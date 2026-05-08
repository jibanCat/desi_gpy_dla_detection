"""Step A.1: Numeric Jacobian sanity for v1 ``spectrum_loss``.

Loads each frozen 2lpt TID and the population init from
``tests/fixtures/2lpt_frozen/`` and checks that the analytic gradients
returned by ``gpy_dla_detection.objective.spectrum_loss`` agree with
central finite differences (eps = 1e-5, the optimal step for float64
central differences) on:

  - log_β, log_τ_0, log_c_0    (3 scalars per spectrum)
  - log_ω[i]                   (5 random valid-pixel indices per spectrum)
  - M[i, j]                    (5 random (pixel, latent) pairs per spectrum)

Per-parameter pass criteria:

  log_c_0, log_τ_0, log_ω, M    rel_err ≤ 1e-4
                                (1e-4 floor accommodates FD precision on
                                 small-magnitude M / log_ω elements; the
                                 typical observed rel_err is 1e-7 to 1e-5)

  log_β                          rel_err ≤ 5e-2  (DOCUMENTED APPROX)
                                 v1 + MATLAB both compute
                                   da_β = da_τ_0 · log(1+z_lya) · β · 1_{forest}
                                 instead of the strict
                                   da_β = β · Σ_k τ_k · log(1+z_k) · 1_{k-forest}
                                 Difference: 1+z_lyb / 1+z_lya = 1216/1026
                                 ≈ 1.185 contributes log(1.185) = 0.170 per
                                 Lyβ-active pixel; similar 0.224 for Lyγ. The
                                 approximation drops the Σ_k log shift, giving
                                 ~0.5–2.5 % systematic bias scaling with z_qso.
                                 v1's MATLAB sibling (`spectrum_loss.m:91-95`)
                                 has the identical formula. Production v1
                                 trained successfully under this approximation
                                 — log_β is one scalar parameter and the
                                 optimizer compensates. Step A.2 confirms v1
                                 Python ≡ MATLAB exactly for this gradient
                                 (both share the approximation). NOT fixed in
                                 this PR by design — fixing it would change
                                 the v1 loss surface.

Run directly:
    python tests/test_v1_spectrum_loss_jacobian.py

Or via pytest:
    pytest tests/test_v1_spectrum_loss_jacobian.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Step A.1 evaluates the v1 reference spectrum_loss directly. The
# byte-identical copy at training_v3/objective.py has a `from .voigt`
# relative import that doesn't resolve from a subpackage; that's a
# wiring detail to fix when we begin modifying training_v3 (Step B+).
# For the Jacobian sanity check the v1 source is the right anchor
# anyway — it's the analytic reference these tests pin down.
from gpy_dla_detection.objective import spectrum_loss  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
DTYPE = torch.float64
EPS = 1e-5  # eps^(1/3) ≈ 1e-5 is the optimal central-FD step for float64
TIDS = [270143607, 250027833, 40000430, 220250636, 180021938, 120046865]
N_FD_SAMPLES = 5
# Per-parameter tolerances — see module docstring for rationale.
TOL_DEFAULT = 1e-4   # log_c_0, log_τ_0, log_ω, M
TOL_LOG_BETA = 5e-2  # documented v1+MATLAB approximation; ~0.5-2.5% bias


def _load_init():
    n = np.load(FIX / "init_params.npz")
    return dict(
        M=torch.tensor(n["M"], dtype=DTYPE),
        log_omega=torch.tensor(n["log_omega"], dtype=DTYPE),
        log_c_0=torch.tensor(np.log(float(n["c_0"])), dtype=DTYPE),
        log_tau_0=torch.tensor(np.log(float(n["tau_0"])), dtype=DTYPE),
        log_beta=torch.tensor(np.log(float(n["beta"])), dtype=DTYPE),
        num_forest_lines=int(n["num_forest_lines"]),
        TW=torch.tensor(n["all_transition_wavelengths"], dtype=DTYPE),
        OS=torch.tensor(n["all_oscillator_strengths"], dtype=DTYPE),
    )


def _load_spec(tid):
    n = np.load(FIX / f"{tid}.npz")
    flux = np.asarray(n["flux"])
    nv = np.asarray(n["noise_variance"])
    lya_1pz = np.asarray(n["lya_1pz"])
    valid_mask = np.asarray(n["valid_mask"]).astype(bool)
    zqso_1pz = float(n["zqso_1pz"])
    # Mask + cast to torch
    y_m = torch.tensor(flux[valid_mask], dtype=DTYPE)
    lya_1pz_m = torch.tensor(lya_1pz[valid_mask], dtype=DTYPE)
    nv_m = torch.tensor(nv[valid_mask], dtype=DTYPE)
    return y_m, lya_1pz_m, nv_m, valid_mask, zqso_1pz


def _eval(y, lya_1pz, nv, M, log_omega, log_c_0, log_tau_0, log_beta,
          num_forest_lines, TW, OS, zqso_1pz):
    """Run spectrum_loss with parameters in (linear-M, log-others) space."""
    omega2 = torch.exp(2 * log_omega)
    c_0 = torch.exp(log_c_0)
    tau_0 = torch.exp(log_tau_0)
    beta = torch.exp(log_beta)
    return spectrum_loss(
        y, lya_1pz, nv, M, omega2, c_0, tau_0, beta,
        num_forest_lines, TW, OS,
        torch.tensor(zqso_1pz, dtype=DTYPE),
    )


def _fd_scalar(eval_fn, base_value, name, eps=EPS):
    """central FD on a scalar log-parameter; returns dL/d(log_param)."""
    base = float(base_value)
    L_plus  = eval_fn(name=name, value=base + eps)
    L_minus = eval_fn(name=name, value=base - eps)
    return (L_plus - L_minus) / (2 * eps)


def _check_one(tid, init, rng):
    y_m, lya_1pz_m, nv_m, valid_mask, zqso_1pz = _load_spec(tid)
    n_valid = y_m.shape[0]
    n_pix = init["M"].shape[0]
    k = init["M"].shape[1]

    # Mask-restricted M and log_omega for spectrum_loss inputs
    M_m_base = init["M"][valid_mask, :].clone()
    log_omega_m_base = init["log_omega"][valid_mask].clone()
    log_c_0_base = init["log_c_0"].clone()
    log_tau_0_base = init["log_tau_0"].clone()
    log_beta_base = init["log_beta"].clone()

    # Analytic gradients at the base point.
    nlog_p, dM_a, dlog_omega_a, dlog_c_0_a, dlog_tau_0_a, dlog_beta_a = _eval(
        y_m, lya_1pz_m, nv_m, M_m_base, log_omega_m_base,
        log_c_0_base, log_tau_0_base, log_beta_base,
        init["num_forest_lines"], init["TW"], init["OS"], zqso_1pz,
    )
    nlog_p_val = float(nlog_p.detach())

    def value_only(*, M=None, log_omega=None, log_c_0=None, log_tau_0=None, log_beta=None):
        nlp, *_ = _eval(
            y_m, lya_1pz_m, nv_m,
            M if M is not None else M_m_base,
            log_omega if log_omega is not None else log_omega_m_base,
            log_c_0 if log_c_0 is not None else log_c_0_base,
            log_tau_0 if log_tau_0 is not None else log_tau_0_base,
            log_beta if log_beta is not None else log_beta_base,
            init["num_forest_lines"], init["TW"], init["OS"], zqso_1pz,
        )
        return float(nlp.detach())

    rows = []
    def _record(name, analytic, fd):
        rel = abs(fd - analytic) / max(abs(analytic), 1e-12)
        rows.append((name, analytic, fd, rel))

    # log_c_0
    fd = (value_only(log_c_0=log_c_0_base + EPS)
          - value_only(log_c_0=log_c_0_base - EPS)) / (2 * EPS)
    _record("dlog_c_0", float(dlog_c_0_a), fd)

    # log_tau_0
    fd = (value_only(log_tau_0=log_tau_0_base + EPS)
          - value_only(log_tau_0=log_tau_0_base - EPS)) / (2 * EPS)
    _record("dlog_tau_0", float(dlog_tau_0_a), fd)

    # log_beta
    fd = (value_only(log_beta=log_beta_base + EPS)
          - value_only(log_beta=log_beta_base - EPS)) / (2 * EPS)
    _record("dlog_beta", float(dlog_beta_a), fd)

    # 5 random log_omega[i]
    idx = rng.choice(n_valid, size=N_FD_SAMPLES, replace=False)
    for i in idx:
        lp = log_omega_m_base.clone(); lp[i] += EPS
        lm = log_omega_m_base.clone(); lm[i] -= EPS
        fd = (value_only(log_omega=lp) - value_only(log_omega=lm)) / (2 * EPS)
        _record(f"dlog_omega[{i}]", float(dlog_omega_a[i]), fd)

    # 5 random M[i, j]
    iM = rng.choice(n_valid, size=N_FD_SAMPLES, replace=False)
    jM = rng.choice(k, size=N_FD_SAMPLES, replace=False)
    for i, j in zip(iM, jM):
        Mp = M_m_base.clone(); Mp[i, j] += EPS
        Mm = M_m_base.clone(); Mm[i, j] -= EPS
        fd = (value_only(M=Mp) - value_only(M=Mm)) / (2 * EPS)
        _record(f"dM[{i},{j}]", float(dM_a[i, j]), fd)

    return nlog_p_val, n_valid, rows


def main():
    init = _load_init()
    rng = np.random.default_rng(0)

    print(f"  v1 spectrum_loss Jacobian sanity (eps={EPS}, "
          f"tol_default={TOL_DEFAULT}, tol_log_beta={TOL_LOG_BETA})")
    print(f"  fixture: {FIX}")
    print(f"{'TID':>10}  {'param':<22}  {'analytic':>14}  {'FD':>14}  {'rel_err':>10}  pass")
    print("-" * 90)

    overall_default = []
    overall_log_beta = []
    for tid in TIDS:
        nlp, n_valid, rows = _check_one(tid, init, rng)
        print(f"  TID {tid:>10}  nlog_p = {nlp:.6f}   n_valid = {n_valid}")
        for name, a, fd, rel in rows:
            tol = TOL_LOG_BETA if name == "dlog_beta" else TOL_DEFAULT
            mark = "✓" if rel < tol else "✗"
            note = " [APPROX]" if name == "dlog_beta" else ""
            print(f"  {tid:>10}  {name:<22}  {a:>14.6e}  {fd:>14.6e}  "
                  f"{rel:>10.2e}  {mark}{note}")
            (overall_log_beta if name == "dlog_beta" else overall_default).append(rel)
        print()

    max_def = max(overall_default) if overall_default else 0.0
    max_lb = max(overall_log_beta) if overall_log_beta else 0.0
    print(f"  MAX rel err   (log_c_0/τ_0/ω, M): {max_def:.3e}  "
          f"(tol {TOL_DEFAULT}; {'PASS' if max_def < TOL_DEFAULT else 'FAIL'})")
    print(f"  MAX rel err   (log_β APPROX):    {max_lb:.3e}  "
          f"(tol {TOL_LOG_BETA}; {'PASS' if max_lb < TOL_LOG_BETA else 'FAIL'})")
    print(f"  log_β is approximate by design (v1 ≡ MATLAB share the same "
          f"single-frame log() simplification — see docstring).")

    if max_def < TOL_DEFAULT and max_lb < TOL_LOG_BETA:
        return 0
    return 1


def test_jacobian():
    """pytest entry: tighter assertion."""
    rc = main()
    assert rc == 0, "spectrum_loss Jacobian sanity FAILED — see printed table"


if __name__ == "__main__":
    sys.exit(main())
