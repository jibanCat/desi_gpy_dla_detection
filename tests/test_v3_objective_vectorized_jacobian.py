"""Step B independent gradient check: numeric Jacobian on
``spectrum_loss_batch`` directly.

Unlike ``tests/test_v3_objective_vectorized_parity.py`` (which compares
batched ↔ per-spectrum-loop, and so misses bug classes where both paths
have the same gradient error), this test takes finite differences of
the **batched** loss only and compares to the batched analytic gradient.
That makes the math check independent of the per-spectrum reference.

Setup mirrors ``tests/test_v1_spectrum_loss_jacobian.py`` but applied to
the batched function:

  - 6 frozen 2lpt TIDs padded to a common (B=6, N=5662) grid with their
    per-spectrum valid_mask.
  - Analytic gradients pulled from one ``spectrum_loss_batch`` call.
  - Central finite differences (eps=1e-5) on:
      log_c_0, log_τ_0, log_β   (3 scalars; shared)
      log_ω[i]                  (5 random pixel indices; shared)
      M[i, j]                   (5 random (pixel, latent) pairs; shared)

  - Per-pixel gradients are accumulated across all 6 spectra by
    ``spectrum_loss_batch`` (this is the design); FD on the batched
    loss should pick up exactly the same accumulated gradient. So we
    compare ``dlog_omega_accum[i]`` and ``dM_accum[i, j]`` to the FD
    values directly.

Tolerances follow the v1 / v3.5 Jacobian tests:
  log_c_0/τ_0/ω, M  rel_err ≤ 1e-4   (FD precision floor for f64)
  log_β             rel_err ≤ 5e-2   (documented v1 dlog_β approximation;
                                      see test_v1_spectrum_loss_jacobian.py
                                      docstring for the analytic background)

Run directly:
    python tests/test_v3_objective_vectorized_jacobian.py

Or via pytest:
    pytest tests/test_v3_objective_vectorized_jacobian.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpy_dla_detection.training_v3.objective_vectorized import spectrum_loss_batch  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
DTYPE = torch.float64
EPS = 1e-5
TIDS = [270143607, 250027833, 40000430, 220250636, 180021938, 120046865]
N_FD_SAMPLES = 5
TOL_DEFAULT = 1e-4
TOL_LOG_BETA = 5e-2  # documented v1 + MATLAB approximation in dlog_β


def _load_init():
    n = np.load(FIX / "init_params.npz")
    return dict(
        M=torch.tensor(n["M"], dtype=DTYPE),
        log_omega=torch.tensor(n["log_omega"], dtype=DTYPE),
        c_0=torch.tensor(float(n["c_0"]), dtype=DTYPE),
        tau_0=torch.tensor(float(n["tau_0"]), dtype=DTYPE),
        beta=torch.tensor(float(n["beta"]), dtype=DTYPE),
        num_forest_lines=int(n["num_forest_lines"]),
        TW=torch.tensor(n["all_transition_wavelengths"], dtype=DTYPE),
        OS=torch.tensor(n["all_oscillator_strengths"], dtype=DTYPE),
    )


def _build_batch(init):
    """Pad the 6 2lpt fixtures to (B, N) with per-spectrum valid_mask."""
    N = init["M"].shape[0]
    B = len(TIDS)

    y_b = torch.zeros((B, N), dtype=DTYPE)
    nv_b = torch.zeros((B, N), dtype=DTYPE)
    lya_b = torch.zeros((B, N), dtype=DTYPE)
    valid_b = torch.zeros((B, N), dtype=torch.bool)
    zqso_b = torch.zeros((B,), dtype=DTYPE)

    for b, tid in enumerate(TIDS):
        n = np.load(FIX / f"{tid}.npz")
        v = np.asarray(n["valid_mask"]).astype(bool)
        y_b[b] = torch.tensor(np.where(v, np.asarray(n["flux"]), 0.0), dtype=DTYPE)
        nv_b[b] = torch.tensor(np.where(v, np.asarray(n["noise_variance"]), 1.0), dtype=DTYPE)
        lya_b[b] = torch.tensor(np.asarray(n["lya_1pz"]), dtype=DTYPE)
        valid_b[b] = torch.tensor(v, dtype=torch.bool)
        zqso_b[b] = float(n["zqso_1pz"])

    return y_b, lya_b, nv_b, valid_b, zqso_b


def _eval_batch(y, lya, nv, valid, zqso,
                M, log_omega, log_c_0, log_tau_0, log_beta,
                num_forest_lines, TW, OS):
    """Run spectrum_loss_batch with parameters in (linear-M, log-others) space."""
    omega2 = torch.exp(2 * log_omega)
    c_0 = torch.exp(log_c_0)
    tau_0 = torch.exp(log_tau_0)
    beta = torch.exp(log_beta)
    return spectrum_loss_batch(
        y, lya, nv, valid, M, omega2, c_0, tau_0, beta,
        num_forest_lines, TW, OS, zqso,
    )


def _check(rng):
    init = _load_init()
    y, lya, nv, valid, zqso = _build_batch(init)

    M_base = init["M"].clone()
    log_omega_base = init["log_omega"].clone()
    log_c_0_base = torch.log(init["c_0"]).clone()
    log_tau_0_base = torch.log(init["tau_0"]).clone()
    log_beta_base = torch.log(init["beta"]).clone()
    num_forest_lines = init["num_forest_lines"]
    TW, OS = init["TW"], init["OS"]

    def value_only(*, M=None, log_omega=None, log_c_0=None, log_tau_0=None, log_beta=None):
        nlp, *_ = _eval_batch(
            y, lya, nv, valid, zqso,
            M if M is not None else M_base,
            log_omega if log_omega is not None else log_omega_base,
            log_c_0 if log_c_0 is not None else log_c_0_base,
            log_tau_0 if log_tau_0 is not None else log_tau_0_base,
            log_beta if log_beta is not None else log_beta_base,
            num_forest_lines, TW, OS,
        )
        return float(nlp.detach())

    # Analytic gradients at the base point.
    nlog_p, dM_a, dlog_omega_a, dlog_c_0_a, dlog_tau_0_a, dlog_beta_a = _eval_batch(
        y, lya, nv, valid, zqso,
        M_base, log_omega_base, log_c_0_base, log_tau_0_base, log_beta_base,
        num_forest_lines, TW, OS,
    )
    nlog_p_val = float(nlog_p.detach())
    n_pix, k = M_base.shape

    rows = []
    def _record(name, analytic, fd):
        rel = abs(fd - analytic) / max(abs(analytic), 1e-12)
        rows.append((name, analytic, fd, rel))

    # log_c_0  (scalar shared across batch)
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

    # 5 random log_omega[i]: pick from pixels that are valid in at least
    # one spectrum (otherwise the FD is identically zero and tells us
    # nothing about the gradient). The batch's any-valid mask ensures
    # the perturbation actually affects the loss.
    any_valid = valid.any(dim=0).numpy()
    valid_pixels = np.where(any_valid)[0]
    idx = rng.choice(valid_pixels, size=N_FD_SAMPLES, replace=False)
    for i in idx:
        lp = log_omega_base.clone(); lp[i] += EPS
        lm = log_omega_base.clone(); lm[i] -= EPS
        fd = (value_only(log_omega=lp) - value_only(log_omega=lm)) / (2 * EPS)
        _record(f"dlog_omega[{i}]", float(dlog_omega_a[i]), fd)

    # 5 random M[i, j]: same any-valid pixel filter.
    iM = rng.choice(valid_pixels, size=N_FD_SAMPLES, replace=False)
    jM = rng.choice(k, size=N_FD_SAMPLES, replace=False)
    for i, j in zip(iM, jM):
        Mp = M_base.clone(); Mp[i, j] += EPS
        Mm = M_base.clone(); Mm[i, j] -= EPS
        fd = (value_only(M=Mp) - value_only(M=Mm)) / (2 * EPS)
        _record(f"dM[{i},{j}]", float(dM_a[i, j]), fd)

    return nlog_p_val, rows


def main():
    rng = np.random.default_rng(0)

    print(f"  spectrum_loss_batch numeric Jacobian sanity (eps={EPS}, "
          f"tol_default={TOL_DEFAULT}, tol_log_beta={TOL_LOG_BETA})")
    print(f"  fixture: {FIX}")
    print(f"  TIDs: {TIDS}")
    print()

    nlog_p, rows = _check(rng)
    print(f"  nlog_p_total (batched, B=6) = {nlog_p:.6f}")
    print()
    print(f"  {'param':<22}  {'analytic':>14}  {'FD':>14}  {'rel_err':>10}  pass")
    print("-" * 90)

    overall_default = []
    overall_log_beta = []
    for name, a, fd, rel in rows:
        is_lb = (name == "dlog_beta")
        tol = TOL_LOG_BETA if is_lb else TOL_DEFAULT
        mark = "✓" if rel < tol else "✗"
        note = " [APPROX]" if is_lb else ""
        print(f"  {name:<22}  {a:>14.6e}  {fd:>14.6e}  {rel:>10.2e}  {mark}{note}")
        (overall_log_beta if is_lb else overall_default).append(rel)

    max_def = max(overall_default) if overall_default else 0.0
    max_lb = max(overall_log_beta) if overall_log_beta else 0.0
    print()
    print(f"  MAX rel err (log_c_0/τ_0/ω, M): {max_def:.3e}  "
          f"(tol {TOL_DEFAULT}; {'PASS' if max_def < TOL_DEFAULT else 'FAIL'})")
    print(f"  MAX rel err (log_β APPROX):    {max_lb:.3e}  "
          f"(tol {TOL_LOG_BETA}; {'PASS' if max_lb < TOL_LOG_BETA else 'FAIL'})")
    print(f"  log_β is approximate by design (v1 + MATLAB share the "
          f"single-frame log() simplification — see "
          f"tests/test_v1_spectrum_loss_jacobian.py docstring).")

    if max_def < TOL_DEFAULT and max_lb < TOL_LOG_BETA:
        return 0
    return 1


def test_jacobian():
    """pytest entry: tighter assertion."""
    rc = main()
    assert rc == 0, "spectrum_loss_batch Jacobian sanity FAILED — see printed table"


if __name__ == "__main__":
    sys.exit(main())
