"""Step A.1 v3.5: Numeric Jacobian sanity for the v3.5 strict-dlog_β
``spectrum_loss``.

v3.5 fixes the chromatic-correction term that v1 + MATLAB approximate.
This test uses the SAME tolerance for log_β as for the other gradients
(unlike test_v1_spectrum_loss_jacobian, which permits 5e-2 because v1
inherits the MATLAB approximation).

Pass criterion (uniform across all gradients):
    rel_err = |FD - analytic| / max(|analytic|, 1e-12) ≤ 1e-4

Run directly:
    python tests/test_v3_5_spectrum_loss_jacobian.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpy_dla_detection.training_v3_5.objective import spectrum_loss  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
DTYPE = torch.float64
EPS = 1e-5
TIDS = [270143607, 250027833, 40000430, 220250636, 180021938, 120046865]
N_FD_SAMPLES = 5
TOL = 1e-4  # uniform — v3.5 should pass at the same threshold as the v1 "exact" gradients


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
    y_m = torch.tensor(flux[valid_mask], dtype=DTYPE)
    lya_1pz_m = torch.tensor(lya_1pz[valid_mask], dtype=DTYPE)
    nv_m = torch.tensor(nv[valid_mask], dtype=DTYPE)
    return y_m, lya_1pz_m, nv_m, valid_mask, zqso_1pz


def _eval(y, lya_1pz, nv, M, log_omega, log_c_0, log_tau_0, log_beta,
          num_forest_lines, TW, OS, zqso_1pz):
    omega2 = torch.exp(2 * log_omega)
    c_0 = torch.exp(log_c_0)
    tau_0 = torch.exp(log_tau_0)
    beta = torch.exp(log_beta)
    return spectrum_loss(
        y, lya_1pz, nv, M, omega2, c_0, tau_0, beta,
        num_forest_lines, TW, OS,
        torch.tensor(zqso_1pz, dtype=DTYPE),
    )


def _check_one(tid, init, rng):
    y_m, lya_1pz_m, nv_m, valid_mask, zqso_1pz = _load_spec(tid)
    n_valid = y_m.shape[0]
    k = init["M"].shape[1]

    M_m_base = init["M"][valid_mask, :].clone()
    log_omega_m_base = init["log_omega"][valid_mask].clone()
    log_c_0_base = init["log_c_0"].clone()
    log_tau_0_base = init["log_tau_0"].clone()
    log_beta_base = init["log_beta"].clone()

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

    fd = (value_only(log_c_0=log_c_0_base + EPS) - value_only(log_c_0=log_c_0_base - EPS)) / (2 * EPS)
    _record("dlog_c_0", float(dlog_c_0_a), fd)

    fd = (value_only(log_tau_0=log_tau_0_base + EPS) - value_only(log_tau_0=log_tau_0_base - EPS)) / (2 * EPS)
    _record("dlog_tau_0", float(dlog_tau_0_a), fd)

    fd = (value_only(log_beta=log_beta_base + EPS) - value_only(log_beta=log_beta_base - EPS)) / (2 * EPS)
    _record("dlog_beta", float(dlog_beta_a), fd)

    idx = rng.choice(n_valid, size=N_FD_SAMPLES, replace=False)
    for i in idx:
        lp = log_omega_m_base.clone(); lp[i] += EPS
        lm = log_omega_m_base.clone(); lm[i] -= EPS
        fd = (value_only(log_omega=lp) - value_only(log_omega=lm)) / (2 * EPS)
        _record(f"dlog_omega[{i}]", float(dlog_omega_a[i]), fd)

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
    print(f"  v3.5 spectrum_loss Jacobian sanity (eps={EPS}, tol={TOL})")
    print(f"  fixture: {FIX}")
    print(f"{'TID':>10}  {'param':<22}  {'analytic':>14}  {'FD':>14}  {'rel_err':>10}  pass")
    print("-" * 90)

    overall = []
    for tid in TIDS:
        nlp, n_valid, rows = _check_one(tid, init, rng)
        print(f"  TID {tid:>10}  nlog_p = {nlp:.6f}   n_valid = {n_valid}")
        for name, a, fd, rel in rows:
            mark = "✓" if rel < TOL else "✗"
            print(f"  {tid:>10}  {name:<22}  {a:>14.6e}  {fd:>14.6e}  {rel:>10.2e}  {mark}")
            overall.append(rel)
        print()

    max_rel = max(overall)
    print(f"  MAX rel error across all tests: {max_rel:.3e}  "
          f"(tolerance {TOL}; {'PASS' if max_rel < TOL else 'FAIL'})")
    return 0 if max_rel < TOL else 1


def test_jacobian():
    rc = main()
    assert rc == 0, "v3.5 spectrum_loss Jacobian sanity FAILED"


if __name__ == "__main__":
    sys.exit(main())
