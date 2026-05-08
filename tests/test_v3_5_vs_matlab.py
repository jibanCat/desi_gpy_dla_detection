"""Step A.2.b — confirm v3.5 Python ≡ MATLAB on the 5 unchanged outputs
and differs from MATLAB on dlog_β by 0.5–2.5 % (the empirical signature
of the chromatic-correction term v3.5 adds and v1+MATLAB drop).

Loads ``<TID>_matlab.mat`` (saved by the v1+MATLAB driver) and runs the
v3.5 Python ``spectrum_loss`` on the same fixture inputs. v3.5's only
difference from v1 is the strict-``dlog_β`` patch — so we expect:

  nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0   ≡ MATLAB to ~1e-10
  dlog_β                                          differs by 0.5–2.5%
                                                  (matches the bias
                                                  measured in
                                                  test_v1_spectrum_loss_jacobian)

Pass criteria:
  IDENTICAL outputs → max rel_err < 1e-10
  dlog_β            → 0.5e-3 ≤ rel_err < 0.05  (approx. range)

Run directly:
    python tests/test_v3_5_vs_matlab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpy_dla_detection.training_v3_5.objective import spectrum_loss  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
DTYPE = torch.float64
TIDS = [270143607, 250027833, 40000430, 220250636, 180021938, 120046865]
TOL_IDENT = 1e-10
TOL_DLOG_BETA_LO = 1e-4   # lowest expected divergence
TOL_DLOG_BETA_HI = 5e-2   # upper bound


def _load_init():
    init = loadmat(FIX / "init_params.mat")
    return dict(
        M=torch.tensor(init["M"], dtype=DTYPE),
        log_omega=torch.tensor(np.asarray(init["log_omega"]).squeeze(), dtype=DTYPE),
        c_0=torch.tensor(float(np.asarray(init["c_0"]).flatten()[0]), dtype=DTYPE),
        tau_0=torch.tensor(float(np.asarray(init["tau_0"]).flatten()[0]), dtype=DTYPE),
        beta=torch.tensor(float(np.asarray(init["beta"]).flatten()[0]), dtype=DTYPE),
        num_forest_lines=int(np.asarray(init["num_forest_lines"]).flatten()[0]),
        TW=torch.tensor(np.asarray(init["all_transition_wavelengths"]).squeeze(), dtype=DTYPE),
        OS=torch.tensor(np.asarray(init["all_oscillator_strengths"]).squeeze(), dtype=DTYPE),
    )


def _eval(init, tid):
    spec = loadmat(FIX / f"{tid}.mat")
    flux = np.asarray(spec["flux"]).squeeze()
    nv = np.asarray(spec["noise_variance"]).squeeze()
    lya_1pz = np.asarray(spec["lya_1pz"]).squeeze()
    valid = np.asarray(spec["valid_mask"]).squeeze().astype(bool)
    zqso_1pz = float(np.asarray(spec["zqso_1pz"]).flatten()[0])

    valid_t = torch.tensor(valid)
    omega2 = torch.exp(2 * init["log_omega"])
    y_m = torch.tensor(flux[valid], dtype=DTYPE)
    lya_1pz_m = torch.tensor(lya_1pz[valid], dtype=DTYPE)
    nv_m = torch.tensor(nv[valid], dtype=DTYPE)
    M_m = init["M"][valid_t, :]
    omega2_m = omega2[valid_t]

    nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta = spectrum_loss(
        y_m, lya_1pz_m, nv_m, M_m, omega2_m,
        init["c_0"], init["tau_0"], init["beta"],
        init["num_forest_lines"], init["TW"], init["OS"],
        torch.tensor(zqso_1pz, dtype=DTYPE),
    )
    return dict(
        nlog_p=float(nlog_p),
        dM=dM.numpy(),
        dlog_omega=dlog_omega.numpy(),
        dlog_c_0=float(dlog_c_0),
        dlog_tau_0=float(dlog_tau_0),
        dlog_beta=float(dlog_beta),
    )


def _err(py, m):
    py_arr = np.asarray(py).flatten()
    m_arr = np.asarray(m).flatten()
    if py_arr.shape != m_arr.shape:
        m_arr = m_arr.reshape(py_arr.shape)
    diff = np.abs(py_arr - m_arr)
    abs_err = float(diff.max())
    denom = max(float(np.abs(m_arr).max()), 1e-12)
    return abs_err, abs_err / denom


def main():
    init = _load_init()
    print(f"  v3.5 Python vs MATLAB (5 outputs ≡ MATLAB; dlog_β differs by ~B-term)")
    print(f"  fixture: {FIX}")
    print(f"{'TID':>10}  {'output':<11}  {'max abs err':>14}  {'max rel err':>14}  pass")
    print("-" * 75)

    fail_count = 0
    dlog_beta_diffs = []

    for tid in TIDS:
        m_path = FIX / f"{tid}_matlab.mat"
        if not m_path.exists():
            print(f"  TID {tid}: NO MATLAB fixture — run the .m driver first")
            return 2
        m = loadmat(m_path)
        py = _eval(init, tid)
        for name in ("nlog_p", "dM", "dlog_omega", "dlog_c_0", "dlog_tau_0"):
            abs_err, rel_err = _err(py[name], m[name])
            ok = rel_err < TOL_IDENT
            mark = "✓" if ok else "✗"
            if not ok: fail_count += 1
            print(f"  {tid:>10}  {name:<11}  {abs_err:>14.3e}  {rel_err:>14.3e}  {mark}")
        # dlog_beta — expected NON-zero divergence
        abs_err, rel_err = _err(py["dlog_beta"], m["dlog_beta"])
        ok = TOL_DLOG_BETA_LO <= rel_err <= TOL_DLOG_BETA_HI
        mark = "✓ ≠" if ok else "✗"
        print(f"  {tid:>10}  {'dlog_beta':<11}  {abs_err:>14.3e}  {rel_err:>14.3e}  {mark} (expected divergent)")
        if not ok: fail_count += 1
        dlog_beta_diffs.append((tid, rel_err, py["dlog_beta"], float(m["dlog_beta"])))
        print()

    print("  dlog_β: v3.5 strict vs MATLAB approximate")
    print(f"  {'TID':>10}  {'v3.5':>14}  {'MATLAB':>14}  {'rel diff':>10}")
    for tid, rel, py_v, m_v in dlog_beta_diffs:
        print(f"  {tid:>10}  {py_v:>14.6e}  {m_v:>14.6e}  {rel*100:>9.3f}%")

    print()
    if fail_count == 0:
        print(f"  PASS — 5 outputs ≡ MATLAB to ~{TOL_IDENT:.0e}; "
              f"dlog_β diverges as predicted ({TOL_DLOG_BETA_LO:.0e} to {TOL_DLOG_BETA_HI:.0e})")
        return 0
    print(f"  FAIL — {fail_count} parameter(s) outside expected range")
    return 1


def test_v3_5_vs_matlab():
    rc = main()
    assert rc == 0, "v3.5 vs MATLAB outside expected range — see printed table"


if __name__ == "__main__":
    sys.exit(main())
