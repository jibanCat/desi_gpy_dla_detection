"""Step A.2.a — confirm v1 Python ≡ MATLAB on every spectrum_loss output.

Loads each ``<TID>_matlab.mat`` (saved by
``tests/matlab/run_spectrum_loss_on_fixture.m``) and runs the v1 Python
``spectrum_loss`` on the same fixture inputs. Both implementations
share the ``dlog_β`` approximation (term-A only) by design — agreement
is expected to ~1e-10 across:

  - nlog_p     (scalar)
  - dM         (n_valid × k)
  - dlog_omega (n_valid)
  - dlog_c_0, dlog_tau_0, dlog_beta (scalars)

Pass criterion:
  max relative error across all 6 outputs and all 6 TIDs ≤ 1e-10.

Run directly:
    python tests/test_v1_matches_matlab.py

Prereq:
    matlab -batch "addpath(fullfile(pwd,'tests','matlab')); run_spectrum_loss_on_fixture"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpy_dla_detection.objective import spectrum_loss  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
DTYPE = torch.float64
TIDS = [120046865, 237926, 250915, 237575, 242431, 243225]
TOL = 1e-10  # machine-precision target


def _load_init_torch():
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


def _eval_python(init, tid):
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
    """Return (max abs err, max rel err) between numpy/scalar py and m."""
    py_arr = np.asarray(py).flatten()
    m_arr = np.asarray(m).flatten()
    if py_arr.shape != m_arr.shape:
        m_arr = m_arr.reshape(py_arr.shape)
    diff = np.abs(py_arr - m_arr)
    abs_err = float(diff.max())
    denom = max(float(np.abs(m_arr).max()), 1e-12)
    rel_err = abs_err / denom
    return abs_err, rel_err


def main():
    init = _load_init_torch()
    print(f"  v1 Python ≡ MATLAB cross-check (tol = {TOL:.0e})")
    print(f"  fixture: {FIX}")
    print(f"{'TID':>10}  {'output':<11}  {'max abs err':>14}  {'max rel err':>14}  pass")
    print("-" * 75)

    overall = []
    for tid in TIDS:
        m_path = FIX / f"{tid}_matlab.mat"
        if not m_path.exists():
            print(f"  TID {tid}: NO MATLAB fixture at {m_path} — run the .m driver first")
            return 2
        m = loadmat(m_path)
        py = _eval_python(init, tid)
        for name in ("nlog_p", "dM", "dlog_omega", "dlog_c_0", "dlog_tau_0", "dlog_beta"):
            abs_err, rel_err = _err(py[name], m[name])
            mark = "✓" if rel_err < TOL else "✗"
            print(f"  {tid:>10}  {name:<11}  {abs_err:>14.3e}  {rel_err:>14.3e}  {mark}")
            overall.append((tid, name, rel_err))
        print()

    max_rel = max(r for _, _, r in overall)
    print(f"  MAX rel err across all TIDs and outputs: {max_rel:.3e}  "
          f"(tol {TOL}; {'PASS' if max_rel < TOL else 'FAIL'})")
    return 0 if max_rel < TOL else 1


def test_v1_matches_matlab():
    rc = main()
    assert rc == 0, "v1 Python != MATLAB beyond tolerance — see printed table"


if __name__ == "__main__":
    sys.exit(main())
