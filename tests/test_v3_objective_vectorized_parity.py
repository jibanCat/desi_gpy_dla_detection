"""Step B: parity test — ``spectrum_loss_batch`` ≡ per-spectrum loop to ~1e-10.

Loads the 6 frozen 2lpt TIDs from ``tests/fixtures/2lpt_frozen/`` plus
the population init, then:

  - Runs the v1 per-spectrum ``spectrum_loss`` inside a Python loop,
    scattering ``dM`` and ``dlog_omega`` into the same accumulators a
    real trainer would use (this is the reference path that A.1–A.4
    pinned down).

  - Runs ``training_v3.objective_vectorized.spectrum_loss_batch`` on the
    same 6 spectra padded to a common (N, ) grid with per-spectrum
    valid_mask.

  - Asserts max absolute difference on each of (nlog_p_total, dM_accum,
    dlog_omega_accum, dlog_c_0, dlog_tau_0, dlog_beta) is below 1e-10
    (float64; loosen to 1e-7 only for log_β where the v1+MATLAB ``log``
    approximation introduces a small extra rounding path through
    ``log(lya_1pz) * beta * indicator``).

Run directly:
    python tests/test_v3_objective_vectorized_parity.py

Or via pytest:
    pytest tests/test_v3_objective_vectorized_parity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpy_dla_detection.objective import spectrum_loss
from gpy_dla_detection.training_v3.objective_vectorized import spectrum_loss_batch  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
DTYPE = torch.float64
TIDS = [270143607, 250027833, 40000430, 220250636, 180021938, 120046865]
# Mixed atol+rtol tolerance: |diff| <= atol + rtol * |ref|. With atol=1e-12 and
# rtol=1e-10 this matches "element-by-element to ~1e-10" at the float64 noise
# floor for accumulators that span many orders of magnitude (dM_accum reaches
# ~13, nlog_p ~24000; the absolute diffs scale linearly with the accumulator
# magnitude under matmul order changes).
TOL_ATOL = 1e-12
TOL_RTOL = 1e-10


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


def _load_spec(tid):
    n = np.load(FIX / f"{tid}.npz")
    return dict(
        flux=np.asarray(n["flux"], dtype=np.float64),
        noise_variance=np.asarray(n["noise_variance"], dtype=np.float64),
        lya_1pz=np.asarray(n["lya_1pz"], dtype=np.float64),
        valid_mask=np.asarray(n["valid_mask"], dtype=bool),
        zqso_1pz=float(n["zqso_1pz"]),
    )


def _per_spectrum_reference(init, specs):
    """Run the v1 spectrum_loss in a Python loop (the reference path).

    Mirrors how a real trainer accumulates: dM_accum has full (N, k) shape
    and we scatter per-spectrum dM into the rows specified by valid_mask.
    """
    M = init["M"]
    omega2_full = torch.exp(2 * init["log_omega"])
    c_0 = init["c_0"]; tau_0 = init["tau_0"]; beta = init["beta"]
    N, k = M.shape

    nlog_total = torch.zeros((), dtype=DTYPE)
    dM_accum = torch.zeros_like(M)
    dlog_omega_accum = torch.zeros_like(init["log_omega"])
    dlog_c_0_accum = torch.zeros((), dtype=DTYPE)
    dlog_tau_0_accum = torch.zeros((), dtype=DTYPE)
    dlog_beta_accum = torch.zeros((), dtype=DTYPE)

    for sp in specs:
        valid_mask = sp["valid_mask"]
        valid_t = torch.tensor(valid_mask, dtype=torch.bool)
        y = torch.tensor(sp["flux"][valid_mask], dtype=DTYPE)
        nv = torch.tensor(sp["noise_variance"][valid_mask], dtype=DTYPE)
        lya_1pz = torch.tensor(sp["lya_1pz"][valid_mask], dtype=DTYPE)
        M_v = M[valid_t, :]
        omega2_v = omega2_full[valid_t]
        zqso_1pz = torch.tensor(sp["zqso_1pz"], dtype=DTYPE)

        nlp, dM, dlogw, dlc0, dlt0, dlb = spectrum_loss(
            y, lya_1pz, nv, M_v, omega2_v, c_0, tau_0, beta,
            init["num_forest_lines"], init["TW"], init["OS"], zqso_1pz,
        )
        nlog_total = nlog_total + nlp.detach()
        dM_accum[valid_t, :] = dM_accum[valid_t, :] + dM.detach()
        dlog_omega_accum[valid_t] = dlog_omega_accum[valid_t] + dlogw.detach()
        dlog_c_0_accum = dlog_c_0_accum + dlc0.detach()
        dlog_tau_0_accum = dlog_tau_0_accum + dlt0.detach()
        dlog_beta_accum = dlog_beta_accum + dlb.detach()

    return (nlog_total, dM_accum, dlog_omega_accum,
            dlog_c_0_accum, dlog_tau_0_accum, dlog_beta_accum)


def _vectorized(init, specs):
    """Pad to (B, N), call spectrum_loss_batch."""
    M = init["M"]
    omega2_full = torch.exp(2 * init["log_omega"])
    N, k = M.shape
    B = len(specs)

    y_b = torch.zeros((B, N), dtype=DTYPE)
    nv_b = torch.zeros((B, N), dtype=DTYPE)
    lya_1pz_b = torch.zeros((B, N), dtype=DTYPE)
    valid_b = torch.zeros((B, N), dtype=torch.bool)
    zqso_1pz_b = torch.zeros((B,), dtype=DTYPE)

    for b, sp in enumerate(specs):
        # Fill flux / nv only at valid; lya_1pz must be finite everywhere
        # so the function's torch.pow(lya_1pz, beta) doesn't see 0 or NaN.
        v = sp["valid_mask"]
        y_b[b, :] = torch.tensor(np.where(v, sp["flux"], 0.0), dtype=DTYPE)
        nv_b[b, :] = torch.tensor(np.where(v, sp["noise_variance"], 1.0), dtype=DTYPE)
        # lya_1pz: use the raw value (it's a function of the rest grid * (1+zqso),
        # always finite and positive) so the indicator gating is the only mask.
        lya_1pz_b[b, :] = torch.tensor(sp["lya_1pz"], dtype=DTYPE)
        valid_b[b, :] = torch.tensor(v, dtype=torch.bool)
        zqso_1pz_b[b] = sp["zqso_1pz"]

    return spectrum_loss_batch(
        y_b, lya_1pz_b, nv_b, valid_b,
        M, omega2_full, init["c_0"], init["tau_0"], init["beta"],
        init["num_forest_lines"], init["TW"], init["OS"],
        zqso_1pz_b,
    )


def _max_abs_diff(a, b):
    return float((a - b).abs().max())


def main():
    init = _load_init()
    specs = [_load_spec(tid) for tid in TIDS]

    print(f"  Step B parity: spectrum_loss_batch vs per-spectrum loop")
    print(f"  fixture: {FIX}")
    print(f"  TIDs: {TIDS}")
    print(f"  M.shape={tuple(init['M'].shape)}  num_forest_lines={init['num_forest_lines']}")
    print()

    ref = _per_spectrum_reference(init, specs)
    vec = _vectorized(init, specs)

    names = ["nlog_p_total", "dM_accum", "dlog_omega_accum",
             "dlog_c_0", "dlog_tau_0", "dlog_beta"]
    print(f"  {'param':<20}  {'|ref|_max':>14}  {'|vec|_max':>14}  "
          f"{'|diff|_max':>14}  {'|diff|/|ref|':>12}")
    print("-" * 92)
    failures = []
    for name, r, v in zip(names, ref, vec):
        diff = _max_abs_diff(r, v)
        rmax = float(r.abs().max())
        vmax = float(v.abs().max())
        rel = diff / max(rmax, 1.0)  # relative-or-absolute floor
        tol = TOL_ATOL + TOL_RTOL * rmax
        ok = diff <= tol
        mark = "✓" if ok else "✗"
        print(f"  {name:<20}  {rmax:>14.6e}  {vmax:>14.6e}  {diff:>14.6e}  "
              f"{rel:>12.3e}  {mark}")
        if not ok:
            failures.append((name, diff, tol))

    print()
    if failures:
        print(f"  FAIL — {len(failures)} parity violation(s):")
        for name, diff, tol in failures:
            print(f"    {name}: |diff|={diff:.3e}  tol={tol:.0e}")
        return 1
    print(f"  PASS — all 6 parity scalars within tolerance")
    return 0


def test_parity():
    """pytest entry."""
    rc = main()
    assert rc == 0, "spectrum_loss_batch parity FAILED — see printed table"


if __name__ == "__main__":
    sys.exit(main())
