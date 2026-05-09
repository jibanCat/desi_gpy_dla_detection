"""Step B follow-up smoke: 3-Adam-iter trained M is identical between
per-spectrum loop and ``spectrum_loss_batch`` paths.

This is the equivalence check the trainer refactor must pass before we
swap the per-spectrum loop in ``phase2_train_dr16.py:_train`` /
``short_retrain_2lpt.py:_full_batch_objective`` for the vectorized
``spectrum_loss_batch`` from
``gpy_dla_detection/training_v3/objective_vectorized.py``.

Setup:
  - Load the 6 frozen 2lpt TIDs + the population init from
    ``tests/fixtures/2lpt_frozen/``.
  - Build (B=6, N=5662) padded inputs.
  - Run 3 Adam iterations under each path with identical:
      M_init, log_omega_init, log_c_0, log_tau_0, log_beta initial values
      Adam(lr=0.01) defaults
      same priors on log_τ_0 / log_β
  - Compare trained M, log_omega, log_c_0, log_tau_0, log_beta.

Tolerance: per-iter parity of dM is ~6.4e-11 relative; over 3 Adam steps
with lr=0.01 the accumulated drift in M is bounded by O(lr · n_iter ·
per_iter_diff) ≈ 1e-11 absolute. We assert max_abs_diff(M) < 1e-9 with
extra margin for f64 momentum-buffer rounding.

Run directly:
    python tests/test_v3_train_step_parity.py

Or via pytest:
    pytest tests/test_v3_train_step_parity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpy_dla_detection.objective import spectrum_loss
from gpy_dla_detection.training_v3.objective_vectorized import spectrum_loss_batch  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
DTYPE = torch.float64
TIDS = [270143607, 250027833, 40000430, 220250636, 180021938, 120046865]
N_ITERS = 3
LR = 0.01
TOL_M = 1e-9
TOL_OMEGA = 1e-10
TOL_SCALAR = 1e-10
# Same DR12Q priors as phase2_train_dr16.py:_train
TAU_0_PRIOR_MU, TAU_0_PRIOR_SIGMA = 0.00554, 0.00064
BETA_PRIOR_MU, BETA_PRIOR_SIGMA = 3.182, 0.074


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


def _load_specs():
    out = []
    for tid in TIDS:
        n = np.load(FIX / f"{tid}.npz")
        out.append(dict(
            flux=np.asarray(n["flux"], dtype=np.float64),
            noise_variance=np.asarray(n["noise_variance"], dtype=np.float64),
            lya_1pz=np.asarray(n["lya_1pz"], dtype=np.float64),
            valid_mask=np.asarray(n["valid_mask"], dtype=bool),
            zqso_1pz=float(n["zqso_1pz"]),
        ))
    return out


def _make_params(init):
    """Fresh nn.Parameters initialised to the population init, with log scalars."""
    M = nn.Parameter(init["M"].clone())
    log_omega = nn.Parameter(init["log_omega"].clone())
    log_c_0 = nn.Parameter(torch.log(init["c_0"]).clone())
    log_tau_0 = nn.Parameter(torch.log(init["tau_0"]).clone())
    log_beta = nn.Parameter(torch.log(init["beta"]).clone())
    return M, log_omega, log_c_0, log_tau_0, log_beta


def _apply_priors(dlog_tau_0, dlog_beta, tau_0, beta):
    """Same priors as phase2_train_dr16.py — DR12Q τ_0/β."""
    dlog_tau_0 = dlog_tau_0 + tau_0 * (tau_0 - TAU_0_PRIOR_MU) / TAU_0_PRIOR_SIGMA**2
    dlog_beta = dlog_beta + beta * (beta - BETA_PRIOR_MU) / BETA_PRIOR_SIGMA**2
    return dlog_tau_0, dlog_beta


def _train_per_spectrum(init, specs):
    """Reference path: per-spectrum loop (mirrors phase2_train_dr16.py:_train)."""
    M, log_omega, log_c_0, log_tau_0, log_beta = _make_params(init)
    optim = torch.optim.Adam([M, log_omega, log_c_0, log_tau_0, log_beta], lr=LR)

    for _ in range(N_ITERS):
        optim.zero_grad()
        omega2 = torch.exp(2 * log_omega)
        c_0 = torch.exp(log_c_0)
        tau_0 = torch.exp(log_tau_0)
        beta = torch.exp(log_beta)

        dM_acc = torch.zeros_like(M)
        dlogw_acc = torch.zeros_like(log_omega)
        dlc0 = torch.zeros((), dtype=DTYPE)
        dlt0 = torch.zeros((), dtype=DTYPE)
        dlb = torch.zeros((), dtype=DTYPE)

        for sp in specs:
            valid_t = torch.tensor(sp["valid_mask"], dtype=torch.bool)
            y = torch.tensor(sp["flux"][sp["valid_mask"]], dtype=DTYPE)
            nv = torch.tensor(sp["noise_variance"][sp["valid_mask"]], dtype=DTYPE)
            lya = torch.tensor(sp["lya_1pz"][sp["valid_mask"]], dtype=DTYPE)
            M_v = M[valid_t, :]
            omega2_v = omega2[valid_t]
            zqso_1pz = torch.tensor(sp["zqso_1pz"], dtype=DTYPE)

            _, dM, dlogw, dc0, dt0, db = spectrum_loss(
                y, lya, nv, M_v, omega2_v, c_0, tau_0, beta,
                init["num_forest_lines"], init["TW"], init["OS"], zqso_1pz,
            )
            dM_acc[valid_t, :] = dM_acc[valid_t, :] + dM.detach()
            dlogw_acc[valid_t] = dlogw_acc[valid_t] + dlogw.detach()
            dlc0 = dlc0 + dc0.detach()
            dlt0 = dlt0 + dt0.detach()
            dlb = dlb + db.detach()

        dlt0, dlb = _apply_priors(dlt0, dlb, tau_0, beta)

        with torch.no_grad():
            M.grad = dM_acc.clone()
            log_omega.grad = dlogw_acc.clone()
            log_c_0.grad = dlc0.clone()
            log_tau_0.grad = dlt0.clone()
            log_beta.grad = dlb.clone()
        optim.step()

    return dict(M=M.detach().clone(), log_omega=log_omega.detach().clone(),
                log_c_0=log_c_0.detach().clone(),
                log_tau_0=log_tau_0.detach().clone(),
                log_beta=log_beta.detach().clone())


def _train_vectorized(init, specs):
    """Test path: pad to (B, N), call spectrum_loss_batch each iter."""
    M, log_omega, log_c_0, log_tau_0, log_beta = _make_params(init)
    optim = torch.optim.Adam([M, log_omega, log_c_0, log_tau_0, log_beta], lr=LR)

    N = M.shape[0]
    B = len(specs)
    y_b = torch.zeros((B, N), dtype=DTYPE)
    nv_b = torch.zeros((B, N), dtype=DTYPE)
    lya_b = torch.zeros((B, N), dtype=DTYPE)
    valid_b = torch.zeros((B, N), dtype=torch.bool)
    zqso_b = torch.zeros((B,), dtype=DTYPE)
    for b, sp in enumerate(specs):
        v = sp["valid_mask"]
        y_b[b] = torch.tensor(np.where(v, sp["flux"], 0.0), dtype=DTYPE)
        nv_b[b] = torch.tensor(np.where(v, sp["noise_variance"], 1.0), dtype=DTYPE)
        lya_b[b] = torch.tensor(sp["lya_1pz"], dtype=DTYPE)
        valid_b[b] = torch.tensor(v, dtype=torch.bool)
        zqso_b[b] = sp["zqso_1pz"]

    for _ in range(N_ITERS):
        optim.zero_grad()
        omega2 = torch.exp(2 * log_omega)
        c_0 = torch.exp(log_c_0)
        tau_0 = torch.exp(log_tau_0)
        beta = torch.exp(log_beta)

        _, dM, dlogw, dc0, dt0, db = spectrum_loss_batch(
            y_b, lya_b, nv_b, valid_b,
            M, omega2, c_0, tau_0, beta,
            init["num_forest_lines"], init["TW"], init["OS"],
            zqso_b,
        )
        dt0, db = _apply_priors(dt0, db, tau_0, beta)

        with torch.no_grad():
            M.grad = dM.detach().clone()
            log_omega.grad = dlogw.detach().clone()
            log_c_0.grad = dc0.detach().clone()
            log_tau_0.grad = dt0.detach().clone()
            log_beta.grad = db.detach().clone()
        optim.step()

    return dict(M=M.detach().clone(), log_omega=log_omega.detach().clone(),
                log_c_0=log_c_0.detach().clone(),
                log_tau_0=log_tau_0.detach().clone(),
                log_beta=log_beta.detach().clone())


def main():
    init = _load_init()
    specs = _load_specs()

    print(f"  Step B trainer-refactor smoke: 3 Adam iter, lr={LR}, B=6 2lpt fixtures")
    print(f"  fixture: {FIX}")
    print(f"  M.shape={tuple(init['M'].shape)}")
    print()

    ref = _train_per_spectrum(init, specs)
    vec = _train_vectorized(init, specs)

    rows = [
        ("M", ref["M"], vec["M"], TOL_M),
        ("log_omega", ref["log_omega"], vec["log_omega"], TOL_OMEGA),
        ("log_c_0", ref["log_c_0"], vec["log_c_0"], TOL_SCALAR),
        ("log_tau_0", ref["log_tau_0"], vec["log_tau_0"], TOL_SCALAR),
        ("log_beta", ref["log_beta"], vec["log_beta"], TOL_SCALAR),
    ]
    print(f"  {'param':<14}  {'|ref|_max':>14}  {'|vec|_max':>14}  "
          f"{'|diff|_max':>14}  {'tol':>10}")
    print("-" * 78)
    failures = []
    for name, r, v, tol in rows:
        diff = float((r - v).abs().max())
        rmax = float(r.abs().max())
        vmax = float(v.abs().max())
        ok = diff <= tol
        mark = "✓" if ok else "✗"
        print(f"  {name:<14}  {rmax:>14.6e}  {vmax:>14.6e}  {diff:>14.6e}  "
              f"{tol:>10.0e}  {mark}")
        if not ok:
            failures.append((name, diff, tol))

    print()
    if failures:
        print(f"  FAIL — {len(failures)} parameter(s) over tolerance:")
        for name, d, tol in failures:
            print(f"    {name}: |diff|={d:.3e}  tol={tol:.0e}")
        return 1
    print(f"  PASS — vectorized trainer step ≡ per-spectrum loop after {N_ITERS} Adam iters")
    return 0


def test_train_step_parity():
    rc = main()
    assert rc == 0, "trainer-refactor smoke FAILED — see printed table"


if __name__ == "__main__":
    sys.exit(main())
