"""Step A.3 — short retrain on the 1300-spectrum 2lpt fixture.

Runs full-batch Adam for N_ITERS=50 iterations on:

  v1 lane    : gpy_dla_detection.objective.spectrum_loss            (approx dlog_β)
  v3.5 lane  : gpy_dla_detection.training_v3_5.objective.spectrum_loss  (strict dlog_β)

Both lanes:
  - use the same starting parameters from tests/fixtures/2lpt_frozen/init_params.npz
  - use the same training data from tests/fixtures/2lpt_frozen/training_set.npz
  - bypass v1's `objective` wrapper (which has a separate zqso_1pz bug —
    see docs/notes/2026-05-07_v1_objective_zqso_bug_finding.md). Each
    spectrum is fed `zqso_1pz = z_qso + 1` correctly.
  - apply the BOSS DR12Q Gaussian priors on (log_τ_0, log_β) matching
    MATLAB legacy defaults, so the converged β is directly comparable
    between Python and MATLAB lanes.
    τ_0 ~ N(0.00554, 0.00064²)
    β   ~ N(3.182, 0.074²)

Saves to tests/fixtures/2lpt_frozen/short_retrain/<lane>.npz:
  loss_history, log_tau_0_history, log_beta_history, log_c_0_history,
  M_final, mu, log_omega_final, log_c_0_final, log_tau_0_final, log_beta_final.

Run:
    python tests/short_retrain_2lpt.py [--lane v1 | v3.5 | both]
                                       [--n-iters 50]
                                       [--lr 0.01]
"""
from __future__ import annotations

# 2026-05-07 PERFORMANCE NOTE: this trainer iterates over 1300 spectra
# inside Python (no batch vectorization yet — Step B). Each call to
# spectrum_loss does small linear-algebra ops (a 30×30 Cholesky + (n×30)
# matmuls). With multi-threaded BLAS (default OMP_NUM_THREADS=#cores),
# every inner-loop iteration spawns a thread storm whose synchronization
# overhead dwarfs the actual compute. Result: per-iter time blows up
# 10–20× under default threading.
#
# Pinning to 1 thread makes the inner loop 10× faster on this machine.
# Step B (vectorize across spectra) will let us re-enable multi-threaded
# BLAS profitably.
import os as _os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_name, "1")

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
OUT = FIX / "short_retrain"
DTYPE = torch.float64

# BOSS DR12Q priors (matches MATLAB objective.m:66-77)
TAU_0_PRIOR_MU = 0.00554
TAU_0_PRIOR_SIGMA = 0.00064
BETA_PRIOR_MU = 3.182
BETA_PRIOR_SIGMA = 0.074


def _import_spectrum_loss(lane):
    if lane == "v1":
        return importlib.import_module("gpy_dla_detection.objective").spectrum_loss
    if lane == "v3.5":
        return importlib.import_module("gpy_dla_detection.training_v3_5.objective").spectrum_loss
    raise ValueError(lane)


def _load_fixture():
    init = np.load(FIX / "init_params.npz")
    train = np.load(FIX / "training_set.npz")
    return init, train


def _build_init_params(init):
    """Returns (mu, M_param, log_omega_param, log_c_0_param,
                 log_tau_0_param, log_beta_param, TW, OS, num_forest_lines).
    Parameters are nn.Parameter so optimizer can update."""
    mu = torch.tensor(init["mu"], dtype=DTYPE)  # NOT optimized (kept fixed at init)
    M = nn.Parameter(torch.tensor(init["M"], dtype=DTYPE))
    log_omega = nn.Parameter(torch.tensor(init["log_omega"], dtype=DTYPE))
    log_c_0 = nn.Parameter(torch.tensor(np.log(float(init["c_0"])), dtype=DTYPE))
    log_tau_0 = nn.Parameter(torch.tensor(np.log(float(init["tau_0"])), dtype=DTYPE))
    log_beta = nn.Parameter(torch.tensor(np.log(float(init["beta"])), dtype=DTYPE))
    TW = torch.tensor(init["all_transition_wavelengths"], dtype=DTYPE)
    OS = torch.tensor(init["all_oscillator_strengths"], dtype=DTYPE)
    num_forest_lines = int(init["num_forest_lines"])
    return mu, M, log_omega, log_c_0, log_tau_0, log_beta, TW, OS, num_forest_lines


def _compute_lya_1pzs(rest, z_qsos, lya_rest=1215.67):
    """rest: (n_pix,); z_qsos: (N,) → returns lya_1pzs (N, n_pix)."""
    rest_t = torch.tensor(rest, dtype=DTYPE)
    z = torch.tensor(z_qsos, dtype=DTYPE)
    return (1.0 + z[:, None]) * rest_t[None, :] / lya_rest


def _full_batch_objective(spectrum_loss, mu, M, log_omega, log_c_0, log_tau_0, log_beta,
                          centered_fluxes, noise_variances, valid_masks,
                          lya_1pzs, z_qsos,
                          num_forest_lines, TW, OS,
                          dM_accum, dlog_omega_accum):
    """Loop over all spectra; accumulate gradients on .grad of params.
    Bypasses the buggy v1 wrapper. Returns total nlog_p (scalar, detached)."""
    omega2 = torch.exp(2 * log_omega)
    c_0 = torch.exp(log_c_0)
    tau_0 = torch.exp(log_tau_0)
    beta = torch.exp(log_beta)

    total = torch.zeros((), dtype=DTYPE)
    dM_accum.zero_()
    dlog_omega_accum.zero_()
    dlog_c_0_acc = torch.zeros((), dtype=DTYPE)
    dlog_tau_0_acc = torch.zeros((), dtype=DTYPE)
    dlog_beta_acc = torch.zeros((), dtype=DTYPE)

    n = centered_fluxes.shape[0]
    for i in range(n):
        valid = valid_masks[i]
        if not valid.any():
            continue
        y_m = centered_fluxes[i, valid]
        nv_m = noise_variances[i, valid]
        lya_1pz_m = lya_1pzs[i, valid]
        M_m = M[valid, :]
        omega2_m = omega2[valid]
        zqso_1pz = torch.tensor(float(z_qsos[i]) + 1.0, dtype=DTYPE)

        nlog_p, dM, dlog_omega, dlog_c_0, dlog_tau_0, dlog_beta = spectrum_loss(
            y_m, lya_1pz_m, nv_m, M_m, omega2_m,
            c_0, tau_0, beta,
            num_forest_lines, TW, OS, zqso_1pz,
        )
        total = total + nlog_p.detach()
        dM_accum[valid, :] += dM.detach()
        dlog_omega_accum[valid] += dlog_omega.detach()
        dlog_c_0_acc = dlog_c_0_acc + dlog_c_0.detach()
        dlog_tau_0_acc = dlog_tau_0_acc + dlog_tau_0.detach()
        dlog_beta_acc = dlog_beta_acc + dlog_beta.detach()

    # Apply BOSS DR12Q priors on log_tau_0, log_beta
    dlog_tau_0_acc = dlog_tau_0_acc + tau_0 * (tau_0 - TAU_0_PRIOR_MU) / TAU_0_PRIOR_SIGMA**2
    dlog_beta_acc = dlog_beta_acc + beta * (beta - BETA_PRIOR_MU) / BETA_PRIOR_SIGMA**2

    return total, dlog_c_0_acc, dlog_tau_0_acc, dlog_beta_acc


def _train_lbfgs_lane(lane, n_iters, init, train, t0_global,
                      use_strong_wolfe=True, history_size=10):
    """Step A.3 4th lane: torch.optim.LBFGS with strong-Wolfe line search.

    Uses v1 spectrum_loss (approximate dlog_β) for an apples-to-apples
    comparison with MATLAB's minFunc/L-BFGS — same loss kernel, same
    optimizer family. If endpoints match MATLAB to a small tolerance,
    that validates torch.optim.LBFGS for this problem class.
    """
    spectrum_loss = _import_spectrum_loss("v1")
    print(f"\n=== lane: {lane} (torch.optim.LBFGS, "
          f"line_search={'strong_wolfe' if use_strong_wolfe else 'None'}, "
          f"history={history_size}, n_iters={n_iters}) ===")

    mu, M, log_omega, log_c_0, log_tau_0, log_beta, TW, OS, num_forest_lines = \
        _build_init_params(init)

    centered_fluxes = torch.tensor(train["centered_fluxes"], dtype=DTYPE)
    centered_fluxes = torch.where(torch.isfinite(centered_fluxes), centered_fluxes,
                                  torch.zeros_like(centered_fluxes))
    noise_variances = torch.tensor(train["noise_variances"], dtype=DTYPE)
    valid_masks = torch.tensor(train["valid_masks"], dtype=torch.bool)
    z_qsos = train["z_qsos"]
    rest_wavelengths = init["rest_wavelengths"]
    lya_1pzs = _compute_lya_1pzs(rest_wavelengths, z_qsos)

    optimizer = torch.optim.LBFGS(
        [M, log_omega, log_c_0, log_tau_0, log_beta],
        lr=1.0,                # line search picks step internally
        max_iter=1,            # one L-BFGS step per .step() call so we can log
        history_size=history_size,
        line_search_fn="strong_wolfe" if use_strong_wolfe else None,
        tolerance_grad=0.0,    # we control termination ourselves
        tolerance_change=0.0,
    )

    history = dict(loss=[], log_c_0=[], log_tau_0=[], log_beta=[], wall_s=[])
    dM_accum = torch.zeros_like(M)
    dlog_omega_accum = torch.zeros_like(log_omega)
    last_loss = [None]   # closure-captured

    def closure():
        # 1) zero existing .grad on params
        optimizer.zero_grad()
        # 2) compute loss + manually-set gradients (mirrors v1 pattern)
        total, dlog_c_0_acc, dlog_tau_0_acc, dlog_beta_acc = _full_batch_objective(
            spectrum_loss, mu, M, log_omega, log_c_0, log_tau_0, log_beta,
            centered_fluxes, noise_variances, valid_masks,
            lya_1pzs, z_qsos, num_forest_lines, TW, OS,
            dM_accum, dlog_omega_accum,
        )
        # 3) populate .grad (L-BFGS reads .grad after closure() returns)
        with torch.no_grad():
            M.grad = dM_accum.clone()
            log_omega.grad = dlog_omega_accum.clone()
            log_c_0.grad = dlog_c_0_acc.clone()
            log_tau_0.grad = dlog_tau_0_acc.clone()
            log_beta.grad = dlog_beta_acc.clone()
        # 4) return loss as a tensor — L-BFGS uses it for line-search comparison
        loss_val = float(total)
        last_loss[0] = loss_val
        return torch.tensor(loss_val, dtype=DTYPE)

    for it in range(n_iters):
        t0 = time.time()
        optimizer.step(closure)
        dt = time.time() - t0
        history["loss"].append(last_loss[0])
        history["log_c_0"].append(float(log_c_0.detach()))
        history["log_tau_0"].append(float(log_tau_0.detach()))
        history["log_beta"].append(float(log_beta.detach()))
        history["wall_s"].append(float(dt))
        if it < 5 or it % 5 == 0 or it == n_iters - 1:
            wall = time.time() - t0_global
            print(f"  it={it:>3d}  loss={last_loss[0]:>14.4f}  "
                  f"τ_0={float(torch.exp(log_tau_0.detach())):.6f}  "
                  f"β={float(torch.exp(log_beta.detach())):.4f}  "
                  f"c_0={float(torch.exp(log_c_0.detach())):.6f}  "
                  f"({dt:.2f}s/iter, total {wall:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{lane}.npz"
    np.savez(out,
        lane=lane, n_iters=n_iters, lr=1.0,
        line_search="strong_wolfe" if use_strong_wolfe else "none",
        history_size=history_size,
        loss_history=np.asarray(history["loss"]),
        log_c_0_history=np.asarray(history["log_c_0"]),
        log_tau_0_history=np.asarray(history["log_tau_0"]),
        log_beta_history=np.asarray(history["log_beta"]),
        wall_s_history=np.asarray(history["wall_s"]),
        M_final=M.detach().numpy(),
        mu=mu.numpy(),
        log_omega_final=log_omega.detach().numpy(),
        log_c_0_final=float(log_c_0.detach()),
        log_tau_0_final=float(log_tau_0.detach()),
        log_beta_final=float(log_beta.detach()),
        c_0_final=float(torch.exp(log_c_0.detach())),
        tau_0_final=float(torch.exp(log_tau_0.detach())),
        beta_final=float(torch.exp(log_beta.detach())),
        rest_wavelengths=init["rest_wavelengths"],
    )
    print(f"  [saved] {out}")


def _train_one_lane(lane, n_iters, lr, init, train, t0_global):
    spectrum_loss = _import_spectrum_loss(lane)
    print(f"\n=== lane: {lane} (Adam, lr={lr}, n_iters={n_iters}) ===")

    mu, M, log_omega, log_c_0, log_tau_0, log_beta, TW, OS, num_forest_lines = \
        _build_init_params(init)

    centered_fluxes = torch.tensor(train["centered_fluxes"], dtype=DTYPE)
    centered_fluxes = torch.where(torch.isfinite(centered_fluxes), centered_fluxes, torch.zeros_like(centered_fluxes))
    noise_variances = torch.tensor(train["noise_variances"], dtype=DTYPE)
    valid_masks = torch.tensor(train["valid_masks"], dtype=torch.bool)
    z_qsos = train["z_qsos"]
    rest_wavelengths = init["rest_wavelengths"]
    lya_1pzs = _compute_lya_1pzs(rest_wavelengths, z_qsos)

    optimizer = torch.optim.Adam([M, log_omega, log_c_0, log_tau_0, log_beta], lr=lr)

    history = dict(loss=[], log_c_0=[], log_tau_0=[], log_beta=[], wall_s=[])
    dM_accum = torch.zeros_like(M)
    dlog_omega_accum = torch.zeros_like(log_omega)

    for it in range(n_iters):
        t0 = time.time()
        optimizer.zero_grad()
        total, dlog_c_0_acc, dlog_tau_0_acc, dlog_beta_acc = _full_batch_objective(
            spectrum_loss, mu, M, log_omega, log_c_0, log_tau_0, log_beta,
            centered_fluxes, noise_variances, valid_masks,
            lya_1pzs, z_qsos, num_forest_lines, TW, OS,
            dM_accum, dlog_omega_accum,
        )
        # Set .grad manually (mirrors v1 pattern; loss is detached)
        M.grad = dM_accum.clone()
        log_omega.grad = dlog_omega_accum.clone()
        log_c_0.grad = dlog_c_0_acc
        log_tau_0.grad = dlog_tau_0_acc
        log_beta.grad = dlog_beta_acc

        optimizer.step()
        dt = time.time() - t0

        history["loss"].append(float(total))
        history["log_c_0"].append(float(log_c_0.detach()))
        history["log_tau_0"].append(float(log_tau_0.detach()))
        history["log_beta"].append(float(log_beta.detach()))
        history["wall_s"].append(float(dt))

        if it % 5 == 0 or it == n_iters - 1:
            wall = time.time() - t0_global
            print(f"  it={it:>3d}  loss={float(total):>14.4f}  "
                  f"τ_0={float(torch.exp(log_tau_0)):.6f}  "
                  f"β={float(torch.exp(log_beta)):.4f}  "
                  f"c_0={float(torch.exp(log_c_0)):.6f}  "
                  f"({dt:.2f}s/iter, total {wall:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{lane}.npz"
    np.savez(out,
        lane=lane, n_iters=n_iters, lr=lr,
        loss_history=np.asarray(history["loss"]),
        log_c_0_history=np.asarray(history["log_c_0"]),
        log_tau_0_history=np.asarray(history["log_tau_0"]),
        log_beta_history=np.asarray(history["log_beta"]),
        wall_s_history=np.asarray(history["wall_s"]),
        M_final=M.detach().numpy(),
        mu=mu.numpy(),
        log_omega_final=log_omega.detach().numpy(),
        log_c_0_final=float(log_c_0.detach()),
        log_tau_0_final=float(log_tau_0.detach()),
        log_beta_final=float(log_beta.detach()),
        c_0_final=float(torch.exp(log_c_0.detach())),
        tau_0_final=float(torch.exp(log_tau_0.detach())),
        beta_final=float(torch.exp(log_beta.detach())),
        rest_wavelengths=init["rest_wavelengths"],
    )
    print(f"  [saved] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lane", choices=["v1", "v3.5", "lbfgs", "lbfgs-no-ls", "both"], default="both")
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    args = p.parse_args()

    init, train = _load_fixture()
    print(f"  loaded {train['centered_fluxes'].shape[0]} spectra "
          f"× {train['centered_fluxes'].shape[1]} pixels")
    print(f"  init c_0={float(init['c_0']):.4f}  tau_0={float(init['tau_0']):.5f}  "
          f"beta={float(init['beta']):.4f}")

    t0_global = time.time()
    if args.lane in ("v1", "both"):
        _train_one_lane("v1", args.n_iters, args.lr, init, train, t0_global)
    if args.lane in ("v3.5", "both"):
        _train_one_lane("v3.5", args.n_iters, args.lr, init, train, t0_global)
    if args.lane == "lbfgs":
        _train_lbfgs_lane("lbfgs", args.n_iters, init, train, t0_global,
                          use_strong_wolfe=True)
    if args.lane == "lbfgs-no-ls":
        # Diagnostic — torch L-BFGS WITHOUT line search (the "doesn't work"
        # configuration). For comparison.
        _train_lbfgs_lane("lbfgs-no-ls", args.n_iters, init, train, t0_global,
                          use_strong_wolfe=False)
    print(f"\n  total wall time: {time.time() - t0_global:.0f}s")


if __name__ == "__main__":
    main()
