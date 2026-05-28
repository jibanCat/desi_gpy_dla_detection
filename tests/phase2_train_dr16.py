"""Phase 2 — train v1 spectrum_loss on DR16 reference data, compare to MATLAB.

Re-uses the MATLAB-faithful preprocessing chain validated in
plot_corr_dr16_comparison.py (max_noise_variance=9 mask + de-forest 31 lines
+ row-median NaN fill + NaN-aware interp). Then runs full-batch Adam with
BOSS DR12Q priors for N iterations and saves endpoints.

Comparison artifacts at end:
  docs/notes/2026-05-08_matlab_dr16_validation/phase2_corr_compare.png
    4-panel: ours initial / ours trained / MATLAB initial / MATLAB final
  docs/notes/2026-05-08_matlab_dr16_validation/phase2_endpoint_table.md
    Final c_0, τ_0, β values: ours vs MATLAB

Usage:
    python tests/phase2_train_dr16.py [--n-spectra 5000] [--n-iters 50] [--lr 0.01]
"""
from __future__ import annotations

import os as _os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_name, "4")

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gpy_dla_detection.objective import spectrum_loss
from gpy_dla_detection.training_v3.objective_vectorized import spectrum_loss_batch
from gpy_dla_detection.effective_optical_depth import effective_optical_depth
from gpy_dla_detection import voigt as _v

REF_DIR = Path("/home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue")
PRELOAD = REF_DIR / "preloaded_qsos.mat"
CATALOG = REF_DIR / "catalog.mat"
LEARNED = REF_DIR / "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat"
OUT_DIR = REPO / "docs/notes/2026-05-08_matlab_dr16_validation"
# Default to scratch (home quota is 80 GiB, cache is ~5.5 GB at 89k spectra).
# Override with --cache-dir / --checkpoint-dir as needed.
SCRATCH_DEFAULT = Path("/scratch/cavestru_root/cavestru0/mfho/phase2_dr16")
CACHE_DIR_DEFAULT = SCRATCH_DEFAULT / "data_cache"
CHECKPOINT_DIR_DEFAULT = SCRATCH_DEFAULT / "checkpoints"
# Fallback locations (if the original home cache exists, use it transparently).
CACHE_DIR_HOME = REPO / "tests/fixtures/dr16_phase2_cache"
PCA_INIT = REPO / "tests/fixtures/dr16_pca_init.npz"

# Set by main(); used by _build_data_cache and _train.
_RUNTIME = {"cache_dir": CACHE_DIR_DEFAULT, "checkpoint_dir": CHECKPOINT_DIR_DEFAULT,
            "save_now": False}

DTYPE = torch.float64
MIN_LAMBDA, MAX_LAMBDA, DLAMBDA = 850.75, 1420.75, 0.25
N_PIX = int(round((MAX_LAMBDA - MIN_LAMBDA) / DLAMBDA)) + 1
NUM_FOREST_LINES = 31
DEFOREST_TAU_0, DEFOREST_BETA = 0.00554, 3.182
MAX_NV = 9.0
K = 20
INITIAL_C_0 = 0.1
INITIAL_TAU_0 = 0.00554
INITIAL_BETA = 3.182
TAU_0_PRIOR_MU, TAU_0_PRIOR_SIGMA = 0.00554, 0.00064
BETA_PRIOR_MU, BETA_PRIOR_SIGMA = 3.182, 0.074


def _build_data_cache(n_spectra=None):
    """Same preprocessing as plot_corr_dr16_comparison.py:_build_cache, but
    saves the per-spectrum centered_fluxes/nv/lya_1pzs needed for training.

    Looks for the cache file at (in order): _RUNTIME['cache_dir'], then the
    legacy CACHE_DIR_HOME path. Builds in _RUNTIME['cache_dir'] if missing.
    """
    cache_dir = _RUNTIME["cache_dir"]
    fname = f"data_cache_n{n_spectra or 'all'}.npz"
    primary = cache_dir / fname
    legacy = CACHE_DIR_HOME / fname
    if primary.exists():
        print(f"[cache] using existing {primary}")
        return primary
    if legacy.exists():
        print(f"[cache] using legacy home cache {legacy}")
        return legacy
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = primary

    print("[cache] loading train_ind from learned_qso_model.mat ...")
    with h5py.File(LEARNED, "r") as f:
        train_ind = np.asarray(f["train_ind"])[0].astype(bool)
    train_idx = np.where(train_ind)[0]
    if n_spectra is not None:
        rng = np.random.default_rng(0)
        train_idx = np.sort(rng.choice(train_idx, size=n_spectra, replace=False))
    n = len(train_idx)
    print(f"[cache] subset = {n} train_ind QSOs")
    rest_grid = np.linspace(MIN_LAMBDA, MAX_LAMBDA, N_PIX)

    with h5py.File(CATALOG, "r") as f:
        z_qsos = np.asarray(f["z_qsos"])[0][train_idx].astype(np.float64)

    interp_flux = np.full((n, N_PIX), np.nan, dtype=np.float64)
    interp_nv = np.full((n, N_PIX), np.nan, dtype=np.float64)

    print("[cache] reading spectra + masking + interpolating ...")
    with h5py.File(PRELOAD, "r") as f:
        all_flux = f["all_flux"]; all_nv = f["all_noise_variance"]
        all_wave = f["all_wavelengths"]; all_mask = f["all_pixel_mask"]
        t0 = time.time()
        for i, qi in enumerate(train_idx):
            if i % 10000 == 0 and i > 0:
                print(f"  [cache] {i}/{n}  ({time.time()-t0:.0f}s)")
            wave = np.asarray(f[all_wave[0, qi]])[0]
            flux = np.asarray(f[all_flux[0, qi]])[0]
            nv = np.asarray(f[all_nv[0, qi]])[0]
            mask = np.asarray(f[all_mask[0, qi]])[0].astype(bool)
            bad = mask | ~np.isfinite(flux) | ~np.isfinite(nv) | ~(nv > 0)
            flux_nan = flux.copy(); flux_nan[bad] = np.nan
            nv_nan = nv.copy();     nv_nan[bad] = np.nan
            if int(np.sum(~bad)) < 100:
                continue
            wave_rest = wave / (1.0 + float(z_qsos[i]))
            idx = np.searchsorted(wave_rest, rest_grid) - 1
            valid_grid = (idx >= 0) & (idx < len(wave_rest) - 1)
            idx_safe = np.clip(idx, 0, len(wave_rest) - 2)
            t = (rest_grid - wave_rest[idx_safe]) / (
                wave_rest[idx_safe + 1] - wave_rest[idx_safe])
            f_interp = (1.0 - t) * flux_nan[idx_safe] + t * flux_nan[idx_safe + 1]
            n_interp = (1.0 - t) * nv_nan[idx_safe]   + t * nv_nan[idx_safe + 1]
            f_interp[~valid_grid] = np.nan
            n_interp[~valid_grid] = np.nan
            interp_flux[i] = f_interp
            interp_nv[i] = n_interp
        print(f"  [cache] done in {time.time()-t0:.0f}s")

    # max_noise_variance mask
    noisy = interp_nv > MAX_NV
    interp_flux[noisy] = np.nan
    interp_nv[noisy] = np.nan

    # de-forest with 31 lines
    print("[cache] de-forest 31 lines ...")
    deforest_flux = np.empty_like(interp_flux)
    deforest_nv = np.empty_like(interp_nv)
    LYA_REST = float(_v.transition_wavelengths[0]) * 1e8
    lya_1pzs = np.empty((n, N_PIX), dtype=np.float64)
    for i in range(n):
        obs_wave = rest_grid * (1.0 + float(z_qsos[i]))
        tau = effective_optical_depth(
            obs_wave, DEFOREST_BETA, DEFOREST_TAU_0,
            float(z_qsos[i]), num_forest_lines=NUM_FOREST_LINES,
        ).sum(axis=1)
        c = np.exp(tau)
        deforest_flux[i] = interp_flux[i] * c
        deforest_nv[i] = interp_nv[i] * c ** 2
        lya_1pzs[i] = (1.0 + float(z_qsos[i])) * rest_grid / LYA_REST

    # center using inverse-variance-weighted mu (matches MATLAB)
    valid = np.isfinite(deforest_flux) & np.isfinite(deforest_nv) & (deforest_nv > 0)
    weights = np.where(valid, 1.0 / deforest_nv, 0.0)
    weighted_sum = np.where(valid, deforest_flux * weights, 0.0).sum(axis=0)
    weight_sum = weights.sum(axis=0)
    mu = np.where(weight_sum > 0, weighted_sum / weight_sum, 0.0)
    centered = np.where(valid, deforest_flux - mu[None, :], np.nan)
    print(f"  [cache] mu computed; centered shape {centered.shape}")

    valid_masks = np.isfinite(centered) & np.isfinite(deforest_nv) & (deforest_nv > 0)
    np.savez(cache_path,
             rest_wavelengths=rest_grid, mu=mu,
             centered_fluxes=centered.astype(np.float64),
             noise_variances=deforest_nv.astype(np.float64),
             lya_1pzs=lya_1pzs.astype(np.float64),
             valid_masks=valid_masks,
             z_qsos=z_qsos.astype(np.float64),
             target_idx=train_idx.astype(np.int64))
    print(f"[saved] {cache_path}  ({cache_path.stat().st_size/1e6:.1f} MB)")
    return cache_path


def _pca_init(centered, k=K, random_state=0):
    """Match MATLAB: row-median NaN fill, then PCA, then M = coeff·sqrt(eigval).

    sklearn's PCA `auto` solver picks `randomized` SVD for n_components << min(n_samples,
    n_features) (true here: k=20 vs (n_spectra, n_pix) = (89408, 2281)). Without an
    explicit random_state, randomized SVD seeds itself per-call → run-to-run drift in
    the top-k eigenvector basis (~1e-7 in eigenvectors, amplifying to ~1e-4 at iter-0
    loss → ~12% relative |dM| over 50+ Adam iters even when gradients are
    bit-identical between paths). See docs/notes/2026-05-09_vec_smoke_vs_phase1_baseline.md.

    Setting random_state=0 makes M_init bit-reproducible across runs — so two retrains
    on the same cached data with the same code path produce identical trained M to f64
    noise. This is the right default for production / regression testing. Override only
    if intentionally probing init sensitivity.
    """
    from sklearn.decomposition import PCA
    pca_input = centered.copy()
    for i in range(pca_input.shape[0]):
        row = pca_input[i]
        finite = np.isfinite(row)
        if finite.any():
            row[~finite] = np.nanmedian(row)
            pca_input[i] = row
        else:
            pca_input[i] = 0.0
    pca_input = np.nan_to_num(pca_input, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(pca_input)
    return pca.components_.T * np.sqrt(pca.explained_variance_)[None, :], pca.explained_variance_


def _train(centered, nv, lya_1pzs, valid_masks, z_qsos, mu, M_init, log_omega_init,
           num_forest_lines, n_iters, lr, checkpoint_every=5, resume_path=None,
           max_walltime_sec=None, vectorized=True, chunk_size=1000):
    """Adam loop matching tests/short_retrain_2lpt.py:_full_batch_objective.
    Bypasses v1's objective.py wrapper (zqso_1pz=z_qso+1 directly).
    Applies BOSS DR12Q priors on log_τ_0 and log_β.

    Two paths give numerically equivalent gradients (verified by
    tests/test_v3_objective_vectorized_parity.py and
    tests/test_v3_train_step_parity.py to ~1e-10 / 2e-10 over 3 Adam iters):

      vectorized=True (default):  spectrum_loss_batch on chunks of `chunk_size`
                                  spectra. Lifts the OMP=1 thread-storm
                                  constraint of the per-spectrum loop.
      vectorized=False:           per-spectrum Python loop calling v1's
                                  spectrum_loss; reference path retained for
                                  cross-validation.

    Saves a checkpoint to _RUNTIME['checkpoint_dir'] every `checkpoint_every`
    iterations, on SIGTERM/SIGINT, or when wall elapsed exceeds max_walltime_sec.
    """
    from gpy_dla_detection.voigt import (
        transition_wavelengths as TW, oscillator_strengths as OS)
    TW_t = torch.tensor(np.asarray(TW), dtype=DTYPE)
    OS_t = torch.tensor(np.asarray(OS), dtype=DTYPE)

    M = nn.Parameter(torch.tensor(M_init, dtype=DTYPE))
    log_omega = nn.Parameter(torch.tensor(log_omega_init, dtype=DTYPE))
    log_c_0 = nn.Parameter(torch.tensor(np.log(INITIAL_C_0), dtype=DTYPE))
    log_tau_0 = nn.Parameter(torch.tensor(np.log(INITIAL_TAU_0), dtype=DTYPE))
    log_beta = nn.Parameter(torch.tensor(np.log(INITIAL_BETA), dtype=DTYPE))

    centered_t = torch.tensor(np.where(valid_masks, centered, 0.0), dtype=DTYPE)
    # Sanitize NaN noise variances at invalid pixels once. Both paths mask out
    # invalid contributions internally; the vectorized path additionally needs
    # finite values everywhere so torch.where/cholesky never sees NaN.
    nv_t = torch.tensor(np.where(valid_masks, nv, 1.0), dtype=DTYPE)
    lya_1pzs_t = torch.tensor(lya_1pzs, dtype=DTYPE)
    valid_t = torch.tensor(valid_masks, dtype=torch.bool)
    zqso_1pz_t = torch.tensor(np.asarray(z_qsos) + 1.0, dtype=DTYPE)
    n = centered.shape[0]

    optimizer = torch.optim.Adam([M, log_omega, log_c_0, log_tau_0, log_beta], lr=lr)
    history = dict(loss=[], log_c_0=[], log_tau_0=[], log_beta=[])
    start_iter = 0
    if resume_path is not None:
        rp = Path(resume_path)
        print(f"\n[resume] loading checkpoint from {rp}")
        ckpt = torch.load(rp, map_location="cpu", weights_only=False)
        with torch.no_grad():
            M.copy_(ckpt["M"])
            log_omega.copy_(ckpt["log_omega"])
            log_c_0.copy_(ckpt["log_c_0"])
            log_tau_0.copy_(ckpt["log_tau_0"])
            log_beta.copy_(ckpt["log_beta"])
        optimizer.load_state_dict(ckpt["optim_state"])
        history = {k: list(v) for k, v in ckpt["history"].items()}
        start_iter = int(ckpt["iter_completed"]) + 1
        print(f"[resume] resuming at iter={start_iter} "
              f"(prior loss={history['loss'][-1]:.4f})")

    dM_accum = torch.zeros_like(M)
    dlog_omega_accum = torch.zeros_like(log_omega)

    ckpt_dir = _RUNTIME["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(it_done, tag="iter"):
        cpath = ckpt_dir / f"phase2_checkpoint_{tag}{it_done:04d}.pt"
        torch.save(dict(
            M=M.detach().clone(),
            log_omega=log_omega.detach().clone(),
            log_c_0=log_c_0.detach().clone(),
            log_tau_0=log_tau_0.detach().clone(),
            log_beta=log_beta.detach().clone(),
            optim_state=optimizer.state_dict(),
            iter_completed=int(it_done),
            history=history,
            mu=mu,
        ), cpath)
        print(f"[checkpoint] saved {cpath} (iter {it_done})")
        return cpath

    # Signal handler: set save_now flag; loop checks it at next iter boundary.
    def _on_signal(signum, _frame):
        print(f"\n[signal] caught {signum}, requesting graceful save at next iter boundary")
        _RUNTIME["save_now"] = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    train_t0 = time.time()
    print(f"\n=== Training: {n} spectra, {n_iters} iter, lr={lr}, "
          f"start_iter={start_iter}, checkpoint_every={checkpoint_every} ===")
    it = start_iter - 1  # ensure defined after loop, even if it doesn't run
    for it in range(start_iter, n_iters):
        t0 = time.time()
        optimizer.zero_grad()
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

        if vectorized:
            # Chunked vectorized path: spectrum_loss_batch over slices of size
            # `chunk_size`. Sums are accumulated across chunks; final result is
            # bit-equivalent to the per-spectrum path within f64 noise (verified
            # by tests/test_v3_train_step_parity.py).
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                nlp_c, dM_c, dlogw_c, dc0_c, dt0_c, db_c = spectrum_loss_batch(
                    centered_t[s:e], lya_1pzs_t[s:e], nv_t[s:e], valid_t[s:e],
                    M, omega2, c_0, tau_0, beta,
                    num_forest_lines, TW_t, OS_t,
                    zqso_1pz_t[s:e],
                )
                total = total + nlp_c.detach()
                dM_accum.add_(dM_c.detach())
                dlog_omega_accum.add_(dlogw_c.detach())
                dlog_c_0_acc = dlog_c_0_acc + dc0_c.detach()
                dlog_tau_0_acc = dlog_tau_0_acc + dt0_c.detach()
                dlog_beta_acc = dlog_beta_acc + db_c.detach()
        else:
            # Per-spectrum reference path (the v1 loop). Retained for
            # cross-validation; `vectorized=False` selects this.
            for i in range(n):
                valid_i = valid_t[i]
                if not valid_i.any():
                    continue
                y = centered_t[i, valid_i]
                nv_i = nv_t[i, valid_i]
                lya_1pz_i = lya_1pzs_t[i, valid_i]
                M_i = M[valid_i, :]
                omega2_i = omega2[valid_i]
                zqso_1pz_i = zqso_1pz_t[i]

                nlog_p, dM_i, dlog_omega_i, dlog_c_0_i, dlog_tau_0_i, dlog_beta_i = \
                    spectrum_loss(y, lya_1pz_i, nv_i, M_i, omega2_i,
                                  c_0, tau_0, beta, num_forest_lines, TW_t, OS_t,
                                  zqso_1pz_i)
                total = total + nlog_p.detach()
                dM_accum[valid_i, :] += dM_i.detach()
                dlog_omega_accum[valid_i] += dlog_omega_i.detach()
                dlog_c_0_acc = dlog_c_0_acc + dlog_c_0_i.detach()
                dlog_tau_0_acc = dlog_tau_0_acc + dlog_tau_0_i.detach()
                dlog_beta_acc = dlog_beta_acc + dlog_beta_i.detach()

        dlog_tau_0_acc = dlog_tau_0_acc + tau_0 * (tau_0 - TAU_0_PRIOR_MU) / TAU_0_PRIOR_SIGMA**2
        dlog_beta_acc = dlog_beta_acc + beta * (beta - BETA_PRIOR_MU) / BETA_PRIOR_SIGMA**2

        with torch.no_grad():
            M.grad = dM_accum.clone()
            log_omega.grad = dlog_omega_accum.clone()
            log_c_0.grad = dlog_c_0_acc.clone()
            log_tau_0.grad = dlog_tau_0_acc.clone()
            log_beta.grad = dlog_beta_acc.clone()

        optimizer.step()
        dt = time.time() - t0
        history["loss"].append(float(total))
        history["log_c_0"].append(float(log_c_0.detach()))
        history["log_tau_0"].append(float(log_tau_0.detach()))
        history["log_beta"].append(float(log_beta.detach()))
        if it < 3 or it % 5 == 0 or it == n_iters - 1:
            print(f"  it={it:>3d}  loss={float(total):>14.4f}  "
                  f"τ_0={float(tau_0.detach()):.6f}  "
                  f"β={float(beta.detach()):.4f}  "
                  f"c_0={float(c_0.detach()):.6f}  ({dt:.2f}s/iter)")

        # Periodic checkpoint
        if checkpoint_every and ((it + 1) % checkpoint_every == 0):
            _save_checkpoint(it)

        # Walltime budget exceeded → save and bail
        if max_walltime_sec is not None and (time.time() - train_t0) > max_walltime_sec:
            print(f"[walltime] elapsed > {max_walltime_sec}s; saving and exiting at iter={it}")
            _save_checkpoint(it, tag="walltime_exit_iter")
            break

        # SIGTERM / SIGINT received → save and bail
        if _RUNTIME["save_now"]:
            _save_checkpoint(it, tag="signal_exit_iter")
            break

    # Final checkpoint at clean exit
    _save_checkpoint(it, tag="final_iter")

    return dict(M=M.detach().numpy(),
                mu=mu,
                log_omega=log_omega.detach().numpy(),
                log_c_0=float(log_c_0.detach()),
                log_tau_0=float(log_tau_0.detach()),
                log_beta=float(log_beta.detach()),
                c_0=float(torch.exp(log_c_0.detach())),
                tau_0=float(torch.exp(log_tau_0.detach())),
                beta=float(torch.exp(log_beta.detach())),
                history=history)


def _corr(M):
    K_mat = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K_mat), 1e-30))
    return np.clip(K_mat / np.outer(d, d), -1.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-spectra", type=int, default=5000,
                   help="Subsample of train_ind for training (default 5000; full = 89408)")
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR_DEFAULT,
                   help=f"Where to write/read the preprocessed npz cache "
                        f"(default: {CACHE_DIR_DEFAULT}). Falls back to "
                        f"{CACHE_DIR_HOME} if a cache file is present there.")
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR_DEFAULT,
                   help=f"Where to write training checkpoints "
                        f"(default: {CHECKPOINT_DIR_DEFAULT}).")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save a checkpoint every N iterations (default 5; 0 disables periodic).")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a .pt checkpoint file to resume from.")
    p.add_argument("--max-walltime-sec", type=int, default=None,
                   help="If set, save and exit when training elapsed exceeds this (seconds).")
    p.add_argument("--vectorized", type=int, default=1,
                   help="1 = use spectrum_loss_batch (default; lifts OMP=1 thread cap); "
                        "0 = use per-spectrum loop (reference path).")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Batch chunk size for vectorized path (default 1000; "
                        "memory ~ chunk * N_PIX * k * 16B).")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help=f"Where to write the final phase2_result.npz, "
                        f"phase2_corr_compare.png, and phase2_endpoint_table.md "
                        f"(default: {OUT_DIR}). Use a separate dir for parallel "
                        f"runs to avoid clobbering each other.")
    args = p.parse_args()

    _RUNTIME["cache_dir"] = args.cache_dir
    _RUNTIME["checkpoint_dir"] = args.checkpoint_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = _build_data_cache(n_spectra=args.n_spectra)
    cache = np.load(cache_path)
    rest = cache["rest_wavelengths"]
    centered = cache["centered_fluxes"]
    nv = cache["noise_variances"]
    lya_1pzs = cache["lya_1pzs"]
    valid_masks = cache["valid_masks"]
    z_qsos = cache["z_qsos"]
    mu = cache["mu"]
    print(f"[loaded] {centered.shape[0]} spectra × {centered.shape[1]} pixels")

    # PCA init
    print("[PCA] init M with k=20 ...")
    M_init, latent = _pca_init(centered, k=K)
    log_omega_init = np.log(np.nanstd(centered, axis=0) + 1e-12)
    print(f"  M_init shape={M_init.shape}; top-5 eigvals={latent[:5].tolist()}")

    # Train
    result = _train(centered, nv, lya_1pzs, valid_masks, z_qsos, mu,
                    M_init, log_omega_init, NUM_FOREST_LINES,
                    args.n_iters, args.lr,
                    checkpoint_every=args.checkpoint_every,
                    resume_path=args.resume,
                    max_walltime_sec=args.max_walltime_sec,
                    vectorized=bool(args.vectorized),
                    chunk_size=args.chunk_size)

    # Save — primary is .h5 in DESI learned-model schema (the same format
    # as learnlogs/model_epoch_*.h5 that null_gp.py:453-468 reads). The
    # .npz is kept as a training-history record (loss/log_*_history etc.),
    # NOT the learned model — see feedback_learned_model_h5_format.md.
    out_h5 = out_dir / "phase2_result.h5"
    # Best-effort git SHA for provenance (matches phase2_train_desi)
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    from datetime import datetime, timezone

    with h5py.File(out_h5, "w") as f:
        # --- 1. Trained kernel (v1 schema, matches null_gp DESI branch) ---
        f.create_dataset("M", data=np.asarray(result["M"], dtype=np.float64))
        f.create_dataset("mu", data=np.asarray(result["mu"], dtype=np.float64))
        f.create_dataset("log_omega", data=np.asarray(result["log_omega"], dtype=np.float64))
        f.create_dataset("log_c_0", data=np.float64(result["log_c_0"]))
        f.create_dataset("log_tau_0", data=np.float64(result["log_tau_0"]))
        f.create_dataset("log_beta", data=np.float64(result["log_beta"]))
        f.create_dataset("rest_wavelengths", data=np.asarray(rest, dtype=np.float64))
        # --- 2. Loader-required scalars ---
        f.create_dataset("max_noise_variance", data=np.float64(MAX_NV))
        f.create_dataset("normalization_min_lambda", data=np.float64(1425.0))
        f.create_dataset("normalization_max_lambda", data=np.float64(1475.0))
        # Note: MATLAB DR16 (preload_qsos.m) normalizes at [1425, 1475] —
        # see docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md.
        # The earlier [1310, 1325] in this trainer was wrong; this is the
        # MATLAB-faithful band.
        # --- 3. Training-hyperparameter manifest (MATLAB-style flat 0-d) ---
        f.create_dataset("num_forest_lines", data=np.int64(NUM_FOREST_LINES))
        f.create_dataset("k",                data=np.int64(K))
        f.create_dataset("n_spectra",        data=np.int64(args.n_spectra))
        f.create_dataset("n_iters",          data=np.int64(args.n_iters))
        f.create_dataset("lr",               data=np.float64(args.lr))
        f.create_dataset("de_forest_tau_0",  data=np.float64(INITIAL_TAU_0))
        f.create_dataset("de_forest_beta",   data=np.float64(INITIAL_BETA))
        f.create_dataset("tau_0_prior_mu",   data=np.float64(TAU_0_PRIOR_MU))
        f.create_dataset("tau_0_prior_sigma", data=np.float64(TAU_0_PRIOR_SIGMA))
        f.create_dataset("beta_prior_mu",    data=np.float64(BETA_PRIOR_MU))
        f.create_dataset("beta_prior_sigma", data=np.float64(BETA_PRIOR_SIGMA))
        f.create_dataset("pca_random_state", data=np.int64(0))  # pinned in _pca_init
        f.create_dataset("chunk_size",       data=np.int64(getattr(args, "chunk_size", 0)))
        f.create_dataset("vectorized",       data=np.int64(bool(args.vectorized)))
        f.create_dataset("normalize_then_mask_order", data=np.int64(1))
        f.create_dataset("optimizer",        data=np.bytes_("Adam"))
        f.create_dataset("training_release", data=np.bytes_("PR6_StepA_DR16"))
        f.create_dataset("git_commit_sha",   data=np.bytes_(git_sha))
        f.create_dataset("training_timestamp",
                         data=np.bytes_(datetime.now(timezone.utc).isoformat()))
        # --- legacy attrs (kept for backward compat) ---
        f.attrs["n_spectra"] = int(args.n_spectra)
        f.attrs["n_iters"] = int(args.n_iters)
        f.attrs["lr"] = float(args.lr)
        f.attrs["vectorized"] = int(bool(args.vectorized))
    print(f"[saved] {out_h5}")

    out_npz = out_dir / "phase2_result.npz"
    np.savez(out_npz, rest_wavelengths=rest, **{k: result[k] for k in
             ["M", "mu", "log_omega", "log_c_0", "log_tau_0", "log_beta",
              "c_0", "tau_0", "beta"]},
             loss_history=np.asarray(result["history"]["loss"]),
             log_c_0_history=np.asarray(result["history"]["log_c_0"]),
             log_tau_0_history=np.asarray(result["history"]["log_tau_0"]),
             log_beta_history=np.asarray(result["history"]["log_beta"]),
             n_spectra=args.n_spectra, n_iters=args.n_iters, lr=args.lr)
    print(f"[saved] {out_npz} (training-history record; learned model is .h5 above)")

    # Compare to MATLAB
    with h5py.File(LEARNED, "r") as f:
        M_init_ref = np.asarray(f["initial_M"]).T
        M_final_ref = np.asarray(f["M"]).T
        c_0_ref = float(np.exp(np.asarray(f["log_c_0"])[0, 0]))
        tau_0_ref = float(np.exp(np.asarray(f["log_tau_0"])[0, 0]))
        beta_ref = float(np.exp(np.asarray(f["log_beta"])[0, 0]))

    # 4-panel corr matrices
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    extent = [rest[0], rest[-1], rest[-1], rest[0]]
    panels = [
        ("ours initial (PCA)", M_init),
        (f"ours trained (Adam, {args.n_spectra} spectra, {args.n_iters} iter)", result["M"]),
        ("MATLAB initial_M", M_init_ref),
        ("MATLAB final M", M_final_ref),
    ]
    for ax, (title, M) in zip(axes.flat, panels):
        if M.shape[0] != rest.shape[0]:
            M = M.T
        im = ax.imshow(_corr(M), cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                        interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("λ′ [Å]"); ax.set_ylabel("λ [Å]")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55,
                  location="right", label="correlation", pad=0.02)
    fig.suptitle("Phase 2: corr(M·M^T) — ours initial / ours trained / MATLAB initial / MATLAB final",
                 fontsize=11, fontweight="bold")
    out = out_dir / "phase2_corr_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")

    # Endpoint table
    rows = ["# Phase 2: trained scalars — ours vs MATLAB DR16",
            "",
            f"## Setup", f"- training subset: {args.n_spectra} of 89408 train_ind QSOs",
            f"- iterations: {args.n_iters}",  f"- optimizer: Adam, lr={args.lr}",
            f"- priors: BOSS DR12Q (τ_0 ~ N(0.00554, 0.00064²); β ~ N(3.182, 0.074²))",
            "",
            "| param | ours (trained) | MATLAB (trained) | Δ |",
            "|---|---:|---:|---:|"]
    for name, ours_val, ref_val in [
        ("c_0", result["c_0"], c_0_ref),
        ("τ_0", result["tau_0"], tau_0_ref),
        ("β",  result["beta"], beta_ref),
    ]:
        rows.append(f"| {name} | {ours_val:.6f} | {ref_val:.6f} | {ours_val-ref_val:+.6f} |")
    md_out = out_dir / "phase2_endpoint_table.md"
    md_out.write_text("\n".join(rows) + "\n")
    print(f"[saved] {md_out}")


if __name__ == "__main__":
    main()
