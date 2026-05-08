"""3-panel corr(M·M^T) comparison: ours initial vs MATLAB initial vs MATLAB final.

Reads:
  /home/mfho/MATLAB/.../learned_qso_model_..._851-1421.mat
  tests/fixtures/dr16_pca_init.npz   (cached from Phase 1; built if absent)
"""
from __future__ import annotations

import os
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_n, "4")

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
REF_DIR = Path("/home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue")
LEARNED = REF_DIR / "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat"
PRELOAD = REF_DIR / "preloaded_qsos.mat"
CATALOG = REF_DIR / "catalog.mat"
OUT_DIR = REPO / "docs/notes/2026-05-08_matlab_dr16_validation"
CACHE = REPO / "tests/fixtures/dr16_pca_init.npz"

MIN_LAMBDA, MAX_LAMBDA, DLAMBDA, NORM_LO, NORM_HI = 850.75, 1420.75, 0.25, 1425.0, 1475.0
N_PIX = int(round((MAX_LAMBDA - MIN_LAMBDA) / DLAMBDA)) + 1
NUM_FOREST_LINES = 31
DEFOREST_TAU_0, DEFOREST_BETA = 0.00554, 3.182
K = 20


def _build_cache():
    """Identical to validate_against_matlab_dr16.py phase1, saves M, mu, log_omega, rest."""
    from gpy_dla_detection.effective_optical_depth import effective_optical_depth

    print("[cache] loading 89408 train_ind QSOs ...")
    with h5py.File(LEARNED, "r") as f:
        train_ind = np.asarray(f["train_ind"])[0].astype(bool)
    train_idx = np.where(train_ind)[0]
    n = len(train_idx)
    rest_grid = np.linspace(MIN_LAMBDA, MAX_LAMBDA, N_PIX)

    with h5py.File(CATALOG, "r") as f:
        z_qsos = np.asarray(f["z_qsos"])[0][train_idx].astype(np.float64)

    interp_flux = np.full((n, N_PIX), np.nan, dtype=np.float64)
    interp_nv = np.full((n, N_PIX), np.nan, dtype=np.float64)

    with h5py.File(PRELOAD, "r") as f:
        all_flux = f["all_flux"]; all_nv = f["all_noise_variance"]
        all_wave = f["all_wavelengths"]; all_mask = f["all_pixel_mask"]
        t0 = time.time()
        for i, qi in enumerate(train_idx):
            if i % 20000 == 0 and i > 0:
                print(f"  [cache] {i}/{n}  ({time.time()-t0:.0f}s)")
            wave = np.asarray(f[all_wave[0, qi]])[0]
            flux = np.asarray(f[all_flux[0, qi]])[0]
            nv = np.asarray(f[all_nv[0, qi]])[0]
            mask = np.asarray(f[all_mask[0, qi]])[0].astype(bool)
            # 2026-05-08 fix per debug-agent #2: MATLAB interp1 keeps full
            # wavelength axis with masked pixels set to NaN; rest-grid points
            # bracketed by a NaN return NaN. np.interp silently linearly
            # bridges masked pixels — produces noise at sky-line gaps. Use
            # NaN-aware linear interp here.
            bad = mask | ~np.isfinite(flux) | ~np.isfinite(nv) | ~(nv > 0)
            flux_nan = flux.copy(); flux_nan[bad] = np.nan
            nv_nan = nv.copy();     nv_nan[bad] = np.nan
            n_finite = int(np.sum(~bad))
            if n_finite < 100: continue
            wave_rest = wave / (1.0 + float(z_qsos[i]))
            # Linear interp where both bracketing points are finite, else NaN
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

    # 2026-05-08 fix per debug-agent report: MATLAB masks pixels with
    # noise_variance > max_noise_variance = 3² = 9 BEFORE de-forest + PCA.
    # See learn_qso_model.m:127-131 and set_parameters.m:38.
    MAX_NV = 9.0
    noisy = interp_nv > MAX_NV
    print(f"[cache] max_noise_variance={MAX_NV}: {noisy.sum()} pixels masked "
          f"({noisy.mean()*100:.2f}% of all entries)")
    interp_flux[noisy] = np.nan
    interp_nv[noisy] = np.nan

    # de-forest with 31 lines
    print("[cache] de-forest ...")
    deforest_flux = np.empty_like(interp_flux)
    deforest_nv = np.empty_like(interp_nv)
    for i in range(n):
        obs_wave = rest_grid * (1.0 + float(z_qsos[i]))
        tau = effective_optical_depth(obs_wave, DEFOREST_BETA, DEFOREST_TAU_0,
                                       float(z_qsos[i]),
                                       num_forest_lines=NUM_FOREST_LINES).sum(axis=1)
        c = np.exp(tau)
        deforest_flux[i] = interp_flux[i] * c
        deforest_nv[i] = interp_nv[i] * c ** 2

    # center
    valid = np.isfinite(deforest_flux) & np.isfinite(deforest_nv) & (deforest_nv > 0)
    weights = np.where(valid, 1.0 / deforest_nv, 0.0)
    weighted_sum = np.where(valid, deforest_flux * weights, 0.0).sum(axis=0)
    weight_sum = weights.sum(axis=0)
    mu = np.where(weight_sum > 0, weighted_sum / weight_sum, 0.0)
    centered = np.where(valid, deforest_flux - mu[None, :], np.nan)

    # 2026-05-08: matches MATLAB learn_qso_model.m EXACTLY (read at line ~199):
    # MATLAB fills NaN with row-median, NOT listwise deletion. The
    # 'rows','complete' option is redundant once row-median fill is applied.
    # The PCA then operates on full N=89k spectra.
    pca_input = centered.copy()
    for i in range(pca_input.shape[0]):
        row = pca_input[i]
        finite = np.isfinite(row)
        if finite.any():
            row[~finite] = np.nanmedian(row)
            pca_input[i] = row
        else:
            pca_input[i] = 0.0
    # Drop fully-empty spectra
    keep = np.any(np.isfinite(centered), axis=1)
    pca_input = pca_input[keep]
    pca_input = np.nan_to_num(pca_input, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[cache] PCA input: {pca_input.shape[0]} spectra (row-median NaN fill, "
          f"matches MATLAB)")
    pca = PCA(n_components=K)
    pca.fit(pca_input)
    M = pca.components_.T * np.sqrt(pca.explained_variance_)[None, :]
    log_omega = np.log(np.nanstd(centered, axis=0) + 1e-12)
    np.savez(CACHE, rest=rest_grid, M=M, mu=mu, log_omega=log_omega,
             latent=pca.explained_variance_,
             n_pca_input=int(pca_input.shape[0]),
             n_train=n)
    print(f"[saved] {CACHE}")


def _corr(M):
    K_mat = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K_mat), 1e-30))
    return np.clip(K_mat / np.outer(d, d), -1.0, 1.0)


def main():
    if not CACHE.exists():
        _build_cache()
    cache = np.load(CACHE)
    rest = cache["rest"]
    M_ours = cache["M"]
    n_in = int(cache.get("n_pca_input", cache.get("n_complete", 0)))
    print(f"ours: M shape={M_ours.shape}, n_pca_input={n_in}/{int(cache['n_train'])}")

    with h5py.File(LEARNED, "r") as f:
        rest_ref = np.asarray(f["rest_wavelengths"])[:, 0]
        M_init_ref = np.asarray(f["initial_M"]).T
        M_final_ref = np.asarray(f["M"]).T
    assert np.allclose(rest, rest_ref), "rest grids don't match"
    print(f"MATLAB: initial_M shape={M_init_ref.shape}, final M shape={M_final_ref.shape}")

    # Compute correlation matrices
    print("computing corr matrices...")
    C_ours = _corr(M_ours)
    C_init = _corr(M_init_ref)
    C_final = _corr(M_final_ref)

    # Side-by-side plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    extent = [rest[0], rest[-1], rest[-1], rest[0]]
    im = None
    for ax, C, title in zip(
        axes,
        [C_ours, C_init, C_final],
        [f"ours initial (DR16, {n_in} spectra, row-median NaN fill = MATLAB convention)",
         "MATLAB initial_M (PCA init from learn_qso_model.m)",
         "MATLAB final M (trained, log_likelihood converged)"],
    ):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                        interpolation="nearest", aspect="auto")
        ax.set_xlabel("λ′ [Å]")
        ax.set_ylabel("λ [Å]")
        ax.set_title(title, fontsize=10)
        ax.axhline(1215.67, color="0.2", lw=0.4, alpha=0.5)
        ax.axvline(1215.67, color="0.2", lw=0.4, alpha=0.5)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                  location="right", label="correlation", pad=0.02)
    fig.suptitle("corr(M·M^T) — ours vs MATLAB DR16 reference\n"
                 "K = M·M^T (no diag noise added); diag-normalized.",
                 fontsize=11, fontweight="bold")
    out = OUT_DIR / "corr_matrix_dr16_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")

    # Also a delta plot to highlight where ours and MATLAB initial differ
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5.5))
    D = C_ours - C_init
    im2 = axes2[0].imshow(D, cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=extent,
                           interpolation="nearest", aspect="auto")
    axes2[0].set_title("ours − MATLAB initial", fontsize=10)
    axes2[0].set_xlabel("λ′ [Å]"); axes2[0].set_ylabel("λ [Å]")
    D2 = C_final - C_init
    im2b = axes2[1].imshow(D2, cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=extent,
                            interpolation="nearest", aspect="auto")
    axes2[1].set_title("MATLAB final − initial (training shifted K)", fontsize=10)
    axes2[1].set_xlabel("λ′ [Å]"); axes2[1].set_ylabel("λ [Å]")
    fig2.colorbar(im2, ax=axes2.ravel().tolist(), shrink=0.7,
                   location="right", label="Δ correlation", pad=0.02)
    out2 = OUT_DIR / "corr_matrix_dr16_delta.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight"); plt.close(fig2)
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
