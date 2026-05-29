"""Step A.5: pipeline validation against MATLAB DR16 reference.

Loads:
  /home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue/
    preloaded_qsos.mat                 (raw spectra of 750414 QSOs)
    learned_qso_model_..._851-1421.mat (gold-standard trained model)

Procedure (single Python file, two phases):

  Phase 1 (cheap, ~30 min):
    - Load 89408 train_ind QSOs from preloaded_qsos.mat
    - Apply v1 preprocessing chain (mask → normalize [1425, 1475]
      → interpolate [850.75, 1420.75] @ dλ=0.25 → de-forest 31 Lyman
      lines → center)
    - PCA at k=20 → init_M_python
    - log_omega = log(nanstd(centered, axis=0))
    - Plot init_M_python vs MATLAB's initial_M side-by-side, top 5
      eigenvectors, with sign correction so visual comparison is fair.

  Phase 2 (expensive, ~few hours):
    - Run our full-batch Adam trainer on the 89k spectra at k=20
    - Apply BOSS DR12Q priors on (τ_0, β) (matches MATLAB objective.m)
    - Compare trained M, μ, log_ω, log_c_0/τ_0/β to MATLAB's converged
      values.

Subcommand mode:
    python tests/validate_against_matlab_dr16.py phase1   # PCA init compare
    python tests/validate_against_matlab_dr16.py phase2   # full training
"""
from __future__ import annotations

# Thread cap (BLAS oversubscription on 89k×2281 work would melt this machine)
import os as _os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_name, "4")

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
from gpy_dla_detection.effective_optical_depth import effective_optical_depth
from gpy_dla_detection import voigt as _v

REF_DIR = Path("/home/mfho/MATLAB/gp_dla_detection_dr16q_public/data/dr16/MATLAB_Catalogue")
PRELOAD = REF_DIR / "preloaded_qsos.mat"
CATALOG = REF_DIR / "catalog.mat"
LEARNED = REF_DIR / "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat"
OUT_DIR = REPO / "docs/notes/2026-05-08_matlab_dr16_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match MATLAB's set_parameters
MIN_LAMBDA = 850.75
MAX_LAMBDA = 1420.75
DLAMBDA = 0.25
N_PIX = int(round((MAX_LAMBDA - MIN_LAMBDA) / DLAMBDA)) + 1   # 2281
NORM_LO = 1425.0
NORM_HI = 1475.0
NUM_FOREST_LINES = 31
DEFOREST_TAU_0 = 0.00554   # Kamble+2019 BOSS DR12Q prior (MATLAB default)
DEFOREST_BETA = 3.182
MAX_NOISE_VARIANCE = 9.0   # 3² per MATLAB set_parameters
K_PCA = 20


def _load_zqso_and_train_ind():
    """Read z_qsos for the 750414 QSOs from learned model's catalog field."""
    # The learned model has train_ind which is binary; we also need z_qsos to
    # compute lya_1pz. z_qsos lives in a separate catalog.mat — skip for
    # phase 1 (we don't need it for centered fluxes; only for de-forest in
    # phase 2 if we actually do training).
    with h5py.File(LEARNED, "r") as f:
        train_ind = np.asarray(f["train_ind"])[0].astype(bool)
        rest_ref = np.asarray(f["rest_wavelengths"])[:, 0]
        initial_M_ref = np.asarray(f["initial_M"])  # (k, n_pix)
        M_ref = np.asarray(f["M"])
        mu_ref = np.asarray(f["mu"])[:, 0]
        log_omega_ref = np.asarray(f["log_omega"])[:, 0]
        initial_log_omega_ref = np.asarray(f["initial_log_omega"])[:, 0]
        log_c_0_ref = float(np.asarray(f["log_c_0"])[0, 0])
        log_tau_0_ref = float(np.asarray(f["log_tau_0"])[0, 0])
        log_beta_ref = float(np.asarray(f["log_beta"])[0, 0])
    return dict(
        train_ind=train_ind, rest=rest_ref,
        initial_M=initial_M_ref.T,  # → (n_pix, k)
        M_final=M_ref.T,
        mu=mu_ref, log_omega=log_omega_ref,
        initial_log_omega=initial_log_omega_ref,
        log_c_0=log_c_0_ref, log_tau_0=log_tau_0_ref, log_beta=log_beta_ref,
    )


def _load_preloaded_subset(train_ind, max_qsos=None):
    """Read flux/noise/wavelength/mask for `sum(train_ind)` QSOs.

    The .mat stores per-QSO arrays via HDF5 references. We resolve them
    one-by-one. Each QSO has variable-length arrays. all_wavelengths
    are in OBSERVER frame; we de-redshift via z_qsos from catalog.mat.
    """
    n_total = train_ind.shape[0]
    train_idx = np.where(train_ind)[0]
    if max_qsos is not None:
        train_idx = train_idx[:max_qsos]
    n_load = len(train_idx)

    # z_qsos for de-redshifting
    print("[load] loading z_qsos from catalog.mat ...")
    with h5py.File(CATALOG, "r") as f:
        z_qsos_all = np.asarray(f["z_qsos"])[0]
    print(f"  z_qsos range: [{z_qsos_all[train_idx].min():.2f}, {z_qsos_all[train_idx].max():.2f}], "
          f"median {np.median(z_qsos_all[train_idx]):.2f}")

    print(f"[load] loading {n_load} train_ind QSOs from preloaded_qsos.mat")
    rest_grid = np.linspace(MIN_LAMBDA, MAX_LAMBDA, N_PIX)

    # Centered fluxes interpolated to common rest grid
    interp_flux = np.full((n_load, N_PIX), np.nan, dtype=np.float64)
    interp_nv = np.full((n_load, N_PIX), np.nan, dtype=np.float64)
    z_qsos = z_qsos_all[train_idx].astype(np.float64)
    normalizers = np.zeros(n_load, dtype=np.float64)

    with h5py.File(PRELOAD, "r") as f:
        all_flux = f["all_flux"]
        all_nv = f["all_noise_variance"]
        all_wave = f["all_wavelengths"]
        all_mask = f["all_pixel_mask"]
        all_norm = np.asarray(f["all_normalizers"])[0]
        # z_qsos is NOT in preloaded_qsos.mat (only in catalog.mat).
        # MATLAB infers z from the obs/rest wavelength relation per QSO;
        # for our purpose (PCA init), we don't need z (centered fluxes
        # are computed in rest frame already in the preloaded data via
        # all_wavelengths being rest already? Let me check by inspecting
        # one entry.)

        t0 = time.time()
        for i, qi in enumerate(train_idx):
            if i % 10000 == 0 and i > 0:
                dt = time.time() - t0
                print(f"  [load] {i}/{n_load}  ({dt:.0f}s elapsed; "
                      f"~{dt/i*n_load:.0f}s total)")
            wave = np.asarray(f[all_wave[0, qi]])[0]   # observed wavelengths
            flux = np.asarray(f[all_flux[0, qi]])[0]
            nv = np.asarray(f[all_nv[0, qi]])[0]
            mask = np.asarray(f[all_mask[0, qi]])[0].astype(bool)

            if i == 0:
                print(f"  [load] sample QSO {qi}: wave[0]={wave[0]:.2f}, "
                      f"wave[-1]={wave[-1]:.2f}, n_pix={len(wave)}, "
                      f"normalizer={all_norm[qi]:.4f}")

            normalizers[i] = all_norm[qi]

            # Apply mask + normalize-by-stored-normalizer (MATLAB does this
            # at preload time; the all_normalizers field is each QSO's
            # median in [1425, 1475] computed during the preload step).
            valid = ~mask & np.isfinite(flux) & np.isfinite(nv) & (nv > 0)
            if valid.sum() < 100:
                continue
            # De-redshift OBSERVER → REST: wave_rest = wave_obs / (1 + z_qso)
            z_qso = float(z_qsos[i])
            wave_rest = wave[valid] / (1.0 + z_qso)
            wave_v = wave_rest
            # 2026-05-08: preloaded flux is ALREADY normalized by the per-QSO
            # median in [1425, 1475] (verified by inspecting QSO 9: median ~1.0).
            # Do NOT divide again.
            flux_v = flux[valid]
            nv_v = nv[valid]

            # Find which observed wavelengths fall onto the rest grid for
            # this QSO. MATLAB's preloaded all_wavelengths is observed-frame
            # (we'll check by inspecting first QSO above). To map onto a
            # rest grid we need z_qso, which we DON'T have here.
            # Instead, infer rest_wavelength from the QSO's individual
            # wavelength range. But this requires knowing z_qso.
            #
            # SHORTCUT: assume `all_wavelengths` is already the REST-frame
            # wavelength (the preload may have already de-redshifted). Test
            # this against the loading bounds: loading_min/max_lambda is
            # 800/1550 in rest frame.
            if i == 0:
                print(f"  [load] wave[0]/wave[-1]: "
                      f"{wave_v.min():.1f}/{wave_v.max():.1f} "
                      f"(rest if these fit in [800, 1550])")

            # Interpolate onto common rest grid (linear; we keep NaN
            # outside the spectrum's coverage)
            interp_flux[i] = np.interp(rest_grid, wave_v, flux_v,
                                        left=np.nan, right=np.nan)
            interp_nv[i] = np.interp(rest_grid, wave_v, nv_v,
                                       left=np.nan, right=np.nan)
        dt = time.time() - t0
        print(f"  [load] done in {dt:.0f}s")
    return rest_grid, interp_flux, interp_nv, normalizers


def _de_forest(rest, fluxes, nvs, z_qsos):
    """v1's effective_optical_depth applied per spectrum, num_forest_lines=31."""
    flux_out = np.empty_like(fluxes)
    nv_out = np.empty_like(nvs)
    for i, z in enumerate(z_qsos):
        obs_wave = rest * (1.0 + z)
        tau = effective_optical_depth(
            obs_wave, DEFOREST_BETA, DEFOREST_TAU_0, float(z),
            num_forest_lines=NUM_FOREST_LINES,
        ).sum(axis=1)
        c = np.exp(tau)
        flux_out[i] = fluxes[i] * c
        nv_out[i] = nvs[i] * c ** 2
    return flux_out, nv_out


def _center(fluxes, nvs):
    valid = np.isfinite(fluxes) & np.isfinite(nvs) & (nvs > 0)
    weights = np.where(valid, 1.0 / nvs, 0.0)
    weighted_sum = np.where(valid, fluxes * weights, 0.0).sum(axis=0)
    weight_sum = weights.sum(axis=0)
    mu = np.where(weight_sum > 0, weighted_sum / weight_sum, 0.0)
    centered = np.where(valid, fluxes - mu[None, :], np.nan)
    return centered, mu


def phase1():
    print("=== Phase 1: side-by-side initial_M vs MATLAB DR16 reference ===")
    ref = _load_zqso_and_train_ind()
    print(f"  reference rest grid: [{ref['rest'].min():.2f}, {ref['rest'].max():.2f}]  n_pix={len(ref['rest'])}")
    print(f"  reference initial_M shape: {ref['initial_M'].shape}")
    print(f"  reference τ_0={float(np.exp(ref['log_tau_0'])):.5f}  β={float(np.exp(ref['log_beta'])):.4f}")

    # 2026-05-08 (per user): use full train_ind = 89408 QSOs. PCA on smaller
    # subsets gives noisier leading eigenvectors because pixel coverage is
    # uneven across z_qso bins (deep-blue rest pixels are observable only by
    # the highest-z QSOs).
    max_qsos = None  # = all train_ind = 89408 spectra
    rest_grid, interp_flux, interp_nv, normalizers = _load_preloaded_subset(
        ref["train_ind"], max_qsos=max_qsos
    )
    print(f"  rest_grid shape: {rest_grid.shape}, equal to MATLAB? {np.allclose(rest_grid, ref['rest'], atol=1e-6)}")

    # Get z_qsos — we don't have them in preloaded_qsos. Take from catalog.mat.
    # For Phase 1 (PCA only), de-forest barely matters as long as it's
    # consistent. Try: skip de-forest for the FIRST plot (easier), then
    # add it for the second plot.
    print("\n--- Phase 1a: full v1 chain (de-forest 31 lines + center) ---")
    # Get z_qsos for de-forest
    with h5py.File(CATALOG, "r") as f:
        z_qsos_all = np.asarray(f["z_qsos"])[0]
    train_idx = np.where(ref["train_ind"])[0][:max_qsos]
    z_qsos = z_qsos_all[train_idx].astype(np.float64)
    print(f"  applying de_forest with num_forest_lines={NUM_FOREST_LINES}, τ_0={DEFOREST_TAU_0}, β={DEFOREST_BETA}")
    deforest_flux, deforest_nv = _de_forest(rest_grid, interp_flux, interp_nv, z_qsos)
    centered_a, mu_a = _center(deforest_flux, deforest_nv)
    # 2026-05-08: match MATLAB's pca('rows','complete') = listwise deletion.
    # Drop spectra with ANY NaN pixel; PCA on the survivor subset only.
    has_full_coverage = np.all(np.isfinite(centered_a), axis=1)
    print(f"  listwise-complete subset: {has_full_coverage.sum()}/{centered_a.shape[0]} "
          f"({has_full_coverage.mean()*100:.1f}%)")
    centered_for_pca = centered_a[has_full_coverage].copy()
    pca = PCA(n_components=K_PCA)
    pca.fit(centered_for_pca)
    coeffs = pca.components_.T
    latent = pca.explained_variance_
    M_init_a = coeffs * np.sqrt(latent)[None, :]
    log_omega_a = np.log(np.nanstd(centered_a, axis=0) + 1e-12)
    print(f"  PCA top 5 eigvals: {latent[:5]}")
    print(f"  log_omega range: [{log_omega_a.min():.2f}, {log_omega_a.max():.2f}]  "
          f"(MATLAB ref initial: [{ref['initial_log_omega'].min():.2f}, {ref['initial_log_omega'].max():.2f}])")

    # Plot side-by-side, top 5 eigenvectors
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True)
    for k in range(5):
        ax_l, ax_r = axes[k]
        # Sign-flip our PCA so the integral matches MATLAB's
        ours = M_init_a[:, k]
        ref_k = ref["initial_M"][:, k]
        if np.dot(ours, ref_k) < 0:
            ours = -ours
        ax_l.plot(rest_grid, ours, color="C0", lw=0.8, label=f"ours k={k}")
        ax_l.plot(ref["rest"], ref_k, color="C3", lw=0.8, ls="--", label=f"MATLAB k={k}")
        ax_l.set_ylabel(f"M[:, {k}]")
        ax_l.legend(fontsize=7)
        ax_l.grid(alpha=0.3)
        # Right column: difference
        # Interpolate ours to MATLAB's grid for diff (they're identical here)
        diff = ours - ref_k
        ax_r.plot(rest_grid, diff, color="C2", lw=0.8)
        ax_r.set_ylabel(f"Δ k={k}")
        ax_r.axhline(0, color="0.5", lw=0.5)
        ax_r.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("rest λ [Å]")
    axes[-1, 1].set_xlabel("rest λ [Å]")
    fig.suptitle(f"initial_M side-by-side (ours, n={max_qsos}, no de-forest) vs MATLAB DR16 reference\n"
                 f"top 5 eigenvectors, sign-corrected", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "initial_M_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")

    # log_omega comparison
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rest_grid, log_omega_a, color="C0", lw=0.9, label="ours")
    ax.plot(ref["rest"], ref["initial_log_omega"], color="C3", ls="--", lw=0.9, label="MATLAB initial")
    ax.set_xlabel("rest λ [Å]"); ax.set_ylabel("log_ω initial")
    ax.legend(); ax.grid(alpha=0.3)
    ax.axvline(1215.67, color="0.5", lw=0.5, ls="--")
    out2 = OUT_DIR / "initial_log_omega_compare.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out2}")

    # mu comparison
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rest_grid, mu_a, color="C0", lw=0.9, label="ours")
    ax.plot(ref["rest"], ref["mu"], color="C3", ls="--", lw=0.9, label="MATLAB final")
    ax.set_xlabel("rest λ [Å]"); ax.set_ylabel("μ")
    ax.legend(); ax.grid(alpha=0.3)
    ax.axvline(1215.67, color="0.5", lw=0.5, ls="--")
    out3 = OUT_DIR / "mu_compare.png"
    fig.savefig(out3, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out3}")
    print("\n[Phase 1] complete. Inspect figures + then decide if Phase 2 (full training) is warranted.")


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    if sub == "phase1":
        phase1()
    else:
        raise SystemExit(f"unknown subcommand: {sub}")
