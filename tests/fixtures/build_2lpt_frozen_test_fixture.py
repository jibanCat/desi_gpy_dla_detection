"""Build a frozen test fixture from 2lpt mock-0 loa-0 trainset for the
v1/MATLAB cross-check tests in PR #6 (Step A.1 / A.2).

The fixture writes:
  - tests/fixtures/2lpt_frozen/init_params.{npz, mat}
      Population-level inputs (same for all spectra; init values from PCA):
        rest_wavelengths   : (n_pix,)     rest-frame grid
        mu                 : (n_pix,)     inverse-variance-weighted mean
        M                  : (n_pix, k)   PCA basis × sqrt(eigenvalue)
        log_omega          : (n_pix,)     log(per-pixel std after centering)
        c_0, tau_0, beta   : scalars      v1 init values (0.1, 0.00246, 3.62)
        num_forest_lines   : 3
        all_transition_wavelengths : (31,) Lyman series
        all_oscillator_strengths   : (31,) Lyman series
        n_train_spectra    : scalar       number used in the PCA computation

  - tests/fixtures/2lpt_frozen/<TID>.{npz, mat}
      Per-spectrum inputs to spectrum_loss:
        target_id      : scalar  int
        z_qso          : scalar  float
        snr_forest     : scalar  float
        flux           : (n_pix,)  CENTERED, DE-FORESTED flux (= y in spectrum_loss)
        noise_variance : (n_pix,)  DE-FORESTED noise variance
        lya_1pz        : (n_pix,)  Lyα 1+z grid per pixel
        valid_mask     : (n_pix,)  bool — pixels with finite flux + nv > 0
        zqso_1pz       : scalar    1 + z_qso (matches MATLAB convention)

Both `.npz` (numpy) and `.mat` (MATLAB) carry IDENTICAL field names and
values. `spectrum_loss` in v1 (`gpy_dla_detection/objective.py:97`) and
MATLAB (`learn_qso_model_dr16q_public/spectrum_loss.m`) have field-identical
input signatures, so loading either format and calling spectrum_loss must
produce equal loss + gradients to ~1e-8.

Source: 2lpt mock-0 loa-0 trainset (300k spectra, includes canonical TID
120046865). Picks happen INSIDE the trainset's TID list so all picks are
guaranteed to be loadable.

Re-run by: python tests/fixtures/build_2lpt_frozen_test_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from scipy.io import savemat
from sklearn.decomposition import PCA


# v2 trainset.h5 — raw fluxes after (mask + interpolate) only.
# We apply the v1 (deforest + center) on top to get the input form
# spectrum_loss expects.
TRAINSET = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_48938765/trainset.h5"
ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
BAL_CAT = f"{ROOT}/bal_cat.fits"
ZCAT = f"{ROOT}/zcat.fits"
SNR_CAT = f"{ROOT}/snr_cat.fits"

OUT_DIR = Path(__file__).resolve().parent / "2lpt_frozen"

# v1 init values (from learn_qso_model.py:440-442)
INITIAL_C_0 = 0.1
INITIAL_TAU_0 = 0.00246
INITIAL_BETA = 3.62
NUM_PCA_K = 30
NUM_FOREST_LINES = 3
DEFOREST_TAU_0 = 0.00246  # same as runtime; v1 SpectrumProcessor default
DEFOREST_BETA = 3.62

# Five frozen TID targets spanning z + SNR. All must be:
#   - in 2lpt_loa0 trainset.h5
#   - non-BAL (BI_CIV == 0 or absent in bal_cat)
#   - ZWARN == 0
PICKS = [
    # (z target, SNR_F target, tier label)
    (2.60, 8.0, "high"),
    (2.90, 3.0, "med"),
    (3.20, 1.0, "low"),
    (3.50, 4.0, "med-hi"),
    (3.80, 2.0, "med"),
]
CANONICAL_TID = 120046865  # NHI=21.26 strong DLA, z=2.96


def _load_lyman_constants():
    """Use the exact same constants the v1 trainer uses (gpy_dla_detection.voigt)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from gpy_dla_detection.voigt import (
        transition_wavelengths as TW,
        oscillator_strengths as OS,
    )
    # v1 uses transition wavelengths in cm; convert to Å for clarity (and matches MATLAB's storage).
    return np.asarray(TW, dtype=np.float64), np.asarray(OS, dtype=np.float64)


def _de_forest(rest_wavelengths, fluxes, noise_variances, z_qsos,
               tau_0=DEFOREST_TAU_0, beta=DEFOREST_BETA):
    """Apply v1 SpectrumProcessor.de_forest_spectra logic (line 349 of
    learn_qso_model.py). Multiplicative correction; pixels above Lyα at
    QSO frame are unchanged. fluxes shape (N, n_pix), z_qsos shape (N,).
    """
    LYA_REST_A = 1215.67  # Å (v1 uses this)
    # observer wavelength of each pixel for each spectrum
    obs = rest_wavelengths[None, :] * (1.0 + z_qsos[:, None])  # (N, n_pix)
    lya_obs = LYA_REST_A * (1.0 + z_qsos[:, None])             # (N, 1)
    z_lya = obs / LYA_REST_A - 1.0                              # absorber 1+z grid
    # only pixels below Lyα at QSO frame get de-forested
    in_forest = obs < lya_obs                                   # (N, n_pix)
    tau = tau_0 * np.power(1.0 + z_lya, beta)                   # (N, n_pix)
    correction = np.exp(tau)                                    # divide by exp(-tau)
    flux_out = np.where(in_forest, fluxes * correction, fluxes)
    nv_out = np.where(in_forest, noise_variances * correction**2, noise_variances)
    return flux_out, nv_out


def _center_fluxes_inv_var(fluxes, noise_variances):
    """v1 SpectrumProcessor.center_fluxes (line 366): subtract the
    inverse-variance-weighted mean per pixel; returns (centered, mu).
    """
    valid = np.isfinite(fluxes) & np.isfinite(noise_variances) & (noise_variances > 0)
    weights = np.where(valid, 1.0 / noise_variances, 0.0)
    weighted_sum = np.where(valid, fluxes * weights, 0.0).sum(axis=0)
    weight_sum = weights.sum(axis=0)
    mu = np.where(weight_sum > 0, weighted_sum / weight_sum, 0.0)
    centered = np.where(valid, fluxes - mu[None, :], np.nan)
    return centered, mu


def _stratified_z_indices(z_qsos, n_per_bin=100, z_lo=2.5, z_hi=3.85, dz=0.1, seed=0):
    """Stratified sample for PCA / mu / log_omega init."""
    rng = np.random.default_rng(seed)
    bins = np.arange(z_lo, z_hi + 1e-6, dz)
    idx_pool = []
    for i in range(len(bins) - 1):
        m = (z_qsos >= bins[i]) & (z_qsos < bins[i + 1])
        idx_in = np.where(m)[0]
        if len(idx_in) == 0:
            continue
        take = min(n_per_bin, len(idx_in))
        idx_pool.extend(rng.choice(idx_in, take, replace=False).tolist())
    return np.asarray(sorted(idx_pool), dtype=np.int64)


def _pick_frozen_tids(tids_in_trainset, z_in_trainset, snr_forest_in_trainset,
                      bal_set, picks):
    """Pick one TID per (z, SNR) target. Returns a list of trainset row indices."""
    chosen = []
    used_idx = set()
    for z_t, snr_t, tier in picks:
        # candidates within ±0.025 of z_t, then loosen to ±0.05 if empty
        for tol in (0.025, 0.05, 0.10):
            cand = np.where(
                (np.abs(z_in_trainset - z_t) <= tol)
                & np.array([int(tid) not in bal_set for tid in tids_in_trainset])
            )[0]
            cand = np.array([i for i in cand if i not in used_idx], dtype=np.int64)
            if cand.size:
                break
        if not cand.size:
            print(f"  ! could not find candidate for z={z_t} (skipping)")
            continue
        # pick by closest SNR to target
        snr_arr = snr_forest_in_trainset[cand]
        # treat NaN SNR as low priority
        snr_arr = np.where(np.isfinite(snr_arr), snr_arr, -np.inf)
        j = cand[np.argmin(np.abs(snr_arr - snr_t))]
        used_idx.add(int(j))
        chosen.append((int(j), z_t, snr_t, tier))
    return chosen


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] trainset: {TRAINSET}")
    with h5py.File(TRAINSET, "r") as f:
        tids_all = np.asarray(f["tids"])
        z_all = np.asarray(f["zqso"]).astype(np.float64)
        rw_all = np.asarray(f["rest_wavelengths"])
        flux_all = np.asarray(f["fluxes"]).astype(np.float64)
        nv_all = np.asarray(f["noise_variance"]).astype(np.float64)
        bluesnr_all = np.asarray(f["bluesnr"]).astype(np.float64)
        n_total, n_pix = flux_all.shape
    rest_wavelengths = rw_all[0].astype(np.float64)  # rest grid identical for all rows
    print(f"  n_total={n_total} n_pix={n_pix} rest=[{rest_wavelengths[0]:.1f}, {rest_wavelengths[-1]:.1f}]")

    # BAL TIDs (universe-level lookup)
    print(f"[load] bal_cat: {BAL_CAT}")
    bal = Table.read(BAL_CAT)[["TARGETID", "BI_CIV"]]
    bal_set = set(int(t) for t in bal["TARGETID"][bal["BI_CIV"] > 0])
    print(f"  BAL TIDs: {len(bal_set)}")

    # Stratified sample for PCA / mu / log_omega
    print("[strat] z-stratified sample for PCA fit (~100/0.1-z bin):")
    strat_idx = _stratified_z_indices(z_all, n_per_bin=100, z_lo=2.5, z_hi=3.85, dz=0.1, seed=0)
    print(f"  n={len(strat_idx)}")
    # Apply v1 (deforest + center) on the stratified set
    rw_2d = np.broadcast_to(rest_wavelengths[None, :], (len(strat_idx), n_pix))
    flux_strat = flux_all[strat_idx]
    nv_strat = nv_all[strat_idx]
    z_strat = z_all[strat_idx]
    flux_def, nv_def = _de_forest(rest_wavelengths, flux_strat, nv_strat, z_strat)
    centered, mu = _center_fluxes_inv_var(flux_def, nv_def)

    # PCA on the centered, NaN-filled fluxes (v1 fills with median per column)
    print("[PCA] fitting on centered fluxes ...")
    centered_for_pca = centered.copy()
    col_med = np.nanmedian(centered_for_pca, axis=0)
    bad = ~np.isfinite(centered_for_pca)
    centered_for_pca[bad] = np.broadcast_to(col_med, centered_for_pca.shape)[bad]
    pca = PCA(n_components=NUM_PCA_K)
    pca.fit(centered_for_pca)
    coefficients = pca.components_.T          # (n_pix, k)
    latent = pca.explained_variance_           # (k,)
    M_init = coefficients * np.sqrt(latent)[None, :]   # (n_pix, k)
    log_omega_init = np.log(np.nanstd(centered, axis=0) + 1e-12)
    print(f"  M shape={M_init.shape}, top eigvals[:5]={latent[:5].tolist()}")
    print(f"  log_omega range=[{log_omega_init.min():.2f}, {log_omega_init.max():.2f}]")

    # Lyman-line constants
    TW, OS = _load_lyman_constants()
    print(f"[lyman] num_forest_lines={NUM_FOREST_LINES} TW[0]={TW[0]} OS[0]={OS[0]}")

    # Save the 1300-spectrum stratified TRAINING SET so both Python and MATLAB
    # short-retrain runs see byte-identical inputs. Each row is post (mask →
    # interpolate → deforest → center). Validity: pixels are valid where
    # `centered` is finite (i.e. nv > 0 and the de-forest division didn't blow
    # up). The trainer derives lya_1pz from rest_wavelengths + z_qso.
    centered_for_save = centered.astype(np.float64)
    nv_for_save = nv_def.astype(np.float64)
    valid_masks = np.isfinite(centered_for_save) & np.isfinite(nv_for_save) & (nv_for_save > 0)
    train = dict(
        centered_fluxes=centered_for_save,                     # (N, n_pix)
        noise_variances=nv_for_save,                            # (N, n_pix)  (de-forested)
        z_qsos=z_strat.astype(np.float64),                      # (N,)
        target_ids=tids_all[strat_idx].astype(np.int64),        # (N,)
        valid_masks=valid_masks,                                # (N, n_pix), bool
        rest_wavelengths=rest_wavelengths,                      # (n_pix,)
        n_train_spectra=np.int64(len(strat_idx)),
    )
    train_npz = OUT_DIR / "training_set.npz"
    train_mat = OUT_DIR / "training_set.mat"
    np.savez(train_npz, **train)
    savemat(train_mat, train)
    print(f"[saved] {train_npz}  ({train_npz.stat().st_size/1e6:.2f} MB)")
    print(f"[saved] {train_mat}  ({train_mat.stat().st_size/1e6:.2f} MB)")

    # Save population init
    init_npz = OUT_DIR / "init_params.npz"
    init_mat = OUT_DIR / "init_params.mat"
    init = dict(
        rest_wavelengths=rest_wavelengths,
        mu=mu, M=M_init, log_omega=log_omega_init,
        c_0=np.float64(INITIAL_C_0),
        tau_0=np.float64(INITIAL_TAU_0),
        beta=np.float64(INITIAL_BETA),
        num_forest_lines=np.int64(NUM_FOREST_LINES),
        all_transition_wavelengths=TW, all_oscillator_strengths=OS,
        n_train_spectra=np.int64(len(strat_idx)),
        latent=latent,
    )
    np.savez(init_npz, **init)
    savemat(init_mat, init)
    print(f"[saved] {init_npz}  ({init_npz.stat().st_size/1e6:.2f} MB)")
    print(f"[saved] {init_mat}  ({init_mat.stat().st_size/1e6:.2f} MB)")

    # Pick the 5 frozen TIDs FROM the trainset (z + SNR diversity)
    chosen = _pick_frozen_tids(tids_all, z_all, bluesnr_all, bal_set, PICKS)
    # Add canonical TID
    canon_idx = np.where(tids_all == CANONICAL_TID)[0]
    if canon_idx.size:
        chosen.append((int(canon_idx[0]), 2.96, 0.59, "canonical"))

    # Apply v1 preprocessing to each frozen TID and save
    LYA_REST_A = 1215.67
    for row_idx, z_t, snr_t, tier in chosen:
        tid = int(tids_all[row_idx])
        z_qso = float(z_all[row_idx])
        flux_raw = flux_all[row_idx][None, :]
        nv_raw = nv_all[row_idx][None, :]
        flux_def, nv_def = _de_forest(rest_wavelengths, flux_raw, nv_raw,
                                       np.array([z_qso]))
        # subtract the SAME population mu (so y is in v1's centered space)
        flux_centered = (flux_def - mu[None, :]).flatten()
        nv_def_1d = nv_def.flatten()
        # valid mask: finite flux + finite nv + nv > 0
        valid_mask = (np.isfinite(flux_centered)
                      & np.isfinite(nv_def_1d) & (nv_def_1d > 0))
        # lya_1pz per pixel: 1 + ((1+z_qso) * rest - lya_rest) / lya_rest
        lya_1pz = 1.0 + ((1.0 + z_qso) * rest_wavelengths - LYA_REST_A) / LYA_REST_A
        per_spec = dict(
            target_id=np.int64(tid),
            z_qso=np.float64(z_qso),
            snr_forest=np.float64(bluesnr_all[row_idx]),
            flux=flux_centered,
            noise_variance=nv_def_1d,
            lya_1pz=lya_1pz,
            valid_mask=valid_mask,
            zqso_1pz=np.float64(1.0 + z_qso),
            tier=tier,
            row_idx=np.int64(row_idx),
        )
        npz = OUT_DIR / f"{tid}.npz"
        mat = OUT_DIR / f"{tid}.mat"
        np.savez(npz, **per_spec)
        savemat(mat, per_spec)
        print(f"  TID={tid:>10} z={z_qso:.3f} SNR_F={bluesnr_all[row_idx]:.2f} "
              f"valid={valid_mask.sum():>4}/{n_pix}  tier={tier}  → {npz.name}")

    # Manifest
    manifest = dict(
        source_trainset=TRAINSET,
        bal_cat=BAL_CAT,
        n_strat=int(len(strat_idx)),
        n_pix=int(n_pix),
        k=int(NUM_PCA_K),
        c_0=INITIAL_C_0, tau_0=INITIAL_TAU_0, beta=INITIAL_BETA,
        deforest_tau_0=DEFOREST_TAU_0, deforest_beta=DEFOREST_BETA,
        num_forest_lines=NUM_FOREST_LINES,
        spectra=[dict(
            target_id=int(tids_all[i]),
            row_idx=int(i),
            z_qso=float(z_all[i]),
            snr_forest=float(bluesnr_all[i]),
            tier=tier,
            z_target=z_t, snr_target=snr_t,
        ) for i, z_t, snr_t, tier in chosen],
    )
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"[saved] {OUT_DIR / 'MANIFEST.json'}")
    print("[done]")


if __name__ == "__main__":
    main()
