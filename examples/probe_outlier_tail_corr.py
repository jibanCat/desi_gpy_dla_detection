"""Probe whether outlier per-spectrum normalization medians cause the
PCA-init corr(M·M^T) noise observed in PR #6 trained 2lpt models.

Compares 5 conditions on the 2lpt loa-0 wide preload:

  (a) CLEAN          — 5000 top-SNR spectra with medians strictly in the bulk
                       [0.5, 2.0]; baseline
  (b) +10 SMALL_POS  — clean + 10 spectra with med ∈ [1.5e-3, 1e-2]
                       (passes the current rejection at |med|<1e-3, but tiny)
  (c) +10 LARGE_POS  — clean + 10 spectra with med ∈ [10, 30]
                       (upper-tail bulk, 1-2% of preload)
  (d) +10 EXTREME    — clean + 10 spectra with med ∈ [50, ∞]
                       (extreme upper tail, ~0.1% of preload)
  (e) +10 NEG (ctrl) — clean + 10 spectra with med ≤ 0
                       (rejected by current logic → should match CLEAN)

For each: applies the standard dataset.py preprocessing (mask + normalize
with the CURRENT rejection rule + de-forest + IV-weighted centering),
runs the PCA init, and measures corr(M·M^T) smoothness (mean adjacent-pixel
|corr| difference).

If LARGE_POS / EXTREME bumps smoothness up by ≫ CLEAN while NEG matches
CLEAN, the upper-tail-median hypothesis is the leading cause of the
~7× corr-noise gap vs v1 production.

Output:
  docs/notes/2026-05-12_2lpt_corr_noise_debug/corr_outlier_tail_test.png
  docs/notes/2026-05-12_2lpt_corr_noise_debug/outlier_tail_smoothness.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gpy_dla_detection.training.dataset import (
    _center_fluxes_inverse_variance,
    _de_forest_batch,
    _mask_high_noise_pixels,
    _normalize_by_rest_median,
)
from tests.phase2_train_dr16 import _pca_init

NOTES = REPO / "docs" / "notes"
OUT_DIR = NOTES / "2026-05-12_2lpt_corr_noise_debug"
OUT_PNG = OUT_DIR / "corr_outlier_tail_test.png"
OUT_JSON = OUT_DIR / "outlier_tail_smoothness.json"

PRELOAD = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5"

NORM_BAND = (1310.0, 1325.0)
Z_MIN, Z_MAX = 2.15, 4.25
MAX_NOISE_VARIANCE = 9.0
N_CLEAN = 5000
N_OUTLIER = 10
K = 30
DE_FOREST = dict(tau_0=0.00246, beta=3.62, num_forest_lines=31)

# Outlier-pool definitions
POOLS = {
    "small_pos": (1.5e-3, 1e-2),
    "large_pos": (10.0, 30.0),
    "extreme":   (50.0, 1e6),
    "neg_ctrl":  (-1e6, 0.0),
}
CLEAN_MED_RANGE = (0.5, 2.0)


def _corr(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def smoothness(M):
    """Mean adjacent-pixel |Δcorr| — same metric as
    tests/validate_corr_smoothness.py and the rest of the corr-noise series."""
    C = _corr(M)
    return float(np.abs(np.diff(C, axis=1)).mean())


def preprocess_and_pca(fluxes, nv, rest, z_qsos, k=K):
    """Apply current dataset.py preprocessing (normalize → mask → de-forest →
    center), then PCA init. Mirrors MATLAB DR16 order (preload normalizes,
    then learn_qso_model masks normalized nv). Returns (M_init, n_rejected)."""
    fluxes = fluxes.astype(np.float32, copy=True)
    nv = nv.astype(np.float32, copy=True)
    fluxes, nv, meds = _normalize_by_rest_median(
        fluxes, nv, rest,
        norm_min_lambda=NORM_BAND[0],
        norm_max_lambda=NORM_BAND[1],
    )
    # Count rows that _normalize_by_rest_median NaN-propagated (its actual
    # rejection, whatever the threshold is). Avoids re-implementing the rule.
    n_rejected = int(np.isnan(fluxes).all(axis=1).sum())
    fluxes, nv = _mask_high_noise_pixels(fluxes, nv, MAX_NOISE_VARIANCE)
    fluxes, nv = _de_forest_batch(fluxes, nv, rest, z_qsos, **DE_FOREST)
    fluxes, _mean = _center_fluxes_inverse_variance(fluxes, nv)
    M, _lat = _pca_init(fluxes, k=k)
    return M, n_rejected


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(PRELOAD, "r") as f:
        rest = np.asarray(f["rest_wavelengths"][0, :], dtype=np.float32)
        n_total = f["fluxes"].shape[0]
        zqso_all = np.asarray(f["zqso"][:], dtype=np.float32)
        redsnr_all = np.asarray(f["redsnr"][:], dtype=np.float32)
        # Compute per-spectrum medians in the norm band for the whole preload.
        # Stream over chunks to bound peak RAM (300k × 100 ≈ 120 MB anyway,
        # but we'll also need flux slices later for the chosen indices).
        m = (rest >= NORM_BAND[0]) & (rest <= NORM_BAND[1])
        n_band = int(m.sum())
        print(f"Norm-band pixels: {n_band}; full preload n={n_total}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluxes_band = f["fluxes"][:, m]  # (N, n_band), small enough
            meds_all = np.nanmedian(fluxes_band, axis=1).astype(np.float64)
        del fluxes_band

    # Build the index pools
    is_finite_z = np.isfinite(zqso_all) & (zqso_all >= Z_MIN) & (zqso_all <= Z_MAX)
    is_finite_snr = np.isfinite(redsnr_all)

    bulk = is_finite_z & is_finite_snr & np.isfinite(meds_all) & (
        meds_all >= CLEAN_MED_RANGE[0]) & (meds_all <= CLEAN_MED_RANGE[1])
    bulk_idx = np.where(bulk)[0]
    # Pick CLEAN as top-SNR within the bulk
    order = np.argsort(redsnr_all[bulk_idx])[::-1]
    clean_idx = bulk_idx[order[:N_CLEAN]]
    print(f"\nCLEAN pool: {len(clean_idx)} spectra "
          f"(top-SNR, med ∈ [{CLEAN_MED_RANGE[0]}, {CLEAN_MED_RANGE[1]}]); "
          f"SNR range {redsnr_all[clean_idx].min():.1f}–{redsnr_all[clean_idx].max():.1f}")

    pool_choices = {}
    rng = np.random.default_rng(0)
    for name, (lo, hi) in POOLS.items():
        if name == "neg_ctrl":
            cand = np.where(is_finite_z & is_finite_snr &
                            np.isfinite(meds_all) & (meds_all <= hi))[0]
        else:
            cand = np.where(is_finite_z & is_finite_snr &
                            np.isfinite(meds_all) &
                            (meds_all >= lo) & (meds_all < hi))[0]
        if len(cand) < N_OUTLIER:
            print(f"  WARN: pool '{name}' has only {len(cand)} candidates "
                  f"(want {N_OUTLIER})")
            pool_choices[name] = cand
        else:
            pool_choices[name] = rng.choice(cand, size=N_OUTLIER, replace=False)
        chosen_meds = meds_all[pool_choices[name]]
        print(f"  pool {name:<11s} [{lo:>10.4g}, {hi:<10.4g}]: "
              f"n_cand={len(cand):>7d}, picked={len(pool_choices[name]):>2d}, "
              f"med range {chosen_meds.min():.4g}…{chosen_meds.max():.4g}")

    # Read in the spectra we need (clean + all pool spectra), de-duplicated
    needed = np.unique(np.concatenate([clean_idx, *pool_choices.values()]))
    sort_order = np.argsort(needed)
    needed_sorted = needed[sort_order]
    # h5py fancy-indexing requires sorted unique indices; build map back.
    idx_to_pos = {int(t): i for i, t in enumerate(needed_sorted)}
    print(f"\nReading {len(needed_sorted)} spectra from preload …")
    with h5py.File(PRELOAD, "r") as f:
        fluxes_pool = f["fluxes"][needed_sorted, :].astype(np.float32)
        nv_pool = f["noise_variance"][needed_sorted, :].astype(np.float32)
        zqso_pool = zqso_all[needed_sorted].astype(np.float32)
    print(f"  loaded fluxes {fluxes_pool.shape}, nv {nv_pool.shape}, "
          f"zqso {zqso_pool.shape}")

    def gather(indices):
        positions = np.array([idx_to_pos[int(i)] for i in indices])
        return (fluxes_pool[positions].copy(),
                nv_pool[positions].copy(),
                zqso_pool[positions].copy())

    conditions = [("clean", clean_idx)]
    for name in ("small_pos", "large_pos", "extreme", "neg_ctrl"):
        conditions.append((name, np.concatenate([clean_idx, pool_choices[name]])))

    results = {}
    panels = []
    for cond_name, idx in conditions:
        flux, nv, zq = gather(idx)
        M, n_rej = preprocess_and_pca(flux, nv, rest, zq)
        adj = smoothness(M)
        C = _corr(M)
        results[cond_name] = {
            "n_spec": len(idx),
            "n_rejected_by_normalize": n_rej,
            "smoothness_adj_diff": adj,
        }
        panels.append((cond_name, C, adj, n_rej, len(idx)))
        print(f"  cond {cond_name:<10s} n={len(idx):>5d}, "
              f"n_rej={n_rej:>3d}, adj_diff={adj:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.8))
    titles = {
        "clean":     "(a) CLEAN baseline\n5000 top-SNR, med ∈ [0.5, 2.0]",
        "small_pos": f"(b) +10 SMALL_POS\nmed ∈ [{POOLS['small_pos'][0]:.0e}, {POOLS['small_pos'][1]:.0e}]",
        "large_pos": f"(c) +10 LARGE_POS\nmed ∈ [{POOLS['large_pos'][0]:g}, {POOLS['large_pos'][1]:g}]",
        "extreme":   f"(d) +10 EXTREME\nmed ≥ {POOLS['extreme'][0]:g}",
        "neg_ctrl":  "(e) +10 NEG (control)\nmed ≤ 0 — rejected by current logic",
    }
    extent = [rest[0], rest[-1], rest[-1], rest[0]]
    for ax, (cond_name, C, adj, n_rej, n_spec) in zip(axes, panels):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=extent, aspect="auto")
        ax.set_title(f"{titles[cond_name]}\n"
                     f"n_spec={n_spec}, rejected={n_rej}, "
                     f"smooth={adj:.4f}")
        ax.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="correlation")
    axes[0].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
    fig.suptitle(
        "Outlier-tail probe — PCA-init corr(M·M$^T$) on 2lpt loa-0 wide preload "
        "(norm band [1310, 1325]). "
        "CLEAN sets the floor; (e) NEG is the control (current rejection should "
        "make it match CLEAN); (b)–(d) test whether outliers that survive current "
        "rejection inflate corr-noise.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT_PNG}")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "preload": PRELOAD,
            "norm_band": list(NORM_BAND),
            "n_clean": N_CLEAN,
            "n_outlier_per_pool": N_OUTLIER,
            "pools": {k: list(v) for k, v in POOLS.items()},
            "clean_med_range": list(CLEAN_MED_RANGE),
            "results": results,
        }, f, indent=2)
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
