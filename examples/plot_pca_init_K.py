"""Plot the K = M·M^T correlation matrix from the PCA initialization,
comparing OLD (per-pixel NaN fill) vs NEW (per-row NaN fill) on the
same trainset.

Lets us verify the PCA bug fix produces physically meaningful initial
correlations (smooth Lyα/Lyβ/metal off-diagonal structure) vs the old
behavior (rank-1 sharp blocks).

Each panel: 2-row grid (μ + |corr K|) × 2 columns (old / new).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _per_column_fill(fluxes):
    """OLD v2 behavior: per-pixel median fill."""
    f = fluxes.copy()
    for j in range(f.shape[1]):
        col = f[:, j]
        finite = np.isfinite(col)
        if finite.any():
            col[~finite] = np.nanmedian(col[finite])
        else:
            col[:] = 0.0
    return f


def _per_row_fill(fluxes):
    """NEW (matches v1 MATLAB / legacy Python): per-spectrum median fill."""
    f = fluxes.copy()
    for i in range(f.shape[0]):
        row = f[i, :]
        finite = np.isfinite(row)
        if finite.any():
            row[~finite] = np.nanmedian(row[finite])
        else:
            row[:] = 0.0
    return f


def _subset_then_per_row_fill(fluxes, max_nan_frac=0.3):
    """RECOMMENDED FIX: drop heavily-NaN pixels first, then per-row fill on
    the high-coverage subset. Returns (filled_subset, keep_mask) so caller
    can pad M back to the full grid."""
    nan_frac = np.isnan(fluxes).mean(axis=0)
    keep = nan_frac < max_nan_frac
    sub = fluxes[:, keep].copy()
    for i in range(sub.shape[0]):
        row = sub[i, :]
        finite = np.isfinite(row)
        if finite.any():
            row[~finite] = np.nanmedian(row[finite])
        else:
            row[:] = 0.0
    return sub, keep


def _pca_M(filled_centered_fluxes, k):
    pca = PCA(n_components=k)
    pca.fit(filled_centered_fluxes)
    coefficients = pca.components_.T
    eigvals = pca.explained_variance_
    return (coefficients * np.sqrt(eigvals)).astype(np.float32), eigvals


def _correlation(M, downsample=4):
    M_ds = M[::downsample]
    K = M_ds @ M_ds.T
    sigma = np.sqrt(np.maximum(np.diag(K), 1e-30))
    Corr = K / np.outer(sigma, sigma)
    return Corr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainset", required=True, help="Path to a trainset.h5")
    p.add_argument("--n-spectra", type=int, default=2000,
                   help="Random subsample for speed")
    p.add_argument("--k", type=int, default=30,
                   help="Number of PCA components (matches production num_pca_components)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[main] loading trainset: {args.trainset}")
    with h5py.File(args.trainset, "r") as f:
        n_total = f["fluxes"].shape[0]
        n_pix = f["fluxes"].shape[1]
        rng = np.random.default_rng(args.seed)
        n = min(args.n_spectra, n_total)
        idx = np.sort(rng.choice(n_total, size=n, replace=False))
        fluxes = f["fluxes"][idx]
        rest_w = f["rest_wavelengths"][idx[0]]   # all spectra share grid
    print(f"  shape={fluxes.shape}  rest=[{rest_w[0]:.1f}, {rest_w[-1]:.1f}]")

    # Center on inverse-variance-weighted μ approximation: just per-pixel mean
    # for visualization purposes (matches what training data already has if
    # the trainer's preload step centered them; but the trainset.h5 is NOT
    # centered, so we do it here for the PCA init step). v2 train_gp.py
    # currently passes centered_fluxes_np = ts.fluxes.numpy() — which is the
    # post-centering per-spectrum. For this diagnostic we just demean per
    # pixel to mirror the test conditions.
    mu = np.nanmean(fluxes, axis=0)
    centered = fluxes - mu[None, :]
    print(f"  center: μ[1100-1180Å] median = {np.nanmedian(mu[(rest_w>1100)&(rest_w<1180)]):.4f}")
    print(f"  fluxes nan fraction: {np.isnan(centered).mean():.4f}")

    # Build M three ways
    print("[main] PCA init OLD (per-column fill)...")
    M_old, eigs_old = _pca_M(_per_column_fill(centered), args.k)
    print(f"  top-5 eigs: {eigs_old[:5]}")
    print(f"  trace_MMT = {(M_old**2).sum():.3e}, eff_rank = {(M_old**2).sum() / eigs_old[0]:.2f}")

    print("[main] PCA init MIDDLE (per-row fill, no subset — matches v1 on its native grid)...")
    M_row, eigs_row = _pca_M(_per_row_fill(centered), args.k)
    print(f"  top-5 eigs: {eigs_row[:5]}")
    print(f"  trace_MMT = {(M_row**2).sum():.3e}, eff_rank = {(M_row**2).sum() / eigs_row[0]:.2f}")

    print("[main] PCA init FIX (subset NaN<30% + per-row fill + zero-pad)...")
    sub_centered, keep_mask = _subset_then_per_row_fill(centered, max_nan_frac=0.3)
    M_sub_only, eigs_fix = _pca_M(sub_centered, args.k)
    M_fix = np.zeros_like(M_old)
    M_fix[keep_mask, :] = M_sub_only
    print(f"  kept pixels: {int(keep_mask.sum())}/{M_fix.shape[0]}")
    print(f"  top-5 eigs: {eigs_fix[:5]}")
    print(f"  trace_MMT = {(M_fix**2).sum():.3e}, eff_rank (subset) = {(M_sub_only**2).sum() / eigs_fix[0]:.2f}")

    Corr_old = _correlation(M_old)
    Corr_row = _correlation(M_row)
    Corr_fix = _correlation(M_fix)
    sub_idx = np.arange(0, M_old.shape[0], 4)
    rest_sub = rest_w[sub_idx]

    LINES = {"Lyα": 1215.67, "Lyβ": 1025.72, "OVI": 1031.93,
             "Lyγ": 972.54, "NV": 1240.0, "CIII*": 1175.7}

    fig, axes = plt.subplots(3, 3, figsize=(18, 13),
                             gridspec_kw=dict(height_ratios=[1, 0.5, 2.5], hspace=0.25, wspace=0.18))
    fig.suptitle(f"PCA init K = M·M^T   trainset = {Path(args.trainset).parent.name}",
                 fontsize=11)

    for col, (label, M, eigs, Corr) in enumerate([
        ("OLD: per-pixel fill", M_old, eigs_old, Corr_old),
        ("v1 BEHAVIOR: per-row fill (no subset)", M_row, eigs_row, Corr_row),
        ("FIX: subset NaN<30% + per-row + zero-pad", M_fix, eigs_fix, Corr_fix),
    ]):
        # Top: top-5 eigenvectors overlaid
        ax_eig = axes[0, col]
        for i in range(min(5, M.shape[1])):
            ax_eig.plot(rest_w, M[:, i], lw=0.5, alpha=0.7,
                        color=plt.get_cmap("tab10")(i),
                        label=f"PC{i+1} (λ={eigs[i]:.2e})")
        for lname, lwave in LINES.items():
            if rest_w[0] < lwave < rest_w[-1]:
                ax_eig.axvline(lwave, color="0.7", ls=":", lw=0.5, alpha=0.6)
        ax_eig.set_xlabel("rest λ [Å]", fontsize=8)
        ax_eig.set_ylabel("M[:, i] eigenvector × √eig", fontsize=8)
        ax_eig.set_title(label, fontsize=9)
        ax_eig.legend(fontsize=6, loc="upper right")
        ax_eig.grid(alpha=0.3)

        # Middle: eigenvalue spectrum (log scale)
        ax_es = axes[1, col]
        ax_es.semilogy(np.arange(1, len(eigs) + 1), eigs, "o-", color="C0", markersize=4)
        ax_es.set_xlabel("PCA component index", fontsize=8)
        ax_es.set_ylabel("eigenvalue (log)", fontsize=8)
        ax_es.set_title(f"Eigenvalue spectrum  (top/2nd ratio = {eigs[0]/max(eigs[1],1e-30):.1f}×)", fontsize=9)
        ax_es.grid(alpha=0.3, which="both")

        # Bottom: |corr K| matrix
        ax_corr = axes[2, col]
        im = ax_corr.imshow(
            np.abs(Corr),
            origin="lower",
            extent=(rest_sub[0], rest_sub[-1], rest_sub[0], rest_sub[-1]),
            aspect="auto",
            cmap="viridis",
            norm=mpl.colors.LogNorm(vmin=1e-3, vmax=1.0),
        )
        for lname, lwave in LINES.items():
            if rest_w[0] < lwave < rest_w[-1]:
                ax_corr.axvline(lwave, color="white", ls=":", lw=0.4, alpha=0.4)
                ax_corr.axhline(lwave, color="white", ls=":", lw=0.4, alpha=0.4)
        ax_corr.set_xlabel("rest λ [Å]")
        ax_corr.set_ylabel("rest λ [Å]")
        ax_corr.set_title("|corr K| (init from PCA only, no training)", fontsize=9)
        if col == 2:
            cbar = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            cbar.set_label("|corr|", fontsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[main] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
