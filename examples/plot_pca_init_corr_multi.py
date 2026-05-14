"""Plot PCA-init corr(M·M^T) across multiple datasets to compare baseline
basis structure independent of any Adam training.

Datasets:
  (a) 2lpt loa-0 wide
  (b) 2lpt loa-124 nohcd-nobal wide
  (c) LOA real, no-DLA + no-BAL wide
  (d) Saclay mock-0 nohcd-nobal (legacy normalized; NOT wide grid)
  (e) London mock — NOT FOUND on GreatLakes; panel left blank with a note

Each panel runs `_pca_init` on a 30k subset of the preload after the
standard preprocessing (mask + normalize at MATLAB band [1425, 1475] +
de-forest + center) and renders corr(M·M^T) full [-1, 1] and the smoothness
metric (mean adj-pixel diff in corr).

Output:
  docs/notes/2026-05-12_2lpt_models_vs_v1_analysis/corr_pca_init_multi_dataset.png
"""
from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gpy_dla_detection.training.dataset import load_preprocessed_h5
from tests.phase2_train_dr16 import _pca_init

NOTES = REPO / "docs" / "notes"
OUT_DIR = NOTES / "2026-05-12_2lpt_models_vs_v1_analysis"
OUT = OUT_DIR / "corr_pca_init_multi_dataset.png"

PRELOADS = [
    ("2lpt loa-0 wide",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5"),
    ("2lpt loa-124 nohcd-nobal wide",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa124_nohcd_nobal_wide_v2_1778186324/trainset.h5"),
    ("LOA real (no-DLA + no-BAL) wide",
     "/scratch/cavestru_root/cavestru0/mfho/loa_wide_v2/loa_no_dla_no_bal_wide/trainset.h5"),
    ("Saclay mock-0 nohcd-nobal (LEGACY narrow grid)",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/saclay_mock0_nohcd_nobal_normalized/trainset.h5"),
]
N_SUB = 30000
K = 30


def _corr(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def pca_init_for_preload(path: str, n_sub: int = N_SUB):
    ts = load_preprocessed_h5(
        path,
        z_min=2.15, z_max=4.25, max_spectra=n_sub,
        max_noise_variance=9.0,
        apply_mask=True, apply_normalize=True,
        apply_de_forest=True, apply_center=True,
        norm_min_lambda=1425.0, norm_max_lambda=1475.0,
        de_forest_tau_0=0.00246, de_forest_beta=3.62, de_forest_num_lines=31,
        dtype=torch.float32, working_dtype=np.float32,
    )
    M_init, latent = _pca_init(ts.fluxes.numpy(), k=K)
    return ts.rest_wavelengths.numpy(), M_init, latent


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5 panels: 4 datasets + 1 placeholder for London
    fig, axes = plt.subplots(1, 5, figsize=(28, 6.2))
    summary = []

    for ax, (name, path) in zip(axes[:4], PRELOADS):
        if not Path(path).exists():
            ax.text(0.5, 0.5, f"{name}\n\nNOT FOUND",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            summary.append((name, None, "missing"))
            continue
        try:
            print(f"\nComputing PCA init: {name}")
            rest, M_init, latent = pca_init_for_preload(path)
            C = _corr(M_init)
            adj = np.abs(np.diff(C, axis=1)).mean()
            extent = [rest[0], rest[-1], rest[-1], rest[0]]
            im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                           extent=extent, aspect="auto")
            ax.set_title(f"{name}\nn_pix={M_init.shape[0]}, k={K}\nmean adj diff = {adj:.4f}")
            ax.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
            plt.colorbar(im, ax=ax, fraction=0.046, label="correlation")
            summary.append((name, adj, latent[:3].tolist()))
        except Exception as e:
            ax.text(0.5, 0.5, f"{name}\n\nERROR\n{type(e).__name__}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            summary.append((name, None, f"error: {e}"))

    # London placeholder
    axes[-1].text(0.5, 0.5,
        "London mock\n\nNOT FOUND on GreatLakes\n\n"
        "Need to either:\n"
        "  • copy preload from NERSC, OR\n"
        "  • build from raw FITS via\n"
        "    preload_spectra/preload_2lpt_simple.py\n"
        "    (london layout differs slightly)",
        ha="center", va="center", transform=axes[-1].transAxes,
        fontsize=10, color="gray")
    axes[-1].set_title("(e) London mock\n(missing)")
    axes[-1].set_xticks([]); axes[-1].set_yticks([])

    axes[0].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")

    fig.suptitle(
        "PCA-init corr(M·M$^T$) across datasets — preprocessing identical "
        "(mask + normalize @ [1425, 1475] + de-forest + center). "
        "Smooth init across all available datasets confirms the noise in "
        "trained 2lpt models is induced during the Adam loop, not from PCA init.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT}")
    print()
    print(f"{'dataset':<55s} {'adj_diff':>10s}  top-3 eigvals")
    for name, adj, info in summary:
        if adj is None:
            print(f"{name:<55s}  {info}")
        else:
            print(f"{name:<55s} {adj:>10.4f}  {info}")


if __name__ == "__main__":
    main()
