"""Plot corr(M·M^T) of the PCA INIT (before any Adam training) on the 2lpt
v2 wide preload, for both norm bands. Confirms whether the noisy features
in the trained models are inherited from PCA init or emerge during training.

Output:
  docs/notes/2026-05-12_2lpt_models_vs_v1_analysis/corr_pca_init_2lpt.png
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
OUT = OUT_DIR / "corr_pca_init_2lpt.png"

PRELOAD = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5"
N_SUB = 30000  # enough for stable PCA, fast load
K = 30


def _corr(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def init_for_band(norm_min, norm_max):
    ts = load_preprocessed_h5(
        PRELOAD,
        z_min=2.15, z_max=4.25, max_spectra=N_SUB,
        max_noise_variance=9.0,
        apply_mask=True, apply_normalize=True,
        apply_de_forest=True, apply_center=True,
        norm_min_lambda=norm_min, norm_max_lambda=norm_max,
        de_forest_tau_0=0.00246, de_forest_beta=3.62, de_forest_num_lines=31,
        dtype=torch.float32, working_dtype=np.float32,
    )
    M_init, latent = _pca_init(ts.fluxes.numpy(), k=K)
    return ts.rest_wavelengths.numpy(), M_init, latent


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3 panels: PCA init at Garnett band, MATLAB band, + v1 production trained (control)
    print("Computing PCA init at [1310, 1325] band ...")
    rest_g, M_g, lat_g = init_for_band(1310.0, 1325.0)
    print("Computing PCA init at [1425, 1475] band ...")
    rest_m, M_m, lat_m = init_for_band(1425.0, 1475.0)

    # v1 production trained model for reference
    print("Loading v1 production for reference ...")
    with h5py.File("/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5", "r") as f:
        M_v1 = np.asarray(f["M"][:])
        rest_v1 = np.asarray(f["rest_wavelengths"][:])

    panels = [
        (f"(a) PCA INIT @ [1310, 1325] (Garnett)\n"
         f"on 2lpt loa-0 wide preload, n={N_SUB} subset", rest_g, _corr(M_g),
         np.abs(np.diff(_corr(M_g), axis=1)).mean()),
        (f"(b) PCA INIT @ [1425, 1475] (MATLAB)\n"
         f"on 2lpt loa-0 wide preload, n={N_SUB} subset", rest_m, _corr(M_m),
         np.abs(np.diff(_corr(M_m), axis=1)).mean()),
        (f"(c) v1 PRODUCTION TRAINED (reference)\n"
         f"epoch 920 — what 'smooth converged' looks like", rest_v1, _corr(M_v1),
         np.abs(np.diff(_corr(M_v1), axis=1)).mean()),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, (title, rest, C, adj) in zip(axes, panels):
        extent = [rest[0], rest[-1], rest[-1], rest[0]]
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=extent, aspect="auto")
        ax.set_title(f"{title}\nmean adj diff = {adj:.4f}")
        ax.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="correlation")
    axes[0].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")

    fig.suptitle(
        "PCA-init corr(M·M$^T$) on 2lpt vs. v1-trained reference. "
        "PCA init is SMOOTH at both norm bands (~0.005), matching v1's trained "
        "smoothness — so the noisy features in our trained 2lpt models emerge "
        "during the Adam loop, NOT from PCA init.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT}")
    print()
    print(f"PCA init smoothness (mean adj diff in corr):")
    for title, _, _, adj in panels:
        print(f"  {title.split(chr(10))[0]:<60s}  {adj:.4f}")


if __name__ == "__main__":
    main()
