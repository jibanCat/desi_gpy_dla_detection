"""Bigger corr(M·M^T) per-lane plot.

Produces a 2×2 grid with separate larger panels:
  init (PCA from fixture), v1 (Adam, 50 iter), v3.5 (Adam, 50 iter),
  MATLAB (L-BFGS, 50 iter).

corr_ij = (M·M^T)_ij / sqrt((M·M^T)_ii · (M·M^T)_jj) — diag-normalized
covariance shape, with per-pixel amplitude divided out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIX = Path(__file__).resolve().parent / "fixtures" / "2lpt_frozen"
OUT = FIX / "short_retrain"


def _corr(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def _load_M(path, key):
    if str(path).endswith(".mat"):
        m = loadmat(path)
        M = np.asarray(m[key])
    else:
        d = np.load(path)
        M = np.asarray(d[key])
    return M


def main():
    init_npz = np.load(FIX / "init_params.npz")
    rest = init_npz["rest_wavelengths"].astype(float)
    M_init = init_npz["M"].astype(float)

    panels = [("init  (PCA, before training)", M_init)]
    if (OUT / "v1.npz").exists():
        panels.append(("v1    (Adam, 50 iter; approx dlog_β)",
                       _load_M(OUT / "v1.npz", "M_final").astype(float)))
    if (OUT / "v3.5.npz").exists():
        panels.append(("v3.5  (Adam, 50 iter; strict dlog_β)",
                       _load_M(OUT / "v3.5.npz", "M_final").astype(float)))
    if (OUT / "matlab.mat").exists():
        panels.append(("MATLAB (L-BFGS, 50 iter)",
                       _load_M(OUT / "matlab.mat", "M_final").astype(float)))

    n_panels = len(panels)
    n_cols = 2 if n_panels > 2 else n_panels
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 6.5 * n_rows),
                             squeeze=False)
    extent = [rest[0], rest[-1], rest[-1], rest[0]]
    for (title, M), ax in zip(panels, axes.flat):
        if M.shape[0] != rest.shape[0]:
            M = M.T
        C = _corr(M)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                       interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("λ′ [Å]", fontsize=9)
        ax.set_ylabel("λ [Å]", fontsize=9)
        ax.axhline(1215.67, color="0.3", lw=0.4, alpha=0.5)
        ax.axvline(1215.67, color="0.3", lw=0.4, alpha=0.5)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55,
                 location="right", label="correlation", pad=0.02)
    fig.suptitle(
        "corr(M·M^T) — diag-normalized GP basis covariance shape\n"
        "Step A.3 short retrains on the 1300-spectrum 2lpt fixture",
        fontsize=12, fontweight="bold",
    )
    out = OUT / "corr_grid_large.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")

    # Also make a delta plot (training - init) per lane to highlight what
    # training learned beyond the PCA prior.
    delta_panels = []
    for title, M in panels[1:]:  # skip init
        delta_panels.append((title.split()[0] + " − init", M))
    if not delta_panels:
        return
    fig2, axes2 = plt.subplots(1, len(delta_panels), figsize=(5.5 * len(delta_panels), 5.5),
                                squeeze=False)
    C0 = _corr(M_init if M_init.shape[0] == rest.shape[0] else M_init.T)
    for ax, (title, M) in zip(axes2[0], delta_panels):
        if M.shape[0] != rest.shape[0]:
            M = M.T
        C = _corr(M)
        D = C - C0
        im = ax.imshow(D, cmap="RdBu_r", vmin=-0.3, vmax=0.3, extent=extent,
                       interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("λ′ [Å]")
        ax.set_ylabel("λ [Å]")
    fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.7,
                  label="Δ correlation (training − init)", pad=0.02)
    fig2.suptitle(
        "How training reshaped corr(M·M^T) (training − init), per lane",
        fontsize=12, fontweight="bold",
    )
    out2 = OUT / "corr_delta_grid.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
