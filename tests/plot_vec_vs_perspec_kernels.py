"""Kernel-level comparison: vec full vs per-spec full at production DR16 scale.

Loads the trained M from both Phase 2 production retrains and renders a
3-panel overlay of M·M^T (the GP covariance kernel, which is gauge-invariant
under right-rotation of M):

  (a) M·M^T from vec full   (49700040, 89k×200, vectorized=1)
  (b) M·M^T from per-spec   (49709974, 89k×200, vectorized=0)
  (c) |Δ| = |C_vec − C_per| (same colour scale as a/b for ratio context;
      log scale to make the structure visible)

Output:
  docs/notes/2026-05-11_vec_vs_perspec_kernels.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
OUT = NOTES / "2026-05-11_vec_vs_perspec_kernels.png"

VEC_FULL = NOTES / "2026-05-08_matlab_dr16_validation_vec_full" / "phase2_result.npz"
PER_SPEC = NOTES / "2026-05-08_matlab_dr16_validation_per_spec" / "phase2_result.npz"


def main():
    vec = np.load(VEC_FULL)
    per = np.load(PER_SPEC)

    Mv, Mp = vec["M"], per["M"]
    rw = vec["rest_wavelengths"]
    assert np.allclose(rw, per["rest_wavelengths"]), "rest_wavelengths must match"

    Cv = Mv @ Mv.T
    Cp = Mp @ Mp.T
    dC = Cv - Cp

    vmax = max(np.max(np.abs(Cv)), np.max(np.abs(Cp)))
    vmin = -vmax
    extent = [rw[0], rw[-1], rw[-1], rw[0]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    # (a) vec full
    im0 = axes[0].imshow(Cv, vmin=vmin, vmax=vmax, cmap="RdBu_r",
                         extent=extent, aspect="auto")
    axes[0].set_title(f"(a) M·M$^T$ — vec full (49700040)\n"
                      f"||C||$_F$ = {np.linalg.norm(Cv):.3e}")
    axes[0].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    axes[0].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # (b) per-spec
    im1 = axes[1].imshow(Cp, vmin=vmin, vmax=vmax, cmap="RdBu_r",
                         extent=extent, aspect="auto")
    axes[1].set_title(f"(b) M·M$^T$ — per-spec (49709974)\n"
                      f"||C||$_F$ = {np.linalg.norm(Cp):.3e}")
    axes[1].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # (c) |Δ| log-scale; floor at 1e-6 of vmax for visibility
    abs_dc = np.abs(dC)
    floor = max(abs_dc[abs_dc > 0].min(), vmax * 1e-6)
    im2 = axes[2].imshow(np.maximum(abs_dc, floor), cmap="viridis",
                         norm=LogNorm(vmin=floor, vmax=vmax),
                         extent=extent, aspect="auto")
    rel = np.linalg.norm(dC) / np.linalg.norm(Cv)
    axes[2].set_title(f"(c) |ΔC| = |C$_\\mathrm{{vec}}$ − C$_\\mathrm{{per}}$|\n"
                      f"||ΔC||$_F$/||C||$_F$ = {rel:.2e}   "
                      f"|ΔC|$_\\mathrm{{max}}$ = {abs_dc.max():.2e}")
    axes[2].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(
        "GP covariance kernel M·M$^T$ on DR16 (89k×200, Adam) — vectorized vs per-spectrum loss path",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT}")
    print()
    print(f"  ||C_vec||_F  = {np.linalg.norm(Cv):.4e}")
    print(f"  ||C_per||_F  = {np.linalg.norm(Cp):.4e}")
    print(f"  ||ΔC||_F     = {np.linalg.norm(dC):.4e}")
    print(f"  |ΔC|_max     = {abs_dc.max():.4e}")
    print(f"  rel Frobenius= {rel:.4e}")


if __name__ == "__main__":
    main()
