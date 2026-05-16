"""Correlation-matrix comparison: vec full vs per-spec full at production DR16 scale.

Same convention as `phase2_train_dr16._corr`:
  C(M) = (M·M^T) / outer(sqrt(diag), sqrt(diag)),  clipped to [-1, 1].

Renders a 3-panel overlay:
  (a) corr(M_vec)
  (b) corr(M_per)
  (c) Δcorr = corr(M_vec) - corr(M_per)   (signed, RdBu_r)

Output:
  docs/notes/2026-05-11_vec_vs_perspec_corr.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
OUT = NOTES / "2026-05-11_vec_vs_perspec_corr.png"

VEC_FULL = NOTES / "2026-05-08_matlab_dr16_validation_vec_full" / "phase2_result.npz"
PER_SPEC = NOTES / "2026-05-08_matlab_dr16_validation_per_spec" / "phase2_result.npz"


def _corr(M: np.ndarray) -> np.ndarray:
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def main():
    vec = np.load(VEC_FULL)
    per = np.load(PER_SPEC)

    rw = vec["rest_wavelengths"]
    assert np.allclose(rw, per["rest_wavelengths"]), "rest_wavelengths mismatch"

    Cv = _corr(vec["M"])
    Cp = _corr(per["M"])
    dC = Cv - Cp

    extent = [rw[0], rw[-1], rw[-1], rw[0]]
    dmax = float(np.max(np.abs(dC)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    im0 = axes[0].imshow(Cv, cmap="RdBu_r", vmin=-1, vmax=1,
                         extent=extent, aspect="auto")
    axes[0].set_title("(a) corr(M·M$^T$) — vec full (49700040)")
    axes[0].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    axes[0].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="correlation")

    im1 = axes[1].imshow(Cp, cmap="RdBu_r", vmin=-1, vmax=1,
                         extent=extent, aspect="auto")
    axes[1].set_title("(b) corr(M·M$^T$) — per-spec (49709974)")
    axes[1].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="correlation")

    im2 = axes[2].imshow(dC, cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                         extent=extent, aspect="auto")
    rel_F = np.linalg.norm(dC) / np.linalg.norm(Cv)
    axes[2].set_title(
        f"(c) Δcorr = corr(M$_\\mathrm{{vec}}$) − corr(M$_\\mathrm{{per}}$)\n"
        f"|Δcorr|$_\\mathrm{{max}}$ = {dmax:.3e}   "
        f"||Δcorr||$_F$/||corr$_\\mathrm{{vec}}$||$_F$ = {rel_F:.2e}")
    axes[2].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="Δ correlation")

    fig.suptitle(
        "Correlation matrix corr(M·M$^T$) on DR16 (89k×200, Adam) — "
        "vectorized vs per-spectrum loss path",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT}")
    print()
    print(f"  corr range  vec : [{Cv.min():+.4f}, {Cv.max():+.4f}]")
    print(f"  corr range  per : [{Cp.min():+.4f}, {Cp.max():+.4f}]")
    print(f"  |Δcorr|_max     : {dmax:.4e}")
    print(f"  ||Δcorr||_F     : {np.linalg.norm(dC):.4e}")
    print(f"  rel Frobenius   : {rel_F:.4e}")
    print(f"  mean |Δcorr|    : {np.mean(np.abs(dC)):.4e}")
    # diagonal sanity (should be 1.0 for both — rounding)
    print(f"  max |diag − 1|  : "
          f"vec={np.max(np.abs(np.diag(Cv) - 1)):.2e}  "
          f"per={np.max(np.abs(np.diag(Cp) - 1)):.2e}")


if __name__ == "__main__":
    main()
