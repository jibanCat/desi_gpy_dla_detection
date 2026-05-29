"""Plot the full kernel hierarchy for the 2lpt-trained DESI models.

Hypothesis (user 2026-05-12): the noisy-looking corr(M·M^T) might be
an artifact of diag-normalization — the actual physical kernel
K = M·M^T + diag(ω²) might be dominated by ω² and look fine.

Output (under docs/notes/2026-05-12_2lpt_models_vs_v1_analysis/):
  full_kernel_2lpt_loa0.png   — 2x2 panel for 2lpt loa-0
  full_kernel_2lpt_loa124.png — 2x2 panel for 2lpt loa-124
  full_kernel_v1_production.png — control (v1 smooth model)

Each 2x2 panel:
  (a) M·M^T               — pure low-rank covariance (raw values)
  (b) corr(M·M^T)         — what we've been calling "noisy"
  (c) K = M·M^T + diag(ω²)— full kernel (with diagonal noise)
  (d) corr(K)             — kernel correlation
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
OUT_DIR = NOTES / "2026-05-12_2lpt_models_vs_v1_analysis"

MODELS = {
    "v1_production": "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",
    "2lpt_loa0_wide": str(NOTES / "2026-05-11_desi_phase2_2lpt_loa0_wide" / "phase2_result.h5"),
    "2lpt_loa124_nohcd_nobal_wide": str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide" / "phase2_result.h5"),
}


def _load(p: str) -> dict:
    with h5py.File(p, "r") as f:
        return dict(
            M=np.asarray(f["M"][:]),
            log_omega=np.asarray(f["log_omega"][:]),
            log_c_0=float(np.asarray(f["log_c_0"])),
            rest=np.asarray(f["rest_wavelengths"][:]),
        )


def _corr(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def make_panel(name: str, model: dict):
    M = model["M"]
    if M.ndim == 2 and M.shape[1] > M.shape[0]:
        M = M.T
    omega2 = np.exp(2 * model["log_omega"])
    rw = model["rest"]
    extent = [rw[0], rw[-1], rw[-1], rw[0]]

    log_c_0 = float(model["log_c_0"])
    c_0 = np.exp(log_c_0)
    MMt = M @ M.T              # low-rank covariance
    K = MMt + np.diag(omega2)  # raw full kernel (no c_0 scaling)
    # Effective inference kernel: what the GP actually uses on observed flux
    # scale (modulo absorption ≈ 1). c_0² * M·M^T is the actual covariance
    # contribution from M; ω² is the per-pixel diagonal noise that doesn't
    # scale with c_0. With c_0~0.004 (degenerate), c_0²·M·M^T gets crushed
    # 6 orders of magnitude → ω² dominates → K_eff is nearly diagonal.
    K_eff = (c_0 ** 2) * MMt + np.diag(omega2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panels = [
        ("(a) M·M$^T$ (raw low-rank covariance)",  MMt, "RdBu_r", None),
        ("(b) corr(M·M$^T$)",                      _corr(MMt), "RdBu_r", (-1, 1)),
        ("(c) K = M·M$^T$ + diag(ω²) (raw kernel)", K,  "RdBu_r", None),
        ("(d) corr(K)",                             _corr(K),   "RdBu_r", (-1, 1)),
        (f"(e) K$_\\mathrm{{eff}}$ = c$_0^2$·M·M$^T$ + diag(ω²) [c_0={c_0:.4g}]",
         K_eff, "RdBu_r", None),
        ("(f) corr(K$_\\mathrm{eff}$) — what GP sees at inference",
         _corr(K_eff), "RdBu_r", (-1, 1)),
    ]
    for ax, (title, mat, cmap, vrange) in zip(axes.ravel(), panels):
        if vrange is None:
            v = float(np.max(np.abs(mat)))
            vmin, vmax = -v, v
        else:
            vmin, vmax = vrange
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
        ax.set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
        plt.colorbar(im, ax=ax, fraction=0.046)
        # Print summary
        d = np.diag(mat)
        if vrange is not None:
            adj = np.abs(np.diff(mat, axis=1)).mean()
            ax.text(0.02, 0.98, f"mean adj diff={adj:.4f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(f"Full kernel hierarchy: {name}\n"
                 f"(M.shape={M.shape}; ω² range [{omega2.min():.3e}, {omega2.max():.3e}])",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / f"full_kernel_{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # Numerical summary
    print(f"  M·M^T range: [{MMt.min():.3e}, {MMt.max():.3e}]  "
          f"diag mean={np.diag(MMt).mean():.3e}")
    print(f"  ω² range:    [{omega2.min():.3e}, {omega2.max():.3e}]  "
          f"mean={omega2.mean():.3e}")
    print(f"  ω² / diag(M·M^T) ratio: median={np.median(omega2/np.diag(MMt)):.3e}  "
          f"max={(omega2/np.diag(MMt)).max():.3e}")
    print(f"  corr(M·M^T) adj diff: {np.abs(np.diff(_corr(MMt), axis=1)).mean():.4f}")
    print(f"  corr(K) adj diff:     {np.abs(np.diff(_corr(K),   axis=1)).mean():.4f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in MODELS.items():
        print(f"\n=== {name} ===")
        make_panel(name, _load(path))


if __name__ == "__main__":
    main()
