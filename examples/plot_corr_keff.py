"""Plot the inference-relevant corr(K_eff) for v1 + the 2 noisy 2lpt models.

K_eff = c_0² · M·M^T + diag(ω²)  is what the GP actually uses on the
observed-flux scale (modulo absorption ≈ 1). corr(K_eff) is therefore
the "physically readable" correlation that matches what the inference
pipeline sees — distinct from raw corr(M·M^T) which doesn't include
the c_0 scaling.

Output (without replacing existing plots):
  docs/notes/2026-05-12_2lpt_models_vs_v1_analysis/corr_Keff_inference_view.png
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
OUT_DIR = NOTES / "2026-05-12_2lpt_models_vs_v1_analysis"
OUT_PATH = OUT_DIR / "corr_Keff_inference_view.png"

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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 2 rows × N cols:
    #   Top row: corr(K_eff) full range [-1, +1] (diagonal dominates)
    #   Bottom row: corr(K_eff) with diagonal masked + colorbar tight to
    #     off-diag 1st/99th percentile so physical off-diag features pop.
    fig, axes = plt.subplots(2, len(MODELS), figsize=(6.5 * len(MODELS), 12.0))
    for i, (name, path) in enumerate(MODELS.items()):
        m = _load(path)
        M = m["M"]
        if M.ndim == 2 and M.shape[1] > M.shape[0]:
            M = M.T
        omega2 = np.exp(2 * m["log_omega"])
        c_0 = np.exp(m["log_c_0"])
        rw = m["rest"]
        K_eff = (c_0 ** 2) * (M @ M.T) + np.diag(omega2)
        C = _corr(K_eff)
        adj = np.abs(np.diff(C, axis=1)).mean()
        extent = [rw[0], rw[-1], rw[-1], rw[0]]

        # Top row: full [-1, +1] view
        ax0 = axes[0, i]
        im0 = ax0.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                         extent=extent, aspect="auto")
        ax0.set_title(f"({chr(ord('a')+i)}) {name}\n"
                      f"corr(K$_\\mathrm{{eff}}$), full range\n"
                      f"c_0 = {c_0:.4g}   mean adj diff = {adj:.4f}")
        ax0.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
        if i == 0:
            ax0.set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
        plt.colorbar(im0, ax=ax0, fraction=0.046, label="correlation")

        # Bottom row: diagonal masked, off-diag-only color range
        ax1 = axes[1, i]
        C_off = C.copy()
        # Mask the main diagonal AND the immediate near-diagonals to let
        # the wider off-diagonal structure dominate the colorbar.
        diag_mask = np.zeros_like(C, dtype=bool)
        for k in range(-1, 2):  # mask diagonal ± 1 pixel
            diag_mask |= np.eye(C.shape[0], C.shape[1], k=k, dtype=bool)
        C_off_masked = np.where(diag_mask, np.nan, C_off)
        # Tight color range based on off-diag percentile
        finite = C_off_masked[np.isfinite(C_off_masked)]
        vmax = float(np.percentile(np.abs(finite), 99))
        vmax = max(vmax, 1e-4)  # avoid degenerate colorbar
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="white")
        im1 = ax1.imshow(C_off_masked, cmap=cmap, vmin=-vmax, vmax=vmax,
                         extent=extent, aspect="auto")
        ax1.set_title(f"corr(K$_\\mathrm{{eff}}$), off-diag only "
                      f"(±{vmax:.4f} = 99th %ile, diag±1 masked)")
        ax1.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
        if i == 0:
            ax1.set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
        plt.colorbar(im1, ax=ax1, fraction=0.046, label="correlation")

    fig.suptitle(
        "corr(K$_\\mathrm{eff}$) — inference-relevant kernel correlation\n"
        "Top row: full [-1, +1] (diagonal dominates because ω² dominates K_eff). "
        "Bottom row: diagonal masked + tight colorbar so physical off-diag features "
        "(emission lines, continuum modes) become visible.",
        fontsize=12, y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
