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
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5 * len(MODELS), 6.2))
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
        im = axes[i].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                            extent=extent, aspect="auto")
        axes[i].set_title(f"({chr(ord('a')+i)}) {name}\n"
                          f"corr(K$_\\mathrm{{eff}}$ = c$_0^2$·M·M$^T$ + diag(ω²))\n"
                          f"c_0 = {c_0:.4g}   mean adj diff = {adj:.4f}")
        axes[i].set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
        if i == 0:
            axes[i].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")
        plt.colorbar(im, ax=axes[i], fraction=0.046, label="correlation")

    fig.suptitle(
        "corr(K$_\\mathrm{eff}$) — inference-relevant kernel correlation. "
        "The 2lpt models look noisy in raw corr(M·M$^T$) but smooth here "
        "because c_0² ≈ 1.6e-5 crushes the M contribution; "
        "ω² dominates on the observed-flux scale.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
