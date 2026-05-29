"""Grid of corr(M·M^T) for all available trained GP models.

Loads M from .h5 (final models) and .pt (live checkpoints from
running/timed-out training). Each panel shows the kernel correlation
matrix + endpoint scalars (c_0, τ_0, β) in the title.

Output:
  docs/notes/2026-05-13_all_trained_kernels.png
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
NOTES = REPO / "docs" / "notes"
OUT = NOTES / "2026-05-13_all_trained_kernels.png"

# Each entry: (display_name, path, kind)
#   kind="h5"  → final .h5 model (look for M, mu, log_omega, scalars)
#   kind="pt"  → live checkpoint (PyTorch state dict with M, log_*)
ENTRIES = [
    ("v1 production\n(real LOA, epoch 920)",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",
     "h5"),
    ("2lpt loa-0 wide (base)\nwide σ, [1310, 1325]",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa0_wide" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-0 wide _g\nstrict Turner σ, [1310, 1325]",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa0_wide_g" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-0 wide _m\nstrict Turner σ, [1425, 1475]",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa0_wide_m" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-124 _g\nstrict Turner σ, [1310, 1325]",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_g" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-124 _m\nstrict Turner σ, [1425, 1475]",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-124 _c0prior\nlog_c_0 prior, [1310, 1325]",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior" / "phase2_result.h5"),
     "h5"),
    ("smoke (post-reorder)\n5k×50, normalize-then-mask FIX",
     str(NOTES / "2026-05-13_desi_smoke_normmask" / "phase2_result.h5"),
     "h5"),
    # Pre-reorder LOA checkpoints from the timed-out runs
    ("LOA no-DLA-no-BAL _g\niter 699 ckpt (TIMEOUT)",
     "/scratch/cavestru_root/cavestru0/mfho/phase2_desi/loa_no_dla_no_bal_wide_g/checkpoints/phase2_desi_checkpoint_iter0699.pt",
     "pt"),
    ("LOA no-DLA-no-BAL _m\niter 699 ckpt (TIMEOUT)",
     "/scratch/cavestru_root/cavestru0/mfho/phase2_desi/loa_no_dla_no_bal_wide_m/checkpoints/phase2_desi_checkpoint_iter0699.pt",
     "pt"),
    ("LOA no-HCD-with-BAL _g\niter 774 ckpt (TIMEOUT)",
     "/scratch/cavestru_root/cavestru0/mfho/phase2_desi/loa_no_hcd_with_bal_wide_g/checkpoints/phase2_desi_checkpoint_iter0774.pt",
     "pt"),
    ("LOA no-HCD-with-BAL _m\niter 799 ckpt (TIMEOUT)",
     "/scratch/cavestru_root/cavestru0/mfho/phase2_desi/loa_no_hcd_with_bal_wide_m/checkpoints/phase2_desi_checkpoint_iter0799.pt",
     "pt"),
]


def _corr(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def _load(path, kind):
    """Return (M, rest, scalars_dict). scalars_dict has c_0, tau_0, beta
    when readable."""
    if kind == "h5":
        with h5py.File(path, "r") as f:
            M = np.asarray(f["M"][:], dtype=np.float64)
            rest = np.asarray(f["rest_wavelengths"][:], dtype=np.float64)
            scalars = {}
            for k in ("log_c_0", "log_tau_0", "log_beta"):
                if k in f:
                    scalars[k] = float(f[k][()])
        out_scalars = dict(
            c_0=float(np.exp(scalars.get("log_c_0", np.nan))),
            tau_0=float(np.exp(scalars.get("log_tau_0", np.nan))),
            beta=float(np.exp(scalars.get("log_beta", np.nan))),
        )
        return M, rest, out_scalars
    elif kind == "pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        M_t = ckpt.get("M")
        if M_t is None:
            raise KeyError(f"no 'M' key in {path}")
        M = M_t.detach().cpu().numpy().astype(np.float64)
        # rest grid isn't in checkpoint; load from any wide preload as a
        # proxy (all LOA preloads share the same wide rest grid)
        ref_h5 = NOTES / "2026-05-13_desi_smoke_normmask" / "phase2_result.h5"
        with h5py.File(ref_h5, "r") as f:
            rest = np.asarray(f["rest_wavelengths"][:], dtype=np.float64)
        log_c_0 = float(ckpt.get("log_c_0", torch.tensor(np.nan)).item())
        log_tau_0 = float(ckpt.get("log_tau_0", torch.tensor(np.nan)).item())
        log_beta = float(ckpt.get("log_beta", torch.tensor(np.nan)).item())
        out_scalars = dict(
            c_0=float(np.exp(log_c_0)),
            tau_0=float(np.exp(log_tau_0)),
            beta=float(np.exp(log_beta)),
        )
        return M, rest, out_scalars
    raise ValueError(f"unknown kind: {kind}")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes_flat = axes.flatten()
    cmap = "RdBu_r"

    for ax, (name, path, kind) in zip(axes_flat, ENTRIES):
        if not Path(path).exists():
            ax.text(0.5, 0.5, f"{name}\n\nNOT FOUND",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        try:
            M, rest, sc = _load(path, kind)
            C = _corr(M)
            adj = float(np.abs(np.diff(C, axis=1)).mean())
            extent = [rest[0], rest[-1], rest[-1], rest[0]]
            im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1,
                           extent=extent, aspect="auto")
            title = (f"{name}\n"
                     f"c_0={sc['c_0']:.4f}, τ_0={sc['tau_0']:.5f}, "
                     f"β={sc['beta']:.2f}\nsmooth={adj:.4f}")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(r"$\lambda_\mathrm{rest}$", fontsize=7)
            ax.set_ylabel(r"$\lambda_\mathrm{rest}$", fontsize=7)
            ax.tick_params(labelsize=6)
        except Exception as e:
            ax.text(0.5, 0.5, f"{name}\n\nERROR\n{type(e).__name__}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            print(f"  ERROR on {name}: {e!r}")

    # Hide unused axes
    for ax in axes_flat[len(ENTRIES):]:
        ax.set_visible(False)

    # One shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(vmin=-1, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="correlation")

    fig.suptitle(
        "All trained GP corr(M·M$^T$) — Step C 2lpt models + smoke + LOA checkpoints. "
        "v1 production = reference (real LOA, epoch 920). 'smooth' = mean adjacent-pixel "
        "|Δcorr|; lower = smoother kernel.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.97])
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
