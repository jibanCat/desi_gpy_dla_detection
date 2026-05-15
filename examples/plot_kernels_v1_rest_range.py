"""Compare trained GP kernels on the v1 rest range [850.90, 1420.60].

Why: the wide v2 rest grid [850.75, 1700] includes the per-spectrum
normalization region [1425, 1475], where by construction the centered
flux variance is near zero → K[i,i] = Σ_k M[i,k]² ≈ 0 → corr[i,j] =
K[i,j]/√(K[i,i]·K[j,j]) blows up. The bright stripes near λ ~ 1450 in
`2026-05-13_all_trained_kernels.png` are this artifact, NOT physics.

Cutting M to rest_λ ≤ 1420.60 (matching v1's grid) eliminates the
normalization region from the corr-normalization denominator, giving an
apples-to-apples comparison with v1 production.

Output: docs/notes/2026-05-13_kernels_v1_rest_range.png
"""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / "docs" / "notes"
OUT = NOTES / "2026-05-13_kernels_v1_rest_range.png"

# v1 rest range — everything else gets cut to this:
V1_REST_MAX = 1420.60
V1_REST_MIN = 850.90

ENTRIES = [
    # (display_name, path, kind)
    # Reference
    ("v1 production\n(real LOA, epoch 920) — reference",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5",
     "h5"),
    # Pre-reorder 2lpt _m baselines (kept as "before" references)
    ("2lpt loa-0 wide _m\n[1425/1475] norm, PRE-reorder",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa0_wide_m" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-124 _m\n[1425/1475] norm, PRE-reorder",
     str(NOTES / "2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m" / "phase2_result.h5"),
     "h5"),
    # 2026-05-14 post-reorder retrains — 5 new models
    ("2lpt loa-0 _m_normmask\n[1425/1475], POST-reorder",
     str(NOTES / "2026-05-14_desi_phase2_2lpt_loa0_wide_m_normmask" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-0 _g_normmask\n[1310/1325], POST-reorder",
     str(NOTES / "2026-05-14_desi_phase2_2lpt_loa0_wide_g_normmask" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-124 _m_normmask\n[1425/1475], POST-reorder",
     str(NOTES / "2026-05-14_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m_normmask" / "phase2_result.h5"),
     "h5"),
    ("2lpt loa-124 _g_normmask\n[1310/1325], POST-reorder",
     str(NOTES / "2026-05-14_desi_phase2_2lpt_loa124_nohcd_nobal_wide_g_normmask" / "phase2_result.h5"),
     "h5"),
    ("LOA no-DLA-no-BAL _m_normmask_3000iter\n[1425/1475], POST-reorder, walltime@2243/3000",
     str(NOTES / "2026-05-13_desi_phase2_loa_no_dla_no_bal_wide_m_normmask_3000iter" / "phase2_result.h5"),
     "h5"),
    ("LOA no-HCD-with-BAL _m_normmask_3000iter\n[1425/1475], POST-reorder, walltime@2461/3000",
     str(NOTES / "2026-05-13_desi_phase2_loa_no_hcd_with_bal_wide_m_normmask_3000iter" / "phase2_result.h5"),
     "h5"),
]


def _load_M_rest(path, kind):
    if kind == "h5":
        with h5py.File(path, "r") as f:
            M = np.asarray(f["M"][:], dtype=np.float64)
            rest = np.asarray(f["rest_wavelengths"][:], dtype=np.float64)
            log_c_0 = float(f["log_c_0"][()]) if "log_c_0" in f else float("nan")
            log_tau_0 = float(f["log_tau_0"][()]) if "log_tau_0" in f else float("nan")
            log_beta = float(f["log_beta"][()]) if "log_beta" in f else float("nan")
    else:  # pt
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        M = ckpt["M"].detach().cpu().numpy().astype(np.float64)
        # Generate rest grid from M shape (all wide v2 preloads start at
        # 850.75 with dλ=0.15; LOA has n_pix=5663, 2lpt has 5662 — proxy
        # h5 lookup gets the count wrong by one).
        n_pix = M.shape[0]
        rest = 850.75 + 0.15 * np.arange(n_pix, dtype=np.float64)
        log_c_0 = float(ckpt["log_c_0"].item())
        log_tau_0 = float(ckpt["log_tau_0"].item())
        log_beta = float(ckpt["log_beta"].item())
    scalars = dict(c_0=float(np.exp(log_c_0)),
                   tau_0=float(np.exp(log_tau_0)),
                   beta=float(np.exp(log_beta)))
    return M, rest, scalars


def _cut_to_v1_range(M, rest):
    """Truncate M to the v1 rest range [V1_REST_MIN, V1_REST_MAX]."""
    keep = (rest >= V1_REST_MIN) & (rest <= V1_REST_MAX)
    return M[keep], rest[keep]


def _corr(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 16))
    axes_flat = axes.flatten()

    for ax, (name, path, kind) in zip(axes_flat, ENTRIES):
        if not Path(path).exists():
            ax.text(0.5, 0.5, f"{name}\n\nNOT FOUND",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        try:
            M, rest, sc = _load_M_rest(path, kind)
            Mc, restc = _cut_to_v1_range(M, rest)
            C = _corr(Mc)
            adj = float(np.abs(np.diff(C, axis=1)).mean())
            extent = [restc[0], restc[-1], restc[-1], restc[0]]
            ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1,
                      extent=extent, aspect="auto")
            title = (f"{name}\n"
                     f"cut to [{restc[0]:.1f}, {restc[-1]:.1f}], "
                     f"n_pix={len(restc)}\n"
                     f"c_0={sc['c_0']:.4f}, τ_0={sc['tau_0']:.5f}, "
                     f"β={sc['beta']:.2f}, smooth={adj:.4f}")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(r"$\lambda_\mathrm{rest}$", fontsize=8)
            ax.set_ylabel(r"$\lambda_\mathrm{rest}$", fontsize=8)
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

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=matplotlib.colors.Normalize(vmin=-1, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="correlation")

    fig.suptitle(
        f"Trained corr(M·M$^T$) cut to v1 rest range [{V1_REST_MIN}, "
        f"{V1_REST_MAX}] Å — excludes the [1425, 1475] normalization band, "
        "so the diag-normalization isn't poisoned by near-zero variance there. "
        "Apples-to-apples with v1 production.",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 0.97])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
