"""Compare trained GP kernels — corr(M·M^T) — on the v1 rest range.

Why the v1-range cut: the wide v2 rest grid [850.75, 1700] includes each
model's per-spectrum normalization band, where by construction the
centered flux variance is near zero -> K[i,i] = Sum_k M[i,k]^2 ~ 0 ->
corr[i,j] = K[i,j]/sqrt(K[i,i]*K[j,j]) blows up.

Two-part normalization-band exclusion:
  1. Cut every model to rest_lambda <= 1420.60 (v1's grid). This drops
     the MATLAB-band [1425, 1475] used by the `_m` models and keeps the
     comparison apples-to-apples with v1 production.
  2. Per model, additionally blank the corr rows/cols inside THAT model's
     own normalization band (read from the .h5). This catches the
     Garnett-band [1310, 1325] used by the `_g` models, which sits inside
     the v1 cut and would otherwise still poison the panel.

Line markers: minimal Ly-series + metal absorption lines are drawn as
faint gridlines on both axes, and three representative cross-correlation
points are marked — Ly-Ly (Lya x Lyb), Ly-metal (Lya x CII), and
metal-metal (CII x SiIV) — so the off-diagonal structure can be read
against known transitions.

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

# Minimal line set for axis gridlines (vacuum rest Å, within the v1 range).
LY_LINES = {"Lyγ": 972.54, "Lyβ": 1025.72, "Lyα": 1215.67}
METAL_LINES = {"SiIII": 1206.50, "OI": 1302.17, "CII": 1334.53, "SiIV": 1393.76}

# Three representative cross-correlation points: (label, kind, λ_x, λ_y).
CROSS_POINTS = [
    ("Lyα×Lyβ  (Ly–Ly)",        "o", 1215.67, 1025.72),
    ("Lyα×CII  (Ly–metal)",     "s", 1215.67, 1334.53),
    ("CII×SiIV  (metal–metal)", "D", 1334.53, 1393.76),
]

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
    """Return (M, rest_wavelengths, scalars, norm_band).

    norm_band is (min, max) Å read from the .h5, or None if absent."""
    norm_band = None
    if kind == "h5":
        with h5py.File(path, "r") as f:
            M = np.asarray(f["M"][:], dtype=np.float64)
            rest = np.asarray(f["rest_wavelengths"][:], dtype=np.float64)
            log_c_0 = float(f["log_c_0"][()]) if "log_c_0" in f else float("nan")
            log_tau_0 = float(f["log_tau_0"][()]) if "log_tau_0" in f else float("nan")
            log_beta = float(f["log_beta"][()]) if "log_beta" in f else float("nan")
            if "normalization_min_lambda" in f and "normalization_max_lambda" in f:
                norm_band = (float(f["normalization_min_lambda"][()]),
                             float(f["normalization_max_lambda"][()]))
    else:  # pt
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        M = ckpt["M"].detach().cpu().numpy().astype(np.float64)
        n_pix = M.shape[0]
        rest = 850.75 + 0.15 * np.arange(n_pix, dtype=np.float64)
        log_c_0 = float(ckpt["log_c_0"].item())
        log_tau_0 = float(ckpt["log_tau_0"].item())
        log_beta = float(ckpt["log_beta"].item())
    scalars = dict(c_0=float(np.exp(log_c_0)),
                   tau_0=float(np.exp(log_tau_0)),
                   beta=float(np.exp(log_beta)))
    return M, rest, scalars, norm_band


def _cut_to_v1_range(M, rest):
    """Truncate M to the v1 rest range [V1_REST_MIN, V1_REST_MAX]."""
    keep = (rest >= V1_REST_MIN) & (rest <= V1_REST_MAX)
    return M[keep], rest[keep]


def _corr(M):
    K = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-30))
    return np.clip(K / np.outer(d, d), -1.0, 1.0)


def _blank_norm_band(C, restc, norm_band):
    """NaN-out corr rows/cols inside the model's own normalization band so
    its near-zero-variance stripe doesn't show. Returns (C, n_blanked)."""
    if norm_band is None:
        return C, 0
    lo, hi = norm_band
    in_band = (restc >= lo) & (restc <= hi)
    if not in_band.any():
        return C, 0
    C = C.copy()
    C[in_band, :] = np.nan
    C[:, in_band] = np.nan
    return C, int(in_band.sum())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 16))
    axes_flat = axes.flatten()

    cmap = matplotlib.cm.RdBu_r.copy()
    cmap.set_bad("lightgrey")   # NaN (blanked norm band) renders grey

    for ax, (name, path, kind) in zip(axes_flat, ENTRIES):
        if not Path(path).exists():
            ax.text(0.5, 0.5, f"{name}\n\nNOT FOUND",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        try:
            M, rest, sc, norm_band = _load_M_rest(path, kind)
            Mc, restc = _cut_to_v1_range(M, rest)
            C = _corr(Mc)
            C, n_blank = _blank_norm_band(C, restc, norm_band)
            adj = float(np.nanmean(np.abs(np.diff(C, axis=1))))
            extent = [restc[0], restc[-1], restc[-1], restc[0]]
            ax.imshow(C, cmap=cmap, vmin=-1, vmax=1,
                      extent=extent, aspect="auto")
            band_txt = (f"norm band [{norm_band[0]:.0f},{norm_band[1]:.0f}] "
                        f"blanked ({n_blank} pix)" if n_blank
                        else "norm band outside cut")
            title = (f"{name}\n"
                     f"cut [{restc[0]:.0f},{restc[-1]:.0f}], {band_txt}\n"
                     f"c_0={sc['c_0']:.4f}, τ_0={sc['tau_0']:.5f}, "
                     f"β={sc['beta']:.2f}, smooth={adj:.4f}")
            ax.set_title(title, fontsize=7.5)
            ax.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]", fontsize=8)
            ax.set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]", fontsize=8)
            ax.tick_params(labelsize=6)

            # Faint gridlines at the Ly + metal lines (both axes).
            for lname, lw_ in {**LY_LINES, **METAL_LINES}.items():
                if not (restc[0] < lw_ < restc[-1]):
                    continue
                is_ly = lname in LY_LINES
                col = "#222222" if is_ly else "#7a4500"
                for axline in (ax.axvline, ax.axhline):
                    axline(lw_, color=col, lw=0.5, ls=":", alpha=0.30)
            # Three representative cross-correlation points.
            for label, marker, lx, ly in CROSS_POINTS:
                if (restc[0] < lx < restc[-1]) and (restc[0] < ly < restc[-1]):
                    ax.plot(lx, ly, marker=marker, ms=8, mfc="none",
                            mec="black", mew=1.4, linestyle="none")
                    ax.plot(ly, lx, marker=marker, ms=8, mfc="none",
                            mec="black", mew=1.4, linestyle="none")
        except Exception as e:
            ax.text(0.5, 0.5, f"{name}\n\nERROR\n{type(e).__name__}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            print(f"  ERROR on {name}: {e!r}")

    for ax in axes_flat[len(ENTRIES):]:
        ax.set_visible(False)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=matplotlib.colors.Normalize(vmin=-1, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label="correlation")

    # Legend for the cross-correlation markers.
    handles = [plt.Line2D([], [], marker=mk, ms=8, mfc="none", mec="black",
                          mew=1.4, linestyle="none", label=lbl)
               for lbl, mk, _, _ in CROSS_POINTS]
    handles += [
        plt.Line2D([], [], color="#222222", lw=1.0, ls=":", label="Ly-series line"),
        plt.Line2D([], [], color="#7a4500", lw=1.0, ls=":", label="metal line"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        f"Trained corr(M·M$^T$), v1 rest range [{V1_REST_MIN}, {V1_REST_MAX}] Å. "
        "Each model's own normalization band is blanked (grey). Dotted lines = "
        "Ly-series (black) / metal (brown); markers = representative "
        "Ly–Ly, Ly–metal, metal–metal cross-correlations.",
        fontsize=10.5, y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 0.92, 0.97])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
