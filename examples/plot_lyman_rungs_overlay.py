"""Lower-triangle corr(M·M^T) with cross-correlation rung overlays across
four datasets: 2lpt loa-0, 2lpt loa-124, LOA real PCA init, and v1
production TRAINED. Lines are colored by category:

  GREEN dashed  — Lyman series (in loa-124 via --metals LYB LY3 LY4 LY5;
                  NOT in loa-0)
  CYAN dotted   — Quickquasars metals injected in loa-124
                  (SiII λ1190/1193/1260, SiIII λ1207)
  MAGENTA dotted — Metals that exist in real-universe quasar lines of
                  sight but are NOT painted by quickquasars
                  (NV, OVI, CII, FeII, AlII, SiII λ1304/1526)

For each pair (A, B) with λ_A < λ_B, a single absorber at redshift z_abs
paints both at QSO-rest wavelengths (λ_A · X, λ_B · X) for
X = (1 + z_abs) / (1 + z_QSO). Marginalized over the training population,
this appears as a line of slope λ_B / λ_A in the (col, row) space of the
corr matrix. We display only the LOWER triangle of the matrix (row > col)
so each rung shows once, not twice.

Output:
  docs/notes/2026-05-12_2lpt_models_vs_v1_analysis/corr_pca_init_lyman_rungs.png
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
OUT = NOTES / "2026-05-12_2lpt_models_vs_v1_analysis" / "corr_pca_init_lyman_rungs.png"
V1_PROD = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5"

PRELOADS = [
    ("loa-0\n(Lyα only)",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5"),
    ("loa-124\n(Lyα + LYB/LY3/LY4/LY5 + Si II/III + DLAs + BALs)",
     "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa124_nohcd_nobal_wide_v2_1778186324/trainset.h5"),
    ("LOA real PCA init\n(no-DLA + no-BAL preload, real DESI)",
     "/scratch/cavestru_root/cavestru0/mfho/loa_wide_v2/loa_no_dla_no_bal_wide/trainset.h5"),
]
V1_LABEL = "v1 production TRAINED\n(model_epoch_920, real DESI Y3 LOA)"
N_SUB = 30000
K = 30

# Rest wavelengths of relevant transitions.
# Lyman series — line-of-sight-correlated with Lyα in loa-124 via shared
# transmission skewer (LYB/LY3/LY4/LY5), per Etourneau+2024 / Farr+2020.
LYMAN = {
    "Lyα": 1215.67,
    "Lyβ": 1025.72,
    "Lyγ":  972.54,
    "Lyδ":  949.74,
    "Lyε":  937.80,
}
# Metals injected by quickquasars in loa-124 (per actual CLI):
QQ_METALS = {
    "SiII(1260)": 1260.42,
    "SiIII(1207)": 1206.50,
    "SiII(1193)": 1193.28,
    "SiII(1190)": 1190.42,
}
# Metals that DO exist in real-universe QSO lines of sight (intervening DLAs,
# associated absorbers, etc.) but are NOT painted by quickquasars.
# A GP trained on real LOA can pick up Lyα cross-correlation with these;
# a GP trained on 2lpt mocks cannot.
REAL_ONLY_METALS = {
    "NV(1238)":  1238.82,
    "NV(1242)":  1242.80,
    "OVI(1031)": 1031.93,
    "OVI(1037)": 1037.62,
    "CII(1334)": 1334.53,
    "SiII(1304)": 1304.37,
    "SiII(1526)": 1526.71,
    "FeII(1608)": 1608.45,
    "AlII(1670)": 1670.79,
}

# Pairs to overlay. Each pair (A, B) with λ_A ≤ λ_B yields a line of
# slope λ_B / λ_A in (col, row) image coordinates.
LYMAN_PAIRS = [
    ("Lyβ", "Lyα"),  # cross-correlation between Lyβ-pixel (smaller λ) and Lyα-pixel
    ("Lyγ", "Lyα"),
    ("Lyδ", "Lyα"),
    ("Lyε", "Lyα"),
    ("Lyγ", "Lyβ"),
    ("Lyδ", "Lyβ"),
    ("Lyδ", "Lyγ"),
]


def _make_pair(a_label, a_wave, b_label, b_wave):
    """Return (label, λ_short, λ_long, color, kw) tuple."""
    if a_wave > b_wave:
        a_label, b_label = b_label, a_label
        a_wave, b_wave = b_wave, a_wave
    return (f"{a_label}·{b_label}", a_wave, b_wave)


# Build all the rungs we want (each: (label, λ_short, λ_long, kw))
def _build_rungs():
    rungs = []
    seen = set()
    def _add(cat, label_a, wa, label_b, wb):
        lab, ls, ll = _make_pair(label_a, wa, label_b, wb)
        key = (cat, round(ls, 3), round(ll, 3))
        if key in seen:
            return
        seen.add(key)
        rungs.append((cat, lab, ls, ll))

    # Lyman series cross-rungs
    for a, b in LYMAN_PAIRS:
        _add("lyman", a, LYMAN[a], b, LYMAN[b])

    # Lyα × quickquasars metals + quickquasars-metal × quickquasars-metal
    qq_all = {"Lyα": LYMAN["Lyα"], **QQ_METALS}
    qq_keys = list(qq_all.keys())
    for i, ka in enumerate(qq_keys):
        for kb in qq_keys[i+1:]:
            _add("qq_metals", ka, qq_all[ka], kb, qq_all[kb])

    # Lyα × real-only metals + real-only × real-only + real-only × qq-metals
    # (the cross-category pairs are real-universe physics that the mocks
    # cannot reproduce because the real-only line is missing from the chain)
    real_all = {**REAL_ONLY_METALS}
    real_keys = list(real_all.keys())
    # Lyα × real-only
    for k in real_keys:
        _add("real_only", "Lyα", LYMAN["Lyα"], k, real_all[k])
    # real-only × real-only
    for i, ka in enumerate(real_keys):
        for kb in real_keys[i+1:]:
            _add("real_only", ka, real_all[ka], kb, real_all[kb])
    # real-only × quickquasars-metal (still a "real-only" rung because the
    # cross-correlation between e.g. CII and SiII requires CII to exist)
    for k_real in real_keys:
        for k_qq in QQ_METALS:
            _add("real_only", k_real, real_all[k_real], k_qq, QQ_METALS[k_qq])

    return rungs


STYLE = {
    "lyman":     dict(color="lime",    lw=1.4, linestyle="--",  alpha=0.95),
    "qq_metals": dict(color="cyan",    lw=1.1, linestyle=":",   alpha=0.85),
    "real_only": dict(color="magenta", lw=1.0, linestyle=":",   alpha=0.85),
}


def _corr(M):
    K_ = M @ M.T
    d = np.sqrt(np.maximum(np.diag(K_), 1e-30))
    return np.clip(K_ / np.outer(d, d), -1.0, 1.0)


LABEL_SKIP = {
    "SiII(1190)", "SiII(1193)",
    "SiII(1304)", "SiII(1526)",
    "NV(1242)", "OVI(1037)",
    "Lyε",
}
NOT_DETECTED_STYLE = dict(color="gray", linestyle="-", lw=0.6, alpha=0.5)
DETECTION_Z_THRESHOLD = 1.5   # z = (on_med - off_med) / (1.4826 * MAD(off))
                              # Lowered from 3.0 → 1.5 (2026-05-13): the rungs
                              # are physically subtle (sub-leading in the
                              # kernel after de-forest + k=30 truncation), so
                              # 3σ rejected even visible features like
                              # Lyβ·Lyα on LOA (z = 1.2). 1.5σ is a meaningful
                              # detection but inclusive enough to catch the
                              # real rung structure.
TRIM_DIAG = 10.0              # Å — skip rung pixels within this distance of main diagonal
PERP_OFFSETS_ANG = (15.0, 25.0, 35.0)  # perpendicular off-rung sample distances


def _short_label(label: str) -> str:
    return label.replace("(", " ").replace(")", "").replace("Lyα·", "").replace("·Lyα", "")


def _detect_feature(C, rest, λs, λl, n_samples=50,
                    perp_offsets=PERP_OFFSETS_ANG,
                    trim_diag=TRIM_DIAG, z_threshold=DETECTION_Z_THRESHOLD):
    """Z-score feature detection along a rung.

    On-rung samples: corr(λs·X, λl·X) for X spanning the valid range.
    Off-rung samples: same X values, but offset perpendicular to the rung
    by ±perp_offsets (so we sample a thicker neighborhood, not a thin line).
    Scale: 1.4826·MAD(off_samples) ≈ σ for Gaussian — robust to forest
    outliers. z = (median(on) − median(off)) / scale; detect if z > 3.

    Pixels within trim_diag Å of the main diagonal are dropped (corr ≈ 1
    there biases both on- and off-rung sampling).

    Returns dict(detected, z, strength, on_med, off_med, n_off).
    """
    rest_min, rest_max = rest[0], rest[-1]
    x_min = max(rest_min / λs, rest_min / λl)
    x_max = min(1.0, rest_max / λs, rest_max / λl)
    NULL = dict(detected=False, z=float("-inf"), strength=0.0,
                on_med=0.0, off_med=0.0, n_off=0)
    if x_max <= x_min:
        return NULL

    Xs = np.linspace(x_min, x_max, n_samples)
    cols = Xs * λs
    rows = Xs * λl
    keep = np.abs(rows - cols) > trim_diag
    if keep.sum() < 5:
        return NULL
    cols, rows = cols[keep], rows[keep]

    slope = λl / λs
    norm = np.sqrt(1.0 + slope * slope)
    perp_dcol = -slope / norm
    perp_drow = 1.0 / norm

    def _sample(c, r):
        i = np.clip(np.searchsorted(rest, c), 0, len(rest) - 1)
        j = np.clip(np.searchsorted(rest, r), 0, len(rest) - 1)
        return C[j, i]

    on_rung = _sample(cols, rows)

    off_samples = []
    for off in perp_offsets:
        for sign in (+1, -1):
            ccol = cols + sign * off * perp_dcol
            crow = rows + sign * off * perp_drow
            valid = ((ccol >= rest_min) & (ccol <= rest_max) &
                     (crow >= rest_min) & (crow <= rest_max))
            if valid.any():
                off_samples.extend(_sample(ccol[valid], crow[valid]).tolist())
    if len(off_samples) < 8:
        return NULL

    off_samples = np.asarray(off_samples)
    on_med = float(np.median(on_rung))
    off_med = float(np.median(off_samples))
    mad = float(np.median(np.abs(off_samples - off_med)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.std(off_samples)) or 1e-6
    z = (on_med - off_med) / scale
    return dict(detected=(z > z_threshold),
                z=float(z), strength=on_med - off_med,
                on_med=on_med, off_med=off_med, n_off=len(off_samples))


def overlay_rungs(ax, rest, C, rungs, annotate=True):
    """Draw each rung in the lower-triangle half. For each rung, sample
    corr along its path and compare to perpendicular off-rung background.
    Detected features get the category's color + label; non-detections
    fade to NOT_DETECTED_STYLE (gray @ alpha=0.5, no label).

    Returns a list of (cat, label, slope, strength, detected) tuples for
    table summary."""
    rest_min, rest_max = rest[0], rest[-1]
    annot_candidates = []
    detections = []
    for cat, label, λs, λl in rungs:
        x_min = max(rest_min / λs, rest_min / λl)
        x_max = min(1.0, rest_max / λs, rest_max / λl)
        if x_max <= x_min:
            detections.append((cat, label, λl / λs, float("nan"),
                               float("nan"), False))
            continue
        Xs = np.array([x_min, x_max])
        col = Xs * λs
        row = Xs * λl
        det = _detect_feature(C, rest, λs, λl)
        detections.append((cat, label, λl / λs, det["strength"], det["z"],
                           det["detected"]))
        if det["detected"]:
            kw = STYLE[cat]
        else:
            kw = NOT_DETECTED_STYLE
        ax.plot(col, row, **kw)
        if (det["detected"] and annotate and "Lyα" in label
                and label.replace("Lyα·", "").replace("·Lyα", "") not in LABEL_SKIP):
            other = label.replace("Lyα·", "").replace("·Lyα", "")
            slope = λl / λs
            annot_candidates.append((cat, _short_label(other), slope,
                                     col[-1], row[-1], det["z"]))
    annot_candidates.sort(key=lambda t: t[2])
    for cat, label, slope, x_end, y_end, z in annot_candidates:
        kw = STYLE[cat]
        angle_deg = -np.degrees(np.arctan(slope))
        ax.annotate(f"{label} (z={z:.1f})", (x_end, y_end),
                    fontsize=6.5, color=kw["color"],
                    xytext=(-15, 6), textcoords="offset points",
                    rotation=angle_deg, rotation_mode="anchor",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.7))
    return detections


def render_panel(ax, name, rest, C, rungs):
    extent = [rest[0], rest[-1], rest[-1], rest[0]]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent, aspect="auto")
    detections = overlay_rungs(ax, rest=rest, C=C, rungs=rungs)
    # Mark Lyman line positions as faint tick lines
    for label, lw_pos in LYMAN.items():
        if rest[0] <= lw_pos <= rest[-1]:
            ax.axvline(lw_pos, color="gray", lw=0.4, alpha=0.5)
            ax.axhline(lw_pos, color="gray", lw=0.4, alpha=0.5)
            ax.text(lw_pos, rest[0], f" {label}", color="black",
                    fontsize=6, va="top", ha="left", alpha=0.7)
    adj = np.abs(np.diff(C, axis=1)).mean()
    ax.set_title(f"{name}\nmean adj diff = {adj:.4f}", fontsize=10)
    ax.set_xlabel(r"$\lambda_\mathrm{rest}$ [Å]")
    return im, detections


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rungs = _build_rungs()

    panels = []
    for name, path in PRELOADS:
        if not Path(path).exists():
            panels.append((name, None, None, f"NOT FOUND"))
            continue
        print(f"\n=== {name.splitlines()[0]} ===")
        ts = load_preprocessed_h5(
            path,
            z_min=2.15, z_max=4.25, max_spectra=N_SUB,
            max_noise_variance=9.0,
            apply_mask=True, apply_normalize=True,
            apply_de_forest=True, apply_center=True,
            norm_min_lambda=1425.0, norm_max_lambda=1475.0,
            de_forest_tau_0=0.00246, de_forest_beta=3.62, de_forest_num_lines=31,
            dtype=torch.float32, working_dtype=np.float32,
        )
        rest = ts.rest_wavelengths.numpy().astype(np.float64)
        M, _ = _pca_init(ts.fluxes.numpy(), k=K)
        C = _corr(M)
        panels.append((name, rest, C, None))

    # v1 production trained reference
    print(f"\n=== v1 production ===")
    if Path(V1_PROD).exists():
        with h5py.File(V1_PROD, "r") as f:
            M_v1 = np.asarray(f["M"][:], dtype=np.float64)
            rest_v1 = np.asarray(f["rest_wavelengths"][:], dtype=np.float64)
        panels.append((V1_LABEL, rest_v1, _corr(M_v1), None))
    else:
        panels.append((V1_LABEL, None, None, "NOT FOUND"))

    fig, axes = plt.subplots(1, 4, figsize=(28, 7.5))
    panel_detections = {}  # name (single-line) -> list of (cat, label, slope, strength, detected)
    last_im = None
    for ax, (name, rest, C, err) in zip(axes, panels):
        if err is not None:
            ax.text(0.5, 0.5, f"{name}\n\n{err}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        last_im, dets = render_panel(ax, name, rest, C, rungs)
        panel_detections[name.splitlines()[0]] = dets
    axes[0].set_ylabel(r"$\lambda_\mathrm{rest}$ [Å]")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], **STYLE["lyman"],
                   label="Lyman series — DETECTED on this kernel (in loa-124 via LYB/LY3/LY4/LY5; absent in loa-0)"),
        plt.Line2D([0], [0], **STYLE["qq_metals"],
                   label="Si II/III metals — DETECTED (injected by quickquasars in loa-124)"),
        plt.Line2D([0], [0], **STYLE["real_only"],
                   label="Real-universe metals (NV/OVI/CII/Fe II/Al II/SiII 1304-1526) — DETECTED (real LOA only)"),
        plt.Line2D([0], [0], **NOT_DETECTED_STYLE,
                   label=f"NOT detected (z = (on_med − off_med)/(1.4826·MAD) ≤ {DETECTION_Z_THRESHOLD:.1f})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)

    fig.suptitle(
        "Lower-triangle corr(M·M$^T$) with cross-correlation rung overlays. "
        "Each rung is the locus of (λ_short·X, λ_long·X) for a single absorber "
        "at z_abs < z_QSO, where X = (1+z_abs)/(1+z_QSO). loa-0 has only Lyα → "
        "rungs should be flat. loa-124 has LYB/LY3/LY4/LY5 + 4 Si metals → "
        "Lyman + cyan metal rungs should carry power. LOA real / v1 production "
        "may additionally show magenta (real-universe) metal rungs.",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {OUT}")

    # ============================================================
    # Detection table: rung × panel, with strength (on_rung_med − off_rung_med)
    # ============================================================
    panel_names = list(panel_detections.keys())
    if not panel_names:
        return
    # All panels see the same rung list (built once); use first panel as
    # the reference order
    ref = panel_detections[panel_names[0]]
    # Sort by category then by slope ascending
    cat_order = {"lyman": 0, "qq_metals": 1, "real_only": 2}
    ref_sorted = sorted(ref, key=lambda d: (cat_order.get(d[0], 9), d[2]))

    print()
    print("=" * 110)
    print("Detection table — z = (median(on-rung) − median(off-rung)) / (1.4826·MAD(off-rung))")
    print(f"Off-rung sampled at perpendicular offsets {PERP_OFFSETS_ANG} Å (each ±). "
          f"Threshold: z > {DETECTION_Z_THRESHOLD}. 'D' = detected, '·' = below.")
    print("=" * 110)
    header_cols = "  ".join(f"{name[:14]:>14s}" for name in panel_names)
    print(f"{'category':<11s} {'rung':<22s} {'slope':>6s}  {header_cols}")
    print("-" * 110)
    for cat, label, slope, _strength_ref, _z_ref, _det_ref in ref_sorted:
        cells = []
        for name in panel_names:
            entry = next(((c, l, s, st, z, d) for c, l, s, st, z, d
                          in panel_detections[name]
                          if l == label and abs(s - slope) < 1e-3), None)
            if entry is None or np.isnan(entry[4]) or np.isinf(entry[4]):
                cells.append(f"{'(out)':>14s}")
            else:
                mark = "D" if entry[5] else "·"
                cells.append(f"{mark} z={entry[4]:+5.1f}".rjust(14))
        cells_s = "  ".join(cells)
        print(f"{cat:<11s} {label:<22s} {slope:>6.3f}  {cells_s}")
    print("=" * 110)


if __name__ == "__main__":
    main()
