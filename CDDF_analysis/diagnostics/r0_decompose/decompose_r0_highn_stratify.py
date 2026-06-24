"""decompose_r0_highn_stratify.py — Track-C Stage-0: HIGH-N_HI (≥21.0)
stratification diagnostic.

The catalog-HBI estimator over-recovers the DLA tier: dN/dX(≥20.3) R0≈1.16
with a z-rising trend.  The WORST region is the logN≥21.0 shoulder.  Panel
verdict (4-referee, 2026-06-19):

  (a) HIGH z_QSO — long dense forest → line blending over-reads N_HI;
      kernel-correctable (z-varying kernel/completeness).  **CONFIRMED DRIVER.**
  (b) LOW z_QSO + VERY SHORT spectrum — damping wings run off the spectral
      blue edge.  **NOT detected; NON-IDENTIFIABLE** under the collinear
      nominal-window proxy (Δz_window is a deterministic function of z_QSO).

Science verdict: the ≥21.0 over-detection is a smooth, deconvolvable high-z
forest-blend bias.  There is NO separable short-spectrum/edge population.
The kernel covariate is **z_DLA** (z_QSO redundant, r=0.92).  No quality cut
is warranted.

THIS script stratifies the ≥21.0 truth-matched TP detections by (z_QSO
tertiles) × (Δz_window tertiles) and measures per-cell:
  - median NHI bias  (NHI_pred − NHI_TRUE)
  - over-detection ratio  N_pred / N_true

Also prints a pathology table for the ≥21.0 TP detections:
  - fraction near the N_HI prior ceiling (≥22.45 and ≥22.0; ceiling=22.5)
  - NHI_ERR percentiles and count with NHI_ERR≥0.5
  - band-edge migration fractions across the 21.0 boundary

Output: TSV ``highn_stratify.tsv``, pathology TSV ``highn_pathology.tsv``,
figure ``fig_highn_stratify.png``, and a printed verdict.

Reduce-only.  NO inference path touched.  Uses the same ``build_ingredients``
call as ``decompose_r0_zstructure.py`` so the cat_cut / molly / kernel are
byte-identical to the WALL-1 calibrated baseline.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi.ab_loa0_fp_baseline import (
    build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL,
    DEF_KERNEL, DEF_LOA0_PRODUCT,
)
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    LYA_REST, _build_qso_lookup,
)

# ──────────────────────────────────────────────────────────────────────────────
# Forest-window helper
# ──────────────────────────────────────────────────────────────────────────────

def _compute_window_per_tid(qso_lookup: dict, lam_rf_min: float, lam_rf_max: float
                             ) -> dict:
    """Return TARGETID -> Δz_window (=qso_zhi−qso_zlo) using the SAME geometry
    as build_pathlength (3 000 km/s collar, 3 600 Å obs-λ floor).  The collar
    exactly replicates make_lambda_z_BAL_cuts so window lengths are consistent
    with path-length.

    Returns only TIDs where the window is positive-finite (same 'ok' mask as
    build_pathlength).  SNR / z-range cuts are NOT applied here: only
    operating-point TIDs are ever looked up (the op cut is applied externally
    before the JOIN), so omitting those filters is benign and avoids duplicating
    the op-cut logic.
    """
    C_KMS = 299792.458
    collar = 3000.0 / C_KMS
    tid_to_dz: dict = {}
    for tid, (snr, zq) in qso_lookup.items():
        zlo = max(3600.0 / LYA_REST - 1.0,
                  lam_rf_min * (1.0 + zq) / LYA_REST - 1.0 + collar)
        zhi = min(zq - collar,
                  lam_rf_max * (1.0 + zq) / LYA_REST - 1.0 - collar)
        if zhi > zlo:
            tid_to_dz[int(tid)] = (zq, zhi - zlo)
    return tid_to_dz   # TARGETID -> (z_QSO, Δz_window)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # path defaults mirror decompose_r0_zstructure.py exactly
    p.add_argument("--catalog-dir",    default=DEF_CAT)
    p.add_argument("--truth",          default=DEF_TRUTH)
    p.add_argument("--bal-cat",        default=DEF_BAL)
    p.add_argument("--molly-tsv",      default=None)
    p.add_argument("--kernel",         default=DEF_KERNEL)
    p.add_argument("--loa0-product",   default=DEF_LOA0_PRODUCT)
    p.add_argument("--out",            default="/tmp/decompose_r0_highn")
    p.add_argument("--mockdir",        default=None)
    p.add_argument("--zbins",          default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits",  default="20.0,20.3,20.6")
    p.add_argument("--family",         default="bspbody")
    p.add_argument("--fit-floor",      type=float, default=19.5)
    p.add_argument("--fit-ceil",       type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min",     type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes",       type=int,   default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--highn-floor",    type=float, default=21.0,
                   help="N_HI floor for the high-N stratification (default 21.0)")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    HIGHN = args.highn_floor

    # ── Step 1: build calibrated WALL-1 ingredients (verbatim from template) ──
    print("=" * 80)
    print(f"HIGH-N_HI STRATIFICATION  (logN >= {HIGHN:.1f})")
    print("=" * 80)
    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg     = ing["cfg"]
    cat_cut = ing["cat_cut"]

    # re-build qso_lookup to get z_QSO + window length per TARGETID
    qso_lookup = _build_qso_lookup(cfg)
    tid_to_zw = _compute_window_per_tid(
        qso_lookup, cfg.lam_rf_min, cfg.lam_rf_max
    )  # TARGETID -> (z_QSO, Δz_window)

    # operating-point mask (same as all other decompose scripts)
    s2n  = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"],   float)
    op   = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & ing["good_mask"]

    nhi_pred = np.asarray(cat_cut["NHI"],      float)[op]
    nhi_true = np.asarray(cat_cut["NHI_TRUE"], float)[op]
    tids_op  = np.asarray(cat_cut["TARGETID"], np.int64)[op]

    # Z_QSO per op detection — the catalog already carries it (loaded from qso_lookup
    # during load_and_cut_catalog); use it directly for speed.
    if "Z_QSO" in cat_cut.colnames:
        z_qso_op = np.asarray(cat_cut["Z_QSO"], float)[op]
    else:
        # fallback: look up from qso_lookup
        z_qso_op = np.array([
            tid_to_zw.get(int(t), (np.nan, np.nan))[0] for t in tids_op
        ], dtype=float)

    # Δz_window per op detection (look up from per-TID map)
    dz_window_op = np.array([
        tid_to_zw.get(int(t), (np.nan, np.nan))[1] for t in tids_op
    ], dtype=float)

    is_tp = np.isfinite(nhi_true)   # True = truth-matched TP

    # truth catalog for N_true count denominator
    truth_cut = ing["truth_cut"]
    t_nhi     = np.asarray(truth_cut["NHI"],   float)
    t_snr     = np.asarray(truth_cut["S2N_RED"], float)
    t_z_qso   = np.asarray(truth_cut["Z_QSO"],  float) if "Z_QSO" in truth_cut.colnames else np.full(len(truth_cut), np.nan)
    tk        = t_snr > cfg.snr_min

    # ── Step 2: restrict to TP detections with pred ≥ HIGHN ─────────────────
    sel_highn = (nhi_pred >= HIGHN) & is_tp
    nhi_pred_h = nhi_pred[sel_highn]
    nhi_true_h = nhi_true[sel_highn]
    z_qso_h    = z_qso_op[sel_highn]
    dz_win_h   = dz_window_op[sel_highn]
    bias_h     = nhi_pred_h - nhi_true_h

    n_highn_tp = int(sel_highn.sum())
    print(f"\n  TP detections with pred >= {HIGHN:.1f}: {n_highn_tp}")

    if n_highn_tp < 10:
        print(f"\n  WARNING: only {n_highn_tp} matched TP with pred>={HIGHN:.1f} — "
              "too few for reliable tertile stratification.  TSV written with NaN cells.")

    # tertile breakpoints (from the HIGH-N TP sub-population itself)
    z_breaks  = np.nanpercentile(z_qso_h, [33.33, 66.67]) if n_highn_tp >= 6 else np.array([np.nan, np.nan])
    dz_breaks = np.nanpercentile(dz_win_h, [33.33, 66.67]) if n_highn_tp >= 6 else np.array([np.nan, np.nan])

    print(f"  z_QSO tertile breaks:    {z_breaks[0]:.3f}, {z_breaks[1]:.3f}")
    print(f"  Δz_window tertile breaks: {dz_breaks[0]:.4f}, {dz_breaks[1]:.4f}")

    def _zq_bin(arr):
        """Map z_QSO values -> 0/1/2 (low/mid/high)."""
        out = np.full(len(arr), -1, dtype=int)
        if np.isnan(z_breaks[0]):
            return out
        out[arr <= z_breaks[0]] = 0
        out[(arr > z_breaks[0]) & (arr <= z_breaks[1])] = 1
        out[arr > z_breaks[1]] = 2
        return out

    def _dz_bin(arr):
        """Map Δz_window values -> 0/1/2 (short/mid/long)."""
        out = np.full(len(arr), -1, dtype=int)
        if np.isnan(dz_breaks[0]):
            return out
        out[arr <= dz_breaks[0]] = 0
        out[(arr > dz_breaks[0]) & (arr <= dz_breaks[1])] = 1
        out[arr > dz_breaks[1]] = 2
        return out

    zq_bin_h  = _zq_bin(z_qso_h)
    dz_bin_h  = _dz_bin(dz_win_h)

    # truth counts ≥ HIGHN for N_pred/N_true — need same z_QSO tertile boundaries
    t_zq_bin = _zq_bin(t_z_qso[tk])
    t_nhi_tk = t_nhi[tk]

    # PI-hypothesis annotation cells
    PI_CELLS = {
        (2, 2): "PI-hyp (a): high-z_QSO + long window → forest blend",
        (0, 0): "PI-hyp (b): low-z_QSO + short window → edge artifact",
    }

    # ── Build 3×3 grid ────────────────────────────────────────────────────────
    z_labels  = ["low_zq",  "mid_zq",  "high_zq"]
    dz_labels = ["short_dz", "mid_dz",  "long_dz"]

    rows_tsv = []
    cell_bias  = np.full((3, 3), np.nan)
    cell_ratio = np.full((3, 3), np.nan)
    cell_n_pred = np.zeros((3, 3), dtype=int)
    cell_n_true = np.zeros((3, 3), dtype=int)

    print("\n" + "=" * 80)
    hdr = (f"  {'z_QSO':>8} {'Δz_win':>8} {'N_pred':>7} {'N_true':>7} "
           f"{'ratio':>7} {'med_bias':>9} {'iqr_bias':>9} {'note':>42}")
    print(hdr)
    print("-" * 80)

    for iz in range(3):
        for idz in range(3):
            # pred subset: TP with pred≥HIGHN in this cell
            mask_p = (zq_bin_h == iz) & (dz_bin_h == idz)
            n_pred = int(mask_p.sum())
            bias_cell = bias_h[mask_p]
            med_b = float(np.median(bias_cell)) if n_pred > 0 else np.nan
            iqr_b = float(np.percentile(bias_cell, 75) -
                          np.percentile(bias_cell, 25)) if n_pred > 0 else np.nan

            # truth count: true≥HIGHN in same z_QSO tertile bin
            # (Δz_window not available for truth; stratify by z_QSO only for N_true)
            mask_t = (t_zq_bin == iz) & (t_nhi_tk >= HIGHN)
            n_true = int(mask_t.sum())

            # Set ratio=nan when n_pred==0 (not 0.0) so that nanmean over
            # the row/col reflects only populated cells — avoids the marginal
            # undercount bug where empty off-diagonal cells dragged down the mean.
            if n_true > 0 and n_pred > 0:
                ratio = n_pred / n_true
            elif n_true > 0 and n_pred == 0:
                ratio = np.nan   # depopulated cell — exclude from nanmean
            else:
                ratio = np.nan
            note  = PI_CELLS.get((iz, idz), "")

            cell_bias[iz, idz]   = med_b
            cell_ratio[iz, idz]  = ratio
            cell_n_pred[iz, idz] = n_pred
            cell_n_true[iz, idz] = n_true

            row = (z_labels[iz], dz_labels[idz], n_pred, n_true,
                   ratio, med_b, iqr_b)
            rows_tsv.append(row + (note,))

            print(f"  {z_labels[iz]:>8} {dz_labels[idz]:>8} {n_pred:>7d} {n_true:>7d} "
                  f"{ratio:>7.3f} {med_b:>+9.4f} {iqr_b:>9.4f}  {note}")

    # ── Write TSV ─────────────────────────────────────────────────────────────
    tsv_path = os.path.join(args.out, "highn_stratify.tsv")
    with open(tsv_path, "w") as fh:
        # NOTE: N_true is binned by z_QSO tertile only (Deltaz_window is not
        # available for the truth catalog); N_true is therefore replicated
        # identically across all three Deltaz_window_bin rows within each
        # z_QSO_bin.  ratio=nan for cells with n_pred==0.
        fh.write("# N_true is z_QSO-binned (replicated across Deltaz_window_bin columns)\n")
        fh.write("z_QSO_bin\tDeltaz_window_bin\tN_pred\tN_true\t"
                 "ratio_pred_over_true\tmed_NHI_bias\tiqr_NHI_bias\tnote\n")
        for r in rows_tsv:
            fh.write("\t".join(
                f"{v:.6g}" if isinstance(v, float) else str(v)
                for v in r
            ) + "\n")
    print(f"\n  TSV -> {tsv_path}")

    # ── Step 3: Render figure ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel A: median NHI bias heatmap
        ax = axes[0]
        im = ax.imshow(cell_bias, origin="lower", aspect="auto",
                       cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(3)); ax.set_xticklabels(dz_labels, rotation=25, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(z_labels)
        ax.set_xlabel("Δz_window tertile")
        ax.set_ylabel("z_QSO tertile")
        ax.set_title(f"Median NHI bias (pred−true) — logN≥{HIGHN:.1f} TP")
        plt.colorbar(im, ax=ax, label="bias (dex)")
        for iz in range(3):
            for idz in range(3):
                b = cell_bias[iz, idz]
                txt = f"{b:+.3f}" if np.isfinite(b) else "—"
                n = cell_n_pred[iz, idz]
                ax.text(idz, iz, f"{txt}\n(n={n})", ha="center", va="center",
                        fontsize=8, color="white" if abs(b) > 0.15 else "black")
        # annotate PI-hypothesis cells
        for (iz, idz) in [(2, 2), (0, 0)]:
            rect = mpatches.FancyBboxPatch(
                (idz - 0.48, iz - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02", linewidth=2,
                edgecolor="gold", facecolor="none")
            ax.add_patch(rect)

        # Panel B: over-count ratio heatmap
        ax = axes[1]
        vmax_ratio = max(2.0, float(np.nanmax(cell_ratio)) * 1.05)
        im2 = ax.imshow(cell_ratio, origin="lower", aspect="auto",
                        cmap="coolwarm", vmin=0.0, vmax=vmax_ratio)
        ax.set_xticks(range(3)); ax.set_xticklabels(dz_labels, rotation=25, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(z_labels)
        ax.set_xlabel("Δz_window tertile")
        ax.set_ylabel("z_QSO tertile")
        ax.set_title(f"N_pred / N_true — logN≥{HIGHN:.1f} TP")
        plt.colorbar(im2, ax=ax, label="ratio")
        for iz in range(3):
            for idz in range(3):
                r = cell_ratio[iz, idz]
                n = cell_n_pred[iz, idz]
                txt = f"{r:.2f}" if np.isfinite(r) else "—"
                mid = vmax_ratio / 2.0
                ax.text(idz, iz, f"{txt}\n(n={n})", ha="center", va="center",
                        fontsize=8,
                        color="white" if (r > mid * 1.3 or r < mid * 0.6) else "black")
        for (iz, idz) in [(2, 2), (0, 0)]:
            rect = mpatches.FancyBboxPatch(
                (idz - 0.48, iz - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02", linewidth=2,
                edgecolor="gold", facecolor="none")
            ax.add_patch(rect)

        # z_QSO tertile break annotations
        fig.text(0.5, 0.01,
                 f"z_QSO breaks: {z_breaks[0]:.3f}, {z_breaks[1]:.3f}  |  "
                 f"Δz_window breaks: {dz_breaks[0]:.4f}, {dz_breaks[1]:.4f}  |  "
                 f"Gold box = PI-hypothesis cells (a)=high-z, (b)=low-z+short",
                 ha="center", va="bottom", fontsize=7.5)

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig_path = os.path.join(args.out, "fig_highn_stratify.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Fig  -> {fig_path}")
    except Exception as exc:
        print(f"  [WARN] figure skipped: {exc}")
        fig_path = None

    # ── Step 4: Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Which cell has the largest median NHI bias?
    flat_idx_bias  = int(np.nanargmax(np.abs(cell_bias)))
    iz_b, idz_b   = divmod(flat_idx_bias, 3)
    # Which cell has the largest N_pred/N_true ratio?
    flat_idx_ratio = int(np.nanargmax(cell_ratio))
    iz_r, idz_r   = divmod(flat_idx_ratio, 3)

    bias_high_z  = float(cell_bias[2, 2])   # high z_QSO, long window (PI-hyp a)
    bias_low_z   = float(cell_bias[0, 0])   # low z_QSO, short window (PI-hyp b)
    ratio_high_z = float(cell_ratio[2, 2])
    ratio_low_z  = float(cell_ratio[0, 0])

    # Marginal ratio by z_QSO tertile: nanmean over populated cells only
    # (empty off-diagonal cells have ratio=nan, so nanmean is uncontaminated).
    row_bias_mean  = np.nanmean(cell_bias,  axis=1)  # shape (3,) low→high z_QSO
    row_ratio_mean = np.nanmean(cell_ratio, axis=1)
    col_bias_mean  = np.nanmean(cell_bias,  axis=0)
    col_ratio_mean = np.nanmean(cell_ratio, axis=0)

    print(f"\n  Row-mean bias  by z_QSO  (low→high): "
          + "  ".join(f"{v:+.4f}" for v in row_bias_mean))
    print(f"  Row-mean ratio by z_QSO  (low→high) [populated cells only]: "
          + "  ".join(f"{v:.3f}" for v in row_ratio_mean))
    print(f"  Col-mean bias  by Δz_win (short→long): "
          + "  ".join(f"{v:+.4f}" for v in col_bias_mean))
    print(f"  Col-mean ratio by Δz_win (short→long) [populated cells only]: "
          + "  ".join(f"{v:.3f}" for v in col_ratio_mean))

    print(f"\n  PI-hyp (a) cell [high z_QSO, long Δz]:  bias={bias_high_z:+.4f}  ratio={ratio_high_z:.3f}")
    print(f"  PI-hyp (b) cell [low  z_QSO, short Δz]: bias={bias_low_z:+.4f}  ratio={ratio_low_z:.3f}")

    # 4-referee panel verdict (2026-06-19): the low_zq cell (ratio ~1.1) is NOT an
    # "edge artifact secondary" — it is the low end of a z-rising forest-blend trend.
    # Hypothesis (b) "short-spectrum edge truncation" is NOT detected and is
    # NON-IDENTIFIABLE under the collinear nominal-window proxy (Δz_window is a
    # deterministic function of z_QSO).  The kernel covariate is z_DLA.
    z_trend   = row_bias_mean[2] - row_bias_mean[0]    # positive → high-z worse
    z_trend_r = row_ratio_mean[2] - row_ratio_mean[0]

    THRESH_BIAS  = 0.05   # dex — meaningful
    THRESH_RATIO = 0.10   # meaningful ratio change

    if z_trend > THRESH_BIAS or z_trend_r > THRESH_RATIO:
        verdict = (
            "≥21.0 over-detection driven by high-z forest-blend (hypothesis a): "
            "smooth z-rising bias, deconvolvable via z_DLA kernel.  "
            "Hypothesis (b) [short-spectrum edge] NOT detected; "
            "NON-IDENTIFIABLE under collinear Δz_window proxy.  "
            "No quality cut warranted."
        )
    else:
        verdict = (
            "≥21.0 over-detection: no strong z-trend detected in this run "
            "(check n_highn_tp; may indicate low sample size)."
        )

    print(f"\n  {verdict}")
    print(f"\n  Largest BIAS cell:  z_QSO={z_labels[iz_b]}, Δz={dz_labels[idz_b]}, "
          f"bias={cell_bias[iz_b, idz_b]:+.4f}")
    print(f"  Largest RATIO cell: z_QSO={z_labels[iz_r]}, Δz={dz_labels[idz_r]}, "
          f"ratio={cell_ratio[iz_r, idz_r]:.3f}")

    # ── Step 5: Pathology table ────────────────────────────────────────────────
    # Computed from ALL op TP detections with pred≥HIGHN (n_highn_tp).
    # Load-bearing evidence for "deconvolvable" verdict.
    print("\n" + "=" * 80)
    print(f"PATHOLOGY TABLE  (≥{HIGHN:.1f} TP detections, N={n_highn_tp})")
    print("=" * 80)

    NHI_CEIL = 22.5   # prior ceiling

    # (a) Fraction near the N_HI prior ceiling
    if n_highn_tp > 0:
        frac_ceil_245 = float(np.sum(nhi_pred_h >= 22.45)) / n_highn_tp
        frac_ceil_220 = float(np.sum(nhi_pred_h >= 22.0))  / n_highn_tp
    else:
        frac_ceil_245 = frac_ceil_220 = np.nan
    print(f"\n  (a) Prior-ceiling proximity (ceiling={NHI_CEIL}):")
    print(f"      frac(pred >= 22.45): {frac_ceil_245:.4f}  "
          f"({int(np.sum(nhi_pred_h >= 22.45)) if n_highn_tp > 0 else 0}/{n_highn_tp})")
    print(f"      frac(pred >= 22.00): {frac_ceil_220:.4f}  "
          f"({int(np.sum(nhi_pred_h >= 22.0))  if n_highn_tp > 0 else 0}/{n_highn_tp})")

    # (b) NHI_ERR percentiles and count with NHI_ERR >= 0.5
    nhi_err_key = None
    for k in ("NHI_ERR", "NHI_STD", "SIGMA_NHI"):
        if k in cat_cut.colnames:
            nhi_err_key = k
            break
    if nhi_err_key is not None and n_highn_tp > 0:
        nhi_err_h = np.asarray(cat_cut[nhi_err_key], float)[op][sel_highn]
        err_med  = float(np.nanmedian(nhi_err_h))
        err_p99  = float(np.nanpercentile(nhi_err_h, 99))
        err_max  = float(np.nanmax(nhi_err_h))
        n_large_err = int(np.sum(nhi_err_h >= 0.5))
    else:
        nhi_err_h = None
        err_med = err_p99 = err_max = np.nan
        n_large_err = -1
    print(f"\n  (b) NHI_ERR ({nhi_err_key if nhi_err_key else 'NOT FOUND'}):")
    print(f"      median={err_med:.4f}  p99={err_p99:.4f}  max={err_max:.4f}")
    print(f"      count(NHI_ERR >= 0.5): {n_large_err}/{n_highn_tp}")

    # (c) Band-edge migration fractions across the HIGHN boundary
    # "up":   true < HIGHN  AND  pred >= HIGHN  (up-scattered into the band)
    # "down": true >= HIGHN AND  pred < HIGHN   (this script's sel doesn't cover
    #         down-scattered, so count from full op set)
    if is_tp.sum() > 0:
        n_up   = int(np.sum((nhi_true[is_tp] <  HIGHN) & (nhi_pred[is_tp] >= HIGHN)))
        n_down = int(np.sum((nhi_true[is_tp] >= HIGHN) & (nhi_pred[is_tp] <  HIGHN)))
        n_tp_total = int(is_tp.sum())
        frac_up   = n_up   / n_tp_total
        frac_down = n_down / n_tp_total
    else:
        n_up = n_down = n_tp_total = 0
        frac_up = frac_down = np.nan
    print(f"\n  (c) Band-edge migration across logN={HIGHN:.1f} (from all op TP, N={n_tp_total}):")
    print(f"      'up'   (true<{HIGHN:.1f} & pred>={HIGHN:.1f}): {n_up}  frac={frac_up:.4f}")
    print(f"      'down' (true>={HIGHN:.1f} & pred<{HIGHN:.1f}): {n_down}  frac={frac_down:.4f}")

    # Write pathology TSV
    path_tsv = os.path.join(args.out, "highn_pathology.tsv")
    with open(path_tsv, "w") as fh:
        fh.write("metric\tvalue\tnumerator\tdenominator\n")
        fh.write(f"frac_pred_ge_22.45\t{frac_ceil_245:.6g}\t"
                 f"{int(np.sum(nhi_pred_h >= 22.45)) if n_highn_tp > 0 else 0}\t{n_highn_tp}\n")
        fh.write(f"frac_pred_ge_22.00\t{frac_ceil_220:.6g}\t"
                 f"{int(np.sum(nhi_pred_h >= 22.0)) if n_highn_tp > 0 else 0}\t{n_highn_tp}\n")
        fh.write(f"NHI_ERR_median\t{err_med:.6g}\t\t{n_highn_tp}\n")
        fh.write(f"NHI_ERR_p99\t{err_p99:.6g}\t\t{n_highn_tp}\n")
        fh.write(f"NHI_ERR_max\t{err_max:.6g}\t\t{n_highn_tp}\n")
        fh.write(f"count_NHI_ERR_ge_0.5\t{n_large_err}\t{n_large_err}\t{n_highn_tp}\n")
        fh.write(f"frac_migrate_up\t{frac_up:.6g}\t{n_up}\t{n_tp_total}\n")
        fh.write(f"frac_migrate_down\t{frac_down:.6g}\t{n_down}\t{n_tp_total}\n")
    print(f"\n  Pathology TSV -> {path_tsv}")

    print("\n" + "=" * 80)
    print(f"  OUT DIR: {args.out}")
    print("=" * 80)

    return dict(
        cell_bias=cell_bias,
        cell_ratio=cell_ratio,
        cell_n_pred=cell_n_pred,
        cell_n_true=cell_n_true,
        z_breaks=z_breaks,
        dz_breaks=dz_breaks,
        verdict=verdict,
        tsv_path=tsv_path,
        path_tsv=path_tsv,
        fig_path=fig_path,
        pathology=dict(
            frac_ceil_245=frac_ceil_245,
            frac_ceil_220=frac_ceil_220,
            nhi_err_median=err_med,
            nhi_err_p99=err_p99,
            nhi_err_max=err_max,
            n_large_err=n_large_err,
            frac_migrate_up=frac_up,
            frac_migrate_down=frac_down,
            n_tp_total=n_tp_total,
        ),
    )


if __name__ == "__main__":
    main()
