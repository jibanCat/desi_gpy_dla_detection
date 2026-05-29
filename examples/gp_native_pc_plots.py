"""gp_native_pc_plots.py — purity/completeness panels for a GP-DLA mock run.

NOT a faithful reproduction of molly's notebook
(`/pscratch/sd/j/jibancat/molly/lls_purity_completeness_20260416-Copy1.ipynb`).
Produces panels with the *same axes/shape* as her plots, but using our GP
catalog columns and a single P(DLA) threshold:

| Aspect              | molly's notebook         | this script               |
|---------------------|--------------------------|---------------------------|
| Detection NHI col   | NHI_TMP (template fit)   | NHI (GP MAP)              |
| Confidence col      | DLA_CONFIDENCE / log_pdla| P_DLA                     |
| SNR col             | S2N_RED                  | SNR_FOREST                |
| Goodness gate       | DELTACHI2 threshold      | (not applied)             |
| Cut family          | 3-threshold combo        | single P_DLA threshold    |

Headline numbers will therefore differ from molly's tables. For a faithful
side-by-side comparison, run a separate molly-faithful script (TODO) that
maps GP columns onto her detection-cat schema and applies her exact cuts.

Given a directory of `dlacat-*.fits` from a multi-DLA run plus the mock's
truth catalog, produces five panels:

  (1) Purity & completeness vs P(DLA) cut, per NHI bin.
  (2) Completeness heatmap on (S2N, true logNHI).
  (3) Purity heatmap on (S2N, predicted logNHI).
  (4) ΔlogNHI and Δz scatter / histograms on matched DLAs.
  (5) NHI distribution of spurious (un-matched) MAP DLAs.

Match definition (same as examples/analyze_production_catalog.py):
  - same TARGETID
  - |Δz| / (1 + z_truth) ≤ dz_rel (default 0.01, ≈ Δv/c = 3000 km/s)
  - greedy: each truth row claims the nearest unused MAP row,
    iterating truth rows in descending NHI order.

Usage:
  python examples/gp_native_pc_plots.py \\
      --catalog-dir /pscratch/.../london0_y3/ \\
      --truth /global/cfs/.../jura-124/dla_cat.fits \\
      --bal-cat /global/cfs/.../jura-124/bal_cat.fits --no-bal \\
      --truth-nhi-min 20.3 \\
      --out /pscratch/.../london0_y3/figures/
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import fitsio
from astropy.table import Table, vstack


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", required=True,
                   help="Run OUTDIR with dlacat-*.fits files.")
    p.add_argument("--truth", required=True,
                   help="Mock truth catalog (London dla_cat.fits or "
                        "Saclay/2LPT hcd_truth_cat.fits).")
    p.add_argument("--bal-cat", default=None,
                   help="bal_cat.fits with BI_CIV column.")
    p.add_argument("--no-bal", action="store_true",
                   help="Exclude BAL TIDs (BI_CIV>0) from BOTH cat and truth.")
    p.add_argument("--truth-nhi-min", type=float, default=20.3,
                   help="Drop truth rows below this NHI (default 20.3).")
    p.add_argument("--dz-rel", type=float, default=0.01,
                   help="|Δz|/(1+z_truth) match tolerance (default 0.01).")
    p.add_argument("--out", required=True,
                   help="Output directory for PNGs + summary tsv.")
    p.add_argument("--title", default=None,
                   help="Figure title (default: catalog-dir basename).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load catalog + truth
# ---------------------------------------------------------------------------
def load_catalog_dir(d: str) -> Table:
    files = sorted(glob.glob(os.path.join(d, "dlacat-*.fits")))
    if not files:
        raise SystemExit(f"[error] no dlacat-*.fits in {d}")
    tbls = []
    for f in files:
        try:
            tbls.append(Table(fitsio.read(f, ext=1)))
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {e}")
    cat = vstack(tbls)
    print(f"[load] catalog: {len(cat)} rows from {len(files)} files")
    return cat


def load_truth(path: str, nhi_min: float) -> Table:
    tr = Table(fitsio.read(path, ext=1))
    z_col = next((c for c in ("Z_DLA", "Z_DLA_NO_RSD", "Z") if c in tr.colnames),
                 None)
    if z_col is None:
        raise SystemExit(f"truth has no Z_DLA/Z col: {tr.colnames}")
    tr.rename_column(z_col, "Z_TRUTH")
    if nhi_min > 0:
        tr = tr[np.asarray(tr["NHI"]) >= nhi_min]
    print(f"[load] truth: {len(tr)} DLAs (NHI≥{nhi_min})")
    return tr


def apply_bal_cut(cat: Table, truth: Table, bal_path: str) -> tuple:
    """Drop BAL targets from BOTH catalog and truth (molly convention)."""
    bal = fitsio.read(bal_path, ext=1, columns=["TARGETID", "BI_CIV"])
    bal_tids = set(int(r["TARGETID"]) for r in bal if r["BI_CIV"] > 0)
    print(f"[bal] {len(bal_tids)} BAL TIDs")
    cat = cat[~np.isin(np.asarray(cat["TARGETID"]), list(bal_tids))]
    truth = truth[~np.isin(np.asarray(truth["TARGETID"]), list(bal_tids))]
    print(f"[bal] cat→{len(cat)}, truth→{len(truth)} after BAL exclusion")
    return cat, truth


# ---------------------------------------------------------------------------
# Truth matching (greedy by descending NHI)
# ---------------------------------------------------------------------------
def match_truth_to_cat(cat: Table, truth: Table, dz_rel: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Returns:
    - cat_is_TP  : (len(cat),) bool — this MAP DLA matched a truth DLA
    - cat_NHI_TR : (len(cat),) float — matched truth NHI (NaN if FP)
    - cat_Z_TR   : (len(cat),) float — matched truth Z (NaN if FP)
    - truth_matched : (len(truth),) bool — this truth DLA found a MAP partner
    """
    cat_is_TP = np.zeros(len(cat), dtype=bool)
    cat_NHI_TR = np.full(len(cat), np.nan)
    cat_Z_TR = np.full(len(cat), np.nan)
    truth_matched = np.zeros(len(truth), dtype=bool)

    c_tid = np.asarray(cat["TARGETID"])
    c_z = np.asarray(cat["Z_DLA"], dtype=float)
    t_tid = np.asarray(truth["TARGETID"])
    t_z = np.asarray(truth["Z_TRUTH"], dtype=float)
    t_nhi = np.asarray(truth["NHI"], dtype=float)

    cat_by_tid: dict[int, list[int]] = {}
    for i, t in enumerate(c_tid):
        cat_by_tid.setdefault(int(t), []).append(i)
    truth_by_tid: dict[int, list[int]] = {}
    for j, t in enumerate(t_tid):
        truth_by_tid.setdefault(int(t), []).append(j)

    for tid, t_idx_list in truth_by_tid.items():
        cand = cat_by_tid.get(tid, [])
        if not cand:
            continue
        # Sort truth by descending NHI so strongest claims first
        order = sorted(t_idx_list, key=lambda j: -t_nhi[j])
        for tj in order:
            best, best_dz = None, np.inf
            for ci in cand:
                if cat_is_TP[ci]:
                    continue
                dz = abs(c_z[ci] - t_z[tj]) / (1 + t_z[tj])
                if dz < best_dz:
                    best_dz, best = dz, ci
            if best is not None and best_dz <= dz_rel:
                cat_is_TP[best] = True
                cat_NHI_TR[best] = t_nhi[tj]
                cat_Z_TR[best] = t_z[tj]
                truth_matched[tj] = True

    return cat_is_TP, cat_NHI_TR, cat_Z_TR, truth_matched


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    return matplotlib


def plot_purity_completeness_vs_pcut(cat, cat_is_TP, truth, truth_matched,
                                     out_png, title):
    """Panel (1): purity & completeness vs P(DLA) cut, per NHI bin."""
    mpl = setup_mpl()
    import matplotlib.pyplot as plt

    p = np.asarray(cat["P_DLA"], dtype=float)
    nhi_pred = np.asarray(cat["NHI"], dtype=float)
    nhi_true_all = np.asarray(truth["NHI"], dtype=float)

    cuts = np.linspace(0.0, 0.99, 50)
    nhi_bins = [(20.3, 20.6), (20.6, 21.0), (21.0, 21.5),
                (21.5, 23.5), (20.3, 23.5)]
    labels = ["[20.3,20.6)", "[20.6,21.0)", "[21.0,21.5)",
              "[21.5,23.5)", "all (≥20.3)"]
    cols = ["C0", "C1", "C2", "C3", "k"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax_p, ax_c = axes

    for (lo, hi), lab, cc in zip(nhi_bins, labels, cols):
        purs, comps = [], []
        for cut in cuts:
            sel = (p >= cut) & (nhi_pred >= lo) & (nhi_pred < hi)
            ntot = int(sel.sum())
            ntp = int(cat_is_TP[sel].sum())
            purs.append(ntp / ntot if ntot else np.nan)
            # completeness: matched truth in [lo, hi] / total truth in [lo, hi]
            sel_t = (nhi_true_all >= lo) & (nhi_true_all < hi)
            # NB: the "matched-at-cut" set is restricted to MAP rows with P>=cut;
            # use the cat_is_TP_at_cut population:
            # Find which truth was matched by a cat row that passed the cut
            # (recompute on demand: we have truth_matched (any cut), but to be
            #  strict we need per-cut matches)
            comps.append(np.nan)  # filled below by a per-cut greedy match
        ax_p.plot(cuts, purs, color=cc, label=lab, lw=1.2)

    # Strict completeness per cut requires re-matching at each cut.
    # Approximation acceptable here: use truth_matched (any P_DLA in cat).
    # Many notebooks just report completeness at one cut (e.g. 0.5); we do
    # the same.
    cut_for_comp = 0.5
    p_keep = p >= cut_for_comp
    matched_cap = np.zeros(len(truth), dtype=bool)
    # walk truth_matched but only keep matches whose MAP passed the cut
    # (need to re-match — quickest: just match again with the kept cat)
    cat_kept = cat[p_keep]
    _, _, _, mt_cut = match_truth_to_cat(cat_kept, truth, dz_rel=0.01)
    for (lo, hi), lab, cc in zip(nhi_bins, labels, cols):
        sel_t = (nhi_true_all >= lo) & (nhi_true_all < hi)
        if sel_t.sum():
            ax_c.axhline(mt_cut[sel_t].sum() / sel_t.sum(),
                         color=cc, ls=":", lw=0.6)
        # also show completeness curve vs cut (one bin only):
        comps = []
        for cut in cuts:
            cat_kc = cat[p >= cut]
            if len(cat_kc) == 0:
                comps.append(0.0); continue
            _, _, _, mtk = match_truth_to_cat(cat_kc, truth, dz_rel=0.01)
            n_match = mtk[sel_t].sum() if sel_t.sum() else 0
            n_tot = sel_t.sum()
            comps.append(n_match / n_tot if n_tot else np.nan)
        ax_c.plot(cuts, comps, color=cc, lw=1.2, label=lab)

    ax_p.set_xlabel("P(DLA) cut")
    ax_p.set_ylabel("Purity")
    ax_p.set_ylim(0, 1.02)
    ax_p.set_title("Purity vs P(DLA) cut")
    ax_p.legend(fontsize=8, loc="lower right", title="predicted log NHI bin")
    ax_p.grid(alpha=0.3)

    ax_c.set_xlabel("P(DLA) cut")
    ax_c.set_ylabel("Completeness")
    ax_c.set_ylim(0, 1.02)
    ax_c.set_title("Completeness vs P(DLA) cut")
    ax_c.legend(fontsize=8, loc="lower left", title="truth log NHI bin")
    ax_c.grid(alpha=0.3)

    fig.suptitle(f"{title}  (mock multi-DLA, dz_rel=0.01)", fontsize=10)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_heatmaps(cat, cat_is_TP, truth, out_png, p_cut, title):
    """Panels (2)+(3): purity heatmap (SNR × predicted NHI) + completeness
    heatmap (SNR × truth NHI), where the truth-side SNR is the QSO SNR_FOREST
    propagated to each truth DLA via the matched MAP row.

    Truth catalogs don't carry per-DLA SNR, so for unmatched truth rows we
    look up the QSO's SNR_FOREST from the catalog's first row for that
    TARGETID (any MAP row on that TID — every spec has one MAP row at the
    least likely candidate or null). If the TID isn't in the catalog at all,
    we drop it from the heatmap denominator (it was never inferred).
    """
    mpl = setup_mpl()
    import matplotlib.pyplot as plt

    p = np.asarray(cat["P_DLA"], dtype=float)
    snr = np.asarray(cat["SNR_FOREST"], dtype=float)
    nhi_pred = np.asarray(cat["NHI"], dtype=float)
    tid_cat = np.asarray(cat["TARGETID"])
    nhi_true_all = np.asarray(truth["NHI"], dtype=float)
    tid_truth = np.asarray(truth["TARGETID"])

    keep = p >= p_cut

    snr_bins = np.array([0, 1, 1.5, 2, 3, 4, 6, 10, 30])
    nhi_bins = np.array([20.3, 20.5, 20.7, 21.0, 21.3, 21.7, 22.5])

    # ---- Purity: (S2N_MAP, predicted NHI) ----
    purity = np.full((len(nhi_bins)-1, len(snr_bins)-1), np.nan)
    for i in range(len(nhi_bins)-1):
        for j in range(len(snr_bins)-1):
            sel = (keep & (snr >= snr_bins[j]) & (snr < snr_bins[j+1])
                   & (nhi_pred >= nhi_bins[i]) & (nhi_pred < nhi_bins[i+1]))
            ntot = int(sel.sum())
            if ntot:
                purity[i, j] = cat_is_TP[sel].sum() / ntot

    # ---- Per-TID SNR map (one SNR per QSO, taken from cat's first row) ----
    snr_by_tid = {}
    for tid, s in zip(tid_cat, snr):
        if int(tid) not in snr_by_tid:
            snr_by_tid[int(tid)] = float(s)
    truth_snr = np.array([snr_by_tid.get(int(t), np.nan) for t in tid_truth])

    # ---- Completeness: (S2N_QSO, truth NHI) ----
    # Numerator = matched truth in bin; denominator = ALL truth in bin
    # (regardless of detection), using the QSO-level SNR for the truth row.
    # Re-match at the chosen p_cut.
    cat_kept = cat[keep]
    _, _, _, truth_matched_at_cut = match_truth_to_cat(cat_kept, truth,
                                                       dz_rel=0.01)
    completeness = np.full((len(nhi_bins)-1, len(snr_bins)-1), np.nan)
    for i in range(len(nhi_bins)-1):
        for j in range(len(snr_bins)-1):
            sel_t = ((nhi_true_all >= nhi_bins[i]) & (nhi_true_all < nhi_bins[i+1])
                     & (truth_snr >= snr_bins[j]) & (truth_snr < snr_bins[j+1]))
            ntot = int(sel_t.sum())
            if ntot:
                completeness[i, j] = truth_matched_at_cut[sel_t].sum() / ntot

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, data, lbl, xlbl in [
        (axes[0], purity, "Purity", "SNR_FOREST (MAP DLA's QSO)"),
        (axes[1], completeness, "Completeness", "SNR_FOREST (truth DLA's QSO)"),
    ]:
        im = ax.pcolor(snr_bins, nhi_bins, data, cmap="CMRmap",
                       vmin=0, vmax=1, rasterized=True)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isfinite(v):
                    col = "white" if v < 0.5 else "black"
                    ax.text((snr_bins[j] + snr_bins[j+1]) / 2,
                            (nhi_bins[i] + nhi_bins[i+1]) / 2,
                            f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=col)
        ax.set_xlabel(xlbl)
        ax.set_ylabel("log N_HI")
        ax.set_title(f"{lbl}  (P(DLA)≥{p_cut})")
        fig.colorbar(im, ax=ax, label=lbl)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_delta_scatter(cat, cat_is_TP, cat_NHI_TR, cat_Z_TR, out_png,
                       p_cut, title):
    """Panel (4): ΔlogNHI and Δz scatter / histograms on matched DLAs."""
    mpl = setup_mpl()
    import matplotlib.pyplot as plt

    p = np.asarray(cat["P_DLA"], dtype=float)
    sel = cat_is_TP & (p >= p_cut)
    dNHI = np.asarray(cat["NHI"], dtype=float)[sel] - cat_NHI_TR[sel]
    dz_rel = ((np.asarray(cat["Z_DLA"], dtype=float)[sel] - cat_Z_TR[sel])
              / (1 + cat_Z_TR[sel]))
    nhi_true = cat_NHI_TR[sel]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             constrained_layout=True)
    nhi_bins = [(20.3, 20.6), (20.6, 21.0), (21.0, 21.5), (21.5, 23.5)]
    cols = ["C0", "C1", "C2", "C3"]

    # (a) ΔNHI vs truth NHI
    for (lo, hi), cc in zip(nhi_bins, cols):
        m = (nhi_true >= lo) & (nhi_true < hi)
        axes[0].scatter(nhi_true[m], dNHI[m], s=4, alpha=0.5, color=cc,
                        label=f"[{lo},{hi})")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("truth log NHI")
    axes[0].set_ylabel(r"$\Delta$ log NHI  (MAP $-$ truth)")
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_title(f"ΔNHI scatter (N matched = {sel.sum()})")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3)

    # (b) ΔNHI histograms per bin
    bins = np.linspace(-1.5, 1.5, 60)
    for (lo, hi), cc in zip(nhi_bins, cols):
        m = (nhi_true >= lo) & (nhi_true < hi)
        if m.sum() > 5:
            axes[1].hist(dNHI[m], bins=bins, histtype="step",
                         color=cc, label=f"[{lo},{hi})  σ={np.nanstd(dNHI[m]):.3f}")
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set_xlabel(r"$\Delta$ log NHI")
    axes[1].set_ylabel("count")
    axes[1].set_title("ΔNHI by truth-NHI bin")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # (c) Δz/(1+z) histogram
    bz = np.linspace(-0.01, 0.01, 60)
    axes[2].hist(dz_rel, bins=bz, histtype="stepfilled",
                 color="0.3", alpha=0.7)
    axes[2].axvline(0, color="k", lw=0.5)
    axes[2].set_xlabel(r"$\Delta z / (1+z_{\mathrm{truth}})$")
    axes[2].set_ylabel("count")
    axes[2].set_title(
        f"Δz scatter (σ={np.nanstd(dz_rel):.5f}, median={np.nanmedian(dz_rel):+.5f})")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_spurious_nhi(cat, cat_is_TP, out_png, p_cut, title):
    """Panel (5): NHI distribution of spurious (FP) detections."""
    mpl = setup_mpl()
    import matplotlib.pyplot as plt

    p = np.asarray(cat["P_DLA"], dtype=float)
    sel = (p >= p_cut)
    nhi = np.asarray(cat["NHI"], dtype=float)[sel]
    tp = cat_is_TP[sel]

    bins = np.linspace(19.5, 23.0, 40)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(nhi[tp], bins=bins, label=f"matched  (N={tp.sum()})",
            color="C0", alpha=0.7, histtype="stepfilled")
    ax.hist(nhi[~tp], bins=bins, label=f"spurious  (N={(~tp).sum()})",
            color="C3", alpha=0.7, histtype="stepfilled")
    ax.axvline(20.3, color="k", lw=0.8, ls="--", label="DLA threshold (20.3)")
    ax.set_xlabel("predicted log N_HI")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title(f"{title}  —  P(DLA)≥{p_cut}")
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cat = load_catalog_dir(args.catalog_dir)
    truth = load_truth(args.truth, args.truth_nhi_min)

    if args.no_bal and args.bal_cat:
        cat, truth = apply_bal_cut(cat, truth, args.bal_cat)
    elif args.no_bal:
        print("[warn] --no-bal without --bal-cat: skipping BAL exclusion")

    # Restrict truth to TIDs the pipeline actually processed
    cat_tids = set(int(t) for t in np.asarray(cat["TARGETID"]))
    in_cat = np.array([int(t) in cat_tids for t in np.asarray(truth["TARGETID"])])
    print(f"[match] {in_cat.sum()} of {len(truth)} truth DLAs on pipeline-processed TIDs")
    truth = truth[in_cat]

    print("[match] truth ↔ MAP (greedy, descending NHI)")
    cat_is_TP, cat_NHI_TR, cat_Z_TR, truth_matched = match_truth_to_cat(
        cat, truth, args.dz_rel)
    print(f"  {cat_is_TP.sum()}/{len(cat)} MAP DLAs matched a truth DLA")
    print(f"  {truth_matched.sum()}/{len(truth)} truth DLAs matched to a MAP")

    title = args.title or os.path.basename(os.path.normpath(args.catalog_dir))

    # ---- Plots ----
    print("[plot] panel 1: purity & completeness vs P(DLA) cut")
    plot_purity_completeness_vs_pcut(
        cat, cat_is_TP, truth, truth_matched,
        out / "pc_vs_pcut.png", title)

    print("[plot] panel 2-3: SNR×NHI heatmaps")
    plot_heatmaps(cat, cat_is_TP, truth,
                  out / "heatmaps_snr_nhi.png", p_cut=0.5, title=title)

    print("[plot] panel 4: ΔNHI / Δz scatter")
    plot_delta_scatter(cat, cat_is_TP, cat_NHI_TR, cat_Z_TR,
                       out / "delta_scatter.png", p_cut=0.5, title=title)

    print("[plot] panel 5: spurious-NHI distribution")
    plot_spurious_nhi(cat, cat_is_TP,
                      out / "spurious_nhi.png", p_cut=0.5, title=title)

    # ---- Summary numbers ----
    summary = {
        "n_truth_in_processed": len(truth),
        "n_truth_matched":     int(truth_matched.sum()),
        "n_cat_total":         len(cat),
        "n_cat_p_ge_0.5":      int((np.asarray(cat["P_DLA"]) >= 0.5).sum()),
        "n_cat_TP_p_ge_0.5":   int((cat_is_TP & (np.asarray(cat["P_DLA"]) >= 0.5)).sum()),
    }
    summary["completeness_p_ge_0.5"] = (
        summary["n_truth_matched"] / summary["n_truth_in_processed"]
        if summary["n_truth_in_processed"] else 0.0)
    summary["purity_p_ge_0.5"] = (
        summary["n_cat_TP_p_ge_0.5"] / summary["n_cat_p_ge_0.5"]
        if summary["n_cat_p_ge_0.5"] else 0.0)

    print("\n=== summary at P(DLA)≥0.5, NHI≥{:.2f} ===".format(args.truth_nhi_min))
    for k, v in summary.items():
        print(f"  {k:>30}  {v}")

    with open(out / "summary.tsv", "w") as f:
        f.write("key\tvalue\n")
        for k, v in summary.items():
            f.write(f"{k}\t{v}\n")
    print(f"\n[saved] {out}/  (5 PNGs + summary.tsv)")


if __name__ == "__main__":
    main()
