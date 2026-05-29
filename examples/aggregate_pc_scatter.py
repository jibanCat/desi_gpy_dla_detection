"""aggregate_pc_scatter.py — purity-vs-completeness scatter for many variants.

Modeled on Molly Wolfson's `read_in_each_up_matched_new_cats_2509.ipynb` cell
that plots P-vs-C for each method (CNN, GP, Template) sweeping the confidence
cut as a color axis, with the canonical operating point shown as a star.

Here we instead sweep one method (GP) across many *configuration variants*
and overlay them on a single scatter, sweeping `P_DLA ∈ {1-10**lp for lp in
log_pdla_range}` for each variant. Marker shape = variant; color = log_pdla.

Re-uses the cut/match helpers from `molly_faithful_pc_plots.py` and
`gp_native_pc_plots.py` so the eval recipe matches `pc_snr2_pdla99.md`
exactly at the headline P_DLA=0.99 row.

USAGE
-----
Add variant rows to VARIANTS below or pass `--variant LABEL=PATH` repeatedly:

    python examples/aggregate_pc_scatter.py \\
        --truth /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits \\
        --bal-cat /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits \\
        --no-bal \\
        --snr-min 2.0 --nhi-min 20.3 --truth-nhi-min 20.3 \\
        --lam-rf-min 911 --lam-rf-max 1216 \\
        --lyb-veto \\
        --out /pscratch/sd/j/jibancat/prod533_5k_20260511/figures/pc_scatter/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Re-use upstream helpers
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from gp_native_pc_plots import (  # noqa: E402
    load_catalog_dir, apply_bal_cut, match_truth_to_cat,
)
from molly_faithful_pc_plots import (  # noqa: E402
    build_per_qso_snr, load_truth_molly, make_lambda_z_BAL_cuts,
    purity_min, completeness_min,
)


# ---------------------------------------------------------------------------
# Variant registry — edit here to add/remove variants.
# label: short tag for the legend
# path:  catalog OUTDIR (containing dlacat-*.fits + processed/)
# marker: matplotlib marker
# color:  base color for this variant
# ---------------------------------------------------------------------------
BASE = "/pscratch/sd/j/jibancat/prod533_5k_20260511"
DEFAULT_VARIANTS = [
    # ( label, path, marker, color )
    ("baseline FILTER=1",        f"{BASE}/london_v3_loa124_pw14_tau_eb",                       "o", "C0"),
    ("early_stop_A",             f"{BASE}/london_v3_loa124_early_stop_A",                      "v", "C8"),
    ("early_stop_D",             f"{BASE}/london_v3_loa124_early_stop_D",                      "^", "C9"),
    ("NFL=31",                   f"{BASE}/london_v3_loa124_pw14_tau_eb_nfl31",                 "D", "gray"),
    ("FILTER=0",                 f"{BASE}/london_v3_loa124_pw14_tau_eb_filter0",               "s", "C3"),
    ("cellA [19,23] md3",        f"{BASE}/joint_dla_subdla_sweep/cellA_md3_nhi19to23",          "P", "C4"),
    ("cellB [19,23] md4",        f"{BASE}/joint_dla_subdla_sweep/cellB_md4_nhi19to23",          "X", "C5"),
    ("cellC [17.2,22] md3",      f"{BASE}/joint_dla_subdla_sweep/cellC_md3_nhi172to22",         "*", "C2"),
    # Knob 2x2 (filled in by the launcher; safe to include before they exist —
    # the loader skips missing dirs with a warning).
    ("k1=10k k4=off",            f"{BASE}/filter1_knob_2x2/k1_10k_k4_off",                     "p", "C6"),
    ("k1=5k k4=on",              f"{BASE}/filter1_knob_2x2/k1_5k_k4_on",                       "H", "C7"),
    ("k1=10k k4=on",             f"{BASE}/filter1_knob_2x2/k1_10k_k4_on",                      "d", "C1"),
]

# P_DLA cuts to sweep — Molly's notebook uses log_pdla ∈ {-1, ..., -8}
LOG_PDLA_SWEEP = np.array([-1., -1.5, -2., -2.5, -3., -4., -5., -6., -7., -8.])


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True)
    p.add_argument("--bal-cat", default=None)
    p.add_argument("--no-bal", action="store_true")
    p.add_argument("--truth-nhi-min", type=float, default=20.3)
    p.add_argument("--nhi-min", type=float, default=20.3,
                   help="Predicted-NHI floor (default 20.3).")
    p.add_argument("--snr-min", type=float, default=2.0,
                   help="SNR_RED floor (default 2.0).")
    p.add_argument("--dz-rel", type=float, default=0.01)
    p.add_argument("--z-qso-min", type=float, default=2.0)
    p.add_argument("--z-qso-max", type=float, default=4.25)
    p.add_argument("--lam-rf-min", type=float, default=911.)
    p.add_argument("--lam-rf-max", type=float, default=1216.)
    p.add_argument("--lyb-veto", action="store_true",
                   help="Apply postprocess.lyb_veto.flag_lybeta.")
    p.add_argument("--lyb-veto-dz", type=float, default=0.005)
    p.add_argument("--zcat", default=None)
    p.add_argument("--variant", action="append", default=[],
                   help='Override VARIANTS list. Format: "LABEL=PATH[:MARKER:COLOR]". '
                        "Repeat for multiple variants.")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    return p.parse_args()


def load_user_variants(spec_list):
    """Parse --variant LABEL=PATH[:MARKER:COLOR] entries; fall back to defaults."""
    out = []
    for s in spec_list:
        if "=" not in s:
            print(f"[warn] ignoring malformed --variant {s!r} (need LABEL=PATH)")
            continue
        label, rest = s.split("=", 1)
        parts = rest.split(":")
        path = parts[0]
        marker = parts[1] if len(parts) > 1 else "o"
        color = parts[2] if len(parts) > 2 else None
        out.append((label.strip(), path.strip(), marker, color or "C0"))
    return out


def evaluate_variant(label: str, catalog_dir: str, truth_path: str,
                     args) -> dict | None:
    """Run the molly-faithful eval and return per-cut purity, completeness."""
    if not os.path.isdir(catalog_dir):
        print(f"[skip] {label}: dir not found — {catalog_dir}")
        return None
    try:
        cat = load_catalog_dir(catalog_dir)
    except SystemExit as e:
        print(f"[skip] {label}: {e}")
        return None

    if "Z_QSO" not in cat.colnames:
        print(f"[skip] {label}: catalog has no Z_QSO column")
        return None

    qso_lookup = build_per_qso_snr(catalog_dir)
    truth = load_truth_molly(truth_path, args.truth_nhi_min, qso_lookup,
                             zcat_path=args.zcat)

    # SNR_RED on the cat side
    if "S2N_RED" not in cat.colnames:
        s2n = np.full(len(cat), np.nan)
        for i, tid in enumerate(np.asarray(cat["TARGETID"], dtype=np.int64)):
            v = qso_lookup.get(int(tid))
            if v is not None:
                s2n[i] = v[0]
        cat["S2N_RED"] = s2n

    bal_tids = None
    if args.no_bal and args.bal_cat and os.path.exists(args.bal_cat):
        cat, truth = apply_bal_cut(cat, truth, args.bal_cat)

    # λ_rf + z_QSO cuts on cat AND truth
    cat = make_lambda_z_BAL_cuts(cat, args.lam_rf_min, args.lam_rf_max,
                                 args.z_qso_min, args.z_qso_max,
                                 bal_tids=bal_tids,
                                 z_col_for_min="Z_DLA",
                                 use_truth_z=True)
    truth = make_lambda_z_BAL_cuts(truth, args.lam_rf_min, args.lam_rf_max,
                                   args.z_qso_min, args.z_qso_max,
                                   bal_tids=bal_tids,
                                   z_col_for_min="Z_DLA",
                                   use_truth_z=False)

    # Optional lyb veto on the cat — flag_lybeta defaults to nhi_col="LOG_NHI"
    # but our dlacat-*.fits stores it as "NHI"
    if args.lyb_veto:
        try:
            from gpy_dla_detection.postprocess import lyb_veto
            nhi_col = "NHI" if "NHI" in cat.colnames else "LOG_NHI"
            tbl = lyb_veto.flag_lybeta(cat, nhi_col=nhi_col,
                                       dz_match=args.lyb_veto_dz)
            flags = np.asarray(tbl["LYBETA_FLAG"], dtype=bool)
            cat = cat[~flags]
            print(f"[lyb-veto] {label}: removed {flags.sum()} rows")
        except Exception as e:
            print(f"[lyb-veto] {label}: skipped ({e})")

    # Truth-match
    tp, _, _, _ = match_truth_to_cat(cat, truth, dz_rel=args.dz_rel)

    good_mask = None
    if "DLAFLAG" in cat.colnames:
        good_mask = (np.asarray(cat["DLAFLAG"], dtype=int) == 0)

    # Sweep P_DLA cuts
    pur = np.full(len(LOG_PDLA_SWEEP), np.nan)
    cmp_ = np.full(len(LOG_PDLA_SWEEP), np.nan)
    n_kept = np.full(len(LOG_PDLA_SWEEP), 0, dtype=int)
    n_tp = np.full(len(LOG_PDLA_SWEEP), 0, dtype=int)
    n_truth = np.full(len(LOG_PDLA_SWEEP), 0, dtype=int)
    for i, lp in enumerate(LOG_PDLA_SWEEP):
        pcut = 1. - 10.**lp
        ntp, ntot, p_val = purity_min(cat, tp, args.snr_min, args.nhi_min,
                                       pcut, good_mask)
        nfound, nfid, c_val = completeness_min(cat, tp, args.snr_min,
                                                args.truth_nhi_min,
                                                args.nhi_min, pcut, truth,
                                                good_mask)
        pur[i] = p_val
        cmp_[i] = c_val
        n_kept[i] = ntot
        n_tp[i] = ntp
        n_truth[i] = nfid

    print(f"[ok] {label}: at P_DLA≥0.99 → P={pur[3]:.3f} C={cmp_[3]:.3f} "
          f"(n_cat={n_kept[3]}, n_truth={n_truth[3]})")

    return {
        "label": label,
        "log_pdla": LOG_PDLA_SWEEP,
        "p_dla_cut": 1. - 10.**LOG_PDLA_SWEEP,
        "purity": pur,
        "completeness": cmp_,
        "n_cat_kept": n_kept,
        "n_tp": n_tp,
        "n_truth_kept": n_truth,
    }


def make_plot(results, out_dir, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True, dpi=120)

    norm = Normalize(vmin=LOG_PDLA_SWEEP.min(), vmax=LOG_PDLA_SWEEP.max())

    for r in results:
        # Connect with a thin line so the variant's trajectory through P_DLA is visible
        ax.plot(r["completeness"], r["purity"], color=r["color"], lw=0.7, alpha=0.4)
        ax.scatter(r["completeness"], r["purity"],
                   marker=r["marker"], c=r["log_pdla"], cmap="viridis_r",
                   norm=norm, s=55, edgecolors=r["color"], linewidths=1.2,
                   label=r["label"], zorder=3)
        # Star at log_pdla = -2 (P_DLA = 0.99) — the headline operating point
        i99 = int(np.argmin(np.abs(LOG_PDLA_SWEEP + 2.)))
        ax.scatter([r["completeness"][i99]], [r["purity"][i99]],
                   marker="*", c=[r["color"]], s=220,
                   edgecolors="black", linewidths=1.2, zorder=4)

    # Reference grid
    ax.axhline(0.85, color="0.5", lw=0.5, ls=":")
    ax.axvline(0.80, color="0.5", lw=0.5, ls=":")
    ax.text(0.79, 0.86, "target 80/85", fontsize=8, color="0.4")

    # Colorbar for log_pdla
    sm = ScalarMappable(norm=norm, cmap="viridis_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("log$_{10}$(1 − P_DLA cut)", fontsize=9)

    # Legend (variants only, no point)
    leg = ax.legend(loc="lower left", fontsize=8, framealpha=0.85,
                    title="(★ marks P_DLA ≥ 0.99)", title_fontsize=8)
    for lh in leg.legend_handles:
        try:
            lh.set_color(lh.get_facecolor()[0] if hasattr(lh, "get_facecolor") else "k")
        except Exception:
            pass

    ax.set_xlabel("Completeness")
    ax.set_ylabel("Purity")
    if title:
        ax.set_title(title)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.2)

    out_png = os.path.join(out_dir, "purity_vs_completeness_all_variants.png")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[plot] wrote {out_png}")
    plt.close(fig)


def write_tsv(results, out_dir):
    tsv_path = os.path.join(out_dir, "purity_vs_completeness_all_variants.tsv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["variant", "log_pdla", "p_dla_cut",
                    "purity", "completeness", "n_cat_kept", "n_tp", "n_truth_kept"])
        for r in results:
            for i in range(len(r["log_pdla"])):
                w.writerow([r["label"], f"{r['log_pdla'][i]:.2f}",
                            f"{r['p_dla_cut'][i]:.6f}",
                            f"{r['purity'][i]:.4f}", f"{r['completeness'][i]:.4f}",
                            int(r["n_cat_kept"][i]), int(r["n_tp"][i]),
                            int(r["n_truth_kept"][i])])
    print(f"[tsv] wrote {tsv_path}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    variants = load_user_variants(args.variant) or DEFAULT_VARIANTS
    results = []
    for label, path, marker, color in variants:
        r = evaluate_variant(label, path, args.truth, args)
        if r is None:
            continue
        r["marker"] = marker
        r["color"] = color
        results.append(r)
    if not results:
        raise SystemExit("[fatal] no variants evaluated")
    write_tsv(results, args.out)
    make_plot(results, args.out,
              args.title or f"GP variants — SNR>{args.snr_min}, NHI_pred>{args.nhi_min}")


if __name__ == "__main__":
    main()
