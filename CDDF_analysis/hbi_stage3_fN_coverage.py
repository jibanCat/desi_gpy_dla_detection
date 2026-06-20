"""hbi_stage3_fN_coverage.py — Stage III differential f(N) per-bin coverage diagnostic.

Measures whether the 2LPT-0 truth f(N) lies inside the Stage-III marginalized
MC band (frozen / step2 / optionally step1) for the purity_mixture (PM) estimator,
bin by bin in logN.

Stage III is the response (θ_K) marginalization — the dominant coverage lever.
The pre-Stage-III (frozen) band is included for comparison.

Outputs (to --out directory):
  fN_coverage.npz         — f_b_samples, map, truth, bins, mode bands
  fN_coverage_report.txt  — per-bin coverage table
  fig_fN_coverage.png     — differential log10 f(N) vs logN, step2 68% band + truth
  (if --skip-pm not set)

Analysis-side ONLY. NO GP inference. Uses the calibrated WALL-1 bundle
(broaden012 2-D posterior kernel + lya_only-nhi195 molly + v3 bspbody, floor 19.5).

Usage:
  python CDDF_analysis/hbi_stage3_fN_coverage.py \\
      --n-mc 120 --workers 4 --modes frozen,step2 \\
      --out /scratch/cavestru_root/cavestru0/mfho/hbi_stage3_fN/
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse stage3 infrastructure verbatim
from CDDF_analysis.hbi_validation_2lpt0_stage3 import (
    DEF_ZNZ, DEF_KERNEL, DEF_LOA0, DEF_CAT, DEF_TRUTH, DEF_BAL,
    MODES, MODE_CFG, run_pm,
)
from CDDF_analysis.ab_loa0_fp_baseline import build_ingredients, DEF_LYAONLY_MOLLY
from CDDF_analysis.cddf_catalog_hbi import truth_reductions


def _cov_bin(samp_col, truth_val):
    """Coverage stats for a single bin (samp_col shape (n_mc,))."""
    s = np.asarray(samp_col, float)
    fin = s[np.isfinite(s)]
    if len(fin) == 0 or not np.isfinite(truth_val) or truth_val <= 0:
        return dict(lo68=np.nan, hi68=np.nan, lo95=np.nan, hi95=np.nan,
                    med=np.nan, cov68=False, cov95=False)
    lo68, hi68 = np.nanpercentile(s, 16), np.nanpercentile(s, 84)
    lo95, hi95 = np.nanpercentile(s, 2.5), np.nanpercentile(s, 97.5)
    med = np.nanpercentile(s, 50)
    return dict(lo68=lo68, hi68=hi68, lo95=lo95, hi95=hi95, med=med,
                cov68=bool(lo68 <= truth_val <= hi68),
                cov95=bool(lo95 <= truth_val <= hi95))


def make_coverage_report(N_b, dN_b, logN_lo, logN_hi, f_truth_b,
                         map_fb, bands, modes_run, fit_floor=19.5):
    """Return a formatted text coverage table + per-bin records."""
    n_bins = len(N_b)
    records = []
    for b in range(n_bins):
        if logN_lo[b] < fit_floor - 1e-9:
            continue
        if f_truth_b[b] <= 0 and np.isnan(f_truth_b[b]):
            continue
        rec = dict(
            b=b,
            logN_lo=float(logN_lo[b]),
            logN_hi=float(logN_hi[b]),
            logN_mid=float(0.5 * (logN_lo[b] + logN_hi[b])),
            f_truth=float(f_truth_b[b]),
            f_map=float(map_fb[b]),
        )
        for mode in modes_run:
            samp_col = bands[mode]["f_b_samples"][:, b]
            cv = _cov_bin(samp_col, f_truth_b[b])
            rec[f"{mode}_lo68"] = cv["lo68"]
            rec[f"{mode}_hi68"] = cv["hi68"]
            rec[f"{mode}_lo95"] = cv["lo95"]
            rec[f"{mode}_hi95"] = cv["hi95"]
            rec[f"{mode}_med"] = cv["med"]
            rec[f"{mode}_cov68"] = cv["cov68"]
            rec[f"{mode}_cov95"] = cv["cov95"]
        records.append(rec)

    # Format text table (step2 focus + frozen for comparison)
    lines = []
    lines.append("=" * 100)
    lines.append("Stage-III differential f(N) per-bin coverage — PM estimator vs 2LPT-0 truth")
    lines.append("=" * 100)
    hdr_parts = [f"{'logN_mid':>8}", f"{'f_truth':>12}", f"{'f_MAP':>12}"]
    for mode in modes_run:
        hdr_parts += [f"{mode+'_lo68':>12}", f"{mode+'_hi68':>12}",
                      f"{'cov68':>5}", f"{'cov95':>5}"]
    lines.append("  ".join(hdr_parts))
    lines.append("-" * 100)

    for rec in records:
        miss_tags = []
        row_parts = [
            f"{rec['logN_mid']:8.3f}",
            f"{rec['f_truth']:12.4e}",
            f"{rec['f_map']:12.4e}",
        ]
        for mode in modes_run:
            lo68 = rec[f"{mode}_lo68"]
            hi68 = rec[f"{mode}_hi68"]
            cov68 = rec[f"{mode}_cov68"]
            cov95 = rec[f"{mode}_cov95"]
            row_parts += [
                f"{lo68:12.4e}", f"{hi68:12.4e}",
                f"{'Y' if cov68 else 'N':5}",
                f"{'Y' if cov95 else 'N':5}",
            ]
            if not cov95 and mode == "step2":
                sign = "OVER" if rec["f_map"] > rec["f_truth"] else "UNDER"
                miss_tags.append(f"MISS-{sign}")
        tag = "  <== " + " ".join(miss_tags) if miss_tags else ""
        lines.append("  ".join(row_parts) + tag)

    lines.append("")
    lines.append("Summary per mode:")
    for mode in modes_run:
        n_rep = sum(1 for r in records if np.isfinite(r["f_truth"]) and r["f_truth"] > 0)
        n68 = sum(1 for r in records if r.get(f"{mode}_cov68", False))
        n95 = sum(1 for r in records if r.get(f"{mode}_cov95", False))
        n_miss_over  = sum(1 for r in records
                          if not r.get(f"{mode}_cov95", False)
                          and r["f_map"] > r["f_truth"])
        n_miss_under = sum(1 for r in records
                          if not r.get(f"{mode}_cov95", False)
                          and r["f_map"] <= r["f_truth"])
        lines.append(f"  {mode:8s}: {n68}/{n_rep} bins in 68%, {n95}/{n_rep} bins in 95%  "
                     f"(miss-over={n_miss_over}, miss-under={n_miss_under})")
    lines.append("")

    # Diagnose the shape of the miss
    step2_key = "step2" if "step2" in modes_run else modes_run[-1]
    if step2_key in modes_run:
        lines.append(f"Shape of miss ({step2_key} mode, bins NOT in 95%):")
        miss_bins = [r for r in records
                     if not r.get(f"{step2_key}_cov95", False)
                     and np.isfinite(r["f_truth"]) and r["f_truth"] > 0]
        if not miss_bins:
            lines.append("  ALL bins covered at 95% — no systematic miss detected.")
        else:
            for r in miss_bins:
                ratio = r["f_map"] / r["f_truth"] if r["f_truth"] > 0 else np.nan
                sign = "OVER" if r["f_map"] > r["f_truth"] else "UNDER"
                lines.append(
                    f"  logN {r['logN_mid']:.2f}: f_truth={r['f_truth']:.3e}  "
                    f"f_MAP={r['f_map']:.3e}  ratio={ratio:.3f}  [{sign}]  "
                    f"95%=[{r[step2_key+'_lo95']:.3e},{r[step2_key+'_hi95']:.3e}]")
        lines.append("")

    return "\n".join(lines), records


def make_figure(out_path, N_b, dN_b, logN_lo, logN_hi,
                f_truth_b, map_fb, bands, modes_run, fit_floor=19.5):
    """Panel 1: step2 68% band vs truth. Panel 2 (if frozen in modes): frozen vs step2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = 0.5 * (logN_lo + logN_hi)
    sel = (logN_lo >= fit_floor - 1e-9) & (logN_lo <= 22.4 + 1e-9)

    # Determine how many panels
    has_frozen = "frozen" in modes_run
    n_panels = 2 if has_frozen else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 6.0))
    if n_panels == 1:
        axes = [axes]

    def _log10safe(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.asarray(a) > 0, np.log10(np.asarray(a)), np.nan)

    # Panel 1: step2 band + truth
    ax = axes[0]
    step2_key = "step2" if "step2" in modes_run else modes_run[-1]
    s2samp = bands[step2_key]["f_b_samples"]  # (n_mc, n_bins)
    s2_q16 = np.nanpercentile(s2samp, 16, axis=0)
    s2_q84 = np.nanpercentile(s2samp, 84, axis=0)
    s2_q025 = np.nanpercentile(s2samp, 2.5, axis=0)
    s2_q975 = np.nanpercentile(s2samp, 97.5, axis=0)
    s2_med = np.nanpercentile(s2samp, 50, axis=0)

    sel_s2 = sel & (s2_q16 > 0) & (s2_q84 > 0)
    ax.fill_between(mid[sel_s2],
                    _log10safe(s2_q025[sel_s2]),
                    _log10safe(s2_q975[sel_s2]),
                    color="C0", alpha=0.15, label=f"{step2_key} 95% MC", zorder=1)
    ax.fill_between(mid[sel_s2],
                    _log10safe(s2_q16[sel_s2]),
                    _log10safe(s2_q84[sel_s2]),
                    color="C0", alpha=0.35, label=f"{step2_key} 68% MC", zorder=2)
    ax.plot(mid[sel_s2], _log10safe(s2_med[sel_s2]),
            color="C0", lw=1.5, ls="--", label=f"{step2_key} MC median", zorder=3)

    # MAP
    sel_map = sel & (map_fb > 0)
    ax.plot(mid[sel_map], _log10safe(map_fb[sel_map]),
            "x", color="C0", ms=7, mew=1.8, label="PM MAP", zorder=4)

    # Truth
    sel_tr = sel & (f_truth_b > 0)
    ax.plot(mid[sel_tr], _log10safe(f_truth_b[sel_tr]),
            "k*-", ms=8, lw=1.8, label="2LPT-0 truth", zorder=5)

    # Annotate coverage per bin
    for b in range(len(mid)):
        if not sel[b]:
            continue
        if f_truth_b[b] <= 0 or not np.isfinite(f_truth_b[b]):
            continue
        samp_col = s2samp[:, b]
        lo95, hi95 = np.nanpercentile(samp_col, 2.5), np.nanpercentile(samp_col, 97.5)
        lo68, hi68 = np.nanpercentile(samp_col, 16), np.nanpercentile(samp_col, 84)
        t = f_truth_b[b]
        if not (lo68 <= t <= hi68):
            sign = "▲" if map_fb[b] > t else "▼"
            ax.annotate(sign, (mid[b], _log10safe(t)),
                        color="red" if not (lo95 <= t <= hi95) else "orange",
                        fontsize=9, ha="center", va="bottom", zorder=6)

    ax.axvline(20.3, color="0.5", ls=":", lw=1.0)
    ax.text(20.32, ax.get_ylim()[0] + 0.05 if ax.get_ylim()[0] > -99 else -22,
            "20.3", fontsize=8, color="0.5")
    ax.set_xlabel(r"$\log_{10}\,N_{\rm HI}$")
    ax.set_ylabel(r"$\log_{10}\,f(N_{\rm HI})$")
    ax.set_title(f"Stage III {step2_key} band vs 2LPT-0 truth\n"
                 "(▲=MAP over truth outside 68%; ▼=under; red=outside 95%)")
    ax.set_xlim(fit_floor - 0.05, 22.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.22)

    # Panel 2: frozen vs step2 overlay (if both modes ran)
    if has_frozen and n_panels > 1:
        ax2 = axes[1]
        mode_styles = {"frozen": ("C1", "-", 0.30),
                       "step1":  ("C2", "--", 0.25),
                       "step2":  ("C0", "-", 0.35)}
        for mode in modes_run:
            c, ls, alpha = mode_styles.get(mode, ("C3", "-", 0.2))
            samp = bands[mode]["f_b_samples"]
            q16 = np.nanpercentile(samp, 16, axis=0)
            q84 = np.nanpercentile(samp, 84, axis=0)
            sel_m = sel & (q16 > 0) & (q84 > 0)
            ax2.fill_between(mid[sel_m],
                             _log10safe(q16[sel_m]), _log10safe(q84[sel_m]),
                             color=c, alpha=alpha, label=f"{mode} 68%", zorder=2)
            ax2.plot(mid[sel_m], _log10safe(np.nanpercentile(samp, 50, axis=0)[sel_m]),
                     color=c, lw=1.3, ls=ls, zorder=3)

        ax2.plot(mid[sel_tr], _log10safe(f_truth_b[sel_tr]),
                 "k*-", ms=8, lw=1.8, label="2LPT-0 truth", zorder=5)
        ax2.axvline(20.3, color="0.5", ls=":", lw=1.0)
        ax2.set_xlabel(r"$\log_{10}\,N_{\rm HI}$")
        ax2.set_ylabel(r"$\log_{10}\,f(N_{\rm HI})$")
        ax2.set_title("Stage III: frozen vs step2 (response-form shift per bin)")
        ax2.set_xlim(fit_floor - 0.05, 22.5)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax2.grid(alpha=0.22)

    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    print(f"[fig] -> {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=DEF_LYAONLY_MOLLY)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--kernel-znz", default=DEF_ZNZ)
    p.add_argument("--loa0-product", default=DEF_LOA0)
    p.add_argument("--out",
                   default="/scratch/cavestru_root/cavestru0/mfho/hbi_stage3_fN")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--n-mc", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--modes", default="frozen,step2",
                   help="Comma-separated subset of frozen,step1,step2 to run.")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    modes_run = [m.strip() for m in args.modes.split(",")]
    # Validate
    for m in modes_run:
        if m not in MODES:
            raise ValueError(f"Unknown mode {m!r}; must be one of {MODES}")

    limits = tuple(float(x) for x in args.report_limits.split(","))

    # Temporarily restrict MODES in stage3 to only what we need (avoid wasted MC)
    # We do this by monkey-patching MODE_CFG visibility — simpler: just call run_pm
    # with a mock that only processes the requested modes.
    import CDDF_analysis.hbi_validation_2lpt0_stage3 as S3
    # Override MODES in the module to restrict which bands are computed
    original_modes = S3.MODES
    S3.MODES = tuple(modes_run)

    t0 = time.time()
    print("=" * 78)
    print(f"[fN-coverage] PM Stage III f(N) coverage diagnostic")
    print(f"  modes={modes_run}  n_mc={args.n_mc}  workers={args.workers}")
    print(f"  kernel={args.kernel}")
    print(f"  kernel_znz={args.kernel_znz}")
    print("=" * 78)
    out_pm = run_pm(args, limits, args.seed)
    S3.MODES = original_modes  # restore
    print(f"[fN-coverage] PM MC bands done ({time.time()-t0:.0f}s)")

    # Extract geometry
    point = out_pm["point"]
    tr    = out_pm["truth"]
    bands = out_pm["bands"]
    f_truth_b = np.asarray(tr["f_truth"], float)
    map_fb    = np.asarray(point["f_b"], float)

    # Need N_b, dN_b, logN_lo, logN_hi — these are stored in the run_pm ingredients
    # We re-build them from the point dict's _v3x internals (fine grid)
    # Easier: re-build from scratch via build_ingredients (already done in run_pm)
    # The fine grid from run_pm is accessible via point["_v3x"]["fwd"]["fine"]
    logN_lo_fine, logN_hi_fine, N_b_fine, dN_b_fine, _ = point["_v3x"]["fwd"]["fine"]
    # These are the fine z-integrated arrays.
    # But truth_reductions uses the coarse grid (build_fine_grid == logN_lo/logN_hi/N_b/dN_b).
    # In run_pm, the truth arrays come from truth_reductions on the same cfg/ing.
    # We can recover the bin arrays directly from the fine grid edges (z-marg bins):
    logN_lo = np.asarray(logN_lo_fine, float)
    logN_hi = np.asarray(logN_hi_fine, float)
    N_b     = np.asarray(N_b_fine, float)
    dN_b    = np.asarray(dN_b_fine, float)

    # Validate shapes
    n_bins = len(N_b)
    for mode in modes_run:
        assert bands[mode]["f_b_samples"].shape[1] == n_bins, \
            f"f_b_samples shape mismatch for mode {mode}"

    # Coverage report
    report_txt, records = make_coverage_report(
        N_b, dN_b, logN_lo, logN_hi, f_truth_b, map_fb, bands, modes_run,
        fit_floor=args.fit_floor)
    print("\n" + report_txt)

    rpt_path = os.path.join(args.out, "fN_coverage_report.txt")
    with open(rpt_path, "w") as fh:
        fh.write(report_txt + "\n")
    print(f"[report] -> {rpt_path}")

    # Figure
    fig_path = os.path.join(args.out, "fig_fN_coverage.png")
    make_figure(fig_path, N_b, dN_b, logN_lo, logN_hi,
                f_truth_b, map_fb, bands, modes_run, fit_floor=args.fit_floor)

    # Save npz
    savez = dict(
        logN_lo=logN_lo, logN_hi=logN_hi, N_b=N_b, dN_b=dN_b,
        f_truth_b=f_truth_b, f_map=map_fb,
        modes_run=np.array(modes_run),
    )
    for mode in modes_run:
        savez[f"f_b_samples_{mode}"] = bands[mode]["f_b_samples"]
        savez[f"f_b_q16_{mode}"] = np.nanpercentile(bands[mode]["f_b_samples"], 16, axis=0)
        savez[f"f_b_q84_{mode}"] = np.nanpercentile(bands[mode]["f_b_samples"], 84, axis=0)
        savez[f"f_b_q025_{mode}"] = np.nanpercentile(bands[mode]["f_b_samples"], 2.5, axis=0)
        savez[f"f_b_q975_{mode}"] = np.nanpercentile(bands[mode]["f_b_samples"], 97.5, axis=0)
        savez[f"f_b_med_{mode}"] = np.nanpercentile(bands[mode]["f_b_samples"], 50, axis=0)
    npz_path = os.path.join(args.out, "fN_coverage.npz")
    np.savez(npz_path, **savez)
    print(f"[npz] -> {npz_path}")

    print(f"\n[done] all outputs in {args.out}  ({time.time()-t0:.0f}s total)")
    return dict(records=records, report=report_txt, out_pm=out_pm)


if __name__ == "__main__":
    main()
