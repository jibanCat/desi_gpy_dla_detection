#!/usr/bin/env python
"""hbi_validation_replot.py — regenerate the two validation figures + summary md
from the already-computed hbi_validation_results.json (NO recompute of the ~35-min
MC bands). Reads the persisted quantiles directly.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main(json_path):
    out_dir = os.path.dirname(json_path)
    with open(json_path) as fh:
        D = json.load(fh)
    limits = [float(x) for x in D["metadata"]["report_limits"]]
    FPS = ("purity_mixture", "loa0")

    # ---------- FIG 1: differential f(N) log-log ----------
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    # truth + per-bin from loa0 block (truth identical across fp; f_b reported 20.0..22)
    fb = {fp: D[fp]["f_b"] for fp in FPS}
    mid = np.array([b["logN_mid"] for b in fb["loa0"]])
    truth = np.array([b["truth"] for b in fb["loa0"]])
    lo_pt = np.array([b["hbi"] for b in fb["loa0"]])
    lo_q16 = np.array([b["mc_q16"] for b in fb["loa0"]])
    lo_q84 = np.array([b["mc_q84"] for b in fb["loa0"]])
    pm_pt = np.array([b["hbi"] for b in fb["purity_mixture"]])
    # recenter-on-point (Track-C #34): per-bin additively shift the MC band so its median
    # (mc_q50) lands on the plug-in MAP point (lo_pt), width-preserving. Without it the
    # convex-bspline-MAP Jensen offset drifts the raw-percentile band off the point.
    lo_q50 = np.array([b["mc_q50"] for b in fb["loa0"]])
    _sh = np.where(np.isfinite(lo_q50) & np.isfinite(lo_pt) & (lo_pt > 0),
                   lo_pt - lo_q50, 0.0)
    lo_q16 = lo_q16 + _sh
    lo_q84 = lo_q84 + _sh

    def _logsafe(a):
        return np.where(np.asarray(a) > 0, np.log10(np.clip(a, 1e-300, None)), np.nan)

    tsel = truth > 0
    ax.plot(mid[tsel], _logsafe(truth)[tsel], "k-", lw=2.2, label="2LPT-0 truth", zorder=5)
    ax.fill_between(mid, _logsafe(lo_q16), _logsafe(lo_q84), color="C3", alpha=0.25,
                    label="HBI loa0 68% MC band", zorder=2)
    psel = lo_pt > 0
    ax.plot(mid[psel], _logsafe(lo_pt)[psel], "C3-", lw=1.8,
            label="HBI loa0 (point)", zorder=4)
    pmsel = pm_pt > 0
    ax.plot(mid[pmsel], _logsafe(pm_pt)[pmsel], "C0--", lw=1.6,
            label="HBI purity_mixture (point)", zorder=3)
    ax.axvline(20.0, color="0.4", ls=":", lw=1.2)
    ax.axvline(20.3, color="0.6", ls=":", lw=1.0)
    y0 = ax.get_ylim()[0]
    ax.text(20.02, y0 + 0.3, "20.0", rotation=90, va="bottom", fontsize=8, color="0.4")
    ax.text(20.32, y0 + 0.3, "20.3", rotation=90, va="bottom", fontsize=8, color="0.6")
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$")
    ax.set_ylabel(r"$\log_{10} f(N_{\rm HI})$")
    ax.set_title("Catalog-HBI differential CDDF vs 2LPT-0 truth (DLA tier)")
    ax.set_xlim(19.9, 22.0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig_hbi_validation_fN.png")
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"[fig] -> {p1}")

    # ---------- FIG 2: dN/dX(>=20.0) and Omega(>=20.0) ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    lim_s = "20.0"

    def _panel(ax, key, ylabel, scale, title):
        labels = ["loa0", "purity_mixture"]; colors = ["C3", "C0"]; xs = [0, 1]
        truth_v = D["loa0"][key][lim_s]["truth"] / scale
        ax.axhline(truth_v, color="k", lw=2.0, ls="-",
                   label=f"2LPT-0 truth = {truth_v:.4g}", zorder=1)
        for x, fp, c in zip(xs, labels, colors):
            blk = D[fp][key][lim_s]; b = blk["band"]
            pt = blk["point"] / scale; R0 = blk["R0"]
            q16 = b["q16"] / scale; q84 = b["q84"] / scale
            q025 = b["q025"] / scale; q975 = b["q975"] / scale; q50 = b["q50"] / scale
            # recenter-on-point (Track-C #34): per-x additive shift by (point - q50),
            # width-preserving, so the HBI marked-Poisson MC band median lands on the
            # plotted MAP point (the convex-bspline-MAP Jensen offset otherwise drifts
            # the raw-percentile band BELOW the point — same primitive as FIG 1).
            _sh = (pt - q50) if (np.isfinite(q50) and np.isfinite(pt)) else 0.0
            q16 += _sh; q84 += _sh; q025 += _sh; q975 += _sh; q50 += _sh
            ax.plot([x, x], [q025, q975], color=c, lw=1.2, alpha=0.5, zorder=2)
            ax.plot([x, x], [q16, q84], color=c, lw=5.0, alpha=0.30, zorder=3)
            ax.plot([x], [q50], marker="_", color=c, ms=16, mew=2.0, zorder=4)
            ax.plot([x], [pt], marker="o", color=c, ms=10, mec="k", mew=0.8, zorder=5,
                    label=f"{fp}: MAP (R0={R0:.3f})")
            ax.annotate(f"R0={R0:.3f}", (x, pt), textcoords="offset points",
                        xytext=(11, 4), fontsize=9, color=c)
        proxies = [Line2D([0], [0], color="0.4", lw=5, alpha=0.3,
                          label="68% MC band (recentered)"),
                   Line2D([0], [0], color="0.4", lw=1.2, alpha=0.5,
                          label="95% MC band (recentered)"),
                   Line2D([0], [0], color="0.4", marker="_", ls="none", ms=14, mew=2,
                          label="band median (= MAP)")]
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlim(-0.5, 1.5); ax.set_ylabel(ylabel); ax.set_title(title)
        h, l_ = ax.get_legend_handles_labels()
        ax.legend(h + proxies, l_ + [p.get_label() for p in proxies],
                  loc="best", fontsize=8.0, framealpha=0.9)
        ax.grid(alpha=0.25)

    _panel(axes[0], "dndx", r"$dN/dX\ (\geq 20.0)$", 1.0,
           r"$dN/dX(\geq 20.0)$ — HBI vs truth")
    _panel(axes[1], "omega", r"$\Omega_{\rm HI}\ (\geq 20.0)\ /\ 10^{-4}$", 1e-4,
           r"$\Omega_{\rm HI}(\geq 20.0)$ — HBI vs truth")
    fig.suptitle("Catalog-HBI integrated DLA recovery on 2LPT-0 "
                 "(MAP point + recentered MC band vs truth)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(out_dir, "fig_hbi_validation_dndx_omega.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"[fig] -> {p2}")

    # ---------- SUMMARY md ----------
    _write_summary(out_dir, D, limits)


def _write_summary(out_dir, D, limits):
    m = D["metadata"]
    def R0(fp, key, l):
        return D[fp][key][str(l)]["R0"]
    md = []
    md.append("# Catalog-HBI DLA validation on 2LPT-0\n")
    md.append("Reduce-only validation of the catalog-HBI DLA measurement against the "
              "2LPT-0 mock truth, on the calibrated WALL-1 bundle (broaden012 2-D "
              "posterior kernel + canonical lya_only-nhi195 molly + v3 bspbody, "
              "fit-floor 19.5, lam_rf_min 1025). NO inference, NO tilt.\n")
    md.append(f"- Kernel: `{m['kernel_path']}` (mtime {m['kernel_mtime']})")
    md.append(f"- Molly (corrected provenance): `{m['molly_path']}` (mtime {m['molly_mtime']})")
    md.append(f"- loa-0 FP product: `{m['loa0_product_path']}` (mtime {m['loa0_product_mtime']})")
    md.append(f"- Truth: `{m['truth_path']}`")
    md.append(f"- n_mc = {m['n_mc']}, family = {m['family']}, fit-floor = {m['fit_floor']}, "
              f"lam_rf_min = {m['lam_rf_min']}\n")

    md.append("## R0 = HBI / truth (recovery ratio; 1.0 = perfect; >1 = over-recovery)\n")
    md.append("| FP estimator | quantity | >=20.0 | >=20.3 | >=20.6 |")
    md.append("|---|---|---|---|---|")
    for fp in ("purity_mixture", "loa0"):
        md.append(f"| {fp} | dN/dX | "
                  + " | ".join(f"{R0(fp,'dndx',l):.4f}" for l in limits) + " |")
        md.append(f"| {fp} | Omega | "
                  + " | ".join(f"{R0(fp,'omega',l):.4f}" for l in limits) + " |")
    md.append("")

    md.append("## Absolute values at >=20.0 (MAP point [68% MC band]) vs truth\n")
    md.append("| FP estimator | dN/dX(>=20.0) | Omega(>=20.0) |")
    md.append("|---|---|---|")
    for fp in ("purity_mixture", "loa0"):
        d = D[fp]["dndx"]["20.0"]; o = D[fp]["omega"]["20.0"]
        md.append(f"| {fp} | {d['point']:.5f} [{d['band']['q16']:.5f}, "
                  f"{d['band']['q84']:.5f}] | {o['point']:.4e} "
                  f"[{o['band']['q16']:.3e}, {o['band']['q84']:.3e}] |")
    tr_d = D["loa0"]["dndx"]["20.0"]["truth"]; tr_o = D["loa0"]["omega"]["20.0"]["truth"]
    md.append(f"| **2LPT-0 truth** | **{tr_d:.5f}** | **{tr_o:.4e}** |")
    md.append("\n(The nuisance-MC band q16/q84 is centred on its own MC median, which on "
              "the steep DLA-tier f(N) sits BELOW the MAP point — the documented σ_i "
              "edge-scatter / +Eddington drift, spec §5. Read the band as an absolute "
              "spread, NOT as MAP ± error.)\n")

    md.append("## Honest interpretation\n")
    md.append(
        f"- **HBI recovers the 2LPT-0 truth to ~5% at >=20.0** (purity_mixture dN/dX "
        f"R0={R0('purity_mixture','dndx',20.0):.3f}, loa0 R0={R0('loa0','dndx',20.0):.3f}). "
        "The recovery is an **over-recovery** (R0>1) driven by the residual "
        "N-measurement / prior-edge Eddington migration that v1's selection correction "
        "does not deconvolve (spec §5/§9 — this is NOT '+0.06 dex gone').")
    md.append(
        "- **The over-recovery GROWS with threshold.** purity_mixture dN/dX R0 climbs "
        f"{R0('purity_mixture','dndx',20.0):.3f} -> {R0('purity_mixture','dndx',20.3):.3f} "
        f"-> {R0('purity_mixture','dndx',20.6):.3f} from >=20.0 to >=20.6; the sharp "
        "prior edge at log N_HI=20.3 piles posterior mass just above it.")
    md.append(
        "- **loa0 is the non-circular FP and reveals the true over-recovery that "
        "purity_mixture artificially masks.** The purity-mixture FP subtracts a per-row "
        "`(1-rho)` contamination that is itself calibrated on the SAME mock truth "
        "(circular) and, at the DLA tier, mechanically pulls the estimate toward truth. "
        "The loa0 product is a frozen forest false-positive intensity measured on a "
        "SEPARATE loa-0 field (non-circular), and shows the honest ~16% over-recovery "
        f"at >=20.3 (loa0 dN/dX R0={R0('loa0','dndx',20.3):.3f} vs purity_mixture "
        f"{R0('purity_mixture','dndx',20.3):.3f}; loa0 Omega R0="
        f"{R0('loa0','omega',20.3):.3f}).")
    md.append(
        "- **alpha(z) = 1/R0 reduce-only calibration closes the residual by "
        "construction.** The headline measurement applies the per-(N,z) completeness "
        "factor alpha(z)=1/R0 measured on this same mock as a REDUCE-ONLY (no "
        "re-inference) correction; by construction it removes the R0 over-recovery, "
        "leaving the bootstrap/MC band as the quoted uncertainty.")
    md.append(
        "- **WALL-1 tilt-robustness caveat (documented systematic, not a showstopper).** "
        "The WALL-1 +/-0.5 slope-tilt closure FAILS with an opposite-sign coherent pull "
        "(`V3_KERNEL_SLOPE_DEPENDENCE`): the empirical (Nhat|N,SNR) migration kernel is "
        "frozen at the untilted slope, so a tilted true slope changes the effective "
        "Eddington correction the same frozen kernel applies. The 2026-06-19 full-GP "
        "injection closure showed this proxy WALL-1 over-stated and mis-oriented the "
        "effect; the genuine slope dependence is ~1.8% at the operating-point Delta-alpha "
        "-- ~20x below the statistical sigma -- so it is carried as a small documented "
        "systematic on the DLA dN/dX/Omega, not a blocker.")
    md.append("")
    md.append("## Figures\n")
    md.append("![Differential f(N) vs truth](fig_hbi_validation_fN.png)\n")
    md.append("![Integrated dN/dX and Omega recovery](fig_hbi_validation_dndx_omega.png)\n")
    p = os.path.join(out_dir, "HBI_VALIDATION_SUMMARY.md")
    with open(p, "w") as fh:
        fh.write("\n".join(md))
    print(f"[md] -> {p}")


if __name__ == "__main__":
    jp = (sys.argv[1] if len(sys.argv) > 1 else
          "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
          "hbi_validation_2lpt0/hbi/hbi_validation_results.json")
    main(jp)
