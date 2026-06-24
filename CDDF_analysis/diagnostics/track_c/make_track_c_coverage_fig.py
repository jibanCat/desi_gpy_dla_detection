#!/usr/bin/env python
"""make_track_c_coverage_fig.py — Track-C SHOULDER coverage figure (the headline result).

Reads the `track_c_td_band` deliverable JSON (`td_band.json` + `td_band_fN.json` written
by CDDF_analysis/track_c_td_band.py with --omega-slope-extrap-integrated --slope-edge 21.1
--band-recenter --omega-slope-extrap) and renders a clean COVERAGE figure showing
truth-in-bar at the DLA headline limits.

Panels:
  (a) dN/dX(≥20.0, ≥20.3): MAP point (×), 68% + 95% band, 2LPT-0 mock truth (★).
  (b) Ω(≥20.0, ≥20.3): same, with the [21,21.5]-shoulder slope-uncertainty-widened band.
  (c) deep-tail Ω(≥21.3): the slope-extrapolation-widened band covering truth.
  (d) f(N) differential with the band (DLA range), truth overlaid.

Coverage is visually clear: each error box is colored GREEN if truth falls inside the 68%
(dark) / 95% (light) band, RED if it misses; the truth marker (★) sits on/off the bar.

Run:
    python -m CDDF_analysis.make_track_c_coverage_fig \
        --band-json /scratch/.../track_c/td_band_shoulder/td_band.json \
        --fN-json   /scratch/.../track_c/td_band_shoulder/td_band_fN.json \
        --out /scratch/.../track_c/td_band_shoulder/fig_track_c_coverage.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


_GREEN = "#1b7837"
_GREEN_L = "#a6dba0"
_RED = "#b2182b"
_RED_L = "#f4a582"
_MAP = "#2166ac"
_TRUTH = "#d6604d"


def _cover_colors(cover68, cover95):
    """dark = 68% box, light = 95% box; green if covered, red if not."""
    c68 = _GREEN if cover68 else _RED
    c95 = _GREEN_L if cover95 else _RED_L
    return c68, c95


def _panel_limits(ax, entries, title, ylabel, scale=1.0, unit=""):
    """entries: list of dicts with keys point, truth, band68, band95, cover68, cover95, label."""
    xs = np.arange(len(entries))
    for i, e in enumerate(entries):
        lo68, hi68 = e["band68"][0] * scale, e["band68"][1] * scale
        lo95, hi95 = e["band95"][0] * scale, e["band95"][1] * scale
        pt = e["point"] * scale
        tr = e["truth"] * scale
        c68, c95 = _cover_colors(e["cover68"], e["cover95"])
        w = 0.34
        # 95% box (light)
        ax.add_patch(Rectangle((i - w, lo95), 2 * w, hi95 - lo95,
                               facecolor=c95, edgecolor="none", alpha=0.55, zorder=1))
        # 68% box (dark edge)
        ax.add_patch(Rectangle((i - w, lo68), 2 * w, hi68 - lo68,
                               facecolor=c68, edgecolor=c68, alpha=0.35, zorder=2,
                               linewidth=1.5))
        # MAP point (x)
        ax.plot([i], [pt], marker="x", color=_MAP, ms=11, mew=2.6, zorder=5)
        # truth (star)
        ax.plot([i], [tr], marker="*", color=_TRUTH, ms=20, mec="k", mew=0.7, zorder=6)
        # annotate R0 + cover
        r0 = e.get("MAP_R0", pt / tr if tr else np.nan)
        cov_txt = ("✓68" if e["cover68"] else ("✓95" if e["cover95"] else "miss"))
        ax.text(i, hi95 + (hi95 - lo95) * 0.10 + 1e-30, f"R0={r0:.3f}\n{cov_txt}",
                ha="center", va="bottom", fontsize=8.5,
                color=(_GREEN if (e["cover68"] or e["cover95"]) else _RED))
    ax.set_xticks(xs)
    ax.set_xticklabels([e["label"] for e in entries])
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel + (f"  [{unit}]" if unit else ""), fontsize=10)
    ax.margins(y=0.22)
    ax.grid(axis="y", ls=":", alpha=0.4)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--band-json", required=True)
    p.add_argument("--fN-json", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--mock-label", default="2LPT-0 mock injection")
    args = p.parse_args(argv)

    with open(args.band_json) as fh:
        B = json.load(fh)
    meta = B["metadata"]
    limits = [str(l) for l in meta["limits"]]

    def _entries(kind):
        out = []
        for l in limits:
            s = B[kind][l]
            out.append(dict(point=s["point"], truth=s["truth"],
                            band68=s["band68"], band95=s["band95"],
                            cover68=s["cover68"], cover95=s["cover95"],
                            MAP_R0=s["MAP_R0"], label=f"≥{l}"))
        return out

    dndx = _entries("dndx")
    omega = _entries("omega")
    dt = B.get("omega_deep_tail", {})
    has_fN = args.fN_json and os.path.exists(args.fN_json)

    n_panels = 4 if has_fN else 3
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    axes = axes.ravel()

    # (a) dN/dX
    _panel_limits(axes[0], dndx, "dN/dX  (DLA headline limits)",
                  r"$dN/dX$", scale=1.0)
    # (b) integrated Ω with shoulder-widened band
    om_se = bool(meta.get("omega_slope_extrap_integrated", False))
    edge = meta.get("omega_slope_extrap_edge", 21.2)
    sig = meta.get("omega_slope_extrap_sigma", 0.5)
    om_title = r"$\Omega_{\rm DLA}\times10^{3}$  (shoulder slope-unc. extended)" if om_se \
        else r"$\Omega_{\rm DLA}\times10^{3}$"
    _panel_limits(axes[1], omega, om_title, r"$\Omega_{\rm DLA}\times10^{3}$", scale=1e3)

    # (c) deep-tail Ω(≥21.3) — slope-extrap widened band
    ax = axes[2]
    if dt:
        se = dt.get("slope_extrap", None)
        tr = dt["truth"] * 1e4
        pt = dt["MAP"] * 1e4
        # carry band (pre-slope-extrap)
        b68 = [dt["band68"][0] * 1e4, dt["band68"][1] * 1e4]
        ax.add_patch(Rectangle((0 - 0.3, b68[0]), 0.6, b68[1] - b68[0],
                               facecolor="#cccccc", edgecolor="#888888", alpha=0.6,
                               label="kernel-carry (no extrap)", zorder=1))
        if se is not None:
            s95 = [se["band95"][0] * 1e4, se["band95"][1] * 1e4]
            s68 = [se["band68"][0] * 1e4, se["band68"][1] * 1e4]
            c68, c95 = _cover_colors(se["cover68"], se["cover95"])
            ax.add_patch(Rectangle((1 - 0.3, s95[0]), 0.6, s95[1] - s95[0],
                                   facecolor=c95, edgecolor="none", alpha=0.5, zorder=1))
            ax.add_patch(Rectangle((1 - 0.3, s68[0]), 0.6, s68[1] - s68[0],
                                   facecolor=c68, edgecolor=c68, alpha=0.35, zorder=2))
            ax.plot([1], [pt], marker="x", color=_MAP, ms=11, mew=2.6, zorder=5)
            cov = "✓95" if se["cover95"] else ("✓68" if se["cover68"] else "miss")
            ax.text(1, s95[1], f"  {cov}", ha="left", va="top", fontsize=9,
                    color=(_GREEN if (se["cover68"] or se["cover95"]) else _RED))
        ax.plot([0], [pt], marker="x", color=_MAP, ms=11, mew=2.6, zorder=5)
        ax.axhline(tr, color=_TRUTH, ls="--", lw=1.3, zorder=3)
        ax.plot([0, 1], [tr, tr], marker="*", color=_TRUTH, ms=18, mec="k",
                mew=0.6, ls="none", zorder=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["carry only", f"+ slope-extrap\n(σ={sig}, edge={edge})"])
        ax.set_title(r"deep-tail $\Omega_{\rm DLA}$ (NHI$\geq$%.1f)$\times10^{4}$"
                     % dt["lo"], fontsize=11)
        ax.set_ylabel(r"$\Omega_{\rm DLA}\times10^{4}$", fontsize=10)
        ax.margins(y=0.2)
        ax.grid(axis="y", ls=":", alpha=0.4)

    # (d) f(N) differential band
    ax = axes[3]
    if has_fN:
        with open(args.fN_json) as fh:
            FN = json.load(fh)
        rows = FN["f_b"]
        mids = np.array([r["logN_mid"] for r in rows])
        hbi = np.array([r["hbi"] for r in rows])
        tr = np.array([r["truth"] for r in rows])
        b68 = np.array([r["band68"] for r in rows])
        b95 = np.array([r["band95"] for r in rows])
        m = (mids >= 20.0) & (mids <= 21.8)
        ax.fill_between(mids[m], b95[m, 0], b95[m, 1], color=_GREEN_L, alpha=0.4,
                        label="95% band", zorder=1)
        ax.fill_between(mids[m], b68[m, 0], b68[m, 1], color=_GREEN, alpha=0.30,
                        label="68% band", zorder=2)
        ax.plot(mids[m], hbi[m], color=_MAP, lw=1.8, marker="x", ms=5,
                label="HBI MAP", zorder=4)
        ax.plot(mids[m], tr[m], color=_TRUTH, lw=1.8, ls="--", marker="*", ms=9,
                label=f"{args.mock_label} truth", zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10} N_{\rm HI}$", fontsize=10)
        ax.set_ylabel(r"$f(N_{\rm HI})$", fontsize=10)
        ax.set_title("differential CDDF  f(N)  (DLA range)", fontsize=11)
        ax.axvline(edge, color="0.5", ls=":", lw=1.2)
        ax.text(edge, ax.get_ylim()[1], f" slope-extrap\n edge {edge}", fontsize=8,
                va="top", ha="left", color="0.4")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(ls=":", alpha=0.4)
    else:
        ax.axis("off")

    # shared legend
    leg_handles = [
        Line2D([0], [0], marker="x", color=_MAP, ls="none", ms=10, mew=2.4,
               label="HBI MAP point"),
        Line2D([0], [0], marker="*", color=_TRUTH, ls="none", ms=15, mec="k",
               label=f"{args.mock_label} truth"),
        Rectangle((0, 0), 1, 1, facecolor=_GREEN, alpha=0.35, label="68% band (covers)"),
        Rectangle((0, 0), 1, 1, facecolor=_GREEN_L, alpha=0.5, label="95% band (covers)"),
        Rectangle((0, 0), 1, 1, facecolor=_RED, alpha=0.35, label="68% band (misses)"),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    se_tag = (f"  |  shoulder slope-unc edge={edge}, σ={sig}" if om_se else "")
    fig.suptitle(f"Track-C coverage — {args.mock_label} (HBI, n_mc={meta['n_mc']})"
                 f"{se_tag}", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
