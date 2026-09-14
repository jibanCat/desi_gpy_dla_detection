"""SEC 5 composite figure: the frozen completeness object C1nsadd.

Four panels, all re-drawn from RELEASED products (no refit, no new fit):
  (a) C vs log10 N_HI by S/N stratum, raw calibration fractions + fitted curve
  (b) C vs S/N at log10 N_HI = 19.7 / 20.0 / 20.3 / 20.6 / 21.0, with the
      production clamp at log10 S/N = 1.0287 (S/N 10.683) and the unphysical
      turn-over of the quadratic at S/N = 11.62 marked
  (c) the 2-D surface with the per-cell calibration support; cells holding
      < 20 truth systems marked (all of them at N >= 21.6)
  (d) the S/N x z structure the frozen g does NOT carry: the (stratum, K)
      residuals of C1nsadd x g against the truth completeness

Read-only.  Deterministic evaluation of the released 6 coefficients only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/cavestru_root/cavestru0/mfho/"
                   "absorber_ladder_2026-09-13/release/completeness")
import calib_common as cc                                        # noqa: E402
import evaluate_completeness as ec                               # noqa: E402

import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

COMP = os.path.join(cc.REL, "completeness")
REVJSON = ("/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_completeness_review"
           "/completeness_review_numbers.json")
OUT = os.path.join(cc.FIGDIR, "fig_sec5_completeness_composite.png")


def main():
    cc.style()
    m = np.load(os.path.join(COMP, "completeness_model.npz"))
    coef = m["coef"]
    N0 = float(m["N0"])
    lsnr_med = m["log10_snr_median_stratum"]
    snr_med = m["snr_median_stratum"]
    live = [int(s) for s in m["live_strata"]]
    dens = pd.read_csv(os.path.join(COMP, "calibration_density.csv"))
    rev = json.load(open(REVJSON))

    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.6))

    # ---------------- (a) C vs N by stratum ----------------
    ax = axes[0, 0]
    show = [2, 3, 5, 7]
    Ngrid = np.linspace(19.0, 22.4, 400)
    for i, s in enumerate(show):
        col = cc.PALETTE[i]
        d = dens[(dens.s == s) & (dens.n_truth > 0)]
        ctr = 0.5 * (d.logN_lo.values + d.logN_hi.values)
        k = d.n_detected.values
        n = d.n_truth.values
        p = k / n
        lo, hi = cc.wilson(k, n)
        ax.errorbar(ctr, p, yerr=[lo, hi], fmt="o", ms=2.4, lw=0.7,
                    color=col, alpha=0.85, zorder=3,
                    label=r"$s=%d$ (S/N med %.2f)" % (s, snr_med[s]))
        C = ec.evaluate_completeness(Ngrid, coef=coef, N0=N0,
                                     log10_snr=float(lsnr_med[s]), clamp=True)
        ax.plot(Ngrid, C, color=col, lw=1.2, zorder=2)
    ax.axvline(20.3, color="0.35", ls=":", lw=0.8)
    ax.text(20.32, 0.06, r"20.3", fontsize=6.5, color="0.35")
    ax.set_xlim(19.0, 22.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
    ax.set_ylabel(r"completeness $C_{\rm det}$")
    ax.set_title(r"(a) $C$ vs $N_{\rm HI}$ by S/N stratum"
                 "\n" r"points = raw $n_{\rm det}/n_{\rm tot}$ (Wilson $1\sigma$),"
                 r" line = C1nsadd")
    ax.legend(loc="lower right", frameon=False)

    # ---------------- (b) C vs S/N at representative N ----------------
    ax = axes[0, 1]
    snr = np.linspace(2.0, 20.0, 600)
    Nreps = [19.7, 20.0, 20.3, 20.6, 21.0]
    clamp_snr = float(m["snr_clamp_hi"])
    turn = float(rev["C2_extrapolation"]["snr_turning_point"])
    for i, Nv in enumerate(Nreps):
        col = cc.PALETTE[i]
        Ccl = ec.evaluate_completeness(np.full_like(snr, Nv), snr=snr,
                                       coef=coef, N0=N0, clamp=True)
        Cno = ec.evaluate_completeness(np.full_like(snr, Nv), snr=snr,
                                       coef=coef, N0=N0, clamp=False)
        ax.plot(snr, Ccl, color=col, lw=1.2,
                label=r"$\log N=%.1f$" % Nv)
        ax.plot(snr[snr > clamp_snr], Cno[snr > clamp_snr], color=col,
                lw=0.8, ls="--", alpha=0.75)
        Ccell = ec.evaluate_completeness(np.full(len(live), Nv), coef=coef,
                                         N0=N0, log10_snr=lsnr_med[live],
                                         clamp=True)
        ax.plot(snr_med[live], Ccell, "o", ms=2.6, color=col, zorder=4)
    ax.axvline(clamp_snr, color="k", lw=0.9)
    ax.axvline(turn, color="crimson", lw=0.9, ls="--")
    ax.text(clamp_snr - 0.22, 0.99, r"production clamp S/N $=10.683$"
            "\n" r"($\log_{10}$ S/N $=1.0287$)", rotation=90, ha="right",
            va="top", fontsize=5.8)
    ax.text(turn + 0.22, 0.99, r"turn-over S/N $=11.62$"
            "\n" r"(unphysical; never evaluated)", rotation=90, ha="left",
            va="top", fontsize=5.8, color="crimson")
    ax.set_xlim(2.0, 20.0)
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel(r"sightline S/N")
    ax.set_ylabel(r"completeness $C_{\rm det}$")
    ax.set_title(r"(b) $C$ vs S/N; dots = the six calibrated strata,"
                 "\n" r"dashed = unclamped form above the cap")
    ax.legend(loc="lower right", frameon=False, ncol=1)

    # ---------------- (c) surface + support ----------------
    ax = axes[1, 0]
    gN = m["grid_logN"]
    gS = m["grid_snr"]
    S = m["C_surface"]
    pm = ax.pcolormesh(gN, gS, S.T, cmap="viridis", vmin=0.0, vmax=1.0,
                       shading="auto", rasterized=True)
    cs = ax.contour(gN, gS, S.T, levels=[0.5, 0.8, 0.9, 0.95, 0.99],
                    colors="w", linewidths=0.55)
    ax.clabel(cs, fmt="%.2f", fontsize=5.5)
    d = dens[(dens.live_stratum == 1)]
    ax.scatter(0.5 * (d.logN_lo + d.logN_hi), d.snr_median,
               s=np.sqrt(np.maximum(d.n_truth.values, 0.0)) * 0.55,
               facecolor="none", edgecolor="w", lw=0.45, alpha=0.85)
    sp = d[d.n_truth < 20]
    ax.scatter(0.5 * (sp.logN_lo + sp.logN_hi), sp.snr_median, s=16,
               marker="x", color="crimson", lw=0.8, zorder=5)
    ax.axvline(21.7, color="crimson", ls="--", lw=0.8)
    ax.text(19.08, 19.4, r"$N\geq21.7$: shape extrapolation"
            "\n" r"(12 live cells $<20$ systems, 1 empty;"
            "\n" r"18 cells $<50$, from the $[21.5,21.7)$ bin up)",
            fontsize=5.8, color="crimson", va="top")
    ax.axhline(clamp_snr, color="k", lw=0.8)
    ax.set_xlim(19.0, 22.4)
    ax.set_ylim(2.0, 20.0)
    ax.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
    ax.set_ylabel(r"sightline S/N")
    ax.set_title(r"(c) fitted surface + calibration support"
                 "\n" r"(circle area $\propto n_{\rm truth}$; $\times$ = cell with $<20$)")
    fig.colorbar(pm, ax=ax, pad=0.015, fraction=0.045, label=r"$C_{\rm det}$")

    # ---------------- (d) (stratum, K) residuals of C1nsadd x g ----------------
    ax = axes[1, 1]
    res = rev["C5"]["effective_completeness_residual_pct_by_stratum_and_K"]
    strata = [2, 3, 4, 5, 6, 7]
    width = 0.26
    xs = np.arange(len(strata))
    for j, Kname in enumerate(["K0", "K1", "K2"]):
        vals = [res["s=%d" % s][Kname] for s in strata]
        ax.bar(xs + (j - 1) * width, vals, width=width * 0.94,
               color=cc.PALETTE[[0, 2, 4][j]],
               label=r"$K_%d$" % j, zorder=3)
    allk = [res["s=%d" % s]["all_K"] for s in strata]
    ax.plot(xs, allk, "kd", ms=4, zorder=5,
            label=r"all $K$ (stratum total)")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"$s=%d$" "\n" r"%.1f--%.1f" %
                        (s, float(dens[dens.s == s].snr_lo.iloc[0]),
                         min(float(dens[dens.s == s].snr_hi.iloc[0]), 99))
                        if s < 7 else r"$s=7$" "\n" r"$\geq 7$"
                        for s in strata], fontsize=6.2)
    ax.set_ylabel(r"$(C_{\rm 1nsadd}\times g) / C_{\rm truth} - 1$  [\%]"
                  .replace("\\%", "%"))
    ax.set_title(r"(d) the S/N $\times$ z structure the frozen $g$ does not carry"
                 "\n" r"effective-completeness residual by (stratum, coarse $z$)")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_ylim(-20, 20)

    fig.tight_layout(pad=0.7, w_pad=1.3, h_pad=1.6)
    os.makedirs(cc.FIGDIR, exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)

    # ---- console echo of the quantities the section quotes ----
    print("clamp S/N", clamp_snr, "turnover", turn)
    for Nv in Nreps:
        vals = ec.evaluate_completeness(np.full(len(live), Nv), coef=coef,
                                        N0=N0, log10_snr=lsnr_med[live],
                                        clamp=True)
        print("N=%.1f" % Nv, " ".join("%.4f" % v for v in vals))
    liv = dens[dens.live_stratum == 1]
    print("live cells", len(liv), "<50:", int((liv.n_truth < 50).sum()),
          "<20:", int((liv.n_truth < 20).sum()),
          "empty:", int((liv.n_truth == 0).sum()),
          "min N of <20 cells:", float(liv[liv.n_truth < 20].logN_lo.min()),
          "min n_truth below 21.6:",
          float(liv[liv.logN_lo < 21.6].n_truth.min()),
          "total truth:", float(liv.n_truth.sum()))


if __name__ == "__main__":
    main()
