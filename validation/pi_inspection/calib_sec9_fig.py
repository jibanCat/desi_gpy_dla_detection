"""SEC 9 figure: the FROZEN false-positive prior -- what is data and what is prior.

(a) the 89-event loa-0 FP template on the (N_hat x S/N) grid: counts per cell,
    25 of the 174 live cells populated, ZERO events at N_hat >= 20.3.
(b) the FP mass the model places at N_hat >= 20.3 as a function of the Perks
    pseudo-count a0 -- 100 % prior regularisation, reproduced in closed form
    from the released template.
(c) the headline mock-bias shifts of the a0 battery (A0_BATTERY_TABLE.md).

Read-only: released fp_template.npz + the published battery table.  No fit.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_common as cc                                        # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import LogNorm                            # noqa: E402

FP = os.path.join(cc.REL, "fp")
OUT = os.path.join(cc.FIGDIR, "fig_sec9_fp_template_and_a0.png")

# A0_BATTERY_TABLE.md (seed 20260811, B + phi_2LPT + C1nsadd under M1CUT)
A0 = [0.0, 0.00143678, 0.00574713, 0.02298851, 0.5]
A0LAB = [r"$0$", r"$1/(4K)$", r"$1/K$" "\n" r"(record)", r"$4/K$",
         r"$1/2$" "\n" r"(Jeffreys)"]
BIAS20 = {"2LPT-0": [-0.31, -0.35, -0.38, -0.54, -1.18],
          "London-0": [+0.15, +0.14, +0.13, +0.01, -0.61],
          "Saclay-0": [-0.06, -0.09, -0.10, -0.23, -0.90]}
BIAS203 = {"2LPT-0": [-0.23, -0.27, -0.33, -0.53, -1.16],
           "London-0": [-0.70, -0.72, -0.72, -0.86, -1.57],
           "Saclay-0": [-0.20, -0.23, -0.29, -0.45, -1.15]}
# FINAL_HBI_..._MATHEMATICS.md sec 6.3, recomputed at t = 0, Lambda = median
MEMO_COUNTS = [0.0, 30.01, 119.05, 460.82, 5296.17]
TRUTH = {"2LPT-0": 149, "London-0": 137, "Saclay-0": 76}
FPW_LEFF = 2255.0
LAM_MED = 6.561265431972927


def main():
    cc.style()
    t = np.load(os.path.join(FP, "fp_template.npz"), allow_pickle=True)
    n = t["fp_counts"].astype(float)              # (29, 8)
    nh = t["nhat_edges"]
    se = t["snr_edges"]
    live_s = t["live_stratum"].astype(bool)
    K = int(t["K_live_cells"])
    NFP = int(t["n_fp_events"])
    live = np.zeros_like(n, dtype=bool)
    live[:, live_s] = True
    print("N_FP", NFP, "= sum", n.sum(), "| live cells", K,
          "=", int(live.sum()), "| populated", int((n[live] > 0).sum()))
    c203 = int(np.where(np.isclose(nh, 20.3))[0][0])
    print("events at Nhat >= 20.3:", float(n[c203:][live[c203:]].sum()),
          "| live cells at Nhat >= 20.3:", int(live[c203:].sum()))

    fig = plt.figure(figsize=(11.2, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.05], wspace=0.42)

    # ---------------- (a) the template ----------------
    ax = fig.add_subplot(gs[0, 0])
    sedge = np.r_[se[:-1], 9.0]                  # render the open >=7 stratum
    m = np.ma.masked_where(~live | (n == 0), n)
    pm = ax.pcolormesh(nh, sedge, m.T, cmap="viridis",
                       norm=LogNorm(vmin=1, vmax=max(n.max(), 2)),
                       shading="flat")
    dead = np.ma.masked_where(live, np.ones_like(n))
    ax.pcolormesh(nh, sedge, dead.T, cmap="Greys", vmin=0, vmax=2,
                  shading="flat", alpha=0.55)
    for ci in range(n.shape[0]):
        for si in range(n.shape[1]):
            if live[ci, si] and n[ci, si] > 0:
                ax.text(0.5 * (nh[ci] + nh[ci + 1]),
                        0.5 * (sedge[si] + sedge[si + 1]), "%d" % n[ci, si],
                        ha="center", va="center", fontsize=4.6,
                        color="w" if n[ci, si] < 8 else "k")
    ax.axvline(20.3, color="crimson", lw=1.3)
    ax.text(20.42, 8.6, "$\\hat{N}\\geq 20.3$: 0 of 89 events\n"
            "126 live cells, 100 % pseudo-count",
            color="crimson", fontsize=5.8, va="top")
    ax.set_xlim(19.5, 22.4)
    ax.set_ylim(0, 9)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_yticklabels(["0", "1", "2", "3", "4", "5", "6", "7", r"$\infty$", ""])
    ax.set_xlabel(r"observed $\log_{10}\hat{N}$")
    ax.set_ylabel(r"sightline S/N (grey = dead strata)")
    ax.set_title(r"(a) loa-0 FP template: 89 events,"
                 "\n" r"25 of 174 live cells populated", fontsize=8.0)
    ax.grid(False)
    fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.045,
                 label=r"FP events per cell")
    ax.figure.axes[-1].yaxis.label.set_size(6.5)

    # ---------------- (b) the >= 20.3 block vs a0 ----------------
    ax = fig.add_subplot(gs[0, 1])
    nlive203 = int(live[c203:].sum())
    a = np.logspace(-4, np.log10(0.5), 300)
    counts = FPW_LEFF * LAM_MED * nlive203 * a / (NFP + K * a)
    ax.plot(a, counts, color=cc.PALETTE[1], lw=1.3, zorder=2,
            label=r"closed form: $f_{\rm w}\ell_{\rm eff}\Lambda_{\rm med}"
                  r"\,K_{\geq}a_0/(N_{\rm FP}+Ka_0)$")
    chk = FPW_LEFF * LAM_MED * nlive203 * np.array(A0) / (NFP + K * np.array(A0))
    print("reproduced a0 battery counts:", np.round(chk, 2),
          "vs memo", MEMO_COUNTS,
          "max rel diff (a0>0):",
          float(np.max(np.abs(chk[1:] / np.array(MEMO_COUNTS[1:]) - 1))))
    ax.plot(A0[1:], MEMO_COUNTS[1:], "o", ms=4.5, color="crimson", zorder=4,
            label=r"battery values (\S6.3 memo)".replace("\\S", "sec. "))
    for fam, col in zip(TRUTH, [cc.PALETTE[0], cc.PALETTE[3], cc.PALETTE[6]]):
        ax.axhline(TRUTH[fam], color=col, ls="--", lw=0.8)
        ax.text(0.55, TRUTH[fam], r"%s: %d" % (fam, TRUTH[fam]),
                fontsize=5.2, color=col, ha="right",
                va="bottom" if fam != "London-0" else "top")
    ax.axvline(1.0 / K, color="k", lw=0.8)
    ax.text(1.0 / K * 1.12, 2.0, r"record $a_0=1/K$", fontsize=6.0, rotation=90)
    ax.axvspan(0.25 / K, 4.0 / K, color="0.5", alpha=0.16, lw=0,
               label=r"predeclared factor-4 bracket")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 0.6)
    ax.set_ylim(1.0, 1e4)
    ax.set_xlabel(r"Perks pseudo-count $a_0$")
    ax.set_ylabel(r"model FP counts at $\hat{N}\geq 20.3$")
    ax.set_title(r"(b) the $\hat{N}\geq 20.3$ FP block is"
                 "\n" r"100\% prior regularisation".replace("\\%", "%"),
                 fontsize=8.0)
    ax.text(0.55, 2.4e2, "mock FP truth", fontsize=5.2, ha="right",
            color="0.3")
    ax.legend(frameon=False, loc="upper left", fontsize=5.4)

    # ---------------- (c) headline shifts ----------------
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(A0))
    for i, fam in enumerate(BIAS20):
        col = [cc.PALETTE[0], cc.PALETTE[3], cc.PALETTE[6]][i]
        ax.plot(x, BIAS20[fam], "-o", ms=3.2, color=col, lw=1.0,
                label=r"%s, $\geq 20.0$" % fam)
        ax.plot(x, BIAS203[fam], "--s", ms=3.0, color=col, lw=0.9, alpha=0.75,
                label=r"%s, $\geq 20.3$" % fam)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvspan(0.5, 3.5, color="0.5", alpha=0.16, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels(A0LAB, fontsize=6.0)
    ax.set_xlabel(r"Perks pseudo-count $a_0$")
    ax.set_ylabel(r"mock $\mathrm{d}N/\mathrm{d}X$ bias  [pp]")
    ax.set_title(r"(c) headline shift: $\leq 0.16/0.20$ pp inside"
                 "\n" r"the bracket, $0.5$--$1.5$ pp at Jeffreys", fontsize=8.0)
    ax.legend(frameon=False, ncol=1, fontsize=5.2, loc="lower left")

    fig.subplots_adjust(left=0.052, right=0.995, top=0.855,
                        bottom=0.155)
    os.makedirs(cc.FIGDIR, exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
