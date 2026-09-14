#!/usr/bin/env python
"""PI inspection packet (2026-09-14) — figures.  READ-ONLY; plots stored numbers only."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.linewidth": 0.7, "savefig.dpi": 130,
    "figure.dpi": 130, "savefig.bbox": "tight", "mathtext.fontset": "dejavuserif"})

C_OLD, C_NEW = "#6f6f6f", "#2c6fbb"
C_T0, C_T3 = "#2c6fbb", "#c8571b"
FAMC = {"2lpt0": "#2c6fbb", "london0": "#1a9e77", "saclay0": "#c8571b"}

D = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def med(qq):
    return qq[2], qq[2] - qq[1], qq[3] - qq[2], qq[2] - qq[0], qq[4] - qq[2]


# ------------------------------------------------------------------ SEC 2 (N)
def fig_sec2():
    nb, ob = D["new_rep_bins"], D["old_rep_bins"]
    x = np.array([0.5 * (b["bin"][0] + b["bin"][1]) for b in nb])
    nm = np.array([b["q"][2] for b in nb]); om = np.array([b["q"][2] for b in ob])
    nlo = nm - np.array([b["q"][1] for b in nb]); nhi = np.array([b["q"][3] for b in nb]) - nm
    olo = om - np.array([b["q"][1] for b in ob]); ohi = np.array([b["q"][3] for b in ob]) - om
    nhw = 0.5 * (np.array([b["q"][3] for b in nb]) - np.array([b["q"][1] for b in nb]))
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained")
    a = ax[0]
    a.errorbar(x - 0.022, om, yerr=[olo, ohi], fmt="o", ms=3.4, mfc="none", lw=0.9,
               capsize=1.8, color=C_OLD, label="old C1 (pre-rebuild)")
    a.errorbar(x + 0.022, nm, yerr=[nlo, nhi], fmt="s", ms=3.4, lw=0.9,
               capsize=1.8, color=C_NEW, label="frozen new C1")
    a.set_yscale("log"); a.set_xlabel(r"$\log_{10} N_{\rm HI}$ bin centre")
    a.set_ylabel(r"$dN/dX$ per 0.2-dex bin")
    for xv, lab in ((20.0, "20.0"), (20.3, "20.3")):
        a.axvline(xv, color="k", lw=0.6, ls=":" if lab == "20.0" else "--")
    a.legend(loc="lower left", frameon=False)
    a.set_title("(a) posterior per reporting bin, 68 % bars", fontsize=8.2)
    b = ax[1]
    frac = 100.0 * (nm / om - 1.0)
    cols = [C_NEW if f >= 0 else C_T3 for f in frac]
    b.bar(x, frac, width=0.16, color=cols, edgecolor="k", linewidth=0.4)
    for xi, fi, hi in zip(x, frac, (nm - om) / nhw):
        b.annotate(f"{hi:+.1f}", (xi, fi), textcoords="offset points",
                   xytext=(0, 3 if fi >= 0 else -9), ha="center", fontsize=6)
    b.axhline(0, color="k", lw=0.7)
    for xv in (20.0, 20.3):
        b.axvline(xv, color="k", lw=0.6, ls=":" if xv == 20.0 else "--")
    b.set_xlabel(r"$\log_{10} N_{\rm HI}$ bin centre")
    b.set_ylabel(r"new $-$ old  [\% of old]" if False else "new - old  [% of old]")
    b.margins(y=0.18)
    b.set_title("(b) change per bin\n(labels: change / new 68 % half-width)", fontsize=8.2)
    fig.suptitle("SEC 2 — old C1 vs frozen new C1, all-z, by column density", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_sec2_old_vs_new_by_N.png")); plt.close(fig)


# ------------------------------------------------------------------ SEC 2b (z)
def fig_sec2b():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained")
    names = [b["bin"] for b in D["new_zbins"]["ge20.0"]]
    xi = np.arange(len(names))
    for j, (t, c) in enumerate((("ge20.0", C_T0), ("ge20.3", C_T3))):
        nm = np.array([b["q"][2] for b in D["new_zbins"][t]])
        om = np.array([b["q"][2] for b in D["old_zbins"][t]])
        nlo = nm - np.array([b["q"][1] for b in D["new_zbins"][t]])
        nhi = np.array([b["q"][3] for b in D["new_zbins"][t]]) - nm
        olo = om - np.array([b["q"][1] for b in D["old_zbins"][t]])
        ohi = np.array([b["q"][3] for b in D["old_zbins"][t]]) - om
        nhw = 0.5 * (np.array([b["q"][3] for b in D["new_zbins"][t]])
                     - np.array([b["q"][1] for b in D["new_zbins"][t]]))
        lab = r"$\geq 20.0$" if t == "ge20.0" else r"$\geq 20.3$"
        ax[0].errorbar(xi - 0.09, om, yerr=[olo, ohi], fmt="o", ms=3.4, mfc="none",
                       lw=0.9, capsize=1.8, color=c, label=f"old {lab}")
        ax[0].errorbar(xi + 0.09, nm, yerr=[nlo, nhi], fmt="s", ms=3.4, lw=0.9,
                       capsize=1.8, color=c, label=f"new {lab}")
        ax[1].bar(xi + (j - 0.5) * 0.34, 100 * (nm / om - 1), width=0.32, color=c,
                  edgecolor="k", linewidth=0.4, label=lab)
        for k in range(len(names)):
            ax[1].annotate(f"{(nm[k]-om[k])/nhw[k]:+.1f}", (xi[k] + (j - 0.5) * 0.34,
                           100 * (nm[k] / om[k] - 1)), textcoords="offset points",
                           xytext=(0, 3 if nm[k] >= om[k] else -9), ha="center", fontsize=6)
    lbl = [f"{n}\n{b['z'][0]:.2f}-{b['z'][1]:.2f}" for n, b in zip(names, D["new_zbins"]["ge20.0"])]
    for a in ax:
        a.set_xticks(xi); a.set_xticklabels(lbl, fontsize=6.5)
    ax[0].set_ylabel(r"$dN/dX$"); ax[0].legend(frameon=False, ncol=2, fontsize=6.5)
    ax[0].set_title("(a) Paper-1 z bins, 68 % bars", fontsize=8.2)
    ax[1].axhline(0, color="k", lw=0.7); ax[1].legend(frameon=False)
    ax[1].set_ylabel("new - old  [% of old]")
    ax[1].margins(y=0.18)
    ax[1].set_title("(b) change per z bin\n(labels: change / new 68 % half-width)", fontsize=8.2)
    ax[0].annotate("B5: 25 % covered", (4, ax[0].get_ylim()[0]), fontsize=6.5,
                   ha="center", va="bottom", color="0.35")
    fig.suptitle("SEC 2b — old C1 vs frozen new C1, by Paper-1 redshift bin", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_sec2b_old_vs_new_by_z.png")); plt.close(fig)


# ------------------------------------------------------------------ SEC 3
def fig_sec3():
    nb = D["new_rep_bins"]
    x = np.array([0.5 * (b["bin"][0] + b["bin"][1]) for b in nb])
    m = np.array([b["q"][2] for b in nb])
    q = np.array([b["q"] for b in nb])
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axvspan(20.3, 21.6, color="#f2c14e", alpha=0.18, lw=0,
               label=r"$\Omega$ window [20.3, 21.6]")
    ax.errorbar(x, m, yerr=[m - q[:, 0], q[:, 4] - m], fmt="none", lw=0.7,
                capsize=1.6, color="#7fa9d6", label="95 %")
    ax.errorbar(x, m, yerr=[m - q[:, 1], q[:, 3] - m], fmt="s", ms=3.6, lw=1.4,
                capsize=2.2, color=C_NEW, label="median, 68 %")
    ax.axvline(20.0, color="k", lw=0.7, ls=":")
    ax.axvline(20.3, color="k", lw=0.7, ls="--")
    ax.annotate("20.0", (20.0, m.max() * 1.25), fontsize=7, ha="center")
    ax.annotate("20.3", (20.3, m.max() * 1.25), fontsize=7, ha="center")
    ax.set_yscale("log"); ax.set_ylim(top=m.max() * 2.2)
    ax.set_xlabel(r"$\log_{10} N_{\rm HI}$ bin centre (0.2-dex reporting bins)")
    ax.set_ylabel(r"$dN/dX$ per bin (all $z$)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("SEC 3 — real pooled posterior per reporting bin", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_sec3_real_reporting_bins.png")); plt.close(fig)


# ------------------------------------------------------------------ SEC 4
def fig_sec4():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1), layout="constrained")
    names = [b["bin"] for b in D["new_zbins"]["ge20.0"]]
    xi = np.arange(len(names))
    for t, c, lab in (("ge20.0", C_T0, r"$\geq 20.0$"), ("ge20.3", C_T3, r"$\geq 20.3$")):
        qq = np.array([b["q"] for b in D["new_zbins"][t]])
        ax[0].errorbar(xi + (0.07 if t == "ge20.3" else -0.07), qq[:, 2],
                       yerr=[qq[:, 2] - qq[:, 1], qq[:, 3] - qq[:, 2]], fmt="s-",
                       ms=3.6, lw=1.0, capsize=2.0, color=c, label=lab)
    ax[0].axvspan(3.5, 4.5, color="0.85", alpha=0.6, lw=0)
    ax[0].set_xlim(-0.5, 4.5)
    ax[0].set_ylabel(r"$dN/dX$ (not bias-corrected)")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("(a) real pooled, Paper-1 z bins", fontsize=8.5)
    ax[0].annotate("B5: only\n[3.40, 3.50)\ncovered (25 %)", (4, ax[0].get_ylim()[0]),
                   fontsize=6.5, ha="center", va="bottom", color="0.3")
    orc = D["oracle_f1_zbins"]; m1 = D["m1cut_J8_zbins_mean_over_j"]
    w = 0.26
    for fi, fam in enumerate(("2lpt0", "london0", "saclay0")):
        k = f"{fam}_s20260811"
        for t, hatch, alpha in (("ge20.0", "", 0.45), ("ge20.3", "///", 1.0)):
            v = [b["median_bias_pct"] or 0.0 for b in orc[k][t]]
            off = (fi - 1) * w + (0.055 if t == "ge20.3" else -0.055)
            ax[1].bar(xi + off, v, width=0.10, color=FAMC[fam], alpha=alpha,
                      edgecolor="k", linewidth=0.3, hatch=hatch,
                      label=(f"{fam} " + (r"$\geq 20.3$" if t == "ge20.3" else r"$\geq 20.0$")))
        ax[1].plot(xi + (fi - 1) * w + 0.055, m1[fam]["ge20.3"], "k_", ms=5, mew=1.0,
                   label="M1CUT J8 mean" if fi == 0 else None)
    ax[1].axhline(0, color="k", lw=0.7)
    for y in (-2, 2):
        ax[1].axhline(y, color="0.6", lw=0.5, ls=":")
    ax[1].set_ylabel("mock transfer residual, median bias [%]")
    ax[1].legend(frameon=False, ncol=2, fontsize=5.6, loc="lower center")
    ax[1].margins(y=0.24)
    ax[1].set_title("(b) ORACLE F1 signed residuals (bars); M1CUT J8 mean (tick)", fontsize=8.5)
    lbl = [f"{n}\n{b['z'][0]:.2f}-{b['z'][1]:.2f}" for n, b in zip(names, D["new_zbins"]["ge20.0"])]
    for a in ax:
        a.set_xticks(xi); a.set_xticklabels(lbl, fontsize=6.5)
    fig.suptitle("SEC 4 — real redshift evolution and the mock transfer limitation", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_sec4_real_z_bins_and_transfer.png")); plt.close(fig)


# ------------------------------------------------------------------ SEC 8
def fig_sec8():
    sets = [("real s20260811", [r for r in D["real_runs"] if r["seed"] == 20260811], "#2c6fbb", "o"),
            ("real s20260812", [r for r in D["real_runs"] if r["seed"] == 20260812], "#1a9e77", "s"),
            ("mock 2LPT-0 J8", D["mock_2lpt0_J8"], "#c8571b", "^")]
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.0), layout="constrained")
    for lab, rows, c, mk in sets:
        L = np.log(np.array([r["lam"] for r in rows]))
        T = np.array([r["t_mean"] for r in rows])
        for K, ls in enumerate(("-", "--", ":")):
            ax[0].plot(L, T[:, K], ls, marker=mk, ms=2.8, lw=0.9, color=c,
                       label=f"{lab}, $K_{K}$" if K == 0 else None)
        LE = np.array([[r["lam"] * np.exp(x) for x in r["t_mean"]] for r in rows])
        jj = np.array([r["j"] for r in rows])
        for K, ls in enumerate(("-", "--", ":")):
            ax[1].plot(jj, LE[:, K] / LE[:, K].mean(), ls, marker=mk, ms=2.8, lw=0.9, color=c)
        ft = np.array([r["mu_fp_total"][1] for r in rows])
        ax[1].plot(jj, ft / ft.mean(), "-", marker=mk, ms=3.4, lw=1.6, color=c, alpha=0.45)
        for t, ls in (("ge20.0", "-"), ("ge20.3", "--")):
            m = np.array([r["thresholds"][t]["median"] for r in rows])
            hw = np.mean([r["thresholds"][t]["hw"] for r in rows])
            ax[2].plot(jj, (m - m.mean()) / hw, ls, marker=mk, ms=2.8, lw=0.9, color=c)
    lam = np.array([r["lam"] for r in D["real_runs"] if r["seed"] == 20260811])
    ax[0].set_xlabel(r"$\ln \Lambda_j$"); ax[0].set_ylabel(r"$t_K$ posterior mean")
    ax[0].set_title(r"(a) $t_K$ vs $\ln\Lambda_j$: $K_0$ solid, $K_1$ dashed, $K_2$ dotted",
                    fontsize=7.2)
    ax[0].legend(frameon=False, fontsize=6.2, loc="lower left")
    ax[0].margins(y=0.16)
    ax[1].axhline(1, color="k", lw=0.7)
    ax[1].set_xlabel("imputation $j$")
    ax[1].set_ylabel(r"value / its mean over $j$", labelpad=1)
    ax[1].set_title(r"(b) $\Lambda_j e^{t_K}$ per block (thin) and posterior"
                    "\n" r"$\mu_{\rm FP}$ total (thick, faded)", fontsize=7.2)
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].fill_between([-0.4, 7.4], -0.25, 0.25, color="0.85", alpha=0.7, lw=0)
    ax[2].set_xlim(-0.4, 7.4)
    ax[2].set_xlabel("imputation $j$")
    ax[2].set_ylabel(r"$(x_j-\bar{x})$ / 68 % half-width", labelpad=1)
    ax[2].set_title(r"(c) $dN/dX$ $\geq$20.0 (solid), $\geq$20.3 (dashed)"
                    "\n" r"grey band $\pm 0.25$ half-width", fontsize=7.2)
    fig.suptitle(r"SEC 8 — the M1CUT $\Lambda$–$t_K$ compensation ridge "
                 rf"($\Lambda_j$ spans {lam.min():.2f}–{lam.max():.2f}, a factor "
                 rf"{lam.max()/lam.min():.2f})", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig_sec8_lambda_tK_ridge.png")); plt.close(fig)


for fn in (fig_sec2, fig_sec2b, fig_sec3, fig_sec4, fig_sec8):
    fn(); print("ok", fn.__name__)
