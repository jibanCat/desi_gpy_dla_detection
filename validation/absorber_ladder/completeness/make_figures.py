#!/usr/bin/env python
"""make_figures.py — figures for the FIXED completeness calibration variants.

Reads ``completeness_variants_summary.json`` + ``completeness_observed.npz``
from the products directory and writes PNGs to the notes repo.
VALIDATION-ONLY.  ENV: ``gpdla`` or ``gpdla-hbi`` (numpy + matplotlib).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

FAMILIES = ("2lpt0", "london0", "saclay0")
FAMLAB = {"2lpt0": "2LPT-0 (calibration)", "london0": "London-0 (transfer)",
          "saclay0": "Saclay-0 (transfer)"}
FAMCOL = {"2lpt0": "#440154", "london0": "#21918c", "saclay0": "#bb3754"}
VARCOL = {"C0": "#000000", "C1g": "#7a7a7a", "C1gz": "#3b528b",
          "C1n": "#21918c", "C1ns": "#e0632b", "C1nsadd": "#5ec962",
          "C1nsz": "#8c2981"}
VARLAB = {"C0": "C0 frozen molly", "C1g": "C1g fine grid",
          "C1gz": "C1gz fine grid, z-resolved", "C1n": "C1n smooth in N",
          "C1ns": "C1ns 2-D tensor", "C1nsadd": "C1ns additive",
          "C1nsz": "C1nsz additive + z (exploratory)"}
KLAB = ("K0  z 2.0-2.5", "K1  z 2.5-3.0", "K2  z 3.0-3.5")


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False})


def load(prod):
    S = json.load(open(os.path.join(prod,
                                    "completeness_variants_summary.json")))
    Z = np.load(os.path.join(prod, "completeness_observed.npz"))
    return S, Z


# ---------------------------------------------------------------------------
def fig_c_vs_n(S, Z, out):
    live = np.asarray(Z["live_idx"], int)
    snr_e = np.asarray(Z["snr_edges"], float)
    Nc = np.asarray(Z["Nc"], float)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.6), sharex=True,
                             sharey=True)
    for i, ax in enumerate(axes.ravel()):
        s = live[i]
        lab = (f"S/N [{snr_e[s]:.0f}, {snr_e[s+1]:.0f})" if np.isfinite(snr_e[s + 1])
               else f"S/N >= {snr_e[s]:.0f}")
        for fam in FAMILIES:
            d = np.asarray(Z[f"det_{fam}"], float)[i]
            t = np.asarray(Z[f"tot_{fam}"], float)[i]
            ok = t > 0
            p = np.where(ok, d / np.maximum(t, 1), np.nan)
            e = np.where(ok, np.sqrt(np.maximum(p * (1 - p), 0) / np.maximum(t, 1)),
                         np.nan)
            ax.errorbar(Nc[ok], p[ok], yerr=e[ok], fmt="o", ms=2.6, lw=0.8,
                        capsize=1.6, color=FAMCOL[fam], alpha=0.85,
                        label=f"truth {FAMLAB[fam].split()[0]}" if i == 0 else None)
        for v in ("C0", "C1n", "C1ns", "C1nsadd"):
            C = np.asarray(S["variants"][v]["C_fixed"], float)[s]
            ax.plot(Nc, C, "-" if v != "C0" else "--", lw=1.6 if v == "C0" else 1.2,
                    color=VARCOL[v], label=VARLAB[v] if i == 0 else None,
                    drawstyle="steps-mid" if v == "C0" else "default")
        ax.set_title(lab, fontsize=8.5)
        ax.set_ylim(0.0, 1.05)
        if i >= 3:
            ax.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
        if i % 3 == 0:
            ax.set_ylabel(r"detection probability $C_{\rm det}$")
        ax.axvline(19.5, color="0.5", lw=0.6, ls=":")
    axes[0, 0].legend(fontsize=6.4, ncol=2, loc="lower right")
    fig.suptitle("Completeness variants vs true N, per S/N stratum "
                 "(points = each mock's own matched truth; C0 is a step "
                 "function on the 12 molly cells)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_residual_map(S, Z, out):
    live = np.asarray(Z["live_idx"], int)
    snr_e = np.asarray(Z["snr_edges"], float)
    Nc = np.asarray(Z["Nc"], float)
    vs = ["C0", "C1g", "C1n", "C1nsadd"]
    fig, axes = plt.subplots(len(vs), 3, figsize=(12.4, 2.35 * len(vs)),
                             sharex=True, sharey=True)
    for r, v in enumerate(vs):
        C = np.asarray(S["variants"][v]["C_fixed"], float)[live]
        for c, fam in enumerate(FAMILIES):
            d = np.asarray(Z[f"det_{fam}"], float)
            t = np.asarray(Z[f"tot_{fam}"], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                res = 100.0 * (C * t / np.where(d > 0, d, np.nan) - 1.0)
            ax = axes[r, c]
            im = ax.pcolormesh(np.arange(len(Nc) + 1) - 0.5,
                               np.arange(len(live) + 1) - 0.5,
                               res, cmap="RdBu_r", vmin=-12, vmax=12,
                               shading="flat")
            ax.set_yticks(range(len(live)))
            ax.set_yticklabels([f"{snr_e[s]:.0f}-"
                               + (f"{snr_e[s+1]:.0f}" if np.isfinite(snr_e[s+1])
                                  else "inf") for s in live], fontsize=6.5)
            ax.set_xticks(range(0, len(Nc), 2))
            ax.set_xticklabels([f"{Nc[b]:.1f}" for b in range(0, len(Nc), 2)],
                               fontsize=6.5)
            ax.grid(False)
            if r == 0:
                ax.set_title(FAMLAB[fam], fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{v}\nS/N stratum", fontsize=7.5)
            if r == len(vs) - 1:
                ax.set_xlabel(r"true $\log_{10} N_{\rm HI}$ bin centre")
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cb.set_label("expected / observed detections - 1  [%]", fontsize=8)
    fig.suptitle("Completeness residual maps: a FIXED 2LPT-0 calibration "
                 "object against each family's own matched truth", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cv(S, out):
    cv = S["cv_curve"]
    fams = {"C1n_deg": ("C1n  per-stratum poly in N", "o-", "#21918c"),
            "C1ns_tensor": ("C1ns  2-D tensor in (N, log S/N)", "s-", "#e0632b"),
            "C1ns_add": ("C1ns  additive in logit", "^-", "#5ec962")}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axes[0]
    L0 = cv["C0_frozen"]["cv_logloss"]
    for pref, (lab, mk, col) in fams.items():
        rows = sorted([(v["n_coef"], v["cv_logloss"]) for k, v in cv.items()
                       if k.startswith(pref)])
        ax.plot([r[0] for r in rows], [r[1] - L0 for r in rows], mk, ms=4,
                lw=1.2, color=col, label=lab)
    ax.axhline(0.0, color="k", lw=1.2, ls="--", label="C0 frozen (reference)")
    g = cv["C1g"]
    ax.plot([g["n_coef"]], [g["cv_logloss"] - L0], "D", ms=6, color="#7a7a7a",
            label=f"C1g free grid ({g['n_coef']} cells)")
    ax.set_xscale("log")
    ax.set_ylim(-750.0, 220.0)
    for pref, (lab, mk, col) in fams.items():
        for k, v in cv.items():
            if k.startswith(pref) and v["cv_logloss"] - L0 > 220.0:
                ax.annotate(f"{k.replace('C1n_','')}: {v['cv_logloss']-L0:+.0f}",
                            (v["n_coef"], 200.0), color=col, fontsize=6.0,
                            ha="center", va="bottom",
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
                            xytext=(v["n_coef"], 120.0))
    ax.set_xlabel("number of fitted coefficients")
    ax.set_ylabel("held-out log-loss - C0  [nats]")
    ax.set_title("CV curve (sightline halves, 2LPT-0)\n"
                 "lower = better; off-scale points annotated", fontsize=8.5)
    ax.legend(fontsize=6.6, loc="center right", bbox_to_anchor=(1.0, 0.42))

    ax = axes[1]
    rows = sorted([(v["n_coef"], v["cv_logloss"], k) for k, v in cv.items()
                   if k.startswith("C1ns")])
    for k, v in cv.items():
        if not k.startswith("C1ns"):
            continue
        du = int(k.split("x")[-1])
        ax.plot(v["n_coef"], v["cv_logloss"] - L0,
                "o" if "tensor" in k else "^", ms=5,
                color={1: "#bb3754", 2: "#21918c", 3: "#3b528b"}[du])
        ax.annotate(k.replace("C1ns_", ""), (v["n_coef"], v["cv_logloss"] - L0),
                    fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    for du, col in ((1, "#bb3754"), (2, "#21918c"), (3, "#3b528b")):
        ax.plot([], [], "o", color=col, label=f"degree {du} in log S/N")
    ax.axhline(0.0, color="k", lw=1.0, ls="--")
    ax.axhline(cv["C1g"]["cv_logloss"] - L0, color="#7a7a7a", lw=1.0, ls=":",
               label="C1g free grid")
    ax.set_xlabel("number of fitted coefficients")
    ax.set_ylabel("held-out log-loss - C0  [nats]")
    ax.set_title("Is the S/N smoothness data-supported?\n"
                 "quadratic in log S/N is REQUIRED", fontsize=8.5)
    ax.legend(fontsize=6.6)

    ax = axes[2]
    zc = S["cv_curve_zlayout"]
    keys = ["z_pooled", "C1nsz", "C1gz"]
    labs = ["z-pooled\n(96 cells)", "additive + 2 z\noffsets (8 coef)",
            "free (b,K,s)\ngrid (%d cells)" % zc["C1gz"]["n_coef"]]
    base = zc["z_pooled"]["cv_logloss"]
    vals = [zc[k]["cv_logloss"] - base for k in keys]
    ax.bar(range(3), vals, color=["#7a7a7a", "#8c2981", "#3b528b"], width=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.0f}", (i, v), ha="center", va="top", fontsize=7.5,
                    xytext=(0, -3), textcoords="offset points")
    ax.set_ylim(min(vals) * 1.18, 60.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("held-out log-loss - z-pooled  [nats]")
    ax.set_title("Coarse-z resolution, scored on (s, K, b)\n"
                 "lower = better", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_by_k(S, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=True)
    vs = ["C0", "C1g", "C1n", "C1ns", "C1nsadd", "C1gz", "C1nsz"]
    w = 0.11
    for ax, fam in zip(axes, FAMILIES):
        for i, v in enumerate(vs):
            y = [100.0 * x for x in S["transfer"][v][fam]["by_K"]]
            ax.bar(np.arange(3) + (i - len(vs) / 2 + 0.5) * w, y, width=w,
                   color=VARCOL[v], label=VARLAB[v] if fam == "2lpt0" else None)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(KLAB, fontsize=7)
        ax.set_title(FAMLAB[fam], fontsize=8.5)
    axes[0].set_ylabel("expected / observed detections - 1  [%]")
    axes[0].legend(fontsize=6.2, ncol=2)
    fig.suptitle("Coarse-z (K) residual of each FIXED completeness object — "
                 "every z-pooled variant leaves the same +5 / -5 / -6 % "
                 "structure", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_sparse(S, Z, out):
    live = np.asarray(Z["live_idx"], int)
    Nc = np.asarray(Z["Nc"], float)
    snr_e = np.asarray(Z["snr_edges"], float)
    tot = np.asarray(S["sparse"]["n_tot"], float)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ax = axes[0]
    for i, s in enumerate(live):
        ax.semilogy(Nc, np.maximum(tot[i], 0.5), "o-", ms=3, lw=1.0,
                    label=f"S/N {snr_e[s]:.0f}-"
                          + (f"{snr_e[s+1]:.0f}" if np.isfinite(snr_e[s + 1])
                             else "inf"))
    ax.axhline(50, color="0.4", ls=":", lw=1.0)
    ax.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
    ax.set_ylabel("truth systems in the cell (2LPT-0)")
    ax.set_title("Calibration cell occupancy", fontsize=8.5)
    ax.legend(fontsize=6.4, ncol=2)
    ax = axes[1]
    for v, mk in (("C1g", "o"), ("C1n", "s"), ("C1nsadd", "^"), ("C0", "d")):
        sd = np.asarray(S["variants"][v]["C_fixed_sd"], float)[live]
        ax.semilogy(Nc, np.maximum(sd.mean(axis=0), 1e-6), mk + "-", ms=3,
                    lw=1.0, color=VARCOL[v], label=VARLAB[v])
    ax.set_xlabel(r"true $\log_{10} N_{\rm HI}$")
    ax.set_ylabel(r"stratum-mean calibration sd of $C$")
    ax.set_title("Calibration uncertainty: the smooth variants BORROW\n"
                 "strength into the sparse high-N cells", fontsize=8.5)
    ax.legend(fontsize=6.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_p6b(S, out):
    p6 = S["p6b"]["families"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    ax.bar(range(3), [100 * p6[f]["residual_total"] for f in FAMILIES],
           color=[FAMCOL[f] for f in FAMILIES], width=0.55)
    for i, f in enumerate(FAMILIES):
        ax.annotate(f"{100*p6[f]['residual_total']:+.1f} %\n"
                    f"({p6[f]['poisson_sigma_total']:+.1f}$\\sigma_P$)",
                    (i, 100 * p6[f]["residual_total"]), ha="center",
                    va="bottom", fontsize=7,
                    xytext=(0, 3), textcoords="offset points")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylim(-2.0, 1.32 * max(100 * p6[f]["residual_total"]
                                 for f in FAMILIES))
    ax.set_xticks(range(3))
    ax.set_xticklabels([FAMLAB[f] for f in FAMILIES], fontsize=7)
    ax.set_ylabel("predicted / observed P6b events - 1  [%]")
    ax.set_title("P6b sub-floor-host rate: transfer of the fixed\n"
                 "2LPT-0 rate per unit path", fontsize=8.5)
    ax = axes[1]
    w = 0.26
    for i, f in enumerate(FAMILIES):
        ax.bar(np.arange(3) + (i - 1) * w,
               [100 * x for x in p6[f]["residual_by_K"]], width=w,
               color=FAMCOL[f], label=FAMLAB[f])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(KLAB, fontsize=7)
    ax.set_ylabel("predicted / observed - 1  [%]")
    ax.set_title("P6b transfer residual by coarse z", fontsize=8.5)
    ax.legend(fontsize=6.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=("/scratch/cavestru_root/cavestru0/"
                                           "mfho/absorber_ladder_2026-09-13/"
                                           "completeness"))
    ap.add_argument("--out", default=("/home/mfho/desi_gpy_dla_notes/figures/"
                                      "2026-09-13_absorber_ladder/"
                                      "completeness"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    style()
    S, Z = load(a.products)
    made = [
        fig_c_vs_n(S, Z, os.path.join(a.out, "comp_fig1_C_vs_N_per_stratum.png")),
        fig_residual_map(S, Z, os.path.join(a.out, "comp_fig2_residual_maps.png")),
        fig_cv(S, os.path.join(a.out, "comp_fig3_cv_curves.png")),
        fig_by_k(S, os.path.join(a.out, "comp_fig4_residual_by_coarse_z.png")),
        fig_sparse(S, Z, os.path.join(a.out, "comp_fig5_sparse_uncertainty.png")),
        fig_p6b(S, os.path.join(a.out, "comp_fig6_p6b_transfer.png")),
    ]
    for m in made:
        print("wrote", m)


if __name__ == "__main__":
    main()
