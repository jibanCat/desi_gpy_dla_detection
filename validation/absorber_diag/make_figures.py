#!/usr/bin/env python
"""make_figures.py — the required plots for the absorber-side operator
forensics (VALIDATION-ONLY).  Reads ``fold_forensics_<fam>.json`` /
``.npz`` and ``empirical_ops_<fam>.npz`` and writes PNGs to the notes repo.

Env: gpdla-hbi (numpy + matplotlib only; no jax, no sampler).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ("2lpt0", "london0", "saclay0")
FAMLAB = {"2lpt0": "2LPT-0", "london0": "London-0", "saclay0": "Saclay-0"}
KLAB = ("K0  z 2.0-2.5", "K1  z 2.5-3.0", "K2  z 3.0-3.5")
KCOL = ("#440154", "#21918c", "#fde725")
#: the ORACLE posterior reporting-bin biases (RUN_ORACLE_*_s20260811.json)
ORACLE_BINS = {
    "2lpt0": [-8.57, 12.19, -10.36, 3.25, 0.58, 2.34, 2.44, 16.12, 0.99, -7.08],
    "london0": [-6.11, 3.57, -3.00, 1.64, 0.49, 2.72, 3.38, 10.53, 6.05, -8.67],
    "saclay0": [-7.84, 6.32, -6.30, 9.95, -3.79, 8.51, -0.92, 8.53, -8.87,
                -13.00]}
ORACLE_EDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False})


def load(fam):
    R = json.load(open(os.path.join(_HERE, f"fold_forensics_{fam}.json")))
    Z = np.load(os.path.join(_HERE, f"fold_forensics_{fam}.npz"))
    O = np.load(os.path.join(_HERE, f"empirical_ops_{fam}.npz"),
                allow_pickle=True)
    return R, Z, O


def fig_zigzag_nhat(out):
    """Fold-of-truth / matched detections vs observed N-hat, per K."""
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)
    for ax, fam in zip(axes, FAMILIES):
        R, _, _ = load(fam)
        lo = np.array([r["lo"] for r in R["by_nhat_by_K"]])
        hi = np.array([r["hi"] for r in R["by_nhat_by_K"]])
        x = 0.5 * (lo + hi)
        for K in range(3):
            y = np.array([np.nan if r["ratio"][K] is None else r["ratio"][K]
                          for r in R["by_nhat_by_K"]])
            n = np.array([r["obs"][K] for r in R["by_nhat_by_K"]])
            e = np.where(n > 0, y / np.sqrt(np.maximum(n, 1)), np.nan)
            m = n >= 20
            ax.errorbar(x[m], y[m], yerr=e[m], marker="o", ms=3, lw=1.1,
                        color=KCOL[K], label=KLAB[K], capsize=2)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_ylim(0.80, 1.25)
        ax.set_ylabel(r"fold($f_{\rm true}$) / matched")
        ax.text(0.01, 0.92, FAMLAB[fam], transform=ax.transAxes,
                fontweight="bold")
    axes[0].legend(ncol=3, loc="lower right")
    axes[-1].set_xlabel(r"observed $\log N_{\rm HI}$ bin centre")
    fig.suptitle("Forward fold of the mock's own truth vs the MATCHED "
                 "detections\n(psi_c = 0; Poisson errors on the matched "
                 "counts; cells with < 20 matches dropped)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "opfor_fig1_fold_vs_matched_by_nhat_perK.png"))
    plt.close(fig)


def fig_snr(out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        R, _, _ = load(fam)
        s = [r["s"] for r in R["by_snr"] if r["obs"] > 0]
        y = [r["ratio"] for r in R["by_snr"] if r["obs"] > 0]
        req = [R["counts_budget"]["by_snr_required"][i] for i in s]
        lab = [f"{int(R['by_snr'][i]['snr'][0])}-"
               f"{'inf' if not np.isfinite(R['by_snr'][i]['snr'][1]) else int(R['by_snr'][i]['snr'][1])}"
               for i in s]
        ax.plot(range(len(s)), y, "o-", color="#3b518b",
                label="fold(truth)/matched")
        ax.plot(range(len(s)), req, "s--", color="#c44e52",
                label="counts-FP required / fold(truth)")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axvspan(-0.4, 0.4, color="#fde725", alpha=0.35, zorder=0)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(lab, rotation=45)
        ax.set_xlabel("S/N stratum")
        ax.set_title(FAMLAB[fam], fontsize=9)
    axes[0].set_ylabel("ratio")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("S/N structure of the absorber-side operator "
                 "(highlighted: S/N 2-3)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "opfor_fig2_residual_vs_snr.png"))
    plt.close(fig)


def fig_migration(out):
    """Adopted skew-normal Mg vs the measured M_true, per stratum and K."""
    for fam in FAMILIES:
        R, Z, O = load(fam)
        ntrue = np.asarray(O["ntrue_edges"], float)
        nhat = np.asarray(O["nhat_edges"], float)
        xc = 0.5 * (nhat[:-1] + nhat[1:])
        Mn = np.asarray(Z["Mg_norm"], float)        # (S,KK,C,B)
        Mt = np.asarray(O["M_true_sKcb"], float)
        Mc = np.asarray(O["M_counts_sKcb"], float)
        bs = [b for b in range(len(ntrue) - 1)
              if (ntrue[b] >= 19.7 - 1e-9 and ntrue[b + 1] <= 20.5 + 1e-9)
              or (ntrue[b] >= 21.1 - 1e-9 and ntrue[b + 1] <= 21.7 + 1e-9)]
        fig, axes = plt.subplots(len(bs), 3, figsize=(11.5, 2.0 * len(bs)),
                                 sharex=True)
        for i, b in enumerate(bs):
            for K in range(3):
                ax = axes[i, K]
                for s, col, lab in ((2, "#c44e52", "S/N 2-3"),
                                    (7, "#3b518b", "S/N>=7")):
                    ax.step(xc, Mn[s, K, :, b], where="mid", color=col, lw=1.0,
                            ls="--", label=f"adopted {lab}")
                    if Mc[s, K, :, b].sum() > 0:
                        ax.step(xc, Mt[s, K, :, b], where="mid", color=col,
                                lw=1.4, label=f"measured {lab} "
                                              f"(n={int(Mc[s, K, :, b].sum())})")
                ax.axvspan(ntrue[b], ntrue[b + 1], color="k", alpha=0.07)
                ax.set_xlim(19.4, 22.0)
                if i == 0:
                    ax.set_title(KLAB[K], fontsize=8)
                if K == 0:
                    ax.set_ylabel(f"b=[{ntrue[b]:.1f},{ntrue[b+1]:.1f})",
                                  fontsize=8)
        axes[0, 2].legend(fontsize=6)
        for K in range(3):
            axes[-1, K].set_xlabel(r"observed $\log N_{\rm HI}$")
        fig.suptitle(f"{FAMLAB[fam]}: migration rows, adopted skew-normal "
                     "(dashed) vs measured (solid); shaded = the true-N bin",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(out, f"opfor_fig3_migration_{fam}.png"))
        plt.close(fig)


def fig_completeness(out):
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.0), sharex=True,
                             sharey=True)
    for r, fam in enumerate(FAMILIES):
        R, _, _ = load(fam)
        cv = R.get("completeness_vs_counting")
        be = np.array(R["completeness"]["b_edges"], float)
        x = 0.5 * (be[:, 0] + be[:, 1])
        for K in range(3):
            ax = axes[r, K]
            if cv is None:
                continue
            ct = np.array(cv["C_det_true"], float)[:, K, :]
            cm = np.array(cv["C_det_model"], float)[:, K, :]
            tcn = np.array(R["completeness"]["truth_counts_bKs"],
                           float)[:, K, :]
            for s, col, lab in ((2, "#c44e52", "S/N 2-3"),
                                (7, "#3b518b", "S/N>=7")):
                m = tcn[:, s] >= 20
                ax.plot(x[m], ct[m, s], "o-", ms=3, color=col,
                        label=f"measured {lab}")
                ax.plot(x[m], cm[m, s], "--", color=col,
                        label=f"calibrated C x g {lab}")
            ax.set_ylim(0, 1.05)
            if r == 0:
                ax.set_title(KLAB[K], fontsize=9)
            if K == 0:
                ax.set_ylabel(f"{FAMLAB[fam]}\ncompleteness")
    axes[0, 2].legend(fontsize=6.5, loc="lower right")
    for K in range(3):
        axes[-1, K].set_xlabel(r"true $\log N_{\rm HI}$ bin centre")
    fig.suptitle("Detection completeness: mock truth vs the frozen 2LPT-0 "
                 "molly calibration x g(N,z)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "opfor_fig4_completeness_truth_vs_calibration.png"))
    plt.close(fig)


def fig_zigzag_latent(out):
    """ORACLE posterior zigzag vs the deterministic operator-level inversion."""
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.4), sharex=True)
    for ax, fam in zip(axes, FAMILIES):
        R, _, _ = load(fam)
        inv = R["deterministic_inversion"]
        xo = 0.5 * (ORACLE_EDGES[:-1] + ORACLE_EDGES[1:])
        ax.plot(xo, ORACLE_BINS[fam], "ks-", ms=5, lw=1.6,
                label="ORACLE posterior median bias")
        for key, col, lab in (
                ("oracle_counts_minus_hostless", "#c44e52",
                 "operator inversion (frozen)"),
                ("fix_migration", "#21918c", "+ measured migration"),
                ("fix_completeness", "#9467bd", "+ measured completeness"),
                ("fix_both", "#3b518b", "+ both")):
            b = inv[key]["reporting_bins"]
            x = np.array([0.5 * (r["bin"][0] + r["bin"][1]) for r in b])
            y = np.array([r["bias_pct"] for r in b])
            m = (x > 19.65) & (x < 21.75)
            ax.plot(x[m], y[m], "o-", ms=3, color=col, lw=1.1, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-20, 22)
        ax.set_ylabel("bias in f  [%]")
        ax.text(0.01, 0.90, FAMLAB[fam], transform=ax.transAxes,
                fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0.0, -0.02))
    axes[-1].set_xlabel(r"0.2-dex reporting bin ($\log N_{\rm HI}$)")
    fig.suptitle("The 0.2-dex zigzag: ORACLE posterior vs the deterministic "
                 "operator-level Poisson MLE\n(no sampler, no population "
                 "prior), before and after exact component replacement",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "opfor_fig5_zigzag_latent_before_after.png"))
    plt.close(fig)


def fig_p6b(out):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, fam in zip(axes, FAMILIES):
        R, _, O = load(fam)
        nhat = np.asarray(O["nhat_edges"], float)
        x = 0.5 * (nhat[:-1] + nhat[1:])
        nt = R["no_term_classes"]
        ax.step(x, nt["P6b_by_nhat"], where="mid", color="#c44e52",
                label="P6b: host [17.2,19.0) (NO term)")
        ax.step(x, nt["hostless_by_nhat"], where="mid", color="#3b518b",
                label="hostless (ORACLE FP pin)")
        ax.step(x, [r["obs"] for r in R["by_nhat"]], where="mid", color="k",
                lw=0.9, label="matched in-basis")
        ax.set_yscale("log")
        ax.set_xlabel(r"observed $\log N_{\rm HI}$")
        ax.set_title(FAMLAB[fam], fontsize=9)
    axes[0].set_ylabel("detections per 0.1-dex bin")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Where the classes the fold has no term for actually land",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "opfor_fig6_no_term_classes.png"))
    plt.close(fig)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--figdir", default=("/home/mfho/desi_gpy_dla_notes/figures/"
                                        "2026-09-13_absorber_diag"))
    ns = a.parse_args(argv)
    os.makedirs(ns.figdir, exist_ok=True)
    style()
    fig_zigzag_nhat(ns.figdir)
    fig_snr(ns.figdir)
    fig_migration(ns.figdir)
    fig_completeness(ns.figdir)
    fig_zigzag_latent(ns.figdir)
    fig_p6b(ns.figdir)
    print("wrote figures to", ns.figdir)


if __name__ == "__main__":
    main()
