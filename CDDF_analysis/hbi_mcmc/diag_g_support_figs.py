#!/usr/bin/env python
"""diag_g_support_figs.py — figures for the 2026-08-20 g(N,z) support-mismatch
diagnostic (finding N1). Reads the JSONs written by diag_g_support.py and
(optionally) the per-z mock-recovery JSONs written by cc_posterior_validation
on the deployed and the DIAGPACK_gcons packs. Presentation only; no science.

  python -m CDDF_analysis.hbi_mcmc.diag_g_support_figs --diag DIAG.json
      --out-dir DIR [--recovery-deployed J1 J2 ...] [--recovery-gcons J1 J2 ...]
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAM = {"2lpt0": "2LPT-0", "london0": "London-0", "saclay0": "Saclay-0"}
COL = {"2lpt0": "C0", "london0": "C1", "saclay0": "C2"}


def _fam(name):
    for k in FAM:
        if k in name:
            return k
    return name


def fig_closure(diag, out):
    zf = np.asarray(diag["zf_edges"]); zc = 0.5 * (zf[:-1] + zf[1:])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    titles = {"deployed_g": "deployed fold (g as shipped)",
              "g_consistent": "g rebuilt on the consistent truth support",
              "g_equals_1": "g ≡ 1 (no z-shape)"}
    for ax, gname in zip(axes, ("deployed_g", "g_consistent", "g_equals_1")):
        for fam, r in diag["closure"].items():
            f = _fam(fam)
            c = r[gname]["nhat_ge20.3"]
            ax.plot(zc, c["ratio_per_cell"], "o-", color=COL[f], ms=4, label=FAM[f])
            bins = c["ratio_paper1_bins"]
            for (name, lo, hi) in (("B1", 2.15, 2.35), ("B2", 2.35, 2.56), ("B3", 2.56, 2.96),
                                   ("B4", 2.96, 3.40), ("B5", 3.40, 3.50)):
                ax.hlines(bins[name], lo, hi, color=COL[f], lw=3, alpha=0.35)
        ax.axhline(1, color="k", lw=0.8)
        for e in (2.5, 3.0):
            ax.axvline(e, color="0.75", lw=0.8, ls="--")
        ax.set_title(titles[gname], fontsize=10)
        ax.set_xlabel("absorber z (native 0.1 cells; bars = Paper-1 bins)")
        ax.set_ylim(0.8, 1.45)
    axes[0].set_ylabel("predicted / observed counts at mock truth, N̂ ≥ 20.3")
    axes[0].legend(fontsize=9)
    fig.suptitle("Per-z closure of the count-conserving fold at mock truth (f = truth, ψ = 0, t = 0, loa-0 FP) — no sampling", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_g(diag, out):
    zf = np.asarray(diag["zf_edges"]); zc = 0.5 * (zf[:-1] + zf[1:])
    rows = [r for r in diag["per_row"] if float(r.strip("[]()").split(",")[0]) >= 20.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for row in rows[:4]:
        v = diag["per_row"][row]
        axes[0].plot(zc, v["excess_ratio"], "o-", ms=3, label=row)
        axes[1].plot(zc, v["g_deployed"], "o-", ms=3, label=f"deployed {row}")
        axes[1].plot(zc, v["g_consistent"], "s--", ms=3, label=f"consistent {row}")
        axes[2].plot(zc, v["raw_C_consistent"], "o-", ms=3, label=row)
    axes[0].set_title("g denominator excess:\nn_true(no SNR cut) / n_true(S2N_RED>2)", fontsize=10)
    axes[0].set_ylabel("ratio"); axes[0].legend(fontsize=8)
    axes[1].set_title("z-shape surface g(N,z):\ndeployed vs consistent-support rebuild", fontsize=10)
    axes[1].axhline(1, color="k", lw=0.8); axes[1].legend(fontsize=7, ncol=2)
    axes[2].set_title("raw C(N,z) = TP / truth\non the consistent support", fontsize=10)
    axes[2].set_ylim(0.6, 1.05); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("absorber z")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _recovery_rows(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        fam = _fam(os.path.basename(d["pack"]))
        seed = os.path.basename(p).split("_s")[-1].split(".")[0]
        out[(fam, seed)] = d
    return out


def fig_recovery(dep, gc, out):
    bins = ["B1", "B2", "B3", "B4", "B5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, thr in zip(axes, ("ge20.3", "ge20.0")):
        for (fam, seed), d in sorted(dep.items()):
            pb = d["perz_recovery"]["estimand"][thr]["paper1_bins"]
            ax.plot(range(5), [b["median_bias_pct"] for b in pb], "o-", color=COL[fam],
                    label=f"{FAM[fam]} s{seed[-2:]} deployed g")
        for (fam, seed), d in sorted(gc.items()):
            pb = d["perz_recovery"]["estimand"][thr]["paper1_bins"]
            ax.plot(range(5), [b["median_bias_pct"] for b in pb], "s--", color=COL[fam],
                    mfc="none", label=f"{FAM[fam]} s{seed[-2:]} g consistent")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{b}" for b in bins])
        ax.set_title(f"dN/dX({thr.replace('ge', '≥ ')}) per Paper-1 bin:\nposterior median bias vs mock truth", fontsize=10)
    axes[0].set_ylabel("median bias [%]  (B5 = [3.40,3.50) coverage 25%)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recovery-deployed", nargs="*", default=[])
    ap.add_argument("--recovery-gcons", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    diag = json.load(open(a.diag))
    fig_closure(diag, os.path.join(a.out_dir, "figA_perz_fold_closure_3variants.png"))
    fig_g(diag, os.path.join(a.out_dir, "figB_g_support_audit.png"))
    if a.recovery_deployed or a.recovery_gcons:
        fig_recovery(_recovery_rows(a.recovery_deployed), _recovery_rows(a.recovery_gcons),
                     os.path.join(a.out_dir, "figC_perbin_mock_recovery.png"))
    print("figures written to", a.out_dir)


if __name__ == "__main__":
    main()
