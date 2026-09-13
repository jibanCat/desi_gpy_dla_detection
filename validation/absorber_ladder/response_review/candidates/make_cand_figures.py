#!/usr/bin/env python
"""make_cand_figures.py -- the five figures of the low-DOF response-candidate
review (sealed opening rule section 3 metrics).

  cand_fig1_rowKL_by_b_s.png       held-out row KL by (b, S/N) per candidate
  cand_fig2_example_rows.png       held-out histogram vs R1c vs candidates
  cand_fig3_boundary_residuals.png boundary-crossing mass residuals
  cand_fig4_pit.png                PIT / CDF calibration
  cand_fig5_transfer.png           London-0 / Saclay-0 transfer + phi

All panels are HELD-OUT (fold-fitted rows scored on the other fold's counts),
using ``cv_fold_rows.npz``.  VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
for _p in (_HERE, os.path.join(_REPO, "validation", "absorber_ladder",
                               "response"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import candlib as CL                                           # noqa: E402
import candmetrics as CM                                       # noqa: E402

ORDER = ["R0", "R1c", "R1d-raw", "A-small", "E", "B", "C", "D"]
COL = {"R0": "#440154", "R1c": "#d95f02", "R1d-raw": "#999999",
       "A-small": "#31688e", "E": "#35b779", "B": "#b52d8c",
       "C": "#1f9e89", "D": "#8b5a2b"}
LS = {"R0": ":", "R1c": "--", "R1d-raw": "-.", "A-small": "-", "E": "-",
      "B": "-", "C": "-", "D": "-"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "legend.fontsize": 7.2, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "figure.constrained_layout.use": True})


def load(out):
    z = np.load(os.path.join(out, "cv_fold_rows.npz"), allow_pickle=True)
    tab = json.load(open(os.path.join(out, "candidate_cv_table.json")))
    geom = dict(ntrue=z["ntrue_edges"], nhat=z["nhat_edges"],
                snr=z["snr_edges"], zc=z["zc_edges"])
    geom["B"] = len(geom["ntrue"]) - 1
    geom["C"] = len(geom["nhat"]) - 1
    geom["S"] = len(geom["snr"]) - 1
    geom["K"] = len(geom["zc"]) - 1
    geom["ccen"] = 0.5 * (geom["nhat"][:-1] + geom["nhat"][1:])
    geom["bcen"] = 0.5 * (geom["ntrue"][:-1] + geom["ntrue"][1:])
    names = [n for n in ORDER if f"{n}__rows_fold0" in z]
    return z, tab, geom, names


def pooled_records(z, geom, name):
    recs = []
    for i in (0, 1):
        recs += CM.row_table(z[f"{name}__rows_fold{i}"],
                             z[f"{name}__heldout_counts_fold{i}"], geom,
                             min_events=20)
    return recs


# ---------------------------------------------------------------------------
def fig1(z, geom, names, out_png):
    _style()
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(1.85 * n + 1.1, 3.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    grids = {}
    for name in names:
        recs = pooled_records(z, geom, name)
        g = np.full((geom["B"], geom["S"]), np.nan)
        acc = np.zeros((geom["B"], geom["S"])); wt = np.zeros_like(acc)
        for r in recs:
            acc[r["b"], r["s"]] += r["n"] * r["kl"]
            wt[r["b"], r["s"]] += r["n"]
        g = np.where(wt > 0, acc / np.maximum(wt, 1e-12), np.nan)
        grids[name] = g
    vmax = np.nanpercentile(np.concatenate([grids[n].ravel()
                                            for n in names]), 97)
    for ax, name in zip(axes, names):
        im = ax.pcolormesh(np.arange(geom["S"] + 1), geom["ntrue"],
                           grids[name], vmin=0.0, vmax=vmax, cmap="magma_r")
        ax.set_title(name)
        ax.set_xlabel(r"S/N stratum")
        ax.set_xticks(np.arange(geom["S"]) + 0.5)
        ax.set_xticklabels([f"{int(geom['snr'][s])}" for s in
                            range(geom["S"])], fontsize=6)
    axes[0].set_ylabel(r"true $\log_{10} N_{\rm HI}$ bin")
    fig.colorbar(im, ax=axes.tolist(), label="held-out row KL (nats)",
                 fraction=0.035)
    fig.suptitle("Held-out row KL(empirical || model) by latent bin and S/N "
                 "stratum (rows with >= 20 held-out events)", fontsize=9)
    fig.savefig(out_png)
    plt.close(fig)


def fig2(z, geom, names, out_png):
    _style()
    targets = [(19.9, 20.1), (20.5, 20.7), (21.3, 21.5)]
    bidx = [int(np.argmin(np.abs(geom["ntrue"][:-1] - t[0])))
            for t in targets]
    strata = [(2, "S/N 2-3"), (7, "S/N >= 7")]
    fig, axes = plt.subplots(len(strata), len(bidx),
                             figsize=(3.1 * len(bidx), 2.5 * len(strata)))
    cc = geom["ccen"]
    for i, (s, slab) in enumerate(strata):
        for j, b in enumerate(bidx):
            ax = axes[i, j]
            cnt = sum(z[f"{names[0]}__heldout_counts_fold{k}"][b, s, 0]
                      for k in (0, 1))
            tot = cnt.sum()
            if tot > 0:
                ax.step(cc, cnt / tot, where="mid", color="k", lw=1.3,
                        label=f"held-out ({int(tot)})")
            for name in names:
                p = 0.5 * sum(z[f"{name}__rows_fold{k}"][b, s, 0]
                              for k in (0, 1))
                ax.plot(cc, p, color=COL.get(name, "#555"),
                        ls=LS.get(name, "-"), lw=1.1, label=name)
            ax.set_xlim(max(geom["nhat"][0] - 0.05,
                            geom["ntrue"][b] - 0.85),
                        min(geom["nhat"][-1], geom["ntrue"][b + 1] + 1.05))
            ax.set_yscale("log"); ax.set_ylim(3e-4, 1.0)
            ax.set_title(f"b = [{geom['ntrue'][b]:.1f}, "
                         f"{geom['ntrue'][b+1]:.1f}), {slab}, K = 0")
            if i == len(strata) - 1:
                ax.set_xlabel(r"$\hat N$ bin centre")
            if j == 0:
                ax.set_ylabel(r"$P(c\,|\,b,s,K)$")
    axes[0, 0].legend(ncol=2, fontsize=6)
    fig.suptitle("Held-out rows: measured histogram vs the fixed calibration "
                 "objects (both folds averaged)", fontsize=9)
    fig.savefig(out_png)
    plt.close(fig)


def fig3(z, geom, names, out_png):
    _style()
    bnd = CL.BOUNDARIES
    fig, axes = plt.subplots(1, len(bnd), figsize=(3.3 * len(bnd), 3.0),
                             sharey=True)
    for ax, beta in zip(np.atleast_1d(axes), bnd):
        for k, name in enumerate(names):
            zz = []
            for i in (0, 1):
                bt = CM.boundary_table(z[f"{name}__rows_fold{i}"],
                                       z[f"{name}__heldout_counts_fold{i}"],
                                       geom, boundaries=(beta,),
                                       min_events=200)
                zz += [r["z"] for r in bt[f"{beta:.1f}"]["records"]]
            if not zz:
                continue
            ax.scatter(np.full(len(zz), k) +
                       np.linspace(-0.22, 0.22, len(zz)), zz, s=5,
                       color=COL.get(name, "#555"), alpha=0.65)
            ax.plot([k - 0.3, k + 0.3], [np.median(zz)] * 2, color="k", lw=1.2)
        ax.axhspan(-3, 3, color="0.85", zorder=0)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_title(rf"boundary {beta:.1f}")
        ax.set_ylim(-20, 20)
    np.atleast_1d(axes)[0].set_ylabel(
        "(predicted - held-out) crossing mass / binomial SE")
    fig.suptitle("Boundary-crossing mass residuals, rows with >= 200 held-out "
                 "events (band = +-3 SE)", fontsize=9)
    fig.savefig(out_png)
    plt.close(fig)


def fig4(tab, names, out_png):
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for name in names:
        h = np.asarray(tab["table"][name]["pit"]["hist"], float)
        e = np.asarray(tab["table"][name]["pit"]["edges"], float)
        ax.step(0.5 * (e[:-1] + e[1:]), h / h.sum() * len(h), where="mid",
                color=COL.get(name, "#555"), ls=LS.get(name, "-"), lw=1.2,
                label=f"{name} ($\\chi^2$ = "
                      f"{tab['table'][name]['pit']['chi2']:.0f})")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("randomised PIT of the held-out detection")
    ax.set_ylabel("density (uniform = 1)")
    ax.legend(ncol=2, fontsize=6.5)
    ax.set_title("CDF calibration of the fixed objects (held-out, both folds)")
    fig.savefig(out_png)
    plt.close(fig)


def fig5(tab, names, out_png):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    fams = ["london0", "saclay0", "2lpt0"]
    x = np.arange(len(names))
    for i, fam in enumerate(fams):
        ax = axes[i]
        v = [tab["table"][n]["transfer"].get(fam, {}).get("wmean_kl", np.nan)
             for n in names]
        ax.bar(x, v, color=[COL.get(n, "#555") for n in names])
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_title(f"transfer to {fam}")
        ax.set_ylabel("row KL($M_{\\rm true}\\,||\\,$model), N_match-weighted")
    fig.suptitle("Transport of the 2LPT-0-fitted fixed object to the measured "
                 "London-0 / Saclay-0 operators", fontsize=9)
    fig.savefig(out_png)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True)
    ap.add_argument("--fig-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.fig_dir, exist_ok=True)
    z, tab, geom, names = load(a.products)
    fig1(z, geom, names, os.path.join(a.fig_dir,
                                      "cand_fig1_rowKL_by_b_s.png"))
    fig2(z, geom, names, os.path.join(a.fig_dir,
                                      "cand_fig2_example_rows.png"))
    fig3(z, geom, names, os.path.join(a.fig_dir,
                                      "cand_fig3_boundary_residuals.png"))
    fig4(tab, names, os.path.join(a.fig_dir, "cand_fig4_pit.png"))
    fig5(tab, names, os.path.join(a.fig_dir, "cand_fig5_transfer.png"))
    print("figures in", a.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
