#!/usr/bin/env python
"""g_kernel_addendum.py — does the frozen g(N, z) already absorb the coarse-z
residual that every z-pooled completeness object leaves?

VALIDATION-ONLY.  Reads the products directory READ-ONLY; writes one figure to
the notes repo and one JSON to the scratchpad.  No delivered product is
modified, nothing is committed, no sampler is run.

The fold multiplies a z-pooled C[b, s] by the frozen z-shape g[b, k]
(``pack.g_grid[b_to_cell, :]``, the per-N-row-normalised completeness z-shape
measured on 2LPT-0 on the S2N_RED > snr_min support — finding N1), so the
EFFECTIVE completeness of a z-pooled object is ``C[b, s] * g[b, k]``.  The
z-carrying objects (C1gz, C1nsz) are passed through the runner's 3-D branch,
which drops g, so they are compared WITHOUT g.

Residual reported, truth-weighted exactly as in the main report:

    r_K = sum_{b, s, k in K} pred[b,k,s] * truth_bks[b,k,s]
          / sum_{b, s, k in K} det_bks[b,k,s]  -  1

ENV: ``gpdla`` or ``gpdla-hbi`` (numpy + matplotlib).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "validation", "absorber_diag"))
from binning import coarse_block_sum                               # noqa: E402
sys.path.insert(0, _HERE)
from cal_fit import expit, eta_hat_jeffreys                        # noqa: E402

FAMILIES = ("2lpt0", "london0", "saclay0")
FAMLAB = {"2lpt0": "2LPT-0 (calibration)", "london0": "London-0 (transfer)",
          "saclay0": "Saclay-0 (transfer)"}
ZPOOLED = ("C0", "C1g", "C1n", "C1ns", "C1nsadd")
ZCARRY = ("C1gz", "C1nsz")
VARCOL = {"C0": "#000000", "C1g": "#7a7a7a", "C1n": "#21918c",
          "C1ns": "#e0632b", "C1nsadd": "#5ec962", "C1gz": "#3b528b",
          "C1nsz": "#8c2981"}
KLAB = ("K0  z 2.0-2.5", "K1  z 2.5-3.0", "K2  z 3.0-3.5")


def resid_by_K(pred_bks, det_bks, truth_bks, kz, b_mask=None):
    """Truth-weighted expected/observed - 1 per coarse block (and the total)."""
    p = np.asarray(pred_bks, float)
    d = np.asarray(det_bks, float)
    t = np.asarray(truth_bks, float)
    if b_mask is not None:
        m = np.asarray(b_mask, bool)
        p, d, t = p[m], d[m], t[m]
    num = coarse_block_sum((p * t).sum(axis=2), kz, 1).sum(axis=0)   # (KK,)
    den = coarse_block_sum(d.sum(axis=2), kz, 1).sum(axis=0)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0) - 1.0, np.nan)
    tot = float((p * t).sum() / d.sum() - 1.0)
    return [float(v) for v in out], tot


def poisson_deviance(mu, obs):
    """2 * sum[ d log(d/mu) - (d - mu) ], the deviance of the fold's own
    Poisson likelihood on expected DETECTION COUNTS.

    This is the correct scoring rule for the EFFECTIVE completeness: ``C * g``
    is a rate multiplier in a Poisson fold, not a Bernoulli probability, and it
    legitimately exceeds 1 (see ``c_times_g_above_one`` below).  The binomial
    log-loss used in the main report scores C alone, which IS a probability.
    """
    mu = np.maximum(np.asarray(mu, float), 1e-12)
    d = np.asarray(obs, float)
    ok = d > 0
    return 2.0 * (np.where(ok, d * np.log(np.where(ok, d, 1.0) / mu), 0.0)
                  - (d - mu))


def z_dof_after_g(products, cal, g_bk, kz):
    """Is a z-resolved C still supported ON TOP OF the frozen g?

    Held-out (sightline-half) Poisson deviance on the fine (s, k, b) cells.
    The z-pooled objects are folded WITH g; the z-carrying objects WITHOUT
    (the runner's 3-D branch drops g).
    """
    live = np.where(np.asarray(cal["truth_bks"], float).sum(axis=(0, 1)) > 0)[0]

    def L(a):
        return np.transpose(np.asarray(a, float), (2, 1, 0))[live]

    dE, tE = L(cal["det_bks_E"]), L(cal["truth_bks_E"])
    dO, tO = L(cal["det_bks_O"]), L(cal["truth_bks_O"])
    t = L(cal["truth_bks"])
    gS = np.transpose(g_bk, (1, 0))[None, :, :]
    C0 = np.asarray(np.load(os.path.join(products, "C_C0_2lpt0.npz"),
                            allow_pickle=True)["C_fixed"], float)[live]

    def pooled(dh, th):
        e, _ = eta_hat_jeffreys(dh.sum(axis=1), th.sum(axis=1))
        return np.where(th.sum(axis=1) > 0, expit(e), C0)

    def zres(dh, th):
        dK = coarse_block_sum(dh, kz, 1)
        tK = coarse_block_sum(th, kz, 1)
        e, _ = eta_hat_jeffreys(dK, tK)
        p = np.where(tK > 0, expit(e), pooled(dh, th)[:, None, :])
        return p[:, kz, :]

    pE, pO = pooled(dO, tO), pooled(dE, tE)          # HELD OUT (other half)
    Cn = np.asarray(np.load(os.path.join(products, "C_C1nsadd_2lpt0.npz"),
                            allow_pickle=True)["C_fixed"], float)[live]
    Cz = np.transpose(np.asarray(
        np.load(os.path.join(products, "C_C1nsz_2lpt0.npz"),
                allow_pickle=True)["C_fixed_bkS"], float), (2, 1, 0))[live]
    bE = np.broadcast_to
    cases = {
        "C0 frozen x g": (bE(C0[:, None, :], dE.shape) * gS,
                          bE(C0[:, None, :], dO.shape) * gS),
        "C1g pooled NO g": (bE(pE[:, None, :], dE.shape).copy(),
                            bE(pO[:, None, :], dO.shape).copy()),
        "C1g pooled x g": (pE[:, None, :] * gS, pO[:, None, :] * gS),
        "C1nsadd (6 coef) x g": (bE(Cn[:, None, :], dE.shape) * gS,
                                 bE(Cn[:, None, :], dO.shape) * gS),
        "C1gz z-resolved NO g": (zres(dO, tO), zres(dE, tE)),
        "C1nsz (8 coef) NO g": (bE(Cz[None], (1,) + Cz.shape)[0],
                                bE(Cz[None], (1,) + Cz.shape)[0]),
    }
    m = t > 0
    dev, cells = {}, {}
    for k, (a, b) in cases.items():
        cE = poisson_deviance(a[m] * tE[m], dE[m])
        cO = poisson_deviance(b[m] * tO[m], dO[m])
        dev[k] = float(cE.sum() + cO.sum())
        cells[k] = np.concatenate([cE, cO])
    ref = "C1g pooled x g"
    out = dict(scoring=("held-out Poisson deviance of mu = pred * truth_bks on "
                        "the fine (s, k, b) cells, sightline halves; the fold's "
                        "own likelihood"),
               reference=ref, deviance=dev, delta={})
    for k in cases:
        if k == ref:
            continue
        dd = cells[k] - cells[ref]
        out["delta"][k] = dict(
            delta=float(dev[k] - dev[ref]),
            cellwise_se=float(np.sqrt(dd.size) * dd.std(ddof=1)))
    p = pooled(np.asarray(L(cal["det_bks"])), t)[:, None, :] * gS
    out["c_times_g_above_one"] = dict(
        max=float(p.max()),
        n_cells=int(((p > 1) & m).sum()), n_live_cells=int(m.sum()),
        truth_weight_pct=float(100 * t[(p > 1) & m].sum() / t[m].sum()),
        note=("C * g legitimately exceeds 1: g is a per-N-row z-RESHAPE, not a "
              "probability reweighting, and the fold is Poisson in counts"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=("/scratch/cavestru_root/cavestru0/"
                                           "mfho/absorber_ladder_2026-09-13/"
                                           "completeness"))
    ap.add_argument("--figdir", default=("/home/mfho/desi_gpy_dla_notes/"
                                         "figures/2026-09-13_absorber_ladder/"
                                         "completeness"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    tabs = {f: np.load(os.path.join(a.products, f"cal_table_{f}.npz"),
                       allow_pickle=True) for f in FAMILIES}
    cal = tabs["2lpt0"]
    kz = np.asarray(cal["kz_to_K"], int)
    ntrue = np.asarray(cal["ntrue_edges"], float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    me = np.asarray(cal["molly_nhi_edges"], float)
    b_to_cell = np.clip(np.digitize(Nc, me) - 1, 0, len(me) - 2)
    g_bk = np.asarray(cal["g_grid"], float)[b_to_cell, :]            # (B, Kf)
    reported = Nc >= 19.5 - 1e-9

    # how much of g is a pure z-RESHAPE (total-preserving) on each family?
    g_norm = {}
    for f in FAMILIES:
        t = np.asarray(tabs[f]["truth_bks"], float)
        g_norm[f] = float((g_bk[:, :, None] * t).sum() / t.sum())

    out = dict(role=("does the frozen g(N,z) absorb the coarse-z residual of a "
                     "z-pooled completeness object?"),
               g_definition=("pack.g_grid[b_to_cell, :]: the per-N-row "
                             "normalised completeness z-shape, measured ONCE "
                             "on 2LPT-0 on the S2N_RED > snr_min support "
                             "(finding N1); FROZEN and identical in every pack"),
               weighting=("truth-weighted: sum(pred * truth_bks) / sum(det_bks) "
                          "- 1, summed over b, s and the fine-z bins of each "
                          "coarse block"),
               g_truth_weighted_mean=g_norm, variants={})

    for v in ZPOOLED + ZCARRY:
        rec = {}
        z = np.load(os.path.join(a.products, f"C_{v}_2lpt0.npz"),
                    allow_pickle=True)
        C2 = np.asarray(z["C_fixed"], float)                         # (S, B)
        C3 = (np.asarray(z["C_fixed_bkS"], float)
              if "C_fixed_bkS" in z.files else None)                 # (B,Kf,S)
        for f in FAMILIES:
            d = np.asarray(tabs[f]["det_bks"], float)
            t = np.asarray(tabs[f]["truth_bks"], float)
            blk = {}
            if v in ZPOOLED:
                p_no = np.broadcast_to(C2.T[:, None, :], d.shape)     # no g
                p_g = C2.T[:, None, :] * g_bk[:, :, None]             # with g
                blk["byK_without_g"], blk["total_without_g"] = resid_by_K(
                    p_no, d, t, kz)
                blk["byK_with_g"], blk["total_with_g"] = resid_by_K(
                    p_g, d, t, kz)
                blk["byK_with_g_reported"], blk["total_with_g_reported"] = \
                    resid_by_K(p_g, d, t, kz, b_mask=reported)
                blk["byK_without_g_reported"], _ = resid_by_K(
                    p_no, d, t, kz, b_mask=reported)
                blk["uses_g"] = True
            else:
                blk["byK_with_g"], blk["total_with_g"] = resid_by_K(
                    C3, d, t, kz)      # C3 already carries the z structure
                blk["byK_with_g_reported"], blk["total_with_g_reported"] = \
                    resid_by_K(C3, d, t, kz, b_mask=reported)
                blk["byK_without_g"] = blk["byK_with_g"]
                blk["total_without_g"] = blk["total_with_g"]
                blk["byK_without_g_reported"] = blk["byK_with_g_reported"]
                blk["uses_g"] = False
                blk["note"] = ("z-carrying object: the runner's 3-D branch "
                               "drops g, so this row is the NO-g fold and it "
                               "is the like-for-like number")
            rec[f] = blk
        out["variants"][v] = rec

    # ---------------- print the table ------------------------------------
    print("\nEFFECTIVE completeness residual by coarse z  "
          "[expected/observed - 1, %]\n")
    hdr = (f"{'variant':9s} {'g?':4s} " + " ".join(
        f"| {FAMLAB[f].split()[0]:>22s}" for f in FAMILIES) + " |")
    print(hdr)
    print(f"{'':9s} {'':4s} " + " ".join(
        "|    K0     K1     K2  " for _ in FAMILIES) + " |")
    print("-" * len(hdr))
    for v in ZPOOLED:
        for tag, key in (("no g", "byK_without_g"), ("x g", "byK_with_g")):
            row = f"{v:9s} {tag:4s} "
            for f in FAMILIES:
                row += "| " + " ".join(
                    f"{100*x:6.2f}" for x in out["variants"][v][f][key]) + " "
            print(row + "|")
    for v in ZCARRY:
        row = f"{v:9s} {'n/a':4s} "
        for f in FAMILIES:
            row += "| " + " ".join(
                f"{100*x:6.2f}" for x in out["variants"][v][f]["byK_with_g"]) \
                + " "
        print(row + "|")
    print("\ntruth-weighted mean of g: " + "  ".join(
        f"{f}={g_norm[f]:.4f}" for f in FAMILIES))

    out["z_dof_after_g"] = z_dof_after_g(a.products, cal, g_bk, kz)
    zz = out["z_dof_after_g"]
    print("\nHeld-out POISSON deviance (fold likelihood), fine (s,k,b):")
    for k, v in zz["deviance"].items():
        extra = ""
        if k in zz["delta"]:
            dd = zz["delta"][k]
            extra = (f"   delta vs '{zz['reference']}' "
                     f"{dd['delta']:+8.2f} +/- {dd['cellwise_se']:.2f}")
        print(f"  {k:24s} {v:10.2f}{extra}")
    c = zz["c_times_g_above_one"]
    print(f"  C*g > 1 in {c['n_cells']}/{c['n_live_cells']} live cells "
          f"(max {c['max']:.4f}); {c['truth_weight_pct']:.1f} % of truth weight")

    # ---------------- figure ---------------------------------------------
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130,
                         "font.size": 8.5, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=True)
    order = list(ZPOOLED) + list(ZCARRY)
    w = 0.11
    for ax, f in zip(axes, FAMILIES):
        for i, v in enumerate(order):
            y = [100 * x for x in out["variants"][v][f]["byK_with_g"]]
            ax.bar(np.arange(3) + (i - len(order) / 2 + 0.5) * w, y, width=w,
                   color=VARCOL[v],
                   label=(v + (r" $\times\,g$" if v in ZPOOLED else " (no g)"))
                   if f == "2lpt0" else None)
        for i, v in enumerate(ZPOOLED):
            y = [100 * x for x in out["variants"][v][f]["byK_without_g"]]
            ax.plot(np.arange(3) + (i - len(order) / 2 + 0.5) * w, y, "_",
                    ms=7, mew=1.4, color="0.25",
                    label="same object WITHOUT g" if (f == "2lpt0" and i == 0)
                    else None)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(KLAB, fontsize=7)
        ax.set_title(FAMLAB[f], fontsize=8.5)
    axes[0].set_ylabel("expected / observed detections - 1  [%]")
    axes[0].legend(fontsize=6.2, ncol=2, loc="lower left")
    fig.suptitle(r"Coarse-z residual of the EFFECTIVE completeness "
                 r"$C[b,s]\,\cdot\,g[b,k]$ — the frozen $g$ removes most of "
                 r"the +5/$-$5/$-$6 % structure", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(a.figdir, "comp_fig7_effective_C_times_g_by_z.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote", p)

    jp = a.json or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "g_kernel_addendum.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", jp)
    return out


if __name__ == "__main__":
    main()
