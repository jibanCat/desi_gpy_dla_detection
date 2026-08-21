#!/usr/bin/env python
"""perz_fold_closure.py — DIAGNOSTIC (2026-08-20, finding N1): per-z closure of
the deployed count-conserving fold evaluated AT MOCK TRUTH, no sampling.

For a mock pack with truth_counts, set f = truth_f (truth_counts / (dX dN)),
psi_c = 0 (molly point completeness), t = 0 and lam_fp = fp_counts / ell_eff
(loa-0 calibration), fold through the SAME precomputed mass tensor the
sampler uses (cc_posterior_validation.build_cc_tensors) and compare the
predicted counts with the observed counts per native z cell, per coarse FP
block and per locked Paper-1 reporting bin, for N_hat >= 20.3 / >= 20.0 and
all N_hat. Also reported: the same closure with the z-shape surface g(N,z)
replaced by 1, to isolate g's contribution.

Why: the posterior validation (ckpt 10.10) checked recovery ALL-z only; g is
normalised per N row to an occupancy-weighted z-mean of 1, so an all-z check
cannot see a z-tilt of the fold. This script measures the tilt directly.
It changes nothing; MOCK-ONLY (refuses packs without truth_counts).

Env: gpdla-hbi.
  python -m CDDF_analysis.hbi_mcmc.perz_fold_closure --pack P1 [P2 ...]
         [--out-json OUT.json] [--out-png OUT.png]
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import (
    build_cc_tensors, PAPER1_LOWZ_BINS, _overlap_w)
from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f


def fold_at_truth(pk, consts, Mg, g_factor):
    ft = truth_f(pk)
    Cc = jax.nn.sigmoid(consts.eta_hat)[:, consts.b_to_cell]          # (S,B)
    w = g_factor * jnp.asarray(ft) * consts.dN_b[:, None]             # (B,Kf)
    tp = jnp.einsum("skcb,sb,bk->cks", Mg, Cc, w) * consts.dX[None, :, :]
    lam = jnp.asarray(np.asarray(pk.fp_counts, float)) / consts.fp_ell_eff
    fp = (consts.fp_w * consts.fp_ell_eff
          * (1.0 - consts.fp_eta_c)[:, None, None]
          * lam[:, None, :] * consts.fp_E[None, :, :])
    return np.asarray(tp), np.asarray(fp)


def closure_table(pk, pred, obs, cmask, label):
    """pred/obs ratios per z cell, coarse block, Paper-1 bin for one N_hat mask."""
    zf = np.asarray(pk.zf_edges, float)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    kz = np.asarray(pk.kz_to_K)
    zc = np.asarray(pk.zc_edges, float)
    p_k = pred[cmask].sum(axis=(0, 2))
    o_k = obs[cmask].sum(axis=(0, 2))

    def ratio(w):
        return float((p_k * (w > 0)).sum() / max((o_k * (w > 0)).sum(), 1e-300))

    def ratio_w(w):
        # overlap-weighted (for reporting bins that cut through a cell): both
        # numerator and denominator scaled by the same cell overlap fraction
        frac = np.where(dX_k > 0, w / np.where(dX_k > 0, dX_k, 1.0), 0.0)
        return float((p_k * frac).sum() / max((o_k * frac).sum(), 1e-300))
    out = dict(label=label,
               obs_per_cell=[int(x) for x in o_k],
               pred_per_cell=[round(float(x), 2) for x in p_k],
               ratio_per_cell=[round(float(p / max(o, 1e-300)), 4)
                               for p, o in zip(p_k, o_k)],
               ratio_coarse={f"block{q}_[{zc[q]},{zc[q+1]})":
                             round(ratio(np.where(kz == q, 1.0, 0.0)), 4)
                             for q in range(len(zc) - 1)},
               ratio_paper1_bins={name: round(ratio_w(_overlap_w(zf, dX_k, lo, hi)), 4)
                                  for name, lo, hi in PAPER1_LOWZ_BINS},
               ratio_allz=round(float(p_k.sum() / o_k.sum()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", nargs="+", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-png", default=None)
    a = ap.parse_args()
    res = {}
    for path in a.pack:
        pk = load_pack(path)
        if pk.truth_counts is None:
            raise SystemExit(f"{path}: no truth_counts — MOCK-ONLY diagnostic")
        consts, Mg = build_cc_tensors(pk)
        obs = np.asarray(pk.counts, float)
        ne = np.asarray(pk.nhat_edges, float)
        masks = {"nhat_ge20.3": ne[:-1] >= 20.3 - 1e-9,
                 "nhat_ge20.0": ne[:-1] >= 20.0 - 1e-9,
                 "nhat_all": np.ones(len(ne) - 1, bool)}
        fam = os.path.basename(path)
        res[fam] = {"pack": path, "g_grid_rows_ge20p3": {}}
        me = np.asarray(pk.molly_nhi_edges, float)
        for j in range(len(me) - 1):
            if me[j] >= 20.3 - 1e-9:
                res[fam]["g_grid_rows_ge20p3"][f"[{me[j]},{me[j+1]})"] = \
                    [round(float(x), 4) for x in np.asarray(pk.g_grid)[j]]
        for gname, gfac in (("deployed_g", consts.g_bk),
                            ("g_equals_1", jnp.ones_like(consts.g_bk))):
            tp, fp = fold_at_truth(pk, consts, Mg, gfac)
            res[fam][gname] = {mk: closure_table(pk, tp + fp, obs, mm, f"{gname}/{mk}")
                               for mk, mm in masks.items()}
            res[fam][gname]["fp_share_allz_ge20.3"] = round(float(
                fp[masks["nhat_ge20.3"]].sum()
                / (tp + fp)[masks["nhat_ge20.3"]].sum()), 5)
        r = res[fam]["deployed_g"]["nhat_ge20.3"]
        print(f"{fam}: ge20.3 deployed-g pred/obs per coarse block "
              f"{r['ratio_coarse']} allz {r['ratio_allz']}; paper1 bins "
              f"{r['ratio_paper1_bins']}")
    out = dict(role=("DIAGNOSTIC per-z closure of the deployed fold at mock "
                     "truth (no sampling); nothing applied"),
               results=res)
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1)
    if a.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, gname, title in zip(axes, ("deployed_g", "g_equals_1"),
                                    ("deployed fold (with g(N,z))",
                                     "same fold, g ≡ 1")):
            for fam, r in res.items():
                pk = load_pack(r["pack"])
                zc = 0.5 * (np.asarray(pk.zf_edges)[:-1] + np.asarray(pk.zf_edges)[1:])
                ax.plot(zc, r[gname]["nhat_ge20.3"]["ratio_per_cell"], "o-",
                        label=fam.replace("scanpack_", "").replace("_b300.npz", ""))
            ax.axhline(1.0, color="k", lw=0.8)
            for e in (2.5, 3.0):
                ax.axvline(e, color="0.7", lw=0.8, ls="--")
            ax.set_title(title)
            ax.set_xlabel("absorber z (native 0.1 cells)")
        axes[0].set_ylabel("predicted / observed counts, N̂ ≥ 20.3, at mock truth")
        axes[0].legend()
        fig.suptitle("Per-z closure of the count-conserving fold evaluated at mock truth "
                     "(f = truth, ψ=0, t=0, loa-0 FP)")
        fig.tight_layout()
        fig.savefig(a.out_png, dpi=140)
    return out


if __name__ == "__main__":
    main()
