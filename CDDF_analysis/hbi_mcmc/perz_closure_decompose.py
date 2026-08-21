#!/usr/bin/env python
"""perz_closure_decompose.py — DIAGNOSTIC (PI ruling 2026-08-20 item 5): decompose
the residual of the per-z fold closure at mock truth by SNR stratum, by N_hat
row and by response z-cell, to see whether the +4..+7 % B3 residual (response
z-cell 1, [2.56, 2.96)) is localized to a selection regime, tied to N_hat
migration, or broad. No sampling; nothing revised.

For each pack (intended: the DIAGPACK_gcons copies, i.e. the consistent-g
fold; the deployed packs may be passed for contrast), predicted (TP+FP) vs
observed counts with f = truth, psi = 0, t = 0, loa-0 FP, tabulated as
pred/obs per (response z-cell x SNR stratum) and (response z-cell x N_hat
cell), plus TP-only shares.

  python -m CDDF_analysis.hbi_mcmc.perz_closure_decompose --packs P.. --out-json J [--out-png F]
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
from CDDF_analysis.hbi_mcmc.perz_fold_closure import fold_at_truth


def decompose(pk, consts, Mg):
    tp, fp = fold_at_truth(pk, consts, Mg, consts.g_bk)
    obs = np.asarray(pk.counts, float)
    pred = tp + fp
    zf = np.asarray(pk.zf_edges, float); zc = 0.5 * (zf[:-1] + zf[1:])
    rze = np.asarray(pk.resp_z_edges, float)
    kcell = np.clip(np.searchsorted(rze, zc, side="right") - 1, 0, len(rze) - 2)
    ne = np.asarray(pk.nhat_edges, float)
    se = np.asarray(pk.snr_edges, float)
    out = dict(resp_z_edges=[float(x) for x in rze],
               native_cells_per_resp_zcell={int(q): [int(k) for k in np.flatnonzero(kcell == q)]
                                            for q in range(len(rze) - 1)},
               nhat_edges=[float(x) for x in ne], snr_edges=[float(x) for x in se],
               by_zcell={})
    m203 = ne[:-1] >= 20.3 - 1e-9
    for q in range(len(rze) - 1):
        ks = kcell == q
        P = pred[:, ks, :]; O = obs[:, ks, :]; T = tp[:, ks, :]

        def r(num, den):
            return float(num.sum() / den.sum()) if den.sum() > 0 else None
        by_s = [dict(snr=[float(se[s]), float(se[s + 1])], obs=int(O[m203][:, :, s].sum()),
                     ratio_ge20p3=r(P[m203][:, :, s], O[m203][:, :, s]),
                     ratio_all=r(P[:, :, s], O[:, :, s])) for s in range(O.shape[2])]
        by_c = [dict(nhat=[float(ne[c]), float(ne[c + 1])], obs=int(O[c].sum()),
                     ratio=r(P[c], O[c]), fp_share_pred=float(fp[c][:, ks, :].sum() / max(P[c].sum(), 1e-300)))
                for c in range(O.shape[0])]
        out["by_zcell"][int(q)] = dict(
            z=[float(rze[q]), float(rze[q + 1])],
            ratio_ge20p3=r(P[m203], O[m203]), ratio_all=r(P, O),
            tp_share_ge20p3=float(T[m203].sum() / max(P[m203].sum(), 1e-300)),
            by_snr=by_s, by_nhat=by_c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-png", default=None)
    a = ap.parse_args()
    res = {}
    for p in a.packs:
        pk = load_pack(p)
        if pk.truth_counts is None:
            raise SystemExit("mock-only")
        consts, Mg = build_cc_tensors(pk)
        res[os.path.basename(p)] = decompose(pk, consts, Mg)
        d = res[os.path.basename(p)]
        print(os.path.basename(p), {q: (round(v["ratio_ge20p3"], 3), round(v["ratio_all"], 3))
                                    for q, v in d["by_zcell"].items()})
        for q, v in d["by_zcell"].items():
            print("  zcell", q, "by SNR (ge20.3):", [None if x["ratio_ge20p3"] is None else round(x["ratio_ge20p3"], 3) for x in v["by_snr"]])
            print("  zcell", q, "by Nhat:", [(x["nhat"][0], None if x["ratio"] is None else round(x["ratio"], 3)) for x in v["by_nhat"] if x["obs"] > 0])
    json.dump(dict(role="DIAGNOSTIC decomposition of the per-z fold closure at truth; nothing revised",
                   results=res), open(a.out_json, "w"), indent=1)
    if a.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        cols = {0: "C0", 1: "C3", 2: "C2"}
        for name, d in res.items():
            fam = name.split("_")[1]
            for q, v in d["by_zcell"].items():
                xs = [0.5 * (x["snr"][0] + min(x["snr"][1], 9)) for x in v["by_snr"] if x["ratio_ge20p3"]]
                ys = [x["ratio_ge20p3"] for x in v["by_snr"] if x["ratio_ge20p3"]]
                axes[0].plot(xs, ys, "o-", color=cols[int(q)], alpha=0.8,
                             label=f"{fam} z-cell {q} {v['z']}" if fam == "2lpt0" else None)
                xs = [x["nhat"][0] + 0.05 for x in v["by_nhat"] if x["ratio"] and x["obs"] > 30]
                ys = [x["ratio"] for x in v["by_nhat"] if x["ratio"] and x["obs"] > 30]
                axes[1].plot(xs, ys, "o-", color=cols[int(q)], alpha=0.8, ms=3)
        for ax in axes:
            ax.axhline(1, color="k", lw=0.8)
        axes[0].set_xlabel("SNR stratum (centre)"); axes[0].set_ylabel("pred/obs at truth, N̂ ≥ 20.3")
        axes[0].set_title("by SNR stratum, per response z-cell (colour)", fontsize=10); axes[0].legend(fontsize=8)
        axes[1].set_xlabel("N̂ cell (lower edge)"); axes[1].set_ylabel("pred/obs at truth, all SNR")
        axes[1].set_title("by N̂ row, per response z-cell (colour); cells with > 30 counts", fontsize=10)
        axes[1].set_ylim(0.6, 1.6)
        fig.suptitle("Fold closure at mock truth under the consistent-support g — residual decomposition", fontsize=11)
        fig.tight_layout(); fig.savefig(a.out_png, dpi=140)


if __name__ == "__main__":
    main()
