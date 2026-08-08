#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""17.2 ↔ nhi195 chain-compatibility bridge (PI §3, 2026-08-07 amended).

Purpose: before the 17.2 chain may be used to estimate below-floor
(N_true < 19.5 → N̂ > 19.5) upward migration, establish exactly what
differs between the two chains on their COMMON truth support
(N_true ≥ 19.5), so chain-competition effects are reported separately
and never absorbed into the migration estimate.

Shared BY CONSTRUCTION (same `_chain` wiring, one executable state,
verified below where cheap): catalogue file + finder output, cut
bundle, masks/λ windows, SNR strata, matcher (dz_rel = 0.01,
nhi_desc), assignment rule, dX/weighting, observed-bin definitions,
normalization. The ONLY designed difference is the truth-side floor
(17.2 vs 19.5), which changes MATCHING COMPETITION: candidates may be
claimed by sub-19.5 truth in the 17.2 chain.

Outputs: `p1_chain_bridge.json` + the 17.2-chain event cache
(`p1_completeness_cache_172.npz` on scratch) for the migration script.
Deterministic; production catalogue only (predates all injection
arms); no holdout, no finder rerun.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

CACHE195 = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
            "p1_completeness_cache.npz")
CACHE172 = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
            "p1_completeness_cache_172.npz")
FLOOR = 19.5


def main():
    t0 = time.time()
    from t1a_reproduce_cmolly import _chain, MOLLY_TSV_172
    mm, cat, tr, tp, good, meta = _chain(MOLLY_TSV_172)

    def col(t, name, dt=float):
        return np.asarray(t[name], dt) if name in t.colnames else None
    save = dict(
        cat_TARGETID=np.asarray(cat["TARGETID"], np.int64),
        cat_Z_DLA=col(cat, "Z_DLA"), cat_NHI=col(cat, "NHI"),
        cat_P_DLA=col(cat, "P_DLA"), cat_S2N=col(cat, "S2N_RED"),
        cat_NHI_TRUE=col(cat, "NHI_TRUE"), cat_Z_TRUE=col(cat, "Z_TRUE"),
        cat_ZQSO=col(cat, "Z_QSO"),
        cat_is_TP=np.asarray(tp, bool), cat_good=np.asarray(good, bool),
        tr_TARGETID=np.asarray(tr["TARGETID"], np.int64),
        tr_NHI=col(tr, "NHI"),
        tr_Z=(col(tr, "Z_TRUTH") if "Z_TRUTH" in tr.colnames
              else col(tr, "Z_DLA")),
        tr_S2N=col(tr, "S2N_RED"), tr_ZQSO=col(tr, "Z_QSO"))
    if "DLAFLAG" in cat.colnames:
        save["cat_DLAFLAG"] = np.asarray(cat["DLAFLAG"], float)
    np.savez(CACHE172, **{k: v for k, v in save.items() if v is not None})

    d5 = np.load(CACHE195)
    out = {"schema": "p1_chain_bridge/v1", "date": time.strftime("%Y-%m-%d"),
           "designed_difference": "truth floor 17.2 vs 19.5 (competition)"}

    # -- catalogue-side identity: same rows in both chains ---------------
    same_cat = (len(save["cat_TARGETID"]) == len(d5["cat_TARGETID"])
                and bool(np.array_equal(save["cat_TARGETID"],
                                        d5["cat_TARGETID"]))
                and bool(np.allclose(save["cat_NHI"], d5["cat_NHI"])))
    out["catalogue_rows_identical"] = same_cat

    # -- truth-side subset relation on common support --------------------
    k172 = set(zip(save["tr_TARGETID"].tolist(),
                   np.round(save["tr_Z"], 6).tolist(),
                   np.round(save["tr_NHI"], 6).tolist()))
    k195 = set(zip(d5["tr_TARGETID"].tolist(),
                   np.round(d5["tr_Z"], 6).tolist(),
                   np.round(d5["tr_NHI"], 6).tolist()))
    k172_ge = {k for k in k172 if k[2] > FLOOR}
    out["truth_common_support"] = {
        "n_172_above_floor": len(k172_ge), "n_195": len(k195),
        "identical": k172_ge == k195,
        "n_only_172": len(k172_ge - k195), "n_only_195": len(k195 - k172_ge)}

    # -- competition on common support: per-N matched sets ---------------
    def sel_tp(d):
        return (d["cat_is_TP"] & d["cat_good"] & (d["cat_P_DLA"] > 0.99)
                & (d["cat_NHI"] > FLOOR) & (d["cat_S2N"] > 2.0))
    tp172 = sel_tp(save)
    tp195 = sel_tp(d5)
    key172 = list(zip(save["cat_TARGETID"][tp172].tolist(),
                      np.round(save["cat_Z_DLA"][tp172], 6).tolist()))
    key195 = list(zip(d5["cat_TARGETID"][tp195].tolist(),
                      np.round(d5["cat_Z_DLA"][tp195], 6).tolist()))
    n172_t = dict(zip(key172, save["cat_NHI_TRUE"][tp172].tolist()))
    n195_t = dict(zip(key195, d5["cat_NHI_TRUE"][tp195].tolist()))
    rows = []
    NB = [(19.5, 20.0), (20.0, 20.3), (20.3, 20.5), (20.5, 21.0),
          (21.0, 21.5), (21.5, 22.5)]
    for lo, hi in NB:
        m195 = {k for k, v in n195_t.items() if lo <= v < hi}
        m172 = {k for k, v in n172_t.items() if lo <= v < hi}
        both = m195 & m172
        rows.append({
            "N_true": [lo, hi], "n_195": len(m195), "n_172": len(m172),
            "n_both_same_truth_bin": len(both),
            "n_195_only": len(m195 - m172), "n_172_only": len(m172 - m195)})
    out["common_support_competition"] = rows

    # -- classification of 172-chain sub-floor TPs vs the 195 chain ------
    sub172 = tp172 & (save["cat_NHI_TRUE"] < FLOOR)
    keys_sub = set(zip(save["cat_TARGETID"][sub172].tolist(),
                       np.round(save["cat_Z_DLA"][sub172], 6).tolist()))
    set195 = set(key195)
    out["subfloor_tp_172"] = {
        "n_total_selected": int(np.sum(sub172 & (save["cat_S2N"] > 2.0))),
        "n_reassigned_in_195": len(keys_sub & set195),
        "n_unmatched_in_195": len(keys_sub - set195),
        "note": ("reassigned = the SAME catalogue row is a ≥19.5-truth TP "
                 "in the 195 chain (pure competition, NOT physical "
                 "migration); unmatched_in_195 = candidate the 195 chain "
                 "leaves as FP — the genuine below-floor attribution "
                 "class for the migration source term")}

    out["executable"] = {k: v for k, v in meta.items()
                         if isinstance(v, (int, float, str, bool))}
    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "p1_chain_bridge.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("catalogue_rows_identical", "truth_common_support",
                       "subfloor_tp_172", "wall_s")}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
