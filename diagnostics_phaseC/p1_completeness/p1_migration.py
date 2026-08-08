#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Below-floor upward migration: N_true < 19.5 → N̂ > 19.5 (PI §3).

Source-term accounting for the low-boundary transport extension. Uses
the 17.2-chain event cache (built by `p1_chain_bridge.py`), with the
chain-competition classes SEPARATED per the bridge:

  * `raw_172`: every selected 17.2-chain TP with N_true < 19.5 — the
    17.2 chain's own attribution;
  * `net_migration`: the subset whose catalogue row is UNMATCHED in
    the nhi195 chain — the genuine below-floor attribution class
    (the same row being a ≥19.5-truth TP in the 195 chain is pure
    competition reassignment, reported separately, never absorbed).

Selection = the production observable: P_DLA > 0.99, DLAFLAG == 0,
S2N_RED > 2, N̂ > 19.5. For every observed bin j:
f_{<19.5→j} = N(true<19.5, N̂∈j, selected) / N(N̂∈j, selected),
where the denominator is ALL selected catalogue rows (TP of any truth
+ unattributed FP) — the observed catalogue content.

Representation decision implemented: the PRIMARY operator stays
truth-side N_true ≥ 19.5; the below-floor inflow is an EXPLICIT
source term (this table). K_natural-pairs is NOT renormalized.

Group propagation: observed-bin contamination fractions mapped onto
the frozen N̂ groups G1 [19.7,20.3) / G2 [20.3,21.0) / G3 [21.0,21.6]
by direct counting on the same selected catalogue.
"""
import json
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE172 = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
            "p1_completeness_cache_172.npz")
CACHE195 = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
            "p1_completeness_cache.npz")
FLOOR = 19.5
TRUE_RANGES = [(17.2, 19.3), (19.3, 19.5)]
OBS_BINS = [(19.5, 20.0), (20.0, 20.3), (20.3, 20.7), (20.7, 21.1),
            (21.1, 22.5)]
GROUPS = {"G1": (19.7, 20.3), "G2": (20.3, 21.0), "G3": (21.0, 21.6)}


def main():
    t0 = time.time()
    d = np.load(CACHE172)
    d5 = np.load(CACHE195)
    sel = ((d["cat_P_DLA"] > 0.99) & d["cat_good"] & (d["cat_S2N"] > 2.0)
           & (d["cat_NHI"] > FLOOR))
    nhat = d["cat_NHI"]
    tp = d["cat_is_TP"]
    ntr = d["cat_NHI_TRUE"]

    # 195-chain TP key set (competition separation)
    sel5 = ((d5["cat_P_DLA"] > 0.99) & d5["cat_good"]
            & (d5["cat_S2N"] > 2.0) & (d5["cat_NHI"] > FLOOR))
    tp195keys = set(zip(d5["cat_TARGETID"][d5["cat_is_TP"] & sel5].tolist(),
                        np.round(d5["cat_Z_DLA"][d5["cat_is_TP"] & sel5],
                                 6).tolist()))
    rowkeys = list(zip(d["cat_TARGETID"].tolist(),
                       np.round(d["cat_Z_DLA"], 6).tolist()))
    in195 = np.array([k in tp195keys for k in rowkeys])

    sub_tp = sel & tp & (ntr < FLOOR)
    net = sub_tp & ~in195          # genuine below-floor attribution
    reas = sub_tp & in195          # competition reassignment (excluded)

    out = {"schema": "p1_migration/v1", "date": time.strftime("%Y-%m-%d"),
           "representation": ("primary operator N_true>=19.5 + EXPLICIT "
                              "below-floor source term; no K "
                              "renormalization"),
           "n_subfloor_tp_selected": int(np.sum(sub_tp)),
           "n_reassigned_in_195_excluded": int(np.sum(reas)),
           "n_net_migration": int(np.sum(net)),
           "matrix": [], "observed_bins": [], "groups": {}}

    for tlo, thi in TRUE_RANGES:
        row = {"N_true": [tlo, thi], "cells": []}
        mt = net & (ntr >= tlo) & (ntr < thi)
        for olo, ohi in OBS_BINS:
            mo = mt & (nhat >= olo) & (nhat < ohi)
            row["cells"].append({"obs": [olo, ohi], "n": int(np.sum(mo))})
        row["n_total"] = int(np.sum(mt))
        out["matrix"].append(row)

    for olo, ohi in OBS_BINS:
        mo_all = sel & (nhat >= olo) & (nhat < ohi)
        mo_net = net & (nhat >= olo) & (nhat < ohi)
        mo_raw = sub_tp & (nhat >= olo) & (nhat < ohi)
        n_all = int(np.sum(mo_all))
        out["observed_bins"].append({
            "obs": [olo, ohi], "n_selected": n_all,
            "n_net_migrants": int(np.sum(mo_net)),
            "n_raw_172_migrants": int(np.sum(mo_raw)),
            "f_net": float(np.sum(mo_net) / n_all) if n_all else None,
            "f_raw": float(np.sum(mo_raw) / n_all) if n_all else None})

    for g, (glo, ghi) in GROUPS.items():
        mg_all = sel & (nhat >= glo) & (nhat < ghi)
        mg_net = net & (nhat >= glo) & (nhat < ghi)
        n_all = int(np.sum(mg_all))
        out["groups"][g] = {
            "obs": [glo, ghi], "n_selected": n_all,
            "n_net_migrants": int(np.sum(mg_net)),
            "f_net": float(np.sum(mg_net) / n_all) if n_all else None}

    # reporting-floor statement: observed [19.7, 20.0)
    m_rf = sel & (nhat >= 19.7) & (nhat < 20.0)
    out["reporting_floor_19p7_20p0"] = {
        "n_selected": int(np.sum(m_rf)),
        "n_net_migrants": int(np.sum(net & (nhat >= 19.7) & (nhat < 20.0))),
        "f_net": float(np.sum(net & (nhat >= 19.7) & (nhat < 20.0))
                       / max(np.sum(m_rf), 1))}
    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "p1_migration.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
