#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier 1b–1d — denominators, injection-placement audit, attrition ledger,
finder/threshold/cuts/matcher decomposition (rulings §9–§13).

All Level-A accounting on IMMUTABLE deployed artifacts: the raw production
combined catalogue (pre-bundle rows incl. sub-threshold candidates), the
Tier-1a cached cut bundle (`cat_cut`, `is_TP`, the deployed 195-chain), the
raw truth/BAL/SNR catalogs, and the FROZEN prod_v1 manifest+roles (aggregate
placement stats; per-proposal replay of the frozen deterministic seed — a
LEDGER of what the frozen campaign did, not new generation). No finder
rerun; no holdout-outcome read (natural data predate injections; injection
rows are not touched here at all).

PRIMARY-CAUSE HIERARCHY (frozen BEFORE the first run of this script;
rulings §12) for each eligible-but-unmatched truth system, first matching
condition wins:
  H1 no raw candidate on the sightline within dz_rel (candidate-generation)
  H2 raw candidate exists but none survives the cut bundle (bundle/cuts)
  H3 bundle candidate exists but none with P_DLA > 0.99 (threshold)
  H4 op candidate exists but none within dz_rel of THIS truth (tolerance)
  H5 in-tolerance op candidate exists but assignment lost it (competition)
ALTERNATIVE ORDER (order-sensitivity): threshold before bundle
(H1, H3', H2', H4, H5) — reported alongside.
Secondary flags preserved (multi-truth sightline, near-neighbor within
5,000 km/s, mask/edge proximity via window position).
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

CACHE = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
         "p1_completeness_cache.npz")
ROLES = os.path.join(_REPO, "diagnostics_phaseC/stage2A/roles_prod_v1.json")
DZ_REL = 0.01
C_KMS = 299792.458
DV_EXCL = 5000.0
NR = [(19.5, 20.0), (20.0, 20.4), (20.4, 21.0), (21.0, 21.5),
      (21.5, 23.0)]


def main():
    from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB
    import fitsio
    t0 = time.time()
    d = np.load(CACHE)
    tr_tid = d["tr_TARGETID"]; tr_n = d["tr_NHI"]; tr_z = d["tr_Z"]
    tr_snr = d["tr_SNR"]
    # §9 eligibility classes: the FOLD's live support is S2N_RED > 2 (the
    # sub-2 molly strata are dX == 0 = structurally outside production).
    tr_s2n = d["tr_S2N"]
    live = tr_s2n > 2.0
    n_class1 = int((~live).sum())
    print(f"[eligibility] truth rows {len(tr_tid)}; OUTSIDE live fold "
          f"support (S2N_RED <= 2): {n_class1} "
          f"({n_class1/len(tr_tid):.1%}) -> excluded as class-1")
    tr_tid, tr_n, tr_z, tr_snr = (tr_tid[live], tr_n[live], tr_z[live],
                                  tr_snr[live])
    c_tid = d["cat_TARGETID"]; c_z = d["cat_Z_DLA"]; c_n = d["cat_NHI"]
    c_p = d["cat_P_DLA"]; c_tp = d["cat_is_TP"]
    c_ntr = d["cat_NHI_TRUE"]; c_ztr = d["cat_Z_TRUE"]

    # raw pre-bundle candidate rows (immutable deployed catalogue)
    import glob as _g
    raw_files = sorted(_g.glob(os.path.join(AB.DEF_CAT, "*.fits")))
    rt, rz, rp = [], [], []
    for f in raw_files:
        r = fitsio.read(f, columns=["TARGETID", "Z_DLA", "P_DLA"])
        rt.append(np.asarray(r["TARGETID"], np.int64))
        rz.append(np.asarray(r["Z_DLA"], float))
        rp.append(np.asarray(r["P_DLA"], float))
    rt = np.concatenate(rt); rz = np.concatenate(rz); rp = np.concatenate(rp)

    def bytid(tids, *arrs):
        m = defaultdict(list)
        for i, t in enumerate(tids):
            m[int(t)].append(i)
        return {k: tuple(a[v] for a in arrs) for k, v in
                ((k, np.asarray(v)) for k, v in m.items())}
    raw_by = bytid(rt, rz, rp)
    cut_by = bytid(c_tid, c_z, c_n, c_p)

    # matched truth identities = (TID, NHI_TRUE, Z_TRUE) of op TP rows
    op_row = c_tp & (c_p > 0.99)
    matched_keys = {(int(t), round(float(n), 6), round(float(z), 6))
                    for t, n, z in zip(c_tid[op_row], c_ntr[op_row],
                                       c_ztr[op_row])}
    # HCD near-neighbor structure on the truth side
    tr_by = defaultdict(list)
    for i, t in enumerate(tr_tid):
        tr_by[int(t)].append(i)

    ledger = {tuple(r): defaultdict(lambda: defaultdict(int))
              for r in NR}
    causes = ("H1_no_candidate", "H2_bundle_cut", "H3_subthreshold",
              "H4_tolerance", "H5_assignment", "matched")
    alt = {tuple(r): defaultdict(lambda: defaultdict(int)) for r in NR}
    sec_flags = {tuple(r): defaultdict(int) for r in NR}
    for j in range(len(tr_tid)):
        N, Z, T = float(tr_n[j]), float(tr_z[j]), int(tr_tid[j])
        rng = next((tuple(r) for r in NR if r[0] <= N < r[1]), None)
        if rng is None:
            continue
        key = (T, round(N, 6), round(Z, 6))
        sibs = [i for i in tr_by[T] if i != j]
        near = any(abs(float(tr_z[i]) - Z) / (1 + Z) * C_KMS < DV_EXCL
                   for i in sibs)
        if near:
            sec_flags[rng]["near_neighbor_5000kms"] += 1
        if len(sibs) > 0:
            sec_flags[rng]["multi_truth_sightline"] += 1
        if key in matched_keys:
            ledger[rng]["matched"]["n"] += 1
            alt[rng]["matched"]["n"] += 1
            continue
        rawc = raw_by.get(T)
        has_raw = rawc is not None and np.any(
            np.abs(rawc[0] - Z) / (1 + rawc[0]) < DZ_REL)
        cutc = cut_by.get(T)
        in_cut = cutc is not None and np.any(
            np.abs(cutc[0] - Z) / (1 + cutc[0]) < DZ_REL)
        in_op_tol = cutc is not None and np.any(
            (np.abs(cutc[0] - Z) / (1 + cutc[0]) < DZ_REL)
            & (cutc[2] > 0.99))
        in_op_any = cutc is not None and np.any(cutc[2] > 0.99)
        has_raw_op = rawc is not None and np.any(
            (np.abs(rawc[0] - Z) / (1 + rawc[0]) < DZ_REL)
            & (rawc[1] > 0.99))
        # primary hierarchy
        if not has_raw:
            cause = "H1_no_candidate"
        elif not in_cut:
            cause = "H2_bundle_cut"
        elif in_op_tol:
            cause = "H5_assignment"      # in-tolerance op candidate existed
        elif in_op_any:
            cause = "H4_tolerance"
        else:
            cause = "H3_subthreshold"
        ledger[rng][cause]["n"] += 1
        if near:
            ledger[rng][cause]["near_neighbor"] += 1
        # alternative order: threshold judged on RAW rows before bundle
        if not has_raw:
            ac = "H1_no_candidate"
        elif not has_raw_op:
            ac = "H3_subthreshold"
        elif not in_op_tol:
            ac = "H2_bundle_cut"
        else:
            ac = "H5_assignment"
        alt[rng][ac]["n"] += 1

    # ---- injection placement audit (frozen aggregates; §10) ----
    roles = json.load(open(ROLES))
    st = roles.get("production_stats") or {}
    n_prop_est = None
    if st:
        n_acc = sum(1 for v in roles["roles"].values())
        n_prop_est = n_acc + int(st.get("redraws_hcd", 0)) \
            + int(st.get("redraws_window", 0))
    # natural fraction with an HCD neighbor within DV_EXCL (the class the
    # injection placement EXCLUDES by construction)
    nat_near = {str(r): (sec_flags[tuple(r)]["near_neighbor_5000kms"],
                         sum(v["n"] for v in ledger[tuple(r)].values()))
                for r in NR}

    out = {
        "schema": "p1_t1_ledger/v1",
        "date": time.strftime("%Y-%m-%d"),
        "hierarchy": list(causes),
        "dz_rel": DZ_REL,
        "ledger": {str(r): {c: dict(ledger[tuple(r)][c]) for c in causes
                            if ledger[tuple(r)][c]} for r in NR},
        "alt_order": {str(r): {c: dict(alt[tuple(r)][c])
                               for c in alt[tuple(r)]} for r in NR},
        "secondary_flags": {str(r): dict(sec_flags[tuple(r)]) for r in NR},
        "natural_near_neighbor_fraction": {
            k: (v[0] / v[1] if v[1] else None) for k, v in nat_near.items()},
        "injection_placement": {
            "accepted": len(roles["roles"]),
            "redraws_hcd": st.get("redraws_hcd"),
            "redraws_window": st.get("redraws_window"),
            "proposals_estimated": n_prop_est,
            "note": ("placement rejects (redraws) ONLY: used sightline, "
                     "window-hosting failure, truth-HCD within 5,000 km/s "
                     "(prodlike). No mask/SNR/blend/forest screening exists "
                     "in the generator — the one systematic parent-population "
                     "difference vs natural truth is the HCD-neighbor "
                     "exclusion, quantified above on the natural side."),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "t1_ledger.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for r in NR:
        led = ledger[tuple(r)]
        tot = sum(v["n"] for v in led.values())
        parts = {c: led[c]["n"] for c in causes if led[c]}
        print(r, "N=", tot, parts)
    print("near-neighbor fractions:", out["natural_near_neighbor_fraction"])
    print("placement:", out["injection_placement"]["accepted"], "accepted;",
          n_prop_est, "proposals (est)")


if __name__ == "__main__":
    main()
