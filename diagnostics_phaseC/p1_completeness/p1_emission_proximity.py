#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quasar-rest-frame / emission-proximity conditional diagnostics
(PI §4–§6, 2026-08-07 amended ruling).

FROZEN DEFINITIONS (stated here BEFORE any conditional aggregate was
inspected; pre-selection quantities only):

  λ_abs^QSO = 1215.67 · (1 + z_abs) / (1 + z_qso)   [Å, quasar rest]

  Regions (mutually exclusive, assigned from λ_abs^QSO):
    LYA_EM    : λ^QSO ≥ 1200            (near quasar Lyα emission)
    LYB_EM    : 1025 ≤ λ^QSO < 1050     (near quasar Lyβ/OVI emission)
    INTERIOR  : 1050 ≤ λ^QSO < 1200     (inter-emission forest)
  EDGE flag (SEPARATE boolean, never collapsed into the regions):
    within 3,000 km/s of either analysis-window edge.
  DV_QSO (reported, not a region): c·(z_qso − z_abs)/(1 + z_abs).
  NOTE: for LYA_EM the emission-feature coordinate and physical
  proximity to the quasar are congruent by definition (λ^QSO ≥ 1200 ⇔
  dv_qso ≲ 3,900 km/s) — they cannot be separated at catalogue level;
  stated, not collapsed. For LYB_EM they are NOT congruent (near-Lyβ =
  far from the quasar), which is what makes the pair discriminating.

  N_true ranges: LOW [19.5,20.0), [20.0,20.4); CONTROL [20.4,21.0),
  [21.0,21.7). z_qso bins: [·,2.8), [2.8,3.3), [3.3,·).
  Materiality (battery scale): 0.015 dex on a conditional K-mean
  difference. NON-ADJUDICABLE cell: n < 50 or se > 0.02 dex — such a
  null is 'non-adjudicable', never 'no dependence'.
  Post-selection variables (N̂, scores) are NOT used for balancing.

Diagnostics: (1) truth occupancy, (2) completeness, (3) subfloor rate,
(4) below-floor upward migration by region (172 cache), (5) K mean,
(6) K width, (7) miss composition, (8) sibling-candidate rate.
Development data only; no holdout rows.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "injection"))
sys.path.insert(0, _REPO)

from build_p1_natpair_ck import extract_kernel_events, C_KMS   # noqa: E402
from gen_phaseC_resp import analysis_window                    # noqa: E402

CACHE172 = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
            "p1_completeness_cache_172.npz")
LYA = 1215.67
REG_EDGES = dict(LYB_EM=(1025.0, 1050.0), INTERIOR=(1050.0, 1200.0),
                 LYA_EM=(1200.0, 1e9))
EDGE_KMS = 3000.0
NRANGES = [(19.5, 20.0), (20.0, 20.4), (20.4, 21.0), (21.0, 21.7)]
ZQ_BINS = [(0.0, 2.8), (2.8, 3.3), (3.3, 99.0)]
MATERIALITY_DEX = 0.015
NMIN, SEMAX = 50, 0.02


def lam_qso(z_abs, z_qso):
    return LYA * (1.0 + np.asarray(z_abs)) / (1.0 + np.asarray(z_qso))


def region_of(z_abs, z_qso):
    lam = lam_qso(z_abs, z_qso)
    out = np.full(len(lam), "OUT", dtype=object)
    for name, (lo, hi) in REG_EDGES.items():
        out[(lam >= lo) & (lam < hi)] = name
    return out


def edge_flag(z_abs, z_qso):
    z_abs = np.asarray(z_abs, float)
    z_qso = np.asarray(z_qso, float)
    fl = np.zeros(len(z_abs), bool)
    for i in range(len(z_abs)):
        lo, hi = analysis_window(float(z_qso[i]))
        dv = min(abs(z_abs[i] - lo), abs(z_abs[i] - hi)) \
            / (1.0 + z_abs[i]) * C_KMS
        fl[i] = dv < EDGE_KMS
    return fl


def st(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return dict(n=int(n), mean=None, se=None, sd=None)
    return dict(n=int(n), mean=float(v.mean()),
                se=float(v.std(ddof=1) / np.sqrt(n)),
                sd=float(v.std(ddof=1)))


def adjudicable(s):
    return s["n"] >= NMIN and s["se"] is not None and s["se"] <= SEMAX


def main():
    t0 = time.time()
    ev, d = extract_kernel_events()
    # sightline z_qso map from the truth table
    zq_map = dict(zip(d["tr_TARGETID"].tolist(), d["tr_ZQSO"].tolist()))
    out = {"schema": "p1_emission_proximity/v1",
           "date": time.strftime("%Y-%m-%d"),
           "frozen": {"regions": {k: list(v) for k, v in REG_EDGES.items()},
                      "edge_kms": EDGE_KMS, "materiality_dex":
                      MATERIALITY_DEX, "nmin": NMIN, "semax": SEMAX}}

    # ---- truth side: occupancy / completeness / miss ------------------
    t_live = d["tr_S2N"] > 2.0
    tN, tZ = d["tr_NHI"][t_live], d["tr_Z"][t_live]
    tZQ = d["tr_ZQSO"][t_live]
    t_reg = region_of(tZ, tZQ)
    t_edge = edge_flag(tZ, tZQ)
    # matched flag per truth via kernel-event keys
    kkeys = set(zip(ev["TID"][ev["IN_KERNEL"]].tolist(),
                    np.round(ev["Z"][ev["IN_KERNEL"]], 6).tolist()))
    tTID = d["tr_TARGETID"][t_live]
    t_matched = np.array([(int(T), round(float(z), 6)) in kkeys
                          for T, z in zip(tTID, tZ)])
    subkeys = set(zip(ev["TID"][ev["CLS_SUBF"]].tolist(),
                      np.round(ev["Z"][ev["CLS_SUBF"]], 6).tolist()))
    t_subf = np.array([(int(T), round(float(z), 6)) in subkeys
                       for T, z in zip(tTID, tZ)])

    occ, comp = [], []
    for lo, hi in NRANGES:
        mN = (tN >= lo) & (tN < hi)
        row_o = {"N": [lo, hi], "n_truth": int(mN.sum())}
        row_c = {"N": [lo, hi]}
        for r in list(REG_EDGES) + ["EDGE"]:
            m = mN & ((t_reg == r) if r != "EDGE" else t_edge)
            n = int(m.sum())
            row_o[r] = {"n": n, "frac": float(n / max(mN.sum(), 1))}
            nm = int((m & t_matched).sum())
            nsf = int((m & t_subf).sum())
            row_c[r] = {"n": n, "C": float(nm / n) if n else None,
                        "C_se": float(np.sqrt(nm / n * (1 - nm / n) / n))
                        if n and 0 < nm < n else None,
                        "subfloor_rate": float(nsf / n) if n else None,
                        "unmatched_rate":
                        float(int((m & ~t_matched & ~t_subf).sum()) / n)
                        if n else None}
        occ.append(row_o)
        comp.append(row_c)
    out["occupancy_truth"] = occ
    out["completeness_missclass"] = comp

    # ---- pair side: K mean / width by region × N (+ z_qso) ------------
    kin = ev["IN_KERNEL"] & (ev["S2N"] > 2.0)
    pZQ = np.array([zq_map.get(int(t), np.nan) for t in ev["TID"]])
    pv = kin & np.isfinite(pZQ)
    pN, pZ, pDX = ev["N"][pv], ev["Z"][pv], ev["DX"][pv]
    pZQv = pZQ[pv]
    p_reg = region_of(pZ, pZQv)
    p_edge = edge_flag(pZ, pZQv)
    kk = []
    for lo, hi in NRANGES:
        mN = (pN >= lo) & (pN < hi)
        row = {"N": [lo, hi]}
        base = st(pDX[mN & (p_reg == "INTERIOR")])
        row["INTERIOR"] = base
        for r in ["LYA_EM", "LYB_EM", "EDGE"]:
            m = mN & ((p_reg == r) if r != "EDGE" else p_edge)
            s = st(pDX[m])
            if s["mean"] is not None and base["mean"] is not None:
                dz = ((s["mean"] - base["mean"])
                      / np.hypot(s["se"], base["se"]))
                s["delta_vs_interior"] = s["mean"] - base["mean"]
                s["z_vs_interior"] = float(dz)
            s["adjudicable"] = adjudicable(s)
            row[r] = s
        for zlo, zhi in ZQ_BINS:
            m = mN & (pZQv >= zlo) & (pZQv < zhi)
            s = st(pDX[m])
            s["adjudicable"] = adjudicable(s)
            row[f"zqso[{zlo},{zhi})"] = s
        kk.append(row)
    out["kernel_by_region"] = kk

    # ---- sibling-candidate rate per region (merge/split proxy) --------
    tid_all, z_all = d["cat_TARGETID"], d["cat_Z_DLA"]
    selc = (d["cat_good"] & (d["cat_P_DLA"] > 0.99)
            & (d["cat_NHI"] > 19.5) & (d["cat_S2N"] > 2.0))
    from collections import defaultdict
    bytid = defaultdict(list)
    for T, zz in zip(tid_all[selc].tolist(), z_all[selc].tolist()):
        bytid[T].append(zz)
    sib = np.zeros(int(pv.sum()), bool)
    pTID = ev["TID"][pv]
    pZD = ev["Z"][pv]
    for i in range(len(pTID)):
        for zz in bytid.get(int(pTID[i]), []):
            dv = abs(zz - pZD[i]) / (1 + pZD[i]) * C_KMS
            if 1e-9 < dv < 5000.0:
                sib[i] = True
                break
    srow = {}
    for r in list(REG_EDGES):
        m = p_reg == r
        srow[r] = {"n": int(m.sum()),
                   "sibling_rate": float(sib[m].mean()) if m.sum() else None}
    out["sibling_rate_by_region"] = srow

    # ---- below-floor migration by region (172 cache) ------------------
    d2 = np.load(CACHE172)
    sel2 = ((d2["cat_P_DLA"] > 0.99) & d2["cat_good"]
            & (d2["cat_S2N"] > 2.0) & (d2["cat_NHI"] > 19.5))
    mig = sel2 & d2["cat_is_TP"] & (d2["cat_NHI_TRUE"] < 19.5)
    mzq = d2["cat_ZQSO"][mig]
    mreg = region_of(d2["cat_Z_TRUE"][mig], mzq)
    denom_reg = region_of(d2["cat_Z_DLA"][sel2], d2["cat_ZQSO"][sel2])
    obs_low = (d2["cat_NHI"][sel2] >= 19.5) & (d2["cat_NHI"][sel2] < 20.0)
    mig_low = (d2["cat_NHI"][mig] >= 19.5) & (d2["cat_NHI"][mig] < 20.0)
    mrow = {}
    for r in list(REG_EDGES):
        nd = int(np.sum(obs_low & (denom_reg == r)))
        nm = int(np.sum(mig_low & (mreg == r)))
        mrow[r] = {"n_obs_19p5_20": nd, "n_migrants": nm,
                   "f": float(nm / nd) if nd else None}
    out["belowfloor_migration_by_region_obs19p5_20"] = mrow

    # ---- transport verdict vs frozen materiality ----------------------
    worst = []
    for row in kk:
        for r in ["LYA_EM", "LYB_EM", "EDGE"]:
            s = row[r]
            if s.get("adjudicable") and "delta_vs_interior" in s:
                worst.append((abs(s["delta_vs_interior"]), row["N"], r,
                              s["delta_vs_interior"], s["z_vs_interior"]))
    worst.sort(reverse=True)
    out["transport_assessment"] = {
        "materiality_dex": MATERIALITY_DEX,
        "worst_adjudicable": [
            {"N": w[1], "region": w[2], "delta": w[3], "z": w[4]}
            for w in worst[:6]],
        "any_material": bool(worst and worst[0][0] > MATERIALITY_DEX)}

    out["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(_HERE, "p1_emission_proximity.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("occupancy (truth, per N):")
    for r in occ:
        print(" ", r["N"], {k: round(r[k]["frac"], 3)
                            for k in list(REG_EDGES) + ["EDGE"]})
    print("K mean deltas vs INTERIOR (adjudicable only):")
    for w in out["transport_assessment"]["worst_adjudicable"]:
        print(f"  N{w['N']} {w['region']}: {w['delta']:+.4f} (z={w['z']:+.1f})")
    print("any_material:", out["transport_assessment"]["any_material"])
    print("migration by region f(19.5-20):",
          {k: (v['f'] and round(v['f'], 3)) for k, v in mrow.items()})


if __name__ == "__main__":
    main()
