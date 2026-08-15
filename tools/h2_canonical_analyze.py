#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2-v2 CANONICAL analysis under the Paper-1 metric contract (v1, 2026-08-15).

Reanalyzes the FROZEN H2 arm finder outputs (no finder rerun; the finder
searched the full modeled window, and the Lyalpha-only / LyAB distinction is
a downstream lambda_rf(Z_DLA) sample cut — molly_faithful_pc_plots.py:110)
under the canonical Molly/Paper-1 contract:

  sample    : SNR_REDSIDE > 2 (per-sightline; validated bit-identical to the
              dlacat column when recomputed from the archive), P_DLA > 0.99,
              DLAFLAG == 0, lambda_rf window (lya_only [1025,1216] primary /
              lya_lyb [911,1216] sensitivity) with 3000 km/s collars —
              identical formula to make_lambda_z_BAL_cuts; optional BAL drop.
  matcher   : examples.molly_faithful_pc_plots.match_truth_to_cat_molly
              (cat NHI-descending, |dz|/(1+z_true) < 0.01, tie-break min
              |dNHI|, greedy 1-to-1) — the ratified matcher of record.
  metrics   : DETECTION completeness; REPORTING completeness at T=20.0/20.3;
              migration decomposition; dlogN/dz response; NON-INJECTION
              DETECTIONS (never labeled FP — real-spectrum substrate).
  intervals : sightline-cluster bootstrap (parent sightline = dependence
              unit; injections on one sightline are NOT IID), 68% pct.

Usage:
  python tools/h2_canonical_analyze.py <realized_plan_arm.csv> <dlacat_glob>
         <injected_archive.h5> <window: lya_only|lya_lyb> <out.json>
         [--bal-tids <file with TARGETIDs to drop>]
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from examples.molly_faithful_pc_plots import (  # noqa: E402
    match_truth_to_cat_molly, LYA, SPEED_C)

WINDOWS = {"lya_only": (1025.0, 1216.0), "lya_lyb": (911.0, 1216.0)}
COLLAR = 3000.0 / SPEED_C
DZ_REL = 0.01
SNR_MIN = 2.0
PDLA_MIN = 0.99
T_LIST = [20.0, 20.3]
ZB = [(3.8, 4.25), (4.25, 4.5), (4.5, 5.0)]
NRE = [("19.5-20.0", 19.5, 20.0), ("20.0-20.3", 20.0, 20.3),
       ("20.3-20.7", 20.3, 20.7), ("20.7-21.1", 20.7, 21.1),
       ("21.1-21.5", 21.1, 21.51)]


def zwin(zq, lam_lo, lam_hi):
    z_lo = max(3600.0 / LYA - 1.0, lam_lo * (1 + zq) / LYA - 1 + COLLAR)
    z_hi = min(zq - COLLAR, lam_hi * (1 + zq) / LYA - 1 - COLLAR)
    return z_lo, z_hi


def main():
    plan_csv, cat_glob, arch_h5, window, outj = sys.argv[1:6]
    bal_tids = set()
    if "--bal-tids" in sys.argv:
        p = sys.argv[sys.argv.index("--bal-tids") + 1]
        bal_tids = {int(x) for x in open(p).read().split()}
    lam_lo, lam_hi = WINDOWS[window]

    # per-sightline sample: SNR_REDSIDE recomputed from the archive arrays
    # (RED_SNR is unpopulated in I/O-only archive builds; the recomputation
    # below is bit-identical to the finder's dlacat SNR_REDSIDE — validated
    # max|diff|=0 on all arm-A detections): mean(flux*sqrt(ivar)) over
    # unmasked rest-frame [1420,1480] A (dlasearch.py:670-676).
    h = h5py.File(arch_h5, "r")
    ac = h["catalog"][:]
    wave = h["wavelength"][:].astype(float)
    zq = {int(t): float(z) for t, z in zip(ac["TARGETID"], ac["Z"])}
    snr = {}
    for i, t in enumerate(ac["TARGETID"]):
        fl = h["flux"][i].astype(float)
        iv = h["ivar"][i].astype(float)
        wrf = wave / (1 + zq[int(t)])
        m = (iv != 0) & (wrf >= 1420) & (wrf <= 1480)
        snr[int(t)] = float(np.mean(fl[m] * np.sqrt(iv[m]))) if m.any() else np.nan
    in_sample = {t: (snr[t] > SNR_MIN) and (t not in bal_tids) for t in snr}

    # truth (frozen realized plan, this arm)
    truth_rows = []
    for r in csv.DictReader(open(plan_csv)):
        t = int(r["TARGETID"])
        zl, zh = zwin(zq[t], lam_lo, lam_hi)
        truth_rows.append(dict(
            tid=t, z=float(r["z_inj"]), n=float(r["logN"]), cell=r["cell"],
            in_window=(zl < float(r["z_inj"]) < zh),
            eligible=in_sample[t] and (zl < float(r["z_inj"]) < zh)))

    # cat (frozen finder output) + canonical selection
    cat_rows = []
    for f in sorted(glob.glob(cat_glob)):
        for r in fits.open(f)[1].data:
            t = int(r["TARGETID"])
            zl, zh = zwin(zq[t], lam_lo, lam_hi)
            sel = (float(r["P_DLA"]) > PDLA_MIN and int(r["DLAFLAG"]) == 0
                   and in_sample[t] and (zl < float(r["Z_DLA"]) < zh))
            cat_rows.append(dict(tid=t, z=float(r["Z_DLA"]),
                                 n=float(r["NHI"]), p=float(r["P_DLA"]),
                                 flag=int(r["DLAFLAG"]),
                                 snr=float(r["SNR_REDSIDE"]), selected=sel))

    sel_cat = [c for c in cat_rows if c["selected"]]
    elig = [t for t in truth_rows if t["eligible"]]

    # matcher of record on (selected cat) x (eligible truth)
    cat_t = Table(dict(TARGETID=[c["tid"] for c in sel_cat],
                       Z_DLA=[c["z"] for c in sel_cat],
                       NHI=[c["n"] for c in sel_cat]))
    tr_t = Table(dict(TARGETID=[t["tid"] for t in elig],
                      Z_DLA=[t["z"] for t in elig],
                      NHI=[t["n"] for t in elig]))
    if len(sel_cat) and len(elig):
        is_tp, nhi_tr, z_tr, truth_matched = match_truth_to_cat_molly(
            cat_t, tr_t, DZ_REL, cat_iter_order="nhi_desc")
    else:
        is_tp = np.zeros(len(sel_cat), bool)
        nhi_tr = np.full(len(sel_cat), np.nan)
        z_tr = np.full(len(sel_cat), np.nan)
        truth_matched = np.zeros(len(elig), bool)

    # per-truth outcome table
    # map matched truth -> its cat row (for measured NHI)
    match_of_truth = {}
    for ci in range(len(sel_cat)):
        if is_tp[ci]:
            key = (sel_cat[ci]["tid"], float(z_tr[ci]), float(nhi_tr[ci]))
            match_of_truth[key] = ci
    per_truth = []
    for ti, t in enumerate(elig):
        ci = match_of_truth.get((t["tid"], t["z"], t["n"]))
        per_truth.append(dict(
            tid=t["tid"], cell=t["cell"], z_true=t["z"], n_true=t["n"],
            detected=bool(truth_matched[ti]),
            n_meas=(sel_cat[ci]["n"] if ci is not None else None),
            z_meas=(sel_cat[ci]["z"] if ci is not None else None)))
    n_noninj = int(np.sum(~is_tp))

    # ---- metric projections -------------------------------------------------
    def boot(fn, units, B=1000, seed=20260815):
        """sightline-cluster bootstrap: resample parent sightlines."""
        rng = np.random.default_rng(seed)
        tids = sorted({u["tid"] for u in units})
        by = {}
        for u in units:
            by.setdefault(u["tid"], []).append(u)
        vals = []
        for _ in range(B):
            pick = rng.choice(tids, size=len(tids), replace=True)
            sample = [u for t in pick for u in by[t]]
            v = fn(sample)
            if v is not None:
                vals.append(v)
        if not vals:
            return None, None
        return float(np.percentile(vals, 16)), float(np.percentile(vals, 84))

    def det_C(rows):
        return (sum(r["detected"] for r in rows) / len(rows)) if rows else None

    def rep_C(rows, T):
        dom = [r for r in rows if r["n_true"] >= T]
        if not dom:
            return None
        k = sum(1 for r in dom if r["detected"] and r["n_meas"] is not None
                and r["n_meas"] >= T)
        return k / len(dom)

    def stratum(label, rows):
        k = sum(r["detected"] for r in rows)
        n = len(rows)
        lo, hi = boot(det_C, rows) if n else (None, None)
        rec = [r for r in rows if r["detected"]]
        dn = [r["n_meas"] - r["n_true"] for r in rec]
        dz = [r["z_meas"] - r["z_true"] for r in rec]
        return dict(stratum=label, n=n, k=k,
                    detection_C=round(k / n, 4) if n else None,
                    C_boot68=[round(lo, 4), round(hi, 4)] if lo is not None else None,
                    dlogN_mean=round(float(np.mean(dn)), 4) if dn else None,
                    dlogN_std=round(float(np.std(dn)), 4) if dn else None,
                    dz_mean=round(float(np.mean(dz)), 5) if dz else None)

    strata = [stratum("all", per_truth)]
    for a, b in ZB:
        strata.append(stratum(f"zbin:[{a},{b})",
                              [r for r in per_truth if a <= r["z_true"] < b]))
    for lab, a, b in NRE:
        strata.append(stratum(f"nre:{lab}",
                              [r for r in per_truth if a <= r["n_true"] < b]))
    for c in sorted({r["cell"] for r in per_truth}):
        strata.append(stratum(f"cell:{c}",
                              [r for r in per_truth if r["cell"] == c]))
    # Molly-matrix truth-NHI bins (the C-table interface used by the
    # real-data estimator: nhi_edges [19.5,20,20.3,20.5,21,21.5,22,inf])
    MOLLY_EDGES = [19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf]
    for a, b in zip(MOLLY_EDGES[:-1], MOLLY_EDGES[1:]):
        strata.append(stratum(f"molly_nhi:[{a},{b})",
                              [r for r in per_truth if a <= r["n_true"] < b]))

    reporting = []
    for T in T_LIST:
        dom = [r for r in per_truth if r["n_true"] >= T]
        k = sum(1 for r in dom if r["detected"] and r["n_meas"] is not None
                and r["n_meas"] >= T)
        det_k = sum(1 for r in dom if r["detected"])
        down = det_k - k
        lo, hi = boot(lambda rows, T=T: rep_C(rows, T), per_truth)
        # measured-side decomposition at T (estimand-purity numerator pieces)
        meas = [c for ci, c in enumerate(sel_cat) if c["n"] >= T]
        meas_tp_in = sum(1 for ci, c in enumerate(sel_cat)
                         if c["n"] >= T and is_tp[ci] and nhi_tr[ci] >= T)
        meas_up = sum(1 for ci, c in enumerate(sel_cat)
                      if c["n"] >= T and is_tp[ci] and nhi_tr[ci] < T)
        meas_noninj = sum(1 for ci, c in enumerate(sel_cat)
                          if c["n"] >= T and not is_tp[ci])
        reporting.append(dict(
            T=T, true_domain=f"logN_true>={T}",
            n_true=len(dom), detected_any=det_k,
            reported_in_domain=k, downward_migrated=down,
            not_detected=len(dom) - det_k,
            reporting_C=round(k / len(dom), 4) if dom else None,
            reporting_C_boot68=[round(lo, 4), round(hi, 4)] if lo is not None else None,
            detection_C=round(det_k / len(dom), 4) if dom else None,
            measured_side=dict(n_measured_ge_T=len(meas),
                               matched_true_in_domain=meas_tp_in,
                               upward_migrants_from_below_T=meas_up,
                               non_injection_detections=meas_noninj,
                               note="non-injection detections are NOT FP "
                                    "(real-spectrum substrate; contract §)")))

    out = dict(
        contract="CANONICAL_PURITY_COMPLETENESS_CONTRACT v1 (2026-08-15)",
        sample=("P1_PRIMARY_LYA" if window == "lya_only" else "P1_SENS_LYAB"),
        header=dict(snr="SNR_REDSIDE>2 (archive RED_SNR, bit-validated)",
                    p_dla="P_DLA>0.99", quality="DLAFLAG==0",
                    bal=("drop-listed" if bal_tids else
                         "none dropped (DLAFLAG POTENTIAL_BAL bit active; "
                         "BAL-drop sensitivity reported separately)"),
                    window=f"{window} lambda_rf [{lam_lo},{lam_hi}] "
                           f"collar 3000 km/s (make_lambda_z_BAL_cuts formula)",
                    matcher="match_truth_to_cat_molly nhi_desc dz_rel=0.01",
                    uncertainty="sightline-cluster bootstrap 68% (B=1000)",
                    dependence_unit="parent sightline"),
        counts=dict(
            plan_injections=len(truth_rows),
            truth_in_window=sum(t["in_window"] for t in truth_rows),
            truth_eligible=len(elig),
            sightlines_in_sample=sum(in_sample.values()),
            sightlines_total=len(in_sample),
            cat_rows_total=len(cat_rows), cat_rows_selected=len(sel_cat),
            matched=int(np.sum(is_tp)), non_injection_detections=n_noninj),
        reporting_completeness=reporting,
        detection_strata=strata)
    json.dump(out, open(outj, "w"), indent=1)
    print(f"[{out['sample']}] elig {len(elig)}/{len(truth_rows)} "
          f"sel_cat {len(sel_cat)} matched {int(np.sum(is_tp))} "
          f"non-inj {n_noninj} -> {outj}")


if __name__ == "__main__":
    main()
