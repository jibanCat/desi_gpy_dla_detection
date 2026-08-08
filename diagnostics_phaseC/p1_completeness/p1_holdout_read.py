#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ONE-TIME holdout read + frozen battery v2 execution (PI-authorized,
2026-08-07 ruling §6).

FIRST-TOUCH RULE: the holdout is CONSUMED at the first access to any
holdout row (= the single invocation of the committed measurement path
with --role held-out-evaluation --evaluation-step), not on successful
completion. On any failure after that invocation begins, this script
writes what it has, classifies F_imp, and STOPS; no rerun without a
new PI ruling. A failure in PRE-FLIGHT (before the invocation) leaves
the holdout sealed (zero-access, verifiable: the pairs output file
does not exist and the measurement subprocess was never launched).

FROZEN IMPLEMENTATION DETAILS (stated here, before the read; all
references come from the RATIFIED p1_holdout_battery_v2.json — nothing
is re-derived):

  * mean tests: two-sided z, var = sd_cal^2/n_hold + se_cal^2.
  * completeness tests: two-sided exact binomial (scipy binomtest;
    normal approximation only if scipy is unavailable, recorded).
  * subfloor test: matched-pair fraction with N̂ = N + dx <= 19.5 in
    [19.5, 20.0), two-sided binomial vs the pinned reference.
  * LYA_EM region test: holdout (LYA_EM - INTERIOR) mean-dx delta vs
    the pinned calibration delta (-0.0562); two-sided z.
  * joint-operator test: pooled 5-category chi^2 (miss / N̂<20.3 /
    G2 [20.3,21.0) / G3 [21.0,21.6) / N̂>=21.6) over design
    injections with logN_true in [20.4, 21.1); expected per injection:
    P(miss) = 1 - yield(bin); detected N̂ ~ Normal(N + mean_cal(bin),
    sd_cal(bin)); df = 4.
  * Holm within each family at alpha = 0.01. Primary family = 5 means
    + pooled + 5 completeness + joint (12 tests; width EXCLUDED =
    diagnostic). Low family = 2 means + 2 completeness + subfloor +
    LYA_EM (6 tests).
  * completeness "coherent failure" (primary no-go contribution):
    >= 2 Holm-rejected completeness bins, OR one Holm-rejected bin
    with |yield difference| >= 5 percentage points.
  * width diagnostic: two-sided variance-ratio at alpha 0.01; review
    flag; contributes to no-go ONLY if the sd ratio deviates from 1
    by more than 25% (battery rule).
  * exploratory (uncorrected, labeled): z_qso terciles (<2.8, 2.8-3.3,
    >=3.3), LYB_EM, EDGE, per-region high-N deltas.
  * verdict mapping: verbatim battery v2 global enumeration.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "injection"))

ARM = "/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/prod_v1"
PAIRS_OUT = os.path.join(ARM, "pairs_holdout.json")
BATTERY = os.path.join(_HERE, "p1_holdout_battery_v2.json")
ARTIFACT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
            "track_c/stage0/p1_natpair_ck_v1.npz")
PYG = "/home/mfho/.conda/envs/gpdla/bin/python"
ALPHA = 0.01
HI_BINS = [(20.4, 20.7), (20.7, 21.0), (21.0, 21.3), (21.3, 21.7),
           (21.7, 22.4)]
LO_BINS = [(19.5, 20.0), (20.0, 20.4)]
POOLED = (20.7, 21.1)
JOINT_RANGE = (20.4, 21.1)
CATS = [(-np.inf, 20.3), (20.3, 21.0), (21.0, 21.6), (21.6, np.inf)]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _phi(z):
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def p_two_z(z):
    return float(2.0 * (1.0 - _phi(abs(z))))


def p_binom(k, n, p0):
    try:
        from scipy.stats import binomtest
        return float(binomtest(int(k), int(n), float(p0),
                               alternative="two-sided").pvalue), "exact"
    except Exception:
        se = np.sqrt(p0 * (1 - p0) / n)
        return p_two_z((k / n - p0) / se), "normal-approx"


def holm(tests, alpha=ALPHA):
    m = len(tests)
    order = sorted(range(m), key=lambda i: tests[i]["p"])
    padj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * tests[i]["p"])
        padj[i] = min(1.0, running)
    for i in range(m):
        tests[i]["p_holm"] = padj[i]
        tests[i]["holm_reject"] = bool(padj[i] < alpha)
    return tests


def main():
    t0 = time.time()
    result = {"schema": "p1_holdout_result/v1",
              "date": time.strftime("%Y-%m-%d %H:%M:%S"),
              "authorization": "PI ruling 2026-08-07 (battery v2 RATIFIED; "
                               "one-time read AUTHORIZED)"}

    # ---------------- PRE-FLIGHT (no holdout access) --------------------
    git = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO,
                         capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                          capture_output=True, text=True).stdout.strip()
    bat = json.load(open(BATTERY))
    preflight = {
        "git_head": head,
        "git_dirty_files": [ln for ln in git.splitlines()],
        "battery_sha256": _sha(BATTERY),
        "battery_schema": bat["schema"],
        "artifact_sha256": _sha(ARTIFACT),
        "measure_script_sha256": _sha(os.path.join(
            _REPO, "injection/measure_phaseC_pairs.py")),
        "pairs_out_preexisting": os.path.exists(PAIRS_OUT)}
    result["preflight"] = preflight
    if bat["schema"] != "p1_holdout_battery/v2" or not bat["frozen"]:
        result["verdict"] = "PRE-READ IMPLEMENTATION FAILURE (battery)"
        _write(result)
        raise SystemExit("pre-flight battery check failed; holdout SEALED")
    if preflight["pairs_out_preexisting"]:
        result["verdict"] = "PRE-READ ABORT: pairs_holdout.json already " \
                            "exists — a prior read may have occurred"
        _write(result)
        raise SystemExit("pre-flight: prior output exists; NOT reading")

    # ---------------- THE ONE-TIME READ (first touch) -------------------
    result["read_started_utc"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cmd = [PYG, os.path.join(_REPO, "injection/measure_phaseC_pairs.py"),
           "--arm", ARM, "--role", "held-out-evaluation",
           "--evaluation-step", "--out", PAIRS_OUT]
    result["read_command"] = " ".join(cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    result["read_returncode"] = proc.returncode
    result["read_stderr_tail"] = proc.stderr[-2000:]
    if proc.returncode != 0 or not os.path.exists(PAIRS_OUT):
        result["verdict"] = "F_imp (read invoked; battery not completed)"
        result["holdout_consumed"] = True
        _write(result)
        raise SystemExit("F_imp: measurement path failed AFTER first "
                         "touch; holdout CONSUMED; stopping per §6.1")
    result["holdout_consumed"] = True
    pj = json.load(open(PAIRS_OUT))
    result["pairs_sha256"] = _sha(PAIRS_OUT)
    result["n_injected_scored"] = pj["n_injected"]
    result["n_matched_op"] = pj["n_matched_op"]

    # ---------------- per-pair arrays from the pairs JSON ---------------
    N, DX, ZT, ZQ = [], [], [], []
    n_inj_bin = {tuple(b): 0 for b in HI_BINS + LO_BINS}
    n_mop_bin = {tuple(b): 0 for b in HI_BINS + LO_BINS}
    n_inj_joint = 0
    for r in pj["per_anchor"]:
        A = float(r["logN"])
        for b in HI_BINS + LO_BINS:
            if b[0] <= A < b[1]:
                n_inj_bin[b] += r["n_inj"]
                n_mop_bin[b] += r["n_matched_op"]
        if JOINT_RANGE[0] <= A < JOINT_RANGE[1]:
            n_inj_joint += r["n_inj"]
        for k in range(len(r["dx"])):
            N.append(A)
            DX.append(float(r["dx"][k]))
            ZT.append(float(r["pair_z_true"][k]))
            ZQ.append(float(r["pair_z_qso"][k]))
    N, DX = np.array(N), np.array(DX)
    ZT, ZQ = np.array(ZT), np.array(ZQ)

    from p1_emission_proximity import region_of, edge_flag
    REG = region_of(ZT, ZQ)
    EDGE = edge_flag(ZT, ZQ)

    def mstats(m):
        v = DX[m]
        if len(v) < 2:
            return None
        return (float(v.mean()), float(v.std(ddof=1)), int(len(v)))

    # ---------------- PRIMARY FAMILY ------------------------------------
    prim = []
    mean_refs = {tuple(b["N"]): b["calibration"]
                 for b in bat["primary_family"]["tests"]["mean_per_bin"]}
    yields = {tuple(b["N"]): b["yield"]
              for b in bat["primary_family"]["tests"]
              ["completeness_per_bin"]}
    for b in HI_BINS:
        ref = mean_refs[b]
        s = mstats((N >= b[0]) & (N < b[1]))
        if s and ref["mean"] is not None:
            se = np.sqrt(ref["sd"] ** 2 / s[2] + ref["se"] ** 2)
            z = (s[0] - ref["mean"]) / se
            prim.append({"test": f"mean[{b[0]},{b[1]})", "hold": s[0],
                         "cal": ref["mean"], "n": s[2], "z": float(z),
                         "p": p_two_z(z)})
        else:
            prim.append({"test": f"mean[{b[0]},{b[1]})", "hold": None,
                         "p": 1.0, "note": "insufficient n (recorded)"})
    refp = bat["primary_family"]["tests"]["pooled_critical"]["calibration"]
    sp = mstats((N >= POOLED[0]) & (N < POOLED[1]))
    zp = (sp[0] - refp["mean"]) / np.sqrt(refp["sd"] ** 2 / sp[2]
                                          + refp["se"] ** 2)
    prim.append({"test": "pooled[20.7,21.1)", "hold": sp[0],
                 "cal": refp["mean"], "n": sp[2], "z": float(zp),
                 "p": p_two_z(zp)})
    comp_rows = []
    for b in HI_BINS:
        y0 = yields[b]
        k, n = n_mop_bin[b], n_inj_bin[b]
        p, kind = p_binom(k, n, y0)
        row = {"test": f"C[{b[0]},{b[1]})", "hold": k / n, "cal": y0,
               "k": k, "n": n, "p": p, "binom": kind,
               "abs_diff": abs(k / n - y0)}
        prim.append(row)
        comp_rows.append(row)
    # joint-operator chi^2
    obs = np.zeros(5)
    exp = np.zeros(5)
    for r in pj["per_anchor"]:
        A = float(r["logN"])
        if not (JOINT_RANGE[0] <= A < JOINT_RANGE[1]):
            continue
        b = next(bb for bb in HI_BINS if bb[0] <= A < bb[1])
        ref = mean_refs[b]
        y0 = yields[b]
        n_inj = r["n_inj"]
        exp[0] += n_inj * (1 - y0)
        mu, sd = A + ref["mean"], ref["sd"]
        edges = [20.3, 21.0, 21.6]
        cdf = [_phi((e - mu) / sd) for e in edges]
        probs = [cdf[0], cdf[1] - cdf[0], cdf[2] - cdf[1], 1 - cdf[2]]
        for ci in range(4):
            exp[1 + ci] += n_inj * y0 * probs[ci]
        obs[0] += r["n_inj"] - r["n_matched_op"]
        for dx in r["dx"]:
            nh = A + float(dx)
            ci = 0 if nh < 20.3 else (1 if nh < 21.0
                                      else (2 if nh < 21.6 else 3))
            obs[1 + ci] += 1
    chi2 = float(np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-9)))
    try:
        from scipy.stats import chi2 as chi2d
        pj_p = float(chi2d.sf(chi2, 4))
    except Exception:
        pj_p = p_two_z(np.sqrt(max(chi2 - 3, 0)))     # rough fallback
    prim.append({"test": "joint_operator[20.4,21.1)", "chi2": chi2,
                 "df": 4, "obs": obs.tolist(),
                 "exp": [float(x) for x in exp], "p": pj_p})
    prim = holm(prim)
    # width diagnostic (NOT in Holm)
    width_diag = []
    for b in HI_BINS:
        ref = mean_refs[b]
        s = mstats((N >= b[0]) & (N < b[1]))
        if s and ref["sd"]:
            ratio = s[1] / ref["sd"]
            width_diag.append({"bin": list(b), "ratio": float(ratio),
                               "flag_review": bool(abs(ratio - 1) > 0.25)})
    mean_joint_fail = any(t["holm_reject"] for t in prim
                          if t["test"].startswith(("mean", "pooled",
                                                   "joint")))
    comp_fail_bins = [r for r in comp_rows if r.get("holm_reject")]
    comp_coherent = (len(comp_fail_bins) >= 2
                     or any(r["abs_diff"] >= 0.05 for r in comp_fail_bins))
    width_no_go = any(abs(w["ratio"] - 1) > 0.25 for w in width_diag)
    primary_pass = not (mean_joint_fail or comp_coherent or width_no_go)
    result["primary_family"] = {
        "tests": prim, "width_diagnostic": width_diag,
        "mean_or_joint_holm_failure": mean_joint_fail,
        "completeness_coherent_failure": comp_coherent,
        "width_review_flags": [w for w in width_diag if w["flag_review"]],
        "verdict": ("HIGH-N PREDICTIVE PASS" if primary_pass
                    else "F_pre (HIGH-N PREDICTIVE NO-GO)")}

    # ---------------- LOW-BOUNDARY FAMILY -------------------------------
    low = []
    lo_refs = {tuple(b["N"]): b["calibration"]
               for b in bat["low_boundary_family"]["tests"]["mean_per_bin"]}
    lo_yields = {tuple(b["N"]): b["yield"]
                 for b in bat["low_boundary_family"]["tests"]
                 ["completeness_per_bin"]}
    for b in LO_BINS:
        ref = lo_refs[b]
        s = mstats((N >= b[0]) & (N < b[1]))
        if s and ref["mean"] is not None:
            se = np.sqrt(ref["sd"] ** 2 / s[2] + ref["se"] ** 2)
            z = (s[0] - ref["mean"]) / se
            low.append({"test": f"mean[{b[0]},{b[1]})", "hold": s[0],
                        "cal": ref["mean"], "n": s[2], "z": float(z),
                        "p": p_two_z(z)})
        k, n = n_mop_bin[b], n_inj_bin[b]
        p, kind = p_binom(k, n, lo_yields[b])
        low.append({"test": f"C[{b[0]},{b[1]})", "hold": k / n,
                    "cal": lo_yields[b], "k": k, "n": n, "p": p,
                    "binom": kind,
                    "fallback_reference": bool(
                        [x for x in bat["low_boundary_family"]["tests"]
                         ["completeness_per_bin"] if tuple(x["N"]) == b
                         and x.get("fallback")])})
    sref = bat["low_boundary_family"]["tests"]["subfloor_rate"][
        "reference"]["[19.5,20.0)"]
    m = (N >= 19.5) & (N < 20.0)
    ksub = int(np.sum((N[m] + DX[m]) <= 19.5))
    p, kind = p_binom(ksub, int(m.sum()), sref["rate"])
    low.append({"test": "subfloor_rate[19.5,20.0)",
                "hold": ksub / max(m.sum(), 1), "cal": sref["rate"],
                "k": ksub, "n": int(m.sum()), "p": p, "binom": kind})
    lref = bat["low_boundary_family"]["tests"]["lya_em_region"]["reference"]
    mi = REG == "INTERIOR"
    ml = REG == "LYA_EM"
    si_, sl_ = mstats(mi), mstats(ml)
    if si_ and sl_:
        delta = sl_[0] - si_[0]
        se = np.sqrt(sl_[1] ** 2 / sl_[2] + si_[1] ** 2 / si_[2])
        z = (delta - lref["delta"]) / se
        low.append({"test": "lya_em_delta", "hold": float(delta),
                    "cal": lref["delta"], "n_lya": sl_[2],
                    "z": float(z), "p": p_two_z(z)})
    else:
        low.append({"test": "lya_em_delta", "hold": None, "p": 1.0,
                    "note": "insufficient n -> non-adjudicable"})
    low = holm(low)
    low_fail = [t["test"] for t in low if t.get("holm_reject")]
    result["low_boundary_family"] = {
        "tests": low, "holm_failures": low_fail,
        "verdict": ("LOW-BOUNDARY CONSISTENT" if not low_fail else
                    "LOW-BOUNDARY QUALIFICATION (see mapping)")}

    # ---------------- EXPLORATORY (uncorrected, labeled) ----------------
    expl = {}
    for name, m in [("LYB_EM", REG == "LYB_EM"), ("EDGE", EDGE),
                    ("zqso<2.8", ZQ < 2.8),
                    ("zqso[2.8,3.3)", (ZQ >= 2.8) & (ZQ < 3.3)),
                    ("zqso>=3.3", ZQ >= 3.3),
                    ("LYA_EM_highN", (REG == "LYA_EM") & (N >= 20.4)),
                    ("INTERIOR_highN", (REG == "INTERIOR") & (N >= 20.4))]:
        s = mstats(m)
        expl[name] = {"n": s[2], "mean": s[0], "sd": s[1]} if s else None
    result["exploratory"] = {"label": "EXPLORATORY ONLY — no verdict "
                             "weight; cannot reject/tune/promote",
                             "cells": expl}

    # ---------------- GLOBAL VERDICT ------------------------------------
    result["global_verdict"] = (
        "HIGH-N PREDICTIVE PASS" if primary_pass else "F_pre")
    result["global_notes"] = {
        "width_no_go_triggered": width_no_go,
        "low_boundary_maps_to_own_outcomes_only": True,
        "gatekeeping": bat["gatekeeping"]}
    result["wall_s"] = round(time.time() - t0, 1)
    _write(result)
    print(json.dumps({
        "global_verdict": result["global_verdict"],
        "primary": result["primary_family"]["verdict"],
        "low": result["low_boundary_family"]["verdict"],
        "n_scored": result["n_injected_scored"],
        "n_matched_op": result["n_matched_op"]}, indent=1))
    for t in prim:
        print(" PRIM", t["test"], "p=", round(t["p"], 5),
              "holm=", round(t["p_holm"], 5),
              "REJECT" if t["holm_reject"] else "")
    for t in low:
        print(" LOW ", t["test"], "p=", round(t["p"], 5),
              "holm=", round(t["p_holm"], 5),
              "REJECT" if t["holm_reject"] else "")


def _write(result):
    with open(os.path.join(_HERE, "p1_holdout_result.json"), "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
