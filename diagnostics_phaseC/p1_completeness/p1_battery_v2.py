#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HIERARCHICAL holdout battery v2 + GLOBAL VERDICT RULE (PI §7–§8,
2026-08-07 amended). Supersedes p1_holdout_battery.json (v1 retained as
record). Calibration+design side only; NO holdout outcome touched.

Structure: primary high-N family (the load-bearing P1 predictive
gate), low-boundary transport family, exploratory subgroup
diagnostics. The global verdict mapping is written HERE, before any
holdout row is read; nothing is left to post-read discretion.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "injection"))

from t2_pairing import truth_neighbors, injected_pairs, ARM  # noqa: E402
from p1_emission_proximity import region_of                  # noqa: E402
from astropy.table import Table                              # noqa: E402

HI_BINS = [(20.4, 20.7), (20.7, 21.0), (21.0, 21.3), (21.3, 21.7),
           (21.7, 22.4)]
LO_BINS = [(19.5, 20.0), (20.0, 20.4)]
POOLED = (20.7, 21.1)


def _stats(v):
    v = np.asarray(v, float)
    if len(v) < 5:
        return dict(n=int(len(v)), mean=None, se=None, sd=None)
    return dict(n=int(len(v)), mean=float(v.mean()),
                se=float(v.std(ddof=1) / np.sqrt(len(v))),
                sd=float(v.std(ddof=1)))


def main():
    by = truth_neighbors()
    inj = injected_pairs(by)
    man = Table.read(os.path.join(ARM, "injection_truth.fits"))
    zk = {round(float(r["z_true"]), 6): float(r["z_qso"]) for r in man}
    za = np.array([r["z"] for r in inj])
    dx = np.array([r["dx"] for r in inj])
    N = np.array([r["N"] for r in inj])
    zq = np.array([zk.get(round(z, 6), np.nan) for z in za])
    v = np.isfinite(zq)
    reg = region_of(za[v], zq[v])
    gate = json.load(open(os.path.join(_HERE, "p1_holdout_gate.json")))
    yields = {tuple(b["N"]): b["calib_pair_yield"] for b in gate["bins"]}

    def bins_ref(bins):
        return [{"N": [lo, hi], "calibration": _stats(dx[(N >= lo)
                                                        & (N < hi)]),
                 "yield": yields.get((lo, hi))}
                for lo, hi in bins]

    # calibration subfloor rate at the low bins (matched, N̂<=19.5)
    nhat = N + dx
    sub_lo = {}
    for lo, hi in LO_BINS:
        m = (N >= lo) & (N < hi)
        sub_lo[f"[{lo},{hi})"] = {
            "rate": float(np.sum(m & (nhat <= 19.5)) / max(m.sum(), 1)),
            "n": int(m.sum())}
    # calibration LYA_EM region reference (pooled over N)
    base = _stats(dx[v][reg == "INTERIOR"])
    lya = _stats(dx[v][reg == "LYA_EM"])
    lya_ref = {"interior": base, "lya_em": lya,
               "delta": lya["mean"] - base["mean"]}

    battery = {
        "schema": "p1_holdout_battery/v2",
        "date": time.strftime("%Y-%m-%d"),
        "supersedes": "p1_holdout_battery/v1 (retained as record)",
        "frozen": True, "no_holdout_outcome_used": True,
        "primary_support_truth": "N_true >= 20.3",
        "primary_family": {
            "role": "the load-bearing P1 predictive gate (confirmatory)",
            "alpha_family": 0.01, "multiplicity": "Holm within family",
            "tests": {
                "mean_per_bin": bins_ref(HI_BINS),
                "pooled_critical": {"N": list(POOLED),
                                    "calibration": _stats(
                                        dx[(N >= POOLED[0])
                                           & (N < POOLED[1])])},
                "completeness_per_bin": [
                    {"N": list(b), "yield": yields.get(b)} for b in HI_BINS],
                "width_per_bin": ("variance-ratio at alpha 0.01; "
                                  "DIAGNOSTIC: review flag unless "
                                  "deviation > 25%"),
                "joint_operator": ("per-group landing probabilities "
                                   "(G2/G3/miss) on [20.4,21.1] x live "
                                   "strata from frozen C_inj + K_natural "
                                   "minus frozen transfer offsets; chi^2 "
                                   "alpha 0.01")},
            "verdict": ("any Holm-failure among mean/pooled/joint or a "
                        "coherent completeness failure => HIGH-N "
                        "PREDICTIVE NO-GO (F-pre primary); width-only "
                        "deviation <= 25% => review flag, not no-go")},
        "low_boundary_family": {
            "role": "transport extension gate (cannot reject high-N)",
            "alpha_family": 0.01, "multiplicity": "Holm within family",
            "tests": {
                "mean_per_bin": bins_ref(LO_BINS),
                "completeness_per_bin": [
                    {"N": list(b), "yield": yields.get(b)} for b in LO_BINS],
                "subfloor_rate": {"reference": sub_lo,
                                  "test": "binomial two-sided alpha 0.01"},
                "lya_em_region": {
                    "reference": lya_ref,
                    "test": ("holdout LYA_EM-minus-interior delta vs the "
                             "frozen calibration delta; two-sided z, "
                             "alpha 0.01; expected n ~ 60-70 => powered "
                             "for a null-region alternative (z ~ 4)")},
                "below_floor_migration": (
                    "NOT holdout-testable: no injections below 19.5 "
                    "exist (Phase-C directive); adjudicated "
                    "development-side only (p1_migration.json); recorded "
                    "here so its absence is never read as a pass")},
            "verdict_mapping": ("failure => exactly one of: low-boundary "
                                "support failure (F-sup-low) / support "
                                "restriction / migration systematic / "
                                "conditional-kernel requirement / "
                                "unadjudicated transport (underpowered). "
                                "It DOES NOT reject the high-N operator "
                                "unless (frozen criterion) the implied "
                                "contamination of high-N observed bins "
                                "exceeds 50 G3-equivalent counts OR the "
                                "primary joint-operator test fails "
                                "simultaneously")},
        "exploratory": {
            "subgroups": ["z_qso terciles", "LYB_EM region", "EDGE flag",
                          "per-region high-N splits"],
            "rules": ("uncorrected, labeled exploratory; cannot reject "
                      "the high-N operator; cannot tune anything after "
                      "the read; cannot be promoted to confirmatory on "
                      "favorable results")},
        "global_verdict_enumeration": [
            "HIGH-N PREDICTIVE PASS (primary family all-pass)",
            "HIGH-N PREDICTIVE FAILURE (F-pre; taxonomy primary)",
            "LOW-BOUNDARY SUPPORT FAILURE (F-sup-low)",
            "LOW-BOUNDARY TRANSPORT QUALIFICATION (systematic/"
            "conditional-kernel/support restriction)",
            "SUBGROUP UNDERPOWER (non-adjudicable, recorded)",
            "IMPLEMENTATION-INVALID (F-imp; nothing scientific "
            "adjudicated)"],
        "gatekeeping": ("hierarchical: the primary family alone decides "
                        "the P1 predictive verdict; the low-boundary "
                        "family is evaluated regardless but maps only to "
                        "its own outcomes; exploratory results carry no "
                        "verdict weight. high-N operator validity != "
                        "low-boundary transport validity, by "
                        "construction."),
        "false_fail": ("<= ~1% per family under the null (Holm, "
                       "alpha 0.01); overall verdict false-fail bounded "
                       "by the primary family's rate"),
        "non_adjudicable_dex": 0.015,
    }
    with open(os.path.join(_HERE, "p1_holdout_battery_v2.json"), "w") as fh:
        json.dump(battery, fh, indent=1)
    print("v2 frozen. primary bins:", [b["N"] for b in
                                       battery["primary_family"]["tests"]
                                       ["mean_per_bin"]])
    print("low bins:", [b["N"] for b in
                        battery["low_boundary_family"]["tests"]
                        ["mean_per_bin"]])
    print("lya_em calib delta:", round(lya_ref["delta"], 4),
          "subfloor refs:", {k: round(v_["rate"], 4)
                             for k, v_ in sub_lo.items()})


if __name__ == "__main__":
    main()
