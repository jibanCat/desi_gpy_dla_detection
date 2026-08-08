#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Finalize (freeze) the one-time holdout test battery — NO holdout read.

Writes `p1_holdout_battery.json`: the frozen test list, calibration-side
reference values, tolerances, family-wise error control and power, per
`docs/P1_FAILURE_TAXONOMY.md` §4 and the PI's 2026-08-07 in-principle
ratification. Inputs are CALIBRATION-role pairs and DESIGN-side holdout
counts only (p1_holdout_gate.json); no holdout outcome is touched.

After this freeze: no tolerance, reference, bin edge or test definition
may be changed using holdout outcomes.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "injection"))

from t2_pairing import truth_neighbors, injected_pairs  # noqa: E402

BINS = [(19.5, 20.0), (20.0, 20.4), (20.4, 20.7), (20.7, 21.0),
        (21.0, 21.3), (21.3, 21.7), (21.7, 22.4)]
POOLED = (20.7, 21.1)
ALPHA_FAMILY = 0.01
NON_ADJUDICABLE_DEX = 0.015
# frozen D1 transfer offsets (natural - injected), t2_pairing.json
TRANSFER = {"[20.4,20.7)": 0.017702285657740312,
            "[20.7,21.0)": 0.024874019979816447,
            "[21.0,21.3)": 0.045362552080292086,
            "[21.3,21.7)": 0.037563641458363405}


def _stats(v):
    v = np.asarray(v, float)
    if len(v) < 5:
        return dict(n=int(len(v)), mean=None, se=None, sd=None, robust=None)
    q16, q84 = np.percentile(v, [15.865, 84.135])
    return dict(n=int(len(v)), mean=float(v.mean()),
                se=float(v.std(ddof=1) / np.sqrt(len(v))),
                sd=float(v.std(ddof=1)),
                robust=float(0.5 * (q84 - q16)))


def main():
    by = truth_neighbors()
    inj = injected_pairs(by)
    gate = json.load(open(os.path.join(_HERE, "p1_holdout_gate.json")))

    refs = []
    for lo, hi in BINS:
        v = [r["dx"] for r in inj if lo <= r["N"] < hi]
        refs.append({"N": [lo, hi], "calibration": _stats(v)})
    vpool = [r["dx"] for r in inj if POOLED[0] <= r["N"] < POOLED[1]]

    battery = {
        "schema": "p1_holdout_battery/v1",
        "date": time.strftime("%Y-%m-%d"),
        "frozen": True,
        "no_holdout_outcome_used": True,
        "alpha_family": ALPHA_FAMILY,
        "multiplicity": "Holm within each family",
        "non_adjudicable_dex": NON_ADJUDICABLE_DEX,
        "families": {
            "mean": {
                "tests": ("per-bin two-sided z of holdout mean dx vs the "
                          "calibration reference mean (variance = "
                          "sd_cal^2/n_holdout_pairs + se_cal^2), the 7 "
                          "bins below, PLUS the pooled critical window"),
                "bins": refs,
                "pooled_window": {"N": list(POOLED),
                                  "calibration": _stats(vpool)},
            },
            "completeness": {
                "tests": ("per-bin two-sided binomial test of holdout "
                          "op-matched pair yield vs the calibration yield "
                          "in p1_holdout_gate.json (design-side "
                          "denominators)"),
                "reference_yields": [
                    {"N": b["N"], "yield": b["calib_pair_yield"],
                     "fallback": b.get("yield_from_t1_fallback", False)}
                    for b in gate["bins"]],
            },
            "width": {
                "tests": ("per-bin two-sided variance-ratio of holdout dx "
                          "sd vs calibration sd at alpha 0.01; DIAGNOSTIC "
                          "weight -- triggers review, automatic no-go only "
                          "if the ratio deviates by more than 25%"),
                "reference": [{"N": r["N"], "sd": r["calibration"]["sd"]}
                              for r in refs],
            },
            "joint_overlap": {
                "tests": ("chi^2 over per-group landing probabilities "
                          "(G1/G2/G3/miss) for holdout injections on "
                          "[20.4,21.1] x live strata, predicted from the "
                          "frozen calibration C_inj yields and "
                          "K_natural minus the frozen transfer offsets; "
                          "alpha 0.01; joint-operator form -- NOT a "
                          "K-only bridge; no acceptance via total-count "
                          "cancellation"),
                "frozen_transfer_offsets_nat_minus_inj": TRANSFER,
            },
        },
        "power": {
            "source": "p1_holdout_gate.json (design-side)",
            "critical_window": gate["critical_window_20p7_21p1"],
            "per_bin_D_mean_31_alpha05": [
                {"N": b["N"], "power": b.get("power_D-mean-31", {}).get("0.05")
                 if isinstance(b.get("power_D-mean-31", {}), dict) else None}
                for b in gate["bins"]],
        },
        "verdict_mapping": ("all families pass -> P1 holdout PASS "
                            "(sufficiency within validated support only); "
                            "any failure -> exactly one PRIMARY taxonomy "
                            "category (P1_FAILURE_TAXONOMY.md #1); no "
                            "re-runs; no tolerance changes; no holdout "
                            "reuse for redesign"),
        "open_preconditions": ("PI full-freeze ratification; artifact + "
                               "guards built (DONE f1eff35); identity PASS "
                               "(DONE); jackknife PASS (DONE); "
                               "hidden-transition audit (spec section)"),
    }
    with open(os.path.join(_HERE, "p1_holdout_battery.json"), "w") as fh:
        json.dump(battery, fh, indent=1)
    print("battery frozen:",
          {k: len(v.get("bins", v.get("reference_yields", [])))
           if isinstance(v, dict) else v
           for k, v in battery["families"].items()})


if __name__ == "__main__":
    main()
