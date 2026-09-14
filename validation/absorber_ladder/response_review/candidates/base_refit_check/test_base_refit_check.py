#!/usr/bin/env python
"""Tests for the E base-refit section 6A recheck.  DIAGNOSTIC ONLY.

T1  wbase reduces EXACTLY to the committed estimator at w == 1
T2  the base-refit path really does move the base under a tilted weight
T3  RC.row_kl reproduces analytic KL on known rows
T4  the harness reproduces the SEALED fixed-base section 6A numbers to 1e-6
    (reads the results JSON; skipped with a loud message if not yet produced)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAND = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, _CAND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_candidates as RC                                    # noqa: E402
import run_base_refit_check as BRC                             # noqa: E402
import wbase as WB                                             # noqa: E402

SEALED = os.path.join(BRC.PRODUCTS, "antitautology_AB_extra.json")
RESULTS = os.path.join(_HERE, "E_base_refit_reweighting_check.json")


def t1_unweighted_reduction(geom, ev, N_ref, spec):
    full = np.ones(ev["n"], bool)
    ref, _ = RC.parametric_rows(ev, full, geom, spec, N_ref)
    got, _ = WB.base_rows_weighted(ev, full, geom, spec, N_ref, w=None)
    d = float(np.max(np.abs(got - ref)))
    ones, _ = WB.base_rows_weighted(ev, full, geom, spec, N_ref,
                                    w=np.ones(ev["n"]))
    d1 = float(np.max(np.abs(ones - ref)))
    ok = (d == 0.0) and (d1 == 0.0)
    print(f"T1 unweighted reduction: max|d| w=None {d:.3e}, w=1 {d1:.3e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok, ref


def t2_base_moves(geom, ev, N_ref, spec, ref):
    w = RC.slope_weights(ev, geom, "steep")
    got, _ = WB.base_rows_weighted(ev, full_mask(ev), geom, spec, N_ref, w=w)
    kl = RC.row_kl(ref, got)
    tid = np.asarray(ev["tid"], np.int64)
    b1 = RC.fit_candidate("R1c", ev, (tid % 4) < 2, geom, N_ref)["rows"]
    b2 = RC.fit_candidate("R1c", ev, (tid % 4) >= 2, geom, N_ref)["rows"]
    noise = 0.5 * (RC.row_kl(b1, b2) + RC.row_kl(b2, b1))
    import candlib as CL
    cnt = CL.held_out_counts(ev, full_mask(ev), geom).sum(axis=-1)
    big = cnt >= 200
    med, nmed = float(np.median(kl[big])), float(np.median(noise[big]))
    ok = (float(np.max(np.abs(got - ref))) > 1e-6) and (med > nmed)
    print(f"T2 tilted weight moves the base: max|d|={np.max(np.abs(got-ref)):.3e} "
          f"medKL={med:.5g} vs split-half noise {nmed:.5g} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def full_mask(ev):
    return np.ones(ev["n"], bool)


def t3_row_kl():
    p = np.array([[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]])
    q = np.array([[0.25, 0.75], [0.25, 0.75], [0.5, 0.5]])
    exp = np.array([0.5 * np.log(2.0) + 0.5 * np.log(0.5 / 0.75),
                    0.0,
                    np.log(2.0)])
    got = RC.row_kl(p, q)
    # unnormalised input must be normalised internally
    got2 = RC.row_kl(p * 7.0, q * 3.0)
    ok = (np.max(np.abs(got - exp)) < 1e-12 and
          np.max(np.abs(got2 - exp)) < 1e-12)
    print(f"T3 row_kl on known rows: max|d|={np.max(np.abs(got-exp)):.3e} "
          f"(unnormalised {np.max(np.abs(got2-exp)):.3e}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def t4_reproduction():
    if not os.path.exists(RESULTS):
        print("T4 reproduction: SKIPPED (results JSON not produced yet)")
        return None
    sealed = {k: v["A"] for k, v in
              json.load(open(SEALED))["results"].items()}
    res = json.load(open(RESULTS))
    worst, worstk = 0.0, ""
    keys = ("kl_median", "kl_p95", "ratio_median", "frac_rows_over_3x")
    for kind in BRC.KINDS:
        for k in keys:
            a = float(sealed["E"][kind][k]); b = float(res["E_fixed_base"][kind][k])
            d = abs(a - b) / max(abs(a), 1e-300)
            if d > worst:
                worst, worstk = d, f"E/{kind}/{k}"
        if sealed["E"][kind]["n_rows_over_3x"] != \
                res["E_fixed_base"][kind]["n_rows_over_3x"]:
            worst, worstk = 1.0, f"E/{kind}/n_rows_over_3x"
    dsamp = abs(float(sealed["E"].get("sampling_kl_median")) -
                float(res["sampling_kl_median_E"])) / \
        max(abs(float(sealed["E"]["sampling_kl_median"])), 1e-300)
    worst = max(worst, dsamp)
    ok = worst < 1e-6
    print(f"T4 reproduction of the sealed fixed-base numbers: worst rel diff "
          f"{worst:.3e} ({worstk or 'sampling_kl'}) -> {'PASS' if ok else 'FAIL'}")
    if "B_equalised_control" in res:
        s = sealed["B"]["equalised"]; g = res["B_equalised_control"]
        db = max(abs(float(s[k]) - float(g[k])) / max(abs(float(s[k])), 1e-300)
                 for k in keys)
        print(f"T4b B equalised control: worst rel diff {db:.3e} "
              f"-> {'PASS' if db < 1e-6 else 'FAIL'}")
        ok = ok and db < 1e-6
    return ok


def main():
    geom, ev, N_ref = BRC.setup()
    spec = BRC.r1c_spec()
    res = []
    ok1, ref = t1_unweighted_reduction(geom, ev, N_ref, spec)
    res.append(ok1)
    res.append(t2_base_moves(geom, ev, N_ref, spec, ref))
    res.append(t3_row_kl())
    r4 = t4_reproduction()
    if r4 is not None:
        res.append(r4)
    print("ALL PASS" if all(res) else "SOME FAILED")
    return 0 if all(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
