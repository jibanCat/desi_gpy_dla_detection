#!/usr/bin/env python
"""r041_candidate_status_gate.py — the PREDECLARED candidate-status gate of the MAX4 P0 core
(PI ruling 2026-09-01 §24 / Q7; PI ruling 2026-09-02: report before stamping if it fails).

Criterion (hard effect size, point estimates):   |dC_cand^MAX4| <= (1/3) |dC_cand^MAX1|
in the cell 20.3 <= log N_true < 20.5 (fiducial-plan points 20.3 and 20.4 = the 600 direct trials), where
    dC_cand = C(candidate-bearing sightlines) - C(candidate-free sightlines)
with the SAME candidate-status definition in both arms (`has_cand_ge20` of the frozen population table,
carried in every truth/analysis row) and the SAME injections (paired by injection_id: the A_shared fiducial
archives run under MAX1/FILTER0/100k (R-041A, diagnostic) and under MAX4/SINGLE1/FILTER1/50k (P0)).
Intervals are reported as UNCERTAINTY ONLY: per-arm Jeffreys (Beta(k+1/2, n-k+1/2)) 68/95 % intervals of each C,
bootstrap intervals of each dC and of the ratio |dC^MAX4|/|dC^MAX1| (injections resampled jointly across the two
arms, B = 4000, seed 20260904). Interval overlap is NOT an alternative pass route. Also reported: the same
statistics at the 20.3 and 20.4 points separately, per S/N stratum, and in the neighbouring cells [20.0,20.3)
and [20.5,21.0) as context (not part of the criterion).
Output: JSON (+ a markdown table) — `PASS` / `FAIL` on the criterion, never softened by the intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import beta as _beta

CELLS = {"[20.3,20.5)": (20.3, 20.5), "[20.0,20.3)": (20.0, 20.3), "[20.5,21.0)": (20.5, 21.0)}
POINTS = (20.3, 20.4)


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        k = f"{r['wave']}:{r['TARGETID']}:{r['inj_idx']}"
        out[k] = dict(logN=float(r["logN"]), stratum=int(r["stratum"]), cand=int(r["has_cand_ge20"]), y=1 if r["detected"] == "True" else 0)
    return out


def jeffreys(k, n):
    if n == 0:
        return [None, None, None, None]
    a, b = k + 0.5, n - k + 0.5
    return [float(_beta.ppf(q, a, b)) for q in (0.16, 0.84, 0.025, 0.975)]


def stats(y1, y4, cand, rng, nboot):
    """dC per arm, ratio, bootstrap intervals; y1/y4 = detections under MAX1/MAX4 for the same injections."""
    def dc(y, c):
        m1, m0 = c == 1, c == 0
        if m1.sum() == 0 or m0.sum() == 0:
            return np.nan
        return y[m1].mean() - y[m0].mean()
    n = y1.size
    d1, d4 = dc(y1, cand), dc(y4, cand)
    boots = np.empty((nboot, 3))
    idx = np.arange(n)
    for b in range(nboot):
        s = rng.choice(idx, n, replace=True)
        b1, b4 = dc(y1[s], cand[s]), dc(y4[s], cand[s])
        boots[b] = (b1, b4, abs(b4) / abs(b1) if b1 not in (0.0,) and np.isfinite(b1) and b1 != 0 else np.nan)
    def ci(x):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, q)) for q in (16, 84, 2.5, 97.5)] if x.size else [None] * 4
    k1c, n1c = int(y1[cand == 1].sum()), int((cand == 1).sum()); k1f, n1f = int(y1[cand == 0].sum()), int((cand == 0).sum())
    k4c, k4f = int(y4[cand == 1].sum()), int(y4[cand == 0].sum())
    ratio = abs(d4) / abs(d1) if np.isfinite(d1) and d1 != 0 else None
    return dict(n=int(n), n_cand=n1c, n_free=n1f,
                MAX1=dict(C_cand=k1c / n1c if n1c else None, C_free=k1f / n1f if n1f else None, jeffreys_cand=jeffreys(k1c, n1c), jeffreys_free=jeffreys(k1f, n1f), dC=float(d1) if np.isfinite(d1) else None, dC_boot_68_95=ci(boots[:, 0])),
                MAX4=dict(C_cand=k4c / n1c if n1c else None, C_free=k4f / n1f if n1f else None, jeffreys_cand=jeffreys(k4c, n1c), jeffreys_free=jeffreys(k4f, n1f), dC=float(d4) if np.isfinite(d4) else None, dC_boot_68_95=ci(boots[:, 1])),
                ratio_abs_dC4_over_dC1=ratio, ratio_boot_68_95=ci(boots[:, 2]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max1", required=True, help="R-041A per-injection CSV (MAX1 diagnostic arm)")
    ap.add_argument("--max4", required=True, help="P0 fiducial per-injection CSV (MAX4 arm)")
    ap.add_argument("--n-boot", type=int, default=4000); ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A, B = load(a.max1), load(a.max4)
    keys = sorted(set(A) & set(B)); lost = dict(missing_in_MAX1=sorted(set(B) - set(A)), missing_in_MAX4=sorted(set(A) - set(B)))
    logN = np.array([A[k]["logN"] for k in keys]); st = np.array([A[k]["stratum"] for k in keys]); cand = np.array([A[k]["cand"] for k in keys])
    assert all(A[k]["cand"] == B[k]["cand"] and A[k]["logN"] == B[k]["logN"] for k in keys), "candidate status / truth differ between arms"
    y1 = np.array([A[k]["y"] for k in keys]); y4 = np.array([B[k]["y"] for k in keys])
    rng = np.random.default_rng(a.seed)
    res = dict(criterion="|dC_cand^MAX4| <= (1/3)|dC_cand^MAX1| in 20.3 <= logN < 20.5 (point estimates); intervals = uncertainty only", n_pairs=len(keys), lost=lost, cells={}, points={}, strata_in_cell={})
    for name, (lo, hi) in CELLS.items():
        m = (logN >= lo) & (logN < hi)
        res["cells"][name] = stats(y1[m], y4[m], cand[m], rng, a.n_boot)
    for p in POINTS:
        m = logN == p
        res["points"][str(p)] = stats(y1[m], y4[m], cand[m], rng, a.n_boot)
    m = (logN >= 20.3) & (logN < 20.5)
    for s in sorted(set(st.tolist())):
        mm = m & (st == s)
        res["strata_in_cell"][str(s)] = stats(y1[mm], y4[mm], cand[mm], rng, max(1000, a.n_boot // 4))
    g = res["cells"]["[20.3,20.5)"]
    d1, d4 = g["MAX1"]["dC"], g["MAX4"]["dC"]
    passed = (d1 is not None and d4 is not None and abs(d4) <= abs(d1) / 3.0)
    res["verdict"] = dict(dC_MAX1=d1, dC_MAX4=d4, one_third_of_MAX1=abs(d1) / 3.0 if d1 is not None else None, n_direct_trials=g["n"], n_direct_trials_ge_600=bool(g["n"] >= 600),
                          PASS=bool(passed), note="hard effect-size criterion on point estimates; interval overlap is not a pass route; FAIL -> report before stamping")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    md = [f"# Candidate-status gate — {'PASS' if passed else 'FAIL'}", "", f"pairs {len(keys)} (lost {len(lost['missing_in_MAX1'])}/{len(lost['missing_in_MAX4'])}); direct trials in [20.3,20.5): {g['n']} (>= 600: {g['n'] >= 600})", "",
          "| cell / point | n (cand/free) | C_cand^MAX1 | C_free^MAX1 | dC^MAX1 [68 %] | C_cand^MAX4 | C_free^MAX4 | dC^MAX4 [68 %] | ratio |dC4|/|dC1| [68 %] |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in list(res["cells"].items()) + [(f"point {p}", v) for p, v in res["points"].items()]:
        f = lambda x: "—" if x is None else f"{x:+.3f}" if isinstance(x, float) and x < 0 or (isinstance(x, float) and name.startswith("point") is False and False) else (f"{x:.3f}" if x is not None else "—")
        ci1 = s["MAX1"]["dC_boot_68_95"]; ci4 = s["MAX4"]["dC_boot_68_95"]; cr = s["ratio_boot_68_95"]
        md.append(f"| {name} | {s['n']} ({s['n_cand']}/{s['n_free']}) | {f(s['MAX1']['C_cand'])} | {f(s['MAX1']['C_free'])} | {f(s['MAX1']['dC'])} [{f(ci1[0])}, {f(ci1[1])}] | {f(s['MAX4']['C_cand'])} | {f(s['MAX4']['C_free'])} | {f(s['MAX4']['dC'])} [{f(ci4[0])}, {f(ci4[1])}] | {f(s['ratio_abs_dC4_over_dC1'])} [{f(cr[0])}, {f(cr[1])}] |")
    open(os.path.splitext(a.out)[0] + ".md", "w").write("\n".join(md) + "\n")
    print(json.dumps(res["verdict"], indent=1)); print("\n".join(md[4:]))


if __name__ == "__main__":
    main()
