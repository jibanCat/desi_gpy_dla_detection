#!/usr/bin/env python
"""Score high-z HBI mock-closure runs against the frozen HZ2 gate (MAX4_HZ2_HBI_CLOSURE_GATE_2026-09-03.md §3–§4; HZ1 §4 criteria).

Per run directory (one realization / population) and MCMC seed:
  G-A            |predictive_total_ratio − 1| ≤ 0.06
  divergences    ≤ 10 (CP-3: a failing seed is deep-rerun; still failing -> excluded and disclosed)
  ≥ 20.3 / ≥ 20.0 integrated dN/dX over [3.8,5.0): truth inside the posterior 95 % interval (perz_recovery.estimand.*.allz)
  reporting bins 19.9–21.5 (0.2 dex, ≥ 20.0 support): truth inside 95 % (cddf_recovery_audit.bin_recovery on the saved f draws)
Fiducial generative closure: PASS iff every retained seed of every realization passes all four.
Native stress (out-of-model): the same numbers are REPORTED, plus the return-to-PI triggers of gate §3:
  |integrated ≥ 20.3 bias| > 30 %; both seeds of an arm excluded for divergences; 95 % width of the integrated ≥ 20.3 incidence > 2 × fiducial.
Usage: hz_closure_eval.py --fiducial LABEL=DIR ... --stress LABEL=DIR ... --out OUT.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from CDDF_analysis.hbi_mcmc.cddf_recovery_audit import bin_recovery, REDGES  # noqa: E402

REPORT_LO, REPORT_HI = 19.9, 21.5
GA_TOL, DIV_MAX = 0.06, 10
STRESS_BIAS_TRIGGER_PCT, STRESS_WIDTH_FACTOR = 30.0, 2.0


def score_seed(js_path):
    j = json.load(open(js_path)); seed = int(j["run_config"]["seed"])
    fd = np.load(js_path.replace(".json", "_fdraws.npz"))
    ga = float(j["diagnostics"]["predictive_total_ratio"]); div = int(j["divergences"])
    out = dict(seed=seed, json=js_path, deep=("deep" in os.path.basename(js_path)), G_A_ratio=ga, G_A_pass=abs(ga - 1.0) <= GA_TOL, divergences=div, div_pass=div <= DIV_MAX,
               code_commit=j["run_config"].get("code_commit"), pack=j.get("pack"))
    for thr in ("ge20.3", "ge20.0"):
        a = j["perz_recovery"]["estimand"][thr]["allz"]; q = a["post_p2p5_16_50_84_97p5"]
        out[thr] = dict(truth=a["truth"], median=q[2], p2p5=q[0], p97p5=q[4], bias_pct=100.0 * (q[2] / a["truth"] - 1.0), truth_in_95=bool(q[0] <= a["truth"] <= q[4]),
                        width95_rel=(q[4] - q[0]) / q[2])
    rows = bin_recovery(fd["f"], fd["truth_f"], fd["ntrue_edges"], fd["zf_edges"], fd["dX_k"], redges=REDGES)
    bins = [dict(bin=r["bin"], truth=r["truth"], median=r["post_p2p5_16_50_84_97p5"][2], bias_pct=r["median_bias_pct"], truth_in_95=r["truth_in_95"], truth_in_68=r["truth_in_68"])
            for r in rows if r["bin"][0] >= REPORT_LO - 1e-9 and r["bin"][1] <= REPORT_HI + 1e-9]
    out["reporting_bins"] = bins; out["bins_pass"] = all(b["truth_in_95"] for b in bins)
    out["seed_pass"] = out["G_A_pass"] and out["div_pass"] and out["ge20.3"]["truth_in_95"] and out["ge20.0"]["truth_in_95"] and out["bins_pass"]
    return out


def score_dir(label, d):
    seeds = {}
    for p in sorted(glob.glob(os.path.join(d, "mockclosure_s*.json"))):
        if p.endswith("_fdraws.npz"):
            continue
        s = score_seed(p); seeds.setdefault(s["seed"], []).append(s)
    # CP-3: prefer the deep rerun of a seed if the base run failed on divergences; a seed is EXCLUDED only if its last available run still fails divergences
    retained, excluded = [], []
    for seed, runs in sorted(seeds.items()):
        base = [r for r in runs if not r["deep"]]; deep = [r for r in runs if r["deep"]]
        pick = base[-1] if base else deep[-1]
        if base and not base[-1]["div_pass"] and deep:
            pick = deep[-1]
        (retained if pick["div_pass"] else excluded).append(pick)
    return dict(label=label, dir=d, n_seeds=len(seeds), retained=retained, excluded=excluded,
                realization_pass=bool(retained) and all(r["seed_pass"] for r in retained))


def fmt(r):
    bins = " ".join(f"{b['bin'][0]:.1f}:{b['bias_pct']:+.0f}%{'✓' if b['truth_in_95'] else '✗'}" for b in r["reporting_bins"])
    return (f"seed {r['seed']}{' (deep)' if r['deep'] else ''}: G-A {r['G_A_ratio']:.4f}{'✓' if r['G_A_pass'] else '✗'} div {r['divergences']}{'✓' if r['div_pass'] else '✗'} | "
            f"≥20.3 truth {r['ge20.3']['truth']:.4f} med {r['ge20.3']['median']:.4f} ({r['ge20.3']['bias_pct']:+.2f} %) 95% [{r['ge20.3']['p2p5']:.4f},{r['ge20.3']['p97p5']:.4f}] "
            f"{'✓' if r['ge20.3']['truth_in_95'] else '✗'} w95/med {r['ge20.3']['width95_rel']:.3f} | ≥20.0 {r['ge20.0']['bias_pct']:+.2f} % {'✓' if r['ge20.0']['truth_in_95'] else '✗'} | bins {bins} | {'PASS' if r['seed_pass'] else 'FAIL'}")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fiducial", nargs="*", default=[]); ap.add_argument("--stress", nargs="*", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = dict(gate="MAX4_HZ2_HBI_CLOSURE_GATE_2026-09-03.md", criteria=dict(G_A_tol=GA_TOL, div_max=DIV_MAX, report_bins=[REPORT_LO, REPORT_HI], stress_bias_trigger_pct=STRESS_BIAS_TRIGGER_PCT,
                                                                         stress_width_factor=STRESS_WIDTH_FACTOR), fiducial=[], stress=[])
    for spec in a.fiducial:
        lab, d = spec.split("=", 1); r = score_dir(lab, d); res["fiducial"].append(r)
        print(f"[fiducial] {lab}: {'PASS' if r['realization_pass'] else 'FAIL'} (retained {len(r['retained'])}, excluded {len(r['excluded'])})")
        for s in r["retained"] + r["excluded"]:
            print("   ", fmt(s), "(EXCLUDED)" if s in r["excluded"] else "")
    fid_pass = bool(res["fiducial"]) and all(r["realization_pass"] for r in res["fiducial"])
    fid_w = [s["ge20.3"]["width95_rel"] for r in res["fiducial"] for s in r["retained"]]
    res["fiducial_gate"] = "PASS" if fid_pass else "FAIL"; res["fiducial_width95_rel_mean"] = float(np.mean(fid_w)) if fid_w else None
    print(f"FIDUCIAL GENERATIVE CLOSURE GATE: {res['fiducial_gate']}  (mean 95 % width / median of ≥20.3: {res['fiducial_width95_rel_mean']})")
    for spec in a.stress:
        lab, d = spec.split("=", 1); r = score_dir(lab, d)
        trig = []
        for s in r["retained"]:
            if abs(s["ge20.3"]["bias_pct"]) > STRESS_BIAS_TRIGGER_PCT:
                trig.append(f"seed {s['seed']}: |≥20.3 bias| {abs(s['ge20.3']['bias_pct']):.1f} % > {STRESS_BIAS_TRIGGER_PCT} %")
            if res["fiducial_width95_rel_mean"] and s["ge20.3"]["width95_rel"] > STRESS_WIDTH_FACTOR * res["fiducial_width95_rel_mean"]:
                trig.append(f"seed {s['seed']}: 95 % width {s['ge20.3']['width95_rel']:.3f} > {STRESS_WIDTH_FACTOR}× fiducial {res['fiducial_width95_rel_mean']:.3f}")
        if r["n_seeds"] and not r["retained"]:
            trig.append("both seeds excluded for divergences")
        r["return_to_pi_triggers"] = trig; res["stress"].append(r)
        print(f"[stress] {lab}: in-model criteria {'PASS' if r['realization_pass'] else 'FAIL'} (reported, not gated); triggers: {trig or 'none'}")
        for s in r["retained"] + r["excluded"]:
            print("   ", fmt(s), "(EXCLUDED)" if s in r["excluded"] else "")
    res["stress_triggers_any"] = any(r["return_to_pi_triggers"] for r in res["stress"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1, default=float); print("wrote", a.out)


if __name__ == "__main__":
    main()
