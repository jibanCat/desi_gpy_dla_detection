#!/usr/bin/env python
"""Associated-absorption response discriminator — reduction (PI ruling 2026-09-03 §11–§17; frozen gate MAX4_ASSOCIATED_ABSORPTION_RESPONSE_GATE_2026-09-03.md).

Stages:
  analyze  run the injection analyzer for arms B and C (same call as the P0 fiducial) -> per-injection tables
  compare  Candidate-E kernel summaries per arm on the PAIRED injections (A = fid), paired differences with a sightline bootstrap, completeness
           separately, propagation through the truth-known N2 population (M only / C+M), native comparison, gate verdict
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, REPO)
from r041_response_population_study import ROOT, load_events  # noqa: E402
from r041_meanflux_response_study import load_inj, subset, stats, paired_diff, propagate, BINS  # noqa: E402
from r041_response_estimator import build_E, crossing_aggregate_events, cells_of  # noqa: E402

FID_ROOT = "/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28"
A_TABLE = f"{ROOT}/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv"


def stage_analyze(a):
    for arm in ("B", "C"):
        root = f"{ROOT}/assoc/arm{arm}"; os.makedirs(f"{root}/analysis", exist_ok=True)
        truths = [f"{root}/r041_assoc{arm}_wave{w}.h5.truth.csv" for w in (0, 1, 2)]; outs = [f"{root}/r041_assoc{arm}_wave{w}_MAX4_outputs" for w in (0, 1, 2)]
        for o in outs:
            n = len([f for f in os.listdir(o) if f.startswith("dlacat-")]) if os.path.isdir(o) else 0
            print(f"arm {arm} {os.path.basename(o)}: {n} catalogue files")
        cmd = ["python", "tools/r041_analyze.py", "--truth"] + truths + ["--outputs"] + outs + ["--population", f"{FID_ROOT}/population/r041_population.csv",
                                                                                              "--out", f"{root}/analysis/analysis_assoc{arm}_MAX4.json", "--label", f"assoc{arm}_MAX4"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True); print("\n".join(l for l in r.stdout.splitlines()[-3:]));
        if r.returncode != 0:
            print(r.stderr[-2000:]); raise SystemExit(f"analyzer failed for arm {arm}")


def stage_compare(a):
    ev = {"A": load_inj(A_TABLE), "B": load_inj(f"{ROOT}/assoc/armB/analysis/analysis_assocB_MAX4_per_injection.csv"), "C": load_inj(f"{ROOT}/assoc/armC/analysis/analysis_assocC_MAX4_per_injection.csv")}
    # attach the realization summaries (EWs) to B/C events for stratified reporting
    for arm in ("B", "C"):
        plan = {f'{r["TARGETID"]}|{r["wave"]}|{r["inj_idx"]}': r for r in csv.DictReader(open(f"{ROOT}/assoc/arm{arm}/plan_assoc{arm}.csv"))}
        ev[arm]["ew1260"] = np.array([float(plan[k]["ew_rest_SiII1260"]) if k in plan else np.nan for k in ev[arm]["key"]])
        ev[arm]["ew1206"] = np.array([float(plan[k]["ew_rest_SiIII1206"]) if k in plan else np.nan for k in ev[arm]["key"]])
    N2 = load_events("N2", f"{ROOT}/response_study"); NL = load_events("NL", f"{ROOT}/response_study")
    res = dict(arms={k: stats(v) for k, v in ev.items()}, paired={}, propagation={}, native={})
    for arm in ("B", "C"):
        pd_, n = paired_diff(ev["A"], ev[arm], n_boot=a.n_boot); res["paired"][f"{arm}-A"] = dict(n_paired=n, **pd_)
        res["propagation"][arm] = propagate(ev[arm], ev["A"], N2)
        p = res["paired"][f"{arm}-A"]
        print(f"{arm} − A (n {n}): ΔU {p['U']['diff']:+.3f} {p['U']['ci95']} ΔD {p['D']['diff']:+.3f} {p['D']['ci95']} | Δ<ΔN> all [20.3,20.5) {p.get('all|[20.3,20.5)|mean',{}).get('diff')} {p.get('all|[20.3,20.5)|mean',{}).get('ci95')} | sr0 {p.get('sr0|[20.3,20.5)|mean',{}).get('diff')} | ΔC all [20.3,20.5) {p.get('all|[20.3,20.5)|C',{}).get('diff')} | ΔP(>0.2) all [20.3,20.5) {p.get('all|[20.3,20.5)|p_gt_0p2',{}).get('diff')}")
        pr = res["propagation"][arm]; print(f"   propagated on N2 truth: M-only ≥20.3 {pr['M_only_ge20p3_pct']:+.2f} % ≥20.0 {pr['M_only_ge20p0_pct']:+.2f} % | C+M ≥20.3 {pr['CM_ge20p3_pct']:+.2f} % ≥20.0 {pr['CM_ge20p0_pct']:+.2f} % | kernel U {pr['kernel_U_fid']:.3f} -> {pr['kernel_U']:.3f} D {pr['kernel_D_fid']:.3f} -> {pr['kernel_D']:.3f}")
        # stratify B/C by realized EW(1260): tertiles
        e = ev[arm]["ew1260"]; q = np.nanpercentile(e, [33.3, 66.7]); strat = {}
        for name, m in (("ew_low", e <= q[0]), ("ew_mid", (e > q[0]) & (e <= q[1])), ("ew_high", e > q[1])):
            sb = stats(subset(ev[arm], m)); ka = {k: i for i, k in enumerate(ev["A"]["key"])}; idx = np.array([ka.get(k, -1) for k in ev[arm]["key"][m]]); sa = stats(subset(ev["A"], idx[idx >= 0]))
            strat[name] = dict(n=int(m.sum()), ew1260_range=[float(np.nanmin(e[m])), float(np.nanmax(e[m]))], dU=round(sb["U"] - sa["U"], 4), dD=round(sb["D"] - sa["D"], 4),
                               d_mean_2035_all=round(sb.get("all|[20.3,20.5)", {}).get("mean", np.nan) - sa.get("all|[20.3,20.5)", {}).get("mean", np.nan), 4))
        res["paired"][f"{arm}-A"]["by_ew1260_tertile"] = strat
        print("   by EW(1260) tertile:", {k: (v["n"], v["dU"], v["d_mean_2035_all"]) for k, v in strat.items()})
    # native comparison (ruling §17): ΔN stats, U/D for A/B/C vs native 2LPT / London (unpaired; same metrics)
    for name, evn in (("native_2LPT", N2), ("native_London", NL)):
        res["native"][name] = stats(evn)
    def row(k, s):
        m = s.get("all|[20.3,20.5)", {}); m0 = s.get("sr0|[20.3,20.5)", {})
        return f"{k:14s} U {s['U']:.3f} D {s['D']:.3f} | all[20.3,20.5) mean {m.get('mean', float('nan')):+.3f} P>0.2 {m.get('p_gt_0p2', float('nan')):.2f} C {m.get('C', float('nan')):.2f} | low-S/N mean {m0.get('mean', float('nan')):+.3f} P>0.2 {m0.get('p_gt_0p2', float('nan')):.2f}"
    for k in ("A", "B", "C"):
        print(row(k, res["arms"][k]))
    for k in ("native_2LPT", "native_London"):
        print(row(k, res["native"][k]))
    # gate verdict (frozen §3): representative arm B decides
    pB = res["paired"]["B-A"]; dN = pB.get("all|[20.3,20.5)|mean", {}).get("diff", np.nan); dU = pB["U"]["diff"]; dD = pB["D"]["diff"]; inc = res["propagation"]["B"]["M_only_ge20p3_pct"]
    n_det_2007 = int(sum(1 for i in range(len(ev["B"]["logN"])) if 20.0 <= ev["B"]["logN"][i] < 20.7 and ev["B"]["matched"][i]))
    if n_det_2007 < 300:
        verdict = "AD — INCONCLUSIVE (paired detections in [20.0,20.7) < 300)"
    elif abs(dN) <= 0.02 and abs(inc) <= 3.0 and abs(dU) <= 0.05 and abs(dD) <= 0.05:
        verdict = "AA — ASSOCIATED ABSORPTION NEGLIGIBLE"
    elif (dN >= 0.05 or dU >= 0.15) and abs(inc) >= 10.0:
        verdict = "AC — ASSOCIATED ABSORPTION NATIVE-LIKE"
    elif (0.02 < abs(dN) < 0.06) or (3.0 < abs(inc) <= 10.0) or (0.05 < abs(dU) < 0.15):
        verdict = "AB — ASSOCIATED ABSORPTION MATERIAL / INTERMEDIATE"
    else:
        verdict = "AB — ASSOCIATED ABSORPTION MATERIAL / INTERMEDIATE (boundary case; B decides)"
    res["gate"] = dict(dN_all_2035=dN, dU=dU, dD=dD, incidence_M_only_ge20p3_pct=inc, n_paired_detections_2007=n_det_2007, verdict=verdict)
    print("GATE:", res["gate"])
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "assoc_response.json"), "w"), indent=1, default=float)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["analyze", "compare"]); ap.add_argument("--out", default=f"{ROOT}/assoc/reductions"); ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv); {"analyze": stage_analyze, "compare": stage_compare}[a.stage](a)


if __name__ == "__main__":
    main()
