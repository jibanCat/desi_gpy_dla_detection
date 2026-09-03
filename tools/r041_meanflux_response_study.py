#!/usr/bin/env python
"""Mean-flux / forest-substrate dependence of the N_HI migration operator (PI amendment 2026-09-02; frozen spec
MAX4_MEANFLUX_RESPONSE_STUDY_SPEC_2026-09-02.md). Mock/injection tables only; Candidate E throughout; no new finder run.

Arms: MOCK ladder (identical injections, forest rescale target varies): fid (2LPT loa-0 random), turner_m1s, turner_p1s, ding; REAL ladder (real-spectrum
injections; the injections shared by the arms): IR fiducial, becker2013, fg2008; cross-substrate: I2 vs IR at matched N / S/N cell / emulated z; native N2.
Outputs: meanflux_response.json (+ printed summary).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, REPO)
from r041_response_population_study import ROOT, load_events, FIT_SNR_EDGES  # noqa: E402
from r041_response_estimator import build_E, population, forward, coarsen, crossing_aggregate_A, crossing_aggregate_events, cells_of, TB, OB  # noqa: E402

TABLES = {
    "fid": f"{ROOT}/p1/reductions/analysis_mock_2lpt_random_MAX4_per_injection.csv",
    "turner_m1s": f"{ROOT}/p1/reductions/analysis_mfctrl_turner2024_m1s_per_injection.csv",
    "turner_p1s": f"{ROOT}/p1/reductions/analysis_mfctrl_turner2024_p1s_per_injection.csv",
    "ding": f"{ROOT}/p1/reductions/analysis_mfctrl_ding2024_hz_per_injection.csv",
    "IR": f"{ROOT}/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv",
    "becker2013": f"{ROOT}/p1/reductions/analysis_mf_becker2013_MAX4_per_injection.csv",
    "fg2008": f"{ROOT}/p1/reductions/analysis_mf_fg2008_MAX4_per_injection.csv",
}
LNTAU = {"fid": 0.0, "turner_m1s": -0.056, "turner_p1s": +0.053, "ding": +0.111, "IR": 0.0, "becker2013": +0.054, "fg2008": +0.178}   # spec §2 (mean over the injections)
BINS = [(20.0, 20.3), (20.3, 20.5), (20.5, 20.7), (20.7, 21.0), (21.0, 21.5)]
SEED = 20260902


def load_inj(path):
    """Events from a per-injection table via the study loader's convention (uses INJ mapping by path)."""
    import csv
    rows = list(csv.DictReader(open(path)))
    f = lambda k, typ=float: np.array([typ(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
    det = np.array([r["detected"] == "True" for r in rows])
    ev = dict(logN=f("logN"), z=f("z_inj"), snr=f("snr"), stratum=f("stratum", int), matched=det, Nhat=f("nhat"), TARGETID=f("TARGETID", int),
              key=np.array([f'{r["TARGETID"]}|{r["wave"]}|{r["inj_idx"]}' for r in rows]), class_20=np.array(["injection"] * len(rows)),
              isolated=np.ones(len(rows), bool), wide_unblended=np.zeros(len(rows), bool), sub20=np.zeros(len(rows), bool), N_nn=np.full(len(rows), np.nan), m20=np.ones(len(rows), int))
    from r041_response_population_study import zblock, SNRGROUP
    ev["zblock"] = np.array([zblock(z) for z in ev["z"]]); ev["snrgroup"] = np.array([SNRGROUP[int(s)] for s in ev["stratum"]]); ev["dN"] = ev["Nhat"] - ev["logN"]
    return ev


def subset(ev, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == ev["logN"].shape else v) for k, v in ev.items()}


def stats(ev, w=None):
    """Per bin x S/N cell (and low-S/N pooled): mean/median/sd/P>0.1/P>0.2 of dN on detected; completeness C; plus aggregated U, D."""
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    isr, _ = cells_of(ev); det = ev["matched"] & np.isfinite(ev["Nhat"]); out = {}
    cells = {"sr0": isr == 0, "sr1": isr == 1, "sr2": isr == 2, "all": np.ones(len(isr), bool), "low_strata01": ev["stratum"] <= 1}
    for cname, cm in cells.items():
        for lo, hi in BINS:
            mb = cm & (ev["logN"] >= lo) & (ev["logN"] < hi); m = mb & det
            if w[mb].sum() < 10:
                continue
            d = ev["dN"][m]; ww = w[m] / max(w[m].sum(), 1e-12); mean = float((ww * d).sum()) if len(d) else np.nan
            order = np.argsort(d); cw = np.cumsum(ww[order])
            out[f"{cname}|[{lo},{hi})"] = dict(n=int(mb.sum()), C=round(float(w[m].sum() / w[mb].sum()), 4), mean=round(mean, 4),
                                                 median=round(float(d[order][np.searchsorted(cw, 0.5)]), 4) if len(d) else None, sd=round(float(np.sqrt((ww * (d - mean) ** 2).sum())), 4),
                                                 p_gt_0p1=round(float((ww * (d > 0.1)).sum()), 4), p_gt_0p2=round(float((ww * (d > 0.2)).sum()), 4))
    U, D = crossing_aggregate_events(ev, w); out["U"] = round(U, 4); out["D"] = round(D, 4)
    # per S/N cell U/D
    for a in range(3):
        m = isr == a
        Ua, Da = crossing_aggregate_events(subset(ev, m), None if w is None else w[m]); out[f"U_sr{a}"] = round(Ua, 4); out[f"D_sr{a}"] = round(Da, 4)
    return out


def paired_diff(ev_a, ev_b, n_boot=200, seed=SEED):
    """Differences (b - a) of key metrics with a sightline bootstrap over the SHARED injections (paired)."""
    ka = {k: i for i, k in enumerate(ev_a["key"])}; idx = np.array([ka.get(k, -1) for k in ev_b["key"]]); ok = idx >= 0
    A = subset(ev_a, idx[ok]); B = subset(ev_b, ok)
    tids = np.unique(A["TARGETID"]); tid_idx = np.searchsorted(tids, A["TARGETID"])
    def metrics(w):
        sa, sb = stats(A, w), stats(B, w)
        keys = ["U", "D", "U_sr0", "D_sr0", "sr0|[20.3,20.5)", "sr0|[20.0,20.3)", "sr0|[20.5,20.7)", "all|[20.3,20.5)", "all|[20.0,20.3)", "all|[20.5,20.7)", "low_strata01|[20.3,20.5)"]
        out = {}
        for k in keys:
            if k in ("U", "D", "U_sr0", "D_sr0"):
                out[k] = sb[k] - sa[k]
            elif k in sa and k in sb:
                out[k + "|mean"] = sb[k]["mean"] - sa[k]["mean"]; out[k + "|p_gt_0p2"] = sb[k]["p_gt_0p2"] - sa[k]["p_gt_0p2"]; out[k + "|C"] = sb[k]["C"] - sa[k]["C"]
        return out
    point = metrics(None); rng = np.random.default_rng(seed); boots = {k: [] for k in point}
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(tids), len(tids)), minlength=len(tids)).astype(float); w = mult[tid_idx]
        m = metrics(w)
        for k in point:
            boots[k].append(m.get(k, np.nan))
    return {k: dict(diff=round(float(v), 4), ci95=[round(float(np.nanpercentile(boots[k], 2.5)), 4), round(float(np.nanpercentile(boots[k], 97.5)), 4)]) for k, v in point.items()}, int(ok.sum())


def propagate(ev_arm, ev_fid, ev_pop):
    """Fold the truth-known N2 population through the arm's E operator vs the fiducial's: predicted >= 20.0 / >= 20.3 counts, (a) M only (C fixed = fiducial), (b) C and M."""
    op_a = build_E(ev_arm); op_f = build_E(ev_fid); T = population(ev_pop)["T"]
    Cf = np.nan_to_num(op_f["C"]); Ca = np.nan_to_num(op_a["C"])
    ge3 = OB[:-1] >= 20.3 - 1e-9; ge0 = OB[:-1] >= 20.0 - 1e-9
    mu_f = forward(coarsen(op_f["M"] * Cf[:, :, None, :]), T); mu_M = forward(coarsen(op_a["M"] * Cf[:, :, None, :]), T); mu_CM = forward(coarsen(op_a["M"] * Ca[:, :, None, :]), T)
    UA_f, DA_f = crossing_aggregate_A(coarsen(op_f["M"]), Cf, T); UA_a, DA_a = crossing_aggregate_A(coarsen(op_a["M"]), Cf, T)
    return dict(M_only_ge20p3_pct=round(100 * (mu_M[ge3].sum() / mu_f[ge3].sum() - 1), 2), M_only_ge20p0_pct=round(100 * (mu_M[ge0].sum() / mu_f[ge0].sum() - 1), 2),
                CM_ge20p3_pct=round(100 * (mu_CM[ge3].sum() / mu_f[ge3].sum() - 1), 2), CM_ge20p0_pct=round(100 * (mu_CM[ge0].sum() / mu_f[ge0].sum() - 1), 2),
                kernel_U=round(UA_a, 4), kernel_D=round(DA_a, 4), kernel_U_fid=round(UA_f, 4), kernel_D_fid=round(DA_f, 4))


def slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); ok = np.isfinite(ys)
    if ok.sum() < 2:
        return None
    A = np.vstack([xs[ok], np.ones(ok.sum())]).T; s, _ = np.linalg.lstsq(A, ys[ok], rcond=None)[0]; return round(float(s), 4)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{ROOT}/response_estimator/meanflux_response"); ap.add_argument("--study", default=f"{ROOT}/response_study")
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    ev = {k: load_inj(p) for k, p in TABLES.items()}
    N2 = load_events("N2", a.study)
    res = dict(lntau=LNTAU, arms={}, paired={}, propagation={}, derivatives={}, cross_substrate={}, decomposition={})
    for k in ev:
        res["arms"][k] = stats(ev[k])
    res["arms"]["N2_native"] = stats(N2)
    for lad, fidk, arms in (("mock", "fid", ["turner_m1s", "turner_p1s", "ding"]), ("real", "IR", ["becker2013", "fg2008"])):
        for k in arms:
            pd_, n = paired_diff(ev[fidk], ev[k]); res["paired"][f"{k}-vs-{fidk}"] = dict(n_paired=n, **pd_)
            print(f"{k} vs {fidk} (n {n}): dU {pd_['U']['diff']:+.3f} {pd_['U']['ci95']} dD {pd_['D']['diff']:+.3f} {pd_['D']['ci95']} d<dN>[20.3,20.5) sr0 {pd_.get('sr0|[20.3,20.5)|mean', {}).get('diff')} all {pd_.get('all|[20.3,20.5)|mean', {}).get('diff')} dC[20.3,20.5) all {pd_.get('all|[20.3,20.5)|C', {}).get('diff')}")
            res["propagation"][k] = propagate(ev[k], ev[fidk], N2)
            print(f"   propagated on N2 truth (vs {fidk}): M-only ≥20.3 {res['propagation'][k]['M_only_ge20p3_pct']:+.2f} % ≥20.0 {res['propagation'][k]['M_only_ge20p0_pct']:+.2f} % | C+M ≥20.3 {res['propagation'][k]['CM_ge20p3_pct']:+.2f} %")
        # derivatives over the ladder
        xs = [LNTAU[fidk]] + [LNTAU[k] for k in arms]
        for metric in ("U", "D", "U_sr0", "D_sr0"):
            ys = [res["arms"][fidk][metric]] + [res["arms"][k][metric] for k in arms]; res["derivatives"][f"{lad}|d{metric}/dlntau"] = slope(xs, ys)
        for cell in ("sr0", "all", "low_strata01"):
            for lo, hi in BINS[:3]:
                key = f"{cell}|[{lo},{hi})"
                ys = [res["arms"][fidk].get(key, {}).get("mean", np.nan)] + [res["arms"][k].get(key, {}).get("mean", np.nan) for k in arms]
                res["derivatives"][f"{lad}|d<dN>({key})/dlntau"] = slope(xs, ys)
                ys = [res["arms"][fidk].get(key, {}).get("C", np.nan)] + [res["arms"][k].get(key, {}).get("C", np.nan) for k in arms]
                res["derivatives"][f"{lad}|dC({key})/dlntau"] = slope(xs, ys)
    print("derivatives:", {k: v for k, v in res["derivatives"].items() if v is not None and ("U" in k or "D" in k or "sr0|[20.3,20.5)" in k or "all|[20.3,20.5)" in k)})
    # cross-substrate at matched N / S/N cell / emulated z: I2 (emulated z = z+1) vs IR; native N2 (emulated) vs I2
    I2 = dict(ev["fid"]); I2["z"] = I2["z"] + 1.0
    from r041_response_population_study import zblock
    I2["zblock"] = np.array([zblock(z) for z in I2["z"]])
    def matched(evA, evB, label):
        # design points common to both (comb) within the emulated z block 0 and 1; per S/N cell
        out = {}
        for cell in ("sr0", "sr1", "sr2", "all"):
            for lo, hi in BINS[:3]:
                kA = evA_stats = None
        sA, sB = stats(evA), stats(evB)
        for k in sA:
            if isinstance(sA[k], dict) and k in sB:
                out[k] = dict(A_mean=sA[k]["mean"], B_mean=sB[k]["mean"], diff=round(sB[k]["mean"] - sA[k]["mean"], 4), A_p02=sA[k]["p_gt_0p2"], B_p02=sB[k]["p_gt_0p2"], A_C=sA[k]["C"], B_C=sB[k]["C"], nA=sA[k]["n"], nB=sB[k]["n"])
        out["U"] = dict(A=sA["U"], B=sB["U"], diff=round(sB["U"] - sA["U"], 4)); out["D"] = dict(A=sA["D"], B=sB["D"], diff=round(sB["D"] - sA["D"], 4))
        out["U_sr0"] = dict(A=sA["U_sr0"], B=sB["U_sr0"], diff=round(sB["U_sr0"] - sA["U_sr0"], 4)); out["D_sr0"] = dict(A=sA["D_sr0"], B=sB["D_sr0"], diff=round(sB["D_sr0"] - sA["D_sr0"], 4))
        res["cross_substrate"][label] = out
        return out
    # restrict IR and I2 to the emulated z range where natives have support ([3.8,4.2) block 0) AND to the full range; report both
    m_IR0 = ev["IR"]["zblock"] == 0; m_I20 = I2["zblock"] == 0
    matched(I2, ev["IR"], "IR_minus_I2 (all z)"); matched(subset(I2, m_I20), subset(ev["IR"], m_IR0), "IR_minus_I2 (z block 0 only)")
    matched(I2, N2, "N2_minus_I2 (same emulated substrate)")
    matched(ev["IR"], N2, "N2_minus_IR (total native-vs-real-injection)")
    # decomposition (spec §4): DeltaM_tot = N2 - IR ; substrate = IR - I2 (matched mean flux) ; morphology = N2 - I2 ; opacity-reachable = slope x 0.111 (Ding) and x 0.79
    def pick(d, k):
        return d[k]["diff"] if k in d else None
    tot = res["cross_substrate"]["N2_minus_IR (total native-vs-real-injection)"]; sub = res["cross_substrate"]["IR_minus_I2 (all z)"]; mor = res["cross_substrate"]["N2_minus_I2 (same emulated substrate)"]
    dec = {}
    for name, key in (("dN_sr0_2035", "sr0|[20.3,20.5)"), ("dN_all_2035", "all|[20.3,20.5)"), ("U", "U"), ("D", "D"), ("U_sr0", "U_sr0")):
        sl = res["derivatives"].get(f"mock|d<dN>({key})/dlntau") if "dN" in name else res["derivatives"].get(f"mock|d{key}/dlntau")
        t = pick(tot, key); s = -pick(sub, key) if pick(sub, key) is not None else None; m = pick(mor, key)
        reach_ding = (sl * 0.111) if sl is not None else None; reach_full = (sl * 0.79) if sl is not None else None
        f_mf = (abs(reach_ding) / abs(t)) if (reach_ding is not None and t not in (None, 0)) else None
        dec[name] = dict(total_N2_minus_IR=t, morphology_N2_minus_I2=m, substrate_I2_minus_IR=s, slope_dlntau=sl, opacity_reachable_ding=(round(reach_ding, 4) if reach_ding is not None else None),
                         opacity_reachable_full_rescale=(round(reach_full, 4) if reach_full is not None else None), f_MF_ding=(round(f_mf, 3) if f_mf is not None else None))
    res["decomposition"] = dec
    fU = dec["U"]["f_MF_ding"]; fN = dec["dN_sr0_2035"]["f_MF_ding"]
    cls = "MF-A" if (fU is not None and fN is not None and fU >= 2 / 3 and fN >= 2 / 3) else ("MF-C" if (fU is not None and fN is not None and fU < 1 / 3 and fN < 1 / 3) else "MF-B")
    res["classification"] = dict(f_MF_U=fU, f_MF_dN_sr0_2035=fN, rule="MF-A both >= 2/3; MF-C both < 1/3; else MF-B", result=cls)
    json.dump(res, open(os.path.join(a.out, "meanflux_response.json"), "w"), indent=1, default=float)
    print("decomposition:", json.dumps(dec, indent=1)); print("classification:", res["classification"])


if __name__ == "__main__":
    main()
