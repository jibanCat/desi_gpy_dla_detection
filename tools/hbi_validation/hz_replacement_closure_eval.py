#!/usr/bin/env python
"""Score HZ2 REPLACEMENT fiducial generative closure runs against the frozen replacement gate
(MAX4_HZ2_HBI_REPLACEMENT_CLOSURE_GATE_2026-09-03.md, frozen 00c9ed9; PI ruling PI_RULING_2026-09-03_REPLACEMENT_CLOSURE_GATE.md).

Per realization (run dir) and retained MCMC seed:
  §2 integrated hard gate: ≥20.3 and ≥20.0 truth inside the posterior 95 % over [3.8,5.0); G-A |ratio − 1| ≤ 0.06
  §3 bias: δ = median/truth − 1, posterior σ (half 68 %), δ/σ; PERSISTENT rule over realizations (same sign ×4, |mean| ≥ 3 %, |mean| ≥ 2 SEM)
  §4 global shape: y = 8 reporting-bin dN/dX (19.9–21.5, 0.2 dex) per draw; T_obs = (m − y_truth)ᵀ Σ⁻¹ (m − y_truth); posterior-predictive
     p_global = P(T_s ≥ T_obs) with T_s = (m − y_s)ᵀ Σ⁻¹ (m − y_s); PASS iff p_global ≥ 0.01; χ²_8 tail reported; Holm fallback if Σ ill-conditioned
  §5 local diagnostics: per-bin truth count, truth, median, 68/95 %, fractional and standardized residual; ISOLATED / COHERENT RUN / RECURRENCE flags
  §6 MCMC: divergences ≤ 10 (CP-3 deep rerun, exclusion, ≥ 1 seed retained), split-R̂ ≤ 1.10 on both estimands (diagnostics.estimand_mixing)
Usage: hz_replacement_closure_eval.py --real LABEL=SHAPE=DIR ... --out OUT.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from CDDF_analysis.hbi_mcmc.cddf_recovery_audit import bin_recovery, REDGES, _overlap_w  # noqa: E402

REPORT_LO, REPORT_HI = 19.9, 21.5
GA_TOL, DIV_MAX, RHAT_MAX, P_GLOBAL_MIN, COND_MAX = 0.06, 10, 1.10, 0.01, 1e8
RUN_MIN_BINS, RUN_Z_MIN, RUN_FRAC_MIN = 3, 1.0, 0.10
PERSIST_ABS_MIN, PERSIST_SEM_FACTOR = 0.03, 2.0


def bin_vectors(fd):
    """Per-draw reporting-bin dN/dX over [3.8,5.0) (same construction as bin_recovery) -> (draws, nbins), truth (nbins,), truth counts."""
    f, truth, ntrue, zf, dX = fd["f"], fd["truth_f"], np.asarray(fd["ntrue_edges"], float), np.asarray(fd["zf_edges"], float), np.asarray(fd["dX_k"], float)
    dN = np.diff(ntrue); w = _overlap_w(zf, dX, zf[0], zf[-1]); W = w / w.sum()
    edges = [(e0, e1) for e0, e1 in zip(REDGES[:-1], REDGES[1:]) if e0 >= REPORT_LO - 1e-9 and e1 <= REPORT_HI + 1e-9]
    Y = []; T = []; names = []
    for e0, e1 in edges:
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        Y.append(((f[:, m, :] * dN[None, m, None]).sum(axis=1) * W[None, :]).sum(axis=1))
        T.append(float(((truth[m, :] * dN[m, None]).sum(axis=0) * W).sum())); names.append([round(float(e0), 1), round(float(e1), 1)])
    return np.array(Y).T, np.array(T), names, edges


def score_seed(js_path, truth_counts_by_bin):
    j = json.load(open(js_path)); seed = int(j["run_config"]["seed"]); fd = np.load(js_path.replace(".json", "_fdraws.npz"))
    ga = float(j["diagnostics"]["predictive_total_ratio"]); div = int(j["divergences"])
    mix = j["diagnostics"].get("estimand_mixing", {}); rh = {k: (v.get("split_rhat") if isinstance(v, dict) else None) for k, v in mix.items()}
    rh_vals = [x for x in rh.values() if x is not None]; rhat_ok = bool(rh_vals) and max(rh_vals) <= RHAT_MAX
    out = dict(seed=seed, json=js_path, deep=("deep" in os.path.basename(js_path)), G_A_ratio=ga, G_A_pass=abs(ga - 1.0) <= GA_TOL, divergences=div,
               div_pass=div <= DIV_MAX, split_rhat=rh, rhat_pass=rhat_ok, code_commit=j["run_config"].get("code_commit"), pack=j.get("pack"))
    for thr in ("ge20.3", "ge20.0"):
        a = j["perz_recovery"]["estimand"][thr]["allz"]; q = a["post_p2p5_16_50_84_97p5"]; sig = 0.5 * (q[3] - q[1])
        out[thr] = dict(truth=a["truth"], median=q[2], p2p5=q[0], p16=q[1], p84=q[3], p97p5=q[4], delta=q[2] / a["truth"] - 1.0, sigma_rel=sig / q[2],
                        delta_over_sigma=(q[2] - a["truth"]) / sig, truth_in_95=bool(q[0] <= a["truth"] <= q[4]), width95_rel=(q[4] - q[0]) / q[2])
    # §4 global shape
    Y, T, names, edges = bin_vectors(fd); m = Y.mean(axis=0); S = np.cov(Y, rowvar=False); cond = float(np.linalg.cond(S))
    glob_ = dict(n_bins=len(T), cond=cond)
    if np.isfinite(cond) and cond <= COND_MAX:
        Si = np.linalg.inv(S); d = m - T; T_obs = float(d @ Si @ d); D = Y - m[None, :]; T_s = np.einsum("ij,jk,ik->i", D, Si, D)
        p_glob = float(np.mean(T_s >= T_obs)); glob_.update(method="posterior_predictive_T", T_obs=T_obs, p_global=p_glob, chi2_8_tail=float(chi2.sf(T_obs, len(T))),
                                                            T_s_p99=float(np.percentile(T_s, 99)), passed=p_glob >= P_GLOBAL_MIN)
    else:
        # Holm fallback: two-sided posterior-predictive marginal p-values
        p = np.array([2.0 * min(np.mean(Y[:, b] <= T[b]), np.mean(Y[:, b] >= T[b])) for b in range(len(T))]); order = np.argsort(p); k = len(p)
        rej = False
        for i, idx in enumerate(order):
            if p[idx] < 0.05 / (k - i):
                rej = True; break
            break
        glob_.update(method="holm_fallback", marginal_p=p.tolist(), passed=not rej)
    out["global_shape"] = glob_
    # §5 local diagnostics
    rows = bin_recovery(fd["f"], fd["truth_f"], fd["ntrue_edges"], fd["zf_edges"], fd["dX_k"], redges=REDGES, truth_counts=fd.get("truth_counts"))
    bins = []
    for r in rows:
        if r["bin"][0] < REPORT_LO - 1e-9 or r["bin"][1] > REPORT_HI + 1e-9:
            continue
        q = r["post_p2p5_16_50_84_97p5"]; sig = 0.5 * (q[3] - q[1]); z = (q[2] - r["truth"]) / sig if sig > 0 else np.nan
        bins.append(dict(bin=r["bin"], truth=r["truth"], truth_count=truth_counts_by_bin.get(tuple(r["bin"])), median=q[2], p16=q[1], p84=q[3], p2p5=q[0], p97p5=q[4],
                         frac_resid=q[2] / r["truth"] - 1.0, z=z, in68=r["truth_in_68"], in95=r["truth_in_95"], isolated_excursion=not r["truth_in_95"]))
    # coherent runs: ≥3 neighbouring bins, same sign, |z| ≥ 1 and |frac| ≥ 10 %
    runs = []; cur = []
    for b in bins:
        mat = abs(b["z"]) >= RUN_Z_MIN and abs(b["frac_resid"]) >= RUN_FRAC_MIN
        if mat and (not cur or np.sign(b["z"]) == cur[-1][1]):
            cur.append((tuple(b["bin"]), float(np.sign(b["z"]))))
        else:
            if len(cur) >= RUN_MIN_BINS:
                runs.append(dict(bins=[list(c[0]) for c in cur], sign=cur[0][1]))
            cur = [(tuple(b["bin"]), float(np.sign(b["z"])))] if mat else []
    if len(cur) >= RUN_MIN_BINS:
        runs.append(dict(bins=[list(c[0]) for c in cur], sign=cur[0][1]))
    out["reporting_bins"] = bins; out["coherent_runs"] = runs
    out["integrated_pass"] = out["G_A_pass"] and out["ge20.3"]["truth_in_95"] and out["ge20.0"]["truth_in_95"]
    out["mcmc_pass"] = out["div_pass"] and out["rhat_pass"]
    return out


def truth_counts_by_bin(pack_path):
    z = np.load(pack_path, allow_pickle=True); tc = np.asarray(z["truth_counts"], float).sum(axis=1); ne = np.asarray(z["ntrue_edges"], float); out = {}
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ne[:-1] >= e0 - 1e-9) & (ne[1:] <= e1 + 1e-9); out[(round(float(e0), 1), round(float(e1), 1))] = float(tc[m].sum())
    return out


def score_dir(label, shape, d):
    seeds = {}
    for p in sorted(glob.glob(os.path.join(d, "mockclosure_s*.json"))):
        j = json.load(open(p)); tcb = truth_counts_by_bin(j["pack"]); s = score_seed(p, tcb); seeds.setdefault(s["seed"], []).append(s)
    retained, excluded = [], []
    for seed, runs in sorted(seeds.items()):
        base = [r for r in runs if not r["deep"]]; deep = [r for r in runs if r["deep"]]
        pick = base[-1] if base else deep[-1]
        if base and not base[-1]["div_pass"] and deep:
            pick = deep[-1]
        (retained if pick["div_pass"] else excluded).append(pick)
    ok = bool(retained) and all(r["integrated_pass"] and r["global_shape"]["passed"] and r["mcmc_pass"] for r in retained)
    return dict(label=label, shape=shape, dir=d, n_seeds=len(seeds), retained=retained, excluded=excluded, realization_pass=ok)


def fmt(r):
    g = r["global_shape"]; gs = f"p_global {g.get('p_global', float('nan')):.3f} (T {g.get('T_obs', float('nan')):.1f}, chi2_8 tail {g.get('chi2_8_tail', float('nan')):.3f})" if g["method"] == "posterior_predictive_T" else f"HOLM {g['passed']}"
    bins = " ".join(f"{b['bin'][0]:.1f}:{100*b['frac_resid']:+.0f}%/z{b['z']:+.1f}{'✗' if not b['in95'] else ''}" for b in r["reporting_bins"])
    return (f"seed {r['seed']}{' (deep)' if r['deep'] else ''}: G-A {r['G_A_ratio']:.4f} div {r['divergences']} R̂ {r['split_rhat']} | ≥20.3 δ {100*r['ge20.3']['delta']:+.2f} % (σ {100*r['ge20.3']['sigma_rel']:.1f} %, δ/σ {r['ge20.3']['delta_over_sigma']:+.2f}) in95 {r['ge20.3']['truth_in_95']} | "
            f"≥20.0 δ {100*r['ge20.0']['delta']:+.2f} % in95 {r['ge20.0']['truth_in_95']} | {gs} {'✓' if g['passed'] else '✗'} | runs {[(x['bins'][0][0], x['bins'][-1][0], x['sign']) for x in r['coherent_runs']]} | bins {bins}")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--real", nargs="+", required=True, help="LABEL=SHAPE=DIR"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = dict(gate="MAX4_HZ2_HBI_REPLACEMENT_CLOSURE_GATE_2026-09-03.md", criteria=dict(GA_tol=GA_TOL, div_max=DIV_MAX, rhat_max=RHAT_MAX, p_global_min=P_GLOBAL_MIN,
               run_rule=dict(min_bins=RUN_MIN_BINS, z_min=RUN_Z_MIN, frac_min=RUN_FRAC_MIN), persist=dict(abs_min=PERSIST_ABS_MIN, sem_factor=PERSIST_SEM_FACTOR)), realizations=[])
    for spec in a.real:
        lab, shape, d = spec.split("=", 2); r = score_dir(lab, shape, d); res["realizations"].append(r)
        print(f"[{shape}] {lab}: {'PASS' if r['realization_pass'] else 'FAIL'} (retained {len(r['retained'])}, excluded {len(r['excluded'])})")
        for s in r["retained"] + r["excluded"]:
            print("   ", fmt(s), "(EXCLUDED)" if s in r["excluded"] else "")
    # §3 persistence over realizations (first retained seed per realization; all seeds listed)
    d203 = [np.mean([s["ge20.3"]["delta"] for s in r["retained"]]) for r in res["realizations"] if r["retained"]]
    d200 = [np.mean([s["ge20.0"]["delta"] for s in r["retained"]]) for r in res["realizations"] if r["retained"]]
    def classify(d):
        d = np.array(d); n = len(d)
        if n < 2:
            return dict(n=n, verdict="INSUFFICIENT")
        mean = float(d.mean()); sem = float(d.std(ddof=1) / np.sqrt(n)); same = bool(np.all(np.sign(d) == np.sign(d[0])))
        persistent = same and abs(mean) >= PERSIST_ABS_MIN and abs(mean) >= PERSIST_SEM_FACTOR * sem
        return dict(n=n, deltas=d.round(4).tolist(), mean=mean, sem=sem, same_sign=same, verdict=("PERSISTENT IN-MODEL CLOSURE BIAS" if persistent else "FINITE-SAMPLE CLOSURE SCATTER"))
    res["bias_persistence"] = dict(ge20p3=classify(d203), ge20p0=classify(d200))
    by_shape = {}
    for r in res["realizations"]:
        for s in r["retained"]:
            by_shape.setdefault(r["shape"], []).append(s["ge20.3"]["delta"])
    res["bias_by_shape_ge20p3"] = {k: dict(deltas=np.round(v, 4).tolist(), mean=float(np.mean(v))) for k, v in by_shape.items()}
    # §5 recurrence of coherent runs across realizations (overlap ≥ 2 bins, same sign, different realizations)
    allruns = [(r["label"], x) for r in res["realizations"] for s in r["retained"] for x in s["coherent_runs"]]
    recur = []
    for i in range(len(allruns)):
        for k in range(i + 1, len(allruns)):
            (la, xa), (lb, xb) = allruns[i], allruns[k]
            if la != lb and xa["sign"] == xb["sign"] and len({tuple(b) for b in xa["bins"]} & {tuple(b) for b in xb["bins"]}) >= 2:
                recur.append(dict(realizations=[la, lb], bins=sorted({tuple(b) for b in xa["bins"]} & {tuple(b) for b in xb["bins"]}), sign=xa["sign"]))
    res["recurrent_coherent_runs"] = recur
    all_pass = bool(res["realizations"]) and all(r["realization_pass"] for r in res["realizations"])
    verdict = "PASS" if (all_pass and not recur and res["bias_persistence"]["ge20p3"]["verdict"] != "PERSISTENT IN-MODEL CLOSURE BIAS") else "FAIL"
    fails = []
    for r in res["realizations"]:
        for s in r["retained"]:
            if not s["integrated_pass"]: fails.append(f"{r['label']} s{s['seed']}: integrated normalization")
            if not s["global_shape"]["passed"]: fails.append(f"{r['label']} s{s['seed']}: global CDDF-shape distortion")
            if not s["mcmc_pass"]: fails.append(f"{r['label']} s{s['seed']}: MCMC/convergence")
        if not r["retained"]: fails.append(f"{r['label']}: all seeds excluded (MCMC)")
    if recur: fails.append("recurrent coherent multi-bin run (PI review trigger)")
    if res["bias_persistence"]["ge20p3"]["verdict"] == "PERSISTENT IN-MODEL CLOSURE BIAS": fails.append("persistent truth-shape bias")
    res["verdict"] = verdict; res["failure_classes"] = fails
    print("BIAS ≥20.3 by realization:", res["bias_persistence"]["ge20p3"]); print("BIAS ≥20.3 by shape:", res["bias_by_shape_ge20p3"]); print("recurrent runs:", recur)
    print(f"REPLACEMENT HZ2 CLOSURE = {verdict}", fails)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1, default=float); print("wrote", a.out)


if __name__ == "__main__":
    main()
