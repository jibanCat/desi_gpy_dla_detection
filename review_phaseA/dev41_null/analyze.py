"""REVIEW-ONLY (Phase A) — aggregate the calibration stream into the verdict
numbers: null/alt/null2 quantiles of T, power, p-values of the observed draw,
GOF-frame distributions, and the FP-error frame."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def q(a, p):
    return float(np.quantile(np.asarray(a, float), p))


def summ(a):
    a = np.asarray(a, float)
    return dict(n=int(a.size), mean=float(a.mean()), sd=float(a.std(ddof=1)),
                min=float(a.min()), q05=q(a, .05), q25=q(a, .25),
                q50=q(a, .50), q75=q(a, .75), q90=q(a, .90), q95=q(a, .95),
                q99=q(a, .99), max=float(a.max()))


def main():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "results_stream.jsonl"))]
    res = {}
    if os.path.exists(os.path.join(HERE, "results.json")):
        res = json.load(open(os.path.join(HERE, "results.json")))
    obs = res.get("observed", {})
    if not obs and os.path.exists(os.path.join(HERE, "out_pilot.json")):
        p = json.load(open(os.path.join(HERE, "out_pilot.json")))
        obs = dict(obs_T=p["obs_delta"], obs_F1=p["obs_F1"], obs_F3=p["obs_F3"],
                   noiseless_delta=p["noiseless_delta"], source="out_pilot.json")
    by = {}
    for r in recs:
        by.setdefault(r["hyp"], []).append(r)
    out = {"n_by_hyp": {h: len(v) for h, v in by.items()}}
    for h, v in by.items():
        out["T_" + h] = summ([r["T"] for r in v])
        out["devF3_" + h] = summ([r["dev_F3"] for r in v])
        out["devF1_" + h] = summ([r["dev_F1"] for r in v])
        out["TB_F3_" + h] = summ([r["TB_F3"] for r in v])
        out["TB_F1_" + h] = summ([r["TB_F1"] for r in v])
        out["TA_F1_" + h] = summ([r["TA_F1"] for r in v])
    qc = [(r["hyp"], r["r"], r.get("qc_warm_minus_cold")) for r in recs
          if "qc_warm_minus_cold" in r]
    out["qc_warm_minus_cold"] = dict(
        n=len(qc), max_abs=float(max(abs(x[2]) for x in qc)) if qc else None)
    out["qc_nit_cap_frac_F1"] = float(np.mean([r["nit_F1"] >= 8000 for r in recs]))
    out["qc_negative_T"] = int(sum(r["T"] < 0 for r in recs))

    Tn = np.array([r["T"] for r in by.get("null", [])])
    Ta = np.array([r["T"] for r in by.get("alt", [])])
    if Tn.size and Ta.size:
        for crit_p, tag in [(.95, "q95"), (.99, "q99")]:
            crit = q(Tn, crit_p)
            out["power_at_null_" + tag] = dict(
                crit=crit, power=float((Ta > crit).mean()),
                n_exceed=int((Ta > crit).sum()), n=int(Ta.size))
        for name, tval in [("probe_85p40", 85.40),
                           ("mine_%.2f" % obs.get("obs_T", 85.60),
                            obs.get("obs_T", 85.60)),
                           ("noiseless_41p20", 41.20)]:
            pv = (1 + int((Tn >= tval).sum())) / (Tn.size + 1)
            out["pvalue_null_" + name] = dict(T=tval, p=float(pv),
                                              n_null=int(Tn.size))
        # GOF frame: could absolute dev(F3) flag the wrong model?
        d3n = np.array([r["dev_F3"] for r in by["null"]])
        d3a = np.array([r["dev_F3"] for r in by["alt"]])
        crit = q(d3n, .95)
        out["gof_power_devF3_at_null_q95"] = dict(
            crit=crit, power=float((d3a > crit).mean()))
        d1n = np.array([r["dev_F1"] for r in by["null"]])
        d1a = np.array([r["dev_F1"] for r in by["alt"]])
        out["gof_shift"] = dict(
            mean_devF3_null=float(d3n.mean()), mean_devF3_alt=float(d3a.mean()),
            mean_devF1_null=float(d1n.mean()), mean_devF1_alt=float(d1a.mean()),
            sqrt_2n=float(np.sqrt(2 * 2610)))
    if "null2" in by and Tn.size:
        T2 = np.array([r["T"] for r in by["null2"]])
        out["null_shift_null2"] = dict(
            mean_null=float(Tn.mean()), mean_null2=float(T2.mean()),
            q95_null=q(Tn, .95), q95_null2=q(T2, .95))
    out["observed"] = obs
    out["claim_under_review"] = (
        "PI_CHECKPOINT_2026-08-05 s8: 'Delta-dev = 41 = 0.6 sigma against survey "
        "Poisson noise, while manufacturing a 9.6x FP error' (0.6 = 41.24/sqrt(2*2610))")
    out["interpretation_notes"] = [
        "noiseless 41.20 is the NONCENTRALITY of the model comparison at survey "
        "exposure (2x min Poisson-KL divergence truth->pad-free family), not a "
        "draw of a test statistic; it scales linearly with exposure",
        "sqrt(2*n_live)=72.25 is the sd of the ABSOLUTE GOF deviance of a FIXED "
        "model; comparing a between-model deviance difference to it is a "
        "category error - the LRT statistic T has its own (much tighter) null",
        "Wilks chi2(75) also does not apply (released params on the "
        "non-negativity boundary, strongly correlated): chi2(75) q95=96.2 would "
        "call even T=85.4 unremarkable (p=0.19); the empirical chi-bar null is "
        "far tighter - miscalibration cuts BOTH ways",
    ]
    json.dump(out, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
