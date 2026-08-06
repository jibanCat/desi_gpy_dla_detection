"""REVIEW-ONLY (Phase A) — empirical calibration of T = dev(F3) - dev(F1).

Parametric bootstrap, three generating measures:
  null : y* ~ Poisson(mu0),  mu0 = the F3 (pad=0) fit to the OBSERVED seed-17
         I1 draw — the best-fitting wrong model an analyst would actually face.
  alt  : y* ~ Poisson(mu_truth), mu_truth = the I1 injection truth
         (T_A=24000 pad, T_B=1086.7 FP, window = pack truth).
  null2: y* ~ Poisson(mu from a PURE-FP truth at matched total counts,
         T_A=0, T_B=25086.7) — does the null move with the generating mu?

Per replicate records: T, dev(F1), dev(F3), T_A/T_B/T_W of both fits, seeds,
timing.  Streams JSONL so partial runs are usable; aggregates to results.json.
Seeds: y_obs=17 (probe's); null r -> 100000+r; alt r -> 200000+r;
null2 r -> 300000+r.  QC: every 25th replicate refits F1 cold and records the
dev discrepancy (warm-start adequacy).
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool

import fitter

HERE = os.path.dirname(os.path.abspath(__file__))
STREAM = os.path.join(HERE, "results_stream.jsonl")

N_NULL, N_ALT, N_NULL2 = 120, 120, 60
WORKERS = 4
WARM = dict(n_em=100, maxiter=8000)

P = None
CTX = {}


def prepare():
    """Anchor fits in the parent (workers inherit via fork)."""
    global P
    P = fitter.Problem()
    f_t, lam_t, mu_t = P.build_truth(24000.0, 1086.7)
    y_obs = np.random.default_rng(17).poisson(mu_t).astype(float)
    rF1_obs = fitter.fit(P, y_obs, pad_free=True)
    rF3_obs = fitter.fit(P, y_obs, pad_free=False)
    mu0 = np.where(P.live, np.clip(rF3_obs["mu"], 0.0, None), 0.0)
    # alt warm anchors: truth params for F1; F3-fit-to-noiseless for F3
    rF3_nl = fitter.fit(P, mu_t, pad_free=False)
    # secondary null: pure-FP truth at matched total counts
    f2, lam2, mu2 = P.build_truth(0.0, 24000.0 + 1086.7)
    CTX.update(
        mu_t=mu_t, mu0=mu0, mu2=mu2,
        u_t=f_t * P.sA, v_t=lam_t * P.scf,
        u3_obs=rF3_obs["u"], v3_obs=rF3_obs["v"],
        u3_nl=rF3_nl["u"], v3_nl=rF3_nl["v"],
        u_t2=f2 * P.sA, v_t2=lam2 * P.scf,
    )
    obs = dict(
        y_obs_seed=17, y_obs_total=float(y_obs[P.live].sum()),
        obs_F1=fit_summary(rF1_obs), obs_F3=fit_summary(rF3_obs),
        obs_T=rF3_obs["dev"] - rF1_obs["dev"],
        noiseless_F3=fit_summary(rF3_nl),
        truth=dict(T_A=24000.0, T_B=1086.7,
                   T_W=float((f_t[P.npad:] * P.sA[P.npad:]).sum())),
        null2_truth=dict(T_A=0.0, T_B=25086.7, mu2_total=float(mu2.sum())),
    )
    return obs


def fit_summary(r):
    return dict(dev=r["dev"], T_A=r["T_A"], T_B=r["T_B"], T_W=r["T_W"],
                nit=r["nit"], success=r["success"])


def one(job):
    hyp, r = job
    t0 = time.time()
    if hyp == "null":
        mu, seed = CTX["mu0"], 100000 + r
        w1 = dict(u0=CTX["u3_obs"], v0=CTX["v3_obs"])
        w3 = w1
    elif hyp == "alt":
        mu, seed = CTX["mu_t"], 200000 + r
        w1 = dict(u0=CTX["u_t"], v0=CTX["v_t"])
        w3 = dict(u0=CTX["u3_nl"], v0=CTX["v3_nl"])
    else:
        mu, seed = CTX["mu2"], 300000 + r
        w1 = dict(u0=CTX["u_t2"], v0=CTX["v_t2"])
        w3 = w1
    y = np.random.default_rng(seed).poisson(mu).astype(float)
    a = fitter.fit(P, y, pad_free=True, **WARM, **w1)
    b = fitter.fit(P, y, pad_free=False, **WARM, **w3)
    rec = dict(hyp=hyp, r=r, seed=seed, T=b["dev"] - a["dev"],
               dev_F1=a["dev"], dev_F3=b["dev"],
               TA_F1=a["T_A"], TB_F1=a["T_B"], TW_F1=a["T_W"],
               TB_F3=b["T_B"], TW_F3=b["T_W"],
               nit_F1=a["nit"], nit_F3=b["nit"], seconds=time.time() - t0)
    if r % 25 == 0:                                  # warm-start QC
        c = fitter.fit(P, y, pad_free=True)          # cold, full budget
        rec["qc_dev_F1_cold"] = c["dev"]
        rec["qc_warm_minus_cold"] = a["dev"] - c["dev"]
    return rec


def main():
    t0 = time.time()
    obs = prepare()
    print("anchors ready %.0fs; obs_T=%.4f" % (time.time() - t0, obs["obs_T"]),
          flush=True)
    jobs = []
    for r in range(max(N_NULL, N_ALT)):              # interleave for balance
        if r < N_NULL:
            jobs.append(("null", r))
        if r < N_ALT:
            jobs.append(("alt", r))
    jobs += [("null2", r) for r in range(N_NULL2)]
    recs = []
    with open(STREAM, "w") as fh, Pool(WORKERS) as pool:
        for rec in pool.imap_unordered(one, jobs, chunksize=1):
            recs.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if len(recs) % 20 == 0:
                print("done %d/%d  %.0fs" % (len(recs), len(jobs),
                                             time.time() - t0), flush=True)
    out = dict(config=dict(N_NULL=N_NULL, N_ALT=N_ALT, N_NULL2=N_NULL2,
                           WORKERS=WORKERS, warm=WARM,
                           pack=fitter.PACK, mock="2lpt0",
                           n_live=P.n_live, seconds=time.time() - t0),
               observed=obs, replicates=recs)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("ALL DONE %.0fs  n=%d" % (time.time() - t0, len(recs)), flush=True)


if __name__ == "__main__":
    main()
