#!/usr/bin/env python
"""cc_config_ambiguity.py — the configuration-ambiguity measurement (memo
A4; PI CP-4 checkpoint 2026-08-21): a named posterior configuration — one
chain of a run that sits in a distinct stationary mode (the s26 "mirror"
mode, t ~ 0) — versus the POOLED candidate, per locked Paper-1 bin and
all-z, at both thresholds. Measurement only; jax-free.

  python CDDF_analysis/hbi_mcmc/cc_config_ambiguity.py --run REAL_ln_deep_s26.json \
      --chain 0 --pooled POOLED.json --pack PACK.npz --out OUT.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

PAPER1_LOWZ_BINS = (("B1", 2.15, 2.35), ("B2", 2.35, 2.56),
                    ("B3", 2.56, 2.96), ("B4", 2.96, 3.40), ("B5", 3.40, 3.80))


def split_chains(f, n_chains):
    n = f.shape[0] // n_chains
    return tuple(f[i * n:(i + 1) * n] for i in range(n_chains))


def _overlap_w(zf, dX_k, lo, hi):
    w = np.zeros(len(dX_k))
    for k in range(len(dX_k)):
        a, b = zf[k], zf[k + 1]
        w[k] = dX_k[k] * max(0.0, min(b, hi) - max(a, lo)) / (b - a)
    return w


def config_vs_pooled(f_conf, f_pool, ntrue, zf, dX_k, floor, thresholds=(20.0, 20.3)):
    ntrue = np.asarray(ntrue, float); zf = np.asarray(zf, float); dX_k = np.asarray(dX_k, float)
    reported = 0.5 * (ntrue[:-1] + ntrue[1:]) >= floor - 1e-9
    out = {}
    for thr in thresholds:
        u = np.where(reported, np.clip(ntrue[1:] - np.maximum(ntrue[:-1], thr), 0.0, None), 0.0)
        pc = np.einsum("dbk,b->dk", f_conf, u); pp = np.einsum("dbk,b->dk", f_pool, u)

        def med(p, w):
            return float(np.median((p * w).sum(axis=1) / w.sum()))
        allz = 100.0 * (med(pc, dX_k) / med(pp, dX_k) - 1.0)
        bins = {}
        for name, lo, hi in PAPER1_LOWZ_BINS:
            w = _overlap_w(zf, dX_k, lo, hi)
            if w.sum() > 0:
                bins[name] = 100.0 * (med(pc, w) / med(pp, w) - 1.0)
        out[f"ge{thr:.1f}"] = dict(allz_pct=allz, bins_pct=bins)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--chain", type=int, required=True)
    ap.add_argument("--n-chains", type=int, default=2)
    ap.add_argument("--pooled", required=True); ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pk = np.load(a.pack, allow_pickle=False)
    f_run = np.load(a.run[:-5] + "_fdraws.npz", allow_pickle=False)["f"]
    f_conf = split_chains(f_run, a.n_chains)[a.chain]
    f_pool = np.load(a.pooled[:-5] + "_fdraws.npz", allow_pickle=False)["f"]
    d = json.load(open(a.run))
    res = config_vs_pooled(f_conf, f_pool, pk["ntrue_edges"], pk["zf_edges"],
                           pk["dX"].sum(axis=1), float(pk["nhat_edges"][0]))
    out = dict(role="configuration-ambiguity measurement (memo A4): named configuration vs pooled candidate",
               run=a.run, chain=a.chain, n_chains=a.n_chains, pooled=a.pooled, pack=a.pack,
               run_diagnostics=dict(perchain_estimand_medians=d["diagnostics"].get("perchain_estimand_medians"),
                                    split_rhat=d["diagnostics"].get("split_rhat"),
                                    t_post_in_prior_sd=d["diagnostics"].get("t_post_in_prior_sd"),
                                    mean_potential_energy_per_chain=d["diagnostics"].get("mean_potential_energy_per_chain")),
               result=res)
    json.dump(out, open(a.out, "w"), indent=1)
    for tag, v in res.items():
        print(f"{tag}: all-z {v['allz_pct']:+.1f} % | " + ", ".join(f"{b} {x:+.1f}" for b, x in v["bins_pct"].items()))


if __name__ == "__main__":
    main()
