#!/usr/bin/env python
"""cc_zdomain_estimand.py — z-domain restricted estimands from EXISTING
posterior draws (PI ruling 2026-08-21 #27, A5 (a+)): dN/dX(>= thr) restricted
to z >= z_lo (path-weighted over the kept native cells), the path share of
the kept domain, and the leverage of a named configuration (mirror chain)
on each restricted estimand. A reduction only — no inference rerun, no model
change. jax-free.

  python CDDF_analysis/hbi_mcmc/cc_zdomain_estimand.py --pooled POOLED.json \
      --pack PACK.npz [--config-run REAL_deep_s26.json --chain 0] --out OUT.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np


def _weights(ntrue, floor, thr):
    reported = 0.5 * (ntrue[:-1] + ntrue[1:]) >= floor - 1e-9
    return np.where(reported, np.clip(ntrue[1:] - np.maximum(ntrue[:-1], thr), 0.0, None), 0.0)


def _restricted(per_k, zf, dX_k, z_lo):
    keep = zf[:-1] >= z_lo - 1e-9
    w = np.where(keep, dX_k, 0.0)
    return (per_k * w[None, :]).sum(axis=1) / w.sum(), float(w.sum() / dX_k.sum())


def zdomain_estimands(f, ntrue, zf, dX_k, floor, z_los=(2.0, 2.3, 2.56), thresholds=(20.0, 20.3)):
    ntrue = np.asarray(ntrue, float); zf = np.asarray(zf, float); dX_k = np.asarray(dX_k, float)
    out = {}
    for thr in thresholds:
        per_k = np.einsum("dbk,b->dk", np.asarray(f), _weights(ntrue, floor, thr))
        block = {}
        for z_lo in z_los:
            dr, share = _restricted(per_k, zf, dX_k, z_lo)
            q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
            block[repr(float(z_lo))] = dict(z_lo=float(z_lo), path_share=share, median=float(q[2]),
                                      post_p2p5_16_50_84_97p5=[float(x) for x in q])
        full = block[repr(float(z_los[0]))]["median"]
        for k, v in block.items():
            v["ratio_to_full_domain"] = float(v["median"] / full)
        out[f"ge{thr:.1f}"] = block
    return out


def config_leverage_by_domain(f_conf, f_pool, ntrue, zf, dX_k, floor, z_los=(2.0, 2.3, 2.56), thresholds=(20.0, 20.3)):
    ntrue = np.asarray(ntrue, float); zf = np.asarray(zf, float); dX_k = np.asarray(dX_k, float)
    out = {}
    for thr in thresholds:
        u = _weights(ntrue, floor, thr)
        pc = np.einsum("dbk,b->dk", np.asarray(f_conf), u); pp = np.einsum("dbk,b->dk", np.asarray(f_pool), u)
        out[f"ge{thr:.1f}"] = {}
        for z_lo in z_los:
            mc = np.median(_restricted(pc, zf, dX_k, z_lo)[0]); mp = np.median(_restricted(pp, zf, dX_k, z_lo)[0])
            out[f"ge{thr:.1f}"][repr(float(z_lo))] = float(100.0 * (mc / mp - 1.0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", required=True); ap.add_argument("--pack", required=True)
    ap.add_argument("--config-run", default=None); ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--n-chains", type=int, default=2)
    ap.add_argument("--z-los", nargs="+", type=float, default=[2.0, 2.3, 2.56])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pk = np.load(a.pack, allow_pickle=False)
    f = np.load(a.pooled[:-5] + "_fdraws.npz", allow_pickle=False)["f"]
    args = (pk["ntrue_edges"], pk["zf_edges"], pk["dX"].sum(axis=1), float(pk["nhat_edges"][0]))
    res = dict(role="z-domain restricted estimands from existing pooled draws (PI 2026-08-21 #27); reduction only",
               pooled=a.pooled, pack=a.pack, z_los=a.z_los,
               estimands=zdomain_estimands(f, *args, z_los=tuple(a.z_los)))
    if a.config_run:
        fr = np.load(a.config_run[:-5] + "_fdraws.npz", allow_pickle=False)["f"]
        n = fr.shape[0] // a.n_chains
        fc = fr[a.chain * n:(a.chain + 1) * n]
        res["config_leverage_pct"] = dict(config_run=a.config_run, chain=a.chain,
                                          by_domain=config_leverage_by_domain(fc, f, *args, z_los=tuple(a.z_los)))
    json.dump(res, open(a.out, "w"), indent=1)
    for tag, block in res["estimands"].items():
        print(tag, " | ".join(f"z>={k}: {v['median']:.5f} (share {v['path_share']:.3f}, ratio {v['ratio_to_full_domain']:.4f})" for k, v in block.items()))
    if a.config_run:
        for tag, d in res["config_leverage_pct"]["by_domain"].items():
            print(tag, "mirror leverage by domain:", {k: round(v, 2) for k, v in d.items()})


if __name__ == "__main__":
    main()
