#!/usr/bin/env python
"""cc_pool_posterior.py — the PREDECLARED pooled real-data posterior
(PI ruling 2026-08-21 #19; memo A2 / R-025).

Pure, jax-free selection + pooling (unit-tested), and a CLI that builds the
serialized artifact from cc_real_posterior run records:

  selection rule (predeclared in notes 2026-08-21_CP3_PREDECLARATION.md):
    a run is INCLUDED iff split-Rhat <= rhat_max on BOTH all-z estimands,
    divergences <= div_max, and the G_A real-mode guard PASSED. A base run
    that fails is flagged for ONE deep-adaptation rerun (same seed, deeper
    warmup); the deep rerun REPLACES its base (never both). A seed whose deep
    rerun still fails is EXCLUDED and DISCLOSED (per-chain medians, Rhat, PE
    carried in the artifact) — never silently dropped.
  pooling: equal-weight concatenation of the included runs' latent draws in
    seed order; reductions through the committed reduce_f_posterior and the
    real runner's own quantile/bin conventions; per-z Paper-1 bins with the
    validator's overlap weights (no truth on real data).

  python -m CDDF_analysis.hbi_mcmc.cc_pool_posterior --runs REAL_*.json
      [--deep REAL_deep_*.json] --rhat-max 1.10 --div-max 10 --out POOLED.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re

import numpy as np

ESTIMANDS = ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz")


def q5(dr):
    return [float(x) for x in np.percentile(np.asarray(dr), [2.5, 16, 50, 84, 97.5])]


def _fails(rec, rule):
    d = rec["diagnostics"]
    for k in ESTIMANDS:
        r = (d.get("split_rhat") or {}).get(k)
        if r is not None and r > rule["rhat_max"]:
            return f"split_rhat {k} {r} > {rule['rhat_max']}"
    dv = d.get("divergences")
    if dv is not None and dv > rule["div_max"]:
        return f"divergences {dv} > {rule['div_max']}"
    ga = ((rec.get("guards") or {}).get("G_A_real_mode") or {}).get("status")
    if ga != "PASS":
        return f"G_A_real_mode {ga}"
    return None


def select_runs(records, rule):
    """records: dicts with seed, deep(bool), file, diagnostics, guards."""
    by_seed = {}
    for r in records:
        by_seed.setdefault(r["seed"], {})["deep" if r.get("deep") else "base"] = r
    included, excluded, needs_deep = [], [], []
    for seed in sorted(by_seed):
        base, deep = by_seed[seed].get("base"), by_seed[seed].get("deep")
        if deep is not None:
            why = _fails(deep, rule)
            if base is not None:
                excluded.append(dict(seed=seed, deep=False, file=base["file"],
                                     reason="replaced by its deep-adaptation rerun",
                                     disclosed=False))
            if why is None:
                included.append(dict(seed=seed, deep=True, file=deep["file"]))
            else:
                excluded.append(dict(seed=seed, deep=True, file=deep["file"],
                                     reason=why, disclosed=True,
                                     perchain=deep["diagnostics"].get("perchain_estimand_medians"),
                                     split_rhat=deep["diagnostics"].get("split_rhat"),
                                     pe=deep["diagnostics"].get("mean_potential_energy_per_chain")))
            continue
        why = _fails(base, rule)
        if why is None:
            included.append(dict(seed=seed, deep=False, file=base["file"]))
        else:
            excluded.append(dict(seed=seed, deep=False, file=base["file"], reason=why,
                                 disclosed=True))
            needs_deep.append(seed)
    return dict(included=included, excluded=excluded, needs_deep_rerun=needs_deep,
                rule=dict(rule))


def pool_draws(seed_arrays):
    """Equal-weight concatenation in seed order. seed_arrays: [(seed, f)]."""
    parts, index, start = [], [], 0
    for seed, f in sorted(seed_arrays, key=lambda t: t[0]):
        f = np.asarray(f)
        parts.append(f)
        index.append(dict(seed=int(seed), n_draws=int(f.shape[0]), start=int(start)))
        start += int(f.shape[0])
    return np.concatenate(parts, axis=0), index


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_of(path):
    m = re.search(r"_s(\d{8})\.json$", os.path.basename(path))
    return int(m.group(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="base-run JSONs")
    ap.add_argument("--deep", nargs="*", default=[], help="deep-rerun JSONs")
    ap.add_argument("--rhat-max", type=float, default=1.10)
    ap.add_argument("--div-max", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rule = dict(rhat_max=a.rhat_max, div_max=a.div_max)
    recs = []
    for p in a.runs:
        d = json.load(open(p)); recs.append(dict(seed=_seed_of(p), deep=False, file=p, **{k: d[k] for k in ("diagnostics", "guards")}))
    for p in a.deep:
        d = json.load(open(p)); recs.append(dict(seed=_seed_of(p), deep=True, file=p, **{k: d[k] for k in ("diagnostics", "guards")}))
    sel = select_runs(recs, rule)
    packs = {json.load(open(r["file"]))["pack"] for r in recs}
    if len(packs) != 1:
        raise SystemExit(f"runs span {len(packs)} packs — refusing to pool")
    pack_path = packs.pop()
    if sel["needs_deep_rerun"]:
        print("DEEP RERUN NEEDED for seeds:", sel["needs_deep_rerun"])
    if not sel["included"]:
        raise SystemExit("no run satisfies the predeclared rule — nothing to pool")
    # --- heavy imports only here (jax via the package __init__) ---
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import (
        PAPER1_LOWZ_BINS, _overlap_w)
    pk = load_pack(pack_path)
    arrays = []
    for r in sel["included"]:
        z = np.load(r["file"][:-5] + "_fdraws.npz", allow_pickle=False)
        arrays.append((r["seed"], np.asarray(z["f"])))
    f, index = pool_draws(arrays)
    red = reduce_f_posterior(f, pk)
    thresholds = {k: dict(post_p2p5_16_50_84_97p5=q5(red[k])) for k in ESTIMANDS}
    ntrue = np.asarray(pk.ntrue_edges, float); dN = np.diff(ntrue)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    REDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)
    binrep = []
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        dr = ((f[:, m, :] * dN[None, m, None]).sum(axis=1) * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        binrep.append(dict(bin=[round(e0, 1), round(e1, 1)], f_post=q5(dr)))
    # per-z Paper-1 bins (validator convention; no truth on real data)
    zf = np.asarray(pk.zf_edges, float)
    reported = 0.5 * (ntrue[:-1] + ntrue[1:]) >= float(np.asarray(pk.nhat_edges, float)[0]) - 1e-9
    perz = {}
    for thr in (20.0, 20.3):
        u = np.where(reported, np.clip(ntrue[1:] - np.maximum(ntrue[:-1], thr), 0.0, None), 0.0)
        per_k = np.einsum("dbk,b->dk", f, u)
        bins = []
        for name, lo, hi in PAPER1_LOWZ_BINS:
            w = _overlap_w(zf, dX_k, lo, hi)
            if w.sum() <= 0:
                bins.append(dict(bin=name, z=[lo, hi], available=False)); continue
            pd = (per_k * w[None, :]).sum(axis=1) / w.sum()
            bins.append(dict(bin=name, z=[float(lo), float(hi)], available=True, dX=float(w.sum()),
                             coverage=float(np.clip(min(hi, zf[-1]) - max(lo, zf[0]), 0, None) / (hi - lo)),
                             post_p2p5_16_50_84_97p5=q5(pd)))
        perz[f"ge{thr:.1f}"] = dict(paper1_bins=bins)
    out = dict(role=("POOLED real-data HBI posterior — predeclared rule (PI ruling 2026-08-21 #19); "
                     "equal-weight concatenation of the included runs' draws; STATISTICAL interval "
                     "only; supersedes the ckpt-10.10 candidate, never combined with it"),
               pack=pack_path, pack_sha256=_sha(pack_path), n_draws=int(f.shape[0]),
               selection=sel, draw_index=index,
               estimand="POSTERIOR_MEDIAN_CI (committed reduce_f_posterior)",
               thresholds=thresholds, reporting_bins=binrep, perz_paper1=perz,
               inputs={r["file"]: _sha(r["file"][:-5] + "_fdraws.npz") for r in sel["included"]})
    json.dump(out, open(a.out, "w"), indent=1)
    np.savez(a.out[:-5] + "_fdraws.npz", f=f, ntrue_edges=ntrue, zf_edges=zf,
             draw_index_seeds=np.array([i["seed"] for i in index]),
             draw_index_starts=np.array([i["start"] for i in index]))
    print(json.dumps(dict(included=[(r["seed"], r["deep"]) for r in sel["included"]],
                          excluded=[(r["seed"], r["deep"], r["reason"]) for r in sel["excluded"]],
                          n_draws=out["n_draws"]), indent=1))


if __name__ == "__main__":
    main()
