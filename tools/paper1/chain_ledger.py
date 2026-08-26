#!/usr/bin/env python3
"""chain_ledger.py — the individual chain -> convergence decision -> included pool ->
frozen posterior reconstruction, printed from the frozen artifacts themselves
(PI requirement 2026-08-26 §8: chains are first-class frozen artifacts).

    python tools/paper1/chain_ledger.py [--cp3-dir <cp3_real>] [--json out.json]

For every REAL_ln*.json in the directory: seed, deep flag, draws, split-Rhat of both
all-z estimands, divergences, G_A real-mode verdict, the predeclared-rule verdict
(cc_pool_posterior.select_runs, rhat 1.10 / div 10), its sha256 and its _fdraws sha256;
then the pooled artifact's own `selection` block and sha256s, cross-checked against the
rule re-applied here (fails closed on disagreement)."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.provenance_util import sha256
from CDDF_analysis.hbi_mcmc.cc_pool_posterior import select_runs, _seed_of

DEF = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp3-dir", default=DEF)
    ap.add_argument("--pooled", default=None)
    ap.add_argument("--rhat-max", type=float, default=1.10); ap.add_argument("--div-max", type=int, default=10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    runs = sorted(glob.glob(os.path.join(a.cp3_dir, "REAL_ln_*.json")))
    recs, rows = [], []
    for p in runs:
        d = json.load(open(p)); deep = "_deep_" in os.path.basename(p)
        dg = d["diagnostics"]
        # real runs carry diagnostics.split_rhat[key]; validation runs estimand_mixing[key].split_rhat
        rh = dg["split_rhat"] if "split_rhat" in dg else {k: v["split_rhat"] for k, v in dg["estimand_mixing"].items()}
        r = dict(file=os.path.basename(p), seed=_seed_of(p), deep=deep, n_draws=d["n_draws"],
                 rhat_20p0=rh["dndx_dla_20p0_allz"], rhat_20p3=rh["dndx_dla_20p3_allz"],
                 mean_potential_energy_per_chain=dg.get("mean_potential_energy_per_chain"),
                 divergences=d["diagnostics"].get("divergences"), G_A_real_mode=d["guards"]["G_A_real_mode"]["status"],
                 dndx_20p3_median=d["thresholds"]["dndx_dla_20p3_allz"]["post_p2p5_16_50_84_97p5"][2],
                 sha256_json=sha256(p), sha256_fdraws=sha256(p[:-5] + "_fdraws.npz") if os.path.isfile(p[:-5] + "_fdraws.npz") else None,
                 run_config=d.get("run_config", "not stamped (pre-2026-08-26 run; config in FROZEN_STATUS.json / sbatch)"))
        rows.append(r); recs.append(dict(seed=r["seed"], deep=deep, file=p, diagnostics=d["diagnostics"], guards=d["guards"]))
    sel = select_runs(recs, dict(rhat_max=a.rhat_max, div_max=a.div_max))
    inc = [(r["seed"], r["deep"]) for r in sel["included"]]; exc = [(r["seed"], r["deep"], r["reason"]) for r in sel["excluded"]]
    pooled = a.pooled or os.path.join(a.cp3_dir, "POOLED_ln_real_v2_20260821.json")
    out = dict(cp3_dir=a.cp3_dir, rule=dict(rhat_max=a.rhat_max, div_max=a.div_max), chains=rows,
               rule_reapplied=dict(included=inc, excluded=exc, needs_deep_rerun=sel["needs_deep_rerun"]))
    print(f"{'file':28s} {'seed':9s} deep  draws  rhat20.0  rhat20.3  div  G_A   dN/dX(>=20.3) med")
    for r in rows:
        print(f"{r['file']:28s} {r['seed']:<9d} {str(r['deep']):5s} {r['n_draws']:<6d} {float(r['rhat_20p0']):<9.3f} {float(r['rhat_20p3']):<9.3f} {str(r['divergences']):4s} {r['G_A_real_mode']:5s} {r['dndx_20p3_median']:.5f}")
    print("rule re-applied -> included:", inc, "| excluded:", exc)
    if os.path.isfile(pooled):
        P = json.load(open(pooled)); ps = P["selection"]
        rec_inc = [(r["seed"], r["deep"]) for r in ps["included"]]; rec_exc = [(r["seed"], r["deep"], r["reason"]) for r in ps["excluded"]]
        agree = (sorted(rec_inc) == sorted(inc))
        out.update(pooled=pooled, pooled_sha256=sha256(pooled), pooled_fdraws_sha256=sha256(pooled[:-5] + "_fdraws.npz"),
                   pooled_selection=dict(included=rec_inc, excluded=rec_exc), pooled_n_draws=P["n_draws"], selection_agrees=agree)
        print(f"pooled artifact {os.path.basename(pooled)}: n_draws {P['n_draws']}, included {rec_inc}; agrees with the re-applied rule: {agree}")
        if not agree:
            raise SystemExit("chain_ledger: the pooled artifact's selection does not match the predeclared rule re-applied to the chains")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
