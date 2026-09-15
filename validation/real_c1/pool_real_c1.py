"""Pooled, truth-free read-out of the blind real-C1 production runs (REAL_C1_BLIND_RUN_PREDECLARATION.md, sealed af74073c).

Equal-weight pool of the f draws over the J = 8 Lambda imputations (per seed and over both seeds), reduced with the SAME
truth-free functions the runner used (validation/real_c1/reduce_truthfree.py). Also: per-run headline medians and 68 %
half-widths, imputation-to-imputation and seed-to-seed spreads in half-width units, sampler health, t_K, FP totals,
predictive marginals. Values are written to the JSON/MD outputs only (private notes repo); stdout prints health and spreads.
Works identically on mock J = 8 runs (used as the tool's test).
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np


def load_runs(runs_dir, pattern="RUN_*.json"):
    R = []
    for p in sorted(glob.glob(os.path.join(runs_dir, pattern))):
        j = json.load(open(p)); fd = p[:-5] + "_fdraws.npz"
        if not os.path.exists(fd):
            continue
        lc = j.get("lam_cut") or (j.get("diagnostics") or {}).get("lam_cut") or {}
        seed = (j.get("run_config") or {}).get("seed")
        R.append(dict(path=p, json=j, fdraws=fd, j=lc.get("j"), J=lc.get("J"), lam=lc.get("lam_fixed"), seed=seed))
    return R


def headline_from_json(j):
    """Per-run headline medians/hw from either runner's JSON schema (real: estimands.thresholds_allz; mock: thresholds)."""
    th = (j.get("estimands") or {}).get("thresholds_allz") or j.get("thresholds")
    out = {}
    for key in ("ge20.0", "ge20.3"):
        t = th[key]; p = t.get("post_p16_50_84") or t.get("p16_50_84")
        out[key] = dict(median=float(p[1]), hw68=float(0.5 * (p[2] - p[0])))
    return out


def pool(R, pk):
    from validation.real_c1 import reduce_truthfree as RT
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
    f = np.concatenate([np.load(r["fdraws"])["f"] for r in R], axis=0)
    ntrue = np.asarray(pk.ntrue_edges, float); dX_k = np.asarray(pk.dX, float).sum(axis=1)
    red = reduce_f_posterior(f, pk)
    return dict(n_runs=len(R), n_draws=int(f.shape[0]),
                thresholds_allz=RT.thresholds_allz(f, pk, red=red),
                reporting_bins_0p2dex=RT.reporting_bins_0p2dex(f, pk),
                perz_posterior=RT.perz_posterior(f, pk),
                omega_20p3_21p6_allz=RT.omega_20p3_21p6_allz(f, ntrue, np.asarray(pk.zf_edges, float), dX_k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True); ap.add_argument("--pack", required=True)
    ap.add_argument("--pattern", default="RUN_*.json")
    ap.add_argument("--out-json", required=True); ap.add_argument("--out-md", required=True)
    ap.add_argument("--print-values", action="store_true", help="print estimand values to stdout (MOCK use only)")
    a = ap.parse_args()
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    pk = load_pack(a.pack)
    R = load_runs(a.runs_dir, a.pattern)
    if not R:
        raise SystemExit("pool_real_c1: no runs found (fail closed)")
    seeds = sorted(set(r["seed"] for r in R))
    per_run = []
    for r in R:
        j = r["json"]; h = headline_from_json(j)
        sh = j.get("sampler_health") or {}; d = j.get("diagnostics") or {}
        per_run.append(dict(file=os.path.basename(r["path"]), seed=r["seed"], j=r["j"], lam=r["lam"],
                            ge20p0=h["ge20.0"], ge20p3=h["ge20.3"],
                            divergences=int(j.get("divergences", sh.get("divergences", -1))),
                            ebfmi=sh.get("ebfmi_per_chain") or d.get("ebfmi_per_chain"),
                            rank_rhat={k: v.get("rank_split_rhat", v.get("split_rhat")) for k, v in (sh.get("estimand_mixing") or {}).items()} or None,
                            ess={k: v.get("ess_bulk", v.get("ess")) for k, v in (sh.get("estimand_mixing") or {}).items()} or None,
                            t_mixing=sh.get("t_mixing_per_K"),
                            t_K=(j.get("t_posterior") or {}).get("mean") or d.get("t_post_mean"),
                            fp_total=(j.get("fp_totals") or {}).get("mu_fp_total_p16_50_84") or ((d.get("fp_by_block") or {}).get("mu_fp_total_p16_50_84"))))
    pools = {"all": pool(R, pk)}
    for s in seeds:
        pools[f"seed{s}"] = pool([r for r in R if r["seed"] == s], pk)
    # spreads in units of the pooled half-width
    spreads = {}
    for key in ("ge20p0", "ge20p3"):
        thr = "ge20.0" if key == "ge20p0" else "ge20.3"
        p = pools["all"]["thresholds_allz"][thr]; pp = p.get("post_p16_50_84") or p.get("p16_50_84"); hw = 0.5 * (pp[2] - pp[0])
        meds = np.array([x[key]["median"] for x in per_run])
        by_seed = {s: float(np.median([x[key]["median"] for x in per_run if x["seed"] == s])) for s in seeds}
        spreads[thr] = dict(pooled_hw68=float(hw), imputation_spread_over_hw=float((meds.max() - meds.min()) / hw),
                            seed_spread_over_hw=float((max(by_seed.values()) - min(by_seed.values())) / hw) if len(seeds) > 1 else None,
                            per_seed_pool_medians_agree_over_hw=(float(abs(pools[f"seed{seeds[0]}"]["thresholds_allz"][thr][("post_p16_50_84" if "post_p16_50_84" in pools[f"seed{seeds[0]}"]["thresholds_allz"][thr] else "p16_50_84")][1] - pools[f"seed{seeds[1]}"]["thresholds_allz"][thr][("post_p16_50_84" if "post_p16_50_84" in pools[f"seed{seeds[1]}"]["thresholds_allz"][thr] else "p16_50_84")][1]) / hw) if len(seeds) > 1 else None))
    health = dict(max_divergences=max(x["divergences"] for x in per_run),
                  min_ebfmi=min(min(x["ebfmi"]) for x in per_run if x["ebfmi"]),
                  n_runs=len(per_run), seeds=seeds, imputations=sorted(set(x["j"] for x in per_run)))
    out = dict(sealed_predeclaration="REAL_C1_BLIND_RUN_PREDECLARATION.md af74073c", pack=a.pack, per_run=per_run, pools=pools, spreads=spreads, health=health)
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    md = ["# Pooled read-out (equal weight over J = 8 imputations, both seeds)", "", f"runs {len(per_run)}; seeds {seeds}; imputations {health['imputations']}; max divergences {health['max_divergences']}; min E-BFMI {health['min_ebfmi']:.3f}", "",
          "| estimand | pooled p2.5 | p16 | median | p84 | p97.5 | 68 % hw | imputation spread / hw | seed spread / hw |", "|---|---|---|---|---|---|---|---|---|"]
    for thr in ("ge20.0", "ge20.3"):
        p = pools["all"]["thresholds_allz"][thr]; p5 = p.get("post_p2p5_97p5") or p.get("p2p5_97p5") or [float("nan")] * 2; pp = p.get("post_p16_50_84") or p.get("p16_50_84")
        s = spreads[thr]
        md.append(f"| dN/dX({thr}) all-z | {p5[0]:.5f} | {pp[0]:.5f} | **{pp[1]:.5f}** | {pp[2]:.5f} | {p5[1]:.5f} | {s['pooled_hw68']:.5f} | {s['imputation_spread_over_hw']:.3f} | {s['seed_spread_over_hw'] if s['seed_spread_over_hw'] is None else round(s['seed_spread_over_hw'],3)} |")
    om = pools["all"]["omega_20p3_21p6_allz"]; md.append(f"| Ω[20.3,21.6] all-z | | {om.get('p16_50_84', om.get('post_p16_50_84', [None]*3))[0]} | **{om.get('p16_50_84', om.get('post_p16_50_84', [None]*3))[1]}** | {om.get('p16_50_84', om.get('post_p16_50_84', [None]*3))[2]} | | | | |")
    md += ["", "## Per-run table", "",
           "Mixing columns: headline = rank-normalised split-R̂ / bulk ESS of dN/dX(≥20.0) and (≥20.3); "
           "t_K mixing = rank-R̂ and bulk ESS of the three coarse-z FP nuisance sites (PI 2026-09-14b §11: the t_K "
           "location is reported WITH its non-convergence and is never a measured FP transfer).", "",
           "| file | seed | j | Λ_j | ≥20.0 median (hw) | ≥20.3 median (hw) | div | E-BFMI | t_K | headline rank-R̂ | headline ESS_bulk | t_K rank-R̂ | t_K ESS_bulk |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in per_run:
        rr = x.get("rank_rhat") or {}; es = x.get("ess") or {}; tm = x.get("t_mixing") or []
        _o = ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz")
        hr = [round(rr[k], 4) for k in _o if rr.get(k) is not None] or None
        he = [round(float(es[k]), 0) for k in _o if es.get(k) is not None] or None
        tr = [round(d.get("rank_split_rhat"), 3) for d in tm if d.get("rank_split_rhat") is not None] or None
        te = [round(d.get("ess_bulk"), 0) for d in tm if d.get("ess_bulk") is not None] or None
        md.append(f"| {x['file']} | {x['seed']} | {x['j']} | {x['lam']:.3f} | {x['ge20p0']['median']:.5f} ({x['ge20p0']['hw68']:.5f}) | {x['ge20p3']['median']:.5f} ({x['ge20p3']['hw68']:.5f}) | {x['divergences']} | {[round(e,2) for e in x['ebfmi']] if x['ebfmi'] else None} | {[round(t,2) for t in x['t_K']] if x['t_K'] else None} | {hr} | {he} | {tr} | {te} |")
    open(a.out_md, "w").write("\n".join(md) + "\n")
    print(json.dumps(dict(health=health, spreads=spreads), indent=1, default=float))
    if a.print_values:
        print(json.dumps(pools["all"]["thresholds_allz"], indent=1, default=float))


if __name__ == "__main__":
    main()
