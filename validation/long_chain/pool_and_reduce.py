"""VALIDATION-ONLY: pool the PASS runs (PREDECLARATION sec.6), reduce with the paper's own
reduction, evaluate the sealed criteria on the pool, and compare against the current C1 pool.

Equal weight, ascending seed order, each run's chains kept separate.  No weighting, no
thinning, no chain dropping.  The pool is a CANDIDATE statistical posterior; it is not
adopted, not frozen, and does not supersede anything.

Usage: pool_and_reduce.py <pack.npz> <outdir> <label> <run.json> [<run.json> ...]
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mfho/Latex/gp_dla_desi_y3/paper_figures")
import hbi_reduction as HR                                     # noqa: E402
import convdiag as C                                           # noqa: E402
from quantities import Q, by_chain, SAMPLED_SITES              # noqa: E402
from analyze_runs import (diagnostics_for, RHAT_MAX, ESS_BULK_MIN,   # noqa: E402
                          ESS_TAIL_MIN, EBFMI_MIN, HEAD)
sys.path.insert(0, "/home/mfho/wt_c1_rebuild_2026-09-11")
from CDDF_analysis.hbi_mcmc.pack import load_pack              # noqa: E402
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior  # noqa: E402

C1_REDUCED = ("/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff/"
              "LOWZ_CLEAN_C1_DURABLE_CANDIDATE_2026-09-11/reductions/"
              "reduced_CLEAN_C1_pooled.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def printed(k, x):
    if k.startswith("dndx"):
        return round(x, 4)
    if k.startswith("omega"):
        return round(x * 1e4, 3)
    return None


def main():
    pack, outdir, label = sys.argv[1], sys.argv[2], sys.argv[3]
    runs = sys.argv[4:]
    os.makedirs(outdir, exist_ok=True)
    recs = sorted(((json.load(open(r)), r) for r in runs),
                  key=lambda t: int(t[0]["run_config"]["seed"]))
    F, NUIS, PE, DIV, chains_per_run = [], {s: [] for s in SAMPLED_SITES}, [], [], []
    for j, r in recs:
        base = r[:-5]
        d = np.load(base + "_fdraws.npz")
        F.append(np.asarray(d["f"], float))
        z = np.load(base + "_bychain.npz")
        for s in SAMPLED_SITES:
            NUIS[s].append(np.asarray(z[s], float))
        PE.append(np.asarray(z["potential_energy"], float))
        DIV.append(np.asarray(z["diverging"]))
        chains_per_run.append(int(j["chains"]))
        ntrue, zf = np.asarray(d["ntrue_edges"], float), np.asarray(d["zf_edges"], float)
    f_all = np.concatenate(F, axis=0)
    pkl = load_pack(pack)

    # ---- POOLED summary in the committed cc_real_posterior format ------------
    red = reduce_f_posterior(f_all, pkl)
    dN = np.diff(ntrue)
    dX_k = np.asarray(pkl.dX, float).sum(axis=1)

    def q(dr):
        return [float(x) for x in np.percentile(dr, [2.5, 16, 50, 84, 97.5])]

    summ = dict(role=("POOLED long-chain CANDIDATE posterior — campaign PREDECLARATION "
                      "sec.6; equal-weight concatenation in ascending seed order; "
                      "STATISTICAL interval only; NOT adopted, NOT frozen"),
                label=label, pack=pack, pack_sha256=sha(pack),
                n_draws=int(f_all.shape[0]),
                seeds=[int(j["run_config"]["seed"]) for j, _ in recs],
                per_run=[{"seed": int(j["run_config"]["seed"]), "file": r,
                          "chains": j["chains"], "samples": j["samples"],
                          "warmup": j["warmup"],
                          "fdraws_sha256": sha(r[:-5] + "_fdraws.npz")} for j, r in recs],
                thresholds={k: dict(post_p2p5_16_50_84_97p5=q(np.asarray(red[k])))
                            for k in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz")},
                reporting_bins=[])
    for e0, e1 in zip(np.arange(19.7, 21.5 + 1e-9, 0.2),
                      np.arange(19.9, 21.7 + 1e-9, 0.2)):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        dr = ((f_all[:, m, :] * dN[None, m, None]).sum(axis=1)
              * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        summ["reporting_bins"].append(dict(bin=[round(e0, 1), round(e1, 1)], f_post=q(dr)))
    sp = os.path.join(outdir, f"POOLED_{label}.json")
    dp = os.path.join(outdir, f"POOLED_{label}_fdraws.npz")
    np.savez(dp, f=f_all, ntrue_edges=ntrue, zf_edges=zf)
    json.dump(summ, open(sp, "w"), indent=1)

    # ---- the paper's own reduction (read-only import; reduce.py logic) -------
    P = HR.Posterior(dp, pack, sp)
    Qty = {}
    for key, thr in HR.THRESHOLDS.items():
        Qty[f"dndx_{key}_allz"] = P.dndx(thr)["quantiles"]
        for rec in P.reduce_to_bins(thr):
            if rec["available"]:
                Qty[f"dndx_{key}_{rec['bin']}"] = rec["dndx"]["quantiles"]
                if key == "20p3" and rec["bin"] != "B5":
                    Qty[f"omega_20p3_21p6_{rec['bin']}"] = rec["omega"]["quantiles"]
    Qty["omega_20p3_21p6_allz"] = P.omega(*HR.OMEGA_NHI)["quantiles"]
    cd = P.cddf()
    for b in cd:
        Qty[f"cddf_{b['lo']:.1f}_{b['hi']:.1f}"] = b["q_per_dex"]
    reduced = dict(label=label, n_draws=int(f_all.shape[0]), quantile_order=HR.QUANTILES,
                   inputs=dict(draws=dp, draws_sha256=sha(dp), pack=pack,
                               pack_sha256=sha(pack), summary=sp, summary_sha256=sha(sp)),
                   quantities=Qty, cddf_bins=[{"lo": b["lo"], "hi": b["hi"],
                                               "tier": b["tier"]} for b in cd],
                   closure_vs_summary=P.closure())
    rp = os.path.join(outdir, f"reduced_{label}.json")
    json.dump(reduced, open(rp, "w"), indent=1)

    # ---- per-draw vectors, cross-checked against the reduction ---------------
    vec = Q(dp, pack).vectors()
    xchk = {}
    for k, v in vec.items():
        got = np.percentile(v, HR.QUANTILES)
        want = np.asarray(Qty[k])
        xchk[k] = float(np.max(np.abs(got / want - 1.0)))
    max_x = max(xchk.values())
    if max_x > 1e-12:
        raise SystemExit(f"BLOCKED: per-draw vectors disagree with the reduction "
                         f"(max rel {max_x:.3g})")

    # ---- pooled diagnostics: chains kept separate ---------------------------
    nch_tot = sum(chains_per_run)
    qd = {}
    off = 0
    for k, v in vec.items():
        mats = []
        o = 0
        for (j, _), fr in zip(recs, F):
            n = fr.shape[0]
            mats.append(by_chain(v[o:o + n], int(j["chains"])))
            o += n
        qd[k] = diagnostics_for(np.concatenate(mats, axis=0))
    _ = off
    nd = {}
    for s in SAMPLED_SITES:
        a = np.concatenate(NUIS[s], axis=0)
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        ev = a.shape[2:]
        if flat.shape[2] == 1:
            nd[s] = diagnostics_for(flat[:, :, 0])
            continue
        for jx in range(flat.shape[2]):
            idx = np.unravel_index(jx, ev)
            nd[s + "[" + ",".join(str(i) for i in idx) + "]"] = diagnostics_for(flat[:, :, jx])
    Eall = np.concatenate(PE, axis=0)
    Dall = np.concatenate(DIV, axis=0)
    eb = [round(C.ebfmi(e), 4) for e in Eall]
    allq = list(qd.values()) + list(nd.values())
    names = list(qd) + list(nd)
    wr = max(allq, key=lambda x: x["rhat_true"])
    wb = min(allq, key=lambda x: x["ess_bulk"])
    wt = min(allq, key=lambda x: x["ess_tail"])
    crit = dict(max_rhat_true=wr["rhat_true"], max_rhat_true_quantity=names[allq.index(wr)],
                min_ess_bulk=wb["ess_bulk"], min_ess_bulk_quantity=names[allq.index(wb)],
                min_ess_tail=wt["ess_tail"], min_ess_tail_quantity=names[allq.index(wt)],
                divergences=int(Dall.sum()), ebfmi_min=float(np.min(eb)),
                n_chains=nch_tot, n_quantities=len(allq),
                pass_rhat=bool(wr["rhat_true"] <= RHAT_MAX),
                pass_ess_bulk=bool(wb["ess_bulk"] >= ESS_BULK_MIN),
                pass_ess_tail=bool(wt["ess_tail"] >= ESS_TAIL_MIN),
                pass_divergences=bool(int(Dall.sum()) == 0),
                pass_ebfmi=bool(np.min(eb) >= EBFMI_MIN))
    crit["POOL_PASS"] = bool(all(crit[k] for k in ("pass_rhat", "pass_ess_bulk",
                                                   "pass_ess_tail", "pass_divergences",
                                                   "pass_ebfmi")))

    # ---- comparison with the current C1 pool of record ----------------------
    c1 = json.load(open(C1_REDUCED))["quantities"]
    cmp = {}
    for k, new in Qty.items():
        old = c1[k]
        hw_o, hw_n = 0.5 * (old[3] - old[1]), 0.5 * (new[3] - new[1])
        cmp[k] = dict(c1_q2p5_16_50_84_97p5=old, new_q2p5_16_50_84_97p5=new,
                      delta_median_pct=100.0 * (new[2] / old[2] - 1.0),
                      delta_median_over_c1_hw=(new[2] - old[2]) / hw_o,
                      c1_halfwidth=hw_o, new_halfwidth=hw_n,
                      halfwidth_ratio_new_over_c1=hw_n / hw_o,
                      intervals_overlap=bool(new[1] <= old[3] and old[1] <= new[3]),
                      printed_c1=printed(k, old[2]), printed_new=printed(k, new[2]),
                      printed_c1_interval=[printed(k, old[1]), printed(k, old[3])],
                      printed_new_interval=[printed(k, new[1]), printed(k, new[3])],
                      printed_digit_change=(None if printed(k, old[2]) is None
                                            else bool(printed(k, old[2]) != printed(k, new[2]))),
                      ess_bulk=qd[k]["ess_bulk"], ess_tail=qd[k]["ess_tail"],
                      mcse_median=qd[k]["mcse_median"],
                      mcse_over_new_hw=qd[k]["mcse_median"] / hw_n if hw_n else None)

    res = dict(label=label, seeds=summ["seeds"], n_draws=summ["n_draws"],
               n_chains=nch_tot, pooled_summary=sp, pooled_summary_sha256=sha(sp),
               pooled_draws=dp, pooled_draws_sha256=sha(dp),
               reduced=rp, reduced_sha256=sha(rp),
               reduction_closure={k: v["max_rel_diff"] for k, v in
                                  reduced["closure_vs_summary"].items()},
               perdraw_vs_reduction_max_rel=max_x,
               criteria=crit, reported=qd, nuisance=nd,
               ebfmi_per_chain=eb, vs_C1_pool=cmp,
               headline_q16_q50_q84={h: qd[h]["q16_q50_q84"] for h in HEAD},
               t_per_chain_medians={f"t[{k}]": nd[f"t[{k}]"]["chain_medians"]
                                    for k in range(3)},
               pe_chain_means=[round(float(e.mean()), 1) for e in Eall])
    op = os.path.join(outdir, f"POOL_EVAL_{label}.json")
    json.dump(res, open(op, "w"), indent=1)
    print(f"pool {label}: seeds={res['seeds']} draws={res['n_draws']} chains={nch_tot}")
    print(f"  PASS={crit['POOL_PASS']} Rhat={crit['max_rhat_true']:.4f} "
          f"({crit['max_rhat_true_quantity']}) ESSb={crit['min_ess_bulk']:.0f} "
          f"ESSt={crit['min_ess_tail']:.0f} div={crit['divergences']} ebfmi={crit['ebfmi_min']:.3f}")
    for h in HEAD:
        c = cmp[h]
        print(f"  {h:22s} C1 {c['c1_q2p5_16_50_84_97p5'][2]:.6g} -> new "
              f"{c['new_q2p5_16_50_84_97p5'][2]:.6g}  d={c['delta_median_over_c1_hw']:+.2f} hw "
              f"({c['delta_median_pct']:+.2f} %)  hw ratio {c['halfwidth_ratio_new_over_c1']:.3f}")
    print("WROTE", op)


if __name__ == "__main__":
    main()
