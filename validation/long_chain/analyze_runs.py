"""VALIDATION-ONLY: evaluate the SEALED campaign criteria on a stage's runs.

PREDECLARATION sec.4 (criteria), sec.5 (mode classification), sec.6 (pooling).  Nothing here
may be relaxed; the thresholds are literals fixed before any sampling.

Usage: analyze_runs.py <pack.npz> <out.json> <run.json> [<run.json> ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convdiag as C                                          # noqa: E402
from quantities import Q, by_chain, nuisance_components       # noqa: E402

RHAT_MAX, ESS_BULK_MIN, ESS_TAIL_MIN, EBFMI_MIN = 1.01, 400.0, 400.0, 0.3
PE_GAP_NATS, BLOCK = 50.0, 100
HEAD = ["dndx_20p3_allz", "dndx_20p0_allz", "omega_20p3_21p6_allz"]


def diagnostics_for(mat):
    m = np.asarray(mat, float)
    return dict(rhat_true=C.az_rhat_true(m), rhat_fold=C.az_rhat_folded(m),
                ess_bulk=C.az_ess_bulk(m), ess_tail=C.az_ess_tail(m),
                mcse_median=C.az_mcse_median(m),
                chain_medians=[float(np.median(c)) for c in m],
                q16_q50_q84=[float(v) for v in np.percentile(m, [16, 50, 84])])


def mode_report(E):
    """PREDECLARATION sec.5.  E: (chains, draws) potential energy."""
    E = np.asarray(E, float)
    mc = E.mean(axis=1)
    order = np.argsort(mc)
    s = mc[order]
    d = np.diff(s)
    if d.size == 0:
        return dict(pe_gap=0.0, classification_basis="single chain")
    j = int(np.argmax(d))
    gap = float(d[j])
    T = float(0.5 * (s[j] + s[j + 1]))
    lower = [int(c) for c in order[:j + 1]]
    upper = [int(c) for c in order[j + 1:]]
    nb = E.shape[1] // BLOCK
    cross_block, cross_draw = [], []
    for c in range(E.shape[0]):
        if nb >= 2:
            bm = np.median(E[c, :nb * BLOCK].reshape(nb, BLOCK), axis=1) - T
            cross_block.append(int(np.sum(np.diff(np.sign(bm)) != 0)))
        else:
            cross_block.append(0)
        sd = np.sign(E[c] - T)
        cross_draw.append(int(np.sum(np.diff(sd) != 0)))
    return dict(pe_chain_means=[round(float(v), 1) for v in mc],
                pe_gap=round(gap, 1), pe_threshold=round(T, 1),
                lower_cluster_chains=lower, upper_cluster_chains=upper,
                crossings_block100_per_chain=cross_block,
                crossings_perdraw_per_chain=cross_draw,
                genuine_between_mode_mixing=bool(gap > PE_GAP_NATS
                                                 and max(cross_block) >= 2))


def evaluate_run(run_json, pack):
    j = json.load(open(run_json))
    base = run_json[:-5]
    nch = int(j["chains"])
    vec = Q(base + "_fdraws.npz", pack).vectors()
    nz = np.load(base + "_bychain.npz")
    E = np.asarray(nz["potential_energy"], float)
    div = np.asarray(nz["diverging"])
    qd = {n: diagnostics_for(by_chain(v, nch)) for n, v in vec.items()}
    nd = {n: diagnostics_for(m) for n, m in nuisance_components(base + "_bychain.npz").items()}
    eb = [round(C.ebfmi(e), 4) for e in E]
    rec = dict(run=run_json, seed=int(nz["seed"]), chains=nch,
               warmup=int(nz["warmup"]), samples=int(nz["samples"]),
               n_draws=int(j["n_draws"]), stage=j.get("stage", ""),
               divergences=int(div.sum()),
               divergences_per_chain=[int(v) for v in div.sum(axis=1)],
               ebfmi_per_chain=eb, ebfmi_min=float(np.min(eb)),
               mode=mode_report(E),
               reported=qd, nuisance=nd,
               t_per_chain_medians={f"t[{k}]": nd[f"t[{k}]"]["chain_medians"]
                                    for k in range(3)})
    allq = list(qd.values()) + list(nd.values())
    names = list(qd) + list(nd)
    ir = max(range(len(allq)), key=lambda i: allq[i]["rhat_true"])
    ib = min(range(len(allq)), key=lambda i: allq[i]["ess_bulk"])
    it = min(range(len(allq)), key=lambda i: allq[i]["ess_tail"])
    worst_r, worst_b, worst_t = allq[ir], allq[ib], allq[it]
    rec["criteria"] = dict(
        max_rhat_true=worst_r["rhat_true"],
        max_rhat_true_quantity=names[ir],
        min_ess_bulk=worst_b["ess_bulk"],
        min_ess_bulk_quantity=names[ib],
        min_ess_tail=worst_t["ess_tail"],
        min_ess_tail_quantity=names[it],
        n_quantities=len(allq),
        pass_rhat=bool(worst_r["rhat_true"] <= RHAT_MAX),
        pass_ess_bulk=bool(worst_b["ess_bulk"] >= ESS_BULK_MIN),
        pass_ess_tail=bool(worst_t["ess_tail"] >= ESS_TAIL_MIN),
        pass_divergences=bool(int(div.sum()) == 0),
        pass_ebfmi=bool(min(eb) >= EBFMI_MIN))
    rec["PASS"] = bool(all(rec["criteria"][k] for k in
                           ("pass_rhat", "pass_ess_bulk", "pass_ess_tail",
                            "pass_divergences", "pass_ebfmi")))
    if rec["PASS"]:
        cls = "PASS"
    elif rec["mode"]["pe_gap"] > PE_GAP_NATS:
        cls = ("MULTIMODAL-MIXING" if rec["mode"]["genuine_between_mode_mixing"]
               else "MULTIMODAL-DISCONNECTED")
    else:
        cls = "SLOW-MIXING/UNRESOLVED"
    rec["classification"] = cls
    rec["headline_q16_q50_q84"] = {h: qd[h]["q16_q50_q84"] for h in HEAD}
    return rec


def main():
    pack, out = sys.argv[1], sys.argv[2]
    runs = sys.argv[3:]
    res = {"criteria_literals": dict(rhat_true_max=RHAT_MAX, ess_bulk_min=ESS_BULK_MIN,
                                     ess_tail_min=ESS_TAIL_MIN, divergences_max=0,
                                     ebfmi_min=EBFMI_MIN, pe_gap_nats=PE_GAP_NATS,
                                     mixing_block=BLOCK),
           "pack": pack, "runs": {}}
    for r in runs:
        rec = evaluate_run(r, pack)
        res["runs"][str(rec["seed"])] = rec
        print(f"{rec['seed']} {rec['classification']:26s} Rhat={rec['criteria']['max_rhat_true']:.4f} "
              f"({rec['criteria']['max_rhat_true_quantity']}) ESSb={rec['criteria']['min_ess_bulk']:.0f} "
              f"ESSt={rec['criteria']['min_ess_tail']:.0f} div={rec['divergences']} "
              f"ebfmi={rec['ebfmi_min']:.3f} dPE={rec['mode']['pe_gap']}", flush=True)
    res["classification"] = {s: r["classification"] for s, r in res["runs"].items()}
    res["PASS_seeds"] = sorted(s for s, r in res["runs"].items() if r["PASS"])
    res["MULTIMODAL_DISCONNECTED_seeds"] = sorted(
        s for s, r in res["runs"].items() if r["classification"] == "MULTIMODAL-DISCONNECTED")
    res["MULTIMODAL_MIXING_seeds"] = sorted(
        s for s, r in res["runs"].items() if r["classification"] == "MULTIMODAL-MIXING")
    json.dump(res, open(out, "w"), indent=1)
    print("WROTE", out)


if __name__ == "__main__":
    main()
