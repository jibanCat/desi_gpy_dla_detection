"""dash_readout.py — READ-ONLY deterministic read-out for the PI inspection sections
SEC_10 (systematics dashboard), SEC_11 (sampler health), SEC_12 (hygiene).

Touches no frozen object: it only opens stored JSON products and writes a
scratch summary JSON + two figures.  No fits, no MCMC, no re-reduction.
"""
import json, glob, os, hashlib, re, sys

R = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
POOLED = R + "/real_c1/REAL_C1_POOLED.json"
SYSJ = R + "/release/systematics/SYSTEMATICS_TABLE.json"
MOCKJ8 = R + "/final/runs/B-phi2lpt-C1nsadd-M1CUTJ8"
REALRUNS = R + "/real_c1/runs"


def sha8(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8]


def rng(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def main():
    out = {}
    pool = json.load(open(POOLED))
    # --- relative 68% half-widths of the real pooled posterior (percent of median)
    hw = {}
    for key, tag in (("ge20.0", "ge20.0"), ("ge20.3", "ge20.3")):
        th = pool["pools"]["all"]["thresholds_allz"][key]
        p16, p50, p84 = th["post_p16_50_84"]
        h = 0.5 * (p84 - p16)
        hw[tag] = dict(hw68_abs=h, median=p50, hw68_pct=100.0 * h / p50)
    om = pool["pools"]["all"]["omega_20p3_21p6_allz"]
    o16, o50, o84 = om["post_p16_50_84"]
    om_hw = 0.5 * (o84 - o16)
    hw["omega"] = dict(hw68_abs=om_hw, median=o50, hw68_pct=100.0 * om_hw / o50)
    out["real_pooled_halfwidths"] = hw
    out["real_spreads"] = pool["spreads"]
    out["real_health"] = pool["health"]
    out["pooled_perz_keys"] = list(pool["pools"]["all"]["perz_posterior"].keys())

    # --- null machine-readable fields in per_run
    nulls = {}
    for r in pool["per_run"]:
        for k, v in r.items():
            if v is None or v == [] or v == {}:
                nulls[k] = nulls.get(k, 0) + 1
    out["pooled_null_per_run_fields"] = nulls
    out["n_per_run"] = len(pool["per_run"])

    # --- systematics sizes (pp) from the released table
    sysd = json.load(open(SYSJ))
    out["sys_units"] = sysd["units"]
    out["sys_ids"] = [s.get("id") for s in sysd["systematics"]] if isinstance(
        sysd["systematics"], list) else list(sysd["systematics"].keys())
    S = sysd["systematics"]
    out["sys_summaries"] = {
        (s.get("id") or k): (s.get("summary") if isinstance(s, dict) else None)
        for k, s in (enumerate(S) if isinstance(S, list) else S.items())
        for _ in [0]
    } if False else None
    # normalise to a list of dicts
    slist = S if isinstance(S, list) else [dict(id=k, **v) for k, v in S.items()]
    out["sys"] = [{k: v for k, v in s.items() if k in ("id", "name", "title",
                   "ruling", "treatment", "summary")} for s in slist]

    # --- mock J=8 sampler health (24 runs)
    mock = []
    for f in sorted(glob.glob(MOCKJ8 + "/*.json")):
        j = json.load(open(f))
        d = j["diagnostics"]
        em = d["estimand_mixing"]
        mock.append(dict(file=os.path.basename(f),
                         family=os.path.basename(f).split("_")[1],
                         seed=int(re.search(r"_s(\d{8})_", os.path.basename(f)).group(1)),
                         div=d["divergences"], ebfmi=d["ebfmi_per_chain"],
                         rhat20p0=em["dndx_dla_20p0_allz"]["split_rhat"],
                         rhat20p3=em["dndx_dla_20p3_allz"]["split_rhat"],
                         ess20p0=em["dndx_dla_20p0_allz"]["ess"],
                         ess20p3=em["dndx_dla_20p3_allz"]["ess"],
                         t_post_sd=d.get("t_post_sd"),
                         t_post_mean=d.get("t_post_mean")))
    out["mock_n"] = len(mock)
    eb = [e for m in mock for e in m["ebfmi"]]
    out["mock"] = dict(
        div=rng([m["div"] for m in mock]),
        rhat=rng([m["rhat20p0"] for m in mock] + [m["rhat20p3"] for m in mock]),
        ess=rng([m["ess20p0"] for m in mock] + [m["ess20p3"] for m in mock]),
        ebfmi=rng(eb), n_chains=len(eb),
        n_ebfmi_lt_0p3=sum(1 for e in eb if e < 0.3),
        frac_ebfmi_lt_0p3=sum(1 for e in eb if e < 0.3) / len(eb))
    out["mock_ebfmi_all"] = eb
    out["mock_rows"] = mock

    # --- real C1 sampler health (16 runs)
    real = []
    for f in sorted(glob.glob(REALRUNS + "/RUN_REALC1_*.json")):
        j = json.load(open(f))
        sh = j["sampler_health"]
        em = sh["estimand_mixing"]
        tm = sh.get("t_mixing_per_K") or []
        real.append(dict(file=os.path.basename(f),
                         seed=int(re.search(r"_s(\d{8})_", os.path.basename(f)).group(1)),
                         div=sh["divergences"], ebfmi=sh["ebfmi_per_chain"],
                         rhat20p0=em["dndx_dla_20p0_allz"]["rank_split_rhat"],
                         rhat20p3=em["dndx_dla_20p3_allz"]["rank_split_rhat"],
                         srhat20p0=em["dndx_dla_20p0_allz"]["split_rhat"],
                         srhat20p3=em["dndx_dla_20p3_allz"]["split_rhat"],
                         essb=[em["dndx_dla_20p0_allz"]["ess_bulk"],
                               em["dndx_dla_20p3_allz"]["ess_bulk"]],
                         esst=[em["dndx_dla_20p0_allz"]["ess_tail"],
                               em["dndx_dla_20p3_allz"]["ess_tail"]],
                         t_rhat=[t["rank_split_rhat"] for t in tm],
                         t_essb=[t["ess_bulk"] for t in tm],
                         t_esst=[t["ess_tail"] for t in tm],
                         t_sd=j["t_posterior"]["sd"],
                         fp_total=j["fp_totals"]["mu_fp_total_p16_50_84"],
                         support=j["support_gate"]["status"],
                         real_gate=j["real_gate"]["truth_counts_sentinel"]))
    ebr = [e for m in real for e in m["ebfmi"]]
    out["real_n"] = len(real)
    out["real"] = dict(
        div=rng([m["div"] for m in real]),
        rank_rhat=rng([m["rhat20p0"] for m in real] + [m["rhat20p3"] for m in real]),
        split_rhat=rng([m["srhat20p0"] for m in real] + [m["srhat20p3"] for m in real]),
        ess_bulk=rng([e for m in real for e in m["essb"]]),
        ess_tail=rng([e for m in real for e in m["esst"]]),
        ebfmi=rng(ebr), n_chains=len(ebr),
        n_ebfmi_lt_0p3=sum(1 for e in ebr if e < 0.3),
        frac_ebfmi_lt_0p3=sum(1 for e in ebr if e < 0.3) / len(ebr),
        t_rank_rhat=rng([t for m in real for t in m["t_rhat"]]),
        t_ess_bulk=rng([t for m in real for t in m["t_essb"]]),
        t_ess_tail=rng([t for m in real for t in m["t_esst"]]),
        support_all_pass=all(m["support"] == "PASS" for m in real),
        real_gate_all=sorted({m["real_gate"] for m in real}))
    out["real_ebfmi_all"] = ebr
    out["real_rows"] = real

    # --- per-run headline medians in half-width units around the pool (real)
    med0 = hw["ge20.0"]["median"]; h0 = hw["ge20.0"]["hw68_abs"]
    med3 = hw["ge20.3"]["median"]; h3 = hw["ge20.3"]["hw68_abs"]
    out["real_run_offsets_hw"] = [
        dict(file=r["file"], seed=r["seed"], j=r["j"],
             d0=(r["ge20p0"]["median"] - med0) / h0,
             d3=(r["ge20p3"]["median"] - med3) / h3)
        for r in pool["per_run"]]

    # --- provenance / hygiene digests
    out["digests"] = {}
    for p in [POOLED, SYSJ, R + "/release/PROVENANCE_MANIFEST.json",
              R + "/release/model_of_record/MODEL_OF_RECORD.json",
              R + "/completeness/C1nsadd_covariance_2lpt0.npz",
              R + "/release/completeness/completeness_model.json",
              R + "/release/README.md", R + "/release/SHA256SUMS"]:
        if os.path.exists(p):
            out["digests"][p] = sha8(p)
    # SHA256SUMS vs actual for the release root
    stale = []
    sums = {}
    for line in open(R + "/release/SHA256SUMS"):
        h, n = line.split()
        sums[n] = h
    for n, h in sums.items():
        p = R + "/release/" + n
        if os.path.exists(p):
            a = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if a != h:
                stale.append(dict(name=n, recorded=h[:8], actual=a[:8]))
        else:
            stale.append(dict(name=n, recorded=h[:8], actual="MISSING"))
    out["release_sha256sums_mismatches"] = stale

    o = "/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/323b5500-b134-4e5f-8dd1-1292b65f6a96/scratchpad/dash_readout.json"
    os.makedirs(os.path.dirname(o), exist_ok=True)
    json.dump(out, open(o, "w"), indent=1)
    print("wrote", o)
    for k in ("real_pooled_halfwidths", "real_spreads", "mock", "real",
              "pooled_null_per_run_fields", "release_sha256sums_mismatches",
              "sys_ids", "real_health", "mock_n", "real_n"):
        print("==", k, "==")
        print(json.dumps(out[k], indent=1)[:1400])


if __name__ == "__main__":
    main()
