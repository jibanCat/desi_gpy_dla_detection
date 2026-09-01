#!/usr/bin/env python3
"""chain_index.py — the CHAIN INDEX required by the PI's MCMC transparency standard (ruling 2026-09-01 #8/#21):
one row per CHAIN of every run of the 2026-09-02 campaign — run, on-disk run dir, ruled run name, arm (seed, deep/base/long),
chain id, initialisation family, warmup, samples, intervention flags, mean potential energy, t means, log Λ mean, estimand
medians, per-chain ESS of Λ and t₀, whether the arm is in the pooled product and why not, and the sha256 of the all-sites file.
Never drops a chain. PRIVATE output (real-data values).

    python tools/hbi_validation/chain_index.py --root ROOT --out-csv CHAIN_INDEX.csv --out-json CHAIN_INDEX.json
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.provenance_util import sha256   # noqa: E402
from tools.hbi_validation.geometry import split_rhat_ess     # noqa: E402

RULED = {"R0": "R0", "R1": "R1", "R2": "R2-A", "R2B": "R2-Binit", "R2b": "R2x4", "R3": "R3", "R4": "R4", "LC": "LC (long chain)"}
INIT = {"R2B": "family B (t=0, Lambda=naive, v=m_cs, psi_c=0)", }


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out-csv", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    rows = []
    for rdir in sorted(glob.glob(os.path.join(a.root, "R*")) + glob.glob(os.path.join(a.root, "LC"))):
        run = os.path.basename(rdir)
        if run not in RULED:
            continue
        pooled = glob.glob(os.path.join(rdir, "POOLED_ln_*.json")); sel = {}
        if pooled:
            P = json.load(open(pooled[0]))
            for x in P["selection"]["included"]:
                sel[(x["seed"], x["deep"])] = ("INCLUDED", "")
            for x in P["selection"]["excluded"]:
                sel[(x["seed"], x["deep"])] = ("EXCLUDED", x["reason"])
        for p in sorted(glob.glob(os.path.join(rdir, "REAL_ln_*s2026????.json"))):
            d = json.load(open(p)); g = d["diagnostics"]; rc = d.get("run_config", {})
            base = os.path.basename(p); seed = int(re.search(r"_s(\d{8})\.json$", base).group(1))
            kind = "deep" if "_deep_" in base else ("long" if "_lc_" in base else "base")
            asp = p[:-5] + "_allsites.npz"; z = np.load(asp) if os.path.isfile(asp) else None
            nch = int(z["t"].shape[0]) if z is not None else int(rc.get("chains", 2))
            status, why = sel.get((seed, kind == "deep"), ("NOT IN SELECTION (long chain / not pooled)", ""))
            for c in range(nch):
                r = dict(run_dir=run, run=RULED[run], seed=seed, arm_kind=kind, chain=c, init_family=INIT.get(run, "production default (init_to_uniform)") if not g.get("validation_flags", {}).get("init_from") else "family B (--init-from)",
                         warmup=rc.get("warmup"), samples=rc.get("samples"), flags=json.dumps(g.get("validation_flags")), t_scale=g.get("t_scale"),
                         pe_mean=(g.get("mean_potential_energy_per_chain") or [None] * nch)[c], divergences_arm=g.get("divergences"),
                         split_rhat_20p0_arm=g["split_rhat"]["dndx_dla_20p0_allz"], split_rhat_20p3_arm=g["split_rhat"]["dndx_dla_20p3_allz"],
                         est_median_20p3=g["perchain_estimand_medians"]["dndx_dla_20p3_allz"][c], est_median_20p0=g["perchain_estimand_medians"]["dndx_dla_20p0_allz"][c],
                         arm_status=status, exclusion_reason=why, json_sha256=sha256(p), allsites_sha256=(sha256(asp) if z is not None else None))
                if z is not None:
                    t = np.asarray(z["t"])[c]; lam = np.asarray(z["lam_fp"])[c].sum(axis=(1, 2))
                    r.update(t_mean=t.mean(axis=0).round(4).tolist(), logL_mean=float(np.log(lam).mean()), ess_t0_chain=float(split_rhat_ess(t[None, :, 0])[1]) if t.std(axis=0)[0] > 0 else None,
                             ess_logL_chain=float(split_rhat_ess(np.log(lam)[None])[1]), n_draws=int(t.shape[0]))
                rows.append(r)
    with open(a.out_csv, "w", newline="") as fh:
        keys = sorted(set(k for r in rows for k in r)); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump(rows, open(a.out_json, "w"), indent=1)
    from collections import Counter
    print(len(rows), "chains indexed;", Counter((r["run"], r["arm_status"].split(" ")[0]) for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
