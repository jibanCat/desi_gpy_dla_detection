#!/usr/bin/env python3
"""r0_bitrepro.py — R0 gate of the 2026-09-02 HBI identifiability campaign: is the
validation-worktree reproduction of the frozen CP-3 battery bit-identical to the
frozen chains and pooled posterior? Fails closed on missing files.

    python tools/hbi_validation/r0_bitrepro.py --new-dir <R0 dir> --ref-dir <cp3_real> --out R0_BITREPRO.json
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.provenance_util import sha256

FROZEN_POOLED_JSON = "ea881b5f05c0b8127f4f417eab81e7bb74b4795e2b6317ca9c2e99793a9081b7"
FROZEN_POOLED_DRAWS = "e43d91484faf13fdded9bf8e717a192b3eb12f7006296417db59534b433645c4"
DIAG_SKIP = ("validation_flags",)


def cmp_run(new_json, ref_json):
    n, r = json.load(open(new_json)), json.load(open(ref_json))
    nd, rd = new_json[:-5] + "_fdraws.npz", ref_json[:-5] + "_fdraws.npz"
    out = dict(new=new_json, ref=ref_json, fdraws_sha_new=sha256(nd), fdraws_sha_ref=sha256(rd))
    out["fdraws_identical_sha"] = out["fdraws_sha_new"] == out["fdraws_sha_ref"]
    fn, fr = np.load(nd)["f"], np.load(rd)["f"]
    out["max_abs_diff_f"] = float(np.max(np.abs(fn - fr))) if fn.shape == fr.shape else None
    out["shape_new"], out["shape_ref"] = list(fn.shape), list(fr.shape)
    out["thresholds_identical"] = n["thresholds"] == r["thresholds"]
    out["reporting_bins_identical"] = n["reporting_bins"] == r["reporting_bins"]
    dn = {k: v for k, v in n["diagnostics"].items() if k not in DIAG_SKIP}
    dr = {k: v for k, v in r["diagnostics"].items() if k not in DIAG_SKIP}
    out["diagnostics_identical"] = dn == dr
    out["diagnostics_diff_keys"] = sorted(k for k in set(dn) | set(dr) if dn.get(k) != dr.get(k))
    out["guards_G_A_real_mode_identical"] = n["guards"]["G_A_real_mode"] == r["guards"]["G_A_real_mode"]
    out["validation_flags"] = n["diagnostics"].get("validation_flags")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True); ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = sorted(glob.glob(os.path.join(a.new_dir, "REAL_ln_*s2026????.json")))
    if not runs:
        raise SystemExit("r0_bitrepro: no run JSONs in new dir")
    rec = {"runs": {}, "pooled": {}}
    ok = True
    for p in runs:
        name = os.path.basename(p); rp = os.path.join(a.ref_dir, name)
        if not os.path.isfile(rp):
            rec["runs"][name] = dict(error="no frozen counterpart"); ok = False; continue
        r = cmp_run(p, rp); rec["runs"][name] = r
        ok &= r["fdraws_identical_sha"] and r["thresholds_identical"] and r["diagnostics_identical"]
        print(f"{name:28s} fdraws sha identical={r['fdraws_identical_sha']} max|df|={r['max_abs_diff_f']} "
              f"thr={r['thresholds_identical']} diag={r['diagnostics_identical']} diffkeys={r['diagnostics_diff_keys']}")
    pooled = sorted(glob.glob(os.path.join(a.new_dir, "POOLED_ln_*.json")))
    if pooled:
        pj = pooled[0]; pd = pj[:-5] + "_fdraws.npz"
        P = json.load(open(pj)); Pref = json.load(open(os.path.join(a.ref_dir, "POOLED_ln_real_v2_20260821.json")))
        rec["pooled"] = dict(json=pj, draws_sha_new=sha256(pd), draws_sha_frozen=FROZEN_POOLED_DRAWS,
                             draws_identical=sha256(pd) == FROZEN_POOLED_DRAWS,
                             thresholds_identical=P["thresholds"] == Pref["thresholds"],
                             reporting_bins_identical=P["reporting_bins"] == Pref["reporting_bins"],
                             perz_identical=P.get("perz_paper1") == Pref.get("perz_paper1"),
                             selection_included_identical=[(x["seed"], x["deep"]) for x in P["selection"]["included"]] ==
                                                          [(x["seed"], x["deep"]) for x in Pref["selection"]["included"]],
                             n_draws=P["n_draws"], n_draws_frozen=Pref["n_draws"])
        ok &= rec["pooled"]["draws_identical"] and rec["pooled"]["thresholds_identical"] and rec["pooled"]["selection_included_identical"]
        print("POOLED:", {k: v for k, v in rec["pooled"].items() if k not in ("json",)})
    else:
        rec["pooled"] = dict(error="no pooled artifact yet"); ok = False
    rec["n_runs_compared"] = len(runs)
    rec["verdict"] = "R0 PASS (bit-identical chains and pooled posterior)" if ok else "R0 FAIL — STOP"
    print("R0_BITREPRO", rec["verdict"])
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
