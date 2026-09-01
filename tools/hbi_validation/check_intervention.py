#!/usr/bin/env python3
"""check_intervention.py — validate that a sensitivity run of the 2026-09-02 matrix changed ONLY its intended block
(kickoff STOP 5) and passed the convergence rule: from the run's all-site draws, (i) the fixed sites are exactly zero
and no longer sampled, (ii) every other site is present, (iii) per-arm estimand split-R̂ / divergences / G_A as in the
pooled selection, (iv) the pooled selection block. Fails closed.

    python tools/hbi_validation/check_intervention.py --run-dir ROOT/R1 --pooled ROOT/R1/POOLED_ln_R1.json --expect-fixed t [--expect-fixed psi_c] --out check_R1.json
"""
import argparse, glob, json, os, sys
import numpy as np
SITES = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z", "psi_c", "fp_lam_total", "fp_shape_v", "t")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--pooled", required=True)
    ap.add_argument("--expect-fixed", action="append", default=[]); ap.add_argument("--expect-t-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ok = True; rep = {"runs": {}, "expect_fixed": a.expect_fixed}
    for p in sorted(glob.glob(os.path.join(a.run_dir, "REAL_ln_*s2026????.json"))):
        d = json.load(open(p)); g = d["diagnostics"]; z = np.load(p[:-5] + "_allsites.npz")
        flags = g.get("validation_flags", {})
        r = dict(fp_mode=g["fp_mode"], t_scale=g["t_scale"], flags=flags, divergences=g["divergences"], split_rhat=g["split_rhat"],
                 G_A=d["guards"]["G_A_real_mode"]["status"], pe=g["mean_potential_energy_per_chain"], t_post_mean=g["t_post_mean"])
        r["fixed_ok"] = {}
        for s in a.expect_fixed:
            v = np.asarray(z[s]); r["fixed_ok"][s] = bool(np.all(v == 0.0)); ok &= r["fixed_ok"][s]
        r["sites_present"] = [s for s in SITES if s in z.files]; r["sites_missing"] = [s for s in SITES if s not in z.files]
        ok &= not r["sites_missing"]
        r["mode_ok"] = (g["fp_mode"] == "informative_ln") and abs(g["t_scale"] - a.expect_t_scale) < 1e-12; ok &= r["mode_ok"]
        # flags consistent with expectation
        r["flags_ok"] = (bool(flags.get("fix_t")) == ("t" in a.expect_fixed)) and (bool(flags.get("fix_psi_c")) == ("psi_c" in a.expect_fixed)); ok &= r["flags_ok"]
        # non-fixed sampled sites really vary
        for s in SITES:
            if s not in a.expect_fixed and s in z.files:
                if float(np.asarray(z[s]).std()) == 0.0:
                    r.setdefault("unexpectedly_constant", []).append(s); ok = False
        rep["runs"][os.path.basename(p)] = r
        print(f"{os.path.basename(p):28s} div={g['divergences']} rhat={g['split_rhat']} G_A={r['G_A']} fixed={r['fixed_ok']} flags_ok={r['flags_ok']} t_mean={[round(x,3) for x in g['t_post_mean']]}")
    P = json.load(open(a.pooled)); rep["pooled"] = dict(included=[(x["seed"], x["deep"]) for x in P["selection"]["included"]], excluded=[(x["seed"], x["deep"], x["reason"]) for x in P["selection"]["excluded"]], n_draws=P["n_draws"])
    rep["n_included"] = len(rep["pooled"]["included"]); rep["verdict"] = "INTERVENTION ISOLATED + POOLED" if ok and rep["n_included"] >= 4 else ("INTERVENTION ISOLATED but < 4 arms pooled" if ok else "STOP — intervention not isolated")
    print("pooled:", rep["pooled"]); print("VERDICT:", rep["verdict"])
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
