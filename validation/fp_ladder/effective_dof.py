"""Nominal and EFFECTIVE degrees of freedom of the absorber-side ladder runs (PI ruling 2026-09-13b §18 item 9).

VALIDATION-ONLY read-out of existing chain draws; no sampler runs, no file of record is modified.

Effective complexity is the prior-whitened Spiegelhalter-style measure already used in
FP_PARAMETERIZATION_PAPER_JUSTIFICATION.md §C:

    p_eff = sum_i (1 - Var_post[u_i]),   u_i = the site whitened to a standard-normal prior,

so a direction the data do not inform contributes ~0 (posterior variance = prior variance) and a direction the
data pin contributes ~1.  Whitening per site of `model_cc_ladder`:
    eps_N, eps_z, t          ~ N(0,1) a priori             -> u = x
    psi_c                    ~ N(0, sigma_hat_cs)          -> u = x / sigma_hat
    theta_level, theta_slope ~ N(0, s)                     -> u = x / s
    sigma_N, sigma_z         ~ HalfNormal(s)               -> u = Phi^{-1}(erf(x / (s sqrt2)))  (probability integral transform)
Under a FIXED completeness (2-D C_fixed) psi_c is sampled but INERT (the fold does not read it), so its
p_eff must come out ~0; this is reported as a self-consistency check, and its nominal count is listed as
"sampled-inert", not as an active parameter.
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np
from scipy.special import erf, ndtri

HYPER = {"sigma_N": ("halfnormal", 0.5), "sigma_z": ("halfnormal", 0.5),
         "theta_level": ("normal", 4.0), "theta_slope": ("normal", 2.0)}


def whiten(name, x, sigma_hat=None, t_sd=1.0):
    x = np.asarray(x, float)
    if name in ("eps_N", "eps_z"):
        return x
    if name == "t":
        return x / float(t_sd)
    if name == "psi_c":
        return x / np.asarray(sigma_hat, float)
    kind, s = HYPER[name]
    if kind == "normal":
        return x / s
    p = erf(x / (s * math.sqrt(2.0)))
    return ndtri(np.clip(p, 1e-12, 1 - 1e-12))


def p_eff_block(u):
    """u: (draws, ...) whitened draws pooled over chains. Returns (p_eff, n_sites, n_informed_2x, n_informed_4x)."""
    v = np.var(u.reshape(u.shape[0], -1), axis=0, ddof=1)
    return float(np.sum(1.0 - v)), int(v.size), int(np.sum(v < 0.5)), int(np.sum(v < 0.25))


def analyse_bychain(path, sigma_hat, t_sd=1.0, c_fixed=False, ladder="ORACLE"):
    d = np.load(path, allow_pickle=True)
    out = {"file": os.path.basename(path), "blocks": {}}
    tot_eff = 0.0; tot_nom_active = 0
    for name in ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z", "psi_c", "t"):
        if name not in d.files:
            continue
        x = np.asarray(d[name], float)
        x = x.reshape((-1,) + x.shape[2:])                                   # pool chains
        if name == "t" and ladder == "ORACLE":
            out["blocks"][name] = {"nominal": 0, "status": "deterministic-zero (ORACLE)", "p_eff": 0.0}
            continue
        u = whiten(name, x, sigma_hat=sigma_hat, t_sd=t_sd)
        pe, n, n2, n4 = p_eff_block(u)
        status = "active"
        if name == "psi_c" and c_fixed:
            status = "sampled-INERT (fixed 2-D completeness; fold does not read psi_c)"
        out["blocks"][name] = {"nominal": n, "status": status, "p_eff": round(pe, 2),
                               "informed_2x": n2, "informed_4x": n4}
        if status == "active":
            tot_nom_active += n; tot_eff += pe
        else:
            out["blocks"][name]["p_eff_inert_check"] = round(pe, 2)
    out["nominal_active_total"] = tot_nom_active
    out["p_eff_total"] = round(tot_eff, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--pack-dir", required=True, help="dir with scanpack_<fam>_b300_A0.npz (for sigma_hat)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    sig = {}
    for fam in ("2lpt0", "london0", "saclay0"):
        pk = load_pack(os.path.join(a.pack_dir, f"scanpack_{fam}_b300_A0.npz"))
        consts, _ = build_cc_tensors(pk)
        sig[fam] = np.asarray(consts.sigma_hat, float)
    rows = []
    for js in sorted(glob.glob(os.path.join(a.runs, "*", "RUN_*.json"))):
        j = json.load(open(js)); g = j["diagnostics"]
        fam = next(f for f in sig if f"_{f}_" in os.path.basename(js))
        variant = os.path.basename(os.path.dirname(js))
        bc = js.replace(".json", "_bychain.npz")
        if not os.path.exists(bc):
            continue
        c_fixed = bool(g.get("fixed_files", {}).get("c"))
        r = analyse_bychain(bc, sig[fam], t_sd=float(g.get("t_sd", 1.0)), c_fixed=c_fixed, ladder=g["ladder"])
        r.update(variant=variant, family=fam, ladder=g["ladder"], nominal_fp_dof=int(g.get("nominal_fp_dof", 0)),
                 seed=j["run_config"]["seed"], lam_cut=g.get("lam_cut"))
        rows.append(r)
    json.dump(rows, open(a.out_json, "w"), indent=1)
    # ---- markdown: per variant, mean over runs -------------------------------------------------------
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    L = ["# Nominal and effective DOF per variant (PI 2026-09-13b §18 item 9)", "",
         "VALIDATION-ONLY read-out of the existing chain draws (`validation/fp_ladder/effective_dof.py`). p_eff = Σ_i (1 − Var_post u_i) on prior-whitened sites (the §C measure of FP_PARAMETERIZATION_PAPER_JUSTIFICATION.md). 'informed 2×' = sites whose posterior variance is below half the prior variance. Values are means over the runs of each variant (families × seeds); the per-run table is in the JSON.", "",
         "| variant | ladder | runs | nominal ACTIVE sites (hyper 4 + eps_N 14 + eps_z 224 [+ psi_c 96 when molly] [+ t 3 under M1CUT]) | nominal FP DOF | p_eff total | p_eff: hyper / eps_N / eps_z / psi_c / t | informed 2× (eps_z, psi_c) | psi_c inert check (p_eff, should be ≈0 when fixed C) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v, rs in by.items():
        def m(key, blk):
            vals = [r["blocks"][blk].get(key) for r in rs if blk in r["blocks"] and r["blocks"][blk].get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        hyper = sum(m("p_eff", b) for b in ("sigma_N", "sigma_z", "theta_level", "theta_slope"))
        psi_active = all(r["blocks"]["psi_c"]["status"] == "active" for r in rs)
        psi_pe = m("p_eff", "psi_c")
        inert = "n/a (active)" if psi_active else f"{m('p_eff_inert_check', 'psi_c'):+.2f}"
        t_pe = m("p_eff", "t")
        L.append(f"| **{v}** | {rs[0]['ladder']} | {len(rs)} | {rs[0]['nominal_active_total']} | {rs[0]['nominal_fp_dof']} | "
                 f"{np.mean([r['p_eff_total'] for r in rs]):.1f} | {hyper:.1f} / {m('p_eff','eps_N'):.1f} / {m('p_eff','eps_z'):.1f} / "
                 f"{(psi_pe if psi_active else 0.0):.1f} / {t_pe:.2f} | {m('informed_2x','eps_z'):.0f}, {(m('informed_2x','psi_c') if psi_active else 0):.0f} | {inert} |")
    L += ["", "Reading: the latent population block (242 nominal) is informed in only ≈ p_eff directions; a fixed completeness object removes the 96 ψ_c sites from the ACTIVE count entirely (they remain in the trace as inert prior draws, which is what the inert check measures), and M1CUT adds exactly the three t_K sites. No variant adds a fitted parameter to the survey fold: R1x/C1x objects are fixed calibration objects whose calibration-side DOF are listed in CALIBRATION_SIDE_VALIDATION_REPORT.md §G (response 84–108; completeness 6–96)."]
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
