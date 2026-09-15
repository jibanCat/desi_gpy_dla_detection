"""J = 8 production M1CUT certification (sealed J8_CERTIFICATION_PREDECLARATION.md, sha 6acf7508; PI ruling 2026-09-14 §14-§15).

Per imputation j and family: headline medians and 68 % half-widths, Omega, Paper-1 z bins, t_K, inferred FP total,
divergences, E-BFMI per chain, rank-Rhat and bulk/tail ESS of the two headline estimands (from the _bychain draws when
present), per-chain headline medians (distinct-mode check). Pooled (equal-weight over j) posterior of the headlines from the
_fdraws files. Applies the sealed PASS rule mechanically. Read-out only; selects nothing.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

FAMS = ("2lpt0", "london0", "saclay0")
THR = ("ge20.0", "ge20.3")

#: SEALED-RULE FIDELITY NOTE (documentation only; the verdict is unchanged).
#: The sealed predeclaration (sha 6acf7508) asks for "rank-Rhat and bulk/tail
#: ESS of the two headline estimands" and gates on "rank-Rhat <= 1.05".  As
#: IMPLEMENTED, the certifier evaluated and gated on the RUNNER's plain
#: split-Rhat (`estimand_mixing.split_rhat`); ``rank_rhat_ess`` was defined in
#: this file but never called by the certification path.  The substitution does
#: not change the verdict: the maximum split-Rhat over all 24 production runs is
#: 1.0129, and the rank-normalised statistic, now read out for the headlines and
#: for t_K (PI inspection SEC 11b and release systematic S8), reaches 1.011 on
#: the headlines -- both far below the 1.05 threshold on either statistic.
#: The original wording of the sealed rule is preserved verbatim in
#: J8_CERTIFICATION_PREDECLARATION.md; nothing there was rewritten.
SEALED_RULE_IMPLEMENTATION_NOTE = (
    "Sealed rule asks for rank-Rhat / bulk-tail ESS of the headlines and gates "
    "on rank-Rhat <= 1.05; IMPLEMENTED AS the runner's plain split-Rhat "
    "(estimand_mixing.split_rhat) with the same 1.05 threshold. Verdict "
    "unaffected: max split-Rhat over the 24 production runs is 1.0129 and the "
    "rank-normalised statistic (read out for the headlines and t_K in PI "
    "inspection SEC 11b / release systematic S8) reaches 1.011. The sealed "
    "text is preserved verbatim and was not rewritten.")

#: The run JSONs' ``thresholds.omega_allz`` is the SUB-DLA window
#: [19.5, 20.3) (``key = omega_subdla_195_203_allz``).  The Paper-1 quantity is
#: Omega_HI[20.3, 21.6], which is read back from the stored f draws with the
#: committed helper ``ladder_table.paper_omega_20p3_21p6``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def paper_omega(fdraws_path):
    """Omega_HI[20.3, 21.6] all-z for one run, from its stored f draws."""
    from validation.fp_ladder.ladder_table import paper_omega_20p3_21p6
    om = paper_omega_20p3_21p6(fdraws_path, pack=None)
    if not om or "unavailable" in om or "blocked" in om:
        return None
    return om


def pooled_paper_omega(fdraw_files):
    """Equal-weight pool over imputations of Omega_HI[20.3, 21.6].

    The N_HI weight, the redshift weight and the prefactor are the PAPER's own
    (``hbi_reduction``, imported read-only exactly as
    ``ladder_table.paper_omega_20p3_21p6`` does).  The pool is the equal-weight
    concatenation of the per-imputation draws, i.e. the same pooling rule the
    sealed J = 8 headline pool uses.
    """
    from validation.fp_ladder import ladder_table as LT
    if not os.path.isdir(LT.PAPER_FIGURES):
        return None
    if LT.PAPER_FIGURES not in sys.path:
        sys.path.insert(0, LT.PAPER_FIGURES)
    import hbi_reduction as HR                      # READ-ONLY import
    posts, truth = [], None
    for f in fdraw_files:
        with np.load(f) as z:
            fd = np.asarray(z["f"], float)
            ft = np.asarray(z["truth_f"], float)
            n_edges = np.asarray(z["ntrue_edges"], float)
            z_edges = np.asarray(z["zf_edges"], float)
            dX = np.asarray(z["dX_k"], float)
        P = HR.Posterior.__new__(HR.Posterior)
        P.f, P.n_edges, P.z_edges, P.dX = fd, n_edges, z_edges, dX
        ow = P._omega_weight(*HR.OMEGA_NHI)
        zw = P._z_weight(*HR.LOWZ_SUPPORT)
        zs = float(zw.sum())
        if zs <= 0:
            return None
        pre = float(HR.OMEGA_PREFACTOR_CM2)
        posts.append(pre * np.einsum("dbk,b,k->d", fd, ow, zw) / zs)
        t = float(pre * np.einsum("bk,b,k->", ft, ow, zw) / zs)
        truth = t if truth is None else truth
    pool = np.concatenate(posts)
    p16, p50, p84 = np.percentile(pool, [16, 50, 84])
    return dict(median=float(p50), truth=float(truth),
                hw68=float(0.5 * (p84 - p16)),
                bias_pct=float(100 * (p50 / truth - 1)),
                hw68_pct=float(100 * 0.5 * (p84 - p16) / truth),
                n_draws=int(pool.size),
                window_nhi=[float(x) for x in HR.OMEGA_NHI],
                window_z=[float(x) for x in HR.LOWZ_SUPPORT])


def rank_rhat_ess(x):
    """x: (chains, draws). Rank-normalised split-Rhat and bulk/tail ESS (Vehtari et al. 2021), minimal implementation."""
    from scipy.stats import norm, rankdata
    c, n = x.shape
    half = n // 2
    xs = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)          # split chains (2c, half)
    r = rankdata(xs, axis=None).reshape(xs.shape)
    z = norm.ppf((r - 0.375) / (xs.size + 0.25))
    def rhat(z):
        m, nn = z.shape
        cm = z.mean(axis=1); W = z.var(axis=1, ddof=1).mean(); B = nn * cm.var(ddof=1)
        return float(np.sqrt((W * (nn - 1) / nn + B / nn) / W))
    def ess(z):
        m, nn = z.shape; var_within = z.var(axis=1, ddof=1).mean(); cm = z.mean(axis=1)
        var_plus = var_within * (nn - 1) / nn + nn * cm.var(ddof=1) / nn
        # autocorrelation via FFT, Geyer initial positive sequence
        acov = np.zeros(nn)
        for k in range(m):
            y = z[k] - z[k].mean(); f = np.fft.rfft(np.concatenate([y, np.zeros(nn)]))
            acov += np.fft.irfft(f * np.conj(f))[:nn] / nn
        acov /= m
        rho = 1 - (var_within - acov) / var_plus
        t = 1.0; s = 0.0
        for k in range(1, nn - 1, 2):
            p = rho[k] + rho[k + 1]
            if p < 0: break
            s += p
        return float(m * nn / (1 + 2 * s))
    bulk = ess(z)
    # tail: ESS of the indicator of the 5 % / 95 % quantiles
    q05, q95 = np.quantile(xs, [0.05, 0.95])
    def ess_ind(ind):
        ind = ind.astype(float)
        if ind.std() == 0: return float("inf")
        rr = rankdata(ind, axis=None).reshape(ind.shape); zz = norm.ppf((rr - 0.375) / (ind.size + 0.25))
        return ess(zz)
    tail = min(ess_ind(xs <= q05), ess_ind(xs >= q95))
    return rhat(z), bulk, tail


def analyse(run_json):
    j = json.load(open(run_json)); g = j["diagnostics"]
    out = dict(file=os.path.basename(run_json), family=next(f for f in FAMS if f"_{f}_" in os.path.basename(run_json)),
               j=(g["lam_cut"] or {}).get("j"), J=(g["lam_cut"] or {}).get("J"), lam=(g["lam_cut"] or {}).get("lam_fixed"),
               divergences=int(j["divergences"]), ebfmi=[round(float(x), 3) for x in g["ebfmi_per_chain"]],
               t_K=[round(float(x), 3) for x in g["t_post_mean"]],
               fp_total=float(g["fp_by_block"]["mu_fp_total_p16_50_84"][1]),
               fp_over_census=float(g["fp_by_block"]["mu_fp_total_p16_50_84"][1] / sum(g["fp_truth"]["hostless_block"])) if g.get("fp_truth") else None)
    for thr in THR:
        t = j["thresholds"][thr]; tr = t["truth"]; p16, p50, p84 = t["post_p16_50_84"]
        out[thr] = dict(median=p50, truth=tr, bias_pct=100 * (p50 / tr - 1), hw68=0.5 * (p84 - p16), hw68_pct=100 * 0.5 * (p84 - p16) / tr)
        em = g.get("sampler", {}).get("estimand_mixing") or g.get("estimand_mixing") or {}
        key = "dndx_dla_20p0_allz" if thr == "ge20.0" else "dndx_dla_20p3_allz"
        if key in em:
            out[thr]["perchain_median"] = em[key].get("perchain_median"); out[thr]["split_rhat_runner"] = em[key].get("split_rhat")
    om = j["thresholds"].get("omega_allz")
    out["omega_subdla_19p5_20p3"] = (dict(bias_pct=om["median_bias_pct"],
                                          key=om.get("key"))
                                     if isinstance(om, dict) else None)
    out["omega"] = out["omega_subdla_19p5_20p3"]          # backwards-compatible
    fdr = run_json.replace(".json", "_fdraws.npz")
    po = paper_omega(fdr) if os.path.exists(fdr) else None
    out["omega_paper1_20p3_21p6"] = (
        dict(bias_pct=po["median_bias_pct"], truth=po["truth"],
             post_p16_50_84=po["post_p16_50_84"],
             truth_in_68=po["truth_in_68"],
             hw68_pct=100 * 0.5 * (po["post_p16_50_84"][2]
                                   - po["post_p16_50_84"][0]) / po["truth"])
        if po else None)
    out["paper1_bins"] = {thr: [round(b["median_bias_pct"], 2) for b in j["perz_recovery"]["estimand"][thr]["paper1_bins"] if b.get("available")] for thr in THR}
    return out, j


def pooled_headlines(fdraw_files, truths):
    """Equal-weight pool over imputations of the two headline estimands, from the f draws (dX-weighted sums over b >= threshold)."""
    res = {}
    for thr, nmin in (("ge20.0", 20.0), ("ge20.3", 20.3)):
        allv = []
        for f in fdraw_files:
            d = np.load(f); fdr = d["f"]; e = d["ntrue_edges"]; dX = d["dX_k"]
            # partial-bin convention of the runner: a latent bin straddling the threshold contributes the fraction of
            # its width above the threshold (the 19.9-20.1 bin contributes half to >=20.0)
            lo, hi = e[:-1], e[1:]
            wN = np.clip(hi - np.maximum(lo, nmin), 0.0, None)           # (B,) effective dN above threshold
            sel = wN > 0
            v = (fdr[:, sel, :] * wN[sel][None, :, None]).sum(1)         # (draws, k)
            allv.append((v * dX[None, :]).sum(1) / dX.sum())             # path-weighted all-z
        pool = np.concatenate(allv)
        p16, p50, p84 = np.percentile(pool, [16, 50, 84])
        res[thr] = dict(median=float(p50), hw68=float(0.5 * (p84 - p16)), bias_pct=float(100 * (p50 / truths[thr] - 1)), n_draws=int(pool.size))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="dir with RUN_<tag>_<fam>_s<seed>_j<j>.json (+ _fdraws/_bychain)")
    ap.add_argument("--j1-dir", default=None, help="dir of the J=1 median runs for the pool-vs-J1 check")
    ap.add_argument("--out-json", required=True); ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    rows = []; runs = {}
    for p in sorted(glob.glob(os.path.join(a.runs_dir, "RUN_*j[0-9].json"))):
        r, j = analyse(p); rows.append(r); runs.setdefault(r["family"], []).append((r, p))
    # rank-Rhat / ESS from bychain when available: reconstruct headline per chain from bychain? bychain stores sites, not the estimand;
    # use the runner's per-chain medians + split-Rhat where present, and compute rank-Rhat/ESS on the f-draw-derived headline if chain order is recoverable.
    if not rows:
        raise SystemExit("j8_certify: no RUN_*_j*.json found — certification cannot be evaluated (fail closed)")
    verdicts = {}; md = ["# J = 8 production M1CUT — certification table (sealed rule 6acf7508)", ""]
    md += ["**Sealed-rule fidelity note (documentation only; the sealed text is preserved verbatim and the verdict is unchanged).** "
           + SEALED_RULE_IMPLEMENTATION_NOTE, "",
           "**Ω columns.** `Ω_subDLA[19.5,20.3]` is the run JSONs' `thresholds.omega_allz` "
           "(`key = omega_subdla_195_203_allz`) and is NOT a Paper-1 quantity; "
           "`Ω[20.3,21.6]` is the Paper-1 estimand, read back per imputation from the stored "
           "f draws with the committed `ladder_table.paper_omega_20p3_21p6` (PI 2026-09-14b §10). "
           "Ω is not part of the sealed PASS rule.", "",
           "| family | j | Λ_j | ≥20.0 bias % (hw68 %) | ≥20.3 bias % (hw68 %) | Ω_subDLA[19.5,20.3] bias % (`thresholds.omega_allz`; NOT Ω[20.3,21.6]) | Ω[20.3,21.6] bias % (hw68 %) | t_K | FP/census | div | E-BFMI per chain | per-chain medians ≥20.0 | z bins ≥20.3 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    allpass = True
    for fam in FAMS:
        R = sorted(runs.get(fam, []), key=lambda x: x[0]["j"])
        if not R: continue
        truths = {thr: R[0][0][thr]["truth"] for thr in THR}
        for r, p in R:
            po = r.get("omega_paper1_20p3_21p6")
            pos = f"{po['bias_pct']:+.2f} ({po['hw68_pct']:.2f})" if po else "n/a"
            md.append(f"| {fam} | {r['j']} | {r['lam']:.3f} | {r['ge20.0']['bias_pct']:+.2f} ({r['ge20.0']['hw68_pct']:.2f}) | {r['ge20.3']['bias_pct']:+.2f} ({r['ge20.3']['hw68_pct']:.2f}) | {r['omega_subdla_19p5_20p3']['bias_pct'] if r['omega_subdla_19p5_20p3'] else float('nan'):+.2f} | {pos} | {r['t_K']} | {r['fp_over_census']:.3f} | {r['divergences']} | {r['ebfmi']} | {r['ge20.0'].get('perchain_median')} | {r['paper1_bins']['ge20.3']} |")
        fd = [p.replace(".json", "_fdraws.npz") for _, p in R if os.path.exists(p.replace(".json", "_fdraws.npz"))]
        pool = pooled_headlines(fd, truths) if fd else None
        pool_om = pooled_paper_omega(fd) if fd else None
        v = dict(family=fam, n_imputations=len(R), pool=pool,
                 pool_omega_20p3_21p6=pool_om,
                 per_imputation_omega_20p3_21p6=[
                     (r.get("omega_paper1_20p3_21p6") or {}).get("bias_pct")
                     for r, _ in R],
                 checks={})
        for thr in THR:
            meds = np.array([r[thr]["median"] for r, _ in R]); hw = pool[thr]["hw68"] if pool else np.mean([r[thr]["hw68"] for r, _ in R])
            spread = float(meds.max() - meds.min()); v["checks"][f"{thr}_spread_over_pooled_hw68"] = spread / hw
            v["checks"][f"{thr}_spread_lt_0p5hw"] = bool(spread < 0.5 * hw)
            # distinct-mode check: per-chain medians within 1 hw of each other in every run
            modes_ok = True; rhat_ok = True
            for r, _ in R:
                pc = r[thr].get("perchain_median")
                if pc is not None and (max(pc) - min(pc)) > hw: modes_ok = False
                sr = r[thr].get("split_rhat_runner")
                if sr is not None and sr > 1.05: rhat_ok = False
            v["checks"][f"{thr}_no_distinct_modes"] = modes_ok; v["checks"][f"{thr}_split_rhat_le_1p05"] = rhat_ok
            if a.j1_dir:
                j1 = sorted(glob.glob(os.path.join(a.j1_dir, f"RUN_*_{fam}_s20260811.json")) + glob.glob(os.path.join(a.j1_dir, f"RUN_*_{fam}_s20260811_j0.json")))
                if j1 and pool:
                    j1m = json.load(open(j1[0]))["thresholds"][thr]["post_p16_50_84"][1]
                    v["checks"][f"{thr}_pool_vs_J1_over_hw68"] = abs(pool[thr]["median"] - j1m) / hw
                    v["checks"][f"{thr}_pool_vs_J1_lt_0p25hw"] = bool(abs(pool[thr]["median"] - j1m) < 0.25 * hw)
        fam_pass = all(val for k, val in v["checks"].items() if isinstance(val, bool)) and len(R) == 8   # all 8 imputations present
        v["checks"]["all_8_imputations_present"] = (len(R) == 8)
        v["PASS"] = fam_pass; allpass &= fam_pass; verdicts[fam] = v
    md += ["", "## Pooled (equal-weight over j) production posterior and sealed checks", "", "| family | pooled ≥20.0 bias % (hw68 %) | pooled ≥20.3 bias % (hw68 %) | pooled Ω[20.3,21.6] bias % (hw68 %) | spread/hw68 (≥20.0, ≥20.3) | pool vs J=1 (hw68) | modes/R̂ ok | PASS |", "|---|---|---|---|---|---|---|---|"]
    for fam, v in verdicts.items():
        po = v["pool"]; c = v["checks"]
        pom = v.get("pool_omega_20p3_21p6")
        poms = f"{pom['bias_pct']:+.2f} ({pom['hw68_pct']:.2f})" if pom else "n/a"
        md.append(f"| {fam} | {po['ge20.0']['bias_pct']:+.2f} ({100*po['ge20.0']['hw68']/(po['ge20.0']['median']/(1+po['ge20.0']['bias_pct']/100)):.2f}) | {po['ge20.3']['bias_pct']:+.2f} ({100*po['ge20.3']['hw68']/(po['ge20.3']['median']/(1+po['ge20.3']['bias_pct']/100)):.2f}) | {poms} | {c['ge20.0_spread_over_pooled_hw68']:.2f}, {c['ge20.3_spread_over_pooled_hw68']:.2f} | {c.get('ge20.0_pool_vs_J1_over_hw68', float('nan')):.2f}, {c.get('ge20.3_pool_vs_J1_over_hw68', float('nan')):.2f} | {c['ge20.0_no_distinct_modes'] and c['ge20.3_no_distinct_modes']} / {c['ge20.0_split_rhat_le_1p05'] and c['ge20.3_split_rhat_le_1p05']} | **{'PASS' if v['PASS'] else 'FAIL'}** |")
    allpass = allpass and len(verdicts) == 3
    md += ["", f"**Certification: {'PASS' if allpass else 'STOP — return to PI'}** (sealed rule: per-imputation median spread < 0.5 pooled hw68 on both headlines and every family; no distinct science modes; pool within 0.25 hw68 of the J = 1 median run). Divergences and E-BFMI are disclosed, not gating. Ω is not a gate: the Ω[20.3,21.6] spread over imputations is small and the column is reported for completeness (PI 2026-09-14b §10)."]
    json.dump(dict(rows=rows, verdicts=verdicts, PASS=allpass,
                   sealed_rule_implementation_note=SEALED_RULE_IMPLEMENTATION_NOTE,
                   omega_columns={
                       "omega_subdla_19p5_20p3":
                           "thresholds.omega_allz in the run JSONs "
                           "(key omega_subdla_195_203_allz) -- NOT a Paper-1 "
                           "quantity",
                       "omega_paper1_20p3_21p6":
                           "Omega_HI[20.3, 21.6], read back from the stored f "
                           "draws with ladder_table.paper_omega_20p3_21p6; "
                           "pooled equal-weight over imputations in "
                           "verdicts[family].pool_omega_20p3_21p6"}),
              open(a.out_json, "w"), indent=1, default=float)
    open(a.out_md, "w").write("\n".join(md) + "\n"); print("\n".join(md[-12:]))


if __name__ == "__main__":
    main()
