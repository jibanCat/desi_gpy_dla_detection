#!/usr/bin/env python
"""run_ladder.py — VALIDATION-ONLY runner for the FP-model ladder on MOCK packs.

Sealed predeclaration: notes/governance/FP_REGULARIZATION_MODEL_LADDER_PREDECLARATION.md
(sha256 3112022a…). MOCK-ONLY: refuses a pack without a nonzero truth_counts (the
real pack carries the all-zero sentinel). Output JSON schema = cc_posterior_validation's
(thresholds / reporting_bins / perz_recovery / diagnostics.estimand_mixing), so
perz_gate.gate_one runs UNCHANGED, plus additive blocks:
  diagnostics.calibration_predictive   (loa-0 deviance p-value; P(0 events at Nhat>=20.2))
  diagnostics.fp_by_block              (posterior median mu_FP per coarse block and Nhat group)
  diagnostics.ebfmi_per_chain, divergences_per_chain, t_post_mean, e_t, lam_over_naive
and saves <out>_fdraws.npz (f, truth_f, grids) and <out>_bychain.npz (all sampled sites,
potential energy, energy, diverging).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import numpyro
from scipy.stats import poisson

from CDDF_analysis.hbi_mcmc.provenance_util import run_config
from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import (build_cc_tensors, perz_recovery)
from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder, FP_SITES, live_mask, NOMINAL_FP_DOF
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f

POP_SITES = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z", "psi_c")
NHAT_GROUPS = (("19.5_20.0", 19.5, 20.0), ("20.0_20.3", 20.0, 20.3), ("20.3_22.4", 20.3, 22.4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ladder", required=True, choices=list(FP_SITES),
                    help="M0..M5 = sealed ladder; ORACLE = DIAGNOSTIC (mu_FP pinned to the mock FP-truth census; needs --census)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--t-sd", type=float, default=1.0)
    ap.add_argument("--tau-scale", type=float, default=0.5)
    ap.add_argument("--calib-weight", type=float, default=1.0)
    ap.add_argument("--census", default=None, help="fp_census_<fam>.npz (hostless@17.2) for the FP-truth soft flag")
    ap.add_argument("--stage", default="")
    ap.add_argument("--mg-fixed-file", default=None, help="LADDER: npz with Mg (S,Kf,C,B) = a FIXED response calibration variant")
    ap.add_argument("--mg-fixed-key", default="Mg", help="key inside --mg-fixed-file: Mg (record: Q x 2LPT-measured phi), Mg_phi_smooth (phi sensitivity), Mg_phi_family (ORACLE DIAGNOSTIC: Q x the family's own measured phi)")
    ap.add_argument("--c-fixed-file", default=None, help="LADDER: npz with C_fixed (S,B) or C_fixed_bks (B,Kf,S) = a FIXED completeness variant")
    ap.add_argument("--extra-fixed-file", default=None, help="LADDER: npz with mu_extra (C,Kf,S) = the FIXED sub-floor-host term (A0)")
    ap.add_argument("--lam-imputations", type=int, default=1, help="M1CUT: number J of stratified imputations of Lambda from p(Lambda|D_loa0)")
    ap.add_argument("--fp-a0", type=float, default=None, help="M1CUT: Perks pseudo-count a0 of the loa-0 template (default = record 1/K); PI 2026-09-13d §12 sensitivity battery")
    ap.add_argument("--lam-imputation", type=int, default=0, help="M1CUT: which imputation j (0..J-1) this run uses")
    ap.add_argument("--require-support", action="store_true",
                    help="LADDER (PI ruling 2026-09-13b §3): fail closed unless pack / census / ops carry stamped, "
                         "identical supports at the 'row' level (truth_host_floor reported alongside)")
    ap.add_argument("--ops", default=None, help="DIAGNOSTIC: empirical_ops_<fam>.npz (matched-truth operators)")
    ap.add_argument("--fix", default="", help="DIAGNOSTIC: comma list of fixed components from --ops/--census: "
                    "P (P6b sub-floor-host term), C (C_true[b,K,s]), Cz (C_true[b,s], z-free), M (M_true[s,K,c,b]), E (E_true[c,K,s,b] full transfer)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    numpyro.set_host_device_count(a.chains)

    support_record = None
    if a.require_support:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "absorber_ladder", "support"))
        import support_contract as SC
        row_fields = tuple(f for f in SC.SUPPORT_FIELDS if f != "truth_host_floor")
        support_record = SC.check_support_consistency(a.pack, a.census, a.ops, fields=row_fields)   # raises on mismatch
        print("SUPPORT GATE PASS (row level):", support_record["support_id_short"], "host floors:", support_record["truth_host_floor"])
    pk = load_pack(a.pack)
    tc = np.asarray(pk.truth_counts)
    if tc.size == 0 or tc.sum() <= 0:
        raise SystemExit("MOCK GATE: pack carries no truth_counts — the ladder is MOCK-ONLY; refusing")
    consts, Mg = build_cc_tensors(pk)
    counts = jnp.asarray(np.asarray(pk.counts, float))
    fpc = jnp.asarray(np.asarray(pk.fp_counts, float))

    mu_fixed = None
    if a.ladder == "ORACLE":
        if not (a.census and os.path.exists(a.census)):
            raise SystemExit("ORACLE diagnostic requires --census")
        cz = np.load(a.census, allow_pickle=True)
        mu_fixed = np.asarray(cz["hostless"], float)            # (C, Kf, S) realised mock FP truth
        if mu_fixed.shape != tuple(np.asarray(pk.counts).shape):
            raise SystemExit(f"census shape {mu_fixed.shape} != counts {np.asarray(pk.counts).shape}")
    fix = [x for x in a.fix.split(",") if x]
    mu_extra = C_fixed = Mg_fixed = E_fixed = None
    lam_fixed = None
    if a.ladder == "M1CUT":
        from CDDF_analysis.hbi_mcmc.fp_ladder import lambda_calibration_posterior_quantiles
        qs = lambda_calibration_posterior_quantiles(pk.fp_counts, consts.fp_ell_eff, a.lam_imputations)
        lam_fixed = float(qs[a.lam_imputation])
    if a.mg_fixed_file:
        _mgf = np.load(a.mg_fixed_file, allow_pickle=True)
        if a.mg_fixed_key == "Mg_phi_family":
            # ORACLE DIAGNOSTIC ONLY (PI 2026-09-13d §9): Q x the FAMILY's own measured phi, built from the
            # stored unit rows; never the formal closure result.
            _kz = np.asarray(consts.kz_to_K, int)                          # the pack's fine-z -> coarse-K map
            _ru = np.asarray(_mgf["rows_unit"], float)                    # (B,S,K,C)
            _ph = np.asarray(_mgf["phi_bsK_family_measured"], float)      # (B,S,K)
            _M = np.einsum("bsKc,bsK->sKcb", _ru, _ph)                     # (S,K,C,B)
            Mg_fixed = _M[:, _kz, :, :]
        else:
            Mg_fixed = np.asarray(_mgf[a.mg_fixed_key], float)
    if a.c_fixed_file:
        cf = np.load(a.c_fixed_file, allow_pickle=True)
        C_fixed = np.asarray(cf["C_fixed_bks"], float) if "C_fixed_bks" in cf.files else np.asarray(cf["C_fixed"], float)
    if a.extra_fixed_file:
        ef = np.load(a.extra_fixed_file, allow_pickle=True)
        key = "mu_extra" if "mu_extra" in ef.files else "mu_P6b_cks"
        mu_extra = (mu_extra if mu_extra is not None else 0.0) + np.asarray(ef[key], float)   # P6b-cal (transported rate)
    kz_np = np.asarray(consts.kz_to_K); KK = consts.n_kk
    if fix:
        # 'P' (the fixed sub-floor-host term from the census, A0) is allowed under any FP member;
        # the truth-pinned absorber-side replacements C/Cz/M/E are DIAGNOSTICS and need ORACLE.
        if a.ladder != "ORACLE" and any(x != "P" for x in fix):
            raise SystemExit("--fix C/Cz/M/E are DIAGNOSTICS and require --ladder ORACLE (FP pinned to truth)")
        if "P" in fix:
            cz = np.load(a.census, allow_pickle=True)
            mu_extra = np.asarray(cz["host_17p2_19p0"], float)      # (C,Kf,S) realised sub-floor-host detections
        need_ops = [x for x in fix if x != "P"]
        if need_ops:
            if not (a.ops and os.path.exists(a.ops)):
                raise SystemExit("--fix C/Cz/M/E requires --ops empirical_ops_<fam>.npz")
            op = np.load(a.ops, allow_pickle=True)
            if "C" in fix:
                Cbk = np.asarray(op["C_true_bKs"], float)              # (B,KK,S)
                C_fixed = Cbk[:, kz_np, :]                             # (B,Kf,S) expanded per fine z
            if "Cz" in fix:
                C_fixed = np.asarray(op["C_true_bs"], float).T         # (S,B) from (B,S)
            if "M" in fix:
                Msk = np.asarray(op["M_true_sKcb"], float)             # (S,KK,C,B)
                Mg_fixed = Msk[:, kz_np, :, :]                         # (S,Kf,C,B)
            if "E" in fix:
                EK = np.asarray(op["E_true_cKsb"], float)              # (C,KK,S,B)
                E_fixed = EK[:, kz_np, :, :]                           # (C,Kf,S,B)
    from numpyro.infer import MCMC, NUTS
    kern = NUTS(model_cc_ladder, target_accept_prob=a.target_accept)
    mcmc = MCMC(kern, num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(a.seed), consts, Mg, counts=counts, fp_counts=fpc,
             ladder=a.ladder, t_sd=a.t_sd, tau_scale=a.tau_scale, calib_weight=a.calib_weight,
             mu_fp_fixed=mu_fixed, mu_extra_fixed=mu_extra, C_fixed=C_fixed, Mg_fixed=Mg_fixed, E_fixed=E_fixed,
             lam_fixed=lam_fixed, fp_a0=a.fp_a0,
             extra_fields=("potential_energy", "energy", "diverging"))
    sam = mcmc.get_samples(group_by_chain=False)
    sam_g = mcmc.get_samples(group_by_chain=True)
    xf_g = mcmc.get_extra_fields(group_by_chain=True)
    f_draws = np.asarray(sam["f"])

    # ---- the committed reduction vs truth (VERBATIM cc_posterior_validation) ----
    ft = np.asarray(truth_f(pk), float)
    red = reduce_f_posterior(f_draws, pk)
    red_t = reduce_f_posterior(ft[None, :, :], pk)
    ntrue = np.asarray(pk.ntrue_edges, float); dN = np.diff(ntrue)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    REDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)
    rep = {}
    for thr, key in ((20.0, "dndx_dla_20p0_allz"), (20.3, "dndx_dla_20p3_allz")):
        dr = np.asarray(red[key]); tv = float(np.asarray(red_t[key])[0])
        q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
        rep[f"ge{thr}"] = dict(truth=tv, post_p16_50_84=[float(x) for x in q[1:4]],
                               post_p2p5_97p5=[float(q[0]), float(q[4])],
                               median_bias_pct=round(100 * (q[2] / tv - 1), 2),
                               truth_in_68=bool(q[1] <= tv <= q[3]), truth_in_95=bool(q[0] <= tv <= q[4]))
    # Omega[20.3,21.6] all-z bias, if the reduction exposes it
    om_key = [k for k in red if k.startswith("omega") and "allz" in k]
    if om_key:
        k = om_key[0]; dr = np.asarray(red[k]); tv = float(np.asarray(red_t[k])[0])
        q = np.percentile(dr, [16, 50, 84])
        rep["omega_allz"] = dict(key=k, truth=tv, post_p16_50_84=[float(x) for x in q],
                                 median_bias_pct=round(100 * (q[1] / tv - 1), 2), truth_in_68=bool(q[0] <= tv <= q[2]))
    binrep = []
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        dr = ((f_draws[:, m, :] * dN[None, m, None]).sum(axis=1) * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        tv = float(((ft[m, :] * dN[m, None]).sum(axis=0) * dX_k).sum() / dX_k.sum())
        q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
        binrep.append(dict(bin=[round(e0, 1), round(e1, 1)], median_bias_pct=round(100 * (q[2] / tv - 1), 2),
                           truth_in_68=bool(q[1] <= tv <= q[3]), truth_in_95=bool(q[0] <= tv <= q[4])))
    perz = perz_recovery(f_draws, ft, pk)

    # ---- diagnostics -----------------------------------------------------------
    div_g = np.asarray(xf_g["diverging"]); pe_g = np.asarray(xf_g["potential_energy"], float)
    en_g = np.asarray(xf_g["energy"], float)
    ebfmi = [float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2)) for e in en_g]
    fg = np.asarray(sam_g["f"])
    mixing = {}
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    for key in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz"):
        cs = np.stack([np.asarray(reduce_f_posterior(fg[ci], pk)[key]) for ci in range(fg.shape[0])])
        W = cs.var(axis=1, ddof=1).mean(); Bv = cs.mean(axis=1).var(ddof=1) * cs.shape[1]
        rh = float(np.sqrt(((cs.shape[1] - 1) / cs.shape[1] * W + Bv / cs.shape[1]) / W)) if cs.shape[0] > 1 else None
        mixing[key] = dict(split_rhat=(round(rh, 4) if rh else None),   # the field perz_gate reads (inherited name)
                           split_rhat_numpyro=round(float(split_gelman_rubin(cs)), 4),
                           ess=round(float(effective_sample_size(cs)), 1),
                           perchain_median=[round(float(np.median(c)), 5) for c in cs])
    lam_draws = np.asarray(sam["lam_fp"])                       # (D, C, S)  (zeros under ORACLE)
    naive = float(np.asarray(pk.fp_counts, float).sum() / consts.fp_ell_eff)
    t_draws = np.asarray(sam["t"]) if "t" in sam else np.zeros((lam_draws.shape[0], consts.n_kk))
    # per-draw mu_FP summed over k within coarse blocks: mu_FP[c,K,s] = w ell (1-eta) e^{t_K} lam E_{K,s}
    kz = np.asarray(consts.kz_to_K); E = np.asarray(consts.fp_E, float)
    EK = np.stack([E[kz == K].sum(axis=0) for K in range(consts.n_kk)])          # (KK, S)
    pref = float(consts.fp_w * consts.fp_ell_eff) * (1.0 - np.asarray(consts.fp_eta_c))  # (C,)
    mu_cKs = pref[None, :, None, None] * np.exp(t_draws)[:, None, :, None] * lam_draws[:, :, None, :] * EK[None, None, :, :]
    mu_cK = mu_cKs.sum(axis=3)                                   # (D, C, KK)
    nh = np.asarray(pk.nhat_edges, float); cc = 0.5 * (nh[:-1] + nh[1:])
    obs_cK = np.stack([np.asarray(pk.counts, float)[:, kz == K, :].sum(axis=(1, 2)) for K in range(consts.n_kk)], axis=1)
    fp_by_block = dict(
        mu_fp_block_p16_50_84=[[float(x) for x in np.percentile(mu_cK[:, :, K].sum(axis=1), [16, 50, 84])] for K in range(consts.n_kk)],
        counts_block=[float(obs_cK[:, K].sum()) for K in range(consts.n_kk)],
        fp_frac_by_c_median=[float(x) for x in np.median(mu_cK.sum(axis=2), axis=0) / np.maximum(obs_cK.sum(axis=1), 1)],
        mu_fp_nhat_group_p16_50_84={name: [float(x) for x in np.percentile(mu_cK[:, (cc >= lo) & (cc < hi), :].sum(axis=(1, 2)), [16, 50, 84])]
                                    for name, lo, hi in NHAT_GROUPS},
        counts_nhat_group={name: float(obs_cK[(cc >= lo) & (cc < hi), :].sum()) for name, lo, hi in NHAT_GROUPS},
        mu_fp_total_p16_50_84=[float(x) for x in np.percentile(mu_cK.sum(axis=(1, 2)), [16, 50, 84])],
        mu_fp_cK_median=np.median(mu_cK, axis=0).tolist())
    # FP-truth soft flag (census = hostless@17.2, P4 ⊕ P6(c))
    fp_truth = None
    if a.census and os.path.exists(a.census):
        cz = np.load(a.census, allow_pickle=True)
        host = np.asarray(cz["hostless"], float)                  # (C, Kf, S)
        host_cK = np.stack([host[:, kz == K, :].sum(axis=(1, 2)) for K in range(consts.n_kk)], axis=1)
        med = np.median(mu_cK, axis=0)
        rb = [float(med[:, K].sum() / max(host_cK[:, K].sum(), 1)) for K in range(consts.n_kk)]
        rg = {name: float(med[(cc >= lo) & (cc < hi), :].sum() / max(host_cK[(cc >= lo) & (cc < hi), :].sum(), 1)) for name, lo, hi in NHAT_GROUPS}
        fp_truth = dict(census=a.census, hostless_block=[float(x) for x in host_cK.sum(axis=0)],
                        ratio_block=rb, ratio_nhat_group=rg,
                        flag=bool(any((r < 0.5 or r > 2.0) for r in rb + list(rg.values()))))
    # calibration-predictive check on the 89 loa-0 counts (live cells)
    live = live_mask(consts); fpc_np = np.asarray(pk.fp_counts, float)
    rng = np.random.default_rng(a.seed)
    nd = lam_draws.shape[0]; sub = np.sort(rng.choice(nd, size=min(2000, nd), replace=False))
    mu_cal = float(consts.fp_ell_eff) * lam_draws[sub][:, :, live]  # (n, C, Ls)
    obs = fpc_np[:, live]
    def dev(n, mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(n > 0, n * np.log(n / mu), 0.0)
        return 2.0 * np.sum(term - (n - mu), axis=(-2, -1))
    D_obs = dev(obs[None], mu_cal); rep_n = rng.poisson(mu_cal); D_rep = dev(rep_n, mu_cal)
    p_dev = float(((D_rep > D_obs).mean() + 0.5 * (D_rep == D_obs).mean())) if a.ladder != "ORACLE" else float("nan")
    hi = cc >= 20.2 - 1e-9
    lam_hi = mu_cal[:, hi, :].sum(axis=(1, 2))                    # expected loa-0 events at Nhat>=20.2 per draw
    p0_hi = float(np.mean(np.exp(-lam_hi)))                       # posterior predictive P(0)
    calib = dict(n_draws=int(len(sub)), deviance_obs_p16_50_84=[float(x) for x in np.percentile(D_obs, [16, 50, 84])],
                 deviance_pvalue=p_dev, expected_loa0_events_ge20p2_p16_50_84=[float(x) for x in np.percentile(lam_hi, [16, 50, 84])],
                 prob_zero_ge20p2=p0_hi, flag=(bool(p_dev < 0.01 or p_dev > 0.99 or p0_hi < 0.05) if a.ladder != "ORACLE" else None),
                 note=("ORACLE: no FP parameters; calibration check not applicable" if a.ladder == "ORACLE" else ""),
                 per_row_expected_median=[float(x) for x in np.median(mu_cal.sum(axis=2), axis=0)], per_row_obs=[float(x) for x in obs.sum(axis=1)])
    # posterior-median predictive vs obs (VERBATIM logic)
    idx_med = int(np.argsort(np.asarray(sam["theta_level"]))[len(sam["theta_level"]) // 2])
    th_med = jnp.asarray(np.asarray(sam["theta_pop"])[idx_med]); pc_med = jnp.asarray(np.asarray(sam["psi_c"])[idx_med])
    t_med = jnp.asarray(t_draws[idx_med]); lf_med = jnp.asarray(lam_draws[idx_med])
    f_med = jnp.exp(th_med)
    if E_fixed is not None:
        tpx = jnp.einsum("cksb,bk->cks", jnp.asarray(E_fixed), f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
    else:
        Mg_use = Mg if Mg_fixed is None else jnp.asarray(Mg_fixed)
        if C_fixed is None:
            Cc = jax.nn.sigmoid(consts.eta_hat + pc_med)[:, consts.b_to_cell]
            tpx = jnp.einsum("skcb,sb,bk->cks", Mg_use, Cc, consts.g_bk * f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
        elif np.asarray(C_fixed).ndim == 2:
            tpx = jnp.einsum("skcb,sb,bk->cks", Mg_use, jnp.asarray(C_fixed), consts.g_bk * f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
        else:
            tpx = jnp.einsum("skcb,bks,bk->cks", Mg_use, jnp.asarray(C_fixed), f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
    fpx = (jnp.asarray(mu_fixed) if a.ladder == "ORACLE" else
           consts.fp_w * consts.fp_ell_eff * (1.0 - consts.fp_eta_c)[:, None, None]
           * jnp.exp(t_med[consts.kz_to_K])[None, :, None] * lf_med[:, None, :] * consts.fp_E[None, :, :])
    if mu_extra is not None:
        fpx = fpx + jnp.asarray(mu_extra)
    tot_obs = float(np.asarray(pk.counts, float).sum())
    obs3 = np.asarray(pk.counts, float); mu3 = np.asarray(tpx) + np.asarray(fpx); live3 = np.asarray(consts.dX) > 0
    def _marg(ax):
        o = obs3.sum(axis=ax); m = mu3.sum(axis=ax); return [float(x) for x in (m / np.maximum(o, 1.0))]
    pred_marg = dict(mu_over_obs_by_nhat=_marg((1, 2)), mu_over_obs_by_z=_marg((0, 2)), mu_over_obs_by_snr=_marg((0, 1)),
                     mu_over_obs_by_nhat_K=[[float(x) for x in (mu3[:, kz_np == K, :].sum((1, 2)) / np.maximum(obs3[:, kz_np == K, :].sum((1, 2)), 1.0))] for K in range(KK)],
                     tp_over_obs_by_nhat=[float(x) for x in (np.asarray(tpx).sum((1, 2)) / np.maximum(obs3.sum((1, 2)), 1.0))],
                     note="posterior-median draw (by theta_level); all fixed components included")
    diag = dict(
        ladder=a.ladder, nominal_fp_dof=NOMINAL_FP_DOF[a.ladder], t_sd=a.t_sd, tau_scale=a.tau_scale, calib_weight=a.calib_weight,
        target_accept=a.target_accept, divergences=int(div_g.sum()), divergences_per_chain=[int(x) for x in div_g.sum(axis=1)],
        ebfmi_per_chain=[round(x, 4) for x in ebfmi], mean_potential_energy_per_chain=[round(float(e.mean()), 1) for e in pe_g],
        sigma_N_post=[float(x) for x in np.percentile(sam["sigma_N"], [16, 50, 84])],
        sigma_z_post=[float(x) for x in np.percentile(sam["sigma_z"], [16, 50, 84])],
        fp_lam_total_over_naive=[round(float(x), 4) for x in np.percentile(lam_draws.sum(axis=(1, 2)) / naive, [16, 50, 84])],
        t_post_mean=[float(x) for x in t_draws.mean(axis=0)], t_post_sd=[float(x) for x in t_draws.std(axis=0)],
        e_t_post_median=[float(x) for x in np.exp(np.median(t_draws, axis=0))],
        t_post_in_record_prior_sd=[float(x) for x in (t_draws.mean(axis=0) / np.asarray(consts.t_sigma))],
        fp_scalar_posts={k: [float(x) for x in np.percentile(np.asarray(sam[k]).reshape(len(sam[k]), -1), [16, 50, 84], axis=0).T.reshape(-1)]
                         for k in FP_SITES[a.ladder] if k in sam and np.asarray(sam[k]).ndim <= 2 and np.asarray(sam[k]).reshape(len(sam[k]), -1).shape[1] <= 8},
        estimand_mixing=mixing,
        predictive_total_ratio=round(float((np.asarray(tpx) + np.asarray(fpx)).sum() / tot_obs), 4),
        predictive_fp_share=round(float(np.asarray(fpx).sum() / (np.asarray(tpx).sum() + np.asarray(fpx).sum())), 4),
        calibration_predictive=calib, fp_by_block=fp_by_block, fp_truth=fp_truth,
        diag_fix=fix, diag_ops=a.ops, predictive_marginals=pred_marg,
        fixed_files=dict(mg=a.mg_fixed_file, c=a.c_fixed_file, extra=a.extra_fixed_file),
        lam_cut=(dict(J=a.lam_imputations, j=a.lam_imputation, lam_fixed=lam_fixed, fp_a0=a.fp_a0) if a.ladder == "M1CUT" else None))
    out = dict(pack=a.pack, ladder=a.ladder, stage=a.stage, n_draws=int(f_draws.shape[0]), chains=a.chains,
               warmup=a.warmup, samples=a.samples, divergences=int(div_g.sum()), thresholds=rep,
               reporting_bins=binrep, perz_recovery=perz, diagnostics=diag, run_config=run_config(a),
               support_gate=support_record,
               role=("DIAGNOSTIC ORACLE run: mu_FP pinned to the mock FP-truth census; OUTSIDE the sealed ladder, never a candidate"
                     if a.ladder == "ORACLE" else
                     "MOCK-ONLY FP-model ladder candidate run (sealed predeclaration 3112022a); NOT a science product"))
    json.dump(out, open(a.out, "w"), indent=1)
    base = a.out[:-5]
    np.savez(base + "_fdraws.npz", f=f_draws, truth_f=ft, ntrue_edges=ntrue, zf_edges=np.asarray(pk.zf_edges), dX_k=dX_k)
    keep = {k: np.asarray(sam_g[k]) for k in POP_SITES + tuple(FP_SITES[a.ladder]) + ("lam_fp", "t") if k in sam_g}
    np.savez(base + "_bychain.npz", seed=a.seed, chains=a.chains, warmup=a.warmup, samples=a.samples, ladder=a.ladder,
             potential_energy=pe_g, energy=en_g, diverging=div_g, **keep)
    print(json.dumps({k: out[k] for k in ("ladder", "thresholds")}, indent=1))
    print(json.dumps({k: diag[k] for k in ("divergences", "ebfmi_per_chain", "fp_lam_total_over_naive", "t_post_mean", "e_t_post_median",
                                            "predictive_fp_share", "calibration_predictive", "estimand_mixing")}, indent=1, default=str))


if __name__ == "__main__":
    main()
