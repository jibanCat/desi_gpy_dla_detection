#!/usr/bin/env python
"""cc_posterior_validation.py — MOCK-SIDE posterior validation of the adopted
count-conserving operator (PI ruling 2026-08-17: prerequisite material for
the correction-vs-systematic decision; MOCK ONLY — refuses to run without
truth_counts).

The generative program is model_a's, verbatim, with two ruled differences:
  * the response fold uses the ADOPTED representation under the
    count-conservation rule, as a PRECOMPUTED mass tensor (kernel FIXED in
    the sampler; its fit uncertainty is the bootstrap CARRIER, propagated as
    a post-hoc predictive band — the adoption's uncertainty structure);
  * psi_k_delta is therefore absent (the retired wide default is never
    sampled; the carrier replaces the fit-uncertainty role).
Population prior (2-D RW), psi_c, t, and the joint FP block are byte-copied
semantics from model_a.

Validation estimand (POSTERIOR_MEDIAN_CI, mock-only): posterior median + CI
of dN/dX(>=20.3), dN/dX(>=20.0) and the 0.2-dex reporting-bin f against the
mock truth.

Env: gpdla-hbi. Usage:
  python -m CDDF_analysis.hbi_mcmc.cc_posterior_validation --pack PACK_v2.npz
      [--samples 500] [--warmup 500] [--chains 2] [--out JSON]
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from CDDF_analysis.hbi_mcmc.pack import ModelAPack, load_pack
from CDDF_analysis.hbi_mcmc.forward import build_consts, eta_hat_sigma_hat
from CDDF_analysis.hbi_mcmc.count_conserving_fold import (surface_masses,
                                                          phi_from_surfaces)


def build_cc_tensors(pack: ModelAPack):
    """Precompute the gathered count-conserving mass tensor + fold pieces."""
    for f in ("adopted_resp_mu_coef", "adopted_phi_ref", "tp_convention_id"):
        if getattr(pack, f, None) is None:
            raise ValueError("cc_posterior_validation: pack lacks the v1.2 "
                             f"adopted-contract stamps ({f}) — rebuild via "
                             "upgrade_packs_v2 first (fail-closed).")
    if pack.truth_counts is None:
        raise ValueError("cc_posterior_validation is MOCK-ONLY: the pack "
                         "carries no truth_counts — refusing (no real-data "
                         "posterior is authorized).")
    consts = build_consts(pack, resp_clamp="both")
    ne = np.asarray(pack.nhat_edges, float)
    masses, phi = surface_masses(
        pack, pack.adopted_resp_mu_coef, pack.adopted_resp_sig_coef,
        pack.adopted_resp_skew_coef,
        np.asarray(pack.adopted_resp_fit_range, float), ne)
    phi_ref = np.asarray(pack.adopted_phi_ref, float)
    d = float(np.max(np.abs(phi_ref - phi_from_surfaces(pack))))
    if d > 1e-9:
        raise ValueError(f"stored phi_ref deviates by {d:.2e} (G-CC)")
    masses = masses / np.maximum(phi, 1e-12)[:, :, None, :] \
        * phi_ref[:, :, None, :]
    s2sr = np.asarray(consts.s_to_sresp)
    kz2K = np.asarray(consts.kz_to_K)
    K2zr = np.asarray(consts.K_to_zresp)
    # gathered (S, Kf, C, B)
    Mg = masses[s2sr[:, None], K2zr[kz2K][None, :], :, :]
    return consts, jnp.asarray(Mg)


def model_cc(consts, Mg, counts=None, fp_counts=None, *,
             fp_mode="joint", fp_eps_rate=1e-6, fp_shape_sd=3.0,
             fp_alpha0=None, fp_total_scale=1.0, t_scale=1.0,
             sigma_N_scale=0.5, sigma_z_scale=0.5,
             level_scale=4.0, slope_scale=2.0):
    """fp_mode: 'joint' = model_a's joint FP block (baseline);
    'anchored' = lam_fp FIXED at the loa-0 forest-only calibration
    (fp_counts/fp_ell_eff) and t FIXED at 0 — the PI-ruled strong
    loa-0 anchor (uncertainties carried as post-hoc named bands, the
    same fixed+carrier structure as the adopted kernel);
    'anchored_t' = lam_fp fixed, t sampled with its calibrated prior
    (isolates which nuisance drives);
    'amplitude' = PI ruling (checkpoint 10.5, item 3): a single overall FP
    normalization amplitude A anchored to the loa-0 forest-only calibration
    with the calibration's own counting uncertainty
    (A ~ Gamma(N_fp+1/2, N_fp), the Jeffreys posterior of the loa-0 total
    rate; sd ~= 1/sqrt(N_fp)); the (C,S) SHAPE is FIXED at the loa-0
    point estimate — no NHI-dependent FP shape freedom; t keeps its
    separately calibrated prior. The fp_counts Poisson term is dropped in
    this mode: the loa-0 information enters ONCE, through the A prior
    (keeping both would double-count the calibration data). The prior
    width is set by the loa-0 counts alone — nothing tuned from closure
    or mock truth."""
    B, Kf = consts.n_b, consts.n_k
    C, S = consts.n_c, consts.n_s
    sigma_N = numpyro.sample("sigma_N", dist.HalfNormal(sigma_N_scale))
    sigma_z = numpyro.sample("sigma_z", dist.HalfNormal(sigma_z_scale))
    level = numpyro.sample("theta_level", dist.Normal(0.0, level_scale))
    slope = numpyro.sample("theta_slope", dist.Normal(0.0, slope_scale))
    eps_N = numpyro.sample(
        "eps_N", dist.Normal(0.0, 1.0).expand([max(B - 2, 0)]).to_event(1))
    eps_z = numpyro.sample(
        "eps_z", dist.Normal(0.0, 1.0).expand([B, max(Kf - 1, 0)]).to_event(2))
    b_idx = jnp.arange(B) - 0.5 * (B - 1)
    curv = jnp.cumsum(jnp.cumsum(jnp.concatenate([jnp.zeros(2), eps_N])))[:B]
    theta_col0 = level + slope * b_idx + sigma_N * curv
    theta = theta_col0[:, None] + jnp.concatenate(
        [jnp.zeros((B, 1)), sigma_z * jnp.cumsum(eps_z, axis=1)], axis=1)
    theta = numpyro.deterministic("theta_pop", theta)
    numpyro.deterministic("f", jnp.exp(theta))

    psi_c = numpyro.sample(
        "psi_c", dist.Normal(0.0, consts.sigma_hat).to_event(2))
    if fp_mode == "joint":
        t = numpyro.sample("t", dist.Normal(0.0, consts.t_sigma).to_event(1))
        lam_total = numpyro.sample("fp_lam_total",
                                   dist.Gamma(0.5, fp_eps_rate))
        v = numpyro.sample(
            "fp_shape_v",
            dist.ZeroSumNormal(fp_shape_sd, event_shape=(C * S,)))
        pi = jax.nn.softmax(v)
        lam_fp = numpyro.deterministic("lam_fp",
                                       (lam_total * pi).reshape(C, S))
        numpyro.sample("fp_counts",
                       dist.Poisson(consts.fp_ell_eff * lam_fp).to_event(2),
                       obs=fp_counts)
    elif fp_mode in ("anchored", "anchored_t"):
        lam_fp = numpyro.deterministic(
            "lam_fp", jnp.asarray(fp_counts) / consts.fp_ell_eff)
        if fp_mode == "anchored_t":
            t = numpyro.sample("t", dist.Normal(0.0, consts.t_sigma)
                               .to_event(1))
        else:
            t = numpyro.deterministic("t", jnp.zeros(consts.t_sigma.shape))
    elif fp_mode == "amplitude":
        n_fp = float(np.asarray(fp_counts).sum())
        lam_hat = jnp.asarray(fp_counts) / consts.fp_ell_eff
        A = numpyro.sample("fp_amp", dist.Gamma(n_fp + 0.5, n_fp))
        lam_fp = numpyro.deterministic("lam_fp", A * lam_hat)
        t = numpyro.sample("t", dist.Normal(0.0, consts.t_sigma).to_event(1))
    elif fp_mode == "informative":
        # PI ruling checkpoint 10.9 (predeclaration @59849c9): the loa-0
        # calibration as the PRIOR, used once (fp_counts likelihood term
        # dropped). Total: Gamma(N_FP+1/2, ell_eff) (Jeffreys posterior of
        # the loa-0 total rate, rel sd 10.6%); shape: Dirichlet(n+alpha0)
        # (the loa-0 multinomial posterior; alpha0=1/K Perks primary for
        # the 207-empty-cell sparse multinomial); t: calibrated, unchanged.
        # Sensitivity knobs (fp_total_scale, fp_alpha0, t_scale) are the
        # PREDECLARED axes only.
        fpc_np = np.asarray(fp_counts, float)
        n_fp = float(fpc_np.sum())
        K_cells = fpc_np.size
        a0 = (1.0 / K_cells) if fp_alpha0 is None else float(fp_alpha0)
        ts = float(fp_total_scale)
        lam_total = numpyro.sample(
            "fp_lam_total",
            dist.Gamma(n_fp * ts + 0.5, float(consts.fp_ell_eff) * ts))
        conc = jnp.asarray(fpc_np.reshape(-1) + a0)
        pi = numpyro.sample("fp_shape_pi", dist.Dirichlet(conc))
        lam_fp = numpyro.deterministic(
            "lam_fp", (lam_total * pi).reshape(C, S))
        t = numpyro.sample(
            "t", dist.Normal(0.0, consts.t_sigma * float(t_scale))
            .to_event(1))
    else:
        raise ValueError(f"unknown fp_mode {fp_mode!r}")

    # count-conserving fold with the FIXED adopted kernel (precomputed Mg)
    Cc = jax.nn.sigmoid(consts.eta_hat + psi_c)[:, consts.b_to_cell]  # (S,B)
    f = jnp.exp(theta)                                                # (B,Kf)
    w = consts.g_bk * f * consts.dN_b[:, None]                        # (B,Kf)
    tp = jnp.einsum("skcb,sb,bk->cks", Mg, Cc, w) * consts.dX[None, :, :]
    fp = (consts.fp_w * consts.fp_ell_eff
          * (1.0 - consts.fp_eta_c)[:, None, None]
          * jnp.exp(t[consts.kz_to_K])[None, :, None]
          * lam_fp[:, None, :] * consts.fp_E[None, :, :])
    mu = tp + fp
    obs_mask = jnp.broadcast_to(jnp.asarray(consts.dX > 0)[None, :, :],
                                mu.shape)
    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("counts", dist.Poisson(jnp.clip(mu, 1e-300, None)),
                       obs=counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--fp-mode", default="joint",
                    choices=["joint", "anchored", "anchored_t", "amplitude",
                             "informative"])
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--fp-alpha0", type=float, default=None)
    ap.add_argument("--fp-total-scale", type=float, default=1.0)
    ap.add_argument("--t-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    numpyro.set_host_device_count(a.chains)

    pk = load_pack(a.pack)
    consts, Mg = build_cc_tensors(pk)
    counts = jnp.asarray(np.asarray(pk.counts, float))
    fpc = jnp.asarray(np.asarray(pk.fp_counts, float))

    from numpyro.infer import MCMC, NUTS
    kern = NUTS(model_cc, target_accept_prob=a.target_accept)
    mcmc = MCMC(kern, num_warmup=a.warmup, num_samples=a.samples,
                num_chains=a.chains, chain_method="sequential",
                progress_bar=True)
    mcmc.run(jax.random.PRNGKey(a.seed), consts, Mg, counts=counts,
             fp_counts=fpc, fp_mode=a.fp_mode, fp_alpha0=a.fp_alpha0,
             fp_total_scale=a.fp_total_scale, t_scale=a.t_scale)
    sam = mcmc.get_samples(group_by_chain=False)
    sam_g = mcmc.get_samples(group_by_chain=True)
    f_draws = np.asarray(sam["f"])                       # (D, B, Kf)

    # --- diagnostics: where does the slack go? ---------------------------
    import jax as _jax
    diag = {}
    diag["sigma_N_post"] = [float(x) for x in
                            np.percentile(sam["sigma_N"], [16, 50, 84])]
    diag["sigma_z_post"] = [float(x) for x in
                            np.percentile(sam["sigma_z"], [16, 50, 84])]
    diag["level_post"] = [float(x) for x in
                          np.percentile(sam["theta_level"], [16, 50, 84])]
    psi_m = np.asarray(sam["psi_c"]).mean(axis=0)
    diag["psi_c_mean_over_cells"] = float(psi_m.mean())
    diag["psi_c_mean_in_prior_sd_units"] = float(
        (psi_m / np.asarray(consts.sigma_hat)).mean())
    lam_draws = np.asarray(sam["lam_fp"]).sum(axis=(1, 2))
    naive = float(np.asarray(pk.fp_counts, float).sum() / consts.fp_ell_eff)
    diag["fp_lam_total_over_naive"] = [round(float(q / naive), 4)
                                       for q in np.percentile(lam_draws,
                                                              [16, 50, 84])]
    diag["t_post_mean"] = ([float(x)
                            for x in np.asarray(sam["t"]).mean(axis=0)]
                           if "t" in sam else [0.0])
    diag["t_post_in_prior_sd"] = (
        [float(x) for x in (np.asarray(sam["t"]).mean(axis=0)
                            / np.asarray(consts.t_sigma))]
        if "t" in sam else [0.0])
    if "fp_amp" in sam:
        diag["fp_amp_post_p16_50_84"] = [
            float(x) for x in np.percentile(sam["fp_amp"], [16, 50, 84])]
        n_fp = float(np.asarray(pk.fp_counts, float).sum())
        diag["fp_amp_prior_sd"] = round(float(1.0 / np.sqrt(n_fp)), 5)
        diag["fp_amp_pull_sd"] = round(float(
            (np.median(sam["fp_amp"]) - 1.0) * np.sqrt(n_fp)), 2)
    diag["fp_mode"] = a.fp_mode
    diag["target_accept"] = a.target_accept
    diag["fp_alpha0"] = a.fp_alpha0
    diag["fp_total_scale"] = a.fp_total_scale
    diag["t_scale"] = a.t_scale
    # split-Rhat + ESS on the threshold estimands (grouped chains)
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior as _red
    fg = np.asarray(sam_g["f"])
    mixing = {}
    for key in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz"):
        cs = np.stack([np.asarray(_red(fg[ci], pk)[key])
                       for ci in range(fg.shape[0])])
        W = cs.var(axis=1, ddof=1).mean()
        Bv = cs.mean(axis=1).var(ddof=1) * cs.shape[1]
        rh = float(np.sqrt(((cs.shape[1] - 1) / cs.shape[1] * W
                            + Bv / cs.shape[1]) / W)) \
            if cs.shape[0] > 1 else None
        from numpyro.diagnostics import effective_sample_size
        ess = float(effective_sample_size(cs))
        mixing[key] = dict(split_rhat=(round(rh, 4) if rh else None),
                           ess=round(ess, 1),
                           perchain_median=[round(float(np.median(c)), 5)
                                            for c in cs])
    diag["estimand_mixing"] = mixing
    # posterior-median predictive vs obs at the reporting grain
    idx_med = int(np.argsort(np.asarray(
        sam["theta_level"]))[len(sam["theta_level"]) // 2])
    th_med = jnp.asarray(np.asarray(sam["theta_pop"])[idx_med])
    pc_med = jnp.asarray(np.asarray(sam["psi_c"])[idx_med])
    t_med = (jnp.asarray(np.asarray(sam["t"])[idx_med]) if "t" in sam
             else jnp.zeros(consts.t_sigma.shape))
    lf_med = jnp.asarray(np.asarray(sam["lam_fp"])[idx_med])
    Cc = _jax.nn.sigmoid(consts.eta_hat + pc_med)[:, consts.b_to_cell]
    w = consts.g_bk * jnp.exp(th_med) * consts.dN_b[:, None]
    tpx = jnp.einsum("skcb,sb,bk->cks", Mg, Cc, w) * consts.dX[None, :, :]
    fpx = (consts.fp_w * consts.fp_ell_eff
           * (1.0 - consts.fp_eta_c)[:, None, None]
           * jnp.exp(t_med[consts.kz_to_K])[None, :, None]
           * lf_med[:, None, :] * consts.fp_E[None, :, :])
    mu_med = np.asarray(tpx + fpx).sum(axis=(1, 2))
    obs_c = np.asarray(pk.counts, float).sum(axis=(1, 2))
    diag["predictive_total_ratio"] = round(float(mu_med.sum()
                                                 / obs_c.sum()), 4)
    diag["predictive_fp_share"] = round(float(
        np.asarray(fpx).sum() / (np.asarray(tpx).sum()
                                 + np.asarray(fpx).sum())), 4)

    # truth comparison via the COMMITTED reduction (pathlength-weighted
    # all-z estimand + reported-support masking — the estimand as reported)
    from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
    ft = np.asarray(truth_f(pk), float)                  # (B, Kf)
    red = reduce_f_posterior(f_draws, pk)
    red_t = reduce_f_posterior(ft[None, :, :], pk)
    ntrue = np.asarray(pk.ntrue_edges, float)
    dN = np.diff(ntrue)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    REDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)

    rep = {}
    for thr, key in ((20.0, "dndx_dla_20p0_allz"), (20.3, "dndx_dla_20p3_allz")):
        dr = np.asarray(red[key])
        tv = float(np.asarray(red_t[key])[0])
        q = np.percentile(dr, [16, 50, 84])
        rep[f"ge{thr}"] = dict(
            truth=tv, post_p16_50_84=[float(x) for x in q],
            median_bias_pct=round(100 * (q[1] / tv - 1), 2),
            truth_in_68=bool(q[0] <= tv <= q[2]))
    binrep = []
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        # pathlength-weighted all-z bin estimand (matches the reduction)
        dr = ((f_draws[:, m, :] * dN[None, m, None]).sum(axis=1)
              * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        tv = float(((ft[m, :] * dN[m, None]).sum(axis=0)
                    * dX_k).sum() / dX_k.sum())
        q = np.percentile(dr, [16, 50, 84])
        binrep.append(dict(bin=[round(e0, 1), round(e1, 1)],
                           median_bias_pct=round(100 * (q[1] / tv - 1), 2),
                           truth_in_68=bool(q[0] <= tv <= q[2])))
    div = int(np.sum(mcmc.get_extra_fields()["diverging"])) \
        if "diverging" in mcmc.get_extra_fields() else None
    out = dict(pack=a.pack, n_draws=int(f_draws.shape[0]),
               divergences=div, thresholds=rep, reporting_bins=binrep,
               diagnostics=diag,
               role=("MOCK-ONLY posterior validation of the adopted "
                     "count-conserving operator; kernel fixed, carrier "
                     "post-hoc; NOT a claim-grade run"))
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
