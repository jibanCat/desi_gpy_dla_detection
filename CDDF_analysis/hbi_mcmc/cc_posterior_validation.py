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
             fp_eps_rate=1e-6, fp_shape_sd=3.0,
             sigma_N_scale=0.5, sigma_z_scale=0.5,
             level_scale=4.0, slope_scale=2.0):
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
    t = numpyro.sample("t", dist.Normal(0.0, consts.t_sigma).to_event(1))

    lam_total = numpyro.sample("fp_lam_total", dist.Gamma(0.5, fp_eps_rate))
    v = numpyro.sample(
        "fp_shape_v", dist.ZeroSumNormal(fp_shape_sd, event_shape=(C * S,)))
    pi = jax.nn.softmax(v)
    lam_fp = numpyro.deterministic("lam_fp", (lam_total * pi).reshape(C, S))
    numpyro.sample("fp_counts",
                   dist.Poisson(consts.fp_ell_eff * lam_fp).to_event(2),
                   obs=fp_counts)

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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    numpyro.set_host_device_count(a.chains)

    pk = load_pack(a.pack)
    consts, Mg = build_cc_tensors(pk)
    counts = jnp.asarray(np.asarray(pk.counts, float))
    fpc = jnp.asarray(np.asarray(pk.fp_counts, float))

    from numpyro.infer import MCMC, NUTS
    kern = NUTS(model_cc, target_accept_prob=0.9)
    mcmc = MCMC(kern, num_warmup=a.warmup, num_samples=a.samples,
                num_chains=a.chains, chain_method="sequential",
                progress_bar=True)
    mcmc.run(jax.random.PRNGKey(a.seed), consts, Mg, counts=counts,
             fp_counts=fpc)
    sam = mcmc.get_samples(group_by_chain=False)
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
    diag["t_post_mean"] = [float(x)
                           for x in np.asarray(sam["t"]).mean(axis=0)]
    # posterior-median predictive vs obs at the reporting grain
    idx_med = int(np.argsort(np.asarray(
        sam["theta_level"]))[len(sam["theta_level"]) // 2])
    th_med = jnp.asarray(np.asarray(sam["theta_pop"])[idx_med])
    pc_med = jnp.asarray(np.asarray(sam["psi_c"])[idx_med])
    t_med = jnp.asarray(np.asarray(sam["t"])[idx_med])
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
    for thr, key in ((20.0, "dndx_20p0_allz"), (20.3, "dndx_20p3_allz")):
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
