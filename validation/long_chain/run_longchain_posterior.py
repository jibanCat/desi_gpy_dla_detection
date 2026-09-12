#!/usr/bin/env python
"""VALIDATION-ONLY wrapper: the low-z HBI real posterior with ADDITIONAL retained diagnostics.

CLASSIFICATION: **VALIDATION-ONLY** (PI ruling 2026-09-12 §8).  This file is NOT a science
authority and must never become one.  It performs no inference of its own: it reproduces the
sampling core of ``CDDF_analysis/hbi_mcmc/cc_real_posterior.py`` @ ``1fd4828`` EXACTLY --
same ``load_pack``, same contract-guard subprocess, same fail-closed real-mode gate, same
``build_cc_tensors``, same ``model_cc``, same ``NUTS(target_accept_prob)``, same
``MCMC(chain_method="sequential")``, same ``jax.random.PRNGKey(seed)``, same committed
``reduce_f_posterior`` summary, same enforced G_A REAL-mode predictive-level check -- and
ADDITIONALLY retains, by chain, the draws of EVERY sampled site plus the per-draw potential
energy and divergence flag, which ``cc_real_posterior`` discards.

Retaining more of what the sampler already produced cannot change the sampler: the extra
sites are drawn whether or not they are stored, and ``get_samples`` does not consume RNG.
The claim is nevertheless CHECKED, not asserted: at the production configuration this driver
must reproduce the candidate's stored draws under ``np.array_equal``
(``verify_bit_identity.py``), and the campaign does not start otherwise.

Derived from the Phase-3 convergence driver
(/home/mfho/lowz_clean_work_2026-09-11/phase3_convergence/code/run_longchain_diag.py), which
passed the same gate on 2026-09-11.

Usage (see PREDECLARATION.md of the campaign for the sealed schedule):
  python validation/long_chain/run_longchain_posterior.py --pack C1.npz --seed 20260821 \
      --chains 4 --warmup 3000 --samples 4000 --target-accept 0.95 \
      --fp-mode informative_ln --out OUT.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
import numpyro

from CDDF_analysis.hbi_mcmc.provenance_util import run_config
from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import (build_cc_tensors,
                                                            model_cc)
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior

# every site model_cc SAMPLES at fp_mode="informative_ln" (574 scalar components on the
# C1 pack: 1+1+1+1+14+224+96+1+232+3).  theta_pop / f / lam_fp are DETERMINISTIC functions
# of these and are not gated separately.
SAMPLED_SITES = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N",
                 "eps_z", "psi_c", "fp_lam_total", "fp_shape_v", "t")


def _real_mode_gate(pack_path, pack):
    """Verbatim in effect from cc_real_posterior._real_mode_gate."""
    prov_path = pack_path[:-4] + ".provenance.json"
    if not os.path.exists(prov_path):
        raise SystemExit("REAL GATE: no provenance sidecar - refusing")
    prov = json.load(open(prov_path))
    if not prov.get("real_data"):
        raise SystemExit("REAL GATE: pack is not stamped real_data")
    if prov.get("truth_counts_sentinel") != "ZEROS_NO_TRUTH":
        raise SystemExit("REAL GATE: truth sentinel missing")
    tc = np.asarray(pack.truth_counts)
    if tc.size == 0 or np.any(tc != 0):
        raise SystemExit("REAL GATE: truth_counts is not the all-zero sentinel")
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--chains", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--fp-mode", default="informative_ln")
    ap.add_argument("--fp-alpha0", type=float, default=None)
    ap.add_argument("--fp-total-scale", type=float, default=1.0)
    ap.add_argument("--t-scale", type=float, default=1.0)
    ap.add_argument("--fp-s-empty", type=float, default=None)
    ap.add_argument("--stage", default="", help="campaign stage label (metadata only)")
    ap.add_argument("--skip-guards", action="store_true",
                    help="skip the contract-guard SUBPROCESS only (it touches no RNG); "
                         "the fail-closed real-mode gate always runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    numpyro.set_host_device_count(a.chains)

    # ---- contract guards (subprocess; no RNG state) --------------------------
    greport = {}
    if not a.skip_guards:
        r = subprocess.run([sys.executable, "-m",
                            "CDDF_analysis.hbi_mcmc.contract_guards_check",
                            "--pack", a.pack], capture_output=True, text=True)
        try:
            greport = json.loads(r.stdout[r.stdout.index("{"):])
        except Exception:
            raise SystemExit(f"guards output unparseable:\n{r.stdout}\n{r.stderr}")
        other_fail = [k for k, v in greport.items()
                      if isinstance(v, dict) and v.get("status") == "FAIL"
                      and k != "G_A_partition"]
        if other_fail:
            raise SystemExit(f"contract guards FAILED (non-G_A): {other_fail}")

    pk = load_pack(a.pack)
    prov = _real_mode_gate(a.pack, pk)

    consts, Mg = build_cc_tensors(pk)
    counts = jnp.asarray(np.asarray(pk.counts, float))
    fpc = jnp.asarray(np.asarray(pk.fp_counts, float))

    from numpyro.infer import MCMC, NUTS
    kern = NUTS(model_cc, target_accept_prob=a.target_accept)
    mcmc = MCMC(kern, num_warmup=a.warmup, num_samples=a.samples,
                num_chains=a.chains, chain_method="sequential",
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(a.seed), consts, Mg, counts=counts,
             fp_counts=fpc, fp_mode=a.fp_mode, fp_alpha0=a.fp_alpha0,
             fp_total_scale=a.fp_total_scale, t_scale=a.t_scale,
             fp_s_empty=a.fp_s_empty,
             extra_fields=("potential_energy", "diverging"))
    sam = mcmc.get_samples(group_by_chain=False)
    sam_g = mcmc.get_samples(group_by_chain=True)
    xf_g = mcmc.get_extra_fields(group_by_chain=True)
    f_draws = np.asarray(sam["f"])

    # ---- the committed reduction, exactly as cc_real_posterior reports it -----
    red = reduce_f_posterior(f_draws, pk)
    ntrue = np.asarray(pk.ntrue_edges, float)
    dN = np.diff(ntrue)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    REDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)

    def q(dr):
        return [float(x) for x in np.percentile(dr, [2.5, 16, 50, 84, 97.5])]

    rep = {k: dict(post_p2p5_16_50_84_97p5=q(np.asarray(red[k])))
           for k in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz")}
    binrep = []
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        dr = ((f_draws[:, m, :] * dN[None, m, None]).sum(axis=1)
              * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        binrep.append(dict(bin=[round(e0, 1), round(e1, 1)], f_post=q(dr)))

    div_g = np.asarray(xf_g["diverging"])
    pe_g = np.asarray(xf_g["potential_energy"], float)
    fg = np.asarray(sam_g["f"])
    perchain, rhat_asrun = {}, {}
    for key in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz"):
        cs = np.stack([np.asarray(reduce_f_posterior(fg[ci], pk)[key])
                       for ci in range(fg.shape[0])])
        perchain[key] = [round(float(np.median(c)), 5) for c in cs]
        W = cs.var(axis=1, ddof=1).mean()
        Bv = cs.mean(axis=1).var(ddof=1) * cs.shape[1]
        # the INHERITED unsplit statistic, named for what it computes; recorded for
        # continuity with the historical record and NEVER used as this campaign's gate
        rhat_asrun[key] = round(float(np.sqrt(
            ((cs.shape[1] - 1) / cs.shape[1] * W + Bv / cs.shape[1]) / W)), 4)

    naive = float(np.asarray(pk.fp_counts, float).sum() / consts.fp_ell_eff)
    diag = dict(
        divergences=int(div_g.sum()),
        divergences_per_chain=[int(v) for v in div_g.sum(axis=1)],
        target_accept=a.target_accept, fp_mode=a.fp_mode,
        sigma_N_post=[float(x) for x in np.percentile(sam["sigma_N"], [16, 50, 84])],
        sigma_z_post=[float(x) for x in np.percentile(sam["sigma_z"], [16, 50, 84])],
        fp_lam_total_over_naive=[
            round(float(x), 4) for x in
            np.percentile(np.asarray(sam["lam_fp"]).sum(axis=(1, 2)) / naive,
                          [16, 50, 84])],
        t_post_mean=[float(x) for x in np.asarray(sam["t"]).mean(axis=0)],
        t_post_in_prior_sd=[float(x) for x in (np.asarray(sam["t"]).mean(axis=0)
                                               / np.asarray(consts.t_sigma))],
        perchain_estimand_medians=perchain,
        rhat_unsplit_inherited_NOT_THE_GATE=rhat_asrun,
        mean_potential_energy_per_chain=[round(float(e.mean()), 1) for e in pe_g])

    # ---- enforced G_A REAL-mode check (verbatim in effect) -------------------
    idx_med = int(np.argsort(np.asarray(sam["theta_level"]))[len(sam["theta_level"]) // 2])
    th_med = jnp.asarray(np.asarray(sam["theta_pop"])[idx_med])
    pc_med = jnp.asarray(np.asarray(sam["psi_c"])[idx_med])
    t_med = jnp.asarray(np.asarray(sam["t"])[idx_med])
    lf_med = jnp.asarray(np.asarray(sam["lam_fp"])[idx_med])
    Cc = jax.nn.sigmoid(consts.eta_hat + pc_med)[:, consts.b_to_cell]
    w_ = consts.g_bk * jnp.exp(th_med) * consts.dN_b[:, None]
    tpx = jnp.einsum("skcb,sb,bk->cks", Mg, Cc, w_) * consts.dX[None, :, :]
    fpx = (consts.fp_w * consts.fp_ell_eff
           * (1.0 - consts.fp_eta_c)[:, None, None]
           * jnp.exp(t_med[consts.kz_to_K])[None, :, None]
           * lf_med[:, None, :] * consts.fp_E[None, :, :])
    level = float((np.asarray(tpx) + np.asarray(fpx)).sum()
                  / np.asarray(pk.counts, float).sum())
    if abs(level - 1.0) > 0.06:
        raise SystemExit(f"G_A REAL-mode FAILED: predictive level {level:.4f}")
    fp_share = float(np.asarray(fpx).sum() / (np.asarray(tpx).sum() + np.asarray(fpx).sum()))
    guards_summary = dict(
        subprocess_report={k: (v.get("status") if isinstance(v, dict) else v)
                           for k, v in greport.items()},
        subprocess_skipped=bool(a.skip_guards),
        G_A_truthpoint="N/A on real pack (zeros sentinel; fail-closed FAIL recorded)",
        G_A_real_mode=dict(predictive_level=round(level, 4),
                           fp_share=round(fp_share, 4), tol=0.06, status="PASS"))

    # ---- outputs --------------------------------------------------------------
    base = a.out[:-5]
    np.savez(base + "_fdraws.npz", f=f_draws, ntrue_edges=ntrue,
             zf_edges=np.asarray(pk.zf_edges))
    bychain = {s: np.asarray(sam_g[s]) for s in SAMPLED_SITES}
    np.savez(base + "_bychain.npz", seed=a.seed, chains=a.chains,
             warmup=a.warmup, samples=a.samples,
             potential_energy=pe_g, diverging=div_g, **bychain)

    out = dict(pack=a.pack, pack_provenance=prov, n_draws=int(f_draws.shape[0]),
               chains=a.chains, warmup=a.warmup, samples=a.samples,
               stage=a.stage, guards=guards_summary,
               estimand=("POSTERIOR_MEDIAN_CI (committed reduce_f_posterior; STATISTICAL "
                         "interval only; NO response or mock-to-real central-value "
                         "correction applied)"),
               thresholds=rep, reporting_bins=binrep, diagnostics=diag,
               run_config=run_config(a),
               sampled_sites={s: list(np.asarray(sam_g[s]).shape[2:]) for s in SAMPLED_SITES},
               role=("VALIDATION-ONLY long-chain FINAL-POSTERIOR campaign run "
                     "(PI ruling 2026-09-12 sec.3); CANDIDATE, not adopted, not frozen"))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("stage", "chains", "warmup", "samples",
                                          "n_draws", "diagnostics")}, indent=1))


if __name__ == "__main__":
    main()
