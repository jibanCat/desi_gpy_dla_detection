#!/usr/bin/env python
"""cc_real_posterior.py — the guarded FINAL real-data HBI posterior.

PI AUTHORIZATION (explicit, checkpoint 10.8): "Proceed: freeze the c=3300
observable-only convention; rebuild/certify the real pack; run the guarded
final real-data HBI posterior with joint model @ target_accept=0.95; keep
statistics and named systematics separate; apply no response or mock-to-real
central-value correction; BH unchanged."

Same generative program as the mock-validated chain (model_cc: adopted
count-conserving operator FIXED, joint effective-nuisance FP block).
Differences from cc_posterior_validation (MOCK-ONLY; not modified):

  * REAL-MODE GATE (fail-closed): the provenance sidecar must carry
    ``real_data: true`` and ``truth_counts_sentinel: "ZEROS_NO_TRUTH"``, and
    the stored truth_counts must be exactly all-zero. Mock packs REFUSED.
  * contract guards must PASS (subprocess) BEFORE the pack is consumed.
  * output = the committed POSTERIOR_MEDIAN_CI reductions
    (reduce_f_posterior) + diagnostics; NO truth comparison exists.
  * PRIVACY: outputs go to scratch; aggregates to the private notes repo.

Env: gpdla-hbi (frozen per REALDATA_ENV_PRESCRIPTION).
Usage:
  python -m CDDF_analysis.hbi_mcmc.cc_real_posterior --pack REAL_PACK_v2.npz
      --out OUT.json [--samples 500 --warmup 500 --chains 2
      --target-accept 0.95]
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
import numpyro

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import (build_cc_tensors,
                                                            model_cc)
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior


def _real_mode_gate(pack_path, pack):
    prov_path = pack_path[:-4] + ".provenance.json"
    if not os.path.exists(prov_path):
        raise SystemExit("REAL GATE: no provenance sidecar — refusing")
    prov = json.load(open(prov_path))
    if not prov.get("real_data"):
        raise SystemExit("REAL GATE: pack is not stamped real_data — mocks "
                         "use cc_posterior_validation")
    if prov.get("truth_counts_sentinel") != "ZEROS_NO_TRUTH":
        raise SystemExit("REAL GATE: truth sentinel missing")
    tc = np.asarray(pack.truth_counts)
    if tc.size == 0 or np.any(tc != 0):
        raise SystemExit("REAL GATE: truth_counts is not the all-zero "
                         "sentinel — refusing")
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--fp-mode", default="joint",
                    choices=["joint", "informative", "informative_ln"])
    ap.add_argument("--fp-alpha0", type=float, default=None)
    ap.add_argument("--fp-total-scale", type=float, default=1.0)
    ap.add_argument("--t-scale", type=float, default=1.0)
    ap.add_argument("--fp-s-empty", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    numpyro.set_host_device_count(a.chains)

    r = subprocess.run([sys.executable, "-m",
                        "CDDF_analysis.hbi_mcmc.contract_guards_check",
                        "--pack", a.pack], capture_output=True, text=True)
    # Parse the guard report. G_A's truth-point form is UNDEFINED on a
    # truth-less real pack (its docstring: "on any future real pack the
    # check runs against the calibrated level band") — the zeros sentinel
    # makes it FAIL, which is the fail-closed behavior working. Its
    # documented REAL-mode semantics are implemented below as an ENFORCED
    # post-run check: the posterior-median predictive level mu/obs must sit
    # within the guard's own tolerance (|level-1| <= 0.06). Every OTHER
    # guard must hard-pass here.
    try:
        greport = json.loads(r.stdout[r.stdout.index("{"):])
    except Exception:
        raise SystemExit(f"guards output unparseable:\n{r.stdout}\n{r.stderr}")
    other_fail = [k for k, v in greport.items()
                  if isinstance(v, dict) and v.get("status") == "FAIL"
                  and k != "G_A_partition"]
    if other_fail:
        raise SystemExit(f"contract guards FAILED (non-G_A): {other_fail}\n"
                         f"{r.stdout}")
    ga_truthpoint = greport.get("G_A_partition", {}).get("status")

    pk = load_pack(a.pack)
    prov = _real_mode_gate(a.pack, pk)
    if ga_truthpoint == "FAIL" and not prov.get("real_data"):
        raise SystemExit("G_A failed on a non-real pack — refusing")

    consts, Mg = build_cc_tensors(pk)
    counts = jnp.asarray(np.asarray(pk.counts, float))
    fpc = jnp.asarray(np.asarray(pk.fp_counts, float))

    from numpyro.infer import MCMC, NUTS
    kern = NUTS(model_cc, target_accept_prob=a.target_accept)
    mcmc = MCMC(kern, num_warmup=a.warmup, num_samples=a.samples,
                num_chains=a.chains, chain_method="sequential",
                progress_bar=True)
    # per-chain mixing diagnostics + potential energy (mode weighing)
    mcmc.run(jax.random.PRNGKey(a.seed), consts, Mg, counts=counts,
             fp_counts=fpc, fp_mode=a.fp_mode, fp_alpha0=a.fp_alpha0,
             fp_total_scale=a.fp_total_scale, t_scale=a.t_scale,
             fp_s_empty=a.fp_s_empty,
             extra_fields=("potential_energy", "diverging"))
    sam = mcmc.get_samples(group_by_chain=False)
    sam_g = mcmc.get_samples(group_by_chain=True)
    f_draws = np.asarray(sam["f"])

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
    xf = mcmc.get_extra_fields()
    div = int(np.sum(xf["diverging"])) if "diverging" in xf else None
    # per-chain estimand medians + split-Rhat on the two thresholds
    fg = np.asarray(sam_g["f"])                    # (chains, draws, B, Kf)
    perchain = {}
    rhat = {}
    for key in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz"):
        cs = []
        for ci in range(fg.shape[0]):
            rc = reduce_f_posterior(fg[ci], pk)
            cs.append(np.asarray(rc[key]))
        cs = np.stack(cs)                           # (chains, draws)
        perchain[key] = [round(float(np.median(c)), 5) for c in cs]
        W = cs.var(axis=1, ddof=1).mean()
        Bv = cs.mean(axis=1).var(ddof=1) * cs.shape[1]
        rhat[key] = round(float(np.sqrt(
            ((cs.shape[1] - 1) / cs.shape[1] * W + Bv / cs.shape[1])
            / W)), 4) if cs.shape[0] > 1 else None
    pe = np.asarray(xf["potential_energy"]) if "potential_energy" in xf \
        else None
    pe_chain = ([round(float(x), 1) for x in
                 np.asarray(pe).reshape(fg.shape[0], -1).mean(axis=1)]
                if pe is not None else None)
    naive = float(np.asarray(pk.fp_counts, float).sum() / consts.fp_ell_eff)
    diag = dict(
        divergences=div, target_accept=a.target_accept, fp_mode=a.fp_mode,
        fp_alpha0=a.fp_alpha0, fp_total_scale=a.fp_total_scale,
        t_scale=a.t_scale,
        sigma_N_post=[float(x) for x in np.percentile(sam["sigma_N"],
                                                      [16, 50, 84])],
        sigma_z_post=[float(x) for x in np.percentile(sam["sigma_z"],
                                                      [16, 50, 84])],
        fp_lam_total_over_naive=[
            round(float(x), 4) for x in
            np.percentile(np.asarray(sam["lam_fp"]).sum(axis=(1, 2)) / naive,
                          [16, 50, 84])],
        t_post_mean=[float(x) for x in np.asarray(sam["t"]).mean(axis=0)],
        t_post_in_prior_sd=[float(x) for x in
                            (np.asarray(sam["t"]).mean(axis=0)
                             / np.asarray(consts.t_sigma))],
        psi_c_mean_in_prior_sd=float(
            (np.asarray(sam["psi_c"]).mean(axis=0)
             / np.asarray(consts.sigma_hat)).mean()),
        perchain_estimand_medians=perchain, split_rhat=rhat,
        mean_potential_energy_per_chain=pe_chain)
    # G_A REAL-mode (ENFORCED, fail-closed): posterior-median predictive
    # level vs observed counts, within the guard's own tolerance.
    import jax as _jax
    idx_med = int(np.argsort(np.asarray(
        sam["theta_level"]))[len(sam["theta_level"]) // 2])
    th_med = jnp.asarray(np.asarray(sam["theta_pop"])[idx_med])
    pc_med = jnp.asarray(np.asarray(sam["psi_c"])[idx_med])
    t_med = jnp.asarray(np.asarray(sam["t"])[idx_med])
    lf_med = jnp.asarray(np.asarray(sam["lam_fp"])[idx_med])
    Cc = _jax.nn.sigmoid(consts.eta_hat + pc_med)[:, consts.b_to_cell]
    w_ = consts.g_bk * jnp.exp(th_med) * consts.dN_b[:, None]
    tpx = jnp.einsum("skcb,sb,bk->cks", Mg, Cc, w_) * consts.dX[None, :, :]
    fpx = (consts.fp_w * consts.fp_ell_eff
           * (1.0 - consts.fp_eta_c)[:, None, None]
           * jnp.exp(t_med[consts.kz_to_K])[None, :, None]
           * lf_med[:, None, :] * consts.fp_E[None, :, :])
    level = float((np.asarray(tpx) + np.asarray(fpx)).sum()
                  / np.asarray(pk.counts, float).sum())
    ga_real_ok = abs(level - 1.0) <= 0.06
    if not ga_real_ok:
        raise SystemExit(f"G_A REAL-mode FAILED: predictive level {level:.4f} "
                         "outside |level-1|<=0.06 — results withheld")
    fp_share = float(np.asarray(fpx).sum()
                     / (np.asarray(tpx).sum() + np.asarray(fpx).sum()))
    guards_summary = dict(
        subprocess_report={k: (v.get("status") if isinstance(v, dict)
                               else v) for k, v in greport.items()},
        G_A_truthpoint="N/A on real pack (zeros sentinel; fail-closed FAIL "
                       "recorded)",
        G_A_real_mode=dict(predictive_level=round(level, 4),
                           fp_share=round(fp_share, 4), tol=0.06,
                           status="PASS"))

    out = dict(pack=a.pack, pack_provenance=prov,
               n_draws=int(f_draws.shape[0]), guards=guards_summary,
               estimand=("POSTERIOR_MEDIAN_CI (committed reduce_f_posterior; "
                         "STATISTICAL interval only — named systematics "
                         "ledger v2.1 reported separately; NO response or "
                         "mock-to-real central-value correction applied; "
                         "BH unchanged)"),
               thresholds=rep, reporting_bins=binrep, diagnostics=diag,
               role=("FINAL guarded real-data HBI posterior — PI checkpoint-"
                     "10.8 explicit authorization; c=3300 observable-only "
                     "convention"))
    json.dump(out, open(a.out, "w"), indent=1)
    np.savez(a.out[:-5] + "_fdraws.npz", f=f_draws, ntrue_edges=ntrue,
             zf_edges=np.asarray(pk.zf_edges))
    print(json.dumps({k: out[k] for k in ("thresholds", "diagnostics")},
                     indent=1))


if __name__ == "__main__":
    main()
