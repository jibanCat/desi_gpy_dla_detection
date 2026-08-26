#!/usr/bin/env python
"""cc_real_ppc.py — posterior-predictive check (PPC) of a model_cc REAL-data
run, from its saved nuisance draws. PI ruling 2026-08-26 (B3 checkpoint,
item 2): a DIAGNOSTIC, not a freeze gate; result disclosed either way.

WHY THIS MODULE EXISTS. `evidence.ppc_block` computes mu through
`forward.fold_mu` — the Model-A fold with a `psi_k_delta` kernel-nuisance
site. The frozen real posterior was sampled under `model_cc`
(cc_posterior_validation): the count-conserving fold with the FIXED adopted
kernel (precomputed `Mg`) and no psi_k. A PPC of that posterior must fold
its draws through model_cc's own expression, so `cc_fold_mu` below is that
expression, copied verbatim and PINNED BY TEST to the rate of model_cc's
`counts` site (tests/test_cc_real_ppc.py). The PPC statistics themselves are
the committed `ppc_block` ones, reached through its additive `mu_draws`
argument; nothing statistical is re-derived here.

Also emitted: the same posterior-predictive comparison aggregated to the
ratified 0.2-dex REPORTING grain (edges 19.7:0.2:21.7, all z and per coarse
z block) — the grain the forward-closure record (memo B3) is stated on.
Presentation of the same replicated counts; no new statistic family.

Nothing here samples, changes the model, prior, calibration or posterior.

Usage:
  python -m CDDF_analysis.hbi_mcmc.cc_real_ppc --pack REAL_PACK_v2.npz
      --draws RUN_nuisance.npz --run RUN.json --out PPC.json
      [--n-rep-draws 300 --ppc-seed 0]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

NUISANCE_SITES = ("theta_pop", "psi_c", "t", "lam_fp", "f")
REPORT_EDGES = tuple(float(x) for x in np.round(np.arange(19.7, 21.7 + 1e-9, 0.2), 3))
Z_BLOCKS = ((2.0, 2.5), (2.5, 3.0), (3.0, 3.5))


def cc_fold_mu(consts, Mg, theta, psi_c, t, lam_fp):
    """model_cc's forward fold, verbatim (cc_posterior_validation.model_cc):
    count-conserving TP term with the FIXED adopted kernel + the FP block."""
    import jax
    import jax.numpy as jnp
    Cc = jax.nn.sigmoid(consts.eta_hat + psi_c)[:, consts.b_to_cell]  # (S,B)
    f = jnp.exp(theta)                                                # (B,Kf)
    w = consts.g_bk * f * consts.dN_b[:, None]                        # (B,Kf)
    tp = jnp.einsum("skcb,sb,bk->cks", Mg, Cc, w) * consts.dX[None, :, :]
    fp = (consts.fp_w * consts.fp_ell_eff
          * (1.0 - consts.fp_eta_c)[:, None, None]
          * jnp.exp(t[consts.kz_to_K])[None, :, None]
          * lam_fp[:, None, :] * consts.fp_E[None, :, :])
    return tp + fp


def nuisance_payload(sam_g):
    """The by-chain draws the fold needs (and f), nothing else."""
    return {k: np.asarray(sam_g[k]) for k in NUISANCE_SITES}


def flatten_by_chain(pay):
    """(chains, draws, ...) -> (chains*draws, ...), chain-major (as numpyro)."""
    return {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:])
            for k, v in pay.items()}


def mu_draws_cc(consts, Mg, flat, n_max=None, seed=0):
    """mu per posterior draw via cc_fold_mu; subsampling convention identical
    to evidence._mu_draws (sorted choice without replacement from rng(seed))."""
    import jax
    import jax.numpy as jnp
    n = int(np.asarray(flat["theta_pop"]).shape[0])
    idx = np.arange(n)
    if n_max is not None and n > n_max:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=n_max, replace=False))
    fold = jax.vmap(cc_fold_mu, in_axes=(None, None, 0, 0, 0, 0))
    mu = fold(consts, Mg,
              jnp.asarray(np.asarray(flat["theta_pop"], float)[idx]),
              jnp.asarray(np.asarray(flat["psi_c"], float)[idx]),
              jnp.asarray(np.asarray(flat["t"], float)[idx]),
              jnp.asarray(np.asarray(flat["lam_fp"], float)[idx]))
    return np.asarray(mu), idx


def _mid_p(rep, obs):
    return (rep > obs).mean(axis=0) + 0.5 * (rep == obs).mean(axis=0)


def report_grain_ppc(mu, obs, dX, nhat_edges, zf_edges, *, report_edges=REPORT_EDGES,
                     z_blocks=Z_BLOCKS, seed=0, pval_min=0.002):
    """Replicated counts vs observed on the 0.2-dex reporting grain.

    mu: (n, C, Kf, S); obs: (C, Kf, S); dX: (Kf, S) (cells with dX == 0 are
    structurally unobserved and excluded, as in the likelihood). For each z
    block: per-bin obs / predictive median / two-sided mid-p, and the
    discrepancy T/n = sum_b (y_b - mubar_b)^2 / mubar_b / n_b with mubar the
    predictive mean, evaluated on the observed counts (T_obs) and on every
    replicate (T_rep); posterior_predictive_p = P(T_rep >= T_obs).
    `T_obs_over_n` is the plug-in Poisson-only figure comparable in form (NOT
    in meaning: fitted, truthless) to the mock truth-fold chi2/dof record."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    obs = np.asarray(obs, float)
    mask = (np.asarray(dX, float) > 0)
    mu = np.where(mask[None, None], mu, 0.0)
    obs = np.where(mask[None], obs, 0.0)
    y_rep = rng.poisson(np.clip(mu, 0.0, None)).astype(float)
    nhat = np.asarray(nhat_edges, float)
    zf = np.asarray(zf_edges, float)
    edges = np.asarray(report_edges, float)
    cen = 0.5 * (nhat[:-1] + nhat[1:])
    rb = np.digitize(cen, edges) - 1
    rb[(cen < edges[0]) | (cen >= edges[-1])] = -1
    zc = 0.5 * (zf[:-1] + zf[1:])
    out = {"report_edges": [float(e) for e in edges], "blocks": []}
    for lo, hi in z_blocks:
        kmask = (zc >= lo) & (zc < hi)
        bins = []
        T_obs = 0.0
        T_rep = np.zeros(mu.shape[0])
        n_used = 0
        n_fail = 0
        for b in range(len(edges) - 1):
            cm = rb == b
            if not cm.any():
                continue
            o_b = float(obs[cm][:, kmask, :].sum())
            r_b = y_rep[:, cm][:, :, kmask, :].sum(axis=(1, 2, 3))
            m_b = mu[:, cm][:, :, kmask, :].sum(axis=(1, 2, 3))
            mubar = float(m_b.mean())
            if mubar <= 0.0:
                continue
            p = float(_mid_p(r_b, o_b))
            two = float(2.0 * min(p, 1.0 - p))
            n_fail += int(two < pval_min)
            T_obs += (o_b - mubar) ** 2 / mubar
            T_rep += (r_b - mubar) ** 2 / mubar
            n_used += 1
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "obs": o_b,
                         "mu_median": float(np.median(m_b)), "mu_mean": mubar,
                         "ratio_mu_over_obs": (mubar / o_b if o_b > 0 else None),
                         "p_mid": p, "p_two_sided": two})
        n_used = max(n_used, 1)
        out["blocks"].append({
            "z_lo": float(lo), "z_hi": float(hi), "n_bins": n_used,
            "n_bins_failed": n_fail, "bins": bins,
            "T_over_n": {"T_obs_over_n": float(T_obs / n_used),
                         "T_rep_over_n_median": float(np.median(T_rep) / n_used),
                         "T_rep_over_n_p2p5_97p5": [float(np.percentile(T_rep, 2.5) / n_used),
                                                    float(np.percentile(T_rep, 97.5) / n_used)],
                         "posterior_predictive_p": float((T_rep >= T_obs).mean())}})
    return out


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--draws", required=True, help="nuisance npz saved by cc_real_posterior")
    ap.add_argument("--run", required=True, help="the run's summary JSON (config echo)")
    ap.add_argument("--n-rep-draws", type=int, default=300)
    ap.add_argument("--ppc-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    from CDDF_analysis.hbi_mcmc.cc_real_posterior import _real_mode_gate
    from CDDF_analysis.hbi_mcmc import evidence as EV

    pk = load_pack(a.pack)
    _real_mode_gate(a.pack, pk)
    consts, Mg = build_cc_tensors(pk)
    z = np.load(a.draws)
    pay = {k: np.asarray(z[k]) for k in NUISANCE_SITES}
    flat = flatten_by_chain(pay)
    mu, idx = mu_draws_cc(consts, Mg, flat, n_max=a.n_rep_draws, seed=a.ppc_seed)
    blk = EV.ppc_block({"samples_by_chain": None}, pk, consts, n_rep_draws=a.n_rep_draws,
                       seed=a.ppc_seed, mu_draws=(mu, idx))
    grain = report_grain_ppc(mu, np.asarray(pk.counts, float), np.asarray(pk.dX, float),
                             np.asarray(pk.nhat_edges, float), np.asarray(pk.zf_edges, float),
                             seed=a.ppc_seed + 1)
    run = json.load(open(a.run))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = "UNKNOWN"
    out = {
        "role": ("POSTERIOR-PREDICTIVE CHECK of a model_cc real-data run — DIAGNOSTIC "
                 "(PI 2026-08-26, B3 item 2); not a freeze gate; the posterior, model, "
                 "prior and calibration are unchanged"),
        "pack": a.pack, "pack_sha256": _sha(a.pack),
        "draws": a.draws, "draws_sha256": _sha(a.draws),
        "run": a.run, "run_sha256": _sha(a.run),
        "run_diagnostics": run.get("diagnostics"), "run_guards": run.get("guards"),
        "run_thresholds": run.get("thresholds"),
        "n_draws_total": int(flat["theta_pop"].shape[0]),
        "n_rep_draws": int(len(idx)), "ppc_seed": a.ppc_seed,
        "policy": {"PPC_PVAL_MIN": EV.PPC_PVAL_MIN,
                   "PPC_MAX_FAILED_CELL_FRAC": EV.PPC_MAX_FAILED_CELL_FRAC,
                   "PPC_OMNIBUS_MIN": EV.PPC_OMNIBUS_MIN},
        "ppc_block": blk, "report_grain": grain, "code_commit": commit,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"PPC written {a.out}: cells {blk['n_cells']} failed {blk['n_cells_failed']} "
          f"omnibus p {blk['omnibus_chi2_discrepancy']['posterior_predictive_p']:.4f} "
          f"checks {blk['checks']}")
    for b in grain["blocks"]:
        print(f"  report grain z[{b['z_lo']},{b['z_hi']}): T_obs/n {b['T_over_n']['T_obs_over_n']:.2f} "
              f"p {b['T_over_n']['posterior_predictive_p']:.3f} bins failed {b['n_bins_failed']}/{b['n_bins']}")


if __name__ == "__main__":
    main()
