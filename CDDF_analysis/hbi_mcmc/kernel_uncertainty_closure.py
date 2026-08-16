#!/usr/bin/env python
"""kernel_uncertainty_closure.py — PI ruling item 3 (2026-08-16):
propagate the CALIBRATION-FIT uncertainty of the response kernel (and,
separately, the completeness fit) through the truth fold, and report the
closure statistic with predictive uncertainty — WITHOUT changing the HBI
science model, without tuning anything to data, and without touching the
ratified gate.

Mathematically: the model's own calibration priors are
    psi_k_delta ~ N(0, fitcov_sd^2)   (order-0 mu/sig response coefs,
                                       per response cell — the kernel phi)
    psi_c       ~ N(0, sigma_hat^2)   (molly-cell completeness logits)
and the predictive fold is
    p(mu | truth) = ∫ dpsi  fold_mu(theta_truth, psi) p(psi | calibration),
evaluated by Monte Carlo at the truth-equivalent point (theta = log
f_truth, log_t = 0, lam_fp = fp_counts/fp_ell_eff — exactly the
forward_selftest point). This uses ONLY machinery and priors already in
the model (fold_mu's psi arguments; the psi_k_delta/psi_c prior widths the
sampler itself uses); nothing is fit to any data here.

Reported per pack (2lpt0/london0/saclay0, ADOPTED config, clamp=both), per
resolution:
    fine    : observed 0.1-dex bins, full grid and window [19.7,21.6]
    report  : the RATIFIED 0.2-dex reporting bins (edges 19.7:0.2:21.7)
              inside the window — 'the actual Paper-1 reporting resolution'
    groups  : G1 [19.7,20.3) / G2 [20.3,21.0) / G3 [21.0,21.6)
and per uncertainty treatment:
    fixed   : var = mu (Poisson only; the ratified gate's form)
    +kernel : var = mu + Var_psi_k(mu)          (diag) and full-covariance
    +k+c    : var = mu + Var_{psi_k,psi_c}(mu)  (diag) and full-covariance
Full-covariance statistic: T/n with T = r^T (diag(mu) + Cov_psi)^{-1} r.

This is a DIAGNOSTIC REPORT for the PI (item 3). It does not alter
run_posterior.GATE and must not be cited as a gate result.

Env: gpdla-hbi. Usage: python -m CDDF_analysis.hbi_mcmc.kernel_uncertainty_closure
     [--packdir DIR] [--n-draws 400] [--out JSON]
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEF_PACKDIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "adopted_packs_20260816")
ADOPTED_TAG = "bw0p2_pad19p0_molly172"
MOCKS = ("2lpt0", "london0", "saclay0")
WIN = (19.7, 21.6)
REPORT_EDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)   # ratified 0.2-dex basis
GROUPS = ((19.7, 20.3), (20.3, 21.0), (21.0, 21.6))


def chi2_stats(obs, mu_bar, cov_psi, label):
    """Diagonal and full-covariance closure statistics on one binning."""
    r = obs - mu_bar
    var_diag = mu_bar + np.diag(cov_psi)
    n = len(obs)
    chi2_diag = float(np.sum(r * r / np.maximum(var_diag, 1e-12)) / n)
    Sig = np.diag(np.maximum(mu_bar, 1e-12)) + cov_psi
    try:
        T = float(r @ np.linalg.solve(Sig, r))
    except np.linalg.LinAlgError:
        T = float(r @ np.linalg.pinv(Sig) @ r)
    return dict(label=label, n_bins=n, chi2_dof_diag=round(chi2_diag, 3),
                chi2_dof_fullcov=round(T / n, 3))


def aggregate(mu_ckS, ne, edges):
    """Aggregate a (C,) c-marginal onto coarser N edges (exact 0.1->0.2)."""
    lo, hi = ne[:-1], ne[1:]
    out = np.zeros(len(edges) - 1)
    for j in range(len(edges) - 1):
        m = (lo >= edges[j] - 1e-9) & (hi <= edges[j + 1] + 1e-9)
        out[j] = mu_ckS[m].sum()
    return out


def run_pack(path, n_draws, rng, ensemble=None):
    import dataclasses
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu
    from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f

    pk = load_pack(path)
    consts = build_consts(pk, resp_clamp="both")
    f = truth_f(pk)
    theta = jnp.asarray(np.log(np.clip(np.asarray(f, float), 1e-300, None)))
    S, M = consts.n_s, consts.n_molly
    lam_fp = jnp.asarray(np.asarray(pk.fp_counts, float)
                         / float(pk.fp_ell_eff))
    zt = jnp.zeros(consts.n_kk)
    psi_c0 = jnp.zeros((S, M))
    fitcov_sd = np.asarray(consts.fitcov_sd, float)          # (2, SR, ZR)
    sigma_hat = np.asarray(consts.sigma_hat, float)          # (S, M)

    def fold(psi_k, psi_c):
        mu = fold_mu(theta, psi_c, jnp.asarray(psi_k), zt, lam_fp, consts)
        return np.asarray(mu).sum(axis=(1, 2))               # c-marginal

    mu0 = fold(np.zeros_like(fitcov_sd), psi_c0)
    obs = np.asarray(pk.counts, float).sum(axis=(1, 2))
    ne = np.asarray(pk.nhat_edges, float)

    ens = {}
    if ensemble is None:
        for tag, use_c in (("kernel", False), ("kernel_completeness", True)):
            draws = np.empty((n_draws, len(mu0)))
            for i in range(n_draws):
                pk_draw = rng.normal(0.0, 1.0, fitcov_sd.shape) * fitcov_sd
                pc_draw = (jnp.asarray(rng.normal(0.0, 1.0, sigma_hat.shape)
                                       * sigma_hat) if use_c else psi_c0)
                draws[i] = fold(pk_draw, pc_draw)
            ens[tag] = draws
    else:
        # FULL kernel-fit covariance (PI item 1): coefficient DELTAS from the
        # committed T-D resample-refit ensemble, applied to the pack's own
        # response surfaces (assert the pack carries the frozen point model).
        e = np.load(ensemble, allow_pickle=True)
        for nm, pknm in (("point_mu", "resp_mu_coef"),
                         ("point_sig", "resp_sig_coef"),
                         ("point_skew", "resp_skew_coef")):
            if not np.allclose(e[nm], np.asarray(getattr(pk, pknm), float),
                               atol=1e-10):
                raise SystemExit(f"pack {pknm} != ensemble point model — "
                                 "cannot apply coefficient deltas")
        n_e = e["mu_coef"].shape[0]
        take = min(n_draws, n_e)
        zeros_k = np.zeros_like(fitcov_sd)
        for tag, use_c in (("kernel_full", False),
                           ("kernel_full_completeness", True)):
            draws = np.empty((take, len(mu0)))
            for i in range(take):
                consts_i = dataclasses.replace(
                    consts,
                    resp_mu_coef=jnp.asarray(e["mu_coef"][i]),
                    resp_sig_coef=jnp.asarray(e["sig_coef"][i]),
                    resp_skew_coef=jnp.asarray(e["skew_coef"][i]))
                pc_draw = (jnp.asarray(rng.normal(0.0, 1.0, sigma_hat.shape)
                                       * sigma_hat) if use_c else psi_c0)
                mu = fold_mu(theta, pc_draw, jnp.asarray(zeros_k), zt,
                             lam_fp, consts_i)
                draws[i] = np.asarray(mu).sum(axis=(1, 2))
            ens[tag] = draws

    def stats_on(mask_or_edges, kind):
        out = {}
        if kind == "fine":
            m = mask_or_edges
            o, mu_b = obs[m], mu0[m]
            sel = o > 0
            o, mu_b = o[sel], mu_b[sel]
            covs = {t: np.cov(d[:, m][:, sel], rowvar=False)
                    for t, d in ens.items()}
            mu_bars = {t: d[:, m][:, sel].mean(axis=0) for t, d in ens.items()}
        else:
            edges = mask_or_edges
            o = aggregate(obs, ne, edges)
            mu_b = aggregate(mu0, ne, edges)
            covs, mu_bars = {}, {}
            for t, d in ens.items():
                agg = np.stack([aggregate(d[i], ne, edges)
                                for i in range(d.shape[0])])
                covs[t] = np.cov(agg, rowvar=False)
                mu_bars[t] = agg.mean(axis=0)
        out["fixed_poisson_only"] = chi2_stats(
            o, mu_b, np.zeros((len(o), len(o))), "fixed")
        for t in ens:
            out[t] = chi2_stats(o, mu_bars[t], np.atleast_2d(covs[t]), t)
        return out

    lo, hi = ne[:-1], ne[1:]
    res = dict(
        pack=os.path.basename(path),
        n_draws=n_draws,
        fine_full_grid=stats_on(np.ones(len(mu0), bool), "fine"),
        fine_window=stats_on((lo >= WIN[0] - 1e-9) & (hi <= WIN[1] + 1e-9),
                             "fine"),
        report_0p2dex_window=stats_on(REPORT_EDGES, "coarse"),
        groups=stats_on(np.asarray([g[0] for g in GROUPS] + [GROUPS[-1][1]]),
                        "coarse"),
        total_ratio=round(float(mu0.sum() / obs.sum()), 5))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packdir", default=DEF_PACKDIR)
    ap.add_argument("--n-draws", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(
        _HERE, "kernel_uncertainty_closure.json"))
    ap.add_argument("--ensemble", default=None,
                    help="kernel_fit_ensemble npz (full-covariance mode)")
    a = ap.parse_args()
    rng = np.random.default_rng(20260816)
    rows = []
    for mock in MOCKS:
        p = os.path.join(a.packdir,
                         f"modelA_pack_{mock}_{ADOPTED_TAG}.npz")
        print(f"[kuc] {mock} ...", flush=True)
        rows.append(run_pack(p, a.n_draws, rng, ensemble=a.ensemble))
        r = rows[-1]
        for res_name in ("fine_window", "report_0p2dex_window", "groups"):
            b = r[res_name]
            keys = [k for k in b if k != "fixed_poisson_only"]
            msg = f"  {res_name}: fixed {b['fixed_poisson_only']['chi2_dof_diag']}"
            for k in keys:
                msg += f" | {k} diag {b[k]['chi2_dof_diag']} full {b[k]['chi2_dof_fullcov']}"
            print(msg)

    def _git():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
        except Exception:
            return "unknown"

    out = dict(
        schema="kernel_uncertainty_closure/v1",
        role=("DIAGNOSTIC REPORT (PI item 3, 2026-08-16). NOT a gate result; "
              "run_posterior.GATE unchanged. Priors used are the model's own "
              "calibration priors (psi_k_delta ~ N(0,fitcov_sd), psi_c ~ "
              "N(0,sigma_hat)); truth-point fold; nothing tuned to data."),
        code_commit=_git(), seed=20260816,
        ratified_reference="chi2/dof <= 3 (PI decision 8) on the Poisson "
                           "form; the predictive forms below are the item-3 "
                           "test, reported separately",
        packs=rows)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[kuc] wrote {a.out}")


if __name__ == "__main__":
    main()
