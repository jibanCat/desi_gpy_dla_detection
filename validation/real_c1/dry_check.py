#!/usr/bin/env python
"""dry_check.py — the NO-LIKELIHOOD consistency run for the blind real C1 inputs.

CLASSIFICATION: **VALIDATION / PREPARATION ONLY.**

HARD RULE (PI ruling 2026-09-14 §17 — the real result stays blind until the
run): this script must never evaluate the real likelihood, run NUTS, or read a
real-data result.  It therefore:

  * loads the real pack and builds the fold tensors (shapes / finiteness /
    row-sum identities / live-cell mask only);
  * traces ``model_cc_ladder`` with ``counts=None``.  In
    ``CDDF_analysis/hbi_mcmc/fp_ladder.model_cc_ladder`` the observed site is
    ``numpyro.sample("counts", Poisson(mu), obs=counts)`` inside a
    ``handlers.mask``; with ``counts=None`` that site SAMPLES (prior
    predictive) and the real ``pack.counts`` array is never passed in, never
    read and never scored.  The real counts array is not loaded into the trace
    at all.
  * reports NOTHING derived from ``pack.counts`` except its shape / dtype /
    finiteness.

``fp_counts`` (the 89-event loa-0 block) IS used — it is a mock/twin
calibration product, not survey data.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

REAL_PACK = ("/home/mfho/lowz_clean_work_2026-09-12/posterior_campaign/"
             "packs/C1_pack.npz")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=REAL_PACK)
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--lam-imputations", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.handlers import trace, seed as nseed

    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    from CDDF_analysis.hbi_mcmc.fp_ladder import (
        model_cc_ladder, live_mask, lambda_calibration_posterior_quantiles)

    rec = dict(schema="absorber_ladder/real_c1_dry_check/v1",
               pack=os.path.abspath(a.pack),
               checked_utc=datetime.datetime.utcnow().isoformat() + "Z",
               likelihood_evaluated=False, mcmc_run=False,
               real_counts_passed_to_the_model=False)

    pk = load_pack(a.pack)
    consts, Mg = build_cc_tensors(pk)
    C, Kf, S = consts.n_c, consts.n_k, consts.n_s
    B, KK = consts.n_b, consts.n_kk
    live = live_mask(consts)
    dX = np.asarray(consts.dX, float)

    Mg_fix = np.asarray(np.load(os.path.join(a.inputs_dir, "Mg_B_real.npz"),
                                allow_pickle=True)["Mg"], float)
    zc = np.load(os.path.join(a.inputs_dir, "C_C1nsadd_real.npz"),
                 allow_pickle=True)
    C_fix = np.asarray(zc["C_fixed"], float)
    mu_extra = np.asarray(np.load(
        os.path.join(a.inputs_dir, "mu_extra_P6bcal_real.npz"),
        allow_pickle=True)["mu_extra"], float)

    # ---- shapes ----------------------------------------------------------
    rec["dims"] = dict(C=int(C), Kf=int(Kf), S=int(S), B=int(B), KK=int(KK))
    rec["shapes"] = dict(
        Mg_from_pack=list(np.asarray(Mg).shape), Mg_fixed=list(Mg_fix.shape),
        Mg_fixed_expected=[S, Kf, C, B],
        C_fixed=list(C_fix.shape), C_fixed_expected=[S, B],
        mu_extra=list(mu_extra.shape), mu_extra_expected=[C, Kf, S],
        counts=list(np.asarray(pk.counts).shape),
        dX=list(dX.shape), fp_counts=list(np.asarray(pk.fp_counts).shape),
        g_bk=list(np.asarray(consts.g_bk).shape),
        fp_E=list(np.asarray(consts.fp_E).shape))
    rec["shapes_ok"] = bool(
        list(Mg_fix.shape) == [S, Kf, C, B]
        and list(C_fix.shape) == [S, B]
        and list(mu_extra.shape) == [C, Kf, S]
        and list(np.asarray(Mg).shape) == [S, Kf, C, B])

    # ---- finiteness (no real-data VALUES reported) ------------------------
    rec["finiteness"] = dict(
        Mg_fixed=bool(np.all(np.isfinite(Mg_fix))),
        C_fixed=bool(np.all(np.isfinite(C_fix))),
        mu_extra=bool(np.all(np.isfinite(mu_extra))),
        dX=bool(np.all(np.isfinite(dX))),
        g_bk=bool(np.all(np.isfinite(np.asarray(consts.g_bk)))),
        eta_hat=bool(np.all(np.isfinite(np.asarray(consts.eta_hat)))),
        sigma_hat=bool(np.all(np.isfinite(np.asarray(consts.sigma_hat)))),
        fp_E=bool(np.all(np.isfinite(np.asarray(consts.fp_E)))),
        counts_finite=bool(np.all(np.isfinite(np.asarray(pk.counts)))))

    # ---- the row-sum identity sum_c Mg = phi ------------------------------
    zmg = np.load(os.path.join(a.inputs_dir, "Mg_B_real.npz"), allow_pickle=True)
    phi = np.asarray(zmg["phi_bsK"], float)
    kz = np.asarray(consts.kz_to_K, int)
    phi_k = np.einsum("bsK->sKb", phi)[:, kz, :]
    rec["row_sums"] = dict(
        max_abs_dev_sum_c_Mg_minus_phi=float(
            np.abs(Mg_fix.sum(axis=2) - phi_k).max()),
        phi_in_unit_interval=bool(phi.min() >= 0.0 and phi.max() <= 1.0),
        tol=1e-12)
    rec["row_sums"]["PASS"] = bool(
        rec["row_sums"]["max_abs_dev_sum_c_Mg_minus_phi"] <= 1e-12)

    # ---- the live-cell mask ----------------------------------------------
    obs_mask = np.broadcast_to((dX > 0)[None, :, :], (C, Kf, S))
    rec["live_cells"] = dict(
        n_live_strata=int(live.sum()), live_strata=[int(i) for i in np.where(live)[0]],
        n_live_ks_cells=int((dX > 0).sum()),
        n_masked_in_cells=int(obs_mask.sum()),
        n_total_cells=int(obs_mask.size),
        mu_extra_zero_off_live=bool(mu_extra[~obs_mask].sum() == 0.0),
        counts_zero_off_live=bool(np.asarray(pk.counts)[~obs_mask].sum() == 0))

    # ---- the M1CUT Lambda imputations (loa-0 calibration ONLY) ------------
    qs = lambda_calibration_posterior_quantiles(pk.fp_counts, consts.fp_ell_eff,
                                                a.lam_imputations)
    qs = np.asarray(qs, float)
    rec["lambda_imputations"] = dict(
        J=int(a.lam_imputations), n_returned=int(qs.size),
        all_finite=bool(np.all(np.isfinite(qs))),
        strictly_increasing=bool(np.all(np.diff(qs) > 0)),
        all_positive=bool(np.all(qs > 0)),
        source=("Gamma(N_FP + 1/2, ell_eff) — the loa-0 calibration posterior; "
                "the 89-event twin block, NOT survey data"))

    # ---- the fold, assembled by hand (no likelihood) ----------------------
    f0 = np.ones((B, Kf))                       # a neutral f; nothing fitted
    w = np.asarray(consts.g_bk, float) * f0 * np.asarray(consts.dN_b, float)[:, None]
    tp = np.einsum("skcb,sb,bk->cks", Mg_fix, C_fix, w) * dX[None, :, :]
    lam = float(qs[0]) * np.exp(np.asarray(
        __import__("CDDF_analysis.hbi_mcmc.fp_ladder", fromlist=["x"])
        .perks_log_share(np.asarray(pk.fp_counts, float), live)))
    lam = lam * live[None, :]
    fp = (float(consts.fp_w) * float(consts.fp_ell_eff)
          * (1.0 - np.asarray(consts.fp_eta_c))[:, None, None]
          * lam[:, None, :] * np.asarray(consts.fp_E)[None, :, :])
    mu = tp + fp + mu_extra
    rec["hand_assembled_fold"] = dict(
        tp_shape=list(tp.shape), finite=bool(np.all(np.isfinite(mu))),
        nonnegative=bool(mu.min() >= 0.0),
        positive_on_every_live_cell=bool(np.all(mu[obs_mask] > 0.0)),
        note=("assembled at f == 1 (a neutral placeholder, NOT a fit) purely "
              "to prove the einsum contracts and that mu is finite and "
              "strictly positive wherever the mask scores it"))

    # ---- the TRACE (prior predictive; counts=None) ------------------------
    with nseed(rng_seed=jax.random.PRNGKey(a.seed)):
        tr = trace(model_cc_ladder).get_trace(
            consts, Mg, counts=None, fp_counts=pk.fp_counts, ladder="M1CUT",
            t_sd=1.0, lam_fixed=float(qs[0]), fp_a0=None,
            mu_extra_fixed=mu_extra, C_fixed=C_fix, Mg_fixed=Mg_fix)
    sites = {k: dict(type=v["type"],
                     shape=list(np.shape(v["value"])),
                     is_observed=bool(v.get("is_observed", False)),
                     finite=bool(np.all(np.isfinite(np.asarray(v["value"],
                                                               float)))))
             for k, v in tr.items()}
    rec["trace"] = dict(
        n_sites=len(sites),
        counts_site_is_observed=bool(tr["counts"].get("is_observed", False)),
        sampled_sites=sorted(k for k, v in sites.items()
                             if v["type"] == "sample" and not v["is_observed"]),
        deterministic_sites=sorted(k for k, v in sites.items()
                                   if v["type"] == "deterministic"),
        all_site_values_finite=bool(all(v["finite"] for v in sites.values())),
        site_shapes={k: v["shape"] for k, v in sites.items()},
        note=("counts=None => the 'counts' site is NOT observed: the real "
              "counts array was never passed to the model. Only shapes and "
              "finiteness are recorded; no drawn value is reported."))
    if rec["trace"]["counts_site_is_observed"]:
        raise SystemExit("BLIND-RUN VIOLATION: the counts site is observed — "
                         "refusing (this dry check must not score real data)")

    rec["VERDICT"] = ("PASS" if (rec["shapes_ok"] and rec["row_sums"]["PASS"]
                                 and all(rec["finiteness"].values())
                                 and rec["hand_assembled_fold"]["finite"]
                                 and rec["hand_assembled_fold"][
                                     "positive_on_every_live_cell"]
                                 and rec["trace"]["all_site_values_finite"]
                                 and not rec["trace"]["counts_site_is_observed"])
                      else "FAIL")
    out = a.out or os.path.join(a.inputs_dir, "DRY_CHECK.json")
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print("VERDICT", rec["VERDICT"])
    print("sampled sites:", rec["trace"]["sampled_sites"])
    print("counts site observed:", rec["trace"]["counts_site_is_observed"])
    print("row-sum max dev:", rec["row_sums"]["max_abs_dev_sum_c_Mg_minus_phi"])
    print("out", out)
    return 0 if rec["VERDICT"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
