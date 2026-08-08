#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""THE GATED P1 REFOLD (PI ruling 2026-08-08 §5.2) — two-phase runner.

ORDER ENFORCED (the committed `predict_and_close.py` discipline):
`--phase predict` computes and WRITES the full calibration-side
prediction (per-bin and per-group, with the (C,K,M) covariance
projection and the G3 decomposition) BEFORE any closure statistic; NO
observed count enters the prediction file.  `--phase close` REFUSES to
run unless the prediction JSON exists, re-derives the prediction
in-memory and verifies it matches the written record, then evaluates
the UNCHANGED frozen closure statistics.  `--phase close` also REFUSES
to run twice: one gated refold, no confirmatory retries (§5.2).

What is folded (every piece committed/frozen; see `p1_refold_fold.py`):
  * (C_molly, K_natural-pairs) — the certified `p1_natpair_ck/v1`
    operator; C path byte-unchanged; K = empirical landing
    distributions of the frozen kernel event set (battery v2
    joint-operator construction), truth support N_true >= 19.5;
  * M_<19.5 — the EXPLICIT committed below-floor net-migration source
    (group totals gate-checked 4088/144/0); K never renormalized;
  * FP — the deployed FP fold, unchanged;
  * truth allocation, g_bk, live support — deployed, unchanged.

Closure statistics (pre-existing rule, nothing new ratified here):
  * Layer A: window [19.7, 21.6] chi^2/dof on the observed-bin
    marginal, variance = predicted mean (ratified threshold <= 3,
    conditional-only label).
  * Layer B: 3-group Mahalanobis T with the frozen parametric-bootstrap
    covariance recipe evaluated at the refolded plug-in (frozen sizes
    B=2000/2000 and seeds 41001/43001; cond>1e6 fallback rule), in TWO
    variants:
      - "frozen-construction": survey + FP-calibration noise only —
        the exact Phase-B Layer-B construction (p < 0.01 ratified for
        that construction, single-mock caveat disclosed);
      - "total": + the frozen (C,K,M) G-level calibration covariance
        (`p1_ckm_cov/v1`), added to the Mahalanobis metric AND to the
        null draws — the PI-mandated total uncertainty of this ruling.
  * Per-group standardized residuals against the total uncertainty.

DIAGNOSTIC ONLY (never a verdict input): the [21.3, 21.7) truth-region
sensitivity — the disclosed holdout residual delta applied to that
region's landing rows — quantifying how much of G3 rides on the
disclosed borderline bin.

MOCK ONLY (2lpt0 / mock-0 loa-124; within-realization).  No holdout
row is touched: the natural-pair kernel uses catalogue TPs, not
injection outcomes.  Nothing is spliced into production.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from p1_refold_fold import (                                   # noqa: E402
    FLOOR, P1RefoldGuardError, build_fold, build_p1_kernel, c_marginal,
    load_kernel_events, load_migration, mu_sig_p1, provenance,
)
from p1_ckm_cov import load_p1_ckm_cov                         # noqa: E402

PRED_JSON = os.path.join(_HERE, "p1_refold_prediction.json")
CLOSE_JSON = os.path.join(_HERE, "p1_refold_closure.json")
HOLDOUT_RESULT = os.path.join(_HERE, "p1_holdout_result.json")
PHASEB_TABLE = os.path.join(_HERE, "..", "..", "CDDF_analysis", "hbi_mcmc",
                            "closure_table_phaseB.json")
WIN = (19.7, 21.6)
SENS_TRUTH_REGION = (21.3, 21.7)


def _holdout_delta_2131():
    """hold - cal mean delta of the disclosed [21.3,21.7) battery bin,
    read from the CONSUMED holdout result record (read-only; no holdout
    data are accessed)."""
    r = json.load(open(HOLDOUT_RESULT))
    for t in r["primary_family"]["tests"]:
        if t["test"] == "mean[21.3,21.7)":
            return float(t["hold"] - t["cal"])
    raise P1RefoldGuardError("holdout record lacks mean[21.3,21.7)")


def _assemble():
    """Deterministic assembly of every prediction-side object."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from p1_refold_fold import PACK

    pk = load_pack(PACK)
    fold = build_fold(pk)
    E, truth, sparse, art, cache = load_kernel_events()
    mig = load_migration(fold["nhat_edges"])
    K_P1, kinfo = build_p1_kernel(E, fold, sparse)
    mu_sig = mu_sig_p1(K_P1, fold)                             # (C,Kf,S)
    mu_sig_c = c_marginal(mu_sig)
    fp_c = c_marginal(fold["mu_fp"])
    M_c = mig["M_c"]
    mu_c = mu_sig_c + M_c + fp_c
    ckm = load_p1_ckm_cov()
    Sigma_G = np.asarray(ckm["Sigma_G"], float)

    # per-truth-bin contributions to each group (for the decomposition);
    # the live (k,s) mask is applied per bin, exactly as in mu_sig_p1
    A = fold["A"]
    alloc = fold["alloc"].copy()
    pad = fold["ntrue_edges"][:-1] < FLOOR - 1e-9
    alloc[pad] = 0.0
    n_c = fold["nhat_edges"].size - 1
    contrib_bc = np.zeros((alloc.shape[0], n_c))
    live = fold["live"]                                        # (Kf,S)
    for b in range(alloc.shape[0]):
        w_ks = (fold["C_bs"][:, b][None, :] * fold["g_bk"][b][:, None]
                * alloc[b])                                    # (Kf,S)
        w_ks = np.where(live, w_ks, 0.0)
        # K_P1[s,k,c,b] -> sum_{k,s} K * w
        contrib_bc[b] = np.einsum("skc,ks->c", K_P1[:, :, :, b], w_ks)
    if np.max(np.abs(contrib_bc.sum(axis=0) - mu_sig_c)) > 1e-8:
        raise P1RefoldGuardError("per-truth-bin decomposition inconsistent")

    G_sig = A @ mu_sig_c
    G_M = A @ M_c
    G_fp = A @ fp_c
    G_pred = A @ mu_c
    sigma_G_ckm = np.sqrt(np.diag(Sigma_G))

    # identity-kernel completeness term (uniform-in-bin observed landing)
    ne = fold["nhat_edges"]
    nt = fold["ntrue_edges"]
    K_id = np.zeros_like(K_P1)
    for b in np.where(~pad)[0]:
        lo, hi = nt[b], min(nt[b + 1], ne[-1])
        ov = np.clip(np.minimum(ne[1:], hi) - np.maximum(ne[:-1], lo),
                     0, None)
        row = ov / (hi - lo)
        K_id[:, :, :, b] = row[None, None, :]
    mu_id_c = c_marginal(mu_sig_p1(K_id, fold))
    G_id = A @ mu_id_c                                         # completeness
    G_redis = G_sig - G_id                                     # kernel term

    # truth-region contributions to G3
    tr_lo = np.asarray(nt[:-1]); tr_hi = np.asarray(nt[1:])
    reg = (tr_lo >= SENS_TRUTH_REGION[0] - 1e-9) \
        & (tr_hi <= SENS_TRUTH_REGION[1] + 1e-9)
    G3_from_region = float((A[2] @ contrib_bc[reg].T).sum())
    per_bin_G3 = {f"[{tr_lo[b]},{tr_hi[b]})":
                  float(A[2] @ contrib_bc[b]) for b in np.where(~pad)[0]}

    # DIAGNOSTIC sensitivity: shift the region's landing rows by the
    # disclosed holdout mean delta (rebuild rows from shifted N-hat)
    delta = _holdout_delta_2131()
    E_shift = {k: (v.copy() if isinstance(v, np.ndarray) else v)
               for k, v in E.items()}
    m_reg = (E["N"] >= SENS_TRUTH_REGION[0]) \
        & (E["N"] < SENS_TRUTH_REGION[1])
    E_shift["NHAT"] = E["NHAT"] + np.where(m_reg, delta, 0.0)
    K_shift, _ = build_p1_kernel(E_shift, fold, sparse)
    mu_shift_c = c_marginal(mu_sig_p1(K_shift, fold))
    dG_shift = A @ (mu_shift_c - mu_sig_c)

    return dict(pk=pk, fold=fold, E=E, mig=mig, K_P1=K_P1, kinfo=kinfo,
                mu_sig_c=mu_sig_c, M_c=M_c, fp_c=fp_c, mu_c=mu_c,
                Sigma_G=Sigma_G, ckm=ckm, A=A,
                G_sig=G_sig, G_M=G_M, G_fp=G_fp, G_pred=G_pred,
                sigma_G_ckm=sigma_G_ckm, G_id=G_id, G_redis=G_redis,
                G3_from_region=G3_from_region, per_bin_G3=per_bin_G3,
                delta_2131=delta, dG_shift=dG_shift,
                contrib_bc=contrib_bc)


def phase_predict():
    if os.path.exists(CLOSE_JSON):
        raise SystemExit("REFUSED: closure already exists — the prediction "
                         "is never regenerated after closure (order rule).")
    t0 = time.time()
    a = _assemble()
    out = {
        "schema": "p1_refold_prediction/v1",
        "date": time.strftime("%Y-%m-%d"),
        "authorization": "PI ruling 2026-08-08 §5.2 (one gated refold; "
                         "prerequisite p1_ckm_cov/v1 passed its gates)",
        "note": ("CALIBRATION-SIDE PREDICTION, written before any closure "
                 "statistic. No observed count appears in this file."),
        "estimand": "p1_natpair_ck/v1 + explicit M_<19.5 + deployed FP",
        "support": ("fold truth >= 19.5 (primary certified >= 20.3; "
                    "[19.5,20.3) truth RESTRICTED low-boundary status); "
                    "observed [19.5,22.4); groups G1/G2/G3 frozen"),
        "groups": {
            "labels": ["G1[19.7,20.3)", "G2[20.3,21.0)", "G3[21.0,21.6)"],
            "mu_signal_CK": a["G_sig"].tolist(),
            "mu_migration_M": a["G_M"].tolist(),
            "mu_fp": a["G_fp"].tolist(),
            "mu_total": a["G_pred"].tolist(),
            "sigma_CKM_calibration": a["sigma_G_ckm"].tolist(),
        },
        "per_bin_mu": {
            "nhat_edges": a["fold"]["nhat_edges"].tolist(),
            "mu_signal_CK": a["mu_sig_c"].tolist(),
            "mu_migration_M": a["M_c"].tolist(),
            "mu_fp": a["fp_c"].tolist(),
            "mu_total": a["mu_c"].tolist(),
        },
        "G3_decomposition": {
            "completeness_identity_kernel": float(a["G_id"][2]),
            "response_kernel_redistribution": float(a["G_redis"][2]),
            "migration_source": float(a["G_M"][2]),
            "fp": float(a["G_fp"][2]),
            "total": float(a["G_pred"][2]),
            "sigma_CKM": float(a["sigma_G_ckm"][2]),
            "per_truth_bin_contribution_to_G3": a["per_bin_G3"],
            "truth_region_21p3_21p7_contribution": a["G3_from_region"],
        },
        "sensitivity_2131_DIAGNOSTIC_ONLY": {
            "note": ("the disclosed holdout residual delta applied to the "
                     "[21.3,21.7) truth region's landing rows; NEVER a "
                     "verdict input; the primary refold uses the "
                     "UNSHIFTED frozen operator"),
            "delta_dex_hold_minus_cal": a["delta_2131"],
            "dG_groups_if_region_shifted": a["dG_shift"].tolist(),
        },
        "kernel": {
            "n_live_events": int(len(a["E"]["N"])),
            "n_out_of_grid_landing": a["kinfo"]["n_out_of_grid"],
            "n_cells_marginal_inherited":
                int(np.sum(a["kinfo"]["sparse"][:14]
                           | (a["kinfo"]["n_cell"][:14] < 25))),
        },
        "migration": {
            "n_net_total": a["mig"]["n_net_total"],
            "n_out_of_grid": a["mig"]["n_out_of_grid"],
            "group_counts": a["G_M"].tolist(),
            "window_caveat": ("the frozen migration definition has no "
                              "analysis-window cut; its selected-row "
                              "universe is ~4% wider than the pack's "
                              "windowed catalogue — a <=~0.4%-of-G1 "
                              "one-sided allowance, disclosed, not "
                              "corrected (the source term is frozen)"),
        },
        "ckm_cov_artifact_sha256": str(
            a["ckm"]["provenance"].get("artifact_sha256", "")) or None,
        "provenance": provenance(),
        "rebuild_rel_err": a["fold"]["rebuild_rel_err"],
        "wall_s": round(time.time() - t0, 1),
    }
    with open(PRED_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print("groups mu_total:", np.round(a["G_pred"], 1).tolist())
    print("  = CK", np.round(a["G_sig"], 1).tolist(),
          "+ M", np.round(a["G_M"], 1).tolist(),
          "+ FP", np.round(a["G_fp"], 1).tolist())
    print("sigma_CKM:", np.round(a["sigma_G_ckm"], 1).tolist())
    print("G3 decomposition: completeness", round(out["G3_decomposition"][
        "completeness_identity_kernel"], 1),
        "kernel", round(out["G3_decomposition"][
            "response_kernel_redistribution"], 1),
        "M", 0.0, "FP", round(out["G3_decomposition"]["fp"], 1))
    print("wrote", PRED_JSON)


def phase_close():
    if not os.path.exists(PRED_JSON):
        raise SystemExit("REFUSED: prediction JSON absent — run "
                         "--phase predict first (order guard).")
    if os.path.exists(CLOSE_JSON):
        raise SystemExit("REFUSED: closure already exists — ONE gated "
                         "refold; no confirmatory retries (§5.2).")
    t0 = time.time()
    from CDDF_analysis.hbi_mcmc import gate_covariance as GC

    a = _assemble()
    pred = json.load(open(PRED_JSON))
    # the closure must evaluate EXACTLY the committed prediction
    if not np.allclose(pred["groups"]["mu_total"], a["G_pred"],
                       rtol=0, atol=1e-9):
        raise P1RefoldGuardError(
            "IMPLEMENTATION-INVALID: in-memory prediction != committed "
            "prediction JSON")

    fold, A = a["fold"], a["A"]
    obs_c = c_marginal(fold["obs_counts"])
    G_obs = A @ obs_c
    d_obs = G_obs - a["G_pred"]

    # ---- Layer A: window chi2/dof (frozen construction) ----------------
    ne = fold["nhat_edges"]
    wmask = (ne[:-1] >= WIN[0] - 1e-9) & (ne[1:] <= WIN[1] + 1e-9)
    z_c = (obs_c - a["mu_c"]) / np.sqrt(np.maximum(a["mu_c"], 1e-12))
    chi2_dof = float(np.sum(z_c[wmask] ** 2) / int(wmask.sum()))

    # ---- Layer B: frozen recipe at the refolded plug-in ----------------
    mu_sig_c, M_c, mu_c = a["mu_sig_c"], a["M_c"], a["mu_c"]
    Sigma_G = a["Sigma_G"]
    n0 = np.asarray(a["pk"].fp_counts, float)
    _, fp_fold, live3 = GC._fold_parts(a["pk"], resp_clamp="both")

    def fp_c_of(n0v):
        return c_marginal(np.where(live3, fp_fold(n0v), 0.0))

    B_cov, B_null = GC.N_COV_DRAWS, GC.N_NULL_DRAWS
    rng = np.random.default_rng(GC.SEED_COV)
    draws = np.empty((B_cov, 3))
    for r in range(B_cov):
        y_star = rng.poisson(np.clip(mu_c, 0, None))
        n0_star = rng.poisson(n0)
        draws[r] = A @ y_star - A @ (mu_sig_c + M_c + fp_c_of(n0_star))
    C_frozen = np.cov(draws, rowvar=False)

    results = {}
    for tag, Sig in (("frozen_construction", C_frozen),
                     ("total_with_CKM", C_frozen + Sigma_G)):
        ev = np.linalg.eigvalsh(Sig)
        cond = float(ev[-1] / max(ev[0], 1e-300))
        fallback = cond > GC.MAX_CONDITION_NUMBER
        z_g = d_obs / np.sqrt(np.diag(Sig))
        if fallback:
            results[tag] = dict(
                T_obs=float(np.max(np.abs(z_g))), p_value=None,
                fallback_1d=True, condition_number=cond,
                residual_z=z_g.tolist())
            continue
        Sinv = np.linalg.inv(Sig)
        T_obs = float(d_obs @ Sinv @ d_obs)
        rng_n = np.random.default_rng(GC.SEED_NULL)
        with_ckm = tag == "total_with_CKM"
        L = np.linalg.cholesky(
            Sigma_G + 1e-12 * np.eye(3) * max(Sigma_G.max(), 1e-300)) \
            if with_ckm else None
        T_null = np.empty(B_null)
        for r in range(B_null):
            y_star = rng_n.poisson(np.clip(mu_c, 0, None))
            n0_star = rng_n.poisson(n0)
            d = A @ y_star - A @ (mu_sig_c + M_c + fp_c_of(n0_star))
            if with_ckm:
                d = d - L @ rng_n.standard_normal(3)
            T_null[r] = float(d @ Sinv @ d)
        n_exceed = int(np.sum(T_null >= T_obs))
        p = (1 + n_exceed) / (B_null + 1)
        results[tag] = dict(
            T_obs=T_obs, p_value=p, p_is_bound=(n_exceed == 0),
            p_mc_error=float(np.sqrt(p * (1 - p) / (B_null + 1))),
            null_quantiles={f"q{int(100*q):02d}":
                            float(np.quantile(T_null, q))
                            for q in (0.05, 0.5, 0.95, 0.99)},
            null_mean=float(T_null.mean()),
            null_sd=float(T_null.std(ddof=1)),
            condition_number=cond, fallback_1d=False,
            covariance=[[float(v) for v in row] for row in Sig],
            residual_z=z_g.tolist(),
            seeds=dict(cov=GC.SEED_COV, null=GC.SEED_NULL,
                       B_cov=B_cov, B_null=B_null))

    # ---- Phase-B baseline (committed record, read-only) ----------------
    base = None
    try:
        tb = json.load(open(PHASEB_TABLE))
        for row in tb["rows"]:
            if "2lpt0" in row["pack"]:
                base = dict(residual=row["predictive"]["residual"],
                            residual_z=row["predictive"]["residual_z"],
                            T_obs=row["predictive"]["T_obs"],
                            p_value=row["predictive"]["p_value"])
    except Exception as e:                                     # noqa: BLE001
        base = {"unavailable": str(e)}

    frac_G3 = (None if not base or "unavailable" in base else
               float(1.0 - d_obs[2] / base["residual"][2]))

    sig_tot = np.sqrt(np.diag(C_frozen + Sigma_G))
    per_group = []
    for i, gname in enumerate(("G1[19.7,20.3)", "G2[20.3,21.0)",
                               "G3[21.0,21.6)")):
        per_group.append(dict(
            group=gname,
            prediction=float(a["G_pred"][i]),
            observed=float(G_obs[i]),
            residual=float(d_obs[i]),
            sigma_total=float(sig_tot[i]),
            z_total=float(d_obs[i] / sig_tot[i]),
            sigma_frozen_construction=float(np.sqrt(C_frozen[i, i])),
            sigma_CKM=float(a["sigma_G_ckm"][i]),
            relies_on=("(C,K) certified >= 20.3 truth + RESTRICTED "
                       "[19.5,20.3) truth rows + explicit M source"
                       if i == 0 else
                       "(C,K) certified + small explicit M (144)"
                       if i == 1 else
                       "(C,K) certified; M = 0 measured")))

    out = {
        "schema": "p1_refold_closure/v1",
        "date": time.strftime("%Y-%m-%d"),
        "authorization": "PI ruling 2026-08-08 §5.2 — ONE gated refold",
        "prediction_file": "p1_refold_prediction.json (verified equal "
                           "in-memory before closure)",
        "observed_total_live": float(obs_c.sum()),
        "groups_observed": G_obs.tolist(),
        "groups_predicted": a["G_pred"].tolist(),
        "groups_residual_obs_minus_pred": d_obs.tolist(),
        "per_group": per_group,
        "layerA_window_chi2_dof": chi2_dof,
        "layerA_threshold": "<= 3 (ratified, conditional-only label)",
        "layerB": results,
        "layerB_rule_note": (
            "p < 0.01 was RATIFIED (2026-08-06 decision 2) for the frozen "
            "multi-mock Layer-B construction; this refold is single-mock "
            "(2lpt0, within-realization by design) and the 'total' variant "
            "adds the (C,K,M) covariance per the 2026-08-08 ruling — both "
            "variants reported; construction difference disclosed"),
        "phaseB_baseline_2lpt0": base,
        "fraction_of_G3_discrepancy_explained": frac_G3,
        "per_bin": {
            "nhat_edges": ne.tolist(),
            "observed": obs_c.tolist(),
            "mu_total": a["mu_c"].tolist(),
            "z_poisson": z_c.tolist(),
            "window_mask": wmask.tolist(),
        },
        "G3_decomposition": pred["G3_decomposition"],
        "sensitivity_2131_DIAGNOSTIC_ONLY":
            pred["sensitivity_2131_DIAGNOSTIC_ONLY"],
        "provenance": provenance(),
        "wall_s": round(time.time() - t0, 1),
    }
    with open(CLOSE_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    print("obs   :", np.round(G_obs, 1).tolist())
    print("pred  :", np.round(a["G_pred"], 1).tolist())
    print("resid :", np.round(d_obs, 1).tolist())
    print("z_tot :", np.round(d_obs / sig_tot, 2).tolist())
    print(f"layerA window chi2/dof = {chi2_dof:.2f}")
    for tag, r in results.items():
        print(f"layerB[{tag}]: T={r['T_obs']:.2f} p={r.get('p_value')}")
    if frac_G3 is not None:
        print(f"fraction of Phase-B G3 discrepancy explained: {frac_G3:.3f}")
    print("wrote", CLOSE_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("predict", "close"), required=True)
    args = ap.parse_args()
    (phase_predict if args.phase == "predict" else phase_close)()


if __name__ == "__main__":
    main()
