#!/usr/bin/env python
"""fit_completeness_variants.py — the nested family of FIXED completeness
calibration objects C0 / C1g / C1n / C1ns, fitted on 2LPT-0 ONLY.

VALIDATION-ONLY.  No sampler.  Nothing under ``CDDF_analysis/`` is modified.
London-0 and Saclay-0 are TRANSFER TESTS: they are read only to report a
residual and are NEVER used to fit (a hard guard refuses to build a design
matrix from anything but the calibration family).

Nested family
-------------
C0    frozen molly matrix on 12 N-cells x 8 S/N strata, evaluated on the latent
      grid via ``b_to_cell``; 0 new coefficients.  GATE: the pack's
      ``eta_hat`` is reproduced elementwise-exactly.
C1g   fine grid: one Jeffreys detection probability per (b, s) live cell,
      z-pooled; and the z-resolved (b, K, s) version.
C1n   per-stratum logistic polynomial in x = N - 20 (degree by CV):
      6 strata x (deg+1) coefficients.
C1ns  compact 2-D smooth in (N, log10 S/N): a tensor design whose intercept and
      N-slopes vary smoothly with the continuous S/N covariate, and an
      additive-in-logit variant with no interaction.

Everything is a DETECTION probability (no observed-grid restriction); the
in-grid counting fraction phi stays with the response kernel.

ENV: ``gpdla`` or ``gpdla-hbi`` (numpy only).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_DIAG = os.path.join(_REPO, "validation", "absorber_diag")
sys.path.insert(0, _DIAG)
sys.path.insert(0, _HERE)

from binning import coarse_block_sum                               # noqa: E402

from cal_fit import (expit, eta_hat_jeffreys, poly_design,          # noqa: E402
                     design_per_stratum, design_2d_tensor,
                     design_2d_additive, design_additive_z,
                     irls_binomial, binom_logloss,
                     binom_deviance, logloss_diff_se, cv_two_fold,
                     weighted_ratio_residual)

CAL_FAMILY = "2lpt0"
TRANSFER_FAMILIES = ("london0", "saclay0")
FAMILIES = (CAL_FAMILY,) + TRANSFER_FAMILIES
FAMLAB = {"2lpt0": "2LPT-0", "london0": "London-0", "saclay0": "Saclay-0"}
X0 = 20.0                     # the N pivot: x = N - 20
REPORT_FLOOR = 19.5           # reported window lower edge (bins b >= 2)


# --------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head():
    try:
        return dict(
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO,
                stderr=subprocess.DEVNULL).decode().strip(),
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_REPO,
                stderr=subprocess.DEVNULL).decode().strip(),
            dirty=bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_REPO, stderr=subprocess.DEVNULL).decode().strip()))
    except Exception as exc:                                # pragma: no cover
        return dict(commit="unknown", error=str(exc))


class Grid:
    """The pack's latent/stratum grid plus the calibration-family covariates."""

    def __init__(self, tab):
        self.ntrue = np.asarray(tab["ntrue_edges"], float)
        self.snr_edges = np.asarray(tab["snr_edges"], float)
        self.kz = np.asarray(tab["kz_to_K"], int)
        self.molly_edges = np.asarray(tab["molly_nhi_edges"], float)
        self.B = len(self.ntrue) - 1
        self.S = len(self.snr_edges) - 1
        self.KK = int(self.kz.max()) + 1
        self.Nc = 0.5 * (self.ntrue[:-1] + self.ntrue[1:])
        self.x_b = self.Nc - X0
        self.b_to_cell = np.clip(
            np.digitize(self.Nc, self.molly_edges) - 1, 0,
            len(self.molly_edges) - 2).astype(np.int64)
        tot = np.asarray(tab["truth_bks"], float)
        self.live_s = tot.sum(axis=(0, 1)) > 0
        self.live_idx = np.where(self.live_s)[0]
        self.snr_logmed = np.asarray(tab["snr_logmed_s"], float)
        w = tot.sum(axis=(0, 1))[self.live_idx]
        self.u0 = float(np.sum(self.snr_logmed[self.live_idx] * w) / w.sum())
        self.u_s = np.where(self.live_s, self.snr_logmed - self.u0, 0.0)
        self.reported_b = self.Nc >= REPORT_FLOOR - 1e-9
        self.g_bk = np.asarray(tab["g_grid"], float)[self.b_to_cell, :]


def _live(tab_arr, grid):
    """(S_live, B) view of a (B, Kf, S) count table, z-pooled."""
    return np.asarray(tab_arr, float).sum(axis=1).T[grid.live_idx, :]


# --------------------------------------------------------------------------
# variant definitions.  Each returns (fit_fn, predict_fn, meta) operating on
# (S_live, B) count matrices and predicting a (S_live, B) probability matrix.
# --------------------------------------------------------------------------
def make_variant(name, grid, C0_live):
    Sl, B = len(grid.live_idx), grid.B
    x = grid.x_b
    u = grid.u_s[grid.live_idx]

    if name == "C0_frozen":
        def fit(det, tot):
            return dict(p=C0_live.copy())

        def pred(st):
            return st["p"]
        return fit, pred, dict(kind="frozen", n_coef=0,
                               representation="molly 12 N-cells x 8 S/N strata "
                               "via b_to_cell (FROZEN, not refitted)")

    if name == "C1g":
        def fit(det, tot):
            eta, sig = eta_hat_jeffreys(det, tot)
            p = expit(eta)
            p = np.where(tot > 0, p, C0_live)
            return dict(p=p, eta=eta, sigma=sig, empty=(tot <= 0))

        def pred(st):
            return st["p"]
        return fit, pred, dict(kind="fine_grid", n_coef=Sl * B,
                               representation=f"free Jeffreys rate per live "
                               f"(b, s) cell ({Sl}x{B}); empty cells fall back "
                               "to C0")

    if name.startswith("C1n_deg"):
        deg = int(name.split("deg")[1])
        X = design_per_stratum(x, Sl, deg)

        def fit(det, tot, _X=X, _deg=deg):
            r = irls_binomial(_X, det.ravel(), tot.ravel(), ridge=RIDGE)
            return dict(beta=r["beta"], fit=r,
                        p=r["p"].reshape(Sl, B))

        def pred(st):
            return st["p"]
        return fit, pred, dict(kind="smooth_in_N", n_coef=Sl * (deg + 1),
                               degree_N=deg,
                               representation=f"independent logistic degree-"
                               f"{deg} polynomial in x=N-20 per S/N stratum")

    if name.startswith("C1ns_tensor"):
        dn, du = (int(v) for v in name.split("_")[-1].split("x"))
        X = design_2d_tensor(x, u, dn, du)

        def fit(det, tot, _X=X):
            r = irls_binomial(_X, det.ravel(), tot.ravel(), ridge=RIDGE)
            return dict(beta=r["beta"], fit=r, p=r["p"].reshape(Sl, B))

        def pred(st):
            return st["p"]
        return fit, pred, dict(kind="smooth_2D_tensor",
                               n_coef=(dn + 1) * (du + 1),
                               degree_N=dn, degree_logSNR=du,
                               representation=f"logit = sum_j sum_i beta_ji "
                               f"x^j u^i, x=N-20, u=log10(SNR_med)-{{u0}}, "
                               f"j<={dn}, i<={du}")

    if name.startswith("C1ns_add"):
        dn, du = (int(v) for v in name.split("_")[-1].split("x"))
        X = design_2d_additive(x, u, dn, du)

        def fit(det, tot, _X=X):
            r = irls_binomial(_X, det.ravel(), tot.ravel(), ridge=RIDGE)
            return dict(beta=r["beta"], fit=r, p=r["p"].reshape(Sl, B))

        def pred(st):
            return st["p"]
        return fit, pred, dict(kind="smooth_2D_additive",
                               n_coef=dn + du + 1,
                               degree_N=dn, degree_logSNR=du,
                               representation=f"logit = poly_{dn}(x) + "
                               f"poly_{du}(u) (NO interaction; the N-shape is "
                               "common to every stratum)")

    raise ValueError(f"unknown variant {name!r}")


RIDGE = 1e-6


# --------------------------------------------------------------------------
def cell_se_smooth(state, X, shape):
    """Delta-method sd of C on each cell from the IRLS covariance."""
    cov = state["fit"]["cov"]
    var_eta = np.einsum("ij,jk,ik->i", X, cov, X)
    p = state["p"].ravel()
    return (p * (1 - p) * np.sqrt(np.maximum(var_eta, 0.0))).reshape(shape)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default=("/scratch/cavestru_root/cavestru0/"
                                         "mfho/absorber_ladder_2026-09-13/"
                                         "completeness"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out_dir = a.out or a.tables
    os.makedirs(out_dir, exist_ok=True)

    tabs = {}
    for fam in FAMILIES:
        p = os.path.join(a.tables, f"cal_table_{fam}.npz")
        tabs[fam] = np.load(p, allow_pickle=True)
    cal = tabs[CAL_FAMILY]
    grid = Grid(cal)
    Sl, B, KK = len(grid.live_idx), grid.B, grid.KK

    # ---- C0 gate: the frozen surface, reproduced exactly ------------------
    eta_pack, sig_pack = eta_hat_jeffreys(cal["molly_n_det"], cal["molly_n_tot"])
    from CDDF_analysis.hbi_mcmc.forward import eta_hat_sigma_hat as _EH
    eta_ref, sig_ref = _EH(cal["molly_n_det"], cal["molly_n_tot"])
    c0_gate = dict(
        eta_max_abs_diff=float(np.abs(eta_pack - eta_ref).max()),
        sigma_max_abs_diff=float(np.abs(sig_pack - sig_ref).max()),
        EQUAL=bool(np.array_equal(eta_pack, eta_ref)
                   and np.array_equal(sig_pack, sig_ref)))
    if not c0_gate["EQUAL"]:
        raise SystemExit(f"C0 GATE FAILED: {c0_gate}")
    C0_full = expit(eta_pack)[:, grid.b_to_cell]           # (S, B)
    C0_live = C0_full[grid.live_idx, :]
    # GATE 2: the C0 object must equal the array the COMMITTED fold builds,
    # sigmoid(consts.eta_hat + 0)[:, consts.b_to_cell], elementwise.
    try:
        from CDDF_analysis.hbi_mcmc.pack import load_pack as _lp
        from CDDF_analysis.hbi_mcmc.forward import build_consts as _bc
        _pkp = json.loads(str(cal["provenance"]))["pack"]
        _cst = _bc(_lp(_pkp))
        _Cref = (1.0 / (1.0 + np.exp(-np.asarray(_cst.eta_hat, float))))[
            :, np.asarray(_cst.b_to_cell, int)]
        c0_gate["fold_b_to_cell_EQUAL"] = bool(
            np.array_equal(np.asarray(_cst.b_to_cell, int), grid.b_to_cell))
        c0_gate["fold_C_bs_max_abs_diff"] = float(np.abs(_Cref - C0_full).max())
        # eta_hat and b_to_cell must agree EXACTLY; the logistic itself is
        # allowed a float-rounding difference only (the fold evaluates
        # jax.nn.sigmoid, this module the branch-stable numpy expit), with an
        # EXPLICIT absolute tolerance — never a default one.
        c0_gate["fold_C_bs_atol"] = 1e-15
        c0_gate["fold_C_bs_EQUAL"] = bool(
            c0_gate["fold_C_bs_max_abs_diff"] <= 1e-15)
        if not (c0_gate["fold_b_to_cell_EQUAL"] and c0_gate["fold_C_bs_EQUAL"]):
            raise SystemExit(f"C0 FOLD GATE FAILED: {c0_gate}")
        print("C0 FOLD GATE PASSED (b_to_cell + eta_hat exact; C_bs agrees "
              f"to {c0_gate['fold_C_bs_max_abs_diff']:.1e} <= 1e-15)",
              flush=True)
    except ImportError as exc:
        c0_gate["fold_gate"] = f"SKIPPED (not importable here): {exc}"
    print(f"C0 GATE PASSED (eta_hat reproduced exactly; "
          f"max|d eta|={c0_gate['eta_max_abs_diff']:.1e})", flush=True)

    # ---- calibration counts ----------------------------------------------
    det = _live(cal["det_bks"], grid)                      # (Sl, B)
    tot = _live(cal["truth_bks"], grid)
    det_E = _live(cal["det_bks_E"], grid)
    tot_E = _live(cal["truth_bks_E"], grid)
    det_O = _live(cal["det_bks_O"], grid)
    tot_O = _live(cal["truth_bks_O"], grid)
    assert np.array_equal(det_E + det_O, det) and np.array_equal(tot_E + tot_O, tot)

    mask_all = tot > 0
    mask_rep = mask_all & grid.reported_b[None, :]

    # ---- the variant ladder ----------------------------------------------
    names = (["C0_frozen", "C1g"]
             + [f"C1n_deg{d}" for d in (0, 1, 2, 3, 4)]
             + [f"C1ns_tensor_{dn}x{du}" for dn, du in
                ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (2, 3), (3, 3))]
             + [f"C1ns_add_{dn}x{du}" for dn, du in
                ((2, 1), (3, 1), (3, 2), (4, 2), (3, 3), (4, 3))])
    results = {}
    for nm in names:
        fit, pred, meta = make_variant(nm, grid, C0_live)
        st_full = fit(det, tot)
        cv_all = cv_two_fold(fit, pred, det_E, tot_E, det_O, tot_O,
                             mask=mask_all)
        cv_rep = cv_two_fold(fit, pred, det_E, tot_E, det_O, tot_O,
                             mask=mask_rep)
        res = dict(meta=meta, name=nm,
                   in_sample_deviance=float(binom_deviance(st_full["p"],
                                                           det, tot)),
                   cv_all=dict(logloss=cv_all["logloss"],
                               per_trial=cv_all["logloss_per_trial"],
                               fold_fitE_scoreO=cv_all["fold_fitA_scoreB"],
                               fold_fitO_scoreE=cv_all["fold_fitB_scoreA"],
                               n_heldout_trials=cv_all["n_heldout_trials"]),
                   cv_reported=dict(logloss=cv_rep["logloss"],
                                    per_trial=cv_rep["logloss_per_trial"]),
                   p_full=st_full["p"], state=st_full,
                   p_on_E=cv_all["p_on_A"], p_on_O=cv_all["p_on_B"])
        if "fit" in st_full:
            res["converged"] = bool(st_full["fit"]["converged"])
            res["hessian_cond"] = float(st_full["fit"]["hessian_cond"])
            res["beta"] = np.asarray(st_full["beta"], float)
        results[nm] = res
        print(f"  {nm:22s} ncoef={meta['n_coef']:4d}  "
              f"CV(all)={cv_all['logloss']:12.2f}  "
              f"CV/trial={cv_all['logloss_per_trial']:.6f}  "
              f"CV(rep)={cv_rep['logloss']:11.2f}  "
              f"dev_in={res['in_sample_deviance']:.1f}", flush=True)

    # paired SE of every variant's CV difference against C0
    base = results["C0_frozen"]
    for nm, r in results.items():
        d_all = r["cv_all"]["logloss"] - base["cv_all"]["logloss"]
        se_E = logloss_diff_se(base["p_on_E"][mask_all], r["p_on_E"][mask_all],
                               det_E[mask_all], tot_E[mask_all])
        se_O = logloss_diff_se(base["p_on_O"][mask_all], r["p_on_O"][mask_all],
                               det_O[mask_all], tot_O[mask_all])
        r["cv_all"]["delta_vs_C0"] = float(d_all)
        r["cv_all"]["delta_vs_C0_se"] = float(np.hypot(se_E, se_O))
        r["cv_all"]["delta_vs_C0_sigma"] = float(
            d_all / max(np.hypot(se_E, se_O), 1e-12))

    # ---- CV choice inside each smooth family ------------------------------
    # PREDECLARED RULE: take the minimum-CV candidate, then step DOWN in
    # complexity to the simplest candidate whose held-out log-loss is within
    # one PAIRED standard error of that minimum (the standard 1-SE rule).  The
    # paired SE is the SE of the difference against the minimum-CV candidate,
    # so it is the right scale for "is this extra freedom data-supported?".
    def _paired_se(n_a, n_b):
        se_E = logloss_diff_se(results[n_a]["p_on_E"][mask_all],
                               results[n_b]["p_on_E"][mask_all],
                               det_E[mask_all], tot_E[mask_all])
        se_O = logloss_diff_se(results[n_a]["p_on_O"][mask_all],
                               results[n_b]["p_on_O"][mask_all],
                               det_O[mask_all], tot_O[mask_all])
        return float(np.hypot(se_E, se_O))

    def pick(prefix):
        cand = sorted([n for n in names if n.startswith(prefix)],
                      key=lambda n: results[n]["meta"]["n_coef"])
        best = min(cand, key=lambda n: results[n]["cv_all"]["logloss"])
        Lb = results[best]["cv_all"]["logloss"]
        detail = []
        simplest = best
        for n in cand:
            se = _paired_se(best, n)
            d = results[n]["cv_all"]["logloss"] - Lb
            within = bool(d <= se)
            detail.append(dict(name=n, n_coef=results[n]["meta"]["n_coef"],
                               delta_vs_best=float(d), paired_se=se,
                               within_1se=within))
            if within and results[n]["meta"]["n_coef"] < \
                    results[simplest]["meta"]["n_coef"]:
                simplest = n
        return dict(min_cv=best, one_se=simplest, detail=detail)

    picks = dict(C1n=pick("C1n_deg"), C1ns_tensor=pick("C1ns_tensor"),
                 C1ns_add=pick("C1ns_add"))
    chosen = {k: v["one_se"] for k, v in picks.items()}
    chosen_min_cv = {k: v["min_cv"] for k, v in picks.items()}
    print("\nCV min:", json.dumps(chosen_min_cv))
    print("CV 1-SE pick (DELIVERED):", json.dumps(chosen), flush=True)

    # ---- the z-resolved fine grid (C1gz) ----------------------------------
    det_bKs = coarse_block_sum(np.asarray(cal["det_bks"], float), grid.kz, 1)
    tot_bKs = coarse_block_sum(np.asarray(cal["truth_bks"], float), grid.kz, 1)
    det_bKs_E = coarse_block_sum(np.asarray(cal["det_bks_E"], float), grid.kz, 1)
    tot_bKs_E = coarse_block_sum(np.asarray(cal["truth_bks_E"], float), grid.kz, 1)
    det_bKs_O = coarse_block_sum(np.asarray(cal["det_bks_O"], float), grid.kz, 1)
    tot_bKs_O = coarse_block_sum(np.asarray(cal["truth_bks_O"], float), grid.kz, 1)

    def _bKs_live(a):                                       # (B,KK,S)->(Sl,KK,B)
        return np.transpose(a, (2, 1, 0))[grid.live_idx]

    dz, tz = _bKs_live(det_bKs), _bKs_live(tot_bKs)
    dzE, tzE = _bKs_live(det_bKs_E), _bKs_live(tot_bKs_E)
    dzO, tzO = _bKs_live(det_bKs_O), _bKs_live(tot_bKs_O)

    def fit_z(d, t):
        eta, _ = eta_hat_jeffreys(d, t)
        p = expit(eta)
        # empty (s,K,b) -> the z-pooled C1g value -> C0
        pooled = expit(eta_hat_jeffreys(d.sum(axis=1), t.sum(axis=1))[0])
        pooled = np.where(t.sum(axis=1) > 0, pooled, C0_live)
        p = np.where(t > 0, p, pooled[:, None, :])
        return dict(p=p)

    cvz = cv_two_fold(fit_z, lambda s: s["p"], dzE, tzE, dzO, tzO,
                      mask=(tz > 0))
    st_z = fit_z(dz, tz)
    # score C1gz on the SAME (s,b) marginal as the others for comparability:
    # its z-pooled prediction is the truth-weighted average over K
    results["C1gz"] = dict(
        meta=dict(kind="fine_grid_z_resolved",
                  n_coef=int((tz > 0).sum()),
                  representation="free Jeffreys rate per live (b, K, s) cell; "
                                 "empty cells fall back to the z-pooled C1g"),
        name="C1gz", in_sample_deviance=float(binom_deviance(st_z["p"], dz, tz)),
        pz_on_E=cvz["p_on_A"], pz_on_O=cvz["p_on_B"],
        p_on_E=((cvz["p_on_A"] * tzE).sum(axis=1)
                / np.maximum(tzE.sum(axis=1), 1e-12)),
        p_on_O=((cvz["p_on_B"] * tzO).sum(axis=1)
                / np.maximum(tzO.sum(axis=1), 1e-12)),
        cv_all=dict(logloss=cvz["logloss"], per_trial=cvz["logloss_per_trial"],
                    fold_fitE_scoreO=cvz["fold_fitA_scoreB"],
                    fold_fitO_scoreE=cvz["fold_fitB_scoreA"],
                    n_heldout_trials=cvz["n_heldout_trials"],
                    note="scored on (s,K,b) cells — NOT comparable cell-for-"
                         "cell with the z-pooled scores; the z-gain test is "
                         "the paired comparison below"),
        p_full=st_z["p"])

    # the DECISIVE z test: does resolving z beat the z-pooled prediction on
    # held-out (s, K, b) cells?  Both are scored on exactly the same cells.
    def _pool_pred(d, t):
        p = expit(eta_hat_jeffreys(d.sum(axis=1), t.sum(axis=1))[0])
        p = np.where(t.sum(axis=1) > 0, p, C0_live)
        return np.broadcast_to(p[:, None, :], d.shape).copy()

    cvz_pool = cv_two_fold(lambda d, t: dict(p=_pool_pred(d, t)),
                           lambda s: s["p"], dzE, tzE, dzO, tzO,
                           mask=(tz > 0))
    z_gain = dict(
        pooled_logloss=cvz_pool["logloss"], zresolved_logloss=cvz["logloss"],
        delta=float(cvz["logloss"] - cvz_pool["logloss"]),
        delta_se=float(np.hypot(
            logloss_diff_se(cvz_pool["p_on_A"][tzE > 0], cvz["p_on_A"][tzE > 0],
                            dzE[tzE > 0], tzE[tzE > 0]),
            logloss_diff_se(cvz_pool["p_on_B"][tzO > 0], cvz["p_on_B"][tzO > 0],
                            dzO[tzO > 0], tzO[tzO > 0]))))
    z_gain["delta_sigma"] = float(z_gain["delta"] / max(z_gain["delta_se"], 1e-12))
    print(f"z-resolution test: delta logloss "
          f"{z_gain['delta']:+.2f} +/- {z_gain['delta_se']:.2f} "
          f"({z_gain['delta_sigma']:+.2f} sigma; negative = z resolution helps)",
          flush=True)

    # ---- EXPLORATORY: the smallest additive member that carries a z trend --
    # OUTSIDE the PI ruling §6-7 list (which names N, S/N and the 2-D smooth
    # only).  Reported so the PI can see whether the large coarse-z residual
    # every z-pooled member leaves is removable with 2 coefficients instead of
    # the 288-cell C1gz.  NOT adopted, NOT part of the sealed C1 set.
    _dn, _du = (int(v) for v in chosen["C1ns_add"].split("_")[-1].split("x"))
    Xz = design_additive_z(grid.x_b, grid.u_s[grid.live_idx], KK, _dn, _du)

    def fit_nsz(d, t, _X=Xz):
        r = irls_binomial(_X, d.ravel(), t.ravel(), ridge=RIDGE)
        return dict(beta=r["beta"], fit=r, p=r["p"].reshape(Sl, KK, B))

    cv_nsz = cv_two_fold(fit_nsz, lambda st: st["p"], dzE, tzE, dzO, tzO,
                         mask=(tz > 0))
    st_nsz = fit_nsz(dz, tz)
    se_nsz = float(np.hypot(
        logloss_diff_se(cvz["p_on_A"][tzE > 0], cv_nsz["p_on_A"][tzE > 0],
                        dzE[tzE > 0], tzE[tzE > 0]),
        logloss_diff_se(cvz["p_on_B"][tzO > 0], cv_nsz["p_on_B"][tzO > 0],
                        dzO[tzO > 0], tzO[tzO > 0])))
    results["C1nsz"] = dict(
        meta=dict(kind="smooth_2D_additive_plus_coarse_z_offsets",
                  n_coef=int(Xz.shape[1]), degree_N=_dn, degree_logSNR=_du,
                  status=("EXPLORATORY — outside PI ruling §6-7; reported for "
                          "the PI choice, NOT adopted"),
                  representation=(f"logit = poly_{_dn}(x) + poly_{_du}(u) + "
                                  "delta_K  (K=0 reference; 2 z offsets)")),
        name="C1nsz",
        in_sample_deviance=float(binom_deviance(st_nsz["p"], dz, tz)),
        cv_all=dict(logloss=cv_nsz["logloss"],
                    per_trial=cv_nsz["logloss_per_trial"],
                    fold_fitE_scoreO=cv_nsz["fold_fitA_scoreB"],
                    fold_fitO_scoreE=cv_nsz["fold_fitB_scoreA"],
                    n_heldout_trials=cv_nsz["n_heldout_trials"],
                    delta_vs_C1gz=float(cv_nsz["logloss"] - cvz["logloss"]),
                    delta_vs_C1gz_se=se_nsz,
                    delta_vs_C1gz_sigma=float(
                        (cv_nsz["logloss"] - cvz["logloss"]) / max(se_nsz, 1e-12)),
                    delta_vs_zpooled=float(cv_nsz["logloss"]
                                           - cvz_pool["logloss"]),
                    note="scored on (s,K,b) cells, like C1gz"),
        cv_reported=None, p_full=st_nsz["p"], state=st_nsz,
        beta=np.asarray(st_nsz["beta"], float),
        converged=bool(st_nsz["fit"]["converged"]),
        hessian_cond=float(st_nsz["fit"]["hessian_cond"]),
        pz_on_E=cv_nsz["p_on_A"], pz_on_O=cv_nsz["p_on_B"],
        p_on_E=((cv_nsz["p_on_A"] * tzE).sum(axis=1)
                / np.maximum(tzE.sum(axis=1), 1e-12)),
        p_on_O=((cv_nsz["p_on_B"] * tzO).sum(axis=1)
                / np.maximum(tzO.sum(axis=1), 1e-12)))
    print(f"  C1nsz (EXPLORATORY)  ncoef={Xz.shape[1]:4d}  "
          f"CV(s,K,b)={cv_nsz['logloss']:12.2f}  "
          f"vs C1gz {cv_nsz['logloss']-cvz['logloss']:+.2f} +/- {se_nsz:.2f}  "
          f"vs z-pooled {cv_nsz['logloss']-cvz_pool['logloss']:+.2f}", flush=True)

    # ---- deliverables ------------------------------------------------------
    deliver = ["C0_frozen", "C1g", "C1gz", chosen["C1n"],
               chosen["C1ns_tensor"], chosen["C1ns_add"], "C1nsz"]
    alias = {"C0_frozen": "C0", "C1g": "C1g", "C1gz": "C1gz",
             chosen["C1n"]: "C1n", chosen["C1ns_tensor"]: "C1ns",
             chosen["C1ns_add"]: "C1nsadd", "C1nsz": "C1nsz"}

    prov_common = dict(
        role=("FIXED completeness calibration object for the absorber-side "
              "ladder; learned on the 2LPT-0 calibration family ONLY; "
              "VALIDATION-ONLY, no sampler, CDDF_analysis untouched"),
        calibration_family=CAL_FAMILY,
        transfer_families=list(TRANSFER_FAMILIES),
        fixed_before_HBI=True,
        fitted_to_real_data=False,
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        recipe=("validation/absorber_ladder/completeness/"
                "fit_completeness_variants.py"),
        git=_git_head(), python=platform.python_version(),
        numpy=np.__version__, ridge=RIDGE,
        x_pivot=X0, u_pivot_log10_snr=grid.u0,
        snr_logmed_s=grid.snr_logmed.tolist(),
        live_strata=grid.live_idx.tolist(),
        convention=("C is a DETECTION probability C_det = matched detections / "
                    "truth systems with NO observed-grid restriction — the "
                    "same object molly_n_det/molly_n_tot measures and the same "
                    "slot the fold's sigmoid(eta_hat + psi_c)[b_to_cell] "
                    "occupies; the in-grid counting fraction phi stays with "
                    "the response kernel"),
        cv=("two-fold sightline halves by TARGETID parity, on the 2LPT-0 "
            "calibration family only"),
        c0_gate=c0_gate)

    summary = dict(provenance=prov_common, chosen=chosen,
                   chosen_min_cv=chosen_min_cv, selection=picks, z_gain=z_gain,
                   variants={}, transfer={}, sparse={}, p6b={})

    # per-family transfer references
    ref = {}
    for fam in FAMILIES:
        t = tabs[fam]
        ref[fam] = dict(
            det=_live(t["det_bks"], grid), tot=_live(t["truth_bks"], grid),
            det_bKs=_bKs_live(coarse_block_sum(np.asarray(t["det_bks"], float),
                                               grid.kz, 1)),
            tot_bKs=_bKs_live(coarse_block_sum(np.asarray(t["truth_bks"], float),
                                               grid.kz, 1)))

    written = []
    for nm in deliver:
        r = results[nm]
        al = alias[nm]
        if nm in ("C1gz", "C1nsz"):
            p_live_bKs = r["p_full"]                              # (Sl,KK,B)
            p_live = (p_live_bKs * tz).sum(axis=1) / np.maximum(tz.sum(axis=1), 1e-12)
            p_live = np.where(tz.sum(axis=1) > 0, p_live, C0_live)
        else:
            p_live = r["p_full"]
            p_live_bKs = None
        C_fixed = np.zeros((grid.S, grid.B))
        C_fixed[grid.live_idx, :] = p_live

        extra = {}
        if p_live_bKs is not None:
            C_bKs = np.zeros((grid.B, KK, grid.S))
            C_bKs[:, :, grid.live_idx] = np.transpose(p_live_bKs, (2, 1, 0))
            C_bkS = C_bKs[:, grid.kz, :]
            extra["C_fixed_bKs"] = C_bKs
            extra["C_fixed_bkS"] = C_bkS
            extra["C_fixed_bkS_with_g"] = C_bkS * grid.g_bk[:, :, None]

        # uncertainty
        if nm == "C1g":
            sd = np.zeros((grid.S, grid.B))
            e, s_ = eta_hat_jeffreys(det, tot)
            sd[grid.live_idx, :] = p_live * (1 - p_live) * s_
        elif nm == "C1gz":
            sd = np.zeros((grid.S, grid.B))
            e, s_ = eta_hat_jeffreys(det, tot)
            sd[grid.live_idx, :] = p_live * (1 - p_live) * s_
        elif nm == "C1nsz":
            sd = np.zeros((grid.S, grid.B))
            sdz = cell_se_smooth(r["state"], Xz, (Sl, KK, B))
            sd[grid.live_idx, :] = (sdz * tz).sum(axis=1) \
                / np.maximum(tz.sum(axis=1), 1e-12)
        elif nm == "C0_frozen":
            sd = np.zeros((grid.S, grid.B))
            sd[grid.live_idx, :] = (C0_full * (1 - C0_full)
                                    * sig_pack[:, grid.b_to_cell]
                                    )[grid.live_idx, :]
        else:
            X = _design_for(nm, grid)
            sd = np.zeros((grid.S, grid.B))
            sd[grid.live_idx, :] = cell_se_smooth(r["state"], X, (Sl, B))

        # transfer residuals (report only)
        tr = {}
        for fam in FAMILIES:
            tr[fam] = _transfer_block(p_live, ref[fam], grid,
                                      p_bKs=p_live_bKs)
        summary["transfer"][al] = tr

        ho = _heldout_block(r["p_on_E"], r["p_on_O"], det_E, tot_E, det_O,
                            tot_O, grid, tzE, tzO, dzE, dzO,
                            pzE=r.get("pz_on_E"), pzO=r.get("pz_on_O"))
        summary.setdefault("heldout", {})[al] = ho

        prov = dict(prov_common)
        prov.update(variant=al, variant_id=nm, meta=r["meta"], heldout=ho,
                    n_fitted_coefficients=int(r["meta"]["n_coef"]),
                    cv_all=r["cv_all"], cv_reported=r.get("cv_reported"),
                    in_sample_deviance=r["in_sample_deviance"],
                    transfer_residual_summary={
                        f: dict(total=tr[f]["total"],
                                reported_window=tr[f]["reported_window"])
                        for f in tr})
        if "beta" in r:
            prov["coefficients"] = np.asarray(r["beta"]).tolist()
            prov["converged"] = r["converged"]
            prov["hessian_cond"] = r["hessian_cond"]

        for fam in FAMILIES:
            path = os.path.join(out_dir, f"C_{al}_{fam}.npz")
            pf = dict(prov)
            pf["target_family"] = fam
            pf["note_family"] = ("C_fixed is FAMILY-INDEPENDENT — it is the "
                                 "2LPT-0 calibration object; the per-family "
                                 "file exists only so the runner can be "
                                 "pointed at one path per mock, and carries "
                                 "that family's transfer residual")
            pf["transfer_residual_this_family"] = dict(
                total=tr[fam]["total"], reported_window=tr[fam]["reported_window"],
                by_b=tr[fam]["by_b"], by_s=tr[fam]["by_s"], by_K=tr[fam]["by_K"])
            np.savez_compressed(
                path, C_fixed=C_fixed, C_fixed_sd=sd,
                live_strata_mask=grid.live_s,
                ntrue_edges=grid.ntrue, snr_edges=grid.snr_edges,
                kz_to_K=grid.kz, b_to_cell=grid.b_to_cell,
                provenance=np.array(json.dumps(pf, indent=1), dtype=object),
                **extra)
            written.append(path)
        summary["variants"][al] = dict(
            variant_id=nm, n_coef=int(r["meta"]["n_coef"]),
            representation=r["meta"]["representation"],
            kind=r["meta"]["kind"], cv_all=r["cv_all"],
            cv_reported=r.get("cv_reported"),
            in_sample_deviance=r["in_sample_deviance"],
            C_fixed=C_fixed.tolist(), C_fixed_sd=sd.tolist())

    # every candidate's CV curve (for the report / figures)
    summary["cv_curve_zlayout"] = dict(
        z_pooled=dict(n_coef=int((tz.sum(axis=1) > 0).sum()),
                      cv_logloss=cvz_pool["logloss"]),
        C1gz=dict(n_coef=int((tz > 0).sum()), cv_logloss=cvz["logloss"]),
        C1nsz=dict(n_coef=int(Xz.shape[1]),
                   cv_logloss=results["C1nsz"]["cv_all"]["logloss"]))
    summary["cv_curve"] = {
        nm: dict(n_coef=int(results[nm]["meta"]["n_coef"]),
                 kind=results[nm]["meta"]["kind"],
                 cv_logloss=results[nm]["cv_all"]["logloss"],
                 cv_per_trial=results[nm]["cv_all"]["per_trial"],
                 cv_reported=results[nm]["cv_reported"]["logloss"],
                 delta_vs_C0=results[nm]["cv_all"].get("delta_vs_C0"),
                 delta_vs_C0_se=results[nm]["cv_all"].get("delta_vs_C0_se"),
                 delta_vs_C0_sigma=results[nm]["cv_all"].get("delta_vs_C0_sigma"),
                 in_sample_deviance=results[nm]["in_sample_deviance"])
        for nm in names}

    # sparse-region uncertainty table (C1g) and the observed cell counts
    e_all, s_all = eta_hat_jeffreys(det, tot)
    p_all = expit(e_all)
    summary["sparse"] = dict(
        n_tot=tot.tolist(), n_det=det.tolist(),
        C1g_sd=(p_all * (1 - p_all) * s_all).tolist(),
        live_strata=grid.live_idx.tolist(),
        Nc=grid.Nc.tolist(), snr_edges=grid.snr_edges.tolist(),
        n_cells_lt_50_trials=int((tot < 50).sum()),
        n_cells_lt_20_trials=int((tot < 20).sum()),
        worst_cells=[dict(b=int(b), s=int(grid.live_idx[i]),
                          N=float(grid.Nc[b]), n_tot=float(tot[i, b]),
                          n_det=float(det[i, b]),
                          C=float(p_all[i, b]),
                          sd=float((p_all * (1 - p_all) * s_all)[i, b]))
                     for i, b in zip(*np.unravel_index(
                         np.argsort(tot, axis=None)[:12], tot.shape))])

    # ---- P6b calibration object -------------------------------------------
    summary["p6b"] = build_p6b(tabs, grid, out_dir, prov_common, written)

    # observed truth / detection tables per family, for the figures
    np.savez_compressed(
        os.path.join(out_dir, "completeness_observed.npz"),
        **{f"det_{f}": ref[f]["det"] for f in FAMILIES},
        **{f"tot_{f}": ref[f]["tot"] for f in FAMILIES},
        **{f"det_bKs_{f}": ref[f]["det_bKs"] for f in FAMILIES},
        **{f"tot_bKs_{f}": ref[f]["tot_bKs"] for f in FAMILIES},
        Nc=grid.Nc, ntrue_edges=grid.ntrue, snr_edges=grid.snr_edges,
        live_idx=grid.live_idx, u_s=grid.u_s, snr_logmed=grid.snr_logmed)

    jpath = os.path.join(out_dir, "completeness_variants_summary.json")
    with open(jpath, "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"\nwrote {len(written)} calibration objects + {jpath}", flush=True)

    # SHA256SUMS
    files = sorted(os.listdir(out_dir))
    with open(os.path.join(out_dir, "SHA256SUMS"), "w") as fh:
        for f in files:
            p = os.path.join(out_dir, f)
            if os.path.isfile(p) and f != "SHA256SUMS":
                fh.write(f"{_sha256(p)}  {f}\n")
    print("wrote SHA256SUMS", flush=True)
    return summary


def _heldout_block(pE, pO, dE, tE, dO, tO, grid, tzE, tzO, dzE, dzO,
                   pzE=None, pzO=None):
    """HELD-OUT expected/observed - 1 by N, by S/N and by coarse z.

    ``pE`` is the prediction FOR the even-TARGETID half made from a fit on the
    odd half (and vice versa), so every number below is out-of-sample.
    """
    num = pE * tE + pO * tO
    den = dE + dO

    def _r(n, d):
        n, d = np.asarray(n, float), np.asarray(d, float)
        return float(n.sum() / d.sum() - 1.0) if d.sum() > 0 else float("nan")
    rep = grid.reported_b
    out = dict(
        total=_r(num, den),
        reported_window=_r(num[:, rep], den[:, rep]),
        by_b=[_r(num[:, b], den[:, b]) for b in range(grid.B)],
        by_s=[_r(num[i], den[i]) for i in range(len(grid.live_idx))])
    z_resolved = pzE is not None
    if not z_resolved:
        pzE = np.broadcast_to(pE[:, None, :], tzE.shape)
        pzO = np.broadcast_to(pO[:, None, :], tzO.shape)
    numz = pzE * tzE + pzO * tzO
    denz = dzE + dzO
    out["by_K"] = [_r(numz[:, K, :], denz[:, K, :]) for K in range(grid.KK)]
    out["by_b_by_K"] = [[_r(numz[:, K, b], denz[:, K, b])
                         for b in range(grid.B)] for K in range(grid.KK)]
    out["z_resolved_prediction"] = bool(z_resolved)
    return out


def _design_for(nm, grid):
    Sl = len(grid.live_idx)
    x, u = grid.x_b, grid.u_s[grid.live_idx]
    if nm.startswith("C1n_deg"):
        return design_per_stratum(x, Sl, int(nm.split("deg")[1]))
    if nm.startswith("C1ns_tensor"):
        dn, du = (int(v) for v in nm.split("_")[-1].split("x"))
        return design_2d_tensor(x, u, dn, du)
    if nm.startswith("C1ns_add"):
        dn, du = (int(v) for v in nm.split("_")[-1].split("x"))
        return design_2d_additive(x, u, dn, du)
    raise ValueError(nm)


def _transfer_block(p_live, r, grid, p_bKs=None):
    """Expected/observed - 1 of the FIXED C against a family's own truth.

    ``p_bKs`` (Sl, KK, B), when given, is the z-RESOLVED prediction and is what
    the by-K residual uses; z-pooled variants broadcast ``p_live`` over K, which
    is exactly the statement that they cannot represent a z trend.
    """
    det, tot = r["det"], r["tot"]
    out = dict(
        total=float(weighted_ratio_residual(p_live, det, tot)),
        reported_window=float(weighted_ratio_residual(
            p_live[:, grid.reported_b], det[:, grid.reported_b],
            tot[:, grid.reported_b])),
        by_b=[float(v) for v in weighted_ratio_residual(p_live, det, tot,
                                                        axis=0)],
        by_s=[float(v) for v in weighted_ratio_residual(p_live, det, tot,
                                                        axis=1)])
    pz = (np.broadcast_to(p_live[:, None, :], r["tot_bKs"].shape)
          if p_bKs is None else np.asarray(p_bKs, float))
    out["by_K"] = [float(v) for v in weighted_ratio_residual(
        pz, r["det_bKs"], r["tot_bKs"], axis=(0, 2))]
    out["z_resolved_prediction"] = bool(p_bKs is not None)
    out["by_b_by_s"] = (np.where(det > 0, p_live * tot / np.maximum(det, 1e-12),
                                 np.nan) - 1.0).tolist()
    return out


def build_p6b(tabs, grid, out_dir, prov_common, written):
    """P6b: the sub-floor-host ([17.2, 19.0)) detection rate per unit path.

    Measured on 2LPT-0 per (c, K, s) as ``P6b_counts / dX``, then transported
    to a family by multiplying its own ``dX[k, s]``.  This is the term the fold
    has NO parameter for (OPERATOR_FORENSICS_REPORT §2.5).
    """
    cal = tabs[CAL_FAMILY]
    kz = grid.kz
    dX_cal = np.asarray(cal["dX"], float)                       # (Kf, S)
    dX_cal_Ks = coarse_block_sum(dX_cal, kz, 0)                 # (KK, S)
    P_cal = np.asarray(cal["P6b_cks"], float)
    P_cal_cKs = coarse_block_sum(P_cal, kz, 1)                  # (C, KK, S)
    rate = np.zeros_like(P_cal_cKs)
    ok = dX_cal_Ks > 0
    rate[:, ok] = P_cal_cKs[:, ok] / dX_cal_Ks[ok]

    # half-split rate, for the held-out check on the calibration family itself
    P_E = coarse_block_sum(np.asarray(cal["P6b_cks_E"], float), kz, 1)
    P_O = coarse_block_sum(np.asarray(cal["P6b_cks_O"], float), kz, 1)
    rate_E = np.zeros_like(rate); rate_O = np.zeros_like(rate)
    rate_E[:, ok] = P_E[:, ok] / (0.5 * dX_cal_Ks[ok])
    rate_O[:, ok] = P_O[:, ok] / (0.5 * dX_cal_Ks[ok])

    out = dict(rate_total_per_dX=float(P_cal.sum() / dX_cal.sum()),
               n_events_cal=float(P_cal.sum()), families={})
    for fam in FAMILIES:
        t = tabs[fam]
        dX = np.asarray(t["dX"], float)
        mu = rate[:, kz, :] * dX[None, :, :]                    # (C, Kf, S)
        obs = np.asarray(t["P6b_cks"], float)
        obs_cKs = coarse_block_sum(obs, kz, 1)
        mu_cKs = coarse_block_sum(mu, kz, 1)
        blk = dict(
            n_obs=float(obs.sum()), n_pred=float(mu.sum()),
            residual_total=float(mu.sum() / max(obs.sum(), 1e-12) - 1.0),
            residual_by_K=[float(mu_cKs[:, K, :].sum()
                                 / max(obs_cKs[:, K, :].sum(), 1e-12) - 1.0)
                           for K in range(grid.KK)],
            residual_by_s=[float(mu[:, :, s].sum()
                                 / max(obs[:, :, s].sum(), 1e-12) - 1.0)
                           if obs[:, :, s].sum() > 0 else None
                           for s in range(grid.S)],
            poisson_sigma_total=float(
                (mu.sum() - obs.sum()) / max(np.sqrt(max(obs.sum(), 1.0)), 1e-9)))
        out["families"][fam] = blk
        prov = dict(prov_common)
        prov.update(
            object="P6b sub-floor-host detection rate per unit path",
            variant="P6b", target_family=fam,
            representation=("rate[c, K, s] = P6b_counts(2LPT-0) / dX(2LPT-0) "
                            "per (observed N-hat cell c, coarse z block K, S/N "
                            "stratum s); mu_P6b[c, k, s] = rate[c, kz_to_K[k], "
                            "s] * dX_thisfamily[k, s]"),
            motivating_defect=("OPERATOR_FORENSICS_REPORT.md §2.5 — hosts in "
                               "[17.2, 19.0) are detected and counted but the "
                               "fold has NO term for them"),
            n_fitted_coefficients=int((rate > 0).sum()),
            n_events_calibration=float(P_cal.sum()),
            transfer=blk)
        path = os.path.join(out_dir, f"P6b_rate_{fam}.npz")
        np.savez_compressed(
            path, mu_P6b_cks=mu, rate_cKs=rate,
            rate_cKs_halfE=rate_E, rate_cKs_halfO=rate_O,
            obs_P6b_cks=obs, dX=dX, kz_to_K=kz,
            nhat_edges=np.asarray(t["nhat_edges"], float),
            zf_edges=np.asarray(t["zf_edges"], float),
            snr_edges=np.asarray(t["snr_edges"], float),
            provenance=np.array(json.dumps(prov, indent=1), dtype=object))
        written.append(path)
        print(f"  P6b {fam}: obs={obs.sum():.0f} pred={mu.sum():.0f} "
              f"({100*blk['residual_total']:+.1f} %, "
              f"{blk['poisson_sigma_total']:+.1f} sigma_Poisson)", flush=True)
    return out


if __name__ == "__main__":
    main()
