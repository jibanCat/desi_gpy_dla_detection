#!/usr/bin/env python
"""run_candidates.py -- fit, cross-validate and deliver the low-DOF response
candidates of the sealed opening rule (RESPONSE_FAMILY_OPENING_RULE_
PREDECLARATION.md, sha256 84ce1de6...; PI ruling 2026-09-13c section 4-7).

CALIBRATION SIDE ONLY.  No HBI/MCMC run.  No real data.  NO mock dN/dX closure
number is computed, read or used -- the only selection device in this program
is the sealed section 5 criterion on held-out calibration rows.

Products -> <out-dir>:
  candidate_cv_table.{json,csv}   the section 3 metric battery per candidate
  section5_decision_trail.json    which levels opened and why
  antitautology_AB.json           section 6 A/B results for PASSING candidates
  rows_<cand>.npz                 the delivered fixed object P(c|b,s,K)
  Mg_<cand>_<fam>.npz             schema absorber_ladder/Mg_fixed/v1
  SHA256SUMS

ENV: gpdla-hbi; PYTHONPATH=<worktree>; JAX_PLATFORMS=cpu.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_RESP = os.path.join(_REPO, "validation", "absorber_ladder", "response")
for _p in (_HERE, _RESP, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import candlib as CL                                           # noqa: E402
import candmetrics as CM                                       # noqa: E402
import families as F                                           # noqa: E402
import phimass as PH                                          # noqa: E402
import respfit as RF                                           # noqa: E402
import opmetrics as OM                                         # noqa: E402
import build_variants as BV                                    # noqa: E402

SCRATCH = ("/scratch/cavestru_root/cavestru0/mfho/"
           "absorber_ladder_2026-09-13")
EVENTS = os.path.join(SCRATCH, "response", "calib_events_2lpt0.npz")
SUPPORT = os.path.join(SCRATCH, "support")
PACKS = "/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/packs"
ADOPTED_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/stage0/adopted_response_v1p1.npz")
FAMS = ("2lpt0", "london0", "saclay0")

LEVEL0 = ("R0", "R1c", "R1d-raw")
LEVEL1 = ("A-small", "E")
LEVEL2 = ("B", "C")
LEVEL3 = ("D",)


# ---------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git():
    try:
        return dict(commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip(),
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=_REPO).decode().strip(),
            dirty=bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=_REPO).decode().strip()))
    except Exception as e:                                   # pragma: no cover
        return dict(commit="unknown", error=str(e))


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


# ===========================================================================
# Level-0 parametric references, expanded onto the (b, s, K) grid
# ===========================================================================
def parametric_rows(ev, mask, geom, spec, N_ref):
    """Fit R0/R1c on ``mask`` and return rows (B, S, K, C) normalised over c."""
    obj = RF.fit_variant(ev["N_true"][mask], ev["dx"][mask], ev["isr"][mask],
                         ev["izr"][mask], N_ref, dict(spec),
                         ntrue_edges=geom["ntrue"])
    masses, _ = OM.model_masses(obj, geom["ntrue"], geom["nhat"],
                                sig_floor=geom["sig_floor"])
    m = masses[geom["s2sr"][:, None], geom["K2zr"][None, :], :, :]
    rows = np.transpose(m, (3, 0, 1, 2))             # (B, S, K, C)
    rows = rows / np.maximum(rows.sum(axis=-1, keepdims=True), 1e-300)
    return rows, obj


# ===========================================================================
# candidate fitting on one training mask
# ===========================================================================
def quadrature_rows_lowrank(fam, par, geom, base_rows=None):
    """Delivered FIXED object: flat quadrature over the latent bin and the
    stratum (predeclaration I2b)."""
    X = CL.quad_design(geom, par["kind"])            # (B,S,K,Q,P)
    B, S, K, Q, P = X.shape
    Xf = X.reshape(-1, P)
    if base_rows is None:
        lb = None
    else:
        lb = np.log(np.maximum(base_rows, CL.BASE_FLOOR))
        lb = np.repeat(lb.reshape(B * S * K, 1, geom["C"]), Q, axis=1
                       ).reshape(-1, geom["C"])
    lp = fam.logp_rows(par, Xf, lb)
    p = np.exp(lp).reshape(B, S, K, Q, geom["C"]).mean(axis=3)
    return p / p.sum(axis=-1, keepdims=True)


def quadrature_rows_generic(fam, par, geom):
    """Delivered FIXED object for the families parametrised by an ABSOLUTE
    location (B, C): the quadrature nodes carry their own N_true."""
    un, vn = CL.quad_grid(geom)
    B, S, K, C = geom["B"], geom["S"], geom["K"], geom["C"]
    Q = CL.N_QUAD_U * CL.N_QUAD_V
    out = np.zeros((B, S, K, C))
    for b in range(B):
        for s in range(S):
            uu, vv = np.meshgrid(un[b], vn[s], indexing="ij")
            uu = uu.ravel(); vv = vv.ravel()
            for k in range(K):
                Xq = CL.design(uu, vv, np.full(Q, k), par["kind"])
                lp = fam.logp_rows(par, Xq, uu + CL.N_REF_U, geom)
                p = np.exp(lp).mean(axis=0)
                out[b, s, k] = p / p.sum()
    return out


def occupancy_rows(fam, par, ev, mask, geom, base_rows=None, absolute=False):
    """Occupancy-WEIGHTED marginalisation (section 4D audit only, I2c)."""
    idx = np.where(mask)[0]
    X = CL.design(ev["u"][idx], ev["v"][idx], ev["K_i"][idx], par["kind"])
    if absolute:
        lp = fam.logp_rows(par, X, ev["N_true"][idx], geom)
    else:
        lb = None if base_rows is None else np.log(np.maximum(
            base_rows[ev["b_i"][idx], ev["s_i"][idx], ev["K_i"][idx]],
            CL.BASE_FLOOR))
        lp = fam.logp_rows(par, X, lb)
    p = np.exp(lp)
    rid = CL.row_index(ev["b_i"][idx], ev["s_i"][idx], ev["K_i"][idx], geom)
    nr = geom["B"] * geom["S"] * geom["K"]
    acc = np.zeros((nr, geom["C"]))
    np.add.at(acc, rid, p)
    n = np.bincount(rid, minlength=nr).astype(float)
    out = np.where(n[:, None] > 0, acc / np.maximum(n[:, None], 1e-12),
                   1.0 / geom["C"])
    out = out / out.sum(axis=-1, keepdims=True)
    return out.reshape(geom["B"], geom["S"], geom["K"], geom["C"])


def fit_candidate(name, ev, mask, geom, N_ref, w=None, base_rows=None,
                  tuned=False, inner_folds=None):
    """Fit one candidate on ``mask``; return dict(rows=..., par=..., meta=...).

    ``rows`` is the DELIVERED FIXED OBJECT (flat quadrature).  ``w`` are event
    weights (section 6 anti-tautology); None means unit weights.
    """
    idx = np.where(mask)[0]
    ww = np.ones(len(idx)) if w is None else np.asarray(w, float)[idx]
    t0 = time.time()
    if name == "R0":
        rows, obj = parametric_rows(ev, mask, geom, BV.specs()["R0"], N_ref)
        return dict(rows=rows, par=dict(kind="parametric"), family=None,
                    meta=dict(nominal_dof=obj["n_coef"]["total_moments"],
                              n_coef=obj["n_coef"], wall_s=time.time() - t0))
    if name == "R1c":
        rows, obj = parametric_rows(ev, mask, geom, BV.specs()["R1c"], N_ref)
        return dict(rows=rows, par=dict(kind="parametric"), family=None,
                    meta=dict(nominal_dof=obj["n_coef"]["total_moments"],
                              n_coef=obj["n_coef"], wall_s=time.time() - t0))
    if name == "R1d-raw":
        cnt = np.zeros((geom["B"], geom["S"], geom["K"], geom["C"]))
        np.add.at(cnt, (ev["b_i"][idx], ev["s_i"][idx], ev["K_i"][idx],
                        ev["c_i"][idx]), ww)
        glob = cnt.sum(axis=(1, 2), keepdims=True)
        glob = glob / np.maximum(glob.sum(axis=-1, keepdims=True), 1e-300)
        glob = np.broadcast_to(glob, cnt.shape)
        rows = F.jeffreys_rows(cnt, 0.5, fallback=glob)
        nrow = int((cnt.sum(axis=-1) > 0).sum())
        n = cnt.sum(axis=-1)
        peff = float(np.sum((geom["C"] - 1) * n / (n + 0.5 * geom["C"])))
        return dict(rows=rows, par=dict(kind="empirical"), family=None,
                    meta=dict(nominal_dof=int(nrow * (geom["C"] - 1)),
                              nominal_dof_label_sealed=702,
                              n_rows_populated=nrow, effective_dof=peff,
                              wall_s=time.time() - t0))
    if name in ("A-small", "E"):
        kind = "asmall" if name == "A-small" else "e"
        fam = F.LowRank(name, kind, R=2, use_base=(name == "E"),
                        level=1)
        X = CL.design(ev["u"][idx], ev["v"][idx], ev["K_i"][idx], kind)
        lb = None
        if name == "E":
            lb = np.log(np.maximum(base_rows[ev["b_i"][idx], ev["s_i"][idx],
                                             ev["K_i"][idx]], CL.BASE_FLOOR))
        par = fam.fit(X, ev["c_i"][idx], lb, ww, geom["C"])
        a, ph, sv = CL.canonicalise(par["alpha"], par["phi"], geom["ccen"])
        par["alpha"], par["phi"], par["sv"] = a, ph, sv
        rows = quadrature_rows_lowrank(fam, par, geom, base_rows)
        base_dof = 108 if name == "E" else 0
        return dict(rows=rows, par=par, family=fam,
                    meta=dict(nominal_dof=fam.nominal_dof(par, base_dof),
                              nominal_dof_own=fam.nominal_dof(par, 0),
                              base_dof=base_dof, sv=sv.tolist(),
                              wall_s=time.time() - t0))
    if name == "B":
        fam = F.Mixture()
        X = CL.design(ev["u"][idx], ev["v"][idx], ev["K_i"][idx], fam.kind)
        par = fam.fit(X, ev["c_i"][idx], ev["N_true"][idx], ww, geom)
        rows = quadrature_rows_generic(fam, par, geom)
        return dict(rows=rows, par=par, family=fam,
                    meta=dict(nominal_dof=fam.nominal_dof(par),
                              wall_s=time.time() - t0))
    if name == "C":
        fam = F.QuantileSpline()
        X = CL.design(ev["u"][idx], ev["v"][idx], ev["K_i"][idx], fam.kind)
        par = fam.fit(X, ev["c_i"][idx], ev["N_true"][idx], ww, geom)
        rows = quadrature_rows_generic(fam, par, geom)
        return dict(rows=rows, par=par, family=fam,
                    meta=dict(nominal_dof=fam.nominal_dof(par),
                              n_knot=fam.n_knot, wall_s=time.time() - t0))
    if name == "D":
        cnt = np.zeros((geom["B"], geom["S"], geom["K"], geom["C"]))
        np.add.at(cnt, (ev["b_i"][idx], ev["s_i"][idx], ev["K_i"][idx],
                        ev["c_i"][idx]), ww)
        lam = (10.0, 10.0, 10.0)
        tune = None
        if tuned and inner_folds is not None:
            lam, tune = tune_D(ev, mask, geom, inner_folds)
        rows, w_own = F.fit_D(cnt, geom, *lam)
        n = cnt.sum(axis=-1)
        peff = float(np.sum((geom["C"] - 1) * w_own))
        return dict(rows=rows, par=dict(kind="empirical", lam=list(lam)),
                    family=None,
                    meta=dict(nominal_dof=int((cnt.sum(axis=-1) > 0).sum() *
                                              (geom["C"] - 1)),
                              n_hyper=3, lam=list(lam), tuning=tune,
                              effective_dof=peff, wall_s=time.time() - t0))
    raise ValueError(name)


def tune_D(ev, mask, geom, inner_folds, grid=(1.0, 3.0, 10.0, 30.0, 100.0)):
    """Inner 2-fold CV on the TRAINING fold only (opened Level 3 only)."""
    ia, ib = inner_folds
    best = None
    for lb_ in grid:
        for ls_ in grid:
            for l0_ in grid:
                sc = 0.0
                for fitm, evm in ((ia, ib), (ib, ia)):
                    fi = mask & fitm
                    ei = mask & evm
                    cnt = np.zeros((geom["B"], geom["S"], geom["K"],
                                    geom["C"]))
                    np.add.at(cnt, (ev["b_i"][fi], ev["s_i"][fi],
                                    ev["K_i"][fi], ev["c_i"][fi]), 1.0)
                    rows, _ = F.fit_D(cnt, geom, lb_, ls_, l0_)
                    sc += float(CM.event_logp_from_rows(rows, ev, ei).sum())
                if best is None or sc > best[0]:
                    best = (sc, (lb_, ls_, l0_))
    return best[1], dict(grid=list(grid), inner_score=best[0],
                         protocol="2-fold TARGETID mod 4 inner split, "
                                  "training fold only")


# ===========================================================================
# held-out scoring of one candidate under the 2-fold protocol
# ===========================================================================
def conditional_logp(name, fit, ev, mask, geom, base_rows):
    """Per-event held-out log P at the event's OWN covariates (I2a)."""
    fam = fit["family"]
    if fam is None:
        return None
    idx = np.where(mask)[0]
    X = CL.design(ev["u"][idx], ev["v"][idx], ev["K_i"][idx],
                  fit["par"]["kind"])
    if name in ("A-small", "E"):
        lb = None
        if name == "E":
            lb = np.log(np.maximum(base_rows[ev["b_i"][idx], ev["s_i"][idx],
                                             ev["K_i"][idx]], CL.BASE_FLOOR))
        lp = fam.logp_rows(fit["par"], X, lb)
    else:
        lp = fam.logp_rows(fit["par"], X, ev["N_true"][idx], geom)
    return lp[np.arange(len(idx)), ev["c_i"][idx]]


def cv_candidate(name, ev, geom, N_ref, tuned=False, w=None):
    """2-fold TARGETID-parity CV.  Returns pooled per-event scores, per-fold
    (rows, held-out counts), and the two fitted objects."""
    A = ev["fold"]
    n = ev["n"]
    ll_fix = np.full(n, np.nan)
    ll_con = np.full(n, np.nan)
    fold_data = []
    fits = {}
    inner = ((np.asarray(ev["tid"], np.int64) % 4) < 2,
             (np.asarray(ev["tid"], np.int64) % 4) >= 2)
    for tag, fitm, evm in (("A", A, ~A), ("B", ~A, A)):
        base = None
        if name == "E":
            base = fit_candidate("R1c", ev, fitm, geom, N_ref)["rows"]
        fit = fit_candidate(name, ev, fitm, geom, N_ref, w=w, base_rows=base,
                            tuned=tuned, inner_folds=inner)
        fit["base_rows"] = base
        fits[tag] = fit
        ll_fix[evm] = CM.event_logp_from_rows(fit["rows"], ev, evm)
        lc = conditional_logp(name, fit, ev, evm, geom, base)
        if lc is not None:
            ll_con[evm] = lc
        fold_data.append((fit["rows"], CL.held_out_counts(ev, evm, geom)))
    return dict(ll_fixed=ll_fix, ll_cond=ll_con, fold_data=fold_data,
                fits=fits)


def effective_dof(name, fit, ev, mask, geom, base_rows):
    """Identifiable rank / ridge p_eff of the observed information."""
    if fit["family"] is None:
        return fit["meta"].get("effective_dof")
    try:
        import jax
        import jax.numpy as jnp
        fam = fit["family"]
        idx = np.where(mask)[0]
        X = CL.apply_std(CL.design(ev["u"][idx], ev["v"][idx], ev["K_i"][idx],
                                   fit["par"]["kind"]),
                         fit["par"].get("xmu", 0.0),
                         fit["par"].get("xsd", 1.0))
        ci = ev["c_i"][idx]
        sub = np.arange(0, len(idx), max(1, len(idx) // 8000))
        Xs = jnp.asarray(X[sub]); cs = jnp.asarray(ci[sub])
        if name in ("A-small", "E"):
            P = X.shape[1]; C = geom["C"]; R = fit["par"]["R"]
            lb = None
            if name == "E":
                lbn = np.log(np.maximum(
                    base_rows[ev["b_i"][idx], ev["s_i"][idx],
                              ev["K_i"][idx]], CL.BASE_FLOOR))
                lb = jnp.asarray(lbn[sub])

            def nll(x):
                a = x[:R * P].reshape(R, P); ph = x[R * P:].reshape(R, C)
                lp = fam._logp(a, ph, Xs, lb)
                return -jnp.mean(lp[jnp.arange(lp.shape[0]), cs])
            x0 = np.concatenate([fit["par"]["alpha"].ravel(),
                                 fit["par"]["phi"].ravel()])
        elif name == "B":
            Ns = jnp.asarray(ev["N_true"][idx][sub])
            edj = jnp.asarray(np.asarray(geom["nhat"], float))
            P = X.shape[1]

            def nll(x):
                b = x.reshape(fam.NPAR, P)
                lp = fam._dens(b, Xs, Ns, edj)
                return -jnp.mean(lp[jnp.arange(lp.shape[0]), cs])
            x0 = fit["par"]["beta"].ravel()
        else:
            Ns = jnp.asarray(ev["N_true"][idx][sub] - CL.N_REF_U)
            Psi = jnp.asarray(fam.psi(geom))
            P = X.shape[1]

            def nll(x):
                b = x.reshape(fam.NPAR, P)
                lp = fam._logp(b, Xs, Psi, Ns, fit["par"]["loc_scale"])
                return -jnp.mean(lp[jnp.arange(lp.shape[0]), cs])
            x0 = fit["par"]["beta"].ravel()
        H = np.asarray(jax.hessian(nll)(jnp.asarray(x0)), float)
        ei = np.linalg.eigvalsh(0.5 * (H + H.T))
        ei = np.clip(ei, 0.0, None)
        mx = float(ei.max())
        rank = int((ei > 1e-8 * mx).sum())
        tau = 1e-6 * mx
        peff = float(np.sum(ei / (ei + tau)))
        return dict(identifiable_rank=rank, p_eff_ridge=peff,
                    n_raw_params=int(len(x0)),
                    note="Hessian of the mean per-event NLL on a "
                         f"{len(sub)}-event subsample; tau = 1e-6 * lam_max")
    except Exception as e:                                   # pragma: no cover
        return dict(error=str(e))


# ===========================================================================
# the section 3 metric battery
# ===========================================================================
def score_candidate(name, cv, ev, geom, refs, ops_paths):
    """Assemble every section 3 metric for one candidate."""
    n_rows = geom["B"] * geom["S"] * geom["K"]
    rid = CL.row_index(ev["b_i"], ev["s_i"], ev["K_i"], geom)
    llf = cv["ll_fixed"]
    out = dict(name=name)
    out["ll_fixed"] = float(np.mean(llf))
    if np.isfinite(cv["ll_cond"]).all():
        out["ll_conditional"] = float(np.mean(cv["ll_cond"]))
        out["marginalisation_cost"] = out["ll_fixed"] - out["ll_conditional"]
    else:
        out["ll_conditional"] = None
        out["marginalisation_cost"] = None
    for rn, rll in refs.items():
        pd = CM.paired_diff(llf, rll)
        out[f"dll_vs_{rn}"] = dict(mean=pd["mean"], se=pd["se"],
                                   n_sigma=pd["mean"] / max(pd["se"], 1e-12))
        rp = CM.row_paired_diff(llf, rll, rid, n_rows)
        out[f"dll_rowcell_vs_{rn}"] = dict(wmean=rp["wmean"],
                                           se=rp["se_between_rows"],
                                           n_rows=rp["n_rows"])
    recs = []
    for rows, counts in cv["fold_data"]:
        recs += CM.row_table(rows, counts, geom, min_events=20)
    big = [r for r in recs if r["n"] >= 200]
    out["rows_ge20"] = len(recs)
    out["rows_ge200"] = len(big)
    out["row_kl_wmean"] = CM.wmean(recs, "kl")
    out["row_kl_wmean_ge200"] = CM.wmean(big, "kl")
    out["row_dev_per_dof_wmean"] = CM.wmean(recs, "dev_per_dof")
    for key in ("d_mean", "r_sd", "d_skew", "d_down", "d_in", "d_up",
                "d_tail"):
        out[f"{key}_wmean"] = CM.wmean(recs, key)
    out["d_skew_wmean_b_ge_21p3"] = CM.wmean(
        recs, "d_skew", sel=lambda r: r["b_lo"] >= 21.3 - 1e-9)
    out["d_mean_wmean_b_ge_21p3"] = CM.wmean(
        recs, "d_mean", sel=lambda r: r["b_lo"] >= 21.3 - 1e-9)
    out["r_sd_wmean_b_ge_21p3"] = CM.wmean(
        recs, "r_sd", sel=lambda r: r["b_lo"] >= 21.3 - 1e-9)
    out["d_skew_wmean_b_lt_19p7"] = CM.wmean(
        recs, "d_skew", sel=lambda r: r["b_hi"] <= 19.7 + 1e-9)
    bt = {}
    for rows, counts in cv["fold_data"]:
        b1 = CM.boundary_table(rows, counts, geom, min_events=200)
        for k, v in b1.items():
            bt.setdefault(k, []).extend(v["records"])
    out["boundary"] = {}
    for k, rr in bt.items():
        nbad = sum(1 for r in rr if abs(r["z"]) > 3.0)
        out["boundary"][k] = dict(
            n_rows=len(rr), n_bad_gt3se=nbad,
            frac_bad=(nbad / len(rr)) if rr else float("nan"),
            wmean_d=CM.wmean(rr, "d") if rr else float("nan"))
    pits = []
    A = ev["fold"]
    for (rows, _), evm in zip(cv["fold_data"], (~A, A)):
        pits.append(CM.pit_hist(rows, ev, evm))
    out["pit"] = dict(chi2=float(np.mean([p["chi2"] for p in pits])),
                      dof=pits[0]["dof"],
                      ks=float(np.mean([p["ks"] for p in pits])),
                      hist=(np.asarray(pits[0]["hist"]) +
                            np.asarray(pits[1]["hist"])).tolist(),
                      edges=pits[0]["edges"])
    out["transfer"] = {}
    full_rows = cv.get("full_rows")
    tr_rows = full_rows if full_rows is not None else cv["fold_data"][0][0]
    for fam, p in ops_paths.items():
        t = CM.transfer_kl(tr_rows, p, geom)
        mo = CM.transfer_moments(tr_rows, p, geom)
        out["transfer"][fam] = dict(wmean_kl=t["wmean_kl"],
                                    median_kl=t["median_kl"],
                                    n_rows=t["n_rows"], moments=mo)
    out["_records"] = recs
    return out


# ===========================================================================
# section 6 anti-tautology A and B
# ===========================================================================
def slope_weights(ev, geom, kind):
    """Event weights, normalised to mean 1 (predeclaration I7)."""
    if kind == "native":
        w = np.ones(ev["n"])
    elif kind in ("steep", "flat"):
        sgn = +0.5 if kind == "steep" else -0.5
        w = 10.0 ** (sgn * ev["u"])
    elif kind == "equalised":
        nb = np.bincount(ev["b_i"], minlength=geom["B"]).astype(float)
        w = np.where(nb[ev["b_i"]] > 0, 1.0 / np.maximum(nb[ev["b_i"]], 1), 0.)
    else:
        raise ValueError(kind)
    return w * (len(w) / max(w.sum(), 1e-300))


def wpaired(d, w):
    d = np.asarray(d, float); w = np.asarray(w, float)
    sw = w.sum()
    m = float(np.sum(w * d) / sw)
    se = float(np.sqrt(np.sum(w ** 2 * (d - m) ** 2)) / sw)
    return m, se


def _safe(f, x, *a):
    x = np.asarray(x, float)
    return float(f(x, *a)) if x.size else float("nan")


def row_kl(p, q, floor=1e-300):
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), floor)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), floor)
    ps = np.maximum(p, floor)
    return np.sum(np.where(p > 0, ps * np.log(ps / np.maximum(q, floor)),
                           0.0), axis=-1)


def antitaut_A(name, ev, geom, N_ref, rows_native):
    """Reweighting invariance: rebuild under +-0.5 dex^-1 slope reweighting and
    equalised occupancy; row KL against the split-half SAMPLING KL."""
    full = np.ones(ev["n"], bool)
    base = None
    if name == "E":
        base = fit_candidate("R1c", ev, full, geom, N_ref)["rows"]
    out = {}
    for kind in ("steep", "flat", "equalised"):
        w = slope_weights(ev, geom, kind)
        r = fit_candidate(name, ev, full, geom, N_ref, w=w,
                          base_rows=base)["rows"]
        out[kind] = r
    tid = np.asarray(ev["tid"], np.int64)
    h1, h2 = (tid % 4) < 2, (tid % 4) >= 2
    b1 = fit_candidate("R1c", ev, h1, geom, N_ref)["rows"] \
        if name == "E" else None
    b2 = fit_candidate("R1c", ev, h2, geom, N_ref)["rows"] \
        if name == "E" else None
    s1 = fit_candidate(name, ev, h1, geom, N_ref, base_rows=b1)["rows"]
    s2 = fit_candidate(name, ev, h2, geom, N_ref, base_rows=b2)["rows"]
    kl_samp = 0.5 * (row_kl(s1, s2) + row_kl(s2, s1))
    cnt = CL.held_out_counts(ev, np.ones(ev["n"], bool), geom).sum(axis=-1)
    big = cnt >= 200
    res = dict(n_rows_ge200=int(big.sum()),
               sampling_kl_median=_safe(np.median, kl_samp[big]))
    for kind, r in out.items():
        kl = row_kl(rows_native, r)
        exc = kl[big] > 3.0 * np.maximum(kl_samp[big], 1e-12)
        res[kind] = dict(
            kl_median=_safe(np.median, kl[big]),
            kl_p95=_safe(np.percentile, kl[big], 95),
            ratio_median=_safe(np.median,
                               kl[big] / np.maximum(kl_samp[big], 1e-12)),
            n_rows_over_3x=int(exc.sum()),
            frac_rows_over_3x=(float(exc.mean()) if exc.size else float("nan")),
            verdict=("OCCUPANCY-IMPRINTED" if exc.mean() > 0.20
                     else "invariant within sampling noise"))
    return res


def antitaut_B(name, ev, geom, N_ref):
    """Train-on-one-slope / evaluate-on-another held-out loglik matrix."""
    A = ev["fold"]
    kinds = ("native", "steep", "flat")
    mat = {}
    per_event = {}
    for tk in kinds:
        w = slope_weights(ev, geom, tk)
        ll = np.full(ev["n"], np.nan)
        for fitm, evm in ((A, ~A), (~A, A)):
            base = fit_candidate("R1c", ev, fitm, geom, N_ref)["rows"] \
                if name == "E" else None
            fit = fit_candidate(name, ev, fitm, geom, N_ref, w=w,
                                base_rows=base)
            ll[evm] = CM.event_logp_from_rows(fit["rows"], ev, evm)
        per_event[tk] = ll
    ref = per_event["native"]
    for ek in kinds:
        we = slope_weights(ev, geom, ek)
        row = {}
        for tk in kinds:
            m, se = wpaired(per_event[tk] - ref, we)
            row[tk] = dict(dll=m, se=se,
                           n_sigma=(m / se if se > 0 else 0.0),
                           ll=float(np.sum(we * per_event[tk]) / we.sum()))
        mat[ek] = row
    worst = max(abs(row[tk]["n_sigma"]) for row in mat.values()
                for tk in kinds if tk != "native")
    return dict(matrix=mat, worst_abs_n_sigma=float(worst),
                verdict=("conditional (all crossed shifts < 2 paired SE)"
                         if worst < 2.0 else
                         "TRAIN-SLOPE DEPENDENT (>= 2 paired SE)"))


# ===========================================================================
# products
# ===========================================================================
def build_mg(rows, fam, geom, phi):
    """(S, Kf, C, B) tensor, schema absorber_ladder/Mg_fixed/v1.

    Rows sum over c to phi(b, s, K) -- the measured in-grid HAD MASS, NOT the
    pack's frozen parametric ``adopted_phi_ref`` (post-seal Science-lane
    instruction of 2026-09-13; the parametric phi_ref is defective by factors
    0.65-1.54 below N_true = 19.7 with a systematic S/N trend).  The pack's
    phi_ref is still returned so the two can be compared downstream.
    """
    pk = np.load(os.path.join(PACKS, f"scanpack_{fam}_b300.npz"),
                 allow_pickle=True)
    for key, src in (("ntrue", "ntrue_edges"), ("nhat", "nhat_edges"),
                     ("snr", "snr_edges"), ("zc", "zc_edges")):
        if not np.array_equal(geom[key], np.asarray(pk[src], float)):
            raise SystemExit(f"pack {fam} geometry differs: {src}")
    phi_ref = np.asarray(pk["adopted_phi_ref"], float)        # (SR, ZR, B)
    kz2K = np.asarray(pk["kz_to_K"], int)
    s2sr = np.clip(np.searchsorted(np.asarray(pk["resp_snr_edges"], float),
                                   geom["snr"][:-1] + 1e-9, "right") - 1, 0, 2)
    K2zr = np.searchsorted(np.asarray(pk["resp_z_edges"], float),
                           0.5 * (geom["zc"][:-1] + geom["zc"][1:]),
                           "right") - 1
    S, Kf, C, B = geom["S"], len(kz2K), geom["C"], geom["B"]
    Mg = np.zeros((S, Kf, C, B))
    phi_ref_g = np.zeros((B, S, geom["K"]))
    for s in range(S):
        for K in range(geom["K"]):
            phi_ref_g[:, s, K] = phi_ref[s2sr[s], K2zr[K], :]
    for s in range(S):
        for kf in range(Kf):
            K = kz2K[kf]
            Mg[s, kf] = (rows[:, s, K, :] * phi[:, s, K][:, None]).T
    return Mg, phi_ref_g


def write_products(out, name, rows, meta, geom, fams, ev, extra,
                   phi=None, phi_smooth=None, phi_meta=None,
                   phi_fam=None):
    os.makedirs(out, exist_ok=True)
    paths = []
    rp = os.path.join(out, f"rows_{name}.npz")
    np.savez_compressed(
        rp, rows=rows, ntrue_edges=geom["ntrue"], nhat_edges=geom["nhat"],
        snr_edges=geom["snr"], zc_edges=geom["zc"],
        provenance=np.array(json.dumps(_jsonable(dict(
            schema="absorber_ladder/response_candidate_rows/v1",
            candidate=name, axes="rows[b, s, K, c]; sum over c == 1 exactly",
            meta=meta, extra=extra,
            calibration=("2LPT-0 natural-pair matched calibration events "
                         f"(n={ev['n']}) ONLY"),
            sealed_rule=("governance/response_review_2026-09-13/"
                         "RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION.md "
                         "sha256 84ce1de6"),
            git=_git())))))
    paths.append(rp)
    for fam in fams:
        Mg, phi_ref_g = build_mg(rows, fam, geom, phi)
        Mg_sm, _ = build_mg(rows, fam, geom, phi_smooth)
        mp = os.path.join(out, f"Mg_{name}_{fam}.npz")
        np.savez_compressed(
            mp, Mg=Mg, Mg_phi_smooth=Mg_sm,
            phi_bsK=phi, phi_bsK_smooth=phi_smooth,
            phi_bsK_family_measured=(phi_fam.get(fam) if phi_fam else phi),
            phi_ref_pack_gathered=phi_ref_g, variant=np.array(name),
            rows_unit=rows,
            coefficients=np.array(json.dumps(_jsonable(
                dict(representation=name, meta=meta,
                     product_file=f"rows_{name}.npz")))),
            provenance=np.array(json.dumps(_jsonable(dict(
                schema="absorber_ladder/Mg_fixed/v1",
                variant=name, family=fam,
                pack=os.path.join(PACKS, f"scanpack_{fam}_b300.npz"),
                pack_sha256=_sha256(os.path.join(
                    PACKS, f"scanpack_{fam}_b300.npz")),
                shape=list(Mg.shape),
                consumer=("CDDF_analysis/hbi_mcmc/fp_ladder.model_cc_ladder "
                          "einsum('skcb,sb,bk->cks', Mg, C, w); "
                          "validation/fp_ladder/run_ladder.py --mg-fixed"),
                count_conservation=(
                    "Mg[s,kf,c,b] = phi_bsK[b,s,K(kf)] * rows_unit[b,s,K,c]; "
                    "rows_unit sums to 1 over c EXACTLY, so Mg rows sum to "
                    "phi -- the MEASURED in-grid had mass (Jeffreys +1/2 "
                    "conditional count ratio N_det_bks_true_z / "
                    "N_det_all_bks_true_z of empirical_ops_2lpt0_A0), NOT "
                    "the pack's frozen parametric adopted_phi_ref (stored as "
                    "phi_ref_pack_gathered for comparison). Mg_phi_smooth is "
                    "the same object with the 6-coefficient logistic phi."),
                phi=_jsonable(phi_meta or {}),
                fitted_on=("2LPT-0 ONLY; the london0/saclay0 files differ "
                           "solely in the pack's adopted_phi_ref and gather "
                           "indices"),
                builder="validation/absorber_ladder/response_review/"
                        "candidates/run_candidates.py",
                git=_git())))))
        paths.append(mp)
    return paths


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=EVENTS)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", default=",".join(FAMS))
    ap.add_argument("--only", default="", help="comma list of candidates")
    ap.add_argument("--no-antitaut", action="store_true")
    ap.add_argument("--antitaut-only", default="",
                    help="comma list: run ONLY the section 6 A/B audit for "
                         "these candidates, reading rows_<name>.npz from the "
                         "products dir as the native object, and write "
                         "antitautology_AB_extra.json")
    ap.add_argument("--subsample", type=int, default=0,
                    help="smoke test only; NEVER used for a reported number")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    fams = [f for f in a.families.split(",") if f]

    geom = CL.load_geom(os.path.join(SUPPORT, "empirical_ops_2lpt0_A0.npz"),
                        os.path.join(PACKS, "scanpack_2lpt0_b300.npz"))
    ev = CL.load_events(a.events, geom)
    if a.subsample:
        rs = np.random.RandomState(0)
        keep = rs.choice(ev["n"], a.subsample, replace=False)
        m = np.zeros(ev["n"], bool); m[keep] = True
        for k in list(ev):
            if isinstance(ev[k], np.ndarray) and ev[k].shape[:1] == (ev["n"],):
                ev[k] = ev[k][m]
        ev["n"] = int(m.sum())
    N_ref = float(np.load(ADOPTED_NPZ, allow_pickle=True)["N_ref"])
    ops_paths = {f: os.path.join(SUPPORT, f"empirical_ops_{f}_A0.npz")
                 for f in fams}
    # ---- phi: the row HAD MASS (post-seal Science-lane instruction) -----
    k_in, n_all = PH.load_phi_counts(
        os.path.join(SUPPORT, "empirical_ops_2lpt0_A0.npz"), geom)
    phi_meas = PH.phi_percell(k_in, n_all)
    phi_sm, phi_sm_meta = PH.fit_phi_smooth(k_in, n_all, geom)
    phi_meas = np.where(n_all > 0, phi_meas, phi_sm)
    phi_cv = PH.phi_cv(k_in, n_all, geom)
    phi_fam = {}
    for f in fams:
        pth = os.path.join(SUPPORT, f"empirical_ops_{f}_A0.npz")
        if os.path.exists(pth):
            kf_, nf_ = PH.load_phi_counts(pth, geom)
            pf_ = PH.phi_percell(kf_, nf_)
            phi_fam[f] = np.where(nf_ > 0, pf_, phi_sm)
    _mgtmp, phi_ref_g = build_mg(
        np.full((geom["B"], geom["S"], geom["K"], geom["C"]),
                1.0 / geom["C"]), "2lpt0", geom, phi_meas)
    phi_block = dict(
        default="measured per-cell Jeffreys +1/2 (no smoothing across b/s/K)",
        smooth_alternative=phi_sm_meta,
        cv=phi_cv,
        summary=PH.phi_summary(phi_meas, phi_ref_g, k_in, n_all, geom),
        insample_ll_per_trial=dict(
            percell=PH.binomial_ll_per_trial(phi_meas, k_in, n_all),
            smooth=PH.binomial_ll_per_trial(phi_sm, k_in, n_all),
            pack_phi_ref=PH.binomial_ll_per_trial(
                np.clip(phi_ref_g, 1e-6, 1 - 1e-6), k_in, n_all)),
        note=("POST-SEAL addition; phi is scored SEPARATELY and does NOT "
              "enter the sealed section 5 criterion, which is applied to the "
              "in-grid row shape only."))
    print(f"[cand] phi: heldout ll/trial percell="
          f"{phi_cv['heldout_ll_per_trial']['percell']:+.5f} smooth="
          f"{phi_cv['heldout_ll_per_trial']['smooth']:+.5f}; "
          f"cells phi<0.98 = {phi_block['summary']['n_cells_phi_lt_0p98']}",
          flush=True)

    print(f"[cand] n={ev['n']} foldA={int(ev['fold'].sum())} "
          f"B={geom['B']} C={geom['C']} S={geom['S']} K={geom['K']} "
          f"N_ref={N_ref} clipped_c={ev['n_clipped_c']}", flush=True)

    if a.antitaut_only:
        res = {}
        for m in [x for x in a.antitaut_only.split(",") if x]:
            rp = os.path.join(out, f"rows_{m}.npz")
            native = np.asarray(np.load(rp, allow_pickle=True)["rows"], float)
            t0 = time.time()
            print(f"[cand] section 6 A/B for {m}", flush=True)
            res[m] = dict(A=antitaut_A(m, ev, geom, N_ref, native),
                          B=antitaut_B(m, ev, geom, N_ref),
                          wall_s=time.time() - t0,
                          native_rows=os.path.basename(rp))
            with open(os.path.join(out, "antitautology_AB_extra.json"),
                      "w") as fh:
                json.dump(_jsonable(dict(
                    note=("section 6 A/B run for EVERY candidate: the sealed "
                          "rule asks for it on candidates that PASS section "
                          "5, and under the literal predeclared section "
                          "5(iii) implementation none did, so the audit is "
                          "run on all of them rather than on a subset chosen "
                          "after seeing the scores."),
                    results=res)), fh, indent=1)
        print("[cand] section 6 extra audit written", flush=True)
        return 0

    names = [n for n in (LEVEL0 + LEVEL1 + LEVEL2 + LEVEL3)]
    if a.only:
        names = [n for n in names if n in a.only.split(",")]

    # ---- 2-fold CV -------------------------------------------------------
    cvs, full = {}, {}
    full_mask = np.ones(ev["n"], bool)
    base_full = None
    for name in names:
        t0 = time.time()
        cvs[name] = cv_candidate(name, ev, geom, N_ref,
                                 tuned=False)
        if name == "E" and base_full is None:
            base_full = fit_candidate("R1c", ev, full_mask, geom,
                                      N_ref)["rows"]
        fit = fit_candidate(name, ev, full_mask, geom, N_ref,
                            base_rows=base_full if name == "E" else None)
        fit["base_rows"] = base_full if name == "E" else None
        full[name] = fit
        cvs[name]["full_rows"] = fit["rows"]
        print(f"[cand] {name}: CV+full in {time.time()-t0:.1f}s  "
              f"ll_fixed={np.nanmean(cvs[name]['ll_fixed']):+.4f}",
              flush=True)

    refs = {k: cvs[k]["ll_fixed"] for k in ("R1d-raw", "R1c") if k in cvs}

    # ---- section 3 battery ----------------------------------------------
    table = {}
    rid = CL.row_index(ev["b_i"], ev["s_i"], ev["K_i"], geom)
    n_rows = geom["B"] * geom["S"] * geom["K"]
    for name in names:
        sc = score_candidate(name, cvs[name], ev, geom, refs, ops_paths)
        fit = full[name]
        sc["nominal_dof"] = fit["meta"].get("nominal_dof")
        sc["meta"] = {k: v for k, v in fit["meta"].items() if k != "tuning"}
        sc["effective_dof"] = effective_dof(
            name, fit, ev, full_mask, geom, fit.get("base_rows"))
        sc["ll_insample"] = float(np.mean(
            CM.event_logp_from_rows(fit["rows"], ev, full_mask)))
        sc["insample_heldout_gap"] = sc["ll_insample"] - sc["ll_fixed"]
        sc["wall_s"] = float(fit["meta"].get("wall_s", np.nan))
        table[name] = sc
        print(f"[cand] {name}: ll={sc['ll_fixed']:+.4f} "
              f"dll_vs_R1d={sc.get('dll_vs_R1d-raw', {}).get('mean', np.nan):+.4f} "
              f"rowKL={sc['row_kl_wmean']:.4f} "
              f"dskew(b>=21.3)={sc['d_skew_wmean_b_ge_21p3']:+.3f}",
              flush=True)

    # ---- section 5 decision trail ---------------------------------------
    trail = dict(rule=("RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION.md "
                       "section 5, verbatim"),
                 reading=("a LEVEL fails only if EVERY candidate in it fails; "
                          "a single passing representation closes the higher "
                          "levels (sealed section 5 'a PASS at a lower level "
                          "closes the higher levels')"),
                 levels={})
    crit = {}
    for name in names:
        if name not in cvs or "R1d-raw" not in refs:
            continue
        crit[name] = CM.criterion_section5(
            cvs[name]["ll_fixed"], refs["R1d-raw"], rid, n_rows,
            cvs[name]["fold_data"], geom)
        table[name]["section5"] = dict(
            fail=crit[name]["fail"], which=crit[name]["which"],
            detail=crit[name]["detail"], boundary=crit[name]["boundary"])
    for lvl, mem in ((0, LEVEL0), (1, LEVEL1), (2, LEVEL2), (3, LEVEL3)):
        mm = [m for m in mem if m in crit]
        if not mm:
            continue
        passes = [m for m in mm if not crit[m]["fail"]]
        trail["levels"][f"level{lvl}"] = dict(
            candidates=mm, passing=passes,
            level_fails=bool(not passes),
            per_candidate={m: crit[m]["which"] for m in mm})
    l1 = trail["levels"].get("level1", {})
    trail["opened"] = dict(level1=True,
                           level2=bool(l1.get("level_fails", True)))
    l2 = trail["levels"].get("level2", {})
    trail["opened"]["level3"] = bool(trail["opened"]["level2"] and
                                     l2.get("level_fails", True))
    for lvl, mem in ((2, LEVEL2), (3, LEVEL3)):
        for m in mem:
            if m in table:
                table[m]["opened"] = trail["opened"][f"level{lvl}"]
                table[m]["tuning"] = ("default only (NOT OPENED)"
                                      if not trail["opened"][f"level{lvl}"]
                                      else "opened")
    for m in LEVEL1:
        if m in table:
            table[m]["opened"] = True
            table[m]["tuning"] = "opened"

    # ---- Level 3 tuning if (and only if) opened --------------------------
    if trail["opened"].get("level3") and "D" in names:
        print("[cand] level 3 OPENED -> tuning D by inner CV", flush=True)
        cvs["D"] = cv_candidate("D", ev, geom, N_ref, tuned=True)
        inner = ((np.asarray(ev["tid"], np.int64) % 4) < 2,
                 (np.asarray(ev["tid"], np.int64) % 4) >= 2)
        full["D"] = fit_candidate("D", ev, full_mask, geom, N_ref,
                                  tuned=True, inner_folds=inner)
        cvs["D"]["full_rows"] = full["D"]["rows"]
        sc = score_candidate("D", cvs["D"], ev, geom, refs, ops_paths)
        sc["nominal_dof"] = full["D"]["meta"]["nominal_dof"]
        sc["meta"] = full["D"]["meta"]
        sc["effective_dof"] = full["D"]["meta"].get("effective_dof")
        sc["ll_insample"] = float(np.mean(
            CM.event_logp_from_rows(full["D"]["rows"], ev, full_mask)))
        sc["insample_heldout_gap"] = sc["ll_insample"] - sc["ll_fixed"]
        sc["opened"] = True
        sc["tuning"] = "opened; 3 shrinkage hypers by inner 2-fold CV"
        crit["D"] = CM.criterion_section5(
            cvs["D"]["ll_fixed"], refs["R1d-raw"], rid, n_rows,
            cvs["D"]["fold_data"], geom)
        sc["section5"] = dict(fail=crit["D"]["fail"], which=crit["D"]["which"],
                              detail=crit["D"]["detail"],
                              boundary=crit["D"]["boundary"])
        table["D"] = sc
        trail["levels"]["level3"] = dict(
            candidates=["D"], passing=([] if crit["D"]["fail"] else ["D"]),
            level_fails=bool(crit["D"]["fail"]),
            per_candidate={"D": crit["D"]["which"]})

    # ---- section 6 A / B for PASSING candidates --------------------------
    at = {}
    passing = [m for m in (LEVEL1 + LEVEL2 + LEVEL3)
               if m in crit and not crit[m]["fail"]]
    at["_passing"] = passing
    if not a.no_antitaut:
        for m in passing + (["R1d-raw"] if "R1d-raw" in crit else []):
            print(f"[cand] anti-tautology A/B for {m}", flush=True)
            t0 = time.time()
            at[m] = dict(A=antitaut_A(m, ev, geom, N_ref, full[m]["rows"]),
                         B=antitaut_B(m, ev, geom, N_ref),
                         wall_s=time.time() - t0)
            table[m]["antitautology"] = dict(
                A_verdicts={k: v["verdict"] for k, v in at[m]["A"].items()
                            if isinstance(v, dict) and "verdict" in v},
                B_verdict=at[m]["B"]["verdict"],
                B_worst_n_sigma=at[m]["B"]["worst_abs_n_sigma"])

    # ---- occupancy audit (section 4D) ------------------------------------
    audit = {}
    for name in names:
        fit = full[name]
        if fit["family"] is not None:
            occ = occupancy_rows(fit["family"], fit["par"], ev, full_mask,
                                 geom, fit.get("base_rows"),
                                 absolute=(name in ("B", "C")))
            cnt = CL.held_out_counts(ev, full_mask, geom).sum(axis=-1)
            big = cnt >= 200
            kl = row_kl(fit["rows"], occ)
            audit[name] = dict(
                occupancy_in_fit="NO (conditional per-event likelihood)",
                occupancy_in_marginalisation=("NO (flat quadrature over the "
                                              "latent bin and the stratum)"),
                kl_flat_vs_occupancy_median=_safe(np.median, kl[big]),
                kl_flat_vs_occupancy_p95=_safe(np.percentile, kl[big], 95))
        elif name in ("R1d-raw", "D"):
            audit[name] = dict(
                occupancy_in_fit=("YES -- the row IS the cell's own count "
                                  "vector; within-bin/within-stratum weight "
                                  "is the calibration mock's own f(N) and S/N "
                                  "distribution"),
                occupancy_in_marginalisation="YES (implicit)",
                shrinkage_depends_on=("row count n_bsK (Jeffreys) " +
                                      ("plus neighbour-row and global-row "
                                       "count pools (D)" if name == "D"
                                       else "")))
        else:
            audit[name] = dict(
                occupancy_in_fit=("sub-bin moment fits weighted by sub-bin "
                                  "COUNT (WLS weight sqrt(n)); adaptive "
                                  "sub-bin edges merge on counts"),
                occupancy_in_marginalisation=("R1c: NO (flat within-bin "
                                              "quadrature); R0: n/a "
                                              "(midpoint)"))

    # ---- write -----------------------------------------------------------
    for name in names:
        write_products(out, name, full[name]["rows"], table[name]["meta"],
                       geom, fams, ev,
                       extra=dict(section5_fail=table[name]
                                  .get("section5", {}).get("fail"),
                                  opened=table[name].get("opened", True)),
                       phi=phi_meas, phi_smooth=phi_sm, phi_meta=phi_block,
                       phi_fam=phi_fam)
        table[name]["phi_carried"] = (
            "measured per-cell Jeffreys (default); Mg_phi_smooth carries the "
            "6-coefficient logistic alternative in the same file")
    slim = {}
    for k, v in table.items():
        slim[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
    with open(os.path.join(out, "candidate_cv_table.json"), "w") as fh:
        json.dump(_jsonable(dict(
            schema="absorber_ladder/response_review_candidates/v1",
            generated_utc=datetime.datetime.utcnow().isoformat() + "Z",
            sealed_rule_sha256="84ce1de6 (opening rule)",
            implementation_predeclaration_sha256=(
                "2772c1c32ccff92ab1d4adc5e3c5e0897b0daea1b399b4346331a40cb4"
                "fa4245"),
            n_events=ev["n"], geometry=dict(B=geom["B"], C=geom["C"],
                                            S=geom["S"], K=geom["K"]),
            table=slim, occupancy_audit=audit, phi=phi_block,
            git=_git())), fh, indent=1)
    with open(os.path.join(out, "section5_decision_trail.json"), "w") as fh:
        json.dump(_jsonable(trail), fh, indent=1)
    with open(os.path.join(out, "antitautology_AB.json"), "w") as fh:
        json.dump(_jsonable(at), fh, indent=1)
    cols = ["name", "nominal_dof", "ll_fixed", "ll_conditional",
            "ll_insample", "insample_heldout_gap", "row_kl_wmean",
            "row_kl_wmean_ge200", "row_dev_per_dof_wmean", "d_mean_wmean",
            "r_sd_wmean", "d_skew_wmean", "d_skew_wmean_b_ge_21p3",
            "d_tail_wmean", "rows_ge20", "rows_ge200", "wall_s"]
    with open(os.path.join(out, "candidate_cv_table.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["dll_vs_R1d-raw", "dll_vs_R1d-raw_se",
                           "dll_vs_R1c", "dll_vs_R1c_se",
                           "frac_bad_20.0", "frac_bad_20.3", "frac_bad_21.0",
                           "pit_chi2", "kl_london0", "kl_saclay0",
                           "section5_fail", "opened"])
        for n in names:
            t = table[n]
            w.writerow([t.get(c) for c in cols] + [
                t.get("dll_vs_R1d-raw", {}).get("mean"),
                t.get("dll_vs_R1d-raw", {}).get("se"),
                t.get("dll_vs_R1c", {}).get("mean"),
                t.get("dll_vs_R1c", {}).get("se"),
                t["boundary"].get("20.0", {}).get("frac_bad"),
                t["boundary"].get("20.3", {}).get("frac_bad"),
                t["boundary"].get("21.0", {}).get("frac_bad"),
                t["pit"]["chi2"],
                t["transfer"].get("london0", {}).get("wmean_kl"),
                t["transfer"].get("saclay0", {}).get("wmean_kl"),
                t.get("section5", {}).get("fail"), t.get("opened")])
    np.savez_compressed(
        os.path.join(out, "cv_fold_rows.npz"),
        **{f"{n}__rows_fold{i}": fd[0]
           for n in names for i, fd in enumerate(cvs[n]["fold_data"])},
        **{f"{n}__heldout_counts_fold{i}": fd[1]
           for n in names for i, fd in enumerate(cvs[n]["fold_data"])},
        ntrue_edges=geom["ntrue"], nhat_edges=geom["nhat"],
        snr_edges=geom["snr"], zc_edges=geom["zc"])
    np.savez_compressed(
        os.path.join(out, "row_records.npz"),
        **{f"{n}__{k}": np.array([r[k] for r in table[n]["_records"]], float)
           for n in names
           for k in ("b", "s", "k", "n", "kl", "d_mean", "r_sd", "d_skew",
                     "d_tail", "deviance")})
    files = sorted(os.listdir(out))
    with open(os.path.join(out, "SHA256SUMS"), "w") as fh:
        for f in files:
            p = os.path.join(out, f)
            if os.path.isfile(p) and f != "SHA256SUMS":
                fh.write(f"{_sha256(p)}  {f}\n")
    print("[cand] products in", out, flush=True)
    print("[cand] section 5 trail:", json.dumps(_jsonable(trail["opened"])),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
