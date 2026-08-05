# -*- coding: utf-8 -*-
"""evidence.py -- the convergence / closure / coverage evidence harness.

WHY THIS EXISTS
---------------
The governing PI decision is that a paper-facing uncertainty band must come
from a faithful joint posterior, and that *sampler convergence alone is not
enough if the forward model fails truth closure*.  Rung 9 is the worked
example: the sampler was healthy (r_hat 1.00978, ESS_bulk 554, ESS_tail 789,
2 divergences / 4000 draws) and the answer was still wrong by a factor 0.165
in the lowest reported bin, because the FORWARD MODEL could not reproduce the
observed counts at any parameter value.  No convergence diagnostic can see
that.  A posterior predictive check can, and does, from inside the sampler.

So this module produces ONE artifact -- ``inference_evidence.json`` -- holding
five independent blocks, and a GATE that is FAIL-CLOSED:

  1. convergence : split-R-hat / ESS_bulk / ESS_tail PER REPORTED QUANTITY
                   (not only per latent), divergence count AND their location
                   in parameter space, tree-depth saturation, E-BFMI.
  2. ppc         : replicated counts simulated from posterior draws, compared
                   to the observed counts per (N, z) cell and in marginals,
                   with Bayesian p-values and the list of cells the model
                   cannot reproduce.
  3. closure     : posterior median / pack truth with its credible interval,
                   per reported quantity and per z-bin, z-scores, and the
                   coverage of the truth by the 68% and 95% intervals.
  4. coverage_sbc: simulation-based calibration rank statistics (see sbc.py).
  5. ztilt       : the estimator's manufactured redshift tilt, and whether an
                   INTEGRATED (z-marginalised) result is the only defensible
                   product.

GATE SEMANTICS (fail-closed, and the reason for the omission-sensitivity test)
  * a MISSING block is a FAILURE, never a pass.  ``stampable`` is the AND of
    "every required block present" and "every check True".  There is no way to
    stamp a run by simply not running a check.
  * a block that ran but could not compute a required quantity records
    ``incomplete: [...]`` and that alone blocks the stamp.  This is what a
    saved rung-9-style artifact hits: ``run_rung9.py`` flattens the chain axis
    away and requests only ``extra_fields=("diverging",)``, so split-R-hat per
    reported quantity, tree-depth saturation and E-BFMI are UNRECOVERABLE from
    the artifact.  The harness says so rather than reporting a subset as if it
    were the whole.

MOCK ONLY where truth is involved (blocks 3-5 require ``pack.truth_counts``).
Omega values stay in the arbitrary-constant units of ``reduce_f_posterior``;
closure is a ratio so the physical constant cancels.
"""
from __future__ import annotations

import collections.abc as _abc
import os
import subprocess
import time
import warnings

import numpy as np

from CDDF_analysis.hbi_mcmc import reporting as RP
from CDDF_analysis.hbi_mcmc.model_a import (
    POLICY, _MASK_LO, _MASK_HI, _THRESHOLDS, reduce_f_posterior)
from CDDF_analysis.hbi_mcmc import ratification as _RAT

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az

__all__ = [
    "REQUIRED_BLOCKS", "REQUIRED_CHECKS",
    "EBFMI_MIN", "TREEDEPTH_SAT_MAX", "PPC_PVAL_MIN",
    "SBC_UNIFORM_P_MIN", "convergence_block", "ppc_block", "closure_block",
    "ztilt_block", "assemble_evidence", "gate", "reported_quantities",
    "posterior_run_from_mcmc", "posterior_run_from_artifact",
]

# --- policy beyond model_a.POLICY (r_hat 1.01 / ESS 400 / 0 divergences) -----
EBFMI_MIN = 0.3            # Stan/Betancourt convention
TREEDEPTH_SAT_MAX = 0.0    # any saturated transition is a failure to report
PPC_PVAL_MIN = 0.002       # two-sided per-cell tail below which a cell "fails"
PPC_MAX_FAILED_CELL_FRAC = 0.0   # fail-closed: ANY unreproducible cell blocks
PPC_OMNIBUS_MIN = 0.01     # omnibus chi2-discrepancy posterior predictive p
SBC_UNIFORM_P_MIN = 0.01   # chi2 uniformity p-value of the rank histogram
CLOSURE_COVER95_MIN = 1.0  # every reported quantity's 95% CI must hold truth

REQUIRED_BLOCKS = ("convergence", "ppc", "closure", "coverage_sbc", "ztilt")

#: STRUCTURALLY REQUIRED CHECKS: ``block -> (check name, ...)`` that a required
#: block MUST volunteer.  A block that simply omits one of these does not get
#: the benefit of the doubt -- the gate synthesises the check as False.
#:
#: WHY THIS EXISTS.  ``gate`` is otherwise entirely at the mercy of whatever
#: keys a block chooses to put in ``checks``: an older ``coverage_sbc`` block,
#: a hand-written one, or a block from a module version predating the
#: matched-configuration ratification, all pass by SILENCE.  The RATIFIED
#: statement (2026-07-29, decision 8) is that an artifact carrying an unmatched
#: *or unspecified* SBC is not stampable, and "unspecified" can only be
#: enforced here, not in the block that failed to specify it.
#:
#: Like ``required``, this may only ever GROW.
REQUIRED_CHECKS = {
    "coverage_sbc": ("sbc_configuration_matches_run",),
}

_SCHEMA = "inference_evidence/v1"


# ============================================================================
# 0. input adapters -- a "posterior run" from a live MCMC or a saved artifact
# ============================================================================

def _tag(thr):
    return f"{thr:.1f}".replace(".", "p")


def reported_quantities(f_by_chain, pack):
    """(chains, draws, B, Kf) f-draws -> the REPORTED scalars, chain-resolved.

    Returns ``dict name -> (chains, draws) float array``.  These are the
    quantities the paper would quote, which is the level R-hat and ESS must be
    reported at -- a latent-only convergence summary can be green while a
    reported functional of it is not.
    """
    f_by_chain = np.asarray(f_by_chain, float)
    n_chain, n_draw = f_by_chain.shape[:2]
    flat = f_by_chain.reshape((n_chain * n_draw,) + f_by_chain.shape[2:])
    red = reduce_f_posterior(flat, pack)

    dX_k = np.asarray(pack.dX, float).sum(axis=1)
    kz = np.asarray(pack.kz_to_K)
    out = {}

    def _put(name, flat_arr):
        out[name] = np.asarray(flat_arr, float).reshape(n_chain, n_draw)

    _put("integrated_total", red["integrated_total"])
    for thr in _THRESHOLDS:
        tg = _tag(thr)
        # z-marginalised (pathlength-weighted over ALL fine z) -- the
        # integrated product the ztilt block argues may be the only
        # defensible one.
        for stat in ("dndx", "omega"):
            fine = np.asarray(red[f"{stat}_{tg}"], float)       # (n, Kf)
            _put(f"{stat}_{tg}_integrated",
                 (fine * dX_k[None, :]).sum(axis=1) / dX_k.sum())
            for K in range(pack.n_kk):
                sel = kz == K
                _put(f"{stat}_{tg}_z{K}",
                     (fine[:, sel] * dX_k[sel][None, :]).sum(axis=1)
                     / dX_k[sel].sum())
    return out


def posterior_run_from_mcmc(mcmc, pack, *, max_tree_depth):
    """Full-evidence adapter: everything the harness can want, from a live run."""
    samples_c = mcmc.get_samples(group_by_chain=True)
    try:
        sampled = set(mcmc.last_state.z.keys())
    except AttributeError:                                   # pragma: no cover
        sampled = None
    latents = {k: np.asarray(v) for k, v in samples_c.items()
               if sampled is None or k in sampled}
    extras = {k: np.asarray(v)
              for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    return {
        "latents_by_chain": latents,
        "f_by_chain": np.asarray(samples_c["f"]),
        "samples_by_chain": {k: np.asarray(v) for k, v in samples_c.items()},
        "extras_by_chain": extras,
        "max_tree_depth": int(max_tree_depth),
        "source": "live_mcmc",
    }


def posterior_run_from_artifact(result, pack, *, n_chains=None,
                                max_tree_depth=None):
    """Degraded adapter for a saved ``run_rung9`` JSON.

    ``run_rung9.py`` writes ``mcmc.get_samples()`` -- ALREADY FLATTENED over
    chains -- and requests only ``extra_fields=("diverging",)``.  So from an
    artifact alone:
      * the chain axis can only be RECONSTRUCTED by assuming numpyro's
        contiguous (chain, draw) reshape, which the caller must assert
        explicitly via ``n_chains``; the assumption is stamped;
      * per-transition divergence flags, ``num_steps`` and ``energy`` are
        simply absent -> divergence LOCATION, tree-depth saturation and E-BFMI
        are unrecoverable.
    The block records those as ``incomplete`` and the gate refuses the stamp.
    """
    red = result["reductions"]
    f = np.asarray(red["f"], float)
    n_draw_total = f.shape[0]
    if n_chains is None:
        n_chains = int((result.get("sampler") or {}).get("chains") or 1)
    if n_draw_total % n_chains:
        raise ValueError(
            f"cannot reshape {n_draw_total} flat draws into {n_chains} chains")
    per = n_draw_total // n_chains
    return {
        "latents_by_chain": None,
        "f_by_chain": f.reshape(n_chains, per, *f.shape[1:]),
        "samples_by_chain": None,
        "extras_by_chain": {},
        "max_tree_depth": max_tree_depth,
        "source": "saved_artifact",
        "chain_axis_assumed_contiguous": True,
        "saved_diagnostics": dict(result.get("diagnostics") or {}),
    }


# ============================================================================
# 1. CONVERGENCE
# ============================================================================

def _rhat_ess(named_2d):
    """dict name -> (chains, draws) -> per-name rank-normalized split-R-hat,
    ESS_bulk, ESS_tail (arviz)."""
    idata = az.from_dict(posterior={k: np.asarray(v)
                                    for k, v in named_2d.items()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rh = az.rhat(idata)          # default method="rank" = split + rank-norm
        eb = az.ess(idata, method="bulk")
        et = az.ess(idata, method="tail")
    out = {}
    for k in named_2d:
        out[k] = {"r_hat": float(np.asarray(rh[k].values)),
                  "ess_bulk": float(np.asarray(eb[k].values)),
                  "ess_tail": float(np.asarray(et[k].values))}
    return out


def _flatten_sites(by_chain):
    """dict site -> (chains, draws, *event) -> dict 'site[i,j]' -> (chains,draws)."""
    flat = {}
    for name, arr in by_chain.items():
        a = np.asarray(arr, float)
        if a.ndim == 2:
            flat[name] = a
            continue
        n_c, n_d = a.shape[:2]
        r = a.reshape(n_c, n_d, -1)
        ev = a.shape[2:]
        for j in range(r.shape[2]):
            idx = np.unravel_index(j, ev)
            flat[f"{name}[{','.join(str(i) for i in idx)}]"] = r[:, :, j]
    return flat


def _divergence_location(latents_flat, diverging):
    """Where in parameter space do the divergent transitions sit?

    For every scalar coordinate, the standardized shift of the divergent
    subsample relative to the non-divergent one,
        z = (mean_div - mean_ok) / sd_all ,
    plus the divergent subsample's mean rank within the full draw pool (0.5 =
    no localization).  Returns the 10 coordinates with the largest |z|.
    """
    div = np.asarray(diverging, bool).reshape(-1)
    n_div = int(div.sum())
    if n_div == 0:
        return {"n_divergent": 0, "localized": False, "top_coords": []}
    rows = []
    for name, arr in latents_flat.items():
        x = np.asarray(arr, float).reshape(-1)
        sd = float(x.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        z = float((x[div].mean() - x[~div].mean()) / sd)
        order = np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
        rows.append({"coord": name, "z_shift": z,
                     "mean_quantile_of_divergent": float(order[div].mean())})
    rows.sort(key=lambda r: -abs(r["z_shift"]))
    top = rows[:10]
    return {"n_divergent": n_div,
            # a divergence cloud displaced by >1 sd in ANY coordinate is
            # localized, i.e. it marks a specific geometry, not bad luck
            "localized": bool(top and abs(top[0]["z_shift"]) > 1.0),
            "top_coords": top}


def convergence_block(run, pack, *, policy=None):
    """Block 1.  Raw numbers always; the POLICY flags on top of them."""
    policy = dict(POLICY if policy is None else policy)
    inc = []
    out = {"policy": policy, "source": run.get("source")}
    if run.get("chain_axis_assumed_contiguous"):
        out["chain_axis_assumed_contiguous"] = True
        out["chain_axis_note"] = (
            "the chain axis was RECONSTRUCTED by assuming numpyro's contiguous "
            "(chain, draw) flattening; it is an assumption, not a measurement")

    f_by_chain = np.asarray(run["f_by_chain"])
    n_chain = int(f_by_chain.shape[0])
    out["n_chains"] = n_chain
    out["n_draws_per_chain"] = int(f_by_chain.shape[1])
    if n_chain < 2:
        inc.append("split_rhat_needs_2_chains")

    # -- 1a. per REPORTED quantity (the level the paper quotes)
    rep = reported_quantities(f_by_chain, pack)
    out["reported"] = _rhat_ess(rep) if n_chain >= 2 else {}
    # -- 1b. per latent site (kept: it is what the legacy summary reported)
    if run.get("latents_by_chain"):
        out["latent"] = _rhat_ess(_flatten_sites(run["latents_by_chain"]))
    else:
        out["latent"] = {}
        inc.append("latent_sites_absent")

    def _agg(d, key, fn):
        return float(fn([v[key] for v in d.values()])) if d else float("nan")

    out["summary"] = {
        "reported_r_hat_max": _agg(out["reported"], "r_hat", np.max),
        "reported_ess_bulk_min": _agg(out["reported"], "ess_bulk", np.min),
        "reported_ess_tail_min": _agg(out["reported"], "ess_tail", np.min),
        "latent_r_hat_max": _agg(out["latent"], "r_hat", np.max),
        "latent_ess_bulk_min": _agg(out["latent"], "ess_bulk", np.min),
        "latent_ess_tail_min": _agg(out["latent"], "ess_tail", np.min),
    }
    worst = (max(out["reported"].items(), key=lambda kv: kv[1]["r_hat"])[0]
             if out["reported"] else None)
    out["summary"]["reported_r_hat_argmax"] = worst
    out["summary"]["reported_ess_bulk_argmin"] = (
        min(out["reported"].items(), key=lambda kv: kv[1]["ess_bulk"])[0]
        if out["reported"] else None)

    ex = run.get("extras_by_chain") or {}

    # -- 1c. divergences + their location
    if "diverging" in ex:
        div = np.asarray(ex["diverging"], bool)
        out["divergences"] = {
            "n_divergent": int(div.sum()),
            "per_chain": [int(c.sum()) for c in div],
            "rate": float(div.mean()),
        }
        if run.get("latents_by_chain"):
            out["divergences"].update(
                _divergence_location(_flatten_sites(run["latents_by_chain"]),
                                     div))
        else:
            inc.append("divergence_location_needs_latent_draws")
    else:
        out["divergences"] = None
        inc.append("divergence_flags_absent")

    # -- 1d. tree-depth saturation
    md = run.get("max_tree_depth")
    if "num_steps" in ex and md:
        ns = np.asarray(ex["num_steps"], float)
        depth = np.ceil(np.log2(ns + 1.0))
        sat = depth >= float(md)
        out["treedepth"] = {
            "max_tree_depth": int(md),
            "depth_max": float(depth.max()), "depth_mean": float(depth.mean()),
            "n_saturated": int(sat.sum()),
            "frac_saturated": float(sat.mean()),
        }
    else:
        out["treedepth"] = None
        inc.append("treedepth_needs_num_steps_extra_field")

    # -- 1e. E-BFMI (per chain)
    if "energy" in ex:
        E = np.asarray(ex["energy"], float)
        ebfmi = []
        for c in range(E.shape[0]):
            e = E[c]
            v = float(e.var(ddof=1))
            ebfmi.append(float(np.sum(np.diff(e) ** 2) / (len(e) - 1) / v)
                         if v > 0 else float("nan"))
        out["ebfmi"] = {"per_chain": ebfmi, "min": float(np.nanmin(ebfmi))}
    else:
        out["ebfmi"] = None
        inc.append("ebfmi_needs_energy_extra_field")

    # -- checks (raw numbers reported above; these are the GATE)
    s = out["summary"]
    checks = {
        "reported_r_hat_ok": bool(s["reported_r_hat_max"]
                                  <= policy["r_hat_max"]),
        "reported_ess_bulk_ok": bool(s["reported_ess_bulk_min"]
                                     >= policy["ess_bulk_min"]),
        "reported_ess_tail_ok": bool(s["reported_ess_tail_min"]
                                     >= policy["ess_tail_min"]),
        "latent_r_hat_ok": bool(s["latent_r_hat_max"] <= policy["r_hat_max"]),
        "latent_ess_bulk_ok": bool(s["latent_ess_bulk_min"]
                                   >= policy["ess_bulk_min"]),
        "latent_ess_tail_ok": bool(s["latent_ess_tail_min"]
                                   >= policy["ess_tail_min"]),
        "divergences_ok": bool(
            out["divergences"] is not None
            and out["divergences"]["n_divergent"] <= policy["n_divergent"]),
        "treedepth_ok": bool(
            out["treedepth"] is not None
            and out["treedepth"]["frac_saturated"] <= TREEDEPTH_SAT_MAX),
        "ebfmi_ok": bool(out["ebfmi"] is not None
                         and out["ebfmi"]["min"] >= EBFMI_MIN),
    }
    out["checks"] = checks
    out["incomplete"] = inc
    return out


# ============================================================================
# 2. POSTERIOR PREDICTIVE CHECKS
# ============================================================================

def _mu_draws(samples_flat, consts, n_max=None, rng=None):
    """mu(theta) per posterior draw, via the model's own forward fold."""
    import jax
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import fold_mu

    theta = np.asarray(samples_flat["theta_pop"], float)
    n = theta.shape[0]
    idx = np.arange(n)
    if n_max is not None and n > n_max:
        rng = rng or np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=n_max, replace=False))
    theta = jnp.asarray(theta[idx])
    psi_c = jnp.asarray(np.asarray(samples_flat["psi_c"], float)[idx])
    psi_k = jnp.asarray(np.asarray(samples_flat["psi_k_delta"], float)[idx])
    t = jnp.asarray(np.asarray(samples_flat["t"], float)[idx])
    if "lam_fp" in samples_flat:
        lam = jnp.asarray(np.asarray(samples_flat["lam_fp"], float)[idx])
    else:
        lam = jnp.zeros((len(idx), consts.n_c, consts.n_s))
    fold = jax.vmap(fold_mu, in_axes=(0, 0, 0, 0, 0, None))
    return np.asarray(fold(theta, psi_c, psi_k, t, lam, consts)), idx


def _mid_p(rep, obs):
    """P(rep > obs) + 0.5 P(rep == obs) along axis 0 -- uniform under the model
    for discrete data (the standard randomized/mid-p correction)."""
    return (rep > obs).mean(axis=0) + 0.5 * (rep == obs).mean(axis=0)


def ppc_block(run, pack, consts, *, n_rep_draws=300, seed=0):
    """Block 2.  Replicate the counts from posterior draws and compare.

    THIS is the check that would have caught rung 9 from inside the sampler:
    a forward model that cannot reach the observed counts produces mid-p
    values pinned at 0 or 1 in exactly the cells where mu/obs is off, no
    matter how healthy R-hat is.
    """
    if run.get("samples_by_chain") is None:
        return {"incomplete": ["ppc_needs_latent_posterior_draws"],
                "checks": {"ppc_cells_ok": False, "ppc_omnibus_ok": False},
                "note": ("a saved reductions-only artifact carries f-draws but "
                         "not the nuisance draws the forward fold needs; the "
                         "PPC cannot be reconstructed from it")}
    rng = np.random.default_rng(seed)
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:])
            for k, v in run["samples_by_chain"].items()}
    mu, idx = _mu_draws(flat, consts, n_max=n_rep_draws, rng=rng)   # (n,C,Kf,S)
    obs = np.asarray(pack.counts, float)
    dxpos = np.asarray(pack.dX, float) > 0                          # (Kf,S)
    mask = np.broadcast_to(dxpos[None, :, :], obs.shape)

    # zero-dX (k, s) cells are structurally unobserved: the model masks them
    # out of the likelihood and the validator guarantees zero counts there, so
    # they must not contribute to any p-value or marginal either.
    mu = np.where(mask[None], mu, 0.0)
    y_rep = rng.poisson(np.clip(mu, 0.0, None)).astype(float)       # (n,C,Kf,S)
    y_rep = np.where(mask[None], y_rep, 0.0)

    def _tail(rep, o):
        p = _mid_p(rep, o)
        return p, 2.0 * np.minimum(p, 1.0 - p)

    # -- per (N, z) cell, summed over strata (the report grain)
    rep_cz = y_rep[:, :, :, :].sum(axis=3)
    obs_cz = np.where(mask, obs, 0.0).sum(axis=2)
    mu_cz = mu.sum(axis=3)
    p_cz, two_cz = _tail(rep_cz, obs_cz[None])
    obs_any = np.broadcast_to(dxpos.any(axis=1)[None, :], obs_cz.shape)

    nhat = np.asarray(pack.nhat_edges, float)
    zf = np.asarray(pack.zf_edges, float)
    bad = []
    ii, jj = np.where(obs_any & (two_cz < PPC_PVAL_MIN))
    order = np.argsort(two_cz[ii, jj])
    for c, k in zip(ii[order], jj[order]):
        bad.append({
            "nhat_lo": float(nhat[c]), "nhat_hi": float(nhat[c + 1]),
            "z_lo": float(zf[k]), "z_hi": float(zf[k + 1]),
            "obs": float(obs_cz[c, k]), "mu_median": float(np.median(mu_cz[:, c, k])),
            "ratio_mu_over_obs": (float(np.median(mu_cz[:, c, k]) / obs_cz[c, k])
                                  if obs_cz[c, k] > 0 else None),
            "p_mid": float(p_cz[c, k]), "p_two_sided": float(two_cz[c, k]),
        })

    # -- marginals
    def _marg(axis_keep):
        ax = tuple(a for a in (1, 2, 3) if a != axis_keep)
        rep = y_rep.sum(axis=ax)
        o = np.where(mask, obs, 0.0).sum(axis=tuple(a - 1 for a in ax))
        p = _mid_p(rep, o[None])
        return [{"i": int(i), "obs": float(o[i]),
                 "mu_median": float(np.median(rep[:, i])),
                 "p_mid": float(p[i]),
                 "p_two_sided": float(2 * min(p[i], 1 - p[i]))}
                for i in range(len(o))]

    tot_rep = y_rep.sum(axis=(1, 2, 3))
    tot_obs = float(np.where(mask, obs, 0.0).sum())
    p_tot = float(_mid_p(tot_rep, tot_obs))

    # -- omnibus chi2 discrepancy T(y, theta) = sum (y-mu)^2 / mu
    def _T(y):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (y - mu) ** 2 / np.clip(mu, 1e-12, None)
        return np.where(np.broadcast_to(mask[None], r.shape), r, 0.0).sum(
            axis=(1, 2, 3))
    T_obs, T_rep = _T(obs[None]), _T(y_rep)
    ppp = float((T_rep >= T_obs).mean())

    n_cells = int(obs_any.sum())
    n_bad = len(bad)
    out = {
        "n_posterior_draws_used": int(len(idx)),
        "n_cells": n_cells,
        "n_cells_failed": n_bad,
        "frac_cells_failed": float(n_bad / max(n_cells, 1)),
        "pval_threshold_two_sided": PPC_PVAL_MIN,
        "failed_cells": bad[:40],
        "marginal_by_nhat": _marg(1),
        "marginal_by_z": _marg(2),
        "marginal_by_snr": _marg(3),
        "total": {"obs": tot_obs, "mu_median": float(np.median(tot_rep)),
                  "p_mid": p_tot,
                  "p_two_sided": float(2 * min(p_tot, 1 - p_tot))},
        "omnibus_chi2_discrepancy": {
            "T_obs_median": float(np.median(T_obs)),
            "T_rep_median": float(np.median(T_rep)),
            "posterior_predictive_p": ppp},
        "incomplete": [],
    }
    out["checks"] = {
        "ppc_cells_ok": bool(out["frac_cells_failed"]
                             <= PPC_MAX_FAILED_CELL_FRAC),
        "ppc_omnibus_ok": bool(PPC_OMNIBUS_MIN <= ppp <= 1.0 - PPC_OMNIBUS_MIN),
    }
    return out


# ============================================================================
# 3. MOCK CLOSURE
# ============================================================================

def _truth_reported(pack):
    """The truth counterpart of every name in ``reported_quantities``.

    🔴 SUPPORT MATCHING IS THE WHOLE POINT OF THIS ROUTINE.  ``reduce_f_posterior``
    integrates a tier with DEX-OVERLAP weights (``reporting.window_overlap_weights``),
    because on the adopted 0.2-dex latent basis a threshold like 20.0 is NOT a
    basis edge.  This function must therefore weight truth COUNTS by the same
    overlap, expressed as a FRACTION of the bin (``reporting.truth_overlap_fractions``).

    It did NOT, until 2026-07-29: it selected basis bins by CENTRE and weighted
    whole bins.  On a 0.2-dex basis the posterior then integrated half of
    [19.9, 20.1) while the truth integrated all of it, and ``closure_block`` /
    ``analyze_rung9`` compared two different estimands -- a ~20% spurious deficit
    on ``dndx_20p0_integrated`` (measured 0.787 at f = f_true) that was PURE
    BOOKKEEPING.  That is [[one-sided support]], the class that has now bitten
    this project five times.  The regression test is
    ``test_truth_side_uses_the_same_window_convention_as_the_posterior`` and it
    runs on a COARSE pack, because the 0.1-dex equivalence test is green under
    both conventions and could never have caught it.
    """
    if pack.truth_counts is None:
        raise ValueError("closure needs a mock pack (truth_counts)")
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    kz = np.asarray(pack.kz_to_K)
    dX_k = np.asarray(pack.dX, float).sum(axis=1)
    tc = np.asarray(pack.truth_counts, float)                   # (B, Kf)
    reported = Nc >= float(np.asarray(pack.nhat_edges, float)[0]) - 1e-9

    out, n_out = {}, {}
    # integrated_total is reduce_f_posterior's sum_{b reported, k} f dN -- a
    # PLAIN sum over fine-z (not a pathlength-weighted mean), so its truth
    # counterpart is sum_k (counts in that z slice) / dX_k of that slice.
    # No window is involved: the reported support is a whole-bin selection on
    # both sides (the observed floor is always an exact basis edge, enforced by
    # validate_pack), so no overlap fraction is needed here.
    out["integrated_total"] = float(
        (tc[reported, :].sum(axis=0) / dX_k).sum())
    n_out["integrated_total"] = float(tc[reported].sum())
    w_omega = 10.0 ** (Nc - 21.0)
    for thr in _THRESHOLDS:
        tg = _tag(thr)
        # THE SAME SUPPORT the posterior integrates: overlap of [thr, inf) with
        # each basis bin, as a fraction of that bin, zeroed off the reported
        # support exactly as reduce_f_posterior's ``_wts`` does.
        frac = np.where(reported,
                        RP.truth_overlap_fractions(ntrue, thr, np.inf), 0.0)
        # 10^(Nc - 21) uses the FULL basis-bin centre on both sides -- matching
        # the estimand is the requirement, not re-centring the sub-interval.
        for stat, wt in (("dndx", np.ones_like(Nc)), ("omega", w_omega)):
            num_k = (tc * (wt * frac)[:, None]).sum(axis=0)      # (Kf,)
            n_k = (tc * frac[:, None]).sum(axis=0)
            out[f"{stat}_{tg}_integrated"] = float(num_k.sum() / dX_k.sum())
            n_out[f"{stat}_{tg}_integrated"] = float(n_k.sum())
            for K in range(pack.n_kk):
                m = kz == K
                out[f"{stat}_{tg}_z{K}"] = float(num_k[m].sum() / dX_k[m].sum())
                n_out[f"{stat}_{tg}_z{K}"] = float(n_k[m].sum())
    return out, n_out


def closure_block(run, pack):
    """Block 3.  posterior median / truth, its CI, z-score, 68/95 coverage."""
    rep = reported_quantities(run["f_by_chain"], pack)
    truth, n_truth = _truth_reported(pack)
    rows = []
    for name, arr in sorted(rep.items()):
        d = np.asarray(arr, float).reshape(-1)
        T = float(truth[name])
        q = np.quantile(d, [0.025, 0.16, 0.5, 0.84, 0.975])
        sd = float(d.std(ddof=1))
        n_t = float(n_truth[name])
        ratio = d / T if T > 0 else np.full_like(d, np.nan)
        rq = np.quantile(ratio, [0.025, 0.16, 0.5, 0.84, 0.975])
        rows.append({
            "quantity": name, "truth": T,
            "post_median": float(q[2]), "post_mean": float(d.mean()),
            "post_sd": sd,
            "q025": float(q[0]), "q16": float(q[1]), "q84": float(q[3]),
            "q975": float(q[4]),
            "ratio_median": float(rq[2]),
            "ratio_q16": float(rq[1]), "ratio_q84": float(rq[3]),
            "ratio_q025": float(rq[0]), "ratio_q975": float(rq[4]),
            "z": float((float(q[2]) - T) / sd) if sd > 0 else None,
            "in68": bool(q[1] <= T <= q[3]),
            "in95": bool(q[0] <= T <= q[4]),
            "truth_rel_poisson_err": (float(np.sqrt(n_t) / n_t)
                                      if n_t > 0 else None),
        })
    c68 = float(np.mean([r["in68"] for r in rows]))
    c95 = float(np.mean([r["in95"] for r in rows]))
    out = {
        "rows": rows,
        "n_quantities": len(rows),
        "coverage68": c68, "coverage95": c95,
        "worst_z": max((abs(r["z"]) for r in rows if r["z"] is not None),
                       default=None),
        "worst_quantity": max(
            (r for r in rows if r["z"] is not None),
            key=lambda r: abs(r["z"]), default={"quantity": None})["quantity"],
        "incomplete": [],
    }
    out["checks"] = {
        "closure_cover95_ok": bool(c95 >= CLOSURE_COVER95_MIN),
        # a nominal 68% interval should hold the truth ~68% of the time; we
        # only fail on a gross miss, since the quantities are correlated
        "closure_cover68_not_pathological": bool(c68 >= 0.5),
    }
    return out


# ============================================================================
# 5. REDSHIFT-TILT DIAGNOSTIC
# ============================================================================

def ztilt_block(run, pack, *, stat="dndx", thr=20.3, forward_fold=True,
                resp_clamp="both"):
    """Block 5.  Does the estimator manufacture a trend in z?

    Reference defect: R0(dN/dX >= 20.3) runs 0.908 -> 1.052 -> 1.189 across
    z in [2.0, 3.5) with 0/3 z-bins covered at 95% against a ~2% statistical
    half-width.  This block measures the same object on ANY run and decides
    whether the INTEGRATED (z-marginalised) result is the only defensible
    product: it is, when the per-z intervals miss the truth but the integrated
    one holds it.

    ``forward_fold`` additionally folds the pack's OWN truth through the pack's
    OWN kernel with NO sampling at all.  Any z-trend there is the FORWARD
    MODEL's, not the sampler's -- which is how much of the tilt survives the
    forward-model fixes.
    """
    tg = _tag(thr)
    rep = reported_quantities(run["f_by_chain"], pack)
    truth, _ = _truth_reported(pack)
    zc = np.asarray(pack.zc_edges, float)
    zmid = 0.5 * (zc[:-1] + zc[1:])

    per_z = []
    for K in range(pack.n_kk):
        name = f"{stat}_{tg}_z{K}"
        d = np.asarray(rep[name], float).reshape(-1)
        T = float(truth[name])
        q = np.quantile(d, [0.025, 0.16, 0.5, 0.84, 0.975])
        per_z.append({
            "coarse_z": int(K), "z_mid": float(zmid[K]), "truth": T,
            "R0_median": float(q[2] / T) if T > 0 else None,
            "R0_q16": float(q[1] / T) if T > 0 else None,
            "R0_q84": float(q[3] / T) if T > 0 else None,
            "R0_q025": float(q[0] / T) if T > 0 else None,
            "R0_q975": float(q[4] / T) if T > 0 else None,
            "in68": bool(q[1] <= T <= q[3]),
            "in95": bool(q[0] <= T <= q[4]),
            "half_width68_frac": (float(0.5 * (q[3] - q[1]) / q[2])
                                  if q[2] > 0 else None),
        })
    R = np.array([p["R0_median"] for p in per_z], float)
    ok = np.isfinite(R)
    slope = (float(np.polyfit(zmid[ok], R[ok], 1)[0])
             if ok.sum() >= 2 else None)
    span = (float(np.nanmax(R) - np.nanmin(R)) if ok.any() else None)
    hw = np.array([p["half_width68_frac"] or np.nan for p in per_z], float)
    n95 = int(sum(p["in95"] for p in per_z))

    iname = f"{stat}_{tg}_integrated"
    di = np.asarray(rep[iname], float).reshape(-1)
    Ti = float(truth[iname])
    qi = np.quantile(di, [0.025, 0.16, 0.5, 0.84, 0.975])
    integ = {"quantity": iname, "truth": Ti,
             "R0_median": float(qi[2] / Ti) if Ti > 0 else None,
             "in68": bool(qi[1] <= Ti <= qi[3]),
             "in95": bool(qi[0] <= Ti <= qi[4])}

    out = {
        "stat": stat, "threshold": thr,
        "per_z": per_z,
        "integrated": integ,
        "R0_span": span,
        "R0_slope_per_unit_z": slope,
        "median_half_width68_frac": (float(np.nanmedian(hw))
                                     if np.isfinite(hw).any() else None),
        "tilt_over_statistical_width": (
            float(span / np.nanmedian(hw))
            if (span is not None and np.isfinite(hw).any()
                and np.nanmedian(hw) > 0) else None),
        "n_z_bins": len(per_z), "n_z_in95": n95,
        "reference_defect": {
            "R0_by_z": [0.908, 1.052, 1.189], "n_z_in95": 0,
            "stat_half_width_frac": 0.02,
            "note": "the standing ZTILT defect this block exists to measure"},
        "incomplete": [],
    }
    # the verdict the task asks for
    out["integrated_only_defensible"] = bool(
        n95 < len(per_z) and integ["in95"])
    out["z_resolved_defensible"] = bool(n95 == len(per_z))

    if forward_fold:
        try:
            from CDDF_analysis.hbi_mcmc import forward_selftest as FS
            res = FS.selftest(pack, resp_clamp=resp_clamp)
            tab = FS.ratio_tables(res, pack)
            zr = [r["ratio"] for r in tab["by_z"]]
            fin = [v for v in zr if np.isfinite(v)]
            out["forward_fold_ztilt"] = {
                "resp_clamp": resp_clamp,
                "mu_over_obs_by_fine_z": [float(v) for v in zr],
                "span": (float(max(fin) - min(fin)) if fin else None),
                "total_ratio": float(tab["total"]["ratio"]),
                "note": ("pure truth-fold, ZERO sampling: any trend here is "
                         "the FORWARD MODEL's z-tilt, not the sampler's"),
            }
        except Exception as exc:                                # pragma: no cover
            out["forward_fold_ztilt"] = {"error": repr(exc)}
            out["incomplete"].append("forward_fold_ztilt_failed")

    out["checks"] = {
        # the gate does NOT require zero tilt -- it requires that the run has
        # DECIDED which product is defensible and that the chosen one covers
        "ztilt_has_a_defensible_product": bool(
            out["z_resolved_defensible"] or out["integrated_only_defensible"]),
        "ztilt_z_resolved_ok": bool(out["z_resolved_defensible"]),
    }
    return out


# ============================================================================
# 6. GATE + ASSEMBLY
# ============================================================================

def _is_bool(v):
    """A GENUINE boolean.  ``np.bool_`` counts (model_a/ppc/sbc build checks
    from numpy comparisons); ``int``/``float``/``str``/``list`` do NOT, however
    truthy they are.  ``np.bool_`` is not a subclass of ``bool``, so it has to
    be named explicitly."""
    return isinstance(v, (bool, np.bool_))


def _is_mapping(v):
    return isinstance(v, _abc.Mapping)


def _is_sequence(v):
    """A genuine sequence of entries.  ``str``/``bytes`` are EXCLUDED: they
    are sequences of characters, and ``incomplete='ppc'`` must be rejected as
    malformed rather than expanded to ``['p', 'p', 'c']``."""
    return (isinstance(v, (_abc.Sequence, _abc.Set))
            and not isinstance(v, (str, bytes, bytearray)))


def _usable_checks(blk):
    """The check mapping of a block, or ``{}`` if it is absent/malformed."""
    c = blk.get("checks")
    return c if _is_mapping(c) else {}


def gate(blocks, *, required=REQUIRED_BLOCKS, bypasses=None):
    """FAIL-CLOSED.  Missing block == failure.  Any False check == failure.

    RATIFIED 2026-07-29 by the PI (decision 8): the fail-closed framework
    itself, matched-configuration SBC, and chi2/dof <= 3 as the closure
    requirement.  The authoritative record, including the two tolerances the PI
    DECLINED to ratify, is ``CDDF_analysis.hbi_mcmc.ratification`` and is
    stamped into the verdict as ``ratification``.

    FIVE fail-open holes closed 2026-07-29.  Each is stated as a CODE PATH,
    because that is what was observed; no claim is made here about any
    particular file having been found on disk.

    1. ``required`` may only ever GROW.  It used to be an override, and
       ``run_evidence --mode sbc`` passed ``required=("coverage_sbc",)``,
       which silenced the four absent blocks: that call path returns
       ``stampable=True, paper_facing=True, n_checks=2``, and anything it
       wrote would carry that verdict.  ``REQUIRED_BLOCKS`` is now unioned in
       unconditionally, so no caller can narrow the gate.  A partial run is
       REPORTABLE; it is never STAMPABLE.

    2. A block whose value is not a dict is INVALID, not absent-and-ignored.
       The old loop did ``if not isinstance(blk, dict): continue`` and the
       missing-list caught only ``None``/``{}`` -- so ``blocks['ppc'] = []``
       (or ``''``, ``0``, ``False``) yielded ``stampable=True, missing=[]``.
       The omission tests used exactly the None/{} pair and so passed
       vacuously.

    3. ``bypasses`` -- any gate-bypass flag actually used by the run (e.g.
       ``--allow-low-farr``, ``--allow-open-forward-model``).  A bypass is
       RECORDED in the verdict and forces ``paper_facing=False`` (and
       ``stampable=False``): a result obtained by switching a gate off cannot
       be certified by that gate.

    4. Hole 2 stopped one level too shallow.  Individual CHECK VALUES were
       still coerced with ``bool(v)``, so ``{'checks': {'ppc_pval_ok': 'no'}}``
       and ``{'checks': {'ppc_pval_ok': [0]}}`` STAMPED -- both are truthy in
       Python and both mean "not ok" to a human.  ``incomplete`` had the
       mirror hole (``list(blk.get('incomplete') or [])`` silently dropped a
       non-sequence such as ``0``), and a non-mapping ``checks`` raised
       ``AttributeError`` out of the gate rather than failing closed.  A check
       value must now be a genuine ``bool``/``np.bool_``, ``checks`` must be a
       mapping, and ``incomplete`` must be a genuine sequence (a ``str`` is
       NOT accepted: it would explode into per-character entries).  Anything
       else is a MALFORMED-EVIDENCE failure, recorded as a False check.

    5. A required block could pass by SILENCE.  ``gate`` only ever inspected
       the checks a block volunteered, so a ``coverage_sbc`` block that simply
       did not mention whether its configuration matched the run's was
       indistinguishable from one that matched.  ``REQUIRED_CHECKS`` now names
       the checks a required block MUST report, and an absent one is
       synthesised as False.  This is what makes the ratified
       matched-configuration-SBC statement enforceable rather than advisory:
       "unspecified" cannot be enforced by the block that failed to specify.
    """
    reasons, checks = [], {}
    # (1) required may only GROW -- narrowing it is how the SBC-only artifact
    #     came to be stamped.  Union, always.
    required = tuple(dict.fromkeys(tuple(REQUIRED_BLOCKS) + tuple(required)))
    bypasses = dict(bypasses or {})
    # (2) any non-dict block is invalid; a required non-dict is ALSO missing
    invalid = [name for name, blk in blocks.items()
               if blk is not None and not isinstance(blk, dict)]
    for name in invalid:
        reasons.append(f"invalid evidence block (not a dict): {name} "
                       f"(got {type(blocks[name]).__name__})")
        checks[f"{name}.__well_formed__"] = False
    missing = [b for b in required
               if b not in blocks or blocks.get(b) is None
               or not isinstance(blocks.get(b), dict)]
    for b in missing:
        reasons.append(f"missing required evidence block: {b}")
    incomplete = {}
    for name, blk in blocks.items():
        if not isinstance(blk, dict):
            continue
        # (4a) ``checks`` must be a MAPPING.  A truthy non-mapping used to
        #      raise AttributeError straight out of the gate; a gate that
        #      crashes is not a gate that fails closed.
        raw_checks = blk.get("checks")
        if raw_checks is None or (not raw_checks and not _is_mapping(raw_checks)):
            raw_checks = {}          # absent/empty -> handled as "no checks"
        elif not _is_mapping(raw_checks):
            reasons.append(f"malformed evidence in {name}: 'checks' is not a "
                           f"mapping (got {type(raw_checks).__name__})")
            checks[f"{name}.__checks_well_formed__"] = False
            raw_checks = {}
        for k, v in raw_checks.items():
            # (4b) a check value must be a GENUINE bool.  `bool(v)` used to
            #      coerce, and 'no' / [0] are truthy while meaning "not ok".
            if not _is_bool(v):
                checks[f"{name}.{k}"] = False
                reasons.append(
                    f"malformed check {name}.{k}: value is not a bool "
                    f"(got {type(v).__name__} {v!r}) -- refusing to coerce")
                continue
            checks[f"{name}.{k}"] = bool(v)
            if not v:
                reasons.append(f"failed check: {name}.{k}")
        # (4c) ``incomplete`` must be a genuine sequence.  A non-sequence was
        #      silently dropped by ``list(... or [])``; a str would have been
        #      exploded into per-character entries.
        raw_inc = blk.get("incomplete")
        if raw_inc is None:
            inc = []
        elif _is_sequence(raw_inc):
            inc = list(raw_inc)
        else:
            inc = []
            reasons.append(f"malformed evidence in {name}: 'incomplete' is "
                           f"not a sequence (got {type(raw_inc).__name__} "
                           f"{raw_inc!r})")
            checks[f"{name}.__incomplete_well_formed__"] = False
        if inc:
            incomplete[name] = inc
            reasons.append(f"incomplete evidence in {name}: "
                           f"{', '.join(str(x) for x in inc)}")
        if not raw_checks and name in required:
            reasons.append(f"block {name} reports no checks at all")
    # a required block that produced no checks cannot pass
    for b in required:
        blk = blocks.get(b)
        if isinstance(blk, dict) and not _usable_checks(blk):
            checks[f"{b}.__present__"] = False
    # (5) STRUCTURALLY REQUIRED CHECKS -- a required block may not pass by
    #     SILENCE.  Closed 2026-07-29 with the matched-configuration-SBC
    #     ratification: before this, a coverage_sbc block that simply did not
    #     mention its configuration was indistinguishable from one whose
    #     configuration matched the run.
    for b, need in REQUIRED_CHECKS.items():
        if b not in required:
            continue
        blk = blocks.get(b)
        if not isinstance(blk, dict):
            continue                 # already recorded as missing/invalid
        have = _usable_checks(blk)
        for k in need:
            if k not in have:
                checks[f"{b}.{k}"] = False
                reasons.append(
                    f"required check {b}.{k} is ABSENT -- the block did not "
                    f"report it, and a required check may not pass by "
                    f"silence (see evidence.REQUIRED_CHECKS)")
    for b in sorted(bypasses):
        reasons.append(f"gate bypass in force: {b} ({bypasses[b]!r}) -- the "
                       f"artifact can never be paper-facing")
    stampable = (not missing) and (not incomplete) and (not invalid) and (
        not bypasses) and bool(checks) and all(checks.values())
    # A SECOND, WEAKER verdict, because "the z-resolved product fails but the
    # z-marginalised one holds" is the project's actual situation and deserves
    # a name rather than a blanket refusal: everything passes EXCEPT that the
    # per-z intervals do not all cover.  ``stampable`` (the full verdict) still
    # requires the z-resolved check, so nothing is silently promoted.
    _Z = "ztilt.ztilt_z_resolved_ok"
    relaxed = {k: v for k, v in checks.items() if k != _Z}
    stampable_integrated_only = bool(
        (not missing) and (not incomplete) and (not invalid)
        and (not bypasses) and relaxed
        and all(relaxed.values()) and not stampable)
    return {
        "checks": checks,
        "required_blocks": list(required),
        "required_checks": {k: list(v) for k, v in REQUIRED_CHECKS.items()},
        # WHICH criteria a deciding authority authorised to refuse this work,
        # and which are report-only because they were NOT ratified.  Carried in
        # the verdict itself so a reader of the JSON never has to guess.
        "ratification": _RAT.ratification_stamp(),
        "invalid_blocks": invalid,
        "bypasses": bypasses,
        "stampable_integrated_only": stampable_integrated_only,
        "stampable_integrated_only_note": (
            "every gate passes except the per-z coverage: only the INTEGRATED "
            "(z-marginalised) product is defensible from this run"),
        "missing_blocks": missing,
        "incomplete": incomplete,
        "n_checks": len(checks),
        "n_failed": int(sum(1 for v in checks.values() if not v)),
        "stampable": bool(stampable),
        # explicit, not merely implied by ``stampable``: a bypassed run is
        # never paper-facing even if some future edit loosens ``stampable``.
        "paper_facing": bool(stampable and not bypasses),
        "estimand": "POSTERIOR_MEDIAN_CI" if stampable else "NOT_STAMPABLE",
        "reasons": reasons,
        "policy_note": (
            "SAMPLER CONVERGENCE ALONE IS NOT ENOUGH: ppc, closure, "
            "coverage_sbc and ztilt are independent gates and each one alone "
            "blocks the stamp."),
    }


def _git(paths=("evidence.py", "sbc.py", "run_evidence.py", "model_a.py",
                "forward.py", "pack.py")):
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                    text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--"]
            + [os.path.join(here, p) for p in paths], cwd=here,
            text=True).strip()
        return c + ("-dirty" if dirty else "")
    except Exception:                                          # pragma: no cover
        return "unknown"


def assemble_evidence(blocks, *, provenance=None, required=REQUIRED_BLOCKS,
                      bypasses=None):
    g = gate(blocks, required=required, bypasses=bypasses)
    prov = dict(provenance or {})
    prov["bypasses"] = dict(bypasses or {})
    prov.setdefault("routine", "CDDF_analysis/hbi_mcmc/run_evidence.py")
    prov.setdefault("module", "CDDF_analysis/hbi_mcmc/evidence.py")
    prov.setdefault("code_commit", _git())
    prov.setdefault("date", time.strftime("%Y-%m-%d"))
    return {"schema": _SCHEMA, "blocks": blocks, "gate": g,
            # top level, not buried in the gate: the ratification state is the
            # first thing a reader of this artifact needs.
            "ratification": _RAT.ratification_stamp(),
            "provenance": prov,
            "scope": prov.get("scope", "MOCK ONLY")}
