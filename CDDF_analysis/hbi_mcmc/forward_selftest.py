# -*- coding: utf-8 -*-
"""forward_selftest.py — the pure forward-model TRUTH-FOLD self-test (no sampling).

The decisive, cheap gate on the Model A / NUTS route.  Take a pack's OWN truth
f(N, z) (from ``truth_counts``), fold it through that pack's OWN
kernel / completeness / g / dX / FP machinery (``forward.build_consts`` +
``forward.fold_mu``) at the truth-equivalent parameter point, and compare the
predicted expected counts ``mu`` against the pack's ACTUAL observed ``counts``.

If the forward model is faithful this must reproduce the counts to within
Poisson noise.  It needs NO MCMC: any failure here is upstream of NUTS and no
amount of sampling can fix it.

Parameter point used for the fold ("truth-equivalent"):

    theta_pop  = log f_truth,  f_truth[b,k] = truth_counts[b,k] / (dX_tot[k] dN_b)
    psi_c      = 0             (completeness at the Jeffreys molly point surface)
    psi_k_delta= 0             (response coefficients at their fitted point)
    log_t      = 0             (transfer factors at their prior centre)
    lam_fp     = fp_counts / ell_eff   (the loa-0 FP point estimate)

TRUTH-SUPPORT EXTENSION (``--truth-floor``)
-------------------------------------------
The pack's ``truth_counts`` is truncated at the bottom of the reporting grid
(N_true >= nhat_edges[0] = 19.5); the extractor's truth cut carries the SAME
19.5 floor as the reporting grid.  But the forward response has a POSITIVE
mean bias (~+0.1 dex) and a ~0.2 dex width, so the observed n-hat bins just
above 19.5 are fed overwhelmingly by TRUE systems BELOW 19.5 that the pack
simply does not carry.  Folding the truncated truth therefore under-predicts
the bottom n-hat bins by construction -- the same CLASS of one-sided-support
bug as B16.  ``--truth-floor 18.5`` (say) extends the true-N grid downward with
a power law fitted to the pack's own truth over a clean, un-truncated window
and re-folds; the difference between the two runs IS the size of the
truncation defect.

Usage
-----
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m CDDF_analysis.hbi_mcmc.forward_selftest \
        --pack /path/modelA_pack_2lpt0.npz [--out out.json] \
        [--truth-floor 18.0] [--fit-lo 19.8 --fit-hi 20.8] [--no-fp]

MOCKS ONLY (truth_counts exists only for mocks); refuses real-LOA packs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np

__all__ = ["truth_f", "extend_pack_truth", "selftest", "ratio_tables",
           "poisson_z", "ratio_span", "ratio_span_null"]


# --------------------------------------------------------------------------
# truth -> f(N, z)
# --------------------------------------------------------------------------
def truth_f(pack):
    """f_truth[b, k] = truth_counts[b, k] / (dX_tot[k] * dN_b)  (per dex, per dX).

    dX_tot[k] = sum_s dX[k, s]: the truth histogram is NOT stratified by the
    fold's SNR axis, so the fold's per-stratum dX[k, s] re-allocates the truth
    across strata in proportion to pathlength (checked separately against
    ``truth_counts_bks``).
    """
    tc = np.asarray(pack.truth_counts, float)             # (B, Kf)
    dX_tot = np.asarray(pack.dX, float).sum(axis=1)       # (Kf,)
    dN = np.diff(np.asarray(pack.ntrue_edges, float))     # (B,)
    denom = dX_tot[None, :] * dN[:, None]
    f = np.zeros_like(tc)
    ok = denom > 0
    f[ok] = tc[ok] / denom[ok]
    return f


def _fit_truth_powerlaw(pack, fit_lo, fit_hi):
    """Least-squares log10 f = a + s * (N - N0) over the clean window, per z-bin.

    Returns (slope_per_dex, log_f_at_grid_bottom) using the POOLED-in-N,
    per-z-bin fit so the downward extension keeps the truth's own z shape.
    """
    f = truth_f(pack)
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    sel = (Nc >= fit_lo - 1e-9) & (Nc < fit_hi - 1e-9)
    if sel.sum() < 3:
        raise ValueError("truth power-law fit window has < 3 bins")
    x = Nc[sel]
    slopes, intercepts = [], []
    for k in range(f.shape[1]):
        y = f[sel, k]
        good = y > 0
        if good.sum() < 3:
            slopes.append(np.nan)
            intercepts.append(np.nan)
            continue
        A = np.vstack([np.ones(good.sum()), x[good] - Nc[0]]).T
        coef, *_ = np.linalg.lstsq(A, np.log(y[good]), rcond=None)
        intercepts.append(coef[0])
        slopes.append(coef[1])
    return np.asarray(slopes), np.asarray(intercepts), Nc[0]


def extend_pack_truth(pack, truth_floor, fit_lo, fit_hi):
    """Return (pack2, f2) whose TRUE-N grid runs down to ``truth_floor``.

    Only the ntrue axis is extended -- the OBSERVED n-hat axis (and therefore
    ``counts``) is untouched, so the comparison stays like-for-like.  The
    extension's f is the power law fitted to the pack's own truth over
    [fit_lo, fit_hi) per z bin.

    NOTE the pack schema requires ntrue_edges == nhat_edges (v1); this function
    deliberately breaks that for the DIAGNOSTIC only and never saves the pack.
    """
    ntrue = np.asarray(pack.ntrue_edges, float)
    step = float(np.diff(ntrue)[0])
    n_extra = int(round((ntrue[0] - truth_floor) / step))
    if n_extra <= 0:
        return pack, truth_f(pack)
    lo_edges = ntrue[0] - step * np.arange(n_extra, 0, -1)
    new_edges = np.concatenate([lo_edges, ntrue])
    slopes, intercepts, N0 = _fit_truth_powerlaw(pack, fit_lo, fit_hi)
    new_Nc = 0.5 * (new_edges[:-1] + new_edges[1:])
    f_old = truth_f(pack)
    Kf = f_old.shape[1]
    f_new = np.zeros((len(new_Nc), Kf))
    f_new[n_extra:, :] = f_old
    for k in range(Kf):
        if not np.isfinite(slopes[k]):
            continue
        f_new[:n_extra, k] = np.exp(
            intercepts[k] + slopes[k] * (new_Nc[:n_extra] - N0))
    pack2 = dataclasses.replace(pack, ntrue_edges=new_edges,
                                truth_counts=None, truth_counts_bks=None)
    return pack2, f_new


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------
def selftest(pack, f=None, *, use_fp=True, psi_c=None, resp_clamp="both"):
    """Fold the truth through the pack's own machinery; return mu and counts."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu, fold_mu_fp

    consts = build_consts(pack, resp_clamp=resp_clamp,
                          allow_unclamped_response=(resp_clamp == "off"))
    if f is None:
        f = truth_f(pack)
    f = np.asarray(f, float)
    theta = np.log(np.clip(f, 1e-300, None))
    C, S = consts.n_c, consts.n_s
    if psi_c is None:
        psi_c = np.zeros((S, consts.n_molly))
    lam_fp = (np.asarray(pack.fp_counts, float) / float(pack.fp_ell_eff)
              if use_fp else np.zeros((C, S)))
    mu = np.asarray(fold_mu(jnp.asarray(theta), jnp.asarray(psi_c),
                            jnp.zeros((2, consts.n_sr, consts.n_zr)),
                            jnp.zeros(consts.n_kk), jnp.asarray(lam_fp),
                            consts))
    # THE fold's own FP term, called with the SAME log_t the fold was given --
    # not a re-typed copy.  The copy that used to live here silently dropped
    # the exp(log_t) factor (inert only because log_t is zero here) and had to
    # be repaired by hand alongside fold_mu on 2026-08-05.
    mu_fp = np.asarray(fold_mu_fp(jnp.zeros(consts.n_kk),
                                  jnp.asarray(lam_fp), consts))
    return dict(mu=mu, mu_fp=mu_fp, mu_sig=mu - mu_fp,
                counts=np.asarray(pack.counts, float), consts=consts, f=f)


#: the floor added to the variance so that a mu==0 cell cannot divide by zero.
#: It is NOT a regularizer with a statistical meaning: at mu==0 and obs>0 the
#: score residual is mathematically +infinity and this floor merely renders it
#: as a very large finite number (obs * 1e6) so the gate FAILS rather than
#: raising.  See ``poisson_z``, "EMPTY AND ZERO-PREDICTION BINS".
_Z_VAR_FLOOR = 1e-12


def poisson_z(mu, obs):
    r"""THE exact definition of the ``z`` that ``|z| <= 5`` thresholds.

    (Decision 8, item 3, 2026-07-29: the criterion was called "malformed as
    currently stated". This docstring is the restatement. Nothing about the
    arithmetic changed; it was hoisted out of a nested closure so it could be
    named, documented and pinned by a test.)

    DEFINITION
    ----------
    For an aggregate cell with predicted expected count ``mu`` and observed
    count ``obs``,

        z = (obs - mu) / sqrt(max(mu, 1e-12))

    i.e. the **Pearson/Poisson score residual of the observed count about the
    predicted count**, with the variance taken as the PREDICTED mean (the
    Poisson variance under the null "the forward model is right"), not the
    observed count and not a fitted variance.

    Every clause of that sentence is load-bearing, so, itemised:

    WHICH RATIO / WHICH ESTIMATOR
        None. ``z`` is NOT computed from the ``mu/obs`` ratio that sits beside
        it in these tables and it involves NO posterior, NO sampler and NO
        estimator. ``mu`` is the pack's own truth folded through the pack's own
        kernel at the truth-equivalent parameter point (see the module
        docstring); ``obs`` is the pack's recorded counts. The whole statistic
        is a deterministic function of the pack.

    SIGN
        ``obs - mu``. z > 0 means the model UNDER-predicts. (The ratio column
        is mu/obs, so ratio > 1 pairs with z < 0. This inversion is a genuine
        readability trap and is why both are reported.)

    OVER WHAT ROWS
        Per AGGREGATE cell, never pooled, and never per (N,z,SNR) voxel:
          * ``total``   -- one z over the sum of all cells with dX > 0.
          * ``by_nhat`` -- one z per observed-N-hat bin (sum over z and SNR).
          * ``by_z``    -- one z per fine-z bin (sum over N-hat and SNR).
          * ``by_snr``  -- one z per SNR stratum (sum over N-hat and z).
        Cells with dX == 0 are zeroed out of BOTH mu and obs before any sum, so
        they can contribute to neither.
        The GATE then thresholds ``max |z|`` over the rows of an arm that have
        ``obs > 0``; for ``by_nhat`` it additionally keeps only rows with
        ``lo >= nhat_edges[0]``, i.e. the REPORTED bins, excluding any
        below-floor basis-pad bins.

    WHAT IS IN THE DENOMINATOR
        ``sqrt(mu)`` only. This is the Poisson standard deviation of ``obs``
        under the null. It contains:
          * NO nuisance uncertainty. psi_c, psi_k, t and lam_fp are all held at
            their point values for the fold, so the uncertainty on completeness,
            on the response coefficients, on the transfer factors and on the FP
            rate contributes NOTHING to the denominator.
          * NO Monte-Carlo or truth-estimation uncertainty on ``mu`` itself,
            although ``mu`` is built from a f_truth that is itself a finite-count
            estimate.
        Both omissions make ``|z|`` LARGER than a fully propagated residual
        would be, i.e. the gate is conservative in the refusing direction. It is
        therefore usable as a tripwire and is NOT usable as a calibrated
        significance.

    EMPTY AND ZERO-PREDICTION BINS
        * ``obs == 0``: the row is DROPPED by the gate (never counted as a
          pass and never counted as a failure). Consequence, stated because it
          is a real hole: a bin the model predicts should be full but which is
          observed empty is INVISIBLE to this arm. That direction of failure is
          caught, if at all, by the total.
        * ``obs > 0`` and ``mu == 0``: z = obs / 1e-6 -- a huge finite number,
          so the gate FAILS, which is the correct verdict (the model assigns
          zero probability to something that happened). The value itself is an
          artifact of the floor and must not be quoted.
        * ``obs == 0`` and ``mu == 0``: dropped, as above.

    CHI2/DOF
        ``chi2/dof = sum(z^2) / n_rows`` over exactly the ``by_nhat`` rows the
        gate keeps. ``dof = n_rows``, NOT ``n_rows - n_params``, because the
        truth fold fits nothing: there is no parameter estimated from ``obs``.

    WHY 5 IS NOT SCALE-FREE
        For a fixed FRACTIONAL model error ``mu = (1+d) * obs_expected``, the
        residual grows as ``z ~ -d * sqrt(mu)``. So the same physical bias
        crosses |z| = 5 at ``mu ~ 25/d^2``: a 10% bias is invisible below ~2500
        counts per bin and unmissable above it. A fixed threshold of 5 therefore
        tightens without limit as the survey grows, and on a count-starved pack
        it will pass a forward model that is wrong by tens of percent. THAT is
        the sense in which "|z| <= 5" is ill-posed as a standalone criterion,
        and it is the exact defect the (unratified) ratio-span arms were
        invented to cover.

        🔴 The criterion is nevertheless kept ARMED, with its purpose narrowed
        to what it can actually do.  An earlier draft of this line said "kept,
        RATIFIED".  IT IS NOT RATIFIED.  Decision 8 item 3 called |z| <= 5
        MALFORMED AS STATED and sent it back for restatement; this docstring IS
        the restatement, and a restatement is not a ratification.  Its status is
        ``ratification.RESTATED_NOT_RATIFIED`` -- it gates, and no deciding
        authority authorised it to (see ``ratification.OPEN_PI_DECISIONS
        ['z_arms_gate_unratified']``).  What it can actually do: it is a
        tripwire against a forward model
        that is wrong by ORDERS OF MAGNITUDE, which is the observed failure
        mode.  Verified against the committed ``rung9_forward_selftest.json``
        (2lpt0, resp_clamp="both", n_pad_bins=0 -- i.e. the UNPADDED pack; the
        earlier draft of this docstring called it "v1.1", which the artifact
        contradicts): total z = +93.3, worst n-hat bin z = +216.4 at
        [19.5, 19.6). The other two mocks in that artifact agree in sign and
        order (london0 +74.7 / +169.1, saclay0 +83.6 / +190.4).
        The closest defensible description of the implemented statistic is
        "the maximum absolute Poisson score residual over aggregate marginal
        cells, with the model's own predicted mean as the variance and no
        nuisance propagation, thresholded at a fixed 5". It is NOT a 5-sigma
        significance statement, NOT multiplicity-corrected over the ~29 + 15 + 8
        rows it maximises over, and NOT a goodness-of-fit test.
    """
    mu = np.asarray(mu, float)
    obs = np.asarray(obs, float)
    return (obs - mu) / np.sqrt(np.maximum(mu, _Z_VAR_FLOOR))


def ratio_span(rows):
    r"""THE exact definition of ``ratio_span_by_z`` / ``ratio_span_by_snr``.

    🔴 UNRATIFIED STATISTIC.  The PI declined (decision 8, 2026-07-29) to
    ratify the thresholds 0.10 / 0.15 that were attached to this statistic, and
    required that it be "defined and calibrated prospectively".  This docstring
    is the definition half; the calibration procedure is
    ``docs/ratio_span_calibration_spec.md`` and the null sampler is
    ``ratio_span_null`` below.  The statistic IS computed and reported on every
    run and does NOT contribute to pass/fail (``ratification.py``).

    DEFINITION
    ----------
    Let ``R`` be the rows of one marginal arm of ``ratio_tables`` -- ``by_z``
    (one row per FINE-z bin) or ``by_snr`` (one row per SNR stratum) -- each row
    an aggregate over the other two axes with ``dX == 0`` cells zeroed out.
    Keep the subset

        R+ = { r in R : obs_r > 0 and mu_r / obs_r is finite }

    and define, with ratio_r = mu_r / obs_r,

        ratio_span(R) = max_{r in R+} ratio_r  -  min_{r in R+} ratio_r      if |R+| >= 2
        ratio_span(R) = 0                                                    otherwise

    It is a RANGE (a max-minus-min), not a variance, not an sd, and not a
    normalised dispersion.  Units: dimensionless, since it is a difference of
    two ratios of counts.

    PROPERTIES THAT MATTER FOR CALIBRATION -- stated because they are the
    reasons a threshold cannot be guessed:

    * It is a RANGE, so its null distribution grows with the NUMBER OF ROWS
      (~sqrt(2 log n) sd for Gaussian-ish rows).  A 15-bin fine-z arm and a
      3-bin one do not share a threshold.  0.10 was chosen with neither row
      count in mind.
    * It is heteroscedastic across rows: the per-row Poisson sd of ratio_r is
      ~ mu_r / obs_r^{3/2} ~ 1/sqrt(obs_r), so a count-starved row dominates the
      range for purely statistical reasons.  The span therefore mixes a real
      shape systematic with the noise of the emptiest row.
    * It uses ``obs`` in the DENOMINATOR, so E[ratio] does not exist in the
      strict sense (Poisson obs has positive mass at 0 and the row is dropped
      rather than being infinite).  Dropping ``obs == 0`` rows makes the
      statistic conditional on which rows happened to be non-empty.
    * The ``|R+| < 2`` case returns 0, i.e. PASSES VACUOUSLY.  On the SBC-style
      1-stratum grid ``ratio_span_by_snr`` is identically 0 and its arm has
      never been able to fire.  A vacuous 0 is indistinguishable, downstream,
      from a measured 0.

    A better-behaved statistic would be a range (or sd) of ``log(mu/obs)``, or
    a dispersion of ``poisson_z`` across rows, both of which have a stable null.
    This function deliberately does NOT redefine anything: it implements what
    the production gate has been computing, so that the calibration is a
    calibration OF THE DEPLOYED STATISTIC.  The candidate replacements are
    written up in the spec and are a PI decision, not a silent edit.

    Parameters
    ----------
    rows : sequence of mapping
        Rows of one arm of ``ratio_tables`` (``by_z`` or ``by_snr``); each row
        needs ``obs`` and ``ratio``.

    Returns
    -------
    dict with ``span``, ``lo``, ``hi``, ``n_rows_used``, ``vacuous``.
    """
    keep = [r for r in (rows or [])
            if float(r.get("obs", 0.0)) > 0
            and np.isfinite(float(r.get("ratio", np.nan)))]
    ratios = np.array([float(r["ratio"]) for r in keep], float)
    if len(ratios) < 2:
        return {"span": 0.0, "lo": None, "hi": None,
                "n_rows_used": int(len(ratios)), "vacuous": True}
    return {"span": float(ratios.max() - ratios.min()),
            "lo": float(ratios.min()), "hi": float(ratios.max()),
            "n_rows_used": int(len(ratios)), "vacuous": False}


def ratio_span_null(pack, *, n_draws=2000, seed=0, resp_clamp="both",
                    arms=("by_z", "by_snr"), res=None):
    """Sample the NULL distribution of ``ratio_span`` (the prospective
    calibration of the two tolerances the PI declined to ratify).

    THE NULL, stated exactly.  "The forward model is right": the folded
    prediction ``mu`` is taken as the true expected counts, the nuisances
    (psi_c, psi_k, t, lam_fp) are held FIXED at the point values the fold used,
    and the observed cell counts are independent Poisson,

        obs*_{c,k,s} ~ Poisson(mu_{c,k,s})     independently over cells,
        obs*_{c,k,s} = 0 wherever dX_{k,s} == 0 (as in ``ratio_tables``),

    Each replicate is marginalised exactly as ``ratio_tables`` marginalises,
    and ``ratio_span`` is recomputed.  The resulting distribution is the
    span you would see from PURE COUNTING NOISE with a perfect forward model,
    for THIS pack's row counts and THIS pack's per-row exposure -- which is why
    the threshold is pack-dependent and cannot be a global constant.

    WHAT THIS NULL DOES NOT CONTAIN (so it is a LOWER bound on the null width):
    nuisance uncertainty, response-coefficient uncertainty, the Monte-Carlo
    error of ``f_truth`` itself, and any real overdispersion of the counts
    (clustering of absorbers along sightlines). A threshold set from this null
    is therefore ANTI-conservative -- it will fire too often. The spec says so.

    Cost: pure numpy, no MCMC, no JAX beyond the single fold. Measured at
    ~1e-4 s per replicate on the SBC-size synthetic pack (see the spec).

    Returns a dict of per-arm ``{"span_draws", "quantiles", "n_rows_used"}``
    plus the observed span, plus the metadata a threshold proposal needs.
    """
    if res is None:
        res = selftest(pack, resp_clamp=resp_clamp)
    mu = np.asarray(res["mu"], float)
    dxpos = np.asarray(pack.dX, float) > 0
    mask3 = np.broadcast_to(dxpos[None, :, :], mu.shape)
    mu_m = np.where(mask3, mu, 0.0)

    axes = {"by_z": (0, 2), "by_snr": (0, 1)}      # axes SUMMED OVER
    rng = np.random.default_rng(seed)
    obs_star = rng.poisson(np.broadcast_to(mu_m, (n_draws,) + mu_m.shape))

    out = {"n_draws": int(n_draws), "seed": int(seed),
           "resp_clamp": str(resp_clamp), "arms": {},
           "null_note": (
               "obs* ~ independent Poisson(mu) with nuisances FIXED; contains "
               "NO nuisance, response-coefficient or f_truth Monte-Carlo "
               "uncertainty and NO count overdispersion, so it is a LOWER "
               "bound on the true null width and a threshold read off it is "
               "ANTI-conservative. See docs/ratio_span_calibration_spec.md.")}
    for arm in arms:
        ax = axes[arm]
        mu_r = mu_m.sum(axis=ax)                              # (n_rows,)
        obs_r = obs_star.sum(axis=tuple(a + 1 for a in ax))   # (n_draws, n_rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(obs_r > 0, mu_r[None, :] / obs_r, np.nan)
        n_used = np.isfinite(ratio).sum(axis=1)
        hi = np.nanmax(np.where(np.isfinite(ratio), ratio, -np.inf), axis=1)
        lo = np.nanmin(np.where(np.isfinite(ratio), ratio, np.inf), axis=1)
        span = np.where(n_used >= 2, hi - lo, 0.0)
        qs = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
        out["arms"][arm] = {
            "n_rows": int(mu_r.size),
            "n_rows_nonempty_median": float(np.median(n_used)),
            "quantiles": {str(q): float(np.quantile(span, q)) for q in qs},
            "mean": float(span.mean()), "sd": float(span.std(ddof=1)),
            "span_draws_summary": [float(span.min()), float(span.max())],
        }
    tab = ratio_tables(res, pack)
    for arm in arms:
        out["arms"][arm]["observed"] = ratio_span(tab.get(arm))
    return out


def ratio_tables(res, pack):
    """Per-cell / marginal mu-over-counts ratios + Poisson z-scores.

    ``z`` is ``poisson_z``; read its docstring for the exact definition, the
    row set, the denominator and the empty-bin convention.
    """
    mu = res["mu"]
    obs = res["counts"]
    nhat = np.asarray(pack.nhat_edges, float)
    zf = np.asarray(pack.zf_edges, float)
    dxpos = np.asarray(pack.dX, float) > 0

    _z = poisson_z          # THE definition; see its docstring

    mask3 = np.broadcast_to(dxpos[None, :, :], mu.shape)
    mu_m = np.where(mask3, mu, 0.0)
    obs_m = np.where(mask3, obs, 0.0)

    out = {
        "total": dict(mu=float(mu_m.sum()), obs=float(obs_m.sum()),
                      ratio=float(mu_m.sum() / max(obs_m.sum(), 1e-30)),
                      z=float(_z(mu_m.sum(), obs_m.sum()))),
        "by_nhat": [], "by_z": [], "by_snr": [],
    }
    for c in range(mu.shape[0]):
        m, o = float(mu_m[c].sum()), float(obs_m[c].sum())
        out["by_nhat"].append(dict(
            lo=float(nhat[c]), hi=float(nhat[c + 1]), mu=m, obs=o,
            ratio=(m / o if o > 0 else float("nan")), z=float(_z(m, o))))
    for k in range(mu.shape[1]):
        m, o = float(mu_m[:, k].sum()), float(obs_m[:, k].sum())
        out["by_z"].append(dict(
            lo=float(zf[k]), hi=float(zf[k + 1]), mu=m, obs=o,
            ratio=(m / o if o > 0 else float("nan")), z=float(_z(m, o))))
    for s in range(mu.shape[2]):
        m, o = float(mu_m[:, :, s].sum()), float(obs_m[:, :, s].sum())
        out["by_snr"].append(dict(
            s=s, mu=m, obs=o,
            ratio=(m / o if o > 0 else float("nan")), z=float(_z(m, o))))
    # chi2/dof over the REPORTED n-hat bins with obs > 0 — the SAME definition
    # run_posterior.forward_closure_gate uses.  It lives in ``total`` because
    # ``_closure_verdict`` reads it from there: before 2026-07-29 the key was
    # never emitted, so ``tot.get("chi2_dof", 0.0)`` always evaluated 0.0 and
    # the chi2 leg of ``--require-closure`` could NEVER fire (fail-OPEN).
    floor = float(nhat[0])
    zs = np.array([b["z"] for b in out["by_nhat"]
                   if b["obs"] > 0 and b["lo"] >= floor - 1e-9], float)
    out["total"]["chi2_dof"] = float((zs ** 2).sum() / max(len(zs), 1))
    out["total"]["n_gate_bins"] = int(len(zs))
    return out


def print_tables(tab, title=""):
    print(f"\n=== {title} ===")
    t = tab["total"]
    print(f"TOTAL   mu={t['mu']:12.1f}  obs={t['obs']:12.1f}  "
          f"ratio={t['ratio']:.4f}  z={t['z']:+.1f}  "
          f"chi2/dof={t.get('chi2_dof', float('nan')):.1f}")
    print(" n-hat bin        mu         obs      mu/obs      z")
    for r in tab["by_nhat"]:
        print(f" [{r['lo']:.1f},{r['hi']:.1f})  {r['mu']:10.2f} {r['obs']:10.0f} "
              f"  {r['ratio']:8.4f}  {r['z']:+8.1f}")
    print("   z bin          mu         obs      mu/obs      z")
    for r in tab["by_z"]:
        print(f" [{r['lo']:.1f},{r['hi']:.1f})  {r['mu']:10.2f} {r['obs']:10.0f} "
              f"  {r['ratio']:8.4f}  {r['z']:+8.1f}")


# --------------------------------------------------------------------------
# structural probes (the hypothesis battery)
# --------------------------------------------------------------------------
def structural_probes(pack):
    """Cheap, decisive structural checks on the fold's ingredients."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K

    consts = build_consts(pack, allow_unclamped_response=True,
                          resp_clamp=("both" if pack.resp_N_fit_range is not None
                                      else "off"))
    probes = {}
    probes["resp_clamp"] = consts.resp_clamp
    probes["has_resp_N_fit_range"] = pack.resp_N_fit_range is not None

    # (c) kernel row mass: sum_c K[s,K,c,b] -- how much probability the observed
    #     n-hat window RETAINS per true-N bin (1.0 = nothing lost off the grid).
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    rowmass = K.sum(axis=2)                       # (S, KK, B)
    probes["kernel_rowmass_min"] = float(rowmass.min())
    probes["kernel_rowmass_max"] = float(rowmass.max())
    probes["kernel_rowmass_by_b"] = rowmass.mean(axis=(0, 1)).tolist()

    # (a) Jacobian / bin-width: dN_b uniform?
    dN = np.diff(np.asarray(pack.ntrue_edges, float))
    probes["dN_b_unique"] = np.unique(np.round(dN, 12)).tolist()

    # (d) g surface: occupancy-weighted mean over z per molly cell (1.0 if the
    #     z-shape is level-preserving and hence NOT double-counting completeness)
    g = np.asarray(pack.g_grid, float)
    occ = np.asarray(pack.g_occupancy, float)
    w = occ / np.maximum(occ.sum(axis=1, keepdims=True), 1e-30)
    probes["g_occ_weighted_mean_by_cell"] = (g * w).sum(axis=1).tolist()

    # (e) truth support: is truth_counts truncated at the grid bottom?
    ntrue = np.asarray(pack.ntrue_edges, float)
    probes["ntrue_lo"] = float(ntrue[0])
    probes["nhat_lo"] = float(np.asarray(pack.nhat_edges, float)[0])
    probes["truth_floor_equals_grid_floor"] = bool(
        abs(float(ntrue[0]) - float(np.asarray(pack.nhat_edges, float)[0])) < 1e-9)

    # (f) index maps
    probes["kz_to_K"] = np.asarray(pack.kz_to_K).tolist()
    probes["s_to_sresp"] = np.asarray(consts.s_to_sresp).tolist()
    probes["K_to_zresp"] = np.asarray(consts.K_to_zresp).tolist()
    probes["b_to_cell"] = np.asarray(consts.b_to_cell).tolist()

    # (g) dX zero pattern vs counts
    dX = np.asarray(pack.dX, float)
    cnt = np.asarray(pack.counts, float).sum(axis=0)
    probes["n_zero_dX_cells"] = int((dX == 0).sum())
    probes["counts_in_zero_dX"] = float(cnt[dX == 0].sum())

    # truth vs counts totals (support asymmetry smoking gun)
    probes["truth_total"] = float(np.asarray(pack.truth_counts, float).sum()) \
        if pack.truth_counts is not None else None
    probes["counts_total"] = float(np.asarray(pack.counts, float).sum())

    # truth_counts_bks vs the dX-proportional re-allocation the fold implies
    if pack.truth_counts_bks is not None:
        tb = np.asarray(pack.truth_counts_bks, float)
        tc = np.asarray(pack.truth_counts, float)
        share = dX / np.maximum(dX.sum(axis=1, keepdims=True), 1e-30)   # (k,s)
        alloc = tc[:, :, None] * share[None, :, :]
        num = np.abs(alloc - tb).sum()
        probes["truth_strat_realloc_L1_frac"] = float(num / max(tb.sum(), 1e-30))
    return probes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _git():
    """FULL 40-char HEAD SHA (+ a dirty flag).

    This used to be ``rev-parse --short HEAD``, and the artifact it stamped
    (``rung9_forward_selftest.json``) carried ``code_commit: 'b76ded7'``.
    Abbreviated SHAs are a known defect class in this repo -- the provenance
    audit's ORPHANED class -- and that stamp was an instance of it: at b76ded7
    ``forward_selftest.py`` did not yet exist (it was added at 85ddd95), so the
    stamp named a commit at which the routine could not have run.  A 40-char
    SHA is checkable with ``git cat-file -e <sha>:<routine>``; a 7-char one
    invites exactly the mis-resolution that happened.

    NOTE the split from the dirty probe (2026-07-29).  A ``-dirty`` SUFFIX
    makes ``code_commit`` unusable with ``git cat-file -e <sha>:<routine>``,
    which is the entire reason the 40-char SHA is stamped.  Dirt is a separate
    BOOLEAN FIELD now, reported alongside the SCOPE it was measured over.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=here, text=True).strip()
    except Exception:
        return "unknown"


# the dirty probe is PATH-SCOPED; this string travels WITH the flag so no
# reader can upgrade it into a claim about the whole working tree.
_DIRTY_SCOPE = ("uncommitted changes under CDDF_analysis/hbi_mcmc/ ONLY -- "
                "this is NOT a whole-tree cleanliness claim")


def _git_dirty():
    """True if CDDF_analysis/hbi_mcmc/ has uncommitted changes.

    Path-scoped by design (the rest of the repo does not affect this routine's
    result), and therefore NOT evidence of a clean tree.  Unknown -> True:
    fail closed, an unprobeable tree is treated as dirty.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", here],
            cwd=here, text=True).strip())
    except Exception:
        return True


def _stamp_fields(mocks):
    """The provenance fields of the aggregate artifact.  Touches no pack, so
    it is directly testable."""
    return {
        "routine": "CDDF_analysis/hbi_mcmc/forward_selftest.py",
        "entry_point": "aggregate_report / --mock NAME=PATH",
        "date": time.strftime("%Y-%m-%d"),
        "code_commit": _git(),
        "code_dirty": bool(_git_dirty()),
        "code_dirty_scope": _DIRTY_SCOPE,
        "scope": (f"MOCK ONLY ({' / '.join(n for n, _ in mocks)}). "
                  f"No real-survey values."),
        "rederive": ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                     "MKL_NUM_THREADS=1 python -m "
                     "CDDF_analysis.hbi_mcmc.forward_selftest "
                     + " ".join(f"--mock {n}={p}" for n, p in mocks)
                     + " --out CDDF_analysis/hbi_mcmc/"
                       "rung9_forward_selftest.json"),
    }


#: the SYNTHETIC pack the ratio-span calibration in
#: ``docs/ratio_span_calibration_spec.md`` was measured on.  Named here so the
#: artifact, the spec and the test all quote the same object.
RATIO_SPAN_NULL_GRID = dict(
    nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
    zf_edges=np.round(np.arange(2.0, 2.4 + 1e-9, 0.1), 10),
    zc_edges=np.array([2.0, 2.2, 2.4]),
    snr_edges=np.array([0.0, 3.0, np.inf]),
    n_molly_cells=3,
)

#: 🔴 THE FALSE-ALARM RATE OF A RANGE STATISTIC DOES NOT TRANSFER BETWEEN GRIDS,
#: which the spec's own §1.1 item 1 predicted and which the first calibration
#: then ignored: 0.3434 for ``ratio_span_by_z_max = 0.10`` was measured on the
#: 5x4x2 pack above -- FOUR fine-z rows.  Production is FIFTEEN.  Every quote of
#: a false-alarm rate must name its geometry, so the calibration now runs on all
#: three and the artifact reports all three.
RATIO_SPAN_NULL_GEOMETRIES = {
    # the pack the spec's §4 table was measured on: C x Kf x S = 5 x 4 x 2
    "calib_5x4x2": dict(RATIO_SPAN_NULL_GRID),
    # production fine-z and SNR axes, reporting window [19.9, 21.6): 17 x 15 x 8
    "prod_17x15x8": dict(
        nhat_edges=np.round(np.arange(19.9, 21.6 + 1e-9, 0.1), 10),
        zf_edges=np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 10),
        zc_edges=np.array([2.0, 2.5, 3.0, 3.5]),
        snr_edges=np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf]),
        n_molly_cells=6),
    # the full REAL grid (pack.REAL_* edges): 29 x 15 x 8
    "prod_29x15x8": dict(
        nhat_edges=np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 10),
        zf_edges=np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 10),
        zc_edges=np.array([2.0, 2.5, 3.0, 3.5]),
        snr_edges=np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf]),
        n_molly_cells=6),
}

#: the injected peak-to-peak fractional z-tilt values of the power curve
RATIO_SPAN_POWER_TILTS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20,
                          0.30, 0.50)

_SPAN_ARM_AXES = {"by_z": (0, 2), "by_snr": (0, 1)}


def _arm_rows_from_obs(mu_m, obs_star, arm):
    """Marginalised (mu_r, obs*_r) for one arm; ``obs_star`` may carry a leading
    draw axis.  Marginalises EXACTLY as ``ratio_tables`` does."""
    ax = _SPAN_ARM_AXES[arm]
    mu_r = mu_m.sum(axis=ax)
    o_ax = tuple(a + 1 for a in ax) if obs_star.ndim == mu_m.ndim + 1 else ax
    return mu_r, obs_star.sum(axis=o_ax)


def _span_and_zmax(mu_r, obs_r):
    """``(span, max|z|)`` per draw, on the rows with ``obs_r > 0``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(obs_r > 0, mu_r[None, :] / obs_r, np.nan)
    fin = np.isfinite(ratio)
    hi = np.max(np.where(fin, ratio, -np.inf), axis=1)
    lo = np.min(np.where(fin, ratio, np.inf), axis=1)
    span = np.where(fin.sum(axis=1) >= 2, hi - lo, 0.0)
    z = np.where(obs_r > 0,
                 (obs_r - mu_r[None, :]) / np.sqrt(np.maximum(mu_r[None, :],
                                                              1e-12)),
                 0.0)
    return span, np.abs(z).max(axis=1)


def ratio_span_power(pack, *, tilts=RATIO_SPAN_POWER_TILTS, n_draws=2000,
                     seed=0, resp_clamp="both", res=None, span_max=None,
                     z_max=5.0):
    """The DETECTION CURVE the spec's §3 step 6 requires -- for BOTH guards.

    This is the measurement the "should the span arm stay disarmed?" question
    turns on, and it was never made.  The declined ``ratio_span_by_z`` arm and
    the still-armed ``z_zbin_max`` arm are exposed to the SAME injected
    systematic, so their sensitivities are directly comparable:

        obs*_{c,k,s} ~ Poisson( mu_{c,k,s} * (1 + d * (z_k - z_bar)/Delta_z) )

    with ``z_k`` the fine-z bin centres and ``Delta_z = z_max - z_min``, so
    ``d`` is the PEAK-TO-PEAK fractional z-tilt of the data relative to the
    forward model.  The model prediction stays the UNTILTED ``mu`` -- exactly
    the situation the arms exist to catch.  ``d = 0`` reproduces the null, so
    the first row of the curve IS the false-alarm rate and the two are measured
    in one place.

    Reports, per arm, the fraction of replicates exceeding the threshold at
    each ``d``, and the smallest ``d`` detected at 50% and at 90% (linear
    interpolation between the bracketing grid points; ``None`` if the curve
    never reaches that level on the grid).

    SAME OMISSIONS AS ``ratio_span_null`` (nuisance, response-coefficient and
    f_truth Monte-Carlo uncertainty; count overdispersion), so these are UPPER
    bounds on power as well as lower bounds on the null width.
    """
    from CDDF_analysis.hbi_mcmc.run_posterior import GATE
    if span_max is None:
        span_max = float(GATE["ratio_span_by_z_max"])
    if res is None:
        res = selftest(pack, resp_clamp=resp_clamp)
    mu = np.asarray(res["mu"], float)
    dxpos = np.asarray(pack.dX, float) > 0
    mu_m = np.where(np.broadcast_to(dxpos[None, :, :], mu.shape), mu, 0.0)

    zf = np.asarray(pack.zf_edges, float)
    zk = 0.5 * (zf[:-1] + zf[1:])
    dz = float(zk[-1] - zk[0]) if len(zk) > 1 else 1.0
    shape = (zk - zk.mean()) / (dz if dz > 0 else 1.0)          # (Kf,)

    mu_r, _ = _arm_rows_from_obs(mu_m, mu_m, "by_z")
    rows = {"span_threshold": float(span_max), "z_threshold": float(z_max),
            "n_rows_by_z": int(mu_r.size), "n_draws": int(n_draws),
            "seed": int(seed), "tilts": [float(d) for d in tilts],
            "curve": []}
    rng = np.random.default_rng(seed)
    for d in tilts:
        factor = 1.0 + float(d) * shape                          # (Kf,)
        mu_tilt = mu_m * factor[None, :, None]
        obs_star = rng.poisson(
            np.broadcast_to(mu_tilt, (n_draws,) + mu_tilt.shape))
        mu_row, obs_row = _arm_rows_from_obs(mu_m, obs_star, "by_z")
        span, zmax = _span_and_zmax(mu_row, obs_row)
        rows["curve"].append({
            "tilt_peak_to_peak": float(d),
            "p_span_arm_fires": float((span > span_max).mean()),
            "p_z_arm_fires": float((zmax > z_max).mean()),
            "median_span": float(np.median(span)),
            "median_zmax": float(np.median(zmax)),
        })

    def _d_at(level, key):
        prev = None
        for row in rows["curve"]:
            p, d = row[key], row["tilt_peak_to_peak"]
            if p >= level:
                if prev is None or prev[1] == p:
                    return float(d)
                d0, p0 = prev
                return float(d0 + (level - p0) * (d - d0) / (p - p0))
            prev = (d, p)
        return None

    rows["span_arm_d50"] = _d_at(0.5, "p_span_arm_fires")
    rows["span_arm_d90"] = _d_at(0.9, "p_span_arm_fires")
    rows["z_arm_d50"] = _d_at(0.5, "p_z_arm_fires")
    rows["z_arm_d90"] = _d_at(0.9, "p_z_arm_fires")
    rows["false_alarm_span_arm"] = rows["curve"][0]["p_span_arm_fires"] \
        if rows["curve"] and rows["curve"][0]["tilt_peak_to_peak"] == 0.0 \
        else None
    rows["false_alarm_z_arm"] = rows["curve"][0]["p_z_arm_fires"] \
        if rows["curve"] and rows["curve"][0]["tilt_peak_to_peak"] == 0.0 \
        else None
    rows["note"] = (
        "d is the PEAK-TO-PEAK fractional z-tilt of the DATA relative to the "
        "forward model. The d=0 row is the false-alarm rate. Same omissions "
        "as ratio_span_null (see its docstring), so power is an UPPER bound. "
        "The span arm's threshold is the DECLINED ratio_span_by_z_max, which "
        "does NOT gate; the z arm's is z_zbin_max, which DOES gate although "
        "nobody ratified it (see ratification.py).")
    return rows


def ratio_span_null_report(*, n_draws=20000, seed=1, pack=None):
    """The committed routine behind ``ratio_span_null_calibration.json``.

    Emits the null distribution of the two UNRATIFIED span statistics plus the
    MEASURED FALSE-ALARM RATE of the two thresholds the PI declined to ratify
    (decision 8, 2026-07-29).  Provenance-stamped, because the spec quotes its
    numbers and this project's rule is committed-routine-plus-git-stamp, never
    a scratch JSON.

    SYNTHETIC pack only.  No survey data, no survey-derived value.
    """
    from CDDF_analysis.hbi_mcmc.pack import synthetic_pack
    from CDDF_analysis.hbi_mcmc.run_posterior import GATE
    from CDDF_analysis.hbi_mcmc import ratification as _RAT

    if pack is None:
        pack = synthetic_pack(0, **RATIO_SPAN_NULL_GRID, fp_frac=0.15,
                              t_true=np.array([0.2, -0.15]))
    res = selftest(pack, resp_clamp="both")
    nul = ratio_span_null(pack, n_draws=n_draws, seed=seed, res=res)

    # the false-alarm rate of each PROPOSED threshold under the same null
    mu = np.asarray(res["mu"], float)
    dxpos = np.asarray(pack.dX, float) > 0
    mu_m = np.where(np.broadcast_to(dxpos[None, :, :], mu.shape), mu, 0.0)
    rng = np.random.default_rng(seed)
    obs_star = rng.poisson(np.broadcast_to(mu_m, (n_draws,) + mu_m.shape))
    for arm, ax, key in (("by_z", (0, 2), "ratio_span_by_z_max"),
                         ("by_snr", (0, 1), "ratio_span_by_snr_max")):
        mu_r = mu_m.sum(axis=ax)
        o = obs_star.sum(axis=tuple(a + 1 for a in ax))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(o > 0, mu_r[None, :] / o, np.nan)
        fin = np.isfinite(r)
        hi = np.max(np.where(fin, r, -np.inf), axis=1)
        lo = np.min(np.where(fin, r, np.inf), axis=1)
        span = np.where(fin.sum(axis=1) >= 2, hi - lo, 0.0)
        thr = float(GATE[key])
        nul["arms"][arm]["proposed_threshold"] = thr
        nul["arms"][arm]["proposed_threshold_name"] = key
        nul["arms"][arm]["measured_false_alarm_rate"] = float((span > thr).mean())
        nul["arms"][arm]["ratification_status"] = _RAT.record(key)["status"]

    # ------------------------------------------------------------------
    # 🔴 THE SAME NULL ON PRODUCTION-SCALE GEOMETRIES.
    # A range statistic's null width depends on the ROW COUNT and the per-row
    # exposure (spec §1.1 item 1), so a false-alarm rate measured on a 4-row
    # by_z arm says nothing about the 15-row production arm.  The v1 artifact
    # quoted 0.3434 with no geometry attached and the spec then said "a third
    # of perfectly correct forward models" unqualified.  Measured here.
    # ------------------------------------------------------------------
    geoms, power = {}, {}
    for gname, grid in sorted(RATIO_SPAN_NULL_GEOMETRIES.items()):
        n_kk = int(len(np.asarray(grid["zc_edges"], float)) - 1)
        gpack = synthetic_pack(0, **grid, fp_frac=0.15,
                               t_true=np.full(n_kk, 0.0))
        gres = selftest(gpack, resp_clamp="both")
        gnul = ratio_span_null(gpack, n_draws=n_draws, seed=seed, res=gres)
        gmu = np.asarray(gres["mu"], float)
        gdx = np.asarray(gpack.dX, float) > 0
        gmu_m = np.where(np.broadcast_to(gdx[None, :, :], gmu.shape), gmu, 0.0)
        grng = np.random.default_rng(seed)
        gobs = grng.poisson(np.broadcast_to(gmu_m, (n_draws,) + gmu_m.shape))
        for arm, key in (("by_z", "ratio_span_by_z_max"),
                         ("by_snr", "ratio_span_by_snr_max")):
            mu_r, obs_r = _arm_rows_from_obs(gmu_m, gobs, arm)
            span, _ = _span_and_zmax(mu_r, obs_r)
            thr = float(GATE[key])
            gnul["arms"][arm]["proposed_threshold"] = thr
            gnul["arms"][arm]["proposed_threshold_name"] = key
            gnul["arms"][arm]["measured_false_alarm_rate"] = float(
                (span > thr).mean())
            gnul["arms"][arm]["ratification_status"] = _RAT.record(key)["status"]
        geoms[gname] = {
            "grid_shape": {"n_nhat": int(gpack.n_c), "n_zf": int(gpack.n_k),
                           "n_snr": int(gpack.n_s)},
            "total_mu": float(gmu_m.sum()),
            "arms": {a: {k: v for k, v in gnul["arms"][a].items()
                         if k != "span_draws_summary"}
                     for a in ("by_z", "by_snr")},
        }
        # the power curve (spec §3 step 6) -- both guards, same injected tilt.
        # TWICE: at the DECLINED 0.10, and at a CALIBRATED threshold read off
        # THIS geometry's own null at the spec's Bonferroni alpha = 0.005
        # (q99.5), which is the only span threshold the spec would let anyone
        # propose.  Without the second one there is no actionable option A.
        n_pw = min(n_draws, 4000)
        power[gname] = ratio_span_power(gpack, n_draws=n_pw, seed=seed,
                                        res=gres)
        thr_cal = float(gnul["arms"]["by_z"]["quantiles"]["0.995"])
        pw_cal = ratio_span_power(gpack, n_draws=n_pw, seed=seed, res=gres,
                                  span_max=thr_cal)
        power[gname]["calibrated"] = {
            "span_threshold": thr_cal,
            "threshold_origin": ("this geometry's own null q99.5 = the spec's "
                                 "Bonferroni alpha=0.005 per arm (§3 step 3/4)"),
            "false_alarm_span_arm": pw_cal["false_alarm_span_arm"],
            "span_arm_d50": pw_cal["span_arm_d50"],
            "span_arm_d90": pw_cal["span_arm_d90"],
            "curve": pw_cal["curve"],
            "note": ("NOT PROPOSED FOR RATIFICATION -- a synthetic pack, and "
                     "it inherits every omission in the null (spec §2.1). It "
                     "exists so option A is a measured option rather than a "
                     "suggestion."),
        }

    far = {g: geoms[g]["arms"]["by_z"]["measured_false_alarm_rate"]
           for g in geoms}

    return {
        "schema": "ratio_span_null_calibration/v2",
        "null": nul,
        "pack": {"kind": "synthetic_pack", "seed": 0, "fp_frac": 0.15,
                 "t_true": [0.2, -0.15],
                 "grid": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in RATIO_SPAN_NULL_GRID.items()},
                 "grid_shape": {"n_nhat": int(pack.n_c), "n_zf": int(pack.n_k),
                                "n_snr": int(pack.n_s)},
                 "total_mu": float(mu_m.sum()),
                 "total_obs": float(np.where(
                     np.broadcast_to(dxpos[None, :, :], mu.shape),
                     res["counts"], 0).sum())},
        "geometries": geoms,
        "power": power,
        "geometry_correction": (
            "🔴 CORRECTION to schema v1 and to the two commit messages of "
            "dbda6a4 / 88f2ecb: the measured false-alarm rate of "
            "ratio_span_by_z_max = 0.10 was quoted as 0.3434 ('refuses 34% of "
            "perfectly correct forward models') with NO GEOMETRY ATTACHED. "
            "That number is specific to the 5x4x2 calibration pack, whose "
            "by_z arm has FOUR rows. On production-scale geometries the SAME "
            "measurement gives by_z FAR = "
            + ", ".join(f"{g}: {far[g]:.4f}" for g in sorted(far))
            + ". Every quote of a span false-alarm rate must name its grid. "
              "The spec's own §1.1 item 1 predicted this and the v1 "
              "calibration ignored it."),
        "verdict": (
            "NO THRESHOLD IS PROPOSED FOR RATIFICATION. This artifact is the "
            "prospective-calibration EVIDENCE that the two declined "
            "thresholds are indefensible as a pair: on EVERY geometry "
            "measured here their false-alarm rates under a null in which the "
            "forward model is exactly right differ by orders of magnitude, in "
            "the opposite direction to the stated rationale (by_snr is inert "
            "at 0.0000 on both production geometries while by_z fires at "
            "~0.08). The pair-mismatch conclusion SURVIVES at production "
            "scale; the MAGNITUDE quoted for by_z does not -- see "
            "geometry_correction. Whether DISARMING the span arms leaves the "
            "z-marginal tilt defect unguarded is a PI TRADEOFF, not resolved "
            "by this routine: the measured detection curves for the disarmed "
            "span arm and the still-armed z_zbin_max arm are in `power`, and "
            "the decision is recorded in "
            "ratification.pi_decision('span_arms_disarmed') -- OPEN when this "
            "routine was written, ANSWERED by the PI direction of 2026-08-05 "
            "(\"keep span-by-z and span-by-SNR active as advisory "
            "diagnostics, not ratified hard gates\"), which changes no arm "
            "and no number here and closes the option of arming a calibrated "
            "threshold. Artifacts generated before that date point at "
            "ratification.OPEN_PI_DECISIONS['span_arms_disarmed'] and are "
            "correct as dated evidence. Procedure, omissions and options: "
            "docs/ratio_span_calibration_spec.md."),
        "metadata": {
            "routine": "CDDF_analysis/hbi_mcmc/forward_selftest.py",
            "entry_point": "ratio_span_null_report / --ratio-span-null",
            "date": time.strftime("%Y-%m-%d"),
            "code_commit": _git(),
            "code_dirty": bool(_git_dirty()),
            "code_dirty_scope": _DIRTY_SCOPE,
            "scope": "SYNTHETIC ONLY. No real-survey data or values.",
            "paper_facing": False,
            "ratification": _RAT.ratification_stamp(),
            "rederive": (
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "python -m CDDF_analysis.hbi_mcmc.forward_selftest "
                f"--ratio-span-null --null-draws {int(n_draws)} "
                f"--null-seed {int(seed)} --out CDDF_analysis/hbi_mcmc/"
                "ratio_span_null_calibration.json"),
        },
    }


def aggregate_report(mocks, *, clamps=("off", "both", "hi"), use_fp=True):
    """The MULTI-MOCK, MULTI-CLAMP report -- the committed routine behind
    ``rung9_forward_selftest.json``.

    ``mocks`` is a list of ``(name, pack_path)``.  This existed only as an
    uncommitted scratch driver, which is how the artifact came to carry a
    hand-written 7-char ``code_commit`` naming a commit at which this file did
    not exist (the ORPHANED provenance class).  It is committed now so the
    artifact has a `rederive` line that actually runs.

    MOCKS ONLY.
    """
    from CDDF_analysis.hbi_mcmc.pack import load_pack

    out_mocks, closes, pads = {}, {}, {}
    for name, path in mocks:
        assert "main_dark" not in path, "REAL-LOA guard: mock packs only"
        pack = load_pack(path)
        # This tool's purpose includes auditing HISTORICAL packs, so the
        # schema-v1.2 legacy migration is applied here EXPLICITLY and logged
        # ((1-eta) restoration 2026-08-06); build_consts stays fail-loud.
        if pack.fp_eta_c is None:
            from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands
            print(f"[selftest] {name}: legacy pack (pre-2026-08-06) — "
                  "attaching fp_eta_c from the committed band table "
                  "(pack.FP_ETA_BANDS_COMMITTED)", file=sys.stderr)
            pack = attach_fp_eta_bands(pack)
        assert "loa_main_dark" not in json.dumps(pack.provenance or {}), \
            "REAL-LOA guard (provenance)"
        if pack.truth_counts is None:
            raise SystemExit(f"{name}: pack carries no truth_counts")
        entry = {"pack": os.path.basename(path),
                 "n_pad_bins": int(pack.n_pad_bins),
                 "probes": structural_probes(pack)}
        for clamp in clamps:
            tab = ratio_tables(selftest(pack, use_fp=use_fp, resp_clamp=clamp),
                               pack)
            entry[f"clamp_{clamp}"] = tab
            if clamp == "both":
                closes[name] = _closure_verdict(tab, 5.0, 5.0, 3.0)
        pads[name] = int(pack.n_pad_bins)
        out_mocks[name] = entry

    return {
        **_stamp_fields(mocks),
        "what": ("pure forward-model truth-fold self-test: the pack's own "
                 "truth f(N,z) folded through the pack's own kernel/"
                 "completeness/g/dX/FP machinery at the truth-equivalent "
                 "parameter point, vs the pack's own observed counts. NO "
                 "SAMPLING."),
        "mocks": out_mocks,
        "closure_verdicts": closes,
        "n_pad_bins": pads,
        "verdict": {
            "D1_basis_pad_low_N": (
                "OPEN — needs a re-extracted basis-padded pack"
                if any(v == 0 for v in pads.values())
                else "a basis-padded pack is in use"),
            "D2_response_extrapolation_high_N":
                "FIXED in-code (resp_clamp, default 'both')",
            "forward_model_closes": bool(
                closes and all(v["closes"] for v in closes.values())),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack")
    ap.add_argument("--mock", action="append", metavar="NAME=PATH",
                    help="AGGREGATE mode: repeat once per mock. Emits the "
                         "multi-mock / multi-clamp report "
                         "(rung9_forward_selftest.json). Mutually exclusive "
                         "with --pack.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--truth-floor", type=float, default=None,
                    help="extend the TRUE-N grid down to this log N_HI using a "
                         "power law fitted to the pack's own truth")
    ap.add_argument("--fit-lo", type=float, default=19.8)
    ap.add_argument("--fit-hi", type=float, default=20.8)
    ap.add_argument("--no-fp", action="store_true")
    ap.add_argument("--resp-clamp", default=None, choices=["both", "hi", "off"],
                    help="response covariate-range guard (finding D2); default "
                         "= 'both' when the pack carries resp_N_fit_range, "
                         "'off' otherwise (pre-fix reproduction)")
    ap.add_argument("--require-closure", action="store_true",
                    help="EXIT NONZERO unless the truth-fold closes within the "
                         "tolerances below. Without this the command is a REPORT "
                         "and always exits 0 -- so `selftest ... || exit 1` in a "
                         "batch script is NOT a gate. Any script that spends "
                         "sampler time must pass this.")
    ap.add_argument("--require-basis-pad", action="store_true",
                    help="EXIT NONZERO unless the pack's true-N basis is padded "
                         "BELOW the reporting floor (n_pad_bins > 0). Finding "
                         "D1: an unpadded pack cannot arithmetically reproduce "
                         "its own lowest observed bins, so no batch script that "
                         "spends sampler time may run on one.")
    ap.add_argument("--min-pad-bins", type=int, default=1)
    ap.add_argument("--max-abs-z-total", type=float, default=5.0)
    ap.add_argument("--max-abs-z-bin", type=float, default=5.0,
                    help="max |z| allowed in EVERY per-bin marginal -- by_nhat, "
                         "by_z AND by_snr. One knob for three arms because "
                         "run_posterior.GATE sets z_bin_max == z_zbin_max == "
                         "z_snrbin_max == 5.0. RESTATED_NOT_RATIFIED: these "
                         "arms refuse work and nobody ratified them.")
    ap.add_argument("--max-chi2-dof", type=float, default=3.0,
                    help="chi2/dof over the reported n-hat bins with obs > 0. "
                         "The ONE ratified numerical closure tolerance "
                         "(PI decision 8): 3.0.")
    ap.add_argument("--ratio-span-null", action="store_true",
                    help="prospective calibration of the two UNRATIFIED "
                         "ratio-span tolerances on a SYNTHETIC pack: null "
                         "distribution + measured false-alarm rate. Emits "
                         "ratio_span_null_calibration.json. See "
                         "docs/ratio_span_calibration_spec.md.")
    ap.add_argument("--null-draws", type=int, default=20000)
    ap.add_argument("--null-seed", type=int, default=1)
    a = ap.parse_args(argv)

    if a.ratio_span_null:
        rep = ratio_span_null_report(n_draws=a.null_draws, seed=a.null_seed)
        for arm, e in rep["null"]["arms"].items():
            print(f"[null] {arm}: rows={e['n_rows']} "
                  f"q50={e['quantiles']['0.5']:.4f} "
                  f"q95={e['quantiles']['0.95']:.4f} "
                  f"q99={e['quantiles']['0.99']:.4f} | proposed "
                  f"{e['proposed_threshold_name']}={e['proposed_threshold']} "
                  f"({e['ratification_status']}) -> measured false-alarm rate "
                  f"{e['measured_false_alarm_rate']:.4f}")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(rep, fh, indent=1)
            print(f"[null] wrote {a.out}")
        return rep

    if a.mock:
        if a.pack:
            raise SystemExit("--mock and --pack are mutually exclusive")
        pairs = []
        for spec in a.mock:
            if "=" not in spec:
                raise SystemExit(f"--mock expects NAME=PATH, got {spec!r}")
            n, _, p = spec.partition("=")
            pairs.append((n, p))
        rep = aggregate_report(pairs)
        for n, e in rep["mocks"].items():
            print_tables(e["clamp_both"], f"{n} (resp_clamp=both, "
                                          f"n_pad_bins={e['n_pad_bins']})")
        print(f"\n[selftest] verdict: {json.dumps(rep['verdict'])}")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(rep, fh, indent=1)
            print(f"[selftest] wrote {a.out}")
        if a.require_closure and not rep["verdict"]["forward_model_closes"]:
            print("\n[selftest] FORWARD MODEL DOES NOT CLOSE -- refusing.",
                  file=sys.stderr)
            raise SystemExit(3)
        return rep

    if not a.pack:
        raise SystemExit("--pack is required (or use --mock NAME=PATH)")

    from CDDF_analysis.hbi_mcmc.pack import load_pack
    assert "main_dark" not in a.pack, "REAL-LOA guard: mock packs only"
    pack = load_pack(a.pack)
    # Same explicit logged legacy migration as aggregate_report above: the
    # CLI/preflight audits HISTORICAL packs ((1-eta) restoration 2026-08-06).
    if pack.fp_eta_c is None:
        from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands
        print(f"[selftest] {os.path.basename(a.pack)}: legacy pack "
              "(pre-2026-08-06) — attaching fp_eta_c from the committed band "
              "table (pack.FP_ETA_BANDS_COMMITTED)", file=sys.stderr)
        pack = attach_fp_eta_bands(pack)
    assert "loa_main_dark" not in json.dumps(pack.provenance or {}), \
        "REAL-LOA guard (provenance)"
    if pack.truth_counts is None:
        raise SystemExit("pack carries no truth_counts — self-test needs a mock")

    # --- the BASIS-PAD gate (finding D1), before any other work -----------
    n_pad = int(getattr(pack, "n_pad_bins", 0))
    if a.require_basis_pad and n_pad < a.min_pad_bins:
        raise SystemExit(
            f"[selftest] REFUSING: pack has n_pad_bins={n_pad} "
            f"(< {a.min_pad_bins}). The true-N basis stops at the reporting "
            f"floor {float(np.asarray(pack.ntrue_edges, float)[0]):.2f}, so "
            f"the truth cannot feed the lowest observed bins and the fold "
            f"cannot close at ANY parameter value (finding D1). Re-extract a "
            f"basis-padded pack (schema v1.1 permits a DOWNWARD pad).")

    t0 = time.time()
    probes = structural_probes(pack)
    clamp = a.resp_clamp or ("both" if pack.resp_N_fit_range is not None else "off")
    res = selftest(pack, use_fp=not a.no_fp, resp_clamp=clamp)
    tab = ratio_tables(res, pack)
    print_tables(tab, f"baseline truth-fold (truth floor {probes['ntrue_lo']:.1f}, "
                      f"resp_clamp={clamp})")

    out = dict(pack=os.path.basename(a.pack), probes=probes, baseline=tab,
               resp_clamp=clamp, n_pad_bins=n_pad,
               provenance=dict(routine="CDDF_analysis/hbi_mcmc/forward_selftest.py",
                               code_commit=_git(),
                               code_dirty=bool(_git_dirty()),
                               code_dirty_scope=_DIRTY_SCOPE,
                               date=time.strftime("%Y-%m-%d"),
                               rederive=("python -m CDDF_analysis.hbi_mcmc."
                                         f"forward_selftest --pack {a.pack}")))

    if a.truth_floor is not None:
        pack2, f2 = extend_pack_truth(pack, a.truth_floor, a.fit_lo, a.fit_hi)
        res2 = selftest(pack2, f=f2, use_fp=not a.no_fp, resp_clamp=clamp)
        tab2 = ratio_tables(res2, pack2)
        print_tables(tab2, f"truth-fold, true-N extended to {a.truth_floor:.1f}")
        out["extended"] = tab2
        out["extended_floor"] = a.truth_floor

    print(f"\n[selftest] {time.time() - t0:.1f}s")

    # --- the actual gate -------------------------------------------------
    # main() is a REPORT by default: it prints the ratio table and returns.  That
    # made `forward_selftest --pack $PACK || exit 1` in rung9v3_2lpt0.sbatch a
    # no-op -- it exited 0 on the very pack this module's own commit message
    # declares broken (total mu/obs 0.7312, chi2/dof 2216), so the "fail-closed
    # pre-flight" would have let ~36 h of sampler time run on a forward model
    # known not to close.  --require-closure makes it a gate for real.
    verdict = _closure_verdict(tab, a.max_abs_z_total, a.max_abs_z_bin,
                               a.max_chi2_dof)
    out["closure_verdict"] = verdict
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[selftest] wrote {a.out}")
    if a.require_closure and not verdict["closes"]:
        print("\n[selftest] FORWARD MODEL DOES NOT CLOSE -- refusing.\n  "
              + "\n  ".join(verdict["reasons"]), file=sys.stderr)
        raise SystemExit(3)
    return out


def _closure_verdict(tab, max_abs_z_total, max_abs_z_bin, max_chi2_dof,
                     max_abs_z_snrbin=None):
    """PASS/FAIL on the truth-fold, from the same table the report prints.

    THIS IS WHAT PRODUCTION RUNS AS ITS PRE-FLIGHT.  ``--require-closure`` is
    invoked by ``slurm/greatlakes/hbi_mcmc/posterior_production.sbatch`` and by
    ``rung9v3_2lpt0.sbatch``, which calls it "LOAD-BEARING"; both take this
    verdict, not ``run_posterior.forward_closure_gate``, as the thing standing
    between a broken forward model and ~36 h of sampler time.

    ALL THREE marginal arms are read -- 'by_nhat', 'by_z' AND 'by_snr'.
    ``ratio_tables`` computes all three; until 2026-08-05 this verdict looped
    over only the first two while ``forward_closure_gate`` gated all three, so
    the PRE-FLIGHT WAS THE WEAKER OF THE TWO CHECKS.  Demonstrated on a
    constructed table: by_snr max|z| = 30.0 against GATE['z_snrbin_max'] = 5.0
    returned {'closes': True, 'reasons': []}.  A forward model that closes in
    total, in N and in z while mis-predicting one SNR stratum 4:1 is exactly
    the completeness/response defect the SNR marginal exists to catch.

    ``max_abs_z_snrbin`` defaults to ``max_abs_z_bin``, matching
    ``run_posterior.GATE`` where z_bin_max == z_zbin_max == z_snrbin_max == 5.0.
    It is a separate parameter so a future split of those tolerances does not
    silently re-couple the arms.

    🔴 AUTHORITY.  The four |z| arms are RESTATED_NOT_RATIFIED: they refuse
    work and no deciding authority ratified them (``ratification.py``).  Only
    chi2/dof <= 3 is a ratified number.  Adding the by_snr arm ARMS one more
    unratified tolerance in the pre-flight; it does so to make the pre-flight
    equal the committed gate, not because the number acquired authority.

    RESIDUAL, KNOWN ASYMMETRY vs ``forward_closure_gate``: that function
    restricts each marginal to rows with ``obs > 0``; this one does not, so on a
    table with an EMPTY aggregate row carrying mu > 0 (|z| = sqrt(mu)) this
    verdict is STRICTER.  Left stricter deliberately -- relaxing a fail-closed
    pre-flight is not a refactor.  Measured inert on all 18 committed mock
    packs: the only obs == 0 rows are the two structurally-empty SNR<=2 op-mask
    strata, which carry mu == 0 and therefore z == 0.0 exactly.
    """
    tot = tab.get("total", {})
    if max_abs_z_snrbin is None:
        max_abs_z_snrbin = max_abs_z_bin
    reasons = []
    zt = abs(float(tot.get("z", 0.0)))
    if zt > max_abs_z_total:
        reasons.append(f"|z_total| {zt:.2f} > {max_abs_z_total}")
    # chi2/dof is COMPUTED here from the n-hat rows rather than read from
    # ``tot``.  🔴 CORRECTION (2026-08-05): until this date this comment stated,
    # in the present tense, that ``ratio_tables``'s ``total`` carries only
    # mu/obs/ratio/z and has at no point emitted a ``chi2_dof`` key.  That was
    # true when written and is FALSE now -- ``ratio_tables`` (see its own
    # comment above ``out["total"]["chi2_dof"]``) emits both ``chi2_dof`` and
    # ``n_gate_bins``, and on ``synthetic_pack(0, **small_test_grid())`` the key
    # reads 0.6479548471525799 over 10 bins, identical to the value computed
    # here.  The retracted sentence is not reproduced verbatim so that a grep
    # for it cannot land on this correction -- read it in git instead.
    # What survives is the RULE, not the claim about the key: this arm computes
    # its own number so it cannot depend on a producer remembering to supply
    # one.  The original defect was real -- ``tot.get("chi2_dof", 0.0)`` read
    # 0.0 unconditionally, so a table of many mildly-off bins, each
    # individually under the per-bin |z| limit, "closed".
    _rows = [r for r in (tab.get("by_nhat") or [])
             if isinstance(r, dict) and r.get("obs", 0) > 0
             and r.get("z") is not None and np.isfinite(float(r["z"]))]
    _z = np.array([float(r["z"]) for r in _rows], float)
    c2 = float((_z ** 2).sum() / len(_z)) if len(_z) else float("nan")
    if np.isfinite(c2) and c2 > max_chi2_dof:
        reasons.append(f"chi2/dof {c2:.2f} > {max_chi2_dof} "
                       f"over {len(_z)} n-hat bins")
    arm_tol = {"by_nhat": max_abs_z_bin, "by_z": max_abs_z_bin,
               "by_snr": max_abs_z_snrbin}
    for key in ("by_nhat", "by_z", "by_snr"):
        tol = arm_tol[key]
        rows = tab.get(key) or {}
        zs = [abs(float(r.get("z", 0.0)))
              for r in (rows.values() if isinstance(rows, dict) else rows)
              if isinstance(r, dict) and r.get("z") is not None]
        if zs and max(zs) > tol:
            reasons.append(f"max|z| in {key} = {max(zs):.2f} > {tol}")
    return dict(closes=not reasons, reasons=reasons,
                chi2_dof=c2, n_bins=int(len(_z)),
                arms_gated=["total", "chi2_dof", "by_nhat", "by_z", "by_snr"],
                tolerances=dict(max_abs_z_total=max_abs_z_total,
                                max_abs_z_bin=max_abs_z_bin,
                                max_abs_z_snrbin=max_abs_z_snrbin,
                                max_chi2_dof=max_chi2_dof))


if __name__ == "__main__":
    main()
