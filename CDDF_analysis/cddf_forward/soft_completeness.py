"""Diagonal soft-completeness Bayesian core (M2 / O3, contract §2).

PURE ARRAYS.  No file I/O, no GP files, no ``calc_cddf`` import.  This module is
the inference layer for the O3 *diagonal soft completeness* correction.  For each
(log N_HI, z) bin ``b`` we correct the raw probabilistic expected count ``F_b``
(the O1 number) by a per-bin scalar completeness ``C_b`` and an additive soft
false-positive deposit ``b_FP_b``::

    n_corr_b = (F_b - b_FP_b) / C_b

"Diagonal" means each bin is corrected by its OWN ``C`` and ``b_FP``; there is NO
cross-bin migration (sub-DLA <-> DLA across logN=20.3, redshift scatter,
LLS <-> DLA leakage).  Those off-diagonal channels are O4 (M3/M4) and are
explicitly OUT OF SCOPE here.  The diagonal correction therefore cannot capture
mass that scatters between bins; treat it as the inference-faithful diagonal
limit of an α(z)-style completeness calibration resolved in (N, z) bins.

"Soft" means both the recovered counts and the truth<->recovered association are
posterior-sample weighted (fractional), NOT catalog-thresholded.  Consequently
the matched / unmatched / truth inputs are real-valued, and the Beta / Gamma
posteriors below are GENERALIZED to fractional successes (see each docstring).

Modeling summary
----------------
* Completeness ``C_b`` : binomial detection rate with a conjugate Beta posterior,
  generalized to fractional successes — ``C_b ~ Beta(f_matched_b + a,
  n_truth_b - f_matched_b + b)``.  For integer counts this is the exact
  Beta-Binomial posterior; for fractional (posterior-weighted) ``f_matched`` it is
  the natural soft-count generalization (the Beta density is well defined for any
  positive shape parameters, and its mean ``(f+a)/(n+a+b)`` is exactly the
  posterior-weighted success fraction shrunk toward the prior).  Approximation:
  this treats the soft success count as if it were a Beta-Binomial sufficient
  statistic; it ignores the (sub-Poisson) variance reduction from averaging over
  posterior samples, so the resulting interval is mildly CONSERVATIVE (wider than
  a fully-pooled treatment).  Stated honestly because it is the modeling choice,
  not a derived fact.

* FP deposit ``b_FP_b`` : Poisson-rate of spurious deposits with a conjugate
  Gamma posterior on fractional counts — ``rate_b ~ Gamma(f_unmatched_b + a,
  exposure_b)`` (shape, rate parameterization), and the DEPOSIT
  ``b_FP_b = rate_b * exposure_b`` is therefore distributed as
  ``Gamma(f_unmatched_b + a, 1)`` — a COUNT in the SAME units as ``F_b`` so that
  ``F_b - b_FP_b`` is dimensionally valid.  ``exposure`` is carried (and used to
  define the rate posterior) but the deposit itself is exposure-invariant by
  construction; this is the units contract demanded by §2.2 / §3.4.

* Propagation : ``apply_diagonal_correction`` combines the THREE uncertainty
  sources (F count CI, C posterior, b_FP posterior) by VECTORIZED ANCESTRAL
  SAMPLING from the already-closed-form posteriors — draw ``C`` from its Beta,
  ``b_FP`` from its Gamma, ``F`` from a count-space proxy matched to ``F_ci``,
  push each draw through ``(F - b_FP)/C``, and take percentiles.  This is direct
  sampling from closed-form posteriors, NOT an MCMC / NUTS sampler over a model —
  it adds no sampler layer.

SBC scope
---------
The ``toy_count_mock`` + ``sbc_coverage`` harness validates THIS INFERENCE LAYER
assuming ``C`` and ``b_FP`` are known (simulate recovered ``F`` from known
``n_true``, ``C``, ``b_FP``; run the estimator; check interval coverage and rank
uniformity).  It does NOT validate the response-matrix build — that requires the
later injection campaign + Campaign D.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import beta as _beta_dist
from scipy.stats import gamma as _gamma_dist

# Default number of ancestral-sampling draws for propagation (contract §2.3).
_DEFAULT_N_MC: int = 4000
# Deterministic base seed so propagation / SBC are reproducible.
_BASE_SEED: int = 20260610


# --------------------------------------------------------------------------- #
# 2.1  estimate_diagonal_completeness
# --------------------------------------------------------------------------- #
def estimate_diagonal_completeness(
    f_matched: np.ndarray,
    n_truth: np.ndarray,
    *,
    prior: Tuple[float, float] = (0.5, 0.5),
) -> Dict[str, np.ndarray]:
    """Per-bin completeness ``C`` with a fractional-success Beta posterior.

    Parameters
    ----------
    f_matched : ndarray[nbin]
        Expected recovered count deposited in bin ``b`` by sightlines that truly
        host an absorber in bin ``b`` (the completeness numerator).  FRACTIONAL
        (posterior-weighted): ``0 <= f_matched_b``; in practice
        ``f_matched_b <= n_truth_b`` modulo scatter, but NOT assumed integer.
    n_truth : ndarray[nbin]
        Number of truth absorbers in bin ``b`` (``>= 0``; may be fractional but
        usually integer).
    prior : (a, b), default (0.5, 0.5)
        Beta prior shape parameters (Jeffreys by default).

    Returns
    -------
    dict
        ``C, C_lo68, C_hi68, C_lo95, C_hi95, valid_mask, upscatter_mask,
        C_alpha, C_beta``.  ``C_alpha`` / ``C_beta`` are the EFFECTIVE Beta
        posterior shape parameters for each valid bin (NaN on invalid bins), so a
        downstream sampler can draw from the EXACT posterior instead of
        moment-matching from the mean and half-width (which drifts in the tails of
        skewed high-completeness bins).  For up-scatter bins the shapes are kept
        from the CLAMPED Beta (the point is still pinned to 1).

    Model
    -----
    ``C_b ~ Beta(f_eff_b + a, n_truth_b - f_eff_b + b)`` where ``f_eff_b`` is
    ``f_matched_b`` clamped to ``[0, n_truth_b]`` (see module docstring for the
    fractional-success justification and its conservative approximation).  The
    reported point estimate ``C`` is the Beta posterior MEAN
    ``(f_eff + a) / (n_truth + a + b)``; intervals are equal-tailed Beta
    quantiles.  ``C`` is clipped to ``(0, 1]``.

    Edge cases (contract §2.1)
    --------------------------
    * ``n_truth_b == 0`` : ``C`` undefined -> NaN, CI NaN, ``valid_mask=False``
      (NEVER 0 -> division blowup; NEVER silently 1).
    * ``f_matched_b > n_truth_b`` (up-scatter) : ``C`` clipped to 1,
      ``upscatter_mask_b=True`` (the Beta is formed on the clamped count, never
      yielding ``C > 1``).
    * ``f_matched_b == 0, n_truth_b > 0`` : ``C -> 0+`` (incomplete), lower CI
      near 0.

    DIAGONAL LIMITATION: this completeness is per-bin only and cannot capture
    truth mass that scatters into a DIFFERENT (N, z) bin (that is O4 migration).
    """
    f_matched = np.asarray(f_matched, dtype=float)
    n_truth = np.asarray(n_truth, dtype=float)
    if f_matched.shape != n_truth.shape:
        raise ValueError(
            f"f_matched {f_matched.shape} and n_truth {n_truth.shape} must match"
        )
    a, b = float(prior[0]), float(prior[1])
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta prior shapes must be > 0, got {prior!r}")

    nbin = f_matched.shape[0] if f_matched.ndim else 1
    f_matched = np.atleast_1d(f_matched)
    n_truth = np.atleast_1d(n_truth)

    valid_mask = n_truth > 0.0
    upscatter_mask = (f_matched > n_truth) & valid_mask

    # Clamp the soft success count into [0, n_truth] so the Beta shapes stay
    # positive and C never exceeds 1 (up-scatter is recorded, not propagated).
    f_eff = np.clip(f_matched, 0.0, n_truth)

    alpha = f_eff + a
    beta_param = n_truth - f_eff + b

    C = np.full(nbin, np.nan)
    C_lo68 = np.full(nbin, np.nan)
    C_hi68 = np.full(nbin, np.nan)
    C_lo95 = np.full(nbin, np.nan)
    C_hi95 = np.full(nbin, np.nan)
    # Effective Beta shapes (NaN on invalid bins) so apply_diagonal_correction can
    # resample the EXACT posterior rather than moment-matching (B3).
    C_alpha = np.full(nbin, np.nan)
    C_beta = np.full(nbin, np.nan)

    if np.any(valid_mask):
        al = alpha[valid_mask]
        be = beta_param[valid_mask]
        mean = al / (al + be)
        C[valid_mask] = mean
        C_alpha[valid_mask] = al
        C_beta[valid_mask] = be
        C_lo68[valid_mask] = _beta_dist.ppf(0.16, al, be)
        C_hi68[valid_mask] = _beta_dist.ppf(0.84, al, be)
        C_lo95[valid_mask] = _beta_dist.ppf(0.025, al, be)
        C_hi95[valid_mask] = _beta_dist.ppf(0.975, al, be)

    # Up-scatter (f_matched > n_truth): the completeness point is clipped to
    # exactly 1 with the flag recording it (contract §2.1).  The CI is kept from
    # the clamped Beta (the conservative interval), but the upper edges cannot
    # exceed 1 and the point sits at the ceiling.
    if np.any(upscatter_mask):
        C[upscatter_mask] = 1.0

    # Clip to the valid completeness support (0, 1].  Strictly-positive lower
    # clamp avoids a later division blowup; NaNs (invalid bins) pass through.
    tiny = np.finfo(float).tiny
    for arr in (C, C_lo68, C_hi68, C_lo95, C_hi95):
        np.clip(arr, tiny, 1.0, out=arr, where=np.isfinite(arr))

    return {
        "C": C,
        "C_lo68": C_lo68,
        "C_hi68": C_hi68,
        "C_lo95": C_lo95,
        "C_hi95": C_hi95,
        "valid_mask": valid_mask,
        "upscatter_mask": upscatter_mask,
        "C_alpha": C_alpha,
        "C_beta": C_beta,
    }


# --------------------------------------------------------------------------- #
# 2.2  estimate_false_positive_deposit
# --------------------------------------------------------------------------- #
def estimate_false_positive_deposit(
    f_unmatched: np.ndarray,
    exposure,
    *,
    prior: Tuple[float] = (0.5,),
) -> Dict[str, np.ndarray]:
    """Per-bin soft false-positive DEPOSIT ``b_FP`` with a Gamma posterior.

    Parameters
    ----------
    f_unmatched : ndarray[nbin]
        Expected recovered count deposited in bin ``b`` by sightlines with NO
        truth absorber in bin ``b`` (the soft FP numerator).  FRACTIONAL.
    exposure : ndarray[nbin] | float
        Normalization that turns a count into a rate consistent with how ``F_b``
        is normalized downstream (e.g. number of contributing sightlines, or the
        total path length ΔX in the bin).  The caller supplies the SAME exposure
        basis it uses for ``F``.
    prior : (a,), default (0.5,)
        Gamma prior shape offset (Jeffreys-like).

    Returns
    -------
    dict
        ``b_FP, b_FP_lo68, b_FP_hi68, b_FP_lo95, b_FP_hi95``.

    Units contract (explicit)
    -------------------------
    The DEPOSIT ``b_FP_b`` is a COUNT in the SAME units as ``F_b`` (NOT a rate),
    so that ``F_b - b_FP_b`` is dimensionally valid.  It is defined as
    ``b_FP_b = rate_b * exposure_b`` with the rate posterior
    ``rate_b ~ Gamma(shape = f_unmatched_b + a, rate = exposure_b)``.  Multiplying
    a ``Gamma(shape, rate=exposure)`` draw by ``exposure`` gives a
    ``Gamma(shape, rate=1)`` draw, so the deposit posterior is
    ``b_FP_b ~ Gamma(f_unmatched_b + a, 1)`` — exposure-invariant by construction.
    ``b_FP >= 0`` always.

    Point estimate — POSTERIOR MODE, not mean (B4)
    ----------------------------------------------
    The reported POINT ``b_FP`` is the Gamma posterior MODE
    ``max(f_unmatched_b + a - 1, 0)`` (for ``Gamma(shape, scale=1)`` the mode is
    ``shape - 1`` for ``shape >= 1`` and ``0`` otherwise).  This is the value that
    is SUBTRACTED in ``(F - b_FP)/C``.  With the Jeffreys-like ``a = 0.5`` prior a
    CLEAN bin (``f_unmatched = 0``) therefore has ``b_FP point == 0`` EXACTLY — no
    phantom half-count is subtracted from clean high-N bins (the Gamma MEAN
    ``f_unmatched + a`` would subtract the prior offset ``a = 0.5`` even with zero
    evidence, biasing clean bins downward; the mode does not).  The ``a = 0.5``
    deposit prior remains a conservative FP floor for the CI (a fully proper but
    diffuse posterior), so the INTERVAL still reflects honest uncertainty about a
    non-zero FP rate even where the point is 0; we simply do not assert a
    half-count of false positives as a central value with no supporting evidence.
    Intervals are equal-tailed ``Gamma(f_unmatched + a, scale=1)`` quantiles.

    Model
    -----
    Poisson-rate of spurious deposits with a conjugate Gamma posterior on
    FRACTIONAL counts (the soft generalization of the Poisson-Gamma posterior;
    the Gamma density is well defined for any positive shape).
    """
    f_unmatched = np.asarray(f_unmatched, dtype=float)
    f_unmatched = np.atleast_1d(f_unmatched)
    nbin = f_unmatched.shape[0]
    a = float(prior[0])
    if a <= 0:
        raise ValueError(f"Gamma prior shape offset must be > 0, got {prior!r}")
    if np.any(f_unmatched < 0):
        raise ValueError("f_unmatched must be non-negative")

    exposure_arr = np.broadcast_to(np.asarray(exposure, dtype=float), (nbin,))
    if np.any(exposure_arr <= 0):
        raise ValueError("exposure must be strictly positive")

    shape = f_unmatched + a  # Gamma shape

    # Deposit posterior is Gamma(shape, rate=1) == scale=1 (exposure cancels).
    # POINT estimate is the posterior MODE max(shape - 1, 0) (B4): this is the
    # value SUBTRACTED in (F - b_FP)/C, and it -> 0 as f_unmatched -> 0, so a clean
    # bin is not over-subtracted by the prior offset a.  The CI below still uses
    # the full Gamma(shape, 1) posterior.
    b_FP = np.maximum(shape - 1.0, 0.0)  # Gamma(shape, scale=1) mode
    b_FP_lo68 = _gamma_dist.ppf(0.16, shape, scale=1.0)
    b_FP_hi68 = _gamma_dist.ppf(0.84, shape, scale=1.0)
    b_FP_lo95 = _gamma_dist.ppf(0.025, shape, scale=1.0)
    b_FP_hi95 = _gamma_dist.ppf(0.975, shape, scale=1.0)

    # Clamp the interval to BRACKET the mode point (B4).  For shape >= 1 the mode
    # is interior and this is (near) a no-op; for shape < 1 (clean / sparse bins)
    # the Gamma mode is at 0 and the density is monotonically decreasing, so the
    # honest highest-posterior-density interval is ``[0, q]`` — i.e. the lower edge
    # IS the mode (0).  Clamping the equal-tailed lower edges down to the mode and
    # the upper edges up to the mode makes the reported interval consistent with
    # the reported (mode) point in every regime.
    b_FP = np.maximum(b_FP, 0.0)
    b_FP_lo68 = np.minimum(np.maximum(b_FP_lo68, 0.0), b_FP)
    b_FP_lo95 = np.minimum(np.maximum(b_FP_lo95, 0.0), b_FP)
    b_FP_hi68 = np.maximum(b_FP_hi68, b_FP)
    b_FP_hi95 = np.maximum(b_FP_hi95, b_FP)

    return {
        "b_FP": b_FP,
        "b_FP_lo68": b_FP_lo68,
        "b_FP_hi68": b_FP_hi68,
        "b_FP_lo95": b_FP_lo95,
        "b_FP_hi95": b_FP_hi95,
        # Exact Gamma(shape, scale=1) deposit-posterior shape per bin, so the
        # downstream sampler draws the EXACT posterior (the reported point is the
        # MODE, not the shape, so it cannot be used to recover the shape).
        "b_FP_shape": shape,
    }


# --------------------------------------------------------------------------- #
# Internal: count-space proxy for F's uncertainty
# --------------------------------------------------------------------------- #
def _sample_F_from_ci(
    F: np.ndarray, F_ci: Dict[str, np.ndarray], n_mc: int, rng: np.random.Generator
) -> np.ndarray:
    """Ancestral draws of the count ``F`` matched to its Poisson-binomial CI.

    ``F`` is the expected count and ``F_ci`` carries the estimator's
    Poisson-binomial 68/95 COUNT intervals.  We do NOT re-derive that
    distribution; we approximate it by a Gamma matched in mean (``F``) and spread
    (half the 68% interval width as a 1σ proxy).  A Gamma is the natural
    non-negative count proxy and recovers Poisson behaviour for unit dispersion.

    Returns an ``(n_mc, nbin)`` array of non-negative draws.  Degenerate bins
    (zero width, e.g. F==0 with a collapsed CI) return the point value.
    """
    F = np.asarray(F, dtype=float)
    nbin = F.shape[0]
    lo68 = np.asarray(F_ci["lo68"], dtype=float)
    hi68 = np.asarray(F_ci["hi68"], dtype=float)
    sigma = np.maximum((hi68 - lo68) / 2.0, 0.0)

    draws = np.empty((n_mc, nbin))
    for j in range(nbin):
        mu = F[j]
        s = sigma[j]
        if s <= 0 or mu <= 0:
            # No usable spread -> treat F as fixed at its point value.
            draws[:, j] = max(mu, 0.0)
            continue
        # Moment-match a Gamma: shape k = (mu/s)^2, scale = s^2/mu.
        k = (mu / s) ** 2
        scale = s * s / mu
        draws[:, j] = _gamma_dist.rvs(k, scale=scale, size=n_mc, random_state=rng)
    return draws


# --------------------------------------------------------------------------- #
# 2.3  apply_diagonal_correction
# --------------------------------------------------------------------------- #
def apply_diagonal_correction(
    F: np.ndarray,
    F_ci: Dict[str, np.ndarray],
    C_est: Dict[str, np.ndarray],
    bfp_est: Dict[str, np.ndarray],
    *,
    n_mc: Optional[int] = None,
    return_draws: bool = False,
) -> Dict[str, np.ndarray]:
    """Apply ``n_corr = (F - b_FP) / C`` with three-source error propagation.

    Parameters
    ----------
    F : ndarray[nbin]
        Raw probabilistic expected counts (O1), COUNT space.
    F_ci : dict
        ``{lo68, hi68, lo95, hi95}`` — the estimator's Poisson-binomial COUNT
        intervals for ``F`` (handed through in count space, contract §3.4).
    C_est : dict
        Output of :func:`estimate_diagonal_completeness`.  If it carries the
        EXACT effective Beta shapes (``C_alpha`` / ``C_beta``, added by B3) we
        sample those directly; otherwise we fall back to moment-matching the Beta
        from the reported mean and 68% half-width.
    bfp_est : dict
        Output of :func:`estimate_false_positive_deposit`.  If it carries the
        EXACT Gamma shape (``b_FP_shape``, added by B4) we sample that directly;
        otherwise we fall back to using the reported point as the shape (only
        correct when the point is the Gamma MEAN).
    n_mc : int, optional
        Number of ancestral draws (default 4000).  Deterministic seed.
    return_draws : bool, optional
        If True, ALSO return the raw per-draw ``n_corr`` array under the key
        ``n_corr_draws`` (shape ``(n_mc, nbin)``; invalid-bin columns are NaN).
        The driver forms Ω(z) PER DRAW from this (see :func:`omega_from_draws`),
        preserving the inter-bin correlation — guaranteeing the Ω point lies
        inside its own interval (B2).

    Returns
    -------
    dict
        ``n_corr, lo68, hi68, lo95, hi95, neg_clip_mask, valid_mask`` (and
        ``n_corr_draws`` if ``return_draws``).

    Propagation
    -----------
    Combines THREE sources — the count CI of ``F``, the Beta posterior of ``C``,
    and the Gamma posterior of ``b_FP`` — by vectorized ANCESTRAL SAMPLING from
    those closed-form posteriors (NOT an MCMC sampler over a model).  For each of
    ``n_mc`` draws we sample ``F`` (count proxy matched to ``F_ci``), ``C`` (Beta,
    EXACT shapes when present), and ``b_FP`` (Gamma, EXACT shape when present),
    form ``(F - b_FP)/C``, clip negatives to 0, and take equal-tailed
    percentiles.  The point estimate is the plug-in ``(F - b_FP)/C`` at the
    reported C MEAN and b_FP MODE (B4: the mode -> 0 for a clean bin).

    Non-negativity (contract §2.3)
    ------------------------------
    Where ``F_b - b_FP_b < 0`` (over-subtracted FP in a starved bin) the point
    ``n_corr`` is clipped to 0 and ``neg_clip_mask_b=True``.  Per-draw negatives
    are likewise clipped before percentiles.  Invalid completeness bins
    (``valid_mask=False``) stay NaN throughout.
    """
    if n_mc is None:
        n_mc = _DEFAULT_N_MC
    n_mc = int(n_mc)

    F = np.atleast_1d(np.asarray(F, dtype=float))
    nbin = F.shape[0]

    C = np.asarray(C_est["C"], dtype=float)
    valid_mask = np.asarray(C_est["valid_mask"], dtype=bool)

    b_FP = np.asarray(bfp_est["b_FP"], dtype=float)

    # Point estimate: C mean, b_FP mode (B4).
    with np.errstate(invalid="ignore", divide="ignore"):
        n_corr = (F - b_FP) / C
    neg_clip_mask = (F - b_FP) < 0.0
    n_corr = np.where(neg_clip_mask, 0.0, n_corr)
    # Invalid completeness bins -> NaN.
    n_corr = np.where(valid_mask, n_corr, np.nan)

    rng = np.random.default_rng(_BASE_SEED)

    # --- ancestral draws of the three sources ---------------------------- #
    # F: count-space proxy matched to F_ci.
    F_draws = _sample_F_from_ci(F, F_ci, n_mc, rng)  # (n_mc, nbin)
    # C: EXACT Beta shapes (B3) when present, else moment-match fallback.
    C_draws = _sample_C_draws(C_est, n_mc, rng)
    # b_FP: EXACT Gamma shape (B4) when present, else reported-point fallback.
    bfp_draws = _sample_bfp_draws(bfp_est, n_mc, rng)

    # --- push through the ratio ----------------------------------------- #
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (F_draws - bfp_draws) / C_draws
    ratio = np.maximum(ratio, 0.0)  # per-draw non-negativity
    # Invalid-completeness columns are NaN (C_draws already NaN there).
    ratio[:, ~valid_mask] = np.nan

    lo68 = np.full(nbin, np.nan)
    hi68 = np.full(nbin, np.nan)
    lo95 = np.full(nbin, np.nan)
    hi95 = np.full(nbin, np.nan)
    for j in range(nbin):
        if not valid_mask[j]:
            continue
        col = ratio[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        lo68[j], hi68[j] = np.percentile(col, [16.0, 84.0])
        lo95[j], hi95[j] = np.percentile(col, [2.5, 97.5])

    out = {
        "n_corr": n_corr,
        "lo68": lo68,
        "hi68": hi68,
        "lo95": lo95,
        "hi95": hi95,
        "neg_clip_mask": neg_clip_mask & valid_mask,
        "valid_mask": valid_mask,
    }
    if return_draws:
        out["n_corr_draws"] = ratio
    return out


def _sample_C_draws(
    C_est: Dict[str, np.ndarray], n_mc: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_mc, nbin)`` Beta draws of ``C`` — EXACT shapes (B3) when available.

    If ``C_est`` carries ``C_alpha`` / ``C_beta`` (the effective Beta posterior
    shapes the estimator actually used) we draw ``Beta(C_alpha, C_beta)``
    DIRECTLY — reproducing the estimator's own quantiles in skewed bins without
    the tail drift of moment-matching from mean+half-width.  Otherwise we fall
    back to moment-matching the Beta from the reported mean and 68% half-width.
    Invalid bins are NaN; draws are clipped to ``(tiny, 1]``.
    """
    C = np.asarray(C_est["C"], dtype=float)
    valid = np.asarray(C_est["valid_mask"], dtype=bool)
    nbin = C.shape[0]
    C_draws = np.empty((n_mc, nbin))

    has_shapes = "C_alpha" in C_est and "C_beta" in C_est
    if has_shapes:
        C_alpha = np.asarray(C_est["C_alpha"], dtype=float)
        C_beta = np.asarray(C_est["C_beta"], dtype=float)
    else:
        C_lo68 = np.asarray(C_est["C_lo68"], dtype=float)
        C_hi68 = np.asarray(C_est["C_hi68"], dtype=float)

    for j in range(nbin):
        if not valid[j] or not np.isfinite(C[j]):
            C_draws[:, j] = np.nan
            continue
        if has_shapes and np.isfinite(C_alpha[j]) and np.isfinite(C_beta[j]):
            al = float(C_alpha[j])
            be = float(C_beta[j])
        else:
            # Moment-match fallback (legacy path).
            mean = float(C[j])
            half = max((float(C_hi68[j]) - float(C_lo68[j])) / 2.0, 1e-6)
            var = half * half  # 68% half-width ~ 1 sigma
            max_var = mean * (1.0 - mean)
            var = min(var, max(max_var * 0.999, 1e-9))
            nu = max(mean * (1.0 - mean) / var - 1.0, 1e-6)
            al = mean * nu
            be = (1.0 - mean) * nu
        C_draws[:, j] = _beta_dist.rvs(al, be, size=n_mc, random_state=rng)
    np.clip(C_draws, np.finfo(float).tiny, 1.0, out=C_draws, where=np.isfinite(C_draws))
    return C_draws


def _sample_bfp_draws(
    bfp_est: Dict[str, np.ndarray], n_mc: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_mc, nbin)`` Gamma draws of ``b_FP`` — EXACT shape (B4) when available.

    The reported ``b_FP`` point is the Gamma MODE (B4), so it can NOT be used to
    recover the Gamma shape.  When ``bfp_est`` carries ``b_FP_shape`` we draw
    ``Gamma(shape, scale=1)`` directly; otherwise we fall back to treating the
    reported point as the shape (only correct if the point is the Gamma MEAN).
    """
    b_FP = np.asarray(bfp_est["b_FP"], dtype=float)
    nbin = b_FP.shape[0]
    if "b_FP_shape" in bfp_est:
        shapes = np.asarray(bfp_est["b_FP_shape"], dtype=float)
    else:
        shapes = b_FP
    bfp_draws = np.empty((n_mc, nbin))
    for j in range(nbin):
        shape = max(float(shapes[j]), 1e-12)
        bfp_draws[:, j] = _gamma_dist.rvs(shape, scale=1.0, size=n_mc, random_state=rng)
    return bfp_draws


# Physical constants for Omega_HI (cgs); identical to the driver's inline values
# so omega_from_draws is a drop-in for the driver WITHOUT importing calc_cddf
# (contract §2: the core must not import calc_cddf).
_PROTON_MASS_G = 1.67262178e-24
_H100_PER_S = 3.2407789e-18  # 100 km/s/Mpc in 1/s
_LIGHT_CM_S = 2.99e10
_GRAV_CGS = 6.674e-8


def _omega_conversion(hubble: float) -> float:
    """Ω_HI conversion factor ``m_p H0 / (c rho_c)`` in cgs (pure formula).

    Mirrors ``calc_cddf.rho_crit`` + the driver's inline conversion so this core
    helper stays self-contained (no ``calc_cddf`` import, contract §2).
    ``rho_c = 3 H0^2 / (8 pi G)`` with ``H0 = _H100_PER_S * hubble``.
    """
    h100 = _H100_PER_S * float(hubble)
    rho_c = 3.0 * h100 ** 2 / (8.0 * np.pi * _GRAV_CGS)
    return _PROTON_MASS_G / _LIGHT_CM_S * h100 / rho_c


def omega_from_draws(
    n_corr_draws: np.ndarray,
    logN_centres: np.ndarray,
    dX: float,
    hubble: float = 0.7,
) -> Dict[str, float]:
    """Ω_HI and its CI formed PER JOINT DRAW (B2; preserves inter-bin correlation).

    The diagonal Ω is ``Ω = (m_p H0 / c rho_c) * Σ_b N_HI_b * n_corr_b / ΔX``.
    Summing PRE-REDUCED per-bin CI edges (the old driver path) destroys the
    inter-bin correlation and can put the Ω POINT outside its own interval.  Here
    we instead form Ω for EACH ancestral draw ``d`` —
    ``Ω_d = conv * Σ_b N_HI_b * n_corr_draws[d, b] / ΔX`` (summing over bins that
    are finite in that draw) — and percentile the resulting Ω distribution.  The
    point Ω is computed from the per-draw MEAN of ``n_corr`` (== the per-draw Ω
    mean), so it lies inside the percentiled interval by construction.

    Parameters
    ----------
    n_corr_draws : ndarray ``(n_mc, nbin)``
        Per-draw corrected counts from
        :func:`apply_diagonal_correction(..., return_draws=True)`.  Invalid-bin
        columns are NaN and are dropped from each draw's sum.
    logN_centres : ndarray ``(nbin,)``
        log10(N_HI) bin centres.
    dX : float
        Absorption path length ΔX over the (single) z window.
    hubble : float, optional
        Dimensionless Hubble ``h`` (default 0.7).

    Returns
    -------
    dict
        ``omega, lo68, hi68, lo95, hi95`` (all scalars).
    """
    draws = np.asarray(n_corr_draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError(f"n_corr_draws must be (n_mc, nbin), got {draws.shape}")
    nhi_cent = 10.0 ** np.asarray(logN_centres, dtype=float)
    if nhi_cent.shape[0] != draws.shape[1]:
        raise ValueError(
            f"logN_centres length {nhi_cent.shape[0]} != nbin {draws.shape[1]}"
        )
    conversion = _omega_conversion(hubble)

    # N-weighted per-draw deposit, NaN bins -> 0 so they drop out of the sum.
    weighted = nhi_cent[None, :] * np.nan_to_num(draws, nan=0.0)
    omega_draws = conversion * np.sum(weighted, axis=1) / float(dX)

    # Point Ω from the per-draw MEAN of n_corr (equivalently the mean of the Ω
    # draws), guaranteeing it lies within the percentiled interval.
    omega_point = float(np.mean(omega_draws))
    lo68, hi68 = np.percentile(omega_draws, [16.0, 84.0])
    lo95, hi95 = np.percentile(omega_draws, [2.5, 97.5])
    return {
        "omega": omega_point,
        "lo68": float(lo68),
        "hi68": float(hi68),
        "lo95": float(lo95),
        "hi95": float(hi95),
    }


# --------------------------------------------------------------------------- #
# 2.4  toy_count_mock + SBC harness
# --------------------------------------------------------------------------- #
def toy_count_mock(
    n_true: np.ndarray,
    C: np.ndarray,
    b_FP: np.ndarray,
    exposure,
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Simulate a recovered count ``F`` from KNOWN ``n_true``, ``C``, ``b_FP``.

    Generative model (the data-generating process the inference layer assumes):

    * ``f_matched_b ~ Binomial(n_true_b, C_b)`` — true absorbers recovered;
    * ``f_unmatched_b ~ Poisson(b_FP_b)`` — spurious deposits with no truth;
    * ``F_b = f_matched_b + f_unmatched_b`` — the observed recovered count;
    * ``n_truth_b = n_true_b`` — the truth count is observed exactly (mock truth).

    Returns ``{F, f_matched, f_unmatched, n_truth, n_true}`` (all length-nbin).
    Used by :func:`sbc_coverage`.  This validates the INFERENCE layer with
    ``C``/``b_FP`` KNOWN; it does NOT test the response-matrix build.
    """
    rng = np.random.default_rng(int(seed))
    n_true = np.atleast_1d(np.asarray(n_true, dtype=float))
    C = np.atleast_1d(np.asarray(C, dtype=float))
    b_FP = np.atleast_1d(np.asarray(b_FP, dtype=float))
    nbin = n_true.shape[0]
    np.broadcast_to(np.asarray(exposure, dtype=float), (nbin,))  # validate shape

    n_int = np.rint(n_true).astype(np.int64)
    f_matched = rng.binomial(n_int, np.clip(C, 0.0, 1.0)).astype(float)
    f_unmatched = rng.poisson(np.maximum(b_FP, 0.0)).astype(float)
    F = f_matched + f_unmatched
    return {
        "F": F,
        "f_matched": f_matched,
        "f_unmatched": f_unmatched,
        "n_truth": n_int.astype(float),
        "n_true": n_true,
    }


def _count_ci_from_point(
    F: np.ndarray,
    f_matched: Optional[np.ndarray] = None,
    f_unmatched: Optional[np.ndarray] = None,
    n_truth: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """A faithful count CI proxy for ``F`` (used by the SBC harness).

    The SBC simulator (:func:`toy_count_mock`) produces ``F = f_matched +
    f_unmatched`` where ``f_matched ~ Binomial(n_truth, C)`` and
    ``f_unmatched ~ Poisson(b_FP)``.  The faithful count variance is therefore the
    BINOMIAL variance of the matched part PLUS the POISSON variance of the FP
    part — NOT a naive ``sqrt(F)`` (which would over-state the spread, since
    binomial thinning is sub-Poisson).  This mirrors what the production estimator
    hands through in §3.4: its Poisson-binomial CI is exactly
    ``Var(F) = sum_i p_i(1-p_i)``, the per-sightline Bernoulli (binomial) variance
    — sub-Poisson by construction.

    If the component counts are not supplied, fall back to the Poisson ``sqrt(F)``
    proxy (used only where the decomposition is unavailable).
    """
    F = np.asarray(F, dtype=float)
    if f_matched is not None and n_truth is not None:
        fm = np.asarray(f_matched, dtype=float)
        nt = np.asarray(n_truth, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            phat = np.where(nt > 0, fm / nt, 0.0)
        var_match = fm * (1.0 - np.clip(phat, 0.0, 1.0))  # n*p*(1-p) = fm*(1-phat)
        var_fp = np.asarray(f_unmatched, dtype=float) if f_unmatched is not None else 0.0
        s = np.sqrt(np.maximum(var_match + var_fp, 0.0))
    else:
        s = np.sqrt(np.maximum(F, 0.0))
    return {
        "lo68": np.maximum(F - s, 0.0),
        "hi68": F + s,
        "lo95": np.maximum(F - 2.0 * s, 0.0),
        "hi95": F + 2.0 * s,
    }


def _mirror_one_mock(
    n_true: np.ndarray,
    C_true: np.ndarray,
    b_true: np.ndarray,
    *,
    build_frac: float,
    soft_fractional: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Draw ONE whole-sample mock and split it 70/30 into BUILD / whole (B1).

    Mirrors the PRODUCTION estimator at the SIGHTLINE level: each truth absorber
    lives on a sightline that lands in BUILD with probability ``build_frac``; its
    recovery (the matched deposit) inherits the SAME role, so a truth absorber and
    its match always go to BUILD together.  This coherent split is what makes the
    BUILD completeness ``Ĉ = f_matched_build / n_truth_build`` an honest binomial
    estimate (numerator and denominator coupled per sightline) — independent
    thinning of numerator and denominator would spuriously inflate ``Var(Ĉ)``.
    The whole sample is the UNION (BUILD ⊂ whole), exactly as production applies a
    BUILD-measured ``C`` to the whole-sample ``F``.

    Returns whole-sample ``F``, ``f_matched_whole``, ``f_unmatched_whole``,
    ``n_true``; and the BUILD-subset ``f_matched_build``, ``f_unmatched_build``,
    ``n_truth_build``.

    ``soft_fractional`` : if True, the matched recovery is FRACTIONAL
    (posterior-weighted): each truth absorber contributes a recovered WEIGHT
    ``w_i = clip(Bernoulli(C_b) + N(0, sigma_soft), 0, 1)`` rather than a hard 0/1
    detection.  This mirrors a real GP posterior recovery probability — bimodal
    (most weight near 0 or 1) but genuinely fractional — so ``f_matched`` and
    ``n_truth`` are non-integer (the "soft" regime), while the matched-mass
    variance stays CLOSE to binomial (a tight Beta around ``C`` would not — it
    would be far sub-binomial and the binomial-style completeness posterior would
    then over-cover).  The per-absorber weight and its sightline role are coherent.
    """
    nbin = n_true.shape[0]
    n_int = np.rint(n_true).astype(np.int64)
    C_clip = np.clip(C_true, 0.0, 1.0)
    b_pos = np.maximum(b_true, 0.0)

    f_matched_whole = np.zeros(nbin)
    f_matched_build = np.zeros(nbin)
    f_matched_held = np.zeros(nbin)
    n_truth_build = np.zeros(nbin)
    n_truth_held = np.zeros(nbin)

    for j in range(nbin):
        k = int(n_int[j])
        if k <= 0:
            continue
        # Per-absorber sightline role: BUILD with prob build_frac (coherent).
        role_build = rng.random(k) < build_frac
        role_held = ~role_build
        if soft_fractional:
            # Soft truth: each truth absorber carries a near-1 fractional
            # truth-confidence weight (high-confidence Beta), so n_truth is also
            # genuinely fractional (posterior-weighted), per the contract.
            t_w = rng.beta(40.0, 1.0, size=k)  # mean ~0.976, fractional
        else:
            t_w = np.ones(k)
        n_truth_build[j] = float(np.sum(t_w[role_build]))
        n_truth_held[j] = float(np.sum(t_w[role_held]))
        if soft_fractional:
            # Soft regime: bimodal posterior recovery weight per absorber —
            # a hard detection softened by Gaussian fuzz, clipped to [0, 1], then
            # capped by the truth weight (can't recover more than the truth mass).
            # Genuinely fractional, but matched-mass variance stays ~binomial so
            # the binomial-style completeness posterior remains well-calibrated.
            sigma_soft = 0.12
            hard = (rng.random(k) < C_clip[j]).astype(float)
            det = np.clip(hard + rng.normal(0.0, sigma_soft, size=k), 0.0, 1.0)
            w = det * t_w  # recovered mass <= truth weight
        else:
            # Hard 0/1 detection per absorber.
            w = (rng.random(k) < C_clip[j]).astype(float)
        f_matched_whole[j] = float(np.sum(w))
        f_matched_build[j] = float(np.sum(w[role_build]))
        f_matched_held[j] = float(np.sum(w[role_held]))

    # FP deposits are independent of truth, so their split is plain binomial
    # thinning (each spurious deposit lands in BUILD with prob build_frac).
    f_unmatched_whole = rng.poisson(b_pos).astype(float)
    f_unmatched_build = rng.binomial(
        f_unmatched_whole.astype(np.int64), build_frac
    ).astype(float)
    f_unmatched_held = f_unmatched_whole - f_unmatched_build

    F = f_matched_whole + f_unmatched_whole
    # HELDOUT-only recovered count (disjoint from BUILD).
    F_held = f_matched_held + f_unmatched_held
    # n_true split coherently so the HELDOUT target is the HELDOUT truth count.
    n_true_held = n_truth_held.copy()

    return {
        "F": F,
        "f_matched_whole": f_matched_whole,
        "f_unmatched_whole": f_unmatched_whole,
        "n_true": n_true,
        "f_matched_build": f_matched_build,
        "f_unmatched_build": f_unmatched_build,
        "n_truth_build": n_truth_build,
        "F_held": F_held,
        "f_matched_held": f_matched_held,
        "f_unmatched_held": f_unmatched_held,
        "n_truth_held": n_truth_held,
        "n_true_held": n_true_held,
    }


def sbc_coverage(
    *,
    n_sims: int = 400,
    nbin: int = 6,
    seed: int = 2026,
    n_true_range: Tuple[float, float] = (200.0, 2000.0),
    C_range: Tuple[float, float] = (0.5, 0.95),
    bfp_range: Tuple[float, float] = (0.0, 5.0),
    n_mc: int = 2000,
    return_ranks: bool = False,
    corrupt_completeness: float = 0.0,
    production_mirror: bool = False,
    build_frac: float = 0.7,
    rebase_bfp: bool = True,
    soft_fractional: bool = False,
) -> Dict[str, object]:
    """Simulation-based calibration of the diagonal-correction inference layer.

    Two modes:

    * ``production_mirror=False`` (legacy): draw TWO independent EQUAL-SCALE mocks
      of the same ``n_true`` — a CALIBRATION mock for ``C`` / ``b_FP`` and an
      independent MEASUREMENT mock for ``F``.  This validates the inference-layer
      independence assumption but — because both mocks are at the SAME scale — it
      CANNOT surface a basis/rebasing bug in the FP deposit (the missing rebasing
      cancels).  Kept for backward compatibility.

    * ``production_mirror=True`` (B1, the FAITHFUL harness): draw ONE mock, split
      it ``build_frac``/``1-build_frac``, estimate ``C`` and the FP deposit on the
      BUILD subset, and apply the correction to the DISJOINT HELDOUT recovered
      ``F`` — mirroring ``driver.heldout_closure`` (the validation path that DOES
      a BUILD→HELDOUT basis rebasing).  Keeping the BUILD/HELDOUT sets disjoint is
      what makes the ancestral-sampling independence assumption faithful, and it is
      the rebasing logic that this harness certifies.  (The SCIENCE path,
      ``compute_o3_products``, instead estimates ``C``/``b_FP``/``F`` all on ONE
      whole-active set with NO split and NO rebasing — trivially consistent, so the
      thing worth SBC-testing is the rebasing in the closure path.)  The FP deposit
      is a per-bin COUNT on the BUILD basis; to subtract it from the HELDOUT ``F``
      it MUST be rebased by the exposure ratio ``N_held / N_build``
      (== ``(1 - build_frac) / build_frac`` for a random sightline split).  With
      ``rebase_bfp=True`` the harness is calibrated; with ``rebase_bfp=False`` (the
      un-rebased BUG) the FP subtraction is mis-scaled and coverage DEGRADES — which
      the old equal-scale two-mock SBC could not reveal.  This is the
      basis-falsifiability guarantee.  (NOTE: the bug is only made VISIBLE at an
      inflated FP scale; at a realistic small-FP operating point the mis-scaling is
      numerically tiny — see the test, which inflates ``bfp_range`` to demonstrate
      sensitivity.)

    In BOTH modes we record per valid bin/sim:

    * whether the 68% / 95% corrected interval covers ``n_true`` (pooled coverage);
    * the SBC rank of ``n_true`` within the posterior draws (should be ~Uniform).

    Parameters
    ----------
    production_mirror : bool, default False
        Use the one-mock 70/30 BUILD/whole harness mirroring production (B1).
    build_frac : float, default 0.7
        BUILD fraction of sightlines (production default 0.7).
    rebase_bfp : bool, default True
        Rebase the BUILD-basis FP deposit to the HELDOUT basis by
        ``(1-build_frac)/build_frac`` before subtracting from the HELDOUT ``F``
        (mirroring ``heldout_closure``).  Set False to reproduce the un-rebased
        bug (coverage must then break at a sufficiently large FP scale).
    soft_fractional : bool, default False
        Exercise the SOFT regime: fractional (posterior-weighted) ``f_matched`` /
        recovered mass rather than integer detections.  Coverage must still hold.
    corrupt_completeness : float, default 0.0
        FALSIFIABILITY knob.  If > 0, deliberately inflate the BUILD ``f_matched``
        toward ``n_truth`` before estimating ``C`` (mis-specifying completeness).
        Coverage MUST then degrade — guards against a vacuous SBC pass.

    Returns
    -------
    dict
        ``coverage68, coverage95`` (pooled over valid bins/sims); plus ``ranks``
        (flat array in [0, 1]) if ``return_ranks``.

    SCOPE: validates the inference layer assuming ``C`` / ``b_FP`` known.  Does
    NOT validate the response-matrix build (injection campaign + Campaign D).
    """
    master = np.random.default_rng(int(seed))
    cov68_hits = 0
    cov95_hits = 0
    total = 0
    ranks_all = []

    for s in range(int(n_sims)):
        n_true = master.uniform(n_true_range[0], n_true_range[1], size=nbin)
        C_true = master.uniform(C_range[0], C_range[1], size=nbin)
        b_true = master.uniform(bfp_range[0], bfp_range[1], size=nbin)

        if production_mirror:
            # ---- ONE mock, coherent BUILD/HELDOUT split (B1) --------------- #
            # Estimate C and the FP deposit on BUILD; apply the correction to the
            # DISJOINT HELDOUT recovered F (independent of the build, so the
            # ancestral-sampling independence assumption is faithful), with the FP
            # deposit REBASED from the BUILD basis to the HELDOUT basis by the
            # exposure ratio N_held / N_build == (1 - build_frac) / build_frac.
            # This mirrors production's 70/30 build/measure separation AND the
            # count-basis rebasing; omitting the rebasing (rebase_bfp=False) is the
            # exact production-path bug and must break coverage.
            mock_rng = np.random.default_rng(int(master.integers(1, 2**31 - 1)))
            mock = _mirror_one_mock(
                n_true, C_true, b_true,
                build_frac=build_frac, soft_fractional=soft_fractional, rng=mock_rng,
            )
            f_matched = mock["f_matched_build"].copy()
            f_unmatched_build = mock["f_unmatched_build"]
            n_truth = mock["n_truth_build"]
            F = mock["F_held"]
            n_target = mock["n_true_held"]
            F_ci = _count_ci_from_point(
                F, mock["f_matched_held"], mock["f_unmatched_held"], n_target
            )
            if corrupt_completeness > 0.0:
                f_matched = f_matched + corrupt_completeness * (n_truth - f_matched)

            C_est = estimate_diagonal_completeness(f_matched, n_truth)
            # FP deposit measured on BUILD, REBASED to the HELDOUT basis
            # (count * N_held/N_build == count * (1-frac)/frac).  rebase_bfp=False
            # keeps the BUILD-basis count -> the production rebasing bug.
            held_over_build = (1.0 - float(build_frac)) / float(build_frac)
            f_unmatched_rebased = (
                f_unmatched_build * held_over_build if rebase_bfp else f_unmatched_build
            )
            bfp_est = estimate_false_positive_deposit(
                f_unmatched_rebased, float(nbin)
            )
        else:
            # ---- legacy: TWO independent equal-scale mocks ----------------- #
            cal_seed = int(master.integers(1, 2**31 - 1))
            meas_seed = int(master.integers(1, 2**31 - 1))
            cal = toy_count_mock(n_true, C_true, b_true, exposure=float(nbin), seed=cal_seed)
            f_matched = cal["f_matched"].copy()
            f_unmatched = cal["f_unmatched"]
            n_truth = cal["n_truth"]
            if corrupt_completeness > 0.0:
                f_matched = f_matched + corrupt_completeness * (n_truth - f_matched)
            C_est = estimate_diagonal_completeness(f_matched, n_truth)
            bfp_est = estimate_false_positive_deposit(f_unmatched, float(nbin))
            meas = toy_count_mock(n_true, C_true, b_true, exposure=float(nbin), seed=meas_seed)
            F = meas["F"]
            n_target = n_true
            F_ci = _count_ci_from_point(
                F, meas["f_matched"], meas["f_unmatched"], meas["n_truth"]
            )

        out = apply_diagonal_correction(F, F_ci, C_est, bfp_est, n_mc=n_mc)

        valid = out["valid_mask"]
        # Coverage on the closed-form intervals (target = n_target: the truth count
        # on the SAMPLE the correction is applied to — whole [legacy] or HELDOUT
        # [mirror]).
        for j in range(nbin):
            if not valid[j] or not np.isfinite(out["lo68"][j]):
                continue
            nt = n_target[j]
            total += 1
            if out["lo68"][j] <= nt <= out["hi68"][j]:
                cov68_hits += 1
            if out["lo95"][j] <= nt <= out["hi95"][j]:
                cov95_hits += 1

        if return_ranks:
            # Rank of n_target within freshly-drawn posterior samples (one cheap
            # ancestral resample per sim, independent of the percentile pass).
            rng = np.random.default_rng(int(master.integers(1, 2**31 - 1)))
            draws = _posterior_draws(F, F_ci, C_est, bfp_est, n_mc, rng)
            for j in range(nbin):
                if not valid[j]:
                    continue
                col = draws[:, j]
                col = col[np.isfinite(col)]
                if col.size == 0:
                    continue
                rank = float(np.mean(col < n_target[j]))
                ranks_all.append(rank)

    coverage68 = cov68_hits / total if total else np.nan
    coverage95 = cov95_hits / total if total else np.nan
    res: Dict[str, object] = {
        "coverage68": coverage68,
        "coverage95": coverage95,
        "n_eval_bins": total,
    }
    if return_ranks:
        res["ranks"] = np.asarray(ranks_all, dtype=float)
    return res


def _posterior_draws(
    F: np.ndarray,
    F_ci: Dict[str, np.ndarray],
    C_est: Dict[str, np.ndarray],
    bfp_est: Dict[str, np.ndarray],
    n_mc: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Raw ``(n_mc, nbin)`` ancestral draws of ``n_corr`` (for the SBC rank).

    Mirrors the sampling inside :func:`apply_diagonal_correction` but RETURNS the
    per-draw ratio array (no percentile reduction) so the SBC harness can compute
    the rank statistic of ``n_true`` within the posterior.
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    valid = np.asarray(C_est["valid_mask"], dtype=bool)

    # Reuse the SHARED ancestral samplers so the rank statistic uses the EXACT
    # Beta/Gamma posteriors (B3/B4) identically to apply_diagonal_correction.
    F_draws = _sample_F_from_ci(F, F_ci, n_mc, rng)
    C_draws = _sample_C_draws(C_est, n_mc, rng)
    bfp_draws = _sample_bfp_draws(bfp_est, n_mc, rng)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (F_draws - bfp_draws) / C_draws
    ratio = np.maximum(ratio, 0.0)
    ratio[:, ~valid] = np.nan
    return ratio
