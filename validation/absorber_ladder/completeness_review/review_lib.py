"""review_lib.py — pure figure-data helpers for the PI-facing C1 completeness
visual review (PI ruling 2026-09-13c §6).

VALIDATION-ONLY.  Nothing under ``CDDF_analysis/`` is modified; no sampler is
run; no real data is read.  Every routine here is plain numpy so the unit tests
(``tests/test_completeness_review.py``) run in seconds in either environment.

Conventions (inherited, not invented here)
------------------------------------------
* ``C`` is the DETECTION probability ``C_det = matched detections / truth
  systems`` with NO observed-grid restriction — the slot the fold's
  ``sigmoid(eta_hat + psi_c)[:, b_to_cell]`` occupies.
* the IN-GRID counting fraction ``phi(b, cell) = sum_c K[c<-b]`` lives with the
  response kernel: ``build_cc_tensors`` returns ``Mg`` with
  ``sum_c Mg[s,k,c,b] = phi_ref(b, cell) <= 1``.
* the truth-pinned forensic completeness ``C_true_bs`` / ``C_true_bKs``
  (``empirical_ops_*.npz``) is the OBSERVED-GRID-RESTRICTED rate
  ``C_det * phi_meas`` — a different object.  ``phi_measured`` below makes the
  difference explicit and ``fold_tp_2d`` lets the two be folded like-for-like.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "wilson_interval", "standardised_residual", "additive_logit_C",
    "phi_measured", "effective_completeness", "fold_tp_2d", "fold_tp_3d",
    "snr_marginal", "pooled_rate",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# binomial interval
# --------------------------------------------------------------------------
def wilson_interval(n_det, n_tot, z=1.0):
    """Wilson score interval for a binomial rate.

    Returns ``(lo, hi)`` arrays broadcast over the inputs.  ``z = 1`` gives the
    68 % (1 sigma) interval, ``z = 1.959964`` the 95 % one.  Cells with
    ``n_tot == 0`` return ``(nan, nan)``.

    The Wilson interval is used rather than the Wald interval precisely because
    the low-N, low-S/N cells this review is about have rates near 0 and 1,
    where the Wald interval leaves the unit interval and its width collapses to
    zero — which would hide exactly the cells under scrutiny.
    """
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    z = float(z)
    ok = t > 0
    tt = np.where(ok, t, 1.0)
    p = d / tt
    denom = 1.0 + z * z / tt
    centre = (p + z * z / (2.0 * tt)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / tt + z * z / (4.0 * tt * tt))
    lo = np.where(ok, np.clip(centre - half, 0.0, 1.0), np.nan)
    hi = np.where(ok, np.clip(centre + half, 0.0, 1.0), np.nan)
    return lo, hi


def standardised_residual(p_pred, n_det, n_tot, min_tot=1.0):
    """``(p_pred - n_det/n_tot) / sigma`` with ``sigma = sqrt(p(1-p)/n_tot)``.

    The predictive (not the observed) rate enters the variance, so a cell whose
    held-out rate happens to be exactly 0 or 1 still has a finite denominator.
    Cells with ``n_tot < min_tot`` return NaN.
    """
    p = np.clip(np.asarray(p_pred, float), _EPS, 1.0 - _EPS)
    d = np.asarray(n_det, float)
    t = np.asarray(n_tot, float)
    ok = t >= float(min_tot)
    tt = np.where(ok, t, 1.0)
    sd = np.sqrt(p * (1.0 - p) / tt)
    out = (p - d / tt) / np.maximum(sd, _EPS)
    return np.where(ok, out, np.nan)


# --------------------------------------------------------------------------
# the delivered C1nsadd functional form
# --------------------------------------------------------------------------
def additive_logit_C(x, u, beta, deg_n=3, deg_u=2):
    """``C = sigmoid(poly_deg_n(x) + poly_deg_u(u))`` — the C1nsadd surface.

    ``beta`` is the delivered coefficient vector in the order the design matrix
    ``cal_fit.design_2d_additive`` builds:
    ``[b0, b_x, b_x2, ..., b_x^deg_n, b_u, b_u2, ..., b_u^deg_u]``.
    ``x`` and ``u`` broadcast against each other, so passing ``x`` with shape
    ``(B, 1)`` and ``u`` with shape ``(1, S)`` returns the ``(B, S)`` surface.

    NOTE there is no interaction term: the N-shape is common to every stratum
    and log S/N only shifts the logit.  That is the whole content of the
    "additive" in C1nsadd, and it is why the curves in figure C2 are parallel
    in logit space.
    """
    beta = np.asarray(beta, float).ravel()
    dn, du = int(deg_n), int(deg_u)
    if beta.size != dn + du + 1:
        raise ValueError(f"additive_logit_C: expected {dn + du + 1} coefficients, got {beta.size}")
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    eta = np.zeros(np.broadcast(x, u).shape, float) + beta[0]
    for j in range(1, dn + 1):
        eta = eta + beta[j] * x ** j
    for i in range(1, du + 1):
        eta = eta + beta[dn + i] * u ** i
    # branch-stable logistic
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    ex = np.exp(eta[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# --------------------------------------------------------------------------
# the phi (in-grid counting fraction) bookkeeping
# --------------------------------------------------------------------------
def phi_measured(n_det_ingrid, n_det_all, axis=1, fill=1.0):
    """Measured in-grid counting fraction ``phi[b,s]``.

    ``n_det_ingrid`` = matched detections whose observed N-hat falls in the
    reported grid (``N_det_bks_true_z``); ``n_det_all`` = every matched
    detection (``N_det_all_bks_true_z``).  Both are ``(B, Kf, S)``; the sum is
    taken over ``axis`` (the fine-z axis) before the ratio, so the result is the
    truth-pooled fraction the z-pooled objects see.

    Cells with no detections get ``fill`` (1.0 = "nothing was lost"), which is
    the only choice that leaves ``C * phi == C`` where phi is unmeasurable.
    """
    a = np.asarray(n_det_ingrid, float).sum(axis=axis)
    b = np.asarray(n_det_all, float).sum(axis=axis)
    return np.where(b > 0, a / np.maximum(b, _EPS), float(fill))


def effective_completeness(C_sb, g_bk):
    """The fold's effective z-pooled completeness ``C[s,b] * g[b,k]``.

    Returns ``(B, Kf, S)``.  This is the object Addendum A of
    ``COMPLETENESS_VARIANTS_REPORT.md`` compares against ``C_true_bKs``; it is
    NOT a probability (``g`` is a per-N z-reshape and the product can exceed 1).
    """
    C = np.asarray(C_sb, float)
    g = np.asarray(g_bk, float)
    if C.shape[1] != g.shape[0]:
        raise ValueError("effective_completeness: C is (S,B) and g is (B,Kf)")
    return C.T[:, None, :] * g[:, :, None]


# --------------------------------------------------------------------------
# the fold, at fixed (truth) f
# --------------------------------------------------------------------------
def fold_tp_2d(Mg, C_sb, g_bk, truth_counts_bks, use_g=True):
    """True-positive counts of the runner's 2-D ``--c-fixed`` branch at truth f.

    ``mu[c,k,s] = sum_b Mg[s,k,c,b] C[s,b] g[b,k] truth_counts[b,k,s]``.

    This is the runner's own expression
    ``einsum("skcb,sb,bk->cks", Mg, C, g*f*dN) * dX`` with
    ``truth_counts[b,k,s] = f[b,k] dN_b dX[k,s]`` substituted, i.e. the fold
    evaluated at the mock's realised truth rather than at a posterior draw.
    """
    Mg = np.asarray(Mg, float)
    g = np.asarray(g_bk, float) if use_g else np.ones_like(np.asarray(g_bk, float))
    return np.einsum("skcb,sb,bk,bks->cks", Mg, np.asarray(C_sb, float), g,
                     np.asarray(truth_counts_bks, float), optimize=True)


def fold_tp_3d(Mg, C_bks, truth_counts_bks):
    """True-positive counts of the runner's 3-D ``--c-fixed`` branch at truth f.

    ``mu[c,k,s] = sum_b Mg[s,k,c,b] C[b,k,s] truth_counts[b,k,s]`` — note the
    3-D branch does NOT multiply by ``g`` (a z-resolved C replaces ``C*g``).
    """
    return np.einsum("skcb,bks,bks->cks", np.asarray(Mg, float),
                     np.asarray(C_bks, float),
                     np.asarray(truth_counts_bks, float), optimize=True)


def snr_marginal(mu_cks, obs_cks, normalise=True):
    """Per-S/N-stratum ``mu/obs``; if ``normalise`` divide by the pooled ratio.

    The sampler is free to rescale the latent f, so only the SHAPE of the S/N
    marginal is comparable between folds built at a fixed f.  ``normalise=True``
    removes the common level and returns exactly that shape; the pooled ratio is
    returned alongside so the level is never silently lost.
    """
    mu = np.asarray(mu_cks, float)
    ob = np.asarray(obs_cks, float)
    num = mu.sum(axis=(0, 1))
    den = ob.sum(axis=(0, 1))
    r = np.where(den > 0, num / np.maximum(den, _EPS), np.nan)
    total = mu.sum() / max(ob.sum(), _EPS)
    return (r / total if normalise else r), float(total)


def pooled_rate(n_det, n_tot, axis=None):
    """``sum(n_det)/sum(n_tot)`` over ``axis``, NaN where the denominator is 0."""
    d = np.asarray(n_det, float).sum(axis=axis)
    t = np.asarray(n_tot, float).sum(axis=axis)
    return np.where(t > 0, d / np.maximum(t, _EPS), np.nan)
