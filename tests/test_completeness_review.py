"""Unit + mutation tests for the C1 completeness visual-review figure data.

VALIDATION-ONLY.  Pure numpy; runs in either environment in well under a second.
Every numeric assertion carries an EXPLICIT tolerance (never ``np.allclose``'s
default atol, which is vacuous on small values — see RULES.md).

Mutation checks are written as explicit "the wrong formula gives a materially
different answer" assertions, so a silent reversion to the wrong formula fails
the suite rather than passing it.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation.absorber_ladder.completeness_review.review_lib import (
    wilson_interval, standardised_residual, additive_logit_C, phi_measured,
    effective_completeness, fold_tp_2d, fold_tp_3d, snr_marginal, pooled_rate)

ATOL = 1e-12


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------
def test_wilson_matches_closed_form():
    """Hand-evaluated Wilson centre/half-width for d=3, t=10, z=1."""
    lo, hi = wilson_interval(3.0, 10.0, z=1.0)
    t, d, z = 10.0, 3.0, 1.0
    p = d / t
    den = 1.0 + z * z / t
    c = (p + z * z / (2 * t)) / den
    h = (z / den) * np.sqrt(p * (1 - p) / t + z * z / (4 * t * t))
    assert abs(lo - (c - h)) < ATOL and abs(hi - (c + h)) < ATOL


def test_wilson_stays_inside_unit_interval_at_the_boundaries():
    """The whole point of Wilson over Wald: p = 0 and p = 1 keep finite width
    INSIDE [0, 1].  The Wald interval would give zero width at both ends, which
    would hide precisely the low-N / low-S/N cells this review is about."""
    for d, t in [(0.0, 40.0), (40.0, 40.0)]:
        lo, hi = wilson_interval(d, t, z=1.0)
        assert 0.0 <= lo < hi <= 1.0
        assert hi - lo > 1e-3                      # MUTATION: Wald gives 0 here
    lo, hi = wilson_interval(0.0, 40.0)
    assert lo == 0.0 and hi > 0.0


def test_wilson_zero_trials_is_nan_not_zero():
    lo, hi = wilson_interval(0.0, 0.0)
    assert np.isnan(lo) and np.isnan(hi)


def test_wilson_width_shrinks_like_one_over_sqrt_n():
    _, h1 = wilson_interval(50.0, 100.0)
    _, h4 = wilson_interval(200.0, 400.0)
    w1, w4 = h1 - 0.5, h4 - 0.5
    assert abs(w1 / w4 - 2.0) < 0.05


# --------------------------------------------------------------------------
# standardised residual
# --------------------------------------------------------------------------
def test_standardised_residual_value_and_sign():
    """pred above the held-out rate must give a POSITIVE residual, scaled by the
    PREDICTIVE sd (not the observed one)."""
    r = standardised_residual(0.6, 50.0, 100.0)
    expect = (0.6 - 0.5) / np.sqrt(0.6 * 0.4 / 100.0)
    assert abs(r - expect) < 1e-10
    assert r > 0


def test_standardised_residual_uses_predicted_not_observed_variance():
    """MUTATION: the observed-rate denominator sqrt(q(1-q)/n) gives a different
    number; the two must not coincide."""
    r = standardised_residual(0.9, 50.0, 100.0)
    wrong = (0.9 - 0.5) / np.sqrt(0.5 * 0.5 / 100.0)
    assert abs(r - wrong) > 1.0


def test_standardised_residual_finite_at_degenerate_heldout_rate():
    """Held-out rate exactly 0 or 1 must still give a finite residual."""
    assert np.isfinite(standardised_residual(0.3, 0.0, 25.0))
    assert np.isfinite(standardised_residual(0.3, 25.0, 25.0))


def test_standardised_residual_masks_empty_cells():
    assert np.isnan(standardised_residual(0.5, 0.0, 0.0))


def test_standardised_residual_scales_with_sqrt_n():
    a = standardised_residual(0.6, 50.0, 100.0)
    b = standardised_residual(0.6, 200.0, 400.0)
    assert abs(b / a - 2.0) < 1e-9


# --------------------------------------------------------------------------
# the C1nsadd functional form
# --------------------------------------------------------------------------
BETA = np.array([3.0324046495187598, 2.3205982668318885, -5.254314882456282e-05,
                 -0.20562775010566917, 5.492141469719244, -7.22937653012373])


def test_additive_logit_matches_explicit_polynomial():
    x, u = 0.35, -0.12
    eta = (BETA[0] + BETA[1] * x + BETA[2] * x ** 2 + BETA[3] * x ** 3
           + BETA[4] * u + BETA[5] * u ** 2)
    p = 1.0 / (1.0 + np.exp(-eta))
    assert abs(additive_logit_C(x, u, BETA) - p) < 1e-13


def test_additive_logit_has_no_interaction():
    """MUTATION guard: an additive logit satisfies
    eta(x1,u1) + eta(x2,u2) == eta(x1,u2) + eta(x2,u1) exactly.  A tensor
    (interaction) surface does not."""
    def lg(p):
        return np.log(p) - np.log1p(-p)
    x1, x2, u1, u2 = 0.3, 1.1, -0.2, 0.34
    a = lg(additive_logit_C(x1, u1, BETA)) + lg(additive_logit_C(x2, u2, BETA))
    b = lg(additive_logit_C(x1, u2, BETA)) + lg(additive_logit_C(x2, u1, BETA))
    assert abs(a - b) < 1e-9


def test_additive_logit_broadcasts_to_a_surface():
    x = np.linspace(-1.0, 2.0, 7)[:, None]
    u = np.linspace(-0.3, 0.35, 5)[None, :]
    C = additive_logit_C(x, u, BETA)
    assert C.shape == (7, 5)
    assert np.all((C > 0.0) & (C < 1.0))
    # monotone increasing in N over the calibrated range
    assert np.all(np.diff(C, axis=0) > 0)


def test_additive_logit_rejects_wrong_coefficient_count():
    with pytest.raises(ValueError):
        additive_logit_C(0.0, 0.0, BETA[:5])


def test_additive_logit_is_branch_stable_at_large_logit():
    """Neither branch may overflow: the negative branch must stay strictly
    positive (exp(-eta) would overflow) and the positive branch must stay
    finite and <= 1 (float64 rounds sigmoid(50) to exactly 1)."""
    lo = additive_logit_C(-50.0, 0.0, np.array([0., 1., 0., 0., 0., 0.]))
    hi = additive_logit_C(50.0, 0.0, np.array([0., 1., 0., 0., 0., 0.]))
    assert 0.0 < lo < 1e-20
    assert np.isfinite(hi) and 0.999 < hi <= 1.0


# --------------------------------------------------------------------------
# phi and the effective completeness
# --------------------------------------------------------------------------
def test_phi_measured_is_ingrid_over_all():
    ing = np.zeros((2, 3, 2))
    alls = np.zeros((2, 3, 2))
    ing[0, :, 0] = [1.0, 2.0, 3.0]
    alls[0, :, 0] = [2.0, 4.0, 6.0]
    phi = phi_measured(ing, alls)
    assert abs(phi[0, 0] - 0.5) < ATOL
    # no detections anywhere -> fill, not 0/0
    assert abs(phi[1, 1] - 1.0) < ATOL


def test_phi_measured_pools_before_dividing():
    """MUTATION: averaging per-k ratios is NOT the pooled ratio."""
    ing = np.array([[[1.0], [9.0]]])          # (B=1, Kf=2, S=1)
    alls = np.array([[[2.0], [10.0]]])
    pooled = phi_measured(ing, alls)[0, 0]
    mean_of_ratios = 0.5 * (1 / 2 + 9 / 10)
    assert abs(pooled - 10.0 / 12.0) < ATOL
    assert abs(pooled - mean_of_ratios) > 0.02


def test_effective_completeness_shape_and_value():
    C = np.arange(6.0).reshape(2, 3) / 10.0        # (S=2, B=3)
    g = np.arange(12.0).reshape(3, 4) / 12.0       # (B=3, Kf=4)
    E = effective_completeness(C, g)
    assert E.shape == (3, 4, 2)
    assert abs(E[1, 2, 0] - C[0, 1] * g[1, 2]) < ATOL


def test_effective_completeness_rejects_transposed_input():
    with pytest.raises(ValueError):
        effective_completeness(np.zeros((3, 2)), np.zeros((3, 4)))


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------
def _toy(seed=7):
    rng = np.random.default_rng(seed)
    S, Kf, C_n, B = 2, 3, 4, 5
    Mg = rng.random((S, Kf, C_n, B))
    Mg = Mg / Mg.sum(axis=2, keepdims=True) * 0.8      # a kernel carrying phi = 0.8
    C = rng.random((S, B))
    g = 0.5 + rng.random((B, Kf))
    tc = rng.integers(1, 50, size=(B, Kf, S)).astype(float)
    return Mg, C, g, tc


def test_fold_2d_matches_an_explicit_loop():
    Mg, C, g, tc = _toy()
    mu = fold_tp_2d(Mg, C, g, tc)
    S, Kf, C_n, B = Mg.shape
    ref = np.zeros((C_n, Kf, S))
    for c in range(C_n):
        for k in range(Kf):
            for s in range(S):
                ref[c, k, s] = sum(Mg[s, k, c, b] * C[s, b] * g[b, k] * tc[b, k, s]
                                   for b in range(B))
    assert np.max(np.abs(mu - ref)) < 1e-11


def test_fold_2d_use_g_false_drops_g():
    Mg, C, g, tc = _toy()
    a = fold_tp_2d(Mg, C, g, tc, use_g=False)
    b = fold_tp_2d(Mg, C, np.ones_like(g), tc)
    assert np.max(np.abs(a - b)) < 1e-12
    # MUTATION: and it is NOT the same as keeping g
    assert np.max(np.abs(a - fold_tp_2d(Mg, C, g, tc))) > 1e-3


def test_fold_3d_equals_2d_when_C_is_broadcast_times_g():
    """The C*g comparison of Addendum A, as an identity: the 3-D branch fed
    C[s,b]*g[b,k] reproduces the 2-D branch exactly.  This is the check that
    makes 'C1nsadd x g vs C_true_bKs' a like-for-like comparison."""
    Mg, C, g, tc = _toy()
    C_bks = effective_completeness(C, g)
    assert np.max(np.abs(fold_tp_3d(Mg, C_bks, tc) - fold_tp_2d(Mg, C, g, tc))) < 1e-11


def test_fold_is_linear_in_the_completeness():
    """Doubling C doubles the predicted counts — so a multiplicative phi applied
    to C is exactly a second application of the kernel's in-grid fraction."""
    Mg, C, g, tc = _toy()
    a = fold_tp_2d(Mg, C, g, tc)
    b = fold_tp_2d(Mg, 2.0 * C, g, tc)
    assert np.max(np.abs(b - 2.0 * a)) < 1e-10


def test_kernel_phi_is_recoverable_from_the_fold():
    """sum_c Mg = phi by construction in the toy kernel; the review's central
    claim rests on this being < 1."""
    Mg, _, _, _ = _toy()
    assert np.max(np.abs(Mg.sum(axis=2) - 0.8)) < 1e-12


# --------------------------------------------------------------------------
# marginals
# --------------------------------------------------------------------------
def test_snr_marginal_normalisation_removes_the_level():
    rng = np.random.default_rng(3)
    obs = rng.integers(10, 100, size=(4, 3, 2)).astype(float)
    mu = obs * 1.3
    r, tot = snr_marginal(mu, obs)
    assert abs(tot - 1.3) < 1e-12
    assert np.max(np.abs(r - 1.0)) < 1e-12
    r_raw, _ = snr_marginal(mu, obs, normalise=False)
    assert np.max(np.abs(r_raw - 1.3)) < 1e-12


def test_snr_marginal_detects_a_pure_shape_tilt():
    obs = np.ones((2, 2, 3)) * 100.0
    mu = obs.copy()
    mu[:, :, 0] *= 0.9
    mu[:, :, 2] *= 1.1
    r, _ = snr_marginal(mu, obs)
    assert r[0] < 0.95 < 1.05 < r[2]
    assert abs(r.mean() - 1.0) < 0.02


def test_pooled_rate_is_a_ratio_of_sums():
    d = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = np.array([[2.0, 8.0], [6.0, 4.0]])
    assert abs(pooled_rate(d, t) - 10.0 / 20.0) < ATOL
    r = pooled_rate(d, t, axis=1)
    assert abs(r[0] - 3.0 / 10.0) < ATOL
    # MUTATION: mean of cell rates would be (1/2+2/8)/2 = 0.375, not 0.3
    assert abs(r[0] - 0.375) > 0.05


def test_pooled_rate_empty_is_nan():
    assert np.isnan(pooled_rate(np.zeros(3), np.zeros(3)))
