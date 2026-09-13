"""Unit tests for the FIXED completeness calibration helpers.

VALIDATION-ONLY code under ``validation/absorber_ladder/completeness/``; these
tests import ONLY the pure numpy helpers (``cal_fit``) so they run in seconds
under either environment (no jax, no sampler, no catalogue I/O).

Every numeric comparison uses an EXPLICIT ``atol`` (never ``np.allclose``'s
vacuous default on tiny values) and several tests are written as exactness
checks — a wrong normalisation or a swapped fold must make them fail, not just
widen a tolerance.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAL = os.path.abspath(os.path.join(
    _HERE, "..", "validation", "absorber_ladder", "completeness"))
if _CAL not in sys.path:
    sys.path.insert(0, _CAL)

from cal_fit import (                                              # noqa: E402
    expit, logit, eta_hat_jeffreys, poly_design, design_per_stratum,
    design_2d_tensor, design_2d_additive, design_additive_z, irls_binomial,
    binom_logloss, binom_deviance, logloss_diff_se, cv_two_fold,
    parity_halves, weighted_ratio_residual)


# ---------------------------------------------------------------- link -----
def test_expit_logit_roundtrip_and_extremes():
    x = np.array([-800.0, -40.0, -1.0, 0.0, 0.3, 12.0, 800.0])
    p = expit(x)
    assert np.all(np.isfinite(p))
    assert p[0] >= 0.0 and p[-1] <= 1.0
    # no overflow warning path: the negative branch must not compute exp(+800)
    mid = x[1:-1]
    assert np.max(np.abs(logit(expit(mid)) - mid)) < 1e-9
    assert abs(expit(np.array([0.0]))[0] - 0.5) < 1e-15


def test_eta_hat_matches_the_committed_jeffreys_formula():
    d = np.array([[0.0, 3.0, 11.0], [99.0, 1883.0, 5.0]])
    t = np.array([[10.0, 12.0, 11.0], [11241.0, 13876.0, 5.0]])
    eta, sig = eta_hat_jeffreys(d, t)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ref_e = math.log((d[i, j] + 0.5) / (t[i, j] - d[i, j] + 0.5))
            ref_s = math.sqrt(1.0 / (d[i, j] + 0.5)
                              + 1.0 / (t[i, j] - d[i, j] + 0.5))
            assert abs(eta[i, j] - ref_e) < 1e-13
            assert abs(sig[i, j] - ref_s) < 1e-13
    # a fully detected cell must stay FINITE (this is why Jeffreys is used)
    assert np.all(np.isfinite(eta))


def test_eta_hat_agrees_with_the_pipeline_implementation_if_importable():
    try:
        from CDDF_analysis.hbi_mcmc.forward import eta_hat_sigma_hat
    except Exception:                       # jax-free env: nothing to compare
        pytest.skip("CDDF_analysis.hbi_mcmc.forward not importable here")
    d = np.array([[0.0, 3.0, 11.0], [99.0, 1883.0, 5.0]])
    t = np.array([[10.0, 12.0, 11.0], [11241.0, 13876.0, 5.0]])
    e1, s1 = eta_hat_jeffreys(d, t)
    e2, s2 = eta_hat_sigma_hat(d, t)
    assert np.array_equal(e1, e2) and np.array_equal(s1, s2)


# -------------------------------------------------------------- designs ----
def test_poly_design_values_and_shape():
    x = np.array([-1.0, 0.0, 2.0])
    X = poly_design(x, 3)
    assert X.shape == (3, 4)
    assert np.max(np.abs(X[:, 0] - 1.0)) == 0.0
    assert np.max(np.abs(X[:, 2] - x ** 2)) == 0.0
    assert np.max(np.abs(X[:, 3] - x ** 3)) == 0.0


def test_design_per_stratum_is_block_diagonal_in_row_order_s_then_b():
    x = np.array([-0.5, 0.0, 0.5, 1.0])
    S, deg = 3, 2
    X = design_per_stratum(x, S, deg)
    assert X.shape == (S * x.size, S * (deg + 1))
    P = poly_design(x, deg)
    for s in range(S):
        blk = X[s * x.size:(s + 1) * x.size]
        assert np.array_equal(blk[:, s * (deg + 1):(s + 1) * (deg + 1)], P)
        other = np.delete(blk, np.arange(s * (deg + 1), (s + 1) * (deg + 1)),
                          axis=1)
        assert np.count_nonzero(other) == 0


def test_design_2d_tensor_row_order_and_degenerate_cases():
    x = np.array([-1.0, 0.0, 1.0])
    u = np.array([-0.2, 0.4])
    X = design_2d_tensor(x, u, 2, 1)
    assert X.shape == (u.size * x.size, 3 * 2)
    # rows must be ordered (s, b) C-style, matching a (S, B) reshape
    for si in range(u.size):
        for bi in range(x.size):
            row = X[si * x.size + bi]
            k = 0
            for j in range(3):
                for i in range(2):
                    assert abs(row[k] - x[bi] ** j * u[si] ** i) < 1e-14
                    k += 1
    # deg_u = 0 collapses to one shared polynomial in x
    X0 = design_2d_tensor(x, u, 2, 0)
    assert X0.shape == (6, 3)
    assert np.array_equal(X0[:3], X0[3:])


def test_design_2d_additive_has_no_interaction_columns():
    x = np.array([-1.0, 0.0, 1.0])
    u = np.array([-0.2, 0.4, 1.1])
    X = design_2d_additive(x, u, 3, 2)
    assert X.shape == (9, 3 + 2 + 1)
    # the x columns repeat identically in every stratum block
    for s in range(1, 3):
        assert np.array_equal(X[s * 3:(s + 1) * 3, 1:4], X[0:3, 1:4])
    # the u columns are constant within a stratum block
    for s in range(3):
        blk = X[s * 3:(s + 1) * 3, 4:]
        assert np.max(np.abs(blk - blk[0])) == 0.0


def test_design_additive_z_dummies_and_reference_level():
    x = np.array([-1.0, 1.0])
    u = np.array([0.0, 0.5])
    nK = 3
    X = design_additive_z(x, u, nK, 2, 1)
    assert X.shape == (u.size * nK * x.size, 1 + 2 + 1 + (nK - 1))
    dum = X[:, -(nK - 1):]
    # rows ordered (s, K, b): the first B rows are K = 0, the reference level
    assert np.count_nonzero(dum[:x.size]) == 0
    assert np.array_equal(dum[x.size:2 * x.size, 0], np.ones(x.size))
    assert np.count_nonzero(dum[x.size:2 * x.size, 1]) == 0
    assert np.array_equal(dum[2 * x.size:3 * x.size, 1], np.ones(x.size))


# ----------------------------------------------------------------- IRLS ----
def test_irls_saturated_design_reproduces_the_cellwise_logit_exactly():
    """A one-hot design must return log(d/(t-d)) in every cell.

    This is an exactness check with no tolerance slack to hide in: any
    mis-scaled weight, gradient or step would move the answer.
    """
    d = np.array([3.0, 40.0, 97.0, 1.0])
    t = np.array([10.0, 80.0, 100.0, 4.0])
    X = np.eye(4)
    r = irls_binomial(X, d, t, ridge=0.0, tol=1e-14, max_iter=400)
    assert r["converged"]
    ref = np.log(d / (t - d))
    assert np.max(np.abs(r["beta"] - ref)) < 1e-9
    assert np.max(np.abs(r["p"] - d / t)) < 1e-11
    assert abs(r["deviance"]) < 1e-9        # saturated => zero deviance


def test_irls_recovers_a_known_logistic_truth_on_large_synthetic_counts():
    rng = np.random.default_rng(20260913)
    x = np.linspace(-1.2, 2.2, 16)
    beta_true = np.array([1.3, 1.9, -0.35])
    X = poly_design(x, 2)
    p = expit(X @ beta_true)
    t = np.full(16, 200000.0)
    d = rng.binomial(t.astype(int), p).astype(float)
    r = irls_binomial(X, d, t, ridge=0.0)
    assert r["converged"]
    assert np.max(np.abs(r["beta"] - beta_true)) < 0.05
    # the fitted probabilities must track the empirical rates to ~1/sqrt(n)
    assert np.max(np.abs(r["p"] - d / t)) < 5e-3


def test_irls_ignores_zero_trial_cells_but_still_predicts_them():
    d = np.array([5.0, 0.0, 20.0])
    t = np.array([10.0, 0.0, 40.0])
    X = np.stack([np.ones(3), np.array([-1.0, 0.0, 1.0])], axis=1)
    r = irls_binomial(X, d, t, ridge=0.0)
    assert r["n_cells_used"] == 2
    assert np.all(np.isfinite(r["p"]))      # the t = 0 cell is still predicted


def test_irls_rejects_impossible_counts():
    X = np.ones((2, 1))
    with pytest.raises(ValueError):
        irls_binomial(X, [3.0, 1.0], [2.0, 1.0])
    with pytest.raises(ValueError):
        irls_binomial(X, [-1.0, 1.0], [2.0, 1.0])
    with pytest.raises(ValueError):
        irls_binomial(np.ones((3, 1)), [1.0], [1.0])


def test_ridge_penalises_only_the_slope_columns():
    d = np.array([5.0, 20.0, 35.0])
    t = np.array([50.0, 50.0, 50.0])
    X = poly_design(np.array([-1.0, 0.0, 1.0]), 1)
    big = irls_binomial(X, d, t, ridge=1e4)
    # a huge ridge must crush the SLOPE, not the intercept
    assert abs(big["beta"][1]) < 1e-2
    assert abs(big["beta"][0] - logit(np.array([60.0 / 150.0]))[0]) < 0.05
    tiny = irls_binomial(X, d, t, ridge=1e-6)
    assert abs(tiny["beta"][1]) > 0.5


# ---------------------------------------------------------------- scores ---
def test_binom_logloss_matches_a_direct_computation():
    p = np.array([0.2, 0.9])
    d = np.array([3.0, 17.0])
    t = np.array([10.0, 20.0])
    ref = -(3 * math.log(0.2) + 7 * math.log(0.8)
            + 17 * math.log(0.9) + 3 * math.log(0.1))
    assert abs(binom_logloss(p, d, t) - ref) < 1e-10
    per = binom_logloss(p, d, t, total=False)
    assert abs(per.sum() - ref) < 1e-10


def test_binom_deviance_is_zero_at_the_saturated_fit_and_detects_mis_scaling():
    d = np.array([3.0, 17.0, 0.0])
    t = np.array([10.0, 20.0, 0.0])
    q = np.array([0.3, 0.85, 0.5])
    assert abs(binom_deviance(q, d, t)) < 1e-12
    # a 2 % multiplicative error on C must be caught, not absorbed
    assert binom_deviance(q * 1.02, d, t) > 0.05
    # and the t = 0 cell must contribute NOTHING (one-sided-support guard)
    assert abs(binom_deviance(np.array([0.3, 0.85, 0.99]), d, t)) < 1e-12


def test_logloss_diff_se_is_zero_for_identical_predictions_and_scales_as_sqrt_n():
    p = np.array([0.4, 0.7])
    d = np.array([40.0, 70.0])
    t = np.array([100.0, 100.0])
    assert logloss_diff_se(p, p, d, t) == 0.0
    se1 = logloss_diff_se(p, p + 0.05, d, t)
    se4 = logloss_diff_se(p, p + 0.05, 4 * d, 4 * t)
    assert abs(se4 / se1 - 2.0) < 1e-9


def test_logloss_diff_se_uses_the_PAIRED_per_trial_difference():
    """Exactness check on the difference variable itself.

    Per held-out trial the log-loss difference is ``-log(p_b/p_a)`` when the
    system is detected and ``-log((1-p_b)/(1-p_a))`` when it is not, so the
    per-trial spread is the DIFFERENCE of those two log-ratios.  Dropping the
    second term (scoring the detections only) must change the answer.
    """
    p_a = np.array([0.30])
    p_b = np.array([0.45])
    d = np.array([25.0])
    t = np.array([100.0])
    q = 0.25
    delta = (math.log(0.45) - math.log(0.30)) \
        - (math.log(0.55) - math.log(0.70))
    ref = math.sqrt(100.0 * q * (1.0 - q) * delta ** 2)
    assert abs(logloss_diff_se(p_a, p_b, d, t) - ref) < 1e-12
    # the unpaired (detections-only) version would give a DIFFERENT number
    unpaired = math.sqrt(100.0 * q * (1.0 - q)
                         * (math.log(0.45) - math.log(0.30)) ** 2)
    assert abs(ref - unpaired) > 0.1


# -------------------------------------------------------------------- CV ---
def test_cv_two_fold_actually_swaps_the_folds():
    """The score on half B must come from the fit on half A, and vice versa."""
    det_A = np.array([[10.0, 90.0]])
    tot_A = np.array([[100.0, 100.0]])
    det_B = np.array([[20.0, 80.0]])
    tot_B = np.array([[100.0, 100.0]])

    def fit(d, t):
        return dict(p=np.where(t > 0, d / np.maximum(t, 1.0), 0.5))

    out = cv_two_fold(fit, lambda st: st["p"], det_A, tot_A, det_B, tot_B)
    assert np.array_equal(out["p_on_B"], det_A / tot_A)     # fitted on A
    assert np.array_equal(out["p_on_A"], det_B / tot_B)     # fitted on B
    ref_B = binom_logloss(det_A / tot_A, det_B, tot_B)
    assert abs(out["fold_fitA_scoreB"] - ref_B) < 1e-10
    assert abs(out["logloss"] - (out["fold_fitA_scoreB"]
                                 + out["fold_fitB_scoreA"])) < 1e-10
    assert out["n_heldout_trials"] == 400.0
    assert abs(out["logloss_per_trial"] - out["logloss"] / 400.0) < 1e-12


def test_cv_two_fold_mask_restricts_only_the_scored_cells():
    det_A = np.array([[10.0, 90.0]])
    tot_A = np.array([[100.0, 100.0]])
    det_B = np.array([[20.0, 80.0]])
    tot_B = np.array([[100.0, 100.0]])

    def fit(d, t):
        # a GLOBAL rate: if the mask leaked into the fit the value would change
        return dict(p=np.full(d.shape, d.sum() / t.sum()))

    m = np.array([[True, False]])
    out = cv_two_fold(fit, lambda st: st["p"], det_A, tot_A, det_B, tot_B,
                      mask=m)
    assert abs(out["p_on_B"][0, 0] - 0.5) < 1e-12       # fit used BOTH cells
    assert out["n_heldout_trials"] == 200.0             # scored ONE cell/fold


def test_parity_halves_partition_exactly():
    tid = np.array([1, 2, 3, 4, 10 ** 16, 10 ** 16 + 1], dtype=np.int64)
    e, o = parity_halves(tid)
    assert np.count_nonzero(e & o) == 0
    assert np.all(e | o)
    assert e.tolist() == [False, True, False, True, True, False]


# ----------------------------------------------------------- residuals -----
def test_weighted_ratio_residual_is_expected_over_observed_minus_one():
    p = np.array([[0.5, 0.25]])
    d = np.array([[40.0, 30.0]])
    t = np.array([[100.0, 200.0]])
    # expected = 0.5*100 + 0.25*200 = 100; observed = 70
    assert abs(weighted_ratio_residual(p, d, t) - (100.0 / 70.0 - 1.0)) < 1e-12
    per_b = weighted_ratio_residual(p, d, t, axis=0)
    assert abs(per_b[0] - (50.0 / 40.0 - 1.0)) < 1e-12
    assert abs(per_b[1] - (50.0 / 30.0 - 1.0)) < 1e-12


def test_weighted_ratio_residual_returns_nan_where_nothing_was_observed():
    p = np.array([[0.5, 0.25]])
    d = np.array([[0.0, 30.0]])
    t = np.array([[100.0, 200.0]])
    per_b = weighted_ratio_residual(p, d, t, axis=0)
    assert np.isnan(per_b[0]) and np.isfinite(per_b[1])


# ------------------------------------------------- grid / C0 conventions ---
def test_b_to_cell_digitize_convention_matches_the_fold():
    """``forward.build_consts``: b_to_cell = clip(digitize(Nc, molly_edges)-1)."""
    ntrue = np.array([19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7,
                      20.9, 21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4])
    me = np.array([17.2, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.3, 20.5, 21.0,
                   21.5, 22.0, np.inf])
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    b2c = np.clip(np.digitize(Nc, me) - 1, 0, len(me) - 2)
    assert b2c.tolist() == [4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11]
    # the grid-resolution defect is exactly this: several 0.2-dex bins share a
    # single molly cell, so C0 cannot vary inside a cell
    assert len(set(b2c.tolist())) == 8 < len(Nc)
