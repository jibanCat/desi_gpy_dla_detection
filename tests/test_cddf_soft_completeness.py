"""Tests for the diagonal soft-completeness Bayesian core
(``CDDF_analysis.cddf_forward.soft_completeness``).

WHY THIS EXISTS
---------------
O3 corrects the raw probabilistic expected count ``F_b`` in each (logN, z) bin
``b`` by a per-bin scalar completeness ``C_b`` and an additive soft
false-positive deposit ``b_FP_b``::

    n_corr_b = (F_b - b_FP_b) / C_b

This module is the PURE-ARRAY Bayesian inference layer (contract §2): given
posterior-weighted (fractional) matched / unmatched / truth counts, it returns
closed-form Beta / Gamma posteriors for ``C`` and ``b_FP``, and propagates the
three uncertainty sources (F count CI, C posterior, b_FP posterior) through the
ratio by *ancestral sampling from the already-closed-form posteriors* (NOT an
MCMC/sampler over a model).

The tests pin:
  * the fractional-success Beta generalization and every contract edge case
    (n_truth==0 -> NaN+masked; f_matched>n_truth -> C clipped to 1 + flag;
    f_matched==0, n_truth>0 -> C->0+);
  * the Gamma FP-deposit posterior and its units contract (b_FP in F's units);
  * three-source error propagation, non-negativity clipping, valid-mask flow;
  * an SBC (simulation-based calibration) harness whose 68/95 interval coverage
    and rank statistic must be calibrated within Monte-Carlo tolerance.

SBC here validates the INFERENCE layer assuming C / b_FP are known; it does NOT
validate the response-matrix build (that is the later injection campaign +
Campaign D).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
# Repo root on sys.path so ``CDDF_analysis`` resolves as a (namespace) package.
sys.path.insert(0, os.path.join(_HERE, ".."))

from CDDF_analysis.cddf_forward.soft_completeness import (  # noqa: E402
    estimate_diagonal_completeness,
    estimate_false_positive_deposit,
    apply_diagonal_correction,
    omega_from_draws,
    toy_count_mock,
    sbc_coverage,
)


# --------------------------------------------------------------------------- #
# 2.1  estimate_diagonal_completeness
# --------------------------------------------------------------------------- #
class TestCompleteness:
    def test_returns_all_required_keys(self):
        out = estimate_diagonal_completeness(
            np.array([5.0, 8.0]), np.array([10.0, 10.0])
        )
        for k in ("C", "C_lo68", "C_hi68", "C_lo95", "C_hi95", "valid_mask"):
            assert k in out, f"missing key {k}"

    def test_shapes_match_input(self):
        out = estimate_diagonal_completeness(
            np.array([5.0, 8.0, 1.0]), np.array([10.0, 10.0, 10.0])
        )
        for k in ("C", "C_lo68", "C_hi68", "C_lo95", "C_hi95", "valid_mask"):
            assert out[k].shape == (3,)

    def test_point_estimate_is_beta_posterior_mean(self):
        # Posterior mean of Beta(f+a, n-f+b) = (f+a)/(n+a+b).
        f = np.array([5.0, 0.0, 10.0])
        n = np.array([10.0, 10.0, 10.0])
        a, b = 0.5, 0.5
        out = estimate_diagonal_completeness(f, n, prior=(a, b))
        expected = (f + a) / (n + a + b)
        np.testing.assert_allclose(out["C"], expected, rtol=1e-10)

    def test_intervals_bracket_point_and_nested(self):
        out = estimate_diagonal_completeness(
            np.array([5.0, 8.0]), np.array([10.0, 10.0])
        )
        assert np.all(out["C_lo95"] <= out["C_lo68"])
        assert np.all(out["C_lo68"] <= out["C"])
        assert np.all(out["C"] <= out["C_hi68"])
        assert np.all(out["C_hi68"] <= out["C_hi95"])

    def test_C_clipped_to_unit_interval(self):
        # All finite C must lie in (0, 1].
        out = estimate_diagonal_completeness(
            np.array([0.0, 5.0, 12.0]), np.array([10.0, 10.0, 10.0])
        )
        C = out["C"]
        finite = np.isfinite(C)
        assert np.all(C[finite] > 0.0)
        assert np.all(C[finite] <= 1.0)
        assert np.all(out["C_hi95"][finite] <= 1.0)
        assert np.all(out["C_lo95"][finite] >= 0.0)

    def test_n_truth_zero_is_nan_and_masked(self):
        out = estimate_diagonal_completeness(
            np.array([0.0, 5.0]), np.array([0.0, 10.0])
        )
        assert np.isnan(out["C"][0])
        assert np.isnan(out["C_lo68"][0])
        assert np.isnan(out["C_hi95"][0])
        assert out["valid_mask"][0] == False  # noqa: E712
        assert out["valid_mask"][1] == True  # noqa: E712
        # Never silently 0 or 1 for the undefined bin.
        assert out["C"][0] != 0.0
        assert out["C"][0] != 1.0

    def test_upscatter_clips_C_to_one_with_flag(self):
        # f_matched > n_truth (up-scatter) -> C clipped to 1, recorded flag.
        out = estimate_diagonal_completeness(
            np.array([12.0, 5.0]), np.array([10.0, 10.0])
        )
        assert out["C"][0] == pytest.approx(1.0)
        assert out["C"][0] <= 1.0
        assert "upscatter_mask" in out
        assert out["upscatter_mask"][0] == True  # noqa: E712
        assert out["upscatter_mask"][1] == False  # noqa: E712

    def test_zero_matched_drives_C_low(self):
        # f_matched=0, n_truth>0 -> C -> 0+ (incomplete), lower CI near 0.
        out = estimate_diagonal_completeness(
            np.array([0.0]), np.array([50.0])
        )
        assert out["C"][0] < 0.05
        assert out["C_lo68"][0] >= 0.0
        assert out["C_lo68"][0] < out["C"][0] + 1e-6

    def test_more_truth_tightens_interval(self):
        # Same fractional completeness, more statistics -> narrower CI.
        small = estimate_diagonal_completeness(
            np.array([5.0]), np.array([10.0])
        )
        big = estimate_diagonal_completeness(
            np.array([500.0]), np.array([1000.0])
        )
        w_small = small["C_hi68"][0] - small["C_lo68"][0]
        w_big = big["C_hi68"][0] - big["C_lo68"][0]
        assert w_big < w_small

    def test_vectorized_matches_scalar_loop(self):
        rng = np.random.default_rng(0)
        n = rng.integers(1, 100, size=20).astype(float)
        f = rng.uniform(0, 1, size=20) * n
        vec = estimate_diagonal_completeness(f, n)
        for i in range(20):
            one = estimate_diagonal_completeness(f[i : i + 1], n[i : i + 1])
            np.testing.assert_allclose(vec["C"][i], one["C"][0], rtol=1e-10)
            np.testing.assert_allclose(
                vec["C_lo68"][i], one["C_lo68"][0], rtol=1e-9
            )


# --------------------------------------------------------------------------- #
# 2.2  estimate_false_positive_deposit
# --------------------------------------------------------------------------- #
class TestFalsePositiveDeposit:
    def test_returns_all_required_keys(self):
        out = estimate_false_positive_deposit(
            np.array([2.0, 0.0]), np.array([100.0, 100.0])
        )
        for k in ("b_FP", "b_FP_lo68", "b_FP_hi68", "b_FP_lo95", "b_FP_hi95"):
            assert k in out, f"missing key {k}"

    def test_point_is_gamma_mode_in_count_units(self):
        # B4: b_FP is a deposit (count) whose POINT is the Gamma posterior MODE
        # max(f_unmatched + a - 1, 0), NOT the mean. With a=0.5 a clean bin
        # (f_unmatched=0) has b_FP point == 0 exactly (no phantom subtraction).
        f = np.array([2.0, 0.0, 10.0])
        a = 0.5
        out = estimate_false_positive_deposit(
            f, np.array([100.0, 100.0, 100.0]), prior=(a,)
        )
        expected = np.maximum(f + a - 1.0, 0.0)  # mode
        np.testing.assert_allclose(out["b_FP"], expected, rtol=1e-10)
        assert out["b_FP"][1] == 0.0  # clean bin: zero point

    def test_deposit_nonnegative(self):
        out = estimate_false_positive_deposit(
            np.array([0.0, 1.0, 5.0]), 100.0
        )
        assert np.all(out["b_FP"] >= 0.0)
        assert np.all(out["b_FP_lo95"] >= 0.0)

    def test_intervals_nested_and_bracket_point(self):
        out = estimate_false_positive_deposit(
            np.array([5.0, 20.0]), np.array([100.0, 100.0])
        )
        assert np.all(out["b_FP_lo95"] <= out["b_FP_lo68"])
        assert np.all(out["b_FP_lo68"] <= out["b_FP"])
        assert np.all(out["b_FP"] <= out["b_FP_hi68"])
        assert np.all(out["b_FP_hi68"] <= out["b_FP_hi95"])

    def test_scalar_exposure_broadcasts(self):
        out = estimate_false_positive_deposit(np.array([2.0, 3.0]), 50.0)
        assert out["b_FP"].shape == (2,)

    def test_exposure_array_accepted(self):
        out = estimate_false_positive_deposit(
            np.array([2.0, 3.0]), np.array([50.0, 80.0])
        )
        assert out["b_FP"].shape == (2,)

    def test_deposit_independent_of_exposure_value(self):
        # b_FP = rate * exposure with rate ~ Gamma(shape, exposure) => the
        # deposit (count) posterior is exposure-invariant by construction.
        a = estimate_false_positive_deposit(np.array([4.0]), 10.0)
        b = estimate_false_positive_deposit(np.array([4.0]), 1000.0)
        np.testing.assert_allclose(a["b_FP"], b["b_FP"], rtol=1e-10)
        np.testing.assert_allclose(a["b_FP_hi95"], b["b_FP_hi95"], rtol=1e-9)


# --------------------------------------------------------------------------- #
# 2.3  apply_diagonal_correction
# --------------------------------------------------------------------------- #
class TestApplyCorrection:
    def _ci(self, F, frac=0.3):
        F = np.asarray(F, float)
        w = frac * np.sqrt(np.maximum(F, 1.0))
        return {
            "lo68": np.maximum(F - w, 0.0),
            "hi68": F + w,
            "lo95": np.maximum(F - 2 * w, 0.0),
            "hi95": F + 2 * w,
        }

    def test_returns_all_required_keys(self):
        F = np.array([10.0, 20.0])
        C = estimate_diagonal_completeness(np.array([8.0, 18.0]), np.array([10.0, 20.0]))
        bfp = estimate_false_positive_deposit(np.array([1.0, 2.0]), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp)
        for k in ("n_corr", "lo68", "hi68", "lo95", "hi95", "neg_clip_mask", "valid_mask"):
            assert k in out, f"missing key {k}"

    def test_point_matches_formula(self):
        F = np.array([10.0, 20.0])
        C = estimate_diagonal_completeness(np.array([8.0, 18.0]), np.array([10.0, 20.0]))
        bfp = estimate_false_positive_deposit(np.array([1.0, 2.0]), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp)
        expected = (F - bfp["b_FP"]) / C["C"]
        np.testing.assert_allclose(out["n_corr"], expected, rtol=1e-10)

    def test_completeness_correction_increases_count(self):
        # C < 1 with negligible FP -> corrected count exceeds raw F.
        F = np.array([50.0])
        C = estimate_diagonal_completeness(np.array([25.0]), np.array([50.0]))  # C~0.5
        bfp = estimate_false_positive_deposit(np.array([0.0]), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp)
        assert out["n_corr"][0] > F[0]

    def test_intervals_nested_and_bracket_point(self):
        rng = np.random.default_rng(1)
        F = rng.uniform(10, 200, size=8)
        nt = rng.uniform(20, 300, size=8)
        fm = rng.uniform(0.4, 0.95, size=8) * nt
        C = estimate_diagonal_completeness(fm, nt)
        bfp = estimate_false_positive_deposit(rng.uniform(0, 5, size=8), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp, n_mc=8000)
        valid = out["valid_mask"]
        assert np.all(out["lo95"][valid] <= out["lo68"][valid] + 1e-9)
        assert np.all(out["lo68"][valid] <= out["hi68"][valid])
        assert np.all(out["hi68"][valid] <= out["hi95"][valid] + 1e-9)
        # point estimate inside the 95% interval
        assert np.all(out["lo95"][valid] <= out["n_corr"][valid] + 1e-6)
        assert np.all(out["n_corr"][valid] <= out["hi95"][valid] + 1e-6)

    def test_deterministic_given_seed(self):
        F = np.array([30.0, 40.0])
        C = estimate_diagonal_completeness(np.array([20.0, 30.0]), np.array([30.0, 40.0]))
        bfp = estimate_false_positive_deposit(np.array([1.0, 1.0]), 100.0)
        a = apply_diagonal_correction(F, self._ci(F), C, bfp, n_mc=2000)
        b = apply_diagonal_correction(F, self._ci(F), C, bfp, n_mc=2000)
        np.testing.assert_array_equal(a["lo68"], b["lo68"])
        np.testing.assert_array_equal(a["hi95"], b["hi95"])

    def test_negative_deposit_clips_to_zero(self):
        # b_FP > F -> (F - b_FP) < 0 -> n_corr clipped to 0, flag set.
        F = np.array([2.0, 50.0])
        C = estimate_diagonal_completeness(np.array([1.0, 40.0]), np.array([2.0, 50.0]))
        bfp = estimate_false_positive_deposit(np.array([20.0, 1.0]), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp)
        assert out["neg_clip_mask"][0] == True  # noqa: E712
        assert out["n_corr"][0] == 0.0
        assert out["neg_clip_mask"][1] == False  # noqa: E712

    def test_invalid_completeness_bin_stays_nan(self):
        F = np.array([10.0, 20.0])
        C = estimate_diagonal_completeness(np.array([0.0, 18.0]), np.array([0.0, 20.0]))
        bfp = estimate_false_positive_deposit(np.array([1.0, 2.0]), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp)
        assert out["valid_mask"][0] == False  # noqa: E712
        assert np.isnan(out["n_corr"][0])
        assert np.isnan(out["lo68"][0])
        assert out["valid_mask"][1] == True  # noqa: E712
        assert np.isfinite(out["n_corr"][1])

    def test_all_outputs_nonnegative_where_valid(self):
        rng = np.random.default_rng(3)
        F = rng.uniform(5, 100, size=10)
        nt = rng.uniform(10, 200, size=10)
        fm = rng.uniform(0.3, 1.0, size=10) * nt
        C = estimate_diagonal_completeness(fm, nt)
        bfp = estimate_false_positive_deposit(rng.uniform(0, 3, size=10), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp)
        v = out["valid_mask"]
        assert np.all(out["n_corr"][v] >= 0.0)
        assert np.all(out["lo68"][v] >= 0.0)
        assert np.all(out["lo95"][v] >= 0.0)


# --------------------------------------------------------------------------- #
# 2.4  toy_count_mock + SBC harness
# --------------------------------------------------------------------------- #
class TestToyMock:
    def test_keys_and_shapes(self):
        out = toy_count_mock(
            n_true=np.array([100.0, 50.0]),
            C=np.array([0.8, 0.6]),
            b_FP=np.array([2.0, 1.0]),
            exposure=200.0,
            seed=0,
        )
        for k in ("F", "f_matched", "f_unmatched", "n_truth", "n_true"):
            assert k in out
            assert np.asarray(out[k]).shape == (2,)

    def test_recovered_below_truth_on_average(self):
        # With C<1 and small FP, recovered F should sit below n_true.
        out = toy_count_mock(
            n_true=np.full(50, 1000.0),
            C=np.full(50, 0.7),
            b_FP=np.full(50, 1.0),
            exposure=2000.0,
            seed=1,
        )
        assert np.mean(out["F"]) < np.mean(out["n_true"])

    def test_deterministic_given_seed(self):
        a = toy_count_mock(np.array([100.0]), np.array([0.8]), np.array([1.0]), 200.0, seed=7)
        b = toy_count_mock(np.array([100.0]), np.array([0.8]), np.array([1.0]), 200.0, seed=7)
        np.testing.assert_array_equal(a["F"], b["F"])


class TestSBC:
    def test_interval_coverage_calibrated(self):
        # Over many sims with KNOWN C/b_FP, the 68% and 95% intervals of the
        # corrected estimate must cover n_true at the nominal rate (within MC
        # tolerance). This is the calibration gate for the inference layer.
        res = sbc_coverage(
            n_sims=400,
            nbin=6,
            seed=2026,
            n_true_range=(200.0, 2000.0),
            C_range=(0.5, 0.95),
            bfp_range=(0.0, 5.0),
            n_mc=2000,
        )
        cov68 = res["coverage68"]
        cov95 = res["coverage95"]
        # Pooled coverage across bins/sims, generous MC tolerance.
        assert abs(cov68 - 0.68) < 0.06, f"68% coverage = {cov68}"
        assert abs(cov95 - 0.95) < 0.04, f"95% coverage = {cov95}"

    def test_rank_statistic_uniform(self):
        # SBC rank of n_true within the posterior draws should be ~Uniform.
        res = sbc_coverage(
            n_sims=400,
            nbin=6,
            seed=99,
            n_true_range=(200.0, 2000.0),
            C_range=(0.5, 0.95),
            bfp_range=(0.0, 5.0),
            n_mc=2000,
            return_ranks=True,
        )
        ranks = np.asarray(res["ranks"])  # in [0, 1]
        ranks = ranks[np.isfinite(ranks)]
        assert ranks.size > 500
        # Mean of a Uniform(0,1) is 0.5; with N>~2000 samples the SE is small.
        assert abs(np.mean(ranks) - 0.5) < 0.05
        # Coarse-bin chi-square-ish uniformity: each decile within tolerance.
        hist, _ = np.histogram(ranks, bins=10, range=(0.0, 1.0))
        frac = hist / ranks.size
        assert np.all(np.abs(frac - 0.1) < 0.04), f"decile fractions {frac}"

    def test_biased_completeness_breaks_coverage(self):
        # Falsifiability: if we DELIBERATELY mis-estimate C (use wrong matched
        # counts), coverage must degrade. Guards against a vacuous SBC pass.
        res = sbc_coverage(
            n_sims=300,
            nbin=6,
            seed=5,
            n_true_range=(200.0, 2000.0),
            C_range=(0.5, 0.95),
            bfp_range=(0.0, 5.0),
            n_mc=1500,
            corrupt_completeness=0.5,  # inflate matched counts -> wrong C
        )
        assert res["coverage68"] < 0.6, f"corrupt cov68={res['coverage68']}"


# --------------------------------------------------------------------------- #
# B1  SBC mirror-production: one mock, 70/30 BUILD/whole, b_FP REBASED
# --------------------------------------------------------------------------- #
class TestSBCProductionMirror:
    """The production estimator (driver.compute_o3_products) measures C and a
    b_FP COUNT on the BUILD split but applies the correction to the WHOLE
    sample's F.  b_FP is a per-bin COUNT, so when the basis changes from BUILD
    (N_build sightlines) to the whole sample (N_whole sightlines) it MUST be
    rebased by the exposure ratio N_whole/N_build, else a real FP-scale bug is
    introduced.  The OLD ``sbc_coverage`` drew two INDEPENDENT EQUAL-SCALE mocks,
    so a missing rebasing cancelled and the SBC green-lit the bug.  The mirror
    harness draws ONE mock, splits it 70/30, and applies at the production
    exposure ratio — so omitting the rebasing MUST now break coverage.
    """

    def test_mirror_harness_is_calibrated_when_rebased(self):
        # With correct rebasing the production-mirrored harness is calibrated.
        res = sbc_coverage(
            n_sims=300,
            nbin=6,
            seed=2027,
            n_true_range=(400.0, 3000.0),
            C_range=(0.5, 0.95),
            bfp_range=(0.0, 5.0),
            n_mc=2000,
            production_mirror=True,
            build_frac=0.7,
            rebase_bfp=True,
        )
        assert abs(res["coverage68"] - 0.68) < 0.07, f"cov68={res['coverage68']}"
        assert abs(res["coverage95"] - 0.95) < 0.05, f"cov95={res['coverage95']}"

    def test_mirror_surfaces_missing_bfp_rebasing(self):
        # FALSIFIABILITY of the basis: if b_FP is NOT rebased from the BUILD
        # basis to the whole-sample basis (the exact production-path bug), the
        # FP subtraction is mis-scaled and the mirror harness MUST reveal it as
        # degraded coverage.  The OLD equal-scale two-mock SBC could not.
        res = sbc_coverage(
            n_sims=300,
            nbin=6,
            seed=2027,
            n_true_range=(400.0, 3000.0),
            C_range=(0.5, 0.95),
            bfp_range=(15.0, 50.0),  # non-trivial FP scale so the basis matters
            n_mc=2000,
            production_mirror=True,
            build_frac=0.7,
            rebase_bfp=False,  # the BUG: keep BUILD-basis b_FP on the held-out F
        )
        # The mis-scaled FP subtraction collapses coverage well below nominal.
        assert res["coverage68"] < 0.55, f"un-rebased cov68={res['coverage68']}"

    def test_mirror_soft_fractional_regime_calibrated(self):
        # The "soft" regime: fractional (posterior-weighted) f_matched / n_truth
        # rather than integer counts. Coverage must still HOLD — and per the
        # documented modeling choice (the binomial-style Beta/Gamma posteriors
        # ignore the sub-Poisson variance reduction of posterior-weighted soft
        # counts) it is calibrated-to-CONSERVATIVE here: coverage must be at least
        # nominal (never anti-conservative) and not wildly over-wide.
        res = sbc_coverage(
            n_sims=300,
            nbin=6,
            seed=4242,
            n_true_range=(400.0, 3000.0),
            C_range=(0.5, 0.95),
            bfp_range=(0.0, 5.0),
            n_mc=2000,
            production_mirror=True,
            build_frac=0.7,
            rebase_bfp=True,
            soft_fractional=True,
        )
        cov68 = res["coverage68"]
        cov95 = res["coverage95"]
        # conservative band: >= nominal (within MC noise) and bounded above.
        assert 0.66 <= cov68 <= 0.85, f"soft cov68={cov68}"
        assert 0.93 <= cov95 <= 0.995, f"soft cov95={cov95}"


# --------------------------------------------------------------------------- #
# B2  Omega from joint per-draw n_corr (preserve inter-bin correlation)
# --------------------------------------------------------------------------- #
class TestOmegaFromDraws:
    def _ci(self, F, frac=0.3):
        F = np.asarray(F, float)
        w = frac * np.sqrt(np.maximum(F, 1.0))
        return {
            "lo68": np.maximum(F - w, 0.0),
            "hi68": F + w,
            "lo95": np.maximum(F - 2 * w, 0.0),
            "hi95": F + 2 * w,
        }

    def test_return_draws_shape_and_consistency(self):
        F = np.array([50.0, 80.0, 30.0])
        C = estimate_diagonal_completeness(
            np.array([40.0, 60.0, 20.0]), np.array([50.0, 80.0, 30.0])
        )
        bfp = estimate_false_positive_deposit(np.array([1.0, 2.0, 0.0]), 100.0)
        out = apply_diagonal_correction(
            F, self._ci(F), C, bfp, n_mc=3000, return_draws=True
        )
        assert "n_corr_draws" in out
        draws = np.asarray(out["n_corr_draws"])
        assert draws.shape == (3000, 3)
        # The per-draw percentiles must reproduce the returned lo68/hi68 closely
        # (same draws, just reduced inside apply_diagonal_correction).
        v = out["valid_mask"]
        for j in np.where(v)[0]:
            col = draws[:, j]
            col = col[np.isfinite(col)]
            lo, hi = np.percentile(col, [16.0, 84.0])
            np.testing.assert_allclose(lo, out["lo68"][j], rtol=0, atol=1e-9)
            np.testing.assert_allclose(hi, out["hi68"][j], rtol=0, atol=1e-9)

    def test_omega_point_inside_interval(self):
        # Forming Omega per-draw (preserving inter-bin correlation) GUARANTEES
        # the Omega point estimate lies inside its own credible interval.
        logN = np.linspace(20.3, 22.5, 6)
        F = np.array([200.0, 120.0, 80.0, 40.0, 15.0, 5.0])
        n_truth = np.array([260.0, 150.0, 95.0, 50.0, 18.0, 6.0])
        f_matched = np.array([210.0, 118.0, 76.0, 39.0, 14.0, 5.0])
        C = estimate_diagonal_completeness(f_matched, n_truth)
        bfp = estimate_false_positive_deposit(np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]), 100.0)
        out = apply_diagonal_correction(
            F, self._ci(F), C, bfp, n_mc=5000, return_draws=True
        )
        om = omega_from_draws(out["n_corr_draws"], logN, dX=12.3, hubble=0.7)
        for k in ("omega", "lo68", "hi68", "lo95", "hi95"):
            assert k in om, f"missing key {k}"
        assert om["lo95"] <= om["omega"] <= om["hi95"]
        assert om["lo68"] <= om["omega"] <= om["hi68"]
        assert om["lo95"] <= om["lo68"] <= om["hi68"] <= om["hi95"]

    def test_omega_scales_with_corrected_counts(self):
        # Omega = (m_p H0 / c rho_c) * sum N_HI * n_corr / dX scales linearly
        # with the n_corr draws (a basic dimensional sanity check).
        logN = np.linspace(20.3, 22.5, 4)
        draws = np.full((1000, 4), 10.0)
        om1 = omega_from_draws(draws, logN, dX=5.0, hubble=0.7)
        om2 = omega_from_draws(2.0 * draws, logN, dX=5.0, hubble=0.7)
        np.testing.assert_allclose(om2["omega"], 2.0 * om1["omega"], rtol=1e-12)
        # double dX -> half Omega
        om3 = omega_from_draws(draws, logN, dX=10.0, hubble=0.7)
        np.testing.assert_allclose(om3["omega"], 0.5 * om1["omega"], rtol=1e-12)


# --------------------------------------------------------------------------- #
# B3  Exact Beta shapes returned + used (no moment-match drift in skew bins)
# --------------------------------------------------------------------------- #
class TestExactBetaShapes:
    def test_estimator_returns_shape_parameters(self):
        out = estimate_diagonal_completeness(
            np.array([9.0, 5.0]), np.array([10.0, 10.0]), prior=(0.5, 0.5)
        )
        assert "C_alpha" in out and "C_beta" in out
        # Beta(f + a, n - f + b) for the valid bins.
        np.testing.assert_allclose(out["C_alpha"], np.array([9.5, 5.5]), rtol=1e-12)
        np.testing.assert_allclose(out["C_beta"], np.array([1.5, 5.5]), rtol=1e-12)

    def test_exact_shapes_reproduce_estimator_quantiles_skewed_bin(self):
        # Skewed high-completeness bin f=9, n=10: moment-matching from mean+
        # half-width drifts in the tails; sampling the EXACT Beta posterior must
        # reproduce the estimator's own C_lo95 / C_hi95 to MC tolerance.
        f = np.array([9.0])
        n = np.array([10.0])
        C_est = estimate_diagonal_completeness(f, n, prior=(0.5, 0.5))
        F = np.array([100.0])
        F_ci = {  # near-degenerate F so the C posterior dominates the spread
            "lo68": np.array([100.0]), "hi68": np.array([100.0]),
            "lo95": np.array([100.0]), "hi95": np.array([100.0]),
        }
        bfp = estimate_false_positive_deposit(np.array([0.0]), 100.0)
        out = apply_diagonal_correction(F, F_ci, C_est, bfp, n_mc=40000, return_draws=True)
        # n_corr = F / C (b_FP point 0), so C = F / n_corr; recover the sampled C
        # quantiles and compare to the estimator's EXACT Beta quantiles.
        draws = out["n_corr_draws"][:, 0]
        draws = draws[np.isfinite(draws) & (draws > 0)]
        C_samples = F[0] / draws
        s_lo95, s_hi95 = np.percentile(C_samples, [2.5, 97.5])
        # If moment-matching were used, the skewed-bin upper tail would drift;
        # the exact-shape path matches the estimator quantiles within MC noise.
        assert abs(s_lo95 - C_est["C_lo95"][0]) < 0.015, (s_lo95, C_est["C_lo95"][0])
        assert abs(s_hi95 - C_est["C_hi95"][0]) < 0.010, (s_hi95, C_est["C_hi95"][0])


# --------------------------------------------------------------------------- #
# B4  Prior offset: f_unmatched=0 => b_FP POINT == 0 (no phantom subtraction)
# --------------------------------------------------------------------------- #
class TestPriorOffsetNoPhantom:
    def _ci(self, F, frac=0.3):
        F = np.asarray(F, float)
        w = frac * np.sqrt(np.maximum(F, 1.0))
        return {
            "lo68": np.maximum(F - w, 0.0),
            "hi68": F + w,
            "lo95": np.maximum(F - 2 * w, 0.0),
            "hi95": F + 2 * w,
        }

    def test_zero_unmatched_gives_zero_bfp_point(self):
        # A clean high-N bin (no unmatched deposits) must NOT have a phantom
        # half-count subtracted: b_FP point == 0 exactly.
        out = estimate_false_positive_deposit(
            np.array([0.0, 3.0, 0.0]), np.array([100.0, 100.0, 100.0])
        )
        assert out["b_FP"][0] == 0.0
        assert out["b_FP"][2] == 0.0
        # A populated bin keeps a positive deposit point.
        assert out["b_FP"][1] > 0.0

    def test_bfp_ci_still_positive_for_clean_bin(self):
        # The point is 0 but the Gamma posterior still provides a defensible
        # (positive) upper CI for the clean bin (we do not claim certainty of 0).
        out = estimate_false_positive_deposit(np.array([0.0]), 100.0)
        assert out["b_FP"][0] == 0.0
        assert out["b_FP_lo68"][0] >= 0.0
        assert out["b_FP_hi95"][0] > 0.0
        # nested + bracket the (zero) point
        assert out["b_FP_lo95"][0] <= out["b_FP_lo68"][0] <= out["b_FP"][0]
        assert out["b_FP"][0] <= out["b_FP_hi68"][0] <= out["b_FP_hi95"][0]

    def test_clean_bin_not_over_subtracted_in_correction(self):
        # End-to-end: with f_unmatched=0 the correction does NOT subtract a
        # phantom FP from a clean bin, so n_corr == F / C (no neg-clip, no
        # downward bias).
        F = np.array([200.0])
        C = estimate_diagonal_completeness(np.array([180.0]), np.array([200.0]))
        bfp = estimate_false_positive_deposit(np.array([0.0]), 100.0)
        out = apply_diagonal_correction(F, self._ci(F), C, bfp)
        expected = F[0] / C["C"][0]  # no b_FP subtracted
        np.testing.assert_allclose(out["n_corr"][0], expected, rtol=1e-12)
        assert out["neg_clip_mask"][0] == False  # noqa: E712
