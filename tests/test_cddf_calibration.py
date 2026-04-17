"""
tests/test_cddf_calibration.py — Unit tests for CDDF_analysis/cddf_calibration.py

All tests use synthetic numpy arrays; no DESI data or desispec required.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CDDF_analysis"))
import cddf_calibration as cal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_out_truth(z, y, err):
    """Minimal out_truth dict as returned by compute_dndx()."""
    return {"z_mid": z, "dndx": y, "err_boot": err}


# ---------------------------------------------------------------------------
# TestSymErrFromBounds
# ---------------------------------------------------------------------------

class TestSymErrFromBounds:
    def test_symmetric_bounds(self):
        y = np.array([1.0, 2.0])
        bounds = np.array([[0.5, 1.5], [1.0, 3.0]])
        sigma = cal.sym_err_from_bounds(y, bounds)
        np.testing.assert_allclose(sigma, [0.5, 1.0])

    def test_asymmetric_bounds(self):
        y = np.array([3.0])
        bounds = np.array([[1.0, 4.0]])
        sigma = cal.sym_err_from_bounds(y, bounds)
        # 0.5 * ((3-1) + (4-3)) = 0.5 * (2 + 1) = 1.5
        np.testing.assert_allclose(sigma, [1.5])

    def test_shape_preserved(self):
        y = np.ones(5)
        bounds = np.column_stack([0.8 * np.ones(5), 1.2 * np.ones(5)])
        sigma = cal.sym_err_from_bounds(y, bounds)
        assert sigma.shape == (5,)
        np.testing.assert_allclose(sigma, 0.2)


# ---------------------------------------------------------------------------
# TestEvalTruthAtZ
# ---------------------------------------------------------------------------

class TestEvalTruthAtZ:
    def test_exact_interpolation(self):
        z = np.array([2.0, 3.0, 4.0])
        y = np.array([0.1, 0.2, 0.15])
        err = np.array([0.01, 0.02, 0.015])
        out_truth = _make_out_truth(z, y, err)

        y_interp, e_interp = cal.eval_truth_at_z(z, out_truth, y_key="dndx")
        np.testing.assert_allclose(y_interp, y)
        np.testing.assert_allclose(e_interp, err)

    def test_interpolation_midpoint(self):
        z = np.array([2.0, 3.0])
        y = np.array([0.0, 1.0])
        err = np.array([0.0, 0.0])
        out_truth = _make_out_truth(z, y, err)

        y_interp, _ = cal.eval_truth_at_z(np.array([2.5]), out_truth, y_key="dndx")
        np.testing.assert_allclose(y_interp, [0.5])

    def test_missing_z_key_raises(self):
        out_bad = {"z_wrong": [1, 2], "dndx": [1, 2], "err_boot": [0.1, 0.1]}
        with pytest.raises(KeyError):
            cal.eval_truth_at_z(np.array([1.5]), out_bad, y_key="dndx")


# ---------------------------------------------------------------------------
# TestCalibrationFactorAlpha
# ---------------------------------------------------------------------------

class TestCalibrationFactorAlpha:
    def test_alpha_one_when_equal(self):
        z = np.array([2.0, 2.5, 3.0])
        y = np.array([0.1, 0.2, 0.15])
        err = np.array([0.01, 0.02, 0.015])
        out_truth = _make_out_truth(z, y, err)
        bounds68 = np.column_stack([y - err, y + err])

        result = cal.calibration_factor_alpha(
            z, y, bounds68, out_truth, truth_y_key="dndx"
        )
        np.testing.assert_allclose(result["alpha"], 1.0, rtol=1e-5)

    def test_alpha_gt_1_when_mock_below_truth(self):
        z = np.array([2.0, 3.0])
        y_truth = np.array([0.2, 0.3])
        y_meas = np.array([0.1, 0.15])  # mock underestimates
        err = np.array([0.01, 0.01])
        out_truth = _make_out_truth(z, y_truth, err)
        bounds68 = np.column_stack([y_meas - err, y_meas + err])

        result = cal.calibration_factor_alpha(
            z, y_meas, bounds68, out_truth, truth_y_key="dndx"
        )
        assert np.all(result["alpha"] > 1.0)

    def test_output_keys(self):
        z = np.array([2.0, 3.0])
        y = np.array([0.1, 0.2])
        err = np.array([0.01, 0.02])
        out_truth = _make_out_truth(z, y, err)
        bounds68 = np.column_stack([y - err, y + err])

        result = cal.calibration_factor_alpha(
            z, y, bounds68, out_truth, truth_y_key="dndx"
        )
        for key in ("z", "alpha", "alpha_err", "y_meas", "y_true", "y_corr"):
            assert key in result, f"Missing key: {key}"

    def test_alpha_err_positive(self):
        z = np.array([2.0, 3.0, 4.0])
        y = np.array([0.1, 0.2, 0.15])
        err = np.array([0.01, 0.02, 0.015])
        out_truth = _make_out_truth(z, 2 * y, err)
        bounds68 = np.column_stack([y - 0.02, y + 0.02])

        result = cal.calibration_factor_alpha(
            z, y, bounds68, out_truth, truth_y_key="dndx"
        )
        assert np.all(result["alpha_err"] >= 0)

    def test_clip_option(self):
        z = np.array([2.0, 3.0])
        y_truth = np.array([10.0, 10.0])
        y_meas = np.array([1.0, 1.0])
        err = np.array([0.1, 0.1])
        out_truth = _make_out_truth(z, y_truth, err)
        bounds68 = np.column_stack([y_meas - err, y_meas + err])

        result = cal.calibration_factor_alpha(
            z, y_meas, bounds68, out_truth, truth_y_key="dndx", clip=(0.5, 3.0)
        )
        assert np.all(result["alpha"] <= 3.0)
        assert np.all(result["alpha"] >= 0.5)


# ---------------------------------------------------------------------------
# TestApplyAlphaToBounds
# ---------------------------------------------------------------------------

class TestApplyAlphaToBounds:
    def _make_cal(self, z, alpha, alpha_err):
        return {"z": z, "alpha": alpha, "alpha_err": alpha_err}

    def test_alpha_one_passes_through(self):
        z = np.array([2.0, 3.0])
        y = np.array([1.0, 2.0])
        lo68, hi68 = y - 0.1, y + 0.1
        lo95, hi95 = y - 0.2, y + 0.2
        cal_dict = self._make_cal(z, np.ones(2), np.zeros(2))

        result = cal.apply_alpha_to_bounds(
            z, y, lo68, hi68, lo95, hi95, cal_dict,
            include_alpha_uncertainty=False
        )
        np.testing.assert_allclose(result["y_corr"], y, rtol=1e-10)

    def test_output_keys(self):
        z = np.array([2.0, 3.0])
        y = np.array([1.0, 2.0])
        lo = y - 0.1
        hi = y + 0.1
        cal_dict = self._make_cal(z, 2.0 * np.ones(2), 0.1 * np.ones(2))

        result = cal.apply_alpha_to_bounds(z, y, lo, hi, lo, hi, cal_dict)
        for key in ("z_cent", "y_raw", "y_corr",
                    "y68_low_corr", "y68_high_corr",
                    "y95_low_corr", "y95_high_corr"):
            assert key in result, f"Missing key: {key}"

    def test_correction_doubles_y(self):
        z = np.array([2.0, 3.0])
        y = np.array([1.0, 2.0])
        lo, hi = y - 0.1, y + 0.1
        cal_dict = self._make_cal(z, 2.0 * np.ones(2), np.zeros(2))

        result = cal.apply_alpha_to_bounds(
            z, y, lo, hi, lo, hi, cal_dict, include_alpha_uncertainty=False
        )
        np.testing.assert_allclose(result["y_corr"], 2.0 * y)

    def test_shape_output(self):
        z = np.array([2.0, 2.5, 3.0])
        y = np.ones(3)
        lo, hi = 0.9 * np.ones(3), 1.1 * np.ones(3)
        cal_dict = self._make_cal(z, np.ones(3), 0.05 * np.ones(3))

        result = cal.apply_alpha_to_bounds(z, y, lo, hi, lo, hi, cal_dict)
        assert result["y_corr"].shape == (3,)
        assert result["y68_low_corr"].shape == (3,)


# ---------------------------------------------------------------------------
# TestCorrectionRatioWithUncertainty
# ---------------------------------------------------------------------------

class TestCorrectionRatioWithUncertainty:
    def test_ratio_equals_one_when_equal(self):
        f = np.array([1.0, 2.0, 0.5])
        sig = np.array([0.1, 0.2, 0.05])
        r, sig_r = cal.correction_ratio_with_uncertainty(f, sig, f, sig)
        np.testing.assert_allclose(r, 1.0)

    def test_ratio_value(self):
        f_true = np.array([2.0])
        f_meas = np.array([1.0])
        sig = np.array([0.0])
        r, sig_r = cal.correction_ratio_with_uncertainty(f_true, sig, f_meas, sig)
        np.testing.assert_allclose(r, [2.0])

    def test_nan_when_nonpositive(self):
        f_true = np.array([1.0, 0.0, -1.0])
        f_meas = np.array([1.0, 1.0, 1.0])
        sig = np.zeros(3)
        r, _ = cal.correction_ratio_with_uncertainty(f_true, sig, f_meas, sig)
        assert np.isfinite(r[0])
        assert np.isnan(r[1])
        assert np.isnan(r[2])

    def test_nan_when_meas_nonpositive(self):
        f_true = np.array([1.0])
        f_meas = np.array([0.0])
        sig = np.zeros(1)
        r, _ = cal.correction_ratio_with_uncertainty(f_true, sig, f_meas, sig)
        assert np.isnan(r[0])

    def test_error_propagation(self):
        # r = 1 when f_true == f_meas; fractional error = sqrt(2) * (sig/f)
        f = np.array([2.0])
        sig = np.array([0.2])
        r, sig_r = cal.correction_ratio_with_uncertainty(f, sig, f, sig)
        expected_err = 1.0 * np.sqrt(2) * (0.2 / 2.0)
        np.testing.assert_allclose(sig_r, [expected_err], rtol=1e-6)


# ---------------------------------------------------------------------------
# TestApplyCorrectionWithUncertainty
# ---------------------------------------------------------------------------

class TestApplyCorrectionWithUncertainty:
    def test_f_corr_equals_r_times_f(self):
        f = np.array([1.0, 2.0])
        sig = np.array([0.1, 0.2])
        r = np.array([2.0, 0.5])
        sig_r = np.zeros(2)

        f_corr, _, _ = cal.apply_correction_with_uncertainty(f, sig, r, sig_r)
        np.testing.assert_allclose(f_corr, [2.0, 1.0])

    def test_lower_bound_clipped_at_zero(self):
        f = np.array([1.0])
        sig = np.array([2.0])   # large uncertainty → lower bound would be negative
        r = np.array([1.0])
        sig_r = np.zeros(1)

        _, _, interval = cal.apply_correction_with_uncertainty(f, sig, r, sig_r)
        assert interval[0, 0] >= 0.0

    def test_nan_propagates(self):
        f = np.array([1.0])
        sig = np.array([0.1])
        r = np.array([np.nan])
        sig_r = np.array([0.0])

        f_corr, _, _ = cal.apply_correction_with_uncertainty(f, sig, r, sig_r)
        assert np.isnan(f_corr[0])

    def test_nsig_scales_interval(self):
        f = np.array([1.0])
        sig = np.array([0.1])
        r = np.array([1.0])
        sig_r = np.zeros(1)

        _, sig_c1, int1 = cal.apply_correction_with_uncertainty(f, sig, r, sig_r, nsig=1.0)
        _, sig_c2, int2 = cal.apply_correction_with_uncertainty(f, sig, r, sig_r, nsig=2.0)
        # 2σ interval should be twice as wide
        w1 = int1[0, 1] - max(int1[0, 0], 0.0)
        w2 = int2[0, 1] - max(int2[0, 0], 0.0)
        np.testing.assert_allclose(w2, 2 * w1, rtol=1e-6)

    def test_output_shapes(self):
        n = 5
        f = np.ones(n)
        sig = 0.1 * np.ones(n)
        r = np.ones(n)
        sig_r = np.zeros(n)

        f_corr, sig_corr, interval = cal.apply_correction_with_uncertainty(f, sig, r, sig_r)
        assert f_corr.shape == (n,)
        assert sig_corr.shape == (n,)
        assert interval.shape == (n, 2)


# ---------------------------------------------------------------------------
# TestBoundsToAsymSigma
# ---------------------------------------------------------------------------

class TestBoundsToAsymSigma:
    def test_symmetric_case(self):
        y = np.array([2.0])
        sm, sp = cal.bounds_to_asym_sigma(y, np.array([1.0]), np.array([3.0]))
        np.testing.assert_allclose(sm, [1.0])
        np.testing.assert_allclose(sp, [1.0])

    def test_asymmetric_case(self):
        y = np.array([3.0])
        sm, sp = cal.bounds_to_asym_sigma(y, np.array([1.0]), np.array([4.0]))
        np.testing.assert_allclose(sm, [2.0])
        np.testing.assert_allclose(sp, [1.0])


# ---------------------------------------------------------------------------
# TestDndxToEllz (in cddf_mock)
# ---------------------------------------------------------------------------

class TestDndxToEllz:
    """Test the dN/dX → dN/dz conversion functions added to cddf_mock."""

    def setup_method(self):
        # Import here so that failures give clearer error messages
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CDDF_analysis"))
        from cddf_mock import dndx_to_ellz, dndx_bounds_to_ellz
        self.dndx_to_ellz = dndx_to_ellz
        self.dndx_bounds_to_ellz = dndx_bounds_to_ellz

    def test_formula_spot_check(self):
        # At z=0, E(z)=1, dX/dz = 1; so dN/dz = dN/dX
        z = np.array([0.0])
        dndx = np.array([0.5])
        ellz = self.dndx_to_ellz(z, dndx)
        np.testing.assert_allclose(ellz, [0.5], rtol=1e-6)

    def test_dX_dz_positive(self):
        z = np.array([2.0, 3.0, 4.0])
        dndx = np.array([0.1, 0.2, 0.15])
        ellz = self.dndx_to_ellz(z, dndx)
        assert np.all(ellz > 0)
        # dN/dz > dN/dX because dX/dz > 1 at z > 0
        assert np.all(ellz > dndx)

    def test_bounds_shape(self):
        z = np.array([2.0, 3.0])
        bounds = np.column_stack([[0.08, 0.18], [0.12, 0.22]])
        ell_bounds = self.dndx_bounds_to_ellz(z, bounds)
        assert ell_bounds.shape == (2, 2)

    def test_bounds_monotone(self):
        z = np.array([2.0, 3.0])
        bounds = np.column_stack([[0.08, 0.18], [0.12, 0.22]])
        ell_bounds = self.dndx_bounds_to_ellz(z, bounds)
        # lower < upper after conversion
        assert np.all(ell_bounds[:, 0] < ell_bounds[:, 1])
