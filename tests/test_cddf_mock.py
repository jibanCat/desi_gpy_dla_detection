"""
tests/test_cddf_mock.py — Smoke and unit tests for CDDF_analysis/cddf_mock.py.

These tests use only synthetic data (no DESI spectra, no desispec required).
They verify:
  - AbsorptionDistance: cosmological path length math
  - truth_cddf_prochaska2014 / truth_dndx_prochaska2014: spline evaluation
  - zbins_from_zmid_uniform: bin-edge reconstruction
  - build_qso_windows / compute_dndx: search window and dN/dX pipeline
  - compute_calibration_alpha / apply_calibration: calibration arithmetic

Run with:
    pytest tests/test_cddf_mock.py -v
"""
import sys
import os

import numpy as np
import pytest
from astropy.table import Table

# Make CDDF_analysis importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CDDF_analysis"))
import cddf_mock as cm


# ------------------------------------------------------------------ #
# 1. AbsorptionDistance
# ------------------------------------------------------------------ #

class TestAbsorptionDistance:
    """Tests for the comoving absorption distance helper."""

    def setup_method(self):
        self.Xcalc = cm.AbsorptionDistance(zmax=5.0, Omega_m=0.279)

    def test_X_at_zero_is_zero(self):
        assert self.Xcalc.X(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_X_monotone_increasing(self):
        zs = np.linspace(0.0, 4.0, 50)
        Xs = self.Xcalc.X(zs)
        assert np.all(np.diff(Xs) > 0)

    def test_deltaX_positive(self):
        assert self.Xcalc.deltaX(2.0, 3.0) > 0

    def test_deltaX_additive(self):
        # X(z1→z3) = X(z1→z2) + X(z2→z3)
        dX_13 = self.Xcalc.deltaX(2.0, 4.0)
        dX_12 = self.Xcalc.deltaX(2.0, 3.0)
        dX_23 = self.Xcalc.deltaX(3.0, 4.0)
        assert dX_12 + dX_23 == pytest.approx(dX_13, rel=1e-6)

    def test_deltaX_asymmetric(self):
        # deltaX(z1, z2) = -deltaX(z2, z1)
        assert self.Xcalc.deltaX(2.0, 3.0) == pytest.approx(
            -self.Xcalc.deltaX(3.0, 2.0), rel=1e-6
        )

    def test_X_at_z2_known_order_of_magnitude(self):
        # For WMAP9 at z=2, dX/dz ≈ (1+z)^2/E(z) ≈ 9/1.73 ≈ 5.2
        # X(z=2) should be roughly in the range [3, 8]
        X2 = self.Xcalc.X(2.0)
        assert 3.0 < X2 < 8.0


# ------------------------------------------------------------------ #
# 2. Prochaska+2014 truth CDDF spline
# ------------------------------------------------------------------ #

class TestTruthCddf:
    """Tests for truth_cddf_prochaska2014 and truth_dndx_prochaska2014."""

    def test_spline_returns_array(self):
        logN = np.array([17.0, 20.0, 21.5])
        result = cm.truth_cddf_prochaska2014(logN)
        assert result.shape == (3,)

    def test_spline_at_node_values(self):
        # At the hardcoded nodes the spline must pass through the values.
        logN_nodes = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
        logf_nodes = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
        result = cm.truth_cddf_prochaska2014(logN_nodes)
        np.testing.assert_allclose(result, logf_nodes, atol=1e-3)

    def test_spline_monotone_decreasing(self):
        # f(N) must decrease as N increases (CDDF property)
        logN = np.linspace(13.0, 21.5, 100)
        logf = cm.truth_cddf_prochaska2014(logN)
        assert np.all(np.diff(logf) < 0)

    def test_spline_clip_outside_range(self):
        # Values outside [12, 22] are clipped to boundary; no NaN
        result = cm.truth_cddf_prochaska2014([5.0, 25.0])
        assert np.all(np.isfinite(result))

    def test_dndx_dla_range_positive_scalar(self):
        # DLA range [20.3, 23): must be a positive float
        val = cm.truth_dndx_prochaska2014(20.3, 23.0)
        assert isinstance(val, float)
        assert val > 0

    def test_dndx_lls_range_positive_scalar(self):
        # LLS range [17.2, 19) must also be positive and larger than DLA
        val_lls = cm.truth_dndx_prochaska2014(17.2, 19.0)
        val_dla = cm.truth_dndx_prochaska2014(20.3, 23.0)
        assert val_lls > 0
        assert val_lls > val_dla  # LLS are more common

    def test_dndx_increases_with_range(self):
        # Integrating more of the CDDF (wider logN range) → larger dN/dX
        val_narrow = cm.truth_dndx_prochaska2014(20.3, 21.0)
        val_wide = cm.truth_dndx_prochaska2014(20.3, 23.0)
        assert val_wide > val_narrow


# ------------------------------------------------------------------ #
# 3. zbins_from_zmid_uniform
# ------------------------------------------------------------------ #

class TestZbinsFromZmid:
    """Tests for zbins_from_zmid_uniform bin-edge reconstruction."""

    def test_roundtrip(self):
        # Midpoints of reconstructed edges should equal the input centers
        z_mid = np.array([2.25, 2.75, 3.25, 3.75])
        zbins = cm.zbins_from_zmid_uniform(z_mid)
        z_mid_recovered = 0.5 * (zbins[:-1] + zbins[1:])
        np.testing.assert_allclose(z_mid_recovered, z_mid, atol=1e-10)

    def test_output_length(self):
        z_mid = np.array([2.0, 2.5, 3.0, 3.5])
        zbins = cm.zbins_from_zmid_uniform(z_mid)
        assert len(zbins) == len(z_mid) + 1

    def test_correct_half_width(self):
        z_mid = np.array([2.0, 2.5, 3.0])
        zbins = cm.zbins_from_zmid_uniform(z_mid)
        assert zbins[0] == pytest.approx(2.0 - 0.25)
        assert zbins[-1] == pytest.approx(3.0 + 0.25)

    def test_raises_non_uniform(self):
        z_mid = np.array([2.0, 2.5, 3.1])  # not uniform
        with pytest.raises(ValueError, match="uniformly spaced"):
            cm.zbins_from_zmid_uniform(z_mid)


# ------------------------------------------------------------------ #
# 4. build_qso_windows
# ------------------------------------------------------------------ #

class TestBuildQsoWindows:
    """Tests for build_qso_windows search-window construction."""

    def _make_qso_cat(self, z_vals):
        return Table({"TARGETID": np.arange(len(z_vals), dtype=int), "Z": np.array(z_vals)})

    def test_z_lo_less_than_z_hi(self):
        qso_cat = self._make_qso_cat([3.0, 3.5, 4.0])
        tid, zlo, zhi = cm.build_qso_windows(qso_cat, zmin=2.0, v_prox_kms=3000.0)
        assert np.all(zlo < zhi)

    def test_z_hi_below_qso_redshift(self):
        # z_hi must be less than z_qso (proximity cut)
        # Note: build_qso_windows only returns windows where z_hi > z_lo (ok mask)
        qso_cat = self._make_qso_cat([3.0, 3.5])
        tid, zlo, zhi = cm.build_qso_windows(qso_cat, zmin=2.0, v_prox_kms=3000.0)
        z_lookup = dict(zip([3.0, 3.5], [3.0, 3.5]))
        for t, zh in zip(tid, zhi):
            # z_hi should be strictly below the corresponding z_qso
            z_qso_val = qso_cat["Z"][qso_cat["TARGETID"] == t][0]
            assert zh < z_qso_val

    def test_lambda_obs_min_raises_zlo(self):
        # With a large lambda_obs_min the blue edge should be higher
        qso_cat = self._make_qso_cat([3.5])
        _, zlo_no_cut, _ = cm.build_qso_windows(qso_cat, zmin=2.0, v_prox_kms=3000.0,
                                                 lambda_obs_min=None)
        _, zlo_with_cut, _ = cm.build_qso_windows(qso_cat, zmin=2.0, v_prox_kms=3000.0,
                                                   lambda_obs_min=4000.0)
        assert zlo_with_cut[0] >= zlo_no_cut[0]

    def test_zmin_floor(self):
        qso_cat = self._make_qso_cat([3.0, 3.5])
        zmin = 2.5
        _, zlo, _ = cm.build_qso_windows(qso_cat, zmin=zmin, v_prox_kms=3000.0)
        assert np.all(zlo >= zmin)


# ------------------------------------------------------------------ #
# 5. compute_dndx — smoke and property tests with synthetic catalogs
# ------------------------------------------------------------------ #

class TestComputeDndx:
    """Smoke tests for compute_dndx using synthetic catalogs."""

    def _make_catalogs(self, n_qso=100, n_dla=20, z_qso_mean=3.0,
                       z_dla_mean=2.5, seed=42):
        rng = np.random.default_rng(seed)
        z_qso = rng.uniform(2.5, 4.0, n_qso)
        targetids = np.arange(n_qso, dtype=int)

        qso_cat = Table({"TARGETID": targetids, "Z": z_qso})

        # Place absorbers at random QSO sightlines with random z_DLA < z_QSO
        qso_idx = rng.choice(n_qso, size=n_dla, replace=True)
        z_dla = rng.uniform(2.0, 2.8, n_dla)
        nhi = rng.uniform(20.3, 21.5, n_dla)

        abs_cat = Table({
            "TARGETID": targetids[qso_idx],
            "Z_DLA": z_dla,
            "NHI": nhi,
        })
        return abs_cat, qso_cat

    _ZBINS = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    _DNDX_KW = dict(zmin=2.0, v_prox_kms=3000.0, logNHImin=20.3, logNHImax=23.0)

    def test_returns_expected_keys(self):
        abs_cat, qso_cat = self._make_catalogs()
        out = cm.compute_dndx(abs_cat, qso_cat, zbins=self._ZBINS, **self._DNDX_KW)
        for key in ("z_mid", "zbins", "dndx", "err_poisson", "N_abs", "X_tot"):
            assert key in out, f"Key '{key}' missing from output"

    def test_output_shapes(self):
        abs_cat, qso_cat = self._make_catalogs()
        out = cm.compute_dndx(abs_cat, qso_cat, zbins=self._ZBINS, **self._DNDX_KW)
        n = len(self._ZBINS) - 1
        assert out["z_mid"].shape == (n,)
        assert out["dndx"].shape == (n,)
        assert out["N_abs"].shape == (n,)
        assert out["X_tot"].shape == (n,)

    def test_dndx_nonnegative(self):
        abs_cat, qso_cat = self._make_catalogs()
        out = cm.compute_dndx(abs_cat, qso_cat, zbins=self._ZBINS, **self._DNDX_KW)
        assert np.all(out["dndx"][out["X_tot"] > 0] >= 0)

    def test_empty_absorber_catalog_gives_zero_dndx(self):
        _, qso_cat = self._make_catalogs()
        empty_cat = Table({"TARGETID": np.array([], dtype=int),
                           "Z_DLA": np.array([]),
                           "NHI": np.array([])})
        zbins = np.array([2.0, 2.5, 3.0, 3.5])
        out = cm.compute_dndx(empty_cat, qso_cat, zbins=zbins, **self._DNDX_KW)
        assert np.all(out["N_abs"] == 0)
        assert np.all(out["dndx"] == 0.0)

    def test_N_abs_matches_manual_count(self):
        # Put exactly 5 absorbers in z=[2.0, 2.5) and none elsewhere
        rng = np.random.default_rng(99)
        z_qso = rng.uniform(3.0, 4.0, 50)
        qso_cat = Table({"TARGETID": np.arange(50, dtype=int), "Z": z_qso})
        abs_cat = Table({
            "TARGETID": np.arange(5, dtype=int),
            "Z_DLA": np.array([2.1, 2.2, 2.3, 2.4, 2.45]),
            "NHI": np.full(5, 20.5),
        })
        zbins = np.array([2.0, 2.5, 3.0])
        out = cm.compute_dndx(abs_cat, qso_cat, zbins=zbins, **self._DNDX_KW)
        assert out["N_abs"][0] == 5
        assert out["N_abs"][1] == 0

    def test_x_tot_positive_where_qsos_exist(self):
        abs_cat, qso_cat = self._make_catalogs()
        zbins = np.array([2.5, 3.0, 3.5])
        out = cm.compute_dndx(abs_cat, qso_cat, zbins=zbins,
                              zmin=2.0, v_prox_kms=3000.0,
                              logNHImin=20.3, logNHImax=23.0)
        assert np.all(out["X_tot"] > 0)


# ------------------------------------------------------------------ #
# 6. compute_calibration_alpha / apply_calibration
# ------------------------------------------------------------------ #

class TestCalibration:
    """Tests for calibration alpha computation and application."""

    def _make_dndx_out(self, z_mid, dndx, err=None):
        if err is None:
            err = dndx * 0.1
        return {
            "z_mid": np.asarray(z_mid),
            "dndx": np.asarray(dndx),
            "err_poisson": np.asarray(err),
            "err_boot": None,
        }

    def test_alpha_one_when_mock_equals_truth(self):
        z_mid = np.array([2.25, 2.75, 3.25])
        dndx = np.array([0.05, 0.06, 0.04])
        out_truth = self._make_dndx_out(z_mid, dndx)
        out_mock = self._make_dndx_out(z_mid, dndx)
        cal = cm.compute_calibration_alpha(out_truth, out_mock)
        np.testing.assert_allclose(cal["alpha"], 1.0, atol=1e-10)

    def test_alpha_value(self):
        # alpha = truth / measured: a 50%-complete mock yields a 2x up-correction.
        z_mid = np.array([2.25, 2.75])
        truth = np.array([0.10, 0.08])
        mock = np.array([0.05, 0.04])  # 50% completeness
        out_truth = self._make_dndx_out(z_mid, truth)
        out_mock = self._make_dndx_out(z_mid, mock)
        cal = cm.compute_calibration_alpha(out_truth, out_mock)
        np.testing.assert_allclose(cal["alpha"], 2.0, atol=1e-10)

    def test_alpha_nan_where_measured_zero(self):
        # With alpha = truth / measured, division-by-zero (NaN) occurs where
        # the MEASURED mock dN/dX is zero, not where truth is zero.
        z_mid = np.array([2.25, 2.75])
        truth = np.array([0.05, 0.08])
        mock = np.array([0.0, 0.04])
        out_truth = self._make_dndx_out(z_mid, truth)
        out_mock = self._make_dndx_out(z_mid, mock)
        cal = cm.compute_calibration_alpha(out_truth, out_mock)
        assert np.isnan(cal["alpha"][0])
        assert np.isfinite(cal["alpha"][1])

    def test_alpha_err_nan_in_truth_empty_bin(self):
        # truth==0 (alpha collapses to 0) is a degenerate, pure-false-positive bin:
        # alpha_err must be NaN (flagged), not 0 (which would hide the real noise).
        z_mid = np.array([2.25, 2.75])
        truth = np.array([0.0, 0.08])
        mock = np.array([0.05, 0.04])
        out_truth = self._make_dndx_out(z_mid, truth)
        out_mock = self._make_dndx_out(z_mid, mock)
        cal = cm.compute_calibration_alpha(out_truth, out_mock)
        assert cal["alpha"][0] == 0.0
        assert np.isnan(cal["alpha_err"][0])
        assert np.isfinite(cal["alpha_err"][1])

    def test_apply_calibration_alpha_one_identity(self):
        # When alpha=1 and alpha_err=0, calibrated == raw
        z_mid = np.array([2.25, 2.75, 3.25])
        dndx = np.array([0.05, 0.06, 0.04])
        out_real = self._make_dndx_out(z_mid, dndx)
        out_cal = {"z": z_mid, "alpha": np.ones(3), "alpha_err": np.zeros(3)}
        result = cm.apply_calibration(out_real, out_cal)
        np.testing.assert_allclose(result["dndx_calibrated"], dndx, atol=1e-14)

    def test_apply_calibration_scales_correctly(self):
        z_mid = np.array([2.25, 2.75])
        dndx_raw = np.array([0.10, 0.08])
        out_real = self._make_dndx_out(z_mid, dndx_raw, err=np.array([0.01, 0.01]))
        out_cal = {"z": z_mid, "alpha": np.array([2.0, 3.0]), "alpha_err": np.zeros(2)}
        result = cm.apply_calibration(out_real, out_cal)
        np.testing.assert_allclose(result["dndx_calibrated"],
                                   np.array([0.20, 0.24]), atol=1e-12)

    def test_apply_calibration_error_propagation(self):
        # err_cal = sqrt((alpha*err_real)^2 + (dndx_real*err_alpha)^2)
        z_mid = np.array([2.5])
        dndx_raw = np.array([0.10])
        err_raw = np.array([0.02])
        alpha = np.array([2.0])
        alpha_err = np.array([0.1])
        out_real = self._make_dndx_out(z_mid, dndx_raw, err_raw)
        out_cal = {"z": z_mid, "alpha": alpha, "alpha_err": alpha_err}
        result = cm.apply_calibration(out_real, out_cal)
        expected_err = np.sqrt((alpha * err_raw) ** 2 + (dndx_raw * alpha_err) ** 2)
        np.testing.assert_allclose(result["err_calibrated"], expected_err, atol=1e-14)

    def test_apply_calibration_output_keys(self):
        z_mid = np.array([2.25, 2.75])
        out_real = self._make_dndx_out(z_mid, np.array([0.05, 0.06]))
        out_cal = {"z": z_mid, "alpha": np.ones(2), "alpha_err": np.zeros(2)}
        result = cm.apply_calibration(out_real, out_cal)
        for key in ("z", "dndx_raw", "dndx_calibrated", "err_raw", "err_calibrated",
                    "alpha", "alpha_err"):
            assert key in result

    def test_apply_calibration_recovers_truth_on_incomplete_mock(self):
        # End-to-end direction check: a mock measured at 50% of truth must,
        # after calibration, recover truth (a factor-2 UP-correction).
        # alpha = truth / measured = 2.0; dndx_calibrated = alpha * real.
        z_mid = np.array([2.25, 2.75, 3.25])
        truth = np.array([0.10, 0.08, 0.06])
        measured = 0.5 * truth  # GP under-counts (50% completeness)
        out_truth = self._make_dndx_out(z_mid, truth)
        out_measured_mock = self._make_dndx_out(z_mid, measured)

        cal = cm.compute_calibration_alpha(out_truth, out_measured_mock)
        # Calibration multiplier must up-correct the under-counted mock.
        np.testing.assert_allclose(cal["alpha"], 2.0, atol=1e-10)

        # Apply to "real" data that has the same incompleteness as the mock.
        out_real = self._make_dndx_out(z_mid, measured)
        result = cm.apply_calibration(out_real, cal)
        np.testing.assert_allclose(result["dndx_calibrated"], truth, atol=1e-12)
