"""Tests for the shared CDDF search-window spec (``CDDF_analysis.cddf_forward.window``).

``WindowSpec`` is the SINGLE source of truth for the DLA search window applied
identically to measurement / truth / injection.  The motivating mismatch is in the
proximity *value*, all in the inference's constant ``v/c`` convention:

  * inference (``set_parameters.kms_to_z``) used 3000 km/s → Δz ≈ 0.0100,
  * ``calc_cddf.py`` hard-codes ``proximity_zone = tail_zone = 0.1`` Δz, which in
    the same constant convention is 0.1 * C_KMS ≈ 30000 km/s (so the inline
    "30000 km/s" comment is CORRECT — the cut is ~10× too WIDE, not mislabelled),
  * ``cddf_mock.py`` defaults ``v_prox_kms = 10000``.

``WindowSpec`` fixes them to 3000 km/s in the inference's constant convention
(default ``velocity_scaled=False``).  This M0 milestone delivers ONLY the object +
tests; rewiring calc_cddf's binning is a later milestone.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
# Repo root on sys.path so ``CDDF_analysis`` resolves as a (namespace) package.
sys.path.insert(0, os.path.join(_HERE, ".."))

from CDDF_analysis.cddf_forward.window import WindowSpec  # noqa: E402
from CDDF_analysis.cddf_forward import assert_filter_off  # noqa: E402

_C_KMS = 299792.458


class TestProxDz:
    def test_prox_dz_default_matches_inference_constant_vc(self):
        # Default velocity_scaled=False reproduces set_parameters.kms_to_z:
        # constant Δz = v/c, independent of z_qso (matches stored min/max_z_dlas).
        w = WindowSpec(v_prox_kms=3000.0)
        expected = 3000.0 / _C_KMS  # == kms_to_z(3000) = 3000*1000/c_m_s
        assert w.prox_dz(2.0) == pytest.approx(expected)
        assert w.prox_dz(3.5) == pytest.approx(expected)  # z-independent

    def test_tail_dz_default_matches_inference_constant_vc(self):
        w = WindowSpec(v_tail_kms=3000.0)
        expected = 3000.0 / _C_KMS
        assert w.tail_dz(2.5) == pytest.approx(expected)

    def test_velocity_scaled_uses_one_plus_z(self):
        w = WindowSpec(v_prox_kms=3000.0, v_tail_kms=5000.0, velocity_scaled=True)
        z_qso = 3.0
        assert w.prox_dz(z_qso) == pytest.approx((1.0 + z_qso) * 3000.0 / _C_KMS)
        assert w.tail_dz(z_qso) == pytest.approx((1.0 + z_qso) * 5000.0 / _C_KMS)

    def test_prox_and_tail_independent(self):
        w = WindowSpec(v_prox_kms=3000.0, v_tail_kms=5000.0)
        assert w.prox_dz(3.0) == pytest.approx(3000.0 / _C_KMS)
        assert w.tail_dz(3.0) == pytest.approx(5000.0 / _C_KMS)

    def test_prox_dz_matches_inference_kms_to_z(self):
        # Pin the byte-identity claim: the constant prox_dz must equal the
        # inference's set_parameters.Parameters.kms_to_z so the WindowSpec stays
        # on the exact convention the stored min/max_z_dlas were built with.
        sp = pytest.importorskip("gpy_dla_detection.set_parameters")
        for v in (3000.0, 5000.0, 1500.0):
            assert WindowSpec(v_prox_kms=v).prox_dz(2.7) == pytest.approx(
                sp.Parameters.kms_to_z(v)
            )


class TestDefaults:
    def test_default_velocities_are_3000(self):
        w = WindowSpec()
        assert w.v_prox_kms == 3000.0
        assert w.v_tail_kms == 3000.0

    def test_default_z_min_lyb_is_true(self):
        # Lyα-only for the CDDF: the minimum search edge is shifted to the Lyβ peak.
        assert WindowSpec().z_min_lyb is True

    def test_default_z_max_lyb_is_false(self):
        assert WindowSpec().z_max_lyb is False

    def test_default_lambda_obs_min_is_none(self):
        assert WindowSpec().lambda_obs_min is None


class TestAssertEqual:
    def test_passes_on_equal(self):
        a = WindowSpec()
        b = WindowSpec()
        # should not raise
        WindowSpec.assert_equal(a, b)

    def test_passes_on_equal_custom(self):
        a = WindowSpec(v_prox_kms=5000.0, v_tail_kms=2000.0, z_min_lyb=False,
                       z_max_lyb=True, lambda_obs_min=4000.0)
        b = WindowSpec(v_prox_kms=5000.0, v_tail_kms=2000.0, z_min_lyb=False,
                       z_max_lyb=True, lambda_obs_min=4000.0)
        WindowSpec.assert_equal(a, b)

    def test_raises_on_differing_v_prox(self):
        a = WindowSpec(v_prox_kms=3000.0)
        b = WindowSpec(v_prox_kms=10000.0)
        with pytest.raises(ValueError):
            WindowSpec.assert_equal(a, b)

    def test_raises_on_differing_z_min_lyb(self):
        a = WindowSpec(z_min_lyb=True)
        b = WindowSpec(z_min_lyb=False)
        with pytest.raises(ValueError):
            WindowSpec.assert_equal(a, b)

    def test_raises_on_differing_lambda_obs_min(self):
        a = WindowSpec(lambda_obs_min=None)
        b = WindowSpec(lambda_obs_min=4000.0)
        with pytest.raises(ValueError):
            WindowSpec.assert_equal(a, b)

    def test_passes_on_nan_lambda_obs_min(self):
        # Two identical NaN fields must compare EQUAL (not nan != nan -> raise).
        a = WindowSpec(lambda_obs_min=float("nan"))
        b = WindowSpec(lambda_obs_min=float("nan"))
        WindowSpec.assert_equal(a, b)  # should not raise

    def test_passes_on_float_arithmetic_drift(self):
        # Physically-identical windows built via different float arithmetic
        # (0.1+0.2 vs 0.3) must NOT spuriously mismatch.
        a = WindowSpec(v_prox_kms=0.1 + 0.2)
        b = WindowSpec(v_prox_kms=0.3)
        WindowSpec.assert_equal(a, b)  # should not raise

    def test_raises_on_genuinely_different_floats(self):
        a = WindowSpec(v_prox_kms=3000.0)
        b = WindowSpec(v_prox_kms=3001.0)
        with pytest.raises(ValueError):
            WindowSpec.assert_equal(a, b)

    def test_ctx_appears_in_error_message(self):
        a = WindowSpec(v_prox_kms=3000.0)
        b = WindowSpec(v_prox_kms=10000.0)
        with pytest.raises(ValueError, match="measurement-vs-truth"):
            WindowSpec.assert_equal(a, b, ctx="measurement-vs-truth")


class TestAssertFilterOff:
    """The CDDF is only valid on FILTER-off runs.

    The processed-HDF5 schema does NOT persist the FILTER flag (verified:
    ``process_helpers.save_results_to_hdf5`` / ``DLAHolder.save_results`` write
    only ``pair_prior_mode`` + ``dla_bias`` as root attrs), so the guard takes
    the FILTER setting explicitly rather than reading it from the file.
    """

    def test_passes_when_filter_off(self):
        # FILTER_LOW_LIKELIHOOD = 0 → CDDF valid; should not raise.
        assert_filter_off(0)

    def test_passes_when_filter_off_bool(self):
        assert_filter_off(False)

    def test_raises_when_filter_on(self):
        with pytest.raises(ValueError):
            assert_filter_off(1)

    def test_raises_when_filter_on_bool(self):
        with pytest.raises(ValueError):
            assert_filter_off(True)

    def test_error_mentions_cddf_validity(self):
        with pytest.raises(ValueError, match="FILTER"):
            assert_filter_off(1)

    def test_ctx_appears_in_error_message(self):
        with pytest.raises(ValueError, match="2lpt0-run"):
            assert_filter_off(1, ctx="2lpt0-run")

    def test_raises_when_flag_none(self):
        # An unknown FILTER setting (None) is unsafe -> refuse, not crash.
        with pytest.raises(ValueError, match="unknown"):
            assert_filter_off(None)

    def test_raises_when_flag_nan(self):
        with pytest.raises(ValueError, match="unknown"):
            assert_filter_off(float("nan"))

    def test_truthy_int_two_raises(self):
        with pytest.raises(ValueError):
            assert_filter_off(2)
