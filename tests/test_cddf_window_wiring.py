"""Tests for wiring the shared :class:`WindowSpec` into the CDDF estimators.

``CDDF_analysis/cddf_forward/window.py::WindowSpec`` is the single source of truth
for the DLA search window.  These tests pin the BACKWARD-COMPATIBLE wiring of that
spec into the two CDDF pathways:

  * Pathway A — ``calc_cddf.DLACatalogue`` (HDF5 model posteriors).
  * Pathway B — ``cddf_mock.build_qso_windows`` / ``compute_dndx`` (FITS catalog).

The HARD requirement: ``window=None`` (and the existing default) must reproduce
today's numbers byte-for-byte.  Only when a ``WindowSpec`` is supplied does the new
constant-``v/c`` window take effect.

Convention reminder: the user-approved default is ``velocity_scaled=False`` — a
CONSTANT Δz = ``v_kms / C_KMS`` (matches the inference's stored search edges), NOT
the ``(1+z)``-scaled form.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
# Repo root on sys.path so ``CDDF_analysis`` resolves as a (namespace) package —
# calc_cddf uses ``from .set_parameters import *`` so it must import as part of the
# package (matching the production CLI's ``from CDDF_analysis.calc_cddf import ...``).
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))

pytest.importorskip("h5py")
pytest.importorskip("astropy")

from CDDF_analysis import calc_cddf  # noqa: E402
from CDDF_analysis import cddf_mock  # noqa: E402
from CDDF_analysis.cddf_forward.window import WindowSpec, C_KMS  # noqa: E402
from gpy_dla_detection.set_parameters import Parameters  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402


@pytest.fixture
def synth(tmp_path):
    return build_synthetic_cddf(tmp_path)


def _make_catalogue(synth, **overrides):
    kwargs = dict(
        processed_file=synth["processed_file"],
        sample_file=synth["sample_file"],
        catalog_file=synth["catalog_file"],
        sub_dla=False,
        snr=-2,
        lowzcut=False,
        highzcut=False,
    )
    kwargs.update(overrides)
    return calc_cddf.DLACatalogue(**kwargs)


# ----------------------------------------------------------------------------
# 1) Backward-compat: window=None reproduces today's estimator EXACTLY.
# ----------------------------------------------------------------------------
class TestBackwardCompatNoWindow:
    def test_default_construction_keeps_constant_zones(self, synth):
        # No window kwarg at all (the historical signature).
        cat = _make_catalogue(synth)
        assert cat.proximity_zone == 0.1
        assert cat.tail_zone == 0.1
        # The window attribute exists but is None when not supplied.
        assert getattr(cat, "window", None) is None

    def test_explicit_window_none_keeps_constant_zones(self, synth):
        cat = _make_catalogue(synth, window=None)
        assert cat.proximity_zone == 0.1
        assert cat.tail_zone == 0.1
        assert cat.window is None

    def test_window_none_does_not_force_zcuts(self, synth):
        # With window=None the lowzcut/highzcut flags are left as the caller set them.
        cat = _make_catalogue(synth, window=None, lowzcut=False, highzcut=False)
        assert cat.lowzcut is False
        assert cat.highzcut is False


# ----------------------------------------------------------------------------
# 2) Wired window: the WindowSpec must NOT re-apply proximity (the stored
#    min/max_z_dlas ALREADY encode the inference proximity cut) and must NOT
#    force lowzcut/highzcut (that double-cuts AND trips the lyb-branch asserts).
#    Its only measurement-side effect is the Lyα-only / Lyman-limit edge.
# ----------------------------------------------------------------------------
class TestWiredWindowCalcCddf:
    def test_window_does_not_change_proximity_zone(self, synth):
        # The stored edges already carry proximity, so re-cutting would DOUBLE it.
        win = WindowSpec(v_prox_kms=3000.0)
        cat = _make_catalogue(synth, window=win)
        assert cat.proximity_zone == 0.1  # unchanged — no re-cut
        assert cat.tail_zone == 0.1

    def test_window_does_not_force_zcuts(self, synth):
        win = WindowSpec(v_prox_kms=3000.0, z_min_lyb=True)
        cat = _make_catalogue(synth, window=win, lowzcut=False, highzcut=False)
        assert cat.lowzcut is False
        assert cat.highzcut is False

    def test_window_sets_lyb_flags_and_records(self, synth):
        win = WindowSpec(v_prox_kms=3000.0, z_min_lyb=True, z_max_lyb=False)
        cat = _make_catalogue(synth, window=win)
        assert cat.z_min_lyb is True
        assert cat.z_max_lyb is False
        assert cat.window is win

    def test_velocity_scaled_window_raises(self, synth):
        with pytest.raises(NotImplementedError):
            _make_catalogue(synth, window=WindowSpec(velocity_scaled=True))

    def test_estimator_runs_with_window_lya_only(self, synth):
        # Regression for the crash: forcing lowzcut/highzcut tripped
        # `assert highzcut == False` inside the z_min_lyb branch. The corrected
        # wiring lets the CDDF computation actually run.
        win = WindowSpec(v_prox_kms=3000.0, z_min_lyb=True)
        cat = _make_catalogue(synth, window=win)
        l_Ncent, cddf, c68, c95, xerrs = cat.column_density_function(
            z_min=2.0, z_max=3.4, lnhi_nbins=8, lnhi_min=20.3, lnhi_max=22.0
        )
        assert np.all(np.isfinite(cddf))
        assert np.all(cddf >= 0)

    def test_z_min_lyb_actually_binds(self, synth):
        # z_min_lyb floors the blue edge at lymanbeta(z_qso) (> the synth min_z_dla),
        # so the searched path SHRINKS vs z_min_lyb off (it was a no-op before).
        cat_off = _make_catalogue(synth)
        cat_on = _make_catalogue(synth, window=WindowSpec(v_prox_kms=3000.0, z_min_lyb=True))
        assert cat_on.path_length(2.0, 3.4) < cat_off.path_length(2.0, 3.4)


# ----------------------------------------------------------------------------
# 3) Lyβ rest wavelength unified to set_parameters (not the legacy 1026.72).
# ----------------------------------------------------------------------------
class TestUnifiedLymanBeta:
    def test_lymanbeta_uses_set_parameters_constant(self, synth):
        cat = _make_catalogue(synth)
        zqso = 3.5
        # Unified value: set_parameters.Parameters.lyb_wavelength (1025.7223),
        # matching cddf_mock.LYB_REST / the inference — NOT the legacy 1026.72.
        expected = (1 + zqso) * (Parameters.lyb_wavelength / Parameters.lya_wavelength) - 1
        assert cat.lymanbeta(zqso) == pytest.approx(expected)

    def test_lymanbeta_not_legacy_1026(self, synth):
        cat = _make_catalogue(synth)
        zqso = 3.5
        legacy = (1 + zqso) * (1026.72 / 1215.67) - 1
        # The change shifts the Lyβ edge by ~0.0037 in z at z_qso=3.5 — assert it moved.
        assert abs(cat.lymanbeta(zqso) - legacy) > 1e-3


# ----------------------------------------------------------------------------
# 4) Matched window: the SAME WindowSpec drives BOTH pathways to identical edges.
# ----------------------------------------------------------------------------
class TestMatchedWindow:
    def test_cddf_mock_default_unchanged(self):
        # Backward-compat: build_qso_windows with no window uses the 10000 default
        # and the (1+z)-scaled proximity (zmax_nonprox).
        from astropy.table import Table

        zq = np.array([3.0, 3.5, 4.0])
        qso = Table({"TARGETID": np.arange(len(zq)), "Z": zq})
        tid, zlo, zhi = cddf_mock.build_qso_windows(
            qso, zmin=2.0, blue_limit_mode="global"
        )
        # default v_prox_kms=10000, (1+z)-scaled
        expected_hi = zq - (1.0 + zq) * (10000.0 / cddf_mock.C_KMS)
        np.testing.assert_allclose(zhi, expected_hi)

    def test_cddf_mock_window_constant_vc_red_edge(self):
        from astropy.table import Table

        win = WindowSpec(v_prox_kms=3000.0)
        zq = np.array([3.0, 3.5, 4.0])
        qso = Table({"TARGETID": np.arange(len(zq)), "Z": zq})
        tid, zlo, zhi = cddf_mock.build_qso_windows(
            qso, zmin=2.0, blue_limit_mode="global", window=win
        )
        # With the window the proximity is the constant v/c Δz (velocity_scaled=False).
        expected_hi = np.array([z - win.prox_dz(z) for z in zq])
        np.testing.assert_allclose(zhi, expected_hi)

    def test_cddf_mock_reproduces_inference_red_edge(self):
        """cddf_mock's red edge from the spec == the inference's stored max_z_dla.

        The matched window is the inference's stored edge (z_qso - kms_to_z(3000));
        calc_cddf USES that stored edge (no re-cut) and cddf_mock REPRODUCES it.
        """
        from astropy.table import Table

        win = WindowSpec(v_prox_kms=3000.0)
        zq = np.array([3.0, 3.5, 4.0])
        qso = Table({"TARGETID": np.arange(len(zq)), "Z": zq})
        _, _, zhi = cddf_mock.build_qso_windows(
            qso, zmin=2.0, blue_limit_mode="global", window=win
        )
        inference_edge = zq - Parameters.kms_to_z(3000.0)
        np.testing.assert_allclose(zhi, inference_edge)

    def test_lya_only_blue_edge_matches_across_pathways(self, synth):
        """With z_min_lyb, cddf_mock's blue edge == calc_cddf's lymanbeta(z_qso)."""
        from astropy.table import Table

        win = WindowSpec(v_prox_kms=3000.0, z_min_lyb=True)
        zq = np.array([3.0, 3.5])
        qso = Table({"TARGETID": np.arange(len(zq)), "Z": zq})
        _, zlo, _ = cddf_mock.build_qso_windows(
            qso, zmin=2.0, blue_limit_mode="lyb", window=win
        )
        cat = _make_catalogue(synth, window=win)
        calc_blue = np.array([cat.lymanbeta(z) for z in zq])  # floor used by z_min_lyb
        np.testing.assert_allclose(zlo, calc_blue, rtol=1e-6)

    def test_compute_dndx_window_overrides_vprox(self, synth):
        """compute_dndx threads the window's v_prox_kms (3000) into the meta."""
        from astropy.table import Table

        win = WindowSpec(v_prox_kms=3000.0)
        zq = np.array([3.0, 3.5, 4.0])
        qso = Table({"TARGETID": np.arange(len(zq)), "Z": zq})
        # Empty absorber catalog (no detections) — we only check meta/window plumbing.
        abs_cat = Table(
            {
                "TARGETID": np.array([], dtype=int),
                "Z_DLA": np.array([], dtype=float),
                "NHI": np.array([], dtype=float),
            }
        )
        zbins = np.array([2.0, 2.5, 3.0])
        out = cddf_mock.compute_dndx(
            abs_cat, qso, zbins=zbins, zmin=2.0, blue_limit_mode="global", window=win
        )
        assert out["meta"]["v_prox_kms"] == pytest.approx(3000.0)


def test_windowspec_disables_both_proximity_and_tail_cuts(tmp_path):
    """Production Lyα-only window (z_min_lyb=True, z_max_lyb=False) must disable
    BOTH lowzcut (proximity) AND highzcut (tail) — the stored min/max_z_dlas already
    carry them. The old code only disabled highzcut (z_min_lyb) / lowzcut (z_max_lyb),
    silently leaving lowzcut=True in production → a proximity DOUBLE-CUT on the F
    deposit that the truth window (_search_edges) does not apply → n_truth inflated
    ~1.43x relative to F.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures", "cddf"))
    from build_synthetic_cddf_fixture import build_synthetic_cddf
    from CDDF_analysis.calc_cddf import DLACatalogue
    from CDDF_analysis.cddf_forward.window import WindowSpec
    fx = build_synthetic_cddf(str(tmp_path), n_spec=4, n_samples=64,
                              p_dla=(1.0, 1.0, 1.0, 0.0),
                              peak_logN=(20.5, 21.0, 21.5, None),
                              peak_z=(2.6, 2.8, 3.0, None))
    cat = DLACatalogue(processed_file=fx["processed_file"], sample_file=fx["sample_file"],
                       catalog_file=fx["catalog_file"], sub_dla=0, snr=-2,
                       high_nhi_cut_value=21.9, window=WindowSpec())  # z_min_lyb=True, z_max_lyb=False
    assert cat.lowzcut is False and cat.highzcut is False
