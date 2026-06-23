"""Tests for the O1 end-to-end CDDF driver (``CDDF_analysis.cddf_forward.driver``).

"O1" = the raw probabilistic CDDF with NO selection correction — i.e. the numbers
the existing Pathway-A estimator (``calc_cddf.DLACatalogue``) already produces.  The
driver is therefore a faithful WRAPPER: it must NOT alter the estimator's arrays.
These tests pin that contract:

  * Faithfulness — driver arrays are byte-identical to direct ``DLACatalogue`` calls.
  * FILTER guard — refuses a FILTER-on catalog before doing any work.
  * Golden snapshot — regression-lock against a committed ``.npz``.
  * IO round-trip — ``save_o1_products`` then ``cddf_io.load_cddf_txt_table`` recovers
    the arrays.
  * @requires_data hook — runs on a real combined catalog only when ``CDDF_TEST_DATA``
    points at one (skipped by default).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
# Repo root on sys.path so ``CDDF_analysis`` resolves as a (namespace) package — calc_cddf
# uses ``from .set_parameters import *`` so it must import as part of the package (matching
# the production CLI's ``from CDDF_analysis.calc_cddf import DLACatalogue``).
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))

pytest.importorskip("h5py")
pytest.importorskip("astropy")
pytest.importorskip("scipy")

from CDDF_analysis import calc_cddf  # noqa: E402
from CDDF_analysis.cddf_forward.driver import (  # noqa: E402
    compute_o1_products,
    save_o1_products,
)
from CDDF_analysis.cddf_forward.window import WindowSpec  # noqa: E402
from CDDF_analysis import cddf_io  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402


# Fixed estimator args used across the faithfulness / golden / IO tests, so the
# driver and the direct comparison run on identical inputs.
_Z_MIN = 2.4
_Z_MAX = 3.1
_LNHI_MIN = 20.3
_LNHI_MAX = 22.5
_LNHI_NBINS = 4
_HUBBLE = 0.7

# DLACatalogue construction kwargs matching the single-absorber (sub_dla=False)
# 2LPT-0 FILTER-off layout the synthetic fixture emulates.
_DLACAT_KWARGS = dict(
    sub_dla=False,
    snr=-2,
    lowzcut=False,
    highzcut=False,
)


@pytest.fixture
def synth(tmp_path):
    return build_synthetic_cddf(tmp_path)


def _direct_products(synth):
    """Call the three DLACatalogue methods directly (the reference numbers)."""
    cat = calc_cddf.DLACatalogue(
        processed_file=synth["processed_file"],
        sample_file=synth["sample_file"],
        catalog_file=synth["catalog_file"],
        **_DLACAT_KWARGS,
    )
    cddf = cat.column_density_function(
        z_min=_Z_MIN,
        z_max=_Z_MAX,
        lnhi_nbins=_LNHI_NBINS,
        lnhi_min=_LNHI_MIN,
        lnhi_max=_LNHI_MAX,
    )
    dndx = cat.line_density(
        z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX
    )
    omega = cat.omega_dla_cddf(
        z_min=_Z_MIN,
        z_max=_Z_MAX,
        hubble=_HUBBLE,
        lnhi_nbins=_LNHI_NBINS,
        lnhi_min=_LNHI_MIN,
        lnhi_max=_LNHI_MAX,
    )
    return cddf, dndx, omega


def _driver_products(synth, **overrides):
    kwargs = dict(
        z_min=_Z_MIN,
        z_max=_Z_MAX,
        lnhi_min=_LNHI_MIN,
        lnhi_max=_LNHI_MAX,
        lnhi_nbins=_LNHI_NBINS,
        hubble=_HUBBLE,
        filter_low_likelihood=0,
    )
    kwargs.update(overrides)
    return compute_o1_products(
        synth["processed_file"],
        synth["sample_file"],
        synth["catalog_file"],
        **_DLACAT_KWARGS,
        **kwargs,
    )


class TestFaithfulness:
    """The driver must not alter the estimator's numbers (byte-identical)."""

    def test_cddf_byte_identical(self, synth):
        (l_Ncent, cddf, cddf68, cddf95, xerrs), _, _ = _direct_products(synth)
        prod = _driver_products(synth)
        np.testing.assert_array_equal(prod["cddf"]["logN"], l_Ncent)
        np.testing.assert_array_equal(prod["cddf"]["f"], cddf)
        np.testing.assert_array_equal(prod["cddf"]["f68"], cddf68)
        np.testing.assert_array_equal(prod["cddf"]["f95"], cddf95)
        np.testing.assert_array_equal(prod["cddf"]["xerrs"][0], xerrs[0])
        np.testing.assert_array_equal(prod["cddf"]["xerrs"][1], xerrs[1])

    def test_dndx_byte_identical(self, synth):
        _, (z_cent, dNdX, dndx68, dndx95, xerrs), _ = _direct_products(synth)
        prod = _driver_products(synth)
        np.testing.assert_array_equal(prod["dndx"]["z"], z_cent)
        np.testing.assert_array_equal(prod["dndx"]["dndx"], dNdX)
        np.testing.assert_array_equal(prod["dndx"]["dndx68"], dndx68)
        np.testing.assert_array_equal(prod["dndx"]["dndx95"], dndx95)

    def test_omega_byte_identical(self, synth):
        _, _, (z_cent, omega, o68, o95, xerrs) = _direct_products(synth)
        prod = _driver_products(synth)
        np.testing.assert_array_equal(prod["omega"]["z"], z_cent)
        np.testing.assert_array_equal(prod["omega"]["omega"], omega)
        np.testing.assert_array_equal(prod["omega"]["omega68"], o68)
        np.testing.assert_array_equal(prod["omega"]["omega95"], o95)


class TestFilterGuard:
    """FILTER-on must raise before constructing the catalogue / touching files."""

    def test_filter_on_raises_valueerror(self, synth):
        with pytest.raises(ValueError, match="FILTER"):
            _driver_products(synth, filter_low_likelihood=1)

    def test_filter_on_raises_before_any_work(self, synth, monkeypatch):
        # The guard runs first: DLACatalogue is never even constructed.
        sentinel = {"built": False}
        orig = calc_cddf.DLACatalogue

        def _spy(*a, **k):
            sentinel["built"] = True
            return orig(*a, **k)

        monkeypatch.setattr(calc_cddf, "DLACatalogue", _spy)
        with pytest.raises(ValueError):
            _driver_products(synth, filter_low_likelihood=1)
        assert sentinel["built"] is False


class TestProvenance:
    """Provenance records the inputs, the FILTER flag, and the WindowSpec."""

    def test_provenance_records_inputs(self, synth):
        prod = _driver_products(synth)
        prov = prod["provenance"]
        assert prov["processed_file"] == synth["processed_file"]
        assert prov["sample_file"] == synth["sample_file"]
        assert prov["catalog_file"] == synth["catalog_file"]
        assert prov["filter_low_likelihood"] == 0
        assert prov["z_min"] == _Z_MIN
        assert prov["z_max"] == _Z_MAX

    def test_provenance_records_windowspec(self, synth):
        win = WindowSpec(v_prox_kms=3000.0)
        prod = _driver_products(synth, window=win)
        # WindowSpec is RECORDED (O1 does not yet re-cut on it) — present in provenance.
        assert prod["provenance"]["window"] is not None

    def test_provenance_window_none_by_default(self, synth):
        prod = _driver_products(synth)
        assert prod["provenance"]["window"] is None


class TestGoldenSnapshot:
    """Regression-lock the seeded synthetic CDDF + dN/dX against a committed .npz."""

    GOLDEN = os.path.join(_HERE, "fixtures", "cddf", "o1_golden.npz")

    def test_golden_matches(self, tmp_path):
        # Rebuild the fixture with the SAME seed used to make the golden file.
        synth = build_synthetic_cddf(tmp_path)
        prod = _driver_products(synth)
        assert os.path.exists(self.GOLDEN), (
            f"golden snapshot missing: {self.GOLDEN} "
            "(regenerate with tests/fixtures/cddf/make_o1_golden.py)"
        )
        gold = np.load(self.GOLDEN)
        np.testing.assert_allclose(prod["cddf"]["logN"], gold["cddf_logN"], rtol=1e-10)
        np.testing.assert_allclose(prod["cddf"]["f"], gold["cddf_f"], rtol=1e-10)
        np.testing.assert_allclose(prod["dndx"]["z"], gold["dndx_z"], rtol=1e-10)
        np.testing.assert_allclose(prod["dndx"]["dndx"], gold["dndx"], rtol=1e-10)


class TestIORoundTrip:
    """save_o1_products → load_cddf_txt_table recovers the arrays."""

    def test_cddf_table_round_trip(self, synth, tmp_path):
        prod = _driver_products(synth)
        out_dir = str(tmp_path / "o1_out")
        paths = save_o1_products(prod, out_dir)
        assert "cddf" in paths and os.path.exists(paths["cddf"])
        loaded = cddf_io.load_cddf_txt_table(paths["cddf"])
        # cddf_io.save_cddf_txt_table writes "%.8e" (8 sig figs) — reuse it verbatim
        # rather than forking the writer, so the round-trip tolerance is ~1e-8.
        np.testing.assert_allclose(loaded["logN"], prod["cddf"]["logN"], rtol=1e-7)
        np.testing.assert_allclose(loaded["f_raw"], prod["cddf"]["f"], rtol=1e-7)

    def test_dndx_table_round_trip(self, synth, tmp_path):
        prod = _driver_products(synth)
        out_dir = str(tmp_path / "o1_out")
        paths = save_o1_products(prod, out_dir)
        assert "dndx" in paths and os.path.exists(paths["dndx"])
        data = np.loadtxt(paths["dndx"])
        if data.ndim == 1:
            data = data.reshape(1, -1)
        np.testing.assert_allclose(data[:, 0], prod["dndx"]["z"], rtol=1e-10)
        # Also assert the dN/dX VALUES round-trip (raw is col 6), not just z.
        np.testing.assert_allclose(data[:, 6], prod["dndx"]["dndx"], rtol=1e-7)

    def test_omega_table_is_written(self, synth, tmp_path):
        prod = _driver_products(synth)
        paths = save_o1_products(prod, str(tmp_path / "o1_out"))
        assert "omega" in paths and os.path.exists(paths["omega"])

    def test_o1_tables_not_labelled_as_alpha_calibrated(self, synth, tmp_path):
        # Regression-lock the O1-honesty fix: the saved tables must NOT carry the
        # reused writer's default "alpha(z) ... london mock" calibration claim.
        prod = _driver_products(synth)
        paths = save_o1_products(prod, str(tmp_path / "o1_out"))
        for key in ("dndx", "omega"):
            txt = open(paths[key]).read().lower()
            assert "london mock" not in txt
            assert "uncorrected" in txt
        cddf_txt = open(paths["cddf"]).read().lower()
        assert "uncorrected" in cddf_txt


@pytest.mark.skipif(
    not os.environ.get("CDDF_TEST_DATA"),
    reason="set CDDF_TEST_DATA=<dir with processed.h5 + sample.mat + catalog.fits> to run",
)
def test_requires_real_data_finite_nonnegative_cddf():
    """On a real 2LPT-0 combined catalog, the O1 CDDF must be finite + non-negative.

    Scaffold only — skipped unless ``CDDF_TEST_DATA`` points at a directory holding
    ``processed.h5``, ``sample.mat``, and ``catalog.fits``.
    """
    data_dir = os.environ["CDDF_TEST_DATA"]
    prod = compute_o1_products(
        os.path.join(data_dir, "processed.h5"),
        os.path.join(data_dir, "sample.mat"),
        os.path.join(data_dir, "catalog.fits"),
        z_min=2.0,
        z_max=4.0,
        lnhi_min=20.3,
        lnhi_max=22.5,
        lnhi_nbins=10,
        hubble=0.7,
        filter_low_likelihood=0,
        sub_dla=False,
    )
    f = prod["cddf"]["f"]
    assert np.all(np.isfinite(f))
    assert np.all(f >= 0.0)
