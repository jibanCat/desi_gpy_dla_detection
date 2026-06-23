"""Tests for the Pathway-A probabilistic CDDF estimator (``CDDF_analysis/calc_cddf.py``).

Tier 0 = pure module functions (no HDF5/instance).
Tier 1 = ``DLACatalogue`` on a tiny synthetic fixture (requires the numpy-2.0 compat fix
so the class can construct on numpy>=2.0).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
# Repo root on sys.path so ``CDDF_analysis`` resolves as a (namespace) package — calc_cddf
# uses ``from .set_parameters import *``, so it must be imported as part of the package
# (matching the production CLI's ``from CDDF_analysis.calc_cddf import DLACatalogue``).
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))

pytest.importorskip("h5py")
pytest.importorskip("astropy")

from CDDF_analysis import calc_cddf  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402


@pytest.fixture
def synth(tmp_path):
    return build_synthetic_cddf(tmp_path)


def _make_catalogue(synth):
    return calc_cddf.DLACatalogue(
        processed_file=synth["processed_file"],
        sample_file=synth["sample_file"],
        catalog_file=synth["catalog_file"],
        sub_dla=False,
        snr=-2,
        lowzcut=False,
        highzcut=False,
    )


class TestDLACatalogueConstruction:
    """The numpy-2.0 compat fix is what lets DLACatalogue construct at all."""

    def test_constructs_single_absorber_layout(self, synth):
        cat = _make_catalogue(synth)
        assert cat.p_dla.shape == (synth["n_spec"],)
        # p_dla is the DLA(1) column of model_posteriors for sub_dla=False
        np.testing.assert_allclose(cat.p_dla, synth["p_dla"])

    def test_log_norm_like_normalization_invariant(self, synth):
        cat = _make_catalogue(synth)
        # Active spectra carry self-normalized sample weights summing to ~1.
        for spec in range(synth["n_spec"]):
            if synth["p_dla"][spec] <= 0:
                continue
            weights = np.exp(cat._log_norm_like(spec))
            assert 0.95 < weights.sum() < 1.05


class TestHighNhiCutDefault:
    """The high-N_HI ceiling default was raised 22.0 → 22.5 (extended prior tail)."""

    def test_default_high_nhi_cut_value_is_22_5(self, synth):
        cat = _make_catalogue(synth)
        assert cat.high_nhi_cut_value == 22.5
