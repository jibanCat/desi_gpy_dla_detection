"""Tests for the count-space CI accessor added to ``calc_cddf.DLACatalogue`` (§3.4).

The estimator returns f(N)/dN/dX/Ω in PHYSICAL units (divided by ΔN·ΔX).  The O3
diagonal correction operates in COUNT space, so we surface — ADDITIVELY, without
touching any existing method's output — the per-bin Poisson-binomial expected count
(MAP) + 68/95 COUNT interval the estimator already computes internally in
``_get_confidence_intervals``.

Pinned here:
  * the count-space MAP per bin equals the existing ``_get_confidence_intervals``
    MAP (i.e. it reuses the internals, not a re-derivation);
  * re-normalizing the count-space MAP by ΔN·ΔX reproduces the O1 f(N) EXACTLY
    (the §3.4 round-trip invariant);
  * the same for dN/dX (count / ΔX).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))

pytest.importorskip("h5py")
pytest.importorskip("astropy")
pytest.importorskip("scipy")

from CDDF_analysis import calc_cddf  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402

_Z_MIN = 2.4
_Z_MAX = 3.1
_LNHI_MIN = 20.3
_LNHI_MAX = 22.5
_LNHI_NBINS = 4


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


class TestCountSpaceCDDF:
    def test_returns_count_mean_and_intervals(self, synth):
        cat = _make_catalogue(synth)
        out = cat.column_density_function_counts(
            z_min=_Z_MIN,
            z_max=_Z_MAX,
            lnhi_nbins=_LNHI_NBINS,
            lnhi_min=_LNHI_MIN,
            lnhi_max=_LNHI_MAX,
        )
        # keys: logN, counts, counts68, counts95, dN, dX
        for key in ("logN", "counts", "counts68", "counts95", "dN", "dX"):
            assert key in out
        assert out["counts"].shape == (_LNHI_NBINS,)
        assert out["counts68"].shape == (_LNHI_NBINS, 2)
        assert out["counts95"].shape == (_LNHI_NBINS, 2)
        # intervals bracket the MAP
        assert np.all(out["counts95"][:, 0] <= out["counts68"][:, 0])
        assert np.all(out["counts68"][:, 0] <= out["counts"])
        assert np.all(out["counts"] <= out["counts68"][:, 1])
        assert np.all(out["counts68"][:, 1] <= out["counts95"][:, 1])

    def test_count_map_matches_internal_confidence_intervals(self, synth):
        # The count-space MAP must BE the internal Poisson-binomial MAP (reuse, not
        # re-derivation): compare against _get_confidence_intervals directly.
        cat = _make_catalogue(synth)
        l_nhi = np.linspace(_LNHI_MIN, _LNHI_MAX, num=_LNHI_NBINS + 1)
        (ndlas, l68, l95) = cat._get_confidence_intervals(
            q_bins=l_nhi, lred=_Z_MIN, ured=_Z_MAX, lnhi_min=_LNHI_MIN, nhi=True
        )
        out = cat.column_density_function_counts(
            z_min=_Z_MIN,
            z_max=_Z_MAX,
            lnhi_nbins=_LNHI_NBINS,
            lnhi_min=_LNHI_MIN,
            lnhi_max=_LNHI_MAX,
        )
        np.testing.assert_array_equal(out["counts"], np.array(ndlas))
        np.testing.assert_array_equal(out["counts68"], np.array(l68))
        np.testing.assert_array_equal(out["counts95"], np.array(l95))

    def test_renormalizing_counts_reproduces_o1_fN_exactly(self, synth):
        # §3.4 round-trip: counts / (dN * dX) == the O1 f(N), byte-identical.
        cat = _make_catalogue(synth)
        (l_Ncent, cddf, cddf68, cddf95, _xerrs) = cat.column_density_function(
            z_min=_Z_MIN,
            z_max=_Z_MAX,
            lnhi_nbins=_LNHI_NBINS,
            lnhi_min=_LNHI_MIN,
            lnhi_max=_LNHI_MAX,
        )
        out = cat.column_density_function_counts(
            z_min=_Z_MIN,
            z_max=_Z_MAX,
            lnhi_nbins=_LNHI_NBINS,
            lnhi_min=_LNHI_MIN,
            lnhi_max=_LNHI_MAX,
        )
        f_from_counts = out["counts"] / out["dX"] / out["dN"]
        np.testing.assert_array_equal(f_from_counts, cddf)
        np.testing.assert_array_equal(out["logN"], l_Ncent)
        f68 = out["counts68"] / out["dX"] / np.vstack([out["dN"], out["dN"]]).T
        np.testing.assert_array_equal(f68, cddf68)


class TestCountSpaceDNDX:
    def test_renormalizing_counts_reproduces_o1_dndx_exactly(self, synth):
        cat = _make_catalogue(synth)
        (z_cent, dNdX, dndx68, dndx95, _xerrs) = cat.line_density(
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX
        )
        out = cat.line_density_counts(
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX
        )
        for key in ("z", "counts", "counts68", "counts95", "dX"):
            assert key in out
        np.testing.assert_array_equal(out["z"], z_cent)
        d_from_counts = out["counts"] / out["dX"]
        np.testing.assert_array_equal(d_from_counts, dNdX)
        d68 = out["counts68"] / np.vstack([out["dX"], out["dX"]]).T
        np.testing.assert_array_equal(d68, dndx68)
