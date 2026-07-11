"""Queue-1 characterization tests: the two calc_cddf blocker behaviors, pinned AS-IS.

These tests pin CURRENT behavior (they are not endorsements of it):

1. NaN ``model_posteriors`` row-drop (calc_cddf.py:522-524): a row with NaN DLA
   columns is removed from ``condition`` and thereafter behaves EXACTLY as if the
   spectrum were never in the file — it leaves the counts AND the path-length
   denominator. (The finder's own convention is NaN-as-zero via ``np.nansum``,
   ``bayesian_model_selection.py:279`` — the two disagree on most production rows;
   resolving that is the Queue-1 science task. If the convention is changed, these
   pins must be consciously re-derived, not silently updated.)

2. ``sub_dla`` column offset: ``p_dla = model_posteriors[:, 1+sub_dla:].sum`` and
   ``p_no_dla = model_posteriors[:, :1+sub_dla].sum``. On a production
   SINGLE_ABSORBER_MODEL=1 layout ``[Null, 1abs, 2abs, ...]`` constructing with
   ``sub_dla=True`` silently reassigns the 1-absorber column into ``p_no_dla`` —
   the exact defect found in the WIP head-to-head harness (2026-07-10 plan note).
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

from CDDF_analysis import calc_cddf  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402

LNHI_BINS = 12


def _make_catalogue(paths, sub_dla=False):
    return calc_cddf.DLACatalogue(
        processed_file=paths["processed_file"],
        sample_file=paths["sample_file"],
        catalog_file=paths["catalog_file"],
        sub_dla=sub_dla,
        snr=-2,
        lowzcut=False,
        highzcut=False,
    )


def _cddf_and_dndx(cat):
    lnhi, cddf, _, _, _ = cat.column_density_function(
        z_min=2.0, z_max=3.4, lnhi_nbins=LNHI_BINS, lnhi_min=20.3, lnhi_max=21.9
    )
    z_cent, dndx, _, _, _ = cat.line_density(z_min=2.0, z_max=3.4, lnhi_min=20.3)
    return lnhi, cddf, z_cent, dndx


class TestNaNRowDrop:
    """Pin: a NaN-DLA-column row is dropped from counts AND path length."""

    ACTIVE = dict(
        n_spec=4,
        p_dla=(1.0, 1.0, 1.0, 1.0),
        peak_logN=(20.5, 21.0, 21.5, 20.7),
        peak_z=(2.6, 2.8, 3.0, 2.7),
    )

    def test_condition_drops_nan_row(self, tmp_path):
        paths = build_synthetic_cddf(tmp_path, nan_dla_rows=(3,), **self.ACTIVE)
        cat = _make_catalogue(paths)
        assert bool(cat.condition[3]) is False
        assert cat.condition[:3].all()

    def test_nan_row_equivalent_to_absent_spectrum(self, tmp_path):
        """The crispest statement of the current convention: NaN row == spectrum
        never present. Counts AND dX identical to a 3-spectrum file."""
        d_nan = tmp_path / "nan4"
        d_abs = tmp_path / "abs3"
        d_nan.mkdir()
        d_abs.mkdir()
        p_nan = build_synthetic_cddf(d_nan, nan_dla_rows=(3,), **self.ACTIVE)
        p_abs = build_synthetic_cddf(
            d_abs,
            n_spec=3,
            p_dla=(1.0, 1.0, 1.0),
            peak_logN=(20.5, 21.0, 21.5),
            peak_z=(2.6, 2.8, 3.0),
        )
        c_nan = _make_catalogue(p_nan)
        c_abs = _make_catalogue(p_abs)
        lnhi_a, cddf_a, zc_a, dndx_a = _cddf_and_dndx(c_nan)
        lnhi_b, cddf_b, zc_b, dndx_b = _cddf_and_dndx(c_abs)
        np.testing.assert_allclose(cddf_a, cddf_b, rtol=1e-12)
        np.testing.assert_allclose(dndx_a, dndx_b, rtol=1e-12)
        # and the drop really removes path length, not just counts:
        d_full = tmp_path / "full4"
        d_full.mkdir()
        dx_nan = c_nan.path_length(2.0, 3.4)
        dx_full = _make_catalogue(
            build_synthetic_cddf(d_full, **self.ACTIVE)
        ).path_length(2.0, 3.4)
        assert dx_nan < dx_full  # 3 windows vs 4 windows

    def test_choice_is_load_bearing(self, tmp_path):
        """The dropped spectrum's contribution is nonzero — i.e. the NaN
        convention materially changes f(N); this is why Queue 1 must trace it."""
        d_b, d_v = tmp_path / "b", tmp_path / "v"
        d_b.mkdir()
        d_v.mkdir()
        base = _make_catalogue(build_synthetic_cddf(d_b, **self.ACTIVE))
        van = _make_catalogue(build_synthetic_cddf(d_v, nan_dla_rows=(3,), **self.ACTIVE))
        _, cddf_base, _, _ = _cddf_and_dndx(base)
        _, cddf_van, _, _ = _cddf_and_dndx(van)
        # atol=0: cddf values are O(1e-22) cm^2, far below allclose's default atol
        assert not np.allclose(cddf_base, cddf_van, rtol=1e-6, atol=0.0)


class TestSubDlaColumnOffset:
    """Pin the ``1 + sub_dla`` column arithmetic in both layouts."""

    def test_subdla_true_layout(self, tmp_path):
        paths = build_synthetic_cddf(
            tmp_path,
            n_spec=4,
            p_dla=(0.7, 0.9, 0.0, 0.5),
            p_sub=(0.2, 0.0, 0.6, 0.1),
            peak_logN=(20.5, 21.0, None, 20.7),
            peak_z=(2.6, 2.8, None, 2.7),
            sub_dla=True,
        )
        cat = _make_catalogue(paths, sub_dla=True)
        mp = paths["model_posteriors"]
        # p_dla excludes the sub-DLA column; p_no_dla includes it
        np.testing.assert_allclose(cat.p_dla, mp[:, 2:].sum(axis=1))
        np.testing.assert_allclose(cat.p_no_dla, mp[:, :2].sum(axis=1))
        # spectrum 2 (pure sub-DLA, p_sub=0.6) contributes NO DLA probability
        assert cat.p_dla[2] == 0.0

    def test_single_absorber_file_with_subdla_true_misassigns(self, tmp_path):
        """THE HARNESS BUG MODE (pinned, not endorsed): production
        SINGLE_ABSORBER_MODEL=1 layout [Null, 1abs, 2abs, 3abs, 4abs] read with
        sub_dla=True treats the 1-absorber column as sub-DLA: it vanishes from
        p_dla and lands in p_no_dla."""
        common = dict(
            n_spec=3,
            p_dla=(0.8, 1.0, 0.6),
            peak_logN=(20.5, 21.0, 20.7),
            peak_z=(2.6, 2.8, 2.7),
            n_extra_zero_cols=3,  # -> 5 columns, all absorber mass in col 1
        )
        paths = build_synthetic_cddf(tmp_path, **common)
        cat_ok = _make_catalogue(paths, sub_dla=False)
        cat_bug = _make_catalogue(paths, sub_dla=True)
        np.testing.assert_allclose(cat_ok.p_dla, [0.8, 1.0, 0.6])
        # sub_dla=True: cols[2:] are the zero padding -> ALL absorber prob lost
        np.testing.assert_allclose(cat_bug.p_dla, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(cat_bug.p_no_dla, [1.0, 1.0, 1.0])
        # and the CDDF built under the wrong flag is empty
        _, cddf_bug, _, _ = _cddf_and_dndx(cat_bug)
        _, cddf_ok, _, _ = _cddf_and_dndx(cat_ok)
        assert cddf_bug.sum() == 0.0
        assert cddf_ok.sum() > 0.0
