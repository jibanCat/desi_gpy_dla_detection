"""Tests for the O3 diagonal soft-completeness CS plumbing
(``CDDF_analysis.cddf_forward.diagonal_deposit``).

Two units, both testable WITHOUT the Bayesian core:

  * ``build_truth_map`` (§3.1) — read a truth FITS catalog, window each absorber
    IDENTICALLY to the measurement via the SAME ``WindowSpec``, bin into (logN, z)
    indices, join to the QSO catalog by TARGETID for each sightline's z_qso, and
    restrict to a BUILD/HELDOUT ``role_mask``.
  * ``DiagonalSoftDeposit`` (§3.2) — mirror ``_split_distributions_single`` but
    partition the deposited per-sample DLA probability mass by truth-presence into
    ``F_matched`` / ``F_unmatched``, with the mass-conservation invariant
    ``F_matched + F_unmatched == F`` to 1e-9, plus ``n_truth`` from the TruthMap.

All fixtures are synthetic. No GP files, no real data.
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
from CDDF_analysis.cddf_forward.window import WindowSpec  # noqa: E402
from CDDF_analysis.cddf_forward.diagonal_deposit import (  # noqa: E402
    build_truth_map,
    TruthMap,
    DiagonalSoftDeposit,
)
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402
from build_synthetic_truth_fixture import write_truth_catalog  # noqa: E402
from build_synthetic_multidla_fixture import (  # noqa: E402
    build_synthetic_multidla_cddf,
)


# Shared bin/window conventions for these tests.
_LNHI_EDGES = np.array([20.3, 20.9, 21.5, 22.5])
_Z_EDGES = np.array([2.4, 2.7, 3.0, 3.3])
_WINDOW = WindowSpec(z_min_lyb=False, z_max_lyb=False)  # no Lyβ edge shift -> simplest


@pytest.fixture
def synth(tmp_path):
    # 4 spectra, peaks chosen to fall inside known (logN, z) bins; z_qso=3.5.
    return build_synthetic_cddf(
        tmp_path,
        p_dla=(1.0, 1.0, 1.0, 0.0),
        peak_logN=(20.5, 21.0, 21.6, None),
        peak_z=(2.55, 2.85, 3.1, None),
        z_qso=3.5,
        z_min=2.4,
        z_max=3.3,
    )


# ---------------------------------------------------------------------------
# §3.1 build_truth_map
# ---------------------------------------------------------------------------
class TestBuildTruthMap:
    def test_returns_truthmap_with_per_target_bins(self, synth, tmp_path):
        # One truth absorber per active sightline, placed in a known bin.
        truth_file = str(tmp_path / "truth.fits")
        write_truth_catalog(
            truth_file,
            target_ids=[1000, 1001, 1002],
            nhi=[20.5, 21.0, 21.6],
            z=[2.55, 2.85, 3.1],
        )
        tmap = build_truth_map(
            truth_file,
            catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"],
            window=_WINDOW,
            lnhi_edges=_LNHI_EDGES,
            z_edges=_Z_EDGES,
        )
        assert isinstance(tmap, TruthMap)
        # logN 20.5 -> bin 0; z 2.55 -> bin 0
        assert (0, 0) in tmap.bins_for_target(1000)
        # logN 21.0 -> bin 1; z 2.85 -> bin 1
        assert (1, 1) in tmap.bins_for_target(1001)
        # logN 21.6 -> bin 2; z 3.1 -> bin 2
        assert (2, 2) in tmap.bins_for_target(1002)

    def test_n_truth_counts_absorbers_per_bin(self, synth, tmp_path):
        truth_file = str(tmp_path / "truth.fits")
        # two absorbers in (0,0), one in (1,1)
        write_truth_catalog(
            truth_file,
            target_ids=[1000, 1001, 1002],
            nhi=[20.5, 20.6, 21.0],
            z=[2.55, 2.5, 2.85],
        )
        tmap = build_truth_map(
            truth_file,
            catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"],
            window=_WINDOW,
            lnhi_edges=_LNHI_EDGES,
            z_edges=_Z_EDGES,
        )
        n_truth = tmap.n_truth_grid()
        # grid shape is (nlnhi_bins, nz_bins)
        assert n_truth.shape == (len(_LNHI_EDGES) - 1, len(_Z_EDGES) - 1)
        assert n_truth[0, 0] == 2
        assert n_truth[1, 1] == 1
        assert n_truth.sum() == 3

    def test_absorber_outside_window_is_excluded(self, synth, tmp_path):
        # An absorber inside the proximity zone (z very close to z_qso=3.5) is
        # windowed out IDENTICALLY to the measurement, so it must NOT appear.
        truth_file = str(tmp_path / "truth.fits")
        write_truth_catalog(
            truth_file,
            target_ids=[1000, 1001],
            nhi=[20.5, 21.0],
            # z 3.49 is within v_prox of z_qso=3.5 -> excluded by the proximity cut.
            z=[2.55, 3.49],
        )
        # Use a window with the default proximity cut active via the estimator edges.
        tmap = build_truth_map(
            truth_file,
            catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"],
            window=_WINDOW,
            lnhi_edges=_LNHI_EDGES,
            z_edges=_Z_EDGES,
        )
        # absorber for 1000 survives; 1001's is windowed out (above max search edge)
        assert len(tmap.bins_for_target(1000)) == 1
        assert len(tmap.bins_for_target(1001)) == 0

    def test_absorber_outside_lnhi_range_excluded(self, synth, tmp_path):
        # A sub-DLA (NHI 18.5) below the lnhi_edges floor must not be binned.
        truth_file = str(tmp_path / "truth.fits")
        write_truth_catalog(
            truth_file,
            target_ids=[1000, 1001],
            nhi=[18.5, 21.0],
            z=[2.55, 2.85],
        )
        tmap = build_truth_map(
            truth_file,
            catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"],
            window=_WINDOW,
            lnhi_edges=_LNHI_EDGES,
            z_edges=_Z_EDGES,
        )
        assert len(tmap.bins_for_target(1000)) == 0
        assert (1, 1) in tmap.bins_for_target(1001)

    def test_active_set_restricts_low_snr_sightline(self, tmp_path):
        # A truth absorber on a sightline that FAILS the estimator's SNR cut must NOT
        # inflate n_truth: the denominator population must match the F_matched
        # numerator (the active filter_dla_spectra set). Contract C3 / decision §3.
        # tid 1002 (spectrum 2) gets a below-threshold SNR -> not active.
        synth = build_synthetic_cddf(
            tmp_path,
            n_spec=3,
            p_dla=(1.0, 1.0, 1.0),
            peak_logN=(20.5, 21.0, 21.6),
            peak_z=(2.55, 2.85, 3.1),
            z_qso=3.5, z_min=2.4, z_max=3.3,
            snr=np.array([5.0, 5.0, -10.0]),  # tid 1002 below the snr cut
        )
        truth_file = str(tmp_path / "truth_snr.fits")
        write_truth_catalog(
            truth_file,
            target_ids=[1000, 1001, 1002],
            nhi=[20.5, 21.0, 21.6],
            z=[2.55, 2.85, 3.1],
        )
        # the estimator's active set (snr_thresh default in DLACatalogue is the
        # passed snr=-2; tid 1002 with snr=-10 is excluded).
        cat = calc_cddf.DLACatalogue(
            processed_file=synth["processed_file"],
            sample_file=synth["sample_file"],
            catalog_file=synth["catalog_file"],
            sub_dla=False, snr=-2, lowzcut=False, highzcut=False, window=_WINDOW,
        )
        active_ids = set(int(t) for t in cat.target_ids[cat.filter_dla_spectra()[0]])
        assert 1002 not in active_ids and 1000 in active_ids

        tmap = build_truth_map(
            truth_file,
            catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"],
            window=_WINDOW,
            lnhi_edges=_LNHI_EDGES,
            z_edges=_Z_EDGES,
            active_target_ids=active_ids,
        )
        # tid 1002's truth absorber (bin (2,2)) must NOT be counted.
        assert len(tmap.bins_for_target(1002)) == 0
        assert tmap.n_truth_grid()[2, 2] == 0
        # the two active sightlines' truths survive.
        assert (0, 0) in tmap.bins_for_target(1000)
        assert (1, 1) in tmap.bins_for_target(1001)
        assert tmap.n_truth_grid().sum() == 2

    def test_role_mask_restricts_to_build(self, synth, tmp_path):
        truth_file = str(tmp_path / "truth.fits")
        write_truth_catalog(
            truth_file,
            target_ids=[1000, 1001, 1002],
            nhi=[20.5, 21.0, 21.6],
            z=[2.55, 2.85, 3.1],
        )
        from CDDF_analysis.cddf_forward.split import sightline_role

        roles = {tid: sightline_role(tid) for tid in (1000, 1001, 1002)}
        build_ids = {tid for tid, r in roles.items() if r == "BUILD"}
        tmap = build_truth_map(
            truth_file,
            catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"],
            window=_WINDOW,
            lnhi_edges=_LNHI_EDGES,
            z_edges=_Z_EDGES,
            role_mask="BUILD",
        )
        # only BUILD targets contribute truth absorbers
        for tid in (1000, 1001, 1002):
            present = len(tmap.bins_for_target(tid)) > 0
            assert present == (tid in build_ids)


# ---------------------------------------------------------------------------
# §3.2 DiagonalSoftDeposit — partitioned, mass-conserving deposit
# ---------------------------------------------------------------------------
def _make_catalogue(synth, window):
    return calc_cddf.DLACatalogue(
        processed_file=synth["processed_file"],
        sample_file=synth["sample_file"],
        catalog_file=synth["catalog_file"],
        sub_dla=False,
        snr=-2,
        lowzcut=False,
        highzcut=False,
        window=window,
    )


def _full_truth_map(synth, tmap_targets, tmap_nhi, tmap_z, tmp_path):
    truth_file = str(tmp_path / "truth_dep.fits")
    write_truth_catalog(truth_file, target_ids=tmap_targets, nhi=tmap_nhi, z=tmap_z)
    return build_truth_map(
        truth_file,
        catalog_file=synth["catalog_file"],
        processed_file=synth["processed_file"],
        window=_WINDOW,
        lnhi_edges=_LNHI_EDGES,
        z_edges=_Z_EDGES,
    )


# ---------------------------------------------------------------------------
# C4 — multi-DLA deposit must honor self.second_dla (sum second=0..second_dla),
# mirroring _split_distributions; for second_dla=0 it reduces to single-DLA and
# truth multiplicity then measures close-pair incompleteness.
# ---------------------------------------------------------------------------
class TestMultiDLADeposit:
    def test_close_pair_incompleteness_single_dla(self, synth, tmp_path):
        # TWO truth absorbers on ONE sightline in the SAME (logN, z) cell. The
        # single-DLA recovery (second_dla=0) deposits ~1 unit of mass there, but
        # n_truth=2 -> C = F_matched / n_truth ~ 0.5 < 1 (close-pair incompleteness).
        # KEEP counting truth multiplicity (contract C4 / decision §4).
        truth_file = str(tmp_path / "truth_pair.fits")
        # spectrum 0 (tid 1000) peaks at (20.5, 2.55) -> bin (0,0). Put TWO truth
        # absorbers in that same cell on tid 1000.
        write_truth_catalog(
            truth_file,
            target_ids=[1000, 1000],
            nhi=[20.4, 20.6],
            z=[2.52, 2.58],
        )
        tmap = build_truth_map(
            truth_file,
            catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"],
            window=_WINDOW,
            lnhi_edges=_LNHI_EDGES,
            z_edges=_Z_EDGES,
        )
        # both truths land in bin (0,0)
        assert tmap.n_truth_grid()[0, 0] == 2
        cat = _make_catalogue(synth, _WINDOW)
        assert cat.second_dla == 0  # single-DLA model
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=_WINDOW
        )
        out = dep.deposit(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        # F_matched in (0,0) is the single-DLA recovered mass (~1, p_dla=1) but
        # n_truth is 2: the implied completeness F_matched/n_truth is ~0.5 < 1.
        fm = out["F_matched"][0, 0]
        nt = out["n_truth"][0, 0]
        assert nt == 2
        assert 0.5 < fm < 1.5  # single-DLA recovery cannot recover both
        assert fm / nt < 0.9   # close-pair incompleteness exposed

    def test_deposit_honors_second_dla_sum(self, tmp_path):
        # With a MAX_DLAS=2 (second_dla=1) catalogue, the deposit must SUM the
        # per-DLA contributions second=0..second_dla, mirroring
        # _split_distributions — so F equals the estimator's count over the SAME
        # _split_distributions path (column_density_function_counts). Pre-fix the
        # deposit hardcodes second=False and skips the second-DLA pass entirely.
        synth = build_synthetic_multidla_cddf(tmp_path)
        z_min, z_max = synth["z_min"], synth["z_max"]
        lnhi_edges = np.array([20.3, 22.5])
        z_edges = np.array([z_min, z_max])
        cat = calc_cddf.DLACatalogue(
            processed_file=synth["processed_file"],
            sample_file=synth["sample_file"],
            catalog_file=synth["catalog_file"],
            sub_dla=False, snr=-2, lowzcut=False, highzcut=False,
            second=1, window=_WINDOW,
        )
        assert cat.second_dla == 1
        truth_file = str(tmp_path / "truth_md.fits")
        write_truth_catalog(truth_file, target_ids=[1000], nhi=[20.5], z=[2.52])
        tmap = build_truth_map(
            truth_file, catalog_file=synth["catalog_file"],
            processed_file=synth["processed_file"], window=_WINDOW,
            lnhi_edges=lnhi_edges, z_edges=z_edges,
        )
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=lnhi_edges, z_edges=z_edges, window=_WINDOW
        )
        # The estimator's count over the SAME _split_distributions (second-summed)
        # path is the reference F: it goes through second=0..second_dla internally.
        cc = cat.column_density_function_counts(
            z_min=z_min, z_max=z_max, lnhi_nbins=1, lnhi_min=20.3, lnhi_max=22.5,
        )
        out = dep.deposit(z_min=z_min, z_max=z_max)
        # Deposit F (single z bin collapsed) must equal the estimator's count, which
        # is the second-summed _split_distributions total.
        np.testing.assert_allclose(out["F"][0, 0], cc["counts"][0], atol=1e-9)
        # And the per-second loop must be honored: the deposit must invoke the
        # second-DLA kernel pass (second=1), not only second=0.
        seconds_seen = []
        orig = dep._iter_sightline_deposits

        def _spy(*args, **kwargs):
            seconds_seen.append(kwargs.get("second", False))
            return orig(*args, **kwargs)

        dep._iter_sightline_deposits = _spy
        dep.deposit(z_min=z_min, z_max=z_max)
        assert 0 in [int(s) for s in seconds_seen]
        assert 1 in [int(s) for s in seconds_seen]


class TestDiagonalSoftDeposit:
    def test_partition_is_mass_conserving(self, synth, tmp_path):
        # Truth absorbers for spectra 0 and 1 (in their peak bins); spectrum 2 has
        # NO truth absorber -> its deposited mass must land in F_unmatched.
        tmap = _full_truth_map(
            synth,
            tmap_targets=[1000, 1001],
            tmap_nhi=[20.5, 21.0],
            tmap_z=[2.55, 2.85],
            tmp_path=tmp_path,
        )
        cat = _make_catalogue(synth, _WINDOW)
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=_WINDOW
        )
        out = dep.deposit(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        F = out["F"]
        Fm = out["F_matched"]
        Fu = out["F_unmatched"]
        assert F.shape == (len(_LNHI_EDGES) - 1, len(_Z_EDGES) - 1)
        # Exhaustive + mass-conserving partition.
        np.testing.assert_allclose(Fm + Fu, F, atol=1e-9, rtol=0)

    def test_F_equals_unpartitioned_windowed_mean(self, synth, tmp_path):
        # F must equal the SAME catalogue's unpartitioned windowed Poisson-binomial
        # mean count per (logN, z) bin (sum of deposited p_dla), independent of truth.
        tmap = _full_truth_map(
            synth,
            tmap_targets=[1000],
            tmap_nhi=[20.5],
            tmap_z=[2.55],
            tmp_path=tmp_path,
        )
        cat = _make_catalogue(synth, _WINDOW)
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=_WINDOW
        )
        out = dep.deposit(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        # Independent reference: the windowed expected count per bin, summed over
        # all active spectra/samples in the (logN, z) cell, via the public
        # count-space accessor on the SAME catalogue (added in calc_cddf).
        ref = dep.reference_count_grid(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        np.testing.assert_allclose(out["F"], ref, atol=1e-9, rtol=0)

    def test_matched_mass_goes_to_truth_bins(self, synth, tmp_path):
        # Spectrum 0 (tid 1000, p_dla=1) has a truth absorber in its OWN peak bin
        # (logN 20.5 -> il 0, z 2.55 -> iz 0). Its deposited mass there is matched.
        tmap = _full_truth_map(
            synth,
            tmap_targets=[1000],
            tmap_nhi=[20.5],
            tmap_z=[2.55],
            tmp_path=tmp_path,
        )
        cat = _make_catalogue(synth, _WINDOW)
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=_WINDOW
        )
        out = dep.deposit(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        # bin (0,0) is a truth bin for tid 1000 -> matched mass there is > 0
        assert out["F_matched"][0, 0] > 0
        # and for a bin with no truth (e.g. 2,2 had no truth absorber at all)
        assert out["F_matched"][2, 2] == 0.0

    def test_no_truth_makes_everything_unmatched(self, synth, tmp_path):
        # Empty truth map -> ALL deposited mass is unmatched, F_matched all zero.
        tmap = _full_truth_map(
            synth,
            tmap_targets=[9999],  # not in the run -> empty truth map
            tmap_nhi=[20.5],
            tmap_z=[2.55],
            tmp_path=tmp_path,
        )
        cat = _make_catalogue(synth, _WINDOW)
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=_WINDOW
        )
        out = dep.deposit(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        np.testing.assert_allclose(out["F_matched"], 0.0, atol=1e-12)
        np.testing.assert_allclose(out["F_unmatched"], out["F"], atol=1e-9)

    def test_n_truth_grid_surfaced(self, synth, tmp_path):
        tmap = _full_truth_map(
            synth,
            tmap_targets=[1000, 1001],
            tmap_nhi=[20.5, 21.0],
            tmap_z=[2.55, 2.85],
            tmp_path=tmp_path,
        )
        cat = _make_catalogue(synth, _WINDOW)
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=_WINDOW
        )
        out = dep.deposit(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        np.testing.assert_array_equal(out["n_truth"], tmap.n_truth_grid())

    def test_deposit_restricted_to_target_subset(self, synth, tmp_path):
        # Restricting the deposit to a TARGETID subset drops the other sightlines'
        # mass entirely (used by the driver to deposit on the BUILD split only).
        tmap = _full_truth_map(
            synth,
            tmap_targets=[1000, 1001],
            tmap_nhi=[20.5, 21.0],
            tmap_z=[2.55, 2.85],
            tmp_path=tmp_path,
        )
        cat = _make_catalogue(synth, _WINDOW)
        dep = DiagonalSoftDeposit(
            cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=_WINDOW
        )
        full = dep.deposit(z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1])
        subset = dep.deposit(
            z_min=_Z_EDGES[0], z_max=_Z_EDGES[-1], target_ids={1000}
        )
        # subset total <= full total (other sightlines removed)
        assert subset["F"].sum() < full["F"].sum()
        # still mass-conserving within the subset
        np.testing.assert_allclose(
            subset["F_matched"] + subset["F_unmatched"], subset["F"], atol=1e-9
        )

    def test_window_mismatch_raises(self, synth, tmp_path):
        # The catalogue and the truth map must share the SAME window.
        tmap = _full_truth_map(
            synth,
            tmap_targets=[1000],
            tmap_nhi=[20.5],
            tmap_z=[2.55],
            tmp_path=tmp_path,
        )
        cat = _make_catalogue(synth, _WINDOW)
        other = WindowSpec(z_min_lyb=True, z_max_lyb=False)
        with pytest.raises(ValueError, match="WindowSpec"):
            DiagonalSoftDeposit(
                cat, tmap, lnhi_edges=_LNHI_EDGES, z_edges=_Z_EDGES, window=other
            )


# ---------------------------------------------------------------------------
# C7 — z_max_lyb cap must use the PER-SIGHTLINE lymanbeta(max_z_dla) clamped at
# the GLOBAL search z_max (mirroring _split_distributions_single), NOT the
# per-sightline stored max_z_dla.
# ---------------------------------------------------------------------------
class TestZMaxLybParity:
    def test_search_edges_caps_upper_z_at_global_z_max_like_estimator(self, tmp_path):
        # _search_edges must reproduce the estimator's z_max_lyb cap EXACTLY:
        #   estimator: upper_z = min(lymanbeta(max_z_dla), GLOBAL z_max)
        # Pre-fix it used min(lymanbeta(max_z_dla), per-sightline max_z_dla), so when
        # the GLOBAL z_max is the binding constraint (below lymanbeta(max_z_dla)) the
        # two disagree. Compare the function output to the estimator's own arithmetic.
        from CDDF_analysis.cddf_forward.diagonal_deposit import _search_edges

        window = WindowSpec(z_min_lyb=False, z_max_lyb=True)
        synth = build_synthetic_cddf(
            tmp_path, n_spec=1, p_dla=(1.0,), peak_logN=(20.5,), peak_z=(2.45,),
            z_qso=3.5, z_min=2.4, z_max=3.3,
        )
        cat = calc_cddf.DLACatalogue(
            processed_file=synth["processed_file"],
            sample_file=synth["sample_file"],
            catalog_file=synth["catalog_file"],
            sub_dla=False, snr=-2, lowzcut=False, highzcut=False, window=window,
        )
        spec = 0
        max_z_dla = float(cat.z_max(spec))   # per-sightline stored edge (3.3)
        z_qso = float(cat.z_qsos[spec])
        # global search z_max BELOW lymanbeta(max_z_dla) so it is the binding cap.
        global_z_min, global_z_max = 2.4, 2.5
        lyb_max = float(cat.lymanbeta(max_z_dla))
        assert global_z_max < lyb_max < max_z_dla  # the binding-cap regime

        # estimator arithmetic (mirrors _split_distributions_single line ~1896)
        estimator_upper = float(np.min([cat.lymanbeta(max_z_dla), global_z_max]))
        _lo, upper = _search_edges(window, z_qso, global_z_min, global_z_max)
        # min(z_min_dla, z_max_dla) stored edges are passed as the base window; the
        # function must clamp at the GLOBAL z_max, not the stored per-sightline edge.
        lower2, upper2 = _search_edges(
            window, z_qso, z_min=2.4, z_max=global_z_max, max_z_dla=max_z_dla
        )
        np.testing.assert_allclose(upper2, estimator_upper, atol=1e-12)
