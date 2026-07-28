"""Tests for CDDF_analysis/diagnostics/deployment/loa_matterhorn_consistency.py.

MOCK-SAFE by construction: every fixture is a synthetic in-memory catalog with
mock-magnitude TARGETIDs (O(1e3)).  No real survey data is read, so this suite
may be committed and run anywhere.  It needs numpy only -- no fitsio, no h5py --
so unlike tests/test_zresolved_pc.py it does NOT silently skip outside `gpdla`.
"""
from __future__ import annotations

import numpy as np
import pytest

from CDDF_analysis.diagnostics.deployment.loa_matterhorn_consistency import (
    DEFAULT_REAL_TID_THRESHOLD,
    _looks_real,
    compare,
    greedy_match,
    select_rows,
)

LYA = 1215.67

_SEL = dict(nhi_min=20.3, snr_min=2.0, gp_conf=0.99, z_qso_min=2.0, z_qso_max=4.25,
            lam_rf_min=1025.0, lam_rf_max=1216.0)


def mkcat(tid, z_dla, nhi, *, z_qso=None, snr=10.0, p_dla=1.0, dlaflag=0):
    n = len(tid)
    z_dla = np.asarray(z_dla, dtype=float)
    if z_qso is None:
        z_qso = z_dla + 0.25
    return {
        "TARGETID": np.asarray(tid, dtype=np.int64),
        "Z_DLA": z_dla,
        "NHI": np.asarray(nhi, dtype=float),
        "Z_QSO": np.broadcast_to(np.asarray(z_qso, dtype=float), (n,)).copy(),
        "SNR_REDSIDE": np.broadcast_to(np.asarray(snr, dtype=float), (n,)).copy(),
        "P_DLA": np.broadcast_to(np.asarray(p_dla, dtype=float), (n,)).copy(),
        "DLAFLAG": np.broadcast_to(np.asarray(dlaflag), (n,)).copy(),
    }


# --------------------------------------------------------------------------
# greedy_match
# --------------------------------------------------------------------------
def test_greedy_match_pairs_nearest_first():
    za = np.array([2.500, 2.900])
    zb = np.array([2.905, 2.502])
    pairs = greedy_match(za, zb, 0.01)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_greedy_match_respects_dz_tolerance():
    # |dz|/(1+z) = 0.05/3.5 ~ 0.0143 > 0.01 -> no pair
    assert greedy_match(np.array([2.5]), np.array([2.55]), 0.01) == []
    assert greedy_match(np.array([2.5]), np.array([2.51]), 0.01) == [(0, 0)]


def test_greedy_match_each_absorber_used_once():
    # two A absorbers compete for one B absorber; only the closer wins
    pairs = greedy_match(np.array([2.500, 2.505]), np.array([2.5001]), 0.01)
    assert pairs == [(0, 0)]


def test_greedy_match_empty_side():
    assert greedy_match(np.array([]), np.array([2.5]), 0.01) == []
    assert greedy_match(np.array([2.5]), np.array([]), 0.01) == []


# --------------------------------------------------------------------------
# select_rows
# --------------------------------------------------------------------------
def test_select_rows_applies_every_cut():
    cat = mkcat([1, 2, 3, 4, 5],
                z_dla=[2.5, 2.5, 2.5, 2.5, 2.5],
                nhi=[20.5, 20.0, 20.5, 20.5, 20.5],
                snr=[10, 10, 1.0, 10, 10],
                p_dla=[1.0, 1.0, 1.0, 0.5, 1.0],
                dlaflag=[0, 0, 0, 0, 16])
    m = select_rows(cat, **_SEL)
    assert m.tolist() == [True, False, False, False, False]


def test_select_rows_lam_rf_window_excludes_proximate_absorber():
    # z_dla == z_qso -> lam_rf = 1215.67 (inside); z_dla far below -> lam_rf < 1025
    cat = mkcat([1, 2], z_dla=[3.0, 2.35], nhi=[21.0, 21.0], z_qso=[3.0, 3.0])
    m = select_rows(cat, **_SEL)
    lam = LYA * (1 + cat["Z_DLA"]) / (1 + cat["Z_QSO"])
    assert lam[0] > 1025.0 and lam[1] < 1025.0
    assert m.tolist() == [True, False]


# --------------------------------------------------------------------------
# compare -- structure and bookkeeping
# --------------------------------------------------------------------------
def test_compare_identical_catalogs_is_perfect():
    tid = [1001, 1001, 1002, 1003]
    z = [2.50, 2.90, 3.00, 2.20]
    n = [20.5, 21.2, 20.9, 22.0]
    cat = mkcat(tid, z, n)
    r = compare(cat, cat, sel_kw=_SEL)
    assert r["sightlines"]["n_shared"] == 3
    assert r["sightlines"]["n_only_a"] == 0 == r["sightlines"]["n_only_b"]
    assert r["absorbers"]["n_matched_pairs"] == 4
    assert r["absorbers"]["match_rate_a_on_shared"] == 1.0
    assert r["count_agreement"]["frac_same_absorber_count"] == 1.0
    assert r["delta_nhi_b_minus_a"]["median"] == 0.0
    assert r["delta_z_rel_b_minus_a"]["median"] == 0.0


def test_compare_taxonomy_buckets_exhaust_every_absorber():
    A = mkcat([1001, 1001, 1002, 1004],
              [2.50, 2.90, 3.00, 2.30], [20.5, 21.0, 20.9, 21.1])
    B = mkcat([1001, 1002, 1003],
              [2.501, 2.60, 2.70], [20.6, 20.8, 20.7])
    r = compare(A, B, sel_kw=_SEL)
    tax = r["taxonomy"]
    assert (tax["matched_pair"] + tax["same_sightline_no_z_counterpart_a"]
            + tax["sightline_absent_from_b"]) == r["absorbers"]["n_a"]
    assert (tax["matched_pair"] + tax["same_sightline_no_z_counterpart_b"]
            + tax["sightline_absent_from_a"]) == r["absorbers"]["n_b"]


def test_compare_counts_shared_and_exclusive_sightlines():
    A = mkcat([1001, 1002, 1004], [2.5, 3.0, 2.3], [20.5, 20.9, 21.1])
    B = mkcat([1001, 1002, 1003], [2.5, 3.0, 2.7], [20.5, 20.9, 20.7])
    r = compare(A, B, sel_kw=_SEL)
    assert r["sightlines"] == pytest.approx(
        {"n_a": 3, "n_b": 3, "n_shared": 2, "n_only_a": 1, "n_only_b": 1,
         "shared_frac_of_a": 2 / 3, "shared_frac_of_b": 2 / 3, "jaccard": 0.5})


def test_compare_count_confusion_matrix():
    A = mkcat([1001, 1001, 1002], [2.5, 2.9, 3.0], [20.5, 21.0, 20.9])
    B = mkcat([1001, 1002], [2.5, 3.0], [20.5, 20.9])
    r = compare(A, B, sel_kw=_SEL)
    assert r["count_agreement"]["confusion"] == {"1->1": 1, "2->1": 1}
    assert r["count_agreement"]["n_same_absorber_count"] == 1


def test_compare_delta_nhi_sign_is_b_minus_a():
    A = mkcat([1001], [2.5], [20.5])
    B = mkcat([1001], [2.5], [20.9])
    r = compare(A, B, sel_kw=_SEL)
    assert r["delta_nhi_b_minus_a"]["median"] == pytest.approx(0.4)


def test_compare_snr_binning_partitions_matched_pairs():
    A = mkcat([1001, 1002, 1003], [2.5, 2.5, 2.5], [20.5, 20.5, 20.5],
              snr=[2.5, 5.0, 20.0])
    B = mkcat([1001, 1002, 1003], [2.5, 2.5, 2.5], [20.6, 20.6, 20.6],
              snr=[2.5, 5.0, 20.0])
    r = compare(A, B, sel_kw=_SEL)
    assert sum(row["n_matched"] for row in r["by_snr"]) == \
        r["absorbers"]["n_matched_pairs"] == 3


def test_compare_processed_lists_split_the_exclusive_bucket():
    A = mkcat([1001, 1004], [2.5, 2.3], [20.5, 21.1])
    B = mkcat([1001, 1003], [2.5, 2.7], [20.5, 20.7])
    # 1004 WAS processed by run B (so a genuine non-detection);
    # 1003 was NOT processed by run A (so not a miss at all).
    r = compare(A, B, sel_kw=_SEL,
                processed_a=[1001, 1004], processed_b=[1001, 1003, 1004])
    assert r["processed_denominator_available"] is True
    assert r["processed"]["n_only_a_and_processed_by_b"] == 1
    assert r["processed"]["n_only_b_and_processed_by_a"] == 0
    assert r["processed"]["n_only_b_not_processed_by_a"] == 1


def test_compare_without_processed_lists_flags_denominator_missing():
    A = mkcat([1001], [2.5], [20.5])
    r = compare(A, A, sel_kw=_SEL)
    assert r["processed_denominator_available"] is False
    assert "processed" not in r


def test_restrict_to_common_drops_sightlines_absent_from_the_other_raw_catalog():
    # 1004 exists only in A's raw catalog; 1003 only in B's.  With the restriction
    # both vanish, leaving a pure "processed by both" population.
    A = mkcat([1001, 1004], [2.5, 2.3], [20.5, 21.1])
    B = mkcat([1001, 1003], [2.5, 2.7], [20.5, 20.7])
    loose = compare(A, B, sel_kw=_SEL)
    strict = compare(A, B, sel_kw=_SEL, restrict_to_common=True)
    assert loose["sightlines"]["n_only_a"] == 1 and loose["sightlines"]["n_only_b"] == 1
    assert strict["sightlines"] == pytest.approx(
        {"n_a": 1, "n_b": 1, "n_shared": 1, "n_only_a": 0, "n_only_b": 0,
         "shared_frac_of_a": 1.0, "shared_frac_of_b": 1.0, "jaccard": 1.0})
    assert strict["restrict_to_common_sightlines"] is True


def test_restrict_to_common_keeps_sightlines_that_fail_the_cut_in_one_run():
    # 1002 is in BOTH raw catalogs but its B-side absorber fails NHI>=20.3.
    # The restriction must KEEP it (both runs looked) and report it as A-only.
    A = mkcat([1001, 1002], [2.5, 2.6], [20.5, 20.9])
    B = mkcat([1001, 1002], [2.5, 2.6], [20.5, 20.0])
    strict = compare(A, B, sel_kw=_SEL, restrict_to_common=True)
    assert strict["sightlines"]["n_only_a"] == 1
    assert strict["taxonomy"]["sightline_absent_from_b"] == 1


def test_compare_is_json_serialisable_with_no_nan():
    import json
    A = mkcat([1001, 1002], [2.5, 3.0], [20.5, 20.9])
    B = mkcat([1001], [2.5], [20.6])
    txt = json.dumps(compare(A, B, sel_kw=_SEL))
    assert "NaN" not in txt and "Infinity" not in txt


# --------------------------------------------------------------------------
# privacy guard
# --------------------------------------------------------------------------
def test_looks_real_discriminates_mock_from_survey_targetids():
    mock = np.array([2155, 470093323], dtype=np.int64)          # O(1e3-1e8)
    real = np.array([39633404408236241], dtype=np.int64)        # O(1e16)
    assert _looks_real(mock, threshold=DEFAULT_REAL_TID_THRESHOLD) is False
    assert _looks_real(real, threshold=DEFAULT_REAL_TID_THRESHOLD) is True
    assert _looks_real(mock, real, threshold=DEFAULT_REAL_TID_THRESHOLD) is True
