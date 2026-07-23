"""tests/test_zresolved_pc.py
============================
Unit tests for the z-resolved purity/completeness reduction
(spec: notes repo `2026-07-22_zresolved_pc_design.md`).

Exercises the three new helpers in ``examples/molly_faithful_pc_plots.py``:
  - ``purity_z_bins``        — P per (z_DLA x z_QSO) cell, binned on PREDICTED z_DLA
  - ``completeness_z_bins``  — C per cell, binned on TRUTH z_DLA on BOTH sides
  - ``wilson_interval``      — 68.3% binomial score interval

All catalogs here are SYNTHETIC in-memory dicts of numpy arrays; no FITS, no
mock data, no network.

The load-bearing test is ``test_marginalization_reproduces_headline``: summing
the z-binned counts over every cell must reproduce the existing, already-validated
``purity_min`` / ``completeness_min`` headline EXACTLY. It is asserted on integer
counts, never on floats — see the note in that test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
pytest.importorskip("fitsio")
from molly_faithful_pc_plots import (  # noqa: E402
    completeness_min,
    completeness_z_bins,
    purity_min,
    purity_z_bins,
    wilson_interval,
)

# Binning fixed by the spec (matches Track-C / decompose_r0_zstructure).
ZDLA_EDGES = [2.0, 2.5, 3.0, 3.5]
ZQSO_EDGES = [2.0, 2.5, 3.0, 4.25]

# Base cuts; every synthetic row below is built to pass these, so the tests
# isolate the z binning rather than re-testing the existing cut bundle.
MIN_SNR = 2.0
MIN_PRED_NHI = 20.3
MIN_TRUE_NHI = 20.3
MIN_GOODNESS = 0.99


def _cat():
    """8 detections, all passing the base cuts, laid out so each populated
    (z_DLA x z_QSO) cell has hand-countable contents.

    z_DLA < z_QSO always, as the forest window requires.

      cell (zdla[2.0,2.5), zqso[2.0,2.5))  rows 0,1     -> 2 det, 1 TP
      cell (zdla[2.5,3.0), zqso[2.5,3.0))  rows 2,3,4   -> 3 det, 2 TP
      cell (zdla[3.0,3.5), zqso[3.0,4.25)) rows 5,6,7   -> 3 det, 2 TP
    """
    z_dla = np.array([2.1, 2.2, 2.6, 2.7, 2.9, 3.1, 3.2, 3.4])
    z_qso = np.array([2.3, 2.3, 2.8, 2.8, 2.95, 3.5, 3.5, 3.9])
    tp = np.array([True, False, True, True, False, True, False, True])
    # Z_TRUE is the matched truth z_DLA; NaN exactly where the row is not a TP.
    z_true = np.where(tp, z_dla, np.nan)
    n = len(z_dla)
    cat = {
        "S2N_RED": np.full(n, 5.0),
        "NHI": np.full(n, 21.0),
        "P_DLA": np.full(n, 0.999),
        "Z_DLA": z_dla,
        "Z_QSO": z_qso,
        "Z_TRUE": z_true,
        "NHI_TRUE": np.where(tp, 21.0, np.nan),
    }
    return cat, tp


def _truth():
    """10 truth systems, all passing the base cuts.

      cell (zdla[2.0,2.5), zqso[2.0,2.5))  -> 3   (1 recovered -> C = 1/3)
      cell (zdla[2.5,3.0), zqso[2.5,3.0))  -> 4   (2 recovered -> C = 1/2)
      cell (zdla[3.0,3.5), zqso[3.0,4.25)) -> 3   (2 recovered -> C = 2/3)
    """
    z_dla = np.array([2.1, 2.15, 2.4, 2.6, 2.7, 2.75, 2.85, 3.1, 3.3, 3.4])
    z_qso = np.array([2.3, 2.3, 2.45, 2.8, 2.8, 2.9, 2.95, 3.5, 3.6, 3.9])
    n = len(z_dla)
    return {
        "S2N_RED": np.full(n, 5.0),
        "NHI": np.full(n, 21.0),
        "Z_DLA": z_dla,
        "Z_QSO": z_qso,
    }


def _cells():
    for i in range(len(ZDLA_EDGES) - 1):
        for j in range(len(ZQSO_EDGES) - 1):
            yield (ZDLA_EDGES[i], ZDLA_EDGES[i + 1],
                   ZQSO_EDGES[j], ZQSO_EDGES[j + 1])


# -----------------------------------------------------------------------------
# purity
# -----------------------------------------------------------------------------
def test_purity_z_bins_exact_counts():
    cat, tp = _cat()
    expect = {
        (2.0, 2.5, 2.0, 2.5): (1, 2),
        (2.5, 3.0, 2.5, 3.0): (2, 3),
        (3.0, 3.5, 3.0, 4.25): (2, 3),
    }
    for cell in _cells():
        ntp, ndet, pur = purity_z_bins(
            cat, tp, MIN_SNR, MIN_PRED_NHI, MIN_GOODNESS, *cell)
        e_tp, e_det = expect.get(cell, (0, 0))
        assert (ntp, ndet) == (e_tp, e_det), f"cell {cell}"
        if e_det:
            assert pur == pytest.approx(e_tp / e_det, abs=0.0)
        else:
            assert np.isnan(pur)


def test_purity_bins_on_predicted_z_not_truth_z():
    """A false positive has no truth z, so purity MUST bin on predicted Z_DLA.

    Move one FP's predicted z into a different cell while leaving everything
    else alone; the cell occupancies must follow the predicted z.
    """
    cat, tp = _cat()
    cat = dict(cat)
    cat["Z_DLA"] = cat["Z_DLA"].copy()
    cat["Z_DLA"][1] = 2.6          # row 1 is an FP: 2.2 -> 2.6, crosses a bin edge
    _, ndet_low, _ = purity_z_bins(
        cat, tp, MIN_SNR, MIN_PRED_NHI, MIN_GOODNESS, 2.0, 2.5, 2.0, 2.5)
    assert ndet_low == 1           # was 2
    # It lands in zdla[2.5,3.0) but its Z_QSO (2.3) keeps it in zqso[2.0,2.5).
    _, ndet_mid, _ = purity_z_bins(
        cat, tp, MIN_SNR, MIN_PRED_NHI, MIN_GOODNESS, 2.5, 3.0, 2.0, 2.5)
    assert ndet_mid == 1


# -----------------------------------------------------------------------------
# completeness
# -----------------------------------------------------------------------------
def test_completeness_z_bins_exact_counts():
    cat, tp = _cat()
    truth = _truth()
    expect = {
        (2.0, 2.5, 2.0, 2.5): (1, 3),
        (2.5, 3.0, 2.5, 3.0): (2, 4),
        (3.0, 3.5, 3.0, 4.25): (2, 3),
    }
    for cell in _cells():
        nf, nt, comp = completeness_z_bins(
            cat, tp, MIN_SNR, MIN_TRUE_NHI, MIN_PRED_NHI, MIN_GOODNESS,
            truth, *cell)
        e_f, e_t = expect.get(cell, (0, 0))
        assert (nf, nt) == (e_f, e_t), f"cell {cell}"
        if e_t:
            assert comp == pytest.approx(e_f / e_t, abs=0.0)
        else:
            assert np.isnan(comp)


def test_completeness_not_inflated_by_nhi_scatter_across_floor():
    """Eddington bias across the N_HI floor must not push completeness above 1.

    A system whose TRUTH N_HI sits below the floor but whose FIT scattered above
    it is a legitimate detection, but it is NOT in the >=floor truth population.
    If the numerator is gated only on predicted N_HI it enters while the
    denominator excludes it, and C exceeds unity.

    This is not hypothetical: on 2LPT-0 with truth loaded at 20.0 and a 20.3
    floor, per-cell completeness reached 1.139 before the numerator was gated on
    NHI_TRUE.
    """
    # One TP with truth N_HI = 20.1 (below floor) but predicted 20.5 (above).
    cat = {
        "S2N_RED": np.array([5.0]),
        "NHI": np.array([20.5]),          # fit scattered ABOVE the floor
        "NHI_TRUE": np.array([20.1]),     # truth is BELOW the floor
        "P_DLA": np.array([0.999]),
        "Z_DLA": np.array([2.2]),
        "Z_QSO": np.array([2.3]),
        "Z_TRUE": np.array([2.2]),
    }
    tp = np.array([True])
    # The truth population >= 20.3 in this cell is empty; the 20.1 system is
    # present in the truth table but below the floor.
    truth = {
        "S2N_RED": np.array([5.0]),
        "NHI": np.array([20.1]),
        "Z_DLA": np.array([2.2]),
        "Z_QSO": np.array([2.3]),
    }
    n_found, n_true, comp = completeness_z_bins(
        cat, tp, MIN_SNR, 20.3, 20.3, MIN_GOODNESS, truth,
        2.0, 2.5, 2.0, 2.5)
    assert n_true == 0                 # nothing in truth is >= 20.3
    assert n_found == 0, ("a sub-floor truth system scattered above the floor "
                          "leaked into the completeness numerator")
    assert np.isnan(comp)


def test_completeness_numerator_uses_truth_z():
    """Numerator and denominator must live on the SAME variable (truth z_DLA).

    Perturbing a TP's PREDICTED z across a bin edge must not move it between
    completeness cells — only its truth z governs.
    """
    cat, tp = _cat()
    truth = _truth()
    before = completeness_z_bins(
        cat, tp, MIN_SNR, MIN_TRUE_NHI, MIN_PRED_NHI, MIN_GOODNESS,
        truth, 2.0, 2.5, 2.0, 2.5)
    cat = dict(cat)
    cat["Z_DLA"] = cat["Z_DLA"].copy()
    cat["Z_DLA"][0] = 2.9          # row 0 is a TP; predicted z jumps a bin
    after = completeness_z_bins(
        cat, tp, MIN_SNR, MIN_TRUE_NHI, MIN_PRED_NHI, MIN_GOODNESS,
        truth, 2.0, 2.5, 2.0, 2.5)
    assert before == after


# -----------------------------------------------------------------------------
# the load-bearing test
# -----------------------------------------------------------------------------
def test_marginalization_reproduces_headline():
    """Summing every z cell must reproduce the existing headline EXACTLY.

    This ties the new reduction to the already-validated `purity_min` /
    `completeness_min` path: if the z binning double-counts, drops edge cases,
    or uses a different mask bundle, this test fails.

    Asserted on INTEGER COUNTS, deliberately. Per the 2026-07-14 finding, a
    default-atol `np.allclose` is vacuously true on CDDF-scale quantities and
    once let a gate pass while it held the wrong array. Integer equality cannot
    be fooled that way. The bins below span the full range of the synthetic
    data, which is what makes the identity hold.
    """
    cat, tp = _cat()
    truth = _truth()

    tot_tp = tot_det = 0
    tot_found = tot_true = 0
    for cell in _cells():
        ntp, ndet, _ = purity_z_bins(
            cat, tp, MIN_SNR, MIN_PRED_NHI, MIN_GOODNESS, *cell)
        nf, nt, _ = completeness_z_bins(
            cat, tp, MIN_SNR, MIN_TRUE_NHI, MIN_PRED_NHI, MIN_GOODNESS,
            truth, *cell)
        tot_tp += ntp
        tot_det += ndet
        tot_found += nf
        tot_true += nt

    h_tp, h_det, _ = purity_min(
        cat, tp, MIN_SNR, MIN_PRED_NHI, MIN_GOODNESS, None)
    h_found, h_true, _ = completeness_min(
        cat, tp, MIN_SNR, MIN_TRUE_NHI, MIN_PRED_NHI, MIN_GOODNESS, truth)

    assert (tot_tp, tot_det) == (h_tp, h_det)
    assert (tot_found, tot_true) == (h_found, h_true)


def test_zdla_bins_are_half_open():
    """Half-open [lo, hi) binning — a point exactly on an edge lands in the
    upper bin, exactly once. This is what makes the marginalization exact."""
    cat, tp = _cat()
    cat = dict(cat)
    cat["Z_DLA"] = cat["Z_DLA"].copy()
    cat["Z_DLA"][0] = 2.5          # sits exactly on the 2.5 edge
    _, n_lower, _ = purity_z_bins(
        cat, tp, MIN_SNR, MIN_PRED_NHI, MIN_GOODNESS, 2.0, 2.5, 2.0, 2.5)
    _, n_upper, _ = purity_z_bins(
        cat, tp, MIN_SNR, MIN_PRED_NHI, MIN_GOODNESS, 2.5, 3.0, 2.0, 2.5)
    assert n_lower == 1            # only row 1 remains
    assert n_upper == 1            # row 0 moved up, counted exactly once


# -----------------------------------------------------------------------------
# Wilson interval
# -----------------------------------------------------------------------------
def test_wilson_empty():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_wilson_bounds_and_containment():
    for k, n in [(0, 10), (10, 10), (5, 10), (1, 1), (0, 1), (3, 7)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0, (k, n, lo, hi)
        assert lo <= k / n <= hi, (k, n, lo, hi)


def test_wilson_never_degenerate_at_extremes():
    """The whole reason for Wilson over the normal approximation: at p = 0 and
    p = 1 the normal interval collapses to zero width, which would render as
    'measured exactly' on the figure. Wilson does not."""
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and hi0 > 0.0
    lo1, hi1 = wilson_interval(20, 20)
    assert hi1 == 1.0 and lo1 < 1.0


def test_wilson_handles_k_greater_than_n():
    """k > n is reachable for completeness -- a TP can survive the catalog cut
    (which uses min/max of predicted AND truth z) while its truth row was
    dropped by the truth cut (truth z only), so a cell can hold more matched
    systems than truth systems. The interval must stay real-valued rather than
    returning NaN from a negative sqrt."""
    lo, hi = wilson_interval(5, 4)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_symmetric_under_reflection():
    lo, hi = wilson_interval(3, 10)
    lo_r, hi_r = wilson_interval(7, 10)
    assert lo == pytest.approx(1.0 - hi_r, abs=1e-12)
    assert hi == pytest.approx(1.0 - lo_r, abs=1e-12)
