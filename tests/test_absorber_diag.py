"""Unit tests for the absorber-side operator-forensics helpers
(``validation/absorber_diag/binning.py``).

VALIDATION-ONLY tooling: these tests pin the binning / normalisation
conventions the forensics rely on.  They are fast (no data, no jax, no
sampler) and run under either environment.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "validation", "absorber_diag"))

from binning import (bin_index, in_range, coarse_block_sum,          # noqa: E402
                     row_normalise, safe_ratio, migration_moments,
                     leakage_fractions, threshold_weights)


# --------------------------------------------------------------------------
# bin_index: the pack's half-open [lo, hi) right-searchsorted convention
# --------------------------------------------------------------------------
def test_bin_index_half_open_and_edges():
    e = np.array([19.5, 19.6, 19.7])
    assert list(bin_index(e, [19.5, 19.55, 19.6, 19.69])) == [0, 0, 1, 1]
    # below the first edge -> -1; at/above the last edge -> len(e)-1
    assert bin_index(e, [19.49])[0] == -1
    assert bin_index(e, [19.7])[0] == 2
    assert bin_index(e, [21.0])[0] == 2


def test_bin_index_matches_extract_pack_idx_semantics():
    """The committed ``_idx`` is searchsorted(side='right') - 1; reproduce it."""
    e = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 3)
    x = np.array([19.5, 19.5001, 20.0, 20.29999, 22.39, 22.4, 19.0])
    ref = np.searchsorted(e, x, side="right") - 1
    assert np.array_equal(bin_index(e, x), ref)


def test_in_range():
    assert list(in_range([-1, 0, 3, 4], 4)) == [False, True, True, False]


# --------------------------------------------------------------------------
# coarse_block_sum: the pack's kz_to_K = repeat([0,1,2], 5)
# --------------------------------------------------------------------------
def test_coarse_block_sum_conserves_and_groups():
    kz = np.repeat([0, 1, 2], 5)
    a = np.arange(2 * 15 * 3, dtype=float).reshape(2, 15, 3)
    out = coarse_block_sum(a, kz, axis=1)
    assert out.shape == (2, 3, 3)
    assert np.isclose(out.sum(), a.sum())
    assert np.allclose(out[:, 1, :], a[:, 5:10, :].sum(axis=1))


def test_coarse_block_sum_rejects_wrong_axis_length():
    with pytest.raises(ValueError):
        coarse_block_sum(np.zeros((3, 4)), np.repeat([0, 1, 2], 5), axis=1)


# --------------------------------------------------------------------------
# row_normalise: unit mass + explicit empty-row bookkeeping
# --------------------------------------------------------------------------
def test_row_normalise_unit_mass_and_empty_rows():
    m = np.array([[1.0, 3.0], [0.0, 0.0]])
    out, had = row_normalise(m, axis=1)
    assert np.allclose(out[0], [0.25, 0.75])
    assert np.allclose(out[1], [0.0, 0.0])
    assert list(had) == [True, False]


def test_row_normalise_fallback_is_used_only_on_empty_rows():
    m = np.array([[2.0, 2.0], [0.0, 0.0]])
    fb = np.array([[0.9, 0.1], [0.9, 0.1]])
    out, had = row_normalise(m, axis=1, fallback=fb)
    assert np.allclose(out[0], [0.5, 0.5])
    assert np.allclose(out[1], [0.9, 0.1])
    assert not had[1]


# --------------------------------------------------------------------------
# safe_ratio: never NaN/inf, zero denominators go to the fill
# --------------------------------------------------------------------------
def test_safe_ratio_guards_zero_denominator():
    out = safe_ratio(np.array([1.0, 2.0, 0.0]), np.array([2.0, 0.0, 0.0]))
    assert np.allclose(out, [0.5, 0.0, 0.0])
    assert np.all(np.isfinite(out))


def test_safe_ratio_atol_zero_style_exactness():
    """A wrong normalisation must not survive: compare with atol=0."""
    num = np.array([1e-12, 2e-12])
    den = np.array([2e-12, 2e-12])
    assert np.allclose(safe_ratio(num, den), [0.5, 1.0], atol=0.0, rtol=1e-12)


# --------------------------------------------------------------------------
# migration_moments
# --------------------------------------------------------------------------
def test_migration_moments_on_a_symmetric_row():
    x = np.array([0.0, 1.0, 2.0])
    mass = np.array([1.0, 2.0, 1.0])
    mean, sd, skew = migration_moments(mass, x)
    assert np.isclose(mean, 1.0, atol=0.0, rtol=1e-12)
    assert np.isclose(sd, np.sqrt(0.5), atol=0.0, rtol=1e-12)
    assert np.isclose(skew, 0.0, atol=1e-12)


def test_migration_moments_unnormalised_equals_normalised():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mass = np.array([1.0, 4.0, 2.0, 1.0])
    a = migration_moments(mass, x)
    b = migration_moments(mass / mass.sum(), x)
    for u, v in zip(a, b):
        assert np.isclose(u, v, atol=0.0, rtol=1e-12)


def test_migration_moments_narrow_row_has_no_variance_floor():
    """A 0.1-dex-wide migration row must not be widened by a hidden floor.

    (Mutation-tested: replacing ``np.maximum(var, 0.0)`` by ``np.maximum(var,
    1e-3)`` — a floor of 0.032 dex, ~30 % of the measured high-N width —
    survives every other test in this file but fails this one.)
    """
    x = np.array([0.0, 0.1, 0.2])
    mass = np.array([1.0, 0.0, 1.0])          # sd = 0.1 exactly
    _, sd, _ = migration_moments(mass, x)
    assert np.isclose(sd, 0.1, atol=0.0, rtol=1e-12)
    mass2 = np.array([0.0, 1.0, 0.0])         # sd = 0 exactly
    _, sd2, _ = migration_moments(mass2, x)
    assert sd2 == 0.0


def test_migration_moments_empty_row_is_nan():
    mean, sd, skew = migration_moments(np.zeros(3), np.arange(3.0))
    assert np.isnan(mean) and np.isnan(sd) and np.isnan(skew)


def test_migration_moments_vectorised_over_leading_axes():
    x = np.arange(4.0)
    mass = np.stack([np.array([1.0, 1.0, 0.0, 0.0]),
                     np.array([0.0, 0.0, 1.0, 1.0])])
    mean, _, _ = migration_moments(mass, x)
    assert np.allclose(mean, [0.5, 2.5], atol=0.0, rtol=1e-12)


# --------------------------------------------------------------------------
# leakage_fractions: down / inside / up must sum to one
# --------------------------------------------------------------------------
def test_leakage_fractions_partition():
    edges = np.array([19.5, 19.7, 19.9, 20.1])
    mass = np.array([2.0, 5.0, 3.0])
    out = leakage_fractions(mass, edges, 19.7, 19.9)
    assert np.allclose(out, [0.2, 0.5, 0.3], atol=0.0, rtol=1e-12)
    assert np.isclose(out.sum(), 1.0, atol=0.0, rtol=1e-12)


def test_leakage_fractions_empty_row_is_nan():
    edges = np.array([19.5, 19.7, 19.9])
    out = leakage_fractions(np.zeros(2), edges, 19.5, 19.7)
    assert np.all(np.isnan(out))


# --------------------------------------------------------------------------
# threshold_weights: the committed open-topped dex weights
# --------------------------------------------------------------------------
def test_threshold_weights_match_the_committed_definition():
    nt = np.array([19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5])
    u = threshold_weights(nt, 20.0, 19.5)
    # bins fully below the threshold get 0; the straddling bin gets its part
    assert np.allclose(u[:4], 0.0, atol=0.0)
    assert np.isclose(u[4], 0.1, atol=1e-12)      # [19.9, 20.1) above 20.0
    assert np.isclose(u[5], 0.2, atol=1e-12)
    assert np.isclose(u[6], 0.2, atol=1e-12)


def test_threshold_weights_zero_below_the_reporting_floor():
    nt = np.array([19.0, 19.2, 19.5, 19.7])
    # every centre below 19.5 must be masked out even if above the threshold
    u = threshold_weights(nt, 19.0, 19.5)
    assert u[0] == 0.0 and u[1] == 0.0
    assert np.isclose(u[2], 0.2, atol=1e-12)      # centre 19.6 >= 19.5
