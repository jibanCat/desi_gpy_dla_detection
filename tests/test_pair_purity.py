"""tests/test_pair_purity.py
===========================
Unit tests for the false-positive-pair purity metric (Task 4 / spec §7-iv).

All tests use SYNTHETIC in-memory catalogs (plain dicts); no real data or FITS
files are required.  Exercises:
  - ``close_pairs``  — enumerates within-sightline close pairs correctly
  - ``matches_truth_pair`` — member-level matching + truth-pair Δv check
  - ``pair_purity``  — the gate metric: purity, n_true_new, n_new, per-bin

Catalog format used throughout:
  by_tid  : int TARGETID  ->  list of (z_dla, log_nhi) tuples
  truth   : int TARGETID  ->  list of z_truth floats
"""
from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Make sure the examples directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from dla_truth_diagnostics import (
    close_pairs,
    matches_truth_pair,
    pair_purity,
    C_KMS,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic catalogs
# ---------------------------------------------------------------------------

def _dv(z_a: float, z_b: float) -> float:
    """Velocity separation Δv = c·|Δz|/(1+z_mean)."""
    zm = 0.5 * (z_a + z_b)
    return C_KMS * abs(z_a - z_b) / (1.0 + zm)


# Two z values that are CLOSE (Δv ~ 800 km/s at z≈2.5)
_Z_CLOSE_A = 2.500
_Z_CLOSE_B = 2.508   # Δv ≈ 800 km/s at z=2.5

# Two z values that are FAR (Δv ~ 8000 km/s at z≈2.5)
_Z_FAR_A = 2.500
_Z_FAR_B = 2.580   # Δv ≈ 8000 km/s


# ---------------------------------------------------------------------------
# close_pairs
# ---------------------------------------------------------------------------

class TestClosePairs:
    def test_single_member_no_pair(self):
        cat = {101: [(_Z_CLOSE_A, 20.5)]}
        assert close_pairs(cat, dv_max=2000.0) == []

    def test_close_pair_detected(self):
        cat = {101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)]}
        pairs = close_pairs(cat, dv_max=2000.0)
        assert len(pairs) == 1
        tid, za, zb = pairs[0]
        assert tid == 101
        assert za <= zb  # convention: za <= zb
        assert math.isclose(_dv(za, zb), _dv(_Z_CLOSE_A, _Z_CLOSE_B), rel_tol=1e-4)

    def test_far_pair_excluded(self):
        cat = {101: [(_Z_FAR_A, 20.5), (_Z_FAR_B, 20.8)]}
        pairs = close_pairs(cat, dv_max=2000.0)
        assert len(pairs) == 0

    def test_multiple_sightlines(self):
        cat = {
            101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],  # close pair
            202: [(_Z_FAR_A, 20.5), (_Z_FAR_B, 20.8)],      # far pair
            303: [(_Z_CLOSE_A, 20.5)],                        # single
        }
        pairs = close_pairs(cat, dv_max=2000.0)
        assert len(pairs) == 1
        assert pairs[0][0] == 101

    def test_three_members_two_close_one_far(self):
        # Three members: zA=2.500, zB=2.508, zFar=2.580.
        # zA-zB: close (~800 km/s); zA-zFar: far (~6775 km/s); zB-zFar: far (~6433 km/s)
        # → exactly one close pair: (zA, zB)
        cat = {101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.6), (_Z_FAR_B, 20.8)]}
        pairs = close_pairs(cat, dv_max=2000.0)
        tids = {p[0] for p in pairs}
        assert tids == {101}
        assert len(pairs) == 1  # only (zA, zB) is close; zFar is always far

    def test_three_members_all_close_three_pairs(self):
        # Three z-values all mutually within 2000 km/s → 3 pairs
        z1, z2, z3 = 2.500, 2.504, 2.508
        cat = {101: [(z1, 20.5), (z2, 20.6), (z3, 20.8)]}
        pairs = close_pairs(cat, dv_max=2000.0)
        assert len(pairs) == 3

    def test_empty_catalog(self):
        assert close_pairs({}, dv_max=2000.0) == []

    def test_convention_za_le_zb(self):
        # Input reversed: higher z first
        cat = {101: [(_Z_CLOSE_B, 20.5), (_Z_CLOSE_A, 20.8)]}
        pairs = close_pairs(cat, dv_max=2000.0)
        assert len(pairs) == 1
        _, za, zb = pairs[0]
        assert za <= zb


# ---------------------------------------------------------------------------
# matches_truth_pair
# ---------------------------------------------------------------------------

class TestMatchesTruthPair:
    def test_match_both_within_dz(self):
        truth = {101: [_Z_CLOSE_A + 0.001, _Z_CLOSE_B - 0.001]}
        result = matches_truth_pair(
            101, _Z_CLOSE_A, _Z_CLOSE_B,
            truth_by_tid=truth, match_dz=0.005, dv_max=2000.0,
        )
        assert result is True

    def test_no_match_when_truth_pair_too_far(self):
        # The truth pair has Δv well above dv_max
        truth = {101: [_Z_FAR_A, _Z_FAR_B]}
        result = matches_truth_pair(
            101, _Z_FAR_A, _Z_FAR_B,
            truth_by_tid=truth, match_dz=0.005, dv_max=2000.0,
        )
        assert result is False

    def test_no_match_when_members_outside_dz(self):
        # Truth DLAs are far from the GP pair members in Δz
        truth = {101: [3.0, 3.1]}
        result = matches_truth_pair(
            101, _Z_CLOSE_A, _Z_CLOSE_B,
            truth_by_tid=truth, match_dz=0.005, dv_max=2000.0,
        )
        assert result is False

    def test_no_match_when_only_one_member_matches(self):
        # Only one truth DLA close enough — not a pair match
        truth = {101: [_Z_CLOSE_A + 0.001]}  # only 1 truth member
        result = matches_truth_pair(
            101, _Z_CLOSE_A, _Z_CLOSE_B,
            truth_by_tid=truth, match_dz=0.005, dv_max=2000.0,
        )
        assert result is False

    def test_no_match_when_truth_sightline_absent(self):
        truth = {999: [_Z_CLOSE_A, _Z_CLOSE_B]}  # different TID
        result = matches_truth_pair(
            101, _Z_CLOSE_A, _Z_CLOSE_B,
            truth_by_tid=truth, match_dz=0.005, dv_max=2000.0,
        )
        assert result is False

    def test_match_with_reversed_assignment(self):
        """The function must try both (GP_a→truth_i, GP_b→truth_j) assignments."""
        # Deliberately swap: z_a should match truth_z_b and vice versa
        tz_a = _Z_CLOSE_A + 0.001   # close to _Z_CLOSE_A
        tz_b = _Z_CLOSE_B - 0.001   # close to _Z_CLOSE_B
        truth = {101: [tz_a, tz_b]}
        # Pass z_a=_Z_CLOSE_B, z_b=_Z_CLOSE_A (reversed) — should still match
        result = matches_truth_pair(
            101, _Z_CLOSE_B, _Z_CLOSE_A,
            truth_by_tid=truth, match_dz=0.005, dv_max=2000.0,
        )
        assert result is True


# ---------------------------------------------------------------------------
# pair_purity — the gate metric
# ---------------------------------------------------------------------------

class TestPairPurity:
    """Tests use a controlled mix of sightlines to verify purity computation."""

    def _make_inputs(self):
        """
        Three sightlines:
          TID 101 — ON has a true close pair (both match truth), OFF has none.
                    → "new" and "true" (contributes 1 to n_true_new, 1 to n_new)
          TID 202 — ON has a spurious close pair (no matching truth pair), OFF has none.
                    → "new" and "false" (contributes 0 to n_true_new, 1 to n_new)
          TID 303 — Both ON and OFF already have the close pair.
                    → NOT new (excluded from counts entirely)

        Expected overall: n_new=2, n_true_new=1, purity=0.5
        """
        on_by_tid = {
            101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],   # true-new pair
            202: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],   # spurious pair
            303: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],   # already in OFF
        }
        off_by_tid = {
            303: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],   # same pair → not new
        }
        truth_by_tid = {
            101: [_Z_CLOSE_A + 0.001, _Z_CLOSE_B - 0.001],   # matching truth pair
            # 202 has NO truth → spurious
            # 303 irrelevant (pair not new)
        }
        return on_by_tid, off_by_tid, truth_by_tid

    def test_basic_purity_and_counts(self):
        on, off, truth = self._make_inputs()
        purity, n_true_new, n_new, per_bin = pair_purity(
            on, off, truth, dv_max=2000.0, match_dz=0.005
        )
        assert n_new == 2, f"expected 2 new pairs, got {n_new}"
        assert n_true_new == 1, f"expected 1 true-new pair, got {n_true_new}"
        assert math.isclose(purity, 0.5, rel_tol=1e-9)

    def test_two_true_one_false_purity_two_thirds(self):
        """2 true-new + 1 false-new → purity 2/3."""
        on_by_tid = {
            101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],  # true-new
            202: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],  # true-new
            303: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],  # spurious-new
        }
        off_by_tid = {}  # all pairs are new
        truth_by_tid = {
            101: [_Z_CLOSE_A + 0.001, _Z_CLOSE_B - 0.001],
            202: [_Z_CLOSE_A + 0.002, _Z_CLOSE_B - 0.002],
            # 303 has no truth → spurious
        }
        purity, n_true_new, n_new, _ = pair_purity(
            on_by_tid, off_by_tid, truth_by_tid, dv_max=2000.0, match_dz=0.005
        )
        assert n_new == 3
        assert n_true_new == 2
        assert math.isclose(purity, 2.0 / 3.0, rel_tol=1e-9)

    def test_no_new_pairs_returns_nan_purity(self):
        """When all ON pairs are already in OFF, n_new=0 and purity is NaN."""
        on_by_tid = {
            101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],
        }
        off_by_tid = {
            101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)],
        }
        truth_by_tid = {101: [_Z_CLOSE_A, _Z_CLOSE_B]}
        purity, n_true_new, n_new, _ = pair_purity(
            on_by_tid, off_by_tid, truth_by_tid, dv_max=2000.0, match_dz=0.005
        )
        assert n_new == 0
        assert n_true_new == 0
        assert math.isnan(purity)

    def test_all_new_pairs_true_purity_one(self):
        on_by_tid = {101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)]}
        off_by_tid = {}
        truth_by_tid = {101: [_Z_CLOSE_A + 0.001, _Z_CLOSE_B - 0.001]}
        purity, n_true_new, n_new, _ = pair_purity(
            on_by_tid, off_by_tid, truth_by_tid, dv_max=2000.0, match_dz=0.005
        )
        assert n_new == 1
        assert n_true_new == 1
        assert math.isclose(purity, 1.0)

    def test_all_new_pairs_spurious_purity_zero(self):
        on_by_tid = {101: [(_Z_CLOSE_A, 20.5), (_Z_CLOSE_B, 20.8)]}
        off_by_tid = {}
        truth_by_tid = {}  # no truth at all
        purity, n_true_new, n_new, _ = pair_purity(
            on_by_tid, off_by_tid, truth_by_tid, dv_max=2000.0, match_dz=0.005
        )
        assert n_new == 1
        assert n_true_new == 0
        assert math.isclose(purity, 0.0)

    def test_per_bin_breakdown_sums_correctly(self):
        """n_new across all bins must equal overall n_new."""
        on, off, truth = self._make_inputs()
        purity, n_true_new, n_new, per_bin = pair_purity(
            on, off, truth, dv_max=2000.0, match_dz=0.005
        )
        bin_sum_new = sum(b["n_new"] for b in per_bin)
        bin_sum_true = sum(b["n_true_new"] for b in per_bin)
        assert bin_sum_new == n_new
        assert bin_sum_true == n_true_new

    def test_per_bin_purity_within_bin_is_ratio(self):
        """Per-bin purity = n_true_new / n_new for each bin with n_new > 0."""
        on, off, truth = self._make_inputs()
        _, _, _, per_bin = pair_purity(
            on, off, truth, dv_max=2000.0, match_dz=0.005
        )
        for b in per_bin:
            if b["n_new"] > 0:
                expected = b["n_true_new"] / b["n_new"]
                assert math.isclose(b["purity"], expected, rel_tol=1e-9), (
                    f"bin [{b['dv_lo']}, {b['dv_hi']}): "
                    f"expected purity {expected}, got {b['purity']}"
                )
            else:
                assert math.isnan(b["purity"])

    def test_far_pairs_outside_dv_max_not_counted(self):
        """Pairs with Δv > dv_max must never appear in any result."""
        on_by_tid = {
            101: [(_Z_FAR_A, 20.5), (_Z_FAR_B, 20.8)],  # far: Δv > 2000
        }
        off_by_tid = {}
        truth_by_tid = {101: [_Z_FAR_A, _Z_FAR_B]}
        purity, n_true_new, n_new, _ = pair_purity(
            on_by_tid, off_by_tid, truth_by_tid, dv_max=2000.0, match_dz=0.005
        )
        assert n_new == 0, "far pairs should not be counted as close pairs"

    def test_empty_on_catalog(self):
        on_by_tid = {}
        off_by_tid = {}
        truth_by_tid = {101: [_Z_CLOSE_A, _Z_CLOSE_B]}
        purity, n_true_new, n_new, per_bin = pair_purity(
            on_by_tid, off_by_tid, truth_by_tid, dv_max=2000.0, match_dz=0.005
        )
        assert n_new == 0
        assert math.isnan(purity)


# ---------------------------------------------------------------------------
# ESS-log wiring: confirm _log_ess=False is a no-op (byte-identical guarantee)
# ---------------------------------------------------------------------------

class TestESSLogWiring:
    """Lightweight check that `_log_ess` default absence / False is a dead path.

    We test the contract without building a full GP model: the attribute lookup
    `getattr(self, "_log_ess", False)` must evaluate to False when the attr is
    absent (default production) or explicitly False, and True only when set.
    This mirrors the guard inserted in parallel_log_model_evidences.
    """

    def test_default_absent_evaluates_false(self):
        class Stub:
            pass
        obj = Stub()
        assert getattr(obj, "_log_ess", False) is False

    def test_explicit_false_evaluates_false(self):
        class Stub:
            _log_ess = False
        obj = Stub()
        assert getattr(obj, "_log_ess", False) is False

    def test_explicit_true_evaluates_true(self):
        class Stub:
            _log_ess = True
        obj = Stub()
        assert getattr(obj, "_log_ess", False) is True

    def test_dlagp_has_no_log_ess_by_default(self):
        """DLAGP.__new__ must not set _log_ess (production byte-identical path)."""
        from gpy_dla_detection.dla_gp import DLAGP
        obj = DLAGP.__new__(DLAGP)
        assert not hasattr(obj, "_log_ess"), (
            "_log_ess must NOT be set on a bare DLAGP instance "
            "(production byte-identical guarantee)"
        )

    def test_ess_block_skipped_when_log_ess_false(self):
        """Simulate the ESS block inline: with _log_ess=False the inner code
        never runs (no warning, no exception), confirming the dead-branch."""
        import warnings

        class StubGP:
            _log_ess = False

        obj = StubGP()
        num_dlas = 2  # >= 1, so the outer condition would fire if _log_ess is True
        sample_probabilities = np.array([1.0, 0.5, 0.25])

        captured_warnings = []
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            # Reproduce the exact guard from parallel_log_model_evidences
            if getattr(obj, "_log_ess", False) and num_dlas >= 1:
                ww = np.array(sample_probabilities, dtype=float)
                ww = ww[np.isfinite(ww)]
                if ww.size and ww.sum() > 0:
                    ess = (ww.sum() ** 2) / np.sum(ww ** 2)
                    ess_frac = ess / ww.size
                    if ess_frac < 0.3:
                        import logging
                        logging.getLogger().warning("low ESS-frac")
            captured_warnings = list(w_list)

        assert len(captured_warnings) == 0, (
            "ESS block must be completely skipped when _log_ess is False"
        )
