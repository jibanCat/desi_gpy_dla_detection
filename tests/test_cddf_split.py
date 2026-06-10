"""Tests for the deterministic sightline train/test split
(``CDDF_analysis.cddf_forward.split``).

WHY THIS EXISTS
---------------
The CDDF response matrix ``R`` is measured by recovered-vs-truth matching on a
mock run, and the forward inference is then VALIDATED by closure.  If ``R`` is
built and validated on the SAME sightlines, the closure is circular and proves
nothing.  The fix is a deterministic, sightline-level train/test split so the
R-build set (``BUILD``) and the validation set (``HELDOUT``) share NO sightlines,
plus a no-leakage guard that fails loudly on any overlap.

The split MUST be keyed on the immutable QSO ``TARGETID`` (NOT array position —
array order varies across loaders), so it is identical across
2LPT/London/Saclay/LOA and reproducible.  These tests pin that contract,
especially the "keyed on ID, not index" property (``test_role_keyed_on_target_id_not_index``).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
# Repo root on sys.path so ``CDDF_analysis`` resolves as a (namespace) package.
sys.path.insert(0, os.path.join(_HERE, ".."))

from CDDF_analysis.cddf_forward.split import (  # noqa: E402
    sightline_role,
    assign_roles,
    split_masks,
    SplitProvenance,
    assert_no_leakage,
)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_sightline_role_is_deterministic_across_calls(self):
        # Same (target_id, seed, frac_build) -> same role, every time.
        tid = 391857462013
        r1 = sightline_role(tid)
        r2 = sightline_role(tid)
        assert r1 == r2
        assert r1 in ("BUILD", "HELDOUT")

    def test_sightline_role_returns_only_valid_labels(self):
        for tid in range(1000):
            assert sightline_role(tid) in ("BUILD", "HELDOUT")

    def test_assign_roles_matches_elementwise_sightline_role(self):
        rng = np.random.default_rng(0)
        tids = rng.integers(1, 2**62, size=500, dtype=np.int64)
        roles = assign_roles(tids)
        for tid, role in zip(tids, roles):
            assert role == sightline_role(int(tid))

    def test_numpy_int64_and_python_int_give_same_role(self):
        # Canonical int64 byte representation must make np.int64 and python int
        # produce an identical role.
        for tid in (0, 1, 7, 12345, 391857462013, 2**61 + 3):
            assert sightline_role(int(tid)) == sightline_role(np.int64(tid))

    def test_role_keyed_on_target_id_not_index(self):
        # THE central contract: shuffling the input array must NOT change any
        # per-ID role.  Build a role map from the original order, shuffle, and
        # confirm each ID keeps its role regardless of its new position.
        rng = np.random.default_rng(42)
        tids = rng.choice(np.arange(1, 10_001, dtype=np.int64), size=2000, replace=False)
        roles_original = assign_roles(tids)
        role_by_id = {int(t): r for t, r in zip(tids, roles_original)}

        shuffled = tids.copy()
        rng.shuffle(shuffled)
        roles_shuffled = assign_roles(shuffled)
        for t, r in zip(shuffled, roles_shuffled):
            assert role_by_id[int(t)] == r, (
                "role changed under permutation -> keyed on index, not TARGETID"
            )


# --------------------------------------------------------------------------- #
# Disjointness / partition / fraction
# --------------------------------------------------------------------------- #
class TestPartition:
    def test_split_masks_disjoint_and_cover_all(self):
        rng = np.random.default_rng(1)
        tids = rng.integers(1, 2**62, size=3000, dtype=np.int64)
        build_mask, heldout_mask = split_masks(tids)
        assert build_mask.dtype == bool
        assert heldout_mask.dtype == bool
        # Disjoint: never both True.
        assert not np.any(build_mask & heldout_mask)
        # Cover all: always exactly one True.
        assert np.all(build_mask | heldout_mask)
        assert np.array_equal(build_mask, ~heldout_mask)

    def test_split_masks_agree_with_assign_roles(self):
        rng = np.random.default_rng(2)
        tids = rng.integers(1, 2**62, size=1000, dtype=np.int64)
        roles = assign_roles(tids)
        build_mask, heldout_mask = split_masks(tids)
        assert np.array_equal(build_mask, roles == "BUILD")
        assert np.array_equal(heldout_mask, roles == "HELDOUT")

    def test_build_fraction_close_to_frac_build(self):
        n = 10_000
        frac = 0.7
        # Distinct synthetic TARGETIDs.
        tids = np.arange(1, n + 1, dtype=np.int64)
        build_mask, _ = split_masks(tids, frac_build=frac)
        observed = build_mask.mean()
        tol = 4.0 * np.sqrt(frac * (1.0 - frac) / n)
        assert abs(observed - frac) < tol, (observed, frac, tol)


# --------------------------------------------------------------------------- #
# frac_build extremes
# --------------------------------------------------------------------------- #
class TestFracExtremes:
    def test_frac_build_one_is_all_build(self):
        tids = np.arange(1, 5001, dtype=np.int64)
        roles = assign_roles(tids, frac_build=1.0)
        assert np.all(roles == "BUILD")
        build_mask, heldout_mask = split_masks(tids, frac_build=1.0)
        assert np.all(build_mask)
        assert not np.any(heldout_mask)

    def test_frac_build_zero_is_all_heldout(self):
        tids = np.arange(1, 5001, dtype=np.int64)
        roles = assign_roles(tids, frac_build=0.0)
        assert np.all(roles == "HELDOUT")
        build_mask, heldout_mask = split_masks(tids, frac_build=0.0)
        assert not np.any(build_mask)
        assert np.all(heldout_mask)


# --------------------------------------------------------------------------- #
# Seed sensitivity
# --------------------------------------------------------------------------- #
class TestSeedSensitivity:
    def test_different_seed_gives_different_partition(self):
        n = 10_000
        tids = np.arange(1, n + 1, dtype=np.int64)
        a = assign_roles(tids, seed=20260609)
        b = assign_roles(tids, seed=99999999)
        # Different partition: a meaningful fraction of IDs flip role.
        flipped = np.mean(a != b)
        assert flipped > 0.05, flipped

    def test_different_seed_still_close_to_frac_build(self):
        n = 10_000
        frac = 0.7
        tids = np.arange(1, n + 1, dtype=np.int64)
        build_mask, _ = split_masks(tids, seed=99999999, frac_build=frac)
        observed = build_mask.mean()
        tol = 4.0 * np.sqrt(frac * (1.0 - frac) / n)
        assert abs(observed - frac) < tol, (observed, frac, tol)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
class TestProvenance:
    def _build_tids(self, tids, **kw):
        build_mask, _ = split_masks(tids, **kw)
        return tids[build_mask]

    def test_provenance_records_seed_and_frac(self):
        prov = SplitProvenance.from_target_ids(
            np.arange(1, 1001, dtype=np.int64), seed=20260609, frac_build=0.7
        )
        assert prov.seed == 20260609
        assert prov.frac_build == 0.7
        assert isinstance(prov.build_tids_hash, str)
        assert len(prov.build_tids_hash) > 0

    def test_build_tids_hash_stable_under_permutation(self):
        tids = np.arange(1, 5001, dtype=np.int64)
        build = self._build_tids(tids)
        # Permute the BUILD ids; hash must be identical (sorted internally).
        rng = np.random.default_rng(7)
        permuted = build.copy()
        rng.shuffle(permuted)
        h1 = SplitProvenance._hash_build_tids(build)
        h2 = SplitProvenance._hash_build_tids(permuted)
        assert h1 == h2

    def test_build_tids_hash_changes_when_build_set_changes(self):
        tids = np.arange(1, 5001, dtype=np.int64)
        build = self._build_tids(tids)
        h1 = SplitProvenance._hash_build_tids(build)
        # Drop one id -> different set -> different hash.
        h2 = SplitProvenance._hash_build_tids(build[1:])
        assert h1 != h2

    def test_provenance_hash_matches_from_target_ids(self):
        tids = np.arange(1, 5001, dtype=np.int64)
        prov = SplitProvenance.from_target_ids(tids)
        build = self._build_tids(tids)
        assert prov.build_tids_hash == SplitProvenance._hash_build_tids(build)


# --------------------------------------------------------------------------- #
# No-leakage guard
# --------------------------------------------------------------------------- #
class TestNoLeakageGuard:
    def test_passes_on_disjoint_sets(self):
        build = np.array([1, 2, 3, 4], dtype=np.int64)
        held = np.array([5, 6, 7, 8], dtype=np.int64)
        # Should NOT raise.
        assert assert_no_leakage(build, held) is None

    def test_raises_on_overlap(self):
        build = np.array([1, 2, 3, 4], dtype=np.int64)
        held = np.array([4, 5, 6], dtype=np.int64)
        with pytest.raises(ValueError):
            assert_no_leakage(build, held)

    def test_error_message_mentions_count(self):
        build = np.array([1, 2, 3, 4, 9], dtype=np.int64)
        held = np.array([3, 4, 9, 100], dtype=np.int64)  # 3 overlapping ids
        with pytest.raises(ValueError) as exc:
            assert_no_leakage(build, held)
        assert "3" in str(exc.value)

    def test_error_message_includes_ctx(self):
        build = np.array([1, 2], dtype=np.int64)
        held = np.array([2, 3], dtype=np.int64)
        with pytest.raises(ValueError) as exc:
            assert_no_leakage(build, held, ctx="London-0 R-build")
        assert "London-0 R-build" in str(exc.value)

    def test_real_split_has_no_leakage(self):
        # The split machinery itself must never leak.
        tids = np.arange(1, 20_001, dtype=np.int64)
        build_mask, heldout_mask = split_masks(tids)
        # Should NOT raise.
        assert_no_leakage(tids[build_mask], tids[heldout_mask], ctx="self-check")
