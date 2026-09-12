"""Fast, data-free unit tests of the grid / host-slot conventions used by
``validation/fp_ladder/build_fp_census.py``.

These pin the two conventions the census depends on:

  1. the pack's (c, k, s) index convention -- half-open [lo, hi), obtained as
     ``searchsorted(edges, x, side="right") - 1``, SNR clipped into [0, n_s-1],
     rows outside the N-hat / z windows DROPPED;
  2. the true-host N_HI slot partition, including its 1e-9 edge tolerance and
     the NaN -> "hostless" rule.

Run: ``pytest tests/test_fp_census_grid.py`` in the ``gpdla`` env.
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "validation", "fp_ladder"))

import build_fp_census as B  # noqa: E402


# --------------------------------------------------------------------------
# the grid edges, rebuilt here exactly as extract_pack builds them
# --------------------------------------------------------------------------
NHAT_EDGES = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 3)
ZF_EDGES = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 3)
SNR_EDGES = np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf])


def _reference_index(edges, x):
    """Brute-force half-open bucket search, independent of searchsorted."""
    out = []
    for v in np.atleast_1d(np.asarray(x, float)):
        i = -1
        for j in range(len(edges) - 1):
            if edges[j] <= v < edges[j + 1]:
                i = j
                break
        if v >= edges[-1]:
            i = len(edges) - 1
        out.append(i)
    return np.asarray(out)


def test_edge_counts():
    assert len(NHAT_EDGES) - 1 == 29
    assert len(ZF_EDGES) - 1 == 15
    assert len(SNR_EDGES) - 1 == 8


def test_bin_index_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.uniform(19.0, 22.9, 5000)
    assert np.array_equal(B.bin_index(NHAT_EDGES, x), _reference_index(NHAT_EDGES, x))
    z = rng.uniform(1.8, 3.7, 5000)
    assert np.array_equal(B.bin_index(ZF_EDGES, z), _reference_index(ZF_EDGES, z))


def test_bin_index_is_left_closed_right_open():
    # an exact left edge belongs to its own bin; the right edge belongs to the next
    assert B.bin_index(NHAT_EDGES, [19.5])[0] == 0
    assert B.bin_index(NHAT_EDGES, [19.6])[0] == 1
    assert B.bin_index(NHAT_EDGES, [19.5999999])[0] == 0
    assert B.bin_index(ZF_EDGES, [2.0])[0] == 0
    assert B.bin_index(ZF_EDGES, [3.4])[0] == 14


def test_bin_index_out_of_window_is_signalled_not_folded():
    # BELOW the first edge -> -1; AT/ABOVE the last edge -> n_bins (both are
    # dropped by bin_counts_cks, never clipped into an end bin)
    assert B.bin_index(NHAT_EDGES, [19.4999])[0] == -1
    assert B.bin_index(NHAT_EDGES, [22.4])[0] == 29
    assert B.bin_index(NHAT_EDGES, [25.0])[0] == 29
    assert B.bin_index(ZF_EDGES, [1.999])[0] == -1
    assert B.bin_index(ZF_EDGES, [3.5])[0] == 15


def test_snr_index_and_clip():
    # the SNR axis IS clipped (that is what makes stratum 7 open-topped)
    raw = B.bin_index(SNR_EDGES, [0.0, 0.5, 2.0, 2.0001, 6.999, 7.0, 1e9, np.inf])
    clipped = np.clip(raw, 0, 7)
    assert list(clipped) == [0, 0, 2, 2, 6, 7, 7, 7]
    # the op cut is S2N_RED > 2 STRICT, so strata 0 and 1 are empty by
    # construction -- a value of exactly 2.0 would land in stratum 2
    assert B.bin_index(SNR_EDGES, [2.0])[0] == 2


# --------------------------------------------------------------------------
# host-slot partition
# --------------------------------------------------------------------------
def test_host_slot_names_and_edges():
    assert B.HOST_SLOT_NAMES == ("host_17p2_19p0", "host_19p0_19p5",
                                 "host_19p5_19p7", "host_19p7_21p6",
                                 "host_ge_21p6")
    los = [lo for _, lo, _ in B.HOST_SLOTS]
    assert los == [17.2, 19.0, 19.5, 19.7, 21.6]
    # contiguous: each slot's top is the next slot's bottom
    for (_, _, hi), (_, lo, _) in zip(B.HOST_SLOTS[:-1], B.HOST_SLOTS[1:]):
        assert hi == lo
    assert np.isinf(B.HOST_SLOTS[-1][2])


def test_nan_is_hostless():
    m = B.host_slot_masks([np.nan, 20.0, np.nan])
    assert list(m["hostless"]) == [True, False, True]
    assert list(m["host_19p7_21p6"]) == [False, True, False]


def test_slots_partition_exactly():
    rng = np.random.default_rng(1)
    n = rng.uniform(17.2, 22.5, 20000)
    n[::7] = np.nan
    m = B.host_slot_masks(n)
    stacked = np.vstack([m[k] for k in ("hostless",) + B.HOST_SLOT_NAMES])
    assert np.all(stacked.sum(axis=0) == 1)
    assert int(stacked.sum()) == len(n)


def test_slot_boundaries_are_left_closed():
    vals = [17.2, 19.0, 19.5, 19.7, 21.6]
    m = B.host_slot_masks(vals)
    for i, name in enumerate(B.HOST_SLOT_NAMES):
        assert m[name][i], f"{vals[i]} should sit in {name}"
        assert sum(m[nm][i] for nm in B.HOST_SLOT_NAMES) == 1


def test_slot_epsilon_pushes_just_below_an_edge_upward():
    # DOCUMENTED asymmetry of the verified recipe: (n >= lo - EPS) & (n < hi - EPS)
    eps = B.SLOT_EPS
    m = B.host_slot_masks([19.5 - eps / 2, 19.5 - 10 * eps])
    assert m["host_19p5_19p7"][0] and not m["host_19p0_19p5"][0]
    assert m["host_19p0_19p5"][1] and not m["host_19p5_19p7"][1]


def test_below_the_census_floor_is_in_no_slot():
    # a finite host below 17.2 would break the partition -- the builder's
    # partition guard is what catches it, so make sure it IS catchable
    m = B.host_slot_masks([17.0])
    assert not m["hostless"][0]
    assert sum(m[nm][0] for nm in B.HOST_SLOT_NAMES) == 0


def test_pack_path_helper():
    p = B.default_pack("london0")
    assert p.endswith("scanpack_london0_b300.npz")
    assert B.PACK_OF_RECORD_DIR in p


def test_controls_agree_with_the_contract_file():
    """No data needed: parses matching_contract.py with ast (it cannot be
    imported under `gpdla` -- its top-level import pulls in jax)."""
    if not os.path.exists(B._CONTRACT):
        pytest.skip("matching_contract.py not present")
    for fam in B.FAMILIES:
        checked = B.assert_controls_match_contract(fam)
        assert checked, f"no contract constants cross-read for {fam}"
    # the fine split must reconstruct the contract's combined P1 slot
    for fam in B.FAMILIES:
        c = B.CONTROLS[fam]
        assert c["host_19p0_19p5"] + c["host_19p5_19p7"] == c["host_19p0_19p7"]
        assert (c["hostless_172"] + c["host_17p2_19p0"] + c["host_19p0_19p7"]
                + c["host_19p7_21p6"] + c["host_ge_21p6"]) == c["n_on_grid_172"]


def test_bin_index_agrees_with_extract_pack_idx():
    """Cross-check against the committed implementation, if it can be loaded."""
    try:
        ep = B._load_ep()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"extract_pack not loadable here: {exc}")
    assert np.array_equal(ep.NHAT_EDGES, NHAT_EDGES)
    assert np.array_equal(ep.ZF_EDGES, ZF_EDGES)
    assert np.array_equal(ep.SNR_EDGES, SNR_EDGES)
    rng = np.random.default_rng(2)
    for edges in (NHAT_EDGES, ZF_EDGES, SNR_EDGES):
        x = rng.uniform(float(edges[0]) - 1.0, float(np.nanmax(edges[:-1])) + 1.0, 2000)
        assert np.array_equal(B.bin_index(edges, x), ep._idx(edges, x))
