"""Tests for the Lya-WG S/N predictive diagnostic read-out (PI ruling 2026-09-14b §5/§14).

These are CERTIFICATION tests for a deterministic read-out of a FROZEN model: they check
(i) that our recomputation of the frozen fold at the stored posterior-median draw
reproduces the value the production runner stored, (ii) that the finer breakdowns satisfy
the counting identities that make them a decomposition of that same number, and (iii) that
the blinding guard actually has power (a count-like value is rejected).

They read the frozen scratch products; they are skipped when those are not mounted.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "validation", "wg_diagnostic"))
import wg_predictive_breakdown as W  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(W.REAL_INPUTS),
                                reason="frozen scratch products not mounted")


@pytest.fixture(scope="module")
def real_run():
    runs = W.real_run_jsons()
    assert len(runs) == 16, f"expected the 16 real C1 runs, found {len(runs)}"
    return runs[0]


@pytest.fixture(scope="module")
def mock_run():
    runs = W.mock_run_jsons("2lpt0")
    assert len(runs) == 8, f"expected the 8 J=8 2lpt0 runs, found {len(runs)}"
    return runs[0]


def test_recomputed_snr_marginal_matches_stored_real(real_run):
    """THE GATE: the all-K S/N marginal we recompute equals the runner's stored
    `mu_over_obs_by_snr` to 1e-10.  Nothing finer is trusted until this passes."""
    d, n = W.verify_stored_snr(real_run, W.real_fixed(), atol=1e-10)
    assert n == 8 and d <= 1e-10


@pytest.mark.parametrize("family", W.MOCK_FAMILIES)
def test_recomputed_snr_marginal_matches_stored_mock(family):
    """Same gate on each mock family, whose fixed objects (including the truth-pinned P6b
    term from the census, `--fix P`) are rebuilt from the run's own argv."""
    p = W.mock_run_jsons(family)[0]
    d, n = W.verify_stored_snr(p, W.mock_fixed(p), atol=1e-10)
    assert n == 8 and d <= 1e-10


def test_gate_has_power_wrong_median_draw_is_detected(real_run):
    """POWER CHECK: the gate above must be able to FAIL.  Shifting the posterior-median
    draw index by one must move the recomputed S/N marginal well above the 1e-10 tolerance
    — otherwise the agreement would be vacuous."""
    import json
    orig = W._median_draw_index
    try:
        W._median_draw_index = lambda b: orig(b) + 1
        with pytest.raises(AssertionError):
            W.verify_stored_snr(real_run, W.real_fixed(), atol=1e-10)
    finally:
        W._median_draw_index = orig
    # and the stored object we compare against is really the runner's
    j = json.load(open(real_run))
    assert "mu_over_obs_by_snr" in (j.get("predictive_marginals") or {})


def test_coarse_K_breakdown_is_a_decomposition_counting_identity(real_run):
    """COUNTING IDENTITY: the observed counts in the three coarse-z blocks partition the
    observed counts of each S/N stratum, so the obs-weighted mean over K of the (S/N x K)
    ratios must return the all-K S/N ratio exactly."""
    mu3, obs3, kz, _ = W.fold_at_median(real_run, W.real_fixed())
    KK = int(kz.max()) + 1
    assert sorted(np.bincount(kz, minlength=KK).tolist()) and kz.min() == 0
    o_K = np.stack([obs3[:, kz == K, :].sum((0, 1)) for K in range(KK)])   # (KK, S)
    m_K = np.stack([mu3[:, kz == K, :].sum((0, 1)) for K in range(KK)])
    assert np.allclose(o_K.sum(0), obs3.sum((0, 1)), rtol=0, atol=0)       # partition of obs
    b = W.breakdown_one(real_run, W.real_fixed())
    live = b["live_snr"]
    recomposed = np.where(o_K.sum(0) > 0, m_K.sum(0) / np.where(o_K.sum(0) > 0, o_K.sum(0), 1.0), np.nan)
    assert np.allclose(recomposed[live], b["by_snr"][live], rtol=0, atol=1e-12)
    # and each cell of the (S/N x K) table is a ratio of the SAME fold
    for K in range(KK):
        cell = np.where(o_K[K] > 0, m_K[K] / np.where(o_K[K] > 0, o_K[K], 1.0), np.nan)
        assert np.allclose(cell[np.isfinite(cell)], b["by_snr_K"][K][np.isfinite(cell)], rtol=0, atol=1e-12)


def test_nhat_groups_partition_the_live_support(real_run):
    """COUNTING IDENTITY: the three observed-N-hat groups (19.5-20.0 / 20.0-20.3 / >=20.3)
    partition the pack's N-hat rows, so their obs-weighted combination returns the all-N-hat
    S/N ratio."""
    mu3, obs3, kz, cc = W.fold_at_median(real_run, W.real_fixed())
    masks = [(cc >= lo) & (cc < hi) for _, lo, hi in W.NHAT_GROUPS]
    stack = np.stack(masks)
    assert stack.sum(0).max() == 1, "the N-hat groups overlap"
    assert stack.sum(0).min() == 1, "the N-hat groups do not cover the pack's rows"
    o_g = np.stack([obs3[m].sum((0, 1)) for m in masks]); m_g = np.stack([mu3[m].sum((0, 1)) for m in masks])
    tot_o = o_g.sum(0)
    rec = np.where(tot_o > 0, m_g.sum(0) / np.where(tot_o > 0, tot_o, 1.0), np.nan)
    b = W.breakdown_one(real_run, W.real_fixed())
    assert np.allclose(rec[b["live_snr"]], b["by_snr"][b["live_snr"]], rtol=0, atol=1e-12)


def test_blinding_guard_rejects_absolute_values():
    """The delivered payload may contain ratios/fractions only.  A count or a dN/dX value
    must be rejected (and a legitimate ratio table must pass)."""
    ok = dict(by_snr=[1.14, 0.93], grp=dict(a=[0.98, np.nan]))
    assert W.assert_ratios_only(ok)
    with pytest.raises(AssertionError):
        W.assert_ratios_only(dict(by_snr=[1.14], counts=[12345.0]))
    with pytest.raises(AssertionError):
        W.assert_ratios_only(dict(dndx=0.0597), lo=0.5, hi=2.0)


def test_readout_is_read_only_no_frozen_object_is_written(real_run, tmp_path):
    """The read-out must not touch the frozen products: mtimes of the pack, the three fixed
    files and the run JSON are unchanged across a full breakdown."""
    fx = W.real_fixed()
    watch = [fx["pack"], fx["mg"], fx["c"],
             os.path.join(W.REAL_INPUTS, "mu_extra_P6bcal_real.npz"), real_run]
    before = [os.stat(p).st_mtime_ns for p in watch]
    W.breakdown_one(real_run, W.real_fixed())
    assert [os.stat(p).st_mtime_ns for p in watch] == before
