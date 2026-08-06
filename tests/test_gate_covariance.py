"""gate_covariance: the calibration-predictive Layer-B gate (Phase B, 2026-08-06).

Pins: group-edge alignment fail-loud; the covariance carrying genuine
cross-bin calibration structure (never per-bin quadrature); null behavior of
the frozen Mahalanobis gate (a null realization is not rejected; an injected
shift is); the delta-method transport variance agreeing with the resampling
ensemble; and frozen-seed determinism.
"""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

from CDDF_analysis.hbi_mcmc import gate_covariance as GC
from CDDF_analysis.hbi_mcmc.pack import synthetic_pack


@pytest.fixture(scope="module")
def pack():
    # a synthetic pack with a real FP share and nonzero eta so every factor
    # of the restored fold is exercised through the gate machinery
    return synthetic_pack(seed=23, fp_frac=0.2, fp_eta=0.1)


@pytest.fixture(scope="module")
def edges(pack):
    ne = np.asarray(pack.nhat_edges, float)
    # three contiguous groups over the pack's own grid (synthetic grids are
    # small; use thirds of the observed range, snapped to bin edges)
    idx = np.linspace(0, len(ne) - 1, 4).round().astype(int)
    return tuple((float(ne[a]), float(ne[b]))
                 for a, b in zip(idx[:-1], idx[1:]))


def test_group_aggregator_rejects_misaligned_edges(pack):
    ne = np.asarray(pack.nhat_edges, float)
    bad_hi = float(0.5 * (ne[1] + ne[2]))          # mid-bin: truncates a bin
    with pytest.raises(ValueError, match="does not align"):
        GC.group_aggregator(pack, ((float(ne[0]), bad_hi),))


def test_covariance_has_cross_group_calibration_structure(pack, edges):
    """The shared calibration amplitude mode must induce POSITIVE cross-group
    covariance in the mu(n0*) part — the structure per-bin quadrature cannot
    represent."""
    cov = GC.estimate_covariance(pack, group_edges=edges, n_draws=400, seed=1)
    C = cov.matrix
    # positive definite, exactly invertible at this dimension
    ev = np.linalg.eigvalsh(C)
    assert ev.min() > 0
    assert cov.condition_number < GC.MAX_CONDITION_NUMBER
    # the calibration-only part, measured directly from the mu(n0*) draws
    # (never inferred as a difference of two MC-noisy large numbers)
    has_fp = np.asarray(cov.calibration_events_per_group) > 0
    assert np.all(cov.calibration_variance[has_fp] > 0), (
        "groups with calibration events must carry a directly-measured "
        "calibration variance component")
    # decomposition consistency: diag(C) ~= survey + calibration within the
    # MC tolerance of a 400-draw variance estimate (~sqrt(2/400) ~ 7% rel)
    expect = cov.survey_variance + cov.calibration_variance
    assert np.allclose(np.diag(C), expect, rtol=0.30), (
        "covariance diagonal is inconsistent with its survey + calibration "
        "decomposition beyond MC tolerance")
    # at least one off-diagonal pair with calibration mass on both sides is
    # positively correlated through the shared amplitude draw
    pairs = [(i, j) for i in range(len(edges)) for j in range(i)
             if has_fp[i] and has_fp[j]]
    if pairs:
        assert max(C[i, j] for i, j in pairs) > 0


def test_gate_accepts_a_null_realization(pack, edges):
    """Counts drawn FROM the model must not be rejected by the frozen gate."""
    cov = GC.estimate_covariance(pack, group_edges=edges, n_draws=400, seed=2)
    mu_sig, fp_fold, live = GC._fold_parts(pack)
    mu = mu_sig + fp_fold(np.asarray(pack.fp_counts, float))
    rng = np.random.default_rng(99)
    null_pack = dataclasses.replace(
        pack, counts=rng.poisson(np.clip(mu, 0, None)).astype(np.int64))
    res = GC.predictive_gate(null_pack, covariance=cov, group_edges=edges,
                             n_null_draws=400, seed_null=3)
    assert res.p_value > 0.01
    assert not res.fallback_1d


def test_gate_rejects_an_injected_group_shift(pack, edges):
    """A +8-sigma shift concentrated in one group must be rejected."""
    cov = GC.estimate_covariance(pack, group_edges=edges, n_draws=400, seed=4)
    mu_sig, fp_fold, live = GC._fold_parts(pack)
    mu = mu_sig + fp_fold(np.asarray(pack.fp_counts, float))
    A = GC.group_aggregator(pack, edges)
    shift = 8.0 * np.sqrt(np.diag(cov.matrix)[0])
    counts = np.asarray(pack.counts, float).copy()
    # add the shift uniformly over the first group's live cells
    sel = (A[0] > 0)
    ncells = int(np.broadcast_to(sel[:, None, None], counts.shape).sum())
    counts += np.where(np.broadcast_to(sel[:, None, None], counts.shape),
                       shift / ncells, 0.0)
    shifted = dataclasses.replace(pack,
                                  counts=np.round(counts).astype(np.int64))
    res = GC.predictive_gate(shifted, covariance=cov, group_edges=edges,
                             n_null_draws=400, seed_null=5)
    assert res.p_value < 0.02


def test_cond_fallback_refuses_inversion_and_is_labeled(pack, edges):
    """The frozen cond>1e6 fallback (spec section 3) — previously untested.

    Two arms, matching the two ways the guard can see trouble:
    (a) a degenerate MATRIX behind a healthy stored condition_number — the
        fallback must engage from the re-measured cond, never trusting the
        stored record alone;
    (b) a healthy matrix with a poisoned stored field — the stored field must
        also be honored (either exceeding the threshold refuses inversion).
    In both arms T degrades to the descriptive max|z| and the report says so.
    """
    cov = GC.estimate_covariance(pack, group_edges=edges, n_draws=150, seed=21)
    G = cov.matrix.shape[0]
    # (a) rank-1 (singular) matrix with a positive diagonal; stored cond stays
    # the healthy one from the estimate — only the re-measured cond can fire
    v = np.sqrt(np.diag(cov.matrix))
    bad = dataclasses.replace(cov, matrix=np.outer(v, v) + 1e-12 * np.eye(G))
    assert bad.condition_number < GC.MAX_CONDITION_NUMBER   # the stored record
    res = GC.predictive_gate(pack, covariance=bad, group_edges=edges,
                             n_null_draws=60, seed_null=22)
    assert res.fallback_1d
    want = float(np.max(np.abs(res.residual) / np.sqrt(np.diag(bad.matrix))))
    assert res.T_obs == pytest.approx(want, rel=1e-12)
    # PI ruling 17 (2026-08-06), conservative fallback reporting: the
    # uncalibrated max|z| tail carries NO p-value and NO pass/fail; no null
    # ensemble is drawn. (Before Phase C this path attached the same p-value
    # machinery; the fallback has never engaged on a produced result.)
    assert res.p_value is None
    assert res.n_null_draws == 0
    assert "FALLBACK ENGAGED" in res.report()["layer"]
    assert "NO p-value" in res.report()["layer"]
    # (b) healthy matrix, poisoned stored field
    poisoned = dataclasses.replace(cov, condition_number=1e9)
    res_b = GC.predictive_gate(pack, covariance=poisoned, group_edges=edges,
                               n_null_draws=60, seed_null=23)
    assert res_b.fallback_1d
    # and the healthy covariance does NOT fall back (label unqualified)
    res_ok = GC.predictive_gate(pack, covariance=cov, group_edges=edges,
                                n_null_draws=60, seed_null=24)
    assert not res_ok.fallback_1d
    assert "FALLBACK" not in res_ok.report()["layer"]


def test_transport_delta_method_matches_the_ensemble(pack, edges):
    """Layer C's delta-method calibration variance on the TOTAL must agree
    with the resampling ensemble's variance of the folded-FP total."""
    stats = GC.transport_stress_stats(pack)
    mu_sig, fp_fold, live = GC._fold_parts(pack)
    n0 = np.asarray(pack.fp_counts, float)
    rng = np.random.default_rng(7)
    tots = np.array([
        float(np.where(live, fp_fold(rng.poisson(n0)), 0.0).sum())
        for _ in range(600)])
    assert np.isclose(stats["var_calibration"], tots.var(ddof=1),
                      rtol=0.25), (
        "delta-method calibration variance disagrees with the resampling "
        "ensemble beyond MC tolerance")
    assert "UNCALIBRATED" in stats["layer"]


def test_frozen_seed_determinism(pack, edges):
    a = GC.estimate_covariance(pack, group_edges=edges, n_draws=150, seed=11)
    b = GC.estimate_covariance(pack, group_edges=edges, n_draws=150, seed=11)
    assert np.array_equal(a.matrix, b.matrix)
    ra = GC.predictive_gate(pack, covariance=a, group_edges=edges,
                            n_null_draws=150, seed_null=12)
    rb = GC.predictive_gate(pack, covariance=b, group_edges=edges,
                            n_null_draws=150, seed_null=12)
    assert ra.T_obs == rb.T_obs and ra.p_value == rb.p_value


def test_covariance_report_exposes_required_provenance(pack, edges):
    cov = GC.estimate_covariance(pack, group_edges=edges, n_draws=150, seed=13)
    rep = cov.report()
    for key in ("axis_labels", "group_edges", "units", "method", "raw_dim",
                "algebraic_rank", "effective_rank", "eigenvalues",
                "condition_number", "regularization",
                "calibration_events_total", "calibration_events_per_group",
                "n_resamples", "seed", "mc_error_rel_max"):
        assert key in rep, f"covariance report missing required field {key}"
