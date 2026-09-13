"""Unit tests for the absorber-ladder response-variant fit/CV helpers
(validation/absorber_ladder/response/).

Every numerical helper that the variant ladder depends on is either checked
against the COMMITTED code it re-implements (element-wise, atol=0) or against
an independently-derived closed form / counting identity.  Tolerances are
explicit and never rely on ``np.allclose``'s default atol, which is vacuous on
small numbers (feedback_allclose_atol_tiny_values).

ENV: gpdla (numpy/scipy only) or gpdla-hbi.
"""
from __future__ import annotations

import importlib.util as ilu
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_RESP = os.path.join(_REPO, "validation", "absorber_ladder", "response")
for p in (_REPO, _RESP):
    if p not in sys.path:
        sys.path.insert(0, p)

import respfit as R                                             # noqa: E402
import opmetrics as OM                                          # noqa: E402
import r1d_empirical as R1D                                     # noqa: E402


def _load_file(name, path):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def committed_shared():
    """The COMMITTED shared-surface builder (numpy-only, safe to load)."""
    return _load_file("_t_run_d2b_lib",
                      os.path.join(_REPO, "CDDF_analysis", "hbi",
                                   "adopted_response", "run_d2b_lib.py"))


@pytest.fixture(scope="module")
def committed_fitlib():
    """The COMMITTED sub-bin/moment fitter.

    ``fitlib`` imports ``_moment_to_skewnormal`` from ``znz_kernel`` after
    inserting a hard-coded worktree on sys.path; pre-seeding sys.modules with
    THIS worktree's module makes that import resolve here instead.
    """
    import CDDF_analysis.hbi.znz_kernel                          # noqa: F401
    return _load_file("_t_fitlib",
                      os.path.join(_REPO, "CDDF_analysis", "hbi",
                                   "adopted_response", "fitlib.py"))


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(20260913)
    n = 20000
    N = 19.0 + 3.0 * rng.beta(1.5, 6.0, n)
    dx = (0.05 + 0.02 * (N - 20.1)) + 0.13 * rng.standard_normal(n) \
        + 0.05 * rng.exponential(1.0, n)
    snr = 2.0 + 8.0 * rng.random(n)
    zq = 2.1 + 1.4 * rng.random(n)
    tid = rng.integers(1, 4000, n).astype(np.int64)
    return dict(N=N, dx=dx, snr=snr, zq=zq, tid=tid)


@pytest.fixture(scope="module")
def ml_rows(sample):
    """The per-cell ML sub-bin rows of the estimator of record, fitted ONCE
    (the Nelder-Mead sub-bin fits dominate the suite's runtime)."""
    isr, izr = R.cell_index(sample["snr"], sample["zq"])
    return [[R.subbin_moments(sample["N"][(isr == i) & (izr == j)],
                              sample["dx"][(isr == i) & (izr == j)],
                              R.R0_EDGES, 25, "ml")
             for j in range(3)] for i in range(3)]


@pytest.fixture(scope="module")
def r0_obj(sample, ml_rows):
    isr, izr = R.cell_index(sample["snr"], sample["zq"])
    spec = dict(R.SPEC_R0); spec["min_n"] = 25
    return R.fit_variant(sample["N"], sample["dx"], isr, izr, 20.1, spec)


# ===========================================================================
# 1. re-implementations match the COMMITTED code element-wise
# ===========================================================================
def test_moment_to_skewnormal_matches_count_conserving_fold():
    ccf = OM._load_ccf()
    mean = np.linspace(19.5, 22.0, 41)
    sd = np.linspace(0.03, 0.45, 41)
    sk = np.linspace(-0.9, 0.9, 41)
    a = R.moment_to_skewnormal(mean, sd, sk)
    b = ccf._m2sn_vec(mean, sd, sk)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_subbin_moments_match_committed_fitlib(committed_fitlib, sample):
    """Element-wise equality with the committed estimator, sample and ML."""
    for mode, edges, min_n in (
            ("sample", np.arange(19.0, 21.4 + 1e-9, 0.1), 50),
            ("ml", np.arange(19.0, 21.4 + 1e-9, 0.4), 50)):
        mine = R.subbin_moments(sample["N"], sample["dx"], edges, min_n, mode)
        theirs = committed_fitlib.subbin_moments(sample["N"], sample["dx"],
                                                 edges, min_n, mode)
        assert len(mine) == len(theirs) > 4
        for a, b in zip(mine, theirs):
            for k in ("c", "n", "mu", "sig", "skew", "ok"):
                assert a[k] == b[k], (mode, k, a, b)


def test_subbin_moments_ml_trunc_matches_committed_fitlib(committed_fitlib):
    """The truncated-ML mode, on a deliberately small, MILDLY truncated case.

    ``ml_trunc`` is not used by any variant in the ladder (R0-R1c use ``ml``;
    the CV-selected R1c uses ``ml``).  It is also NOT numerically robust: the
    committed Nelder-Mead wanders badly when the x_hat >= 19.5 selection bites
    hard (single sub-bin fits ranging from 2 s to > 600 s were measured), which
    is why it is exercised here only on a well-conditioned sample.
    """
    rng = np.random.default_rng(20260913)
    n = 1500
    N = 20.0 + 1.4 * rng.random(n)          # far from the 19.5 floor
    dx = 0.05 + 0.12 * rng.standard_normal(n)
    edges = np.arange(20.0, 21.4 + 1e-9, 0.5)
    mine = R.subbin_moments(N, dx, edges, 50, "ml_trunc")
    theirs = committed_fitlib.subbin_moments(N, dx, edges, 50, "ml_trunc")
    assert len(mine) == len(theirs) >= 2
    for a, b in zip(mine, theirs):
        for k in ("c", "n", "mu", "sig", "skew", "ok"):
            assert a[k] == b[k], (k, a, b)


def test_surfaces_percell_matches_committed_fitlib(committed_fitlib, sample):
    isr, izr = R.cell_index(sample["snr"], sample["zq"])
    edges = np.arange(19.0, 21.4 + 1e-9, 0.1)
    rows = [[R.subbin_moments(sample["N"][(isr == i) & (izr == j)],
                              sample["dx"][(isr == i) & (izr == j)],
                              edges, 25, "sample")
             for j in range(3)] for i in range(3)]
    s1, r1 = R.surfaces_percell(rows, 20.1, 2)
    s2, r2 = committed_fitlib.surfaces_from_rows(rows, 20.1, 2)
    for k in ("mu", "sig", "skew"):
        assert np.array_equal(s1[k], s2[k])
    assert np.array_equal(r1, r2)


def test_surfaces_shared_matches_committed_builder(committed_shared, sample):
    isr, izr = R.cell_index(sample["snr"], sample["zq"])
    edges = np.arange(19.0, 21.4 + 1e-9, 0.1)
    rows = [[R.subbin_moments(sample["N"][(isr == i) & (izr == j)],
                              sample["dx"][(isr == i) & (izr == j)],
                              edges, 25, "sample")
             for j in range(3)] for i in range(3)]
    s1, r1, sh1 = R.surfaces_shared(rows, 20.1, 3, deg_cell=2)
    s2, r2, sh2 = committed_shared.shared_surfaces(rows, 3, 20.1)
    for k in ("mu", "sig", "skew"):
        assert np.max(np.abs(s1[k] - s2[k])) == 0.0
    assert np.array_equal(r1, r2)
    assert np.max(np.abs(sh1 - sh2)) == 0.0


def test_shared_surfaces_mutation_guard(committed_shared, sample):
    """A one-iteration shared fit must NOT equal the two-iteration one — the
    n_iter semantics are load-bearing and a silent change must be detectable."""
    isr, izr = R.cell_index(sample["snr"], sample["zq"])
    edges = np.arange(19.0, 21.4 + 1e-9, 0.1)
    rows = [[R.subbin_moments(sample["N"][(isr == i) & (izr == j)],
                              sample["dx"][(isr == i) & (izr == j)],
                              edges, 25, "sample")
             for j in range(3)] for i in range(3)]
    _, _, sh2 = R.surfaces_shared(rows, 20.1, 3, deg_cell=2, n_iter=2)
    _, _, sh1 = R.surfaces_shared(rows, 20.1, 3, deg_cell=2, n_iter=1)
    assert np.max(np.abs(sh2 - sh1)) > 0.0


def test_cell_index_matches_committed_digitize(sample):
    snr_edges = np.array([2.0, 3.5, 6.5, np.inf])
    z_edges = np.array([0.0, 2.56, 2.96, np.inf])
    isr, izr = R.cell_index(sample["snr"], sample["zq"])
    assert np.array_equal(
        isr, np.clip(np.digitize(sample["snr"], snr_edges) - 1, 0, 2))
    assert np.array_equal(
        izr, np.clip(np.digitize(sample["zq"], z_edges) - 1, 0, 2))


# ===========================================================================
# 2. the CV split
# ===========================================================================
def test_parity_fold_is_a_sightline_split(sample):
    A = R.parity_fold(sample["tid"])
    tids = sample["tid"]
    assert A.sum() > 0 and (~A).sum() > 0
    # no TARGETID may appear in both folds (the split unit is the sightline)
    assert len(np.intersect1d(np.unique(tids[A]), np.unique(tids[~A]))) == 0
    # complete and disjoint
    assert int(A.sum()) + int((~A).sum()) == len(tids)


# ===========================================================================
# 3. adaptive sub-binning
# ===========================================================================
def test_adaptive_edges_properties(sample):
    N = sample["N"]
    e = R.adaptive_edges(N, 19.0, 22.4, 0.1, 50, hard_min_n=25, max_width=0.3)
    assert e[0] == 19.0 and e[-1] == pytest.approx(22.4, abs=1e-9)
    assert np.all(np.diff(e) > 0)
    counts, _ = np.histogram(N, e)
    # every bin but the last carries at least hard_min_n
    assert np.all(counts[:-1] >= 25)
    # and the grid never splits finer than the base step
    assert np.min(np.diff(e)) >= 0.1 - 1e-9


def test_adaptive_edges_extends_past_the_r0_clamp(sample):
    """The whole point of R1a: the fixed 0.1-dex/min_n=50 grid stops at 21.35,
    the merged grid must reach further."""
    N = sample["N"]
    fixed = R.subbin_moments(N, sample["dx"],
                             np.arange(19.0, 21.4 + 1e-9, 0.1), 50, "sample")
    e = R.adaptive_edges(N, 19.0, 22.4, 0.1, 50, hard_min_n=25, max_width=0.3)
    adapt = R.subbin_moments_weighted_centre(N, sample["dx"], e, 25, "sample")
    assert max(r["c"] for r in adapt) > max(r["c"] for r in fixed)


# ===========================================================================
# 4. the mixture / marginalisation algebra (the width repair)
# ===========================================================================
def test_mixture_moments_degenerate_case():
    m, s, k = R._mixture_moments(np.array([0.3]), np.array([0.12]),
                                 np.array([0.4]), np.array([1.0]))
    assert abs(m - 0.3) < 1e-15
    assert abs(s - 0.12) < 1e-15
    assert abs(k - 0.4) < 1e-14


def test_mixture_moments_two_normal_components_closed_form():
    """Two symmetric normal components: closed-form mean/variance/third moment."""
    mean = np.array([0.0, 1.0]); sd = np.array([0.2, 0.5])
    sk = np.array([0.0, 0.0]); w = np.array([0.3, 0.7])
    m, s, k = R._mixture_moments(mean, sd, sk, w)
    m_ref = 0.3 * 0.0 + 0.7 * 1.0
    v_ref = (0.3 * (0.2 ** 2 + (0.0 - m_ref) ** 2)
             + 0.7 * (0.5 ** 2 + (1.0 - m_ref) ** 2))
    m3_ref = (0.3 * (3 * (0.0 - m_ref) * 0.2 ** 2 + (0.0 - m_ref) ** 3)
              + 0.7 * (3 * (1.0 - m_ref) * 0.5 ** 2 + (1.0 - m_ref) ** 3))
    assert abs(m - m_ref) < 1e-14
    assert abs(s - np.sqrt(v_ref)) < 1e-14
    assert abs(k - m3_ref / v_ref ** 1.5) < 1e-13


def _flat_object(sig=0.15, slope=0.0, N_ref=20.1, deg=3):
    """A calibration object with constant sigma, linear mu and zero skew."""
    surf = {k: np.zeros((3, 3, deg + 1)) for k in ("mu", "sig", "skew")}
    surf["sig"][..., 0] = sig
    surf["mu"][..., 1] = slope
    rng = np.tile(np.array([19.0, 22.4]), (3, 3, 1))
    return dict(surf=surf, rng=rng, N_ref=N_ref, ramp=R.NO_RAMP,
                rng_data=rng.copy(), shared=np.zeros((3, deg + 1)),
                spec=dict(deg_cell=deg, deg_shared=deg, fit_rng="full"))


def test_bin_marginal_variance_matches_closed_form():
    """For constant sigma and mu(N) = slope*(N-N_ref) the marginalised variance
    over a bin of width w with a FLAT weight is sigma^2 + (1+slope)^2 * Var(N),
    and Var(N) of the n_quad midpoint nodes is known exactly."""
    for slope in (0.0, -0.1, 0.25):
        obj = _flat_object(sig=0.15, slope=slope)
        ne = np.array([20.0, 20.2, 20.5])
        nq = 17
        _, sd_t, _ = R.bin_marginal_targets(obj, ne, n_quad=nq)
        for b, (lo, hi) in enumerate(((20.0, 20.2), (20.2, 20.5))):
            q = np.linspace(lo, hi, nq + 2)[1:-1]
            var_N = float(np.var(q))
            ref = np.sqrt(0.15 ** 2 + (1.0 + slope) ** 2 * var_N)
            assert abs(sd_t[0, 0, b] - ref) < 1e-12, (slope, b)


def test_marginalisation_is_a_strict_widening():
    obj = _flat_object(sig=0.15, slope=0.0)
    ne = np.linspace(19.0, 22.2, 17)
    _, sd_t, _ = R.bin_marginal_targets(obj, ne, n_quad=17)
    assert np.all(sd_t > 0.15)


def test_marginalise_object_removes_the_clamp_and_the_ramp():
    obj = _flat_object(sig=0.15, slope=-0.05)
    obj["ramp"] = (21.0, 0.5)
    ne = np.linspace(19.0, 22.2, 17)
    new = R.marginalise_object(obj, ne, n_quad=17, deg=4)
    assert np.array_equal(new["ramp"], np.asarray(R.NO_RAMP)) or \
        tuple(new["ramp"]) == R.NO_RAMP
    assert np.all(new["rng"][..., 0] == ne[0])
    assert np.all(new["rng"][..., 1] == ne[-1])
    assert set(new["marginal_resid"]) == {"mu", "sig", "skew"}


# ===========================================================================
# 5. operator metrics
# ===========================================================================
def test_row_stats_known_distribution():
    centers = np.array([0.0, 1.0, 2.0, 3.0])
    mass = np.array([0.1, 0.4, 0.4, 0.1])
    m, sd, sk, down, inb, up = OM.row_stats(mass, centers, 1.0, 3.0)
    assert abs(m - 1.5) < 1e-14
    var = 0.1 * 2.25 + 0.4 * 0.25 + 0.4 * 0.25 + 0.1 * 2.25
    assert abs(sd - np.sqrt(var)) < 1e-14
    assert abs(sk) < 1e-14                      # symmetric
    assert abs(down - 0.1) < 1e-15
    assert abs(up - 0.1) < 1e-15
    assert abs(down + inb + up - 1.0) < 1e-15


def test_row_stats_is_normalisation_invariant():
    centers = np.arange(5.0)
    mass = np.array([1.0, 3.0, 7.0, 2.0, 1.0])
    a = OM.row_stats(mass, centers, 1.0, 4.0)
    b = OM.row_stats(mass * 13.7, centers, 1.0, 4.0)
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-13


def test_empirical_rows_counting_identity():
    """The decisive check for the one-sided-support bug class: the histogram
    must contain EXACTLY the in-grid events, no more and no fewer."""
    rng = np.random.default_rng(7)
    n = 5000
    nhat = np.arange(19.5, 22.4 + 1e-9, 0.1)
    xhat = 19.0 + 4.0 * rng.random(n)
    b_i = rng.integers(0, 16, n)
    s_i = rng.integers(0, 8, n)
    K_i = rng.integers(0, 3, n)
    out = OM.empirical_rows(xhat, b_i, s_i, K_i, nhat, 16, 8, 3)
    n_in = int(((xhat >= nhat[0]) & (xhat < nhat[-1])).sum())
    assert out.sum() == n_in
    assert out.shape == (8, 3, len(nhat) - 1, 16)


def test_gather_Mg_shape_and_selection():
    rng = np.random.default_rng(3)
    masses = rng.random((3, 3, 29, 16))
    s2sr = np.array([0, 0, 0, 0, 1, 1, 1, 2])
    kz2K = np.repeat([0, 1, 2], 5)
    K2zr = np.array([0, 1, 2])
    mg = OM.gather_Mg(masses, s2sr, kz2K, K2zr)
    assert mg.shape == (8, 15, 29, 16)
    for s in range(8):
        for k in range(15):
            assert np.array_equal(mg[s, k], masses[s2sr[s], K2zr[kz2K[k]]])


# ===========================================================================
# 6. R1d empirical representation
# ===========================================================================
def test_r1d_rows_are_unit_mass_and_conserve_counts():
    rng = np.random.default_rng(11)
    n = 40000
    ntrue = np.linspace(19.0, 22.2, 17)
    nhat = np.arange(19.5, 22.4 + 1e-9, 0.1)
    N = 19.0 + 3.0 * rng.beta(1.5, 6.0, n)
    xhat = N + 0.05 + 0.15 * rng.standard_normal(n)
    isr = rng.integers(0, 3, n); izr = rng.integers(0, 3, n)
    b_i = np.clip(np.digitize(N, ntrue) - 1, 0, 15)
    out = R1D.fit_r1d(N, xhat, isr, izr, b_i, ntrue, nhat, 20.1, deg=2,
                      lam=1.0, J=15)
    s = out["masses"].sum(axis=2)
    assert np.all(np.abs(s - 1.0) < 1e-12)
    assert np.all(out["masses"] >= 0.0)
    A, tot, _ = R1D.raw_masses(N, xhat, isr, izr, b_i, ntrue, nhat, 15)
    assert A.sum() == tot.sum()
    j, c_idx, _ = R1D.offset_index(xhat, N, nhat)
    n_in = int(((c_idx >= 0) & (c_idx < len(nhat) - 1)
                & (np.abs(j) <= 15)).sum())
    assert A.sum() <= n_in + 1e-9


def test_r1d_ridge_shrinks_the_curvature():
    rng = np.random.default_rng(13)
    n = 20000
    ntrue = np.linspace(19.0, 22.2, 17)
    nhat = np.arange(19.5, 22.4 + 1e-9, 0.1)
    N = 19.0 + 3.0 * rng.beta(1.5, 6.0, n)
    xhat = N + 0.05 + 0.15 * rng.standard_normal(n)
    isr = rng.integers(0, 3, n); izr = rng.integers(0, 3, n)
    b_i = np.clip(np.digitize(N, ntrue) - 1, 0, 15)
    lo = R1D.fit_r1d(N, xhat, isr, izr, b_i, ntrue, nhat, 20.1, deg=2,
                     lam=0.0, J=15)
    hi = R1D.fit_r1d(N, xhat, isr, izr, b_i, ntrue, nhat, 20.1, deg=2,
                     lam=1e5, J=15)
    assert hi["n_coef"]["effective"] < lo["n_coef"]["effective"]


# ===========================================================================
# 7. the variant plumbing
# ===========================================================================
def test_fit_variant_R0_equals_the_explicit_two_step(r0_obj, ml_rows):
    obj = r0_obj
    surf, rng, _ = R.surfaces_shared(ml_rows, 20.1, 3, deg_cell=2)
    for k in ("mu", "sig", "skew"):
        assert np.max(np.abs(obj["surf"][k] - surf[k])) == 0.0
    assert np.array_equal(obj["rng"], rng)


def test_coefficient_count_is_the_declared_one(r0_obj):
    obj = r0_obj
    # 9 cells x (deg_cell+1) + 1 shared cubic = 28 per moment, 84 in total,
    # plus 18 data-determined fit-range numbers
    assert obj["n_coef"]["per_moment"] == 28
    assert obj["n_coef"]["total_moments"] == 84
    assert obj["n_coef"]["fit_range_numbers"] == 18


def test_per_event_loglik_is_finite_and_scales(sample, r0_obj):
    isr, izr = R.cell_index(sample["snr"], sample["zq"])
    obj = r0_obj
    ll, n = R.per_event_loglik(obj["surf"], obj["rng"], 20.1, sample["N"],
                               sample["dx"], isr, izr, truncated=True,
                               skew_ramp=obj["ramp"])
    assert np.isfinite(ll) and n == len(sample["N"])
    half = slice(0, len(sample["N"]) // 2)
    ll2, n2 = R.per_event_loglik(obj["surf"], obj["rng"], 20.1,
                                 sample["N"][half], sample["dx"][half],
                                 isr[half], izr[half], truncated=True,
                                 skew_ramp=obj["ramp"])
    assert n2 == len(sample["N"][half])
    assert abs(ll2) < abs(ll) * 1.05


def test_no_ramp_is_a_true_no_op():
    obj = _flat_object()
    obj["surf"]["skew"][..., 0] = 0.4
    N = np.linspace(19.0, 22.4, 35)
    i = np.zeros(len(N), int)
    _, _, sk = R.eval_moments(obj["surf"], obj["rng"], obj["N_ref"], N, i, i,
                              skew_ramp=R.NO_RAMP)
    assert np.max(np.abs(sk - 0.4)) < 1e-15
    _, _, sk2 = R.eval_moments(obj["surf"], obj["rng"], obj["N_ref"], N, i, i,
                               skew_ramp=(21.0, 0.5))
    assert np.max(np.abs(sk2[N >= 21.5])) == 0.0
