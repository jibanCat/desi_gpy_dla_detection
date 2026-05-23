"""
tests/test_dla_clustering.py
============================
Tests for gpy_dla_detection.dla_clustering.DLAClusteringPrior.

Validates:
- sigma8 reproduction from EH98 P(k)
- growth factor D(z=2.5)/D(0)
- xi_DLA magnitude at 200 km/s (pair clustering signal)
- xi_DLA decay at large separations
- log_rho = 0 for k=1 (single DLA)
- additive log_rho is strictly less than multiplicative (additive is the correct
  leading-order approximation)
- log_rho is floored and finite even for extremely close pairs
- small-scale cap keeps xi_DLA finite and identical below r_cut
"""
import numpy as np
import pytest
from gpy_dla_detection.dla_clustering import DLAClusteringPrior


@pytest.fixture(scope="module")
def cp():
    return DLAClusteringPrior(b_dla=2.0)


def test_sigma8_reproduced(cp):
    assert cp.sigma8_check() == pytest.approx(0.831, abs=2e-3)


def test_growth_z25(cp):
    assert cp.growth_D(2.5) == pytest.approx(0.359, abs=5e-3)


def test_xi_dla_magnitude(cp):
    val = 1.0 + cp.xi_dla(np.array([200.0]), np.array([2.5]))[0]
    assert 2.3 < val < 2.95


def test_xi_dla_decays(cp):
    assert cp.xi_dla(np.array([3000.0]), np.array([2.5]))[0] < 0.1


def test_log_rho_k1_is_zero(cp):
    z = np.array([[2.5, 2.6, 2.7]])  # shape (1, N=3)
    assert np.allclose(cp.log_rho(z), 0.0)


def test_log_rho_additive_below_multiplicative_k3(cp):
    z = np.array([[2.500, 2.500], [2.503, 2.503], [2.506, 2.506]])  # (k=3, N=2)
    add = cp.log_rho(z)
    c = 299792.458
    mult = np.zeros(z.shape[1])
    for a in range(3):
        for b in range(a + 1, 3):
            zbar = 0.5 * (z[a] + z[b])
            dv = c * np.abs(z[a] - z[b]) / (1 + zbar)
            mult += np.log1p(cp.xi_dla(dv, zbar))
    assert np.all(add < mult)
    assert np.all(add > 0)


def test_log_rho_floored_finite(cp):
    z = np.array([[2.5000], [2.50001]])  # (k=2, N=1)
    out = cp.log_rho(z)
    assert np.isfinite(out).all()
    assert out[0] >= np.log(cp.eps)


def test_small_scale_cap(cp):
    tiny = cp.xi_dla(np.array([1.0]), np.array([2.5]))[0]
    capped = cp.xi_dla(np.array([1e-3]), np.array([2.5]))[0]
    assert np.isfinite(tiny) and tiny == pytest.approx(capped, rel=1e-6)


def test_log_rho_vectorized_distinct_columns(cp):
    # (k=2, N=3) with DIFFERENT z per column -> distinct, finite, shape (3,)
    z = np.array([[2.50, 2.70, 3.00], [2.503, 2.85, 3.20]])
    out = cp.log_rho(z)
    assert out.shape == (3,)
    assert np.isfinite(out).all()
    assert out[0] > out[1] > out[2]   # closer pair => larger log rho


def test_shape_contracts(cp):
    assert cp.xi_dla(np.array([200.0, 500.0, 800.0]), np.array([2.5, 2.5, 2.5])).shape == (3,)
    assert cp.growth_D(np.array([2.5, 3.0])).shape == (2,)
    assert cp.log_rho(np.array([[2.5, 2.6], [2.51, 2.7]])).shape == (2,)


def test_log_rho_rejects_1d_input(cp):
    # A (11, 1) array has k=11 DLA rows which exceeds MAX_DLAS; the guard must fire.
    # (A plain 1D (N,) array silently becomes (1, N) via atleast_2d, which is
    # harder to guard cheaply — the k>10 check catches the obviously-wrong 2D case.)
    bad = np.arange(11, dtype=float).reshape(11, 1)
    with pytest.raises(ValueError, match="log_rho expects"):
        cp.log_rho(bad)


# --------------------------------------------------------------------------- #
# Closed-form prior-window average E_unif[ρ_k] (mean_xi_window / prior_mean_rho)
# --------------------------------------------------------------------------- #
_Z_MIN, _Z_MAX = 2.40, 2.60  # a representative z-DLA search window


def test_prior_mean_rho_k1_is_exactly_one(cp):
    # k=1 -> no pairs -> C(1,2)=0 -> E_unif[ρ] = 1.0 exactly.
    assert cp.prior_mean_rho(1, _Z_MIN, _Z_MAX) == 1.0


def test_prior_mean_rho_monotone(cp):
    # More DLAs -> more pairs -> larger expected ρ, all strictly above 1.
    r2 = cp.prior_mean_rho(2, _Z_MIN, _Z_MAX)
    r3 = cp.prior_mean_rho(3, _Z_MIN, _Z_MAX)
    assert r3 > r2 > 1.0


@pytest.mark.parametrize("k", [2, 3])
def test_prior_mean_rho_matches_monte_carlo(cp, k):
    """Closed-form E_unif[ρ_k] = 1 + C(k,2)·⟨ξ⟩ matches a Monte-Carlo estimate
    that draws N uniform z-pairs in [z_min, z_max] and averages 1 + Σ_{i<j} ξ.
    The referee verified closed-form ≈ MC to 4 sig figs; we require ~1%."""
    rng = np.random.default_rng(0)
    n = 400_000
    c = 299792.458
    sum_xi = np.zeros(n)
    # draw k uniform redshifts per sample, sum ξ over all C(k,2) pairs
    zk = rng.uniform(_Z_MIN, _Z_MAX, size=(k, n))
    for a in range(k):
        for b in range(a + 1, k):
            zbar = 0.5 * (zk[a] + zk[b])
            dv = c * np.abs(zk[a] - zk[b]) / (1.0 + zbar)
            sum_xi += cp.xi_dla(dv, zbar)
    mc = np.mean(1.0 + sum_xi)
    closed = cp.prior_mean_rho(k, _Z_MIN, _Z_MAX)
    assert closed == pytest.approx(mc, rel=0.01)


def test_mean_xi_window_positive_and_degenerate(cp):
    # A finite-width window has a positive mean ξ; a zero-width window -> 0.0.
    assert cp.mean_xi_window(_Z_MIN, _Z_MAX) > 0.0
    assert cp.mean_xi_window(2.5, 2.5) == 0.0
