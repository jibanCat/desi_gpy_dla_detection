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
