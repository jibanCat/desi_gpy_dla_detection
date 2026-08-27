"""Regression for the numpy >= 2.4 compatibility of the DEFAULT-OFF clustering prior
(pre-tag review 2026-08-26, PI ruling 7).

`np.trapz` was removed in numpy 2.4; `dla_clustering.mean_xi_window` used to evaluate it
eagerly as a `getattr` default, so every import-time-fine call crashed on the new numpy.
The module is not on the deployed finder path (`pair_prior_mode="off"` by default and in
every production environment); this test pins (i) the numpy-compat behaviour under a numpy
WITHOUT `trapz` and (ii) that the DLA GP's pair prior is still default-off.
"""
import inspect
import numpy as np
import pytest


def test_mean_xi_window_works_without_np_trapz(monkeypatch):
    from gpy_dla_detection.dla_clustering import DLAClusteringPrior
    if hasattr(np, "trapz"):
        monkeypatch.delattr(np, "trapz")
    assert hasattr(np, "trapezoid"), "numpy without both trapz and trapezoid is unsupported"
    p = DLAClusteringPrior()
    v = p.mean_xi_window(2.0, 2.5, n=64)
    assert np.isfinite(v) and v >= 0.0
    assert p.prior_mean_rho(2, 2.0, 2.5) == pytest.approx(1.0 + v)


def test_mean_xi_window_matches_trapezoid_reference():
    from gpy_dla_detection.dla_clustering import DLAClusteringPrior
    p = DLAClusteringPrior()
    v = p.mean_xi_window(2.2, 2.4, n=128)
    # independent reference with the same quadrature nodes
    from gpy_dla_detection.dla_clustering import _C_KMS
    zbar = 2.3; L = _C_KMS * 0.2 / (1 + zbar)
    d = np.linspace(1.0, L, 128); pdf = 2.0 * (L - d) / L ** 2
    ref = float(np.trapezoid(p.xi_dla(d, np.full_like(d, zbar)) * pdf, d))
    assert v == pytest.approx(ref, rel=0, abs=0)


def test_pair_prior_is_default_off():
    from gpy_dla_detection.dla_gp import DLAGP
    sig = inspect.signature(DLAGP.__init__)
    assert sig.parameters["pair_prior_mode"].default == "off"
