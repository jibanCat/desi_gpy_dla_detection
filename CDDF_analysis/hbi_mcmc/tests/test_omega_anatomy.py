"""Synthetic guards for omega_anatomy.py (R-037 generator). No real-data values."""
import math

import numpy as np

from CDDF_analysis.hbi_mcmc import omega_anatomy as OA


def _toy(seed=0, D=50):
    rng = np.random.default_rng(seed)
    ne = np.array([19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7, 20.9, 21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4])
    ze = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 10)
    B, K = len(ne) - 1, len(ze) - 1
    base = 10.0 ** (-1.5 * (0.5 * (ne[:-1] + ne[1:]) - 20.3))            # falling power law per dex
    f = base[None, :, None] * np.exp(0.05 * rng.standard_normal((D, B, K))) * (1.0 + 0.1 * np.arange(K))[None, None, :]
    dXk = 1000.0 * (1.0 + 0.2 * rng.random(K))
    return f, ne, ze, dXk


def test_contributions_sum_to_omega_and_weights_clip_the_window():
    f, ne, ze, dXk = _toy()
    zw = OA.z_weight(ze, dXk, 2.0, 3.5)
    om = OA.omega_per_draw(f, ne, zw)
    C, wb = OA.contributions(f, ne, zw)
    assert np.allclose(C.sum(axis=1), om, rtol=1e-12, atol=0.0)
    assert [(a, b) for (_, a, b, _, _) in wb][-1] == (21.5, 21.6)       # partial top bin clipped at the ceiling
    w = OA.omega_weight(ne, 20.3, 21.6)
    assert w[ne[:-1] < 20.3].sum() == 0.0 and w[ne[:-1] >= 21.7].sum() == 0.0
    assert math.isclose(w[list(ne[:-1]).index(21.5)], (10 ** 21.6 - 10 ** 21.5) / math.log(10.0))


def test_powerlaw_intrabin_reduces_to_adopted_when_f_is_flat_in_logN():
    f, ne, ze, dXk = _toy()
    f[:] = 1.0e-2                                                      # flat per dex: beta = 0 everywhere
    zw = OA.z_weight(ze, dXk, 2.0, 3.5)
    assert np.allclose(OA.powerlaw_intrabin_omega(f, ne, zw), OA.omega_per_draw(f, ne, zw), rtol=1e-10, atol=0.0)


def test_powerlaw_intrabin_moves_mass_toward_the_heavy_edge_for_a_rising_shape():
    f, ne, ze, dXk = _toy()
    f[:] = 10.0 ** (+1.0 * (0.5 * (ne[:-1] + ne[1:]) - 20.3))[None, :, None]   # rising per dex: beta = +1
    zw = OA.z_weight(ze, dXk, 2.0, 3.5)
    # fully-contained bins only: a rising shape loads the heavy edge -> larger Omega
    assert np.all(OA.powerlaw_intrabin_omega(f, ne, zw, 20.3, 21.5) > OA.omega_per_draw(f, ne, zw, 20.3, 21.5))
    # the clipped ceiling bin alone: the same rising shape moves [21.5,21.7)'s mass ABOVE the
    # 21.6 cut, so the reported part [21.5,21.6] gets LESS than the flat share
    assert np.all(OA.powerlaw_intrabin_omega(f, ne, zw, 21.5, 21.6) < OA.omega_per_draw(f, ne, zw, 21.5, 21.6))


def test_upper_limit_monotone_and_z_weight_overlap():
    f, ne, ze, dXk = _toy()
    zw = OA.z_weight(ze, dXk, 2.0, 3.5)
    vals = [np.median(OA.omega_per_draw(f, ne, zw, 20.3, X)) for X in OA.UPPER_LIMIT_SCAN]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    zwb = OA.z_weight(ze, dXk, 3.4, 3.8)                              # quarter-covered bin: only [3.4,3.5) carries weight
    assert zwb[:-1].sum() == 0.0 and math.isclose(zwb[-1], dXk[-1])


def test_treatment_record_fields():
    r = OA.treatment_record(np.linspace(0.9, 1.1, 1001), 1.0)
    assert set(r) == {"q_p2p5_16_50_84_97p5", "median_over_adopted", "median_shift_pct", "halfwidth68_pct", "halfwidth95_pct"}
    assert math.isclose(r["median_shift_pct"], 0.0, abs_tol=1e-9)
