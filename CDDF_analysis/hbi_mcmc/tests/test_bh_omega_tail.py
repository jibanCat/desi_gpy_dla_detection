"""Synthetic guards for bh_omega_tail.py and the BH producer's NaN-SNR guard. No real-data values."""
import math

import numpy as np

from CDDF_analysis.hbi_mcmc import bh_omega_tail as BT


def _grid():
    edges = np.round(np.arange(17.2, 22.4 + 0.05, 0.1), 10)
    lo, hi = edges[:-1], edges[1:]
    N_b = 10.0 ** (0.5 * (lo + hi))
    dN_b = 10.0 ** hi - 10.0 ** lo
    return lo, hi, N_b, dN_b


def test_omega_grid_is_additive_over_disjoint_ranges_and_band_recentres_on_the_point():
    lo, hi, N_b, dN_b = _grid()
    fb = 1e-21 * (N_b / 1e20) ** -2.0
    K = 1.0e-3
    tot = BT.omega_grid(fb, lo, N_b, dN_b, K, 20.3, 22.4 + 1e-6)
    assert math.isclose(tot, BT.omega_grid(fb, lo, N_b, dN_b, K, 20.3, 21.6) + BT.omega_grid(fb, lo, N_b, dN_b, K, 21.6, 22.4 + 1e-6), rel_tol=1e-12)
    b = BT.band(np.linspace(0.8, 1.2, 1001), point=2.0, recenter=True)
    assert math.isclose(b["q50_recentred"], 2.0) and b["q16"] < 2.0 < b["q84"]


def test_powerlaw_tail_anchor_and_convergence():
    lo, hi, N_b, dN_b = _grid()
    K = 1.0
    for s_true, converges in ((-2.5, True), (-1.5, False)):
        fb = 1e-21 * (N_b / 1e20) ** s_true
        om, slope, inf_extra = BT.powerlaw_tail(fb, lo, hi, N_b, dN_b, K, 21.6, 22.4)
        assert math.isclose(slope, s_true, abs_tol=1e-9)              # exact power law recovered
        assert math.isclose(om, BT.omega_grid(fb, lo, N_b, dN_b, K, 21.6, 22.4), rel_tol=1e-9)
        assert (np.isfinite(inf_extra) and inf_extra > 0) == converges


def test_phw05_tail_amplitude_matches_itself():
    lo, hi, N_b, dN_b = _grid()
    params = (10.0 ** -23.52, 10.0 ** 21.48, -1.8)
    k2, Ng, a = params
    fb = k2 * (N_b / Ng) ** a * np.exp(-N_b / Ng)
    om, amp, inf_extra = BT.phw05_tail(fb, lo, hi, N_b, dN_b, 1.0, 21.6, 22.4, params)
    assert math.isclose(amp, 1.0, rel_tol=1e-9)
    assert math.isclose(om, BT.omega_grid(fb, lo, N_b, dN_b, 1.0, 21.6, 22.4), rel_tol=1e-9)
    assert inf_extra > 0


def test_finite_snr_guard_counts_and_drops_nan_entries():
    from CDDF_analysis.hbi.track_c_tf_hz import finite_snr_guard
    lk = {1: (3.0, 4.5), 2: (float("nan"), 6.3), 3: (0.5, 5.0), 4: (float("nan"), 6.5)}
    kept, n_bad = finite_snr_guard(dict(lk), drop=False)
    assert n_bad == 2 and kept == lk                                   # default: unchanged, counted
    kept, n_bad = finite_snr_guard(dict(lk), drop=True)
    assert n_bad == 2 and set(kept) == {1, 3}
