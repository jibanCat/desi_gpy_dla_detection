"""Unit tests for injection/noise_preserving.py (R-041A A3). Synthetic spectra only."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from injection import noise_preserving as NP  # noqa: E402


def _spec(seed=1, n=4000, snr=3.0):
    rng = np.random.default_rng(seed)
    wave = np.linspace(3600.0, 6800.0, n)
    cont = 1.0 + 0.2 * np.sin(wave / 300.0)
    forest = np.clip(1.0 - 0.6 * rng.random(n) * (wave < 6200.0), 0.05, 1.0)
    S = cont * forest
    sigma = S.mean() / snr
    ivar = np.full(n, 1.0 / sigma ** 2)
    flux = S + rng.standard_normal(n) * sigma
    mask = np.zeros(n, dtype=np.uint32); mask[100:110] = 1; ivar[200:205] = 0.0
    return wave, flux, ivar, mask, S, sigma


def test_variance_preserving_identity_and_invariants():
    wave, flux, ivar, mask, S, sigma = _spec()
    ab = [{"nhi": 10 ** 21.0, "z_dla": 4.0}]
    out, parts = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=7, return_parts=True)
    T, eps = parts["T"], parts["eps"]
    good = (ivar > 0) & (mask == 0)
    synth = np.zeros_like(flux); synth[good] = np.sqrt(1 - T[good] ** 2) * eps[good] / np.sqrt(ivar[good])
    assert np.max(np.abs((out - synth) - T * flux)) < 1e-9          # deterministic part is exactly T * F
    outside = np.abs(T - 1.0) < 1e-12
    assert np.array_equal(out[outside], flux[outside])                # untouched outside the profile
    assert np.array_equal(NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=7), out)   # seeded -> reproducible
    assert not np.array_equal(NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=8), out)


def test_trough_noise_variance_matches_ivar_on_average():
    stds_new, stds_old = [], []
    for seed in range(40):
        wave, flux, ivar, mask, S, sigma = _spec(seed=seed, snr=4.0)
        z = 4.0; ab = [{"nhi": 10 ** 21.3, "z_dla": z}]
        new = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=seed)
        old = NP.inject_multiplicative(wave, flux, ab)
        lam0 = NP.LYA_REST * (1 + z)
        core = np.abs(wave / lam0 - 1.0) * 299792.458 < 300.0
        stds_new.append(np.std(new[core] * np.sqrt(ivar[core]))); stds_old.append(np.std(old[core] * np.sqrt(ivar[core])))
    assert abs(np.mean(stds_new) - 1.0) < 0.1                          # ivar-consistent trough
    assert np.mean(stds_old) < 0.02                                    # the old operation leaves a noiseless trough


def test_meanflux_ratio_only_in_forest_and_scales_signal_not_noise():
    wave, flux, ivar, mask, S, sigma = _spec(seed=3)
    zq = 4.2
    r = NP.meanflux_ratio(wave, zq, NP.taueff("fg2008"), NP.taueff("finder_fiducial"))
    assert np.all(r[wave >= NP.LYA_REST * (1 + zq)] == 1.0)
    assert np.all(r[wave < NP.LYA_REST * (1 + zq)] < 1.0)              # FG08 is lower transmission than the fiducial at z ~ 3-4
    out = NP.inject_noise_preserving(wave, flux, ivar, mask, [], r=r, seed=0)
    Sest = NP.signal_estimate(flux, ivar, mask)
    ok = np.isfinite(Sest)
    assert np.allclose(out[ok], flux[ok] + (r[ok] - 1.0) * Sest[ok])   # noise untouched, signal rescaled


def test_taueff_models_are_ordered_as_documented_at_high_z():
    z = np.array([3.8, 4.2, 4.8])
    fid, fg, bk = NP.taueff("finder_fiducial")(z), NP.taueff("fg2008")(z), NP.taueff("becker2013")(z)
    assert np.all(fg > fid) and np.all(bk > fid * 0.9)
