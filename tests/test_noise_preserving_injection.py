"""Unit tests for injection/noise_preserving.py (R-041A A3). Synthetic spectra only."""
import hashlib

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


# ---- MAX4 repair cycle (2026-09-01): prescription B + a bit-for-bit guard on prescription A ----
# Stored expectation computed from the code at 82a3359 (before residual_preserving was added), env
# gpdla (numpy 2.4.4), on the _spec() fixture: sha256 of out.tobytes() plus four sampled values.
_VP_SHA_82A3359 = "b7bef2cd871933df42131aec464a0fa6f3d8c606c28ae3a63b9feb363b19fa89"
_VP_SAMPLES_82A3359 = (0.4252662209636783, 0.6100446177257272, 1.0957480962748956, 1.0344196251070354)
_SE_SHA_82A3359 = "15d06a8bbc9d8cc76448a5c3883a31739e437596b3a602d7aadff36f26d6c2d4"


def test_variance_preserving_is_bit_for_bit_unchanged_since_82a3359():
    wave, flux, ivar, mask, S, sigma = _spec()
    ab = [{"nhi": 10 ** 21.0, "z_dla": 4.0}]
    out = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=7)
    assert hashlib.sha256(out.tobytes()).hexdigest() == _VP_SHA_82A3359
    assert (out[0], out[1500], out[2500], out[-1]) == _VP_SAMPLES_82A3359
    se = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=7, method="signal_estimate")
    assert hashlib.sha256(se.tobytes()).hexdigest() == _SE_SHA_82A3359


def test_residual_preserving_carries_the_observed_residual():
    wave, flux, ivar, mask, S_true, sigma = _spec()
    ab = [{"nhi": 10 ** 21.0, "z_dla": 4.0}]
    out, parts = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=7, method="residual_preserving",
                                            return_parts=True)
    T, S = parts["T"], parts["S"]
    assert parts["eps"] is None                                            # no synthetic noise
    ok = np.isfinite(S)
    assert ok.all() and (T != 1.0).all()      # this 3-line profile never returns to T == 1 on the fixture grid
    assert np.max(np.abs((out - T * S) - (flux - S))) < 1e-12              # F' - T S_est == F - S_est, every pixel
    assert np.array_equal(out, T * S + parts["resid"])                     # the implemented expression, bitwise
    core = T < 1e-6
    # trough = the observed residual up to the T S term (|T S| < 1e-6 there; the exact identity is asserted above)
    assert core.sum() > 10 and np.allclose(out[core], (flux - S)[core], atol=2e-6, rtol=0)
    # seed-independent; differs from prescription A; equals the algebraically identical signal_estimate form
    assert np.array_equal(out, NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=99, method="residual_preserving"))
    assert not np.array_equal(out, NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=7))
    se = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, method="signal_estimate")
    assert np.allclose(out, se, atol=1e-12, rtol=0)
    # with a mean-flux rescaling the residual F - S is still carried through unchanged
    r = NP.meanflux_ratio(wave, 4.2, NP.taueff("fg2008"), NP.taueff("finder_fiducial"))
    out_r, pr = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, r=r, method="residual_preserving", return_parts=True)
    assert np.max(np.abs(pr["resid"] - (flux - S))) < 1e-12
    assert np.max(np.abs((out_r - T * r * S) - (flux - S))) < 1e-12


def test_residual_preserving_keeps_flux_bitwise_where_T_is_exactly_one(monkeypatch):
    # a transmission with a genuine outside region (T == 1 exactly): those pixels keep F bit-for-bit,
    # the inside follows T S + (F - S)
    wave, flux, ivar, mask, S_true, sigma = _spec()
    T_syn = np.ones(wave.size); win = (wave > 6000.0) & (wave < 6150.0); T_syn[win] = 0.1
    monkeypatch.setattr(NP, "transmission", lambda w, absorbers, num_lines=3: T_syn.copy())
    out, parts = NP.inject_noise_preserving(wave, flux, ivar, mask, [{"nhi": 1e21, "z_dla": 4.0}],
                                            method="residual_preserving", return_parts=True)
    S = parts["S"]
    assert (~win).sum() > 50 and np.array_equal(out[~win], flux[~win])
    assert np.array_equal(out[win], (0.1 * S + (flux - S))[win])
    # NaN-S spectra (unusable) are returned unchanged by construction
    bad = np.zeros_like(mask); bad[:] = 1
    out2 = NP.inject_noise_preserving(wave, flux, ivar, bad, [{"nhi": 1e21, "z_dla": 4.0}], method="residual_preserving")
    assert np.array_equal(out2, flux)


def test_unknown_method_raises():
    wave, flux, ivar, mask, S, sigma = _spec()
    with pytest.raises(ValueError):
        NP.inject_noise_preserving(wave, flux, ivar, mask, [], method="no_such_method")
