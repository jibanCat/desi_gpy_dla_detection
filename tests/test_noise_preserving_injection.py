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
    # The former check here — `outside = np.abs(T - 1.0) < 1e-12; assert np.array_equal(out[outside],
    # flux[outside])` — was VACUOUS on this fixture (PI ruling 2026-09-01 §8): the 3-line Voigt profile of the
    # frozen primitive never reaches |1 - T| < 1e-12 anywhere on the 3600-6800 A grid (its damping wings decay
    # as 1/dv^2; min |1 - T| ~ 2e-4 at the blue end), so the selected set was EMPTY and array_equal([], [])
    # is True. Pinned here so the reason is visible; the outside-profile guarantee is now tested for BOTH
    # methods, with an asserted non-empty outside set, in
    # test_flux_outside_the_injected_profile_untouched_both_methods.
    assert (np.abs(T - 1.0) < 1e-12).sum() == 0
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


PROFILE_WINDOW_KMS = 3000.0


def _spec_with_outside(monkeypatch, z_dla=4.0, logN=21.0, window_kms=PROFILE_WINDOW_KMS):
    """Fixture with a NON-EMPTY outside region BY CONSTRUCTION (PI ruling 2026-09-01 §8): the frozen Voigt
    primitive never returns T == 1.0 exactly on the 3600-6800 A grid (damping wings), so the profile is
    truncated by an explicit mask — the real Voigt transmission inside |v| < window_kms of the line, exactly
    1.0 outside. The injector's own `transmission` hook is replaced so both methods run their production
    arithmetic on this profile. Returns the spectrum, the absorber list, and the inside/outside masks."""
    wave, flux, ivar, mask, S_true, sigma = _spec()
    lam0 = NP.LYA_REST * (1.0 + z_dla)
    inside = np.abs(wave / lam0 - 1.0) * 299792.458 < window_kms
    real_transmission = NP.voigt_transmission

    def truncated(w, absorbers, num_lines=3):
        T = np.ones(np.asarray(w).size)
        for ab in absorbers:
            T = T * real_transmission(np.asarray(w, float), float(ab["nhi"]), float(ab["z_dla"]), int(ab.get("num_lines", num_lines)))
        T[~inside] = 1.0
        return T

    monkeypatch.setattr(NP, "transmission", truncated)
    return wave, flux, ivar, mask, [{"nhi": 10 ** logN, "z_dla": z_dla}], inside, ~inside


@pytest.mark.parametrize("method", ["variance_preserving", "residual_preserving"])
def test_flux_outside_the_injected_profile_untouched_both_methods(monkeypatch, method):
    wave, flux, ivar, mask, ab, inside, outside = _spec_with_outside(monkeypatch)
    # (c) the outside set is non-empty by construction and asserted, so this test can never be vacuous again
    assert outside.sum() > 100 and inside.sum() > 10
    out, parts = NP.inject_noise_preserving(wave, flux, ivar, mask, ab, seed=7, method=method, return_parts=True)
    T = parts["T"]
    assert np.all(T[outside] == 1.0) and np.all(T[inside] < 1.0)      # the constructed profile is what the injector used
    # (a) bit-for-bit unchanged outside the injected profile
    assert np.array_equal(out[outside], flux[outside])
    # (b) changed inside it (a saturated log N = 21 core: essentially every inside pixel moves)
    assert not np.array_equal(out[inside], flux[inside])
    assert np.mean(out[inside] != flux[inside]) > 0.95
    assert np.max(np.abs(out[inside] - flux[inside])) > 0.1
    if method == "residual_preserving":
        S = parts["S"]
        assert np.max(np.abs((out - T * S) - (flux - S))[inside]) < 1e-12     # the prescription-B identity inside
        # an unusable spectrum (S undefined everywhere) is returned unchanged
        bad = np.ones_like(mask)
        assert np.array_equal(NP.inject_noise_preserving(wave, flux, ivar, bad, ab, method=method), flux)


def test_unknown_method_raises():
    wave, flux, ivar, mask, S, sigma = _spec()
    with pytest.raises(ValueError):
        NP.inject_noise_preserving(wave, flux, ivar, mask, [], method="no_such_method")


def test_meanflux_control_variants_match_the_predeclared_construction():
    """P1 mean-flux control (spec MAX4_MEANFLUX_CONTROL_SPEC_2026-09-02.md §2): the +-1 sigma variants are the fiducial scaled by the
    Turner+2024 nearest-bin relative error (4.2 % at z = 4.15), the envelope rises to 11 % at z = 4.5; the Ding+2024 variant reproduces the
    published spline points exactly and is +13 % above the fiducial at z = 4.20; the FG08 stress arm stays +20 %."""
    fid = NP.taueff("finder_fiducial"); p = NP.taueff("turner2024_p1s"); m = NP.taueff("turner2024_m1s"); d = NP.taueff("ding2024_hz")
    assert abs(fid(4.15) - 0.928) < 0.01                                        # the fiducial IS the Turner fit
    assert abs(p(4.15) / fid(4.15) - (1 + 0.039 / 0.928)) < 1e-9 and abs(m(4.15) / fid(4.15) - (1 - 0.039 / 0.928)) < 1e-9
    assert abs(NP.meanflux_envelope_s(4.5) - 0.11) < 1e-12 and abs(NP.meanflux_envelope_s(4.35) - 0.5 * (0.039 / 0.928 + 0.11)) < 1e-9
    assert abs(NP.meanflux_envelope_s(3.05) - 0.023 / 0.373) < 1e-12
    for z, t in NP.DING2024_SPLINE:
        assert abs(d(z) - t) < 1e-9
    assert abs(d(4.20) / fid(4.20) - 1.09 / (0.00246 * 5.2 ** 3.62)) < 1e-9 and 1.10 < d(4.20) / fid(4.20) < 1.16
    assert abs(d(4.6) - 1.09 * (5.6 / 5.2) ** 4.85) < 1e-9
    fg = NP.taueff("fg2008"); assert 1.15 < fg(4.3) / fid(4.3) < 1.25
