"""tests/test_lls_mirror_noise.py
================================
Tests for the noise-correct Lyman-limit-drop injection in
``gpy_dla_detection.lls.mirror`` (Build P-b).

The mirror mock injects bound-free LyC opacity T = exp(-tau_LL) into 2LPT quickquasars
spectra.  A quickquasars pixel is F_obs = S + n, n ~ N(0, 1/ivar).  The legacy path did
``flux *= T`` which attenuates the noise realization too (region artificially quiet,
true var T^2/ivar while ivar still advertises 1/ivar).  The fix regenerates statistically
correct noise consistent with the (unchanged) ivar:
    F_new = T*F_obs + eps,  eps ~ N(0, (1 - T^2)/ivar)  ->  Var(F_new) = 1/ivar.

Regime assumption (measured on 2LPT-0 loa-124): blue-edge noise is SKY/READ-dominated,
so 1/ivar is independent of the source and stays fixed under attenuation.

These tests are pure-numpy (no compiled Voigt / desispec needed).
"""
import numpy as np
import pytest
from astropy.io import fits

from gpy_dla_detection.lls import mirror
from gpy_dla_detection.lls.mirror import attenuate_with_noise, row_rng, inject_file
from CDDF_analysis.lyc import lyc_transmission, LYMAN_LIMIT, SIGMA_912


# ---------------------------------------------------------------------------
# 1. Synthetic MC: recovered pixel variance in the attenuated region == 1/ivar
# ---------------------------------------------------------------------------
def test_recovered_variance_matches_ivar_all_depths():
    """Inject a known T on a synthetic spectrum with known sigma; the Monte-Carlo
    pixel variance of the mirror flux must equal 1/ivar at every attenuation depth
    (the whole point of the fix — the absorbed region is NOT artificially quiet).

    Tolerance: the sample-variance estimator over Ntrial independent realizations has
    relative std ~ sqrt(2/(Ntrial-1)) ~ 2.2% for Ntrial=4000.  The MEAN over all npix
    attenuated pixels averages this down, so we require it within 3% (the tight, meaningful
    bound).  The MAX over npix pixels is an extreme-value: E[max|z|] ~ sqrt(2 ln npix) ~ 3.4
    per-pixel sigma ~ 7.6% for npix=300, so we allow 12% (~1.6x that) to stay non-flaky while
    still catching any real bias.
    """
    npix = 300
    sigma = 0.7
    ivar = np.full(npix, 1.0 / sigma ** 2)
    S = 3.0 * np.ones(npix)               # constant source (isolates the noise term)
    T = np.linspace(1.0, 0.02, npix)      # ramp: unabsorbed edge -> nearly opaque
    ntrial = 4000

    draws = np.empty((ntrial, npix))
    for k in range(ntrial):
        # fresh independent observed spectrum each trial: F_obs = S + n
        Fobs = S + np.random.default_rng(10_000 + k).standard_normal(npix) * sigma
        draws[k] = attenuate_with_noise(Fobs, ivar, T, np.random.default_rng([7, k, 0]))

    mc_var = draws.var(axis=0)
    mc_mean = draws.mean(axis=0)

    # variance is restored to 1/ivar everywhere (including deep attenuation)
    rel_var_err = np.abs(mc_var - 1.0 / ivar) / (1.0 / ivar)
    assert rel_var_err.mean() < 0.03, f"mean var error {rel_var_err.mean():.4f}"
    assert rel_var_err.max() < 0.12, f"max var error {rel_var_err.max():.4f}"

    # signal is correctly attenuated to T*S
    mc_sig_err = np.abs(mc_mean - T * S)
    # MC error on the mean per pixel ~ sigma/sqrt(ntrial) ~ 0.011; allow 5 sigma
    assert mc_sig_err.max() < 5 * sigma / np.sqrt(ntrial), mc_sig_err.max()


def test_legacy_region_is_artificially_quiet():
    """Sanity: the LEGACY path (flux*=T) really does under-produce the variance
    (true var T^2/ivar), which is the bug being fixed.  This guards against the
    two paths accidentally becoming identical."""
    npix = 200
    sigma = 0.5
    ivar = np.full(npix, 1.0 / sigma ** 2)
    S = np.zeros(npix)
    T = np.full(npix, 0.3)
    ntrial = 3000
    legacy = np.empty((ntrial, npix))
    for k in range(ntrial):
        Fobs = S + np.random.default_rng(k).standard_normal(npix) * sigma
        legacy[k] = Fobs * T                      # exactly the old behaviour
    # legacy variance should be ~ T^2 * sigma^2, far below 1/ivar
    assert np.isclose(legacy.var(0).mean(), (T[0] ** 2) * sigma ** 2, rtol=0.1)
    assert legacy.var(0).mean() < 0.2 * sigma ** 2


# ---------------------------------------------------------------------------
# 2. Edge handling: masked (ivar<=0), T==1, T->0
# ---------------------------------------------------------------------------
def test_masked_pixels_untouched_and_no_nans():
    flux = np.array([5.0, 2.0, 2.0, -1.0, 9.9])
    ivar = np.array([0.0, 4.0, 4.0, 4.0, np.nan])   # pixels 0,4 masked
    T = np.array([0.3, 1.0, 0.0, 0.5, 0.2])
    out = attenuate_with_noise(flux, ivar, T, np.random.default_rng([1, 2, 0]))
    assert out[0] == 5.0            # ivar==0 -> untouched
    assert out[4] == 9.9            # ivar nan -> untouched
    assert out[1] == 2.0            # T==1 -> unchanged (eps term vanishes)
    assert np.isfinite(out[2])      # T==0 -> pure noise, finite (no NaN)
    assert np.all(np.isfinite(out))


def test_above_limit_pixels_bit_identical():
    """T==1 pixels (above the observed Lyman limit) must be byte-identical to input."""
    flux = np.array([1.0, 2.5, -0.3, 4.2])
    ivar = np.array([4.0, 4.0, 4.0, 4.0])
    T = np.ones_like(flux)
    out = attenuate_with_noise(flux, ivar, T, np.random.default_rng([3, 3, 0]))
    assert np.array_equal(out, flux)


def test_no_negative_variance_extreme_tau():
    """Very deep tau (T -> 0) must not produce NaN / negative variance."""
    flux = np.full(50, 1.0)
    ivar = np.full(50, 4.0)
    T = np.full(50, 1e-12)
    out = attenuate_with_noise(flux, ivar, T, np.random.default_rng([4, 4, 0]))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------
def test_determinism_same_seed_identical():
    flux = np.random.default_rng(0).standard_normal(500) + 3.0
    ivar = np.full(500, 4.0)
    T = np.linspace(1.0, 0.1, 500)
    a = attenuate_with_noise(flux, ivar, T, row_rng(99, 12345, 0))
    b = attenuate_with_noise(flux, ivar, T, row_rng(99, 12345, 0))
    assert np.array_equal(a, b)


def test_determinism_seed_and_key_sensitivity():
    flux = np.random.default_rng(0).standard_normal(500) + 3.0
    ivar = np.full(500, 4.0)
    T = np.linspace(1.0, 0.1, 500)
    base = attenuate_with_noise(flux, ivar, T, row_rng(99, 12345, 0))
    assert not np.array_equal(base, attenuate_with_noise(flux, ivar, T, row_rng(100, 12345, 0)))
    assert not np.array_equal(base, attenuate_with_noise(flux, ivar, T, row_rng(99, 12346, 0)))
    assert not np.array_equal(base, attenuate_with_noise(flux, ivar, T, row_rng(99, 12345, 1)))


# ---------------------------------------------------------------------------
# 4. End-to-end inject_file: legacy byte-identity, new path, ivar untouched
# ---------------------------------------------------------------------------
def _make_synthetic_spectra(path, tids, wave, flux, ivar):
    """Minimal spectra-16-style FITS with a B camera only (R/Z absent -> skipped)."""
    prim = fits.PrimaryHDU()
    fmap = fits.BinTableHDU(
        fits.FITS_rec.from_columns([fits.Column(name="TARGETID", format="K", array=np.asarray(tids))]),
        name="FIBERMAP",
    )
    hdus = [prim, fmap,
            fits.ImageHDU(np.asarray(wave, float), name="B_WAVELENGTH"),
            fits.ImageHDU(np.asarray(flux, float), name="B_FLUX"),
            fits.ImageHDU(np.asarray(ivar, float), name="B_IVAR")]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _setup(tmp_path):
    z_abs, z_qso, logN = 3.4, 3.6, 17.5     # tau at limit = 10^17.5 * SIGMA_912 ~ 2 (depth 0.86)
    edge = LYMAN_LIMIT * (1 + z_abs)
    wave = np.linspace(3600.0, edge + 400.0, 900)   # spans below & above the limit
    tids = [1001, 1002]                              # 1001 has HCD, 1002 does not
    rng = np.random.default_rng(0)
    flux = 3.0 + rng.standard_normal((2, wave.size)) * 0.5
    ivar = np.full((2, wave.size), 4.0)
    ivar[0, 100:130] = 0.0                           # a masked stretch on the HCD sightline
    src = tmp_path / "spectra-16-0.fits"
    _make_synthetic_spectra(src, tids, wave, flux, ivar)
    hcd_by_tid = {1001: (np.array([z_abs]), np.array([logN]))}
    zq_of = {1001: z_qso, 1002: z_qso}
    return src, wave, flux, ivar, hcd_by_tid, zq_of, z_abs, logN


def test_legacy_reproduces_old_behaviour_byte_identical(tmp_path):
    src, wave, flux0, ivar0, hcd, zq, z_abs, logN = _setup(tmp_path)
    out = tmp_path / "legacy" / "spectra-16-0.fits"
    inject_file(src, out, hcd, zq, legacy_noise=True, seed=123)
    with fits.open(out) as h:
        got = np.asarray(h["B_FLUX"].data, float)
        got_ivar = np.asarray(h["B_IVAR"].data, float)
    # independent reference of the OLD behaviour: flux *= exp(-tau_LL) on every pixel
    ref = flux0.copy()
    ref[0] = flux0[0] * lyc_transmission(wave, np.array([z_abs]), np.array([logN]))
    assert np.array_equal(got, ref)                 # byte-identical
    assert np.array_equal(got_ivar, ivar0)          # ivar untouched


def test_new_path_keeps_ivar_and_untouched_masked(tmp_path):
    src, wave, flux0, ivar0, hcd, zq, z_abs, logN = _setup(tmp_path)
    out = tmp_path / "new" / "spectra-16-0.fits"
    inject_file(src, out, hcd, zq, legacy_noise=False, seed=123)
    with fits.open(out) as h:
        got = np.asarray(h["B_FLUX"].data, float)
        got_ivar = np.asarray(h["B_IVAR"].data, float)
    T = lyc_transmission(wave, np.array([z_abs]), np.array([logN]))
    # ivar never modified
    assert np.array_equal(got_ivar, ivar0)
    # non-HCD sightline completely unchanged
    assert np.array_equal(got[1], flux0[1])
    # above-limit pixels (T==1) unchanged on the HCD sightline
    above = T >= 1.0
    assert np.allclose(got[0][above], flux0[0][above])
    # masked stretch untouched even though below the limit
    masked = ivar0[0] <= 0
    assert np.array_equal(got[0][masked], flux0[0][masked])
    # below-limit, unmasked pixels ARE changed (noise regenerated)
    changed = (~above) & (~masked)
    assert not np.allclose(got[0][changed], flux0[0][changed])
    assert np.all(np.isfinite(got))


def test_inject_file_deterministic_same_seed(tmp_path):
    src, *_rest, hcd, zq, _z, _n = _setup(tmp_path)
    o1 = tmp_path / "a" / "spectra-16-0.fits"
    o2 = tmp_path / "b" / "spectra-16-0.fits"
    inject_file(src, o1, hcd, zq, legacy_noise=False, seed=2024)
    inject_file(src, o2, hcd, zq, legacy_noise=False, seed=2024)
    with fits.open(o1) as h1, fits.open(o2) as h2:
        assert np.array_equal(np.asarray(h1["B_FLUX"].data), np.asarray(h2["B_FLUX"].data))
    # different seed -> different flux
    o3 = tmp_path / "c" / "spectra-16-0.fits"
    inject_file(src, o3, hcd, zq, legacy_noise=False, seed=2025)
    with fits.open(o1) as h1, fits.open(o3) as h3:
        assert not np.array_equal(np.asarray(h1["B_FLUX"].data), np.asarray(h3["B_FLUX"].data))
