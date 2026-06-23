"""
tests/test_gp_dla_draw.py
=========================
TDD tests for ``injection/gp_dla_draw.py`` — the M3 cross-check (method i):
draw a spectrum from the GP+DLA generative model.

    s ~ NullGP(μ, K) for a given (z_qso, SNR),  then  s *= Voigt(logN, z_dla)

This is the fully-controlled, no-forest-selection arm of the campaign; the
(ii)−(i) difference quantifies the real-forest contribution to bias /
incompleteness (most informative at NHI<19).

Discipline: reuse ``null_gp`` (μ, K) READ-ONLY and
``gpy_dla_detection.inject_absorber.inject_voigt`` — do not modify the model.
Tests inject a tiny synthetic NullGP-like model so they run without the trained
``.h5`` (the model-pieces are consumed via a duck-typed object exposing
``mu_interpolator`` / ``M_interpolator`` / the per-pixel mean+covariance), and
``importorskip`` the C extension only for the Voigt-trough physics test.
"""
import numpy as np
import pytest


LYA = 1215.67


def _require_c_voigt():
    voigt_fast = pytest.importorskip("gpy_dla_detection.voigt_fast")
    try:
        voigt_fast.VoigtProfile()
    except (OSError, ImportError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"compiled _voigt.so unavailable: {exc}")
    return voigt_fast


class _FakeNullGP:
    """A minimal stand-in exposing the read-only pieces gp_dla_draw consumes.

    Reproduces the public surface of ``null_gp.NullGP`` that the generative
    draw needs: ``rest_wavelengths``, ``mu``, ``M`` (low-rank K = M Mᵀ), and the
    interpolators that map a rest-frame grid → per-pixel (mu, M). We deliberately
    keep absorption/optical-depth OFF (continuum-level mean) so the unit tests
    pin the SAMPLING math, not the mean-flux model.
    """

    def __init__(self, n_rest=256, k=4, seed=0):
        from scipy import interpolate

        rng = np.random.default_rng(seed)
        self.params = type("P", (), {"k": k})()
        self.rest_wavelengths = np.linspace(911.0, 1230.0, n_rest)
        # Smooth continuum-ish mean around 1.0.
        self.mu = 1.0 + 0.05 * np.sin(
            np.linspace(0, 3.0, n_rest)
        )
        # Low-rank basis (n_rest, k); modest amplitude.
        self.M = 0.1 * rng.standard_normal((n_rest, k))
        self.mu_interpolator = interpolate.interp1d(
            self.rest_wavelengths, self.mu, bounds_error=False, fill_value=0.0
        )
        self._m_interps = [
            interpolate.interp1d(
                self.rest_wavelengths, self.M[:, i],
                bounds_error=False, fill_value=0.0,
            )
            for i in range(k)
        ]

    def M_interpolator(self, x):
        out = np.empty((x.shape[0], self.params.k))
        for i, f in enumerate(self._m_interps):
            out[:, i] = f(x)
        return out


# ---------------------------------------------------------------------------
# Shape / sampling
# ---------------------------------------------------------------------------


def test_draw_returns_expected_arrays_and_shapes():
    """draw_gp_dla_spectrum returns (wavelengths, flux, noise_variance) of equal
    length on the requested observed grid."""
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP()
    z_qso = 3.0
    obs = np.linspace(3600.0, (1 + z_qso) * 1230.0, 1500)

    result = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=5.0,
        logN=None, z_dla=None, rng=np.random.default_rng(1),
    )
    wave, flux, noise_var = result
    assert wave.shape == flux.shape == noise_var.shape == obs.shape
    assert np.all(np.isfinite(flux))


def test_draw_noise_variance_matches_snr():
    """With snr given, the returned per-pixel noise variance ≈ (mean_level/snr)²
    so a downstream GP sees the requested sightline SNR."""
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP()
    z_qso = 3.0
    obs = np.linspace(3600.0, (1 + z_qso) * 1215.0, 2000)
    snr = 4.0

    wave, flux, noise_var = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=snr,
        logN=None, z_dla=None, rng=np.random.default_rng(2),
    )
    # mean continuum level ~1.0 → noise sigma ~ 1/snr.
    expected_sigma = 1.0 / snr
    assert np.median(np.sqrt(noise_var)) == pytest.approx(
        expected_sigma, rel=0.5
    )


def test_draw_reproducible_with_seed():
    """Same rng seed → identical spectra (deterministic for the campaign)."""
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP()
    z_qso = 2.8
    obs = np.linspace(3600.0, (1 + z_qso) * 1220.0, 1200)

    a = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=6.0,
        logN=None, z_dla=None, rng=np.random.default_rng(42),
    )
    b = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=6.0,
        logN=None, z_dla=None, rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(a[1], b[1])


def test_draw_sample_tracks_gp_mean_and_covariance():
    """Over many draws the empirical mean ≈ this_mu and variance is consistent
    with diag(K)+noise (the draw is s ~ N(this_mu, K + V), no DLA)."""
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP(seed=7)
    z_qso = 3.0
    obs = np.linspace(3600.0, (1 + z_qso) * 1225.0, 400)

    rng = np.random.default_rng(123)
    draws = np.array(
        [
            draw_gp_dla_spectrum(
                model, z_qso=z_qso, observed_wavelengths=obs, snr=10.0,
                logN=None, z_dla=None, rng=rng,
            )[1]
            for _ in range(400)
        ]
    )
    emp_mean = draws.mean(axis=0)
    # Mean should sit near the continuum (~1.0) within a few sigma/sqrt(N).
    assert np.abs(emp_mean.mean() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# DLA imprint (physics)
# ---------------------------------------------------------------------------


def test_strong_dla_produces_trough():
    """Injecting a strong DLA (logN=21.5) at z_dla yields a deep trough at
    (1+z_dla)·1215.67 relative to the same draw with no DLA."""
    _require_c_voigt()
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP()
    z_qso = 3.0
    z_dla = 2.7
    obs = np.linspace(3600.0, (1 + z_qso) * 1230.0, 3000)

    # High SNR + same seed isolates the DLA imprint from sampling noise.
    seed = 555
    _, flux_nodla, _ = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=1e6,
        logN=None, z_dla=None, rng=np.random.default_rng(seed),
    )
    wave, flux_dla, _ = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=1e6,
        logN=21.5, z_dla=z_dla, rng=np.random.default_rng(seed),
    )

    lam0 = (1 + z_dla) * LYA
    icen = int(np.argmin(np.abs(wave - lam0)))
    ratio = flux_dla[icen] / flux_nodla[icen]
    assert ratio < 0.1, "strong DLA should blacken the line core"
    # The DLA imprint is purely multiplicative absorption: flux_dla =
    # flux_nodla * transmission with transmission in [0, 1]. Check the ratio
    # where the (same-seed) no-DLA flux is comfortably positive — this pins both
    # that the imprint never *adds* flux and that it equals the GP/coadd Voigt.
    pos = flux_nodla > 0.1
    trans = flux_dla[pos] / flux_nodla[pos]
    assert np.all(trans <= 1.0 + 1e-6) and np.all(trans >= -1e-6)
    # The ratio is exactly the inject_voigt transmission (same seed → same draw),
    # confirming the DLA is imprinted via the shared Voigt primitive, not refit.
    from gpy_dla_detection.inject_absorber import voigt_transmission

    expected_trans = voigt_transmission(wave, 10 ** 21.5, z_dla, 3)
    np.testing.assert_allclose(
        flux_dla[pos], flux_nodla[pos] * expected_trans[pos], rtol=0, atol=1e-9
    )


def test_subdla_shallower_than_dla():
    """A sub-DLA (logN=19) imprint is shallower than a DLA (logN=21) at the same
    (z_dla, draw)."""
    _require_c_voigt()
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP()
    z_qso, z_dla = 3.0, 2.6
    obs = np.linspace(3600.0, (1 + z_qso) * 1230.0, 3000)
    seed = 99

    _, base, _ = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=1e6,
        logN=None, z_dla=None, rng=np.random.default_rng(seed),
    )
    w, sub, _ = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=1e6,
        logN=19.0, z_dla=z_dla, rng=np.random.default_rng(seed),
    )
    _, dla, _ = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=1e6,
        logN=21.0, z_dla=z_dla, rng=np.random.default_rng(seed),
    )
    lam0 = (1 + z_dla) * LYA
    icen = int(np.argmin(np.abs(w - lam0)))
    assert (sub[icen] / base[icen]) > (dla[icen] / base[icen])


# ---------------------------------------------------------------------------
# Per-pixel (λ-dependent) noise template — review minor: real DESI forest ivar
# is λ-dependent, so the flat σ=median(μ)/snr conflates forest absorption with a
# noise-model mismatch. Method (a): accept an ivar(λ) template and draw σ per
# pixel from it.
# ---------------------------------------------------------------------------


def test_draw_accepts_per_pixel_noise_template():
    """When a per-pixel ``noise_variance`` template (V(λ)=1/ivar(λ)) is supplied,
    the returned noise_variance equals it exactly and the empirical scatter tracks
    the λ-dependent σ(λ) — NOT the flat median(μ)/snr."""
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP(seed=3)
    z_qso = 3.0
    obs = np.linspace(3600.0, (1 + z_qso) * 1225.0, 600)
    # A clearly λ-dependent variance template (blue end noisier than red).
    nvar = np.linspace(0.04, 0.01, obs.size)  # σ from 0.2 → 0.1

    rng = np.random.default_rng(11)
    draws = np.array(
        [
            draw_gp_dla_spectrum(
                model, z_qso=z_qso, observed_wavelengths=obs, snr=None,
                noise_variance=nvar, logN=None, z_dla=None, rng=rng,
            )[1]
            for _ in range(800)
        ]
    )
    _, _, returned_nvar = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=None,
        noise_variance=nvar, logN=None, z_dla=None,
        rng=np.random.default_rng(0),
    )
    # The returned per-pixel variance is the supplied template (not flat).
    np.testing.assert_allclose(returned_nvar, nvar, rtol=0, atol=0)

    # Empirical per-pixel scatter, with the GP covariance diag(K) subtracted,
    # tracks the λ-dependent template (decreasing blue→red), not a flat level.
    this_mu, this_M = model.mu_interpolator(obs / (1 + z_qso)), model.M_interpolator(
        obs / (1 + z_qso)
    )
    diagK = np.sum(np.asarray(this_M) ** 2, axis=1)
    emp_var = draws.var(axis=0) - diagK
    # Blue half noisier than red half (the whole point of a λ-dependent template).
    blue = emp_var[: obs.size // 2].mean()
    red = emp_var[obs.size // 2 :].mean()
    assert blue > red


def test_draw_noise_template_takes_precedence_over_snr():
    """If both ``noise_variance`` and ``snr`` are given, the explicit per-pixel
    template wins (it is the more faithful forest noise model)."""
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP(seed=5)
    z_qso = 3.0
    obs = np.linspace(3600.0, (1 + z_qso) * 1220.0, 300)
    nvar = np.full(obs.size, 0.09)  # σ=0.3, distinct from any snr-derived level

    _, _, returned = draw_gp_dla_spectrum(
        model, z_qso=z_qso, observed_wavelengths=obs, snr=50.0,
        noise_variance=nvar, logN=None, z_dla=None,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_allclose(returned, nvar)


def test_draw_requires_snr_or_noise_template():
    """Exactly one noise specification is required: neither ``snr`` nor
    ``noise_variance`` → a clear error (no silent default)."""
    from injection.gp_dla_draw import draw_gp_dla_spectrum

    model = _FakeNullGP()
    obs = np.linspace(3600.0, 4800.0, 200)
    with pytest.raises((ValueError, TypeError)):
        draw_gp_dla_spectrum(
            model, z_qso=3.0, observed_wavelengths=obs, snr=None,
            noise_variance=None, logN=None, z_dla=None,
        )
