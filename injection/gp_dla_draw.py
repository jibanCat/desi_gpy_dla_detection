"""
examples/inject/gp_dla_draw.py
==============================
M3 cross-check arm (method i): draw a spectrum from the GP+DLA generative model.

    s ~ NullGP(μ, K)   for a given (z_qso, SNR)
    s *= Voigt(logN, z_dla)            # optional injected DLA

This is the fully-controlled, no-clean-sightline-selection arm. It isolates the
INFERENCE self-consistency (no real Lyα forest), so the difference between the
coadd-injection arm (method ii) and this arm quantifies the real-forest
contribution to bias / incompleteness — most informative at NHI<19.

Discipline
----------
* The NullGP model pieces (μ, the low-rank basis M, the rest-frame
  interpolators) are consumed READ-ONLY — ``null_gp`` / ``dla_gp`` are never
  modified.
* The DLA imprint reuses ``gpy_dla_detection.inject_absorber.inject_voigt`` (the
  same Voigt the coadd arm and the GP forward model use), so both injection arms
  are byte-for-byte consistent in the absorber profile.

Sampling math
-------------
On the observed grid (rest = obs / (1 + z_qso)) we build the per-pixel mean
``this_mu`` and the low-rank basis ``this_M`` (n, k) via the model's
interpolators, so K = this_M this_Mᵀ. A draw is

    s = this_mu + this_M @ η + σ ⊙ ξ ,   η ~ N(0, I_k),  ξ ~ N(0, I_n)

where the per-pixel noise σ comes from EITHER:

* ``noise_variance`` — an explicit λ-dependent variance template V(λ)=1/ivar(λ),
  e.g. a representative clean-sightline ivar at the target SNR (PREFERRED — real
  DESI forest ivar is λ-dependent, noisier in the blue), or
* ``snr`` — an IDEALIZED FLAT level σ = median(this_mu)/snr (fallback).

This is exactly s ~ N(this_mu, K + V) with V = diag(σ²).

Caveat on method (i) vs (ii)
----------------------------
The FLAT-σ (``snr``) draw is an *idealized-noise self-consistency* arm. Because
real DESI forest noise is λ-dependent, the (ii)−(i) difference (coadd-injection
minus this arm) conflates the real-forest absorption term with a NOISE-MODEL
mismatch unless (i) is drawn with the SAME λ-dependent noise. To attribute
(ii)−(i) to the forest, pass a representative ``noise_variance`` template here so
both arms share the noise model; otherwise scope (i) explicitly as
idealized-noise self-consistency and do NOT subtract it as the forest term.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gpy_dla_detection.inject_absorber import inject_voigt


def _interp_mu_M(model, rest_wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate the model's μ and low-rank basis M onto a rest-frame grid.

    Uses the model's own ``mu_interpolator`` and ``M_interpolator`` (the exact
    read-only surface ``null_gp.NullGP`` exposes), so the sampled covariance is
    the GP's K = M Mᵀ, restricted to the observed pixels.
    """
    this_mu = np.asarray(model.mu_interpolator(rest_wavelengths), dtype=np.float64)
    this_M = np.asarray(model.M_interpolator(rest_wavelengths), dtype=np.float64)
    if this_M.ndim != 2:
        raise ValueError(
            f"M_interpolator must return (n, k); got shape {this_M.shape}"
        )
    return this_mu, this_M


def draw_gp_dla_spectrum(
    model,
    *,
    z_qso: float,
    observed_wavelengths: np.ndarray,
    snr: Optional[float] = None,
    noise_variance: Optional[np.ndarray] = None,
    logN: Optional[float] = None,
    z_dla: Optional[float] = None,
    num_lines: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a GP(+DLA) spectrum on a given observed grid (M3 cross-check arm).

    Provide EITHER ``noise_variance`` (a per-pixel, λ-dependent template —
    preferred, faithful to real DESI forest noise) OR ``snr`` (an idealized FLAT
    level). Exactly one is required (see the module docstring for the method-(i)
    vs (ii) caveat).

    Parameters
    ----------
    model : object
        A NullGP-like model exposing ``mu_interpolator(rest)`` and
        ``M_interpolator(rest) -> (n, k)`` (e.g. ``null_gp.NullGP`` or
        ``NullGPMAT``). Consumed READ-ONLY.
    z_qso : float
        QSO redshift (sets the rest frame: rest = obs / (1 + z_qso)).
    observed_wavelengths : (n,) array
        Observed-frame wavelength grid (Å) to draw the spectrum on.
    snr : float or None
        Idealized-FLAT native SNR: per-pixel noise σ = median(this_mu)/snr.
        Ignored if ``noise_variance`` is given. Use a very large value to suppress
        sampling noise (isolate the DLA).
    noise_variance : (n,) array or None
        PREFERRED: explicit per-pixel noise variance V(λ)=1/ivar(λ) (e.g. a
        representative clean-sightline ivar at the target SNR), giving the GP the
        λ-dependent forest noise. Takes precedence over ``snr`` if both are given.
    logN : float or None
        log10 N_HI of the injected DLA; ``None`` → no DLA (a clean draw).
    z_dla : float or None
        Absorber redshift; required when ``logN`` is given.
    num_lines : int
        Lyman-series line count for the DLA Voigt (match NUM_FOREST_LINES).
    rng : numpy.random.Generator or None
        Random generator (for reproducibility). ``None`` → fresh default_rng.

    Returns
    -------
    (wavelengths, flux, noise_variance) : tuple of (n,) arrays
        ``wavelengths`` echoes ``observed_wavelengths``; ``flux`` is the sampled
        (and optionally DLA-absorbed) spectrum; ``noise_variance`` = σ² per pixel
        (the supplied template, or the flat σ² derived from ``snr``).
    """
    if rng is None:
        rng = np.random.default_rng()

    obs = np.asarray(observed_wavelengths, dtype=np.float64)
    if obs.ndim != 1:
        raise ValueError("observed_wavelengths must be 1-D")
    rest = obs / (1.0 + z_qso)

    this_mu, this_M = _interp_mu_M(model, rest)
    n, k = this_M.shape

    if noise_variance is not None:
        # PREFERRED: per-pixel λ-dependent variance template.
        nvar = np.asarray(noise_variance, dtype=np.float64)
        if nvar.shape != (n,):
            raise ValueError(
                f"noise_variance shape {nvar.shape} != ({n},) (observed grid)"
            )
        if np.any(nvar < 0):
            raise ValueError("noise_variance must be non-negative")
        sigma = np.sqrt(nvar)
        noise_variance_out = nvar
    elif snr is not None:
        # Fallback: idealized FLAT noise from the requested SNR.
        mean_level = float(np.nanmedian(this_mu))
        if not np.isfinite(mean_level) or mean_level == 0.0:
            mean_level = 1.0
        sigma = np.full(n, abs(mean_level) / float(snr), dtype=np.float64)
        noise_variance_out = sigma ** 2
    else:
        raise ValueError(
            "provide exactly one of `noise_variance` (per-pixel template, "
            "preferred) or `snr` (idealized flat noise)"
        )

    # s ~ N(this_mu, K + V):  low-rank draw + diagonal (possibly λ-dependent) noise.
    eta = rng.standard_normal(k)
    xi = rng.standard_normal(n)
    flux = this_mu + this_M @ eta + sigma * xi

    # Optional DLA imprint via the same Voigt the GP / coadd arm use.
    if logN is not None:
        if z_dla is None:
            raise ValueError("z_dla is required when logN is given")
        flux = inject_voigt(obs, flux, 10.0 ** float(logN), float(z_dla), num_lines=num_lines)

    return obs, flux, noise_variance_out
