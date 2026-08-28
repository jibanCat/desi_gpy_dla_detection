"""noise_preserving.py — DLA injection that preserves a realistic, ivar-consistent noise
realization (Paper-1 high-z repair cycle, R-041A item A3, 2026-08-28).

The established primitive (gpy_dla_detection.inject_absorber.inject_voigt) does
F_injected = F_observed * T with T = exp(-tau_DLA): it attenuates the observed noise
together with the signal, so a saturated injected trough is noiseless (F ~ 0 exactly),
unlike a real DLA trough (zero signal PLUS the real noise).

Corrected operation (the PI's F_obs + (T - 1) S with the noise term made exactly
variance-preserving). Write F_obs = S + n with n the noise (variance 1/ivar, the finder's
own noise model). The absorber must act on S only; n must keep its variance:

    F' = T * F_obs + sqrt(1 - T^2) * eps / sqrt(ivar),    eps ~ N(0, 1) (seeded)

so that F' = T S + [T n + sqrt(1 - T^2) eps/sqrt(ivar)] and the bracket has variance
exactly 1/ivar at every pixel: inside a saturated trough (T = 0) the flux is a fresh
ivar-consistent noise realization, outside the profile (T = 1) the spectrum is untouched
bit-for-bit, and in the wings the real and the synthetic noise mix with the right total
variance. This needs NO estimate of S (the observed forest structure is absorbed by T
along with the signal, as in a real DLA), which is why it was preferred over the
smoothing-based estimate: on these high-z spectra the forest structure below any
smoothing scale is ~1.7x the pixel noise and would have survived inside the troughs
(measured by the validation suite, kept as the documented rejected alternative
`method="signal_estimate"`).

Mean-flux rescaling (R-041B): the forest SIGNAL is rescaled by r(lambda) =
exp(-(tau_alt - tau_fid)) using a smooth signal estimate S (running median + Gaussian
smoothing over unmasked pixels), F_obs -> F_obs + (r - 1) S, which leaves the real noise
in place; the DLA is then injected with the variance-preserving operation. Real spectra
have no latent noise-free flux, so S is an estimate; its smoothing scale is a documented
parameter and the suite reports the residual statistics.

ivar, mask, resolution and the wavelength grid are never modified. The Voigt transmission
is the SAME frozen primitive the finder and the old campaign used (voigt_transmission).
Limitation: the synthetic noise component is white (diagonal ivar) — the same assumption
the finder's likelihood makes; pixel-to-pixel noise correlations of the coadd are not
reproduced inside troughs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from gpy_dla_detection.inject_absorber import voigt_transmission

LYA_REST = 1215.67
DEFAULT_MEDIAN_PX = 9
DEFAULT_SIGMA_PX = 2.5


def signal_estimate(flux, ivar, mask, median_px=DEFAULT_MEDIAN_PX, sigma_px=DEFAULT_SIGMA_PX):
    """Smooth estimate S of the underlying signal of one spectrum.

    Pixels with mask != 0, ivar <= 0 or non-finite flux are excluded from the estimate and
    linearly interpolated over (for the estimate only; they keep their observed values in
    the injected spectrum). Returns S with the same shape as flux (NaN where the whole
    spectrum is unusable)."""
    flux = np.asarray(flux, float)
    good = np.isfinite(flux) & np.isfinite(ivar) & (np.asarray(ivar, float) > 0) & (np.asarray(mask) == 0)
    if good.sum() < 10:
        return np.full_like(flux, np.nan)
    x = np.arange(flux.size)
    f = np.interp(x, x[good], flux[good])
    m = median_filter(f, size=int(median_px), mode="nearest")
    return gaussian_filter1d(m, float(sigma_px), mode="nearest")


def transmission(wave, absorbers, num_lines=3):
    """Product of the frozen Voigt transmissions of every absorber ({'nhi' linear, 'z_dla'})."""
    T = np.ones(np.asarray(wave).size)
    for ab in absorbers:
        T = T * voigt_transmission(np.asarray(wave, float), float(ab["nhi"]), float(ab["z_dla"]),
                                   int(ab.get("num_lines", num_lines)))
    return T


def meanflux_ratio(wave, z_qso, taueff_alt, taueff_fid):
    """r(lambda) = exp(-(tau_alt - tau_fid)) evaluated at the Lyα absorption redshift of each
    pixel, applied only blueward of the quasar's Lyα emission (the forest); 1 elsewhere.
    taueff_* are callables tau_eff(z)."""
    wave = np.asarray(wave, float)
    z_abs = wave / LYA_REST - 1.0
    r = np.ones_like(wave)
    forest = (wave < LYA_REST * (1.0 + z_qso)) & (z_abs > 0)
    r[forest] = np.exp(-(taueff_alt(z_abs[forest]) - taueff_fid(z_abs[forest])))
    return r


def inject_noise_preserving(wave, flux, ivar, mask, absorbers, *, z_qso=None, r=None, seed=0,
                            num_lines=3, median_px=DEFAULT_MEDIAN_PX, sigma_px=DEFAULT_SIGMA_PX,
                            method="variance_preserving", return_parts=False):
    """Corrected injection. method:
      'variance_preserving' (default): F' = T (F + (r-1) S) + sqrt(1 - T^2) eps / sqrt(ivar)
      'signal_estimate' (rejected alternative, kept for the record): F' = F + (T r - 1) S
    ivar/mask are returned unchanged. eps is drawn from numpy default_rng(seed) so a wave is
    reproducible from its plan + seed. Pixels with ivar <= 0 or mask != 0 get no synthetic
    noise (they keep T * F)."""
    flux = np.asarray(flux, float)
    ivar = np.asarray(ivar, float)
    T = transmission(wave, absorbers, num_lines)
    S = signal_estimate(flux, ivar, mask, median_px, sigma_px)
    base = flux.copy()
    if r is not None:                                    # mean-flux rescaling of the forest signal (noise untouched)
        ok = np.isfinite(S)
        base[ok] = flux[ok] + (np.asarray(r, float)[ok] - 1.0) * S[ok]
    if method == "signal_estimate":
        out = base.copy()
        ok = np.isfinite(S)
        out[ok] = base[ok] + (T[ok] - 1.0) * S[ok]
        parts = dict(T=T, S=S, eps=None)
    elif method == "variance_preserving":
        rng = np.random.default_rng(int(seed))
        eps = rng.standard_normal(flux.size)
        good = np.isfinite(ivar) & (ivar > 0) & (np.asarray(mask) == 0)
        sig = np.zeros_like(flux)
        sig[good] = 1.0 / np.sqrt(ivar[good])
        out = T * base + np.sqrt(np.clip(1.0 - T ** 2, 0.0, 1.0)) * eps * sig
        parts = dict(T=T, S=S, eps=eps)
    else:
        raise ValueError(method)
    if return_parts:
        return out, parts
    return out


def inject_multiplicative(wave, flux, absorbers, num_lines=3):
    """The OLD operation (F * T), kept only for the old-vs-corrected comparison."""
    return np.asarray(flux, float) * transmission(wave, absorbers, num_lines)


# ---- effective optical depth models for the mean-flux sensitivity (R-041B) -----------------
# Only supported literature / project inputs; each is a power law tau_eff = tau_0 (1+z)^beta
# except Becker+13. References to be verified by the paper lane before citation.
TAUEFF_MODELS = {
    "finder_fiducial": {"form": "tau0*(1+z)**beta", "tau0": 0.00246, "beta": 3.62,
                        "source": "the deployed finder's mean-flux suppression (PREV_TAU_0 / PREV_BETA of the production config; Kim et al. 2007 form)"},
    "kim2007": {"form": "tau0*(1+z)**beta", "tau0": 0.0023, "beta": 3.65,
                "source": "Kim et al. 2007 (the finder module default; dla_meanflux_gp.py)"},
    "fg2008": {"form": "tau0*(1+z)**beta", "tau0": 0.0018, "beta": 3.92,
               "source": "Faucher-Giguere et al. 2008, ApJ 681, 831 (0.0018 (1+z)^3.92)"},
    "becker2013": {"form": "0.751*((1+z)/4.5)**2.90-0.132", "source": "Becker et al. 2013, MNRAS 430, 2067 (2 < z < 5)"},
}


def taueff(model):
    m = TAUEFF_MODELS[model]
    if model == "becker2013":
        return lambda z: np.maximum(0.751 * ((1.0 + np.asarray(z, float)) / 4.5) ** 2.90 - 0.132, 0.0)
    return lambda z: m["tau0"] * (1.0 + np.asarray(z, float)) ** m["beta"]
