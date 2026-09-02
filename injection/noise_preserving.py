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

Prescription B (PI ruling 2026-08-28/29 item 6, the injection-prescription gate of the MAX4
repair cycle), `method="residual_preserving"`:

    F' = T * S_est + (F_obs - S_est),      S_est = signal_estimate(F_obs)

the smooth signal estimate is absorbed by the profile and the observed residual (pixel noise
PLUS the forest structure below the smoothing scale) is carried through unchanged — no
synthetic noise, no seed. Outside the profile (T == 1 exactly) and where S_est is undefined
the observed flux is kept bit-for-bit. Algebraically this is F_obs + (T - 1) S_est, i.e. the
same operation as the documented `signal_estimate` alternative; it is exposed under its own
name so the A (variance-preserving) vs B (residual-preserving) gate is explicit and testable.
The default remains `variance_preserving` (prescription A, the current fiducial).

Mean-flux rescaling (R-041B): the forest SIGNAL is rescaled by r(lambda) =
exp(-(tau_alt - tau_fid)) using a smooth signal estimate S (running median + Gaussian
smoothing over unmasked pixels), F_obs -> F_obs + (r - 1) S, which leaves the real noise
in place; the DLA is then injected with the variance-preserving operation. Real spectra
have no latent noise-free flux, so S is an estimate; its smoothing scale is a documented
parameter and the suite reports the residual statistics.

DETERMINISTIC STATE OF PRESCRIPTION A (documented 2026-09-01, PI follow-up; no behaviour change)
- The noise draw is eps = numpy.random.default_rng(int(seed)).standard_normal(n_pix), one
  vector per call of inject_noise_preserving, indexed by pixel of the spectrum's own grid.
  There is no other entropy source: the same (inputs, seed) give the same F' bit-for-bit.
- `seed` is a plain keyword argument (default 0). Who sets it:
    * the REAL-SPECTRUM archive route, tools/r041_build_archive.py (R-041A fiducial / cmp,
      R-041B mean-flux, R-041D pairs), passes NO seed -> seed = 0 for EVERY sightline of every
      wave. Because all archive spectra share one wavelength grid, every sightline of a wave
      receives the SAME eps vector (pixel-aligned), scaled by its own 1/sqrt(ivar); two
      injections at the same observed wavelength on different sightlines therefore carry the
      same trough-noise pattern. This is deterministic but NOT the "seeded per sightline"
      wording of the 2026-08-28 progress record — recorded here as a fact, not changed
      (a per-sightline seed would alter every existing archive's bytes and is a PI decision).
    * the MOCK coadd route, injection/coadd_injection.py (R-041C/E), seeds per
      (seed_salt, target_id, camera): seed = first 4 bytes of sha256(f"{salt}:{tid}:{cam}").
- The injection PLAN (which z, log N per sightline) is realised in tools/r041_plan.py with
  numpy default_rng(seed_for(TARGETID, k, seed_salt)) — independent of eps.
- Rebuilding a wave from the same plan + source archive + code reproduces the injected flux
  arrays bit-for-bit (tests/test_r041_build_archive_methods.py); the HDF5 container bytes
  may still differ through h5py/HDF5 object timestamps — the build summary therefore records
  the archive sha256 of the build that was actually run.
- Prescription B (residual_preserving) draws no noise and ignores `seed`.

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
      'residual_preserving' (prescription B, gate item 6): F' = T S_r + (F_r - S_r) with
          F_r = F + (r-1) S the (optionally mean-flux-rescaled) flux and S_r = r S its signal
          estimate, so the observed residual F - S is untouched; T == 1 pixels and pixels with
          undefined S keep F_r bit-for-bit; no synthetic noise (seed unused).
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
    elif method == "residual_preserving":
        ok = np.isfinite(S)
        S_sig = S.copy()
        if r is not None:
            S_sig[ok] = np.asarray(r, float)[ok] * S[ok]
        resid = base - S_sig                             # == F - S wherever S is defined
        out = base.copy()
        inside = ok & (T != 1.0)
        out[inside] = T[inside] * S_sig[inside] + resid[inside]
        parts = dict(T=T, S=S, eps=None, resid=resid)
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
    # --- P1 mean-flux CONTROL variants (PI ruling 2026-09-02 §9-§12; spec MAX4_MEANFLUX_CONTROL_SPEC_2026-09-02.md §2) -------------
    # Relative envelope s(z) about the finder fiducial (= the Turner et al. 2024 fit): sigma_tot/tau of the nearest Turner bin for
    # 2.05 <= z <= 4.15 (Table 3 of arXiv:2405.06743), held at the last-bin value (4.2 %) to z = 4.2, rising linearly to 11 % at z = 4.5
    # (the Ding, Madau & Prochaska 2024 stated 10-12 % precision at z > 4; arXiv:2310.00524) and constant beyond. No (tau0, gamma)
    # covariance is published, so the power-law parameters are NOT pushed jointly.
    "turner2024_m1s": {"form": "finder_fiducial*(1-s(z))", "source": "Turner+2024 Table 3 binned sigma_tot/tau (2.05-4.15); Ding+2024 10-12 % beyond 4.2"},
    "turner2024_p1s": {"form": "finder_fiducial*(1+s(z))", "source": "idem"},
    # Ding+2024 spline points (metal- and optically-thick-corrected tau_Lya, mean of the posterior; Table 3 of arXiv:2310.00524v2), log-linear
    # between points, extrapolated above z = 4.20 with the last-segment slope d ln tau / d ln(1+z) = ln(1.09/0.85)/ln(5.20/4.94) = 4.85.
    "ding2024_hz": {"form": "spline(Ding+2024) ; tau = 1.09*((1+z)/5.20)**4.85 above z = 4.20", "source": "Ding, Madau & Prochaska 2024, MNRAS 532, 2082"},
}

TURNER2024_BINS = [(2.05, 0.147, 0.012), (2.15, 0.158, 0.012), (2.25, 0.179, 0.015), (2.35, 0.200, 0.016), (2.45, 0.226, 0.016), (2.55, 0.235, 0.018),
                   (2.65, 0.268, 0.019), (2.75, 0.292, 0.020), (2.85, 0.316, 0.021), (2.95, 0.342, 0.022), (3.05, 0.373, 0.023), (3.15, 0.410, 0.023),
                   (3.25, 0.455, 0.022), (3.35, 0.498, 0.025), (3.45, 0.527, 0.030), (3.55, 0.579, 0.032), (3.65, 0.638, 0.031), (3.75, 0.694, 0.032),
                   (3.85, 0.770, 0.033), (3.95, 0.830, 0.034), (4.05, 0.854, 0.036), (4.15, 0.928, 0.039)]
DING2024_SPLINE = [(2.50, 0.27), (2.64, 0.32), (2.80, 0.38), (3.11, 0.46), (3.27, 0.52), (3.42, 0.58), (3.63, 0.68), (3.94, 0.85), (4.20, 1.09)]


def meanflux_envelope_s(z):
    """Relative 1-sigma envelope s(z) of the fiducial tau_eff (spec §2): nearest-bin sigma_tot/tau from Turner+2024 for z <= 4.2 (constant 8.2 %
    below the first bin), then linear from 4.2 % at z = 4.2 to 11 % at z = 4.5, constant beyond."""
    z = np.asarray(z, float)
    zc = np.array([b[0] for b in TURNER2024_BINS]); rel = np.array([b[2] / b[1] for b in TURNER2024_BINS])
    s = rel[np.abs(z[..., None] - zc[None, :]).argmin(axis=-1)]
    s = np.where(z > 4.2, np.interp(z, [4.2, 4.5], [rel[-1], 0.11]), s)
    return s


def taueff_ding2024(z):
    z = np.asarray(z, float)
    zs = np.array([p[0] for p in DING2024_SPLINE]); ts = np.log(np.array([p[1] for p in DING2024_SPLINE]))
    lo = np.exp(np.interp(z, zs, ts))
    hi = 1.09 * ((1.0 + z) / 5.20) ** 4.85
    return np.where(z > 4.20, hi, np.where(z < zs[0], np.exp(ts[0] + (ts[1] - ts[0]) / np.log((1 + zs[1]) / (1 + zs[0])) * np.log((1 + z) / (1 + zs[0]))), lo))


def taueff(model):
    m = TAUEFF_MODELS[model]
    if model == "becker2013":
        return lambda z: np.maximum(0.751 * ((1.0 + np.asarray(z, float)) / 4.5) ** 2.90 - 0.132, 0.0)
    if model in ("turner2024_m1s", "turner2024_p1s"):
        fid = taueff("finder_fiducial"); sign = -1.0 if model.endswith("m1s") else 1.0
        return lambda z: fid(z) * (1.0 + sign * meanflux_envelope_s(z))
    if model == "ding2024_hz":
        return taueff_ding2024
    return lambda z: m["tau0"] * (1.0 + np.asarray(z, float)) ** m["beta"]
