"""Core Lyman-continuum bound-free opacity physics (reusable).

The HI photoionization cross-section below the limit is a power law,
    sigma(nu) = SIGMA_912 * (nu/nu_912)^-beta  ==  SIGMA_912 * (lambda_rest/912)^beta ,
with beta=3 the near-threshold hydrogenic value (Verner+1996; correct for a discrete-absorber
INJECTION) and beta=2.75 the Worseck+2014 population-averaged effective slope. A single absorber
at redshift z_abs, column N_HI, imposes on the observed spectrum
    tau_LL(lambda_obs) = N_HI * sigma_912 * (lambda_obs / (912*(1+z_abs)))^beta   for lambda_obs < 912*(1+z_abs).
The mean-transmission observable is tau_eff,LL = -ln<exp(-tau)> (Meiksin & Madau 1993).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

SIGMA_912 = 6.35e-18   # cm^2, HI photoionization cross-section at 1 Ryd (Verner et al. 1996)
LYMAN_LIMIT = 911.76   # Angstrom
BETA_LL = 3.0          # cross-section index sigma ~ nu^-BETA (3 = Verner/PWO09; 2.75 = Worseck14)
C_KMS = 299792.458


@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM; ONE shared instance keeps the counting (dX) and drop (dl) channels consistent."""
    H0: float = 70.0
    Om: float = 0.3        # standalone default; the JOINT fit MUST pass Cosmology(Om=cfg.Omega_m)
                           # so the counting (dX) and drop (dl) channels share one cosmology.
    def Ez(self, z):
        return np.sqrt(self.Om * (1 + np.asarray(z, float)) ** 3 + (1 - self.Om))
    def proper_distance(self, z_lo, z_hi, n=512):
        """Proper (physical) distance between z_lo<z_hi in Mpc."""
        zz = np.linspace(z_lo, z_hi, n)
        integ = C_KMS / (self.H0 * (1 + zz) * self.Ez(zz))
        return float(np.sum(0.5 * (integ[1:] + integ[:-1]) * np.diff(zz)))


DEFAULT_COSMO = Cosmology()


def sigma_ll(lambda_rest, beta=BETA_LL):
    """Bound-free cross-section (cm^2) at absorber-rest wavelength lambda_rest (<912 Å); 0 above."""
    lr = np.asarray(lambda_rest, float)
    return np.where(lr < LYMAN_LIMIT, SIGMA_912 * (lr / LYMAN_LIMIT) ** beta, 0.0)


def tau_ll(wave_obs, nhi, z_abs, beta=BETA_LL):
    """LyC optical depth on the observed grid from ONE absorber (log or linear N accepted)."""
    wave_obs = np.asarray(wave_obs, float)
    edge = LYMAN_LIMIT * (1.0 + z_abs)
    N = 10.0 ** nhi if np.ndim(nhi) == 0 and nhi < 30 else np.asarray(nhi, float)
    tau = np.zeros_like(wave_obs)
    below = wave_obs < edge
    tau[below] = N * SIGMA_912 * (wave_obs[below] / edge) ** beta
    return tau


def lyc_optical_depth(wave_obs, z_abs, nhi, beta=BETA_LL):
    """Total LyC optical depth summed over a sightline's HCDs (arrays z_abs, nhi)."""
    wave_obs = np.asarray(wave_obs, float)
    tau = np.zeros_like(wave_obs)
    for zk, nk in zip(np.atleast_1d(z_abs), np.atleast_1d(nhi)):
        edge = LYMAN_LIMIT * (1.0 + zk)
        below = wave_obs < edge
        N = 10.0 ** nk if nk < 30 else nk
        tau[below] += N * SIGMA_912 * (wave_obs[below] / edge) ** beta
    return tau


def lyc_transmission(wave_obs, z_abs, nhi, beta=BETA_LL):
    """exp(-tau) for a sightline's HCDs — the multiplicative LyC attenuation for injection."""
    return np.exp(-lyc_optical_depth(wave_obs, z_abs, nhi, beta=beta))


def effective_opacity(N, fN, beta_unused=None):
    """Population effective LyC opacity per unit path: INT f(N)(1-e^{-N sigma_912}) dN
    (the drop model integrand; N linear, fN = f(N) on the same grid). Returns the integral."""
    N = np.asarray(N, float); fN = np.asarray(fN, float)
    integrand = fN * (1.0 - np.exp(-N * SIGMA_912))
    return float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(N)))


def tau_eff_kernel_basis(z912, z_q, beta=BETA_LL, cosmo=DEFAULT_COSMO):
    """The tau_eff,LL(z912,z_q) SHAPE (everything except kappa_912): matched to sigma~nu^-beta.
    tau_eff = kappa_912 * (c/H0) * (1+z912)^beta * INT_{z912}^{z_q} (1+z')^-(beta+2.5) dz'."""
    e = beta + 2.5
    z912 = np.asarray(z912, float)
    integral = ((1 + z912) ** (1 - e) - (1 + z_q) ** (1 - e)) / (e - 1)
    return (C_KMS / cosmo.H0) * (1 + z912) ** beta * integral


def fit_kappa(z912, tau, z_q, beta=BETA_LL, cosmo=DEFAULT_COSMO):
    """Least-squares kappa_912 for tau = kappa*basis (through origin)."""
    b = tau_eff_kernel_basis(z912, z_q, beta=beta, cosmo=cosmo)
    good = np.isfinite(tau) & np.isfinite(b) & (b > 0)
    if good.sum() < 3:
        return np.nan
    return float(np.sum(b[good] * tau[good]) / np.sum(b[good] ** 2))


def lambda_mfp_from_kappa(kappa, z_q, beta=BETA_LL, cosmo=DEFAULT_COSMO):
    """Proper distance (Mpc) from z_q to the z912 where tau_eff=1, given kappa_912."""
    zg = np.linspace(max(z_q - 1.5, 0.0), z_q - 1e-3, 3000)
    tau = kappa * tau_eff_kernel_basis(zg, z_q, beta=beta, cosmo=cosmo)
    if not np.isfinite(tau[0]) or tau[0] < 1:
        return np.nan
    z_tau1 = float(np.interp(1.0, tau[::-1], zg[::-1]))
    return cosmo.proper_distance(z_tau1, z_q)


def break_matched_filter_snr(wave_obs, ivar, cont, mask, nhi, z_abs,
                             beta=BETA_LL, wave_min=3590.0):
    """Per-sightline matched-filter detection S/N of the LyC break of ONE absorber, given the
    (best-case, true) continuum: S/N = sqrt( sum_pix ivar * (cont*(1-e^{-tau}))^2 ) below the
    absorber's observed Lyman limit. High S/N (>~3-5) => the break is a usable per-sightline
    signal (a break-aware finder can detect the LLS by it). Returns np.nan if no coverage."""
    wave_obs = np.asarray(wave_obs, float)
    edge = LYMAN_LIMIT * (1.0 + z_abs)
    good = ((wave_obs < edge) & (wave_obs > wave_min) & (mask == 0)
            & (ivar > 0) & np.isfinite(cont) & (cont > 0))
    if good.sum() < 5:
        return np.nan
    tau = (10.0 ** nhi if nhi < 30 else nhi) * SIGMA_912 * (wave_obs[good] / edge) ** beta
    decrement = cont[good] * (1.0 - np.exp(-tau))
    return float(np.sqrt(np.sum(ivar[good] * decrement ** 2)))
