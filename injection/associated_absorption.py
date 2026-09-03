"""DLA-associated metal-line absorption for the paired real-spectrum response discriminator (PI ruling 2026-09-03 §5–§9).

Observationally motivated, deliberately simple: each associated line is a Voigt profile with column density N_ion (cm^-2), Doppler parameter b
(km/s) and a velocity offset dv (km/s) relative to the H I redshift, optionally split into a few components; the optical depth is convolved
with a Gaussian instrumental LSF (FWHM in Å at the observed wavelength) and applied as T_metal = exp(-tau) multiplying the DLA transmission.
NOT the quickquasars construction (no Lyα-shaped profiles, no scaled Lyα optical depths).

The realization of every injected absorber is a JSON list of line dicts {"line": name, "logN": ..., "b_kms": ..., "dv_kms": ...} carried in the
plan (column metals_json) so the analysis knows exactly what was injected. Atomic data are looked up from LINES (values must be verified against a
published table before a campaign; see the frozen model document).
"""
import json
import math

import numpy as np
from scipy.special import wofz

C_KMS = 299792.458
# Atomic data: rest wavelength [Å], oscillator strength f, damping constant Gamma [s^-1]. FILLED FROM THE VERIFIED TABLE recorded in the frozen
# model document (MAX4_ASSOCIATED_ABSORPTION_MODEL_2026-09-03.md); the campaign generator asserts that every line used carries verified=True.
LINES = {
    "SiII1190": dict(lambda0=1190.4158, f=0.2502, gamma=6.53e8, verified=False),
    "SiII1193": dict(lambda0=1193.2897, f=0.4991, gamma=2.69e9, verified=False),
    "SiIII1206": dict(lambda0=1206.5000, f=1.63, gamma=2.48e9, verified=False),
    "SiII1260": dict(lambda0=1260.4221, f=1.18, gamma=2.53e9, verified=False),
}
# Voigt prefactor: tau_0 = (pi e^2 / m_e c) f lambda0 N / (sqrt(pi) b) in cgs with lambda0 in cm, b in cm/s
_PI_E2_MEC = 2.654008854e-2   # pi e^2 / (m_e c)  [cm^2 s^-1]


def voigt_tau(wave_A, lambda0_A, f, gamma, logN, b_kms, dv_kms):
    """Optical depth of one Voigt line on the observed grid wave_A [Å] for a line centred at lambda0 (already redshifted) shifted by dv."""
    wave = np.asarray(wave_A, float)
    lam_c = lambda0_A * (1.0 + dv_kms / C_KMS)
    b = b_kms * 1e5                                                         # cm/s
    nu = C_KMS * 1e5 / (wave * 1e-8); nu0 = C_KMS * 1e5 / (lam_c * 1e-8)   # Hz
    dnu_D = nu0 * b / (C_KMS * 1e5)                                          # Doppler width in Hz
    a = gamma / (4.0 * math.pi * dnu_D)
    u = (nu - nu0) / dnu_D
    H = np.real(wofz(u + 1j * a))
    tau0 = _PI_E2_MEC * f * (10.0 ** logN) / (math.sqrt(math.pi) * dnu_D)
    return tau0 * H


def metal_transmission(wave_A, z_abs, lines, lsf_fwhm_A=None, lines_table=None, fine_dlam_A=0.02, pad_A=25.0):
    """T_metal on the OBSERVED grid wave_A [Å] (coadd pixels): each line's optical depth is evaluated on a fine sub-grid (fine_dlam_A) around its
    observed wavelength, T = exp(-tau) is convolved with a Gaussian LSF of FWHM lsf_fwhm_A (if given) on the fine grid, and the result is
    box-averaged over each coadd pixel (the spectrograph delivers pixel-integrated flux). Pixels outside every line window keep T = 1.
    lines: list of {"line", "logN", "b_kms", "dv_kms"}."""
    tab = LINES if lines_table is None else lines_table
    wave = np.asarray(wave_A, float); T_out = np.ones(wave.size)
    if not lines:
        return T_out
    dpix = float(np.median(np.diff(wave))); edges = np.concatenate([[wave[0] - dpix / 2], 0.5 * (wave[1:] + wave[:-1]), [wave[-1] + dpix / 2]])
    centres = [tab[ln["line"]]["lambda0"] * (1.0 + z_abs) * (1.0 + float(ln.get("dv_kms", 0.0)) / C_KMS) for ln in lines]
    lo = min(centres) - pad_A; hi = max(centres) + pad_A
    fine = np.arange(lo, hi, fine_dlam_A); tau = np.zeros(fine.size)
    for ln in lines:
        p = tab[ln["line"]]
        tau += voigt_tau(fine, p["lambda0"] * (1.0 + z_abs), p["f"], p["gamma"], float(ln["logN"]), float(ln["b_kms"]), float(ln.get("dv_kms", 0.0)))
    Tf = np.exp(-tau)
    if lsf_fwhm_A is not None and lsf_fwhm_A > 0:
        from scipy.ndimage import gaussian_filter1d
        Tf = 1.0 - gaussian_filter1d(1.0 - Tf, (lsf_fwhm_A / 2.3548200450309493) / fine_dlam_A, mode="nearest")
    # box-average onto the coadd pixels that overlap the fine window
    j0 = max(0, int(np.searchsorted(edges, lo)) - 1); j1 = min(wave.size, int(np.searchsorted(edges, hi)) + 1)
    cum = np.concatenate([[0.0], np.cumsum(Tf) * fine_dlam_A])
    for j in range(j0, j1):
        a_, b_ = max(edges[j], lo), min(edges[j + 1], hi)
        if b_ <= a_:
            continue
        ia = int((a_ - lo) / fine_dlam_A); ib = int((b_ - lo) / fine_dlam_A)
        inside = (cum[min(ib, fine.size)] - cum[ia]) / max((min(ib, fine.size) - ia) * fine_dlam_A, 1e-12) if ib > ia else float(Tf[min(ia, fine.size - 1)])
        frac = (b_ - a_) / (edges[j + 1] - edges[j])
        T_out[j] = (1.0 - frac) * 1.0 + frac * inside
    return T_out


def equivalent_width_A(wave_A, T):
    """Rest-frame-agnostic observed-frame equivalent width [Å] of 1 - T over the grid."""
    wave = np.asarray(wave_A, float); return float(np.trapezoid(1.0 - np.asarray(T, float), wave))


def parse_metals(s):
    """metals_json column -> list of line dicts (empty for '' / None)."""
    if s is None or (isinstance(s, str) and s.strip() in ("", "[]", "nan")):
        return []
    return json.loads(s) if isinstance(s, str) else list(s)
