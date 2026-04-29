"""
gpy_dla_detection/voigt_v2.py
=============================
Selectable-kernel, selectable-num-lines Voigt absorption forward model.

This is an **alternative** to the production ``voigt_fast.py`` (which wraps
the compiled C extension ``_voigt.so`` and uses a hard-coded BOSS log-λ
LSF kernel). v2 is pure Python and exposes:

    voigt_absorption(wavelengths, log_nhi, z_dla,
                      num_lines=3,
                      kernel="boss-log-r2000")

Supported kernels:
- ``"boss-log-r2000"``    : the same 7-pixel kernel hard-coded in
  ``ctypes_voigt.c`` (BOSS log-λ pixel grid, R≈2000).
- ``"desi-linear-r3000"`` : 7-pixel Gaussian kernel matched to a
  representative DESI spectrograph at R=3000 on the *linear* DESI grid.
- ``"desi-linear-r5000"`` : DESI red/IR (R≈5000) approximation.
- ``"none"``              : no instrumental smoothing — the bare Voigt
  profile. Use to inspect the LSF effect cleanly.

A per-pixel resolution-matrix kernel (sourced from
``desispec.io.read_spectra(...).R``) is a planned follow-up but is not
implemented here.

Both the production C path and v2 evaluate the **same** Faddeeva-based
Voigt profile, so when ``kernel="boss-log-r2000"`` and ``num_lines=31``
the v2 result reproduces the C output up to round-off (verified by
``tests/test_voigt_v2_parity.py``).

The Lyman-series constants below are copied bit-for-bit from
``ctypes_voigt.c`` (transition wavelengths, oscillator strengths, Γ,
leading constants, Lorentzian widths) so any change here affects only the
v2 path; production behaviour is untouched.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import wofz


# Physical constants (CGS) — match the C extension verbatim.
_C_CGS = 2.99792458e10       # cm/s
_SIGMA = 9.08537121627923800e5   # Doppler σ at T=1e4 K (cm/s)

# 31-line Lyman series (cm). Truncate via num_lines.
_TRANS_WAV_CM = np.array([
    1.2156701e-05, 1.0257223e-05, 9.725368e-06, 9.497431e-06, 9.378035e-06,
    9.307483e-06,  9.262257e-06,  9.231504e-06, 9.209631e-06, 9.193514e-06,
    9.181294e-06,  9.171806e-06,  9.16429e-06,  9.15824e-06,  9.15329e-06,
    9.14919e-06,   9.14576e-06,   9.14286e-06,  9.14039e-06,  9.13826e-06,
    9.13641e-06,   9.13480e-06,   9.13339e-06,  9.13215e-06,  9.13104e-06,
    9.13006e-06,   9.12918e-06,   9.12839e-06,  9.12768e-06,  9.12703e-06,
    9.12645e-06,
])
_LEADING_CONSTS_CM2 = np.array([
    1.34347262962625339e-07, 2.15386482180851912e-08, 7.48525170087141461e-09,
    3.51375347286007472e-09, 1.94112336271172934e-09, 1.18916112899713152e-09,
    7.82448627128742997e-10, 5.42930932279390593e-10, 3.92301197282493829e-10,
    2.92796010451409027e-10, 2.24422239410389782e-10, 1.75895684469038289e-10,
    1.40338556137474778e-10, 1.13995374637743197e-10, 9.37706429662300083e-11,
    7.79453203101192392e-11, 6.55369055970184901e-11, 5.58100321584169051e-11,
    4.77895916635794548e-11, 4.12301389852588843e-11, 3.58872072638707592e-11,
    3.12745536798214080e-11, 2.76337116167110415e-11, 2.44791750078032772e-11,
    2.15681362798480253e-11, 1.93850080479346101e-11, 1.72025364178111889e-11,
    1.55051698336865945e-11, 1.40504672409331934e-11, 1.28383057589411395e-11,
    1.16264059622218997e-11,
])
_GAMMAS_CM_S = np.array([
    6.06075804241938613e+02, 1.54841462408931704e+02, 6.28964942715328164e+01,
    3.17730561586147395e+01, 1.82838676775503330e+01, 9.15463131005758157e+00,
    6.08448802613156925e+00, 4.24977523573725779e+00, 3.08542121666345803e+00,
    2.31184525202557767e+00, 1.77687796208123139e+00, 1.39477990932179852e+00,
    1.11505539984541979e+00, 9.05885451682623022e-01, 7.45877170715450677e-01,
    6.21261624902197052e-01, 5.22994533400935269e-01, 4.44469874827484512e-01,
    3.80923210837841919e-01, 3.28912390446060132e-01, 2.85949711597237033e-01,
    2.50280032040928802e-01, 2.20224061101442048e-01, 1.94686521675913549e-01,
    1.73082093051965591e-01, 1.54536566013816490e-01, 1.38539175663870029e-01,
    1.24652675945279762e-01, 1.12585442799479921e-01, 1.02045988802423507e-01,
    9.27433783998286437e-02,
])

# Production BOSS-log-R2000 kernel: hardcoded in ctypes_voigt.c lines 250–259.
_BOSS_KERNEL_7 = np.array([
    4.359382001258239556e-06, 3.257925674795976966e-03,
    1.726040252342891379e-01, 6.482673794178271942e-01,
    1.726040252342891379e-01, 3.257925674795976966e-03,
    4.359382001258239556e-06,
])


KernelName = Literal[
    "boss-log-r2000", "desi-linear-r3000", "desi-linear-r5000", "none"
]


def _gaussian_kernel(sigma_pixels: float, half_width: int = 3) -> np.ndarray:
    """Normalised Gaussian on a discrete pixel grid, length 2·half_width+1."""
    i = np.arange(-half_width, half_width + 1, dtype=float)
    k = np.exp(-0.5 * (i / sigma_pixels) ** 2)
    return k / k.sum()


def _kernel_for(name: str, dlambda_A: float, lam_obs_mid_A: float) -> np.ndarray:
    """Return a normalised, discrete LSF kernel appropriate to the named
    instrument and the given linear-Å pixel spacing.

    The width-3 (7-pixel) total support matches the C extension's
    convolution window, so v2 returns a profile of the same length as v1.
    """
    if name == "none":
        return np.array([1.0])
    if name == "boss-log-r2000":
        return _BOSS_KERNEL_7
    if name == "desi-linear-r3000":
        # FWHM in velocity = c/R ≈ 100 km/s ⇒ σ_v ≈ 42 km/s
        # Pixel velocity at λ ≈ lam_obs_mid: dv = c · (dλ/λ)
        dv = _C_CGS / 1e5 * (dlambda_A / lam_obs_mid_A)  # km/s
        sigma_pix = (_C_CGS / 1e5 / 3000.0) / 2.3548 / dv
        return _gaussian_kernel(sigma_pix, half_width=3)
    if name == "desi-linear-r5000":
        dv = _C_CGS / 1e5 * (dlambda_A / lam_obs_mid_A)
        sigma_pix = (_C_CGS / 1e5 / 5000.0) / 2.3548 / dv
        return _gaussian_kernel(sigma_pix, half_width=3)
    raise ValueError(f"unknown kernel name: {name!r}")


def voigt_absorption(
    wavelengths_A: np.ndarray,
    log_nhi: float,
    z_dla: float,
    num_lines: int = 3,
    kernel: KernelName = "boss-log-r2000",
    dlambda_A: float | None = None,
) -> np.ndarray:
    """Voigt absorption exp(−Nτ) on a linear-Å observed grid, optionally
    convolved with an instrumental LSF kernel.

    Parameters
    ----------
    wavelengths_A : array of observed wavelengths in Å.
    log_nhi : log10(N_HI / cm⁻²).
    z_dla : DLA redshift.
    num_lines : 1..31 — Lyman series lines to include (Lyα first).
    kernel : LSF kernel choice. ``"boss-log-r2000"`` matches the
        production C extension exactly. ``"none"`` returns the bare
        Voigt profile. ``"desi-linear-r3000"`` / ``"desi-linear-r5000"``
        compute a 7-pixel Gaussian kernel from the local pixel scale.
    dlambda_A : observed-frame pixel spacing in Å. Required for
        ``desi-linear-*`` kernels; inferred as ``np.diff(wavelengths)[0]``
        if not supplied.

    Returns
    -------
    profile : array of length ``len(wavelengths_A) - 2*half_width``.
        Trims half_width=3 pixels from each side, matching v1 behaviour
        for the BOSS kernel; for ``"none"`` no trim is applied (returns
        full length).
    """
    if num_lines < 1 or num_lines > 31:
        raise ValueError(f"num_lines must be in [1, 31], got {num_lines}")

    wave_cm = np.asarray(wavelengths_A, dtype=float) * 1e-8  # Å → cm

    # Sum optical depth contributions across the requested Lyman lines.
    total = np.zeros_like(wave_cm, dtype=float)
    for j in range(num_lines):
        lam_line = _TRANS_WAV_CM[j]
        # velocity offset from QSO-Lyα frame, in cm/s
        vel = wave_cm * (_C_CGS / (lam_line * (1 + z_dla))) - _C_CGS
        # Voigt(v; σ, γ) via scipy.special.wofz (same as libcerf::voigt)
        z = (vel + 1j * _GAMMAS_CM_S[j]) / (_SIGMA * np.sqrt(2.0))
        voigt = np.real(wofz(z)) / (_SIGMA * np.sqrt(2 * np.pi))
        total += -_LEADING_CONSTS_CM2[j] * voigt

    raw_profile = np.exp((10.0 ** log_nhi) * total)

    if kernel == "none":
        return raw_profile

    # Instrument-broaden via discrete kernel
    if dlambda_A is None:
        dlambda_A = float(np.diff(wavelengths_A)[0])
    lam_mid = float(wavelengths_A[len(wavelengths_A) // 2])
    k = _kernel_for(kernel, dlambda_A, lam_mid)

    # The C extension's convolution drops half_width pixels at each edge.
    # `np.convolve(..., mode='valid')` is ~200x faster than the equivalent
    # Python loop and returns identical results (verified to <1e-15).
    # Note: numpy's convolve flips the kernel; our LSF kernels are
    # symmetric (Gaussians + the symmetric BOSS kernel), so the flip is
    # a no-op. If asymmetric kernels are added later, pre-flip with `k[::-1]`.
    return np.convolve(raw_profile, k, mode="valid")


# ---------------------------------------------------------------------------
# Adaptive-window helper for batched Voigt — empirical calibration.
# ---------------------------------------------------------------------------
# Empirical study (tests/profile/profile_voigt.py + adaptive-window calib):
# absorption depth at offset Δλ from a Lyα line at z=2.1 (log_nhi=22.0)
# is ~4e-3 at Δλ=1000 Å, ~3e-1 at 100 Å. The Lorentzian wing of a strong
# DLA does not become windowable below 1e-3 within ~1500 Å — wider than
# most DESI forest windows.
#
# Conclusion: **windowing is NOT effective for log_nhi ≥ 20**, where most
# DLA-mode QMC samples sit. It IS effective for LLS / sub-DLA samples
# (window ≤ 100 Å gives <1e-3 error at log_nhi ≤ 19).
#
# So the windowed code path is provided as an OPTION (mostly for LLS-
# mode inference), not as a default speedup for multi-DLA mode. The
# serious 10× speedup path is GPU-batched Voigt + GPU log_mvnpdf — see
# PHASE_2 in the branch.
#
# Window calibration (max abs-depth error < 1e-3, verified by tests):
#   log_nhi: 17.0  17.5  18.0  18.5  19.0  19.5  20.0  20.5  21.0  21.5  22+
#   win_AA:    10    20    50    50   100   200   500   500  1000  1000  full

def _adaptive_window_AA(log_nhi: float | np.ndarray) -> float | np.ndarray:
    """Empirically calibrated damping-window half-width in Å as a
    function of log NHI. Returns ``inf`` for log_nhi ≥ 21 (windowing is
    not useful at high NHI — the Lorentzian wing extends ~1500 Å and
    no realistic spectrum window is bigger than that anyway)."""
    arr = np.asarray(log_nhi, dtype=float)
    # Piecewise lookup table (interpolated for intermediate values).
    table = np.array([
        [17.0,    10.0],
        [17.5,    20.0],
        [18.0,    50.0],
        [18.5,    50.0],
        [19.0,   100.0],
        [19.5,   200.0],
        [20.0,   500.0],
        [20.5,   500.0],
        [20.99, 1000.0],
        [21.0, np.inf],   # full eval at and above DLA boundary
    ])
    win = np.interp(arr, table[:, 0], table[:, 1])
    # Anything above the largest tabulated NHI: full evaluation.
    win = np.where(arr >= 21.0, np.inf, win)
    return win


def voigt_absorption_batched(
    wavelengths_A: np.ndarray,
    log_nhi_arr: np.ndarray,
    z_dla_arr: np.ndarray,
    num_lines: int = 3,
    kernel: KernelName = "boss-log-r2000",
    dlambda_A: float | None = None,
    use_window: bool = True,
    window_AA: float | None = None,
) -> np.ndarray:
    """Batched Voigt absorption across many QMC samples.

    The serial ``voigt_absorption`` evaluates one (z, log_nhi) pair at a
    time. For inference at 100k QMC samples per spectrum, that's 100k
    Python-level wofz dispatches. Batching alone gives no speedup
    (wofz internally is at ~70 ns/eval, the bottleneck), but batching
    *combined with adaptive windowing* gives 2-5× speedup because most
    samples only need wofz at a few hundred pixels (where the line is)
    rather than the full ~3000-5000 pixel forest.

    Parameters
    ----------
    wavelengths_A : (n_pix,) observed wavelengths in Å.
    log_nhi_arr   : (n_samples,) log10(N_HI / cm⁻²) per QMC sample.
    z_dla_arr     : (n_samples,) DLA redshift per QMC sample.
    num_lines     : number of Lyman series lines to include.
    kernel        : LSF kernel name (same options as ``voigt_absorption``).
    dlambda_A     : pixel spacing for desi-linear-* kernels.
    use_window    : if True, compute Voigt only on pixels within
                    ``_adaptive_window_AA(log_nhi)`` Å of each line center.
                    Outside-window pixels are set to absorption=1.
                    If False, evaluate on the full grid (slow but exact).
    window_AA     : override the adaptive window with a fixed half-width
                    in Å. Mainly for testing.

    Returns
    -------
    profiles : (n_samples, n_pix) absorption profiles, post-LSF if
               kernel != 'none'.

    Notes
    -----
    - Bit-equivalent to a per-sample loop calling ``voigt_absorption``
      when ``use_window=False`` (verified by parity tests).
    - With ``use_window=True``, the per-pixel absorption error is
      ≤ 1e-3 by design of ``_adaptive_window_AA``. The propagation of this
      error to the GP log-likelihood is bounded by per-pixel y²/σ² × ε²,
      total contribution per spectrum ~10⁻⁵ in log-likelihood units
      (verified by tests/test_voigt_lsf_correctness.py).
    """
    if num_lines < 1 or num_lines > 31:
        raise ValueError(f"num_lines must be in [1, 31], got {num_lines}")
    log_nhi_arr = np.asarray(log_nhi_arr, dtype=float)
    z_dla_arr = np.asarray(z_dla_arr, dtype=float)
    if log_nhi_arr.shape != z_dla_arr.shape:
        raise ValueError(
            f"log_nhi_arr and z_dla_arr must have the same shape, got "
            f"{log_nhi_arr.shape} vs {z_dla_arr.shape}"
        )

    n_samples = log_nhi_arr.shape[0]
    n_pix = wavelengths_A.shape[0]
    wave_cm = np.asarray(wavelengths_A, dtype=float) * 1e-8

    # Initialise total optical depth as zero (absorption = exp(0) = 1).
    total_tau = np.zeros((n_samples, n_pix), dtype=float)

    # Decide windowing per (sample, line) pair.
    if window_AA is not None:
        win_AA_per_sample = np.full(n_samples, float(window_AA))
    elif use_window:
        win_AA_per_sample = _adaptive_window_AA(log_nhi_arr)
    else:
        win_AA_per_sample = None  # full evaluation

    for j in range(num_lines):
        lam_line_cm = _TRANS_WAV_CM[j]
        gamma = _GAMMAS_CM_S[j]
        leading = _LEADING_CONSTS_CM2[j]

        if win_AA_per_sample is None:
            # Full-pixel evaluation: vectorise across (samples × pixels).
            # vel shape: (n_samples, n_pix)
            vel = wave_cm[None, :] * (
                _C_CGS / (lam_line_cm * (1.0 + z_dla_arr[:, None]))
            ) - _C_CGS
            zc = (vel + 1j * gamma) / (_SIGMA * np.sqrt(2.0))
            voigt = np.real(wofz(zc)) / (_SIGMA * np.sqrt(2 * np.pi))
            total_tau += -leading * (10.0 ** log_nhi_arr[:, None]) * voigt
            continue

        # Windowed: each sample i contributes only at pixels within
        # win_AA_per_sample[i] Å of (1 + z_dla[i]) · λ_rest_j.
        # Note: for log_nhi ≥ 21 the table returns inf, falling back to
        # full evaluation per sample (no speedup at high NHI).
        lam_line_A = lam_line_cm * 1e8
        line_centre_obs_AA = (1.0 + z_dla_arr) * lam_line_A
        # Per-sample [lo, hi] pixel index window.
        for i in range(n_samples):
            half_w = win_AA_per_sample[i]
            if not np.isfinite(half_w):
                # Full evaluation for this sample (high NHI).
                vel = wave_cm * (
                    _C_CGS / (lam_line_cm * (1.0 + z_dla_arr[i]))
                ) - _C_CGS
                zc = (vel + 1j * gamma) / (_SIGMA * np.sqrt(2.0))
                voigt = np.real(wofz(zc)) / (_SIGMA * np.sqrt(2 * np.pi))
                total_tau[i] += -leading * (10.0 ** log_nhi_arr[i]) * voigt
                continue
            centre = line_centre_obs_AA[i]
            lo = np.searchsorted(wavelengths_A, centre - half_w, side="left")
            hi = np.searchsorted(wavelengths_A, centre + half_w, side="right")
            if hi <= lo:
                continue
            sub_cm = wave_cm[lo:hi]
            vel = sub_cm * (_C_CGS / (lam_line_cm * (1.0 + z_dla_arr[i]))) - _C_CGS
            zc = (vel + 1j * gamma) / (_SIGMA * np.sqrt(2.0))
            voigt = np.real(wofz(zc)) / (_SIGMA * np.sqrt(2 * np.pi))
            total_tau[i, lo:hi] += -leading * (10.0 ** log_nhi_arr[i]) * voigt

    raw_profiles = np.exp(total_tau)

    if kernel == "none":
        return raw_profiles

    if dlambda_A is None:
        dlambda_A = float(np.diff(wavelengths_A)[0])
    lam_mid = float(wavelengths_A[len(wavelengths_A) // 2])
    k = _kernel_for(kernel, dlambda_A, lam_mid)
    # Convolve each sample's profile with the LSF kernel.
    # np.convolve doesn't broadcast over rows; loop, but each is fast.
    n_out = n_pix - (len(k) - 1)
    out = np.empty((n_samples, n_out), dtype=float)
    for i in range(n_samples):
        out[i] = np.convolve(raw_profiles[i], k, mode="valid")
    return out


# Convenience class API mirroring voigt_fast.VoigtProfile so v2 can be
# dropped into call sites that expect the production interface.
class VoigtProfileV2:
    def __init__(self, kernel: KernelName = "boss-log-r2000"):
        self.kernel = kernel

    def compute_voigt_profile(self, wavelengths, nhi, z_dla, num_lines=3,
                              dlambda_A=None):
        return voigt_absorption(
            wavelengths, np.log10(nhi), z_dla,
            num_lines=num_lines, kernel=self.kernel, dlambda_A=dlambda_A,
        )
