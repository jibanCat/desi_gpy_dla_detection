"""
gpy_dla_detection/inject_absorber.py
====================================
Self-consistent Voigt-absorber INJECTION into observed-frame flux — the
foundational module for the M3 DLA-recovery injection campaign.

Why this exists
---------------
The GP's DLA forward model (``dla_gp.DLAGP.this_dla_gp``) multiplies the spectral
mean by ``VoigtProfile().compute_voigt_profile(wavelengths, nhi, z_dla, num_lines)``
(the compiled C extension ``_voigt.so``; transmission = exp(-tau)). To inject an
absorber that the GP can recover *faithfully*, we multiply the OBSERVED FLUX by
the **same** profile. The GP (``dla_gp.py`` / inference) is never modified —
injection is purely input-flux preprocessing.

Frame & edge convention (matched EXACTLY to ``dla_gp.this_dla_gp``)
-------------------------------------------------------------------
``wavelengths`` is the OBSERVED-frame grid (Å), equally log-spaced (the GP runs
on observed wavelengths; ``z_dla`` is the absorber's observed redshift).

The C extension trims the profile by ``2 * width`` pixels, with ``width = 3``
hardcoded in ``ctypes_voigt.c`` (see ``num_points - 2 * 3`` there and the same in
``voigt_fast.VoigtProfile.compute_voigt_profile``). Concretely, for an input of
``n`` wavelengths the output has ``n - 6`` pixels, and output pixel ``i`` is the
instrument-broadening convolution of input pixels ``[i, i+6]`` — i.e. it aligns
with input pixel ``i + 3``.

``dla_gp`` restores full length by PADDING the wavelength grid with
``params.width = 3`` log-spaced pixels on each side (``null_gp.py`` builds
``padded_wavelengths``), calling Voigt on the padded grid, and using the trimmed
result, which then aligns pixel-for-pixel with the *unpadded* grid
(``params.width == 3`` matches the C extension's ``width``). We reproduce that
padding here so the injected absorber sits at the SAME pixels the GP will score —
**no off-by-3-pixel shift** (a 3-pixel shift would bias the recovered N_HI).

N_HI units
----------
``nhi`` is the **linear** column density in cm^-2 (i.e. ``10 ** log10(N_HI)``),
exactly what ``dla_gp`` passes to the C extension (it uses ``nhis``, not
``log_nhis``). Pass ``10**21.5`` for log N_HI = 21.5, NOT ``21.5``.

Grid spacing (log vs linear) — note for the M3 injection campaign
-----------------------------------------------------------------
The "equally log-spaced" wording above describes the eBOSS/BOSS observed grid
``dla_gp`` was originally built on, and the ``1e-4``-dex edge pad reproduces
``null_gp.padded_wavelengths`` on that grid. The Voigt PROFILE itself is computed
at the supplied wavelengths (the C kernel reads the wavelength array; it does not
assume a fixed pitch), and the edge pad only affects the ``_EDGE_WIDTH`` (=3)
pixels at each END — far from any line core. So ``voigt_transmission`` is equally
valid on a **linear** grid: in the DESI mock path the GP scores on a RESAMPLED
0.8 Å LINEAR ``brz`` grid (coadd → ``resample_spectra_lin_or_log``), and calling
this profile on that grid reproduces ``dla_gp``'s imprint to <1 % in equivalent
width (round-trip validated in ``tests/test_coadd_injection.py::test_m4_round
trip_*``). The 3-pixel log-pad is numerically inert for the trough at the 0.8 Å
linear scale; it is retained only for byte-identical parity with the historical
log-grid edge handling.

Public API
----------
``inject_voigt(wavelengths, flux, nhi, z_dla, num_lines=3) -> injected_flux``
``inject_multiple(wavelengths, flux, absorbers) -> injected_flux``  (close pairs)
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

# Fixed instrument-broadening half-width (pixels), hardcoded in ctypes_voigt.c
# (``width = 3``) and used as ``params.width`` by null_gp's padded_wavelengths.
# The C extension trims ``2 * _EDGE_WIDTH`` pixels (3 from each end).
_EDGE_WIDTH = 3

# Pixel spacing in dex used to construct the log-spaced edge padding. Matches
# ``set_parameters.Parameters.pixel_spacing`` (1e-4 dex = DESI/BOSS DLAMBDA grid),
# which is what null_gp.padded_wavelengths uses to build the pad pixels.
_PIXEL_SPACING_DEX = 1e-4

# Lazily-instantiated singleton VoigtProfile so the module imports even when the
# compiled C extension (_voigt.so) is unavailable (pure-logic tests still run).
_VOIGT_PROFILE = None


def _get_voigt_profile():
    """Lazy-load and cache the compiled VoigtProfile (so import never fails)."""
    global _VOIGT_PROFILE
    if _VOIGT_PROFILE is None:
        from .voigt_fast import VoigtProfile

        _VOIGT_PROFILE = VoigtProfile()
    return _VOIGT_PROFILE


def voigt_transmission(
    wavelengths: np.ndarray,
    nhi: float,
    z_dla: float,
    num_lines: int = 3,
) -> np.ndarray:
    """Voigt transmission profile aligned pixel-for-pixel with ``wavelengths``.

    Reproduces ``dla_gp.this_dla_gp``'s broadening path: pad the (observed-frame,
    log-spaced) grid with ``_EDGE_WIDTH`` log-spaced pixels on each side, call the
    C-extension Voigt on the padded grid, and return the (trimmed) result, which
    aligns to the original grid with NO edge shift.

    Parameters
    ----------
    wavelengths : (n,) array
        Observed-frame wavelength grid (Å), equally log-spaced.
    nhi : float
        LINEAR column density in cm^-2 (``10 ** log10 N_HI``).
    z_dla : float
        Absorber redshift (observed-frame; line centre at ``(1 + z_dla) * lambda_rest``).
    num_lines : int
        Number of Lyman-series lines (matches the run's ``NUM_FOREST_LINES``).

    Returns
    -------
    transmission : (n,) array
        ``exp(-tau)`` in [0, 1], same length as ``wavelengths``.
    """
    wavelengths = np.ascontiguousarray(wavelengths, dtype=np.float64)
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths must be 1-D")
    if wavelengths.size <= 2 * _EDGE_WIDTH:
        raise ValueError(
            f"wavelengths must have > {2 * _EDGE_WIDTH} pixels to survive the "
            f"{2 * _EDGE_WIDTH}-pixel edge trim; got {wavelengths.size}"
        )

    log_min = np.log10(wavelengths.min())
    log_max = np.log10(wavelengths.max())

    # Edge padding identical in form to null_gp.padded_wavelengths (uses
    # ``unmasked_wavelengths.min()/.max()`` and ``params.width`` log-spaced pixels).
    left = np.logspace(
        log_min - _EDGE_WIDTH * _PIXEL_SPACING_DEX,
        log_min - _PIXEL_SPACING_DEX,
        _EDGE_WIDTH,
    )
    right = np.logspace(
        log_max + _PIXEL_SPACING_DEX,
        log_max + _EDGE_WIDTH * _PIXEL_SPACING_DEX,
        _EDGE_WIDTH,
    )
    padded = np.concatenate([left, wavelengths, right])

    profile = _get_voigt_profile().compute_voigt_profile(
        padded, nhi, z_dla, num_lines
    )

    # After the 2*_EDGE_WIDTH trim the profile aligns pixel-for-pixel with the
    # original (unpadded) grid.
    if profile.shape[0] != wavelengths.shape[0]:  # pragma: no cover - defensive
        raise RuntimeError(
            "Voigt edge convention mismatch: padded grid produced "
            f"{profile.shape[0]} pixels, expected {wavelengths.shape[0]}. "
            "Check that the C extension trims exactly 2*3 pixels."
        )
    return profile


def inject_voigt(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    nhi: float,
    z_dla: float,
    num_lines: int = 3,
) -> np.ndarray:
    """Inject a single Voigt absorber into ``flux`` (multiplicative transmission).

    ``injected_flux = flux * transmission(nhi, z_dla, num_lines)``, where
    ``transmission`` is the SAME profile ``dla_gp`` multiplies into its mean model
    (see module docstring for the frame/edge convention and N_HI units). The
    per-pixel noise is intentionally left untouched: the absorber multiplies the
    signal only, and the GP's noise model is unchanged.

    Parameters
    ----------
    wavelengths : (n,) array
        Observed-frame, log-spaced wavelength grid (Å).
    flux : (n,) array
        Observed flux on the same grid.
    nhi : float
        LINEAR column density in cm^-2 (``10 ** log10 N_HI``).
    z_dla : float
        Absorber redshift.
    num_lines : int, default 3
        Number of Lyman-series lines (match ``NUM_FOREST_LINES``).

    Returns
    -------
    injected_flux : (n,) array
        ``flux`` with the absorber imprinted; same shape as ``flux``.
    """
    flux = np.asarray(flux, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if flux.shape != wavelengths.shape:
        raise ValueError(
            f"flux shape {flux.shape} != wavelengths shape {wavelengths.shape}"
        )

    transmission = voigt_transmission(wavelengths, nhi, z_dla, num_lines)
    return flux * transmission


def inject_multiple(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    absorbers: Iterable[Mapping[str, float]],
) -> np.ndarray:
    """Inject several Voigt absorbers multiplicatively (e.g. Campaign-B close pairs).

    Each transmission profile multiplies the flux in turn, so overlapping troughs
    blend exactly as ``dla_gp.this_dla_gp`` blends multiple DLAs (it multiplies the
    per-DLA absorptions before applying them to the mean).

    Parameters
    ----------
    wavelengths : (n,) array
        Observed-frame, log-spaced wavelength grid (Å).
    flux : (n,) array
        Observed flux on the same grid.
    absorbers : iterable of mappings
        Each item provides ``nhi`` (linear cm^-2) and ``z_dla``; ``num_lines`` is
        optional (default 3).

    Returns
    -------
    injected_flux : (n,) array
        ``flux`` with all absorbers imprinted; same shape as ``flux``.
    """
    out = np.asarray(flux, dtype=np.float64).copy()
    for absorber in absorbers:
        out = inject_voigt(
            wavelengths,
            out,
            nhi=absorber["nhi"],
            z_dla=absorber["z_dla"],
            num_lines=int(absorber.get("num_lines", 3)),
        )
    return out
