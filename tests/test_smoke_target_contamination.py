"""
tests/test_smoke_target_contamination.py
=========================================
Verify that the smoke-test target (2LPT mock-0 contaminated, TARGETID
120046865) actually contains the truth-catalog DLA — i.e. the loa-124
spectrum has flux suppressed in the expected DLA wing region while the
matching loa-0 (uncontaminated) spectrum does not.

This test is a guard against the failure mode of running the smoke against
the wrong file (uncontaminated) and falsely concluding nothing is wrong.

Skipped automatically if the per-spectrum FITS files are not on this machine
(e.g. NERSC vs GreatLakes vs developer laptop).
"""

from __future__ import annotations

import os

import numpy as np
import pytest


PATH_LOA124 = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-124/spectra-16/7/789/spectra-16-789.fits"
)
PATH_LOA0 = (
    "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
    "mock-0/loa-0/spectra-16/7/789/spectra-16-789.fits"
)
TARGET_ID = 120046865
TRUTH_Z = 2.773
TRUTH_LOG_NHI = 21.26
LYA_REST = 1215.67


pytestmark = pytest.mark.skipif(
    not (os.path.exists(PATH_LOA124) and os.path.exists(PATH_LOA0)),
    reason="2LPT mock spectra not present on this machine",
)


def _load_flux(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (wave_brz, flux_brz) for TARGET_ID from the given spec file."""
    import fitsio
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    sp = read_spectra(path, targetids=[TARGET_ID])
    try:
        sp = coadd_cameras(sp)
        band = "brz"
    except Exception:
        if sp.resolution_data is None:
            truth = path.replace("spectra-16-", "truth-16-")
            sp.resolution_data = {}
            for cam in "brz":
                tres = fitsio.read(truth, ext=f"{cam}_RESOLUTION")
                tresdata = np.empty(
                    [sp.flux[cam].shape[0], tres.shape[0], sp.flux[cam].shape[1]],
                    dtype=float,
                )
                for i in range(sp.flux[cam].shape[0]):
                    tresdata[i] = tres
                sp.resolution_data[cam] = tresdata
        sp = resample_spectra_lin_or_log(
            sp, linear_step=0.8,
            wave_min=float(np.min(sp.wave["b"])),
            wave_max=float(np.max(sp.wave["z"])),
            fast=True,
        )
        band = "brz" if "brz" in sp.wave else list(sp.wave.keys())[0]

    i = int(np.where(np.asarray(sp.fibermap["TARGETID"]) == TARGET_ID)[0][0])
    return sp.wave[band].astype(np.float64), sp.flux[band][i].astype(np.float64)


def test_targetid_present_in_both_files():
    """The smoke target must exist in both contaminated and uncontaminated mocks."""
    wave_c, flux_c = _load_flux(PATH_LOA124)
    wave_u, flux_u = _load_flux(PATH_LOA0)
    assert wave_c.size > 0
    assert wave_u.size > 0


def test_dla_absorption_visible_only_in_loa124():
    """The contaminated mock flux must drop in the DLA wing region;
    the uncontaminated mock must keep the normal forest flux."""
    wave_c, flux_c = _load_flux(PATH_LOA124)
    wave_u, flux_u = _load_flux(PATH_LOA0)

    # Truth DLA Lyα observed wavelength
    lya_obs = LYA_REST * (1 + TRUTH_Z)

    # DLA wing region: ±20 Å around the line core.
    # For log NHI = 21.26, the damping wings are ~10-15 Å wide.
    wing_c = (wave_c > lya_obs - 20) & (wave_c < lya_obs + 20)
    wing_u = (wave_u > lya_obs - 20) & (wave_u < lya_obs + 20)

    mean_c = np.nanmean(flux_c[wing_c])
    mean_u = np.nanmean(flux_u[wing_u])

    # Empirical numbers measured 2026-04-27 on this machine:
    #   loa-124 (contaminated):  mean ≈ -0.04
    #   loa-0   (uncontaminated): mean ≈  +0.31
    # The DLA suppresses flux to roughly 0; the forest baseline is positive.
    assert mean_c < 0.10, (
        f"Contaminated mock flux at DLA wing should be near zero, got {mean_c:.3f}. "
        f"This means the contamination is missing — check that the spec file path "
        f"is loa-124, not loa-0."
    )
    assert mean_u > 0.20, (
        f"Uncontaminated mock flux at the same wavelength should be ~forest "
        f"baseline (~0.3), got {mean_u:.3f}."
    )
    # And the contaminated case must have measurably less flux than the uncontaminated.
    assert mean_u - mean_c > 0.20, (
        f"Contaminated minus uncontaminated mean flux at DLA position should be "
        f"clearly negative (DLA absorbs flux), got Δ={mean_u - mean_c:.3f}."
    )
