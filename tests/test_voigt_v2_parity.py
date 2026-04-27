"""tests/test_voigt_v2_parity.py
=================================
Verify the alternative pure-Python ``voigt_v2`` reproduces the production
``voigt_fast`` C extension when run in the same configuration:

    kernel = "boss-log-r2000"
    num_lines = 31

This is a regression test: any future tweak to voigt_v2 that breaks parity
will fail here, alerting us before that change propagates to mock studies.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def voigt_fast():
    try:
        from gpy_dla_detection.voigt_fast import VoigtProfile  # noqa: PLC0415
    except (OSError, ImportError) as e:
        pytest.skip(f"voigt_fast C extension unavailable: {e}")
    return VoigtProfile()


def _grid(z_qso=2.7, n=2000, dlambda=0.25):
    lam_obs_min = 911.75 * (1 + z_qso) - 50
    lam_obs_max = 1216.75 * (1 + z_qso) + 50
    return np.arange(lam_obs_min, lam_obs_max, dlambda, dtype=np.float64)


@pytest.mark.parametrize("log_nhi,z_dla", [
    (20.5, 2.45),
    (21.0, 2.55),
    (21.5, 2.65),
    (22.0, 2.50),
])
def test_v2_matches_c_extension_under_default_kernel(voigt_fast, log_nhi, z_dla):
    """v2 with kernel=boss-log-r2000 + num_lines=31 reproduces v1 to ~1e-10."""
    from gpy_dla_detection.voigt_v2 import voigt_absorption

    wavelengths = _grid()
    p_v1 = voigt_fast.compute_voigt_profile(
        wavelengths, nhi=10**log_nhi, z_dla=z_dla, num_lines=31
    )
    p_v2 = voigt_absorption(
        wavelengths, log_nhi, z_dla,
        num_lines=31, kernel="boss-log-r2000",
    )
    assert p_v1.shape == p_v2.shape
    diff = np.max(np.abs(p_v1 - p_v2))
    assert diff < 1e-9, f"v1 vs v2 max abs diff = {diff:.2e} (expected < 1e-9)"


def test_no_kernel_is_deeper_than_smoothed_for_strong_dla():
    """Sanity: at z=2.5, NHI=21.5, the unsmoothed Voigt core is deeper
    than the BOSS-kernel-smoothed one."""
    from gpy_dla_detection.voigt_v2 import voigt_absorption
    wave = _grid()
    raw = voigt_absorption(wave, 21.5, 2.50, num_lines=3, kernel="none")
    smoothed = voigt_absorption(wave, 21.5, 2.50, num_lines=3,
                                kernel="boss-log-r2000")
    assert raw.min() <= smoothed.min() + 1e-3, (
        f"unsmoothed core (min={raw.min():.4f}) should be at least as "
        f"deep as smoothed core (min={smoothed.min():.4f})"
    )


def test_desi_kernel_smooths_more_than_boss_kernel_on_dlambda_015_grid():
    """When applied on the DESI linear pixel grid (dlambda=0.15 Å), the
    DESI-R3000 kernel has a wider σ (≈ 4.2 px) than the BOSS-R2000 kernel
    (which is ≈ 0.92 px wide and was actually calibrated for a log-λ
    grid with ~23 km/s/pixel). The narrower BOSS kernel under-smooths,
    so its profile has a sharper transition between the saturated core
    and the un-absorbed continuum than the DESI-broadened profile.

    This test asserts the DIRECTION of that physical effect: in the
    Voigt damping wings (where the line is partially absorbing, not
    saturated to zero), the BOSS-kernel profile is DEEPER (lower flux)
    than the DESI-kernel profile."""
    from gpy_dla_detection.voigt_v2 import voigt_absorption
    z_dla = 2.45
    wave = _grid(z_qso=2.6, dlambda=0.15)
    # Use a moderate NHI=20.7 so the wings are partially-absorbing
    # rather than saturated to zero on both kernels.
    boss = voigt_absorption(wave, 20.7, z_dla, num_lines=3,
                            kernel="boss-log-r2000", dlambda_A=0.15)
    desi = voigt_absorption(wave, 20.7, z_dla, num_lines=3,
                            kernel="desi-linear-r3000", dlambda_A=0.15)

    # Sanity: same shape and not coincidentally identical.
    assert boss.shape == desi.shape
    assert np.max(np.abs(boss - desi)) > 1e-3, (
        "BOSS and DESI kernels produced indistinguishable profiles — "
        "kernel-dependent broadening is not active"
    )

    # Direction-of-effect check: in the line wings, BOSS profile is deeper.
    # Wing region: pixels where the BOSS profile is between 0.05 and 0.5
    # (i.e. partially absorbing, not saturated to zero, not on the
    # un-absorbed continuum).
    wing_mask = (boss > 0.05) & (boss < 0.5)
    assert wing_mask.sum() > 5, (
        f"too few wing pixels for the test ({wing_mask.sum()}); "
        "increase the wave grid or revisit NHI"
    )
    boss_wing_mean = float(np.mean(boss[wing_mask]))
    desi_wing_mean = float(np.mean(desi[wing_mask]))
    assert boss_wing_mean < desi_wing_mean, (
        f"expected narrower BOSS kernel to give DEEPER wings than the "
        f"broader DESI kernel; got BOSS wings={boss_wing_mean:.4f}, "
        f"DESI wings={desi_wing_mean:.4f}"
    )
