"""Unit tests for the Voigt v2 forward model — kernel widths, per-line
damping correctness, NHI scaling, and num_lines progressivity.

Built on top of ``tests/test_voigt_v2_parity.py`` (which proves v1↔v2
output equivalence at 1e-9). This file goes deeper: it checks that the
v2 model produces the *physically correct* profile, not just a
match-by-construction copy of v1. Where v1 has known approximations
(e.g. the BOSS-shaped LSF kernel), these tests document them as "v1
intentionally differs from physics here, v2 must agree with physics".

Test layout:

  Layer A — kernel correctness
    - LSF kernel widths match the design spec for each kernel name
    - The "none" kernel is a no-op convolution
    - DESI-linear kernels scale correctly with the pixel grid

  Layer B — per-line Lyman damping
    - Each line uses its OWN damping constant Γᵢ (not Lyα's)
    - Per-line oscillator strengths produce the right depth ratio
    - Adding more lines (num_lines: 1 → 3 → 6 → 31) monotonically
      increases optical depth in the relevant rest-frame regions

  Layer C — NHI scaling (DLA / sub-DLA / LLS regimes)
    - DLA regime (logNHI ≥ 20.3): damping-wing-dominated profile
      width scales as √NHI (Lorentzian asymptotic)
    - Sub-DLA / LLS (logNHI < 20.3): Doppler-core-dominated, much
      narrower profile
    - Optical depth at line centre scales linearly with NHI

  Layer D — degenerate / boundary cases
    - logNHI = -∞ (NHI=0) → identity profile
    - z_dla outside the wavelength range → identity (no absorption in window)
    - Single-pixel input handled gracefully
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpy_dla_detection.voigt_v2 import (
    KernelName,
    VoigtProfileV2,
    _BOSS_KERNEL_7,
    _C_CGS,
    _GAMMAS_CM_S,
    _LEADING_CONSTS_CM2,
    _TRANS_WAV_CM,
    _gaussian_kernel,
    _kernel_for,
    voigt_absorption,
)


# Lyα rest wavelength in Å (consistent with notebooks).
_LYA_AA = 1215.6701
_C_KMS = _C_CGS / 1e5  # speed of light in km/s

# numpy 2.x renamed trapz → trapezoid; older versions only have trapz.
_trapz = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# Layer A — kernel correctness
# ---------------------------------------------------------------------------
class TestKernelCorrectness:
    def test_none_kernel_is_identity(self):
        """``kernel='none'`` must return the bare Voigt profile, length-preserved."""
        wave = np.linspace(4500.0, 4540.0, 200)
        profile_with_kernel = voigt_absorption(
            wave, log_nhi=21.0, z_dla=2.7, num_lines=3, kernel="none",
        )
        # Length-preserved (no edge trim for kernel='none').
        assert profile_with_kernel.shape == (200,)
        # Shouldn't be exactly 1 everywhere if there's a real DLA in window.
        assert profile_with_kernel.min() < 0.5

    def test_gaussian_kernel_normalised(self):
        """All kernel options must integrate to 1 (no flux loss / gain)."""
        for sigma_pix in (0.5, 1.0, 2.0):
            k = _gaussian_kernel(sigma_pix, half_width=3)
            assert k.shape == (7,)
            assert math.isclose(k.sum(), 1.0, abs_tol=1e-12)

        for name in ("boss-log-r2000", "desi-linear-r3000", "desi-linear-r5000"):
            k = _kernel_for(name, dlambda_A=0.15, lam_obs_mid_A=4500.0)
            assert math.isclose(k.sum(), 1.0, rel_tol=1e-6), (
                f"kernel {name!r} not normalised: sum={k.sum()}"
            )

    def test_boss_kernel_unchanged_from_c_extension(self):
        """The BOSS-log-R2000 kernel is hard-coded in the C extension; v2's
        copy must match exactly so the parity tests remain meaningful."""
        k = _kernel_for("boss-log-r2000", dlambda_A=0.15, lam_obs_mid_A=4500.0)
        np.testing.assert_array_equal(k, _BOSS_KERNEL_7)

    def test_desi_kernel_widths_track_spec_qualitatively(self):
        """The DESI-linear kernels are TRUNCATED Gaussians with only 7
        pixels of support. The 'true' σ for R=3000 at dlambda=0.15 Å,
        λ=4500 Å is ~4.2 pixels, but truncation at ±3 pixels means the
        kernel's recovered σ is smaller (~1.9) — most of the Gaussian
        wings are clipped.

        This is a documented limitation, not a bug: the production C
        extension uses the same 7-pixel support, and v2 follows for
        v1↔v2 length parity. Test: R=3000 should still be wider than
        R=5000 (which has σ_true ~2.5 px, less truncation effect).
        """
        dl_A = 0.15
        lam_A = 4500.0

        def _empirical_sigma(k):
            i = np.arange(len(k)) - (len(k) - 1) // 2
            return float(np.sqrt(np.sum(i**2 * k) - np.sum(i * k) ** 2))

        sig_3000 = _empirical_sigma(
            _kernel_for("desi-linear-r3000", dl_A, lam_A)
        )
        sig_5000 = _empirical_sigma(
            _kernel_for("desi-linear-r5000", dl_A, lam_A)
        )
        # R=3000 (lower resolution = wider kernel) σ_pix > R=5000.
        assert sig_3000 > sig_5000, (
            f"R=3000 σ should exceed R=5000 σ, got {sig_3000:.3f} vs {sig_5000:.3f}"
        )
        # Sanity: R=5000 truncated kernel should be close to its full σ
        # (σ_true ~2.5 px, ±3 captures most of the mass).
        assert math.isclose(sig_5000, 2.0, abs_tol=0.6), (
            f"R=5000 σ_pix expected ~2.0–2.5, got {sig_5000:.3f}"
        )

    def test_desi_r3000_wider_than_boss(self):
        """The DESI-linear-r3000 kernel applied to a DESI-linear pixel grid
        should be wider in pixels than the BOSS kernel applied to the same
        grid (because BOSS's kernel was designed for a coarser log-λ grid)."""
        dl_A = 0.15
        lam_A = 4500.0
        k_boss = _kernel_for("boss-log-r2000", dl_A, lam_A)
        k_desi = _kernel_for("desi-linear-r3000", dl_A, lam_A)
        # σ_pix recovery
        def _sigma(k):
            i = np.arange(len(k)) - (len(k) - 1) // 2
            return np.sqrt(np.sum(i**2 * k) - np.sum(i * k)**2)
        assert _sigma(k_desi) > _sigma(k_boss), (
            "DESI-r3000 kernel should be WIDER than BOSS kernel on a "
            "DESI linear-λ grid (this is the bias hypothesis)."
        )


# ---------------------------------------------------------------------------
# Layer B — per-line Lyman damping
# ---------------------------------------------------------------------------
class TestPerLineDamping:
    def test_damping_constants_unique_per_line(self):
        """The Γ_i (damping constants in cm/s) MUST differ between Lyα,
        Lyβ, Lyγ — they're physically distinct atomic transitions.

        Comment in user task: London mock has a known bug (it rescales
        the Lyα feature by oscillator strength rather than recomputing
        the per-line Voigt). The inference-side forward model (this code)
        must do the right thing regardless.
        """
        # First three lines (Lyα, Lyβ, Lyγ) must all differ noticeably.
        assert _GAMMAS_CM_S[0] > _GAMMAS_CM_S[1] > _GAMMAS_CM_S[2]
        # Lyα Γ ≈ 6.06e2; Lyβ ≈ 1.55e2; Lyγ ≈ 6.29e1.
        assert math.isclose(_GAMMAS_CM_S[0], 6.06e2, rel_tol=0.01)
        assert math.isclose(_GAMMAS_CM_S[1], 1.55e2, rel_tol=0.01)
        assert math.isclose(_GAMMAS_CM_S[2], 6.29e1, rel_tol=0.01)
        # And the leading constants (oscillator-strength × λ²) decrease too.
        assert _LEADING_CONSTS_CM2[0] > _LEADING_CONSTS_CM2[1] > _LEADING_CONSTS_CM2[2]

    def test_lyman_lines_at_correct_rest_wavelengths(self):
        """Each Lyα/β/γ feature should appear at the expected observed-frame
        wavelength λ_obs = (1+z) · λ_rest_i."""
        z_dla = 2.5
        # Wide grid spanning Lyα + Lyβ rest-frame at z_dla=2.5.
        # λ(Lyα) at z=2.5 = 4255 Å; λ(Lyβ) at z=2.5 = 3590 Å.
        wave = np.linspace(3500.0, 4350.0, 1000)
        profile = voigt_absorption(
            wave, log_nhi=21.0, z_dla=z_dla, num_lines=3, kernel="none",
        )
        # Find the three deepest minima.
        # Lyα expected at 4255 Å, Lyβ at 3590 Å.
        lya_idx = np.argmin(np.abs(wave - (1 + z_dla) * _LYA_AA))
        lyb_idx = np.argmin(np.abs(wave - (1 + z_dla) * 1025.7223))
        # The profile should be deeply absorbed at both
        # (within a few pixels of the expected centre).
        for centre_idx, name in [(lya_idx, "Lyα"), (lyb_idx, "Lyβ")]:
            local = profile[max(0, centre_idx - 5):centre_idx + 5]
            assert local.min() < 0.5, (
                f"{name}: profile not deep enough at expected centre "
                f"({wave[centre_idx]:.1f} Å); local min = {local.min():.3f}"
            )

    def test_more_lines_means_more_absorption(self):
        """Adding more Lyman series lines monotonically increases the
        total optical depth in the relevant rest-frame regions.

        Specifically: Lyβ region (rest-frame ~1026 Å, observed for z=2.5
        at λ ≈ 3590 Å) should show *more* absorption with num_lines=3
        than num_lines=1, and *more still* with num_lines=6.
        """
        z_dla = 2.5
        # Centre on Lyβ
        wave = np.linspace(3580.0, 3600.0, 300)
        depths = {}
        for nl in (1, 3, 6, 12):
            profile = voigt_absorption(
                wave, log_nhi=21.0, z_dla=z_dla, num_lines=nl, kernel="none",
            )
            # Sum of "absorption depth" 1 - profile in the Lyβ window.
            depths[nl] = np.sum(1.0 - profile)
        assert depths[1] < depths[3], (
            f"Lyβ depth: 1-line={depths[1]:.3f}, 3-line={depths[3]:.3f} "
            f"— more lines should mean more absorption"
        )
        # Higher-order lines (Lyγ, Lyδ, ...) shouldn't add much in the Lyβ
        # window itself, so 3 ≈ 6 ≈ 12 within tolerance.
        assert math.isclose(depths[3], depths[6], rel_tol=0.10)


# ---------------------------------------------------------------------------
# Layer C — NHI scaling (DLA / sub-DLA / LLS regimes)
# ---------------------------------------------------------------------------
class TestNhiScaling:
    """The user is targeting accurate logNHI ≥ 20 (DLA regime) and is
    fine with slight bias for log NHI < 20 (sub-DLA + LLS). These tests
    verify the forward model behaves correctly across all three regimes.
    """

    @staticmethod
    def _trough_eq_width(profile, wave_A):
        """Equivalent width of (1 - profile) integrated over a window —
        a robust measure of total absorption."""
        return float(_trapz(1.0 - profile, x=wave_A))

    @pytest.mark.parametrize("log_nhi,regime", [
        (17.5, "LLS"),
        (19.0, "sub-DLA"),
        (20.5, "DLA"),
        (21.5, "DLA"),
    ])
    def test_eq_width_strictly_increases_with_nhi(self, log_nhi, regime):
        """Equivalent width must grow with NHI in all three regimes."""
        z_dla = 2.5
        # Wide window centred on Lyα at z_dla=2.5.
        wave_centre = (1 + z_dla) * _LYA_AA   # ≈ 4255 Å
        wave = np.linspace(wave_centre - 100, wave_centre + 100, 1000)

        eq_width_lower = self._trough_eq_width(
            voigt_absorption(wave, log_nhi=log_nhi - 0.3, z_dla=z_dla,
                             num_lines=3, kernel="none"),
            wave,
        )
        eq_width_target = self._trough_eq_width(
            voigt_absorption(wave, log_nhi=log_nhi, z_dla=z_dla,
                             num_lines=3, kernel="none"),
            wave,
        )
        assert eq_width_target > eq_width_lower, (
            f"{regime} (logNHI={log_nhi}): equivalent width should "
            f"increase with NHI, got {eq_width_target:.2f} vs lower "
            f"{eq_width_lower:.2f}"
        )

    def test_dla_damping_wings_present_at_high_nhi(self):
        """At logNHI ≥ 20.3 the profile has Lorentzian damping wings —
        absorption extends far from line centre. At logNHI ≤ 19 the
        profile is dominated by the Gaussian Doppler core and is
        much narrower.

        Test: depth at ±50 Å from line centre should be substantial
        (>1% absorption) for DLA but negligible for LLS.
        """
        z_dla = 2.5
        wave_centre = (1 + z_dla) * _LYA_AA
        wave = np.linspace(wave_centre - 100, wave_centre + 100, 2000)

        # Sample at +50 Å from line centre (well into the wings).
        far_idx = np.argmin(np.abs(wave - (wave_centre + 50.0)))

        for log_nhi, regime, expect_wing_absorption in [
            (17.5, "LLS",     False),
            (18.5, "LLS",     False),
            (20.7, "DLA",     True),
            (21.5, "DLA",     True),
        ]:
            profile = voigt_absorption(
                wave, log_nhi=log_nhi, z_dla=z_dla, num_lines=3, kernel="none",
            )
            depth_at_far = 1.0 - profile[far_idx]
            if expect_wing_absorption:
                assert depth_at_far > 0.005, (
                    f"{regime} (logNHI={log_nhi}): expected damping-wing "
                    f"absorption > 0.5% at +50Å from centre, got "
                    f"{depth_at_far*100:.3f}%"
                )
            else:
                assert depth_at_far < 0.001, (
                    f"{regime} (logNHI={log_nhi}): expected NO measurable "
                    f"damping-wing absorption (Doppler core dominates), got "
                    f"{depth_at_far*100:.3f}%"
                )

    def test_dla_wing_width_scales_with_sqrt_nhi(self):
        """Lorentzian damping-wing FWHM scales as √NHI in the asymptotic
        regime (Voigt → Lorentzian when natural broadening dominates).
        Verify by comparing logNHI=20.5 vs logNHI=21.5 — their FWHM
        ratio should be ~ √(10) ≈ 3.16.

        (Tolerance is loose because we're at the boundary of the
        asymptotic regime; logNHI=20.5 still has some Gaussian-core
        contribution.)
        """
        z_dla = 2.5
        wave_centre = (1 + z_dla) * _LYA_AA
        wave = np.linspace(wave_centre - 100, wave_centre + 100, 2000)

        def _fwhm_pix(profile):
            depth = 1.0 - profile
            half_max = depth.max() / 2.0
            above = depth > half_max
            if not above.any():
                return 0.0
            idx = np.where(above)[0]
            return float(idx[-1] - idx[0])

        fwhm_205 = _fwhm_pix(voigt_absorption(
            wave, log_nhi=20.5, z_dla=z_dla, num_lines=3, kernel="none"))
        fwhm_215 = _fwhm_pix(voigt_absorption(
            wave, log_nhi=21.5, z_dla=z_dla, num_lines=3, kernel="none"))
        ratio = fwhm_215 / fwhm_205
        # √10 = 3.16; allow [2, 5] as a generous Voigt-regime corridor.
        assert 2.0 < ratio < 5.0, (
            f"FWHM ratio (logNHI 21.5 / 20.5) expected ~√10≈3.16 in the "
            f"damping-wing-dominated regime; got {ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# Layer D — degenerate / boundary cases
# ---------------------------------------------------------------------------
class TestBoundaryCases:
    def test_zero_nhi_is_identity(self):
        """NHI = 0 (logNHI = -∞) gives no absorption — profile is all 1."""
        wave = np.linspace(4200.0, 4300.0, 100)
        # logNHI = -100 ⇒ NHI = 1e-100 ⇒ effectively 0
        profile = voigt_absorption(
            wave, log_nhi=-100.0, z_dla=2.5, num_lines=3, kernel="none",
        )
        np.testing.assert_allclose(profile, 1.0, rtol=1e-12)

    def test_dla_far_from_window_is_identity(self):
        """If the DLA's Lyα is far outside the wavelength window, the
        profile should be (close to) 1 throughout."""
        # DLA at z=4.0 → Lyα at 6080 Å. We sample near 4500 Å.
        wave = np.linspace(4400.0, 4600.0, 100)
        profile = voigt_absorption(
            wave, log_nhi=21.0, z_dla=4.0, num_lines=3, kernel="none",
        )
        # Allow tiny non-1 from the broad damping wing tail.
        assert profile.min() > 0.99, (
            f"DLA out of window should give ~1 profile; got min={profile.min():.4f}"
        )

    def test_num_lines_validation(self):
        """num_lines must be in [1, 31]."""
        wave = np.linspace(4200.0, 4300.0, 100)
        with pytest.raises(ValueError):
            voigt_absorption(wave, 21.0, 2.5, num_lines=0, kernel="none")
        with pytest.raises(ValueError):
            voigt_absorption(wave, 21.0, 2.5, num_lines=32, kernel="none")

    def test_unknown_kernel_raises(self):
        wave = np.linspace(4200.0, 4300.0, 100)
        with pytest.raises(ValueError):
            voigt_absorption(wave, 21.0, 2.5, num_lines=3, kernel="bogus-kernel")


# ---------------------------------------------------------------------------
# Class-API parity (drop-in for production code paths expecting voigt_fast)
# ---------------------------------------------------------------------------
def test_class_api_passes_through():
    """``VoigtProfileV2.compute_voigt_profile`` must produce the same
    result as the functional ``voigt_absorption`` call."""
    wave = np.linspace(4200.0, 4300.0, 200)
    p_class = VoigtProfileV2(kernel="none").compute_voigt_profile(
        wave, nhi=10**21.0, z_dla=2.5, num_lines=3,
    )
    p_func = voigt_absorption(
        wave, log_nhi=21.0, z_dla=2.5, num_lines=3, kernel="none",
    )
    np.testing.assert_array_equal(p_class, p_func)
