"""Tests for voigt_absorption_batched — correctness vs serial loop and
Bayesian-correctness vs the GP log-likelihood it propagates into.

The function under test is a vectorised + windowed Voigt evaluator
intended to replace the per-QMC-sample loop in
``gpy_dla_detection.dla_gp.this_dla_gp``. It must produce profiles
that are bit-equivalent to the serial loop in the unwindowed case,
and only mildly different in the windowed case — small enough that
the resulting GP log-likelihood doesn't change beyond floating-point
noise relevant to Bayesian model selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpy_dla_detection.voigt_v2 import (
    _adaptive_window_AA,
    voigt_absorption,
    voigt_absorption_batched,
)


_LYA_AA = 1215.6701


def _make_qmc_samples(n_samples: int, log_nhi_lo: float, log_nhi_hi: float,
                      z_lo: float = 2.0, z_hi: float = 3.5, seed: int = 0):
    """Realistic QMC sample distribution: uniform z, uniform log_nhi
    within the requested range."""
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(log_nhi_lo, log_nhi_hi, n_samples),
        rng.uniform(z_lo, z_hi, n_samples),
    )


def _serial_loop_reference(
    wavelengths_A: np.ndarray, log_nhi_arr: np.ndarray,
    z_dla_arr: np.ndarray, num_lines: int, kernel: str,
):
    """Independent reference: call the (already-tested) single-sample
    voigt_absorption in a Python loop. This is the slow but trusted
    answer; the batched version must match it."""
    profiles = []
    for log_nhi, z in zip(log_nhi_arr, z_dla_arr):
        p = voigt_absorption(
            wavelengths_A, log_nhi=float(log_nhi), z_dla=float(z),
            num_lines=num_lines, kernel=kernel,
        )
        profiles.append(p)
    return np.array(profiles)


# ---------------------------------------------------------------------------
# Layer A — bit-equivalent to serial for use_window=False
# ---------------------------------------------------------------------------
class TestUnwindowedParity:
    """With use_window=False the batched evaluator must produce
    identical results to the serial loop down to float64 precision."""

    @pytest.mark.parametrize("kernel,num_lines,n_samples,n_pix", [
        ("none",            3, 50, 1500),
        ("none",            6, 30, 1500),
        ("boss-log-r2000",  3, 50, 1500),
        ("desi-linear-r3000", 3, 30, 1500),
    ])
    def test_batched_full_matches_serial_loop(
        self, kernel, num_lines, n_samples, n_pix
    ):
        wave = np.linspace(3700.0, 3700.0 + 0.15 * (n_pix - 1), n_pix)
        log_nhi_arr, z_dla_arr = _make_qmc_samples(
            n_samples, log_nhi_lo=20.0, log_nhi_hi=22.5, z_lo=2.1, z_hi=2.3,
        )

        ref = _serial_loop_reference(wave, log_nhi_arr, z_dla_arr,
                                      num_lines, kernel)
        batched = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr,
            num_lines=num_lines, kernel=kernel, use_window=False,
        )
        assert ref.shape == batched.shape, (
            f"shape mismatch: ref={ref.shape}, batched={batched.shape}"
        )
        max_abs = np.max(np.abs(ref - batched))
        # Full-evaluation case: should be floating-point identical.
        assert max_abs < 1e-12, f"max |Δ| = {max_abs:.3e}"


# ---------------------------------------------------------------------------
# Layer B — adaptive windowing keeps absorption error ≤ 1e-3 per pixel
# ---------------------------------------------------------------------------
class TestAdaptiveWindowing:
    """The adaptive window function must be wide enough that the
    truncation error in the absorption profile is bounded across all
    NHI regimes."""

    def test_window_grows_with_nhi(self):
        """Higher NHI → larger damping wing → wider safe window."""
        windows = [_adaptive_window_AA(log_nhi)
                   for log_nhi in (17.0, 18.0, 19.0, 20.0, 21.0, 22.0)]
        for a, b in zip(windows[:-1], windows[1:]):
            assert b >= a, f"windows non-monotonic: {windows}"

    def test_windowed_low_error_across_nhi(self):
        """Per-pixel absorption error from windowing < 1e-3 for the
        sweep's full NHI range. This is the key correctness gate.

        Tested across LLS / sub-DLA / DLA / strong-DLA, kernel='none'."""
        n_pix = 4000
        wave = np.linspace(3500.0, 3500.0 + 0.15 * (n_pix - 1), n_pix)

        for log_nhi in (17.5, 18.5, 19.5, 20.5, 21.5, 22.0):
            log_nhi_arr = np.array([log_nhi])
            z_dla_arr = np.array([2.1])  # fixed centre

            full = voigt_absorption_batched(
                wave, log_nhi_arr, z_dla_arr,
                num_lines=3, kernel="none", use_window=False,
            )
            win = voigt_absorption_batched(
                wave, log_nhi_arr, z_dla_arr,
                num_lines=3, kernel="none", use_window=True,
            )
            max_abs = float(np.max(np.abs(full - win)))
            assert max_abs < 1e-3, (
                f"log_nhi={log_nhi}: windowing error {max_abs:.3e} > 1e-3"
            )

    def test_windowed_with_lsf_low_error(self):
        """After convolution with the production LSF kernel, the same
        error budget should hold (LSF can't amplify a 1e-3 absorption
        error by more than O(kernel-width) which is < 7)."""
        n_pix = 1500
        wave = np.linspace(3700.0, 3700.0 + 0.15 * (n_pix - 1), n_pix)
        log_nhi_arr = np.array([21.5])
        z_dla_arr = np.array([2.1])

        full = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr,
            num_lines=3, kernel="boss-log-r2000", use_window=False,
        )
        win = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr,
            num_lines=3, kernel="boss-log-r2000", use_window=True,
        )
        max_abs = float(np.max(np.abs(full - win)))
        # Bound: 1e-3 (window error) × 7 (kernel taps × max amplitude) ≈ 7e-3.
        assert max_abs < 7e-3, (
            f"LSF-convolved windowing error {max_abs:.3e} > 7e-3"
        )


# ---------------------------------------------------------------------------
# Layer C — Bayesian-correctness: GP log-likelihood error is bounded
# ---------------------------------------------------------------------------
class TestBayesianCorrectness:
    """The downstream consumer of voigt_absorption is the GP
    log-likelihood ``log p(y | M_DLA(z, NHI))``. The windowing error
    must propagate to a log-likelihood error small enough that
    Bayesian model selection (Δlog p ~ 1+ between hypotheses) is not
    affected.

    Strategy: synthesise an absorbed spectrum, then compute the
    log-likelihood with full-Voigt and windowed-batched-Voigt. Verify
    Δlog p < 0.1 across NHI regimes — well below the typical 10–100
    log-evidence differences in the inference.
    """

    def test_windowing_log_likelihood_error_bounded(self):
        """For a synthetic GP-like spectrum, the |log_p_full - log_p_window|
        must be < 0.1 nat across all NHI regimes — well below typical
        Bayes-factor differences (10-100 nats).

        Use kernel='none' to keep shapes simple. The LSF is independent
        of windowing correctness anyway."""
        n_pix = 2000
        wave = np.linspace(3700.0, 3700.0 + 0.15 * (n_pix - 1), n_pix)

        rng = np.random.default_rng(7)
        mu = 1.0 + 0.05 * np.sin(np.arange(n_pix) * 0.01)
        k_rank = 5
        M = rng.normal(0, 0.1, size=(n_pix, k_rank))
        noise_v = np.full(n_pix, 0.05 ** 2)
        omega2 = np.full(n_pix, 0.1 ** 2)

        # For each log_nhi we test at, generate matching truth data and
        # then check |log p_full - log p_win| at that same log_nhi.
        # This isolates the windowing error from the well-known
        # bad-fit-amplifies-tiny-changes regime.
        from gpy_dla_detection.null_gp import NullGP

        truth_z = 2.07
        for log_nhi in (17.5, 19.5, 20.5):
            # Synthetic data WITH ABSORPTION AT THIS log_nhi (good fit).
            truth_abs = voigt_absorption(
                wave, log_nhi, truth_z, num_lines=3, kernel="none",
            )
            y = mu * truth_abs + rng.normal(0, 0.05, n_pix)

            log_nhi_arr = np.array([log_nhi])
            z_arr = np.array([truth_z])

            full = voigt_absorption_batched(
                wave, log_nhi_arr, z_arr, num_lines=3,
                kernel="none", use_window=False,
            )[0]
            win = voigt_absorption_batched(
                wave, log_nhi_arr, z_arr, num_lines=3,
                kernel="none", use_window=True,
            )[0]

            # Apply absorption to the GP mean function and low-rank basis.
            mu_full = mu * full
            M_full = M * full[:, None]
            omega2_full = omega2 * full ** 2

            mu_win = mu * win
            M_win = M * win[:, None]
            omega2_win = omega2 * win ** 2

            log_p_full = NullGP.log_mvnpdf_low_rank(
                y, mu_full, M_full, omega2_full + noise_v,
            )
            log_p_win = NullGP.log_mvnpdf_low_rank(
                y, mu_win, M_win, omega2_win + noise_v,
            )
            d_log = abs(log_p_full - log_p_win)
            assert d_log < 0.1, (
                f"log_nhi={log_nhi}: |Δ log p| = {d_log:.4e} > 0.1"
            )


# ---------------------------------------------------------------------------
# Layer D — wofz invariance under broadcast vs serial scattering
# ---------------------------------------------------------------------------
class TestVectorisationInvariance:
    """When a sample's window is constructed from `np.searchsorted`,
    the same wave_cm subset must be passed to wofz as in the serial
    loop. Catch any off-by-one in the slice."""

    def test_searchsorted_endpoints(self):
        """Boundary cases for window-pixel indexing — at log_nhi=20.0
        (in the windowable regime), the windowed and full results must
        agree to better than 1e-3."""
        wave = np.linspace(4000.0, 4500.0, 5000)
        log_nhi_arr = np.array([20.0])
        # z chosen so the line falls EXACTLY at a grid pixel.
        z = 4254.85 / _LYA_AA - 1   # ≈ 2.5
        z_dla_arr = np.array([z])
        full = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr, num_lines=3,
            kernel="none", use_window=False,
        )[0]
        win = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr, num_lines=3,
            kernel="none", use_window=True,
        )[0]
        max_abs = np.max(np.abs(full - win))
        assert max_abs < 1e-3, f"max_abs={max_abs:.3e}"

    def test_window_off_grid_low_nhi(self):
        """If a Lyman line falls off the grid AND we're in the windowable
        regime (log_nhi < 21), the windowed evaluator must still return
        identity (no IndexError, no spurious absorption)."""
        wave = np.linspace(4000.0, 4500.0, 5000)
        # Lyα at z=10 → 13373 Å, way off grid.
        log_nhi_arr = np.array([18.0])  # LLS — windowable
        z_dla_arr = np.array([10.0])
        win = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr, num_lines=3,
            kernel="none", use_window=True,
        )[0]
        # Should be identity (no absorption) within rounding.
        assert np.max(np.abs(win - 1.0)) < 1e-12

    def test_window_off_grid_high_nhi_falls_back(self):
        """At log_nhi ≥ 21 the table returns inf → full evaluation is
        used. With a far-off-grid line, the Lorentzian wing may contribute
        a tiny absorption (~1e-5) — verify it's bounded, not zero."""
        wave = np.linspace(4000.0, 4500.0, 5000)
        log_nhi_arr = np.array([22.0])
        z_dla_arr = np.array([10.0])  # line off grid by ~9000 Å
        full = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr, num_lines=3,
            kernel="none", use_window=False,
        )[0]
        win = voigt_absorption_batched(
            wave, log_nhi_arr, z_dla_arr, num_lines=3,
            kernel="none", use_window=True,
        )[0]
        # Windowed should equal full (both do full eval at log_nhi=22).
        assert np.max(np.abs(full - win)) < 1e-12, (
            "windowed at log_nhi=22 must fall back to full evaluation"
        )


# ---------------------------------------------------------------------------
# Layer E — speed sanity
# ---------------------------------------------------------------------------
def test_batched_windowed_speed_in_lls_regime(capsys):
    """Sanity check: in the LLS/sub-DLA regime (log_nhi < 20), where
    windowing IS effective, batched-windowed should be at least 1.5×
    faster than serial-full. For DLA-mode (log_nhi ≥ 21) windowing
    falls back to full evaluation and gives no speedup — that's expected
    and tested via test_high_nhi_falls_back_to_full above."""
    import time
    n_pix = 4000
    wave = np.linspace(3700.0, 3700.0 + 0.15 * (n_pix - 1), n_pix)
    n_samples = 500
    # LLS / sub-DLA range: windowing is most effective here.
    log_nhi_arr, z_dla_arr = _make_qmc_samples(
        n_samples, 17.5, 19.5, z_lo=2.1, z_hi=2.3,
    )

    t0 = time.perf_counter()
    _serial_loop_reference(wave, log_nhi_arr, z_dla_arr, 3, "none")
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    voigt_absorption_batched(
        wave, log_nhi_arr, z_dla_arr,
        num_lines=3, kernel="none", use_window=True,
    )
    t_batched_win = time.perf_counter() - t0

    speedup = t_serial / t_batched_win
    with capsys.disabled():
        print(f"\n  serial-full (log_nhi 17.5-19.5):  {t_serial:.2f} s for {n_samples} samples")
        print(f"  batched-window:                    {t_batched_win:.2f} s ({speedup:.2f}× speedup)")

    assert speedup > 1.5, f"speedup only {speedup:.2f}× in LLS regime — investigate"
