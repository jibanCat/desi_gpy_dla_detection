"""
tests/test_inject_absorber.py
=============================
TDD tests for ``gpy_dla_detection.inject_absorber`` — the M3 injection module.

Injection is INPUT-FLUX preprocessing: we multiply an observed-frame flux by the
*same* Voigt transmission profile the GP's DLA forward model would multiply in
(``dla_gp.this_dla_gp``), so a GP run on the injected spectrum is a faithful test
of recovery. ``dla_gp.py`` / the GP are never modified.

Edge / frame convention under test (must match ``dla_gp.this_dla_gp`` exactly):
  - ``wavelengths`` is the OBSERVED-frame grid (Å), equally log-spaced.
  - ``VoigtProfile().compute_voigt_profile`` returns a profile trimmed by
    ``2 * width`` pixels (``width = 3``, hardcoded in ctypes_voigt.c) — 3 from
    each end. Output pixel ``i`` is the broadening-convolution of input pixels
    ``[i, i+6]``, i.e. it aligns with input pixel ``i + 3``.
  - ``dla_gp`` restores full length by PADDING the wavelength grid with
    ``params.width = 3`` log-spaced pixels on each side, calling Voigt on the
    padded grid, and using the (trimmed) result aligned pixel-for-pixel with the
    original (unpadded) grid. ``inject_voigt`` reproduces this so there is NO
    off-by-3-pixel shift between the injected absorber and the grid the GP scores.
  - ``nhi`` is the LINEAR column density (cm^-2), i.e. ``10**logNHI`` — exactly
    what ``dla_gp`` passes to the C extension (``nhis``, not ``log_nhis``).

The physics tests ``importorskip`` the compiled C extension; the pure-logic
tests (shape, import) run without it.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LYA = 1215.67  # Å, Lyman-alpha rest wavelength


def _log_grid(lam_min, lam_max, n):
    """Equally log-spaced observed-frame grid (matches the GP's pixel spacing)."""
    return np.logspace(np.log10(lam_min), np.log10(lam_max), n)


def _equivalent_width(wavelengths, transmission):
    """Rest-frameless EW proxy: integral of (1 - transmission) dlambda."""
    absorbed = 1.0 - transmission
    return np.trapezoid(absorbed, wavelengths)


# Skip physics tests if the compiled Voigt C extension is unavailable.
def _require_c_voigt():
    voigt_fast = pytest.importorskip("gpy_dla_detection.voigt_fast")
    try:
        voigt_fast.VoigtProfile()
    except (OSError, ImportError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"compiled _voigt.so unavailable: {exc}")
    return voigt_fast


# ---------------------------------------------------------------------------
# Pure-logic tests (no C extension required)
# ---------------------------------------------------------------------------

def test_module_imports_without_c_extension():
    """The module must import even if _voigt.so is missing (lazy-loaded profile)."""
    import gpy_dla_detection.inject_absorber as inj

    assert hasattr(inj, "inject_voigt")
    assert hasattr(inj, "inject_multiple")


def test_shape_preserved_and_transmission_bounded():
    """injected_flux has the SAME shape as flux; transmission in [0, 1]."""
    _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt

    z = 3.0
    lam0 = (1 + z) * LYA
    w = _log_grid(lam0 - 400, lam0 + 400, 600)
    flux = np.ones_like(w) * 2.5  # arbitrary continuum level

    out = inject_voigt(w, flux, nhi=10**21.0, z_dla=z, num_lines=3)

    assert out.shape == flux.shape
    transmission = out / flux
    assert np.all(transmission <= 1.0 + 1e-9)
    assert np.all(transmission >= -1e-9)


# ---------------------------------------------------------------------------
# Physics tests
# ---------------------------------------------------------------------------

def test_strong_dla_damped_trough_at_right_place():
    """A strong DLA (logN=21.5) → deep trough centred at (1+z)*1215.67, flux->~0;
    far from the line transmission -> ~1 (continuum unchanged)."""
    _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt

    z = 3.0
    lam0 = (1 + z) * LYA
    w = _log_grid(lam0 - 500, lam0 + 500, 800)
    flux = np.ones_like(w)

    out = inject_voigt(w, flux, nhi=10**21.5, z_dla=z, num_lines=3)
    transmission = out / flux

    # Trough centre at (1+z)*Lya
    i_center = int(np.argmin(np.abs(w - lam0)))
    assert transmission[i_center] < 0.05, "DLA core should be near-black"

    # Deep: the minimum drops toward ~0
    assert transmission.min() < 0.01

    # The minimum is located at the Lya line, not some spurious place.
    i_min = int(np.argmin(transmission))
    assert abs(w[i_min] - lam0) < 5.0, "trough not centred on (1+z)*Lya"

    # Far blue of the line (well outside the damping wings), continuum ~unchanged.
    far_blue = w < (lam0 - 300)
    assert np.all(transmission[far_blue] > 0.95)


def test_monotonic_in_nhi_subdla_shallower_than_dla():
    """EW increases with N_HI; a sub-DLA (logN=19) trough is shallower/narrower
    than a DLA (logN=21). Covers the NHI<20 regime."""
    _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt

    z = 2.5
    lam0 = (1 + z) * LYA
    w = _log_grid(lam0 - 500, lam0 + 500, 900)
    flux = np.ones_like(w)

    ews = []
    mins = []
    for logN in (18.0, 19.0, 20.0, 21.0, 22.0):
        out = inject_voigt(w, flux, nhi=10**logN, z_dla=z, num_lines=3)
        t = out / flux
        ews.append(_equivalent_width(w, t))
        mins.append(t.min())

    ews = np.array(ews)
    mins = np.array(mins)

    # Equivalent width strictly increases with N_HI.
    assert np.all(np.diff(ews) > 0), f"EW not monotonic in NHI: {ews}"

    # Sub-DLA (logN=19) is shallower (higher min transmission) than DLA (logN=21).
    # ews/mins index: [18, 19, 20, 21, 22]
    assert mins[1] > mins[3], "sub-DLA should be shallower than DLA"


def test_self_consistency_with_gp_forward_model():
    """Injected transmission == the profile dla_gp itself multiplies in, for the
    SAME (nhi, z, num_lines), with NO off-by-3-pixel shift.

    dla_gp.this_dla_gp builds the absorption by padding the wavelength grid with
    params.width=3 pixels on each side and calling
    voigt_absorption(padded, z_dla, nhi, num_lines), whose output (trimmed by 6)
    aligns pixel-for-pixel with the UNPADDED grid. We reproduce that here and
    require an exact match.
    """
    voigt_fast = _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt

    z = 2.8
    lam0 = (1 + z) * LYA
    # Use an equally log-spaced grid so the padding is well-defined.
    w = _log_grid(lam0 - 400, lam0 + 400, 700)
    flux = np.full_like(w, 3.0)
    nhi = 10**20.7
    num_lines = 3

    out = inject_voigt(w, flux, nhi=nhi, z_dla=z, num_lines=num_lines)
    transmission = out / flux

    # Reconstruct EXACTLY as dla_gp.this_dla_gp does: pad by width=3 log-spaced
    # pixels each side, call Voigt, output aligns to the original grid.
    width = 3
    pixel_spacing = 1e-4  # dex, Parameters default DLAMBDA spacing
    left = np.logspace(
        np.log10(w.min()) - width * pixel_spacing,
        np.log10(w.min()) - pixel_spacing,
        width,
    )
    right = np.logspace(
        np.log10(w.max()) + pixel_spacing,
        np.log10(w.max()) + width * pixel_spacing,
        width,
    )
    padded = np.concatenate([left, w, right])
    expected = voigt_fast.VoigtProfile().compute_voigt_profile(
        padded, nhi, z, num_lines
    )

    assert expected.shape == w.shape, "padded Voigt output must align to grid"
    np.testing.assert_allclose(transmission, expected, rtol=0, atol=1e-12)


def test_no_off_by_three_pixel_shift():
    """The trough minimum sits at the same observed-pixel index whether computed
    via inject_voigt or via the raw (unpadded) Voigt offset by +3 pixels. Guards
    against the classic 2*3 edge-trim misalignment that would bias recovered N_HI.
    """
    voigt_fast = _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt

    z = 3.1
    lam0 = (1 + z) * LYA
    w = _log_grid(lam0 - 300, lam0 + 300, 600)
    flux = np.ones_like(w)

    t = inject_voigt(w, flux, nhi=10**21.0, z_dla=z, num_lines=3) / flux
    i_inject = int(np.argmin(t))

    # Raw (unpadded) Voigt: output index j corresponds to input pixel j+3.
    raw = voigt_fast.VoigtProfile().compute_voigt_profile(w, 10**21.0, z, 3)
    j_raw = int(np.argmin(raw))
    i_raw_in_full_grid = j_raw + 3

    assert i_inject == i_raw_in_full_grid, (
        f"off-by edge misalignment: inject min @ {i_inject}, "
        f"raw+3 @ {i_raw_in_full_grid}"
    )


def test_num_lines_plumbing_adds_blueward_troughs():
    """More Lyman lines (num_lines>1) adds Lybeta/Lygamma troughs BLUEWARD of
    Lyalpha. With num_lines=1 those extra troughs are absent."""
    _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt

    z = 3.0
    lam_lya = (1 + z) * LYA
    lam_lyb = (1 + z) * 1025.7222  # Lyman-beta rest wavelength
    # Grid must span from Lybeta to a bit redward of Lyalpha.
    w = _log_grid(lam_lyb - 100, lam_lya + 200, 1400)
    flux = np.ones_like(w)

    t1 = inject_voigt(w, flux, nhi=10**21.0, z_dla=z, num_lines=1) / flux
    t3 = inject_voigt(w, flux, nhi=10**21.0, z_dla=z, num_lines=3) / flux

    # Around Lyman-beta: with 3 lines there is extra absorption; with 1 line ~none.
    near_lyb = np.abs(w - lam_lyb) < 15.0
    assert near_lyb.any()
    assert t3[near_lyb].min() < 0.5, "Lybeta trough missing with num_lines=3"
    assert t1[near_lyb].min() > 0.95, "Lybeta should be absent with num_lines=1"

    # Both share the Lyalpha trough.
    near_lya = np.abs(w - lam_lya) < 10.0
    assert t1[near_lya].min() < 0.1
    assert t3[near_lya].min() < 0.1


def test_inject_multiple_close_pair_multiplicative():
    """inject_multiple applies several profiles multiplicatively (Campaign B
    close pairs). Result equals the product of single-absorber injections."""
    _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt, inject_multiple

    z1, z2 = 2.90, 2.92
    lam0 = (1 + z1) * LYA
    w = _log_grid(lam0 - 500, lam0 + 500, 900)
    flux = np.full_like(w, 1.7)

    absorbers = [
        {"nhi": 10**20.8, "z_dla": z1, "num_lines": 3},
        {"nhi": 10**20.5, "z_dla": z2, "num_lines": 3},
    ]
    out_multi = inject_multiple(w, flux, absorbers)

    # Equivalent: inject sequentially.
    step = inject_voigt(w, flux, nhi=10**20.8, z_dla=z1, num_lines=3)
    step = inject_voigt(w, step, nhi=10**20.5, z_dla=z2, num_lines=3)

    np.testing.assert_allclose(out_multi, step, rtol=0, atol=1e-12)
    # And the pair is deeper than either alone.
    single = inject_voigt(w, flux, nhi=10**20.8, z_dla=z1, num_lines=3)
    assert out_multi.min() <= single.min() + 1e-12
