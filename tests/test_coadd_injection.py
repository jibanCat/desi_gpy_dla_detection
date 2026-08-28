"""
tests/test_coadd_injection.py
=============================
TDD tests for ``injection/coadd_injection.py`` — the M3 injection-campaign
I/O machinery owned by the scientific-software / CS agent:

  1. ``build_clean_table``     — clean-sightline selection table
  2. ``inject_into_coadd``     — Voigt injection into a desispec coadd (+schema proof)
  3. ``write_campaign``        — orchestrate: group manifest by healpix, inject, write
                                  an injectable healpix tree + truth manifest.

Discipline: reuse ``gpy_dla_detection.inject_absorber.inject_voigt`` and
``desispec.io`` (never reimplement Voigt or the coadd schema); never touch
``dla_gp.py`` / inference. Real LOA spectra never leave NERSC — these tests use
synthetic desispec ``Spectra`` objects built in-process (no real-data dependency).

Pure-logic tests (clean-table set algebra) ``importorskip`` astropy only; the
coadd I/O tests ``importorskip`` desispec + the compiled Voigt C extension.
"""
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------

LYA = 1215.67


def _require_desispec():
    return pytest.importorskip("desispec.io")


def _require_c_voigt():
    voigt_fast = pytest.importorskip("gpy_dla_detection.voigt_fast")
    try:
        voigt_fast.VoigtProfile()
    except (OSError, ImportError) as exc:  # pragma: no cover - env dependent
        pytest.skip(f"compiled _voigt.so unavailable: {exc}")
    return voigt_fast


# ---------------------------------------------------------------------------
# Synthetic catalog / spectra fixtures (no real data)
# ---------------------------------------------------------------------------


def _fake_catalogs():
    """Return (zcat, hcd_truth, bal_cat, snr_cat) astropy Tables with a known
    clean/contaminated split so the set algebra is fully pinned.

    TARGETIDs: 100..107 (int64). Contamination:
      - 102, 103 carry a truth HCD absorber
      - 104       carries a BAL
      - 105       is BOTH HCD and BAL
      - 100, 101, 106, 107 are CLEAN.
    """
    Table = pytest.importorskip("astropy.table").Table

    tids = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.int64)
    # RA/DEC chosen so several land on the same nside=16 healpix (grouping test).
    ra = np.array([10.0, 10.1, 20.0, 20.1, 30.0, 30.1, 40.0, 40.1])
    dec = np.array([5.0, 5.0, -5.0, -5.0, 0.0, 0.0, 15.0, 15.0])
    z = np.array([2.5, 2.6, 3.0, 3.1, 2.8, 2.9, 3.3, 2.4])

    zcat = Table(
        {"TARGETID": tids, "TARGET_RA": ra, "TARGET_DEC": dec, "Z": z}
    )
    hcd = Table(
        {
            "TARGETID": np.array([102, 103, 105], dtype=np.int64),
            "NHI": np.array([20.5, 21.0, 19.5]),
            "Z": np.array([2.4, 2.7, 2.5]),
        }
    )
    bal = Table(
        {"TARGETID": np.array([104, 105], dtype=np.int64)}
    )
    snr = Table(
        {
            "TARGETID": tids,
            "SNR_FOREST": np.array([1.0, 3.0, 5.0, 2.0, 4.0, 1.5, 8.0, 0.5]),
            "SNR_REDSIDE": np.array([2.0, 4.0, 6.0, 3.0, 5.0, 2.5, 9.0, 1.5]),
        }
    )
    return zcat, hcd, bal, snr


def _make_spectra(targetids, ra, dec, *, flux_level=1.0):
    """Build a minimal 3-camera desispec ``Spectra`` covering the Lyα forest.

    Wavelength ranges mimic DESI b/r/z so a z~2.8 Lyα trough lands in the b camera.
    """
    desispec_io = _require_desispec()  # noqa: F841 ensure available
    from desispec.spectra import Spectra
    from astropy.table import Table

    n = len(targetids)
    waves = {
        "b": np.arange(3600.0, 5800.0, 0.8),
        "r": np.arange(5760.0, 7620.0, 0.8),
        "z": np.arange(7520.0, 9824.0, 0.8),
    }
    flux, ivar, mask = {}, {}, {}
    for cam, w in waves.items():
        flux[cam] = np.full((n, w.size), float(flux_level), dtype=np.float64)
        ivar[cam] = np.full((n, w.size), 4.0, dtype=np.float64)  # noise var 0.25
        mask[cam] = np.zeros((n, w.size), dtype=np.uint32)

    fibermap = Table(
        {
            "TARGETID": np.asarray(targetids, dtype=np.int64),
            "TARGET_RA": np.asarray(ra, dtype=np.float64),
            "TARGET_DEC": np.asarray(dec, dtype=np.float64),
            "FIBER": np.arange(n, dtype=np.int32),
        }
    )
    return Spectra(
        bands=["b", "r", "z"],
        wave=waves,
        flux=flux,
        ivar=ivar,
        mask=mask,
        fibermap=fibermap,
    )


# ---------------------------------------------------------------------------
# M4 round-trip helpers: a coadd that EXERCISES the production resample path
# (mismatched camera grids → coadd_cameras fails → resample_spectra_lin_or_log
# at linear_step=0.8 → brz), so we validate the GRID the GP actually scores on.
# ---------------------------------------------------------------------------


def _gauss_resmatrix(nwave, ndiag=11, sigma_pix=1.1):
    """A desispec diagonal-storage resolution matrix ``(ndiag, nwave)``.

    Each column is a normalized Gaussian across the ``ndiag`` band offsets,
    matching the real 2LPT ``truth-16-*.fits`` ``{cam}_RESOLUTION`` HDU layout
    (shape ``(ndiag, nwave)``, ndiag=11) that ``process_spectra_group`` reads.
    """
    offsets = np.arange(ndiag) - ndiag // 2
    g = np.exp(-0.5 * (offsets / sigma_pix) ** 2)
    g /= g.sum()
    return np.repeat(g[:, None], nwave, axis=1).astype(np.float64)


def _make_resolved_spectra(targetids, ra, dec, *, flux_level=1.0):
    """3-camera ``Spectra`` with DESI-like MISMATCHED grids + a resolution matrix.

    The b/r/z grids are ~0.8 Å linear but with the real-data quirks that make
    ``coadd_cameras`` reject them up front (camera r at step 0.7998, overlapping
    ranges) — so the production fallback (resample to a common 0.8 Å brz grid
    using the resolution data) is what runs. Resolution data is attached so the
    resample broadens exactly as production does.
    """
    _require_desispec()
    from desispec.spectra import Spectra
    from astropy.table import Table

    n = len(targetids)
    waves = {
        "b": np.arange(3600.0, 5800.0, 0.8),
        "r": np.arange(5760.0, 7620.0, 0.7998),
        "z": np.arange(7520.0, 9824.0, 0.7998),
    }
    flux, ivar, mask, res = {}, {}, {}, {}
    for cam, w in waves.items():
        flux[cam] = np.full((n, w.size), float(flux_level), dtype=np.float64)
        ivar[cam] = np.full((n, w.size), 25.0, dtype=np.float64)
        mask[cam] = np.zeros((n, w.size), dtype=np.uint32)
        rm = _gauss_resmatrix(w.size)
        res[cam] = np.repeat(rm[None, :, :], n, axis=0)  # (n, ndiag, nwave)

    fibermap = Table(
        {
            "TARGETID": np.asarray(targetids, dtype=np.int64),
            "TARGET_RA": np.asarray(ra, dtype=np.float64),
            "TARGET_DEC": np.asarray(dec, dtype=np.float64),
            "FIBER": np.arange(n, dtype=np.int32),
        }
    )
    return Spectra(
        bands=["b", "r", "z"],
        wave=waves,
        flux=flux,
        ivar=ivar,
        mask=mask,
        resolution_data=res,
        fibermap=fibermap,
    )


def _production_resample(spec):
    """Replicate ``dlasearch.process_spectra_group``'s coadd→brz exactly.

    Tries ``coadd_cameras`` first (fails on mismatched grids, as for mocks), then
    resamples to the common linear 0.8 Å grid and coadds — the grid the GP scores
    on. Returns the resampled ``Spectra`` with a ``brz`` band.
    """
    import copy

    from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log

    spec = copy.deepcopy(spec)
    try:
        return coadd_cameras(spec)
    except Exception:
        spec = resample_spectra_lin_or_log(
            spec,
            linear_step=0.8,
            wave_min=np.min(spec.wave["b"]),
            wave_max=np.max(spec.wave["z"]),
            fast=True,
        )
        return coadd_cameras(spec)


# ===========================================================================
# 1. Clean-sightline table
# ===========================================================================


def test_build_clean_table_set_difference():
    """clean = zcat.TARGETID − hcd.TARGETID − bal.TARGETID, with SNR + healpix
    joined. 102/103 (HCD), 104 (BAL), 105 (both) are removed; 100/101/106/107
    remain."""
    pytest.importorskip("astropy.table")
    from injection.coadd_injection import build_clean_table

    zcat, hcd, bal, snr = _fake_catalogs()
    table = build_clean_table(zcat, hcd, bal, snr)

    clean_tids = set(np.asarray(table["TARGETID"]).tolist())
    assert clean_tids == {100, 101, 106, 107}
    # Contaminated never present.
    assert not ({102, 103, 104, 105} & clean_tids)


def test_build_clean_table_int64_targetid_and_columns():
    """TARGETID stays int64 (no float coercion → no precision loss on 19-digit
    DESI IDs); the table carries SNR_FOREST and a HEALPIX column."""
    pytest.importorskip("astropy.table")
    from injection.coadd_injection import build_clean_table

    zcat, hcd, bal, snr = _fake_catalogs()
    table = build_clean_table(zcat, hcd, bal, snr)

    assert np.asarray(table["TARGETID"]).dtype == np.int64
    for col in ("TARGETID", "Z", "SNR_FOREST", "HEALPIX"):
        assert col in table.colnames, f"missing column {col}"
    # SNR joined correctly for tid=106 (SNR_FOREST=8.0 in the fake snr_cat).
    row106 = table[np.asarray(table["TARGETID"]) == 106][0]
    assert row106["SNR_FOREST"] == pytest.approx(8.0)


def test_build_clean_table_healpix_matches_ang2pix():
    """HEALPIX equals hp.ang2pix(16, RA, DEC, nest=True, lonlat=True) — the DESI
    nside=16 nested assignment the GP driver's tree uses."""
    hp = pytest.importorskip("healpy")
    pytest.importorskip("astropy.table")
    from injection.coadd_injection import build_clean_table

    zcat, hcd, bal, snr = _fake_catalogs()
    table = build_clean_table(zcat, hcd, bal, snr)

    for row in table:
        expected = hp.ang2pix(
            16, row["TARGET_RA"], row["TARGET_DEC"], nest=True, lonlat=True
        )
        assert int(row["HEALPIX"]) == int(expected)


# ===========================================================================
# 2. Coadd injector + schema-preservation proof
# ===========================================================================


def test_inject_into_coadd_roundtrip_and_schema(tmp_path):
    """A NEW coadd is written that ``desispec.io.read_spectra`` re-reads with the
    SAME bands / wave grid / fibermap as the input — so ``dlasearch`` runs on it
    unchanged."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import inject_into_coadd

    tids = np.array([100, 101, 102], dtype=np.int64)
    ra = np.array([10.0, 10.1, 10.2])
    dec = np.array([5.0, 5.0, 5.0])
    spec = _make_spectra(tids, ra, dec, flux_level=2.0)

    in_path = tmp_path / "coadd-in.fits"
    out_path = tmp_path / "coadd-out.fits"
    desispec_io.write_spectra(str(in_path), spec)

    # Inject a strong DLA into the FIRST fiber only.
    injections = [
        {"target_id": 100, "logN_true": 21.5, "z_true": 2.8, "num_lines": 3},
    ]
    inject_into_coadd(str(in_path), str(out_path), injections, num_lines=3)

    out = desispec_io.read_spectra(str(out_path))
    in_spec = desispec_io.read_spectra(str(in_path))

    # Schema preserved: bands, wave grids, fibermap order/content identical.
    assert out.bands == in_spec.bands
    for cam in out.bands:
        np.testing.assert_array_equal(out.wave[cam], in_spec.wave[cam])
        assert out.flux[cam].shape == in_spec.flux[cam].shape
        # ivar + mask byte-preserved.
        np.testing.assert_array_equal(out.ivar[cam], in_spec.ivar[cam])
        np.testing.assert_array_equal(out.mask[cam], in_spec.mask[cam])
    np.testing.assert_array_equal(
        np.asarray(out.fibermap["TARGETID"]),
        np.asarray(in_spec.fibermap["TARGETID"]),
    )


def test_inject_into_coadd_injected_fiber_shows_trough(tmp_path):
    """The injected fiber's b-camera flux develops a damped trough at
    (1+z)·1215.67; un-injected fibers are byte-unchanged."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import inject_into_coadd

    tids = np.array([100, 101, 102], dtype=np.int64)
    ra = np.array([10.0, 10.1, 10.2])
    dec = np.array([5.0, 5.0, 5.0])
    spec = _make_spectra(tids, ra, dec, flux_level=2.0)

    in_path = tmp_path / "coadd-in.fits"
    out_path = tmp_path / "coadd-out.fits"
    desispec_io.write_spectra(str(in_path), spec)

    z = 2.8
    injections = [{"target_id": 100, "logN_true": 21.5, "z_true": z, "num_lines": 3}]
    inject_into_coadd(str(in_path), str(out_path), injections, num_lines=3)

    out = desispec_io.read_spectra(str(out_path))
    in_spec = desispec_io.read_spectra(str(in_path))

    fm = list(np.asarray(out.fibermap["TARGETID"]))
    idx100 = fm.index(100)

    # Injected fiber: deep trough at (1+z)*Lya in the b camera.
    wb = out.wave["b"]
    lam0 = (1 + z) * LYA
    icen = int(np.argmin(np.abs(wb - lam0)))
    cont = in_spec.flux["b"][idx100]
    trans = out.flux["b"][idx100][icen] / cont[icen]
    assert trans < 0.1, "injected fiber should show a damped (near-black) trough"
    # Minimum located at the line.
    imin = int(np.argmin(out.flux["b"][idx100]))
    assert abs(wb[imin] - lam0) < 5.0

    # Un-injected fibers: byte-identical across ALL cameras.
    for tid in (101, 102):
        j = fm.index(tid)
        for cam in out.bands:
            np.testing.assert_array_equal(
                out.flux[cam][j], in_spec.flux[cam][j]
            )


def test_inject_into_coadd_multiplies_only_signal_not_ivar(tmp_path):
    """Injection multiplies the FLUX by transmission; ivar/mask untouched (native
    SNR preserved per the design)."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import inject_into_coadd
    from gpy_dla_detection.inject_absorber import inject_voigt

    tids = np.array([100, 101], dtype=np.int64)
    spec = _make_spectra(tids, [10.0, 10.1], [5.0, 5.0], flux_level=3.0)
    in_path = tmp_path / "in.fits"
    out_path = tmp_path / "out.fits"
    desispec_io.write_spectra(str(in_path), spec)

    z, logN = 2.7, 20.3
    inject_into_coadd(
        str(in_path),
        str(out_path),
        [{"target_id": 100, "logN_true": logN, "z_true": z, "num_lines": 3}],
        num_lines=3,
    )
    out = desispec_io.read_spectra(str(out_path))
    in_spec = desispec_io.read_spectra(str(in_path))
    j = list(np.asarray(out.fibermap["TARGETID"])).index(100)

    # ivar unchanged on the injected fiber.
    for cam in out.bands:
        np.testing.assert_array_equal(out.ivar[cam][j], in_spec.ivar[cam][j])

    # Flux equals inject_voigt applied to the b-camera continuum. DESI coadds
    # store flux as float32 on disk (wave float64), so compare at the float32
    # round-trip floor, not bit-exactly — the in-memory injection IS exact.
    wb = out.wave["b"]
    expected = inject_voigt(wb, in_spec.flux["b"][j], 10 ** logN, z, num_lines=3)
    np.testing.assert_allclose(
        out.flux["b"][j], expected, rtol=1e-6, atol=1e-6
    )


def test_inject_into_coadd_multiple_injections_same_fiber(tmp_path):
    """Two injected absorbers on one fiber (close pair) blend multiplicatively."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import inject_into_coadd

    tids = np.array([100, 101], dtype=np.int64)
    spec = _make_spectra(tids, [10.0, 10.1], [5.0, 5.0], flux_level=1.0)
    in_path = tmp_path / "in.fits"
    out_path = tmp_path / "out.fits"
    desispec_io.write_spectra(str(in_path), spec)

    injections = [
        {"target_id": 100, "logN_true": 20.8, "z_true": 2.80, "num_lines": 3},
        {"target_id": 100, "logN_true": 20.5, "z_true": 2.82, "num_lines": 3},
    ]
    inject_into_coadd(str(in_path), str(out_path), injections, num_lines=3)
    out = desispec_io.read_spectra(str(out_path))
    j = list(np.asarray(out.fibermap["TARGETID"])).index(100)
    # Two distinct troughs → at least two well-separated deep pixels.
    deep = np.where(out.flux["b"][j] < 0.5)[0]
    assert deep.size > 0
    assert deep.max() - deep.min() > 5  # spread over two troughs


# ===========================================================================
# 2b. M4 — round-trip on the grid the GP actually scores on
# ===========================================================================
#
# Finding M4 (2026-06-10 review): inject_voigt runs on the LINEAR 0.8 Å camera
# grids, but the GP scores on the RESAMPLED 0.8 Å brz grid (mock coadds have no
# RESOLUTION HDU → coadd_cameras fails → resample_spectra_lin_or_log builds a
# NEW common 0.8 Å linear grid). These tests inject → write → read → coadd_cameras
# → resample (the EXACT production path) and assert the recovered absorption
# trough matches dla_gp's OWN forward-model Voigt at the same (N, z) — to <1% in
# equivalent width across the whole campaign N_HI range — confirming no EW/N_HI
# bias from a grid mismatch.


def _ew_of_transmission(wave, T, mask):
    """Absorption equivalent width  ∫ (1 − T) dλ  over ``mask`` pixels (Å)."""
    dlam = np.gradient(wave)
    return float(np.sum((1.0 - T[mask]) * dlam[mask]))


def test_m4_roundtrip_ew_matches_gp_forward_model():
    """ROUND-TRIP (M4): inject on the camera grids, run the production
    coadd→resample to the 0.8 Å brz grid, and confirm the recovered trough's EW
    matches what ``dla_gp``'s own Voigt (``voigt_transmission`` on the brz grid)
    imprints — to <1% across log N_HI ∈ [17.5, 20.3] (LLS → DLA).

    This validates that injecting on the camera pitch survives the resample
    FAITHFULLY, so the GP recovers the SAME absorber with no grid-mismatch bias.
    The weakest LLS (17.5) is the worst case (unsaturated wings carry all the
    N_HI information); the saturated DLA core is trivially matched.
    """
    import copy

    _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import inject_into_coadd  # noqa: F401
    from gpy_dla_detection.inject_absorber import inject_voigt, voigt_transmission

    tids = np.array([100], dtype=np.int64)
    base = _make_resolved_spectra(tids, [10.0], [5.0], flux_level=1.0)

    # Clean reference on the brz grid (what the continuum resamples to).
    rs_clean = _production_resample(base)
    wbrz = rs_clean.wave["brz"]
    f_clean = rs_clean.flux["brz"][0]
    assert "brz" in rs_clean.bands
    # The grid the GP scores on is LINEAR 0.8 Å (not log-spaced).
    assert np.allclose(np.diff(wbrz), 0.8, atol=1e-6)

    z_dla = 2.8
    lam0 = (1 + z_dla) * LYA
    near = (wbrz > lam0 - 120) & (wbrz < lam0 + 120)

    worst = 0.0
    for logN in (17.5, 18.0, 18.5, 19.0, 20.3):
        nhi = 10.0 ** logN
        # Inject on the CAMERA grids exactly as inject_into_coadd does.
        spec_inj = copy.deepcopy(base)
        for cam in spec_inj.bands:
            w = np.asarray(spec_inj.wave[cam], dtype=np.float64)
            f = np.asarray(spec_inj.flux[cam][0], dtype=np.float64)
            spec_inj.flux[cam][0] = inject_voigt(w, f, nhi, z_dla, num_lines=3)
        rs = _production_resample(spec_inj)
        f_inj = rs.flux["brz"][0]

        with np.errstate(divide="ignore", invalid="ignore"):
            T_rec = f_inj / f_clean
        T_mod = voigt_transmission(wbrz, nhi, z_dla, 3)  # the GP forward model
        m = near & np.isfinite(T_rec) & (np.abs(f_clean) > 1e-3)

        ew_rec = _ew_of_transmission(wbrz, T_rec, m)
        ew_mod = _ew_of_transmission(wbrz, T_mod, near)
        rel = abs(ew_rec - ew_mod) / ew_mod
        worst = max(worst, rel)
        assert rel < 0.01, (
            f"log N_HI={logN}: recovered EW {ew_rec:.4f} Å vs GP forward-model "
            f"EW {ew_mod:.4f} Å differ by {rel*100:.2f}% (> 1% → grid mismatch)"
        )
    # Tightest regime (weak LLS wings) is the bound that matters for M4.
    assert worst < 0.01


def test_m4_roundtrip_trough_centred_on_line():
    """The recovered trough centre lands on (1+z)·Lyα to within one brz pixel —
    i.e. the camera-grid injection + resample introduces NO wavelength shift that
    would bias z_DLA (and hence N_HI via the line-centre constraint)."""
    import copy

    _require_desispec()
    _require_c_voigt()
    from gpy_dla_detection.inject_absorber import inject_voigt

    base = _make_resolved_spectra(np.array([100], dtype=np.int64), [10.0], [5.0])
    z_dla, logN = 2.8, 19.5
    spec_inj = copy.deepcopy(base)
    for cam in spec_inj.bands:
        w = np.asarray(spec_inj.wave[cam], dtype=np.float64)
        f = np.asarray(spec_inj.flux[cam][0], dtype=np.float64)
        spec_inj.flux[cam][0] = inject_voigt(w, f, 10.0 ** logN, z_dla, num_lines=3)
    rs = _production_resample(spec_inj)
    wbrz = rs.wave["brz"]
    imin = int(np.argmin(rs.flux["brz"][0]))
    lam0 = (1 + z_dla) * LYA
    assert abs(wbrz[imin] - lam0) < 0.8, "trough shifted >1 brz pixel off the line"


@pytest.mark.skipif(
    not os.path.exists(
        "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
        "mock-0/loa-0/spectra-16/19/1960/spectra-16-1960.fits"
    ),
    reason="real 2LPT mock coadd (GL only) not present",
)
def test_m4_roundtrip_real_mock_coadd():
    """M4 round-trip against a REAL 2LPT mock coadd + its truth-16 RESOLUTION
    (GL-only; skipped elsewhere). Uses the genuine resolution matrix and camera
    grids, so this is the production-faithful check; EW match must stay <1% for
    the weakest LLS (the tightest tolerance)."""
    import copy

    _require_desispec()
    _require_c_voigt()
    import fitsio
    import desispec.io
    from gpy_dla_detection.inject_absorber import inject_voigt, voigt_transmission

    P = (
        "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/"
        "mock-0/loa-0/spectra-16/19/1960/spectra-16-1960.fits"
    )
    TRUTH = P.replace("spectra-16-", "truth-16-")

    def prod_resample(spec):
        from desispec.coaddition import (
            coadd_cameras,
            resample_spectra_lin_or_log,
        )

        spec = copy.deepcopy(spec)
        spec.resolution_data = {}
        for cam in ["b", "r", "z"]:
            tres = fitsio.read(TRUTH, ext=f"{cam}_RESOLUTION")
            n = spec.flux[cam].shape[0]
            td = np.empty([n, tres.shape[0], spec.flux[cam].shape[1]], dtype=float)
            for i in range(n):
                td[i] = tres
            spec.resolution_data[cam] = td
        spec = resample_spectra_lin_or_log(
            spec,
            linear_step=0.8,
            wave_min=np.min(spec.wave["b"]),
            wave_max=np.max(spec.wave["z"]),
            fast=True,
        )
        return coadd_cameras(spec)

    base = desispec.io.read_spectra(
        P, skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"]
    )
    row = 0
    rs_clean = prod_resample(base)
    wbrz = rs_clean.wave["brz"]
    f_clean = rs_clean.flux["brz"][row]

    z_dla, logN = 2.8, 17.5  # weakest LLS — tightest grid-fidelity test
    nhi = 10.0 ** logN
    spec_inj = copy.deepcopy(base)
    for cam in spec_inj.bands:
        w = np.asarray(spec_inj.wave[cam], dtype=np.float64)
        f = np.asarray(spec_inj.flux[cam][row], dtype=np.float64)
        spec_inj.flux[cam][row] = inject_voigt(w, f, nhi, z_dla, num_lines=3)
    rs = prod_resample(spec_inj)
    f_inj = rs.flux["brz"][row]

    lam0 = (1 + z_dla) * LYA
    near = (wbrz > lam0 - 120) & (wbrz < lam0 + 120)
    with np.errstate(divide="ignore", invalid="ignore"):
        T_rec = f_inj / f_clean
    T_mod = voigt_transmission(wbrz, nhi, z_dla, 3)
    m = near & np.isfinite(T_rec) & (np.abs(f_clean) > 1e-3)
    ew_rec = _ew_of_transmission(wbrz, T_rec, m)
    ew_mod = _ew_of_transmission(wbrz, T_mod, near)
    assert abs(ew_rec - ew_mod) / ew_mod < 0.01


# ===========================================================================
# 2c. Injection minor — pre-existing forest-blend guard
# ===========================================================================


def test_inject_into_coadd_flags_forest_blend(tmp_path):
    """A guard flags injections that land where the PRE-injection forest flux at
    the trough centre is already near-zero (a strong pre-existing forest/blend),
    so an LLS that imprints nothing distinguishable from the blend isn't mistaken
    for the injected response. The flag is reported, not silently dropped."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import inject_into_coadd

    tids = np.array([100, 101], dtype=np.int64)
    spec = _make_spectra(tids, [10.0, 10.1], [5.0, 5.0], flux_level=1.0)

    # Pre-existing near-zero forest at (1+z)*Lyα on fiber 100's b camera.
    z = 2.8
    lam0 = (1 + z) * LYA
    j100 = list(np.asarray(spec.fibermap["TARGETID"])).index(100)
    wb = spec.wave["b"]
    blendmask = np.abs(wb - lam0) < 6.0  # a few-Å absorbed patch
    spec.flux["b"][j100][blendmask] = 1e-4

    in_path = tmp_path / "in.fits"
    out_path = tmp_path / "out.fits"
    desispec_io.write_spectra(str(in_path), spec)

    blend_report = []
    inject_into_coadd(
        str(in_path),
        str(out_path),
        [
            {"target_id": 100, "logN_true": 18.0, "z_true": z, "num_lines": 3},
            {"target_id": 101, "logN_true": 18.0, "z_true": z, "num_lines": 3},
        ],
        num_lines=3,
        blend_report=blend_report,
    )

    by_tid = {r["target_id"]: r for r in blend_report}
    assert by_tid[100]["forest_blend"] is True
    assert by_tid[100]["forest_flux_frac"] < 0.1
    # Fiber 101 had clean (flat) continuum at the trough → not flagged.
    assert by_tid[101]["forest_blend"] is False


# ===========================================================================
# 3. Campaign writer + injectable-tree layout
# ===========================================================================


def _fake_manifest(rows):
    """Build a manifest Table matching campaign_grid's schema from row dicts."""
    Table = pytest.importorskip("astropy.table").Table
    cols = {
        "inj_id": [r["inj_id"] for r in rows],
        "campaign": [r["campaign"] for r in rows],
        "method": [r["method"] for r in rows],
        "target_id": np.array([r["target_id"] for r in rows], dtype=np.int64),
        "healpix": np.array([r["healpix"] for r in rows], dtype=np.int64),
        "z_qso": [r["z_qso"] for r in rows],
        "snr_bin": [r["snr_bin"] for r in rows],
        "native_snr": [r["native_snr"] for r in rows],
        "logN_true": [r["logN_true"] for r in rows],
        "z_true": [r["z_true"] for r in rows],
        "num_lines": np.array([r["num_lines"] for r in rows], dtype=np.int32),
    }
    return Table(cols)


def test_write_campaign_builds_injectable_tree(tmp_path):
    """write_campaign groups manifest rows by healpix, injects into each source
    coadd, and writes a GP-driver-compatible tree:
        {out_root}/spectra-16/{hp//100}/{hp}/spectra-16-{hp}.fits
    plus the companion truth-16 file (symlink/copy) so resample has resolution
    data, and a per-injection truth manifest.
    """
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import write_campaign

    # Build a fake mock dir with TWO healpix, both supplying source coadds.
    mockdir = tmp_path / "mock"
    hp_a, hp_b = 1960, 1995
    tid_map = {hp_a: [100, 101], hp_b: [200, 201]}
    for hp_id, tids in tid_map.items():
        d = mockdir / "spectra-16" / str(hp_id // 100) / str(hp_id)
        d.mkdir(parents=True)
        spec = _make_spectra(
            np.array(tids, dtype=np.int64),
            [10.0 + i for i in range(len(tids))],
            [5.0] * len(tids),
            flux_level=1.0,
        )
        desispec_io.write_spectra(str(d / f"spectra-16-{hp_id}.fits"), spec)
        # a stand-in truth file (campaign writer must carry it over)
        (d / f"truth-16-{hp_id}.fits").write_bytes(b"TRUTHSTUB")

    manifest = _fake_manifest(
        [
            dict(inj_id=0, campaign="A", method="coadd", target_id=100,
                 healpix=hp_a, z_qso=2.5, snr_bin=0, native_snr=1.0,
                 logN_true=21.0, z_true=2.3, num_lines=3),
            dict(inj_id=1, campaign="A", method="coadd", target_id=201,
                 healpix=hp_b, z_qso=3.3, snr_bin=2, native_snr=8.0,
                 logN_true=19.0, z_true=3.0, num_lines=3),
        ]
    )

    out_root = tmp_path / "campaign"
    truth_path = write_campaign(
        manifest, None, out_root=str(out_root), mockdir=str(mockdir), num_lines=3
    )

    # Injectable tree exists in the GP driver's exact layout.
    for hp_id in (hp_a, hp_b):
        f = (
            out_root / "spectra-16" / str(hp_id // 100) / str(hp_id)
            / f"spectra-16-{hp_id}.fits"
        )
        assert f.exists(), f"missing injected coadd for healpix {hp_id}"
        # truth-16 companion carried over (needed for resolution during resample).
        t = (
            out_root / "spectra-16" / str(hp_id // 100) / str(hp_id)
            / f"truth-16-{hp_id}.fits"
        )
        assert t.exists(), f"missing truth-16 companion for healpix {hp_id}"

    # Injection landed only on the manifest fibers.
    a = desispec_io.read_spectra(
        str(out_root / "spectra-16" / str(hp_a // 100) / str(hp_a)
            / f"spectra-16-{hp_a}.fits")
    )
    fm = list(np.asarray(a.fibermap["TARGETID"]))
    j100 = fm.index(100)
    j101 = fm.index(101)
    # 100 injected (trough), 101 untouched (flat at 1.0).
    assert a.flux["b"][j100].min() < 0.5
    np.testing.assert_allclose(a.flux["b"][j101], 1.0, atol=1e-9)

    # Truth manifest written and re-readable, one row per injection.
    assert os.path.exists(truth_path)
    from astropy.table import Table

    tman = Table.read(truth_path)
    assert len(tman) == 2
    assert set(np.asarray(tman["target_id"]).tolist()) == {100, 201}
    for col in ("inj_id", "target_id", "healpix", "logN_true", "z_true"):
        assert col in tman.colnames
    # Forest-blend guard columns are surfaced into the truth manifest (per M3
    # minor): flat continua → not flagged, fraction ≈ 1.
    for col in ("forest_blend", "forest_flux_frac"):
        assert col in tman.colnames, f"truth manifest missing {col}"
    assert not np.any(np.asarray(tman["forest_blend"]).astype(bool))


def test_write_campaign_injects_both_absorbers_of_a_close_pair(tmp_path):
    """Campaign-B close-pair rows carry a SECOND absorber (logN_true2/z_true2); the
    manifest keeps ONE row per sightline (validate_manifest enforces target_id
    uniqueness), so write_campaign must inject BOTH Voigt absorbers into that single
    spectrum — not just the first.  Regression for the review finding that the
    second absorber was silently dropped (Campaign B injected 1 trough vs a 2-trough
    truth)."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import write_campaign

    mockdir = tmp_path / "mock"
    hp, tid = 1960, 300
    d = mockdir / "spectra-16" / str(hp // 100) / str(hp)
    d.mkdir(parents=True)
    spec = _make_spectra(np.array([tid, 301], np.int64), [10.0, 11.0], [5.0, 5.0],
                         flux_level=1.0)
    desispec_io.write_spectra(str(d / f"spectra-16-{hp}.fits"), spec)
    (d / f"truth-16-{hp}.fits").write_bytes(b"STUB")

    z1, z2 = 2.40, 2.46  # both Lyα land in the b camera (3600-5800 Å)
    manifest = [dict(inj_id=0, campaign="B", method="coadd", target_id=tid,
                     healpix=hp, z_qso=2.9, snr_bin=0, native_snr=1.0,
                     logN_true=20.8, z_true=z1, num_lines=3,
                     logN_true2=20.6, z_true2=z2, dv_kms=5000.0)]
    out_root = tmp_path / "camp"
    write_campaign(manifest, None, out_root=str(out_root), mockdir=str(mockdir),
                   num_lines=3)

    a = desispec_io.read_spectra(
        str(out_root / "spectra-16" / str(hp // 100) / str(hp) / f"spectra-16-{hp}.fits")
    )
    fm = list(np.asarray(a.fibermap["TARGETID"]))
    j = fm.index(tid)
    w = a.wave["b"]
    f = a.flux["b"][j]
    for zc in (z1, z2):
        c = int(np.argmin(np.abs(w - (1.0 + zc) * LYA)))
        assert f[c] < 0.5, f"close-pair absorber at z={zc} was not injected"
    # the other fiber is untouched
    np.testing.assert_allclose(a.flux["b"][fm.index(301)], 1.0, atol=1e-9)


def test_write_campaign_leaves_control_fibers_clean_not_nan(tmp_path):
    """CONTROL rows (logN_true=NaN) must NOT be injected: 10**NaN=NaN would blank the
    control fiber to all-NaN, the GP would crash on it, and b_FP would collapse to a
    FAKE zero.  write_campaign must leave the control fiber as the clean source flux."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import write_campaign

    mockdir = tmp_path / "mock"
    hp = 1960
    d = mockdir / "spectra-16" / str(hp // 100) / str(hp)
    d.mkdir(parents=True)
    # fiber 300 = injection target, 301 = control (no injection)
    spec = _make_spectra(np.array([300, 301], np.int64), [10.0, 11.0], [5.0, 5.0],
                         flux_level=1.0)
    desispec_io.write_spectra(str(d / f"spectra-16-{hp}.fits"), spec)
    (d / f"truth-16-{hp}.fits").write_bytes(b"STUB")

    manifest = [
        dict(inj_id=0, campaign="A", method="coadd", target_id=300, healpix=hp,
             z_qso=2.9, snr_bin=0, native_snr=1.0, logN_true=20.5, z_true=2.40,
             num_lines=3, control=False, zqso_bin=-1),
        dict(inj_id=1, campaign="A", method="coadd", target_id=301, healpix=hp,
             z_qso=2.9, snr_bin=0, native_snr=1.0, logN_true=float("nan"),
             z_true=float("nan"), num_lines=3, control=True, zqso_bin=-1),
    ]
    out_root = tmp_path / "camp"
    write_campaign(manifest, None, out_root=str(out_root), mockdir=str(mockdir),
                   num_lines=3)
    a = desispec_io.read_spectra(
        str(out_root / "spectra-16" / str(hp // 100) / str(hp) / f"spectra-16-{hp}.fits"))
    fm = list(np.asarray(a.fibermap["TARGETID"]))
    # control fiber 301 stays clean/finite (NOT all-NaN), == flat source flux
    ctrl_flux = a.flux["b"][fm.index(301)]
    assert np.all(np.isfinite(ctrl_flux)), "control fiber was NaN-poisoned by injection"
    np.testing.assert_allclose(ctrl_flux, 1.0, atol=1e-9)
    # injected fiber 300 DID get the trough
    assert a.flux["b"][fm.index(300)].min() < 0.5


def test_write_campaign_truth_manifest_heterogeneous_pair_and_control_rows(tmp_path):
    """A Campaign-B manifest mixes pair rows (carrying logN_true2/z_true2/dv_kms) with
    control rows that LACK those keys.  _write_truth_manifest must take the UNION of
    columns and fill missing values (NaN), not index every row by the first row's keys
    (which crashed with KeyError: 'logN_true2')."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import write_campaign

    mockdir = tmp_path / "mock"
    hp = 1960
    d = mockdir / "spectra-16" / str(hp // 100) / str(hp)
    d.mkdir(parents=True)
    spec = _make_spectra(np.array([300, 301], np.int64), [10.0, 11.0], [5.0, 5.0],
                         flux_level=1.0)
    desispec_io.write_spectra(str(d / f"spectra-16-{hp}.fits"), spec)
    (d / f"truth-16-{hp}.fits").write_bytes(b"STUB")

    manifest = [
        dict(inj_id=0, campaign="B", method="coadd", target_id=300, healpix=hp,
             z_qso=2.9, snr_bin=0, native_snr=1.0, logN_true=20.6, z_true=2.40,
             num_lines=3, control=False, zqso_bin=1,
             logN_true2=20.2, z_true2=2.43, dv_kms=400.0, _dlogN=-0.4),
        dict(inj_id=1, campaign="B", method="coadd", target_id=301, healpix=hp,
             z_qso=2.9, snr_bin=0, native_snr=1.0, logN_true=float("nan"),
             z_true=float("nan"), num_lines=3, control=True, zqso_bin=1),  # no pair keys
    ]
    out_root = tmp_path / "camp"
    truth_path = write_campaign(manifest, None, out_root=str(out_root), mockdir=str(mockdir),
                                num_lines=3)
    from astropy.table import Table
    tman = Table.read(truth_path)
    assert len(tman) == 2
    assert "logN_true2" in tman.colnames          # union column present
    # the control row's pair fields are NaN-filled, not a crash
    ctrl = tman[np.asarray(tman["control"]).astype(bool)]
    assert len(ctrl) == 1 and not np.isfinite(ctrl["logN_true2"][0])


def test_write_campaign_truth_manifest_records_blend_flag(tmp_path):
    """write_campaign records the forest-blend flag per injection in the truth
    manifest: a sightline pre-absorbed to near-zero at the trough centre is
    flagged ``forest_blend=True`` (and ``forest_flux_frac`` small), so M4 can
    exclude/annotate it rather than mistaking the blend for the LLS response."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    from injection.coadd_injection import write_campaign

    mockdir = tmp_path / "mock"
    hp_id = 1960
    d = mockdir / "spectra-16" / str(hp_id // 100) / str(hp_id)
    d.mkdir(parents=True)
    spec = _make_spectra(
        np.array([100, 101], dtype=np.int64), [10.0, 10.1], [5.0, 5.0],
        flux_level=1.0,
    )
    # Pre-existing near-zero forest at the trough centre of fiber 100.
    z = 2.8
    lam0 = (1 + z) * LYA
    wb = spec.wave["b"]
    spec.flux["b"][0][np.abs(wb - lam0) < 6.0] = 1e-4
    desispec_io.write_spectra(str(d / f"spectra-16-{hp_id}.fits"), spec)
    (d / f"truth-16-{hp_id}.fits").write_bytes(b"TRUTHSTUB")

    manifest = _fake_manifest(
        [
            dict(inj_id=0, campaign="A", method="coadd", target_id=100,
                 healpix=hp_id, z_qso=3.0, snr_bin=0, native_snr=2.0,
                 logN_true=18.0, z_true=z, num_lines=3),
            dict(inj_id=1, campaign="A", method="coadd", target_id=101,
                 healpix=hp_id, z_qso=3.0, snr_bin=0, native_snr=2.0,
                 logN_true=18.0, z_true=z, num_lines=3),
        ]
    )

    out_root = tmp_path / "campaign"
    truth_path = write_campaign(
        manifest, None, out_root=str(out_root), mockdir=str(mockdir), num_lines=3
    )
    from astropy.table import Table

    tman = Table.read(truth_path)
    flag = {int(t): bool(b) for t, b in zip(tman["target_id"], tman["forest_blend"])}
    assert flag[100] is True   # pre-existing blend
    assert flag[101] is False  # clean continuum


def test_inject_into_coadd_noise_preserving_keeps_ivar_mask_and_is_T_times_F(tmp_path):
    """R-041 (2026-08-28): method="variance_preserving" leaves ivar/mask untouched, the
    deterministic part of the injected flux is exactly T * F, and the mean-flux-only
    rescaling (R-041C) changes the forest SIGNAL without touching the noise arrays."""
    desispec_io = _require_desispec()
    _require_c_voigt()
    pytest.importorskip("scipy")
    from injection.coadd_injection import inject_into_coadd
    from injection.noise_preserving import transmission

    tids = np.array([100, 101], dtype=np.int64)
    spec = _make_spectra(tids, [10.0, 10.1], [5.0, 5.0], flux_level=3.0)
    rng = np.random.default_rng(0)
    for cam in spec.bands:
        spec.flux[cam] = spec.flux[cam] + rng.standard_normal(spec.flux[cam].shape) * 0.5
        spec.mask[cam][0, 50:60] = 1
    in_path = tmp_path / "in.fits"; out_path = tmp_path / "out.fits"; out2 = tmp_path / "out2.fits"
    desispec_io.write_spectra(str(in_path), spec)
    z, logN, zq = 2.7, 21.0, 3.2
    rec = [{"target_id": 100, "logN_true": logN, "z_true": z, "num_lines": 3, "z_qso": zq}]
    inject_into_coadd(str(in_path), str(out_path), rec, num_lines=3, method="variance_preserving")
    out = desispec_io.read_spectra(str(out_path)); inp = desispec_io.read_spectra(str(in_path))
    j = list(np.asarray(out.fibermap["TARGETID"])).index(100)
    for cam in out.bands:
        np.testing.assert_array_equal(out.ivar[cam][j], inp.ivar[cam][j])
        np.testing.assert_array_equal(out.mask[cam][j], inp.mask[cam][j])
        np.testing.assert_array_equal(out.flux[cam][1], inp.flux[cam][1])          # other fiber untouched
        w = out.wave[cam]; T = transmission(w, [{"nhi": 10 ** logN, "z_dla": z, "num_lines": 3}])
        f_in = inp.flux[cam][j].astype(float); f_out = out.flux[cam][j].astype(float)
        outside = np.abs(T - 1) < 1e-12
        np.testing.assert_allclose(f_out[outside], f_in[outside], rtol=1e-6, atol=1e-6)
        core = T < 0.01
        if core.any():
            # noiseless-trough check fails for the old operation; the new one keeps ~ivar noise
            assert np.std(f_out[core] * np.sqrt(inp.ivar[cam][j][core])) > 0.5
    # mean-flux-only rescale with NO absorber: forest signal darkened, red side and noise arrays untouched
    inject_into_coadd(str(in_path), str(out2), [{"target_id": 100, "logN_true": None, "z_true": None, "z_qso": zq}],
                      num_lines=3, method="variance_preserving",
                      meanflux={"fiducial": "finder_fiducial", "model": "finder_fiducial", "delta_z": 1.0})
    o2 = desispec_io.read_spectra(str(out2))
    wb = o2.wave["b"]; fb_in = inp.flux["b"][j].astype(float); fb_out = o2.flux["b"][j].astype(float)
    forest = wb < 1215.67 * (1 + zq) * 0.97
    assert np.mean(fb_out[forest]) < 0.9 * np.mean(fb_in[forest])
    wz = o2.wave["z"]
    np.testing.assert_allclose(o2.flux["z"][j], inp.flux["z"][j], rtol=1e-6, atol=1e-6)
    for cam in o2.bands:
        np.testing.assert_array_equal(o2.ivar[cam][j], inp.ivar[cam][j])
