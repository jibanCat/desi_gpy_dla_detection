"""
tests/test_gen_wall1_inject.py
==============================
Dedicated unit tests for ``injection/gen_wall1_inject.py`` — the WALL-1
FULL-INJECTION truth-catalog generator (loa-0 substrate).  Previously ZERO
tests.

The two load-bearing helpers tested here both feed the n_true^tilt truth that
the slope-closure / HBI reads back.  A schema or join error in either silently
biases the DLA dN/dX / Omega headline:

  * ``build_loa0_clean_table(mockdir, snr_cut)`` — reads ``zcat.fits`` +
    ``snr_cat.fits`` from a mockdir, ang2pix's the (RA, DEC) into HEALPIX, does a
    TARGETID left-join (sorted-copy + searchsorted, with a ``found`` mask) of the
    snr_cat columns onto the zcat rows, NaN-fills snr columns for TIDs missing
    from snr_cat, and finally keeps only ``SNR_REDSIDE > snr_cut`` (finite) rows.
  * ``write_injected_truth(manifest_rows, out_root)`` — writes the injected-truth
    FITS in the hcd_truth_cat schema the HBI consumes:
    ``NHI, Z, TARGETID, DLAID, SNR`` (ext=1).  ``DLAID = TARGETID * 1000`` (slot
    0, the real hcd_truth_cat 3-digit-slot convention).

These are unit-testable in isolation on tiny SYNTHETIC FITS fixtures (astropy
Table -> FITS in a tmp dir); no real catalog, no inference, no desispec.

``main()`` is NOT unit-tested: it requires the real loa-0 mockdir + the
``coadd_injection.write_campaign`` Voigt-injection I/O (desispec) and the loa-124
truth catalog.  A module-level import/smoke + symbol-presence check stands in for
it (see ``test_module_imports_and_exposes_helpers``).
"""
import os
import tempfile

import numpy as np
import pytest
from astropy.table import Table

healpy = pytest.importorskip("healpy")
import healpy as hp  # noqa: E402

from injection import gen_wall1_inject as g  # noqa: E402


_NSIDE = 16  # matches gen_wall1_inject._NSIDE (the DESI healpix nside)


# --------------------------------------------------------------------------- #
# fixtures — tiny synthetic mockdirs / manifests built entirely in the test
# --------------------------------------------------------------------------- #
def _write_mockdir(tmp_path, zcat, snr_cat):
    """Write a minimal synthetic mockdir (zcat.fits + snr_cat.fits) and return it."""
    d = str(tmp_path)
    Table(zcat).write(os.path.join(d, "zcat.fits"), overwrite=True)
    Table(snr_cat).write(os.path.join(d, "snr_cat.fits"), overwrite=True)
    return d


def _expected_healpix(ra, dec):
    return hp.ang2pix(_NSIDE, ra, dec, nest=True, lonlat=True).astype(np.int64)


# --------------------------------------------------------------------------- #
# module smoke (stands in for the un-unit-testable main())
# --------------------------------------------------------------------------- #
def test_module_imports_and_exposes_helpers():
    # The module imports cleanly (incl. its top-level ``coadd_injection`` /
    # ``campaign_grid`` deps) and exposes the two helpers + the DLAID/_NSIDE
    # conventions the tests below pin.  main() itself needs real loa-0 paths +
    # desispec coadd I/O, so it is exercised only at this smoke level.
    assert callable(g.build_loa0_clean_table)
    assert callable(g.write_injected_truth)
    assert g._NSIDE == _NSIDE
    assert callable(g.main)


# =========================================================================== #
# build_loa0_clean_table
# =========================================================================== #
def test_clean_table_columns_and_basic_alignment(tmp_path):
    """The clean table emits the documented column set and aligns each TID's
    Z / RA / DEC / HEALPIX + the joined snr columns to the right zcat row."""
    zcat = {
        "TARGETID": np.array([100, 200, 300], dtype=np.int64),
        "TARGET_RA": np.array([10.0, 20.0, 30.0]),
        "TARGET_DEC": np.array([5.0, -5.0, 15.0]),
        "Z": np.array([2.5, 3.0, 2.8]),
    }
    snr_cat = {
        "TARGETID": np.array([100, 200, 300], dtype=np.int64),
        "SNR_REDSIDE": np.array([3.0, 5.0, 9.0]),
        "SNR_OTHER": np.array([7.0, 8.0, 9.5]),
    }
    d = _write_mockdir(tmp_path, zcat, snr_cat)
    out = g.build_loa0_clean_table(d, snr_cut=2.0)

    # documented column set: TARGETID, Z, TARGET_RA, TARGET_DEC, HEALPIX + snr cols
    assert set(out.colnames) == {
        "TARGETID", "Z", "TARGET_RA", "TARGET_DEC", "HEALPIX",
        "SNR_REDSIDE", "SNR_OTHER",
    }
    # all three pass the SNR cut, so all three survive, in input (zcat) order
    assert list(np.asarray(out["TARGETID"])) == [100, 200, 300]
    # per-row values are exactly the zcat values (hand-checked)
    np.testing.assert_array_equal(np.asarray(out["Z"]), [2.5, 3.0, 2.8])
    np.testing.assert_array_equal(np.asarray(out["TARGET_RA"]), [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(np.asarray(out["TARGET_DEC"]), [5.0, -5.0, 15.0])
    # HEALPIX is ang2pix(nside=16, nest=True, lonlat=True) of (RA, DEC)
    np.testing.assert_array_equal(
        np.asarray(out["HEALPIX"]),
        _expected_healpix(np.array([10.0, 20.0, 30.0]),
                          np.array([5.0, -5.0, 15.0])),
    )
    # the joined snr columns are aligned to the matching TID
    np.testing.assert_array_equal(np.asarray(out["SNR_REDSIDE"]), [3.0, 5.0, 9.0])
    np.testing.assert_array_equal(np.asarray(out["SNR_OTHER"]), [7.0, 8.0, 9.5])


def test_clean_table_snr_cut_drops_below_threshold(tmp_path):
    """A sightline whose SNR_REDSIDE <= snr_cut is dropped; one strictly above is
    kept.  Verifies the cut is strict (> snr_cut) and on SNR_REDSIDE."""
    zcat = {
        "TARGETID": np.array([100, 200, 300], dtype=np.int64),
        "TARGET_RA": np.array([10.0, 20.0, 30.0]),
        "TARGET_DEC": np.array([5.0, -5.0, 15.0]),
        "Z": np.array([2.5, 3.0, 2.8]),
    }
    snr_cat = {
        "TARGETID": np.array([100, 200, 300], dtype=np.int64),
        # TID100 below cut (1.0), TID200 EXACTLY at cut (2.0 -> dropped, strict >),
        # TID300 above (3.0 -> kept)
        "SNR_REDSIDE": np.array([1.0, 2.0, 3.0]),
    }
    d = _write_mockdir(tmp_path, zcat, snr_cat)
    out = g.build_loa0_clean_table(d, snr_cut=2.0)

    assert list(np.asarray(out["TARGETID"])) == [300]
    np.testing.assert_array_equal(np.asarray(out["SNR_REDSIDE"]), [3.0])

    # a different cut keeps both >2 and the boundary one when cut is lowered
    out2 = g.build_loa0_clean_table(d, snr_cut=1.0)
    assert list(np.asarray(out2["TARGETID"])) == [200, 300]


def test_clean_table_searchsorted_alignment_with_unsorted_inputs(tmp_path):
    """The TARGETID left-join must NOT assume zcat or snr_cat is sorted.

    Both tables are written in DELIBERATELY UNSORTED TARGETID order, and the snr
    values are chosen so that any sorted-assumption / mis-alignment bug would
    move a value onto the wrong sightline.  We assert each surviving TID carries
    its OWN snr_cat row's values (hand-checked), proving the sorted-copy +
    searchsorted machinery (lines 81-93) aligns correctly regardless of order.
    """
    # zcat TID order: 300, 100, 400, 200  (unsorted)
    zcat = {
        "TARGETID": np.array([300, 100, 400, 200], dtype=np.int64),
        "TARGET_RA": np.array([30.0, 10.0, 40.0, 20.0]),
        "TARGET_DEC": np.array([15.0, 5.0, -20.0, -5.0]),
        "Z": np.array([2.8, 2.5, 3.2, 3.0]),
    }
    # snr_cat TID order: 200, 400, 100, 300  (a DIFFERENT unsorted order)
    # SNR_REDSIDE encodes the TID (tid/100) so a misalignment is unmistakable.
    snr_cat = {
        "TARGETID": np.array([200, 400, 100, 300], dtype=np.int64),
        "SNR_REDSIDE": np.array([2.0, 4.0, 1.0, 3.0]),
        "SNR_TAG": np.array([222.0, 444.0, 111.0, 333.0]),
    }
    d = _write_mockdir(tmp_path, zcat, snr_cat)
    out = g.build_loa0_clean_table(d, snr_cut=2.0)

    # cut keeps SNR_REDSIDE>2: TID400 (4.0) and TID300 (3.0).  TID200 is exactly
    # at 2.0 (dropped, strict), TID100 is 1.0 (dropped).  Output stays in zcat
    # (input) order, so 300 precedes 400.
    assert list(np.asarray(out["TARGETID"])) == [300, 400]
    # each survivor carries ITS OWN snr_cat row's values (the alignment proof)
    by_tid = {int(t): (float(rs), float(tg))
              for t, rs, tg in zip(out["TARGETID"], out["SNR_REDSIDE"], out["SNR_TAG"])}
    assert by_tid[300] == (3.0, 333.0)
    assert by_tid[400] == (4.0, 444.0)
    # and Z / RA / DEC stay glued to the zcat row for each survivor
    zby = {int(t): (float(z), float(ra), float(dec))
           for t, z, ra, dec in zip(out["TARGETID"], out["Z"],
                                    out["TARGET_RA"], out["TARGET_DEC"])}
    assert zby[300] == (2.8, 30.0, 15.0)
    assert zby[400] == (3.2, 40.0, -20.0)


def test_clean_table_nanfill_for_missing_snr_then_dropped(tmp_path):
    """A zcat TID absent from snr_cat is NaN-filled in every snr column, and
    therefore fails the finite-SNR_REDSIDE cut and is dropped.

    This is the documented NaN-fill path (``joined = full(nan); joined[found] =
    ...``) AND the final ``np.isfinite(rs) & (rs > snr_cut)`` filter together.
    """
    zcat = {
        "TARGETID": np.array([100, 200, 300], dtype=np.int64),
        "TARGET_RA": np.array([10.0, 20.0, 30.0]),
        "TARGET_DEC": np.array([5.0, -5.0, 15.0]),
        "Z": np.array([2.5, 3.0, 2.8]),
    }
    # TID200 is MISSING from snr_cat -> NaN SNR -> dropped
    snr_cat = {
        "TARGETID": np.array([100, 300], dtype=np.int64),
        "SNR_REDSIDE": np.array([5.0, 6.0]),
        "SNR_OTHER": np.array([7.0, 8.0]),
    }
    d = _write_mockdir(tmp_path, zcat, snr_cat)
    out = g.build_loa0_clean_table(d, snr_cut=2.0)

    # only the two present-and-above-cut TIDs survive
    assert list(np.asarray(out["TARGETID"])) == [100, 300]
    np.testing.assert_array_equal(np.asarray(out["SNR_REDSIDE"]), [5.0, 6.0])
    np.testing.assert_array_equal(np.asarray(out["SNR_OTHER"]), [7.0, 8.0])


def test_clean_table_nanfill_preserved_when_missing_tid_passes_other_cols(tmp_path):
    """Directly observe the NaN-fill BEFORE the final cut removes it.

    We give the only missing TID a LOWER snr_cut sibling so we can see the table
    shape, then confirm that a TID missing from snr_cat would carry NaN — by
    checking it never appears in the output for ANY non-negative cut (a NaN can
    never satisfy ``rs > snr_cut``).  This isolates the NaN-fill semantics from
    the cut: a missing-SNR sightline is unconditionally excluded.
    """
    zcat = {
        "TARGETID": np.array([100, 200], dtype=np.int64),
        "TARGET_RA": np.array([10.0, 20.0]),
        "TARGET_DEC": np.array([5.0, -5.0]),
        "Z": np.array([2.5, 3.0]),
    }
    snr_cat = {  # TID200 missing
        "TARGETID": np.array([100], dtype=np.int64),
        "SNR_REDSIDE": np.array([5.0]),
    }
    d = _write_mockdir(tmp_path, zcat, snr_cat)
    for cut in (-1.0, 0.0, 2.0, 4.0):
        out = g.build_loa0_clean_table(d, snr_cut=cut)
        assert 200 not in set(int(t) for t in out["TARGETID"]), (
            f"missing-SNR TID 200 must never survive (NaN !> {cut})")


def test_clean_table_ra_dec_fallback_column_names(tmp_path):
    """``build_loa0_clean_table`` falls back to plain RA/DEC when TARGET_RA /
    TARGET_DEC are absent (line 66-67), and the HEALPIX is computed from them."""
    zcat = {
        "TARGETID": np.array([100, 200], dtype=np.int64),
        "RA": np.array([10.0, 20.0]),       # NOT TARGET_RA
        "DEC": np.array([5.0, -5.0]),       # NOT TARGET_DEC
        "Z": np.array([2.5, 3.0]),
    }
    snr_cat = {
        "TARGETID": np.array([100, 200], dtype=np.int64),
        "SNR_REDSIDE": np.array([5.0, 6.0]),
    }
    d = _write_mockdir(tmp_path, zcat, snr_cat)
    out = g.build_loa0_clean_table(d, snr_cut=2.0)
    assert list(np.asarray(out["TARGETID"])) == [100, 200]
    # the output still names the columns TARGET_RA/TARGET_DEC and uses the RA/DEC values
    np.testing.assert_array_equal(np.asarray(out["TARGET_RA"]), [10.0, 20.0])
    np.testing.assert_array_equal(np.asarray(out["TARGET_DEC"]), [5.0, -5.0])
    np.testing.assert_array_equal(
        np.asarray(out["HEALPIX"]),
        _expected_healpix(np.array([10.0, 20.0]), np.array([5.0, -5.0])),
    )


def test_clean_table_targetid_dtype_int64(tmp_path):
    """TARGETID must remain int64 (the join key + downstream qsocat ``isin`` key)."""
    zcat = {
        "TARGETID": np.array([100, 200], dtype=np.int64),
        "TARGET_RA": np.array([10.0, 20.0]),
        "TARGET_DEC": np.array([5.0, -5.0]),
        "Z": np.array([2.5, 3.0]),
    }
    snr_cat = {
        "TARGETID": np.array([100, 200], dtype=np.int64),
        "SNR_REDSIDE": np.array([5.0, 6.0]),
    }
    d = _write_mockdir(tmp_path, zcat, snr_cat)
    out = g.build_loa0_clean_table(d, snr_cut=2.0)
    assert np.asarray(out["TARGETID"]).dtype == np.int64
    assert np.asarray(out["HEALPIX"]).dtype == np.int64


# =========================================================================== #
# write_injected_truth
# =========================================================================== #
def _manifest_row(target_id, logN_true, z_true, native_snr):
    """A minimal manifest row carrying only the keys write_injected_truth reads."""
    return {
        "target_id": int(target_id),
        "logN_true": float(logN_true),
        "z_true": float(z_true),
        "native_snr": float(native_snr),
    }


def test_injected_truth_schema_matches_hbi_loader(tmp_path):
    """The written FITS carries EXACTLY the hcd_truth_cat columns the HBI reads
    back: NHI, Z, TARGETID, DLAID, SNR (ext=1).

    cddf_catalog_hbi.load_and_cut_catalog reads TARGETID + NHI + (Z->Z_DLA->
    Z_TRUTH) from cfg.truth_path ext=1; this file IS the n_true^tilt the closure
    compares against, so the column names are a hard contract.
    """
    manifest = [
        _manifest_row(100, 20.5, 2.5, 4.0),
        _manifest_row(200, 21.0, 3.0, 8.0),
    ]
    path = g.write_injected_truth(manifest, str(tmp_path))
    assert path == os.path.join(str(tmp_path), "injected_truth_cat.fits")
    assert os.path.exists(path)

    t = Table.read(path)
    # the exact column set the HBI truth loader expects
    assert set(t.colnames) == {"NHI", "Z", "TARGETID", "DLAID", "SNR"}
    # the three columns the HBI actually consumes are present & typed
    assert "TARGETID" in t.colnames and "NHI" in t.colnames and "Z" in t.colnames
    # int64 (FITS round-trips as big-endian '>i8'; check kind/itemsize, not ==)
    tid_dt = np.asarray(t["TARGETID"]).dtype
    assert tid_dt.kind == "i" and tid_dt.itemsize == 8


def test_injected_truth_values_roundtrip(tmp_path):
    """Every manifest row round-trips: NHI/Z/SNR/TARGETID values preserved, in
    manifest order."""
    manifest = [
        _manifest_row(100, 20.50, 2.50, 4.0),
        _manifest_row(200, 21.00, 3.00, 8.0),
        _manifest_row(300, 19.75, 2.80, 3.5),
    ]
    path = g.write_injected_truth(manifest, str(tmp_path))
    t = Table.read(path)

    np.testing.assert_array_equal(np.asarray(t["TARGETID"]), [100, 200, 300])
    np.testing.assert_allclose(np.asarray(t["NHI"]), [20.50, 21.00, 19.75])
    np.testing.assert_allclose(np.asarray(t["Z"]), [2.50, 3.00, 2.80])
    np.testing.assert_allclose(np.asarray(t["SNR"]), [4.0, 8.0, 3.5])


def test_injected_truth_dlaid_convention_slot0(tmp_path):
    """DLAID = TARGETID * 1000 (slot 0) for EVERY row — the real hcd_truth_cat
    3-digit-slot convention as IMPLEMENTED (line 112 always uses slot 0).

    NOTE on the slot suffix: the docstring mentions a per-absorber slot, but the
    code hard-codes slot 0 (``dlaid = tid * 1000``) for every row.  In the WALL-1
    pipeline this is unambiguous because the upstream tilted manifest is
    ONE-absorber-per-sightline (build_tilted_manifest + validate_manifest reject
    duplicate target_ids), so each TARGETID appears exactly once and DLAID is
    consequently unique-per-row (see the dedicated uniqueness test below).
    """
    manifest = [
        _manifest_row(100, 20.5, 2.5, 4.0),
        _manifest_row(200, 21.0, 3.0, 8.0),
        _manifest_row(1234567, 20.9, 3.2, 5.0),
    ]
    path = g.write_injected_truth(manifest, str(tmp_path))
    t = Table.read(path)
    np.testing.assert_array_equal(
        np.asarray(t["DLAID"]),
        np.asarray(t["TARGETID"], dtype=np.int64) * 1000,
    )
    # int64 (FITS round-trips as big-endian '>i8'; check kind/itemsize, not ==)
    dla_dt = np.asarray(t["DLAID"]).dtype
    assert dla_dt.kind == "i" and dla_dt.itemsize == 8
    # explicit hand-checked values
    assert list(np.asarray(t["DLAID"])) == [100000, 200000, 1234567000]


def test_injected_truth_dlaid_unique_per_sightline_wall1_manifest(tmp_path):
    """For a WALL-1 manifest (one absorber per TARGETID, which is what
    build_tilted_manifest + validate_manifest enforce), DLAID is UNIQUE per row
    — exactly the (TID, slot=0) join key the HBI uses to count absorbers.

    This is the load-bearing invariant: a non-unique DLAID over distinct truth
    rows would double-count or collide n_true^tilt absorbers.
    """
    manifest = [_manifest_row(tid, 20.0 + 0.1 * i, 2.5 + 0.05 * i, 4.0)
                for i, tid in enumerate([100, 200, 300, 400, 500])]
    path = g.write_injected_truth(manifest, str(tmp_path))
    t = Table.read(path)
    dlaids = np.asarray(t["DLAID"])
    assert len(set(dlaids.tolist())) == len(dlaids), "DLAID must be unique per row"
    # and the (TID, slot) decode is exact: DLAID // 1000 == TARGETID, slot 0
    np.testing.assert_array_equal(dlaids // 1000, np.asarray(t["TARGETID"]))
    np.testing.assert_array_equal(dlaids % 1000, np.zeros(len(dlaids), dtype=np.int64))


def test_injected_truth_overwrites_existing(tmp_path):
    """A second call with a different manifest overwrites cleanly (the driver
    re-runs an arm into the same out_root)."""
    g.write_injected_truth([_manifest_row(100, 20.5, 2.5, 4.0)], str(tmp_path))
    path = g.write_injected_truth(
        [_manifest_row(900, 21.5, 3.5, 9.0), _manifest_row(901, 20.1, 2.4, 2.5)],
        str(tmp_path))
    t = Table.read(path)
    assert list(np.asarray(t["TARGETID"])) == [900, 901]
    assert len(t) == 2


def test_injected_truth_empty_manifest(tmp_path):
    """An empty manifest writes a well-formed empty table with the full schema
    (defensive: the driver SystemExits on an empty manifest BEFORE this, but the
    helper must not crash and must keep the column contract)."""
    path = g.write_injected_truth([], str(tmp_path))
    t = Table.read(path)
    assert set(t.colnames) == {"NHI", "Z", "TARGETID", "DLAID", "SNR"}
    assert len(t) == 0
