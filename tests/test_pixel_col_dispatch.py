"""Tests for the --pixel_col {HPXPIXEL, UNIQPIX} dispatch path.

These cover the matterhorn UNIQPIX grouping/coadd-layout feature added across
stages S1-S3, and the parity contract that ``pixel_col='HPXPIXEL'`` (the
default) keeps the existing loa/mock behaviour byte-identical.

The tests use mocking + tiny synthetic data only -- they require fitsio, numpy
and astropy, but NOT real spectra and nothing from the desi environment beyond
those three packages plus ``desispec.io.findfile`` (a pure path constructor).
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
from astropy.table import Table

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_desi_dlagp():
    """Import the hyphenated entry-point module ``desi-DLAGP.py``."""
    path = os.path.join(REPO_ROOT, "desi-DLAGP.py")
    spec = importlib.util.spec_from_file_location("desi_DLAGP_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_balance():
    """Import ``tools/loa_balance_boundaries.py``."""
    path = os.path.join(REPO_ROOT, "tools", "loa_balance_boundaries.py")
    spec = importlib.util.spec_from_file_location("loa_balance_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_spec_counts():
    """Import ``tools/loa_hpx_spec_counts.py`` (the count-table generator)."""
    path = os.path.join(REPO_ROOT, "tools", "loa_hpx_spec_counts.py")
    spec = importlib.util.spec_from_file_location("loa_hpx_spec_counts_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# (1) read_catalog column set: UNIQPIX requested only when pixel_col=UNIQPIX
# --------------------------------------------------------------------------- #
def test_read_catalog_requests_uniqpix_only_when_selected(monkeypatch):
    desi = _load_desi_dlagp()

    captured = {}

    def fake_fitsio_read(qsocat, ext=1, columns=None):
        # Record exactly which columns desi-DLAGP asked for ...
        captured["columns"] = list(columns)
        # ... and return a synthetic table whose schema matches the request so
        # the downstream z-mask in read_catalog does not crash.
        n = 3
        data = {}
        for c in columns:
            if c == "Z":
                data[c] = np.array([2.5, 3.0, 3.5], dtype=float)  # all in z-range
            elif c in ("TARGETID", "HPXPIXEL", "UNIQPIX", "TILEID",
                       "PETAL_LOC", "NCIV_450", "ZWARN"):
                data[c] = np.arange(n, dtype=np.int64)
            elif c == "SPECTYPE":
                data[c] = np.array(["QSO"] * n)
            else:
                data[c] = np.zeros(n, dtype=float)
        return Table(data)

    # read_catalog calls fitsio.read via the module-level `fitsio` binding.
    monkeypatch.setattr(desi.fitsio, "read", fake_fitsio_read)

    # Default / explicit HPXPIXEL: UNIQPIX must NOT be requested.
    desi.read_catalog("dummy.fits", balmask=False, bytile=False,
                      pixel_col="HPXPIXEL")
    cols_hpx = captured["columns"]
    assert "HPXPIXEL" in cols_hpx
    assert "UNIQPIX" not in cols_hpx

    # UNIQPIX: the extra column must be appended to the request.
    desi.read_catalog("dummy.fits", balmask=False, bytile=False,
                      pixel_col="UNIQPIX")
    cols_uniq = captured["columns"]
    assert "UNIQPIX" in cols_uniq
    # Parity: UNIQPIX is purely additive -- every HPXPIXEL-default column stays.
    assert set(cols_hpx).issubset(set(cols_uniq))
    assert cols_uniq == cols_hpx + ["UNIQPIX"]


def test_read_catalog_uniqpix_appended_in_balmask_branch(monkeypatch):
    desi = _load_desi_dlagp()
    captured = {}

    def fake_fitsio_read(qsocat, ext=1, columns=None):
        captured["columns"] = list(columns)
        n = 2
        data = {}
        for c in columns:
            if c == "Z":
                data[c] = np.array([2.5, 3.0], dtype=float)
            elif c == "SPECTYPE":
                data[c] = np.array(["QSO"] * n)
            else:
                data[c] = np.arange(n, dtype=np.int64)
        return Table(data)

    monkeypatch.setattr(desi.fitsio, "read", fake_fitsio_read)

    # balmask=True hits the AI_CIV/NCIV_450 column branch; UNIQPIX still added.
    desi.read_catalog("dummy.fits", balmask=True, bytile=False,
                      pixel_col="UNIQPIX")
    assert "UNIQPIX" in captured["columns"]
    assert "AI_CIV" in captured["columns"]  # confirms we exercised balmask branch


# --------------------------------------------------------------------------- #
# (2) dispatch column: np.unique on the chosen col selects the right TARGETIDs
# --------------------------------------------------------------------------- #
def _grouping_targetids(desi, catalog, pixel_col):
    """Drive desi-DLAGP's PRODUCTION dispatch helpers (``unique_pixel_cells`` +
    ``select_pixel_cell``) -- NOT a re-implementation. ``main`` uses these exact
    helpers, so a regression that ignored ``pixel_col`` (e.g. hard-coding
    HPXPIXEL back in) would make this test fail."""
    cells = desi.unique_pixel_cells(catalog, pixel_col)
    return {
        int(cell): set(
            np.asarray(desi.select_pixel_cell(catalog, pixel_col, cell)["TARGETID"])
        )
        for cell in cells
    }


def test_dispatch_column_selects_right_targetids():
    desi = _load_desi_dlagp()
    # HPXPIXEL and UNIQPIX intentionally disagree so the two groupings differ.
    catalog = Table(
        {
            "TARGETID": np.array([10, 11, 12, 13, 14], dtype=np.int64),
            "HPXPIXEL": np.array([100, 100, 200, 200, 200], dtype=np.int64),
            "UNIQPIX": np.array([7000000, 7000001, 7000000, 8000000, 8000000],
                                dtype=np.int64),
        }
    )

    by_hpx = _grouping_targetids(desi, catalog, "HPXPIXEL")
    assert set(by_hpx) == {100, 200}
    assert by_hpx[100] == {10, 11}
    assert by_hpx[200] == {12, 13, 14}

    by_uniq = _grouping_targetids(desi, catalog, "UNIQPIX")
    assert set(by_uniq) == {7000000, 7000001, 8000000}
    assert by_uniq[7000000] == {10, 12}   # spans two healpix
    assert by_uniq[7000001] == {11}
    assert by_uniq[8000000] == {13, 14}

    # Each grouping partitions the catalog: every TARGETID exactly once.
    all_hpx = set().union(*by_hpx.values())
    all_uniq = set().union(*by_uniq.values())
    assert all_hpx == all_uniq == {10, 11, 12, 13, 14}


# --------------------------------------------------------------------------- #
# (2b) loa_hpx_spec_counts: --pixel-col groups QSO counts by the chosen column
# --------------------------------------------------------------------------- #
def test_counts_from_qsocat_groups_by_selected_column(monkeypatch):
    import fitsio
    sc = _load_spec_counts()

    # Z all inside (constants.zmin_qso, zmax_qso) so the z-mask keeps every row.
    cat = {
        "Z": np.array([2.5, 3.0, 3.5, 2.8, 3.1], dtype=float),
        "HPXPIXEL": np.array([100, 100, 200, 200, 200], dtype=np.int64),
        "UNIQPIX": np.array([7000000, 7000001, 7000000, 8000000, 8000000],
                            dtype=np.int64),
    }
    captured = {}

    def fake_read(path, columns=None):
        captured["columns"] = list(columns)
        return {c: cat[c] for c in columns}

    monkeypatch.setattr(fitsio, "read", fake_read)

    # Default: HPXPIXEL (the parity-preserving desi-DLAGP index space).
    out_hpx = sc.counts_from_qsocat("dummy.fits")
    assert captured["columns"] == ["Z", "HPXPIXEL"]
    assert out_hpx == {100: 2, 200: 3}

    # UNIQPIX: groups by the per-UNIQPIX column instead (matterhorn path).
    out_uniq = sc.counts_from_qsocat("dummy.fits", pixel_col="UNIQPIX")
    assert captured["columns"] == ["Z", "UNIQPIX"]
    assert out_uniq == {7000000: 2, 7000001: 1, 8000000: 2}


def test_counts_from_qsocat_applies_zmask(monkeypatch):
    import fitsio
    sys.path.insert(0, REPO_ROOT)
    import constants  # the very z-bounds the tool applies
    sc = _load_spec_counts()

    lo, hi = constants.zmin_qso, constants.zmax_qso
    cat = {
        # one below lo, one above hi -> both dropped; two in-range kept.
        "Z": np.array([lo - 0.5, hi + 0.5, lo + 0.1, hi - 0.1], dtype=float),
        "HPXPIXEL": np.array([100, 100, 100, 200], dtype=np.int64),
    }

    def fake_read(path, columns=None):
        return {c: cat[c] for c in columns}

    monkeypatch.setattr(fitsio, "read", fake_read)
    out = sc.counts_from_qsocat("dummy.fits")
    assert out == {100: 1, 200: 1}  # the two out-of-range rows are excluded


def test_spec_counts_cli_rejects_unknown_pixel_col(monkeypatch):
    sc = _load_spec_counts()
    monkeypatch.setattr(
        sys, "argv",
        ["loa_hpx_spec_counts.py", "--qsocat", "x.fits",
         "--out", "y.txt", "--pixel-col", "BOGUS"],
    )
    # argparse choices=[HPXPIXEL, UNIQPIX] rejects the bad value -> exit 2,
    # before any file I/O.
    with pytest.raises(SystemExit):
        sc.main()


# --------------------------------------------------------------------------- #
# (3) CLI contract + launcher emission + the UNIQPIX healpix-mode guard
#
# PORT NOTE (2026-08-12 pixel_col repair): the original section (3) exercised
# the UNIQPIX/matterhorn coadd-path dispatch inside ``dlasearch_hpx``. That
# half was deliberately deferred to a separate PR (98b50a4: the findfile call
# is incompatible with desispec 0.70.0; PI decision "ship separately") and is
# NOT in this state; ``dlasearch_hpx`` keeps its legacy signature. The repair
# restores the driver-side interface + grouping and adds a FAIL-LOUD guard for
# UNIQPIX in healpix mode. These tests pin exactly that contract.
# --------------------------------------------------------------------------- #
def test_cli_accepts_pixel_col_hpxpixel():
    desi = _load_desi_dlagp()
    args = desi.parse(["--qsocat", "x.fits", "--release", "loa",
                       "--outdir", "/tmp/o", "--pixel_col", "HPXPIXEL"])
    assert args.pixel_col == "HPXPIXEL"
    # default is HPXPIXEL (parity with the pre-repair hardcoded behaviour)
    args2 = desi.parse(["--qsocat", "x.fits", "--release", "loa",
                        "--outdir", "/tmp/o"])
    assert args2.pixel_col == "HPXPIXEL"


def test_cli_rejects_unknown_pixel_col():
    desi = _load_desi_dlagp()
    with pytest.raises(SystemExit):
        desi.parse(["--qsocat", "x.fits", "--release", "loa",
                    "--outdir", "/tmp/o", "--pixel_col", "BOGUS"])


def test_cli_parses_full_launcher_emission():
    """Every --flag the production submit script emits must parse.

    This is the regression that would have caught the 56807753 failure: the
    launcher chain (launch_nersc.sh -> submit_desi_loa_nersc.sh) emitted
    --pixel_col HPXPIXEL while the frozen driver defined no such argument, so
    all 32 pilot tasks died at argparse before inference.
    """
    desi = _load_desi_dlagp()
    argv = [
        "--qsocat", "x.fits", "--release", "loa", "--program", "dark",
        "--survey", "main", "--outdir", "/tmp/o",
        "--learned_file", "m.h5", "--catalog_name", "c.mat",
        "--los_catalog", "los", "--dla_catalog", "dla",
        "--dla_samples_file", "d.mat", "--sub_dla_samples_file", "s.mat",
        "--min_z_separation", "3000.0", "--prev_tau_0", "0.00246",
        "--prev_beta", "3.62", "--max_dlas", "1", "--plot_figures", "0",
        "--filter_low_likelihood", "0", "--single_absorber_model", "1",
        "--max_workers", "8", "--batch_size", "1250",
        "--loading_min_lambda", "910", "--loading_max_lambda", "1550",
        "--normalization_min_lambda", "1425",
        "--normalization_max_lambda", "1475",
        "--min_lambda", "911.75", "--max_lambda", "1250",
        "--dlambda", "0.15", "--k", "30",
        "--num_dla_samples", "100000", "--num_subdla_samples", "100000",
        "--max_z_cut", "3000.0", "--min_z_cut", "3000.0",
        "--max_noise_variance", "9", "--num_forest_lines", "31",
        "--num_lines", "3", "--enable_tau_eb", "1",
        "--tau_eb_objective", "null", "--early_stop_mode", "baseline",
        "--pair_prior_mode", "off", "--dla_bias", "2.0",
        "--figure_dir", "/tmp/f",
        "--pixel_col", "HPXPIXEL",
        "--hpx_start", "0", "--hpx_end", "1",
    ]
    args = desi.parse(argv)
    assert args.pixel_col == "HPXPIXEL"


def test_uniqpix_healpix_mode_fails_loudly(monkeypatch):
    """UNIQPIX in healpix mode must exit(1) BEFORE any catalog read: the
    matterhorn coadd-path half ships separately (98b50a4), and a silent
    half-dispatch (UNIQPIX grouping + legacy coadd paths) must be impossible."""
    desi = _load_desi_dlagp()

    def _boom(*a, **k):                      # pragma: no cover - guard test
        raise AssertionError("read_catalog must not be reached")

    monkeypatch.setattr(desi, "read_catalog", _boom)
    with pytest.raises(SystemExit):
        desi.main(["--qsocat", "x.fits", "--release", "loa",
                   "--outdir", "/tmp/o", "--pixel_col", "UNIQPIX"])


# --------------------------------------------------------------------------- #
def test_balance_ignores_id_values_and_tiles_exactly(tmp_path):
    bal = _load_balance()

    # A counts table whose *id* column is sparse, non-contiguous UNIQPIX values
    # (huge ints, gaps). The balancer must key off INDEX/order only.
    rng = np.random.default_rng(0)
    n = 40
    uniqpix_ids = np.sort(rng.choice(np.arange(7_000_000, 9_000_000),
                                     size=n, replace=False))
    spec_counts = rng.integers(1, 50, size=n)
    counts_path = tmp_path / "matterhorn_counts.txt"
    counts_path.write_text(
        "\n".join(f"{int(u)} {int(c)}" for u, c in zip(uniqpix_ids, spec_counts))
        + "\n"
    )

    # read_counts must keep ONLY the count (last whitespace field), ignoring ids.
    counts = bal.read_counts(str(counts_path))
    assert counts == [int(c) for c in spec_counts]

    start, end, ntasks = 5, 33, 7
    boundaries = bal.compute_boundaries(counts, start, end, ntasks)

    # Exact tiling of [start, end): endpoints, monotone, no gap / no overlap.
    assert len(boundaries) == ntasks + 1
    assert boundaries[0] == start
    assert boundaries[-1] == end
    for k in range(ntasks):
        assert boundaries[k] <= boundaries[k + 1]  # monotone, contiguous
    # Adjacent tasks share exactly one edge -> [b_k, b_{k+1}) cover with no gap.
    covered = []
    for k in range(ntasks):
        covered.extend(range(boundaries[k], boundaries[k + 1]))
    assert covered == list(range(start, end))  # every index once, in order

    # Value-agnosticism proof: shifting the ids by an arbitrary huge offset
    # (so the id column is completely different) yields IDENTICAL boundaries,
    # because only the order/counts matter.
    shifted_path = tmp_path / "shifted_counts.txt"
    shifted_path.write_text(
        "\n".join(f"{int(u) + 10**12} {int(c)}"
                  for u, c in zip(uniqpix_ids, spec_counts)) + "\n"
    )
    counts_shifted = bal.read_counts(str(shifted_path))
    boundaries_shifted = bal.compute_boundaries(counts_shifted, start, end, ntasks)
    assert boundaries_shifted == boundaries


def test_balance_tiles_exactly_for_degenerate_zero_counts():
    bal = _load_balance()
    # All-zero counts -> equal-index split, still tiles [start, end) exactly.
    counts = [0] * 20
    start, end, ntasks = 2, 18, 4
    boundaries = bal.compute_boundaries(counts, start, end, ntasks)
    assert boundaries[0] == start and boundaries[-1] == end
    covered = []
    for k in range(ntasks):
        covered.extend(range(boundaries[k], boundaries[k + 1]))
    assert covered == list(range(start, end))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
