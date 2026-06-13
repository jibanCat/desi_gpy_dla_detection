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
def _grouping_targetids(catalog, pixel_col):
    """Replicate desi-DLAGP's dispatch: unique cells over pixel_col, then the
    per-cell sub-catalog selection ``catalog[catalog[pixel_col] == cell]``."""
    cells = np.unique(catalog[pixel_col])
    return {
        int(cell): set(np.asarray(catalog[catalog[pixel_col] == cell]["TARGETID"]))
        for cell in cells
    }


def test_dispatch_column_selects_right_targetids():
    # HPXPIXEL and UNIQPIX intentionally disagree so the two groupings differ.
    catalog = Table(
        {
            "TARGETID": np.array([10, 11, 12, 13, 14], dtype=np.int64),
            "HPXPIXEL": np.array([100, 100, 200, 200, 200], dtype=np.int64),
            "UNIQPIX": np.array([7000000, 7000001, 7000000, 8000000, 8000000],
                                dtype=np.int64),
        }
    )

    by_hpx = _grouping_targetids(catalog, "HPXPIXEL")
    assert set(by_hpx) == {100, 200}
    assert by_hpx[100] == {10, 11}
    assert by_hpx[200] == {12, 13, 14}

    by_uniq = _grouping_targetids(catalog, "UNIQPIX")
    assert set(by_uniq) == {7000000, 7000001, 8000000}
    assert by_uniq[7000000] == {10, 12}   # spans two healpix
    assert by_uniq[7000001] == {11}
    assert by_uniq[8000000] == {13, 14}

    # Each grouping partitions the catalog: every TARGETID exactly once.
    all_hpx = set().union(*by_hpx.values())
    all_uniq = set().union(*by_uniq.values())
    assert all_hpx == all_uniq == {10, 11, 12, 13, 14}


# --------------------------------------------------------------------------- #
# (3) coadd path: UNIQPIX -> spectra/<u//100>/<u>/..., HPXPIXEL -> healpix/...
# --------------------------------------------------------------------------- #
def _drive_dlasearch_path(monkeypatch, pixel_col, cell, datapath, release):
    """Run dlasearch.dlasearch_hpx far enough to build `coadd`, capture it.

    `os.path.exists` is patched to return False (coadd "missing"), so the
    function returns () immediately after constructing the path -- no model
    work, no real spectra. We capture the exact path it built.
    """
    import dlasearch

    captured = {}

    def fake_exists(path):
        captured["coadd"] = path
        return False  # short-circuit before any DLAHolder / spectra reads

    monkeypatch.setattr(dlasearch.os.path, "exists", fake_exists)

    result = dlasearch.dlasearch_hpx(
        healpix=cell,
        survey="main",
        program="dark",
        datapath=datapath,
        hpxcat=Table({"TARGETID": np.array([1], dtype=np.int64)}),
        model_params={},  # never touched: coadd is "missing"
        release=release,
        pixel_col=pixel_col,
    )
    assert result == ()  # missing-coadd early return
    return captured["coadd"]


def test_uniqpix_coadd_path_uses_spectra_uniqpix_layout(monkeypatch):
    # findfile needs a redux root; provide a throwaway one so it is pure.
    monkeypatch.setenv("DESI_SPECTRO_REDUX", "/tmp/nope_redux")
    cell = 12345
    path = _drive_dlasearch_path(
        monkeypatch, pixel_col="UNIQPIX", cell=cell,
        datapath="/ignored/for/uniqpix", release="matterhorn",
    )
    # .../<specprod>/spectra/main/dark/<u//100>/<u>/coadd-main-dark-<u>.fits
    expected_tail = os.path.join(
        "matterhorn", "spectra", "main", "dark",
        str(cell // 100), str(cell), f"coadd-main-dark-{cell}.fits",
    )
    assert path.endswith(expected_tail), path
    assert os.sep + "spectra" + os.sep in path
    assert os.sep + "healpix" + os.sep not in path


def test_hpxpixel_coadd_path_uses_healpix_layout(monkeypatch):
    cell = 12345
    # In production this datapath is .../<specprod>/healpix/main/dark
    datapath = os.path.join("/somewhere", "loa", "healpix", "main", "dark")
    path = _drive_dlasearch_path(
        monkeypatch, pixel_col="HPXPIXEL", cell=cell,
        datapath=datapath, release="loa",
    )
    expected = os.path.join(
        datapath, str(cell // 100), str(cell), f"coadd-main-dark-{cell}.fits",
    )
    assert path == expected
    assert os.sep + "healpix" + os.sep in path
    assert os.sep + "spectra" + os.sep not in path


def test_path_branches_byte_identical_for_default(monkeypatch):
    """Parity contract: the default pixel_col yields the legacy healpix path
    exactly as the pre-feature one-liner did."""
    cell = 987
    datapath = "/d/loa/healpix/main/dark"
    legacy = os.path.join(
        datapath, str(cell // 100), str(cell),
        f"coadd-main-dark-{cell}.fits",
    )
    path = _drive_dlasearch_path(
        monkeypatch, pixel_col="HPXPIXEL", cell=cell,
        datapath=datapath, release="loa",
    )
    assert path == legacy


# --------------------------------------------------------------------------- #
# (4) balance value-agnosticism: tiles [start, end) regardless of id values
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
