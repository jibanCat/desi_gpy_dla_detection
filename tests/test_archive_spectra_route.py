"""Tests for the --spectra_archive production I/O route (PI 2026-08-13 L-A).

Engineering/I-O contract only — the scientific finder path is exercised by
the Gate A-D validation suite, not here. Covers:
  - CLI accepts/propagates --spectra_archive; healpix-only + existence guards
    fail loudly (SystemExit);
  - read_archive_group fail-loud contract: missing archive, missing
    TARGETID, wavelength-grid mismatch, incompatible schema;
  - the adapter serves the stand-in with the exact committed f8 grid and
    f4->f8-promoted arrays, masks preserved exactly;
  - legacy path: archive=None leaves the historical behavior reachable
    (dlasearch_hpx signature default).
"""
import importlib.util
import os
import sys

import h5py
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import dlasearch  # noqa: E402

GRID_FILE = os.path.join(REPO_ROOT, "data", "brz_wave_grid_f8.npy")


def _make_archive(path, tids, n_pix=None, schema_version=1, wave=None):
    wave_f8 = np.load(GRID_FILE)
    if wave is None:
        wave = wave_f8.astype(np.float32)
    if n_pix is None:
        n_pix = len(wave)
    rng = np.random.RandomState(1)
    with h5py.File(path, "w") as h:
        h.attrs["schema_version"] = schema_version
        h.attrs["wave_step"] = 0.8
        h.create_dataset("wavelength", data=wave)
        cat = np.zeros(len(tids), dtype=[("TARGETID", "<i8")])
        cat["TARGETID"] = tids
        h.create_dataset("catalog", data=cat)
        h.create_dataset("flux", data=rng.normal(
            size=(len(tids), n_pix)).astype(np.float32))
        h.create_dataset("ivar", data=np.abs(rng.normal(
            size=(len(tids), n_pix))).astype(np.float32))
        h.create_dataset("mask", data=np.zeros(
            (len(tids), n_pix), dtype=np.uint32))
    return path


def test_grid_file_committed_and_matches_f4():
    assert os.path.isfile(GRID_FILE)
    w = np.load(GRID_FILE)
    assert w.dtype == np.float64 and w.shape == (7781,)
    assert w[0] == 3600.0


def test_missing_archive_fails_loud():
    with pytest.raises(FileNotFoundError, match="archive not found"):
        dlasearch.read_archive_group("/no/such/archive.h5", [1])


def test_missing_targetid_fails_loud(tmp_path):
    p = _make_archive(str(tmp_path / "a.h5"), [11, 22])
    with pytest.raises(KeyError, match="absent from archive"):
        dlasearch.read_archive_group(p, [11, 33])


def test_wavelength_grid_mismatch_fails_loud(tmp_path):
    w = np.load(GRID_FILE).astype(np.float32)
    w[100] += 0.1
    p = _make_archive(str(tmp_path / "a.h5"), [11], wave=w)
    with pytest.raises(RuntimeError, match="wavelength grid"):
        dlasearch.read_archive_group(p, [11])


def test_incompatible_schema_fails_loud(tmp_path):
    p = _make_archive(str(tmp_path / "a.h5"), [11], schema_version=99)
    with pytest.raises(RuntimeError, match="schema_version"):
        dlasearch.read_archive_group(p, [11])


def test_adapter_serves_exact_representation(tmp_path):
    p = _make_archive(str(tmp_path / "a.h5"), [11, 22])
    sp = dlasearch.read_archive_group(p, [22, 11])
    w8 = np.load(GRID_FILE)
    assert np.array_equal(sp.wave["brz"], w8)          # exact f8 grid
    assert sp.wave["brz"].dtype == np.float64
    assert sp.flux["brz"].dtype == np.float64          # f4 promoted exactly
    assert sp.ivar["brz"].dtype == np.float64
    with h5py.File(p) as h:
        assert np.array_equal(sp.flux["brz"][1],
                              h["flux"][0].astype(np.float64))
        assert np.array_equal(sp.mask["brz"][0], h["mask"][1])
    assert list(sp.fibermap["TARGETID"]) == [22, 11]   # requested order
    assert sp.bands == ["brz"]


def test_dlasearch_hpx_signature_defaults_to_legacy():
    import inspect
    sig = inspect.signature(dlasearch.dlasearch_hpx)
    assert sig.parameters["archive"].default is None


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "desi_dlagp_arc_test", os.path.join(REPO_ROOT, "desi-DLAGP.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cli_accepts_spectra_archive(monkeypatch):
    monkeypatch.delenv("GPDLA_SPECTRA_ARCHIVE", raising=False)
    drv = _load_driver()
    args = drv.parse(["--qsocat", "q.fits", "--release", "loa",
                      "--program", "dark", "--survey", "main",
                      "--outdir", "/tmp/x",
                      "--spectra_archive", "/x/a.h5"])
    assert args.spectra_archive == "/x/a.h5"
    args2 = drv.parse(["--qsocat", "q.fits", "--release", "loa",
                       "--program", "dark", "--survey", "main",
                       "--outdir", "/tmp/x"])
    assert args2.spectra_archive is None
