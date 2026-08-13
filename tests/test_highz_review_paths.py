"""Tests for the highz_review_package path parameterization (engineering only).

Covers the 2026-08-13 portability contract:
  - the three GL-historical input constants stay the defaults;
  - env vars HZ_CDDF_CAT / HZ_QSO_CAT / HZ_HPX_ROOT and the CLI flags
    override them;
  - missing required inputs fail loudly (SystemExit) instead of silently
    falling back or quietly producing zero pages;
  - the LoaArchive spectrum adapter returns the same {tid: {cam: (w,f,iv)}}
    shape as read_coadd_rows, so stitch()/page() are untouched.

No science/selection/plotting behaviour is exercised beyond shape contracts.
Synthetic stubs only; no real spectra.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIST_CDDF = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
             "loa_cddf_main_dark_v1/dlacat-loa-cddf-main-dark-v1.fits")
HIST_QSO = ("/nfs/turbo/lsa-cavestru/mfho/DESI/loa/"
            "QSO_cat_loa_main_dark_healpix_v2-altbal.fits")
HIST_HPX = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/healpix/main/dark"


def _load(monkeypatch=None, env=None):
    if monkeypatch is not None:
        for k in ("HZ_CDDF_CAT", "HZ_QSO_CAT", "HZ_HPX_ROOT", "HZ_ARCHIVE",
                  "HZ_REVIEW_OUT"):
            monkeypatch.delenv(k, raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
    path = os.path.join(REPO_ROOT, "tools", "highz_review_package.py")
    spec = importlib.util.spec_from_file_location("hzrev_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_defaults_preserved(monkeypatch):
    m = _load(monkeypatch)
    assert m.CDDF_CAT == HIST_CDDF
    assert m.QSO_CAT == HIST_QSO
    assert m.HPX_ROOT == HIST_HPX


def test_env_overrides(monkeypatch):
    m = _load(monkeypatch, env={"HZ_CDDF_CAT": "/x/c.fits",
                                "HZ_QSO_CAT": "/x/q.fits",
                                "HZ_HPX_ROOT": "/x/hpx"})
    assert m.CDDF_CAT == "/x/c.fits"
    assert m.QSO_CAT == "/x/q.fits"
    assert m.HPX_ROOT == "/x/hpx"


def _run_main(module, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["highz_review_package.py"] + argv)
    module.main()


def test_missing_cddf_cat_fails_loud(monkeypatch, tmp_path):
    m = _load(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _run_main(m, monkeypatch,
                  ["--cddf-cat", str(tmp_path / "absent.fits")])
    assert "FATAL" in str(e.value) and "absent.fits" in str(e.value)


def test_missing_qso_cat_fails_loud(monkeypatch, tmp_path):
    real = tmp_path / "cddf.fits"
    real.write_bytes(b"")
    m = _load(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _run_main(m, monkeypatch,
                  ["--cddf-cat", str(real),
                   "--qso-cat", str(tmp_path / "absent.fits")])
    assert "FATAL" in str(e.value)


def test_missing_hpx_root_fails_loud_without_archive(monkeypatch, tmp_path):
    c = tmp_path / "cddf.fits"
    q = tmp_path / "qso.fits"
    c.write_bytes(b"")
    q.write_bytes(b"")
    m = _load(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _run_main(m, monkeypatch,
                  ["--cddf-cat", str(c), "--qso-cat", str(q),
                   "--hpx-root", str(tmp_path / "no_such_tree")])
    assert "FATAL" in str(e.value) and "no_such_tree" in str(e.value)


def test_missing_archive_fails_loud(monkeypatch, tmp_path):
    c = tmp_path / "cddf.fits"
    q = tmp_path / "qso.fits"
    c.write_bytes(b"")
    q.write_bytes(b"")
    m = _load(monkeypatch)
    with pytest.raises(SystemExit) as e:
        _run_main(m, monkeypatch,
                  ["--cddf-cat", str(c), "--qso-cat", str(q),
                   "--archive", str(tmp_path / "absent.h5")])
    assert "FATAL" in str(e.value) and "absent.h5" in str(e.value)


class _StubSpectrum:
    def __init__(self, w, f, iv):
        self.wavelength, self.flux, self.ivar = w, f, iv


class _StubArchive:
    def __init__(self, specs):
        self._specs = specs

    def get_spectrum(self, tid):
        if tid not in self._specs:
            raise KeyError(tid)
        return self._specs[tid]


def test_archive_adapter_matches_stitch_contract(monkeypatch):
    m = _load(monkeypatch)
    w = np.linspace(3600.0, 9800.0, 50)
    f = np.random.RandomState(0).normal(size=50)
    iv = np.ones(50)
    ar = _StubArchive({7: _StubSpectrum(w, f, iv)})
    out = m.read_archive_rows(ar, [7, 8])
    assert set(out) == {7}          # missing tid skipped, not fatal
    spec = out[7]
    assert set(spec) == {"B", "R", "Z"}
    sw, sf, siv = m.stitch(spec)
    np.testing.assert_array_equal(sw, w)
    np.testing.assert_array_equal(sf, f)
    np.testing.assert_array_equal(siv, iv)
