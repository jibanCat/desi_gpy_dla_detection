"""Tests for the NO-COMBINE streaming O3/O1 CDDF driver
(``CDDF_analysis.cddf_forward.streaming``).

THE correctness pin
--------------------
``compute_o3_products_streaming([fileA, fileB, fileC])`` must equal
``compute_o3_products(<combined file built from A,B,C>)`` to floating-point
tolerance for EVERY output array (o1 + o3_cddf f/f68/f95, o3_dndx, completeness
C/b_FP/n_truth, o3_omega).  Likewise O1 streaming == O1 combined.

The per-(logN, z)-bin Poisson-binomial deposit is ADDITIVE over sightlines: the
``(probs, poissons)`` ingredients from ``_split_distributions_single`` concatenate
(probs lists) / sum (poisson totals) across files, and the partitioned
matched/unmatched deposit + n_truth + dX accumulate likewise.  Running the
EXISTING CI-combine once on the accumulated totals therefore reproduces the
single-combined-file result exactly (each TARGETID lives in exactly one file, so
no double-counting).

These tests build 2-3 small SYNTHETIC processed files + their concatenation and
assert equality.  They inject the SAME fake Bayesian core the O3 driver tests use
(so the CS-side streaming logic is tested without the real core's sampler RNG,
which is deterministic but easier to pin via the fake), and ALSO exercise the
real core on the omega path indirectly via the driver.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "fixtures", "cddf"))
sys.path.insert(0, _HERE)  # so we can reuse test_cddf_o3_driver._FakeCore

pytest.importorskip("h5py")
pytest.importorskip("astropy")
pytest.importorskip("scipy")

import h5py  # noqa: E402

from CDDF_analysis.cddf_forward import driver as o3driver  # noqa: E402
from CDDF_analysis.cddf_forward import streaming as o3stream  # noqa: E402
from CDDF_analysis.cddf_forward.window import WindowSpec  # noqa: E402
from build_synthetic_cddf_fixture import build_synthetic_cddf  # noqa: E402
from build_synthetic_truth_fixture import write_truth_catalog  # noqa: E402

# Reuse the O3 driver test's fake core (identical §2 signatures) so the streaming
# accumulation is pinned against the SAME deterministic correction the combined
# driver uses — making streaming==combined a pure-array equality, not an RNG race.
from test_cddf_o3_driver import _FakeCore  # noqa: E402


_Z_MIN = 2.4
_Z_MAX = 3.3
_LNHI_MIN = 20.3
_LNHI_MAX = 22.5
_LNHI_NBINS = 3
_WINDOW = WindowSpec(z_min_lyb=False, z_max_lyb=False)
_DLACAT_KWARGS = dict(sub_dla=False, snr=-2, lowzcut=False, highzcut=False)


# --------------------------------------------------------------------------- #
# Fixture: 3 disjoint per-healpix processed files + their concatenation.
# --------------------------------------------------------------------------- #
def _build_one_file(out_dir, *, tag, base_tid, n_spec, lnhi_hi):
    """One synthetic per-healpix processed file with disjoint TARGETIDs."""
    rng_dummy = None  # determinism comes from build_synthetic_cddf's own layout
    p_dla = tuple(1.0 if i % 4 != 3 else 0.0 for i in range(n_spec))
    peak_logN = tuple(
        None if p == 0 else float(20.4 + 0.6 * (i % 3)) for i, p in enumerate(p_dla)
    )
    peak_z = tuple(
        None if p == 0 else float(2.5 + 0.25 * (i % 3)) for i, p in enumerate(p_dla)
    )
    sub = os.path.join(out_dir, tag)
    os.makedirs(sub, exist_ok=True)
    synth = build_synthetic_cddf(
        sub,
        n_spec=n_spec,
        p_dla=p_dla,
        peak_logN=peak_logN,
        peak_z=peak_z,
        z_qso=3.6,
        z_min=2.4,
        z_max=3.3,
        lnhi_hi=lnhi_hi,
    )
    # Rename to the per-healpix glob (processed-*-*.h5) and shift TARGETIDs so the
    # files are DISJOINT (each TARGETID lives in exactly one healpix file).
    proc = os.path.join(sub, f"processed-{tag}-100.h5")
    os.replace(synth["processed_file"], proc)
    with h5py.File(proc, "r+") as f:
        tids = np.asarray(f["target_ids"][:]).astype(np.int64)
        del f["target_ids"]
        f["target_ids"] = tids + np.int64(base_tid)
    synth["processed_file"] = proc
    synth["target_ids"] = (1000 + np.arange(n_spec)).astype(np.int64) + np.int64(base_tid)
    synth["p_dla"] = np.asarray(p_dla, float)
    synth["peak_logN"] = peak_logN
    synth["peak_z"] = peak_z
    return synth


def _concat_processed(files, out_path):
    """Build a single COMBINED processed HDF5 by concatenating per-file datasets.

    This is the on-disk monolith the streaming driver AVOIDS; the test builds it
    only to pin streaming==combined.  All files share the same sample grid, so the
    concatenation along axis 0 of the per-spectrum datasets is the exact combined
    file ``combine_processed_h5.py`` would produce.
    """
    keys_1d = ["min_z_dlas", "max_z_dlas", "z_qsos", "target_ids", "snrs",
               "log_likelihoods_dla"]
    keys_2d = ["model_posteriors", "sample_log_likelihoods_dla"]
    bufs = {k: [] for k in keys_1d + keys_2d}
    for fp in files:
        with h5py.File(fp, "r") as f:
            for k in keys_1d + keys_2d:
                bufs[k].append(np.asarray(f[k][:]))
    with h5py.File(out_path, "w") as f:
        for k in keys_1d + keys_2d:
            f[k] = np.concatenate(bufs[k], axis=0)
    return out_path


def _write_combined_truth(per_file_synths, out_path):
    """One truth FITS spanning all files (1 absorber per active sightline)."""
    tids, nhis, zs = [], [], []
    for synth in per_file_synths:
        tarr = synth["target_ids"]
        for i in range(len(synth["p_dla"])):
            if synth["p_dla"][i] == 0:
                continue
            tids.append(int(tarr[i]))
            nhis.append(synth["peak_logN"][i])
            zs.append(synth["peak_z"][i])
    write_truth_catalog(out_path, target_ids=tids, nhi=nhis, z=zs)
    return out_path


@pytest.fixture
def three_files(tmp_path):
    """3 disjoint per-healpix files (8, 12, 8 spectra), a combined file, truth."""
    s0 = _build_one_file(tmp_path, tag="000", base_tid=0, n_spec=8, lnhi_hi=_LNHI_MAX)
    s1 = _build_one_file(tmp_path, tag="001", base_tid=100000, n_spec=12, lnhi_hi=_LNHI_MAX)
    s2 = _build_one_file(tmp_path, tag="002", base_tid=200000, n_spec=8, lnhi_hi=_LNHI_MAX)
    synths = [s0, s1, s2]
    files = [s["processed_file"] for s in synths]
    combined = _concat_processed(files, str(tmp_path / "combined.h5"))
    truth = _write_combined_truth(synths, str(tmp_path / "truth_all.fits"))
    # all files share one sample grid / catalog grid; the combined-driver path needs
    # a catalog spanning ALL combined TARGETIDs.
    from astropy.table import Table
    all_tids = np.concatenate([s["target_ids"] for s in synths]).astype(np.int64)
    cat_path = str(tmp_path / "catalog_all.fits")
    Table({"TARGETID": all_tids, "Z": np.full(all_tids.size, 3.6)}).write(
        cat_path, overwrite=True
    )
    return {
        "files": files,
        "combined": combined,
        "truth": truth,
        "catalog": cat_path,
        "sample": s0["sample_file"],
        "synths": synths,
        "tmp_path": tmp_path,
    }


def _common_kwargs():
    return dict(
        z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
        lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
    )


def _combined_o3(three_files, monkeypatch):
    monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
    return o3driver.compute_o3_products(
        three_files["combined"], three_files["sample"], three_files["catalog"],
        three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(),
    )


def _streaming_o3(three_files, monkeypatch):
    monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
    return o3stream.compute_o3_products_streaming(
        three_files["files"], three_files["sample"], three_files["catalog"],
        three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(),
    )


def _assert_array_close(a, b, name):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    np.testing.assert_allclose(
        np.nan_to_num(a, nan=-12345.0), np.nan_to_num(b, nan=-12345.0),
        rtol=1e-9, atol=1e-9, err_msg=name,
    )


class TestStreamingEqualsCombined:
    """THE pin: streaming == combined for every output array."""

    def test_o3_cddf_matches_combined(self, three_files, monkeypatch):
        comb = _combined_o3(three_files, monkeypatch)
        strm = _streaming_o3(three_files, monkeypatch)
        for k in ("logN", "f", "f68", "f95", "f_raw", "f68_raw", "f95_raw", "n_corr"):
            _assert_array_close(strm["o3_cddf"][k], comb["o3_cddf"][k], f"o3_cddf[{k}]")

    def test_o3_dndx_matches_combined(self, three_files, monkeypatch):
        comb = _combined_o3(three_files, monkeypatch)
        strm = _streaming_o3(three_files, monkeypatch)
        for k in ("z", "dndx", "dndx68", "dndx95", "dndx_raw", "n_corr"):
            _assert_array_close(strm["o3_dndx"][k], comb["o3_dndx"][k], f"o3_dndx[{k}]")

    def test_completeness_matches_combined(self, three_files, monkeypatch):
        comb = _combined_o3(three_files, monkeypatch)
        strm = _streaming_o3(three_files, monkeypatch)
        for k in ("C", "b_FP", "n_truth", "F_matched", "F_unmatched"):
            _assert_array_close(
                strm["completeness"][k], comb["completeness"][k], f"completeness[{k}]"
            )

    def test_omega_matches_combined(self, three_files, monkeypatch):
        comb = _combined_o3(three_files, monkeypatch)
        strm = _streaming_o3(three_files, monkeypatch)
        for k in ("omega", "omega68", "omega95"):
            _assert_array_close(strm["o3_omega"][k], comb["o3_omega"][k], f"o3_omega[{k}]")

    def test_o1_block_matches_combined(self, three_files, monkeypatch):
        comb = _combined_o3(three_files, monkeypatch)
        strm = _streaming_o3(three_files, monkeypatch)
        for blk in ("cddf", "dndx", "omega"):
            cb = comb["o1"][blk]
            sb = strm["o1"][blk]
            for k in cb:
                if k == "xerrs":
                    continue
                _assert_array_close(sb[k], cb[k], f"o1[{blk}][{k}]")


class TestO1StreamingEqualsCombined:
    """O1-only streaming == O1 combined (independent of the O3 core)."""

    def test_o1_products_streaming_matches_compute_o1(self, three_files):
        comb = o3driver.compute_o1_products(
            three_files["combined"], three_files["sample"], three_files["catalog"],
            **_DLACAT_KWARGS, **_common_kwargs(),
        )
        strm = o3stream.compute_o1_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            **_DLACAT_KWARGS, **_common_kwargs(),
        )
        for blk in ("cddf", "dndx", "omega"):
            for k in comb[blk]:
                if k == "xerrs":
                    continue
                _assert_array_close(strm[blk][k], comb[blk][k], f"{blk}[{k}]")


class TestNoCombineDiscipline:
    """Streaming never builds a combined file and opens each file exactly once."""

    def test_processes_list_without_writing_combined(self, three_files, monkeypatch):
        # Count DLACatalogue constructions; must equal n_files (one per file, no
        # combined-file construction). Spy on the constructor in the streaming module.
        import CDDF_analysis.cddf_forward.streaming as st

        constructed = {"n": 0, "paths": []}
        orig = st.DLACatalogue

        class _Spy(orig):
            def __init__(self, *a, **k):
                constructed["n"] += 1
                constructed["paths"].append(k.get("processed_file", a[0] if a else None))
                super().__init__(*a, **k)

        monkeypatch.setattr(st, "DLACatalogue", _Spy)
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        prod = o3stream.compute_o3_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(),
        )
        assert constructed["n"] == len(three_files["files"])
        # none of the constructed catalogs is a combined file.
        for p in constructed["paths"]:
            assert "combined" not in str(p)
        assert prod["provenance"]["streaming"] is True
        assert prod["provenance"]["n_files"] == len(three_files["files"])

    def test_accepts_directory_glob(self, three_files, monkeypatch, tmp_path):
        # A directory of processed-*-*.h5 files must be globbed (same result as the
        # explicit list). The fixture's files live in per-tag subdirs, so copy them
        # flat into one dir to exercise the glob.
        import shutil
        glob_dir = tmp_path / "healpix_dir"
        glob_dir.mkdir()
        for fp in three_files["files"]:
            shutil.copy(fp, glob_dir / os.path.basename(fp))
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        prod_dir = o3stream.compute_o3_products_streaming(
            str(glob_dir), three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(),
        )
        prod_list = _streaming_o3(three_files, monkeypatch)
        _assert_array_close(
            prod_dir["o3_cddf"]["f"], prod_list["o3_cddf"]["f"], "glob f"
        )
        assert prod_dir["provenance"]["n_files"] == len(three_files["files"])


class TestStreamingProvenance:
    def test_per_healpix_coverage_recorded(self, three_files, monkeypatch):
        strm = _streaming_o3(three_files, monkeypatch)
        cov = strm["provenance"]["coverage"]
        # union TARGETID counts accumulate across files.
        assert cov["n_both"] >= 1
        # per-file (per-healpix) provenance is recorded.
        assert "per_file" in strm["provenance"]
        assert len(strm["provenance"]["per_file"]) == len(three_files["files"])
        assert strm["provenance"]["window_applied"] is True

    def test_filter_on_raises_first(self, three_files, monkeypatch):
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        with pytest.raises(ValueError, match="FILTER"):
            o3stream.compute_o3_products_streaming(
                three_files["files"], three_files["sample"], three_files["catalog"],
                three_files["truth"], **_DLACAT_KWARGS,
                z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
                lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=1, window=_WINDOW,
            )


class TestParallelStreaming:
    """THE parallel pin: ``n_workers > 1`` == sequential to floating-point.

    The per-file deposit is INDEPENDENT (each file -> additive per-bin ingredients),
    so mapping it across a ``multiprocessing`` pool and REDUCING the ingredients —
    then running the SAME single correction at the end — must reproduce the
    sequential streaming result bit-for-bit modulo float summation order (allclose).
    """

    def test_o3_parallel_equals_sequential(self, three_files, monkeypatch):
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        seq = o3stream.compute_o3_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(),
        )
        par = o3stream.compute_o3_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(),
            n_workers=3,
        )
        for blk in ("o3_cddf",):
            for k in ("logN", "f", "f68", "f95", "f_raw", "n_corr"):
                _assert_array_close(par[blk][k], seq[blk][k], f"par {blk}[{k}]")
        for k in ("z", "dndx", "dndx68", "dndx95", "n_corr"):
            _assert_array_close(par["o3_dndx"][k], seq["o3_dndx"][k], f"par o3_dndx[{k}]")
        for k in ("C", "b_FP", "n_truth", "F_matched", "F_unmatched"):
            _assert_array_close(
                par["completeness"][k], seq["completeness"][k], f"par completeness[{k}]"
            )
        for k in ("omega", "omega68", "omega95"):
            _assert_array_close(par["o3_omega"][k], seq["o3_omega"][k], f"par o3_omega[{k}]")
        for blk in ("cddf", "dndx", "omega"):
            for k in seq["o1"][blk]:
                if k == "xerrs":
                    continue
                _assert_array_close(par["o1"][blk][k], seq["o1"][blk][k], f"par o1[{blk}][{k}]")

    def test_o1_parallel_equals_sequential(self, three_files):
        seq = o3stream.compute_o1_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            **_DLACAT_KWARGS, **_common_kwargs(),
        )
        par = o3stream.compute_o1_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            **_DLACAT_KWARGS, **_common_kwargs(), n_workers=2,
        )
        for blk in ("cddf", "dndx", "omega"):
            for k in seq[blk]:
                if k == "xerrs":
                    continue
                _assert_array_close(par[blk][k], seq[blk][k], f"par {blk}[{k}]")

    def test_parallel_provenance_records_workers(self, three_files, monkeypatch):
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        par = o3stream.compute_o3_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(), n_workers=3,
        )
        assert par["provenance"]["streaming"] is True
        assert par["provenance"]["n_workers"] == 3
        assert par["provenance"]["n_files"] == len(three_files["files"])

    def test_n_workers_one_is_default_sequential(self, three_files, monkeypatch):
        # n_workers=1 must take the exact sequential path (back-compat) and record it.
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        prod = o3stream.compute_o3_products_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS, **_common_kwargs(), n_workers=1,
        )
        assert prod["provenance"]["n_workers"] == 1

    def test_closure_parallel_equals_sequential(self, three_files, monkeypatch):
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        seq = o3stream.heldout_closure_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS,
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
        )
        par = o3stream.heldout_closure_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS,
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
            n_workers=2,
        )
        _assert_array_close(par["corrected"], seq["corrected"], "par closure.corrected")
        _assert_array_close(par["truth"], seq["truth"], "par closure.truth")
        _assert_array_close(par["residual"], seq["residual"], "par closure.residual")
        assert par["bfp_rebase_ratio"] == pytest.approx(seq["bfp_rebase_ratio"])
        assert bool(par["passed"]) == bool(seq["passed"])


class TestHeldoutClosureStreaming:
    def test_closure_streaming_runs_and_reports_pass_flag(self, three_files, monkeypatch):
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        out = o3stream.heldout_closure_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS,
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
        )
        for key in ("residual", "standardized_residual", "passed", "n_valid_bins",
                    "bfp_rebase_ratio"):
            assert key in out
        assert isinstance(bool(out["passed"]), bool)

    def test_closure_streaming_matches_combined(self, three_files, monkeypatch):
        # The streaming closure must reproduce the single-combined-file closure.
        monkeypatch.setattr(o3driver, "soft_completeness", _FakeCore, raising=False)
        comb = o3driver.heldout_closure(
            three_files["combined"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS,
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
        )
        monkeypatch.setattr(o3stream, "soft_completeness", _FakeCore, raising=False)
        strm = o3stream.heldout_closure_streaming(
            three_files["files"], three_files["sample"], three_files["catalog"],
            three_files["truth"], **_DLACAT_KWARGS,
            z_min=_Z_MIN, z_max=_Z_MAX, lnhi_min=_LNHI_MIN, lnhi_max=_LNHI_MAX,
            lnhi_nbins=_LNHI_NBINS, filter_low_likelihood=0, window=_WINDOW,
        )
        _assert_array_close(strm["corrected"], comb["corrected"], "closure.corrected")
        _assert_array_close(strm["truth"], comb["truth"], "closure.truth")
        assert strm["bfp_rebase_ratio"] == pytest.approx(comb["bfp_rebase_ratio"])


class TestUnreadableFileRobustness:
    """A truncated/corrupt processed file must be SKIPPED + recorded, not crash."""

    def test_truncated_file_skipped_and_recorded(self, three_files, monkeypatch, tmp_path):
        monkeypatch.setattr(o3stream, "_require_core", lambda: _FakeCore, raising=False)
        # a 96-byte HDF5 stub (what a killed job leaves), named like a real output
        bad = str(tmp_path / "processed-spectra-16-9999.h5")
        with open(bad, "wb") as fh:
            fh.write(b"\x89HDF\r\n\x1a\n" + b"\x00" * 88)
        good = list(three_files["files"])

        prod = o3stream.compute_o3_products_streaming(
            good + [bad], three_files["sample"], three_files["catalog"],
            three_files["truth"], n_workers=1, **_DLACAT_KWARGS, **_common_kwargs(),
        )
        unread = prod["provenance"]["unreadable_files"]
        assert len(unread) == 1
        assert "9999" in unread[0]["file"]
        assert prod["provenance"]["n_files"] == len(good)  # bad one excluded

        # result is identical to running on ONLY the good files
        ref = o3stream.compute_o3_products_streaming(
            good, three_files["sample"], three_files["catalog"],
            three_files["truth"], n_workers=1, **_DLACAT_KWARGS, **_common_kwargs(),
        )
        np.testing.assert_allclose(
            prod["o3_dndx"]["dndx"], ref["o3_dndx"]["dndx"], rtol=0, atol=1e-12
        )

    def test_all_unreadable_raises(self, tmp_path):
        bad = str(tmp_path / "processed-spectra-16-9998.h5")
        with open(bad, "wb") as fh:
            fh.write(b"\x00" * 96)
        with pytest.raises(ValueError, match="no readable"):
            o3stream.compute_o3_products_streaming(
                [bad], "s", "c", "t", **_common_kwargs(),
            )
