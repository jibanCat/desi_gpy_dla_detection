"""Tests for the lossless gzip compression of saved processed-h5 result files.

Covers the shared `_gzip_kwargs` helper and BOTH writers that persist
per-spectrum GP-DLA results:
  - `process_helpers.save_results_to_hdf5`  (run_bayes_select standalone path)
  - `run_bayes_select.DLAHolder.save_results` (the mock/DESI/LOA production path,
    called from dlasearch.py:621 model.save_results)

The compression is purely an on-disk encoding change: every value (including
NaN fill) must round-trip bit-identically. The big per-sample arrays are mostly
NaN in FILTER+multi-DLA+early-stop runs, so they compress heavily; these tests
assert both losslessness and that a NaN-heavy array shrinks substantially.
"""
import os

import h5py
import numpy as np
import pytest

from gpy_dla_detection.process_helpers import _gzip_kwargs, save_results_to_hdf5

GZIP = {"compression": "gzip", "compression_opts": 4}


# --- _gzip_kwargs ------------------------------------------------------------
def test_gzip_kwargs_float_array():
    assert _gzip_kwargs(np.zeros((5, 3))) == GZIP


def test_gzip_kwargs_int_array():
    assert _gzip_kwargs(np.arange(10, dtype=np.int32)) == GZIP


def test_gzip_kwargs_1d_array():
    assert _gzip_kwargs(np.array([1.0, 2.0, 3.0])) == GZIP


def test_gzip_kwargs_scalar_gets_no_compression():
    # gzip needs a chunked layout, impossible for a scalar (ndim 0)
    assert _gzip_kwargs(np.float64(1.0)) == {}
    assert _gzip_kwargs(5) == {}


def test_gzip_kwargs_empty_gets_no_compression():
    assert _gzip_kwargs(np.array([])) == {}


# --- shared fixture ----------------------------------------------------------
def _nan_heavy_results(n=4, ns=2000, k=4):
    """A results dict shaped like real output: a ~98% NaN per-sample array, an
    int index array, plus small per-spectrum vectors and a target_ids axis.
    (No z_qsos: save_results_to_hdf5 writes z_qsos from a separate argument, so
    including it here would collide; DLAHolder.save_results writes only the dict.)"""
    rs = np.random.RandomState(0)
    big = np.full((n, ns, k), np.nan)
    big[:, :40, :] = rs.randn(n, 40, k)  # ~98% NaN
    return {
        "sample_log_likelihoods_dla": big,
        "base_sample_inds": rs.randint(0, ns, (n, k - 1, ns)).astype(np.int32),
        "model_posteriors": rs.rand(n, 5),
        "p_dlas": rs.rand(n),
        "target_ids": np.arange(n, dtype=np.int64),
    }


def _assert_lossless(h5file, results):
    with h5py.File(h5file, "r") as h:
        for key, val in results.items():
            a, b = np.asarray(val), h[key][()]
            assert a.shape == b.shape and a.dtype == b.dtype, key
            if a.dtype.kind in "fc":
                assert np.array_equal(a, b, equal_nan=True), key
            else:
                assert np.array_equal(a, b), key


# --- save_results_to_hdf5 (process_helpers) ----------------------------------
def test_save_results_to_hdf5_compresses_and_round_trips(tmp_path):
    results = _nan_heavy_results()
    fn = str(tmp_path / "ph.h5")
    z_qsos = np.linspace(2.1, 3.5, 4)
    save_results_to_hdf5(fn, results, [f"s{i}" for i in range(4)], z_qsos)
    with h5py.File(fn, "r") as h:
        assert h["sample_log_likelihoods_dla"].compression == "gzip"
        assert h["base_sample_inds"].compression == "gzip"
        assert h["spectrum_ids"].compression == "gzip"
    _assert_lossless(fn, results)


def test_save_results_with_run_attrs_compresses_and_writes_attrs(tmp_path):
    """Regression for the PR #7 (gzip) x clustering_prior (run_attrs) merge:
    the compressed datasets AND the root-group provenance attrs must coexist,
    and each result dataset must be written exactly once. A duplicate
    create_dataset (the bad keep-both resolution) would raise ValueError at
    the save call below, so this fails loudly on a regressed merge."""
    results = _nan_heavy_results()
    fn = str(tmp_path / "ph_attrs.h5")
    z_qsos = np.linspace(2.1, 3.4, 4)
    attrs = {"pair_prior_mode": "clustering", "dla_bias": 2.0}
    save_results_to_hdf5(
        fn, results, [f"s{i}" for i in range(4)], z_qsos, run_attrs=attrs
    )
    with h5py.File(fn, "r") as h:
        # provenance attrs present + correct
        assert h.attrs["pair_prior_mode"] == "clustering"
        assert float(h.attrs["dla_bias"]) == 2.0
        # datasets still gzip-compressed (conflict-1 kept the gzip loop body)
        assert h["sample_log_likelihoods_dla"].compression == "gzip"
        assert h["base_sample_inds"].compression == "gzip"
    _assert_lossless(fn, results)


def test_nan_heavy_array_shrinks_substantially(tmp_path):
    big = np.full((2, 5000, 4), np.nan)
    big[:, :50, :] = 1.23  # 97.5% NaN
    fn = str(tmp_path / "shrink.h5")
    save_results_to_hdf5(
        fn, {"sample_log_likelihoods_dla": big}, ["a", "b"], np.array([2.0, 3.0])
    )
    # a 97.5%-NaN array must end well under a third of its raw byte size
    assert os.path.getsize(fn) < big.nbytes * 0.3


# --- DLAHolder.save_results (run_bayes_select; the production/LOA writer) -----
def test_dlaholder_save_results_compresses_and_round_trips(tmp_path):
    from run_bayes_select import DLAHolder

    results = _nan_heavy_results()
    holder = object.__new__(DLAHolder)  # bypass __init__; save_results only reads .results
    holder.results = results
    fn = str(tmp_path / "holder.h5")
    holder.save_results(output_file=fn)
    with h5py.File(fn, "r") as h:
        assert h["sample_log_likelihoods_dla"].compression == "gzip"
        assert h["base_sample_inds"].compression == "gzip"
    _assert_lossless(fn, results)


def test_dlaholder_handles_scalar_value(tmp_path):
    """A scalar in results must not raise (gzip can't chunk a scalar)."""
    from run_bayes_select import DLAHolder

    holder = object.__new__(DLAHolder)
    holder.results = {"a_scalar": np.float64(3.14), "a_vec": np.arange(5)}
    fn = str(tmp_path / "scalar.h5")
    holder.save_results(output_file=fn)
    with h5py.File(fn, "r") as h:
        assert h["a_scalar"].compression is None  # scalar: uncompressed, no error
        assert h["a_vec"].compression == "gzip"
        assert h["a_scalar"][()] == np.float64(3.14)


# --- provenance attrs: the PRODUCTION writer + the combine round-trip ---------
def test_dlaholder_save_results_writes_clustering_provenance(tmp_path):
    """The production writer (DLAHolder.save_results) must stamp the clustering
    provenance attrs from the holder's own fields when set."""
    from run_bayes_select import DLAHolder

    holder = object.__new__(DLAHolder)
    holder.results = _nan_heavy_results()
    holder.pair_prior_mode = "clustering"
    holder.dla_bias = 2.0
    fn = str(tmp_path / "prov_on.h5")
    holder.save_results(output_file=fn)
    with h5py.File(fn, "r") as h:
        assert h.attrs["pair_prior_mode"] == "clustering"
        assert float(h.attrs["dla_bias"]) == 2.0


def test_dlaholder_save_results_provenance_defaults_off(tmp_path):
    """A holder built without the clustering fields (e.g. object.__new__ / any
    legacy path) must still write the production default ('off', 2.0) via the
    getattr fallback — never raise, never silently omit the stamp."""
    from run_bayes_select import DLAHolder

    holder = object.__new__(DLAHolder)
    holder.results = _nan_heavy_results()  # no pair_prior_mode / dla_bias set
    fn = str(tmp_path / "prov_default.h5")
    holder.save_results(output_file=fn)
    with h5py.File(fn, "r") as h:
        assert h.attrs["pair_prior_mode"] == "off"
        assert float(h.attrs["dla_bias"]) == 2.0


def test_combine_propagates_clustering_provenance(tmp_path):
    """End-to-end: the clustering provenance written by the production writer
    must survive combine_processed_h5 into combined.h5, so a downstream reader
    can verify which prior produced a catalog. Guards the combine attr-copy
    (combine_processed_h5.py: copy pair_prior_mode/dla_bias from processed[0])."""
    from astropy.table import Table

    from run_bayes_select import DLAHolder
    from combine_processed_h5 import combine_processed_files

    survey, program = "main", "dark"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Two per-healpix processed files, each row-aligned on target_ids, each
    # stamped clustering/2.0 by the production writer.
    all_ids = []
    for hp, ids in ((100, [10, 11, 12, 13]), (200, [20, 21, 22, 23])):
        holder = object.__new__(DLAHolder)
        res = _nan_heavy_results()
        res["target_ids"] = np.asarray(ids, dtype=np.int64)
        holder.results = res
        holder.pair_prior_mode = "clustering"
        holder.dla_bias = 2.0
        holder.save_results(
            output_file=str(processed_dir / f"processed-{survey}-{program}-{hp}.h5")
        )
        all_ids.extend(ids)

    out = str(tmp_path / "combined.h5")
    target_catalog = Table({"TARGETID": np.asarray(all_ids, dtype=np.int64)})
    combine_processed_files(
        str(processed_dir), [100, 200], out, survey, program, target_catalog
    )

    with h5py.File(out, "r") as h:
        assert h.attrs["pair_prior_mode"] == "clustering"
        assert float(h.attrs["dla_bias"]) == 2.0
        assert h.attrs["combined_files"] == 2
        # rows from both healpix made it through
        assert h["target_ids"].shape[0] == 8
