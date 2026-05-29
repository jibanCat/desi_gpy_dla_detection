# Processed-h5 gzip compression

Added 2026-05-23. The per-spectrum GP-DLA result files (`processed-spectra-16-*.h5`
for mocks, `processed-main-dark-*.h5` for real/LOA) are written with **lossless
gzip compression**. This is a pure on-disk-encoding change: every dataset — every
value, including NaN fill — round-trips bit-identically, and all readers
(`qso_loader`, `calc_cddf`, `combine_processed_h5`) open via `h5py`, which
decompresses transparently. **No read-side change is required.**

## Why

The processed h5 is dominated by two arrays that scale with `num_dla_samples`:

| dataset | shape | dtype | raw size @100k samples |
|---|---|---|---|
| `sample_log_likelihoods_dla` | (n, num_samples, max_dlas) | f64 | ~3.6 GB |
| `base_sample_inds` | (n, max_dlas−1, num_samples) | i32 | ~1.4 GB |

In the headline **mock production config** (`FILTER_LOW_LIKELIHOOD=1`,
multi-DLA `MAX_DLAS≥3`, early-stop) most QMC samples are never evaluated, so
`sample_log_likelihoods_dla` is **~93–96 % NaN fill** — the same 8-byte NaN
pattern repeated millions of times. gzip's DEFLATE (LZ77 + Huffman) collapses
those identical byte-runs to almost nothing. The real values are kept exactly.

## Where (the code)

- **`process_helpers.py` → `_gzip_kwargs(value)`** — the single source of truth.
  Returns `{"compression": "gzip", "compression_opts": 4}` for array datasets
  (`ndim ≥ 1`, non-empty) and `{}` for scalars / empty arrays (gzip requires a
  chunked layout, impossible for a scalar — applying it would raise).
- Two writers call it on every `create_dataset`:
  - `process_helpers.py → save_results_to_hdf5()` — the standalone path
    (`run_bayes_select.py` direct use; writes a `spectrum_ids` axis).
  - **`run_bayes_select.py → DLAHolder.save_results()`** — the writer that
    actually persists **mock, DESI, and real/LOA production** output. It is
    called from `dlasearch.py:621` (`model.save_results(...)`), shared by both
    `dlasearch_mock` and `dlasearch_hpx` via `process_spectra_group`. Mock/LOA
    files use a `target_ids` axis (not `spectrum_ids`).

> Gotcha for future edits: the production per-healpix file is written by
> `DLAHolder.save_results`, **not** `save_results_to_hdf5`. They are separate
> code paths with different id-axis names (`target_ids` vs `spectrum_ids`).

The separate `loa_archive.py` spectra store is **not** affected by this change.

## Expected data size — the ratio tracks the NaN fraction, which depends on run mode

Compression gain is *not* universal; it scales with how much of the per-sample
array is NaN fill, which is set by the run configuration. Measured:

| run / config | per-sample NaN % | per-file size | ratio |
|---|---|---|---|
| **Mock V1 production** (FILTER=1, MAX_DLAS=4, 100k samples, early-stop) | ~93–96 % | ~5.0 GB → ~0.1–0.3 GB | **~15–25×** |
| Whole London-0 run, 1150 files | — | ~3.1 TB → **~0.2 TB** | ~15× |
| One-time in-place repack of the 815 existing files | — | **2.0 TB → 99 GB** | ~20× |
| **LLS single-absorber LOA** (50k samples, fully evaluated) | **0 %** | 42 MB → 38 MB | ~1.1× |
| Tiny LLS LOA healpix (few spectra) | 0 % | 0.4 MB → 0.4 MB | ~1.0× |

Takeaways:
- **FILTER + multi-DLA + early-stop** runs (the mock production baseline) are
  NaN-heavy → huge wins (15–25×).
- **Single-absorber / no-filter** runs (e.g. the real-LOA LLS catalogs) evaluate
  every sample → ~0 % NaN → only ~1.1×. gzip **never meaningfully grows** a file
  (worst case ~1.0×), so it is safe to leave on for all modes.
- Rule of thumb: compressed size ≈ raw size × (non-NaN fraction) + small overhead.

## One-time repack of legacy (uncompressed) files

Files written before this change can be compressed in place, losslessly:

```
PROCDIR=<run>/outputs/figures/processed \
  sbatch --export=ALL,PROCDIR slurm/greatlakes/production/repack_gzip.sh
```

It runs `h5repack -f GZIP=4` per file (16-way), structurally verifies the result
(`repack_verify.py`: opens + matching keys/shapes/dtypes), then atomically
replaces the original. Truncated/corrupt files are refused and left untouched, so
the repack doubles as a corruption scan (its `[FAIL]` list = the unreadable files).

## Tests

`tests/test_h5_compression.py` (9 tests): `_gzip_kwargs` array-vs-scalar behaviour;
both writers produce gzip-compressed datasets; NaN-aware lossless round-trip; a
NaN-heavy array shrinks > 3×; a scalar value writes uncompressed without error.
