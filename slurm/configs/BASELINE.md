# Baseline production settings (Y3 multi-DLA, BAL-included)

These are the hyperparameters every `*_y3.env` flavour config bakes in via
`_base.env`. They were the settings used for the 2025-09-12 LOA + London Y3
production runs (the historic `desi-loa-gpdla-20250912-desi-learned/` and
`desi-mock-gpdla-20250912-y3-learned-epoch920-filter/`).

Drop this file as `<OUTDIR>/BASELINE.md` after launching, so the run's settings
travel with the outputs.

## GP model

| Knob | Value | Notes |
|---|---|---|
| `LEARNED_FILE` | `learnlogs/model_epoch_920.h5` | Y3 GP trained model |
| `DLAMBDA` | 0.15 Å | rest-frame pixel spacing |
| `K` | 30 | GP rank |
| `MIN_LAMBDA / MAX_LAMBDA` | 911.75 / 1216.75 Å | rest-frame forest grid |
| `LOADING_MIN/MAX_LAMBDA` | 910 / 1550 Å | raw-spectrum load window |
| `NORMALIZATION_MIN/MAX_LAMBDA` | 1425 / 1475 Å | rest-frame median window for flux normalization |
| `NUM_FOREST_LINES` | 3 | Lyα + Lyβ + Lyγ |
| `NUM_LINES` | 3 | Voigt absorber lines |
| `MAX_NOISE_VARIANCE` | 9 | pixel-level mask |

## Mean-flux prior (Turner+2024)

| Knob | Value |
|---|---|
| `PREV_TAU_0` | 0.00246 |
| `PREV_BETA` | 3.62 |

## Absorber-model mode

| Mode | `MAX_DLAS` | `SINGLE_ABSORBER_MODEL` | `FILTER_LOW_LIKELIHOOD` | Use |
|---|---|---|---|---|
| Multi-DLA (`*_y3.env`)         | 3 | 0 | 1 | headline DLA catalog |
| LLS single (`*_y3_lls172.env`) | 1 | 1 | 0 | LLS/subDLA, NHI floor 17.2 |
| LLS single (`*_y3_lls190.env`) | 1 | 1 | 0 | LLS/subDLA, NHI floor 19.0 |

## QMC samples — **match the .mat file's row-count**

Note: this differs from the historic 2025-09-12 production which used a
`dla_samples_a03_100000.mat` (100k-row) file that no longer exists on disk.
Today's defaults pair with the still-available 10k file:

| Mode | `DLA_SAMPLES_FILE` | `NUM_DLA_SAMPLES` | `SUB_DLA_SAMPLES_FILE` | `NUM_SUBDLA_SAMPLES` |
|---|---|---|---|---|
| multi-DLA | `dla_samples_a03.mat` | **10 000** | `subdla_samples.mat` | 10 000 |
| LLS nhi172 | `pw_samples_a3_172_220_50000.mat` | 50 000 | `subdla_samples_a03_191_200_100000.mat` | 100 000 |
| LLS nhi190 | `pw_samples_a3_190_220_50000.mat` | 50 000 | `subdla_samples_a03_191_200_100000.mat` | 100 000 |

**Impact of 10 k vs 100 k DLA samples:** on the 16-LOA-TID cross-check
against the historic Y3 catalog, p(DLA), MAP z_DLA, and log NHI agree to
≤0.01, 0.001, 0.01 dex respectively. The 10 k cap mostly costs precision
on borderline-NHI sub-DLA-vs-DLA calls. Regenerate
`dla_samples_a03_100000.mat` (e.g. via `python gpy_dla_detection/generate_samples.py
--num_dla_samples 100000 --out data/dr12q/processed/dla_samples_a03_100000.mat`)
if you want bit-for-bit agreement with the historic catalog.

## BAL handling

| Knob | Value |
|---|---|
| `BALMASK` | `false` | BAL QSOs are **included**, no pixel masking |

## Parallelism

| Path | `parallel-files` | `MAX_WORKERS` | Cores busy | Notes |
|---|---|---|---|---|
| Production sbatch | 32 (`ntasks=32`) | 8 | 256 | One srun per file slice |
| Local (this repo's `slurm/run_local.sh`) | configurable (default 4) | 64 | parallel × max_workers | Mirrors srun parallelism with background `&` |

`desi-DLAGP.py` hardcodes `nproc_futures = 1` (line 551) — so a single
python process only processes ONE spectra-16 file at a time. Both paths
above get node-wide parallelism by launching multiple python processes,
not by tuning `max_workers` alone.

## Outputs

Each run writes per-file HDF5 (`processed-spectra-16-N.h5`) + per-range
catalog FITS (`dlacat-<release>-mockcat-<a>-<b>.fits`, or
`dlacat-<release>-<survey>-<program>-hpx-<a>-<b>.fits` for LOA) to `OUTDIR`.

Combine with:
```bash
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5" \
    [--mock]
```

Run purity/completeness:
```bash
# Mock (vs truth)
python examples/analyze_production_catalog.py \
    --catalog-dir "$OUTDIR" \
    --truth /path/to/dla_cat.fits  # or hcd_truth_cat.fits for Saclay/2LPT \
    --zcat  /path/to/zcat.fits \
    --truth-nhi-min 20.3 \
    --bal-cat /path/to/bal_cat.fits --no-bal \
    --out "$OUTDIR/purity_completeness.md"
```
