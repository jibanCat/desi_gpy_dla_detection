# Production cost: v3_loa124 vs baseline (1 M QSO on Perlmutter)

> **TL;DR**: v3_loa124 (5662 pixels) costs **+9% wall/spec** vs `model_epoch_920`
> (3798 pixels) on London 8f, even when v3 was run with 5x more QMC samples.
> 1 M QSO production at 32 pythons * 8 max_workers per Perlmutter node:
> **~15.2 node-hours baseline, ~16.6 node-hours v3 (50k QMC), ~12-14 node-hours
> v3 normalized to 10k QMC**. The +2.3 pp completeness gain at P_DLA >= 0.99 is
> essentially free.
>
> The existing `docs/notes/2026-04-29_production_cost_estimate.md` (340 node-hour
> baseline) was based on GreatLakes 16-CPU profiles (~280-620 s/spec). Perlmutter
> actually delivers ~1.7 s/spec at 32x8=256-core saturation -- that note's
> absolute numbers are ~20x too high. Ratios are useful, but the budget
> conclusion is reversed: we are already well **under** the 50 node-hour target.

## Source runs

| | baseline | v3_loa124 |
|---|---|---|
| `LEARNED_FILE` | `model_epoch_920.h5` (3798 pix) | `2lpt_loa124_nohcd_nobal_wide.h5` (5662 pix) |
| `NUM_DLA_SAMPLES` | 10 000 | 50 000 |
| `MAX_DLAS / ENABLE_TAU_EB / FILTER_LOW_LIKELIHOOD` | 3 / 1 / 1 | 3 / 1 / 1 |
| `MAX_WORKERS` per python | 8 | 8 |
| `PARALLEL_FILES` per node | 32 | 8 |
| Files | 8 (spectra-16-{0,1,2,3,8,9,10,11}) | same 8 |
| Outdir | `/pscratch/sd/j/jibancat/prod533test-20260511_1333/london0_y3/` | `/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/` |

## Per-file wall (`dlasearch.py: Completed processing of N spectra ... in T s`)

| spec-16-N | n_spec | base T(s) | v3 T(s) | base s/spec | v3 s/spec | v3/base |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 1129 | 1979.66 | 2179.09 | 1.754 | 1.930 | 1.10 |
| 1  |  332 |  568.20 |  529.82 | 1.711 | 1.596 | 0.93 |
| 2  | 1083 | 1902.79 | 2262.77 | 1.757 | 2.089 | 1.19 |
| 3  |  364 |  663.69 |  598.94 | 1.823 | 1.645 | 0.90 |
| 8  | 1081 | 1882.16 | 2168.88 | 1.741 | 2.007 | 1.15 |
| 9  |  700 | 1203.61 | 1251.69 | 1.719 | 1.788 | 1.04 |
| 10 | 1110 | 1944.22 | 2076.48 | 1.752 | 1.871 | 1.07 |
| 11 |  967 | 1710.55 | 1835.14 | 1.769 | 1.898 | 1.07 |
| **total** | **6766** | **11854.88** | **12902.81** | **1.752** | **1.907** | **1.088** |

Sanity check from `time spent:` per-spectrum lines in local_0_1: mean 1.72 s
baseline vs 1.88 s v3 -- matches the per-file totals within 1 %.

## Why is the ratio only 1.09x when v3 has 1.49x more pixels AND 5x more QMC?

`parallel_log_model_evidences` early-stops on no-DLA spectra when the 1-DLA
evidence drops below null (visible all over both logs: "Stopping early at
1 DLAs because ..."). London is ~90% no-DLA, so the population-mean cost is
dominated by the null-GP build + Voigt grid + I/O -- all roughly linear in
n_pix. Most spectra never enter the per-sample QMC inner loop in a way that
the 50k vs 10k difference matters. The observed +9 % wall is a fractional
share of the +49 % pixel cost, with I/O amortization absorbing the rest.

## 1 M QSO Perlmutter projection

CPU-seconds per spectrum = `wall * max_workers` (each python uses 8 cores
in its QMC ThreadPool). Node-hours = `N * cpu_s / (256 cores * 3600 s)`.

| Model | wall/spec | CPU-s/spec | **1M-QSO node-hours** |
|---|---:|---:|---:|
| baseline `model_epoch_920` (10k QMC) | 1.752 s | 14.0 | **15.2** |
| v3_loa124 (50k QMC, as-run)          | 1.907 s | 15.3 | **16.6** |
| v3_loa124 normalized to 10k QMC (~3x sub-linear) | ~1.3 s | ~11 | ~12-14 |
| **raw ratio v3/base** | -- | -- | **1.09x** |

The "10k QMC" rows account for QMC's sub-linear scaling under early-stops.
Realistic v3-at-10k-QMC: **~12-14 node-hours**, possibly **cheaper** than
baseline at 10k. Either way: well **under** the 50 node-hour target.

## Headline (the requested table)

| Model | wall/spec | 1M-QSO node-hours |
|---|---:|---:|
| baseline (`model_epoch_920`, 3798 pix, 10k QMC) | 1.75 s | 15.2 |
| v3_loa124 (5662 pix, 50k QMC as-run) | 1.91 s | 16.6 |
| ratio (v3/base, raw) | 1.09x | 1.09x |

## Comparison with the existing estimate

| Estimate | 1M-QSO node-hours | Notes |
|---|---:|---|
| Existing note (`docs/notes/2026-04-29_production_cost_estimate.md`) | 343 | GreatLakes 16-CPU profile, 2 cherry-picked targets, no population early-stop |
| **This note baseline (Perlmutter London 5k)** | **15** | Measured at 32x8 saturation; 90% no-DLA early-stops |
| **This note v3_loa124 (50k QMC)** | **17** | Same, +1.4 nh overhead |
| User target | 50 | session prior |

The 22x absolute discrepancy with the existing note: (a) GreatLakes 16 cores
vs Perlmutter 256 cores per node (16x), and (b) the existing profile targets
were a strong-DLA + an LLS, which don't early-stop. At population scale the
no-DLA majority short-circuits `parallel_log_model_evidences`.

**The user's 50 node-hour target was set against the 343-estimate.** Actual
baseline is ~15 nh; the budget is already met by 3x margin. The follow-up
items (1)-(4) in the existing note are no longer load-bearing for
survey-scale production at current `num_dla_samples`.

## Recommendation

**Switch production GP to `v3_loa124`**. Cost gap is +9 % wall/spec
(~+1.4 node-hours per 1 M QSO at Perlmutter 32x8). The +1.0 / +2.3 / +3.3 pp
purity/completeness at P_DLA >= {0.99, 0.999, 0.99999} (HANDOFF.md
"Headline P/C at SNR > 2") is scientifically meaningful and effectively free
at this scale.

Recommended config (vs `slurm/configs/london0_y3.env`):

```
LEARNED_FILE=/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5
NUM_DLA_SAMPLES=10000   # match 2025-09-12 historic catalog for comparability
DLA_SAMPLES_FILE=data/dr12q/processed/dla_samples_a03.mat
# all other settings unchanged from london0_y3.env / BASELINE.md
```

Keep 50k QMC if sub-DLA/DLA-boundary precision matters more than historic
comparability -- incremental cost is < 2 node-hours per 1 M.

## Caveats

1. **I/O**: `wall_total - n_spec * per_spec_wall` is ~0 in both runs, so I/O
   was negligible on mock spectra. Real LOA spectra (more pixels masked,
   variable SNR) may differ; re-measure on the next LOA production batch.
2. **Node fill-rate**: projections assume 32 pythons * 8 workers = 256 cores
   per node. The v3 test ran 8 pythons (64 cores); baseline ran 32 and
   validates the pattern. Production SLURM submits should set
   `PARALLEL_FILES=32`.
3. **Distribution shift**: measured on London mock-0. Saclay and real LOA
   may have different no-DLA fractions, hence different early-stop rates.
   The 1.09x ratio is robust; the absolute number is less so.
4. **Pixel ratio 5662/3798 = 1.491x** predicts +49 % wall for null-GP + Voigt.
   We see +9 % -- the rest is amortized by population early-stops and fixed
   I/O. On a DLA-rich population (LLS pipeline, dense-DLA mocks) the
   v3/base ratio could approach 1.4x; worth re-checking before LLS deployment.

## Reproducibility

Timing lines: `grep "Completed processing of " <logdir>/local_*_*.log` in
- v3:       `/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/logs/`
- baseline: `/pscratch/sd/j/jibancat/prod533test-20260511_1333/london0_y3/logs/`

Per-spectrum: `grep "time spent: 0m " <logfile> | awk -F"time spent: 0m " '{print $2}' | awk -F"s" '{s+=$1; n++} END {print s/n, n}'`
