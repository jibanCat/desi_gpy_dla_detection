# Saclay mock-0 8f — v3_loa124 distribution-shift sanity check

> 2026-05-12. Inference complete. Validates v3 phase2_desi `2lpt_loa124_nohcd_nobal_wide`
> against London (cleanest 8f comparison so far).

## Setup

| Param | Value |
|---|---|
| Model | `null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5` (v3 phase2_desi) |
| Mock | Saclay v4.7.5 / mock-0 / juraLy8-124 |
| Spectra | 8 spectra-16 files (level2 slices 0..8), ~6700 QSOs |
| DLA samples | 50 000 PW14 (NHI ∈ [19, 22]) |
| τ-EB | objective=null, HCD mask OFF (default) |
| MAX_DLAS | 3, FILTER_LOW_LIKELIHOOD=1 |
| BAL | included in inference (no `--balmask`); excluded in molly via `--no-bal` |
| Truth | `hcd_truth_cat.fits` (Saclay HCD truth, NHI≥20.3) |
| Inference wall | ~45 min on this jupyter node (8 slices × 8 workers; relaunch needed for 5 of 8) |

## Headline P/C at SNR > 2 (lya_lyb window [911, 1216] Å rest)

### Saclay v3_loa124 (this run)

| P_DLA cut | Purity | Completeness |
|---:|---:|---:|
| ≥ 0.99    | **0.8707** | **0.7710** |
| ≥ 0.999   | 0.8845 | 0.7475 |
| ≥ 0.99999 | 0.9013 | 0.6768 |

(cat_post_cuts = 1381, truth_post_cuts = 533)

### Saclay baseline (production GP `model_epoch_920.h5`, NO τ-EB, NUM_DLA_SAMPLES=10k, PW10 [19,20] prior)

| P_DLA cut | Purity | Completeness |
|---:|---:|---:|
| ≥ 0.99    | 0.7883 | 0.8653 |
| ≥ 0.999   | 0.8111 | 0.8384 |
| ≥ 0.99999 | 0.8529 | 0.7811 |

(cat_post_cuts = 593, truth_post_cuts = 533; re-run molly at SNR>2 from `saclay0_y3/`)

### London v3_loa124 reference (HANDOFF.md)

| P_DLA cut | Purity | Completeness |
|---:|---:|---:|
| ≥ 0.99    | 0.8452 | 0.7661 |
| ≥ 0.999   | 0.8547 | 0.7398 |
| ≥ 0.99999 | 0.8872 | 0.6901 |

(cat_post_cuts = 1242, truth_post_cuts = 618)

## Distribution-shift verdict — Saclay v3 vs London v3

| P_DLA cut | London P / C | Saclay P / C | ΔP / ΔC |
|---:|---:|---:|---:|
| ≥ 0.99    | 0.8452 / 0.7661 | **0.8707 / 0.7710** | **+2.6pp / +0.5pp** |
| ≥ 0.999   | 0.8547 / 0.7398 | 0.8845 / 0.7475 | +3.0pp / +0.8pp |
| ≥ 0.99999 | 0.8872 / 0.6901 | 0.9013 / 0.6768 | +1.4pp / −1.3pp |

**No regression — v3_loa124 generalises off-distribution.** Saclay purity is uniformly higher
than London v3 (+1.4 to +3.0 pp). Completeness matches London to within ≤1.3 pp at every cut.
At the headline operating point (SNR>2, P_DLA≥0.99) Saclay is essentially as good as or better
than London on both axes (P +2.6pp, C +0.5pp).

## Saclay v3 vs Saclay baseline — same model swap, same mock

| P_DLA cut | Baseline P / C | v3_loa124 P / C | ΔP / ΔC |
|---:|---:|---:|---:|
| ≥ 0.99    | 0.7883 / 0.8653 | **0.8707 / 0.7710** | **+8.2pp / −9.4pp** |
| ≥ 0.999   | 0.8111 / 0.8384 | 0.8845 / 0.7475 | +7.3pp / −9.1pp |
| ≥ 0.99999 | 0.8529 / 0.7811 | 0.9013 / 0.6768 | +4.8pp / −10.4pp |

Caveat — the baseline run also differs in NUM_DLA_SAMPLES (10k vs 50k), DLA prior
(`dla_samples_a03.mat` [19.5, 22] vs `pw_samples_a3_190_220_50000.mat` [19, 22]),
and τ-EB OFF vs ON. So this column is a **PR-stack delta**, not a v3-only delta.
On London the same PR stack moved by +1.0 / +2.3 (HANDOFF baseline vs v3 entry);
here Saclay shows a much bigger purity/completeness trade — v3+50k_PW14+τ-EB
aggressively trims many ~85%-confidence baseline DLAs into the "not DLA" bin.

That can be ~half "ghost DLAs removed" (good — purity↑) and ~half "real DLAs lost
to prior-volume dilution" (bad — completeness↓), the mechanism documented in
`HANDOFF.md` §3 and `memory/project_prior_dilution_finding.md`.

## Run sequence notes

The first nohup launch of `slurm/run_local.sh slurm/configs/saclay0_y3.env` started
at 14:54:44 on `nid004210` (job 52877772 jupyter). 3 of 8 slices (file indices 1, 3,
5 → produced `processed-spectra-16-{1,3,9}.h5` + matching `dlacat-*-{1,3,5}-{2,4,6}.fits`)
completed in 10–19 min before the harness terminated the background task at ~15:17.
Orphan ProcessPoolExecutor workers (PGID == killed launcher's PGID) hung sleeping
and were killed before relaunch.

The 5 remaining slices (0, 2, 4, 6, 7) were relaunched via
`/pscratch/sd/j/jibancat/prod533_5k_20260511/relaunch_saclay_v3.sh` at 15:20:45 and
finished at 16:05:09 (44.5 min wall) — the longest single-file slice in the wave
was the bottleneck.

All 8 dlacat-v4.7.5-mockcat-*-*.fits and 8 processed-spectra-16-*.h5 files exist
in this OUTDIR. Molly was run on the joint catalog.

## File layout

```
saclay0_v3_loa124_pw14_tau_eb/
├── dlacat-v4.7.5-mockcat-{0,1,2,3,4,5,6,7}-{1,2,3,4,5,6,7,8}.fits   # 8 GP-DLA catalogs
├── processed/processed-spectra-16-{0,1,2,3,8,9,10,11}.h5             # 8 per-QSO posterior h5
├── figures_molly/
│   ├── snr2_pdla0.99/{lya_only,lya_lyb}/    # P=0.8707, C=0.7710 in lya_lyb
│   ├── snr2_pdla0.999/{lya_only,lya_lyb}/   # P=0.8845, C=0.7475
│   └── snr2_pdla0.99999/{lya_only,lya_lyb}/ # P=0.9013, C=0.6768
├── logs/local_*.log + relaunch_*.log
└── RUN_SETTINGS.md  (auto-emitted by run_local.sh)
```

## Reproduce

```bash
# RECOMMENDED — single command (8-file run, no relaunch needed)
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main

cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
export LEARNED_FILE=/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5
export DLA_SAMPLES_FILE=data/dr12q/processed/pw_samples_a3_190_220_50000.mat
export NUM_DLA_SAMPLES=50000
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
export MAX_DLAS=3
export FILTER_LOW_LIKELIHOOD=1

bash slurm/run_local.sh slurm/configs/saclay0_y3.env \
    --outdir /pscratch/sd/j/jibancat/prod533_5k_20260511/saclay0_v3_loa124_pw14_tau_eb/ \
    --window 8 --parallel-files 8 --max-workers 8

# Then:
python examples/molly_faithful_pc_plots.py \
    --catalog-dir /pscratch/sd/j/jibancat/prod533_5k_20260511/saclay0_v3_loa124_pw14_tau_eb \
    --truth /global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/hcd_truth_cat.fits \
    --bal-cat /global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/bal_cat.fits \
    --no-bal --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out /pscratch/sd/j/jibancat/prod533_5k_20260511/saclay0_v3_loa124_pw14_tau_eb/figures_molly/snr2_pdla0.99
```
