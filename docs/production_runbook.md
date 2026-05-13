# GP-DLA Production Runbook — NERSC Perlmutter

> **Audience**: next-Claude in this repo. You are launching full GP-DLA inference
> over a dataset (mock or real LOA). This document tells you the exact paths,
> commands, hyperparameters, expected wall-time, and expected P/C numbers.
>
> **Written**: 2026-05-13, branch `production_533`, after the Var[Δ_marg] gating
> diagnostic concluded production N=50k QMC is sampling-converged
> (see `docs/notes/2026-05-13_var_delta_marg_diagnostic.md`).
>
> **DO NOT submit new jobs without reading §10 (Gotchas) first** — the production
> sbatch scripts `slurm/submit_desi_{mock,loa}.sh` **do not forward**
> `--enable_tau_eb` / `--tau_eb_objective` / `--early_stop_mode` to the python
> CLI. You will silently lose τ-EB in production unless you patch them or use
> `slurm/run_local.sh` (which does forward them). See §10.1.

---

## 0. WINNING BASELINE (copy-paste this)

This is the *current best-known config* (2026-05-13). Validated on London mock-0
(8 spectra-16 files, ~6.6k QSOs) and Saclay mock-0 (8 files, ~6.7k QSOs).
Cross-validated by `docs/runs/2026-05-12_saclay_v3_loa124_results.md`.

```bash
# v3 GP model (phase2_desi, 2LPT loa124 training filter, HCD- and BAL-excluded)
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"

# QMC sample grid: PW14 prior over NHI ∈ [19, 22], 50k samples (file rowcount MUST match)
export DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat"
export NUM_DLA_SAMPLES=50000

# Sub-DLA: keep production 10k default (the 100k file with NHI floor 19.1 is
# recommended for headline catalogs; the 10k default is what's been validated
# at scale).
# To switch: export SUB_DLA_SAMPLES_FILE=".../subdla_samples_a03_191_200_100000.mat"; export NUM_SUBDLA_SAMPLES=100000

# Per-spectrum empirical-Bayes τ_eff (PR #5) — ON
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null              # cheap; matched by "dla" objective on canonical targets
# (default tau_eb_factors = 0.5 1.0 1.5 2.0 3.0 4.0 5.0 6.0 — keep)

# Multi-DLA early-stop policy: default "baseline" matches the validated production
# configuration. Variants A/D are under evaluation (see early_stop_fix_test/)
# but have NOT been promoted to production. KEEP "baseline".
export EARLY_STOP_MODE=baseline

# DLA-mode (multi-DLA catalog, NHI ≥ 20.3)
export MAX_DLAS=3
export SINGLE_ABSORBER_MODEL=0
export FILTER_LOW_LIKELIHOOD=1

# BAL: included in inference (no --balmask). BAL exclusion is applied at eval time.
export BALMASK=false
```

**Expected P/C at SNR_RED > 2, P_DLA ≥ 0.99, full forest λ_rf ∈ [911, 1216] Å,
BAL excluded** (`docs/runs/2026-05-12_saclay_v3_loa124_results.md`):

| Mock | Purity | Completeness | n_cat | n_truth |
|---|---:|---:|---:|---:|
| London 8f (this baseline) | **0.8452** | **0.7661** | 1242 | 618 |
| Saclay 8f (this baseline) | **0.8707** | **0.7710** | 1381 | 533 |

At stricter cuts:

| Cut | London P / C | Saclay P / C |
|---|---|---|
| ≥ 0.99    | 0.8452 / 0.7661 | 0.8707 / 0.7710 |
| ≥ 0.999   | 0.8547 / 0.7398 | 0.8845 / 0.7475 |
| ≥ 0.99999 | 0.8872 / 0.6901 | 0.9013 / 0.6768 |

**Headline: at SNR_RED > 2 nothing currently passes 85/85 jointly.** The
completeness gap is intrinsic at low SNR (model-side limiter — sampling is
already converged; see `docs/notes/2026-05-13_var_delta_marg_diagnostic.md`).

Per `memory/feedback_snr_canonical.md`: report **SNR_RED > 2** as headline,
acceptable > 1. **Do NOT** report SNR > 6 as primary.

---

## 1. Datasets

### 1.1 Per-dataset spectrum counts

Counted 2026-05-13 from the zcat.fits and the spectra-16 directory tree.
Production cut applied: Z ≥ 1.96 (see `desi-DLAGP.py:read_in_each_plots_*`).

| Dataset      | zcat path                                                                                                  | rows (zcat) | Z ≥ 1.96   | spectra-16 files |
|--------------|------------------------------------------------------------------------------------------------------------|------------:|-----------:|-----------------:|
| **2LPT-0**   | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/zcat.fits` | 1 213 217 | 977 268 | 1 150 |
| **London-0** | `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits`         | 1 217 878 | 982 313 | 1 150 |
| **London-1** | `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-1/jura-124/zcat.fits`         | 1 218 470 | 982 672 | 1 148 |
| **Saclay-0** | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/zcat.fits` | 1 221 478 | 985 282 | 1 127 |
| **Saclay-1** | `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-1/jura-124/zcat.fits`    | 1 221 416 | 984 074 | 1 130 |
| **LOA real** | `/global/cfs/cdirs/desi/science/lya/y3/loa/catalogs/QSO_cat_loa_main_dark_healpix_v2-altbal-20241115.fits`  | 2 776 520 | 1 000 854 | 16 586 unique HPXPIXEL |

**Additional cuts applied at inference time** (in `desi-DLAGP.py` / `dlasearch.py`):
- BAL inclusion (no `--balmask`) — but `BI_CIV > 0` BALs are excluded at *eval time*
- Per-spectrum pixel masking: `MAX_NOISE_VARIANCE=9`, `NUM_FOREST_LINES=3`
- `MIN_LAMBDA / MAX_LAMBDA` rest-frame window 911.75 / 1216.75 Å — spectra with
  zero in-window pixels are skipped
- SNR_RED > 1 (acceptable) / > 2 (headline) cut applied at *catalog cuts*, not
  inference time. All spectra are inferred; downstream P/C scripts apply the cut.

### 1.2 Saclay mock-0 vs mock-1 subdir naming gotcha

Saclay mock-0 lives in `juraLy8-124/`; Saclay mock-1 lives in `jura-124/`.
Note: `Ly8` vs no `Ly8`. Both config files already encode this correctly.

### 1.3 Truth catalogs (for P/C eval)

| Dataset | Truth file |
|---|---|
| London-0 / 1 | `<mockdir>/dla_cat.fits` |
| Saclay-0 / 1 | `<mockdir>/hcd_truth_cat.fits` |
| 2LPT-0 | `<mockdir>/dla_cat.fits` (verify) |
| LOA real | no truth — P/C not applicable; use eBOSS overlap or DR9/DR12 concordance for sanity-check |

---

## 2. Two run modes per dataset

### 2.1 Multi-DLA mode (the headline DLA catalog, NHI ≥ 20.3)

| Knob                    | Value                       |
|-------------------------|-----------------------------|
| `MAX_DLAS`              | 3                           |
| `SINGLE_ABSORBER_MODEL` | 0                           |
| `FILTER_LOW_LIKELIHOOD` | 1                           |
| `DLA_SAMPLES_FILE`      | `pw_samples_a3_190_220_50000.mat` (PW14, NHI ∈ [19, 22], 50k) |
| `NUM_DLA_SAMPLES`       | 50000                       |
| `SUB_DLA_SAMPLES_FILE`  | `subdla_samples.mat` (10k) — production-validated |
| `NUM_SUBDLA_SAMPLES`    | 10000                       |

Configs (use these as-is — they `source _base.env`):
| Dataset    | Config file                                          | Outer loop |
|------------|------------------------------------------------------|-----------:|
| 2LPT-0     | `slurm/configs/2lpt0_y3.env`                         | 0..1150 step 64 |
| London-0   | `slurm/configs/london0_y3.env`                       | 0..1150 step 64 |
| London-1   | (none yet — copy london0 and swap `mock-0` → `mock-1`, update `OUTER_MAX_INDEX=1148`) | 0..1148 step 64 |
| Saclay-0   | `slurm/configs/saclay0_y3.env`                       | 0..1127 step 64 |
| Saclay-1   | (none yet — copy saclay0 and swap path; OUTER=1130)  | 0..1130 step 64 |
| LOA real   | `slurm/configs/loa_y3.env`                           | 0..16519 step 1664 |

### 2.2 LLS single-absorber mode (sub-DLA + DLA, NHI ∈ [17.2, 22] or [19, 22])

| Knob                    | LLS NHI≥17.2                | LLS NHI≥19.0                |
|-------------------------|-----------------------------|-----------------------------|
| `MAX_DLAS`              | 1                           | 1                           |
| `SINGLE_ABSORBER_MODEL` | 1                           | 1                           |
| `FILTER_LOW_LIKELIHOOD` | 0                           | 0                           |
| `DLA_SAMPLES_FILE`      | `pw_samples_a3_172_220_50000.mat` | `pw_samples_a3_190_220_50000.mat` |
| `NUM_DLA_SAMPLES`       | 50000                       | 50000                       |
| `SUB_DLA_SAMPLES_FILE`  | `subdla_samples_a03_191_200_100000.mat` (100k, NHI ∈ [19.1, 20]) | same |
| `NUM_SUBDLA_SAMPLES`    | 100000                      | 100000                      |
| `BATCH_SIZE`            | 6250                        | 6250                        |

Configs:
| Dataset  | NHI 17.2 config                            | NHI 19.0 config                            |
|----------|--------------------------------------------|--------------------------------------------|
| 2LPT-0   | `slurm/configs/2lpt0_y3_lls172.env`        | `slurm/configs/2lpt0_y3_lls190.env`        |
| London-0 | `slurm/configs/london0_y3_lls172.env`      | `slurm/configs/london0_y3_lls190.env`      |
| Saclay-0 | `slurm/configs/saclay0_y3_lls172.env`      | `slurm/configs/saclay0_y3_lls190.env`      |
| LOA real | `slurm/configs/loa_y3_lls172.env`          | `slurm/configs/loa_y3_lls190.env`          |

LLS configs are **NOT** updated to v3 model + τ-EB. To deploy the WINNING BASELINE
in LLS mode, override via env vars before launching (see §6.3).

---

## 3. All hyperparameters that can be tuned

### 3.1 GP model (rarely change)

| Knob | Default (Y3) | Meaning |
|---|---|---|
| `LEARNED_FILE` | `learnlogs/model_epoch_920.h5` (production); `2lpt_loa124_nohcd_nobal_wide.h5` (v3, **recommended**) | Trained GP weights (μ, M, log_omega, log_c_0, log_tau_0, log_beta). Pre-trained, you do not retrain in production. |
| `DLAMBDA` | 0.15 Å | Rest-frame pixel spacing of the GP grid. Must match the trained model. |
| `K` | 30 | GP rank. Must match the trained model. |
| `MIN_LAMBDA` / `MAX_LAMBDA` | 911.75 / 1216.75 Å | Rest-frame forest grid edges. |
| `LOADING_MIN_LAMBDA` / `LOADING_MAX_LAMBDA` | 910 / 1550 Å | Raw-spectrum load window (must contain the normalization band). |
| `NORMALIZATION_MIN_LAMBDA` / `NORMALIZATION_MAX_LAMBDA` | 1425 / 1475 Å | Rest-frame median window for flux normalization. |
| `NUM_FOREST_LINES` | 3 | Number of forest Lyman lines (Lyα, β, γ). |
| `NUM_LINES` | 3 | Number of Lyman lines in the Voigt absorber model. |
| `MAX_NOISE_VARIANCE` | 9 | Pixel-level mask (pixels with noise² > 9 dropped). |

### 3.2 Mean-flux prior (Turner+2024) — production default

| Knob | Value | Meaning |
|---|---|---|
| `PREV_TAU_0` | 0.00246 | τ_0 in τ_eff = τ_0 (1+z)^β. Per-spectrum τ-EB *seeds* from this. |
| `PREV_BETA`  | 3.62  | β     in τ_eff = τ_0 (1+z)^β. |

### 3.3 Mode / sampling knobs

| Knob | Default | Meaning |
|---|---|---|
| `MAX_DLAS` | 3 (multi-DLA), 1 (LLS) | Max absorbers in the multi-DLA recursion. |
| `SINGLE_ABSORBER_MODEL` | 0 (multi-DLA), 1 (LLS) | If 1: only the 1-absorber model is evaluated; no sub-DLA branch. |
| `FILTER_LOW_LIKELIHOOD` | 1 (multi-DLA), 0 (LLS) | If 1: truncated-sampler shortcut over QMC samples. |
| `DLA_SAMPLES_FILE` | `dla_samples_a03.mat` (10k) | QMC sample grid. **`NUM_DLA_SAMPLES` MUST equal the .mat row count.** |
| `NUM_DLA_SAMPLES` | 10000 (default), 50000 (WINNING BASELINE) | Row count of `DLA_SAMPLES_FILE`. Today's Var[Δ_marg] diagnostic shows N=50k is already sampling-converged. |
| `SUB_DLA_SAMPLES_FILE` | `subdla_samples.mat` (10k) | Sub-DLA QMC grid. |
| `NUM_SUBDLA_SAMPLES` | 10000 | Row count of sub-DLA file. |

**Sample file rowcounts (must match `NUM_*_SAMPLES`):**

| File                                                     | rows  | NHI prior        |
|----------------------------------------------------------|------:|------------------|
| `data/dr12q/processed/dla_samples_a03.mat`               | 10000 | (legacy) [19.5, 22], extrap to 23 |
| `data/dr12q/processed/pw_samples_a3_190_220_50000.mat`   | 50000 | PW14 [19.0, 22.0] |
| `data/dr12q/processed/pw_samples_a3_172_220_50000.mat`   | 50000 | PW14 [17.2, 22.0] |
| `data/dr12q/processed/subdla_samples.mat`                | 10000 | sub-DLA [19.5, 20], extrap to 23 |
| `data/dr12q/processed/subdla_samples_a03_191_200_100000.mat` | 100000 | sub-DLA [19.1, 20], extrap to 22.6 |

### 3.4 τ-EB (per-spectrum empirical Bayes; PR #5)

| Knob | Default | Meaning |
|---|---|---|
| `ENABLE_TAU_EB`   | 0 (CLI default) — **set 1 for WINNING BASELINE** | Per-spectrum τ_0 fit. |
| `TAU_EB_FACTORS`  | `(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)` | Grid factors multiplied by `PREV_TAU_0`. |
| `TAU_EB_APPLY_HCD_MASK` | 0 | HCD masking during τ-fit. At scale the mask over-corrects (see `docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md`). **Keep 0.** |
| `TAU_EB_OBJECTIVE` | `"null"` | `"null"` (cheap, K=8 null-GP rebuild) or `"dla"` (more rigorous, ~5× cost). |

### 3.5 Early-stop policy (new today; commit `2c499a8`)

| Knob | Default | Meaning |
|---|---|---|
| `EARLY_STOP_MODE` | `"baseline"` | `"baseline"`: historical penalized-likelihood-vs-null heuristic. `"A"`: disable null-early-stop entirely. `"D"`: compare pre-Occam likelihood to null. |

**Status**: variants A and D have inference complete on London 8f as of today but
NO production P/C measurement yet. **Keep `baseline` for production until A/D
P/C are published.** Track `prod533_5k_20260511/early_stop_fix_test/RESULTS.md`.

### 3.6 BAL handling

| Knob | Default | Meaning |
|---|---|---|
| `BALMASK` | `false` | If `true`: pass `--balmask`, masks BAL absorption pixels. Production has NEVER used BAL masking; BAL exclusion is applied at eval time only. |

### 3.7 Parallelism

| Knob | Default | Meaning |
|---|---|---|
| `MAX_WORKERS` | 8 | Inner-loop ThreadPool workers (per python process), used for QMC sample evaluation. |
| `BATCH_SIZE`  | 1250 (multi-DLA), 6250 (LLS) | QMC sample batch size for memory chunking. |
| `PARALLEL_FILES` (run_local.sh only) | 4 (script default), 32 (production) | Number of python processes per node. |
| `--ntasks` (sbatch) | 32 | Number of srun python procs per node. Each does one spectra-16 (mock) or 52-healpix (LOA) chunk at a time. |
| `--cpus-per-task` | 8 | = `MAX_WORKERS`. |
| Per-node total cores | 32 × 8 = **256** | Perlmutter CPU node = 256 cores. |

**Per `RECOMMENDATIONS.md` §6: PARALLEL_FILES=32 × MAX_WORKERS=8 is already optimal.**
Inner-thread parallelism saturates quickly; to go faster, scale to more nodes.

### 3.8 Outer launcher loop

| Knob | Meaning |
|---|---|
| `OUTER_MAX_INDEX` | Max index of the file/healpix axis (set per dataset). |
| `OUTER_STEP`      | Stride per sbatch job (= number of files per sbatch). 64 for mocks, 1664 for LOA. |
| `OUTER_WINDOW`    | Inner chunk-size *inside* one sbatch (typically `step - 2`). |

---

## 4. Wall-time / node-hour estimates per dataset

### 4.1 Source data: prod533 5k London/Saclay v3 runs

Per-spectrum throughput from `prod533_5k_20260511/{london,saclay0}_v3_loa124_pw14_tau_eb/logs/*.log`
(WINNING BASELINE config, 32 × 8 saturation):

| Source              | Wall/spec | Notes |
|---------------------|----------:|---|
| London 8f, v3, 50k QMC, τ-EB on | **1.91 s** | 6766 spec / 12 902 s total |
| Saclay 8f, v3, 50k QMC, τ-EB on | **1.97 s** | 6690 spec / 13 667 s total |
| London 8f, baseline (model_epoch_920, 10k, no τ-EB) | 1.75 s | 6766 spec / 11 855 s total |
| LOA real, LLS-mode (legacy 51695* jobs) | **~7.3 s** | dense-target population, no early-stops |

Both v3 mock runs match (~1.9 s/spec). LLS-mode is ~4× slower because the
single-absorber model doesn't early-stop on no-DLA spectra.

### 4.2 Per-dataset estimates (WINNING BASELINE, multi-DLA mode)

Cores per node = 256. `cpu_seconds / spec = wall * MAX_WORKERS = wall × 8`.
`node_hours = N_spec × cpu_s / (256 × 3600)`.

| Dataset    | N_spec (Z≥1.96) | wall/spec assumption | total CPU-s   | **node-hours** | wall on 1 node | wall on 36 nodes (sbatch fan-out) |
|------------|-----------------|---------------------:|--------------:|---------------:|---------------:|---------------------:|
| 2LPT-0     | 977 268         | 1.95 s (v3-Saclay)   | 15.25 M       | **16.5**       | 132 h ≈ 5.5 d   | ~3.7 h |
| London-0   | 982 313         | 1.91 s (v3-London)   | 15.01 M       | **16.3**       | 130 h ≈ 5.4 d   | ~3.6 h |
| London-1   | 982 672         | 1.91 s               | 15.02 M       | **16.3**       | 130 h ≈ 5.4 d   | ~3.6 h |
| Saclay-0   | 985 282         | 1.97 s               | 15.53 M       | **16.9**       | 134 h ≈ 5.6 d   | ~3.7 h |
| Saclay-1   | 985 282         | 1.97 s               | 15.53 M       | **16.9**       | 134 h ≈ 5.6 d   | ~3.7 h |
| LOA real   | 1 000 854       | 2.5 s (estimate; +30 % v3 cost for variable real-data masking) | 20.02 M | **21.7** | 174 h ≈ 7.2 d | ~4.8 h |

**Total all 6 datasets ≈ 105 node-hours** for multi-DLA mode. Well under any
plausible NERSC budget.

### 4.3 LLS-mode estimates (3.8× the cost per spectrum)

LLS mode uses `MAX_DLAS=1, SINGLE_ABSORBER_MODEL=1, FILTER_LOW_LIKELIHOOD=0` —
no early-stops, and 50k samples → ~3.8× wall/spec from the 51695* job logs.

| Dataset    | wall/spec | **node-hours** | wall on 36 nodes |
|------------|----------:|---------------:|---------------------:|
| 2LPT-0     | ~7.3 s    | **~62**        | ~14 h |
| London-0   | ~7.3 s    | **~62**        | ~14 h |
| Saclay-0   | ~7.3 s    | **~63**        | ~14 h |
| LOA real   | ~9 s      | **~78**        | ~16 h |

**Two LLS variants per dataset (nhi172 + nhi190)** doubles this — budget
**~140 node-hours per dataset** if running both, **~70** if running only one.

### 4.4 Sanity check vs older estimate

`docs/notes/2026-04-29_production_cost_estimate.md` claimed **343 node-hours** for
1 M QSO. That estimate was from GreatLakes (16-core profile) on two cherry-picked
*strong-DLA / LLS targets* that don't trigger early-stops. Perlmutter at 256-core
saturation with the population-mean early-stop rate (90 % no-DLA) is **~20× cheaper**
in practice. See `prod533_5k_20260511/v3_production_cost.md` for the reconciliation.

---

## 5. Multi-node parallelism strategy

### 5.1 The math

The outer driver `slurm/launch.sh` submits **one sbatch per `OUTER_STEP` files**.
Each sbatch is 1 node, 32 srun python processes, each handling ~`OUTER_WINDOW / 32` files.

For a mock dataset with 1150 spectra-16 files and `OUTER_STEP=64, OUTER_WINDOW=62`:
- Number of sbatch jobs = ⌈1150 / 64⌉ = **18 sbatch jobs**
- Each sbatch = 1 node × 32 srun python × ~2 files per srun
- Wall per sbatch ≈ (62 / 32) × file_wall ≈ 2 × ~30 min ≈ **~60 min per sbatch**

So **per mock dataset**: 18 sbatch jobs in parallel ≈ 1 hour wall-clock if queue
allows, with total cost ≈ 18 node-hours (within ~10% of the 16.5 estimate).

For LOA (1 M QSOs, 16519 healpix, `OUTER_STEP=1664, OUTER_WINDOW=1612`):
- Number of sbatch jobs = ⌈16519 / 1664⌉ = **10 sbatch jobs**
- Each sbatch = 1 node × 32 srun × 52 healpix
- Wall per sbatch ≈ (1612 / 32) × ~5 min/healpix ≈ ~4 h
- → **~22 node-hours, ~4 h wall** if all 10 run concurrently

### 5.2 Finer-grain parallelism

Override `--window` and `--end` to split sbatches into smaller chunks:
```bash
# Mock: 36 sbatches of 32 files each instead of 18 of 64
bash slurm/launch.sh slurm/configs/london0_y3.env --window 32 --end 1152
```
Doesn't reduce node-hours but lets more jobs run concurrently if queue is wide.

### 5.3 Queue selection

Perlmutter `regular` queue: max 12 h wall. All per-sbatch estimates above fit
inside 12 h with 2-3× margin. For LOA real, `--time=08:00:00` is already in
`slurm/submit_desi_loa.sh`. For mocks, `--time=05:00:00` in
`slurm/submit_desi_mock.sh`. Both are conservative.

### 5.4 Recommended dataset launch sequence

If you want all 6 datasets done in minimum wall-time and the queue allows it:

```bash
# Multi-DLA mode for all four mocks (60 sbatches total, parallel):
bash slurm/launch.sh slurm/configs/2lpt0_y3.env     --outdir /pscratch/sd/j/jibancat/desi-mock-2lpt0-prod-$(date +%Y%m%d)/
bash slurm/launch.sh slurm/configs/london0_y3.env   --outdir /pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)/
bash slurm/launch.sh slurm/configs/saclay0_y3.env   --outdir /pscratch/sd/j/jibancat/desi-mock-saclay0-prod-$(date +%Y%m%d)/
bash slurm/launch.sh slurm/configs/loa_y3.env       --outdir /pscratch/sd/j/jibancat/desi-loa-prod-$(date +%Y%m%d)/
```

This launches 72 sbatch jobs (18 × 4 mocks + 10 × LOA = 82 actually, but several
will queue). At 10-20 concurrent jobs typical NERSC dispatch, completion in
6-12 hours wall-clock for the full multi-DLA stack.

---

## 6. Exact launch commands

### 6.1 Multi-DLA, mock (using WINNING BASELINE config)

The `slurm/configs/*.env` files default to the historic baseline (10k samples,
model_epoch_920, no τ-EB). You MUST export the v3-stack overrides before launch:

```bash
# 1. Source DESI env (mandatory for astropy/desispec)
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main

cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection

# 2. WINNING BASELINE overrides (§0 above)
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat"
export NUM_DLA_SAMPLES=50000
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
export EARLY_STOP_MODE=baseline

# 3. Dry-run first (no mkdir, no submit)
bash slurm/launch.sh slurm/configs/london0_y3.env --dry-run --no-sleep | head

# 4. Submit. The launcher refuses any OUTDIR outside the allowed write roots.
bash slurm/launch.sh slurm/configs/london0_y3.env \
    --outdir /pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)-v3/
```

**WARNING — production sbatch scripts don't forward τ-EB / early_stop_mode.**
See §10.1 for details. The exports above will reach the python CLI only if you
either (a) use `slurm/run_local.sh` instead, or (b) patch the sbatch scripts as
described in §10.1. For one-node-at-a-time on a salloc'd compute node:

```bash
salloc -N 1 -C cpu -q interactive -t 04:00:00 -A desi
bash slurm/run_local.sh slurm/configs/london0_y3.env \
     --outdir /pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)-v3/ \
     --parallel-files 32 --max-workers 8
```

### 6.2 Multi-DLA, real LOA

```bash
# Same env + overrides as §6.1, then:
bash slurm/launch.sh slurm/configs/loa_y3.env \
    --outdir /pscratch/sd/j/jibancat/desi-loa-prod-$(date +%Y%m%d)-v3/
```

### 6.3 LLS mode (sub-DLA + DLA, NHI ∈ [17.2, 22] or [19, 22])

LLS configs already set `MAX_DLAS=1, SINGLE_ABSORBER_MODEL=1, FILTER_LOW_LIKELIHOOD=0`
and select the right `pw_samples_a3_172_220_50000.mat` or `pw_samples_a3_190_220_50000.mat`.
To use the v3 model + τ-EB on top:

```bash
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
export EARLY_STOP_MODE=baseline

bash slurm/launch.sh slurm/configs/london0_y3_lls172.env \
    --outdir /pscratch/sd/j/jibancat/desi-mock-london0-lls172-$(date +%Y%m%d)-v3/

bash slurm/launch.sh slurm/configs/loa_y3_lls190.env \
    --outdir /pscratch/sd/j/jibancat/desi-loa-lls190-$(date +%Y%m%d)-v3/
```

LLS mode is ~3.8× the wall-time of multi-DLA mode (~62 node-hours per million
spectra; see §4.3).

### 6.4 Combine results

```bash
# Multi-DLA mock
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5" \
    --mock

# Multi-DLA real LOA
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5"
```

### 6.5 Run P/C eval (mock only)

```bash
# Molly-faithful (matches Molly's 2509 notebook headline)
python examples/molly_faithful_pc_plots.py \
    --catalog-dir "$OUTDIR" \
    --truth /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits \
    --bal-cat /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits \
    --no-bal --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out "$OUTDIR/figures_molly/snr2_pdla0.99"
```

For Saclay use `hcd_truth_cat.fits` and the Saclay mockdir. For 2LPT, see truth
catalog at `<mockdir>/dla_cat.fits` (or equivalent).

---

## 7. Expected P/C numbers for the WINNING BASELINE

### 7.1 Classical DLAs (NHI ∈ [20.3, 23]), SNR_RED > 2

| Mock     | P_DLA ≥ 0.99    | P_DLA ≥ 0.999   | P_DLA ≥ 0.99999 |
|----------|-----------------|-----------------|------------------|
| London 8f | P=0.845, C=0.766 | P=0.855, C=0.740 | P=0.887, C=0.690 |
| Saclay 8f | P=0.871, C=0.771 | P=0.884, C=0.748 | P=0.901, C=0.677 |
| 2LPT 8f (baseline GP, projection) | P≈0.78, C≈0.91 (no v3 run yet on 2LPT) | — | — |

**Headline at the recommended operating point (P_DLA ≥ 0.99, SNR_RED > 2):**
Purity ~0.85, Completeness ~0.77. The 8-pp completeness gap below 85 % is
*intrinsic at SNR > 2* — sampling-noise is 130× below the signal-null gap, so
this is a model-side / forward-model limiter (see today's diagnostic).

### 7.2 SNR > 1 (acceptable per project memory)

Completeness drops to ~65 % at SNR > 1 because the FILTER mechanism under-weights
marginal-z modes. See `MOLLY_TABLES_SNR_CUTS.md` SNR>1 table.

### 7.3 SNR > 4 (cosmology-grade subset; not the headline)

P ≈ 0.83 / C ≈ 0.90 at P_DLA ≥ 0.999 with the full stack — approximately 85/85.

### 7.4 Sub-DLA catalog (NHI ∈ [19, 20.3])

**Not yet validated** with the WINNING BASELINE. The recommendation in
`RECOMMENDATIONS.md` §3b is to switch `SUB_DLA_SAMPLES_FILE` to the 100k file
with NHI floor 19.1; the impact on multi-DLA-mode sub-DLA P/C has not been
measured at scale. Run `molly_faithful_subdla_pc_plots.py` (TODO — script
doesn't exist yet; analogue of `molly_faithful_pc_plots.py` with `--nhi-min 19.5`
and a sub-DLA truth filter).

### 7.5 LLS catalog (NHI ∈ [17.2, 20.3], from LLS-mode runs)

Not validated by the WINNING BASELINE harness yet. Use Pathway A CDDF
(`CDDF_analysis/calc_cddf.py`) on LLS-mode HDF5 posteriors, not the
catalog-time P/C tool, because LLS-mode emits a single-absorber catalog with
NHI floor 17.2.

---

## 8. Quick lookup: "I want to run X"

| Goal | Config | Outdir suffix |
|------|--------|---------------|
| DLA catalog on London-0 (production) | `slurm/configs/london0_y3.env` + §0 exports | `desi-mock-london0-prod-YYYYMMDD-v3` |
| DLA catalog on London-1 | (TODO: create `london1_y3.env`) | `desi-mock-london1-prod-YYYYMMDD-v3` |
| DLA catalog on Saclay-0 | `slurm/configs/saclay0_y3.env` + §0 exports | `desi-mock-saclay0-prod-YYYYMMDD-v3` |
| DLA catalog on Saclay-1 | (TODO: create `saclay1_y3.env`) | `desi-mock-saclay1-prod-YYYYMMDD-v3` |
| DLA catalog on 2LPT-0 | `slurm/configs/2lpt0_y3.env` + §0 exports | `desi-mock-2lpt0-prod-YYYYMMDD-v3` |
| DLA catalog on DESI LOA | `slurm/configs/loa_y3.env` + §0 exports | `desi-loa-prod-YYYYMMDD-v3` |
| LLS catalog NHI≥17.2 (any dataset) | `slurm/configs/<flavour>_y3_lls172.env` + v3+τ-EB exports | `…-lls172-YYYYMMDD-v3` |
| LLS catalog NHI≥19.0 (any dataset) | `slurm/configs/<flavour>_y3_lls190.env` + v3+τ-EB exports | `…-lls190-YYYYMMDD-v3` |

---

## 9. After all jobs finish

```bash
# 1. Combine per-file HDF5 → single combined.h5
python combine_processed_h5.py --processed_dir "$OUTDIR" --output_file "$OUTDIR/combined.h5" [--mock]

# 2. P/C eval (mock only, see §6.5)

# 3. Catalog-time post-process: lyb_veto (free ~+1.7 pp purity)
python -c "
from astropy.table import Table
from gpy_dla_detection.postprocess.lyb_veto import flag_lybeta
cat = Table.read('$OUTDIR/combined_dlacat.fits')
cat = flag_lybeta(cat, dz_match=0.005, targetid_col='TARGETID', z_col='Z_DLA', nhi_col='NHI')
cat = cat[~cat['LYBETA_FLAG']]
cat.write('$OUTDIR/combined_dlacat_lybveto.fits', overwrite=True)
"

# 4. CDDF / Ω_HI (population statistics)
# See docs/tutorial_population_statistics.md
```

---

## 10. Gotchas (READ BEFORE LAUNCHING)

### 10.1 Production sbatch scripts don't forward τ-EB / early_stop_mode

`slurm/submit_desi_mock.sh` and `slurm/submit_desi_loa.sh` do **not** pass
`--enable_tau_eb`, `--tau_eb_objective`, or `--early_stop_mode` to the python
CLI. The flags exist in `desi-DLAGP.py:parse()` and the env vars exist in the
exports, but the sbatch scripts' python command lines were last updated before
those flags landed.

Only `slurm/run_local.sh` forwards them (see lines 187-204).

**Fix options:**
- (a) Use `slurm/run_local.sh` on a salloc'd interactive node. Lose the queue
  fan-out but get correct hyperparams.
- (b) Patch `slurm/submit_desi_{mock,loa}.sh` to add three lines at the end of
  the python command:
  ```bash
  $(if [ "$ENABLE_TAU_EB" = "1" ]; then echo "--enable_tau_eb 1 --tau_eb_objective $TAU_EB_OBJECTIVE"; fi) \
  --early_stop_mode "${EARLY_STOP_MODE:-baseline}"
  ```
  before the trailing `&`. Then launch via `slurm/launch.sh`. **Not yet patched
  in `production_533`.**

This is a deployment blocker for any production run that needs τ-EB or A/D
early-stop modes via sbatch. Validate by looking at the actual `python …`
command in `slurm/submit_desi_{mock,loa}.sh` before believing the env vars
made it through.

### 10.2 NUM_DLA_SAMPLES must equal the .mat rowcount

`pw_samples_a3_190_220_50000.mat` is 50k rows. Running with `NUM_DLA_SAMPLES=10000`
will silently read only the first 10k samples — but the QMC weight normalization
assumes `1/N` where N is the *file* count. The math goes wrong. Always pair the
file with its rowcount.

### 10.3 Saclay mock-0 subdir is `juraLy8-124`, mock-1 is `jura-124`

Already encoded in the configs but worth knowing if you write a new one.

### 10.4 BAL inclusion vs exclusion

Inference: BAL QSOs are **included** (`BALMASK=false`). No pixel masking.
P/C eval: BAL QSOs are **excluded** via `--no-bal` flag in `molly_faithful_pc_plots.py`
(filters `BI_CIV > 0` from the truth + catalog). Always match this convention.

### 10.5 Z ≥ 1.96 cut is at inference time

The QSO catalog gets filtered in `desi-DLAGP.py` (look for `Z >= 1.96` or
similar near the `read_in_each_plots_*` logic). All numbers in §1.1 reflect
this cut. If you want to keep low-z QSOs, you have to patch the catalog
filter.

### 10.6 `EARLY_STOP_MODE` default is `baseline`, set via env

The `--early_stop_mode` CLI flag defaults to `os.environ.get("EARLY_STOP_MODE",
"baseline")`. If you don't set it, production behavior is unchanged
(`baseline`). Variants A and D are under evaluation and **not yet promoted to
production**.

### 10.7 The launcher refuses unsafe `OUTDIR`s

`slurm/launch.sh` refuses to submit if `OUTDIR` falls outside
`/pscratch/sd/j/jibancat/`, `/global/homes/j/jibancat/`, or
`/global/cfs/cdirs/desicollab/users/jibancat/`. This is by design; do not work
around it. See `docs/nersc_write_permissions.md`.

### 10.8 The current jupyter node has 22 inference slices running

As of 2026-05-13 12:00 PT, `nid004179` (job 52907557) is running 22 resume
slices inline. **Do not submit new jobs or kill these.** They populate
`prod533_5k_20260511/{london_v3_loa124_early_stop_A,early_stop_D,joint_dla_subdla_sweep/*}`.

---

## 11. Reproduce the current best result (copy-pasteable)

```bash
# === Step 0: env ===
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection

# === Step 1: WINNING BASELINE exports ===
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat"
export NUM_DLA_SAMPLES=50000
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
export EARLY_STOP_MODE=baseline
export MAX_DLAS=3
export FILTER_LOW_LIKELIHOOD=1

OUTDIR="/pscratch/sd/j/jibancat/desi-mock-london0-prod-$(date +%Y%m%d)-v3"

# === Step 2: dry-run ===
bash slurm/launch.sh slurm/configs/london0_y3.env \
    --outdir "$OUTDIR" --dry-run --no-sleep | head

# === Step 3: PATCH submit_desi_mock.sh FIRST (§10.1) ===
# Add to the python command lines, before the trailing `&`:
#     --enable_tau_eb "$ENABLE_TAU_EB" \
#     --tau_eb_objective "$TAU_EB_OBJECTIVE" \
#     --early_stop_mode "$EARLY_STOP_MODE" \

# Either patch the file then:
bash slurm/launch.sh slurm/configs/london0_y3.env --outdir "$OUTDIR"

# OR use run_local.sh on a salloc'd node (forwarding already correct):
salloc -N 1 -C cpu -q interactive -t 04:00:00 -A desi
bash slurm/run_local.sh slurm/configs/london0_y3.env \
    --outdir "$OUTDIR" --parallel-files 32 --max-workers 8

# === Step 4: combine + P/C ===
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5" \
    --mock

python examples/molly_faithful_pc_plots.py \
    --catalog-dir "$OUTDIR" \
    --truth /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits \
    --bal-cat /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits \
    --no-bal --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto \
    --out "$OUTDIR/figures_molly/snr2_pdla0.99"

# Expected at SNR>2, P_DLA≥0.99, lya_lyb [911, 1216] Å:
#   Purity  ~0.845
#   Completeness ~0.766
# (per docs/runs/2026-05-12_saclay_v3_loa124_results.md and the 8f reference run)
```
