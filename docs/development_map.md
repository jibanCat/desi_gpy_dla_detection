# Development map — repository structure, key entry points, reproduction workflows

> Written 2026-05-12. Intended as the **first doc to read** when starting a new session on this repo. Pairs with `CLAUDE.md` (session handoff state) and `docs/notes/` (investigation logs). Branch: `production_533`.

## What this repo does

Gaussian-process Bayesian detection of damped Lyα absorbers (DLAs), sub-DLAs / LLS in DESI quasar spectra. Outputs per-spectrum HDF5 with model evidences + per-DLA FITS catalogs. Downstream computes column-density distribution f(N,z), dN/dX, Ω_HI.

Three run modes:

| Mode | Flag combo | Detects |
|---|---|---|
| **DLA mode** | `MAX_DLAS=3 SINGLE_ABSORBER_MODEL=0` | up to 3 DLAs/spec + competing sub-DLA hypothesis |
| **LLS / sub-DLA mode** | `MAX_DLAS=1 SINGLE_ABSORBER_MODEL=1` | single absorber over a narrower NHI range |
| **High-z DLA mode** | `--highz` (separate entry) | DLAs at z > 4 (legacy code path) |

## Top-level entry points

| File | Role |
|---|---|
| `desi-DLAGP.py` | **Primary CLI** — healpix/tile-based DESI runs. Loads spectra, runs Bayesian model selection per spectrum, emits HDF5 + FITS catalog. |
| `desi-DLAGP-highz.py` | High-z variant (separate, to-be-merged). |
| `dlasearch.py` | Healpix/mock processing engine. Handles parallelism, BAL mask gating, the per-file loop. Called by `desi-DLAGP.py`. |
| `run_bayes_select.py` | The `DLAHolder` class: model init (NullGP + DLAGP + SubDLAGP) + per-spectrum dispatch + Bayesian model selection. |
| `combine_processed_h5.py` | Merge per-healpix HDF5 outputs → single `combined.h5`. Run after each production run. |

## Key modules (`gpy_dla_detection/`)

| File | Role |
|---|---|
| `null_gp.py` | Base GP (no DLA). Woodbury Cholesky O(n·k²). |
| `dla_gp.py` | DLA GP. Voigt absorption, multi-DLA recursion. |
| `subdla_gp.py` | Sub-DLA / LLS GP model. |
| `voigt_fast.py` | Production Voigt-profile evaluator. C extension + Python fallback. |
| `voigt_v2.py` | Newer Voigt with selectable LSF kernel and configurable `num_lines`. |
| `set_parameters.py` | All GP hyperparameters: `K=30`, `DLAMBDA=0.15`, `NUM_FOREST_LINES=3`, NHI prior ranges, z-search window, mean-flux τ₀/β. |
| `bayesian_model_selection.py` | Evidence aggregation. **Critical**: `p_dla = sum(model_posteriors[2:])`; `p_no_dla = 1 - p_dla` *includes* SubDLA. See "Score aggregation" below. |
| `tau_eb.py` | Empirical-Bayes τ_eff per-spectrum fit (PR #5). Enable with `--enable_tau_eb 1`. |
| `generate_samples.py` | QMC sample generation for the DLA prior (`pw_samples_a3_*.mat`) and sub-DLA prior (`subdla_samples_*.mat`). |
| `process_helpers.py` | HDF5 result schema + write. **Read this docstring** to understand the per-spectrum h5 layout. |
| `postprocess/lyb_veto.py` | Post-hoc Lyβ-misID detection + flag. Used after MAP for single-DLA framing. |

## Slurm workflow (`slurm/`)

Production runs use a **layered env-override** system:

```
slurm/configs/_base.env             # shared: τ₀, β, NUM_DLA_SAMPLES, K, DLAMBDA, FILTER_LOW_LIKELIHOOD, ENABLE_TAU_EB, …
slurm/configs/<mock>_y3.env         # per-mock: QSOCAT, MOCKDIR, OUTDIR, NSIDE, LEARNED_FILE, DLA_SAMPLES_FILE
slurm/configs/<mock>_y3_lls172.env  # LLS mode override (NHI ∈ [17.2, 22.0])
slurm/configs/<mock>_y3_lls190.env  # LLS mode override (NHI ∈ [19.0, 22.0])
```

`slurm/run_local.sh` sources `_base.env` first, then the mock-specific env, then env vars from the command line. Last write wins. Example:

```bash
LEARNED_FILE=/path/to/v3_loa124.h5 \
ENABLE_TAU_EB=1 \
bash slurm/run_local.sh slurm/configs/london0_y3.env \
  --outdir /pscratch/sd/j/jibancat/<run-name>/ \
  --window 8 --parallel-files 8 --max-workers 8
```

`slurm/launch.sh` is the sbatch wrapper (currently doesn't honor all env-overrides — `run_local.sh` is the safer path for v3/τ-EB experiments per `docs/runs/2026-05-12_v3_production_cost.md`).

## Output schema (per spectrum h5)

The h5 layout (see `gpy_dla_detection/process_helpers.py:34-50` docstring):

```
target_ids                 (N,)                                    int64
z_qsos                     (N,)
snrs                       (N,)         red-side SNR (canonical for DLA P/C)
snrs_blue                  (N,)         blue-side SNR
model_posteriors           (N, 1+num_subdla+K) — DLA mode default: (N, 5)
                            col 0 = P(Null | D)
                            col 1 = P(SubDLA | D),  NHI ∈ [19.1, 20.0]
                            col 2 = P(1 DLA | D),   NHI ∈ [19, 22]  (PW14 prior)
                            col 3 = P(2 DLAs | D)
                            col 4 = P(3 DLAs | D)   (if MAX_DLAS=3)
p_dlas                     (N,)         = sum(model_posteriors[:, 2:])  — DLAs only
p_no_dlas                  (N,)         = 1 - p_dlas  = P(Null) + P(SubDLA)
log_likelihoods_dla        (N, K)       per DLA-count model
log_likelihoods_no_dla     (N,)
log_priors_dla, log_priors_no_dla
log_posteriors_dla, log_posteriors_no_dla
sample_log_likelihoods_dla (N, NUM_DLA_SAMPLES, K)   QMC samples
MAP_z_dlas                 (N, K)
MAP_log_nhis               (N, K)
z_dla_errs, log_nhi_errs   (N, K)       1-σ errors (curvature-based)
min_z_dlas, max_z_dlas     (N,)         search window per spectrum
```

Per-DLA `dlacat-*.fits` (one row per predicted DLA after thresholding):
`TARGETID, RA, DEC, Z_QSO, SNR_FOREST, SNR_REDSIDE, DLAID, Z_DLA, Z_DLA_ERR, NHI, NHI_ERR, DLAFLAG, P_DLA, P_NULL, LOGP_DLA, LOGP_NULL, MODEL_P`.

## Score aggregation — important subtlety

```
SubDLA NHI prior:    log NHI ∈ [19.1, 20.0]   (subdla_samples_a03_191_200_100000.mat)
DLA NHI prior:       log NHI ∈ [19.0, 22.0]   (pw_samples_a3_190_220_50000.mat)
Overlap:             [19.0, 20.0]             (both models can fit; marginal splits mass)
```

The **production p_DLA classifier** counts SubDLA as **no DLA** (`p_no_dla = P(Null) + P(SubDLA)`). Verified at `gpy_dla_detection/bayesian_model_selection.py:240-275`.

Consequence: low-SNR weak DLAs whose NHI posterior smears into the SubDLA range get classified as non-detections. This is **~5pp of the SNR>2 completeness ceiling** for classical DLAs at the P_DLA≥0.99 cut. See `docs/notes/2026-05-12_mlmc_design.md` for measurement and remediation options.

The docstrings in `process_helpers.py:38` and `subdla_samples.py:27` say "[19, 20.3]" — **stale**; actual is [19.1, 20.0]. Worth a docstring fix.

## Validation workflow

The standard P/C evaluator is `examples/molly_faithful_pc_plots.py`. It reproduces Molly Wolfson's notebook recipe exactly:

```bash
python examples/molly_faithful_pc_plots.py \
    --catalog-dir <outdir-with-dlacat-and-processed> \
    --truth /global/cfs/projectdirs/desi/mocks/.../dla_cat.fits \
    --bal-cat .../bal_cat.fits --no-bal \
    --truth-nhi-min 20.3 \
    --out <figures-dir>/
```

Outputs `summary.tsv` with rows `[P_DLA_cut, NHI_bin, snr_min, n_TP, n_kept, n_truth_kept, purity, completeness]`. Pairs each row of the predicted catalog against truth with `|Δz|/(1+z) < 0.01` greedy matching per TARGETID.

For investigations that need the underlying logic (e.g., re-aggregating p_DLA), reuse `examples/gp_native_pc_plots.match_truth_to_cat` directly — see `tools/research/test_unified_pdla_perdla.py` for an example.

## Two science targets and current numbers

Per `docs/notes/2026-05-12_mlmc_design.md`, on London 8f v3_loa124 (BAL-excl, lya_lyb [911, 1216], SNR>2):

| Target | NHI range | Aim P / C | Best measured |
|---|---|---|---|
| Classical DLA | ≥ 20.3 | 85-90% / 85% | **84.6 / 83.5** @ P_DLA≥0.99 (baseline) |
| Sub-DLA | [19.1, 20.0) | 85% / 70% | **77.8 / 3.7** @ P(SubDLA)≥0.99 (sweep: 59/56 @ 0.5) |

Saclay v3_loa124 confirmed off-distribution generalization at **87.07 / 77.10** at P_DLA≥0.99 (see `docs/runs/2026-05-12_saclay_v3_loa124_results.md`). Production cost: **17 node-hours per 1M QSOs** for v3_loa124 vs 15.2 for baseline (`docs/runs/2026-05-12_v3_production_cost.md`).

## Path forward — open problems

| Problem | Path | Status |
|---|---|---|
| Classical DLA ~0.5-2pp short of 85/85 strict | MLMC / adaptive importance sampling at MAP seed | Designed but not implemented; see `docs/notes/2026-05-12_mlmc_design.md` |
| Sub-DLA model far from 85/70 | (a) tighter NHI prior, (b) better training, (c) MLMC | Open — needs its own investigation |
| MAP+LR detection contaminated by logN ≈ 20.5 ghosts | Add Laplace correction (DOES NOT WORK — measured 2026-05-12) → use MLMC instead | Resolved: don't pursue MAP+LR |
| Prior dilution at narrow likelihood peaks | MLMC / IS | Same as MLMC item above |
| SubDLA-as-null siphoning ~5pp completeness | Drop SubDLA + extend DLA prior (LOSES sub-DLA detector — bad trade); or keep + recognize as structural | Open structural decision |

## Compute environment

The Jupyter session bash shell needs a clean subshell to source the DESI env (sourcing fails silently in the parent shell):

```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
python ...
'
```

This gets Python 3.10.14 with `desispec 0.71.2.dev10051`, `h5py`, `scipy`, `astropy`, `fitsio`, `torch`. The standalone scripts in `tools/research/` all rely on this pattern.

For interactive compute: `salloc -N 1 -C cpu -q interactive -t 4:00:00 -A desi`.

## Reproducing the 2026-05-12 SubDLA-mechanism finding

```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
python tools/research/test_unified_pdla_perdla.py
'
```

Outputs three tables (classical DLA + sub-DLA from two angles) matching the numbers in `docs/notes/2026-05-12_mlmc_design.md`.

## Files NOT documented here

The investigation harnesses at repo top-level (`_*.py`, `investigate_*.py`) are one-off, undocumented, and date from prior sessions. They are listed in `HANDOFF.md`. Do not assume they work without reading the relevant `docs/notes/` entry first. If you need to keep one, move it to `tools/research/` and add a row to that README.

## Where related docs live

| Doc | Topic |
|---|---|
| `docs/architecture.md` | High-level pipeline architecture and data flow |
| `docs/tutorial_quickstart.md` | First-time run setup (libcerf, conda env, etc.) |
| `docs/tutorial_population_statistics.md` | dN/dX and CDDF calibration workflow |
| `docs/data_inputs.md` | Input file schemas |
| `docs/nersc_write_permissions.md` | NERSC writeable-path policy |
| `docs/greatlakes_setup.md` | GreatLakes alternative environment |
| `docs/notes/2026-04-*` | Pre-PR investigation logs |
| `docs/notes/2026-05-12_*` | This session's MAP+LR failure + MLMC design |
| `docs/runs/2026-05-12_*` | Selected production run results |
| `HANDOFF.md` | Per-session state handoff (read for current session) |
| `CLAUDE.md` | Session memory — local, not committed |
