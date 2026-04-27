# CLAUDE.md — Session Handoff & Project Memory

> **Written**: 2026-04-17, after PR #2 merged into `desi_y3`.
> **Next session**: GreatLakes HPC (University of Michigan).
> Read this file in full before starting any work.

---

## 1. What This Repository Is

**GP-DLA**: Gaussian-Process Bayesian detection of DLA / sub-DLA / LLS absorbers in
DESI quasar spectra.  Outputs absorber catalogs, dN/dX, f(N,z), Ω_HI.

- **Primary entry point**: `desi-DLAGP.py` (healpix-based DESI LOA/mock runs)
- **Population statistics**: `CDDF_analysis/cddf_mock.py` + `CDDF_analysis/cddf_calibration.py`
- **SLURM production**: `slurm/` (NERSC Perlmutter) and `slurm/lls_runs/` (LLS/single-absorber)
- **Tests**: `tests/` — 65 passing, no desispec required for CDDF tests

---

## 2. Cluster Context

### Was: NERSC Perlmutter

All existing SLURM scripts target Perlmutter:
- Scratch: `/pscratch/sd/j/jibancat/`
- DESI software env: `source /global/cfs/cdirs/desi/software/desi_environment.sh main`
- DESI spectra + mocks: `/global/cfs/projectdirs/desi/` and `/global/cfs/cdirs/desicollab/`
- Account: `-A desi`
- Queue: `-q regular` / `-q debug`

### Now: GreatLakes (University of Michigan)

GreatLakes uses different paths, partitions, and environment setup. **All SLURM scripts
need adaptation** before running there. Key differences to expect:

| Item | NERSC | GreatLakes |
|------|-------|------------|
| Scratch | `/pscratch/sd/j/jibancat/` | `$SCRATCH` or `/scratch/jibancat/` |
| Account | `desi` | UMich allocation (TBD) |
| Queue | `regular` / `debug` | `standard` / `gpu` / etc. |
| DESI env | `desi_environment.sh main` | custom conda env (no DESI module) |
| Spectra access | full `/global/cfs/projectdirs/desi/` | partial — mocks + subset of real spectra |

**Spectra available on GreatLakes** (per user): mock catalogs + part of real DESI spectra.
London/Saclay mock spectra (the actual per-healpix FITS files) may be present.
Real LOA spectra may be partial.

---

## 3. Current State of the Codebase (as of 2026-04-17)

### Git

- **Active branch**: `desi_y3` (main development branch, NOT `main`)
- **PR #2** (`claude/friendly-allen` → `desi_y3`): **MERGED** ✓
- **`main` branch**: stable older code — do not push science changes there

### What PR #2 Added (all now in `desi_y3`)

#### CDDF analysis modules (notebook extraction)
| File | Description |
|------|-------------|
| `CDDF_analysis/cddf_calibration.py` | Alpha(z) calibration: `calibration_factor_alpha()`, `apply_alpha_to_bounds()`, `correction_ratio_with_uncertainty()`, `apply_correction_with_uncertainty()` |
| `CDDF_analysis/cddf_io.py` | Save/load calibrated dN/dX + f(N,z) text tables |
| `CDDF_analysis/cddf_mock.py` | Three new utilities: `logN_bins_from_mids()`, `dndx_to_ellz()`, `dndx_bounds_to_ellz()`; full docstrings added throughout |
| `notebooks/Paper_plots.ipynb` | Skeleton notebook wiring all modules end-to-end |

#### Tests (65 passing, no desispec needed)
| File | Tests | Covers |
|------|-------|--------|
| `tests/test_cddf_calibration.py` | 31 | `cddf_calibration.py` — alpha(z), error propagation, bounds |
| `tests/test_cddf_mock.py` | 34 | `cddf_mock.py` — dN/dX, CDDF, Omega, windows |
| `tests/test_generate_samples.py` | 15 | `generate_samples.py` — QMC sample generation |

Run all: `python -m pytest tests/test_cddf_mock.py tests/test_cddf_calibration.py tests/test_generate_samples.py -v`

#### SLURM / LLS runs
| File | Description |
|------|-------------|
| `slurm/lls_runs/run_saclay_mock0_nhi172.sh` | Saclay mock-0, NHI 17.2–22.0 |
| `slurm/lls_runs/run_saclay_mock0_nhi19.sh` | Saclay mock-0, NHI 19.0–22.0 |
| `slurm/lls_runs/run_saclay_mock1_nhi172.sh` | Saclay mock-1, NHI 17.2–22.0 |
| `slurm/lls_runs/run_saclay_mock1_nhi19.sh` | Saclay mock-1, NHI 19.0–22.0 |
| `slurm/lls_runs/README.md` | Table of all 10 LLS run scripts |
| `slurm/lls_runs/run_reference_mock1_nhi172.sh` | Fixed: resumes from START_INDEX=702 (removed stale break at 960) |

#### Data transfer (NERSC → local)
- `data/scripts/rsync_mock_catalogs.sh`: downloads `zcat.fits`, `bal_cat.fits`, truth catalogs for all 4 mocks; SSH ControlMaster (authenticate once)

#### Bug fix
- `dlasearch.py` line 317: duplicate `"""` removed (caused `SyntaxError` at import)

#### Docs
| File | Description |
|------|-------------|
| `docs/architecture.md` | Code flow, module map, three run modes, two CDDF pathways |
| `docs/tutorial_quickstart.md` | Corrected libcerf-from-source, Y3 flags, BAL note |
| `docs/tutorial_population_statistics.md` | Full calibration workflow |
| `docs/data_inputs.md` | All input file schemas |
| `README.md` | Run modes section, HDF5 schema, absorber modes table |

---

## 4. Local Data State (on Mac, NOT on GreatLakes)

The Mac has the following mock **catalog** files (small, ~MB each) downloaded via
`rsync_mock_catalogs.sh`. These will NOT be on GreatLakes automatically.

```
data/mocks/
├── london/v5.9.5/
│   ├── mock-0/  zcat.fits  bal_cat.fits  dla_cat.fits  SOURCE.txt
│   └── mock-1/  zcat.fits  bal_cat.fits  dla_cat.fits  SOURCE.txt
└── saclay/v4.7.5/
    ├── mock-0/  zcat.fits  bal_cat.fits  hcd_truth_cat.fits  SOURCE.txt
    └── mock-1/  zcat.fits  bal_cat.fits  hcd_truth_cat.fits  SOURCE.txt
```

**The actual mock spectra** (per-healpix FITS, ~GB) live only on NERSC:
- London: `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-{0,1}/jura-124/`
- Saclay mock-0: `/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/`
- Saclay mock-1: `.../mock-1/jura-124/`  ← note: different subdir name from mock-0

**On GreatLakes**: you will have access to mock spectra and partial real spectra.
The catalog files (zcat, bal_cat, truth) may need to be copied there separately
(use `rsync_mock_catalogs.sh` adapted for GreatLakes paths, or scp from Mac).

---

## 5. Inference Output Locations (NERSC `/pscratch`)

These are the GP-DLA inference results currently on NERSC scratch:

### Multi-DLA production runs (MAX_DLAS=3, standard catalog)
| Directory | Data | Status |
|-----------|------|--------|
| `desi-loa-gpdla-20250912-desi-learned/` | Real DESI LOA | Done |
| `desi-mock-gpdla-20251229-y3-learned-epoch920/` | London mock-0 | Done |

### LLS single-absorber runs (MAX_DLAS=1, SINGLE_ABSORBER_MODEL=1)
| Directory | Data | NHI | Status |
|-----------|------|-----|--------|
| `desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/` | London mock-0 | 17.2–22.0 | Done |
| `desi-mock-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/` | London mock-0 | 19.0–22.0 | Done |
| `desi-mock-1-gpdla-20260119-y3-learned-epoch920-lls_run-nhi172/` | London mock-1 | 17.2–22.0 | Partially done (702→1150 resumed) |
| `desi-mock-saclay-0-gpdla-20260415-y3-learned-epoch920-lls_run-nhi172/` | Saclay mock-0 | 17.2–22.0 | Submitted |
| `desi-mock-saclay-0-gpdla-20260415-y3-learned-epoch920-lls_run-nhi190/` | Saclay mock-0 | 19.0–22.0 | Submitted |
| `desi-mock-saclay-1-gpdla-20260415-y3-learned-epoch920-lls_run-nhi172/` | Saclay mock-1 | 17.2–22.0 | Submitted |
| `desi-mock-saclay-1-gpdla-20260415-y3-learned-epoch920-lls_run-nhi190/` | Saclay mock-1 | 19.0–22.0 | Submitted |
| `desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi172/` | Real DESI LOA | 17.2–22.0 | Done |
| `desi-loa-gpdla-20251229-y3-learned-epoch920-lls_run-nhi190/` | Real DESI LOA | 19.0–22.0 | Done |

After each run completes, combine with:
```bash
python combine_processed_h5.py \
    --processed_dir /pscratch/sd/j/jibancat/<OUTDIR>/ \
    --output_file /pscratch/sd/j/jibancat/<OUTDIR>/combined.h5 \
    [--mock]   # add for mock runs
```

---

## 6. Key Model Settings (Y3 Production)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `LEARNED_FILE` | `model_epoch_920.h5` | Y3 GP trained model |
| `PREV_TAU_0` | 0.00246 | Turner+2024 mean-flux prior |
| `PREV_BETA` | 3.62 | Turner+2024 mean-flux prior |
| `DLAMBDA` | 0.15 Å | Pixel spacing |
| `K` | 30 | GP rank |
| `NUM_FOREST_LINES` | 3 | Lyα + Lyβ + Lyγ |
| Multi-DLA mode | `MAX_DLAS=3, SINGLE_ABSORBER_MODEL=0` | Standard DLA catalog |
| LLS mode | `MAX_DLAS=1, SINGLE_ABSORBER_MODEL=1` | LLS/subDLA catalog |

**BAL treatment**: `--balmask` is NOT passed in any production run. BAL QSOs are
included; no pixel masking is applied. The "altbal" LOA catalog was used as input
but BAL columns are never loaded.

---

## 7. Two CDDF Analysis Pathways

| Pathway | Module | Input | CI method | Use when |
|---------|--------|-------|-----------|----------|
| **A: Bayesian** | `CDDF_analysis/calc_cddf.py` | HDF5 model posteriors | Poisson-binomial | Full Bayesian CI propagation |
| **B: Direct** | `CDDF_analysis/cddf_mock.py` | FITS absorber catalog | Bootstrap over QSOs | Mock validation, calibration |

Notebooks use **Pathway B**. The calibration workflow is:
1. `compute_dndx()` on mock truth → `dNdX_truth(z)`
2. `compute_dndx()` on GP-DLA mock output → `dNdX_measured(z)`
3. `calibration_factor_alpha()` → `alpha(z)` (correction factor; alpha>1 = incompleteness)
4. `apply_alpha_to_bounds()` on real DESI data → calibrated dN/dX
5. Save with `cddf_io.save_dndx_combined()`

---

## 8. Remaining TODO Tasks

### High priority (science)
- [ ] **Check Saclay LLS run completion** on NERSC — verify job outputs in
  `desi-mock-saclay-{0,1}-gpdla-20260415-*` directories; combine once done
- [ ] **Combine London mock-1 LLS run** (`desi-mock-1-gpdla-20260119-*`) after
  resumed jobs finish (START_INDEX 702→1150)
- [ ] **Run CDDF analysis on Saclay mocks** — once combined.h5 files exist,
  use `CDDF_dNdX_all.ipynb` + `CDDF_fN_z.ipynb` with Saclay truth (`hcd_truth_cat.fits`)
- [ ] **Cross-validate London vs Saclay calibration** — do alpha(z) values agree?
  This is the core science check for the LLS/subDLA calibration

### Medium priority (code)
- [ ] **Adapt SLURM scripts for GreatLakes** — new partition names, account, paths,
  env setup (no `desi_environment.sh`; need conda env with desispec)
- [ ] **Write GreatLakes submit scripts** analogous to `slurm/submit_desi_mock.sh`
  and `slurm/submit_desi_loa.sh` for GreatLakes scheduler
- [ ] **Copy/rsync mock catalog files to GreatLakes** — the `data/mocks/` files on
  Mac (zcat, truth catalogs) need to go to GreatLakes; adapt `rsync_mock_catalogs.sh`
- [ ] **`Paper_plots.ipynb`**: populate with real data paths once GreatLakes outputs
  are available

### Lower priority (cleanup/docs)
- [ ] Merge `desi-DLAGP.py` + `desi-DLAGP-highz.py` with `--highz` flag
- [ ] Merge `combine_dlakibo.py` + `combine_dlamocks.py` with `--mock` flag
- [ ] `DLAHolder` refactor: separate model-init from inference dispatch
- [ ] `effective_optical_depth.py`: add explicit link to Turner+2024 τ₀/β values
- [ ] `docs/faq.md`: LinAlgError, model_posteriors index, voigt_fast C extension
- [ ] `docs/tutorial_run_one_spectrum.md`

---

## 9. GreatLakes Adaptation Checklist

When starting on GreatLakes, work through this before running anything:

1. **Clone / pull** the repo from GitHub (`git checkout desi_y3`)
2. **Conda environment**: create env with `desispec`, `numpy`, `scipy`, `astropy`,
   `h5py`, `torch`, `pytest`; build `voigt_fast.py` C extension:
   ```bash
   # Compile libcerf from source first (see docs/tutorial_quickstart.md)
   cc -fPIC -shared -o gpy_dla_detection/_voigt.so \
       gpy_dla_detection/ctypes_voigt.c \
       -I$HOME/.local/usr/local/include \
       -L$HOME/.local/usr/local/lib64 -lcerf
   ```
3. **Run tests** to verify env: `python -m pytest tests/test_cddf_mock.py tests/test_cddf_calibration.py -v`
4. **Copy data files** to GreatLakes scratch:
   - GP model: `learnlogs/model_epoch_920.h5`
   - DLA/subDLA sample grids: `data/dr12q/processed/pw_samples_a3_*.mat`, `subdla_samples_*.mat`
   - Prior catalogs: `data/dr12q/processed/catalog.mat`, los/dla catalogs
   - Mock catalog files: `data/mocks/` tree (zcat, truth cats)
5. **Write new SLURM submit scripts** for GreatLakes scheduler — copy
   `slurm/submit_desi_mock.sh` as template, adjust:
   - `#SBATCH -A <greatlakes_account>`
   - `#SBATCH -p standard` (or appropriate partition)
   - Remove `source /global/cfs/cdirs/desi/software/desi_environment.sh main`
   - Add `conda activate <your_env>`
   - Update all `/pscratch/sd/j/jibancat/` paths to GreatLakes scratch
6. **Update run scripts** — copy relevant scripts from `slurm/lls_runs/` as
   templates; update QSOCAT, MOCKDIR, OUTDIR, LEARNED_FILE paths

---

## 10. Repository Map (Key Files)

```
/
├── desi-DLAGP.py           PRIMARY ENTRY: CLI for healpix/tile DESI runs
├── dlasearch.py            Healpix/mock processing engine (parallelism, BAL mask)
├── run_bayes_select.py     DLAHolder: model init + Bayesian selection
├── combine_processed_h5.py Merge per-healpix HDF5 → single combined.h5
├── constants.py            Global constants (z range, SNR, optical depth, search window)
├── fitwarning.py           DLAFLAG bitmask definitions
│
├── gpy_dla_detection/
│   ├── null_gp.py          Base GP (no DLA); Woodbury O(nk²)
│   ├── dla_gp.py           DLA GP; Voigt absorption; multi-DLA recursion
│   ├── subdla_gp.py        Sub-DLA / LLS GP model
│   ├── voigt_fast.py       PRODUCTION Voigt: C extension + Python fallback
│   ├── set_parameters.py   All GP hyperparameters
│   ├── generate_samples.py QMC sample generation (CLI + module)
│   ├── process_helpers.py  HDF5 result schema + write
│   └── bayesian_model_selection.py  Bayes factor + stopping criterion
│
├── CDDF_analysis/
│   ├── cddf_mock.py        MAIN population statistics engine (dN/dX, CDDF, Omega)
│   ├── cddf_calibration.py Alpha(z) calibration + error propagation [NEW]
│   ├── cddf_io.py          Save/load calibrated output text tables [NEW]
│   └── calc_cddf.py        Bayesian CDDF (Pathway A, uses HDF5 posteriors)
│
├── slurm/
│   ├── submit_desi_loa.sh      NERSC LOA production submit
│   ├── submit_desi_mock.sh     NERSC mock production submit
│   └── lls_runs/               LLS/single-absorber run scripts (see README.md there)
│
├── data/scripts/
│   └── rsync_mock_catalogs.sh  Download mock catalogs from NERSC (SSH mux, dry-run)
│
├── docs/
│   ├── architecture.md         Code flow + module map
│   ├── tutorial_quickstart.md  Run pipeline on London mock
│   ├── tutorial_population_statistics.md  Full calibration workflow
│   └── data_inputs.md          All input file schemas
│
├── tests/                      65 passing (CDDF + generate_samples, no desispec)
└── notebooks/
    ├── CDDF_dNdX_all.ipynb     Main dN/dX calibration workflow (76 cells)
    ├── CDDF_fN_z.ipynb         f(N,z) calibration (31 cells)
    └── Paper_plots.ipynb       Clean skeleton using extracted modules [NEW]
```

---

## 11. Known Issues / Gotchas

- **`NUM_FOEST_LINES`** typo in `slurm/run_loa_desi_y3_learned.sh` (missing R) — doesn't break anything since the env var fallback works, but note for new scripts use `NUM_FOREST_LINES`
- **Saclay mock subdirs differ**: mock-0 uses `juraLy8-124/`, mock-1 uses `jura-124/`
- **`model_posteriors` column layout differs by mode**:
  - `single_absorber_model=False`: `[:,0]`=Null, `[:,1]`=SubDLA, `[:,2]`=1DLA, `[:,3]`=2DLA, `[:,4]`=3DLA
  - `single_absorber_model=True`: `[:,0]`=Null, `[:,1]`=absorber
- **BAL QSOs included** in all production runs — no masking, no exclusion
- **Alpha calibration direction**: `alpha = dNdX_truth / dNdX_measured_mock` — alpha>1 means the GP is *under-counting* (incompleteness); alpha<1 means over-counting
- **`voigt_fast.py` C extension** must be compiled on each new machine — pure Python fallback works but is ~10× slower
- **Tests require `desc` conda env** (or equivalent with scipy/astropy/h5py) — the default `miniforge3` base env does not have pytest

---

## 12. Useful Commands

```bash
# Run all CDDF tests (no desispec needed)
python -m pytest tests/test_cddf_mock.py tests/test_cddf_calibration.py tests/test_generate_samples.py -v

# Check syntax of any SLURM script
bash -n slurm/lls_runs/run_saclay_mock0_nhi172.sh

# Combine mock HDF5 outputs
python combine_processed_h5.py \
    --processed_dir /path/to/outdir/ \
    --output_file /path/to/outdir/combined.h5 \
    --mock

# Dry-run rsync of mock catalogs
bash data/scripts/rsync_mock_catalogs.sh --dry-run

# Import check
python -c "import ast; ast.parse(open('dlasearch.py').read()); print('syntax OK')"

# Quick import check for new modules
python -c "
from CDDF_analysis import cddf_calibration, cddf_io
from CDDF_analysis.cddf_mock import dndx_to_ellz, logN_bins_from_mids, dndx_bounds_to_ellz
print('all imports OK')
"
```

---

## 13. GreatLakes session — 2026-04-27

This session brought the pipeline up on UMich GreatLakes and added
three analysis capabilities. Reference docs:

| Topic | File |
|-------|------|
| Env setup (conda, libcerf, voigt.so) | `docs/greatlakes_setup.md` |
| NERSC ↔ GreatLakes path mapping | `docs/nersc_greatlakes_mapping.md` |
| Voigt v2 module (selectable LSF kernel, num_lines) | `gpy_dla_detection/voigt_v2.py` + `tests/test_voigt_v2_parity.py` |
| Lyβ misID + LLS xref postprocessing | `gpy_dla_detection/postprocess/` (with README) |
| Smoke-test runners | `examples/smoke_one_spectrum.py`, `run_smoke_batch.sh`, `pick_smoke_targets.py`, … |
| Production-catalog analyzer | `examples/analyze_production_catalog.py`, `examples/scan_pdla_cuts.py` |

Investigation logs (each is a falsifiable-test write-up, not a
reference doc — read the prose, not just the tables):

| File | Contents |
|------|---------|
| `docs/notes/2026-04-25_smoke_and_model_comparison.md` | eBOSS / Y3 / London model side-by-side on a strong-DLA target |
| `docs/notes/2026-04-25_filter_samples_sweep.md` | FILTER ∈ {0,1} × N_DLA ∈ {10k, 100k} on 20 strong DLAs |
| `docs/notes/2026-04-27_filter_completeness_explanation.md` | 200-target FILTER-1 completeness drop in [20.3, 20.6) and 5 fixes |
| `docs/notes/2026-04-27_lybeta_persistence_hypotheses.md` | Why GP fits Lyβ as a DLA; 4 hypotheses to test |
| `docs/notes/2026-04-27_subdla_model_improvements.md` | 4 ranked sub-DLA model improvements |
| `docs/notes/2026-04-27_bayesian_correctness_plan.md` | 4-step plan to discriminate integral / forward-model / prior contributions |
| `docs/notes/2026-04-27_london_pdla_scan_no_bal.md` | P_DLA cut sweep on London production (recovers historic ~78/80%) |
| `docs/notes/2026-04-27_london_postprocess_p99_no_bal.md` | Post-processing efficacy at the realistic operating point |
| `docs/notes/2026-04-27_pr_readiness_checklist.md` | What's done vs. what should land before merging |

**Production-code changes from this session**, intentionally minimal:
- `CDDF_analysis/cddf_mock.py`: numpy 2.x compat alias for `np.trapz` → `np.trapezoid`. Behaviour-preserving.
- `gpy_dla_detection/plottings/plot_model.py:237`: `argmax` → `nanargmax` to fix a NaN-label bug in `plot_samples_vs_this_mu`. Behaviour-preserving (the unfixed code crashes the model overlay; this restores it).

**93 tests pass**: the original 80 + 6 voigt_v2 parity + 5 lyb_veto unit + 2 smoke-target contamination.
