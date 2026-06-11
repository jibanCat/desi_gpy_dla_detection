# NERSC (Perlmutter) production drivers

NERSC mirror of `slurm/greatlakes/production/`. Runs the **V1** GP-DLA production
recipe on Perlmutter with **identical science knobs** to the GreatLakes config —
only paths, SLURM headers, env activation, and the per-cluster parallelism packing
differ. Built + cost-characterised 2026-06-01/02; all four V1 catalogs run since (see "Runs" below).

> **Before any `Write`/`mkdir` here read `docs/nersc_write_permissions.md`.** Writable
> roots: `/pscratch/sd/j/jibancat/`, `/global/homes/j/jibancat/`,
> `/global/cfs/cdirs/desicollab/users/jibancat/`. Everything else on CFS is read-only.

## Layout

| File | Purpose |
|------|---------|
| `_base_nersc.env` | Overlay: sources the in-repo `slurm/configs/_base.env` (all science defaults), then sets NERSC scheduler (`-A desi -q regular -C cpu`), output root, CFS model root, the writable-prefix allowlist, `desi_environment.sh` activation, and the **measured packing** (`NERSC_NTASKS=32`, `MAX_WORKERS=8`). |
| `london0_nersc_v1.env` | London mock-0 catalog (FILTER-on). The canonical mock flavour; the others source it. `NUM_SAMPLES` default **50000**. |
| `2lpt1_nersc_v1.env`, `2lpt2_nersc_v1.env` | 2LPT mock-1 / mock-2 catalog — source london0, swap mock paths + truth cat (`hcd_truth_cat.fits`). |
| `loa_nersc_v1.env` | **Real DESI Y3 LOA** catalog (`MODE=loa`, healpix mode). No truth cat (real data). |
| `loa_cddf_nersc.env` | Real-LOA **CDDF** input — sources `loa_nersc_v1`, sets `FILTER_LOW_LIKELIHOOD=0`, `MAX_DLAS=1`, `NUM_SAMPLES` default **100000** (dense Pathway-A integral). |
| `launch_nersc.sh` | Outer driver. Sources a flavour, validates OUTDIR ∈ allowed roots, submits one `sbatch` per `--window` chunk. `MODE=mock`→mock inner, `MODE=loa`→loa inner. |
| `submit_desi_mock_nersc.sh` | Mock inner. ONE `srun -n NTASKS` (PROCID-dispatched level2 file chunks). |
| `submit_desi_loa_nersc.sh` | LOA inner. Same pattern over **healpix** (`--hpx_start/--hpx_end`). |
| `parallelism_sweep_nersc.sh` + `_sweep_cell_nersc.sh` + `analyze_sweep.py` | Packing calibration (debug-QOS cells; one `srun -n N` per cell, NOT backgrounded). |
| `samplecost_sweep_nersc.sh` | Node-hours-vs-`NUM_SAMPLES` sweep. |
| `package_catalog.sh`, `_write_catalog_readme.py`, `resume_positions.py`, `repack_gzip.sh`, `repack_verify.py` | Cluster-agnostic post-run utilities (copied verbatim from GL). |
| (repo) `tools/make_subsampled_grids.py` | Makes the PW {10k,30k,50k} grids by subsampling the 100k Halton grids. |

## How it differs from the GreatLakes pipeline

| | GreatLakes | NERSC |
|---|---|---|
| Scheduler | `-A cavestru0 -p standard` | `-A desi -q regular -C cpu` |
| Env | `conda activate gpdla` + libcerf `LD_LIBRARY_PATH` | `source /global/cfs/cdirs/desi/software/desi_environment.sh main` (no conda; `_voigt.so` prebuilt). **Not `set -u`-clean** — sourced inside `set +u`. |
| Parallelism | N=2 × W=16 (36-core node optimum) | **N=32 × W=8** (256-logical-core Perlmutter optimum; measured — GL's W=16 is 14% slower here) |
| Concurrency | backgrounded `srun &` per slice | **one `srun -n NTASKS`** multi-task launch (NERSC rejects backgrounded `srun &` with "step creation disabled") |
| Output root | `/scratch/cavestru_root/...` | `/pscratch/sd/j/jibancat/` |
| Real LOA | not available | full access (the headline NERSC deliverable) |

**Science knobs are byte-identical to GL V1** (verified: PW100k reproduces the GL P/C
reference regime 0.818/0.904). The only non-plumbing change is `MAX_WORKERS` (per-cluster
parallelism, not a science knob — does not affect inference results).

## How to read a config

Configs **layer** via `source`: `_base.env` (science defaults) ← `_base_nersc.env`
(NERSC plumbing) ← `london0_nersc_v1.env` (V1 science + London paths) ← `2lpt*/loa*`
(swap paths) ← `loa_cddf` (FILTER/MAXDLAS override). Later assignments win.

**Sample count is one knob:** `NUM_SAMPLES` sets `NUM_DLA_SAMPLES`, `NUM_SUBDLA_SAMPLES`,
**and** the `.mat` grid paths together. Override it at launch:
`NUM_SAMPLES=100000 bash launch_nersc.sh london0_nersc_v1.env …`.

## The V1 production config (what it does)

`MAX_DLAS=4`, `SINGLE_ABSORBER_MODEL=1` (2-way single-absorber), `FILTER_LOW_LIKELIHOOD=1`,
`FILTER_N_INITIAL_FLOOR=5000`, `NUM_FOREST_LINES=31`, `NUM_LINES=3`, `MAX_LAMBDA=1250`,
`DLAMBDA=0.15`, `K=30`, `ENABLE_TAU_EB=1` / `TAU_EB_OBJECTIVE=null`, `EARLY_STOP_MODE=baseline`,
`PAIR_PRIOR_MODE=off` (clustering prior off), `BALMASK=false`, Turner `τ0=0.00246`/`β=3.62`,
PW prior NHI [17.2, 22.5]. **Production PW = 50k** (catalogs), **PW = 100k** (CDDF).

## Usage

```bash
# Mock catalog (PW50k default). --window large for load-balancing (~12-20 files/task).
bash slurm/nersc/production/launch_nersc.sh london0_nersc_v1.env --start 0 --end 1150 --window 384 --time 06:00:00

# Real LOA catalog (healpix mode; window = HPX chunk per sbatch).
bash slurm/nersc/production/launch_nersc.sh loa_nersc_v1.env --start 0 --end 16519 --window 512 --time 06:00:00

# Real LOA CDDF (PW100k, FILTER=0, MAX_DLAS=1).
bash slurm/nersc/production/launch_nersc.sh loa_cddf_nersc.env --start 0 --end 16519 --window 512 --time 12:00:00

# Always dry-run first (prints sbatch lines, no FS mutation, no submit):
bash slurm/nersc/production/launch_nersc.sh <flavour> … --dry-run
```
`launch_nersc.sh` flags: `--start/--end` (index window), `--window` (files-or-hpx per sbatch
— **drives load balancing**), `--qos`, `--time`, `--outdir`, `--dry-run`, `--no-sleep`.
Each run dir gets a `BASELINE.env` with the resolved knobs + `CODE_COMMIT` for provenance.

## Measured cost (Perlmutter, N32×W8, 2026-06-02)

| Run | config | node-hours |
|---|---|---|
| Mock catalog (London-0) | FILTER1, MAXDLA4, PW50k (measured 452 spec/min, ~4.25 s/spec) | **~36** |
| Mock catalog (2LPT-1/2) | same (assumed ≈ London; 2LPT per-spec not yet measured) | **~36** each |
| LOA catalog | FILTER1, MAXDLA4, PW50k (measured 2.657 s/spec, 942,946 QSO) | **~21.7** |
| LOA CDDF | FILTER0, MAXDLA1, **PW100k** (62 cpu-s/spec @ PW50k ×1.8–2) | **~120** |

**Mock vs LOA per-spec (measured V1, PW50k):** mock spectra are DLA-richer → more
refinement → ~452 spec/min (London, ~4.25 s/spec) vs real LOA ~722 spec/min (~2.66 s/spec).
So **mock catalogs ~36 nh, LOA catalog ~21.7 nh** — the difference is the DATASET, not packing.

**Load balancing is small (~7%), NOT the 1.7× first claimed:** the 722-vs-423 gap was a
dataset confound (LOA cheap vs DLA-rich mock), not imbalance. London spread-healpix (452)
≈ London completion (423) → ~7%. A larger `--window` (~12–20 files/task) still helps a
little — it shrinks the per-window completion tail (slowest file) and cuts the number of
sbatch jobs — but it is not a major node-hour lever. CDDF (FILTER=0, uniform) gets nothing.

## P/C calibration (London-0, 32-healpix slice, canonical cuts SNR>2 / NHI≥20.3 / pDLA>0.99 / no-BAL / lyβ-veto)

| PW | P (lyα) | C (lyα) | 85/85? |
|---|---|---|---|
| 10k | 0.840 | 0.897 | ✗ (purity, not pDLA-recoverable) |
| 30k | 0.860 | 0.907 | ✅ |
| **50k** | **0.852** | **0.906** | **✅ (production choice)** |
| 100k | 0.856 | 0.913 | ✅ (≈ GL ref → port faithful) |

FILTER-floor sweep (PW30k): P/C flat 0.853–0.860 across floor 2500→15000 → floor is past
diminishing returns; production keeps the default 5000.

## Runs

### Calibration (2026-06-01/02)

- **Parallelism sweep** (`nersc_parallelism_sweep_*`): N32×W8 optimal (301 spec/min PW100k).
- **Sample-count sweep** (`nersc_samplecost_sweep_*`): node-hours vs PW (above).
- **P/C calibration** (`nersc_calib_london0_S{10000,30000,50000,100000}`): the P/C table above.
- **FILTER-floor sweep** (`nersc_floor_S30000_F*`): floor is a no-op on P/C at PW30k.
- **Real-LOA cost benches** (`loa_cost_bench/`): LOA catalog + CDDF costs above.
- Investigation write-ups are in the **private notes repo** `github.com/jibanCat/desi_gpy_dla_notes`
  (`notes/2026-06-0{1,2}_*`), not committed here.

### Production — V1 catalogs

The three completed catalogs are packaged + shared to `desicollab/users/jibancat/DLA/...`.
P/C is the full [911,1216] window, NHI>20 / >20.3.

| Run | Output | P/C | nh |
|-----|--------|-----|----|
| London-0 mock (`nersc_prod_london0_v1_*`) | 1149/1150 hpx | 0.837/0.885 · 0.828/0.896 | ~36 |
| 2LPT-1 mock | 1149/1150 hpx, 799,162 rows / 362,445 TIDs | 0.818/0.874 · 0.815/0.888 | ~36 |
| **Real LOA catalog** (`nersc_prod_loa_v1_20260606`) | 16,519/16,519 hpx, 801,761 rows / 358,835 sightlines, 90.3% DLAFLAG-clean | — (real data) | ~34 |
| **Real LOA CDDF** (`nersc_cddf_loa_v1_20260609`, PW100k) | **in flight** (2026-06-11: hpx 0..9216 done, 22 jobs, 0 failures) | — (Pathway-A input) | ~125-130 proj |

## After a run finishes

```bash
# Combine per-healpix dlacat → one catalog (gap-checked + provenance):
python examples/combine_dlacat.py --procdir <OUTDIR> --out <OUTDIR>/combined_dlacat.fits
# P/C vs mock truth (mocks only):
python examples/molly_faithful_pc_plots.py --catalog-dir <OUTDIR> --truth <MOCKDIR>/dla_cat.fits \
    --bal-cat <MOCKDIR>/bal_cat.fits --no-bal --mockdir <MOCKDIR> --snr-min 2.0 --nhi-min 20.3 \
    --gp-conf 0.99 --lyb-veto --restrict-truth-to-processed --out <OUTDIR>/pc/
# Package + share (cluster-agnostic):
bash slurm/nersc/production/package_catalog.sh --rundir <OUTDIR> --share-to <CFS share>
```
Note: desi-DLAGP writes the processed h5 under `<OUTDIR>/figures/processed/`; the molly eval
expects `<OUTDIR>/processed/` — symlink it: `ln -sfn <OUTDIR>/figures/processed <OUTDIR>/processed`.

## Memory + termination (verified 2026-06-03)

**Memory fits N32×W8 with huge margin** (measured, 3 methods — meminfo/cgroup/sacct):
catalog PW50k peaks **102 GB** (20% of the 503 GB node), CDDF PW100k peaks **76 GB** (15%)
— ~0.3–0.4 GB/worker. The model+grids are shared page cache; per-spectrum arrays scale with
MAX_DLAS, so CDDF (MAXDLAS=1) is *lighter* than the catalog (MAXDLAS=4) despite 2× the QMC bag.
Keep N32×W8 for both; no repacking. (The GL OOM was on a 64 GB node — not relevant here.)

**Termination is graceful + recoverable.** Each healpix's `processed-*.h5` is written/closed
as that file finishes (`dlasearch.py:631`), so a kill keeps all completed healpix — only the
in-flight window is lost. A mid-write h5 truncation is caught (`resume_positions.py` validates
on open + core keys → truncated = not-done). Caveat: the `dlacat` FITS is written only at each
task's window-end (`desi-DLAGP.py:681`), so a window killed after its h5s but before its dlacat
leaves h5-but-no-dlacat — **`combine_dlacat.py --expect-positions N --fail-on-gap` is the
authoritative completeness gate** that catches it (always run it before trusting a catalog).

**Recovery recipe** (timeout/node-failure):
```bash
python slurm/nersc/production/resume_positions.py --mockdir <MOCKDIR> --procdir <OUTDIR>/figures/processed --summary
bash slurm/nersc/production/launch_nersc.sh <env> --outdir <OUTDIR> --start <first_not_done> --end <end> --window <W>
python examples/combine_dlacat.py --procdir <OUTDIR> --out <OUTDIR>/combined.fits --expect-positions <N> --fail-on-gap
```
Writes are idempotent (overwrite-safe), so `--start/--end` re-launch suffices; a
`launch_nersc_resume.sh` is only worth porting if a run leaves a *sparse scattered* gap set
(the NERSC inner script lacks the GL `LEVEL2_LIST` task branch).

## Pre-launch TODO / open items

- [x] **Load-balancing / throughput** — confirmed: London-0 spread-healpix = 452 spec/min (~36 nh);
  balancing is ~7% (dataset, not packing — see cost section).
- [x] **nfl=31** — confirmed correct (P/C insensitive to nfl=3 vs 31; `2026-05-13_filter_nfl_confirmation.md`).
- [x] **Memory + termination/resume** — verified (this section).
- [x] **CDDF PW100k cost** — measured on the real run (`nersc_cddf_loa_v1_20260609`):
  13.07 s/spec, ~125-130 nh projected with Option B, in the ~115–120 nh band. No longer extrapolated.
- [x] **2LPT per-spec** — 2LPT-1 ran at ≈ London (~36 nh); the assumption held.
