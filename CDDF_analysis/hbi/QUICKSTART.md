# `CDDF_analysis/hbi/` — QUICKSTART

Shortest runnable path from a frozen GP-DLA catalog to a selection-corrected
dN/dX / Omega_DLA / f(N_HI) band.

## Input (what you need before starting)

| item | example path |
|---|---|
| Frozen combined catalog dir (FILTER=on, MAXDLA=4) | `/scratch/.../gl_prod_2lpt0_v1_20260526/combined_catalog/` |
| Mock truth catalog (for kernel build) | `/nfs/turbo/.../hcd_truth_cat.fits` |
| BAL catalog | `/nfs/turbo/.../bal_cat.fits` |
| Molly purity/completeness matrix (`.tsv`) | `/scratch/.../figures_molly/molly_matrix.tsv` |
| Per-healpix processed-h5 glob (stage1 SIR kernel only) | `/scratch/.../outputs/figures/processed/processed-spectra-16-*.h5` |
| PW samples (stage1 SIR kernel only) | `/scratch/.../data/dr12q/processed/pw_samples_a3_172_225_100000.mat` |
| WALL1_GATE_FROZEN.md (committed PASS/KILL criteria, must exist in OUT before running) | place in `$OUT/WALL1_GATE_FROZEN.md` |

Repo: `/home/mfho/desi_gpy_dla_detection`. Conda env: `gpdla`.

---

## Two kernel options

### Option A — R_emp kernel (lighter, recommended for cross-mock)

Builds the empirical truth-match response R_emp(x_true|x_hat,SNR) — reads only
the truth catalog, no processed-h5 files required. ~1–3 hours on a 16-core GL
`standard` node.

```bash
# Submit via SLURM (GreatLakes):
sbatch slurm/greatlakes/production/remp_kernel.sbatch

# Or run interactively from the repo root (override paths as needed):
export OUT=/scratch/.../r_emp
export CAT=/scratch/.../combined_catalog/
python CDDF_analysis/hbi/run_remp_kernel.py \
    --stage all \
    --out "${OUT}" \
    --catalog-dir "${CAT}" \
    --truth /nfs/turbo/.../hcd_truth_cat.fits \
    --bal-cat /nfs/turbo/.../bal_cat.fits \
    --molly-tsv /scratch/.../molly_matrix.tsv \
    --zbins "2.0,2.5,3.0,3.5" \
    --report-limits "20.0,20.3,20.6" \
    --fit-floor 19.5 \
    --lambda-bspbody 30.0 \
    --lam-rf-min 911.0 \
    --dalpha 0.5 \
    --host-truth-floor 19.0 \
    --smooth-bins 1.0 \
    --n-floor 20 \
    --host-col NHI_TILT_HOST \
    --n-mc 200 \
    --seed 0 \
    --n-jobs 16
```

Key flags:
- `--stage all` — build R_emp kernel, run bspbody MAP fit, run WALL-1 tilt closure (runs the three stages sequentially in one invocation; pass `--stage build`, `--stage 2`, or `--stage 3` to run just one — the SLURM wrapper loops over them)
- `--fit-floor 19.5` — detection-row floor for the bspbody fit
- `--lambda-bspbody 30.0` — 2nd-diff curvature penalty (swept minimal-DOF rung)
- `--dalpha 0.5` — WALL-1 tilt magnitude (runs +/- dalpha)
- `--report-limits "20.0,20.3,20.6"` — integrated dN/dX and Omega closure thresholds
- `--n-mc 200` — MC draws for the marginalized band (use >= 240 for publication)

Outputs land in `$OUT/`: `posterior_kernel_2lpt0.npz`, `wall1_result.tsv`,
`wall1_pulls_*.csv`, and figures in `$OUT/figures/`.

---

### Option B — SIR posterior kernel (full pipeline, requires processed-h5)

Builds the 2-D posterior kernel from the per-healpix processed-h5 files (heavier,
~3–8 hours). Submit via:

```bash
sbatch slurm/greatlakes/production/phase3d_postkernel.sbatch
```

Or run interactively:

```bash
python CDDF_analysis/hbi/run_phase3d_postkernel.py \
    --stage all \
    --out "${OUT}" \
    --catalog-dir "${CAT}" \
    --truth /nfs/turbo/.../hcd_truth_cat.fits \
    --bal-cat /nfs/turbo/.../bal_cat.fits \
    --molly-tsv /scratch/.../molly_matrix.tsv \
    --processed-glob "/scratch/.../processed/processed-spectra-16-*.h5" \
    --pw-samples /scratch/.../pw_samples_a3_172_225_100000.mat \
    --zbins "2.0,2.5,3.0,3.5" \
    --report-limits "20.0,20.3,20.6" \
    --fit-floor 19.5 \
    --lambda-bspbody 30.0 \
    --dalpha 0.5 \
    --host-truth-floor 19.0 \
    --n-mc 200 \
    --seed 0 \
    --n-jobs 16
```

Stage meanings: `stage1` = build kernel cache, `stage2` = bspbody MAP fit,
`stage3` = WALL-1 tilt closure; `--stage all` runs all three sequentially.
S3 falsifiers (prior-null + dense-synthetic injection) run via `--stage s3_all`.

---

## Where figures land

Both drivers write figures to `$OUT/figures/`. Key outputs:

| file | content |
|---|---|
| `figures/fig_compare_integrated.png` | integrated dN/dX and Omega: HBI vs raw feed-forward vs truth, R0 annotated |
| `figures/fig_compare_fN.png` | differential f(N_HI): HBI vs raw tail |
| `wall1_result.tsv` | WALL-1 tilt closure R0 pulls per tilt arm |
| `posterior_kernel_2lpt0.npz` | cached 2-D kernel (reusable across stages) |

---

## Approximate runtime (GreatLakes standard, 16 cores)

| stage | R_emp path | SIR-kernel path |
|---|---|---|
| kernel build | ~20 min | ~3–5 h (1150 processed-h5) |
| bspbody fit (stage 2) | ~10 min | ~10 min |
| WALL-1 tilt closure (stage 3) | ~30 min | ~30 min |
| **total** | **~1 h** | **~4–6 h** |

---

## SLURM resource requirements

Both sbatch scripts request: `-p standard`, `-N 1`, `-n 16`, `--mem 48–64G`.
Account: `-A cavestru1`. BLAS threads pinned to 1 (set in sbatch header).

---

## Known constraints

- `WALL1_GATE_FROZEN.md` MUST exist in `$OUT/` before submitting — the script
  aborts with exit code 2 if it is missing.
- Use `--n-mc >= 240` for publication-quality bands (200 is sufficient for
  development checks).
- The on-mock R0 is a self-consistency check (not a calibration transfer claim).
  See `hbi/README.md` for the uncertainty budget and cross-mock validation status.
