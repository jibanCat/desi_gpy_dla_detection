# GreatLakes production drivers

Additive GL-side replicas of the NERSC production pipeline (`slurm/configs/` +
`slurm/launch.sh` + `slurm/submit_desi_*.sh`). **Does not touch any
NERSC-side script.** The intent is to replicate the PR #7 production
runs (London / 2LPT / Saclay mocks) on GreatLakes using the mock
spectra mirrored at `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/`.

> First scaffolded 2026-05-20 on the `production_533` branch.
> Active NERSC jobs on `production_533` are not affected.

## Layout

```
slurm/greatlakes/production/
  _base_gl.env            # GL overlay: sources slurm/configs/_base.env then
                          # repoints REPO_ROOT, catalog/sample paths to /nfs/turbo
                          # and bakes in GL SLURM defaults (-A cavestru0 etc).
  london0_gl_v1.env       # London-0 mock, V1 production candidate (PR #7
                          # headline: P=0.8357 / C=0.8978 on the 5k slice).
  submit_desi_mock_gl.sh  # Inner sbatch script — GL analog of
                          # slurm/submit_desi_mock.sh. Body is identical;
                          # SBATCH directives + conda/libcerf setup differ.
  launch_gl.sh            # Outer driver — GL analog of slurm/launch.sh.
  logs/                   # Created on first launch; collects SLURM stdout/err
                          # for each sbatch job.
```

## How it differs from the NERSC pipeline

| Aspect | NERSC | GreatLakes |
|---|---|---|
| Repo root | `/pscratch/sd/j/jibancat/desi_gpy_dla_detection` | `/home/mfho/desi_gpy_dla_detection` (code), `/nfs/turbo/.../data/` (catalogs+samples) |
| Models | `${REPO_ROOT}/learnlogs/model_epoch_920.h5` | `/scratch/cavestru_root/cavestru0/mfho/phase2_desi/<run>/phase2_result.h5` |
| Mocks | `/global/cfs/projectdirs/desi/mocks/...` | `/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/...` |
| Output write area | `/pscratch/sd/j/jibancat/` | `/scratch/cavestru_root/cavestru0/mfho/` |
| SLURM account | `desi` | `cavestru0` |
| Partition / queue | `-q regular -C cpu` | `-p standard` |
| Env setup | `source /global/cfs/cdirs/desi/software/desi_environment.sh main` | `conda activate gpdla` + `LD_LIBRARY_PATH` for libcerf |

All *scientific* knobs (PREV_TAU_0, K, DLAMBDA, NUM_FOREST_LINES,
NUM_LINES, MIN/MAX_LAMBDA, normalization band, …) are sourced verbatim
from `slurm/configs/_base.env` via `_base_gl.env`. Flavour configs layer
the same per-mock overrides on top.

## Usage — V1 candidate replication smoke

Dry-run the launch (prints sbatch commands, doesn't submit):
```bash
bash slurm/greatlakes/production/launch_gl.sh london0_gl_v1.env --dry-run
```

1-window smoke (~62 spectra-16 files / ~1000 spectra, 1 sbatch):
```bash
bash slurm/greatlakes/production/launch_gl.sh london0_gl_v1.env --end 0
```

5k slice (matches PR #7 headline n_truth ≈ 581, 5 sbatch windows):
```bash
bash slurm/greatlakes/production/launch_gl.sh london0_gl_v1.env --end 320
```

Full London-0 mock (~1150 spectra-16 files, 18 sbatch windows):
```bash
bash slurm/greatlakes/production/launch_gl.sh london0_gl_v1.env
```

## What the V1 candidate config does

`london0_gl_v1.env` targets the PR #7 "Updated headline P/C — V1
production candidate" recipe (post-2026-05-19 matcher fix). Key
overrides over `_base_gl.env`:

- `LEARNED_FILE`: `2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5`
  (PR #6 corrected trainer, 2LPT-trained, matched normalization band
  [1425, 1475]).
- `MAX_LAMBDA = 1250` (overrides the `_base.env` default 1216.75).
- `MAX_DLAS = 4`, `SINGLE_ABSORBER_MODEL = 1` (2-way single-absorber;
  bumped 3 → 4 for DR2 alignment, P/C ~unchanged on the 5 k smoke).
- `FILTER_LOW_LIKELIHOOD = 1` (PR #5 FILTER fix #5).
- `NUM_DLA_SAMPLES = 100000`, PW prior NHI [17.2, 22.5]
  (`pw_samples_a3_172_225_100000.mat`) — production best baseline.
- `NUM_FOREST_LINES = 31` — must match the training-time setting
  (running inference at the 3 default biases log N_HI ~+0.06 dex high).
- `ENABLE_TAU_EB = 1`, `TAU_EB_OBJECTIVE = null` — τ-EB ON with the
  null objective (production best baseline).

The +log(N) and −log_ratio log-evidence patches are already in the code
path (PR #7 commits) — no flag needed.

> See the [PR #7 description](https://github.com/jibanCat/desi_gpy_dla_detection/pull/7)
> "Updated headline P/C — V1 production candidate" section for the exact
> recipe + the 2026-05-19 matcher-fix details.

## After a run finishes

Combine the per-window HDF5 catalogs:
```bash
python combine_processed_h5.py \
    --processed_dir "$OUTDIR" \
    --output_file   "$OUTDIR/combined.h5" \
    --mock
```

Run P/C eval with PR #7's exact recipe (NHI-desc matcher default after
the 2026-05-19 fix `b410393`):
```bash
python examples/molly_faithful_pc_plots.py \
    --catalog-dir "$OUTDIR" \
    --truth       "$MOCKDIR/dla_cat.fits" \
    --zcat        "$MOCKDIR/zcat.fits" \
    --bal-cat     "$MOCKDIR/bal_cat.fits" --no-bal \
    --truth-nhi-min 20.3 \
    --pdla-cut    0.99 \
    --lyb-veto \
    --restrict-truth-to-processed \
    --out "$OUTDIR/purity_completeness.md"
```

Compare against the PR #7 headline (P = 0.8357 / C = 0.8978 on the 5k
slice). The intra-batch P/C scatter on a 5k slice is ~0.6 pp; deltas
under ~1 pp are within the noise floor (per
`docs/notes/2026-05-16_config_confirmations.md`).

## Outputs

Each run writes to `${OUTDIR}/`:
- `outputs/processed-spectra-16-N.h5` — per-file inference HDF5.
- `outputs/dlacat-<release>-mockcat-<a>-<b>.fits` — per-window catalog.
- `outputs/logs/` — per-srun stdout/err.
- `BASELINE.env` — resolved env at submission time (audit trail).
- `<flavour>.env` — copy of the flavour config used.

## Known open items (ambiguities, not blockers)

- **Model name disambiguation**: PR #7 description says
  `2lpt_loa124_nohcd_nobal_wide_m`; the 05-16 sweep notes
  (`lambda_fine_and_gp_range.md`, `config_confirmations.md`) write the
  same model as `2lpt_loa124_nohcd_nobal_wide` (no `_m`). The `_m`
  variant uses MATLAB-matched norm band [1425,1475]; `_wide` (no `_m`)
  uses Garnett band [1310,1325]. We use `_wide_m` based on the PR
  description being most recent; if the replicated P/C deviates from
  the headline, retry against `_wide`.
- **PW sample-grid floor**: V1 candidate uses 50k PW samples; the
  NHI-low floor is ambiguous from the PR description alone. We use
  `pw_samples_a3_172_220_50000.mat` (matches the 2-way single-absorber
  convention in `slurm/configs/london0_y3_lls172.env`). The
  `_nhi198`-named NERSC OUTDIR is *output*-name choice, not the sample
  floor.
