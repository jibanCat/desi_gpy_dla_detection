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
  london0_gl_v1.env       # London-0 mock, V1 candidate (see PR #7 description
                          # for current headline P/C — the canonical merge-gate
                          # baseline is now the 2LPT-0 V1 run, see below).
  2lpt0_gl_v1.env         # 2LPT-0 mock, V1 production BASELINE — sources the
                          # London-0 config and only overrides mock paths /
                          # RELEASE / OUTDIR / TRUTH_CAT. The verified
                          # merge-gate catalog (PR #7 baseline table).
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

## Usage — V1 baseline replication

The canonical merge-gate baseline is `2lpt0_gl_v1.env` (2LPT-0 mock,
fully sourced from `london0_gl_v1.env`, only mock paths / RELEASE / OUTDIR
differ). `london0_gl_v1.env` runs the same recipe against the London-0
mock — kept as a sibling target.

Dry-run the launch (prints sbatch commands, doesn't submit):
```bash
bash slurm/greatlakes/production/launch_gl.sh 2lpt0_gl_v1.env --dry-run
```

1-window smoke (~12 spectra-16 files, 1 sbatch at OUTER_WINDOW=10):
```bash
bash slurm/greatlakes/production/launch_gl.sh 2lpt0_gl_v1.env --end 0
```

Full 2LPT-0 mock (1150 spectra-16 files; chunked into ~96 sbatches at
OUTER_WINDOW=10 / OUTER_STEP=12):
```bash
bash slurm/greatlakes/production/launch_gl.sh 2lpt0_gl_v1.env
```

Subset (`--end N` runs positions `0 .. N*OUTER_STEP`; pick `N` to size
the slice you want):
```bash
bash slurm/greatlakes/production/launch_gl.sh 2lpt0_gl_v1.env --end 40
```

## What the V1 baseline config does

`london0_gl_v1.env` (and `2lpt0_gl_v1.env`, which sources it) target the
PR #7 V1 production-baseline recipe. Key overrides over `_base_gl.env`:

- `LEARNED_FILE`: `2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5`
  (PR #6 rebuilt trainer, 2LPT-trained, MATLAB-matched normalization
  band [1425, 1475] — the `_m` suffix). Resolved 2026-05-26: the `_wide`
  (no `_m`) Garnett-band variant is NOT the production choice.
- `MAX_LAMBDA = 1250` (overrides the `_base.env` default 1216.75).
- `MAX_DLAS = 4`, `SINGLE_ABSORBER_MODEL = 1` (2-way single-absorber;
  bumped 3 → 4 for DR2 alignment; P/C ~unchanged on the
  `gl_maxdla4_london0_v1` sensitivity full-London run).
- `FILTER_LOW_LIKELIHOOD = 1` (truncated importance sampler — FILTER
  fix #5 plus the 2026-05-26 −log_ratio region-A/B correction at
  `25f32ae`).
- `NUM_DLA_SAMPLES = 100000`, PW prior NHI [17.2, 22.5]
  (`pw_samples_a3_172_225_100000.mat`) — extended ceiling to 22.5,
  matches the production best baseline.
- `NUM_FOREST_LINES = 31` — must match the GP's training-time
  `de_forest_num_lines` to avoid μ/Ω² mean-flux mis-scaling. NOTE: the
  +0.06 dex N_HI bias seen on production is INTRINSIC to the sub-DLA
  prior edge (resolved 2026-05-22), NOT caused by NUM_FOREST_LINES —
  NF=3 vs NF=31 inference give numerically-identical NHI on controlled
  comparisons.
- `ENABLE_TAU_EB = 1`, `TAU_EB_OBJECTIVE = null` — τ-EB on with the
  null objective (production best baseline).

The +log(N) and −log_ratio log-evidence patches are always-on code
changes on this branch (`25f32ae` and an earlier log-N commit) — no
runtime knob; any catalog produced from `production_533` HEAD has both.

> See the [PR #7 description](https://github.com/jibanCat/desi_gpy_dla_detection/pull/7)
> baseline-config table for the canonical recipe and the current
> headline P/C numbers on the 2LPT-0 V1 catalog. The earlier London-0
> 5k preview (P=0.8357/C=0.8978, pre-`25f32ae`) is superseded by the
> 2LPT-0 V1 full-catalog baseline.

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

Compare against the PR #7 baseline-config table (current 2LPT-0 V1
headline at NHI≥20.3: P=0.8181 / C=0.8910). The intra-batch P/C scatter
on a 5k slice is ~0.6 pp; deltas under ~1 pp are within the noise floor
(per `docs/notes/2026-05-16_config_confirmations.md`).

## Outputs

Each run writes to `${OUTDIR}/`:
- `outputs/processed-spectra-16-N.h5` — per-file inference HDF5.
- `outputs/dlacat-<release>-mockcat-<a>-<b>.fits` — per-window catalog.
- `outputs/logs/` — per-srun stdout/err.
- `BASELINE.env` — resolved env at submission time (audit trail).
- `<flavour>.env` — copy of the flavour config used.

## Resolved items (kept for trail)

- **Model name (`_wide_m` vs `_wide`)** — RESOLVED 2026-05-26. `_wide_m`
  (MATLAB-matched norm band [1425,1475]) is the production choice, used
  in both `london0_gl_v1.env` and `2lpt0_gl_v1.env`. The `_wide` Garnett
  band [1310,1325] variant is not used. Earlier README hedge ("if the
  replicated P/C deviates, retry against `_wide`") is no longer relevant.
- **PW sample-grid floor** — RESOLVED. V1 uses `NUM_DLA_SAMPLES=100000`
  with `pw_samples_a3_172_225_100000.mat` (NHI prior [17.2, 22.5]) — see
  the "What the V1 baseline config does" bullets above. The earlier 50k
  / [17.2, 22.0] grid (`pw_samples_a3_172_220_50000.mat`) was the
  pre-extension setting; superseded by the [17.2, 22.5] PW-100k baseline.
