# Handoff — 2026-05-13 12:00 PT

> Today's session resumed yesterday's interrupted work and ran the gating
> diagnostic that yesterday's design notes flagged as next-most-important.
> Resume jobs are running inline on this jupyter compute node (256 CPUs,
> nid004179, job 52907557, expires 15:44 PT). Partial outputs are durable
> on disk; nothing relies on a process that won't survive session expiry.

## TL;DR for next-Claude

1. **Var[Δ_marg] gating diagnostic is done.** Verdict in
   [`docs/notes/2026-05-13_var_delta_marg_diagnostic.md`](docs/notes/2026-05-13_var_delta_marg_diagnostic.md):
   **statistic-limited, not sampling-limited** at production N=50k. Sampling
   noise σ ≈ 0.1, signal-null gap ≈ 13 — noise is ~130× below the signal.
   This kills the case for MLMC / pocoMC / variance-reduction sampler work
   on the low-SNR P/C ceiling. The lever is model-side. **Read the verdict
   before proposing any sampler upgrades.**

2. **PR #7 still draft.** Two new commits on `production_533` after PR #7
   was raised, both relevant to today's findings:
   - `2c499a8 feat: EARLY_STOP_MODE flag for multi-DLA inference + resume scripts`
   - `86ad225 diag: Var[Δ_marg] gating diagnostic + 2026-05-13 verdict note`

3. **Yesterday's "in-flight" pickup list was misleading.** The 5 investigations
   were running *inline in the previous jupyter session*, not as sbatch.
   When that session expired they died. On resume today, 3 of 8 slices were
   on disk per cell (durable); 5 of 8 needed re-running. Lesson saved to
   memory ([`feedback_long_runs_need_sbatch.md`](../../global/homes/j/jibancat/.claude/projects/-pscratch-sd-j-jibancat-desi-gpy-dla-detection/memory/feedback_long_runs_need_sbatch.md)).

4. **Today's resume is running inline in this jupyter session** because the
   regular sbatch queue is ~3 days deep. Tracked via watcher
   `b9f0r2a9j` (Bash background); will notify when all 22 missing slices
   complete. Partial slices' outputs are durable on disk.

---

## Resume run state (as of 12:00 PT)

| OUTDIR | Done before today | Missing | Resume started |
|---|---:|---|---|
| `london_v3_loa124_early_stop_A` | 3 | 0 2 4 6 7 (5) | 11:56 PT (re-launch, see "Gotcha" below) |
| `london_v3_loa124_early_stop_D` | 3 | 0 2 4 6 7 (5) | 11:56 PT (re-launch) |
| `joint_dla_subdla_sweep/cellA_md3_nhi19to23` | 3 | 2 4 5 6 7 (5) | 11:45 PT |
| `joint_dla_subdla_sweep/cellB_md4_nhi19to23` | 6 | 0 6 (2) | 11:45 PT |
| `joint_dla_subdla_sweep/cellC_md3_nhi172to22` | 3 | 0 2 4 6 7 (5) | 11:45 PT |

**Watcher**: Bash background `b9f0r2a9j`, polls for `total run time:` lines
across all slice logs, exits when 22 are present.

**If jupyter dies before resume completes**: the durable outputs survive on
disk. To resume the remaining missing slices, re-run:

```bash
cd /pscratch/sd/j/jibancat/desi_gpy_dla_detection
# Re-derive the missing list per cell by checking which
# processed-spectra-16-*.h5 are present in each OUTDIR's processed/ dir.
# Then for early-stop A and D, USE THIS EXPORT BLOCK:
export LEARNED_FILE="/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted/2lpt_loa124_nohcd_nobal_wide.h5"
export DLA_SAMPLES_FILE="/pscratch/sd/j/jibancat/desi_gpy_dla_detection/data/dr12q/processed/pw_samples_a3_190_220_50000.mat"
export NUM_DLA_SAMPLES=50000
export ENABLE_TAU_EB=1
export TAU_EB_OBJECTIVE=null
# For cellA/B/C, the config under joint_dla_subdla_sweep/configs/ is self-contained.
CONFIG_PATH=... OUTDIR=... MISSING_SLICES="..." EARLY_STOP_MODE=A \
    bash slurm/resume_local.sh
```

### Gotcha caught and fixed this session

`slurm/configs/london0_y3.env` falls back to `_base.env` defaults for
`NUM_DLA_SAMPLES` (10k), `DLA_SAMPLES_FILE` (`dla_samples_a03.mat`), and
`LEARNED_FILE` (`learnlogs/model_epoch_920.h5`). The original early-stop A/D
run had *overridden* these via exported env vars before sourcing the config
(v3 GP, 50k pw_samples, `ENABLE_TAU_EB=1`). The first resume launch this
session inherited the config defaults instead, producing a 10k-sample run.
Caught at ~10 min in, killed cleanly (no h5 written), and re-launched
correctly. The `RESUME_LOCAL_*.md` in each OUTDIR records the actual values
used per relaunch.

**Rule for next-Claude**: when resuming a partial run, diff the proposed
python command against the original `RUN_SETTINGS.md` in the OUTDIR *before*
launching. Don't trust the config file alone — production-style configs at
this repo defer most knobs to `${VAR:-default}` and the original launch
exported them externally.

---

## Pickup work, in order of priority

### When resume completes (notification will arrive from watcher `b9f0r2a9j`)

1. Verify all 22 slices wrote h5s + dlacats:
   ```bash
   for d in london_v3_loa124_early_stop_A london_v3_loa124_early_stop_D \
            joint_dla_subdla_sweep/cellA_md3_nhi19to23 \
            joint_dla_subdla_sweep/cellB_md4_nhi19to23 \
            joint_dla_subdla_sweep/cellC_md3_nhi172to22; do
     echo "$d: $(ls /pscratch/sd/j/jibancat/prod533_5k_20260511/$d/processed/*.h5 | wc -l) h5"
   done
   ```
   Each should report 8.

2. Combine per cell:
   ```bash
   for d in london_v3_loa124_early_stop_A london_v3_loa124_early_stop_D \
            joint_dla_subdla_sweep/cellA_md3_nhi19to23 \
            joint_dla_subdla_sweep/cellB_md4_nhi19to23 \
            joint_dla_subdla_sweep/cellC_md3_nhi172to22; do
     OUTDIR=/pscratch/sd/j/jibancat/prod533_5k_20260511/$d
     python combine_processed_h5.py --processed_dir "$OUTDIR" \
         --output_file "$OUTDIR/combined.h5" --mock
   done
   ```

3. P/C tables, per the early-stop-bug pickup plan in yesterday's handoff:
   - `examples/molly_faithful_pc_plots.py` for each of {baseline, A, D}
   - SNR cuts {2, 4, 6} × P_DLA cuts {0.99, 0.999, 0.99999}
   - Baseline reference: `/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/combined.h5`
   - Decision: deploy A, deploy D, or neither?

4. Joint sweep evaluation, per yesterday's handoff:
   - For each of cellA/B/C, build a dlacat with per-DLA NHI binning
     (each MAP DLA → "sub-DLA" if `log N_HI ∈ [19, 20.3]`, else "classical")
   - Two P/C tables per cell against `dla_cat.fits` truth
     (classical N_HI ≥ 20.3, sub-DLA [19, 20.3])
   - Does any cell hit user targets (classical 85-90/85, sub-DLA 85/70)?
   - Helper: `joint_dla_subdla_sweep/_evaluate_cell.py` is on disk; adapt or rerun.

5. Update PR #7 description with the new findings.

### Lower priority (the long horizon)

- **Var[Δ_marg] verdict implies dropping bespoke MLMC/pocoMC for this purpose.**
  See [§"Implications"](docs/notes/2026-05-13_var_delta_marg_diagnostic.md#implications)
  for what *does* remain on the table (Marginal-MAP, prior reweighting,
  model-side levers).

- **The 4 medium-priority audit items** (filter_sweep, dilution_test caveat,
  tau_eb_5cand, map_detection_test) were flagged for inclusion in PR #7
  yesterday; user deferred — still open.

---

## What today added to PR #7

Two new commits, both already on `production_533`:

1. `2c499a8 feat: EARLY_STOP_MODE flag for multi-DLA inference + resume scripts`
   - Plumbs `--early_stop_mode {baseline,A,D}` through `desi-DLAGP.py` →
     `DLAHolder` → `dlasearch_{hpx,mock}` → `DLAGP`. Default unchanged.
   - Adds `slurm/resume_missing_slices.sh` (sbatch) and
     `slurm/resume_local.sh` (inline for queue-impractical sessions).
2. `86ad225 diag: Var[Δ_marg] gating diagnostic + 2026-05-13 verdict note`
   - `examples/var_delta_marg.py` re-analysis (read-only, ~70 s wall).
   - `docs/notes/2026-05-13_var_delta_marg_diagnostic.md` falsifiable-test
     write-up + verdict.

Push of these commits to GitHub will happen at end of session.

---

## Memory updates this session

- Updated [`feedback_long_runs_need_sbatch`](.../memory/feedback_long_runs_need_sbatch.md)
  with three rules now:
  1. Default sbatch BUT check queue depth — fall back to inline jupyter
     + nohup if queue >> remaining session.
  2. Handoff notes must reflect reality (no in-flight entries for compute
     that won't outlive the session).
  3. When resuming a partial run, the original `RUN_SETTINGS.md` is
     authoritative; diff the proposed python command against it before
     relaunching.

---

## Compute env

Same shell preamble as before:
```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
python ...
'
```

Current jupyter compute node `nid004179` (256 CPUs, 487 GB mem, jupyter job
52907557). Started 09:44 PT, expires 15:44 PT — plenty of margin for
resume completion. If pickup work spans past 15:44, salloc:
```bash
salloc -N 1 -C cpu -q interactive -t 4:00:00 -A desi
```
Or just open a fresh jupyter session — most pickup work after the resume is
re-analysis on the combined.h5 outputs and doesn't need a full compute node.

---

## Key files to read in this order on next session

1. `HANDOFF.md` (this file)
2. `docs/notes/2026-05-13_var_delta_marg_diagnostic.md` (the verdict that
   redirects future sampler work)
3. `docs/notes/2026-05-12_multidla_early_stop_bug.md` (the bug A/D test)
4. `docs/development_map.md` (repo orientation)
5. `RESUME_LOCAL_*.md` files in each resumed OUTDIR (authoritative
   reproduce-this-run config)
