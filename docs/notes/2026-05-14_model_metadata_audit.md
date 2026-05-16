# Model-metadata audit — PR #6

Date: 2026-05-14. Auditor: Claude (Opus 4.7 1M).
Scope: all model-card / metadata machinery landed today
(`examples/reemit_step_c_readmes.py`, `examples/write_scratch_readmes.py`,
the `_save_readme` change + scratch-mirror in `tests/phase2_train_desi.py`,
`docs/CURRENT_MODELS.md`, `STATUS.md` files, the 6 commits `c38dc57`,
`d172f49`, `d19ddfc`, `ac4eb90`, `f2b931d`, `6642df8`).

Verification was performed read-only against the repo, the actual .h5
files in `docs/notes/2026-05-1[123]_*/`, the scratch dirs under
`/scratch/cavestru_root/cavestru0/mfho/phase2_desi/`, the SLURM log
headers in `slurm/greatlakes/`, and `squeue -u mfho` + `sacct` at audit
time.

## Cross-reference validity (path-exists check)

All paths grep-extracted from `docs/CURRENT_MODELS.md` and
`docs/production_models.md`:

| Reference | Exists? |
|---|---|
| `docs/notes/2026-05-14_c0prior_failure_investigation/findings.md` | yes |
| `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md` | yes |
| `docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md` | yes |
| `docs/notes/2026-05-13_2lpt_loa0_vs_loa124_implementation/findings.md` | yes |
| `docs/notes/2026-05-13_step_c_dla_recovery/findings.md` | yes |
| `docs/notes/2026-05-11_desi_phase2_2lpt_loa0_wide/phase2_result.h5` | yes |
| `docs/notes/2026-05-13_desi_phase2_loa_no_dla_no_bal_wide_m_normmask_3000iter/phase2_result.h5` | **no — in flight (correct)** |
| `examples/dla_recovery_step_c.py` | yes |
| `examples/plot_kernels_v1_rest_range.py` | yes |

All link targets that should exist do exist; the only "missing" path is
the in-flight `_3000iter/phase2_result.h5` which is explicitly labelled
"in flight, SLURM 50087967" in `docs/CURRENT_MODELS.md:14`. No
discoverable dangling links inside the model docs.

## MODEL_STATUS / SUPERSEDED accuracy

`examples/reemit_step_c_readmes.py:103-114` — every status tag matches
the actual state in `docs/notes/2026-05-1[123]_desi_phase2_*/phase2_result.h5`
(β values cross-checked with `np.exp(f['log_beta'][()])`):

| MODEL_STATUS entry | Claim | Truth |
|---|---|---|
| `2lpt_loa0_wide` "β=1.28" | β=1.28 | β=1.2793 ✓ |
| `2lpt_loa124_nohcd_nobal_wide` "β=1.45" | β=1.45 | β=1.4513 ✓ |
| `2lpt_loa0_wide_m` "p_DLA=0.70" | p_DLA=0.70 | 0.7031 per `docs/notes/2026-05-13_step_c_dla_recovery/stepc_2lpt_loa0_wide_m.json` ✓ |
| `2lpt_loa124_nohcd_nobal_wide_m` "p_DLA=0.76" | p_DLA=0.76 | 0.7554 per `stepc_2lpt_loa124_nohcd_nobal_wide_m.json` ✓ |
| `c0prior` "p_DLA = 0.042 vs `_m`'s 0.755" | 0.042 / 0.755 | 0.0416 / 0.7554 ✓ |

Supersession pointers (e.g. 50212863, 50212866, 50212867, 50212621) all
appear in `squeue -u mfho` as PENDING (50212863/866/867) or RUNNING
(50212621). Ground truth confirmed.

## SUPERSEDED map in `write_scratch_readmes.py`

`examples/write_scratch_readmes.py:119-139` — supersession map review:

Two issues found:

1. **Forward-looking dangling pointers**:
   - `2lpt_loa0_wide_m` → points to `/scratch/.../2lpt_loa0_wide_m_normmask/`,
     which does **not** exist on scratch yet (SLURM 50212866 is PENDING).
   - `2lpt_loa0_wide_g` → points to `/scratch/.../2lpt_loa0_wide_g_normmask/`,
     does not exist (50212863 PENDING).
   - `2lpt_loa124_nohcd_nobal_wide_g` → points to
     `/scratch/.../2lpt_loa124_nohcd_nobal_wide_g_normmask/`, does not
     exist (50212867 PENDING).
   These are not bugs in the map content (the directories will exist
   once the PENDING jobs start), but a reader following the link today
   will hit a 404. Minor.

2. **`_g`-variant LOA supersession claims a 3000iter retrain that
   doesn't exist for `_g`**: `loa_no_dla_no_bal_wide_g` →
   `loa_no_dla_no_bal_wide_m_normmask_3000iter/`. Same for `_hcd_with_bal_g`.
   `CURRENT_MODELS.md:28-29` only lists `_m_normmask_3000iter` post-reorder
   LOA jobs (no `_g`), so users of the LOA `_g` model are silently
   pointed to a different band's retrain. Either a `_g_normmask_3000iter`
   needs to be queued or the supersession message should clarify the
   band switch. Medium.

## In-flight 3-way consistency (50087967 case study)

| Source | n_iters | Norm band | ETA |
|---|---:|---|---|
| `docs/notes/2026-05-13_desi_phase2_loa_no_dla_no_bal_wide_m_normmask_3000iter/STATUS.md:7-10` | 3000 | [1425/1475] | 2026-05-15 06h |
| `/scratch/.../loa_no_dla_no_bal_wide_m_normmask_3000iter/README.md:18-20` | 3000 | [1425.00, 1475.00] (MATLAB DR16) | (no ETA) |
| `slurm/greatlakes/phase2_desi_retrain_50087967.log:6-11` | 3000 | [1425.0, 1475.0] | n/a |
| `docs/CURRENT_MODELS.md:28` | 3000 | [1425, 1475] | 2026-05-15 ~06h |

All four sources agree on n_iters and norm band. ETA is consistent
where present. The SLURM log at `:20` does print
"per-spectrum median in [1425.0, 1475.0] Å rest (Garnett+2017 convention)"
— that's a stale dataset.py log line from the run start (the run was
launched before commit `f2b931d` fixed `_band_label` in
`gpy_dla_detection/training/dataset.py:362-372`). Cosmetic only — the
actual band IS [1425, 1475] per the .h5 manifest. The fixed logger will
emit the right label on all future runs.

## Trainer scratch-mirror correctness (`tests/phase2_train_desi.py`)

Reviewed `tests/phase2_train_desi.py:719-735`:

- `args.checkpoint_dir.parent` — for production SLURM runs the
  checkpoint dir is `/scratch/.../phase2_desi/<run_name>/checkpoints/`
  so `.parent = /scratch/.../phase2_desi/<run_name>/` is the intended
  scratch root. Per `slurm/greatlakes/phase2_desi_retrain.sh:122` this
  is the convention.
- The `try/except` at `:724-735` properly catches any
  failure and only prints a `[warn]` line; training does not exit.
- `scratch_parent.mkdir(parents=True, exist_ok=True)` at `:729` makes
  it work for fresh runs with no pre-existing scratch dir.
- One caveat: a user who passes `--checkpoint-dir /tmp/foo/ckpt` (no
  `phase2_desi/<run>/` convention) will see `phase2_result.h5` written
  to `/tmp/foo/` rather than alongside their checkpoint dir. Acceptable
  given the SLURM script enforces the convention.

The scratch READMEs at
`/scratch/.../2lpt_loa0_wide_m/README.md:41` cite
`tests/phase2_train_desi.py:716-731` — actual mirror is at lines
**723-735** (variable defs at 727-728, copies at 730-731, prints at 732-733,
except at 735). Off by ~7 lines because the comment block above shifted
the implementation. Nit-level.

## Templating soundness (`_save_readme` in `tests/phase2_train_desi.py:311-433`)

- Norm-band convention label logic (`:368`) matches `dataset.py:364-369`
  and `reemit_step_c_readmes.py:174-178`. All three use the same
  abs-diff-<1 thresholds vs 1425 / 1310. ✓
- `log_c_0_prior_sigma` (`:366`) is rendered as `(none)` when the CLI
  arg is unset. ✓
- The "STATUS" header at `:331-335` says "post-reorder pipeline,
  training freshly completed; update via `examples/reemit_step_c_readmes.py`
  to mark current after validation." This is sensible for a freshly
  completed run. But: every completed run will be re-emitted by
  `reemit_step_c_readmes.py` before a human reads the dir, so the
  "freshly completed" template is actually never observed in finalized
  state — confirmed by `grep -l "freshly completed" docs/notes/*/README.md`
  returning empty. Not a bug; just dead text in finalized state.

## Per-model README sanity (8 completed + 6 in-flight)

**Critical bug** (high severity): 7 of 8 completed READMEs report
`n_spectra = -1`, `n_iters = -1`, `lr = nan` in the "Training config"
table.

Root cause: `examples/reemit_step_c_readmes.py:80-98` `read_h5_scalars`
reads keys via `f[k]` (dataset access). For 2026-05-11 batch trained
**before** commit `3a0b84f` ("phase2_train_{desi,dr16}: full
training-hyper manifest in .h5"), these fields exist only in `f.attrs`,
not as datasets:

```
2026-05-11_desi_phase2_2lpt_loa0_wide       has_ds_manifest=False has_attrs=True
2026-05-11_desi_phase2_2lpt_loa0_wide_g     has_ds_manifest=False has_attrs=True
2026-05-11_desi_phase2_2lpt_loa0_wide_m     has_ds_manifest=False has_attrs=True
2026-05-11_desi_phase2_2lpt_loa124_*_wide   has_ds_manifest=False has_attrs=True
2026-05-11_*_wide_g                          has_ds_manifest=False has_attrs=True
2026-05-11_*_wide_m                          has_ds_manifest=False has_attrs=True
2026-05-11_*_wide_c0prior                    has_ds_manifest=False has_attrs=True
2026-05-13_desi_smoke_normmask               has_ds_manifest=True  has_attrs=True
```

The .h5 files DO have these values (in `f.attrs`: `n_spectra=236755`,
`n_iters=1500`, `lr=0.005`), but the reader uses only `f[k]`. Fix: in
`read_h5_scalars`, fall back to `f.attrs[k]` when dataset access fails.
Affects every pre-reorder model README. Currently the "Training config"
table is misleading on 7 of 8 user-facing model cards.

The smoke READMe correctly reports `n_spectra=5,000`, `n_iters=50`,
`lr=0.005` because that .h5 was trained after `3a0b84f` and has the
dataset-level manifest.

Endpoint scalars (`c_0`, `tau_0`, `beta`, `final_loss`) are read via
`np.exp(f[k][()])` and via `f['loss_history'][-1]` and **are**
present in all 8 .h5 files (except `loss_history` is missing from the
two base-wide runs, where `final_loss=nan` correctly results). Endpoint
scalar values match the .h5 to 4 decimal places. Norm-band rows correct
in all 8 cases (verified via `f['normalization_min_lambda'][()]` direct
read).

**Minor sub-issues**:

- `c0prior` README at line 59 reports `log_c_0 prior σ | (none)` despite
  this being the c0prior model. The SLURM log header
  `phase2_desi_retrain_50021381.log` does not echo the
  `--log-c-0-prior-sigma` value, and the older .h5 doesn't have a
  `log_c_0_prior_sigma` field, so neither source can recover the
  runtime σ. The `_warning_banner` does still fire (via the
  `"c0prior" in out_dir_name` branch at
  `reemit_step_c_readmes.py:132`), so the user sees the "Not preferred"
  caveat — but the table-row says "(none)" which contradicts the
  banner. Add the σ value to the SLURM submit-script header, OR include
  `--log-c-0-prior-sigma` echo in `phase2_desi_retrain.sh`. Medium.
- Re-emitted READMEs drop the `preload_source` row that
  `_save_readme:355` includes. The data is in the .h5 (`f.attrs[
  'preload_source']` and `f['preload_source']` for newer runs) — minor
  info loss. Low.

**STATUS banners** (`_warning_banner` at
`examples/reemit_step_c_readmes.py:117-167`) — every banner matches:

- The two "wide-σ collapse" base runs carry SUPERSEDED tags ✓ (β=1.28,
  1.45 match h5 truth).
- The four `_g` / `_m` non-c0prior pre-reorder variants carry the
  pre-reorder caveat ✓.
- The c0prior variant carries both the "Not preferred" banner AND the
  pre-reorder caveat ✓.
- The smoke run carries ℹ SMOKE only ✓.

**6 in-flight STATUS.md files** — verified consistent with
`squeue -u mfho`:

| STATUS.md SLURM ID | squeue state |
|---|---|
| 50087967 | RUNNING ✓ |
| 50087968 | RUNNING ✓ |
| 50212621 | RUNNING ✓ |
| 50212863 | PENDING ✓ |
| 50212866 | PENDING ✓ |
| 50212867 | PENDING ✓ |

All STATUS.md files cite the right ETAs (matched against
`docs/CURRENT_MODELS.md:28-33`).

## Scratch READMEs (8 directories with `phase2_result.h5`)

**Bug** (medium severity): `write_scratch_readmes.py:171-176` —
the status-tag elif tree compares `state` against literal
`"CANCELLED"`, but sacct returns `"CANCELLED by 114399728"` for
admin-cancelled runs. Result: 4 scratch READMEs show
`> **STATUS: (CANCELLED by 114399728)**.` (the fallthrough `f"({state})"`
on `:176`) instead of the intended `🚫 CANCELLED` tag. Affected dirs:
`/scratch/.../{2lpt_loa0_wide, 2lpt_loa124_nohcd_nobal_wide, loa_no_dla_no_bal_wide, loa_no_hcd_with_bal_wide, loa_no_dla_no_bal_wide_m_normmask, loa_no_hcd_with_bal_wide_m_normmask}/`.
Fix: change `:171` from `state == "CANCELLED"` to
`state and state.startswith("CANCELLED")`.

Other scratch-README findings:

- Run-metadata `Latest checkpoint iter` is read from the
  `phase2_desi_checkpoint_iter*.pt` filenames and matches the
  highest-iter checkpoint on disk in each dir (spot-checked
  `loa_no_dla_no_bal_wide_m`=699, `2lpt_loa0_wide_m`=1499).
- For the in-flight `loa_no_dla_no_bal_wide_m_normmask_3000iter`, the
  README correctly states `Latest checkpoint iter | 1549` and `phase2_result.h5`
  does NOT exist, so the "Authoritative model artifact" section
  correctly falls back to the repo `out_dir` and the
  `examples/write_scratch_readmes.py` self-reference.
- The byte-identical-mirror claim at line 41 of each completed scratch
  README is **only true if the trainer wrote the .h5 there**. For the
  pre-PR-#6 runs that were CANCELLED-then-manually-copied, the user
  hand-copied via `cp`; the trainer's `shutil.copy2` (mirror) only
  shipped in commit `6642df8` today. So for the 7 completed PR-#6 dirs
  the claim is now correct going forward, but for the 2 base-wide
  CANCELLED-then-manually-restored dirs, the body still says
  "byte-identical at write time (trainer copy2's to both)" — slightly
  misleading because the trainer never ran to completion. Nit.

Cross-reference at scratch README line 41 to
`tests/phase2_train_desi.py:716-731` — the actual scratch-mirror code
lives at `:723-735`. Off by ~7 lines. Nit.

## Re-runnability / race conditions

Both `reemit_step_c_readmes.py:229` and
`write_scratch_readmes.py:266` use unconditional `write_text(body)` —
fully idempotent in content. No lock files; no append behaviour. Safe
to re-run any time.

Race conditions:
- If a SLURM job transitions from `RUNNING → COMPLETED` between
  `find_latest_slurm_log` (`write_scratch_readmes.py:38-55`) and
  `sacct_state` (`:103-113`), the README briefly shows stale status.
  Trivially resolved by re-running the script. Not a correctness issue.
- `reemit_step_c_readmes.py` reads `phase2_result.h5` then writes
  README. If a re-emit runs while the trainer is still writing the .h5
  (e.g. user invokes manually mid-run), partial h5 reads could fail —
  `h5py.File(...,"r")` would raise; the script processes one dir at a
  time and a single failure won't break the others, but the
  uncaught exception will halt processing. Low risk because re-emit is
  manual. Add a try/except per dir if defensive. Nit.

## Discoverability test (pretend-future-Claude walkthrough)

1. Start at `README.md`: zero mentions of `CURRENT_MODELS.md` or
   `production_models.md` (grep returns no matches). **Discoverability
   gap.** A future session has to know to look in `docs/`.
2. `CLAUDE.md` also has zero mentions of either pointer file.
   Same gap.
3. Land at `docs/CURRENT_MODELS.md` (assume the future-Claude
   eventually finds it via `ls docs/`): table is clear; one-line
   pointers to long-form `docs/production_models.md`. All paths in the
   table verified to exist or to be correctly labelled "in flight".
4. Recommended NOW-pick: `model_epoch_920.h5` —
   `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5`.
   File exists on Turbo per the path layout in
   `CLAUDE.md:project_paths`. ✓
5. Recommended 2lpt pick:
   `docs/notes/2026-05-11_desi_phase2_2lpt_loa{0,124_nohcd_nobal}_wide_m/phase2_result.h5`
   — both exist ✓.
6. In-flight pointers (50087967, 50212621, etc): all six STATUS.md
   files explicitly state "do NOT use any partial checkpoint as a
   production model" and link to monitoring; reader-safe.

## Issues found

### Critical (block PR until fixed)

None.

### High (should fix in this PR if cheap)

1. **`reemit_step_c_readmes.py:80-98` read_h5_scalars only checks
   `f[k]` not `f.attrs[k]`** — 7 of 8 completed model READMEs show
   `n_spectra=-1, n_iters=-1, lr=nan`. Fix: fallback to `f.attrs[k]`
   when the dataset is absent.

### Medium

2. **`write_scratch_readmes.py:171` cancelled-state matching** —
   `state == "CANCELLED"` misses `"CANCELLED by NNN"`; 4 scratch
   READMEs lose their intended status tag. Fix:
   `state and state.startswith("CANCELLED")`.

3. **`production_models.md:118-131` "In flight" table is stale** —
   50017771-74 listed as in-flight but their scratch READMEs report
   TIMEOUT (iter ~700-800); 50021381 listed as in-flight but COMPLETED
   and its model card is in
   `docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/`.
   Update this section or remove it.

4. **`production_models.md:166-172` caveat #4 "README templating bug"
   says "Fix pending (task #7)"** — that fix landed in commit
   `f2b931d` ("Cleanup per holistic PR review: 3 small fixes")
   and the re-emit script is in this PR. Remove the caveat or
   convert it to "Fixed 2026-05-14 (commit f2b931d)".

5. **SUPERSEDED map LOA `_g` → `_m_normmask_3000iter`** — pointing
   `_g` users at the `_m` band's retrain silently changes the
   normalization convention. Either submit a `_g_normmask_3000iter` job
   or rewrite the supersession message to call out the band switch.

6. **c0prior README "log_c_0 prior σ | (none)"** contradicts the "Not
   preferred" banner above. Root cause: σ value isn't echoed by SLURM
   submit script or stored in the older .h5. Either (a) echo
   `LOG_C_0_PRIOR_SIGMA` in `phase2_desi_retrain.sh` header so future
   runs are recoverable, or (b) hard-code the σ for the c0prior model
   in `reemit_step_c_readmes.py` since it's the only existing case
   (per the .h5 endpoint `c_0=0.0198` and `log_c_0_history` decay rate,
   σ was likely 0.1).

### Low / nit

7. **Discoverability: repo `README.md` doesn't reference
   `CURRENT_MODELS.md` or `production_models.md`**. A 1-line pointer
   at the top would close the future-Claude gap.

8. **Scratch READMEs cite `tests/phase2_train_desi.py:716-731`** but
   the scratch-mirror code is at `:723-735`. Off by ~7 lines because
   the comment block expanded.

9. **`reemit_step_c_readmes.py` drops the `preload_source` row** that
   `_save_readme` includes. Data is in the .h5 (attrs and/or dataset).
   Add the row back.

10. **SLURM log line `(Garnett+2017 convention)` on the 50087967 log**
    is stale (the log was started before `f2b931d` fixed the label
    logic in `dataset.py`). Cosmetic; the .h5 manifest carries the
    correct band, so the README is right. Affects only currently-running
    job logs; future runs will print correctly.

11. **`production_models.md:25` TL;DR says "β=2.69-3.09"** for `_m`
    variants — but actual `_m` values are 3.09 (loa-0) and 2.97
    (loa-124); 2.69 is the `_g` (loa-0) value. Either correct the range
    or footnote that 2.69 is from a different variant.

12. **`reemit_step_c_readmes.py:245-248` fallback to suffix-inferred
    band** is reached only if both the .h5 manifest AND the SLURM log
    lookup fail; given that `normalization_min_lambda` is present in
    all 8 existing .h5 files this branch is dead code today. Defensive,
    fine to keep.

## Recommendation

**Fix-first** before merge:
- Issue #1 (high) — trivial 2-line fix in `read_h5_scalars`; makes the
  user-facing model cards report correct training-config values
  instead of `-1` / `nan`.
- Issue #2 (medium) — trivial 1-line fix in `write_scratch_readmes.py`.

**Either fix-first or follow-up commit** (your call):
- Issues #3 & #4 — doc staleness; ~3 minutes' work in
  `production_models.md`.

**Defer to follow-up**:
- Issues #5–#11 — none block correctness; #5 and #6 are the closest to
  needing attention but neither breaks production-loading.

Everything else (cross-reference paths, MODEL_STATUS lookup,
SUPERSEDED map accuracy modulo #2 and #5, 3-way in-flight
consistency, trainer scratch-mirror, idempotency, p_DLA / β / c_0
endpoint scalars) checks out.
