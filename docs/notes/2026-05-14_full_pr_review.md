# PR #6 holistic review — 2026-05-14

> Reviewer: independent audit (separate from the three 2026-05-13 reviews).
> Scope: 140 commits on `claude/debug-trainer-from-v1`, +30861 / -70 lines.
> Branch HEAD: `660ee34` (Cleanup: README templating fix + Saclay norm band).
> Base: `desi_y3` HEAD `6415fb7` (PR #5 merge).
> Cross-references already on disk and not repeated:
> `docs/notes/2026-05-13_code_review_dataset_math.md` (✓ SHIP, dataset math),
> `docs/notes/2026-05-13_code_review_pr_diff.md` (✓ SHIP, 5-commit diff),
> `docs/notes/2026-05-13_beta_drift_investigation/findings.md` (no bug),
> `docs/notes/2026-05-13_step_c_dla_recovery/findings.md` (2/3 pass).

## Top-line verdict: FIX-FIRST (small)

The science core ships. The two behaviourally-substantive changes (dataset
reorder + 1e-2 threshold) are tested, MATLAB-equivalent, and inference-safe
on the canonical TID for the two main `_m` models. Inference loader is
already wired to read the v2 .h5 schema (`null_gp.py:478-482`).

Before merge, **three small things should land** (each <10 lines):

1. **`gpy_dla_detection/training/dataset.py:362-364`** — log message
   still prints "(Garnett+2017 convention)" unconditionally; was correct
   pre-reorder when the band was hardcoded, is now misleading for the
   `[1425, 1475]` MATLAB band. Either drop the convention label or
   replicate the templating logic from
   `tests/phase2_train_desi.py:362`.
2. **Model-card discoverability** — the c0prior README in
   `docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/README.md`
   has **no in-line warning** that this model collapses DLA detection.
   Only `docs/production_models.md:68` carries the "DO NOT USE FOR
   PRODUCTION" flag. A user navigating to the dir does not see it.
3. **Uncommitted state** — `git status` is dirty (see §"Merge readiness"
   below). Either commit or revert before tagging the PR ready.

None of the above is a science blocker. The base ships as-is if these are
deferred to a follow-up, but they are cheap and ship-blocking from a
documentation-hygiene standpoint.

## Cross-commit issues

I traced the dataset reorder (`aa36205`) against all earlier commits that
might assume the old order:

- **Training scripts** (`tests/phase2_train_desi.py`, `tests/phase2_train_dr16.py`):
  both call `load_preprocessed_h5(…, apply_normalize=True, apply_mask=True,
  …)`. They do not assert the internal order; the change is opaque to the
  caller. Safe.
- **LoaArchive adapter** (`preload_spectra/preload_from_loa_archive.py`):
  emits a `trainset.h5` that is *not* yet normalized/masked at write time
  (those steps happen at training load). Order change does not affect it.
- **Test fixtures** (`tests/test_normalize_by_rest_median.py:108-167`,
  `:261`): no assertions on order. The new regression test
  `test_normalize_rejection_threshold_is_1e_minus_2` (line 126) directly
  tests the new threshold semantics. Safe.
- **Stale comment** at `tests/test_preload_from_loa_archive.py:264-265`:
  "mask high-noise, normalize, de-forest, center" — reflects the old
  order. The test sets `apply_normalize=False`, so behaviour is
  unaffected; comment is cosmetic but should be flipped to
  "normalize, mask, de-forest, center". (low)
- **Stale comment** at `gpy_dla_detection/training/dataset.py:362-364`:
  the print after `_normalize_by_rest_median` says
  `(Garnett+2017 convention)` regardless of the runtime band. This is
  the inverse of the README templating bug fixed by `660ee34`
  (`tests/phase2_train_desi.py:362` templates it correctly via the band
  value). One trainer surface now reports truth, the other does not.
  (medium — log hygiene; misleading at scale when a user greps the SLURM
  log to confirm what was trained.)

The earlier reviews cited `findings.md:53` as a stale `dataset.py:169`
line reference. The number 169 is wrong post-reorder; current line is
`:177`. Doc-only.

No silent functional break across commits. The only contradiction I find
is the print-message bug above.

## PR description fidelity

The PR description draft (`docs/notes/2026-05-13_pr6_description.md`) is
accurate in structure. Three minor inaccuracies:

- Line 78 claims **225 tests pass**; actual collected count on this
  machine, after excluding 8 `desispec`/`desiutil`/`fitsio`-blocked
  modules, is **224**. Off-by-one (probably PR-diff agent counted before
  `67700d8` was pushed). Update before merge.
- Line 80 references `test_v3_objective_vectorized_parity.py` — confirmed
  exists, runs green (3 passed in §"Merge readiness").
- Line 87 reports `c0prior p_DLA = 0.04, NaN, do not use for production`
  — matches `findings.md:21`. Accurate.

Claimed file paths I spot-checked: all exist on disk
(`docs/notes/2026-05-11_desi_phase2_2lpt_loa*/phase2_result.h5` confirmed
by `ls`; tests in `tests/test_*.py` collected by pytest).

Claimed performance numbers:
- "28× speedup at 5k smoke, 3× at full 89k DR16 retrain" (line 21): not
  re-verified here, trusted from `docs/notes/2026-05-09_*` (commit
  `1c45986`).
- "max rel_err = 5.3e-11" v1≡MATLAB (line 81): matches
  `tests/test_v1_matches_matlab.py` semantics (test runs and passes).

The body otherwise matches what's in the diff.

## Test gaps and production-facing surfaces

### Test coverage

Of the load-bearing PR-#6 new code:

| Module | Tested? | Notes |
|---|---|---|
| `gpy_dla_detection/training_v3/objective_vectorized.py::spectrum_loss_batch` | ✓ | `test_v3_objective_vectorized_jacobian.py`, `test_v3_objective_vectorized_parity.py`, `test_v3_train_step_parity.py` — Jacobian, per-spec parity, and Adam-step parity |
| `gpy_dla_detection/training/dataset.py` reorder | ✓ | `test_normalize_by_rest_median.py` (20 tests, including the new `test_normalize_rejection_threshold_is_1e_minus_2`) |
| `preload_spectra/preload_from_loa_archive.py` | ✓ | `tests/test_preload_from_loa_archive.py` (9 tests; all green) |
| `gpy_dla_detection/training/trainer_v2.py` jitter fix | ✓ | `test_objective_v2_jitter.py` |
| `tests/phase2_train_desi.py` | ✗ | Production script, no unit test. Smoke-via-SLURM only (job 49913952 referenced in script header). |
| `tests/phase2_train_dr16.py` | ✗ | Same — production script, no unit test. |
| `examples/dla_recovery_step_c.py` (NEW today) | ✗ | One-shot operational diagnostic, no test |
| `examples/reemit_step_c_readmes.py` (NEW today) | ✗ | One-shot script, no test |
| `examples/plot_lyman_rungs_overlay.py` (REWRITTEN today, +57 lines vs `64e9f49`) | ✗ | Plot-only; no test |
| `tests/phase2_train_*.py::_save_readme` (CHANGED today, signature) | ✗ | No unit test asserts the README band reflects `--norm-min-lambda` |

**Gap that matters**: there is no test asserting that the README emitted
by `_save_readme` contains the actual `--norm-min-lambda` value. The
templating bug fix at `tests/phase2_train_desi.py:362` could regress and
not be caught — only by manually inspecting the next training run's
README. **Suggested**: a 15-line unit test that monkeypatches
`_save_readme(out_dir, …, norm_min_lambda=1425.0, norm_max_lambda=1475.0)`
and asserts `"[1425.00, 1475.00] Å rest"` appears in the written README.
Low priority but cheap.

The two one-shot example scripts (`dla_recovery_step_c.py`,
`reemit_step_c_readmes.py`) don't really need tests — they're
analysis-time scripts run once and committed for reproducibility, not
production code paths. Fine as-is.

### Production-facing surfaces

| Surface | Touched by PR? | Risk |
|---|---|---|
| `gpy_dla_detection/null_gp.py` (inference) | YES — `NullGPMAT.__init__` calls `apply_normalization_from_h5(params, learned)` at `:481-482`, mutating `params` from v2 schema | Low; tested by `test_null_gp_mat_picks_up_norm_from_v2_h5` |
| `run_bayes_select.py::DLAHolder` (inference) | YES — 5 new τ-EB kwargs, all default-OFF | Low; default behaviour preserved. Pre-existing tests cover the OFF path. |
| `desi-DLAGP.py` (CLI) | YES — 5 new CLI args, all default-OFF | Low; same |
| `dlasearch.py` (pipeline) | YES — passes new τ-EB kwargs via `model_params.get(…, default)` | Low; default behaviour preserved |
| .h5 schema (model file) | YES — manifest fields added in `3a0b84f` (`num_forest_lines`, `k`, `lr`, prior σs, etc.) | Low; `NullGPMAT.__init__` reads exactly 9 keys (`:462-476`); unknown keys are ignored. Other tools that read the .h5 are not in this repo. |
| SLURM scripts | YES — new files only; `MAX_WALLTIME_SEC` env override is purely additive | Low |
| Default behaviour of existing SLURM scripts | NO regression — they are new files, not modifications to existing scripts | Safe |

Production-inference *behaviour* on a real run is unchanged unless a
user explicitly passes `--enable_tau_eb 1` or loads a v2-schema .h5
(which they already can do via PR #5; the only PR-#6 addition is the
extra metadata keys, which are ignored by `NullGPMAT`).

## Caveat discoverability

**Two known caveats** documented in the PR:

### Pre-reorder corr-noise on the 6 Step C 2lpt models

- ✓ Flagged in `docs/production_models.md:78-84` (text "Pre-reorder
  caveat", explicit "applies to all *future* retrains; the 6 trained
  models above were trained before the fix and inherit the roughness").
- ✓ Flagged in `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`.
- ✗ **NOT** flagged in any of the 8 model-card READMEs in
  `docs/notes/2026-05-11_desi_phase2_*/README.md`. Each README has only
  a "corr-noise debug arc: see findings.md" provenance line at line 44 —
  no in-line "this model is pre-reorder, has 7× corr-roughness". A user
  navigating to the dir does not see the caveat.

### c0prior collapses DLA detection

- ✓ Flagged in `docs/production_models.md:68` ("DO NOT USE FOR PRODUCTION").
- ✓ Flagged in `docs/notes/2026-05-13_step_c_dla_recovery/findings.md:47`
  (p_DLA = 0.04, MAP log NHI = NaN).
- ✗ **NOT** flagged in
  `docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/README.md`.
  This is the most-likely-to-be-confused model in the entire PR, and its
  own model card is silent.

**Recommendation**: 3-line markdown banner at the top of each
pre-reorder Step C README, e.g.:

```
> ⚠ **Pre-reorder caveat**: trained before the 2026-05-13 dataset.py
> reorder; corr(M·M^T) is ~7× rougher than v1 production
> (see `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`).
```

And for c0prior specifically, add:

```
> 🚫 **Not production-ready**: DLA detection collapses on canonical TID
> (p_DLA = 0.04, MAP = NaN). See
> `docs/notes/2026-05-13_step_c_dla_recovery/findings.md:21`.
```

`examples/reemit_step_c_readmes.py` is already in the PR — extending it
to inject these banners is a 10-line addition.

## Style and hygiene at PR scale

### Duplicated code

- `gpy_dla_detection/training_v3/` is a **deliberate verbatim copy** of
  v1 (per `gpy_dla_detection/training_v3/README.md:8-20`). Intentional.
  Documented. Not a style issue.
- `desi_learn_qsos_model.py` (repo root) and
  `gpy_dla_detection/training_v3/desi_learn_qsos_model.py` are
  byte-equivalent except for the absolute-import fix (`from .voigt` →
  `from gpy_dla_detection.voigt`, doc'd in commit `ba55c01`). Pre-existing.

### Dead files / dead code

- `examples/plot_all_trained_kernels.py:152-153` — "Hide unused axes"
  loop is dead (12 entries fill 12 panels). Flagged in earlier review.
  Nit.
- `examples/plot_pca_init_corr_multi.py:24` (`import h5py`) and
  `examples/plot_all_trained_kernels.py:13` (`import sys`) — unused.
  Nit.

### Misleading commit messages

I scanned all 140 commits — none are misleading. A few are terse but
none contradict their diff. Note `ec4c26e` says "Step C status: 2lpt
jobs DONE, LOA jobs resubmitted at chunk=7500" which was accurate at the
time but is now stale (LOA jobs since landed). No issue — provenance.

### TODOs / FIXMEs introduced

3 TODOs in the diff:
- `desi_learn_qsos_model.py:180` "TODO : understand if I can use these
  as init points" — pre-existing v1 code, just copied into
  `training_v3/`. Not introduced.
- `preload_spectra/preload_from_loa_archive.py` (docstring)
  "TODO; for now this profile exercises" — placeholder doc, low impact.
- `tests/profile/profile_voigt.py` similar.

None block the merge.

### Docstring drift

- `run_bayes_select.py:354` — docstring says default `(0.5, 1.0, 1.5,
  2.0, 3.0, 4.0)` but the actual default at `:334` is `(0.5, 1.0, 1.5,
  2.0, 3.0, 4.0, 5.0, 6.0)`. Off by 2 elements. (low)

### Naming idiosyncrasy

`tests/phase2_train_desi.py` and `tests/phase2_train_dr16.py` are
**scripts**, not tests. Pytest correctly skips them (no `test_` prefix
functions). Documented intent per `tests/phase2_train_dr16.py` header.
Idiosyncratic but established convention. Not a blocker.

## Merge readiness checklist

### Tests

Pytest collection: **224 tests** across 41 test files, after excluding
8 modules blocked by missing `desispec` / `desiutil` / `fitsio` / `healpy`
(all environmental, pre-existing). Subset run on this machine:

- `test_normalize_by_rest_median.py`: 20 passed, 0 failed (48.7s)
- `test_v3_objective_vectorized_parity.py` + 3 sibling v3 modules: all
  passed (8 tests in 6s)
- `test_review_fixes.py` + `test_tau_eb_wiring.py` + `test_lyb_veto.py`
  + 5 others: 102 passed, 0 failed (5.9s)
- `test_v1_matches_matlab.py` + 3 v3-matlab tests: 21 passed, 1 skipped (2.9s)
- `test_voigt_*.py` set: 23 passed, 1 failed (the failure is
  `test_voigt_sweep_targets.py::test_pick_for_mock_returns_all_three_regimes`
  due to missing `healpy`, environmental, pre-existing — see
  `tests/test_voigt_sweep_targets.py:72`)
- `test_preload_from_loa_archive.py`: 9 passed (2.4s)

Pre-existing failures (not introduced by PR #6):
- `tests/test_loss_history.py::test_gp_training_convergence` —
  `learn_qso_model.py:433`: `centered_rest_fluxes.copy()` on a `Tensor`.
  Pre-existing bug from 2025-02-19 commit `06aba1c`.
- `tests/test_smoke_target_contamination.py` — depends on `fitsio`,
  skip-by-path-existence guard doesn't fire because path exists on this
  machine but module is missing. Pre-existing environmental.

**Net**: all PR-relevant tests pass. The 1 collection error and 3
runtime failures all pre-date this PR. Aligns with the PR description's
"all tests pass" modulo the "225 vs 224" off-by-one.

### Uncommitted state

`git status` shows:
- Modified: `docs/notes/2026-04-29_voigt_lsf_sweep/per_target_scatter.png`,
  `docs/notes/2026-04-29_voigt_lsf_sweep/report.md` (legitimate analysis
  re-run — different `voigt_sweep_fixed_kernel` source)
- Modified: `examples/render_story_figures.sh` (legitimate bug fix —
  `dirname` chain depth was off by one for the spectra-16 path)
- Untracked: `.claude/scheduled_tasks.lock` (harness lock file —
  shouldn't be committed; `.gitignore` it)
- Untracked: `docs/notes/2026-05-03_multidla_velocity_separation_prior_design.md`
  and `docs/notes/2026-05-04_joint_inference_design.md` (design docs;
  per CLAUDE.md they belong in `docs/notes/` — should be added or
  explicitly deferred)

**Action**: commit the 3 modified files (small, legitimate fixes),
decide whether to ship the 2 design docs in this PR or defer, and ensure
the lock file is gitignored.

### Merge conflicts with desi_y3

Merge base = `desi_y3` HEAD = `6415fb7`. The PR branch is a clean
fast-forward (no parallel changes upstream).

### PR description up to date

Mostly. Three minor inaccuracies (test count, docstring drift on
factors, c0prior README banner absent) — all noted above.

## Issues found

**Critical**: none.

**High**: none.

**Medium**:

1. **`gpy_dla_detection/training/dataset.py:362-364`** — print message
   unconditionally claims `(Garnett+2017 convention)` after the
   normalize step, regardless of `norm_min_lambda`. With the new
   `[1425, 1475]` MATLAB band, the SLURM logs of `_m` runs show the
   wrong convention label. Fix: drop the label or template it like
   `tests/phase2_train_desi.py:362` does post-`660ee34`.

2. **Step C model-card discoverability** — the c0prior model
   (`docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/README.md`)
   has **no in-line warning** that DLA detection collapses; the only
   "DO NOT USE FOR PRODUCTION" flag is in `docs/production_models.md:68`.
   A user navigating to the dir directly does not see it. Same for the
   "pre-reorder corr-roughness" caveat across all 8 Step C READMEs.
   Fix: extend `examples/reemit_step_c_readmes.py` to inject the
   banner; re-run once.

**Low**:

3. **Test count off-by-one** in PR description
   (`docs/notes/2026-05-13_pr6_description.md:78`) — claim "225 tests"
   vs actual "224 tests collected". Update before merge.

4. **Docstring drift** at `run_bayes_select.py:354` — claims default
   `tau_eb_factors = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)` but actual default
   has 8 elements (incl. 5.0 and 6.0). Both the kwargs default at line
   334 and the CLI default at `desi-DLAGP.py:213` use the 8-element
   tuple. Docstring is the only one wrong.

5. **Stale comment** in `tests/test_preload_from_loa_archive.py:264-265`
   reflecting old "mask, normalize, …" order. Cosmetic; test behaviour
   unaffected.

6. **Stale comment** in `slurm/greatlakes/phase2_desi_retrain.sh:21-23`
   "Walltime budget 8h" vs `#SBATCH -t 12:00:00`. Pre-existing from
   commit `1dcc13b`. Not introduced by today's commits.

7. **Stale doc line reference** in
   `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md:53` —
   cites `dataset.py:169`, current line is `:177` post-reorder. Doc-only.

8. **Uncommitted state** in working tree (see "Merge readiness" §) —
   3 legitimate-looking modifications + 1 lock file + 2 untracked
   design docs. Commit or revert before tagging the PR done.

**Nit**:

9. `examples/plot_pca_init_corr_multi.py:24` — unused `import h5py`.
10. `examples/plot_all_trained_kernels.py:13` — unused `import sys`.
11. `examples/plot_all_trained_kernels.py:152-153` — dead "hide unused
    axes" loop (12 entries fill 12 panels).

## Recommendations

**Before merge** (cheap, totaling <30 lines of new code):

1. Fix the dataset.py print message stale label (Medium #1) — 5 lines.
2. Re-run `examples/reemit_step_c_readmes.py` after adding a 5-line
   banner-injection helper for pre-reorder caveat + c0prior warning
   (Medium #2) — 20 lines total in the script.
3. Update the test-count claim in the PR description (Low #3) —
   1 character (5→4).
4. Fix the `tau_eb_factors` docstring (Low #4) — add `5.0, 6.0`,
   <10 chars.
5. Commit or revert the 3 modified files and the 2 design docs
   (#8) — user decision.

**After merge** (defer cleanly):

- Probe `med ∈ [1e-2, 1e-1)` band (from the dataset-math review's
  follow-up — line 92 in that doc). Open question whether the next
  band up the threshold ladder is also contaminating.
- Investigate c0prior collapse mechanism (PR description §"What's
  deliberately NOT in this PR" line 72; tracked).
- Post-reorder LOA retrains 50087967/68 (in flight; will supersede
  the 6 pre-reorder Step C 2lpt models per `production_models.md:113-125`).
- Pre-existing `learn_qso_model.py:433` `.copy()` on tensor bug
  (unrelated to PR scope; fix in follow-up).

**Net**: SHIP after the 5 cheap fixes above. The science is sound, the
math is tested, the inference path is safe, the caveats are real but
already mapped — they just need to be visible at the model-card level
where users will land first.
