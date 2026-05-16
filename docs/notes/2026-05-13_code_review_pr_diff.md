# PR #6 diff review — 2026-05-13 commits

Scope: `badf0c2..HEAD` (5 commits) on `claude/debug-trainer-from-v1`.

## Commit-by-commit summary

| SHA | Title | Lines | Risk |
|---|---|---:|---|
| `aa36205` | Corr-noise fix: reorder normalize→mask + threshold 1e-3→1e-2 | `dataset.py` +35/-22, `test_normalize_by_rest_median.py` +44/-0 | Medium — changes preprocessing semantics for all future trainings |
| `3a0b84f` | Phase 2 trainers — full training-hyper manifest in .h5 | `phase2_train_desi.py` +90/-30, `phase2_train_dr16.py` +43/-7 | Low — additive schema; loaders are forward-compatible |
| `9f321f6` | `phase2_desi_retrain.sh` — `MAX_WALLTIME_SEC` env override | +6/-1 | Low — pure parameterization |
| `834fb78` | Step C 2lpt model cards + post-reorder smoke README | docs only | None |
| `64e9f49` | Analysis scripts + plots + docs from corr-noise session | +2200 lines docs/examples | None on production code paths |

## Schema change side effects (commit `3a0b84f`)

The DESI inference loader `gpy_dla_detection/null_gp.py:453-482` reads exactly
nine top-level keys (`M`, `mu`, `log_omega`, `log_c_0`, `log_tau_0`,
`log_beta`, `rest_wavelengths`) plus the optional normalization scalars via
`gpy_dla_detection/_h5_helpers.py:39`. Extra keys are silently ignored — the
HDF5 file pointer is only queried for the names listed. DESI-detection
hinge is `log_tau_0.ndim == 0` (`null_gp.py:455`), and `log_tau_0` retains
its scalar 0-d datatype, so detection is unaffected.

The fake-model fixture in `tests/test_normalize_by_rest_median.py:377-396`
writes only the legacy minimal schema, so
`test_null_gp_mat_picks_up_norm_from_v2_h5` and siblings still pass against
both pre- and post-manifest .h5 files. Verified: all 20 tests pass
(`tests/test_normalize_by_rest_median.py`, 48.95s).

No code path in `gpy_dla_detection/` or `tests/` reads
`num_forest_lines`, `k`, `n_spectra`, `n_iters`, `lr`, `tau_0_prior_*`,
`pca_random_state`, `optimizer`, `training_release`, etc. The manifest
is pure provenance.

Hard-coded "expected key list" audit — `null_gp.NullGPMAT.__init__`,
`_h5_helpers.apply_normalization_from_h5`, and the `_make_fake_v2_model_h5`
fixture are the only readers; none enumerate keys exhaustively.

## Test regression risk

Collection on GreatLakes: 225 tests collect cleanly (after excluding 6
modules that import `desispec`/`desiutil`, which has always been missing
on GL — pre-existing).

I searched for tests that depend on `mask-then-normalize` order or
`|med| < 1e-3` threshold:
- `tests/test_normalize_by_rest_median.py:108-123` — `test_normalize_handles_bad_spectra` tests NaN/zero rows. The new `1e-2` threshold doesn't change behavior for these cases. PASS.
- `tests/test_normalize_by_rest_median.py:60` — `test_load_preprocessed_h5_normalize_path_smoke` invokes the full pipeline; no order assertion. PASS.
- `tests/test_preload_from_loa_archive.py:269` — uses `apply_normalize=False`, no order dependence. The docstring at `:264-265` still reads "normalize, de-forest, center" in the *old* order, which is now stale — comment-only.
- `tests/validate_against_matlab_dr16.py:161` — loads MATLAB preload directly, bypasses `dataset.py`. Unaffected.

Ran 118 tests (`test_normalize_by_rest_median.py` + `test_voigt_v2_parity.py`
+ `test_voigt_batched.py` + `test_cddf_mock.py` + `test_cddf_calibration.py`
+ `test_generate_samples.py`) — all 118 pass in 114s.

## Inter-trainer inconsistencies

Manifest field comparison between `phase2_train_desi.py:529-562` and
`phase2_train_dr16.py:530-548`:

| Field | DESI | DR16 | Reason for divergence |
|---|---|---|---|
| `num_forest_lines` | 31 | 31 | Match — both `NUM_FOREST_LINES=31` |
| `k` | `args.k` (default `K_DESI=30`) | `K=20` (module constant) | DESI Y3 vs SDSS DR16 convention; intentional |
| `n_spectra` | `ts.n_spectra` | `args.n_spectra` | Both correct |
| `lr`, `n_iters`, `chunk_size` | from args | from args | Match |
| `de_forest_tau_0`/`beta` | `TAU_0_PRIOR_MU`/`BETA_PRIOR_MU` (Turner+2024) | `INITIAL_TAU_0`/`INITIAL_BETA` | Different constants but semantically equivalent ("seed used to deforest at preprocessing") |
| `tau_0_prior_mu/sigma`, `beta_prior_mu/sigma` | yes | yes | Match |
| `log_c_0_prior_mu/sigma` | yes | **absent** | DR16 trainer has no log_c_0 prior at all (PR #5 was DESI-only). Appropriate. |
| `z_min`, `z_max`, `min_snr` | yes | **absent** | DR16 trainer doesn't z/SNR filter (uses preload `train_idx`). Appropriate. |
| `pca_random_state` | 0 | 0 | Match |
| `vectorized` | int | `int(bool(...))` | Cosmetic inconsistency (DR16 normalizes truthiness). Both serializable as int64. Low impact. |
| `normalize_then_mask_order` | 1 | 1 | Match |
| `optimizer` | `b"Adam"` | `b"Adam"` | Match |
| `training_release` | `b"PR6_StepC"` | `b"PR6_StepA_DR16"` | Distinguishable provenance — good |
| `git_commit_sha` | yes | yes | Match — both via `subprocess.check_output(["git", "rev-parse", "HEAD"])` |
| `training_timestamp` | yes | yes | Match — both ISO-8601 UTC |
| `preload_source` | yes | **absent** | DR16 trainer doesn't track a preload — uses MATLAB-style cache_dir. Appropriate. |

Note: `phase2_train_dr16.py:520-521` writes
`normalization_min_lambda = 1425.0` / `normalization_max_lambda = 1475.0`
(MATLAB DR16 band). `phase2_train_desi.py:502-503` writes whatever the
runtime CLI passed (`_RUNTIME.get("norm_min_lambda", args.norm_min_lambda)`).
This divergence is correct — DR16 trainer always uses the MATLAB-faithful
band, DESI trainer is CLI-parameterized.

Constants `K_DESI=30` (`phase2_train_desi.py:70`) and `K=20`
(`phase2_train_dr16.py:69`) — both match the corresponding conventions
documented in `docs/architecture.md` and CLAUDE.md §6. `NUM_FOREST_LINES=31`
is consistent (both at line 66). Verified via grep on both files.

## SLURM script (commit `9f321f6`)

The `MAX_WALLTIME_SEC="${MAX_WALLTIME_SEC:-41000}"` default at
`slurm/greatlakes/phase2_desi_retrain.sh:75` is 11h23m = 41000s. The
`#SBATCH -t 12:00:00` (line 8) = 43200s. Buffer = 36 min. Trainer
`tests/phase2_train_desi.py:287-296` triggers graceful-exit save on
elapsed > `max_walltime_sec`. Safe at default.

**Caveat for users**: if a user submits `sbatch --time=24:00:00 …` without
also exporting `MAX_WALLTIME_SEC=79200`, the trainer exits at 11h23m
even though SLURM would allow 24h. Output is still saved cleanly, so
this is wasted compute, not data loss. The header comment at
`slurm/greatlakes/phase2_desi_retrain.sh:73-74` documents the override
path. **Minor**: the unchanged block-comment at lines 21-23 still claims
"8h budget" but `#SBATCH -t 12:00:00` — pre-existing stale (commit
`1dcc13b`), not introduced by this PR. Not blocking.

The `phase2_dr16_retrain.sh` script (`slurm/greatlakes/phase2_dr16_retrain.sh:48`)
already had `MAX_WALLTIME_SEC="${MAX_WALLTIME_SEC:-82800}"`. With this PR,
both DESI and DR16 retrain scripts expose the same env knob — convergent.

## Scripts under examples/

All six new scripts compile via `python -m py_compile`. Two minor unused
imports:

- `examples/plot_pca_init_corr_multi.py:24` — `import h5py` is never used; the script reaches data via `load_preprocessed_h5` and `_pca_init`. Nit.
- `examples/plot_all_trained_kernels.py:13` — `import sys` is never used; `Path(__file__).resolve().parent.parent` does the work directly. Nit.

`examples/plot_all_trained_kernels.py:152-153` "Hide unused axes" loop is
dead code: `ENTRIES` has 12 items, the grid is `rows=3, cols=4 = 12`,
so `axes_flat[len(ENTRIES):]` is empty. Nit; harmless.

`examples/probe_outlier_tail_corr.py` properly uses `warnings.catch_warnings`
(line 127). `examples/compare_inference_norm_band.py` correctly builds
`Parameters(num_lines=3, num_forest_lines=preset.num_forest_lines, …)`
matching `gpy_dla_detection/set_parameters.py:61`. `examples/plot_kernels_v1_rest_range.py`
uses `torch.load` at line 71. `examples/plot_lyman_rungs_overlay.py`
is large (490 lines) but I sampled the imports and structure; nothing
appears dead or contradictory.

No typos in the user-facing strings I read.

## Documentation accuracy

`docs/production_models.md` — verified paths:
- `model_epoch_920.h5` at `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/` (not checked from filesystem; user-managed scratch). The repo-local README references to `docs/notes/2026-05-11_desi_phase2_2lpt_loa{0,124}_wide{,_g,_m}/phase2_result.h5` all exist on disk (verified by `ls`).
- The caveats section §4 correctly flags the README templating bug (`phase2_train_desi.py:358` hard-codes `[1310, 1325]`).
- Cross-reference to `null_gp.py:440-503` at `docs/production_models.md:208` is accurate (`NullGPMAT.__init__` does live at lines 440-497).

`docs/notes/2026-05-13_pr6_description.md`:
- Claim "All 65+ pre-existing tests still pass" (line 71) — 225 tests collect on GL after excluding `desispec`-dependent modules. The "65+" number is dated (was true post-PR-#5); current count is much higher. **Minor doc lag**.
- Test plan checkbox "All 65+ tests pass" (line 173) — same.

`docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md:96` — references
`dataset.py:177` for the `1e-2` threshold. Current line is 177 (verified).
Findings doc line 53 references `dataset.py:169`; current line is 177
(off by 8 — minor stale-line-number drift after the reorder, doc is still
semantically correct since it points to "the rejection rule").

`docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md`
correctly flags the prior ✓ on the mask line as a 2026-05-13 correction
(line 24, "🟡 Correction landed 2026-05-13"). Good.

Model card READMEs in `docs/notes/2026-05-11_desi_phase2_*/` all have
the templating bug already disclosed (`phase2_train_desi.py:358`).

## Issues found

**Critical**: none.

**High**: none.

**Medium**:
1. **README templating bug** (`tests/phase2_train_desi.py:358`) — hard-codes `normalize | [1310, 1325]` in the auto-emitted model card regardless of `--norm-min-lambda`/`--norm-max-lambda`. The `_m` variants trained on `[1425, 1475]` say `[1310, 1325]` in their README. Disclosed in PR description task #7, `docs/production_models.md` §4. Should be fixed before downstream users start relying on the README as ground truth. Fix is one-line: use `{_RUNTIME.get("norm_min_lambda", args.norm_min_lambda)}` like `_save_h5` does.

**Low**:
2. **DR16 `vectorized` cast** (`tests/phase2_train_dr16.py:541`) — saves `np.int64(bool(args.vectorized))` whereas DESI saves `np.int64(vectorized)`. Cosmetic; both round-trip fine as int.
3. **SLURM header comment lag** (`slurm/greatlakes/phase2_desi_retrain.sh:21-23`) — claims "8h budget" but `-t 12:00:00`. Pre-existing (commit `1dcc13b`), not introduced here.
4. **Test count "65+"** (`docs/notes/2026-05-13_pr6_description.md:71,173`) — actual is 225 collected. Update before the PR body is final.
5. **Stale doc comment** (`tests/test_preload_from_loa_archive.py:264-265`) — says "mask high-noise, normalize, de-forest, center" reflecting the old order. The test itself uses `apply_normalize=False` so it's not affected functionally, but the comment should read "normalize, mask, de-forest, center" to match the post-reorder dataset.py.

**Nit**:
6. `examples/plot_pca_init_corr_multi.py:24` — unused `import h5py`.
7. `examples/plot_all_trained_kernels.py:13` — unused `import sys`.
8. `examples/plot_all_trained_kernels.py:152-153` — dead "hide unused axes" loop (12 entries fill 12 panels).
9. `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md:53` cites `dataset.py:169` — current line 177 after the reorder. Off by 8.

## Net verdict: SHIP

The two behaviorally-substantive changes — preprocessing reorder
(`dataset.py:333-371`) and `1e-3 → 1e-2` rejection threshold
(`dataset.py:177`) — are well-tested: 20/20 in `test_normalize_by_rest_median.py`
pass including the new regression guard
`test_normalize_rejection_threshold_is_1e_minus_2`. The reorder is also
backed by the MATLAB audit (`docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md`
post-correction) and the falsification probe `examples/probe_outlier_tail_corr.py`.

The h5 manifest addition is additive — no inference-path code reads the
new fields; `null_gp.NullGPMAT` ignores unknown keys; the fake fixture
in tests still validates. Inter-trainer divergences are all intentional
and traceable to design (DESI has `log_c_0` prior, DR16 doesn't; DESI
z-filters at training, DR16 doesn't; DR16 hard-codes MATLAB band, DESI
is CLI-parameterized).

The SLURM `MAX_WALLTIME_SEC` env override is purely additive and matches
the DR16 retrain script's existing pattern.

All scripts compile; minor nits are cosmetic. The medium-priority item
(README templating bug) is **already disclosed** in `docs/production_models.md`
§4 and PR description task #7; consumers are warned. If a fix is desired
before merge, it's one line in `tests/phase2_train_desi.py:358`.

The dead-comment / stale-line-number / unused-import nits can ship as-is
or be cleaned up in a follow-up. None of them affect runtime behavior.
