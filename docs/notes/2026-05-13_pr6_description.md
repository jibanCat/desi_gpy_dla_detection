# PR #6 description draft — 2026-05-13

Paste into `gh pr edit 6 --body "$(cat docs/notes/2026-05-13_pr6_description.md)"`
or the GitHub web UI when ready. Trim sections that aren't ready to ship.

---

## Title

```
Rebuild GP trainer from v1 reference; vectorize spectrum_loss; DESI port; corr-noise fix
```

## Summary

Rebuilds the Phase 2 GP continuum trainer in three parts:

1. **Step A — verify against MATLAB DR16**: the vectorized loss
   (`spectrum_loss_batch`) matches MATLAB `spectrum_loss.m` to
   `max rel_err ≈ 5e-11` per pixel on five frozen 2lpt fixtures; the
   v1 Python `zqso_1pz` bug is correctly bypassed in both new trainers.
2. **Step B — vectorize the loss**: replaces the per-spectrum Python
   loop with a chunked vectorized path (`training_v3/objective_vectorized.py`),
   28× faster at 5k smoke and 3.05× at full 89k DR16 scale, with
   parity tests asserting `< 1e-10` per pixel and a kernel-Frobenius
   distance of 1.7% on full retrains.
3. **Step C — DESI port**: `tests/phase2_train_desi.py` + LoaArchive
   adapter + parameterized GPU SLURM scripts. Trains a Step-C model
   in ~5 h on an A40 (300k × 5662 × k=30, 1500 Adam iter).

Plus a **corr-noise debug arc** discovered while validating Step C:
the trained M matrices on 2lpt mocks had ~7× rougher off-diagonal
structure than v1 production. Root cause was a divergence from
MATLAB in the preprocessing order — fixed in `dataset.py` (normalize
then mask, matching `learn_qso_model.m:128` on already-normalized
nv) plus a tighter `|med| < 1e-2` rejection threshold.

## What landed

### Trainer + math (Steps A, B)

| File | Purpose |
|---|---|
| `gpy_dla_detection/training_v3/objective_vectorized.py` | Chunked vectorized `spectrum_loss_batch` |
| `tests/phase2_train_dr16.py` | DR16 reference trainer (MATLAB validation harness) |
| `tests/phase2_train_desi.py` | DESI port — k=30, num_lines=31, Turner+2024 priors, optional log_c_0 prior |
| `gpy_dla_detection/training/dataset.py::load_preprocessed_h5` | Normalize → mask → de-forest → IV-weighted center (MATLAB-faithful order, 2026-05-13) |
| `gpy_dla_detection/training/preload_from_loa_archive.py` | LoaArchive (compressed observed coadds) → trainset.h5 adapter, chunked-read to bound host RAM |
| `slurm/greatlakes/phase2_desi_smoke.sh` + `phase2_desi_retrain.sh` | GPU SLURM scripts parameterized by `RUN_NAME`, `PRELOAD`, norm band, optional log_c_0 prior |

### Tests

13 new tests added or strengthened (see `docs/test_results_overview.md` §1
for the full list). Highlights:

- `test_v1_matches_matlab.py` — v1 Python `spectrum_loss` ≡ MATLAB on
  5 frozen 2lpt spectra (`max rel_err = 5.30e-11`).
- `test_v3_objective_vectorized_parity.py` — `spectrum_loss_batch` ≡
  per-spectrum loop on 6 fixtures (`max rel = 1e-10`).
- `test_v3_train_step_parity.py` — 3-iter Adam parity vec ≡ per-spec
  (`max rel ~2e-10`).
- `test_v3_objective_vectorized_jacobian.py` — independent FD-vs-analytic
  on the batched function (`max rel = 4.01e-5`).
- `test_preload_from_loa_archive.py` — 9 sub-tests covering the
  LoaArchive adapter end-to-end (schema, z/ZWARN filter, NHI
  threshold, rest-frame interpolation, mask propagation,
  reproducibility, downstream `load_preprocessed_h5` compat).
- **NEW**: `test_normalize_rejection_threshold_is_1e_minus_2` —
  regression guard for the 2026-05-13 corr-noise fix.

All 65+ pre-existing tests still pass.

### Validation experiments (Step A.5, B kernel, B inference)

| Experiment | Result | Source |
|---|---|---|
| DR16 89k × 200 vec full retrain | 8h03m wall, c_0=0.106, τ_0=0.00449, β=3.03; `|Δβ|` vs MATLAB = 2.13 (Adam-vs-L-BFGS) | `docs/notes/2026-05-09_phase2_vec_full_vs_matlab.md` |
| DR16 89k × 200 per-spec full | 21h31m wall, scalars match vec to ~3 sig figs | (vec/per-spec comparison series) |
| Vec smoke 5k × 50 vs Phase-1 baseline | 28× speedup, scalars match | `docs/notes/2026-05-09_vec_smoke_vs_phase1_baseline.md` |
| Kernel agreement vec vs per-spec (full DR16) | 1.7% Frobenius on M·M^T, 0.95% on corr | `docs/notes/2026-05-11_vec_vs_perspec_full_comparison.md` |
| Inference consistency on canonical TID 120046865 | p_DLA Δ = 2.9e-3, log-evidence Δ ≈ 0.05 nats | `docs/notes/2026-05-11_vec_vs_perspec_inference_consistency.md` |
| Step C smoke 5k × 50 on 2lpt loa-0 wide (A40) | 0.43 s/iter, all 3 outputs (`.h5` + `.npz` + auto-README) | `docs/notes/2026-05-11_desi_smoke/` |

### Step C 2lpt trained models (6 variants)

3 variants × 2 datasets (`2lpt_loa0_wide`, `2lpt_loa124_nohcd_nobal_wide`):

- **base**: norm [1310, 1325], wide prior σ → β = 1.28 (heavy underfit)
- **_g**: norm [1310, 1325], strict Turner σ → β = 2.69
- **_m**: norm [1425, 1475], strict Turner σ → β = 3.09 ✓ recommended

Output dirs: `docs/notes/2026-05-11_desi_phase2_<RUN_NAME>/`.
Each contains `phase2_result.h5` (DESI inference schema), `phase2_result.npz`
(training history), and a model-card `README.md`.

### Corr-noise debug + 2026-05-13 fix

Trained Step C 2lpt models had corr(M·M^T) mean adj-diff ≈ 0.004,
~7× rougher than v1 production's 0.0006. Investigation chain:

1. PCA-init smoothness confirmed clean on 2lpt at both norm bands
   (`corr_pca_init_2lpt.png`, commit `badf0c2`) → noise emerges during
   Adam, not init.
2. Falsification probe (`examples/probe_outlier_tail_corr.py`) showed
   the smoking gun is **lower-tail marginal medians** (med ∈ [1.5e-3, 1e-2])
   — they pass the previous `|med| < 1e-3` rejection by 10× but
   produce `flux/med` 100–1000× bulk, dominating PCA's top eigenvector
   (14.9× smoothness blowup with just 10 such spectra in a 5000 batch).
3. **Root cause**: `dataset.py` masked raw nv against threshold 9 BEFORE
   normalizing, whereas MATLAB preload writes already-normalized nv and
   masks `nv/med² > 9` (`preload_qsos.m:63-64` + `learn_qso_model.m:128`).
   The MATLAB order is self-protecting against marginal medians; our
   Python wasn't.
4. **Fix**: (a) reorder `load_preprocessed_h5` to normalize → mask;
   (b) tighten threshold from `|med| < 1e-3` to `|med| < 1e-2`. Both are
   defense-in-depth; either alone closes the small-N gap.

Documentation: `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`
+ updated `docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md`
(prior ✓ on the mask line corrected).

### Other artifacts

- `docs/training_overview.md` — every GP-training file in the repo
  with status.
- `docs/test_results_overview.md` — correctness tests + benchmarks +
  SLURM job ledger.
- `docs/production_models.md` — guidance for downstream inference and
  the NERSC sampler-fix work on which `learned_file` to use.
- `docs/notes/2026-05-13_qso_emission_absorption_correlations/findings.md`
  — literature-review note on why real-LOA PCA-init kernels carry
  richer cross-correlation than 2lpt mocks (Baldwin, EV1, BAL-EV1
  coupling, intervening DLA metal forests).

## What's deliberately NOT in this PR

- **DLA-recovery validation** at scale on the Step C models — task
  filed (`docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md` §
  Next; tasks #6, #10 of the corr-noise arc).
- **β-drift investigation** — all 2lpt-trained models converge to
  β ≈ 1.3–3.1 not the Turner prior 3.62 (task #9).
- **README templating fix** — `phase2_train_desi.py` auto-emits a
  README that hard-codes `normalize | [1310, 1325]` regardless of
  runtime CLI; the _m variants say Garnett band but trained on MATLAB
  band. Cross-reference SLURM log header for the truth (task #7).
- **Saclay panel** in `examples/plot_pca_init_corr_multi.py` errored;
  fix in a follow-up (task #8).
- **c0prior gauge analysis** — SLURM 50021381 still running; will land
  ~2026-05-13 PM.

These are deferred to a follow-up PR (or kept open as standalone work).

## In-flight SLURM as of 2026-05-13 16:00

These were submitted under the pre-reorder pipeline; results land
~2026-05-13 PM → 2026-05-14 AM. They serve as the pre-reorder baseline
against which the post-reorder duplicates will be compared:

| JobID | Run | ETA |
|---|---|---|
| 50017771 | `loa_no_dla_no_bal_wide_g` (norm [1310, 1325]) | 2026-05-14 06h |
| 50017772 | `loa_no_dla_no_bal_wide_m` (norm [1425, 1475]) | 2026-05-14 07h |
| 50017773 | `loa_no_hcd_with_bal_wide_g` | 2026-05-14 05h |
| 50017774 | `loa_no_hcd_with_bal_wide_m` | 2026-05-14 05h |
| 50021381 | `2lpt_loa124_nohcd_nobal_wide_c0prior` | 2026-05-13 21h |
| 50072213 | smoke `desi_smoke_normmask` (validates the reorder fix at 5k×50) | (queued) |

Post-reorder production duplicates will be submitted after smoke 50072213
validates the fix at scale.

## Test plan

- [ ] All 65+ tests pass (`python -m pytest tests/ -v`)
- [ ] Smoke 50072213 lands; verify trained corr smoothness drops toward
      v1's 0.0006
- [ ] At least one 2lpt _m_normmask duplicate retrain confirms the
      production-scale corr improvement
- [ ] DLA-recovery quick check on canonical TID 120046865 against
      `2lpt_loa0_wide_m/phase2_result.h5` and `model_epoch_920.h5` —
      p_DLA delta within ~3e-3

## Notes for reviewers

- The MATLAB-faithful reorder of `dataset.py` is the single
  highest-impact behavioral change in this PR. The audit doc explicitly
  flags the prior ✓ as corrected, so reviewers can verify the new
  order matches MATLAB exactly.
- The Adam-vs-L-BFGS β discrepancy on the DR16 retrain (`|Δβ| = 2.13`)
  is an optimizer choice, not a math bug — `test_v3_objective_vectorized_jacobian.py`
  asserts the analytic gradient matches FD; `test_v1_matches_matlab.py`
  asserts the loss values match.
- The 6 Step C 2lpt trained models were trained pre-reorder. They are
  documented as "best available pre-reorder 2lpt" in
  `docs/production_models.md`; the post-reorder retrains will supersede
  them.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
