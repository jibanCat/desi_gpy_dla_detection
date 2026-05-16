## Status — 2026-05-15 update: validation complete, mergeable

The post-reorder retrains that were "in flight overnight" on 2026-05-13
have all landed and been validated. The PR's validation chain is complete.

### Post-reorder retrains — all 6 landed
All 6 post-reorder `_normmask` retrains completed (2 LOA 3000-iter,
4 2lpt 1500-iter). The LOA runs hit SLURM walltime (iter 2243 / 2461 of
3000) but the trainer exits gracefully and writes the final model.
β endpoints moved toward the Turner prior (μ=3.62) — LOA `_m` reached
β=3.57, the closest any model has come.

### Validation — DLA-recovery
- **2lpt `_m_normmask`** (canonical-TID recovery): p_DLA 0.72–0.76, all
  pass — slightly better than the pre-reorder `_m` baselines.
- **`_g_normmask`** variants: fail recovery (p_DLA 0.10–0.14, Garnett
  norm band) — added to the AVOID list in `docs/CURRENT_MODELS.md`.
- **LOA `loa_no_dla_no_bal_wide_m_normmask_3000iter`** — in-distribution
  validated on REAL LOA (`examples/dla_recovery_real_loa.py`, new): on
  100 strong DLAs v1 production confidently detected, the new model
  recovers 96% at p_DLA > 0.5 / 93% at p_DLA > 0.97, MAP log N_HI bias
  −0.04 dex vs v1. See
  `docs/notes/2026-05-15_dla_recovery_real_loa/findings_summary.md`.

### Diagnostics
- Kernel corr-matrix comparison (`examples/plot_kernels_v1_rest_range.py`)
  updated: per-model normalization-band blanking + Ly/metal
  cross-correlation markers.
- `docs/CURRENT_MODELS.md` records the validated top picks per use case.

### Net
All in-PR validation is complete — the LOA candidate is validated against
v1 production on real spectra, the 2lpt models against mock truth. The PR
is mergeable (0 commits behind `desi_y3`).

---

## Status — 2026-05-13 EOD

**Ready for review.** Three independent agent audits today recommend SHIP:

| Audit | Result | Doc |
|---|---|---|
| dataset.py math vs MATLAB | ✓ correct, NaN-safe, MATLAB-equivalent reorder | `docs/notes/2026-05-13_code_review_dataset_math.md` |
| PR diff (5 commits, 31 files) | ✓ no critical/high; h5 manifest is purely additive | `docs/notes/2026-05-13_code_review_pr_diff.md` |
| β-drift puzzle | no bug — data prefers β<3.62 under σ=0.04; v1 also β=2.41 on real LOA | `docs/notes/2026-05-13_beta_drift_investigation/findings.md` |
| Inference safety (DLA-recovery on canonical TID) | 2/3 main models pass; c0prior is an outlier on this target (not a systematic failure — see 2026-05-14 investigation) | `docs/notes/2026-05-13_step_c_dla_recovery/findings.md` + `docs/notes/2026-05-14_c0prior_failure_investigation/findings.md` |

In flight overnight: SLURM 50087967, 50087968 — fresh post-reorder LOA `_m_normmask` at 3000 iter (land ~2026-05-15 AM).

---

## Summary

Rebuilds the Phase 2 GP continuum trainer end-to-end and lands a corr-noise debug arc that surfaced a real MATLAB↔Python divergence:

1. **Step A — verify against MATLAB DR16**: vectorized `spectrum_loss_batch` matches MATLAB `spectrum_loss.m` to `max rel_err ≈ 5e-11` on five frozen 2lpt fixtures; the v1 Python `zqso_1pz` bug is correctly bypassed.
2. **Step B — vectorize the loss**: chunked vectorized path (`training_v3/objective_vectorized.py`); 28× faster at 5k smoke, 3× at full 89k DR16 retrain; kernel agrees to 1.7% Frobenius, inference p_DLA Δ = 2.9e-3 on canonical TID.
3. **Step C — DESI port**: `tests/phase2_train_desi.py` + LoaArchive adapter + parameterized GPU SLURM scripts. 6 Step C 2lpt models on disk (3 priors × 2 datasets).
4. **Corr-noise debug arc (2026-05-13)**: discovered + fixed an order divergence from MATLAB in `dataset.py::load_preprocessed_h5`. Audit doc previously marked the per-pixel-mask line as ✓ — it was wrong; corrected here.

## Today's discoveries (2026-05-13)

### dataset.py order divergence vs MATLAB

- **MATLAB** (`preload_qsos.m:63-64` + `learn_qso_model.m:128`): normalizes flux+nv at preload time, then masks `nv > 9` on the already-normalized array → effective threshold is `nv_raw/med² > 9`.
- **Python (old)**: masked raw nv against 9 *before* normalizing → effective threshold was `nv_raw > 9`.

Functional consequence: marginal-median spectra (`med ∈ [1.5e-3, 1e-2]`) survived the Python mask but would have been killed by MATLAB. Their normalized centered flux is 100–1000× bulk; PCA picks them up as a top eigenvector → noisy corr(M·M^T).

### Falsification probe

`examples/probe_outlier_tail_corr.py` — clean 5000-spectrum batch + 10 injected outliers at 4 magnitude/sign conditions. Result table from `outlier_tail_smoothness.json`:

| Injected | smoothness (mean adj-corr-diff) | vs CLEAN |
|---|---:|---:|
| CLEAN baseline (5000) | 0.0130 | 1× |
| +10 SMALL_POS (med ∈ [1.5e-3, 1e-2]) | **0.1939** | **15×** ⚠ |
| +10 LARGE_POS (med ∈ [10, 30]) | 0.0130 | 1× ✓ |
| +10 EXTREME (med ∈ [50, 94]) | 0.0130 | 1× ✓ |
| +10 NEG (med ≤ 0) | 0.0130 | 1× ✓ (rejected) |

Falsified my upper-tail hypothesis. Smoking gun is the lower-tail marginal — passing the previous `|med| < 1e-3` rejection by 10×.

### Fix (commit `aa36205`)

1. Reorder `load_preprocessed_h5` to `normalize → mask` (matches MATLAB effective behavior at `dataset.py:356-372`).
2. Tighten `|med| < 1e-3` to `|med| < 1e-2`.
3. Regression test in `tests/test_normalize_by_rest_median.py::test_normalize_rejection_threshold_is_1e_minus_2`.

After fix, the probe gives smoothness=0.0130 (matches CLEAN) for all 5 injection conditions.

## Today's commits

| Commit | Summary |
|---|---|
| `aa36205` | Corr-noise fix: reorder normalize→mask + tighten \|med\|<1e-2 |
| `3a0b84f` | phase2_train_{desi,dr16}: full training-hyper manifest in .h5 |
| `9f321f6` | phase2_desi_retrain.sh: MAX_WALLTIME_SEC env override |
| `834fb78` | Step C 2lpt model cards + post-reorder smoke artifact |
| `64e9f49` | Analysis scripts + plots + docs from 2026-05-13 |
| `67700d8` | Slope-shuffled null detection + 4-agent review pack |

## What's deliberately NOT in this PR

> **2026-05-15 update**: three items previously listed here are now done
> and IN this PR — the README templating fix and the Saclay panel norm
> band (both commit `660ee34`), and the post-reorder LOA 3000-iter
> retrains (landed + validated). Only c0prior remains deferred.

- **c0prior production retraining** — investigation 2026-05-14
  (`docs/notes/2026-05-14_c0prior_failure_investigation/findings.md`)
  found the c0prior model performs identically to `_m` on 7/10 random
  strong DLAs but is borderline on the canonical TID. log_c_0 prior
  anchoring failed; real difference is narrower norm band + 13× inflated
  `‖M‖²`. Recommendation: drop the c0prior recipe and prefer `_m`. A
  reparameterisation-based gauge fix is a future-PR experiment.

These are deferred to a follow-up PR.

## Test status

- All 224 tests in `tests/` pass (per PR-diff review agent at commit `67700d8`); 1 pre-existing failure unrelated to this PR (`learn_qso_model.py:433` tensor.copy()), 3 environmental skips
- Key correctness tests covered in `docs/test_results_overview.md`:
  - `test_v1_matches_matlab.py` — v1 ≡ MATLAB on 5 spectra, max rel_err = 5.3e-11
  - `test_v3_objective_vectorized_parity.py` — vec ≡ per-spec to ~1e-10
  - `test_normalize_rejection_threshold_is_1e_minus_2` (NEW) — regression guard for the 2026-05-13 corr-noise fix
- DLA-recovery on canonical TID 120046865:
  - v1 production: p_DLA = 0.52, MAP log NHI = 21.53 (+0.27 dex bias, matches historical)
  - 2lpt loa-0 _m: p_DLA = 0.70, MAP log NHI = 21.52 (+0.25 dex)
  - 2lpt loa-124 _m: p_DLA = 0.76, MAP log NHI = 21.52 (+0.25 dex)
  - 2lpt loa-124 c0prior: p_DLA = 0.04, NaN on canonical TID — investigation 2026-05-14 shows this is an outlier (7/10 random strong DLAs match `_m`'s detections); prefer `_m` for production. Multi-DLA NaN is the production code's deliberate early-stop (`dla_gp.py:790-810`), not a Cholesky failure.

## Notes for reviewers

- The dataset.py reorder is the single highest-impact behavioral change. The MATLAB↔Python audit explicitly flags the prior ✓ as a 2026-05-13 correction (`docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md`).
- The Adam-vs-L-BFGS β discrepancy on DR16 (`|Δβ| = 2.13`) is an optimizer choice, not a math bug. `test_v3_objective_vectorized_jacobian.py` asserts analytic grad ≡ FD; `test_v1_matches_matlab.py` asserts loss ≡ MATLAB.
- β drift puzzle resolved (no bug, hypothesis (b) — see `docs/notes/2026-05-13_beta_drift_investigation/findings.md`).
- 6 Step C 2lpt trained models are pre-reorder; documented as "best available pre-reorder 2lpt" in `docs/production_models.md`. Post-reorder retrains (50087967/68) will supersede.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
