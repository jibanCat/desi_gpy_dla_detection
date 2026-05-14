## Status — 2026-05-13 EOD

**Ready for review.** Three independent agent audits today recommend SHIP:

| Audit | Result | Doc |
|---|---|---|
| dataset.py math vs MATLAB | ✓ correct, NaN-safe, MATLAB-equivalent reorder | `docs/notes/2026-05-13_code_review_dataset_math.md` |
| PR diff (5 commits, 31 files) | ✓ no critical/high; h5 manifest is purely additive | `docs/notes/2026-05-13_code_review_pr_diff.md` |
| β-drift puzzle | no bug — data prefers β<3.62 under σ=0.04; v1 also β=2.41 on real LOA | `docs/notes/2026-05-13_beta_drift_investigation/findings.md` |
| Inference safety (DLA-recovery on canonical TID) | 2/3 main models pass; **c0prior collapses** (flagged) | `docs/notes/2026-05-13_step_c_dla_recovery/findings.md` |

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

- **README templating fix** (`tests/phase2_train_desi.py:358` hard-codes `[1310, 1325]`) — already noted in `docs/production_models.md`; one-line fix
- **Saclay panel error** in `examples/plot_pca_init_corr_multi.py` — separate follow-up
- **Production retrain at 3000 iter** of post-reorder LOA — in flight (50087967/68)
- **c0prior gauge-analysis investigation** — c0prior model collapses DLA detection (`docs/notes/2026-05-13_step_c_dla_recovery/findings.md`); mechanism not yet investigated, flagged "not production ready" in `docs/production_models.md`

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
  - 2lpt loa-124 c0prior: p_DLA = 0.04, NaN — **flagged, do not use for production**

## Notes for reviewers

- The dataset.py reorder is the single highest-impact behavioral change. The MATLAB↔Python audit explicitly flags the prior ✓ as a 2026-05-13 correction (`docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md`).
- The Adam-vs-L-BFGS β discrepancy on DR16 (`|Δβ| = 2.13`) is an optimizer choice, not a math bug. `test_v3_objective_vectorized_jacobian.py` asserts analytic grad ≡ FD; `test_v1_matches_matlab.py` asserts loss ≡ MATLAB.
- β drift puzzle resolved (no bug, hypothesis (b) — see `docs/notes/2026-05-13_beta_drift_investigation/findings.md`).
- 6 Step C 2lpt trained models are pre-reorder; documented as "best available pre-reorder 2lpt" in `docs/production_models.md`. Post-reorder retrains (50087967/68) will supersede.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
