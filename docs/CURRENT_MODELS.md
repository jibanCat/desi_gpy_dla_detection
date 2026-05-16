# CURRENT_MODELS.md

> **Read this first** if you're picking a `learned_file` for inference.
> Updated when new training runs land or supersede.
> Last touched: 2026-05-15.
>
> Long-form decision matrix + caveats: `docs/production_models.md`.
> DLA-recovery results for this update: `docs/notes/2026-05-15_dla_recovery_post_reorder/findings.md`.

## Use-case → recommended model

| Use case | Model | Path | Status |
|---|---|---|---|
| **Real DESI Y3 LOA inference — proven baseline** | v1 production, epoch 920 | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5` | ✓ known-good (β=2.41, c_0=0.174); production. p_DLA=0.52, MAP log NHI=21.53 on canonical 2lpt TID (Δ=+0.27 dex) |
| **Real DESI Y3 LOA inference — validated post-reorder candidate** | `loa_no_dla_no_bal_wide_m_normmask_3000iter` (walltime@2243/3000) | `docs/notes/2026-05-13_desi_phase2_loa_no_dla_no_bal_wide_m_normmask_3000iter/phase2_result.h5` | ✓ COMPLETED (SLURM 50087967). β=3.57 (vs v1 2.41 — closer to Turner prior μ=3.62). **Real-LOA in-distribution recovery PASSED** (2026-05-15): recovers 96% of v1's confident strong DLAs at p_DLA>0.5 (93% at p_DLA>0.97), MAP log N_HI bias −0.04 dex vs v1. See `docs/notes/2026-05-15_dla_recovery_real_loa/findings.md`. |
| **2lpt mock inference — top pick** | `2lpt_loa124_nohcd_nobal_wide_m_normmask` (post-reorder) | `docs/notes/2026-05-14_desi_phase2_2lpt_loa124_nohcd_nobal_wide_m_normmask/phase2_result.h5` | ✓ COMPLETED (SLURM 50212621). Best DLA-recovery: p_DLA=0.762, MAP log NHI=21.52 (Δ=+0.25 dex). Supersedes pre-reorder `_m`. β=3.29, c_0=0.028 |
| **2lpt mock inference — loa-0 variant** | `2lpt_loa0_wide_m_normmask` (post-reorder) | `docs/notes/2026-05-14_desi_phase2_2lpt_loa0_wide_m_normmask/phase2_result.h5` | ✓ COMPLETED (SLURM 50212866). p_DLA=0.724, MAP log NHI=21.52 (Δ=+0.25 dex). β=3.31, c_0=0.026 |

## Models to AVOID

- **All `_g_normmask` / `_g` variants (Garnett norm band [1310, 1325])** — fail DLA-recovery on canonical TID (p_DLA = 0.10–0.14, vs `_m` band's 0.72–0.76 on the same target). Verified across both pre-reorder and post-reorder pipelines; the Garnett band is genuinely worse for DLA detection on this configuration. Use `_m` (MATLAB DR16 [1425, 1475]) variants instead.
- **`2lpt_loa124_nohcd_nobal_wide_c0prior`** — outlier on canonical TID (p_DLA=0.042), log_c_0 prior anchoring failed; equivalent to `_m` on 10-target random sample but worse on the canonical TID due to ‖M‖² inflation. See `docs/notes/2026-05-14_c0prior_failure_investigation/findings.md`. Prefer `_m`.
- **All `2026-05-11_*` pre-reorder Step C models** — superseded by the 2026-05-14 `_normmask` retrains (same `_m` variant gives marginally higher p_DLA: 0.762 vs 0.755). The pre-reorder pipeline had `corr(M·M^T)` mean adj-diff ≈ 0.004, ~7× rougher than v1 production (0.0006); the reorder + threshold fix tightens this.
- **`loa_no_hcd_with_bal_wide_m_normmask_3000iter`** — fails 2lpt OOD recovery (p_DLA=0.22). The BAL-included training set may be the cause; not recommended for production inference until validated on real LOA data.
- The `base` wide-σ variants (`2lpt_loa{0,124_nohcd_nobal}_wide`) — β collapsed to 1.28 from too-loose prior. Deprecated.

## Completed retrains landing log (2026-05-15)

All 6 post-reorder retrains have now landed:

| JobID | Run | n_iter (actual / target) | β endpoint | c_0 | τ_0 | DLA-recovery p_DLA on canonical TID |
|---|---|---:|---:|---:|---:|---:|
| 50087967 | LOA `loa_no_dla_no_bal_wide_m_normmask_3000iter` | 2243 / 3000 (walltime) | 3.566 | 0.079 | 0.00189 | 0.503 (OOD ✓ marginal) |
| 50087968 | LOA `loa_no_hcd_with_bal_wide_m_normmask_3000iter` | 2461 / 3000 (walltime) | 3.061 | 0.069 | 0.00228 | 0.215 (OOD ✗) |
| 50212621 | 2lpt `loa124_nohcd_nobal_wide_m_normmask` | 1500 / 1500 | 3.289 | 0.028 | 0.00452 | **0.762** (best) |
| 50212863 | 2lpt `loa0_wide_g_normmask` | 1500 / 1500 | 3.329 | 0.025 | 0.00392 | 0.104 (✗ _g band) |
| 50212866 | 2lpt `loa0_wide_m_normmask` | 1500 / 1500 | 3.311 | 0.026 | 0.00417 | 0.724 |
| 50212867 | 2lpt `loa124_nohcd_nobal_wide_g_normmask` | 1500 / 1500 | 3.315 | 0.028 | 0.00444 | 0.135 (✗ _g band) |

Headline: the reorder + threshold fix pulled β meaningfully toward Turner prior μ=3.62 (LOA _m at 3.57; pre-reorder _m models were 2.57–3.09). All `_m` variants pass DLA-recovery; all `_g` variants fail.

## How to load (DESI inference)

```python
from gpy_dla_detection.null_gp import NullGPMAT
gp = NullGPMAT(params, prior,
               learned_file="/path/from/table/above.h5")
```

`NullGPMAT.__init__` reads the .h5 and mutates `params.normalization_*_lambda`
in place if the file carries those fields (all post-2026-05-08 trained
models do).

## Updates / supersession history

| Date | Change |
|---|---|
| 2026-05-13 | Step C 2lpt `_m` variants pass DLA-recovery; c0prior flagged "not preferred" |
| 2026-05-14 | c0prior investigation: actually NOT broken, just outlier on canonical TID; updated guidance. dataset.py reorder + \|med\|<1e-2 threshold landed (commit aa36205). Post-reorder retrains submitted. |
| 2026-05-15 | All 6 post-reorder retrains landed. 2lpt `_m_normmask` retrains supersede pre-reorder Step C as 2lpt top picks (p_DLA 0.72–0.76, slightly better than pre-reorder counterparts). LOA `_m_normmask_3000iter` no-DLA-no-BAL marginally passes 2lpt OOD; flagged as leading real-LOA candidate pending real-LOA recovery test. All `_g_normmask` variants fail DLA-recovery — added to "AVOID" list. See `docs/notes/2026-05-15_dla_recovery_post_reorder/findings.md`. |
| 2026-05-15 | Real-LOA in-distribution recovery test for `loa_no_dla_no_bal_wide_m_normmask_3000iter` — PASSED. On 100 strong DLAs v1 confidently detected, the new model recovers 96% at p_DLA>0.5 / 93% at p_DLA>0.97, MAP log N_HI bias −0.04 dex (MAD 0.08). Validates the LOA candidate against v1 production on real spectra. See `docs/notes/2026-05-15_dla_recovery_real_loa/findings.md`. |
