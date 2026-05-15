# CURRENT_MODELS.md

> **Read this first** if you're picking a `learned_file` for inference.
> Updated when new training runs land or supersede.
> Last touched: 2026-05-14.
>
> Long-form decision matrix + caveats: `docs/production_models.md`.

## Use-case → recommended model

| Use case | Model | Path | Status |
|---|---|---|---|
| **Real DESI Y3 LOA inference (NOW)** | v1 production, epoch 920 | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5` | ✓ current; known-good baseline (β=2.41, c_0=0.174) |
| **Real DESI Y3 LOA inference (will supersede when landed)** | post-reorder LOA `_m_normmask` 3000-iter retrains | `docs/notes/2026-05-13_desi_phase2_loa_no_dla_no_bal_wide_m_normmask_3000iter/phase2_result.h5` (in flight, SLURM 50087967) | ⏳ landing ~2026-05-15 AM |
| **2lpt mock inference (NOW)** | pre-reorder Step C `_m` | `docs/notes/2026-05-11_desi_phase2_2lpt_loa{0,124_nohcd_nobal}_wide_m/phase2_result.h5` | ⚠ pre-reorder caveat (corr ~7× rougher than v1 but inference works: p_DLA=0.70-0.76 on canonical TID) |
| **2lpt mock inference (will supersede when landed)** | post-reorder 2lpt `_m_normmask` 1500-iter retrains | `docs/notes/2026-05-14_desi_phase2_2lpt_{loa0,loa124_nohcd_nobal}_wide_m_normmask/phase2_result.h5` (in flight) | ⏳ landing ~2026-05-14 evening / 2026-05-15 AM |

## Models to AVOID

- `2lpt_loa124_nohcd_nobal_wide_c0prior` — outlier on canonical TID, not better than `_m` on a 10-target sample, log_c_0 anchoring failed. See `docs/notes/2026-05-14_c0prior_failure_investigation/findings.md`. Prefer the `_m` variant instead.
- Any pre-reorder `_g` (norm [1310, 1325]) variants — superseded by `_m` ([1425, 1475] MATLAB band). Kept for trail only.
- The `base` wide-σ variants (`2lpt_loa{0,124_nohcd_nobal}_wide`) — β collapsed to 1.28 from too-loose prior. Deprecated.

## In-flight retrains (post-reorder, expected to become current when they land)

| JobID | Run | Iter | Norm band | ETA |
|---|---|---:|---|---|
| 50087967 | LOA `loa_no_dla_no_bal_wide_m_normmask_3000iter` | 3000 | [1425, 1475] | 2026-05-15 ~06h |
| 50087968 | LOA `loa_no_hcd_with_bal_wide_m_normmask_3000iter` | 3000 | [1425, 1475] | 2026-05-15 ~09h |
| 50212621 | 2lpt `loa124_nohcd_nobal_wide_m_normmask` | 1500 | [1425, 1475] | 2026-05-14 evening |
| 50212863 | 2lpt `loa0_wide_g_normmask` | 1500 | [1310, 1325] | 2026-05-14 evening |
| 50212866 | 2lpt `loa0_wide_m_normmask` | 1500 | [1425, 1475] | 2026-05-14 evening |
| 50212867 | 2lpt `loa124_nohcd_nobal_wide_g_normmask` | 1500 | [1310, 1325] | 2026-05-14 evening |

Validation chain after each lands:
1. Render corr(M·M^T) at v1 rest range (`examples/plot_kernels_v1_rest_range.py`)
2. DLA-recovery on canonical TID 120046865 (`examples/dla_recovery_step_c.py`)
3. Update this doc + the "Use case" table

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
| 2026-05-14 | c0prior investigation: actually NOT broken, just outlier on canonical TID; updated guidance. dataset.py reorder + |med|<1e-2 threshold landed (commit aa36205). Post-reorder retrains submitted. |
