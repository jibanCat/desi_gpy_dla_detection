# Production model recommendations

> 🟢 **Quick pick: see `docs/CURRENT_MODELS.md`** for the one-screen
> "use this model" pointer. This document is the long-form decision
> matrix with full caveats / endpoint scalars / DLA-recovery numbers.
>
> **Audience**: anyone running DLA-detection inference (`desi-DLAGP.py`,
> `DLAHolder`, the NERSC SLURM stack, the sampler-fix work).
> **Last updated**: 2026-05-14. Re-update each time a new trained model
> lands or a known caveat changes.
>
> See also:
> - `docs/architecture.md` — pipeline overview
> - `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md` — why the
>   pre-reorder Step C models have ~7× rougher kernel than v1
> - `docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/findings.md`
>   — MATLAB ↔ trainer audit (corrected 2026-05-13)
> - `docs/test_results_overview.md` — SLURM job ledger

## TL;DR

| Use case | Recommended `learned_file` | Reason |
|---|---|---|
| **Production inference NOW (real DESI Y3 LOA)** | `model_epoch_920.h5` (v1 production) | Known good; used in all 2025 inference runs; corr kernel is the cleanest available. |
| **Production inference on a 2lpt mock** | `2lpt_loa0_wide_m/phase2_result.h5` or `2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5` | Best of the PR #6 Step C 2lpt models. Norm band [1425, 1475] (MATLAB), strict Turner+2024 σ. β=2.69–3.09 closest to Turner 3.62. |
| **Real DESI Y3 inference at production scale** | **wait** | Post-reorder LOA retrains are not done yet; pre-reorder LOA runs land ~2026-05-14 PM but carry the same corr-roughness caveat as the 2lpt models. |
| **Sampler-fix correctness check** | v1 production *and* one Step C 2lpt _m model | Compare against both. If a sampler fix changes p_DLA more than ~3e-3 on a fixed model, that's a sampler effect, not a model effect. |

## Available trained models — 2026-05-13

All paths below assume the GreatLakes mount. NERSC equivalents:
replace `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/`
with `/pscratch/sd/j/jibancat/desi_gpy_dla_detection/` (when synced).

### Reference — v1 production (used for all 2025 inference)

| Field | Value |
|---|---|
| Path | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/learnlogs/model_epoch_920.h5` |
| Rest grid | [850.90, 1420.60], n_pix=3798, dλ=0.15 |
| k | 30 |
| Trained on | real DESI Y3 LOA spectra |
| Endpoint scalars | c_0=0.174, τ_0=0.00210, β=2.41 |
| corr(M·M^T) mean adj-diff | **0.0006** (smooth) |
| Status | ✓ production-ready |

### PR #6 Step C — 2lpt × 3 variants × 2 datasets (pre-reorder pipeline)

All trained 2026-05-12 on commit `c8a5464` or `badf0c2`. All 1500 Adam
iter, lr=0.005, k=30, num_lines=31, PCA init, hand-coded gradient via
`spectrum_loss_batch`. Same preload, three normalization/prior knobs.

| Variant | Norm band | Prior σ | c_0 | τ_0 | β | Path |
|---|---|---|---:|---:|---:|---|
| base | [1310, 1325] | wide (0.00064 / 0.074) | 0.0040 | 0.00054 | **1.28** | `docs/notes/2026-05-11_desi_phase2_2lpt_loa0_wide/phase2_result.h5` |
| _g | [1310, 1325] | strict Turner (0.00014 / 0.04) | 0.0193 | 0.00154 | 2.69 | `…_2lpt_loa0_wide_g/phase2_result.h5` |
| **_m** | **[1425, 1475]** | **strict Turner** | **0.0230** | **0.00173** | **3.09** | `…_2lpt_loa0_wide_m/phase2_result.h5` |

Same 3 variants exist for `2lpt_loa124_nohcd_nobal_wide` (HCD+BAL masked).
Endpoint scalars there are within ~10% of the loa-0 numbers above.

### DLA-recovery on canonical TID 120046865 (truth log_NHI = 21.263)

Per `docs/notes/2026-05-13_step_c_dla_recovery/findings.md` — operational
check that the corr-noise debug arc hasn't broken inference:

| Model | p_DLA | MAP log NHI | Δ vs truth | Status |
|---|---:|---:|---:|---|
| v1 production (epoch 920) | 0.52 | 21.53 | +0.27 dex | ✓ historical baseline (matches v1 +0.27 dex bias) |
| 2lpt loa-0 _m | **0.70** | 21.52 | +0.25 dex | ✓ passes (p_DLA > 0.5, MAP ±0.5 dex) |
| 2lpt loa-124 _m | **0.76** | 21.52 | +0.25 dex | ✓ passes |
| 2lpt loa-124 c0prior | 0.04 | NaN | — | ⚠ **outlier on this target — use `_m` instead** (see below) |
| Smoke post-reorder (50 iter only) | 0.85 | 21.63 | +0.36 dex | undertrained but detects (sanity only) |

**Headline**: the two main `_m` models pass — the corr-noise debug arc is
inference-safe.

**Update 2026-05-14 on the c0prior model** (after focused investigation,
`docs/notes/2026-05-14_c0prior_failure_investigation/findings.md`):

- On a 10-target random sample of strong 2lpt-loa-124 DLAs the c0prior
  model performs **identically to `_m`** (7/10 detected, same 3/10 missed).
- The canonical-TID 0.04 vs 0.76 divergence is an outlier in a narrow
  gap between the two models' truncated-marginal thresholds; not a
  systematic detection failure.
- The log_c_0 prior **failed to anchor c_0** (c_0 still drifted from 0.1
  → 0.020, vs `_m`'s 0.024 — endpoints differ by only 0.17 dex).
- Real difference: norm band [1310, 1325] vs `_m`'s [1425, 1475] + 13×
  inflated `‖M‖_F²` (21317 vs 1648). The wider M envelope drags
  borderline QMC marginals below null.
- Multi-DLA NaN is the production pipeline's deliberate early-stop at
  `gpy_dla_detection/dla_gp.py:790-810`, **NOT** a Cholesky failure —
  both models hit it for k ≥ 3 on this target.
- **Recommendation: use `_m` for production**, drop the c0prior recipe.
  If you want to fix the (c_0, M) gauge degeneracy in the future, prefer
  reparameterisation or direct L2 on M over a weak Gaussian prior on log_c_0.

⚠ **Pre-reorder caveat**: all 6 carry corr(M·M^T) mean adj-diff ≈ 0.004,
~7× rougher than v1 production. Root cause: the mask order in the
training-time preprocessing was reversed vs MATLAB, letting marginal-
median spectra (`med ∈ [1e-3, 1e-2]`) reach PCA with inflated centered
values. Fixed in `dataset.py` 2026-05-13 (commit pending). The fix
applies to all *future* retrains; the 6 trained models above were
trained before the fix and inherit the roughness.

**Inference impact (TBD)**: not yet measured. The corr-roughness is
visible in the kernel but its effect on `p_DLA` is unknown. Task #6 in
the corr-noise debug arc.

### Recommended of the 2lpt set: `_m` variants

| Why | What |
|---|---|
| Norm band matches MATLAB DR16 (and the new dataset.py default) | [1425, 1475] |
| Strict Turner+2024 prior σ; β=3.09 is closest to the prior 3.62 | β |
| Loss endpoint highest of the 3 (less overfit to forest physics) | log p(D \| Adam) = 8.83e8 |

## Pre-reorder LOA runs (TIMED OUT 2026-05-13, superseded by post-reorder)

The first wave of LOA training (50017771-74) hit the 12h SLURM kill
before saving a final .h5. Last checkpoints preserved on scratch
(iter 699-799 of 1500). Status:

| JobID | Run name | Final state | Last iter |
|---|---|---|---:|
| 50017771 | `loa_no_dla_no_bal_wide_g` | TIMEOUT | 699 |
| 50017772 | `loa_no_dla_no_bal_wide_m` | TIMEOUT | 699 |
| 50017773 | `loa_no_hcd_with_bal_wide_g` | TIMEOUT | 774 |
| 50017774 | `loa_no_hcd_with_bal_wide_m` | TIMEOUT | 799 |
| 50021381 | `2lpt_loa124_nohcd_nobal_wide_c0prior` | COMPLETED (1500 iter) | — |
| 50072213 | `desi_smoke_normmask` (5k × 50 iter) | COMPLETED | — |

All superseded by the post-reorder retrains below.

## In flight (post-reorder, submitted 2026-05-13/14)

The 2026-05-13 dataset.py reorder + `|med| < 1e-2` threshold closes
the corr-noise gap at small scale. Six new training runs now in flight
on the post-reorder pipeline; will replace the pre-reorder Step C
models as production once they pass DLA-recovery validation:

| JobID | Run name | n_iter | ETA |
|---|---|---:|---|
| 50087967 | `loa_no_dla_no_bal_wide_m_normmask_3000iter` | 3000 | 2026-05-15 ~06h |
| 50087968 | `loa_no_hcd_with_bal_wide_m_normmask_3000iter` | 3000 | 2026-05-15 ~09h |
| 50212621 | `2lpt_loa124_nohcd_nobal_wide_m_normmask` | 1500 | 2026-05-14 evening |
| 50212863 | `2lpt_loa0_wide_g_normmask` | 1500 | 2026-05-14 evening |
| 50212866 | `2lpt_loa0_wide_m_normmask` | 1500 | 2026-05-14 evening |
| 50212867 | `2lpt_loa124_nohcd_nobal_wide_g_normmask` | 1500 | 2026-05-14 evening |

Output dirs: `docs/notes/2026-05-1[34]_desi_phase2_<RUN_NAME>/`.
Live ETA + last-checkpoint-iter + sacct state in each scratch dir's
`README.md` (`/scratch/.../phase2_desi/<run>/README.md`).

## Known caveats (apply to all current models)

1. **corr(M·M^T) roughness** (pre-reorder only, fix in flight) — see
   `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`. Inference
   impact not yet measured (task #6).

2. **β drift** — all 2lpt-trained models converge to β ≈ 1.3–3.1
   regardless of prior σ. Even strict Turner σ=0.04 isn't enough to
   anchor β at 3.62. v1 production also landed at β=2.41 (also below
   3.62). Hypothesis: the data genuinely prefers lower β, OR the prior
   strength is being computed inconsistently with the likelihood
   gradient. Investigation pending (task #9).

3. **c_0 difference (2lpt vs real)** — 2lpt c_0 ≈ 0.004–0.023, v1 c_0 =
   0.174 (30–40× higher). Reflects different absolute flux normalization
   in the LyaColore mocks vs real DESI spectra, not a bug. The 2lpt
   models can still be used for 2lpt inference because c_0 is part of
   the per-mock calibration.

4. **README templating bug — FIXED 2026-05-14** (commits `660ee34` +
   `c38dc57`): `phase2_train_desi.py::_save_readme` now interpolates
   the runtime `--norm-min-lambda`/`--norm-max-lambda` correctly, and
   `examples/reemit_step_c_readmes.py` re-emitted all 8 existing
   READMEs with the correct band + status header.

5. **Rest grid mismatch with v1** — Step C 2lpt models are on the
   "wide v2" rest grid `[850.75, 1700]`, n_pix=5662, dλ=0.15. v1
   production is `[850.90, 1420.60]`, n_pix=3798, same dλ. Inference
   loader (`NullGPMAT`) handles this via `set_parameters` overrides
   when the .h5 carries `normalization_*_lambda` — already wired.

## Recommendations by use case (decision matrix)

### A. NERSC production sampler-fix correctness check

Use **two** models and check the sampler change doesn't move `p_DLA`
more than the model-to-model dispersion would explain:

- Model 1 (clean): `model_epoch_920.h5` — v1 production.
- Model 2 (Step C control): `2lpt_loa0_wide_m/phase2_result.h5` — best
  pre-reorder 2lpt model.

If the sampler fix changes `p_DLA` by ≥ a few × 10⁻³ on a fixed model,
that's a sampler effect. Inter-model dispersion across the two should
be `O(10⁻²)`; anything bigger flags a real model difference (likely the
corr-roughness; see caveat #1).

### B. NERSC 2lpt mock production inference

Pick `2lpt_loa<N>_wide_m/phase2_result.h5` matching the mock's contamination
treatment. **Note**: in the 2LPT mock-0 naming convention, `loa-0` is the
*uncontaminated* baseline (pure Lyα forest + continuum + DESI noise) and
`loa-124` is the *contaminated* preset (DLAs + sub-DLAs + BALs + simple
H-correlated metals overlaid). See
`docs/notes/2026-05-13_2lpt_loa0_vs_loa124_implementation/findings.md`.

- 2lpt loa-0 (uncontaminated baseline): `2lpt_loa0_wide_m`
- 2lpt loa-124 nohcd-nobal (contaminated mock with HCDs/BALs anti-joined
  out at preload time): `2lpt_loa124_nohcd_nobal_wide_m`

These are the cleanest pre-reorder 2lpt models. The corr-roughness
caveat applies but it's the best we have until the post-reorder
retrain completes.

### C. NERSC real-LOA production inference

Two paths:

1. **Conservative** (recommended unless time-critical): keep using
   `model_epoch_920.h5`. The pre-reorder Step C LOA models (50017771-74)
   will land ~2026-05-14 but carry the same corr-roughness caveat as
   the 2lpt models. The post-reorder LOA retrain has not been queued yet.

2. **Aggressive**: when 50017772 (`loa_no_dla_no_bal_wide_m`) lands,
   compare its `p_DLA` to v1 production on a canonical TID + a small
   batch. If consistent within ~3e-3, switch. Otherwise wait for the
   post-reorder version.

### D. Per-spectrum smoke / 1-target tests

Either model works. Prefer v1 for known-good baselines; the Step C 2lpt
models are useful for 2lpt-mock-specific calibration.

## How to load (production-ready)

```python
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog

params = Parameters(...)
prior = PriorCatalog(params, catalog_path, los_path, dla_path)
gp = NullGPMAT(
    params, prior,
    learned_file="/path/to/phase2_result.h5",   # or model_epoch_920.h5
)
# gp picks up normalization_{min,max}_lambda from the .h5 automatically.
```

The DESI inference loader at `null_gp.py:440-503` reads both the v1
schema (`M`, `mu`, `log_omega`, `log_c_0`, `log_tau_0`, `log_beta`)
and the v2/Step-C schema (adds `normalization_{min,max}_lambda`,
`max_noise_variance`, `rest_wavelengths`).

## Coordination with NERSC sampler-fix agent

- The reorder + threshold fix in `dataset.py` only affects *training*.
  Inference is untouched. Sampler-fix work proceeds on any of the
  models above without dependency on the training pipeline fix.
- If the sampler agent needs a clean kernel to validate the fix
  against, use v1 production. The Step C trained models are not yet
  "verified" for production inference (task #6).
- When new trained models land in `docs/notes/2026-05-1[123]_*`, this
  doc gets updated. Pull the latest `claude/debug-trainer-from-v1`
  before pinning a model path in a long-running NERSC job.
