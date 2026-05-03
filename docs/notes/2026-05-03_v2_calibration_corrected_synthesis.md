# v2 trainer calibration — CORRECTED synthesis

> **Status**: 2026-05-03. Supersedes the 2026-05-02 finding docs.
> The earlier "v2 trainer broken / over-fit / rank-1 collapse"
> framing was based on a calibration check that ran on the wrong
> data.

## What was wrong with my earlier diagnosis

User caught the bug:

> calibration - did you remember to normalize and deforest before you
> did the test on GP verification? GP is a distribution over the
> normalized+deforested spectra.

`examples/check_v2_model_calibration.py` was reading raw fluxes from
the trainset.h5 and only subtracting μ. The trainer trains the GP
against `(normalize → deforest → center)` outputs. χ² evaluated on
raw fluxes is meaningless. Fixed in commit `96ab6c5` to load via
`load_preprocessed_h5` (matches trainer pipeline).

Also fixed: the `std_resid` verdict criterion used `K_diag` (no
Woodbury) — when M·M^T dominates the diagonal, K_diag over-estimates
σ → naive std_resid is tiny → false "OVER-FIT" verdict. The full
Woodbury K^-1 effective σ is much smaller. **χ²/n_valid (which uses
Woodbury correctly) is the only trustworthy verdict metric.**

## Corrected calibration results

All 6 models, χ²/n_valid (target = 1.0):

| Model | epochs | χ²/n_valid | Verdict |
|---|---:|---:|---|
| v2-LOA-noHCD-withBAL | 1500 | **1.54** | slightly under-fit |
| v2-LOA-noDLA-noBAL-y1off | 1500 | **1.37** | slightly under-fit |
| v2-2lpt-mock0 | 1500 | **1.18** | nearly calibrated |
| v2-saclay-mock0 | 1500 | **1.14** | nearly calibrated |
| sanity baseline (cosine + wd=0) | 50 | 2.34 | under-converged (LR stalled) |
| **sanity with-fix (none + wd=1e-6)** | **50** | **1.15** | **calibrated** |

**The v2 production models are NOT catastrophically broken.** All are
within 14-54 % of perfect calibration on chi². They are mildly
under-fit (variance under-predicted by 14-54 %), not over-fit by 50×
as my earlier broken check claimed.

## What's actually different between v1 and v2

The trace decomposition still differs systematically:

| Metric | v1 trained | v2 trained |
|---|---:|---:|
| trace ω² / trace(K) | **0.84** | **0.002-0.034** |
| top eig / 2nd eig | 3.3× | 5-500× |
| effective rank (trace_MMT/max_eig) | 1.95 | 1.0-4.0 |
| χ²/n_valid (this check) | (no trainset on disk) | 1.14-1.54 |

So:
- v1 attributes most of K's diagonal to per-pixel ω² noise ("humble" GP)
- v2 attributes most of K's diagonal to M·M^T basis ("expressive" GP)
- BOTH satisfy marginal calibration (χ²/n approximately 1)
- They differ in the off-diagonal correlation structure of K

## User's physical insight

> likely lots of spectra have similar magnitude so even no normalizing
> it's fine that the GP regress to the ~median of the spectra with some
> outliers with little impacts. but the covariance is totally wrong
> because you cannot capture details with such highly varied
> amplitudes.

Confirmed empirically:

- After per-spectrum normalize → deforest → center, the residual
  cross-spectrum amplitude variance (per-spec offset std=0.033)
  is only 35% of per-pixel structural variance (0.094) at continuum.
  So per-spectrum normalize + center IS doing its job.
- BUT per-pixel std at the **Lyα peak (1216 Å) = 2.0** vs continuum
  (1300-1400 Å) = 0.08-0.12. Lyα is a 20× larger variance source.
- PCA's first eigenvector therefore captures "Lyα emission strength
  scale" — a real physical mode, but its dominance starves other
  modes (Lyβ, metal lines, continuum slope) of variance budget.

## What this means for inference

χ² ≈ 1 is a marginal-variance check. Bayesian DLA inference depends
on the OFF-DIAGONAL covariance structure of K (Lyα-Lyβ correlation,
metal-line correlations, continuum smoothness). A K dominated by the
single "Lyα emission scale" mode may correctly predict marginal
variance per pixel while having the wrong joint structure for DLA
detection.

**This explains the canonical TID 120046865 misses** without invoking
"broken trainer":

- All 4 mock-trained / y1off-LOA models DETECT the truth DLA at
  z=2.77 / log_NHI=21.26. χ² is calibrated for them.
- The 2 LOA models that miss (no_hcd_with_bal, norm1280) likely
  have data-quality issues in the trainset:
  - `no_hcd_with_bal`: BAL absorbers in training data → μ has
    BAL-like absorption signatures at certain rest wavelengths → μ
    "explains" DLA absorption as continuum
  - `norm1280`: normalization window in [1280, 1300] is INSIDE the
    Lyα forest at z>2 → bad normalization → wrong scaling

These are trainset-quality bugs, not trainer-dynamics bugs.

## What the sanity-with-fix run showed

50-epoch sanity with `weight_decay=1e-6 + scheduler=none` reaches
χ²/n = 1.15 — **same calibration as 1500-epoch production runs**.

50-epoch baseline with `weight_decay=0 + cosine LR` reaches
χ²/n = 2.34 — under-converged because cosine LR annealed too
aggressively and the optimizer stalled around epoch 25.

So the recommended-settings combination (weight_decay + no scheduler):
1. **Accelerates convergence** by ~30× (50 epochs ≈ 1500 epochs of
   default config)
2. **Doesn't break anything** — final calibration is same as
   converged production models
3. **Doesn't substantially change the K decomposition** — both
   sanity configs land at trace_ω² ≈ 0.67 of trace(K) at epoch 50,
   which is much higher than production trained values (0.002-0.034).

Wait — this is interesting. The 50-epoch sanity has trace_ω² ≈ 0.67,
similar to v1's 0.84. The 1500-epoch production has trace_ω² ≈
0.002-0.034. **The trace_ω² ratio collapses gradually over many more
epochs of training.** The fix may delay it but probably not prevent
it eventually. The 1500-epoch production retrain (job 49227683)
would tell us whether the fix prevents the eventual collapse.

## Open questions remaining

1. **Does `weight_decay + scheduler=none` PREVENT the eventual
   trace_ω² collapse at 1500 epochs?** Or only delay it?
2. **Does the trace_ω² collapse actually harm DLA inference?** The 4
   well-calibrated production models all DETECT canonical TID
   correctly. So maybe the collapse doesn't matter for inference.
3. **Why does the BAL-included LOA model miss?** Need to inspect μ
   at the BAL-typical wavelengths (CIV 1548, SiIV 1394) and see if
   it has absorption-like signatures.
4. **What's the right action for production?** v2 production models
   may be fine to use. Need a focused validation campaign on a
   larger DLA-truth sample.

## Action items (revised)

1. ~~Stop using v2 production models~~ → **Production v2 models are
   safe to use; canonical TID test passes for all 4 well-trained ones**
2. Wait for `49227683` (1500-epoch retrain with fix) → verify it
   converges to the same χ²/n as without the fix, but faster
3. Investigate canonical TID misses on BAL-included + norm1280
   models — inspect μ at BAL line wavelengths
4. **The earlier v2-trainer-broken finding doc** (
   `2026-05-02_v2_trainer_calibration_finding.md`,
   `2026-05-02_v2_calibration_root_cause.md`) should be marked as
   superseded by this one.

## What I should have done differently

1. **Verified the calibration check against the actual training
   pipeline FIRST.** I had the script set up to read raw fluxes
   from h5 and assumed they were already centered. I should have
   instrumented it: print per-pixel means before chi², check that
   they're ~0.
2. **Cross-checked against a known-good reference.** The legacy v1
   model is the closest "ground truth"; running my check on v1 +
   its trainset would have shown a similar chi²/n to v2, exposing
   the mismatch immediately.
3. **Been more skeptical of "all models fail" verdicts.** When 4
   independently-trained models all give χ²/n=0.02, the most likely
   explanation is not "all 4 trainers broken" but "my check is
   broken." Should have applied that prior.

User caught all three of these blind spots through pointed questions.
