# 2lpt corr(M·M^T) noise — debug findings

**Date**: 2026-05-12 → 2026-05-13
**Branch**: `claude/debug-trainer-from-v1` (PR #6)
**Question**: why is corr(M·M^T) in the PR #6 trained 2lpt models ~7× rougher
than v1 production trained on real LOA? (mean adj-pixel-corr-diff: 2lpt ≈ 0.004
vs v1 production ≈ 0.0006)

This note records the experiments run on 2026-05-12 and the remaining
hypothesis to test.

## TL;DR

1. The 2lpt trained-model corr(M·M^T) carries off-diagonal "noise" stripes
   that v1 production does not. The roughness metric is ~7× v1.
2. PCA init on 2lpt is **smooth** at both norm bands ([1310, 1325] and
   [1425, 1475]); adj-diff ≈ 0.0045, matching v1's *trained* level. → the
   stripes do NOT come from PCA init; they are induced during the Adam loop.
3. PCA init on real LOA looks superficially busier than on 2lpt, but per
   review with the science lead this is **richer cross-correlation of
   emission/absorption features** in the real-universe data (not noise).
   2lpt mocks look blockier because they lack physics the GP picks up on
   real data — see `2026-05-13_qso_emission_absorption_correlations/`
   (literature review, in progress).
4. The clean/dirty experiment (`corr_emergence_clean_vs_dirty.png`) shows
   that **10 outlier-median spectra detonate corr(M·M^T) smoothness from
   ~0.001 to ~0.47 at iter 0** when mixed into a 5000-spectrum batch.
   This drove the lower-tail median rejection in commit `3e76056`
   (`bad = ¬isfinite | (med ≤ 0) | (|med| < 1e-3)`).
5. **After** the lower-tail fix, trained 2lpt corr is still ~7× rougher
   than v1. The lower-tail rejection helped but did not fully close the
   gap. The remaining candidate is the **upper tail of normalization
   medians**, which is not rejected today.

## Median distribution in the 2lpt loa-0 wide preload

50k random sample from `2lpt_loa0_wide_v2_1778186324/trainset.h5`
(N=299811, n_pix=5662, rest [850.75, 1700]):

| | norm band [1310, 1325] | norm band [1425, 1475] |
|---|---:|---:|
| NaN | 0.00% | 0.00% |
| median ≤ 0 | **0.51%** | 0.00% |
| 0 < med < 1e-3 | 0.008% | 0.002% |
| 1e-3 ≤ med < 1e-2 | 0.094% | 0.034% |
| 1e-2 ≤ med < 1e-1 | 1.92% | 2.12% |
| 1e-1 ≤ med < 10 (bulk) | 95.32% | 96.34% |
| **med ≥ 10 (upper tail)** | **2.15%** | **1.50%** |
| 99.9-pct / median | 47.8× | 46.5× |
| max / median | **114×** | **111×** |

Current rejection (`dataset.py:169`) zeros only the rows where
median is NaN / ≤ 0 / |·|<1e-3 — i.e. roughly the *first three rows*
of the table. The 2.15% / 1.50% upper-tail (med ≥ 10) passes unscathed.

## Falsification result — 2026-05-13

Ran `examples/probe_outlier_tail_corr.py` on the 2lpt loa-0 wide preload
(norm band [1310, 1325]). For each condition, 5000 top-SNR bulk-median
spectra + 10 injected outliers, full dataset.py preprocessing with
current rejection rule, PCA init, smoothness = mean adjacent-pixel
|Δcorr|. Output: `corr_outlier_tail_test.png` + `outlier_tail_smoothness.json`.

| Condition | n_rejected | smoothness |
|---|---:|---:|
| CLEAN (5000) | 0 | 0.0130 |
| **+10 SMALL_POS (med ∈ [1.5e-3, 1e-2])** | **0** | **0.1939** ⚠ |
| +10 LARGE_POS (med ∈ [10, 30]) | 0 | 0.0130 |
| +10 EXTREME (med ∈ [50, 94]) | 0 | 0.0130 |
| +10 NEG (med ≤ 0) | 10 | 0.0130 ✓ |

**Conclusion**:

- The upper-tail hypothesis (large medians inflating IV weight) is **wrong**.
  Calibration invariance of the IV-weighted centering is preserved when both
  flux and noise variance scale together, which they do at the dataset level.
  Adding 10 spectra with med ∈ [10, 94] left the smoothness unchanged at 0.0130.
- The neg-median control behaves correctly: 10 spectra rejected, smoothness
  matches CLEAN.
- The **smoking gun is the lower-tail marginal**: 10 spectra with
  med ∈ [1.5e-3, 1e-2] — passing the current `|med| < 1e-3` rejection
  by ~10× — bumped smoothness 14.9× (0.0130 → 0.1939).
- Mechanism: flux/med becomes 100–1000× the bulk scale; the IV centering
  weight for these spectra is small (∝ med²) so the *mean* shifts little,
  but the *centered* flux for those rows is huge, and PCA picks up the
  resulting idiosyncratic high-variance direction as a top eigenvector,
  poisoning corr(M·M^T).

⚠ The CLEAN baseline at 5000 spectra is itself rough (0.0130 vs 30000-N
PCA init at 0.0045 and 237k full-train at 0.0041) — small-N PCA is
intrinsically noisier. The 14.9× small_pos / CLEAN ratio at matched N is
the real signal, not the absolute number.

## Fix applied 2026-05-13

Tightened `dataset.py:177` threshold from `|med| < 1e-3` to `|med| < 1e-2`,
plus an extra `[1e-3, 1e-2)` count in the diagnostic print so production
logs surface how many marginal-tail spectra are being caught.

Regression-guard test landed at `tests/test_normalize_by_rest_median.py
::test_normalize_rejection_threshold_is_1e_minus_2`: asserts a synthetic
spectrum at med=0.005 is now rejected, while med=0.02 and med=100 are
kept. All 20 tests in the file pass.

**Re-probe at the new threshold** (same `examples/probe_outlier_tail_corr.py`,
no other change):

| Condition | n_rejected (NaN rows) | smoothness | before fix |
|---|---:|---:|---:|
| CLEAN (5000) | 0 | 0.0130 | 0.0130 |
| **+10 SMALL_POS** ([1e-3, 1e-2)) | **10** | **0.0130** | 0.1939 |
| +10 LARGE_POS | 0 | 0.0130 | 0.0130 |
| +10 EXTREME | 0 | 0.0130 | 0.0130 |
| +10 NEG | 10 | 0.0130 | 0.0130 |

SMALL_POS smoothness drops from 0.1939 → 0.0130 (15× → 1×). Verdict:
the fix closes the small-N injection gap completely.

## Additional MATLAB-faithful fix — reorder normalize/mask (2026-05-13 PM)

After landing the `|med| < 1e-2` threshold, found a deeper divergence
from MATLAB. The training-pipeline audit doc had marked the per-pixel
mask as ✓, but:

- **MATLAB** (`preload_qsos.m:63-64` + `learn_qso_model.m:128`):
  preload normalizes flux/nv by med/med², then learn masks
  `nv > 9` on the already-normalized array. Effective threshold is
  `nv_raw/med² > 9`.
- **Our Python (old)** (`dataset.py::load_preprocessed_h5`):
  masked raw nv against 9, THEN normalized. So `nv_raw > 9` was the
  effective threshold.

Functional consequence: in MATLAB, a `med=0.005` spectrum gets
`nv_normed ≈ nv_raw / 2.5e-5 ≈ 400×` → most/all pixels masked →
spectrum effectively rejected. In our old Python, those pixels passed
the mask and the spectrum reached PCA. This is precisely why we saw the
trained-model gap.

**Fix** (`dataset.py:333-371`): reordered `load_preprocessed_h5` so
`_normalize_by_rest_median` runs before `_mask_high_noise_pixels`. The
mask now sees `nv_normed = nv_raw/med²`, matching MATLAB.

**Audit doc** (`../2026-05-12_training_pipeline_audit_vs_matlab/findings.md`)
updated to flag the prior ✓ as a 2026-05-13 correction.

## Probe at the reordered pipeline (2026-05-13)

| Condition | n_rejected (NaN rows) | smoothness |
|---|---:|---:|
| CLEAN (5000) | 0 | 0.0130 |
| +10 SMALL_POS ([1.5e-3, 1e-2)) | 10 | 0.0130 ✓ |
| +10 LARGE_POS | 0 | 0.0130 ✓ |
| +10 EXTREME | 0 | 0.0130 ✓ |
| +10 NEG | 10 | 0.0130 ✓ |

All conditions match CLEAN — defense in depth: SMALL_POS is now caught
by both the median threshold AND the reordered mask. Either fix on its
own would have closed the small-N gap; together they're redundant by
design.

## Next: validate at production scale

1. **Smoke** (in flight): SLURM 50072213 — 5k×50 on 2lpt loa-0 wide
   with the fixed pipeline. Target dir
   `docs/notes/2026-05-13_desi_smoke_normmask/`. Expected ~10 min wall.
2. If smoke shows the trained corr(M·M^T) smoothness drops from ~0.0041
   toward v1 production's ~0.0006, submit duplicate production runs
   (suffix `_normmask`) into new dirs alongside the in-flight ones so we
   can compare old-vs-new pipeline at full scale:
   - 2lpt loa-124 nohcd-nobal wide × 2 norm bands (1310/1325, 1425/1475)
     — fastest turnaround (270k → ~7h)
   - LOA real × selected variant — biggest impact (638k → ~22h)
3. In-flight runs (50017771-74, 50021381) are continuing on the old
   pipeline; they'll provide a contemporaneous baseline.

## What's NOT load-bearing

- Upper-tail medians (now ruled out by probe).
- PCA init smoothness intrinsic to 2lpt (confirmed clean at 30k-N).
- Norm band choice [1310, 1325] vs [1425, 1475] (both bands have
  comparable lower-tail fractions; the [1425, 1475] band actually has
  fewer marginals — 0.034% vs 0.094% — so it's somewhat self-protective).
- Strict vs wide τ_0/β prior σ.
- c_0 prior — testing on 50021381 (in flight, iter ~400/1500); will
  update this note when it lands.

## What's NOT load-bearing for the corr noise

- PCA init smoothness — confirmed clean on 2lpt at both norm bands.
- Strict vs wide τ_0/β prior σ — both produce noisy corr (see
  `2lpt_loa0_wide_g/_m` README endpoints).
- Norm band choice [1310, 1325] vs [1425, 1475] — both bands have
  comparable upper tails (table above).
- c_0 prior — testing on 50021381 (in flight, iter 400/1500); will
  update this note when it lands.

## Related artifacts

- `corr_emergence_clean_vs_dirty.png` (2026-05-12 12:49) — ad-hoc plot
  that drove the lower-tail rejection.
- `../2026-05-12_2lpt_models_vs_v1_analysis/corr_pca_init_2lpt.png`
  (commit badf0c2) — confirms PCA init is smooth on 2lpt.
- `../2026-05-12_2lpt_models_vs_v1_analysis/corr_pca_init_multi_dataset.png`
  (uncommitted) — extends to LOA-real + Saclay; Saclay panel errored.
- `../2026-05-12_2lpt_models_vs_v1_analysis/corr_Keff_inference_view.png`
  (commit 7dafd7a) — diag-masked off-diagonal-only view, surfaces the
  ridge structure that should be physical emission/absorption features.
- `../2026-05-12_training_pipeline_audit_vs_matlab/findings.md` —
  confirms the loss math is MATLAB-faithful (no bug there).
- `dataset.py:163-184` (commit 3e76056) — the lower-tail rejection.

## Cross-reference

- Memory `project_corrected_retrains_regression_2026_05_06` for the
  earlier (May 6) round of corr-noise hunting that led to the PCA-init
  rewrite.
