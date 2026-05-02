# Critical finding: v2 trainer produces over-fit / mode-collapsed GPs

> **Status**: Empirical finding flagged 2026-05-02 by the user (Q1+Q2+Q3
> session). All 4 v2-trained models tested fail Mahalanobis χ²
> calibration vs their own trainsets. v2 trainer is NOT production-ready.

## The user's observation (corrected interpretation)

User: "v1's correlation matrix has rich emission-absorption features
— Lyα and Lyβ, metals + emission off-diagonal — that have clear
physical meaning. v2 correlation matrices are just sharp squares
without physical interpretation."

This was the correct read. My earlier framing ("v2 has higher
off-diagonal density") missed the point: density isn't structure.
A K dominated by ONE collapsed eigenvector has high overall
correlation density but no meaningful per-line covariance.

## Three lines of evidence (all consistent)

### Evidence 1: K decomposition

| Model | trace ω² / trace(K) | top eig / 2nd eig | effective rank | interpretation |
|---|---:|---:|---:|---|
| v1 | **0.843** | 3.3× | 1.95 | ω²-dominated, broad basis |
| LOA-noHCD-withBAL | 0.034 | 1.3× | 4.03 | M·M^T-dominated, semi-flat eigvals |
| LOA-noDLA-noBAL-norm1280 | 0.004 | 4.0× | 1.30 | M·M^T-dominated, top mode dominant |
| LOA-noDLA-noBAL-y1off | 0.020 | 4.8× | 1.70 | same |
| **2lpt-mock0** | **0.002** | **500×** | **1.00** | **ONE eigenvector explains everything** |
| 2lpt-loa124 | 0.018 | 1.2× | 2.02 | same |
| saclay-mock0 | 0.013 | 11× | 1.28 | same |

v1 splits 16% of K diagonal into M·M^T and 84% into ω². This is the
classical GP regression regime: model captures the smooth physical
modes, ω² absorbs the per-pixel noise.

v2 models put 97-99.6% of K diagonal into M·M^T. Combined with
effective rank ≈ 1-2, this means M·M^T is essentially a rank-1
or rank-2 matrix that captures one or two dominant modes (likely
"per-spectrum scale factor"), with negligible residual variance
budget for everything else.

**The visual "sharp squares" the user sees in the correlation matrix
are the rank-1 M M^T pattern leaking through everywhere, not real
emission-line covariances.**

### Evidence 2: Mahalanobis χ² calibration

For each model, I evaluated χ² = (y - μ)^T K^-1 (y - μ) on 500
random training spectra:

| Model | χ²/n_valid mean | χ² z-score (target N(0,1)) | per-pixel std_resid (target ~1) |
|---|---:|---:|---:|
| LOA-noHCD-withBAL | 0.02 | mean=-33.3, std=4.95 | 0.07 |
| LOA-noDLA-noBAL-y1off | 0.02 | mean=-34.1, std=7.57 | 0.09 |
| 2lpt-mock0 | 1.00 (mean) | mean=-0.08, std=**116** | 0.06 |
| saclay-mock0 | 0.88 | mean=-4.2, std=**101** | 0.04 |

All FAIL. Two distinct failure modes:

- **LOA-trained**: χ²/n is universally ~0.02 — the model's predicted
  σ is ~7× too large at every pixel. Per-pixel residuals are 0.07-0.09
  in std vs target 1. The model thinks the data has 100× more variance
  than it actually does.

- **2lpt/saclay-trained**: χ²/n averages around 1 but with std=100+.
  This is bimodal: most spectra get tiny χ² (well-fit), a minority get
  enormous χ² (mode-collapsed mode doesn't fit them). The "average is
  fine" illusion masks the failure.

A well-calibrated GP should give χ²/n ≈ 1 with std ≈ √(2/n) ≈ 0.023
(small spread, since χ² has variance 2n). Observed std of 100+ is
**4 orders of magnitude worse** than expected.

### Evidence 3: Canonical TID 120046865 missed detections

From `2026-05-02_v2_canonical_tid_comparison.md`: the LOA-trained
models with the worst calibration miss the truth DLA at log_NHI=21.26
entirely (p_dla=0.037 and 0.136). The over-fit M·M^T can absorb DLA
absorption as a "small variance excursion in the dominant mode" —
the Bayes factor between Null and DLA is muted because K^-1 (DLA
residual) is tiny when K diag is dominated by inter-spectrum mode
variance.

## Diagnosis: probable root causes

The v2 trainer (`gpy_dla_detection/training/`) is a from-scratch
rewrite from v1 (`gpy_dla_detection/objective.py` +
`learn_qso_model.py`). Several differences could be at fault.

### Suspect 1 — Wrong split between M and ω at convergence

In v1, the optimizer finds an equilibrium where most variance gets
attributed to ω² (84%) because the L2 regularization on M parameters
implicitly constrains its magnitude. v2's Adam optimizer has
`weight_decay=0.0` (TrainConfig default) — no L2 regularization.

→ M can grow unbounded, ω can shrink to ~0, both happen.

### Suspect 2 — Cosine LR schedule too aggressive

v2 uses `cosine_t_max=50` with `cosine_eta_min=1e-5`. This anneals
the LR to near-zero before convergence on a non-trivial loss
surface. The trainer ends up "stuck" at a local optimum with one
dominant mode + small ω².

### Suspect 3 — Initialization

`init_M = randn * 0.05` (small random). With Adam + cosine LR, the
first few epochs aggressively grow M to fit the training data (which
has a dominant scale-factor mode), then the LR decays before M can
diversify into multiple modes.

### Suspect 4 — Different loss function vs v1

The vectorized NLL (`objective_v2.py`) is supposed to match the
legacy `spectrum_loss` (verified by `test_objective_v2_parity.py`).
But the parity tests run on small synthetic batches; the actual
trainer behavior on real 300k-spectrum trainsets could differ if
there's an aggregation bug.

## Recommendations

### Immediate (before any v2 model promotion):

1. **DO NOT promote any v2 model to production.** Including
   `loa_no_hcd_with_bal_normalized` (which we recommended earlier
   for the 50k campaign). All v2 models fail calibration.

2. **Run the same calibration check on v1's training spectra.** I
   can't compare directly because there's no v1 trainset.h5 on disk
   — only the model. If we can dump the v1-equivalent trainset
   (preloaded LOA spectra used for v1 training), we can confirm v1
   passes calibration on its own data.

3. **Pause the 50k mock-vs-LOA campaign plan.** Comparing two
   uncalibrated models doesn't tell us anything useful.

### Investigation steps (next session):

4. **Add weight decay to v2 trainer.** `TrainConfig.weight_decay=1e-6`
   is a cheap test. Re-train one model, compare K decomposition.

5. **Disable cosine scheduler.** Train with `scheduler="none"` for
   1500 epochs, see if M·M^T diversifies.

6. **Check the loss curve.** `loss_history.json` exists for all v2
   runs — plot it for v1 (if available) and v2 to see whether v2
   converged to a local optimum vs continued descent.

7. **Compare v2 vs v1 training data preprocessing in detail.**
   The fix for the missing per-spectrum normalization (PR #5) should
   have made v2 → v1 equivalent at the data level. Verify:
   - Same masking
   - Same normalization (median in [1310, 1325] vs v1's [1425, 1475] —
     this was the noted difference; could it cause ω collapse?)
   - Same de-forest (Turner+2024 in both)
   - Same centering (inverse-variance-weighted in both)

8. **Re-train ONE model with v1-equivalent settings.** Same
   normalization window [1425, 1475] (if the v2 trainset rest grid
   extends that far — it doesn't, ends at 1421 Å), same scheduler,
   same weight decay. Compare.

### Tier-1 #1 design impact:

The sub-DLA / DLA prior fix was scoped against working models. With
all v2 models failing calibration, the prior fix can't be validated
against them. **Recommend deferring Tier 1 #1 implementation until a
calibrated v2 model exists.**

## What this means for PR #5

PR #5 (already merged) shipped:
- τ-EB recipe ✓ (independent of v2 training; tunes runtime mean-flux)
- v2 trainer normalization fix ✓ (correctly addresses the bug found
  empirically; v2 trainsets ARE now per-spectrum normalized)
- ABCD other things ✓

But PR #5 did NOT make any claim about v2 trained models being
production-ready. The `2026-05-01_trained_gp_models_comparison.md`
doc that ships with PR #5 lists trained-model hyperparameters but
doesn't claim calibration. So PR #5 itself is not invalidated — what
this finding invalidates is the path forward of "promote v2 model to
production after retraining."

The right next step is investigation (per items 4-8 above), not
production rollout.

## Artifacts

- `examples/check_v2_model_calibration.py` — the calibration script
- `docs/notes/2026-05-02_calibration_*.png` — per-model figures
  (4 figures, one per model checked)
- `examples/plot_v2_model_diagnostics.py` — per-model μ/ω/K viz
- `docs/notes/2026-05-02_v2_model_diagnostics.png` — original side-
  by-side figure (the "sharp squares" the user noticed)
- `examples/plot_v2_model_mu_detailed.py` — μ-only per-model viz with
  diff vs baseline
- `docs/notes/2026-05-02_v2_mu_detailed.png` — same
