# Multilevel adaptive Monte Carlo for narrow-peak DLA detection

> **2026-05-12.** Discussion note for future iteration. Context: the prior-dilution memory note (`project_prior_dilution_finding.md`) and the MAP+LR failure note (`2026-05-12_map_lr_failure.md`). Captures the #6/#7 design discussion so we can resume later.

## The Bayesian-purist tension

The current production pipeline is end-to-end Bayesian:

```
p(D | 1 DLA) = ∫ p(D | z, NHI) p(z, NHI) dz dNHI
```

evaluated via QMC over a uniform prior in (z, NHI). CDDF Pathway A propagates the full `model_posteriors` through Poisson-binomial uncertainty.

The narrow-peak problem (peaks <±0.001 z × <±0.01 log NHI vs a prior covering ±0.5 z × ±3 log NHI) means uniform QMC samples 99% empty space. Single-point evaluation at truth gives Δ_at_truth = +5 to +10 logL, but the marginal is −3 to −8 logL — a 10-18 logL gap explained by the prior-volume Occam factor.

Three families of fixes:

| Approach | Bayesian? | Preserves CDDF Pathway A? | Engineering cost |
|---|---|---|---|
| MAP + LR | No (test statistic, not posterior) | No (no posterior) | Low |
| MAP + Laplace correction | Approximate (Gaussian-around-MAP) | Partial (Gaussian posterior approx) | Medium |
| Adaptive importance sampling | Yes | Yes | Medium |
| Multilevel Monte Carlo (MLMC) | Yes | Yes | Medium-high |

## Why MAP + LR fails

`log_LR = log p(D|MAP) − log p(D|null)` is a test statistic, not a posterior probability. As a detection criterion it has three serious problems:

1. **No Occam volume penalty.** A narrow MAP fit and a broad MAP fit get the same `log_LR`. The Bayesian framework correctly penalizes narrow fits via the `−½ log|H|` term that emerges from the saddle-point integral around the MAP. **This is the dominant mechanism for the logN ≈ 20.5 ghost detections** — those peaks are real (noise-shaped like damped wings) but vanishingly narrow against the prior, so the marginal sees a tiny integral. MAP+LR misses this dilution entirely.
2. **Doesn't include competing hypotheses.** Production `p_DLA` includes sub-DLA evidence in the denominator. MAP-LR doesn't. Affects detections in the overlap region [19, 20.3] where sub-DLA and DLA both fit, but is *not* the mechanism for the logN ≈ 20.5 ghost detections (those live outside sub-DLA range).
3. **Breaks downstream CDDF.** No proper posterior over NHI per detection means f(N,z) loses uncertainty propagation. Pathway A (the Bayesian CDDF, current default) cannot consume MAP-only catalogs.

Measured this 2026-05-12: Method B (MAP+LR) had P=20-29% standalone, contaminated by the ghost-DLA tail.

## Why MAP + Laplace partially helps

The saddle-point approximation of the evidence:

```
log p(D | 1 DLA) ≈ log p(D|θ_MAP) + (d/2) log(2π) − ½ log|H| + log p(θ_MAP)
```

Adds back the Occam factor and the prior. The earlier MAP-detection prototype on n=48 nulls showed this drops the null FP rate from ~30% to ~22%. Better, but still worse than the prior-marginal score (Method A), per the prior session's investigation log.

The Gaussian-around-MAP assumption breaks at prior boundaries and in multimodal posteriors — exactly where the ghost-DLA pathology lives. So Laplace is a useful diagnostic but not a deployment-quality fix.

## The right Bayesian fix: adaptive importance sampling

Reframe QMC as importance sampling:

```
p(D | 1 DLA) ≈ (1/N) Σ_i [p(D|θ_i) · p(θ_i)] / q(θ_i),     θ_i ~ q
```

where `q` is an informed proposal centered on a MAP seed (or top-K MAP modes). This is the **same target integral** as the current pipeline — the prior is unchanged, only the sampling distribution changes. The estimator is consistent (N → ∞ recovers the true evidence), unbiased, and preserves the entire downstream CDDF math.

**Concrete scheme**:
1. Level 0: cheap coarse QMC over the full prior (e.g., 5k samples). Identify candidate peaks via top-K of `log_likelihoods_dla` sample-wise.
2. For each candidate peak, propose `q_k = N(θ_MAP, c · H⁻¹)` with `c ∈ [1, 4]` for fattened tails.
3. Level 1: draw `N_1` samples from a mixture `q = Σ_k w_k q_k`, compute weighted log-likelihoods. Combine with level 0 via the telescoping sum estimator.

For narrow-peak problems the variance reduction at fixed N is 10-100×.

**Why this beats MAP-Laplace**:
- No Gaussian-around-MAP assumption; the actual likelihood is evaluated at sampled points.
- Multi-modal posteriors handled naturally (mixture proposal).
- Final number IS the evidence — no calibration of an LR threshold needed.

## MLMC vs simple adaptive IS

MLMC is the natural generalization: an indexed family of estimators at increasing resolution, combined via a telescoping sum to get variance reduction proportional to *between-level* variance, not total variance. Two levels (coarse uniform + fine importance-sampled near MAP) is the minimum useful case and what I'd recommend prototyping first.

True multi-level (3+ levels, e.g., coarse → mode-locating → mode-refining) is overkill for our 2D (z, NHI) problem per absorber. For k-DLA models with k > 1, the 2k-D space might justify it.

## Engineering scope to ship

Rough 1-2 week estimate:

1. **Module**: `gpy_dla_detection/importance_sampling.py`
   - `propose_around_map(sample_log_likelihoods, top_k=3, hessian_scale=2.0) → proposal mixture`
   - `weighted_log_evidence(samples_l0, samples_l1, weights_l1) → log p(D)`
2. **Hook into `dla_gp.py`**: after level-0 QMC, identify top-K modes, draw level-1 samples, combine. Optional, gated by `--enable_adaptive_is`.
3. **h5 schema additions**: store level-1 samples + weights alongside existing level-0. Backward compatible (new fields, old readers unaffected).
4. **`bayesian_model_selection.py`**: minor — combine level-0 + level-1 estimates via telescoping sum; output `log p(D | M_k)` unchanged in interface.
5. **CDDF Pathway A**: should work unchanged. `model_posteriors` has the same shape; just lower variance.
6. **Validation**:
   - Reproduce baseline on 5 known-missed candidates: do they now have `Δ_marg` > 0?
   - 5k London 8f run: P/C at SNR > 2 vs baseline. Target the 85/85 line.
   - Cost overhead: should be <2× baseline (level-1 adds N_1 = 1000-2000 likelihood evals per spec).

## #7 — production-catalog ghost-DLA audit

Independent of the MLMC plan. Worth doing as a data quality check on the current production catalog.

**Hypothesis**: at strict P_DLA cuts (≥0.999, ≥0.99999) some fraction of "high-confidence" detections are noise-overfit broad-weak DLAs at logN ≈ 20.5 (the same population that contaminated Method B). The marginal score doesn't always wash these out because:
- Some spectra have noise structures that genuinely peak narrowly enough to survive the prior-dilution penalty.
- The sub-DLA model's NHI range [19.5, 20.3] only partially overlaps the ghost regime — at logN > 20.3 the DLA model has no Bayesian competitor.

**Cheap test** on the existing v3_loa124 catalog:
1. Cross-reference each high-P detection's `MAP_log_nhis` and `MAP_z_dlas`.
2. Flag rows where MAP logN ∈ [20.3, 21] AND z_MAP is far from any high-NHI truth match.
3. Truth-match against `dla_cat.fits`. Is the FP rate higher in the flagged subset than the body of the catalog?

If yes → ghost pathology is real in production. Mitigation options:
- Extend DLA model to compete with itself at narrow-peak/broad-peak (a "peak width" hyperparameter).
- MLMC-with-stratification: explicitly stratify QMC by NHI to under-sample [20.3, 20.7] where noise overfits live.
- Train a narrow-peak detector and use it as a veto post-hoc.

## Sub-DLA model interaction with extended NHI prior — open question

Actual ranges in production (verified 2026-05-12 from samples file + code):
```
Sub-DLA: log NHI ∈ [19.1, 20.0]  (per subdla_samples_a03_191_200_100000.mat filename)
DLA:     log NHI ∈ [19.0, 22.0]  (PW14 prior, pw_samples_a3_190_220_50000.mat)
                                  ^^^ overlap [19.0, 20.0] — both models can fit here
```
Note: docstrings in `process_helpers.py` and `subdla_samples.py` say "[19, 20.3]" — stale, should be fixed.

**Critical**: SubDLA counts as **no DLA** in the catalog, not as a DLA detection.
```
p_dla     = P(1DLA) + P(2DLA) + ... (only the DLA(k) columns)
p_no_dla  = 1 - p_dla = P(Null) + P(SubDLA)
```
(verified in `bayesian_model_selection.py:240-275`)

The two models overlap in [19, 20.3] — the marginal correctly splits mass between them in that band. This is design-intentional (sub-DLA is a "narrow" version of DLA at lower NHI) but it's bookkeeping overhead.

**Possible simplification**: drop sub-DLA, extend DLA NHI prior to [17, 22] or [19, 23], so a single DLA model spans the full NHI range. Method B implicitly already did this with its [17, 22] optimizer.

What changes:
- The prior shape used in the CDDF integral (extended NHI support → re-derive p(N|D) normalizations).
- The interpretation of "DLA detection" vs "sub-DLA detection" downstream — relabelling, not new physics.
- The f(N, z) binning at the low-NHI end.

**Important**: this simplification does *not* address the logN ≈ 20.5 ghost-DLA problem (false positives), which lives outside the sub-DLA range. The Occam volume penalty (Laplace correction or MLMC marginalization) is the fix for that. The sub-DLA cleanup and the ghost fix are orthogonal.

**But** the simplification DOES directly address a separate structural source of **low-SNR weak-DLA misses** (completeness ceiling):
- A real DLA at logN_truth = 20.4 in a low-SNR spectrum has its NHI posterior smeared toward lower NHI (forest noise looks like weaker absorption).
- If the smear lands in [19.1, 20.0], SubDLA and weak-DLA evidences are comparable.
- `P(1DLA)` and `P(SubDLA)` split mass; `p_DLA = P(1DLA) + P(2DLA) + ...` falls below threshold.
- The detection goes to `p_no_dla` (which includes SubDLA) — counted as a miss.
- Dropping SubDLA + extending DLA prior would send these to DLA → counted as positives → direct completeness boost at the low-SNR end.

**Cheap test** (no new inference): on the existing v3_loa124 combined catalog, compute `p_DLA_unified = model_posteriors[1] + model_posteriors[2:].sum()` (SubDLA + all DLA cols) and re-evaluate P/C. ~10 min script.

**Open question**: is the right move to (a) keep sub-DLA + add MLMC, or (b) drop sub-DLA + add MLMC + extended prior? (a) is incremental; (b) is a structural simplification that better matches the underlying physics (DLA and sub-DLA differ only in NHI; same absorber population) AND gives a free completeness boost at low NHI, but requires re-deriving downstream f(N, z) normalizations and re-running calibration.

## Decision points to revisit

1. **Headline strategy** for the immediate paper:
   - **A.** Ship v3_loa124 + baseline P_DLA cut. Saclay generalization confirmed (87.07/77.10). Fully Bayesian, clean PR. Classical DLAs sit at 84.6/83.5 — close to 85/85.
   - **B.** Pause headline, build MLMC (1-2 weeks), push for 85/85 strict.
2. **Sub-DLA structural question**: keep separate model. **Resolved** — DLA-prior extension is worse than the dedicated SubDLA model (see Two-target validation below).
3. **#7 audit**: run now (cheap re-analysis) or defer until MLMC built?

## Two-target validation (2026-05-12 measurement)

The end-state catalog has two science targets with different P/C requirements:

### Classical DLA target (NHI ≥ 20.3): aim P > 85-90%, C > 85%
Measured on London 8f v3_loa124, per-DLA, BAL-excl, SNR>2 (lya_lyb window):

| Cut | Score | P | C |
|---|---|---:|---:|
| ≥ 0.99 | baseline | **84.6%** | **83.5%** |
| ≥ 0.99 | unified | 81.5% | 86.3% |
| ≥ 0.99999 | baseline | 88.5% | 74.2% |

**Status**: baseline is ~0.5pp short on both axes at ≥0.99. Unified score gets above C target but drops below P target. MLMC could plausibly close the last 0.5-2pp at ≥0.99 by suppressing the narrow-peak Occam dilution that currently kills marginal weak-DLA detections in low-SNR spectra.

### Sub-DLA target (NHI ∈ [19.1, 20.0)): aim P > 85%, C > 70%
Measured per-spec via P(SubDLA|D); n_truth_pure_subdla = 384:

| Cut | n_pred | TP | P | C |
|---|---:|---:|---:|---:|
| ≥ 0.50 | 368 | 217 | 59.0% | **56.5%** |
| ≥ 0.90 | 133 | 93 | 70.0% | 24.2% |
| ≥ 0.99 | 18 | 14 | **77.8%** | 3.7% |

For reference, **DLA-model MAP in [19, 20.3]** (the alternative if we drop SubDLA + extend DLA prior to absorb the sub-DLA range): max P = 19%, max C = 37%. Strictly worse than the dedicated SubDLA model.

**Status**: SubDLA model is far from target — best operating point is 77.8/3.7 or 59/56. **Improving this requires a dedicated effort**:
1. Tighter NHI prior on the SubDLA model (currently [19.1, 20.0] uniform; a peaked prior at the high-frequency NHI band might suppress noise overfits).
2. Better training data curation (loa124-style for SubDLA).
3. MLMC at the SubDLA peak — same principle as classical-DLA MLMC; adaptive importance sampling around the SubDLA model's MAP.
4. Optional post-hoc filter using MAP NHI + truth z-matching to reduce contamination from spectra where a classical DLA is also present but at different z.

## Laplace correction is not sufficient (2026-05-12 measurement)

Per the Laplace agent's run on the full v3_loa124 London 8f:
- Raw MAP+LR: peaks at 74.1% P at p99.7 null-quantile threshold.
- MAP + Laplace: peaks at 74.5% P at p99.9 null-quantile.
- Both 11pp short of the 85% P target.
- Laplace shifts the null distribution down by 8-12 logL (matches the dilution finding's expected Occam magnitude) but shifts the signal distribution by the SAME amount → no improvement in discrimination.

Mechanism: in low-SNR spectra, BOTH noise-fitted ghost peaks AND legitimate weak-DLA peaks are narrow. The Hessian penalty `½ log|H|` is large for both, so the Occam correction doesn't separate signal from noise. The relevant discriminator is in the prior — narrow peaks at z's where a real absorber exists (correlated with Lyα flux structure) vs at z's where they don't (random forest noise). Importance sampling over the proposal `q ∝ likelihood × prior` naturally does this; Laplace correction does not.

**Implication**: the MAP-based detection path is a dead end for the SNR>2 ceiling. **MLMC / adaptive importance sampling is the principled fix and the next investment**. See agent's full report at `/pscratch/sd/j/jibancat/prod533_5k_20260511/laplace_map_test/RESULTS.md`.

## Related notes
- `project_prior_dilution_finding.md` (memory) — the original mechanism
- `2026-05-12_map_lr_failure.md` — the empirical failure of MAP+LR
- `2026-04-29_bayesian_correctness_synthesis.md` — earlier hypothesis ledger
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/laplace_map_test/RESULTS.md` — Laplace correction also fails
- `/pscratch/sd/j/jibancat/prod533_5k_20260511/null_quantile_map_combined/unified_pdla_perdla.json` — two-target P/C measurement
