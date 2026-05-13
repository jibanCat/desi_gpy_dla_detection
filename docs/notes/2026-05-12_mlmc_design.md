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

---

## Literature review and refined plan (2026-05-12)

> **Provenance note.** WebSearch was operational for this revision (unlike the sibling MAP+LR session that had to fall back to training knowledge); the family-level summaries below were cross-checked against fresh arXiv/journal links. Where I report a specific *number* (compute scaling, variance ratio, etc.) and could not verify it on a primary source within budget, it is flagged `[memory]`. Method recommendations integrate the MAP+LR agent's `2026-05-12_map_lr_failure.md` Literature section — in particular its Marginal-MAP (MMAP) finding and the proposed `Var[Δ_marg]`-across-seeds diagnostic.
>
> **Bottom line.** Build the diagnostic first (1 day), build MMAP next (1-2 days) — *both* are stepping stones that make the eventual MLMC/AIS design measurably better-targeted. Then commit to **pocoMC** (Preconditioned Monte Carlo: SMC + normalizing-flow preconditioner with built-in bridge-sampling evidence) as the primary algorithm, with **two-level MLMC-as-AIS-around-MAP** (the original design above) as the backup. **Revised total scope: 2-4 weeks** (was 1-2), most of which is the validation grid, not the implementation.

### 1. Family-level summary

**(A) Quasi-Monte Carlo + importance sampling (QMC-IS).** Owen & Zhou's "deterministic multiple mixture" + randomized lattice rules. Liu et al. 2024 (arXiv:2403.11374) proved that *with a proper IS proposal* a randomly-shifted rank-1 lattice rule achieves O(N⁻¹) error **uniformly in the noise level / posterior concentration** — i.e. the narrow-peak pathology disappears. Without IS, QMC over the prior degrades catastrophically as the posterior concentrates. This is the formal version of our prior-dilution memory note. Conclusion: **QMC alone is the wrong tool; QMC-IS is the right tool**.

**(B) Adaptive Importance Sampling (AIS family).**
- *Population Monte Carlo* (Cappé+2004): iterative IS where the proposal is re-fit each round from weighted samples.
- *AMIS* (Cornuet+2012, Scand. J. Stat. 39, 798; arXiv:0907.1254): like PMC but **re-weights all past samples each iteration via Owen-Zhou multiple mixture** — substantial ESS improvement on banana-shape and multimodal targets.
- *Defensive mixture* (Hesterberg 1995, Technometrics 37, 185): `q = 0.5 q_narrow + 0.5 q_wide` — bounds the importance weights from above, makes IS robust to proposal misspecification. **Critical for our problem**: a too-narrow proposal around the wrong MAP gives weights that blow up; defensive mixture is the standard protection.

**(C) Annealed / Sequential Importance Sampling.**
- *Annealed IS* (Neal 2001, arXiv:physics/9803008): bridge prior → posterior via tempered sequence `π_β ∝ prior · likelihood^β`. Gives an unbiased evidence estimator as a by-product. Robust to multi-modality (early stages walk freely; late stages localize).
- *SMC samplers* (Del Moral, Doucet, Jasra 2006, JRSS B 68, 411): population version of AIS with adaptive temperature ladder and resampling. **Evidence falls out of the algorithm for free** as the product of inter-temperature normalization ratios. This is the modern workhorse.
- *Adaptive AIS* (Goshtasbpour+2023, ICML): constant-rate-progress schedules dramatically reduce wasted iterations.

**(D) Nested sampling.**
- *Skilling 2004/2006* + implementations: MultiNest (Feroz+2009 arXiv:0809.3437), PolyChord (Handley+2015), **dynesty** (Speagle 2020 arXiv:1904.02180).
- dynesty's `'multi'` bound (multi-ellipsoidal decomposition) is the standard tool for multi-modal targets; default `nlive=500-1000`. Cost on a 2-D unimodal problem with sharp peak: typically `ncall ~ 30 × nlive × log(prior/posterior volume ratio)` `[memory]`. For our 2-D box covering ~5×3 with peaks of width 10⁻³ × 10⁻²:  `log(vol ratio) ~ log(15 / 10⁻⁵) ~ 14`, so `ncall ~ 30 × 500 × 14 ≈ 200k` likelihoods per spec. **This is ~10× more expensive than our current QMC budget** (5k–50k) — not viable for a 1M-QSO run without major engineering. Saving grace: nested sampling on a per-spec basis is **embarrassingly parallel** at the QSO level; cost concern is total node-hours, not wall-clock.

**(E) Multilevel Monte Carlo (Giles 2008, 2015 Acta Num. arXiv:1410.5847).** Telescoping `E[P] = E[P_0] + Σ_l E[P_l − P_{l−1}]`. **Crucial caveat for our problem**: classical MLMC was designed around *discretization-level hierarchies* (coarse PDE solver → fine PDE solver), where adjacent levels are *coupled* by sharing input randomness. Our problem has no PDE; the analogue is *sample-budget* hierarchy (coarse QMC → fine IS-around-MAP). The proper umbrella for that variant is **Multi-Index SMC** / **MLSMC** (Beskos+2017 arXiv:1709.09763, Latz+2018, Jasra+2023). The two-level design in §"MLMC vs simple adaptive IS" above is closer to a Chib-style estimator with a coarse-correction term than to canonical Giles-MLMC — and that's fine, but call it that.

**(F) Variational + normalizing flows.** Standard VI gives an ELBO **lower bound** on log evidence — not the evidence itself. **pocoMC** (Karamanis+2022 arXiv:2207.05660) finesses this: it uses a normalizing flow as a *preconditioner* inside an SMC sampler, then uses a bridge-sampling estimator over flow-transformed coordinates to get an unbiased evidence. Reports 1–2 orders of magnitude speedup vs nested sampling on multi-modal targets `[memory: claimed in their paper]`. **Stands out as a candidate** because (i) it produces the *unbiased* evidence we need, (ii) it natively handles multi-modal targets via the flow + SMC combination, (iii) it's a working pip-installable library, (iv) it's authored by an astronomy/cosmology group so the API and defaults are aligned with our use case.

**(G) Bridge / stepping-stone sampling.** Meng & Wong 1996 (Stat. Sin. 6, 831); Xie+2011 (Syst. Biol. 60, 150). The geometric bridge `q_β ∝ q₀^(1−β) q₁^β` gives a low-variance evidence estimator if you can sample both ends. In our context: between a coarse Gaussian-around-MAP proposal and the true posterior. **This is what pocoMC's evidence estimator actually does internally** post-flow-transformation.

### 2. Domain precedents

| Paper | Method | Notes |
|---|---|---|
| Garnett 2017 (arXiv:1605.04460) | **QMC over (z, logN_HI) prior** | Same estimator as our pipeline; explicitly chose marginal-evidence for Occam protection |
| Ho, Bird & Garnett 2020 (arXiv:2003.11036) | Same; multi-DLA greedy recursion | §3.3 cited in `bayesian_model_selection.py:41` |
| Hobson & McLachlan 2003 (MNRAS 338, 765) | **Marginal-evidence matched filter** for source detection (CMB/X-ray) | Closest published precedent to the MMAP idea; integrates over flux, optimizes over position |
| Feroz+2009 (MultiNest) | Importance Nested Sampling | Astronomy go-to before dynesty |
| Karamanis+2022 (pocoMC, arXiv:2207.05660) | SMC + normalizing flow | Astronomy/cosmology framing; pip-installable |

**No published GP-DLA paper has applied MLMC, AIS, or nested sampling to spectroscopic absorption detection.** We would be first; this is publishable methodology in its own right.

### 3. Comparison table

| Method | Bayesian-consistent | Narrow-peak efficient | Mode-aware | Open-source Python | Fit score (1-5) |
|---|---|---|---|---|---|
| Current: prior QMC | yes | NO | weak (top-K sample-wise) | in-repo | 2 — what we have |
| MAP + LR | NO (test statistic) | n/a | weak | in-repo | 1 — refuted |
| MAP + Laplace | approx (Gaussian) | partial | NO (unimodal) | in-repo | 1 — refuted |
| **MMAP** (integrate logNHI, scan z) | **yes** (marginal evidence) | **yes** (1D inner integral O(50) cheaper than 2D QMC) | partial (multi-peak in z survives) | in-repo + scipy | **4 — stepping stone** |
| Two-level AIS-around-MAP (current note's design) | yes | yes | yes (mixture proposal over top-K modes) | needs implementation | 4 |
| AMIS (Cornuet+2012) | yes | yes | yes | partial (no canonical lib; rewrite) | 3 |
| Annealed IS (Neal 2001) | yes | yes | yes (built-in via temperature) | available in pymc/blackjax | 3 |
| **pocoMC (SMC + NF)** | **yes** (bridge-sampling evidence) | **yes** | **yes** (flow handles geometry) | **yes** (pip install pocomc) | **5 — primary recommendation** |
| dynesty (nested sampling) | yes | yes (in principle) | yes | yes (pip install dynesty) | 3 — too expensive at scale |
| MultiNest / PolyChord | yes | yes | yes | yes | 3 — Fortran wrap pain |
| Full MLMC (Giles) | yes | n/a (designed for PDE) | n/a | not for our problem class | 2 — wrong abstraction |
| MLSMC / Multi-index SMC | yes | yes | yes | research code | 2 — too research-y for now |
| VI / ELBO | LOWER BOUND only | partial | poor | yes | 2 — wrong target |

### 4. Refined design recommendation

**Primary algorithm: pocoMC.**
- Library: `pip install pocomc` (Karamanis+2022 arXiv:2207.05660; docs https://pocomc.readthedocs.io).
- API: pass `log_prior`, `log_likelihood`, prior `bounds`. Returns posterior samples *and* `log_evidence` *and* its uncertainty.
- For our 2-D (z, log N_HI) problem with `n_effective ~ 500` and `n_total ~ 4000` (defaults), expect ~`10k-30k` likelihood evals per spec `[memory; verify with a single-spec smoke test]`. That's 2-6× the current QMC budget — within the "2-3× acceptable" envelope and 5-10× cheaper than dynesty.
- Bridge-sampling evidence (Meng & Wong 1996; pocoMC paper §3.3) is the right estimator for our concentrated-posterior case.
- Multi-DLA: pocoMC handles it natively for k=2 (4-D) and k=3 (6-D); current GP-DLA recursion stays unchanged.

**Backup: two-level "MLMC-as-AIS-around-MAP"** (the original design in this note).
- Implementation already specced above; ~1-2 weeks.
- Use as fallback if pocoMC's per-spec cost turns out too high on real London spectra, OR if pocoMC's evidence agrees with prior QMC on the easy cases but diverges on the hard ones (would suggest implementation bug in pocoMC's bridge-sampler at our regime, and a hand-rolled estimator becomes safer).
- The level-1 proposal should be a **defensive mixture** `q = 0.5 N(θ_MAP, c·H⁻¹) + 0.5 prior` per Hesterberg 1995 — protects against the case where the MAP is in the wrong basin.

### 5. Addressing MMAP and the diagnostic-first option

The MAP+LR agent identified two priors-to-MLMC that should be re-evaluated against the pocoMC recommendation:

**Option A (diagnostic-first, ~2-4 hours):** Compute `Var[Δ_marg(θ_spec; seed) | N_QMC]` for `N ∈ {1k, 5k, 10k, 50k, 200k}` across 4 QMC seeds, on the 1683 SNR>2 nulls and 309 truth-positives. **This is what the MAP+LR agent recommended as the next experiment**, and I agree it should be **done first** before any production code. Why:

1. If `Var[Δ_marg | N=5k]` is comparable to the signal–null separation, we are **QMC-noise-limited** → IS/MLMC/pocoMC will straightforwardly help. Expected outcome.
2. If `Var[Δ_marg | N=5k]` is already small compared to the separation, we are **sufficient-statistic-limited** → no amount of sampler engineering helps; the answer lies in the model (Voigt profile, GP kernel, mean-flux prior, BAL handling). This would be a much bigger pivot.
3. Either way, the variance curve calibrates the *target* of any subsequent algorithm: if you need 100× variance reduction at fixed N, that's a different design than 5×.

**Cost**: this is just re-running the existing QMC pipeline at multiple N values on the same spectra — ~2-4 hours on the existing London 8f catalog. Almost free.

**Option B (MMAP as stepping stone, ~1-2 days).** `argmax_z ∫ p(D|z, logN_HI) p(logN_HI) dlogN_HI` evaluated by a 1D inner QMC + a coarse 1D outer scan over z. Compute: ~50× a single likelihood per spec, vs ~5k for current 2D QMC, so **~100× cheaper** than the existing pipeline `[memory: rough scaling]`. The MAP+LR agent flagged this as 5× MAP+LR cost — that's correct if you compare to MAP, but compared to *current QMC* it's actually cheaper. Hobson & McLachlan 2003 is the published precedent.

Recommended sequencing: **A → B → primary (pocoMC) build, backup (level-1 AIS) ready in parallel.**

| Stage | Time | Deliverable |
|---|---|---|
| A. Variance diagnostic | 0.5-1 day | Plot: `Var[Δ_marg]` vs N vs signal-null separation. Decide: sampling-limited (proceed) or model-limited (pivot). |
| B. MMAP prototype | 1-2 days | Per-spec MMAP catalog on London 8f. Re-evaluate P/C at SNR>2. **If MMAP alone hits 85/85, MLMC/pocoMC may be unnecessary for the headline paper.** |
| C. pocoMC integration | 3-5 days | `gpy_dla_detection/pocomc_inference.py` + CLI flag `--enable_pocomc 1`. Single-spec smoke test, then 5k London validation. |
| D. AIS-around-MAP backup | 3-5 days (parallel to C) | The original design in §"Engineering scope" above. Implement to a working prototype; benchmark vs pocoMC. |
| E. Production validation | 1 week | 5k London + 5k Saclay parity, BAL-incl/excl, P/C at SNR>{1, 2, 6}, cost budget |

**Total: 2-4 weeks** (was 1-2), but with two important changes vs the original plan:
- **Stage A is gating**: if it shows we're not sampling-limited, C-D are skipped and the work pivots to model changes.
- **Stage B is independently valuable**: MMAP yields an Occam-protected detector with no new sampler, in 1-2 days. This may close enough of the 85/85 gap to ship the headline paper without C-D, which would push C-D out to a follow-up paper.

### 6. Pitfalls to design around

1. **Importance-weight degeneracy**. If the level-1 proposal misses the true peak (wrong MAP, wrong width), weights blow up at the few good samples and ESS collapses to ~1. Standard mitigations: defensive mixture (Hesterberg 1995), Pareto-smoothed IS diagnostic (Vehtari+2024 `psis` package; flags ESS issues automatically), truncated weights.

2. **Multi-modality + flow training instability**. pocoMC trains a normalizing flow at each SMC step. On low-SNR spectra with weak peaks, the flow may overfit a noise mode. Mitigation: pocoMC's `n_effective` and `flow_lr` defaults are tuned for cosmology problems with smooth posteriors; we should sweep them on a strong-DLA spec first.

3. **Per-spec wall-clock vs cluster wall-clock**. pocoMC's per-spec cost (~10k-30k likelihoods, single-threaded) is bigger than our current 5k QMC. At 1M QSOs, even 3× per-spec is 3× the node-hours. The 17→50 nh budget has only 3× headroom. **The first single-spec benchmark must verify per-spec cost before any catalog-scale work.**

4. **Reproducibility / seeds**. pocoMC is stochastic; our pipeline is currently deterministic-given-QMC-seed. We need a per-spec seed plumbing convention (already present in `generate_samples.py`).

5. **CDDF Pathway A compatibility**. The CDDF Pathway A code consumes `model_posteriors` from HDF5. pocoMC outputs posterior samples + an evidence scalar — both are derivable. **Plan**: store both the evidence array (`log_p_D_M_k` for k=0..3) and the posterior samples in the existing schema; CDDF code reads only the evidence array, unchanged. Posterior samples become available for diagnostics and for future Pathway-B-style analyses.

6. **Boundary / prior-edge MAPs**. Same caveat as Laplace: if the MAP sits at the prior boundary, level-1 IS proposals with infinite support will leak mass outside the prior. Use truncated-Gaussian proposals or include a hard reweight by `1[θ ∈ prior_support]`.

7. **The MMAP integrand for noise-dominated spectra is flat in z**. For a pure-null spec, the marginal-over-logNHI is a slowly-varying function of z (no preferred z). The "MAP z" is then driven by tiny QMC variance — operationally a noise eigenvalue, scientifically meaningless. Mitigation: report MMAP only above an evidence threshold; below threshold use the marginal directly.

### 7. Surprising findings vs the current note

- **The "MLMC" in our title is the wrong umbrella term**. Classical Giles MLMC is for discretization hierarchies, not sample-budget hierarchies. The right name for the two-level design above is "Chib-style estimator with coarse-IS correction" or "multilevel SMC" (MLSMC, Beskos+2017). Cosmetic but matters for paper references.
- **pocoMC exists and matches our requirements almost exactly**. Astronomy-authored, pip-installable, normalizing-flow preconditioner, bridge-sampling evidence, ~10⁴ likelihoods per problem. We should try it before building bespoke MLMC.
- **MMAP could close the gap with minimal engineering**. ~100× cheaper than the existing 2D QMC `[memory; verify]`, Occam-protected via the 1D logNHI integral, and the MAP+LR agent already laid out the recipe. If MMAP gets us to 85/85 at SNR>2, the entire MLMC/pocoMC build moves to the next paper.
- **QMC-IS theory (Liu+2024) directly supports the level-1 AIS-around-MAP design**: with a proper importance proposal, lattice-rule QMC achieves O(N⁻¹) error *uniformly in posterior concentration*. The narrow-peak pathology vanishes the moment we add IS. This is the formal version of our prior-dilution memory note.
- **The "1-2 week MLMC build" estimate is revised to 2-4 weeks total**, but most of that is the validation grid (C+E above), not the algorithm. The diagnostic (A, ~1 day) is gating; the MMAP prototype (B, 1-2 days) is independently valuable; pocoMC integration (C, 3-5 days) is the actual algorithm work.

### 8. Open questions for the next session

1. Does Stage A diagnostic show sampling-limited or model-limited behavior?
2. Does MMAP at Stage B hit 85/85 on SNR>2 BAL-excl per-DLA, removing the need for Stages C-D?
3. pocoMC single-spec smoke test: is the per-spec cost actually within 3× of QMC?
4. Does the sub-DLA model interact cleanly with pocoMC's SMC schedule, or does the tempering interleave with the sub-DLA boundary in pathological ways?

### References (verified via WebSearch this session unless flagged)

Primary:
- Giles (2008), Operations Research 56, 607 — original MLMC.
- Giles (2015), Acta Numerica 24, 259, arXiv:1410.5847 — MLMC review.
- Del Moral, Doucet, Jasra (2006), JRSS B 68, 411 — SMC samplers.
- Neal (2001), Stat. Comput. 11, 125, arXiv:physics/9803008 — annealed IS.
- Karamanis, Beutler, Peacock, Nabergoj, Seljak (2022), arXiv:2207.05660 — pocoMC.
- Speagle (2020), MNRAS 493, 3132, arXiv:1904.02180 — dynesty.
- Skilling (2004, 2006) — nested sampling.
- Cornuet, Marin, Mira, Robert (2012), Scand. J. Stat. 39, 798, arXiv:0907.1254 — AMIS.
- Cappé, Guillin, Marin, Robert (2004), JCGS 13, 907 — Population Monte Carlo.
- Hesterberg (1995), Technometrics 37, 185 — defensive mixture IS.
- Meng & Wong (1996), Stat. Sin. 6, 831 — bridge sampling.
- Xie, Lewis, Liu, Fan (2011), Syst. Biol. 60, 150 — stepping-stone sampling.
- Liu, Wang, Owen, et al. (2024), arXiv:2403.11374 — QMC-IS for concentrated Bayesian posteriors.
- Beskos, Jasra, Law, Marzouk, Zhou (2017), arXiv:1709.09763 — Multilevel SMC² for inverse problems.
- Hobson & McLachlan (2003), MNRAS 338, 765 — Bayesian matched filter (closest precedent to MMAP).

Implementations:
- pocoMC: https://pocomc.readthedocs.io ; https://github.com/minaskar/pocomc
- dynesty: https://dynesty.readthedocs.io ; https://github.com/joshspeagle/dynesty
- BlackJAX SMC: https://blackjax-devs.github.io/blackjax/ (SMC + tempering, JAX-native)
- PyMC SMC: `pm.sample_smc()` — full SMC sampler with evidence as by-product
- MultiNest (Python wrapper PyMultiNest), PolyChord — older but well-tested

Cross-references in our repo:
- `docs/notes/2026-05-12_map_lr_failure.md` "Literature review and refined diagnostic plan" — sibling literature review (MAP+LR family) that recommended the diagnostic-first experiment.
- `docs/notes/2026-04-29_bayesian_correctness_synthesis.md` — earlier hypothesis ledger.
- Memory: `project_prior_dilution_finding.md`.
