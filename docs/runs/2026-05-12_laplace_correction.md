# Method B + Laplace volume penalty — RESULTS

Operating point: SNR>2, BAL-excluded, classical-DLA truth (NHI≥20.3), MAP NHI≥19.0.
Null = no truth absorber NHI≥17.2 in [min_z_dla, max_z_dla].

- Total Method-B specs: **3345**
- Hessian PD at MAP:    **3267** (non-PD: 78)
- Null pop (NHI_MAP≥19): raw 377 / Laplace 347 (PD subset)
- Truth DLA denom (specs, SNR>2, non-BAL):  **309**

## Null `log_LR` quantiles (NHI_MAP≥19 subset)

| Quantile | Method B (raw) | Method B + Laplace |
|----------|---------------:|-------------------:|
|      p50 |     +14.03 |      +6.31 |
|      p90 |     +45.21 |     +36.78 |
|      p95 |     +65.29 |     +53.53 |
|      p99 |     +93.21 |     +80.83 |
|    p99.9 |    +145.97 |    +133.34 |

## Null FP rate at each method's own p-quantile (sanity)

Each method calibrated to its own null distribution at p95 → 5% by construction.

| q | Raw thr | n_FP_raw / n_null_raw  | Laplace thr | n_FP_lap / n_null_lap |
|---|--------:|-----------------------:|------------:|----------------------:|
| p90 | +45.21 | 38 / 377 (10.1%) | +36.78 | 35 / 347 (10.1%) |
| p95 | +65.29 | 19 / 377 (5.0%) | +53.53 | 18 / 347 (5.2%) |
| p99 | +93.21 | 4 / 377 (1.1%) | +80.83 | 4 / 347 (1.2%) |

## P/C at p95 (classical DLA, NHI≥20.3, SNR>2)

| Method | Threshold (p95) | n_det | n_tp | Purity | Completeness |
|--------|----------------:|------:|-----:|-------:|-------------:|
| MethodB+Laplace | +53.53 | 407 | 199 | 48.89% | 64.40% |
| MethodB(raw) | +65.29 | 402 | 200 | 49.75% | 64.72% |

## Full sweep across p90 / p95 / p99

| Method | q | Threshold | n_det | n_tp | P% | C% |
|--------|---|----------:|------:|-----:|---:|---:|
| MethodB+Laplace | p90 | +36.78 | 546 | 228 | 41.76 | 73.79 |
| MethodB(raw) | p90 | +45.21 | 563 | 234 | 41.56 | 75.73 |
| MethodB+Laplace | p95 | +53.53 | 407 | 199 | 48.89 | 64.40 |
| MethodB(raw) | p95 | +65.29 | 402 | 200 | 49.75 | 64.72 |
| MethodB+Laplace | p99 | +80.83 | 260 | 153 | 58.85 | 49.51 |
| MethodB(raw) | p99 | +93.21 | 253 | 149 | 58.89 | 48.22 |

## Finer threshold sweep (raw vs Laplace at matched null FP rates)

Each row uses each method's own null-distribution percentile as its threshold,
so the null FP rate is identical across rows by construction. Comparing P/C at
the SAME percentile is the apples-to-apples comparison.

| q | raw thr | raw n_det | raw P% | raw C% | lap thr | lap n_det | lap P% | lap C% |
|---|--------:|----------:|-------:|-------:|--------:|----------:|-------:|-------:|
| 50.0  | +14.03 | 1106 | 26.40 | 94.50 |  +6.31 | 1073 | 26.19 | 90.94 |
| 75.0  | +26.69 |  836 | 32.42 | 87.70 | +18.63 |  803 | 32.63 | 84.79 |
| 85.0  | +37.60 |  661 | 37.82 | 80.91 | +27.23 |  659 | 37.03 | 78.96 |
| 90.0  | +45.21 |  563 | 41.56 | 75.73 | +36.78 |  546 | 41.76 | 73.79 |
| 92.5  | +54.40 |  481 | 44.49 | 69.26 | +44.94 |  467 | 45.61 | 68.93 |
| 95.0  | +65.29 |  402 | 49.75 | 64.72 | +53.53 |  407 | 48.89 | 64.40 |
| 97.0  | +73.40 |  337 | 52.52 | 57.28 | +60.61 |  352 | 51.99 | 59.22 |
| 98.0  | +78.67 |  313 | 54.63 | 55.34 | +68.28 |  313 | 54.63 | 55.34 |
| 99.0  | +93.21 |  253 | 58.89 | 48.22 | +80.83 |  260 | 58.85 | 49.51 |
| 99.5  | +100.54|  221 | 64.25 | 45.95 | +95.22 |  208 | 66.83 | 44.98 |
| 99.7  | +131.85|  166 | 74.10 | 39.81 | +122.27|  165 | 73.94 | 39.48 |
| 99.9  | +145.97|  149 | 73.83 | 35.60 | +133.34|  153 | 74.51 | 36.89 |

**Verdict**: Laplace and raw track each other within ±2 pp of purity at every
quantile. Maximum purity reached is ~74 % at p99.7 — still 11 pp short of 85 %.
Neither method approaches the 85 % purity target. Reference: the baseline
prior-marginal `P_DLA > 0.99` cut on the same v3_loa124 catalog gives
P=84.5 % / C=76.6 % (HANDOFF), so Method B (raw or Laplace) is significantly
WORSE than the prior-marginal score at every operating point.

## 5 known-missed candidates (HANDOFF)

| TID | in v3 scope? | NHI_MAP | log_LR_raw | log_LR_lap | hess_pd | half_logdet | log_prior | tp_by_z | truth-NHI | survives p95 (raw / lap) |
|-----|-------------:|--------:|-----------:|-----------:|:-------:|------------:|----------:|:-------:|----------:|:------------------------:|
| 105798 | yes | 18.79 | +9.16 | +0.97 | PD | 9.20 | -0.82 | no | – | n / n |
| 1798 | yes | 20.66 | +23.30 | +15.66 | PD | 8.36 | -1.11 | yes | 20.54 | n / n |
| 80198262 | NO | – | – | – | – | – | – | – | – | – |
| 64988 | yes | 20.25 | +18.48 | +11.07 | PD | 8.25 | -1.00 | yes | 20.41 | n / n |
| 20115135 | NO | – | – | – | – | – | – | – | – | – |

**Sanity-check finding**: The Laplace correction subtracts ~7-9 logL from BOTH
ghost noise detections AND legitimate signal detections, so it does not
discriminate. TID 1798 (real DLA logN=20.54): raw +23.30 → Laplace +15.66
(penalty 7.6). TID 64988 (real DLA logN=20.41): raw +18.48 → Laplace +11.07
(penalty 7.4). Both correctly z-match truth, but **neither survives the p95
threshold** in raw OR Laplace — the threshold is set high precisely because
the null-pop tail (forest noise at logN ≈ 20.5) extends to similar log_LR
values. The detection signal in low-SNR spectra (TID 1798 SNR=2.06, TID 64988
SNR=3.41) is comparable in magnitude to the noise-overfit tail in null
spectra, and Laplace correction does not break that degeneracy.

## Conclusion

**Laplace alone does NOT push Method B past 85 % purity** on London 8f v3_loa124
at SNR>2. At p95 of the null distribution it gives **P=48.9 % / C=64.4 %**,
essentially identical to raw MAP+LR (P=49.8 % / C=64.7 %). At p99 it tops out
at P=58.9 % / C=49.5 %. The Laplace penalty (`½ log|H| − log p(θ_MAP)`,
typically 7-10 logL) is applied uniformly to both noise and signal — the noise
ghosts have narrow peaks (large `½ log|H|`) but so do the legitimate weak DLA
detections in low-SNR spectra. The Laplace correction does not break the
signal-vs-noise degeneracy that drives the false-positive tail.

**Gap to baseline**: the prior-marginal `P_DLA > 0.99` cut on the same catalog
gives P=84.5 % / C=76.6 % (HANDOFF table); Method B+Laplace at its best
operating point (p99.7) reaches only P=74 % / C=40 %. The end-to-end Bayesian
marginal evidence remains the better detector — the Occam volume penalty
encoded in the marginal integral does what Laplace approximates but more
accurately, because the actual likelihood-peak shape can be far from Gaussian
(especially at the [17, 22] log NHI optimizer boundary, where 78/3345 specs
already have non-PD Hessians).

**Implications for the path forward** (per `2026-05-12_mlmc_design.md`):
- MAP+Laplace cannot replace the prior-marginal score as a detector.
- The remaining principled fix is adaptive importance sampling / MLMC at the
  MAP seed — see the design note. This keeps the marginal-evidence framing,
  reduces variance for narrow-peak likelihoods, and preserves CDDF Pathway A.
- The "v3_loa124 + null-quantile threshold on Δ_marg" experiment (HANDOFF #1)
  is still the most promising short-term win for the headline P/C at SNR>2.
