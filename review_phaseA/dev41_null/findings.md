# REVIEW-ONLY (Phase A) — referee report: the Δdev = 41 → "0.6σ" claim

*Recorded by the orchestrator from the referee agent's final report (the agent
environment could not write report files); the complete numeric record is
`summary.json` + `results_stream.jsonl` (all 300 replicates, seeds, timing) in
this directory. Independent fitter: `fitter.py` (EM + L-BFGS-B, convex problem;
gradient check 3.1e-7; validated against the archived probe: noiseless
Δ = 41.196 vs 41.21, seed-17 observed Δ = 85.60 vs 85.4).*

**Claim under test** (PI checkpoint §8 @ 9d73365): "'FP only, pad off' on data
injected with the opposite truth fits at Δdev = 41, i.e. 0.6σ against survey
Poisson noise, while manufacturing a 9.6× FP error."

## Verdict

**(a) The "0.6σ" derivation is INVALID — three separate errors.**
1. Category error: it divides a between-model LRT statistic by the sampling sd
   of the absolute GOF deviance of a fixed model (√(2·2610) = 72). The LRT
   statistic T = dev(F3) − dev(F1) has its own null distribution: empirical sd
   **7.3**, not 72.
2. The noiseless 41.20 is the **noncentrality** — twice the minimum Poisson-KL
   divergence from the injected truth to the pad-free family at survey
   exposure — not a test statistic. It is precisely the quantity that powers
   detection.
3. Naive Wilks χ²(75) fails in the opposite direction (q95 = 96.2). The 75
   released pad amplitudes sit on the non-negativity boundary and are strongly
   correlated → chi-bar-squared with empirical effective dof ≈ 23.

**(b) Empirical null (parametric bootstrap, N = 120 per hypothesis).**
Null (y* ~ Poisson(F3-fit-to-data)): mean 23.2, sd 7.3, q05 13.5, q50 22.1,
q95 35.8, q99 40.7, max 48.0. Alternative (y* ~ Poisson(injection truth)):
mean 69.9, sd 15.2, min 40.5, max 124.8. The distributions are essentially
disjoint. **Power = 120/120 = 1.00** at the null q95 (≥ 0.975 at 95%
confidence); 0.983 at q99. **p(observed Δ = 85.6) ≤ 0.0083** (0/120 null
exceedances; largest null draw 48.0). Even the bare noncentrality 41.2 sits at
the null's 98th percentile.

**(c) "A wrong model is undetectable" — REJECTED as stated.** Correct
statement: undetectable by the **absolute-GOF** check (measured GOF power
0.058 — the alternative shifts fitted dev(F3) by ~10 against sd ~61–67; the
session's intuition was right for the test it implicitly ran), but the nested
LRT calibrated by parametric bootstrap — a test the pipeline does not
perform — detects the pad↔FP misattribution with power ≈ 1 at this exposure.

**(d) "Manufactures a 9.6× FP error" — survives, reframed.** Noiseless F3:
T_B = 10,653 (9.8×; the argmin is near-flat, the multiple is not unique).
Under noise: 16,740 ± 1,472 (12–19×), so 9.6× is a floor. However the
anchor-free *correct* model F1 also fails on the same data (T_B = 8,452 ±
1,609 vs truth 1,087), while the loa-0-anchored F1b recovers 1,272. The large
FP error is chiefly the **unanchored FP-total non-identifiability**, not a
signature of the wrong model.

## Additional findings

- **The null is not pivotal**: a pure-FP generating point at matched totals
  gives null mean 10.9 / q95 16.6 (vs 23.2 / 35.8) — **no single
  σ-equivalent number for Δdev is meaningful**. Every candidate null lies far
  below the alternative, so the power verdict is robust; the analyst-facing
  null (F3 fit) is the wider, conservative one.
- The observed Δ = 85.6 is a typical alternative draw (~79th percentile).
- Correct-model pad recovery under noise is poor: T_A = 20,483 ± 2,519 vs
  24,000 injected (−15% mean; range 11.0k–25.7k) — the checkpoint's noiseless
  −7.7–8.5% under-recovery worsens under noise.

Seeds: y_obs = 17; null 100000+r; alt 200000+r; secondary null 300000+r.
Runtime: 300 fit-pairs in 1,376 s at 4 workers. Nothing committed by the
agent; no production file touched.
