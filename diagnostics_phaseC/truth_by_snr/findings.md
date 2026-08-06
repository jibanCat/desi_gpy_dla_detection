# Phase-C truth-by-SNR refold (PI §16) — EXPLORATORY, one bounded pass

Authorized diagnostic; zero model freedom (an existing pack array replaces
the pathlength-proportional truth allocation; same kernel, completeness, g,
FP). Prespecified signature in `run_truth_by_snr.py`'s header. Sanity:
strata sum reproduces the marginal truth histogram exactly; rebuilding the
baseline through this script's plumbing reproduces `selftest`'s μ_sig to
<1e-8 on every mock.

## Result: ALLOCATION/COMPOSITION IS REFUTED as the tilt driver

The truth's actual SNR allocation differs from pathlength-proportional by
only **6.6–6.7% (L1 fraction)**, and folding the actual allocation moves
the closure statistics by amounts that are negligible against the failure:

| mock | G3 residual base → refold | Δ | window χ²/dof base → refold |
|---|---|---|---|
| 2LPT-0 | +450.2 → +452.7 | **+2.5** | 22.09 → 21.99 |
| London-0 | +464.9 → +463.5 | −1.4 | 28.16 → 28.41 |
| Saclay-0 | +191.0 → +186.3 | −4.7 | 25.57 → 25.39 |

G1/G2 move by ≤ 25 counts (< 1.5% of the G1 residual). The per-stratum
structure also survives essentially unchanged (twin):

* G1 z per stratum: +6.07 … −12.61 (base) → +5.84 … −12.13 (refold) — the
  H10 monotone G1-vs-SNR tilt is NOT a truth-allocation artifact; it lives
  in the calibration surfaces themselves (completeness-vs-SNR and/or
  kernel-width-vs-SNR).
* G3 z per stratum: +6.09/+2.91/+1.97/+2.58/+1.77/−0.55 → +5.90/+3.32/
  +1.80/+1.85/+1.55/−0.02 — **the high-N̂ deficit does not concentrate in
  any SNR range** under either allocation.

## Answers to the §16 questions

* Contribution of each SNR stratum to G1/G2/G3: in `truth_by_snr.json`
  (`per_stratum_groups`), both allocations, all mocks.
* Does the high-N̂ deficit concentrate in one SNR range? **No** —
  SNR-near-uniform under both allocations (slightly flatter under refold).
* Replication: yes — the Δs are ≤5 counts on all three mocks.
* Implication for high-N calibration conditioning: the §10 SNR
  stratification of anchors REMAINS required (the per-stratum G1 pattern is
  real calibration-side structure), but no SNR-composition mechanism can
  produce the G3 deficit: **the result supports response shape, not an SNR
  nuisance, and no SNR nuisance function is introduced.**
* Did it change the calibration design? No change to anchor placement or
  support; it confirms the design's existing requirement that anchors span
  all live SNR strata.

Label: EXPLORATORY (outside any frozen confirmatory spec). One pass, as
authorized; no rerun, no fitting, nothing adopted into the model.
