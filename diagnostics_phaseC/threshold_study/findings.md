# Layer-B threshold operating study (§14.3) — findings for PI ratification

Routine: `run_threshold_study.py` (this directory); artifact
`threshold_study.json`. Independently simulated operating characteristics
on a production-geometry (29×15×8) `synthetic_pack` universe in the
production FP regime (94 events carrying a 12.3% μ_FP share), running the
DEPLOYED Layer-B procedure per replicate (E_cov B=2000 seed 41001; null
B=2000 seed 43001; exact 3-group Mahalanobis; fixed deployed seeds).
Faithfulness guards: the committed `fold_mu_fp` linearity probe (1e-9) and
exact T-statistic agreement with the committed `predictive_gate`
(|ΔT| < 1e-12). The observed Phase-B failures enter ONLY as the ε=1
alternative SHAPE (the twin's per-bin fractional residual, committed as
`observed_tilt_shape.npz`) — the threshold candidates were never evaluated
against the observed data.

## Operating characteristics (per-mock rates; family = 3 mocks sharing one calibration draw)

| config | α=0.01 | α=0.05 |
|---|---|---|
| healthy, 89-event regime (n=2000): per-mock type-I | **0.0167 ± 0.0032** | 0.0573 ± 0.0059 |
| — P(≥1 of 3 healthy mocks fails) | **0.034** | **0.117** |
| — P(all 3 pass) | 0.966 | 0.883 |
| healthy, B_null=500 (finite-null sensitivity) | 0.0180 | 0.0610 |
| healthy, 400-event regime (κ=4.5) | 0.0133 | 0.0500 |
| healthy, 1111-event regime (κ=12.5) | 0.0050 | 0.0517 |
| power, observed-scale tilt (ε=1, per mock) | 0.508 | 0.762 |
| power, half-scale defect (ε=0.5) | 0.077 | 0.192 |
| power, quarter-scale defect (ε=0.25) | 0.025 | 0.102 |
| power at κ=4.5 / κ=12.5 (ε=1) | 0.417 / 0.417 | 0.647 / 0.670 |

## Reading (recommendation to the PI, not a self-ratification)

* **p < 0.01 is not evidently too strict.** Its healthy per-mock type-I
  is mildly inflated at the CURRENT 89-event calibration (0.0167 — the
  finite-calibration/finite-null price) and converges to ≤ nominal as the
  calibration grows (0.005 at the 1111-event regime); the family-wise
  false-alarm rate is 1-in-29 healthy triples, versus **1-in-9 at
  α = 0.05**. The α=0.05 power advantage at the observed scale (0.76 vs
  0.51 per mock) is immaterial for the current verdict: the actual
  Phase-B failures sit at p ≤ 5·10⁻⁴ on ALL three mocks, far beyond both
  candidates, and family-wise (≥1 of 3) detection of an observed-scale
  common tilt is high under either.
* Neither threshold reliably detects a HALF-scale defect per mock
  (0.08/0.19) — material-but-smaller biases are caught by replication
  across mocks and by the calibration precision program, not by this
  gate; this limitation is threshold-independent.
* Per-mock vs joint: rates above are per-mock with the measured
  shared-calibration family correlation; a joint (all-3) rule was not
  proposed and none is introduced.

## Stated limitations (MC + design)

* Type-I MC 95% half-width at α=0.01 is ±0.0032 at n=2000 replicates —
  the §15.5 ≤0.002 target needs ~9,500; the current value is reported as
  a resolved-to-this-precision estimate, not a bound violation; the run
  costs 3.4 CPU-min and extends trivially if the PI wants the tighter
  figure.
* The synthetic survey (35.9k window counts) is ~2.4× smaller than the
  production window (88k), so the power column is CONSERVATIVE (real
  power at the observed scale is higher — consistent with the observed
  p ≤ 5e-4 on all mocks).
* The three family members share one truth template (the real mocks share
  the signal calibration bit-identically but differ in truth) — stated
  approximation for the family-wise rates.

Status: PROVISIONAL p < 0.01 stands; continuous calibrated p-values
remain the report; ratification is the PI's decision at the checkpoint.
The frozen Phase-B verdict is untouched by construction.
