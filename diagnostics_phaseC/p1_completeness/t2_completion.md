# Tier-2 COMPLETION — the sole gate is resolved at catalog level

Frozen verdict rule (`t2_pairing.py` header, stated before the run):
**IMPRINT-SUPPORTED (environment-flat)**, with the coupling contribution
bounded. The stopping rule's criterion 3 is now MET (attribution +
bound); the completeness investigation is COMPLETE. No forced fits, no
holdout rows, no Stage-2B, no P2 were touched.

## 1. The discrimination result (t2_pairing.json)

Common-substrate (isolated naturals mirroring the injection exclusion;
injected-cell reweighting on pre-selection covariates only):

* D1 — offset (natural − injected) by host N: +0.018 ± 0.007 (z 2.4) →
  +0.025 ± 0.007 (3.7) → **+0.045 ± 0.006 (8.0)** → +0.038 ± 0.008
  (4.6) over [20.4, 21.7). Significant and N-rising.
* D2 — shell-density (5–10k km/s catalogued neighbors) slopes: natural
  +0.0068 ± 0.0025 vs injected +0.0025 ± 0.0079; difference z = +0.51.
  **Environment-flat** → host-coupling (as proxied by catalogued shell
  density) is NOT the driver; its residual contribution is bounded by
  the slope-difference CI (≲0.015 dex per shell count at 95%). Stated
  limitation: the proxy cannot see uncatalogued sub-17.2 correlated
  absorption; within catalog reach, the N-rising + environment-flat
  pattern is the imprint-realism signature (quickquasars' natural-
  absorber imprint reads systematically higher to the finder than
  `inject_voigt`'s profile at the same catalog NHI, growing into the
  damping-wing regime).

## 2. Level-A capstone: the NATURAL pairs refute the deployed clamp directly

The production estimand itself (natural matched op pairs, live support —
no injections, no transfer question), per bin against the deployed
clamped surface (pair-weighted):

| true N | n pairs | natural dx | clamped surface | pairs − surface |
|---|---|---|---|---|
| [21.0,21.2) | 2,787 | +0.0546 ± 0.0027 | +0.0488 | +0.006 (+2.2σ) |
| [21.2,21.4) | 1,629 | +0.0443 ± 0.0032 | +0.0529 | −0.009 (−2.7σ) |
| [21.4,21.6) | 792 | +0.0221 ± 0.0047 | +0.0524 | **−0.030 (−6.5σ)** |
| [21.6,21.8) | 345 | +0.0086 ± 0.0082 | +0.0527 | **−0.044 (−5.4σ)** |
| [21.8,22.1) | 156 | +0.0127 ± 0.0113 | +0.0554 | **−0.043 (−3.8σ)** |
| [22.1,22.5) | 30 | +0.0198 ± 0.0189 | +0.0537 | −0.034 (−1.8σ) |

The true mean-bias FALLS from +0.055 to ≈+0.01 through [21.0, 22.1)
while the clamp holds ≈+0.05. Together with the Tier-2 surface-vs-pairs
mid-range misfit (−0.035/−0.028 at [20.4,21.0)), the deployed surface
misrepresents its own calibration pairs on BOTH sides of 21.0, and both
errors suppress μ(G3).

## 3. Scale note (BACK-OF-ENVELOPE, labeled — not a refold, nothing adopted)

Projecting the pairs-minus-surface corrections through the committed
preimage sensitivities: mid-range [20.4,21.0) ≈ +400–450 counts on
μ(G3); clamp region [21.4,22.1) ≈ +120 counts (plus reduced 21.7+
over-prediction). Combined ≈ the full +450-count G3 discrepancy at the
right order. The exact number requires folding a pairs-faithful kernel —
that is the GATED P1 predict step, not performed here.

## 4. Consequences for the P1 design (§22 report)

* The natural-pair estimand at high N is measurable at production
  precision FROM THE EXISTING mock-0 catalog (n = 4,871 + 868 above
  21.0 on live support) — much of what P2 was scoped to buy (~1,500
  CPU-h on mock-1) already exists within-realization for free;
  mock-1 would add only realization independence.
* The injected operator carries an N-rising imprint-transfer bias
  (+0.02…+0.045 dex) relative to the production estimand: any P1 use of
  injected K requires that measured correction (natural-anchored) or a
  natural-pair kernel outright, with injections supplying
  completeness/miss-state and low-density support.
* Denominator/parent population: LIVE support (S2N_RED > 2), decided in
  Tier 1. Blend class (7.5–8%, +0.03…+0.10 dex) carried as a
  composition term. Transition overlap [20.4, 21.0] remains valid.
* The estimand freeze can now be written; per the PI's binding sequence
  it follows at the next work step, with the failure taxonomy and the
  holdout adjudicability gate, before any holdout read.
