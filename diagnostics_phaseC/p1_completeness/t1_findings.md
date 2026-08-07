# Tier-1 findings — the completeness gap is an ACCOUNTING artifact of dead strata

Status: Tier 1 complete (t1a + ledger). Evidence Level A throughout
(immutable deployed artifacts; the frozen primary-cause hierarchy in
`t1_ledger.py`'s header was committed before its first run; alternative
ordering reported in `t1_ledger.json`).

## 1. Deployed C_molly: REPRODUCED (stopping-rule criterion 1 MET)

Integer-exact in all 96 cells after replicating the deployed TWO-CHAIN
SPLICE (≥19.5 cells from the canonical nhi195 chain; <19.5 cells from
the floor-17.2 chain; `extract_pack.py:556-589`). A single-chain
reconstruction differs by up to 473 counts at [19.5,20.0)×mid-SNR —
matching competition against differently-floored truth is EMBEDDED in
the deployed numerators (recorded; relevant to any future estimand
work).

## 2. The eligibility split (§9): 47.0% of in-window truth is OUTSIDE the live fold

The truth table joins S2N_RED from the sightline catalog; the molly
matrix CONTAINS sub-2 SNR strata, but the FOLD zeroes them (dX = 0 —
the op-mask S2N_RED > 2; the F5 guard's "structurally empty strata").
69,008 of 146,792 in-window truth systems (47.0%) sit on S2N_RED ≤ 2
sightlines: **class-1, scientifically outside the production analysis**,
NOT finder or matcher losses. (Their truth-catalog local SNR medians
~0.55 and blue positions explain the earlier zero-row profile.)

## 3. Live-support attrition (S2N_RED > 2): natural ≈ injected

| true N | eligible | matched | C_live | H1 no-cand | H3 subthr | H5 assign | H2 bundle |
|---|---|---|---|---|---|---|---|
| [19.5,20.0) | 32,360 | 25,895 | **0.800** | 3,834 (11.8%) | 1,826 (5.6%) | 766 (2.4%) | 39 |
| [20.0,20.4) | 21,363 | 19,174 | **0.898** | 1,035 (4.8%) | 767 (3.6%) | 364 (1.7%) | 23 |
| [20.4,21.0) | 18,197 | 17,323 | **0.952** | 327 (1.8%) | 357 (2.0%) | 166 (0.9%) | 24 |
| [21.0,21.5) | 4,975 | 4,871 | **0.979** | 34 | 42 | 16 | 12 |
| ≥21.5 | 889 | 868 | **0.976** | 11 | 7 | 0 | 3 |

Injected campaign completeness over the same anchors: 0.81–0.99.
**The natural-vs-injection completeness gap on the production-relevant
population is ≈ 0–2 points, not 38 points.** The 43–58% figures quoted
at the bridge pooled the dead strata into the denominator.

Cross-checks: pooled-all-strata matched fractions reproduce the deployed
molly ratios exactly (0.458 / 0.694 at the earlier-quoted cells) —
the deployed PER-STRATUM completeness surface is correct and the FOLD
uses it correctly; nothing in production mis-multiplies. H4
(tolerance) = 0 everywhere: no truth with an op candidate on the
sightline lacked one within dz_rel. Near-neighbor (≤5,000 km/s truth
HCD) fraction: 6.5–7.1% at every N — the one parent-population class
the injection placement excludes by construction (bounded, small).
Placement audit: 2,597 accepted of ~3,632 draws (546 HCD-neighbor +
489 window redraws); the generator screens NOTHING else (no
mask/SNR/blend avoidance) — the injected denominator is parent-
equivalent to natural live-support hosts up to the near-neighbor class.

## 4. Consequences

* **Bridge criterion 4 as implemented was mis-pooled** (all-strata
  molly vs live-only injections). The bridge verdict is UNCHANGED —
  criteria 1–3 (pair means/widths/G3-projection) fail independently of
  any completeness pooling — but the criterion-4 construction must be
  corrected in any future bridge, and the Stage-2A verdict's
  "completeness far outside 3σ" line must not be quoted as a physical
  finding. Recorded here per the honest-record rule; no re-adjudication
  is performed unilaterally.
* **The detection-conditioned-selection explanation of the sub-20.4
  mean offset is WEAKENED**: with both selections retaining ~80–90%
  there, survivor bias of the missing ~10–20% cannot obviously carry a
  +0.12 dex mean offset at 19.6 decaying to zero at 20.4. The offset is
  REAL (pair-level) and its mechanism moves to Tier 2 (mismatch/blend
  composition of natural pairs — natural detections can be claimed by
  blended sub-floor structure; injections cannot within 5,000 km/s).
* **The high-N conclusion STRENGTHENS**: at [21.0,21.5) both pipelines
  are ≥97.9% complete — selection is absent — and they still disagree
  by −0.051 ± 0.011 dex at 21.0 with the measured ≈0…−0.04 vs clamp
  +0.03…+0.09 above. The boundary-continuation error now stands free of
  the selection confound in the region that feeds G3.
* **P1 shape**: on live support above ~20.4, (C_natural, ·) and
  (C_inj, ·) are numerically equal and both selections are
  near-saturated — the coherent-pair question reduces there to the
  KERNEL difference, with the sub-20.4 offset bounded by its measured
  decay-to-zero at 20.4. Tier 2 must bound the mismatch/blend mechanism
  before the transition freeze.

## Stopping-rule position

Criterion 1 MET. Criterion 2: the original gap is attributed at
Level A — ~100% of the pooled gap = dead-strata dilution; the live-gap
(0–2 points) attribution per range: H1+H3+H5 decompositions above (all
Level A). Criterion 3 (residual projection ≤ 50 counts on G3): the
remaining UNEXPLAINED object is no longer a completeness gap but the
sub-20.4 pair-mean offset — projected through the preimage S_b over
bridge bins it contributes ≤ ~25 counts to G3 (S_b ≤ 385 counts/dex
below 20.4 × ≤0.12 dex × decaying) — under the bound, PROVIDED the
mechanism does not extend above 20.4 (measured: Δ(20.4) = +0.004 ±
0.013). Criterion 4 (design-change screen): the P1 parent population
must be defined on S2N_RED > 2 live support (decided by this tier);
denominator = live eligible truth; miss state = live-support
non-matches; conditioning must include the live SNR strata; transition
support unchanged ([20.4, 21.0] overlap remains valid); holdout
criterion unaffected. Tier 2 (bounded): the survivor/mismatch tests for
the sub-20.4 offset — required before the estimand freeze, per the rule.
