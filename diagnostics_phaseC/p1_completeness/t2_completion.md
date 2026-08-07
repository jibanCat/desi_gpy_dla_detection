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

## 5. POST-SPECIFIED power/robustness addendum (2026-08-07, `t2_power.py`)

Everything in this section was specified AFTER the frozen verdict was
seen; it diagnoses, it does not re-adjudicate. Frozen bins/proxies only.

**Power of D2 against the material coupling alternative (P1).** The
isolation-mirrored naturals do NOT sit at higher catalogued shell
density than the injected sightlines (mean shell 0.144 vs 0.166 —
lever −0.023). For the catalogued-neighbor channel to carry the
+0.045 dex offset at [21.0,21.3) it would need a slope difference of
+0.32 dex/count, which is **38σ above the measured** +0.0042 ± 0.0083;
even crediting the channel with the FULL natural mean shell count at
the 95% upper slope difference (+0.021), its contribution is
**≤ +0.003 dex** vs +0.045 required. D2 was NOT underpowered for the
channel it proxies; that channel is excluded at ~15× margin.

**Mechanical bound on the un-proxied near-field channel (P2).**
Injected dx vs `forest_flux_frac` (PRE-injection forest flux at the
trough centre — a design covariate recorded at generation, before any
selection): slope +0.037 ± 0.017 (n = 1,047; fff sd 0.186). Two
consequences: (i) magnitude — a 1σ population shift in pre-existing
central absorption moves dx by ≤ 0.007 dex (worst-conceivable
full-range shift ≤ 0.037), 6× below the offset; (ii) **sign — the
slope is POSITIVE (cleanest sites read HIGHEST): pre-existing
absorption at the trough slightly DEPRESSES fitted N̂**, the opposite
of what coupling-inflation requires. Correlated near-field absorption
therefore cannot manufacture the positive natural−injected offset
through the GP's measured response to local absorption (caveat: the
response is measured on `inject_voigt` imprints; attenuation from fff
point-noise could hide magnitude but cannot flip the sign).

**Robustness (R1–R3).** R1 joint dependence: per-bin D2 slope
differences z = −0.61 / +2.48 / −1.13 / +2.07 — sign-alternating, no
coherent pattern, and z = −1.13 (wrong sign for coupling) at
[21.0,21.3) where the offset peaks. R2: the offset persists at full
size in shell = 0 pairs on both sides (+0.0184/+0.0185/**+0.0475**/
+0.0331 — cf. unrestricted +0.0177/+0.0249/+0.0454/+0.0376). R3 —
**decision-relevant for the freeze: the natural kernel is WIDER than
the injected one at every bin** (robust σ 0.100–0.116 vs 0.088–0.099;
sd 0.124–0.136 vs 0.097–0.108, ~15–25%): an injected-K-plus-mean-
correction design would still mis-state the kernel width (worth
~50–150 G3 counts through the committed width sensitivities).

**Projection (R4, labeled back-of-envelope).** The frozen D1 offsets
folded through the committed preimage mean-shift sensitivities
(overlap-weighted 0.2-dex bins; sub-20.4 excluded, bounded ≤ ~25
counts): **ΔG3 = +387 ± 76 counts** — the scale of the full +450-count
discrepancy. The exact number remains the GATED P1 refold.

**Alternatives that remain viable at catalog level.** (i) imprint
realism (supported); (ii) an environmental variable coupled to N̂ yet
uncorrelated with BOTH catalogued shell density AND central forest
transmission — no physical candidate in the LyaCoLoRe→quickquasars
generative chain is known to us, but it cannot be excluded from
catalogs; (iii) any truth-side property correlated with being-natural
that alters the imprint at fixed catalog NHI — which IS the imprint
hypothesis, not a rival. Verdict wording stays: **consistent with
imprint-realism differences and disfavors the tested environmental
explanations** — uniqueness is NOT claimed.
