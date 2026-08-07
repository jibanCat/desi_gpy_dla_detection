# STAGE-2A VERDICT — the production bridge FAILED; response QUARANTINED; stopped for PI review

Per rulings §3.1 (no-go): the new production response is quarantined
(`quarantined_forward_response_2lpt0_phaseC.npz`), nothing is spliced,
no closure statistic was computed with it, the bridge criterion is
untouched, and the majority of the Stage-2B FP allocation is NOT
committed. **A failed bridge is a scientific result — and this one is
informative.** All artifacts: `diagnostics_phaseC/stage2A/`.

## The campaign (facts)

2,597 prodlike + 260 clean-probe injections through the frozen
executable state (jobs 56619743/56619744, **66.9 CPU-h consumed** — 
Stage-2A total incl. pilot ≈ 75 of the 1,850 ceiling). Scoring at full
production fidelity (roles enforced; holdout untouched; sentinel/
DLAFLAG/analysis-window cuts). Yields: bridge 694/769 op-matched,
production 1,163/1,167, ZERO injections out of the analysis window.
Precision/power go-condition PASSED on the measured variances:
σ(G3) = 99.0 counts ≤ 116.7; power 0.976 ≥ 0.90.

## The bridge verdict (frozen criteria)

ALL FOUR criteria FAILED: G3-projected old−new difference
**D = −476.1 ± 104.7 counts** (|D| ≥ 75; CI-upper 681 ≥ 116.7); global
coherence z_mean = 7.9, z_width = 10.5 (≥ 3); max local |z| = 3.89 with
a coherent pattern (≥ criterion via the ensemble); completeness
criterion failed at every bridge anchor. Ĉ_shared split-half ratio 1.00
(no inflation needed — the failure is not shared-forest noise).

## Diagnosis (the §3.1 source identification)

* **Component 1 — detection-conditioned estimand mismatch, true N ≲
  20.4.** The per-bin old−new mean difference runs −0.121 ± 0.016 dex
  (19.6) → −0.064 (19.8) → −0.042 (20.0) → −0.028 (20.2) → **+0.004 ±
  0.013 (20.4: exact agreement)**, tracking completeness: injected
  detection 0.81→0.99 versus molly 0.43→0.58. The old natural matched
  pairs are the upward-fluctuated SURVIVORS of a 43–58%-complete
  production selection; the injected sample (neighbor-free by
  construction, 5,000 km/s exclusion) detects nearly everything. The
  clean-probe arm shows NO substrate dependence within the new pipeline
  (all |z| ≤ 2.0 across 14 anchors), so this is a BETWEEN-pipeline
  estimand difference, not an injection artifact. Crucially this is NOT
  evidence the old response is wrong for production use at low N: the
  fold pairs the kernel with the molly completeness, which soaks
  exactly those matching losses — **(C_molly, K_natural-matched) is a
  jointly-defined, self-consistent pair**, and the injected campaign
  measures a different pair (C≈1, K_clean). Replacing K alone would
  double-count selection — which is precisely what the quarantine
  prevented.
* **Component 2 — the high-N boundary discrepancy (the original
  Phase-C target).** At 21.0 the pipelines disagree by **−0.051 ±
  0.011 dex** (dominating D: S₉ ≈ 9,300 counts/dex), and the new
  measurement above the old top anchor runs ≈ 0 to −0.042 dex
  (21.2→22.25) against the deployed clamp's +0.03…+0.09 — the pilot's
  engineering observation, now at production precision on 1,163 pairs.
  Interpretation is entangled with component 1 to the extent molly
  completeness at [21.0, 21.5) is 0.69 (selection can still act);
  above ~21.5 both selections saturate and the measured ≈ −0.03
  vs clamped +0.03…+0.09 is a genuine boundary-continuation error of
  order the preimage's G3-explaining scale.
* Old/new calibrations, truth construction, matcher, normalization,
  finder: the finder+matcher are IDENTICAL objects in both pipelines
  (frozen state); truth construction differs (natural vs injected) —
  the identified source is **truth-construction/selection conditioning**,
  the first item on the rulings' §3.1 source list.

## Diagnostic-only projection (quarantined kernel; labeled, no closure)

IF the invalid wholesale refit were adopted, group-μ would shift by
(ΔG1, ΔG2, ΔG3) ≈ (−3,700, +1,200, −470) on all three mocks —
demonstrating the low-N estimand contamination would corrupt every
group. No closure statistic was computed; the runner refuses the
quarantined artifact by design.

## Decision paths for the PI (none taken unilaterally)

* **P1 — coherent pair replacement above a support boundary:** adopt
  the injected (C, K) as a PAIR above ~20.4–20.5 (where injected
  selection saturates), keep (C_molly, K_natural) below. This changes
  the production completeness definition in the high-N region — a
  production-definition change requiring your ruling; it would need a
  revised bridge (overlap [20.4, 21.0] where both selections are
  near-saturated) and a coherent re-fold validation. Cost: analysis
  only (~0 CPU-h; the measurements exist).
* **P2 — estimand-matched remeasurement:** measure the high-N response
  from NATURAL pairs on the independent realization (mock-1/loa-124
  HAS spectra + truth) — the same estimand as the old response, new
  events. Natural high-N systems are rare: ~300 pairs above 20.9 needs
  ~30 healpix ≈ **1,500 CPU-h** (81% of the ceiling) for σ(mean) ≈
  0.008 dex — feasible only by dropping most of the FP program or with
  a new envelope.
* **P3 — bounded-systematic treatment:** keep the old response;
  convert the clamp region's support label from a point assumption to
  a MEASURED systematic band (the injected clean-response bound:
  −0.03…−0.09 dex below clamp above 21.2), propagated as a Layer-C-
  style reported systematic, not a covariance term. Cost ~0; leaves
  the G3 question quantified but not resolved within the frozen gate.

Stage 2B (FP expansion) note: its scientific value (FP precision,
transport, low-boundary accounting) is INDEPENDENT of this bridge
failure, but per your sequential mandate the majority FP spend is
withheld pending this review. Budget consumed: ≈ 75 of 1,850 CPU-h.

## Cosmetic defects recorded (no verdict impact)

Criterion 4's 21.0 row shows C_inj = 0 (vacuously inside-3σ): the
completeness pool drew from bridge-role pairs only and 21.0 is a
production-role bin — the row is inert (the criterion already failed on
the five real bridge anchors); to be fixed with the next builder
amendment. The completeness comparison also pools molly across all SNR
strata (stated in the code); a stratum-matched comparison would shrink
but not close the gap (0.43 vs 0.81 at 19.6 is far beyond weighting).
