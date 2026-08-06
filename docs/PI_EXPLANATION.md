# For the PI: the state of the analysis after Phase B (2026-08-06)
### Plain-language companion to `docs/MODEL_SPEC.md` and the closure table; all numbers mock-derived

## 1. What Phase B did, in one paragraph

We froze the statistics before computing anything new; built the honest
error accounting for the false-positive calibration; restored a small
missing factor the FP definition always contained; corrected the public and
private record of the retracted claims; produced the first closure table in
which every layer of evidence is labeled by what it can actually test; and
ran one bounded diagnostic pass on what remains broken. No model freedom
was added anywhere. The model still fails — but for the first time the
failure is precisely characterized, and most of what previously looked
broken was the measuring stick, not the model.

## 2. Why calibration noise changed the picture

Our forest-false-positive rate comes from 89 events found in an
absorber-free calibration mock. Each of those events, scaled to the survey,
stands for about 166 predicted counts. The old closure statistics treated
those 89 events as exact truth and measured every residual against survey
noise alone. That overstated significance wherever false positives matter:
it manufactured the "SNR axis is the leading residual" claim (rejected in
Phase A), and — as the Phase-B diagnosis now shows — 58–66% of the per-bin
window χ² as well (the low-N̂ sawtooth collapses once the calibration band
is included). This cannot be fixed by inflating per-bin error bars: one
calibration event feeds many bins at once, so the uncertainty is
correlated across bins, carries a shared amplitude mode, and must be
propagated by resampling the calibration through the entire
template→normalization→fold pipeline. That is what the new Layer-B gate
does.

## 3. Why transport bias is not calibration variance

Calibration variance is the noise of the 89-event sample; transport bias is
the error of carrying the calibration mock's FP rate to a *different* mock
by a single sightline ratio. We measured that these are cleanly separable
here: the calibration twin shares its actual skewers with the calibration
mock (we verified sightline-by-sightline: identical quasar redshifts), so
on the twin the transport is exact by construction — and indeed the twin's
total closes at +0.06%. The two held-out mocks are fully independent of the
calibration (zero shared sightlines) and show a 31–45% transport
over-prediction. Absorbing that bias into the predictive covariance would
manufacture agreement; per your ruling it is reported separately as an
uncalibrated systematic (Layer C), and the t prior was not touched.

## 4. Why the statistic had to be frozen first

Phase A demonstrated how a plausible statistic chosen after seeing the
residuals produced a wrong headline ("0.6σ"). The Phase-B spec therefore
fixed — before any new residual was computed — the diagnostic axes, the
grouping, the covariance construction, the ensemble sizes and seeds, the
null-calibration procedure, and the interpretation rule; every choice is
classified by provenance, and anything chosen later is labeled exploratory
and cannot be promoted. The finite calibration also limits how many
covariance dimensions are trustworthy: with 89 events (29 in the reporting
window, all below N̂ 20.3) a 19-bin joint statistic would be noise-
dominated, so the confirmatory statistic lives in 3 prespecified,
physically meaningful groups, inverted exactly, with the fallback rule
frozen in advance.

## 5. What the honest closure table says

**Conditional layer** (implementation check against the realized
calibration, the historical χ²/dof ≤ 3 diagnostic): 22.09 / 28.16 / 25.57 —
unchanged in character, now labeled as what it is.
**Predictive layer** (the confirmatory science gate): the model **fails on
all three mocks** — T = 40.7 / 46.9 / 32.7 against a simulated null whose
99th percentile is ≈ 11; p ≤ 5×10⁻⁴ (the resolution bound of 2000 null
draws). The failure is a smooth, common-signed shape tilt in observed N̂:
over-prediction at 19.9–20.5, under-prediction from 20.5 up to the window
ceiling — largest in [21.0, 21.6] (+5.9σ on the twin) — and renewed
over-prediction above the window.
**Transport layer**: twin exact (−0.03σ); held-outs −2.7σ / −1.5σ once the
template's own noise is counted.
**The (1−η̄) restoration**: −0.576% on the FP term, exactly as the
definition predicts, every re-pinned value verified to be the pure effect.

## 6. What the bounded diagnosis established

One pass, ten prespecified discriminants, no new model freedom:

- The per-bin χ² inflation is mostly the calibration-shape noise of item 2
  — already priced by the new gate. A reporting-metric correction, not a
  model repair.
- The surviving tilt is **not** explained by any calibrated existing
  mechanism, each refuted with measured effect sizes: sub-floor promotion
  (extending the truth floor to 17.2 moves the [21.0,21.6] residual by
  0.001 counts); completeness offsets (would need ~10⁵ prior widths);
  in-span kernel perturbations (need up to 320 prior widths and do not
  transport); the clamp bracket (both settings leave the high-N̂ residual;
  the pre-fix "off" convention makes it worse); bin-edge mismatches and
  stale artifacts (excluded bit-level).
- What remains is most consistent with a **response-kernel shape error at
  and above the weakly-measured region (≳ 21.05)** — precisely where the
  kernel's anchors stop. Testing or fixing that requires new kernel
  freedom or a response re-measurement: your call, not ours.
- A cheap, zero-freedom discriminant exists but sits outside the
  prespecified list (refolding with the pack's own truth-by-SNR allocation
  instead of the path-length-proportional one); it needs your
  authorization.

## 7. What is measured, what is anchored, what is prior-driven

Measured by the survey data: the window population shape (within the tilt
that fails the gate); the total count; the detectability of a missing
sub-floor population. Anchored by the 89-event calibration: the FP total.
Supplied by priors: the sub-floor amplitude (smoothness prior), the FP
shape across cells (149/174 cells have no calibration events), the FP
redshift allocation (imposed ∝ path length), the transfer factors (a prior
the measured transport bias exceeds). None of this changed in Phase B; it
is now stated on every artifact.

## 8. Confirmatory vs exploratory — the ledger

Confirmatory (frozen before evaluation): the three Layer-B gate results
(fail, p ≤ 5×10⁻⁴ each). Independently validated: the (1−η̄) effect; the
bit-identity of the re-extracted calibration; the twin transport closure.
Exploratory (labeled, not promotable): the diagnosis rankings, the
secondary axes, the per-bin tables, the SNR-stratum structure inside G1,
the pad-17.2 fold. Sensitivity-only: everything in MODEL_SPEC §8,
including any frozen-FP-shape variant and the un-ratified p<0.01 threshold.

## 9. What still fails after honest accounting, and what does not

Fails: the observed-N̂ shape, dominated by the high end — a real,
well-characterized model failure that no existing calibrated freedom
absorbs. Does not fail (any more): the total normalization on the twin
(+0.06%); the SNR axis as a leading residual (3.6–5.9 with honest errors,
under the old gate's own threshold); the FP normalization itself
(twin-validated at +0.2% in Phase A, now with η restored).

## 10. Decisions that are yours (nothing below was implemented)

1. **Response kernel above ~21.05** — the leading failure points there.
   Options: re-measure the response with higher-N anchors; add bounded
   kernel freedom (new moment terms) with priors; or shrink the reporting
   ceiling. Each changes a ratified quantity (kernel definition or
   window); we implemented none.
2. **Gate governance** — adopt the Layer-B gate as the production science
   gate with a ratified threshold (p < 0.01 proposed)? Re-scope the
   Layer-A χ²/dof ≤ 3 as conditional-only (it currently reads as the
   gate)? Reclassify the H9 share of Layer-A χ² in reporting?
3. **Transport** — the prepared proposal (below) for independent loa-0-
   style calibration volume; the t prior stays untouched until then.
4. **The authorization-pending diagnostic** (truth-by-SNR refold).
5. **The cond-fallback detail** flagged by code review (may the fallback
   report a descriptive max|z| p at all, or standardized residuals only?).
6. **The r5 test guard** — under-powered by measurement; re-power or
   replace.

## 11. The transport-calibration proposal (prepared, not launched)

To make transport a measurement instead of an assumption: run the
production finder over additional absorber-free realizations matched to
each held-out mock's forest/noise/continuum family (the loa-0 recipe,
~2,000–5,000 searched sightlines each — the current calibration searched
2,255 and its 89-event Poisson noise contributes ±10.6% to μ_FP, the
single largest FP uncertainty). Each new realization makes response-free
FP supply directly measurable per mock family, turns the transport ratio
into a measured distribution, and shrinks the calibration variance ∝ 1/N.
Compute is finder-dominated and modest compared to the (unfunded) blue-end
campaign; exact sizing belongs to the run plan if you approve the
direction. Until then, nothing is launched.
