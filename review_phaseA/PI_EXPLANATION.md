# For the PI: where the DLA/sub-DLA forward-model analysis actually stands
### After the independent adversarial review of 2026-08-06 (all numbers mock-derived; no real-DESI value appears here)

## 1. Executive summary (one page)

On 2026-08-05 a work session reported four headline results: (1) a real
normalization bug in the false-positive part of the forward model, repaired;
(2) the conclusion that the two populations living at the reporting floor —
genuine absorbers migrating up from below 19.5, and Lyman-α-forest false
positives — cannot be told apart by our data ("not separately identifiable"),
which was the stated reason the session stopped; (3) the statement that a
wrong model fits the data at "0.6σ", i.e. that we could not even detect the
error; and (4) a new "leading residual" on the signal-to-noise axis.

A fresh, fully independent review has now re-derived, re-implemented, and
stress-tested each of these. The outcome:

- **The bug and its repair are correct — strengthened.** An independent
  counting derivation reproduces the repaired equation exactly, and on the
  calibration mock (where no transfer assumption is involved) the corrected
  false-positive prediction now matches the measured supply to +0.2%.
- **The "not separately identifiable" headline is wrong as stated.** The
  numbers behind it reproduce exactly, but they were measuring something
  else: the near-degenerate directions are between the sub-floor population
  and the *completeness calibration* (and, at fine binning, between adjacent
  population bins), not between absorbers and false positives. Those two are
  in fact among the best-separated pieces of the model.
- **"A wrong model is undetectable (0.6σ)" is wrong.** The 0.6σ number came
  from comparing against the wrong statistical yardstick. Calibrated
  properly (by simulation), the wrong model is detected essentially every
  time. What *is* true: the split between the two floor populations is set
  by the priors and the loa-0 calibration, not by the survey data alone.
- **The "SNR residual" was ~85–95% an artifact** of treating an 89-event
  calibration sample as exact. Once that sampling noise is propagated, the
  SNR axis is not the leading residual; the N_HI axis is. A real but much
  smaller SNR effect remains, on the signal side (not the FP side).
- **The model still does not close** — that negative result survives review,
  at roughly 3× the ratified tolerance instead of 7–10× — and the failure
  now has a cleaner shape: on the calibration mock it is a residual
  N_HI-shape misfit; on the two held-out mocks it is dominated by the
  false-positive *transport* assumption (carrying the loa-0 rate to another
  mock by a single sightline ratio over-predicts by 31–45%).

Five specific published statements should be corrected in the project record
(§12). None of the corrections weakens the case for caution: the analysis is
still blocked from producing a posterior — but for different, better-founded
reasons than the ones reported. Several next steps now need your ruling (§16).

## 2. What the analysis is trying to measure

The incidence of strong H I absorbers — how many systems per unit absorption
path per interval of column density, dN/dX(N_HI) — over
log N_HI ∈ [19.7, 21.6], in three redshift bins, from the mock DESI-like
catalogues. Everything else (total Ω_HI, the LLS population, quantitative
redshift evolution) is out of scope for Paper 1 by earlier rulings.

## 3. How genuine absorbers and false positives enter the catalogue

The finder's catalogue mixes two populations. Genuine absorbers enter after
three filters the model must represent: a completeness probability (was the
system found at all), a response kernel (found at what apparent N_HI — a
system below the floor can be measured above it, "migration"), and the path
length surveyed. Forest false positives — noise and forest structure that
the finder reports as absorbers — are calibrated on **loa-0**, an
absorber-free twin mock: every detection there is by construction a false
positive. That calibration contains **89 events** in the analysis range. Its
per-sightline rate is transported to each analysis mock by a sightline-count
ratio.

## 4. What the false-positive normalization defect was

The model defines the FP intensity per unit of a calibration "exposure"
(ℓ ≈ 13.59). The calibration side used that convention; the forward fold —
the equation predicting survey counts — omitted the factor, so every folded
false positive was under-counted by exactly 13.59×. It survived because the
factor cancels identically on the calibration side (which is also why no
synthetic test could catch it — the synthetic generator shared the bug).

## 5. What changed after the repair — and what the review adds

The repair multiplies the folded FP term by ℓ at all four code sites. The
independent review re-derived the equation from a head-count argument
(FP-per-sightline × number of sightlines) and confirms the repaired code
matches it exactly; the alternative way of writing the repair is provably
equivalent. Two additions: (i) a small factor (1−η̄ ≈ 0.994) that the written
definition contains is still missing from the code (+0.58% on the FP term) —
a one-line fix awaiting your confirmation; (ii) the impressive-looking exact
identity used as evidence (w·ℓ = 2255 on all packs) is an algebraic
consequence of the definitions — it checks data integrity, not physics.

## 6. Why closure still fails — the corrected picture

"Closure" asks: fold the known truth through the model — do predicted counts
match observed counts? It still fails, and that is a real result. But the
review changed both its size and its interpretation. The previous gate
treated the 89-event FP calibration as exact; each of those events carries a
~166-count footprint in the prediction, so wherever the FP matters the gate
was comparing against far too small an error bar. With that propagated, the
failure is ~3× the ratified tolerance (was 7–10×), it lives on the N_HI axis,
and it decomposes cleanly: on the calibration mock the total is right to
+0.16% and what fails is the *shape* across N_HI; on London-0/Saclay-0 the
excess is quantitatively the transported FP amplitude. In plain terms: **the
model normalization is now right where we can check it absolutely; what
fails is the shape, and the assumption that one mock's FP rate carries to
another.**

## 7. What the SNR residual means

Mostly: sampling noise. The "coherent SNR tilt, worse than the main gate"
was computed against survey-noise-only error bars; 85–95% of it disappears
when the calibration sample's own noise is included, and its agreement
across the three mocks is guaranteed (they share the same 89 events), so it
was one observation, not three. A genuine, much smaller SNR effect survives
(the lowest-SNR stratum is under-predicted ~17–19% even with the FP off) —
it sits in the signal model (completeness or kernel SNR dependence, or
absorbers promoted from below 19.0), and it does **not** justify adding any
SNR flexibility to the model. One warning for later: the FP's SNR profile
measured on *all* loa-0 detections is opposite to the profile of those in
the analysis range — transferring the all-N profile makes things 3–4×
worse. If an FP shape is ever frozen, it must be the in-range one.

## 8. What "non-identifiable" actually means here

Two different statements were conflated. (i) *"You cannot tell the model
with no sub-floor absorbers apart from the truth"* — false: a properly
calibrated comparison detects the difference essentially always; the "0.6σ"
came from dividing by the spread of the wrong quantity. (ii) *"The data
alone do not pin down the sizes of the two floor populations"* — true, but
the culprit was misidentified: the sub-floor amplitude is degenerate with
the *completeness calibration* (an absorber added below the floor looks like
a completeness offset plus a small redistribution below N_HI 20), not with
the false positives; and the FP total, without its calibration anchor, can
be absorbed by the other components. With the full production priors and
the anchor, both totals are pinned to a few hundredths of a dex — **by the
priors and the anchor, not by the survey data**. That distinction — data-
identified vs prior-identified — is the honest language going forward.

## 9. What the data constrain directly

The window-region population shape (within the N_HI-axis misfit that still
fails closure); the total detection count; on the calibration twin, the
absolute FP normalization (+0.2%); and the *detectability* of a missing
sub-floor population (power ≈ 1 in the calibrated test).

## 10. What is supplied by priors or external calibration

The sub-floor (pad) amplitude — by the smoothness prior; the FP total — by
the 89-event loa-0 anchor; the FP shape across (N̂, SNR) — by its prior
(149 of 174 cells have zero calibration events); the FP redshift allocation
— imposed ∝ path length; the transfer factors — by a prior whose width the
measured cross-mock bias exceeds. Any reported number inherits these; the
prior-sensitivity axes to quote are the bottom completeness cell, the
transfer prior, and the smoothness scale.

## 11. What London-0 and Saclay-0 do and do not validate

Every calibration block in the three packs is bit-identical — one frozen
2LPT-0 calibration. So they test whether that fixed calibration predicts
*independent realizations* (different truth, path length, noise draw). They
cannot validate the calibration itself; an error common to it is invisible
by construction. No "validated on three mocks" statement about response,
completeness, FP template, or transfer widths is supportable.

## 12. Statements that should be corrected in the record

1. "Δdev = 41 is 0.6σ — a wrong model is undetectable" → rejected (invalid
   yardstick; calibrated power ≈ 1; no single σ-number is meaningful).
2. "16 of 75 pad directions within 1° of the FP+window span" as absorber-
   vs-FP evidence → 15 of 16 are pad↔window at fine binning (0 of 30 on the
   ratified basis); pad↔FP is well separated.
3. "Sub-floor migration and forest FP are not separately identifiable" →
   retire; replace with the prior-identified statement (§8).
4. "The leading residual is the SNR axis" → rejected (calibration-noise
   artifact; N_HI axis leads; the committed numbers were also silently
   window-restricted).
5. "μ_FP exceeds the mock's entire FP supply; no parameter can fix this" →
   comparator mislabeled and direction inverted; on the twin the repaired
   normalization is validated (+0.2%); the real finding is the cross-mock
   transport failure (~1.9–2.8σ, effectively one observation).
   Also: issue #30's body still carries a claim formally retracted on
   2026-08-05 (the +2519/+3478/+3089 "one dof" deviance verdict).

## 13. What remains reliable

The FP defect and repair; the closure failure (N_HI axis, ~3× tolerance,
all mocks); the mock-calibration bit-identity and its consequences; the
"resp_N_fit_range is a binning knob" finding; the guard-layer work; the
two-lineage discipline. The earlier kernel-support boundary measurements
(§4 of the 2026-08-05 checkpoint) were *not* re-reviewed this session.

## 14. What remains uncertain

The mechanism of the small signal-side SNR effect; whether the transport
failure is a property of loa-0 or of the single-ratio transport model; the
(1−η̄) placement; the clamp-convention bracket; the true prior-sensitivity
of any reported dN/dX (the 16–21% shape-freedom figure likely stands but
was not re-derived).

## 15. Scientifically distinct next options

A. Adopt calibration-noise-aware gate statistics (changes the ratified
   gate's meaning; makes every closure number honest about the 89-event
   template). B. Enlarge the loa-0 calibration (more absorber-free volume →
   directly shrinks the dominant FP uncertainty and tests transport).
C. Treat transport as a measured systematic (widen/refit the transfer
   prior). D. Sub-floor injections — still valuable, but re-founded: they
   probe the pad↔completeness degeneracy and the sub-19.0 promotion
   channel, not the (retired) absorber↔FP degeneracy. E. Attack the
   N_HI-shape misfit on the twin (the actual leading residual).
F. Drop quantitative sub-DLA inference from Paper 1 (unchanged option).

## 16. Recommendation and the decisions that are yours

Recommendation: **first correct the record (§12) and adopt the corrected
statistical accounting (A); then prioritize B and E over new model freedom;
keep D re-founded as above; hold F until A+B are assessed.** The rulings
needed from you: the gate variance model (A); the transport treatment (C);
the FP-shape treatment if any freeze is contemplated (in-range conditional
only); confirmation of the (1−η̄) restoration; whether injections (D) are
commissioned under the corrected rationale; and the manuscript question —
the current TeX describes the *previous* pipeline generation, so bringing
it to the current model is a scope decision, not a patch.

---

### Status table

| claim (2026-08-05) | review outcome |
|---|---|
| FP normalization defect + repair | **Upheld, strengthened** (twin +0.2%) |
| Not separately identifiable (A vs B) | **Rejected as stated** → pad is prior-identified; degeneracy is with completeness |
| Δdev=41 = 0.6σ, wrong model undetectable | **Rejected** (power ≈ 1; invalid yardstick) |
| Leading residual = SNR axis | **Rejected** (85–95% calibration noise; N_HI leads) |
| μ_FP exceeds mock FP supply | **Rejected as stated** → cross-mock transport failure only |
| Closure fails everywhere | **Upheld** (softened to ~3× gate, N_HI axis) |
| London/Saclay = transfer tests | **Upheld** (prediction transfer only) |
| Kernel support boundary [19.5, 21.1) | Not re-reviewed |

### The model in one line each

Predicted counts = (path length) × Σ over true-N bins [kernel × completeness
× population] + (transported loa-0 FP rate); survey counts and the 89
calibration counts are both Poisson; smoothness/shape priors supply what the
data do not. Full specification with per-term verification status:
`review_phaseA/MODEL_SPEC.md`.
