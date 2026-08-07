# P1 ESTIMAND SPECIFICATION — PROPOSED FREEZE (for PI review)

**Status: PROPOSED. Nothing here is self-ratified.** This document is
the estimand freeze the PI's binding sequence calls for (P1 rulings
§25/§33; handoff "Exact next step"). It becomes FROZEN only on PI
acceptance at the Tier-2 attribution / P1-design checkpoint. Until
then: no K artifact is built, nothing is spliced into production, the
holdout stays unread, Stage-2B stays withheld, P2/P3 are not begun.

Inputs: Tier-1 (`t1_findings.md`, C_molly reproduced integer-exact;
live-support completeness; attrition taxonomy), Tier-2
(`t2_findings.md`, `t2_completion.md` incl. the post-specified power
addendum, `t2_pairing.json`, `t2_power.json`), the Stage-2A verdict
(`PHASEC_STAGE2A_BRIDGE_VERDICT.md`), and the preimage sensitivities
(`diagnostics_phaseC/preimage/`).

## 1. The estimand (definition)

The **deployed-pipeline detection-conditioned joint operator** on
mock-0 loa-124's live parent population:

    R(o | N_true, x),   o ∈ {miss} ∪ {N̂}

* **C(N,x) = P(o ≠ miss | N, x)** — the deployed C_molly (two-chain
  splice, ≥19.5 from the nhi195 chain, <19.5 from the 17.2 chain),
  reproduced integer-exactly in all 96 cells (Tier 1). UNCHANGED.
* **K(N̂ | N, x, o ≠ miss)** — the NATURAL matched-pair kernel,
  measured from the SAME matched event set whose counts form C's
  numerators (production matcher `match_truth_to_cat_molly`,
  dz_rel = 0.01, nhi_desc; P_DLA > 0.99; DLAFLAG = 0; live support).
* **Identity: R = C·K must close on that event set by construction**;
  the artifact build carries an executable closure test.

(C, K) are jointly conditioned by one detection/matching process —
this is exactly the coherent pair the rulings require; the estimand is
the one production has always used. What changes is only the kernel's
REPRESENTATION (§5): the deployed degree-2 + edge-clamp surface
misfits its own calibration pairs (+0.071 low edge / −0.035 mid /
−0.030…−0.044 clamp region, 4–7σ) and is replaced by a pairs-faithful
representation of the same pairs.

## 2. Parent population (denominator)

In-window truth of loa-124 on **live support**: S2N_RED > 2, z_qso in
the production window, absorber z inside `analysis_window(z_qso)` —
the fold's op-mask, decided in Tier 1. The S2N_RED ≤ 2 strata (47.0%
of raw in-window truth, dX = 0 in the fold) are class-1: outside the
analysis, recorded, never part of any P1 number. The natural-truth
near-neighbor class (≤5,000 km/s truth HCD neighbor, 6.5–7.1%) is
INSIDE the parent population and inside K's pair population (§6).

## 3. Miss state

Explicit: P(miss | N,x) = 1 − C(N,x). Diagnostic sub-states recorded
from the Tier-1 ledger (H1 no-candidate / H3 sub-threshold / H5
assignment / H2 bundle; H4 tolerance = 0). Sub-states never
renormalize C silently; they are labels on the miss mass.

## 4. Conditioning set and support map

* x = (z-cell ∈ {<2.56, 2.56–2.96, ≥2.96}) × (SNR stratum ∈
  {≤3.5, 3.5–6.5, >6.5}) — the frozen 9-cell stratification.
* Kernel N-grid: 0.2-dex bins over **[19.5, 22.5)**; reporting/gate
  bins 19.5/20.0/20.4/20.7/21.0/21.3/21.7/22.4 as committed.
* Fold window and groups unchanged (ratified [19.7, 21.6], G1/G2/G3
  edges 19.7/20.3/21.0/21.6).
* **No claims below 19.7; no extension into 19.3–19.5** (rulings).
* Support map ships INSIDE the artifact with per-bin n, SE, width;
  bins below minimum occupancy are flagged and inherit the N-marginal
  with inflated variance — never extrapolated by a parametric form.

## 5. Kernel representation (the fix that motivated Phase C)

Pairs-faithful, non-parametric: per (N-bin × cell) pair mean + width
(+ SE), linear interpolation of bin means in N inside the populated
range; **no polynomial refit; no edge clamp; no extrapolation beyond
the last populated bin** (above it, the last bin's value with flagged,
inflated variance). Rationale: the Stage-2A high-N finding and Tier-2
Level-A capstone show the true mean-bias FALLS +0.055 → ≈+0.01
through [21.0, 22.1) while the deployed clamp holds ≈ +0.05 — a shape
no low-order polynomial + clamp represents; the natural pairs measure
it directly at production precision (2,787/1,629/792/345/156/30 pairs
per bin above 21.0).

## 6. The kernel-anchor decision (the ONE pick, justified)

**PICKED: natural-pair K across the full support.** REJECTED:
injected K + the measured N-rising mean correction. Reasons:

1. **Coherence** — (C_molly, K_natural) is the jointly-defined pair;
   the rulings state a new K cannot combine with the old C. An
   injected K would import the injected campaign's OWN conditioning
   and need the natural-anchored correction anyway (circular).
2. **Width** — the natural kernel is 15–25% WIDER than the injected
   one at every bin (t2_power R3); a mean-only correction would carry
   a width error worth ~50–150 G3 counts through the committed width
   sensitivities.
3. **Attribution robustness** — Tier-2's verdict is imprint-supported
   (environment-flat) with the catalogued-neighbor channel excluded at
   ~15× margin and the near-field channel mechanically bounded
   (≤0.007 dex per 1σ population shift, wrong sign). But the freeze
   does NOT need the attribution to be unique: **whatever mechanism
   produces the natural−injected offset is inside the natural pairs
   and therefore inside K by construction.** The residual ambiguity
   attaches to the injected campaign's uses only (§7).
4. **Precision** — live natural pairs: 25,895/19,174/17,323/4,871/868
   per Tier-1 range; per-bin kernel-mean SE 0.003–0.011 dex ≈
   production-grade.

Consequence to state plainly: this is a within-realization (mock-0)
natural remeasurement. **Realization independence is the ONE thing it
does not buy** — that is P2's content (~1,500 CPU-h, mock-1), a
separate PI decision; nothing here pre-empts it.

## 7. Role of the injected campaign (transfer rule)

The injected campaign (bridge + production-calibration roles) serves
validation, never production K:

* **Completeness cross-check:** live-support C_inj ≈ C_molly within
  ~2 points (Tier 1) — a standing check, re-verified on the holdout.
* **Frozen transfer map (injected → natural), for validation and the
  holdout only:** mean offset by N (D1: +0.0177/+0.0249/+0.0454/
  +0.0376 ± 0.006–0.008 over 20.4–21.7) and width ratio (R3). Never
  spliced into production; never used to "correct" natural pairs.
* **Overlap validation** on [20.4, 21.1] (both selections ≥95%
  complete; injected anchors dense): the JOINT-operator comparison of
  the rulings — per-truth-class P(land in group g, incl. miss) — not
  a K-only bridge; no acceptance via total-count cancellation.

## 8. Blend composition term

Natural pairs with a catalogued 17.2–19.5 neighbor ≤3,000 km/s are
7.5–8.0% of pairs with dx elevated +0.03…+0.10 dex. They are part of
the production estimand and STAY inside K's pair population; the
class fraction and elevation ship in the artifact as a composition
diagnostic. Injected comparisons (§7) use isolated naturals only —
the frozen Tier-2 discipline — because injections exclude the class
by the 5,000 km/s placement rule.

## 9. Transition statement

**There is no estimand transition in this design.** One coherent
(C, K) pair covers the full support; the deployed↔proposed difference
is representation only. [20.4, 21.1] is a VALIDATION overlap (chosen
by the measured completeness saturation of both selections and the
injected anchor density — not by the 20.4 crossing per se), frozen
here, never re-chosen after any closure viewing.

## 10. Versioning and guards (rulings §25)

Estimand ID `p1_natpair_ck/v1`. C and K artifacts are generated
atomically by one committed builder run, share the estimand ID +
version + support map, and fail loudly on any mismatch. The fold's C
path is byte-unchanged (C_molly as deployed); the K artifact's loader
refuses: a K without the matching estimand ID; any use that would
apply completeness twice or renormalize the miss state; any mixed
version. Guards land WITH the builder, with tests, before any
evaluation.

## 11. Stability requirements (pre-holdout, part of the freeze)

Before the holdout gate can be evaluated: (i) the R = C·K identity
test passes on the calibration event set; (ii) a whole-healpix
jackknife of the natural-pair kernel demonstrates per-bin mean
stability consistent with its quoted SEs (no single-healpix
domination); (iii) the artifact reproduces the Tier-2 pair tables
bit-consistently from the committed cache.

## 12. What PI acceptance authorizes — and what it does not

Authorizes: building the K artifact + guards + stability checks
(§10–§11), then evaluating the FROZEN holdout gate
(`P1_FAILURE_TAXONOMY.md`) on design-side terms, and ONLY if that
gate says "open" — the one-time holdout read.

Does NOT authorize (each needs its own gate/ruling): the refold /
G1-G2-G3 prediction (after holdout pass only); any production splice;
Stage-2B FP launches; P2; P3; any claim that G3 is resolved or that
the low-N boundary is closed.

## 13. Design-screen answers (stopping rule §4)

* Parent population: **decided** (§2, live support).
* Completeness denominator: **decided** (§2; deployed C_molly).
* Miss-state definition: **decided** (§3).
* Conditioning set: **decided** (§4).
* Transition support: **decided** (§9 — none; validation overlap
  frozen).
* Holdout criterion: **proposed-frozen** in `P1_FAILURE_TAXONOMY.md`
  (power computed design-side in `p1_holdout_gate.json`).

## 14. Honest-record notes

* Natural pairs were never blind within mock-0 (they are the
  production catalog; every Phase-B closure number already used the
  full footprint). The holdout blinds INJECTION outcomes only; the 13
  holdout healpix's natural rows appear in Tier-1/2 aggregates by
  construction. The natural kernel's out-of-sample adjudication is
  therefore: injected-holdout overlap (§7) + jackknife (§11) within
  mock-0; realization independence only via P2.
* The Tier-2 attribution wording remains: consistent with
  imprint-realism differences; disfavors the TESTED environmental
  explanations; uniqueness not claimed. The freeze is robust to that
  residual ambiguity (§6.3).
* Stage-2A criterion 4 stays retracted as physics; criteria 1–3 stand.
