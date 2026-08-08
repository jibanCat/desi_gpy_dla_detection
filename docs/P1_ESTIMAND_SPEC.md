# P1 ESTIMAND SPECIFICATION — kernel anchor APPROVED; complete freeze PENDING mechanical gates

**Status (2026-08-07 PI ruling): the natural-pair kernel anchor
(C_molly, K_natural-pairs) is APPROVED as the selected P1 design
direction. The COMPLETE estimand freeze is NOT yet ratified** — final
ratification is conditional on: (1) explicit overlap/blend coherence
closure; (2) C/K parent-population compatibility; (3) atomic C/K
artifact construction; (4) the explicit miss state and the R = C·K
identity; (5) fail-loud provenance/estimand guards; (6) the frozen
healpix-jackknife gate. §15–§16 below record the gate results
(engineering phase, same ruling). Nothing is spliced into production,
the holdout stays unread, Stage-2B stays withheld, P2 is deferred
until after the holdout (PI sequencing decision).

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

## 15. BINDING overlap/blend coherence section (PI conditions §2.1–§2.5, 2026-08-07)

### 15.1 Parent population of K — exact

**K is constructed from ALL eligible production-matched natural pairs
— exactly the deployed completeness-numerator events** of the
canonical nhi195 chain: `is_TP & (N̂ > 19.5) & (P_DLA > 0.99) &
(DLAFLAG == 0)`, strict deployed cell bounds. It is NOT the isolated
subset: the isolation restriction (no catalogued ≥17.2 neighbor within
5,000 km/s) was and remains ONLY the frozen construction for the
injected transfer map (§7), because injections exclude that class by
placement. Blended pairs stay inside K; their composition is recorded
per bin in the artifact (`composition`).

### 15.2 C/K compatibility — verified, integer-exact

C_molly (≥19.5 columns) and K derive from ONE chain and ONE matcher
run (`t1a_reproduce_cmolly.py` chain; cache = its saved event tables):
same truth population (nhi195-floored, in-window), same live-support
definition, same z/N support, same finder catalogue and cut bundle,
same matching rule (`match_truth_to_cat_molly`, dz_rel = 0.01,
nhi_desc), same one-to-one assignment, same split/merge treatment.
**Proof is the load-time identity: per-cell kernel event counts equal
the pack's `molly_n_det` INTEGER-EXACTLY in all 56 ≥19.5 cells**
(build `f1eff35`, re-verified at every load). The
"C_all·K_isolated" incoherence the PI flagged does not arise: K is
all-class. A marginalization argument is therefore NOT needed.

**Decomposition finding surfaced by this closure:** the deployed
numerator excludes matched detections with N̂ ≤ 19.5 (the SUBFLOOR
class): 1,650 live events, 1,611 of them at [19.5, 20.0), five in
[20.5, 21.0), zero above 21.0. Under the estimand convention the
deployed C_live at [19.5, 20.0) is **0.7504**; the Tier-1 ledger's
0.800 counted subfloor detections as matches (a different, now
explicitly labeled object). The load-bearing region is unaffected.

### 15.3 Width excess — subset results (frozen rule, `p1_width_checks`)

Natural robust width vs injected, ratio (n):

| N | all (K parent) | iso5k | shell0 | no-nb-30k |
|---|---|---|---|---|
| [20.4,20.7) | 1.24 (10,589) | 1.16 (8,867) | 1.16 (7,670) | 1.16 (5,023) |
| [20.7,21.0) | 1.31 (6,726) | 1.21 (5,553) | 1.20 (4,830) | 1.22 (3,185) |
| [21.0,21.3) | 1.27 (3,700) | 1.17 (3,106) | 1.15 (2,720) | 1.17 (1,786) |
| [21.3,21.7) | 1.09 (1,720) | 1.03 (1,434) | 1.03 (1,240) | 1.01 (805) |

The excess PERSISTS in the isolated/shell-zero/no-catalogued-neighbor
subsets over [20.4, 21.3) → **catalogued-shell classes are ruled out
as a SUFFICIENT explanation.** Per the frozen interpretation rule this
does NOT exclude: sub-threshold neighboring absorbers, unresolved
multi-component structure, non-catalogued overlap, quickquasars
imprint complexity, or truth-side profile variation. All overlap is
NOT claimed excluded. The catalogued-blend contribution to the K
parent's width (~5–8%) is real and belongs inside the estimand.

### 15.4 Merge/split accounting — every truth counted exactly once

Matcher semantics (production object, documented): greedy one-to-one;
cat rows walked in DESCENDING N̂; each row claims at most one
still-unmatched truth on its TARGETID within dz_rel; tie-break by
min |N̂ − N_truth|; a claimed truth is never re-claimed.

| class | enters |
|---|---|
| one truth, multiple candidates (split) | strongest candidate → the K pair + C numerator; sibling rows are non-TP → purity/FP side ONLY (never K, never C) |
| multiple truths, one candidate (merge) | the candidate claims ONE truth (its K pair); other truths → miss (unmatched) |
| assignment competition | deterministic greedy above; losses land in miss (unmatched) |
| unmatched secondary truth | miss (unmatched) |
| duplicated/split catalogue entries | non-TP → purity side only |
| matched, N̂ ≤ 19.5 | miss (subfloor class, explicit) |
| matched, P_DLA ≤ 0.99 | miss (lowP class, explicit) |
| matched, DLAFLAG ≠ 0 | miss (flag class, explicit; count 0 in this build) |

**Exactly-once demonstration (enforced at build AND load):** per cell,
`n_det + subfloor + lowP + flag + unmatched == molly_n_tot`,
integer-exact in all 56 ≥19.5 cells; matched-truth keys verified
unique (no truth carries two TP rows); no negative class anywhere. A
merged candidate contributes exactly one K pair; multiplicity is not
part of the forward model and never enters twice.

### 15.5 Marginal vs multi-object response — the marginal operator CLOSES

The per-absorber marginal operator's R = C·K identity closes
integer-exactly (15.2/15.4) with the overlap classes explicitly
placed. No blend kernel, mixture model, event-level response or
support restriction is REQUIRED for closure. Should any future
accounting break this identity, the build/loader fails loudly and the
choice among those extensions returns to the PI (no auto-escalation).

## 16. Representation integrity + hidden-transition audit (PI §3)

Occupancy of the reporting grid (live support, 9 cells/bin;
measured = cell n ≥ 25):

* **[19.5, 21.7): 11 bins, ALL 9 cells measured in every bin**
  (marginal n = 8,658 … 549). Directly measured; no pooling.
* [21.7, 21.9): 3 cells measured / 6 sparse (marginal n = 207).
* [21.9, 22.1): all cells sparse; marginal n = 82 (marginal-inherited,
  flagged).
* [22.1, 22.5): marginal n = 22 / 8 — sparse even marginally;
  wide-error flagged; **no extrapolation beyond [22.3, 22.5)** and no
  scientific claim there.

Audit findings: NO parametric refit, NO clamp, NO smoothing beyond
linear interpolation between bin centers (which reproduces the bin
means identically at the centers — pairs-faithful by construction);
ONE source chain for the entire ≥19.5 support (no source-chain
transition); sparse-cell inheritance uses the LIVE N-marginal only,
flagged with inflated variance (frozen rule in the artifact
provenance); the de-clamped shape is carried directly by the data
(+0.053/+0.052/+0.053 through [20.7, 21.3) → +0.034/+0.012 at
[21.3, 21.7) → +0.017/+0.005 above — vs the deployed clamp's ≈+0.05).
**No hidden estimand or representation transition exists.** The
paper-facing restriction stands: no scientific claim below N = 19.7;
formal artifact support to 19.5 does not close the low-N boundary.

## 17. Authoritative status (PI §7, 2026-08-07)

Tier-2 attribution: imprint-supported; tested environmental channels
excluded or bounded; uniqueness not claimed. Natural-pair kernel
anchor: APPROVED as the selected P1 design direction. Complete
estimand freeze: NOT yet ratified — pending PI review of §15–§16 and
the pre-read checkpoint. Injected response: validation and diagnostic
object only; not a production kernel source. Holdout framework:
ratified in principle; still sealed; battery frozen
(`p1_holdout_battery.json`). P2: deferred until after the holdout.
Stage-2B: withheld. G3: not resolved. Low-N boundary: not closed.

## 18. TWO-LAYER RATIFICATION STRUCTURE (PI amended ruling, 2026-08-07)

### 18.1 High-N primary operator

**Support: N_true ≥ 20.3** (a deployed molly cell edge; the committed
preimage puts 99% of the G3 feed at true [20.3, 21.7)). Content: §1–§16
unchanged — (C_molly, K_natural-pairs), all-pair parent, explicit miss
states, exact merge/split accounting, no clamp, no hidden transition,
atomic guards. **Below-floor migrants are measured at ZERO events in
the observed G3 window, 0.60% (144 ± 12 events) in observed G2, 0.83%
at observed [20.3, 20.7)** (`p1_migration.json`) — carried as an
explicit measured source term, negligible for G3, stated for G2.

### 18.2 Low-boundary transport extension (truth < 20.3 + below-floor)

A separate conditional-support problem. Measured content:
* **Below-floor inflow (N_true < 19.5 → N̂ > 19.5):** f_net = 22.7% of
  the selected catalogue at observed [19.5, 20.0), 15.2% at the 19.7
  reporting floor, 4.1% at [20.0, 20.3). Chain-compatibility bridge
  PASSED first (`p1_chain_bridge.json`): truth common support
  identical; 4 competition reassignments of 10,687 (excluded); one
  0.13% catalogue-cut difference reported separately. Representation:
  the primary operator keeps truth ≥ 19.5; the inflow is an EXPLICIT
  source term; **K is never renormalized to hide it.**
* **Emission-proximity dependence** (frozen regions,
  `p1_emission_proximity.json`): K mean near quasar Lyα emission is
  −0.083…−0.093 dex vs interior at every N (z −18…−32) — largely
  COMMON-MODE with injections (−0.056, z −7.7; pipeline-mechanical);
  low-N completeness deficit near Lyα em (0.517 vs 0.773 at
  [19.5, 20.0)), confined to low N. Truth-vs-pair mixture shift on
  the marginal K: ≤ 0.003 dex everywhere — below materiality ⇒ the
  MARGINAL kernel remains valid for the within-realization fold; the
  frozen conditional table (kernel_by_region) is REQUIRED for any
  cross-mixture transport. LYA_EM and physical quasar proximity are
  congruent by construction near Lyα (stated, never collapsed);
  EDGE is a separate flag.
* Failures here map to support restriction / migration systematic /
  conditional transport uncertainty / no-claim — never to automatic
  high-N rejection (frozen exception: implied high-N contamination
  > 50 G3-equivalent counts, or a simultaneous primary joint failure).

### 18.3 Authoritative completeness nomenclature (binding)

* **`C_fm` — finder-matched completeness:** truth matched at the
  operating point regardless of the N̂ floor (the Tier-1 ledger
  object; 0.800 at [19.5, 20.0) live).
* **`C_paf` — production-above-floor completeness:** the deployed
  estimand's C; matched AND N̂ > 19.5 AND P_DLA > 0.99 AND
  DLAFLAG = 0 (0.7504 at [19.5, 20.0) live; the artifact's
  `C_molly_n_det/n_tot`).
These names are mandatory in artifacts, docs, figures,
manuscript-facing results and checkpoints; never interchangeable.

### 18.4 Joint (C, K) calibration covariance

Frozen construction (`p1_joint_cov.json`): whole-healpix delete-one
jackknife, same realizations for C and K; corr(C, K) = −0.04…+0.04 in
every battery bin — **not material**; ESS 242–896 blocks; max block
share ≤ 1.2%; jackknife/naive SE ratios 0.98–1.06. Full G1/G2/G3
covariance propagation = a refold-stage deliverable (before the gated
refold, per the ruling).

### 18.5 Hierarchical holdout verdict (battery v2)

`p1_holdout_battery_v2.json` (supersedes v1, retained): primary
high-N family (5 mean bins ≥ 20.4 + pooled [20.7, 21.1) +
completeness + width-diagnostic + joint-operator; Holm α = 0.01)
ALONE decides the P1 predictive verdict. Low-boundary family (2 low
bins + subfloor rate + LYA_EM region test + low completeness; Holm
α = 0.01) maps only to its own outcomes; below-floor migration is
recorded as NOT holdout-testable (no sub-19.5 injections exist) and
is adjudicated development-side. Exploratory subgroups are
uncorrected, labeled, and cannot reject, tune, or be promoted. The
global verdict enumeration and gatekeeping are frozen in the JSON —
**high-N operator validity ≠ low-boundary transport validity.**
