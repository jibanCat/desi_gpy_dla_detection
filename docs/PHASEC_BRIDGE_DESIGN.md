# Phase-C bridge-anchor design and acceptance criterion

**Status: FROZEN 2026-08-06, before any bridge data exist.** Companion to
`docs/PHASEC_CALIB_DESIGN.md`. The bridge tests whether the NEW
calibration pipeline (controlled injections → production finder →
contract matching → moment construction) measures the SAME scientific
quantity as the OLD one (natural matched pairs from the production
catalogue, `track_c/stage0/forward_response_2lpt0.npz`) inside the
region the old response is considered well measured. Per PI §7: a failed
bridge is a scientific result; nothing is spliced or smoothed over a
failure.

## 1. Bridge anchors [FRZ]

* **Locations:** the five 0.2-dex bins [19.5, 20.5) (b2–b6), i.e. the
  well-populated heart of the old anchor sets (old anchors 1–6 of 7 per
  cell lie in [19.34, 20.77]), PLUS the old TOP-anchor region [20.9, 21.1]
  which is shared with production bin b9 (dual-role: its bridge
  comparison uses the old top anchor; its pairs feed production).
* **Counts:** 150 pairs per bridge bin pooled over the 9 cells, allocated
  ∝ the old anchors' per-cell density, floor 8/(bin, cell) — sized so the
  per-bin paired-difference sd on the kernel mean ≈ 0.19/√150 ≈ 0.016 dex,
  giving the tolerance test below ≥ 90% power against a 0.05-dex estimand
  offset (2× the largest old per-cell mean bias in the bridge region).
* **Old cells to be reproduced:** all 9 (SR × ZR) cells; per cell the
  empirical anchor moments (mean, sd) of the old envelope's anchors
  falling in [19.5, 20.8], re-derived from the STORED per-anchor rho
  distributions (n ≈ 40 each), plus the top anchor.

## 2. What the bridge tests (dimensions) [FRZ]

Truth-generation consistency (natural vs injected truth); injection-path
consistency (flux-level imprint vs real absorption); matching definitions
and multiple-candidate rates (contract vs stage-0 convention — the rate
itself is a compared statistic); completeness normalization (injected
completeness vs molly surface in the overlap cells); response
normalization (unit row mass in-window); observed-N̂ migration (moment
agreement); z and SNR conditioning (per-cell agreement pattern);
old–new boundary compatibility (continuity of the stitched surfaces at
the old fit-range edge).

## 3. Bridge covariance treatment (§7.1) [FRZ]

Overlap structure, stated exactly:

* **Shared at the event level: NOTHING.** Old pairs = natural absorbers
  on their host sightlines; new pairs = injections on disjoint sightlines
  (the generator excludes truth-HCD hosts at the injection redshift and
  uses each sightline once). No shared truth systems, spectra, noise or
  continuum realizations. Paired differences are therefore NOT available;
  the comparison is between-sample.
* **Shared at the family level:** the same 2LPT-0 forest realization,
  pipeline binaries and finder configuration. Deterministic pipeline
  sharing is part of the ESTIMAND (wanted). Forest-family sharing is a
  common-mode systematic that cancels to first order in the old–new
  DIFFERENCE — but it is not assumed away:
* **C_bridge = C_old + C_new + Ĉ_shared**, with C_old from the stored
  per-anchor rho spreads (sd²/n_eff, n_eff ≈ 40; jackknife over the rho
  support), C_new from the injection pairs (empirical, per cell), and
  Ĉ_shared bounded empirically by splitting the new bridge pairs across
  disjoint healpix groups and comparing the between-group scatter of the
  old–new difference to its nominal (within-group) variance. The
  variance-ratio estimate (and its uncertainty) is reported; if the
  between/within ratio exceeds 1.5, the shared term is inflated to the
  measured ratio (conservative direction). Effective rank, overlap
  counts (= 0 events; family-level structures named), and sensitivity of
  the verdict to the defensible covariance range (Ĉ_shared ∈ [0, measured
  bound]) are all reported [FRZ].

## 4. Acceptance criterion [FRZ — chosen before any bridge data]

Let Δ_a = (new − old) anchor mean and ω_a = (new/old − 1) width, per
bridge anchor a (bin × cell where the old anchor exists), each
standardized by its C_bridge sd. The bridge PASSES iff ALL of:

1. **G3-projected difference:** |Σ_a w_a Δ_a| projected through the
   preimage sensitivity map onto predicted G3 counts is **< 75 counts**
   (half the production σ target — a bridge failure at the size of the
   production error budget must not be smoothed into the splice), AND the
   95% CI UPPER BOUND of that projected difference is **< 116.7 counts**
   (the binding production σ target).
   *[AMENDMENT 2026-08-06, PRE-DATA, PI to ratify: the original second
   clause — "its 95% CI does not exclude values < 75" — was VACUOUS (a CI
   always contains its point estimate; independent review finding F5).
   Replaced, before any bridge datum existed, by the real dispersion
   guard above: a bridge measurement too noisy to bound the estimand
   difference below the production error budget cannot pass. This is a
   TIGHTENING recorded in writing per the design's own amendment rule.]*
2. **No coherent offset:** the precision-weighted global mean-shift
   z = |Σ Δ_a/σ²_a| / √(Σ 1/σ²_a) < 3, and the same for widths.
3. **No localized break:** max standardized |Δ_a| < 4 (Bonferroni-aware
   for ~50 anchor-cells), and the multi-candidate-rate difference per
   cell < 3 σ (binomial).
4. **Completeness consistency:** injected completeness within 3 σ of the
   molly surface per overlap cell (Jeffreys intervals).

FAIL ⇒ stop, no splice, diagnose source (truth construction / injection
path / matching / conditioning / artifact definitions / normalization);
if resolving it changes the estimand or production definition → PI ruling
(§7). The tolerance was NOT selected to let the pipeline pass: it derives
from the production error budget (item 1) and standard multiplicity
practice (items 2–3), fixed at design time.

## 5. Transition and stitching rule [FRZ]

Only on a PASSED bridge:

* Per cell, refit the SAME degree-2 moment surfaces on the UNION of the
  old anchors (re-derived moments, weight ∝ n_eff = 40) and the new
  production+bridge anchors (weight ∝ n pairs), with `resp_N_fit_range`
  extended to the new top anchor. Degree does NOT increase [INH]; the
  clamp convention (`resp_clamp="both"`) is unchanged.
* **Lack-of-fit test:** per cell, χ² of anchor moments about the refit
  surface; if p < 0.01 in any cell (the quadratic cannot represent the
  widened range), STOP → PI ("further response structure" is a checkpoint
  C decision; not adopted unilaterally) [PI].
* **Interpolation:** within the extended fit range only (polynomial
  evaluation at clamped covariate outside — unchanged semantics). The
  no-extrapolation boundary moves from 21.04–21.22 up to the new top
  anchors (~22.3); BELOW 19.5 nothing changes (the pad region keeps its
  existing labels).
* **Continuity report:** old vs stitched surface maximum |Δmean| and
  |Δsd/sd| over the old fit range, per cell — must satisfy the §4
  tolerance recast on the surface (< 0.02 dex / < 5% except where the
  new data dominate by weight); violations → the bridge-failure path.
* Support labels after stitching: [19.5, 20.5) bridge-validated;
  [20.5, new top anchor] directly-measured (within covered conditioning
  strata); above the top anchor clamped; below 19.5 unchanged.
* The correlated old/new uncertainty in the stitched surface is carried
  by keeping BOTH anchor sets (with weights and provenance) in the new
  envelope — downstream uncertainty propagation resamples anchors, never
  the smoothed surface alone [FRZ].

## 6. Reporting [FRZ]

The bridge validation artifact reports: overlap counts (0 events;
family-level sharing named), C_old / C_new / Ĉ_shared construction and
effective ranks, per-anchor Δ and ω tables with CIs, the four criterion
outcomes, the G3-projected difference with CI, sensitivity across the
defensible Ĉ_shared range, the multi-candidate and completeness
comparisons, and the pass/fail verdict. Pilot-precision bridge numbers
are labeled engineering-validation; the production bridge verdict uses
production bridge pairs only.
