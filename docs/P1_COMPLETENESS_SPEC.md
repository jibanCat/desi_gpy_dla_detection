# P1 / natural-match completeness investigation — SPECIFICATION

**Branch `calibration/phaseC-p1-coherent-ck-2026-08-06` (root `60cef40`).**
Implements the PI's P1 rulings (2026-08-06, verbatim in the notes repo,
`notes/2026-08-06_phaseC_p1_rulings.md` @ `e3930cb`). Binding sequence:
bounded completeness investigation → P1 design freeze → failure taxonomy +
holdout adjudicability gate → one-time holdout → P1 verdict. The 25%
whole-healpix holdout of `prod_v1` stays UNREAD throughout this
investigation; the quarantined K-only artifact stays quarantined; the main
Stage-2B FP spend stays withheld; P2/P3 are not begun.

## The question

Why does natural matching retain ~43–58% of eligible truth systems where
the injection pipeline retains ~81–99% — decomposed by mechanism with
evidence levels, sufficient to freeze the P1 joint-operator design
(parent population, denominators, miss state, conditioning, transition,
holdout criteria) or to establish that injections cannot represent the
dominant natural selection process.

## Tier structure (sequential; rulings §5)

* **Tier 1 — deployed accounting and mechanical attrition:**
  (t1a) event-level reproduction of the EXACT deployed C_molly
  (pack `molly_n_det/molly_n_tot`) from the immutable production
  catalogue + truth via the committed recipe, bin-by-bin, BEFORE any
  interpretation; (t1b) frozen natural-denominator eligibility ledger +
  the injection-placement/denominator audit (deterministic REPLAY of the
  frozen generator seed with full per-proposal logging — no new
  generation; the replay is a ledger of what the frozen campaign did);
  (t1c) event-level attrition ledger with one primary mutually-exclusive
  cause + secondary flags + order-sensitivity bounds; (t1d) the
  C_candidate × C_threshold × C_cuts × C_matcher decomposition with
  nearest/highest-scoring pre-threshold candidate records for every
  unmatched truth.
* **Tier 2 — selection conditioning:** survivor tests (matched vs
  unmatched naturals vs injections per stratum), threshold-proximity,
  paired/common-substrate comparisons, common-support restriction,
  prespecified reweighting on PRE-selection covariates only
  (post-selection variables diagnose, never balance — rulings §16).
* **Tier 3 — natural-profile/environment:** only if the frozen stopping
  rule is unmet after Tiers 1–2.

## Data sources (immutable; no finder reruns, no new injections)

* Production catalogue: `gl_prod_2lpt0_v1_20260526/combined_catalog/`
  (all finder rows incl. sub-op-threshold candidates, DLAFLAG, sentinels).
* Truth: loa-124 `hcd_truth_cat.fits`; sightlines: `zcat`/`snr_cat`/
  `bal_cat`; deployed molly artifact: the frozen Phase-B pack fields +
  `ff_fp_cache/molly_counts_2lpt0_lyaonly195.npz` + the TSVs named in the
  pack provenance.
* Injection side: the frozen `prod_v1` manifest/roles (+ the committed
  generator at the frozen seed for the placement replay), pilot arms
  (engineering), the Stage-2A pairs JSONs. NO holdout-healpix rows are
  read in any analysis (role enforcement + an explicit healpix guard in
  every Tier script).
* Matcher: THE production object (`match_truth_to_cat_molly`,
  dz_rel=0.01, nhi_desc). Shadow-matching counterfactuals (§14) run this
  object or clearly-labeled variants on immutable inputs.

## Candidate-stream fidelity (rulings §8)

All Tier-1 attribution uses the immutable deployed catalogue (Level A).
No instrumented finder rerun is planned; if one ever becomes necessary it
requires a frozen instrumented state + behavioral-equivalence demo and is
labeled counterfactual, never Level A.

## Resource statement (rulings §21, recorded BEFORE Tier 1)

Expected compute: ≤ 10 CPU-h total (catalogue reads ~770k rows ×
few passes, matcher replays, generator replay, stratified analyses) on
the login node in short interactive steps (<30 min each) or one small
sbatch if any step exceeds that. Storage: ≤ 2 GB derived ledgers/JSON
under `diagnostics_phaseC/p1_completeness/` + scratch caches. Finder
reruns: NONE. Forced-fit campaign: NONE planned (Tier 2 uses existing
sub-threshold catalogue rows; if a forced-fit need emerges it goes back
to the PI as a separate budget). New simulations/injections/mocks: NONE.
This spend is inside the ~1,850 CPU-h envelope's diagnostic margin and
does not draw on the withheld FP allocation.

## Evidence classification (rulings §19)

A = directly observed accounting loss; B = controlled counterfactual;
C = paired/common-substrate; D = covariate-adjusted association;
E = residual inference. "Root cause" is reserved for A/B.

## Deliverables

Per-mechanism gap decomposition in the frozen N ranges (19.3–19.5 where
available, 19.5–20.0, 20.0–20.4, 20.4–21.0, 21.0–21.5, >21.5) with
evidence level, stability/order bounds, residual unexplained gap, and
the projected consequence for the P1 joint operator; then the §22
consequence report (denominator equivalence, placement selection, miss
state, conditioning, common support, transfer justification, transition
support, P2 likelihood). The stopping rule lives in
`docs/P1_STOPPING_RULE.md` and was frozen before any aggregate figure
was inspected.
