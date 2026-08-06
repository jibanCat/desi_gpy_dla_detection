# Phase-C high-N response calibration — DESIGN SPECIFICATION

**Status: FROZEN-BEFORE-PRODUCTION except where an item is explicitly
labeled pilot-adjustable or PI-gated.** Written after the committed
preimage analysis (`diagnostics_phaseC/preimage/`, commit `d07c4ff`) and
BEFORE any injection data exist. Per PI rulings 2026-08-06 §8: no
production anchor placement, endpoint definition, bridge tolerance or
stitching rule may change after production closure is seen. The bridge
half of the design is `docs/PHASEC_BRIDGE_DESIGN.md`.

Classification labels used throughout: **[INH]** inherited (ratified),
**[PHY]** physically required, **[PBM]** Phase-B motivated, **[PA]**
pilot-adjustable, **[FRZ]** frozen before production, **[PI]** PI-gated.

## 1. Scientific purpose and frozen context

The model fails the Layer-B gate on all three mocks through a smooth
observed-N̂ tilt with the leading violation in G3 [21.0, 21.6]
(+450.25 counts, z = +5.93 on the twin). The frozen Phase-B conclusion
attributes this to a response-model deficiency or calibration gap in a
materially contributing true-N region not directly measured. This design
turns that region into a direct measurement. It adds NO model freedom:
the deliverable is a measured calibration artifact, not parameters.

The preimage analysis established [PBM]:

* 99% of G3's predicted signal comes from true N ∈ [20.3, 21.7);
  12.7% migrates up from below 21.0, 3.3% down from above 21.6.
* 47.3% of G3's μ sits on covariates CLAMPED above the top anchors
  (21.04–21.22); the nominally measured [20.6, 21.1] band hangs on ONE
  ~40-pair anchor per cell.
* Sensitivity peaks at true [20.7, 21.1): 14,300 counts/dex; the +450
  discrepancy ≡ +0.031 dex coherent mean bias there, or ≈ −0.13 dex over
  the clamped [21.1, 21.5) (which also reduces the 21.7+ over-run), or a
  mixed mean/width change of the same scale. These set the §9 effect size.

## 2. Estimand [FRZ]

* **Response estimand:** the deployed-pipeline response — the conditional
  law of (detection, N̂) given a true absorber (N_true, z_true) on a
  production sightline of given SNR, under the PRODUCTION finder
  configuration (the same GP config, priors and model files as the
  `gl_prod_*` runs; no FILTER-off topology, no finder modification) and
  the PRODUCTION op-mask (S2N_RED > 2 strict, P_DLA > 0.99 strict,
  DLAFLAG == 0, BAL veto, z_QSO ∈ (2, 4.25), λ_rf ∈ [1025, 1216]).
  Kernel representation: the existing `skewnormal_per_cell` moment
  construction (same envelope schema as
  `track_c/stage0/forward_response_2lpt0.npz`), per-cell empirical anchor
  distributions → moment surfaces of UNCHANGED polynomial degree 2 [INH];
  fit range per cell extended to the new top anchor. Observed axis runs
  to the full pack grid top 22.4 (NOT truncated at the 21.6 reporting
  ceiling) so ceiling migration is measured on both sides [PHY].
* **Completeness estimand:** P(detection passing the op-mask | injected
  truth) per molly cell (same `molly172`/`molly195` cell definitions)
  [INH]; reported per stratum alongside the response.
* **Old–new relation:** same estimand as the old measurement (natural
  matched pairs from the production catalogue) up to truth-generation
  differences; the bridge (separate doc) tests exactly that equivalence.
* Mock family: 2LPT-0 [INH] — the signal-side calibration stays frozen on
  2LPT-0 as ratified; cross-family transport remains Layer C and is NOT
  re-opened by this campaign.

## 3. Production anchor design

* **True-N support [FRZ]:** production anchors at every 0.2-dex basis bin
  over **[20.5, 21.9) dense + [21.9, 22.4] thin** (bins b7–b13 + b14–b15).
  Support floor 20.5 = the 99% G3-feed edge (20.3) minus nothing — the
  [20.3, 20.5) bin itself is covered by the top BRIDGE anchor (see bridge
  doc), so the union of bridge+production anchors covers [19.5, 22.4]
  with no gap; margin above the ceiling to 22.4 covers the measured
  downward migration (189.5 counts into G3 from true > 21.6) [PHY].
* **Anchor density [FRZ locations / PA counts]:** density peaks at
  [20.5, 21.3] per the sensitivity map, NOT at the highest N (hard
  constraint §23: anchors must not sit only above 21.05). Per-bin pair
  counts from the committed sizing (`diagnostics_phaseC/design_sizing/`,
  Neyman-allocated, floors applied): b7 108, b8 238, b9 359, b10 108,
  b11 108, b12 108, b13 108, tail 60+60 — TOTALS 1,257 production pairs.
  Counts are [PA] upward or reallocated by the pilot's measured yields
  and variances; the CRITERIA they must meet (§7 below) are [FRZ].
* **Conditioning stratification [FRZ]:** every production bin is measured
  in ALL 9 response cells (z edges [0, 2.56, 2.96, ∞] × SNR edges
  [2, 3.5, 6.5, ∞]), pairs allocated ∝ the cell's G3 share (4.6–20.7%)
  with a floor of 12 pairs/(bin, cell). Within a cell: injection z drawn
  ∝ production dX(z) of that cell; host-sightline SNR drawn from the
  production SNR distribution within the stratum; fine-SNR coverage of
  the extreme strata [2,3) and [7,∞) verified explicitly (they carry the
  largest G3 shares). Noise, continuum, wavelength coverage and masking
  are the substrate's own (injection preserves them) [PHY].

## 4. Truth construction and injection path

* **Mechanism [INH]:** the committed, round-trip-validated
  `gpy_dla_detection/inject_absorber.py` (M4 validation < 0.5% EW) — flux
  × Voigt transmission on the GP's own grid convention; the finder is
  NEVER modified; injection is input preprocessing only.
* **Substrate [FRZ definition / PA mechanics]:** 2LPT-0 production
  coadds. A sightline is eligible for injection at z_inj if it passes the
  production op-mask preconditions and carries NO truth HCD within the
  matching window of z_inj (unambiguous truth ownership); sightlines with
  truth HCDs ELSEWHERE on the forest remain eligible (production-like
  environments). One injected system per sightline [FRZ] (multi-injection
  confounds matching; close-pair response is out of scope [PI]).
* **Environment-consistency sub-arm [FRZ]:** 10% of production
  injections duplicated onto fully-clean sightlines (no truth HCD
  anywhere) to measure environment sensitivity of the response — a
  consistency probe, NOT a second calibration; role-labeled and excluded
  from the production artifact.
* **Truth manifest [FRZ]:** every injection carries
  (TARGETID, z_inj, N_inj, cell, role, seed, healpix) in a validated
  manifest (the M3 MANIFEST schema); roles frozen at generation time.
* **Matching [FRZ]:** `matching_contract.py` P1–P6 fail-closed
  classification against the injection manifest; the velocity/N window
  convention IDENTICAL to the old (track_c stage-0) natural-pair
  matching; every injection resolves to exactly one slot (detected-
  matched / missed / multi-candidate per contract rule). Multi-candidate
  treatment: the contract's pinned order; the multi-candidate RATE is a
  reported pilot output and a bridge consistency dimension.

## 5. Roles, independence, holdout [FRZ]

Roles (frozen at generation, carried in the manifest, checked by code):
`pilot-validation`, `bridge`, `production-calibration`,
`environment-probe`, `held-out-evaluation`.

* **Holdout:** 25% of production injections, assigned by WHOLE HEALPIX
  blocks at generation, untouched until the frozen-statistic evaluation
  (Phase C3). No quantity derived from them may enter the calibration,
  the bridge, or any tuning decision.
* **Independence:** injections spread over ≥ 40 distinct healpix; ≤ 2
  injections per sightline-family (a sightline is used at most once per
  role); seeds recorded; effective-sample accounting (shared-sightline,
  shared-healpix) reported with the artifact.
* Pilot data are engineering data: they never enter the production
  artifact and are never reinterpreted as confirmation (§11 ruling).

## 6. Artifact and support labels [FRZ schema]

The new calibration ships as a versioned envelope
(`forward_response_2lpt0_phaseC.npz` + provenance JSON) containing: the
per-cell anchor sets (bridge + production), empirical rho distributions,
moment surfaces (degree 2), per-cell fit ranges, per-(cell, N-interval)
**support labels** from the frozen vocabulary {directly-measured,
bridge-validated, interpolated-within-support, transferred, extrapolated,
unsupported, clamped}, joint-conditioning metadata (which (N, z, SNR)
boxes the label covers), anchor-role metadata, seeds, and the old–new
transition record. Fail-loud loaders: production code refuses an
artifact whose support labels do not cover its evaluation point
(no silent extrapolation) [PHY]. A region is labeled directly-measured
ONLY within the jointly covered conditioning strata and only after
passing the §21 validation ladder (truth validation, matching
validation, normalization closure, bridge consistency, precision,
power, schema validation).

## 7. Precision and power criteria [FRZ — may not be weakened by pilot or production]

Primary endpoint: observed G3 [21.0, 21.6], planning discrepancy 450.25
counts (twin closure table @ `df29c78`).

1. σ(response-induced predicted G3) ≤ 450.25/3 = **150.1 counts**.
2. Power ≥ **0.90** to distinguish the current response from a
   perturbation explaining the full G3 discrepancy, at two-sided
   α = 0.01 on the G3-projected difference ⇒ σ ≤ **116.7 counts**
   (the binding criterion). Assumed effect size: the preimage
   equivalents (+0.031 dex at [20.7, 21.1); −0.13 dex over [21.1, 21.5);
   width analogues) — all of the SAME G3-projected magnitude, so the
   projected test is sensitive to each.
3. Bridge consistency within the frozen covariance-aware tolerance
   (`PHASEC_BRIDGE_DESIGN.md`).

Sizing model and achieved plan (committed:
`diagnostics_phaseC/design_sizing/sizing.py|json`): per-bin independent
measurement, σ(mean) = sd/√n, σ_frac(width) = 1/√(2n), sd from the
current response [PA: pilot re-measures], completeness from the pack
molly surface [PA]. Achieved at the frozen counts: **σ(G3) = 112.0
counts, power = 0.926**. Implied (reported, non-driving): σ(G1) = 618,
σ(G2) = 445 counts. Sensitivities to tails, z- and SNR-stratification
and transition rules: the per-bin/per-cell tables in `sizing.json` and
the preimage conditioning decomposition; the per-bin tail beyond ±3 sd
is NOT constrained at these counts (stated limitation — tail behavior
enters through the width/skew moments only) [PBM].

If the pilot's measured variances make these criteria unattainable at
reasonable cost: report achievable precision and STOP for the PI (§9);
do not weaken the criteria.

## 8. Compute, storage, Stage-2 authorization

* Unit costs (measured): 167 CPU-s/injected spectrum (loa-124 production
  log; wall1 pilot gate re-confirms) [PA: pilot re-measures]; generation
  minutes/arm; extraction 21 s/mock.
* Projected production campaign: 2,533 injections (incl. completeness
  and 15% retry allowance) ≈ **118 CPU-h finder + pilot ~25 CPU-h +
  margin ≈ 150 CPU-h total**; storage ≈ 8 GB scratch (spectra trees +
  catalogs + manifests). Job shape: standard `launch_gl.sh` window-loop,
  ≈ 40 healpix jobs, wall ≈ 2–4 h on cavestru0.
* **Stage-2 authorization status: NOT AUTHORIZED.** No documented
  PI-approved compute envelope for Phase C exists in the repo, the notes
  repo, or the rulings (checked 2026-08-06); the rulings' default rule
  therefore applies: complete C1, return this budget, do not launch.
  (The standing project rule — sbatch > ~500 CPU-h needs PI sign-off —
  is a NECESSARY not sufficient condition and does not constitute the
  §12 envelope even though this campaign sits below it.)

## 9. Prohibition boundary (§5) [FRZ]

This campaign REUSES the committed injection MACHINERY (`injection/`
package, `inject_absorber`, manifest schema, clean-table builder) that
the M3/wall1 work built and validated. It is NOT the prohibited pad–FP
identifiability campaign: no dense low-N (< 19.5) grid, no pad–FP
identifiability objective, no population inference, no LLS scope. The
lowest anchor of ANY kind is the 19.5 bridge anchor. Any proposal to
inject below 19.5 requires a new PI ruling. The independent reviewer
checklist (§21) includes verifying this boundary in the generator
config.

## 10. What the pilot may and may not change

Pilot-adjustable [PA]: yields, detection rates, runtimes, storage,
failure/retry rates, empirical variances and covariances used in the
sizing, pair counts needed to meet the frozen criteria, mechanical
implementation (file routing, job shapes, healpix selection), repair of
malformed truth construction / job config / deterministic matching bugs
/ schema bugs.

NOT changeable without a written amendment + PI approval [PI]: the G3
endpoint; the effect size; criteria 1–3 of §7; the decision rule; the
calibration/evaluation separation; the estimand (§2); the reporting
interval; anchor LOCATIONS; the bridge tolerance; the transition rule.

## 11. Choice classification table

| choice | class |
|---|---|
| adopted config / window / basis / clamp / molly | [INH] |
| response representation (skewnormal_per_cell, degree 2) | [INH] |
| production finder config as the estimand | [PHY] |
| observed axis to 22.4 (ceiling migration measured) | [PHY] |
| anchor support [20.5, 22.4] + bridge [19.5, 21.1] | [PBM]+[PHY], [FRZ] |
| density peak at [20.5, 21.3] | [PBM], [FRZ] |
| 9-cell stratification + within-cell allocation | [PHY], [FRZ] |
| substrate eligibility + one-injection-per-sightline | [FRZ] |
| environment sub-arm 10% | [PBM], [FRZ] |
| matching = contract + old window convention | [INH], [FRZ] |
| roles + 25% healpix holdout | [FRZ] |
| precision/power criteria (§7) | [FRZ] (PI §9) |
| pair counts per bin/cell | [PA] |
| unit costs, variances | [PA] |
| degree-2 lack-of-fit escalation | [PI] |
| any anchor below 19.5 | [PI] (prohibited absent a ruling) |
| Stage-2 launch | [PI] (§12 — not authorized) |
