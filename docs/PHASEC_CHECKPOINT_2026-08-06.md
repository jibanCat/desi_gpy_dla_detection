# PHASE-C PI CHECKPOINT — 2026-08-06 (Phase C1 complete; Stage 2 awaiting authorization)

Branch `calibration/phaseC-highN-fp-2026-08-06` (root `a56e3c8`, the
frozen Phase-B tip). Phase C1 (§ execution precedence items 1–8) is
COMPLETE; per the §12 default rule Stage 2 was NOT launched — the
measured budget is in `docs/PHASEC_BUDGET.md` and section C below.
Everything quoted is mock-derived; no real-DESI value anywhere.

## A. Verdict

* **What feeds observed G3 [21.0, 21.6]** (committed preimage,
  `diagnostics_phaseC/preimage/`): 99% of predicted G3 signal comes from
  true N ∈ [20.3, 21.7) — 12.7% migrates up from below 21.0, 84.0%
  in-band, 3.3% down across the 21.6 ceiling. **47.3% of G3's μ sits on
  covariates CLAMPED above the current top anchors (21.04–21.22)**, and
  the nominally measured [20.6, 21.1] band — which carries the peak
  sensitivity, 14,300 counts/dex — rests on ONE ~40-pair anchor per
  cell. Conditioning: all 9 response cells and both SNR extremes are
  material. The +450-count G3 discrepancy ≡ a +0.031 dex coherent mean
  bias at [20.7, 21.1), or −0.10…−0.15 dex over the clamped [21.1, 21.5)
  (which also relieves the 21.7+ over-prediction), or width analogues —
  the frozen §9 effect size.
* **Does the new pipeline reproduce the old response in bridge support?**
  At pilot precision, YES: no bridge anchor-cell deviates at |z| > 2
  (n ≥ 5) on either substrate arm. The production covariance-aware
  acceptance criterion is frozen pre-data
  (`docs/PHASEC_BRIDGE_DESIGN.md`).
* **Paired/shared bridge covariance handled?** Yes, by design and
  stated exactly: zero shared events (old = natural pairs, new =
  injections on disjoint sightlines → no paired differences exist);
  shared forest family carried as an empirically bounded Ĉ_shared via
  disjoint-healpix splits; C_old from the stored per-anchor rho spreads.
  Never C_old + C_new alone unless the shared bound is negligible.
* **Directly measured?** Not yet — that is Stage 2. The pilot verified
  the full chain (injection → unmodified production GP → production
  matcher → per-anchor moments) end-to-end at 310 injections.
* **Does the measured response explain G3?** Not answerable pre-Stage-2.
  One pilot-precision, engineering-labeled observation points the right
  way: the clamped region measures −0.03…−0.14 dex BELOW the frozen
  clamp (both arms), the exact direction/range the preimage identifies
  as feeding G3. Per §11 this is NOT confirmation; the production
  campaign with its frozen 25% whole-healpix holdout decides.
* **Fraction of the tilt remaining / replication:** unchanged from
  Phase B (no production measurement yet). The preimage and truth-by-SNR
  results replicate across all three mocks.
* **FP uncertainty decrease:** none yet (design + measured costing only:
  ±10.6% → 5% at ≈ 710 CPU-h, → 3% at ≈ 2,340; genuinely independent
  substrates verified on disk, including the independent 2LPT mock-1
  realization and the Saclay HCD-free twin).
* **Transport measured without leakage?** Designed, not yet run: the
  three FP roles are frozen at generation; Saclay's twin+with-HCD pair
  measures the natural-control method bias, making the London control
  (no twin exists) a corrected measurement rather than an assumption.
* **Is p < 0.01 too strict?** On independently simulated operating
  characteristics (deployed procedure, production FP regime): NO
  evidence it is. Healthy per-mock type-I 0.0167 ± 0.003 at the 89-event
  calibration (→ 0.005 as calibration grows); healthy-triple false-alarm
  3.4% vs **11.7% at α = 0.05**. The power difference at the
  observed-scale tilt (0.51 vs 0.76 per mock) is immaterial to the
  current verdict (actual p ≤ 5e-4 on all mocks). Ratification is yours;
  0.01 stays provisional.
* **Did truth-by-SNR change the calibration design?** No. The refold
  (authorized §16, one pass) REFUTES allocation/composition as the tilt
  driver: the truth's real SNR allocation differs from
  pathlength-proportional by 6.6% L1 and moves G3 by ≤ 5 counts of +450
  on every mock; the G3 deficit is SNR-near-uniform under both
  allocations. SNR stratification of anchors stays required; no SNR
  nuisance introduced.
* **Is another PI model decision required?** Not on the model. The
  decisions required are in section C (all authorization/ratification,
  no science reopened). The frozen Phase-B conclusion stands untouched.

## B. Evidence (all committed on this branch)

Preimage table + M matrices + conditioning decomposition + sensitivity
map: `diagnostics_phaseC/preimage/` (`d07c4ff`, `d2adbe3`). Bridge and
production anchors, density, support, conditioning, roles, holdout:
`docs/PHASEC_CALIB_DESIGN.md` (`a6e434b`); bridge covariance treatment,
acceptance criterion, transition/stitching + lack-of-fit escalation:
`docs/PHASEC_BRIDGE_DESIGN.md` (`b5d927e`). Sizing/precision/power:
`diagnostics_phaseC/design_sizing/sizing.json` — σ(G3) = 113.0 over all
re-measured bins (review-corrected), power 0.920, implied σ(G1)/σ(G2) =
618/445. Pilot verdict + pairs + hashes:
`diagnostics_phaseC/pilot/` (`cd6044a`); measured 106 CPU-s/spec, 96%
yield, matching/accounting exact. Truth-by-SNR:
`diagnostics_phaseC/truth_by_snr/` (`ccf9d6d`). Threshold operating
study: `diagnostics_phaseC/threshold_study/` (`90d5b91`). FP expansion
design/costing: `docs/PHASEC_FP_EXPANSION_DESIGN.md` (`13e0d46`). H9
definition + Layer-A conditional ratification:
`docs/PHASEC_GATE_GOVERNANCE.md` (`f70541c`). Conservative fallback
reporting implemented + tested (`b63a076`). r5 deterministic merge guard
+ release-cadence stochastic spec (`c3b0941`). Budget:
`docs/PHASEC_BUDGET.md`. Independent code review (§21):
`docs/PHASEC_CODE_REVIEW_2026-08-06.md` — verdict
**PASS-WITH-FINDINGS**; every re-run reproduced committed numbers
(several bit-identically), no frozen criterion weakened, the prohibition
boundary enforced in code; all eight findings dispositioned same-day
(record at the end of that file), the two record-keeping defects fixed,
F3/F7 tracked as pre-Stage-2 blockers. Tests: closure-path suites all
green in `gpdla-hbi` (354 passed + 1 explained skip after the r5
restructure); the full-suite baseline is environment-split and is
documented with its caveats in `docs/PHASEC_HANDOFF.md` (the ruling's
"939 passed/1 xfailed" is not reproducible in any current single env;
the 1 xfail IS r5, now restructured; the 35 gpdla-side failures are
torch/numpy version drift in training paths untouched since long before
Phase B, plus one 1.2e-11 bit-identity drift in an env whose numpy
exceeds its scipy's supported range).

## C. Decisions requested (nothing here proceeds without you)

1. **Stage-2 authorization + envelope** (`docs/PHASEC_BUDGET.md`):
   option 5% ≈ 1,850 CPU-h (37% of cap) or option 3% ≈ 3,480 CPU-h
   (70%). The response campaign alone is ≤ 110 CPU-h. Recommendation:
   option 5% (the 3% increment stays purchasable under the same frozen
   roles).
2. **Layer-B threshold ratification**: keep p < 0.01 (evidence above),
   or direct otherwise. Until ruled, it remains provisional and
   continuous p-values are the report.
3. **Reporting ceiling**: unchanged per your ruling; the design measures
   ceiling migration directly, so this decision is best made AFTER the
   production measurement (no action requested now).
4. **r5 stochastic validation cadence**: release-cadence re-powered run
   (~1–2 CPU-h) as specified in the governance doc — approve cadence or
   direct otherwise.
4b. **Bridge criterion 1 amendment ratification**: the independent
   review found the frozen criterion's CI clause vacuous (F5); it was
   replaced PRE-DATA by a real dispersion guard (95% CI upper bound of
   the G3-projected difference < 116.7 counts — a tightening, recorded
   in `PHASEC_BRIDGE_DESIGN.md` §4). Ratify or direct otherwise before
   Stage 2.
5. Deferred by rule (not asked now): whether the measured response
   suffices for Paper 1; further response structure (only if the frozen
   degree-2 lack-of-fit test fails); transport-prior recalibration;
   Model-A advancement.
