# Phase-C gate governance — Layer A, H9, Layer-B threshold, fallback, r5

## 0. Ruled interpretation language (Stage-2 rulings §1, 2026-08-06 — BINDING)

* **The clamp** is the conservative boundary continuation
  K(x > x_max) = K(x_max): a numerical boundary treatment, NOT evidence
  that the true response is constant outside calibrated support. "47.3%
  of the predicted G3 mean lies on clamped covariates" MEANS: nearly
  half of the current G3 prediction depends on a boundary-continuation
  assumption rather than a direct response measurement in the relevant
  joint conditioning support. Clamped cells are never "directly
  measured". The pilot's −0.03…−0.14 dex observation is pilot evidence
  only, never a confirmatory response measurement.
* **A bridge pass** establishes pipeline and estimand COMPATIBILITY and
  supports a continuous old–new combination; it is NOT sufficient
  evidence that the old response is scientifically correct (the two
  pipelines may share truth-construction assumptions, matching
  conventions, estimators, normalization choices, or defects).
* **Low-N boundary (binding wording):** "The evidential strength and
  apparent significance of the earlier low-boundary failure have been
  substantially reduced and reinterpreted after correcting FP accounting
  and propagating calibration uncertainty. Much of the earlier evidence
  for a distinct low-boundary defect has been weakened or reclassified,
  but the underlying low-N response has not yet been directly
  validated." Never state the low-boundary residual disappeared or that
  the 19.5 boundary passed closure. The Stage-2 checkpoint keeps the
  high-N response verdict and the low-boundary status in SEPARATE
  sections.
* **Phase-C2 scope:** a domain-of-validity and calibration-support
  exercise — replace the clamped/thinly-measured region with a direct,
  precision-qualified measurement and test closure under the unchanged
  Phase-B statistics. Not a kernel refit; not a low-boundary campaign.

Implements PI rulings 2026-08-06 §14, §17, §18. Nothing here alters any
frozen Phase-B statistic, covariance, binning, group, or produced number;
raw historical values are preserved everywhere. Secondary to the active
calibration path (§ execution precedence) and completed after the C1
items were already committed.

## 1. Layer A (§14.1) — ratified as a CONDITIONAL implementation diagnostic

The legacy statistic — per-arm Poisson score residuals with variance =
predicted mean, window by_nhat χ²/dof with the historical χ²/dof ≤ 3
threshold — is henceforth described ONLY as a **conditional implementation
diagnostic**: it tests consistency CONDITIONAL on the realized calibration
artifact (response, completeness, g, FP counts held fixed at their point
values; `forward_selftest.poisson_z` docstring is the exact definition).
It is NOT the predictive science-validity gate and must never be described
as one; Layer B is. The raw statistic continues to be computed and
reported unchanged on every run (continuity, debugging, comparison with
older records — `closure_table` keeps its `conditional` block untouched).

## 2. H9 (§14.2) — explicit definition and reclassification

* **What H9 names.** Hypothesis 9 of the frozen Phase-B twin-diagnosis
  battery (`docs/PHASEB_STATS_SPEC_2026-08-06.md` §7 item 9;
  `diagnostics_phaseB/twin_nhat/findings.md`): *finite FP-calibration
  shape noise* — the per-observed-bin sampling noise of the 89-event
  loa-0 `fp_counts` profile, which enters every pack's μ directly.
* **Exact statistic.** For each window by_nhat bin c:
  z_survey(c) = (obs_c − μ_c)/√μ_c (the Layer-A definition) versus
  z_cal(c) = (obs_c − μ_c)/√(μ_c + Var_cal,c), with
  Var_cal,c = Σ_s (∂μ_c/∂n0[c,s])² n0[c,s] the delta-method calibration
  variance (EXACT here: the FP fold is linear in n0 and the resampling
  unit is n0* ~ Poisson(n0)). H9's quantification = the fraction of the
  survey-metric window χ² absorbed by the calibration band:
  Σz_survey² − Σz_cal² over Σz_survey².
* **Inputs.** The pack's `fp_counts`, `fp_ell_eff`, `fp_eta_c`,
  `fp_w_sightline_ratio`, `fp_E_alloc`; the fold at the truth-equivalent
  point; nothing else.
* **Original layer / previous interpretation.** Layer A. The per-bin
  window χ²/dof (twin 22.09) was previously read, in its entirety, as
  model misfit magnitude.
* **Why finite calibration noise changes that.** The conditional
  statistic conditions on the realized artifact, so variance originating
  IN the artifact is mis-attributed to the model. Measured (all three
  mocks, same definition): the calibration band absorbs **58% / 66% /
  65%** (twin/London/Saclay) of the survey-metric per-bin window χ²,
  entirely in the N̂ < 20.3 bins where the FP mass sits; **0% of G3**
  (no FP mass above 20.3).
* **Revised classification.** ANSWERED-BY-GATE (a variance
  re-classification): this component is calibration-sampling variance
  that the Layer-B covariance already carries — it is not a model error
  and not a repair.
* **What stays / what changes.** Raw values unchanged: every recorded
  Layer-A χ²/dof and per-bin z stands as written. Derived attributions
  change: any statement equating the full per-bin window χ² with model
  misfit is corrected to the decomposition above. Historical pass/fail:
  UNCHANGED — every recorded Layer-A verdict keeps its raw statistic, and
  the with-calibration per-bin χ²/dof (9.37 / 9.66 / 8.94) still exceeds
  3 on all mocks, so no verdict flips.
* Applied identically across mocks (measured on all three, above). H9 is
  never to appear as a bare label in PI- or manuscript-facing text; cite
  this section.

## 3. Layer-B threshold — RATIFIED 2026-08-06

**p < 0.01 is RATIFIED by the PI (Stage-2 rulings, Decision 2) as the
rejection threshold for THE CURRENT FROZEN Layer-B calibration-predictive
gate**, on the operating-study evidence (healthy-triple false alarm 3.4%
vs 11.7% at 0.05; observed failures at p ≤ 5e-4, so the verdict is
insensitive to the choice; the power difference immaterial at the
observed magnitude). Binding conditions of the ratification: continuous
p-values and finite-ensemble MC uncertainty stay in all reporting; the
threshold is NOT universal outside this frozen multi-mock gate; any
change in the number of mocks, tested projections, null construction,
calibration size, multiplicity rule, or fallback behavior REQUIRES a
fresh operating-characteristic assessment before reuse.

Pre-ratification record (kept for provenance): p < 0.01 was provisional
through Phase C1. The operating study
(`diagnostics_phaseC/threshold_study/` — independently simulated
characteristics on a production-geometry synthetic universe running the
DEPLOYED procedure; the observed failures enter only as the ε = 1
alternative SHAPE for the required power target) reports, for
α ∈ {0.01, 0.05} (no third candidate — none had a frozen rationale):
per-mock type-I, P(≥1 of 3 healthy shared-calibration mocks fails),
P(all pass), power against the observed-scale tilt and 0.5×/0.25×
scaled defects, sensitivity to calibration size (89/400/1111-event
regimes) and to finite null size (B = 500 vs 2000). The study returns to
the PI for ratification; until then continuous calibrated p-values are
reported, 0.01 is labeled provisional, and the current Phase-B verdict is
untouched (the observed failures are far beyond either candidate).

## 4. Conditional-covariance fallback (§17) — conservative reporting IMPLEMENTED

The frozen fallback (spec §3: cond(Ĉ) > 1e6 ⇒ no inversion) previously
degraded T to a descriptive max|z| but still attached the same p-value
machinery. Per the ruling, the fallback path now reports: the three
standardized group residuals; max|z| DESCRIPTIVELY; **no p-value and no
pass/fail** (`p_value=None`, labeled). A max|z| p-value would be
admissible only under a frozen prespecification whose simulation-
calibrated null covers the full fallback procedure including mode
selection and multiplicity; no such prespecification exists, and none is
created here. The fallback has never engaged on any produced result
(Phase-B cond ≈ 176‑equivalent 3×3 systems were well-conditioned), so no
historical number changes. The fallback is a contingency, never an
alternate result.

## 5. r5 (§18) — contract, deterministic merge guard, re-powered stochastic validation

* **18.1 The contract.** r5 guards SCIENTIFIC RECOVERY: the posterior
  interval width must respond to the calibration information content
  (shrinking the molly calibration ×1/16 must widen the posterior). That
  is inherently stochastic at the posterior level. Its MERGE-BLOCKING
  core, however, is deterministic: the calibration-variance plumbing —
  the Jeffreys completeness widths `sigma_hat` and the fold's linearized
  calibration-variance component — must scale correctly with the
  calibration counts. A width test and a recovery test are NOT the same
  contract; both levels are kept, explicitly.
* **18.2 Deterministic merge guard** (in the normal suite,
  `tests/test_modelA_rungs.py::test_r5_deterministic_calibration_width_contract`):
  on the synthetic pack, the Jeffreys `sigma_hat` surface and the
  delta-method calibration variance of the folded group prediction must
  grow by the analytic factor (×16 in variance, tolerance from the
  Jeffreys +1/2 offsets, stated in the test) when `molly_scale = 1/16`.
  Fixed inputs, no sampling, fails reproducibly. The existing
  deterministic Farr-gate check (`test_r5_farr_gate_fires_on_shrunk_
  calibration`) stays.
* **18.3 Stochastic validation — cadence APPROVED 2026-08-06** (Stage-2
  rulings, Decision 4): release-cadence, ~1–2 CPU-h/run; the
  deterministic guard (18.2) remains the ordinary merge-blocking
  contract; the under-powered test is never restored as a per-commit
  blocker. Required records per run: power target; seeds/realizations;
  false-failure and false-pass estimates; the exact release trigger
  (defined here as: any tag/release candidate on this analysis line, and
  any PI checkpoint that adopts a new calibration artifact); results
  retained under `diagnostics_phaseC/r5_release_validation/` with
  provenance stamps. The existing posterior-width comparison, re-powered:
  Measured state (3 seeds, Phase-B tree): sd ratios 0.958/0.998/1.040 —
  MC noise ±4–10% exceeds the effect at 2×200/150 draws. Requirement:
  false-failure ≤ 5% and false-pass ≤ 10% against the ×1/16 shrink,
  which per the measured seed spread needs the posterior-sd MC error
  ≤ 1/3 of the effect — to be sized by a dedicated pre-run measuring the
  effect (est. ≥ 2×2000 draws × ≥ 5 seeds ≈ 1–2 CPU-h; runs with
  release validation, cost recorded there). Until that job exists the
  per-commit test is marked `skip` with THIS section as the reason and
  an opt-in env flag (`RUN_R5_STOCHASTIC=1`) — it is no longer an
  unexplained xfail; a genuine width regression is caught by the new
  deterministic guard.
