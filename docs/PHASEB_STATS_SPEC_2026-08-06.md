# FROZEN statistical-analysis specification — Phase B (2026-08-06)

**Status: FROZEN before any Phase-B closure output is generated.** This
document fixes every statistic, covariance construction, dimensional
reduction, null-calibration procedure, and interpretation rule used by the
Phase-B confirmatory gates. Inputs to these choices: calibration structure
(measured 2026-08-06, §6 below), mathematical considerations, and the frozen
Phase-A conclusions (`review/phaseA-adversarial-2026-08-05` @ `a11dae0`) —
never Phase-B residuals. Any later deviation demotes the affected result to
exploratory.

Branch: `repair/phaseB-gate-fp-2026-08-05` rooted at `9d73365`.

## 1. Analysis configuration (inherited)

Zero-sampling truth fold on the adopted configuration: packs
`modelA_pack_{2lpt0,london0,saclay0}` at window `lya_only`, pad_floor 19.0,
completeness `molly172`, basis 0.2 dex; `resp_clamp="both"`; live cells
dX > 0; reporting window N̂ ∈ [19.7, 21.6]. Packs are re-extracted at the
Phase-B tip (with the `(1−η̄)` schema field, §5) with recorded commands and
hashes; the Phase-A exact-reproduction recipe is
`review_phaseA/snr_residual/findings.md`. The plug-in FP intensity is
λ̂ = n0/ℓ (n0 = pack `fp_counts`, ℓ = `fp_ell_eff`).

## 2. Three diagnostic layers (never interchangeable)

**Layer A — conditional implementation gate.** Conditions on the realized
calibration artifact as fixed. Statistic: the existing
`forward_selftest.ratio_tables` / `poisson_z` arms (total, by_nhat, by_z,
by_snr), variance = predicted mean, χ²/dof per arm; historical threshold
χ²/dof ≤ 3 retained AS A CONDITIONAL-IMPLEMENTATION DIAGNOSTIC ONLY. Code
path unchanged. [Provenance: inherited/ratified.]

**Layer B — calibration-predictive science gate.** Propagates the finite
loa-0 calibration sample through template → normalization → fold →
diagnostic vector. Defined in §3–4. [Provenance: prespecified here.]

**Layer C — transport stress test.** Cross-mock (loa-0 → London-0/Saclay-0)
mismatch reported separately as an UNCALIBRATED transfer systematic; never
absorbed into Layer-B covariance. Statistics: (i) full-grid total residual
z with variance = survey + calibration (delta method); (ii) the
FP-attributable excess share (Phase-A r04 decomposition method); (iii)
descriptive common-sign/shape comparison London vs Saclay. No threshold; no
pass/fail. [Provenance: prespecified; method inherited from Phase-A review.]

## 3. Layer-B primary statistic (confirmatory)

- **Primary residual axis: observed-N̂** [inherited from the frozen Phase-A
  conclusion; disclosed as motivated by Phase-A residuals].
- **Dimensional reduction:** the 19 window by_nhat bins are aggregated into
  **3 prespecified groups**: G1 = [19.7, 20.3), G2 = [20.3, 21.0),
  G3 = [21.0, 21.6]. Justification uses calibration-side information only:
  20.3 is the external physical DLA threshold; 21.0 separates the
  directly-measured kernel region from the weakly-measured [21.1, 21.5)
  region (Phase-A frozen support classification). [Prespecified here.]
- **Statistic:** T = d^T Ĉ⁻¹ d with d = G(y_obs) − G(μ̂(n0_obs)) the
  3-vector of group residuals (window bins, all z and SNR summed), Ĉ the
  frozen 3×3 covariance of §4. Exact inverse; **no shrinkage, no
  pseudoinverse, no mode truncation** at this dimension. Frozen fallback
  rule: if cond(Ĉ) > 1e6, DO NOT invert — report the three 1-dim
  standardized group residuals only. [Prespecified.]
- **Interpretation:** simulation-calibrated p-value only (§4); **no scalar
  χ²-threshold is assigned to Layer B**. A proposed (NOT ratified) future
  threshold p < 0.01 will be put to the PI at the checkpoint. Effective
  dof are reported descriptively from the null mean/variance if the null is
  approximately χ²-like; otherwise quantiles only. [Prespecified.]

**Secondary axes (descriptive, unadjusted — stated as such wherever
reported):** window total; coarse-z linear tilt contrast; SNR contrast
(stratum [2,3) minus strata [5,∞)); per-bin by_nhat residual table; per-arm
Layer-B analogues of the Layer-A arms. None of these is promotable to a
headline claim. Multiplicity treatment: exactly ONE confirmatory statistic
per layer per mock (Layer A: window by_nhat χ²/dof; Layer B: T above;
Layer C: none); everything else descriptive.

## 4. Layer-B covariance and null calibration

**Resampling unit (primary):** per-(c,s)-cell Poisson resampling of the
calibration counts, n0* ~ Poisson(n0_obs) — NOT independent per-bin error
bars added in quadrature: every resample is pushed through normalization
(λ* = n0*/ℓ) and the full fold, so cross-bin covariance, the shared
amplitude mode, and the imposed-E structure propagate exactly.
**Validation (frozen switch rule):** a sightline-block bootstrap of the raw
loa-0 catalogue (resample the 2,255 searched sightlines with replacement —
1,614 event-bearing + 641 empty — recount, re-select support, re-bin,
refold; B = 500, seed 42001). If any component of the 3-group calibration
sd differs from the Poisson-cell result by more than 10%, the sightline
bootstrap becomes the primary for ALL Layer-B quantities. The comparison is
reported either way. [Prespecified; decision rule frozen before results.]

**Covariance ensemble E_cov** (B = 2000, seed 41001): draws
d_r = G(y*_r) − G(μ̂(n0*_r)) with y*_r ~ Poisson(μ̂(n0_obs)) and n0*_r an
independent calibration resample; Ĉ = sample covariance of {d_r}. Reported
with: raw dimension (3), calibration event count (89 in-support; per-group
counts), resample count, algebraic and effective rank, eigenvalues,
condition number, MC error on elements (jackknife over the ensemble).

**Null ensemble E_null** (B = 2000, seed 43001, independent of E_cov):
T_r = q(y*_r, μ̂(n0*_r)) using the FROZEN Ĉ from E_cov.
p = (1 + #{T_r ≥ T_obs}) / (B + 1), with binomial MC error; tail
probabilities are never quoted beyond the resolution of B (minimum
reportable p = 1/2001 → reported as a bound).

The null procedure reproduces every analysis choice applied to the observed
vector (same grouping, same frozen Ĉ, same plug-in convention). Covariance
estimation (E_cov), null calibration (E_null), and final evaluation
(observed) use disjoint randomness. No statistic or covariance choice may
be revised after T_obs is computed; if any is, the result is exploratory.

## 5. The (1−η̄) restoration (applies to μ̂ everywhere above)

η is the **host-occlusion fraction** (build_loa0_fp_product.py: the fraction
of a production sightline's searchable forest occluded by a true HCD; a
forest FP can only occur in un-occluded forest). The written definition is
**per-band**: μ_FP,cell = n̂_FP·(N_prod/N_sl)·(1−η_band), with η_DLA ≡ 0
(forced; documented 1.73× over-subtraction otherwise), η_subdla =
0.005757, η_lls = 0.011187 (band-averaged from loa-124 hcd_truth; the
product carries the per-fine-bin vector `band_eta_per_nbin`). Restoration =
carry a per-observed-bin vector `fp_eta_c` (C,) into the pack (extractor →
schema → consts) and apply (1−η_c) once in the FP fold at all sites
including the synthetic generator. The scalar-vs-binwise comparison
Σ(1−η_c)μ_c vs (1−η̄)Σμ_c is reported on all three mocks; the binwise form
IS the artifact's definition (not a new convention). Expected effect ≈
−0.58% on the sub-DLA-band FP, 0 on the DLA band. The closure table reports
before/after. [Restoration of the stated definition; PI-approved ruling 8.]

## 6. Target–calibration independence (measured 2026-08-06)

- **2LPT-0: NOT independent.** 1,586/1,614 loa-0 FP-run TARGETIDs appear in
  the 2LPT-0 production catalogue with **identical Z_QSO** (max|Δz| = 0):
  loa-0 is the HCD-free twin of the same skewer set; the calibration
  footprint (2,255 sightlines) sits inside the 374,177-sightline target.
  Cross-covariance bound: shared events enter y once and μ̂ with weight
  w ≈ 166, so |2 Cov| / Var_cal ≤ 2 s̄/w ≈ 1.2% (s̄ ≤ 1 the shared-survival
  fraction) — negligible for Ĉ, and its neglect is CONSERVATIVE for a
  failure claim (true predictive variance is smaller). Consequence stated
  wherever the twin gate is quoted: **the 2LPT-0 Layer-B gate tests
  within-realization prediction** (same forest, noise, continuum family),
  not across-realization transport.
- **London-0, Saclay-0: independent** (TARGETID overlap 0 with loa-0).
  Layer-B covariance is C_target + C_calibration with Cov = 0 established.

## 7. Bounded twin observed-N̂ diagnosis — prespecified discriminants

One pass, ruling §15 order. Predicted signature stated before each test;
existing calibrated components only; no new model freedom. Prespecified
discriminating statistics:

1. **Stale/inconsistent artifact definitions** — re-extract packs at the
   Phase-B tip; discriminant: any bit-level difference in calibration
   blocks vs the Phase-A re-extraction; predicted signature if causal:
   residual changes when artifacts are rebuilt.
2. **N̂ bin-edge/normalization mismatch** — discriminant: residual parity
   under ±half-bin shifts of the aggregation edges (a true edge mismatch
   produces an alternating/sawtooth per-bin pattern and moves group sums
   under shift; a smooth shape misfit does not).
3. **Response-kernel shape** — discriminant: correlation of the per-bin
   residual with the kernel's per-bin sensitivity to the fitted moment
   perturbations (columns of the ψ_k Jacobian aggregated to N̂ bins);
   predicted signature: residual lies substantially inside the span of the
   2×(SR×ZR) ψ_k directions.
4. **Completeness vs true N** — discriminant: residual projection onto the
   ψ_c Jacobian directions (per molly cell); predicted signature: excess
   concentrated below N̂ = 20.0 aligned with molly-cell boundaries.
5. **Matching / multiple-candidate accounting** — discriminant: recompute
   counts under the alternate sibling-treatment convention available in the
   contract tooling; predicted signature: localized change where
   multi-candidate rows concentrate.
6. **Clamp behavior** — discriminant: residual delta between
   resp_clamp="both" and "hi" (diagnostic bracket only); predicted
   signature: differences confined to bins fed by clamped covariates.
7. **Below-support promotion** — discriminant: fraction of residual
   absorbed by extending the truth fold's pad floor 19.0 → 17.2 with the
   EXISTING molly172 completeness (no new model freedom; the pad-ladder
   machinery exists); predicted signature: low-N̂-group deficit shrinks.
8. **Conditional FP shape within support** — discriminant: replace the
   imposed within-group FP allocation by the calibration's own in-support
   conditional profile (a measured alternative, not a fit); predicted
   signature: redistribution among G1 bins only.
9. **Finite calibration-shape noise** — discriminant: fraction of per-bin
   residual χ² within the E_cov per-bin band.
10. **N–SNR interactions** — discriminant: stability of the group residuals
    across SNR strata (descriptive table).

Evidence rules: effect size and morphology over p-values; replication on
London/Saclay checked for any surviving mechanism; every result labeled
confirmatory (only if it uses §3–4 unchanged) or exploratory. Stop rule as
ruled: one pass; stop early only if a calibrated existing mechanism
explains a material, reproducible fraction of the residual.

## 8. Provenance classification of every choice

| choice | class |
|---|---|
| adopted config, window, basis, clamp | inherited (ratified) |
| Layer-A statistic + ≤3 threshold | inherited (ratified; conditional-only label is new, PI-approved) |
| primary axis = N̂ | inherited from frozen Phase-A conclusion (residual-motivated, disclosed) |
| 3-group edges 20.3 / 21.0 | prespecified here (external physical / Phase-A support structure) |
| Mahalanobis T, exact inverse, cond<1e6 fallback | prespecified here |
| resampling unit + 10% switch rule | prespecified here |
| B = 2000/2000/500, seeds 41001/43001/42001 | prespecified here |
| p-value rule, no Layer-B threshold, p<0.01 proposed-not-ratified | prespecified here |
| (1−η̄) binwise application | restoration of the artifact definition (PI ruling 8) |
| transport = Layer C, never in Ĉ | PI ruling 10 |
| diagnosis discriminants §7 | prespecified here |
| anything introduced after observing Phase-B results | exploratory by definition |

## 9. What this spec does NOT authorize

No new FP shape freedom, SNR functions, splines, logits, transfer
amplitudes, priors, or clamp conventions; no t-prior change; no absorption
of transport mismatch into covariance; no threshold ratification; no
posterior production. Per PI rulings 10–13 and the hard constraints.
