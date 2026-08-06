# Authoritative mathematical-model specification — Model A forward fold
### Phase-B revision (2026-08-06), branch `repair/phaseB-gate-fp-2026-08-05`
### Supersedes `review_phaseA/MODEL_SPEC.md` @ a420abd on the review branch

Three layers, marked throughout:
- **[A: implemented]** — what the production code computes at this branch tip.
- **[B: verified]** — survived the 2026-08-06 Phase-A adversarial review
  (`review/phaseA-adversarial-2026-08-05`, verdict @ a11dae0) or carries a
  deterministic regression test added in Phase B.
- **[C: proposed / sensitivity-only]** — not production truth. Never cite as
  implemented or verified.

## 1. Scientific estimand [A]

dN/dX(N_HI) over log₁₀N_HI ∈ [19.7, 21.6] (floor `reporting.NONIDENT_EDGE`,
ceiling `RESPONSE_ANCHOR_CEILING` — a reporting cap, not the top of the
calibrated response; 0.38–0.56 dex of extrapolated response lies inside the
window), full sample and 3 coarse-z bins. Outside Paper 1: total Ω_HI, the
LLS population, quantitative z-evolution claims, anything built from
`f_truth` (B16 class).

## 2. Indices, axes, units [A]

| symbol | meaning | size (0.2-dex adopted basis) | edges |
|---|---|---|---|
| b | true-N basis bin | pad [19.0,19.5) + window [19.5,22.4) | `ntrue_edges` |
| c | observed N̂ bin | C = 29 (0.1 dex) | `nhat_edges` [19.5, 22.4] |
| k | fine z bin | Kf = 15 | `zf_edges` |
| s | SNR stratum | S = 8 (6 live; [0,2) empty) | `snr_edges` |
| K | coarse z | 3 | `zc_edges`, map `kz_to_K` |
| m | molly completeness cell | `n_molly` (molly172: floor 17.2) | `molly_nhi_edges` |
| (sr,zr) | response cell | 3×3 | `resp_snr_edges` [2,3.5,6.5,∞) |

Adopted config: window `lya_only`, pad_floor 19.0, completeness `molly172`,
basis 0.2 dex, `resp_clamp="both"`. dX[k,s] = absorption path; counts and μ
are dimensionless catalogue counts; live cells: dX > 0.

## 3. THE forward-count equation [A; FP term B-verified]

For every live (c, k, s) (`forward.fold_mu`; independent numpy oracle
`fold_mu_reference` at rtol 1e-10):

    mu[c,k,s] = dX[k,s] · Σ_b K[c←b](ψ_k; sr(s), zr(K(k)))          (response)
                         · C[cell(b), s](ψ_c) · g[b,k]              (completeness)
                         · exp(θ[b,k]) · dN_b                       (population)
              + w · ℓ · (1 − η_c[c]) · exp(t[K(k)]) · λ[c,s] · E[k,s]   (FP)

    counts[c,k,s]  ~ Poisson(mu[c,k,s])         (masked to live cells)
    fp_counts[c,s] ~ Poisson(ℓ · λ[c,s])        (loa-0 calibration block — NO η:
                                                 loa-0 is HCD-free)

FP-term symbols [B — counting-exact, twin-validated +0.2%]:
- w = `fp_w_sightline_ratio` = N_prod/N_sl (per-mock transport ratio);
- ℓ = `fp_ell_eff` = N_sl²/N_prod (calibration exposure; the w·ℓ ≡ N_sl
  identity is an algebraic tautology — an integrity check, not physics);
- λ[c,s] = FP intensity per unit ℓ; plug-in λ̂ = n0/ℓ;
- E[k,s] = imposed z-allocation (∝ dX; Σ_k E = 1 per live stratum);
- **η_c[c] = host-occlusion fraction** per observed bin (restored 2026-08-06,
  commit 85bdba5, PI ruling 8): a forest FP can only occur in un-occluded
  forest; per-band by definition (η_DLA ≡ 0 forced; η_subdla = 0.005757;
  η_lls = 0.011187), carried by pack `fp_eta_c` from the product's
  `band_eta_per_nbin` via the single canonical `pack.eta_from_intervals`
  (straddle-asserting). Applied exactly once, fold-side only, generator
  included. Measured effect: μ_FP × (1 − 0.005757) exactly on all three
  mocks (−85.01 counts on 2LPT-0); binwise ≡ global-scalar to 0.0 counts
  (all 89 calibration events sit below N̂ = 20.3). Tests:
  `tests/test_fp_eta.py` (presence/uniqueness/weighting/round-trip).
- Legacy packs: fail-loud (`build_consts`); explicit idempotent migration
  `pack.attach_fp_eta_bands` (committed band table); the selftest CLI
  migrates historical packs with a logged note.

Response, completeness, g as in the Phase-A spec §3 (unchanged; kernel
support classes and the `resp_N_fit_range`-is-a-binning-knob caveat carry
over; §4 of the 2026-08-05 checkpoint remains not re-reviewed).

## 4. Diagnostic layers, likelihood, and null calibration [A+B]

**Never interchangeable** (frozen spec `docs/PHASEB_STATS_SPEC_2026-08-06.md`):

- **Layer A — conditional implementation gate** [A]:
  `forward_selftest.ratio_tables`/`poisson_z`, variance = predicted mean,
  χ²/dof per arm, historical ≤ 3 threshold. Tests that the fold implements
  the REALIZED calibration artifact; NOT a predictive science test. [B]: the
  Phase-A review showed this variance model overstates significance wherever
  FP matters (89 events × ~166-count footprints).
- **Layer B — calibration-predictive gate** [A, new]:
  `gate_covariance.predictive_gate` — full-pipeline parametric bootstrap
  (n0* → λ* = n0*/ℓ → production `fold_mu_fp`; never per-bin quadrature),
  frozen 3-group N̂ Mahalanobis (G1 [19.7,20.3) / G2 [20.3,21.0) /
  G3 [21.0,21.6]), exact inverse (cond<1e6 fallback to 1-dim), disjoint
  E_cov/E_null ensembles (2000/2000, seeds 41001/43001), simulation-
  calibrated p with MC-resolution bounds. **No ratified threshold** —
  p < 0.01 is PROPOSED to the PI, not adopted. Covariance objects expose
  full provenance (axes, edges, ranks, eigenvalues, condition number,
  calibration event counts, seeds, jackknife MC error, survey/calibration
  decomposition). Tests: `tests/test_gate_covariance.py`.
- **Layer C — transport stress** [A, new]:
  `gate_covariance.transport_stress_stats` — cross-mock loa-0 transport
  mismatch as an UNCALIBRATED systematic (PI ruling 10); never absorbed
  into Layer-B covariance; London-0 and Saclay-0 reported separately.

**Target–calibration independence** [B — measured 2026-08-06]: 2LPT-0 is NOT
independent of loa-0 (same skewer set: 1,586/1,614 FP-run TARGETIDs with
identical Z_QSO; |2·Cov|/Var_cal ≤ ~1.2%, neglected — conservative for
failure claims; the twin's Layer-B gate is a WITHIN-realization test).
London-0/Saclay-0: disjoint (overlap 0) — Cov = 0 established.

**Model comparison** [B]: no σ-equivalent for a Δdeviance between nested
fits here (boundary + correlation; non-pivotal null). Calibrate by
parametric bootstrap (Phase-A `dev41_null`: null mean 23.2, q99 40.7 for
the 75-pad-release; power ≈ 1 against the FP-only alternative).

**Confirmatory vs exploratory** [A]: exactly one confirmatory statistic per
layer per mock; everything else (secondary axes, per-bin residuals, the
diagnosis scan) is descriptive/exploratory and is labeled so in every
artifact. A choice revised after seeing Phase-B residuals demotes its
result to exploratory (frozen-spec provenance table).

## 5. Priors and identification [A; identification B]

Priors as in Phase-A spec §5 (σ_N/σ_z HalfNormal(0.5); θ level/slope weak;
non-centered RW; ψ_c ~ N(0,σ̂); ψ_k ~ N(0, fitcov_sd) with the hard-coded
(0.02², 0.10²) fallback still flagged; t ~ N(0, t_sigma) — 🔴 [B] t_sigma
under-covers the measured cross-mock transport bias and is FF-route-
calibrated (cross-link unresolved; ruling 10: do NOT widen in this phase);
fp_lam_total Gamma(0.5, 1e-6) — scale-inert to 9.3e-7; fp_shape_v
ZeroSumNormal(3.0) — prior-dominated, 149/174 live cells without
calibration counts).

**Identifiability [B — Phase-A corrected language, ruling 19]:**
- pad↔FP is NOT the principal degeneracy (best-separated pair;
  data-supported ≥ 6.6°, 18.2° on the adopted basis);
- the main likelihood degeneracies involve **pad ↔ (completeness ×
  sub-20.0 window population)** (exact on the v11 grid; real-but-not-exact
  under molly172) and the **t↔λ global scale** (prior-controlled);
- the sub-floor amplitude T_A is **prior-identified** (≥ 0.6 dex
  likelihood-only → ~0.05 dex under full priors, Laplace);
- the FP total is **anchor-identified** (the loa-0 block);
- the FP (c,s) shape is **prior-dominated**;
- prior-sensitivity axes to report with any pad-related number: ψ_c in the
  bottom molly cell, the t priors, the RW smoothness across the floor.

## 6. Calibration and transfer [B]

One frozen 2LPT-0 calibration, bit-identical across the three packs:
London-0/Saclay-0 are prediction-transfer tests only; common calibration
bias invisible by construction (manuscript-safe wording:
`review_phaseA/mock_transfer_audit.md`). loa-0→mock transport is exact on
the twin (same realization) and over-predicts the held-outs by 31–45%
(≈1.9–2.8σ incl. template noise; effectively one observation) — Layer-C
material; the proposal for independent transport calibration is a PI
checkpoint item (ruling 10).

## 7. Equation-to-code map [A]

| object | code | tests | status |
|---|---|---|---|
| fold μ (jit) | `forward.fold_mu` | `test_modelA_forward.py` | [B] oracle rtol 1e-10 |
| FP term | `forward.fold_mu_fp` | `test_fp_eta.py`, contract guard | [B] counting-exact + η restored |
| η schema/migration | `pack.eta_from_intervals`, `attach_fp_eta_bands`, `extract_pack.build_fp_block` | `test_fp_eta.py`, `test_window_study.py` | [B] |
| consts / clamps | `forward.build_consts` | `test_forward_response.py` | [A] |
| priors / model | `model_a.model_a` | `test_modelA_rungs.py` | [A] (one seed-marginal width guard flagged, §9) |
| Layer-A gate | `forward_selftest.ratio_tables`, `poisson_z` | `test_modelA_forward_selftest.py` | [A]; variance-model caveat [B] |
| Layer-B gate + covariance | `gate_covariance.py` | `test_gate_covariance.py` | [A+B] |
| Layer C | `gate_covariance.transport_stress_stats` | same | [A+B] |
| closure product | `closure_table.py` (thin CLI) | smoke via synthetic pack | [A] |
| contract / audits | `matching_contract.py` (`fp_normalisation_audit`, hostless-census comparison — corrected language) | `test_matching_contract.py` | [A+B] |
| estimand reductions | `model_a.reduce_f_posterior`, `reporting.py` | `test_adopted_reporting.py` | [A] |
| FP product extractor | guard lineage `CDDF_analysis/hbi/build_loa0_fp_product.py` | — | [B] definitions re-derived |

## 8. Layer C — proposed / sensitivity-only (NOT production)

1. Layer-B threshold p < 0.01 — PROPOSED, awaiting ratification.
2. Frozen FP (c,s) shape — exploratory sensitivity ONLY (ruling 11); must
   use the in-support conditional profile; conditional-identification
   language mandatory.
3. Transport-prior widening / refit — BLOCKED this phase (ruling 10);
   proposal to be presented at the checkpoint.
4. Injections — design/estimand work only (ruling 12); re-founded targets:
   pad↔completeness degeneracy, sub-19.0 promotion channel, low-N response
   support, completeness-controlled recovery.
5. `resp_clamp="hi"` — systematic bracket; defensibility unadjudicated.
6. Model A as the Paper-1 method — NOT adopted (ruling 13).

## 9. Mathematical-change log (Phase B)

| # | change | commit | class | evidence |
|---|---|---|---|---|
| 1 | frozen statistical spec | ef67a6d | documentation | — |
| 2 | record corrections (docs, issue, notes) | 374ab18 (+notes 26a267a) | record | ratification scan 0 violations |
| 3 | (1−η̄) restoration, binwise, all sites + generator + schema v1.2 | 85bdba5 | **behavior (restoration of the stated definition)** | test_fp_eta.py; −0.5757% on μ_FP, all mocks; signal-side bit-unchanged |
| 4 | Layer-B/C gate machinery | 5b14cbe | new diagnostics (no model change) | test_gate_covariance.py |
| 5 | closure-table CLI | 808ca46 | diagnostics | smoke |
| 6 | jax-free extractor fix | (follow-up) | mechanical | import check |

Known open items (§16 of the rulings; investigate-only): `resp_clamp="hi"`;
t_sigma FF→forward cross-link; the 16–21% FP-shape sensitivity
(reinterpreted as prior-sensitivity, not re-derived); the kernel-support
boundary (§4, not re-reviewed); prior-assisted sub-floor extension; one
seed-marginal sampler-width guard
(`test_r5_posterior_width_grows_with_shrunk_calibration`) flipped by an XLA
graph perturbation with bit-identical inputs/fold — under-powered by
construction (~4% margin vs ~10% MC noise), disposition at the checkpoint.
