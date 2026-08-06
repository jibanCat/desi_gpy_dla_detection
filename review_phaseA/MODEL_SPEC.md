# Authoritative mathematical-model specification — Model A forward fold
### As implemented at `hbi-mcmc-threeroute` @ `9d73365`, annotated with the Phase-A verification status (review branch `review/phaseA-adversarial-2026-08-05`)

This document distinguishes three layers throughout:

- **[A: implemented]** — what the production code computes today.
- **[B: verified]** — components that survived the 2026-08-05 adversarial
  review (independent implementation or independent derivation; pointers to
  `review_phaseA/*/findings.md`).
- **[C: proposed / sensitivity-only]** — alternatives not adopted as
  production truth. Never cite a [C] item as implemented or verified.

---

## 1. Scientific estimand [A]

Target: the incidence rate of H I absorbers per unit absorption path,
**dN/dX (N_HI)**, over the ratified reporting window
**log₁₀N_HI ∈ [19.7, 21.6]**, for the full sample and per coarse redshift bin
(3 bins). The window floor is `reporting.NONIDENT_EDGE = 19.7` (inherited,
never re-derived); the ceiling 21.6 is `RESPONSE_ANCHOR_CEILING` — a
**reporting cap** motivated by the D2 residual excess, *not* the top of the
calibrated response (0.38–0.56 dex of extrapolated response lies inside the
window; `reporting.extrapolated_response_inside_window()`).

Explicitly outside Paper 1 (ratified 2026-07-29): total Ω_HI (no tail
treatment), the LLS population, any per-bin quantity built from `f_truth`
(B16 leak class), quantitative z-evolution claims (objective, not claim).

**[B]** The estimand reductions (`report_q.reductions`, mirroring
`model_a.reduce_f_posterior`) were exercised unchanged by the review; the
estimand itself was not contested.

## 2. Indices, axes, units [A]

| symbol | meaning | size (0.1-dex packs) | edges |
|---|---|---|---|
| b | true-N basis bin | B = 34 = 5 pad + 29 window | `ntrue_edges`: [19.0, 22.4], Δ=0.1 (pad [19.0,19.5)) |
| c | observed N̂ bin | C = 29 | `nhat_edges`: [19.5, 22.4] |
| k | fine redshift bin | Kf = 15 | `zf_edges` |
| s | SNR stratum | S = 8 (6 live) | `snr_edges` = [0,1,…,7,∞); strata [0,2) have dX ≡ 0 |
| K | coarse redshift bin | 3 | `zc_edges`; map `kz_to_K` |
| m | molly completeness cell | `n_molly` | `molly_nhi_edges` (v11: floor 19.5; molly172: floor 17.2) |
| (sr, zr) | response cell | 3 × 3 | `resp_snr_edges` = [2, 3.5, 6.5, ∞), `resp_z_edges` |

The **ratified analysis basis is 0.2 dex** (`coarsen_basis(pack, 0.2,
pad_floor=19.0)`); 0.1 dex is plotting-only. dX[k,s] is the absorption path
length per (k,s); counts[c,k,s] are the observed catalogue counts. Live cells:
dX > 0 (2,610 at 0.1 dex). dN_b = bin width in log₁₀N.

## 3. The forward fold — THE authoritative expected-count equation [A]

For every live cell (c, k, s) (`forward.fold_mu`; independent numpy oracle
`fold_mu_reference`, agreement required at rtol 1e-10):

    mu[c,k,s] = dX[k,s] · Σ_b  K[c←b](ψ_k; sr(s), zr(K(k)))          (response)
                         · C[cell(b), s](ψ_c)                        (completeness)
                         · g[b,k]                                    (z-resolved corr.)
                         · exp(θ[b,k]) · dN_b                        (population)
              + w · ℓ · exp(t[K(k)]) · λ[c,s] · E[k,s]               (false positives)

with observation models

    counts[c,k,s]  ~ Poisson(mu[c,k,s])        over live cells (masked, model_a.py:246-259)
    fp_counts[c,s] ~ Poisson(ℓ · λ[c,s])       the loa-0 calibration block (model_a.py:230-233)

Terms, dimensions, and status:

- **θ[b,k] = log f** — latent log-population; f = exp(θ) is the incidence
  density per unit log₁₀N per unit dX; f·dN_b·dX = expected true systems in
  (b,k,s) before selection. [A]
- **K[c←b]** — response kernel: analytic skew-normal bin mass
  F(hi_c) − F(lo_c), moment surfaces per response cell evaluated at the
  clamped covariate u = clip(N_b, resp_N_fit_range) − resp_N_ref
  (`resp_clamp="both"` in production; fail-closed on missing `resp_N_ref` /
  `resp_N_fit_range`). Support classes: **directly measured** on
  [19.5, 21.1), **weakly measured** [21.1, 21.5), **clamped/extrapolated**
  outside; 100% of the pad's kernel runs on frozen/extrapolated covariate.
  🔴 `resp_N_fit_range` is a **fit/binning-dependent range** (moves
  [19.34,21.22]→[19.10,21.62] when `n_N_cells` 7→40 on identical data) — it
  must not be described as a measured physical boundary. [A; boundary claims
  §4 of the 2026-08-05 checkpoint were NOT re-reviewed in Phase A]
- **C[cell(b), s] = logistic(η̂ + ψ_c)** — completeness on molly (s,m) cells,
  Jeffreys point surface η̂ = log((n_det+½)/(n_tot−n_det+½)), gathered to
  true-N bins by digitization. [A] **[B]** the review established the pad's
  exact (v11-grid) likelihood degeneracy is with this surface
  (`geometry/findings.md`).
- **g[b,k]** — z-resolved completeness correction (pack `g_grid`). [A]
- **FP term**: w = `fp_w_sightline_ratio` = N_prod/N_sl (sightline transport
  ratio); ℓ = `fp_ell_eff` = N_sl²/N_prod (calibration exposure); λ[c,s] =
  FP intensity per unit ℓ; E[k,s] = imposed z-allocation (∝ dX, Σ_k E = 1
  per live stratum); t[K] = coarse-z transfer factors.
  **[B — verified exactly]**: with λ̂ = n0/ℓ the folded FP equals the
  counting expectation r̂·N_prod (r̂ = n0/N_sl per-sightline rate) to
  1.2e-16; `w·ℓ ≡ N_sl` is an algebraic tautology of the definitions (an
  integrity check, not physics); the parameterization choice is
  posterior-identical to 9.3e-7 (`fp_normalization/findings.md`).
  🔴 **Known deviation from the stated definition**: the cited product
  definition carries (1−η̄) (η̄ = sub-DLA reclassification fraction,
  0.00576); the fold carries it **zero** times → FP term high by +0.58%.
  [C: restoring it is a prepared one-line Phase-B change awaiting PI
  confirmation.]

**Accounting identity** [A]: the matching-contract ledger
T·C·ρ + T·C·(1−ρ) + T·(1−C) ≡ T is **tautological** (holds for any C, ρ;
detects only shape errors) — stated as such since 9d73365; real value guards
exist per term. The non-tautological accounting (genuine / missed / migrated
/ FP not double-counted) is enforced by `matching_contract.py` predicates
and, for the FP term, verified by the generator↔fold round trip (exact
0.15/1.15 share; `fp_normalization/results.json`).

## 4. Likelihood, deviance, gate statistics

- Likelihood [A]: independent Poisson over live cells + the loa-0 block +
  priors (§5). Zero-dX strata are masked (per-element batch masking).
- Deviance [A]: 2Σ[y log(y/μ) − (y−μ)] over live cells.
- **Gate statistics** [A]: `forward_selftest.poisson_z` = (obs−μ)/√μ per
  aggregate row (total / by_nhat / by_z / by_snr), variance = predicted mean
  **only**, with λ at the plug-in n0/ℓ.
  🔴 **[B]**: this variance model omits the loa-0 calibration sampling noise
  (89 events, per-event folded footprint w ≈ 166 counts; var_cal/var_surv up
  to 17 per SNR stratum). Propagating it: by_snr χ²/dof 36.6/62.9/54.6 →
  3.6/5.9/4.9; window by_nhat 22.2/28.4/25.8 → 9.4/9.7/8.9. The N̂ axis is
  the leading residual; the closure failure stands at ~3× the ratified gate
  (`snr_residual/findings.md`). [C: a calibration-noise-aware gate statistic
  is PROPOSED, not adopted — it changes the ratified gate's meaning and
  needs a PI ruling.]
- **Model comparison** [B]: no single σ-equivalent interpretation of a
  Δdeviance between nested fits is meaningful here (boundary + correlation;
  the null is not pivotal). Calibrate by parametric bootstrap
  (`dev41_null/findings.md`: null mean 23.2, q99 40.7 for the 75-pad-param
  release; power ≈ 1 against the FP-only alternative).

## 5. Priors and constraints [A], with classification (§15.8)

| site | prior | class | identifies |
|---|---|---|---|
| σ_N, σ_z | HalfNormal(0.5) | regularization (RW scales) | — |
| θ level / slope | N(0,4), N(0,2) | weak regularization | — |
| eps_N (curvature), eps_z | N(0,1) non-centered RW | **regularization — supplies the pad's identification** | T_A (prior-identified) |
| ψ_c | N(0, σ̂) per (s,m) | calibration uncertainty (Jeffreys width) | completeness offsets |
| ψ_k_delta | N(0, fitcov_sd) | calibration uncertainty — 🔴 `resp_fitcov_diag` absent from all packs → **hard-coded fallback (0.02², 0.10²)**, a guess not a measurement | response perturbations |
| t[K] | N(0, t_sigma=[0.127,0.165,0.100]) | calibration uncertainty — 🔴 **[B] under-covers the measured cross-mock transport bias** (ln 1.447, ln 1.307 ≈ 1.9–2.8σ incl. template noise); also calibrated on the FF route and applied to the forward route (cross-link unresolved) | transfer |
| fp_lam_total | Gamma(0.5, 1e-6) | numerical (proper Jeffreys; scale-inert to 9.3e-7) | — |
| fp_shape_v | ZeroSumNormal(3.0) over C·S logits | **regularization — 149/174 live cells have zero calibration counts: the FP (c,s) shape is prior-dominated** | FP shape (prior-identified) |
| fp_counts block | Poisson(ℓλ) | **the anchor** — external calibration | T_B (anchor-identified) |

**Identification summary [B]** (Laplace, production coordinates,
`geometry/findings.md`): likelihood-identified — window f above the floor
(within its own N̂-axis misfit); anchor-identified — the FP total
(sd ≈ 0.015–0.12 dex); prior-identified — the pad total T_A
(≥ 0.6 dex likelihood-only → ≈ 0.05 dex with priors) and the FP (c,s) shape;
exactly degenerate without priors — pad↔(completeness × sub-20.0 window) [on
the v11 grid; real-but-not-exact on molly172] and t↔λ scale. The pad↔FP pair
is geometrically well separated (data-supported ≥ 6.6°; 18.2° on the ratified
basis). **The statement "sub-floor migration and forest FP are not separately
identifiable" is retired** (Phase-A verdict §2).

## 6. Calibration and transfer [B]

One frozen 2LPT-0 calibration is spliced bit-identically into all three
packs (24/32 arrays identical; only counts, dX, truth, and the per-mock FP
exposure scalars differ). London-0/Saclay-0 therefore test **prediction
transfer only**; no nuisance-calibration transfer is validated by them, and a
common calibration bias is invisible by construction
(`mock_transfer_audit.md`, manuscript-safe wording included). The loa-0→mock
FP transport (global sightline ratio) is **assumed**; measured: exact on the
twin (+0.2% after chance-coincidence correction), over-predicting the
held-outs by 31–45% (≈1.9–2.8σ with all noise sources; effectively one
observation — shared template draw).

## 7. Posterior products (gated) [A]

`model_a.reduce_f_posterior`: f-draws → dN/dX points and intervals from the
**same joint posterior** (points and bands must never come from different
posteriors); differential-reporting mask nulls [19.5, 19.7). Production
sampling remains **gated**: forward closure fails (window by_nhat 8.9–9.7 vs
gate 3 under the corrected variance model), the transport prior under-covers
the measured bias, and the pad/FP-shape are prior-identified — a posterior
would be reportable only with those prior-sensitivity axes stated (ψ_c bottom
cell, t, RW smoothness, fp_shape_sd).

## 8. Layer C — proposed / sensitivity-only (NOT production)

1. Calibration-noise-aware gate statistics (variance += delta-method loa-0
   term; or bootstrap over n0). Changes the ratified gate — PI decision.
2. (1−η̄) restoration in the FP fold (−0.58%): prepared, awaiting PI.
3. Frozen FP (c,s) shape: if ever adopted, must freeze the **in-support
   conditional** profile — the all-N loa-0 SNR profile makes closure 3–4×
   worse (refuted as a transfer; `snr_residual/findings.md`).
4. Transport-prior widening / refit for t_sigma.
5. `resp_clamp="hi"`: systematic bracket only; defensibility unadjudicated.
6. Sub-floor injections (option B): re-founded targets = pad↔completeness
   degeneracy, sub-19.0 promotion channel, FP shape calibration — not the
   (retired) A↔B degeneracy.

## 9. Equation-to-code map

| object | code | tests | verification |
|---|---|---|---|
| fold μ (jit) | `forward.fold_mu` (:491) | `test_modelA_forward.py` | [B] oracle rtol 1e-10; FP term exact (r01) |
| fold μ (oracle) | `forward.fold_mu_reference` (:528) | same | [B] |
| FP term | `forward.fold_mu_fp` (:455–488) | contract guard `matching_contract.py:952–1069` | [B] counting-exact; (1−η̄) deviation flagged |
| consts / clamps | `forward.build_consts` (:256) | `test_forward_response.py` | [A] |
| kernel moments | pack `resp_*_coef` via `znz_kernel` semantics | `test_modelA_vs_legacy.py` | [A; not re-reviewed] |
| completeness η̂ | `forward.eta_hat_sigma_hat` | `test_modelA_forward.py` | [A] |
| priors / model | `model_a.model_a` (:187–259) | `test_modelA_rungs.py` | [A]; geometry [B] |
| calibration block | `model_a.py:230–233` | — | [B] |
| gate z / arms | `forward_selftest.poisson_z`, `ratio_tables` (:186, :305–490) | `test_modelA_forward_selftest.py` | [A]; variance-model defect flagged [B] |
| estimand reductions | `model_a.reduce_f_posterior`, `reporting.py` | `test_adopted_reporting.py` | [A] |
| pack schema | `pack.py` (`load_pack`, `coarsen_basis`, `synthetic_pack` :1054–1073) | `test_modelA_pack.py` | generator [B] (round trip) |
| FP product extractor | guard lineage `CDDF_analysis/hbi/build_loa0_fp_product.py` | — | [B] definitions re-derived (r01) |

## 10. Mathematical-change log (this session)

**No equation or production behavior was changed in Phase A.** Proposed
corrections ledger (all pending PI):

| # | target | old | proposed | reason | status |
|---|---|---|---|---|---|
| 1 | `forward.fold_mu`/`fold_mu_fp` (+3 sites) | FP term without (1−η̄) | ×(1−η̄) = ×0.99424 | restores the stated product definition | prepared, unapplied |
| 2 | `matching_contract.py:572–574` | pre-repair forward_term prose | repaired convention | documentation matches code | trivial, unapplied |
| 3 | `forward_selftest.poisson_z` callers / gate | var = μ | var = μ + delta-method loa-0 term | gate overstates significance where FP matters | PI decision 1 |
| 4 | `reporting.py` `RESPONSE_ANCHOR_MEASURED` | "measured" anchors | + "fit-design-dependent (n_N_cells)" caveat | §4 binning-knob finding | unapplied |
| 5 | checkpoint/issue texts | five rejected statements (verdict) | retractions/reframings | Phase-A verdict | PI decision 6 |

Known failure modes to carry with any future use: one-sided-support class
(12+ occurrences project-wide); pinv on prior-less singular Fisher matrices
(produces regulator artifacts, not model properties); σ-equivalents for
non-pivotal statistics; shared-calibration "coherence across mocks".
