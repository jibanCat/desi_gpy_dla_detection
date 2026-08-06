# REVIEW-ONLY (Phase A) — referee report: FP normalization defect, repair, ceiling

*Recorded by the orchestrator from the referee agent's final report (the agent
environment could not write report files); `results.json` in this directory
carries the same content machine-readably, with the site table (file:line),
all numbers, and scripts `r01`–`r05`.*

## Verdict

**Defect: CONFIRMED. Repair: CORRECT (exact) — not overshooting. Ceiling
claim: NOT same-estimand, and its bound direction is inverted; after
correction the "impossible excess" vanishes on 2LPT-0 and survives only
cross-mock, as a transport failure.**

## 1. Independent dimensional derivation (r01)

From the extractor sources alone: r̂[c,s] = n0[c,s]/n_sl (FP per searched
loa-0 sightline; Σn0 = 89, n_sl = 2255). The counting expectation for a mock
with N_prod searched sightlines is r̂·N_prod·(1−η). The calibration
n0 ~ Poisson(ell·lam) with ell = n_sl²/N_prod gives lam̂ = n0·N_prod/n_sl²,
and the fold w·ell·lam̂ = (N_prod/n_sl)·n0 = r̂·N_prod — the counting answer
**exactly once**, no double count (Σ_k E = 1 verified; exp(t) applied once,
fold-side only). The counting total equals the repaired `fold_mu_fp` to
1.2e-16; re-typing the defect reproduces 1086.687 vs 14767.961 (ratio ≡
ell_eff = 13.5899) from an independent path. Three sharpenings:

- (a) **`fp_w·fp_ell_eff == n_sl` is an algebraic tautology** of the
  extractor's definitions — the numerical check verifies pack integrity, not
  physics.
- (b) ell = n_sl²/N_prod vs n_sl is likelihood-inert — the convention makes
  the Gamma-posterior variance match the frequentist production-extrapolation
  variance w²·n0.
- (c) 🔴 **The (1−η) in the commit message's equality chain is not in the
  fold at all** — Model A carries η zero times (the FF route carries it).
  Real but small omission: η_subdla = 0.00576 → the FP term is biased
  **+0.58%**. (Confirmed definitional inconsistency; Phase-B candidate.)

Only 5 of the 6 requested packs exist — `modelA_pack_saclay0.npz` (non-v11)
is absent (provenance sidecar only).

## 2. Site audit at 9d73365 (r05 + grep)

One convention at every executable site: forward.py:455–488/519/674–675,
model_a.py:230–233/598–600, forward_selftest.py:161–172, pack.py:1054–1073,
extract_pack.py:998–1002, matching_contract.py:952–1069 (guard executes the
repair), evidence.py:409–430 (PPC). Generator→fold round trips exact
(FP share 0.15/1.15 exactly; fold vs oracle 1.6e-15). Two non-executable
inconsistencies: **matching_contract.py:572–574 still states the pre-repair
forward_term prose (no fp_ell_eff)**, and **every committed hbi_mcmc/*.json
closure artifact is pre-repair** with nothing marking them superseded.

## 3. Prior sensitivity of the repair choice (r02)

The two repair options (intensity + ell-in-fold vs count-rescale) are
**posterior-identical to 9.26e-7 relative** — exactly the predicted
eps·(1/ell − 1) tilt of the Gamma(½, 1e-6) cutoff; the softmax shape
coordinate cannot see the choice (identical gradients). Genuinely
immaterial — which is also why the defect survived: ell is pure
reparameterization on the loa-0 source side.

## 4. Ceiling-claim estimand (r03, r03b) — the headline finding

Independent census (support reproduces pack totals 88071/87840/86763
exactly):

- The **committed primary "unmatched"** (truth floor 19.5): 24,181 / 19,197 /
  20,225 — **μ_FP is BELOW it by 39/23/27%**. Under the stated estimand the
  ceiling claim fails outright.
- The claimed +6.55/+53.3/+38.9% instead match the **hostless-at-truth-floor-
  17.2** class (this review's ANY-host census: 13,491 / 9,253 / 10,273). For
  that comparator the direction claim ("unmatched over-counts forest FP") is
  wrong: ~10k sub-19.5-host rows are **~92% genuine sub-floor-host
  detections** (host-N composition 69% in [19.0,19.5) vs 24% chance-weight
  expectation).
- Dominant systematic of the hostless class: **chance z-coincidence with the
  ≥17.2 truth, measured by z-scrambling at 8.5–9.0%** (±0.10/±0.15
  displacements, same matcher; ~±0.8pp offset systematic).
- Corrected forest-FP supply S = hostless/(1−p): **μ_FP/S = 1.0020 (2LPT-0),
  1.447 (London-0), 1.307 (Saclay-0).** On the twin — where transport is
  exact by construction — the repaired normalization is **validated
  absolutely at +0.2%**. The excess is real only cross-mock: the loa-0→mock
  transport (global sightline ratio + exp(t), t_sigma = [0.127,0.165,0.100])
  is **assumed**, and the measured bias ln(1.447) = 0.37 / ln(1.307) = 0.27
  exceeds the t prior by **2.2–3.7σ** if coherent — the prior under-covers
  it.

## 5. The overshoot (r04)

"1.8–3.5%" located and reproduced: the reporting-window [19.7,21.6) ratios
1.0180/1.0345/1.0261 at the reviewed tip (full-grid 1.00155/1.05009/1.02814,
matching the prior session's probe). Attribution, measured: London's total
overshoot +4,400 counts ≈ its FP transport excess +4,549; Saclay
transport-dominated (+2,441 vs +3,456 with a ~−1,000-count signal deficit);
2LPT-0 total closes at **+0.16%** (+137 counts; FP excess +29), so its +1.8%
window residual is a **signal-shape effect compensated outside the window**.
**Both prior readings fail:** (a) "~6% loa-0 template bias" was the
uncorrected chance blur of the comparator; (b) no admissible reading makes
ell wrong — the only defensible correction the fold omits is (1−η) at
−0.58%.

## Remaining ambiguities (agent-stated)

1. saclay0 non-v11 pack npz absent (5/6 identity checks + 3 fresh
   extractions).
2. z-scramble chance rate carries ~±0.8pp offset-dependence; twin agreement
   1.0020 is consistent with both 0 and the −0.58% η omission.
3. The claim's exact comparators (13,860/9,600/10,590 implied) came from a
   greedy floor-17.2 matcher whose dump files no longer exist; this review's
   ANY-host comparator differs by ~2–4% in the expected direction.
4. t_sigma is calibrated from FF-route (FP-subtracted) closure ratios and
   applied to an FP-forward-modelled term; measured cross-mock transport bias
   exceeds it.
5. Per-(c,s) FP shape is prior-dominated (149/174 live cells with zero
   calibration counts) — total-level conclusions do not certify the shape.
