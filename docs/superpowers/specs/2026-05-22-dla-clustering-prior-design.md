# Spec: DLA velocity-separation clustering prior

> **Date**: 2026-05-22 · **Status**: design, pre-implementation · **Branch target**: `production_533`
> **Author**: session 2026-05-22 (GreatLakes). Brainstormed + math-verified by an independent agent.
> **✅ REVIEW GATES COMPLETE (2026-05-22)**: a math-verification agent, a **cosmology referee**, and
> a **Bayesian-statistics referee** have reviewed the equations, physics, and prior settings. The
> cosmology referee gave a conditional sign-off; the Bayesian referee required changes (the SIR/Z_k
> normalization being the key one). All required changes are incorporated into §3–§7; verdicts are
> recorded in §9. Implementation may proceed via writing-plans.

## 1. Motivation

The multi-DLA evidence in `dla_gp.py` currently assumes the DLA redshifts in a sightline are
**independent and uniform** over the forest window. Real DLAs cluster: they trace the matter field
with a linear bias, so close pairs (small Δv) are more probable than uniform. The GP therefore
under-weights physically-likely close DLA pairs. This spec adds a **physical clustering prior**
`1+ξ_DLA(Δv)` to the multi-DLA evidence, improving close-pair recovery.

This is an **add-on**, default-OFF. It is orthogonal to the +0.05–0.06 dex NHI prior-edge bias
(see `docs/notes/2026-05-22_nhi_bias_meanflux_investigation.md`); it is about multi-DLA pair
*completeness*, not single-DLA NHI accuracy. The pre-clustering V1 production runs proceed
unchanged (`gl_prod_london0_v1_preclustering_*`).

## 2. Background / current code

- **Sampling** (`dla_samples.py:147-157`): `z_DLA = z_min + (z_max−z_min)·offset`, `offset~U[0,1]`,
  drawn independently per DLA. The k-DLA evidence (`dla_gp.py` `parallel_log_model_evidences`,
  ~L789-867) is an importance-sampling-from-the-uniform-prior MC estimator:
  `log Z_k ≈ logmeanexp_i [ L_i ]` where `L_i = sample_log_likelihood_k_dlas(z_i, N_i)`.
- **Floor** (`dla_gp.py:777-787`): samples with any pair closer than `MIN_Z_SEPARATION` (default
  3000 → kms_to_z → Δz=0.01 ≈ 812 km/s at z~2.5) are NaN-masked (excluded). This blocks the most
  strongly-clustered pairs entirely.
- **Measured clustering** (`examples/measure_dla_pair_clustering.py`, London-0 NHI≥20.3,
  `dla_clustering_london0_nhi203.npz`): `1+ξ ≈ 2.58` at Δv<500 km/s — strong small-scale excess.

**Scope [RC-5, C-1 — revised after design discussion]:** the prior is **real-data-applicable, with
discipline** — NOT mock-only. Rationale: without it, real catalogs have ≈0 close-pair completeness
(uniform prior + the `MIN_Z_SEPARATION` floor + the evidence penalty against overlapping Voigt
profiles structurally suppress close pairs), a known incompleteness; the prior recovers those pairs
with a *calibratable* selection. The injected prior is "undone" downstream exactly as the existing
`alpha(z)` completeness/selection correction for dN/dX: characterize `C(Δv)` on mocks (how the b=2
prior maps to recovered close-pair completeness), then correct real-data counts. Required discipline:
(i) mock-calibrated `C(Δv)` for any downstream clustering/dN/dX; (ii) **propagate b-uncertainty**
(b≈2.0±0.1–0.2, ξ∝b²) or marginalize it (Option C) so the headline isn't over-confident; (iii) the
one still-circular case to fence: *measuring b_DLA itself* from a b=2-prior catalog without forward-
modeling the selection's b-dependence — that needs Option C or explicit selection modeling. The
`--dla-bias` value and prior-mode are recorded in the catalog header as part of the selection
function. (This revises the referee's stricter "mock-only" RC-5; the calibration discipline replaces
the prohibition.)

1. **Correctness**: with the prior OFF, `dla_gp.py` is byte-identical to current behaviour
   (parity test). With it ON, the multi-DLA evidence is reweighted by a self-normalized clustering
   prior, and the **null-clustering invariance test [RC-3]** passes (no leak into p(k)).
2. **Recovery**: with the prior ON and `MIN_Z_SEPARATION` lowered, the GP recovers more of the
   truth's close DLA pairs — measured by the **pair-Δv recovery** metric (`dla_truth_diagnostics`),
   NOT global P/C (the 2026-05-15 smoke note showed P/C is insensitive to the floor). The prior must
   NOT degrade single-DLA ΔNHI, and the **ΔZ/ΔNHI of detected pair members vs truth [RC-6]** must
   not over-cluster (pulling pairs together *beyond* truth).
3. **Physics**: the analytic `ξ_DLA = b²·ξ_matter` (b=2) reproduces the empirical 1+ξ(Δv) measured
   from the mock truth in *shape* (the mock plants DLAs at linear bias 2). NB: the data 1+ξ is
   *redshift-space* (RSD-included Δv) while the analytic is real-space, so agreement is ~shape to
   ~15-30%, not bit-for-bit; this is a prior/regularizer, exact agreement is not required.

## 4. The math (verified by two Bayesian-referee rounds)

The clustered prior enters as an **evidence-only, Occam-style factor**. The SIR sampler is left
**completely untouched** — it resamples ∝ the *bare* likelihood, preserving full exploration of near
*and* far z_DLA (critical with sparse QMC samples; we don't starve far-separation regions). ρ never
enters `sample_log_likelihoods` or the resampling weights; it only reweights the *finalized* k-DLA
marginal evidence — exactly where the existing Occam penalty `−log(N)·k` lives.

**Per-tuple clustering weight (additive form):**
```
ρ_k(z_1..z_k) = 1 + Σ_{i<j} ξ_DLA(Δv_ij, z̄_ij)          # leading order, 2-point only
log ρ_k       = log1p( Σ_{i<j} ξ_DLA(Δv_ij) )            # feeds the evidence factor below, NOT added to L_i
```
This is the **leading-order (tree-level)** N-point density for a Gaussian/linear-bias field
(connected 3+point terms ζ small). NB: CoLoRe applies a *lognormal* transform before Poisson-
sampling DLAs at b(z) (Farr+2019 §2.1), which generates nonzero ζ ∝ ξ² — so the additive ρ_k is a
leading-order approximation at k≥3, not exact (a sound, conservative prior regardless). The
**multiplicative** form
`Σ log(1+ξ_ij)` ( = `Π(1+ξ_ij)` ) silently injects ξ·ξ products that are not true higher-order
correlations and **overcounts** at k≥3 (numerically ⟨ρ_mult⟩=3.20 vs ⟨ρ_add⟩=2.39 at k=3). Since
production runs MAX_DLAS=4, the additive form is required. For **k=2** the two forms coincide and
equal `1+ξ_12` exactly (the textbook 2-point definition `n̄²[1+ξ]`).

**The evidence factor (exact algebra; round-2 referee confirmed to machine precision).** For the
normalized clustered prior `π_clust = π_unif·ρ_k/C`, `C = E_unif[ρ_k]`:
```
Z_k^clust = ∫ L π_clust = E_unif[L ρ_k]/E_unif[ρ_k] = Z_k^unif · E_post[ρ_k] / E_unif[ρ_k]
log Z_k  +=  log E_post[ρ_k]  −  log E_unif[ρ_k]        # added to the FINALIZED bare k-DLA log-evidence
```
- **Z_k^unif** = the existing, **UNCHANGED** bare evidence (and the bare resampling weights `W`).
- **E_post[ρ_k]** = posterior (likelihood-weighted) average = `Σ_i p_i ρ_i / Σ_i p_i`, `p_i` = the
  **bare** `sample_probabilities` (= exp(L_k−max), `dla_gp.py:791`), `ρ_i = exp(log ρ_k)`.
  Equivalently `logmeanexp_i(L_i+log ρ_i) − logmeanexp_i(L_i)` computed on the bare column (do **not**
  mutate it). The `+log(N)`/`−lognorm·k` cancel in this ratio.
- **E_unif[ρ_k]** = prior average = closed form **`1 + C(k,2)·⟨ξ⟩_window`** (⟨ξ⟩_window = the
  triangular-pdf `2(L−d)/L²` integral over the spectrum's z-DLA window; analytic, *exact in
  expectation* by linearity — verified vs MC to 4 sig figs). This is the genuine prior-normalization
  constant `C`, computed once per spectrum.
- **k=1**: ρ≡1 → factor 0 → the **DLA-vs-null factor (Z_1) is untouched**; only the multi-DLA
  ladder (k vs k−1) shifts.

**⚠ E_post is a mildly biased estimator (the residual multiplicity-inflation risk).** The round-2
referee showed `Σ p_i ρ_i/Σ p_i` is a self-normalized ratio over the **SIR proposal q ≠ uniform**;
the `+log(N)`/`−lognorm·k` cancel but **q does not** (no `1/q` weight in `p`). So E_post is biased
**upward**, ESS-dependent (~+0.03 logZ at ESS≈0.4 → ~+0.10 at ESS≈0.13), one-signed toward more DLAs
— exactly the user's Comment-1 concern, in attenuated form. Crucially it **does NOT compound**
(applied once per k, *post*-resampling), so it is bounded and far smaller than the rejected in-place
option. It is monitored, not eliminated: the ESS-frac diagnostic (§7-i, warn <0.3) bounds it, and the
**false-positive-pair purity** check (§7-iv) is the *gating* empirical guard. Accepted as the inherent
SIR limitation for a default-off, mock-validated, calibratable prior.

**Wiring (non-negotiable — round-2 referee).**
1. `W` (resampling weights, `dla_gp.py:920-929`) computed from the **bare** likelihood — ρ excluded
   (no proposal change, no compounding; this is what makes RC-2 hold and what was wrong in the first
   in-place attempt).
2. `Δ_k = log E_post[ρ_k] − log E_unif[ρ_k]` added to the **finalized** `log_likelihoods_dla[k]` in
   all branches (no-truncation + FILTER partition: add to the *combined* value; for FILTER=1 compute
   E_post over the dominant `_valid_mask` samples).
3. Do **not** mutate the per-sample `sample_log_likelihoods` column. Default-off ⇒ byte-identical.

**Model-selection separability [RC-3].** ρ_k re-distributes probability mass *within fixed k*; it
must not leak into the existing occurrence prior `p(k) = (M/N)^k − (M/N)^{k+1}`
(`dla_gp.py:1042-1070`, applied in `bayesian_model_selection.py`). Guarantee = a **null-clustering
invariance test**: a spectrum with no true close pairs gives identical p_DLA / per-k posteriors
prior-ON vs prior-OFF (to MC tol). `Z_1`-untouched (k=1 factor=0) is the exact part of this.

**RC-1 correction (superseded by round-2).** An earlier round-1 RC-1 used *only* the realized-sample
`logmeanexp_i[log ρ]` (= log E_post, the **numerator**) and demoted the closed-form `1+C(k,2)⟨ξ⟩` to
"a cross-check" — that was **incomplete**: the closed form is the genuine **denominator** `C = E_unif[ρ]`.
The correct factor is `E_post[ρ]/E_unif[ρ]` — **both** terms.

## 5. The clustering function `ξ_DLA(Δv, z)`

```
ξ_DLA(Δv, z) = b_DLA² · [D(z)/D(0)]² · ξ_matter(r, 0),   b_DLA = 2 (Farr+2019)
r(Δv, z)     = Δv · (1+z) / H(z)                          # LOS comoving separation [Mpc]
ξ_matter(r,0)= (1/2π²) ∫ P_lin(k) k² j0(kr) dk            # FT of the z=0 linear matter P(k)
```
- **P_lin(k)**: Eisenstein-Hu 1998 *no-wiggle* transfer function (analytic, no camb/classy
  dependency; BAO wiggles at ~100 Mpc are irrelevant to the ≤40 Mpc forest window), normalized to
  σ8. **Cosmology = LyaCoLoRe's Planck-2015 input** (Planck 2015 XIII Table 3 base-ΛCDM TT+lowP,
  Farr+2019 §4.1; referee-confirmed): Ωm=0.3156, Ωb h²=0.02222, H0=67.31, ns=0.9645, **σ8=0.831**.
  (Planck18 σ8=0.811 was a ~5% ξ under-prediction; the corrected value gives 1+ξ(200 km/s)=2.63,
  matching the empirical ~2.58 better.)
- **Small-scale cap [RC-4]**: linear-bias ξ→∞ as r→0 is unphysical for discrete absorbers, and a
  hard `MIN_Z_SEPARATION` floor that truncates the very region the soft prior upweights is
  incoherent. Resolution: fold a small-scale turnover/cap into `xi_dla` (so `1+ξ` is the *complete*
  pair prior) and demote `MIN_Z_SEPARATION` to an explicitly-labeled likelihood/identifiability
  regularizer (guarding GP degeneracy of near-collinear Voigt profiles + double-counting), set far
  below the science scale and shown not to clip the ξ mass that drives recovery. Also floor the
  additive form: `max(1+Σξ, ε)` before `log1p` (ξ can dip <0 only at BAO scales, outside the
  window, but guard for robustness).
- **D(z)**: linear growth factor from the ΛCDM growth integral (or Carroll-Press-Turner fit);
  `H(z)` from astropy `FlatLambdaCDM`.
- **Cross-check**: analytic ξ_DLA(Δv) must reproduce the empirical 1+ξ(Δv) from the mock truth
  (the b=2 validation). Fallback if the analytic ξ_matter proves awkward: use the empirical
  1+ξ(Δv) table directly (it *is* b²ξ_matter for the mock) — A≡B here.

## 6. Components

| Unit | What | Depends on |
|------|------|-----------|
| `gpy_dla_detection/dla_clustering.py` (new) | `xi_dla(dv,z,b)`, `log_rho_k(dv_pairs,z,b)`, `mean_xi_window(z_min,z_max,b)`, `log_Zk(k,z_min,z_max,b)`; EH98 P(k) + ξ_matter(r) + D(z); module-level cosmology constants | astropy, scipy, numpy |
| `gpy_dla_detection/dla_gp.py` hook | in the multi-DLA evidence: `if pair_prior_mode=="clustering"`: add `log_rho_k` to each surviving sample's log-lik; subtract `log_Zk` from the k-DLA log-evidence. `pair_prior_mode="off"` (default) → block skipped → byte-identical | dla_clustering |
| Wiring | `--pair-prior-mode {off,clustering}`, `--dla-bias` (default 2.0) through `desi-DLAGP.py → run_bayes_select(DLAHolder) → DLAGP`. `--min_z_separation` already exists. **[RC-5]** clustering mode either hard-gates to mock runs or prints a runtime over-confidence caveat on real data (no silent fixed-b) | argparse |
| Tests | (a) parity: mode=off ⇒ byte-identical evidences on a fixture; (b) `dla_clustering` units: ξ at known Δv, additive≡multiplicative at k=2, additive<multiplicative at k=3, Z_1=1, small-scale cap, `1+Σξ` floor; (c) **null-clustering invariance [RC-3]**: no-close-pair spectrum ⇒ identical p_DLA/per-k ON vs OFF to MC tol; (d) hook on a 2-DLA spectrum (ρ in evidence only, not SIR weights [RC-2]) | pytest |

**Honors the don't-change-proven-path rule**: `dla_gp.py` gains a default-off branch only;
production (off) is byte-identical. The change to `dla_gp.py` is the minimal hook, gated.

## 7. Validation plan (separate runs, post-implementation)

- Sweep `MIN_Z_SEPARATION` ∈ {3000, 1500, 800, 400, 160} km/s with `--pair-prior-mode clustering`
  on a London-0 slice; baseline = current (mode off, 3000).
- Primary metric: **pair-Δv recovery** (truth vs GP close-pair counts vs Δv, `dla_truth_diagnostics`),
  plus global P/C and single-DLA ΔNHI not degraded.
- **Diagnostics required for sign-off [RC-6 + design-discussion comments]**:
  (i) per-spectrum reweighted **ESS-fraction** `(Σw)²/Σw²` for the k≥2 stages (warn <0.3; decide
  whether to raise `num_dla_samples` *from* this, not a priori) **[Comment 2]**;
  (ii) **ΔZ_DLA and ΔNHI of detected pair members** vs truth, ON vs OFF (does the prior pull pairs
  together beyond truth?);
  (iii) **b ∈ {1.8, 2.0, 2.2} robustness swing** on p(k≥2);
  (iv) **THE GATING METRIC [Comment 1] — purity of newly-detected close pairs**: of the close pairs
  the prior adds (present ON, absent OFF), the fraction matching a *true* truth pair vs spurious. If
  small-Δv purity holds, the Q3 completeness "undo" is clean; if it degrades, false positives don't
  divide out and the prior must be retuned (or the floor raised). This is the make-or-break check;
  (v) **MAX_DLAS cap pile-up**: does the prior push configurations into the MAX_DLAS=4 cap? If the
  cap is hit materially more often with the prior ON, raise MAX_DLAS and re-check.
- Compare against the `gl_prod_london0_v1_preclustering_*` baseline run.

## 8. Intermediate figures (for user review before implementation)

A prototype (`examples/prototype_dla_clustering.py`) generates, for review:
1. `ξ_DLA(Δv)` and `1+ξ_DLA(Δv)` at z=2.5 (b=2) vs the empirical mock 1+ξ(Δv) — the b=2 check.
2. `ξ_matter(r,0)` and the r↔Δv mapping (annotate MIN_Z_SEPARATION floors).
3. Per-sample `log ρ_k` weight vs Δv (k=2), and additive-vs-multiplicative at k=3.
4. `⟨ξ⟩_window` and `Z_k` vs window width / z_qso (the normalization).

## 9. Referee verdicts (completed 2026-05-22) — merge gates

**(A) Math verification agent** — additive ρ_k correct (multiplicative overcounts at k≥3); Z_k
normalization needed; importance-reweighting valid; magnitudes consistent. ✅

**(B) Cosmology referee** — **SIGN-OFF, conditional.** Verified: Δv↔r `r=Δv(1+z)/H(z)` correct AND
consistent with the empirical Δv definition (`measure_dla_pair_clustering.py` uses Δv=c·Δz/(1+z̄);
reproduces astropy comoving distance to 4 ppm); growth D(2.5)/D(0)=0.359 (CPT to 0.1%);
ξ_matter(8 Mpc/h)=0.50 correct; EH98 no-wiggle algebraically canonical; b_DLA=2 confirmed
(Farr+2019 §5.1, constant b_HCD=2.0, supported by Font-Ribera+2012/Pérez-Ràfols+2018). Required
changes (incorporated): use Planck-2015 σ8=0.831 (§5); relabel "exact"→"leading-order" (CoLoRe's
lognormal map gives nonzero ζ — §4 reflects this as leading-order); compare at matched bin centers.
Caveats (in §10): real-space theory vs RSD data; floor `1+Σξ`.

**(C) Bayesian-statistics referee** — **REQUIRED CHANGES (now incorporated).** Key catch: the
estimator is **SIR, not i.i.d.-uniform** → the closed-form Z_k is the wrong normalizer (biases
toward more DLAs); use the self-normalized `logmeanexp_i[log ρ_k]` over realized samples [RC-1, §4].
Also: ρ_k in the evidence only, not SIR weights [RC-2, §4]; null-clustering invariance test [RC-3,
§4/§6]; resolve the hard-floor/soft-prior incoherence via a small-scale cap [RC-4, §5]; scope-guard
b=2 to mock/caveat real data [RC-5, §3/§6]; validation must add ESS + pair ΔZ/ΔNHI + b-swing
diagnostics [RC-6, §7]; flag prior-ON catalogs out of clustering science [C-1, §3/§10]. Composes
correctly with the existing `+log(N)` patch and the `p(k)=(M/N)^k−(M/N)^{k+1}` occurrence prior.

**(C-round2) Bayesian referee, round 2** (after the design discussion chose evidence-only) —
verified the exact evidence factor `log Z_k += log E_post[ρ] − log E_unif[ρ]` (confirmed to machine
precision) with the **sampler left untouched** (your sparse-sample exploration concern). Two outcomes:
(i) **round-1 RC-1 was incomplete** — it kept the realized-sample numerator (E_post) and wrongly
dropped the closed-form denominator (E_unif = C); the correct form needs *both* (§4). (ii) `Σpρ/Σp`
is **not** proposal-invariant — it is biased mildly *upward* (ESS-dependent, one-signed toward more
DLAs), but bounded and non-compounding; monitored by ESS (§7-i) and gated by pair-purity (§7-iv).
The §4 math + §7 diagnostics reflect this.

All required changes from (B), (C), and (C-round2) are folded into §3–§7 above. **Implementation may
proceed** (Task 2 re-worked from the rejected in-place form to the evidence-only factor).

## 10. Open questions / risks / caveats

- **Real-space theory vs redshift-space data** (cosmology caveat i): the empirical 1+ξ uses
  RSD-included Δv; the analytic ξ is real-space. Linear Kaiser LOS boost for b=2 at z=2.5 is large
  ((1+β)²≈2.2 at μ=1, β≈0.49), partly offset by fingers-of-god at small Δv — likely most of the
  residual analytic-vs-empirical scatter. As a *prior*, exact agreement isn't required (don't claim
  "analytic ≡ empirical" as clean validation).
- **Circularity** (C-1): catalogs with the prior ON are flagged and **excluded from downstream DLA
  clustering / bias / ξ_DLA measurements** — those would recover the assumed b=2. Legitimate use:
  completeness studies + recovery validated against *truth* (independent of the assumed ξ).
- **ξ_matter source**: EH98-analytic (now σ8=0.831) matches the empirical to ~shape; the empirical
  table is the fallback (it carries the mock's true, RSD-included ξ). Decide at implementation.
- **Z_k cost**: the self-normalized `logmeanexp_i[log ρ_k]` is computed over the existing samples
  (no extra integral). The closed-form cross-check's per-spectrum 1-D integral is cheap.
- **Real data**: linear-bias / leading-order approximation degrades and b_DLA≠2 (and is uncertain);
  this prior is mock-scoped. Option C (hierarchical marginalized b_DLA) is the real-data path.
- **QMC** (C-2): the k≥2 stages are already plain MC (SIR uses `np.random.rand`); the reweight
  doesn't worsen QMC order but inherits that MC variance — hence the ESS diagnostic.
