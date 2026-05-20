# Tier 1 #1 Phase 2 — fast joint multi-(subDLA + DLA) inference: design (rev 2)

> **Status**: 2026-05-04, draft for user sign-off.
> **Supersedes**: rev 1 of this doc (heuristic 2-stage scan), rejected
> by user in favor of a principled adaptive Bayesian sampler.
> **NHI search range** (decided): `[19, 23]` for the next PR; LLS
> coverage [17.2, 19] is a future extension.
> **Goal**: Replace the artificial sub-DLA / DLA cut with a single
> inference engine over `[19, 23]`, using adaptive importance
> sampling (AIS / SMC-flavored) to keep cost competitive with the
> current pipeline.

## Why this design and not the heuristic two-stage

The previous draft proposed a hand-tuned peak-finder (cheap z-only
sweep at fiducial NHI = 20.5, then focused QMC at peaks). User
pushed back:

> compute a coarse bayesian evidence with some smart choices of
> samples (uniform first to locate the rough regions to focus, then
> progressively adaptively sample the peaks to build joint model;
> something like my iteratively sampling but more principled)

The user's existing iterative sampling already does the right basic
thing: at each absorber count k, resample with replacement weighted by
likelihood (`dla_meanflux_gp.py:367-372`). What's missing — and what
"more principled" means here — is **proposal refinement between
iterations** with proper importance-weight bookkeeping so that
- the Bayesian evidence at each k is unbiased,
- the resampling variance is bounded (effective sample size > N_eff_min),
- coverage of all modes is guaranteed by the uniform Stage 1.

The framework is **Adaptive Multiple Importance Sampling (AMIS)** /
sequential Monte Carlo (SMC) — well-established in the Bayesian
inference literature (Cornuet+2012, Del Moral+2006).

## Architecture

### Stage 1 — Coarse uniform pass (1 absorber)

Draw `N_coarse = 2000` samples uniformly over the joint search space:

```
z_i  ~ Uniform(z_min(λ_obs, z_QSO), z_max(λ_obs, z_QSO))
NHI_i ~ Uniform(19.0, 23.0)             # log-NHI uniform, NOT PW14
```

Why uniform on log NHI rather than PW14? Because the proposal must
have **support everywhere we care about**. PW14 already heavily
down-weights the high-NHI tail; if we use it as the proposal, our
high-NHI MAP precision degrades. Importance weighting handles the
PW14 prior shape:

```
w_i ∝ p_PW14(NHI_i) × p(z_i | z_QSO) × p(D | z_i, NHI_i)
```

(The `p(z | z_QSO)` term is the per-spectrum z-search-window prior;
`p_PW14` is the PW14-mixture used in Phase 1.)

**Outputs**:
- Per-sample log-weight `log_w_i = log p_PW14(NHI_i) + log p(D | z_i, NHI_i) - log q_uniform(z_i, NHI_i)`
- Coarse 1-absorber log-evidence `log Z_1^(coarse) = logsumexp(log_w_i) - log N_coarse`
- Effective sample size `N_eff = (Σ w_i)² / Σ w_i²`

If `N_eff / N_coarse > 0.5` (well-sampled, no narrow peaks): stop here,
report `Z_1^(coarse)` and skip to Stage 3 with the uniform pool.

If `N_eff / N_coarse < 0.5` (concentrated posterior, evidence of
narrow peaks): proceed to Stage 2.

### Stage 2 — Adaptive refinement (per high-weight cluster)

Identify "modes" in the weighted Stage 1 samples:

1. Sort samples by weight, keep the top `f × N_coarse` (default
   `f = 0.05` → 100 high-weight samples).
2. Cluster in (z, NHI) space using a simple linkage with cutoff
   3000 km/s in z (the existing minimum-DLA-separation scale) — same
   distance metric as `min_z_separation`. Result: `K_modes` clusters.
3. For each cluster, fit a 2D Gaussian `q_k(z, NHI)` from the
   weight-weighted mean and covariance of cluster samples, with a
   floor on σ_z (≥ 500 km/s) and σ_NHI (≥ 0.1 dex) to avoid
   pathological narrow proposals.

Draw `N_refine = 500` new samples from each `q_k`. Compute their
log-weights with the **mixture proposal** (Multiple Importance
Sampling, balance heuristic):

```
q_total(x) = (N_coarse × q_uniform + Σ_k N_refine × q_k) / (N_coarse + K_modes × N_refine)
log_w_i = log p_PW14(NHI_i) + log p(D | x_i) - log q_total(x_i)
```

This is the AMIS recycling formula — every sample (Stage 1 or any
Stage 2 cluster) is reweighted under the mixture proposal, so the
combined estimator is unbiased and has lower variance than any
single-proposal estimator.

**Outputs**:
- Refined log-evidence `log Z_1 = logsumexp(log_w_all) - log N_total`
- A pool of (sample, log-weight) tuples with global N_eff > 0.5 × N_total
  (if not, expand K_modes or N_refine — but in practice 1-3 modes
  per spectrum)

### Stage 3 — Joint K-absorber model

Now we want `Z_k` for k = 2, 3, ..., K_max. The joint k-absorber
posterior factorizes:

```
p(z_1, NHI_1, ..., z_k, NHI_k | D) ∝ p_PW14(NHI_1) ... p_PW14(NHI_k)
    × Θ(min |z_i - z_j| > min_z_separation)
    × p(D | {z_i, NHI_i})
```

To sample from this, build joint proposals as **K-tuples of refined
samples**:

```
For each combination (i_1, i_2, ..., i_k) of K_modes refined clusters
  taken k at a time (or with replacement if a single cluster can host
  multiple absorbers):
    Draw one sample from each of those k cluster proposals q_{i_j}
    Form the joint proposal q_joint = product q_{i_j}
    Compute joint log-weight:
      log_w = Σ log p_PW14(NHI_j) + log p(D | {x_j}) - log q_joint
            + log Θ(separation OK)
```

For typical `K_modes = 1-3`, the number of joint configurations to
explore is small: `(K_modes choose k)` → k=2 needs 1-3 combos, k=3
needs 1 combo. For each combo we draw `N_joint = 500` joint samples
(one per absorber). Total cost: ~500 × `(K_modes choose k)` joint
likelihood evaluations per k.

`Z_k` is the importance-sampling estimate over the joint proposal,
again with the mixture-balance recycling formula.

### Stage 4 — Model selection

Plug `(Z_0, Z_1, Z_2, ..., Z_K_max)` into the existing
`BayesModelSelect.model_selection`. The downstream catalog interprets
each MAP absorber's NHI to assign sub-DLA / DLA / LLS labels post hoc
(Stage 4 is just relabeling — no new inference). The **boundary
becomes a labeling-only choice**, decoupled from the inference, which
is the user's headline ask.

## Why this is "more principled" than heuristic two-stage

| Property | Heuristic 2-stage | Adaptive (this doc) |
|---|---|---|
| Peak detection | hand-tuned threshold + fiducial NHI | implicit, weight-driven |
| Mode coverage | dependent on fiducial NHI choice | guaranteed by uniform Stage 1 |
| Bayesian evidence | derived per-stage, possibly biased | unbiased AMIS estimator at each k |
| Error budget | undefined | quantified by N_eff and importance variance |
| Tuning knobs | NHI fiducial, threshold, grid Δz | N_coarse, N_refine, f, σ_floors |
| Cost on clean LOS | ~3-5k evals | ~2-3k evals (skips Stage 2 via N_eff check) |
| Cost on busy LOS | ~5-10k evals | ~5-8k evals |
| Connection to existing iterative sampling | none | direct generalization with proposal refinement |

## Cost estimate

| Approach | Total log-likelihood evals per spectrum |
|---|---:|
| Production multi-DLA (max_dlas=3) | ~30,000 |
| Naive widen DLA samples to [19, 23] | ~30,000–60,000 |
| **Adaptive sampler (this design)** | **~3,000–8,000** (4×–10× faster) |

Same speedup target as the heuristic 2-stage, but with unbiased
evidence and guaranteed coverage.

## Open implementation choices

These are knobs, not blockers — sensible defaults are noted, and a
quick sensitivity sweep on a few canonical targets can lock them in
once the framework is up:

1. **N_coarse**: 2000 default. If evidence variance too noisy, bump
   to 5000.
2. **f (top-weight fraction for clustering)**: 0.05 default. If too
   few clusters identified, bump to 0.10.
3. **K_modes cap**: 5 default. Above this, evidence integrals
   over k-tuples blow up combinatorially.
4. **N_refine per cluster**: 500 default.
5. **σ_floors**: 500 km/s in z (matches min_z_separation), 0.1 dex
   in log NHI. Keeps proposals from collapsing to point masses.
6. **N_eff threshold for skipping Stage 2**: 0.5 × N_coarse. Lower
   means more aggressive refinement (slower but more accurate);
   higher means more spectra skip Stage 2 (faster but maybe
   under-resolved).
7. **PW14 prior in proposal vs likelihood**: Stage 1 proposes
   uniform on log NHI; PW14 enters via the importance weight.
   Alternative is to propose from PW14 directly (less coverage at
   high NHI). Default to uniform proposal.
8. **Min absorber separation between sub-DLAs and DLAs**: production
   uses 3000 km/s for DLA pairs only. For joint sub-DLA + DLA,
   should the same cutoff apply or a smaller one (since sub-DLAs
   and DLAs are different population)? **Recommend: same 3000 km/s
   for all pairs** as a starting point (they're observationally
   degenerate within that window). Open for review.

## Backward compatibility

Ship as a new CLI flag `--inference-mode {classic, adaptive}` (default
`classic` for the next 1-2 PRs). The classic path stays unchanged so
nothing in production breaks. Once Phase 1 + Phase 2 validation
establishes parity-or-better on the held-out set, deprecate `classic`.

## Validation plan (Phase 2)

Reuse the held-out n=200 truth-NHI ∈ [19.0, 20.6] set from Phase 1
(2LPT mock-1). Configurations:

- `production` — current pipeline + production sub-DLA samples
- `variant_alpha` — current pipeline + new sub-DLA samples (Phase 1 ✓)
- `adaptive` — adaptive sampler + unified PW14 prior on [19, 23]

Metrics:
- Pile-up at log NHI = 20.3 (production should show, α should reduce, adaptive should remove)
- Median |Δlog_NHI| in [19.5, 20.5] truth band
- DLA P/R, sub-DLA P/R (post-hoc Stage 4 labels)
- Wall-clock per spectrum (target: adaptive ≤ 1.2 × classic on average)
- log-evidence agreement between adaptive and classic on n=20 strong-DLA targets (sanity: should agree to ~0.1 in log Z)

## What I will NOT do without sign-off

1. Touch `dla_meanflux_gp.py` recursion or the existing
   QMC sample-grid code.
2. Add the `--inference-mode` flag.
3. Implement any of Stages 1-4. The order above is the implementation
   order; each stage is independently testable on a canonical TID
   before moving to the next.

## Effort estimate

- Stage 1 (coarse uniform + importance weights): 1.5 days
- Stage 2 (clustering + AMIS refinement): 2 days, the trickiest piece
- Stage 3 (joint k-absorber via cluster combinatorics): 1.5 days
- Stage 4 (existing BayesModelSelect glue): 0.5 day
- CLI flag + tests: 1 day
- Validation campaign: 1 day to run + 1 day to write up

Total: ~1.5 weeks of focused work after design sign-off. Slower than
the heuristic 2-stage estimate (~1 week) because of the proposal
clustering + AMIS bookkeeping, but the result is rigorous Bayesian
inference instead of a heuristic pipeline.

## Decisions captured (2026-05-04 user sign-off so far)

| Question | Decision |
|---|---|
| NHI search range | **[19, 23]** for next PR; LLS later |
| Inference style | Principled adaptive importance sampling (AMIS / SMC), not heuristic 2-stage scan |

## Files this builds on

- `2026-05-04_subdla_variant_alpha.md` — Phase 1 of the same PR
- `2026-05-02_subdla_dla_prior_design.md` — original Tier 1 #1 design
  (Option C lives here, this doc is the realisation)
- `gpy_dla_detection/dla_meanflux_gp.py:216-380` — current iterative
  sampler to generalize from
