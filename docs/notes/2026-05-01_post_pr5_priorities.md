# Post-PR-#5 priorities — what to work on next

> Distillation of the user-Claude conversation on 2026-05-01 after the
> τ-EB recipe was validated at scale (146 k mock + 5 k real LOA, 56–65 %
> bias closure on DLA-regime). The user's specific feedback on each
> tier is captured below; this doc is the next-session starting point.

---

## Tier 1 — Direct production improvement (next 1–2 PRs)

### 1. Sub-DLA / LLS prior boundary fix (H4) — but NOT just "extend prior to 19.5"

**Problem**: current DLA prior boundary is `min_log_nhi = 20.0` (or 20.3
depending on file). Targets with truth NHI < 20.3 snap to MAP = 20.3
giving +0.4 to +2.7 dex bias on the n=54 LLS / sub-DLA scale-out.
Sub-DLA / LLS catalogs are heavily impacted.

**Naïve fix (rejected)**: extend the DLA prior down to NHI 19.5 and
crank `max_dlas` up to absorb the resulting sub-DLA detections.
Won't work cleanly — at logNHI = 20 there's a real population
boundary (sub-DLAs vs DLAs are physically different) and we still
lack a **joint sub-DLA + DLA model** in the multi-DLA search.

**Current architecture (the problem framed)**:
- Multi-DLA search ranges over `[20.3, 23]` (or `[20.0, 23]` in some
  configs). When a real sub-DLA at NHI 19.7 is in a spectrum, it
  cannot be modeled inside the DLA search → pile-up at NHI 20.
- A separate sub-DLA model exists as a **penalizing alternative**
  (Bayesian model selection between DLA and sub-DLA hypotheses).
  This helps purity at the catalog level but **biases the CDDF at
  logNHI ≈ 20** because real DLAs near the boundary get pulled
  toward the sub-DLA classification.
- The two models cannot fit JOINTLY (e.g. one sub-DLA + one DLA on
  the same LOS) because the multi-DLA search doesn't admit
  sub-DLA-strength peaks.

**Earlier suggestion (jibanCat)**: extend the sub-DLA model range
from `[19.5, 20]` to `[19.5, 20.3]` to soak up the pile-up at the
DLA prior boundary. Probably the cheap fix, but doesn't solve the
joint-model problem.

**Better idea (jibanCat)**: a two-stage scan —

  > "Scan [17.2, 23] range and find the rough peaks then run
  > multi-DLA algorithm to sample those peaks."

so sub-DLA peaks become candidate centers for the multi-DLA search,
allowing **joint sub-DLA + DLA fits** without changing the GP model
itself. Cost is O(sub-DLA scan + multi-DLA local search) per spectrum
— probably tractable if the initial scan is cheap.

**Open question**: how to keep model evidence unbiased when joint
fits include peaks at very different NHI scales. Needs design.

**Scope for the next PR**: probably (a) ship the cheap sub-DLA range
extension as a stop-gap, and (b) prototype the two-stage scan as a
research follow-up. Scientifically the two-stage scan is the
right answer.

### 2. Promote v2 LOA-trained model AND test mock-trained model

`loa_no_dla_no_bal_52198069/model_epoch_1499.h5` is the candidate
v2 production replacement (cleaner data than v1, 1500 epochs vs 953).
But also worth testing the mock-trained models
(`v2_runs/2lpt_loa{0,124}_*`) because **mock-trained models have
better-controlled contaminations** — we know exactly what was in
the trainset.

If the mock-trained GP gives equal or better validation results on
real LOA, that's interesting (says the GP is robust to training
distribution and what we measure is mostly intrinsic). If mock-trained
gives worse results, the v2-LOA-trained is the production answer.

The 2×2 anchor experiment in flight (jobs 49108430-49108443) is the
first read on this.

---

## Tier 2 — Research with clear payoff

### 1. Cost gap closure for 1 M QSOs (currently 6× over budget)

User's verdicts on the three options I sketched:

- (a) **Reduce num_dla_samples 10k → 2-3k** with adaptive sampler:
  USEFUL if the new sampler can satisfy three properties simultaneously:
    - **parallelizable** (current QMC parallelizes well across samples)
    - **fast** per-sample
    - **accurate on peaks** AND **unbiased model evidence**
  This is non-trivial — most adaptive samplers (e.g. nested) are
  sequential and don't parallelize. The model-evidence requirement
  rules out cheap importance-sampling tricks. **High payoff if
  achievable.**

- (b) **Drop max_dlas 3 → 2 for survey scale**: NOT NEEDED. The
  bayes step already early-stops when log evidence drops below null
  (we saw this in the FILTER fix #5 commits). Current logic naturally
  truncates to 2-DLA on most spectra; explicit cap doesn't help.

- (c) **Batch QMC across spectra** within a healpix: not discussed
  by the user. Probably non-trivial and orthogonal to (a).

**Recommended path**: focus on (a). The user's long-run-sampler
direction (Tier 3 #2) IS this work; not a separate item.

### 2. Overlapping DLAs — multi-issue, two distinct sub-problems

This came up because of the multi-DLA conflation case (TID 60167537,
truth NHI 20.6 fit at NHI 22.3). User pointed out two separate
mechanisms:

#### (a) Lyβ / Lyγ should disambiguate — but in practice doesn't

The Voigt code includes Lyβ and Lyγ lines (`num_forest_lines = 3`).
In theory: when two DLAs are within ~3000 km/s, their overlapping
Lyα damping wings are saturated, but Lyβ/Lyγ at different rest
wavelengths should reveal them separately.

In practice: the user reports rarely seeing Lyβ / Lyγ
distinguishing two close DLAs because **the Lyβ and Lyγ features
are too small** (lower oscillator strength → shallower damping
wings → swamped by forest noise).

**Open question**: is there a regime where Lyβ + Lyγ would
help? Larger spectra? Higher SNR? Worth a sensitivity study.

#### (b) Multi-DLA prior currently uses a constant velocity-separation prior

The DLA prior for k=2 DLAs assumes uniform separation (with the
3000 km/s minimum separation). User's idea: pull HCD clustering
from 2LPT mocks — DLA bias `b_DLA ≈ 2` (verified from eBOSS) —
and use the **measured clustering** as a prior on
`p(2 DLAs | Δv_separation)`.

Implementation:
1. Measure the DLA-DLA pair correlation function on 2LPT (or
   verify on eBOSS).
2. Convert to a velocity-separation distribution.
3. Use as prior in the multi-DLA model evidence calculation.

This would be a real improvement to the multi-DLA fit accuracy —
not just for overlapping DLAs but for the overall multi-DLA prior.

### 3. τ-EB on 50 k real LOA (NERSC has full LOA)

GreatLakes only mirrors ~5 % of LOA healpixes. NERSC has the full
catalog. A 50 k LOA Phase B at NERSC tightens the τ_factor result
to per-NHI bin precision and also lets us measure detection rate
shift (BASELINE vs ENABLED) at survey-relevant statistical power.

### 4. 2×2 anchor experiment results (in flight)

Currently running on GreatLakes (jobs 49108430-49108443). Will tell
us how much μ-shape / Ω-calibration matters vs runtime mean-flux.
Free — just wait for results.

---

## Tier 3 — Architectural cleanup

### 1. Unify the two-fold (τ_0, β) parameterization

The codebase has two distinct (τ_0, β) pairs:
- Mean-flux A multiplier (runtime, prev_tau_0/prev_beta, NEVER trained)
- Ω-kernel diagonal (trained log_tau_0/log_beta)

These should be the same physical thing. User confirmed this is on
a future to-do list, "more elegant" but not blocking. Should be
done in conjunction with re-architecting training to optimize the
mean-flux suppression too — i.e. the trainer learns the per-mock
(τ_0, β), and τ-EB at inference becomes a finer per-spectrum tuning
on top.

### 2. Long-run sampler (H6) — the user's flagged direction

Adaptive nested sampler instead of QMC. Tier 2 #1(a) overlaps with
this. The trade-off space is: parallelism + speed + peak accuracy +
unbiased evidence. Hard to satisfy all four.

### 3. Student-t / Huber residuals (H7)

Tests whether non-Gaussian forest residuals account for the residual
~0.04 dex bias after τ-EB. Modest payoff; non-trivial likelihood-code
change. Lowest priority of Tier 3.

---

## Tier 4 — Additional validation / research

### 1. BAL-only GP (user expressed mild interest)

Train a GP on BAL spectra only (BI_CIV > 0). Two use cases:

(a) **BAL-aware DLA catalog**: at inference time, run each spectrum
    through both non-BAL and BAL-only GPs; pick the higher-evidence
    model. Proper Bayesian handling of BAL contamination instead of
    just dropping BAL targets at picker time.

(b) **BAL physics characterization**: the learned μ/Ω structure of
    a BAL-only GP at population level is itself a science output.

Cost: ~150k spectra × 1500 epochs ≈ 4-5 h on NERSC, similar to existing
LOA trainings. Cheap.

The current "drop BAL at picker time" works fine for catalog FPR (down
to 0 %), so this is research rather than catalog production.

### 2. Mock-1 variants of each pipeline

We only validated on mock-0. Each mock has a mock-1 that's a free
statistical check.

### 3. BOSS DR16 cross-validation

Apply the recipe to a different real survey. If τ_factor distribution
on BOSS lands close to LOA's (median 1.5×), the mock-vs-real divergence
is a mock issue. If BOSS gives a different number, real DESI is
non-typical.

### 4. Run the 4 v2 trained models through canonical-target inference

For each new trained GP, run inference on TID 120046865 (truth NHI 21.26).
Tells us which one fixes the historical +0.34 dex bias best. ~1 min
each — cheap and discriminating.

---

## Recommended sequencing (next 6 months)

```
Now ────► Next PR ──────► PR after ─────► Tier-3/4 follow-ups
          (Tier 1)        (Tier 2 a/b)    (research)
```

**Next PR**:
- Sub-DLA range extension (cheap stop-gap) — ships immediately
- Two-stage scan prototype (research lead)
- v2 / mock-trained model evaluation on canonical + LOA Phase B

**Following PR**:
- Multi-DLA velocity-separation prior from HCD clustering
- 50 k NERSC LOA Phase B
- Adaptive sampler prototype (long-run direction)

**Tier-3 cleanup**:
- Unify (τ_0, β) parameterization (architectural)
- Pursue Long-run sampler in earnest

**Research / validation (parallel)**:
- BAL-only GP (when convenient)
- Mock-1 variants
- BOSS DR16 cross-validation

---

## Files this builds on

- `docs/notes/2026-04-29_bayesian_correctness_synthesis.md` — full hypothesis ledger
- `docs/notes/2026-05-01_tau_factor_distributions.md` — measured τ_factors across 4 populations
- `docs/notes/2026-05-01_trained_gp_models_comparison.md` — what each trained GP encodes
- `docs/stories/tau_eb_story_{2lpt,london,saclay,loa}.md` — per-mock + real-data narratives
- `feedback_dla_prior_edge_bias.md` (memory) — the original sub-DLA / LLS bias finding
- `feedback_lls_vs_multidla_modes.md` (memory) — distinction between LLS and multi-DLA modes
- `project_long_run_sampler.md` (memory) — the user's adaptive-sampler direction
