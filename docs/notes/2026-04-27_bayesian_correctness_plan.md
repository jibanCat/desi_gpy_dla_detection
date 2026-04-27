# Is the multi-DLA Bayesian computation strictly correct?

> User question: *"So you suggest there are some place the Bayesian
> computation part is not strictly theoretically correct? Maybe it's
> good to have a plan to pinpoint the reason (put after validating
> training GP). Do you think it's likely linked to the instrumental
> profile mismatch?"*

## Short answer (calibrated as a hypothesis, not a conclusion)

I do **not yet know** where the Lyβ-misidentification slack lives.
Three plausible loci, each falsifiable:

1. **The integration formulation is correct, but finite-sample QMC
   noise lets a small bias survive.** Would be falsified if the
   spurious-Lyβ rate is invariant under integration-method swaps
   (Step 4 below).
2. **The forward model is miscalibrated** (LSF kernel and/or num_lines
   wrong), so the per-sample log-likelihood at the Lyβ-shifted z is
   artificially close to the M_DLA(1) value. Would be confirmed if
   swapping in a DESI-shaped LSF (Steps 1–3) reduces the rate.
3. **The DLA prior pile-up at NHI=20.3 gives spurious low-NHI second
   DLAs extra weight.** Would be confirmed if the rate moves with
   `--alpha` or `--min-log-nhi` changes.

These hypotheses are not exclusive. The 4-step plan below is designed
to discriminate among them; **I do NOT have data to commit to any one
yet**. (Note added 2026-04-27 after user pointed out that an earlier
draft of this doc asserted the answer was forward-model miscalibration
without having tested it.)

## Is it linked to the instrumental profile mismatch?

**Unknown until tested.** A plausible mechanism, *if* the LSF mismatch
matters:

- True data at λ_obs corresponding to z_lybeta_apparent has a Lyβ
  trough whose depth and width are set by the *real* LSF.
- Production model predicts the trough using a *BOSS-shaped* LSF, which
  is too narrow (in DESI velocity units) by a factor of ~3-5.
- The model trough is therefore deeper-than-data in the very core and
  narrower in the wings. The χ² residuals at the Lyβ position are not
  zero, even when DLA1's NHI is exactly correct.
- Adding a second DLA at z₂ = z_lybeta_apparent with low NHI gives a
  small additional broad absorption — and this can reduce the residuals
  at the Lyβ position more than chance, producing a *real* likelihood
  improvement under the wrong model.

If the LSF mismatch is real *and* dominant, fixing it should reduce
the spurious-Lyβ rate. If it isn't, fixing it will move nothing. Step 1
of the plan tests this.

Other candidate contributors that **don't** depend on the LSF (and
should be tested independently):
- The QMC sample density: M_DLA(2) evidence depends on how the (z₂,
  NHI₂) prior is sampled. Sparse sampling near the Lyβ-shifted z lets
  individual high-likelihood samples carry disproportionate weight.
- The DLA NHI prior at the [20.0, 20.3] edge is steep (Ho+2020 mixture
  α=0.97), giving extra prior mass near the boundary. A spurious DLA
  with NHI driven to 20.3 by the redundancy with DLA1's Lyβ benefits
  from this prior pile-up.

## A plan to pinpoint the cause

### Step 0: verify the GP training first (per the user's ordering)

Validate the trained GP on injection-recovery before changing the
inference. Otherwise we'd be fixing inference around a moving target.

### Step 1: isolate the LSF effect

Use `gpy_dla_detection/voigt_v2.py` (already implemented, parity-
tested) to swap the LSF kernel without touching anything else. Run a
small set of spectra (~20 LOS that have a known parent + Lyβ-spurious
pair) under three configurations:

| run | kernel | num_lines |
|-----|---|---|
| A   | boss-log-r2000 (production) | 3 |
| B   | desi-linear-r3000           | 3 |
| C   | desi-linear-r3000           | 6 |

Measure: log p(D | M_DLA(2), z₂=z_lyb) per spectrum. If B drops below A
substantially, the LSF mismatch is contributing. If C drops further
than B, higher-order Lyman lines also contribute. If neither drops,
neither effect is the dominant cause and the integral itself is the
suspect.

### Step 2: test sample-density sensitivity

Re-run the same spectra under FILTER=0 with N_DLA = {10k, 50k, 100k}.
If the spurious M_DLA(2) rate decreases monotonically with N_DLA, the
finite-sample integration is non-trivially contributing. If it's flat,
the integral is converged at 10k and the issue is purely in the model.

### Step 3: prior-shape test

Same spectra under FILTER=0, varying the DLA prior `alpha` ∈ {0.3, 0.97}.
Alpha=0.3 makes the prior less steep at the boundary. If spurious
M_DLA(2) drops with alpha=0.3, the prior pile-up is contributing.

### Step 4: integration-method swap

Re-run with the same forward model but a different evidence
estimator: harmonic mean estimator, importance sampling around the
M_DLA(1) MAP, or nested sampling on a small case. If the spurious
M_DLA(2) rate is dramatically different across estimators, the issue
is the QMC integral itself, not the model.

After steps 1-4 we will have separated the four candidate causes
quantitatively: (i) LSF, (ii) num_lines, (iii) sample density, (iv)
prior shape. The fix targets whichever dominates.

## I do NOT have a current best guess

Pre-announcing which hypothesis I expect to win biases the test
design. The four steps above are mutually-discriminating; we'll
report the result of each step, then update the model of what
matters from data, not before.

## Long-run direction (per user)

User noted (2026-04-27): "in the long-run, I would like to have a
much more efficient and better sampler for my GP finder (something
like a super hand-written fast nest sampler)".

If the falsification plan above identifies sample-density / QMC
noise as a meaningful contributor, the next-iteration sampler is
the natural fix. Specifically a hand-rolled nested-sampler tuned
for the (z_DLA, NHI) parameter space — small dimensionality (2 per
absorber), well-defined likelihood plateau, and the existing
truncated-sampling scheme already does adaptive partitioning of
the prior. Saved as a project memory item; not addressed in this
plan.
