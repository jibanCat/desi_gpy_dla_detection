# Is the multi-DLA Bayesian computation strictly correct?

> User question: *"So you suggest there are some place the Bayesian
> computation part is not strictly theoretically correct? Maybe it's
> good to have a plan to pinpoint the reason (put after validating
> training GP). Do you think it's likely linked to the instrumental
> profile mismatch?"*

## Short answer

The integration scheme is **theoretically valid** in the limit of
infinite samples and an unbiased forward model. It's the **finite-sample
behaviour combined with a slightly miscalibrated forward model** that
lets the Lyβ-misidentification mode sneak through. Specifically:

1. The forward model predicts the data correctly *only* if the LSF
   kernel matches the instrument. Today's BOSS-shaped kernel on a DESI
   linear-λ grid systematically under-broadens the line core.
2. With a miscalibrated model, the per-sample log-likelihood at
   "spurious" (z₂, NHI₂) samples near the Lyβ-shifted z is **closer
   to** the M_DLA(1) likelihood than it should be, because the model's
   broadening doesn't fully account for what the data actually shows.
3. Marginalised over the (z₂, NHI₂) prior with finite samples, the
   M_DLA(2) evidence integral can edge above M_DLA(1).

So the "incorrectness" is not in the integral *formulation*; it is in
the **forward model fed to the integral**. Fix the LSF, and the
spurious M_DLA(2) likelihood at Lyβ-of-real-DLA samples should drop
because the model would correctly explain that feature with DLA1's Lyβ
line alone.

## Is it linked to the instrumental profile mismatch?

**Probably yes, partially.** The mechanism is concretely:

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

So the LSF mismatch creates a systematic incentive for the multi-DLA
fitter to add a low-NHI second DLA at the Lyβ position. Fix the LSF,
and that incentive partially disappears.

But there are other contributors that **don't** depend on the LSF:
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

## My current best guess

Most likely contributors, in order:
1. **Sample density / integration noise** — FILTER=1 reduces the
   spurious rate by 2× already, just by truncation. Suggests M_DLA(2)
   is sensitive to which 10k samples land where.
2. **LSF kernel mismatch** — under-broadening makes the Lyβ residual
   non-zero under M_DLA(1), creating the incentive for a second DLA.
3. **Prior pile-up at NHI=20.3 edge** — gives spurious second DLAs
   benefit-of-the-doubt prior mass.
4. **num_lines** — already at 3 in production, going to 6 unlikely
   to move much because Lyγ contribution is small at modest NHI.

So I'd expect Step 1 (LSF) and Step 2 (sample density) to dominate,
with Steps 3 and 4 second-order.
