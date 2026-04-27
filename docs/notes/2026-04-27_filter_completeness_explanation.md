# Why FILTER=1 drops completeness — and how to fix it

200-target stratified sample (100 Saclay `juraLy8-124` + 100 2LPT `loa-124`),
4 NHI bins × 25 each per mock, SNR ≥ 1.5. eBOSS preset, multi-DLA mode.

## Completeness by truth NHI bin

| condition         | total | [20.3, 20.6) | [20.6, 21.0) | [21.0, 21.5) | [21.5, 23] | median ΔlogN_HI |
|-------------------|------:|------------:|------------:|------------:|-----------:|----------------:|
| FILTER=0, N=10k   | 87.8 % (209/238) | **89.0 %** | 86.4 % | 92.7 % | 82.4 % | +0.029 |
| FILTER=1, N=10k   | 84.9 % (202/238) | **79.5 %** | 84.7 % | 92.7 % | 84.3 % | +0.029 |
| FILTER=1, N=100k  | 86.6 % (206/238) | 78.1 % | 86.4 % | 90.9 % | **94.1 %** | +0.049 |

Total MAP DLAs and purity:

| condition         | MAP DLAs | matched | spurious | purity | Lyβ-explained spurious |
|-------------------|--------:|--------:|--------:|-------:|----------------------:|
| FILTER=0, N=10k   | 434     | 209     | 225     | 48 %   | 49 / 225 = 22 %       |
| FILTER=1, N=10k   | 292     | 202     |  90     | 69 %   | 17 /  90 = 19 %       |
| FILTER=1, N=100k  | 285     | 206     |  79     | 72 %   | 22 /  79 = 28 %       |

## Where the completeness drop comes from

The 2.9 % completeness penalty for FILTER=1 over FILTER=0 (87.8 → 84.9 at
N=10k) is **almost entirely in the [20.3, 20.6) NHI bin** — 89.0 % → 79.5 %,
i.e. **~9.5 percentage points lost** for the weakest DLAs near the prior
edge. The other three NHI bins are within 1–2 % of each other.

This is consistent with the FILTER=1 algorithm description:

> *Coarse QMC sample (~5,000) → keep only samples whose log-likelihood
> exceeds the null evidence → re-sample the surviving z_DLA window with
> ×20 more samples.*

For weak DLAs near the prior edge:
1. The Lyα feature is shallow.
2. The "null evidence" cut is calibrated against the QMC samples in the
   bulk of (z, NHI) space, where most are noise. Near a marginal DLA the
   true high-likelihood mode is only a small bump above null.
3. The coarse 5k QMC sample has roughly 5,000 / 4 NHI-bins ≈ 1,250 samples
   per NHI bin. At [20.3, 20.6), if no sample lands within Δz ≲ 0.005 of
   the truth, the surviving "high-likelihood" set after the null-evidence
   cut may be empty, and the algorithm misses the DLA entirely.
4. Increasing N to 100k recovers some of the [21.5, 23] cases (84.3 → 94.1 %)
   because high-NHI DLAs become harder to reach with sparse QMC at the
   tail of the prior — but doesn't help [20.3, 20.6) because the issue
   there is the null-evidence cut threshold, not sample density.

## Confirming the user's intuition about high-NHI splitting

The user wrote:

> *"With N_sample = 10,000, a large DLA sometimes is fit with two
> overlapping DLAs because lacking QMC samples at high NHI region."*

Concretely visible in the [21.5, 23] bin: completeness goes from
**84.3 % at N=10k → 94.1 % at N=100k**, a 10-percentage-point recovery.
The "two overlapping DLAs" failure mode would (a) add a spurious second
MAP DLA on these LOS at FILTER=1, N=10k and (b) be partially cured by
more samples. The N MAP DLAs total drops from 292 to 285 going to 100k
samples — consistent with these large-DLA splittings being un-split.

## Why FILTER=1 is *still* the right default for catalog production

Even with the 2.9 % completeness loss:

- **Purity 72 % vs 48 %** (N=100k vs FILTER=0). For a science catalog
  used in cosmology / BAO, contamination at the 50 % level dilutes
  signal far more than missing 3 % of the weakest DLAs — those weak
  DLAs (NHI < 20.6) contribute ~5 % of dN/dX and ~1 % of Ω_HI.
- **The completeness loss is in a single NHI bin** that can be filled
  in by the LLS-mode catalog (which has a better prior near 20.0–20.3
  and is rerun anyway).
- **Spurious MAP DLAs scale 4×** at FILTER=0 (434 vs 292), and they
  cluster near the prior edge — exactly where they damage the catalog
  most.

## How to fix the FILTER=1 completeness drop

In rough order of effort vs. expected gain:

### 1. Adaptive null-evidence cut by NHI bin (cheapest, biggest gain)

Instead of a single global "p(D | DLA sample) > p(D | Null)" cut over
the coarse QMC, compute the cut per NHI bin in the prior. Weak DLAs
near 20.3 should be allowed through with a relaxed margin so that an
actual marginal-detection mode survives the truncation.

Risk: if mis-calibrated this could re-introduce noise samples and
revert toward FILTER=0 behaviour. Easy to falsify on the same 200-
target sample.

### 2. Stratified QMC sampling (medium effort)

Replace the uniform Halton QMC with a stratified design that puts
fixed sample counts in each NHI quintile. Production currently
samples uniformly in log NHI ∈ [20.0, 23.0]; a stratified scheme
gives the same total N samples but guarantees coverage at the prior
edge.

Risk: changing the QMC sampler subtly changes the evidence integral
under FILTER=0 too. Need to verify the CDDF/dN/dX path (which uses
FILTER=0) is unchanged or trivially-different.

### 3. Multi-pass refinement (large effort, large gain)

The current scheme does coarse-then-fine sampling. Iterate: after the
fine pass, re-evaluate the null-evidence cut with the updated
posterior, and refine the z_DLA window again if a new high-likelihood
mode appeared. Two iterations would likely close the [20.3, 20.6) gap
without hurting purity. The user's draft text already alludes to this
direction — "iteratively sample the high-probability regions with more
samples".

### 4. Use the LLS-mode posterior to seed the multi-DLA fit (architectural)

LLS-mode is run on the same spectrum with prior NHI ∈ [17.2, 22.0] and
gives a posterior peak even for marginal absorbers near 20.3. Feed
those peaks as seed locations for the multi-DLA fine-pass. This is
"adaptive sampling driven by the LLS catalog" and should be near-
costless because LLS-mode is run anyway.

### 5. Don't filter the M_DLA(1) evidence — only filter M_DLA(2+)

The single-DLA hypothesis doesn't have a sample-density crisis (only
2 free parameters). The two-overlapping-DLA failure mode and the
weak-DLA failure mode are both *multi-DLA* problems. A simpler patch:
keep FILTER=0 behaviour for M_DLA(1), use FILTER=1 only for M_DLA(2+).
This preserves completeness at the prior edge where multi-DLA
splitting can't help anyway.

## Test plan for FILTER=1 (after Voigt unbiasedness work)

The FILTER=1 implementation has not been thoroughly tested. Run the
following falsification suite, in order:

1. **Re-weighting unbiasedness.** Construct a synthetic spectrum with
   a known DLA. Compare M_DLA(1) evidence under (a) full QMC, no
   filter and (b) FILTER=1 truncated-and-reweighted. The two should
   agree to within MC noise (~ 1 %). If they disagree systematically,
   the re-weighting step is biased.

2. **Multi-DLA equivalence.** Inject two truth DLAs at separated z
   on a synthetic LOS. M_DLA(2) evidence should agree under both
   FILTER settings if the algorithm is unbiased. Fail if FILTER=1
   under-estimates M_DLA(2) on real two-DLA cases (which would
   manifest as missed detections of secondary DLAs).

3. **Coverage scan.** 500 spectra spanning weak (20.3 ≤ NHI < 20.6)
   to strong (NHI > 22), SNR ∈ [0.5, 10]. Per-bin completeness for
   FILTER=0 vs FILTER=1 vs the proposed fixes. Falsify any fix that
   doesn't recover the [20.3, 20.6) bin.

4. **No-DLA control.** 200 mock LOS that have **no** truth absorber
   in the search window. Both FILTER settings must give p(DLA) below
   a sensible floor (say < 0.05). Catches the "FILTER=1 hallucinates
   detection at the prior edge" failure mode.

5. **CDDF path.** Run the population-statistics pipeline (CDDF /
   dN/dX) on a London mock with FILTER=0 and FILTER=1, compare against
   truth. Confirm the user's policy that FILTER=0 is the right setting
   for population stats — or, if FILTER=1 happens to be unbiased there
   too, document it.

Each test is a discrete pass/fail criterion, not a metric to interpret.

## Recommendation summary

| For                        | Setting                              | Why |
|----------------------------|--------------------------------------|-----|
| DLA catalog (BAO / Cosmo)  | FILTER=1, N=10k or 100k              | Purity dominates the science budget; small completeness loss is acceptable and concentrated in a bin filled in by the LLS-mode catalog. |
| LLS / sub-DLA catalog      | FILTER=0, N=50k+                     | Population statistics; want full QMC posterior over the prior volume. |
| Validation studies         | FILTER=0, full QMC                   | Bypass the truncation algorithm to keep the inference unbiased while testing other components (Voigt, training model, etc.). |
| Anything fragile to NHI<20.6 completeness | FILTER=0 until §1 above is implemented and validated | The [20.3, 20.6) bin loses ~10 percentage points under FILTER=1 today. |
