# FILTER=1 knob tuning — make FILTER=1's completeness match FILTER=0

> **UPDATE 2026-05-13 evening — hypothesis substantially REFUTED by the 2×2
> ablation.** The 2×2 (`n_initial ∈ {5k, 10k}` × `empty-mask-fallthrough ∈
> {off, on}`) on London v3 8f 5k showed knob 1 alone gives only **+0.6 pp
> completeness** vs baseline, knob 4 is essentially a **no-op** (empty-mask
> branch is too rare to move the headline metric). Both knobs together: +0.3
> pp C. The FILTER=0 / FILTER=1 completeness gap of ~3 pp therefore does
> NOT come from coarse-scan miss (knob 1) or empty-mask early-stop (knob 4)
> as conjectured below. The remaining gap is unexplained — knobs 2 (`z_tol`),
> 3 (`null_threshold_delta`), and 5 (truncated-region estimator for `num_dlas
> ≥ 1`) are not yet tested. See `HANDOFF.md` 2026-05-13 21:30 evening top
> block and `docs/notes/2026-05-13_cellC_mechanism_verdict.md` for the
> alternative (cellC's posterior-arithmetic) that actually does close the gap.
>
> The knob-plumbing CLI flags `--filter_n_initial_floor` and
> `--filter_empty_mask_fallthrough` landed in commit `2e3642b` and remain
> useful as debugging tools; the production runbook §3.6 documents the
> recommendation to keep defaults.
>
> The rest of this note is preserved as the original (now-superseded)
> investigation plan.

---

> **Written 2026-05-13 after user reframe**: "The goal is to tune the knobs in
> filter=1 so that the completeness match to filter=0, because the underlying
> integration math should be the same."
>
> The FILTER=0 vs FILTER=1 completeness regression in `[20.3, 20.6)` NHI bin
> ([`2026-04-27_filter_completeness_explanation.md`](2026-04-27_filter_completeness_explanation.md),
> confirmed weakly today in
> [`2026-05-13_filter_nfl_confirmation.md`](2026-05-13_filter_nfl_confirmation.md))
> is a **knob-tuning bug**, not a fundamental algorithm choice. Both modes
> approximate the same marginal `p(D | k-DLA) = ∫ p(D | z, log N_HI)
> p(z) p(log N_HI) d(z, log N_HI)`. FILTER=0 averages all `NUM_DLA_SAMPLES`
> uniformly; FILTER=1 does a coarse-then-refine truncated-region scheme
> that *should* asymptote to the same answer with sufficient samples.
> Today's task: identify the specific knobs in the FILTER=1 code that, when
> tuned, close the gap.

## The FILTER=1 algorithm in code

All references to `gpy_dla_detection/dla_gp.py` on branch `production_533`
at commit `bb218c5` (or later).

The mechanism, in code order:

### Step 0 — Choose `n_initial` (the coarse-scan budget)

```python
# line 575
n_initial = max(int(self.params.num_dla_samples // 20), 5000)
```

For `NUM_DLA_SAMPLES = 50000` (production) → `n_initial = max(2500, 5000) = 5000`.
For `NUM_DLA_SAMPLES = 100000` → `n_initial = max(5000, 5000) = 5000`.
For `NUM_DLA_SAMPLES = 10000` → `n_initial = max(500, 5000) = 5000`.

**The 5000 floor dominates at every production sample budget.** The `// 20`
divisor matters only above `NUM_DLA_SAMPLES = 100000` where it would start
to push `n_initial` past 5000.

### Step 1 — Coarse scan over the first `n_initial` samples

The first `n_initial` QMC samples are evaluated at `num_dlas = 0` (the
1-DLA marginal). `initial_logL[i]` holds the per-sample log-likelihood.

This is the bottleneck for *catching* weak DLAs: if no sample within the
first 5000 happens to land near the truth's (z, log N_HI), the truth's
high-likelihood region is invisible to the coarse scan, and the next step
returns an empty mask → 1-DLA marginal is computed from only the 5000
coarse samples (no refinement) → may not pass the P_DLA cut.

### Step 2 — Build the "valid region" mask using `select_region_indices_searchsorted`

```python
# line 622, 626-632
z_tol = 0.02   # TODO: find the best value  ← explicit knob TODO!
valid_mask = select_region_indices_searchsorted(
    z_all=sample_z_dlas,                   # full 50000
    initial_logL=initial_logL,             # only 5000
    initial_z=sample_z_dlas[:n_initial],   # only 5000
    z_tol=z_tol,
    logL_null=null_evidence,
)
```

`select_region_indices_searchsorted` (line 91):
- Keeps `initial_z[i]` if `initial_logL[i] > logL_null` (== `null_evidence`).
- Returns the boolean mask over `z_all`: True where any retained `initial_z`
  is within `±z_tol` redshift.

So the retained samples are those whose redshift is near the redshift of at
least one coarse-scan winner.

### Step 3 — Two branches

**Branch A — empty mask (line 635):** if `valid_mask.sum() == 0`, the
coarse scan found no sample above null evidence. Compute the 1-DLA marginal
from the coarse scan alone (5000 samples), log "Stopping early at 1 DLAs",
and **return immediately — no 2-DLA / 3-DLA exploration**. This is the
single largest mechanism by which FILTER=1 loses weak DLAs that FILTER=0
catches: FILTER=0 always evaluates all 50000 samples and so has 10× more
chances to find the high-likelihood region.

**Branch B — non-empty mask:** refine on the surviving samples. For
`num_dlas == 0`, FILTER fix #5 (2026-04-29) applies: use the *unbiased*
mean over all `initial_logL` for the 1-DLA evidence (line 762-766) — this
is essentially FILTER=0 at `n_initial` samples, so 1-DLA evidence is fine
in branch B. For `num_dlas ≥ 1`, do the truncated-region correction (line
770-797) using the retained mask + a "rejected region" likelihood estimate.

## The knobs, ranked by how much they should affect completeness

### Knob 1 — `n_initial` floor (line 575)

The 5000 floor dictates the coarse-scan coverage. Doubling it to 10000
halves the chance that a weak DLA's high-likelihood region is missed by
the coarse scan, at 2× the per-spectrum cost of the initial scan (which
is the same 1-DLA work that FILTER=0 does on all samples — so the marginal
cost is small).

**Experiment**: rerun FILTER=1 with `n_initial_floor ∈ {5000, 10000, 25000}`
on the 5k London v3 sample. The expectation: at `n_initial = NUM_DLA_SAMPLES`
the FILTER=1 result asymptotes to FILTER=0 (modulo the truncated-region
correction for num_dlas ≥ 1).

**Code change**: parametrize `n_initial`. Two natural options:
- A new constructor argument `n_initial_floor: int = 5000`.
- A new ratio knob `n_initial_ratio: int = 20` (currently hardcoded `// 20`).

Either way, plumb through `DLAHolder` → `dlasearch.{hpx,mock}` → CLI as
`--filter_n_initial_floor` or `--filter_n_initial_ratio`.

### Knob 2 — `z_tol` (line 622)

The TODO at line 622 explicitly flags this as "find the best value".
Currently 0.02. Widening it captures more samples near the high-likelihood
z; narrowing it tightens the truncation.

For the [20.3, 20.6) bin specifically: the high-likelihood region for a
weak DLA is **narrow in (z, log N_HI)** (small damping wings, shallow Lyα
core). The z-width of the region scales with NHI — weaker DLAs have a
*narrower* z-window than strong ones. So `z_tol = 0.02` may be too wide
for strong DLAs (admits noise) and too narrow for weak ones (clips the
truth window when the coarse-scan winner is offset).

**Experiment**: rerun FILTER=1 with `z_tol ∈ {0.01, 0.02, 0.05, 0.10}`.
Track completeness in each NHI bin.

**Code change**: parametrize `z_tol` via the same plumbing as knob 1.

### Knob 3 — `null_evidence` threshold relaxation (line 631)

`logL_null = null_evidence` means "a coarse-scan sample is retained only
if its log-likelihood exceeds the null evidence". For weak DLAs the
high-likelihood region is only marginally above null, so the coarse-scan
winners are sparse. Relaxing the threshold to `null_evidence − δ` admits
more samples into the valid mask, widening the refinement region.

**Experiment**: rerun FILTER=1 with `null_threshold_delta ∈ {0.0 (current),
0.5, 1.0, 2.0}`. Each unit relaxation should retain ~e=2.7× more samples.

**Code change**: add an argument `null_threshold_delta: float = 0.0` to
`select_region_indices_searchsorted` (use `logL_null - delta` as the
retention threshold).

### Knob 4 — fall through on empty mask, don't early-stop (line 635)

When `valid_mask.sum() == 0` (no coarse-scan winner), the code currently
returns the 5000-sample marginal and stops. The principled alternative is
to **fall through to the FILTER=0 path** (evaluate all `NUM_DLA_SAMPLES`)
when the coarse scan fails. This makes FILTER=1 strictly bound by FILTER=0
completeness — at the price of full-sample cost on spectra where the
coarse scan was unlucky.

**Experiment**: code change the line-635 branch to dispatch to the
`filter_low_likelihood=False` path rather than return. Rerun 5k London v3.
This is the cleanest fix that guarantees `FILTER=1 ≥ FILTER=0` completeness.

**Code change**: a few lines around dla_gp.py:635 — instead of computing
the 1-DLA marginal and returning, set `filter_low_likelihood = False` and
fall into the else-branch at line 694.

### Knob 5 — truncated-correction below-null estimator (lines 679-690, 770-797)

Used only for `num_dlas ≥ 1`. The current estimator approximates the
rejected-region marginal using the *mean log-likelihood* of the rejected
samples (line 684-687), with a fallback to `null_evidence` if too few
samples (line 690). For weak multi-DLA cases where the rejected region
carries non-negligible mass, this approximation can bias `log Z_B` low and
therefore the total log evidence low.

**Lower priority** than knobs 1-4 because the multi-DLA case isn't the
dominant pathology — most [20.3, 20.6) misses are 1-DLA.

## Suggested next-session experiment matrix

A 2×2 (knob 1, knob 4) ablation on the existing 5k London v3 sample —
*this is the cleanest test of whether the issue is "missing coarse-scan
coverage" or "early-stop on empty mask"*:

| n_initial floor | empty-mask fall-through | Expected effect |
|---|---|---|
| 5000 (current) | NO (early stop) | baseline FILTER=1 — losing C in [20.3, 20.6) |
| 10000 | NO | partial recovery if coarse-scan coverage is the issue |
| 5000 | YES | partial recovery if empty-mask early-stop is the issue |
| 10000 | YES | full recovery (likely ≈ FILTER=0) |

If the 2×2 ablation cleanly identifies the dominant cause, we can ship
a one-line code change as the FILTER=1 fix and recover production
completeness without paying the FILTER=0 2.4× cost.

Cost: each cell ~30 min wall on a real compute node (FILTER=1 fast path
for spectra where the knobs don't help; FILTER=0-like for the spectra
where they do). All 4 cells parallel on a 256-CPU node = ~30 min total
wall. Inline-via-jupyter or salloc.

## Reference: the FILTER=0 + FILTER fix #5 result

FILTER fix #5 (2026-04-29) already fixed the `num_dlas == 0` case (use
unbiased mean over `initial_logL`) when the coarse scan returns winners.
The remaining FILTER=1 vs FILTER=0 completeness gap therefore comes from:

1. Cases where the coarse scan returns no winners (knob 4 — early stop).
2. Cases where multi-DLA truncated correction underestimates evidence
   (knob 5).
3. Subtle interactions with `z_tol` (knob 2) when the coarse-scan winner
   isn't at the exact truth z.

Knob 1 (`n_initial` floor) is the prevention; knob 4 is the rescue.
Together they should close the gap. Knobs 2 and 3 are fine-tuning.

## Notes for the trainer agent

The FILTER=1 knobs are inference-side. The trainer-PR (running on
GreatLakes) does not touch these. But if the trainer's new clean-forest
LOA model widens the GP signal vs noise (the goal in
[`2026-05-13_model_side_improvements.md`](2026-05-13_model_side_improvements.md)),
the FILTER=1 issue *automatically* gets smaller — because a model with
better discrimination has a wider Δ_marg gap on borderline DLAs, which
makes more coarse-scan samples pass the null threshold, which makes the
valid mask non-empty more often. So the trainer-PR and the FILTER=1
knob-tuning PR work toward the same goal from two sides.
