# Design note — band Bayes-factor as a DLA/sub-DLA boundary discriminator

**Date**: 2026-05-18   **Status**: RESEARCH / DESIGN — the `BAND_LOGBF`
flag is **NOT yet added to `dlacat.fits`**. It is deferred pending the
open questions below. This note is the research plan.

## 1. Where this came from

The FN/FP deep-dive (`2026-05-18_fn_fp_deepdive.md`) found the 85/85
purity gap is dominated by the NHI 20.3 boundary: ~75% of false positives
are *real* sub-DLAs (true NHI 19.0–20.3) detected with NHI_pred > 20.3.
Post-hoc NHI point-estimate debiasing was tested and **refuted**
(`2026-05-18_boundary_purity_tests.md`) — it is a 1:1 P↔C trade.

A prototype **band Bayes factor** was then tried: from the QMC samples,
form band-restricted evidences and take a ratio,

    log BF = log( Z[logNHI ∈ 20.3,20.6] / Z[logNHI ∈ 20.0,20.3] )

Prototype result (`band_bf_test/`): on borderline detections it
discriminates true DLAs from over-estimated sub-DLAs at **AUC 0.726** —
a real, modest signal, ~2:1-favourable trade. Promising enough to
research properly, **not** promising enough to ship as-is.

## 2. Open questions — must be answered before adding the flag

### Q1 — Local (per-absorber) vs global posterior

The prototype summed over **all** QMC `z_DLA` samples on the sightline —
that is a *global* band evidence, not specific to the absorber being
classified. A sightline with a second absorber elsewhere contaminates it.
**The BF must be computed *local* to the absorber in question**: restrict
the QMC samples to those whose `z_DLA` is close to that absorber's
`z_DLA_MAP` (the MAP redshift of that specific detection). Re-do the
band-BF with a local (z-windowed) posterior and compare AUC local vs
global. Decide the z-window width.

### Q2 — Which band pair(s)?

`[20.3,20.6] / [20.0,20.3]` was an arbitrary first guess. The BF is known
to be sensitive to the integration range (see the literature review,
Q5). Test a family and pick on discriminating power:
- wide:    `log Z[20.3,21.6] / Z[19.0,20.3]`
- nominal: `log Z[20.3,20.6] / Z[20.0,20.3]`
- narrow:  `log Z[20.3,20.4] / Z[20.2,20.3]`
- possibly a **vector / profile** of nested band ratios rather than a
  single scalar, if no single pair dominates.
Report AUC (both directions, see Q3) for each.

### Q3 — Both misclassification directions

The prototype only measured sub-DLA→DLA (the false positives). We **also**
care about **DLA→sub-DLA** — true DLAs (NHI_true ≥ 20.3) whose NHI_pred
fell below 20.3 and dropped out of the catalog (boundary false
negatives). The band-BF must be evaluated as a **two-way** classifier and
**both** directions reported: how well does it (a) flag over-estimated
sub-DLAs sitting in the catalog, and (b) rescue under-estimated DLAs that
fell out. A flag that only does (a) trades completeness silently.

### Q4 — Resolve the bias-vs-scatter paradox

Apparent contradiction to resolve: the NHI-debias test found the NHI bias
on truth-matched **TP DLAs** is ≈ 0 near the 20.3 floor — yet ~75% of FPs
are sub-DLAs promoted across 20.3. How can both be true?

**Working hypothesis** (to verify): this is **scatter-driven boundary
leakage at a hard threshold**, not a mean bias.
- The "bias ≈ 0" was measured on true-DLA detections (NHI_true ≥ 20.3) —
  one population.
- The promoted sub-DLAs are a *different* population (NHI_true < 20.3),
  and they are **selected** by NHI_pred > 20.3 — i.e. the positive tail
  of the sub-DLA NHI_pred error distribution. You cannot see this from a
  TP-DLA bias fit.
- With NHI_ERR under-estimated ~1.6× (real scatter larger than reported),
  a hard cut at 20.3 leaks **both ways**: sub-DLAs scatter up (FP),
  DLAs scatter down (boundary FN). A ~0 mean bias is fully consistent
  with large symmetric leakage.
**Verify** by characterising NHI_pred − NHI_true on the *full sub-DLA
population* (true NHI 19–20.3, regardless of detection) and on TP DLAs,
and quantifying the two-way leakage rate across 20.3. If confirmed, the
flag's job is to catch *scatter* leakage — which is exactly why an
evidence-ratio (posterior-shape) discriminator can beat a point-estimate
correction.

### Q5 — Statistical grounding

How do statisticians use Bayes factors / evidence ratios for subtle
borderline classification? Range sensitivity, interval/partial Bayes
factors, Savage–Dickey, ROPE, the Lindley–Bartlett paradox, etc. — see
the companion literature review (separate agent). Feeds Q2.

## 3. Decision

`BAND_LOGBF` is **deferred** — do not add it to `dlacat.fits` until Q1–Q5
are resolved. Two agents are dispatched 2026-05-18:
- a literature-review agent (Q5);
- a methodology agent (Q1–Q4, compute on a debug node) → writes to
  `band_bf_research/`.
The flag spec (column name, exact band definition, local-posterior
window, threshold, one scalar vs a profile) will be finalised here once
both report.
