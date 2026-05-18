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

## 3. Findings (2026-05-18) — Q1–Q5 answered

Methodology agent → `band_bf_research/FINDINGS.md`; literature agent →
`~/band_bf_literature_review.md`.

- **Q1 (local vs global): LOCAL wins.** Restricting the band evidence to
  QMC samples with |z_DLA_sample − Z_DLA| ≤ **0.02** lifts AUC 0.726 →
  **0.759**. Wider windows revert toward the contaminated global value;
  narrower starve the sample. Use the local, per-absorber posterior.
- **Q2 (band pair): one scalar, edges don't matter if anchored at 20.3.**
  Wide/nominal/high-wide all tie at AUC 0.759; the split must sit at
  20.3 and bands must not be starved. A nested-band *profile* / LDA does
  **not** beat the single scalar (ratios near-collinear).
- **Q3 (both directions): the band-BF is purity-only, with NO
  completeness cost.** Sub-DLA→DLA AUC 0.759. DLA→sub-DLA: out of
  jurisdiction — 332/338 (98%) of missed DLAs produced *no detection at
  all*, and the band-BF only re-scores existing detections. So it cannot
  rescue boundary FNs, but it also cannot *create* them.
- **Q4 (bias/scatter paradox): RESOLVED — hypothesis confirmed.** The
  mean NHI bias near the floor is small (~0–0.06 dex) but the **scatter
  is σ ≈ 0.35–0.42 dex — ~3× the reported `NHI_ERR`** (median 0.11).
  Two-way leakage across 20.3: sub-DLA→DLA 36.6%, DLA→sub-DLA 16.8%. It
  is **scatter-driven symmetric boundary leakage with a ~3×
  under-estimated NHI posterior width**, not a mean bias. "75% of FPs
  are sub-DLAs" and "bias ≈ 0" are the up-scatter tail and the
  centred-but-wide error of the *same* scatter.
- **Q5 (statistics): the prototype statistic is not formally a Bayes
  factor.** Per the encompassing-prior / Savage–Dickey result, an
  interval BF is (posterior mass in band) / (**prior** mass in band) —
  so the correct statistic uses the per-band **mean** exp(L) (or
  equivalently the local posterior mass `P(NHI ≥ 20.3 | local data)`),
  not the **sum** the prototype used. The literature also flags:
  marginal-likelihood ratios are range-sensitive (Lindley–Bartlett);
  fix band edges on physical grounds; expect a ROPE-style "undecided"
  zone near 20.3; propagate the QMC error.

## 4. Decision & the one remaining step

**The discriminator is real and worth shipping** — local, single scalar,
AUC 0.759, ~2:1-favourable purity trade, *zero* completeness cost (Q3).
But **it must be re-expressed in the statistically-sound form before it
becomes a column**: the **prior-mass-corrected** statistic — i.e. the
local posterior mass `P(NHI ≥ 20.3 | local data)` (bounded [0,1],
range-stable, a genuine probability), not the raw `log(Z_A/Z_B)`. The
AUC is rank-based so the *discrimination* should carry over, but the
threshold and interpretation only make sense in the corrected form.

**Remaining step**: one more debug-node test — recompute on the local
posterior using the prior-mass-corrected statistic `P(NHI ≥ 20.3 |
local)`, confirm the AUC, and set the threshold + a ROPE "undecided"
band. Then finalise the flag spec:
- `P_DLA_LOCAL` (float32) — local posterior mass `P(NHI ≥ 20.3)` for the
  absorber, from QMC samples within ±0.02 in z_DLA. Informational column,
  **not** folded into `DLAFLAG` (like `NHI_CONSISTENCY_FLAG`).
- Downstream: a higher-purity boundary cut + an "undecided" flag for
  objects whose local posterior straddles 20.3.

`BAND_LOGBF`/`P_DLA_LOCAL` stays **deferred** until that final test
confirms the corrected form. It is a partial, no-cost-to-completeness
purity proxy — **not** a route to 85/85; the root cause (3× under-
estimated NHI posterior width) needs a model-side fix.
