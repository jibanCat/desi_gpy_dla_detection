# Design note — band Bayes-factor as a DLA/sub-DLA boundary discriminator

**Date**: 2026-05-18   **Status**: **CLOSED 2026-05-18.** The research
questions Q1–Q5 (§2) are answered (§3); the decision and the final flag
spec are in §4. Outcome: ship **`BF_BAND`** = local posterior mass
`P(NHI ≥ 20.3 | local)` as an informational `dlacat.fits` column (not a
`DLAFLAG` gate). Full production spec:
`band_bf_research/PRODUCTION_FLAG_SPEC.md`. §1–§2 below are kept as the
original research plan.

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

## 4. Decision — CLOSED 2026-05-18

The final debug-node test ran (job `53144090`, `band_bf_finalize.py`,
T1–T5). Full spec: `band_bf_research/PRODUCTION_FLAG_SPEC.md`.

- **T1**: the prior-mass-corrected statistic — local posterior mass
  `P(NHI ≥ 20.3 | local)` (bounded [0,1]) — reproduces the raw-ratio
  discrimination **exactly**: AUC **0.759** (Δ = −0.000; rank-corr with
  the raw ratio ρ = 0.994). Median 0.93 (true DLA) vs 0.56 (promoted
  sub-DLA). Ship the corrected form.
- **T2**: z-window ±0.02 **confirmed** — flat AUC plateau ±0.005→±0.05,
  reverts at ±0.10 (sightline contamination).
- **T3**: NHI scatter is QMC-**independent** (MAP scatter flat
  0.276→0.269 dex, 10k→50k) — it is **genuine inference uncertainty**,
  not QMC sparsity. The flag's AUC stabilises by 20k; 50k is past the
  knee; 100k would not help.
- **T4**: NHI is **not** biased high (near-floor mean bias ≈ +0.016 dex).
  The sub-DLA→DLA > DLA→sub-DLA asymmetry is **Eddington bias** — a
  mildly asymmetric per-object leakage rate (11.5 % vs 8.4 %) amplified
  by the steep-CDDF population ratio (detected near-boundary sub-DLAs :
  DLAs ≈ 1.6 : 1).
- **T5**: final column **`BF_BAND`** (float32) = `P(NHI ≥ 20.3 | local)`,
  ±0.02 z-window; informational, **not** in `DLAFLAG`. Higher-purity cut
  `BF_BAND ≥ 0.7`; ROPE "undecided" zone `0.4 ≤ BF_BAND ≤ 0.6`. Companion
  `BF_BAND_NLOCAL` (int32) QMC-noise diagnostic.

`BF_BAND` is a **partial purity proxy — NOT a route to 85/85**; the root
cause (≈3× under-estimated NHI posterior width + Eddington bias at a hard
cut) needs a model-side fix.

## 5. Implementation + validation (2026-05-18)

`BF_BAND` (+ `BF_BAND_NLOCAL`) is implemented in
`tools/postprocess/add_dla_flags.py` — an **informational column, NOT
folded into `DLAFLAG`** (so `DLAFLAG == 0` is unchanged); `run_local.sh`'s
postprocess hook passes `DLA_SAMPLES_FILE` so production runs get it.
`molly_faithful_pc_plots.py` gained `--bf-band-min` as an optional cut.

**Validation** (`bf_band_validation/`, job 53145330) — `BF_BAND` added to
the V1 baseline run, molly headline P/C re-measured:

| variant | purity | completeness |
|---|---:|---:|
| baseline | 0.804 | 0.864 |
| BF_BAND ≥ 0.5 | 0.824 (+2.0) | 0.842 (−2.2) |
| BF_BAND ≥ 0.7 | 0.843 (+3.9) | 0.814 (−5.0) |

**It hurts completeness — it is a P↔C trade, not a free purity gain.** At
≥0.5 the trade is ≈1:1; at ≥0.7 it is worse than 1:1 on the headline. The
earlier "~2:1-favourable / no completeness cost" reading was a
borderline-subset recall figure and does **not** carry to the headline
P/C — applying `BF_BAND` as a filter is an ordinary slide along the P↔C
frontier. This is exactly why it ships as an *optional, informational*
column: the default `DLAFLAG == 0` catalog is unaffected; a consumer who
needs higher boundary purity opts in and accepts the completeness cost.
See `bf_band_validation/FINDINGS.md`.

## 6. Follow-up deep-dive (2026-05-18) — P/C re-sweep + root cause

After the validation showed an unfavourable trade, two further debug-node
investigations: a full design re-sweep scored by P/C (not AUC), and a
root-cause study of the borderline cases.

### 6a. P/C re-sweep — no configuration works (`bf_band_pc_sweep/`, job 53146802)

Re-swept **63 configs** = 3 statistic forms (raw ratio, prior-mass-corrected
ratio, `P(NHI≥20.3|local)`) × 4 band definitions (disjoint, wide, narrow,
overlapping) × 7 z-windows (±0.005…0.10 + global), each over a fine
threshold grid — scored by the **actual molly headline P/C**, not AUC.

**Every one of the 63 configs is a pure P↔C slide along the same
frontier.** Past a ≥10 % cut the trade slope `ΔP/|ΔC|` is ≤ ~1.0 for all of
them; none reaches 85/85; the whole-design-space spread is within the ~1pp
noise floor. Band edges, z-window, raw-vs-corrected form — all immaterial.
AUC misled us because it is a *ranking* metric; thresholding a ranking is a
ROC point, which maps to a frontier slide — AUC > 0.5 only guarantees the
slide is no worse than the p_DLA knob, never that it moves the frontier.

### 6b. Root cause of the borderline cases (`borderline_rootcause/`, jobs 53146968/53147178)

**Why `BF_BAND` is weak**: the borderline-case NHI posteriors genuinely
**straddle 20.3** — single-peaked, ~0.08–0.12 dex wide, real physical
ambiguity (over-estimated sub-DLAs: posterior mean median 0.094 dex from
the cut, `P(NHI≥20.3|local)` median 0.564, 39 % in the [0.3,0.7] ROPE). An
evidence ratio cannot separate classes whose posteriors overlap at the cut
— the AUC-0.76 ceiling is the *data*, not the statistic.

**The root cause** is a two-part NHI noise floor: (1) an **information
limit** — 58 % of borderline cases are SNR<4, corr(logSNR, posterior
width) = −0.76; and (2) an **SNR-independent ~0.19 dex
model-misspecification floor** at all SNR — the decisive evidence is that
the pull `(NHI_pred−NHI_true)/NHI_ERR` *grows* with SNR (1.5→2.9) while the
MAP-error scatter stays flat ~0.24 dex; a systematic the likelihood is
blind to (continuum / mean-flux τ_eff / metals / RSD / Voigt shape).

> **Correction to §3 Q4 / §4 T4.** The earlier "σ ≈ 0.35–0.42 dex, ~3×
> under-estimated NHI_ERR" is superseded. The 0.35–0.42 was a *plain* std
> inflated by ~5 % truth-match artefacts; the **robust scatter is ~0.24
> dex**. `NHI_ERR` is the QMC posterior weighted-RMS (not a parabola-fit
> error) and it *faithfully* reports the per-spectrum posterior width
> (~0.12 dex) — the box truncation is harmless. The real ratio is **~1.9×**,
> and the gap is **not** an under-sized posterior — it is an extra error
> term (model misspecification) that no per-spectrum posterior can contain.

**What fixes it** — neither a sharper posterior-shape statistic nor
post-hoc NHI debiasing can (the defect is scatter, not a mean bias, and the
posteriors genuinely straddle the cut). The durable, model-side moves:
(i) shrink the ~0.19 dex model-misspecification floor via per-spectrum
continuum / mean-flux (τ_eff) marginalisation — evaluate the τ-EB recipe on
σ(dNHI), not just mean bias; (ii) propagate continuum/mean-flux uncertainty
into `NHI_ERR` so the posterior is *calibrated* (pull→1); (iii) replace the
hard 20.3 cut with a continuous-NHI catalog + post-hoc bands so the catalog
stops manufacturing boundary FP/FN. **85/85 at a hard 20.3 cut is
unreachable by any catalog-level knob.** Full detail:
`borderline_rootcause/FINDINGS.md`, `bf_band_pc_sweep/FINDINGS.md`.

## 7. Overnight investigations (2026-05-18) — the root cause, pinned

A five-agent batch, all on debug/regular sbatch (no login-node compute).

### 7.1 — SNR 2–4 root cause: it is unmodelled metal/forest wing absorption

`snr24_rootcause/FINDINGS.md` (job 53156351). **This corrects §6b.** The
"58 % information limit (SNR<4)" framing was wrong — it used the h5 `snrs`
key (narrow Lyα-window forest SNR), not `SNR_RED`. By `SNR_RED`, the **2–4
band has the *highest* borderline misclassification rate (37 %)**, not the
lowest (0–2: 28%, 4–6: 26%, 6–12: 22%, 12+: 19%) — so it is not an SNR
information limit at all.

**The dominant cause (38 % of SNR-2–4 misclassification, 5× any rival):
unmodelled extra absorption in the DLA damping wings** — coincident metal
lines (foreground CIV/SiIV/FeII) and chance Lyα-forest pile-ups. The
single-Voigt forward model (continuum + Lyα forest + one Voigt DLA) has no
term for them, so the GP launders the extra wing absorption into a higher
NHI and pushes the absorber across 20.3. Quantified by `voigt_wing_deficit`
(observed flux vs a Voigt at the *true* NHI, in the 3–15 Å wings):
misclassified −0.158 vs correct −0.066 — the misclassified wings are 2.4×
darker than the true NHI warrants. Continuum placement is **ruled out**
(GP normalisation sub-percent accurate, uncorrelated with dNHI). The
posterior is narrow-but-biased (MAP error 1.9× the posterior width) — the
calibration signature of a systematic the likelihood treats as real DLA
opacity. **This is the concrete identity of the ~0.19 dex
"model-misspecification floor" of §6b.**

### 7.2 — gapped-band BF (`gapped_band_bf/`, job 53155536): no help

11 band-pair configs with the bands placed *away* from 20.3 (gap straddling
the posterior peak). Best gapped AUC 0.777 vs the touching control 0.782 —
*worse*; every config is still a ≤1:1 P↔C slide. The premise (peak-location
degeneracy) is refuted: it is the genuine posterior *overlap*, not the peak
location, that bounds the trade.

### 7.3 — mock vs GP Voigt (`voigt_compare/`, job 53155493): match — ruled out

The DESI mock injects DLAs (`quickquasars` → `desisim.dla` → `linetools`
`voigt_tau`) with an exact `wofz` Voigt, *same* Lyα f=0.4164 and Γ=6.265e8
as the GP detector (`ctypes_voigt.c`/libcerf). Differences (Doppler b, line
count, injection-LSF) do not touch the NHI-setting damping wing. Measured
effective-NHI offset **< 0.001 dex** — 100× below the +0.05 dex bias. A
Voigt mismatch is **not** a contributor. (Side note: `voigt_v2`'s BOSS LSF
kernel is mis-scaled on the fine DESI grid — cosmetic, zero NHI impact.)

### 7.4 — visual review figures (`visual_FP/`, job 53155949)

99 figures in `visual_FP/figures/` — each misclassified spectrum (FP + FN)
with the GP null + GP+DLA Voigt at predicted *and* true NHI overplotted,
every flag annotated, plus `visual_FP/README.md` (flag glossary + index).

### 7.5 — finer τ-EB (`tau_eb_finer/`, job 53155471): NULL — does not help

Re-inference with a 23-point τ-EB factor grid (step 0.25) vs the 8-point
production grid. The finer grid *is* exercised — 43.5 % of spectra land on
a factor the coarse grid lacks — yet every NHI metric is flat-to-slightly-
worse, all within noise: σ(dNHI) 0.238→0.245, pull std 1.845→1.870,
excess std 0.191→0.198, P/C 0.799/0.851→0.797/0.848. **The ~0.19 dex
misspecification floor is τ-EB-grid-independent — it is NOT mean-flux/τ_0
granularity.** This corroborates §7.1: the floor is *localised* wing
absorption, which a better mean-flux point estimate cannot touch. Keep the
production 8-point grid.

### Net

The 85/85 purity gap is now concretely diagnosed and the alternatives are
exhausted: **the GP forward model lacks a term for metal lines / Lyα-forest
pile-ups coincident with the DLA damping wings**, so a fraction of sub-DLAs
get an over-estimated NHI and cross the 20.3 cut. Ruled out as the cause:
the Voigt implementation (§7.3), continuum placement (§7.1 C1), τ-EB grid
granularity (§7.5), and SNR information limit (§7.1). Not a fix: any
catalog knob / p_DLA cut / band-BF re-scoring (§6, §7.2), post-hoc NHI
debias, or higher-SNR data. **The fix is model-side** — a metal-line veto
(DESI foreground-z catalogs) before/during the DLA fit, and a heavy-tailed
wing-residual nuisance term so `NHI_ERR` becomes calibrated — plus
replacing the hard 20.3 cut with a continuous-NHI catalog + post-hoc bands.
