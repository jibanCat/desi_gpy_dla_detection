# Phase-B bounded diagnosis (exploratory)

One bounded diagnostic pass of the observed-N̂ shape failure on the
calibration twin (2LPT-0), per frozen spec §7
(`docs/PHASEB_STATS_SPEC_2026-08-06.md`) and PI ruling 15.
Worktree `/home/mfho/wt_repair_phaseB` @ `853550d`; packs = the fresh
Phase-B `phaseB_packs/modelA_pack_{2lpt0,london0,saclay0}_winlya_only_pad19p0_molly172_bw0p2.npz`
(with `fp_eta_c`); fold = `forward_selftest.selftest`, `resp_clamp="both"`,
zero-sampling truth fold; window N̂ ∈ [19.7, 21.6].

**Every result below is EXPLORATORY** — the scan was prespecified only at the
discriminant level; predicted signatures were stated in each script header
BEFORE the test ran. Ranking is by effect size, morphology match and
cross-mock replication, never by smallest p-value. No new model freedom was
introduced anywhere: every refold uses committed components at committed
conventions; the H3/H4 projections are diagnostic projections, not fits
adopted into the model.

**Sanity gate** (`h00_base_folds.py`): the recomputation reproduces the
closure table exactly — window by_nhat χ²/dof 22.094021781801782 and 3-group
residual (−1760.82, +130.46, +450.25), bit-for-bit vs
`CDDF_analysis/hbi_mcmc/closure_table_phaseB.json`.

## The observed failure, restated

Twin window: χ²/dof 22.09 (survey-only metric); 3-group residual
z = [−1.99, +0.86, +5.93] with full calibration covariance
(G1 [19.7,20.3) over-predicted, G3 [21.0,21.6] under-predicted); twin TOTAL
closes at +0.06%. Same signed tilt on London-0 [−3.26, +1.13, +5.83] and
Saclay-0 [−2.75, +4.30, +2.53].

**Standing caveat (stated up front, applies to every "replication" claim
below):** the signal-side calibration (response, completeness, g, FP counts)
is bit-identical across the three packs, so cross-mock coherence of the tilt
is CONSISTENT WITH a shared signal-calibration shape error — it is not
independent evidence against one. What cross-mock refolds CAN falsify is
whether a fitted correction direction transports.

## Result 1 — the per-bin χ² and the group tilt are two different objects

The survey-metric per-bin window χ² (22.09/dof) conflates two separable
structures:

1. **A low-end sawtooth** (bins 19.7–20.0, survey-only z = −4.8/+8.6/−10.0 on
   the twin, same signs on London/Saclay). This is finite FP-calibration
   shape noise: the 89-event loa-0 `fp_counts` profile enters every pack's μ
   directly per observed bin, and all its in-window mass sits below 20.3.
   Dividing by the per-bin survey+calibration sd (delta method — EXACT here,
   the FP fold is linear in n0 and the E_cov resampling unit is
   n0* ~ Poisson(n0)) collapses those bins to z = −0.7/+2.0/−1.9 and absorbs
   **58% / 66% / 65%** of the per-bin window χ² on twin/London/Saclay
   (χ²/dof 22.09 → 9.37, 28.16 → 9.66, 25.57 → 8.94). G3 is untouched
   (χ² 58.5 → 58.5; zero FP mass above 20.3). This is H9 =
   **ANSWERED-BY-GATE**, now quantified per-bin: it is not a model error, it
   is variance the Layer-B covariance already carries (which is why the G1
   group z is −1.99, not −8.83).

2. **A smooth signed tilt** that survives the calibration band: with-cal
   twin z per bin
   `−0.7 +2.0 −1.9 −1.2 −3.0 −4.1 −5.7 −0.1 −1.7 +3.2 +1.9 +5.0 +3.1 +3.4 +5.3 +3.2 +2.2 +0.3 −2.0`.
   Decomposition of the remaining with-cal window χ² = 178.0 (twin):
   over-prediction run 19.9–20.5 → 35.5%; under-prediction run 20.5–21.0 →
   29.0%; G3 under-prediction 21.0–21.6 → 32.9%; the FP-noise bins 19.7–19.9
   → 2.6%. The full grid adds: over-prediction of every bin 21.7–22.3
   (z −3.5 … −1.0) — mass sits too high at the top. Cross-mock coherence of
   the with-cal per-bin z vector: r(twin, London) = 0.92,
   r(twin, Saclay) = 0.87, r(London, Saclay) = 0.88 (caveat above applies).

Everything below is about structure 2, which carries the leading violation.

## Hypothesis-by-hypothesis record (spec §7 order)

### H1 — stale/inconsistent artifacts: REFUTED
Predicted signature: bit-level diffs in calibration blocks vs the
Phase-A-era window-study packs; residual changes on rebuild.
Observed (`h01_bitcompare.json`): all 32 shared npz keys **bit-identical**
for all three mocks; the only addition is the new `fp_eta_c` field.
Explains 0%.

### H2 — N̂ bin-edge/normalization mismatch: REFUTED
Predicted signature: every-bin alternating residual signs (lag-1 sign
correlation → −1) and strong parity change of the 3-group residual under
±one-bin edge shifts (the spec's ±0.05 dex snapped to the adjacent ±0.1 dex
one-bin shifts — 0.05 cannot land on a bin edge).
Observed (`h02_edges_morphology.json`): lag-1 sign correlation **positive**
(+0.56 / +0.67 / +0.33); only 4/18, 3/18, 6/18 sign changes; group residual
signs STABLE under ±1-bin shifts (G1 always negative, G3 always positive,
G2 crossing smoothly). Rebinning the residual onto the 0.2-dex basis pairs
does NOT collapse χ²/dof (22.09 → 29.92 on pairs): the failure is not
basis-width ripple either. The only sawtooth present is the 19.7–20.0
FP-noise wiggle (see Result 1), which is phase-locked to the FP count
profile, not to the observed edges. Explains ~0% of the tilt.

### H3 — response-kernel shape (ψ_k span): MORPHOLOGY-CONSISTENT, MAGNITUDE- AND TRANSPORT-INSUFFICIENT within existing freedom
Predicted signature (stated first): smooth tilt concentrated at high N̂ (the
D2-clamp/moment-extrapolation region — the anchors top out at 21.04–21.22,
so all of G3 sits where the kernel is weakly measured/clamped); residual
substantially inside the 18-column ψ_k Jacobian span at plausible magnitudes,
replicating cross-mock.
Observed (`h03_h04_jacobians.json`, `h03b_prior_scale_projection.json`):
* Morphology: matches qualitatively — the surviving residual is smooth,
  high-N̂-weighted, G3 is SNR-near-uniform (H10), and the 21.7+ over-run is
  exactly what mass pushed past 21.6 looks like.
* Unrestricted LS on the 18 ψ_k columns (19 bins — near-saturated, caveat
  recorded in the script BEFORE fitting): "explains" 99.5%, but demands
  |ψ_k| up to **320× the prior sd** (mean-surface shifts up to −6.4 dex),
  and the EXACT REFOLD at that point gives χ² 7458 vs the 178 baseline —
  the linearization is invalid; the fit is not a physical kernel
  perturbation. Leading-mode content: top-5 SVD modes carry 74%.
* At the CALIBRATED scale (ridge with the model's own prior sd — CAVEAT:
  `resp_fitcov_diag` is None in these packs, so the sd is the hard-coded
  (0.02, 0.10) fallback): refold-verified absorption **35% (twin),
  39% (London), −10% (Saclay)**; max|ψ|/sd = 1.85. At 3× the scale it
  overshoots and worsens all three (−70%).
* Verdict: the residual is NOT absorbable by the existing calibrated
  kernel-shape freedom (order-0 mu/sig perturbations at the fit-cov scale);
  the fitted direction does not transport to Saclay-0. What remains
  morphologically indicated but UNTESTABLE with existing components is a
  kernel error OUTSIDE the calibrated span — higher-order moment/anchor
  behavior in the weakly-measured region above ~21.05. That repair is new
  model freedom → **PI-GATE**.

### H4 — completeness vs true N (ψ_c span): REFUTED at scale
Predicted signature: excess below N̂ ≈ 20.0 aligned with molly-cell
boundaries (in-window edges 20.0, 20.3, 20.5, 21.0, 21.5).
Observed: the 48 live ψ_c columns saturate the 19-bin marginal exactly
(rank 19) with absurd offsets (max ~3.4 × 10⁵ σ̂) and catastrophic
transport (−59× / −61× on London/Saclay). At the calibrated σ̂ scale the
refold absorbs **3% / 3% / 2%**; at 3× σ̂, 20% / 20% / 14%. The surviving
residual runs cross the molly boundaries smoothly (no step at 20.0 or 20.5
in the with-cal z). Joint ψ_k+ψ_c note: the two spans are almost fully
degenerate on this 19-bin marginal (18 of 19 principal-angle cosines
> 0.9999) — the marginal cannot distinguish kernel-shape from
completeness-shape freedom; only magnitudes/priors and transport separate
them (both fail).

### H5 — matching / multiple-candidate accounting: NOT-TESTABLE-WITH-EXISTING-COMPONENTS
`matching_contract.py` defines ONE fail-closed candidate classification
(P1–P6, pinned order, exactly-one-slot guard); it classifies per-record
inputs that the pack does not carry (the pack holds only the aggregated
`counts`). `extract_pack.py` exposes no alternate sibling-treatment
convention (CLI: pad-floor / completeness-below-floor / basis-width /
window / tag only). Recomputing counts under an alternate convention would
require new extraction machinery — out of bounds for this pass. Recorded,
no computation.

### H6 — clamp behavior: REFUTED as the G3 cause (bracketed both sides)
Predicted signature: changes concentrated in bins fed by clamped covariates
(top and bottom of the basis).
Observed (`h00_base_folds.json`, `h06_clamp_off.json`): the bracket moves
**G1 only**. "hi" (low side unclamped): G1 −1760.8 → −3011.2, G2 130.5 →
40.7, G3 450.2 → **449.6** (Δ < 1 count). The committed diagnostic "off"
(both sides unclamped — the pre-D2 defect) makes G3 WORSE, not better:
450 → 742 (twin), 465 → 803 (London), 191 → 495 (Saclay) — the quadratic
extrapolation pushes basis mass past 21.6, deepening the G3 deficit while
over-predicting 21.7+. Neither end of the committed bracket straddles the
observed G3 residual, and the adopted "both" is the best of the three on
window χ²/dof (22.09 vs 27.93 vs 32.40). The clamp convention is not the
mechanism; note it IS G1-material (−1250), consistent with "clamped-covariate
bins" being the pad-fed bottom.

### H7 — below-support promotion: REFUTED for the leading violation
Predicted signature: G1 changes (more sub-floor truth promoted); if G3 is
untouched, promotion cannot explain the leading violation.
Observed (`h07_pad17p2_fold.json`; pack freshly extracted by the COMMITTED
extractor at this tip, pad 17.2, molly172, 0.2-dex, lya_only —
`h07_extract.log`): Δ(3-group residual) = **[−143.2, −1.3, −0.001]**.
G1's over-prediction deepens slightly (as predicted — more promoted μ), G3
changes by one part in 4×10⁵. Window χ²/dof 22.09 → 22.40. Promotion is
immaterial and cannot touch G3.

### H8 — conditional FP shape within support: STRUCTURALLY INERT (no computation)
For the by_nhat marginal the FP term's N̂ profile is the measured `n0[c,s]`
itself — `fold_mu_fp` is diagonal in c and multiplies `lam_fp = n0/ℓ` by
c-independent factors (w, ℓ, exp(t), E[k,s], (1−η_c) with η piecewise
constant per band). Nothing is imposed on the N̂ axis that could be swapped
for "the calibration's own in-support conditional profile": on THIS marginal
they are the same object. The imposed-E structure lives on the (k,s) axes
and cannot move counts between N̂ bins. Recorded inert for this axis.

### H9 — finite calibration-shape noise: ANSWERED-BY-GATE, quantified
See Result 1. Absorbs 58/66/65% of the survey-metric per-bin window χ²;
collapses the low-end sawtooth; explains 0% of G3 (no FP mass above 20.3;
E_cov calibration variance in G3 ≈ 7×10⁻²¹). Group-level numbers were
already in the gate: G1 z −8.83 (survey-only) → −1.99 (with calibration);
G3 z +5.97 → +5.93.

### H10 — N–SNR interaction (descriptive table)
`h09_h10_calband_snr.json`. Per-stratum 3-group survey-only z (twin):

| SNR stratum | G1 | G2 | G3 |
|---|---|---|---|
| [2,3) | +6.07 | +3.10 | +6.09 |
| [3,4) | +0.58 | −4.35 | +2.91 |
| [4,5) | +0.19 | +0.41 | +1.97 |
| [5,6) | −7.01 | +1.48 | +2.58 |
| [6,7) | −9.22 | −0.51 | +1.77 |
| [7,∞) | −12.61 | +1.44 | −0.55 |

* **G3 is SNR-near-uniform** (positive in 5/6 strata on twin and London,
  4/6 on Saclay, no monotone trend) → kernel-shape-like, not selection-like.
* **G1 hides a large monotone SNR tilt** (+6 at SNR [2,3) → −13/−15 at
  SNR ≥ 7; cross-mock correlation of the stratum pattern 0.97/0.94) that
  cancels in the G1 sum. This is a distinct, real, shared-calibration-shaped
  signature (completeness-vs-SNR or kernel-width-vs-SNR or the truth-fold's
  pathlength-proportional SNR allocation — see "cheapest next test"), and it
  is NOT the G3 driver.

## Ranked table

| rank | mechanism | predicted signature | observed | explained fraction (window residual χ²) | replication | label |
|---|---|---|---|---|---|---|
| 1 | H9 finite FP-calibration shape noise | per-bin wiggles only where FP mass sits (< 20.3), coherent across packs (shared n0) | low-end sawtooth collapses under the survey+cal band; G3 untouched | 58% / 66% / 65% of the SURVEY-metric per-bin χ² (twin/London/Saclay); 0% of G3 | yes (all three mocks, by construction of the shared n0) | ANSWERED-BY-GATE (already in Layer-B covariance; not a model error) |
| 2 | H3 response-kernel shape | smooth high-N̂ tilt in the weakly-measured/clamped region | morphology matches (smooth runs, SNR-uniform G3, 21.7+ over-run); calibrated-scale absorption insufficient; unrestricted fit unphysical (320 prior-sd, refold breaks) | 35% (twin) at the calibrated ψ_k scale | London 39%, **Saclay −10%** — fails transport | exploratory; the in-span version REFUTED at scale; the out-of-span version (high-N moment/anchor behavior) UNTESTABLE here → PI-GATE |
| 3 | H10 N–SNR interaction | selection-like if stratum-concentrated | G1 has a large monotone SNR tilt (+6 → −13) canceling in the sum; G3 SNR-uniform | not quantifiable on the by_nhat marginal (cancels in it) | stratum pattern corr 0.97/0.94 across mocks (shared-calibration caveat) | exploratory, descriptive |
| 4 | H6 clamp behavior | changes confined to clamp-fed bins | G1-material (−1250 under "hi"); G3 Δ<1; "off" worsens G3 (+292) | ~0% of G3; bracket does not straddle the residual | consistent on all three mocks | exploratory; refuted as the cause |
| 5 | H7 below-support promotion | G1 changes; G3 untouched ⇒ cannot explain leading violation | Δ = [−143, −1.3, −0.001] | ≈0% (window χ²/dof 22.09→22.40, slightly worse) | not run on held-outs (twin-only per spec) | exploratory; refuted for G3 |
| 6 | H2 edge/normalization mismatch | every-bin sawtooth + parity flip under shifts | lag-1 sign corr POSITIVE; group signs stable under ±1-bin shifts | ~0% | uniform across mocks | exploratory; refuted |
| 7 | H4 completeness vs true N | excess < 20.0 aligned with molly boundaries | 3% at σ̂ scale; runs cross molly edges smoothly; span degenerate with ψ_k on this marginal | 3% / 3% / 2% | transport catastrophic (−59×) | exploratory; refuted at scale |
| 8 | H1 stale artifacts | bit-diffs; residual changes on rebuild | 32/32 shared keys bit-identical (×3 mocks) | 0% | n/a | exploratory; refuted |
| 9 | H5 matching conventions | localized change where multi-candidate rows concentrate | no alternate convention recomputable from the pack; no extractor flag | — | — | NOT-TESTABLE-WITH-EXISTING-COMPONENTS |
| 10 | H8 conditional FP shape in support | redistribution among G1 bins | the FP N̂ profile IS n0[c,s] on this marginal; nothing imposed to swap | — | — | STRUCTURALLY INERT (this axis) |

## Verdict

* **Per-bin χ²/dof 22.09:** majority-explained (58–66%, reproducible on all
  three mocks) by finite FP-calibration shape noise that the Layer-B
  covariance already carries. This fraction is MATERIAL and REPRODUCIBLE,
  but it is a variance re-classification, not a model repair — the Layer-A
  survey-only metric overstates the per-bin failure.
* **The leading violation (G3 z = +5.93 and the smooth 19.9→21.6 tilt,
  with-cal χ²/dof still 9.4): UNEXPLAINED.** Every mechanism testable with
  existing calibrated components is refuted (H1, H2, H4-at-scale, H6, H7)
  or insufficient and non-transporting (H3-at-scale). The morphology
  (smoothness, high-N̂ weighting, SNR-uniformity of G3, the 21.7+ over-run,
  G3's position entirely above the top response anchors at 21.04–21.22)
  points at response-kernel shape in the weakly-measured region — but
  demonstrating or repairing that requires freedom outside the calibrated
  ψ_k span (higher-order moment terms, re-measured anchors, or a
  recalibrated response product). That is new model freedom / a new
  calibration product → **PI-GATE required**.

## Cheapest discriminating next test (named, not run — outside the frozen list)

Refold the twin selftest with the truth allocated to SNR strata by the
pack's OWN `truth_counts_bks` (an existing pack array) instead of
`forward_selftest.truth_f`'s pathlength-proportional allocation. One script,
one fold, zero new model freedom — but it is not among the ten prespecified
discriminants, so running it belongs to the next PI-approved step. It is
maximally discriminating because the kernel K and completeness C are
SNR-dependent: if the true absorbers are not distributed across strata in
proportion to pathlength, the current truth fold mis-weights the per-stratum
kernels, which can generate BOTH the smooth by_nhat tilt and H10's G1 SNR
tilt with no calibration error at all. If the tilt survives, the
kernel-shape hypothesis (rank 2) is what remains.

## PI gate

**YES.** Required for: (i) any new kernel freedom (moment terms, anchor
extension, response re-measurement) to address the G3 under-prediction;
(ii) any promotion of the H9 variance re-classification into a change of the
Layer-A reporting metric; (iii) authorization of the truth_counts_bks
allocation diagnostic above (outside the frozen §7 list). Nothing in this
pass was adopted into the model; no production code was touched.

## File inventory (this directory)

| file | content |
|---|---|
| `h00_base_folds.py/.json`, `base_*_{both,hi}.npz` | base folds ×3 mocks ×2 clamps + closure-table sanity gate |
| `h01_bitcompare.py/.json` | H1 pack bit-comparison |
| `h02_edges_morphology.py/.json` | H2 morphology + edge shifts + 0.2-dex rebin |
| `h03_h04_jacobians.py/.json`, `jacobians_2lpt0.npz` | H3/H4 unrestricted projections, joint span, principal angles |
| `h03b_prior_scale_projection.py/.json` | H3/H4 calibrated-scale ridge + exact refolds + transport |
| `h06_clamp_off.json` | H6 "off" bracket ×3 mocks |
| `h07_extract.log`, `h07_pad17p2_fold.py/.json` | H7 committed pad-17.2 extraction + fold |
| `h09_h10_calband_snr.py/.json` | H9 per-bin calibration band; H10 SNR-stratum table |
| `findings.md` | this document |

Extraction side-product (left in place, tagged `_pad17p2_diag`):
`/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/modelA_pack_2lpt0_pad17p2_diag.npz` (+ provenance).
