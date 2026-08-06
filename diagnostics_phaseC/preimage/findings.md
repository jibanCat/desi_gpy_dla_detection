# Phase-C response-preimage analysis (PI ruling 2026-08-06 §6)

**Status: PLANNING MAP ONLY.** Every number here is computed THROUGH the
current response (`build_K` at the truth-equivalent point, `resp_clamp="both"`),
so this analysis cannot validate its own preimage (PI §6). It exists to place
anchors and support margins; it is not evidence about where the failure
physically originates.

Routine: `run_preimage.py` (this directory). Packs: the Phase-B
`modelA_pack_{2lpt0,london0,saclay0}_winlya_only_pad19p0_molly172_bw0p2.npz`.
Sanity gates (all three mocks): (i) recomputed 3-group residual ==
`closure_table_phaseB.json` to 1e-6 (fail-loud); (ii) Σ_b M[b,c] reconstructs
the fold's signal μ to <1e-8 relative; (iii) the diagnostic numpy-oracle
kernel copy reproduces the committed `build_K` bin masses to 1.0e-14.

## 1. The preimage of observed G3 [21.0, 21.6] (twin; London/Saclay replicate)

Signal μ(G3) = 5685.8 counts (FP contributes 0 above N̂ = 20.3). Split by
true-N origin — the numbers the anchor design must cover:

| true-N origin | counts | share |
|---|---|---|
| true N < 21.0 (upward migration) | 720.9 | **12.7%** |
| true N ∈ [21.0, 21.6] | 4775.3 | 84.0% |
| true N > 21.6 (downward migration across the ceiling) | 189.5 | 3.3% |

plus 213.4 counts of true-[21.0,21.6] mass predicted to land ABOVE the
ceiling (the other side of the ceiling exchange). Replication: London
12.5% / 83.8% / 3.6%, Saclay 12.5% / 83.9% / 3.6% (shares nearly identical
by construction — the signal-side calibration is bit-identical across packs;
only the truth populations differ).

**G3 feed coverage (min true-N set):** 95% from true [20.7, 21.7); 99% from
[20.3, 21.7); 99.9% from [19.9, 21.9). Identical bins on all three mocks.

Per-true-bin table (twin), `preimage.json` `table`:

| true bin | truth | →G1 | →G2 | →G3 | →[21.6,22.4) | G3 share |
|---|---|---|---|---|---|---|
| [20.3,20.5) | 8455 | 1719 | 6214 | 36 | 0 | 0.6% |
| [20.5,20.7) | 6598 | 29 | 6193 | 134 | 0 | 2.4% |
| [20.7,20.9) | 4823 | 0 | 4121 | 534 | 1 | 9.4% |
| [20.9,21.1) | 3257 | 0 | 1324 | 1886 | 6 | **33.2%** |
| [21.1,21.3) | 2108 | 0 | 90 | 1952 | 37 | **34.3%** |
| [21.3,21.5) | 1127 | 0 | 3 | 938 | 171 | **16.5%** |
| [21.5,21.7) | 530 | 0 | 0 | 180 | 339 | 3.2% |
| [21.7,21.9) | 196 | 0 | 0 | 9 | 185 | 0.2% |

## 2. Support classification of the G3 feed (the core design fact)

The current response's per-cell anchor ranges (`resp_N_fit_range`) top out at
**21.04–21.22** (9 cells; z edges [0, 2.56, 2.96, ∞] × SNR edges
[2, 3.5, 6.5, ∞]). Against the feed above:

* **47.3% of G3's predicted μ comes from true bins whose response covariate
  is CLAMPED** (bin center above the cell's top anchor): 27.1% from
  [21.1,21.3), 16.6% from [21.3,21.5), 3.3% from [21.5,21.7)+ (twin 0.473,
  London 0.472, Saclay 0.476).
* The remaining ~53% is dominated by [20.9,21.1) (33%) and [20.7,20.9)
  (9.4%) — nominally inside the anchored range, **but thinly**: the source
  envelope (`track_c/stage0/forward_response_2lpt0.npz`) has SEVEN
  equal-count empirical anchors per cell with ~40 matched pairs each, and the
  TOP anchor (centered 21.04–21.22) is the ONLY one above ~20.77. The
  quadratic moment fit therefore interpolates over ~0.3–0.5 dex exactly
  where the G3 sensitivity peaks (§3), supported by ~40 pairs/cell ≈ 360
  pairs total.
* No G3-relevant true-N region is "unsupported" in the label sense (the
  clamp guarantees no silent extrapolation), but [21.1, 22.1] is
  **clamped-not-measured** and [20.6, 21.1] is **measured-but-single-anchor**.

## 3. Sensitivity map = the §9 effect-size scale

dG3 per +0.02 dex kernel-mean shift of one true bin (twin / London / Saclay
agree to ~10%):

| true bin | dG3 (+0.02 dex mean) | dG3 (+10% width) |
|---|---|---|
| [20.5,20.7) | +27 | +49 |
| [20.7,20.9) | +100 | +78 |
| [20.9,21.1) | **+186** | −44 |
| [21.1,21.3) | +12 | −45 |
| [21.3,21.5) | −37 | −26 |
| [21.5,21.7) | −26 | 6 |

Readings for the design:

* The G3 prediction is most sensitive to the kernel MEAN at true
  [20.7, 21.1) — ≈ **14,300 counts/dex** summed over the two bins. The
  +450-count discrepancy corresponds to a coherent **+0.031 dex** mean bias
  confined there. That is the scale of "a perturbation large enough to
  explain G3" (criterion 2, §9).
* Inside the clamped region the sign flips (a +mean shift pushes mass past
  the ceiling): a −0.10 to −0.15 dex mean correction over [21.1, 21.5)
  yields both ΔG3 > 0 AND less predicted mass at 21.7+, matching the
  observed two-sided morphology (G3 deficit + 21.7+ over-prediction). A
  width reduction over [20.9, 21.5) acts in the same direction.
* Precision consequence: to hold the response-induced σ(G3 prediction)
  ≤ 150 counts (= 1/3 × 450), the aggregated kernel-mean uncertainty over
  true [20.7, 21.1) must be ≲ **0.010 dex**, with proportionally looser
  requirements per conditioning cell (exact allocation in the design doc).
  For reference, the CURRENT top-anchor sampling gives ≈ 0.15 dex/√40 ≈
  0.024 dex per cell ≈ 0.008 dex pooled — comparable to the requirement,
  with the clamped-region systematic on top of it. The new measurement must
  therefore both DENSIFY [20.5, 21.1] and EXTEND [21.1, 22.1].

## 4. Conditioning decomposition of G3 (what "measured" must cover)

* By response cell (sr × zr), twin: all 9 cells contribute 262–1178 counts;
  no cell is negligible (min share 4.6%).
* By SNR stratum: strata [2,3)…[7,∞) contribute 1477 / 1020 / 712 / 503 /
  385 / 1588 — BOTH ends of the SNR axis are material; anchors cannot sit
  only at high SNR (PI §10).
* By coarse z: 2738 / 1943 / 1004 — all three z cells material.

## 5. Design consequences (feed §8 of the rulings; frozen in PHASEC_CALIB_DESIGN.md)

1. **Production anchor support: true N ∈ [20.3, 22.1]** = the 99% feed
   region [20.3, 21.7) widened by the conservative margin (PI §6): one
   0.2-dex bin below (→ 20.3, already the 99% edge; 0.999 reaches 19.9 via
   bins that contribute <0.1% each — covered by bridge anchors instead),
   and above the ceiling to 22.1, because true [21.7, 21.9) still lands
   189.5 counts in G3 and the ceiling exchange (213 counts) must be
   measured, not extrapolated.
2. **Bridge anchors: true N ∈ [19.5, 21.1]** overlapping the old anchors
   (esp. the well-populated 19.5–20.8 range) — same estimand test, PI §7.
3. **Anchor density must peak at [20.5, 21.3]** (sensitivity + single-anchor
   thinness), not at the highest N.
4. Anchors must be stratified over all 9 response cells AND spot-check the
   fine-SNR extremes (s=[2,3) and [7,∞) carry the largest G3 shares).
5. The response estimand includes the ABOVE-CEILING columns (migration
   accounting on both sides of 21.6), so the new measurement's observed
   axis must extend past the reporting ceiling exactly as the pack grid
   does (to 22.4), NOT stop at 21.6.

## Files

| file | content |
|---|---|
| `run_preimage.py` | the committed routine (sanity-gated, provenance-stamped) |
| `preimage.json` | full tables ×3 mocks (schema `phaseC_preimage/v1`) |
| `preimage_M_{mock}.npz` | the full M[b,c] preimage matrices |
| `findings.md` | this document |
