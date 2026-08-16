# JOINT C/K/FP LOW-N CONTRACT — PROPOSAL (PI ruling 2026-08-17 item 3)

**Status: PROPOSAL. Ratification is a PI decision.** This formalizes the
low-N response of the Model A fold as a joint (C, K, FP) contract, per the
2026-08-17 ruling: "Do not de-censor K independently. Completeness, kernel
TP matching, and FP/association definitions must use a mutually consistent
truth/detection convention." It extends the ratified estimand principle of
`P1_ESTIMAND_SPEC.md` (the atomic coherent (C, K) pair, the R = C·K
identity, the explicit miss state) to the Model A fold's calibration
triple. Nothing here changes code, packs, guards, or priors.

## 1. The invariants (the contract)

**I1 — K is the detected-and-counted conditional.** The fold's kernel
K(x̂ | N, cell) is defined CONDITIONAL on the detection-and-counting
process that produced the calibration pairs (op-cut S2N_RED > 2,
P_DLA > 0.99, good_mask; x̂ ≥ 19.5). It must never be "de-censored" in
isolation: replacing K with an inferred untruncated kernel while C and FP
keep their deployed definitions double-discounts the below-floor mass.
Measured consequence of violating I1 (2026-08-17 diagnostic D1, all three
families): folded level −9% to −14%, both residual arms inverted,
reporting-grain chi2/dof 450+.

**I2 — one truth/detection convention across the triple.** The TP
definition that selects K's calibration pairs, the definition under which
C's numerators count a truth object as recovered, and the definition that
makes the FP arm the complement (detections attributable to no in-support
truth) must be mutually consistent, so that the fold's partition holds:

    every counted detection is modeled by exactly one arm:
    obs ≈ (C · K · f)-arm  +  FP-arm            (partition invariant)

**I3 — below the identifiability floor, (C·K) is the defined object.**
For N ≲ 19.7 the x̂ ≥ 19.5 floor censors the calibration pairs (measured
censored fraction: ~5% at N = 19.55 rising to ~52% at N = 19.05), so K
alone is not identifiable from natural pairs. Any convention there
(polynomial extension to the anchor edge, moment clamp at 19.7, censored
anchors excluded) is a CHOICE, not a measurement, and must be carried as a
propagated systematic in the spirit of the 2026-07-29 pad ruling. Measured
envelope of defensible conventions (diagnostic D3, three families): folded
total level ±4–5% per family; dN/dX(≥20.0) ±2%; dN/dX(≥20.3) ±0.5%;
reporting-grain arms robust.

**I4 — TP-definition changes are atomic.** Changing the matching/TP
convention for ANY member of the triple requires re-deriving the other two
under the same convention in one atomic build (the `p1_natpair_ck/v1`
atomicity precedent), with the partition invariant re-checked. Swapping
one member alone is refused.

## 2. Current state of the triple (recorded, not changed)

| object | artifact | truth/detection convention |
|---|---|---|
| C | molly TSV (deployed two-chain splice; `p1_natpair_ck_v1` era) | molly recipe recovery counting; live support |
| K | `forward_response_2lpt0.npz` (frozen; deg-2 skew-normal moment surfaces) | natural-pair matching, host = `NHI_TILT_HOST`, x̂ ≥ 19.5, op-cut |
| FP | loa-0 twin product (`build_loa0_fp_product.py`, purity_mixture, (1−η̄)) | HCD-free twin: ALL loa-0 detections are FP by construction (no matching) |

## 3. Partition audit (2026-08-17; 2LPT-0 adopted pack)

Fold side: obs 88,071; mu 88,123 = TP-arm 73,440 + FP-arm 14,683
(FP share 16.7%). Event side (op-cut, x̂ ≥ 19.5): 91,607 detections;
tilt-host TPs 73,845 (non-TP 19.4%); NHI_TRUE TPs 66,481 (non-TP 27.4%).

**Reading: the deployed triple approximately satisfies the partition
invariant under the tilt-host convention** (TP 80.6% + FP 16.7% ≈ 97%,
with the remainder inside the domain differences between the raw op-cut
and the pack's z-window/op-mask). Under the NHI_TRUE convention ~10% of
counted detections would be modeled by neither arm — quantitatively
matching the −0.5% to −5.3% level breaks measured when the NHI_TRUE
kernel is folded against the deployed C/FP (diagnostic D4). The 19,738
tilt-only associations are therefore load-bearing for consistency: they
are either K-population events (current convention) or must move to a
redefined FP/association arm (alternative convention) — never dropped
unilaterally.

## 4. Checkable guards this contract implies (future work, PI-gated)

* G-A (partition): |TP_frac + FP_share − 1| ≤ tol inside the fold's
  counting domain, evaluated at pack-build time.
* G-B (identity): an R = C·K-style closure test on the Model A
  calibration event set (the `P1_ESTIMAND_SPEC.md` §10 pattern).
* G-C (atomicity): pack builds stamp one TP-convention ID shared by the
  C, K, FP inputs; loaders refuse mixed IDs.

## 5. Relation to the 2026-08-17 diagnostics

The calibration-selected kernel representation (ML per-cell deg-2 +
shared cubic; leave-one-group-out validated 15/15) is a REPRESENTATION
change inside the SAME convention (tilt-host, detected-conditional,
censored anchors excluded per I3) — it does not touch this contract. The
independently re-derived deg-2 mean-shape misfit converges with the
already-ratified `P1_ESTIMAND_SPEC.md` §5 finding on the refold side
(+0.071 low edge / −0.035 mid, 4–7σ): the same defect, found twice by
independent routes, in the two kernels' shared estimator class.
