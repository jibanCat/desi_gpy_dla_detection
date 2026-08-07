# FROZEN diagnostic stopping rule — P1 completeness investigation

**Frozen 2026-08-06, BEFORE any aggregate cause fraction, ledger total,
or gap decomposition was computed or inspected** (rulings §6). The only
completeness numbers seen at freeze time are the ones already in the
Stage-2A bridge verdict (the pooled 0.43–0.58 vs 0.81–0.99 comparison
and the per-bin bridge Δ's) — no mechanism attribution existed.

## Numerical criteria (may not be weakened after aggregates are seen)

1. **C_molly reproduction tolerance:** the event-level reconstruction
   must reproduce the deployed `molly_n_det` and `molly_n_tot` EXACTLY
   (integer equality) in every (SNR, N) cell, OR every non-reproducing
   cell must carry an explicit, individually diagnosed discrepancy
   source. No scientific interpretation before this gate.
2. **Attribution requirement:** in each load-bearing true-N range —
   [20.0, 20.4), [20.4, 21.0), [21.0, 21.5) — at least **80%** of the
   natural-vs-injection completeness gap must be assigned to mechanisms
   at evidence Level A, B, or C (rulings §19). The ranges
   [19.5, 20.0) and ≥ 21.5 are reported with the same machinery but do
   not gate (the P1 transition support will not reach below 20.0, and
   above 21.5 both selections saturate).
3. **Residual bound:** the unexplained residual gap (Level D/E), treated
   as a bounding transfer uncertainty on C_inj over the candidate P1
   support, must project through the preimage sensitivity map to
   **≤ 50 counts on G3** (i.e. below the 75-count bridge tolerance with
   margin) and ≤ 500 counts on G1. Projection formula: for each range,
   |residual-gap fraction| × (range's folded μ contribution to the
   group), summed over ranges in the P1 support, taken at the
   conservative (upper) end of the order-sensitivity bounds.
4. **Design-change screen:** the investigation is NOT complete while any
   open finding could still change: the P1 parent population; the
   completeness denominator; the miss-state definition; the conditioning
   set; the transition support; or the holdout criterion. Each Tier
   report must answer these six explicitly (yes/no/bounded).

## Stop conditions

* **STOP (success)** when 1–4 all hold. Optional further Tier-3
  diagnostics are then PROHIBITED (rulings §6: no continuation past the
  frozen decision requirement).
* **STOP (escalate)** if after Tier 2 the attribution requirement (2) is
  unreachable with authorized data — report the achievable attribution,
  the bounded residual, and its P1 consequence; the checkpoint then
  carries an estimand/support finding rather than a completed
  decomposition. Do not silently start Tier-3-style campaigns to force
  the number.
* **Never stop merely because one plausible narrative exists** — every
  gate above is quantitative.

## Order-sensitivity requirement (rulings §12)

The primary-cause hierarchy is frozen in the ledger script header before
its first run; the cause budget is re-computed under at least one
defensible alternative ordering; ranges are reported as [lower, upper]
attribution bounds; criterion 2 uses the LOWER bound.
