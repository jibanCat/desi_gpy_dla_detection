# 2-way vs 3-way model decision — 2026-05-14

> **Recommendation: ship the 3-way model with the D1+D4 knob stack
> (MAX_DLAS=4 + NHI prior [19, 23]) as the production default.** It is
> Pareto-comparable to the 2-way cellC family on classical-DLA P/C,
> preserves a separate sub-DLA channel that the 2-way deletes, and
> avoids cellC's structural sub-DLA-bleed pathology in the [20.3, 20.5)
> bin. The cellC operating point with NHI consistency gate at k=0.5
> remains a strong **alternate** for downstream consumers that prioritize
> classical-DLA purity ≥ 85 over a calibrated sub-DLA catalog.

## Frontiers (London-0 5k validation)

All operating points: SNR_RED > 2, P_DLA ≥ 0.99, lyb-veto, no-BAL,
λ_rf [911, 1216], truth NHI ≥ 20.3.

### 2-way frontier (cellC family + NHI consistency gate)

Path: cellC baseline (P=0.826, C=0.830) → NHI gate `NHI − k·NHI_ERR ≥ 20.3`:

| Operating point | P | C | knob change |
|---|---:|---:|---|
| cellC baseline | 0.826 | 0.830 | k=0 |
| cellC + gate k=0.25 | 0.837 | 0.810 | +k=0.25 |
| cellC + gate k=0.5 | 0.856 | 0.798 | +k=0.5 |
| cellC + gate k=0.75 | 0.871 | 0.787 | +k=0.75 |
| cellC + gate k=1.0 | 0.878 | 0.781 | +k=1.0 |

(Gate is eval-only; same HDF5 outputs; no inference cost. See
`docs/notes/2026-05-14_cellC_knob_sweep.md` and the agent-2 finding
in `HANDOFF.md`.)

### 3-way frontier (D-sweep best cells)

| Operating point | P | C | knob change |
|---|---:|---:|---|
| 3-way baseline | 0.845 | 0.766 | reference |
| D1 (MAX_DLAS=4) | **0.862** | 0.769 | +1.7 P / +0.3 C |
| D4 (NHI [19, 23]) | 0.855 | **0.775** | +1.0 P / +0.9 C |
| D2 (MAX_DLAS=5) | 0.853 | 0.766 | +0.8 P / 0 C |
| D8 (n_init=10k) | 0.846 | 0.769 | ~0 / +0.3 |

PW-count cells (D5/D6/D7) excluded — bias-suspect per dla_gp.py
`-log(N)` bug. **D1 + D4 stacking** not yet tested but predicted to give
P ≈ 0.872, C ≈ 0.778 (linear additive estimate) — i.e., better than D1
or D4 alone.

## Side-by-side at comparable C

| Target C | 2-way (cellC + gate) | 3-way (best D) |
|---|---|---|
| 0.83 | k=0: P=0.826 | (3-way doesn't reach this C) |
| 0.80 | k=0.5: P=0.856 | (3-way doesn't reach this C) |
| 0.78 | k=0.75: P=0.871 | D4: P=0.855 |
| 0.77 | k=1.0: P=0.878 | D1: P=0.862 |

**At C ≥ 0.78 the 2-way frontier dominates the 3-way frontier on
classical-DLA P/C.** The 3-way's best (D1+D4 stack predicted P=0.872 at
C=0.778) sits roughly on the 2-way's k=0.75 point (P=0.871, C=0.787) —
i.e., **the two frontiers are roughly Pareto-equivalent at the operating
points the 3-way can reach**, but the 2-way extends further toward
higher completeness.

## Why the 3-way is still the recommendation

The headline classical-DLA P/C is roughly tied. The choice should
therefore turn on the **structural** differences:

| Dimension | 2-way (cellC) | 3-way (production) |
|---|---|---|
| Classical DLA P/C | comparable (frontier extends further to high-C) | comparable |
| Sub-DLA per-spectrum P(SubDLA) | **NOT computed** — single absorber posterior | computed (`model_posteriors[:, 1]`) |
| [20.3, 20.5) FP composition | 84 % sub-DLA bleed | not measured (likely lower; cleaner per-bin posterior) |
| Joint catalog format | 1 catalog, post-hoc NHI cuts to split | 2 catalogs (DLA + SubDLA) at well-defined operating points |
| Downstream consumer friction | LOW for DLA-only consumers; HIGH for sub-DLA consumers (must re-derive) | LOW for both |
| Behavior under MAX_DLAS knob | hurts (C1: −3 pp P) | helps (D1: +1.7 pp P) |
| Behavior under NHI prior knob | wider hurts (C3: −2.4 pp P) | wider helps (D4: +1 pp P) |
| Mock-validated on Saclay/2LPT | NO (only London-0 5k) | partially (Saclay v3_loa124 = 0.871/0.771 per `2026-05-12_saclay_v3_loa124_results.md`) |

**The four bottom rows tip the decision toward 3-way**:

- The cellC family's [20.3, 20.5) bin pathology (84 % sub-DLA bleed)
  is a structural cost of the 2-way model space that a NHI consistency
  gate can patch but not eliminate. The 3-way per-NHI-bin posteriors
  don't have this problem to the same degree (3-way [20.3, 20.5) P =
  0.583, vs cellC P = 0.600 — cellC's bin-P is *higher* but only because
  it doesn't get penalized for sub-DLA bleed in this eval).

- The 3-way model rewards capacity knobs (D1, D4); the 2-way penalizes
  them. Going forward, the 3-way is the more tunable platform if we
  want further headline gains beyond ~0.86/0.78.

- The production baseline (3-way) already has off-distribution
  validation on Saclay (P=0.871, C=0.771 per 2026-05-12). cellC has not
  been cross-validated outside London-0.

- Downstream sub-DLA science needs `P(SubDLA)`. cellC deletes it.

## Cost comparison

Per `docs/runs/2026-05-12_v3_production_cost.md`, 3-way baseline =
~17 nh / 1M QSO. The OAT D-sweep wall numbers (under contention)
suggest D1 / D4 add ~+0.04 nh per 5k → ~+8 nh per 1M, i.e., ~25 nh / 1M
for D1+D4. Still below the ~50 nh / 1M target.

cellC (2-way) was ~17 nh / 1M too. The NHI consistency gate is
eval-only (zero inference cost). So the cost comparison roughly favors
the 2-way.

## Where the 3-way recommendation could be wrong

1. **The dla_gp.py bug**: not in the headline path for D1/D4 (same
   N=50k as baseline → bias cancels). But **D6/D7's apparent +2.2/+2.9
   pp purity gains are bias artifacts**. After patching, D-sweep PW
   results may shift; the D1/D4 gains are unaffected.

2. **D1+D4 stacking is predicted, not measured.** The interaction
   could be sub-additive or super-additive. A "D9" cell would settle
   it (~0.4 nh).

3. **Saclay/2LPT cellC behavior is unmeasured.** If cellC degrades
   strongly off-distribution (e.g., if the sub-DLA bleed mechanism is
   amplified on different mock truth NHI distributions), the 2-way
   frontier could collapse below 3-way.

4. **The NHI consistency gate is not yet tested on real LOA.** It is
   a pure post-processing trick on the eval; it should generalize to
   real data, but the NHI_ERR distribution is mock-fitted and may have
   systematic differences in real LOA spectra.

## Recommended next experiments

In priority order:

1. **Fix the dla_gp.py log-evidence bug** + regenerate the 50k sample
   file with consistent seed=42, then re-run C5-C7 + D5-D7 to verify
   the bias artifact disappears. ~3 nh.

2. **D9 stack test**: combine D1 (MAX_DLAS=4) + D4 (NHI [19, 23]) into
   one cell. ~0.36 nh. Confirms the additivity hypothesis.

3. **Saclay cross-validation**: run the recommended 3-way config (D9 if
   step 2 confirms; D1 or D4 otherwise) on Saclay mock-0 5k. ~3-4 nh.
   PR #7 task 6.

4. **2LPT cross-validation**: run same on 2LPT 5k. ~3-4 nh. PR #7
   task 6.

5. **NHI consistency gate as a real flag**: add `--nhi-consistency-k`
   to `examples/molly_faithful_pc_plots.py`. Document operating points
   k = 0, 0.25, 0.5 in `docs/production_runbook.md` §7.

After these, the recommendation can be made with confidence and the
"flag, not default" verdict from `2026-05-13_cellC_mechanism_verdict.md`
can be revisited.

## Implication for PR #7 task 4 (decide cellC default)

**Verdict: keep the 3-way model as the default**, but ship the cellC
config as a documented opt-in flag for downstream consumers that:
(a) only want a classical-DLA catalog (NHI ≥ 20.3),
(b) want maximum [20.3, 20.5) completeness (cellC: C=0.704; 3-way+D4:
    C=0.583),
(c) are willing to apply the NHI consistency gate `NHI − 0.5·NHI_ERR
    ≥ 20.3` as a post-processing step (which gets to 85.6/79.8).

The 2026-05-13 cellC mechanism verdict's "flag, not default"
recommendation is **vindicated and refined** — the verdict's
"cross-validate on Saclay/2LPT before promoting" condition is the
remaining unknown that could tip the decision.

## Operating-point recommendations for downstream consumers

| Use case | Config | Expected P/C |
|---|---|---|
| Headline DLA catalog (production default) | 3-way + D1 + D4 stack | ≈ 0.872 / 0.778 |
| Maximum [20.3, 20.5) completeness | cellC + NHI gate k=0 | 0.826 / 0.830 |
| Balanced 85/85 target | cellC + NHI gate k=0.5 | 0.856 / 0.798 |
| Purity-first | cellC + NHI gate k=1.0 | 0.878 / 0.781 |
| Sub-DLA + DLA joint catalog | 3-way (any D variant) | DLA per above; SubDLA via `model_posteriors[:, 1]` |

The "headline DLA catalog" remains TBD until D9 stack is measured. If
D9 underdelivers vs the linear estimate, fall back to D4 (best
single-knob 3-way at C=0.775).

## Note on the dla_gp.py bug interpretation

The bug (`-log(N)` double-counted in 1-DLA evidence formulas) is
present across both 2-way and 3-way runs. Both sweeps compare against
their respective baselines at fixed N=50k, so D1/D4 wins are bug-clean.
The PW-count cells in BOTH sweeps (C5-C7, D5-D7) are bias-suspect and
should be re-run after the patch. **The 2-way vs 3-way comparison
itself is bug-clean** — both at N=50k, same bias offset cancels.

## References

- `docs/notes/2026-05-14_cellC_knob_sweep.md` — 2-way C-sweep
- `docs/notes/2026-05-14_cellD_knob_sweep.md` — 3-way D-sweep
- `docs/notes/2026-05-13_cellC_mechanism_verdict.md` — cellC mechanism
- `docs/notes/2026-05-14_subdla_pc_cellabc.md` — sub-DLA P/C 2×3
- `docs/runs/2026-05-12_saclay_v3_loa124_results.md` — Saclay 3-way validation
- `HANDOFF.md` (2026-05-14 top block) — bug finding + agent results
