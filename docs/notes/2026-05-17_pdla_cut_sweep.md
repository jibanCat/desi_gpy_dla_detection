# 2026-05-17 — p_DLA-cut sweep: is 85/85 reachable?

> **Status**: DONE (sbatch job 53089164, debug QOS).
> **Verdict: NO — a stricter p_DLA cut cannot reach the 85/85 P/C target
> on the current (β-collapsed baseline) GP model.** Purity never reaches
> 0.85 at any cut, on either of the two best 2-way cells. The p_DLA cut is
> a real purity lever (~+2–3pp from loosest to tightest) but it is not
> enough — there is a ~2–3pp purity shortfall that knob-tuning cannot
> close. The remaining lever is the **GP-model swap** (`model_sweep`).
>
> Sweep dir: `/pscratch/sd/j/jibancat/prod533_5k_20260511/pdla_cut_sweep/`

## Why

Production target = **85% purity AND 85% completeness**. The best configs
sit at completeness ~0.85–0.87 but purity ~0.78–0.81 — so the gap is
purity, and the p_DLA cut is its dedicated lever (tightening the cut drops
marginal detections → purity up, completeness down). This sweep asks
directly: is there a p_DLA cut where **both** ≥ 0.85?

## Method

Catalog analysis only (no inference). For the two strongest 2-way cells —
**F2** (MAX_LAMBDA=1250, the production candidate) and **C7** (PW 100k) —
swept the p_DLA cut over a fine grid (`1 − 10^x`, x from −1 to −9 in
0.25 steps) using the molly machinery, new DLAFLAG convention.

## Result

| cell | P at default 0.99 | C at default 0.99 | max purity (any cut) | balanced point (max min(P,C)) |
|---|---:|---:|---:|---|
| F2 (MAX_LAMBDA=1250) | 0.810 | 0.870 | **0.833** (at C=0.786) | P=0.823 / C=0.820 |
| C7 (PW 100k) | 0.791 | 0.855 | **0.816** (at C=0.768) | P=0.806 / C=0.811 |

**Neither cell's purity ever reaches 0.85** — F2 caps at 0.833, C7 at 0.816,
both only at the extreme cut `p_DLA ≥ 1−1e-9` where completeness has
collapsed to ~0.79. The P and C curves cross near `log_pdla ≈ −6` at
**~0.82 / ~0.82** for F2 and **~0.81 / ~0.81** for C7 — i.e. the best
*balanced* operating point the p_DLA cut can deliver is ~0.82/0.82,
about 3pp short of 85/85 on both axes.

F2 dominates C7 across the whole curve (the MAX_LAMBDA=1250 advantage
holds at every p_DLA cut).

## Interpretation

The p_DLA cut works exactly as a purity lever should — tightening it
moves ~+2.3pp purity for ~−8pp completeness across the full range — but
the **purity frontier of the current model is too low**: even spending
all available completeness it tops out ~0.83. 85/85 requires lifting the
whole P/C frontier ~3pp, which is not a catalog-cut operation.

This is consistent with the standing finding that the sweep baseline GP
model is **β-collapsed** (memory `project_baseline_model_beta_collapse`).
The next lever is the **GP-model swap**: the `model_sweep` (job 53077686)
tests 4 healthy phase2_desi `_m` models. If a healthy model lifts the
frontier, re-run this p_DLA sweep on the winner to locate the 85/85 point.
If even the healthy models fall short, 85/85 needs deeper work (NHI-bias
fix, better inference) — not knob tuning.

## Production p_DLA-cut recommendation

Until the model swap lands, no p_DLA cut delivers 85/85. The pragmatic
operating points on F2 are:
- **p_DLA ≥ 0.99** (default): P=0.810 / C=0.870 — completeness-rich.
- **p_DLA ≥ 1−1e-6** (balanced): P=0.823 / C=0.820.

Pick after the model swap; the convention is otherwise an open item.
