# 2026-05-17 — NHI_INCONSISTENT flag investigation

> **Status**: DONE (sbatch job 53078990, debug QOS, 2026-05-17).
> **Verdict: NHI gate OFF (k=0) for the production headline** under the
> completeness-first directive. The `NHI_INCONSISTENT` flag is FP-enriched
> (genuinely discriminating) but not a clean filter, and every k>0 trades
> completeness for purity. Keep it as an informational column only.
>
> Investigation dir:
> `/pscratch/sd/j/jibancat/prod533_5k_20260511/nhi_flag_investigation/`

## Why

`molly_faithful_pc_plots.py` gates the headline P/C on `DLAFLAG==0`, and
`DLAFLAG` is dominated by `NHI_INCONSISTENT` (bit 5: `NHI − k·NHI_ERR < 20.3`,
default k=0.5) — which flags ~79–86% of catalog rows. The gate's `k` had
never been validated and the flag's usefulness was unproven. The question,
under the completeness-first directive (memory `feedback_completeness_first`):
is it a **smart** false-positive filter, or a **blunt** P↔C knob?

## Method

`_investigate.py` on cellC C0 (London-0 5k, baseline GP model), fixed molly
recipe, catalog analysis only (no inference). Reuses molly's catalog load +
truth match + cuts.

## A. NHI_ERR is mis-calibrated; NHI is biased high

265 truth-matched headline DLAs (5 excluded for degenerate `NHI_ERR<1e-3`):

- **Raw NHI bias** (`NHI_pred − NHI_true`): mean **+0.077 dex**, median
  **+0.055 dex**, std 0.155. The GP over-estimates NHI by ~0.05–0.08 dex.
- **Pull** (`resid / NHI_ERR`), robust: median **+0.66**, MAD-σ **1.42**.
  `NHI_ERR` **under-estimates** the true NHI scatter by ~40%, and the
  non-zero pull median reflects the +0.06 dex central bias.

So the `k·NHI_ERR` threshold is built on an under-estimated error bar around
a biased central value — it cannot be made clean by tuning k alone.

## B+C. k-sweep — FP-enriched but not clean

| k | P | C | n_flagged | FP%_flagged | FP%_kept | TP_lost |
|--:|--:|--:|--:|--:|--:|--:|
| 0.0 | 0.780 | **0.836** | 0 | — | 22.0% | 0 |
| 0.25 | 0.799 | 0.814 | 17 | 58.8% | 20.1% | 7 |
| 0.5 | 0.814 | 0.799 | 29 | 58.6% | 18.6% | 12 |
| 0.75 | 0.834 | 0.793 | 39 | 64.1% | 16.6% | 14 |
| 1.0 | 0.847 | 0.771 | 52 | 59.6% | 15.3% | 21 |
| 2.0 | 0.885 | 0.694 | 93 | 50.5% | 11.5% | 46 |

- The flag **is FP-enriched**: flagged rows are ~3× more FP-rich than kept
  rows at every k (≈50–64% vs ≈9–22%). It discriminates — not random noise.
- But it is **not clean**: at k=0.5, 41% of flagged rows are real DLAs; the
  gate discards 12 true DLAs (21 at k=1.0). Every k>0 trades completeness
  for purity (k=0.5: −3.7pp C / +3.4pp P; k=1.0: −6.5pp C / +6.7pp P).

## Verdict

**NHI gate OFF (k=0) for the production headline P/C.** It is a P↔C trade,
and the completeness-first directive says don't make that trade now. Keep
`NHI_CONSISTENCY_FLAG` / `NHI_INCONSISTENT` as an **informational column**
(a 59%-FP-rate subset is a useful handle for a later high-purity cut), but
do not gate the catalog or the headline metric on it.

Re-evaluate this once the GP model is swapped (the β-collapsed baseline may
itself drive part of the NHI bias) and after the NHI-bias / `NHI_ERR`
recalibration task — only then could the flag become a trustworthy gate.

## Follow-up: the NHI-bias task is now concretely characterised

`NHI_pred` biased **+0.06 dex high**, `NHI_ERR` under-estimated by **~1.4×**.
That is the scope of the deferred "make NHI less biased" work.
