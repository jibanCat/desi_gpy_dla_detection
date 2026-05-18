# 2026-05-16 — GP modeling-window sweeps: lambda_fine + gp_range

> **Status**: DONE. Two follow-up sweeps to `2026-05-15_lambda_range_smoke.md`
> (which found extending MAX_LAMBDA redward of Lyα HELPS). Together they
> locate the production optimum.
>
> **Verdict: MAX_LAMBDA = 1250 is the recommended production value**
> (cell F2: P=0.810 / C=0.870 on London-0 5k). Blue-side (MIN_LAMBDA)
> changes do not help — keep MIN_LAMBDA = 911.75.
>
> **All numbers refreshed 2026-05-17** under the new DLAFLAG convention
> (NHI_INCONSISTENT no longer gated — informational only). Purity is
> ~3pp lower / completeness ~4pp higher than the pre-refresh draft;
> the MAX_LAMBDA=1250 verdict is unchanged — F2 has the sweep's highest
> completeness (0.870) and a competitive purity (0.810).
>
> Sweep roots:
> `/pscratch/sd/j/jibancat/prod533_5k_20260511/lambda_fine_sweep/`
> `/pscratch/sd/j/jibancat/prod533_5k_20260511/gp_range_sweep/`

All cells: post-patch 2-way cellC recipe (PW 50k, MAX_DLAS=3,
SINGLE_ABSORBER_MODEL=1, FILTER=1, τ-EB null, NHI [17.2,22]), London-0
5k slice, fixed molly eval recipe (SNR>2, p_DLA≥0.99, lyb-veto, no-BAL,
λ_rf∈[911,1216], NHI≥20.3, n_truth=581). Same recipe as the lambda_range
sweep, so all numbers here are directly comparable to it.

## 1. gp_range_sweep — blue + red GP-window endpoints (job 53014199)

Varies *both* ends of the GP fit window `[MIN_LAMBDA, MAX_LAMBDA]`.

Numbers refreshed 2026-05-17 (new DLAFLAG convention).

| Cell | range (Å) | P | C | note |
|---|---|---:|---:|---|
| B0 | [851.0, 1216.75] blue extended | 0.7697 | 0.8173 | blue end −61 Å |
| B1 | [920.0, 1216.75] blue narrowed | 0.7717 | 0.8266 | blue end +8 Å |
| baseline | [911.75, 1216.75] | 0.7719 | 0.8173 | = lambda_range L0 |
| R1 | [911.75, 1240] | 0.8041 | 0.8514 | red end +23 Å |
| L1 | [911.75, 1260] | 0.8118 | 0.8545 | = lambda_range L1 |
| L2 | [911.75, 1300] | 0.8088 | 0.8514 | = lambda_range L2 |
| R2 | [911.75, 1330] | 0.8017 | 0.8514 | red end +113 Å |
| R3 | [911.75, 1360] | 0.8035 | 0.8483 | red end +143 Å |

**Blue side is inert.** Extending blue to 851 (B0) or narrowing to 920
(B1) both leave P/C within ~1pp of baseline. The blue Lyα-forest region
is dominated by absorption noise the GP already models; moving the
endpoint adds no clean constraint. **Keep MIN_LAMBDA = 911.75.**

**Red side reproduces lambda_range.** R1/L1/L2/R2/R3 trace the same
red-extension Pareto gain: extending the red endpoint past Lyα raises
purity ~+3pp (0.77→0.80–0.81) *and* completeness ~+3.4pp (0.817→0.85),
then plateaus — completeness stays ~0.85 flat all the way out to 1360,
with no fall-off. (A pre-refresh draft saw a decline at R3; that was a
DLAFLAG-gating artifact.)

## 2. lambda_fine_sweep — fine MAX_LAMBDA grid 1228–1300 (job 53019665)

Fine-steps the [1240, 1260] transition where lambda_range jumped from
C=0.817 to C=0.855.

Numbers refreshed 2026-05-17 (new DLAFLAG convention).

| MAX_LAMBDA | cell | P | C | P·C |
|---:|---|---:|---:|---:|
| 1216.75 | L0 (coarse) | 0.7719 | 0.8173 | 0.631 |
| 1228 | F0 | 0.7924 | 0.8390 | 0.665 |
| 1240 | R1 (coarse) | 0.8041 | 0.8514 | 0.685 |
| 1244 | F1 | 0.8041 | 0.8514 | 0.685 |
| **1250** | **F2** | **0.8098** | **0.8700** | **0.704** |
| 1256 | F3 | 0.8118 | 0.8545 | 0.694 |
| 1260 | L1 (coarse) | 0.8118 | 0.8545 | 0.694 |
| 1272 | F4 | 0.7982 | 0.8452 | 0.675 |
| 1288 | F5 | 0.7994 | 0.8390 | 0.671 |
| 1300 | L2 (coarse) | 0.8088 | 0.8514 | 0.689 |

**F2 (1250) is the Pareto-best balanced point**: the **highest
completeness in the sweep (0.870)**, purity (0.810) within ~0.2pp of the
sweep top, and the highest P·C product (0.704). It strictly
Pareto-dominates the historical 1216.75 baseline (+3.8pp P, +5.3pp C).

The completeness column has a clear F2 peak at 1250 (0.870), with
neighbours 1244/1256 at 0.851 — a ~2pp bump, above the ~1pp determinism
noise floor. 1250 is the clear pick — top completeness *and* the highest
P·C product, with no purity sacrifice vs neighbours. **1250 is
recommended.**

## 3. Production recommendation

**Set MAX_LAMBDA = 1250 for the 1M production run.** Keep MIN_LAMBDA at
911.75. The London-0 evidence is strong (F2 vs the 1216.75 baseline:
+3.8pp purity *and* +5.3pp completeness — a strict Pareto win); the
off-distribution check is the `lambda1250_crossval` sweep (Saclay-0 +
2LPT-0, 1216.75-vs-1250 paired) — confirm 1250 still Pareto-leads there
before locking it in.

Cost: extending the window to 1250 adds ~14% pixels to the Woodbury
inversion; wall impact is small (O(n_px²) on a sub-dominant term).
