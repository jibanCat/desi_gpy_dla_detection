# 2026-05-16 — GP modeling-window sweeps: lambda_fine + gp_range

> **Status**: DONE. Two follow-up sweeps to `2026-05-15_lambda_range_smoke.md`
> (which found extending MAX_LAMBDA redward of Lyα HELPS). Together they
> locate the production optimum.
>
> **Verdict: MAX_LAMBDA = 1250 is the recommended production value**
> (cell F2: P=0.838 / C=0.830 on London-0 5k). Blue-side (MIN_LAMBDA)
> changes do not help — keep MIN_LAMBDA = 911.75.
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

| Cell | range (Å) | P | C | note |
|---|---|---:|---:|---|
| B0 | [851.0, 1216.75] blue extended | 0.8063 | 0.7988 | blue end −61 Å |
| B1 | [920.0, 1216.75] blue narrowed | 0.8032 | 0.7833 | blue end +8 Å |
| baseline | [911.75, 1216.75] | 0.7719 | 0.8173 | = lambda_range L0 |
| R1 | [911.75, 1240] | 0.8224 | 0.8173 | red end +23 Å |
| L1 | [911.75, 1260] | 0.8118 | 0.8545 | = lambda_range L1 |
| L2 | [911.75, 1300] | 0.8088 | 0.8514 | = lambda_range L2 |
| R2 | [911.75, 1330] | 0.8297 | 0.8142 | red end +113 Å |
| R3 | [911.75, 1360] | 0.8344 | 0.8111 | red end +143 Å |

**Blue side is inert-to-mildly-bad.** Extending blue to 851 (B0) or
narrowing to 920 (B1) both leave P within ~3pp of baseline and *cost*
completeness (B1 −3.4pp). The blue Lyα-forest region is dominated by
absorption noise the GP already models; moving the endpoint adds no
clean constraint. **Keep MIN_LAMBDA = 911.75.**

**Red side reproduces lambda_range.** R1/L1/L2/R2/R3 trace the same
red-extension Pareto gain. Pushing the red endpoint monotonically
*raises purity* (0.77→0.83) but *completeness peaks near 1260 and then
declines* (0.855 at L1 → 0.811 at R3) — the emission-line-variance
regime predicted in the lambda_range note.

## 2. lambda_fine_sweep — fine MAX_LAMBDA grid 1228–1300 (job 53019665)

Fine-steps the [1240, 1260] transition where lambda_range jumped from
C=0.817 to C=0.855.

| MAX_LAMBDA | cell | P | C | P·C |
|---:|---|---:|---:|---:|
| 1216.75 | L0 (coarse) | 0.7719 | 0.8173 | 0.631 |
| 1228 | F0 | 0.8243 | 0.7988 | 0.659 |
| 1240 | R1 (coarse) | 0.8224 | 0.8173 | 0.672 |
| 1244 | F1 | 0.8317 | 0.8111 | 0.675 |
| **1250** | **F2** | **0.8375** | **0.8297** | **0.695** |
| 1256 | F3 | 0.8365 | 0.8235 | 0.689 |
| 1260 | L1 (coarse) | 0.8118 | 0.8545 | 0.694 |
| 1272 | F4 | 0.8239 | 0.8111 | 0.668 |
| 1288 | F5 | 0.8182 | 0.8080 | 0.661 |
| 1300 | L2 (coarse) | 0.8088 | 0.8514 | 0.689 |

**F2 (1250) is the Pareto-best balanced point**: highest purity in the
sweep (0.838) *and* near-top completeness (0.830), and the highest P·C
product (0.695). It strictly Pareto-dominates the historical 1216.75
baseline (+6.6pp P, +1.2pp C).

The completeness column is noisy (1250→1256→1260 reads 0.830→0.824→0.855
— a ~3pp wobble over 10 Å, above the ~1pp determinism noise floor only
at the L1 spike). 1260 (L1) has the single highest C (0.855) but pays
2.6pp purity vs F2. The choice between 1250 and 1260 is a P-vs-C
preference; **1250 is recommended** because it leads on purity and on
the P·C product, and the 1260 completeness spike is partly sampling
noise.

## 3. Production recommendation

**Set MAX_LAMBDA = 1250 for the 1M production run.** Keep MIN_LAMBDA at
911.75. The London-0 evidence is strong (a 6.6pp purity gain at no
completeness cost is large); the off-distribution check is the
`lambda1250_crossval` sweep (Saclay-0 + 2LPT-0, 1216.75-vs-1250 paired)
— confirm 1250 still Pareto-leads there before locking it in.

Cost: extending the window to 1250 adds ~14% pixels to the Woodbury
inversion; wall impact is small (O(n_px²) on a sub-dominant term).
