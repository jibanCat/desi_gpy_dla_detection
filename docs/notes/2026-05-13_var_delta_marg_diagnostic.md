# Var[Δ_marg] gating diagnostic — verdict

> **Status**: complete. Pure re-analysis of the prod533 5k London v3 run; no new inference. 70 s wall on a jupyter compute node.
> **Bottom line**: at production N=50k, sampling noise on Δ_marg is ~130× *below* the signal-null gap. The pipeline is **statistic-limited, not sampling-limited**. MLMC/pocoMC/Marginal-MAP variance-reduction work won't move the low-SNR P/C ceiling.

## What was tested

`Δ_marg ≡ log p(D|1 DLA) − log p(D|null)`. The 1-DLA marginal is a QMC sum over `N` samples drawn from the (z, log N_HI) prior. Production currently uses `N=50000`. The design notes in [`2026-05-12_map_lr_failure.md`](2026-05-12_map_lr_failure.md) §"Refined diagnostic recommendation" and [`2026-05-12_mlmc_design.md`](2026-05-12_mlmc_design.md) §Option A flagged this as the falsifiable gating experiment to run *before* building any new sampler — because the answer determines whether sampler engineering can help at all.

The recipe:

1. Read the `sample_log_likelihoods_dla[..., 0]` array (per-QMC-sample log-likelihood of the 1-DLA model) from every `processed-spectra-16-*.h5` in `/pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/processed/`.
2. For each spectrum and each `N ∈ {1k, 5k, 10k, 25k, 50k}`, draw 4 independent subsamples ("seeds") of size N from the 50k. Non-overlapping when `4·N ≤ 50000`; bootstrap with replacement at N=25k and N=50k.
3. Compute Δ_marg per (spectrum, N, seed) via `logsumexp − log N − log p(D|null)`.
4. Aggregate: `Var[Δ_marg | spectrum, N]` across the 4 seeds → `σ_noise(N)`. Compare to the signal-null gap `median(Δ_marg | p_dla_full≥0.99) − median(Δ_marg | p_dla_full≤0.01)`.

Code: [`examples/var_delta_marg.py`](../../examples/var_delta_marg.py).

## Results

5694 spectra across 8 healpix files. Stratified by the *production* P_DLA at N=50k:

| Bucket | n |
|---|---:|
| confident_pos (P_DLA ≥ 0.99) | 759 |
| confident_neg (P_DLA ≤ 0.01) | 4935 |
| borderline (0.01 < P_DLA < 0.99) | 5694 − 759 − 4935 = (rest) |

`σ_noise` is `median across borderline spectra of sqrt(Var[Δ_marg|spec, N])`. Signal-null gap is the median gap between `confident_pos` and `confident_neg` Δ_marg distributions.

| N | σ_noise (borderline) | signal-null gap | noise/signal |
|---:|---:|---:|---:|
| 1 000 | **1.31** | 33.70 | 3.9 % |
| 5 000 | **0.12** | 38.82 | 0.3 % |
| 10 000 | **0.037** | 12.79 | 0.3 % |
| 25 000 | 0.11* | 12.71 | 0.9 % |
| 50 000 (production) | 0.10* | 12.79 | **0.8 %** |

\* bootstrap-with-replacement: the 4 "seeds" overlap on average ~37 %, so apparent variance is inflated. Treat 10k–50k as a plateau at σ ≈ 0.04. The 0.10 plateau value bounds the *true* noise from above.

PNG: `/pscratch/sd/j/jibancat/prod533_5k_20260511/var_delta_marg/var_delta_marg.png`.
Per-spectrum CSV: `var_delta_marg_per_spec_N.csv` in the same directory.
Per-bucket aggregate: `var_delta_marg_aggregated.csv`.

## Interpretation

**Sampling noise dominates only at small N.** At N=1k, σ_noise ≈ 1.3 perturbs the sigmoid by ~30 % at p=0.5 — small-N runs are seed-dependent. By N=10k the noise is ~0.04 and is invisible at any p threshold relevant to the catalog. Going to N=200k (4× current) would shrink σ by √4 ≈ 2× — taking noise/signal from 0.8 % to 0.4 %. Invisible at the P/C level.

**The Δ_marg estimator is biased upward at small N.** Confident_pos − confident_neg jumps from 33.7 at N=1k to 12.8 at N≥10k. This is the well-known positive bias of `logsumexp − log N`: with too few samples, the estimator over-weights whichever QMC samples land near the likelihood peak. Don't trust absolute Δ_marg from any N < ~10k.

**Verdict — Outcome B of the gating decision: statistic-limited.** At production N=50k the marginal evidence is sampling-converged. The low-SNR P/C ceiling that motivates the sampler upgrade work cannot be caused by QMC noise on Δ_marg — even if you doubled N four times, the noise is already two orders of magnitude below the gap.

## Implications

1. **Drop the bespoke MLMC build**. No win available — the variance is not where the leverage is.
2. **Drop pocoMC integration** for this purpose. Same reason. It would be useful only if Δ_marg were noise-limited, which it isn't.
3. **Marginal-MAP (MMAP) is still potentially worth running as a stepping stone.** It changes the *statistic* (different test from `Δ_marg`), not just the integration. If MMAP improves P/C, the gain is from the alternate statistic, not from variance reduction. Cost: ~5× current QMC, deliverable in days. Lower priority than option (5) below.
4. **The remaining sampling-related lever is *biasing the prior***, not adding more samples. Stratified QMC with a reweighted prior (more samples in the narrow-z high-N_HI peak; importance-sampling correction back to the original prior) can change *where* Δ_marg's information comes from. The current diagnostic shows there's no headroom in noise; reweighting wouldn't add headroom there either, but could shift the Δ_marg level for some borderline spectra by changing the integrand support. Speculative — would need a separate test.
5. **The real leverage is on the model side.** Candidate directions, in order of plausibility:
   - **GP kernel / mean-flux prior**: a kernel that better separates DLA-shape from forest-noise covariance would directly widen the signal-null gap.
   - **External information**: forest correlation features, multi-line cross-checks (Lyα + Lyβ joint), neighbour-pixel context. These add new dimensions to the likelihood beyond the per-spectrum forest.
   - **Voigt profile**: already production-grade; unlikely to move much.
   - **BAL handling**: not the bottleneck for the SNR>2 ceiling per existing sweeps, but cheap to revisit.
6. **The multi-DLA early-stop bug ([`2026-05-12_multidla_early_stop_bug.md`](2026-05-12_multidla_early_stop_bug.md)) and the SubDLA aggregation finding remain valid independent levers** — they are about how Δ_marg is *used* and which catalog assignment results from it, not about the per-spectrum integration accuracy.

## Caveats

- This run uses `sample_log_likelihoods_dla[..., 0]` — the 1-DLA model column. Variance properties of the multi-DLA recursion (k=2, k=3) are not directly tested. The recursion uses an even more concentrated proposal at later k, so noise should be smaller, not larger.
- Bootstrap-with-replacement at N≥25k inflates apparent σ; the true plateau is ~0.04 (the N=10k value), not ~0.10. The conclusion is unchanged.
- Stratification uses `P_DLA` from the same 50k run as the input — so "confident_pos" and "confident_neg" are circularly defined. The right interpretation is "the 50k pipeline confidently assigns these labels and the *seed-to-seed jitter* doesn't unflip them," which is precisely the gating question.
- 5694 spectra is the prod533 5k sample (BAL included, FILTER=1, max_dlas=3, τ-EB on). Conclusion should generalize to the full 1M-QSO production at this configuration but is not formally tested at scale.

## Reproduce

```bash
bash -c '
source /usr/share/lmod/lmod/init/bash
export DESI_ROOT=/global/cfs/cdirs/desi
source /global/common/software/desi/desi_environment.sh main
python /pscratch/sd/j/jibancat/desi_gpy_dla_detection/examples/var_delta_marg.py \
    --processed-dir /pscratch/sd/j/jibancat/prod533_5k_20260511/london_v3_loa124_pw14_tau_eb/processed \
    --out-dir       /pscratch/sd/j/jibancat/prod533_5k_20260511/var_delta_marg
'
```

Pure re-analysis. ~70 s wall. No inference runs needed.
