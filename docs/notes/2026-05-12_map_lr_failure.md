# MAP+LR detection failure on London 8f

> **2026-05-12.** Hybrid catalog experiment: v3_loa124 + Method A (null-quantile prior-marginal) + Method B (MAP detection with NHI optimizer over [17, 22]). Method B failed; this note documents what happened and what to test next.

## Setup

- **Catalog**: London mock-0 8 healpix files, v3_loa124 GP inference (PW14 [19, 22] for the marginal QMC; **Method B used `pw_samples_a3_172_220_50000.mat` for the wider [17, 22] optimizer scan**).
- **n_initial sweep** (3 missed cands in v3 scope + 10 strong-truth + 20 SNR>2 nulls): output `(z_MAP, log NHI_MAP, log_LR)` is **identical to 4+ sig figs at n=5k / 10k / 50k**. Timing 3.0 / 4.9 / 22.4 s per spec. **Picked n_initial = 5000.**
- **Method A**: null-quantile of `Δ_marg = log_likelihoods_dla[0] − log_likelihoods_no_dla` on the SNR>2 BAL-excl null population (n_null=1683).
  - p90=+6.31, p95=+15.86, p99=+35.70.
- **Method B**: scipy.optimize.minimize over (z, log NHI) with bounds (z_search, [17, 22]). Classify:
  - MAP log NHI < 19 → `b_subdla_lls` (null).
  - MAP log NHI ≥ 19 AND `log_LR = log p(D|MAP) − log p(D|null) > τ_LR` → B-detected.
  - Lyβ veto applied post-MAP.
- **τ_LR** = p95 of `log_LR` on the same null population (NHI≥19 nulls, n=377).
  - log_LR p90=+45.2, p95=+65.3, p99=+93.2 — a **fat right tail**.

## Result (per-spec, SNR>2, BAL-excl, truth=309)

| Cell | n_det | P | C |
|---|---:|---:|---:|
| baseline P_DLA>0.99 | 229 | 85.6% | 63.4% |
| A only @ p95 | 227 | 85.0% | 62.5% |
| A∪B @ p90/p90 | 603 | **43.6%** | 85.1% |
| A∪B @ p95/p99 | 361 | 65.1% | 76.1% |
| A∩B @ p95/p99 | 119 | **92.4%** | 35.6% |

(Caveat: per-spec eval, so multi-DLA truth specs count once. Per-DLA molly C numbers ~10pp higher in absolute terms; the *shape* — A∪B trades P for C, A∩B trades C for P — is the load-bearing observation, not the absolute C.)

## What broke Method B

**The [17, 22] optimizer over-fits forest noise as broad weak DLAs at logN ∈ [20.5, 21]** — not at the [17, 19] boundary as the working hypothesis had it. Out of 377 NHI≥19 nulls, log_LR has a fat tail extending to +90+, driving B-only purity to 20-29% at every threshold.

**Mechanism (corrected 2026-05-12)**: MAP+LR removes the **Occam volume penalty** that protects the baseline marginal. The baseline `p_DLA = p(D|DLA) / [p(D|null) + p(D|subDLA) + p(D|DLA)]` evaluates `p(D|DLA)` as an integral over the (z, NHI) prior. A narrow likelihood peak averaged against ~5 z-units × ~3 log-NHI-units of mostly-empty prior gets a large `−½ log|H|` Occam factor. Even though `p(D|θ_peak)` is high, the marginal `∫p(D|θ)p(θ)dθ` is moderate. Method B computes `log p(D|θ_MAP) − log p(D|null)` — peak value, no integral, no Occam factor. The peak height alone wins. Noise-overfit ghosts pass.

**Sub-DLA does NOT compete directly for the logN=20.5 region**: the sub-DLA prior is [19, 20.3]. Ghosts at logN ≈ 20.5 are *outside* that range. So this is not a "missing sub-DLA term in the denominator" problem — it's specifically a missing Occam volume penalty (a separate problem). The sub-DLA omission is a second issue that affects detections in the [19, 20.3] band, but is not the cause of the logN ≈ 20.5 ghost tail.

(Earlier draft of this note attributed the failure to sub-DLA competition — that was wrong; corrected.)

## Recoveries on the 5 known-missed candidates

| TID | In v3 scope? | Method B verdict |
|---|---|---|
| 105798 | ✓ | MAP logN=18.79 → correctly `b_subdla_lls` |
| 1798 | ✓ | B-detected |
| 80198262 | ✗ (different healpix) | — |
| 64988 | ✓ | B-detected |
| 20115135 | ✗ (different healpix) | — |

3/3 in-scope. Mixed: 1 correctly downgraded as sub-DLA range, 2 recovered.

## Lyβ veto: minimal impact

Only 2 specs flagged across the full catalog, neither in the p95 B-set. The veto matters more for multi-DLA hybrids — single-DLA MAP rarely locks onto Lyβ in practice because the Lyα fit is geometrically constrained.

## Next experiments

1. **Laplace-correct the MAP-LR** — the direct fix for the logN ≈ 20.5 ghost problem. Add the saddle-point Occam factor: `log p(D|MAP) − log p(D|null) − ½ log|H| + (d/2) log(2π) + log p(θ_MAP) − log p(null)`. The earlier MAP-Laplace prototype on n=48 nulls dropped FP rate from ~30% → ~22%. Worth re-running on full London 8f with the [17, 22] prior to see if it survives population scale.
2. **Adaptive importance sampling / MLMC** at the MAP seed — keep Bayesian framing throughout. See `docs/notes/2026-05-12_mlmc_design.md` for the longer-form proposal. This is the principled fix.
3. **Optional structural cleanup**: drop sub-DLA model and extend DLA NHI prior to [17, 22] or [19, 23]. The current sub-DLA [19, 20.3] overlaps DLA [19, 22] in production — the marginal correctly splits mass there, but it's bookkeeping overhead. Dropping sub-DLA and absorbing its range into DLA simplifies the pipeline. Does NOT directly help the logN ≈ 20.5 ghost problem (those live outside sub-DLA range) but cleans up the model hierarchy. Affects downstream f(N,z) labelling, not the math.

## Artifacts

`/pscratch/sd/j/jibancat/prod533_5k_20260511/null_quantile_map_combined/`:
- `dlacat_v3_loa124_combined.fits` (458 rows: 57 A_only, 170 A_and_B, 231 B_only, 0 B_filtered_lyb)
- `RESULTS.md`, `combined_summary.json`, `method_a_summary.json`, `method_a_per_spec.fits`, `method_b_all.json`
- `nsweep/`, `figures/`, `logs/`
- Scripts: `step1_method_a.py`, `step2_nsweep.py`, `step3_method_b.py`, `step4_combine_eval.py`

## Status of the working hypothesis

The prior-session belief that **widening NHI to [17, 22] would let ghost peaks slide below 19 and be correctly classified as null** is **partially refuted**: it works for the few cases where the truth is genuinely a sub-DLA (e.g., TID 105798 — MAP found logN=18.79), but does NOT eliminate the much larger population of noise-overfit "weak DLAs" at logN ≈ 20.5. **The boundary stack at 19 was never the dominant pathology; it was a small subset of the actual ghost problem.**
