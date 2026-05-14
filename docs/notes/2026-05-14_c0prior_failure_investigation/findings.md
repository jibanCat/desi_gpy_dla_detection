# c0prior model DLA-detection failure — investigation

Date: 2026-05-14. Branch: `claude/debug-trainer-from-v1`.

## Summary

The c0prior hypothesis as stated is **largely wrong**. The c0prior model is **not** fundamentally broken; its single-point 1-DLA log-likelihood at the canonical-TID truth is **+18 nats above null** — well into "DLA preferred" territory. The `p_DLA = 0.042` failure is the production pipeline's **early-stopping fallback** because the QMC-marginalised 1-DLA evidence (a different quantity from the point likelihood at truth) lands ~3 nats below prior-weighted null. The multi-DLA NaN is the consequence of that early stop at `gpy_dla_detection/dla_gp.py:790-810`, not a Cholesky singularity (`_m` also returns NaN for k=3,4 on the same target — see `docs/notes/2026-05-13_step_c_dla_recovery/stepc_2lpt_loa124_nohcd_nobal_wide_m.json:17-19`).

Headline structural difference between the two models is **not** the log_c_0 endpoint (both ended at log_c_0 ≈ -3.8, i.e. c_0 ≈ 0.02 — the c0prior anchoring failed). It is the **normalisation band** ([1310, 1325] rest vs [1425, 1475]) and the resulting 13× larger `‖M‖_F²`. On 10 random strong DLAs the two models behave identically (7/10 detected, same 3/10 missed). Canonical TID 120046865 is the *only* divergent case — an outlier sitting right at the truncated-sampling threshold.

Recommendation: drop the c0prior recipe. Fix the (c_0, M) degeneracy via reparameterisation or L2 on M; keep the standard [1425, 1475] norm band.

## Trained endpoint scalars (c0prior vs _m)

Read from each `phase2_result.h5` (`kernel_conditioning.log`):

| quantity | c0prior | _m |
|---|---:|---:|
| `log_c_0` | -3.9210 | -3.7476 |
| c_0 | 0.01982 | 0.02357 |
| `log_tau_0` | -6.4828 | -6.3473 |
| τ_0 | 1.529e-3 | 1.751e-3 |
| `log_beta` | 0.9427 | 1.0884 |
| β | 2.5668 | 2.9694 |
| norm band (Å rest) | **[1310, 1325]** | **[1425, 1475]** |
| ‖M‖_F² | **21,317** | 1,648 |
| M max singular value | **144.4** | 31.95 |
| cond(K) | **9.6e4** | 1.3e4 |
| K max eigenvalue | **2.09e4** | 1.03e3 |
| K min eigenvalue | 0.217 | 0.079 |
| c_0² × eigval_max | **8.20** | 0.57 |
| μ mean | 1.204 | 1.415 |

The `log_c_0_history` for both models initialises at log(0.1) = -2.3026 and drifts monotonically over 1500 Adam iterations down to ~-3.8. The c0prior Gaussian prior (`tests/phase2_train_desi.py:261-262`) merely **slowed** the drift — the endpoint differs from `_m` by only 0.17 dex. **The "c0prior" label is misleading in inference terms.** The README's "log_c_0 prior σ: (none)" is a re-emit artifact (`examples/reemit_step_c_readmes.py:110-112` reads the field from `.h5` which doesn't preserve it). The prior σ value is not recoverable from artifacts on disk.

Critical second-order effect: because c0prior's `c_0` decayed slightly slower, the (c_0, M) factor-analysis degeneracy resolved by inflating `‖M‖_F` 13×. The combined `c_0² × eigval_max(K)` is 8.2 for c0prior vs 0.57 for `_m` — c0prior's effective continuum prior is 14× too wide for normalised flux.

## Multi-target inference test (10 random strong DLAs)

Selection: 2lpt loa-124 mock-0 truth DLAs with logNHI ∈ [20.6, 21.5] and SNR > 3 (n=10, seed 13). Script: `sample_strong_dlas.py`. Inference: same `DLAHolder` config as `examples/dla_recovery_step_c.py`. Full log: `multi_target.log`.

| TID | truth NHI | c0prior p_DLA | c0prior ΔNHI | _m p_DLA | _m ΔNHI |
|---:|---:|---:|---:|---:|---:|
| 180058672 | 20.789 | 1.0000 | +0.015 | 1.0000 | +0.018 |
| 350004868 | 20.992 | 1.0000 | +0.045 | 1.0000 | +0.021 |
| 160221272 | 20.712 | **0.0001** | **NaN** | **0.0005** | **NaN** |
|  40194438 | 21.074 | 1.0000 | +0.148 | 1.0000 | +0.148 |
| 300075262 | 20.671 | 1.0000 | -0.007 | 1.0000 | -0.040 |
| 170022839 | 20.951 | **0.0000** | **NaN** | **0.0000** | **NaN** |
| 270119597 | 20.681 | 1.0000 | -0.198 | 1.0000 | -0.198 |
| 390002315 | 20.759 | 1.0000 | +0.020 | 1.0000 | +0.020 |
| 170146976 | 20.701 | **0.0000** | **NaN** | **0.0000** | **NaN** |
| 180163996 | 20.622 | 1.0000 | -0.039 | 1.0000 | -0.020 |

**The two models are indistinguishable on this random sample**: both detect 7/10 with high confidence and miss the SAME 3/10. The canonical-TID divergence (0.042 vs 0.755) is an outlier, not a generic failure mode. The 3 jointly-missed targets all have z_DLA ≈ 1.9-2.0 (just above prior `min_z_dla`), suggesting a proximity-region issue unrelated to c0prior.

## NaN trace — k ≥ 2 DLA posteriors

Production run on TID 120046865 with c0prior reproduces deterministically (`reproduce_full.log`):

```
log p(D | z_QSO, no DLA)     = -3029.930
WARNING (dla_gp.py:620)      : "No valid regions found in the initial scan"      ← SubDLA
INFO (dla_gp.py:633)         : "Stopping early at 1 DLAs because log_lik -3039.377 < null -3027.477"  ← SubDLA
log p(D | z_QSO, 1 subDLAs)  = -3039.377
INFO (dla_gp.py:804)         : "Stopping early at 1 DLAs because log_lik -3030.538 < null -3027.477"  ← DLA
log p(D | z_QSO, 1 DLAs)     = -3030.538
log p(D | z_QSO, 2..4 DLAs)  = NaN
```

The NaN for k ≥ 2 is the production code's **deliberate early-stop** at `gpy_dla_detection/dla_gp.py:790-810` (the `parallel_log_model_evidences` branch returns NaN for k+1..max_dlas when the k-DLA truncated marginal falls below null). **It is not a Cholesky failure.** Both models exhibit this for k ≥ 3 on the same target (see `stepc_2lpt_loa124_nohcd_nobal_wide_m.json` lines 17-19). The c0prior-vs-`_m` divergence is only at k=1.

I verified directly (`probe_canonical.py`, "c0prior" block) that the c0prior model evaluates the 1-DLA log-likelihood at truth (z=2.7748, log_NHI=21.26) to **logL = -3011.84**, while null = -3029.93. **Δ = +18.09 nats**. The kernel is fine; the failure is in the QMC marginalisation.

## Kernel condition / singular values

`kernel_conditioning.log`:

- c0prior: cond(K) = 9.63e4, M top SVs = [144.4, 12.3, 9.2, 7.4, 4.7], `‖M‖_F²` = 21,317
- `_m`: cond(K) = 1.31e4, M top SVs = [32.0, 15.5, 11.6, 5.7, 5.4], `‖M‖_F²` = 1,648

Both eigenvalue ranges are within `dpotrf`'s working range (smallest eigenvalues > 0.07; 1e-15 ulp ≪ 1e-5 ratio). cond(K) is dominated by a single very-large top eigenvalue, not approach-to-zero. **Neither model's K is singular.** The k ≥ 2 NaN is algorithmic, not numerical.

What the larger K spectrum actually does on the canonical TID:

- 1-DLA point-at-truth: -3011.84 (vs null -3029.93) → **Δ = +18.09 nats**
- 1-DLA QMC truncated marginal: -3030.54 → **0.6 nats BELOW null** (-3029.93), or **3.1 nats below prior-weighted null** (-3027.48)
- For `_m`: point-at-truth -2857.29 (Δ = +22.22) and truncated marginal ≈ -2856.8 (well above null)

The c0prior model's QMC integral suffers because the wider kernel absorbs more bulk-NHI samples into "plausible" mass, dragging the marginal down. Not a representation problem — a variance problem.

## c_0-override experiment

`probe_canonical.py` mutates `null_gp.log_c_0` / `dla_gp.log_c_0` in-memory before `set_data`:

| variant | null | L@truth | Δ vs null | best surf logL |
|---|---:|---:|---:|---:|
| c0prior native | -3029.93 | -3011.84 | **+18.09** | -3012.29 |
| _m baseline | -2879.51 | -2857.29 | **+22.22** | -2856.81 |
| c0prior with c_0 = _m's value (0.02357) | -3030.60 | -3012.40 | **+18.20** | -3012.80 |
| c0prior with c_0 = 0.1 (anchor) | -3054.20 | -3032.19 | **+22.01** | -3031.00 |

1. Setting c_0 to `_m`'s value **does not move the needle** (Δ improves by 0.11 nats — noise). The endpoint c_0 values are too close (0.020 vs 0.024) to matter.
2. Setting c_0 to the prior anchor 0.1 **recovers `_m`'s Δ** (18.09 → 22.01). Because c_0 multiplies the entire reconstructed continuum, the 5× boost scales up DLA absorption SNR.
3. The c_0 = 0.1 override shifts both null and L@truth together by ~24 nats (continuum now too bright in absolute terms); detectability depends on the **difference** Δ = L@truth − null, which recovers.

Conclusion: c_0 is not the direct cause of the canonical-TID failure — the c0prior model with the right c_0 still fails by ~4 nats. The 22-nat recovery from c_0=0.1 override is artificial — it compensates for the 13× inflated M by rescaling the whole kernel back. The "true" mechanism is the inflated M (combined with the narrower norm band), not the c_0 endpoint itself.

## Mechanism — which hypothesis wins

Brief's three hypotheses:

1. **"Kernel doesn't fit the DLA absorption shape well, evidence ~20× smaller"** — partially true but not as framed. The c0prior kernel fits the DLA fine at truth (+18 nats); its truncated QMC integral averages down because the 13× larger ‖M‖² lets too many low-NHI samples look plausible.
2. **"Multi-DLA hypotheses produce NaN due to Cholesky failures"** — **wrong**. NaN is the deliberate early-stopping shortcut at `dla_gp.py:790-810`. Cholesky of K = MMᵀ + diag(ω²) succeeds for both models. Both produce k=3, k=4 NaN even when they pass p_DLA > 0.5.
3. **"Anchoring log_c_0 at log(0.1) leaves the kernel 5× inflated"** — the **anchoring failed**. Both models ended at c_0 ≈ 0.02. But c0prior's slowed `log_c_0` drift let `‖M‖_F` balloon 13× in compensation. The spirit of the hypothesis (c_0/M factor-analysis degeneracy) is the mechanism, with the resolution direction inverted: M ballooned instead of c_0 stalling.

Concrete two-factor explanation for canonical TID 120046865:

- **Norm band [1310, 1325] vs [1425, 1475]**. The narrower 15-Å band closer to Lyα gives noisier per-spectrum median estimates and a different μ shape (μ_mean 1.204 vs 1.415). Contributes most of the ~3 nat null-evidence gap. (Visible in the kernel-conditioning scalars and in the README explicitly recording "Garnett+2017 convention" for c0prior vs "MATLAB DR16 convention" for `_m`.)
- **Inflated M.** ‖M‖² 13× larger → wider QMC prior envelope at any (z_DLA, NHI) → off-truth samples less penalised → marginal drags down ~1 nat.

Together these put c0prior's 1-DLA truncated marginal 3 nats BELOW prior-weighted null while `_m`'s is 22 nats ABOVE. On 7/10 other strong DLAs the gap is large enough either way; on 3/10 both fail. Canonical TID 120046865 happens to sit in the gap between c0prior and `_m`.

## Recommendation

1. **Stop using the c0prior recipe.** The prior failed at anchoring c_0 to 0.1 (drift dominates) and made things worse via the M-inflation it allowed. A weak Gaussian prior on a logarithmic parameter is not the right tool for the factor-analysis degeneracy.
2. **Do not change the norm band.** [1310, 1325] (Garnett+2017 convention) is correlated with the prior change in this experiment and contributed independently to the regression. Keep [1425, 1475] (MATLAB DR16 convention; used by `_m` and v1 production).
3. **If you want to fix the (c_0, M) degeneracy properly:**
   - **Reparameterise**: train on `(M̃, c_0̃)` with `M̃ = c_0·M` and fix one (e.g. `c_0̃ = 1` or `‖M̃‖_F = 1`). The likelihood is gauge-invariant; fixing a gauge kills the degeneracy without biasing anything.
   - **Tighter Gaussian prior**: σ = 0.5 was clearly too weak (gradient strength = σ⁻² = 4, far below the τ_0/β prior strengths). Try σ = 0.1 (strength 100) or σ = 0.2 (strength 25). Be aware aggressive σ inflates ‖M‖ — monitor with `kernel_conditioning.py`.
   - **Direct L2 on M**: add `λ · ‖M‖_F²` to the loss. Kills the degeneracy without touching c_0.
4. **Improve marginal-evidence robustness** for borderline targets:
   - Increase FILTER's `n_initial` from 5000 to 10000+ to denoise the threshold (`gpy_dla_detection/dla_gp.py:559`).
   - When `k=1 truncated marginal < null` AND the **max single-sample** `logL` was > null, fall back to the unrestricted nested-sampling marginal instead of returning NaN for k ≥ 2 (current behaviour at `dla_gp.py:619-641` is "give up").
5. **Document the README "(none)" misleadingness**. The c0prior README claims no prior, but the SLURM submit script presumably set `--log-c-0-prior-sigma`. Either recover the value from the cluster log archive (SLURM 50021381) or re-train and let `tests/phase2_train_desi.py:548-549` write the sigma into the `.h5` (the dataset is being written for new runs; for this model it just wasn't yet, or was NaN).

## Analysis artifacts

All in `/home/mfho/desi_gpy_dla_detection/docs/notes/2026-05-14_c0prior_failure_investigation/`:

| File | Contents |
|---|---|
| `sample_strong_dlas.py` + `sampled_dlas.json` | Truth-catalog DLA selection (n=10, seed 13) |
| `multi_target_inference.py` + `multi_target.log` + `multi_target_results.json` | 10-target inference (both models) |
| `kernel_conditioning.py` + `kernel_conditioning.log` | M, ω², K spectrum diagnostics |
| `probe_canonical.py` + `probe_canonical.log` + `probe_canonical.json` | Single-target likelihood at truth + c_0 override |
| `nan_trace_canonical.py` + `nan_trace.log` | First NaN-trace attempt (errored mid-run on a holder attribute lookup; superseded by `reproduce_full.log` + `reproduce_filter.log`) |
| `initial_scan_diagnosis.py` + `initial_scan_diagnosis.log` | First 5000 raw-QMC-sample distribution (both models) |
| `reproduce_filter_path.py` + `reproduce_filter.log` | Exact reproduction of `parallel_log_model_evidences`' initial scan with shifted-logL bookkeeping; confirms valid_mask.sum() > 0 for c0prior at proper threshold (the "No valid regions" warning in the production log was from SubDLA, not DLA — the DLA path stopped early because the truncated marginal lost to null) |
| `reproduce_full_inference.py` + `reproduce_full.log` | Full holder.process_qso reproduction confirming p_DLA = 0.04160 is deterministic |
