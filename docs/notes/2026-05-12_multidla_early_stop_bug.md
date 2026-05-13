# Multi-DLA early-stop bug — TID 47767 (London 8f)

> **Status**: real bug, found 2026-05-12 by hypothesis #1 sweep agent. Not yet fixed.
> Worth its own PR — small, localized fix in `gpy_dla_detection/dla_gp.py`.

## The finding

On TID 47767 (London mock-0):

| k-DLA model | log_likelihood | Δ vs null |
|---|---:|---:|
| Null | (baseline) | 0 |
| 1-DLA | ≪ null evidence | < 0 |
| **2-DLA** | **null + ~11** | **+11** |

The 2-DLA evidence is **+11 logL above 1-DLA** and above null. The catalog should report a 2-DLA detection. Instead it reports a non-detection, because the inference loop never evaluated the 2-DLA model — it hit an early-stop after computing the 1-DLA likelihood.

(Numbers from hypothesis1_sweep agent's manual evaluation. Reproducible by overriding the early-stop condition and rerunning.)

## Where the bug lives

`gpy_dla_detection/dla_gp.py`, the multi-DLA inference loop has three early-stop conditions inside the `for num_dlas in range(max_dlas)` body. The relevant block (lines ~795-820):

```python
# ========= Early stopping logic =========
if (num_dlas + 1) == max_dlas or np.isnan(log_likelihoods_dla[num_dlas]):
    break

# If null_evidence is provided and the current log likelihood is less than it,
# stop further computation
if null_evidence is not None:
    if log_likelihoods_dla[num_dlas] < null_evidence:
        log.info(
            f"Stopping early at {num_dlas + 1} DLAs because the log likelihood "
            f"{log_likelihoods_dla[num_dlas]} is less than the null model evidence "
            f"{null_evidence}."
        )
        break

# If log likelihood is smaller than the previous one by 10 times, stop further computation
if num_dlas > 0:
    if log_likelihoods_dla[num_dlas] < log_likelihoods_dla[num_dlas - 1]:
        log.info(
            f"Stopping early at {num_dlas + 1} DLAs because the log likelihood "
            f"{log_likelihoods_dla[num_dlas]} is less than the previous one."
        )
        break
```

The second condition is the one that fires on TID 47767: `log_likelihoods_dla[0] < null_evidence` is true (1-DLA fits one of the two real absorbers poorly), so the loop breaks **before** ever computing the 2-DLA likelihood — even though 2-DLA would have been the right answer.

## Why this is the wrong heuristic

The greedy assumption embedded in the early-stop is:
> If k-DLA doesn't beat null, k+1-DLA can't either.

That's false for spectra with **multiple real absorbers**: a single DLA fit attempts to "explain" both, and ends up doing badly on each. Adding a second DLA can suddenly fit them properly, jumping the likelihood by +11 logL or more.

In Bayesian terms: the likelihood as a function of k-DLA is **not monotonic** for spectra with multiple absorbers. Greedy early-stop assumes monotonicity and is wrong.

Bookkeeping note: there's also an Occam-like penalty `- lognorm * num_dlas` (line 444) and a QMC normalization `- np.log(num_dla_samples)` (line 412) baked into `log_likelihoods_dla`. These are model-complexity penalties, not the same thing as the early-stop. The 2-DLA evidence in this case is +11 logL above 1-DLA **after** those penalties — so even Occam-corrected, 2-DLA wins.

## Suspected scope

Unknown how many spectra are affected — TID 47767 was found by spot-checking the hypothesis #1 sweep's missed-candidate population. A targeted scan would:

1. For each spectrum where the catalog reports < MAX_DLAS detections AND `log_likelihoods_dla` shows an early-stop, recompute k+1 onwards.
2. Flag any case where a later k has higher likelihood.

Plausibly a non-trivial fraction in the SNR>2 multi-absorber tail (these are exactly the low-SNR weak-DLA spectra that drive the completeness ceiling).

## Suggested fix

Replace the second condition (the `null_evidence` early-stop) with one of:

- **Option A — disable entirely**: rely only on `MAX_DLAS` and the "likelihood decreased from previous k" check (the third condition is still defensible: if k+1 > k didn't help, k+2 > k+1 probably won't either, though even this assumption is shaky).
- **Option B — delay one step**: only allow null_evidence early-stop at `num_dlas ≥ 1` (after at least one DLA has been evaluated AND below null), so two-absorber spectra get a chance.
- **Option C — peek ahead**: compute the next k anyway and only break if both k and k+1 are below null. More expensive but principled.

Option A is the cleanest and reduces inference cost only marginally — `MAX_DLAS=3` means at worst 3 evaluations instead of 1, on spectra that would otherwise get 0 DLAs reported. The number of affected spectra is small (most catalog rows already use the full `MAX_DLAS`), so the cost overhead is minor.

## Relation to the SNR>2 completeness ceiling

This bug is **independent** of the prior-volume dilution mechanism documented in `project_prior_dilution_finding` (memory) and `docs/notes/2026-05-12_mlmc_design.md`. The dilution problem is "the marginal under-counts narrow peaks"; this bug is "the inference loop never gets to the model that would fit". Both contribute to the SNR>2 ceiling; both need separate fixes.

Plausibly: fixing this bug recovers some multi-absorber detections (estimate ≪ 5pp of the C gap on its own). MLMC / adaptive importance sampling addresses the dilution. The two together get closer to 85/85.

## How to verify the fix

1. Comment out the `null_evidence` early-stop (Option A) or apply Option B/C.
2. Re-run inference on London 8f (or just the slice containing TID 47767).
3. Check that TID 47767 now reports a 2-DLA detection with log_likelihoods_dla[1] - log_likelihoods_dla[0] ≈ +11.
4. Run molly-faithful P/C on the full output and confirm completeness goes up without purity drop.

Expected wall-time cost increase: < 5% (most spectra already iterate to `MAX_DLAS` because they have at least one DLA in scope).

## Artifacts

- `/pscratch/sd/j/jibancat/prod533_5k_20260511/hypothesis1_sweep/RESULTS.md` — the original sweep that found the bug
- `gpy_dla_detection/_pme_patched.py` (in scratch, not in repo) — the agent's patched copy of `parallel_log_model_evidences` that bypasses the early-stop. Useful starting point if implementing a fix.

## Related

- `docs/notes/2026-05-12_mlmc_design.md` — the broader path to 85/85; this bug is one orthogonal contributor
- `[[project-prior-dilution-finding]]` (memory) — the *other* SNR>2 contributor
- `[[project-subdla-dla-joint-design]]` (memory) — the SubDLA-as-no-DLA aggregation; a third orthogonal contributor
