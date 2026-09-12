# `validation/long_chain/` — VALIDATION-ONLY long-chain posterior wrapper

**Classification: VALIDATION-ONLY** (PI ruling 2026-09-12 §8: "Every new component classified
science-producing / validation-only / provenance-only / release-packaging-only. A validation tool
must never quietly become a second science authority.")

Nothing in this directory is production code, nothing here is imported by any pipeline, and nothing
here defines a science quantity of its own. The sampler, the model, the likelihood, the priors, the
FP mode and the reduction are all taken unchanged from the frozen machinery:

| object | authority |
|---|---|
| sampler / model / priors | `CDDF_analysis/hbi_mcmc/cc_posterior_validation.py`, `cc_real_posterior.py`, `pack.py`, `model_a.py` — byte-identical to `1fd4828` on this branch |
| reduction | `paper_figures/hbi_reduction.py` in the manuscript repo, **read-only import** (identical quantity set to `reductions/tools/reduce.py`, R-042a) |
| convergence estimators | ArviZ 0.23.4 (the reference implementation of Vehtari et al. 2021), with an independent hand-written cross-check in `convdiag.py` |

## Why this wrapper exists

PI ruling 2026-09-12 §3 requires a prospectively declared long-chain final-posterior campaign whose
runs retain "nuisance draws and per-draw potential energy". `cc_real_posterior.py` discards both.
This wrapper adds only that retention, plus the chain/warmup/draw counts of the sealed schedule.

PI ruling 2026-09-12 §3 also requires that "any wrapper used only to retain additional diagnostics
must first reproduce a production-config run bit-for-bit". `verify_bit_identity.py` is that gate and
it is run before any campaign sampling.

## Files

| file | role |
|---|---|
| `run_longchain_posterior.py` | the wrapper: `cc_real_posterior`'s sampling core + guards + gate + committed reduction, with by-chain draws of every sampled site, per-draw potential energy and per-draw divergence flag retained |
| `verify_bit_identity.py` | the mandatory `np.array_equal` gate against a stored production run |
| `convdiag.py` | convergence estimators (copied unchanged from the Phase-3 study) |
| `quantities.py` | per-draw vectors for the 30 reported scalars and the 574 sampled nuisance components |
| `analyze_runs.py` | the sealed criteria + mode classification, applied per run |
| `pool_and_reduce.py` | pooling rule, the paper's reduction, pooled criteria, comparison against the current C1 pool |

The campaign's sealed predeclaration, stage schedule, criteria literals and stop conditions live in
`/home/mfho/lowz_clean_work_2026-09-12/posterior_campaign/PREDECLARATION.md`
(sha256 `73ab274184e00b1f865db5351e8ecdc77e4982f858fa2180b906e53eccc526fc`).
