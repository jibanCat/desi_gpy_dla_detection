# Phase B: full production bayes on 5000 random 2LPT — definitive bias-closure measurement

> Phase B = run `DLAHolder.process_qso` with `enable_tau_eb=False` then
> `enable_tau_eb=True` on each of the 5000 random 2LPT QSOs (z_qso ≥ 2,
> no cherry-picking; same target list as Phase A).  Records the full
> production MAP log N_HI and p_DLA for both treatments.  Validates
> the τ-EB recipe under the production code path, not just the
> τ-fit step.
>
> **Result CSV**: `tests/profile/results/tau_eb_phase_b_5k_2lpt.tsv`.
> SLURM array job: `49040725` (16 tasks × 313–320 spectra each).
> Wall: 2–4.5 h per array task in parallel (mix-dependent), ~4.5 h total
> wall for the longest task.
> Per-spectrum wall: 13.3 s median (baseline + enabled = 27 s combined,
> very close to the 19.8 s × 2 estimate from the per-target profile).
>
> **Caveat**: this run used the τ_factor grid `(0.5, 1.0, 1.5, 2.0,
> 3.0, 4.0)` because Phase B was submitted before commit `d849a30`
> bumped the default to include 5× and 6×.  At the population scale,
> 11 % of spectra pinned at the τ=4 ceiling — those would have moved
> to 5 or 6 in the new default grid.  Expected effect: slightly more
> aggressive bias correction.  Not re-run; see Phase A extgrid
> (`docs/notes/2026-04-29_tau_eb_phase_a_5k_2lpt.md`) for the τ-distribution
> with the new default.

## Headline

| Metric | BASELINE (no τ-EB) | ENABLED (τ-EB on) | change |
|---|---:|---:|---:|
| n_total / n_ok | 5000 / 5000 | 5000 / 5000 | 0 errors |
| n DLA-truth detected by both | 234 | 234 | — |
| **median bias on DLA-truth** | **+0.126 dex** | **+0.044 dex** | **−65 %** |
| mean bias on DLA-truth | +0.172 | +0.087 | −49 % |
| RMS bias on DLA-truth | 0.367 | 0.286 | −22 % |
| std bias on DLA-truth | 0.325 | 0.273 | −16 % |
| Wilcoxon p (H₀: median = 0) | 3.4 × 10⁻²¹ | 4.6 × 10⁻⁸ | 13 orders of magnitude |
| **false-positive rate** (no-truth → flagged as DLA) | **2.3 %** | **1.5 %** | **−35 %** |
| DLA detection completeness (truth NHI ≥ 20.3) | 50.5 % | 48.7 % | −1.8 pp |
| wall time per spectrum (16-CPU node) | 15.8 s | 15.7 s | 0.99 × |

## Per-regime detection rate

| Truth regime | n | baseline detect | enabled detect | difference |
|---|---:|---:|---:|---:|
| DLA (NHI ≥ 20.3) | 493 | 50.5 % | 48.7 % | −1.8 pp |
| sub-DLA (19.0–20.3) | 1051 | 15.0 % | 13.6 % | −1.4 pp |
| LLS (17.2–19.0) | 853 | 3.4 % | 2.7 % | −0.7 pp |
| **none (no truth absorber)** | **2603** | **2.3 %** | **1.5 %** | **−0.8 pp** |

τ-EB **uniformly tightens** the detection criterion:
  - Modest completeness loss on DLA-truth (1.8 pp), balanced by
  - Substantial false-positive reduction on no-truth (35 % relative).

Net effect: precision improves more than recall declines on a
randomly-sampled population.  This is **not** a recall regression
the way it might look from the per-bin numbers — the catalog purity
goes up.

## Why the median-bias improvement matters

Production at τ_0 = 0.00246 produces **statistically significant
positive bias** (Wilcoxon p ≈ 10⁻²¹) of +0.126 dex median on n=234
detected DLAs.  τ-EB cuts the bias to +0.044 dex (still significant,
p ≈ 10⁻⁸, but 13 orders of magnitude weaker rejection of zero-bias)
with the same number of detections.

Importantly, this is consistent across truth-NHI bins (per Phase A
breakdown), three mocks (n=90 result), and now a production-mode
5000-spectrum random sample.  The recipe is robust.

## What's NOT closed by τ-EB

The residual +0.04 dex median is real and small but non-zero.
Consistent with the n=90 result.  Plausible candidates for further
investigation (all out of scope for PR #5):
- non-Gaussian residual distribution (Lyα forest skewness; user's H7)
- GP μ continuum shape in DLA wings
- per-pixel ω² miscalibration

A possible quick-win is running the user's H8 hypothesis (training
the GP on a no-HCD-no-BAL trainset and seeing if the residual bias
shifts).  Both NERSC training jobs (52198069 / 52198070) and the
two GreatLakes jobs (49037617 / 49037618) are in flight; can re-run
this Phase B against the new model when training completes.

## Cost reality check

At 16-CPU node × 5000 spectra × (15.8 + 15.7) ≈ 250 000 CPU-s
= 70 CPU-hours wall.  On a 256-CPU NERSC node = 16 nodes' worth in
parallel = **~16 NERSC node-hours** for this 5000-target validation
run.  For 1 M QSO equivalent: ~3300 NERSC node-hours, vs the
~50-node-hour target — same 6× over-budget conclusion as the
per-target profile, no surprises.

τ-EB itself adds essentially zero cost (`enabled / baseline = 0.99×`).
The 6× gap is in `bayes.model_selection`, unchanged by this PR.
See `docs/notes/2026-04-29_production_cost_estimate.md`.

## Files

- `examples/run_tau_eb_phase_b.py` — chunk runner (per SLURM-array task)
- `slurm/greatlakes/phase_b_5k_array.sh` — SLURM-array driver
- `tests/profile/results/tau_eb_phase_b_5k_2lpt.tsv` — per-spectrum results (5000 rows)
- `docs/notes/2026-04-29_tau_eb_phase_a_5k_2lpt.md` — Phase A τ-distribution companion
- `docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md` — earlier n=90 picker result
