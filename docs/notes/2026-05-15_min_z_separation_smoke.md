# 2026-05-15 — MIN_Z_SEPARATION smoke test

> **Status**: DONE (job 53018910; results refreshed 2026-05-17 under the
> new DLAFLAG convention). **Verdict: NO-CHANGE (do not retune).** The
> M0–M3 spread is ≤0.7pp P / 0.6pp C — entirely within the ~1pp
> run-to-run noise floor. Keep the production default 3000 km/s.
>
> **Sweep root**: `/pscratch/sd/j/jibancat/prod533_5k_20260511/min_z_separation_sweep/`
>
> Evaluated with the fixed molly recipe (n_truth = 581); numbers
> refreshed 2026-05-17 (NHI_INCONSISTENT no longer gated).

## Hypothesis

The `MIN_Z_SEPARATION` knob (production = **3000 km/s**) imposes a minimum
velocity separation between QMC-sampled DLA centers within a single QSO during
the inner Bayesian integration over k-DLA models. Tighter values *could* let
the model resolve overlapping DLAs that are currently merged into a single
detection (helping completeness on close pairs); looser values *could*
spuriously inflate higher-k evidences by allowing two samples to land on the
same physical absorber (hurting purity by spurious 2-DLA / 3-DLA detections).
Either way it shifts MAP P/C.

The proximity collar applied at eval time
(`molly_faithful_pc_plots.py`, 3000 km/s) is a *different* thing — it lives
post-catalog. So MIN_Z_SEPARATION is genuinely an **inner-inference** knob,
not eval-time dedup.

## Mechanism inspection (code reading, before launching)

`MIN_Z_SEPARATION` (km/s) is converted via `Parameters.kms_to_z()` into a Δz
threshold on construction of `DLAGPMAT` (`gpy_dla_detection/dla_gp.py:329`):

```python
self.min_z_separation = self.params.kms_to_z(min_z_separation)
```

It is then enforced inside the joint k-DLA evidence integration loop. After
QMC sampling positions for the k-th DLA conditioned on the resampled
positions for DLA 1..k-1, the code masks out any joint sample where any pair
of DLA centers is closer than the threshold (`dla_gp.py:434-447` for the
serial path, `dla_gp.py:777-787` for the parallel path):

```python
ind = np.any(
    np.diff(np.sort(all_z_dlas, axis=0), axis=0) < self.min_z_separation,
    axis=0,
)
sample_log_likelihoods[ind, num_dlas] = np.nan
```

That NaN sample is then dropped from the MC integral (`np.nanmean`). The
1-DLA evidence (`num_dlas == 0`) is unaffected — the constraint kicks in at
k ≥ 2. So this knob acts on **multi-DLA inner evidences only**.

The same `min_z_separation` (in Δz units) is also passed into
`process_batch` so the sub-process branch enforces the same rule on its
`sample_log_likelihood_k_dlas` call (line 760). No other module reads it.

There is **no** post-Bayesian dedup that uses this knob. The catalog
construction in `process_helpers.py` writes one row per (DLA-index)
combination based on the model with the highest `model_posteriors[k]`; the
3000 km/s collar that filters the predicted catalog at eval time is hardcoded
inside `molly_faithful_pc_plots.py`.

The QMC sampler is shared across cells (same `pw_samples_a3_172_220_50000.mat`
file), so sampling noise is matched between cells — the only thing changing
is which samples get NaN'd. M3 (= 0 km/s) is the limit case where no samples
are NaN'd at all.

## Configs

Cloned from `cellC_knob_sweep/configs/C0.env` (post-patch 2-way cellC
baseline: PW 50k, MAX_DLAS=3, FILTER=1, τ-EB=on, NHI [17.2, 22], SubDLA
samples 100k). Single knob varied.

| Cell | MIN_Z_SEPARATION (km/s) | Δz at z=2.5 | rationale |
|------|------------------------:|------------:|-----------|
| M0   | 3000  | 0.0350 | baseline = clone of C0 (sanity check) |
| M1   | 2000  | 0.0233 | moderately looser; resolves pairs ~3.7 Mpc/h apart |
| M2   | 1000  | 0.0117 | aggressively looser; ~1.9 Mpc/h |
| M3   | 0     | 0      | limit case; no separation enforcement |

(Δz computed as `kms × 1000 / c`. Comoving separation at z=2.5 from
`c × Δz / H(z)` is order-of-magnitude only — for intuition, not science.)

## Operating point (eval recipe, identical to cellC sweep)

`molly_faithful_pc_plots.py` with: SNR_RED > 2, p_DLA ≥ 0.99, lyb-veto on,
`--no-bal`, λ_rf ∈ [911, 1216], NHI ≥ 20.3 truth+predicted,
`--restrict-truth-to-processed`, external `--snr-cat` + `--zcat`. Mock =
London v5.9.5 mock-0, 5k slice (8 × spectra-16 files starting at index 0).

## Reference baseline

C0 (post-patch, same eval recipe) at the same operating point reads from
`cellC_knob_sweep/HEADLINE.tsv`. M0 should reproduce this within the
sampling noise from the τ-EB seed search and Monte Carlo integration.

## Cost prediction

Wall-time per cell ≈ 90 min, identical to cellC C0 cadence — the knob is
free in inference cost (it's a numpy mask). 4 cells in parallel × 64 cores
each = 256 cores = full jupyter node. Total ≈ 90 min wall + ~5 min eval.

## P/C table — DONE (refreshed 2026-05-17, new DLAFLAG convention)

Source: `min_z_separation_sweep/HEADLINE.tsv`. n_truth = 581. Numbers
refreshed 2026-05-17 (NHI_INCONSISTENT no longer gated).

| Cell | knob (km/s) | P | C | ΔP vs M0 | ΔC vs M0 | n_cat | wall_min | node_h |
|------|------------:|---:|---:|---:|---:|---:|---:|---:|
| M0   | 3000 (baseline) | 0.7775 | 0.8328 | ref | ref | 4663 | 44.7 | 0.186 |
| M1   | 2000 | 0.7843 | 0.8328 | +0.7 | 0.0 | 4682 | 44.7 | 0.186 |
| M2   | 1000 | 0.7797 | 0.8328 | +0.2 | 0.0 | 4678 | 44.6 | 0.186 |
| M3   | 0    | 0.7784 | 0.8266 | +0.1 | −0.6 | 4685 | 44.6 | 0.186 |

## Verdict — NO-CHANGE

Loosening MIN_Z_SEPARATION has **no resolvable effect**. On the refreshed
numbers the M0–M3 spread is 0.7pp on purity (0.777–0.784) and 0.6pp on
completeness (0.827–0.833) — entirely within the ~1pp run-to-run noise
floor (`determinism_sweep`). The earlier draft's "M1 looks +2.3pp better"
hint was a DLAFLAG-gating artifact; post-refresh M1 is +0.7pp P / 0.0pp C
vs M0 — noise. n_cat barely moves (4663 → 4685, +0.5%), confirming the
knob only touches k≥2 inner evidences and almost never flips a headline
detection at this operating point.

So the knob is **inert at production scale**: the multi-DLA NaN-masking it
controls rarely changes the MAP k-model after the p_DLA≥0.99 cut.
**Keep MIN_Z_SEPARATION = 3000 km/s.** (The `min_z_separation_sweep_50k`
re-run at 10× statistics will give the definitive word, but the 5k
refreshed result already shows nothing to chase.)

The knob is also free in compute (identical 44.6–44.7 min wall across all
cells — it is a numpy mask), so there is no cost argument either way.

## Files

- Configs: `min_z_separation_sweep/configs/M{0,1,2,3}.env`
- Launcher: `min_z_separation_sweep/_launch.sh` (1 cell)
- Chain runner: `min_z_separation_sweep/_chain.sh` (waits for lambda L3 to
  finish, then launches 4 cells in parallel, emits `ALL_DONE` then auto-runs eval)
- Eval+aggregate: `min_z_separation_sweep/_eval_and_aggregate.sh`
- Headline: `min_z_separation_sweep/HEADLINE.tsv` (created by eval)
