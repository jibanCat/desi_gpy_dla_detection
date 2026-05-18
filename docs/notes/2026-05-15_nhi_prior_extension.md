# Extended NHI prior for DLA detection — design + smoke test

**Date**: 2026-05-15
**Branch**: `production_533`
**Sweep dir**: `/pscratch/sd/j/jibancat/prod533_5k_20260511/nhi_prior_ext_sweep/`

> **Status**: DONE (job 53013668; results refreshed 2026-05-17 under the
> new DLAFLAG convention). **Verdict: marginal P/C cost — provisional.**
> Refreshed: E1 (extended prior [17.2,22.5]) is −1.3pp purity / −0.9pp
> completeness vs E0 on the 5k slice (near the noise floor; the
> pre-refresh "byte-identical" result was a DLAFLAG-gating artifact). The
> extension still fixes a real modelling defect (NHI clipping above 22.0),
> but it is no longer the "free" change the gated eval implied —
> **re-check at larger scale before committing the DLA_SAMPLES_FILE
> swap.** See §6.

## 1. Motivation

The production C7 cell uses the DLA column-density prior QMC grid
`data/dr12q/processed/pw_samples_a3_172_220_100000.mat`, which covers
`log10(N_HI) ∈ [17.2, 22.0]`.

London mock-0 truth (`dla_cat.fits`) contains DLAs up to `logNHI ≈ 22.47`, and
the real DESI universe has rare DLAs above 22.0. A prior with a hard ceiling at
22.0 **cannot represent** those absorbers: their evidence integral has no
support above 22.0, so the model is forced to under-quantify the highest-NHI
DLAs (NHI pinned to the 22.0 edge, biased low). The user wants the production
prior extended to **[17.2, 22.5]**.

## 2. The PW-vs-uniform mixture is already the production machinery

The user's specified approach — "mix the PW prior over [17.2, 22.0] with an
alpha-weighted uniform tail over [22.0, 22.5]" — is **exactly** what
`gpy_dla_detection/generate_samples.py::build_pw14_prior` already implements.
That function builds

    p(logN) = alpha * p_PW14(logN) + (1 - alpha) * Uniform(min_log_nhi, max_log_nhi)

where the `Uniform` component spans the **entire** requested range. The PW14
component is the Prochaska+2014 CDDF (`f(N) · N · ln10`, normalised over the
range). Crucially, `f_pw14` clips its input to the spline node range
`[12.0, 22.0]`:

```python
_LOGNHI_NODES = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
log_nhi_clip = np.clip(log_nhi, _LOGNHI_NODES[0], _LOGNHI_NODES[-1])
```

So above `logNHI = 22.0` the PW14 component is held **flat** at its 22.0 value.
It does **not** extrapolate to zero (which would zero out the tail) and does
**not** blow up. Consequence: the mixture is **continuous at 22.0** by
construction — the PW14 part is C0-continuous across the node, and the uniform
part is constant — so there is **no discontinuity** at the 22.0 boundary. The
`(1 - alpha)` uniform floor then guarantees finite, non-zero sampling density
across the whole `[22.0, 22.5]` tail where the PW fit has no data.

This is **not** a case requiring a bespoke f(N) extrapolation. Generating the
extended file is a one-liner: call `generate_pw14_samples` with
`max_log_nhi = 22.5`.

## 3. Alpha choice

**alpha = 0.97** — kept unchanged.

Rationale:
- 0.97 is the default in `generate_samples.py` and the value used to build the
  legacy `dla_samples_a03.mat` and **every** production `pw_samples_a3_*` file
  (the docstring states "alpha=0.97 by default (97% data-driven, 3%
  uniform)"). Keeping it constant means E1 differs from the C7 baseline E0
  **only** in the prior's upper bound — a clean, single-variable ablation.
- The `(1 - alpha) = 3%` uniform component, spread over the
  `22.5 - 17.2 = 5.3` dex range, contributes a flat density of
  `0.03 / 5.3 ≈ 0.0057` per dex. Over the `[22.0, 22.5]` tail this is the
  *dominant* contribution (the PW14 part there is the tiny, flat clipped
  value), so the tail is sampled at roughly the uniform-floor density —
  small but strictly non-zero, which is the goal: rare high-NHI DLAs get
  finite evidence support without the prior pretending the PW fit extends
  there.
- Picking a *larger* alpha-for-the-tail (a separately-weighted tail uniform)
  was considered and rejected: it would (a) break the single-variable ablation
  vs C7, and (b) require re-deriving normalisation. The existing global-alpha
  mixture already delivers a populated, continuous tail; no extra knob is
  warranted for a smoke test.

`_gen_extended_prior.py` prints a continuity check (PDF at 21.99 vs 22.01) and
the tail sampling fraction so this can be verified empirically.

## 4. New sample file

```
data/dr12q/processed/pw_samples_a3_172_225_100000.mat
  log10(N_HI) ∈ [17.2, 22.5]
  N = 100000   alpha = 0.97   seed = 42
  schema: log_nhi_samples, nhi_samples, offset_samples (N,1);
          alpha, fit_min_log_nhi, fit_max_log_nhi,
          uniform_min_log_nhi, uniform_max_log_nhi (1,1)
```

Built by `nhi_prior_ext_sweep/_gen_extended_prior.py`. Histogram (baseline vs
extended) saved to `nhi_prior_ext_sweep/nhi_prior_histogram.png` by
`_plot_nhi_hist.py`.

**NHI histogram description** *(to fill after generation)*: the bulk of the
mass sits at low NHI (LLS/sub-DLA regime, steep PW14 CDDF); the DLA regime
`[20.3, 22.0]` carries the production-relevant samples; the new `[22.0, 22.5]`
tail is populated at the ~uniform-floor density (expected ≈ a few × 0.1% of
total) — sparse but non-empty, with no gap or step at 22.0.

## 5. Smoke test (E0 vs E1)

Two cells on the same 8-healpix London mock-0 slice (`OUTER_WINDOW=8`),
`cellC_knob_sweep` recipe:

| cell | NHI prior      | sample file                          | role            |
|------|----------------|--------------------------------------|-----------------|
| E0   | [17.2, 22.0]   | `pw_samples_a3_172_220_100000.mat`   | C7 baseline ref |
| E1   | [17.2, 22.5]   | `pw_samples_a3_172_225_100000.mat`   | extended prior  |

Identical model, `MAX_DLAS=3`, `SINGLE_ABSORBER_MODEL=1`,
`FILTER_LOW_LIKELIHOOD=1`, τ-EB null, 100k samples. Eval: canonical recipe
(`P_DLA≥0.99`, `SNR>2`, `λ_rf ∈ [911,1216]`, lyb-veto, no-BAL, NHI≥20.3 truth +
predicted, `--restrict-truth-to-processed`).

### Headline P/C — DONE (refreshed 2026-05-17, new DLAFLAG convention)

Source: `nhi_prior_ext_sweep/HEADLINE.tsv`. Numbers refreshed 2026-05-17
(NHI_INCONSISTENT no longer gated).

| cell | knob                 | purity | completeness | n_cat | n_truth | wall_min | node_h |
|------|----------------------|-------:|-------------:|------:|--------:|---------:|-------:|
| E0   | NHI [17.2, 22.0]     | 0.7907 | 0.8421       | 4520  | 581     | 52.4 | 0.437 |
| E1   | NHI [17.2, 22.5]     | 0.7775 | 0.8328       | 4790  | 581     | 54.2 | 0.452 |

E1 (extended prior) is **−1.3pp purity and −0.9pp completeness** vs E0.
(The pre-refresh gated eval showed the two as byte-identical — that was a
DLAFLAG-gating artifact; un-gated, a small difference appears.) The raw
catalog count rises +270 rows (+6%) with the extended prior.

### Per-SNR-bin P/C

The [22.0, 22.5] tail is a sub-percent population in a 5k London-0 slice
(London-0 truth tops out near logNHI≈22.47), so the hypothesised
high-NHI×high-SNR *recovery* is below this slice's sensitivity. What the
refreshed eval does show is a small *net* cost — the extended prior's extra
support mostly adds marginal low-NHI catalog rows here, not high-NHI wins.

## 6. Verdict + production recommendation — marginal P/C cost; adopt for modelling correctness, re-check at scale

Refreshed result: the extension is **not** P/C-neutral — it costs ~1pp on
both purity and completeness on the 5k slice (near, but not clearly below,
the ~1pp noise floor). It is near cost-neutral in wall time (+3%).

The case for adopting it is now purely the **modelling defect** it fixes:
with the hard 22.0 ceiling, rare real DLAs above 22.0 have no prior support
and their NHI is pinned low at the 22.0 edge. That benefit is real but
invisible at 5k.

**Recommendation (revised):** the extension is defensible for modelling
correctness, but the refreshed 5k smoke shows a marginal P/C cost rather
than the "free" result the gated eval suggested — a −1.3pp purity /
−0.9pp completeness hit is a real (if near-noise) cost against the 85/85
target. **Re-check E0 vs E1 at larger scale (or on the 1M catalog) before
committing the `DLA_SAMPLES_FILE` swap.** This downgrades the runbook's "firm" status for
the [17.2, 22.5] prior to "provisional — pending a scale re-check".

## 7. Execution status

**2026-05-15 (authoring session)**: blocked — the harness denied Python/bash
execution, so `_gen_extended_prior.py`, `_plot_nhi_hist.py` and `_chain.sh`
were authored but not run.

**2026-05-15 17:23–18:18 (later session)**: unblocked and executed as sbatch
job `53013668` (`gpdla_nhiprior`, regular QOS, completed 18:18). The
extended prior `pw_samples_a3_172_225_100000.mat` was generated, E0+E1 ran,
postprocessed, and were evaluated — see the filled-in P/C table in §5 and
the verdict in §6. The design conclusion (alpha=0.97, single-variable
extension via the existing mixture machinery) held: no bespoke
extrapolation was needed.
