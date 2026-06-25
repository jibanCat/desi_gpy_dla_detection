# `CDDF_analysis/hbi/` — Catalog-HBI DLA measurement

For the shortest runnable path from a frozen catalog to a dN/dX/Omega/f(N) band,
see **[QUICKSTART.md](QUICKSTART.md)**.


The **catalog-HBI** estimator turns a GP-DLA detection catalog into a
selection-corrected DLA population measurement — the column-density distribution
**f(N_HI)**, the line density **dN/dX**, and **Ω_DLA** — using a GW-style
rate-form (marked-Poisson) hierarchical Bayesian model that deconvolves the
measurement kernel and subtracts the false-positive rate.

## Discipline

- **Reduce-only.** The pipeline reuses the *frozen* GP posteriors byte-for-byte.
  No inference code is touched (`gpy_dla_detection/dla_gp.py`,
  `run_bayes_select.py`, `dlasearch.py` are never modified) and there is **zero
  re-inference**.
- **Mock figures only in-repo.** The figures here are from a **mock-injection
  validation** (truth known) and are regenerated from a committed mock table
  (`figures/compare_mock_data.csv` + `figures/make_compare_figs.py`). Real-survey
  figures and numbers are **not** committed.

## Headline result (mock-injection validation)

On a mock-injection test (DLA truth known), at the **logN_HI ≥ 20.3 DLA headline**
the catalog-HBI estimator recovers the injected DLA population, and in particular
corrects the two failure modes of a raw feed-forward (uncorrected-posterior)
measurement: the **incompleteness deficit** in dN/dX and the **Ω over-statement**
from the uncorrected high-N_HI posterior tail.

Integrated recovery ratios **R0 = method / injected-truth** at logN_HI ≥ 20.3
(`z ∈ [2.0, 3.5]`, the numbers annotated on the figure):

| method | dN/dX R0 | Ω R0 |
|---|---|---|
| raw feed-forward (uncorrected) | **0.904** (−9.6% deficit) | **1.468** (Ω blow-up; 1.61 at ≥20.6 → "40–61%") |
| HBI purity_mixture (headline) | **1.090** (+9%) | **1.029** (+3%) |
| HBI loa0 (conservative cross-check) | **1.159** (+16%) | **1.114** (+11%) |

i.e. HBI recovers the injected truth at ≥20.3 to **~9% (purity_mixture headline) /
~16% (loa0) in dN/dX** and **~3% / ~11% in Ω**, while the raw feed-forward
*under-counts* dN/dX by ~10% yet *over-states* Ω by ~47–61% (the un-deconvolved
high-N posterior tail).

| ![integrated](figures/fig_compare_integrated.png) | ![f(N)](figures/fig_compare_fN.png) |
|:--:|:--:|
| Integrated dN/dX & Ω_DLA at **logN_HI ≥ 20.3**: HBI vs raw feed-forward vs injected truth | Differential f(N_HI): the high-N tail HBI re-steepens |

- `figures/fig_compare_integrated.png` — integrated dN/dX and Ω_DLA at
  **logN_HI ≥ 20.3** for both HBI FP variants vs the raw feed-forward vs the
  injected truth, with R0 (= method/truth) annotated.
- `figures/fig_compare_fN.png` — the differential f(N_HI): the raw feed-forward
  tail is too flat (drives the Ω over-statement); HBI's kernel deconvolution
  re-steepens it back onto the injected truth.

Both figures are regenerated reproducibly from the committed mock table by
**[`figures/make_compare_figs.py`](figures/make_compare_figs.py)** (reads only
`figures/compare_mock_data.csv`; see *Reproducing the figures* below).

> **What the mock validates — and what it does NOT.** The on-mock R0 ≈ 1 at the
> ≥20.3 headline (HBI over-recovers dN/dX by ~9% (purity_mixture) to ~16% (loa0),
> and Ω by ~3% to ~11%) is a **self-consistency** check, *not* an
> external-calibration claim. The completeness / kernel correction is fit on the
> *same* mock's truth, so by construction the corrected estimate is pulled toward
> that truth — the residual α = 1/R0 → 1 is a near-**tautology**. This demonstrates
> the estimator's internal machinery (kernel deconvolution + FP subtraction + the
> marginalized band) is arithmetically sound; it does **not** prove the calibration
> *transfers*. The real, non-circular test is **cross-mock**: build the
> kernel/completeness on one mock and check α(z) ≈ 1 on a *held-out* mock or survey
> (London, Saclay, real LOA) **without refitting**. The cross-mock **drivers are
> included** here — `track_c_tf_2lpt1.py`, `track_c_tf_london0.py`,
> `track_c_tf_loa.py` (transfer / freeze-kernel-and-apply runners) — and the
> validated transfer result (London-0 done, Saclay pending) is reported in the
> analysis-notes paper draft, not duplicated in this README. Treat the mock R0 as
> self-consistency; treat the cross-mock / real-LOA literature agreement as the
> calibration evidence.

## Uncertainty budget (read before quoting an error bar)

The HBI MC band is a **statistical** band only — label it "statistical (indep. MC)".
A 3-referee stat-panel audit (re-running the band machinery on the 2LPT-0 mock)
established:

- **The statistical band is correctly sized**, not under-dispersed: the dN/dX and
  f(N) 68% half-widths are ~2–3× the irreducible Poisson floor, MC-converged
  (use `n_mc ≥ 240`), and ~1.2–1.6× wider than a plain sightline bootstrap. It
  carries the truth-match bootstrap (C/ρ/g), the inner Laplace, the per-object
  N_HI width, and the kernel-calibration MC.
- **It is NOT the total uncertainty.** For a science claim, combine the *symmetric*
  terms in quadrature — and keep the *one-sided* high-N kernel-shape term as a
  **separate signed line**, because it is a bias (the truth lies above the high-N
  points), not scatter, and folding it into the symmetric quadrature double-counts
  and over-inflates σ_tot:

  | term | dN/dX | Ω |
  |---|---|---|
  | statistical band (this MC) | ~0.7–1.7% | ~1% |
  | model-shape / B-spline penalty (λ) | **≈ band-sized (→ ~√2× wider)** | <1% (shape-robust) |
  | mean-flux | 1–2% | 1–2% |
  | integrated DLA-tier method/kernel spread (pm↔loa0 pre-α) | ~8–9% | — |
  | Ω deep-tail / high-N shoulder (symmetric part) | — | ~11–12% |
  | **σ_tot (symmetric, rough)** | **±8–9%** | **±12–16%** |
  | high-N kernel-shape deep-tail (ONE-SIDED, *not* summed in) | bias: truth **above** points (R0 0.68→0.40 at logN≳21.6) | bias: Ω point ~11% low |

  Under σ_tot, literature anchors that differ from each other by ~10–15% are mutually
  consistent at ≲1σ — a ~1% statistical band cannot envelope them, by construction.
  The one-sided deep-tail row is a *systematic flag*, reported separately, never
  added into the symmetric ±8–9% / ±12–16% quadrature.

- **Two caveats the band does NOT absorb:**
  1. **High-N tail bias (one-sided).** On-mock the MAP under-recovers the high-N
     end (R0 0.92→0.95 at logN 20.4–20.9, falling to **0.68→0.40 at logN ≳ 21.6** —
     the prior-ceiling / forward-kernel deep-tail systematic). It is a *bias*, not
     scatter: the true f(N) lies **above** the plotted high-N points. Do not widen
     the band to "cover" it — reduce/flag it.
  2. **Ω deep-tail slope-extrap over-inflation.** With `omega_slope_extrap` on
     (σ_slope=0.5) the Ω band is inflated ~100× and its coverage is uninformative;
     the Ω point itself is ~11% biased low. Do not cite Ω band coverage as
     unbiasedness evidence.

- **f(N) band centering.** The differential f(N) band is recentered on the plug-in
  MAP point (`band_recenter`, the same first-order Jensen correction as the
  integrated dN/dX/Ω); without it the band floats ~17.5% above the line.

## Reproducing the figures

The two README figures are regenerated from a small **committed, self-contained**
mock table — no private-repo or scratch-cache dependency at run time:

```bash
HDF5_USE_FILE_LOCKING=FALSE conda run -n gpdla python \
    CDDF_analysis/hbi/figures/make_compare_figs.py
```

- `figures/make_compare_figs.py` — the committed generator; reads **only**
  `figures/compare_mock_data.csv` and writes `fig_compare_integrated.png`
  (integrated dN/dX & 10³·Ω at logN_HI ≥ 20.3) and `fig_compare_fN.png`
  (differential f(N_HI)). It prints the headline R0 values for a self-check
  against the table above.
- `figures/compare_mock_data.csv` — the committed mock-injection numbers
  (2LPT-0 validation; truth known). Integrated ≥20.3 block + the differential
  f(N_HI) arrays (truth / HBI loa0 / HBI purity_mixture / raw feed-forward, plus
  the HBI loa0 MC 68% band). **MOCK values only — no real-survey numbers.**

## Full documentation

The figure *generator* is committed here (above); the **full end-to-end pipeline
tutorial** (runnable notebook + script that re-runs the estimator from a frozen
catalog), the complete reporting conventions and known limitations, the synthesis
reducer, and the per-mock figures live in the **private analysis-notes repository**
(not in this public code repo), under `hbi/`.

## Code referenced (not modified)

| module | role |
|---|---|
| `CDDF_analysis/hbi/cddf_catalog_hbi.py` | the catalog-HBI estimator (1/Vmax + FP subtraction + marked-Poisson MAP fit) |
| `CDDF_analysis/hbi/run_remp_kernel.py` | empirical truth-match response (R_emp) kernel driver — builds the forward-response kernel (QUICKSTART Option A) |
| `CDDF_analysis/hbi/run_phase3d_postkernel.py` | stage runner (kernel build, point fit, tilt-closure gate) |
| `CDDF_analysis/hbi/track_c_tf_2lpt1.py`, `track_c_tf_london0.py`, `track_c_tf_loa.py` | cross-mock transfer drivers (freeze the 2LPT-0 kernel, apply to a held-out mock / real LOA — the non-circular α(z)≈1 test) |
| `CDDF_analysis/calc_cddf.py` | raw feed-forward (uncorrected) baseline |
