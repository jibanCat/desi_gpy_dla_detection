# `CDDF_analysis/hbi/` — Catalog-HBI DLA measurement (quick start)

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
  validation** (truth known). Real-survey figures and numbers are not committed.

## Headline result (mock-injection validation)

On a mock-injection test (DLA truth known), the catalog-HBI estimator recovers
the injected DLA population, and in particular corrects the two failure modes of
a raw feed-forward (uncorrected-posterior) measurement: the **incompleteness
deficit** in dN/dX and the **Ω over-statement** from the uncorrected high-N_HI
posterior tail.

| ![integrated](figures/fig_compare_integrated.png) | ![f(N)](figures/fig_compare_fN.png) |
|:--:|:--:|
| Integrated dN/dX & Ω_DLA: HBI vs raw feed-forward vs injected truth | Differential f(N_HI): the high-N tail HBI re-steepens |

- `figures/fig_compare_integrated.png` — integrated dN/dX and Ω_DLA for both HBI
  FP variants vs the raw feed-forward vs the injected truth, with R0 (=
  method/truth) annotated.
- `figures/fig_compare_fN.png` — the differential f(N_HI): the raw feed-forward
  tail is too flat (drives the Ω over-statement); HBI's kernel deconvolution
  re-steepens it.

## Full documentation

The end-to-end tutorial (runnable notebook + script), the complete reporting
conventions and known limitations, the synthesis reducer, and the per-mock
figures live in the **private analysis-notes repository** (not in this public
code repo), under `hbi/`.

## Code referenced (not modified)

| module | role |
|---|---|
| `CDDF_analysis/cddf_catalog_hbi.py` | the catalog-HBI estimator (1/Vmax + FP subtraction + marked-Poisson MAP fit) |
| `CDDF_analysis/run_phase3d_postkernel.py` | stage runner (kernel build, point fit, tilt-closure gate) |
| `CDDF_analysis/calc_cddf.py` | raw feed-forward (uncorrected) baseline |
