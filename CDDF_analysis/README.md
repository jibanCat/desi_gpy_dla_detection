# CDDF_analysis

Population statistics (dN/dX, f(N), Omega_DLA) from GP-DLA catalogs. Two supported pathways:

## Pathway A — Bayesian posteriors (Bird/Ho+2021 reproduction)

`calc_cddf.py` consumes GP-DLA HDF5 posteriors to produce dN/dX, f(N), and Omega with
Poisson-binomial CIs. Figures/tables via `make_plots.py`, `make_tables.py`,
`make_multi_dla_plots.py`. Real-LOA / raw-feed-forward drivers: `loa_literal_calccddf.py`,
`rawff_2lpt0.py`. Direct-catalog stats: `cddf_mock.py`; calibration/IO:
`cddf_calibration.py`, `cddf_io.py`.

### `calc_cddf.py` retirement status — SETTLED 2026-07-28: **PARTIAL, not full**

The module is **NOT retired**. Its **single-absorber path is LIVE** and is the
paper's feed-forward (FF) estimator. **Only the multi-DLA increment path is
retired.**

* **What production exercises.** Every packaged `BASELINE.env` runs
  `SINGLE_ABSORBER_MODEL=1`, so the FF drivers construct
  `DLACatalogue(..., sub_dla=False, second=0)`. With `second` falsy,
  `_split_distributions` never enters its `if self.second_dla:` increment loop
  and `_get_prob_dla_this_bin` returns at `if second == False` before the
  defective block. The whole production FF surface —
  `loa_literal_calccddf.py` (FF-A) and `hbi/calccddf_vs_hbi.py` (FF-B) — is
  provably disjoint from the defect.
* **The defect** (commit `b00e6e4`, 2020-03-31), in the `second != 0` branch of
  `_get_prob_dla_this_bin` only, two independent bugs:
  1. `model_posteriors[index, ...]` addresses the **spectrum** axis with
     **sample-grid** indices (`index` comes from
     `_split_distributions_single`); it must be `spec`;
  2. the accumulator is initialized to the **scalar `-1e30`** (the `np.empty`
     on the preceding line is discarded).
  Any `second != 0` number from this module is therefore meaningless. Nothing
  depends on it and it has never been repaired.
* **Guard.** That branch now raises `RuntimeError` unless the module-level
  opt-out `CDDF_analysis.calc_cddf.ALLOW_BROKEN_MULTI_DLA` is set True. Only a
  characterization test that deliberately pins the broken loop's *shape* may set
  it (`tests/test_cddf_diagonal_deposit.py::test_deposit_honors_second_dla_sum`).
* **Consequence for the FF numbers.** Slot-0 counting means ~7–8% of injected
  DLAs — the 2nd/3rd absorber in a sightline — are not separately counted, so FF
  `R0` is ~7% conservative (LOW) at the DLA tier. Fixing the increment path is
  separate, referee-reviewed debt (PI decision C3, 2026-07-11).

Full statement: the `MODULE STATUS` block at the top of `calc_cddf.py`.

### FF arm aggregation

`hbi/calccddf_vs_hbi.py` runs the literal (NaN-safe) `calc_cddf` closure per mock;
`hbi/calccddf_vs_hbi_artifact.py` is **the aggregation entry point** and writes
the stamped `hbi/calccddf_vs_hbi.json`. That artifact carries dN/dX and f(N)
only (**no Omega** — B16-contaminated), stamps 2LPT-0 as the **on-mock
calibration / recovery floor** rather than a held-out leg, declares the FF
**estimand** (a posterior-weighted plug-in CDDF that a naive mock correction
alpha is later applied to), and carries a **sampling** interval on that plug-in —
explicitly *not* a posterior credible interval.

## Pathway B — selection-corrected catalog-HBI / Track-C  ->  `hbi/`

Reduce-only estimator over a frozen catalog (no re-inference). See `hbi/README.md` and
`hbi/QUICKSTART.md`. Feed-forward building blocks live in `cddf_forward/`.

## Layout

- root: Pathway-A + direct-stats + legacy library
- `hbi/`: catalog-HBI / Track-C estimator + reproduction drivers
- `cddf_forward/`: feed-forward subpackage
- `diagnostics/`: archived one-off audit scripts (see `diagnostics/README.md`)

The 7 bare `*.py` files at root matching `hbi/` module names (`cddf_catalog_hbi.py`, `cddf_tilt_closure.py`, `znz_kernel.py`, `track_c_td_band.py`, `track_c_tf_loa.py`, `run_phase3d_postkernel.py`, `run_remp_kernel.py`) are back-compat shims (4-line `sys.modules` aliases) so pre-reorg imports `from CDDF_analysis.<mod> import ...` keep working; prefer the `CDDF_analysis.hbi.<mod>` paths.

## Legacy (SDSS / plotting)

`qso_loader.py` (+ `set_parameters.py`, `voigt.py`) — the QSOLoader plotting utilities
from the SDSS DR12/DR16 era. To reproduce Bird (2017) / Ho+2021 CDDF/dN/dX/OmegaDLA
plots use `calc_cddf.py`; to manipulate a MATLAB catalogue without
`sample_log_likelihoods_dla` use `qso_loader.py`.

```python
from CDDF_analysis.qso_loader import QSOLoader

qsos = QSOLoader(
    preloaded_file="preloaded_qsos.mat", catalogue_file="catalog.mat",
    learned_file="learned_qso_model_dr9q_minus_concordance.mat",
    processed_file="processed_qsos_multi_dr12q.mat",
    dla_concordance="dla_catalog", los_concordance="los_catalog",
    snrs_file="snrs_qsos_multi_dr12q.mat",
    sub_dla=True)

# Plot GP prior mean × MAP DLAs for spectrum at catalogue index nspec
qsos.plot_this_mu(nspec)

# With Lyman-series suppression + three Voigt lines + Parks predictions
qsos.plot_this_mu(nspec, suppressed=True,
    num_voigt_lines=3, num_forest_lines=31,
    Parks=True, dla_parks="predictions_DR12.json")
```

Concordance catalogs are downloaded via `data/scripts/download_catalogs.sh`.
ROC/CDDF comparison plots: `make_multi_dla_plots.py`.
