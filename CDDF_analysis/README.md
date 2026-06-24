# CDDF_analysis

Population statistics (dN/dX, f(N), Omega_DLA) from GP-DLA catalogs. Two supported pathways:

## Pathway A — Bayesian posteriors (Bird/Ho+2021 reproduction)

`calc_cddf.py` consumes GP-DLA HDF5 posteriors to produce dN/dX, f(N), and Omega with
Poisson-binomial CIs. Figures/tables via `make_plots.py`, `make_tables.py`,
`make_multi_dla_plots.py`. Real-LOA / raw-feed-forward drivers: `loa_literal_calccddf.py`,
`rawff_2lpt0.py`. Direct-catalog stats: `cddf_mock.py`; calibration/IO:
`cddf_calibration.py`, `cddf_io.py`.

## Pathway B — selection-corrected catalog-HBI / Track-C  ->  `hbi/`

Reduce-only estimator over a frozen catalog (no re-inference). See `hbi/README.md` and
`hbi/QUICKSTART.md`. Feed-forward building blocks live in `cddf_forward/`.

## Layout

- root: Pathway-A + direct-stats + legacy library
- `hbi/`: catalog-HBI / Track-C estimator + reproduction drivers
- `cddf_forward/`: feed-forward subpackage
- `diagnostics/`: archived one-off audit scripts (see `diagnostics/README.md`)

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
