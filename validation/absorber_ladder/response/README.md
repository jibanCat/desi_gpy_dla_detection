# `validation/absorber_ladder/response` — R1 response fixed-calibration variants

VALIDATION-ONLY. No sampler is run; nothing under `CDDF_analysis/` is modified. Governing ruling:
`governance/PI_RULING_2026-09-13b_ABSORBER_LADDER_GO_CUT_FEEDBACK.md` §4, §5, §9, §15, §19.

Read `RESPONSE_VARIANTS_REPORT.md` first — it carries the variant table, the held-out numbers, the
diagnosed cause of the width defect and the open items.

| file | env | what |
|---|---|---|
| `extract_calib_events.py` | `gpdla` | rebuilds the 2LPT-0 natural-pair matched calibration events (the raw pair table is not on disk); loader block copied verbatim from `CDDF_analysis/hbi_mcmc/build_kernel_fit_ensemble.py` |
| `respfit.py` | `gpdla` | sub-bin moment estimators (`sample` / `ml` / `ml_trunc`), per-cell + shared-order moment surfaces, the bin-marginal quadrature repair, CV helpers |
| `opmetrics.py` | `gpdla` | row moments / leakage, empirical rows, stratified aggregation, the `Mg` gather; calls the COMMITTED `count_conserving_fold.surface_masses` file-directly |
| `r1d_empirical.py` | `gpdla` | the alternative smoothed-empirical mass representation |
| `build_variants.py` | `gpdla` | the driver: four fail-closed gates, CV model selection, all products |
| `make_figures.py` | `gpdla` | the three figures |
| `../../../tests/test_response_variants.py` | either | unit tests, incl. element-wise equality against the committed `fitlib` / `run_d2b_lib` |

Gates the driver enforces (all fail-closed):

1. refitting the estimator of record reproduces `adopted_response_v1p1.npz` with max |Δ| **exactly 0**;
2. R0's bin masses reproduce the pack's deployed kernel bit-for-bit;
3. `adopted_phi_ref` equals the DEPLOYED kernel's in-grid fraction (the count-conservation rule);
4. (checked separately, recorded in the report) `Mg_R0_2lpt0.npz` is bit-identical to
   `cc_posterior_validation.build_cc_tensors`'s tensor.

Products land in `/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/response/`
(`response_<variant>.npz`, `Mg_<variant>_<fam>.npz`, `cv_records_<variant>.npz`,
`variants_report.json`, `SHA256SUMS`). Figures and a copy of the report go to
`/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_absorber_ladder/response/`.

**Nothing here is adopted.** Per the ruling, a PASS is a candidate for a PI choice, not a decision.
