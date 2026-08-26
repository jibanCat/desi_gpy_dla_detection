# Archived in the Paper-1 code review (2026-08-26)

Moved here, unchanged, because they obscured the production path and nothing on the
Paper-1 path (no production script, no test, no frozen artifact's provenance) references them:

| file | was | why |
|---|---|---|
| `research_unified_pdla.py`, `research_unified_pdla_perdla.py` | `tools/research/test_unified_pdla*.py` | not tests (no `test_*` functions), NERSC-only paths; the `test_` prefix made pytest collect and import them |
| `phase2_pt_to_h5.py` | `tests/phase2_pt_to_h5.py` | zero inbound references; `tools/research/convert_phase2_pt_to_h5.py` duplicates it |
| `combine_dlakibo.py` | repo root | referenced only by three stale docs; no code path, test or sbatch |
| `broaden_kernel.py` | `CDDF_analysis/hbi/` | one-off width hack on a cached kernel, superseded by the forward-response kernel |
| `wall1_explain_partA.py` | `CDDF_analysis/hbi/` | explanatory-doc plumbing |
| `hbi_figures/` | `CDDF_analysis/hbi/figures/` | two committed PNGs excluded from packaging, referenced by no manifest |

Deliberately NOT moved (referenced by production, tests, sbatch scripts or frozen provenance):
`tests/phase2_train_{desi,dr16}.py` (training entry points referenced by 3 sbatch scripts and 6 docs — a
relocation would break them; documented in `docs/PAPER1_REPRODUCTION.md`), `tests/a4_inference.py`
(imported by `test_review_fixes.py`), the `tests/plot_*`/`compare_a3_results.py`/`_overnight_a3_a4_chain.sh`
GP-training investigation set (documented in `docs/training_overview.md`), `diagnostics_phaseB/`,
`diagnostics_phaseC/` (provenance of `p1_natpair_ck_v1.npz`), `CDDF_analysis/diagnostics/` (referenced by
`REPRODUCE_HEADLINE.md`), `CDDF_analysis/hbi_mcmc/diag_*` (provenance of the Battery-2/3 gate product),
`notebooks/` (provenance of the FIG-04/05-era raw tables), `examples/`.
