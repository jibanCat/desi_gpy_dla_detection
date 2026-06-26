# `tutorial_data/` — small mock-only fixtures for the HBI tutorial notebooks

These are the committable, machine-independent fixtures the HBI tutorial
notebooks (`notebooks/HBI_0*.ipynb`) load so they run end-to-end on a fresh
checkout **without** the scratch validation caches, the 1.7 GB posterior
kernel, or any GPFS spectra read.

**Everything here is 2LPT-0 *mock* (injected truth known). No real-survey
(LOA) values.** Safe to commit publicly. The notebooks reconstruct the real
forward kernel / fit results / band / validation from these cached coefficients
via the production dataclasses — the only steps that still require scratch are
the full from-scratch MAP refit (the 1.7 GB kernel) and the cross-mock / real-LOA
application, which the notebooks show as commands, not run.

## Phase 1 (NB0, NB1)

| file | what it is | provenance |
|------|------------|------------|
| `compare_synthesis.json` | 2LPT-0 injection-recovery summary: `table.truth` + `table.methods` (`raw_feedforward`, `HBI_purity_mixture`, `HBI_loa0`) × dN/dX,Ω × {20.0,20.3,20.6}. | `hbi_validation_2lpt0/figures/compare_synthesis.json` |
| `compare_R0_table.md` | The same numbers as a human-readable R0 table + reporting conventions. | `hbi_validation_2lpt0/figures/compare_R0_table.md` |

## Phase 2 — NB2 (forward-response kernel)

| file | what it is | provenance |
|------|------------|------------|
| `znz_2lpt0.npz` | bias/scatter polynomial coefficients (`b_coef`,`sig_coef`) → reconstruct `ZNZModel` (the forward bias b(x̂,z) + scatter σ(x̂,z) surfaces). | `track_c/stage0/znz_2lpt0.npz` |
| `forward_response_2lpt0.npz` | empirical forward density (`emp_rho`,`emp_N_anchors`,`emp_r_grid`) + parametric coeffs → reconstruct `EmpiricalForwardDensity` p(x̂\|N,SNR,z) (headline `resp_family="empirical"`). | `track_c/stage0/forward_response_2lpt0.npz` |

## Phase 2 — NB3 (likelihood, MAP, band)

| file | what it is | provenance |
|------|------------|------------|
| `point_kernel_pm.npz` | the **headline** (purity_mixture) MAP fit: `f_b` (52), `theta_map` (15), `logN_lo/hi`, and `dndx_total_*`/`omega_*` at 20.0/20.3/20.6. | `hbi_validation_2lpt0/hbi/pm_verify_canonical/phase3d_v3_point_kernel.npz` |
| `point_kernel_loa0.npz` | the **loa0 cross-check** MAP fit (same schema). | `hbi_validation_2lpt0/hbi/loa0_run/phase3d_v3_point_kernel.npz` |
| `stage2_band_compare.npz` | the correlated MC band (the statistical band, recentered on the MAP point). | `hbi_validation_2lpt0/stage2/stage2_band_compare.npz` |
| `molly_matrix_nhi195_lyaonly.tsv` | per-(SNR,logN) completeness/purity (the lya_only matrix that reproduces the baseline). | `gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv` |
| `loa0_fp_product_lyaonly1025.npz` | the directly-measured loa-0 forest false-positive product (per-detection share). | `gl_loa0_fp_v1_20260615/outputs/loa0_fp_product_lyaonly1025.npz` |

## Phase 2 — NB4 (validation, mock-only)

| file | what it is | provenance |
|------|------------|------------|
| `recovery_mock_data.csv` | per-z dN/dX,Ω recovery (truth / raw-FF / HBI-loa0 / HBI-pm). | notes-draft `scripts/recovery_mock_data.csv` |
| `h4_zstructure.tsv` | the R0(z) decomposition (raw-migration × 1/C × kernel residual) showing the kernel residual is the climbing factor (the z-gradient). | notes-draft `scripts/r0_decompose_cache/h4_zstructure.tsv` |
| `rawff_results.json`, `rawff_zresolved.json` | the raw feed-forward (uncorrected) mock CDDF, integrated + per-z. | `hbi_validation_2lpt0/rawff/` |
| `null_field_hbi_results.json` | HBI on an HCD-free control field → dN/dX(≥20.3) ≈ 0 (FP subtraction validated). | `cddf_o3_realdata/null_field/hbi_out/` |
| `sbc_realizations.npz` | block-bootstrap realizations for the SBC coverage check. | `cddf_o3_realdata/sbc/sbc_realizations.npz` |
| `wall1_full_injection_baseline_divided.tsv` | the full-injection slope-closure (inject ±tilt, re-run GP, baseline-divided) → slope dependence ~1.8% at the operating point. | `wall1_inject/reduce_out/` |
| `competed_completeness.npz` | the injection-campaign completeness C(logN_true, SNR). | `cddf_o3_realdata/campaignS_2lpt0/competed_completeness.npz` |

Reference values (≥20.3 DLA headline; for notebook self-checks): truth dN/dX = 0.05434,
10³Ω = 0.6288; R0 — raw-FF 0.904 / 1.468; HBI_purity_mixture 1.090 / 1.029; HBI_loa0 1.159 / 1.114.

## Regenerating

Run `./regen_tutorial_fixtures.sh` to re-copy these from the scratch caches (a
copy, not a recompute — the provenance is the 2LPT-0 injection validation run).
Mock data only; the script never touches a real-survey (LOA) result file.
