# CDDF_analysis diagnostics — archived audit trail

One-off investigation scripts (reduce-only). Kept for provenance; not part of the
supported reproduction path. Grouped by theme.

| script | theme | investigated | first commit |
|---|---|---|---|
| `decompose_r0_dla_tier.py` | r0_decompose | Four-hypothesis decompose of the DLA-tier R0=1.16 over-recovery on 2LPT-0: completeness, N_HI bias, z-trend, and kernel-width contributions | `2736ad0 2026-06-19` |
| `decompose_r0_highn_stratify.py` | r0_decompose | High-N (>=21.0) stratification diagnostic: isolates whether the shoulder over-recovery comes from high-z_QSO line blending, NHI bias, or kernel width | `e489e8c 2026-06-19` |
| `decompose_r0_zstructure.py` | r0_decompose | Phase-1 drill into the z-rising R0 trend: decomposes the 0.91->1.19->1.42 gradient into NHI-bias x z, z-flat-C, and kernel z-residual | `2736ad0 2026-06-19` |
| `sbatch_reunion.py` | diagnostics | Legacy utility to re-combine split .mat SLURM output files (SDSS era) | `27579e6 2020-03-31` |
| `subdla_basis_pad_bracket.py` | subdla | Bracket test for cfg.basis_pad_floor: extends the deconvolution basis below the 19.5 fit floor without admitting FP-heavy detections; tests whether the low-edge R0 deficit is a basis-truncation artifact | `2736ad0 2026-06-19` |
| `subdla_floor_mc_band.py` | subdla | MC error bar on the sub-DLA integrated [19.5,20.3) R0 for floor-19.5 vs floor-19.0, using the same joint-MC recipe as the estimator's joint_mc_errors | `b905a54 2026-06-17` |
| `subdla_loa0_validation_floor190.py` | subdla | Floor-19.0 variant of the sub-DLA edge recovery validation: tests whether lowering the fit floor rescues the [19.5,19.7) non-identifiable edge | `2736ad0 2026-06-19` |
| `subdla_loa0_validation.py` | subdla | Sub-DLA-tier validation of the corrected loa-0 forest-FP HBI estimator against the 2LPT-0 truth; first end-to-end R0 check for the [19.5,20.3) band | `d253de0 2026-06-17` |
| `make_track_c_coverage_fig.py` | track_c | Track-C shoulder coverage figure: reads td_band.json deliverable and renders the headline truth-in-bar coverage plot at the DLA report limits | `e976657 2026-06-20` |
| `track_c_bref_r0check.py` | track_c | After-the-fact R0(z) check under each non-circular re-center recipe (MEAN_bref0, MEDIAN, MODE variants) to select the least-biased b_ref functional | `e8d98b1 2026-06-20` |
| `track_c_bref_skew.py` | track_c | Non-circular diagnosis of the kernel re-center statistic: resolves the b_ref contradiction by measuring the conditional dx=xhat-xtrue skew distribution per (xhat,z) cell from the truth-match alone | `e8d98b1 2026-06-20` |
| `track_c_czresolve_diag.py` | track_c | Step-0 diagnosis for Track-C #39: measures C_true(N,z) directly from the truth-match (non-circular TP/truth counts) and answers whether z-dependent completeness explains the z-tilt | `722edf0 2026-06-21` |
| `track_c_czresolve_point_ab.py` | track_c | Fast point-only A/B for z-resolved completeness: compares C·g(N,z) ON vs OFF (z-marginalized molly) to verify the per-z amplitude tilt flattens before paying for the full MC band | `722edf0 2026-06-21` |
| `track_c_eddington_verify.py` | track_c | Reduce-only Eddington-bias falsifier: decides whether the kernel's strong left skew (p(x_true|xhat) skew -1.5..-1.9) is Eddington bias from a steep f convolved with a symmetric forward response, rather than per-system N_HI over-estimation | `e8d98b1 2026-06-20` |
| `track_c_tbc_smoke.py` | track_c | Track-C T-BC smoke: runs the v3 bspbody point estimate twice under resp_kind="kappa" (GP posterior kernel) vs resp_kind="forward" (T-A ForwardResponseModel), to verify the +9% over-recovery traced to the wrong kernel object | `32a2bd4 2026-06-20` |
| `track_c_ztilt_guard.py` | track_c | Track-C #38 Step-0 guard: decides whether the dN/dX(z) tilt (0.91->1.19) is a real z-dependent truth shape vs a completeness/response artifact before adding a z-slope parameter to the separable population model | `10229ae 2026-06-21` |
| `track_c_shape_diag.py` | track_c | Track-C f(N) shape-prior diagnosis (mock, MAP-point only, no MC band): sweeps the bspbody curvature penalty / edge-slope anchor / family to test whether the shape prior bends f(N) down at the low-N (~20.0) and high-N (>21) edges | `chore/repo-hygiene 2026-07-05` |
| `freebin_localizer.py` | validation | Decisive 2x-under-recovery localizer: runs the free-bin (non-parametric, one-DOF-per-(N,z)) v2 estimator in place of v3 bspbody on the same kernel, to localize which N bins drive the under-recovery | `62d38cb 2026-06-14` |
| `hbi_fNz_coverage.py` | validation | Per-z differential CDDF f(N,z) coverage deliverable: checks whether f_truth[b,k] sits inside the HBI marginalized band for each (logN bin, coarse-z bin) cell | `d5cebce 2026-06-20` |
| `hbi_stage3_fN_coverage.py` | validation | Stage-III differential f(N) per-bin coverage diagnostic: measures whether the 2LPT-0 truth f(N) lies inside the Stage-III (response marginalization) MC band, bin by bin in logN | `3b0f885 2026-06-20` |
| `hbi_validation_2lpt0.py` | validation | Clean single-source validation of the catalog-HBI DLA measurement on the 2LPT-0 mock: reduce-only, no tilt, reuses the exact calibrated WALL-1 bundle | `2736ad0 2026-06-19` |
| `hbi_validation_2lpt0_stage2.py` | validation | Stage II validation on 2LPT-0: independent-Beta vs shared-bootstrap calibration nuisance band, at mc_inner=laplace, for both loa0 and purity_mixture FP paths | `dc537e1 2026-06-20` |
| `hbi_validation_2lpt0_stage3.py` | validation | Stage III validation on 2LPT-0: the response (theta_K) marginalization coverage test — the dominant coverage lever — for the faithful composed Stage I+II+III band | `1438d59 2026-06-20` |
| `hbi_validation_replot.py` | validation | Regenerate the two validation figures + summary md from a persisted hbi_validation_results.json without re-running the ~35-min MC bands | `2736ad0 2026-06-19` |
| `wall1_explain_figures.py` | wall1 | Render the 5 explanatory figures for the WALL-1 tilt-closure + recovered-CDDF-posterior doc, from the partA/partB npz products (no MC, no inference) | `2736ad0 2026-06-19` |
| `wall1_explain_partB.py` | wall1 | WALL-1 tilt-closure data + tilt-mechanism arrays for the explanatory doc: extracts cached closure arrays into a single npz and recomputes the truth-side injected-tilt mechanism | `2736ad0 2026-06-19` |
| `wall1_full_injection.py` | wall1 | WALL-1 full-injection reduce orchestrator: loads the re-inferred detection catalogs from the tilted-injection arms and runs the HBI closure vs the injected truth to confirm the slope-response is genuine | `2736ad0 2026-06-19` |
| `wall1_is_faithfulness.py` | wall1 | WALL-1 full-injection IS-faithfulness check: tests whether reweighting the untilted GP detections by w(logN)=10^(Dalpha*(N-20.3)) is a faithful proxy for genuinely re-inferring a tilted-slope population | `2736ad0 2026-06-19` |
