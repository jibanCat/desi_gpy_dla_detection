# Mock-transfer audit — what London-0 and Saclay-0 genuinely validate

REVIEW-ONLY (Phase A). Independent re-measurement, 2026-08-05 adversarial review.

## Measurement

Bit-level comparison of all 32 arrays in the three v11 packs
(`/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/PRESERVED_2026-07-28_small_artifacts/modelA_packs/modelA_pack_{2lpt0,london0,saclay0}_v11.npz`),
via `np.array_equal` per key (command in this review's session log; independent of
the previous session's comparison script).

**IDENTICAL across all three packs (24 keys):** `fp_counts`, `g_grid`,
`g_occupancy`, `kz_to_K`, `molly_n_det`, `molly_n_tot`, `molly_nhi_edges`,
`molly_snr_edges`, `nhat_edges`, `nhat_masked_bins`, `ntrue_edges`,
`resp_N_fit_range`, `resp_N_ref`, `resp_mu_coef`, `resp_sig_coef`,
`resp_sig_floor`, `resp_skew_coef`, `resp_skew_ramp`, `resp_snr_edges`,
`resp_z_edges`, `snr_edges`, `t_sigma`, `zc_edges`, `zf_edges`.

**DIFFER (8 keys):** `counts`, `dX`, `dX_coarse_committed`, `fp_E_alloc`,
`fp_ell_eff`, `fp_w_sightline_ratio`, `truth_counts`, `truth_counts_bks`.

## Per-item classification (does it vary independently across mocks?)

| calibration item | varies? | consequence |
|---|---|---|
| response (all `resp_*` coefficients) | NO | response transfer never tested |
| completeness (`molly_n_det`, `molly_n_tot`) | NO | completeness transfer never tested |
| FP calibration counts (`fp_counts` = 89 loa-0 events) | NO | FP-template transfer never tested |
| `g` (`g_grid`, `g_occupancy`) | NO | not tested |
| `t_sigma` (transfer-prior width) | NO | asserted, never measured per-mock |
| path-length construction (`dX`, `fp_E_alloc`) | YES | per-mock exposure genuinely differs |
| FP exposure scalars (`fp_w`, `fp_ell_eff`) | YES | per-mock sightline ratios genuinely differ |
| truth population (`truth_counts`) | YES | genuinely different truths |
| observed realization (`counts`) | YES | genuinely different realizations |

## Verdict

The checkpoint §7 claim is **UPHELD and independently re-measured**: every
calibration block is one frozen 2LPT-0 product spliced into all three packs.
London-0 and Saclay-0 therefore test **prediction transfer** — the fixed
calibration applied to a different truth population, different exposure, and a
different observed realization. They do **not** test nuisance-calibration
transfer: a bias common to the shared calibration (response, completeness, FP
template, `t_sigma`) is invisible to the three-mock comparison **by
construction**. This also explains mechanically why the degeneracy geometry
agrees across mocks to ~3 significant figures — the operator is the same object.

## Manuscript-safe wording (strongest supportable claim)

> "London-0 and Saclay-0 provide held-out tests of forward prediction under a
> single fixed calibration derived from 2LPT-0: the truth population, the
> absorption-path exposure, and the observed catalogue realization vary, while
> the response, completeness, and false-positive calibrations are held fixed by
> construction. These comparisons test whether the fixed calibration predicts
> independent realizations; they do not constitute independent validations of
> the calibration itself, and any error common to the shared calibration would
> not be detected by them."

Forbidden wording (unsupported): "nuisances validated on three mocks",
"calibration transfer-tested", "independently calibrated per mock".
