# Absorber-side OPERATOR forensics — where the ORACLE-pinned residual enters

**Date 2026-09-13. VALIDATION-ONLY. No sampler was run. Nothing under `CDDF_analysis/` was
modified.** Worktree `/home/mfho/wt_abs_diag_2026-09`, branch `absorber-diag-2026-09`, nothing
committed. Governing ruling: `governance/PI_RULING_2026-09-13_FP_RULE_ABSORBER_SIDE_OPTION2.md` §3
(diagnose first), §4 (anti-garbage-can), §10 (what to return).

Figures: `/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_absorber_diag/opfor_fig*.png`.

---

## 0. What was computed, and the gates it passed

Two new validation-only programs, plus pure helpers and unit tests:

| file | env | what |
|---|---|---|
| `validation/absorber_diag/binning.py` | either | pure binning / normalisation helpers |
| `validation/absorber_diag/build_matched_ops.py` | `gpdla` | matched-pair tables → `empirical_ops_<fam>.npz` |
| `validation/absorber_diag/analyze_fold.py` | `gpdla-hbi` | operator comparison → `fold_forensics_<fam>.json/.npz` |
| `validation/absorber_diag/make_figures.py` | `gpdla-hbi` | the required plots |
| `tests/test_absorber_diag.py` | either | 18 unit tests (all pass; mutation-checked) |

`build_matched_ops.py` re-cuts and re-matches each mock catalogue through the **committed**
machinery (`extract_pack.load_mock_bundle`, `cddf_catalog_hbi.load_and_cut_catalog`,
`track_c_tf_saclay._snap_off_molly_edges`), exactly as `validation/fp_ladder/build_fp_census.py`
does, at **two** truth floors, and **fails closed** on three gates:

* **TRUTH GATE** — the rebuilt `truth_counts_bks[b,k,s]` (floor 19.0, `S2N_RED > 2` strict,
  `build_truth_counts` semantics) equals the adopted pack's `truth_counts_bks` **elementwise,
  exactly**: 101 949 / 106 705 / 104 561 systems, max |diff| = 0. PASSED on all three families.
* **CENSUS GATE** — the floor-17.2 `host_17p2_19p0` and `hostless` arrays equal the census NPZ on
  disk elementwise (P6b 3200 / 2611 / 2668; hostless 13 860 / 9598 / 10 592). PASSED.
* **INDEX CONVENTION** — the local `bin_index` is asserted equal to the committed
  `extract_pack._idx` on the real arrays before use, and `KZ_TO_K == repeat([0,1,2],5)` is asserted.

**Convention used for the matched set.** Primary natural-pair truth match at floor 19.0 (so
`NHI_TILT_HOST == NHI_TRUE`), op cut `S2N_RED>2 & P_DLA>0.99 & DLAFLAG==0`, counting domain the
observed grid [19.5, 22.4). This is the pack's own adopted TP convention,
`tp_convention_id = "tp_natpair_tilthost_op/v1"`, so the empirical operators are like-for-like with
the adopted response.

**Informational (not a gate).** The rebuilt observed `counts` differ from the adopted pack's by
7 / 6 / 14 of ~88 000 (≤ 0.016 %): lowering the truth floor to 19.0 moves `Z_TRUE` for a handful of
rows through `make_lambda_z_BAL_cuts(use_truth_z=True)`, the same documented perturbation the census
records (18 / 9 / 18 at floor 17.2).

### The collar — stated as instructed

`load_and_cut_catalog` applies the **3000 km/s** collar. `build_scan_packs.py` rebuilds only
`counts`, `dX`, `dX_coarse_committed`, `fp_E_alloc` at **3300 km/s** and copies *every other array
byte-identically* — verified here: `scanpack.truth_counts_bks` is byte-identical to the adopted
pack's. The posterior runs therefore evaluate `truth_f = truth_counts / (dX_tot * dN)` with a
**collar-3000 numerator over a collar-3300 denominator**. All operator work below is done on the
**adopted** packs (collar matched to the matched-pair tables); the scan-pack mismatch is costed
separately in §7.

---

## 1. The fold, and what each object is compared against

Frozen fold (`CDDF_analysis/hbi_mcmc/cc_posterior_validation.py::model_cc` lines 195–208; tensors
`build_cc_tensors` lines 42–70; index maps `forward.py::build_consts` lines 265–458; kernel mass
`count_conserving_fold.surface_masses` / `phi_from_surfaces`):

```
mu_TP[c,k,s] = dX[k,s] * sum_b Mg[s,k,c,b] * C[s,b] * g[b,k] * f[b,k] * dN_b
```

Under the count-conservation rule (module docstring of `count_conserving_fold.py`) the kernel splits
into a **unit-mass redistribution** and a **counting probability**
`phi[s,K,b] = sum_c Mg[s,k,c,b]` that belongs to `C`. The operator that can be compared to data is
therefore

```
A_mod[b,k,s] = C_molly[s,b] * g[b,k] * phi[s,K,b]      ("detected AND on the observed grid")
```

and its measured counterpart is `C_true_bKs = N_match / truth_counts_bks`. `psi_c = 0` (the
calibrated completeness) everywhere; the ORACLE posterior-mean `psi_c` is reported in §6 and is
negligible.

**All absorber-side calibration blocks are the FROZEN 2LPT-0 ones, byte-identical in all three
packs** — verified here: `adopted_resp_{mu,sig,skew}_coef`, `adopted_phi_ref`, `molly_n_det`,
`molly_n_tot`, `g_grid`, `t_sigma` are equal across families (only `fp_ell_eff` differs). So on
2LPT-0 any mismatch is **representation error**, and on London-0/Saclay-0 it is representation +
transfer error.

---

## 2. Deliverable 1 — forward fold of truth vs the realised catalogue

### 2.1 Totals (collar 3000, `dX>0` cells only)

| | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| detections on grid | 88 064 | 87 834 | 86 749 |
| matched in basis (host ≥ 19.0) | 70 993 | 75 619 | 73 482 |
| host [19.0,19.5) / host ≥ 19.5 | 7 106 / 63 887 | 6 983 / 68 636 | 6 949 / 66 533 |
| **P6b** host [17.2,19.0) — *no term in the fold* | 3 200 | 2 611 | 2 668 |
| hostless (the ORACLE FP pin) | 13 860 | 9 598 | 10 592 |
| host above the basis top (≥22.4) | 0 | 3 | 3 |
| residual unaccounted (floor-difference rows) | 11 | 6 | 7 |
| **fold of truth, mu_TP(f_true)** | 72 510.6 | 76 541.3 | 74 123.4 |
| **mu_TP / matched** | **1.0214** | **1.0122** | **1.0087** |
| mu_TP/matched per K0/K1/K2 | 1.0228 / 1.0133 / 1.0356 | 1.0083 / 1.0119 / 1.0265 | 1.0025 / 1.0132 / 1.0235 |

### 2.2 By S/N stratum — `fold(truth)/matched` (`opfor_fig2_residual_vs_snr.png`)

| stratum | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| **S/N 2–3** | **0.9741** | **0.9362** | **0.9266** |
| 3–4 | 1.0166 | 1.0006 | 1.0065 |
| 4–5 | 1.0279 | 1.0292 | 1.0210 |
| 5–6 | 1.0315 | 1.0488 | 1.0495 |
| 6–7 | 1.0509 | 1.0638 | 1.0268 |
| ≥7 | 1.0441 | 1.0436 | 1.0480 |

A 7–12 % spread across strata, monotone, **family-common**, present with truth pinned.

### 2.3 By latent true-N bin b — `fold(truth)/matched` (the fold's implied detections per b)

| b | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| [19.0,19.2) | 1.1740 | 1.2789 | 1.2776 |
| [19.2,19.5) | 1.2781 | 1.2986 | 1.3120 |
| [19.5,19.7) | 1.1012 | 1.0689 | 1.0623 |
| [19.7,19.9) | 0.9207 | 0.9043 | 0.8885 |
| [19.9,20.1) | 1.0160 | 0.9989 | 1.0023 |
| [20.1,20.3) | 0.9752 | 0.9686 | 0.9609 |
| [20.3,20.5) | 0.9867 | 0.9805 | 0.9763 |
| [20.5,20.7) | 0.9948 | 0.9909 | 0.9863 |
| [20.7,20.9) | 0.9813 | 0.9805 | 0.9758 |
| [20.9,21.1) | 0.9964 | 0.9911 | 0.9890 |
| [21.1,21.3) | 0.9902 | 0.9880 | 0.9886 |
| [21.3,21.5) | 0.9924 | 0.9890 | 0.9848 |
| [21.5,21.7) | 0.9877 | 0.9755 | 0.9841 |
| [21.7,21.9) | 0.9863 | 0.9846 | 0.9881 |
| [21.9,22.1) | 0.9364 | 0.9555 | 0.9717 |
| [22.1,22.4) | 0.7897 | 0.8405 | 0.7448 |

The two **basis-pad** bins [19.0,19.5) are over-predicted by 17–31 %; [19.5,19.7) by 6–10 %;
[19.7,19.9) under-predicted by 8–11 %. Above 20.3 the per-b efficiency is within 1–2.5 % (always
slightly low) until the very sparse top bins.

### 2.4 By observed N̂ bin per coarse block (the zigzag at operator level)

`opfor_fig1_fold_vs_matched_by_nhat_perK.png`. 2LPT-0 example (ratio K0/K1/K2, cells with ≥ 20
matches): [19.5,19.6) 1.170/1.106/1.437 → [19.8,19.9) 1.010/1.042/0.951 → [20.2,20.3)
1.003/1.066/0.922 → a clear deficit band at **[21.1,21.5): 0.87–0.93 in every K and every family** →
an excess again above 21.7. The full per-(c,K) table (mu, obs, ratio, Poisson z with the committed
`forward_selftest.poisson_z` definition, line 185) is in `fold_forensics_<fam>.json:by_nhat_by_K`;
`by_nhat_by_snr` holds the (c,s) map.

### 2.5 The class with no term — P6b (host [17.2, 19.0))

`opfor_fig6_no_term_classes.png`. 3.6 / 3.0 / 3.1 % of all on-grid detections; only 0.15 / 0.15 /
0.08 % of the detections above N̂ = 20.3. **It rises steeply with z** — P6b / matched per block:

| | K0 | K1 | K2 |
|---|---|---|---|
| 2LPT-0 | 3.14 % | 5.40 % | 7.21 % |
| London-0 | 1.90 % | 4.18 % | 7.17 % |
| Saclay-0 | 2.22 % | 4.44 % | 7.68 % |

and it is concentrated at low S/N (2LPT-0: 789 / 755 / 535 / 333 / 212 / 576 across strata 2…7).
Per-(c,k,s) it is stored as `P6b_cks`.

---

## 3. Deliverable 2 — the empirical operators (file contract)

`validation/absorber_diag/empirical_ops_<fam>.npz` (also copied to
`/scratch/cavestru_root/cavestru0/mfho/absorber_diag_2026-09-13/ops/`, with `SHA256SUMS` in both
places):

| key | shape | definition |
|---|---|---|
| `N_match_cksb` | (29,15,8,16) | matched in-basis detections, host bin b → observed cell (c,k,s), **observed** z |
| `N_match_true_z_cksb` | (29,15,8,16) | same with k from `Z_TRUE` |
| `C_true_bKs` | (16,3,8) | on-grid matched detections / `truth_counts_bks`, summed within K; 0 where truth is 0; dead strata 0 |
| `C_true_bs` | (16,8) | the same pooled over z |
| `C_true_bk`, `C_true_bKs_obsz` | (16,15), (16,3,8) | z-resolved / observed-z variants |
| `M_true_sKcb` | (8,3,29,16) | migration conditional on detection, unit mass over c; **rows with no matches fall back to the adopted kernel row renormalised**, flagged by `M_had_mass_sKb` |
| `M_counts_sKcb` | (8,3,29,16) | the raw counts behind `M_true` (so noise is visible) |
| `E_true_cKsb` | (29,3,8,16) | `N_match_true_z` summed within K / `truth_counts_bks` summed within K |
| `E_true_cksb_fine` | (29,15,8,16) | the same at fine z (Poisson-noisy: typically 0–5 counts per cell) |
| `P6b_cks`, `hostless_cks`, `host_above_top_cks` | (29,15,8) | the classes with no term / the ORACLE FP pin |
| `N_det_all_bks_true_z`, `N_det_bks_true_z`, `N_det_bks_obs_z` | (16,15,8) | detections with / without the observed-grid restriction |
| `truth_counts_bks`, `truth_counts_bks_collar3300`, `counts_obs_cks`, `counts_obs_cks_collar3300`, `hostless_cks_collar3300`, `P6b_cks_collar3300` | | truth and data at both collars |
| grids + `provenance` | | `ntrue_edges`, `nhat_edges`, `zf_edges`, `snr_edges`, `zc_edges`, `kz_to_K`; provenance as a 0-d object array with catalogue/truth/BAL/molly paths, code commit, pack and census sha256, collar record |

**Noise.** `E_true_cKsb` has, for the reported bins, O(10²–10³) counts per (K,s,b) row and O(1–100)
per (c,K,s,b) cell; the fine-z `E_true_cksb_fine` has O(1–10) per cell. Use the coarse-K form for
anything quantitative; the fine form is diagnostic only.

**Consumer warning (load-bearing).** `C_true_*` already carries the on-grid factor `phi`, and
`M_true_sKcb` is unit-mass. **They compose exactly as a pair.** Substituting `M_true` for `Mg`
alone removes `phi` (mass 1 instead of `phi_ref`) and substituting `C_true` alone double-counts it.
`analyze_fold.py` therefore always swaps the pair `(A_true, M_true)` or holds the other's mass fixed.

---

## 4. Deliverable 3 — migration around the alternating reporting bins

`opfor_fig3_migration_<fam>.png` overlays the adopted skew-normal row against the measured row for
every b in [19.7,20.5) and [21.1,21.7), per K, for S/N 2–3 and S/N ≥ 7.
Full moment/leakage tables: `fold_forensics_<fam>.json:migration`.

Family-common findings (2LPT-0 / London-0 / Saclay-0 all show the same sign):

1. **The modelled kernel is too NARROW.** At S/N ≥ 7 the measured sd exceeds the modelled sd in
   almost every row: e.g. b=[20.1,20.3) K1 0.112 vs 0.133/0.128/0.124; b=[21.1,21.3) K1 0.091 vs
   0.110/0.117/0.107; b=[19.7,19.9) K2 0.140 vs 0.152/0.166/0.153. Typically 8–25 % too narrow at
   high S/N, 3–10 % at S/N 2–3.
2. **The modelled mean is too HIGH at high N.** `mean_model − mean_true` = +0.03…+0.08 dex for
   b=[21.3,21.5) and b=[21.5,21.7) at S/N ≥ 7 on all three families; the adopted covariate clamp
   `adopted_resp_fit_range[...,1] = 21.35` freezes the moment polynomials at 21.35 for every latent
   bin above it.
3. **The modelled skew is structurally ZERO above 21.5.** `resp_skew_ramp = (21.0, 0.5)` forces
   `skew *= 1 − clip((N−21.0)/0.5, 0, 1)`, so for b=[21.5,21.7) the model skew is exactly +0.000
   (see the table) while the measured skew is +0.3…+1.3.
4. **Up/down leakage is badly mis-apportioned at high N.** b=[21.5,21.7) K0 S/N≥7: model
   down/in/up = 0.066 / 0.624 / 0.310 vs measured 0.116 / 0.812 / 0.072 (2LPT-0); London-0
   0.116/0.789/0.095, Saclay-0 0.139/0.772/0.089. The model believes ~31 % of that bin's detections
   scatter **up** out of the bin; the truth is 7–10 %. The same pattern holds for b=[21.3,21.5)
   (model up 0.31–0.44 vs measured 0.15–0.26).
5. **A K1-specific mean offset of about −0.02 dex** around 20.1–20.5 at high S/N:
   `mean_model − mean_true` at b=[20.1,20.3) S/N≥7 is −0.002 / −0.025 / −0.002 (K0/K1/K2) on
   2LPT-0, −0.008 / −0.017 / −0.008 on London-0, −0.011 / −0.024 / −0.004 on Saclay-0. K1 is the
   only block with a systematic offset there.

---

## 5. Deliverable 4 — completeness truth vs calibration (K1 and low S/N)

`opfor_fig4_completeness_truth_vs_calibration.png`. Because the matched table was also binned with
**no** observed-grid restriction, the detection probability and the counting fraction separate
exactly: `C_true = C_det × phi`.

**(a) Does the calibrated C under-estimate completeness at S/N 2–3? YES — and by a lot in the
lowest reported bins.** `C_det(model)/C_det(true)`, truth-weighted, at S/N 2–3 vs S/N ≥ 7:

| b | 2LPT-0 s2 / s7 | London-0 s2 / s7 | Saclay-0 s2 / s7 |
|---|---|---|---|
| [19.5,19.7) | 1.054 / 0.954 | 0.997 / 0.944 | 0.964 / 0.944 |
| **[19.7,19.9)** | **0.866 / 0.940** | **0.799 / 0.935** | **0.778 / 0.927** |
| [19.9,20.1) | 1.063 / 0.999 | 0.996 / 1.001 | 0.994 / 1.001 |
| [20.1,20.3) | 0.931 / 0.996 | 0.893 / 1.000 | 0.888 / 1.000 |
| [20.3,20.5) | 0.959 / 0.995 | 0.945 / 1.000 | 0.937 / 0.996 |
| [20.5,20.7) | 0.982 / 1.002 | 0.973 / 1.001 | 0.966 / 0.999 |
| [20.7,20.9) | 0.950 / 0.997 | 0.936 / 0.999 | 0.933 / 0.998 |
| [20.9,21.1) | 0.995 / 0.997 | 0.974 / 0.997 | 0.981 / 1.000 |
| [21.1,21.3) | 0.969 / 1.003 | 0.972 / 0.997 | 0.983 / 1.002 |
| [21.3,21.5) | 0.978 / 0.996 | 0.973 / 0.999 | 0.962 / 1.004 |
| [21.5,21.7) | 0.986 / 0.998 | 0.941 / 0.998 | 0.949 / 0.996 |

At S/N ≥ 7 the calibrated completeness is accurate to ≤ 0.5 % for every reported bin above 19.9 on
**all three** families. At S/N 2–3 it is low by 1–11 % almost everywhere, and the sign **alternates
within each molly N-cell**: cell [20.0,20.3) gives +6.3 % at [19.9,20.1) and −6.9 % at [20.1,20.3)
on 2LPT-0; cell [19.5,20.0) gives +5.4 % at [19.5,19.7) and −13.4 % at [19.7,19.9). That is the
signature of a **12-cell molly N-grid read at 0.2-dex latent resolution**: `b_to_cell` assigns one
constant completeness to bins whose true completeness is still climbing steeply, and the cell
average is dominated by the (more numerous) low-N end.

**(b) Is the shortfall z-dependent?** Yes, mildly, and in the pad region strongly.
`C_det(model)/C_det(true)` per K at S/N 2–3 shows K0 systematically closer to 1 than K1/K2 in the
pad bins; in the reported window the per-K spread at fixed b is ≲ 2 % (full array
`fold_forensics_<fam>.json:completeness_vs_counting`). The z-shape surface `g(N,z)` is itself the
frozen 2LPT-0 one and carries the pad-cell z-trend that Fig. 4 shows failing at 19.0–19.8 in K0.

**(c) The counting fraction `phi` is exact where it matters and broken below 19.7.**
`phi(model)/phi(true)` = 1.000 ± 0.003 for every b ≥ [19.9,20.1) at every stratum. It is
1.11 / 1.13 (2LPT-0 s2 / s7) at [19.5,19.7), and 0.70…1.62 across strata at [19.0,19.5) — i.e. the
kernel's in-grid mass for latent bins at and below the observed floor is wrong by tens of percent.
This is where the `adopted_resp_fit_range` **lower** clamp (19.05–19.65 depending on cell) bites:
for latent centres 19.1 and 19.35 the moment polynomials are evaluated at the clamp, not at the bin.

---

## 6. The deterministic (no-sampler) inversion — the operator reproduces the posterior

The fold is **linear in f and does not mix fine-z cells**, so for each k the latent solution is a
16-parameter non-negative Poisson MLE, obtainable by EM. `analyze_fold.py` solves it exactly
(4000 EM iterations from `f_true`; **no population prior**, which is the undamped limit of the
committed 2-D RW model) and reduces it with the committed threshold weights.

| ≥20.0 bias [%] | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| ORACLE posterior (median) | +0.99 | +1.19 | +1.42 |
| operator inversion, frozen | **+1.37** | **+1.83** | **+2.03** |
| ≥20.3 bias [%] | | | |
| ORACLE posterior (median) | +2.95 | +2.17 | +3.29 |
| operator inversion, frozen | **+2.82** | **+2.57** | **+3.60** |

Per block (≥20.3), ORACLE vs inversion: 2LPT-0 [+0.60, +7.41, +1.45] vs [−0.95, +9.64, +1.33];
London-0 [+0.17, +5.31, +2.92] vs [+0.16, +6.13, +3.71]; Saclay-0 [+1.83, +5.16, +4.95] vs
[+1.88, +6.04, +4.86]. The reporting-bin zigzag is reproduced bin-by-bin
(`opfor_fig5_zigzag_latent_before_after.png`): e.g. 2LPT-0 [21.1,21.3) +16.1 (posterior) vs +17.6
(inversion), [21.5,21.7) −7.1 vs −8.8, [19.9,20.1) +12.2 vs +10.4; Saclay-0 [20.3,20.5) +10.0 vs
+10.6, [20.7,20.9) +8.5 vs +8.8, [20.9,21.1) −0.9 vs −1.8.

**Self-test.** Folding the *empirical* operator against the *matched* counts returns the truth to
−0.00 / −0.13 / −0.03 % (≥20.0) and +0.05 / −0.07 / +0.23 % (≥20.3). That residual is the whole
budget of z-migration + stratum allocation, and it bounds the machinery's own error.

**Exact component replacement on the ORACLE data (before → after):**

| arm | 2LPT-0 ≥20.0 / ≥20.3 | London-0 | Saclay-0 |
|---|---|---|---|
| frozen | +1.37 / +2.82 | +1.83 / +2.57 | +2.03 / +3.60 |
| − P6b (add the missing class) | +1.15 / +2.62 | +1.67 / +2.45 | +1.90 / +3.51 |
| + measured **completeness** | +0.20 / +1.34 | −0.23 / +0.42 | −0.37 / +1.13 |
| + measured **migration** | +1.14 / +1.31 | +2.23 / +2.27 | +2.45 / +2.67 |
| + **both** | −0.15 / +0.05 | −0.21 / −0.02 | −0.05 / +0.17 |
| + both + P6b | −0.00 / +0.05 | −0.13 / −0.07 | −0.03 / +0.23 |
| response-z **gather** fine-z instead of coarse | +1.45 / +2.92 | +1.88 / +2.60 | +2.08 / +3.64 |

Per-block (≥20.3) K1: frozen +9.64 / +6.13 / +6.04 → measured completeness +7.29 / +3.49 / +2.94 →
measured migration **+2.36 / +3.30 / +3.20** → both +0.22 / +0.15 / +0.24.

`opfor_fig5` also shows that **migration alone flattens the zigzag** (2LPT-0 [21.1,21.3)
+17.6 → +1.5; [21.5,21.7) −8.8 → −2.8; Saclay-0 ±10–16 % → ±3 %) while **completeness alone leaves
it intact** (2LPT-0 [21.1,21.3) +16.8).

**ORACLE posterior `psi_c`.** Mean over cells −0.0073 (2LPT-0), 0.034 prior-sd units; folding with
it changes `mu_TP/matched` from 1.0214 to 1.0232 and the S/N 2–3 ratio from 0.9741 to 0.9809. The
sampled completeness offsets are **not** absorbing the completeness error.

---

## 7. Two accounting defects found in the run configuration (not in the fold maths)

**(i) Collar mismatch in the scan packs — the reported bias is understated by ~0.45–0.49 pp.**
`truth_counts` at 3300 km/s / at 3000 km/s = 0.99554 / 0.99557 / 0.99554 (flat in K: 0.9954–0.9957;
flat in b: 0.9925–0.9973 over the reported bins). The estimand actually used is therefore
`truth_ge20.0 = 0.086604` where the collar-matched value is `0.086191` (2LPT-0). Correcting it:

| | ≥20.0 | ≥20.3 |
|---|---|---|
| 2LPT-0 | +0.99 → **+1.48** | +2.95 → **+3.45** |
| London-0 | +1.19 → **+1.65** | +2.17 → **+2.63** |
| Saclay-0 | +1.42 → **+1.88** | +3.29 → **+3.71** |

This is a **numerator/denominator-on-different-supports** instance (the project's recurring bug
class, `feedback_one_sided_support_bug_class`): `truth_counts` on the 3000 km/s selection divided by
`dX` on the 3300 km/s selection. It makes the defect *larger*, not smaller, and it is uniform in N
and z, so it does not explain the K1 concentration or the zigzag.

**(ii) The ORACLE FP pin is also on the wrong collar.** `run_ladder.py:68` pins `mu_FP` to the
census `hostless` array, which is built at 3000 km/s, against `counts` at 3300 km/s. Measured
excess: 16 / 12 / 18 detections (13 860 vs 13 844; 9598 vs 9586; 10 592 vs 10 574) — i.e. the FP arm
is over-pinned by ≈ 0.1 %, pushing the TP arm ≈ 0.02 % low. Negligible, but it should be fixed
together with (i).

---

## 8. Deliverable 5 — the causal map

### (i) The K1 (z 2.5–3.0) concentration

* **Can account for it: the RESPONSE / migration operator.** Replacing `Mg` by the measured
  migration collapses the ≥20.3 K1 bias from +9.6/+6.1/+6.0 to +2.4/+3.3/+3.2 on the three families
  — the single largest lever on K1, and the only component whose replacement moves K1 more than K0
  or K2. The mechanism is visible in the moments: a K1-only mean offset of ≈ −0.02 dex at
  20.1–20.5, high S/N, on all three families (§4.5).
* **Partly: COMPLETENESS.** Replacing `C·g·phi` by the measured completeness takes ≥20.3 K1 from
  +9.6/+6.1/+6.0 to +7.3/+3.5/+2.9. It carries roughly a third of K1 (more on London-0/Saclay-0).
* **Partly: the class with NO term (P6b).** P6b/matched rises 3.1 → 5.4 → 7.2 % from K0 to K2
  (family-common), so the unmodelled sub-basis class pushes the required TP level up with z. At the
  estimand level it is only +0.1…+0.2 pp, because the basis pad [19.5,19.7) absorbs nearly all of it
  (the inversion's [19.5,19.7) bin moves −8.8 → +0.1 when P6b is added).
* **CANNOT account for it: the latent basis / population representation** — the `truth_stratified`
  swap and the self-test show the latent basis reproduces the matched data to ≲ 0.2 % once the
  operators are right. **CANNOT: the response-z GATHER.** `build_cc_tensors` gathers the response
  z-cell through the coarse block (`K_to_zresp[kz_to_K[k]]`); with `resp_z_edges = [0, 2.56, 2.96,
  ∞]` exactly one fine cell moves under a direct fine-z gather (k=5, z [2.5,2.6), centre 2.55 < 2.56)
  and the effect is **+0.1…+0.3 pp in the wrong direction** — tested, ruled out.
  **CANNOT: z-migration** (the fold has no z term): observed-z vs true-z detection totals agree to
  ≤ 0.08 % per block and ≤ 0.3 % per fine-z cell. **CANNOT: the collar mismatch** (flat in z).
  **CANNOT: `psi_c`.**

### (ii) The 0.2-dex zigzag

* **The RESPONSE / migration operator accounts for it, essentially completely.** It is a
  *deterministic* property of the operator, not a sampler or prior artefact: an unpenalised Poisson
  MLE with no population prior reproduces the posterior zigzag bin-by-bin (§6), and replacing the
  migration rows flattens it on all three families while replacing completeness does not.
* Named mechanisms inside the response representation, all family-common and all present on 2LPT-0
  itself (so they are **representation**, not transfer, error): a kernel 8–25 % too narrow; a mean
  frozen by the `adopted_resp_fit_range` upper clamp at 21.35; a skew forced to exactly zero above
  21.5 by `resp_skew_ramp = (21.0, 0.5)`; and a consequent 3–4× over-statement of upward leakage out
  of [21.3,21.7). These produce the +16/+11/+9 % excess at [21.1,21.3) and the −7/−9/−13 % deficit at
  [21.5,21.7).
* **Completeness contributes a second, smaller zigzag** through the 12-cell molly N-grid read at
  0.2-dex latent resolution (alternating ±6 % within a cell), but only at S/N 2–3, which is 21 % of
  the matched sample — hence its small effect on the reporting bins.
* **The latent basis cannot account for it**: the 0.2-dex reporting bins are exact unions of single
  latent bins, and with both operators replaced every reported bin of 2LPT-0 and Saclay-0 lands
  within ±2 %.

### (iii) The S/N 2–3 under-prediction (4–6 %)

* **COMPLETENESS accounts for it.** With truth pinned the operator already under-predicts S/N 2–3 by
  2.6 / 6.4 / 7.3 % while over-predicting S/N ≥ 7 by 4.4 / 4.4 / 4.8 % (§2.2). The separation in §5
  localises this entirely in the **detection** probability `C_det` (the molly matrix), not in `phi`,
  which is exact for b ≥ 19.9. A single f cannot satisfy both ends of a 7–12 % stratum gradient, so
  the fit compromises and leaves S/N 2–3 under-predicted.
* **P6b adds to it in the same direction**: P6b is concentrated at low S/N (25 % of it in the S/N
  2–3 stratum), so the *required* TP level at S/N 2–3 is 1.08 / 1.12 / 1.13 × the truth fold while at
  S/N ≥ 7 it is 0.98 / 0.97 / 0.97 ×.
* **CANNOT account for it: the fold's stratum ALLOCATION of the truth.** The fold uses
  `f[b,k]·dX[k,s]`, i.e. it re-allocates truth across strata in proportion to pathlength rather than
  using `truth_counts_bks`. Measured: allocated/measured truth per stratum = 0.996–1.010, and the
  completeness-weighted error per latent bin is ≤ 0.2 % (≤ 0.4 % in the pad). Ruled out.
* **CANNOT: the response**, whose replacement leaves the S/N pattern unchanged
  (`component_swaps.migration_only.by_snr` ≡ `frozen.by_snr` to 4 decimals).

### Family-common vs family-specific

**Family-common** (same sign and comparable size on all three; all of these are frozen 2LPT-0
calibration objects, so 2LPT-0 shows them as representation error and the other two as
representation + transfer error):
the +2.2…+2.7 % level the TP arm is forced to; the S/N gradient (low at 2–3, high at ≥7); the
over-prediction of the [19.0,19.5) pad by 17–31 % and of [19.5,19.7) by 6–10 %; the under-prediction
of [19.7,19.9) by 8–11 %; the too-narrow kernel; the up-shifted, skew-zeroed high-N kernel and its
3–4× excess upward leakage; the K1-only −0.02 dex mean offset; the rise of P6b with z; the 0.45 pp
collar understatement.

**Family-specific**: the *detailed* zigzag pattern (2LPT-0 peaks at [19.9,20.1) and [21.1,21.3);
Saclay-0 has a strong +10 %/−4 %/+8.5 % alternation through [20.3,21.1) that 2LPT-0 does not);
the size of the S/N 2–3 completeness shortfall (2.6 % on 2LPT-0, its own calibration, vs 6.4–7.3 %
on the transfer families); the K2 behaviour (2LPT-0 ≈ 0, London-0 +2.1 %, Saclay-0 +3.3 % at ≥20.0).
The inversion reproduces each family's own pattern, so these differences are genuine operator
transfer error, not noise.

---

## 9. Suspected defects — with evidence

1. **Collar mismatch inside `scanpack_*_b300.npz`** (`build_scan_packs.py` rebuilds `counts`/`dX`/
   `fp_E_alloc` at 3300 km/s and copies `truth_counts*` byte-identically at 3000 km/s). Evidence:
   `scanpack.truth_counts_bks` byte-identical to the adopted pack's; rebuilt truth at 3300 is
   0.99554× the 3000 one. Consequence: every reported mock recovery bias is understated by
   0.45–0.49 pp. **This is the sixth-plus instance of the one-sided-support class.**
2. **The ORACLE FP pin uses a collar-3000 census against collar-3300 counts** (`run_ladder.py:68`):
   16 / 12 / 18 counts of over-pinning.
3. **The `adopted_resp_fit_range` covariate clamp is load-bearing outside [≈19.6, 21.35]**, where
   4 of the 16 latent bins live. Evidence: `phi(model)/phi(true)` = 0.70…1.62 for b < 19.7 and
   1.000 ± 0.003 for b ≥ 19.9; `mean_model − mean_true` = +0.03…+0.08 dex for b ≥ 21.3. The clamp is
   a documented guard (finding D2), not a coding error — but the bins outside the calibrated range
   carry an *unvalidated* response, and that is where the two largest per-b efficiency errors sit.
4. **`resp_skew_ramp = (21.0, 0.5)` sets the modelled skew to exactly 0 for every latent bin above
   21.5**, while the measured skew there is +0.3…+1.3. Directly visible in the moment table
   (`skew_model = +0.000`).
5. **`b_to_cell` reads a 12-cell molly N-grid at 0.2-dex latent resolution**, producing an
   alternating ±6 % completeness error within each molly cell at S/N 2–3.
6. **The latent bin [19.9,20.1) straddles the molly edge 20.0** (centre 20.0 → cell [20.0,20.3)),
   the only reported bin that does; it is also the one with the largest S/N 2–3 completeness error of
   the wrong sign (+6.3 %).

**No off-by-one or digitize-convention error was found.** `bin_index` matches `extract_pack._idx`
on the real arrays; `kz_to_K`, `s_to_sresp`, `K_to_zresp`, `b_to_cell` were each re-derived and
agree with `build_consts`; the response-z gather was tested by explicit replacement and is *not* the
driver; the truth histogram reproduces the pack's `truth_counts_bks` bit-exactly; `M_true` rows sum
to 1 within 3e-16; the empirical operator inverts back to the truth to ≤ 0.23 %.

---

## 10. What this implies for the ladder (diagnosis only — no model is proposed here)

The diagnosis points at exactly two blocks, in this order of leverage:

* **Response/migration** — carries the zigzag and the largest single share of the K1 excess. The
  named sub-causes are low-dimensional and interpretable: kernel width, the high-N covariate clamp,
  and the skew ramp.
* **Completeness (the molly `C`, its N-resolution and its S/N gradient)** — carries the level
  (≥20.0) and the S/N 2–3 shortfall.
* **Accounting**: the P6b class has no term at all (worth ~0.1–0.2 pp on the estimand but ~3 % of
  the catalogue, and it is what the pad bin currently absorbs), and the two collar mismatches must
  be repaired before any bias number is quoted.

Anything else — latent basis, population prior, stratum allocation, z-migration, `psi_c`, the
response-z gather — is measured here and cannot carry the residual.

---

## 11. Provenance

* worktree `/home/mfho/wt_abs_diag_2026-09`, branch `absorber-diag-2026-09`, **not committed**;
* packs: `adopted_packs_v2p2_20260821/modelA_pack_<fam>_bw0p2_pad19p0_molly172_v2.npz`
  (operator work) and `fp_ladder_2026-09-12/packs/scanpack_<fam>_b300.npz` (collar costing);
  census `fp_ladder_2026-09-12/census/fp_census_<fam>.npz`; ORACLE runs
  `fp_ladder_2026-09-12/diag_oracle/RUN_ORACLE_<fam>_s20260811*.{json,npz}`;
* sha256 of every input is stamped in each `empirical_ops_<fam>.provenance.json` and in
  `fold_forensics_<fam>.json:provenance`; `SHA256SUMS` accompanies the NPZs in both locations;
* outputs: `validation/absorber_diag/{empirical_ops,fold_forensics}_<fam>.{npz,json}` and
  `/scratch/cavestru_root/cavestru0/mfho/absorber_diag_2026-09-13/ops/`;
* figures: `.../figures/2026-09-13_absorber_diag/opfor_fig1…fig6*.png`;
* tests: `tests/test_absorber_diag.py`, 18 passed, mutation-checked (a `searchsorted(side=...)` flip,
  a threshold-weight mis-definition, a leakage edge-sign flip and a hidden variance floor are each
  caught).

---

## 12. Twelve-line summary

1. With FP pinned to mock truth and `psi_c = 0`, folding each mock's own truth through the frozen absorber-side operator OVER-predicts the matched in-basis detections by +2.14 / +1.22 / +0.87 % (2LPT-0 / London-0 / Saclay-0).
2. The same fold under-predicts the S/N 2–3 stratum by 2.6 / 6.4 / 7.3 % and over-predicts S/N ≥ 7 by 4.4 / 4.4 / 4.8 % — a 7–12 % stratum gradient one global f cannot satisfy.
3. A deterministic, unpenalised Poisson MLE inversion of the frozen operator (no sampler, no prior) reproduces the ORACLE posterior: ≥20.0 +1.37/+1.83/+2.03 % vs +0.99/+1.19/+1.42 %, ≥20.3 +2.82/+2.57/+3.60 % vs +2.95/+2.17/+3.29 %, including the K1 concentration and the reporting-bin zigzag bin-by-bin.
4. Therefore the bias, the K1 concentration and the zigzag are deterministic properties of the absorber-side operator, not sampler or prior artefacts.
5. Replacing the adopted skew-normal migration by the measured one flattens the zigzag on all three families (2LPT-0 [21.1,21.3) +17.6 → +1.5 %, [21.5,21.7) −8.8 → −2.8 %) and cuts the ≥20.3 K1 excess from +9.6/+6.1/+6.0 to +2.4/+3.3/+3.2 %.
6. Replacing the calibrated completeness by the measured one removes the level and the whole S/N gradient (≥20.0 +1.37/+1.83/+2.03 → +0.20/−0.23/−0.37 %) but leaves the zigzag intact.
7. Replacing both returns every estimand to |bias| ≤ 0.21 % and every reported 0.2-dex bin to ≲ 2 % on 2LPT-0 and Saclay-0; the empirical operator folded against the matched counts self-tests to ≤ 0.23 %.
8. Named response defects, family-common and present on 2LPT-0 itself: the kernel is 8–25 % too narrow; the `adopted_resp_fit_range` clamp at 21.35 leaves the mean +0.03…+0.08 dex high above 21.3; `resp_skew_ramp=(21.0,0.5)` forces skew to exactly 0 above 21.5; upward leakage out of [21.5,21.7) is modelled at 31 % vs 7–10 % measured.
9. Named completeness defects: at S/N ≥ 7 the frozen molly C is accurate to ≤ 0.5 % above 19.9 on all three families, but at S/N 2–3 it is 1–11 % low, with an alternating ±6 % error inside each 12-cell molly N-bin read at 0.2-dex latent resolution; the in-grid fraction phi is exact (±0.3 %) above 19.9 and wrong by 11–62 % below 19.7.
10. The class the fold has no term for — matched hosts in [17.2,19.0) — is 3.6/3.0/3.1 % of the catalogue, rises with z (3.1→5.4→7.2 % of the matched set from K0 to K2), sits mostly at low S/N, and is worth only +0.1…+0.2 pp on the estimand because the [19.5,19.7) pad bin absorbs it.
11. Two accounting defects found: the collar-scan packs divide a 3000 km/s `truth_counts` by a 3300 km/s `dX` (a one-sided-support instance), understating every reported bias by 0.45–0.49 pp (2LPT-0 ≥20.3 +2.95 → +3.45 %); and the ORACLE FP pin uses the 3000 km/s census against 3300 km/s counts (16/12/18 counts).
12. Explicitly ruled out with measurements: the latent basis / population representation, the fold's pathlength stratum allocation of truth (≤ 0.2 %), z-migration (≤ 0.08 % per block), the posterior `psi_c` (0.03 prior sd), and the coarse response-z gather (tested by direct fine-z replacement: +0.1…+0.3 pp, wrong sign); no off-by-one, digitize-convention or gather bug was found.
