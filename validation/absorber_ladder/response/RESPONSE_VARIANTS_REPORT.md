# R1 — response fixed-calibration variants: fits, cross-validation, and the diagnosed cause of the width defect

**Date 2026-09-13. VALIDATION-ONLY. No sampler was run. Nothing under `CDDF_analysis/` was
modified, and nothing was committed.** Worktree `/home/mfho/wt_abs_diag_2026-09`, branch
`absorber-diag-2026-09` @ `daa93af6`. Governing ruling:
`governance/PI_RULING_2026-09-13b_ABSORBER_LADDER_GO_CUT_FEEDBACK.md` §4 (GO the first ladder),
§5 (R1 as explicit nested variants), §9 (opening rules), §15 (calibration-side predictive checks
mandatory), §19 (return early on an unexpectedly high-dimensional calibration representation).

**No mock-closure number was read, computed or used anywhere in this work.** Every model-selection
decision is made on held-out 2LPT-0 calibration events alone.

Products: `/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/response/`
(`SHA256SUMS` there). Figures:
`/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_absorber_ladder/response/`.

---

## 0. Files, environments, and the gates they passed

| file | env | what |
|---|---|---|
| `validation/absorber_ladder/response/extract_calib_events.py` | `gpdla` | rebuilds the 2LPT-0 natural-pair matched CALIBRATION event table |
| `validation/absorber_ladder/response/respfit.py` | `gpdla` | sub-bin moment estimators, moment surfaces, the bin-marginal quadrature repair, CV helpers |
| `validation/absorber_ladder/response/opmetrics.py` | `gpdla` | operator-side row metrics + the `Mg` gather |
| `validation/absorber_ladder/response/r1d_empirical.py` | `gpdla` | the alternative (smoothed-empirical) representation |
| `validation/absorber_ladder/response/build_variants.py` | `gpdla` | the driver: gates, CV, model selection, products |
| `validation/absorber_ladder/response/make_figures.py` | `gpdla` | the three figures |
| `tests/test_response_variants.py` | either | unit tests for every fit/CV helper |

### The calibration events

`forward_response_2lpt0.npz` stores only coefficients and the smoothed-empirical density — **the raw
pair table is not on disk**, so it was **rebuilt** through the committed machinery. The loader block
of `extract_calib_events.py` is a verbatim copy of
`CDDF_analysis/hbi_mcmc/build_kernel_fit_ensemble.py` (`track_c_tf_loa._C0_*` catalogue,
`load_and_cut_catalog(truth_nhi_floor=molly[0], host_truth_floor=19.0)`, op cut
`S2N_RED > snr_min & P_DLA > p_dla_min & good_mask`, `host_col = NHI_TILT_HOST`,
`xhat_floor = 19.5`), which is the same selection `adopted_response`'s own recovered builder
(`CDDF_analysis/hbi/adopted_response/stage1b_events_full.py`) applies.

**n = 73,845 matched detections on 65,011 unique sightlines.** (The "66,481 events" in the
`adopted_response/v1.1` provenance string is the G-B *audit* population, a different count; the fit
population is the 73,845 above — as the gate below proves.)

### Gate 1 — the event table reproduces the frozen response BIT-FOR-BIT

Refitting the estimator of record on these events reproduces `adopted_response_v1p1.npz`:

```
mu_coef 0.0   sig_coef 0.0   skew_coef 0.0   fit_rng 0.0      (max |difference|, exactly zero)
```

This is the strongest available proof that the rebuilt table IS the calibration set of record and
that the re-implementation in `respfit.py` is exact.

### Gate 2 — R0 reproduces the deployed bin masses bit-for-bit

`count_conserving_fold.surface_masses` on the refit R0 coefficients vs on the pack's stored
`adopted_resp_*` arrays: `max |Δ mass| = 0.0`, `max |Δ φ| = 0.0`.

### Gate 3 — `adopted_phi_ref` provenance (the count-conservation rule)

`adopted_phi_ref` equals the **deployed** (`resp_*`) kernel's in-grid fraction to `0.0`, **not** the
adopted surfaces' (which differ by up to 0.349). That is the ratified rule — `C_molly` was
calibrated jointly with the deployed kernel — so every variant inherits the *same* counting
probability and can only **redistribute** detections.

### Gate 4 — consumability

`Mg_R0_2lpt0.npz["Mg"]` is **bit-identical** (`max|Δ| = 0.0`) to
`cc_posterior_validation.build_cc_tensors(load_pack(scanpack_2lpt0_b300.npz))[1]`, shape
`(8, 15, 29, 16)` — the exact object `validation/fp_ladder/run_ladder.py` passes as `Mg_fixed`.
Every variant's `Mg` has row sums equal to the pack's `adopted_phi_ref` to ≤ 4.4e-16.

---

## 1. The variant ladder

All of R0–R1c are the SAME functional object the production fold consumes: moment polynomials
`(mu, sig, skew)` of shape `(SR=3, ZR=3, D)` in `u = N_true − resp_N_ref`, plus a covariate clamp
range `fit_rng (3,3,2)`, evaluated by the committed `surface_masses`. All are **fixed before any HBI
run**; none carries an HBI degree of freedom.

| variant | representation | fitted calibration coefficients | motivating defect |
|---|---|---|---|
| **R0** (baseline) | per-cell deg-2 + ONE shared cubic; untruncated skew-normal ML on a fixed 0.1-dex sub-bin grid over [19.0, 21.4], min 50 events; `fit_rng` = min/max fitted sub-bin centre; skew ramp (21.0, 0.5) | **84** moment coefficients (3 moments × [9 cells × 3 + 1 shared]) **+ 18** data-determined fit-range numbers = **102** | — (the frozen `adopted_response/v1.1`) |
| **R1a** | as R0, but the sub-bin grid runs over the FULL latent support [19.0, 22.4], 0.1-dex bins merged upward to ≥50 events (cap 0.3 dex, floor 25), row centre = count-weighted mean N; `fit_rng` = (19.0, 22.4) — **no clamp anywhere inside the latent support** | **84** (+0 range numbers) | the clamp `adopted_resp_fit_range[...,1] = 21.35` freezes every moment polynomial for the 4 latent bins above it (forensics §9.3); mean +0.03…+0.08 dex too high at b ≥ 21.3 |
| **R1b** | R1a **+ the skew ramp removed** (`skew_ramp` declared as a no-op; skew is fitted wherever measured, and above the last fitted sub-bin centre it is the same polynomial extrapolation as the other two moments — documented in §4) | **84** | `resp_skew_ramp = (21.0, 0.5)` forces the modelled skew to EXACTLY 0 above 21.5 while the measured skew there is +0.3…+1.3 (forensics §9.4) |
| **R1c** | R1b **+ the width representation repaired**: the object is re-expressed so that the MIDPOINT evaluation `surface_masses` performs returns the bin-**marginalised** first three moments (§3). Deterministic quadrature of the same fitted conditional surfaces; no new data fit. Polynomial degree for the re-expression and the sub-bin estimator chosen by CV | **108** (3 moments × 9 cells × 4) | kernel rows 8–25 % too narrow at high S/N (forensics §9.1) |
| **R1d** | ALTERNATIVE representation (opened only on R1c's residual, §6): smoothed **empirical** masses `p[sr, zr, offset, b]`, ridge-penalised low-order polynomial smoothing along N | **702 nominal / 531 effective** | R1c's residual row-skew and down/in split, which no 3-moment skew-normal can carry |

Cross-validation protocol (PI §15): **2-fold over SIGHTLINES — TARGETID parity**, fit on one half
(36,813 events) and score the held-out half (37,032), both directions pooled. No TARGETID appears in
both folds (unit-tested). Scores are computed on the SAME discretised rows the fold consumes
(`surface_masses` masses on the pack's `nhat_edges`), per `(b, s, K)` block with ≥ 40 held-out
detections.

---

## 2. Held-out calibration-side predictive performance

Count-weighted MEAN residuals, model − empirical (width as a ratio − 1), pooled over all strata:

| variant | n_coef | Δmean (dex) | width | Δskew | Δdown | Δin | Δup | loglik/event |
|---|---|---|---|---|---|---|---|---|
| R0  | 102 | −0.0047 | **−0.0617** | −0.174 | +0.0006 | −0.0002 | −0.0004 | 0.4934 |
| R1a | 84  | −0.0013 | −0.0508 | −0.191 | +0.0046 | −0.0074 | +0.0028 | 0.4871 |
| R1b | 84  | −0.0013 | −0.0508 | −0.187 | +0.0047 | −0.0073 | +0.0026 | 0.4775 |
| **R1c** | **108** | **−0.0001** | **−0.0058** | −0.249 | +0.0174 | −0.0317 | +0.0143 | 0.4890 |
| R1d | 702 (eff 531) | −0.0006 | +0.0583 | +0.096 | +0.0247 | −0.0269 | +0.0022 | n/a |

Held-out RMS (per-block, dominated by held-out counting noise):

| variant | Δmean | width | Δskew | Δdown | Δin | Δup |
|---|---|---|---|---|---|---|
| R0  | 0.0330 | 0.1631 | 0.560 | 0.0453 | 0.0573 | 0.0566 |
| R1a | 0.0319 | 0.1422 | 0.566 | 0.0530 | 0.0635 | 0.0691 |
| R1b | 0.0319 | 0.1422 | 0.557 | 0.0532 | 0.0636 | 0.0690 |
| R1c | 0.0320 | 0.1400 | 0.577 | 0.0575 | 0.0683 | 0.0673 |
| R1d | 0.0461 | 0.1560 | 0.577 | 0.0954 | 0.0674 | 0.0950 |

### by true-N bin (held-out width residual)

| bin | R0 | R1a | R1b | R1c | R1d |
|---|---|---|---|---|---|
| [19.0,19.2) | −0.387 | −0.223 | −0.223 | −0.221 | +0.050 |
| [19.2,19.5) | −0.189 | −0.098 | −0.098 | −0.091 | +0.039 |
| [19.5,19.7) | −0.056 | −0.045 | −0.045 | −0.025 | −0.001 |
| [19.7,19.9) | −0.041 | −0.044 | −0.044 | −0.005 | +0.019 |
| [19.9,20.1) | −0.031 | −0.035 | −0.035 | +0.012 | +0.082 |
| [20.1,20.3) | −0.062 | −0.064 | −0.064 | −0.014 | +0.058 |
| [20.3,20.5) | −0.042 | −0.041 | −0.041 | +0.014 | +0.083 |
| [20.5,20.7) | −0.033 | −0.030 | −0.030 | +0.032 | +0.097 |
| [20.7,20.9) | −0.041 | −0.035 | −0.035 | +0.030 | +0.089 |
| [20.9,21.1) | −0.039 | −0.031 | −0.031 | +0.040 | +0.103 |
| [21.1,21.3) | −0.082 | −0.077 | −0.077 | −0.003 | +0.064 |
| [21.3,21.5) | −0.040 | −0.046 | −0.046 | +0.039 | +0.132 |
| [21.5,21.7) | −0.229 | −0.204 | −0.204 | −0.137 | −0.110 |

(Bins above 21.7 carry < 40 held-out detections in every block and are not scored.)

### by true-N bin (held-out mean residual, dex)

| bin | R0 | R1a/R1b | R1c | R1d |
|---|---|---|---|---|
| [19.0,19.2) | −0.085 | −0.046 | −0.045 | +0.018 |
| [19.2,19.5) | −0.032 | −0.008 | −0.006 | +0.015 |
| [20.1,20.3) | −0.001 | −0.002 | −0.002 | +0.027 |
| [20.9,21.1) | −0.006 | −0.022 | −0.022 | −0.010 |
| [21.1,21.3) | −0.006 | −0.023 | −0.023 | −0.013 |
| **[21.3,21.5)** | **+0.018** | **+0.002** | **+0.002** | +0.015 |
| [21.5,21.7) | +0.067 | +0.058 | +0.058 | +0.073 |

### by S/N stratum (held-out width residual)

| variant | S/N 2–3 | 3–4 | 4–5 | 5–6 | 6–7 | ≥7 |
|---|---|---|---|---|---|---|
| R0  | −0.089 | +0.042 | −0.094 | −0.014 | +0.053 | **−0.126** |
| R1a/R1b | −0.078 | +0.059 | −0.089 | −0.010 | +0.058 | −0.113 |
| R1c | −0.055 | +0.083 | −0.050 | +0.032 | +0.103 | **−0.039** |
| R1d | −0.016 | +0.126 | +0.012 | +0.102 | +0.168 | +0.053 |

### by coarse z cell (held-out width residual, mean residual)

| variant | K0 (2.0–2.5) | K1 (2.5–3.0) | K2 (3.0–3.5) |
|---|---|---|---|
| R0  | −0.065, −0.004 | −0.064, −0.010 | −0.047, +0.005 |
| R1a/R1b | −0.056, −0.001 | −0.052, −0.007 | −0.030, +0.009 |
| R1c | −0.010, +0.000 | −0.006, −0.006 | +0.010, +0.011 |
| R1d | +0.062, +0.000 | +0.053, −0.007 | +0.056, +0.011 |

### boundaries (PI §15)

| variant | b < 19.7: Δmean / width / Δskew / Δup | b ≥ 21.3: Δmean / width / Δskew / Δdown / Δin / Δup |
|---|---|---|
| R0  | −0.020 / −0.128 / −0.193 / +0.000 | **+0.024** / −0.062 / −0.126 / −0.029 / −0.046 / **+0.074** |
| R1a | +0.003 / −0.077 / −0.281 / +0.028 | **+0.009** / −0.065 / −0.142 / +0.012 / −0.076 / +0.065 |
| R1b | +0.003 / −0.077 / −0.281 / +0.028 | +0.009 / −0.065 / **−0.021** / +0.015 / −0.074 / +0.059 |
| R1c | +0.006 / **−0.063** / −0.297 / +0.036 | +0.009 / **+0.019** / −0.066 / +0.030 / **−0.105** / +0.074 |
| R1d | +0.016 / +0.018 / **−0.008** / +0.023 | +0.021 / +0.104 / −0.129 / −0.011 / −0.051 / +0.062 |

### in-sample vs held-out, and against the forensics' empirical `M_true`

The in-sample and held-out aggregates agree to **≤ 0.005 in every entry for R0–R1c** (e.g. R1c width
−0.0049 in sample vs −0.0058 held out; mean −0.0011 vs −0.0001) — the fit population is 73,845
events over 65,011 sightlines, so the two-fold split costs essentially nothing and there is **no
evidence of overfitting for R0–R1c**. R1d is the only variant with a visible in-sample/held-out gap
(row skew +0.072 in sample vs +0.096 held out; width +0.054 vs +0.058), consistent with its much
larger effective dimension. Against `validation/absorber_diag/empirical_ops_<fam>.npz`
(`M_true_sKcb`), in sample, count-weighted overall width residual:

| variant | 2LPT-0 | London-0 | Saclay-0 |
|---|---|---|---|
| R0  | −0.041 | −0.016 | +0.008 |
| R1a/R1b | −0.034 | −0.009 | +0.016 |
| R1c | +0.013 | +0.039 | +0.064 |
| R1d | +0.073 | +0.100 | +0.127 |

The **family spread is ±2.5 %** and is not removed by any variant — a fixed calibration object fitted
on 2LPT-0 transports to London-0 and Saclay-0 with a residual width offset of that size. R1c is
closest to zero on 2LPT-0 (its calibration family) and runs mildly wide on Saclay-0; R1d is wide on
all three.

---

## 3. THE IDENTIFIED CAUSE OF THE WIDTH DEFECT

**It is not the σ estimator. It is a midpoint-quadrature error in how the fixed response is
deployed.**

The fold needs `M[c, b, s, K] = P(ĉ | a system in latent bin b, block (s,K))`, i.e. the row
**marginalised** over the joint within-block distribution of `(N_true, S/N, Z_QSO)`.
`count_conserving_fold.surface_masses` instead evaluates the response at a **single point** —
`N = centre(b)`, one `(SR, ZR)` response cell per `(s, K)` block. The dropped variance is exactly

```
sd_row^2  =  E[ sigma^2(N, cell) ]  +  Var[ N + mu(N, cell) ]      (over the block's own covariates)
             \_____ dropped by using one cell _____/   \__ dropped by evaluating at the bin centre __/
```

Measured on the calibration events themselves (`respvar_fig3_width_cause.png`; K1 blocks):

| b | S/N | n | measured sd | R0 model (midpoint) | `E[σ²]^½` | `(E[σ²]+Var[N+μ])^½` | model/measured |
|---|---|---|---|---|---|---|---|
| [19.9,20.1) | ≥7 | 988 | 0.1340 | 0.1155 | 0.1224 | **0.1331** | 0.862 |
| [20.1,20.3) | ≥7 | 867 | 0.1288 | 0.1079 | 0.1146 | **0.1281** | 0.838 |
| [21.1,21.3) | ≥7 | 178 | 0.1050 | 0.0862 | 0.0893 | **0.1082** | 0.821 |
| [21.3,21.5) | ≥7 | 103 | 0.1054 | 0.0860 | 0.0884 | **0.1047** | 0.816 |
| [19.9,20.1) | 2–3 | 772 | 0.2174 | 0.2106 | 0.2147 | **0.2206** | 0.969 |
| [20.1,20.3) | 2–3 | 829 | 0.2195 | 0.2047 | 0.2060 | **0.2127** | 0.933 |
| [21.1,21.3) | 2–3 | 181 | 0.1631 | 0.1479 | 0.1468 | **0.1582** | 0.907 |

The two-term decomposition reproduces the measured row width to ≈ 1 % in every well-populated block,
while the midpoint value is 3–18 % short. Roughly **one third** of the gap is the response-cell
mismatch (a `(s, K)` block spans several `Z_QSO` response cells, because the cells are binned on
`Z_QSO` while the fold's K blocks are `z_DLA`) and **two thirds** is the bin-centre evaluation
(`Var[N + μ]` over a 0.2–0.3 dex latent bin, ≈ 0.055–0.08 dex added in quadrature). The effect is
largest at high S/N precisely because σ is smallest there, so the same additive variance is a larger
fractional error — which is exactly the 8–25 % (high S/N) / 3–10 % (S/N 2–3) pattern the forensics
reported.

**The candidate causes that were checked and are NOT the driver.** Per 0.1-dex sub-bin, in the
calibration cells:

* **moment-vs-ML estimation**: over all 213 populated sub-bins in the nine cells the untruncated
  skew-normal ML σ has median ratio **0.984** to the sample sd (mean 0.971, 16–84 % 0.941–0.994) —
  a 1.6 % median narrowing, an order of magnitude short of the row-level defect. After the
  quadrature repair, choosing the ML estimator gives a held-out width bias of **−0.6 %** and the
  sample-moment estimator **+2.3 %** — ML is *better*, so the estimator class was not the defect;
* **the fitted σ polynomial**: `σ_surface(N)/σ_sample(N)` over the same 213 sub-bins has median
  **1.000** (mean 0.999, 16–84 % 0.933–1.070) — the surface reproduces the conditional width, with
  scatter but no bias;
* **the σ floor** (`SIG_MIN = 0.02`, `resp_sig_floor = 1e-3`) never binds (smallest fitted σ = 0.075);
* **outlier clipping**: only the skewness is clipped (`SKEW_CAP = 0.95`), never σ;
* **polynomial degree**: CV over the re-expression degree {3, 4, 5} moves the held-out width bias by
  < 0.001.

**The repair (R1c).** The object is re-expressed so the midpoint evaluation returns the
bin-marginalised moments: for every cell and every latent bin, the exact first three moments of the
mixture of skew-normals over the bin are computed by 17-node quadrature with a **flat** within-bin
weight, and the three moment polynomials are refitted through those 16 per-bin targets. The
within-bin weight is deliberately **flat and not the mock's f(N)**: a weight taken from the
population would make the fixed calibration object depend on the science parameter. No new data fit
and no new data-driven freedom is introduced — only the deterministic quadrature of the same fitted
conditional surfaces plus a degree choice. The re-expression is exact to
`max|Δ| = 6.0e-4` (μ), `1.0e-2` (σ), `5.9e-2` (skew) at the 16 bin centres. **R1c is tied to the
latent grid it was marginalised on** (`ntrue_edges` of the `b300` packs); this is stamped in its
provenance.

R1c leaves the *response-cell* third of the gap unrepaired by construction (one coefficient set
serves several `(s, K)` blocks) — visible as the residual +0.10 width at S/N 6–7 and −0.04 at
S/N ≥ 7.

### Model selection for R1c (CV, no closure)

Rule, causal and fixed in code: **primary = |held-out count-weighted MEAN width residual|** (the
defect being repaired is a width *bias*; the per-block RMS is dominated by held-out counting noise
and barely discriminates), **tie-break = held-out RMS mean residual**.

| candidate | \|mean width resid\| | RMS width | RMS mean | RMS up | loglik/event |
|---|---|---|---|---|---|
| **ML, deg 3** | **0.0058** | 0.1400 | 0.0320 | 0.0673 | **0.4890** |
| ML, deg 4 | 0.0060 | 0.1404 | 0.0318 | 0.0673 | 0.4880 |
| ML, deg 5 | 0.0067 | 0.1404 | 0.0319 | 0.0672 | 0.4883 |
| sample, deg 3 | 0.0226 | 0.1375 | 0.0361 | 0.0722 | 0.4713 |
| sample, deg 4 | 0.0220 | 0.1385 | 0.0362 | 0.0731 | 0.4711 |
| sample, deg 5 | 0.0218 | 0.1389 | 0.0363 | 0.0727 | 0.4723 |

ML/deg-3 wins the primary score and is simultaneously best on RMS mean, RMS up-leakage and per-event
log-likelihood — the choice is not a trade-off.

---

## 4. R1a and R1b: what "no clamp" and "no ramp" actually buy, and what they extrapolate

**R1a.** The merged grid pushes the last fitted sub-bin centre from a uniform **21.35** (R0, all nine
cells) to **21.50–21.82** in eight cells (cell (SNR≥6.5, z 2.56–2.96) still stops at 21.35 because
its high-N sub-bin ML fits fail the `ok` test). `fit_rng` is then declared as the full latent support
`(19.0, 22.4)` so no clamp acts anywhere. **The documented extrapolation:** latent bin centres above
the last fitted centre — i.e. 21.9, 22.0/22.1 and 22.25 depending on the cell — are served by the
degree-2 + shared-cubic polynomial extrapolated by at most **0.9 dex** (versus up to **0.9 dex of
frozen clamp** in R0, which is worse: a clamp is a zeroth-order extrapolation). Those bins carry
< 40 held-out detections, so they are unvalidated in either variant; this is stated rather than
hidden. Effect: the mean residual at b ≥ 21.3 falls from **+0.024 to +0.009 dex** and at
[21.3,21.5) from **+0.018 to +0.002 dex**; the overall mean residual falls from −0.0047 to −0.0013.

**R1b.** With the ramp removed, the skew above 21.5 is the fitted polynomial rather than exactly 0.
Effect is confined to where the ramp acted: the held-out row-skew residual at b ≥ 21.3 improves from
**−0.142 to −0.021**, and the up-leakage residual there from +0.065 to +0.059. Overall aggregates
are unchanged because those bins carry ~1 % of the detections.

**Honest note (PI §16 — "a repair that moves the defect elsewhere is not a PASS").** At the single
bin [21.5,21.7) the held-out up-leakage residual *worsens* monotonically along the ladder:
R0 +0.140 → R1a +0.181 → R1b +0.188 → R1c +0.192. The driver there is the **mean**, which stays
+0.058 dex high in every variant: the polynomial extrapolation (and, at ~20 % of the effect, the flat
within-bin weight versus the steep true f(N)) puts that bin's row too far right. This is the clearest
un-repaired defect in the family and it is not fixed by R1c.

---

## 5. The objects and how to consume them

Per variant: `response_<variant>.npz` (`mu_coef`, `sig_coef`, `skew_coef`, `fit_rng`, `N_ref`,
`skew_ramp`, `fit_rng_data`, `shared`, `provenance`) — the `adopted_response` schema plus the
variant's skew ramp, which R0's schema kept in the pack.

Per variant × family: `Mg_<variant>_<fam>.npz` with `Mg (8, 15, 29, 16)`, `phi_ref_used (3,3,16)`,
`variant`, `coefficients`, `provenance`. Built by the COMMITTED
`count_conserving_fold.surface_masses` under the ratified count-conservation rule
(`masses / φ × adopted_phi_ref`), i.e. the identical arithmetic to
`cc_posterior_validation.build_cc_tensors`, and passed straight to
`validation/fp_ladder/run_ladder.py`'s `Mg_fixed` hook. The three families' tensors are identical
because the packs share the geometry and the frozen `adopted_phi_ref`; they are delivered separately
so a pack-side divergence would be caught.

The variant's skew ramp is carried **inside** the calibration object (a shim carrying
`ntrue_edges / resp_N_ref / resp_skew_ramp / resp_sig_floor` is handed to the unmodified
`surface_masses`, the same device `validation/absorber_diag/build_matched_ops.py` already uses). No
pack field is changed, and the delivered product is the precomputed tensor, so nothing downstream
needs to re-read the ramp.

---

## 6. R1d — opened, implemented, and NOT recommended

**Opening test** (declared before fitting): open R1d only if R1c leaves a residual the 3-moment
skew-normal family cannot carry — |held-out row-skew residual| > 0.05 overall, or |in-bin fraction
residual| > 0.05 at b ≥ 21.3. R1c gives **−0.249** and **−0.105** → opened.

Representation: per response cell and per observed-grid offset from the cell holding `centre(b)`,
the empirical mass fraction, smoothed along N by a ridge-penalised low-order polynomial weighted by
the per-bin event count; degree and ridge chosen by the same 2-fold sightline CV on the held-out RMS
in-bin-fraction residual.

**Result.** CV selects degree 2 with **ridge λ = 0**, i.e. *no shrinkage at all* — the CV is asking
for maximum flexibility. The selected object has **702 nominal / 531 effective** calibration
coefficients against 84–108 for R0–R1c, and it buys:

* row skew: −0.249 → **+0.096** (the one place it genuinely wins, as expected from a non-parametric
  shape), and at b < 19.7 −0.297 → −0.008;
* width bias: −0.0058 → **+0.0583** (six times *worse* than R1c; wide, because its rows marginalise
  the whole response cell's covariate spread rather than the block's);
* RMS mean residual 0.032 → **0.046**, RMS down-leakage 0.058 → **0.095**, RMS up-leakage
  0.067 → **0.095** — all worse;
* per-bin residuals that are visibly erratic (spikes at 19.8 and 20.8 in
  `respvar_fig2_cv_residuals.png`), the signature of a smoother that is following counting noise;
* transport to the other two families is the worst of all variants (+0.073 / +0.100 / +0.127).

**Recommendation: R1d is implemented and reported, not recommended.** It is exactly the
"zero-HBI-DOF change that is still an overly flexible calibration model" the ruling warns about
(§5), and its dimensionality is the §19 trigger — flagged here rather than carried forward.

---

## 7. What is still open (no adoption is proposed)

1. **The response-cell / block mismatch.** The `(SR, ZR)` cells are binned on `Z_QSO`; the fold's K
   blocks are `z_DLA`. One third of the width gap and the residual S/N-stratum pattern come from
   this. Repairing it means re-blocking the cell axis (or resolving SR at the pack's 8 strata),
   which changes `resp_z_edges` / `resp_snr_edges` and therefore the pack — **not attempted here**,
   and it belongs to a PI ruling, not to this ladder.
2. **[21.5,21.7)**: mean +0.058 dex and up-leakage +0.19 in every variant (§4).
3. **b < 19.5**: the observed grid starts at 19.5, so the rows of the two lowest latent bins are
   grid-edge objects; R1c improves their width from −39 %/−19 % to −22 %/−9 % but they remain the
   worst-modelled bins and are outside the reporting floor.
4. **Family transport**: a ±2.5 % width offset between 2LPT-0, London-0 and Saclay-0 survives every
   variant.
5. **Latent-grid tie**: R1c's object is defined against the `b300` packs' `ntrue_edges`. A change of
   latent binning requires re-marginalising (the code does this deterministically).

---

## 7b. Unit tests

`tests/test_response_variants.py` — **25 tests, all passing (48 s, env `gpdla`)**. They cover:

* element-wise equality of every re-implemented estimator with the COMMITTED code it mirrors —
  `moment_to_skewnormal` vs `count_conserving_fold._m2sn_vec`, `subbin_moments` (sample / ML /
  truncated ML) vs `adopted_response/fitlib.subbin_moments`, `surfaces_percell` vs
  `fitlib.surfaces_from_rows`, `surfaces_shared` vs `adopted_response/run_d2b_lib.shared_surfaces`
  (`max |Δ| == 0.0`, never a default `np.allclose` tolerance);
* the CV split is a genuine SIGHTLINE split (no TARGETID in both folds, complete and disjoint);
* the adaptive sub-binning: monotone, covers the support, every bin but the last ≥ the floor, never
  finer than the base step, and provably reaches past R0's 21.35;
* the mixture algebra against closed forms — a degenerate one-component mixture, a two-normal
  mixture's mean/variance/third moment, and `sd_marginal² = σ² + (1+μ′)²·Var(N)` for a constant-σ /
  linear-μ object at every quadrature node count;
* `row_stats` against a hand-computed discrete distribution and for normalisation invariance;
* **a counting identity** on `empirical_rows` (total == the number of in-grid events) — the decisive
  check for the recurring one-sided-support bug class;
* the `Mg` gather's shape and per-`(s,k)` cell selection;
* R1d's rows being unit mass, non-negative and count-conserving, and the ridge reducing the
  effective DOF;
* the coefficient count being the declared one, and `NO_RAMP` being a true no-op while the
  production ramp zeroes the skew above 21.5.

**Mutation-checked before this report was written** (feedback_allclose_atol_tiny_values): four
deliberate defects — dropping the `3·d·σ²` term in the mixture third moment, replacing the
marginalised width by the mean conditional width, transposing the `Mg` gather indices, and inverting
the skew-ramp sign — each turned the suite red, and the suite is green again with the code restored.

**Numerical note.** The committed `ml_trunc` estimator is not robust: its Nelder-Mead wanders badly
when the `x̂ ≥ 19.5` selection bites hard (single sub-bin fits from 2 s to > 600 s were measured).
No variant in this ladder uses it — R0–R1c use `ml` or `sample` — but anyone considering a
truncation-aware estimator should know.

---

## 8. Reproduction

```bash
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate gpdla
export PYTHONPATH=/home/mfho/wt_abs_diag_2026-09 OMP_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd /home/mfho/wt_abs_diag_2026-09
OUT=/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/response

python validation/absorber_ladder/response/extract_calib_events.py \
    --out $OUT/calib_events_2lpt0.npz --expect-events 73845
python validation/absorber_ladder/response/build_variants.py \
    --events $OUT/calib_events_2lpt0.npz --out-dir $OUT
python validation/absorber_ladder/response/make_figures.py \
    --out-dir $OUT --events $OUT/calib_events_2lpt0.npz \
    --fig-dir /home/mfho/desi_gpy_dla_notes/figures/2026-09-13_absorber_ladder/response
python -m pytest tests/test_response_variants.py -q
```

Runtime: event extraction 13 s; the full variant build (all CV grids, 5 variants, 3 families)
**≈ 5 min on one core**; figures 40 s. Total well under the < 30 core-h estimate.

---

## 9. Twelve-line factual summary

1. The 2LPT-0 calibration event table was **rebuilt** (the raw pair table is not on disk): 73,845
   matched detections on 65,011 sightlines.
2. Refitting the estimator of record on it reproduces `adopted_response_v1p1.npz`'s
   `mu/sig/skew/fit_rng` with **max |difference| exactly 0.0**, and R0's bin masses reproduce the
   pack's deployed kernel bit-for-bit.
3. `Mg_R0_2lpt0.npz` is bit-identical to `cc_posterior_validation.build_cc_tensors`'s tensor, shape
   `(8,15,29,16)` — consumable by `run_ladder.py --fix M` as `Mg_fixed`.
4. **Cause of the 8–25 % width defect: a midpoint-quadrature error in the deployment, not the σ
   estimator.** `surface_masses` evaluates the response at `N = centre(b)` in one `(SR,ZR)` cell,
   dropping `E[σ²] − σ²(centre)` (the block's covariate spread, ≈ 1/3) and `Var[N + μ]` (the latent
   bin's own width, ≈ 2/3); the two-term reconstruction matches the measured row width to ≈ 1 %.
5. The estimator class was checked and exonerated: over all 213 populated sub-bins untruncated-ML σ
   has median ratio 0.984 to the sample sd, the fitted σ surface reproduces the conditional sub-bin
   width to a median ratio of 1.000, and the σ floors never bind (smallest fitted σ = 0.075).
6. **R0** (84 + 18 coefficients) held-out: width **−6.2 %**, mean −0.005 dex, +0.024 dex and
   +0.074 up-leakage at b ≥ 21.3, width −12.8 % at b < 19.7.
7. **R1a** (84, clamp removed, last fitted centre 21.35 → 21.50–21.82): mean residual at b ≥ 21.3
   +0.024 → **+0.009 dex**; width −5.1 %.
8. **R1b** (84, skew ramp removed): row-skew residual at b ≥ 21.3 **−0.142 → −0.021**.
9. **R1c** (108, bin-marginal quadrature repair, flat within-bin weight; ML estimator and degree 3
   chosen by CV): width **−6.2 % → −0.6 %** overall, −12.8 % → −6.3 % at b < 19.7, −6.2 % → +1.9 %
   at b ≥ 21.3, mean residual −0.0001 dex — and the best per-event held-out log-likelihood of the
   ladder.
10. **R1d** (702 nominal / 531 effective; CV selects zero ridge) fixes the row skew (−0.25 → +0.10)
    but is worse on width (+5.8 %), mean, both leakages and family transport — **implemented,
    reported, not recommended**; its dimensionality is the PI §19 trigger.
11. Residuals no variant repairs: the `Z_QSO`-cell / `z_DLA`-block mismatch (1/3 of the width gap),
    [21.5,21.7) mean +0.058 dex and up-leakage +0.19, and a ±2.5 % width offset between the three
    mock families.
12. Nothing was committed; no sampler was run; **no closure number was read or used**; the ladder
    returns for a PI ruling before any adoption.
