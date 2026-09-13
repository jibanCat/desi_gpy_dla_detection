# C1 — FIXED completeness calibration variants (nested family, CV-selected)

**Date 2026-09-13. VALIDATION-ONLY. No sampler was run. Nothing under `CDDF_analysis/` was read for
writing or modified; no existing file in the worktree was modified. Nothing committed.**
Worktree `/home/mfho/wt_abs_diag_2026-09`. Governing ruling:
`governance/PI_RULING_2026-09-13b_ABSORBER_LADDER_GO_CUT_FEEDBACK.md` §6–7 (C1 may include smooth
variants, complexity by CV, S/N as a continuous covariate), §15 (calibration-side predictive checks
mandatory), §9 (causal, progressive opening), §18 (return options, do not adopt).

Diagnosis this repairs: `validation/absorber_diag/OPERATOR_FORENSICS_REPORT.md` §5 (completeness
truth vs calibration) and §2.5 (the P6b class with no term).

Figures: `/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_absorber_ladder/completeness/comp_fig*.png`.
Products: `/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/completeness/`
(33 files, `SHA256SUMS` included).

---

## 0. Code, environments, gates

| file (all NEW) | env | what |
|---|---|---|
| `validation/absorber_ladder/completeness/cal_fit.py` | either | pure binomial-GLM / IRLS / CV / scoring helpers |
| `validation/absorber_ladder/completeness/build_cal_tables.py` | `gpdla` | matched table **with TARGETID**, split into sightline halves |
| `validation/absorber_ladder/completeness/fit_completeness_variants.py` | either | the nested variant family, CV, deliverables, P6b |
| `validation/absorber_ladder/completeness/make_figures.py` | either | the six figures |
| `validation/absorber_ladder/completeness/verify_runner_hook.py` | `gpdla-hbi` | `--c-fixed` consumability smoke test |
| `tests/test_completeness_variants.py` | either | 23 unit tests |

`build_cal_tables.py` **imports** `validation/absorber_diag/build_matched_ops.py` and re-uses its
cut/match/op-mask verbatim; the only thing it adds is the per-detection and per-truth `TARGETID`
that the empirical operators do not carry and the sightline-half CV needs. Everything it produces
is gated **elementwise** against objects that already exist on disk, and it fails closed:

* **TRUTH GATE** — rebuilt `truth_bks[b,k,s]` == the adopted pack's `truth_counts_bks`, max |diff| 0
  (101 949 / 106 705 / 104 561 systems). PASSED on all three families.
* **DETECTION GATE** — rebuilt `det_bks` == `empirical_ops_<fam>.npz:N_det_all_bks_true_z`,
  max |diff| 0 (84 470 / 90 167 / 88 406 matched detections). PASSED.
* **P6b GATE** — rebuilt `P6b_cks` == `empirical_ops_<fam>.npz:P6b_cks`, max |diff| 0
  (3 200 / 2 611 / 2 668). PASSED.
* **PARITY PARTITION GATE** — even-TARGETID + odd-TARGETID halves sum back to the full table
  elementwise. PASSED. Balance: 49.90 % of truth systems and 49.85 % of detections in the even
  half (169 140 distinct truth sightlines, 84 472 even) — the halves are not adversarially split.
* **C0 GATE** — `eta_hat` / `sigma_hat` recomputed here equal
  `CDDF_analysis/hbi_mcmc/forward.eta_hat_sigma_hat(molly_n_det, molly_n_tot)` **elementwise
  exactly** (max |Δη| = 0).
* **C0 FOLD GATE** — the delivered `C_C0_*.npz:C_fixed` equals
  `sigmoid(consts.eta_hat)[:, consts.b_to_cell]` built by the committed
  `forward.build_consts(load_pack(...))`: `b_to_cell` identical elementwise, C agrees to
  5.6e-17 against an **explicit** atol of 1e-15 (the residual is the difference between
  `jax.nn.sigmoid` and the branch-stable numpy `expit`, nothing else).

**Tests.** 23 tests, 0.4–1.0 s, pass in **both** environments (`gpdla`: 22 passed 1 skipped — the
skip is the `forward` cross-check, which needs jax; `gpdla-hbi`: 23 passed, the cross-check
running and exact). **Mutation-checked before this report was written**: 13 deliberate mutants
(Jeffreys ½ removed, CV folds not swapped, residual denominator switched from observed detections
to truth trials, deviance factor/saturated term, IRLS weight missing the trial count, tensor-design
row order transposed, per-stratum design not block-diagonal, additive design given an interaction,
coarse-z dummy reference level shifted, log-loss missing the non-detection term, unpaired SE) —
**12 killed**. The one survivor (dropping the `t > 0` mask inside `binom_deviance`) is *provably
equivalent*: a zero-trial cell contributes `0·log q + 0·log(1−q) = 0` either way. Every numeric
assertion uses an explicit tolerance; none relies on `np.allclose`'s default atol.

---

## 1. The calibration data, and the convention every variant obeys

**Calibration family: 2LPT-0 only.** London-0 and Saclay-0 are read *only* to report a transfer
residual; no design matrix is ever built from them (`fit_completeness_variants.py` takes its grid,
its covariates and every count it fits from `CAL_FAMILY = "2lpt0"`).

**The object is a DETECTION probability.** `det_bks` counts a matched detection irrespective of
where its observed N̂ lands, so `det/truth` is `C_det` — the same object `molly_n_det/molly_n_tot`
measures and the same slot the fold's `C_bs = sigmoid(eta_hat + psi_c)[:, b_to_cell]` occupies. The
in-grid counting fraction `phi` stays with the response kernel, where the forensics showed it is
exact (≤ 0.3 %) above 19.9 (§5c). This is a deliberate difference from the existing `--fix Cz`
diagnostic, which passes `C_true_bs` (an observed-grid-restricted rate, i.e. `C_det × phi`) into a
fold whose kernel already carries `phi`. **All seven objects delivered here are on the `C_det`
convention, C0 included, so they are like-for-like with each other and with the frozen surface.**

Grid (the pack's own): B = 16 latent 0.2-dex bins on `ntrue_edges` 19.0 → 22.4; S = 8 S/N strata of
which **6 are live** (s = 2…7; s = 0,1 carry zero exposure and zero truth and are written as
**exact zeros**); K = 3 coarse-z blocks, Kf = 15 fine-z bins via `kz_to_K`.
Covariates: `x = N_c − 20`, and `u = log10(median S/N of the stratum) − u0` with
`u0 = 0.685231` (the truth-weighted mean over live strata) — stratum medians
0.3871 / 0.5380 / 0.6492 / 0.7378 / 0.8105 / 1.0287 in log10.

**A second, separate defect found while setting this up (not in the forensics report).** The frozen
C0's *denominator population is not the population `truth_counts_bks` defines*. Summed over the
molly cells the basis reaches (≥ 19.0) and the live strata, the frozen matrix carries
**n_tot = 107 759, n_det = 86 101, C = 0.7990**, while the pack's own window carries
**n_tot = 101 949, n_det = 84 470, C = 0.8286** — the frozen denominator is **5.7 % larger** and its
pooled completeness **3.6 % lower**. The molly TSV was generated on molly's own z / proximity-collar
window, not the pack's. This is a member of the recurring one-sided-support class (numerator and
denominator on different supports) and it is a *first-order* contributor to C0's deficit: it is a
population mismatch, not a grid-resolution effect, and re-fitting on the pack window removes it by
construction.

---

## 2. The nested family

| variant | calibration data | representation | fitted coef. | fixed before the HBI? | motivating defect |
|---|---|---|---|---|---|
| **C0** | frozen `molly_matrix.tsv` (2LPT-0, molly's own window) | `sigmoid(eta_hat)` on 12 N-cells × 8 S/N strata, gathered to the latent grid by `b_to_cell` | **0** (frozen; not refitted) | yes — unchanged baseline | none (it *is* the baseline) |
| **C1g** | 2LPT-0 matched table, pack window, z-pooled | free Jeffreys rate per live (b, s) cell | **96** (96 live cells) | yes | §5a grid resolution: 16 latent bins share 8 molly cells, so C0 is a step function across a steeply climbing C |
| **C1gz** | 2LPT-0, resolved in coarse z | free Jeffreys rate per live (b, K, s) cell | **279** (live cells of 288) | yes | §5b the z-dependence of completeness at fixed S/N stratum |
| **C1n** | 2LPT-0, z-pooled | independent logistic **degree-1** polynomial in x per stratum | **12** = 6 × 2 | yes | §5a, with the step function replaced by a smooth monotone curve |
| **C1ns** | 2LPT-0, z-pooled | 2-D tensor `logit = Σ_{j≤3} Σ_{i≤2} β_ji x^j u^i` | **12** | yes | §5a + §5(a) S/N trend, treating S/N as a continuous covariate |
| **C1nsadd** | 2LPT-0, z-pooled | additive-in-logit `poly_3(x) + poly_2(u)`, no interaction | **6** | yes | the most compact form that still repairs §5a and the S/N trend |
| *C1nsz* | 2LPT-0, (s,K,b) | `poly_3(x) + poly_2(u) + δ_K` (K0 reference) | **8** | yes | **EXPLORATORY — outside the §6–7 list.** Reported only, for the PI choice: can the coarse-z residual be removed with 2 coefficients instead of 279 cells? |

Fit: binomial IRLS (damped Newton on the canonical logit link) with a ridge of **1e-6** on
non-intercept columns only — a conditioning term, not a prior: raising it to 1e4 crushes the slope
(tested), at 1e-6 the slope is unchanged to the printed digits and every fit converged
(`converged = True`, Hessian condition recorded in each object's provenance).

`C1nsadd` coefficients (the delivered object):
`[β0, βx, βx², βx³, βu, βu²] = [3.0324, 2.3206, −0.0001, −0.2056, 5.4921, −7.2294]`.

---

## 3. Cross-validation — which smoothness level the data support

Two folds by **sightline halves (TARGETID parity)**: fit on the even half, score the odd half, and
vice versa. Score = binomial log-loss in nats (the combinatorial constant, which depends only on the
data, is dropped so variants are directly comparable). 101 949 held-out truth trials; the
comparison statistic is the **paired per-trial difference**, whose SE is the root of the summed
held-out Bernoulli variances. `comp_fig3_cv_curves.png`.

| variant | coef. | CV log-loss (all b) | Δ vs C0 | σ | CV log-loss (reported b ≥ 19.5) | in-sample deviance |
|---|---|---|---|---|---|---|
| C0 frozen | 0 | 33 492.35 | — | — | 19 734.09 | 1 443.8 |
| C1n deg 0 | 6 | 39 165.38 | **+5 673.0** | +57.2 | 22 801.01 | 12 775.7 |
| **C1nsadd** (add 3×2) | **6** | 32 849.39 | **−642.96 ± 36.03** | **−17.8** | 19 249.71 | 128.8 |
| C1ns add 4×3 | 8 | 32 848.47 | −643.88 ± 36.35 | −17.7 | 19 249.36 | 124.9 |
| C1n deg 1 | 12 | 32 833.87 | −658.49 ± 36.41 | −18.1 | 19 241.94 | 101.7 |
| C1ns tensor 3×2 | 12 | **32 830.38** | **−661.98 ± 35.83** | −18.5 | 19 234.07 | 89.6 |
| C1n deg 3 | 24 | 32 831.87 | −660.49 ± 36.08 | −18.3 | 19 239.36 | 68.9 |
| C1g free grid | 96 | 32 873.77 | −618.58 ± 35.73 | −17.3 | 19 280.26 | 22.2 |

Selection rule, **predeclared in the code** and applied unchanged: take the minimum-CV candidate,
then step *down* in complexity to the simplest candidate within one **paired** SE of that minimum
(the standard 1-SE rule). Both answers are reported.

* `C1n`: min-CV = degree 3 (24 coef); 1-SE pick = **degree 1** (12 coef; Δ = +2.00 vs a paired SE of
  6.01). Degree 2 is *outside* 1 SE (+7.13 ± 4.45) — the N-curve wants odd-order freedom, not
  quadratic. Degree 0 is catastrophic (+5 673), so the **N-dependence is unambiguously
  data-supported**.
* `C1ns` tensor: min-CV = 3×2 and the 1-SE pick is the same (2×2 is +8.32 ± 4.38, outside 1 SE).
* `C1nsadd`: min-CV = 4×3 (8 coef); 1-SE pick = **3×2** (6 coef; Δ = +0.92 ± 1.95).

**Is smoothness data-supported, or imposed?** Data-supported, and on two axes separately.
(i) *In N*: degree 0 loses 5 673 nats; degree 1 gains 658; degrees 2–4 add nothing outside noise.
(ii) *In log S/N*: with the S/N covariate entered **linearly** the whole family stalls at
Δ ≈ −230 nats; going to **quadratic** in log S/N jumps to Δ ≈ −643; cubic adds nothing
(−643.9 vs −643.0, well inside the paired SE). So the S/N smoothness is not an assumption that
happens to fit — the CV curve has a sharp, reproducible step exactly at the degree the data
demand, and stops there (`comp_fig3`, middle panel).
**The decisive parsimony result: 6 smooth coefficients beat 96 free cells.** C1nsadd
(−643.0 ± 36.0) is *better* out of sample than C1g (−618.6 ± 35.7) with 1/16 of the freedom, and
within 19 nats of the global best. The free grid pays for its 96 cells in held-out noise.

**Coarse z, scored on (s, K, b) cells so the comparison is like-for-like** (`comp_fig3`, right):

| representation | coef. | CV log-loss | Δ vs z-pooled |
|---|---|---|---|
| z-pooled (the C1g prediction broadcast over K) | 96 | 32 873.77 | — |
| *C1nsz* additive + 2 z offsets | 8 | 31 871.61 | **−1 002.15** |
| C1gz free (b, K, s) grid | 279 | 31 584.77 | **−1 289.00 ± 53.43 (−24.1 σ)** |

z structure is the **largest single data-supported gain in this whole exercise**, larger than the
N/S-N grid repair. Eight coefficients recover 78 % of it; the free grid is still 286.85 ± 37.95
(+7.6 σ) better, so the z shape is not fully captured by a constant logit offset per block.

---

## 4. Held-out predictive performance (PI ruling §15)

Held-out expected/observed detections − 1, on 2LPT-0, from the parity-half fits (every number below
is out of sample). `comp_fig2_residual_maps.png`, `comp_fig4_residual_by_coarse_z.png`.

| variant | total | reported window | by K (K0 / K1 / K2) | by S/N stratum (s2 … s7) |
|---|---|---|---|---|
| C0 | **−2.55 %** | −2.61 % | +2.41 / **−7.49** / **−8.45** | −2.89 −3.58 −3.44 −2.56 −2.21 −1.53 |
| C1g | −0.09 % | −0.11 % | +4.97 / −5.13 / −6.10 | −0.05 −0.09 −0.11 −0.17 −0.18 −0.05 |
| C1n | +0.00 % | +0.03 % | +5.07 / −5.05 / −6.01 | +0.02 0.00 0.00 0.00 0.00 0.00 |
| C1ns | +0.00 % | +0.02 % | +5.07 / −5.05 / −6.02 | −0.14 +0.44 −0.10 −0.63 +0.34 +0.02 |
| C1nsadd | +0.00 % | +0.03 % | +5.07 / −5.05 / −6.02 | −0.13 +0.41 −0.09 −0.62 +0.34 +0.02 |
| **C1gz** | −0.26 % | −0.32 % | **−0.17 / −0.24 / −0.63** | −0.18 −0.25 −0.33 −0.43 −0.57 −0.16 |
| *C1nsz* | +0.00 % | +0.03 % | **−0.00 / +0.00 / +0.01** | −0.13 +0.40 −0.12 −0.58 +0.36 +0.01 |

Held-out, by latent N bin (%), reported window 19.5 → 21.7:

| variant | 19.5 | 19.7 | 19.9 | 20.1 | 20.3 | 20.5 | 20.7 | 20.9 | 21.1 | 21.3 | 21.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | −3.31 | **−9.40** | +1.15 | −2.55 | −1.59 | −0.50 | −1.95 | −0.41 | −0.94 | −0.86 | −0.97 |
| C1g | −0.04 | −0.04 | −0.05 | −0.06 | −0.07 | −0.10 | −0.12 | −0.15 | −0.28 | −0.53 | −1.06 |
| C1n | +0.19 | +0.43 | −0.16 | +0.14 | −0.50 | −0.13 | −0.18 | +0.01 | +0.09 | +0.58 | +1.40 |
| C1nsadd | −0.18 | +0.31 | −0.03 | +0.40 | −0.26 | −0.00 | −0.19 | −0.13 | −0.16 | +0.24 | +0.93 |
| C1gz | −0.10 | −0.15 | −0.17 | −0.19 | −0.21 | −0.27 | −0.36 | −0.49 | −0.85 | −1.64 | **−3.27** |

Readings. (a) The frozen C0's alternating-sign, ±6–9 % pattern inside each molly cell — the
signature §5a identified — is *entirely* removed by every C1 member, held out, at every N and every
stratum. (b) ⚠️ **Qualified by Addendum A — read it before quoting this paragraph.** The numbers here are for C *alone*; the fold applies `C·g`, and the frozen `g` removes most of this structure. The N/S-N repair leaves the **coarse-z residual of C completely untouched**: +5 / −5 / −6 %
for C1g, C1n and C1ns alike, and *larger in K1/K2* than C0's because the total is now pinned. A
repair that fixes the total and moves nothing in z is exactly the "moves the defect elsewhere"
pattern §16 warns about, and it is visible here as a first-class gate. (c) Only a z-resolved object
closes it (C1gz ≤ 0.63 % in every block), at the price of a growing high-N residual
(−1.6 %, −3.3 % in the two top reported bins) where its (b, K, s) cells run out of systems.

**Sparse-region uncertainty** (`comp_fig5_sparse_uncertainty.png`). Of the 96 live (b, s) cells,
**18 hold < 50 truth systems and 12 hold < 20**; all of them sit at N ≥ 21.7 (b ≥ 13). The worst are
(b = 15, s = 6) with **zero** trials, then 1, 1, 4, 5, 5, 5, 7 trials. Cell-level Jeffreys sd there
is 0.06 → 0.50 (C1g) and 0.003 → 0.09 (C0, whose coarser cells pool more). The smooth variants
borrow strength: at N = 21.6 the delivered calibration sd is 0.005–0.030 for C1g, 0.0015–0.0047 for
C1ns, and 0.0002–0.0047 for C1nsadd. **This is a caution, not a selling point** — a smooth model's
reported sd is the *fit* uncertainty of an imposed shape, and C1n's 2e-4 at N = 21.6 is an
extrapolation confidence the 11 systems in that cell cannot support. The honest statement is:
above 21.7 no representation here is data-constrained per cell; the smooth ones are constrained by
the *shape* fitted below, and C1g's empty cell falls back to C0 (recorded in the object).

---

## 5. Transfer residuals — REPORT ONLY (nothing was fitted to these families)

Expected/observed detections − 1 for the **fixed 2LPT-0** object against each family's own matched
truth (`comp_fig2`, `comp_fig4`):

| variant | 2LPT-0 total / reported | London-0 total / reported | Saclay-0 total / reported |
|---|---|---|---|
| C0 | −2.55 % / −2.61 % | **−4.15 % / −3.73 %** | **−4.56 % / −4.15 %** |
| C1g | −0.05 % / −0.06 % | −1.73 % / −1.26 % | −2.13 % / −1.65 % |
| C1gz | −0.13 % / −0.16 % | −1.82 % / −1.37 % | −2.22 % / −1.75 % |
| C1n | −0.00 % / +0.03 % | **−1.69 % / −1.19 %** | **−2.08 % / −1.56 %** |
| C1ns | +0.00 % / +0.02 % | −1.69 % / −1.19 % | −2.08 % / −1.57 % |
| C1nsadd | −0.00 % / +0.03 % | −1.69 % / −1.18 % | −2.08 % / −1.56 % |
| *C1nsz* | — | −1.69 % / −1.18 % | −2.08 % / −1.56 % |

By coarse z (transfer families):

| variant | London-0 K0 / K1 / K2 | Saclay-0 K0 / K1 / K2 |
|---|---|---|
| C0 | +0.27 / −8.65 / −9.34 | −0.64 / −9.03 / −9.26 |
| C1g | +2.78 / −6.32 / −7.07 | +1.89 / −6.71 / −6.90 |
| C1n, C1ns, C1nsadd | +2.83 / −6.29 / −7.02 | +1.93 / −6.67 / −6.86 |
| **C1gz** | **−2.15 / −1.51 / −1.32** | **−3.07 / −1.86 / −1.24** |
| *C1nsz* | −2.06 / −1.38 / −1.11 | −2.99 / −1.71 / −0.90 |

Three facts. (i) The N/S-N repair transfers: the ~4 % deficit C0 shows on families it was never
fitted to drops to **1.2–1.6 % in the reported window**, and the residual that remains is
*family-to-family*, not grid resolution. (ii) All the z-pooled variants transfer identically to
three decimals — the compact 6-coefficient object is as transferable as the 96-cell grid, which is
the strongest argument for the compact form. (iii) The +5/−5/−6 % K structure is **family-common**:
a z-resolved 2LPT-0 object removes it on London-0 and Saclay-0 too (≤ 3.1 %, and ≤ 1.9 % in K1/K2),
so it is a property of the detector's z response, not of any one mock's population.
⚠️ **This table compares C alone. The fold multiplies C by the frozen `g[b,k]`, which already
encodes that same z response — see Addendum A for the effective-completeness version, which is the
one that describes the fold.**

---

## 6. P6b — the sub-floor-host term with no parameter in the fold

`P6b_rate_<fam>.npz`. Measured on 2LPT-0 as `rate[c, K, s] = P6b_counts / dX` (hosts in
[17.2, 19.0), from the floor-17.2 matched pass, gated against the census), then transported by
multiplying each family's own `dX[k, s]`: `mu_P6b[c, k, s] = rate[c, kz_to_K[k], s] · dX[k, s]`.
Calibration: 3 200 events, pooled rate 6.394e-3 per unit dX.

| family | observed | predicted | residual | Poisson σ | by K (K0 / K1 / K2) |
|---|---|---|---|---|---|
| 2LPT-0 (calibration) | 3 200 | 3 200 | 0.0 % | 0.0 | 0 / 0 / 0 |
| London-0 | 2 611 | 3 170 | **+21.4 %** | **+10.9** | +53.1 / +21.4 / −8.0 |
| Saclay-0 | 2 668 | 3 051 | **+14.4 %** | **+7.4** | +32.7 / +15.0 / −11.4 |

`comp_fig6_p6b_transfer.png`. **P6b is the worst-transferring object in this package by a wide
margin** — a fixed 2LPT-0 rate per unit path over-predicts by 14–21 % overall and by 33–53 % in K0,
while *under*-predicting by 8–11 % in K2. The sub-DLA host population itself differs between mocks
(this is a population property, not a detector property), so a fixed-rate transfer is not
defensible as a survey-facing object on this evidence. Reported, not recommended.

---

## 7. Deliverables and the `--c-fixed` contract

Products dir `/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/completeness/`:

* `C_{C0,C1g,C1gz,C1n,C1ns,C1nsadd,C1nsz}_{2lpt0,london0,saclay0}.npz` — 21 files.
  Keys: **`C_fixed` (S = 8, B = 16)** z-pooled, **dead strata s = 0,1 exactly zero**; `C_fixed_sd`
  (S, B); `live_strata_mask` (8,); `ntrue_edges`, `snr_edges`, `kz_to_K`, `b_to_cell`;
  `provenance` (0-d object array holding indented JSON: variant id, representation, coefficient
  vector, coefficient count, CV block, held-out block, transfer block, the pivots, the git head,
  and the explicit statements `fixed_before_HBI: true` / `fitted_to_real_data: false`).
  For the z-carrying variants (`C1gz`, `C1nsz`) additionally **`C_fixed_bKs` (B, 3, S)**,
  **`C_fixed_bkS` (B, Kf = 15, S)** expanded with the pack's `kz_to_K`, and
  `C_fixed_bkS_with_g` (B, Kf, S).
  `C_fixed` is **family-independent** (it is the 2LPT-0 object); the per-family file exists so the
  runner can be pointed at one path per mock, and each carries that family's transfer residual.
* `P6b_rate_{fam}.npz` — keys `mu_P6b_cks` **(C = 29, Kf = 15, S = 8)** expected counts from the
  2LPT-0 rate against that family's `dX`; `rate_cKs` (29, 3, 8); `rate_cKs_halfE/halfO`;
  `obs_P6b_cks`; `dX`; grids; `provenance`.
* `cal_table_{fam}.npz` (+ `.provenance.json`) — the gated count tables with sightline halves.
* `completeness_variants_summary.json`, `completeness_observed.npz`, `runner_hook_smoke.json`,
  `SHA256SUMS` (33 entries).

**Consumability, verified (`verify_runner_hook.py`, no sampler).** For all 3 packs × 7 variants the
delivered `C_fixed` was pushed through the runner's own 2-D expression
`einsum("skcb,sb,bk->cks", Mg, C_fixed, g_bk·f·dN) · dX` at the pack's truth f: correct shape,
all-finite, dead strata zero. TP/total-counts at truth f: C0 0.8233 / 0.8714 / 0.8543 →
C1nsadd 0.8478 / 0.8966 / 0.8795 (2LPT-0 / London-0 / Saclay-0), i.e. the repair moves ≈ +2.4 % of
all observed counts into the TP prediction.

⚠️ **One contract defect to flag.** The runner's **3-D** branch
(`fp_ladder.model_cc_ladder`, `C_fixed.ndim == 3`) uses `w0 = f·dN` and therefore **drops `g_bk`**,
whereas the 2-D branch keeps it. A pure completeness passed to the 3-D branch is consequently folded
*without* the z-shape surface: the smoke test measures 0.8470 vs 0.8506 on 2LPT-0 (0.4 % of counts).
`C_fixed_bkS_with_g` is provided as the like-for-like array for that branch and the difference is
recorded in each object's provenance. **No existing file was modified to fix this** — it is reported
for a PI/implementation decision.

---

## 8. What this does not do, and what needs a ruling

1. **Nothing is adopted.** Seven objects, options only, per §18.
2. **C1nsz is outside the §6–7 list** (it adds a coarse-z term). It is labelled EXPLORATORY in the
   code, in its provenance and in every table here, and is delivered only so the PI can see that
   2 coefficients recover 78 % of a −1 289-nat effect. It must not enter a run without a ruling.
3. **The z finding is a candidate C2, and C2 is HOLD** (§10). Reported as a remaining
   calibration-side residual, not opened. **Addendum A downgrades it substantially**: on the fold's
   own scale, after the frozen `g`, the residual z gain is ≈ 3 σ, not 24 σ.
4. **The molly-window population mismatch (§1) is a diagnosis, not a repair of the frozen object.**
   Re-fitting on the pack window removes it by construction in every C1 member; the frozen C0 is
   left exactly as it is.
5. **Above N = 21.7 nothing here is per-cell data-constrained** (12 live cells with < 20 systems,
   one empty). The smooth objects extrapolate a shape; their tiny reported sd is fit uncertainty
   only. Reporting-window conclusions (≥ 19.5, and the ≥ 20.0 / ≥ 20.3 estimands) do not rest on
   those cells, but any open-topped Ω use would.
6. **P6b should not be transported as a fixed rate** on this evidence (+14–21 %, up to +53 % in K0).
7. **No mock closure run, no sampler, no real data** touched this package. The closure gates of
   §16 are the next step and are not reported here.

---

## 9. Twelve-line summary

1. Built a nested family of **seven FIXED completeness objects**, all learned on the **2LPT-0
   calibration family only**, with complexity chosen by **two-fold sightline-half (TARGETID parity)
   cross-validation**; London-0 and Saclay-0 were used **only** as transfer tests.
2. Every rebuilt count table is gated **elementwise, max |diff| = 0** against the adopted pack's
   `truth_counts_bks`, the empirical operators' matched detections, and the FP census's P6b; C0
   reproduces the committed `eta_hat` exactly and the fold's `C_bs` to 5.6e-17 (atol 1e-15).
3. **CV prefers a compact smooth surface over both the frozen grid and the free grid.** The
   parsimony winner is **C1nsadd — additive in logit, `poly_3(N−20) + poly_2(log S/N)`, 6
   coefficients** at Δ = **−643.0 ± 36.0 nats vs C0 (−17.8 σ)**; the global CV minimum is the
   12-coefficient tensor `C1ns` at **−662.0 ± 35.8**; the 96-cell free grid `C1g` is *worse* out of
   sample (**−618.6 ± 35.7**) than 6 smooth coefficients.
4. **Smoothness is data-supported, not imposed, on both axes:** degree 0 in N costs +5 673 nats,
   degree 1 gains 658, degrees 2–4 add nothing outside the paired SE; **linear** in log S/N stalls
   at −230 nats while **quadratic** jumps to −643 and cubic adds 0.9 ± 1.5.
5. The frozen C0's diagnosed defect is reproduced and then removed: its alternating ±6–9 %
   within-molly-cell error (worst −9.40 % held out at [19.7, 19.9)) goes to ≤ 0.5 % at every N and
   every stratum for every C1 member, held out.
6. A **second, previously unreported defect**: the frozen matrix's denominator population is
   **5.7 % larger** than the pack's truth population (107 759 vs 101 949 systems; pooled C 0.7990
   vs 0.8286) — molly's window, not the pack's. A member of the one-sided-support class; removed by
   construction in every C1 member.
7. **The N/S-N repair does not touch the coarse-z residual.** C1g, C1n and C1ns all leave
   **+5.1 / −5.1 / −6.0 %** by K0/K1/K2 (C0: +2.4 / −7.5 / −8.5 %) — the total is fixed and the
   defect stays put, exactly the §16 failure mode, visible as a first-class gate.
8. ⚠️ **Superseded by Addendum A, line A6 — this line scores C alone, not the effective `C·g`.**
   On C alone, z resolution is the largest data-supported gain in the package: **−1 289.0 ± 53.4 nats
   (−24.1 σ)** for a free (b, K, s) grid over the z-pooled prediction, scored on identical cells;
   `C1gz` drives the K residual to ≤ 0.63 % held out and ≤ 3.1 % on transfer, and the 8-coefficient
   `C1nsz` drives it to ≤ 0.01 % held out and ≤ 3.0 % on transfer. `C1nsz` recovers 78 % of the
   log-loss gain and is delivered **EXPLORATORY / not adopted** (C2 is HOLD).
9. **Transfer residuals (fit to 2LPT-0, evaluated on the other two, never fitted):** reported window
   C0 **−3.73 % (London-0) / −4.15 % (Saclay-0)** → C1nsadd **−1.18 % / −1.56 %**, with C1g, C1n,
   C1ns and C1nsadd agreeing to three decimals — the 6-coefficient object is as transferable as the
   96-cell one.
10. **P6b** (sub-floor-host [17.2, 19.0) rate per unit path, 3 200 calibration events, fixed and
    transportable) **transfers badly: +21.4 % (+10.9 σ_P) on London-0 and +14.4 % (+7.4 σ_P) on
    Saclay-0**, +33…+53 % in K0 and −8…−11 % in K2. Delivered as requested, **not recommended** as
    a survey-facing fixed object.
11. Above N = 21.7 no representation is per-cell data-constrained (12 live cells with < 20 truth
    systems, one empty); the smooth objects' small reported sd is fit uncertainty of an imposed
    shape, and this is stated as a caution, not a gain.
12. Deliverables are consumable: 21 `C_*` objects (`C_fixed` (8, 16), dead strata zero; `C_fixed_bKs`
    (16, 3, 8) and `C_fixed_bkS` (16, 15, 8) for the z variants) + 3 `P6b_rate_*` objects
    (`mu_P6b_cks` (29, 15, 8)) + `SHA256SUMS`, all smoke-tested through the runner's own einsum;
    **one contract defect flagged**: the runner's 3-D `C_fixed` branch drops `g_bk` (0.4 % of
    counts), for which `C_fixed_bkS_with_g` is supplied. 23 unit tests pass in both environments;
    13 mutants attempted, 12 killed, 1 provably equivalent. Nothing adopted, nothing committed, no
    existing file modified, no sampler run.


---

# Addendum A — the EFFECTIVE completeness `C[b,s] · g[b,k]`

**Added 2026-09-13 in response to a coordinator query.** Reproduced by
`validation/absorber_ladder/completeness/g_kernel_addendum.py` (NEW file, read-only on the products
directory). **No delivered product was changed; `SHA256SUMS` is untouched and still verifies;
nothing committed; no sampler run.** Figure:
`comp_fig7_effective_C_times_g_by_z.png`.

## A1. The correction

§4 and §5 compare each completeness object against `C_true[b,K,s]` **as a bare completeness**. That
is the right comparison for the *calibration object* — C is what the molly matrix measures and what
the binomial CV scores — but it is **not** what the fold applies. In the runner's 2-D `--c-fixed`
branch the fold forms

```
mu[c,k,s] = Σ_b Mg[s,k,c,b] · C[s,b] · g[b,k] · f[b,k] · dN_b · dX[k,s]
```

so the **effective completeness of a z-pooled object is `C[b,s] · g[b,k]`**, with
`g_bk = pack.g_grid[b_to_cell, :]` — the per-N-row-normalised completeness z-shape, measured once on
2LPT-0 on the `S2N_RED > snr_min` support (finding N1's source fix) and frozen byte-identically into
every pack. The z-carrying objects (`C1gz`, `C1nsz`) go through the 3-D branch, which **drops** `g`,
so they replace `C·g` and are compared without it.

`g` is a pure z-**reshape**: its truth-weighted mean is 1.0001 (2LPT-0), 1.0002 (London-0), 0.9981
(Saclay-0). It therefore cannot change the level of any residual, only its distribution in z — which
is exactly the K structure §4/§5 reported.

## A2. The K-residual table (the answer to the query)

Truth-weighted `Σ pred·truth_bks / Σ det_bks − 1` over all b and live s, per coarse block, in %.
Rows marked `× g` are the fold's effective completeness; rows marked `no g` are the §5 numbers,
repeated so the difference is visible in one place. `C1gz` / `C1nsz` are shown **without** `g`,
which is their like-for-like fold.

| variant | g | 2LPT-0 K0 / K1 / K2 | London-0 K0 / K1 / K2 | Saclay-0 K0 / K1 / K2 |
|---|---|---|---|---|
| C0 | no g | +2.41 / −7.49 / −8.45 | +0.27 / −8.65 / −9.34 | −0.64 / −9.03 / −9.26 |
| **C0** | **× g** | **−2.15 / −2.95 / −2.84** | **−4.11 / −4.23 / −3.96** | **−5.04 / −4.63 / −3.50** |
| C1g | no g | +5.01 / −5.09 / −6.06 | +2.78 / −6.32 / −7.07 | +1.89 / −6.71 / −6.90 |
| **C1g** | **× g** | **+0.31 / −0.40 / −0.26** | **−1.74 / −1.76 / −1.51** | **−2.66 / −2.17 / −0.94** |
| C1n | no g | +5.07 / −5.05 / −6.02 | +2.83 / −6.29 / −7.02 | +1.93 / −6.67 / −6.86 |
| **C1n** | **× g** | **+0.36 / −0.36 / −0.21** | **−1.70 / −1.73 / −1.46** | **−2.62 / −2.12 / −0.90** |
| C1ns | no g | +5.06 / −5.05 / −6.02 | +2.83 / −6.28 / −7.02 | +1.94 / −6.66 / −6.87 |
| **C1ns** | **× g** | **+0.36 / −0.36 / −0.22** | **−1.69 / −1.72 / −1.46** | **−2.61 / −2.11 / −0.91** |
| C1nsadd | no g | +5.07 / −5.05 / −6.02 | +2.83 / −6.28 / −7.02 | +1.94 / −6.66 / −6.87 |
| **C1nsadd** | **× g** | **+0.36 / −0.36 / −0.22** | **−1.69 / −1.72 / −1.45** | **−2.61 / −2.11 / −0.91** |
| C1gz | n/a (3-D branch) | −0.08 / −0.13 / −0.31 | −2.15 / −1.51 / −1.32 | −3.07 / −1.86 / −1.24 |
| *C1nsz* | n/a (3-D branch) | +0.00 / −0.00 / −0.00 | −2.06 / −1.38 / −1.11 | −2.99 / −1.71 / −0.90 |

## A3. Does the frozen `g` remove the +5 / −5 / −6 % structure? **Yes — most of it.**

Measured as the **K spread** (max − min across K0/K1/K2), in percentage points:

| variant | 2LPT-0 no g → × g | London-0 no g → × g | Saclay-0 no g → × g |
|---|---|---|---|
| C0 | 10.85 → **0.81**  (92.6 % removed) | 9.61 → **0.27**  (97.2 %) | 8.62 → **1.54**  (82.1 %) |
| C1g | 11.07 → **0.71**  (93.6 %) | 9.85 → **0.26**  (97.4 %) | 8.79 → **1.72**  (80.4 %) |
| C1n | 11.09 → **0.72**  (93.5 %) | 9.85 → **0.27**  (97.2 %) | 8.79 → **1.72**  (80.4 %) |
| C1ns | 11.09 → **0.72**  (93.5 %) | 9.85 → **0.26**  (97.3 %) | 8.81 → **1.70**  (80.7 %) |
| C1nsadd | 11.09 → **0.72**  (93.5 %) | 9.84 → **0.27**  (97.3 %) | 8.81 → **1.70**  (80.7 %) |
| C1gz (no g, 3-D branch) | — → 0.23 | — → 0.84 | — → 1.84 |
| *C1nsz* (no g, 3-D branch) | — → 0.00 | — → 0.95 | — → 2.09 |

**On the calibration family the frozen `g` removes 93–94 % of the K spread** (11.1 pp → 0.7 pp),
and on London-0 97 % (9.8 pp → 0.3 pp). On Saclay-0 it removes 80 % (8.8 pp → 1.7 pp) — Saclay-0
keeps a genuine residual z slope of ≈ 1.7 pp that the frozen 2LPT-0 `g` does not describe, and the
z-resolved objects do not fix it either (C1gz 1.84 pp, C1nsz 2.09 pp), so it is a Saclay-0
population/geometry difference rather than a missing completeness DOF.

**What remains after `g` is a LEVEL, not a shape**, and it is the level the C1 repair was built to
fix: C0 × g sits at **−2.2 to −5.0 %** in every block on every family, while C1nsadd × g sits at
**+0.4 %** (2LPT-0), **−1.5 to −1.7 %** (London-0) and **−0.9 to −2.6 %** (Saclay-0). The frozen `g`
and the C1 repair are therefore **complementary and non-overlapping**: `g` fixes the z *shape*, C1
fixes the *level* and the N/S-N *shape*, and neither substitutes for the other.

## A4. The main report's §4(b) reading, corrected

§4(b) said the N/S-N repair "leaves the coarse-z residual completely untouched" and called it a §16
"the defect moved elsewhere" pattern. **That reading was on C alone and is wrong for the fold.** On
the effective completeness the C1 repair leaves the calibration-family K residual at ±0.4 % with a
K spread of 0.7 pp, i.e. it moves nothing into z. The correct statement is:

> The +5 / −5 / −6 % K structure that every z-pooled C shows against `C_true` is *by construction*
> the z-shape the frozen `g` carries. It is not an unrepaired defect of the C objects; it is the
> part of completeness the fold has deliberately factored out of C and into `g`.

## A5. `C · g` exceeds 1

On 2LPT-0, `C1g[b,s]·g[b,k]` exceeds 1 in **341 of 1 291 live (s, k, b) cells**, carrying
**19.8 %** of the truth weight, with a maximum of **1.0708**. This is not an error: `g` is a per-N
z-reshape and the fold is Poisson in counts, so the product is a rate multiplier, not a probability.
It does mean two things. (i) The binomial log-loss used for model selection in §3 scores **C**, which
is a probability; it must **not** be applied to `C·g` (doing so is dominated by the clipping and
gives a meaningless answer — 42 676 vs 32 874 nats). (ii) Any future reparameterisation that treats
the effective completeness as a probability (a logit-additive z term inside C, say) is **not**
equivalent to the current `C × g` form and would be a genuine model change, not a refactor.

## A6. Is a z-resolved C still supported *on top of* `g`? Weakly — ≈ 3 σ, not 24 σ

Scored on the fold's own likelihood: held-out (sightline-half) **Poisson deviance** of
`mu = pred · truth_bks` on the fine (s, k, b) cells, 2LPT-0.

| representation | held-out Poisson deviance | Δ vs `C1g × g` |
|---|---|---|
| C0 frozen × g | 2 099.94 | **+286.35 ± 69.19** |
| C1g pooled, NO g | 2 510.42 | **+696.83 ± 158.36** |
| **C1g pooled × g** (reference) | **1 813.59** | — |
| C1nsadd (6 coef) × g | 1 806.40 | −7.19 ± 11.37 |
| C1gz z-resolved, NO g | 1 447.38 | **−366.22 ± 119.37** |
| *C1nsz* (8 coef), NO g | 1 532.60 | −280.99 ± 98.53 |

(SE = cellwise `sqrt(n)·sd` of the paired per-cell deviance difference; a crude but honest scale.)

Four readings.
1. **`g` is strongly data-supported**: dropping it costs **+696.8 ± 158.4** — the frozen z-shape is
   doing real work and must not be removed when a repaired C is installed.
2. **The C1 repair is still worth +286.4 ± 69.2 after `g`** — the level/N-shape defect is real and
   `g` does not touch it. This is the number that justifies opening C1 at all.
3. **The 6-coefficient C1nsadd is indistinguishable from the 96-cell C1g on the fold's own scale**
   (−7.2 ± 11.4), confirming §3's parsimony conclusion on the scale that matters.
4. **The residual z gain after `g` is −366 ± 119 (≈ −3.1 σ), not the −24 σ of §3 line 8.** §3's
   −1 289-nat, −24 σ figure scores C against `C_true` with no `g` anywhere, so it measures the z
   structure `g` already supplies. The honest statement for the PI is: **most of the apparent z
   signal is already in the frozen `g`; what remains is a ~3 σ effect on 100 k held-out systems, and
   it is a C2-class DOF which is HOLD.** I do not recommend opening it on this evidence.

## A7. Addendum summary (6 lines)

* **A1** The fold's effective z-pooled completeness is `C[b,s]·g[b,k]`, not C; §4/§5 compared C alone.
* **A2** With `g` included, the K residual of every C1 member on 2LPT-0 is **+0.36 / −0.36 / −0.22 %**
  (was +5.07 / −5.05 / −6.02), and C0's is −2.15 / −2.95 / −2.84 (was +2.41 / −7.49 / −8.45).
* **A3** The frozen `g` removes **93–94 %** of the K spread on 2LPT-0, **97 %** on London-0 and
  **80 %** on Saclay-0 (which keeps a real ≈ 1.7 pp residual z slope); `g` is total-preserving to
  0.02 %, so it fixes shape only.
* **A4** What survives `g` is a **level**: C0 × g = −2.2…−5.0 % everywhere; C1nsadd × g = +0.4 %
  (2LPT-0), −1.5…−1.7 % (London-0), −0.9…−2.6 % (Saclay-0). `g` and C1 are complementary.
* **A5** `C·g` > 1 in 341/1 291 cells (19.8 % of truth weight, max 1.0708) — legitimate in a Poisson
  fold, but it forbids scoring the effective completeness with a binomial log-loss.
* **A6** On the fold's own Poisson deviance: dropping `g` costs +697 ± 158; the C1 repair is worth
  +286 ± 69 **after** `g`; the extra z DOF is worth only **−366 ± 119 (≈ 3 σ)**, so §3's −24 σ z
  result is largely **redundant with the frozen `g`** and the C2 z-DOF case is much weaker than the
  main report implied. **Nothing adopted; no delivered product changed.**
