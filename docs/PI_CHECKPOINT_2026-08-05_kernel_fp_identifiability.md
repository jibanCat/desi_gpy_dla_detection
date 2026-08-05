# PI SCIENCE CHECKPOINT — 2026-08-05
## Low-N response, completeness, and false-positive separation

**Why this stops here.** The session brief says to halt at a consolidated checkpoint if the
components remain non-identifiable without a substantive prior, external calibration or new
injections. **They do.** The measurement is in §8 and it is a clean negative result, not a
defect awaiting a fix.

**Nothing here is paper-facing.** No artifact in either lineage is marked `paper_facing: true`
(verified by walking every tracked JSON). No rung 10, no campaign, no freeze, no tag, no lineage
merge, no inference PR. All values are **mock-derived** (2LPT-0 / London-0 / Saclay-0 / loa-0).

---

## 1. Pushed branch tips and issue status

| branch | tip | pushed | role |
|---|---|---|---|
| `hbi-mcmc-threeroute` | `b81be9f` | ✅ origin | inference core; kernel + FP repair |
| `lls-subdla-cddf` | `1533333` | ✅ origin | provenance, guards, tombstones, paper-safety |
| `hbi-readme-figs-fix` | `d943bfe` | ✅ (PR #21) | tutorial series — untouched |

Worktrees at the intended endpoint (9 → 3): primary, `hbi_mcmc_wt`, `hbi_tutorial_wt`.
Tracking issue **#30**, one checklist labelled engineering / scientific validation / possible
future PI decision, updated with progress.

## 2. Commits and bounded refactors completed

**Integration** (three streams, dependency order, each refereed): `c4cd67c` gate-ratification ·
`bdae1a4` adopted-basis + ONE authority source · `9461d5a` spectral-window ·
`a12385f`+`bed0483` the `lya_lyb × 0.2-dex` cell.

**Bounded refactor** (behaviour-preserving unless marked): `f88baa3` one closure metric replacing
six implementations, proven **bitwise identical on 19 packs** · `6b86728` **BEHAVIOUR CHANGE** —
the SLURM pre-flight now gates `by_snr` as the production gate does · `d5b702d` skew-ramp width
serialized rather than re-typed as a magic literal · `a3bbf23` `_SN_SKEW_MAX` pinned by a test ·
`1ab1840` `legacy_oracle`'s convention map corrected (it documented the **inverted** mapping —
measured 0.281 dex error vs 4.4e-16) and the missing D2 clamp added.

**FP repair**: `7707c8e` **BEHAVIOUR CHANGE** — `fp_ell_eff` at all four sites *including the
synthetic generator* · `2b436df` one FP term (`forward.fold_mu_fp`); the re-typed copy had already
drifted, dropping `exp(log_t)` · `b81be9f` the contract guard now verifies the repair.

**Contract**: `748ebc2`, `47897e8`, `878e087`.

## 3. The frozen forward-model and matching contract

`CDDF_analysis/hbi_mcmc/matching_contract.py` (67 tests). Six populations with **executable
predicates**, a checkable accounting identity, declared support classes, parameter classes
(fixed calibration product vs inferred nuisance), and fail-closed pack validation.

**The truth ledger closes exactly** on all six packs — but a referee established it is an
**algebraic tautology** (`T·C·ρ + T·C·(1−ρ) + T·(1−C) ≡ T` for any C, ρ, T; injecting ρ ∈ {0, 1,
random} leaves the residual at 0.0). It detects only a shape/broadcast crash. Now stated as such,
with real value guards added on the three terms individually.

**Retracted after adversarial review, both recorded with measurements:**
- The "attainable efficiency" was presented as a **sharper bound**. It is the value at one point
  in an **unbounded** nuisance space. The adopted-geometry "infeasible" verdict is withdrawn: the
  gap closes at **0.56σ** of the FP's own Poisson width, 0.87σ of a uniform transfer shift, or
  2.1σ of one response nuisance. The **unpadded** refutation survives — on 2lpt0_v11 as a genuine
  bound (required 0.9958 > supremum 0.9938), on the others at prior cost χ² ~ 10⁴.
- The **one-sided BAL veto** in the FP weight: arithmetic exact (351/2255 sightlines, 19/89 FPs)
  but the premise fails. Tested on the full loa-0 catalogue — **2,378 events, 27× the
  statistics** — there is **no BAL signal at all** (ratio 1.0027, z = +0.05); the 19-of-89 excess
  is 1.5σ. The support-matching observation is kept; the magnitude, direction and proposed fix
  are retracted.

## 4. The measured kernel-support boundary

**The kernel is genuinely measured on [19.5, 21.1), weakly measured on [21.1, 21.5), and clamped
or unsupported outside.** Criterion stated before application, applied per (SNR,z) cell, three
conjunctive arms: measured resolution `d* ≤ 0.05 dex` (a detection curve, so agreement cannot be
manufactured by sparsity), agreement `|Δμ| ≤ 0.05 dex` in **every** cell, and interpolation
(the bin lies inside that cell's fit range).

| target | verdict | evidence |
|---|---|---|
| **19.5** | **YES** — DIRECT in 9/9 cells | n = 410–1241/cell, `d*` 0.020–0.050, \|Δμ\| 0.028–0.049 |
| **19.3** | **NO as delivered** | data adequate (266–907/cell) but 7/9 cells clamp, \|Δμ\| up to **0.119 dex**; τ = 0.44–0.52, half the response censored |
| **19.0** | **NO** | 9/9 clamped, \|Δμ\| **0.262–0.389 dex**; τ = 0.186 (81% censored); 56–88% of the model's own column falls below the observed floor |
| **21.3** | **WEAK YES** — a constant, not a function | no cell has an anchor above 21.2164 |
| **21.5** | **NO at basis resolution** | 43–84 rows/cell, \|Δμ\| up to 0.099 dex |
| **>21.6** | **NO** | 15–32 rows/cell; `d*` 0.095–0.185, so nothing could have been falsified |

🔴 **`resp_N_fit_range` is a binning knob, not a measurement.** Re-running on *identical* data with
only the septile count changed moves the "calibrated covariate range" from [19.34, 21.22]
(`n_N_cells=7`) to [19.10, 21.62] (`n_N_cells=40`), with no sub-bin ever below `min_count`. **It is
the clamp boundary** (`forward.py:294-306`) and therefore load-bearing. Making the schema finer
makes 21.5 *worse*: the clamped up-bias moves +0.083 → +0.173 dex against a measured +0.026.
`reporting.py:159` and `RESPONSE_ANCHOR_MEASURED` currently present it as measured.

🔴 **`min_count = 60` is inert.** Sweeping 30 → 200 changes the fitted surfaces **exactly zero**
(the smallest septile sub-bin holds 853 rows). Any argument resting on it — including earlier
feasibility reasoning in this project — needs re-grounding. The binding constraints are truncation
and fit design, never statistics, anywhere on [19.0, 21.6].

**Extending below 19.5 requires THREE coupled changes** — `xhat_floor` (`znz_kernel.py:1675,1730`),
`cmp_min_pred_nhi` (`cddf_catalog_hbi.py:752`), and `host_truth_floor` (`ab_loa0_fp_baseline.py:190`).
Moving any one alone breaks the C·K factorisation. Moving all three **replaces the currently-DIRECT
[19.5,19.7) column** (TV 0.222 against a null p95 of 0.07–0.14) — a re-measurement of the headline
region, not an extension of it.

## 5. Prior-assisted, extrapolated and clamped regions

- **21.6 ceiling**: contains **0.384–0.559 dex of extrapolated response**. `[21.5,21.6)` is inside
  the authorized reporting window but **is not measured at basis resolution** — the window ceiling
  and the support ceiling are 0.1 dex apart, in the unhelpful direction.
- **100% of the sub-floor population's kernel** runs on a frozen/extrapolated covariate; the top
  three in-window basis bins are on frozen covariate in 8–9 of 9 cells.
- Per-bin, in the repaired fold: `[19.7,19.8)` is **60.6% pad-sourced** and 27.6% clamp-sourced;
  `[21.5,21.6)` is **96.3% clamp-sourced**. Both ends of the reporting window are held up by
  support that is either measured-but-unreported or frozen-not-measured.
- The unclamped deg-2 extrapolation is **refuted by the data at both ends** (+17.4σ at
  `[21.5,21.6)`; −14.6σ at `[19.0,19.1)`). **The D2 clamp is the right call and is priced**: its own
  error is ≤0.036 dex to 21.3, ≤0.054 to 21.5, ≤0.099 to 21.9, against the unclamped surface's
  +0.10 to +0.55 dex.
- **A prior is needed only above ~22.1** (n = 1–7/cell, empty cells at 0.1 dex; and 17–38% of the
  top bin's kernel mass falls off the top of the observed grid). **Below 19.5 no prior is needed** —
  the data exist; what is needed is a coupled re-extraction.
- 🔴 **3,340 analysis detections (4.33% of the FP-removed set) originate below the basis floor of
  19.0.** No f on [19.0, 22.4] can produce them, and they are invisible to the response fit because
  `host_truth_floor = 19.0` censors them.

## 6. The corrected loa-0 FP model

**The defect, confirmed three ways and adversarially attacked twice:** `forward.py` folded
`fp_w · exp(t) · λ · E`, omitting `fp_ell_eff`, while the likelihood constrains
`fp_counts ~ Poisson(ell_eff·λ)` and the loa-0 product defines
`μ_FP = (N_prod/N_sl_loa0)·N_FP·(1−η̄)`. Measured: implemented **1086.687** vs contract
**14767.961**, ratio **13.589891949532 == `fp_ell_eff` exactly**; `fp_w · fp_ell_eff = 2255.0 ==
n_sl_loa0` exactly on all six packs.

🔴 **No test could have caught it**: `pack.synthetic_pack` shared the same convention, so no
synthetic rung and no SBC could detect it. **Repaired at all four sites including the generator**,
with `w·ℓ == n_sl_loa0` now a pinned invariant and a guard that raises if the omission returns.

**Stratification: NOT implemented, and the reason is the finding.** The schema would carry it
(`fp_E_alloc` is already `(Kf,S)`), but:
- **96.16%** of loa-0's op+Lyα FPs (2229 of 2318 in-window) sit at **N̂ < 19.5** — off the pack's
  observed grid. Only **89** are on it.
- In-support coarse-z shape **0.4831/0.4045/0.1124** vs below-floor **0.6716/0.2638/0.0646**:
  2×3 homogeneity **χ²(2) = 13.807, p = 0.0010**. The z-shape is *different* above and below the
  floor — occurrence #12 of the one-sided-support class.
- The in-support shape is unusable on its own: 89 counts over 15 z × 8 SNR, 47/120 cells nonzero,
  max cell 5; its own offset from the imposed shape is 2.2σ.
- **Both measured shapes make z-closure worse** on all three mocks (`by_z` χ²/dof 9.94/9.72/7.85 →
  15.65/18.42/18.92 or 14.74/17.30/16.79).

Choosing which support to transfer from is a modelling decision with a **p = 0.001** conflict
between the candidates. Stopped there deliberately.

## 7. Held-out transfer error and nuisance calibration

🔴 **The corrected μ_FP exceeds the mock's entire false-positive supply — on all three mocks, far
worse off the calibration twin.** Re-measured on the 17.2-truth-floor bundle, every partition
summing exactly to the on-grid total:

| mock | on-grid | P1 [19.0,19.7) | P2 [19.7,21.6) | ≥21.6 | <19.0 | unmatched | μ_FP | excess |
|---|---|---|---|---|---|---|---|---|
| 2LPT-0 | 88053 | 15438 | 55058 | 497 | 3200 | 13860 | 14767.96 | **+907.96 (+6.55%)** |
| London-0 | 87831 | 15834 | 59186 | 602 | 2611 | 9598 | 14716.38 | **+5118.38 (+53.33%)** |
| Saclay-0 | 86745 | 15733 | 57213 | 539 | 2668 | 10592 | 14707.06 | **+4115.06 (+38.85%)** |

`unmatched` **over-counts** true forest FP (blends, second candidates), so these are **lower
bounds**. No parameter can fix this: it bounds the loa-0 template itself.

🔴 **London-0 and Saclay-0 are NOT transfer tests of the nuisances.** Bit-for-bit comparison of the
three packs: `molly_n_det`, `molly_n_tot`, `molly_nhi_edges`, `g_grid`, `g_occupancy`,
`resp_*_coef`, `resp_N_fit_range`, `fp_counts` (89) and `t_sigma` are **IDENTICAL** — one frozen
2LPT-0 calibration spliced into every pack. Only `counts`, `dX`, `truth_counts`, `fp_w`,
`fp_ell_eff`, `fp_E_alloc` differ. **A common held-out calibration bias is invisible by
construction** — which is why the degeneracy geometry matches to three significant figures across
mocks. **Any "validated on three mocks" statement about the nuisances is unsupported.**

**`resp_fitcov_diag` is absent from all six packs**, so `build_consts` falls back to
`_DEFAULT_FITCOV_DIAG = (0.02², 0.10²)`. The `psi_k_delta` prior — whose width sets the cost of
closing the residual — is a hard-coded guess, not a measured fit covariance.

## 8. 🔴 IDENTIFIABILITY — THE DECISIVE RESULT

**Populations A (sub-floor migration) and B (forest FP) are NOT separately identifiable from these
observables.**

| evidence | measurement |
|---|---|
| principal angles, pad ⟷ [window ∪ FP] | **16 of 75** pad directions within **1°**; smallest **0.0176°** (cos = 0.99999995) |
| pad ⟷ FP alone | smallest **0.668°**; 3 within 4.4° |
| pad ⟷ window alone (basis truncation, geometric form) | **15 of 75** within **0.283°** |
| adding the 3 coarse-z transfer factors `t_K` | 0.0176° → **0.0175°** — **no discrimination whatsoever** |
| "FP only, pad off" on data injected with the opposite truth | fits at **Δdev = 41**, i.e. **0.6σ** against survey Poisson noise, while manufacturing a **9.6× FP error** |
| two clamp conventions the codebase itself calls defensible | **Δdev = 0.46**, giving a **24× different** FP amplitude and −24% to −31% on the sub-floor total |

**They become identifiable under exactly one substantive addition: freezing the FP's (c,s) shape
to its external calibration.** That single restriction collapses the admissible FP total ~**50×**
(Δdev ≤ 4 range 10,000–21,000 → 0–300), makes the profile exactly convex, and reduces the induced
spread on reported dN/dX[19.7,21.6] from **16–21% to 0.2–0.4%** and on the z-tilt from **9–10% to
0.1–0.5%**, consistently on all three mocks.

**The production model does not impose it.** `model_a.py:219-220` gives each of 232 FP cells a free
logit at `fp_shape_sd = 3.0`, and **149 of the 174 live cells carry zero calibration counts**.

**FP changes masquerade as absorber evolution** — the result that matters most for the z-objective:
- An FP z-error the loa-0 data **cannot exclude** manufactures **5,958–11,401 phantom sub-floor
  absorbers** from a truth containing ~zero.
- The imposed z-shape is **already marginally refuted** by loa-0's own counts: Pearson χ² = 5.767,
  2 dof, **p = 0.056**. The test that could refute it has power **0.104** at ±10%, **0.277** at
  ±20% — while `t_sigma = [0.127, 0.165, 0.100]` asserts the shape is right to 10–18%.
- **The FP z-allocation is invisible to the ratified gate.** Because `Σ_k fp_E[k,s] = 1` per
  populated stratum, it cancels out of the n̂ marginal: window χ²/dof agrees to ≤9.65e-16 relative
  under all three allocations. The only arm that moves is `by_z` max\|z\| (7.64 → 3.75 or → 10.26)
  — **and that arm is restated-but-not-ratified.** The leg with authority behind it is blind to the
  one statistic that separates A from B.

**Even the correct model under-recovers T_A by 7.7–8.5%** on noiseless data, consistently, from the
pad↔window degeneracy. That is a floor on sub-floor accuracy independent of the FP.

🔴 **Correction:** the previously-circulating Δdeviance +2519/+3478/+3089 ("the survey decisively
prefers sub-floor absorbers over 13.6× more forest FP") reproduces the **frozen-shape** case. Under
the production model's **free** shape the survey prefers the **opposite** convention by 165–265
deviance units. The claim was conditional on a restriction the model does not impose.

## 9–10. Closure before and after, integrated and z-resolved

Adopted config, zero-sampling truth fold, all three mocks:

| mock | window χ²/dof | full χ²/dof | full mu/obs | window z_tot |
|---|---|---|---|---|
| 2LPT-0 | 56.58 → **22.22** | 441.41 → 19.07 | 0.8462 → **1.0016** | +12.86 → −4.62 |
| London-0 | 40.16 → **28.39** | 224.04 → 29.04 | 0.8948 → **1.0501** | +7.86 → −8.94 |
| Saclay-0 | 44.21 → **25.77** | 305.77 → 20.33 | 0.8711 → **1.0281** | +10.55 → −6.68 |

**The ratified gate χ²/dof ≤ 3 still fails on all three.** But the failure has changed character:

- 🔴 **The floor effect was largely this bug.** `[19.7,19.8)` carried **55.3/58.2/57.0%** of window
  χ² before; **5.8/10.2/9.6%** after (z +24.38 → −4.93 on 2LPT-0).
- **The residual flipped sign**: +4.85/+2.94/+3.98% → **−1.80/−3.45/−2.61%**. The corrected FP
  overshoots.
- **New dominant bin** `[19.9,20.0)`: 24.0/40.4/35.0% of window χ². The FP over-fills 19.9–20.2.
- **Coherent signed pattern** persists: over-prediction at 19.9–20.5, under-prediction at
  20.6–21.5; 4–7 sign runs.
- **z-resolved**: low-z now closes (+0.36) but **high z coherently over-predicts** (ratio 1.08–1.24
  for z ≥ 3.0 on all three). `by_z` χ²/dof = 9.94/9.72/7.85.
- 🔴 **The leading residual is now the SNR axis, not N̂.** `by_snr` χ²/dof = **36.61/62.91/54.61** —
  *worse* than the n̂ gate value on all three — with a coherent monotone tilt: SNR [2,3)
  under-predicted (z +8.14/+9.67/+10.93), SNR ≥ 5 over-predicted (z −4 to −9.8). Separately
  measured, the `by_snr` arm is violated at **6.9×–10.8×** tolerance on all 18 committed packs.
- **Only 32.6% of μ_FP lands in [19.7,21.6]**; 67.4% lands in the masked [19.5,19.7) strip. The FP
  term is mostly constrained by bins that are not reported.

🔴 **A committed science number moved, and it reassigns blame.** Rung-9 signature, v1 2LPT-0 pack,
clamp off: bottom-bin ratio **0.1655 → 0.7042**, total **0.7307 → 0.8860**; the [19.5,19.6) deficit
falls **6.0× → 1.4×**. **Most of the reported low-N deficit was the FP normalisation, not D1 basis
truncation.** This weakens the evidentiary weight previously assigned to D1. (The basis pad is
still supported independently: removing it costs +4,242 deviance on 75 parameters.)

**Stale committed artifacts** (not regenerated, per instruction): `adopted_config_closure.json`
(427 `chi2_dof` entries), `spectral_window_study.json` and `_bw0p2.json` (147 each),
`d1_basis_pad_ladder.json` (66), `rung9_forward_selftest.json` (6),
`posterior_synthetic_smoke.json` (2).

## 11. Lyα-only versus nominal window

**0 of 24 configurations close** across the matched cross (2 windows × 2 basis widths × 3 mocks).

| basis | best margin over the ratified gate | worst |
|---|---|---|
| 0.1 dex | **9.8685×** (`london0 \| lya_only \| clamp=hi`, χ²/dof 29.6056) | 35.53× |
| 0.2 dex | **9.8009×** (same config, 29.4027) | 30.37× |

`lya_lyb` is worse than `lya_only` on all three mocks at both clamps — unanimous 6/6 on the
scale-free `rms_frac_dev` and 6/6 on χ²/dof, the same verdict as at 0.1 dex. **Neither window
closes; Lyα-only does not solve the model.** The blue-end campaign remains unfunded; the standing
result is that a large blue-end N_HI shift is unsupported at current sensitivity while shifts below
~0.09 dex remain insufficiently constrained.

## 12. Sensitivity

- **Basis width** 0.1 → 0.2 dex improves χ²/dof in all 12 window cells by 0.69–15.85%, but
  `rms_frac_dev` gets **worse** on the *best* configuration (0.084360 → 0.086348) — the gain is in
  the level, not the shape — and the excluded ≥21.6 residual moves **further from 1 in all 12**.
- **Pad**: removing it costs +4,242 deviance on 75 parameters (decisively refuted). Pad effect on
  window χ²/dof: 219.80 → 56.58 (2lpt0), 170.29 → 40.16, 185.24 → 44.21.
- **Binning**: `n_N_cells` is the dominant knob (§4). `min_count` is inert. `deg_N` is the whole
  story outside the anchors (μ_b at N=22.0 = −0.134 / +0.524 / −0.702 for deg 1/2/3).
- **Model complexity**: the FP shape freedom alone drives a ~50× swing in the admissible FP total
  and 16–21% on the reported estimand (§8).

## 13. Posterior predictive, SBC, coverage

**Not applicable — sampling was never justified and was not run.** Forward closure fails, transfer
is not credible, and the components are not identifiable. Per the brief's own gate, posterior
production remains blocked. Rung 10 stays gated.

## 14. Referee verdicts

Every implementation stream was refereed by an independent agent that re-ran the code rather than
reading it. Referees **overturned four claims**, including two the orchestrator had already
reported: the "sharper bound" (BLOCKING), the BAL effect, the `a_fp` deviance verdict, and the
cosine-similarity identifiability statistic (which spans 0.078–0.760 depending on the assumed FP
shape — the thing being inferred). Implementation agents in turn corrected referees twice where
their numbers did not reproduce. Mutation testing throughout; several mutants **survived** first
passes and were reported rather than papered over.

Fail-open guards closed this session, each proven open beforehand: the tombstone registry (a rogue
tombstone smuggled a float science value and `paper_facing: true` past 31 green tests); the CLEAN
record's recoverability claim; the ratification scanner's **116 false alarms on correctly-retracted
code** (now 0, detection retained, both directions pinned by a corpus test); the SLURM pre-flight
that did not gate `by_snr`; and the prose guard that read a correctly-`_restated(...)` entry as an
unqualified ratified claim.

## 15. Scientifically citable results

1. The forward model **does not close** in any configuration tested: 0/48 adopted-config, 0/24
   window × basis. Margins 9.80×–35.53× the ratified gate.
2. The **kernel is measured on [19.5, 21.1)** and nowhere lower; the 21.6 ceiling carries
   0.384–0.559 dex of extrapolated response.
3. The **FP normalisation defect** and its repair, with before/after closure on three mocks.
4. **A and B are not separately identifiable** without freezing the FP shape to external
   calibration (§8) — with the degeneracy directions, angles and profile likelihoods that
   establish it.
5. The **loa-0 template over-predicts every mock's FP supply**, by 6.55% on its own twin and
   38.85–53.33% on the held-outs.

**Not citable:** any nuisance value "validated on three mocks" (§7); any statement resting on
`resp_N_fit_range` as measured; any dN/dX from the current model in [19.7,21.6] (it carries the
unmodelled-FP bias one-sided).

## 16. Remaining distinct science options

**A. Freeze the FP (c,s) shape to the loa-0 calibration.** Buys identifiability (§8) at the cost of
adopting a template that already over-predicts every mock's FP supply and whose z-shape is
refuted at p = 0.001 across the floor.
**B. New sub-floor injections** below 19.5 with known abundance — the only route that measures A
directly rather than inferring it.
**C. A z-resolved loa-0 FP calibration on the observed grid.** Currently 89 counts; would need a
larger loa-0 volume.
**D. Extend completeness below N̂ = 19.5** — requires the three coupled changes (§4) and
re-measures the currently-DIRECT headline column.
**E. Drop quantitative sub-DLA inference from Paper 1** and report only where the kernel is
measured and the components separable.
**F. Marginalise over z** — the fallback, but §8 shows FP errors masquerade as absorber evolution,
so this hides rather than resolves the problem.

## 17. Recommended PI choice, and its consequence for Paper 1

**Recommend B + E, with A as an explicitly-labelled sensitivity, and defer C.**

*Reasoning.* Option A is the only route that makes the current data identifiable, but it buys that
by asserting a template that fails three independent checks — it exceeds every mock's FP supply,
its z-shape differs across the observed floor at p = 0.001, and its in-support calibration is 89
counts. Adopting it as primary would convert a measurement into an assumption at exactly the point
the analysis is weakest. Option B is the only route that breaks the degeneracy with data rather
than prior, and the injections needed are sub-floor systems in an existing mock — far cheaper than
the ~17,000 CPU-h blue-end campaign, which stays unfunded.

*Consequence for Paper 1.* dN/dX over the region where the kernel is **measured and the components
separable** is defensible; the sub-DLA band that depends on sub-floor migration is not, until B
lands. z-evolution should stay in scope as an objective but **cannot be claimed** on the current
model — the FP z-shape is invisible to the ratified gate and an FP z-error the data cannot exclude
manufactures thousands of phantom absorbers.

**This is a recommendation, not a decision.** The genuinely PI-level items are: whether a
prior-assisted kernel extension is adopted; the physical definition of the FP population; whether
sub-DLA inference and z-evolution remain in Paper 1; and whether new injections are commissioned.

## 18. The smallest next implementation phase after the ruling

**If B (new injections):** specify the injection sub-floor N_HI distribution and count, extend
`host_truth_floor` and `xhat_floor` together with `cmp_min_pred_nhi` (§4), re-extract, and re-run
the injection–recovery battery of §8 to confirm the degeneracy is broken *before* any posterior
work. Estimated cost is extraction-dominated (~21 s/mock/pack), not sampler-dominated.

**If A (freeze the FP shape):** implement `fp_shape_sd → 0` as a config-level option (not a
default), re-run zero-sampling closure and the §8 profile, and propagate the loa-0 supply-ceiling
violation as a stated systematic rather than absorbing it.

**Either way, first:** the `by_snr` axis is now the leading residual (χ²/dof 36.6–62.9, worse than
the n̂ gate) and has never been diagnosed. That is the cheapest next measurement and it does not
depend on the ruling.

---

## Standing constraints, unchanged

No push of a new inference PR; no lineage merge; no freeze or tag; no rung 10; no major campaign;
no unblinding; nothing marked `paper_facing`. Adopted working baseline frozen for controlled
comparison: floor 19.7, upper limit 21.6, 0.2-dex latent basis, provisional 19.0/molly172 pad,
Lyα-only 1025–1216 Å primary with the nominal Molly window as reference, dN/dX as the target
observable, no Ω_HI headline, no LLS population in Paper 1, 2LPT-0 as calibration and London-0 /
Saclay-0 as transfer tests.
