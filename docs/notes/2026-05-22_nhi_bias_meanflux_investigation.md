# NHI-bias investigation: mean-flux is NOT the cause (the bias is intrinsic / prior-edge)

> **Date**: 2026-05-22 (GreatLakes). Falsifiable-test log — read the prose, not just the tables.
> **Question**: a **+0.05–0.06 dex log-NHI over-estimate** (ΔNHI = NHI_GP − NHI_true) on the
> London-0 mock (V1 production config). What causes it, and does it require a model retrain?
> **Bottom line**: it is **intrinsic to the GP single-absorber model (prior-edge effect)** —
> *not* the mean-flux model, *not* NUM_FOREST_LINES, *not* the matcher, *not* the trained model.
> **No production-config change is warranted** by this investigation.

## TL;DR — what was ruled out (each with a controlled test)

| Hypothesis | Test | Verdict |
|---|---|---|
| Pure-Python Voigt fallback | `_voigt.so` loads; no fallback warning; `dla_gp.py:1021`→ctypes | **C Voigt in use; not it** |
| Matcher artifact (nhi-desc ordering) | z-only matcher ≡ nhi-desc (+0.067 vs +0.066); single-DLA sightlines still +0.049 | **Real bias, not matcher** |
| NUM_FOREST_LINES (training=31 vs inference=3) | controlled τEB-off NF=3 vs NF=31, same slice: +0.044 vs +0.046 | **Identical; not it** (higher Lyman lines' absorbers are beyond the QSO ⇒ zeroed in the forest, so GP-3line ≡ GP-31line ≡ Turner-Lyα) |
| Trained-model-specific (2lpt de-forest) | eBOSS DR16Q model (independently trained) on London: +0.057–0.060 | **Same bias ⇒ not model-specific** |
| Inference mean-flux mismatch | eBOSS model + Kim (τ_eff(2.5)=0.229) vs + Kamble (0.298): +0.060 vs +0.057 | **±30% mean-flux ⇒ ΔNHI unchanged; not it** |
| Training de-forest residual in `mu` | re-derive `mu` with the clean loa-0 Lyα de-forest (0.01258/2.385, +2.84% forest mu), hold M/Ω, re-infer (job 50705918) | **CONFIRMED not it**: ΔNHI=+0.044, *identical* to the Turner-deforest baseline +0.044 (same 72 TP) |

**Why mean-flux doesn't move NHI** (physical): the mean-flux *level* is absorbed into the
continuum/normalization fit, while NHI comes from the Voigt **shape**, so a smooth mean-flux
shift barely changes NHI. Confirmed empirically by the eBOSS Kim-vs-Kamble null.

## Mean-flux characterization (the thread that clarified, but isn't the cause)

Measured τ_eff(z) = −ln⟨F⟩ (F = flux / TRUE_CONT) in the forest [1040,1185] Å, fit τ₀(1+z)^β:

- **GP mean-flux model = Turner24 in the forest.** Computed with the per-QSO z_qso (as
  `null_gp.py:249` applies it), the 3-line and 31-line totals equal the Turner Lyα-only term to
  machine precision. (An earlier curve used a fixed z_qso=4.5 and wrongly kept higher lines —
  that was a bug, since corrected.)
- **Full-mock (790k QSOs) clean Lyα mean-flux (2lpt loa-0):** τ₀=0.01258, β=2.385,
  τ_eff(2.5)=0.250 — **+9% above Turner** (0.229), with a much **flatter slope** (β 2.385 vs 3.62).
  The earlier "+23–28%" figure was the **contaminated loa-124** value (τ₀=0.0136, β=2.45,
  τ_eff(2.5)=0.293); loa-124−loa-0 ≈ +17% is **DLA/BAL-dominated contamination** (quickquasars
  metals are only ~0.7% per the `--metal-strengths`).
- **Literature consistency (verified, arXiv):** Turner+2024 (2405.06743, LyCAN/DESI-Y1,
  0.00246/3.62) and Kim+2007 (0711.1862, 0.0023/3.65) agree (both **metal- & continuum-corrected**).
  Kamble+2020 (1904.01110, 0.00554/3.182) is 2.25× higher only because it is **uncorrected**
  (metals left in τ_eff + a continuum the authors flag as biased low) — same functional form, not
  a physical disagreement. The mocks (uncorrected, metals-in) naturally track Kamble, not Turner.

## Conclusion + production implications

- The bias is **intrinsic — the GP single-absorber over-estimate / prior-edge effect**
  (sharp DLA prior at log NHI=20.3 inflates the posterior near 20.3). See
  `feedback_dla_prior_edge_bias` (extend the NHI prior past 20.3 + post-cut). τ-EB recovers part
  of the apparent bias by per-spectrum adaptation but is not addressing a mean-flux error.
- **Production config UNCHANGED.** The V1 settings (Turner mean-flux, NF=31 [matches training,
  numerically a no-op in the forest], τ-EB on, MAX_DLAS=4) stand. The mean-flux investigation
  did not surface a production fix.
- **Retrain NOT warranted — CONFIRMED.** The mu-only test (clean loa-0 de-forest mu, M/Ω held)
  gives ΔNHI=+0.044, *bit-for-bit the Turner-deforest baseline* — so the training de-forest residual
  is also ruled out. Every mean-flux lever (training + inference) is null; the bias is intrinsic.

## Tooling added (this investigation)

- `examples/measure_mock_mean_flux.py` — τ_eff(z) from mock TRUE_CONT vs Turner/GP curves
  (per-QSO z_qso; `--no-gp` fast path for full-mock runs).
- `examples/measure_dla_pair_clustering.py` — 1+ξ(Δv) DLA-pair clustering from truth.
- `examples/dla_truth_diagnostics.py` — GP-vs-truth ΔNHI / Δz / pair-Δv diagnostics.
- `examples/make_snr_cat_from_processed.py` — small snr_cat from per-file processed h5s (no combine).
- Test configs: `london0_gl_eboss_kamble_notaueb.env` (eBOSS + correct Kamble mean-flux),
  `london0_gl_muhybrid_notaueb.env` (mu-only retrain test), plus the τEB-off NF/eBOSS variants.

## Incidental code fixes (separate commits)

- `constants.py`/`constants_highz.py`: Turner24 Lyα slope 3.182→3.62 (was a Kamble legacy value;
  the `Lyman_series` table is unused by inference).
- `run_bayes_select.py`: `--prev_tau_0`/`--prev_beta` CLI defaults 0.00554/3.182 (Kamble) →
  0.00246/3.62 (Turner); production always overrides via config (safety fix only).
- `smoke_one_spectrum.py`/`demo_desi_spectrum.py`: 0.0023/3.65 mislabeled "Kamble+2020" → Kim+2007.
