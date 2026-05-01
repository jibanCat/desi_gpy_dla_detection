# Trained GP models — what we have, what they encode, where they came from

> One reference doc covering the **5 trained GP forward-model files
> currently on this filesystem**, what each was trained on, and how
> the learned hyperparameters compare. Companion to
> `docs/notes/2026-05-01_tau_factor_distributions.md` (the τ-EB
> measurements that revealed the LOA-vs-mock divergence).
>
> Source: `examples/compare_trained_gp_models.py` extracts μ, ω,
> τ₀, β, c₀ from each ``.h5`` and writes
> `docs/story_figures/trained_gp_models_compare.png` and
> `docs/story_figures/trained_gp_models_hyperparameters.png`.
> Re-runnable any time as new training jobs land.

## The 5 models

| # | Path | Trainer | Trained on | Epochs | Notes |
|---|---|---|---|---:|---|
| 1 | `pscratch/.../learnlogs/model_epoch_920.h5` | v1 (matlab → torch port) | real DESI Y3 LOA | 953 | **Current production** (Apr 25) |
| 2 | `GP_trained/loa_no_dla_no_bal_52198069/model_epoch_1499.h5` | v2 | real LOA, **DLAs (NHI ≥ 20.3) + BALs masked**, sub-DLAs + LLS kept | 1500 | NERSC train, Apr 30 |
| 3 | `GP_trained/loa_no_hcd_with_bal_52198070/model_epoch_1499.h5` | v2 | real LOA, **all HCDs (NHI ≥ 17.2) masked**, BALs kept | 1500 | NERSC train, Apr 30 |
| 4 | `pscratch/.../v2_runs/2lpt_loa0_48938765/model_epoch_0799.h5` | v2 | 2lpt mock, loa-0 (forest-only by construction) | 800 | GreatLakes train, Apr 29 |
| 5 | `pscratch/.../v2_runs/2lpt_loa124_nohcd_nobal_48938766/model_epoch_0799.h5` | v2 | 2lpt mock loa-124, HCDs + BALs masked | 800 | GreatLakes train, Apr 29 |

The v1 trainer is the original Ho+2020 / Bird+2017 codebase ported to
torch. The v2 trainer (commits in `gpy_dla_detection/training/`) was
written for the GreatLakes session and uses centered de-forest +
slightly different normalization. As a result **the v1 production
model and the v2-trained models live on different normalization scales**
(see panel C of the hyperparameter figure: production c₀ = 0.17 vs
others 0.04 or below). This matters when comparing μ values directly,
but the high-level shape of the bias-fix story is the same regardless.

## Hyperparameter side-by-side

![Hyperparameters](../story_figures/trained_gp_models_hyperparameters.png)

| model | τ₀ | β | c₀ | ω̄ | loss[end] | epochs |
|---|---:|---:|---:|---:|---:|---:|
| PROD_y3_LOA | 0.00210 | 2.41 | 0.1738 | 1.682 | (v1 scale) | 953 |
| LOA_no_dla_no_bal | **0.00480** | 3.39 | 0.0245 | 2.081 | 2597.4 | 1500 |
| LOA_no_hcd_with_bal | ~0 | **3.62** | 0.0414 | 1.382 | 2077.8 | 1500 |
| MOCK_2lpt_loa0 | 0.00189 | 4.04 | 0.0021 | 1.618 | 2427.5 | 800 |
| MOCK_2lpt_loa124_nohcd_nobal | 0.00120 | **4.59** | 0.0009 | 1.463 | 2199.3 | 800 |

Reference: **Turner+2024 τ₀ = 0.00246, β = 3.62**. The training pipeline
de-forests at Turner before fitting, so each model's learned (τ₀, β)
is the *residual* it found after Turner subtraction.

### β (forest opacity power-law) — the cleanest signal

The β ordering is monotonic from real-LOA to mock:

```
PROD_y3_LOA            β = 2.41   ← lower than Turner; pre-fix-era v1 trainer
LOA_no_dla_no_bal      β = 3.39   ← close to Turner (real LOA)
LOA_no_hcd_with_bal    β = 3.62   ← exactly Turner (real LOA)
MOCK_2lpt_loa0         β = 4.04   ← higher (forest-only mock)
MOCK_2lpt_loa124       β = 4.59   ← highest (mock with HCDs/BALs masked)
```

**Mocks bake in a steeper β than real DESI Y3 forest.** Both 2lpt
training runs (independent of input filtering) settle at β ≈ 4.0–4.6
vs LOA's β ≈ 3.4–3.6. This corroborates the finding from the τ-EB
measurements: mocks have systematically steeper forest-opacity
evolution with z than real data. **β = 3.62 = Turner+2024** for the
LOA-with-BALs model is suggestive that Turner+2024's β was originally
calibrated on similar DESI data including BALs.

### τ₀ (mean-flux opacity residual after Turner deforest)

```
PROD_y3_LOA            τ₀ = 0.00210  ← close to Turner (v1 LOA, current production)
LOA_no_dla_no_bal      τ₀ = 0.00480  ← 2× Turner (cleanest LOA still has residual)
LOA_no_hcd_with_bal    τ₀ ≈ 0        ← Turner is exactly right (BALs absorbed it)
MOCK_2lpt_loa0         τ₀ = 0.00189  ← below Turner
MOCK_2lpt_loa124       τ₀ = 0.00120  ← well below Turner
```

τ₀ residuals after Turner are small (< 2× Turner), as expected since
Turner was calibrated on similar real data. Mock-trained models pick
*lower* residual τ₀ but compensate with higher β — they trade off
mean-flux scale for steeper z-evolution.

## Learned μ + ω + loss — split into v1 / v2 panels

![μ + ω + loss comparison](../story_figures/trained_gp_models_compare.png)

> **Why the figure is split into v1 vs v2 panels (not all in one)**:
> the two trainers use different normalization conventions, so the
> absolute numerical values of μ, ω, and loss live on different
> scales by construction. Same physics, different conventions —
> values within each side are directly comparable, but cross-side
> comparison requires the conversions below.

### Why v1 production μ ≈ 1 (with peaks at Lyα), v2 μ in absolute flux units

**Correction (2026-05-01)**: an earlier draft of this section claimed
v1 doesn't centre and v2 does. That was wrong — **both v1 and v2 apply
inverse-variance-weighted mean centering**, and in BOTH the μ stored
in the .h5 is the centring target (passed at construction, NOT learned
as a parameter):

  - v1 `learn_qso_model.SpectrumProcessor.center_fluxes` (line 366):
    > `Centers fluxes by subtracting the inverse-variance weighted mean.`
  - v2 `training/dataset._weighted_mean_centering`: same operation.

The actual difference is v1 has an **extra per-spectrum normalization
step** before centering, which v2 skips:

| step | v1 (`SpectrumProcessor`) | v2 (`training/dataset`) |
|---|---|---|
| 1. mask high-noise pixels | yes | yes |
| 2. interpolate to rest grid | yes | yes |
| 3. **per-spectrum normalize by median in [1425, 1475]** | **YES** (`normalize_spectra`) | NO |
| 4. de-forest at fixed Turner+2024 | yes | yes |
| 5. inverse-variance-weighted mean centering | yes | yes |
| 6. μ saved to .h5 | the population mean of *normalized* fluxes (≈ 1, with QSO emission peaking above) | the population mean of *absolute* fluxes |

So the v1 production μ ≈ 1 baseline is because v1 normalizes-each-spectrum-to-1
*before* centring → centring subtracts a population mean that's also ≈ 1.
v2 skips the per-spectrum normalize step → centring subtracts a population
mean in absolute flux units.

Same physics. Apples-to-apples comparison would require dividing v2 μ by
the median flux in [1425, 1475], which we haven't done because the v2
trainset already discarded the per-spectrum medians.

### ⚠ BUG — v2 preload skips per-spectrum normalization

**2026-05-01, found by jibanCat raising a sanity-check question**:

The v2 preload scripts (`preload_spectra/preload_loa_real.py`,
`preload_spectra/preload_2lpt_simple.py`) and the alternative
`preload_spectra/prepare_trainset.py` **do not apply per-spectrum
median normalization** before saving the trainset.h5. They only
mask + interpolate.

Verified empirically on the 2lpt v2 trainset:
- `fluxes` range: −6.87 to 134.76 (raw DESI absolute flux)
- Per-spectrum median flux at rest [1100, 1180] Å: 5th pct = 0.15,
  95th pct = 6.12 → **42× dynamic range across the population**
- The rest grid [850.8, 1420.8] doesn't even include the standard
  [1425, 1475] normalization region, so a normalize step couldn't
  even use that range without re-grid

When the v2 trainer's `_center_fluxes_inverse_variance` runs on
this data, the inverse-variance-weighted mean is heavily weighted
toward bright QSOs. The resulting μ doesn't represent a typical
QSO; it represents a bright-source-weighted average.

The CLI args `--norm_min_lambda=1425 --norm_max_lambda=1475` exist
in `prepare_trainset.py` but the corresponding `normalize_spectra`
call is **never made** in the pipeline. Likely an oversight.

**Implications for the 4 v2 trained models** (LOA_no_dla_no_bal,
LOA_no_hcd_with_bal, MOCK_2lpt_loa0, MOCK_2lpt_loa124_nohcd_nobal):
their μ is biased; they should NOT be promoted to production as-is.

**The v1 production model is correctly normalized** —
`SpectrumProcessor.normalize_spectra` (line 290) divides each spectrum
by its own median in [1425, 1475] before centring.

**Impact on the τ-EB story (PR #5)**: largely unaffected. τ-EB tunes
runtime `prev_tau_0` (mean-flux A), not the trained μ. So the bias-
closure measurements still stand even though the trained-μ underlying
the inference happens to be the v1 production (correctly normalized)
model.

Filed as a Tier 1 follow-up in
`docs/notes/2026-05-01_post_pr5_priorities.md`. Fix:
add a per-spectrum normalize step before `_center_fluxes_inverse_variance`
(either at preload time or in `dataset.load_preprocessed_h5`).

### Train-time z range — likely also differs

| Trainer | z range |
|---|---|
| **v2 (all 4 trainings)** | `[2.0, 4.25]` (per each preload's README) |
| **v1 production model** | NOT recoverable from `model_epoch_920.h5` alone. The v1 source has two catalog defaults: `LegacyGPCatalog z_range=(3.0, 4.25)` and another at `(2.15, 4.25)`. User recalls v1 was trained on `~(2.5, 4.25)` but never verified. |

If v1 was trained on a higher-z subset (z ≥ 2.5 or z ≥ 3.0), it didn't
see the low-z forest behaviour where mocks-vs-real divergence is largest
(per the τ-EB measurements: at z_qso ≥ 3.0 even real LOA wants τ_factor=1.0×;
the divergence is at z_qso = 2.0–2.5). This affects how μ and ω compare
between v1 and v2 — but the comparison is between trainsets that don't
fully overlap in z, so caveat the cross-trainer scale comparison.

### Why ω looks very different across trainers

ω scales with the same normalization as μ:

- **v1 ω**: in *fractional* flux units (since the data was divided by
  median ≈ 1 before training). Smooth curve ranging 0.5-2.
- **v2 ω**: in *absolute* flux units (since data was only mean-subtracted).
  Has wild dynamic range with spikes near Lyα emission edges.

The spikes in v2 ω near 1215 Å rest are at the boundary between
the QSO emission line and the forest, where the GP has high uncertainty.
v1 doesn't show these spikes because (a) it normalizes per-spectrum
so variance scales differently, and (b) v1's μ absorbs more of the
emission-line shape into its learned parameters.

The v2 `LOA_no_dla_no_bal` ω (panel B2) has the largest dynamic range
of the v2 set — sub-DLAs and LLS in the trainset add absorption
features that go into ω.

### Why loss y-scales differ by 8 orders of magnitude

| Trainer | Loss reported |
|---|---|
| v1 | Total log-likelihood (or negative log-likelihood × N_pix × N_spectra), absolute matlab-era scale |
| v2 | Per-pixel normalized loss (typically the per-pixel −log p(D | model) averaged over the batch) |

So v1 production loss reports ~6 × 10⁸ while v2 models all converge
in a 2 000-2 700 normalised-loss range. **They're not the same number;
direct comparison is meaningless.**

The v1 loss panel (C1) also shows an odd jump from very low to 6e8
around epoch 200 — likely an artifact of v1's two-phase training
schedule (PCA initialization phase has different loss scale than full
GP-fitting phase). For convergence purposes, only the late-epoch slope
matters; both v1 and v2 are converged.

For a proper comparable view, you could re-run the figure script with
each model's loss normalized by its `loss[0]` (or `loss[end]`). I've
left it un-normalized so the absolute-scale issue is visible.

## What the trainer actually optimises (architectural clarification, 2026-05-01)

> **The codebase carries two distinct (τ_0, β) parameter pairs**
> for what is physically the same forest-opacity quantity:
>
> 1. **Mean-flux suppression** in the GP likelihood, used to build
>    `A = exp(−τ_eff(z))` that multiplies BOTH the mean and the
>    covariance:
>    ```
>    y ~ N(A·μ ,  A^T K A  +  Ω²  +  V)
>    ```
>    Here A appears on μ (mean-flux suppression of the QSO emission
>    model) AND on the rank-k covariance K via `A^T K A`. At inference
>    these are computed from `prev_tau_0` / `prev_beta` — **runtime
>    constants** (Turner+2024 by default), not learnable.
>
> 2. **Ω-kernel diagonal** for per-pixel forest absorption noise.
>    This uses a SEPARATE pair of parameters (`log_tau_0`, `log_beta`
>    in the .h5) that ARE learnable in training. They parameterize
>    `Ω² ∝ (1 − A)² c_0 + ω²` for the diagonal noise term.
>
> Conceptually these two (τ_0, β) pairs should be the same
> physical thing; the codebase keeps them separate, and the user
> (jibanCat) has noted this is on the to-do list to unify but is
> not part of PR #5.
>
> The training data is pre-deforested at FIXED Turner+2024 before
> training (`gpy_dla_detection/training/dataset.py`,
> `_de_forest_spectra`), so the bulk mean-flux opacity is "consumed"
> at dataset prep time; the trained `log_tau_0` / `log_beta` are
> Ω-kernel residuals.
>
> **Therefore the τ-EB recipe in this PR is tuning `prev_tau_0` —
> the mean-flux-A coefficient — at INFERENCE time.** That parameter
> is **never touched by training**, so the trained model's identity
> (LOA-trained vs mock-trained vs production v1) is largely
> orthogonal to the τ-EB story. The mock-vs-real τ_factor divergence
> we measure is the actual mean-flux opacity gap between mocks /
> real LOA / Turner+2024.
>
> The trained `log_tau_0` / `log_beta` shown in the bar chart above
> are the **Ω-kernel** parameters. They corroborate the same
> mock-vs-real story (mocks need steeper β in Ω too) but they are
> a different number from the runtime mean-flux β.

## Convergence — should we train more?

```
LOA_no_dla_no_bal             1500 ep, loss 2724 → 2597, slope last-100: −0.002 / ep  CONVERGED
LOA_no_hcd_with_bal           1500 ep, loss 2239 → 2078, slope last-100: +0.001 / ep  converged (oscillating)
MOCK_2lpt_loa0                 800 ep, loss 2768 → 2428, slope last-100: −0.002 / ep  CONVERGED
MOCK_2lpt_loa124_nohcd_nobal   800 ep, loss 2493 → 2199, slope last-100: −0.002 / ep  CONVERGED
```

All four are converged. Extending the 2 GL models from 800 → 1500
epochs would reduce loss by ~2 (negligible vs final 2199-2428 range).
Worth doing for apples-to-apples comparison if epoch-count differences
matter for a referee, but the science conclusion is unchanged.

## Trainset filter differences (corrected — 2026-05-01)

The two NERSC-trained datasets are NOT symmetric in their filter
choices:

|  | `loa_no_dla_no_bal_52198069` | `loa_no_hcd_with_bal_52198070` |
|---|---|---|
| n_total | 300 008 | 300 032 |
| z range | [2.0, 4.25] | [2.0, 4.25] |
| ZWARN | =0 | =0 |
| BAL filter | **`exclude_bal=true`** (BI_CIV>0 dropped) | **`exclude_bal=false`** (BALs kept) |
| HCD filter | NHI ≥ 20.3 dropped (**DLAs only**; sub-DLAs+LLS kept) | NHI ≥ 17.2 dropped (**all HCDs**: DLAs+sub-DLAs+LLS) |
| BALs in data? | No | Yes |
| Sub-DLAs/LLS in data? | **Yes** | No |

Neither is "BAL-only". The `with_bal` in `loa_no_hcd_with_bal` means
"BAL spectra are kept alongside non-BAL spectra" (i.e. not excluded),
not "BAL spectra only".

The asymmetric filtering explains the trained-parameter pattern in
the bar chart above:

- `loa_no_dla_no_bal` keeps sub-DLAs + LLS in the data → there's
  residual absorption above what Turner-deforest removed → optimizer
  drives the Ω-kernel τ₀ up to **0.0048 (2× Turner)** to absorb that.
- `loa_no_hcd_with_bal` masks all HCDs (down to NHI 17.2) → no extra
  absorption signature in the data → optimizer drives the Ω-kernel
  τ₀ down to **~0** (BAL features get modelled by μ/M instead).

Both are converged. Different optima reflect different filtered
training distributions, not training failure.

## Why the τ-EB recipe behaves as it does — the linking story

τ-EB picks per-spectrum τ_factor that maximises log-evidence under the
production GP forward model. With the LOA-trained production GP:

- On **real LOA**: GP's μ × A_lyα(Turner) ≈ data → τ_factor ≈ 1× wins.
  Median 1.5×, frac ≥ 2× = 41 %. Recipe is a near-no-op.
- On **mocks**: GP's μ × A_lyα(Turner) under-predicts absorption (mocks
  have ~2× more opacity than LOA at low z). τ-EB compensates by
  picking τ_factor ≈ 3×. Median 3.0×, frac ≥ 2× = 76-77 %.

**The τ-EB recipe is, in effect, calibrating per-spectrum mean-flux to
bridge the gap between the GP's training distribution (LOA) and
whatever the test data actually is.** This frames the recipe more
honestly: it's not "fixing a bias", it's "adapting the forward
model's mean-flux opacity per spectrum, away from its training
anchor". The bias closure on mocks (56-65 % at production cut) is
the magnitude of that adaptation.

## In flight — 2×2 training-data anchor experiment

The proposition is testable: train a GP on mocks and run it on mocks
(matched), and a separate GP on LOA and run on LOA (matched). If both
matched cases produce τ_factor ≈ 1×, the anchor hypothesis is confirmed
and the τ_factor we observed in production simply measures
mock-vs-LOA opacity ratio.

| Dataset \ GP | LOA_no_dla_no_bal | MOCK_2lpt_loa124 |
|---|---|---|
| 2lpt mock | (control: predicts ~3×) | **predicts ~1× (matched)** |
| LOA real | **predicts ~1× (matched)** | predicts <1× (over-absorbed) |

Submitted as jobs 49108430 / 49108431 / 49108432 / 49108443. ETA
~2 h wall. Will append the τ_factor distributions to this doc and
to `2026-05-01_tau_factor_distributions.md` once the runs land.

## Practical implications for production

1. **The current production model (`learnlogs/model_epoch_920.h5`,
   v1 LOA-trained) is fine for production LOA inference.** The
   per-spectrum τ-EB recipe is closer to a no-op on real data
   (median τ_factor=1.5) and only kicks in significantly on mock
   inference.
2. **The new v2 LOA-trained model (`loa_no_dla_no_bal_52198069`) is a
   candidate replacement** with cleaner training data (DLAs/BALs
   masked) and substantially more epochs (1500 vs 953). However it
   uses v2 normalization, so swapping into the v1 inference pipeline
   may require a small calibration. **Out of scope for this PR.**
3. **The 2lpt-trained models (`v2_runs/2lpt_loa{0,124}_*`) are
   research models for the anchor experiment**, not for production
   inference (they encode mock physics, not real DESI).

## Files

| File | What |
|---|---|
| `examples/compare_trained_gp_models.py` | extractor + figure generator |
| `docs/story_figures/trained_gp_models_compare.png` | μ + ω + loss panels |
| `docs/story_figures/trained_gp_models_hyperparameters.png` | τ₀, β, c₀, ω̄ bars |
| `docs/story_figures/trained_gp_models_table.md` | embeddable hyperparameter table |
| `docs/notes/2026-05-01_tau_factor_distributions.md` | companion τ measurements |
| `docs/stories/tau_eb_story_loa.md` | the LOA-vs-mock writeup |
