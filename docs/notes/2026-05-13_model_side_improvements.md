# Model-side improvements for the trainer agent — what to do RIGHT NOW

> Written 2026-05-13 after the Var[Δ_marg] gating diagnostic
> ([`2026-05-13_var_delta_marg_diagnostic.md`](2026-05-13_var_delta_marg_diagnostic.md))
> showed the pipeline is statistic-limited, not sampling-limited, at
> production N=50k. The lever is now squarely model-side. This doc gathers
> the concrete trainer-actionable improvements from the existing
> hypothesis-ledger and trained-model-comparison notes into one
> prioritized list, scoped specifically to what a GreatLakes trainer agent
> can act on without coordinating inference-side changes.
>
> Companion docs (read these first if you have time):
> - [`2026-04-29_bayesian_correctness_synthesis.md`](2026-04-29_bayesian_correctness_synthesis.md) — H1-H8 hypothesis ledger
> - [`2026-05-01_trained_gp_models_comparison.md`](2026-05-01_trained_gp_models_comparison.md) — 5 trained models on disk + v2 normalization bug
> - [`2026-04-27_subdla_model_improvements.md`](2026-04-27_subdla_model_improvements.md) — 4 ranked sub-DLA model improvements
> - [`2026-04-27_lybeta_persistence_hypotheses.md`](2026-04-27_lybeta_persistence_hypotheses.md) — Lyβ misID hypotheses H1-H4

## Verdict that motivates this doc

At production N=50k QMC samples, `Var[Δ_marg]` across seeds is ~130× *below* the signal–null gap (σ ≈ 0.1 vs gap ≈ 13). The borderline P/C ceiling at SNR > 2 is NOT a sampling problem — increasing N or replacing QMC with importance sampling / MLMC / pocoMC will not move the science result at production cost. **The remaining headroom is in the GP forward model itself**: its μ shape, its Ω calibration, its rest-frame z range, and the trainset filters that shape both.

The trainer's job is therefore: *change the inputs to training, or the architecture parameters of the GP, in ways that widen the signal–null Δ_marg gap on the truth-positive set* without inflating it on truth-negatives.

## Scope clarifier — trainer-only vs joint-with-inference

Each improvement below is tagged:

- **[T]** trainer can ship alone — produces a new `.h5`, drops into the existing inference path with no code change beyond `LEARNED_FILE` swap.
- **[T+I]** needs new trainer output AND an inference-side change (new sample file path, new prior config, new model variant index). Coordinate with the production team before landing.

Inference-only levers (τ-EB, BAL pixel masking, multi-line cross-checks, neighbor-pixel features, postprocess vetos) are **not in this doc** — they belong to the production-side roadmap, not the trainer PR.

---

## Tier 0 — Critical bug fix (blocks everything else)

### 0.1 — Fix the v2 preload normalization bug [T]

**Status: known bug, root-cause identified, no fix landed yet.** See
[`2026-05-01_trained_gp_models_comparison.md`](2026-05-01_trained_gp_models_comparison.md)
§ "⚠ BUG — v2 preload skips per-spectrum normalization".

The v2 preload scripts (`preload_spectra/preload_loa_real.py`,
`preload_spectra/preload_2lpt_simple.py`,
`preload_spectra/prepare_trainset.py`) skip the per-spectrum median-flux
normalization in `[1425, 1475]` Å rest-frame before passing fluxes to
`_center_fluxes_inverse_variance`. The v1 production trainer
(`SpectrumProcessor.normalize_spectra`) does normalize, so v1's learned μ
is on the right scale. **All four v2 trained models on disk
(`LOA_no_dla_no_bal`, `LOA_no_hcd_with_bal`, `MOCK_2lpt_loa0`,
`MOCK_2lpt_loa124_nohcd_nobal`) have biased μ.** The bias-fix story (τ-EB)
holds because τ-EB only tunes runtime `prev_tau_0`, not the trained μ —
but any v2 model promoted to production as-is would carry the bright-QSO-
weighted μ bias forward.

**Fix:** Add a per-spectrum normalize step before
`_center_fluxes_inverse_variance` — either at preload time (in
`preload_*.py` after the mask+interpolate step) or in
`dataset.load_preprocessed_h5` (after loading, before centering). The
`prepare_trainset.py` CLI already has `--norm_min_lambda=1425
--norm_max_lambda=1475` flags; the corresponding `normalize_spectra`
call is missing. The rest-grid currently `[850.8, 1420.8]` doesn't
include `[1425, 1475]` so the normalize step needs to use a re-grid or
pull from the raw spectrum before grid-projection.

**Validation after the fix:**
1. Re-run `examples/compare_trained_gp_models.py` and verify v2 μ is now
   on a flux-units-1 scale comparable to v1 (after dividing v1 μ by its
   own median-normalized population mean, which should be ≈ 1).
2. Verify `c_0` (multiplicative noise) is order ~0.1, comparable to v1's
   0.17 — not v2's current 0.001-0.04 (which is artifact of the un-
   normalized fluxes).
3. Smoke-compare a few targets between fix-on and fix-off v2 models. The
   GP continuum (`this_mu`) should hug the data better at the QSO emission
   peak.

**Why this is Tier 0:** every other trainer improvement below assumes you
can compare v2 models against each other on a fair scale. With the
normalization bug, the comparisons are confounded.

**Cost:** ~1 day. Re-training each of the 4 v2 models from the fixed
preload is ~16 GPU-hours each on GreatLakes per the existing
`gpy_dla_detection/training/` configs.

---

## Tier 1 — High-leverage, well-defined

These are concrete training-side knobs whose effects we can predict and
test in isolation. Pick one at a time.

### 1.1 — Train on the cleanest forest possible: LOA real, all HCDs masked [T]

The trainer agent already submitted `loa_no_hcd_with_bal_52198069` (mask
all NHI ≥ 17.2, keep BALs). After the Tier 0 fix re-train it. This is
the **cleanest-forest model**: every known absorber is masked out of the
trainset, so the learned μ + ω reflect pure intervening forest + QSO
emission. Compared to the v1 production model (which trained on data
that still contained absorbers), this should:

- Tighten ω at locations where production-v1 had high uncertainty because
  of trainset absorber contamination
- Stabilize the QSO emission-line shape (Lyα peak, NV, etc.)
- Reduce the spurious 2-DLA-at-Lyβ rate

**Expected gain:** the relevant comparison is `loa_no_hcd_with_bal` vs
v1 production on a stratified 200-target sample (the same one used in
[`2026-04-25_filter_samples_sweep.md`](2026-04-25_filter_samples_sweep.md)).
Reduction in spurious 2-DLA rate of 10-20 % is plausible; reduction in
the [20.0, 20.3] CDDF dip is also plausible (the cleanest forest doesn't
have residual absorption pulling μ down at those wavelengths).

**Action for the trainer agent:**
1. Apply the Tier 0 fix.
2. Re-run `loa_no_hcd_with_bal_*` from the fixed preload.
3. Run the comparison: convert to inference-compatible `.h5`
   (`null_gp_test/converted/`), then run inference on the 200-target
   stratified sample with both models, compare P/C.
4. If the cleanest-forest model wins, this becomes the new production
   GP candidate.

### 1.2 — Test GP rank K > 30 [T]

Current production K=30. The GP forward model is a rank-K factor model
on the (M, K) basis. Higher K means less smoothing of the QSO emission
shape — sharper peaks at Lyα / NV / etc., potentially better
discrimination between emission-line residuals and weak DLA signatures.

The synthesis note marks H1 (LSF) and H2 (num_lines) as ruled out for
the DLA regime but **doesn't test rank**. There's no theoretical reason
K=30 is optimal; it's a legacy choice from Ho+2020.

**Action for the trainer agent:**
1. Train at K ∈ {30 (baseline), 50, 64} with the fixed preload + the
   same trainset as Tier 1.1.
2. Compare loss-at-convergence per rank.
3. Compare P/C on the 200-target stratified sample.
4. **Stop early if K=50 doesn't show a clear improvement** — diminishing
   returns are likely and the inference cost scales as K² for the
   covariance solve.

**Cost:** ~16 GPU-h × 3 ranks = 48 GPU-h. Cheap.

**Expected gain:** uncertain. Plausibly 5-10 % reduction in borderline
spurious detections if the emission-line shape was previously under-fit
at K=30. Could be zero if K=30 is already saturated.

### 1.3 — Widen the training z range [T]

The four v2 models all use z ∈ [2.0, 4.25]. v1 production is suspected
to be trained on z ≈ (2.5, 4.25) or (3.0, 4.25) — the user recalls
~(2.5, 4.25) but verification from the .h5 alone isn't possible. The
τ-EB story (mocks at 3× Turner, real LOA at 1.5×) **is largest at low
z (2.0–2.5)**. If v1 wasn't trained on this low-z forest behaviour, it
may under-fit there.

**Action for the trainer agent:**
1. Use the v2 trainer (post-Tier-0-fix) at z ∈ [2.0, 4.25] as the
   baseline.
2. Optionally also try z ∈ [1.96, 4.25] to capture the entire DESI
   inference cut.
3. Compare on a *low-z*-stratified target subset (z_qso ∈ [2.0, 2.5]) —
   this is where the divergence is largest.

**Cost:** included in Tier 1.1 if the baseline z range is already [2.0,
4.25]. Optional sweep adds ~16 GPU-h.

**Expected gain:** modest at high-z (where the production data already
trained), substantial at low-z (potentially 10-15 pp completeness gain
on z_qso < 2.5 targets, since the low-SNR forest is also the low-z
regime per the τ-EB measurements).

### 1.4 — Train on LOA + 2LPT hybrid [T] (speculative — discuss before doing)

Current options for the inference GP are LOA-trained (v1, current
production) or 2LPT-trained (v2_2lpt_loa124, the v3 model used in this
session's prod533 runs). The 2LPT models bake in mock physics (β ≈ 4.6
vs LOA's β ≈ 3.6 — see the trained-models doc) and shouldn't be promoted
to production for LOA inference. But a **hybrid trainset** — both 2LPT
and LOA mixed — might give a GP that generalizes to both regimes.

**Caveat:** the τ-EB story shows the mock/real divergence is REAL physics
(mocks have ~2× more opacity than real LOA). A hybrid trainset would
either average β = 4.0 (compromise that's wrong for both) or learn a
bi-modal distribution (which the rank-K factor model can't represent).
**Likely fails**, but cheap to try.

**Action:** lower priority than 1.1–1.3. Discuss with user before
training.

---

## Tier 2 — Sub-DLA model improvements

These are tagged [T+I] because they require BOTH a new trainer output AND
inference-side plumbing changes. Coordinate with production.

### 2.1 — Extend sub-DLA NHI prior to overlap with DLA [T+I]

From [`2026-04-27_subdla_model_improvements.md`](2026-04-27_subdla_model_improvements.md)
§ A. Current sub-DLA prior is NHI ∈ [19.5, 20.0]; DLA prior is NHI ∈
[20.0, 23.0]. No overlap. Truth DLAs at NHI ≈ 20.05 fall in a no-man's
land where Bayesian model selection assigns them somewhat arbitrarily,
biasing the dN/dX in the [20.0, 20.3] bin low.

**Action for the trainer agent:**
1. Generate a new sub-DLA QMC sample file with NHI ∈ [19.0, 20.3]:
   ```bash
   python gpy_dla_detection/generate_samples.py \
       --num_dla_samples 10000 \
       --log_nhi_min 19.0 --log_nhi_max 20.3 \
       --out data/dr12q/processed/subdla_samples_a03_190_203_10000.mat
   ```
2. Validate the sample on a small mock to confirm the prior is correctly
   loaded (`SUB_DLA_SAMPLES_FILE` env var path).
3. Document the new file path so production can pick it up.

**Inference-side needed:** swap `SUB_DLA_SAMPLES_FILE` to the new path
in production configs. Validate that the column layout of
`model_posteriors` is unchanged.

**Expected gain:** closes the [20.0, 20.3] CDDF bias. ~2-4 pp dN/dX
recovery in that bin.

### 2.2 — Re-tune the Ho+2020 alpha mixture weight for the sub-DLA prior [T]

From [`2026-04-27_subdla_model_improvements.md`](2026-04-27_subdla_model_improvements.md)
§ C. The Ho+2020 α = 0.97 mixture weight was tuned on the DLA NHI
distribution and inherited for the sub-DLA prior. Prochaska & Wolfe 2014
show sub-DLAs have a shallower NHI distribution; the inherited α is
likely too aggressive.

**Action for the trainer agent:**
1. Pull the sub-DLA truth catalog (`hcd_truth_cat.fits` for Saclay or
   `dla_cat.fits` filtered to NHI ∈ [19, 20.3] for London).
2. Fit the PW14 mixture: maximum-likelihood α on the truth sample.
3. Generate a new sub-DLA sample file with the re-tuned α.

**Expected gain:** better posterior calibration at the NHI boundary,
~1-2 pp improvement in sub-DLA purity (the trade-off is slight
completeness loss at NHI ≈ 19, which is the dominant bin and acceptable).

### 2.3 — Strong sub-DLA model — make it multi-absorber [T+I]

From [`2026-04-27_subdla_model_improvements.md`](2026-04-27_subdla_model_improvements.md)
§ B. **Larger architectural change** — keep on the radar but don't
attempt without discussion.

Current sub-DLA model in `single_absorber_model=0` mode competes for the
SAME observed feature as the 1-DLA model — it can't model a LOS with a
real strong DLA AND a real LLS at a different z. Allowing
`M_subdla(N)` for N ≥ 1, in parallel with `M_dla(N)`, fixes this.

**Cost:** ~1 week of refactor in `run_bayes_select.py` and
`gpy_dla_detection/bayesian_model_selection.py`. Defer until the simpler
items land.

---

## Tier 3 — Speculative (not for the first trainer PR)

### 3.1 — Student-t residuals (H7) [T+I]

The GP likelihood currently assumes Gaussian residuals around the mean.
H7 in the hypothesis ledger speculates that heavy-tailed (Student-t)
residuals would better capture sky-line residuals, BAL-trough-edges, and
emission-line scatter. Untested.

**Cost:** substantial — touches the GP likelihood module. Discuss before
attempting.

### 3.2 — num_forest_lines > 3 for low-SNR [T]

H2 in the ledger was ruled out for the DLA regime (`num_lines=3` vs 6
gives bit-identical MAP). But the test was on DLA-regime targets. For
the **low-SNR, low-NHI** regime where Δ_marg is borderline, the higher
Lyman series lines might matter — adding Lyδ + Lyε could shift Δ_marg
on sub-DLA borderline cases.

**Action:** repeat the H2 test specifically on low-NHI (sub-DLA range)
targets. If still null, drop. If positive, retrain models with
`num_forest_lines = 5` or 6 baked in.

**Cost:** ~1 day to test, plus retraining if positive.

### 3.3 — Voigt LSF kernel revisit [T]

H1 ruled out for DLA regime, but the docs note **kernel matters in the
sub-DLA / LLS regime** (config-spread 0.029 → 0.384 dex on saclay
target). For the sub-DLA PR specifically, revisiting the LSF could be
worthwhile.

See `docs/notes/2026-04-29_voigt_lsf_sweep/` for the existing sweep
infrastructure.

---

## Recommended order of operations for the trainer PR

1. **First commit**: Tier 0.1 — fix the v2 preload normalization bug.
2. **Second commit**: Tier 1.1 — re-train `loa_no_hcd_with_bal` with the
   fix. This becomes the candidate production v2 model.
3. **Third commit**: Tier 1.2 — K-rank sweep at K ∈ {30, 50, 64} on the
   trainset from step 2.
4. **Fourth commit**: Tier 2.2 — re-tune alpha and produce a new sub-DLA
   sample file. (Trainer can do this without retraining the GP.)
5. **Fifth commit** (if there's a follow-up PR): Tier 2.1 — extended
   sub-DLA NHI prior sample file.

Items 3, 4, 5 are independent of each other — they can be parallelized
on GreatLakes if you have the budget.

After each item, run the 200-target stratified P/C comparison
(`examples/molly_faithful_pc_plots.py` against the truth catalog) and
record the result. The acceptance criterion at this point is: **does the
new model widen the signal-null gap relative to v1 production at
SNR > 2?** That's the metric the Var[Δ_marg] verdict tells us to optimize.

---

## What success looks like

A trainer PR that lands at least Tier 0.1 + Tier 1.1 + Tier 1.2, with:

1. Tier 0.1 fix verified by smoke-comparison plots (added to `examples/`).
2. Tier 1.1: new `loa_no_hcd_with_bal_v2_fixed.h5` shipping in
   `learnlogs/` (or wherever production picks up models from).
3. Tier 1.2: a K-rank comparison figure in `docs/story_figures/`
   showing loss-at-convergence and P/C at each K.
4. A short note (`docs/notes/2026-MM-DD_trained_v2_validation.md`)
   capturing the comparison vs v1 production on the 200-target sample.
5. **Net change to signal-null Δ_marg gap** on the 5k stratified
   borderline sample — this is the headline number to put in the PR
   description. Target: gap widens by ≥ 1 (in log-evidence units)
   without inflating Δ_marg on truth-negatives.

If item 5 doesn't move, the trainer PR has surfaced more about the
limits of the current architecture than it has improved them — that's
still useful but the next step would be tier 2.3 (multi-absorber
sub-DLA hypothesis) which is the larger architectural change.

---

## What NOT to do in the trainer PR (don't be tempted)

- **Don't change the inference τ-EB recipe.** It's PR #5's deliverable
  and is already validated at scale. Train new models compatible with
  τ-EB-default-OFF first; turn it on at inference time only.
- **Don't introduce Student-t residuals (Tier 3.1) in the first PR.**
  Speculative + invasive — wait for the simpler items to land first.
- **Don't combine multiple Tier 1 items into one model** before the
  comparison. Train one knob at a time so the A/B ablation is clean.
- **Don't promote any v2 model to production before validating the
  Tier 0.1 bugfix actually closes the μ-bias.** A μ-biased model that
  happens to perform better than v1 on one mock might fail catastrophically
  on a different distribution.
