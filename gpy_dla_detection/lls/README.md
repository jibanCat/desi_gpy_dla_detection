# `gpy_dla_detection.lls` — the LLS (Lyman-limit system) sub-module

Everything LLS-specific lives here, **disentangled from the main DLA / sub-DLA path** so the
NERSC-proven DLA inference stays byte-identical. Nothing in this package modifies
`dla_gp.py`, `subdla_gp.py`, `voigt.py`, or `voigt_lls.py`.

The LLS regime is `17.2 <~ log N_HI <~ 19`. There the Lyα line is **saturated but not damped**
(flat curve of growth), so line-only detection is only ~5% pure. The distinguishing feature of
an LLS is instead the **bound-free Lyman-limit break** at rest 912 Å:

```
tau_LL(lambda_rest) = N_HI * sigma_912 * (lambda_rest / 912)^3    for lambda_rest < 912 A
sigma_912 = 6.35e-18 cm^2   (== 1 / 10^17.2, the self-shielding column)
```

which carries a per-sightline matched-filter S/N of ~10–18 on 2LPT mocks. So the strategy is a
GP forward model that scores **both** the Lyman-series lines and the 912 Å break in one
likelihood, and an inference window wide enough to actually contain the break.

---

## Contents

| module | what it is |
|---|---|
| `gp.py` | Break-aware GP: `SubDLAGP{,MAT}LymanBreak` — the GP+Voigt forward model **with the 912 Å drop folded into one likelihood**. Plus `load_lls_gp` / `extend_window_to_drop` (blueward window) and the LLS-only model path. |
| `mirror.py` | Mirror-quickquasar LyC injection: add the bound-free 912 Å drop to 2LPT spectra (which carry quickquasars' Lyman-*series* lines only). Writes a mirror `spectra-16` tree the finder reads unchanged. **No quickquasars re-run.** |
| `train.py` *(planned)* | Relearn the QSO GP down to the drop on **HCD-free 2LPT-0** spectra (extended grid → ~800 Å rest), GPU recipe. See the training doc. |

---

## The window problem (why a naive break-aware model does nothing)

A foreground absorber's 912 Å break is observed at `lambda_obs = 912 (1+z_abs)`, which in the
**quasar** rest frame is `912 (1+z_abs)/(1+z_qso) < 912` — i.e. **blueward of the quasar's own
Lyman limit**. Measured on the 2LPT-0 truth (358k LLS):

- break-edge quasar-rest wavelength: **median 836 Å**, 5–95% [676, 908]

The standard model window is `[911.75, 1215.75]` Å (rest) and the counting search is Lyα-only
(`lam_rf_min=1025`). So **~99.6% of LLS breaks fall below the window** — a break-aware model run
on the default window sees no break and gains nothing. The break has to be brought *in-window*.

### Two tiers of window extension

The production DESI model `model_epoch_920.h5` already has a rest grid **[850.90, 1420.60] Å** —
it *already* models the Lyman-continuum region; inference merely clips it at 911.75. So:

| tier | window floor | LLS breaks in-window (rest-frame grid) | cost | frozen code |
|---|---|---|---|---|
| **Tier 1** | 850.9 Å (config-only) | ~41% | none — existing model | untouched |
| **Tier 2** | ~800 Å (relearn) | ~66% | preload regen + GPU train | untouched |

Tier 1 is `load_lls_gp()` below — pure config, no retrain. Inspection of `model_epoch_920`'s
[851,912) band (407 px): mean declines smoothly 1.47→0.48, ω=1.18 (below the forest's 2.29) —
well-behaved, usable.

> ⚠️ **STALE — computed on the pre-fix (legacy-noise) mirror mock (2026-07-09).** The `log-BF ≈ +80`
> and `~+0.05 detection recall` figures below, and the `mirror/orig = 0.320` measurement further
> down, were all measured on a mirror whose below-break noise was artificially quiet (the legacy
> `flux *= T` attenuated the noise realization along with the signal; measured pull σ = 0.235 where
> it should be 1.0). They **must be re-derived** on a mirror built with the corrected default path,
> as must SLURM `52949574` / `52950580`. The *direction* of the fix (a noisier break) can only
> **reduce** the recall gain, so the qualitative conclusion — the Lyα line already saturates, the
> break adds little to *counting* — is safe and in fact reinforced. The `~10% observable` figure is
> geometric (blue cutoff) and is **not** affected by the noise model. See `mirror.py` docstring.

> **Observability caveat (2026-07-05 referee finding).** The "~41% in-window" is a *rest-frame
> model-grid* count. Only **~10%** of foreground LLS have an *observable* break — the break sits at
> `912(1+z_abs)`, which for the z_qso≈2.26-dominated mock falls below the DESI blue cutoff (~3600 Å)
> for ~90% of sightlines. And even where the break IS observable and is a huge per-sightline feature
> (log-BF ≈ +80), the break-aware finder gains only **~+0.05 detection recall**: the Lyα line already
> saturates P→1 for those systems. **The break's survey value is the aggregate λ_mfp (drop channel)
> and N_HI refinement — NOT per-sightline counting** (which the line already does). See the
> walkthrough `notes/2026-07-05_lls_break_aware_walkthrough.md` §9–10.

---

## Usage

### Break-aware LLS finder (Tier 1, no retrain)

```python
from gpy_dla_detection.lls import load_lls_gp

# params/prior/dla_samples built as for the normal sub-DLA run; load_lls_gp extends the
# window to 850.9 A and loads the LLS-only model, returning a break-aware GP.
gp = load_lls_gp(params, prior, subdla_samples)   # SubDLAGPMATLymanBreak, window -> 850.9
# ... then the usual set_data / model_selection loop, unchanged.
```

`SubDLAGP{,MAT}LymanBreak` override only `this_dla_gp`, routing per-absorber absorption through
`voigt_lls.voigt_absorption = exp(-(tau_lines + tau_LLS_break))` instead of the line-only
`voigt_absorption`. Everything else (priors, samples, model-selection) is inherited from the
frozen `SubDLAGP` / `DLAGP`.

### Mirror mock (add the 912 Å drop to 2LPT-0)

```bash
python -m gpy_dla_detection.lls.mirror --limit-healpix 200 --out /scratch/.../mirror_2lpt0
```

Optical depth is additive (`tau_tot = tau_lines + tau_LL`), so multiplying each existing
spectrum by `exp(-tau_LL)` for its truth HCDs is identical to re-doing the HCD absorption with
the break-aware Voigt. Verified on 2LPT-0: a log N=17.32 LLS shows mirror/orig = 0.320 below
912(1+z) vs predicted `exp(-tau_LL)` = 0.296, and 1.000 above.

---

## Relearn the GP down to the drop (Tier 2)

Train a **mock-matched** extended GP on **HCD-free 2LPT-0** sightlines (clean forest → clean
null model; forest statistics match the mirror-mock test data, no mock-vs-real transfer gap),
with the preload grid floor pushed to ~800 Å, using the same PCA-init + GPU `train_gp.py`
recipe. Two stages (reuse the proven mock path — no new code):

```bash
# 1) preload (CPU, ~4h): HCD-free 2LPT-0 loa-124, grid floor extended to 800 A
sbatch --export=ALL,VARIANT=loa124_nohcd_nobal,MIN_LAMBDA=800.0,Z_MIN=2.0,Z_MAX=4.0,\
RUN_TAG=2lpt_loa124_nohcd800 slurm/greatlakes/preload_2lpt_only.sh

# 2) train (GPU, ~4h): same PCA-init + autograd recipe, k=30, 800 epochs
sbatch --export=ALL,RUN_TAG=2lpt_loa124_nohcd800,Z_MIN=2.0,Z_MAX=4.0,NUM_EPOCHS=800,NUM_PCA=30 \
    slurm/greatlakes/train_only_gpu.sh
# -> .../v2_runs/2lpt_loa124_nohcd800/model_epoch_0799.h5  (point load_lls_gp(learned_file=...) here)
```

**Ceiling:** the LyC region is only constrained by z_qso≳3 quasars — ~43–50% of LLS breaks is
the practical counting ceiling (real-LOA limit too), the rest need the drop channel. Full steps
+ diagnostic figures: `notes/2026-07-05_lls_gp_relearn_hcdfree_2lpt0.md` *(private notes repo)*.

---

## Frozen-code discipline

- New files only; `dla_gp.py`, `subdla_gp.py`, `voigt.py`, `voigt_lls.py` are **byte-identical**.
- The LLS model (`model_epoch_920.h5` for Tier 1, or the relearned mock model for Tier 2) is
  loaded **only here** — DLA and sub-DLA keep their own model and window.
- The break-aware classes reuse the frozen `SubDLAGP`/`DLAGP` verbatim and override only
  `this_dla_gp`.
