# Tier 1 #1 — sub-DLA / DLA prior boundary fix: design proposal

> **Status**: Draft for user review before implementation.
> **Author**: Claude, 2026-05-02 session.
> **Goal**: Improve purity + completeness near log_NHI ~ 20 (the
> sub-DLA / DLA boundary).

## Problem characterization

The production GP-DLA pipeline runs Bayesian model selection with
*disjoint* absorber priors:

| Model | log_NHI range | source file | n_samples |
|---|---|---|---:|
| sub-DLA (LLS-side) | [19.5, 20.0] | `subdla_samples.mat` | 10,000 |
| DLA (k=1..max) | [20.0, ~23] | `dla_samples_a03.mat` | 10,000 |

**No coverage below log_NHI = 19.5.** Anything in [17.2, 19.5) is
silently invisible to the multi-DLA pipeline. The LLS production mode
(SINGLE_ABSORBER_MODEL=1) uses different sample files
(`pw_samples_a3_172_220_50000.mat` covering [17.2, 22.0]) but is
mutually exclusive with multi-DLA mode.

### Three failure modes this creates

1. **Pile-up at the boundary (purity).** Truth absorbers with
   log_NHI ∈ [19.5, 20.3] often get fit by the 1-DLA model with
   MAP_log_NHI = 20.3 (lower bound of the DLA prior). The CDDF at
   log_NHI = 20 is artifactually high.

2. **Gap [17.2, 19.5) entirely missed.** Truth absorbers in this
   regime register as "no absorber" — the spectrum's absorption is
   absorbed (no pun intended) into the GP's per-pixel ω noise term,
   not flagged as a discrete absorber.

3. **Sub-DLA detection has no z/NHI estimate.** The sub-DLA model is
   a *penalizing alternative* in the Bayes factor — it raises
   p(no DLA) in the catalog but doesn't fit a sub-DLA absorption
   profile. So `MAP_z_subdla` and `MAP_log_nhi_subdla` columns don't
   exist; only the model-selection probability does.

### Direct evidence — canonical TID 120046865

The 2026-05-02 canonical comparison ran 7 trained models on a 2lpt
mock-0 spectrum with **two truth absorbers**:

| Truth | z | log_NHI | What models found |
|---|---:|---:|---|
| sub-DLA | 2.287 | **19.41** | **All 7 models miss it entirely** (falls in the 17.2-19.5 gap) |
| DLA | 2.773 | 21.26 | 5 of 7 models find it at MAP log_NHI=21.63 (+0.36 dex bias) |

The 19.41 sub-DLA is not even on the radar of any model — illustrating
failure mode #2 in production-realistic conditions.

## Three options of increasing scope

### Option A — extend sub-DLA range only (cheap stop-gap)

Generate a new `subdla_samples.mat` with log_NHI ∈ [19.5, **20.3**]
instead of [19.5, **20.0**]. The 0.3-dex overlap with the DLA prior
soaks up the boundary pile-up — a truth-19.6 absorber can be fit by
*either* the sub-DLA or DLA model, and Bayesian model selection will
prefer the sub-DLA when the absorption is weak.

**What it fixes**: failure mode #1 (boundary pile-up) — partially.
**What it does NOT fix**: #2 (the [17.2, 19.5) gap) and #3 (sub-DLA
catalog has no z/NHI). The canonical TID 19.41 absorber would still
be missed.

**Effort**: ~1 day. Generate new .mat samples + update the loader
defaults + retrain validation campaign on a subset (n=50 sub-DLA-rich
mocks) to confirm the boundary pile-up moves.

### Option B — wider sub-DLA range + estimate z/NHI in catalog (medium scope)

Two changes from option A:
1. Extend sub-DLA range further to **[17.2, 20.3]** (covers LLS + sub-DLA
   + boundary), giving full coverage in concert with the DLA model.
2. Modify `SubDLAGPMAT` to actually fit a sub-DLA Voigt at the MAP
   sample (currently it only computes evidence). Add `MAP_z_subdla`
   and `MAP_log_nhi_subdla` to the results dict.

**What it fixes**: #1, partially fixes #2 (now sub-DLAs in [17.2, 19.5)
are detectable but with a wide single-absorber prior — won't capture
multi-absorber LOS), partially fixes #3 (z/NHI for the *single best*
sub-DLA, not multi-sub-DLA cases).

**Effort**: ~3-5 days. The MAP z/NHI extraction is straightforward
(it's already done for DLAs in `dla_gp.py`); just need to wire the
analogous sub-DLA path. Wider sample range needs a regenerated .mat.
A small but non-trivial change to the catalog schema downstream.

### Option C — two-stage scan (full architectural rework)

User's original proposal from `project_post_pr5_priorities_2026_05_01.md`:

> Better plan (user): two-stage scan — scan [17.2, 23] for rough
> peaks first, then run multi-DLA on those peaks. Allows joint
> sub-DLA + DLA fits without breaking the multi-DLA model.

**Stage 1 — peak detection.** Sweep the spectrum with a wide-prior
1-absorber model over log_NHI ∈ [17.2, 22.0]. Record local maxima of
the per-pixel log-evidence vs z. Filter peaks above some
significance threshold. (Could reuse the LLS-mode `pw_samples_*.mat`
infrastructure — already exists.)

**Stage 2 — multi-absorber fit at the peaks.** For each peak, run a
focused Bayesian model selection that tries: this-peak-is-LLS vs
this-peak-is-sub-DLA vs this-peak-is-DLA, simultaneously across
peaks. This is the joint multi-absorber semantics that the current
"sub-DLA as penalizing alternative" architecture doesn't support.

**What it fixes**: #1, #2, #3 — all three.
**What it requires**: a redesign of `BayesModelSelect`, likely a new
inference engine class. The current `BayesModelSelect([0, max_dlas], 1)`
treats absorber count as a model-index axis and only allows one
absorber type per inference run; joint sub-DLA + DLA needs a
hierarchical model.

**Effort**: ~3-4 weeks of dedicated design + implementation +
validation. Not a single-PR scope.

## Recommendation

**Ship Option A in the next PR.** Reasons:
1. Fastest path to a measurable improvement on the boundary
   pile-up, which is the headline science concern at log_NHI ~ 20.
2. Doesn't change the inference engine — only the sample priors.
   So existing tests, downstream catalogs, and population statistics
   code (`CDDF_analysis/`) keep working.
3. Provides empirical evidence (post-fix CDDF at log_NHI=20.3) for
   whether the boundary issue is now resolved or whether we need
   to do option B / C.
4. Low risk: if the wider sub-DLA prior over-penalizes DLAs and we
   lose completeness, we can revert by swapping the .mat file.

**Plan Option B as the follow-up if Option A's CDDF improvement is
inadequate.** The MAP z/NHI extraction for sub-DLAs is independently
useful (the catalog gets richer), and the [17.2, 20.3] coverage
addresses failure mode #2.

**Plan Option C as a multi-quarter project.** It's the right
architectural answer but the cost-benefit only makes sense once
options A and B have been measured and shown to be insufficient.

## Test plan (Option A)

1. **Generate** `subdla_samples_extended_19_5_to_20_3.mat` with the
   same Prochaska+2014 prior shape, just on the wider [19.5, 20.3]
   range. Use the existing sample-generation MATLAB code or port to
   `gpy_dla_detection/generate_samples.py`.

2. **Smoke test on canonical TID** — does `subdla_lls.mat` extension
   change the canonical comparison output? Sub-DLA truth at 19.41
   still misses (Option A doesn't address that), but the DLA at 21.26
   should still be found by all the working models. p_subdla for the
   DLA-finding models should remain near zero.

3. **Sub-DLA-rich mock validation** — pick 50 known sub-DLA truth
   targets (truth log_NHI ∈ [19.5, 20.3]) from a 2lpt mock. Run
   inference with the OLD subdla_samples.mat and the NEW one. Compare:
   (a) p_subdla distribution, (b) CDDF at log_NHI=19.7 vs 20.3 vs
   20.5, (c) DLA false-positive rate at the boundary.

4. **Population CDDF re-run** on the existing 50k Phase-B output
   datasets — apply the new prior at the post-processing CDDF
   estimation step (where the prior shape enters via the
   `cddf_calibration` module). Verify the artifactual peak at
   log_NHI=20.3 is reduced.

5. **Production cross-check** on a 5k LOA random subset (BAL-excl,
   FILTER=1, max_dlas=4): is the catalog DLA count materially changed
   at log_NHI < 20.6? If yes, the production CDDF needs a re-run; if
   no, the change is purity-improvement only and can ship.

## Open questions for the user

1. **Confirm scope** — Option A first? Skip directly to B?

2. **For Option A — exact upper bound of the extended sub-DLA range.**
   - 20.3 (matches DLA lower bound, no gap, no overlap) — cleanest
   - 20.5 (overlaps DLA prior by 0.5 dex) — more aggressive, more
     boundary-soaking, but Bayes factor between sub-DLA and DLA
     models becomes ambiguous in [20.0, 20.3]
   - I recommend 20.3 unless you have specific reason for the wider
     overlap.

3. **For Option B — wide-range sub-DLA, what's the correct prior
   shape?** Prochaska+2014 is a power-law fit to known DLA columns;
   extrapolating it down to log_NHI = 17.2 is an extrapolation past
   the calibrating data. Two choices:
   - Pure uniform on [17.2, 20.3] (no Prochaska weighting)
   - Power-law extrapolation (matches Prochaska shape down to 17.2)
   - Hybrid (uniform below 19.5, Prochaska-shape above)

4. **Generate samples in MATLAB or port to Python?** The existing
   sample generation lives in Bird/Garnett's MATLAB code. Porting
   `generate_samples.py` to support the new ranges is ~1 day of work.
   MATLAB is a 1-line change but requires a working MATLAB env.

5. **Boundary-pile-up validation target**: when we say "the pile-up
   at log_NHI=20.3 is reduced", what threshold counts as success?
   - A 50% reduction in the integrated CDDF excess in [20.2, 20.4]?
   - A specific absolute number?
   - "Reduced to within Poisson noise of the smooth Prochaska CDDF"?

## What I will NOT do without explicit user sign-off

- Generate new sample files (.mat) — needs choice of prior shape
  (question 3 above)
- Modify `SubDLASamplesMAT` to load wider-range files — small but
  non-trivial; depends on questions 2 + 3
- Touch the `BayesModelSelect` engine — Option C only

## Once approved (Option A path)

1. Generate `subdla_samples_extended.mat` with the agreed range +
   prior shape.
2. Write a 1-line patch to `SubDLASamplesMAT.__init__` to accept
   wider extrapolate_min/max ranges (currently it just asserts; need
   to allow the range or relax the assertion).
3. Update `slurm/submit_desi_loa.sh` to point at the new file.
4. Run smoke + 50-target validation (steps 2 + 3 above).
5. Run 5k LOA cross-check (step 5 above).
6. If results are clean, run the full 50k mock-vs-LOA campaign with
   the new prior — surfaces both purity (boundary pile-up) and
   completeness (gap-filling) impacts.
