# Fresh-eyes review: PR #6

> Reviewer: cold reader (no prior context with PR #6).
> Branch: `claude/debug-trainer-from-v1` → `desi_y3`.
> Approach: started at `README.md`, followed the trail.
> Goal: surface confusion a new contributor would hit, NOT re-litigate math
> already validated by the three prior reviewers (dataset-math 2026-05-13,
> PR-diff 2026-05-13, full PR 2026-05-14).

## What I think this PR is doing (in my own words)

PR #6 rebuilds the GP continuum trainer. The repo had three generations of
trainer code (v1 MATLAB-faithful, v2 randn-init + autograd which silently
trains broken models, v3 PR-#6 working area). PR #6 lands the corrected v3
trainer — PCA-initialised M plus hand-coded analytic gradients running on a
batched (`spectrum_loss_batch`) loss — and trains 6+ models on 2LPT and LOA
preloads as the first end-to-end DESI deliverable. Three milestones called
Step A / Step B / Step C structure the work: Step A re-verifies the per-spectrum
loss matches MATLAB to 1e-11, Step B batches it (28× speedup at 5k), Step C
ports the trainer to DESI v2 preloads (`tests/phase2_train_desi.py`) and emits
13 trained models in `docs/notes/2026-05-11_desi_phase2_*/`. Along the way the
PR catches a real MATLAB↔Python divergence in `dataset.py` (mask order),
re-orders normalize → mask, and ships a regression test. Net behavioural
change for inference: zero unless you load a Step C `.h5`; the inference code
auto-detects v2 schema (`gpy_dla_detection/null_gp.py:478-482`).

## Concepts I had to puzzle out

| Term | Where I encountered it | Where it's defined | Hours of confusion |
|---|---|---|---|
| **Step A / Step B / Step C** | `docs/notes/2026-05-13_pr6_description.md:18-23`, `docs/training_overview.md:21`, `docs/production_models.md:43` | `docs/training_overview.md:174-184` (only) — `## Status of PR #6` table. **No standalone glossary; the README never mentions Step A/B/C.** | 20 min — had to grep the whole tree |
| **`loa-0` / `loa-124`** | `docs/production_models.md:21`, `docs/notes/2026-05-11_desi_phase2_2lpt_loa0_wide/` (dir name) | `docs/notes/2026-05-13_2lpt_loa0_vs_loa124_implementation/findings.md:13-18` and `docs/production_models.md:197-199`. Numbers imply a continuum (loa-1, loa-2, … loa-124?); it is **binary** — loa-0=uncontaminated, loa-124=contaminated. Naming derives from `qq_desi_y3/v2.8.5/mock-0/{loa-0,loa-124}/` upstream LyaCoLoRe layout. | 15 min |
| **`_g` / `_m` suffixes** | every model card in `docs/notes/2026-05-11_desi_phase2_*_g/` and `*_m/` | **Nowhere explicit.** I had to compare three model READMEs to infer: `_g` = "Garnett+2017 normalization band [1310, 1325]" + strict Turner σ; `_m` = "MATLAB DR16 normalization band [1425, 1475]" + strict Turner σ; the unsuffixed variant = wide BOSS DR12Q σ + [1310, 1325]. The print convention labels in `gpy_dla_detection/training/dataset.py:362-369` quietly imply this but the model cards never say it. | 30 min |
| **`wide`** | dir names like `2lpt_loa0_wide`, `loa_no_dla_no_bal_wide` | **Implicit.** Comparing to `docs/production_models.md:170-174` it means "wide v2 rest grid [850.75, 1700], n_pix=5662, dλ=0.15" vs v1's [850.90, 1420.60] n_pix=3798. No glossary. | 10 min |
| **`normmask`** | `desi_smoke_normmask`, `*_m_normmask` in `docs/production_models.md:138-141` | Inferable: post-`aa36205` (normalize → mask reorder) retrain. **No definition.** Reader could plausibly think it means "normalize and apply mask" (i.e. both flags on). It really means "post-reorder pipeline". | 10 min |
| **`c0prior`** | dir `docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/` | The README banner at lines 2-17 explains the outcome but not the **mechanism**: it's a Gaussian prior on `log_c_0` added to the Adam loss to break the factor-analysis (c_0, M) gauge degeneracy. Defined in `tests/phase2_train_desi.py:84-91` and `:78-91` (comment), but no doc surface points there. | 15 min |
| **`v2 preload` / `trainset.h5`** | `tests/phase2_train_desi.py:35`, SLURM scripts | `docs/v2_runs_layout.md` (had to find it) plus `gpy_dla_detection/training/dataset.py::load_preprocessed_h5`. Reasonable to find but not linked from production_models.md. | 5 min |
| **`pre-reorder` / `post-reorder`** | `docs/production_models.md:43, 94-100`, all model READMEs | The 2026-05-13 fix in `dataset.py` swapped normalize↔mask. "Pre-reorder" = trained before commit `aa36205`. Banner in each README explains, but the term lives or dies on the reader knowing what was reordered. | 5 min |

**Net glossary gap**: the PR introduces eight new short-form labels that
together are load-bearing for choosing a model. None of them are defined in a
single discoverable place. `docs/production_models.md` is the closest thing,
and even it leaves `_g`, `_m`, `wide`, `normmask`, `c0prior` to inference.

## Confusing or under-documented bits

1. **The README doesn't acknowledge PR #6.** `README.md` ends at the τ-EB
   recipe (line 264-298). A new contributor reading top-down has no signal
   that there is a new trainer, a new schema, 13 new models, or a Step A/B/C
   workflow. `docs/training_overview.md` is reachable only by knowing it
   exists. **Suggest**: a "## Training your own GP model" subsection in
   README pointing at `docs/training_overview.md` and `docs/production_models.md`.

2. **`docs/notes/2026-05-13_pr6_description.md` (the PR-body draft) assumes
   prior context.** Line 1 jumps straight to "Status — 2026-05-13 EOD" with no
   one-paragraph "what does this PR do at all". The summary section starts at
   `:16` and works, but a stranger lands at the top and reads "Three
   independent agent audits today recommend SHIP" first — not orienting. The
   bullet list at `:18-23` is a good summary but doesn't reach the reader
   until they've scrolled past two tables.

3. **Step A / Step B / Step C** are referenced 47+ times across the PR but
   *defined* only inside a table at `docs/training_overview.md:174-184`
   titled "Status of PR #6 (this file written mid-PR)". The table reads like
   internal status, not a glossary. The terms should be promoted to a named
   section, or at least the PR description should open with a one-line
   "Step A = MATLAB parity verification, Step B = vectorisation, Step C = DESI
   retrain" so readers don't have to reverse-engineer the milestones.

4. **`_m` is the recommendation but `_g` is what the SLURM defaults
   train.** `slurm/greatlakes/phase2_desi_retrain.sh:68-69` defaults to
   `NORM_MIN_LAMBDA=1425.0`, which is the `_m` band — good. But the in-flight
   table in `docs/production_models.md:121-125` shows BOTH `_g` and `_m`
   variants being retrained, with no doc explaining why we still train `_g` if
   `_m` is recommended. (Implicit answer: to be able to compare. But state it.)

5. **The c0prior README banner** (`docs/notes/2026-05-11_desi_phase2_2lpt_loa124_nohcd_nobal_wide_c0prior/README.md:2-17`)
   says "Not preferred for production — use `_m` instead" and explains the
   13× `‖M‖²` finding. **Good** — this addresses the previous reviewer's
   concern. But the banner doesn't define what the c0prior **recipe** is
   (`tests/phase2_train_desi.py:84-91` is the only place). A new reader sees
   "c0prior" in the dir name, finds a banner saying don't use it, but doesn't
   know what was being tried. **Suggest**: 2 lines at the bottom: "This
   variant adds a Gaussian prior on `log_c_0` with σ=… to break the (c_0, M)
   factor-analysis gauge degeneracy. The prior failed to anchor."

6. **`production_models.md` decision matrix (§"Recommendations by use case",
   `:176-227`) is actionable** for the four use cases listed (sampler
   correctness, 2lpt mock, real LOA, single-spectrum smoke). The
   in-flight/forthcoming tables at `:114-141` introduce JobIDs and "TBD" rows
   that age fast — they will be stale within a week. Consider marking those
   tables explicitly as "as of 2026-05-13; check git log for fresher state".

7. **The dataset.py reorder is documented in the source code** (`gpy_dla_detection/training/dataset.py:336-355`)
   with an excellent inline `ORDERING NOTE (2026-05-13)` comment. The first
   reviewer noted this. **No issue here** — the code is self-explanatory; my
   concern is only that a reader of `dataset.py` who doesn't know about the
   preceding bug might be confused about why such a long block-comment exists.
   The comment is worth keeping but a one-line "see `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`"
   at the top would short-cut that.

8. **The h5 manifest's new fields are self-describing per-key**
   (`tests/phase2_train_desi.py:529-562` writes well-named keys like
   `num_forest_lines`, `optimizer`, `git_commit_sha`). **No issue.** But
   nothing in `docs/` enumerates the schema for the reader who picks up a
   `.h5` and wants to know what every key means. The closest is
   `tests/phase2_train_desi.py:376-393` (the schema table inside the README
   template). Promote it to `docs/learned_model_schema.md`.

## Trail-of-bread-crumbs test

I picked `docs/notes/2026-05-11_desi_phase2_2lpt_loa0_wide_g/` cold (no prior
context) and tried to answer (a) why it exists, (b) how to reproduce it,
(c) whether it's production-ready, (d) what caveats apply.

- (a) **Why does this model exist?** The README (lines 13-15) says it's
  trained by `tests/phase2_train_desi.py`. The `_g` suffix is not defined.
  The README says norm band is `[1310.00, 1325.00] Å rest (Garnett+2017
  convention)`. To figure out it's the "Garnett+2017 band variant with strict
  Turner prior σ" I had to compare to `_m` and the unsuffixed model. **Partial
  pass** — provenance through SLURM job is "(not tracked)" in this model,
  which means the chain back to a sbatch invocation is broken.

- (b) **How to reproduce it?** The README doesn't tell me. `slurm/greatlakes/phase2_desi_retrain.sh:30-36`
  shows the sbatch usage. To reproduce `_g` specifically I have to know to
  set `NORM_MIN_LAMBDA=1310.0 NORM_MAX_LAMBDA=1325.0` and **also** what the
  strict Turner σ knob is — which I'd have to read in `tests/phase2_train_desi.py:75-82`.
  **Fail** — the model card should contain the exact `sbatch --export=...`
  command that reproduces it.

- (c) **Production ready?** Banner at `:3-11` says no, it's pre-reorder and
  recommends `_m`. But then `docs/production_models.md:21` says `_m` IS
  recommended (for 2lpt mocks). **Pass.**

- (d) **Caveats?** Banner mentions the corr-roughness; the SLURM job link is
  missing; β=2.69 is not flagged here even though `docs/production_models.md:148-153`
  flags β drift as caveat #2 globally. **Partial pass** — global caveats
  should be cross-linked from each model card.

Verdict: the trail works only if the reader knows to consult
`docs/production_models.md` as the index. Without that pointer the model
cards are isolated.

## "Reproduce a Step C training" walkthrough — where I got stuck

I pretended to be tasked with retraining on a new mock (say a hypothetical
`saclay_mock0_wide_m`). Walking through:

1. **Find the entrypoint.** README says nothing. `docs/training_overview.md`
   says use `tests/phase2_train_desi.py`. **OK**, 5 min.
2. **Find an example invocation.** `tests/phase2_train_desi.py:25-39`
   docstring gives a smoke command and a production command. **OK**, 1 min.
3. **Need a preload (`trainset.h5`).** The docstring shows
   `/nfs/turbo/.../v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5`. **How
   do I make one for Saclay?** Grep shows `preload_spectra/preload_2lpt_simple.py`
   and `preload_spectra/preload_loa_real.py`. No `preload_saclay_*.py`. **Got
   stuck — 25 min.** The `training_overview.md:120-129` table mentions the
   preload modules but doesn't tell me which to use for a new mock. Reading
   `preload_2lpt_simple.py` source eventually reveals it's parameterised but
   the SLURM wrappers for it aren't listed anywhere.
4. **Choose hyperparameters.** SLURM script `phase2_desi_retrain.sh:55-85`
   exposes them all. **OK**, but `LOG_C_0_PRIOR_SIGMA="${LOG_C_0_PRIOR_SIGMA:-}"`
   (line 70) is an empty default, and the comment says "optional Gaussian
   prior on log_c_0 to prevent gauge collapse". From the c0prior README I
   know that didn't work. So **what should I set it to?** Nothing in
   `docs/` says "leave it empty for production".
5. **Submit the job.** SLURM template at lines 30-37 works. **OK**.
6. **Verify the result.** README is auto-emitted. **OK** — and per
   `tests/phase2_train_desi.py:362` the bug from the previous reviewer is
   fixed; the band is now templated correctly.
7. **Compare to v1 production.** `examples/dla_recovery_step_c.py` exists
   (4 commits ago) but has no docstring at the top — I have to read 30 lines
   of source before I can guess what it does. **15 min.** Once I figure out
   it's a canonical-TID smoke test, fine.

**Net**: about 70 minutes for a reasonably experienced engineer who is new
to the repo, with two stuck points (preload selection for a new mock; the
`log_c_0` knob's recommended value).

## Documentation gaps

- **No glossary** for `_g`, `_m`, `wide`, `normmask`, `c0prior`, `loa-0`,
  `loa-124`. Single Markdown table at the top of `docs/production_models.md`
  fixes this in 10 lines.
- **`tests/phase2_train_desi.py`** has a 14-line module docstring (lines
  1-39) — **excellent**, this is the best-documented file in the PR. Compare
  to `tests/phase2_train_dr16.py`, which I'd expect to be similar; spot-check
  shows it is.
- **`docs/training_overview.md:174` "Status of PR #6"** table is marked
  "this file written mid-PR" — once PR #6 merges, that line becomes false.
  Update at merge.
- **`docs/notes/2026-05-13_pr6_description.md:84`** claims 224 tests but
  earlier reviewers said 225; the full-PR review (`2026-05-14_full_pr_review.md:84`)
  says 224. PR description is correct as of latest pull but needs a final
  re-count before tagging the PR.
- **`run_bayes_select.py:354` docstring** still says
  `tau_eb_factors=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0)` while the actual default
  has 8 elements. Flagged by previous reviewer (full PR review §"Docstring
  drift"); not yet fixed.

## TODOs / FIXMEs in production code paths

I greppped:
- `tests/phase2_train_desi.py:69`: `INITIAL_BETA = TAU_0_PRIOR_MU * 0 + 3.62  # avoid stale-cache typo`
  — clearly a workaround comment that's now confusing. Line 70 redundantly
  sets `INITIAL_BETA = 3.62`. **Cleanup nit**: drop line 69.
- `desi_learn_qsos_model.py:180`: pre-existing TODO from v1, just copied
  into `training_v3/`. Not introduced by this PR.
- `preload_spectra/preload_from_loa_archive.py` docstring TODO. Low impact.

## PR cohesion check

This PR feels like **3 coherent pieces and 1 surprise**:

1. **The trainer rebuild (Steps A/B/C)** — coherent, well-staged, validated.
   Lines roughly: `gpy_dla_detection/training_v3/*` + `tests/phase2_train_*.py` +
   SLURM scripts + tests.
2. **The corr-noise debug arc (`dataset.py` reorder + threshold + probe)** —
   coherent, falsifiable, well-tested. Strictly a bug fix discovered DURING
   the Step C runs; could in principle have been a separate PR, but the
   models in this PR depend on it (post-reorder retrains supersede
   pre-reorder), so co-shipping makes sense.
3. **13 trained models + model cards** — coherent product of (1) and (2).
4. **The surprise**: `gpy_dla_detection/postprocess/`, `voigt_v2.py`,
   `tau_eb.py`, LoaArchive — these touched files appear in the diff but are
   not the focus. Some are PR #5 material (τ-EB, voigt_v2, postprocess)
   that's already merged into `desi_y3` and just shows up in this branch's
   diff because the branch is many commits ahead of the merge base I
   examined. I confirmed by checking `git log --oneline desi_y3..HEAD` would
   filter these — they're not new in PR #6. **Not a cohesion issue** but a
   new reviewer scanning the diff sees +33,917 lines and panics; only a
   fraction is actually PR #6.

Overall: **coherent for what it claims to be**. The PR description correctly
scopes the changes; the "many commits, much line churn" feels like a 76-commit
investigation, not duct-tape.

## Recommendations for new-contributor onboarding

In priority order (effort is the dimension I'd optimise; all are <50 lines):

1. **One glossary** at the top of `docs/production_models.md` defining
   `_g`, `_m`, `wide`, `normmask`, `c0prior`, `loa-0`, `loa-124`. 10 lines.
2. **One README diff** adding a "Training your own GP model" subsection
   that points at `docs/training_overview.md` and `docs/production_models.md`.
   5 lines.
3. **One sentence in each model card** saying what hyperparameter knob
   produced the suffix. e.g. `_g` README: "The `_g` suffix indicates the
   Garnett+2017 normalization band [1310, 1325] Å. Reproduce with
   `NORM_MIN_LAMBDA=1310.0 NORM_MAX_LAMBDA=1325.0 sbatch …`". Could be
   added via `examples/reemit_step_c_readmes.py` in 10 lines.
4. **Promote the Step A/B/C definitions** out of the mid-PR status table at
   `docs/training_overview.md:174` into a labelled section near the top.
   5 lines.
5. **Add a top-level docstring** to `examples/dla_recovery_step_c.py` and
   `examples/reemit_step_c_readmes.py`. 6 lines each.
6. **Add a `docs/learned_model_schema.md`** describing the 9-key v2 schema +
   the manifest extension. Lift the table from `tests/phase2_train_desi.py:376-393`.
   ~30 lines.

None of these block the merge — the science ships and the prior reviewers
correctly recommended SHIP. The above improves onboarding for the next
contributor, which is the next-most-leveraged thing after the science.

## Final verdict

I agree with the three prior reviewers: the math, the tests, the inference
safety, the caveat discoverability (after the 2026-05-14 c0prior banner
update) are all in good shape. My added contribution is the cold-reader
glossary debt — which is a real cost on a project this size but cheap to
amortise across a couple of follow-up commits.

**Ship.** Prefer the 6 onboarding fixes above land before merge if cheap;
otherwise queue them as the first item in a "PR #7 docs polish" follow-up.
