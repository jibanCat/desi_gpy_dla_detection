# Handoff — 2026-05-13 14:40 PT (jupyter 52907557 expires 15:44 PT)

> Big session. Var[Δ_marg] verdict landed, all 5 yesterday-in-flight runs
> resumed and finished, P/C eval done on all of them, 2-way joint-sweep
> cellC (single-absorber NHI [17.2, 22]) is a clear baseline-candidate
> winner, FILTER=1 knob tuning is the actionable lever to close the
> remaining completeness gap. 4 commits pushed to `production_533`
> (`c8ba76b..bb218c5`). PR #7 diff auto-updated.

## TL;DR for next-Claude

1. **Today's headline result**: cell C of the joint sweep
   (`SINGLE_ABSORBER_MODEL=1`, MAX_DLAS=3, NHI prior `[17.2, 22]`, PW 50k)
   gives **P=0.83 / C=0.83 at SNR>2, P_DLA≥0.99 vs dla_cat NHI≥20.3** —
   beats the FILTER=1 v3 baseline by **+7 pp completeness at only −2 pp
   purity**. This is the closest any tested config gets to balanced 85/85
   so far. Read [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md)
   before deciding next steps.

2. **Var[Δ_marg] verdict (done today)**: pipeline is statistic-limited
   at production N=50k, not sampling-limited. σ_noise ≈ 0.1 vs signal
   gap ≈ 13. Drop bespoke MLMC / pocoMC for the SNR>2 ceiling. Read
   [`docs/notes/2026-05-13_var_delta_marg_diagnostic.md`](docs/notes/2026-05-13_var_delta_marg_diagnostic.md).

3. **FILTER=1 has tunable knobs** that should close the completeness
   gap vs FILTER=0 (which is the same Bayesian computation, slower). Per
   user reframe: don't switch to FILTER=0; tune FILTER=1. The
   knob doc [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md)
   has a 2×2 ablation matrix ready to run.

4. **Compute environment is bad right now**: regular_milan_ss11 queue is
   **11 170 jobs deep**, sbatch start estimate is **2026-05-23** (10 days
   from now). All inference today ran inline on the jupyter compute node
   via `slurm/resume_local.sh` + `nohup` + `disown`. The session
   expires 15:44 PT — next compute should either be a fresh jupyter or
   `salloc -q interactive` (much faster queue).

5. **2 untouched workstreams continue elsewhere**: (a) trainer PR on
   GreatLakes is running the LOA-no-HCD-with-BAL retrain after PR #6
   fixed the v2 preload normalization bug; (b) PR #7 is still draft.

---

## Today's P/C results — single comparison table (SNR>2, P_DLA≥0.99, lyb-veto, no-BAL, full forest λ_rf∈[911, 1216])

| Variant | Purity | Completeness | cat rows | truth | vs baseline |
|---|---:|---:|---:|---:|---|
| **baseline** v3+PW14[19,22]+τ-EB+FILTER=1+md=3 | 0.8452 | 0.7661 | 1242 | 618 | reference |
| early_stop_A (no null-vs-current early-stop) | 0.8466 | 0.7749 | 1468 | 618 | +1pp C, more cat |
| early_stop_D (pre-Occam likelihood for null cmp) | 0.8466 | 0.7749 | 1399 | 618 | +1pp C, modest cat |
| NFL=31 (test trainer-mismatch hypothesis) | 0.8534 | 0.7661 | 1241 | 618 | **null** — reject |
| FILTER=0 (s1+s3 only, n_truth=54) | 0.8519 | **0.8846** | 144 | 54 | **+12pp C** (suggestive, undersized) |
| joint cellA: SINGLE_ABS=1, md=3, NHI[19,23], PW50k | 0.7906 | 0.8392 | 2668 | 618 | −5pp P, +8pp C |
| joint cellB: SINGLE_ABS=1, md=4, NHI[19,23], PW50k | 0.7950 | 0.8392 | 2912 | 618 | similar to A |
| **joint cellC**: SINGLE_ABS=1, md=3, NHI[17.2,22], LLS50k | **0.8256** | **0.8304** | 3268 | 618 | **−2pp P, +7pp C — best yet** |

**Source**: `examples/molly_faithful_pc_plots.py` against London mock-0
`dla_cat.fits` (NHI≥20.3 truth). Per-variant log files:
`/pscratch/sd/j/jibancat/prod533_5k_20260511/resume_local_logs/pc_*.log`.

### How to interpret each row

- **early_stop A/D** (today's fix variants): Both lift completeness by
  ~1 pp without hurting purity much, at the cost of more catalog rows
  (more multi-DLA hypotheses survive). Marginal value. **Default
  EARLY_STOP_MODE=baseline still reasonable.**

- **NFL=31**: Tested whether NUM_FOREST_LINES mismatch with the
  trainer (user belief: trainer used 31) was the issue. Today's A1 agent
  verified the v3 GP was **actually trained at NFL=3** — so NFL=31 at
  inference is *more* lines than training. Result: indistinguishable
  from baseline. **Reject the hypothesis. Drop NFL=31.** The
  `submit_desi_{mock,loa}.sh` default of 31 is a latent bug.

- **FILTER=0** (only s1+s3 ran due to wall-time budget): +12 pp
  completeness in headline. Same-n_truth (54) comparison: 0.769 → 0.885
  C. Direction is consistent across all 4 P_DLA cuts. Statistical
  significance ~1.1 σ (n_truth=10 in [20.3, 20.6) bin). **Promising but
  needs full 8-slice confirmation.** See `docs/notes/2026-05-13_filter_nfl_confirmation.md`.

- **Joint sweep cells A/B/C** (today's joint catalog test — Option B
  from `project_subdla_dla_joint_design` memory): single DLA model
  with widened NHI prior. All three improve completeness +7-8 pp. **Cell
  C ([17.2, 22] prior, the LLS-extended one) wins on purity-completeness
  balance.** The [19, 23] prior (cells A/B) hurts purity more (−5 pp).
  These were run with `SINGLE_ABSORBER_MODEL=1` so the model is 2-way
  [null, k-DLA-with-widened-prior].

### NHI-bin-stratified completeness — cellC vs baseline

Computed via `_nhi_bin_compare.py` (which reuses molly_faithful helpers).
Same operating point: SNR>2, P_DLA≥0.99, lyb-veto, no-BAL, λ_rf ∈ [911, 1216].
Counts here are post-all-cuts (the molly `n_*_post_cuts` are pre-P_DLA-cut,
so they're larger; the per-bin C ratios are correct either way).

**Completeness per truth-NHI bin:**

| Bin | baseline | cellC [17.2, 22] | Δ |
|---|---:|---:|---:|
| [20.3, 20.5) | 62/108 = **0.574** | 76/108 = **0.704** | **+0.130** |
| [20.5, 21.0) | 129/158 = 0.816 | 138/158 = 0.873 | +0.057 |
| [21.0, 21.5) | 58/62  = 0.935 | 57/62  = 0.919 | −0.016 |
| [21.5, 22.0) | 13/14  = 0.929 | 13/14  = 0.929 | 0.000  |
| **overall**  | 262/342 = **0.766** | 284/342 = **0.830** | **+0.064** |

**Purity per predicted-NHI bin** (using cat NHI for the bin assignment):

| Bin | baseline | cellC [17.2, 22] | Δ |
|---|---:|---:|---:|
| [20.3, 20.5) | 42/72  = 0.583 | 57/95  = 0.600 | +0.017 |
| [20.5, 21.0) | 136/146 = 0.932 | 139/152 = 0.914 | −0.018 |
| [21.0, 21.5) | 63/70  = 0.900 | 66/74  = 0.892 | −0.008 |
| [21.5, 22.0) | 21/22  = 0.955 | 22/23  = 0.957 | +0.002 |
| **overall**  | 262/310 = **0.845** | 284/344 = **0.826** | **−0.019** |

**The cellC win is concentrated almost entirely in the [20.3, 20.5)
regression bin** (+13 pp completeness). Mid-NHI [20.5, 21.0) picks up
another +6 pp. Strong DLAs (NHI ≥ 21.0) are flat — those weren't broken.
Purity drops a uniform ~2 pp because the wider NHI prior +
`single_absorber_model=1` admits more cat candidates per spectrum; some
are spurious. Notably the weakest bin's purity actually *rises* slightly
(0.58 → 0.60).

**Why this maps onto the FILTER=1 knob-tuning story**: cellC uses the
SAME FILTER=1 algorithm with a wider NHI prior `[17.2, 22]` plus
`single_absorber_model=1`. The wider prior gives the FILTER=1 coarse
5000-sample scan a better chance of finding a sample near a weak truth's
high-likelihood mode → fewer spectra hit the "early-stop on empty
valid_mask" branch (`dla_gp.py:635`). This is mechanistically the same
fix as knob 1 (`n_initial` floor) and knob 4 (empty-mask fall-through)
in [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md).
Cell C achieves the fix "for free" by widening the prior support; tuning
the FILTER=1 knobs directly should achieve the same [20.3, 20.5) recovery
**without** the −2 pp purity hit, because the knob tuning fixes the
coarse-scan miss without changing the prior support.

The script `_nhi_bin_compare.py` at the repo root is the analysis tool
that produced these tables. Untracked (matches the `_*.py` scratch
convention from earlier sessions). Re-run with the same DESI env preamble.

Other variants' per-bin completeness in [20.3, 20.5) for reference:
- early_stop_A: 0.602 (modest +3 pp vs baseline)
- early_stop_D: 0.593 (modest +2 pp)
- NFL=31: 0.574 (=baseline, no effect — confirms NFL is irrelevant)
- FILTER=0 (s1+s3 only, n_truth=6 in this bin): 0.833 (suggestive, undersized)
- cellA [19,23] md3: 0.722
- cellB [19,23] md4: 0.694

---

## What was committed today

| Commit | Title |
|---|---|
| `2c499a8` | feat: EARLY_STOP_MODE flag for multi-DLA inference + resume scripts |
| `86ad225` | diag: Var[Δ_marg] gating diagnostic + 2026-05-13 verdict note |
| `f08d63b` | docs: 2026-05-13 handoff — Var[Δ_marg] verdict, resume status, lessons |
| `c8ba76b` | docs: production runbook + model-side improvements suggestions |
| `bb218c5` | docs: runbook fixes + BAL scope + FILTER/NFL confirmation + model-side updates |

This handoff will be the 6th commit (push at end of session).

---

## Pickup priorities for next-Claude

### Priority 1 — Run the FILTER=1 knob 2×2 ablation

Read [`docs/notes/2026-05-13_filter1_knob_tuning.md`](docs/notes/2026-05-13_filter1_knob_tuning.md).
The 4-cell ablation:
- (n_initial floor, empty-mask fall-through) ∈ {(5000, no), (10000, no), (5000, yes), (10000, yes)}
- Goal: identify whether the FILTER=1 completeness gap is from coarse-scan miss
  (knob 1), early-stop on empty mask (knob 4), or both.
- Cost: 4 × 30 min wall in parallel = 30 min on one 256-CPU node.
- Code changes needed before running: parametrize `n_initial` and add a
  fall-through branch at `dla_gp.py:635`. Touches `dla_gp.py`,
  `run_bayes_select.py`, `desi-DLAGP.py`, `slurm/run_local.sh`.

This is the most direct path to closing the completeness regression
without paying FILTER=0's 2.4× cost.

### Priority 2 — Validate cellC as the new production baseline

Cell C looks like the best operating point in today's data (0.83/0.83
balanced). Before claiming it:
1. Verify on full London 26k (today was just 5k 8f).
2. Verify on Saclay and 2LPT (today was London-only).
3. Verify on real LOA (today was mock-only).
4. NHI-bin-stratified completeness — esp. [20.3, 20.6) — make sure cell C
   doesn't have a different pathology than the v3 baseline.

If cellC validates at full scale, it becomes a **strong candidate** to
ship as the production DLA catalog. Note: cellC's catalog row count is
3268 (vs baseline 1242), so the post-cut work is harder (more candidate
DLAs to filter). The lyb_veto + P_DLA≥0.99 already handle this in the
P/C eval; production catalog work needs to follow the same recipe.

### Priority 3 — Sub-DLA P/C eval on cellA/B/C against truth

Today's cellA/B/C P/C was at NHI ≥ 20.3 (classical DLAs). The whole
*point* of widening the NHI prior to [19, 22] or [17.2, 22] is to ALSO
get sub-DLA detection. Need to:
1. Re-run `examples/molly_faithful_pc_plots.py` with `--truth-nhi-min 19.0`
   `--nhi-min 19.0` and a truth filter `NHI ≤ 20.3` on each cell.
2. Compare against the LLS truth catalog `hcd_truth_cat.fits` (Saclay) or
   filtered `dla_cat.fits` (London).
3. Report the 2×3 table: (classical DLA, sub-DLA) × (cellA, cellB, cellC).

This is the "Option B" deliverable from `project_subdla_dla_joint_design`
memory.

### Priority 4 — Wait for trainer-PR retrain to land + retrain validation

The user has a retrain running on GreatLakes (`loa_no_hcd_with_bal` model
+ PR #6 v2-preload-bug fix). When it lands:
1. Convert the trained .h5 to inference format (`null_gp_test/converted/`
   pattern, see existing v3 conversion).
2. Run the 5k London smoke comparison: new-trained-model vs v3 model.
3. P/C delta at the canonical operating point.

### Lower priority / parked

- **BAL masking smoke test** ([`docs/notes/2026-05-13_bal_masking_scope.md`](docs/notes/2026-05-13_bal_masking_scope.md)):
  4-line config + 30 min run, falsifies whether `--balmask` lets us
  recover BAL QSOs in the catalog. Worth running but not blocking.
  Also fix `constants.bal_lines` duplicate-CIII bug (line 85 vs 95).
- **K-rank sweep** (Tier 1.2 in
  [`docs/notes/2026-05-13_model_side_improvements.md`](docs/notes/2026-05-13_model_side_improvements.md)):
  needs trainer-side action on GreatLakes after PR #6 merges.
- **Sub-DLA prior overlap test** (Tier 2.0): does `[19, 20.3]` sub-DLA
  prior improve DLA detection? Open question, untested.

---

## Compute env — what's running, what's expired, what's queue-bound

- **Current jupyter**: job 52907557 on `nid004210` → wait, that's wrong;
  it's `nid004179` per `scontrol show job 52907557`. 256 CPUs, 487 GB
  mem, expires **15:44:40 PT** (= ~1 hr from this handoff write time).
- **Background procs**: should be quiet now — all P/C eval done, all 5
  resume runs finished, all 2 confirmation runs finished. Verify via
  `pgrep -af desi-DLAGP.py | wc -l` (expect 0 or near-0).
- **Regular sbatch queue**: 11 170 jobs pending, start estimate 10 days
  out. **Do not submit large sbatch jobs today** — won't run.
- **Interactive queue** (`-q interactive`): much shorter wait. For
  next-session compute, prefer salloc on interactive over regular queue.
- **GreatLakes**: separate cluster, user's trainer PR is running there.

---

## File-by-file summary of today's commits

- `2c499a8`: EARLY_STOP_MODE plumbing (5 files in core + 2 new scripts in
  slurm/). Default "baseline" → bit-for-bit production unchanged.
- `86ad225`: Var[Δ_marg] re-analysis script + 2026-05-13 verdict note.
  ~70 s wall to reproduce.
- `f08d63b`: this morning's first handoff.
- `c8ba76b`: production runbook v1 + model-side improvements v1. Stale
  for hours — superseded by `bb218c5`.
- `bb218c5`: runbook corrections (regression callout, NUM_FOREST_LINES
  verified-trained-at-3, sub-DLA table, [19,22] prior verified, node-hours
  re-validated, 2-way subDLA mode documented), BAL scope, FILTER+NFL
  confirmation note, model-side improvements v2.

**Uncommitted at handoff write time**: this `HANDOFF.md` itself, the new
`docs/notes/2026-05-13_filter1_knob_tuning.md`. Will commit + push at the
end of the session.

---

## Memory state (per `/global/homes/j/jibancat/.claude/.../memory/MEMORY.md`)

No new memory items added today; existing memory items still load-bearing:
- `feedback-long-runs-need-sbatch` (updated this morning: 3 rules; the
  queue-vs-jupyter exception was vindicated today).
- `project-base-branch` (`desi_y3`).
- `feedback-snr-canonical` (SNR_RED > 2).
- `project-prior-dilution-finding` (now substantially refuted by today's
  Var[Δ_marg] verdict — should add a "superseded by 2026-05-13_var_delta_marg_diagnostic"
  note next session).
- `project-subdla-dla-joint-design` (today's cellC result corroborates).

---

## Key files to read in this order on next session

1. `HANDOFF.md` (this file)
2. `docs/notes/2026-05-13_filter1_knob_tuning.md` (the actionable knob doc)
3. `docs/notes/2026-05-13_var_delta_marg_diagnostic.md` (the verdict that redirects sampler work)
4. `docs/notes/2026-05-13_filter_nfl_confirmation.md` (today's confirmation runs)
5. `docs/production_runbook.md` (full production runbook with today's corrections)
6. `docs/notes/2026-05-13_model_side_improvements.md` (trainer PR roadmap)

---

## Open questions for human at next session

1. **Cell C ([17.2, 22] single-absorber-mode) as the new production
   baseline?** Today's evidence (0.83/0.83 vs baseline 0.85/0.77)
   strongly favors it. Validate on full London 26k + Saclay + 2LPT + real
   LOA before claiming.
2. **FILTER=1 knob 2×2**: which knob actually matters? Run today's
   ablation matrix to find out.
3. **Sub-DLA P/C from cellA/B/C** — what's the [19, 20.3] truth match
   look like? This is the "is cellC also a usable sub-DLA catalog"
   question.
4. **EARLY_STOP_MODE A or D for production**? Today's data says +1 pp C
   for both, modest. Probably defer until cellC / FILTER=1 tuning lands.
