# BAL-only GP — design proposal

> **Status**: Draft for user review before any training is launched.
> **Author**: Claude, 2026-05-02 session.
> **Decision needed**: which of the two scopes (A vs B) we ship in
> the post-PR-#5 work.

## Context

Standard production runs include BAL QSOs without masking (CLAUDE.md §6).
The standard GP's μ is therefore an average over both BAL and non-BAL
QSOs. This is fine for the DLA-detection statistic *on average*, but it
has known failure modes near the BAL absorption troughs (typically
C IV 1548, Si IV 1394, N V 1240, sometimes Lyα 1216 itself):

- The mean continuum μ at those rest wavelengths is depressed.
- Per-pixel absorption noise `omega^2` captures *some* of the variance,
  but the BAL pattern is too systematic to be modeled as pixel noise.
- DLAs near C IV / Si IV troughs are misidentified at non-trivial rates
  (the GP "explains" the trough as a DLA absorption feature).

A BAL-aware DLA catalog needs *some* model that distinguishes
"this is a QSO with broad outflows" from "this is a DLA". Two paths.

## Option A — BAL-only μ + Bayesian model selection at inference

**Train:** run `train_gp.py` on the `loa_bal_only` trainset (BAL QSOs
only, identified via `BI_CIV > 0`). Output: a separate `.h5` GP whose
μ is the average BAL-QSO continuum. Same code path as production
training; no new inference code.

**Inference:** for each QSO, build BOTH GPs (standard + BAL-only),
compute `log_evidence` under each (no DLA in the model), and pick the
higher-evidence one as the "continuum model". Then run the existing
DLA inference on top of the chosen continuum.

**Pros:**
- Reuses 100% of existing training + inference code. Cheap (~5 h NERSC
  for the train; ~2× inference cost since we build two GPs per QSO).
- The "BAL detection" is a side effect of model selection: if the
  BAL GP wins, the QSO is BAL.
- Decouples cleanly from DLA detection — DLA results just become
  conditional on the chosen continuum.

**Cons:**
- Doesn't model *individual* BAL variability. BAL absorption ranges
  enormously (BI_CIV from 0.1 to ~10⁴ km/s, multiple troughs at
  different velocities, "mini-BAL" vs "broad" vs "low-ionization"
  morphologies). The single BAL-only μ will be the *average* BAL —
  fits the median QSO well, fits high-BI tails poorly.
- Doesn't reduce DLA false positives at BAL troughs unless the BAL
  GP's μ is pulled enough to "explain" the depression as continuum.
  The Voigt-shape DLA profile is still a much better fit to a deep
  trough than a smooth μ depression.

**Implementation:** zero code changes. Just need the BAL trainset on
disk + train + a small inference-side glue function that compares
log evidences (probably ~30 lines in `run_bayes_select.py`).

## Option B — BAL as a stochastic absorber, modeled like DLAs

**Train:** keep the standard GP. Don't train a separate BAL GP.

**Inference:** add a BAL absorption model to the Bayesian model
selection. Analogous to how DLAs are modeled (Voigt profile + sample
grid), but with BAL physics:
- **Profile shape**: a single broad gaussian-like trough in the QSO
  rest frame (not the absorber rest frame). Width 1000–10,000 km/s,
  central optical depth 0.1–10.
- **Position prior**: blueward of C IV 1548 by 0–30,000 km/s
  (consistent with outflow physics; BAL ≠ Lyα-forest absorber).
- **Sample grid**: O(10⁴) samples of `(velocity_offset, width,
  central_depth)` — analogous to `dla_samples_a03.mat`.
- **Bayes factor**: model selection now has 4+ options: `Null`, `DLA`,
  `BAL`, `DLA+BAL`, `DLA+DLA+BAL`, etc.

**Pros:**
- Correct in principle: BAL absorption modeled as a physical
  outflow, not as "average BAL continuum".
- Reduces DLA false positives at BAL troughs *by construction*: the
  BAL model gets credit for explaining the trough.
- Catalog of BAL parameters as a side product (BAL physics).

**Cons:**
- Requires new sample grid generation (analog of QMC samples for DLA).
- Requires inference-engine changes:
  - new likelihood in `bayesian_model_selection.py`
  - new GP class (`BalGP`?) analogous to `DlaGP`
  - new params (`Parameters.num_bal_samples`, etc.)
- 2-3 weeks of design + implementation. Not 1-PR scope.

## Recommendation

**Ship Option A in the next PR.** Reasons:
1. Trainset is almost ready (`loa_bal_only_52338154/` exists; preload
   will produce trainset.h5 once it runs). Just need the train + glue.
2. Existing infrastructure is sufficient.
3. The inference-time model selection between standard / BAL GP is a
   well-defined Bayesian operation; the catalog gets a `is_bal`
   flag for free.
4. DLA-false-positive reduction at BAL troughs is *not* the headline
   science goal of Option A, but the BAL catalog itself is a useful
   product for cross-correlation with absorber catalogs (e.g., DLA
   environment near BAL host quasars).

**Plan Option B as a separate, longer-running project.** Reasons:
1. Requires new sample grid + inference-engine changes — not a
   straightforward extension.
2. Should wait until we have empirical evidence from Option A about
   whether DLA-false-positives at BAL troughs are actually a
   significant population statistic (vs a qualitative concern).
3. If DLA-false-positives at BAL troughs ARE significant, then Option
   B is well-motivated and we can scope it from the empirical
   measurement.

## Open questions for the user

1. **Confirm Option A scope** — agree the next PR ships the BAL-only
   μ + inference-side model-selection, with no new absorber model?

2. **BAL trainset filter strictness** — `BI_CIV > 0` is the canonical
   "weak" cut. Stricter cuts: `BI_CIV > 500 km/s` (BAL by Trump+2006);
   `AI_CIV > 0` (absorption index, picks up mini-BALs). Which do we
   use? Default `--bal-min 0` in `preload_loa_real.py` matches the
   weak cut. The `loa_bal_only_52338154/` job (if it ran) used
   defaults. Worth checking once trainset.h5 is on disk.

3. **Inference-side glue** — for Option A, do we want
   per-spectrum *hard* model selection (pick the higher-evidence GP)
   or *soft* model averaging (weight by evidence ratio)? Hard is
   simpler; soft is more Bayesian. Recommend hard for first
   implementation; revisit if BAL-fraction near-edge cases are noisy.

4. **DLA detection on BAL-flagged QSOs** — once a QSO is BAL-flagged,
   do we (a) skip DLA detection entirely (current production behavior
   ~equivalent if we trust BI_CIV), (b) run DLA detection on the
   BAL-GP continuum (Option A's natural choice), or (c) run DLA
   detection on the standard-GP continuum but flag the result as
   BAL-contaminated? Recommend (b) since it uses the better-fitting
   continuum.

## What I will NOT do without explicit user sign-off

- Launch BAL training. Trainset must exist + design choices above
  must be confirmed first.
- Implement Option B. Out of scope for the next PR.
- Modify how the production catalog handles BAL flags — the
  `--balmask` CLI flag exists but is not used in production; that's
  an orthogonal decision.

## Once approved

1. Confirm `loa_bal_only_*/trainset.h5` is on disk (NERSC) — if not,
   re-trigger the preload.
2. Train the BAL GP (~5 h NERSC, same `train_only_nersc.sh` recipe
   as the LOA retrains).
3. Add ~30 lines to `run_bayes_select.py` for the glue (build two
   holders, compare `log_evidence_no_dlas`, pick the winner).
4. Validate on a small set (n=50 known BAL QSOs from the QSO catalog
   + n=50 random non-BAL): confirm BAL GP wins for known BALs,
   standard GP wins for non-BALs.
5. Run the n=50 validation through full DLA inference under both
   GPs — measure how many DLA false positives at C IV / Si IV
   troughs got removed.
6. If improvement is substantial, run the 50k LoaArchive comparison
   campaign through this glue (BAL-aware vs original) and publish
   the comparison.
