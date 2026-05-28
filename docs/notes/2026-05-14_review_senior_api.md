# Senior reviewer pass: PR #6

> Reviewer angle: API design, abstractions, long-term maintainability.
> Three prior reviews (`2026-05-13_code_review_dataset_math.md`,
> `2026-05-13_code_review_pr_diff.md`, `2026-05-14_full_pr_review.md`)
> covered the math, the diff, and the holistic picture; all three said
> SHIP / FIX-FIRST. This pass focuses on what those reviews did not:
> what does this code look like to someone who picks it up in six
> months and has to extend it? Did the PR draw the right module
> boundaries, or did it punt and leave a mess for next time?

## Top-line verdict

**SHIP, but flag four maintainability debts for follow-up.**

The science is correct and the load-bearing surfaces (`load_preprocessed_h5`,
`spectrum_loss_batch`, `LoaArchive`, `loa_archive_to_trainset`) are well-
shaped functions with clear contracts. The two trainer scripts (`tests/
phase2_train_*.py`) are competent operational code but live in the wrong
directory and share too much logic by copy-paste rather than refactor;
the next person to add a third trainer variant will pay for that.

The debts are: (1) trainer scripts in `tests/` and re-importing private
helpers across files, (2) a `_train` function that has grown to ~190
lines and mixes five concerns, (3) the `_RUNTIME` dict + `global` mutation
pattern for CLI plumbing, (4) a public `LoaArchive` attribute (`_h`) that
external code already reaches into. None block the merge, but they should
be on the list before any fourth variant lands.

## API surface concerns

**`load_preprocessed_h5`** (`gpy_dla_detection/training/dataset.py:224-414`)
is the strongest new API surface in the PR. Keyword-only after the
positional path, sensible defaults, clear semantic name. One issue:
**no preprocessing-config dataclass**. The function takes 14 keyword
arguments, of which 9 control preprocessing toggles + bands + priors
(`max_noise_variance`, `apply_mask`, `apply_normalize`, `apply_de_forest`,
`apply_center`, `norm_min_lambda`, `norm_max_lambda`, `de_forest_tau_0`,
`de_forest_beta`, `de_forest_num_lines`). The trainer at
`tests/phase2_train_desi.py:640-652` then has to spell out 11 of those
kwargs every time. A `PreprocessConfig` dataclass would: (a) make
`_RUNTIME` mutation (`tests/phase2_train_desi.py:624-625`) unnecessary,
since the same config flows through `_save_h5` for serialization; (b)
make a future "load with the same config as this training run" trivial
by deserialising the manifest in the .h5; (c) shrink the kwarg surface
from 14 to 4 (path, config, max_spectra, catalog). Low priority but
this is the kind of thing that compounds.

**`spectrum_loss_batch`** (`gpy_dla_detection/training_v3/objective_vectorized.py:36-182`)
takes 13 positional arguments. Several are tightly coupled:
`num_forest_lines`, `all_transition_wavelengths`, `all_oscillator_strengths`
always travel together (they're the Lyman-series spec); `M, omega2, c_0,
tau_0, beta` always travel together (they're the GP parameters);
`y, lya_1pz, noise_variance, valid_mask` always travel together (they're
one batch of data). Keeping all 13 positional is faithful to v1's
`spectrum_loss` signature — which is the right call for parity testing
(`test_v3_objective_vectorized_parity.py`) — but at the trainer call
site (`tests/phase2_train_desi.py:240-245`, 6 lines of arg shuffling) it
hurts readability. A future cleanup could group these into three
named tuples without touching the math.

**`LoaArchive`** (`gpy_dla_detection/loa_archive.py:296-384`) has one
visible API hole: `LoaArchive._h` is consumed by an external caller at
`preload_spectra/preload_from_loa_archive.py:225-227` (`ar._h["flux"][sorted_idx]`).
The author needed sorted-index bulk read for memory-bounded chunking and
the public `get_spectrum(targetid)` doesn't support that, so they
reached past the `_h` underscore. Two options: (a) add
`get_spectra_by_indices(indices)` to the public surface, or (b) drop
the underscore on `h` and document the public h5py handle. Right now
this is a private-API violation hiding in production code.

**`LoaArchive`** also writes `schema_version=1` (`loa_archive.py:182`)
but the reader never checks it (`loa_archive.py:317-327`). The first
breaking change to the schema will silently mis-read v1 files until
someone runs into a downstream error. A 2-line check in `open()` would
fix this and is the single cheapest forward-compatibility improvement
in the PR.

**`loa_archive_to_trainset`** (`preload_spectra/preload_from_loa_archive.py:110-292`)
takes `verbose: bool = True` mid-keyword-args. Logging-as-parameter is
fine for one-off scripts but anti-pattern for module-level functions —
a caller who wants structured logging has to pipe through `verbose=False`
and lose all the per-stage QSO counts. Acceptable for now; replace with
a `logging.Logger` in any cleanup pass.

**`tests/phase2_train_desi.py` CLI surface** has 17 args. They're
organised reasonably (data, model, optimizer, IO, walltime, priors),
all have help strings, all have sensible defaults. The one structural
issue: `--norm-min-lambda` / `--norm-max-lambda` defaults are 1425.0 /
1475.0 (line 605-608), but `tests/phase2_train_dr16.py` *hardcodes*
1425.0 / 1475.0 with no CLI override (`phase2_train_dr16.py:523-524`).
Inconsistent: DESI is CLI-driven, DR16 is hardcoded. The 2026-05-13_code_review_pr_diff.md
review correctly noted this is intentional (DR16 must match MATLAB
faithfully), but the asymmetry surfaces in the manifest schema (DESI .h5
has `preload_source`; DR16 .h5 does not — `phase2_train_dr16.py:529-549`)
and will trip up anyone writing a generic .h5 inspector.

**The new h5 schema** is the riskiest forward-compatibility surface in
the PR. Twenty new top-level keys, all flat 0-d datasets, no namespacing.
A future `lr_schedule` field, `prior_v2` field, or `de_forest_tau_eb` field
clashes with the trainer's flat layout. A single `training_manifest` group
(or even just a prefix like `train.lr`, `train.k`, `prior.tau_0_mu`)
would have made the schema self-describing and reservation-safe. The
inference loader at `null_gp.py:462-476` reads exactly 9 keys, so it's
forward-compatible by accident — but the schema is not designed for
forward-compatibility, just accidentally tolerant of it.

## Function complexity

**`tests/phase2_train_desi.py::_train`** (lines 104-308, ~205 lines).
This is the heaviest function in the PR. It does six things:

1. **Tensor materialization + pin-memory** (lines 117-156) — building
   `M, log_omega, log_c_0, log_tau_0, log_beta` Parameters, pinning
   CPU data, in-place NaN sanitization.
2. **Checkpoint resume** (lines 158-173).
3. **Adam optimizer + signal handler setup** (lines 175-208).
4. **The actual training loop** (lines 215-295), itself containing:
   - chunked CPU→GPU per-step transfer (235-239)
   - vectorized loss call (240-251)
   - prior gradient injection (256-262)
   - manual `.grad` assignment (264-269)
   - history logging + console print (275-283)
   - periodic / walltime / signal checkpoint logic (285-295)
5. **Final checkpoint** (line 297).
6. **Return-dict construction** (lines 299-308).

Steps 1, 2, 3, 5, 6 are mechanical; step 4 is the science. The two are
intertwined and there's no test for `_train` directly — only via SLURM
smoke runs. A `Trainer` class with `setup_parameters()`,
`load_checkpoint(path)`, `step(chunk)`, `save_checkpoint(tag)` methods
would split testable units cleanly. This is what `trainer_v2.py` aimed at
and got wrong in a different way; the new code is the right algorithm
but in the wrong shape.

The **prior-gradient injection at lines 256-262** is the most fragile
fragment in the function: it adds gradients in three places
(`dlog_tau_0_acc`, `dlog_beta_acc`, conditionally `dlog_c_0_acc`) using
chain-rule derivations that are mathematically correct but not directly
unit-tested. If you ever add a fourth prior, the analogous derivation
mistake will not be caught by any existing test. Suggested
follow-up: extract `_apply_priors(grads, params)` and unit-test it on
known closed-form derivatives.

**`tests/phase2_train_dr16.py::_train`** (lines 220-422) is even longer
because it carries the `vectorized=True/False` branch plus all the same
concerns. It also has a per-spectrum reference path (lines 348-371)
inside the same function as the vectorized path (lines 329-347). The
two paths are tested for numeric parity (`test_v3_train_step_parity.py`)
but live in one function that's already 200 lines. A clean refactor
would push them into separate functions `_step_vectorized` /
`_step_per_spectrum` both returning the same six accumulators, and the
outer loop just picks which to call.

**`spectrum_loss_batch`** at ~150 lines is well-structured. The
five sections (sanitize, forward model, Woodbury, gradients, scalar
gradients) are flagged with comment dividers (lines 93, 99, 121, 148,
165). The hand-coded gradient blocks are dense but each is one logical
step. This function is doing exactly what its docstring promises and
nothing more. Best-shaped new function in the PR.

**`_normalize_by_rest_median`** at 90 lines is on the upper edge of
readability — the `bad` mask uses three predicates ORed together
(`dataset.py:177`) followed by four diagnostic counts (`dataset.py:180-187`),
all wrapped in conditional print. The four-bucket diagnostic is
operationally useful at scale (you want to know if 0.094% of your
preload is being rejected for the marginal reason vs. 0.51% for the
negative reason), but the logic-vs-print ratio in lines 163-194 is
unfortunate. Extracting a tiny `_diagnose_bad_medians(medians)` would
clean this up.

## Naming consistency

**"preload" vs "trainset" vs "training set"** — *inconsistent*.
- `dataset.py:13` "older preload"
- `dataset.py:15` "current production preload"
- `dataset.py:54` `TrainingSet` (class name)
- `dataset.py:245` "preprocessed GP training set HDF5 file"
- `preload_from_loa_archive.py:1` "Adapter: LoaArchive → v2 trainset.h5"
- `preload_from_loa_archive.py:140` "v2 preload trainset.h5 files"
- `phase2_train_desi.py:582` `--preload` argparse name
- `phase2_train_desi.py:5` "v2 preprocessed `trainset.h5`"

Three names for the same artifact. A reader is left to infer that
"preload" = "trainset" = "training set HDF5" = "the file produced by
`preload_*` scripts and consumed by `load_preprocessed_h5`". Lock one
name down — `trainset` is the closest to descriptive (it's what the file
is, not how it's produced) and is what the CLI flag in
`preload_from_loa_archive.py:329` already uses (`--out trainset.h5`).
Then either: keep `--preload` (operationally entrenched, refers to the
*source*, semantically "the preloaded file") OR rename to `--trainset`
(semantically "the training-set artifact"). Either is fine; mixing both
is the problem.

**"normalize_then_mask_order"** (`phase2_train_desi.py:560`,
`phase2_train_dr16.py:544`) — both write `np.int64(1)`. The convention
is **1 = normalize-then-mask (new, MATLAB-faithful), 0 = mask-then-normalize
(old, pre-2026-05-13)**. The name encodes the new direction, which is
the right call. The comment at `phase2_train_desi.py:556-559` documents
this. Minor nit: the name reads slightly ambiguous on its own ("does
0 mean 'no ordering' or the reverse ordering?"). A renamed
`preprocess_order` field with string values (`b"normalize_then_mask"` /
`b"mask_then_normalize"`) would be self-describing without a comment.

**`log_c_0` vs `c_0`** — handled cleanly. The convention is consistently
"log_*" for trained parameters in log-space (which Adam updates) and
"*" for the exponentiated value (which the loss uses). Trainer returns
both in the result dict (`phase2_train_desi.py:302-307`). The .h5 stores
only `log_*` (the canonical form). Good.

**"Step C" vs "Phase 2 desi" vs "PR6_StepC"** — *inconsistent*.
- `training_v3/README.md:40` "Step C — production retrain"
- `phase2_train_desi.py:1` "DESI Phase 2 trainer"
- `phase2_train_desi.py:562` `training_release = b"PR6_StepC"`
- `phase2_train_dr16.py:546` `training_release = b"PR6_StepA_DR16"`
- `docs/production_models.md:43` "PR #6 Step C"

The terminology is: "Phase 2" = the training architecture (post-PR-#5);
"Step C" = the production stage of the Phase-2 plan (`training_v3/README.md`
Plan items 1/2/3 = Step A/B/C). The trainer file is named after Phase 2,
the model card and .h5 manifest are named after Step C. This is
internally coherent but a new reader has to triangulate three names to
understand they refer to overlapping concepts. Documentation in
`training_v3/README.md` makes it clear; the trainer-file naming does
not. Acceptable as is — terminology is settled and the next PR can
either keep "PR6_StepC" or shift to a more durable naming scheme.

## Error handling + edge cases

**All-NaN row** is well-handled. The chain is:
- `_normalize_by_rest_median` (`dataset.py:189-193`) zeroes the row to NaN
- `_mask_high_noise_pixels` (`dataset.py:65-74`) leaves NaN as NaN (since
  `nan > 9.0` returns False)
- `_pca_init` (`phase2_train_dr16.py:204-214`) catches all-NaN rows and
  zeroes them
- Trainer `valid_masks = np.isfinite(centered) & np.isfinite(nv) & (nv > 0)`
  (`phase2_train_desi.py:667`) drops those pixels from the loss
- `spectrum_loss_batch` masks at `valid_mask` (line 96-97, 118-119)

This path is the most carefully thought-through error handling in the PR.
Verified by the math review (`2026-05-13_code_review_dataset_math.md:63-73`).

**All-zero row** — handled identically to all-NaN (median=0 →
`(medians <= 0)` → bad → row NaN'd, `dataset.py:177`).

**Single-row input** — `load_preprocessed_h5` does not explicitly handle
n=1. Will pass through (`fluxes_raw[mask]` with one True row works), but
`_pca_init` with k=30 components and n=1 spectrum will fail at sklearn.
Currently undocumented. Low priority; production never sees this case.

**Optimization divergence (Adam loss → NaN)** is *not* handled. There's
no NaN-check on `total.cpu()` (`phase2_train_desi.py:275`) and no
mechanism to abort cleanly when the optimizer blows up. The c0prior
collapse model (`docs/production_models.md:68`) is a real-world example
where Adam endpoint produced NaN-MAP-NHI; the training itself completed
"successfully" because nothing checked. A `if not torch.isfinite(total):
raise RuntimeError(...)` at line 275 would have caught it. Suggested
follow-up.

**Resume from corrupted checkpoint** — `torch.load(rp, weights_only=False)`
at `phase2_train_desi.py:162` will raise on a corrupted file but the
caller doesn't wrap; the SLURM script will see an unhandled exception.
This is the right behavior (fail loud, don't proceed silently with a
half-loaded state). Not a defect; flagging for awareness.

**Network filesystem (`/nfs/turbo`) read failures** — `h5py.File(...,
"r")` will raise an `OSError` on read failure; the caller `_train` does
not catch. Same fail-loud pattern as above. Production-correct.

**Mismatched array shapes** — `_train` assumes `centered.shape ==
nv.shape == lya_1pzs.shape == valid_masks.shape`. No assertion. If
the caller passes inconsistent shapes, torch will raise at the first
broadcast op with a confusing error. A 4-line shape check at the top
of `_train` would convert that into a clear message at low cost.

## Logging quality

**Log discoverability is mostly good**. Tags are consistent (`[dataset]`,
`[pca]`, `[config]`, `[data]`, `[loa_adapter]`, `[cache]`, `[checkpoint]`,
`[resume]`, `[signal]`, `[walltime]`, `[saved]`) and self-explanatory.

The standout self-describing log line is `dataset.py:184-187` (the
four-bucket bad-median diagnostic), which gives an operator everything
they need to verify the preprocessing rejection threshold is working
correctly.

**The standout misleading log line** is `dataset.py:362-372`. After
2026-05-12 the trainer started running with `[1425, 1475]` MATLAB band
but the print statement said "(Garnett+2017 convention)". The PR fixed
this at commit 660ee34 (now templated by the band value) — but the same
templating bug existed in `phase2_train_desi.py:362` for the README.
**Both fixes are in.** Note: the third reviewer flagged this; it's now
addressed. Verified at `dataset.py:364-372`.

**One missing log line**: `_pca_init` at `phase2_train_dr16.py:189-217`
prints nothing about which rows were zeroed. At scale you want
"`[pca] 23 of 89408 rows have all-NaN, zeroed for PCA input`" to detect
preprocessing drift. The trainer's `valid_pix_frac` log at
`phase2_train_desi.py:668-670` partly covers this but it's at pixel
granularity, not row granularity. Low priority.

**`tests/phase2_train_dr16.py` lacks the `[config]` line** the DESI
trainer has at line 620. A user trying to grep "what device / preload /
priors did this run use" can find it in `phase2_train_desi.py` but
not the DR16 trainer. Cosmetic.

## Configuration discoverability

**Module-level constants are well-discoverable** in both trainers.
`phase2_train_desi.py:66-99` and `phase2_train_dr16.py:63-74` group them
at the top of the file, just below imports, with comments. The
inconsistency is the *prior σ* values are flagged via Turner+2024 ref
in DESI (line 81-84) and via "BOSS DR12Q" in DR16 (line 73-74); both
are correct but the asymmetric documentation is on the order of "DESI
file has a 6-line comment explaining the audit, DR16 file has none."

**The `LOG_C_0_PRIOR_SIGMA = None  # set via --log-c-0-prior-sigma at
submit time` global** (`phase2_train_desi.py:99`) is then mutated by
`main()` at line 616-617 (`global LOG_C_0_PRIOR_SIGMA; LOG_C_0_PRIOR_SIGMA
= args.log_c_0_prior_sigma`). This is the most surprising pattern in
the PR. The `_train` function reads the global at line 261. Three
problems:

1. Two ways to set it (module-level constant + CLI flag), creating an
   "which one wins" mystery resolvable only by reading `main()`.
2. The global makes `_train` non-thread-safe and non-reusable from
   tests.
3. The same pattern is used for `_RUNTIME["norm_min_lambda"]` etc.
   (`phase2_train_desi.py:622-625`) without a `global` keyword (`_RUNTIME`
   is a dict so mutation doesn't need `global` to work, but it's
   doing the same thing).

A clean fix is `_train(..., log_c_0_prior_sigma=None, ...)` as an
explicit kwarg. The current pattern works but signals to a future
reader "be careful about modifying _train".

**Module-level `MAX_NOISE_VARIANCE` is hidden**. The value 9.0 appears
**three times hardcoded** in `phase2_train_desi.py` (lines 413, 510, 644)
and once as the default in `dataset.py:232`. Should be a module-level
`MAX_NOISE_VARIANCE = 9.0` constant referenced everywhere — same way
`NUM_FOREST_LINES = 31` is handled at line 67. DR16 trainer does this
right (`MAX_NV = 9.0` at line 68).

**CLI args** are discoverable and well-documented. The 17 args in
`phase2_train_desi.py:580-613` are grouped logically. The only
discoverability gap is that `--log-c-0-prior-sigma` (`line 609`) does
not appear in the docstring usage examples (`line 25-37`), so a user
running `--help` sees it but a user reading the module docstring does not.

**`.h5 manifest fields`** (`phase2_train_desi.py:537-567`) — 25 fields
in a flat namespace, no grouping. As noted in "API surface concerns,"
this will pay technical-debt interest in the next manifest extension.

## Testability

| Surface | Testable? | Tested? |
|---|---|---|
| `_normalize_by_rest_median` | yes | yes (`test_normalize_by_rest_median.py`, 20 tests including the regression guard at line 126) |
| `_mask_high_noise_pixels` | yes | indirectly (via end-to-end smoke at `test_load_preprocessed_h5_normalize_path_smoke`) — no targeted unit test |
| `_de_forest_batch` | yes | yes (`test_normalize_by_rest_median.py:194-226`) |
| `_center_fluxes_inverse_variance` | yes | yes (`test_normalize_by_rest_median.py:232-255`) |
| `load_preprocessed_h5` | yes (smoke) | yes (`test_load_preprocessed_h5_normalize_path_smoke`) |
| `spectrum_loss_batch` | yes | yes (Jacobian + parity + train-step parity, 3 test files) |
| `LoaArchive` r/w roundtrip | yes | yes (`test_loa_archive.py`) |
| `loa_archive_to_trainset` | yes | yes (`test_preload_from_loa_archive.py`, 9 tests) |
| `_pca_init` | yes | indirectly (via parity tests) — no targeted unit test for `random_state=0` determinism |
| `_train` in either trainer | **no** | **no** — only via SLURM smoke runs |
| `_save_h5` / `_save_readme` | yes | **no** — no test asserts the manifest fields are present or the README templating works |
| `_apply_priors` (would-be helper) | n/a | n/a (does not exist; prior code is inlined in `_train`) |

The big testability gaps are:

1. **`_train` is monolithic** and only tested by running the whole
   trainer end-to-end (SLURM jobs). A refactor splitting `setup`,
   `step`, `save` would make each unit-testable. The math content of
   one Adam iteration is *already* covered by `test_v3_train_step_parity.py`
   — what's not covered is the orchestration (resume, checkpoint cadence,
   walltime exit, signal handling). If any of those break, only an
   ops failure on a long SLURM job will surface it.

2. **No test for `_save_h5` manifest output**. The 25 fields written at
   `phase2_train_desi.py:537-567` could be wrong in ways the inference
   loader won't catch (since it ignores unknown keys). A test that writes
   a dummy result + args, reads the .h5 back, and asserts the 25
   expected keys are present with the right dtypes would be 30 lines.

3. **No test for `_save_readme`**. The third reviewer noted this gap
   (`2026-05-14_full_pr_review.md:122-130`); fix is a 15-line test.

4. **`LOG_C_0_PRIOR_SIGMA` global + CLI override** has no test that
   confirms the prior gradient fires only when `--log-c-0-prior-sigma`
   is set. A regression where someone "fixes" the if-check at line 261
   to `if LOG_C_0_PRIOR_SIGMA:` (Falsy on 0.0!) would silently disable
   the prior for `σ=0` configs. Easy to test.

## Specific findings (severity: critical / high / medium / low / nit)

**Critical**: none.

**High**:

1. **`tests/phase2_train_desi.py:147` in-place mutation of caller's
   array** — `centered[~valid_masks] = 0.0` mutates the numpy array
   `_train` was passed (`main()` line 683 hands in `centered` from
   `ts.fluxes.numpy()`, which is the *same* memory as the
   `TrainingSet.fluxes` tensor's storage). The trainer trusts that `main`
   won't use `centered` again post-call (it doesn't), but if any caller
   ever does, they'll silently get the masked-out version. Either
   `centered = np.where(valid_masks, centered, 0.0)` (a copy) or
   document the contract loudly. The comment at line 141 hints at this
   but doesn't say "warning: in-place".

**Medium**:

2. **`LoaArchive._h` reached by external code**
   (`preload_from_loa_archive.py:225-227`). Private-attribute consumption.
   Either add a public `get_bulk(indices)` to `LoaArchive` or drop the
   underscore.

3. **`LoaArchive.write_archive` writes `schema_version=1` but the reader
   never validates it** (`loa_archive.py:317-327`). Add a check in
   `open()` to fail on unknown versions.

4. **`chunks_2d = (min(chunk_qsos, 1), n_pix)`** at `loa_archive.py:179`
   forces single-row HDF5 chunks regardless of the `chunk_qsos` kwarg.
   Looks like a typo — probably intended `(chunk_qsos, n_pix)` or
   `(min(chunk_qsos, max_rows_per_chunk), n_pix)`. With `chunk_qsos=256`
   default, the buffer batches 256 rows in Python (line 280) but writes
   1-row chunks to HDF5 — read perf for fancy-index lookup is fine, but
   the chunking is wasteful. Worth verifying intent.

5. **`LOG_C_0_PRIOR_SIGMA` global + `_RUNTIME` dict + `global` keyword
   pattern** (`phase2_train_desi.py:99, 615-625, 261`) is the most
   surprising configuration pattern in the PR. Replace with an explicit
   kwarg in `_train` and a config object passed through `_save_h5`.

6. **`_train` is ~205 lines doing six things**
   (`phase2_train_desi.py:104-308`). Refactor into a `Trainer` class
   with `setup_parameters / load_checkpoint / step / save_checkpoint`
   methods to make each unit-testable.

7. **Hardcoded `9.0` for max_noise_variance** appears three times in
   `phase2_train_desi.py` (lines 413, 510, 644) and as a default in
   `dataset.py:232`. Should be a module-level constant
   `MAX_NOISE_VARIANCE = 9.0` referenced everywhere, mirroring how
   `phase2_train_dr16.py:68 MAX_NV = 9.0` is handled.

**Low**:

8. **Trainers live in `tests/`**. `tests/phase2_train_desi.py` and
   `tests/phase2_train_dr16.py` are production scripts, not tests.
   Pytest correctly skips them (no `test_` prefix), but the convention is
   confusing. Move to `examples/` or `scripts/`. Cross-imports
   (`from tests.phase2_train_dr16 import _pca_init` at
   `phase2_train_desi.py:61` and 4 example files) hardcode the wrong
   location.

9. **`_pca_init` is private and re-imported**
   (`phase2_train_desi.py:61`, `examples/probe_outlier_tail_corr.py:52`,
   3 more example files). It's the right function and the right
   behavior, but it should be `gpy_dla_detection.training.pca_init`
   (a public module-level symbol) rather than imported through the
   trainer.

10. **`tests/phase2_train_desi.py:89-90` defensive double-assignment**
    `INITIAL_BETA = TAU_0_PRIOR_MU * 0 + 3.62  # avoid stale-cache typo`
    followed by `INITIAL_BETA = 3.62`. The comment confirms the author
    knew the first line was a workaround for some prior bug; the second
    line makes it dead code. Delete line 89.

11. **`vectorized=1`** as the default kwarg in `_save_h5`
    (`phase2_train_desi.py:455`) is int-typed because of the .h5 schema;
    `_save_h5` is called with the int directly, but the call site at
    line 695 doesn't pass it. So the manifest field `vectorized` is
    always 1 for the DESI trainer (no per-spectrum path exists there).
    Either drop the kwarg or make it explicit in the contract that
    DESI trainer is vectorized-only.

12. **Naming: "preload" vs "trainset"** inconsistency
    (4 names: preload, preprocessed h5, trainset, training set HDF5)
    — pick one.

13. **`loa_archive_to_trainset` takes `verbose: bool`** — replace with
    `logging.Logger` for caller flexibility.

14. **`spectrum_loss_batch` takes 13 positional args**. Group into
    `(BatchData, GPParams, LymanSeries)` named tuples for readability;
    keep current signature as a thin shim if parity tests need it.

**Nit**:

15. The four-bucket diagnostic in `_normalize_by_rest_median`
    (`dataset.py:180-187`) is operationally great but the logic-vs-print
    ratio in 30 lines is high. Extract `_diagnose_bad_medians(medians)`.

16. `_save_readme` and `_save_h5` are file-local helpers
    (`phase2_train_desi.py:311, 455`) — fine for now; would migrate
    naturally if the trainer moves out of `tests/`.

17. The DR16 trainer lacks a `[config]` print at start of `main`
    (`phase2_train_dr16.py`). DESI trainer has one at line 620.

18. `loa_archive.py:175` reject `wavelength.ndim != 1 or n_pix < 2` is
    a useful sanity check; equivalent check is missing in
    `preload_from_loa_archive.py` for the rest grid (`rest_dlambda > 0`,
    `rest_min < rest_max`). Defensive but not critical.

## Recommendations for long-term maintainability

**Before any fourth trainer variant lands**, do the following:

1. **Move trainers out of `tests/`** to `scripts/` or `gpy_dla_detection/
   training/`. Make `_pca_init` and `_apply_priors` public module-level
   functions in `gpy_dla_detection.training.utils`.
2. **Extract a `Trainer` class** with `setup_parameters / load_checkpoint
   / step / save_checkpoint` methods. The two `_train` functions become
   thin wrappers that wire data → Trainer → save.
3. **Replace the global `LOG_C_0_PRIOR_SIGMA` + `_RUNTIME` dict pattern**
   with explicit kwargs.
4. **Group `_save_h5` manifest fields** under named subgroups
   (`training/`, `priors/`, `provenance/`, `data/`).

**Cheap follow-ups not on the critical path**:

5. Add `schema_version` check in `LoaArchive.open()` (2 lines).
6. Add `_save_readme` and `_save_h5` unit tests (~50 lines).
7. Add a NaN-on-loss runtime check in `_train` to fail loudly on Adam
   divergence (~3 lines).
8. Add `get_spectra_by_indices(indices)` to `LoaArchive` and drop the
   `_h` underscore consumption in `preload_from_loa_archive.py:225-227`.
9. Lock naming: pick one of "preload" / "trainset" / "training set"
   and propagate.

**Net**: this PR moves the trainer from a half-broken v2 path to a
working v3 path with the math correct, the tests in place, and the
production runs queued. The structure is operational, not yet idiomatic.
That's acceptable for a science-blocking PR; the maintainability debt
is recorded and the fixes are all in the small-to-medium range.
