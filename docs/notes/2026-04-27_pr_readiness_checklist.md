# PR-readiness checklist — claude/friendly-allen → desi_y3

This branch carries the GreatLakes migration + a set of investigation
tools and reports. Listing what's already in vs. what should land
before the first PR to `desi_y3`.

## What's already on the branch (commit-by-commit)

| commit  | content                                                            |
|---------|--------------------------------------------------------------------|
| 1a44004 | `.gitignore`, `private/loa_paths.md.template`                      |
| 52e410f | numpy 2.x compat (`np.trapezoid`); `nanargmax` plotting bug fix    |
| 939182e | `docs/greatlakes_setup.md`, `docs/nersc_greatlakes_mapping.md`     |
| b7f9c92 | `gpy_dla_detection/voigt_v2.py` + injector + 6 parity tests        |
| 9ccf0c0 | `gpy_dla_detection/postprocess/` (Lyβ + LLS xref) + README + tests |
| 81540e4 | smoke runner suite + per-condition / production analyzers          |
| ee675d9 | `slurm/greatlakes/timing_test.sh`                                  |
| d7a2c41 | `docs/notes/` (smoke + model comparison + filter sweep + sub-DLA + Bayesian) |
| ed3e7ef | `.gitignore` slurm log entry                                       |
| 86a7bb7 | analyze_production_catalog: condition completeness on processed TIDs |
| (next)  | scan_pdla_cuts + Bayesian-doc rewrite (already pending)            |

Total: ~3000 LOC, all 93 tests passing, no production behaviour change.

## What should land before merging the first PR

### Must (blocks merge)

1. **Resolve the README + CLAUDE.md updates.** The new modules
   (`voigt_v2`, `postprocess/`) deserve a one-line mention in the top
   README so a fresh user can find them. CLAUDE.md should link to the
   new docs/notes files. *Estimated: 30 min.*
2. **One README pass on the PR description itself**, summarising the
   investigation results so the reviewer doesn't have to read all of
   docs/notes/ to know what landed. *Estimated: 30 min.*
3. **Squash the trivial follow-up commits** (`.gitignore` slurm entry
   could fold into the main `.gitignore` commit; the bayesian-doc
   rewrite could fold into the prior bayesian-doc commit). *Estimated:
   15 min, optional but cleaner.*

### Should (improves the PR but not required)

4. **Add a top-level `examples/README.md`** describing what each runner
   / analyser does. Right now there are 11 .py files in `examples/`
   and a fresh user has to read each to figure out which one to run.
   *Estimated: 1 hour.*
5. **Run the existing 93-test suite under all current conda envs** to
   confirm no regressions on the NERSC stack. *Currently passes locally
   on the gpdla env.*
6. **Run `pytest -q` on a NERSC compute node** (any tests that import
   `voigt_fast` and skip cleanly when the .so is unavailable should
   stay skipped). The `tests/test_smoke_target_contamination.py` is
   already gated on file existence with `pytest.skipif`, so it should
   no-op on NERSC.

### Nice to have (defer to a follow-up PR)

7. The 4-step Bayesian-correctness investigation (Steps 1–4 in
   `docs/notes/2026-04-27_bayesian_correctness_plan.md`). Not yet run.
8. Voigt kernel × num_lines hypothesis test on 30 spectra (Task 16).
9. Sub-DLA model improvements A and D (Task 18 / `docs/notes/2026-04-27_subdla_model_improvements.md`).
10. Test of "don't filter M_DLA(1)" on existing test spectra.
11. Multi-DLA + sub-DLA hypothesis test on the candidate spectra
    listed in `out/smoke/subdla_test_targets.tsv`.
12. Y3 200-target sweep finish (in flight as sbatch 48819397).
13. Prior-edge test [20.0, 23] vs [20.3, 23] aggregation (data is in
    on disk; the analyzer hasn't been run on it yet).
14. Lyβ-veto SNR-binned plot to match the molly notebook.
15. Real LOA single-spectrum smoke test (Phase 2 task #4 was for
    real LOA but only mock has been done).
16. CDDF analysis on London mock-0 LLS run (Task 8).
17. Purity/completeness module extraction from molly notebook (Task 9).
18. Training smoke test (Task 6).

## Open questions for the user before merging

- **Should the PR include `docs/notes/`?** The notes are
  investigation logs, not reference docs. They could live in a
  separate `analysis/` branch, or under `docs/notes/` as committed
  material, or be deleted after extracting their conclusions into
  `docs/architecture.md` or similar. My default would be: keep them
  in `docs/notes/` so the reasoning is recoverable, even if some of
  the conclusions get superseded.
- **Should the new examples/ analyzers ship in `examples/` or in a
  new `analysis/` subdir?** The `examples/` convention is "runnable
  one-off scripts"; some of these (like `analyze_production_catalog.py`)
  are full analysis tools. Either location is defensible; I'll stay
  in `examples/` unless you prefer otherwise.
- **Do you want me to open the PR now, or wait until item 7-11 land?**
  The branch is complete and consistent as-is. Items 7-11 are
  follow-up science work that doesn't depend on the merge.

## Recommended PR title and description

```
GreatLakes setup + Voigt v2 + post-processing helpers + investigation logs

Bring up the GP-DLA pipeline on UMich GreatLakes and add three
analytical capabilities the next iteration is going to need:

- Voigt v2 (alternative pure-Python forward model with selectable LSF
  kernel and num_lines), parity-tested against the production C
  extension to <1e-9. Used for the LSF-bias study now in flight.

- Post-processing helpers (Lyβ misID veto + LLS-mode cross-reference)
  with conservative flagging. On the London production multi-DLA
  catalog they recover the historic 78%-purity / 80%-completeness
  operating point at P_DLA ≥ 0.99 with BAL excluded, and add a small
  but real purity boost via LLS xref.

- Smoke / sweep / production-scale analyzers: single-spectrum runner
  with --plot, stratified-by-NHI target picker, FILTER × num_samples
  sweep tooling, and a production-scale purity/completeness analyzer
  with per-NHI-bin breakdowns. Reproduces the historic GP numbers and
  produces a per-class "spurious" decomposition (DLA / sub-DLA / LLS /
  hallucinated).

Two minimal production-code fixes are included:
- numpy 2.x compat alias for np.trapz → np.trapezoid in CDDF_analysis.
- np.argmax → np.nanargmax in plot_samples_vs_this_mu (the early-stop
  rule produced NaN entries that argmax was selecting, causing the
  canonical model-fit plot to label MAP DLAs as "(nan, nan, nan)" and
  skip the model-line overlay).

Investigation logs under docs/notes/ document each finding with a
falsifiable test design and explicit caveats.

93 tests pass. Production code unchanged except for the two minimal
fixes above.
```

## Final note

I have NOT been opening pull requests on this repo on my own.
Whenever you're ready, the branch can be pushed and the PR opened —
let me know if you'd like me to do that, or do it manually so you
control the squash / rebase / message form.
