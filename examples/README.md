# `examples/` — runners and analysers

Single-file CLI tools for running the GP-DLA pipeline on individual
spectra, picking targets from mock truth catalogs, sweeping
configurations, and measuring purity / completeness on the resulting
catalogs. Nothing here is imported by the production pipeline; these
are downstream / development utilities.

If you're new to the repo, the typical entry point sequence is:

1. Set up the environment (`docs/greatlakes_setup.md` for GreatLakes,
   `README.md` for general).
2. `smoke_one_spectrum.py` to verify your env reproduces a known DLA
   recovery.
3. `pick_smoke_targets.py` to build a stratified test set, then
   `run_smoke_batch.sh` to run it.
4. `analyze_production_catalog.py` to score a real catalog against
   mock truth.

## Single-spectrum

| Script                           | What it does                                                                |
|----------------------------------|-----------------------------------------------------------------------------|
| `demo_desi_spectrum.py`          | Original Ho+ demo: London eBOSS-format mock spectra + DR16Q model.          |
| `smoke_one_spectrum.py`          | Modern single-spectrum runner. Three model presets (`eboss` / `y3` / `london`), explicit FILTER and num_samples flags, optional `--plot` to wire into the project's `plot_samples_vs_this_mu` canonical plot. The basic "is the env working" smoke test. |
| `plot_smoke_result.py`           | Lightweight plotter: reads a smoke `.pkl` and produces a 2-panel plot (spectrum + (z, NHI) sample posterior contour). |
| `plot_smoke_v2.py`               | Unified plotter. Re-instantiates `DLAGPMAT` to compute the GP-mean continuum and `GP × Voigt(MAP)` model overlay; multi-DLA-aware (handles k≥2 selections); shows truth marker box on the spectrum panel and an unconvolved analytical Voigt at truth. The "what's the model actually doing" diagnostic. |
| `plot_visual_inspect.py`         | Older one-off visual inspector for hand-picked spectra (kept for reference). |
| `plot_mcmc.py`                   | One-off MCMC posterior diagnostic (kept for reference).                      |

## Target picking + batch running

| Script                             | What it does                                                                 |
|------------------------------------|------------------------------------------------------------------------------|
| `pick_smoke_targets.py`            | Picks N stratified-by-NHI test targets per mock (4 NHI bins × N/4 each), with all-truth-on-LOS columns for downstream Lyβ / LLS analysis. Outputs `targets.tsv` consumed by the batch runner. |
| `run_smoke_batch.sh`               | Bash batch runner. Args: `PRESET FILTER N_DLA N_SUBDLA TARGETS [DLA_MAT_OVERRIDE]`. Writes per-condition output dirs under `out/smoke/batch/`. |
| `extend_targets_with_all_truth.py` | Migration helper: take a legacy 9-column `targets.tsv` and add `all_truth_z` / `all_truth_nhi` columns from the mock truth catalog. |
| `finalize_smoke_batch.py`          | Walks a batch output dir, builds a clean `summary.tsv`, and (optionally with `--plot`) regenerates the `plot_smoke_v2.py` figure for every target. |
| `aggregate_sweep.py`               | Combines multiple per-condition `summary.tsv` files into a single Markdown comparison table (FILTER × N_DLA sweep). |

## Catalog analysis

| Script                              | What it does                                                                     |
|-------------------------------------|----------------------------------------------------------------------------------|
| `analyze_purity_completeness.py`    | Per-truth-DLA matching (greedy nearest-z), with Lyβ misidentification cross-check on each spurious MAP DLA. Operates on a single batch dir's `.pkl` files; small-scale. |
| `analyze_production_catalog.py`     | Production-scale: walks 500+ chunks of `dlacat-*.fits`, matches against truth, and applies Lyβ veto + LLS cross-reference. Reports both strict (1-to-1) and loose (any-truth-match) purity, completeness per NHI bin, before vs. after each post-processing step. Supports `--no-bal` and `--p-dla-cut`. |
| `scan_pdla_cuts.py`                 | Scans P_DLA ∈ {0.5, 0.9, 0.99, 0.999} on a production catalog with optional BAL exclusion. Use to find the operating point that matches a target purity / completeness budget. |

## Data files

| Path                              | Contents                                                                       |
|-----------------------------------|--------------------------------------------------------------------------------|
| `data/subdla_test_targets.tsv`    | 5 high-SNR 2LPT loa-124 sightlines with ≥1 truth DLA + ≥2 truth sub-DLA / LLS. Curated for testing the multi-DLA + sub-DLA improvement plan in `docs/notes/2026-04-27_subdla_model_improvements.md`. |

## Conventions

- All scripts that run inference or load DESI spectra need
  `LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH` set
  for the compiled `_voigt.so` (see `docs/greatlakes_setup.md`).
- All output paths default to `out/smoke/...` or `figures/smoke_v2/...`,
  which are gitignored.
- All target / batch IO uses TSV (tab-separated) so it's grep-friendly.
- All analysers emit a Markdown report to `--out` and also print to
  stdout, so the result is reproducible from the report alone.

## Adding a new runner

If you write a new script in this directory, please:
- Put a short description and the typical-use one-liner in this README.
- If it generates output files, default to `out/...` (already ignored).
- If it reads truth or production catalogs, parameterise the path
  rather than hard-coding `/nfs/turbo/...` — both NERSC and GreatLakes
  callers need to swap.
