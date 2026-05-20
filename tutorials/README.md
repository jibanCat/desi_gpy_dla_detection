# Tutorials

Hand-written walkthroughs for new users. Each notebook builds up a
concept from first principles — start at `00`, work upward.

| # | Notebook | What it covers |
|---|---|---|
| 00 | [`00_quasar_redshift_estimation.ipynb`](00_quasar_redshift_estimation.ipynb) | What a quasar spectrum is, what redshift is, and how to estimate it from emission-line positions and from a trained GP model. Uses one SDSS DR12 spectrum. |
| 01 | [`01_lyman_alpha_absorption_detection.ipynb`](01_lyman_alpha_absorption_detection.ipynb) | Lyman-α absorption physics (DLA / sub-DLA / LLS), the Voigt profile, and how Bayesian model selection chooses between "absorber" and "no absorber" hypotheses. End-to-end pipeline on one quasar spectrum. |

## Before running

Set up the environment first per [`../docs/tutorial_quickstart.md`](../docs/tutorial_quickstart.md) — the
notebooks assume `libcerf` is built and the Voigt C extension
(`gpy_dla_detection/_voigt.so`) is compiled.

**Tutorial 01, cell 1 has a hard-coded `base_dir`** pointing at a
macOS install path (`/Users/jibanmac/...`). If you're on Linux /
GreatLakes / NERSC, swap it for your local repo root, or replace
that cell with the cross-platform `os.chdir("..")` used by tutorial 00.

## Style conventions

For anyone adding new tutorials here:

- **Numeric prefix** (`02_…`) for ordering. Snake_case filenames.
- **Pedagogical pacing**: a short markdown cell explaining the
  concept, then a small code cell demonstrating it on concrete
  numbers (e.g. "Let's pick z = 2.5"). Avoid 200-line monolithic
  cells.
- **Heavy comments** inside code cells — assume the reader is new
  to the codebase.
- **Build up incrementally**: minimal example → "now let's
  automate it" → real workflow on production-style data.
- **Stay self-contained**: include any data-download cell at the
  top; don't depend on un-versioned files under `data/`.

## See also

- [`../docs/tutorial_quickstart.md`](../docs/tutorial_quickstart.md) — environment setup.
- [`../docs/tutorial_population_statistics.md`](../docs/tutorial_population_statistics.md) — population-statistics workflow.
- [`../docs/architecture.md`](../docs/architecture.md) — the code map these tutorials are demonstrating.
- [`../notebooks/`](../notebooks/) — intermediate analysis notebooks (less polished, often used to back paper-figure work).
