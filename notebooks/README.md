# Notebooks (analysis / intermediate)

This folder is the **working scratchpad** for the GP-DLA project —
intermediate analysis, paper-figure prep, sample generation, and
visualisation experiments. Polish level varies.

> **Looking for a tutorial?** Go to [`../tutorials/`](../tutorials/) — that's
> where the maintained, hand-written walkthroughs for new users live.

## What lives here

### CDDF / population statistics (Paper 1 source material)

- `CDDF_dNdX_all.ipynb` — main dN/dX calibration workflow (76 cells).
- `CDDF_fN_z.ipynb` — f(N, z) calibration.
- `CDDF_all.ipynb`, `CDDF plots.ipynb`, `CDDF_zsep.ipynb` — older
  variants and plotting passes.
- `CIV_CDDF.ipynb` — CIV absorber population statistics.
- `Paper_plots.ipynb` — clean skeleton wiring the extracted
  `CDDF_analysis/` modules; the target home for Paper 1's headline
  figures (still being populated; see [`../docs/paper_figures.md`](../docs/paper_figures.md)).

### Demos / visualisation

- `Demo DESI Spectra.ipynb` — load and visualise a DESI coadded spectrum.
- `Demo GP Training.ipynb` — illustrate the GP training pipeline.
- `Visualize Data.ipynb`, `Visualize Model.ipynb` — diagnostic plots.
- `No_BAL_no_DLAs.ipynb` — BAL-free / DLA-free subset analysis.

### Sample-grid generation (utility)

- `GenerateSamples.ipynb` — QMC sample generation for the DLA model.
- `GenerateSamples_subDLA.ipynb` — same, sub-DLA prior.
- `GenerateSamples_PW14.ipynb` — Prochaska-Worseck-2014 prior variant.

### Other

- `GP CIV using MCMC.ipynb` — CIV-feature MCMC exploration.

## archive/

[`archive/`](archive/) holds notebooks that have been **retired**
from active use but kept for git-history. Don't run them; don't edit
them as templates for new work. Current residents:

- `Demo DESI Spectra-Copy1.ipynb` — accidental duplicate of `Demo DESI Spectra.ipynb`.
- `IntrumentalProfile.ipynb` — early instrumental-profile exploration (also note the filename typo: "Intrumental").

## Conventions

- This folder is **not the place to put a new user-facing tutorial**.
  If you want to write a polished walkthrough, add it under
  [`../tutorials/`](../tutorials/) following the conventions in that folder's
  README.
- These notebooks may reference paths that only exist on NERSC /
  GreatLakes scratch (`/pscratch/...`, `/nfs/turbo/...`). Don't
  expect them to run on a fresh laptop without setup.
