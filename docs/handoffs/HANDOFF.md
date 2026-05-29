# HANDOFF — current (2026-05-27)

> **Canonical, cross-machine handoff.** Lives in the repo so any clone sees it.
> Older/superseded handoffs are archived locally under `docs/handoffs/legacy/`
> (gitignored — not pushed). Claude's per-session working notes live in its
> local memory (`~/.claude/.../memory/`), not here.

- **Repo:** `/home/mfho/desi_gpy_dla_detection`, branch **`production_533`** (PR #7 → `desi_y3`).
- **Cluster:** GreatLakes. Scratch outputs under `/scratch/cavestru_root/cavestru0/mfho/`.
  Persistent shareables under `/nfs/turbo/lsa-cavestru/mfho/`.

## State of the science (as of 2026-05-27)

- **2LPT-0 loa-124 V1 baseline — DONE, 1150/1150** (`gl_prod_2lpt0_v1_20260526`, FILTER-on,
  fixed code @ `b219996`). The trustworthy post-fix reference.
  - Molly P/C (NHI-desc matcher, `--no-bal`, lyb-veto, SNR>2, p_DLA>0.99):
    **NHI≥20.3 → P 0.8181 / C 0.8910**; NHI≥20 → 0.8232/0.8742; NHI≥19 → 0.4318/0.7733
    (≥19 purity drop is the sub-DLA regime; the DLA model isn't a sub-DLA finder).
  - Molly P/C eval **verified faithful** to Molly Wolfson's notebook (only intentional diff =
    the NHI-descending matcher; dlacat SNR == snr_cat SNR byte-identical).
- **Shareable catalog (for collaborators) on Turbo:**
  `/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/2lpt0_loa124_v1/`
  — `dlacat-v2.8.5-mockcat.fits` (769,833 rows / 361,167 sightlines; flag columns added,
  `DLAFLAG==0` = clean incl. lyb+BAL; provenance + `CODECMT=b219996` in header) +
  `README.md` + `BASELINE.env` + `example_spectra/` (3 annotated example plots).
- **Bayesian-CDDF input** = the separate FILTER-off MAX_DLAS=1 run
  (`gl_cddf_2lpt0_v1_filteroff_maxdla1_20260526`) — partial (~700/1150); see budget below.
- **London-0 V1 catalog is NOT usable** (`gl_prod_london0_v1_preclustering_20260522`):
  161/1150 healpix have an h5 but no dlacat (resume gap; `resume_positions.py` checks h5 only).
  Its P/C is a coverage artifact. **Deferred** to the full fixed-code re-run (the PR #7 merge gate),
  which fixes both this and the pre−log_ratio-fix staleness.

## Budget / allocation (⚠️ live decision)

- **cavestru0 nearly exhausted:** ~75.5k / 80k CPU-hr used (~4.5k left). Jobs block at the
  GrpTRESMins cap — no overspend.
- **CDDF FILTER-off run:** ~700/1150 done. Finishing needs ~15.3k CPU-hr (≈3.5× the cavestru0
  remainder). A *partial* CDDF is usable (path-length normalization handles any sightline subset).
- **Action taken (2026-05-27):** cancelled the 22 pending cavestru0 CDDF jobs; **a background
  watcher waits for the ~17 running cavestru0 CDDF jobs to finish, then auto-submits the
  remaining ~449 healpix (~15k CPU-hr) on `--account=yueyingn0`** (the resume re-computes
  not-done at submit time → no overlap/double-write). ⚠️ Confirm yueyingn0 can absorb ~15k CPU-hr.

## Code changes this session (all pushed to `production_533`)

- `0ceacfc` configurable `--nhi-bins`/`--snr-bins` + `molly_matrix.tsv` in `molly_faithful_pc_plots.py`.
- `71e1e1d` `examples/combine_dlacat.py` — glob-based dlacat combiner (gap-checked + provenance),
  replaces the rigid `combine_dlamocks.py` grid-walk that silently dropped resume slices.
- `30774b4` `slurm/greatlakes/production/package_catalog.sh` (+ `_write_catalog_readme.py`) — the
  standard post-run packaging routine (combine→flag→stamp commit→env→README→`--share-to`);
  `launch_gl.sh` now records `CODE_COMMIT` in each run's `BASELINE.env`.
- `079f1ca` + `9166b3d` `2lpt` smoke preset + `--truth-cat`/`--bal-cat` overlays + `--learned-file`
  override in `plot_smoke_v2.py`.

## Next, in order

1. **Confirm/let the yueyingn0 CDDF resume run** (watcher auto-submits when cavestru0 jobs finish).
   Then P/C + CDDF (`calc_cddf`) on the 2lpt catalog (`hcd_truth_cat.fits`).
2. **London full re-run with the fixed code** (PR #7 pre-merge gate; fixes the coverage gap +
   −log_ratio staleness) → re-tune p_DLA + re-calibrate P/C. Budget-gated.
3. **PR #7 → desi_y3**, then the `clustering_prior` PR (still unpushed, merge-ready).
4. (done 2026-05-27) `docs/notes/` migrated to a separate notes repo — see Pointers.

## Pointers

- **Investigation notes + figures: separate private repo `github.com/jibanCat/desi_gpy_dla_notes`**
  (NOT in this codebase — `docs/notes/` is gitignored + untracked). **Read it for prior
  investigation context** (`git clone`/`git pull`, then read `notes/*.md`). Add new write-ups
  there, not here.
- Detailed prior handoffs: `docs/handoffs/legacy/` (local archive, gitignored).
- Per-knob production baseline + justification: top of PR #7 description.
- Packaging routine: `slurm/greatlakes/production/package_catalog.sh`.
