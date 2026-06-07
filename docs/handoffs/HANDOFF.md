# HANDOFF — current (2026-06-03, NERSC production launched)

> **Canonical, cross-machine handoff.** Lives in the repo so any clone sees it.
> Older/superseded handoffs are archived locally under `docs/handoffs/legacy/`
> (gitignored — not pushed). Claude's per-session working notes live in its
> local memory (`~/.claude/.../memory/`), not here.

---

## 2026-06-03 — NERSC Perlmutter session (READ THIS FIRST)

**Where we are:** on NERSC, branch **`nersc_production`** (off merged `desi_y3`), repo at
`/pscratch/sd/j/jibancat/desi_gpy_dla_detection`. The NERSC production port is built,
committed, pushed, and **PR #12 → `desi_y3` is open**. The first production run (London-0)
is **launched and in flight**.

### London-0 catalog — DONE ✅ (2026-06-04)
- **Result: 1149/1150 healpix, P 0.844 / C 0.907 (lyα)** — faithful to the 32-hpx calibration (0.852/0.906) and above the GL reference purity (0.818/0.904 → no drift). Catalog: `…/nersc_prod_london0_v1_20260603/outputs/combined_dlacat.fits` (741,760 rows / 348,272 sightlines). Ready for the last-year comparison.
- The "missing" healpix 715 is **benign** — its 1 QSO is z=1.926 (< the z>2 cut), so it correctly produces no h5. 1149/1150 is complete.
- **Gap-detection lesson (important for LOA scale):** `combine --fail-on-gap` checks *slice* coverage (misses per-healpix holes); `resume_positions` checks per-healpix h5 but **false-positives on z<2-only healpix**. A flagged position is a REAL gap only if that healpix has z>2 QSOs (cross-ref spectra-16 fibermap TARGETIDs vs the z>2 zcat). Verify before re-running. A true silent mid-file failure (no traceback, loop continues, no "Completed processing" line) IS possible — re-run `--start <idx> --end <idx+1> --window 1` (idempotent).

### 2LPT-1 catalog — DONE ✅ (2026-06-05)
- **6 sbatch jobs `53950094`–`53950099`** all COMPLETED (exit 0:0) overnight 2026-06-04 19:56 → 06-05 01:19 (PW50k, N32×W8).
- **Completeness: 1149/1150** — the lone missing h5 (position 550 → healpix 1046) is **benign** (its 1 QSO is z=1.825 < z>2 cut, so correctly no h5). Verified via `resume_positions --summary` + zcat cross-ref.
- **P/C faithful, no drift** (BAL-excl, lyβ-veto, P_DLA>0.99, SNR_RED>2; NHI-desc matcher):
  | window | logNHI>20 (P/C) | logNHI>20.3 (P/C) |
  |---|---|---|
  | **full [911,1216]** (headline) | **0.818 / 0.874** | **0.815 / 0.888** |
  | lyα-only [1025,1216] | 0.836 / 0.892 | 0.835 / 0.903 |
  (2LPT-0 GL ref lyα-only NHI≥20.3 = 0.818/0.891 → consistent.)
- **Packaged + shared:** `…/DLA/2lpt/mock-1/2lpt1_loa124_v1/` (FITS 799,162 rows / 362,445 TIDs, 23 cols w/ flag columns, DLAFLAG==0 clean=656,102/82.1%, P_DLA clipped, commit 891db99) + README + BASELINE.env + `diagnostics/nhi20.{0,3}/{lya_lyb,lya_only}`. Bundle: `/pscratch/sd/j/jibancat/nersc_prod_2lpt1_v1_20260604/combined_catalog/`.
- **NOTE:** the earlier packaging run died right after combine (the 07:22 bundle FITS was a raw combine, no flags/provenance/README) — re-ran `package_catalog.sh` cleanly 2026-06-05. Watch for this if a session disconnects mid-package.
- node-hour estimate 36 vs London-0 actual 32.2 (~13% conservative); 2LPT per-spec ≈ London confirmed.
### LOA catalog (REAL DESI Y3) — DONE ✅ (2026-06-06)
- **The headline NERSC deliverable.** Run `nersc_prod_loa_v1_20260606`, **9 sbatch jobs `54044805`–`54044860`** (window 2048 → 9 jobs; clean BASELINE @ commit `84fa654`). All COMPLETED, ~34 nh actual (vs 21.7 est — the per-job 3 GB altbal-cat read adds overhead at 9 jobs). **16,519/16,519 healpix processed** (full z>2 coverage).
- **Catalog:** `…/DLA/loa/loa_main_dark_v1/` (bundle: FITS + README + BASELINE.env + diagnostics/). FITS `dlacat-loa-main-dark-v1.fits` = 801,761 rows / 358,835 sightlines, DLAFLAG==0 clean 723,805 (90.3%), P_DLA clipped. At clean cut (DLAFLAG==0, p>0.99, SNR_RED>2): **NHI>20 = 91,900 DLAs/74,981 sightlines; NHI>20.3 = 65,325/56,155**. 77% sightline overlap w/ prior epoch-920 LOA cat (different model — sane).
- **No truth (real data):** P/C cited from mocks (London-0 0.84/0.91, 2LPT-1 0.82/0.87, full [911,1216] NHI>20). BAL = `BI_CIV>0` from the altbal QSO cat (56,740 TIDs), built via new `tools/postprocess/build_bal_cat_from_qsocat.py`.
- **⚠ TWO LOA-path bugs fixed this session (commit `84fa654`)** — the MODE=loa path had never been exercised (GL lacked real LOA):
  1. `desi-DLAGP.py:534` iterated a big-endian memoryview (`all_hpxs.data[...]`) → `NotImplementedError: memoryview: unsupported format >q` under py3.13/current numpy. Crashed every task **but the sbatch wrapper still exited 0** (silent failure — first 33-job run reported COMPLETED with ZERO output). Fixed: drop `.data` (matches the external-list branch). **Lesson: for LOA always confirm `processed-*.h5` count, NOT just sacct State.**
  2. `launch_nersc.sh` `MOCKDIR=${MOCKDIR}` aborted under `set -u` for LOA. Fixed: `${MOCKDIR:-}`.
- (Stale `nersc_prod_loa_v1_20260605/` = the first crashed run, 0 outputs — safe to delete.)
### NEXT: **2LPT-2** (`2lpt2_nersc_v1.env`, ~36 nh, budget-gated) and **LOA CDDF** (`loa_cddf_nersc.env`, PW100k, FILTER=0, MAX_DLAS=1, ~120 nh) — both NOT launched; confirm before spending. LOA CDDF reuses the same BAL helper + finish recipe.

### (historical) London-0 launch — 6 sbatch jobs `53867901`–`53867906`
- Config: `slurm/nersc/production/london0_nersc_v1.env`, **PW50k**, N32×W8, ~36 nh, OUTDIR
  `/pscratch/sd/j/jibancat/nersc_prod_london0_v1_20260603/outputs/`.
- **Check status:** `sacct -j 53867901,53867902,53867903,53867904,53867905,53867906 --format=JobID,State,Elapsed`
- **When all 6 COMPLETE**, finish the run:
  ```bash
  source /global/cfs/cdirs/desi/software/desi_environment.sh main   # (wrap in set +u)
  OUT=/pscratch/sd/j/jibancat/nersc_prod_london0_v1_20260603/outputs
  python examples/combine_dlacat.py --procdir $OUT --out $OUT/combined_dlacat.fits --expect-positions 1150 --fail-on-gap
  ln -sfn $OUT/figures/processed $OUT/processed
  python examples/molly_faithful_pc_plots.py --catalog-dir $OUT \
     --truth /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits \
     --bal-cat .../bal_cat.fits --no-bal --mockdir .../jura-124 \
     --snr-min 2.0 --nhi-min 20.3 --gp-conf 0.99 --lyb-veto --restrict-truth-to-processed --out $OUT/pc/
  ```
  Expect P≈0.85 / C≈0.91 (calibration gave 0.852/0.906). Then compare to last year's London-0.
- **If a job times out / a position is missing** (combine `--fail-on-gap` flags it): re-launch the
  not-done range — `launch_nersc.sh london0_nersc_v1.env --outdir $OUT --start <first_not_done> --end <end> --window 192 --time 08:00:00` (idempotent), then re-combine. Per-healpix h5 is checkpointed so completed healpix survive a kill.

### Remaining production plan (user's order; budget-gated)
663 nh left; **preserve ~300 nh reserve** → ~363 spendable.
1. **London-0** (running) — last-year comparison.
2. **2LPT-1, 2LPT-2** catalog → `2lpt1_nersc_v1.env`, `2lpt2_nersc_v1.env` (~36 nh each; 2LPT per-spec assumed≈London, pin on first run).
3. **LOA catalog** → `loa_nersc_v1.env` (`--window ~4500`, ~22 nh measured).
4. **LOA CDDF** (PW100k) → `loa_cddf_nersc.env` (`--window ~1600`, ~120 nh; PW100k is extrapolated — optionally pin with one regular slice). **CDDF on LOA only.**
5. Hold ≥300 nh for 2–3 DR3 Matterhorn (next-gen LOA) catalog iterations (~22 nh each).

### Key findings (don't re-litigate — see memory `nersc-parallelism-and-cost` + notes repo)
- Packing **N32×W8** optimal (GL's W=16 doesn't transfer). P/C: PW50k 0.852/0.906, PW≥30k all hit 85/85; PW10k misses purity (not pDLA-recoverable). PW100k reproduces GL 0.818/0.904 → **port faithful**.
- Model + grids **md5-verified byte-identical to GL** (`…/learned/greatlakes/`); PW50k = first-50k prefix of the 100k.
- **Woodbury batching REFUTED** (~1.4× e2e, not 2×; don't do the PR). **K-reduction rejected** (fidelity). **nfl=31 correct** (P/C-insensitive). FILTER-floor & load-balancing are minor (~7–13%). Real 10× = GPU only.
- Memory fits N32×W8 easily (102/76 GB of 503). Investigation write-ups: private notes repo `desi_gpy_dla_notes` (`notes/2026-06-0{1,2,3}_*`).

### What this session changed in git (PR #12, base `desi_y3`)
`slurm/nersc/production/` (full port + README), `tools/make_subsampled_grids.py`, `slurm/run_local.sh` (+1 line `--max_z_cut`). **No inference-code changes** (diff vs `gpy_dla_detection/`/`dla_gp.py`/`dlasearch.py`/`desi-DLAGP.py` is empty).

---

### Prior GreatLakes context (2026-05-27, superseded by the above for NERSC work)

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
