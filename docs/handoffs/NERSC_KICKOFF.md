# NERSC kickoff — note for the NERSC Claude agent

> Written 2026-05-28, **updated 2026-05-30** by the GL-side Claude as a
> cross-machine handoff. Read the **2026-05-30 UPDATE** block first (it reflects
> the merged state), then the original Steps 0–5 below (still the correct
> recipe for the NERSC config port). Pair with `docs/handoffs/HANDOFF.md`.

---

## 2026-05-30 UPDATE — read this first

### What changed since this note was first written

Both PRs are now **MERGED into `desi_y3`**:

- **PR #7** (`production_533` → `desi_y3`) — **MERGED.** Carries all the GL
  production work and the three correctness fixes you must have before any run:
  the `−log_ratio` multi-DLA evidence fix in `dla_gp.py`, the order-agnostic
  BAL-mask fix in `dlasearch.py`, and the `+log(N)` MC-evidence patch. So the
  "Step 0 — wait for PR #7" gate below is **already satisfied.**
- **PR #10** (`clustering_prior` → `desi_y3`) — **MERGED.** Adds an
  evidence-only DLA close-pair clustering prior, **gated default-off and
  byte-identical when off** (verified by an `np.array_equal` parity test).
  **This does NOT change production behavior** unless someone explicitly passes
  `--pair_prior_mode clustering`. **Leave it off for all production runs.** A
  ~87k-DLA London-0 A/B (2026-05-29) showed the prior is a *close-pair no-op*
  (1 false new pair) — the bottleneck is the SIR proposal, not the weighting, so
  enabling it buys nothing today. Full write-up is in the notes repo
  (`notes/2026-05-29_clustering_prior_pair_purity_ab.md`). The real fix
  (future, not now) is a clustering-informed *proposal*; don't spend NERSC
  cycles enabling the prior.

Net: **`desi_y3` is now the single source of truth** with all fixes + both PRs.
Base every NERSC branch off it.

### Step A — get the NERSC clone unstuck (it's mid-PR-7)

The NERSC repo was left part-way through PR #7 and never advanced; it predates
both merges and the notes repo. Recover it carefully — **do not blow away local
NERSC work without checking** (there may be a half-built `launch_nersc.sh` etc.;
see Step 1 below):

```bash
cd /global/homes/j/jibancat/desi_gpy_dla_detection   # or wherever the NERSC clone is
git status --short                 # inspect FIRST — note any local/untracked NERSC work
git stash list                     # and any stashes
# if a merge/rebase/cherry-pick is half-applied and you don't need it:
git merge --abort 2>/dev/null; git rebase --abort 2>/dev/null; git cherry-pick --abort 2>/dev/null
git fetch origin
git checkout desi_y3 && git pull --ff-only origin desi_y3
# verify you have the fixes + both PRs:
git log --oneline -5                      # should show the PR #10 + PR #7 merge commits
git log --oneline | grep -iE "log_ratio|BAL masking|clustering" | head
```

If `git status` shows real NERSC WIP, `git stash` it (or commit to a wip branch)
before checking out `desi_y3`, then reapply onto the new `nersc_production`
branch. Ask the user before discarding anything unfamiliar.

### Step B — build the private notes repo on NERSC (it isn't there yet)

The NERSC side has **no clone of the investigation-notes repo**. Investigation
notes/figures live in a **separate private repo**, NOT in this codebase
(`docs/notes/` here is gitignored). Bootstrap it:

```bash
cd /global/homes/j/jibancat            # or any home-dir location you can write
gh repo clone jibanCat/desi_gpy_dla_notes   # needs `gh auth login` (or use the SSH URL)
# then read the recent context, most-recent first:
ls desi_gpy_dla_notes/notes/ | sort | tail
#   2026-05-29_clustering_prior_pair_purity_ab.md   ← the clustering A/B no-op result
#   2026-05-26 handoff, −log_ratio + BAL fixes, etc.
```

Write any NEW investigation write-ups THERE (`notes/*.md`), never into
`desi_gpy_dla_detection/docs/notes/`. Keep real-LOA spectra out of it — mocks
and derived numbers only (real-data privacy rule).

### Step C — configure NERSC production + run the remaining productions

Do the config port exactly as **Steps 1–5 below** (mirror
`slurm/greatlakes/production/` → `slurm/nersc/production/`; science knobs
**identical**, only paths/headers/env activation change). The V1 knobs to carry
verbatim are in Step 3's table and in each GL `*_v1.env`.

**Remaining productions** — confirm the exact list + priority **with the user**
before launching (I'm GL-side and don't have the authoritative NERSC job
ledger), but the known candidates from the canonical handoff are:

1. **Real DESI Y3 (LOA) production with the fixed code** — this is the headline
   NERSC deliverable: NERSC has the full real-spectra access GL lacks. Use the
   V1 knobs + `--pair_prior_mode off`. Mirror `slurm/submit_desi_loa.sh` into the
   new `slurm/nersc/production/` flavour, then `launch_nersc.sh`.
2. **London-0 full re-run with the fixed code** — the GL London-0 V1 catalog has
   a 161-healpix dlacat coverage gap and predates the `−log_ratio` fix, so it was
   deferred to a clean fixed-code re-run. Then re-tune p_DLA + re-calibrate P/C.
3. **CDDF input run** (FILTER-off, MAX_DLAS=1) to completion + `calc_cddf` on the
   2LPT catalog, if not already finished on GL.

**Before trusting any NERSC catalog**, run the Step 4 5k cross-check: NERSC P/C
must match the GL reference (2LPT-0 NHI≥20.3 → **P 0.8181 / C 0.8910**; London-0
5k PW-100k → **0.818 / 0.904**). A divergence ≥ a few pp means a knob drifted in
the port.

---

## Context

The user spent the recent sessions porting/improving production on **GreatLakes**
(GL) and is now switching back to NERSC, where they were **part way through the
NERSC-side port** before pivoting. PR #7 (`production_533 → desi_y3`) carries the
GL work; the user has decided **2LPT-0 is the merge gate** (no London re-run
needed). On the NERSC side, no twin of `slurm/greatlakes/production/` exists yet
— only the pre-V1 originals at `slurm/` top level.

## Decision (set by the user)

**Merge PR #7 first, then do NERSC config on a fresh branch off the merged `desi_y3`.**
Reasoning: PR #7 is already large (143 files, +19.7k/−4.6k, 71 commits ahead of
`desi_y3`); the NERSC work will inherit all of it cleanly via `desi_y3` as base, and
the two reviews stay small and focused. Do **NOT** bundle NERSC into PR #7.

## Step 0 — wait for PR #7 to land (or check it's merged)

```bash
gh pr view 7 --json state,mergedAt --jq '.state, .mergedAt'
```

Until PR #7 is `MERGED`, don't branch off `desi_y3` for NERSC work — you'd miss
the GL changes (the `−log_ratio` evidence fix in `dla_gp.py`, the BAL-mask order-
agnostic fix in `dlasearch.py`, the new `examples/combine_dlacat.py` +
`tools/postprocess/add_dla_flags.py` + `slurm/greatlakes/production/package_catalog.sh`,
the `examples/molly_faithful_pc_plots.py` overhauls). If it isn't merged yet and
the user wants NERSC work to start, you can base off `production_533` instead —
but the NERSC PR then has to wait on PR #7 anyway.

## Step 1 — grep NERSC FIRST for the half-finished work (do this before writing anything)

The user said they were "halfway into NERSC." Before recreating anything, look
for it. Likely places:

```bash
# the NERSC clone of the repo
ls /global/homes/j/jibancat/desi_gpy_dla_detection/slurm/ 2>/dev/null
git -C /global/homes/j/jibancat/desi_gpy_dla_detection status --short 2>/dev/null
git -C /global/homes/j/jibancat/desi_gpy_dla_detection branch -a 2>/dev/null | grep -iE "nersc|perlmutter|wip"
git -C /global/homes/j/jibancat/desi_gpy_dla_detection stash list 2>/dev/null
# untracked env / scripts on NERSC scratch
find /pscratch/sd/j/jibancat -maxdepth 3 -name '*.env' -o -name 'launch_*.sh' -o -name 'submit_*.sh' 2>/dev/null | head
```

If you find a partial `launch_nersc.sh`, `_base_nersc.env`, a stash, or an
uncommitted branch — **fold that in**, don't start blank. Ask the user to confirm
before deleting anything.

## Step 2 — branch off the merged `desi_y3`

```bash
git fetch origin
git checkout -b nersc_production origin/desi_y3
```

## Step 3 — mirror `slurm/greatlakes/production/` to `slurm/nersc/production/`

Add a new dir paralleling the GL one. **Identical science knobs**; only paths,
SLURM headers, and env activation change.

| GL file (reference) | New NERSC file | What changes |
|---|---|---|
| `_base_gl.env` | `_base_nersc.env` | `GL_SLURM_ACCOUNT=desi` (or rename → `NERSC_SLURM_ACCOUNT`); partition `regular`/`debug` instead of `standard`; constraint `-C cpu`; `GL_OUTPUT_ROOT=/pscratch/sd/j/jibancat/`; `GL_DATA_ROOT=/global/cfs/cdirs/desi/.../data/dr12q/processed/` (the user's NERSC data root); `GL_CONDA_SETUP`/`GL_CONDA_ENV` swapped for `source /global/cfs/cdirs/desi/software/desi_environment.sh main`; libcerf path drops (NERSC env has it) |
| `london0_gl_v1.env`, `2lpt0_gl_v1.env`, `2lpt0_gl_v1_filteroff_maxdla1.env` | `london0_nersc_v1.env`, `2lpt0_nersc_v1.env`, `2lpt0_nersc_v1_filteroff_maxdla1.env` | `source "$(dirname …)/_base_nersc.env"`; same `MAX_DLAS=4`, `SINGLE_ABSORBER_MODEL=1`, `FILTER_LOW_LIKELIHOOD=1`, `MAX_LAMBDA=1250`, `NUM_FOREST_LINES=31`, `ENABLE_TAU_EB=1`, `TAU_EB_OBJECTIVE=null`, PW-100k, `BALMASK=false`; swap `MOCKDIR` to `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124` (London) and the corresponding 2lpt path |
| `launch_gl.sh` | `launch_nersc.sh` | `ALLOWED_PREFIXES=("/pscratch/sd/j/jibancat/" "/global/cfs/cdirs/desi/users/jibancat/")` per the NERSC write-permission note; conda activation block swapped; `CODE_COMMIT` recording stays (the lines added by `30774b4`) |
| `submit_desi_mock_gl.sh` (+ `_resume`) | `submit_desi_mock_nersc.sh` (+ `_resume`) | `#SBATCH -A desi -q regular -C cpu -t HH:MM:SS`; remove `--mem` (NERSC partitions don't use it the same way); the inner srun block stays |
| `launch_gl_resume.sh`, `resume_positions.py`, `repack_gzip.sh`, `make_snr_cat_from_processed.py` | reuse as-is | already cluster-agnostic — only the env var defaults need pointing at the new `_base_nersc.env` |
| `slurm/greatlakes/production/package_catalog.sh` + `_write_catalog_readme.py` | reuse as-is | already cluster-agnostic; just point `--share-to` at a NERSC-accessible share location instead of the GL Turbo path |

The science knobs are **non-negotiably identical** to the GL config; only the
plumbing differs. The user's project memory has a hard rule: **don't modify
`dla_gp.py` / inference code for cluster differences — speedups are config-only
per cluster.** Same applies in reverse: the only thing the NERSC port should
touch is paths, SLURM headers, env activation, and the writable-prefix allowlist.

## Step 4 — validate: 5k slice cross-check against GL

The strongest validation: run the same 5k slice on NERSC and confirm the P/C
matches GL to within run-to-run shuffle. The truth catalog + matcher are cluster-
agnostic, so any divergence ≥ a few pp means a config drift.

```bash
# example: 5k slice of London-0 V1 on NERSC, then molly P/C against
# /global/cfs/projectdirs/desi/mocks/.../jura-124/dla_cat.fits
bash slurm/nersc/production/launch_nersc.sh london0_nersc_v1.env --start 0 --end 0 --window 5
# when done, run molly_faithful_pc_plots.py with the same headline cuts as GL
```

GL reference numbers to match: **5k London-0 PW-100k → P 0.818 / C 0.904**
(full London headline). 2LPT-0 baseline NHI≥20.3 → **0.8181 / 0.8910** (this is
what the verified Turbo catalog gives — see `docs/handoffs/HANDOFF.md`).

## Step 5 — open the NERSC PR

Small, focused, against `desi_y3`. Should be ~15-20 new files all under
`slurm/nersc/production/`. The PR description should:

- state what cluster it adds support for;
- confirm the science knobs are byte-identical to the GL V1 config (cite the env
  files);
- include the 5k validation P/C number side by side with GL's;
- explicitly note: no changes to `dla_gp.py`, `dlasearch.py`, `desi-DLAGP.py`, or
  any examples/ that aren't path-related.

## Invariants and don'ts

- **Don't touch inference code** (`gpy_dla_detection/`, `dla_gp.py`, `dlasearch.py`,
  `desi-DLAGP.py`, etc.) for cluster differences. Cluster-specific work is
  config + SLURM wrapping only. (Project memory: `feedback_dont_modify_dla_gp_behaviour`,
  `gl-blas-oversubscription`.)
- **Don't kill in-flight jobs without asking.** If there's a yueyingn0 CDDF
  resume still running on GL when you start, leave it.
- **Real-data privacy** — only mock catalogs go in public/share locations; real
  LOA spectra stay behind NERSC paths and never get committed to the notes repo
  or the codebase.
- **The notes repo is separate.** New investigation notes go to
  `github.com/jibanCat/desi_gpy_dla_notes` (private), not into `desi_gpy_dla_detection/docs/notes/`
  (which is gitignored). See memory `reference-notes-repo`.

## Useful pointers

- **GL infra to mirror:** `slurm/greatlakes/production/` (env files, launchers,
  resume driver, `package_catalog.sh`, `_write_catalog_readme.py`,
  `resume_positions.py`, `repack_gzip.sh`).
- **Verified GL reference catalog:** Turbo `/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/2lpt0_loa124_v1/`
  (catalog + README + BASELINE.env). NERSC's first 2LPT run should converge to
  the same P/C; if it doesn't, look for a knob drift.
- **Packaging routine:** `slurm/greatlakes/production/package_catalog.sh` — works
  cross-cluster; use it on NERSC too so the shared NERSC catalog has the same
  schema + provenance header.
- **NERSC paths cheat-sheet (from `CLAUDE.md`):** scratch `/pscratch/sd/j/jibancat/`;
  data `/global/cfs/projectdirs/desi/` + `/global/cfs/cdirs/desicollab/`;
  env activation `source /global/cfs/cdirs/desi/software/desi_environment.sh main`;
  account `-A desi`; queue `-q regular` / `-q debug`. Write permissions strictly
  under `/pscratch/sd/j/jibancat/`, `/global/homes/j/jibancat/`,
  `/global/cfs/cdirs/desicollab/users/jibancat/` — read `docs/nersc_write_permissions.md`
  before the first `mkdir`.

## What this gives the user (the end state)

When NERSC PR lands, the user can:

```bash
# launch a production run on either cluster with identical science knobs
bash slurm/greatlakes/production/launch_gl.sh 2lpt0_gl_v1.env       # GL
bash slurm/nersc/production/launch_nersc.sh 2lpt0_nersc_v1.env      # NERSC
# combine + flag + ship the same way from either side
bash slurm/<cluster>/production/package_catalog.sh --rundir … --share-to …
```

— and the headline P/C numbers will agree by construction (science knobs identical,
inference code identical).
