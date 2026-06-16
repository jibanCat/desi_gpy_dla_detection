# Where the investigation notes went (read this first, next Claude)

The Claude-oriented **working docs** — investigation notes, run notes, per-mock
stories, session handoffs, and design memos — have been **moved out of this code
repo** to keep it clean. They now live in the **private notes repo**.

This is a deliberate hygiene choice: these are working notes meant to be read by
Claude, not part of the shipped codebase. They are **not secret science** (the
team's P/C, dN/dX, Ω_HI and DLA-catalog results stay public in this repo); the QSO
*input-catalog* counts are the only thing scrubbed for confidentiality.

## Where they are now

**Private notes repo:** `jibanCat/desi_gpy_dla_notes`
- On NERSC: `/global/homes/j/jibancat/desi_gpy_dla_notes/`
- The full mirror of what used to be here is under **`from_code_repo/docs/…`**,
  preserving the original tree.

| Used to be (this repo) | Now (private repo) | Files |
|---|---|---|
| `docs/notes/` | `from_code_repo/docs/notes/` | 147 |
| `docs/handoffs/` | `from_code_repo/docs/handoffs/` | 2 |
| `docs/runs/` | `from_code_repo/docs/runs/` | 7 |
| `docs/stories/` | `from_code_repo/docs/stories/` | 4 |
| `docs/superpowers/` | `from_code_repo/docs/superpowers/` | 6 |

To read them: `ls /global/homes/j/jibancat/desi_gpy_dla_notes/from_code_repo/docs/`
(or `git -C /global/homes/j/jibancat/desi_gpy_dla_notes pull` first to refresh).

## What stayed in this code repo

- All **reference / results docs**: tutorials, `architecture.md`, `data_inputs.md`,
  `paper_figures.md`, `production_runbook.md`, `production_models.md`,
  `CURRENT_MODELS.md`, `test_results_overview.md`, NERSC setup/permissions, etc.
- All **env / config** under `slurm/nersc/production/` (so others can reproduce).
- Code, tests, and the packaging guide.

## Note

`CLAUDE.md` and some kept docs still contain `docs/notes/…` links. Those targets now
live in the private repo at the `from_code_repo/` paths above (and remain in this
repo's git *history* — this was a tip removal, not a history rewrite).
