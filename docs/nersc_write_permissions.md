# NERSC write-permission rules for this repo

**Read this first before any `Write` / `Edit` / `Bash` that creates files.**

This repo is checked out at NERSC (Perlmutter). Outside of a small set of
paths, the filesystem is **read-only for the `jibancat` account**, and trying
to write there will either fail with `Permission denied` or — worse — succeed
in a place that other users will not expect to have files appear in.

## Allowed write paths (the only places you may create/modify files)

| Path | Purpose |
|------|---------|
| `/pscratch/sd/j/jibancat/` | Scratch (Lustre). Inference outputs, intermediate HDF5, logs, big I/O. **Default for any new artifact.** Note: `$PSCRATCH` is purged after ~8 weeks of inactivity — move keepers to a CFS project area before they get culled. |
| `/global/homes/j/jibancat/` | `$HOME`. Code, dotfiles, venvs, shell config, `~/.claude/`. Backed up. 40 GB / 1 M inode quota — do **not** dump millions of small files here (pip caches, log dirs). |
| `/global/cfs/cdirs/desicollab/users/jibancat/` | DESI collab CFS area for durable outputs you want other DESI members to see. Group-writable, not purged. Use this for the final catalogs / combined HDF5 / paper-figure inputs. |

The current working tree, `/pscratch/sd/j/jibancat/desi_gpy_dla_detection/`,
is on `$PSCRATCH` — so editing the repo itself is fine.

## Read-only (do NOT try to write here)

- `/global/cfs/cdirs/desi/` (DESI shared production area — Y3 LOA spectra, public mocks)
- `/global/cfs/cdirs/desicollab/mocks/` (collab mock spectra: London, Saclay, 2LPT)
- `/global/cfs/projectdirs/desi/` (legacy mirror of the above; same files)
- `/global/cfs/cdirs/cosmo/`, anyone else's `users/<them>/`
- `/global/common/software/` (shared software installs)
- Any other `/global/cfs/cdirs/*/` not listed above

If you need to "modify" something that lives in a read-only area, **copy** it
into one of the allowed write paths first and edit the copy.

## SLURM output / log paths

The current submit scripts under `slurm/` write `*.log` / `*.err` to the
working directory (relative paths in `#SBATCH --output=` and `--error=`).
Because the working directory at submit time is the repo root on `$PSCRATCH`,
that already lands in an allowed write path. If you templatize / move scripts,
keep this invariant.

## DESI account / repo

- **CPU jobs:** `#SBATCH -A desi`
- **GPU jobs:** `#SBATCH -A desi_g`  (different repo — required suffix per NERSC)
- QOS namespace is split too: CPU uses `regular`/`shared`/`debug`/`interactive`;
  GPU uses `gpu_regular`/`gpu_shared`/`gpu_debug`/`gpu_interactive`.

## Login-node discipline

Compute-heavy work (`python desi-DLAGP.py`, anything multi-core or many-GB)
**must** be inside an `sbatch` or `salloc`, not on the login node. The login
node is fine for `git`, `pytest` (a quick run), editing, building a single
`voigt_fast.so`, and small data poking. Multi-minute compute on the login node
will be throttled (silently slow) and may be killed.

## Compute nodes lack internet

`pip install`, `curl https://...`, `gh auth`, etc. will all fail inside an
sbatch job. Install dependencies on the login node first.

## Common-software / conda-env caveat

`PYTHONPATH` set in `~/.bashrc` is prepended to every Python invocation and
will silently shadow venv / conda installs. If a venv import fails inexplicably
in a job, check `echo $PYTHONPATH` first.
