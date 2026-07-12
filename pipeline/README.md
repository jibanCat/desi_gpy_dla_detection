# `pipeline/` — the HBI intermediate-results recompute pipeline

**One line:** a deterministic DAG driver that recomputes every intermediate the
catalog-HBI CDDF measurement depends on by *chaining the existing producers* into a
structured, provenance-stamped, write-once **results store** — so nothing the notebooks
or the paper rely on is a hand-made JSON floating on scratch.

It does **not** re-implement any science. Each stage is a thin wrapper that runs a
producer that already exists in the repo, captures its output as an immutable *leaf*,
and stamps it with a README + a git commit hash + a privacy class.

---

## The mental model

```
primary inputs                producers (already exist)            results store
(GP inference, mock      ──►   wrapped as pipeline STAGES    ──►   $CDDF_STORE/<privacy>/<dataset>/<stage>/<slug>__<hash8>/
 catalog, truth, bal)          (run_pipeline topo-sorts them)        ├── <result file(s)>   (the producer's real output)
                                                                     ├── README.md          (human-readable provenance)
                                                                     └── provenance.json    (config, inputs, code commit, privacy)
```

- **A leaf is addressed by its config.** The directory name ends in `__<hash8>`, an 8-char
  hash of the stage's full configuration. Change a science knob → new hash → new leaf.
  Same config → same leaf (idempotent; `--resume` skips it).
- **Leaves are write-once.** A committed leaf is never mutated. Re-running with a new
  config makes a *new* leaf; the old one stays as the record of what produced the old number.
- **Provenance travels with the data.** Every leaf carries the exact code commit, the CLI
  that made it, its inputs (by leaf id), and a regen command — so any number in a notebook
  or the paper traces back to "this code, this config, these inputs."

---

## Quickstart

```bash
export CDDF_STORE=/scratch/.../cddf_store         # the store lives OUTSIDE the repo

# see the whole DAG without running anything
python -m pipeline.run_pipeline --dataset 2lpt0 --stage all --dry-run

# recompute one intermediate (and whatever it depends on)
python -m pipeline.run_pipeline --dataset 2lpt0 --stage reduction --resume

# recompute the full in-session 2LPT-0 chain (~2 min; the cluster stages defer)
python -m pipeline.run_pipeline --dataset 2lpt0 --stage all --resume --cluster-emit
```

Flags: `--dataset` (`2lpt0`, `real_loa`, …) · `--stage` (one name or `all`) ·
`--store` (else `$CDDF_STORE`) · `--resume` (skip leaves that already exist) ·
`--dry-run` (print the topo plan, run nothing) · `--cluster-emit` (print the sbatch line
for cluster-only stages and defer their descendants instead of crashing).

---

## The stages (the DAG)

| stage | producer wrapped | output | where |
|---|---|---|---|
| `completeness_molly` | `examples/molly_faithful_pc_plots.py` | molly C/ρ matrix | in-session (~min) |
| `kernel_znz` | `znz_kernel.py build-cache` | `znz_*.npz` | in-session |
| `kernel_fwd` | `znz_kernel.py build-forward-cache` | forward-response kernel | in-session (deterministic) |
| `fp_loa0` | `build_loa0_fp_product.py` | loa-0 false-positive product | in-session (~min) |
| `kernel_remp` | `run_remp_kernel.py --stage build` | R_emp posterior kernel | in-session (~30 s) |
| `kernel_sir` | `run_phase3d_postkernel.py --stage 1` | SIR kernel (1.6 GB) | **cluster** (1150 processed-h5) |
| `fit_map` | `run_phase3d_postkernel.py --stage 2` | MAP point fit | cluster (needs `kernel_sir`) |
| `band` | `run_phase3d_postkernel.py --stage 3` | MC band | cluster (needs `fit_map`) |
| `reduction` | `track_c_tf_*.py` | per-z dN/dX, Ω, f(N) | in-session (~min) |

**In-session vs cluster is honest, not arbitrary.** The headline measurement
(`reduction`) runs in-session in ~minutes — it computes its uncertainty band *internally*
from the frozen forward kernel, so it does **not** need the heavy phase3d path. The
`kernel_sir → fit_map → band` diagnostic path genuinely needs the cluster (the SIR kernel
is 1.6 GB built over 1150 per-spectrum HDF5 files). For those, `--cluster-emit` prints:

```
sbatch --export=ALL,STAGE=1 slurm/greatlakes/production/phase3d_postkernel_staged.sbatch
```

run STAGE=1/2/3 on the cluster, then the produced leaf is registered in the store.

---

## Provenance & privacy

- `CDDF_analysis/hbi/provenance.py` — `git_stamp()` (commit + dirty flag), `config_hash()`
  (sha1 of sorted-json, first 8 chars), `privacy_class()` (**contagious**: a leaf is
  real-LOA if *any* input is real-LOA), `write_provenance()` (atomic README + provenance.json).
- Mock leaves live under `$CDDF_STORE/mock/…` and are shareable. Real-LOA leaves live under
  `$CDDF_STORE/real_loa/…` and must **never** be committed — `tools/provenance/precommit_privacy_guard.py`
  is a pre-commit hook that blocks staging any real-LOA leaf (by partition, by
  `provenance.json` privacy field, or by being a sibling of one; fail-closed on a malformed
  provenance). The store itself lives outside the repo on scratch by default.

---

## How the notebooks consume it

`CDDF_analysis/hbi/tutorial_data/fixtures.py::tutorial_fixture(name, ...)` is the bridge:

- `$CDDF_STORE` set **and** a matching committed leaf exists → use the fresh store leaf.
- otherwise → use the committed, version-controlled mock fixture in `tutorial_data/`.

So the tutorial notebooks are reproducible two ways: out of the box from the committed
mock fixtures (no scratch, CI-safe), or against a fresh recompute when you point
`$CDDF_STORE` at a store you built with this pipeline. Either way they never read an
ad-hoc scratch JSON.

---

## What this pipeline is *not* (yet)

Each producer's output is stored as an opaque leaf payload. A canonical
`cddf-result/1.0.0` schema (`cddf_result_io.py`) that unifies the payload formats, plus a
`migrate_legacy.py` for the historical scratch files, are a deliberate follow-up (see
`IMPLEMENTATION_PLAN.md`, task 9). The `--ingest` step that stamps provenance onto a leaf
*after* a cluster job lands is currently handled by re-running the stage; a first-class
`--ingest` is also follow-up.

See `IMPLEMENTATION_PLAN.md` for the task-by-task build log and
`../CDDF_analysis/RESULTS_STORE_PLAN.md` for the original design rationale.
