# desi_gpy_dla_detection — GP-DLA detection for DESI, and the Paper-1 hierarchical CDDF measurement

Two things live here, deliberately separated:

1. **The GP-DLA finder** (`gpy_dla_detection/`, `dlasearch.py`, `run_bayes_select.py`): the Gaussian-process
   detection of damped Lyman-α absorbers in quasar spectra (Garnett+17 / Ho+20 lineage, ported to DESI).
   It writes per-sightline DLA posteriors and catalogues. Its own README is `docs/FINDER_README.md`;
   its production runbook is `docs/production_runbook.md`.
2. **The hierarchical population inference (HBI)** (`CDDF_analysis/hbi_mcmc/`, `CDDF_analysis/hbi/`):
   selection-, response- and false-positive-aware inference of the DLA column-density distribution
   f(N_HI), the line density dN/dX and Ω_HI from those catalogues, calibrated on mocks and in situ
   injections. This is the **Paper-1 measurement of record (frozen 2026-08-26)**.

```
spectra / mocks ──▶ GP-DLA fits & catalogues ──▶ [mock truth, H2 injections]
      ──▶ completeness / FP / response calibration ──▶ certified HBI pack ──▶ chains ──▶ pooled
      frozen posterior ──▶ CDDF, dN/dX, Ω_HI ──▶ systematics products ──▶ paper figures & tables
```
The finder never imports the HBI code and the HBI engine consumes only a *pack* (`CDDF_analysis/hbi_mcmc/pack.py`);
the full dependency map and the future standalone-package boundary are in **`docs/HBI_ARCHITECTURE.md`**.

## Where to start
| you want to … | read / run |
|---|---|
| reproduce the Paper-1 figures and tables from the frozen artifacts (minutes) | `docs/PAPER1_REPRODUCTION.md` §7 (mode A): verify manifest → `build_all.py` → ledger/provenance checks |
| reproduce the Paper-1 science products (≈ 15 core-h) | `docs/PAPER1_REPRODUCTION.md` §3–§6 (mode B): the six sbatch scripts + the BH launch script in `slurm/greatlakes/production/paper1/` |
| know exactly which files are frozen, and check them | `docs/PAPER1_FROZEN_MANIFEST.json` (165 entries, sha256) — `python tools/paper1/frozen_manifest.py --verify docs/PAPER1_FROZEN_MANIFEST.json`; archive copy on Turbo (see the runbook §2) |
| see how the frozen posterior was assembled from its chains | `python tools/paper1/chain_ledger.py` (chain → convergence decision → pool → frozen posterior) |
| understand every producer edge, seed and hash from spectra to figures | `docs/PAPER1_PROVENANCE_DAG.md` |
| know which branch is authoritative and how the branches relate | `docs/PAPER1_BRANCH_TOPOLOGY.md` |
| run the tests | `tools/paper1/run_tests.sh hbi|finder|training` — one environment per profile (`tests/profiles/README.md`) |
| run the finder itself | `docs/FINDER_README.md`, `docs/production_runbook.md` |
| understand the estimator and its validation | `docs/P1_ESTIMAND_SPEC.md`, `docs/P1_COMPLETENESS_SPEC.md`, `docs/P1_STOPPING_RULE.md`, `CDDF_analysis/hbi_mcmc/*.py` docstrings |

## Environments
| env | what runs in it | lock of record |
|---|---|---|
| `gpdla` | the finder, catalogue post-processing, pack extraction, scan packs, certification, the BH arm, training (torch), finder/training tests | `slurm/greatlakes/production/env_pip_gpdla_2026-08-17.txt` on the python of `env_lock_gpdla_2026-08-17.txt` |
| `gpdla-hbi` | everything invoked as `python -m CDDF_analysis.hbi_mcmc.<module>` (the package imports jax): validation runs, real chains, pooling, PPC, audits; HBI tests | `slurm/greatlakes/production/env_lock_gpdla-hbi_2026-08-17.txt` (explicit conda) |
| paper figures | `gp_dla_desi_y3/paper_figures/*` (matplotlib decides the PDF bytes) | `gp_dla_desi_y3/paper_figures/ENV_LOCK_2026-08-26.txt` |

## Authoritative state for Paper 1 (2026-08-26)
- Science branch: `hbi/forward-2026-08` (this checkout). Guard-test/tombstone branch: `lls-subdla-cddf`. Main line: `desi_y3`. Roles, required commits and dispositions: `docs/PAPER1_BRANCH_TOPOLOGY.md`.
- Frozen low-z posterior `POOLED_ln_real_v2_20260821` (sha256 `ea881b5f…`) on the certified pack (`219c43aa…`); ratified BH arm `…gapc0.496_RATIFIED_20260826.json` (`62446b47…`); the paper repo refuses to build against anything else (`paper_figures/common.py::verify_frozen_status`).
- Superseded look-alikes (same basenames in `real_pack_v1/`, thirteen `gapc*` variants, five `adopted_packs_*` directories) are listed in the runbook §2 and are refused by the pooling and figure code.

## Layout
`gpy_dla_detection/` finder · `CDDF_analysis/hbi_mcmc/` HBI engine + pack producers · `CDDF_analysis/hbi/` DESI calibration producers (catalogue-HBI ingredients, kernels, FP product, BH arm) · `CDDF_analysis/cddf_forward/`, `CDDF_analysis/calc_cddf.py` the earlier feed-forward CDDF lineage (single-absorber path live; multi-DLA path retired) · `tools/h2_*` H2 injection campaign · `tools/paper1/` manifest, chain ledger, test runner · `slurm/` launch scripts (`production/paper1/` = the scripts of record) · `docs/` runbooks and specs · `tests/` (+ `tests/profiles/`) · `archive/` retired code kept for the record · `notebooks/`, `examples/` historical.

Private material (session notes, science numbers, real-data values) lives in a separate private notes repository, never here.
