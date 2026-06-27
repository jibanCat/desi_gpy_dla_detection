# HBI results-store pipeline — implementation plan

Recompute every intermediate the HBI tutorial notebooks (NB0–NB5) + the paper TeX
draft depend on, in a structured/reproducible store, implementing
`CDDF_analysis/RESULTS_STORE_PLAN.md`. **This pipeline CHAINS existing producers; it
does not rewrite the science.**

## Producers chained (already exist)
| producer | stage | output | in-session? |
|---|---|---|---|
| `examples/molly_faithful_pc_plots.py` | completeness_molly | `molly_matrix.tsv` | yes (~min) |
| `CDDF_analysis/hbi/znz_kernel.py build-cache` | kernel_znz | `znz_2lpt0.npz` | yes |
| `…znz_kernel.py build-forward-cache` | kernel_fwd | `forward_response_2lpt0.npz` | yes (deterministic) |
| `CDDF_analysis/hbi/run_remp_kernel.py --stage build` | kernel_remp | R_emp `posterior_kernel.npz` | yes (~20 min) |
| `CDDF_analysis/hbi/build_loa0_fp_product.py` | fp_loa0 | `loa0_fp_product.npz` | yes (~min) |
| `run_phase3d_postkernel.py --stage 1` | kernel_sir | SIR kernel (1.6 GB) | **CLUSTER** (3–5h, 1150 h5) |
| `run_phase3d_postkernel.py --stage 2/3` | fit_map / band | point_kernel.npz, band npz | yes (~10/~30 min) |
| `track_c_tf_{loa,2lpt1,london0}.py` | reduction | `track_c_tf_*.json` | yes (~min, n_mc=120) |
| `CDDF_analysis/loa_literal_calccddf.py` | reduction_raw | `loa_literal_calccddf_*.json` | yes |
| `injection/measure_competed.py` | completeness_camp | `competed_completeness.npz` | CLUSTER (needs campaignS gp_out) |

**Primary inputs (NOT recomputed — upstream GP inference):** the dlacat catalogs,
`hcd_truth_cat.fits`, `bal_cat.fits`, `processed-main-dark-*.h5`. Registered in the
store as external leaves with a privacy class.

## Files to create
```
CDDF_analysis/results_store.py        # resolver: ResultStore.get/by_id/new/list + sqlite manifest
CDDF_analysis/hbi/provenance.py       # git_stamp, write_provenance (README+provenance.json), privacy, config_hash, slug
pipeline/{__init__,datasets,stages,run_pipeline}.py   # dataset bundles, stage catalog, DAG driver/CLI
tools/provenance/precommit_privacy_guard.py           # reject staged real_loa/shareable:false leaves
tests/{test_results_store,test_provenance,test_pipeline_stages}.py
$CDDF_STORE/{mock,real_loa}/{dataset}/{stage}/{slug}__{hash8}/{result.*,run.json,README.md,run.log}  # outside repo
```

## Tasks (bite-sized; in-session unless noted)
1. **provenance.py** — `git_stamp()` (clean/dirty/unknown, generalize `track_c_tf_loa._git_commit`), `config_hash()` (sha1 sorted-json[:8]), `make_slug()`, `privacy_class()` (contagious: real-LOA iff any input real-LOA), `write_provenance()` (README + provenance.json atomically). + `tests/test_provenance.py`.
2. **results_store.py** — `ResultStore.{leaf_path, new, commit_leaf, get(strict: 0/>1 raises), by_id, list, rebuild_manifest}` over `MANIFEST.sqlite` (+ json mirror, rebuildable from leaves). + `tests/test_results_store.py`.
3. **pipeline/datasets.py** — per-dataset primary-input bundles (defaults = producers' current constants → byte-identical repro), each with `privacy`.
4. **pipeline/stages.py** — the stage catalog: thin wrappers that resolve upstream leaves via `store.get`, invoke the existing producer's `main(argv)`/build fn with `--out`=fresh leaf, then `store.commit_leaf` stamping provenance. Cluster-only stages raise `ClusterOnlyStage` emitting the sbatch line.
5. **pipeline/run_pipeline.py** — topo-sort DAG driver; `--dataset --stage{|all} --resume --dry-run --store --cluster-emit --ingest`. + `tests/test_pipeline_stages.py` (dry-run topo order).
6. **Wire NB0–5 + scratch_build_nb*.py + regen script to the store** — `tutorial_fixture(name, dataset, stage, selection)`: `$CDDF_STORE` set → `store.get(...).path()`, else committed `tutorial_data/` (the no-scratch fallback). NB5 already runs live → point its `--forward-model/--molly-tsv/--loa0-product` at `store.get(...).path()`.
7. **Privacy guard** — `tools/provenance/precommit_privacy_guard.py` (reject staged real-LOA leaves) + `real_loa/.gitignore '*'` + DO_NOT_COMMIT sentinel.
8. **SLURM ingest** — `--cluster-emit` prints the existing sbatch (phase3d_postkernel_staged / remp_kernel) pointed at a leaf; `--ingest` stamps provenance after the cluster job lands.
9. **Deferred (separate PR):** `cddf_result_io.py` + canonical `cddf-result/1.0.0` schema unification; `migrate_legacy.py` of the 104 legacy scratch files. This pipeline treats each producer's existing output as an opaque leaf payload (buildable now).

Build order: 1→2→3→4→5 (the store+driver spine, TDD-gated) → 6→7 → 8.

## Recompute run plan — 2LPT-0 end-to-end (in-session, ~1h)
```bash
export CDDF_STORE=/scratch/.../cddf_store
python -m pipeline.run_pipeline --dataset 2lpt0 --stage completeness_molly   # ~min
python -m pipeline.run_pipeline --dataset 2lpt0 --stage kernel_znz           # stage-0
python -m pipeline.run_pipeline --dataset 2lpt0 --stage kernel_fwd           # forward_response_2lpt0.npz
python -m pipeline.run_pipeline --dataset 2lpt0 --stage kernel_remp          # ~20 min (R_emp, no processed-h5)
python -m pipeline.run_pipeline --dataset 2lpt0 --stage fp_loa0              # ~min
python -m pipeline.run_pipeline --dataset 2lpt0 --stage fit_map              # ~10 min
python -m pipeline.run_pipeline --dataset 2lpt0 --stage band                 # ~30 min
python -m pipeline.run_pipeline --dataset 2lpt0 --stage reduction            # ~min (track_c_tf, n_mc=120)
```
- **Cluster-only (honest):** `kernel_sir` (SIR 1.6 GB, 1150 processed-h5 → `phase3d_postkernel_staged.sbatch STAGE=1`, 3–5h); `completeness_camp` (needs the injection campaignS GP re-run). The headline runs on the **R_emp** kernel which IS in-session.
- **Other datasets:** `2lpt1`/`london0` reuse steps with `--dataset`; `real_loa --stage reduction` only (frozen 2LPT-0 calibration + real catalog = NB5's live path), every leaf under `real_loa/` (unstageable by the guard).
