# v2 training run output layout

All v2 training submits (NERSC + GreatLakes) write to a uniform
`v2_runs/<RUN_TAG>/` layout so each run is one self-contained folder
that can be rsync'd to / from any cluster in one shot.

## Layout

```
${OUTDIR_BASE}/v2_runs/<RUN_TAG>/
├── trainset.h5                    preload output (legacy gp_interp_trainset schema)
├── config.json                    TrainConfig snapshot (lr, batch, epochs, ...)
├── loss_history.json              per-epoch loss list
├── slurm.log                      SLURM stdout (copied at end of script)
├── checkpoint_epoch_NNNN.pt       full Adam state (resume), every save_every
└── model_epoch_NNNN.h5            inference-ready models, legacy schema —
                                   drop-in for `dlasearch.py` / `DLAHolder`
```

`OUTDIR_BASE` defaults:
- NERSC: `/pscratch/sd/j/jibancat/desi_gpy_dla_detection`
- GreatLakes: `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection`

`RUN_TAG` defaults (overridable via `--export=ALL,RUN_TAG=...`):

| submit | default tag |
|---|---|
| `slurm/train/submit_e2e_train_loa_nersc.sh`        | `loa_${VARIANT}_${SLURM_JOB_ID}` |
| `slurm/train/submit_e2e_train_loa_nersc_debug.sh`  | `loa_${VARIANT}_dbg_${SLURM_JOB_ID}` |
| `slurm/train/preload_train_2lpt_nersc.sh`          | `2lpt_${TAG}_${SLURM_JOB_ID}` |
| `slurm/greatlakes/preload_train_2lpt.sh`           | `2lpt_${TAG}_${SLURM_JOB_ID}` |

So a finished run might live at e.g.:
- `/pscratch/sd/j/jibancat/desi_gpy_dla_detection/v2_runs/loa_no_hcd_with_bal_52234567/`
- `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_48881057/`

## Move runs between clusters (Globus)

To move EVERYTHING in one shot, transfer the whole `v2_runs/` tree:

| cluster | source / destination |
|---|---|
| NERSC | `/pscratch/sd/j/jibancat/desi_gpy_dla_detection/v2_runs/` |
| GreatLakes | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/` |

Each subdirectory under `v2_runs/` is one fully-self-contained run.
You can also pick individual run folders if you only want some.

The `trainset.h5` files are large (~hundreds of MB) — if you only
need the trained models for inference, you can deselect them in the
Globus UI and keep only `model_epoch_*.h5`, `config.json`,
`loss_history.json`, `slurm.log`.

## Resume a partial run

The trainer auto-resumes from the latest `checkpoint_epoch_NNNN.pt` in
`OUTPUT_DIR`. If a `-q debug` run capped at 200 epochs and you want to
continue to 800, just resubmit the regular-queue script with the same
`RUN_TAG`:

```bash
sbatch --export=ALL,VARIANT=no_hcd_with_bal,RUN_TAG=loa_no_hcd_with_bal_dbg_52234567 \
    slurm/train/submit_e2e_train_loa_nersc.sh
```

The trainer reads `checkpoint_epoch_0199.pt` from that dir and
continues training from epoch 200 onward (the regular submit's default
`NUM_EPOCHS=800`). The trainset.h5 from the debug run is also reused —
no preload re-run.

## Inspect outputs

`examples/diagnose_trained_gp.py visualize` works on any
`model_epoch_*.h5` directly:

```bash
python examples/diagnose_trained_gp.py visualize \
    --model legacy_y3:/path/to/model_epoch_920.h5 \
    --model v2_loa_no_hcd_with_bal:${OUTDIR_BASE}/v2_runs/loa_no_hcd_with_bal_<jobid>/model_epoch_0799.h5 \
    --out-dir /path/to/diagnostics/
```
