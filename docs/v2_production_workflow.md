# v2 production workflow — preload separately, train cheaply

End-to-end guide for producing all v2 GP models for PR #4 / task #20.
Decoupled into preload (CPU, regular queue) and train (GPU, debug
queue) so the GPU isn't paid to do FITS I/O.

## Where each step runs

|   | dataset | preload | train |
|---|---|---|---|
| 1 | LOA real (`no_dla_no_bal`) | NERSC `-q regular -C cpu` | NERSC `-q debug -C gpu` |
| 2 | LOA real (`no_hcd_with_bal`) | NERSC `-q regular -C cpu` | NERSC `-q debug -C gpu` |
| 3 | LOA real (`no_hcd_no_bal`) | NERSC `-q regular -C cpu` | NERSC `-q debug -C gpu` |
| 4 | 2LPT mock (`loa0`) | GreatLakes `-p standard` | GreatLakes `-p gpu_mig40` |
| 5 | 2LPT mock (`loa124_nohcd_nobal`) | GreatLakes `-p standard` | GreatLakes `-p gpu_mig40` |

Real LOA stays on NERSC (privacy + access). Mocks run on GreatLakes
(public, fast queues there). All runs land in
`<base>/v2_runs/<RUN_TAG>/` so each is one self-contained folder.

## Step 1 — submit ALL preloads in one wave

### NERSC LOA (3 variants):

```bash
ssh perlmutter
cd ~/desi_gpy_dla_detection && git pull

sbatch --export=ALL,VARIANT=no_dla_no_bal     slurm/train/preload_loa_only_nersc.sh
sbatch --export=ALL,VARIANT=no_hcd_with_bal   slurm/train/preload_loa_only_nersc.sh
sbatch --export=ALL,VARIANT=no_hcd_no_bal     slurm/train/preload_loa_only_nersc.sh
```

`-q regular -C cpu` walltime 6 h, MAX_SPECTRA=300,000 default.
CPU-regular queue typically lands within an hour.

### GreatLakes 2LPT (2 variants):

```bash
cd /home/mfho/desi_gpy_dla_detection && git pull

sbatch --export=ALL,VARIANT=loa0               slurm/greatlakes/preload_2lpt_only.sh
sbatch --export=ALL,VARIANT=loa124_nohcd_nobal slurm/greatlakes/preload_2lpt_only.sh
```

`-p standard` walltime 4 h, MAX_SPECTRA=300,000 default.
The standard partition is uncrowded.

## Step 2 — for each preload's RUN_TAG, submit train

The **preload's stdout** prints the RUN_TAG and the exact `sbatch`
command for the train step at the very end:

```
=== PRELOAD COMPLETE  loa_no_dla_no_bal_52234567
  RUN_DIR:   /pscratch/sd/j/jibancat/desi_gpy_dla_detection/v2_runs/loa_no_dla_no_bal_52234567

  NEXT STEP: train on this dataset
  sbatch --export=ALL,RUN_TAG=loa_no_dla_no_bal_52234567 slurm/train/train_only_nersc.sh
```

Just copy/paste. Then on each cluster:

### NERSC train:
```bash
sbatch --export=ALL,RUN_TAG=loa_no_dla_no_bal_<jobid>     slurm/train/train_only_nersc.sh
sbatch --export=ALL,RUN_TAG=loa_no_hcd_with_bal_<jobid>   slurm/train/train_only_nersc.sh
sbatch --export=ALL,RUN_TAG=loa_no_hcd_no_bal_<jobid>     slurm/train/train_only_nersc.sh
```

`-q debug -C gpu --gpus=1` 30-min walltime (200 epochs default).
A100 → ~10 min train per variant on a 300k trainset.

### GreatLakes train:
```bash
sbatch --export=ALL,RUN_TAG=2lpt_loa0_<jobid>               slurm/greatlakes/train_only_gpu.sh
sbatch --export=ALL,RUN_TAG=2lpt_loa124_nohcd_nobal_<jobid> slurm/greatlakes/train_only_gpu.sh
```

`-p gpu_mig40 --gpus=1` 2-h walltime (800 epochs default; auto-resume).
A100 MIG → ~30 min for 800 epochs on 50k spectra.

## Step 3 — Globus the run folders

After both NERSC training is done, transfer all 3 LOA runs to GreatLakes:

| | path |
|---|---|
| NERSC source | `/pscratch/sd/j/jibancat/desi_gpy_dla_detection/v2_runs/` |
| GreatLakes destination | `/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/` |

Each subfolder is one complete run (trainset + models + README + slurm logs).

## What's in each `v2_runs/<RUN_TAG>/` folder

```
README.md                        ← human-readable description (filter pipeline,
                                   sources, suggested train command)
dataset_metadata.json            ← machine-readable (used by train_only*.sh
                                   to auto-detect z_min/z_max from preload)
trainset.h5                      ← preload output, legacy schema
preload.slurm.log                ← preload SLURM stdout
                                                          (after train step:)
config.json                      ← TrainConfig snapshot
loss_history.json                ← per-epoch loss list
checkpoint_epoch_NNNN.pt         ← Adam state (resume support)
model_epoch_NNNN.h5              ← inference-ready models
train.slurm.log                  ← train SLURM stdout
```

## To extend training (debug → regular)

The trainer **auto-resumes** from the latest `checkpoint_epoch_NNNN.pt`.
If a `-q debug` run capped at 200 epochs and you want 800:

```bash
sbatch --export=ALL,RUN_TAG=loa_no_hcd_with_bal_<jobid>,NUM_EPOCHS=800 \
    slurm/train/submit_e2e_train_loa_nersc.sh   # the chained regular-queue submit
```

It'll skip preload (trainset.h5 already exists) and continue training from
epoch 200 → 800. (Alternatively, the `train_only_nersc.sh` script — but
debug walltime caps it at 30 min ⇒ stays at 200 epochs unless you switch
queue.)

## Diagnose any model after training

```bash
python examples/diagnose_trained_gp.py visualize \
    --model legacy_y3:/path/to/legacy/model_epoch_920.h5 \
    --model v2_<run>:<RUN_DIR>/model_epoch_<final>.h5 \
    --out-dir docs/notes/<comparison_tag>
```

Produces μ(λ), ω(λ), eigenspectra, correlation matrices for direct
comparison.

## Why split preload from train?

| | chained (old) | split (new) |
|---|---|---|
| Preload step | runs on GPU (idle) | runs on CPU |
| Queue wait | regular GPU (~3 days on NERSC) | regular CPU (~hours) |
| Train step | runs after preload | runs separately on GPU |
| Train queue wait | n/a | debug GPU (~minutes) |
| Total turnaround | ~3 days | **~hours** |
| GPU-hour cost | ~10× | **~1×** |

The chained submits remain in the repo (`submit_e2e_train_loa_nersc.sh`,
`preload_train_2lpt.sh`) for one-shot small-scale runs. For production use
the split workflow.
