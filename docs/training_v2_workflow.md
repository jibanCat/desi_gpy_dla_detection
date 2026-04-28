# Streamlined GP training (v2)

End-to-end workflow for retraining the QSO emission GP, replacing the
legacy ``desi_learn_qsos_model.py`` + ``learn_qso_model.Trainer`` stack.
The legacy code remains in place; v2 lives under
``gpy_dla_detection/training/`` and is a drop-in replacement at the
data-tensor and HDF5-output level.

## Why v2

Layer 3 profiling on the legacy trainer (n_pix=600, k=30, 128 synthetic
spectra) showed:

| metric | legacy | v2 (CPU bs=32) | v2 (CPU bs=128) |
|---|---:|---:|---:|
| per-epoch wall | 36 s | 2.5 s | 3.6 s |
| per-spectrum   | 276 ms | 19.7 ms | 28.4 ms |
| **CPU speedup** | 1× | **14×** | **9.9×** |

Two underlying changes:

1. **Vectorized NLL across the batch** — one ``torch.linalg.cholesky`` /
   ``torch.bmm`` / ``torch.einsum`` call per epoch step instead of a
   Python ``for`` loop over each spectrum.
2. **Autograd backward** — replaces the manual gradient accumulation in
   ``objective.objective``. Mathematically equivalent (verified by
   ``tests/test_objective_v2_parity.py``), and side-fixes the legacy
   ``dlog_beta`` approximation (see "Caveat" below).

GPU speedup is expected to be substantially larger than CPU because the
Python-loop overhead disappears; submit
``slurm/greatlakes/profile_training_gpu.sh`` to measure it.

## Caveat: legacy ``dlog_beta`` approximation

The legacy analytical gradient formula in
``gpy_dla_detection.objective.spectrum_loss`` line 188:

```python
da_beta = da_tau0 * torch.log(lya_1pz) * beta * indicator
```

uses ``log(lya_1pz)`` for **every** Lyman line, but each Lyman line
should use its own ``log(lyman_1pz_i)``. The two differ by
``log(λ_α / λ_i) ≈ 0.17–0.29``. The DR16Q-public MATLAB reference
``spectrum_loss.m`` line 94 carries the same approximation
(Layer 4 parity passes byte-stably), so the production GP has been
trained with this approximate gradient.

v2 uses ``loss.backward()`` over the actual NLL, so its ``dlog_beta``
is the correct full derivative. Implications:

- The **loss** is identical (parity-tested to 1e-9).
- After enough Adam steps, v2 converges to a slightly different ``β``
  than legacy.
- When loading a v2-trained ``model_epoch_<N>.h5`` with the legacy
  inference code (``run_bayes_select.DLAHolder``), no changes are
  needed — the H5 schema is identical.

If you want a strict legacy-compatible β optimisation for backwards
comparability, use the legacy trainer for that single training run.
For all new training, prefer v2.

## Layout

```
gpy_dla_detection/training/
├── __init__.py
├── dataset.py        # load_preprocessed_h5() — reads gp_interp_trainset.h5
├── model_v2.py       # GPModelV2 — pure parameter container
├── objective_v2.py   # vectorized_nll() — batched NLL + autograd-friendly
└── trainer_v2.py     # train() + checkpointing/resume

train_gp.py           # top-level CLI
slurm_train/submit_train_gp_v2_loa_nersc.sh
slurm/greatlakes/train_gp_v2_2lpt.sh
```

## NERSC submit (LOA / Y3 production)

```bash
ssh perlmutter
cd /global/u2/j/jibancat/desi_gpy_dla_detection
git checkout claude/training-and-lsf-validation
sbatch slurm_train/submit_train_gp_v2_loa_nersc.sh
```

The script defaults match the production config in
``slurm_train/submit_train_gp_loa_full.sh``:

| flag | default | matches legacy |
|---|---|---|
| ``--num-epochs`` | 800 | yes |
| ``--batch-size`` | 12,500 | half of legacy 205,516 — keeps memory bounded |
| ``--learning-rate`` | 5e-3 | yes |
| ``--max-spectra`` | 300,000 | yes |
| ``--num-pca-components`` | 30 | yes |
| ``--num-forest-lines`` | 3 | yes |
| ``--scheduler`` | cosine | yes |

Outputs:

```
$OUTPUT_DIR/
├── config.json                    # snapshot of TrainConfig
├── loss_history.json              # per-epoch loss
├── checkpoint_epoch_NNNN.pt       # full resume state, every save_every (default 10)
└── model_epoch_NNNN.h5            # compact H5 for inference, same schema as legacy
```

## GreatLakes submit (development / 2LPT)

```bash
sbatch slurm/greatlakes/train_gp_v2_2lpt.sh
```

Uses ``-A cavestru0 -p gpu --gpus=1``. Same script as NERSC, just
different SLURM directives + ``module load cuda/12.4.0`` + conda env
activation.

## Resume on preemption

The trainer auto-resumes from the latest ``checkpoint_epoch_NNNN.pt`` in
``$OUTPUT_DIR``. Just re-submit the same sbatch script — no extra args
needed.

## Validation tests

Run before committing any change to ``training/``:

```bash
python -m pytest tests/test_objective_math.py     # Layer 1, 11 tests, ~2 s
python -m pytest tests/test_objective_v2_parity.py # Parity, 12 tests, ~3 s
```

For Layer 4 (MATLAB parity, requires ``module load matlab``):

```bash
python tests/parity/matlab_parity_check.py
```

## Speed expectations

Empirical CPU numbers (login node, small batch):

```
batch_size=32  n_spectra=128  n_pix=600  k=30
legacy:  36.2 s/epoch  (276 ms/spectrum)
v2:       2.5 s/epoch  (19.7 ms/spectrum)  → 14.4x

batch_size=128 n_spectra=256
legacy:  71.9 s/epoch  (281 ms/spectrum)
v2:       7.3 s/epoch  (28.4 ms/spectrum)  → 9.9x
```

For a full training run (300k spectra, 800 epochs):

| machine | legacy estimate | v2 estimate |
|---|---|---|
| CPU (logn) | ~24 days | ~2 days |
| GPU (Perlmutter A100) | unknown | TBD via the GPU profile |

GPU benchmark pending — ``sbatch slurm/greatlakes/profile_training_gpu.sh``
takes ~10 min wall.

## Architecture decisions documented

- **fp32**, not fp64 — matches the legacy production trainer. Layer 1
  / Layer 4 use fp64 for tight tolerances; production training uses fp32
  to fit memory and exploit GPU mixed-precision tensor cores.
- **No DataParallel / DDP** — the per-epoch wall is small enough at
  batch_size=12,500 that single-GPU is fine. Multi-GPU adds a non-trivial
  amount of code complexity; defer until needed.
- **No L-BFGS option** — Adam is fine; L-BFGS adds bookkeeping.
- **Save every 10 epochs** instead of every epoch — Layer 3 noted h5+pt
  + covariance plot save at every epoch in the legacy trainer was a
  noticeable overhead. Configurable via ``--save-every``.
- **No covariance plotting in the trainer** — that's a separate
  diagnostic; do it from the saved h5 files offline.
