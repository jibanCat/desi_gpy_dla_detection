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

## End-to-end pipeline overview

The pipeline is **three steps**, only the third is implemented in v2:

| step | code | input | output |
|---|---|---|---|
| 1. Preload spectra | `preload_spectra/desi-preload.py` | DESI spectra-16 fits | per-healpix HDF5 (`preloaded_*.h5`) |
| 2. Build trainset  | `preload_spectra/prepare_trainset.py` | per-healpix HDF5 | consolidated `gp_interp_trainset.h5` (rest-grid-interpolated) |
| 3. Train (v2 NEW)  | `train_gp.py` (this PR) | `gp_interp_trainset.h5` | `learnlogs_v2/.../model_epoch_NNNN.h5` |

Steps 1 and 2 are the existing legacy pipeline (in `preload_spectra/`)
and are **out of scope for v2** — they're separately tested. Step 3 is
where the speed/correctness fixes live.

The v2 trainer applies the train-time preprocessing (mask high-noise
pixels + de-forest with Turner+2024 τ₀/β + inverse-variance-weighted
centering) inside `dataset.py` so the inputs to `vectorized_nll` match
what the legacy `objective.objective` saw after
`GPModelTrainer.prepare_data` ran.

## NERSC submit (LOA / Y3 production)

**ALWAYS debug-submit first** — the `-q debug` queue is short and the
script does pre-flight import + path checks before training:

```bash
ssh perlmutter
cd ~/desi_gpy_dla_detection
git fetch && git checkout claude/training-and-lsf-validation
sbatch slurm_train/debug_train_gp_v2_nersc.sh
```

Once that succeeds (5 epochs on 5,000 spectra in ≤ 30 min), submit production:

```bash
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

---

## 2026-05-02 update — training-dynamics findings + recommended settings

A user-flagged investigation in May 2026 found that all 4 v2-trained
models (LOA + 2lpt + saclay) on the wide [851, 1421] Å rest grid were
**over-fit / mode-collapsed**: trace(ω²) shrunk to 0.2-3.4 % of
trace(K), top eigenvalue 5-500× the second, χ²/n_valid on the
training spectra averaged 0.02-1.0 (vs the well-calibrated target
of ~1). Consequences:

- Inference under-attributes residuals to noise → DLA absorption gets
  absorbed as a "small variance excursion in the dominant continuum
  mode" → DLA detection p_dla collapses for some configurations.
- Demonstrated on canonical TID 120046865: the LOA-trained-with-BAL
  model gave p_dla = 0.037 vs v1's 0.920 on a known DLA.

### Diagnosis (corrected after multiple iterations)

The first guess — that PCA NaN-fill choice (per-pixel vs per-row) was
the bug — turned out to be a red herring. Once the trainset.h5 fluxes
go through `load_preprocessed_h5`'s normalize → de-forest → center
pipeline, PCA init produces a healthy basis (eff_rank 1.78 on LOA,
3.11 on 2lpt). The per-row vs per-column NaN-fill choice barely
matters on properly normalized data.

**The actual cause is in training dynamics**: Adam optimizer +
`weight_decay=0` + cosine LR with `eta_min=1e-5`. Adam momentum
amplifies the dominant-eigenvector gradient, no L2 constrains M
growth, and the LR anneals before the basis can diversify.

The legacy v1 trainer used L-BFGS (no momentum, line-search step
sizes), which avoids this trap. v2 was switched to Adam for batch
vectorization; the cost of that switch is the need for explicit
regularization.

### Recommended training settings (post-2026-05-02)

For all NEW v2 training, prefer:

```
--weight-decay 1e-6          # constrains M growth
--scheduler none             # constant LR, no premature anneal
--num-epochs 1500            # production: more epochs help
--min-valid-pixels-lyman 200 # v1-equivalent quality filter
--norm-min-lambda 1310       # Garnett+2017 (forced by v2 grid coverage)
--norm-max-lambda 1325
```

The CLI defaults preserve backward compatibility (cosine LR,
weight_decay=0) so existing runs reproduce. New runs should override.

### Validation workflow (mandatory before model promotion)

After every training run, run all four diagnostics. If any fails the
model is **not production-ready**:

1. **Calibration check** —
   `examples/check_v2_model_calibration.py`. Verifies χ²/n_valid ≈ 1
   on a 500-spectrum sample of the trainset. Verdict printed.

2. **Per-model μ + ω + K viz** —
   `examples/plot_v2_model_diagnostics.py`. Look at the trace ratio
   (ω² should be O(0.1-1.0) of trace(K), not 0.001) and the
   correlation matrix (should show emission-line off-diagonal
   structure, not rank-1 sharp blocks).

3. **PCA init K viz** —
   `examples/plot_pca_init_K.py`. Run on the same trainset. If init
   eff_rank is healthy (≥ 1.5) but trained eff_rank is rank-1, you
   have a training-dynamics regression — re-check weight_decay +
   scheduler.

4. **Canonical TID test** —
   `examples/compare_v2_models_canonical.py` on TID 120046865 (2lpt
   mock-0). Should detect a DLA at z=2.77, log NHI≈21.6 with
   p_dla > 0.5. If it misses, the model is unsuitable for production.

Full evidence trail in:
- `docs/notes/2026-05-02_v2_calibration_root_cause.md`
- `docs/notes/2026-05-02_v2_trainer_calibration_finding.md`
- `docs/notes/2026-05-02_v2_canonical_tid_comparison.md`

### v1-style quality filter (added 2026-05-02)

`load_preprocessed_h5` now drops:
- spectra with NaN normalization median (v1 `preload_qsos.m` bit 2)
- spectra with < `min_valid_pixels_lyman` valid pixels in
  [`lyman_min_lambda`, `lyman_max_lambda`] = [911, 1216] Å rest
  (v1 `preload_qsos.m` bit 3)

Default `min_valid_pixels_lyman=200` is conservative (~10% of the
2030-pixel range at dlambda=0.15). Set to 0 to disable.

### What's still being tested

SLURM job 49227683 (queued at the time of writing): a 1500-epoch
retrain with the recommended settings on `loa_no_dla_no_bal` trainset
+ z [2.5, 4.25]. After it finishes the calibration verdict will tell
us whether weight_decay + scheduler=none alone is sufficient or
whether deeper trainer changes are needed.

For onboarding new contributors: **don't promote a v2 trained model
to production without running the four-step validation workflow
above.** The parity tests (`tests/test_objective_v2_parity.py`)
verify per-spectrum NLL math but do NOT catch training-dynamics
issues that emerge only at 100s-1000s of epochs on a real trainset.
