# Layer 3 — GP training speed profile

Empirical profile of `gpy_dla_detection.objective.objective` (the per-batch
NLL accumulator used by `Trainer`) on synthetic data sized like a real DESI
Y3 training batch (n_pix=600, k=30, num_forest_lines=3).

## How to run

CPU (laptop / login node):

```bash
python tests/profile/profile_training.py \
   --device cpu --num-spectra 128 --epochs 3 --batch-size 32
```

GPU (GreatLakes compute):

```bash
sbatch slurm/greatlakes/profile_training_gpu.sh
```

Output goes to `tests/profile/results/profile_<tag>.txt` (table) and
`tests/profile/results/trace_<tag>.json` (Chrome trace).

## What I found on CPU (128 spectra × 3 epochs, login node)

**Wall time**: 36 s per epoch → **0.28 s per spectrum per epoch**.
**Implication for production**: 300k spectra × 800 epochs ≈ **24 days on CPU**.

### Top ops by self CPU time (3 epochs, 384 total `spectrum_loss` calls)

| % | self CPU | op | calls | per-call | what it is |
|---:|---:|---|---:|---:|---|
| 49.3 % | 53.6 s | `aten::mm` | 2688 | 19.9 ms | matrix mults inside `spectrum_loss` (≈7 per spectrum) |
| 16.6 % | 18.0 s | `aten::index` | 2688 | 6.7 ms | boolean indexing `M[valid_mask, :]` — pure Python-loop overhead |
| 16.2 % | 17.6 s | `aten::linalg_solve_triangular` | 768 | 23.0 ms | 2 per spectrum (forward + back subs) |
|  8.4 % |  9.1 s | `aten::_index_put_impl_` | 768 | 11.9 ms | `dM_accum[valid_mask, :] += dM` |
|  8.0 % |  8.7 s | `aten::tril` | 384 | 22.7 ms | inside `linalg_cholesky_ex` |

### Diagnosis

The work is dominated by **matrix multiplications and triangular solves on
(600 × 30) and (30 × 30) tensors**, evaluated **one spectrum at a time** in a
Python loop inside `objective()`. Two structural problems:

1. **No batching across spectra.** PyTorch supports
   `torch.linalg.cholesky(B)` and `torch.bmm(...)` on stacked tensors of
   shape `(B, n, k)` and `(B, k, k)`. The Cholesky of 30×30 matrices is so
   cheap that BLAS overhead dominates per-spectrum; one batched call should
   be ~5–10× faster on CPU and ~50–100× faster on GPU.

2. **Per-spectrum boolean indexing for NaN masking.** `M[valid_mask, :]`
   re-allocates a new tensor every call. With 5 % NaN pixels per spectrum
   the alternative is to pad the kept-flux to a fixed length and mask via
   `torch.where(valid_mask, contribution, 0)` inside the per-pixel ops.
   This eliminates 16.6 % + 8.4 % = 25 % of CPU time outright.

3. **Manual gradient accumulation** (`dM_accum[valid_mask, :] += dM`)
   bypasses autograd. Layer 1 confirmed analytical gradients match
   autograd to 1e-9, so the math is correct; but we're paying for the
   indexing twice (once in `spectrum_loss`, once in `objective`). A clean
   `loss.backward()` on a fully-vectorized forward pass eliminates this.

### Per-epoch wall components (rough split)

| component | wall fraction (CPU) | comment |
|---|---:|---|
| Pure linear algebra (mm + tril + solve) | ~73 % | unavoidable but **batch it** |
| Python-loop / indexing overhead | ~25 % | **eliminable** via vectorization |
| Optimizer step / misc | ~2 % | not a bottleneck |

## Recommended speedups

1. **Vectorize `spectrum_loss` across spectra** (high payoff, medium effort).
   - Pad valid pixels with zeros, carry a `valid_mask` (B, n_pix).
   - Replace per-spectrum Cholesky / triangular solves with `torch.linalg.cholesky`
     / `torch.linalg.solve_triangular` on `(B, k, k)` and `(B, k, n_pix)` tensors.
   - Use `torch.bmm` for matrix products.
   - Use `loss.backward()` instead of manual gradient accumulation. Cleaner;
     the analytic Layer 1 gradients don't disappear — they're rederived by
     autograd, identical to 1e-9.
   - Estimated CPU speedup: 5–10×; GPU speedup: 50–100×.

2. **Move all per-batch tensors to GPU** (high payoff, low effort once #1 done).
   Currently the `Trainer` class tries to use GPU but the per-spectrum
   loop in `objective` still incurs Python overhead that GPU doesn't help
   with. Once `objective` is vectorized, GPU should dominate.

3. **Avoid per-batch CPU sync** (small payoff, low effort).
   The current `Trainer.train()` does `loss.detach().cpu().item()` every
   batch, plus `print(f"Max gradient norm: ...")`. Each forces a host-device
   sync. Move both to per-epoch.

4. **Save model less often** (small payoff, low effort).
   The current code saves `.pt` + `.h5` + plots a covariance matrix every
   epoch. The h5 + .pt save block alone is ~0.5–1 s per epoch on this
   setup; covariance-plot save is ~3–5 s. Save every 5–10 epochs instead.

## Why batch sizes matter

Production uses `batch_size=205516` (entire dataset in one go) with Adam.
That's a lot of memory but eliminates the per-batch Python loop overhead
across the dataset. The 32-batch profile here is the worst case for Python
overhead. If we ran at `batch_size=205516`, the indexing overhead would
amortize but each `aten::mm` call would balloon — and we'd hit OOM on most
GPUs without vectorization.

The right answer is **vectorize first, then pick a sensible batch size for
the available GPU memory**.

## Open question for the GPU profile

We don't yet know how much of the 49 % `aten::mm` time stays the same on
GPU. Small matrices (30×30) often have BLAS-launch overhead that GPU
amortizes only at larger batches. Run `slurm/greatlakes/profile_training_gpu.sh`
to find out — should take ~10 minutes wall.
