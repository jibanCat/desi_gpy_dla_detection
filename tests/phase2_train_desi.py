"""DESI Phase 2 trainer — mirrors tests/phase2_train_dr16.py for DESI v2 preloads.

Key differences from `phase2_train_dr16.py`:

  - **Data source**: reads v2 preprocessed `trainset.h5`
    (`gpy_dla_detection.training.dataset.load_preprocessed_h5` schema)
    instead of the MATLAB DR16 `preloaded_qsos.mat`.
  - **GPU support**: `--device cuda` (default if available) moves tensors
    + parameters to GPU. trainer_v2 hits ~3.2 s/iter on A100 80GB for
    118k×3801; phase2_train_dr16 (CPU) hits ~144 s/iter for 89k×2281.
  - **Priors**: Turner+2024 (`τ_0=0.00246, β=3.62`) instead of BOSS DR12Q.

What's the SAME:
  - PCA init for M (with `random_state=0` for reproducibility)
  - Hand-coded analytic gradients via `spectrum_loss_batch`
    (`gpy_dla_detection.training_v3.objective_vectorized`)
  - Adam optimizer, checkpoint+resume, signal handlers
  - Output `.h5` in DESI learned-model schema (production-loadable by
    `null_gp.NullGPMAT`)

This trainer DOES NOT use the broken `gpy_dla_detection.training.trainer_v2`
path (randn-init M + autograd loss; see `docs/training_overview.md` and
`project_corrected_retrains_regression_2026_05_06.md`).

Usage:
    # smoke (5k spectra, 50 iter, GPU)
    python tests/phase2_train_desi.py \\
        --preload /nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection/v2_runs/2lpt_loa0_wide_v2_1778186324/trainset.h5 \\
        --max-spectra 5000 --n-iters 50 \\
        --out-dir /tmp/desi_smoke

    # production (full 300k spectra, 1500 iter, GPU)
    python tests/phase2_train_desi.py \\
        --preload <PATH>/trainset.h5 \\
        --n-iters 1500 \\
        --out-dir /scratch/.../desi_production
"""
from __future__ import annotations

import os as _os
# Don't pin OMP_NUM_THREADS=1 by default — the GPU path doesn't need
# the per-spectrum loop's thread-storm protection. CPU runs may want
# `OMP_NUM_THREADS=4 python tests/phase2_train_desi.py ...` upstream.

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

# Reuse the corrected DR16 trainer's PCA init function (it's
# k-agnostic — pass the desired k explicitly).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.phase2_train_dr16 import _pca_init

from gpy_dla_detection.training_v3.objective_vectorized import spectrum_loss_batch
from gpy_dla_detection.training.dataset import load_preprocessed_h5

DTYPE = torch.float32  # GPU memory budget — match trainer_v2's dtype
NUM_FOREST_LINES = 31  # Match phase2_train_dr16.py:66 (deep Lyman series).
                       # Used for BOTH de-forest preprocessing AND spectrum_loss
                       # GP forward model — must be consistent. Different from
                       # trainer_v2's default of 3.
K_DESI = 30  # DESI Y3 production convention (matches trainer_v2 corrected runs).
             # Different from DR16's k=20 — don't reuse the DR16 K constant.

# Turner+2024 priors (Y3 production). Use strict Turner sigmas to
# match v1 production (`gpy_dla_detection/objective.py:65,67`). The
# audit (docs/notes/2026-05-12_training_pipeline_audit_vs_matlab/)
# identified the earlier (0.00064, 0.074) as BOSS DR12Q sigmas mixed
# with Turner means — too loose, allowed 2lpt mocks to drift to
# τ_0~0.0006, β~1.3 (well below physical Turner mean). Strict Turner
# σ pulls the trained scalars back toward the published mean.
TAU_0_PRIOR_MU = 0.00246
TAU_0_PRIOR_SIGMA = 0.00014   # Turner+2024 (was 0.00064 = BOSS DR12Q)
BETA_PRIOR_MU = 3.62
BETA_PRIOR_SIGMA = 0.04       # Turner+2024 (was 0.074 = BOSS DR12Q)

# Initial point for hyperparameters
INITIAL_C_0 = 0.1
INITIAL_TAU_0 = TAU_0_PRIOR_MU
INITIAL_BETA = TAU_0_PRIOR_MU * 0 + 3.62  # avoid stale-cache typo
INITIAL_BETA = 3.62

_RUNTIME = dict(checkpoint_dir=None, save_now=False)


def _train(centered, nv, lya_1pzs, valid_masks, z_qsos, mu, M_init, log_omega_init,
           num_forest_lines, n_iters, lr, device,
           checkpoint_every=5, resume_path=None,
           max_walltime_sec=None, chunk_size=12500,
           rest_wavelengths=None):
    """Adam loop on `device` (CPU or CUDA) with hand-coded gradients via
    `spectrum_loss_batch`. Mirrors `phase2_train_dr16._train` exactly except:
      - tensors moved to `device`
      - DESI Y3 priors (Turner+2024)
      - default chunk_size=12500 (matches trainer_v2)
    Always uses the vectorized loss path (no per-spectrum fallback — the
    DESI scale (300k×5662) makes the per-spectrum loop infeasible).
    """
    from gpy_dla_detection.voigt import (
        transition_wavelengths as TW, oscillator_strengths as OS)
    TW_t = torch.tensor(np.asarray(TW), dtype=DTYPE, device=device)
    OS_t = torch.tensor(np.asarray(OS), dtype=DTYPE, device=device)

    M = nn.Parameter(torch.tensor(M_init, dtype=DTYPE, device=device))
    log_omega = nn.Parameter(torch.tensor(log_omega_init, dtype=DTYPE, device=device))
    log_c_0 = nn.Parameter(torch.tensor(np.log(INITIAL_C_0), dtype=DTYPE, device=device))
    log_tau_0 = nn.Parameter(torch.tensor(np.log(INITIAL_TAU_0), dtype=DTYPE, device=device))
    log_beta = nn.Parameter(torch.tensor(np.log(INITIAL_BETA), dtype=DTYPE, device=device))

    # Sanitize NaN at invalid pixels once. Both fluxes and noise variance
    # are masked-where-invalid by load_preprocessed_h5 (flux→0, nv→NaN);
    # the vectorized path needs finite values everywhere so cholesky
    # never sees NaN.
    #
    # IMPORTANT: keep data tensors on CPU (pinned if GPU). Transfer one
    # chunk per Adam iter inside the loop. Loading all 300k × 5662 × 4B
    # to GPU upfront wastes 7+ GB and competes with the per-chunk
    # intermediates (which can hit 5+ GB at chunk=5k×5662×k=30). This
    # mirrors trainer_v2's per-batch CPU→GPU transfer pattern.
    pin = (device.type == "cuda")
    # torch.tensor(np_array, pin_memory=...) doesn't accept pin_memory
    # for numpy inputs. Construct via from_numpy().to(dtype).pin_memory().
    # IN-PLACE mask operations to avoid duplicating arrays in host RAM.
    def _mk(arr_np, dtype):
        t = torch.from_numpy(np.ascontiguousarray(arr_np)).to(dtype)
        return t.pin_memory() if pin else t
    # In-place sanitize: write 0 / 1 directly into centered / nv at invalid
    # pixels (no temporary copy from np.where).
    centered[~valid_masks] = 0.0
    nv[~valid_masks] = 1.0
    centered_cpu = _mk(centered, DTYPE)
    nv_cpu = _mk(nv, DTYPE)
    lya_1pzs_cpu = _mk(lya_1pzs, DTYPE)
    valid_cpu = _mk(valid_masks.astype(bool), torch.bool)
    zqso_1pz_cpu = _mk((z_qsos + 1.0).astype(np.float32), DTYPE)
    n = centered.shape[0]

    optimizer = torch.optim.Adam([M, log_omega, log_c_0, log_tau_0, log_beta], lr=lr)
    history = dict(loss=[], log_c_0=[], log_tau_0=[], log_beta=[])
    start_iter = 0
    if resume_path is not None:
        rp = Path(resume_path)
        print(f"\n[resume] loading checkpoint from {rp}")
        ckpt = torch.load(rp, map_location=device, weights_only=False)
        with torch.no_grad():
            M.copy_(ckpt["M"].to(device))
            log_omega.copy_(ckpt["log_omega"].to(device))
            log_c_0.copy_(ckpt["log_c_0"].to(device))
            log_tau_0.copy_(ckpt["log_tau_0"].to(device))
            log_beta.copy_(ckpt["log_beta"].to(device))
        optimizer.load_state_dict(ckpt["optim_state"])
        history = {k: list(v) for k, v in ckpt["history"].items()}
        start_iter = int(ckpt["iter_completed"]) + 1
        print(f"[resume] resuming at iter={start_iter} "
              f"(prior loss={history['loss'][-1]:.4f})")

    dM_accum = torch.zeros_like(M)
    dlog_omega_accum = torch.zeros_like(log_omega)

    ckpt_dir = _RUNTIME["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(it_done, tag="iter"):
        cpath = ckpt_dir / f"phase2_desi_checkpoint_{tag}{it_done:04d}.pt"
        # Move to CPU before saving so the checkpoint is portable
        # (CUDA-saved tensors require the same device on load).
        # Include rest_wavelengths so the .pt → .h5 converter doesn't
        # need to re-derive the rest grid from the preload file.
        torch.save(dict(
            M=M.detach().cpu().clone(),
            log_omega=log_omega.detach().cpu().clone(),
            log_c_0=log_c_0.detach().cpu().clone(),
            log_tau_0=log_tau_0.detach().cpu().clone(),
            log_beta=log_beta.detach().cpu().clone(),
            optim_state=optimizer.state_dict(),
            iter_completed=int(it_done),
            history=history,
            mu=mu,
            rest_wavelengths=(np.asarray(rest_wavelengths)
                              if rest_wavelengths is not None else None),
            num_forest_lines=int(num_forest_lines),
        ), cpath)
        print(f"[checkpoint] saved {cpath} (iter {it_done})")
        return cpath

    def _on_signal(signum, _frame):
        print(f"\n[signal] caught {signum}, requesting graceful save at next iter boundary")
        _RUNTIME["save_now"] = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    train_t0 = time.time()
    print(f"\n=== Training: {n} spectra, {n_iters} iter, lr={lr}, "
          f"device={device}, chunk_size={chunk_size}, "
          f"start_iter={start_iter}, checkpoint_every={checkpoint_every} ===")
    it = start_iter - 1
    for it in range(start_iter, n_iters):
        t0 = time.time()
        optimizer.zero_grad()
        omega2 = torch.exp(2 * log_omega)
        c_0 = torch.exp(log_c_0)
        tau_0 = torch.exp(log_tau_0)
        beta = torch.exp(log_beta)

        total = torch.zeros((), dtype=DTYPE, device=device)
        dM_accum.zero_()
        dlog_omega_accum.zero_()
        dlog_c_0_acc = torch.zeros((), dtype=DTYPE, device=device)
        dlog_tau_0_acc = torch.zeros((), dtype=DTYPE, device=device)
        dlog_beta_acc = torch.zeros((), dtype=DTYPE, device=device)

        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            # Per-chunk CPU→GPU transfer (non_blocking=pin enables async
            # copy when src is pinned-memory CPU). Free chunks at end via
            # del + (optionally) torch.cuda.empty_cache to keep peak low.
            cb = centered_cpu[s:e].to(device, non_blocking=pin)
            lb = lya_1pzs_cpu[s:e].to(device, non_blocking=pin)
            nb = nv_cpu[s:e].to(device, non_blocking=pin)
            vb = valid_cpu[s:e].to(device, non_blocking=pin)
            zb = zqso_1pz_cpu[s:e].to(device, non_blocking=pin)
            nlp_c, dM_c, dlogw_c, dc0_c, dt0_c, db_c = spectrum_loss_batch(
                cb, lb, nb, vb,
                M, omega2, c_0, tau_0, beta,
                num_forest_lines, TW_t, OS_t,
                zb,
            )
            total = total + nlp_c.detach()
            dM_accum.add_(dM_c.detach())
            dlog_omega_accum.add_(dlogw_c.detach())
            dlog_c_0_acc = dlog_c_0_acc + dc0_c.detach()
            dlog_tau_0_acc = dlog_tau_0_acc + dt0_c.detach()
            dlog_beta_acc = dlog_beta_acc + db_c.detach()
            del cb, lb, nb, vb, zb, nlp_c, dM_c, dlogw_c, dc0_c, dt0_c, db_c

        # Turner+2024 priors on (τ_0, β). dlog_τ_0 += τ_0 (τ_0 - μ)/σ²
        # follows from chain rule on log-parameter prior.
        dlog_tau_0_acc = dlog_tau_0_acc + tau_0 * (tau_0 - TAU_0_PRIOR_MU) / TAU_0_PRIOR_SIGMA**2
        dlog_beta_acc = dlog_beta_acc + beta * (beta - BETA_PRIOR_MU) / BETA_PRIOR_SIGMA**2

        with torch.no_grad():
            M.grad = dM_accum.clone()
            log_omega.grad = dlog_omega_accum.clone()
            log_c_0.grad = dlog_c_0_acc.clone()
            log_tau_0.grad = dlog_tau_0_acc.clone()
            log_beta.grad = dlog_beta_acc.clone()

        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        history["loss"].append(float(total.cpu()))
        history["log_c_0"].append(float(log_c_0.detach().cpu()))
        history["log_tau_0"].append(float(log_tau_0.detach().cpu()))
        history["log_beta"].append(float(log_beta.detach().cpu()))
        if it < 3 or it % 5 == 0 or it == n_iters - 1:
            print(f"  it={it:>4d}  loss={float(total.cpu()):>14.4f}  "
                  f"τ_0={float(tau_0.detach().cpu()):.6f}  "
                  f"β={float(beta.detach().cpu()):.4f}  "
                  f"c_0={float(c_0.detach().cpu()):.6f}  ({dt:.2f}s/iter)")

        if checkpoint_every and ((it + 1) % checkpoint_every == 0):
            _save_checkpoint(it)

        if max_walltime_sec is not None and (time.time() - train_t0) > max_walltime_sec:
            print(f"[walltime] elapsed > {max_walltime_sec}s; saving and exiting at iter={it}")
            _save_checkpoint(it, tag="walltime_exit_iter")
            break

        if _RUNTIME["save_now"]:
            _save_checkpoint(it, tag="signal_exit_iter")
            break

    _save_checkpoint(it, tag="final_iter")

    return dict(M=M.detach().cpu().numpy().astype(np.float64),
                mu=mu,
                log_omega=log_omega.detach().cpu().numpy().astype(np.float64),
                log_c_0=float(log_c_0.detach().cpu()),
                log_tau_0=float(log_tau_0.detach().cpu()),
                log_beta=float(log_beta.detach().cpu()),
                c_0=float(torch.exp(log_c_0.detach()).cpu()),
                tau_0=float(torch.exp(log_tau_0.detach()).cpu()),
                beta=float(torch.exp(log_beta.detach()).cpu()),
                history=history)


def _save_readme(out_dir: Path, result: dict, rest: np.ndarray,
                 n_spectra: int, n_iters: int, lr: float, k: int,
                 chunk_size: int, device: str, preload: Path) -> Path:
    """Write a README.md in the output dir documenting the trained model.

    Includes: training config, endpoint scalars, DESI schema reference,
    production-loader compatibility notes. Exists alongside phase2_result.h5
    so anyone who picks up the model file knows what it is and how it was
    trained without needing to dig into the source.
    """
    readme = out_dir / "README.md"
    n_pix = len(rest)
    rest_min = float(rest[0]); rest_max = float(rest[-1])
    d_lambda = float(rest[1] - rest[0])
    final_loss = result["history"]["loss"][-1] if result["history"]["loss"] else float("nan")

    body = f"""# Phase 2 DESI trained GP — model card

This directory contains a GP model trained by `tests/phase2_train_desi.py`
(PR #6 corrected trainer; PCA init + hand-coded gradient via
`gpy_dla_detection/training_v3/objective_vectorized.spectrum_loss_batch`).

## Files

| File | Purpose |
|---|---|
| `phase2_result.h5` | **Learned model** in DESI schema. Production-loadable by `gpy_dla_detection.null_gp.NullGPMAT(learned_file=...)`. |
| `phase2_result.npz` | Training-history record (loss + log_*_history per iter, n_spectra, n_iters, lr). Not loaded by the inference pipeline. |
| `README.md` | This file. |

## Training config

| Parameter | Value | Source |
|---|---|---|
| preload source | `{preload}` | `--preload` |
| n_spectra (after filter) | {n_spectra} | post z/SNR/cap |
| n_pix (rest) | {n_pix} | preload |
| rest grid | [{rest_min:.2f}, {rest_max:.2f}] Å, dλ={d_lambda:.4f} | preload |
| k (PCA components) | {k} | `--k` |
| n_iters (Adam) | {n_iters} | `--n-iters` |
| lr | {lr} | `--lr` |
| chunk_size (vec) | {chunk_size} | `--chunk-size` |
| device | {device} | `--device` |
| optimizer | `torch.optim.Adam` | trainer |
| loss path | `spectrum_loss_batch` (training_v3, vectorized, hand-coded grad) | trainer |
| τ_0 prior | N({TAU_0_PRIOR_MU}, {TAU_0_PRIOR_SIGMA}²) — Turner+2024 | trainer |
| β prior | N({BETA_PRIOR_MU}, {BETA_PRIOR_SIGMA}²) — Turner+2024 | trainer |
| de-forest | τ_0={TAU_0_PRIOR_MU}, β={BETA_PRIOR_MU}, num_lines={NUM_FOREST_LINES} | dataset.py |
| normalize | per-spectrum median in [1310, 1325] Å rest (Garnett+2017) | dataset.py |
| max_noise_variance | 9.0 | dataset.py |
| PCA init `random_state` | 0 (pinned, ac7bed8) | `_pca_init` |

## Endpoint scalars

| Parameter | Value |
|---|---:|
| c_0 | {result['c_0']:.6f} |
| τ_0 | {result['tau_0']:.6f} |
| β | {result['beta']:.4f} |
| log p(D \| Adam endpoint) | {final_loss:.4f} |

## DESI .h5 schema (`phase2_result.h5`)

The DESI inference loader (`null_gp.NullGPMAT.__init__`, `null_gp.py:440-503`)
reads these keys at top level:

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `M` | (n_pix, k) | float64 | GP low-rank basis. `K = M·M^T + diag(omega²)` is the GP prior covariance. |
| `mu` | (n_pix,) | float64 | GP mean function (per-pixel inverse-variance-weighted training mean). |
| `log_omega` | (n_pix,) | float64 | log of per-pixel variance addition. ω² adds to the noise diagonal. |
| `log_c_0` | scalar | float64 | log of mean-flux scale c_0. Reconstructed flux = c_0 × A_lyα(z) × (μ + Mη). |
| `log_tau_0` | scalar | float64 | log Lyα optical-depth normalization. Final τ_eff(z) = τ_0 × (1+z)^β. |
| `log_beta` | scalar | float64 | log Lyα optical-depth power-law index. |
| `rest_wavelengths` | (n_pix,) | float64 | Rest-wavelength grid (Å). |
| `max_noise_variance` | scalar | float64 | Pixel-mask threshold used during preprocessing. |
| `normalization_min_lambda` | scalar | float64 | Per-spectrum normalization band (Å rest), lower edge. |
| `normalization_max_lambda` | scalar | float64 | Per-spectrum normalization band (Å rest), upper edge. |

This schema matches `learnlogs/model_epoch_*.h5` from production runs.
Detection mode is set automatically by `NullGPMAT` based on
`log_tau_0.ndim == 0` (DESI = scalar; SDSS = (1,1)).

## How to load this model for inference

```python
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.set_parameters import Parameters

# Build Parameters with the inference-time settings YOU want. The
# trained model's `normalization_min/max_lambda` will be auto-applied
# by NullGPMAT's loader (overrides params.normalization in place).
# k and rest range MUST match the trained model — k={k}, rest=[{rest_min:.0f}, {rest_max:.0f}] Å.
params = Parameters(
    k={k},
    min_lambda={rest_min:.2f},
    max_lambda={rest_max:.2f},
    dlambda={d_lambda:.4f},
    max_noise_variance=9.0,
    num_lines=3,
    num_forest_lines={NUM_FOREST_LINES},
    # ... other params per your inference setup
)
prior = ...  # PriorCatalog instance
gp = NullGPMAT(params, prior, learned_file="phase2_result.h5")
```

## Training provenance

Trained on commit `{_git_head_sha()}`.
"""
    readme.write_text(body)
    return readme


def _git_head_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_h5(out_path, result, rest, n_spectra, n_iters, lr, vectorized=1,
             initial_M=None, initial_log_omega=None,
             initial_log_c_0=None, initial_log_tau_0=None, initial_log_beta=None):
    """Write learned model in DESI schema (production-loadable).

    Schema matches v1 production (`learnlogs/model_epoch_*.h5`) where it
    overlaps, plus the DESI-loader-required normalization scalars:

      M, mu, log_omega                               trained kernel
      log_c_0, log_tau_0, log_beta                   trained scalars
      rest_wavelengths, max_noise_variance           required by loader
      normalization_min_lambda, normalization_max_lambda  v2-only,
                                                     auto-applied by NullGPMAT

    Plus v1-compatible provenance + history (so plot_corr_dr16_comparison
    and other v1-era scripts work directly on our outputs):

      initial_M, initial_log_omega                   PCA init / data-driven
      initial_log_c_0, initial_log_tau_0, initial_log_beta   initial scalars
      loss_history, log_c_0_history, log_tau_0_history, log_beta_history
    """
    with h5py.File(out_path, "w") as f:
        # --- Trained kernel (v1 schema) ---
        f.create_dataset("M", data=np.asarray(result["M"], dtype=np.float64))
        f.create_dataset("mu", data=np.asarray(result["mu"], dtype=np.float64))
        f.create_dataset("log_omega", data=np.asarray(result["log_omega"], dtype=np.float64))
        f.create_dataset("log_c_0", data=np.float64(result["log_c_0"]))
        f.create_dataset("log_tau_0", data=np.float64(result["log_tau_0"]))
        f.create_dataset("log_beta", data=np.float64(result["log_beta"]))
        f.create_dataset("rest_wavelengths", data=np.asarray(rest, dtype=np.float64))
        # --- Required-for-loader scalars ---
        f.create_dataset("max_noise_variance", data=np.float64(9.0))
        f.create_dataset("normalization_min_lambda", data=np.float64(1425.0))
        f.create_dataset("normalization_max_lambda", data=np.float64(1475.0))
        # --- v1 provenance: initial values (PCA / data-driven init) ---
        if initial_M is not None:
            f.create_dataset("initial_M", data=np.asarray(initial_M, dtype=np.float64))
        if initial_log_omega is not None:
            f.create_dataset("initial_log_omega",
                             data=np.asarray(initial_log_omega, dtype=np.float64))
        if initial_log_c_0 is not None:
            f.create_dataset("initial_log_c_0", data=np.float64(initial_log_c_0))
        if initial_log_tau_0 is not None:
            f.create_dataset("initial_log_tau_0", data=np.float64(initial_log_tau_0))
        if initial_log_beta is not None:
            f.create_dataset("initial_log_beta", data=np.float64(initial_log_beta))
        # --- v1 training history (embedded so .h5 is self-contained) ---
        hist = result.get("history", {}) or {}
        if "loss" in hist:
            f.create_dataset("loss_history", data=np.asarray(hist["loss"], dtype=np.float64))
        if "log_c_0" in hist:
            f.create_dataset("log_c_0_history", data=np.asarray(hist["log_c_0"], dtype=np.float64))
        if "log_tau_0" in hist:
            f.create_dataset("log_tau_0_history", data=np.asarray(hist["log_tau_0"], dtype=np.float64))
        if "log_beta" in hist:
            f.create_dataset("log_beta_history", data=np.asarray(hist["log_beta"], dtype=np.float64))
        # --- attrs ---
        f.attrs["n_spectra"] = int(n_spectra)
        f.attrs["n_iters"] = int(n_iters)
        f.attrs["lr"] = float(lr)
        f.attrs["vectorized"] = int(vectorized)
        f.attrs["preload_source"] = str(_RUNTIME.get("preload_source", ""))
    print(f"[saved] {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--preload", type=Path, required=True,
                   help="Path to v2 preprocessed trainset.h5 (e.g. "
                        "/nfs/turbo/.../v2_runs/2lpt_loa0_wide_v2_*/trainset.h5)")
    p.add_argument("--max-spectra", type=int, default=None,
                   help="Cap on number of spectra (top-SNR; default: all)")
    p.add_argument("--n-iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.005,
                   help="Default 0.005 matches trainer_v2 production")
    p.add_argument("--k", type=int, default=K_DESI,
                   help=f"Number of GP basis functions (default {K_DESI} — DESI Y3 convention)")
    p.add_argument("--device", default=None,
                   help="cuda or cpu (default: cuda if available)")
    p.add_argument("--chunk-size", type=int, default=12500,
                   help="Batch chunk for spectrum_loss_batch (default 12500; "
                        "memory ~ chunk * n_pix * k * 4B)")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Save a checkpoint every N iter (default 25)")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--max-walltime-sec", type=int, default=None)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Where to write phase2_result.h5/.npz")
    p.add_argument("--z-min", type=float, default=2.15)
    p.add_argument("--z-max", type=float, default=4.25)
    p.add_argument("--min-snr", type=float, default=0.0)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[config] device={device} preload={args.preload}")

    _RUNTIME["checkpoint_dir"] = args.checkpoint_dir
    _RUNTIME["preload_source"] = str(args.preload)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + filter + preprocess via the existing v2 dataset loader.
    # Normalization band: [1425, 1475] Å rest matches MATLAB DR16
    # (`set_parameters.m:30-31`) and v1 production. The earlier choice
    # of Garnett+2017 [1310, 1325] was driven by the legacy narrow
    # trainsets (which ended at 1421 Å); the wide v2 preloads include
    # both bands. Switching to [1425, 1475] reduces bad-median outliers
    # 35× (31 vs 1101 on 2lpt loa-0) — redder band has less Lyα forest
    # contamination + 3× more pixels → more robust median estimate.
    # Documented in docs/notes/2026-05-12_2lpt_corr_noise_debug/.
    #
    # working_dtype=float32 to bound host RAM at 600k+ spectra scale
    # (default f64 needs ~110 GB; matches trainer_v2 dtype).
    ts = load_preprocessed_h5(
        args.preload,
        z_min=args.z_min, z_max=args.z_max, min_snr=args.min_snr,
        max_spectra=args.max_spectra,
        max_noise_variance=9.0,
        apply_mask=True, apply_normalize=True,
        apply_de_forest=True, apply_center=True,
        norm_min_lambda=1425.0, norm_max_lambda=1475.0,
        de_forest_tau_0=TAU_0_PRIOR_MU, de_forest_beta=BETA_PRIOR_MU,
        de_forest_num_lines=NUM_FOREST_LINES,
        dtype=torch.float32,
        working_dtype=np.float32,
    )

    # Keep large arrays at float32 to fit in host RAM. At 300k × 5662
    # spectra, an f64 cast would need 13 GB per array × 3 = 39 GB just
    # for centered/nv/lya_1pzs (plus the f32 originals from load_preprocessed_h5
    # before GC) → blows past 64 GB SLURM mem budget.
    # PCA can run on f32 (sklearn handles it natively).
    centered = ts.fluxes.numpy()             # already f32 from load_preprocessed_h5
    nv = ts.noise_variances.numpy()          # f32
    lya_1pzs = ts.lya_1pzs.numpy()           # f32
    z_qsos = ts.z_qsos.numpy().astype(np.float32)
    rest = ts.rest_wavelengths.numpy().astype(np.float64)  # tiny — keep f64 for save
    mu = (ts.mu.numpy().astype(np.float32) if ts.mu is not None
          else np.zeros(ts.n_pix, dtype=np.float32))

    valid_masks = np.isfinite(centered) & np.isfinite(nv) & (nv > 0)
    print(f"[data] {ts.n_spectra} spectra × {ts.n_pix} pix, "
          f"valid_pix_frac={valid_masks.mean():.3f}, "
          f"dtype={centered.dtype}")

    # 2. PCA init on CPU (sklearn). Pin random_state=0 (per ac7bed8).
    print(f"[pca] computing init for k={args.k}")
    t0 = time.time()
    M_init, latent = _pca_init(centered, k=args.k)
    print(f"[pca] done in {time.time()-t0:.1f}s; top-3 eigvals = "
          f"{latent[:3]}")
    # Data-driven log_omega init: log of per-pixel std of centered fluxes.
    # Matches phase2_train_dr16.py:482. Adam will refine from there.
    log_omega_init = np.log(np.nanstd(centered, axis=0) + 1e-12)

    # 3. Train.
    result = _train(centered, nv, lya_1pzs, valid_masks, z_qsos, mu,
                    M_init, log_omega_init,
                    num_forest_lines=NUM_FOREST_LINES,
                    n_iters=args.n_iters, lr=args.lr, device=device,
                    checkpoint_every=args.checkpoint_every,
                    resume_path=args.resume,
                    max_walltime_sec=args.max_walltime_sec,
                    chunk_size=args.chunk_size,
                    rest_wavelengths=rest)

    # 4. Save: .h5 (DESI schema, primary) + .npz (training history) + README.md.
    out_h5 = args.out_dir / "phase2_result.h5"
    _save_h5(out_h5, result, rest, ts.n_spectra, args.n_iters, args.lr,
             initial_M=M_init,
             initial_log_omega=log_omega_init,
             initial_log_c_0=float(np.log(INITIAL_C_0)),
             initial_log_tau_0=float(np.log(INITIAL_TAU_0)),
             initial_log_beta=float(np.log(INITIAL_BETA)))

    out_npz = args.out_dir / "phase2_result.npz"
    np.savez(out_npz, rest_wavelengths=rest, **{k: result[k] for k in
             ["M", "mu", "log_omega", "log_c_0", "log_tau_0", "log_beta",
              "c_0", "tau_0", "beta"]},
             loss_history=np.asarray(result["history"]["loss"]),
             log_c_0_history=np.asarray(result["history"]["log_c_0"]),
             log_tau_0_history=np.asarray(result["history"]["log_tau_0"]),
             log_beta_history=np.asarray(result["history"]["log_beta"]),
             n_spectra=ts.n_spectra, n_iters=args.n_iters, lr=args.lr)
    print(f"[saved] {out_npz} (training-history record)")

    out_readme = _save_readme(args.out_dir, result, rest,
                              ts.n_spectra, args.n_iters, args.lr,
                              args.k, args.chunk_size, str(device),
                              args.preload)
    print(f"[saved] {out_readme}")

    # 5. Endpoint summary
    print(f"\n=== Endpoint scalars ===")
    print(f"  c_0   = {result['c_0']:.6f}")
    print(f"  tau_0 = {result['tau_0']:.6f}  (Turner+2024 prior μ={TAU_0_PRIOR_MU})")
    print(f"  beta  = {result['beta']:.4f}    (Turner+2024 prior μ={BETA_PRIOR_MU})")


if __name__ == "__main__":
    main()
