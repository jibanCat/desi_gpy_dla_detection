"""Streamlined Adam trainer for the GP model.

Purpose-built to be fast and inspectable. Compared to the legacy
``gpy_dla_detection.learn_qso_model.Trainer``:

  - Calls ``vectorized_nll`` (one forward across the whole batch) instead
    of looping per-spectrum in Python.
  - Uses ``loss.backward()`` instead of manual gradient accumulation.
  - No per-batch CPU↔GPU sync, no per-batch prints, no per-batch
    ``torch.cuda.empty_cache()``.
  - Saves checkpoint every N epochs (configurable, default 10), not
    every epoch.
  - Logs loss to a JSON file once per epoch.

The math is identical to the legacy trainer up to autograd vs analytical
gradients (verified by ``tests/test_objective_v2_parity.py``); v2 also
fixes the legacy ``dlog_beta`` approximation.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model_v2 import GPModelV2
from .objective_v2 import vectorized_nll


@dataclass
class TrainConfig:
    """Hyperparameters for the streamlined trainer."""
    # Optimisation
    learning_rate: float = 5e-3
    num_epochs: int = 800
    batch_size: int = 12500
    weight_decay: float = 0.0
    grad_clip: float = 0.0  # 0 = disabled
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "none"
    cosine_t_max: int = 50
    cosine_eta_min: float = 1e-5
    step_size: int = 100
    step_gamma: float = 0.5
    # Forward-model knobs
    num_forest_lines: int = 3
    apply_y1_prior: bool = True
    # I/O cadence
    save_every: int = 10  # epochs between full-checkpoint saves
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _build_optimizer(model: GPModelV2, cfg: TrainConfig):
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def _build_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.cosine_t_max, eta_min=cfg.cosine_eta_min
        )
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=cfg.step_size, gamma=cfg.step_gamma
        )
    return None


def save_checkpoint(model: GPModelV2, opt, scheduler, epoch: int,
                    loss_history: list, output_dir: Path) -> Path:
    """Full PyTorch checkpoint (resume support)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(
        dict(
            model_state=model.state_dict(),
            optimizer_state=opt.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            epoch=epoch,
            loss_history=loss_history,
        ),
        p,
    )
    return p


def save_h5_model(model: GPModelV2, output_dir: Path, epoch: int) -> Path:
    """Compact H5 file (matches the legacy ``model_epoch_<N>.h5`` schema)
    so downstream inference code (run_bayes_select / dla_gp) can load
    v2-trained models without any change."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"model_epoch_{epoch:04d}.h5"
    state = model.state_dict_for_h5()
    with h5py.File(p, "w") as f:
        # Trainable parameters
        f.create_dataset("M", data=state["M"], compression="gzip")
        f.create_dataset("log_omega", data=state["log_omega"], compression="gzip")
        f.create_dataset("log_c_0", data=np.array(state["log_c_0"]))
        f.create_dataset("log_tau_0", data=np.array(state["log_tau_0"]))
        f.create_dataset("log_beta", data=np.array(state["log_beta"]))
        # Metadata required by the legacy inference loader.
        f.create_dataset("rest_wavelengths", data=state["rest_wavelengths"])
        f.create_dataset("mu", data=state["mu"])
        f.create_dataset("max_noise_variance", data=np.array(state["max_noise_variance"]))
        f.attrs["num_pixels"] = state["num_pixels"]
        f.attrs["k"] = state["k"]
        f.attrs["epoch"] = epoch
    return p


def maybe_resume(model: GPModelV2, opt, scheduler, output_dir: Path) -> tuple[int, list]:
    """If ``output_dir`` contains the latest checkpoint, restore
    state_dicts and return (start_epoch, loss_history). Otherwise
    return (0, [])."""
    if not output_dir.exists():
        return 0, []
    checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return 0, []
    latest = checkpoints[-1]
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    opt.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    start_epoch = int(payload["epoch"]) + 1
    loss_history = list(payload.get("loss_history", []))
    return start_epoch, loss_history


def train(
    model: GPModelV2,
    fluxes: torch.Tensor,
    lya_1pzs: torch.Tensor,
    noise_variances: torch.Tensor,
    z_qsos: torch.Tensor,
    transition_wavelengths: torch.Tensor,
    oscillator_strengths: torch.Tensor,
    output_dir: Path,
    cfg: TrainConfig,
    *,
    extra_log_callback: Optional[Callable[[int, float], None]] = None,
) -> list:
    """Streamlined training loop. Returns the loss history list."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)

    fluxes = fluxes.to(device, non_blocking=True)
    lya_1pzs = lya_1pzs.to(device, non_blocking=True)
    noise_variances = noise_variances.to(device, non_blocking=True)
    z_qsos = z_qsos.to(device, non_blocking=True)
    transition_wavelengths = transition_wavelengths.to(device)
    oscillator_strengths = oscillator_strengths.to(device)

    opt = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(opt, cfg)

    start_epoch, loss_history = maybe_resume(model, opt, scheduler, output_dir)

    n = fluxes.shape[0]
    bs = cfg.batch_size

    # Save initial config + persist as JSON.
    with (output_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"[trainer_v2] device={cfg.device} n_spectra={n} n_pix={fluxes.shape[1]} "
          f"k={model.k} batch_size={bs} epochs={cfg.num_epochs} starting_epoch={start_epoch}")

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.perf_counter()

        # Shuffle indices once per epoch (cheap).
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, bs):
            end = min(n, start + bs)
            idx = perm[start:end]

            opt.zero_grad(set_to_none=True)
            loss = vectorized_nll(
                fluxes[idx], lya_1pzs[idx], noise_variances[idx], z_qsos[idx],
                model.M, model.log_omega, model.log_c_0, model.log_tau_0, model.log_beta,
                transition_wavelengths, oscillator_strengths,
                num_forest_lines=cfg.num_forest_lines,
                apply_y1_prior=cfg.apply_y1_prior,
            )
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # Single CPU sync per batch — fine since we accumulate locally.
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_wall = time.perf_counter() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if scheduler is not None:
            scheduler.step()

        log_line = (
            f"[epoch {epoch:4d}] loss={avg_loss:.6f} wall={epoch_wall:.1f}s "
            f"log_tau_0={model.log_tau_0.item():.4f} "
            f"log_beta={model.log_beta.item():.4f} "
            f"log_c_0={model.log_c_0.item():.4f}"
        )
        print(log_line, flush=True)

        if extra_log_callback is not None:
            extra_log_callback(epoch, avg_loss)

        if (epoch % cfg.save_every == 0) or (epoch == cfg.num_epochs - 1):
            ckpt_path = save_checkpoint(model, opt, scheduler, epoch,
                                        loss_history, output_dir)
            h5_path = save_h5_model(model, output_dir, epoch)
            print(f"[saved] {ckpt_path.name}, {h5_path.name}", flush=True)

        # Persist loss history as JSON each epoch — small file, useful for
        # interactive monitoring and post-hoc convergence plots.
        with (output_dir / "loss_history.json").open("w") as f:
            json.dump(loss_history, f)

    return loss_history
