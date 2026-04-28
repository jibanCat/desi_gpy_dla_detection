"""Layer 3 — speed/memory profiling of GP training.

Runs the existing ``Trainer`` (gpy_dla_detection.learn_qso_model.Trainer)
on synthetic data sized like a real DESI Y3 training batch, with
``torch.profiler`` enabled, and dumps:

  - Per-op timing (sorted by self CPU/CUDA time)
  - Memory peak
  - Per-epoch wall time

The aim is to identify the bottlenecks before refactoring (Layer 3 is the
"why is training so slow" diagnostic for the larger streamlining task).

Usage::

    # CPU profile, small toy size
    python tests/profile/profile_training.py --device cpu --num-spectra 64 --epochs 2

    # GPU profile (after ``module load matlab/...`` is irrelevant; need
    # GreatLakes GPU): ``salloc -A cavestru0 -p gpu --gpus=1 -t 30:00 --mem=32G``
    python tests/profile/profile_training.py --device cuda --num-spectra 1024 --epochs 3

Output:
  - tests/profile/results/profile_<tag>.txt  — table of top ops
  - tests/profile/results/trace_<tag>.json   — Chrome trace (load via chrome://tracing)

Notes:
  - Synthetic data deliberately mirrors the production grid:
    n_pix=600 (~ DESI Y3 forest pixel count), k=30, num_forest_lines=3.
  - Uses fp32 to match the production trainer.
  - Does NOT save model checkpoints / plots / h5 files — those are NOT what
    we want to profile (they are a known per-epoch overhead, separately
    timed below).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

# Make repo root importable.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from gpy_dla_detection.objective import objective  # noqa: E402
from gpy_dla_detection.voigt import (  # noqa: E402
    transition_wavelengths as TRANSITION_WAVELENGTHS_NP,
    oscillator_strengths as OSCILLATOR_STRENGTHS_NP,
)


def _make_synthetic_batch(num_spectra: int, n_pix: int, k: int, device: torch.device,
                          seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Realistic centered-flux scale: median normalised to ~1, residual ~0.2 RMS.
    fluxes = torch.randn(num_spectra, n_pix, generator=g, dtype=torch.float32) * 0.2
    # Insert ~5 % NaNs at random pixels (mirrors masked-noisy + de-forest pipeline).
    mask = torch.rand(num_spectra, n_pix, generator=g) < 0.05
    fluxes[mask] = float("nan")

    noise_variances = (0.05 + 0.05 * torch.rand(num_spectra, n_pix, generator=g)).float()
    z_qsos = (2.5 + (4.25 - 2.5) * torch.rand(num_spectra, generator=g)).float()

    # lya_1pz: per-pixel (1+z_lya) for a fixed rest-wavelength grid spanning
    # ~ 911 - 1216 Å (forest), broadcast against (1+z_qso).
    rest_lambda = torch.linspace(911.0, 1216.0, n_pix, dtype=torch.float32)
    one_plus_z_qso = (1.0 + z_qsos).unsqueeze(-1)
    lya_wavelength = 1216.0
    lya_1pz = 1.0 + (one_plus_z_qso * rest_lambda - lya_wavelength) / lya_wavelength
    lya_1pz = lya_1pz.float()

    return (
        fluxes.to(device),
        lya_1pz.to(device),
        noise_variances.to(device),
        z_qsos.to(device),
    )


class _MinModel(torch.nn.Module):
    """Minimal stand-in for GaussianProcessModel — just the parameters and a
    forward that calls objective(). Avoids dragging in the ``Trainer``
    class's per-epoch save/plot machinery, which we want to time separately.
    """
    def __init__(self, n_pix: int, k: int, device: torch.device):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(1)
        # Initialize M from random PCA-like vectors.
        M_init = torch.randn(n_pix, k, generator=g, dtype=torch.float32) * 0.05
        log_omega_init = torch.log(0.1 + 0.05 * torch.rand(n_pix, generator=g)).float()
        self.M = torch.nn.Parameter(M_init.to(device))
        self.log_omega = torch.nn.Parameter(log_omega_init.to(device))
        self.log_c_0 = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32, device=device))
        self.log_tau_0 = torch.nn.Parameter(torch.tensor(np.log(0.00246), dtype=torch.float32, device=device))
        self.log_beta = torch.nn.Parameter(torch.tensor(np.log(3.62), dtype=torch.float32, device=device))


def _run_one_epoch(model, fluxes, lya_1pzs, noise_vars, z_qsos, num_forest_lines,
                   tw, os_, optimizer, batch_size: int) -> dict:
    """One training epoch using the manual-gradient pattern of objective().

    Returns dict of timings (forward+grad accumulate, optimizer step, total).
    """
    timings = {"per_batch_forward": [], "per_batch_step": []}
    total_loss = 0.0

    n = fluxes.shape[0]
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        bf = fluxes[start:end]
        bl = lya_1pzs[start:end]
        bn = noise_vars[start:end]
        bz = z_qsos[start:end]

        optimizer.zero_grad()

        t0 = time.perf_counter()
        with record_function("objective_forward"):
            loss = objective(model, bf, bl, bn, num_forest_lines, tw, os_, bz)
        if torch.cuda.is_available() and bf.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        with record_function("optimizer_step"):
            optimizer.step()
        if torch.cuda.is_available() and bf.is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        timings["per_batch_forward"].append(t1 - t0)
        timings["per_batch_step"].append(t2 - t1)
        total_loss += float(loss.detach().cpu().item())

    timings["loss"] = total_loss
    return timings


def _format_table(prof, top_n: int = 25) -> str:
    """Top ops by self time."""
    return prof.key_averages().table(
        sort_by="self_cpu_time_total" if not torch.cuda.is_available() else "cuda_time_total",
        row_limit=top_n,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--num-spectra", type=int, default=128, help="batch size for the "
                   "profile run (one full pass per epoch)")
    p.add_argument("--n-pix", type=int, default=600)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--num-forest-lines", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--out-dir", default=str(_HERE / "results"))
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[profile] CUDA requested but not available — falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    tag = args.tag or f"{args.device}_n{args.num_spectra}_k{args.k}_npix{args.n_pix}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[profile] device={args.device} spectra={args.num_spectra} "
          f"n_pix={args.n_pix} k={args.k} epochs={args.epochs} batch={args.batch_size}")

    # Build synthetic data.
    fluxes, lya_1pzs, noise_vars, z_qsos = _make_synthetic_batch(
        args.num_spectra, args.n_pix, args.k, device, seed=0
    )
    tw = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=torch.float32, device=device)
    os_ = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=torch.float32, device=device)

    model = _MinModel(args.n_pix, args.k, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Warm-up: one epoch outside the profiler so we don't capture compilation /
    # memory pool initialization.
    print("[profile] warm-up epoch...")
    _run_one_epoch(model, fluxes, lya_1pzs, noise_vars, z_qsos,
                   args.num_forest_lines, tw, os_, optimizer, args.batch_size)

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"[profile] running {args.epochs} epochs under torch.profiler...")
    epoch_walls = []
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for ep in range(args.epochs):
            t0 = time.perf_counter()
            with record_function(f"epoch_{ep}"):
                _run_one_epoch(model, fluxes, lya_1pzs, noise_vars, z_qsos,
                               args.num_forest_lines, tw, os_, optimizer, args.batch_size)
            if args.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            epoch_walls.append(t1 - t0)
            print(f"[profile] epoch {ep}: {epoch_walls[-1]:.2f} s")

    # Summary
    table = _format_table(prof, top_n=25)
    summary = []
    summary.append(f"# Layer 3 — training profile ({tag})")
    summary.append("")
    summary.append("## Setup")
    summary.append(f"- device: `{args.device}`")
    summary.append(f"- spectra (per epoch): `{args.num_spectra}`")
    summary.append(f"- n_pix: `{args.n_pix}`, k: `{args.k}`, num_forest_lines: `{args.num_forest_lines}`")
    summary.append(f"- batch_size: `{args.batch_size}`, epochs: `{args.epochs}`")
    summary.append(f"- optimizer: Adam(lr={args.lr})")
    summary.append("")
    summary.append("## Per-epoch wall time")
    summary.append("")
    summary.append("| epoch | seconds |")
    summary.append("|:---:|---:|")
    for i, w in enumerate(epoch_walls):
        summary.append(f"| {i} | {w:.3f} |")
    summary.append(f"| **mean** | **{np.mean(epoch_walls):.3f}** |")
    summary.append("")
    summary.append("## Top 25 ops by self time")
    summary.append("")
    summary.append("```")
    summary.append(table)
    summary.append("```")

    txt_path = out_dir / f"profile_{tag}.txt"
    txt_path.write_text("\n".join(summary))
    print(f"[profile] wrote {txt_path}")

    trace_path = out_dir / f"trace_{tag}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"[profile] wrote {trace_path} (load via chrome://tracing)")


if __name__ == "__main__":
    main()
