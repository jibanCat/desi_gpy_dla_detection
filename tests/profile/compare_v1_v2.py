"""Speed comparison: legacy ``objective.objective`` vs vectorized
``training.objective_v2.vectorized_nll`` on the same synthetic batch.

This is a focused benchmark — same inputs, same shapes (n_pix=600, k=30,
num_forest_lines=3), measure forward + backward wall time per epoch.

Usage::

    python tests/profile/compare_v1_v2.py [--num-spectra 128] [--epochs 3]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from gpy_dla_detection.objective import objective as legacy_objective  # noqa: E402
from gpy_dla_detection.training.objective_v2 import vectorized_nll  # noqa: E402
from gpy_dla_detection.voigt import (  # noqa: E402
    transition_wavelengths as TRANSITION_WAVELENGTHS_NP,
    oscillator_strengths as OSCILLATOR_STRENGTHS_NP,
)


def _make_batch(num_spectra: int, n_pix: int, k: int, device: torch.device,
                seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    fluxes = torch.randn(num_spectra, n_pix, generator=g, dtype=torch.float32) * 0.2
    mask = torch.rand(num_spectra, n_pix, generator=g) < 0.05
    fluxes[mask] = float("nan")
    noise_variances = (0.05 + 0.05 * torch.rand(num_spectra, n_pix, generator=g)).float()
    z_qsos = (2.5 + 1.75 * torch.rand(num_spectra, generator=g)).float()
    rest_lambda = torch.linspace(911.0, 1216.0, n_pix, dtype=torch.float32)
    one_plus_z_qso = (1.0 + z_qsos).unsqueeze(-1)
    lya_wavelength = 1216.0
    lya_1pz = 1.0 + (one_plus_z_qso * rest_lambda - lya_wavelength) / lya_wavelength
    return (
        fluxes.to(device), lya_1pz.to(device),
        noise_variances.to(device), z_qsos.to(device),
    )


class _MinModel(torch.nn.Module):
    def __init__(self, n_pix: int, k: int, device: torch.device):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(1)
        self.M = torch.nn.Parameter(
            (torch.randn(n_pix, k, generator=g, dtype=torch.float32) * 0.05).to(device)
        )
        self.log_omega = torch.nn.Parameter(
            torch.log(0.1 + 0.05 * torch.rand(n_pix, generator=g)).float().to(device)
        )
        self.log_c_0 = torch.nn.Parameter(
            torch.tensor(np.log(0.1), dtype=torch.float32, device=device)
        )
        self.log_tau_0 = torch.nn.Parameter(
            torch.tensor(np.log(0.00246), dtype=torch.float32, device=device)
        )
        self.log_beta = torch.nn.Parameter(
            torch.tensor(np.log(3.62), dtype=torch.float32, device=device)
        )


def _zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--num-spectra", type=int, default=128)
    p.add_argument("--n-pix", type=int, default=600)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--num-forest-lines", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.005)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[bench] CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    fluxes, lya_1pzs, noise_variances, z_qsos = _make_batch(
        args.num_spectra, args.n_pix, args.k, device, seed=0
    )
    tw = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=torch.float32, device=device)
    os_ = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=torch.float32, device=device)

    # Run legacy
    print(f"[bench] device={args.device} spectra={args.num_spectra} epochs={args.epochs} batch={args.batch_size}")

    def _run_legacy(epochs: int) -> float:
        model = _MinModel(args.n_pix, args.k, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Warm-up
        _zero_grads(model)
        legacy_objective(
            model, fluxes[: args.batch_size], lya_1pzs[: args.batch_size],
            noise_variances[: args.batch_size], args.num_forest_lines,
            tw, os_, z_qsos[: args.batch_size],
        )
        opt.step()

        t0 = time.perf_counter()
        for ep in range(epochs):
            for start in range(0, args.num_spectra, args.batch_size):
                end = min(args.num_spectra, start + args.batch_size)
                _zero_grads(model)
                legacy_objective(
                    model, fluxes[start:end], lya_1pzs[start:end],
                    noise_variances[start:end], args.num_forest_lines,
                    tw, os_, z_qsos[start:end],
                )
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
        return (time.perf_counter() - t0) / epochs

    def _run_v2(epochs: int) -> float:
        model = _MinModel(args.n_pix, args.k, device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Warm-up
        opt.zero_grad()
        loss = vectorized_nll(
            fluxes[: args.batch_size], lya_1pzs[: args.batch_size],
            noise_variances[: args.batch_size], z_qsos[: args.batch_size],
            model.M, model.log_omega, model.log_c_0, model.log_tau_0, model.log_beta,
            tw, os_, num_forest_lines=args.num_forest_lines, apply_y1_prior=True,
        )
        loss.backward()
        opt.step()

        t0 = time.perf_counter()
        for ep in range(epochs):
            for start in range(0, args.num_spectra, args.batch_size):
                end = min(args.num_spectra, start + args.batch_size)
                opt.zero_grad()
                loss = vectorized_nll(
                    fluxes[start:end], lya_1pzs[start:end],
                    noise_variances[start:end], z_qsos[start:end],
                    model.M, model.log_omega, model.log_c_0, model.log_tau_0, model.log_beta,
                    tw, os_, num_forest_lines=args.num_forest_lines, apply_y1_prior=True,
                )
                loss.backward()
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
        return (time.perf_counter() - t0) / epochs

    legacy_per_epoch = _run_legacy(args.epochs)
    v2_per_epoch = _run_v2(args.epochs)

    speedup = legacy_per_epoch / v2_per_epoch
    print()
    print(f"[bench] legacy   per-epoch wall: {legacy_per_epoch:.3f} s")
    print(f"[bench] v2       per-epoch wall: {v2_per_epoch:.3f} s")
    print(f"[bench] speedup (legacy/v2)    : {speedup:.2f}x")
    print(f"[bench] per-spectrum: legacy={legacy_per_epoch / args.num_spectra * 1000:.1f} ms, "
          f"v2={v2_per_epoch / args.num_spectra * 1000:.1f} ms")


if __name__ == "__main__":
    main()
