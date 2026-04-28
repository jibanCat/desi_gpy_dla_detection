"""Smoke test: run the streamlined trainer for a few epochs on synthetic
data, confirm it converges (loss decreases) and produces a valid HDF5
checkpoint that mirrors the legacy schema.

Not a math validation — that's Layer 1 / Layer 4. This just confirms the
end-to-end wiring (data → vectorized_nll → autograd → optimizer step →
checkpoint → resume) works without crashing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from gpy_dla_detection.training.model_v2 import GPModelV2
from gpy_dla_detection.training.objective_v2 import vectorized_nll
from gpy_dla_detection.training.trainer_v2 import TrainConfig, train, save_h5_model
from gpy_dla_detection.voigt import (
    transition_wavelengths as TRANSITION_WAVELENGTHS_NP,
    oscillator_strengths as OSCILLATOR_STRENGTHS_NP,
)


def _make_synthetic_dataset(n_spectra=64, n_pix=128, k=4, seed=42):
    g = torch.Generator().manual_seed(seed)
    fluxes = (torch.randn(n_spectra, n_pix, generator=g, dtype=torch.float32) * 0.2)
    # Inject a few NaN-padded pixels.
    nan_mask = torch.rand(n_spectra, n_pix, generator=g) < 0.03
    fluxes[nan_mask] = float("nan")
    noise_variances = (0.05 + 0.05 * torch.rand(n_spectra, n_pix, generator=g)).float()
    z_qsos = (2.5 + 1.0 * torch.rand(n_spectra, generator=g)).float()
    rest_lambda = torch.linspace(911.0, 1216.0, n_pix, dtype=torch.float32)
    one_plus_z_qso = (1.0 + z_qsos).unsqueeze(-1)
    lya_1pzs = one_plus_z_qso * rest_lambda / 1215.6701
    return fluxes, lya_1pzs.float(), noise_variances, z_qsos


def test_trainer_v2_converges_on_synthetic(tmp_path):
    """3-epoch run on n=64 synthetic spectra. Loss must decrease, h5
    checkpoint must round-trip back through h5py."""
    fluxes, lya_1pzs, noise_variances, z_qsos = _make_synthetic_dataset()
    n_spectra, n_pix = fluxes.shape
    k = 4

    # Seed *before* model construction so the random init_M / init_log_omega
    # are deterministic. Otherwise the loss-decrease assertion below can
    # be flaky across runs / PyTorch versions (Copilot review #3, PR #4).
    torch.manual_seed(0)
    model = GPModelV2(num_pixels=n_pix, k=k)

    cfg = TrainConfig(
        learning_rate=1e-2,
        num_epochs=4,
        batch_size=32,
        scheduler="none",
        save_every=2,
        seed=0,
        device="cpu",
    )
    out_dir = tmp_path / "smoke_out"
    tw = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=torch.float32)
    os_ = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=torch.float32)

    history = train(
        model, fluxes, lya_1pzs, noise_variances, z_qsos,
        tw, os_, out_dir, cfg,
    )

    assert len(history) == cfg.num_epochs
    # Should generally decrease after ≥ 2 epochs (Adam should help on a
    # convex-near-quadratic loss surface like this synthetic).
    assert history[-1] <= history[0], (
        f"Loss didn't decrease: history={history}"
    )

    # config + loss_history JSON written
    assert (out_dir / "config.json").exists()
    assert (out_dir / "loss_history.json").exists()
    with (out_dir / "loss_history.json").open() as f:
        json_history = json.load(f)
    assert json_history == history

    # h5 checkpoint exists and has the legacy-inference-compatible schema.
    h5_files = sorted(out_dir.glob("model_epoch_*.h5"))
    assert len(h5_files) >= 1
    final_h5 = h5_files[-1]
    with h5py.File(final_h5, "r") as f:
        # Trainable parameters
        for key in ["M", "log_omega", "log_c_0", "log_tau_0", "log_beta"]:
            assert key in f, f"missing {key} in {final_h5}"
        # Metadata that the legacy inference loader (dla_gp.py L1042-1056)
        # requires for v2-trained models to be drop-in for DLAHolder.
        for key in ["rest_wavelengths", "mu", "max_noise_variance"]:
            assert key in f, f"missing {key} in {final_h5} (inference loader needs it)"
        M_loaded = f["M"][:]
        assert M_loaded.shape == (n_pix, k)
        assert "num_pixels" in f.attrs
        assert "k" in f.attrs


def test_trainer_v2_resumes(tmp_path):
    """After 2 epochs, training stops; relaunch should resume to 4 epochs."""
    fluxes, lya_1pzs, noise_variances, z_qsos = _make_synthetic_dataset()
    n_pix = fluxes.shape[1]

    out_dir = tmp_path / "resume_out"
    tw = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=torch.float32)
    os_ = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=torch.float32)

    # First run: 2 epochs.
    model = GPModelV2(num_pixels=n_pix, k=4)
    cfg1 = TrainConfig(num_epochs=2, batch_size=32, scheduler="none",
                       save_every=1, seed=0, device="cpu")
    h1 = train(model, fluxes, lya_1pzs, noise_variances, z_qsos, tw, os_, out_dir, cfg1)
    assert len(h1) == 2

    # Second run: resume to 4 epochs total.
    model2 = GPModelV2(num_pixels=n_pix, k=4)  # fresh; trainer should restore state
    cfg2 = TrainConfig(num_epochs=4, batch_size=32, scheduler="none",
                       save_every=1, seed=0, device="cpu")
    h2 = train(model2, fluxes, lya_1pzs, noise_variances, z_qsos, tw, os_, out_dir, cfg2)
    # Should now have 4 entries (2 from before + 2 new).
    assert len(h2) == 4
    # First 2 entries should match the first run exactly (resumed state).
    assert h2[:2] == h1
