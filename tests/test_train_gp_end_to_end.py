"""End-to-end smoke test of the v2 training stack.

Builds a synthetic ``gp_interp_trainset.h5`` matching the schema produced
by ``preload_spectra/prepare_trainset.py``, runs ``load_preprocessed_h5``
(which applies mask + de-forest + center), and trains for 2 epochs via
``trainer_v2.train``.

The point is to **catch any wiring bug in the preprocessing/loading path
locally**, before submitting a multi-hour job to NERSC. Layer 1 / parity /
trainer-smoke tests cover the math, but they bypass the preload schema.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from gpy_dla_detection.training.dataset import load_preprocessed_h5
from gpy_dla_detection.training.model_v2 import GPModelV2
from gpy_dla_detection.training.trainer_v2 import TrainConfig, train
from gpy_dla_detection.voigt import (
    transition_wavelengths as TRANSITION_WAVELENGTHS_NP,
    oscillator_strengths as OSCILLATOR_STRENGTHS_NP,
)


def _build_synthetic_trainset_h5(path: Path, *, n_spectra=64, n_pix=200,
                                  z_min=2.5, z_max=4.0, schema="legacy",
                                  inject_high_noise=True, seed=0):
    """Write a synthetic preloaded trainset HDF5 in either the legacy or
    newer schema. Mirrors what ``preload_spectra/prepare_trainset.py``
    produces.
    """
    rng = np.random.default_rng(seed)
    rest_wave = np.linspace(911.0, 1216.0, n_pix).astype(np.float32)
    z_qsos = (z_min + (z_max - z_min) * rng.random(n_spectra)).astype(np.float32)
    # Synthetic flux: a smooth continuum with small per-pixel scatter.
    fluxes = np.tile(np.linspace(1.0, 0.7, n_pix), (n_spectra, 1)).astype(np.float32)
    fluxes += 0.05 * rng.standard_normal((n_spectra, n_pix)).astype(np.float32)
    noise_variance = (0.01 + 0.02 * rng.random((n_spectra, n_pix))).astype(np.float32)
    if inject_high_noise:
        # Inject a few pixels with noise variance > 9 → should get masked.
        bad = rng.random((n_spectra, n_pix)) < 0.02
        noise_variance[bad] = 20.0  # above default max_noise_variance=9
    redsnr = (1.0 + 5.0 * rng.random(n_spectra)).astype(np.float32)
    bluesnr = (0.5 + 3.0 * rng.random(n_spectra)).astype(np.float32)
    tids = np.arange(n_spectra, dtype=np.int64)

    # rest_wavelengths field is per-spectrum but identical row-by-row.
    rest_wavelengths_per_spectrum = np.tile(rest_wave, (n_spectra, 1))

    with h5py.File(path, "w") as f:
        if schema == "legacy":
            f.create_dataset("tids", data=tids)
            f.create_dataset("rest_wavelengths", data=rest_wavelengths_per_spectrum)
            f.create_dataset("fluxes", data=fluxes)
            f.create_dataset("noise_variance", data=noise_variance)
            f.create_dataset("zqso", data=z_qsos)
            f.create_dataset("redsnr", data=redsnr)
            f.create_dataset("bluesnr", data=bluesnr)
        elif schema == "newer":
            f.create_dataset("tidlist", data=tids)
            f.create_dataset("rest_wavelength_list", data=rest_wavelengths_per_spectrum)
            f.create_dataset("flux_list", data=fluxes)
            f.create_dataset("noise_variance_list", data=noise_variance)
            f.create_dataset("zqsolist", data=z_qsos)
            f.create_dataset("redsnrlist", data=redsnr)
            f.create_dataset("bluesnrlist", data=bluesnr)
        else:
            raise ValueError(schema)


@pytest.mark.parametrize("schema", ["legacy", "newer"])
def test_load_preprocessed_h5_both_schemas(tmp_path, schema):
    """Both schemas should load with identical filtered counts."""
    h5_path = tmp_path / f"trainset_{schema}.h5"
    _build_synthetic_trainset_h5(h5_path, n_spectra=64, n_pix=128, schema=schema)

    ts = load_preprocessed_h5(
        h5_path, z_min=2.0, z_max=5.0, min_snr=0.0,
        apply_mask=True, apply_de_forest=True, apply_center=True,
    )
    assert ts.n_spectra == 64
    assert ts.n_pix == 128
    assert ts.fluxes.shape == (64, 128)
    assert ts.lya_1pzs.shape == (64, 128)
    assert ts.noise_variances.shape == (64, 128)
    assert ts.z_qsos.shape == (64,)


def test_de_forest_changes_flux(tmp_path):
    """De-forest must alter the flux at forest pixels (lya_1pz < 1+z_qso)
    but leave side-band pixels (lya_1pz > 1+z_qso) ~unchanged."""
    h5_path = tmp_path / "trainset.h5"
    _build_synthetic_trainset_h5(h5_path, n_spectra=8, n_pix=100,
                                  z_min=2.5, z_max=2.5, schema="legacy",
                                  inject_high_noise=False)

    ts_no_deforest = load_preprocessed_h5(
        h5_path, z_min=2.0, z_max=5.0,
        apply_mask=False, apply_de_forest=False, apply_center=False,
    )
    ts_deforest = load_preprocessed_h5(
        h5_path, z_min=2.0, z_max=5.0,
        apply_mask=False, apply_de_forest=True, apply_center=False,
    )
    diff = (ts_deforest.fluxes - ts_no_deforest.fluxes).abs().max(dim=0).values
    # In the forest region (rest λ < ~1216 Å, all pixels here), de-forest
    # divides by exp(-tau_eff) > 0 and < 1, so flux should change.
    assert diff.max().item() > 1e-3, "de-forest didn't change any flux"


def test_train_gp_runs_two_epochs_end_to_end(tmp_path):
    """Build synthetic preloaded HDF5 → load → train 2 epochs → assert
    output H5 is well-formed.

    This is the smoke we run before submitting NERSC.
    """
    h5_path = tmp_path / "trainset.h5"
    _build_synthetic_trainset_h5(h5_path, n_spectra=32, n_pix=128,
                                  schema="legacy", inject_high_noise=True)

    ts = load_preprocessed_h5(h5_path, z_min=2.0, z_max=5.0)

    model = GPModelV2(num_pixels=ts.n_pix, k=3)
    cfg = TrainConfig(
        num_epochs=2, batch_size=16, scheduler="none",
        save_every=1, seed=0, device="cpu",
    )

    tw = torch.tensor(TRANSITION_WAVELENGTHS_NP, dtype=torch.float32)
    os_ = torch.tensor(OSCILLATOR_STRENGTHS_NP, dtype=torch.float32)
    out_dir = tmp_path / "out"

    history = train(
        model, ts.fluxes, ts.lya_1pzs, ts.noise_variances, ts.z_qsos,
        tw, os_, out_dir, cfg,
    )

    assert len(history) == 2
    h5_files = sorted(out_dir.glob("model_epoch_*.h5"))
    assert len(h5_files) >= 1
    with h5py.File(h5_files[-1], "r") as f:
        for key in ["M", "log_omega", "log_c_0", "log_tau_0", "log_beta"]:
            assert key in f

    # All loaded model params must be finite (catches NaN-leak bugs in
    # the preprocessing / training path).
    for p in model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after training"
