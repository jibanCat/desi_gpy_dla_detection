"""Streamlined data loader for GP training.

Reads the preprocessed HDF5 trainset that the legacy
``gpy_dla_detection.learn_qso_model.GPTrainingSetLoader`` writes and
returns torch tensors ready for ``vectorized_nll``.

Two read paths supported:

  - **Legacy keys** (older preload):
    ``tids``, ``rest_wavelengths``, ``fluxes``, ``noise_variance``,
    ``zqso``, ``redsnr``.
  - **Newer keys** (current production preload):
    ``tidlist``, ``rest_wavelength_list``, ``flux_list``,
    ``noise_variance_list``, ``zqsolist``, ``redsnrlist``.

Filters by z-range, SNR and optional QSO catalog (``TARGETID`` join).

Computes ``lya_1pzs`` per spectrum from the rest wavelengths and z_qso —
this duplicates the formula used in
``gpy_dla_detection/desi_learn_qsos_model.py`` so that v2 is a
drop-in replacement at the data-tensor level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch


# Lyα rest wavelength in Å.
_LYA_WAVELENGTH_AA = 1215.6701


@dataclass
class TrainingSet:
    """Container for the tensors that ``trainer_v2.train`` needs."""
    fluxes: torch.Tensor          # (N, n_pix), centered residual flux
    lya_1pzs: torch.Tensor        # (N, n_pix), 1 + z_lya per pixel
    noise_variances: torch.Tensor  # (N, n_pix)
    z_qsos: torch.Tensor          # (N,)
    rest_wavelengths: torch.Tensor  # (n_pix,)
    n_pix: int
    n_spectra: int


def load_preprocessed_h5(
    h5_path: str | Path,
    *,
    z_min: float = 2.15,
    z_max: float = 4.25,
    min_snr: float = 0.0,
    max_spectra: Optional[int] = None,
    catalog_targetids: Optional[set[int]] = None,
    dtype: torch.dtype = torch.float32,
) -> TrainingSet:
    """Load and filter a preprocessed GP training set HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to e.g. ``gp_interp_trainset.h5`` (legacy) or the newer
        ``preload-loa-gpdla-*/gp_interp_trainset.h5``.
    z_min, z_max : float
        QSO redshift filter.
    min_snr : float
        Minimum red-side SNR (column ``redsnr``).
    max_spectra : int, optional
        Cap on number of spectra; if exceeded, keep top-SNR spectra.
    catalog_targetids : set[int], optional
        If given, restrict to these TARGETIDs (legacy "nonBAL-nonDLA"
        catalog join).
    dtype : torch.dtype
        Target tensor dtype (default fp32).

    Returns
    -------
    TrainingSet
        Tensors ready to pass to ``trainer_v2.train``.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        # Try legacy then newer key naming.
        keys = set(f.keys())
        if {"tids", "rest_wavelengths", "fluxes", "noise_variance", "zqso", "redsnr"} <= keys:
            tids = f["tids"][:]
            rest_wavelengths = f["rest_wavelengths"][:]
            fluxes_raw = f["fluxes"][:]
            noise_variance_raw = f["noise_variance"][:]
            z_qsos_raw = f["zqso"][:]
            redsnrs = f["redsnr"][:]
        elif {"tidlist", "rest_wavelength_list", "flux_list",
              "noise_variance_list", "zqsolist", "redsnrlist"} <= keys:
            tids = f["tidlist"][:]
            rest_wavelengths = f["rest_wavelength_list"][:]
            fluxes_raw = f["flux_list"][:]
            noise_variance_raw = f["noise_variance_list"][:]
            z_qsos_raw = f["zqsolist"][:]
            redsnrs = f["redsnrlist"][:]
        else:
            raise KeyError(
                f"{h5_path} does not contain expected keys for either the "
                f"legacy or newer preload schemas. Found: {sorted(keys)}"
            )

    # Filtering
    n_total = len(fluxes_raw)
    mask = (z_qsos_raw >= z_min) & (z_qsos_raw <= z_max)
    mask &= np.isfinite(redsnrs) & (redsnrs >= min_snr)
    if catalog_targetids is not None:
        mask &= np.isin(tids, list(catalog_targetids))

    n_filtered = int(mask.sum())
    if max_spectra is not None and n_filtered > max_spectra:
        valid_indices = np.where(mask)[0]
        top = valid_indices[np.argsort(redsnrs[mask])[::-1][:max_spectra]]
        new_mask = np.zeros_like(mask)
        new_mask[top] = True
        mask = new_mask
    n_kept = int(mask.sum())

    fluxes = fluxes_raw[mask].astype(np.float32)
    noise_variance = noise_variance_raw[mask].astype(np.float32)
    z_qsos = z_qsos_raw[mask].astype(np.float32)

    # The HDF5 stores per-spectrum rest_wavelengths but they're typically
    # the same grid — pull the first row as the canonical grid.
    if rest_wavelengths.ndim == 2:
        rest_wave = rest_wavelengths[0].astype(np.float32)
    else:
        rest_wave = rest_wavelengths.astype(np.float32)

    print(f"[dataset] {h5_path.name}: {n_total} total → {n_filtered} after "
          f"z/SNR/catalog filter → {n_kept} after max_spectra cap")

    # lya_1pz per pixel per spectrum:
    # lya_1pz[i, j] = (1 + z_qso[i]) * rest_wavelengths[j] / λ_Lya
    one_plus_z_qso = z_qsos[:, None] + 1.0  # (N, 1)
    lya_1pzs = one_plus_z_qso * rest_wave[None, :] / _LYA_WAVELENGTH_AA  # (N, n_pix)

    return TrainingSet(
        fluxes=torch.from_numpy(fluxes).to(dtype),
        lya_1pzs=torch.from_numpy(lya_1pzs).to(dtype),
        noise_variances=torch.from_numpy(noise_variance).to(dtype),
        z_qsos=torch.from_numpy(z_qsos).to(dtype),
        rest_wavelengths=torch.from_numpy(rest_wave).to(dtype),
        n_pix=int(rest_wave.shape[0]),
        n_spectra=int(n_kept),
    )


def load_targetids_from_catalog(catalog_path: str | Path) -> set[int]:
    """Load TARGETIDs from a FITS catalog (e.g. gp_trainset_loa.fits).

    Tiny wrapper using ``astropy.table`` because the existing pipeline
    already depends on it.
    """
    from astropy.table import Table
    t = Table.read(str(catalog_path))
    return set(int(x) for x in t["TARGETID"])
