"""Streamlined data loader for GP training.

Reads the preprocessed HDF5 trainset that the legacy
``gpy_dla_detection.learn_qso_model.GPTrainingSetLoader`` writes,
applies the train-time preprocessing (mask high-noise pixels +
de-forest + inverse-variance-weighted centering) that the legacy
``SpectrumProcessor`` ran inside ``GPModelTrainer.prepare_data``,
and returns torch tensors ready for ``vectorized_nll``.

Two HDF5 schemas supported:

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

The HDF5 is expected to have already been **interpolated to a common
rest-wavelength grid** by the existing preload pipeline
(``preload_spectra/prepare_trainset.py``). De-forest and center happen
here so the data fed to ``vectorized_nll`` matches what the legacy
trainer fed to ``spectrum_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from gpy_dla_detection.effective_optical_depth import effective_optical_depth


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


def _mask_high_noise_pixels(fluxes: np.ndarray, noise_variances: np.ndarray,
                            max_noise_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """Replace flux/variance with NaN where noise variance exceeds threshold.

    Mirrors ``SpectrumProcessor.mask_noisy_pixels`` but vectorized.
    """
    bad = noise_variances > max_noise_variance
    fluxes = np.where(bad, np.nan, fluxes)
    noise_variances = np.where(bad, np.nan, noise_variances)
    return fluxes, noise_variances


def _de_forest_batch(fluxes: np.ndarray, noise_variances: np.ndarray,
                     rest_wavelengths: np.ndarray, z_qsos: np.ndarray,
                     tau_0: float = 0.00246, beta: float = 3.62,
                     num_forest_lines: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized de-forest: divide each spectrum by exp(-τ_eff) at its
    observed wavelengths.

    Mirrors ``SpectrumProcessor.de_forest_spectra`` exactly (uses the
    same ``effective_optical_depth`` helper). Performance: per-spectrum
    Python loop, but each iteration is a small numpy op so it's fast on
    CPU even for 300k spectra (~30 s in practice).
    """
    n_spectra, n_pix = fluxes.shape
    de_fluxes = np.empty_like(fluxes)
    de_noise = np.empty_like(noise_variances)
    for i in range(n_spectra):
        obs_wave = rest_wavelengths * (1.0 + float(z_qsos[i]))
        tau_per_line = effective_optical_depth(
            obs_wave, beta=beta, tau_0=tau_0, z_qso=float(z_qsos[i]),
            num_forest_lines=num_forest_lines,
        )
        lya_absorption = np.exp(-np.sum(tau_per_line, axis=1))
        # lya_absorption == 1 outside the forest; safe to divide.
        de_fluxes[i] = fluxes[i] / lya_absorption
        de_noise[i] = noise_variances[i] / (lya_absorption ** 2)
    return de_fluxes, de_noise


def _center_fluxes_inverse_variance(fluxes: np.ndarray, noise_variances: np.ndarray,
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance-weighted per-pixel mean subtraction.

    Mirrors ``SpectrumProcessor.center_fluxes``. Returns
    ``(centered_fluxes, mean_flux)``. ``mean_flux`` is also useful as
    ``μ`` (the GP mean function).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = np.where((noise_variances > 0) & np.isfinite(noise_variances),
                        1.0 / noise_variances, 0.0)
    num = np.nansum(fluxes * ivar, axis=0)
    den = np.nansum(ivar, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_flux = num / den
    # Fill NaN pixels (where no spectrum had ivar>0) with the median of
    # the rest of the mean profile.
    finite = np.isfinite(mean_flux)
    if not finite.all():
        if finite.any():
            mean_flux[~finite] = np.nanmedian(mean_flux[finite])
        else:
            mean_flux[:] = 0.0
    centered = fluxes - mean_flux
    return centered, mean_flux


def load_preprocessed_h5(
    h5_path: str | Path,
    *,
    z_min: float = 2.15,
    z_max: float = 4.25,
    min_snr: float = 0.0,
    max_spectra: Optional[int] = None,
    catalog_targetids: Optional[set[int]] = None,
    max_noise_variance: float = 9.0,
    apply_mask: bool = True,
    apply_de_forest: bool = True,
    apply_center: bool = True,
    de_forest_tau_0: float = 0.00246,
    de_forest_beta: float = 3.62,
    de_forest_num_lines: int = 3,
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

    fluxes = fluxes_raw[mask].astype(np.float64)  # promote for preprocessing precision
    noise_variance = noise_variance_raw[mask].astype(np.float64)
    z_qsos = z_qsos_raw[mask].astype(np.float64)

    # The HDF5 stores per-spectrum rest_wavelengths but they're typically
    # the same grid — pull the first row as the canonical grid.
    if rest_wavelengths.ndim == 2:
        rest_wave = rest_wavelengths[0].astype(np.float64)
    else:
        rest_wave = rest_wavelengths.astype(np.float64)

    print(f"[dataset] {h5_path.name}: {n_total} total → {n_filtered} after "
          f"z/SNR/catalog filter → {n_kept} after max_spectra cap")

    # Train-time preprocessing — mirrors the legacy SpectrumProcessor steps
    # 3 (mask) + 6 (de-forest) + 7 (center). Skipped via flags if the caller
    # wants to inspect intermediate state.
    if apply_mask:
        fluxes, noise_variance = _mask_high_noise_pixels(
            fluxes, noise_variance, max_noise_variance
        )
        print(f"[dataset] mask: max_noise_variance={max_noise_variance}")

    if apply_de_forest:
        fluxes, noise_variance = _de_forest_batch(
            fluxes, noise_variance, rest_wave, z_qsos,
            tau_0=de_forest_tau_0, beta=de_forest_beta,
            num_forest_lines=de_forest_num_lines,
        )
        print(f"[dataset] de-forest: tau_0={de_forest_tau_0} beta={de_forest_beta} "
              f"num_lines={de_forest_num_lines}")

    if apply_center:
        fluxes, mean_flux = _center_fluxes_inverse_variance(fluxes, noise_variance)
        print(f"[dataset] centered (inverse-variance weighted mean subtracted)")

    # lya_1pz per pixel per spectrum:
    # lya_1pz[i, j] = (1 + z_qso[i]) * rest_wavelengths[j] / λ_Lya
    one_plus_z_qso = z_qsos[:, None] + 1.0  # (N, 1)
    lya_1pzs = one_plus_z_qso * rest_wave[None, :] / _LYA_WAVELENGTH_AA  # (N, n_pix)

    # Cast back to target dtype for training memory budget.
    return TrainingSet(
        fluxes=torch.from_numpy(fluxes.astype(np.float32)).to(dtype),
        lya_1pzs=torch.from_numpy(lya_1pzs.astype(np.float32)).to(dtype),
        noise_variances=torch.from_numpy(noise_variance.astype(np.float32)).to(dtype),
        z_qsos=torch.from_numpy(z_qsos.astype(np.float32)).to(dtype),
        rest_wavelengths=torch.from_numpy(rest_wave.astype(np.float32)).to(dtype),
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
