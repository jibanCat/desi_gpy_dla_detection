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
this duplicates the formula used in the v1 frozen reference
``gpy_dla_detection/training_v3/desi_learn_qsos_model.py`` so that v2
is a drop-in replacement at the data-tensor level.

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
    mu: Optional[torch.Tensor]    # (n_pix,), inverse-variance-weighted mean
                                  # flux from centering (= μ in Ho 2020 eq.).
                                  # None if --no-center.
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


def _normalize_by_rest_median(
    fluxes: np.ndarray,
    noise_variances: np.ndarray,
    rest_wavelengths: np.ndarray,
    norm_min_lambda: float,
    norm_max_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-spectrum median-flux normalization in [norm_min_lambda, norm_max_lambda].

    Mirrors ``SpectrumProcessor.normalize_spectra`` from the v1 trainer
    (``learn_qso_model.py:290``) but operates on already-interpolated
    rest-frame fluxes.

    Garnett+2017 (arXiv:1605.04460) recommended region is [1310, 1325] Å
    rest, redward of the Lyα forest and void of strong emission lines.
    The v1 production model used [1425, 1475] Å instead. v2 trainsets
    do not include [1425, 1475] in their rest grid (they end at ~1421 Å)
    so [1310, 1325] is the appropriate choice.

    Parameters
    ----------
    fluxes : (N, n_pix) np.ndarray
    noise_variances : (N, n_pix) np.ndarray
    rest_wavelengths : (n_pix,) np.ndarray
    norm_min_lambda, norm_max_lambda : float
        Inclusive boundaries for the median window.

    Returns
    -------
    fluxes_normed : (N, n_pix) np.ndarray
        ``fluxes / median_flux`` per spectrum.
    nv_normed : (N, n_pix) np.ndarray
        ``noise_variance / median_flux**2`` per spectrum (preserves SNR).
    medians : (N,) np.ndarray
        The per-spectrum median used for normalization. Spectra with too
        few valid pixels in the window get NaN here AND have their
        flux row zeroed out (they're effectively unusable).
    """
    norm_mask = (rest_wavelengths >= norm_min_lambda) & (rest_wavelengths <= norm_max_lambda)
    if norm_mask.sum() < 2:
        raise ValueError(
            f"Normalization window [{norm_min_lambda}, {norm_max_lambda}] Å "
            f"contains < 2 pixels of the rest grid "
            f"[{rest_wavelengths.min():.1f}, {rest_wavelengths.max():.1f}] — "
            f"choose a different window."
        )
    # Suppress the "All-NaN slice" RuntimeWarning here — it fires for any
    # row that's entirely NaN inside the normalization window, which is a
    # legitimate (if rare) outcome (e.g. localized pixel-masking removed
    # all pixels in [1310, 1325]). The downstream check on `bad` below
    # zeros those spectra out cleanly, and the print line announces the
    # count, so the warning itself is just noise.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore",
                                 category=RuntimeWarning,
                                 message="All-NaN slice encountered")
        medians = np.nanmedian(fluxes[:, norm_mask], axis=1)  # (N,)
    # Spectra with median == 0, NaN, negative, or |median| < 1e-2 are
    # unusable — divide-by-tiny-median produces |flux| > 100 outliers
    # whose centered values then become a high-variance direction PCA
    # locks onto. Verified on 2lpt v2 wide preloads:
    #   - lower-tail rejection ≤0 / |·|<1e-3 (commit 3e76056, 2026-05-12)
    #     caught the obvious cases.
    #   - The probe at examples/probe_outlier_tail_corr.py (2026-05-13)
    #     showed 10 spectra with med ∈ [1.5e-3, 1e-2] — passing the
    #     previous threshold by ~10× — still bumped PCA-init corr
    #     smoothness 14.9× (0.0130 → 0.1939) when injected into a
    #     5000-spectrum batch. Upper-tail medians (med ∈ [10, 100])
    #     did NOT contaminate (calibration invariance of IV centering).
    # Threshold widened to 1e-2 to capture the [1e-3, 1e-2] marginal tail.
    # See docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md.
    bad = ~np.isfinite(medians) | (medians <= 0) | (np.abs(medians) < 1e-2)
    n_bad = int(bad.sum())
    if n_bad > 0:
        n_neg = int((medians <= 0).sum())
        n_tiny = int(((medians > 0) & (np.abs(medians) < 1e-3)).sum())
        n_marginal = int(((medians >= 1e-3) & (np.abs(medians) < 1e-2)).sum())
        n_nan = int((~np.isfinite(medians)).sum())
        print(f"[dataset] normalize: {n_bad} of {len(medians)} spectra have bad "
              f"median in [{norm_min_lambda}, {norm_max_lambda}] (NaN={n_nan}, "
              f"≤0={n_neg}, |·|<1e-3={n_tiny}, "
              f"[1e-3, 1e-2)={n_marginal}); these will be zeroed out.")
    safe_med = np.where(bad, 1.0, medians)  # avoid div-by-zero; we'll zero them below
    fluxes_normed = fluxes / safe_med[:, None]
    nv_normed = noise_variances / (safe_med[:, None] ** 2)
    if n_bad > 0:
        fluxes_normed[bad] = np.nan
        nv_normed[bad] = np.nan
    return fluxes_normed, nv_normed, medians


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
    apply_normalize: bool = True,
    apply_de_forest: bool = True,
    apply_center: bool = True,
    norm_min_lambda: float = 1310.0,
    norm_max_lambda: float = 1325.0,
    de_forest_tau_0: float = 0.00246,
    de_forest_beta: float = 3.62,
    de_forest_num_lines: int = 3,
    dtype: torch.dtype = torch.float32,
    working_dtype: np.dtype = np.float64,
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
        # Try legacy then newer key naming. For 2D rest_wavelengths
        # (one row per spectrum, all identical), only read the first row —
        # the full 2D array is multiple GB redundant on production sizes
        # (300k × 3801 × 4B ≈ 4.5 GB). Fall back to the full read only if
        # the dataset is 1D.
        def _read_rest_wavelengths(dset):
            if dset.ndim == 2:
                return dset[0]
            return dset[:]

        keys = set(f.keys())
        if {"tids", "rest_wavelengths", "fluxes", "noise_variance", "zqso", "redsnr"} <= keys:
            tids = f["tids"][:]
            rest_wavelengths = _read_rest_wavelengths(f["rest_wavelengths"])
            fluxes_raw = f["fluxes"][:]
            noise_variance_raw = f["noise_variance"][:]
            z_qsos_raw = f["zqso"][:]
            redsnrs = f["redsnr"][:]
        elif {"tidlist", "rest_wavelength_list", "flux_list",
              "noise_variance_list", "zqsolist", "redsnrlist"} <= keys:
            tids = f["tidlist"][:]
            rest_wavelengths = _read_rest_wavelengths(f["rest_wavelength_list"])
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

    # `working_dtype` controls the precision used during preprocessing.
    # Default float64 promotes for precision (matches legacy behaviour).
    # Pass `working_dtype=np.float32` at very large scale (300k+ spectra
    # × 5662 px) to keep host RAM bounded — saves ~50% memory at the
    # cost of a small precision drop in the centering / normalization
    # path. trainer_v2 production also runs in f32 throughout.
    fluxes = fluxes_raw[mask].astype(working_dtype)
    noise_variance = noise_variance_raw[mask].astype(working_dtype)
    z_qsos = z_qsos_raw[mask].astype(working_dtype)

    # The HDF5 stores per-spectrum rest_wavelengths but they're typically
    # the same grid — _read_rest_wavelengths already extracted the 1D grid
    # (first row if 2D, or the array directly if 1D).
    rest_wave = rest_wavelengths.astype(working_dtype)

    print(f"[dataset] {h5_path.name}: {n_total} total → {n_filtered} after "
          f"z/SNR/catalog filter → {n_kept} after max_spectra cap")

    # Train-time preprocessing — mirrors MATLAB DR16 (`preload_qsos.m` +
    # `learn_qso_model.m`):
    # 1. normalize per-spectrum by median in [norm_min_lambda, norm_max_lambda]
    #    (`preload_qsos.m:63-64` writes normalized arrays into the .mat) —
    #    Garnett+2017 [1310, 1325]
    # 2. mask high-noise pixels against the NORMALIZED nv
    #    (`learn_qso_model.m:128` runs `nv > 9` on the pre-normalized arrays
    #    it reads from the preload; the effective threshold is `nv_raw/med² > 9`)
    # 3. de-forest at fixed Turner+2024
    # 4. inverse-variance-weighted mean centering
    #
    # ORDERING NOTE (2026-05-13): we previously masked-then-normalized,
    # which let marginal-median spectra (med ∈ [1e-3, 1e-2]) slip through:
    # raw nv passes the threshold normally, but after normalize their
    # centered values are 100-1000× bulk and dominate PCA. MATLAB is
    # self-protecting against these because mask runs on already-normalized
    # nv (which for a med=0.005 spectrum is ~400× larger → all pixels
    # masked). See `docs/notes/2026-05-12_2lpt_corr_noise_debug/findings.md`
    # and `examples/probe_outlier_tail_corr.py`.
    if apply_normalize:
        fluxes, noise_variance, _meds = _normalize_by_rest_median(
            fluxes, noise_variance, rest_wave,
            norm_min_lambda=norm_min_lambda,
            norm_max_lambda=norm_max_lambda,
        )
        # Auto-detect convention label from the band values; default to a
        # generic "(custom)" if the band doesn't match a known one.
        if abs(norm_min_lambda - 1425.0) < 1 and abs(norm_max_lambda - 1475.0) < 1:
            _band_label = "(MATLAB DR16 convention)"
        elif abs(norm_min_lambda - 1310.0) < 1 and abs(norm_max_lambda - 1325.0) < 1:
            _band_label = "(Garnett+2017 convention)"
        else:
            _band_label = "(custom)"
        print(f"[dataset] normalize: per-spectrum median in "
              f"[{norm_min_lambda}, {norm_max_lambda}] Å rest "
              f"{_band_label}")

    if apply_mask:
        fluxes, noise_variance = _mask_high_noise_pixels(
            fluxes, noise_variance, max_noise_variance
        )
        print(f"[dataset] mask: max_noise_variance={max_noise_variance} "
              f"(applied to normalized nv = nv_raw/med²; matches MATLAB "
              f"learn_qso_model.m:128)")

    if apply_de_forest:
        fluxes, noise_variance = _de_forest_batch(
            fluxes, noise_variance, rest_wave, z_qsos,
            tau_0=de_forest_tau_0, beta=de_forest_beta,
            num_forest_lines=de_forest_num_lines,
        )
        print(f"[dataset] de-forest: tau_0={de_forest_tau_0} beta={de_forest_beta} "
              f"num_lines={de_forest_num_lines}")

    mean_flux = None
    if apply_center:
        fluxes, mean_flux = _center_fluxes_inverse_variance(fluxes, noise_variance)
        print(f"[dataset] centered (inverse-variance weighted mean subtracted)")

    # lya_1pz per pixel per spectrum:
    # lya_1pz[i, j] = (1 + z_qso[i]) * rest_wavelengths[j] / λ_Lya
    one_plus_z_qso = z_qsos[:, None] + 1.0  # (N, 1)
    lya_1pzs = one_plus_z_qso * rest_wave[None, :] / _LYA_WAVELENGTH_AA  # (N, n_pix)

    # Cast back to target dtype for training memory budget.
    mu_t = None
    if mean_flux is not None:
        mu_t = torch.from_numpy(mean_flux.astype(np.float32)).to(dtype)
    return TrainingSet(
        fluxes=torch.from_numpy(fluxes.astype(np.float32)).to(dtype),
        lya_1pzs=torch.from_numpy(lya_1pzs.astype(np.float32)).to(dtype),
        noise_variances=torch.from_numpy(noise_variance.astype(np.float32)).to(dtype),
        z_qsos=torch.from_numpy(z_qsos.astype(np.float32)).to(dtype),
        rest_wavelengths=torch.from_numpy(rest_wave.astype(np.float32)).to(dtype),
        mu=mu_t,
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
