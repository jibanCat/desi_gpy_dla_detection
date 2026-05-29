"""
gpy_dla_detection/process_helpers.py — HDF5 result I/O for GP-DLA inference.

Overview
--------
Provides two functions used by the inference pipeline (run_bayes_select.py,
dlasearch.py) to initialize and persist per-spectrum GP-DLA outputs:

  initialize_results(num_spectra, max_dlas, num_dla_samples, single_absorber_model)
      → dict of pre-allocated numpy arrays (NaN-filled)

  save_results_to_hdf5(filename, results, spectrum_ids, z_qsos)
      → writes the results dict to an HDF5 file

HDF5 output schema
------------------
The output HDF5 file (processed-{survey}-{program}-{hpx}.h5) contains the
following datasets.  All shapes use ``N`` = num_spectra, ``K`` = max_dlas,
``S`` = num_dla_samples.

Primary per-spectrum arrays:
  target_ids                 int64    (N,)     DESI TARGETID for each spectrum
  z_qsos                     float64  (N,)     QSO emission redshift
  min_z_dlas                 float64  (N,)     minimum DLA search redshift
  max_z_dlas                 float64  (N,)     maximum DLA search redshift
  snrs                       float64  (N,)     SNR in the DLA search window
  snrs_blue                  float64  (N,)     SNR on the blue side of the spectrum

Model posteriors (KEY OUTPUT — see layout note below):
  model_posteriors           float64  (N, 1+num_subdla+K)  posterior per model
  p_dlas                     float64  (N,)     P(≥1 DLA | D, z_QSO)
  p_no_dlas                  float64  (N,)     P(no DLA | D, z_QSO)

model_posteriors column layout
  DLA run (single_absorber_model=False, num_subdla=1):
    col 0  → P(Null | D)       no absorber
    col 1  → P(SubDLA | D)     log NHI ∈ [19, 20.3]
    col 2  → P(DLA(1) | D)     1 DLA
    col 3  → P(DLA(2) | D)     2 DLAs
    col 4  → P(DLA(3) | D)     3 DLAs  (if max_dlas=3)

  Sub-DLA / LLS run (single_absorber_model=True, num_subdla=0):
    col 0  → P(Null | D)       no absorber
    col 1  → P(DLA(1) | D)     1 absorber

  IMPORTANT: The column index of DLA(k) = k + num_subdla.
  DLACatalogue (CDDF_analysis/calc_cddf.py) uses sub_dla=True/False
  to account for this shift when loading the processed file.

MAP parameter estimates (NaN if no DLA detected):
  MAP_z_dlas                 float64  (N, K)   MAP DLA redshift per k
  MAP_log_nhis               float64  (N, K)   MAP log10(N_HI) per k
  z_dla_errs                 float64  (N, K)   1-sigma error on z_DLA
  log_nhi_errs               float64  (N, K)   1-sigma error on log N_HI

Evidence / likelihood components (for diagnostics):
  log_priors_no_dla          float64  (N,)     log P(no DLA | z_QSO)
  log_priors_dla             float64  (N, K)   log P(DLA(k) | z_QSO)
  log_likelihoods_no_dla     float64  (N,)     log P(D | Null model)
  log_likelihoods_dla        float64  (N, K)   log P(D | DLA(k) model)
  log_posteriors_no_dla      float64  (N,)     log P(Null | D, z_QSO)
  log_posteriors_dla         float64  (N, K)   log P(DLA(k) | D, z_QSO)

QMC sample likelihoods (large arrays — for CDDF posterior computation):
  sample_log_likelihoods_dla float64  (N, S, K)  log P(D | DLA(k), θ_j) for
                                                  each sample j (used by DLACatalogue)
  base_sample_inds           int32    (N, K-1, S) resampled indices for DLA(k>1)
                                                  samples (1-indexed in MATLAB convention,
                                                  subtract 1 when loading in Python)

Detection quality:
  detection_flags            bool     (N,)     True if any DLAFLAG bit was set for this
                                               spectrum (i.e. np.sum(fitwarn) > 0);
                                               False = no flags. See fitwarning.py for
                                               per-DLA bitmask values (fitwarn array).

Notes
-----
- ``spectrum_ids`` is saved separately as byte strings (dtype "S") under the
  key ``spectrum_ids``.  ``target_ids`` (int64) is also saved from results.
- Arrays are NaN-initialized; NaN entries mean the spectrum was skipped or
  the model was not evaluated (e.g., z_QSO too low, bad redshift warning).
- The file is written in a single pass (mode "w"); use combine_processed_h5.py
  to merge multiple per-healpix files into a single combined file.
"""

import numpy as np
import h5py
from typing import List


def _gzip_kwargs(value) -> dict:
    """h5py create_dataset kwargs for lossless gzip compression.

    Full rationale + measured size expectations: docs/h5_compression.md.

    The dominant datasets (sample_log_likelihoods_dla (n,num_samples,k) f64 and
    base_sample_inds (n,k-1,num_samples) i32) are ~93-96% NaN/fill because FILTER
    truncation + early-stop leave most QMC samples un-computed; gzip collapses
    those identical-byte runs (measured 15-25x smaller, fully lossless). gzip
    needs a chunked layout, which is impossible for scalar (ndim 0) or empty
    datasets, so fall back to contiguous/uncompressed for those. CDDF readers
    (qso_loader / calc_cddf) decompress transparently.
    """
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.size > 0:
        return {"compression": "gzip", "compression_opts": 4}
    return {}


def initialize_results(num_spectra: int, max_dlas: int, num_dla_samples: int, single_absorber_model: bool = False) -> dict:
    """Pre-allocate the results dict for all spectra (NaN/zero-filled).

    All arrays are pre-filled with NaN (or 0 for integer arrays) and populated
    during inference.  NaN values in the output mean the spectrum was not
    evaluated (e.g., skipped due to z < 2.0 or bad ZWARN).

    Parameters
    ----------
    num_spectra : int
        Number of spectra to process.
    max_dlas : int
        Maximum number of DLAs to model per spectrum (default 3).
    num_dla_samples : int
        Number of QMC samples in the DLA sample grid (e.g. 10000).
    single_absorber_model : bool, optional
        If False (default, DLA run): includes Sub-DLA model column in
        ``model_posteriors`` — layout is [Null, SubDLA, DLA(1), ..., DLA(K)].
        If True (sub-DLA/LLS run): no Sub-DLA column — layout is [Null, DLA(1)].

    Returns
    -------
    dict
        Keys and shapes (N=num_spectra, K=max_dlas, S=num_dla_samples):

        target_ids                 int64    (N,)       DESI TARGETID
        z_qsos                     float64  (N,)       QSO redshift
        min_z_dlas                 float64  (N,)       min DLA search redshift
        max_z_dlas                 float64  (N,)       max DLA search redshift
        snrs                       float64  (N,)       SNR in search window
        snrs_blue                  float64  (N,)       SNR blue side
        model_posteriors           float64  (N, 1+num_subdla+K)  see layout above
        p_dlas                     float64  (N,)       P(>=1 DLA | D, z_QSO)
        p_no_dlas                  float64  (N,)       P(no DLA | D, z_QSO)
        MAP_z_dlas                 float64  (N, K)     MAP DLA redshift
        MAP_log_nhis               float64  (N, K)     MAP log10(N_HI)
        z_dla_errs                 float64  (N, K)     1-sigma z_DLA error
        log_nhi_errs               float64  (N, K)     1-sigma log NHI error
        log_priors_no_dla          float64  (N,)       log P(no DLA | z_QSO)
        log_priors_dla             float64  (N, K)     log P(DLA(k) | z_QSO)
        log_likelihoods_no_dla     float64  (N,)       log P(D | Null)
        log_likelihoods_dla        float64  (N, K)     log P(D | DLA(k))
        log_posteriors_no_dla      float64  (N,)       log P(Null | D)
        log_posteriors_dla         float64  (N, K)     log P(DLA(k) | D)
        sample_log_likelihoods_dla float64  (N, S, K)  per-sample log-likelihoods
        base_sample_inds           int32    (N, K-1, S) DLA(k>1) resample indices
        detection_flags            int32    (N,)       DLAFLAG bitmask (fitwarning.py)
    """

    if single_absorber_model:
        num_subdla = 0
    else:
        num_subdla = 1

    results = {
        "target_ids": np.full(
            (num_spectra,), -1, dtype=np.int64
        ),  # Target IDs for each spectrum
        "z_qsos": np.full(
            (num_spectra,), np.nan
        ),  # Redshifts of the Quasi-Stellar Objects (QSOs)
        "min_z_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Minimum DLA redshift for each spectrum
        "max_z_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Maximum DLA redshift for each spectrum
        "log_priors_no_dla": np.full(
            (num_spectra,), np.nan
        ),  # Log prior for no-DLA model
        "log_priors_dla": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # Log priors for DLA models
        "log_likelihoods_no_dla": np.full(
            (num_spectra,), np.nan
        ),  # Log likelihood for no-DLA model
        "log_likelihoods_dla": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # Log likelihoods for DLA models
        "log_posteriors_no_dla": np.full(
            (num_spectra,), np.nan
        ),  # Log posteriors for no-DLA model
        "log_posteriors_dla": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # Log posteriors for DLA models
        "sample_log_likelihoods_dla": np.full(
            (num_spectra, num_dla_samples, max_dlas), np.nan
        ),  # Sampled log likelihoods for DLA models
        # Correct shape for base_sample_inds: (num_spectra, max_dlas - 1, num_dla_samples)
        "base_sample_inds": np.zeros(
            (num_spectra, max_dlas - 1, num_dla_samples), dtype=np.int32
        ),  # Indices for base samples
        "MAP_z_dlas": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # MAP redshift estimates for DLAs
        "MAP_log_nhis": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # MAP log N_HI estimates for DLAs
        "z_dla_errs": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # 1-sigma errors for DLA redshifts
        "log_nhi_errs": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # 1-sigma errors for log N_HI values
        "model_posteriors": np.full(
            (num_spectra, 1 + num_subdla + max_dlas), np.nan
        ),  # Model posterior probabilities
        "p_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Posterior probability for DLA models
        "p_no_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Posterior probability for no-DLA model
        "snrs": np.full(
            (num_spectra,), np.nan
        ),  # Signal-to-noise ratios for each spectrum
        "snrs_blue": np.full(
            (num_spectra,), np.nan
        ),  # Blue-side signal-to-noise ratios
        "detection_flags": np.full(
            (num_spectra,), 0
        ),  # Detection flags for each spectrum
        # "sample_z_dlas": np.full(
        #     (num_spectra, num_dla_samples), np.nan
        # ),  # Sampled redshifts for DLAs
        # "log_nhi_samples": np.full(
        #     (num_spectra, num_dla_samples), np.nan
        # ),  # Sampled log N_HI values
    }
    return results


def save_results_to_hdf5(
    filename: str,
    results: dict,
    spectrum_ids: List[str],
    z_qsos: np.ndarray,
    run_attrs: dict = None,
) -> None:
    """
    Save the results of the DLA detection process into an HDF5 file.

    This function writes the results from Bayesian model selection and DLA detection into an
    HDF5 file. It saves all relevant information, including spectrum IDs, QSO redshifts,
    and computed posteriors, priors, and likelihoods.

    Parameters:
    ----------
    filename : str
        The name of the HDF5 file to save the results.
    results : dict
        The results dictionary containing the processed outputs (e.g., priors, likelihoods, posteriors).
    spectrum_ids : List[str]
        List of spectrum IDs that were processed.
    z_qsos : np.ndarray
        Array of redshift values for each Quasi-Stellar Object (QSO) corresponding to the spectra.
    run_attrs : dict, optional
        Optional dict of scalar provenance values (str or numeric) written as
        HDF5 root-group attributes.  Use to record run-level parameters such as
        ``pair_prior_mode`` and ``dla_bias`` so each catalog is self-describing.

    Keys in `results`:
    -----------------
    Each key in the `results` dictionary corresponds to a specific output of the DLA detection process.
    They include priors, likelihoods, and model estimates for each spectrum.
    """

    with h5py.File(filename, "w") as f:
        # Save spectrum IDs and QSO redshifts (gzip — see _gzip_kwargs)
        spectrum_ids_arr = np.array(spectrum_ids, dtype="S")
        f.create_dataset(
            "spectrum_ids", data=spectrum_ids_arr, **_gzip_kwargs(spectrum_ids_arr)
        )  # Save spectrum IDs as strings
        f.create_dataset(
            "z_qsos", data=z_qsos, **_gzip_kwargs(z_qsos)
        )  # Save QSO redshifts

        # Loop through the results dictionary and save each key-value pair as an HDF5 dataset
        for key, value in results.items():
            f.create_dataset(
                key, data=value, **_gzip_kwargs(value)
            )  # Save each result in the HDF5 file (gzip-compressed)

        # Write provenance attributes (run-level parameters) to the root group.
        if run_attrs:
            for attr_key, attr_val in run_attrs.items():
                f.attrs[attr_key] = attr_val
