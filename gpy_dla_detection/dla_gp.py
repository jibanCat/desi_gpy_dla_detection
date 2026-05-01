"""
dla_gp.py — Gaussian Process model for quasar spectra with DLA absorption.

Overview
--------
Extends NullGP (QSO-emission-only GP) by adding one or more Damped Lyman-Alpha
(DLA) absorbers along the line of sight.  Each DLA multiplies the QSO mean
model and absorption-noise model by a Voigt absorption profile.

The primary class is ``DLAGP``.  For loading from a MATLAB ``.mat`` file use
``DLAGPMAT``.

Science
-------
Model evidence for k DLAs is approximated via Quasi-Monte Carlo (QMC)
integration over (z_DLA, log NHI) drawn from flat priors in the search window.
The Woodbury matrix identity reduces the GP log-likelihood from O(n³) to
O(n k²), where n is the number of observed pixels and k is the GP rank.

For the default DLA run (Ho et al. 2020, arxiv 2003.11036):
  - log NHI ∈ [20.3, 23]  (uniform prior)
  - z_DLA search window defined by set_parameters.Parameters
  - Sample file: dla_samples_a03.mat  (same as Ho+2020)
  - Multi-DLA: up to ``max_dlas`` (default 3) via recursive importance resampling

Key functions
-------------
select_region_indices_searchsorted : adaptive window selection for QMC samples
DLAGP.set_data : preprocess a single spectrum and attach to the model
DLAGP.sample_log_likelihood_k_dlas : QMC evidence for k DLAs
DLAGP.maximum_a_posteriori : MAP (z_DLA, log NHI) estimates

Dependencies
------------
voigt_fast.VoigtProfile (C extension) — falls back to voigt.voigt_absorption
    with a RuntimeWarning if the C extension is not built.
null_gp.NullGP — base class providing the Woodbury log-likelihood
bayesian_model_selection.BayesModelSelect — callers use this for model comparison

References
----------
Ho, Bird & Garnett (2020) https://arxiv.org/abs/2003.11036
Garnett et al. (2017) https://arxiv.org/abs/1605.04538
"""
import time

from typing import Tuple, Optional, Callable, List
import os

import concurrent.futures
from concurrent.futures import as_completed
from desiutil.log import log


import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.special import logsumexp

import h5py

from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .null_gp import NullGP

# Attempt to import VoigtProfile from voigt_fast, and fall back to voigt if it fails
try:
    from .voigt_fast import VoigtProfile

    voigt_absorption = VoigtProfile().compute_voigt_profile
# OSError, ImportError:
except (OSError, ImportError):
    import warnings
    warnings.warn(
        "Could not load the compiled C Voigt extension (_voigt.so). "
        "Falling back to the pure-Python voigt_absorption, which is ~100x slower. "
        "To fix, rebuild the C extension: see gpy_dla_detection/voigt_fast.py.",
        RuntimeWarning,
        stacklevel=2,
    )
    from .voigt import voigt_absorption

# this could be replaced to DLASamples in the future;
# I import this is for the convenient of my autocomplete
from .dla_samples import DLASamplesMAT

# Limit the number of workers to the number of CPU cores
# max_workers = os.cpu_count() * 2

# fast search method for adapative truncated sampling
def select_region_indices_searchsorted(initial_z, initial_logL, z_all, z_tol=0.02, logL_null=None):
    """
    Selects a mask for z_all entries that are within `z_tol` of any high-likelihood z values from an initial scan.

    This function efficiently identifies entries in `z_all` whose values lie within a tolerance
    range (`z_tol`) of any `initial_z` sample whose corresponding log-likelihood exceeds a given threshold.

    This is useful in truncated likelihood integration schemes where the parameter space
    (e.g., z_DLA) is multi-modal and one wants to restrict evaluation to promising subregions.

    Parameters
    ----------
    initial_z : np.ndarray of shape (N_scan,)
        The redshift samples from the initial low-resolution scan.
    initial_logL : np.ndarray of shape (N_scan,)
        The corresponding log-likelihood values of `initial_z`.
    z_all : np.ndarray of shape (N_total,)
        The full set of redshift values (e.g., from dense QMC samples) to be filtered.
    z_tol : float, optional
        The redshift tolerance window. A point in `z_all` is kept if it lies within
        ±`z_tol` of any high-likelihood `initial_z`. Default is 0.02.
    logL_null : float or None, optional
        The log-likelihood threshold to define "high-likelihood" points. If None, the maximum
        value of `initial_logL` is used.

    Returns
    -------
    mask : np.ndarray of shape (N_total,)
        A boolean array indicating which entries in `z_all` fall near any high-likelihood
        `initial_z` within the specified tolerance.

    Notes
    -----
    Internally, this uses `np.searchsorted` on the sorted array of `z_good` for efficient
    vectorized interval checks, avoiding nested loops.

    Examples
    --------
    >>> mask = select_region_indices_searchsorted(initial_z, initial_logL, z_all, z_tol=0.01)
    >>> filtered_z = z_all[mask]
    """
    if logL_null is None:
        logL_null = np.max(initial_logL)

    z_good = initial_z[initial_logL > logL_null]
    if len(z_good) == 0:
        return np.array([], dtype=int)  # no valid points

    z_good_sorted = np.sort(z_good)
    left = np.searchsorted(z_good_sorted, z_all - z_tol, side='left')
    right = np.searchsorted(z_good_sorted, z_all + z_tol, side='right')

    # Valid if at least one good z is within [z - z_tol, z + z_tol]
    mask = (right > left)
    return mask    

# fast searchsorted method for resampling
def searchsorted_method(W, N):
    """
    Fast searchsorted method for resampling indices based on weights.
    equivalent to MATLAB's randsample with replacement.
    """
    W = W / np.sum(W)
    cumsum = np.cumsum(W)
    u = np.random.rand(N)
    return np.searchsorted(cumsum, u)


def process_sample(
    i: int,
    num_dlas: int,
    sample_z_dlas: np.ndarray,
    base_sample_inds_T: np.ndarray,
    dla_samples: DLASamplesMAT,
    params: Parameters,
    sample_log_likelihood_k_dlas: Callable[[np.ndarray, np.ndarray], float],
    min_z_separation: float,
) -> float:
    """
    Process a single sample by querying DLA parameters and computing the log likelihood.

    This function retrieves the DLA parameters (redshift `z_dlas` and column density `logNHI`) for the
    current sample. If `num_dlas` > 0, it retrieves additional parameters for multiple DLAs. Finally, it
    computes the log likelihood of the k-DLA model for the given sample.

    Args:
        i (int): Index of the current sample.
        num_dlas (int): Number of DLAs in the model for this sample.
        sample_z_dlas (np.ndarray): Array of sampled redshift values for DLAs.
        base_sample_inds_T (np.ndarray): "The transpose" of Base indices to be resampled according to the prior.
            Transpose is faster for indexing.
        dla_samples ('DLASamplesMAT'): Object containing the DLA sample catalog.
        params ('Parameters'): Model parameters object.
        sample_log_likelihood_k_dlas (Callable): Function to compute the log likelihood of k-DLA model.
        min_z_separation (float): Minimum redshift separation for the DLA samples.

    Returns:
        float: The computed log likelihood for this sample.
    """

    # Query the 1st DLA parameter {z_dla, logNHI}_{i=1} from the given DLA samples
    z_dlas = np.array([sample_z_dlas[i]])
    log_nhis = np.array([dla_samples.log_nhi_samples[i]])
    nhis = np.array([dla_samples.nhi_samples[i]])

    # Query the 2:k DLA parameters {z_dla, logNHI}_{i=2}^k_dlas
    if num_dlas > 0:
        # use transpose of base_sample_inds to speed up indexing
        base_ind = base_sample_inds_T[i, :num_dlas] #base_sample_inds[:num_dlas, i]
        z_dlas_2_k = sample_z_dlas[base_ind]
        log_nhis_2_k = dla_samples.log_nhi_samples[base_ind]
        nhis_2_k = dla_samples.nhi_samples[base_ind]

        # Append to samples to be applied on calculating the log likelihood
        z_dlas = np.append(z_dlas, z_dlas_2_k)
        log_nhis = np.append(log_nhis, log_nhis_2_k)
        nhis = np.append(nhis, nhis_2_k)

        del z_dlas_2_k, log_nhis_2_k, nhis_2_k

    # Compute the sample log likelihoods conditioned on k-DLAs
    log_likelihood = sample_log_likelihood_k_dlas(z_dlas, nhis) - np.log(
        params.num_dla_samples
    )

    return log_likelihood


def process_batch(
    batch_indices: List[int],
    num_dlas: int,
    sample_z_dlas: np.ndarray,
    base_sample_inds: np.ndarray,
    dla_samples: DLASamplesMAT,
    params: Parameters,
    sample_log_likelihood_k_dlas: callable,
    min_z_separation: float,  # Add min_z_separation as an argument
) -> List[float]:
    """
    Process a batch of samples. For each sample in the batch, this function computes
    the log likelihood using `process_sample` and returns the results as a list.

    Args:
        batch_indices (List[int]): Indices of the samples in the batch.
        num_dlas (int): Number of DLAs to consider in the model.
        sample_z_dlas (np.ndarray): Array of sampled redshift values for DLAs.
        base_sample_inds (np.ndarray): Base indices for resampling according to the prior.
        dla_samples ('DLASamplesMAT'): Object containing the DLA sample catalog.
        params ('Parameters'): Model parameters object.
        sample_log_likelihood_k_dlas (callable): Function to compute log likelihood for each sample.
        min_z_separation (float): Minimum redshift separation for DLA pairs.

    Returns:
        List[float]: List of log likelihoods for each sample in the batch.
    """
    batch_results = []  # This will store the results for the entire batch

    # vectorize the loop
    batch_results = np.empty(len(batch_indices), dtype=np.float64)
    base_sample_inds_T = base_sample_inds.T  # Transpose for faster indexing

    # t0 = time.time()

    for j, i in enumerate(batch_indices):
        # Process each sample using the same logic as process_sample
        batch_results[j] = process_sample(
            i,
            num_dlas,
            sample_z_dlas,
            base_sample_inds_T,
            dla_samples,
            params,
            sample_log_likelihood_k_dlas,
            min_z_separation,  # Pass the missing argument
        )
    # print(f"Batch {batch_indices[0]}-{batch_indices[-1]} took {time.time() - t0:.3f} sec")

    return batch_results  # Return the list of results for the batch


class DLAGP(NullGP):
    """
    DLA GP model for QSO emission + DLA intervening:
        p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, z_dla, logNHI)

    additional two parameters (z_dla, logNHI) will control the position
    and the strength of the absorption intervening on the QSO emission.

    Since the integration is not tractable, so we use QMC to approximate
    the model evidence.

    How many QMC samples will be defined in Parameters and DLASamples.

    :param rest_wavelengths: λ, the range of λ you model your GP on QSO emission
    :param mu: mu, the mean model of the GP.
    :param M: M, the low rank decomposition of the covariance kernel: K = MM^T.
    :param log_omega: log ω, the pixel-wise noise of the model. Used to model absorption noise.
    :param log_c_0: log c₀, the constant in the Lyman forest noise model,
        Lyman forest noise := s(z) = 1 - exp(-effective_optical_depth) + c_0.
    :param log_tau_0: log τ₀, the scale factor of effective optical depth in the absorption noise,
        effective_optical_depth := ∑ τ₀ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
    :param log_beta: log β, the exponent of the effective optical depth in the absorption noise.
    :param prev_tau_0: τ_kim, the scale factor of effective optical depth used in mean-flux suppression.
    :param prev_beta: β_kim, the exponent of the effective optical depth used in mean-flux suppression.
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: DLASamplesMAT,
        rest_wavelengths: np.ndarray,
        mu: np.ndarray,
        M: np.ndarray,
        log_omega: np.ndarray,
        log_c_0: float,
        log_tau_0: float,
        log_beta: float,
        prev_tau_0: float = 0.0023,
        prev_beta: float = 3.65,
        min_z_separation: float = 3000.0,
        broadening: bool = True,
    ):
        super().__init__(
            params,
            prior,
            rest_wavelengths,
            mu,
            M,
            log_omega,
            log_c_0,
            log_tau_0,
            log_beta,
            prev_tau_0,
            prev_beta,
        )

        self.min_z_separation = self.params.kms_to_z(min_z_separation)

        self.dla_samples = dla_samples

        self.broadening = broadening

        # Initialize a cache for Voigt profiles
        self.voigt_cache = {}

    def log_model_evidences(self, max_dlas: int) -> np.ndarray:
        """
        marginalize out the DLA parameters, {(z_dla_i, logNHI_i)}_{i=1}^k_dlas,
        and return an array of log_model_evidences for 1:k DLA models

        Note: we provide an integration method here to reproduce the functionality
        in Ho-Bird-Garnett's code, but we encourage users to improve this sampling
        scheme to be more efficient with another external script by calling
        self.sample_log_likelihood_k_dlas directly.

        :param max_dlas: the number of DLAs we want to marginalise

        :return: [P(D | 1 DLA), ..., P(D | k DLAs)]
        """
        # allocate the final log model evidences
        log_likelihoods_dla = np.empty((max_dlas,))
        log_likelihoods_dla[:] = np.nan

        # base inds to store the QMC samples to be resampled according
        # the prior, which is the posterior of the previous run.
        base_sample_inds = np.zeros(
            (
                max_dlas - 1,
                self.params.num_dla_samples,
            ),
            dtype=np.int32,
        )

        # sorry, let me follow the convention of the MATLAB code here
        # could be changed to (max_dlas, num_dla_samples) in the future.
        sample_log_likelihoods = np.empty((self.params.num_dla_samples, max_dlas))
        sample_log_likelihoods[:] = np.nan

        # preallocate sample probabilities
        sample_probabilities = np.empty(self.params.num_dla_samples)

        # prepare z_dla samples
        sample_z_dlas = self.dla_samples.sample_z_dlas(
            self.this_wavelengths, self.z_qso
        )
        # move this to the top of the function to avoid re-computation
        lognorm = np.log(self.params.num_dla_samples)

        # compute probabilities under DLA model for each of the sampled
        # (normalized offset, log(N HI)) pairs
        for num_dlas in range(max_dlas):  # count from zero to max_dlas - 1

            # [Need to be parallelized]
            # Roman's code has this part to be parallelized.
            for i in range(self.params.num_dla_samples):
                # query the 1st DLA parameter {z_dla, logNHI}_{i=1} from the
                # given DLA samples.
                z_dlas = np.array([sample_z_dlas[i]])
                log_nhis = np.array([self.dla_samples.log_nhi_samples[i]])
                nhis = np.array([self.dla_samples.nhi_samples[i]])

                # query the 2:k DLA parameters {z_dla, logNHI}_{i=2}^k_dlas
                if num_dlas > 0:
                    base_ind = base_sample_inds[:num_dlas, i]

                    z_dlas_2_k = sample_z_dlas[base_ind]
                    log_nhis_2_k = self.dla_samples.log_nhi_samples[base_ind]
                    nhis_2_k = self.dla_samples.nhi_samples[base_ind]

                    # append to samples to be applied on calculating the log likelihood
                    z_dlas = np.append(z_dlas, z_dlas_2_k)
                    log_nhis = np.append(log_nhis, log_nhis_2_k)
                    nhis = np.append(nhis, nhis_2_k)

                    del z_dlas_2_k, log_nhis_2_k, nhis_2_k

                # store the sample log likelihoods conditioned on k-DLAs
                sample_log_likelihoods[i, num_dlas] = self.sample_log_likelihood_k_dlas(
                    z_dlas, nhis
                ) - np.log(
                    self.params.num_dla_samples
                )  # additional occams razor

            # check if any pair of dlas in this sample is too close this has to
            # happen outside the parfor because "continue" slows things down
            # dramatically
            if num_dlas > 0:
                # all_z_dlas : (num_dlas, num_dla_samples)
                ind = base_sample_inds[:num_dlas, :]  # (num_dlas - 1, num_dla_samples)

                all_z_dlas = np.concatenate(
                    [sample_z_dlas[None, :], sample_z_dlas[ind]], axis=0
                )  # (num_dlas, num_dla_samples)

                ind = np.any(
                    np.diff(np.sort(all_z_dlas, axis=0), axis=0)
                    < self.min_z_separation,
                    axis=0,
                )
                sample_log_likelihoods[ind, num_dlas] = np.nan

            # to prevent numerical underflow
            max_log_likelihood = np.nanmax(sample_log_likelihoods[:, num_dlas])

            sample_probabilities = np.exp(
                sample_log_likelihoods[:, num_dlas] - max_log_likelihood
            )

            log_likelihoods_dla[num_dlas] = (
                max_log_likelihood
                + np.log(np.nanmean(sample_probabilities))
                - lognorm * num_dlas
            )  # occams razor for more DLA parameters

            # no needs for re-sample the QMC samples for the last run
            if (num_dlas + 1) == max_dlas:
                break

            # if p(D | z_QSO, k DLA) is NaN, then
            # finish the loop.
            # It's usually because p(D | z_QSO, no DLA) is very high, so
            # the higher order DLA model likelihoods already underflowed
            if np.isnan(log_likelihoods_dla[num_dlas]):
                print(
                    "Finish the loop earlier because NaN value in log p(D | z_QSO, {} DLAs)".format(
                        num_dlas
                    )
                )
                break

            # avoid nan values in the randsample weights
            nanind = np.isnan(sample_probabilities)
            W = sample_probabilities
            W[nanind] = 0.0

            # resample the base sample indices using searchsorted method
            base_sample_inds[num_dlas, :] = searchsorted_method(
                W,
                self.params.num_dla_samples,
            )

            # base_sample_inds[num_dlas, :] = np.random.choice(
            #     np.arange(self.params.num_dla_samples).astype(np.int32),
            #     size=self.params.num_dla_samples,
            #     replace=True,
            #     p=W / W.sum(),
            # )

        # store sample likelihoods for MAP value calculation
        # this could cause troubles for parallelization in the future
        self.sample_log_likelihoods = sample_log_likelihoods
        self.base_sample_inds = base_sample_inds

        return log_likelihoods_dla

    def parallel_log_model_evidences(
        self,
        max_dlas: int,
        max_workers: int = 32,
        batch_size: int = 313,
        executor=None,
        null_evidence: Optional[float] = None,
        filter_low_likelihood: bool = False,
    ) -> np.ndarray:
        """
        Parallelized version of the log model evidences computation using process-based parallelization.

        This method computes the log likelihoods of the k-DLA models in parallel using `ProcessPoolExecutor`.
        The process is repeated for each number of DLAs (up to `max_dlas`), and the results are stored
        in an array.

        If `null_evidence` is provided, the method performs an initial scan to
        identify high-likelihood regions and estimates the marginal likelihood
        with a truncated sampling correction (weighted by prior volume).

        Args:
            max_dlas (int): The maximum number of DLAs to be considered in the model.
            max_workers (int, optional): Maximum number of workers to use. Defaults to number of CPU cores * 2.
            batch_size (int, optional): Number of samples per batch. Defaults to 100.
            executor (ProcessPoolExecutor, optional): An existing executor to reuse; if not provided, a new one is created.
            null_evidence (float, optional): The log likelihood of the null model.
            If provided, it will be used to stop the computation early if the null model likelihood is higher.
            filter_low_likelihood (bool, optional): Whether to filter out low-likelihood samples based on initial scan. Defaults to True.

        Returns:
            np.ndarray: Array containing the computed log likelihoods for 1 to `max_dlas` DLAs.
        """
        # Set default number of workers if not provided
        if max_workers is None:
            max_workers = os.cpu_count() * 2

        # Allocate the final log model evidences
        log_likelihoods_dla = np.empty((max_dlas,))
        log_likelihoods_dla[:] = np.nan

        # Base indices to store the QMC samples to be resampled according to the prior
        base_sample_inds = np.zeros(
            (max_dlas - 1, self.params.num_dla_samples), dtype=np.int32
        )

        # Allocate sample log likelihoods array
        sample_log_likelihoods = np.empty((self.params.num_dla_samples, max_dlas))
        sample_log_likelihoods[:] = np.nan

        # Preallocate sample probabilities
        sample_probabilities = np.empty(self.params.num_dla_samples)
        sample_probabilities[:] = np.nan

        # precomputed log normalization factor
        lognorm = np.log(self.params.num_dla_samples)

        # Prepare z_dla samples
        sample_z_dlas = self.dla_samples.sample_z_dlas(
            self.this_wavelengths, self.z_qso
        )

        # Check if an executor is passed; if not, create one locally
        local_executor = False
        if executor is None:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            local_executor = True

        # ========= Adative truncated sampling =========
        # Your first 10000 samples scan is to find the regions of high likelihood,
        # and then you only sample the regions of high likelihood
        # 
        # Step 1: only use a small subset of QMC samples for the initial scan
        n_initial = max(int(self.params.num_dla_samples // 20) , 5000)  # only scan the first 5000 samples
        initial_logL = np.empty(n_initial)
        initial_logL[:] = np.nan
        # # Select slice of samples for initial scan
        # z_dla_subset = sample_z_dlas[:n_initial]
        # log_nhi_subset = self.dla_samples.log_nhi_samples[:n_initial]
        # get the new batch_size based on the n_workers
        batch_size_subset = int(n_initial / max_workers)
        if batch_size_subset * max_workers < n_initial:
            batch_size_subset += 1

        # Build initial batches
        initial_indices = list(range(n_initial))
        initial_batches = [
            initial_indices[i : i + batch_size_subset]
            for i in range(0, n_initial, batch_size_subset)
        ]

        try:
            if filter_low_likelihood and (null_evidence is not None):
                # ========= Initial scan on the 10000 subset =========
                # Step 2: Run scan on the 10000 subset
                init_num_dla = 0  # num_dlas = 0 for initial scan
                futures = {
                    executor.submit(
                        process_batch,
                        batch,
                        init_num_dla,  # num_dlas = 0 for initial scan
                        sample_z_dlas,
                        base_sample_inds,
                        self.dla_samples,
                        self.params,
                        self.sample_log_likelihood_k_dlas,
                        self.min_z_separation,
                    ): batch
                    for batch in initial_batches
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        results = future.result()
                        for i, res in zip(batch, results):
                            initial_logL[i] = res
                            sample_log_likelihoods[i, init_num_dla] = res
                    except Exception as e:
                        log.error(f"Initial scan error: {e}")
                # Step 3: define the log likelihood "zDLA" mask for the initial scan
                z_tol = 0.02 # TODO: find the best value
                # These are the indices of the z_dlas that are within z_tol of the high likelihood regions
                # in the initial scan. You can think of this as nested sampling
                # in general it reduces samples by a factor of ~100 (100000 to ~2000; 5000 to ~100)
                valid_mask = select_region_indices_searchsorted(
                    z_all=sample_z_dlas,                 # full 100000
                    initial_logL=initial_logL,           # only 5000
                    initial_z=sample_z_dlas[:n_initial], # only 5000
                    z_tol=z_tol,
                    logL_null=null_evidence,
                ) # this is boolean mask of the full 100000 samples

                # ========== Early stopping if initial scan returns no valid regions ==========
                if np.sum(valid_mask) == 0:
                    log.warning(
                        "No valid regions found in the initial scan. Returning NaN for all log likelihoods."
                    )
                    # Save the approximated log likelihoods for 1 DLA model
                    max_log_likelihood = np.nanmax(sample_log_likelihoods[:, init_num_dla])
                    sample_probabilities[:] = np.exp(
                        sample_log_likelihoods[:, init_num_dla] - max_log_likelihood
                    )
                    log_likelihoods_dla[init_num_dla] = (
                        max_log_likelihood
                        + np.log(np.nanmean(sample_probabilities))
                        - lognorm * init_num_dla
                    )
                    log.info(
                        f"Stopping early at {init_num_dla + 1} DLAs because the log likelihood {log_likelihoods_dla[init_num_dla]} is less than the null model evidence {null_evidence}."
                    )
                    # break the try block to avoid further processing
                    # Store results for future use
                    self.sample_log_likelihoods = sample_log_likelihoods
                    self.base_sample_inds = base_sample_inds

                    return log_likelihoods_dla

                # Step 4: Filter the batch indices based on the valid mask,
                #         so this is filtered both lognhi and z_dla
                # 
                # Avoid using initial scan samples again for refined sampling
                _valid_mask = valid_mask.copy() # retain the original mask
                valid_mask[:n_initial] = False  # Exclude the initial scan samples
                indices = np.where(valid_mask)[0]
                batch_size = int(len(indices) / max_workers)
                if batch_size * max_workers < len(indices):
                    batch_size += 1
                batches = [
                    indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
                ]
                # Estimate average log-likelihood in the rejected region (logL < null_evidence)
                # If few values, fallback to minimum or null_evidence
                # below_null = initial_logL[initial_logL < null_evidence]
                # 
                # this should be the samples out side of the valid mask
                below_null = initial_logL[~_valid_mask[:n_initial]]

                if below_null.size > 5:
                    max_log_below_null = np.nanmax(below_null)
                    probabilities_below_null = np.exp(
                        below_null - max_log_below_null
                    )
                    log_initial_logL = (
                        max_log_below_null
                        + np.log(np.nanmean(probabilities_below_null))
                    )
                else:
                    log.warning(f"Only {below_null.size} samples in low-likelihood region; correction may be unreliable.")                    
                    log_initial_logL = null_evidence 

            # ========= Not adaptive truncated sampling =========
            # this is a safegard for the case we want the original sampling
            else:
                indices = list(range(self.params.num_dla_samples))
                batches = [
                    indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
                ]


            # ========= Select regions of high likelihood =========
            for num_dlas in range(max_dlas):  # Iterate from 0 to max_dlas - 1
                # Submit the tasks for each batch to the executor
                futures = {
                    executor.submit(
                        process_batch,
                        batch,
                        num_dlas,
                        sample_z_dlas,
                        base_sample_inds,
                        self.dla_samples,
                        self.params,
                        self.sample_log_likelihood_k_dlas,
                        self.min_z_separation,
                    ): batch
                    for batch in batches
                }

                # Process the results as each batch completes
                for future in as_completed(futures):
                    batch_indices = futures[future]
                    try:
                        batch_results = future.result()
                        # Store the results
                        for i, result in zip(batch_indices, batch_results):
                            sample_log_likelihoods[i, num_dlas] = result
                    except Exception as e:
                        log.error(f"Error in batch processing: {e}")

                # Handle NaN values and resampling logic
                if num_dlas > 0:
                    ind = base_sample_inds[:num_dlas, :]
                    all_z_dlas = np.concatenate(
                        [sample_z_dlas[None, :], sample_z_dlas[ind]], axis=0
                    )
                    ind = np.any(
                        np.diff(np.sort(all_z_dlas, axis=0), axis=0)
                        < self.min_z_separation,
                        axis=0,
                    )
                    sample_log_likelihoods[ind, num_dlas] = np.nan

                # Compute the log likelihood for each number of DLAs
                max_log_likelihood = np.nanmax(sample_log_likelihoods[:, num_dlas])
                sample_probabilities[:] = np.exp(
                    sample_log_likelihoods[:, num_dlas] - max_log_likelihood
                )
                # FILTER fix #5 (2026-04-29): the truncated-region bias correction
                # below produces unreliable values for the 1-DLA evidence when
                # the initial-scan valid_mask is small or degenerate (the failure
                # mode that produced p_DLA = 0.05 on TID 120046865 despite a
                # real DLA). For the **single-DLA evidence (num_dlas == 0)** we
                # therefore always use the unbiased initial-scan estimate (a
                # uniform prior sample of n_initial=5000, which is enough for
                # 1-DLA marginalization) rather than the truncated correction.
                # The truncated correction still applies for num_dlas >= 1 where
                # the dimensionality justifies the cost / variance tradeoff.
                if (filter_low_likelihood
                        and (null_evidence is not None)
                        and num_dlas == 0
                        and not np.all(np.isnan(initial_logL))):
                    initial_max_L = np.nanmax(initial_logL)
                    initial_probs = np.exp(initial_logL - initial_max_L)
                    log_likelihoods_dla[num_dlas] = (
                        initial_max_L + np.log(np.nanmean(initial_probs))
                    )
                    # Skip the multi-DLA truncated-correction branch — early-stop
                    # check below still applies normally.
                elif filter_low_likelihood and (null_evidence is not None):
                    # ===== Bias correction for truncated region using initial scan =====
                    # We are approximating the model evidence (log Z) by partitioning the sample space
                    # into two regions:
                    #   - Region A: retained samples with logL > null_evidence (fraction w)
                    #   - Region B: rejected samples with logL <= null_evidence (fraction 1 - w)
                    #
                    # The total marginal likelihood is:
                    #   Z ≈ w * Z_A + (1 - w) * Z_B
                    #     = w * mean(exp(logL_A)) + (1 - w) * mean(exp(logL_B))
                    #
                    # Taking log:
                    #   log Z ≈ log( w * exp(log_Z_A) + (1 - w) * exp(log_Z_B) )
                    #
                    # log_Z_A is estimated from the retained high-likelihood region:
                    log_Z_trunc = np.log(np.nanmean(sample_probabilities[_valid_mask])) + max_log_likelihood

                    # log_Z_B is approximated from the mean log-likelihood of the *rejected* region
                    # (e.g. those from the initial scan with logL < null_evidence), stored as log_initial_logL

                    # Compute total log evidence as a weighted log-sum-exp over A and B
                    eps = 1e-10
                    w = np.clip(_valid_mask.mean(), eps, 1 - eps)
                    log.info(
                        f"Fraction of prior retained: {w:.4f} for {num_dlas + 1} DLAs."
                    )
                    log_ratio = np.log(self.params.num_dla_samples) - np.log(n_initial)
                    log_likelihoods_dla[num_dlas] = (
                        logsumexp([
                            log_Z_trunc - log_ratio + np.log(w),
                            log_initial_logL + np.log(1 - w),
                        ])
                        - lognorm * num_dlas
                    )
                else:
                    # No truncation: standard marginal likelihood estimate from unweighted average
                    log_likelihoods_dla[num_dlas] = (
                        max_log_likelihood
                        + np.log(np.nanmean(sample_probabilities))
                        - lognorm * num_dlas
                    )

                # ========= Early stopping logic =========
                if (num_dlas + 1) == max_dlas or np.isnan(
                    log_likelihoods_dla[num_dlas]
                ):
                    break
                # If null_evidence is provided and the current log likelihood is less than it,
                # stop further computation
                if null_evidence is not None:
                    if log_likelihoods_dla[num_dlas] < null_evidence:
                        log.info(
                            f"Stopping early at {num_dlas + 1} DLAs because the log likelihood {log_likelihoods_dla[num_dlas]} is less than the null model evidence {null_evidence}."
                        )
                        break
                # If log likelihood is smaller than the previous one by 10 times,
                # stop further computation
                if num_dlas > 0:
                    if (
                        log_likelihoods_dla[num_dlas]
                        < log_likelihoods_dla[num_dlas - 1] #- 2.302585092994046 # log(10)
                    ):
                        log.info(
                            f"Stopping early at {num_dlas + 1} DLAs because the log likelihood {log_likelihoods_dla[num_dlas]} is less than the previous one."
                        )
                        break

                # Resampling logic to update base sample indices
                nanind = np.isnan(sample_probabilities)
                W = sample_probabilities
                W[nanind] = 0.0

                # resample the base sample indices using searchsorted method
                base_sample_inds[num_dlas, :] = searchsorted_method(
                    W,
                    self.params.num_dla_samples,
                )
                # base_sample_inds[num_dlas, :] = np.random.choice(
                #     np.arange(self.params.num_dla_samples).astype(np.int32),
                #     size=self.params.num_dla_samples,
                #     replace=True,
                #     p=W / W.sum(),
                # )

        finally:
            # Only shut down the executor if it was created locally
            if local_executor:
                executor.shutdown()

        # Store results for future use
        self.sample_log_likelihoods = sample_log_likelihoods
        self.base_sample_inds = base_sample_inds

        return log_likelihoods_dla

    def sample_log_likelihood_k_dlas(
        self, z_dlas: np.ndarray, nhis: np.ndarray
    ) -> float:
        """
        Compute the log likelihood of k DLAs within a quasar spectrum:
            p(y | λ, σ², M, ω, c₀, τ₀, β, τ_kim, β_kim, {z_dla, logNHI}_{i=1}^k)

        :param z_dlas: an array of z_dlas you want to condition on
        :param nhis: an array of nhis you want to condition on
        """
        assert len(z_dlas) == len(nhis)

        # Reuse or cache the Voigt profiles
        dla_mu, dla_M, dla_omega2 = self.this_dla_gp(z_dlas, nhis)

        # Compute the log-likelihood
        sample_log_likelihood = self.log_mvnpdf_low_rank(
            self.y, dla_mu, dla_M, dla_omega2 + self.v
        )

        return sample_log_likelihood

    def this_dla_gp(
        self, z_dlas: np.ndarray, nhis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the DLA GP model with k intervening DLA profiles onto
        the mean and covariance.

        :param z_dlas: (k_dlas, ), the redshifts of intervening DLAs
        :param nhis: (k_dlas, ), the column densities of intervening DLAs

        :return: (dla_mu, dla_M, dla_omega2)
        :return dla_mu: (n_points, ), the GP mean model with k_dlas DLAs intervening.
        :return dla_M: (n_points, k), the GP covariance with k_dlas DLAs intervening.
        :return dla_omega2: (n_points), the absorption noise with k_dlas DLAs intervening.S

        Note: the number of Voigt profile lines is controlled by self.params : Parameters,
        I prefer to not to allow users to change from the function arguments since that
        would easily cause inconsistent within a pipeline. But if a user want to change
        the num_lines, they can change via changing the instance attr of the self.params:Parameters
        like:
            self.params.num_lines = <the number of lines preferred to be used>
        This would happen when a user want to know whether the result would converge with increasing
        number of lines.
        """
        assert len(z_dlas) == len(nhis)

        k_dlas = len(z_dlas)

        # to retain only unmasked pixels from computed absorption profile
        mask_ind = ~self.pixel_mask[self.ind_unmasked]

        # [broadening] use the padded wavelengths for convolution
        # otherwise, should use unmasked wavelengths.
        if self.broadening:
            wavelengths = self.padded_wavelengths
        else:
            wavelengths = self.unmasked_wavelengths

        # Initialize the absorption profile for the DLA model
        absorption = np.ones(self.unmasked_wavelengths.shape[0])

        # Loop through each DLA and compute/reuse the Voigt profiles
        for j in range(k_dlas):
            # Create a unique cache key for the current (z_dla, nhi) pair
            cache_key = (z_dlas[j], nhis[j], self.broadening)

            if cache_key in self.voigt_cache:
                # Retrieve from cache if available
                cached_absorption = self.voigt_cache[cache_key]
            else:
                # Otherwise, compute the Voigt profile and store in cache
                cached_absorption = voigt_absorption(
                    wavelengths,
                    z_dla=z_dlas[j],
                    nhi=nhis[j],
                    num_lines=self.params.num_lines,
                )
                self.voigt_cache[cache_key] = cached_absorption

            # Multiply the absorption profiles for all DLAs
            absorption *= cached_absorption

        absorption = absorption[mask_ind]

        assert len(absorption) == len(self.this_mu)

        dla_mu = self.this_mu * absorption
        dla_M = self.this_M * absorption[:, None]
        dla_omega2 = self.this_omega2 * absorption**2

        return dla_mu, dla_M, dla_omega2

    def log_priors(self, z_qso: float, max_dlas: int) -> float:
        """
        get the model prior of null model, this is defined to be:
            P(k DLA | zQSO) = P(at least k DLAs | zQSO) - P(at least (k + 1) DLAs | zQSO),

        where

            P(at least 1 DLA | zQSO) = M / N

        M : number of DLAs below this zQSO
        N : number of quasars below this zQSO

        and

            P(at least k DLA | zQSO) = (M / N)^k

        Note: I did not overwrite the NullGP log prior, name of this method is log_prior's'
        for multi-DLAs
        """
        this_num_dlas, this_num_quasars = self.prior.less_ind(z_qso)

        p_dlas = (this_num_dlas / this_num_quasars) ** np.arange(1, max_dlas + 1)

        for i in range(max_dlas - 1):
            p_dlas[i] = p_dlas[i] - p_dlas[i + 1]

        log_priors_dla = np.log(p_dlas)

        return log_priors_dla

    def maximum_a_posteriori(self):
        """
        Find the maximum a posterior parameter pair {(z_dla, logNHI)}_{i=1}^k.

        :return (MAP_z_dla, MAP_log_nhi): shape for each is (max_dlas, max_dlas),
            the 0 dimension is for DLA(k) model and the 1 dimension is for
            the MAP estimates.
        """
        # maxinds = np.nanargmax(self.sample_log_likelihoods, axis=0)
        # Example array for demonstration; replace this with your actual array
        sample_log_likelihoods = self.sample_log_likelihoods

        # Identify columns that are not all NaNs
        valid_columns = ~np.isnan(sample_log_likelihoods).all(axis=0)

        # Apply np.nanargmax only on columns that are not all NaNs
        maxinds = np.full(
            sample_log_likelihoods.shape[1], None
        )  # Default to None for all-NaN columns
        if valid_columns.any():  # Ensure there are valid columns to process
            maxinds[valid_columns] = np.nanargmax(
                sample_log_likelihoods[:, valid_columns], axis=0
            )

        max_dlas = self.sample_log_likelihoods.shape[1]

        MAP_z_dla = np.empty((max_dlas, max_dlas))
        MAP_log_nhi = np.empty((max_dlas, max_dlas))
        MAP_z_dla[:] = np.nan
        MAP_log_nhi[:] = np.nan

        # prepare z_dla samples
        sample_z_dlas = self.dla_samples.sample_z_dlas(
            self.this_wavelengths, self.z_qso
        )

        for num_dlas, maxind in enumerate(maxinds):
            # skip if maxind is NaN
            if maxind is None:
                continue

            # store k MAP estimates for DLA(k) model
            if num_dlas > 0:
                # all_z_dlas : (num_dlas, num_dla_samples)
                ind = self.base_sample_inds[
                    :num_dlas, maxind
                ]  # (num_dlas - 1, num_dla_samples)

                MAP_z_dla[num_dlas, : (num_dlas + 1)] = np.concatenate(
                    [[sample_z_dlas[maxind]], sample_z_dlas[ind]]
                )  # (num_dlas, )
                MAP_log_nhi[num_dlas, : (num_dlas + 1)] = np.concatenate(
                    [
                        [self.dla_samples.log_nhi_samples[maxind]],
                        self.dla_samples.log_nhi_samples[ind],
                    ]
                )
            # for DLA(1) model, only store one MAP estimate
            else:
                MAP_z_dla[num_dlas, 0] = sample_z_dlas[maxind]
                MAP_log_nhi[num_dlas, 0] = self.dla_samples.log_nhi_samples[maxind]

        return MAP_z_dla, MAP_log_nhi


class DLAGPMAT(DLAGP):
    """
    Load learned model from .mat file
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        dla_samples: DLASamplesMAT,
        min_z_separation: float = 3000.0,
        learned_file: str = "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        broadening: bool = True,
        prev_tau_0: float = 0.0023,
        prev_beta: float = 3.65,
    ):
        # See NullGPMAT for the rationale: v2 trained .h5 carries its own
        # normalization region; mutate params in place if present so set_data
        # picks it up.
        with h5py.File(learned_file, "r") as learned:

            # Check if the learned model is DESI or not
            if learned["log_tau_0"].ndim == 0:
                print("DESI DLA model detected.")
                is_desi = True
            else:
                is_desi = False

            if is_desi is True:
                rest_wavelengths = learned["rest_wavelengths"][:]
                mu = learned["mu"][:]
                M = learned["M"][:]
                log_omega = learned["log_omega"][:]
                log_c_0 = learned["log_c_0"][()]
                log_tau_0 = learned["log_tau_0"][()]
                log_beta = learned["log_beta"][()]
            else:
                rest_wavelengths = learned["rest_wavelengths"][:, 0]
                mu = learned["mu"][:, 0]
                M = learned["M"][()].T
                log_omega = learned["log_omega"][:, 0]
                log_c_0 = learned["log_c_0"][0, 0]
                log_tau_0 = learned["log_tau_0"][0, 0]
                log_beta = learned["log_beta"][0, 0]

            if "normalization_min_lambda" in learned:
                new_min = float(learned["normalization_min_lambda"][()])
                new_max = float(learned["normalization_max_lambda"][()])
                params.normalization_min_lambda = new_min
                params.normalization_max_lambda = new_max

        super().__init__(
            params,
            prior,
            dla_samples,
            rest_wavelengths,
            mu,
            M,
            log_omega,
            log_c_0,
            log_tau_0,
            log_beta,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
            min_z_separation=min_z_separation,
            broadening=broadening,
        )
