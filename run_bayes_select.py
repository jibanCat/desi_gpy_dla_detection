"""
run_bayes_select.py

This script processes DESI spectra using Bayesian model selection for Damped Lyman-Alpha (DLA) detection.
"""

import os
import time
import numpy as np
import h5py
from typing import List
import argparse
from matplotlib import pyplot as plt

from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT
from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.subdla_samples import SubDLASamplesMAT
from gpy_dla_detection.bayesian_model_selection import BayesModelSelect
from gpy_dla_detection.desi_spectrum_reader import DESISpectrumReader
from gpy_dla_detection.process_helpers import (
    initialize_results,
    save_results_to_hdf5,
    _gzip_kwargs,
)
from gpy_dla_detection.plottings.plot_model import plot_samples_vs_this_mu

from gpy_dla_detection.compute_1sigma_errors import compute_1sigma_errors_fast

from desiutil.log import log


def process_single_spectrum(
    idx: int,
    target_id: str,
    z_qso: float,
    wavelengths: np.ndarray,
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
    noise_variance: np.ndarray,
    pixel_mask: np.ndarray,
    params: Parameters,
    prior: PriorCatalog,
    dla_samples: DLASamplesMAT,
    subdla_samples: SubDLASamplesMAT,
    bayes: BayesModelSelect,
    results: dict,
    max_dlas: int,
    broadening: bool,
    gp: NullGPMAT,  # Pre-initialized NullGPMAT
    dla_gp: DLAGPMAT,  # Pre-initialized DLAGPMAT
    subdla_gp: SubDLAGPMAT,  # Pre-initialized SubDLAGPMAT
    min_z_separation: float,
    plot_figures: bool,
    max_workers: int,
    batch_size: int,
    figure_dir: str,
    snr_blue: float = None,
    snr_red: float = None,
    filter_low_likelihood: bool = False,
    filter_n_initial_floor: int = 5000,
    filter_empty_mask_fallthrough: bool = False,
    single_absorber_model: bool = False,
):
    """
    Process a single spectrum using pre-initialized Null, DLA, and SubDLA models.

    Parameters:
    ----------
    idx : int
        Index of the spectrum being processed.
    target_id : str
        Identifier of the spectrum being processed.
    z_qso : float
        Redshift of the quasar for the spectrum.
    wavelengths : np.ndarray
        Observed wavelengths of the spectrum.
    rest_wavelengths : np.ndarray
        Rest-frame wavelengths of the spectrum.
    flux : np.ndarray
        Flux values of the spectrum.
    noise_variance : np.ndarray
        Noise variance per pixel in the spectrum.
    pixel_mask : np.ndarray
        Mask indicating which pixels are flagged as bad or good.
    params : Parameters
        Parameters instance for the analysis.
    prior : PriorCatalog
        Prior catalog instance for the analysis.
    dla_samples : DLASamplesMAT
        DLA samples data for the analysis.
    subdla_samples : SubDLASamplesMAT
        SubDLA samples data for the analysis.
    bayes : BayesModelSelect
        Bayesian model selection object for DLA detection.
    results : dict
        Dictionary to store the results.
    max_dlas : int
        Maximum number of DLAs to model.
    broadening : bool
        Whether to include instrumental broadening.
    gp : NullGPMAT
        Pre-initialized NullGPMAT object.
    dla_gp : DLAGPMAT
        Pre-initialized DLAGPMAT object.
    subdla_gp : SubDLAGPMAT
        Pre-initialized SubDLAGPMAT object.
    min_z_separation : float
        Minimum redshift separation for DLA models.
    plot_figures : bool
        If True, generates plots for each processed spectrum.
    max_workers : int
        Number of workers for parallel processing.
    batch_size : int
        Batch size for parallel model evidence computation.
    figure_dir : str
        Directory to save the figures.
    filter_low_likelihood : bool
        If True, filters out low likelihood samples during model evidence computation.
    single_absorber_model : bool
        If True, uses a single absorber model for DLA detection.
        That is, the DLA model includes NHI = [10^19.5, 10^22.5] cm^-2.
    """
    if single_absorber_model:
        # Set data for the Null and DLA models
        for model, name in zip([gp, dla_gp], ["Null", "DLA"]):
            model.set_data(
                rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
            )

        # Run Bayesian model selection with parallelized model evidence computation
        bayes.model_selection(
            [gp, dla_gp],
            z_qso,
            max_workers=max_workers,
            batch_size=batch_size,
            filter_low_likelihood=filter_low_likelihood,
            filter_n_initial_floor=filter_n_initial_floor,
            filter_empty_mask_fallthrough=filter_empty_mask_fallthrough,
        )
    else:
        # Set data for the Null, DLA, and Sub-DLA models
        for model, name in zip([gp, dla_gp, subdla_gp], ["Null", "DLA", "Sub-DLA"]):
            model.set_data(
                rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
            )

        # Run Bayesian model selection with parallelized model evidence computation
        bayes.model_selection(
            [gp, subdla_gp, dla_gp],
            z_qso,
            max_workers=max_workers,
            batch_size=batch_size,
            filter_low_likelihood=filter_low_likelihood,
            filter_n_initial_floor=filter_n_initial_floor,
            filter_empty_mask_fallthrough=filter_empty_mask_fallthrough,
        )

    # Store basic results
    results["z_qsos"][idx] = z_qso
    results["target_ids"][idx] = target_id
    results["min_z_dlas"][idx] = dla_gp.params.min_z_dla(wavelengths, z_qso)
    results["max_z_dlas"][idx] = dla_gp.params.max_z_dla(wavelengths, z_qso)
    results["log_priors_no_dla"][idx] = bayes.log_priors[0]
    results["log_priors_dla"][idx, :] = bayes.log_priors[-max_dlas:]
    results["log_likelihoods_no_dla"][idx] = bayes.log_likelihoods[0]
    results["log_likelihoods_dla"][idx, :] = bayes.log_likelihoods[-max_dlas:]
    results["log_posteriors_no_dla"][idx] = bayes.log_posteriors[0]
    results["log_posteriors_dla"][idx, :] = bayes.log_posteriors[-max_dlas:]

    # Store base sample indices (ensure this is set correctly in dla_gp)
    results["base_sample_inds"][idx, :, :] = dla_gp.base_sample_inds
    results["sample_log_likelihoods_dla"][idx, :, :] = dla_gp.sample_log_likelihoods

    # Save the DLA samples
    sample_z_dlas = dla_gp.dla_samples.sample_z_dlas(
        dla_gp.this_wavelengths, dla_gp.z_qso
    )
    # results["sample_z_dlas"][idx, :] = sample_z_dlas
    # results["log_nhi_samples"][idx, :] = dla_samples.log_nhi_samples

    # Obtain MAP estimates for z_DLA and log_NHI
    MAP_z_dla, MAP_log_nhi = dla_gp.maximum_a_posteriori()

    # Identify the most probable model
    # -----------------------------------------------------------------------
    # model_posteriors array layout
    # -----------------------------------------------------------------------
    # The length and index meaning depend on the run mode (single_absorber_model).
    #
    # DLA run (single_absorber_model=False) — length = 1 + 1 + max_dlas:
    #   index 0          → Null model  (no absorber)
    #   index 1          → Sub-DLA model  (log NHI in [19, 20.3])
    #   index 2          → 1-DLA model
    #   index 3          → 2-DLA model
    #   ...
    #   index 1+max_dlas → max_dlas-DLA model
    #
    # Sub-DLA / LLS run (single_absorber_model=True) — length = 1 + max_dlas:
    #   index 0          → Null model  (no absorber)
    #   index 1          → 1-absorber model
    #   index 2          → 2-absorber model  (usually disabled; max_dlas=1 in practice)
    #   ...
    #
    # argmaxind convention:
    #   when single_absorber_model=True:
    #     0  → no absorber detected (Null model most probable)
    #     k  → k absorbers detected
    #
    #   when single_absorber_model=False:
    #    -1  → Null model is most probable (np.nanargmax=0, minus 1 offset)
    #     0  → Sub-DLA model is most probable (treated as "no DLA detected")
    #     k>0 → k DLA absorbers
    #
    # Note: in the single_absorber_model=False path, index 0=Null and index
    # 1=Sub-DLA, so subtracting 1 makes the Null model map to argmaxind=-1
    # and the Sub-DLA model map to argmaxind=0. Both are treated as "no DLA
    # detected" for the purpose of saving MAP parameters.
    # -----------------------------------------------------------------------
    model_posteriors = bayes.model_posteriors[:]
    if single_absorber_model:
        argmaxind = np.nanargmax(model_posteriors)  # No absorber vs single absorber only
    else:
        argmaxind = np.nanargmax(model_posteriors) - 1  # offset: index 0=Null, 1=SubDLA, 2+=DLA(k)

    # Check if any DLA detection is made
    if argmaxind > 0:
        # Filter out NaNs in the MAP values
        MAP_z_dla = MAP_z_dla[argmaxind - 1, :argmaxind]
        MAP_log_nhi = MAP_log_nhi[argmaxind - 1, :argmaxind]

        # Compute 1-sigma errors using the fast method (Gaussian approximation)
        z_dla_errs, log_nhi_errs = compute_1sigma_errors_fast(
            MAP_z_dla,
            MAP_log_nhi,
            sample_z_dlas,
            dla_samples.log_nhi_samples,
            dla_gp.sample_log_likelihoods[:, 0],
        )

        # Save MAP estimates and associated 1-sigma errors
        results["MAP_z_dlas"][idx, :argmaxind] = MAP_z_dla
        results["MAP_log_nhis"][idx, :argmaxind] = MAP_log_nhi
        results["z_dla_errs"][idx, :argmaxind] = z_dla_errs
        results["log_nhi_errs"][idx, :argmaxind] = log_nhi_errs

    # Save posterior probabilities
    results["model_posteriors"][idx, :] = model_posteriors
    results["p_dlas"][idx] = bayes.p_dla
    results["p_no_dlas"][idx] = bayes.p_no_dla

    # Log the results
    log.info(
        f"Results for spectrum {idx + 1}/{len(results['z_qsos'])} (ID: {target_id})"
    )
    if argmaxind > 0:
        log.info(f" ...     MAP z_DLA: {MAP_z_dla}")
        log.info(f" ...     z_DLA errors: {z_dla_errs}")
        log.info(f" ...     MAP log N_HI: {MAP_log_nhi}")
        log.info(f" ...     log N_HI errors: {log_nhi_errs}")
    # log.info(f" ...     Model posteriors: {model_posteriors}")
    log.info(f" ...     p(DLA): {bayes.p_dla:.3f}")
    log.info(f" ...     p(no DLAs): {bayes.p_no_dla:.3f}")

    # Generate plots if enabled
    if plot_figures:
        title = f"Spectrum {target_id}; zQSO: {z_qso:.2f}"
        out_filename = f"spec-{target_id}-zqso-{z_qso:.2f}"
        plot_samples_vs_this_mu(
            dla_gp, bayes, filename=out_filename, sub_dir=figure_dir, title=title
        )
        plt.clf()
        plt.close()


class DLAHolder:
    """
    Class to handle Bayesian model selection for Damped Lyman-Alpha (DLA) system detection in DESI spectra.

    Parameters:
    ----------
    num_spectra : int
        Number of spectra to process.
    learned_file : str
        Learned QSO model file path.
    catalog_name : str
        Catalog file path.
    los_catalog : str
        Line-of-sight catalog file path.
    dla_catalog : str
        DLA catalog file path.
    dla_samples_file : str
        DLA samples file path.
    sub_dla_samples_file : str
        Sub-DLA samples file path.
    params : Parameters
        Parameters object containing various settings and hyperparameters.
    min_z_separation : float
        Minimum redshift separation between DLAs.
    prev_tau_0 : float
        Previous value of the DLA optical depth.
    prev_beta : float
        Previous value of the DLA power-law index.
    max_dlas : int, optional
        Maximum number of DLAs to consider per spectrum (default is 4).
    broadening : bool, optional
        Flag indicating whether to apply broadening to the DLA profiles (default is True).
    plot_figures : bool, optional
        Flag indicating whether to plot diagnostic figures during processing (default is False).
    max_workers : int, optional
        Maximum number of parallel workers to use for processing (default is None).
    batch_size : int, optional
        Batch size for parallel processing (default is 100).
    single_absorber_model : bool, optional
        If True, uses a single absorber model for DLA detection.
        That is, the DLA model includes NHI = [10^19.5, 10^22.5] cm^-2.
    """

    def __init__(
        self,
        learned_file: str,
        catalog_name: str,
        los_catalog: str,
        dla_catalog: str,
        dla_samples_file: str,
        sub_dla_samples_file: str,
        params: Parameters,
        min_z_separation: float,
        prev_tau_0: float,
        prev_beta: float,
        max_dlas: int = 3,
        broadening: bool = True,
        plot_figures: bool = False,
        max_workers: int = None,
        batch_size: int = 100,
        figure_dir: str = "figures/",
        params_subdla=None,
        filter_low_likelihood: bool = False,
        filter_n_initial_floor: int = 5000,
        filter_empty_mask_fallthrough: bool = False,
        single_absorber_model: bool = False,
        enable_tau_eb: bool = False,
        tau_eb_factors: tuple = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
        tau_eb_apply_hcd_mask: bool = False,
        tau_eb_mask_threshold_sigma: float = 1.5,
        tau_eb_objective: str = "null",
        early_stop_mode: str = "baseline",
    ):
        """
        Initialize the DLAProcessor class with necessary data files and parameters.

        Additional parameters
        ---------------------
        enable_tau_eb : bool, default False
            If True, run the per-spectrum empirical-Bayes τ_0 fit (see
            ``gpy_dla_detection/tau_eb.py`` and ``docs/tau_eb_hcd_mask.md``)
            and use the chosen τ_0 in place of ``prev_tau_0`` for the
            production inference. At n=90 mock DLA targets the recipe drops
            median bias from +0.135 → +0.026 dex (no HCD mask) or −0.131 dex
            (with HCD mask, default OFF). See
            ``docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md``.
        tau_eb_factors : tuple of float
            τ-grid for the EB scan: candidate τ_0 = factor * prev_tau_0.
            Default (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0) covers the
            range observed in mock validation.
        tau_eb_apply_hcd_mask : bool, default False
            If True, mask pixels with negative residual < -N σ during the
            τ-fit step. Default OFF — at scale this *over-corrects* the
            τ-fit. Keep True only for saturated-DLA-dominated targets.
        tau_eb_mask_threshold_sigma : float
            HCD-flag threshold; only used when tau_eb_apply_hcd_mask=True.
        tau_eb_objective : {"null", "dla"}
            "null" (default, cheap): fit τ on null-model log evidence.
            "dla": match the validated diagnostic at higher cost.
        """

        self.learned_file = learned_file
        self.catalog_name = catalog_name
        self.los_catalog = los_catalog
        self.dla_catalog = dla_catalog
        self.dla_samples_file = dla_samples_file
        self.sub_dla_samples_file = sub_dla_samples_file
        self.min_z_separation = min_z_separation
        self.prev_tau_0 = prev_tau_0
        self.prev_beta = prev_beta
        self.max_dlas = max_dlas
        self.broadening = broadening
        self.plot_figures = plot_figures
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Filter low likelihood samples
        self.filter_low_likelihood = filter_low_likelihood
        # FILTER=1 knobs (see docs/notes/2026-05-13_filter1_knob_tuning.md).
        # Defaults reproduce historical behavior:
        #   n_initial = max(num_dla_samples // 20, 5000)
        #   empty valid_mask → early-stop with 1-DLA evidence from coarse samples
        self.filter_n_initial_floor = int(filter_n_initial_floor)
        self.filter_empty_mask_fallthrough = bool(filter_empty_mask_fallthrough)

        # Single absorber model flag: No Sub-DLA model, only Null and DLA models
        self.single_absorber_model = single_absorber_model

        # τ-EB knobs (see gpy_dla_detection.tau_eb.fit_tau_eb).
        self.enable_tau_eb = enable_tau_eb
        self.tau_eb_factors = tuple(tau_eb_factors)
        self.tau_eb_apply_hcd_mask = bool(tau_eb_apply_hcd_mask)
        self.tau_eb_mask_threshold_sigma = float(tau_eb_mask_threshold_sigma)
        self.tau_eb_objective = tau_eb_objective

        # Multi-DLA early-stop policy (see DLAGP for documentation).
        # See docs/notes/2026-05-12_multidla_early_stop_bug.md.
        if early_stop_mode not in ("baseline", "A", "D"):
            raise ValueError(
                f"early_stop_mode must be one of 'baseline', 'A', 'D'; got {early_stop_mode!r}"
            )
        self.early_stop_mode = early_stop_mode

        self.params = params  # Pass in the Parameters object here
        if params_subdla is None:
            params_subdla = params.copy() # Use the same parameters for Sub-DLA
        self.params_subdla = params_subdla

        # Initialize prior catalog and Bayesian model selection
        self.prior = PriorCatalog(self.params, catalog_name, los_catalog, dla_catalog)
        self.dla_samples = DLASamplesMAT(self.params, self.prior, dla_samples_file)

        if not self.single_absorber_model:
            self.subdla_samples = SubDLASamplesMAT(
                self.params_subdla, self.prior, sub_dla_samples_file
            )
        else:
            self.subdla_samples = None
        # self.bayes = BayesModelSelect([0, 1, max_dlas], 2)

        self.figure_dir = figure_dir

    def initialize_results(self, num_spectra: int):
        """
        Initialize the results dictionary
        """
        self.results = initialize_results(
            num_spectra,
            self.max_dlas,
            self.params.num_dla_samples,
            single_absorber_model=self.single_absorber_model,
        )
        self.num_spectra = num_spectra

    def process_qso(
        self,
        idx,
        target_id,
        wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso,
    ):
        """
        Process all spectra in the DESI file.

        idx : int
            Index of the spectrum being processed.
        """
        tic = time.time()

        rest_wavelengths = self.params.emitted_wavelengths(wavelengths, z_qso)

        # Optional per-spectrum empirical-Bayes τ_0 fit
        # (see gpy_dla_detection/tau_eb.py + docs/tau_eb_hcd_mask.md +
        # docs/notes/2026-04-29_tau_eb_n90_unbiasedness.md).
        # When enabled, this replaces ``self.prev_tau_0`` for THIS spectrum
        # only; the rest of the inference is unchanged.
        prev_tau_0_eff = self.prev_tau_0
        if self.enable_tau_eb:
            from gpy_dla_detection.tau_eb import fit_tau_eb
            prev_tau_0_eff, tau_eb_info = fit_tau_eb(
                params=self.params,
                prior=self.prior,
                learned_file=self.learned_file,
                rest_wavelengths=rest_wavelengths,
                flux=flux,
                noise_variance=noise_variance,
                pixel_mask=pixel_mask,
                z_qso=z_qso,
                prev_tau_0_seed=self.prev_tau_0,
                prev_beta=self.prev_beta,
                tau_factors=self.tau_eb_factors,
                apply_hcd_mask=self.tau_eb_apply_hcd_mask,
                mask_threshold_sigma=self.tau_eb_mask_threshold_sigma,
                objective=self.tau_eb_objective,
                dla_samples=self.dla_samples,
            )
            log.info(
                f" ...     τ-EB[{self.tau_eb_objective}, "
                f"hcd_mask={self.tau_eb_apply_hcd_mask}]: "
                f"factor_best={tau_eb_info['tau_factor_best']:.2f}  "
                f"τ_0={prev_tau_0_eff:.5f}  "
                f"n_hcd={tau_eb_info['n_hcd']}"
            )

        # Initialize the Null and DLA models for this spectrum
        null_gp = NullGPMAT(
            self.params,
            self.prior,
            learned_file=self.learned_file,
            prev_tau_0=prev_tau_0_eff,
            prev_beta=self.prev_beta,
        )
        dla_gp = DLAGPMAT(
            self.params,
            self.prior,
            self.dla_samples,
            min_z_separation=self.min_z_separation,
            learned_file=self.learned_file,
            broadening=self.broadening,
            prev_tau_0=prev_tau_0_eff,
            prev_beta=self.prev_beta,
            early_stop_mode=self.early_stop_mode,
        )
        if self.single_absorber_model:
            subdla_gp = None
            bayes = BayesModelSelect([0, self.max_dlas], 1)
        else:
            subdla_gp = SubDLAGPMAT(
                self.params_subdla,
                self.prior,
                self.subdla_samples,
                min_z_separation=self.min_z_separation,
                learned_file=self.learned_file,
                broadening=self.broadening,
                prev_tau_0=prev_tau_0_eff,
                prev_beta=self.prev_beta,
            )
            bayes = BayesModelSelect([0, 1, self.max_dlas], 2)

        # Log the processing of the spectrum
        log.info(
            f"Processing spectrum {idx + 1}/{self.num_spectra} (ID: {target_id}) zQSO: {z_qso:.2f}"
        )
        # Process single spectrum
        process_single_spectrum(
            idx,
            target_id,
            z_qso,
            wavelengths,
            rest_wavelengths,
            flux,
            noise_variance,
            pixel_mask,
            self.params,
            self.prior,
            self.dla_samples,
            self.subdla_samples,
            bayes,
            self.results,
            self.max_dlas,
            self.broadening,
            null_gp,
            dla_gp,
            subdla_gp,
            self.min_z_separation,
            self.plot_figures,
            self.max_workers,
            self.batch_size,
            self.figure_dir,
            filter_low_likelihood=self.filter_low_likelihood,
            filter_n_initial_floor=self.filter_n_initial_floor,
            filter_empty_mask_fallthrough=self.filter_empty_mask_fallthrough,
            single_absorber_model=self.single_absorber_model,
        )
        # Clean up to free memory
        if self.single_absorber_model:
            del null_gp, dla_gp
        else:
            del null_gp, dla_gp, subdla_gp

        toc = time.time()
        log.info(
            f"Processed spectrum {idx + 1}/{self.num_spectra} (ID: {target_id}), time spent: {(toc - tic) // 60:.0f}m {(toc - tic) % 60:.0f}s"
        )

    def save_results(self, output_file: str):

        # Save results to HDF5 file (gzip — the per-sample arrays are mostly
        # NaN fill, so this is ~15-25x smaller and lossless; see _gzip_kwargs)
        with h5py.File(output_file, "w") as f:
            # Loop through the results dictionary and save each key-value pair as an HDF5 dataset
            for key, value in self.results.items():
                f.create_dataset(
                    key, data=value, **_gzip_kwargs(value)
                )  # Save each result in the HDF5 file (gzip-compressed)


class DLAProcessor:
    """
    Class to handle Bayesian model selection for Damped Lyman-Alpha (DLA) system detection in DESI spectra.

    Parameters:
    ----------
    spectra_filename : str
        DESI spectra FITS filename.
    zbest_filename : str
        DESI redshift catalog filename.
    learned_file : str
        Learned QSO model file path.
    catalog_name : str
        Catalog file path.
    los_catalog : str
        Line-of-sight catalog file path.
    dla_catalog : str
        DLA catalog file path.
    dla_samples_file : str
        DLA samples file path.
    sub_dla_samples_file : str
        Sub-DLA samples file path.
    params : Parameters
        Parameters object containing various settings and hyperparameters.
    min_z_separation : float
        Minimum redshift separation between DLAs.
    prev_tau_0 : float
        Previous value of the DLA optical depth.
    prev_beta : float
        Previous value of the DLA power-law index.
    max_dlas : int, optional
        Maximum number of DLAs to consider per spectrum (default is 4).
    broadening : bool, optional
        Flag indicating whether to apply broadening to the DLA profiles (default is True).
    plot_figures : bool, optional
        Flag indicating whether to plot diagnostic figures during processing (default is False).
    max_workers : int, optional
        Maximum number of parallel workers to use for processing (default is None).
    batch_size : int, optional
        Batch size for parallel processing (default is 100).
    """

    def __init__(
        self,
        spectra_filename: str,
        zbest_filename: str,
        learned_file: str,
        catalog_name: str,
        los_catalog: str,
        dla_catalog: str,
        dla_samples_file: str,
        sub_dla_samples_file: str,
        params: Parameters,
        min_z_separation: float,
        prev_tau_0: float,
        prev_beta: float,
        max_dlas: int = 4,
        broadening: bool = True,
        plot_figures: bool = False,
        max_workers: int = None,
        batch_size: int = 100,
    ):
        """
        Initialize the DLAProcessor class with necessary data files and parameters.
        """

        self.spectra_filename = spectra_filename
        self.zbest_filename = zbest_filename
        self.learned_file = learned_file
        self.catalog_name = catalog_name
        self.los_catalog = los_catalog
        self.dla_catalog = dla_catalog
        self.dla_samples_file = dla_samples_file
        self.sub_dla_samples_file = sub_dla_samples_file
        self.min_z_separation = min_z_separation
        self.prev_tau_0 = prev_tau_0
        self.prev_beta = prev_beta
        self.max_dlas = max_dlas
        self.broadening = broadening
        self.plot_figures = plot_figures
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.params = params  # Pass in the Parameters object here

        # Initialize prior catalog and Bayesian model selection
        self.prior = PriorCatalog(self.params, catalog_name, los_catalog, dla_catalog)
        self.dla_samples = DLASamplesMAT(self.params, self.prior, dla_samples_file)
        self.subdla_samples = SubDLASamplesMAT(
            self.params, self.prior, sub_dla_samples_file
        )
        self.bayes = BayesModelSelect([0, 1, max_dlas], 2)

        # Initialize reader for DESI spectra
        self.reader = DESISpectrumReader(spectra_filename, zbest_filename)
        self.reader.read_spectra()
        self.reader.read_redshift_catalog()
        self.redshift_data = self.reader.get_redshift_data()
        self.all_spectrum_ids = self.reader.get_all_spectrum_ids()
        self.results = initialize_results(
            len(self.all_spectrum_ids), max_dlas, self.params.num_dla_samples
        )

    def process_all_spectra(self):
        """
        Process all spectra in the DESI file.
        """
        for idx, spectrum_id in enumerate(self.all_spectrum_ids):
            tic = time.time()

            z_qso = self.redshift_data["Z"][idx]
            spectrum_data = self.reader.get_spectrum_data(spectrum_id)
            wavelengths = spectrum_data.wavelengths
            flux = spectrum_data.flux
            noise_variance = spectrum_data.noise_variance
            pixel_mask = spectrum_data.pixel_mask

            rest_wavelengths = self.params.emitted_wavelengths(wavelengths, z_qso)

            # Initialize the Null and DLA models for this spectrum
            null_gp = NullGPMAT(
                self.params,
                self.prior,
                learned_file=self.learned_file,
                prev_tau_0=self.prev_tau_0,
                prev_beta=self.prev_beta,
            )
            dla_gp = DLAGPMAT(
                self.params,
                self.prior,
                self.dla_samples,
                min_z_separation=self.min_z_separation,
                learned_file=self.learned_file,
                broadening=self.broadening,
                prev_tau_0=self.prev_tau_0,
                prev_beta=self.prev_beta,
            )
            subdla_gp = SubDLAGPMAT(
                self.params,
                self.prior,
                self.subdla_samples,
                min_z_separation=self.min_z_separation,
                learned_file=self.learned_file,
                broadening=self.broadening,
                prev_tau_0=self.prev_tau_0,
                prev_beta=self.prev_beta,
            )

            # Process single spectrum
            process_single_spectrum(
                idx,
                spectrum_id,
                z_qso,
                wavelengths,
                rest_wavelengths,
                flux,
                noise_variance,
                pixel_mask,
                self.params,
                self.prior,
                self.dla_samples,
                self.subdla_samples,
                self.bayes,
                self.results,
                self.max_dlas,
                self.broadening,
                null_gp,
                dla_gp,
                subdla_gp,
                self.min_z_separation,
                self.plot_figures,
                self.max_workers,
                self.batch_size,
            )

            toc = time.time()
            print(
                f"Processed spectrum {idx + 1}/{len(self.all_spectrum_ids)} (ID: {spectrum_id}), time spent: {(toc - tic) // 60:.0f}m {(toc - tic) % 60:.0f}s"
            )

        # Save results to HDF5 file
        save_results_to_hdf5(
            "processed_desi_spectra.h5",
            self.results,
            self.all_spectrum_ids,
            self.redshift_data["Z"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DESI spectra with Bayesian model selection for DLA detection."
    )

    # Spectra and file-related arguments
    parser.add_argument(
        "--spectra_filename",
        required=True,
        help="DESI spectra FITS filename (e.g., spectra-*.fits).",
    )
    parser.add_argument(
        "--zbest_filename",
        required=True,
        help="DESI redshift catalog filename (zbest-*.fits).",
    )
    parser.add_argument(
        "--learned_file",
        default="data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        help="Learned QSO model file path.",
    )
    parser.add_argument(
        "--catalog_name",
        default="data/dr12q/processed/catalog.mat",
        help="Catalog file path.",
    )
    parser.add_argument(
        "--los_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        help="Line-of-sight catalog file path.",
    )
    parser.add_argument(
        "--dla_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        help="DLA catalog file path.",
    )
    parser.add_argument(
        "--dla_samples_file",
        default="data/dr12q/processed/dla_samples_a03.mat",
        help="DLA samples file path.",
    )
    parser.add_argument(
        "--sub_dla_samples_file",
        default="data/dr12q/processed/subdla_samples.mat",
        help="Sub-DLA samples file path.",
    )

    # DLA-related arguments
    parser.add_argument(
        "--min_z_separation",
        type=float,
        default=3000.0,
        help="Minimum redshift separation for DLA models.",
    )
    parser.add_argument(
        # Turner+2024 (DESI Y1) mean-flux defaults. Were 0.00554/3.182 (Kamble+2020) —
        # a mismatched default for the Turner-trained DESI models; production always
        # overrides via config (PREV_TAU_0/PREV_BETA), so this is a safety fix only.
        "--prev_tau_0", type=float, default=0.00246, help="Previous value for tau_0 (Turner+2024)."
    )
    parser.add_argument(
        "--prev_beta", type=float, default=3.62, help="Previous value for beta (Turner+2024)."
    )
    parser.add_argument(
        "--max_dlas", type=int, default=3, help="Maximum number of DLAs to model."
    )
    parser.add_argument(
        "--plot_figures",
        type=int,
        default=0,
        help="Set to 1 to generate plots, 0 otherwise.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Number of workers for parallel processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=313,
        help="Batch size for parallel model evidence computation.",
    )

    # Parameter-related arguments
    # These are the values used in the trained GP model, don't change them unless you change the trained model
    parser.add_argument(
        "--loading_min_lambda",
        type=float,
        default=800,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--loading_max_lambda",
        type=float,
        default=1550,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--normalization_min_lambda",
        type=float,
        default=1425,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--normalization_max_lambda",
        type=float,
        default=1475,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--min_lambda",
        type=float,
        default=850.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=1420.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--dlambda", type=float, default=0.25, help="Separation of wavelength grid (Å)."
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Rank of non-diagonal contribution."
    )
    parser.add_argument(
        "--max_noise_variance",
        type=float,
        default=9,
        help="Maximum pixel noise allowed during model training.",
    )

    args = parser.parse_args()

    # Initialize Parameters object with user inputs
    params = Parameters(
        loading_min_lambda=args.loading_min_lambda,
        loading_max_lambda=args.loading_max_lambda,
        normalization_min_lambda=args.normalization_min_lambda,
        normalization_max_lambda=args.normalization_max_lambda,
        min_lambda=args.min_lambda,
        max_lambda=args.max_lambda,
        dlambda=args.dlambda,
        k=args.k,
        max_noise_variance=args.max_noise_variance,
    )

    processor = DLAProcessor(
        spectra_filename=args.spectra_filename,
        zbest_filename=args.zbest_filename,
        learned_file=args.learned_file,
        catalog_name=args.catalog_name,
        los_catalog=args.los_catalog,
        dla_catalog=args.dla_catalog,
        dla_samples_file=args.dla_samples_file,
        sub_dla_samples_file=args.sub_dla_samples_file,
        params=params,
        min_z_separation=args.min_z_separation,
        prev_tau_0=args.prev_tau_0,
        prev_beta=args.prev_beta,
        max_dlas=args.max_dlas,
        plot_figures=bool(args.plot_figures),
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )

    processor.process_all_spectra()
