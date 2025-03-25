#!/usr/bin/env python
"""
plot_visual_inspect.py
======================
This script performs a visual inspection of DESI spectra by reading the quasar catalog,
loading CNN/TEMP DLA finder results, and then processing the spectrum with two different
DLA detection models (eBOSS and DESI trained). It produces several plots (observed spectrum,
zoomed Lyman-α, and comparisons of CNN/TEMP/GP results) and saves the outputs in a structured
folder hierarchy. The script accepts command-line arguments to easily loop over multiple TARGETIDs.

Usage:
    python plot_visual_inspect.py --catalog /path/to/catalog.fits --output_dir output \
         --tid_list 123456,789012 --release kibo --survey main --program dark

Author: [Your Name]
Date: [Date]
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.table import Table, vstack
import fitsio
from collections import namedtuple
from scipy.interpolate import interp1d

# DESI-related imports
import desispec.io
from desispec.interpolation import resample_flux
from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log
from desiutil.log import log
import constants

# DLA detection imports
from run_bayes_select import process_single_spectrum
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.subdla_samples import SubDLASamplesMAT
from gpy_dla_detection.bayesian_model_selection import BayesModelSelect
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT
from gpy_dla_detection.process_helpers import initialize_results
from gpy_dla_detection.voigt import voigt_absorption


def read_catalog(qsocat, balmask, bytile):
    """
    Read the quasar catalog from a FITS file and apply redshift, BAL, ZWARN, and spectype cuts.

    Parameters
    ----------
    qsocat : str
        Path to the quasar catalog file.
    balmask : bool
        Whether to read BAL attributes.
    bytile : bool
        If the catalog is tile-based.

    Returns
    -------
    catalog : astropy.table.Table
        Filtered catalog table.
    """
    if constants.no_bal:
        balmask = True

    if balmask:
        try:
            cols = [
                "TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "HPXPIXEL",
                "AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450",
                "SPECTYPE", "ZWARN",
            ]
            if bytile:
                cols = [
                    "TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "TILEID", "PETAL_LOC",
                    "AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450",
                    "SPECTYPE", "ZWARN",
                ]
            catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
        except Exception as e:
            log.error(f"Error reading catalog columns: {e}")
            sys.exit(1)
    else:
        cols = [
            "TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "HPXPIXEL",
            "SPECTYPE", "ZWARN",
        ]
        if bytile:
            cols = [
                "TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "TILEID", "PETAL_LOC",
                "SPECTYPE", "ZWARN",
            ]
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    log.info(f"Successfully read quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"Catalog objects: {len(catalog)}; After redshift cuts: {np.sum(zmask)}")

    # Apply BAL mask if necessary
    if constants.no_bal:
        balind = catalog["NCIV_450"] > 0
        zmask = zmask & ~balind
        log.info(f"Objects without BAL: {np.sum(zmask)}")

    # Apply ZWARN mask
    if constants.zwarning:
        zmask = zmask & (catalog["ZWARN"] == 0)
        log.info(f"Objects with ZWARN=0: {np.sum(zmask)}")

    # Apply spectype mask
    if constants.is_qso:
        zmask = zmask & (catalog["SPECTYPE"] == "QSO")
        log.info(f"Objects with SPECTYPE QSO: {np.sum(zmask)}")

    return catalog[zmask]


def load_cnn_temp_results():
    """
    Load the CNN/TEMP DLA finder catalogs and apply SNR, BAL, and NHI cuts.

    Returns
    -------
    mollycat : astropy.table.Table
        Raw combined catalog.
    mollycat_gp : astropy.table.Table
        Catalog with overlapping DLAs from GP.
    target_ids : numpy.ndarray
        Unique TARGETIDs from the GP catalog that pass the cuts.
    """
    # Hard-coded paths; adjust if necessary
    mollycat_path = "/global/cfs/cdirs/desi/users/mwolfson/DLA_cat/loa_combined_cat_raw.fits"
    mollycat_gp_path = "/global/cfs/cdirs/desi/users/mwolfson/DLA_cat/loa_dla_cat_close_gp_bal_col.fits"
    mollycat = Table.read(mollycat_path)
    mollycat_gp = Table.read(mollycat_gp_path)

    # Apply cuts for a cleaner plot
    snr = mollycat_gp["SNR_REDSIDE"]
    bal_mask_arr = mollycat_gp["AI_CIV"]
    ind = (snr > 15) & (bal_mask_arr == 0) & (mollycat_gp["NHI"] > 20.3)
    target_ids = np.unique(mollycat_gp[ind]["TARGETID"])
    print("CNN/TEMP GP catalog selection:")
    print(mollycat_gp[ind])
    return mollycat, mollycat_gp, target_ids


def load_spectrum(catalog, tid, release, survey, program):
    """
    Read the DESI spectrum for a given TARGETID.

    Parameters
    ----------
    catalog : astropy.table.Table
        Quasar catalog.
    tid : int or str
        TARGETID to process.
    release : str
        Data release identifier.
    survey : str
        Survey name.
    program : str
        Program name (e.g., dark, bright).

    Returns
    -------
    specobj : desispec.io.spectra.Spectra
        The spectrum object read from file.
    idx : int
        Index of the TARGETID in the catalog.
    hpx : int
        Healpix pixel corresponding to the TARGETID.
    """
    try:
        idx = np.where(catalog["TARGETID"] == tid)[0][0]
    except IndexError:
        log.error(f"TARGETID {tid} not found in catalog!")
        raise

    print("Selected object from catalog:")
    print(catalog[idx])
    hpx = catalog[idx]["HPXPIXEL"]
    datapath = f"/global/cfs/cdirs/desi/spectro/redux/{release}/healpix/{survey}/{program}"
    coaddname = f"coadd-{survey}-{program}-{str(hpx)}.fits"
    coadd = os.path.join(datapath, str(hpx // 100), str(hpx), coaddname)
    hpxcatalog = catalog[catalog["HPXPIXEL"] == hpx]
    specobj = desispec.io.read_spectra(
        coadd,
        targetids=hpxcatalog["TARGETID"],
        skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"],
    )
    specobj = coadd_cameras(specobj)
    return specobj, idx, hpx


def extract_spectrum_data(specobj, catalog, tid, idx):
    """
    Extract wavelength, flux, noise variance, and pixel mask for the given TARGETID.

    Parameters
    ----------
    specobj : desispec.io.spectra.Spectra
        Spectrum object.
    catalog : astropy.table.Table
        Quasar catalog.
    tid : int or str
        TARGETID.
    idx : int
        Index of the TARGETID in the catalog.

    Returns
    -------
    spectrum_data : namedtuple
        Contains wavelengths, flux, noise_variance, and pixel_mask.
    z_qso : float
        Redshift of the quasar.
    """
    SpectrumData = namedtuple("SpectrumData", ["wavelengths", "flux", "noise_variance", "pixel_mask"])
    z_qso = catalog[idx]["Z"]

    # Find the index within the healpix grouping corresponding to tid
    this_cat = catalog["TARGETID"][catalog["HPXPIXEL"] == catalog[idx]["HPXPIXEL"]]
    this_idx = np.where(this_cat == tid)[0][0]

    ivar = specobj.ivar["brz"][this_idx]
    noise_variance = np.full_like(ivar, np.nan, dtype=float)
    valid = ivar != 0
    noise_variance[valid] = 1 / ivar[valid]
    pixel_mask = specobj.mask["brz"][this_idx].astype(bool)
    pixel_mask[~valid] = True

    spectrum_data = SpectrumData(
        wavelengths=specobj.wave["brz"],
        flux=specobj.flux["brz"][this_idx],
        noise_variance=noise_variance,
        pixel_mask=pixel_mask
    )
    return spectrum_data, z_qso


def save_plot(fig, filename):
    """
    Save a matplotlib figure to the specified filename and then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Path to save the figure.
    """
    fig.savefig(filename, format=os.path.splitext(filename)[1][1:], dpi=150)
    plt.close(fig)


def plot_observed_spectrum(wavelengths, flux, tid, z_qso, out_dir):
    """
    Plot and save the spectrum in observed wavelengths.

    Parameters
    ----------
    wavelengths : array_like
        Observed wavelengths.
    flux : array_like
        Flux values.
    tid : int or str
        TARGETID.
    z_qso : float
        Redshift.
    out_dir : str
        Output directory where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(wavelengths, flux)
    ax.set_xlabel("Observed Wavelengths [$\AA$]")
    ax.set_ylabel("Flux")
    ax.set_title(f"Spectrum {tid} in Observed Wavelengths (z = {z_qso:.2f})")
    ax.grid(True)
    save_plot(fig, os.path.join(out_dir, "spectrum_observed.png"))


def plot_zoomed_lya(rest_wavelengths, flux, tid, out_dir):
    """
    Plot and save the normalized spectrum zoomed into the Lyman-alpha region.

    Parameters
    ----------
    rest_wavelengths : array_like
        Rest-frame wavelengths.
    flux : array_like
        Flux values.
    tid : int or str
        TARGETID.
    out_dir : str
        Output directory where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(rest_wavelengths, flux / np.mean(flux))
    ax.set_xlabel("Rest-frame Wavelengths [$\AA$]")
    ax.set_ylabel("Normalized Flux")
    ax.set_ylim(-1, 5)
    ax.set_xlim(750, 1415)
    ax.set_title(f"Zoomed-in Lya Region for Spectrum {tid}")
    ax.grid(True)

    # Emission line markers for Lyα, Lyβ, and Lyman Limit
    lya_wavelength = 1215.24
    lyb_wavelength = 1025.72
    ly_limit_wavelength = 911.76
    ax.axvline(lya_wavelength, color="C3", ls="--")
    ax.text(lya_wavelength, 4, "Lyα", rotation="vertical", color="C3")
    ax.axvline(lyb_wavelength, color="C2", ls="--")
    ax.text(lyb_wavelength, 4, "Lyβ", rotation="vertical", color="C2")
    ax.axvline(ly_limit_wavelength, color="C1", ls="--")
    ax.text(ly_limit_wavelength, 4, "Lyman Limit", rotation="vertical", color="C1")
    save_plot(fig, os.path.join(out_dir, "spectrum_zoomed.png"))


class SpectrumProcessor:
    """
    Class for processing a single spectrum with DLA detection models.

    This class initializes parameters, priors, and Gaussian Process models,
    and then processes the spectrum via Bayesian model selection.
    """
    def __init__(self, spectra_filename, zbest_filename, learned_file, catalog_name,
                 los_catalog, dla_catalog, dla_samples_file, sub_dla_samples_file,
                 max_dlas=3, min_z_separation=3000.0, prev_tau_0=0.00554, prev_beta=3.182,
                 k=20, dlambda=0.25, min_lambda=912.75, max_lambda=1216.75):
        self.spectra_filename = spectra_filename
        self.zbest_filename = zbest_filename
        self.learned_file = learned_file
        self.catalog_name = catalog_name
        self.los_catalog = los_catalog
        self.dla_catalog = dla_catalog
        self.dla_samples_file = dla_samples_file
        self.sub_dla_samples_file = sub_dla_samples_file
        self.max_dlas = max_dlas
        self.min_z_separation = min_z_separation
        self.prev_tau_0 = prev_tau_0
        self.prev_beta = prev_beta

        # Initialize parameters for loading and normalization
        self.params = Parameters(
            loading_min_lambda=910,
            loading_max_lambda=1550,
            normalization_min_lambda=1425,
            normalization_max_lambda=1475,
            min_lambda=min_lambda,
            max_lambda=max_lambda,
            dlambda=dlambda,
            k=k,
            max_noise_variance=3 ** 2,
        )

        # Initialize priors and sample catalogs
        self.prior = PriorCatalog(self.params, self.catalog_name, self.los_catalog, self.dla_catalog)
        self.dla_samples = DLASamplesMAT(self.params, self.prior, self.dla_samples_file)
        self.subdla_samples = SubDLASamplesMAT(self.params, self.prior, self.sub_dla_samples_file)

        # Bayesian model selection
        self.bayes = BayesModelSelect([0, 1, self.max_dlas], 2)

        # Instantiate Gaussian Process models
        self.null_gp = NullGPMAT(self.params, self.prior, self.learned_file,
                                 prev_tau_0=self.prev_tau_0, prev_beta=self.prev_beta)
        self.dla_gp = DLAGPMAT(
            params=self.params, prior=self.prior, dla_samples=self.dla_samples,
            min_z_separation=self.min_z_separation, learned_file=self.learned_file,
            broadening=True, prev_tau_0=self.prev_tau_0, prev_beta=self.prev_beta
        )
        self.subdla_gp = SubDLAGPMAT(
            params=self.params, prior=self.prior, dla_samples=self.subdla_samples,
            min_z_separation=self.min_z_separation, learned_file=self.learned_file,
            broadening=True, prev_tau_0=self.prev_tau_0, prev_beta=self.prev_beta
        )

        # Initialize results dictionary
        num_spectra = 1
        num_dla_samples = self.dla_samples.log_nhi_samples.shape[0]
        self.results = initialize_results(num_spectra, self.max_dlas, num_dla_samples=num_dla_samples)

    def process_spectrum(self, idx, target_id, z_qso, wavelengths, rest_wavelengths,
                         flux, noise_variance, pixel_mask):
        """
        Process a single spectrum using the Bayesian model selection procedure.

        Parameters
        ----------
        idx : int
            Index of the target in the catalog.
        target_id : int or str
            TARGETID.
        z_qso : float
            Redshift of the quasar.
        wavelengths : array_like
            Observed wavelengths.
        rest_wavelengths : array_like
            Rest-frame wavelengths.
        flux : array_like
            Flux values.
        noise_variance : array_like
            Noise variance.
        pixel_mask : array_like
            Pixel mask.

        Returns
        -------
        results : dict
            Dictionary containing the model posteriors and related information.
        """
        process_single_spectrum(
            idx=idx,
            target_id=target_id,
            z_qso=z_qso,
            wavelengths=wavelengths,
            rest_wavelengths=rest_wavelengths,
            flux=flux,
            noise_variance=noise_variance,
            pixel_mask=pixel_mask,
            params=self.params,
            prior=self.prior,
            dla_samples=self.dla_samples,
            subdla_samples=self.subdla_samples,
            bayes=self.bayes,
            results=self.results,
            max_dlas=self.max_dlas,
            broadening=True,
            gp=self.null_gp,
            dla_gp=self.dla_gp,
            subdla_gp=self.subdla_gp,
            min_z_separation=self.min_z_separation,
            plot_figures=False,
            max_workers=32,
            batch_size=313,
            figure_dir="figures"
        )
        return self.results

    def plot_results(self, z_qso, return_variables=False):
        """
        Plot the detection results in both the spectrum space and the posterior space.

        Parameters
        ----------
        z_qso : float
            Redshift of the quasar.
        return_variables : bool, optional
            If True, also return variables for further plotting.

        Returns
        -------
        fig, ax : matplotlib objects
            The created figure and axes.
        Additional returns if return_variables is True.
        """
        results = self.results
        if self.bayes.p_dla > 0.9:
            nth_lya = 1 + results["model_posteriors"][0, 2:].argmax()
        else:
            nth_lya = 0

        lya_gp = self.dla_gp
        gp = self.null_gp
        sample_z_dlas = lya_gp.dla_samples.sample_z_dlas(lya_gp.this_wavelengths, lya_gp.z_qso)
        sample_log_likelihoods = lya_gp.sample_log_likelihoods[:, 0]
        max_like = np.nanmax(sample_log_likelihoods)
        min_like = np.nanmin(sample_log_likelihoods)
        colours = (sample_log_likelihoods - min_like) / (max_like - min_like)
        colours = colours * 5 - 4
        colours[colours < 0] = 0

        fig, ax = plt.subplots(2, 1, figsize=(16, 10))

        # Real spectrum space
        MAP_z_dla, MAP_log_nhi = lya_gp.maximum_a_posteriori()
        map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
        map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]
        lya_mu, lya_M, lya_omega2 = lya_gp.this_dla_gp(map_z_dlas, 10 ** map_log_nhis)
        absorption = lya_mu / lya_gp.this_mu
        _this_mu = lya_gp.mu_interpolator(lya_gp.X)
        lya_mu = _this_mu * absorption

        this_rest_wavelengths = lya_gp.x
        ind = this_rest_wavelengths < lya_gp.params.lya_wavelength
        this_rest_wavelengths = this_rest_wavelengths[ind]
        lya_mu = lya_mu[ind]

        ax[0].plot((this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1, lya_gp.Y[ind])
        ax[0].plot((this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
                   lya_gp.this_mu[ind], color="red", ls="--", label="GP meanflux")
        ax[0].plot((this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
                   lya_mu,
                   label=(r"$\mathcal{M}$ HCD({n}); z_dlas = ({}); lognhi = ({})"
                          .format(nth_lya,
                                  ",".join("{:.3g}".format(z) for z in map_z_dlas),
                                  ",".join("{:.3g}".format(n) for n in map_log_nhis))),
                   color="red")
        ax[0].fill_between((this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
                           gp.Y[ind] - 2*np.sqrt(gp.v[ind]),
                           gp.Y[ind] + 2*np.sqrt(gp.v[ind]),
                           label="SDSS Instrumental Uncertainty (95%)",
                           color="C0", alpha=0.3)

        # Posterior space
        ax[1].scatter(sample_z_dlas, lya_gp.dla_samples.log_nhi_samples, c=colours, marker="o", alpha=0.5)
        ax[1].scatter(map_z_dlas, map_log_nhis, marker="*", s=100, color="C3")
        ax[1].set_xlim(sample_z_dlas.min(), z_qso)
        ax[1].set_ylim(lya_gp.dla_samples.log_nhi_samples.min(), lya_gp.dla_samples.log_nhi_samples.max())
        ax[1].set_xlabel(r"$z_{Lya}$")
        ax[1].set_ylabel(r"$log N_{HI}$")
        ax[0].set_xlim(sample_z_dlas.min(), z_qso)
        ax[0].set_ylim(-1, 5)
        ax[0].legend()

        if return_variables:
            return fig, ax, nth_lya, map_z_dlas, map_log_nhis, this_rest_wavelengths, lya_mu
        return fig, ax


def plot_added_finder(ax, lya_gp, map_z_dlas, map_log_nhis, z_qso, num_lines=2,
                      marker="x", label="CNN", color="C2"):
    """
    Add results from another DLA finder (CNN/TEMP) to an existing plot.

    Parameters
    ----------
    ax : list
        List of matplotlib axes (first axis for spectrum, second for posterior).
    lya_gp : object
        GP DLA detection object.
    map_z_dlas : array_like
        Array of redshifts from the finder.
    map_log_nhis : array_like
        Array of log(N_HI) values from the finder.
    z_qso : float
        Redshift of the quasar.
    num_lines : int, optional
        Number of Voigt profile lines to use.
    marker : str, optional
        Marker style for the posterior scatter.
    label : str, optional
        Label for the plot.
    color : str, optional
        Color for the plot.

    Returns
    -------
    ax : list
        Updated list of axes.
    """
    nth_lya = len(map_z_dlas)
    absorption = np.ones_like(lya_gp.X)
    for i in range(len(map_log_nhis)):
        absorption *= voigt_absorption(
            lya_gp.X * (lya_gp.z_qso + 1),
            10**map_log_nhis[i],
            map_z_dlas[i],
            broadening=False,
            num_lines=num_lines,
        )
    _this_mu = lya_gp.mu_interpolator(lya_gp.X)
    lya_mu = _this_mu * absorption
    this_rest_wavelengths = lya_gp.x
    ind = this_rest_wavelengths < lya_gp.params.lya_wavelength
    this_rest_wavelengths = this_rest_wavelengths[ind]
    lya_mu = lya_mu[ind]
    ax[0].plot((this_rest_wavelengths * (1 + z_qso)) / lya_gp.params.lya_wavelength - 1,
               lya_mu,
               label=(f"{label}({nth_lya}); z_dlas = ({', '.join(f'{z:.3g}' for z in map_z_dlas)}); "
                      f"lognhi = ({', '.join(f'{n:.3g}' for n in map_log_nhis)})"),
               color=color)
    ax[1].scatter(map_z_dlas, map_log_nhis, marker=marker, s=100, color=color)
    return ax


def process_target(tid, catalog, release, survey, program, output_dir,
                   spectra_filename, zbest_filename, 
                   learned_file_eBOSS, learned_file_desi,
                   catalog_name, los_catalog, dla_catalog,
                   dla_samples_file, sub_dla_samples_file):
    """
    Process a single TARGETID:
      - Loads the spectrum.
      - Extracts the data and plots the observed and zoomed Lya spectrum.
      - Runs eBOSS and DESI trained DLA models.
      - Creates comparison plots (including CNN/TEMP overlays).
      - Saves all outputs to a dedicated directory.

    Parameters
    ----------
    tid : int or str
        TARGETID to process.
    catalog : astropy.table.Table
        Quasar catalog.
    release : str
        Data release identifier.
    survey : str
        Survey name.
    program : str
        Program name.
    output_dir : str
        Base output directory.
    (The remaining parameters are file paths and model parameters.)

    Returns
    -------
    None
    """
    print(f"Processing TARGETID: {tid}")
    # Create a subdirectory for this target
    target_dir = os.path.join(output_dir, str(tid))
    os.makedirs(target_dir, exist_ok=True)

    try:
        specobj, idx, _ = load_spectrum(catalog, tid, release, survey, program)
    except Exception as e:
        print(f"Skipping TARGETID {tid}: {e}")
        return

    spectrum_data, z_qso = extract_spectrum_data(specobj, catalog, tid, idx)
    rest_wavelengths = spectrum_data.wavelengths / (1 + z_qso)

    # Save observed and zoomed plots
    plot_observed_spectrum(spectrum_data.wavelengths, spectrum_data.flux, tid, z_qso, target_dir)
    plot_zoomed_lya(rest_wavelengths, spectrum_data.flux, tid, target_dir)

    # Load CNN/TEMP results and extract for current TARGETID
    mollycat, mollycat_gp, _ = load_cnn_temp_results()
    ind = mollycat["TARGETID"] == tid
    # CNN results
    z_dla_cnn = mollycat[ind]["Z_DLA_CNN"]
    _ind_nan = ~np.isnan(z_dla_cnn.value.data)
    z_dla_cnn = z_dla_cnn.value.data[_ind_nan]
    nhi_cnn = mollycat[ind]["NHI_CNN"]
    nhi_cnn = nhi_cnn.value.data[_ind_nan]
    # TEMP results
    z_dla_temp = mollycat[ind]["Z_DLA_TEMP"]
    _ind_nan = ~np.isnan(z_dla_temp.value.data)
    z_dla_temp = z_dla_temp.value.data[_ind_nan]
    nhi_temp = mollycat[ind]["NHI_TEMP"]
    nhi_temp = nhi_temp.value.data[_ind_nan]

    # ----- eBOSS Trained Model -----
    processor_eBOSS = SpectrumProcessor(
        spectra_filename=spectra_filename,
        zbest_filename=zbest_filename,
        learned_file=learned_file_eBOSS,
        catalog_name=catalog_name,
        los_catalog=los_catalog,
        dla_catalog=dla_catalog,
        dla_samples_file=dla_samples_file,
        sub_dla_samples_file=sub_dla_samples_file,
        max_dlas=3,
        min_z_separation=3000.0, prev_tau_0=0.00554, prev_beta=3.182
    )

    start_time = time.time()
    results_eBOSS = processor_eBOSS.process_spectrum(
        idx=0,
        target_id=tid,
        z_qso=z_qso,
        wavelengths=spectrum_data.wavelengths,
        rest_wavelengths=rest_wavelengths,
        flux=spectrum_data.flux,
        noise_variance=spectrum_data.noise_variance,
        pixel_mask=spectrum_data.pixel_mask
    )
    print("eBOSS Model Results:")
    print("p(Null) =", results_eBOSS["model_posteriors"][0, 0])
    print("p(SubDLA/Alternative) =", results_eBOSS["model_posteriors"][0, 1])
    print("p(DLA+) =", results_eBOSS["model_posteriors"][0, 2:])
    print("eBOSS processing time: {:.2f} s".format(time.time() - start_time))

    # ----- DESI Trained Model -----
    processor_desi = SpectrumProcessor(
        spectra_filename=spectra_filename,
        zbest_filename=zbest_filename,
        learned_file=learned_file_desi,
        catalog_name=catalog_name,
        los_catalog=los_catalog,
        dla_catalog=dla_catalog,
        dla_samples_file=dla_samples_file,
        sub_dla_samples_file=sub_dla_samples_file,
        max_dlas=3,
        min_z_separation=3000.0, prev_tau_0=0.00246, prev_beta=3.62,
        dlambda=0.15, k=30, min_lambda=912.75, max_lambda=1420,
    )
    start_time = time.time()
    results_desi = processor_desi.process_spectrum(
        idx=0,
        target_id=tid,
        z_qso=z_qso,
        wavelengths=spectrum_data.wavelengths,
        rest_wavelengths=rest_wavelengths,
        flux=spectrum_data.flux,
        noise_variance=spectrum_data.noise_variance,
        pixel_mask=spectrum_data.pixel_mask
    )
    print("DESI Model Results:")
    print("p(Null) =", results_desi["model_posteriors"][0, 0])
    print("p(SubDLA/Alternative) =", results_desi["model_posteriors"][0, 1])
    print("p(DLA+) =", results_desi["model_posteriors"][0, 2:])
    print("DESI processing time: {:.2f} s".format(time.time() - start_time))

    # ----- Plotting and Comparison -----
    fig_eBOSS, ax_eBOSS, nth_lya, map_z_dlas, map_log_nhis, this_rest_wavelengths, lya_mu = \
        processor_eBOSS.plot_results(z_qso, return_variables=True)
    plt.suptitle("TARGETID = " + str(tid))
    eBOSS_plot_path = os.path.join(target_dir, f"plot_tid_{tid}_eBOSS.pdf")
    save_plot(fig_eBOSS, eBOSS_plot_path)

    fig_desi, ax_desi = processor_desi.plot_results(z_qso)
    # Overlay eBOSS MAP estimates on DESI plot
    ax_desi[1].scatter(map_z_dlas, map_log_nhis, marker="o", s=100, color="black", alpha=0.8)
    ax_desi[0].plot((this_rest_wavelengths * (1 + z_qso)) / processor_desi.dla_gp.params.lya_wavelength - 1,
                    lya_mu,
                    label=(r"$\mathcal{M}$ eBOSS HCD({n}); z_dlas = ({}); lognhi = ({})"
                           .format(nth_lya,
                                   ",".join("{:.3g}".format(z) for z in map_z_dlas),
                                   ",".join("{:.3g}".format(n) for n in map_log_nhis))),
                    color="black", ls="--")
    ax_desi[0].legend()
    desi_plot_path = os.path.join(target_dir, f"plot_tid_{tid}_DESI.pdf")
    save_plot(fig_desi, desi_plot_path)

    # ----- Comparison with CNN/TEMP Finders -----
    fig_comp, ax_comp = processor_desi.plot_results(z_qso)
    ax_comp = plot_added_finder(ax_comp, processor_desi.dla_gp, z_dla_cnn, nhi_cnn, z_qso,
                                num_lines=1, marker="v", color="C4")
    ax_comp = plot_added_finder(ax_comp, processor_desi.dla_gp, z_dla_temp, nhi_temp, z_qso,
                                num_lines=2, marker="x", color="C2", label="TEMP")

    # Add GP Lyman series markers (Lyα, Lyβ, Lyγ)
    nth_lya_gp = np.argmax(processor_desi.bayes.model_posteriors) - 1
    MAP_z_dla, MAP_log_nhi = processor_desi.dla_gp.maximum_a_posteriori()
    map_z_dlas_gp = MAP_z_dla[nth_lya_gp - 1, :nth_lya_gp]
    map_log_nhis_gp = MAP_log_nhi[nth_lya_gp - 1, :nth_lya_gp]
    ax_comp[0].vlines(map_z_dlas_gp, 1.5, 2.5, color="red")
    for z in map_z_dlas_gp:
        ax_comp[0].text(z, 2.4, r"Ly$\alpha$", rotation=90, color="red")
    map_z_dlbs = (map_z_dlas_gp + 1) * 1025.7 / 1215.67 - 1
    ind = map_z_dlbs > 912 * (1 + z_qso) / 1216 - 1
    ax_comp[0].vlines(map_z_dlbs[ind], 1.5, 2, ls="--", color="red")
    for z in map_z_dlbs[ind]:
        ax_comp[0].text(z, 2, r"Ly$\beta$", rotation=90, color="red")
    map_z_dlgs = (map_z_dlas_gp + 1) * 972.5 / 1215.67 - 1
    ind = map_z_dlgs > 912 * (1 + z_qso) / 1216 - 1
    ax_comp[0].vlines(map_z_dlgs[ind], 1.5, 1.7, ls="dotted", color="red")
    for z in map_z_dlgs[ind]:
        ax_comp[0].text(z, 1.7, r"Ly$\gamma$", rotation=90, color="red")

    # Markers for CNN/TEMP results on the comparison plot
    ax_comp[0].vlines(z_dla_cnn, 1.5, 2.5, color="C4")
    for z in z_dla_cnn:
        ax_comp[0].text(z, 2.4, r"Ly$\alpha$", rotation=90, color="C4")
    ax_comp[0].vlines(z_dla_temp, -0.2, -0.9, color="C2")
    for z in z_dla_temp:
        ax_comp[0].text(z, -0.5, r"Ly$\alpha$", rotation=90, color="C2")
    z_dlb_temp = (z_dla_temp + 1) * 1025.7 / 1215.67 - 1
    ind = z_dlb_temp > 912 * (1 + z_qso) / 1216 - 1
    ax_comp[0].vlines(z_dlb_temp[ind], -0.4, -0.9, ls="--", color="C2")
    for z in z_dlb_temp[ind]:
        ax_comp[0].text(z, -0.5, r"Ly$\beta$", rotation=90, color="C2")
    ax_comp[0].legend()
    ax_comp[1].set_ylim(19.7, 23)
    comp_plot_path = os.path.join(target_dir, f"plot_tid_{tid}_comparison.pdf")
    save_plot(fig_comp, comp_plot_path)
    print(f"Finished processing TARGETID {tid}.\n")


def main():
    """
    Main routine:
      - Parses command-line arguments.
      - Reads the catalog and CNN/TEMP results.
      - Loops over the specified TARGETIDs.
      - Processes each TARGETID and saves outputs in a structured directory.
    """
    parser = argparse.ArgumentParser(description="Process DESI spectra and DLA finder results for multiple TARGETIDs.")
    parser.add_argument("--catalog", type=str, required=True,
                        help="Path to the quasar catalog FITS file.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to store output files and plots.")
    parser.add_argument("--tid_list", type=str, default="",
                        help="Comma-separated list of TARGETIDs to process. If empty, use all from CNN/TEMP selection.")
    parser.add_argument("--balmask", action="store_true", default=True,
                        help="Enable BAL mask (default: True).")
    parser.add_argument("--tilebased", action="store_true", default=False,
                        help="Catalog is tile-based (default: False).")
    parser.add_argument("--release", type=str, default="kibo",
                        help="Data release identifier (default: kibo).")
    parser.add_argument("--survey", type=str, default="main",
                        help="Survey name (default: main).")
    parser.add_argument("--program", type=str, default="dark",
                        help="Program name (default: dark).")
    # Additional arguments for model file paths (adjust defaults as needed)
    parser.add_argument("--spectra_filename", type=str, default="/path/to/spectra-16-724.fits",
                        help="Path to the spectra file (placeholder).")
    parser.add_argument("--zbest_filename", type=str, default="/path/to/zbest-16-724.fits",
                        help="Path to the zbest file (placeholder).")
    parser.add_argument("--learned_file_eBOSS", type=str,
                        default="../data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
                        help="Path to the eBOSS learned model file.")
    parser.add_argument("--learned_file_desi", type=str,
                        default="../learnlogs/model_epoch_682.h5",
                        help="Path to the DESI learned model file.")
    parser.add_argument("--catalog_name", type=str,
                        default="../data/dr12q/processed/catalog.mat",
                        help="Path to the processed catalog for the models.")
    parser.add_argument("--los_catalog", type=str,
                        default="../data/dla_catalogs/dr9q_concordance/processed/los_catalog",
                        help="Path to the LOS catalog.")
    parser.add_argument("--dla_catalog", type=str,
                        default="../data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
                        help="Path to the DLA catalog.")
    parser.add_argument("--dla_samples_file", type=str,
                        default="../data/dr12q/processed/dla_samples_a03.mat",
                        help="Path to the DLA samples file.")
    parser.add_argument("--sub_dla_samples_file", type=str,
                        default="../data/dr12q/processed/subdla_samples.mat",
                        help="Path to the sub-DLA samples file.")

    args = parser.parse_args()

    # Create the base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the quasar catalog
    catalog = read_catalog(args.catalog, args.balmask, args.tilebased)

    # Load CNN/TEMP results to get the default list of TARGETIDs
    _, _, cnn_temp_tids = load_cnn_temp_results()
    if args.tid_list:
        # Parse comma-separated list of TARGETIDs
        tid_list = [int(t.strip()) for t in args.tid_list.split(",")]
    else:
        tid_list = cnn_temp_tids.tolist()

    print(f"Processing {len(tid_list)} TARGETIDs...")

    # Loop over each TARGETID and process
    for tid in tid_list:
        process_target(
            tid=tid,
            catalog=catalog,
            release=args.release,
            survey=args.survey,
            program=args.program,
            output_dir=args.output_dir,
            spectra_filename=args.spectra_filename,
            zbest_filename=args.zbest_filename,
            learned_file_eBOSS=args.learned_file_eBOSS,
            learned_file_desi=args.learned_file_desi,
            catalog_name=args.catalog_name,
            los_catalog=args.los_catalog,
            dla_catalog=args.dla_catalog,
            dla_samples_file=args.dla_samples_file,
            sub_dla_samples_file=args.sub_dla_samples_file
        )

if __name__ == "__main__":
    main()