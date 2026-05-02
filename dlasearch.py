#!/usr/bin/env python

"""
dlasearch.py — DLA search orchestration for DESI healpix and mock data.

This module sits between the CLI entry point (``desi-DLAGP.py``) and the
per-spectrum Bayesian inference engine (``run_bayes_select.DLAHolder``).
It handles:

- Loading DESI coadded spectra (b/r/z cameras) from FITS files
- Coadding the three camera bands into a single ``brz`` wavelength grid
  (with a fallback to ``resample_spectra_lin_or_log`` when resolution data
  is missing, as occurs for London mock spectra without truth files)
- Applying BAL masking using CIV velocity windows from the QSO catalog
- Enforcing the search-window quality cut (>20% unmasked pixels)
- Dispatching each spectrum to ``DLAHolder.process_qso()`` for GP-DLA inference
- Assembling per-spectrum results into an Astropy Table for the FITS catalog

Main functions
--------------
dlasearch_hpx(healpix, ..., model_params)
    Entry point for real DESI data, indexed by healpix pixel.

dlasearch_mock(specfile, catalog, model_params)
    Entry point for London mock spectra, indexed by FITS file path.

process_spectra_group(coaddpath, catalog, model)
    Core workhorse: loads, preprocesses, and runs DLA inference on all
    spectra in a single FITS file.

Important conventions
---------------------
``model_params`` is a plain dict (not a Parameters object) because
``ProcessPoolExecutor`` must pickle arguments across process boundaries.
Both ``dlasearch_hpx`` and ``dlasearch_mock`` reconstruct ``Parameters``
and ``DLAHolder`` from this dict inside the worker process.

``model_posteriors`` index convention (line ~498):
    - DLA run (``single_absorber_model=False``): num_subdla=1
        index = 1 + 1 + n  →  SubDLA at [1], DLA(n) at [2+n]
    - LLS/sub-DLA run (``single_absorber_model=True``): num_subdla=0
        index = 1 + 0 + n  →  single absorber at [1]
"""

import numpy as np
import os
import fitsio
from astropy.table import Table, vstack

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from functools import partial
import time

# desi packages - TO DO : remove or isolate desi dependencies
import desispec.io
from desispec.interpolation import resample_flux
from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log
from desiutil.log import log

import constants

# import dlaprofile
from fitwarning import DLAFLAG

import warnings
from scipy.optimize import OptimizeWarning

from run_bayes_select import DLAHolder
from gpy_dla_detection.set_parameters import Parameters

warnings.simplefilter("error", OptimizeWarning)

#### FOR TESTING ONLY ####
# import matplotlib.pyplot as plt
##########################


def dlasearch_hpx(healpix, survey, program, datapath, hpxcat, model_params):
    """
    Find the best fitting DLA profile(s) for spectra in hpx catalog.

    Arguments
    ---------
    healpix (int): N64 healpix
    survey (str): e.g., main, sv1, sv2, etc.
    program (str): e.g., bright, dark, etc.
    datapath (str): path to coadd files
    hpxcat (table): collection of spectra to search for DLAs, all belonging to a single healpix
    model_params (dict): dictionary of parameters for the DLAHolder model

    Returns
    -------
    fitresults (table): attributes of detected DLAs
    """
    t0 = time.time()

    # Read spectra from healpix
    coaddname = f"coadd-{survey}-{program}-{str(healpix)}.fits"
    coadd = os.path.join(datapath, str(healpix // 100), str(healpix), coaddname)

    if os.path.exists(coadd):
        # Reconstruct the Parameters instance from the dictionary
        params = Parameters(**model_params["params_dict"])
        params_subdla = Parameters(**model_params["params_subdla_dict"])

        # Reconstruct the DLAHolder instance using the reconstructed Parameters
        model = DLAHolder(
            learned_file=model_params["learned_file"],
            catalog_name=model_params["catalog_name"],
            los_catalog=model_params["los_catalog"],
            dla_catalog=model_params["dla_catalog"],
            dla_samples_file=model_params["dla_samples_file"],
            sub_dla_samples_file=model_params["sub_dla_samples_file"],
            params=params,
            min_z_separation=model_params["min_z_separation"],
            prev_tau_0=model_params["prev_tau_0"],
            prev_beta=model_params["prev_beta"],
            max_dlas=model_params["max_dlas"],
            plot_figures=model_params["plot_figures"],
            max_workers=model_params["max_workers"],
            batch_size=model_params["batch_size"],
            figure_dir=model_params["figure_dir"],
            params_subdla=params_subdla,  # Pass the Sub-DLA Parameters
            filter_low_likelihood=model_params["filter_low_likelihood"],  # Filter low likelihood samples
            single_absorber_model=model_params["single_absorber_model"],  # single absorber model only
            enable_tau_eb=model_params.get("enable_tau_eb", False),
            tau_eb_factors=model_params.get("tau_eb_factors", (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)),
            tau_eb_apply_hcd_mask=model_params.get("tau_eb_apply_hcd_mask", False),
            tau_eb_mask_threshold_sigma=model_params.get("tau_eb_mask_threshold_sigma", 1.5),
            tau_eb_objective=model_params.get("tau_eb_objective", "null"),
        )

        fitresults = process_spectra_group(coadd, hpxcat, model)

    else:
        log.error(f"could not locate coadd file for healpix {healpix}")
        return ()

    t1 = time.time()
    total = np.round(t1 - t0, 2)
    log.info(
        f"Completed processing of {len(hpxcat)} spectra from healpix {healpix} in {total}s"
    )

    return fitresults


def dlasearch_tile(tileid, datapath, tilecat, model, nproc):
    """
    Find the best fitting DLA profile(s) for spectra in hpx catalog

    Arguments
    ---------
    tileid (int) : tile no.
    datapath (str) : path to coadd files
    tilecat (table) : collection of spectra to search for DLAs, all belonging to
                     single tile
    model (dict) : flux model dictionary containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys
    nproc (int) : number of multiprocessing processes for solve_DLA, default=64

    Returns
    -------
    fitresults (table) : fit attributes for detected DLAs
    """

    t0 = time.time()

    # do tile based search, will need to save tileid in catalog since targetid is not unique to a tile
    # call process_spectra_group, append tileid and petal id to fitresults
    # loop over petal number

    # e.g. for petal in np.unique(tilecat['PETAL_LOC']):
    #           petcat = tilecat[tilecat['PETAL_LOC'] == petal]
    #           coadd = 'path to tile-petal coadd'
    #           #check if pool should be set up
    #           process_spectr_group(coadd, petcat, model, pool)
    #           # apeend tile and petal columns

    t1 = time.time()
    total = t1 - t0
    log.info(
        f"Completed processing of {len(tilecat)} spectra from tile {tileid} in {total}s"
    )


def dlasearch_mock(specfile, catalog, model_params):
    """
    Find the best fitting DLA profile(s) for spectra in the mock catalog.

    Arguments
    ---------
    specfile (str): Path to the mock spectra file.
    catalog (table): Catalog of spectra to search for DLAs.
    model_params (dict): Dictionary containing parameters for the DLAHolder model.

    Returns
    -------
    fitresults (table): Fit attributes for detected DLAs.
    """
    t0 = time.time()

    if os.path.exists(specfile):
        fm = desispec.io.read_fibermap(specfile)
        tidmask = np.in1d(catalog["TARGETID"], fm["TARGETID"])
        catalog = catalog[tidmask]
        if len(catalog) < 1:
            return ()

        # Reconstruct the Parameters instance from the dictionary
        params = Parameters(**model_params["params_dict"])
        params_subdla = Parameters(**model_params["params_subdla_dict"])

        # Log the parameters
        log.info(f"Parameters: ---")
        for key, value in model_params["params_dict"].items():
            log.info(f"{key}: {value}")
            log.info(f"---")

        # Reconstruct the DLAHolder instance using the reconstructed Parameters
        model = DLAHolder(
            learned_file=model_params["learned_file"],
            catalog_name=model_params["catalog_name"],
            los_catalog=model_params["los_catalog"],
            dla_catalog=model_params["dla_catalog"],
            dla_samples_file=model_params["dla_samples_file"],
            sub_dla_samples_file=model_params["sub_dla_samples_file"],
            params=params,
            min_z_separation=model_params["min_z_separation"],
            prev_tau_0=model_params["prev_tau_0"],
            prev_beta=model_params["prev_beta"],
            max_dlas=model_params["max_dlas"],
            plot_figures=model_params["plot_figures"],
            max_workers=model_params["max_workers"],
            batch_size=model_params["batch_size"],
            figure_dir=model_params["figure_dir"],
            params_subdla=params_subdla,  # Pass the Sub-DLA Parameters
            filter_low_likelihood=model_params["filter_low_likelihood"],  # Filter low likelihood samples
            single_absorber_model=model_params["single_absorber_model"],  # single absorber model only
            enable_tau_eb=model_params.get("enable_tau_eb", False),
            tau_eb_factors=model_params.get("tau_eb_factors", (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)),
            tau_eb_apply_hcd_mask=model_params.get("tau_eb_apply_hcd_mask", False),
            tau_eb_mask_threshold_sigma=model_params.get("tau_eb_mask_threshold_sigma", 1.5),
            tau_eb_objective=model_params.get("tau_eb_objective", "null"),
        )

        fitresults = process_spectra_group(specfile, catalog, model)
    else:
        log.error(f"could not locate coadd file for {specfile}")
        return ()

    t1 = time.time()
    total = np.round(t1 - t0, 2)
    log.info(
        f"Completed processing of {len(catalog)} spectra from {specfile} in {total}s"
    )

    return fitresults


def process_spectra_group(coaddpath, catalog, model: DLAHolder):
    """
    Pre-process spectra from a single coadd file and run GP-DLA inference.

    This is the core processing function called by both ``dlasearch_hpx``
    (for real DESI data) and ``dlasearch_mock`` (for London mock data).

    Processing steps per file:
    1. Load b/r/z camera coadds from the FITS file.
    2. Coadd cameras → single ``brz`` wavelength grid.
       Fallback path when ``resolution_data`` is missing (London mocks):
       read resolution from the companion ``truth-16-*.fits`` file, then
       resample to a linear grid (step=0.8 Å) before coadding.
    3. Per spectrum:
       a. Extract flux, ivar, rest-frame wavelengths.
       b. Optionally apply BAL masking: set ``pixel_mask=True`` and ``ivar=0``
          for pixels within CIV (and other line) velocity windows read from
          the QSO catalog columns ``NCIV_450``, ``VMIN_CIV_450``, ``VMAX_CIV_450``.
       c. Enforce search-window quality: skip spectra where >80% of the
          search region (constants.search_minlam–search_maxlam) is masked.
       d. Call ``model.process_qso()`` → GP-DLA Bayesian inference.
       e. Extract MAP z_DLA, log_NHI, errors, and model posteriors.
       f. Check for potential BAL contamination of detected DLAs (DLAFLAG).
    4. Write per-file HDF5 results via ``model.save_results()``.
    5. Assemble FITS catalog table of detected absorbers.

    Parameters
    ----------
    coaddpath : str
        Path to the DESI coadded spectra FITS file (e.g., ``coadd-main-dark-705.fits``
        or a London mock ``spectra-16-705.fits``).
    catalog : astropy.table.Table
        Sub-catalog of spectra to process from this file.
        Required columns: TARGETID, Z, TARGET_RA/DEC (or RA/DEC for mocks).
        Optional BAL columns: NCIV_450, VMIN_CIV_450, VMAX_CIV_450.
    model : DLAHolder
        Initialized DLA model object (GP matrices + QMC samples loaded).

    Returns
    -------
    fitresults : astropy.table.Table or ()
        Table of detected DLA/absorber entries with columns:
        TARGETID, RA, DEC, Z_QSO, SNR_FOREST, SNR_REDSIDE, DLAID,
        Z_DLA, Z_DLA_ERR, NHI, NHI_ERR, DLAFLAG,
        P_DLA, P_NULL, LOGP_DLA, LOGP_NULL, MODEL_P.
        Returns an empty tuple ``()`` if no absorbers were detected.

    Notes
    -----
    BAL masking velocity convention:
        The velocity columns VMIN_CIV_450 and VMAX_CIV_450 are in km/s
        (absolute velocity relative to QSO). They are converted to
        rest-frame wavelength offsets via ``v/c``:
            lambda_mask = lambda_line * (1 ± v/c)
        where v > 0 is blueward of QSO. The mask is applied to all
        ``constants.bal_lines`` simultaneously.

    model_posteriors indexing:
        Only the posterior for the k-th DLA model is saved in the catalog
        (column ``MODEL_P``). The index into ``model_posteriors`` is:
            index = 1 + num_subdla + n
        where num_subdla=1 for DLA runs and 0 for single-absorber runs,
        and n is the 0-based DLA index (0 for first DLA, 1 for second, ...).
    """

    specobj = desispec.io.read_spectra(
        coaddpath,
        targetids=catalog["TARGETID"],
        skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"],
    )
    try:
        specobj = coadd_cameras(specobj)
    except:
        if specobj.resolution_data is not None:
            # resample on linear grid
            wave_min = np.min(specobj.wave["b"])
            wave_max = np.max(specobj.wave["z"])
            specobj = resample_spectra_lin_or_log(
                specobj,
                linear_step=0.8,
                wave_min=wave_min,
                wave_max=wave_max,
                fast=True,
            )
            specobj = coadd_cameras(specobj)
        else:
            # check if mock truth file exists
            truthfile = coaddpath.replace("spectra-16-", "truth-16-")
            if not (os.path.exists(truthfile)):
                log.error(
                    f"cannot process {coaddpath}; no mock truth file or resolution data"
                )
            specobj.resolution_data = {}
            for cam in ["b", "r", "z"]:
                tres = fitsio.read(truthfile, ext=f"{cam}_RESOLUTION")
                tresdata = np.empty(
                    [
                        specobj.flux[cam].shape[0],
                        tres.shape[0],
                        specobj.flux[cam].shape[1],
                    ],
                    dtype=float,
                )
                for i in range(specobj.flux[cam].shape[0]):
                    tresdata[i] = tres
                specobj.resolution_data[cam] = tresdata
            specobj = resample_spectra_lin_or_log(
                specobj,
                linear_step=0.8,
                wave_min=np.min(specobj.wave["b"]),
                wave_max=np.max(specobj.wave["z"]),
                fast=True,
            )

    # for each entry in passed catalog, fit spectrum with intrinsic model + N DLA
    wave = specobj.wave["brz"]

    # lists shared with Allyson's finder
    tidlist, ralist, declist, zqsolist, bluesnrlist, redsnrlist, dlaidlist = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    zlist, nhilist, zerrlist, nhierrlist, fitwarnlist = (
        [],
        [],
        [],
        [],
        [],
    )
    # lists for GP-DLA results
    pdlalist = []
    pnulllist = []
    logpdlalist = []
    logpnulllist = []
    modelplist = []

    # set up results dict for GPDLA
    num_spectra = len(catalog)
    model.initialize_results(num_spectra=num_spectra)

    # for each entry in passed catalog, fit spectrum with intrinsic model + N DLA
    for entry in range(len(catalog)):

        tid = catalog["TARGETID"][entry]
        try:
            ra = catalog["TARGET_RA"][entry]
            dec = catalog["TARGET_DEC"][entry]
        except:
            # mock catalog
            ra = catalog["RA"][entry]
            dec = catalog["DEC"][entry]
        zqso = catalog["Z"][entry]

        try:
            idx = np.nonzero(specobj.fibermap["TARGETID"] == tid)[0][0]
        except:
            log.error(
                f"Targetid {tid} NOT FOUND on healpix {catalog['HPXPIXEL'][entry]}"
            )
            continue

        # TODO: Do the GP finder here

        flux = specobj.flux["brz"][idx]
        ivar = specobj.ivar["brz"][idx]
        wave_rf = wave / (1 + zqso)
        pixel_mask = specobj.mask["brz"][idx].astype(np.bool_)

        # Apply BAL masking using CIV velocity windows from the QSO catalog.
        # NCIV_450 is the number of CIV absorption systems with v > 450 km/s.
        # VMIN/VMAX_CIV_450 are the velocity bounds (km/s, positive = blueward of QSO).
        # We mask all lines in constants.bal_lines within each velocity window.
        # The velocity-to-wavelength conversion:
        #   lambda_obs = lambda_rest * (1 - v/c)  for blueward velocity
        # so the mask covers rest-frame wavelengths from lam*(1-v_max/c) to lam*(1-v_min/c).
        if "NCIV_450" in catalog.columns:
            nbal = catalog["NCIV_450"][entry]
            bal_locs = []
            for n in range(nbal):
                # velocity factor: (1 - v/c) for each edge of the BAL trough
                # VMAX_CIV_450 is the high-velocity (blueshifted) edge
                # VMIN_CIV_450 is the low-velocity edge
                v_max = -catalog[entry]["VMAX_CIV_450"][n] / constants.c + 1.0
                v_min = -catalog[entry]["VMIN_CIV_450"][n] / constants.c + 1.0

                for line, lam in constants.bal_lines.items():
                    # rest-frame wavelength range to mask for this BAL trough + line
                    mask = np.logical_and(wave_rf > lam * v_max, wave_rf < lam * v_min)
                    # track Lyα and NV observed-frame ranges for post-detection BAL check
                    if (line == "Lya") or (line == "NV"):
                        rededge = (lam * v_min) * (1 + zqso)
                        blueedge = (lam * v_max) * (1 + zqso)
                        bal_locs.append((rededge, blueedge))

                    # Update pixel mask and zero out inverse variance
                    pixel_mask[mask] = True

                    ivar[mask] = 0

        # Convert inverse variance to variance
        noise_variance = np.zeros(ivar.shape)
        ind = ivar == 0
        noise_variance[:] = np.nan
        noise_variance[~ind] = 1 / ivar[~ind]

        # Append ivar=0 to pixel mask
        pixel_mask[ind] = True

        # This part set by Allyson, leave it as it is to match the final catalog filtering
        # only searching to rest frame 900 A (TODO: make this match GPDLA search range)
        fitmask = wave_rf > constants.search_minlam
        # limit our bestfit comparision w/ and w/o DLAs to search region of spectrum
        searchmask = np.ma.masked_inside(
            wave_rf[fitmask], constants.search_minlam, constants.search_maxlam
        ).mask
        # check if too much of the spectrum is masked
        if np.sum(ivar[fitmask][searchmask] != 0) / np.sum(searchmask) < 0.2:
            log.warning(f"Targetid {tid} skipped - SEARCH WINDOW >80% MASKED")
            continue

        # Allyson's code to get fitwarning
        # TODO: replace this specific to GP
        fitwarn = np.full(model.max_dlas, 0)

        try:
            # Process each QSO, resampling model to observed wavelength grid
            model.process_qso(
                entry,
                tid,
                wavelengths=wave,
                flux=flux,
                noise_variance=noise_variance,
                pixel_mask=pixel_mask,
                z_qso=zqso,
            )

        except np.linalg.LinAlgError:
            # Catch any LinAlgError and set a flag
            print(f"Warning: LinAlgError for target ID {tid}. Setting error flag.")
            # error_flags[tid] = "non_pos_def_matrix"
            fitwarn |= DLAFLAG.BAD_ZFIT  # TODO: Placeholder - change to GPDLA flag

        except ValueError as e:
            if "All-NaN slice encountered" in str(e):
                print(
                    f"Warning: All-NaN slice encountered for target ID {tid}. Setting error flag."
                )
                # error_flags[tid] = "all_nan_slice"
                fitwarn |= (
                    DLAFLAG.BAD_NHIFIT
                )  # TODO: Placeholder - change to GPDLA flag
            else:
                # If it's an unexpected ValueError, re-raise it
                raise

        # Get zerr and nhierr from GPDLA
        # TODO: check the robustness of zerr and nhierr
        # model w/o DLAs
        log_posteriors_no_dla = model.results["log_posteriors_no_dla"][entry]
        p_no_dla = model.results["p_no_dlas"][entry]

        zdla = model.results["MAP_z_dlas"][entry]
        zerr = model.results["z_dla_errs"][entry]
        nhi = model.results["MAP_log_nhis"][entry]
        nhierr = model.results["log_nhi_errs"][entry]
        log_posteriors_dla = model.results["log_posteriors_dla"][entry]
        p_dla = model.results["p_dlas"][entry]
        model_posteriors = model.results["model_posteriors"][entry]

        # replace nan with -1 for Allysion's convention
        zdla[np.isnan(zdla)] = -1
        zerr[np.isnan(zerr)] = -1
        nhi[np.isnan(nhi)] = -1
        nhierr[np.isnan(nhierr)] = -1

        # check for potential BAL contamination in solution
        # false positive should only come from Lya and NV - all other lines too weak
        if ("nbal" in locals()) & np.any(zdla != -1):
            lam_center_dla = constants.Lya_line * (1 + zdla)
            for window in bal_locs:
                balflag = (lam_center_dla < window[0]) & (lam_center_dla > window[1])
                fitwarn[balflag] |= DLAFLAG.POTENTIAL_BAL

        # average signal to noise computation
        mask = np.logical_and(
            ivar != 0,
            np.ma.masked_inside(
                wave_rf, constants.bluesnr_min, constants.bluesnr_max
            ).mask,
        )
        bluesnr = np.mean((flux[mask] * np.sqrt(ivar[mask])))
        mask = np.logical_and(
            ivar != 0,
            np.ma.masked_inside(
                wave_rf, constants.redsnr_min, constants.redsnr_max
            ).mask,
        )
        redsnr = np.mean((flux[mask] * np.sqrt(ivar[mask])))
        # save SNR values
        model.results["snrs"][entry] = redsnr
        model.results["snrs_blue"][entry] = bluesnr
        # save detection flag
        model.results["detection_flags"][entry] = np.sum(fitwarn) > 0

        ndla = np.sum(zdla != -1)

        # whether use single model only
        if model.single_absorber_model:
            num_subdla = 0
        else:
            num_subdla = 1

        for n in range(ndla):
            tidlist.append(tid)
            dlaid = str(tid) + "00" + str(n)
            dlaidlist.append(dlaid)
            ralist.append(ra)
            declist.append(dec)
            zqsolist.append(zqso)

            # DLA parameters
            zlist.append(zdla[n])
            zerrlist.append(zerr[n])
            nhilist.append(nhi[n])
            nhierrlist.append(nhierr[n])
            fitwarnlist.append(fitwarn[n])

            bluesnrlist.append(bluesnr)
            redsnrlist.append(redsnr)

            # GP-DLA results
            pdlalist.append(p_dla)
            pnulllist.append(p_no_dla)
            logpdlalist.append(log_posteriors_dla[n])
            logpnulllist.append(log_posteriors_no_dla)
            # model_posteriors index: [Null, (SubDLA), DLA(0), DLA(1), ...]
            # DLA run (num_subdla=1): index 2+n picks DLA(n) posterior
            # Single-absorber run (num_subdla=0): index 1+n picks absorber(n) posterior
            modelplist.append(model_posteriors[1 + num_subdla + n])

    # TODO: Intermediate results saving for debugging - this is the same format as Roman's code
    processed_filename = "processed-" + coaddpath.split("/")[-1].replace("coadd-", "")
    if os.path.exists(os.path.join(model.figure_dir, "processed")) is False:
        os.makedirs(os.path.join(model.figure_dir, "processed"), exist_ok=True)
    processed_filename = os.path.join(
        model.figure_dir, "processed", processed_filename.replace(".fits", ".h5")
    )
    model.save_results(output_file=processed_filename)

    if len(tidlist) == 0:
        # avoid vstack error for empty tables
        return ()

    # DLACAT create table of fit results
    fitresults = Table(
        data=(
            tidlist,
            ralist,
            declist,
            zqsolist,
            bluesnrlist,
            redsnrlist,
            dlaidlist,
            zlist,
            zerrlist,
            nhilist,
            nhierrlist,
            fitwarnlist,
            # GP-DLA results
            pdlalist,  # posterior probability of DLA model
            pnulllist,  # posterior probability of no DLA model
            logpdlalist,  # log posterior probability of DLA model
            logpnulllist,  # log posterior probability of no DLA model
            modelplist,  # model posterior probabilities
        ),
        names=[
            "TARGETID",
            "RA",
            "DEC",
            "Z_QSO",  # QSO redshift: 2024-10-25 changed from Z, so remember to update the old dlacat files
            "SNR_FOREST",
            "SNR_REDSIDE",
            "DLAID",
            "Z_DLA",
            "Z_DLA_ERR",
            "NHI",
            "NHI_ERR",
            "DLAFLAG",
            "P_DLA",
            "P_NULL",
            "LOGP_DLA",
            "LOGP_NULL",
            "MODEL_P",
        ],
        dtype=(
            "int",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "str",
            "float64",
            "float64",
            "float64",
            "float64",
            "int",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
        ),
    )

    return fitresults
