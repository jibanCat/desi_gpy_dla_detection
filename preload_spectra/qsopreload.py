#!/usr/bin/env python

"""
dlasearch.py

Search for DLAs in spectra from a given catalog.
"""
# include the .. to import from the parent directory
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import h5py
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


import warnings
from scipy.optimize import OptimizeWarning


# import dlaprofile
from fitwarning import DLAFLAG

import constants

from run_bayes_select import DLAHolder
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.null_gp import NullGPMAT

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
    pre-process group of spectra in same file and run DLA searching tools

    Arguments
    ---------
    coaddpath (str) : path to file containing spectra
    catalog (table) : collection of spectra in file to search for DLAs
    model (DLAHolder) : DLA model object

    Returns
    -------
    fitresults (table) : attributes of detected DLAs
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
    # list of quasar spectra data (wavelengths, fluxes, noise variances, pixel masks)
    rest_wavelength_list, flux_list, noise_variance_list, pixel_mask_list = (
        [],
        [],
        [],
        [],
    )

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

        # apply mask to BAL features, if available
        if "NCIV_450" in catalog.columns:
            nbal = catalog["NCIV_450"][entry]
            bal_locs = []
            for n in range(nbal):
                # Compute velocity ranges
                v_max = -catalog[entry]["VMAX_CIV_450"][n] / constants.c + 1.0
                v_min = -catalog[entry]["VMIN_CIV_450"][n] / constants.c + 1.0

                for line, lam in constants.bal_lines.items():
                    # Mask wavelengths within the velocity ranges
                    mask = np.logical_and(wave_rf > lam * v_max, wave_rf < lam * v_min)
                    if (line == "Lya") or (line == "NV"):
                        rededge = (lam * v_min) * (1 + zqso)
                        blueedge = (lam * v_max) * (1 + zqso)
                        bal_locs.append((rededge, blueedge))

                    # Update pixel mask
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

        # Initialize the NullGPMAT object, and then set the data
        gp = NullGPMAT(
            model.params,
            model.prior,
            learned_file=model.learned_file,
            prev_tau_0=model.prev_tau_0,
            prev_beta=model.prev_beta,
        )
        rest_wavelengths = model.params.emitted_wavelengths(wave, zqso)
        gp.set_data(
            rest_wavelengths, flux, noise_variance, pixel_mask, zqso, build_model=True
        )
        # Save the quasar data to the lists
        # here use the data from the GPDLA model, which is already preprocessed with normalization
        rest_wavelength_list.append(gp.X)
        flux_list.append(gp.Y)
        noise_variance_list.append(gp.v)
        pixel_mask_list.append(gp.pixel_mask)

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

        # save results to lists
        tidlist.append(tid)
        ralist.append(ra)
        declist.append(dec)
        zqsolist.append(zqso)

        # DLA parameters
        fitwarnlist.append(fitwarn[n])

        bluesnrlist.append(bluesnr)
        redsnrlist.append(redsnr)

    # TODO: Intermediate results saving for debugging - this is the same format as Roman's code
    processed_filename = "preloaded-" + coaddpath.split("/")[-1].replace("coadd-", "")
    if os.path.exists(os.path.join(model.figure_dir, "preloaded")) is False:
        os.makedirs(os.path.join(model.figure_dir, "preloaded"), exist_ok=True)
    processed_filename = os.path.join(
        model.figure_dir, "preloaded", processed_filename.replace(".fits", ".h5")
    )
    # Save the preprocessed data
    # these are various lengths, so save as lists of arrays
    with h5py.File(processed_filename, "w") as f:
        vlen_dtype = h5py.vlen_dtype(np.float64)  # Variable-length float arrays
        f.create_dataset("rest_wavelength_list", data=rest_wavelength_list, dtype=vlen_dtype)
        f.create_dataset("flux_list", data=flux_list, dtype=vlen_dtype)
        f.create_dataset("noise_variance_list", data=noise_variance_list, dtype=vlen_dtype)
        vlen_dtype = h5py.vlen_dtype(np.bool_)  # Variable-length float arrays
        f.create_dataset("pixel_mask_list", data=pixel_mask_list, dtype=vlen_dtype)
        # save the targetids, ra, dec, zqso, bluesnr, redsnr
        f.create_dataset("tidlist", data=np.array(tidlist, dtype=np.int64))
        f.create_dataset("zqsolist", data=np.array(zqsolist, dtype=np.float64))
        f.create_dataset("bluesnrlist", data=np.array(bluesnrlist, dtype=np.float64))
        f.create_dataset("redsnrlist", data=np.array(redsnrlist, dtype=np.float64))

        # save the metadata
        f.attrs["min_lambda"] = model.params.min_lambda
        f.attrs["max_lambda"] = model.params.max_lambda
        f.attrs["normalization_min_lambda"] = model.params.normalization_min_lambda
        f.attrs["normalization_max_lambda"] = model.params.normalization_max_lambda
        f.attrs["min_num_pixels"] = model.params.min_num_pixels


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
            fitwarnlist,
         ),
        names=[
            "TARGETID",
            "RA",
            "DEC",
            "Z_QSO",  # QSO redshift: 2024-10-25 changed from Z, so remember to update the old dlacat files
            "SNR_FOREST",
            "SNR_REDSIDE",
            "DLAFLAG",
        ],
        dtype=(
            "int",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "int",
        ),
    )

    return fitresults
