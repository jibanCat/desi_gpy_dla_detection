import os
import numpy as np
import fitsio

import h5py
from astropy.table import Table
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log
from desiutil.log import log
import constants

from gpy_dla_detection.set_parameters import Parameters

params_dict = {
    "loading_min_lambda": 800.0,
    "loading_max_lambda": 1550.0,
    "normalization_min_lambda": 1425.0,
    "normalization_max_lambda": 1475.0,
    "min_lambda": 850.75,
    "max_lambda": 1420.75,
    "dlambda": 0.25,
    "k": 20,
    "max_noise_variance": 9,
}


param = Parameters(**params_dict)

# Constants
lyman_limit = param.lyman_limit  # Angstroms
lya_wavelength = param.lya_wavelength  # Angstroms
loading_min_lambda = param.loading_min_lambda  # Angstroms
loading_max_lambda = param.loading_max_lambda  # Angstroms
normalization_min_lambda = param.normalization_min_lambda  # Angstroms
normalization_max_lambda = param.normalization_max_lambda  # Angstroms
min_num_pixels = param.min_num_pixels  # Minimum number of pixels in the sampling range


def read_catalog(qsocat, balmask, bytile):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes from baltools be read in?
    bytile (bool) : catalog is tilebased, default assumption is healpix

    Returns
    -------
    table of relevant attributes for quasars defined in constants.py

    """
    if constants.no_bal:
        balmask = True

    if balmask:
        try:
            # read the following columns from qsocat
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "Z",
                "HPXPIXEL",
                "AI_CIV",
                "NCIV_450",
                "VMIN_CIV_450",
                "VMAX_CIV_450",
                "SPECTYPE",
                "ZWARN",
            ]
            if bytile:
                cols = [
                    "TARGETID",
                    "TARGET_RA",
                    "TARGET_DEC",
                    "Z",
                    "TILEID",
                    "PETAL_LOC",
                    "AI_CIV",
                    "NCIV_450",
                    "VMIN_CIV_450",
                    "VMAX_CIV_450",
                    "SPECTYPE",
                    "ZWARN",
                ]
            catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
        except:
            log.error(f"cannot find {cols} in quasar catalog")
            exit(1)
    else:
        # read the following columns from qsocat
        cols = [
            "TARGETID",
            "TARGET_RA",
            "TARGET_DEC",
            "Z",
            "HPXPIXEL",
            "SPECTYPE",
            "ZWARN",
        ]
        if bytile:
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "Z",
                "TILEID",
                "PETAL_LOC",
                "SPECTYPE",
                "ZWARN",
            ]
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    log.info(f"Successfully read quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    # Apply bal mask
    if constants.no_bal:
        balind = catalog["NCIV_450"] > 0
        zmask = zmask & ~balind
        log.info(f"objects in catalog without BAL: {np.sum(zmask)}")

    # Apply zwarning mask
    if constants.zwarning:
        zmask = zmask & (catalog["ZWARN"] == 0)
        log.info(f"objects in catalog without ZWARN: {np.sum(zmask)}")

    # Apply spectype mask
    if constants.is_qso:
        zmask = zmask & (catalog["SPECTYPE"] == "QSO")
        log.info(f"objects in catalog with SPECTYPE QSO: {np.sum(zmask)}")

    catalog = catalog[zmask]

    return catalog, zmask


def read_coadded_spectrum(datapath, healpix, survey, program):
    """
    Read coadded spectrum file for a given healpix pixel.

    Args:
        datapath (str): Base path to the spectra directory.
        healpix (int): Healpix pixel number.
        survey (str): Survey name.
        program (str): Observing program.

    Returns:
        Spectra object: Coadded spectrum object for the given healpix.
    """
    coaddname = f"coadd-{survey}-{program}-{healpix}.fits"
    coadd_path = os.path.join(datapath, str(healpix // 100), str(healpix), coaddname)

    if not os.path.exists(coadd_path):
        log.error(f"Coadded spectrum file not found: {coadd_path}")
        return None

    specobj = read_spectra(
        coadd_path, skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"]
    )

    try:
        specobj = coadd_cameras(specobj)
    except Exception:
        log.warning(
            f"Error coadding cameras for healpix {healpix}. Resampling spectra."
        )
        wave_min = np.min(specobj.wave["b"])
        wave_max = np.max(specobj.wave["z"])
        specobj = resample_spectra_lin_or_log(
            specobj, linear_step=0.8, wave_min=wave_min, wave_max=wave_max, fast=True
        )
        specobj = coadd_cameras(specobj)

    return specobj


def preload_qsos(catalog_path, spectra_dir, output_file, survey, program, release):
    """
    Preload QSO spectra with normalization and filtering applied, saving to HDF5.

    Args:
        catalog_path (str): Path to the QSO catalog file.
        spectra_dir (str): Base directory of coadded spectra files.
        output_file (str): Path to save the preloaded QSO data.
        survey (str): Survey name.
        program (str): Observing program.
        release (str): Data release version.
    """
    # Load QSO catalog
    catalog, zmask = read_catalog(catalog_path, balmask=True, bytile=False)
    z_qsos = catalog["Z"]
    target_ids = catalog["TARGETID"]
    hpxpixels = catalog["HPXPIXEL"]

    num_quasars = len(z_qsos)
    filter_flags = np.zeros(num_quasars, dtype=int)

    # Initialize containers
    all_wavelengths = []
    all_flux = []
    all_noise_variance = []
    all_pixel_mask = []
    all_normalizers = np.zeros(num_quasars)
    all_target_ids = []
    all_zqsos = []

    # Process each healpix
    datapath = os.path.join(spectra_dir, release, "healpix", survey, program)
    unique_hpxpixels = np.unique(hpxpixels)

    for healpix in unique_hpxpixels:
        log.info(f"Processing healpix {healpix}...")
        specobj = read_coadded_spectrum(datapath, healpix, survey, program)

        if specobj is None:
            log.warning(f"Skipping healpix {healpix} due to missing spectrum file.")
            continue

        # Process each QSO in this healpix
        hpx_indices = np.where(hpxpixels == healpix)[0]
        for idx in hpx_indices:
            tid = target_ids[idx]
            z_qso = z_qsos[idx]

            try:
                spec_idx = np.nonzero(specobj.fibermap["TARGETID"] == tid)[0][0]
            except IndexError:
                log.error(f"Targetid {tid} not found in healpix {healpix}. Skipping.")
                continue

            wave = specobj.wave["brz"]
            flux = specobj.flux["brz"][spec_idx]
            ivar = specobj.ivar["brz"][spec_idx]
            mask = specobj.mask["brz"][idx].astype(np.bool_)

            # Convert inverse variance to variance
            noise_variance = np.zeros(ivar.shape)
            ind = ivar == 0
            noise_variance[:] = np.nan
            noise_variance[~ind] = 1 / ivar[~ind]
            # Append ivar=0 to pixel mask
            mask[ind] = True

            # Process rest wavelengths
            wave_rf = wave / (1 + z_qso)

            # Normalize flux
            norm_mask = (
                (wave_rf >= normalization_min_lambda)
                & (wave_rf <= normalization_max_lambda)
                & (~mask)
            )

            if not np.any(norm_mask):
                filter_flags[idx] |= 1 << 2  # Bit 2: cannot normalize
                # continue

            median_flux = np.nanmedian(flux[norm_mask])
            if np.isnan(median_flux):
                filter_flags[idx] |= 1 << 2
                # continue

            # Sampling range
            sampling_mask = (
                (wave_rf >= lyman_limit) & (wave_rf <= lya_wavelength) & (~mask)
            )

            if np.sum(sampling_mask) < min_num_pixels:
                filter_flags[idx] |= 1 << 3  # Bit 3: not enough pixels
                # continue

            # Normalize flux and noise variance
            all_normalizers[idx] = median_flux
            flux /= median_flux
            noise_variance /= median_flux**2

            # Limit wavelength range to loading region
            loading_mask = (wave_rf >= loading_min_lambda) & (
                wave_rf <= loading_max_lambda
            )

            # Extend by one pixel on either side if available
            available_indices = np.where(~mask)[0]
            if available_indices.size > 0:
                first_idx = available_indices[
                    available_indices > np.where(loading_mask)[0][-1]
                ][0]
                last_idx = available_indices[
                    available_indices < np.where(loading_mask)[0][0]
                ][-1]
                loading_mask[first_idx] = True
                loading_mask[last_idx] = True

            # Store the preprocessed data
            all_wavelengths.append(wave[loading_mask])
            all_flux.append(flux[loading_mask])
            all_noise_variance.append(
                noise_variance[loading_mask]
            )  # Convert back to variance
            all_pixel_mask.append(mask[loading_mask])

            all_target_ids.append(tid)
            all_zqsos.append(z_qso)

            log.info(
                f"Processed QSO {idx + 1}/{num_quasars} (TARGETID={tid}) in healpix {healpix}"
            )

    # Save to HDF5
    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset(
            "all_wavelengths", data=np.array(all_wavelengths, dtype=object)
        )
        h5f.create_dataset("all_flux", data=np.array(all_flux, dtype=object))
        h5f.create_dataset(
            "all_noise_variance", data=np.array(all_noise_variance, dtype=object)
        )
        h5f.create_dataset(
            "all_pixel_mask", data=np.array(all_pixel_mask, dtype=object)
        )
        h5f.create_dataset("all_normalizers", data=all_normalizers)
        h5f.create_dataset("filter_flags", data=filter_flags)
        h5f.create_dataset("all_target_ids", data=np.array(all_target_ids))
        h5f.create_dataset("all_zqsos", data=np.array(all_zqsos))
        h5f.create_dataset("zmasks", data=zmask)

        h5f.attrs["loading_min_lambda"] = loading_min_lambda
        h5f.attrs["loading_max_lambda"] = loading_max_lambda
        h5f.attrs["normalization_min_lambda"] = normalization_min_lambda
        h5f.attrs["normalization_max_lambda"] = normalization_max_lambda
        h5f.attrs["min_num_pixels"] = min_num_pixels

        # Save metadata in HDF5 file
        h5f.attrs["healpix_processed"] = list(unique_hpxpixels)
        h5f.attrs["release"] = release
        h5f.attrs["survey"] = survey
        h5f.attrs["program"] = program

    # Summary logs
    processed_qsos = len(all_target_ids)
    log.info(
        f"Processed {processed_qsos} QSOs. Skipped {num_quasars - processed_qsos}."
    )
    log.info(f"Saved preloaded QSO data to {output_file}")

    return filter_flags


# Example usage
if __name__ == "__main__":
    catalog_path = "/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits"
    spectra_dir = "/global/cfs/cdirs/desi/spectro/redux/"
    output_file = "preloaded_qsos.h5"
    survey = "main"
    program = "dark"
    release = "kibo"

    filter_flags = preload_qsos(
        catalog_path, spectra_dir, output_file, survey, program, release
    )
    log.info("Preloading complete.")
