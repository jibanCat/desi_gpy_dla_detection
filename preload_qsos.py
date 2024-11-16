import os
import numpy as np
import fitsio
from concurrent.futures import ProcessPoolExecutor

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


def process_healpix(healpix, catalog, datapath, survey, program, temp_file):
    """
    Process a single healpix pixel and save results to a temporary HDF5 file.

    Args:
        healpix (int): Healpix pixel number.
        catalog (Table): Catalog of quasars.
        datapath (str): Path to coadded spectra.
        survey (str): Survey name.
        program (str): Program name.
        temp_file (str): Path to the temporary HDF5 file for results.

    Returns:
        None
    """
    log.info(f"Processing healpix {healpix}...")
    specobj = read_coadded_spectrum(datapath, healpix, survey, program)

    if specobj is None:
        log.warning(f"Skipping healpix {healpix} due to missing spectrum file.")
        return

    # Containers for healpix results
    wavelengths, fluxes, noise_variances, pixel_masks = [], [], [], []
    normalizers, target_ids, zqsos, flags = [], [], [], []

    hpx_indices = np.where(catalog["HPXPIXEL"] == healpix)[0]
    for idx in hpx_indices:
        tid = catalog["TARGETID"][idx]
        z_qso = catalog["Z"][idx]

        try:
            spec_idx = np.nonzero(specobj.fibermap["TARGETID"] == tid)[0][0]
        except IndexError:
            log.error(f"Targetid {tid} not found in healpix {healpix}. Skipping.")
            continue

        wave = specobj.wave["brz"]
        flux = specobj.flux["brz"][spec_idx]
        ivar = specobj.ivar["brz"][spec_idx]
        mask = specobj.mask["brz"][spec_idx].astype(np.bool_)

        # Convert inverse variance to variance
        noise_variance = np.zeros(ivar.shape)
        ind = ivar == 0
        noise_variance[:] = np.nan
        noise_variance[~ind] = 1 / ivar[~ind]
        mask[ind] = True

        wave_rf = wave / (1 + z_qso)

        # Normalize flux
        norm_mask = (
            (wave_rf >= normalization_min_lambda)
            & (wave_rf <= normalization_max_lambda)
            & (~mask)
        )

        if not np.any(norm_mask):
            flags.append(1 << 2)  # Bit 2: cannot normalize
            continue

        median_flux = np.nanmedian(flux[norm_mask])
        if np.isnan(median_flux):
            flags.append(1 << 2)
            continue

        sampling_mask = (wave_rf >= lyman_limit) & (wave_rf <= lya_wavelength) & (~mask)

        if np.sum(sampling_mask) < min_num_pixels:
            flags.append(1 << 3)  # Bit 3: not enough pixels
            continue

        flux /= median_flux
        noise_variance /= median_flux**2

        loading_mask = (wave_rf >= loading_min_lambda) & (wave_rf <= loading_max_lambda)

        wavelengths.append(wave[loading_mask])
        fluxes.append(flux[loading_mask])
        noise_variances.append(noise_variance[loading_mask])
        pixel_masks.append(mask[loading_mask])
        normalizers.append(median_flux)
        target_ids.append(tid)
        zqsos.append(z_qso)
        flags.append(0)

    # Save results to temporary HDF5 file
    with h5py.File(temp_file, "w") as h5f:
        h5f.create_dataset("all_wavelengths", data=np.array(wavelengths, dtype=object))
        h5f.create_dataset("all_flux", data=np.array(fluxes, dtype=object))
        h5f.create_dataset(
            "all_noise_variance", data=np.array(noise_variances, dtype=object)
        )
        h5f.create_dataset("all_pixel_mask", data=np.array(pixel_masks, dtype=object))
        h5f.create_dataset("all_normalizers", data=np.array(normalizers))
        h5f.create_dataset("all_target_ids", data=np.array(target_ids))
        h5f.create_dataset("all_zqsos", data=np.array(zqsos))
        h5f.create_dataset("filter_flags", data=np.array(flags))

        # Save metadata
        h5f.attrs["healpix"] = healpix
        h5f.attrs["loading_min_lambda"] = loading_min_lambda
        h5f.attrs["loading_max_lambda"] = loading_max_lambda
        h5f.attrs["normalization_min_lambda"] = normalization_min_lambda
        h5f.attrs["normalization_max_lambda"] = normalization_max_lambda
        h5f.attrs["min_num_pixels"] = min_num_pixels


def combine_hdf5_files(temp_files, output_file, release, survey, program):
    """
    Combine temporary HDF5 files into a single HDF5 file while preserving attributes.

    Args:
        temp_files (list): List of temporary HDF5 files.
        output_file (str): Path to the final HDF5 file.
        release (str): Data release version.
        survey (str): Survey name.
        program (str): Observing program.

    Returns:
        None
    """
    with h5py.File(output_file, "w") as h5f_out:
        for temp_file in temp_files:
            if not os.path.exists(temp_file):
                continue
            with h5py.File(temp_file, "r") as h5f_in:
                for key in h5f_in.keys():
                    if key not in h5f_out:
                        h5f_out.create_dataset(key, data=h5f_in[key])
                    else:
                        h5f_out[key].resize(
                            (h5f_out[key].shape[0] + h5f_in[key].shape[0]), axis=0
                        )
                        h5f_out[key][-h5f_in[key].shape[0] :] = h5f_in[key]

                # Copy attributes
                for attr in h5f_in.attrs:
                    h5f_out.attrs[attr] = h5f_in.attrs[attr]

        # Add additional global attributes
        h5f_out.attrs["release"] = release
        h5f_out.attrs["survey"] = survey
        h5f_out.attrs["program"] = program

    # # Cleanup temporary files
    # for temp_file in temp_files:
    #     os.remove(temp_file)


def parallel_preload_qsos(
    catalog_path, spectra_dir, output_file, survey, program, release, n_workers=16
):
    """
    Parallel preload of QSO spectra using healpix-based division.

    Args:
        catalog_path (str): Path to the QSO catalog.
        spectra_dir (str): Base directory for spectra.
        output_file (str): Path for final HDF5 file.
        survey (str): Survey name.
        program (str): Program name.
        release (str): Data release version.
        n_workers (int): Number of parallel workers.

    Returns:
        None
    """
    catalog, _ = read_catalog(catalog_path, balmask=True, bytile=False)
    datapath = os.path.join(spectra_dir, release, "healpix", survey, program)
    unique_hpxpixels = np.unique(catalog["HPXPIXEL"])

    # Create a folder for temporary files
    if not os.path.exists("temp"):
        os.makedirs("temp", exist_ok=True)

    temp_files = [
        os.path.join("temp", f"temp_hpx_{hpx}.h5") for hpx in unique_hpxpixels
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        executor.map(
            process_healpix,
            unique_hpxpixels,
            [catalog] * len(unique_hpxpixels),
            [datapath] * len(unique_hpxpixels),
            [survey] * len(unique_hpxpixels),
            [program] * len(unique_hpxpixels),
            temp_files,
        )

    combine_hdf5_files(temp_files, output_file, release, survey, program)
    log.info(f"Saved preloaded QSO data to {output_file}")


# Example usage
if __name__ == "__main__":
    catalog_path = "/global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits"
    spectra_dir = "/global/cfs/cdirs/desi/spectro/redux/"
    output_file = "preloaded_qsos.h5"
    survey = "main"
    program = "dark"
    release = "kibo"

    parallel_preload_qsos(
        catalog_path, spectra_dir, output_file, survey, program, release, n_workers=256
    )
