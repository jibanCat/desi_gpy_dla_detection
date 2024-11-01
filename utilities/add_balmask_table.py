#!/usr/bin/env python
import os
import argparse
import numpy as np
from astropy.table import Table
import constants
from fitwarning import DLAFLAG
import fitsio

from desiutil.log import log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check for BAL contamination in DLA catalog."
    )

    parser.add_argument(
        "--dlacat",
        type=str,
        required=True,
        help="Path to the DLA catalog FITS file.",
    )

    parser.add_argument(
        "-q",
        "--qsocat",
        type=str,
        default=None,
        required=True,
        help="path to quasar catalog",
    )

    parser.add_argument(
        "--mocks",
        default=False,
        required=False,
        action="store_true",
        help="is this a mock catalog? Default is False",
    )

    parser.add_argument(
        "--mockdir",
        type=str,
        default=None,
        required=False,
        help="path to mock directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the updated DLA catalog with BAL contamination flags.",
    )
    return parser.parse_args()


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
    table of relevant attributes for z>2 quasars

    """

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
                ]
            catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
        except:
            log.error(f"cannot find {cols} in quasar catalog")
            exit(1)
    else:
        # read the following columns from qsocat
        cols = ["TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "HPXPIXEL"]
        if bytile:
            cols = ["TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "TILEID", "PETAL_LOC"]
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    log.info(f"Successfully read quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    catalog = catalog[zmask]

    return catalog


def read_mock_catalog(qsocat, balmask, mockpath):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes be read in?
    mockpath (str) : path to mock data

    Returns
    -------
    table of relevant attributes for z>2 quasars

    """
    # read the following columns from qsocat
    cols = ["TARGETID", "RA", "DEC", "Z"]
    catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
    log.info(f"Successfully read mock quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    catalog = catalog[zmask]

    if balmask:
        try:
            # open bal catalog
            balcat = os.path.join(mockpath, "bal_cat.fits")
            cols = ["TARGETID", "AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450"]
            balcat = Table(fitsio.read(balcat, ext=1, columns=cols))

            # add columns to catalog
            ai = np.full(len(catalog), 0.0)
            nciv = np.full(len(catalog), 0)
            vmin = np.full((len(catalog), balcat["VMIN_CIV_450"].shape[1]), -1.0)
            vmax = np.full((len(catalog), balcat["VMIN_CIV_450"].shape[1]), -1.0)

            for i, tid in enumerate(catalog["TARGETID"]):
                if np.any(tid == balcat["TARGETID"]):
                    match = balcat[balcat["TARGETID"] == tid]
                    ai[i] = match["AI_CIV"]
                    nciv[i] = match["NCIV_450"]
                    vmin[i] = match["VMIN_CIV_450"]
                    vmax[i] = match["VMAX_CIV_450"]

            catalog.add_columns(
                [ai, nciv, vmin, vmax],
                names=["AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450"],
            )

        except:
            log.error(f"cannot find mock bal_cat.fits in {mockpath}")
            exit(1)

    return catalog


def main():
    args = parse_args()

    # Load the DLA and BAL catalogs
    dlacat = Table.read(args.dlacat)

    if args.mocks:
        log.info("Reading mock catalog")
        catalog = read_mock_catalog(args.qsocat, True, args.mockdir)
    else:
        log.info("Reading quasar catalog")
        catalog = read_catalog(args.qsocat, True, False)

    num_dla = len(dlacat)
    for i in range(num_dla):
        # Find the indices of the target ID in the BAL catalog
        entries = np.where(catalog["TARGETID"] == dlacat["TARGETID"][i])[0]

        if len(entries) == 0:
            # No match found in the BAL catalog; skip this entry
            continue

        entry = entries[0]  # Take the first matching entry if duplicates exist

        # Apply mask to BAL features if they are available
        nbal = catalog["NCIV_450"][entry]

        if nbal == 0:
            # No BAL features; continue to the next DLA
            continue

        # Get the fit warning flags for the DLA entry
        fitwarn = dlacat["DLAFLAG"][i]
        # Get the redshift of the QSO and the DLA
        zqso = dlacat["Z_QSO"][i]
        zdla = dlacat["Z_DLA"][i]

        # Track BAL contamination regions
        bal_locs = []

        for n in range(nbal):
            # Compute velocity ranges
            v_max = -catalog[entry]["VMAX_CIV_450"][n] / constants.c + 1.0
            v_min = -catalog[entry]["VMIN_CIV_450"][n] / constants.c + 1.0

            for line, lam in constants.bal_lines.items():
                # Mask wavelengths within the velocity ranges
                # mask = np.logical_and(wave_rf > lam * v_max, wave_rf < lam * v_min)
                if (line == "Lya") or (line == "NV"):
                    rededge = (lam * v_min) * (1 + zqso)
                    blueedge = (lam * v_max) * (1 + zqso)
                    bal_locs.append((rededge, blueedge))

        # check for potential BAL contamination in solution
        # false positive should only come from Lya and NV - all other lines too weak
        if "nbal" in locals():
            lam_center_dla = constants.Lya_line * (1 + zdla)
            for window in bal_locs:
                balflag = (lam_center_dla < window[0]) & (lam_center_dla > window[1])
                if balflag:
                    log.info(
                        f"Potential BAL contamination in DLA {i} in target {dlacat['TARGETID'][i]}"
                    )
                    fitwarn |= DLAFLAG.POTENTIAL_BAL

        # Update the DLA catalog with the updated flag
        dlacat["DLAFLAG"][i] = fitwarn

    # Save the updated DLA catalog
    dlacat.write(args.output, overwrite=True)
    log.info(f"Updated DLA catalog saved to {args.output}")


if __name__ == "__main__":
    main()

# python utilities/add_balmask_table.py \
# --dlacat ../desi-mock-gpdla/dlacat-v5.9.5-mockcat.fits \
# -q /global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/zcat.fits \
# --mocks \
# --mockdir /global/cfs/projectdirs/desi/mocks/lya_forest/develop/london/qq_desi_y3/v5.9.5/mock-0/jura-124/ \
# --output dlacat-v5.9.5-mockcat-balflag.fits


# python utilities/add_balmask_table.py \
#     --dlacat ../desi-kibo-gpdla/dlacat-kibo-main-dark.fits \
#     -q /global/cfs/cdirs/desi/users/martini/bal-catalogs/kibo/QSO_cat_kibo_main_dark_healpix_v3-altbal.fits \
#     --output dlacat-kibo-main-dark-balflag.fits
