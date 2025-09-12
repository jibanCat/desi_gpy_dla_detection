from astropy.table import Table
import numpy as np
import fitsio

import os

# include the path to the gpy_dla_detection module
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants

from desiutil.log import log



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
    try:
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
    except:
        print("[Warning] cannot find TARGETID, RA, DEC, Z in quasar catalog")
        print("... using Saclay cols instead.")
        cols = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z']
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
