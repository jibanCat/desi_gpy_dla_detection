# include .. in the path
import os
import sys
sys.path.append("..")

import constants
from desiutil.log import log
import fitsio

from astropy.table import Table, vstack
import numpy as np
from matplotlib import pyplot as plt

import argparse


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
    cols = ["TARGETID", "RA", "DEC", "Z", "ZWARN", "SPECTYPE"]
    try:
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
    except:
        print("[Warning] cannot find TARGETID, RA, DEC, Z in quasar catalog")
        print("... using Saclay cols instead.")
        cols = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z', "ZWARN", "SPECTYPE"]
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

def filter_qsos(qso_catalog, dla_catalog, bal_catalog=None):
    """Apply filtering criteria to QSO catalog."""
    ind = (qso_catalog["Z"] > 2.15) & (qso_catalog["Z"] < 4.25)
    print("Number of QSOs in the catalog:", ind.sum())

    if bal_catalog is not None:
        balind = np.isin(qso_catalog["TARGETID"], bal_catalog["TARGETID"])
        ind &= ~balind
        print("Number of QSOs without BALs:", ind.sum())
    else:
        balind = qso_catalog["NCIV_450"] > 0
        ind &= ~balind
        print("Number of QSOs without BALs:", ind.sum())

    zwarnind = qso_catalog["ZWARN"] == 0
    ind &= zwarnind
    print("Number of QSOs without ZWARN:", ind.sum())

    if "SPECTYPE" in qso_catalog.colnames:
        spectypeind = qso_catalog["SPECTYPE"] == "QSO"
        ind &= spectypeind
        print("Number of QSOs with SPECTYPE == QSO:", ind.sum())

    dlaind = np.isin(qso_catalog["TARGETID"], dla_catalog["TARGETID"])
    ind &= ~dlaind
    print("Number of QSOs without DLAs:", ind.sum())

    return qso_catalog[ind]

def main():
    parser = argparse.ArgumentParser(description="Prepare the training set for the GP model.")
    parser.add_argument("--qsocat", type=str, required=True, help="Path to the QSO catalog.")
    parser.add_argument("--dlacat", type=str, required=True, help="Path to the DLA catalog.")
    parser.add_argument("--balcat", type=str, default=None, help="Path to the BAL catalog (optional).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the filtered QSO catalog.")
    parser.add_argument("--is_mock", action="store_true", help="Use this flag if processing mock catalog.")
    args = parser.parse_args()

    if args.is_mock:
        catalog = read_mock_catalog(args.qsocat, False, "data/london/")
    else:
        catalog = read_catalog(args.qsocat, True, False)
    
    dla_catalog = Table(fitsio.read(args.dlacat, ext=1))
    bal_catalog = Table(fitsio.read(args.balcat, ext=1)) if args.balcat else None
    
    filtered_catalog = filter_qsos(catalog, dla_catalog, bal_catalog)
    filtered_catalog.write(args.output, overwrite=True)
    print(f"Filtered catalog saved to {args.output}")

if __name__ == "__main__":
    main()
