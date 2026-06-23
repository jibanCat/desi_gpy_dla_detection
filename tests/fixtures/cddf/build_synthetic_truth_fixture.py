"""Synthetic truth-catalog writer for the O3 diagonal-deposit tests.

Writes a tiny ``hcd_truth_cat.fits``-style FITS table with the exact columns the
real 2LPT-0 truth catalog carries (``NHI`` log10, ``Z`` absorber redshift,
``TARGETID``), so ``build_truth_map`` can be exercised on the same schema without
touching real data.
"""
import numpy as np
from astropy.table import Table


def write_truth_catalog(path, *, target_ids, nhi, z, dlaid=None):
    """Write a minimal HCD truth FITS catalog (NHI, Z, TARGETID[, DLAID])."""
    target_ids = np.asarray(target_ids, dtype=np.int64)
    nhi = np.asarray(nhi, dtype=float)
    z = np.asarray(z, dtype=float)
    assert target_ids.shape == nhi.shape == z.shape
    cols = {
        "NHI": nhi,
        "Z": z,
        "TARGETID": target_ids,
    }
    if dlaid is None:
        dlaid = target_ids.astype(np.int64) * 1000
    cols["DLAID"] = np.asarray(dlaid, dtype=np.int64)
    Table(cols).write(path, overwrite=True)
    return path
