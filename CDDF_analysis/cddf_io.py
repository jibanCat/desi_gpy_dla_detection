"""
cddf_io.py — I/O helpers for saving and loading CDDF/dN/dX calibrated results.

Overview
--------
After running the calibration workflow (cddf_mock.py + cddf_calibration.py), the
pipeline saves per-redshift-bin results to plain-text files for downstream use
(paper figures, further analysis).  This module provides:

    save_dndx_combined(...)           Save calibrated + raw dN/dX table to text
    save_cddf_txt_table(...)          Save one z-bin CDDF table (calibrated+raw+ratio)
    save_all_cddf_txt_tables(...)     Save all z-bin CDDF tables from panel_data list
    load_cddf_txt_table(...)          Load a saved CDDF txt table back into a dict

Column conventions (dN/dX files)
---------------------------------
    z
    dNdX_calibrated
    dndx68_low_calibrated  dndx68_high_calibrated
    dndx95_low_calibrated  dndx95_high_calibrated
    dNdX_raw
    dndx68_low_raw  dndx68_high_raw
    dndx95_low_raw  dndx95_high_raw

Column conventions (CDDF files)
---------------------------------
    logN
    f_cal  f_cal_68_lo  f_cal_68_hi  f_cal_95_lo  f_cal_95_hi
    f_raw  f_raw_68_lo  f_raw_68_hi  f_raw_95_lo  f_raw_95_hi
    r_cal  r_68_lo  r_68_hi  r_95_lo  r_95_hi
    [optional] f_true_mock  sigma_true_mock

All 68/95 columns are absolute lower/upper bounds (NOT symmetric errors).
Calibrated CDDF is always saved before raw CDDF.

Reference
---------
Extracted from CDDF_analysis/notebooks/CDDF_dNdX_all.ipynb (cell 52, cell 67)
and CDDF_analysis/notebooks/CDDF_fN_z.ipynb (cell 20).
"""
import os
import re

import numpy as np


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #

def _sanitize_title_to_tag(title):
    """
    Convert a panel title like r"$2.0 \\le z < 2.5$" into a safe filename tag.

    Examples
    --------
    >>> _sanitize_title_to_tag(r"$2.0 \\le z < 2.5$")
    'z2025'
    >>> _sanitize_title_to_tag("3.0 <= z < 3.5")
    'z3035'
    """
    s = str(title)
    nums = re.findall(r"\d+\.\d+", s)
    if len(nums) >= 2:
        a = nums[0].replace(".", "")
        b = nums[1].replace(".", "")
        return f"z{a}{b}"
    return "zbin"


# --------------------------------------------------------------------------- #
# dN/dX output
# --------------------------------------------------------------------------- #

def save_dndx_combined(
    filename,
    z,
    y_raw,
    y68_raw,
    y95_raw,
    y_calibrated,
    y68_calibrated,
    y95_calibrated,
    *,
    meta=None,
):
    """
    Save combined calibrated + raw dN/dX (or Omega_HI) table to a text file.

    Calibrated values are saved first (before raw) so that the most relevant
    measurement is in the leading columns.

    Parameters
    ----------
    filename : str or path-like
        Output file path.
    z : array-like, shape (N,)
        Redshift bin centers.
    y_raw : array-like, shape (N,)
        Raw (uncalibrated) dN/dX central values.
    y68_raw : array-like, shape (N, 2)
        Absolute [low, high] 68% bounds on y_raw.
    y95_raw : array-like, shape (N, 2)
        Absolute [low, high] 95% bounds on y_raw.
    y_calibrated : array-like, shape (N,)
        Calibrated dN/dX central values (alpha(z) corrected).
    y68_calibrated : array-like, shape (N, 2)
        Absolute [low, high] 68% bounds on y_calibrated.
    y95_calibrated : array-like, shape (N, 2)
        Absolute [low, high] 95% bounds on y_calibrated.
    meta : dict, optional
        Metadata key-value pairs written as comment lines in the header
        (e.g. absorber type, NHI range, date).

    Notes
    -----
    - 68/95 bounds are absolute values (NOT ± offsets from the central value).
    - The calibration factor alpha(z) is derived from London mock spectra (z=2–4.25).
    """
    z = np.asarray(z, float)
    y_raw = np.asarray(y_raw, float)
    y68_raw = np.asarray(y68_raw, float)
    y95_raw = np.asarray(y95_raw, float)
    y_calibrated = np.asarray(y_calibrated, float)
    y68_calibrated = np.asarray(y68_calibrated, float)
    y95_calibrated = np.asarray(y95_calibrated, float)

    out = np.column_stack([
        z,
        y_calibrated,
        y68_calibrated[:, 0], y68_calibrated[:, 1],
        y95_calibrated[:, 0], y95_calibrated[:, 1],
        y_raw,
        y68_raw[:, 0], y68_raw[:, 1],
        y95_raw[:, 0], y95_raw[:, 1],
    ])

    header_lines = [
        "z  dNdX_calibrated  dndx68_low_calibrated  dndx68_high_calibrated  "
        "dndx95_low_calibrated  dndx95_high_calibrated  "
        "dNdX_raw  dndx68_low_raw  dndx68_high_raw  dndx95_low_raw  dndx95_high_raw",
        "Calibrated values include alpha(z) correction derived from london mock (z=2-4.25).",
        "68/95 columns are absolute lower/upper bounds (NOT symmetric errors).",
    ]

    if meta is not None:
        for k, v in meta.items():
            header_lines.append(f"{k} = {v}")

    # np.savetxt prepends "# " to every header line automatically, so we join
    # with plain "\n" (not "\n# ") to avoid double comment markers "# # ...".
    header = "\n".join(header_lines)
    np.savetxt(filename, out, header=header)
    print(f"Saved: {filename}")


# --------------------------------------------------------------------------- #
# CDDF f(N,z) output
# --------------------------------------------------------------------------- #

def save_cddf_txt_table(outpath, d, include_truth=True):
    """
    Save one redshift-bin CDDF table as a text file.

    Calibrated CDDF is saved first, followed by raw CDDF, then the
    calibration function (correction ratio r), and optionally the truth values.

    Parameters
    ----------
    outpath : str or path-like
        Output file path.
    d : dict
        Per-redshift-bin data dict with keys:

        Required:
            logN          (N,)          log10(N_HI) bin centers
            f_corr        (N,)          calibrated CDDF f(N,z)
            f68_corr      (N, 2)        absolute 68% [lo, hi] on f_corr
            f95_corr      (N, 2)        absolute 95% [lo, hi] on f_corr
            f_raw         (N,)          raw (uncalibrated) f(N,z)
            f68_raw       (N, 2)        absolute 68% [lo, hi] on f_raw
            f95_raw       (N, 2)        absolute 95% [lo, hi] on f_raw
            r             (N,)          correction ratio r = f_true / f_meas_mock
            r68           (N, 2)        absolute 68% [lo, hi] on r
            r95           (N, 2)        absolute 95% [lo, hi] on r

        Optional:
            title         str           redshift bin label (default 'unknown')
            f_true        (N,)          truth CDDF (Prochaska+2014 or mock truth)
            sig_true      (N,)          1σ error on f_true

    include_truth : bool
        Whether to append f_true_mock and sigma_true_mock columns (default True).
        Skipped if d lacks 'f_true' or 'sig_true'.

    Notes
    -----
    All values are stored in linear space (not log).
    All 68/95 columns are absolute lower/upper bounds (NOT ± offsets).
    """
    logN = np.asarray(d["logN"], dtype=float)

    cols = [
        logN,
        np.asarray(d["f_corr"], dtype=float),
        np.asarray(d["f68_corr"][:, 0], dtype=float),
        np.asarray(d["f68_corr"][:, 1], dtype=float),
        np.asarray(d["f95_corr"][:, 0], dtype=float),
        np.asarray(d["f95_corr"][:, 1], dtype=float),
        np.asarray(d["f_raw"], dtype=float),
        np.asarray(d["f68_raw"][:, 0], dtype=float),
        np.asarray(d["f68_raw"][:, 1], dtype=float),
        np.asarray(d["f95_raw"][:, 0], dtype=float),
        np.asarray(d["f95_raw"][:, 1], dtype=float),
        np.asarray(d["r"], dtype=float),
        np.asarray(d["r68"][:, 0], dtype=float),
        np.asarray(d["r68"][:, 1], dtype=float),
        np.asarray(d["r95"][:, 0], dtype=float),
        np.asarray(d["r95"][:, 1], dtype=float),
    ]

    names = [
        "logN",
        "f_cal", "f_cal_68_lo", "f_cal_68_hi", "f_cal_95_lo", "f_cal_95_hi",
        "f_raw", "f_raw_68_lo", "f_raw_68_hi", "f_raw_95_lo", "f_raw_95_hi",
        "r_cal", "r_68_lo", "r_68_hi", "r_95_lo", "r_95_hi",
    ]

    if include_truth and "f_true" in d and "sig_true" in d:
        cols.extend([
            np.asarray(d["f_true"], dtype=float),
            np.asarray(d["sig_true"], dtype=float),
        ])
        names.extend(["f_true_mock", "sigma_true_mock"])

    arr = np.column_stack(cols)

    header_lines = [
        f"title: {d.get('title', 'unknown')}",
        "columns: " + " ".join(names),
        "notes: values in linear space; calibrated first, raw second, "
        "calibration ratio included; 68/95 columns are absolute lower/upper bounds",
    ]
    header = "\n".join(header_lines)

    np.savetxt(outpath, arr, header=header, fmt="%.8e")
    print(f"Saved: {outpath}")


def save_all_cddf_txt_tables(
    panel_data, outdir, prefix="cddf_calibrated", include_truth=True
):
    """
    Save all redshift-bin CDDF tables from a panel_data list into one directory.

    Creates one text file per z bin, named ``{prefix}_{tag}.txt`` where
    ``tag`` is derived from the bin title (e.g. "z2025" for z=[2.0, 2.5)).

    Parameters
    ----------
    panel_data : list of dict
        Each element is a dict in the format accepted by ``save_cddf_txt_table()``.
        Must contain a 'title' key for file naming.
    outdir : str or path-like
        Output directory (created if it does not exist).
    prefix : str
        Filename prefix (default 'cddf_calibrated').
    include_truth : bool
        Whether to include truth columns (default True).
    """
    os.makedirs(outdir, exist_ok=True)
    for d in panel_data:
        tag = _sanitize_title_to_tag(d.get("title", "zbin"))
        outpath = os.path.join(outdir, f"{prefix}_{tag}.txt")
        save_cddf_txt_table(outpath, d, include_truth=include_truth)


def load_cddf_txt_table(path):
    """
    Load a saved CDDF txt table back into a dict with named columns.

    Parameters
    ----------
    path : str or path-like
        Path to the text file produced by ``save_cddf_txt_table()``.

    Returns
    -------
    dict
        Keys: 'title', 'path', plus one entry per column name.
        Column arrays are 1-D float ndarrays.

    Raises
    ------
    ValueError
        If the file header does not contain a 'columns:' line.
    """
    with open(path, "r") as f:
        header_lines = [line.strip() for line in f if line.startswith("#")]

    col_line = None
    title = None
    for line in header_lines:
        s = line.lstrip("#").strip()
        if s.startswith("columns:"):
            col_line = s
        if s.startswith("title:"):
            title = s.replace("title:", "").strip()

    if col_line is None:
        raise ValueError(f"Could not find 'columns:' header in {path}")

    colnames = col_line.replace("columns:", "").strip().split()
    data = np.loadtxt(path)

    out = {"title": title, "path": str(path)}
    for i, name in enumerate(colnames):
        out[name] = data[:, i]
    return out
