"""O1 end-to-end CDDF driver — a faithful wrapper over the Pathway-A estimator.

"O1" is the raw probabilistic CDDF with **no selection correction**: exactly the
numbers the existing estimator (``CDDF_analysis.calc_cddf.DLACatalogue``) produces.
This driver is therefore a thin, parameterized WRAPPER that:

  1. refuses FILTER-on catalogs (``assert_filter_off``) — the Monte-Carlo CDDF is
     only valid on FILTER-off runs, and the schema does not persist the FILTER flag
     (see ``filter_guard.py``), so the caller supplies it;
  2. constructs ONE :class:`~CDDF_analysis.calc_cddf.DLACatalogue`; and
  3. calls its three public statistical methods —
     ``column_density_function`` (CDDF f(N)), ``line_density`` (dN/dX), and
     ``omega_dla_cddf`` (Omega_DLA) — and packages the results into a dict.

It does **not** alter any estimator number (pinned by the faithfulness test in
``tests/test_cddf_driver.py``).  Later milestones (O3/O4) add the selection
corrections; this is the un-corrected baseline.

WindowSpec scope (O1)
---------------------
A :class:`~CDDF_analysis.cddf_forward.window.WindowSpec`, if supplied, is RECORDED
in ``provenance`` but is **not yet used** to re-cut the search window — wiring a
``WindowSpec`` into ``calc_cddf``'s binning (replacing its hard-coded
``proximity_zone``/``tail_zone``) is a LATER milestone (see ``window.py`` "M0 scope").
For O1 the estimator's own window cuts (governed by ``lowzcut``/``highzcut`` etc.
passed through ``dlacat_kwargs``) are what take effect.
"""
import os
from typing import Optional

from .filter_guard import assert_filter_off
from .window import WindowSpec
from .. import cddf_io
from ..calc_cddf import DLACatalogue


def compute_o1_products(
    processed_file: str,
    sample_file: str,
    catalog_file: str,
    *,
    z_min: float,
    z_max: float,
    lnhi_min: float = 20.3,
    lnhi_max: float = 22.5,
    lnhi_nbins: int = 30,
    hubble: float = 0.7,
    filter_low_likelihood: int = 0,
    window: Optional[WindowSpec] = None,
    **dlacat_kwargs,
) -> dict:
    """Compute the O1 (uncorrected) CDDF, dN/dX, and Omega_DLA from one catalog.

    Faithful wrapper: the returned arrays are byte-identical to calling
    :class:`~CDDF_analysis.calc_cddf.DLACatalogue` directly with the same args.

    Parameters
    ----------
    processed_file : str
        HDF5 output of the GP-DLA inference pipeline (per-spectrum posteriors).
    sample_file : str
        QMC sample grid (.mat HDF5) — ``offset_samples`` + ``log_nhi_samples``.
    catalog_file : str
        FITS QSO catalog with ``TARGETID`` (aligns the processed file to a reference).
    z_min, z_max : float
        Redshift range over which all three products are computed.
    lnhi_min : float, default 20.3
        Lower log10(N_HI) bound (DLA threshold).
    lnhi_max : float, default 22.5
        Upper log10(N_HI) bound.
    lnhi_nbins : int, default 30
        Number of log10(N_HI) bins for the CDDF and the Omega_DLA integration.
    hubble : float, default 0.7
        H_0 / (100 km/s/Mpc), used by ``omega_dla_cddf``.
    filter_low_likelihood : int, default 0
        The ``FILTER_LOW_LIKELIHOOD`` setting of the run that produced
        ``processed_file``.  Must be 0 (FILTER-off); anything else raises
        ``ValueError`` *before* any work — the CDDF is invalid on FILTER-on runs and
        the schema does not persist this flag, so the caller supplies it.
    window : WindowSpec, optional
        Search-window spec.  RECORDED in ``provenance`` but NOT yet used to re-cut
        the window for O1 (see module docstring) — a later milestone wires it in.
    **dlacat_kwargs
        Forwarded verbatim to the ``DLACatalogue`` constructor (e.g. ``sub_dla``,
        ``snr``, ``lowzcut``, ``highzcut``, ``occams_razor``, ``second``).

    Returns
    -------
    dict
        ``{"cddf": {...}, "dndx": {...}, "omega": {...}, "provenance": {...}}``.
        ``cddf``  : ``logN, f, f68, f95, xerrs``
        ``dndx``  : ``z, dndx, dndx68, dndx95, xerrs``
        ``omega`` : ``z, omega, omega68, omega95, xerrs``
        ``provenance`` : the inputs, the WindowSpec (if any), ``filter_low_likelihood``,
        the z/logN/hubble bin params, and the sample-file path.

    Raises
    ------
    ValueError
        If ``filter_low_likelihood`` is non-zero / unknown (FILTER guard).
    """
    import warnings

    # FILTER guard FIRST — before constructing the catalogue or touching any file,
    # so a FILTER-on catalog is refused with no side effects.
    assert_filter_off(filter_low_likelihood, ctx="compute_o1_products")

    # Fail fast with a directed message if pointed at a per-healpix directory: the
    # real GP-DLA run is ~1000 per-healpix .h5 files; DLACatalogue needs the single
    # COMBINED file (run combine_processed_h5.py first).
    if os.path.isdir(processed_file):
        raise ValueError(
            f"processed_file={processed_file!r} is a directory. The CDDF needs a "
            "single COMBINED HDF5 (run combine_processed_h5.py on the per-healpix "
            "outputs first), not the per-healpix processed/ directory."
        )

    # The estimator's sample-selection cap (high_nhi_cut_value) must track lnhi_max,
    # else CDDF bins above the cap receive zero samples -> a spurious empty high-N
    # tail (worst for Omega_DLA, which is N-weighted). Default the cap to lnhi_max so
    # the bin range and the sample cut are consistent; the caller can still override.
    dlacat_kwargs.setdefault("high_nhi_cut_value", lnhi_max)

    if window is not None:
        warnings.warn(
            "WindowSpec is RECORDED ONLY in O1 (provenance['window_applied']=False); "
            "the active search window is the estimator's own cuts "
            "(proximity_zone/tail_zone gated by lowzcut/highzcut). With the CLI "
            "default lowzcut=highzcut=0, NO proximity/tail cut is applied. Wiring the "
            "WindowSpec into the binning is a later milestone.",
            stacklevel=2,
        )

    cat = DLACatalogue(
        processed_file=processed_file,
        sample_file=sample_file,
        catalog_file=catalog_file,
        **dlacat_kwargs,
    )

    l_Ncent, cddf, cddf68, cddf95, cddf_xerrs = cat.column_density_function(
        z_min=z_min,
        z_max=z_max,
        lnhi_nbins=lnhi_nbins,
        lnhi_min=lnhi_min,
        lnhi_max=lnhi_max,
    )
    z_cent_d, dNdX, dndx68, dndx95, dndx_xerrs = cat.line_density(
        z_min=z_min, z_max=z_max, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    z_cent_o, omega, omega68, omega95, omega_xerrs = cat.omega_dla_cddf(
        z_min=z_min,
        z_max=z_max,
        hubble=hubble,
        lnhi_nbins=lnhi_nbins,
        lnhi_min=lnhi_min,
        lnhi_max=lnhi_max,
    )

    provenance = {
        "processed_file": processed_file,
        "sample_file": sample_file,
        "catalog_file": catalog_file,
        "filter_low_likelihood": int(filter_low_likelihood),
        "z_min": z_min,
        "z_max": z_max,
        "lnhi_min": lnhi_min,
        "lnhi_max": lnhi_max,
        "lnhi_nbins": lnhi_nbins,
        "hubble": hubble,
        "dlacat_kwargs": dict(dlacat_kwargs),
        # Recorded but not yet used to re-cut the window (O1); see module docstring.
        "window": window,
        "window_applied": False,
    }

    return {
        "cddf": {
            "logN": l_Ncent,
            "f": cddf,
            "f68": cddf68,
            "f95": cddf95,
            "xerrs": cddf_xerrs,
        },
        "dndx": {
            "z": z_cent_d,
            "dndx": dNdX,
            "dndx68": dndx68,
            "dndx95": dndx95,
            "xerrs": dndx_xerrs,
        },
        "omega": {
            "z": z_cent_o,
            "omega": omega,
            "omega68": omega68,
            "omega95": omega95,
            "xerrs": omega_xerrs,
        },
        "provenance": provenance,
    }


def save_o1_products(products: dict, out_dir: str) -> dict:
    """Write the O1 CDDF + dN/dX text tables via ``cddf_io`` (reusing its writers).

    For O1 there is no calibration, so the "calibrated" columns of the reused
    ``cddf_io`` writers are filled with the same raw values and the correction ratio
    ``r`` is identically 1 — keeping the on-disk schema readable by
    ``cddf_io.load_cddf_txt_table`` / ``save_dndx_combined`` without forking the IO.

    Parameters
    ----------
    products : dict
        The dict returned by :func:`compute_o1_products`.
    out_dir : str
        Output directory (created if absent).

    Returns
    -------
    dict
        ``{"cddf": <cddf table path>, "dndx": <dndx table path>}``.
    """
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    prov = products["provenance"]
    meta = {
        "product": "O1 (raw probabilistic CDDF, no selection correction)",
        "processed_file": prov["processed_file"],
        "sample_file": prov["sample_file"],
        "catalog_file": prov["catalog_file"],
        "filter_low_likelihood": prov["filter_low_likelihood"],
        "z_min": prov["z_min"],
        "z_max": prov["z_max"],
    }

    # --- CDDF f(N) table (reuse cddf_io.save_cddf_txt_table) ----------------------
    c = products["cddf"]
    n = np.asarray(c["f"]).shape[0]
    ones2 = np.ones((n, 2), dtype=float)
    cddf_dict = {
        "title": (
            f"O1 UNCORRECTED (f_cal == f_raw, no selection/alpha correction) "
            f"{prov['z_min']} <= z < {prov['z_max']}"
        ),
        "logN": c["logN"],
        # O1 has no calibration: calibrated == raw, ratio == 1.
        "f_corr": c["f"],
        "f68_corr": c["f68"],
        "f95_corr": c["f95"],
        "f_raw": c["f"],
        "f68_raw": c["f68"],
        "f95_raw": c["f95"],
        "r": np.ones(n, dtype=float),
        "r68": ones2,
        "r95": ones2,
    }
    cddf_path = os.path.join(out_dir, "o1_cddf.txt")
    cddf_io.save_cddf_txt_table(cddf_path, cddf_dict, include_truth=False)

    o1_note = "O1: UNCORRECTED — 'calibrated' columns == raw; no alpha(z) applied."

    # --- dN/dX table (reuse cddf_io.save_dndx_combined) --------------------------
    d = products["dndx"]
    dndx_path = os.path.join(out_dir, "o1_dndx.txt")
    cddf_io.save_dndx_combined(
        dndx_path,
        d["z"],
        d["dndx"],
        d["dndx68"],
        d["dndx95"],
        d["dndx"],  # calibrated == raw for O1
        d["dndx68"],
        d["dndx95"],
        meta=meta,
        calibration_note=o1_note,
    )

    # --- Omega_DLA table (reuse the generic dN/dX writer; y = Omega) -------------
    # Convention: Omega is Omega_HI in DLAs = (m_p H0 / c rho_c) integral N f dN,
    # neutral-H only (NO helium / X_H correction; multiply by ~1/0.76 for total gas).
    o = products["omega"]
    omega_path = os.path.join(out_dir, "o1_omega.txt")
    omega_meta = dict(meta)
    omega_meta["quantity"] = "Omega_HI in DLAs (neutral-H only, no X_H/He correction)"
    cddf_io.save_dndx_combined(
        omega_path,
        o["z"],
        o["omega"],
        o["omega68"],
        o["omega95"],
        o["omega"],  # calibrated == raw for O1
        o["omega68"],
        o["omega95"],
        meta=omega_meta,
        calibration_note=o1_note,
    )

    return {"cddf": cddf_path, "dndx": dndx_path, "omega": omega_path}


def main(argv=None):
    """Thin argparse CLI: run the O1 driver and save text tables.

    Example
    -------
        python -m CDDF_analysis.cddf_forward.driver \\
            --processed_file processed.h5 --sample_file dla_samples_a03.mat \\
            --catalog_file catalog.fits --out_dir o1_out \\
            --z_min 2.0 --z_max 4.0 --filter_low_likelihood 0 --sub_dla 0
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="O1 end-to-end CDDF driver (raw probabilistic CDDF, no correction)."
    )
    parser.add_argument(
        "--processed_file",
        required=True,
        help="Single COMBINED processed HDF5 (run combine_processed_h5.py first; "
        "NOT the per-healpix processed/ directory).",
    )
    parser.add_argument("--sample_file", required=True)
    parser.add_argument("--catalog_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--z_min", type=float, default=2.0)
    parser.add_argument("--z_max", type=float, default=4.0)
    parser.add_argument("--lnhi_min", type=float, default=20.3)
    parser.add_argument("--lnhi_max", type=float, default=22.5)
    parser.add_argument("--lnhi_nbins", type=int, default=30)
    parser.add_argument("--hubble", type=float, default=0.7)
    parser.add_argument(
        "--filter_low_likelihood",
        type=int,
        required=True,
        help="FILTER_LOW_LIKELIHOOD of the run, from its .env (must be 0; the CDDF is "
        "FILTER-off only). REQUIRED — the schema does not persist it and the V1 "
        "production baseline is FILTER-ON, so it must be declared, not defaulted.",
    )
    # Common DLACatalogue cut flags (mirrors desi_cddf.py); 0/1 for the bool ones so
    # the default can be expressed explicitly.
    parser.add_argument("--sub_dla", type=int, default=0)
    parser.add_argument("--snr", type=float, default=-2)
    parser.add_argument("--lowzcut", type=int, default=0)
    parser.add_argument("--highzcut", type=int, default=0)
    parser.add_argument("--occams_razor", type=float, default=1)
    parser.add_argument("--second", type=int, default=0)

    args = parser.parse_args(argv)

    products = compute_o1_products(
        args.processed_file,
        args.sample_file,
        args.catalog_file,
        z_min=args.z_min,
        z_max=args.z_max,
        lnhi_min=args.lnhi_min,
        lnhi_max=args.lnhi_max,
        lnhi_nbins=args.lnhi_nbins,
        hubble=args.hubble,
        filter_low_likelihood=args.filter_low_likelihood,
        sub_dla=bool(args.sub_dla),
        snr=args.snr,
        lowzcut=bool(args.lowzcut),
        highzcut=bool(args.highzcut),
        occams_razor=args.occams_razor,
        second=args.second,
    )
    paths = save_o1_products(products, args.out_dir)
    print("Wrote O1 products:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return paths


if __name__ == "__main__":
    main()
