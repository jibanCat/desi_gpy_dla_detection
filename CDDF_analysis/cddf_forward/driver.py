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

import numpy as np

from .filter_guard import assert_filter_off, assert_filter_off_from_file
from .window import WindowSpec
from .split import split_masks, assert_no_leakage
from .diagonal_deposit import build_truth_map, DiagonalSoftDeposit
from .. import cddf_io
from ..calc_cddf import DLACatalogue

# The Bayesian core is built in parallel (contract §2). Import it lazily/softly so
# this module imports even before the core lands; the O3 driver dereferences
# ``soft_completeness`` at call time (and tests inject a fake via monkeypatch).
try:  # pragma: no cover - import wiring
    from . import soft_completeness  # type: ignore
except Exception:  # pragma: no cover
    soft_completeness = None  # type: ignore


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


# =============================================================================
# O3 — diagonal soft-completeness correction (contract §3.3 / §3.5 / §3.6)
# =============================================================================
#
# O3 corrects the raw O1 count F_b in each (logN, z) bin by its OWN completeness
# C_b and soft false-positive deposit b_FP_b:  n_corr_b = (F_b - b_FP_b) / C_b.
# DIAGONAL: no cross-bin migration (sub-DLA<->DLA across logN=20.3, z-scatter,
# LLS<->DLA leakage) is modelled — those are off-diagonal channels (O4 / M3-M4),
# explicitly out of scope.  Every artifact is labelled accordingly.
#
# This driver is the CS plumbing; the per-bin C / b_FP / n_corr maths lives in the
# Bayesian core (``soft_completeness``), built in parallel.  The driver:
#   1. refuses FILTER-on (§3.3);
#   2. measures C & b_FP on the BUILD split (truth map + partitioned deposit + core
#      2.1/2.2);
#   3. applies the diagonal correction (core 2.3) to the WHOLE-sample F (the
#      science product) in COUNT space, then re-normalizes to f(N)/dN/dX.

_O3_LABEL = "O3 DIAGONAL SOFT-COMPLETENESS CORRECTED"
_O3_LIMITATION = (
    "DIAGONAL correction only: each (logN, z) bin is corrected by its OWN "
    "completeness C and soft false-positive deposit b_FP. NO cross-bin migration "
    "(sub-DLA<->DLA across logN=20.3, z-scatter, LLS<->DLA leakage) is modelled — "
    "those off-diagonal channels are out of scope (O4 / M3-M4)."
)


def _require_core():
    """Return the Bayesian core module, or raise a directed error if absent."""
    if soft_completeness is None:
        raise ImportError(
            "the O3 Bayesian core CDDF_analysis.cddf_forward.soft_completeness is "
            "not available; it is built in parallel (contract §2). Install it, or "
            "inject a fake matching the §2 signatures in tests."
        )
    return soft_completeness


def _correct_1d(core, counts, counts_ci_lo68, counts_ci_hi68,
                counts_ci_lo95, counts_ci_hi95, f_matched, f_unmatched, n_truth,
                exposure, *, return_draws=False):
    """Run core 2.1/2.2/2.3 on a 1-D (collapsed) count vector → correction dict.

    ``counts`` and the ``f_*`` / ``n_truth`` are 1-D over the SURVIVING axis (logN
    for the CDDF; z for dN/dX), the orthogonal axis already collapsed to a single
    window bin.  Returns the core's ``apply_diagonal_correction`` dict plus the
    completeness / FP estimator dicts for provenance.

    ``return_draws`` asks the core for the JOINT per-draw ``n_corr`` array
    (``n_corr_draws``), used by the C6 Ω-from-draws derivation.  Passed through only
    if the core advertises the keyword (it is added by the parallel B2 core fix);
    older cores silently fall back to the no-draws path.
    """
    C_est = core.estimate_diagonal_completeness(f_matched, n_truth)
    bfp_est = core.estimate_false_positive_deposit(f_unmatched, exposure)
    F_ci = {
        "lo68": counts_ci_lo68,
        "hi68": counts_ci_hi68,
        "lo95": counts_ci_lo95,
        "hi95": counts_ci_hi95,
    }
    if return_draws and _core_supports_return_draws(core):
        corr = core.apply_diagonal_correction(
            counts, F_ci, C_est, bfp_est, return_draws=True
        )
    else:
        corr = core.apply_diagonal_correction(counts, F_ci, C_est, bfp_est)
    return corr, C_est, bfp_est


def _core_supports_return_draws(core) -> bool:
    """True iff the core's ``apply_diagonal_correction`` accepts ``return_draws``."""
    import inspect

    try:
        sig = inspect.signature(core.apply_diagonal_correction)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return "return_draws" in sig.parameters


def compute_o3_products(
    processed_file: str,
    sample_file: str,
    catalog_file: str,
    truth_file: str,
    *,
    z_min: float,
    z_max: float,
    lnhi_min: float = 20.3,
    lnhi_max: float = 22.5,
    lnhi_nbins: int = 30,
    hubble: float = 0.7,
    filter_low_likelihood: int = 0,
    window: Optional[WindowSpec] = None,
    split_seed: int = 20260609,
    build_frac: float = 0.7,
    **dlacat_kwargs,
) -> dict:
    """Compute the O3 diagonal soft-completeness-corrected CDDF / dN/dX / Ω.

    Contract §3.3 + review-fix decision §1 (the count-basis fix).  The SCIENCE
    product estimates F, C, b_FP, and n_truth ALL on the **WHOLE ACTIVE sightline
    set** ("active" = the estimator's ``filter_dla_spectra()`` set, with the
    SNR / catalog-membership / z-range cuts) — NO BUILD/HELDOUT split and NO
    rescaling, so F, b_FP, and n_truth share one population and the subtraction
    ``F − b_FP`` is dimensionally consistent.  F is the partitioned MEAN-count
    deposit total (``F_matched + F_unmatched``), NOT the Poisson-binomial MAP from
    ``column_density_function_counts`` (the MAP supplies only the F-uncertainty
    WIDTH, re-centred on the deposit mean); O3 therefore corrects the posterior-
    MEAN counts while O1 reports the MAP.  The BUILD/HELDOUT split is used ONLY by
    :func:`heldout_closure` (the anti-circularity validation), which rebases the
    BUILD b_FP to the HELDOUT basis — it is NOT the science estimate.

    Truth absorbers and recovered samples are windowed IDENTICALLY via the SAME
    ``window``.  Diagonal only — no cross-bin migration (see module banner).

    Returns
    -------
    dict
        ``o1``           : the raw O1 products dict (from :func:`compute_o1_products`);
        ``completeness`` : ``{C, b_FP, n_truth, F_matched, F_unmatched, C_est,
                           bfp_est}`` on the CDDF (logN) grid (WHOLE active set);
        ``o3_cddf``      : ``{logN, f, f68, f95, valid_mask, neg_clip_mask}``;
        ``o3_dndx``      : ``{z, dndx, dndx68, dndx95, valid_mask, neg_clip_mask}``;
        ``o3_omega``     : Ω(z) integrated from the corrected count-space CDDF;
        ``closure``      : ``{}`` placeholder (run :func:`heldout_closure` separately);
        ``provenance``   : inputs, split, window, ``window_applied=True``, edges.
    """
    assert_filter_off_from_file(
        processed_file, supplied=filter_low_likelihood, ctx="compute_o3_products"
    )
    if os.path.isdir(processed_file):
        raise ValueError(
            f"processed_file={processed_file!r} is a directory; the CDDF needs a "
            "single COMBINED HDF5 (run combine_processed_h5.py first)."
        )
    if window is None:
        window = WindowSpec()

    core = _require_core()

    dlacat_kwargs.setdefault("high_nhi_cut_value", lnhi_max)

    # --- O1 raw products (faithful, reused verbatim) ---------------------------
    o1 = compute_o1_products(
        processed_file,
        sample_file,
        catalog_file,
        z_min=z_min,
        z_max=z_max,
        lnhi_min=lnhi_min,
        lnhi_max=lnhi_max,
        lnhi_nbins=lnhi_nbins,
        hubble=hubble,
        filter_low_likelihood=filter_low_likelihood,
        window=window,
        **dlacat_kwargs,
    )

    # --- one windowed catalogue for the count-space + deposit work -------------
    cat = DLACatalogue(
        processed_file=processed_file,
        sample_file=sample_file,
        catalog_file=catalog_file,
        window=window,
        **dlacat_kwargs,
    )

    # C1/C2 (cross-cutting decision §1): the SCIENCE path estimates F, C, b_FP, and
    # n_truth ALL on the WHOLE ACTIVE sightline set — "active" = the estimator's
    # filter_dla_spectra() (SNR / catalog-membership / z-range cuts).  NO BUILD/
    # HELDOUT split, NO rescaling.  F is the partitioned MEAN-count deposit total
    # (F_matched + F_unmatched), NOT column_density_function_counts (the
    # Poisson-binomial MAP).  O3 therefore corrects the posterior-MEAN counts; the
    # MAP (O1) remains available in the "o1" block and as the count-CI WIDTH source.
    active_target_ids = set(int(t) for t in cat.target_ids[cat.filter_dla_spectra()[0]])

    # ===== CDDF correction (logN bins; z collapsed to [z_min, z_max]) ==========
    cddf_counts = cat.column_density_function_counts(
        z_min=z_min, z_max=z_max, lnhi_nbins=lnhi_nbins,
        lnhi_min=lnhi_min, lnhi_max=lnhi_max,
    )
    lnhi_edges = np.linspace(lnhi_min, lnhi_max, lnhi_nbins + 1)
    z_edges_cddf = np.array([z_min, z_max])  # single z bin spanning the window

    tmap_cddf = build_truth_map(
        truth_file, catalog_file=catalog_file, processed_file=processed_file,
        window=window, lnhi_edges=lnhi_edges, z_edges=z_edges_cddf,
        active_target_ids=active_target_ids,
    )
    dep_cddf = DiagonalSoftDeposit(
        cat, tmap_cddf, lnhi_edges=lnhi_edges, z_edges=z_edges_cddf,
        window=window,
    )
    # WHOLE active set (no target_ids subset): the science deposit.
    part_cddf = dep_cddf.deposit(z_min=z_min, z_max=z_max)
    # collapse the singleton z axis → 1-D over logN
    f_matched = part_cddf["F_matched"][:, 0]
    f_unmatched = part_cddf["F_unmatched"][:, 0]
    n_truth_cddf = part_cddf["n_truth"][:, 0]
    # F fed to the correction IS the deposit mean-count total (C1/C2), NOT the MAP.
    F_cddf = f_matched + f_unmatched

    # exposure for the FP rate: the whole-active path length (same count basis as F,
    # which is a per-bin expected COUNT) — documented count↔rate contract in §2.2.
    exposure_cddf = float(cddf_counts["dX"])

    # The count-space CI from the estimator supplies the F-uncertainty WIDTH (its
    # Poisson-binomial 68/95); the central value handed to the core is the deposit
    # mean F.  We re-center the MAP CI on the deposit mean so the half-widths carry
    # through without re-introducing the MAP as the point.
    cddf_ci = _recenter_count_ci(cddf_counts, F_cddf)
    corr_cddf, C_est, bfp_est = _correct_1d(
        core,
        F_cddf,
        cddf_ci["lo68"], cddf_ci["hi68"], cddf_ci["lo95"], cddf_ci["hi95"],
        f_matched, f_unmatched, n_truth_cddf, exposure_cddf,
        return_draws=True,  # C6: joint per-draw n_corr for Ω
    )
    dN = cddf_counts["dN"]
    dX = cddf_counts["dX"]
    o3_cddf = {
        "logN": cddf_counts["logN"],
        "f": corr_cddf["n_corr"] / dX / dN,
        "f68": np.vstack([corr_cddf["lo68"], corr_cddf["hi68"]]).T / dX / np.vstack([dN, dN]).T,
        "f95": np.vstack([corr_cddf["lo95"], corr_cddf["hi95"]]).T / dX / np.vstack([dN, dN]).T,
        # windowed RAW reference = the SAME deposit MEAN-count basis the correction
        # operates on (NOT the MAP, NOT the unwindowed O1 wrapper), so the
        # corrected-vs-raw ratio is self-consistent.
        "f_raw": F_cddf / dX / dN,
        "f68_raw": np.vstack([cddf_ci["lo68"], cddf_ci["hi68"]]).T / dX / np.vstack([dN, dN]).T,
        "f95_raw": np.vstack([cddf_ci["lo95"], cddf_ci["hi95"]]).T / dX / np.vstack([dN, dN]).T,
        "valid_mask": corr_cddf["valid_mask"],
        "neg_clip_mask": corr_cddf["neg_clip_mask"],
        "n_corr": corr_cddf["n_corr"],
    }

    # ===== dN/dX correction (z bins; logN collapsed to [lnhi_min, lnhi_max]) ====
    dndx_counts = cat.line_density_counts(
        z_min=z_min, z_max=z_max, lnhi_min=lnhi_min, lnhi_max=lnhi_max
    )
    z_cent = dndx_counts["z"]
    # reconstruct the z bin EDGES used by line_density (only dX>0 bins survive, but
    # the synthetic/real windows keep them all here; rebuild from bins_per_z).
    z_edges_dndx = _dndx_z_edges(cat, z_min, z_max)
    lnhi_edges_single = np.array([lnhi_min, lnhi_max])

    tmap_dndx = build_truth_map(
        truth_file, catalog_file=catalog_file, processed_file=processed_file,
        window=window, lnhi_edges=lnhi_edges_single, z_edges=z_edges_dndx,
        active_target_ids=active_target_ids,
    )
    dep_dndx = DiagonalSoftDeposit(
        cat, tmap_dndx, lnhi_edges=lnhi_edges_single, z_edges=z_edges_dndx,
        window=window,
    )
    part_dndx = dep_dndx.deposit(z_min=z_min, z_max=z_max)
    # collapse the singleton logN axis → 1-D over z
    fm_d = part_dndx["F_matched"][0, :]
    fu_d = part_dndx["F_unmatched"][0, :]
    nt_d = part_dndx["n_truth"][0, :]
    F_dndx = fm_d + fu_d  # deposit mean-count total (C1/C2)
    dX_d = dndx_counts["dX"]
    dndx_ci = _recenter_count_ci(dndx_counts, F_dndx)
    corr_dndx, C_est_d, bfp_est_d = _correct_1d(
        core,
        F_dndx,
        dndx_ci["lo68"], dndx_ci["hi68"], dndx_ci["lo95"], dndx_ci["hi95"],
        fm_d, fu_d, nt_d, dX_d,
    )
    o3_dndx = {
        "z": z_cent,
        "dndx": corr_dndx["n_corr"] / dX_d,
        "dndx68": np.vstack([corr_dndx["lo68"], corr_dndx["hi68"]]).T / np.vstack([dX_d, dX_d]).T,
        "dndx95": np.vstack([corr_dndx["lo95"], corr_dndx["hi95"]]).T / np.vstack([dX_d, dX_d]).T,
        # raw = the deposit MEAN-count basis (C1/C2), NOT the MAP.
        "dndx_raw": F_dndx / dX_d,
        "dndx68_raw": np.vstack([dndx_ci["lo68"], dndx_ci["hi68"]]).T / np.vstack([dX_d, dX_d]).T,
        "dndx95_raw": np.vstack([dndx_ci["lo95"], dndx_ci["hi95"]]).T / np.vstack([dX_d, dX_d]).T,
        "valid_mask": corr_dndx["valid_mask"],
        "neg_clip_mask": corr_dndx["neg_clip_mask"],
        "n_corr": corr_dndx["n_corr"],
    }

    # ===== Ω(z) from the corrected count-space CDDF ============================
    # C6: prefer the core's JOINT per-draw n_corr (omega_from_draws) — it preserves
    # inter-bin correlation and guarantees the Ω point lies inside its interval.
    # Fall back to the summed per-bin CI proxy only if the core lacks the helper.
    o3_omega = _omega_from_draws_or_proxy(
        core, corr_cddf, o3_cddf, cddf_counts, hubble=hubble
    )

    # ===== C8 join-coverage + C9 boundary/ceiling provenance ===================
    coverage = _join_coverage(truth_file, processed_file, tmap_cddf)
    boundary_flags = _boundary_ceiling_flags(
        lnhi_edges, lnhi_max, truth_file, sample_file, active_target_ids,
        processed_file, catalog_file, window,
    )

    provenance = {
        "processed_file": processed_file,
        "sample_file": sample_file,
        "catalog_file": catalog_file,
        "truth_file": truth_file,
        "filter_low_likelihood": int(filter_low_likelihood),
        "z_min": z_min, "z_max": z_max,
        "lnhi_min": lnhi_min, "lnhi_max": lnhi_max, "lnhi_nbins": lnhi_nbins,
        "hubble": hubble,
        "split_seed": split_seed, "build_frac": build_frac,
        "window": window, "window_applied": True,
        "lnhi_edges": lnhi_edges, "z_edges": z_edges_cddf,
        "z_edges_dndx": z_edges_dndx,
        "dlacat_kwargs": dict(dlacat_kwargs),
        "correction": _O3_LABEL, "limitation": _O3_LIMITATION,
        "n_active_sightlines": len(active_target_ids),
        # C8: join-coverage so a partial-coverage gap is VISIBLE, not mistaken for
        # incompleteness (cf. the 161-healpix London dlacat gap).
        "coverage": coverage,
        # C9: 20.3 Eddington-boundary per-bin flag + count of truth absorbers above
        # the lnhi_max grid ceiling (the diagonal correction can't capture migration
        # across the boundary nor recover mass beyond the ceiling).
        "boundary_flags": boundary_flags,
    }

    completeness = {
        "logN": cddf_counts["logN"],
        "C": np.asarray(C_est["C"]),
        "C_lo68": np.asarray(C_est["C_lo68"]),
        "C_hi68": np.asarray(C_est["C_hi68"]),
        "b_FP": np.asarray(bfp_est["b_FP"]),
        "n_truth": n_truth_cddf,
        "F_matched": f_matched,
        "F_unmatched": f_unmatched,
        "C_est": C_est,
        "bfp_est": bfp_est,
    }

    return {
        "o1": o1,
        "completeness": completeness,
        "o3_cddf": o3_cddf,
        "o3_dndx": o3_dndx,
        "o3_omega": o3_omega,
        "closure": {},
        "provenance": provenance,
    }


def _build_target_set(processed_file, *, split_seed, build_frac):
    """Return the set of BUILD TARGETIDs present in the processed file."""
    import h5py

    with h5py.File(processed_file, "r") as f:
        tids = np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
    build_mask, _ = split_masks(tids, seed=split_seed, frac_build=build_frac)
    return set(int(t) for t in tids[build_mask])


def _recenter_count_ci(counts_dict, F_center):
    """Re-center the estimator's count-space CI on the deposit-mean ``F_center``.

    The science basis (C1/C2) is the deposit MEAN count, but the estimator's
    Poisson-binomial CI brackets the MAP.  The core uses the CI only for its WIDTH
    (``sigma = (hi68 - lo68)/2``), centering its F-proxy at the passed point.  We
    therefore preserve the MAP CI HALF-WIDTHS but re-anchor them on ``F_center``
    (the deposit mean) so the F-uncertainty carries through WITHOUT re-introducing
    the MAP as the central value.  Lower edges are floored at 0 (counts >= 0).
    """
    F_center = np.asarray(F_center, dtype=float)
    map_counts = np.asarray(counts_dict["counts"], dtype=float)
    if F_center.shape != map_counts.shape:
        # The deposit F (all bins) and the estimator counts must be bin-aligned.
        # For dN/dX, line_density_counts drops dX==0 bins; if that ever fires while
        # the deposit keeps all z bins, the two are misaligned — fail loudly instead
        # of silently mis-broadcasting the CI half-widths onto the wrong bins.
        raise ValueError(
            f"_recenter_count_ci: F_center {F_center.shape} and estimator counts "
            f"{map_counts.shape} are not bin-aligned (a dX==0 z bin was dropped by "
            "line_density_counts but kept by the deposit). Restrict the deposit z "
            "grid to the dX>0 bins before correcting."
        )
    c68 = np.asarray(counts_dict["counts68"], dtype=float)
    c95 = np.asarray(counts_dict["counts95"], dtype=float)
    h68_lo = map_counts - c68[:, 0]
    h68_hi = c68[:, 1] - map_counts
    h95_lo = map_counts - c95[:, 0]
    h95_hi = c95[:, 1] - map_counts
    return {
        "lo68": np.maximum(F_center - h68_lo, 0.0),
        "hi68": F_center + h68_hi,
        "lo95": np.maximum(F_center - h95_lo, 0.0),
        "hi95": F_center + h95_hi,
    }


def _dndx_z_edges(cat, z_min, z_max):
    """The z bin edges ``line_density`` / ``line_density_counts`` use (bins_per_z)."""
    nbins = int(np.max([int((z_max - z_min) * cat.bins_per_z), 1]))
    return np.linspace(z_min, z_max, nbins + 1)


def _join_coverage(truth_file, processed_file, truth_map):
    """TARGETID join-coverage provenance between the truth and processed catalogs (C8).

    Reports how the truth TARGETID set and the processed-file TARGETID set overlap:

    * ``n_truth_only``     — TARGETIDs in truth but NOT in the processed run (truth
      absorbers the GP never even searched -> a COVERAGE gap, not incompleteness);
    * ``n_processed_only`` — processed sightlines with no truth absorber listed;
    * ``n_both``           — TARGETIDs present in both;
    * ``n_absorbers_in`` / ``n_absorbers_kept`` — from the windowed truth map (how
      many truth absorbers entered vs survived the window/range/active cuts);
    * ``n_healpix`` / ``healpix_coverage`` — per-healpix sightline coverage IF the
      processed file carries a ``healpix`` dataset (None otherwise).

    Surfaced so a partial-coverage gap (cf. the 161-healpix London dlacat gap) is
    VISIBLE in provenance and not silently absorbed into a low completeness.
    """
    import h5py
    from astropy.table import Table

    table = Table.read(truth_file)
    tcol = None
    for name in ("TARGETID",):
        if name in table.colnames:
            tcol = name
            break
    truth_tids = (
        set(int(t) for t in np.asarray(table[tcol]).astype(np.int64))
        if tcol is not None else set()
    )

    healpix_coverage = None
    n_healpix = None
    with h5py.File(processed_file, "r") as f:
        proc_tids = set(int(t) for t in np.asarray(f["target_ids"][:]).astype(np.int64).ravel())
        if "healpix" in f:
            hp = np.asarray(f["healpix"][:]).ravel()
            uniq = np.unique(hp)
            n_healpix = int(uniq.size)
            healpix_coverage = {int(h): int(np.sum(hp == h)) for h in uniq}

    both = truth_tids & proc_tids
    return {
        "n_truth_only": int(len(truth_tids - proc_tids)),
        "n_processed_only": int(len(proc_tids - truth_tids)),
        "n_both": int(len(both)),
        "n_truth_targets": int(len(truth_tids)),
        "n_processed_targets": int(len(proc_tids)),
        "n_absorbers_in": int(getattr(truth_map, "n_absorbers_in", 0)),
        "n_absorbers_kept": int(getattr(truth_map, "n_absorbers_kept", 0)),
        "n_healpix": n_healpix,
        "healpix_coverage": healpix_coverage,
    }


def _boundary_ceiling_flags(lnhi_edges, lnhi_max, truth_file, sample_file,
                            active_target_ids, processed_file, catalog_file, window):
    """20.3-boundary per-bin flag + above-ceiling truth count provenance (C9).

    * ``eddington_boundary_bins`` — boolean per logN bin flagging the lowest 1-2 DLA
      bins adjacent to the logN = 20.3 (sub-DLA↔DLA) boundary, where the diagonal
      correction CANNOT capture cross-boundary migration / Eddington up-scatter (an
      off-diagonal O4 channel).  These bins' O3 values must be read with that caveat.
    * ``n_truth_above_ceiling`` — number of (active-sightline) truth absorbers with
      logN above ``lnhi_max`` (the 22.5 grid ceiling): mass the diagonal correction
      cannot recover because the sample grid does not extend there.
    * ``lnhi_max`` / ``sample_grid_ceiling`` — recorded; the ``lnhi_max ==
      sample-grid ceiling`` assertion (C9) is enforced below.
    """
    import h5py
    from astropy.table import Table

    lnhi_edges = np.asarray(lnhi_edges, float)
    nbin = lnhi_edges.size - 1

    # The lowest 1-2 DLA bins at the 20.3 Eddington boundary.  Flag the lowest two
    # (or one if only one bin), where DLA<->sub-DLA migration concentrates.
    eddington = np.zeros(nbin, dtype=bool)
    n_flag = min(2, nbin)
    eddington[:n_flag] = True

    # Sample-grid ceiling: the max log_nhi_sample in the QMC grid.  Assert it equals
    # lnhi_max so CDDF bins above the cap are not silently starved (C9).
    with h5py.File(sample_file, "r") as f:
        log_nhi_samples = np.asarray(f["log_nhi_samples"][:]).ravel()
    sample_grid_ceiling = float(np.max(log_nhi_samples))
    assert sample_grid_ceiling >= float(lnhi_max) - 1e-6, (
        f"lnhi_max={lnhi_max} exceeds the QMC sample-grid ceiling "
        f"{sample_grid_ceiling}: bins above the grid would be starved. Lower "
        "lnhi_max to the sample-grid ceiling or extend the grid."
    )

    # Count truth absorbers above the ceiling on ACTIVE sightlines.
    table = Table.read(truth_file)
    nhi_col = None
    for name in ("NHI", "N_HI"):
        if name in table.colnames:
            nhi_col = name
            break
    tid_col = "TARGETID" if "TARGETID" in table.colnames else None
    n_above = 0
    if nhi_col is not None and tid_col is not None:
        nhis = np.asarray(table[nhi_col]).astype(float)
        tids = np.asarray(table[tid_col]).astype(np.int64)
        active = set(int(t) for t in active_target_ids)
        for nh, t in zip(nhis, tids):
            if int(t) in active and nh >= float(lnhi_max):
                n_above += 1

    return {
        "eddington_boundary_bins": eddington,
        "eddington_boundary_note": (
            "Lowest 1-2 DLA bins at the logN=20.3 boundary: the DIAGONAL correction "
            "cannot capture sub-DLA<->DLA migration / Eddington up-scatter (O4)."
        ),
        "n_truth_above_ceiling": int(n_above),
        "lnhi_max": float(lnhi_max),
        "sample_grid_ceiling": sample_grid_ceiling,
    }


def _omega_from_draws_or_proxy(core, corr_cddf, o3_cddf, cddf_counts, *, hubble):
    """Ω(z) from the core's JOINT per-draw ``n_corr`` (C6), else the CI-sum proxy.

    Preferred (C6 / B2): the corrected CDDF carries ``n_corr_draws`` (an
    ``(n_mc, nbin)`` joint posterior over the logN bins).  Forming Ω per-draw
    (``Σ_N N·n_corr / ΔX``) and percentiling preserves the inter-bin correlation
    and GUARANTEES the Ω point estimate lies inside its interval — unlike summing
    pre-reduced per-bin f-CI edges, which over-states the width and can place the
    point outside.  We call the core's documented ``omega_from_draws`` helper.

    Fallback: if the core does not expose ``omega_from_draws`` / ``n_corr_draws``
    (older core), use the conservative summed-CI proxy
    (:func:`_omega_from_corrected_cddf`) so the driver still produces an Ω block.
    """
    draws = corr_cddf.get("n_corr_draws") if isinstance(corr_cddf, dict) else None
    helper = getattr(core, "omega_from_draws", None)
    if draws is not None and callable(helper):
        dX = float(cddf_counts["dX"])
        out = helper(np.asarray(draws, float), np.asarray(o3_cddf["logN"], float),
                     dX, hubble)
        z_cent = 0.5 * (cddf_counts.get("z_min", 0) + cddf_counts.get("z_max", 0))
        # The core returns the FLAT scalar schema {omega, lo68, hi68, lo95, hi95}
        # (B2); also accept the nested {omega, omega68, omega95} for forward-compat.
        omega = float(np.asarray(out["omega"]).ravel()[0])
        if "lo68" in out:
            o68 = np.array([[float(out["lo68"]), float(out["hi68"])]])
            o95 = np.array([[float(out["lo95"]), float(out["hi95"])]])
        else:
            o68 = np.asarray(out["omega68"], float).reshape(1, 2)
            o95 = np.asarray(out["omega95"], float).reshape(1, 2)
        return {
            "z": np.array([z_cent]) if np.ndim(z_cent) == 0 else z_cent,
            "omega": np.array([omega]),
            "omega68": o68,
            "omega95": o95,
            "method": "joint_draws",
        }
    proxy = _omega_from_corrected_cddf(o3_cddf, cddf_counts, hubble=hubble)
    proxy["method"] = "summed_ci_proxy"
    return proxy


def _omega_from_corrected_cddf(o3_cddf, cddf_counts, *, hubble):
    """Ω(z) from the corrected COUNT-space CDDF (single z window).

    Diagonal limit: Ω = (m_p H0 / c ρ_c) * Σ N_HI * n_corr / ΔX, summing the
    N-weighted corrected counts over the logN bins of the single [z_min, z_max]
    window.  CI is propagated from the corrected count CIs (bin-summed in
    quadrature on the N-weighted deposits — a conservative diagonal proxy).
    """
    from ..calc_cddf import rho_crit

    protonmass = 1.67262178e-24
    h100 = 3.2407789e-18 * hubble
    light = 2.99e10
    conversion = protonmass / light * h100 / rho_crit(hubble)

    nhi_cent = 10 ** np.asarray(o3_cddf["logN"], float)
    n_corr = np.asarray(o3_cddf["n_corr"], float)
    valid = np.asarray(o3_cddf["valid_mask"], bool)
    dX = float(cddf_counts["dX"])

    w = np.where(valid, nhi_cent * np.nan_to_num(n_corr, nan=0.0), 0.0)
    omega = conversion * np.sum(w) / dX
    # CI: N-weighted corrected count interval, summed (diagonal proxy). We derive
    # the Ω interval from the corrected f-CI by mapping back to counts via dX*dN
    # (keeps this additive + simple, no separate count-CI carry).
    dN = np.asarray(cddf_counts["dN"], float)
    f68 = np.asarray(o3_cddf["f68"], float)  # (nbins, 2) f-values
    f95 = np.asarray(o3_cddf["f95"], float)
    counts68 = f68 * dX * np.vstack([dN, dN]).T  # back to counts
    counts95 = f95 * dX * np.vstack([dN, dN]).T
    w68 = np.where(valid[:, None], nhi_cent[:, None] * np.nan_to_num(counts68, nan=0.0), 0.0)
    w95 = np.where(valid[:, None], nhi_cent[:, None] * np.nan_to_num(counts95, nan=0.0), 0.0)
    omega68 = conversion * np.sum(w68, axis=0) / dX
    omega95 = conversion * np.sum(w95, axis=0) / dX
    z_cent = 0.5 * (cddf_counts.get("z_min", 0) + cddf_counts.get("z_max", 0))
    return {
        "z": np.array([z_cent]) if np.ndim(z_cent) == 0 else z_cent,
        "omega": np.array([omega]),
        "omega68": omega68.reshape(1, 2),
        "omega95": omega95.reshape(1, 2),
    }


def heldout_closure(
    processed_file: str,
    sample_file: str,
    catalog_file: str,
    truth_file: str,
    *,
    z_min: float,
    z_max: float,
    lnhi_min: float = 20.3,
    lnhi_max: float = 22.5,
    lnhi_nbins: int = 30,
    hubble: float = 0.7,
    filter_low_likelihood: int = 0,
    window: Optional[WindowSpec] = None,
    split_seed: int = 20260609,
    build_frac: float = 0.7,
    pass_frac: float = 0.68,
    **dlacat_kwargs,
) -> dict:
    """HELDOUT closure harness (contract §3.5 + fix C5).

    Estimate C & b_FP on the BUILD-ACTIVE split, apply the diagonal correction to
    the HELDOUT-ACTIVE recovered count-space F, and compare the corrected CDDF count
    to the HELDOUT **truth** count per logN bin.  ``assert_no_leakage`` between BUILD
    and HELDOUT TARGETIDs runs FIRST (non-circularity is enforced, not assumed).

    C5 fixes
    --------
    (a) The BUILD ``b_FP`` COUNT is rebased to the HELDOUT basis by the
        active-sightline ratio ``N_held_active / N_build_active`` before subtraction
        (a FP COUNT scales with the number of contributing sightlines).
    (b) The ``F_ci`` handed to the correction is the REAL count-space 68/95 CI
        (the Poisson-binomial interval restricted to the HELDOUT-ACTIVE sightlines),
        NOT the degenerate ``lo == hi == F`` point.
    (c) The pass gate combines marginal COVERAGE (~95% of valid bins within 2σ, or
        ~68% within 1σ) AND a COHERENT-BIAS test (mean standardized residual ≈ 0),
        to catch a uniform multiplicative bias that marginal coverage alone misses.

    Returns
    -------
    dict
        ``logN, corrected, truth, residual, standardized_residual, n_valid_bins``;
        ``bfp_rebase_ratio``           : N_held_active / N_build_active;
        ``mean_standardized_residual`` : coherent-bias statistic;
        ``coverage_ok``                : marginal-coverage gate;
        ``coherent_bias_ok``           : |mean standardized residual| small;
        ``passed``                     : ``coverage_ok and coherent_bias_ok``.
    """
    assert_filter_off_from_file(
        processed_file, supplied=filter_low_likelihood, ctx="heldout_closure"
    )
    if window is None:
        window = WindowSpec()
    core = _require_core()
    dlacat_kwargs.setdefault("high_nhi_cut_value", lnhi_max)

    cat = DLACatalogue(
        processed_file=processed_file, sample_file=sample_file,
        catalog_file=catalog_file, window=window, **dlacat_kwargs,
    )
    lnhi_edges = np.linspace(lnhi_min, lnhi_max, lnhi_nbins + 1)
    z_edges = np.array([z_min, z_max])

    # Active set FIRST, then split it (so BUILD/HELDOUT are both ACTIVE — same
    # population basis the C / b_FP / F are measured on).
    active_ids = set(int(t) for t in cat.target_ids[cat.filter_dla_spectra()[0]])
    active_arr = np.array(sorted(active_ids), dtype=np.int64)
    build_mask, held_mask = split_masks(
        active_arr, seed=split_seed, frac_build=build_frac
    )
    build_ids = set(int(t) for t in active_arr[build_mask])
    held_ids = set(int(t) for t in active_arr[held_mask])
    # Enforce non-circularity FIRST.
    assert_no_leakage(
        np.array(sorted(build_ids), dtype=np.int64),
        np.array(sorted(held_ids), dtype=np.int64),
        ctx="heldout_closure",
    )
    n_build_active = max(len(build_ids), 1)
    n_held_active = len(held_ids)
    bfp_rebase_ratio = n_held_active / n_build_active

    # BUILD-active: C & b_FP (b_FP estimated on the BUILD path length).
    build_tmap = build_truth_map(
        truth_file, catalog_file=catalog_file, processed_file=processed_file,
        window=window, lnhi_edges=lnhi_edges, z_edges=z_edges,
        active_target_ids=build_ids,
    )
    dep = DiagonalSoftDeposit(
        cat, build_tmap, lnhi_edges=lnhi_edges, z_edges=z_edges, window=window
    )
    build_part = dep.deposit(z_min=z_min, z_max=z_max, target_ids=build_ids)
    C_est = core.estimate_diagonal_completeness(
        build_part["F_matched"][:, 0], build_part["n_truth"][:, 0]
    )
    # exposure = BUILD-active path length (the basis the BUILD b_FP count lives on).
    build_dX = _role_restricted_dX(cat, build_ids, z_min, z_max)
    bfp_est_build = core.estimate_false_positive_deposit(
        build_part["F_unmatched"][:, 0], float(max(build_dX, np.finfo(float).tiny))
    )
    # (a) Rebase the BUILD b_FP COUNT to the HELDOUT basis by the active-sightline
    # ratio (a FP count scales with the number of contributing sightlines).
    bfp_est = _rebase_bfp(bfp_est_build, bfp_rebase_ratio)

    # HELDOUT-active: recovered F (deposit mean count) + REAL count-space CI.
    held_tmap = build_truth_map(
        truth_file, catalog_file=catalog_file, processed_file=processed_file,
        window=window, lnhi_edges=lnhi_edges, z_edges=z_edges,
        active_target_ids=held_ids,
    )
    dep_h = DiagonalSoftDeposit(
        cat, held_tmap, lnhi_edges=lnhi_edges, z_edges=z_edges, window=window
    )
    held_part = dep_h.deposit(z_min=z_min, z_max=z_max, target_ids=held_ids)
    F_held = (held_part["F_matched"] + held_part["F_unmatched"])[:, 0]
    truth_held = held_part["n_truth"][:, 0].astype(float)

    # (b) REAL count-space CI on the HELDOUT-ACTIVE subset (the Poisson-binomial
    # interval restricted to HELDOUT sightlines), re-centered on the deposit mean.
    held_counts = _role_restricted_count_counts(
        cat, held_ids, z_min=z_min, z_max=z_max, lnhi_nbins=lnhi_nbins,
        lnhi_min=lnhi_min, lnhi_max=lnhi_max,
    )
    F_ci = _recenter_count_ci(held_counts, F_held)
    corr = core.apply_diagonal_correction(F_held, F_ci, C_est, bfp_est)
    corrected = np.asarray(corr["n_corr"], float)
    valid = np.asarray(corr["valid_mask"], bool)

    residual = corrected - truth_held
    # CI half-width from the corrected interval; fall back to sqrt(N) Poisson.
    hi = np.asarray(corr["hi68"], float)
    lo = np.asarray(corr["lo68"], float)
    half = 0.5 * (hi - lo)
    poisson = np.sqrt(np.clip(corrected, 1.0, None))
    half = np.where(np.isfinite(half) & (half > 0), half, poisson)
    with np.errstate(invalid="ignore", divide="ignore"):
        std_resid = np.where(half > 0, residual / half, np.nan)

    finite = valid & np.isfinite(std_resid) & (truth_held > 0)
    n_valid = int(np.sum(finite))

    # (c) Gate: marginal coverage AND coherent-bias.
    if n_valid == 0:
        coverage_ok = False
        mean_std = np.nan
        coherent_bias_ok = False
    else:
        sr = std_resid[finite]
        within2 = np.mean(np.abs(sr) <= 2.0)
        within1 = np.mean(np.abs(sr) <= 1.0)
        coverage_ok = bool(within2 >= 0.95 or within1 >= pass_frac)
        mean_std = float(np.mean(sr))
        # coherent (uniform) bias: the mean standardized residual must be ~0.
        # tolerance ~ 1σ/sqrt(n) for the mean of n unit-variance residuals, with a
        # small floor; a uniform multiplicative bias shifts this away from 0.
        bias_tol = max(2.0 / np.sqrt(n_valid), 0.5)
        coherent_bias_ok = bool(abs(mean_std) <= bias_tol)
    passed = bool(coverage_ok and coherent_bias_ok)

    return {
        "logN": held_counts["logN"],
        "corrected": corrected,
        "truth": truth_held,
        "residual": residual,
        "standardized_residual": std_resid,
        "n_valid_bins": n_valid,
        "bfp_rebase_ratio": bfp_rebase_ratio,
        "mean_standardized_residual": mean_std,
        "coverage_ok": coverage_ok,
        "coherent_bias_ok": coherent_bias_ok,
        "passed": passed,
    }


def _rebase_bfp(bfp_est, ratio):
    """Scale a b_FP COUNT estimator dict by ``ratio`` (BUILD→HELDOUT basis, C5a).

    A false-positive DEPOSIT is a COUNT proportional to the number of contributing
    sightlines (and the path length), so rebasing from the BUILD basis to the
    HELDOUT basis multiplies every b_FP count (and its CI edges) by
    ``N_held_active / N_build_active``.

    KNOWN MINOR APPROXIMATION (re-verifier finding 4, closure-only): the core's
    b_FP point is the Gamma posterior MODE ``max(f_unmatched + a − 1, 0)``.  Linear
    scaling here multiplies that finished mode (and the Gamma ``b_FP_shape``) by
    ``ratio``, so the rebased point is no longer EXACTLY the mode of a
    ``Gamma(shape·ratio, 1)`` draw distribution — the strictly consistent rebasing
    would scale the underlying unmatched COUNT (``f_unmatched·ratio``) and
    re-derive the posterior.  Impact is negligible and bounded: this is used ONLY
    in :func:`heldout_closure` (a validation diagnostic, never the science
    estimate, which does no rebasing), the rebased point stays inside its own
    rebased CI in every tested regime, and the clean-bin ``f_unmatched=0 ⇒ b_FP=0``
    property survives (0·ratio=0).  Tracked as a follow-up for the closure path; it
    does not affect the corrected CDDF/dN/dX/Ω the science path reports.
    """
    out = {}
    for k, v in bfp_est.items():
        out[k] = np.asarray(v, dtype=float) * float(ratio)
    return out


def _role_restricted_dX(cat, target_ids, z_min, z_max):
    """``path_length`` over only the given (active) TARGETIDs (via ``condition``).

    Temporarily restricts the catalogue's ``condition`` mask to the requested
    sightlines so ``path_length`` integrates dX over exactly that subset, then
    restores the original mask.  ADDITIVE: leaves the catalogue unchanged on exit.
    """
    keep = np.isin(np.asarray(cat.target_ids).astype(np.int64),
                   np.array(sorted(int(t) for t in target_ids), dtype=np.int64))
    saved = cat.condition
    try:
        cat.condition = saved & keep
        return float(cat.path_length(z_min, z_max))
    finally:
        cat.condition = saved


def _role_restricted_count_counts(cat, target_ids, *, z_min, z_max, lnhi_nbins,
                                  lnhi_min, lnhi_max):
    """``column_density_function_counts`` over only the given (active) TARGETIDs.

    Restricts the catalogue's ``condition`` mask to the subset (so both the
    Poisson-binomial count CI and dX are computed on exactly those sightlines),
    calls the additive count accessor, and restores the mask.  This is the REAL
    count-space CI restricted to the role, per C5(b).
    """
    keep = np.isin(np.asarray(cat.target_ids).astype(np.int64),
                   np.array(sorted(int(t) for t in target_ids), dtype=np.int64))
    saved = cat.condition
    try:
        cat.condition = saved & keep
        return cat.column_density_function_counts(
            z_min=z_min, z_max=z_max, lnhi_nbins=lnhi_nbins,
            lnhi_min=lnhi_min, lnhi_max=lnhi_max,
        )
    finally:
        cat.condition = saved


def save_o3_products(products: dict, out_dir: str) -> dict:
    """Write O3 CDDF + dN/dX + completeness text tables with HONEST labels (§3.6).

    Labels every table "O3 DIAGONAL SOFT-COMPLETENESS CORRECTED" and states the
    cross-bin-migration limitation.  Reuses ``cddf_io`` writers (raw = O1, calibrated
    = O3-corrected, ratio = corrected/raw) without forking the IO.
    """
    os.makedirs(out_dir, exist_ok=True)
    prov = products["provenance"]
    o1 = products["o1"]
    note = f"{_O3_LABEL}. {_O3_LIMITATION}"
    # ``processed_file`` (single-file driver) OR ``processed_files``/``n_files``
    # (no-combine streaming) — summarize the latter so the streaming provenance
    # schema does not KeyError here.
    _proc = prov.get("processed_file")
    if _proc is None:
        _n_skip = len(prov.get("unreadable_files", []))
        _proc = (
            f"{prov.get('n_files', '?')} per-healpix files (streaming"
            + (f", {_n_skip} skipped" if _n_skip else "")
            + ")"
        )
    meta = {
        "product": _O3_LABEL,
        "limitation": _O3_LIMITATION,
        "processed_file": _proc,
        "truth_file": prov["truth_file"],
        "split_seed": prov["split_seed"],
        "build_frac": prov["build_frac"],
        "z_min": prov["z_min"], "z_max": prov["z_max"],
    }

    # --- CDDF f(N): windowed raw vs corrected (self-consistent ratio) --------
    c3 = products["o3_cddf"]
    n = np.asarray(c3["f"]).shape[0]
    raw_f = np.asarray(c3["f_raw"], float)  # windowed raw from the SAME catalogue
    corr_f = np.asarray(c3["f"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(raw_f > 0, corr_f / raw_f, np.nan)
    ones2 = np.ones((n, 2), dtype=float)
    cddf_dict = {
        "title": (
            f"{_O3_LABEL} (f_cal = diagonal soft-completeness corrected; "
            f"f_raw = windowed uncorrected) {prov['z_min']} <= z < {prov['z_max']}. "
            f"{_O3_LIMITATION}"
        ),
        "logN": c3["logN"],
        "f_corr": corr_f, "f68_corr": c3["f68"], "f95_corr": c3["f95"],
        "f_raw": raw_f, "f68_raw": c3["f68_raw"], "f95_raw": c3["f95_raw"],
        "r": r, "r68": ones2, "r95": ones2,
    }
    cddf_path = os.path.join(out_dir, "o3_cddf.txt")
    cddf_io.save_cddf_txt_table(cddf_path, cddf_dict, include_truth=False)

    # --- dN/dX: windowed raw vs corrected ------------------------------------
    d3 = products["o3_dndx"]
    dndx_path = os.path.join(out_dir, "o3_dndx.txt")
    cddf_io.save_dndx_combined(
        dndx_path, d3["z"],
        d3["dndx_raw"], d3["dndx68_raw"], d3["dndx95_raw"],
        d3["dndx"], d3["dndx68"], d3["dndx95"],
        meta=meta, calibration_note=note,
    )

    # --- completeness C(N) + b_FP table --------------------------------------
    comp = products["completeness"]
    comp_path = os.path.join(out_dir, "o3_completeness.txt")
    comp_cols = np.column_stack([
        np.asarray(comp["logN"], float),
        np.asarray(comp["C"], float),
        np.asarray(comp["C_lo68"], float),
        np.asarray(comp["C_hi68"], float),
        np.asarray(comp["b_FP"], float),
        np.asarray(comp["n_truth"], float),
        np.asarray(comp["F_matched"], float),
        np.asarray(comp["F_unmatched"], float),
    ])
    comp_header = "\n".join([
        _O3_LABEL,
        _O3_LIMITATION,
        "columns: logN  C  C_lo68  C_hi68  b_FP  n_truth  F_matched  F_unmatched",
        "C is the per-bin diagonal completeness (BUILD split); b_FP the soft "
        "false-positive deposit in count units.",
    ])
    np.savetxt(comp_path, comp_cols, header=comp_header)

    return {"cddf": cddf_path, "dndx": dndx_path, "completeness": comp_path}


# How low / unreliable a per-bin completeness must be before the fine-logN f(N)
# point is annotated "DIAGONAL — needs off-diagonal R (M3)" (N-scatter dominates).
_C_UNRELIABLE = 0.2


def _band(ax, x, lo, hi, *, color, alpha=0.25, label=None):
    """A 68% (or any) shaded band ``[lo, hi]`` along ``x`` (NaN-safe)."""
    x = np.asarray(x, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    finite = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    if np.any(finite):
        ax.fill_between(x[finite], lo[finite], hi[finite], color=color, alpha=alpha,
                        linewidth=0, label=label)


def plot_o3_diagnostics(products: dict, *, save_path: Optional[str] = None,
                        title: Optional[str] = None):
    """Reviewable multi-panel O3 DIAGONAL diagnostic figure (§3.6).

    Six panels on a streaming/combined O3-products dict (the shape
    :func:`compute_o3_products` / ``compute_o3_products_streaming`` return):

    1. **f(N)**         raw vs corrected (log-log) + 68% band; fine-logN bins where
       completeness is low/unreliable are annotated "DIAGONAL — needs off-diagonal
       R (M3)" (the N-scatter limitation lives precisely there).
    2. **dN/dX(z)**     raw vs corrected + 68% band — the ROBUST wide-N-bin product.
    3. **Ω_DLA(z)**     value + CI, with the diagonal caveat.
    4. **C(N)**         per-bin completeness; NaN/masked bins greyed.
    5. **b_FP(N)**      per-bin soft false-positive deposit; NaN bins greyed.
    6. **coverage**     join-coverage / per-healpix counts so a partial-healpix GAP
       is visible (cf. the 161-healpix London dlacat gap).

    Honest everywhere: the suptitle carries the "O3 DIAGONAL SOFT-COMPLETENESS
    CORRECTED" banner + the off-diagonal/N-scatter limitation; no panel claims an
    alpha(z) / London-mock calibration.  Agg backend when ``save_path`` is given (no
    display).  Returns the matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c3 = products["o3_cddf"]
    d3 = products["o3_dndx"]
    om = products.get("o3_omega", {})
    comp = products["completeness"]
    prov = products.get("provenance", {})
    grey = "0.7"

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    banner = f"{_O3_LABEL}\n{_O3_LIMITATION}"
    if title:
        banner = f"{title}\n{banner}"
    fig.suptitle(banner, fontsize=8)

    # ---- 1. f(N): raw vs corrected (log-log) + 68% band -----------------------
    ax = axes[0, 0]
    logN = np.asarray(c3["logN"], float)
    f_raw = np.asarray(c3["f_raw"], float)
    f_cor = np.asarray(c3["f"], float)
    f68 = np.asarray(c3["f68"], float)
    _band(ax, logN, f68[:, 0], f68[:, 1], color="C0", label="O3 68%")
    ax.plot(logN, f_raw, "o-", color=grey, label="raw (windowed)")
    ax.plot(logN, f_cor, "s-", color="C0", label="O3 corrected")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10} N_{HI}$"); ax.set_ylabel(r"$f(N_{HI})$")
    ax.set_title("CDDF f(N) [DIAGONAL]", fontsize=9)
    # Annotate fine-logN bins where C is low/unreliable: there the DIAGONAL f(N) is
    # corrupted by N-scatter and would need the off-diagonal response R (M3).
    C = np.asarray(comp.get("C", np.full(logN.shape, np.nan)), float)
    unreliable = ~np.isfinite(C) | (C < _C_UNRELIABLE)
    if np.any(unreliable & np.isfinite(f_cor)):
        ax.annotate(
            "DIAGONAL — needs off-diagonal R (M3)\n(low-C fine-logN bins unreliable)",
            xy=(0.02, 0.03), xycoords="axes fraction", fontsize=6, color="C3",
            va="bottom",
        )
    ax.legend(fontsize=6)

    # ---- 2. dN/dX(z): raw vs corrected + 68% band (robust wide-N product) ------
    ax = axes[0, 1]
    z = np.asarray(d3["z"], float)
    dndx_raw = np.asarray(d3["dndx_raw"], float)
    dndx_cor = np.asarray(d3["dndx"], float)
    d68 = np.asarray(d3["dndx68"], float)
    _band(ax, z, d68[:, 0], d68[:, 1], color="C0", label="O3 68%")
    ax.plot(z, dndx_raw, "o-", color=grey, label="raw (windowed)")
    ax.plot(z, dndx_cor, "s-", color="C0", label="O3 corrected")
    ax.set_xlabel("z"); ax.set_ylabel("dN/dX")
    ax.set_title("Line density dN/dX (robust wide-N bin)", fontsize=9)
    ax.legend(fontsize=6)

    # ---- 3. Omega_DLA(z): value + CI, diagonal caveat -------------------------
    ax = axes[0, 2]
    if om:
        z_o = np.asarray(om["omega"]).ravel().size  # noqa: F841 (presence check)
        zc = np.atleast_1d(np.asarray(om.get("z", [np.nan]), float))
        omega = np.atleast_1d(np.asarray(om["omega"], float))
        o68 = np.asarray(om.get("omega68", np.full((omega.size, 2), np.nan)), float
                         ).reshape(-1, 2)
        yerr = np.vstack([omega - o68[:, 0], o68[:, 1] - omega])
        ax.errorbar(zc, omega, yerr=np.abs(yerr), fmt="s", color="C0", capsize=3,
                    label=r"$\Omega_{DLA}$ (68%)")
        ax.set_xlabel("z"); ax.set_ylabel(r"$\Omega_{DLA}$")
        ax.legend(fontsize=6)
    ax.set_title(r"$\Omega_{DLA}(z)$ [DIAGONAL caveat]", fontsize=9)
    ax.annotate("N-weighted; DIAGONAL — high-N scatter not modelled (M3)",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=6, color="C3",
                va="bottom")

    # ---- 4. C(N): completeness; NaN/masked bins greyed ------------------------
    ax = axes[1, 0]
    cl = np.asarray(comp.get("C_lo68", np.full(C.shape, np.nan)), float)
    ch = np.asarray(comp.get("C_hi68", np.full(C.shape, np.nan)), float)
    finite = np.isfinite(C)
    _band(ax, logN, cl, ch, color="C2", label="68%")
    ax.plot(logN[finite], C[finite], "o-", color="C2", label="C (valid)")
    masked = ~finite
    if np.any(masked):
        # grey vertical spans mark the masked / NaN-completeness bins.
        for xb in logN[masked]:
            ax.axvspan(xb - 0.03, xb + 0.03, color=grey, alpha=0.5)
        ax.plot([], [], color=grey, lw=6, alpha=0.5, label="masked / NaN")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\log_{10} N_{HI}$"); ax.set_ylabel("completeness C")
    ax.set_title("Diagonal completeness C(N)", fontsize=9)
    ax.legend(fontsize=6)

    # ---- 5. b_FP(N): soft false-positive deposit; NaN bins greyed -------------
    ax = axes[1, 1]
    bfp = np.asarray(comp.get("b_FP", np.full(logN.shape, np.nan)), float)
    fb = np.isfinite(bfp)
    ax.plot(logN[fb], bfp[fb], "o-", color="C3", label=r"$b_{FP}$ (valid)")
    bmask = ~fb
    if np.any(bmask):
        for xb in logN[bmask]:
            ax.axvspan(xb - 0.03, xb + 0.03, color=grey, alpha=0.5)
        ax.plot([], [], color=grey, lw=6, alpha=0.5, label="masked / NaN")
    ax.set_xlabel(r"$\log_{10} N_{HI}$"); ax.set_ylabel(r"$b_{FP}$ (counts)")
    ax.set_title("Soft false-positive deposit b_FP(N)", fontsize=9)
    ax.legend(fontsize=6)

    # ---- 6. coverage / gap panel ----------------------------------------------
    ax = axes[1, 2]
    _draw_coverage_panel(ax, prov, grey=grey)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path is not None:
        fig.savefig(save_path, dpi=110)
    return fig


def _draw_coverage_panel(ax, prov, *, grey):
    """Per-healpix coverage bars + join-coverage counts so a GAP is visible.

    A zero-count healpix is drawn in red and the panel title is annotated "GAP" so a
    partial-coverage hole (cf. the 161-healpix London dlacat gap) cannot be mistaken
    for low completeness.  Absent ``provenance.coverage`` -> an honest "N/A" panel.
    """
    cov = (prov or {}).get("coverage")
    ax.set_title("Coverage / per-healpix gap", fontsize=9)
    if not cov:
        ax.annotate("coverage provenance N/A", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center", va="center", fontsize=8,
                    color=grey)
        ax.set_xticks([]); ax.set_yticks([])
        return

    hp_cov = cov.get("healpix_coverage")
    has_gap = False
    if hp_cov:
        items = sorted(hp_cov.items())
        counts = np.array([c for _, c in items], float)
        idx = np.arange(counts.size)
        is_gap = counts <= 0
        has_gap = bool(np.any(is_gap))
        colors = ["C3" if g else "C0" for g in is_gap]
        ax.bar(idx, counts, color=colors)
        ax.set_xlabel(f"healpix index (n={counts.size})")
        ax.set_ylabel("sightlines / healpix")
        if has_gap:
            n_gap = int(np.sum(is_gap))
            ax.annotate(f"GAP: {n_gap} zero-coverage healpix (red)",
                        xy=(0.02, 0.92), xycoords="axes fraction", fontsize=7,
                        color="C3", va="top")
    else:
        # no per-healpix breakdown: show the join-coverage counts as a bar summary.
        keys = ["n_both", "n_truth_only", "n_processed_only"]
        vals = [int(cov.get(k, 0)) for k in keys]
        ax.bar(range(len(keys)), vals, color=["C0", "C3", "0.5"])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["both", "truth-only", "proc-only"], fontsize=7)
        ax.set_ylabel("TARGETID count")
        if vals[1] > 0:
            ax.annotate("truth-only = coverage GAP (not incompleteness)",
                        xy=(0.02, 0.92), xycoords="axes fraction", fontsize=7,
                        color="C3", va="top")

    # always surface the headline join-coverage numbers as text.
    txt = (f"both={cov.get('n_both', '?')}  truth-only={cov.get('n_truth_only', '?')}"
           f"  proc-only={cov.get('n_processed_only', '?')}"
           f"  n_healpix={cov.get('n_healpix', '?')}")
    ax.annotate(txt, xy=(0.02, 0.02), xycoords="axes fraction", fontsize=6,
                color="0.3", va="bottom")


def plot_o3_products(products: dict, *, save_path: Optional[str] = None):
    """Back-compat shim: delegates to :func:`plot_o3_diagnostics` (§3.6).

    The original 4-panel entry point is preserved for existing callers; it now
    produces the richer multi-panel diagnostic figure.  Honest title + the migration
    caveat are carried by the delegate.
    """
    return plot_o3_diagnostics(products, save_path=save_path)


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
