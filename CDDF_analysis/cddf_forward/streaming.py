"""NO-COMBINE streaming O1/O3 CDDF driver (file-by-file, bounded memory).

WHY
---
A FILTER-off processed file carries ~100k QMC-sample log-likelihoods per spectrum
(~750 MB each); the 2LPT-0 run is ~1081 such files (~800 GB).  Building a
monolithic ``combined.h5`` and feeding it to :func:`driver.compute_o3_products` is
wasteful and OOM-prone.

The per-(logN, z)-bin Poisson-binomial deposit is **ADDITIVE over sightlines**, so
we can stream file-by-file, accumulate the per-bin ingredients, and run the
EXISTING CI-combine ONCE on the totals — mathematically IDENTICAL to one combined
file, with bounded memory (one file open at a time) and NO combined file on disk.

THE ADDITIVE SEAM (what accumulates, and why it is exact)
---------------------------------------------------------
For each file we construct ONE :class:`~CDDF_analysis.calc_cddf.DLACatalogue` and a
:class:`~CDDF_analysis.cddf_forward.diagonal_deposit.DiagonalSoftDeposit`, then call
its :meth:`deposit_raw` to obtain, per surviving bin ``b``:

* the deposit-MEAN partition ``F_matched_b`` / ``F_unmatched_b`` / ``n_truth_b``
  (these SUM across files); and
* the estimator's Poisson-binomial MAP ingredients ``(probs_b, poissons_b)`` from
  ``calc_cddf.DLACatalogue._split_distributions`` — ``probs_b`` a LIST of large-p
  arrays (these CONCATENATE across files), ``poissons_b`` the summed small-p mass
  (these SUM across files).

Plus the per-file path length ``dX`` (sums across files).  Because each TARGETID
lives in exactly ONE per-healpix file, no sightline is double-counted; the
accumulated ``(probs, poissons)`` are exactly what ``_split_distributions`` would
return on the single combined file, so the EXISTING CI-combine
(``DLACatalogue._count_ci_from_probs_poissons`` /
``_omega_ci_from_probs_poissons``) reproduces the combined-file CIs to
floating-point.

AFTER the loop we run the UNCHANGED Bayesian core
(``soft_completeness.estimate_diagonal_completeness`` /
``estimate_false_positive_deposit`` / ``apply_diagonal_correction`` /
``omega_from_draws``) ONCE on the totals and package the SAME dict shape as
:func:`driver.compute_o3_products`.

This module is ADDITIVE: it adds new streaming entry points and never alters the
existing single-file driver / estimator outputs.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Union

import numpy as np

from .filter_guard import assert_filter_off, assert_filter_off_from_file
from .window import WindowSpec
from .split import sightline_role, assert_no_leakage
from .diagonal_deposit import build_truth_map, DiagonalSoftDeposit
from ..calc_cddf import DLACatalogue, rho_crit

# Reuse the single-file driver's helpers verbatim (no reimplementation) so the
# streaming products are renormalized / corrected EXACTLY like compute_o3_products.
from . import driver as _driver

try:  # pragma: no cover - import wiring (mirrors driver.py)
    from . import soft_completeness  # type: ignore
except Exception:  # pragma: no cover
    soft_completeness = None  # type: ignore


# --------------------------------------------------------------------------- #
# file discovery
# --------------------------------------------------------------------------- #
def _dndx_z_edges_from_kwargs(z_min, z_max, dlacat_kwargs):
    """The dN/dX z bin edges WITHOUT opening a file.

    ``line_density`` / ``line_density_counts`` use
    ``nbins = max(int((z_max - z_min) * bins_per_z), 1)``.  ``bins_per_z`` is a
    file-independent constructor parameter (default 6), so the z grid is identical
    across every per-healpix file — we read it from ``dlacat_kwargs`` (or its
    default) instead of constructing a throwaway catalogue, keeping the streaming
    construction count == n_files.
    """
    bins_per_z = int(dlacat_kwargs.get("bins_per_z", 6))
    nbins = int(np.max([int((z_max - z_min) * bins_per_z), 1]))
    return np.linspace(z_min, z_max, nbins + 1)


def _resolve_files(processed_files: Union[str, Sequence[str]]) -> List[str]:
    """A directory (glob ``processed-*-*.h5``) OR an explicit list -> sorted paths."""
    if isinstance(processed_files, (str, os.PathLike)):
        path = os.fspath(processed_files)
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "processed-*-*.h5")))
            if not files:
                raise ValueError(
                    f"no processed-*-*.h5 files found under directory {path!r}; "
                    "point at the per-healpix processed/ directory or pass an "
                    "explicit list of file paths."
                )
            return files
        # a single explicit file path
        return [path]
    files = [os.fspath(p) for p in processed_files]
    if not files:
        raise ValueError("processed_files is an empty list.")
    return files


# --------------------------------------------------------------------------- #
# per-bin accumulator
# --------------------------------------------------------------------------- #
class _BinAccumulator:
    """Accumulate the ADDITIVE per-bin ingredients across files.

    ``probs`` is a per-bin LIST that we EXTEND (concatenate) across files;
    ``poissons`` / ``F_matched`` / ``F_unmatched`` / ``n_truth`` are per-bin arrays
    that we SUM.  ``dX`` is a scalar (CDDF) or per-bin array (dN/dX) that sums.
    """

    def __init__(self, nbin: int, *, dx_is_scalar: bool):
        self.nbin = int(nbin)
        self.probs = [list() for _ in range(self.nbin)]
        self.poissons = np.zeros(self.nbin, dtype=float)
        self.F_matched = np.zeros(self.nbin, dtype=float)
        self.F_unmatched = np.zeros(self.nbin, dtype=float)
        self.n_truth = np.zeros(self.nbin, dtype=float)
        self.dx_is_scalar = bool(dx_is_scalar)
        self.dX = 0.0 if dx_is_scalar else np.zeros(self.nbin, dtype=float)

    def add(self, raw: dict, dX):
        if len(raw["probs"]) != self.nbin:
            raise ValueError(
                f"bin count mismatch: accumulator {self.nbin}, file {len(raw['probs'])}"
            )
        for b in range(self.nbin):
            # EXTEND the per-bin list with this file's large-p arrays (concatenate).
            self.probs[b].extend(raw["probs"][b])
        self.poissons += np.asarray(raw["poissons"], float)
        self.F_matched += np.asarray(raw["F_matched"], float)
        self.F_unmatched += np.asarray(raw["F_unmatched"], float)
        self.n_truth += np.asarray(raw["n_truth"], float)
        if self.dx_is_scalar:
            self.dX += float(dX)
        else:
            self.dX = self.dX + np.asarray(dX, float)

    def count_ci(self, ref_cat: DLACatalogue):
        """MAP + 68/95 COUNT CI from the accumulated ``(probs, poissons)``.

        Calls the EXISTING Poisson-binomial + Poisson combine
        (``DLACatalogue._count_ci_from_probs_poissons``) on the TOTALS — exactly the
        combine ``column_density_function_counts`` / ``line_density_counts`` run on
        a single combined file.
        """
        maxlikes, l68, l95 = ref_cat._count_ci_from_probs_poissons(
            self.probs, self.poissons
        )
        return {
            "counts": np.asarray(maxlikes, float),
            "counts68": np.asarray(l68, float),
            "counts95": np.asarray(l95, float),
        }


# --------------------------------------------------------------------------- #
# per-file ingredient extraction (one DLACatalogue per file; closed after)
# --------------------------------------------------------------------------- #
def _per_file_ingredients(
    processed_file, sample_file, catalog_file, truth_file, *,
    z_min, z_max, lnhi_edges, z_edges_cddf, z_edges_dndx, window,
    role_mask=None, split_seed=20260609, build_frac=0.7, dlacat_kwargs,
    keep_open=False,
):
    """Open ONE file, extract the streaming ingredients, then close it.

    Constructs exactly ONE :class:`DLACatalogue` (so a streaming run builds exactly
    ``n_files`` catalogues — no combined-file construction).  Returns a dict
    carrying the CDDF (logN) raw ingredients, the dN/dX (z) raw ingredients,
    per-file dX (CDDF scalar + dN/dX per-bin), coverage provenance, and the active
    TARGETID set (optionally restricted to a split ``role_mask``).

    ``keep_open`` returns the OPEN catalogue under the ``"cat"`` key (the caller is
    then responsible for closing it) so the very first file's catalogue can be
    reused as the CI-combine reference WITHOUT a second construction.  The
    Poisson-binomial combine (``_count_ci_from_probs_poissons`` /
    ``_omega_ci_from_probs_poissons``) only reads ``tophat_prior`` (False) and the
    pure module-level PB functions, so any constructed catalogue is a valid
    reference for combining the accumulated TOTALS.
    """
    cat = DLACatalogue(
        processed_file=processed_file, sample_file=sample_file,
        catalog_file=catalog_file, window=window, **dlacat_kwargs,
    )
    try:
        # Active set for THIS file (the estimator's filter_dla_spectra cuts).
        active_all = set(int(t) for t in cat.target_ids[cat.filter_dla_spectra()[0]])
        if role_mask is not None:
            active = set(
                t for t in active_all
                if sightline_role(t, seed=split_seed, frac_build=build_frac) == role_mask
            )
        else:
            active = active_all
        active_arr = active if role_mask is not None else None

        # ----- CDDF (logN bins, single z window) -----
        tmap_cddf = build_truth_map(
            truth_file, catalog_file=catalog_file, processed_file=processed_file,
            window=window, lnhi_edges=lnhi_edges, z_edges=z_edges_cddf,
            active_target_ids=active,
        )
        dep_cddf = DiagonalSoftDeposit(
            cat, tmap_cddf, lnhi_edges=lnhi_edges, z_edges=z_edges_cddf, window=window
        )
        raw_cddf = dep_cddf.deposit_raw(
            z_min=z_min, z_max=z_max, target_ids=active_arr,
            nhi=True, q_bins=lnhi_edges,
        )
        # CDDF dX over THIS file's active sightlines (scalar).
        if role_mask is None:
            dX_cddf = float(cat.path_length(z_min, z_max))
        else:
            dX_cddf = _driver._role_restricted_dX(cat, active, z_min, z_max)

        # ----- dN/dX (z bins, single logN window) -----
        lnhi_edges_single = np.array([lnhi_edges[0], lnhi_edges[-1]])
        tmap_dndx = build_truth_map(
            truth_file, catalog_file=catalog_file, processed_file=processed_file,
            window=window, lnhi_edges=lnhi_edges_single, z_edges=z_edges_dndx,
            active_target_ids=active,
        )
        dep_dndx = DiagonalSoftDeposit(
            cat, tmap_dndx, lnhi_edges=lnhi_edges_single, z_edges=z_edges_dndx,
            window=window,
        )
        raw_dndx = dep_dndx.deposit_raw(
            z_min=z_min, z_max=z_max, target_ids=active_arr,
            nhi=False, q_bins=z_edges_dndx,
        )
        # dN/dX dX per z bin over THIS file's active sightlines.
        z_lo = z_edges_dndx[:-1]
        z_hi = z_edges_dndx[1:]
        if role_mask is None:
            dX_dndx = np.array(
                [cat.path_length(zl, zh) for zl, zh in zip(z_lo, z_hi)], float
            )
        else:
            dX_dndx = np.array(
                [_driver._role_restricted_dX(cat, active, zl, zh)
                 for zl, zh in zip(z_lo, z_hi)], float
            )

        # ----- O1 omega (per z bin: (probs, poissons) over the lnhi grid) -----
        # Reuse the estimator's per-z-bin omega split so the streaming Ω-O1 CI is
        # byte-identical to omega_dla_cddf on the combined file.
        omega_raw = []  # list over z bins of (probs, poissons)
        for zl, zh in zip(z_lo, z_hi):
            pr, po = _omega_split_one_zbin(
                cat, lnhi_edges, zl, zh, target_ids=active_arr,
            )
            omega_raw.append((pr, np.asarray(po, float)))

        # ----- coverage provenance for THIS file (per-healpix natural) -----
        coverage = _file_coverage(truth_file, processed_file, tmap_cddf)
        coverage["n_active"] = int(len(active))
        coverage["n_active_all"] = int(len(active_all))

        out = {
            "raw_cddf": raw_cddf,
            "dX_cddf": dX_cddf,
            "raw_dndx": raw_dndx,
            "dX_dndx": dX_dndx,
            "omega_raw": omega_raw,
            "active": active,
            "coverage": coverage,
        }
        if keep_open:
            out["cat"] = cat
        return out
    finally:
        # Close the big HDF5 handle so we never hold >1 file's arrays (unless the
        # caller asked to keep it open to reuse as the CI-combine reference).
        if not keep_open:
            try:
                cat.filehandle.close()
            except Exception:  # pragma: no cover
                pass


def _omega_split_one_zbin(cat, lnhi_edges, z_lo, z_hi, *, target_ids=None):
    """``_split_distributions`` over the lnhi grid for ONE z bin (Ω ingredient).

    Mirrors ``_get_omega_confidence_intervals``'s split call; optionally restricted
    to a TARGETID subset via a temporary condition mask (restored on exit).
    """
    lnhi_bins = np.asarray(lnhi_edges, float)
    if target_ids is None:
        return cat._split_distributions(
            lnhi_bins, lred=z_lo, ured=z_hi,
            lnhi_min=lnhi_bins[0], lnhi_max=lnhi_bins[-1], nhi=True,
        )
    keep = np.isin(
        np.asarray(cat.target_ids).astype(np.int64),
        np.array(sorted(int(t) for t in target_ids), dtype=np.int64),
    )
    saved = cat.condition
    try:
        cat.condition = saved & keep
        return cat._split_distributions(
            lnhi_bins, lred=z_lo, ured=z_hi,
            lnhi_min=lnhi_bins[0], lnhi_max=lnhi_bins[-1], nhi=True,
        )
    finally:
        cat.condition = saved


def _file_coverage(truth_file, processed_file, truth_map):
    """Per-file join-coverage (truth-only / processed-only / both + n_absorbers)."""
    import h5py
    from astropy.table import Table

    table = Table.read(truth_file)
    truth_tids = (
        set(int(t) for t in np.asarray(table["TARGETID"]).astype(np.int64))
        if "TARGETID" in table.colnames else set()
    )
    healpix_coverage = None
    n_healpix = None
    with h5py.File(processed_file, "r") as f:
        proc_tids = set(
            int(t) for t in np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
        )
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
        "n_processed_targets": int(len(proc_tids)),
        "n_absorbers_in": int(getattr(truth_map, "n_absorbers_in", 0)),
        "n_absorbers_kept": int(getattr(truth_map, "n_absorbers_kept", 0)),
        "n_healpix": n_healpix,
        "healpix_coverage": healpix_coverage,
    }


# --------------------------------------------------------------------------- #
# O1 + O3 streaming products from accumulated ingredients
# --------------------------------------------------------------------------- #
def _o1_blocks_from_accumulators(
    ref_cat, acc_cddf, acc_dndx, omega_accum, *, lnhi_edges, z_cent_dndx,
    z_min, z_max, hubble,
):
    """Build the O1 (uncorrected) cddf/dndx/omega blocks from the totals.

    Uses the SAME MAP CI-combine the single-file estimator uses
    (``column_density_function`` / ``line_density`` / ``omega_dla_cddf``), so these
    are byte-identical to the combined-file O1.
    """
    # --- CDDF f(N) ---
    ci_c = acc_cddf.count_ci(ref_cat)
    dX_c = acc_cddf.dX
    dN = np.array([10**e2 - 10**e1 for e1, e2 in zip(lnhi_edges[:-1], lnhi_edges[1:])])
    l_Ncent = np.array([(e1 + e2) / 2.0 for e1, e2 in zip(lnhi_edges[:-1], lnhi_edges[1:])])
    cddf = ci_c["counts"] / dX_c / dN
    cddf68 = ci_c["counts68"] / dX_c / np.vstack([dN, dN]).T
    cddf95 = ci_c["counts95"] / dX_c / np.vstack([dN, dN]).T
    xerrs_c = (10**l_Ncent - 10**lnhi_edges[:-1], 10**lnhi_edges[1:] - 10**l_Ncent)

    # --- dN/dX ---
    ci_d = acc_dndx.count_ci(ref_cat)
    dX_d = acc_dndx.dX
    ii = np.where(dX_d > 0)
    dX_d_keep = dX_d[ii]
    dNdX = ci_d["counts"][ii] / dX_d_keep
    dndx68 = ci_d["counts68"][ii] / np.vstack([dX_d_keep, dX_d_keep]).T
    dndx95 = ci_d["counts95"][ii] / np.vstack([dX_d_keep, dX_d_keep]).T
    z_cent_keep = np.asarray(z_cent_dndx)[ii]

    # --- Omega (per z bin, N-weighted PDF convolution on the accumulated probs) ---
    protonmass = 1.67262178e-24
    h100 = 3.2407789e-18 * hubble
    light = 2.99e10
    conversion = protonmass / light * h100 / rho_crit(hubble)
    lnhi_bins = np.asarray(lnhi_edges, float)
    om, om68, om95, z_om = [], [], [], []
    for zb in range(len(z_cent_dndx)):
        if dX_d[zb] <= 0.0:
            continue
        probs_z, poissons_z = omega_accum[zb]
        nhi_like, nhi_68, nhi_95 = ref_cat._omega_ci_from_probs_poissons(
            probs_z, poissons_z, lnhi_bins
        )
        assert nhi_95[0] <= nhi_68[0] <= nhi_like
        assert nhi_95[1] >= nhi_68[1] >= nhi_like
        om.append(conversion * nhi_like / dX_d[zb])
        om68.append([conversion * nhi_68[0] / dX_d[zb], conversion * nhi_68[1] / dX_d[zb]])
        om95.append([conversion * nhi_95[0] / dX_d[zb], conversion * nhi_95[1] / dX_d[zb]])
        z_om.append(z_cent_dndx[zb])

    omega = np.array(om)
    omega68 = np.array(om68).reshape(-1, 2)
    omega95 = np.array(om95).reshape(-1, 2)
    z_om = np.array(z_om)

    return {
        "cddf": {
            "logN": l_Ncent, "f": cddf, "f68": cddf68, "f95": cddf95,
            "xerrs": xerrs_c,
        },
        "dndx": {
            "z": z_cent_keep, "dndx": dNdX, "dndx68": dndx68, "dndx95": dndx95,
            "xerrs": None,
        },
        "omega": {
            "z": z_om, "omega": omega, "omega68": omega68, "omega95": omega95,
            "xerrs": None,
        },
    }


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def compute_o1_products_streaming(
    processed_files: Union[str, Sequence[str]],
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
    """Streaming O1 (uncorrected) CDDF / dN/dX / Ω — no combined file on disk.

    Equals :func:`driver.compute_o1_products` on the single combined file built from
    the same ``processed_files`` (pinned by ``tests/test_cddf_streaming.py``).
    """
    files = _resolve_files(processed_files)
    if window is None:
        window = WindowSpec()
    dlacat_kwargs.setdefault("high_nhi_cut_value", lnhi_max)

    lnhi_edges = np.linspace(lnhi_min, lnhi_max, lnhi_nbins + 1)
    z_edges_cddf = np.array([z_min, z_max])

    acc_cddf = _BinAccumulator(lnhi_nbins, dx_is_scalar=True)
    # dN/dX z grid: file-INDEPENDENT (only bins_per_z) — computed without a file.
    z_edges_dndx = _dndx_z_edges_from_kwargs(z_min, z_max, dlacat_kwargs)
    z_cent_dndx = 0.5 * (z_edges_dndx[:-1] + z_edges_dndx[1:])
    n_z = z_edges_dndx.size - 1
    acc_dndx = _BinAccumulator(n_z, dx_is_scalar=False)
    omega_accum = None  # list over z bins of (probs-list, poissons-array)

    for fp in files:
        assert_filter_off_from_file(
            fp, supplied=filter_low_likelihood, ctx="compute_o1_products_streaming"
        )
    # O1 needs NO truth file: we accumulate the (probs, poissons) directly per file
    # (no truth partition needed for the uncorrected products). One DLACatalogue per
    # file; the FIRST is kept open to reuse as the CI-combine reference.
    per_file_prov = []
    ref_cat = None
    for i, fp in enumerate(files):
        cat = DLACatalogue(
            processed_file=fp, sample_file=sample_file, catalog_file=catalog_file,
            window=window, **dlacat_kwargs,
        )
        try:
            # CDDF (probs, poissons) + dX
            pr_c, po_c = cat._split_distributions(
                lnhi_edges, lred=z_min, ured=z_max,
                lnhi_min=lnhi_min, lnhi_max=lnhi_max, nhi=True,
            )
            acc_cddf.add(
                {"probs": pr_c, "poissons": po_c,
                 "F_matched": np.zeros(lnhi_nbins), "F_unmatched": np.zeros(lnhi_nbins),
                 "n_truth": np.zeros(lnhi_nbins)},
                float(cat.path_length(z_min, z_max)),
            )
            # dN/dX (probs, poissons) per z bin + per-bin dX
            pr_d, po_d = cat._split_distributions(
                z_edges_dndx, lred=z_min, ured=z_max,
                lnhi_min=lnhi_min, lnhi_max=lnhi_max, nhi=False,
            )
            dX_d = np.array(
                [cat.path_length(zl, zh)
                 for zl, zh in zip(z_edges_dndx[:-1], z_edges_dndx[1:])], float
            )
            acc_dndx.add(
                {"probs": pr_d, "poissons": po_d,
                 "F_matched": np.zeros(n_z), "F_unmatched": np.zeros(n_z),
                 "n_truth": np.zeros(n_z)},
                dX_d,
            )
            # Omega per z bin (probs over lnhi grid)
            if omega_accum is None:
                omega_accum = [([list() for _ in range(lnhi_nbins)],
                                np.zeros(lnhi_nbins)) for _ in range(n_z)]
            for zb, (zl, zh) in enumerate(zip(z_edges_dndx[:-1], z_edges_dndx[1:])):
                pr_o, po_o = cat._split_distributions(
                    lnhi_edges, lred=zl, ured=zh,
                    lnhi_min=lnhi_min, lnhi_max=lnhi_max, nhi=True,
                )
                acc_probs, acc_po = omega_accum[zb]
                for b in range(lnhi_nbins):
                    acc_probs[b].extend(pr_o[b])
                omega_accum[zb] = (acc_probs, acc_po + np.asarray(po_o, float))
            per_file_prov.append({"processed_file": fp})
        finally:
            # Keep the FIRST file's catalogue open to reuse as the CI-combine ref.
            if i == 0:
                ref_cat = cat
            else:
                try:
                    cat.filehandle.close()
                except Exception:  # pragma: no cover
                    pass

    # CI-combine ONCE on the totals, reusing the first file's (still-open) catalogue.
    try:
        blocks = _o1_blocks_from_accumulators(
            ref_cat, acc_cddf, acc_dndx, omega_accum,
            lnhi_edges=lnhi_edges, z_cent_dndx=z_cent_dndx,
            z_min=z_min, z_max=z_max, hubble=hubble,
        )
    finally:
        try:
            ref_cat.filehandle.close()
        except Exception:  # pragma: no cover
            pass

    blocks["provenance"] = {
        "processed_files": files, "n_files": len(files),
        "sample_file": sample_file, "catalog_file": catalog_file,
        "filter_low_likelihood": int(filter_low_likelihood),
        "z_min": z_min, "z_max": z_max,
        "lnhi_min": lnhi_min, "lnhi_max": lnhi_max, "lnhi_nbins": lnhi_nbins,
        "hubble": hubble, "window": window, "window_applied": True,
        "streaming": True, "dlacat_kwargs": dict(dlacat_kwargs),
        "per_file": per_file_prov,
    }
    return blocks


def compute_o3_products_streaming(
    processed_files: Union[str, Sequence[str]],
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
    """Streaming O3 diagonal soft-completeness CDDF / dN/dX / Ω — no combined file.

    Equals :func:`driver.compute_o3_products` on the single combined file built from
    ``processed_files`` for EVERY output array (o1 + o3_cddf + o3_dndx +
    completeness + o3_omega), to floating-point tolerance — pinned by
    ``tests/test_cddf_streaming.py``.  Memory is bounded: one file open at a time.

    See the module docstring for the additive seam.  ``processed_files`` is a
    directory (glob ``processed-*-*.h5``) OR an explicit list of file paths.
    """
    files = _resolve_files(processed_files)
    if window is None:
        window = WindowSpec()
    core = _require_core()
    dlacat_kwargs.setdefault("high_nhi_cut_value", lnhi_max)

    # FILTER guard on EVERY file FIRST (refuse a FILTER-on run with no work done).
    for fp in files:
        assert_filter_off_from_file(
            fp, supplied=filter_low_likelihood, ctx="compute_o3_products_streaming"
        )

    lnhi_edges = np.linspace(lnhi_min, lnhi_max, lnhi_nbins + 1)
    z_edges_cddf = np.array([z_min, z_max])

    # dN/dX z grid: file-INDEPENDENT (only bins_per_z, the constructor default 6 or
    # the caller's dlacat_kwargs override) — so we compute it WITHOUT opening a file.
    z_edges_dndx = _dndx_z_edges_from_kwargs(z_min, z_max, dlacat_kwargs)
    z_cent_dndx = 0.5 * (z_edges_dndx[:-1] + z_edges_dndx[1:])
    n_z = z_edges_dndx.size - 1

    acc_cddf = _BinAccumulator(lnhi_nbins, dx_is_scalar=True)
    acc_dndx = _BinAccumulator(n_z, dx_is_scalar=False)
    omega_accum = [([list() for _ in range(lnhi_nbins)], np.zeros(lnhi_nbins))
                   for _ in range(n_z)]

    # union TARGETID coverage accumulators
    n_absorbers_in = 0
    n_absorbers_kept = 0
    per_file_prov = []
    active_union = set()

    # Process each file with exactly ONE DLACatalogue construction; keep the FIRST
    # file's catalogue OPEN to reuse as the CI-combine reference (so a streaming run
    # builds exactly n_files catalogues — no combined file, no extra construction).
    ref_cat = None
    for i, fp in enumerate(files):
        ing = _per_file_ingredients(
            fp, sample_file, catalog_file, truth_file,
            z_min=z_min, z_max=z_max, lnhi_edges=lnhi_edges,
            z_edges_cddf=z_edges_cddf, z_edges_dndx=z_edges_dndx, window=window,
            role_mask=None, split_seed=split_seed, build_frac=build_frac,
            dlacat_kwargs=dlacat_kwargs, keep_open=(i == 0),
        )
        if i == 0:
            ref_cat = ing["cat"]
        acc_cddf.add(ing["raw_cddf"], ing["dX_cddf"])
        acc_dndx.add(ing["raw_dndx"], ing["dX_dndx"])
        for zb in range(n_z):
            pr_o, po_o = ing["omega_raw"][zb]
            acc_probs, acc_po = omega_accum[zb]
            for b in range(lnhi_nbins):
                acc_probs[b].extend(pr_o[b])
            omega_accum[zb] = (acc_probs, acc_po + po_o)
        cov = ing["coverage"]
        n_absorbers_in += cov["n_absorbers_in"]
        n_absorbers_kept += cov["n_absorbers_kept"]
        active_union |= ing["active"]
        per_file_prov.append({"processed_file": fp, "coverage": cov})

    # ---- reuse the FIRST file's (still-open) catalogue for the CI-combine ----
    try:
        # O1 blocks from the SAME accumulated ingredients.
        o1 = _o1_blocks_from_accumulators(
            ref_cat, acc_cddf, acc_dndx, omega_accum,
            lnhi_edges=lnhi_edges, z_cent_dndx=z_cent_dndx,
            z_min=z_min, z_max=z_max, hubble=hubble,
        )
        o1.pop("provenance", None)

        # ===== CDDF correction (logN bins) =====
        ci_c = acc_cddf.count_ci(ref_cat)
        F_cddf = acc_cddf.F_matched + acc_cddf.F_unmatched
        dX_c = acc_cddf.dX
        dN = np.array([10**e2 - 10**e1 for e1, e2 in zip(lnhi_edges[:-1], lnhi_edges[1:])])
        l_Ncent = np.array(
            [(e1 + e2) / 2.0 for e1, e2 in zip(lnhi_edges[:-1], lnhi_edges[1:])]
        )
        cddf_counts = {
            "logN": l_Ncent, "counts": ci_c["counts"],
            "counts68": ci_c["counts68"], "counts95": ci_c["counts95"],
            "dN": dN, "dX": dX_c, "z_min": z_min, "z_max": z_max,
        }
        cddf_ci = _driver._recenter_count_ci(cddf_counts, F_cddf)
        corr_cddf, C_est, bfp_est = _driver._correct_1d(
            core, F_cddf,
            cddf_ci["lo68"], cddf_ci["hi68"], cddf_ci["lo95"], cddf_ci["hi95"],
            acc_cddf.F_matched, acc_cddf.F_unmatched, acc_cddf.n_truth, float(dX_c),
            return_draws=True,
        )
        o3_cddf = {
            "logN": l_Ncent,
            "f": corr_cddf["n_corr"] / dX_c / dN,
            "f68": np.vstack([corr_cddf["lo68"], corr_cddf["hi68"]]).T / dX_c / np.vstack([dN, dN]).T,
            "f95": np.vstack([corr_cddf["lo95"], corr_cddf["hi95"]]).T / dX_c / np.vstack([dN, dN]).T,
            "f_raw": F_cddf / dX_c / dN,
            "f68_raw": np.vstack([cddf_ci["lo68"], cddf_ci["hi68"]]).T / dX_c / np.vstack([dN, dN]).T,
            "f95_raw": np.vstack([cddf_ci["lo95"], cddf_ci["hi95"]]).T / dX_c / np.vstack([dN, dN]).T,
            "valid_mask": corr_cddf["valid_mask"],
            "neg_clip_mask": corr_cddf["neg_clip_mask"],
            "n_corr": corr_cddf["n_corr"],
        }

        # ===== dN/dX correction (z bins; only dX>0 bins, matching line_density) =====
        dX_d = acc_dndx.dX
        ii = np.where(dX_d > 0)[0]
        ci_d = acc_dndx.count_ci(ref_cat)
        dX_d_keep = dX_d[ii]
        F_dndx = (acc_dndx.F_matched + acc_dndx.F_unmatched)[ii]
        fm_d = acc_dndx.F_matched[ii]
        fu_d = acc_dndx.F_unmatched[ii]
        nt_d = acc_dndx.n_truth[ii]
        dndx_counts = {
            "z": z_cent_dndx[ii],
            "counts": ci_d["counts"][ii],
            "counts68": ci_d["counts68"][ii],
            "counts95": ci_d["counts95"][ii],
            "dX": dX_d_keep,
        }
        dndx_ci = _driver._recenter_count_ci(dndx_counts, F_dndx)
        corr_dndx, C_est_d, bfp_est_d = _driver._correct_1d(
            core, F_dndx,
            dndx_ci["lo68"], dndx_ci["hi68"], dndx_ci["lo95"], dndx_ci["hi95"],
            fm_d, fu_d, nt_d, dX_d_keep,
        )
        o3_dndx = {
            "z": z_cent_dndx[ii],
            "dndx": corr_dndx["n_corr"] / dX_d_keep,
            "dndx68": np.vstack([corr_dndx["lo68"], corr_dndx["hi68"]]).T / np.vstack([dX_d_keep, dX_d_keep]).T,
            "dndx95": np.vstack([corr_dndx["lo95"], corr_dndx["hi95"]]).T / np.vstack([dX_d_keep, dX_d_keep]).T,
            "dndx_raw": F_dndx / dX_d_keep,
            "dndx68_raw": np.vstack([dndx_ci["lo68"], dndx_ci["hi68"]]).T / np.vstack([dX_d_keep, dX_d_keep]).T,
            "dndx95_raw": np.vstack([dndx_ci["lo95"], dndx_ci["hi95"]]).T / np.vstack([dX_d_keep, dX_d_keep]).T,
            "valid_mask": corr_dndx["valid_mask"],
            "neg_clip_mask": corr_dndx["neg_clip_mask"],
            "n_corr": corr_dndx["n_corr"],
        }

        # ===== Ω(z) from the corrected count-space CDDF (joint draws) =====
        o3_omega = _driver._omega_from_draws_or_proxy(
            core, corr_cddf, o3_cddf, cddf_counts, hubble=hubble
        )

        # ===== boundary/ceiling provenance (uses the union active set) =====
        boundary_flags = _boundary_ceiling_flags_streaming(
            lnhi_edges, lnhi_max, truth_file, sample_file, active_union,
        )
    finally:
        ref_cat.filehandle.close()

    # union coverage: truth-only over the WHOLE truth catalog vs the union of all
    # processed TARGETIDs (computed once over the truth file + per-file proc sets).
    coverage = _union_coverage(
        truth_file, files, n_absorbers_in, n_absorbers_kept,
    )

    completeness = {
        "logN": l_Ncent,
        "C": np.asarray(C_est["C"]),
        "C_lo68": np.asarray(C_est["C_lo68"]),
        "C_hi68": np.asarray(C_est["C_hi68"]),
        "b_FP": np.asarray(bfp_est["b_FP"]),
        "n_truth": acc_cddf.n_truth,
        "F_matched": acc_cddf.F_matched,
        "F_unmatched": acc_cddf.F_unmatched,
        "C_est": C_est,
        "bfp_est": bfp_est,
    }

    provenance = {
        "processed_files": files, "n_files": len(files),
        "sample_file": sample_file, "catalog_file": catalog_file,
        "truth_file": truth_file,
        "filter_low_likelihood": int(filter_low_likelihood),
        "z_min": z_min, "z_max": z_max,
        "lnhi_min": lnhi_min, "lnhi_max": lnhi_max, "lnhi_nbins": lnhi_nbins,
        "hubble": hubble, "split_seed": split_seed, "build_frac": build_frac,
        "window": window, "window_applied": True, "streaming": True,
        "lnhi_edges": lnhi_edges, "z_edges": z_edges_cddf,
        "z_edges_dndx": z_edges_dndx,
        "dlacat_kwargs": dict(dlacat_kwargs),
        "correction": _driver._O3_LABEL, "limitation": _driver._O3_LIMITATION,
        "n_active_sightlines": int(len(active_union)),
        "coverage": coverage,
        "boundary_flags": boundary_flags,
        "per_file": per_file_prov,
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


def _require_core():
    if soft_completeness is None:
        raise ImportError(
            "the O3 Bayesian core CDDF_analysis.cddf_forward.soft_completeness is "
            "not available; install it or inject a fake in tests."
        )
    return soft_completeness


def _union_coverage(truth_file, files, n_absorbers_in, n_absorbers_kept):
    """Union join-coverage over the truth catalog vs ALL processed TARGETIDs."""
    import h5py
    from astropy.table import Table

    table = Table.read(truth_file)
    truth_tids = (
        set(int(t) for t in np.asarray(table["TARGETID"]).astype(np.int64))
        if "TARGETID" in table.colnames else set()
    )
    proc_tids = set()
    healpix_coverage = {}
    for fp in files:
        with h5py.File(fp, "r") as f:
            proc_tids |= set(
                int(t) for t in np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
            )
            if "healpix" in f:
                hp = np.asarray(f["healpix"][:]).ravel()
                for h in np.unique(hp):
                    healpix_coverage[int(h)] = (
                        healpix_coverage.get(int(h), 0) + int(np.sum(hp == h))
                    )
    both = truth_tids & proc_tids
    return {
        "n_truth_only": int(len(truth_tids - proc_tids)),
        "n_processed_only": int(len(proc_tids - truth_tids)),
        "n_both": int(len(both)),
        "n_truth_targets": int(len(truth_tids)),
        "n_processed_targets": int(len(proc_tids)),
        "n_absorbers_in": int(n_absorbers_in),
        "n_absorbers_kept": int(n_absorbers_kept),
        "n_healpix": (len(healpix_coverage) if healpix_coverage else None),
        "healpix_coverage": (healpix_coverage or None),
    }


def _boundary_ceiling_flags_streaming(
    lnhi_edges, lnhi_max, truth_file, sample_file, active_target_ids
):
    """20.3-boundary per-bin flag + above-ceiling truth count (streaming union).

    Mirrors ``driver._boundary_ceiling_flags`` but counts above-ceiling truth
    absorbers over the UNION active set (the streaming science population).
    """
    import h5py
    from astropy.table import Table

    lnhi_edges = np.asarray(lnhi_edges, float)
    nbin = lnhi_edges.size - 1
    eddington = np.zeros(nbin, dtype=bool)
    eddington[: min(2, nbin)] = True

    with h5py.File(sample_file, "r") as f:
        log_nhi_samples = np.asarray(f["log_nhi_samples"][:]).ravel()
    sample_grid_ceiling = float(np.max(log_nhi_samples))
    assert sample_grid_ceiling >= float(lnhi_max) - 1e-6, (
        f"lnhi_max={lnhi_max} exceeds the QMC sample-grid ceiling "
        f"{sample_grid_ceiling}: bins above the grid would be starved."
    )

    table = Table.read(truth_file)
    nhi_col = next((c for c in ("NHI", "N_HI") if c in table.colnames), None)
    n_above = 0
    if nhi_col is not None and "TARGETID" in table.colnames:
        nhis = np.asarray(table[nhi_col]).astype(float)
        tids = np.asarray(table["TARGETID"]).astype(np.int64)
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


def heldout_closure_streaming(
    processed_files: Union[str, Sequence[str]],
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
    """Streaming HELDOUT closure — equals :func:`driver.heldout_closure` on combined.

    Accumulates BUILD-active and HELDOUT-active ingredients SEPARATELY (split by
    TARGETID via ``split.sightline_role`` — file-independent), then runs the
    existing closure logic (b_FP rebased BUILD→HELDOUT, real count CI on HELDOUT,
    coverage + coherent-bias gate).  ``assert_no_leakage`` runs on the accumulated
    BUILD vs HELDOUT TARGETID sets.
    """
    files = _resolve_files(processed_files)
    if window is None:
        window = WindowSpec()
    core = _require_core()
    dlacat_kwargs.setdefault("high_nhi_cut_value", lnhi_max)
    for fp in files:
        assert_filter_off_from_file(
            fp, supplied=filter_low_likelihood, ctx="heldout_closure_streaming"
        )

    lnhi_edges = np.linspace(lnhi_min, lnhi_max, lnhi_nbins + 1)
    z_edges_cddf = np.array([z_min, z_max])

    acc_build = _BinAccumulator(lnhi_nbins, dx_is_scalar=True)
    acc_held = _BinAccumulator(lnhi_nbins, dx_is_scalar=True)
    build_ids = set()
    held_ids = set()

    for fp in files:
        ing_b = _per_file_ingredients(
            fp, sample_file, catalog_file, truth_file,
            z_min=z_min, z_max=z_max, lnhi_edges=lnhi_edges,
            z_edges_cddf=z_edges_cddf, z_edges_dndx=z_edges_cddf, window=window,
            role_mask="BUILD", split_seed=split_seed, build_frac=build_frac,
            dlacat_kwargs=dlacat_kwargs,
        )
        acc_build.add(ing_b["raw_cddf"], ing_b["dX_cddf"])
        build_ids |= ing_b["active"]

        ing_h = _per_file_ingredients(
            fp, sample_file, catalog_file, truth_file,
            z_min=z_min, z_max=z_max, lnhi_edges=lnhi_edges,
            z_edges_cddf=z_edges_cddf, z_edges_dndx=z_edges_cddf, window=window,
            role_mask="HELDOUT", split_seed=split_seed, build_frac=build_frac,
            dlacat_kwargs=dlacat_kwargs,
        )
        acc_held.add(ing_h["raw_cddf"], ing_h["dX_cddf"])
        held_ids |= ing_h["active"]

    # Enforce non-circularity on the ACCUMULATED role sets.
    assert_no_leakage(
        np.array(sorted(build_ids), dtype=np.int64),
        np.array(sorted(held_ids), dtype=np.int64),
        ctx="heldout_closure_streaming",
    )
    n_build_active = max(len(build_ids), 1)
    n_held_active = len(held_ids)
    bfp_rebase_ratio = n_held_active / n_build_active

    # BUILD-active: C & b_FP (b_FP on the BUILD path length).
    C_est = core.estimate_diagonal_completeness(
        acc_build.F_matched, acc_build.n_truth
    )
    bfp_est_build = core.estimate_false_positive_deposit(
        acc_build.F_unmatched, float(max(acc_build.dX, np.finfo(float).tiny))
    )
    bfp_est = _driver._rebase_bfp(bfp_est_build, bfp_rebase_ratio)

    # HELDOUT-active: recovered F + REAL count CI restricted to HELDOUT.
    F_held = acc_held.F_matched + acc_held.F_unmatched
    truth_held = acc_held.n_truth.astype(float)
    ref_cat = DLACatalogue(
        processed_file=files[0], sample_file=sample_file, catalog_file=catalog_file,
        window=window, **dlacat_kwargs,
    )
    try:
        ci_h = acc_held.count_ci(ref_cat)
    finally:
        ref_cat.filehandle.close()
    held_counts = {
        "logN": 0.5 * (lnhi_edges[:-1] + lnhi_edges[1:]),
        "counts": ci_h["counts"],
        "counts68": ci_h["counts68"],
        "counts95": ci_h["counts95"],
    }
    F_ci = _driver._recenter_count_ci(held_counts, F_held)
    corr = core.apply_diagonal_correction(F_held, F_ci, C_est, bfp_est)
    corrected = np.asarray(corr["n_corr"], float)
    valid = np.asarray(corr["valid_mask"], bool)

    residual = corrected - truth_held
    hi = np.asarray(corr["hi68"], float)
    lo = np.asarray(corr["lo68"], float)
    half = 0.5 * (hi - lo)
    poisson = np.sqrt(np.clip(corrected, 1.0, None))
    half = np.where(np.isfinite(half) & (half > 0), half, poisson)
    with np.errstate(invalid="ignore", divide="ignore"):
        std_resid = np.where(half > 0, residual / half, np.nan)
    finite = valid & np.isfinite(std_resid) & (truth_held > 0)
    n_valid = int(np.sum(finite))

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
        "streaming": True,
        "n_files": len(files),
    }
