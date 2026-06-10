"""O3 diagonal soft-completeness — CS plumbing (truth map + partitioned deposit).

This module is the **CS plumbing** half of the O3 diagonal soft-completeness
correction (contract ``2026-06-10_m2_o3_interface_contract.md`` §3).  It is
ADDITIVE: it never alters any existing ``calc_cddf`` estimator number.  The
Bayesian core (``soft_completeness.py``) turns the count arrays produced here into
``C``/``b_FP``/``n_corr``; it is built in parallel and imported only at the driver.

What "diagonal" means (and its limitation)
------------------------------------------
The O3 correction is *diagonal*: each (logN, z) bin is corrected by its OWN
completeness ``C`` and soft false-positive deposit ``b_FP``.  There is **no
cross-bin migration** modelled here — sub-DLA↔DLA leakage across logN=20.3,
z-scatter, LLS↔DLA contamination are OFF-diagonal channels (O4 / M3-M4),
explicitly out of scope.  Every artifact this module feeds is labelled
"O3 DIAGONAL SOFT-COMPLETENESS" with that caveat.

Two units, both pure-array / file-IO and testable WITHOUT the Bayesian core:

* :func:`build_truth_map` (§3.1) — read the HCD truth FITS catalog, window each
  truth absorber IDENTICALLY to the measurement (the SAME :class:`WindowSpec`),
  bin into (logN, z), and yield a per-TARGETID set of occupied bins, optionally
  restricted to a BUILD / HELDOUT split role.
* :class:`DiagonalSoftDeposit` (§3.2) — a variant of
  ``calc_cddf.DLACatalogue._split_distributions_single`` that routes each
  per-sample DLA probability deposit into ``F_matched`` (the sightline truly hosts
  a truth absorber in that bin) or ``F_unmatched`` (it does not), so that
  ``F_matched + F_unmatched == F`` to 1e-9 (mass-conserving partition).  Reuses
  the existing windowing and ``_get_prob_dla_this_bin`` — the probabilities are
  NOT re-derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import numpy as np
from astropy.table import Table

from .window import WindowSpec
from .split import sightline_role


# Truth-catalog column aliases.  The real 2LPT-0 ``hcd_truth_cat.fits`` uses
# ``NHI`` (log10) + ``Z`` + ``TARGETID`` (confirmed against the file); the contract
# also allows ``N_HI``.  We normalize to those three.
_TARGETID_COLS = ("TARGETID",)
_Z_COLS = ("Z",)
_NHI_COLS = ("NHI", "N_HI")


def _pick_column(table: Table, candidates) -> str:
    for name in candidates:
        if name in table.colnames:
            return name
    raise KeyError(
        f"truth catalog is missing any of {candidates}; has {list(table.colnames)}"
    )


@dataclass
class TruthMap:
    """Per-TARGETID set of occupied (logN, z) bins for windowed truth absorbers.

    Attributes
    ----------
    bins_by_target : dict[int, set[tuple[int, int]]]
        TARGETID -> set of ``(ilnhi, iz)`` bin-index pairs the sightline's
        windowed truth absorbers fall into.  Only sightlines with >=1 surviving
        absorber appear; query a missing one safely with :meth:`bins_for_target`.
    n_lnhi_bins, n_z_bins : int
        Grid dimensions (= ``len(edges) - 1``).
    lnhi_edges, z_edges : np.ndarray
        The edge arrays used to bin (recorded for provenance / re-use).
    window : WindowSpec
        The window applied to the truth absorbers (the SAME spec the measurement
        uses; asserted equal at deposit time).
    """

    bins_by_target: Dict[int, Set[Tuple[int, int]]]
    n_lnhi_bins: int
    n_z_bins: int
    lnhi_edges: np.ndarray
    z_edges: np.ndarray
    window: WindowSpec
    role_mask: Optional[str] = None
    n_absorbers_in: int = 0
    n_absorbers_kept: int = 0

    def bins_for_target(self, target_id: int) -> Set[Tuple[int, int]]:
        """Set of occupied ``(ilnhi, iz)`` bins for ``target_id`` (empty if none)."""
        return self.bins_by_target.get(int(target_id), set())

    def has_truth(self, target_id: int, ilnhi: int, iz: int) -> bool:
        """True iff ``target_id`` has a windowed truth absorber in bin (ilnhi, iz)."""
        return (int(ilnhi), int(iz)) in self.bins_for_target(target_id)

    def n_truth_grid(self) -> np.ndarray:
        """``(n_lnhi_bins, n_z_bins)`` integer count of truth absorbers per bin.

        Counts MULTIPLICITY (a sightline with two truth absorbers in one bin
        contributes 2), consistent with the §2.1 ``n_truth`` definition (number of
        truth absorbers in the bin, not number of hosting sightlines).
        """
        grid = np.zeros((self.n_lnhi_bins, self.n_z_bins), dtype=np.int64)
        for counts in self._multiplicity.values():
            for (il, iz), c in counts.items():
                grid[il, iz] += c
        return grid

    # multiplicity store (TARGETID -> {(il,iz): count}); set by the builder.
    _multiplicity: Dict[int, Dict[Tuple[int, int], int]] = field(
        default_factory=dict, repr=False
    )


def _search_edges(
    window: WindowSpec,
    z_qso: float,
    z_min: float,
    z_max: float,
    max_z_dla: Optional[float] = None,
):
    """Reproduce the measurement's per-sightline ``[lower_z, upper_z]`` search edges.

    Mirrors ``calc_cddf.DLACatalogue._split_distributions_single`` for the
    window's Lyβ-edge selection byte-for-byte, anchored on the SAME constants so
    truth and recovered are windowed identically.  The estimator's base window is
    the GLOBAL ``[lred, ured]`` (here ``[z_min, z_max]``) — the WindowSpec branch
    sets ``lowzcut == highzcut == False`` so NO proximity/tail re-cut is applied
    (the stored ``min_z_dlas``/``max_z_dlas`` already encode it) — and on top of
    that:

    * ``z_min_lyb`` floors the search at ``max(lymanbeta(z_qso), z_min)`` (the QSO
      Lyβ emission, clamped at the GLOBAL ``z_min`` — NOT a per-sightline edge);
    * ``z_max_lyb`` caps it at ``min(lymanbeta(max_z_dla), z_max)`` — the
      PER-SIGHTLINE ``lymanbeta(max_z_dla)`` clamped at the GLOBAL ``z_max``
      (``self.z_max(spec)`` inside the lymanbeta, ``ured`` as the clamp), exactly
      as the estimator does.  ``max_z_dla`` defaults to ``z_max`` only for callers
      that have no separate stored edge (which then reduces to ``lymanbeta(z_max)``).

    The Lyβ rest/Lyα wavelengths come from ``calc_cddf`` (the unified
    ``set_parameters`` values), so this matches the estimator byte-for-byte.
    """
    from ..calc_cddf import lyb_wavelength, lya_wavelength

    if max_z_dla is None:
        max_z_dla = z_max

    lower_z = z_min
    upper_z = z_max
    if window.z_max_lyb:
        # PER-SIGHTLINE lymanbeta(max_z_dla), clamped at the GLOBAL z_max (ured).
        zlyb_max = (1.0 + max_z_dla) * (lyb_wavelength / lya_wavelength) - 1.0
        upper_z = min(zlyb_max, z_max)
    if window.z_min_lyb:
        zlyb_qso = (1.0 + z_qso) * (lyb_wavelength / lya_wavelength) - 1.0
        lower_z = max(zlyb_qso, z_min)
    return lower_z, upper_z


def _load_zqso_by_target(processed_file: str, catalog_file: str):
    """Map each TARGETID present in the processed file to its z_qso + stored edges.

    Returns ``{TARGETID: (z_qso, z_min_dla, z_max_dla)}``.  z_qso comes from the
    processed file (``z_qsos``), the stored search edges from
    ``min_z_dlas``/``max_z_dlas`` — exactly the arrays the measurement windows on.
    The ``catalog_file`` is accepted for symmetry with the estimator (TARGETID
    alignment); z_qso is taken from the processed file, which is authoritative for
    the run.
    """
    import h5py

    with h5py.File(processed_file, "r") as f:
        tids = np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
        z_qsos = np.asarray(f["z_qsos"][:]).astype(float).ravel()
        z_min_dlas = np.asarray(f["min_z_dlas"][:]).astype(float).ravel()
        z_max_dlas = np.asarray(f["max_z_dlas"][:]).astype(float).ravel()
    out = {}
    for tid, zq, zlo, zhi in zip(tids, z_qsos, z_min_dlas, z_max_dlas):
        out[int(tid)] = (float(zq), float(zlo), float(zhi))
    return out


def build_truth_map(
    truth_file: str,
    *,
    catalog_file: str,
    processed_file: str,
    window: WindowSpec,
    lnhi_edges,
    z_edges,
    role_mask: Optional[str] = None,
    split_seed: int = 20260609,
    build_frac: float = 0.7,
    active_target_ids=None,
) -> TruthMap:
    """Build a :class:`TruthMap` from an HCD truth FITS catalog (contract §3.1).

    Each truth absorber is windowed IDENTICALLY to the measurement (via the SAME
    ``window`` and the sightline's stored search edges from ``processed_file``),
    then binned into ``(ilnhi, iz)`` by ``lnhi_edges`` / ``z_edges``.  Absorbers
    outside the window, the logN range, or the z range are dropped.  Sightlines not
    present in ``processed_file`` are skipped (no z_qso / search edges to window
    against).

    Active-set restriction (contract C3 / decision §3)
    --------------------------------------------------
    When ``active_target_ids`` is supplied, ONLY truth absorbers on sightlines in
    that set count.  The set must be the estimator's ACTIVE sightlines
    ``cat.target_ids[cat.filter_dla_spectra()[0]]`` (the SAME SNR / catalog-
    membership / z-range cuts the ``F_matched`` numerator is measured on), so the
    ``n_truth`` DENOMINATOR matches the numerator population.  A truth absorber on a
    low-SNR or catalog-absent sightline (one the GP never effectively searched)
    must NOT inflate ``n_truth`` — otherwise completeness is biased low against a
    population that was never in the measurement.

    Parameters
    ----------
    truth_file : str
        HCD truth FITS catalog with ``NHI``/``N_HI`` (log10), ``Z``, ``TARGETID``.
    catalog_file : str
        QSO FITS catalog (TARGETID alignment; accepted for symmetry with the
        estimator).
    processed_file : str
        The processed HDF5 of the SAME run (supplies per-sightline z_qso and the
        stored ``min_z_dlas``/``max_z_dlas`` search edges the measurement windowed on).
    window : WindowSpec
        Search window; truth and recovered must use the SAME spec (``velocity_scaled``
        must be False — see :class:`WindowSpec`).
    lnhi_edges, z_edges : array-like
        Bin EDGE arrays (bin ``b`` spans ``[edge[b], edge[b+1])``).
    role_mask : {None, "BUILD", "HELDOUT"}
        If given, restrict truth absorbers to sightlines whose TARGETID hashes into
        that split role (``cddf_forward.split.sightline_role``).
    split_seed, build_frac : int, float
        Split parameters for ``role_mask`` (defaults match ``split._DEFAULT_*``).
    active_target_ids : set/iterable of int, optional
        The estimator's ACTIVE TARGETID set.  If given, truth absorbers on
        sightlines outside it are dropped (see "Active-set restriction").

    Returns
    -------
    TruthMap
    """
    if window.velocity_scaled:
        raise NotImplementedError(
            "build_truth_map requires WindowSpec(velocity_scaled=False) — the "
            "constant v/c convention that matches the stored inference search edges."
        )
    lnhi_edges = np.asarray(lnhi_edges, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)
    n_lnhi_bins = lnhi_edges.size - 1
    n_z_bins = z_edges.size - 1

    zqso_by_target = _load_zqso_by_target(processed_file, catalog_file)

    active_set = (
        None if active_target_ids is None else set(int(t) for t in active_target_ids)
    )

    table = Table.read(truth_file)
    tid_col = _pick_column(table, _TARGETID_COLS)
    z_col = _pick_column(table, _Z_COLS)
    nhi_col = _pick_column(table, _NHI_COLS)

    tids = np.asarray(table[tid_col]).astype(np.int64)
    zs = np.asarray(table[z_col]).astype(float)
    nhis = np.asarray(table[nhi_col]).astype(float)

    bins_by_target: Dict[int, Set[Tuple[int, int]]] = {}
    multiplicity: Dict[int, Dict[Tuple[int, int], int]] = {}
    n_in = int(tids.size)
    n_kept = 0

    for tid, z_abs, lnhi in zip(tids, zs, nhis):
        tid = int(tid)
        if active_set is not None and tid not in active_set:
            # not in the estimator's active set (SNR / catalog / z-range cut) ->
            # excluded so n_truth matches the F_matched numerator population.
            continue
        if role_mask is not None:
            if sightline_role(tid, seed=split_seed, frac_build=build_frac) != role_mask:
                continue
        info = zqso_by_target.get(tid)
        if info is None:
            # sightline not in this run's processed file -> nothing to window against
            continue
        z_qso, z_min_dla, z_max_dla = info

        # Mirror the deposit's windowing EXACTLY: the estimator's base window is the
        # GLOBAL [z_edges[0], z_edges[-1]] (the deposit() z_min/z_max), with the
        # z_max_lyb cap using the PER-SIGHTLINE lymanbeta(max_z_dla) clamped at the
        # global z_max.  The recovered samples are additionally bounded by the stored
        # per-sightline [z_min_dla, z_max_dla], so the truth window is the
        # INTERSECTION of that stored range with the global+Lyβ search edges.
        lower_z, upper_z = _search_edges(
            window,
            z_qso,
            z_min=float(z_edges[0]),
            z_max=float(z_edges[-1]),
            max_z_dla=z_max_dla,
        )
        lower_z = max(lower_z, z_min_dla)
        upper_z = min(upper_z, z_max_dla)

        # window + range cut (half-open on the lower logN/z edge, mirroring the
        # estimator's strict ``>``/``<`` on lnhi/z and the edge-array convention).
        if not (lower_z < z_abs < upper_z):
            continue
        if not (lnhi_edges[0] <= lnhi < lnhi_edges[-1]):
            continue
        if not (z_edges[0] <= z_abs < z_edges[-1]):
            continue

        ilnhi = int(np.searchsorted(lnhi_edges, lnhi, side="right") - 1)
        iz = int(np.searchsorted(z_edges, z_abs, side="right") - 1)
        if not (0 <= ilnhi < n_lnhi_bins and 0 <= iz < n_z_bins):
            continue

        bins_by_target.setdefault(tid, set()).add((ilnhi, iz))
        cnt = multiplicity.setdefault(tid, {})
        cnt[(ilnhi, iz)] = cnt.get((ilnhi, iz), 0) + 1
        n_kept += 1

    tmap = TruthMap(
        bins_by_target=bins_by_target,
        n_lnhi_bins=n_lnhi_bins,
        n_z_bins=n_z_bins,
        lnhi_edges=lnhi_edges,
        z_edges=z_edges,
        window=window,
        role_mask=role_mask,
        n_absorbers_in=n_in,
        n_absorbers_kept=n_kept,
    )
    tmap._multiplicity = multiplicity
    return tmap


class DiagonalSoftDeposit:
    """Truth-partitioned probabilistic deposit (contract §3.2).

    Mirrors ``calc_cddf.DLACatalogue._split_distributions_single``'s per-sightline
    deposit loop, but routes each per-sample DLA probability mass deposited in a
    ``(logN, z)`` bin into one of two accumulators by truth-presence:

    * ``F_matched[b]``   — the sightline HAS a windowed truth absorber in bin ``b``;
    * ``F_unmatched[b]`` — it does NOT.

    The partition is exhaustive and mass-conserving:
    ``F_matched + F_unmatched == F`` to 1e-9, where ``F[b]`` is the windowed
    Poisson-binomial MEAN count (the sum of deposited per-sample DLA probabilities)
    in bin ``b`` — the SAME count the O1 estimator's count-space accessor returns
    (cross-checked by :meth:`reference_count_grid`).

    This is a 2-D (logN × z) generalization of the estimator's 1-D deposit; it
    reuses the estimator's windowing (``z_min_lyb``/``z_max_lyb`` via the shared
    ``window``), its sample-parameter helper, ``_get_prob_dla_this_bin`` (the
    probabilities are NOT re-derived), and ``filter_dla_spectra`` (the same active
    set).  Diagonal only — no cross-bin migration is modelled (O4 / out of scope).

    Parameters
    ----------
    catalogue : DLACatalogue
        A constructed estimator on the recovered (FILTER-off) run, built with the
        SAME ``window``.
    truth_map : TruthMap
        Truth occupancy for the SAME split (BUILD or HELDOUT).  Its ``window`` and
        edge arrays must match those passed here (asserted).
    lnhi_edges, z_edges : array-like
        Bin EDGE arrays — must equal the truth map's edges.
    window : WindowSpec
        The shared window; asserted equal to the truth map's window.
    """

    def __init__(self, catalogue, truth_map: TruthMap, *, lnhi_edges, z_edges, window):
        WindowSpec.assert_equal(
            window, truth_map.window, ctx="DiagonalSoftDeposit truth-vs-measurement"
        )
        lnhi_edges = np.asarray(lnhi_edges, dtype=float)
        z_edges = np.asarray(z_edges, dtype=float)
        if not np.array_equal(lnhi_edges, np.asarray(truth_map.lnhi_edges, float)):
            raise ValueError("lnhi_edges must match the truth map's lnhi_edges.")
        if not np.array_equal(z_edges, np.asarray(truth_map.z_edges, float)):
            raise ValueError("z_edges must match the truth map's z_edges.")
        self.cat = catalogue
        self.truth_map = truth_map
        self.lnhi_edges = lnhi_edges
        self.z_edges = z_edges
        self.window = window
        self.n_lnhi_bins = lnhi_edges.size - 1
        self.n_z_bins = z_edges.size - 1

    # -- shared windowed-deposit kernel ----------------------------------------
    def _seconds(self):
        """The ``second`` values to sum over, mirroring ``_split_distributions``.

        ``_split_distributions`` runs ``second=False`` (DLA1) and then, if
        ``self.second_dla``, ``second = k-1`` for ``k = 2 .. second_dla+1``.  We
        reproduce that EXACT sequence so the partitioned deposit's total ``F``
        matches what ``column_density_function`` counts (contract C4 / decision §4):
        ``second_dla == 0`` ⇒ ``[False]`` (single-DLA); ``second_dla == m`` ⇒
        ``[False, 1, 2, ..., m]``.
        """
        seconds = [False]
        second_dla = int(getattr(self.cat, "second_dla", 0) or 0)
        for k in range(2, second_dla + 2):
            seconds.append(k - 1)
        return seconds

    def _iter_sightline_deposits(self, *, z_min, z_max, target_ids=None, second=False):
        """Yield ``(target_id, ilnhi, iz, p_dla)`` for every windowed sample deposit.

        Reproduces the estimator's per-sightline windowing + sample selection +
        ``_get_prob_dla_this_bin`` for the requested ``second`` DLA model, binning
        each surviving sample into a ``(logN, z)`` cell.  This single kernel feeds
        BOTH the partitioned deposit and the unpartitioned reference grid, so they
        are guaranteed consistent.  Callers loop it over :meth:`_seconds` to honor
        ``self.second_dla`` (the multi-DLA sum), exactly as ``_split_distributions``.

        ``target_ids`` (set/iterable, optional) restricts the deposit to those
        TARGETIDs — the driver uses this to deposit on the BUILD split only.
        ``second`` selects the DLA model (``False`` = DLA1; ``k`` = DLA(k+1)); the
        sample parameters / probabilities are taken from the estimator for that
        ``second`` (NOT re-derived).
        """
        cat = self.cat
        allowed = None if target_ids is None else set(int(t) for t in target_ids)
        lnhi_min = float(self.lnhi_edges[0])
        lnhi_max_grid = float(self.lnhi_edges[-1])
        dla_ind = cat.filter_dla_spectra(second=second)
        for spec in dla_ind[0]:
            if allowed is not None and int(cat.target_ids[spec]) not in allowed:
                continue
            (lnhi_vals, redshifts) = cat._get_sample_params(spec, second=second)

            lower_z = z_min
            upper_z = z_max
            if cat.z_max_lyb:
                upper_z = np.min([cat.lymanbeta(cat.z_max(spec)), z_max])
            if cat.z_min_lyb:
                lower_z = np.max([cat.lymanbeta(cat.z_qsos[spec]), z_min])
            if cat.lowzcut:
                upper_z = np.min([cat.proximity(cat.z_max(spec)), z_max])
            if cat.highzcut:
                lower_z = np.max([cat.tail(cat.z_min(spec)), z_min])

            # the estimator caps the high-N tail via high_nhi_cut_value; here the
            # grid's own upper edge is the cap (samples above it bin out anyway).
            hi = lnhi_max_grid
            if cat.high_nhi_cut:
                hi = min(hi, cat.high_nhi_cut_value)

            desired = (
                (lnhi_vals > lnhi_min)
                * (lnhi_vals < hi)
                * (redshifts < upper_z)
                * (redshifts > lower_z)
            )
            ind = np.where(desired)
            if np.size(ind) == 0:
                continue
            p_dla = cat._get_prob_dla_this_bin(spec, ind[0], second=second)
            keep = np.where(p_dla > cat.p_thresh_sample)[0]
            if keep.size == 0:
                continue
            lnhi_k = lnhi_vals[ind][keep]
            z_k = redshifts[ind][keep]
            p_k = p_dla[keep]

            target_id = int(cat.target_ids[spec])
            for lnhi_v, z_v, p_v in zip(lnhi_k, z_k, p_k):
                if not (self.lnhi_edges[0] <= lnhi_v < self.lnhi_edges[-1]):
                    continue
                if not (self.z_edges[0] <= z_v < self.z_edges[-1]):
                    continue
                il = int(np.searchsorted(self.lnhi_edges, lnhi_v, side="right") - 1)
                iz = int(np.searchsorted(self.z_edges, z_v, side="right") - 1)
                if not (0 <= il < self.n_lnhi_bins and 0 <= iz < self.n_z_bins):
                    continue
                yield target_id, il, iz, float(p_v)

    def deposit(self, *, z_min, z_max, target_ids=None) -> dict:
        """Partition the windowed probabilistic deposit by truth-presence.

        Honors ``self.second_dla`` by SUMMING the deposit over ``second=0..second_dla``
        (via :meth:`_seconds`), mirroring ``_split_distributions`` so the total ``F``
        matches ``column_density_function`` (contract C4).  ``n_truth`` keeps truth
        MULTIPLICITY, so for ``second_dla==0`` a close pair (two truths in one cell)
        correctly registers as single-DLA incompleteness (``C<1``); for
        ``second_dla>0`` the second-DLA pass recovers the pair toward ``C→1``.

        ``target_ids`` (optional) restricts the deposit to those sightlines (e.g.
        the BUILD split); the partition stays mass-conserving within the subset.

        Returns
        -------
        dict
            ``F``           : (nlnhi, nz) total windowed expected counts;
            ``F_matched``   : deposited by sightlines hosting a truth absorber in
                              that bin;
            ``F_unmatched`` : deposited by sightlines with NO truth absorber there;
            ``n_truth``     : (nlnhi, nz) integer truth counts (from the TruthMap).
        ``F_matched + F_unmatched == F`` to 1e-9.
        """
        shape = (self.n_lnhi_bins, self.n_z_bins)
        F = np.zeros(shape, dtype=float)
        F_matched = np.zeros(shape, dtype=float)
        F_unmatched = np.zeros(shape, dtype=float)
        tmap = self.truth_map
        for second in self._seconds():
            for target_id, il, iz, p in self._iter_sightline_deposits(
                z_min=z_min, z_max=z_max, target_ids=target_ids, second=second
            ):
                F[il, iz] += p
                if tmap.has_truth(target_id, il, iz):
                    F_matched[il, iz] += p
                else:
                    F_unmatched[il, iz] += p
        return {
            "F": F,
            "F_matched": F_matched,
            "F_unmatched": F_unmatched,
            "n_truth": tmap.n_truth_grid(),
        }

    def reference_count_grid(self, *, z_min, z_max) -> np.ndarray:
        """Unpartitioned windowed expected-count grid (independent of truth).

        A second accumulation over the SAME windowed deposit kernel (summed over
        ``second=0..second_dla`` via :meth:`_seconds`, exactly like :meth:`deposit`),
        summing all deposited ``p_dla`` per ``(logN, z)`` cell without partitioning.
        Used to pin ``F == reference`` so the partition's total is the genuine
        windowed Poisson-binomial mean count, not an artifact of the matched/
        unmatched split.
        """
        grid = np.zeros((self.n_lnhi_bins, self.n_z_bins), dtype=float)
        for second in self._seconds():
            for _tid, il, iz, p in self._iter_sightline_deposits(
                z_min=z_min, z_max=z_max, second=second
            ):
                grid[il, iz] += p
        return grid
