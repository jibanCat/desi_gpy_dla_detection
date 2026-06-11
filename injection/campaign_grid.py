"""
examples/inject/campaign_grid.py
================================
M3 injection-campaign GRID + clean-sightline SAMPLER + MANIFEST SCHEMA.

Scope (the Bayesian-modeling owner's half of the M3 design,
``2026-06-10_m3_injection_campaign_design.md``).  PURE python/numpy — NO
desispec, NO coadd I/O (that is the CS agent's ``coadd_injection.py``, which
CONSUMES the manifest this module emits).

What this builds
----------------
1. ``build_injection_grid(...) -> list[dict]`` — the (logN_true × z_true ×
   SNR_bin) injection cells, DENSE in [17.2, 19.0] (the LLS/sub-DLA regime where
   the single-absorber GP is weakest), moderate in [19.0, 20.3], coarse in
   [20.3, 22.5].  One manifest row per injection.  Sized to ``target_injections``
   so the driver can hit a CPU-h budget (≤4000 CPU-h cap; ~131.5 s/spec).
2. ``sample_clean_sightlines(...) -> assignment`` — deterministic (seeded),
   SNR-bin-balanced draw of CLEAN (HCD-free ∩ BAL-free) TARGETIDs, no reuse
   across cells.
3. ``MANIFEST_FIELDS`` + ``validate_manifest(...)`` — the EXACT manifest schema
   (the CONTRACT for the CS injector) and a guard.

Manifest schema (CONTRACT — keep EXACT)
---------------------------------------
One row per injection::

    {inj_id:int, campaign:str['A'|'B'|'D'], method:str['coadd'|'gpdraw'],
     target_id:int, healpix:int, z_qso:float, snr_bin:int, native_snr:float,
     logN_true:float, z_true:float, num_lines:int}

Campaign B (close pairs) additionally carries the OPTIONAL fields
``logN_true2, z_true2, dv_kms``.

The injected absorber is at OBSERVED redshift ``z_true``; the CS injector calls
``inject_voigt(observed_wavelengths, flux, 10**logN_true, z_true,
num_lines=num_lines)`` on the clean coadd's flux, keeping ivar/noise as-is so the
native SNR is preserved (Campaign A varies SNR by clean-sightline selection).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

# Reuse the REAL inference constants so the injection z-window tracks exactly the
# GP search window (``Parameters.min_z_dla`` / ``max_z_dla``).  Do NOT hardcode
# the Lyman limit / Lyα / the 3000 km/s proximity convention — import them, so a
# change in the inference is reflected here and the campaign never injects outside
# the GP-searchable z (M1 review finding).
from gpy_dla_detection.set_parameters import Parameters as _Parameters

# --------------------------------------------------------------------------- #
# Manifest schema — the EXACT contract the CS coadd injector consumes.
# --------------------------------------------------------------------------- #
# The first 11 keys are the FROZEN contract (kept byte-stable for the CS
# injector).  ``control`` is an ADDITIVE backward-compatible flag (M2): True on
# the no-injection control rows that carry NaN ``logN_true`` / ``z_true`` and
# supply the data-driven ``b_FP``; False on every ordinary injection row.
MANIFEST_FIELDS = (
    "inj_id",
    "campaign",
    "method",
    "target_id",
    "healpix",
    "z_qso",
    "snr_bin",
    "native_snr",
    "logN_true",
    "z_true",
    "num_lines",
    "control",
)

# Optional close-pair fields (Campaign B only).
CLOSE_PAIR_FIELDS = ("logN_true2", "z_true2", "dv_kms")

# The 2LPT-0 FILTER-off production run used NUM_FOREST_LINES = 31 (the GP's
# Lyman-series line count).  The injection MUST match it so recovery is faithful
# (the injected Voigt and the GP's forward model use the same line count).
DEFAULT_NUM_LINES = 31

# Full QMC sample-grid log N_HI range (the design's R range): the sub-DLA/LLS
# regime is covered, 20.3 is an INTERIOR point.
LOGN_MIN = 17.2
LOGN_MAX = 22.5
LOGN_KNEE = 20.3  # sub-DLA <-> DLA migration knee (interior grid point)

# Global DESI absorber-redshift window (from constants.py): zmin_search = 2.15
# floor, zmax_qso = 4.25 ceiling.  The z grid lives strictly inside this; each
# injected z_true is additionally clamped into its host z_qso's GP search window.
Z_SEARCH_MIN = 2.15
Z_SEARCH_MAX = 4.25

# Real inference constants (imported from ``Parameters`` — NOT hardcoded) that
# define the per-sightline GP search window so the injection window == the
# inference window (M1).  ``Parameters.min_z_dla`` uses the Lyman limit + a
# 3000 km/s blue buffer; ``max_z_dla`` uses z_qso - 3000 km/s AND the model's
# ``max_lambda`` red edge.
_LYA_REST = float(_Parameters.lya_wavelength)        # 1215.6701 Å
_LYB_REST = float(_Parameters.lyb_wavelength)        # 1025.7223 Å  (provenance only)
_LYMAN_LIMIT = float(_Parameters.lyman_limit)        # 911.7633 Å — the GP blue edge
_KMS_TO_Z_3000 = float(_Parameters.kms_to_z(3000.0))  # the v/c = 3000 km/s convention

# The 2LPT-0 FILTER-off V1 production run sets MAX_LAMBDA = 1250 Å (overrides the
# _base default 1216.75).  The GP red search edge is (1+z_qso)*MAX_LAMBDA/Lyα - 1,
# so the injection red ceiling must use the SAME value.  Imported convention, not
# a hardcode of the window geometry; this is the run-config knob the campaign
# corrects against (see slurm/greatlakes/production/london0_gl_v1.env).
_MAX_LAMBDA = 1250.0


# --------------------------------------------------------------------------- #
# Default grids
# --------------------------------------------------------------------------- #
def default_logn_grid(
    refine_around: Optional[Sequence[float]] = None,
    refine_step: float = 0.1,
    refine_halfwidth: float = 0.75,
) -> np.ndarray:
    """Default log10(N_HI) injection grid — DENSE below 19, coarse above 20.3.

    Three regimes, each at a science-motivated spacing:

    * ``[17.2, 19.0]`` LLS regime, **Δ = 0.2 dex** (10 points 17.2..19.0) — the
      single-absorber GP is weakest here (faint likelihood → prior-dominated
      posterior), so it needs the finest resolution to measure the few-% detection
      rate and the large prior-pulled N_HI bias.
    * ``(19.0, 20.3]`` sub-DLA, **Δ ≈ 0.26 dex** (5 steps) — moderate; the 20.3
      knee is INCLUDED as an interior point.
    * ``(20.3, 22.5]`` DLA, **Δ ≈ 0.4 dex** (6 steps) — coarse; recovery is robust
      and the diagonal already works here.

    Returns a strictly-increasing edge array spanning the full QMC range.

    ``refine_around`` — POST-PILOT HOOK (minor review finding)
    ----------------------------------------------------------
    The flat Δ=0.2 dex below 19 is the PILOT grid.  The detection turn-on
    ``C_det(N)`` is SNR-dependent and not yet measured, so we do NOT guess where to
    refine here.  After the pilot measures ``C_det(N)``, pass the measured turn-on
    log N_HI value(s) as ``refine_around`` to insert a ~``refine_step`` (default
    0.1 dex) sub-grid within ``±refine_halfwidth`` of each, WITHOUT changing the
    coarse backbone.  Default ``None`` → the pilot grid unchanged.
    """
    below19 = np.round(np.arange(17.2, 19.0 + 1e-9, 0.2), 6)          # 10 pts: 17.2..19.0
    subdla = np.round(np.linspace(19.0, 20.3, 6)[1:], 6)             # 5 pts to the knee
    dla = np.round(np.linspace(20.3, 22.5, 7)[1:], 6)               # 6 pts above the knee
    grid = np.concatenate([below19, subdla, dla])
    # Optional post-pilot refinement around the measured SNR-dependent turn-on.
    if refine_around is not None:
        extra = []
        for center in np.atleast_1d(np.asarray(refine_around, dtype=float)):
            lo = max(float(center) - refine_halfwidth, LOGN_MIN)
            hi = min(float(center) + refine_halfwidth, LOGN_MAX)
            if hi > lo:
                n = int(round((hi - lo) / float(refine_step))) + 1
                extra.append(np.round(np.linspace(lo, hi, max(n, 2)), 6))
        if extra:
            grid = np.concatenate([grid] + extra)
    # de-dup the shared 19.0 / 20.3 join points (and any refinement overlaps), sort
    grid = np.unique(grid)
    return grid.astype(float)


def default_z_grid() -> np.ndarray:
    """Default injected-absorber redshift grid — a few bins across the window.

    Four points spanning the DESI absorber-redshift window; each is additionally
    clamped blueward of the host sightline's z_qso at build time (an absorber
    cannot sit redward of its QSO).
    """
    return np.round(np.linspace(2.3, 3.8, 4), 6).astype(float)


# --------------------------------------------------------------------------- #
# clean-sightline sampler
# --------------------------------------------------------------------------- #
def _snr_bin_index(snr: float, snr_bins: Sequence[float]) -> int:
    """Index ``b`` such that ``snr_bins[b] <= snr < snr_bins[b+1]`` (-1 if out)."""
    b = int(np.searchsorted(snr_bins, snr, side="right") - 1)
    if 0 <= b < len(snr_bins) - 1:
        return b
    return -1


def sample_clean_sightlines(
    clean_table,
    snr_table: Mapping[int, float],
    *,
    n_per_cell: int,
    snr_bins: Sequence[float],
    seed: int,
    allow_reuse: bool = False,
) -> Dict[int, np.ndarray]:
    """Deterministically draw ``n_per_cell`` CLEAN TARGETIDs per SNR bin.

    Parameters
    ----------
    clean_table : iterable of int
        CLEAN (HCD-free ∩ BAL-free) TARGETIDs — the CS agent supplies this set
        (``zcat - hcd_truth - bal_cat``).  Accepts an array or any int-iterable.
    snr_table : mapping {TARGETID -> native SNR}
        Per-sightline SNR (from ``snr_cat``).  Sightlines absent here are skipped.
    n_per_cell : int
        Number of sightlines to draw per SNR bin.  If a bin has fewer than this,
        ALL of its sightlines are returned (no error, no duplication).
    snr_bins : sequence of float
        Monotone SNR bin EDGES; bin ``b`` spans ``[snr_bins[b], snr_bins[b+1])``.
    seed : int
        Deterministic seed.  Same seed → identical assignment (pinned by tests).
    allow_reuse : bool, default False
        If False (default) a TARGETID drawn for one bin is removed from the pool so
        NO sightline is reused across bins (each clean sightline hosts at most one
        injection cell's draw).  Set True to allow reuse (flagged caller intent).

    Returns
    -------
    dict {snr_bin_index -> np.ndarray[TARGETID]}
        One entry per SNR bin (0 .. len(snr_bins)-2).  Deterministic order.

    Determinism
    -----------
    A single ``np.random.default_rng(seed)`` drives a per-bin
    ``Generator.choice(..., replace=False)`` over the SORTED candidate TARGETIDs,
    so the draw is reproducible and independent of dict iteration order.
    """
    snr_bins = np.asarray(snr_bins, dtype=float)
    if snr_bins.ndim != 1 or snr_bins.size < 2 or np.any(np.diff(snr_bins) <= 0):
        raise ValueError("snr_bins must be a strictly-increasing edge array of >=2.")
    n_bins = snr_bins.size - 1
    rng = np.random.default_rng(int(seed))

    # Bucket clean TARGETIDs (that have a known SNR) into SNR bins; SORT each
    # bucket for order-independent, reproducible sampling.
    buckets: Dict[int, list] = {b: [] for b in range(n_bins)}
    for t in clean_table:
        t = int(t)
        snr = snr_table.get(t)
        if snr is None or not np.isfinite(snr):
            continue
        b = _snr_bin_index(float(snr), snr_bins)
        if b >= 0:
            buckets[b].append(t)

    assignment: Dict[int, np.ndarray] = {}
    used: set = set()
    for b in range(n_bins):
        candidates = sorted(buckets[b])
        if not allow_reuse:
            candidates = [t for t in candidates if t not in used]
        k = min(int(n_per_cell), len(candidates))
        if k <= 0:
            assignment[b] = np.empty(0, dtype=np.int64)
            continue
        picked = rng.choice(np.asarray(candidates, dtype=np.int64), size=k, replace=False)
        picked = np.sort(picked)  # deterministic order in the manifest
        assignment[b] = picked.astype(np.int64)
        if not allow_reuse:
            used.update(int(t) for t in picked)
    return assignment


# --------------------------------------------------------------------------- #
# grid builder
# --------------------------------------------------------------------------- #
def _per_sightline_forest_window(z_qso: float):
    """The [z_lo, z_hi] absorber-redshift window for a sightline of given z_qso.

    This is the GP's OWN search window (``Parameters.min_z_dla`` /
    ``max_z_dla``), reconstructed from the imported inference constants so the
    injection campaign never lands an absorber outside the GP-searchable z (M1).

    * BLUE floor — the GP searches down to the **Lyman limit** (911.7633 Å), with
      a 3000 km/s buffer::

          z_lo = max((1+z_qso)*lyman_limit/lya - 1 + kms_to_z(3000), Z_SEARCH_MIN)

      (reaches the Lyman limit, NOT the Lyβ rest — the old Lyβ floor cut Δz≈0.33
      of the bluest forest, exactly the worst Lyβ-misID / LLS-leakage region.)
    * RED ceiling — the proximity buffer (z_qso - 3000 km/s) AND the model's
      ``max_lambda`` red edge::

          z_hi = min(z_qso - kms_to_z(3000),
                     (1+z_qso)*max_lambda/lya - 1,
                     Z_SEARCH_MAX)

      (NOT the bare z_qso — that ignored the proximity margin and MAX_LAMBDA, so
      injections in the proximity zone fell above ``max_z_dla`` → false
      non-detections at the edge z-bins.)
    """
    z_lo_limit = (1.0 + z_qso) * (_LYMAN_LIMIT / _LYA_REST) - 1.0 + _KMS_TO_Z_3000
    z_lo = max(z_lo_limit, Z_SEARCH_MIN)
    z_hi = min(
        z_qso - _KMS_TO_Z_3000,
        (1.0 + z_qso) * (_MAX_LAMBDA / _LYA_REST) - 1.0,
        Z_SEARCH_MAX,
    )
    return z_lo, z_hi


def _clamp_z_into_window(z: float, z_lo: float, z_hi: float) -> float:
    """Clamp an injected ``z`` into the GP search window ``[z_lo, z_hi]`` (M1).

    Returns ``np.nan`` if the window is empty (``z_lo > z_hi``) — the caller must
    then skip the cell (no valid injection z exists for that sightline).
    """
    if z_lo > z_hi:
        return float("nan")
    return float(min(max(z, z_lo), z_hi))


def _normalize_clean_sightlines(clean_sightlines) -> Dict[int, dict]:
    """Normalize the clean-sightline table to {TARGETID -> {healpix,z_qso,native_snr}}.

    Accepts the dict-of-arrays form
    ``{"target_id","healpix","z_qso","native_snr"}`` (the toy/test + the CS
    agent's table) or a list of per-row dicts.
    """
    if isinstance(clean_sightlines, Mapping) and "target_id" in clean_sightlines:
        tids = np.asarray(clean_sightlines["target_id"]).astype(np.int64)
        hp = np.asarray(clean_sightlines["healpix"]).astype(np.int64)
        zq = np.asarray(clean_sightlines["z_qso"]).astype(float)
        snr = np.asarray(clean_sightlines["native_snr"]).astype(float)
        return {
            int(t): {"healpix": int(h), "z_qso": float(z), "native_snr": float(s)}
            for t, h, z, s in zip(tids, hp, zq, snr)
        }
    out: Dict[int, dict] = {}
    for row in clean_sightlines:
        out[int(row["target_id"])] = {
            "healpix": int(row["healpix"]),
            "z_qso": float(row["z_qso"]),
            "native_snr": float(row["native_snr"]),
        }
    return out


def _resolve_n_per_cell(n_cells, n_per_cell, target_injections) -> int:
    """Per-cell injection count honoring ``target_injections`` (the CPU-h budget).

    If ``target_injections`` is given, choose the LARGEST ``n_per_cell`` whose
    total (``n_cells * n_per_cell``) does not exceed it (floor division, min 1 so
    every cell gets at least one injection — if even 1/cell overflows the budget,
    the caller must coarsen the grid, which is surfaced by the cap).
    """
    if target_injections is not None:
        if n_cells <= 0:
            return 0
        n = max(int(target_injections) // int(n_cells), 1)
        return n
    if n_per_cell is None:
        raise ValueError("provide either n_per_cell or target_injections")
    return int(n_per_cell)


def build_injection_grid(
    clean_sightlines,
    *,
    logN_grid: Optional[Sequence[float]] = None,
    z_grid: Optional[Sequence[float]] = None,
    snr_bins: Sequence[float],
    n_per_cell: Optional[int] = None,
    target_injections: Optional[int] = None,
    seed: int,
    campaign: str = "A",
    method: str = "coadd",
    num_lines: int = DEFAULT_NUM_LINES,
) -> List[dict]:
    """Build the (logN_true × z_true × SNR_bin) injection grid → manifest rows.

    One dict per injection, carrying EXACTLY the :data:`MANIFEST_FIELDS` keys (the
    contract for the CS coadd injector).  For each (logN, z, SNR-bin) cell we draw
    ``n_per_cell`` CLEAN sightlines whose native SNR falls in that bin AND whose
    forest window can host the injected ``z`` (``z < z_qso``), via
    :func:`sample_clean_sightlines` (deterministic, no reuse across cells within a
    given (logN, z) — each draw is independent per cell but balanced per SNR bin).

    Parameters
    ----------
    clean_sightlines : dict-of-arrays or list-of-dicts
        CLEAN sightline table with ``target_id, healpix, z_qso, native_snr`` (the
        CS agent supplies it; HCD-free ∩ BAL-free).
    logN_grid, z_grid : sequences, optional
        Injection grids; default to :func:`default_logn_grid` / :func:`default_z_grid`.
    snr_bins : sequence of float
        Monotone native-SNR bin EDGES.
    n_per_cell : int, optional
        Sightlines per cell.  Mutually-informative with ``target_injections``.
    target_injections : int, optional
        Cap on the TOTAL injection count (the CPU-h budget knob).  If given,
        ``n_per_cell`` is sized down so ``len(rows) <= target_injections``.
    seed : int
        Deterministic master seed.
    campaign, method : str
        Manifest ``campaign`` ∈ {'A','B','D'} and ``method`` ∈ {'coadd','gpdraw'}.
    num_lines : int, default 31
        Lyman-series line count (MUST match the run's NUM_FOREST_LINES).

    Returns
    -------
    list[dict]
        Manifest rows with contiguous ``inj_id`` 0..N-1.
    """
    logN_grid = np.asarray(
        default_logn_grid() if logN_grid is None else logN_grid, dtype=float
    )
    z_grid = np.asarray(default_z_grid() if z_grid is None else z_grid, dtype=float)
    snr_bins = np.asarray(snr_bins, dtype=float)
    if snr_bins.ndim != 1 or snr_bins.size < 2 or np.any(np.diff(snr_bins) <= 0):
        raise ValueError("snr_bins must be a strictly-increasing edge array of >=2.")
    n_snr = snr_bins.size - 1

    sl = _normalize_clean_sightlines(clean_sightlines)
    snr_table = {t: info["native_snr"] for t, info in sl.items()}
    all_tids = np.array(sorted(sl.keys()), dtype=np.int64)

    n_cells = int(logN_grid.size) * int(z_grid.size) * int(n_snr)
    per_cell = _resolve_n_per_cell(n_cells, n_per_cell, target_injections)

    rng = np.random.default_rng(int(seed))
    rows: List[dict] = []
    inj_id = 0
    # Iterate cells in a fixed (logN, z, snr_bin) order for determinism.
    for logN in logN_grid:
        for z_true in z_grid:
            # Candidate sightlines whose GP search window can host this grid z
            # (i.e. the grid z falls inside [z_lo, z_hi]).  This is a per-(logN,z)
            # draw.  The emitted z_true is additionally CLAMPED into the window
            # below so float-edge cases never escape it (M1).
            hostable = []
            for t in all_tids:
                z_lo, z_hi = _per_sightline_forest_window(sl[int(t)]["z_qso"])
                if z_lo <= z_true <= z_hi:
                    hostable.append(int(t))
            hostable = np.array(hostable, dtype=np.int64)
            if hostable.size == 0:
                continue
            sub_snr = {int(t): snr_table[int(t)] for t in hostable}
            # Per-cell seed derived from the master rng so the whole build is
            # reproducible yet each (logN,z) cell draws an independent balanced set.
            cell_seed = int(rng.integers(1, 2**31 - 1))
            assign = sample_clean_sightlines(
                hostable, sub_snr,
                n_per_cell=per_cell, snr_bins=snr_bins, seed=cell_seed,
            )
            for b in range(n_snr):
                for t in assign.get(b, ()):
                    info = sl[int(t)]
                    z_lo, z_hi = _per_sightline_forest_window(info["z_qso"])
                    z_emit = _clamp_z_into_window(float(z_true), z_lo, z_hi)
                    if not np.isfinite(z_emit):
                        continue  # empty window — no valid injection z (skip)
                    rows.append({
                        "inj_id": inj_id,
                        "campaign": str(campaign),
                        "method": str(method),
                        "target_id": int(t),
                        "healpix": int(info["healpix"]),
                        "z_qso": float(info["z_qso"]),
                        "snr_bin": int(b),
                        "native_snr": float(info["native_snr"]),
                        "logN_true": float(logN),
                        "z_true": float(z_emit),
                        "num_lines": int(num_lines),
                        "control": False,
                    })
                    inj_id += 1
    # Enforce the total cap exactly (the per-cell floor can leave a small surplus
    # only if n_cells does not divide target_injections; trim deterministically).
    if target_injections is not None and len(rows) > int(target_injections):
        rows = rows[: int(target_injections)]
    return rows


# --------------------------------------------------------------------------- #
# M2 — control-row generator (clean sightlines, NO injection) for b_FP.
# --------------------------------------------------------------------------- #
def build_control_rows(
    clean_sightlines,
    *,
    snr_bins: Sequence[float],
    n_per_cell: Optional[int] = None,
    target_controls: Optional[int] = None,
    seed: int,
    campaign: str = "A",
    method: str = "coadd",
    num_lines: int = DEFAULT_NUM_LINES,
    inj_id_start: int = 0,
    exclude_target_ids: Optional[Iterable] = None,
) -> List[dict]:
    """CLEAN no-injection CONTROL rows — the data path for ``b_FP`` (M2).

    Each control row is a clean (HCD-free ∩ BAL-free) sightline run through the
    SAME GP driver but with NO absorber injected, so any recovered deposit is a
    spurious false positive.  These rows carry ``logN_true = NaN`` /
    ``z_true = NaN`` and ``control = True``; ``measurements.response_matrix``
    measures ``b_FP`` from them (the ``estimate_false_positive_deposit`` path),
    making ``b_FP`` data-driven instead of collapsing to the bare Gamma prior.

    The draw is SNR-bin-balanced (so b_FP is resolved by SNR) and deterministic,
    reusing :func:`sample_clean_sightlines`.  ``inj_id`` starts at
    ``inj_id_start`` so control rows can be appended to an injection manifest with
    globally-contiguous ids.

    Parameters
    ----------
    clean_sightlines : dict-of-arrays or list-of-dicts
        CLEAN sightline table (see :func:`build_injection_grid`).
    snr_bins : sequence of float
        Monotone native-SNR bin EDGES (same basis as the injection grid).
    n_per_cell : int, optional
        Control sightlines per SNR bin.  Mutually-informative with
        ``target_controls``.
    target_controls : int, optional
        Cap on the TOTAL control count (sizes ``n_per_cell`` down to fit).
    seed : int
        Deterministic seed.
    campaign, method, num_lines : str / int
        Manifest tags (default ``"A"`` / ``"coadd"`` / 31), as in the grid.
    inj_id_start : int, default 0
        First ``inj_id`` (so controls append to an injection manifest contiguously).

    Returns
    -------
    list[dict]
        Control manifest rows (full :data:`MANIFEST_FIELDS`, ``control=True``,
        NaN ``logN_true`` / ``z_true``).
    """
    snr_bins = np.asarray(snr_bins, dtype=float)
    if snr_bins.ndim != 1 or snr_bins.size < 2 or np.any(np.diff(snr_bins) <= 0):
        raise ValueError("snr_bins must be a strictly-increasing edge array of >=2.")
    n_snr = snr_bins.size - 1

    sl = _normalize_clean_sightlines(clean_sightlines)
    # Controls MUST be injection-free: drop any sightline carrying an injection (else
    # a "control" sits on an injected sightline → contaminated b_FP).
    if exclude_target_ids is not None:
        _excl = {int(t) for t in exclude_target_ids}
        sl = {t: v for t, v in sl.items() if int(t) not in _excl}
        if not sl:
            raise ValueError("no clean sightlines left for controls after excluding the "
                             "injected targets — widen the clean pool / healpix.")
    snr_table = {t: info["native_snr"] for t, info in sl.items()}
    all_tids = np.array(sorted(sl.keys()), dtype=np.int64)

    per_cell = _resolve_n_per_cell(n_snr, n_per_cell, target_controls)

    assign = sample_clean_sightlines(
        all_tids, snr_table, n_per_cell=per_cell, snr_bins=snr_bins, seed=int(seed),
    )
    rows: List[dict] = []
    inj_id = int(inj_id_start)
    for b in range(n_snr):
        for t in assign.get(b, ()):
            info = sl[int(t)]
            rows.append({
                "inj_id": inj_id,
                "campaign": str(campaign),
                "method": str(method),
                "target_id": int(t),
                "healpix": int(info["healpix"]),
                "z_qso": float(info["z_qso"]),
                "snr_bin": int(b),
                "native_snr": float(info["native_snr"]),
                "logN_true": float("nan"),   # NO injection — control sightline
                "z_true": float("nan"),
                "num_lines": int(num_lines),
                "control": True,
            })
            inj_id += 1
    if target_controls is not None and len(rows) > int(target_controls):
        rows = rows[: int(target_controls)]
        for i, r in enumerate(rows):
            r["inj_id"] = int(inj_id_start) + i
    return rows


# --------------------------------------------------------------------------- #
# Campaign B — close-pair grid
# --------------------------------------------------------------------------- #
_C_KMS = 299792.458  # km/s (matches window.C_KMS / set_parameters)


def build_close_pair_grid(
    clean_sightlines,
    *,
    logN_grid: Sequence[float],
    z_grid: Sequence[float],
    dv_kms_grid: Sequence[float],
    dlogN_grid: Sequence[float],
    snr_bins: Sequence[float],
    n_per_cell: Optional[int] = None,
    target_injections: Optional[int] = None,
    seed: int,
    method: str = "coadd",
    num_lines: int = DEFAULT_NUM_LINES,
) -> List[dict]:
    """Campaign-B close-pair grid: inject PAIRS at varying Δv and ΔN.

    Each row is a TWO-absorber injection (the blending / R-nonlinearity probe).
    The first absorber is at ``(logN_true, z_true)`` (as in Campaign A); the second
    is offset in velocity by ``dv_kms`` and in column by ``dlogN``::

        z_true2  = z_true + dv_kms / C_KMS * (1 + z_true)
        logN_true2 = logN_true + dlogN

    Rows carry the full :data:`MANIFEST_FIELDS` (with ``campaign="B"``) PLUS the
    :data:`CLOSE_PAIR_FIELDS` ``(logN_true2, z_true2, dv_kms)`` the CS injector
    passes to ``inject_multiple`` (two Voigt profiles multiplied into the flux).

    Parameters
    ----------
    clean_sightlines : dict-of-arrays or list-of-dicts
        CLEAN sightline table (see :func:`build_injection_grid`).
    logN_grid, z_grid : sequences
        First-absorber column-density and redshift grids.
    dv_kms_grid : sequence of float (> 0)
        Velocity separations (km/s) of the second absorber from the first.
    dlogN_grid : sequence of float
        Column-density offset of the second absorber (``logN_true2 - logN_true``).
    snr_bins, n_per_cell, target_injections, seed, method, num_lines
        As in :func:`build_injection_grid`.

    Returns
    -------
    list[dict]
        Close-pair manifest rows (contiguous ``inj_id``).  ``_dlogN`` is also
        stored for provenance (the offset that produced ``logN_true2``).
    """
    dv_kms_grid = np.asarray(dv_kms_grid, dtype=float)
    dlogN_grid = np.asarray(dlogN_grid, dtype=float)
    if np.any(dv_kms_grid <= 0):
        raise ValueError("dv_kms_grid entries must be > 0 (a close PAIR).")

    # Reuse the single-absorber builder for the FIRST absorber + sightline draw,
    # then expand each base row across the (dv, dlogN) pair grid.
    base = build_injection_grid(
        clean_sightlines,
        logN_grid=logN_grid,
        z_grid=z_grid,
        snr_bins=snr_bins,
        n_per_cell=n_per_cell,
        target_injections=None,  # cap applied AFTER pair expansion below
        seed=seed,
        campaign="B",
        method=method,
        num_lines=num_lines,
    )
    rows: List[dict] = []
    inj_id = 0
    for br in base:
        z1 = br["z_true"]
        lN1 = br["logN_true"]
        z_lo, z_hi = _per_sightline_forest_window(br["z_qso"])
        for dv in dv_kms_grid:
            z2 = z1 + float(dv) / _C_KMS * (1.0 + z1)
            # The second absorber must ALSO sit in the GP search window (M1).  We
            # DROP pairs whose z2 falls outside it rather than clamp — clamping
            # would silently falsify the dv separation the campaign measures.
            if not (z_lo <= z2 <= z_hi):
                continue
            for dlogN in dlogN_grid:
                lN2 = lN1 + float(dlogN)
                if not (LOGN_MIN - 1e-6 <= lN2 <= LOGN_MAX + 1e-6):
                    continue
                row = dict(br)
                row["inj_id"] = inj_id
                row["logN_true2"] = float(lN2)
                row["z_true2"] = float(z2)
                row["dv_kms"] = float(dv)
                row["_dlogN"] = float(dlogN)
                rows.append(row)
                inj_id += 1
    if target_injections is not None and len(rows) > int(target_injections):
        rows = rows[: int(target_injections)]
        # re-contiguous inj_id after the trim
        for i, r in enumerate(rows):
            r["inj_id"] = i
    return rows


# --------------------------------------------------------------------------- #
# manifest guard
# --------------------------------------------------------------------------- #
def validate_manifest(rows: Sequence[Mapping]) -> None:
    """Assert ``rows`` is a valid manifest (the CS-injector contract).

    For ordinary INJECTION rows: every required :data:`MANIFEST_FIELDS` key
    present; ``inj_id`` unique; ``campaign`` ∈ {'A','B','D'}; ``method`` ∈
    {'coadd','gpdraw'}; ``LOGN_MIN <= logN_true <= LOGN_MAX``; and the z-window
    bounds (M1) ``z_lo <= z_true <= z_hi`` AND ``z_true < z_qso`` (absorber inside
    the GP search window and blueward of the QSO).

    CONTROL rows (``control=True``, M2) carry ``logN_true = NaN`` /
    ``z_true = NaN`` by design — they are EXEMPT from the logN-range and z-window
    checks (a clean no-injection sightline has no truth absorber).  A NaN
    ``logN_true`` WITHOUT the control flag is a corrupt injection row → rejected.

    Close-pair rows (Campaign B) may additionally carry :data:`CLOSE_PAIR_FIELDS`;
    when present they are sanity-checked (``logN_true2`` in range, ``dv_kms > 0``,
    ``z_lo <= z_true2 <= z_hi`` and ``z_true2 < z_qso``).
    """
    seen_ids = set()
    for i, r in enumerate(rows):
        for k in MANIFEST_FIELDS:
            if k not in r:
                raise KeyError(f"manifest row {i} missing required field {k!r}")
        iid = r["inj_id"]
        if iid in seen_ids:
            raise ValueError(f"duplicate inj_id {iid!r} in manifest")
        seen_ids.add(iid)
        if r["campaign"] not in ("A", "B", "D"):
            raise ValueError(f"row {i}: campaign {r['campaign']!r} not in A/B/D")
        if r["method"] not in ("coadd", "gpdraw"):
            raise ValueError(f"row {i}: method {r['method']!r} not in coadd/gpdraw")

        is_control = bool(r.get("control", False))
        logN_finite = np.isfinite(r["logN_true"])

        # M2: control rows legitimately carry NaN logN/z — exempt them from the
        # injection-truth checks.  A NaN logN on a NON-control row is corrupt.
        if is_control:
            continue
        if not logN_finite:
            raise ValueError(
                f"row {i}: logN_true is NaN but control flag is not set "
                f"(corrupt injection row)"
            )

        # M1: z_true must sit inside the GP search window AND blueward of z_qso.
        z_lo, z_hi = _per_sightline_forest_window(float(r["z_qso"]))
        if not (z_lo - 1e-9 <= r["z_true"] <= z_hi + 1e-9):
            raise ValueError(
                f"row {i}: z_true {r['z_true']} outside GP search window "
                f"[{z_lo}, {z_hi}] for z_qso {r['z_qso']}"
            )
        if not (r["z_true"] < r["z_qso"]):
            raise ValueError(
                f"row {i}: z_true {r['z_true']} not blueward of z_qso {r['z_qso']}"
            )
        if not (LOGN_MIN - 1e-6 <= r["logN_true"] <= LOGN_MAX + 1e-6):
            raise ValueError(
                f"row {i}: logN_true {r['logN_true']} outside [{LOGN_MIN}, {LOGN_MAX}]"
            )
        # Optional close-pair sanity (Campaign B second absorber).
        if "logN_true2" in r and r["logN_true2"] is not None:
            if not (LOGN_MIN - 1e-6 <= r["logN_true2"] <= LOGN_MAX + 1e-6):
                raise ValueError(f"row {i}: logN_true2 out of range")
            if "dv_kms" in r and r["dv_kms"] is not None and not (r["dv_kms"] > 0):
                raise ValueError(f"row {i}: dv_kms must be > 0 for a close pair")
            if "z_true2" in r and r["z_true2"] is not None:
                if not (z_lo - 1e-9 <= r["z_true2"] <= z_hi + 1e-9):
                    raise ValueError(
                        f"row {i}: z_true2 {r['z_true2']} outside GP search window "
                        f"[{z_lo}, {z_hi}]"
                    )
                if not (r["z_true2"] < r["z_qso"]):
                    raise ValueError(
                        f"row {i}: z_true2 {r['z_true2']} not blueward of z_qso"
                    )
