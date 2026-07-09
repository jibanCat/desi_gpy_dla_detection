"""survival.py — model-free LLS incidence via a Nelson-Aalen / exposure (g(z)) estimator.

WHAT THIS MEASURES
------------------
Lyman-limit systems (LLS) are counted by their 912 Angstrom continuum BREAK, not by the
Lyman-alpha line. A break counter, scanning a sightline from the QSO downward in redshift,
sees only the FIRST (highest-z) tau>=2 system: everything blueward of that break is opaque
(this is "blocking"). We want ell(z) -- the incidence of tau>=2 systems per unit redshift
(and per unit absorption distance dX) -- from these blocked, first-break-only observations.

THE ESTIMATOR (why blocking does NOT bias the incidence)
-------------------------------------------------------
This is textbook right-censored survival analysis with "time" running from the QSO (high z)
downward toward the blue cutoff (low z). A sightline is AT RISK at redshift z if it is
break-observable at z AND has not yet had a detected break at any z' > z. The moment a break
is detected at z_detect the sightline leaves the risk set (blocked); a sightline that reaches
the blue cutoff with no break is right-censored there.

The Aalen (exposure-based Nelson-Aalen) hazard estimator in a redshift bin b is

        ell_hat(b) = D_b / R_b ,     R_b = INT_b n_atrisk(z) dz   [person-redshift],

where D_b = number of detected first-breaks with z_detect in bin b, and R_b is the OBSERVED
at-risk exposure -- the total searchable path in z accrued by sightlines while they are at
risk (i.e. from each sightline's high-z window edge DOWN TO its detected break, or down to the
blue cutoff if it survived). Because the denominator counts exposure only where a sightline is
actually searchable, a higher-z break -- which removes a sightline from the risk set below it
-- reduces the numerator and the denominator IN LOCKSTEP. Under independent absorbers (a
Poisson point process along the sightline, so a high-z break carries no information about the
lower-z incidence: "independent censoring"), D_b/R_b is an unbiased estimator of the intensity
ell(z). This is exactly why blocking does not bias the incidence, and why this plug-in
estimator recovers the SAME ell(z) as directly counting every tau>=2 absorber over the full
(un-blocked) observable window -- the self-consistency check in break_census.py.

CLUSTERING IS A CARRIED SYSTEMATIC (the independent-censoring assumption's one real crack).
Independent censoring holds for a POISSON absorber field. Real LLS trace haloes (bias ~2), and
positive line-of-sight clustering biases this estimator LOW by approximately

        ell_NA / ell_true  ~  1 / (1 + ell * INT xi(r) dr)

(sign and magnitude validated against a Cox-process simulation to <10% across three clustering
amplitudes). On the 2LPT-0 mock the measured LOS correlation is INT xi dr ~ 1e-4, so the induced
bias is ~0.02% -- i.e. the mock's injected, non-hydro HCDs are effectively Poisson and CANNOT
bound this term. Do not read the mock's direct-vs-NA agreement as validating away real-data
clustering: carry it as a systematic (plausibly ~1-few % for real LLS).

NOT a model-based g_i. A "P_clear(z; Lambda)" clear-fraction estimator that plugs a MODEL of
ell back into the risk set is a fixed-point equation, not a plug-in estimator, and is biased
under misspecification (-15%/+18% when Lambda is scaled by 0.6/1.5). This module deliberately
implements only the model-free version: R_b is the OBSERVED exposure, never a modelled clear
probability.

UNITS
-----
ell_hat is returned PER UNIT REDSHIFT (dN/dz). The cosmology-independent measurement is dN/dz;
the population quantity usually quoted is dN/dX (per unit absorption distance). Convert with
`ell_per_dz_to_dX`, which uses the repo's single absorption-distance convention
`CDDF_analysis.cddf_mock.path_length_int` (dX/dz = (1+z)^2 / E(z), Omega_m=0.279 by default) --
the SAME cosmology as the counting channel. We do NOT introduce a second cosmology here.
(Note: `CDDF_analysis.lyc.opacity.Cosmology` defaults to Om=0.3 and is used for the *proper
distance* lambda_mfp inversion in the drop channel; for dX we intentionally use path_length_int
so counting and incidence share Omega_m=0.279. Pass Omega_m explicitly to align if needed.)

ERRORS
------
Sightline bootstrap: resample SIGHTLINES with replacement (not absorbers), so correlated
multi-absorber sightlines and the exposure/count coupling are propagated correctly. Reported
`ell_err` is the bootstrap standard deviation per bin.

Physics anchors: Nelson (1972) / Aalen (1978) hazard estimator; Prochaska+ 2010 (0912.0292),
Worseck+ 2014 (1402.4154) for the LLS break-counting geometry.
"""
from __future__ import annotations

import sys

import numpy as np

from .opacity import LYMAN_LIMIT, C_KMS

__all__ = [
    "blue_cutoff_z",
    "proximity_z_max",
    "build_break_census",
    "ell_nelson_aalen",
    "ell_direct_incidence",
    "ell_per_dz_to_dX",
]


# ---------------------------------------------------------------------------
# Geometry helpers (the searchable window per sightline)
# ---------------------------------------------------------------------------
def blue_cutoff_z(wave_obs_min: float = 3600.0, lyman_limit: float = LYMAN_LIMIT) -> float:
    """Lowest absorber redshift whose 912 A break lands ABOVE the instrument's blue cutoff.

    A break at z_abs sits at observed wavelength 912*(1+z_abs); it is only observable when that
    exceeds wave_obs_min. Returns z_cut = wave_obs_min/lyman_limit - 1 (DESI: 3600 A -> 2.948).
    """
    return float(wave_obs_min) / float(lyman_limit) - 1.0


def proximity_z_max(z_qso, dv_kms: float = 3000.0):
    """High-z edge of the searchable window: z_qso minus a proximity velocity zone.

    Absorbers within dv_kms of the QSO are excluded (proximity effect / associated systems).
    Uses the standard low-order velocity-redshift relation dv = c * (z_qso - z)/(1+z_qso), so
    z_max = z_qso - dv_kms*(1+z_qso)/c. Vectorized over z_qso.
    """
    z_qso = np.asarray(z_qso, float)
    return z_qso - float(dv_kms) * (1.0 + z_qso) / C_KMS


# ---------------------------------------------------------------------------
# Blocking + windowing: turn per-absorber truth into a per-sightline first-break census
# ---------------------------------------------------------------------------
def build_break_census(sl_ids, z_abs, z_qso_map, cutoff, proximity_dv_kms: float = 3000.0):
    """Reduce a per-absorber catalog to the per-sightline BLOCKED first-break census.

    Parameters
    ----------
    sl_ids : array-like
        Sightline id for every tau>=2 absorber (parallel to z_abs). Only sightlines that
        appear in `z_qso_map` are considered; the returned arrays are one row per UNIQUE
        sightline present in z_qso_map (so exposure is counted even for sightlines with no
        observable absorber).
    z_abs : array-like
        Absorber redshifts (parallel to sl_ids).
    z_qso_map : dict {sl_id: z_qso}
        QSO redshift per sightline. Defines the searchable sample and each window's high edge.
    cutoff : float
        Blue-cutoff redshift (low edge of every searchable window); see `blue_cutoff_z`.
    proximity_dv_kms : float
        Proximity velocity zone excluded below the QSO (see `proximity_z_max`).

    Returns
    -------
    dict with per-sightline arrays (indexed by the sorted unique sightlines of z_qso_map):
        sl        : sightline ids (sorted)
        z_qso     : QSO redshift
        z_stop    : high-z searchable edge = proximity_z_max(z_qso)
        z_cut     : low-z searchable edge  = cutoff (same for all)
        z_detect  : highest-z OBSERVABLE tau>=2 absorber (NaN if none) -- the counted break
        z_start   : at-risk low edge = z_detect if a break, else cutoff
        has_break : bool mask
        observable: bool mask, z_stop > cutoff (a non-empty searchable window exists)
    Also returns the per-absorber `obs_mask` (which input absorbers are break-observable) so a
    caller can build the un-blocked direct-incidence input from the SAME selection.
    """
    sl_ids = np.asarray(sl_ids)
    z_abs = np.asarray(z_abs, float)

    sl_sorted = np.array(sorted(z_qso_map.keys()))
    n = sl_sorted.size
    idx_of = {s: i for i, s in enumerate(sl_sorted.tolist())}
    z_qso = np.array([z_qso_map[s] for s in sl_sorted.tolist()], float)
    z_stop = proximity_z_max(z_qso, proximity_dv_kms)
    z_cut = np.full(n, float(cutoff))
    observable = z_stop > z_cut

    z_detect = np.full(n, np.nan)
    # per-absorber observability: within [cutoff, z_stop_of_its_sightline]
    obs_mask = np.zeros(z_abs.size, dtype=bool)
    for k in range(z_abs.size):
        s = sl_ids[k].item() if hasattr(sl_ids[k], "item") else sl_ids[k]
        j = idx_of.get(s, None)
        if j is None:
            continue
        if (z_abs[k] > cutoff) and (z_abs[k] < z_stop[j]):
            obs_mask[k] = True
            if np.isnan(z_detect[j]) or z_abs[k] > z_detect[j]:
                z_detect[j] = z_abs[k]

    has_break = np.isfinite(z_detect)
    z_start = np.where(has_break, z_detect, z_cut)
    return dict(sl=sl_sorted, z_qso=z_qso, z_stop=z_stop, z_cut=z_cut,
                z_detect=z_detect, z_start=z_start, has_break=has_break,
                observable=observable, obs_mask=obs_mask)


# ---------------------------------------------------------------------------
# Core binning primitives (shared by the Nelson-Aalen and direct estimators)
# ---------------------------------------------------------------------------
def _exposure_matrix(z_start, z_stop, edges):
    """(n_sl, n_bin) matrix: person-redshift each sightline contributes to each bin.

    Overlap of [z_start_i, z_stop_i] with each [edges[b], edges[b+1]]. Empty windows
    (z_start >= z_stop) contribute 0; fully robust to out-of-range windows.
    """
    z_start = np.asarray(z_start, float)
    z_stop = np.asarray(z_stop, float)
    edges = np.asarray(edges, float)
    lo = edges[:-1][None, :]
    hi = edges[1:][None, :]
    a = np.maximum(z_start[:, None], lo)
    b = np.minimum(z_stop[:, None], hi)
    return np.clip(b - a, 0.0, None)


def _count_matrix(z_events, event_sl_row, n_sl, edges):
    """(n_sl, n_bin) integer matrix: events per sightline per bin.

    z_events : event redshifts (any number per sightline). event_sl_row : the ROW index
    (0..n_sl-1) each event belongs to. Events outside [edges[0], edges[-1]] are dropped.
    """
    edges = np.asarray(edges, float)
    C = np.zeros((n_sl, edges.size - 1), float)
    z_events = np.asarray(z_events, float)
    event_sl_row = np.asarray(event_sl_row, int)
    finite = np.isfinite(z_events)
    z_events = z_events[finite]
    event_sl_row = event_sl_row[finite]
    if z_events.size:
        # np.digitize: bin index in [1, n_bin]; 0 or n_bin+? => out of range
        bidx = np.digitize(z_events, edges) - 1
        inside = (bidx >= 0) & (bidx < edges.size - 1)
        np.add.at(C, (event_sl_row[inside], bidx[inside]), 1.0)
    return C


def _estimate_from_matrices(C, E, edges, n_boot, seed):
    """ell = colsum(C)/colsum(E) per bin (per unit z); sightline-bootstrap std.

    C, E : (n_sl, n_bin) count and exposure matrices. Bins with zero total exposure -> NaN.
    """
    edges = np.asarray(edges, float)
    n_sl = C.shape[0]
    n_det = C.sum(axis=0)
    exposure = E.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ell = np.where(exposure > 0, n_det / exposure, np.nan)

    ell_err = np.full(edges.size - 1, np.nan)
    if n_boot and n_sl > 0:
        rng = np.random.default_rng(seed)
        boot = np.empty((n_boot, edges.size - 1))
        for t in range(n_boot):
            pick = rng.integers(0, n_sl, size=n_sl)
            cb = C[pick].sum(axis=0)
            eb = E[pick].sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                boot[t] = np.where(eb > 0, cb / eb, np.nan)
        with np.errstate(invalid="ignore"):
            ell_err = np.nanstd(boot, axis=0)

    dz = np.diff(edges)
    return dict(
        z_mid=0.5 * (edges[:-1] + edges[1:]),
        z_lo=edges[:-1], z_hi=edges[1:],
        ell=ell, ell_err=ell_err,
        n_det=n_det.astype(int),
        exposure=exposure,                       # R(z_bin): person-redshift denominator
        n_risk=np.where(dz > 0, exposure / dz, np.nan),  # mean # sightlines at risk over bin
        n_sl_risk=(E > 0).sum(axis=0).astype(int),       # integer # sightlines touching bin
    )


# ---------------------------------------------------------------------------
# Primary estimator: Nelson-Aalen on the BLOCKED first-break census
# ---------------------------------------------------------------------------
def ell_nelson_aalen(z_detect, z_start, z_stop, zbins, n_boot: int = 1000, seed: int = 0):
    """Model-free incidence ell(z) [per unit z] from first-break observations (see module doc).

    Parameters
    ----------
    z_detect : array-like, one per sightline
        Redshift of the counted (highest-z) break, or NaN if the sightline has no detected break.
    z_start, z_stop : array-like, one per sightline
        At-risk window: z_start = z_detect (if a break) or the blue cutoff (if censored);
        z_stop = proximity-excluded high-z edge. Empty windows (z_start >= z_stop) are allowed
        and contribute zero exposure.
    zbins : array-like
        Bin EDGES (monotone increasing), length n_bin+1.
    n_boot : int
        Sightline-bootstrap draws for `ell_err` (0 disables -> ell_err all NaN).
    seed : int
        RNG seed for reproducible bootstrap.

    Returns
    -------
    dict(z_mid, z_lo, z_hi, ell, ell_err, n_det, exposure, n_risk, n_sl_risk).
    `ell` and `ell_err` are per unit redshift. `exposure` is the R(z_bin) person-z denominator.
    """
    z_detect = np.asarray(z_detect, float)
    z_start = np.asarray(z_start, float)
    z_stop = np.asarray(z_stop, float)
    edges = np.asarray(zbins, float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("zbins must be monotone-increasing edges of length >= 2")
    n_sl = z_start.size
    if not (z_detect.size == z_start.size == z_stop.size):
        raise ValueError("z_detect, z_start, z_stop must be the same length (one per sightline)")

    # Events outside their own at-risk window are inconsistent -> drop + warn. This is a
    # SILENT-DEGRADATION footgun for callers who thread a different proximity/cutoff convention
    # into z_detect than into z_start/z_stop: a mismatched convention drops real detections and
    # biases ell LOW (~10% in a review stress-test) with only a stderr line. `n_dropped` is
    # returned so callers can `assert out["n_dropped"] == 0`.
    with np.errstate(invalid="ignore"):
        bad = np.isfinite(z_detect) & ((z_detect < z_start - 1e-9) | (z_detect > z_stop + 1e-9))
    n_dropped = int(bad.sum())
    if n_dropped:
        print(f"  [survival WARN] {n_dropped} detection(s) outside their at-risk window "
              f"were dropped (proximity/window mismatch).", file=sys.stderr)
        z_detect = z_detect.copy()
        z_detect[bad] = np.nan

    E = _exposure_matrix(z_start, z_stop, edges)
    C = _count_matrix(z_detect, np.arange(n_sl), n_sl, edges)
    out = _estimate_from_matrices(C, E, edges, n_boot, seed)
    out["n_dropped"] = n_dropped
    out["estimator"] = "nelson_aalen_first_break"
    return out


# ---------------------------------------------------------------------------
# Direct incidence from ALL tau>=2 absorbers over the FULL observable window
# (the un-blocked truth cross-check; must match ell_nelson_aalen on the same geometry)
# ---------------------------------------------------------------------------
def ell_direct_incidence(z_abs, abs_sl_row, z_start_full, z_stop_full, zbins,
                         n_boot: int = 1000, seed: int = 0):
    """Direct incidence ell(z) [per unit z] from EVERY observable tau>=2 absorber (no blocking).

    Parameters
    ----------
    z_abs : array-like
        Redshifts of ALL observable tau>=2 absorbers (multiple per sightline allowed).
    abs_sl_row : array-like
        Row index (0..n_sl-1) of each absorber's sightline (parallel to z_abs).
    z_start_full, z_stop_full : array-like, one per sightline
        FULL observable window [blue cutoff, proximity edge] -- NOT truncated at any break,
        because there is no blocking in the direct count.
    zbins : array-like
        Bin edges (as `ell_nelson_aalen`).

    Returns the same dict shape as `ell_nelson_aalen`. On an independent-absorber population
    this equals `ell_nelson_aalen` bin-by-bin (the Nelson-Aalen self-consistency check).
    """
    edges = np.asarray(zbins, float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("zbins must be monotone-increasing edges of length >= 2")
    n_sl = np.asarray(z_start_full, float).size
    E = _exposure_matrix(z_start_full, z_stop_full, edges)
    C = _count_matrix(z_abs, abs_sl_row, n_sl, edges)
    out = _estimate_from_matrices(C, E, edges, n_boot, seed)
    out["estimator"] = "direct_all_absorbers"
    return out


# ---------------------------------------------------------------------------
# Units: per dz  ->  per dX (repo's single absorption-distance cosmology)
# ---------------------------------------------------------------------------
def ell_per_dz_to_dX(ell_dz, z, Omega_m: float = 0.279):
    """Convert an incidence per unit redshift to per unit absorption distance dX.

    dN = ell_dz dz = ell_dX dX with dX/dz = (1+z)^2 / E(z) (path_length_int), so
    ell_dX = ell_dz / (dX/dz). Uses CDDF_analysis.cddf_mock.path_length_int -- the ONE
    absorption-distance convention shared with the counting channel (Omega_m=0.279 default).
    """
    from CDDF_analysis.cddf_mock import path_length_int  # local import: keep lyc core light
    dXdz = path_length_int(np.asarray(z, float), Omega_m=Omega_m)
    return np.asarray(ell_dz, float) / dXdz
