"""
examples/inject/measurements.py
===============================
M3 injection-campaign MEASUREMENT estimators (the Bayesian-modeling owner's
scope).  Three estimators operate on the RECOVERED processed-HDF5 readout
(produced by the streaming deposit machinery) keyed by the injection MANIFEST:

1. :func:`detection_completeness` — ``C_det(logN_true, z, SNR)``: the fraction of
   injected absorbers the GP recovers with ``p_dla > thresh``, per cell, with a
   conjugate **Beta-Binomial** credible interval (Jeffreys prior).  This is the
   genuine SNR-resolved detectability, free of the diagonal N-scatter artifact.
2. :func:`nhi_bias` — ``b_N(logN_true, SNR) = <logN_rec> - logN_true`` and the
   scatter ``σ_N``, conditioned on RECOVERY.  The **headline NHI<19 bias** (the
   prior-pull toward the PW14 mode where the likelihood is faint).
3. :func:`response_matrix` — ``R[(N_rec, z_rec), (N_true, z_true, SNR)]``: the
   off-diagonal kernel built from the per-injection posterior DEPOSIT (the same
   ``p_dla`` mass ``DiagonalSoftDeposit`` routes), column-normalized; plus
   ``b_FP`` from no-injection CONTROL cells (a **Poisson-rate Gamma** posterior).
   Feeds the M4 forward model ``E[n_rec] = R · n_true + b_FP``.

Recovered-record schema (the seam the CS deposit fills — one per manifest row)::

    {inj_id:int,
     p_dla:float,                      # SINGLE-ABSORBER posterior P(absorber) =
                                       #   model_posteriors[:, 1] (M3, NOT a DLA-only
                                       #   p_DLA): a true LLS recovered as a low-N
                                       #   absorber is COUNTED here, not zeroed.
     logN_rec:float, z_rec:float,      # recovered MAP / posterior-mean (NaN if none)
     deposit:[(logN, z, weight), ...]} # per-(N,z) posterior probability mass

The campaign topology is PINNED in :data:`CAMPAIGN_TOPOLOGY` (single-absorber
FILTER-off, ``sub_dla=False``); see its comment for why a ``sub_dla=True`` run
would zero-out the NHI<19 recoveries this campaign is built to measure.

``deposit`` is the streaming-machinery output: a list of posterior probability
mass deposits in (logN, z) space — exactly the ``p_dla`` mass the production
``DiagonalSoftDeposit._iter_sightline_deposits`` yields per surviving sample.

Bayesian assumptions (stated honestly)
--------------------------------------
* **Completeness** — binomial detection per cell: each injected absorber is an
  independent Bernoulli(C) trial (recovered iff ``p_dla > thresh``).  Conjugate
  ``Beta(k + a, n - k + b)`` posterior, Jeffreys ``(0.5, 0.5)`` default; point =
  posterior MEAN, intervals = equal-tailed Beta quantiles.  A missing recovered
  record counts as a non-detection (never dropped from the denominator).
* **R columns** — the DEPOSIT-MEAN response: column ``j`` (a true (N,z,SNR) cell)
  is the posterior probability mass deposited into each recovered (N_rec,z_rec)
  cell, summed over that cell's injections and (when ``normalize``) divided by the
  injected count so the column is a per-true-absorber recovered-mass distribution.
  Off-diagonal entries are genuine migration (sub-DLA↔DLA across 20.3, z-scatter).
  A column's sum ≤ 1 (un-normalized: ≤ ``n_true``); the deficit is incompleteness.
* **b_FP** — Poisson rate of spurious deposits in the no-injection control cells:
  conjugate ``Gamma(mass + a, scale = 1)`` deposit posterior (matching the O3
  ``soft_completeness.estimate_false_positive_deposit`` convention exactly), point
  = posterior MODE ``max(mass + a - 1, 0)``, CI = equal-tailed Gamma quantiles
  clamped to bracket the mode.  A COUNT in the same units as the R deposit.

These reuse the O3 Bayesian core (``CDDF_analysis.cddf_forward.soft_completeness``)
for the Beta/Gamma posteriors so the M3 measurement layer is numerically
consistent with the diagonal-correction layer the M4 deconvolution feeds.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as _beta_dist

# Reuse the O3 Bayesian core's FP-deposit estimator so b_FP is the SAME
# Poisson-rate Gamma posterior the diagonal-correction layer uses.
from CDDF_analysis.cddf_forward.soft_completeness import (
    estimate_false_positive_deposit as _estimate_fp_deposit,
)

# Jeffreys Beta prior for the binomial completeness (matches the O3 core default).
_BETA_PRIOR = (0.5, 0.5)

# --------------------------------------------------------------------------- #
# M3 — CAMPAIGN TOPOLOGY (pinned + LOUD).  ⚠️ READ THIS BEFORE RUNNING. ⚠️
# --------------------------------------------------------------------------- #
# The M3 injection campaign MUST run the SINGLE-ABSORBER FILTER-off model — the
# exact topology of the 2LPT-0 FILTER-off V1 production run being corrected:
#
#   * ``sub_dla = False`` — ONE absorber model, NO separate sub-DLA leg.  Under
#     ``sub_dla = True`` a true LLS recovered as a sub-DLA deposits ZERO DLA-mass
#     into ``p_DLA``, so ``C_det(NHI < 19)`` collapses to a spurious ~0 artifact.
#   * The recovery readout reads ``model_posteriors[:, 1]`` = p(absorber) — the
#     SINGLE absorber posterior — NOT a DLA-only ``p_DLA``.  The recovered-record
#     ``p_dla`` field IS this col-1 absorber posterior; a true LLS recovered as a
#     LOW-N absorber (e.g. recovered logN ≈ 18) is therefore COUNTED as recovered,
#     never zeroed.  (For ``single_absorber_model=True`` the layout is
#     ``[:, 0] = Null, [:, 1] = absorber`` — see CLAUDE.md §11.)
#   * The QMC sample grid spans log N_HI ∈ [17.2, 22.5] (the full LLS→DLA range);
#     20.3 is an INTERIOR point, so sub-DLA↔DLA migration across 20.3 is genuine
#     off-diagonal R mass, not an edge effect.
#
# If a future run uses ``sub_dla=True`` (the two-leg topology), the LLS recoveries
# would be routed to the sub-DLA posterior and these estimators (which read the
# absorber col) would under-count them — the campaign would mis-measure exactly
# the NHI<19 regime it targets.  ``CAMPAIGN_TOPOLOGY`` pins the requirement so a
# topology mismatch is a loud, testable assertion rather than a silent ~0 artifact.
CAMPAIGN_TOPOLOGY = {
    "sub_dla": False,                 # single-absorber model (NOT the 2-leg topology)
    "filter": "off",                  # FILTER-off (matches the 2LPT-0 V1 run)
    "absorber_posterior_col": 1,      # model_posteriors[:, 1] = p(absorber)
    "logn_grid_min": 17.2,            # full QMC sample-grid log N_HI range ...
    "logn_grid_max": 22.5,            #   ... LLS→DLA, 20.3 an interior point
    "note": (
        "Read model_posteriors[:, 1] (p(absorber)), NOT a DLA-only p_DLA.  A true "
        "LLS recovered as a low-N absorber must be COUNTED, not zeroed.  A "
        "sub_dla=True run would zero-out LLS recoveries -> do NOT use it."
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bin_index(value: float, edges: np.ndarray) -> int:
    """Index ``i`` with ``edges[i] <= value < edges[i+1]`` (-1 if outside)."""
    if not np.isfinite(value):
        return -1
    i = int(np.searchsorted(edges, value, side="right") - 1)
    if 0 <= i < edges.size - 1:
        return i
    return -1


def _cell_key_NzSNR(row: Mapping) -> Tuple[float, float, int]:
    """The (logN_true, z_true, snr_bin) coordinate of a manifest row."""
    return (float(row["logN_true"]), float(row["z_true"]), int(row["snr_bin"]))


def _is_control(row: Mapping) -> bool:
    """A control (no-injection) manifest row: logN_true is NaN."""
    return not np.isfinite(row.get("logN_true", np.nan))


# --------------------------------------------------------------------------- #
# 1. detection completeness
# --------------------------------------------------------------------------- #
def detection_completeness(
    recovered: Mapping[int, Mapping],
    manifest: Sequence[Mapping],
    *,
    p_dla_thresh: float = 0.5,
    prior: Tuple[float, float] = _BETA_PRIOR,
) -> Dict[str, np.ndarray]:
    """Per-cell detection completeness ``C_det(logN_true, z, SNR)`` (Beta-Binomial).

    An injected absorber is RECOVERED iff its recovered ``p_dla`` STRICTLY exceeds
    ``p_dla_thresh``.  A manifest row whose ``inj_id`` is absent from ``recovered``
    (GP produced no output) counts as a non-detection — it stays in the denominator
    (never silently dropped), so completeness is not biased high by lost spectra.

    Parameters
    ----------
    recovered : mapping {inj_id -> recovered-record}
        Recovered-record schema (see module docstring); needs at least ``p_dla``.
    manifest : sequence of manifest rows
        The injection manifest (one row per injection).  Control rows
        (``logN_true`` NaN) are EXCLUDED (they have no truth to recover).
    p_dla_thresh : float, default 0.5
        Operating threshold on ``p_dla`` for a "detection".
    prior : (a, b), default (0.5, 0.5)
        Beta prior shape parameters (Jeffreys).

    Returns
    -------
    dict
        ``logN_true, z_true, snr_bin`` (cell coordinates),
        ``n_injected, n_recovered`` (integer per cell),
        ``C, C_lo68, C_hi68, C_lo95, C_hi95`` (Beta posterior mean + equal-tailed
        intervals).  One entry per occupied (logN_true, z_true, snr_bin) cell, in
        sorted coordinate order.

    Model
    -----
    ``k_b`` recoveries out of ``n_b`` injections in cell ``b`` →
    ``C_b ~ Beta(k_b + a, n_b - k_b + b)`` (conjugate Beta-Binomial).  The reported
    POINT ``C`` is the BINOMIAL MLE ``k/n`` (the directly-interpretable recovered
    fraction the campaign measures); the CREDIBLE INTERVAL is the equal-tailed
    Jeffreys-Beta posterior quantile (so a clean cell ``k=0`` / a full cell ``k=n``
    still carries an honest one-sided uncertainty instead of collapsing to a point).
    Binomial because each injection is an independent Bernoulli(C) detection trial.
    For ``n=0`` cells (none in practice) the MLE is undefined → reported as NaN.
    """
    a, b = float(prior[0]), float(prior[1])
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta prior shapes must be > 0, got {prior!r}")

    # Tally injections + recoveries per (logN, z, SNR) cell.
    n_inj: Dict[Tuple[float, float, int], int] = {}
    n_rec: Dict[Tuple[float, float, int], int] = {}
    for row in manifest:
        if _is_control(row):
            continue
        key = _cell_key_NzSNR(row)
        n_inj[key] = n_inj.get(key, 0) + 1
        recd = recovered.get(row["inj_id"])
        detected = (
            recd is not None
            and np.isfinite(recd.get("p_dla", np.nan))
            and float(recd["p_dla"]) > p_dla_thresh
        )
        n_rec[key] = n_rec.get(key, 0) + (1 if detected else 0)

    keys = sorted(n_inj.keys())
    logN = np.array([k[0] for k in keys], dtype=float)
    z = np.array([k[1] for k in keys], dtype=float)
    snr = np.array([k[2] for k in keys], dtype=int)
    n = np.array([n_inj[k] for k in keys], dtype=int)
    k = np.array([n_rec[k] for k in keys], dtype=int)

    alpha = k + a
    beta_p = n - k + b
    # POINT: binomial MLE k/n (the directly-interpretable recovered fraction);
    # NaN where n==0.  INTERVAL: equal-tailed Jeffreys-Beta posterior quantiles.
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    C_lo68 = _beta_dist.ppf(0.16, alpha, beta_p)
    C_hi68 = _beta_dist.ppf(0.84, alpha, beta_p)
    C_lo95 = _beta_dist.ppf(0.025, alpha, beta_p)
    C_hi95 = _beta_dist.ppf(0.975, alpha, beta_p)

    return {
        "logN_true": logN,
        "z_true": z,
        "snr_bin": snr,
        "n_injected": n,
        "n_recovered": k,
        "C": C,
        "C_lo68": C_lo68,
        "C_hi68": C_hi68,
        "C_lo95": C_lo95,
        "C_hi95": C_hi95,
    }


# --------------------------------------------------------------------------- #
# 2. N_HI bias
# --------------------------------------------------------------------------- #
def nhi_bias(
    recovered: Mapping[int, Mapping],
    manifest: Sequence[Mapping],
    *,
    p_dla_thresh: float = 0.5,
    n_min: int = 20,
) -> Dict[str, np.ndarray]:
    """Per-cell N_HI bias ``b_N(logN_true, SNR)`` + scatter ``σ_N`` (headline NHI<19).

    For every RECOVERED injected absorber (``p_dla > thresh`` and finite
    ``logN_rec``) the residual is ``logN_rec - logN_true``.  Per (logN_true, SNR)
    cell (z is MARGINALIZED — the bias is reported over logN×SNR per the design):

    * ``b_N``     = recovery-conditioned mean residual (positive = recovered above
                    truth, the prior-pull toward the PW14 mode expected for NHI<19);
    * ``σ_N``     = population std of the recovered residual (ddof=0);
    * ``b_N_se``  = standard error of the mean ``σ_N / sqrt(n_used)``.

    M5 — recovery-conditioning / Malmquist (the review finding)
    -----------------------------------------------------------
    ``b_N`` is conditioned on RECOVERY: at few-% LLS completeness the survivors are
    the UPWARD-scattered tail, so ``<logN_rec>`` over survivors UNDER-STATES the
    population bias.  This estimator therefore ALSO returns:

    * ``logN_rec_dist`` — the FULL per-cell recovered-N distribution (the array of
      recovered ``logN_rec`` over survivors), so M4 can use the whole distribution
      rather than only its conditioned mean;
    * ``b_N_deposit``   — the DEPOSIT-WEIGHTED bias over ALL injections (recovered
      AND not): ``Σ_i p_i·(logN_rec_i − logN_true_i) / n_injected``, where ``p_i``
      is the recovered posterior deposit mass (non-recoveries carry ~0 mass).
      Dividing by ``n_injected`` (not the recovered count) lets incompleteness drag
      the bias toward 0 — the population-level number M4 should use;
    * ``bias_unreliable`` — a boolean flag, True where ``n_used < n_min`` (default
      20): the recovery-conditioned mean of a near-empty cell is not trustworthy.

    Parameters
    ----------
    recovered : mapping {inj_id -> recovered-record}
        Needs ``p_dla`` and ``logN_rec`` (the recovered MAP or posterior-mean N_HI).
    manifest : sequence of manifest rows
        Control rows (``logN_true`` NaN) are excluded.
    p_dla_thresh : float, default 0.5
        Recovery threshold; non-detections carry no meaningful ``logN_rec`` and are
        excluded from the recovery-conditioned average (but still counted in the
        deposit-weighted ``b_N_deposit`` denominator).
    n_min : int, default 20
        Cells with fewer than ``n_min`` recovered absorbers are flagged
        ``bias_unreliable`` (the recovery-conditioned mean is Malmquist-biased and
        statistically thin there).

    Returns
    -------
    dict
        ``logN_true, snr_bin`` (cell coords), ``n_used`` (recovered count),
        ``n_injected`` (all injections in the cell), ``b_N, sigma_N, b_N_se``
        (recovery-conditioned), ``b_N_deposit`` (deposit-weighted over all
        injections), ``logN_rec_dist`` (per-cell recovered-N arrays), and
        ``bias_unreliable`` (n_used < n_min).  One entry per occupied (logN_true,
        snr_bin) cell with ``n_used >= 1``, in sorted order.

    Assumption
    ----------
    The residual distribution is summarized by its first two moments (mean + std);
    no Gaussianity is assumed for the point/scatter, but ``b_N_se`` is the usual
    ``σ/√n`` Gaussian-CLT standard error of the mean (valid for n not tiny).
    """
    # Recovered residuals (survivors) per cell + the full recovered-N list, and
    # the deposit-weighted numerator / injection-count denominator over ALL
    # injections in the cell (M5).
    resid_by_cell: Dict[Tuple[float, int], List[float]] = {}
    lnrec_by_cell: Dict[Tuple[float, int], List[float]] = {}
    dep_num_by_cell: Dict[Tuple[float, int], float] = {}   # Σ p_i·resid_i
    n_inj_by_cell: Dict[Tuple[float, int], int] = {}       # all injections (denom)
    for row in manifest:
        if _is_control(row):
            continue
        key = (float(row["logN_true"]), int(row["snr_bin"]))
        n_inj_by_cell[key] = n_inj_by_cell.get(key, 0) + 1  # counts non-recoveries too
        recd = recovered.get(row["inj_id"])
        if recd is None:
            continue
        p = recd.get("p_dla", np.nan)
        lr = recd.get("logN_rec", np.nan)
        if not (np.isfinite(p) and float(p) > p_dla_thresh and np.isfinite(lr)):
            continue
        resid = float(lr) - float(row["logN_true"])
        resid_by_cell.setdefault(key, []).append(resid)
        lnrec_by_cell.setdefault(key, []).append(float(lr))
        # deposit-weighted contribution: weight the residual by the recovered
        # posterior deposit mass p_i (non-recoveries above contribute nothing).
        dep_num_by_cell[key] = dep_num_by_cell.get(key, 0.0) + float(p) * resid

    keys = sorted(resid_by_cell.keys())
    logN = np.array([k[0] for k in keys], dtype=float)
    snr = np.array([k[1] for k in keys], dtype=int)
    n_used = np.array([len(resid_by_cell[k]) for k in keys], dtype=int)
    n_injected = np.array([n_inj_by_cell[k] for k in keys], dtype=int)
    b_N = np.empty(len(keys))
    sigma_N = np.empty(len(keys))
    b_N_se = np.empty(len(keys))
    b_N_deposit = np.empty(len(keys))
    logN_rec_dist: List[np.ndarray] = []
    for i, k in enumerate(keys):
        r = np.asarray(resid_by_cell[k], dtype=float)
        b_N[i] = float(np.mean(r))
        sigma_N[i] = float(np.std(r, ddof=0))
        b_N_se[i] = sigma_N[i] / np.sqrt(r.size) if r.size > 0 else np.nan
        # deposit-weighted bias over ALL injections in the cell (Malmquist-honest):
        # incomplete cells divide by the full injection count, dragging it toward 0.
        b_N_deposit[i] = (
            dep_num_by_cell.get(k, 0.0) / n_inj_by_cell[k]
            if n_inj_by_cell[k] > 0 else np.nan
        )
        logN_rec_dist.append(np.asarray(lnrec_by_cell[k], dtype=float))

    bias_unreliable = n_used < int(n_min)

    return {
        "logN_true": logN,
        "snr_bin": snr,
        "n_used": n_used,
        "n_injected": n_injected,
        "b_N": b_N,
        "sigma_N": sigma_N,
        "b_N_se": b_N_se,
        "b_N_deposit": b_N_deposit,
        "logN_rec_dist": logN_rec_dist,
        "bias_unreliable": bias_unreliable,
    }


# --------------------------------------------------------------------------- #
# 3. response matrix + b_FP
# --------------------------------------------------------------------------- #
def response_matrix(
    recovered: Mapping[int, Mapping],
    manifest: Sequence[Mapping],
    *,
    lnhi_edges: Sequence[float],
    z_edges: Sequence[float],
    normalize: bool = True,
    fp_prior: Tuple[float] = (0.5,),
) -> Dict[str, object]:
    """Off-diagonal response ``R[(N_rec,z_rec), (N_true,z_true,SNR)]`` + ``b_FP``.

    Builds, from the per-injection posterior DEPOSIT, the probability that a true
    absorber at (N_true, z_true, SNR) deposits recovered posterior mass at
    (N_rec, z_rec) — the full off-diagonal kernel (sub-DLA↔DLA migration across
    20.3, z-scatter, prior-edge pull).  ``b_FP`` is the spurious-deposit rate from
    the no-injection CONTROL rows (``logN_true`` NaN), as a Poisson-rate Gamma.

    Parameters
    ----------
    recovered : mapping {inj_id -> recovered-record}
        Needs ``deposit`` (the per-(N,z) posterior probability mass list).
    manifest : sequence of manifest rows
        Injection rows define the R columns; control rows (``logN_true`` NaN)
        define ``b_FP``.
    lnhi_edges, z_edges : sequences of float
        Recovered-axis bin EDGES.  Truth columns are binned on the SAME logN/z
        edges (plus the discrete SNR bin) so R is square in (N,z) up to the SNR
        resolution.
    normalize : bool, default True
        If True (default), each true column is divided by ``n_true`` (its injected
        count) so the column is a per-true-absorber recovered-mass DISTRIBUTION
        (entries in [0,1], column sum = recovered fraction ≤ 1).  If False, the
        column holds the raw summed deposit mass (sum ≤ ``n_true``).
    fp_prior : (a,), default (0.5,)
        Gamma prior shape offset for ``b_FP`` (matches the O3 FP core).

    Returns
    -------
    dict
        ``R``          : (n_rec_cells, n_true_cells) response matrix;
        ``true_cells`` : list of (ilnhi_true, iz_true, snr_bin) — the columns;
        ``rec_cells``  : list of (ilnhi_rec, iz_rec) — the rows;
        ``n_true``     : (n_true_cells,) injected count per true column;
        ``lnhi_edges, z_edges`` : the edge arrays (provenance);
        ``b_FP``       : dict ``{deposit, lo68, hi68, lo95, hi95, b_FP_shape,
                         n_control}`` — per recovered (N,z) cell (row-aligned to
                         ``rec_cells``), the spurious-deposit Gamma posterior.

    Modeling
    --------
    R column = deposit-mean response (see module docstring).  Off-diagonal mass is
    genuine migration; the column deficit (1 - sum, normalized) is incompleteness.
    b_FP per recovered cell = ``Gamma(FP_mass + a, scale=1)`` deposit posterior
    (the SAME convention as ``soft_completeness.estimate_false_positive_deposit``);
    point = posterior MODE, CI = equal-tailed Gamma quantiles bracketing the mode.
    """
    lnhi_edges = np.asarray(lnhi_edges, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)
    n_lnhi = lnhi_edges.size - 1
    n_z = z_edges.size - 1

    # ---- recovered-cell index: every (ilnhi_rec, iz_rec) on the grid ----
    rec_cells: List[Tuple[int, int]] = [
        (il, iz) for il in range(n_lnhi) for iz in range(n_z)
    ]
    rec_index = {c: i for i, c in enumerate(rec_cells)}
    n_rec_cells = len(rec_cells)

    # ---- true columns: occupied (ilnhi_true, iz_true, snr_bin) cells ----
    true_keys: Dict[Tuple[int, int, int], int] = {}  # cell -> n_injected
    # raw deposited mass per (rec_cell, true_cell)
    mass: Dict[Tuple[int, int], float] = {}  # (rec_idx, true_idx) -> mass
    # FP mass per recovered cell (control rows)
    fp_mass = np.zeros(n_rec_cells, dtype=float)
    n_control = 0

    # First pass: enumerate true columns (so column indexing is stable/sorted).
    for row in manifest:
        if _is_control(row):
            continue
        il = _bin_index(float(row["logN_true"]), lnhi_edges)
        iz = _bin_index(float(row["z_true"]), z_edges)
        if il < 0 or iz < 0:
            continue
        key = (il, iz, int(row["snr_bin"]))
        true_keys[key] = true_keys.get(key, 0) + 1
    true_cells = sorted(true_keys.keys())
    true_index = {c: j for j, c in enumerate(true_cells)}
    n_true = np.array([true_keys[c] for c in true_cells], dtype=int)

    # Second pass: deposit posterior mass into (rec_cell, true_col) and FP.
    for row in manifest:
        recd = recovered.get(row["inj_id"])
        deposit = (recd or {}).get("deposit", []) or []
        if _is_control(row):
            n_control += 1
            for (lN, zz, w) in deposit:
                ir = rec_index.get((_bin_index(lN, lnhi_edges), _bin_index(zz, z_edges)))
                if ir is not None:
                    fp_mass[ir] += float(w)
            continue
        il = _bin_index(float(row["logN_true"]), lnhi_edges)
        iz = _bin_index(float(row["z_true"]), z_edges)
        if il < 0 or iz < 0:
            continue
        jt = true_index[(il, iz, int(row["snr_bin"]))]
        for (lN, zz, w) in deposit:
            ir = rec_index.get((_bin_index(lN, lnhi_edges), _bin_index(zz, z_edges)))
            if ir is None:
                continue  # deposit outside the recovered grid is dropped
            mass[(ir, jt)] = mass.get((ir, jt), 0.0) + float(w)

    R = np.zeros((n_rec_cells, len(true_cells)), dtype=float)
    for (ir, jt), m in mass.items():
        R[ir, jt] = m
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = n_true.astype(float)
            denom[denom == 0] = np.nan
            R = R / denom[None, :]
        R = np.nan_to_num(R, nan=0.0)

    # ---- b_FP per recovered cell: Poisson-rate Gamma (O3 core convention) ----
    # exposure = number of control sightlines (the deposit is exposure-invariant by
    # construction in the core; pass a strictly-positive basis).
    exposure = float(max(n_control, 1))
    bfp_est = _estimate_fp_deposit(fp_mass, exposure, prior=fp_prior)
    b_FP = {
        "deposit": np.asarray(bfp_est["b_FP"], dtype=float),
        "lo68": np.asarray(bfp_est["b_FP_lo68"], dtype=float),
        "hi68": np.asarray(bfp_est["b_FP_hi68"], dtype=float),
        "lo95": np.asarray(bfp_est["b_FP_lo95"], dtype=float),
        "hi95": np.asarray(bfp_est["b_FP_hi95"], dtype=float),
        "b_FP_shape": np.asarray(bfp_est["b_FP_shape"], dtype=float),
        "fp_mass": fp_mass,
        "n_control": int(n_control),
    }

    return {
        "R": R,
        "true_cells": true_cells,
        "rec_cells": rec_cells,
        "n_true": n_true,
        "lnhi_edges": lnhi_edges,
        "z_edges": z_edges,
        "b_FP": b_FP,
    }


# --------------------------------------------------------------------------- #
# 4. GP+DLA generative-draw SPEC (the cross-check method — DESIGN ONLY)
# --------------------------------------------------------------------------- #
def gpdraw_spec(*, z_qso: float, native_snr: float) -> Dict[str, object]:
    """SPEC for the GP+DLA generative-draw cross-check (method (i)) — DESIGN ONLY.

    The campaign has TWO injection methods (M3 design, Refinement 2):

    * **(ii) PRIMARY** — inject a Voigt absorber into a CLEAN 2LPT-0 coadd
      (``inject_voigt`` on the real coadded flux), preserving the REAL Lyα forest /
      continuum / noise the GP sees in the run being corrected.  This is the
      faithful R for the real 2LPT-0 FILTER-off CDDF.
    * **(i) CROSS-CHECK** — draw a spectrum from the GP+DLA GENERATIVE model:
      ``s ~ NullGP(μ, K)`` for a given ``(z_qso, native_snr)``, multiply by the DLA
      Voigt at ``(N, z)``, run the GP.  Fully controlled (no clean-sightline
      selection), isolates INFERENCE self-consistency.  The **(ii)−(i) difference
      quantifies the real-forest contribution** to bias/incompleteness — most
      informative at NHI<19 (the forest can mask/mimic weak HCDs).

    This function returns the SPEC for the CS agent to IMPLEMENT the draw — it does
    NOT draw (the draw needs the GP model files: μ, M, the learned τ₀/β; the CS
    agent owns ``coadd_injection.py`` and the GP files).  The MEASUREMENT is
    IDENTICAL to method (ii): the drawn-and-absorbed spectrum runs through the SAME
    GP driver, and the recovered output feeds the SAME three estimators above.

    The generative draw (what the CS agent implements)
    --------------------------------------------------
    For a chosen QSO redshift ``z_qso`` build the null-GP model on the observed-
    frame grid (``NullGP.set_data`` → ``get_interp(z_qso)``), which yields

    * ``this_mu``     — (n,) the forest-absorbed mean model ``μ · a_lya``;
    * ``this_M``      — (n, k) the forest-absorbed low-rank covariance factor, so
                        the QSO-fluctuation covariance is ``K = this_M @ this_M.T``;
    * ``this_omega2`` — (n,) the Lyman-series-scaled diagonal model variance;
    * ``v``           — (n,) the instrumental noise variance (set the per-pixel
                        ``v`` to hit the target ``native_snr`` — e.g. scale ``v`` so
                        the in-forest SNR matches the SNR bin being populated).

    Draw a sample flux (the FULL null-GP generative model the inference scores
    against, ``N(this_mu, this_M this_M.T + diag(this_omega2 + v))``)::

        eps_k = rng.standard_normal(k)          # low-rank QSO fluctuations
        eps_n = rng.standard_normal(n)          # diagonal model + noise
        s = this_mu + this_M @ eps_k + sqrt(this_omega2 + v) * eps_n

    then imprint the absorber with the SAME Voigt as method (ii)::

        s_inj = inject_voigt(observed_wavelengths, s, 10**logN_true, z_true,
                             num_lines=num_lines)

    Write ``s_inj`` (with per-pixel noise ``v``) into the GP input schema and run the
    SAME production driver.  Because the draw uses the GP's OWN μ/K and the absorber
    its OWN Voigt, recovery is the cleanest possible test of inference self-
    consistency (no forest systematic) — the controlled baseline the real-forest
    method (ii) is differenced against.

    Parameters
    ----------
    z_qso : float
        QSO emission redshift the null-GP model is built at.
    native_snr : float
        Target in-forest SNR the per-pixel noise ``v`` is scaled to (the SNR-bin
        the draw populates), so method (i) and (ii) share the SNR parameterization.

    Returns
    -------
    dict
        A self-describing spec: ``method`` (the manifest tag ``"gpdraw"``),
        ``z_qso``, ``native_snr``, ``null_gp_pieces`` (the exact ``NullGP``
        attributes the draw needs), ``draw_formula`` (the sampling recipe),
        ``absorber_step`` (the Voigt imprint), ``measurement`` (identical to the
        coadd method), and ``implemented=False`` (this owner provides the spec, the
        CS agent implements the draw).
    """
    return {
        "method": "gpdraw",
        "z_qso": float(z_qso),
        "native_snr": float(native_snr),
        "null_gp_pieces": ("this_mu", "this_M", "this_omega2", "v"),
        "draw_formula": (
            "s = this_mu + this_M @ N(0, I_k) + sqrt(this_omega2 + v) * N(0, I_n) "
            "[the full null-GP generative model N(this_mu, this_M@this_M.T + "
            "diag(this_omega2 + v)) the inference scores against]"
        ),
        "absorber_step": (
            "s_inj = inject_voigt(observed_wavelengths, s, 10**logN_true, z_true, "
            "num_lines=num_lines)  # same Voigt profile as the coadd method (ii)"
        ),
        "snr_parameterization": (
            "scale the per-pixel noise variance v so the in-forest SNR equals "
            "native_snr (the SNR bin); v is the only knob, mu/M are z_qso-driven"
        ),
        "build_recipe": (
            "NullGP(params, prior, mu, M, ...).set_data(observed_wavelengths, flux, "
            "noise_variance); .get_interp(z_qso) -> this_mu/this_M/this_omega2"
        ),
        "measurement": (
            "identical to coadd: detection_completeness / nhi_bias / response_matrix"
        ),
        "implemented": False,  # design-only; the CS agent implements the draw
    }
