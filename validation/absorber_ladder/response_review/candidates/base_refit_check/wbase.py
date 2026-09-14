"""wbase.py -- WEIGHTED refit of the R1c-form base kernel.

DIAGNOSTIC ONLY.  Calibration side; no HBI run; no real data; no tracked file
is modified.  This module exists for exactly one reason: ``respfit.fit_variant``
(and therefore ``run_candidates.parametric_rows``) takes NO event weights, so
the sealed section 6A audit could not refit E's base under the reweightings and
held the base FIXED (mathematics review F1/F2).  Here the SAME estimator is
expressed with per-event weights so that the base can be refitted.

Reduction guarantee (tested): with w == 1 every routine below returns bit-wise
what the respfit routine it mirrors returns, and ``fit_variant_weighted``
returns an object whose rows equal ``run_candidates.parametric_rows``.

Weight convention = FREQUENCY weights (an event of weight w counts as w
events), which is what the section 6A reweightings mean:
  * ``adaptive_edges``     : the min_n / hard_min_n merge thresholds are
                             applied to the WEIGHTED count;
  * sub-bin admission      : weighted count >= min_n AND RAW count >= 5
                             (the raw guard keeps the moment estimators
                             defined; it is the literal ``n < 5`` guard of
                             ``respfit.subbin_moments_weighted_centre``);
  * sub-bin centre/moments : weighted mean / weighted sd (ddof-1 analogue,
                             denominator sum(w) - 1) / weighted skewness on
                             the population sd, capped at SKEW_CAP;
  * WLS surface weight     : sqrt(weighted count), i.e. the row ``n`` carried
                             into ``respfit.surfaces_shared`` is sum(w).
Everything downstream (``surfaces_shared``, ``marginalise_object``,
``count_coefficients``, ``opmetrics.model_masses``) is the COMMITTED code,
imported, not copied.
"""
from __future__ import annotations

import numpy as np

import respfit as RF
import opmetrics as OM


# ---------------------------------------------------------------------------
def adaptive_edges_w(N, w, lo, hi, step, min_n, hard_min_n=25, max_width=None):
    """``respfit.adaptive_edges`` with weighted counts."""
    N = np.asarray(N, float)
    w = np.ones(len(N)) if w is None else np.asarray(w, float)
    grid = np.arange(lo, hi + 1e-9, step)
    if grid[-1] < hi - 1e-9:
        grid = np.append(grid, hi)
    counts, _ = np.histogram(N, grid, weights=w)
    edges = [grid[0]]
    acc = 0.0
    for i in range(len(counts)):
        acc += float(counts[i])
        width = grid[i + 1] - edges[-1]
        wide = (max_width is not None) and (width >= max_width - 1e-9)
        if acc >= min_n or (wide and acc >= hard_min_n):
            edges.append(grid[i + 1])
            acc = 0.0
    if edges[-1] < grid[-1] - 1e-12:
        if acc >= hard_min_n:
            edges.append(grid[-1])
        elif len(edges) > 1:
            edges[-1] = grid[-1]
        else:
            edges.append(grid[-1])
    return np.asarray(edges, float)


def subbin_moments_wc_w(N, dx, w, edges, min_n, mode):
    """``respfit.subbin_moments_weighted_centre`` with frequency weights."""
    N = np.asarray(N, float); dx = np.asarray(dx, float)
    w = np.ones(len(N)) if w is None else np.asarray(w, float)
    edges = np.asarray(edges, float)
    rows = []
    ib = np.digitize(N, edges) - 1
    for b in range(len(edges) - 1):
        m = ib == b
        n_raw = int(m.sum())
        sw = float(w[m].sum())
        if sw < min_n or n_raw < 5:
            continue
        dn, dd, ww = N[m], dx[m], w[m]
        c = float(np.sum(ww * dn) / sw)
        if mode == "sample":
            mu = float(np.sum(ww * dd) / sw)
            e = dd - mu
            var_u = float(np.sum(ww * e ** 2) / max(sw - 1.0, 1e-12))
            var_p = float(np.sum(ww * e ** 2) / sw)
            sg = float(np.sqrt(max(var_u, 0.0)))
            m3 = float(np.sum(ww * e ** 3) / sw)
            sk = float(np.clip(m3 / max(np.sqrt(max(var_p, 0.0)), 1e-6) ** 3,
                               -RF.SKEW_CAP, RF.SKEW_CAP))
            rows.append(dict(c=c, n=sw, mu=mu, sig=sg, skew=sk, ok=True))
        elif mode in ("ml", "ml_trunc"):
            p, ok, _ = RF.fit_subbin_ml(dd, RF.XHAT_FLOOR - dn,
                                        truncated=(mode == "ml_trunc"), w=ww)
            rows.append(dict(c=c, n=sw, mu=float(p[0]), sig=float(p[1]),
                             skew=float(p[2]), ok=ok))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return rows


def _rows_for_cells_w(N, dx, isr, izr, w, spec):
    rows = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            m = (isr == i) & (izr == j)
            if spec["edges"] != "adaptive":
                raise ValueError("weighted refit implemented for the adaptive "
                                 "sub-bin grid (R1a/R1b/R1c) only")
            e = adaptive_edges_w(N[m], w[m], spec["lo"], spec["hi"],
                                 spec["step"], spec["min_n"],
                                 hard_min_n=spec.get("hard_min_n", 25),
                                 max_width=spec.get("max_width"))
            rows[i][j] = subbin_moments_wc_w(N[m], dx[m], w[m], e,
                                             spec["min_n"], spec["estimator"])
    return rows


def fit_variant_weighted(N, dx, isr, izr, N_ref, spec, w=None,
                         ntrue_edges=None):
    """``respfit.fit_variant`` with frequency weights on the events."""
    N = np.asarray(N, float)
    w = np.ones(len(N)) if w is None else np.asarray(w, float)
    rows = _rows_for_cells_w(N, dx, isr, izr, w, spec)
    if spec["deg_shared"] > spec["deg_cell"]:
        surf, rng, shared = RF.surfaces_shared(rows, N_ref, spec["deg_shared"],
                                               deg_cell=spec["deg_cell"])
    else:
        surf, rng = RF.surfaces_percell(rows, N_ref, spec["deg_cell"])
        shared = np.zeros((3, spec["deg_cell"] + 1))
    rng_data = rng.copy()
    if spec["fit_rng"] == "full":
        if ntrue_edges is None:
            raise ValueError("fit_rng='full' needs ntrue_edges")
        ne = np.asarray(ntrue_edges, float)
        rng = np.tile(np.array([ne[0], ne[-1]]), (3, 3, 1))
    ramp = spec["ramp"] if spec["ramp"] is not None else RF.NO_RAMP
    out = dict(surf=surf, rng=rng, rng_data=rng_data, shared=shared,
               ramp=ramp, N_ref=float(N_ref), spec=dict(spec), rows=rows)
    if spec.get("marginalise"):
        out = RF.marginalise_object(out, ntrue_edges, **spec["marginalise"])
    out["n_coef"] = RF.count_coefficients(out)
    return out


def base_rows_weighted(ev, mask, geom, spec, N_ref, w=None):
    """The R1c-form base rows (B, S, K, C) refitted under event weights.

    Mirrors ``run_candidates.parametric_rows`` -- same masses -> same gather ->
    same normalisation -- with the weighted fit substituted for fit_variant.
    """
    ww = None if w is None else np.asarray(w, float)[mask]
    obj = fit_variant_weighted(ev["N_true"][mask], ev["dx"][mask],
                               ev["isr"][mask], ev["izr"][mask], N_ref,
                               dict(spec), w=ww, ntrue_edges=geom["ntrue"])
    masses, _ = OM.model_masses(obj, geom["ntrue"], geom["nhat"],
                                sig_floor=geom["sig_floor"])
    m = masses[geom["s2sr"][:, None], geom["K2zr"][None, :], :, :]
    rows = np.transpose(m, (3, 0, 1, 2))
    rows = rows / np.maximum(rows.sum(axis=-1, keepdims=True), 1e-300)
    return rows, obj
