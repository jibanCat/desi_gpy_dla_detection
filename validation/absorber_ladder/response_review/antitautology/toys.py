#!/usr/bin/env python
"""toys.py — the MUTATION controls for the anti-tautology battery.

Test A is only evidence if it can actually fail.  Two toy estimators are
therefore pushed through the identical pipeline:

  ``toy_imprinted``   — the saturated empirical row SHRUNK TOWARD THE POOLED
                        N-hat MARGINAL of the whole calibration sample.  The
                        shrinkage target is a function of the calibration
                        population p(N_true) (it is that population folded
                        through the response), so the operator carries the
                        mock's CDDF slope.  Test A MUST flag it.
  ``toy_conditional`` — a purely conditional analytic row: a Gaussian on the
                        observed grid centred on the latent-bin centre with a
                        width that depends only on (b, S/N cell).  No
                        calibration count enters at all, so it is invariant
                        under ANY reweighting by construction.  Test A MUST
                        pass it.
  ``toy_rowfit``      — a realistic conditional control: per-row Gaussian with
                        mean and sd estimated from THAT ROW's own events only
                        (no pooling across rows).  Under reweighting only the
                        within-row composition moves, so the change must stay
                        at the sampling-noise level.

VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import numpy as np

from opbuild import Operator, row_index, row_shape, _weighted_counts

IMPRINT_LAMBDA = 0.30


def build_toy_imprinted(ev, w, geom, sel=None, grid="resp",
                        lam=IMPRINT_LAMBDA):
    """DELIBERATELY occupancy-imprinted: rows shrunk toward the pooled
    (population-dependent) observed marginal."""
    A, n_raw = _weighted_counts(ev, w, grid, geom, sel)
    pool = A.sum(axis=0)
    pool = pool / max(pool.sum(), 1e-300)
    P = A + 0.5
    P = P / P.sum(axis=1, keepdims=True)
    P = (1.0 - lam) * P + lam * pool[None, :]
    return Operator("toy_imprinted", P, grid, geom, row_n=n_raw,
                    meta=dict(shrink_lambda=lam,
                              target="pooled observed N-hat marginal"))


def _gauss_rows(centres, cen_grid, sds):
    d = cen_grid[None, :] - np.asarray(centres, float)[:, None]
    P = np.exp(-0.5 * (d / np.asarray(sds, float)[:, None]) ** 2)
    return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)


def build_toy_conditional(ev, w, geom, sel=None, grid="resp"):
    """Exactly conditional: an analytic row that never touches a count."""
    sh = row_shape(grid, geom)
    meta = np.array(list(np.ndindex(*sh)), int)
    b = meta[:, 2]
    i0 = meta[:, 0]
    sd = 0.10 + 0.02 * i0 + 0.01 * (geom["Nc"][b] - 20.5)
    P = _gauss_rows(geom["Nc"][b], geom["cen"], np.maximum(sd, 0.05))
    n_raw = np.zeros(int(np.prod(sh)))
    m = np.ones(ev["n"], bool) if sel is None else np.asarray(sel, bool)
    np.add.at(n_raw, row_index(grid, ev, geom, m & ev["in_grid"]), 1.0)
    return Operator("toy_conditional", P, grid, geom, row_n=n_raw,
                    meta=dict(analytic=True))


def build_toy_rowfit(ev, w, geom, sel=None, grid="resp"):
    """Conditional, data-driven, no cross-row pooling: per-row Gaussian whose
    (mean, sd) are the row's OWN weighted first two moments of N-hat."""
    A, n_raw = _weighted_counts(ev, w, grid, geom, sel)
    cen = geom["cen"]
    tot = A.sum(axis=1)
    m = np.divide(A @ cen, np.where(tot > 0, tot, 1.0))
    v = np.divide(A @ (cen ** 2), np.where(tot > 0, tot, 1.0)) - m ** 2
    sh = row_shape(grid, geom)
    b = np.array(list(np.ndindex(*sh)), int)[:, 2]
    m = np.where(tot > 0, m, geom["Nc"][b])
    sd = np.sqrt(np.maximum(v, 0.0))
    # the sparse-row guard keys on the RAW event count, never on the weighted
    # total: keying it on the weighted total would make the toy occupancy-
    # dependent through the back door and it would stop being a clean control.
    sd = np.where(n_raw >= 5, np.maximum(sd, 0.05), 0.15)
    P = _gauss_rows(m, cen, sd)
    return Operator("toy_rowfit", P, grid, geom, row_n=n_raw,
                    meta=dict(pooling="none"))


TOY_BUILDERS = {
    "toy_imprinted": build_toy_imprinted,
    "toy_conditional": build_toy_conditional,
    "toy_rowfit": build_toy_rowfit,
}
