#!/usr/bin/env python
"""metrics.py — row KL, boundary-crossing / leakage summaries and the paired
held-out predictive score used by anti-tautology tests A and B.

Conventions (stated once, applied everywhere):
  * every row is floored by ``opbuild.floor_rows`` (uniform eps = 1e-6) before
    any log, so KL is finite and a zero cell in one operator cannot make a
    comparison infinite;
  * KL(native || other) = sum_c p_nat[c] log(p_nat[c] / p_oth[c])  (nats);
  * a row's "boundary-crossing mass" at threshold T is the row mass that ends
    up on the WRONG side of T relative to the latent bin the row belongs to:
    P(N-hat >= T) if the bin lies below T, else P(N-hat < T);
  * "tail mass beyond +-0.3 dex" is P(|centre(c) - centre(b)| > 0.3).

VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import numpy as np

from opbuild import EPS_FLOOR, floor_rows

THRESHOLDS = (20.0, 20.3, 21.0)
TAIL_DEX = 0.3


def row_kl(P_ref, P_oth, eps=EPS_FLOOR):
    """(n_rows,) KL(P_ref || P_oth) in nats, both floored."""
    a = floor_rows(P_ref, eps)
    b = floor_rows(P_oth, eps)
    return np.sum(a * (np.log(a) - np.log(b)), axis=1)


def row_leakage(P, geom, grid_row_b):
    """Per-row leakage summaries.

    Returns a dict of (n_rows,) arrays:
      ``cross_20.0`` / ``cross_20.3`` / ``cross_21.0`` — boundary-crossing
      mass; ``ge_20.0`` ... — plain P(N-hat >= T) (useful when the sign of a
      change matters); ``tail_0.3`` — mass beyond +-0.3 dex of the bin centre.
    """
    cen = geom["cen"]
    Nc = geom["Nc"][np.asarray(grid_row_b, int)]            # (n_rows,)
    P = np.asarray(P, float)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
    out = {}
    for T in THRESHOLDS:
        ge = P[:, cen >= T].sum(axis=1)
        out[f"ge_{T}"] = ge
        out[f"cross_{T}"] = np.where(Nc < T, ge, 1.0 - ge)
    d = np.abs(cen[None, :] - Nc[:, None])
    out[f"tail_{TAIL_DEX}"] = (P * (d > TAIL_DEX + 1e-12)).sum(axis=1)
    return out


def wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    g = np.isfinite(x) & (w > 0)
    if not g.any():
        return float("nan")
    return float(np.sum(w[g] * x[g]) / np.sum(w[g]))


def paired_loglik(op_a, op_b, ev, sel, eval_w, cluster=None):
    """Weighted paired held-out per-event multinomial log-likelihood.

    Returns (l_a, l_b, d_mean, d_se) where d_mean = l_a - l_b and ``d_se`` is
    a TARGETID-clustered (sightline-clustered) standard error of the weighted
    mean difference, because calibration events share sightlines.
    """
    la = op_a.logp_events(ev, sel)
    lb = op_b.logp_events(ev, sel)
    w = np.asarray(eval_w, float)[sel]
    W = w.sum()
    l_a = float(np.sum(w * la) / W)
    l_b = float(np.sum(w * lb) / W)
    d = la - lb
    dm = float(np.sum(w * d) / W)
    r = w * (d - dm)                                        # influence terms
    if cluster is None:
        cluster = ev["tid"][sel]
    cl = np.asarray(cluster)
    _, inv = np.unique(cl, return_inverse=True)
    g = np.bincount(inv, weights=r)
    se = float(np.sqrt(np.sum(g ** 2)) / W)
    return l_a, l_b, dm, se


def per_event_loglik(op, ev, sel, eval_w):
    ll = op.logp_events(ev, sel)
    w = np.asarray(eval_w, float)[sel]
    return float(np.sum(w * ll) / w.sum())
