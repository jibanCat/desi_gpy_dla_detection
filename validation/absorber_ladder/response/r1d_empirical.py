#!/usr/bin/env python
"""r1d_empirical.py — R1d: the ALTERNATIVE fixed response representation.

Opened only because R1c leaves a DIAGNOSED residual that the skew-normal
3-moment family cannot carry (held-out row skew still -0.11 overall and -0.33
for b < 19.7, and a down/in mis-split at b >= 21.3): the true row is a narrow
core plus a broad tail, and NO (mean, sd, skew) skew-normal can be both.  R1d
therefore drops the parametric shape and keeps a smoothed EMPIRICAL mass row.

Representation (a FIXED calibration object, frozen before any HBI run):

    p[sr, zr, j, b]  = the migration mass of latent bin b into the observed
                       cell at OFFSET j (in units of the 0.1-dex observed grid)
                       from the observed cell containing centre(b),

estimated from the 2LPT-0 calibration events and SMOOTHED ALONG N: for every
(response cell, offset) the per-bin masses are fitted by a RIDGE-penalised
low-order polynomial in u = centre(b) - N_ref, weighted by the bin's event
count.  The polynomial degree and the ridge strength are chosen by the same
2-fold sightline CV as the other variants; nothing is fitted to, or scored
against, any closure quantity.

HONEST COMPLEXITY WARNING (PI ruling 2026-09-13b §5, §19).  This is a
high-dimensional calibration model: nominally 9 cells x n_offset x (deg+1)
coefficients, versus 84 for R0/R1a/R1b and 108 for R1c.  The effective number
of parameters (the trace of the ridge hat matrix, summed over cells and
offsets) is reported alongside, and the report flags the variant as one whose
dimensionality must be ruled on by the PI before it may be adopted.  Its
within-bin marginalisation weight is the CALIBRATION MOCK's own N distribution
(the events' own density inside each latent bin), not a flat weight — a
documented dependence on the calibration population that R1c deliberately
avoids.

VALIDATION-ONLY.  ENV: gpdla.
"""
from __future__ import annotations

import numpy as np


def offset_index(xhat, Nc_b, nhat_edges):
    """Offset (in observed-grid cells) of each detection from the cell holding
    its latent bin centre.  Returns (j, c_idx, c0)."""
    ne = np.asarray(nhat_edges, float)
    C = len(ne) - 1
    c_idx = np.digitize(xhat, ne) - 1
    c0 = np.clip(np.digitize(Nc_b, ne) - 1, 0, C - 1)
    return c_idx - c0, c_idx, c0


def raw_masses(N_true, xhat, isr, izr, b_i, ntrue_edges, nhat_edges, J):
    """(3, 3, 2J+1, B) raw counts and (3, 3, B) per-row totals."""
    nt = np.asarray(ntrue_edges, float)
    ne = np.asarray(nhat_edges, float)
    Nc = 0.5 * (nt[:-1] + nt[1:])
    B = len(Nc); C = len(ne) - 1
    c_idx = np.digitize(xhat, ne) - 1
    c0 = np.clip(np.digitize(Nc, ne) - 1, 0, C - 1)
    j = c_idx - c0[b_i]
    ok = (c_idx >= 0) & (c_idx < C) & (np.abs(j) <= J)
    A = np.zeros((3, 3, 2 * J + 1, B))
    np.add.at(A, (isr[ok], izr[ok], j[ok] + J, b_i[ok]), 1.0)
    return A, A.sum(axis=2), c0


def smooth_along_N(A, tot, Nc, N_ref, deg, lam):
    """Ridge-penalised degree-``deg`` polynomial in u = centre(b) - N_ref,
    fitted per (cell, offset) to the per-bin mass fractions, weighted by the
    bin's event count.  Returns (p_smooth (3,3,2J+1,B), effective DOF)."""
    B = A.shape[-1]
    u = np.asarray(Nc, float) - float(N_ref)
    X = np.vander(u, deg + 1, increasing=True)             # (B, deg+1)
    P = np.eye(deg + 1)
    P[0, 0] = 0.0                                          # never penalise the level
    out = np.zeros_like(A)
    edof = 0.0
    for i in range(3):
        for j in range(3):
            w = tot[i, j]                                  # (B,)
            sw = np.sqrt(np.clip(w, 0.0, None))
            Xw = X * sw[:, None]
            G = Xw.T @ Xw + float(lam) * P
            try:
                Gi = np.linalg.inv(G)
            except np.linalg.LinAlgError:                  # pragma: no cover
                Gi = np.linalg.pinv(G)
            H = Xw @ Gi @ Xw.T
            n_off = 0
            for k in range(A.shape[2]):
                y = np.where(w > 0, A[i, j, k] / np.maximum(w, 1e-12), 0.0)
                beta = Gi @ (Xw.T @ (sw * y))
                out[i, j, k] = X @ beta
                if A[i, j, k].sum() > 0:
                    n_off += 1
            edof += float(np.trace(H)) * n_off
    out = np.clip(out, 0.0, None)
    s = out.sum(axis=2, keepdims=True)
    out = np.divide(out, np.maximum(s, 1e-300))
    return out, edof


def to_masses(p, c0, C):
    """(3, 3, 2J+1, B) offset rows -> (3, 3, C, B) masses on the observed grid,
    with mass that falls off the grid dropped (then the row renormalised, the
    count-conservation convention)."""
    SR, ZR, nJ, B = p.shape
    J = (nJ - 1) // 2
    out = np.zeros((SR, ZR, C, B))
    for b in range(B):
        for k in range(nJ):
            c = c0[b] + (k - J)
            if 0 <= c < C:
                out[:, :, c, b] += p[:, :, k, b]
    s = out.sum(axis=2, keepdims=True)
    return np.divide(out, np.maximum(s, 1e-300)), s[:, :, 0, :]


def fit_r1d(N_true, xhat, isr, izr, b_i, ntrue_edges, nhat_edges, N_ref,
            deg=2, lam=1.0, J=15):
    nt = np.asarray(ntrue_edges, float)
    Nc = 0.5 * (nt[:-1] + nt[1:])
    C = len(np.asarray(nhat_edges, float)) - 1
    A, tot, c0 = raw_masses(N_true, xhat, isr, izr, b_i, ntrue_edges,
                            nhat_edges, J)
    p, edof = smooth_along_N(A, tot, Nc, N_ref, deg, lam)
    masses, phi = to_masses(p, c0, C)
    n_off = int(np.sum(A.sum(axis=(0, 1, 3)) > 0))
    return dict(masses=masses, phi=phi, p_offset=p, raw=A, tot=tot, c0=c0,
                J=int(J), deg=int(deg), lam=float(lam),
                n_offsets_populated=n_off,
                n_coef=dict(nominal=int(9 * n_off * (deg + 1)),
                            effective=float(edof),
                            note=("nominal = 9 response cells x populated "
                                  "offsets x (deg+1); effective = trace of "
                                  "the ridge hat matrix summed over cells and "
                                  "populated offsets")),
                N_ref=float(N_ref), ramp=None, spec=dict(
                    name="R1d", representation="smoothed empirical masses",
                    deg=int(deg), ridge_lambda=float(lam), J=int(J),
                    fit_rng="n/a (no covariate clamp; no polynomial in the "
                            "response moments)",
                    marginalisation_weight="calibration mock f(N) within bin"))
