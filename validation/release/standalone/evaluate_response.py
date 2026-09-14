#!/usr/bin/env python
"""evaluate_response.py -- STANDALONE evaluator for the DESI GP-DLA Paper-1
forward response operator  M = phi(b,s,K) * Q(c | b,s,K).

SHIPPED IN THE ZENODO RELEASE.  numpy only: no project code.

Two distinct conditional objects (PI ruling 2026-09-13d §3)
-----------------------------------------------------------
``Q(c | b, s, K)``  the SHAPE of the reported-column distribution GIVEN that a
    truth absorber in latent cell (b = truth log N_HI bin, s = S/N stratum,
    K = coarse redshift block) is detected AND its reported N_hat lands on the
    observed grid.  Rows sum to one EXACTLY over the 29 observed bins.
``phi(b, s, K)``    the in-grid HAD MASS: P(N_hat lands on the >= 19.5 observed
    grid | detected, b, s, K).  Measured per cell with a Jeffreys +1/2; a
    6-coefficient smooth alternative is shipped as the predeclared response-
    calibration sensitivity.

Two ways to evaluate Q
----------------------
1. **Cell lookup** in the released tensor (``response_row``) -- this is exactly
   what the inference does, and the tensor is the OBJECT OF RECORD.
2. **Rebuild from the fitted coefficients** (``rebuild_Q_E`` / ``rebuild_Q_B``),
   using ``response_coefficients_<var>.npz``.  The ladder objects were
   evaluated tensors; the coefficients were recovered POST HOC by a
   deterministic refit and are shipped with the measured reproduction error
   (see ``response_coefficients.json``).

Family forms
------------
``B``  Q proportional to the bin mass of
       pi * SplitNormal(N + d1, w1, kappa) + (1 - pi) * Normal(N + d1 + d2, w2),
       each of (logit pi, d1, log w1, kappa, d2, log w2) linear in the design
       [1, u, u^2, v, 1[K=1], 1[K=2]] -- 6 x 6 = 36 coefficients.
``E``  Q = normalise_c[ base(c) * exp( sum_{r<=2} a_r(x) phi_r(c) ) ] with the
       base the R1c parametric kernel row (released as the evaluated tensor
       ``base_rows_R1c``; label it exactly as built: estimator "sample",
       marginalise n_quad 17 / deg 4 / flat weight) and a_r linear in the
       design [1, u, v, 1[K=1], 1[K=2]] -- alpha (2 x 5) and phi_r (2 x 29).

Both are evaluated by the predeclared FLAT quadrature: 17 interior nodes across
the latent N bin and 5 interior nodes across the stratum in log10(S/N), the
open top stratum capped at S/N = 40 and the empty bottom stratum floored at 1,
with the row averaged over nodes and renormalised.

Grid
----
b : 16 truth bins, edges ``ntrue_edges``  (19.0 ... 22.4)
s :  8 S/N strata, edges ``snr_edges``    (0,1,2,3,4,5,6,7,inf)
K :  3 coarse z blocks, edges ``zc_edges``(2.0, 2.5, 3.0, 3.5)
c : 29 observed N_hat bins, edges ``nhat_edges`` (19.5 ... 22.4)

phi_smooth
----------
    logistic( a0 + a1 w + a2 w^2 + a3 v_s + a4 1[K=1] + a5 1[K=2] )
with ``w = N_true_bin_centre - 20.5`` and ``v_s`` the stratum's flat-quadrature
midpoint of ``log10(S/N) - 0.7`` (shipped as ``vmid_s``; it is a property of
the stratum, not of an individual sightline).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["cell_index", "response_row", "phi_at", "phi_smooth",
           "operator_M", "row_sum_check", "quad_nodes", "design",
           "rebuild_Q_E", "rebuild_Q_B"]

N_REF_U = 20.5          # N pivot of the design and of phi_smooth
V_REF = 0.7             # log10 S/N pivot of the design and of phi_smooth
N_QUAD_U = 17           # flat interior quadrature nodes across a latent bin
N_QUAD_V = 5            # flat interior quadrature nodes across a stratum
SNR_TOP_CAP = 40.0      # finite cap for the open top stratum [7, inf)
SNR_BOT_FLOOR = 1.0     # finite floor for the (empty) bottom stratum [0, 1)
BASE_FLOOR = 1e-10      # floor under the E base kernel before the log


def _ndtr(x):
    """Standard normal CDF (stdlib erf; no scipy)."""
    x = np.asarray(x, dtype=float)
    flat = np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0)))
                     for v in x.ravel()], dtype=float)
    return flat.reshape(x.shape)


def quad_nodes(ntrue_edges, snr_edges, n_quad_u=N_QUAD_U, n_quad_v=N_QUAD_V):
    """(u_nodes (B, n_quad_u), v_nodes (S, n_quad_v)) -- the flat quadrature.

    ``u = N_true - 20.5`` and ``v = log10(S/N) - 0.7`` at the INTERIOR nodes of
    a uniform partition of each latent bin / stratum.
    """
    nt = np.asarray(ntrue_edges, dtype=float)
    se = np.asarray(snr_edges, dtype=float)
    u = np.stack([np.linspace(nt[b], nt[b + 1], n_quad_u + 2)[1:-1] - N_REF_U
                  for b in range(len(nt) - 1)])
    v = []
    for s in range(len(se) - 1):
        lo = max(float(se[s]), SNR_BOT_FLOOR)
        hi = float(se[s + 1])
        hi = SNR_TOP_CAP if not np.isfinite(hi) else min(hi, SNR_TOP_CAP)
        hi = max(hi, lo * 1.0001)
        v.append(np.linspace(np.log10(lo), np.log10(hi),
                             n_quad_v + 2)[1:-1] - V_REF)
    return u, np.stack(v)


def design(u, v, K, kind):
    """Coefficient-function design.

    kind ``'e'``    : [1, u, v, 1[K=1], 1[K=2]]        (P = 5; family E)
    kind ``'full'`` : [1, u, u^2, v, 1[K=1], 1[K=2]]   (P = 6; family B)
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    K = np.asarray(K, dtype=int)
    one = np.ones_like(u)
    k1 = (K == 1).astype(float)
    k2 = (K == 2).astype(float)
    if kind == "e":
        cols = [one, u, v, k1, k2]
    elif kind == "full":
        cols = [one, u, u * u, v, k1, k2]
    else:
        raise ValueError("unknown design kind %r" % kind)
    return np.stack(cols, axis=-1)


def _softmax(eta):
    m = eta.max(axis=-1, keepdims=True)
    e = np.exp(eta - m)
    return e / e.sum(axis=-1, keepdims=True)


def rebuild_Q_E(coef, n_K=3):
    """Rebuild Q[b, s, K, c] for family E from ``response_coefficients_E.npz``.

    Needs ``E_alpha``, ``E_phi``, ``E_xmu``, ``E_xsd``, ``base_rows_R1c``,
    ``ntrue_edges`` and ``snr_edges``.
    """
    alpha = np.asarray(coef["E_alpha"], dtype=float)          # (R, P)
    phi_r = np.asarray(coef["E_phi"], dtype=float)            # (R, C)
    xmu = np.asarray(coef["E_xmu"], dtype=float)
    xsd = np.asarray(coef["E_xsd"], dtype=float)
    base = np.asarray(coef["base_rows_R1c"], dtype=float)     # (B,S,K,C)
    un, vn = quad_nodes(coef["ntrue_edges"], coef["snr_edges"])
    B, S, C = un.shape[0], vn.shape[0], phi_r.shape[1]
    out = np.zeros((B, S, n_K, C))
    for b in range(B):
        for s in range(S):
            uu, vv = np.meshgrid(un[b], vn[s], indexing="ij")
            uu = uu.ravel()
            vv = vv.ravel()
            for k in range(n_K):
                X = design(uu, vv, np.full(uu.size, k), "e")
                Xs = (X - xmu) / xsd
                eta = (Xs @ alpha.T) @ phi_r
                eta = eta + np.log(np.maximum(base[b, s, k], BASE_FLOOR))
                p = _softmax(eta).mean(axis=0)
                out[b, s, k] = p / p.sum()
    return out


def rebuild_Q_B(coef, n_K=3):
    """Rebuild Q[b, s, K, c] for family B from ``response_coefficients_B.npz``.

    Needs ``B_beta``, ``B_xmu``, ``B_xsd``, ``ntrue_edges``, ``nhat_edges``
    and ``snr_edges``.
    """
    beta = np.asarray(coef["B_beta"], dtype=float)            # (6, P)
    xmu = np.asarray(coef["B_xmu"], dtype=float)
    xsd = np.asarray(coef["B_xsd"], dtype=float)
    edges = np.asarray(coef["nhat_edges"], dtype=float)
    un, vn = quad_nodes(coef["ntrue_edges"], coef["snr_edges"])
    B, S, C = un.shape[0], vn.shape[0], len(edges) - 1
    out = np.zeros((B, S, n_K, C))
    for b in range(B):
        for s in range(S):
            uu, vv = np.meshgrid(un[b], vn[s], indexing="ij")
            uu = uu.ravel()
            vv = vv.ravel()
            Nabs = uu + N_REF_U
            for k in range(n_K):
                X = design(uu, vv, np.full(uu.size, k), "full")
                t = ((X - xmu) / xsd) @ beta.T                # (n, 6)
                pi = 1.0 / (1.0 + np.exp(-t[:, 0]))
                m1 = Nabs + t[:, 1]
                w1 = np.exp(np.clip(t[:, 2], -4.0, 2.0))
                ka = np.clip(t[:, 3], -3.0, 3.0)
                m2 = m1 + t[:, 4]
                w2 = np.exp(np.clip(t[:, 5], -4.0, 2.0))
                wm = w1 * np.exp(-0.5 * ka)
                wp = w1 * np.exp(0.5 * ka)
                e = edges[None, :]
                sm = (wm + wp)[:, None]
                left = (2.0 * wm[:, None] / sm) * _ndtr((e - m1[:, None])
                                                        / wm[:, None])
                right = wm[:, None] / sm + (2.0 * wp[:, None] / sm) * (
                    _ndtr((e - m1[:, None]) / wp[:, None]) - 0.5)
                F1 = np.where(e < m1[:, None], left, right)
                F2 = _ndtr((e - m2[:, None]) / w2[:, None])
                mass = (pi[:, None] * np.diff(F1, axis=1)
                        + (1.0 - pi[:, None]) * np.diff(F2, axis=1))
                mass = np.clip(mass, 1e-300, None)
                p = (mass / mass.sum(axis=1, keepdims=True)).mean(axis=0)
                out[b, s, k] = p / p.sum()
    return out


def _bin_index(values, edges, name):
    """Index of the half-open bin ``[edge_i, edge_{i+1})``; -1 outside."""
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.where(values >= edges[-1], -1, idx)
    idx = np.where(values < edges[0], -1, idx)
    if np.any(idx < 0):
        bad = values[np.asarray(idx) < 0] if values.ndim else values
        raise ValueError("%s value(s) outside the calibrated grid: %r"
                         % (name, bad))
    return np.asarray(idx, dtype=int)


def cell_index(logN, snr, z, model):
    """(b, s, K) for truth log N_HI, sightline S/N and absorber/QSO redshift."""
    b = _bin_index(logN, model["ntrue_edges"], "log N_HI")
    s = _bin_index(snr, model["snr_edges"], "S/N")
    K = _bin_index(z, model["zc_edges"], "z")
    return b, s, K


def response_row(logN, snr, z, model):
    """The 29-vector Q(. | b, s, K) for one (logN, snr, z)."""
    b, s, K = cell_index(logN, snr, z, model)
    return np.asarray(model["Q"], dtype=float)[b, s, K]


def phi_at(logN, snr, z, model, smooth=False):
    """The in-grid had mass phi(b, s, K)."""
    b, s, K = cell_index(logN, snr, z, model)
    key = "phi_smooth" if smooth else "phi"
    return float(np.asarray(model[key], dtype=float)[b, s, K])


def phi_smooth(b, s, K, coef, bcen, vmid):
    """Closed-form 6-coefficient phi on the calibration grid."""
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.size != 6:
        raise ValueError("phi_smooth takes 6 coefficients, got %d" % coef.size)
    w = float(np.asarray(bcen, dtype=float)[b]) - N_REF_U
    v = float(np.asarray(vmid, dtype=float)[s])
    eta = (coef[0] + coef[1] * w + coef[2] * w * w + coef[3] * v
           + coef[4] * (K == 1) + coef[5] * (K == 2))
    return 1.0 / (1.0 + np.exp(-eta))


def operator_M(model, smooth_phi=False):
    """``M[b, s, K, c] = phi(b,s,K) * Q(c | b,s,K)`` -- the object the forward
    count model contracts against the completeness surface and the latent CDDF.
    """
    Q = np.asarray(model["Q"], dtype=float)
    phi = np.asarray(model["phi_smooth" if smooth_phi else "phi"], dtype=float)
    return phi[:, :, :, None] * Q


def row_sum_check(model, atol=1e-12):
    """max |sum_c Q - 1| over every populated cell; raises above ``atol``."""
    Q = np.asarray(model["Q"], dtype=float)
    dev = float(np.max(np.abs(Q.sum(axis=-1) - 1.0)))
    if dev > atol:
        raise ValueError("released Q rows do not sum to one: max dev %g" % dev)
    return dev


if __name__ == "__main__":                                   # pragma: no cover
    import sys
    m = np.load(sys.argv[1] if len(sys.argv) > 1 else "response_model_E.npz")
    print("row-sum deviation:", row_sum_check(m))
    row = response_row(20.35, 4.0, 2.6, m)
    print("Q(.|N=20.35,S/N=4,z=2.6) argmax bin:", int(np.argmax(row)),
          "mass:", float(row.max()))
