#!/usr/bin/env python
"""candlib.py -- low-DOF response-representation candidates (sealed opening
rule ``RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION.md``, sha256 84ce1de6...,
PI ruling 2026-09-13c section 4-7).

CALIBRATION SIDE ONLY.  No HBI/MCMC run, no real data, no mock dN/dX closure
number is computed, read or used anywhere in this module or its driver.

The object every family estimates is the SAME one the sealed rule defines:

    P(c | b, s, K)   rows summing to ONE exactly over the 29 observed bins,

with b = 16 latent ``ntrue_edges`` bins, c = 29 ``nhat_edges`` bins, s = 8
``snr_edges`` strata and K = 3 coarse-z blocks (z_DLA).  The delivered FIXED
object is the flat-quadrature average of the fitted conditional row over the
latent bin and the S/N stratum (implementation predeclaration I2b): the
analogue of the R1c bin-marginal width fix, and it uses NO calibration
occupancy.

Every implementation choice here was sealed BEFORE any fit in
``governance/response_review_2026-09-13/CANDIDATE_IMPLEMENTATION_CHOICES_
PREDECLARATION.md`` (sha256 2772c1c3...).

ENV: gpdla-hbi (numpy/scipy + jax).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_RESP = os.path.join(_REPO, "validation", "absorber_ladder", "response")
for _p in (_REPO, _RESP, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# sealed constants (predeclaration I1, I2, I4)
# ---------------------------------------------------------------------------
N_REF_U = 20.5            # u = N_true - 20.5
V_REF = 0.7               # v = log10(S/N) - 0.7
N_QUAD_U = 17             # flat interior quadrature points across a latent bin
N_QUAD_V = 5              # flat interior quadrature points across a stratum
SNR_TOP_CAP = 40.0        # finite cap for the open top stratum [7, inf)
SNR_BOT_FLOOR = 1.0       # finite floor for the (empty) stratum [0, 1)
RIDGE = 1.0e-6            # ridge per unit weight on coefficient vectors ONLY
BASE_FLOOR = 1.0e-10      # numerical floor on the E base row (I4)
N_RESTART = 4             # deterministic restarts, seeds 0..3
BOUNDARIES = (20.0, 20.3, 21.0)


# ===========================================================================
# geometry and events
# ===========================================================================
def load_geom(ops_path, pack_path):
    """Geometry from the A0 empirical-operator pack + the scanpack (for the
    response-cell maps and the frozen ``adopted_phi_ref``)."""
    z = np.load(ops_path, allow_pickle=True)
    pk = np.load(pack_path, allow_pickle=True)
    g = dict(
        ntrue=np.asarray(z["ntrue_edges"], float),
        nhat=np.asarray(z["nhat_edges"], float),
        snr=np.asarray(z["snr_edges"], float),
        zc=np.asarray(z["zc_edges"], float),
        zf=np.asarray(z["zf_edges"], float),
        kz2K=np.asarray(z["kz_to_K"], int),
        resp_snr=np.asarray(pk["resp_snr_edges"], float),
        resp_z=np.asarray(pk["resp_z_edges"], float),
        sig_floor=float(pk["resp_sig_floor"]),
        phi_ref=np.asarray(pk["adopted_phi_ref"], float))
    for key, src in (("ntrue", "ntrue_edges"), ("nhat", "nhat_edges"),
                     ("snr", "snr_edges"), ("zc", "zc_edges")):
        if not np.array_equal(g[key], np.asarray(pk[src], float)):
            raise SystemExit(f"geometry mismatch between ops and pack: {src}")
    g["B"] = len(g["ntrue"]) - 1
    g["C"] = len(g["nhat"]) - 1
    g["S"] = len(g["snr"]) - 1
    g["K"] = len(g["zc"]) - 1
    g["Kf"] = len(g["kz2K"])
    g["ccen"] = 0.5 * (g["nhat"][:-1] + g["nhat"][1:])
    g["bcen"] = 0.5 * (g["ntrue"][:-1] + g["ntrue"][1:])
    # response-cell maps, identical arithmetic to build_variants.main
    g["s2sr"] = np.clip(np.searchsorted(g["resp_snr"], g["snr"][:-1] + 1e-9,
                                        "right") - 1, 0, 2)
    g["K2zr"] = np.searchsorted(g["resp_z"],
                                0.5 * (g["zc"][:-1] + g["zc"][1:]),
                                "right") - 1
    return g


def load_events(path, geom):
    """Events + every index and covariate the families use (predeclaration
    I1: clipping of the 8 out-of-grid x_hat and of N_true > 22.4)."""
    d = np.load(path, allow_pickle=True)
    ev = {k: np.asarray(d[k]) for k in
          ("N_true", "snr", "zqso", "zdla", "dx", "xhat", "tid")}
    ev["provenance"] = str(d["provenance"])
    B, C, S, K = geom["B"], geom["C"], geom["S"], geom["K"]
    ev["b_i"] = np.clip(np.digitize(ev["N_true"], geom["ntrue"]) - 1, 0, B - 1)
    ev["s_i"] = np.clip(np.digitize(ev["snr"], geom["snr"]) - 1, 0, S - 1)
    ev["K_i"] = np.clip(np.digitize(ev["zdla"], geom["zc"]) - 1, 0, K - 1)
    c_raw = np.digitize(ev["xhat"], geom["nhat"]) - 1
    ev["n_clipped_c"] = int(((c_raw < 0) | (c_raw >= C)).sum())
    ev["c_i"] = np.clip(c_raw, 0, C - 1)
    ev["u"] = ev["N_true"] - N_REF_U
    ev["v"] = np.log10(np.maximum(ev["snr"], 1e-3)) - V_REF
    ev["fold"] = (np.asarray(ev["tid"], np.int64) % 2) == 0
    # response-cell index of each event (for the R0/R1c/E base rows)
    ev["isr"] = np.clip(np.digitize(ev["snr"], geom["resp_snr"]) - 1, 0, 2)
    ev["izr"] = np.clip(np.digitize(ev["zqso"], geom["resp_z"]) - 1, 0, 2)
    ev["n"] = len(ev["dx"])
    return ev


def row_index(b, s, K, geom):
    """Flat row id of (b, s, K)."""
    return (np.asarray(b) * geom["S"] + np.asarray(s)) * geom["K"] + \
        np.asarray(K)


def held_out_counts(ev, mask, geom):
    """(B, S, K, C) held-out count tensor."""
    out = np.zeros((geom["B"], geom["S"], geom["K"], geom["C"]))
    np.add.at(out, (ev["b_i"][mask], ev["s_i"][mask], ev["K_i"][mask],
                    ev["c_i"][mask]), 1.0)
    return out


# ===========================================================================
# design matrices (predeclaration I1, I4)
# ===========================================================================
def design(u, v, K, kind):
    """Coefficient-function design.

    kind 'asmall' : [1, u, u^2, v, 1[K=1], 1[K=2]]   (P = 6)
    kind 'e'      : [1, u, v, 1[K=1], 1[K=2]]        (P = 5)
    """
    u = np.asarray(u, float); v = np.asarray(v, float)
    K = np.asarray(K, int)
    one = np.ones_like(u)
    k1 = (K == 1).astype(float); k2 = (K == 2).astype(float)
    if kind == "asmall":
        cols = [one, u, u * u, v, k1, k2]
    elif kind == "e":
        cols = [one, u, v, k1, k2]
    elif kind == "full":            # B and C families (I6)
        cols = [one, u, u * u, v, k1, k2]
    else:
        raise ValueError(kind)
    return np.stack(cols, axis=-1)


def standardiser(X):
    """Column mean/sd of a design (the intercept left alone).

    A pure linear reparametrisation of the coefficient functions: it does not
    change the model, only the conditioning of the optimiser.  Stored with the
    fitted object so evaluation applies the identical transform.
    """
    mu = X.mean(axis=0).copy(); sd = X.std(axis=0).copy()
    mu[0] = 0.0; sd[0] = 1.0
    sd[sd < 1e-9] = 1.0
    return mu, sd


def apply_std(X, mu, sd):
    return (np.asarray(X, float) - mu) / sd


def quad_grid(geom):
    """Flat (u, v) quadrature nodes per (b, s) -- predeclaration I2b.

    Returns u_nodes (B, N_QUAD_U), v_nodes (S, N_QUAD_V).  Flat weights.
    """
    nt = geom["ntrue"]
    u_nodes = np.stack([
        np.linspace(nt[b], nt[b + 1], N_QUAD_U + 2)[1:-1] - N_REF_U
        for b in range(geom["B"])])
    se = geom["snr"]
    v_nodes = []
    for s in range(geom["S"]):
        lo = max(float(se[s]), SNR_BOT_FLOOR)
        hi = float(se[s + 1])
        hi = SNR_TOP_CAP if not np.isfinite(hi) else min(hi, SNR_TOP_CAP)
        hi = max(hi, lo * 1.0001)
        v_nodes.append(np.linspace(np.log10(lo), np.log10(hi),
                                   N_QUAD_V + 2)[1:-1] - V_REF)
    return u_nodes, np.stack(v_nodes)


def quad_design(geom, kind):
    """(B, S, K, Q, P) flat-quadrature design and Q = N_QUAD_U * N_QUAD_V."""
    un, vn = quad_grid(geom)
    B, S, K = geom["B"], geom["S"], geom["K"]
    Q = N_QUAD_U * N_QUAD_V
    P = design(np.zeros(1), np.zeros(1), np.zeros(1, int), kind).shape[-1]
    X = np.zeros((B, S, K, Q, P))
    for b in range(B):
        for s in range(S):
            uu, vv = np.meshgrid(un[b], vn[s], indexing="ij")
            uu = uu.ravel(); vv = vv.ravel()
            for k in range(K):
                X[b, s, k] = design(uu, vv, np.full(Q, k), kind)
    return X


def occupancy_design(ev, mask, geom, kind):
    """Per-cell list of the TRAINING events' own design rows (the occupancy
    -weighted marginalisation used ONLY in the section 4D audit, I2c)."""
    idx = np.where(mask)[0]
    X = design(ev["u"][idx], ev["v"][idx], ev["K_i"][idx], kind)
    rid = row_index(ev["b_i"][idx], ev["s_i"][idx], ev["K_i"][idx], geom)
    return X, rid


# ===========================================================================
# identifiability canonicalisation (predeclaration I4)
# ===========================================================================
def canonicalise(alpha, phi, ccen):
    """Put (alpha (R,P), phi (R,C)) in the sealed canonical gauge.

    The likelihood depends only on M = alpha^T phi (P, C) modulo an additive
    row-constant (softmax / row-normalisation invariance) and modulo any
    GL(R) reparametrisation.  The canonical representative is:
      * phi_r sum-to-zero over c;
      * phi orthonormal rows (phi phi^T = I_R);
      * the R x R mixing fixed by the SVD of the sum-zero part of M, so the
        rows are ordered by singular value;
      * the sign of phi_r fixed by <phi_r, e_r> > 0 with e_1 = (c - cbar),
        e_2 = (c - cbar)^2 - mean.
    """
    alpha = np.asarray(alpha, float); phi = np.asarray(phi, float)
    R, C = phi.shape
    M = alpha.T @ phi                                  # (P, C)
    M = M - M.mean(axis=1, keepdims=True)              # gauge: sum-zero rows
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    U, sv, Vt = U[:, :R], sv[:R], Vt[:R]
    e = np.stack([ccen - ccen.mean(),
                  (ccen - ccen.mean()) ** 2 - ((ccen - ccen.mean()) ** 2).mean()
                  ])[:R]
    for r in range(R):
        if float(Vt[r] @ e[r]) < 0:
            Vt[r] = -Vt[r]; U[:, r] = -U[:, r]
    return (U * sv[None, :]).T, Vt, sv


def nominal_dof_lowrank(P, C, R):
    """Identifiable nominal DOF of a rank-R softmax/multiplicative low-rank
    row model with P covariate columns and C observed bins:
    R * (P + (C - 1) - R)  -- the rank-R manifold inside the sum-zero
    (C-1)-dimensional observed-bin space, modulo GL(R)."""
    return int(R * (P + (C - 1) - R))
