#!/usr/bin/env python
"""families.py -- the candidate response families of the sealed ladder.

Level 0 references : R0, R1c (parametric, from ``respfit``), R1d-raw (Jeffreys).
Level 1 candidates : A-small (low-rank multinomial logit), E (low-rank
                     residual correction to the R1c kernel).
Level 2 candidates : B (two-component mixture), C (monotone quantile spline).
Level 3 candidate  : D (regularised empirical matrix, 3 shrinkage hypers).

EVERY family delivers the SAME object: rows P(c | b, s, K) summing to one
exactly over the 29 observed bins.  Nothing here reads a closure number.

ENV: gpdla-hbi.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import candlib as CL

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _HAVE_JAX = True
except Exception:                                          # pragma: no cover
    _HAVE_JAX = False


# ===========================================================================
# generic JAX fitting harness
# ===========================================================================
def _fit_lbfgs(loss_fn, x0_list, maxiter=2000):
    """L-BFGS on ``loss_fn`` (jax scalar) from each start; best kept."""
    vg = jax.jit(jax.value_and_grad(loss_fn))

    def f(x):
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, float)

    best = None
    for x0 in x0_list:
        r = minimize(f, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, maxfun=maxiter * 2,
                                  ftol=1e-15, gtol=1e-11, maxcor=50))
        if best is None or r.fun < best.fun:
            best = r
    return best


def _rng_starts(shape_flat, n, scale, seed0=0):
    out = [np.zeros(shape_flat)]
    for k in range(1, n):
        rs = np.random.RandomState(seed0 + k)
        out.append(rs.normal(scale=scale, size=shape_flat))
    return out


# ===========================================================================
# LOW-RANK families: A-small (logbase = 0) and E (logbase = log M_R1c)
# ===========================================================================
class LowRank:
    """P(c|.) = normalise_c[ base(c) * exp( sum_r a_r(x) phi_r(c) ) ].

    base == 1 gives the A-small multinomial logit (exact softmax
    normalisation); base = the R1c kernel row gives E (exact explicit row
    normalisation).  R = 2 by the sealed rule.
    """

    def __init__(self, name, kind, R=2, use_base=False, level=1):
        self.name = name
        self.kind = kind
        self.R = int(R)
        self.use_base = bool(use_base)
        self.level = level

    # ---- forward -------------------------------------------------------
    @staticmethod
    def _logp(alpha, phi, X, logbase):
        eta = (X @ alpha.T) @ phi                     # (n, C)
        if logbase is not None:
            eta = eta + logbase
        return eta - jax.scipy.special.logsumexp(eta, axis=1, keepdims=True)

    def unpack(self, x, P, C):
        R = self.R
        return x[:R * P].reshape(R, P), x[R * P:].reshape(R, C)

    # ---- fit -----------------------------------------------------------
    def fit(self, X, c_i, logbase, w, C, verbose=False):
        """ML fit of the rank-R model.

        Conditioning (numerical, disclosed): the design is standardised and
        the deterministic start is the rank-R truncated SVD of the
        UNCONSTRAINED (full-rank, CONVEX) multinomial-logit fit of the same
        data.  The three further starts are seeded perturbations of it.  The
        likelihood is invariant under GL(R) and under an additive row
        constant, so the raw optimum is a manifold; the reported object is the
        canonical representative (``candlib.canonicalise``), and the spread of
        the four starts' training objectives is reported as the convergence
        witness.
        """
        P = X.shape[1]
        xmu, xsd = CL.standardiser(X)
        Xs = CL.apply_std(X, xmu, xsd)
        Xj = jnp.asarray(Xs); cj = jnp.asarray(c_i)
        wj = jnp.asarray(w)
        lb = None if logbase is None else jnp.asarray(logbase)
        sw = float(np.sum(w))
        ridge = CL.RIDGE
        ar = jnp.arange(Xs.shape[0])

        def loss_full(x):
            M = x.reshape(P, C)
            eta = Xj @ M
            if lb is not None:
                eta = eta + lb
            lp = eta - jax.scipy.special.logsumexp(eta, axis=1, keepdims=True)
            return -jnp.sum(wj * lp[ar, cj]) / sw

        rf = _fit_lbfgs(loss_full, [np.zeros(P * C)], maxiter=800)
        Mh = np.asarray(rf.x, float).reshape(P, C)
        Mh = Mh - Mh.mean(axis=1, keepdims=True)
        U, sv0, Vt = np.linalg.svd(Mh, full_matrices=False)
        R = self.R
        a0 = (U[:, :R] * sv0[:R]).T
        ph0 = Vt[:R]
        x_svd = np.concatenate([a0.ravel(), ph0.ravel()])

        def loss(x):
            a, ph = self.unpack(x, P, C)
            lp = self._logp(a, ph, Xj, lb)
            return -jnp.sum(wj * lp[ar, cj]) / sw + ridge * jnp.sum(a ** 2)

        starts = [x_svd]
        for k in range(1, CL.N_RESTART):
            rs = np.random.RandomState(k)
            starts.append(x_svd * (1.0 + 0.2 * rs.normal(size=x_svd.shape))
                          + 0.02 * rs.normal(size=x_svd.shape))
        objs = []
        best = None
        for i, x0 in enumerate(starts):
            r = _fit_lbfgs(loss, [x0], maxiter=(3000 if i == 0 else 1500))
            objs.append(float(r.fun))
            if best is None or r.fun < best.fun:
                best = r
        a, ph = self.unpack(np.asarray(best.x, float), P, C)
        return dict(alpha=a, phi=ph, P=P, C=C, R=self.R, kind=self.kind,
                    use_base=self.use_base, train_loss=float(best.fun),
                    nit=int(best.nit), name=self.name, xmu=xmu, xsd=xsd,
                    full_rank_loss=float(rf.fun),
                    start_objectives=objs,
                    start_spread=float(max(objs) - min(objs)),
                    svd_values=sv0.tolist())

    # ---- evaluate ------------------------------------------------------
    def logp_rows(self, par, X, logbase):
        Xs = CL.apply_std(X, par.get("xmu", 0.0), par.get("xsd", 1.0))
        eta = (Xs @ par["alpha"].T) @ par["phi"]
        if logbase is not None:
            eta = eta + logbase
        m = eta.max(axis=-1, keepdims=True)
        e = np.exp(eta - m)
        return np.log(e / e.sum(axis=-1, keepdims=True))

    def nominal_dof(self, par, base_dof=0):
        return CL.nominal_dof_lowrank(par["P"], par["C"], par["R"]) + base_dof


# ===========================================================================
# B -- two-component mixture (Level 2)
# ===========================================================================
_GL5_X = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                   0.5384693101056831, 0.9061798459386640])
_GL5_W = np.array([0.2369268850561891, 0.4786286704993665,
                   0.5688888888888889, 0.4786286704993665,
                   0.2369268850561891])


def bin_quad_nodes(nhat_edges):
    """(C, 5) Gauss-Legendre nodes and (C, 5) weights on the observed bins."""
    ne = np.asarray(nhat_edges, float)
    lo, hi = ne[:-1], ne[1:]
    mid = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
    x = mid[:, None] + half[:, None] * _GL5_X[None, :]
    w = half[:, None] * _GL5_W[None, :]
    return x, w


class Mixture:
    """P(c|.) proportional to the mass in bin c of

        pi * SplitNormal(x; N + d1, w1, kappa) + (1 - pi) * N(x; N + d1 + d2,
                                                              w2)

    renormalised over the 29 observed bins (exact row normalisation).  Each of
    (logit pi, d1, log w1, kappa, d2, log w2) is linear in the design
    [1, u, u^2, v, K1, K2]: location, width, fraction and ONE skew parameter
    smooth in the covariates, as the sealed section 4 specifies.

    IMPLEMENTATION NOTE (disclosed; decided before any B number was seen): the
    skewed component is a TWO-PIECE (split) normal rather than a skew-normal.
    Both carry exactly one skew parameter; the split normal has a CLOSED-FORM
    CDF, so the bin masses are exact instead of quadrature and the fit is ~30x
    cheaper.  The point of family B is the MIXTURE (a narrow core plus a broad
    tail, which no single skew-normal can be), not the particular skewed
    kernel -- the skew-normal kernel is precisely the one R0/R1c already use
    and that the first ladder closed negative.
    """

    NPAR = 6

    def __init__(self, name="B", level=2):
        self.name = name
        self.kind = "full"
        self.level = level

    @staticmethod
    def _split_normal_cdf(x, m, wm, wp):
        s = wm + wp
        left = (2.0 * wm / s) * jax.scipy.stats.norm.cdf((x - m) / wm)
        right = wm / s + (2.0 * wp / s) * (
            jax.scipy.stats.norm.cdf((x - m) / wp) - 0.5)
        return jnp.where(x < m, left, right)

    @staticmethod
    def _dens(beta, X, Nabs, edges, _unused=None):
        t = X @ beta.T                                 # (n, 6)
        pi = jax.nn.sigmoid(t[:, 0])
        m1 = Nabs + t[:, 1]
        w1 = jnp.exp(jnp.clip(t[:, 2], -4.0, 2.0))
        ka = jnp.clip(t[:, 3], -3.0, 3.0)
        m2 = m1 + t[:, 4]
        w2 = jnp.exp(jnp.clip(t[:, 5], -4.0, 2.0))
        wm = w1 * jnp.exp(-0.5 * ka); wp = w1 * jnp.exp(0.5 * ka)
        e = edges[None, :]                             # (1, C+1)
        F1 = Mixture._split_normal_cdf(e, m1[:, None], wm[:, None],
                                       wp[:, None])
        F2 = jax.scipy.stats.norm.cdf((e - m2[:, None]) / w2[:, None])
        mass = pi[:, None] * jnp.diff(F1, axis=1) + \
            (1.0 - pi[:, None]) * jnp.diff(F2, axis=1)
        mass = jnp.clip(mass, 1e-300, None)
        return jnp.log(mass) - jnp.log(jnp.sum(mass, axis=1, keepdims=True))

    def fit(self, X, c_i, Nabs, w, geom, verbose=False):
        P = X.shape[1]
        xmu, xsd = CL.standardiser(X)
        ed = jnp.asarray(np.asarray(geom["nhat"], float))
        Xj = jnp.asarray(CL.apply_std(X, xmu, xsd))
        cj = jnp.asarray(c_i); wj = jnp.asarray(w)
        Nj = jnp.asarray(Nabs)
        sw = float(np.sum(w))
        ridge = CL.RIDGE
        ar = jnp.arange(X.shape[0])

        def loss(x):
            b = x.reshape(self.NPAR, P)
            lp = self._dens(b, Xj, Nj, ed)
            return -jnp.sum(wj * lp[ar, cj]) / sw + ridge * jnp.sum(b ** 2)

        nx = self.NPAR * P
        x0 = np.zeros((self.NPAR, P))
        x0[0, 0] = 1.0          # pi ~ 0.73 on the skewed component
        x0[2, 0] = np.log(0.12)
        x0[4, 0] = 0.10
        x0[5, 0] = np.log(0.35)
        starts = [x0.ravel()]
        for k in range(1, CL.N_RESTART):
            rs = np.random.RandomState(k)
            starts.append(x0.ravel() + 0.15 * rs.normal(size=nx))
        objs = []; best = None
        for i, xs in enumerate(starts):
            r = _fit_lbfgs(loss, [xs], maxiter=(3000 if i == 0 else 1500))
            objs.append(float(r.fun))
            if best is None or r.fun < best.fun:
                best = r
        return dict(beta=np.asarray(best.x, float).reshape(self.NPAR, P), P=P,
                    train_loss=float(best.fun), nit=int(best.nit),
                    name=self.name, kind=self.kind, xmu=xmu, xsd=xsd,
                    start_objectives=objs,
                    start_spread=float(max(objs) - min(objs)))

    def logp_rows(self, par, X, Nabs, geom):
        Xs = CL.apply_std(X, par.get("xmu", 0.0), par.get("xsd", 1.0))
        return np.asarray(self._dens(jnp.asarray(par["beta"]),
                                     jnp.asarray(Xs), jnp.asarray(Nabs),
                                     jnp.asarray(np.asarray(geom["nhat"],
                                                            float))), float)

    def nominal_dof(self, par):
        return int(self.NPAR * par["P"])


# ===========================================================================
# C -- monotone quantile spline (Level 2)
# ===========================================================================
class QuantileSpline:
    """Cumulative-link rows with a MONOTONE piecewise-linear spline on the
    observed axis:  F(edge_j) = sigmoid(theta_j),  F(edge_0) = 0,
    F(edge_C) = 1,  theta_j = theta0(x) + sum_m softplus(g_m(x)) psi_m(edge_j)
    with psi_m the m-th cumulative ramp of ``n_knot`` equal knots.  Monotone by
    construction, so p_c = F_{c+1} - F_c >= 0 and the row sums to one exactly.
    Each of theta0 and g_1..g_M is linear in the design.
    """

    def __init__(self, name="C", n_knot=6, level=2):
        self.name = name
        self.kind = "full"
        self.n_knot = int(n_knot)
        self.level = level
        self.NPAR = self.n_knot + 1

    def psi(self, geom):
        """(C-1, n_knot) cumulative ramps at the INTERIOR observed edges."""
        ne = np.asarray(geom["nhat"], float)
        ed = ne[1:-1]                                   # interior edges
        kn = np.linspace(ne[0], ne[-1], self.n_knot + 1)
        out = np.zeros((len(ed), self.n_knot))
        for m in range(self.n_knot):
            out[:, m] = np.clip((ed - kn[m]) / (kn[m + 1] - kn[m]), 0.0, 1.0)
        return out

    @staticmethod
    def _logp(beta, X, Psi, Nabs, loc_scale):
        t = X @ beta.T                                  # (n, NPAR)
        th0 = t[:, 0] - loc_scale * Nabs
        g = jax.nn.softplus(jnp.clip(t[:, 1:], -20.0, 20.0))   # (n, M)
        th = th0[:, None] + g @ Psi.T                   # (n, C-1)
        F = jax.nn.sigmoid(th)
        one = jnp.ones((F.shape[0], 1))
        zero = jnp.zeros((F.shape[0], 1))
        Ff = jnp.concatenate([zero, F, one], axis=1)
        p = jnp.clip(jnp.diff(Ff, axis=1), 1e-12, None)
        return jnp.log(p) - jnp.log(jnp.sum(p, axis=1, keepdims=True))

    def fit(self, X, c_i, Nabs, w, geom, loc_scale=8.0):
        P = X.shape[1]
        xmu, xsd = CL.standardiser(X)
        Psi = self.psi(geom)
        Xj = jnp.asarray(CL.apply_std(X, xmu, xsd))
        cj = jnp.asarray(c_i); wj = jnp.asarray(w)
        Pj = jnp.asarray(Psi); Nj = jnp.asarray(Nabs - CL.N_REF_U)
        sw = float(np.sum(w)); ridge = CL.RIDGE * sw

        def loss(x):
            b = x.reshape(self.NPAR, P)
            lp = self._logp(b, Xj, Pj, Nj, loc_scale)
            ll = jnp.sum(wj * lp[jnp.arange(lp.shape[0]), cj])
            return -ll / sw + ridge * jnp.sum(b ** 2) / sw

        nx = self.NPAR * P
        starts = _rng_starts(nx, CL.N_RESTART, 0.05)
        x0 = np.zeros((self.NPAR, P))
        x0[0, 0] = -3.0
        x0[1:, 0] = 1.0
        starts[0] = x0.ravel()
        objs = []; best = None
        for i, xs in enumerate(starts):
            r = _fit_lbfgs(loss, [xs], maxiter=(2000 if i == 0 else 1000))
            objs.append(float(r.fun))
            if best is None or r.fun < best.fun:
                best = r
        return dict(beta=np.asarray(best.x, float).reshape(self.NPAR, P), P=P,
                    loc_scale=float(loc_scale), n_knot=self.n_knot,
                    train_loss=float(best.fun), nit=int(best.nit),
                    name=self.name, kind=self.kind, xmu=xmu, xsd=xsd,
                    start_objectives=objs,
                    start_spread=float(max(objs) - min(objs)))

    def logp_rows(self, par, X, Nabs, geom):
        Psi = self.psi(geom)
        return np.asarray(self._logp(jnp.asarray(par["beta"]),
                                     jnp.asarray(CL.apply_std(
                                         X, par.get("xmu", 0.0),
                                         par.get("xsd", 1.0))),
                                     jnp.asarray(Psi),
                                     jnp.asarray(Nabs - CL.N_REF_U),
                                     par["loc_scale"]), float)

    def nominal_dof(self, par):
        return int(self.NPAR * par["P"])


# ===========================================================================
# R1d-raw -- the saturated Jeffreys row (Level 0 reference)
# ===========================================================================
def jeffreys_rows(counts, alpha=0.5, fallback=None):
    """(B,S,K,C) counts -> Jeffreys rows (n + alpha) / (N + alpha*C).

    Rows with NO events fall back to ``fallback`` (the global row at that b)
    if given, else to the uniform row; the fallback is never used in any
    held-out score (those rows have no held-out events either) and exists only
    so the delivered tensor is complete.
    """
    counts = np.asarray(counts, float)
    C = counts.shape[-1]
    tot = counts.sum(axis=-1, keepdims=True)
    p = (counts + alpha) / (tot + alpha * C)
    if fallback is not None:
        empty = (tot[..., 0] <= 0)
        p[empty] = fallback[empty]
    return p


# ===========================================================================
# D -- regularised empirical matrix with 3 shrinkage hyper-parameters (L3)
# ===========================================================================
def _offset_align(counts, geom):
    """Rows re-expressed on the OFFSET axis relative to the observed bin that
    holds the latent bin centre, so adjacent-b rows may be pooled coherently.
    Returns (B,S,K,2C-1) offset counts and the per-b anchor c0."""
    B, S, K, C = counts.shape
    c0 = np.clip(np.digitize(geom["bcen"], geom["nhat"]) - 1, 0, C - 1)
    out = np.zeros((B, S, K, 2 * C - 1))
    for b in range(B):
        j = np.arange(C) - c0[b] + (C - 1)
        out[b, :, :, j] = counts[b].transpose(2, 0, 1)
    return out, c0


def _offset_to_c(off, c0, C):
    """Inverse of ``_offset_align`` (mass off the observed grid is dropped)."""
    B, S, K, _ = off.shape
    out = np.zeros((B, S, K, C))
    for b in range(B):
        j = np.arange(C) - c0[b] + (C - 1)
        out[b] = off[b][:, :, j]
    return out


def fit_D(counts, geom, lam_b, lam_s, lam_0, eps=1e-3):
    """p proportional to n + lam_b * (offset-aligned b-neighbour rows)
    + lam_s * (s-neighbour rows) + lam_0 * (global row at this b) + eps.

    The three lambdas are the ONLY hyper-parameters; they are chosen by inner
    CV on the training fold (driver).  Occupancy dependence is explicit: the
    neighbour and global pools are COUNT pools, so this family is the one the
    section 4D audit is expected to flag.
    """
    counts = np.asarray(counts, float)
    B, S, K, C = counts.shape
    off, c0 = _offset_align(counts, geom)
    nb = np.zeros_like(off)
    for b in range(B):
        acc = np.zeros_like(off[b]); m = 0
        for bb in (b - 1, b + 1):
            if 0 <= bb < B:
                acc += off[bb]; m += 1
        nb[b] = acc / max(m, 1)
    nb_c = _offset_to_c(nb, c0, C)
    ns = np.zeros_like(counts)
    for s in range(S):
        acc = np.zeros_like(counts[:, 0]); m = 0
        for ss in (s - 1, s + 1):
            if 0 <= ss < S:
                acc += counts[:, ss]; m += 1
        ns[:, s] = acc / max(m, 1)
    glob = counts.sum(axis=(1, 2), keepdims=True)
    glob = glob / np.maximum(glob.sum(axis=-1, keepdims=True), 1e-300)
    n_row = counts.sum(axis=-1, keepdims=True)
    nb_n = nb_c / np.maximum(nb_c.sum(axis=-1, keepdims=True), 1e-300)
    ns_n = ns / np.maximum(ns.sum(axis=-1, keepdims=True), 1e-300)
    num = counts + lam_b * nb_n + lam_s * ns_n + lam_0 * glob + eps
    p = num / num.sum(axis=-1, keepdims=True)
    w_own = (n_row[..., 0] /
             np.maximum(n_row[..., 0] + lam_b + lam_s + lam_0 + eps * C, 1e-12))
    return p, w_own
