"""REVIEW-ONLY (Phase A) — independent non-negative Poisson MLE for the
pad + window + FP linear fold model, plus the injection-truth constructor.

FRESH implementation for the adversarial review of the claim
"Δdev = 41 ↔ 0.6σ — a wrong model is undetectable".  It deliberately does NOT
import or copy the archived probe's fitters (prof2.ProfEM / core.fit): the
solver below is an independently derived multiplicative-EM warm start followed
by L-BFGS-B with an analytic gradient in attributed-count coordinates.
Because the Poisson NLL with mu linear in non-negative amplitudes is convex,
the optimum VALUE is unique — agreement with the probe validates both codes.

Model (identical estimand to the probe, by construction from the pack):
    data index i = (c, s) flattened;  live cells = dX > 0 repeated over c
    mu[k, i] = sum_b M[k, i, b] * f[b, k]  +  coefv[k, i] * lam[i]
    M from the PRODUCTION fold operator A[c,k,s,b] (cached by build_cache.py)
    coefv[k, (c,s)] = fp_w_sightline_ratio * fp_E_alloc[k, s]  (log_t = 0)

Fits:
    F1: pad + window + FP all free (non-negative)
    F3: pad EXACTLY 0 (cleaner than the probe's total=1e-9), window + FP free

Deviance: 2 * sum_live [ y*log(y/mu) - (y - mu) ]   (y>0 log terms only).
"""
import os
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
        "PRESERVED_2026-07-28_small_artifacts/modelA_packs/modelA_pack_2lpt0_v11.npz")


# ---------------------------------------------------------------------------
# Problem container (numpy only — safe to fork into worker processes)
# ---------------------------------------------------------------------------
class Problem:
    def __init__(self, packpath=PACK):
        # pack loader is numpy-only; the JAX-built operator comes from cache
        import sys
        sys.path.insert(0, "/home/mfho/wt_review_phaseA")
        from CDDF_analysis.hbi_mcmc.pack import load_pack, coarsen_basis
        pk = coarsen_basis(load_pack(packpath), 0.1, pad_floor=19.0)
        A = np.load(os.path.join(HERE, "cache", "A_2lpt0_both.npy"))
        self.f_window_truth = np.load(os.path.join(HERE, "cache", "truthf_2lpt0.npy"))
        self.pack = pk
        C, Kf, S, B = A.shape
        self.C, self.Kf, self.S, self.B = C, Kf, S, B
        self.CS = C * S
        self.npad = int(pk.n_pad_bins)
        dX = np.asarray(pk.dX, float)                                   # (Kf,S)
        self.live = np.repeat((dX > 0)[:, None, :], C, axis=1).reshape(Kf, C * S)
        M = np.ascontiguousarray(A.transpose(1, 0, 2, 3).reshape(Kf, C * S, B))
        self.M = M * self.live[:, :, None]
        coeff = float(pk.fp_w_sightline_ratio) * np.asarray(pk.fp_E_alloc, float)
        self.coefv = np.repeat(coeff[:, None, :], C, axis=1).reshape(Kf, C * S) * self.live
        self.n0 = np.asarray(pk.fp_counts, float).reshape(C * S)
        edges = np.asarray(pk.ntrue_edges, float)
        self.Nc = 0.5 * (edges[:-1] + edges[1:])
        self.sA = self.M.sum(axis=1).T                                  # (B,Kf)
        self.scf = self.coefv.sum(axis=0)                               # (CS,)
        self.fp_live = self.scf > 0
        self.n_live = int(self.live.sum())
        # normalized (unit-column) operators: params become ATTRIBUTED COUNTS
        sAT = self.sA.T[:, None, :]                                     # (Kf,1,B)
        self.Au = np.where(sAT > 0, self.M / np.maximum(sAT, 1e-300), 0.0)
        self.Cu = np.where(self.scf[None, :] > 0,
                           self.coefv / np.maximum(self.scf, 1e-300)[None, :], 0.0)

    # -- injection truth, independent re-derivation of the probe's spec ------
    def build_truth(self, T_A, T_B):
        """Truth (f, lam, mu): window rows = the pack's own truth in the fold
        coordinate; pad rows = per-z-column power-law continuation of log f
        fitted on the 5 lowest WINDOW bins, block-rescaled so the pad's
        expected survey count is T_A; FP shape = pack fp_counts (loa-0 shape)
        on fp-live columns, scaled so the expected FP survey count is T_B."""
        f = self.f_window_truth.copy()                                  # (B,Kf)
        npad = self.npad
        lo, hi = npad, npad + 5
        for k in range(self.Kf):
            col = f[lo:hi, k]
            m = col > 0
            yv = np.log(np.clip(col, 1e-300, None))
            if m.sum() >= 2:
                slope, icpt = np.polyfit(self.Nc[lo:hi][m], yv[m], 1)
            else:
                slope, icpt = -1.5, 0.0
            f[:npad, k] = np.exp(icpt + slope * self.Nc[:npad])
        cur = float((f[:npad] * self.sA[:npad]).sum())
        f[:npad] *= (T_A / cur) if cur > 0 else 0.0
        sh = self.n0 * self.fp_live
        if sh.sum() <= 0:
            sh = self.fp_live.astype(float)
        xf = T_B * sh / sh.sum()
        lam = np.where(self.fp_live, xf / np.maximum(self.scf, 1e-300), 0.0)
        mu = np.einsum("kib,bk->ki", self.M, f) + self.coefv * lam[None, :]
        mu = np.where(self.live, np.clip(mu, 0.0, None), 0.0)
        return f, lam, mu

    def deviance(self, y, mu):
        yl = y[self.live]
        ml = np.clip(mu[self.live], 1e-300, None)
        t = np.zeros_like(yl)
        nz = yl > 0
        t[nz] = yl[nz] * np.log(yl[nz] / ml[nz])
        return float(2.0 * np.sum(t - (yl - ml)))


# ---------------------------------------------------------------------------
# Independent solver: EM warm start + L-BFGS-B polish (analytic gradient)
# ---------------------------------------------------------------------------
def fit(P, y, pad_free=True, n_em=400, maxiter=30000, ftol=1e-17, gtol=1e-9,
        u0=None, v0=None):
    """Global non-negative Poisson MLE.

    pad_free=False pins the 5*Kf pad amplitudes to EXACTLY 0 (config F3).
    Parameters are attributed counts u[b,k] (signal) and v[i] (FP), so every
    design column sums to 1 and the problem is well scaled.
    Returns dict with dev, T_A, T_W, T_B, f, lam, mu, convergence info.
    """
    B, Kf, CS, npad = P.B, P.Kf, P.CS, P.npad
    rows_free = np.ones(B, bool)
    if not pad_free:
        rows_free[:npad] = False
    ok_f = rows_free[:, None] & (P.sA > 0)                              # (B,Kf)
    ok_l = P.fp_live.copy()                                             # (CS,)
    nf, nl = int(ok_f.sum()), int(ok_l.sum())
    ylive_sum = max(float(y[P.live].sum()), 1.0)

    def mu_of(u, v):
        return np.einsum("kib,bk->ki", P.Au, u) + P.Cu * v[None, :]

    # --- init: equal split of the observed total over free columns
    if u0 is None:
        u = np.zeros((B, Kf)); u[ok_f] = ylive_sum / max(nf + nl, 1)
    else:
        u = np.array(u0, float)
        u[~ok_f] = 0.0
        u[ok_f] = np.maximum(u[ok_f], 1e-8)
    if v0 is None:
        v = np.zeros(CS); v[ok_l] = ylive_sum / max(nf + nl, 1)
    else:
        v = np.array(v0, float)
        v[~ok_l] = 0.0
        v[ok_l] = np.maximum(v[ok_l], 1e-8)

    # --- multiplicative EM (monotone; my own two-line derivation:
    #     u_j <- u_j * sum_i col_ij * y_i / mu_i  for unit-sum columns)
    for _ in range(n_em):
        m = np.clip(mu_of(u, v), 1e-300, None)
        r = np.where(P.live, y / m, 0.0)
        u = np.where(ok_f, u * np.einsum("kib,ki->bk", P.Au, r), u)
        v = np.where(ok_l, v * (P.Cu * r).sum(axis=0), v)

    # --- L-BFGS-B polish
    def unpack(x):
        uu = np.zeros((B, Kf)); uu[ok_f] = x[:nf]
        vv = np.zeros(CS); vv[ok_l] = x[nf:]
        return uu, vv

    def obj(x):
        uu, vv = unpack(x)
        m = np.clip(mu_of(uu, vv), 1e-300, None)
        nll = float(np.sum((m - np.where(y > 0, y * np.log(m), 0.0))[P.live]))
        w = np.where(P.live, 1.0 - y / m, 0.0)
        gu = np.einsum("kib,ki->bk", P.Au, w)[ok_f]
        gv = (P.Cu * w).sum(axis=0)[ok_l]
        return nll, np.concatenate([gu, gv])

    x0 = np.concatenate([u[ok_f], v[ok_l]])
    res = minimize(obj, x0, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * x0.size,
                   options=dict(maxiter=maxiter, maxfun=2 * maxiter,
                                ftol=ftol, gtol=gtol, maxcor=40))
    u, v = unpack(res.x)
    mu = mu_of(u, v)
    dev = P.deviance(y, mu)
    f = np.where(P.sA > 0, u / np.maximum(P.sA, 1e-300), 0.0)
    lam = np.where(P.fp_live, v / np.maximum(P.scf, 1e-300), 0.0)
    return dict(u=u, v=v, f=f, lam=lam, mu=mu, dev=dev, nll=float(res.fun),
                T_A=float(u[:npad].sum()), T_W=float(u[npad:].sum()),
                T_B=float(v.sum()), nit=int(res.nit), nfev=int(res.nfev),
                success=bool(res.success), msg=str(res.message))


def grad_check(P, seed=0):
    """Finite-difference check of the analytic gradient at a random point."""
    rng = np.random.default_rng(seed)
    B, Kf, CS = P.B, P.Kf, P.CS
    ok_f = P.sA > 0
    ok_l = P.fp_live
    nf, nl = int(ok_f.sum()), int(ok_l.sum())
    y = rng.poisson(10.0, size=(Kf, CS)) * P.live

    def obj(x):
        u = np.zeros((B, Kf)); u[ok_f] = x[:nf]
        v = np.zeros(CS); v[ok_l] = x[nf:]
        m = np.clip(np.einsum("kib,bk->ki", P.Au, u) + P.Cu * v[None, :],
                    1e-300, None)
        nll = float(np.sum((m - np.where(y > 0, y * np.log(m), 0.0))[P.live]))
        w = np.where(P.live, 1.0 - y / m, 0.0)
        return nll, np.concatenate([np.einsum("kib,ki->bk", P.Au, w)[ok_f],
                                    (P.Cu * w).sum(axis=0)[ok_l]])

    x = rng.uniform(5.0, 50.0, nf + nl)
    f0, g = obj(x)
    idx = rng.choice(nf + nl, 25, replace=False)
    errs = []
    for j in idx:
        h = 1e-5 * max(abs(x[j]), 1.0)
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        fp, _ = obj(xp); fm, _ = obj(xm)
        num = (fp - fm) / (2 * h)
        errs.append(abs(num - g[j]) / max(abs(num), abs(g[j]), 1e-12))
    return float(np.max(errs))
