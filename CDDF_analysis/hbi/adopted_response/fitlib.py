"""Shared machinery for the PI-ruled diagnostics D1/D2 (env gpdla).

Everything stays inside the model's own family: per-sub-bin (mu, sigma, skew)
of dx, converted with the committed _moment_to_skewnormal, likelihood
optionally TRUNCATED at dx >= t_i (t_i = 19.5 - N_true,i per event); then
degree-d WLS moment polynomials per response cell in u = N - N_ref
(the committed two-step structure, with sample moments replaced by ML).
"""
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skewnorm

sys.path.insert(0, "/home/mfho/wt_forward_2026_08")
from CDDF_analysis.hbi.znz_kernel import (_moment_to_skewnormal,
                                          _moment_to_skewnormal_vec)

SKEW_CAP = 0.95
SIG_MIN = 0.02


def _nll(params, dx, t, truncated, w=None):
    m, s, sk = params
    if not (SIG_MIN < s < 1.0 and -SKEW_CAP < sk < SKEW_CAP):
        return 1e9
    xi, om, al = _moment_to_skewnormal(m, s, sk)
    ll = skewnorm.logpdf(dx, al, loc=xi, scale=om)
    if not np.all(np.isfinite(ll)):
        return 1e9
    if truncated:
        sf = skewnorm.sf(t, al, loc=xi, scale=om)
        ll = ll - np.log(np.clip(sf, 1e-12, 1.0))
    return -(ll.sum() if w is None else (w * ll).sum())


def fit_subbin(dx, t, truncated, w=None):
    """ML (mu, sigma, skew) for one sub-bin; returns (params, ok)."""
    if w is None:
        m0 = float(dx.mean())
        s0 = float(max(dx.std(ddof=1), SIG_MIN * 1.5))
        d = dx - m0
    else:
        ws = max(w.sum(), 1e-9)
        m0 = float((w * dx).sum() / ws)
        d = dx - m0
        s0 = float(max(np.sqrt((w * d ** 2).sum() / ws), SIG_MIN * 1.5))
    sk0 = float(np.clip((d ** 3).mean() / max(d.std(), 1e-6) ** 3, -0.9, 0.9))
    best = None
    for x0 in ([m0, s0, sk0], [m0, s0 * 1.3, 0.5 * sk0]):
        r = minimize(_nll, x0, args=(dx, t, truncated, w), method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-5, fatol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    p = best.x
    ok = (best.fun < 1e8 and SIG_MIN * 1.01 < p[1] < 0.95
          and abs(p[2]) < SKEW_CAP * 0.999)
    return p, bool(ok), float(best.fun)


def subbin_moments(N, dx, edges, min_n, mode, weights=None, fixed_bins=None):
    """Per-sub-bin moment estimates. mode: 'sample'|'ml'|'ml_trunc'.
    weights: optional per-event weights (bootstrap multiplicities; ml modes
    only). fixed_bins: optional explicit list of bin indices to fit (the
    adopted estimator's frozen anchor set), bypassing the min_n selection.
    Returns list of dict(rows) with center, n, mu, sig, skew, ok."""
    rows = []
    ib = np.digitize(N, edges) - 1
    bins = (range(len(edges) - 1) if fixed_bins is None else fixed_bins)
    for b in bins:
        m = ib == b
        if weights is not None:
            m = m & (weights > 0)
        n = int(m.sum()) if weights is None else float(weights[m].sum())
        if fixed_bins is None and n < min_n:
            continue
        if n < 5:
            continue
        d = N[m], dx[m]
        t = 19.5 - d[0]
        c = 0.5 * (edges[b] + edges[b + 1])
        if mode == "sample":
            mu = float(d[1].mean())
            sg = float(d[1].std(ddof=1))
            dd = d[1] - mu
            sk = float(np.clip((dd ** 3).mean() / max(d[1].std(), 1e-6) ** 3,
                               -SKEW_CAP, SKEW_CAP))
            rows.append(dict(c=c, n=n, mu=mu, sig=sg, skew=sk, ok=True))
        else:
            p, ok, fun = fit_subbin(d[1], t, truncated=(mode == "ml_trunc"),
                                    w=None if weights is None else weights[m])
            rows.append(dict(c=c, n=n, mu=float(p[0]), sig=float(p[1]),
                             skew=float(p[2]), ok=ok))
    return rows


def surfaces_from_rows(rows_per_cell, N_ref, deg):
    """WLS degree-`deg` moment polynomials per cell from sub-bin rows.
    Returns dict(mu, sig, skew) each (3,3,deg+1) + fit_range (3,3,2)."""
    D = deg + 1
    out = {k: np.zeros((3, 3, D)) for k in ("mu", "sig", "skew")}
    rng = np.zeros((3, 3, 2))
    for i in range(3):
        for j in range(3):
            rows = [r for r in rows_per_cell[i][j] if r["ok"]]
            if len(rows) < D + 1:
                raise RuntimeError(f"cell {i}{j}: only {len(rows)} usable rows")
            c = np.array([r["c"] for r in rows])
            w = np.sqrt([r["n"] for r in rows])
            u = c - N_ref
            rng[i, j] = (c.min(), c.max())
            for key in ("mu", "sig", "skew"):
                y = np.array([r[key] for r in rows])
                out[key][i, j] = np.polynomial.polynomial.polyfit(
                    u, y, deg, w=w)
    return out, rng


def eval_loglik(N, dx, isr, izr, surf, rng, N_ref, truncated=True):
    """Per-event (truncated) log-likelihood under moment surfaces, covariate
    clamped to each cell's fit range. Scalar sum + n."""
    total, ntot = 0.0, 0
    for i in range(3):
        for j in range(3):
            m = (isr == i) & (izr == j)
            if not m.any():
                continue
            Ncl = np.clip(N[m], rng[i, j, 0], rng[i, j, 1])
            u = Ncl - N_ref
            up = u[:, None] ** np.arange(surf["mu"].shape[-1])[None, :]
            mu = (surf["mu"][i, j] * up).sum(1)
            sg = np.clip((surf["sig"][i, j] * up).sum(1), SIG_MIN, 0.95)
            sk = np.clip((surf["skew"][i, j] * up).sum(1),
                         -SKEW_CAP, SKEW_CAP)
            xi, om, al = _moment_to_skewnormal_vec(mu, sg, sk)
            ll = skewnorm.logpdf(dx[m], al, loc=xi, scale=om)
            if truncated:
                sf = np.clip(skewnorm.sf(19.5 - N[m], al, loc=xi, scale=om),
                             1e-12, 1.0)
                ll = ll - np.log(sf)
            good = np.isfinite(ll)
            total += float(ll[good].sum()) - 50.0 * int((~good).sum())
            ntot += int(good.sum())
    return total, ntot
