#!/usr/bin/env python
"""respfit.py — response calibration-object fitting, CV and diagnostics for the
first absorber-side ladder (PI ruling 2026-09-13b §5, §15).

VALIDATION-ONLY.  Nothing under ``CDDF_analysis/`` is modified, and no sampler
is run.  Every object this module produces is a FIXED calibration object of the
SAME functional form the production fold consumes:

    (mu_coef, sig_coef, skew_coef) : (SR=3, ZR=3, D) moment polynomials in
                                     u = N_true - resp_N_ref
    fit_rng                        : (SR, ZR, 2) covariate clamp range

evaluated by the committed ``count_conserving_fold.surface_masses``.  Nothing
here is fitted to, or scored against, any mock-closure or survey quantity:
model selection is by 2-fold cross-validation over SIGHTLINES (TARGETID) on the
2LPT-0 calibration events only.

The estimator of record (R0) is reproduced here EXACTLY as the committed
recovered chain implements it
(``CDDF_analysis/hbi/adopted_response/{fitlib,run_d2b_lib}.py``):

  * 0.1-dex sub-bins of N_true on ``arange(19.0, 21.4, 0.1)``, min 50 events;
  * per sub-bin an UNTRUNCATED skew-normal maximum-likelihood (mu, sigma, skew)
    (Nelder-Mead, two starts, SIG_MIN 0.02, |skew| < 0.95);
  * per-(SNR, z)-cell degree-2 WLS moment polynomials (weight = sub-bin count)
    plus ONE cubic coefficient shared across the 9 cells, 2 refit iterations;
  * ``fit_rng`` = (min, max) sub-bin CENTRE that survived the count cut — which
    is where the load-bearing upper clamp 21.35 comes from.

``tests/test_response_variants.py`` gates this re-implementation against the
committed ``fitlib`` / ``run_d2b_lib`` element-wise.

ENV: gpdla (numpy/scipy only).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import skewnorm

# --- committed constants (fitlib.py) ---------------------------------------
SKEW_CAP = 0.95
SIG_MIN = 0.02
XHAT_FLOOR = 19.5                 # the calibration selection x_hat >= 19.5
R0_EDGES = np.arange(19.0, 21.4 + 1e-9, 0.1)
R0_MIN_N = 50
SNR_EDGES = np.array([2.0, 3.5, 6.5, np.inf])
Z_EDGES = np.array([0.0, 2.56, 2.96, np.inf])

# the committed skew-normal skewness ceiling (znz_kernel._SN_SKEW_MAX)
SN_SKEW_MAX = 0.5 * (4.0 - np.pi) * (np.sqrt(2.0 / np.pi) ** 3) / \
    (1.0 - 2.0 / np.pi) ** 1.5


# ===========================================================================
# moment <-> skew-normal (committed semantics, count_conserving_fold._m2sn_vec)
# ===========================================================================
def moment_to_skewnormal(mean, sd, skew):
    """Vector moment -> skew-normal (xi, omega, alpha); committed semantics."""
    mean = np.asarray(mean, float)
    bb = np.sqrt(2.0 / np.pi)
    s_ = np.clip(np.asarray(skew, float), -0.995 * SN_SKEW_MAX,
                 0.995 * SN_SKEW_MAX)
    sd = np.maximum(np.asarray(sd, float), 1e-9)
    cc = 0.5 * (4.0 - np.pi)
    r = (np.abs(s_) / cc) ** (2.0 / 3.0)
    gg = r / (1.0 + r)
    delta = np.clip(np.sign(s_) * np.sqrt(gg) / bb, -0.999, 0.999)
    delta = np.where(np.abs(s_) < 1e-9, 0.0, delta)
    al = delta / np.sqrt(np.maximum(1.0 - delta * delta, 1e-12))
    om = sd / np.sqrt(np.maximum(1.0 - (bb * delta) ** 2, 1e-12))
    xi = mean - om * bb * delta
    return xi, om, al


# ===========================================================================
# events
# ===========================================================================
def load_events(path):
    """Load the calibration-event NPZ written by extract_calib_events.py."""
    d = np.load(path, allow_pickle=True)
    ev = {k: np.asarray(d[k]) for k in
          ("N_true", "snr", "zqso", "zdla", "dx", "xhat", "tid")}
    ev["provenance"] = str(d["provenance"])
    return ev


def cell_index(snr, zqso, snr_edges=SNR_EDGES, z_edges=Z_EDGES):
    """(i_snr, i_z) response-cell indices, committed digitize convention."""
    isr = np.clip(np.digitize(np.asarray(snr, float), np.asarray(snr_edges))
                  - 1, 0, len(snr_edges) - 2)
    izr = np.clip(np.digitize(np.asarray(zqso, float), np.asarray(z_edges))
                  - 1, 0, len(z_edges) - 2)
    return isr, izr


def parity_fold(tid):
    """2-fold CV split over SIGHTLINES by TARGETID parity (PI §15 'split
    matched events by TARGETID parity').  Returns a bool mask (fold A)."""
    return (np.asarray(tid, np.int64) % 2) == 0


# ===========================================================================
# per-sub-bin moment estimators
# ===========================================================================
def _nll(params, dx, t, truncated, w=None):
    m, s, sk = params
    if not (SIG_MIN < s < 1.0 and -SKEW_CAP < sk < SKEW_CAP):
        return 1e9
    xi, om, al = moment_to_skewnormal(m, s, sk)
    ll = skewnorm.logpdf(dx, al, loc=xi, scale=om)
    if not np.all(np.isfinite(ll)):
        return 1e9
    if truncated:
        sf = skewnorm.sf(t, al, loc=xi, scale=om)
        ll = ll - np.log(np.clip(sf, 1e-12, 1.0))
    return -(ll.sum() if w is None else (w * ll).sum())


def fit_subbin_ml(dx, t, truncated, w=None):
    """ML (mu, sigma, skew) of one sub-bin; committed fitlib.fit_subbin."""
    dx = np.asarray(dx, float)
    if w is None:
        m0 = float(dx.mean())
        s0 = float(max(dx.std(ddof=1), SIG_MIN * 1.5))
        d = dx - m0
    else:
        ws = max(float(w.sum()), 1e-9)
        m0 = float((w * dx).sum() / ws)
        d = dx - m0
        s0 = float(max(np.sqrt(float((w * d ** 2).sum()) / ws), SIG_MIN * 1.5))
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


def subbin_moments(N, dx, edges, min_n, mode):
    """Per-sub-bin (mu, sigma, skew).

    mode:
      'sample'    — sample moments (2nd/3rd central moments; the MOMENT
                    estimator: reproduces the sub-bin's variance by
                    construction);
      'ml'        — untruncated skew-normal ML  (THE ESTIMATOR OF RECORD);
      'ml_trunc'  — ML with the x_hat >= 19.5 selection accounted for
                    (per-event left truncation at t_i = 19.5 - N_i).
    Returns a list of dict rows (c, n, mu, sig, skew, ok).
    """
    N = np.asarray(N, float); dx = np.asarray(dx, float)
    edges = np.asarray(edges, float)
    rows = []
    ib = np.digitize(N, edges) - 1
    for b in range(len(edges) - 1):
        m = ib == b
        n = int(m.sum())
        if n < min_n or n < 5:
            continue
        dn, dd = N[m], dx[m]
        t = XHAT_FLOOR - dn
        c = 0.5 * (edges[b] + edges[b + 1])
        if mode == "sample":
            mu = float(dd.mean())
            sg = float(dd.std(ddof=1))
            e = dd - mu
            sk = float(np.clip((e ** 3).mean() / max(dd.std(), 1e-6) ** 3,
                               -SKEW_CAP, SKEW_CAP))
            rows.append(dict(c=c, n=n, mu=mu, sig=sg, skew=sk, ok=True))
        elif mode in ("ml", "ml_trunc"):
            p, ok, _ = fit_subbin_ml(dd, t, truncated=(mode == "ml_trunc"))
            rows.append(dict(c=c, n=n, mu=float(p[0]), sig=float(p[1]),
                             skew=float(p[2]), ok=ok))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return rows


def adaptive_edges(N, lo, hi, step, min_n, hard_min_n=25, max_width=None):
    """Sub-bin edges on [lo, hi] of width ``step`` in the well-populated region,
    MERGED upward wherever a bin would hold fewer than ``min_n`` events.

    Used by the fit-range-extension variants (R1a+): the estimator of record's
    fixed 0.1-dex grid with min_n=50 simply DROPS every sub-bin above 21.35,
    which is what creates the load-bearing upper clamp.  Merging instead keeps
    the high-N region IN the fit at the cost of coarser N resolution.  The row
    centre used downstream is the count-weighted mean N of the members
    (``subbin_moments_weighted_centre``), so a merge does not bias the
    covariate.

    ``max_width`` caps how far a merge may run: a bin is emitted once it
    reaches that width provided it holds at least ``hard_min_n`` events, which
    keeps the sparse high-N tail resolved instead of collapsing it into one
    0.9-dex block.  A final remainder too small to stand alone is absorbed into
    its predecessor.
    """
    N = np.asarray(N, float)
    grid = np.arange(lo, hi + 1e-9, step)
    if grid[-1] < hi - 1e-9:
        grid = np.append(grid, hi)
    counts, _ = np.histogram(N, grid)
    edges = [grid[0]]
    acc = 0
    for i in range(len(counts)):
        acc += int(counts[i])
        width = grid[i + 1] - edges[-1]
        wide = (max_width is not None) and (width >= max_width - 1e-9)
        if acc >= min_n or (wide and acc >= hard_min_n):
            edges.append(grid[i + 1])
            acc = 0
    if edges[-1] < grid[-1] - 1e-12:
        if acc >= hard_min_n:
            edges.append(grid[-1])
        elif len(edges) > 1:
            edges[-1] = grid[-1]
        else:
            edges.append(grid[-1])
    return np.asarray(edges, float)


def subbin_moments_weighted_centre(N, dx, edges, min_n, mode):
    """As ``subbin_moments`` but the row centre is the MEAN N of the sub-bin
    (correct for the merged, variable-width bins of ``adaptive_edges``)."""
    N = np.asarray(N, float); dx = np.asarray(dx, float)
    edges = np.asarray(edges, float)
    rows = []
    ib = np.digitize(N, edges) - 1
    for b in range(len(edges) - 1):
        m = ib == b
        n = int(m.sum())
        if n < min_n or n < 5:
            continue
        dn, dd = N[m], dx[m]
        c = float(dn.mean())
        if mode == "sample":
            mu = float(dd.mean()); sg = float(dd.std(ddof=1))
            e = dd - mu
            sk = float(np.clip((e ** 3).mean() / max(dd.std(), 1e-6) ** 3,
                               -SKEW_CAP, SKEW_CAP))
            rows.append(dict(c=c, n=n, mu=mu, sig=sg, skew=sk, ok=True))
        else:
            p, ok, _ = fit_subbin_ml(dd, XHAT_FLOOR - dn,
                                     truncated=(mode == "ml_trunc"))
            rows.append(dict(c=c, n=n, mu=float(p[0]), sig=float(p[1]),
                             skew=float(p[2]), ok=ok))
    return rows


# ===========================================================================
# surfaces
# ===========================================================================
def surfaces_percell(rows_per_cell, N_ref, deg):
    """Per-cell degree-``deg`` WLS moment polynomials (committed
    fitlib.surfaces_from_rows)."""
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
                out[key][i, j] = np.polynomial.polynomial.polyfit(u, y, deg,
                                                                  w=w)
    return out, rng


def surfaces_shared(rows_per_cell, N_ref, deg_shared, deg_cell=2, n_iter=2):
    """Per-cell degree-``deg_cell`` + orders ``deg_cell+1..deg_shared`` SHARED
    across the 9 cells (committed run_d2b_lib.shared_surfaces, generalised in
    ``deg_cell``; deg_cell=2, deg_shared=3 IS the adopted estimator)."""
    Ds = deg_shared + 1
    lo = deg_cell + 1
    keys = ("mu", "sig", "skew")
    shared = np.zeros((3, Ds))
    cell = {k: np.zeros((3, 3, deg_cell + 1)) for k in keys}
    rng = np.zeros((3, 3, 2))
    for _ in range(n_iter):
        for i in range(3):
            for j in range(3):
                rr = [r for r in rows_per_cell[i][j] if r["ok"]]
                if len(rr) < deg_cell + 2:
                    raise RuntimeError(f"cell {i}{j}: {len(rr)} usable rows")
                c = np.array([r["c"] for r in rr]); u = c - N_ref
                w = np.sqrt([r["n"] for r in rr])
                rng[i, j] = (c.min(), c.max())
                up = u[:, None] ** np.arange(Ds)[None, :]
                for ki, k in enumerate(keys):
                    y = np.array([r[k] for r in rr]) - up[:, lo:] @ shared[ki, lo:]
                    cell[k][i, j] = np.polynomial.polynomial.polyfit(
                        u, y, deg_cell, w=w)
        for ki, k in enumerate(keys):
            uu, yy, ww = [], [], []
            for i in range(3):
                for j in range(3):
                    rr = [r for r in rows_per_cell[i][j] if r["ok"]]
                    c = np.array([r["c"] for r in rr]); u = c - N_ref
                    up2 = u[:, None] ** np.arange(deg_cell + 1)[None, :]
                    y = np.array([r[k] for r in rr]) - up2 @ cell[k][i, j]
                    uu.append(u); yy.append(y)
                    ww.append(np.sqrt([r["n"] for r in rr]))
            uu = np.concatenate(uu); yy = np.concatenate(yy)
            ww = np.concatenate(ww)
            if Ds > lo:
                X = uu[:, None] ** np.arange(lo, Ds)[None, :]
                beta, *_ = np.linalg.lstsq(ww[:, None] * X, ww * yy,
                                           rcond=None)
                shared[ki, lo:] = beta
    surf = {}
    for ki, k in enumerate(keys):
        arr = np.zeros((3, 3, Ds))
        arr[..., :deg_cell + 1] = cell[k]
        arr[..., lo:] = shared[ki, lo:][None, None, :]
        surf[k] = arr
    return surf, rng, shared


# ===========================================================================
# evaluation of a fixed calibration object
# ===========================================================================
def eval_moments(surf, rng, N_ref, N, isr, izr, sig_floor=1e-3,
                 skew_ramp=None):
    """(mean_dx, sd, skew) of a calibration object at (N, cell), with the
    covariate clamp and (optionally) the skew ramp applied — the SAME
    arithmetic ``count_conserving_fold.surface_masses`` performs."""
    N = np.asarray(N, float)
    D = surf["mu"].shape[-1]
    mu = np.zeros_like(N); sg = np.zeros_like(N); sk = np.zeros_like(N)
    for i in range(3):
        for j in range(3):
            m = (isr == i) & (izr == j)
            if not np.any(m):
                continue
            Ncl = np.clip(N[m], rng[i, j, 0], rng[i, j, 1])
            up = (Ncl - N_ref)[:, None] ** np.arange(D)[None, :]
            mu[m] = up @ surf["mu"][i, j]
            sg[m] = np.maximum(up @ surf["sig"][i, j], sig_floor)
            sk[m] = np.clip(up @ surf["skew"][i, j],
                            -0.995 * SN_SKEW_MAX, 0.995 * SN_SKEW_MAX)
    if skew_ramp is not None:
        rc, rw = float(skew_ramp[0]), float(skew_ramp[1])
        sk = sk * (1.0 - np.clip((N - rc) / rw, 0.0, 1.0))
    return mu, sg, sk


def per_event_loglik(surf, rng, N_ref, N, dx, isr, izr, truncated=True,
                     skew_ramp=None):
    """Held-out per-event (floor-conditional) log-likelihood — the committed
    D2b CV score (fitlib.eval_loglik), generalised over the skew ramp."""
    mu, sg, sk = eval_moments(surf, rng, N_ref, N, isr, izr,
                              sig_floor=SIG_MIN, skew_ramp=skew_ramp)
    sg = np.clip(sg, SIG_MIN, 0.95)
    sk = np.clip(sk, -SKEW_CAP, SKEW_CAP)
    xi, om, al = moment_to_skewnormal(mu, sg, sk)
    ll = skewnorm.logpdf(dx, al, loc=xi, scale=om)
    if truncated:
        sf = np.clip(skewnorm.sf(XHAT_FLOOR - N, al, loc=xi, scale=om),
                     1e-12, 1.0)
        ll = ll - np.log(sf)
    good = np.isfinite(ll)
    return float(ll[good].sum()) - 50.0 * int((~good).sum()), int(good.sum())


# ===========================================================================
# variant specification and fitting
# ===========================================================================
NO_RAMP = (1.0e9, 1.0)      # a skew ramp that is identically 0 on any real N


class VariantSpec(dict):
    """A declarative, fully-serialisable description of a FIXED response
    calibration object.  Everything that can change between R0 and R1x is a
    field here, so the variant table and the provenance stamp are generated
    from the same object the fit reads.

    Fields
    ------
    name          : variant id ("R0", "R1a", ...)
    edges         : "r0_fixed" (the 0.1-dex 19.0-21.4 grid of record) or
                    "adaptive" (0.1-dex merged upward to ``min_n`` over
                    [lo, hi] — the fit-range extension)
    lo, hi, step  : adaptive sub-bin grid
    min_n         : minimum events per sub-bin
    estimator     : "ml" (of record), "sample" (moments), "ml_trunc"
    deg_cell      : per-cell polynomial degree (2 of record)
    deg_shared    : highest order, shared across the 9 cells (3 of record)
    fit_rng       : "data" (min/max fitted sub-bin centre — of record) or
                    "full" (the latent support; no clamp)
    ramp          : (collapse, width) skew ramp, or None for NO ramp
    marginalise   : None, or dict(weight="flat", n_quad=..., deg=...) —
                    the latent-bin quadrature repair (R1c)
    """


SPEC_R0 = VariantSpec(
    name="R0", edges="r0_fixed", lo=19.0, hi=21.4, step=0.1, min_n=50,
    estimator="ml", deg_cell=2, deg_shared=3, fit_rng="data",
    ramp=(21.0, 0.5), marginalise=None)


def _rows_for_cells(N, dx, isr, izr, spec):
    rows = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            m = (isr == i) & (izr == j)
            if spec["edges"] == "r0_fixed":
                e = np.arange(spec["lo"], spec["hi"] + 1e-9, spec["step"])
                rows[i][j] = subbin_moments(N[m], dx[m], e, spec["min_n"],
                                            spec["estimator"])
            elif spec["edges"] == "adaptive":
                e = adaptive_edges(N[m], spec["lo"], spec["hi"], spec["step"],
                                   spec["min_n"],
                                   hard_min_n=spec.get("hard_min_n", 25),
                                   max_width=spec.get("max_width"))
                rows[i][j] = subbin_moments_weighted_centre(
                    N[m], dx[m], e, spec["min_n"], spec["estimator"])
            else:
                raise ValueError(spec["edges"])
    return rows


def fit_variant(N, dx, isr, izr, N_ref, spec, ntrue_edges=None):
    """Fit one FIXED calibration object.  Returns a dict with
    surf/rng/ramp/n_coef/rows (everything needed to evaluate and to report)."""
    rows = _rows_for_cells(N, dx, isr, izr, spec)
    if spec["deg_shared"] > spec["deg_cell"]:
        surf, rng, shared = surfaces_shared(rows, N_ref, spec["deg_shared"],
                                            deg_cell=spec["deg_cell"])
    else:
        surf, rng = surfaces_percell(rows, N_ref, spec["deg_cell"])
        shared = np.zeros((3, spec["deg_cell"] + 1))
    rng_data = rng.copy()
    if spec["fit_rng"] == "full":
        if ntrue_edges is None:
            raise ValueError("fit_rng='full' needs ntrue_edges")
        ne = np.asarray(ntrue_edges, float)
        rng = np.tile(np.array([ne[0], ne[-1]]), (3, 3, 1))
    ramp = spec["ramp"] if spec["ramp"] is not None else NO_RAMP
    out = dict(surf=surf, rng=rng, rng_data=rng_data, shared=shared,
               ramp=ramp, N_ref=float(N_ref), spec=dict(spec), rows=rows)
    if spec.get("marginalise"):
        out = marginalise_object(out, ntrue_edges, **spec["marginalise"])
    out["n_coef"] = count_coefficients(out)
    return out


def count_coefficients(obj):
    """Number of FREE fitted calibration coefficients in the object.

    Per-cell orders 0..deg_cell are free per (SNR, z) cell (9 cells); orders
    above deg_cell are ONE shared coefficient each.  ``fit_rng`` contributes
    2 numbers per cell that are DATA-DETERMINED (min/max fitted sub-bin
    centre), not free parameters, and 0 when the range is the fixed latent
    support; they are reported separately.
    """
    sp = obj["spec"]
    D = obj["surf"]["mu"].shape[-1]
    dc = min(sp["deg_cell"], D - 1)
    per_moment = 9 * (dc + 1) + max(D - 1 - dc, 0)
    rng_n = 0 if sp["fit_rng"] == "full" else 9 * 2
    return dict(per_moment=int(per_moment), total_moments=int(3 * per_moment),
                fit_range_numbers=int(rng_n),
                total_including_range=int(3 * per_moment + rng_n))


# ---------------------------------------------------------------------------
# the latent-bin quadrature repair (R1c)
# ---------------------------------------------------------------------------
def _mixture_moments(mean, sd, skew, w):
    """First three moments of a finite mixture of skew-normals given each
    component's (mean, sd, skewness) and mixture weights ``w`` (sum 1)."""
    m1 = float(np.sum(w * mean))
    d = mean - m1
    m2 = float(np.sum(w * (sd ** 2 + d ** 2)))
    m3 = float(np.sum(w * (skew * sd ** 3 + 3.0 * d * sd ** 2 + d ** 3)))
    sd_m = np.sqrt(max(m2, 1e-12))
    return m1, sd_m, m3 / sd_m ** 3


def bin_marginal_targets(obj, ntrue_edges, n_quad=17, weight="flat"):
    """Moments of the LATENT-BIN-MARGINALISED response row, per (cell, bin).

    ``surface_masses`` builds the row for latent bin b by evaluating the
    response at the SINGLE point N = centre(b).  The fold's M[c,b] is by
    definition the row marginalised over the bin,
        M[c,b] = int_b p(x_hat | N) w(N) dN / int_b w(N) dN,
    so the midpoint evaluation is a QUADRATURE approximation whose leading
    error is a MISSING VARIANCE equal to Var_{N in b}[N + mu(N)] — the
    diagnosed width narrowing.  This routine computes the exact first three
    moments of that mixture with a FLAT within-bin weight (deliberately NOT
    the mock's f(N): a weight taken from the population would make the fixed
    calibration object depend on the science parameter).

    Returns arrays (3, 3, B) of (mu_target relative to the bin centre,
    sd_target, skew_target).
    """
    ne = np.asarray(ntrue_edges, float)
    Nc = 0.5 * (ne[:-1] + ne[1:])
    B = len(Nc)
    mu_t = np.zeros((3, 3, B)); sd_t = np.zeros((3, 3, B))
    sk_t = np.zeros((3, 3, B))
    for i in range(3):
        for j in range(3):
            for b in range(B):
                q = np.linspace(ne[b], ne[b + 1], n_quad + 2)[1:-1]
                if weight == "flat":
                    w = np.full(len(q), 1.0 / len(q))
                else:
                    raise ValueError(weight)
                mu, sg, sk = eval_moments(
                    obj["surf"], obj["rng"], obj["N_ref"], q,
                    np.full(len(q), i), np.full(len(q), j),
                    skew_ramp=obj["ramp"])
                m1, s1, k1 = _mixture_moments(q + mu, sg, sk, w)
                mu_t[i, j, b] = m1 - Nc[b]
                sd_t[i, j, b] = s1
                sk_t[i, j, b] = k1
    return mu_t, sd_t, sk_t


def marginalise_object(obj, ntrue_edges, n_quad=17, deg=4, weight="flat"):
    """Re-express a fitted object so that the MIDPOINT evaluation performed by
    ``surface_masses`` reproduces the bin-MARGINALISED first three moments.

    No new calibration data is touched and no new degree of freedom is opened
    against the data: the targets are a deterministic quadrature of the SAME
    fitted conditional surfaces.  The only choice is the polynomial degree used
    to re-express the targets on the latent-bin centres, which is picked by CV.
    The resulting object is TIED TO THE LATENT GRID it was marginalised on
    (stated in the provenance).
    """
    mu_t, sd_t, sk_t = bin_marginal_targets(obj, ntrue_edges, n_quad=n_quad,
                                            weight=weight)
    ne = np.asarray(ntrue_edges, float)
    Nc = 0.5 * (ne[:-1] + ne[1:])
    u = Nc - obj["N_ref"]
    D = deg + 1
    surf = {k: np.zeros((3, 3, D)) for k in ("mu", "sig", "skew")}
    resid = {}
    for k, tgt in (("mu", mu_t), ("sig", sd_t), ("skew", sk_t)):
        r = []
        for i in range(3):
            for j in range(3):
                c = np.polynomial.polynomial.polyfit(u, tgt[i, j], deg)
                surf[k][i, j] = c
                r.append(float(np.max(np.abs(
                    np.polynomial.polynomial.polyval(u, c) - tgt[i, j]))))
        resid[k] = float(np.max(r))
    new = dict(obj)
    new["surf"] = surf
    new["rng"] = np.tile(np.array([ne[0], ne[-1]]), (3, 3, 1))
    new["ramp"] = NO_RAMP
    new["marginal_targets"] = dict(mu=mu_t, sd=sd_t, skew=sk_t)
    new["marginal_resid"] = resid
    new["spec"] = dict(obj["spec"])
    new["spec"]["deg_cell"] = deg          # every order is per-cell here
    new["spec"]["deg_shared"] = deg
    new["spec"]["fit_rng"] = "full"
    return new
