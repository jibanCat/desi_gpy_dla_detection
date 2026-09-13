#!/usr/bin/env python
"""opmetrics.py — OPERATOR-side predictive metrics for a fixed response
calibration object, and the Mg tensor build (PI ruling 2026-09-13b §15).

The scores here are the ones the ruling names: held-out mean / width / skew
residuals and up / in / down leakage fractions, resolved by true-N bin, S/N
cell, coarse z cell and near the boundaries.  They are computed on the SAME
discretised rows the fold consumes — the kernel masses produced by the
COMMITTED ``count_conserving_fold.surface_masses`` (loaded file-directly so the
jax-importing package ``__init__`` is not needed), not on an analytic density.

VALIDATION-ONLY.  ENV: gpdla.
"""
from __future__ import annotations

import importlib.util as ilu
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))


def _load_ccf():
    """Load the COMMITTED count_conserving_fold file-directly."""
    key = "_absladder_ccf"
    if key in sys.modules:
        return sys.modules[key]
    spec = ilu.spec_from_file_location(
        key, os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc",
                          "count_conserving_fold.py"))
    mod = ilu.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


class _Shim:
    """The four attributes ``surface_masses`` reads off a pack.

    Using a shim (the same device ``validation/absorber_diag/
    build_matched_ops.py`` already uses) lets a variant declare its OWN skew
    ramp without touching the pack or the committed function: the ramp is part
    of the variant's fixed calibration object, and the DELIVERED product is the
    precomputed Mg tensor, so nothing downstream has to re-read it.
    """

    def __init__(self, ntrue_edges, n_ref, ramp, sig_floor):
        self.ntrue_edges = np.asarray(ntrue_edges, float)
        self.resp_N_ref = float(n_ref)
        self.resp_skew_ramp = np.asarray(ramp, float)
        self.resp_sig_floor = float(sig_floor)


def model_masses(obj, ntrue_edges, nhat_edges, sig_floor=1e-3):
    """(SR, ZR, C, B) kernel masses of a fixed object, via the COMMITTED
    ``surface_masses``.  Returns (masses, phi)."""
    ccf = _load_ccf()
    shim = _Shim(ntrue_edges, obj["N_ref"], obj["ramp"], sig_floor)
    return ccf.surface_masses(shim, obj["surf"]["mu"], obj["surf"]["sig"],
                              obj["surf"]["skew"],
                              np.asarray(obj["rng"], float),
                              np.asarray(nhat_edges, float))


def gather_Mg(masses, s2sr, kz2K, K2zr):
    """(S, Kf, C, B) gathered tensor — the runner's ``--mg-fixed`` shape."""
    s2sr = np.asarray(s2sr, int); kz2K = np.asarray(kz2K, int)
    K2zr = np.asarray(K2zr, int)
    return masses[s2sr[:, None], K2zr[kz2K][None, :], :, :]


# ===========================================================================
# row statistics
# ===========================================================================
def row_stats(mass, centers, lo, hi):
    """(mean, sd, skew, down, inb, up) of a unit-mass row on the observed grid.

    ``lo``/``hi`` are the reporting-bin edges the leakage is measured against
    (the latent bin's own edges), matching the forensics' convention.
    """
    mass = np.asarray(mass, float)
    tot = mass.sum()
    if tot <= 0:
        return (np.nan,) * 6
    p = mass / tot
    m = float(np.sum(p * centers))
    d = centers - m
    v = float(np.sum(p * d * d))
    sd = np.sqrt(max(v, 1e-30))
    sk = float(np.sum(p * d ** 3)) / sd ** 3 if v > 1e-12 else np.nan
    down = float(p[centers < lo].sum())
    up = float(p[centers >= hi].sum())
    return m, sd, sk, down, 1.0 - down - up, up


def empirical_rows(xhat, b_i, s_i, K_i, nhat_edges, B, S, K):
    """(S, K, C, B) histogram of held-out detections — the empirical rows."""
    ne = np.asarray(nhat_edges, float)
    C = len(ne) - 1
    c = np.clip(np.digitize(xhat, ne) - 1, -1, C)
    ok = (c >= 0) & (c < C)
    out = np.zeros((S, K, C, B))
    np.add.at(out, (s_i[ok], K_i[ok], c[ok], b_i[ok]), 1.0)
    return out


def compare_rows(masses, emp, s2sr, K2zr, ntrue_edges, nhat_edges,
                 min_events=40):
    """Per-(b, s, K) held-out comparison of model vs empirical rows.

    Returns a list of dict records; the aggregation is done by the caller so
    every stratification the ruling asks for is computed from one table.
    """
    ne = np.asarray(nhat_edges, float)
    cen = 0.5 * (ne[:-1] + ne[1:])
    nt = np.asarray(ntrue_edges, float)
    S, K, C, B = emp.shape
    recs = []
    for b in range(B):
        lo, hi = nt[b], nt[b + 1]
        for s in range(S):
            for k in range(K):
                n = float(emp[s, k, :, b].sum())
                if n < min_events:
                    continue
                em = row_stats(emp[s, k, :, b], cen, lo, hi)
                mm = row_stats(masses[s2sr[s], K2zr[k], :, b], cen, lo, hi)
                recs.append(dict(
                    b=b, s=s, k=k, n=n, b_lo=float(lo), b_hi=float(hi),
                    mean_emp=em[0], mean_mod=mm[0],
                    sd_emp=em[1], sd_mod=mm[1],
                    skew_emp=em[2], skew_mod=mm[2],
                    down_emp=em[3], down_mod=mm[3],
                    in_emp=em[4], in_mod=mm[4],
                    up_emp=em[5], up_mod=mm[5],
                    d_mean=mm[0] - em[0],
                    r_sd=(mm[1] / em[1] - 1.0) if em[1] > 0 else np.nan,
                    d_skew=mm[2] - em[2],
                    d_down=mm[3] - em[3], d_in=mm[4] - em[4],
                    d_up=mm[5] - em[5]))
    return recs


def _wstat(recs, key, wkey="n"):
    if not recs:
        return dict(n=0)
    v = np.array([r[key] for r in recs], float)
    w = np.array([r[wkey] for r in recs], float)
    g = np.isfinite(v)
    if not g.any():
        return dict(n=0)
    v, w = v[g], w[g]
    mean = float(np.sum(w * v) / np.sum(w))
    return dict(n=int(g.sum()), n_events=float(np.sum(w)),
                wmean=mean, median=float(np.median(v)),
                p16=float(np.percentile(v, 16)),
                p84=float(np.percentile(v, 84)),
                absmax=float(np.max(np.abs(v))))


KEYS = ("d_mean", "r_sd", "d_skew", "d_down", "d_in", "d_up")


def aggregate(recs, ntrue_edges, snr_edges):
    """Stratified aggregation: overall, per true-N bin, per S/N stratum, per
    coarse z cell, and at the two boundary regions the ruling names."""
    nt = np.asarray(ntrue_edges, float)
    out = {"overall": {k: _wstat(recs, k) for k in KEYS}}
    byb = {}
    for b in sorted({r["b"] for r in recs}):
        sel = [r for r in recs if r["b"] == b]
        byb[f"[{nt[b]:.1f},{nt[b+1]:.1f})"] = {k: _wstat(sel, k) for k in KEYS}
    out["by_true_N_bin"] = byb
    bys = {}
    se = np.asarray(snr_edges, float)
    for s in sorted({r["s"] for r in recs}):
        sel = [r for r in recs if r["s"] == s]
        bys[f"S/N[{se[s]:.0f},{se[s+1]:.0f})"] = {k: _wstat(sel, k)
                                                  for k in KEYS}
    out["by_snr_stratum"] = bys
    byk = {}
    for k_ in sorted({r["k"] for r in recs}):
        sel = [r for r in recs if r["k"] == k_]
        byk[f"K{k_}"] = {k: _wstat(sel, k) for k in KEYS}
    out["by_coarse_z"] = byk
    lowb = [r for r in recs if r["b_hi"] <= 19.7 + 1e-9]
    hib = [r for r in recs if r["b_lo"] >= 21.3 - 1e-9]
    out["boundary_low_b_lt_19p7"] = {k: _wstat(lowb, k) for k in KEYS}
    out["boundary_high_b_ge_21p3"] = {k: _wstat(hib, k) for k in KEYS}
    return out
