"""binning.py — pure binning / normalisation helpers for the absorber-side
operator forensics (VALIDATION-ONLY; nothing under CDDF_analysis/ is touched).

Every helper here is pure numpy with no heavy imports so it can be unit-tested
under either environment (``gpdla`` for the matched-pair builder, ``gpdla-hbi``
for the fold analysis).  The binning convention is the pack's own
(``extract_pack._idx``): half-open ``[lo, hi)``, right-searchsorted.  The
builder cross-checks ``bin_index`` against the committed ``_idx`` on the real
arrays at run time (``_assert_index_convention``), exactly as
``validation/fp_ladder/build_fp_census.py`` does.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bin_index", "in_range", "coarse_block_sum", "row_normalise",
           "safe_ratio", "migration_moments", "leakage_fractions",
           "threshold_weights"]


def bin_index(edges, x):
    """The pack's grid index convention: half-open [lo, hi), right-searchsorted.

    Identical to ``CDDF_analysis/hbi_mcmc/extract_pack.py::_idx``.  Values below
    ``edges[0]`` give -1; values >= ``edges[-1]`` give ``len(edges) - 1``.
    """
    return np.searchsorted(np.asarray(edges, float),
                           np.asarray(x, float), side="right") - 1


def in_range(idx, n):
    """Mask of indices that land strictly inside ``[0, n)``."""
    idx = np.asarray(idx)
    return (idx >= 0) & (idx < int(n))


def coarse_block_sum(a, kz_to_K, axis):
    """Sum a fine-z axis into coarse blocks K using the pack's ``kz_to_K`` map.

    ``a`` may have any rank; ``axis`` names the fine-z axis.  The result has the
    same rank with that axis of length ``kz_to_K.max() + 1``.
    """
    a = np.asarray(a, float)
    kz = np.asarray(kz_to_K, int)
    if a.shape[axis] != kz.size:
        raise ValueError(f"axis {axis} has length {a.shape[axis]}, "
                         f"kz_to_K has {kz.size}")
    n_K = int(kz.max()) + 1
    out = np.zeros(a.shape[:axis] + (n_K,) + a.shape[axis + 1:], float)
    for K in range(n_K):
        sel = np.where(kz == K)[0]
        out[(slice(None),) * axis + (K,)] = a.take(sel, axis=axis).sum(axis=axis)
    return out


def row_normalise(m, axis, fallback=None):
    """Normalise ``m`` to unit sum along ``axis``; return (normalised, had_mass).

    Rows whose sum is zero are filled from ``fallback`` (already normalised,
    broadcastable to ``m``) when given, else left at zero.  ``had_mass`` is the
    boolean mask (``m.sum(axis)`` > 0) with ``axis`` removed — the caller is
    expected to RECORD it, so a fallback row is never mistaken for a
    measurement.
    """
    m = np.asarray(m, float)
    tot = m.sum(axis=axis)
    had = tot > 0
    denom = np.where(had, tot, 1.0)
    out = m / np.expand_dims(denom, axis)
    if fallback is not None:
        fb = np.broadcast_to(np.asarray(fallback, float), m.shape)
        out = np.where(np.expand_dims(had, axis), out, fb)
    else:
        out = np.where(np.expand_dims(had, axis), out, 0.0)
    return out, had


def safe_ratio(num, den, fill=0.0):
    """``num/den`` with ``den <= 0`` cells set to ``fill`` (never NaN/inf)."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    out = np.full(np.broadcast(num, den).shape, float(fill))
    ok = den > 0
    np.divide(num, den, out=out, where=ok)
    return np.where(ok, out, float(fill))


def migration_moments(mass, centres):
    """Mean / sd / (Fisher) skew of a discrete mass row over ``centres``.

    ``mass`` may be unnormalised; rows of zero mass return NaNs.  ``mass`` is
    (..., C) and ``centres`` is (C,).
    """
    mass = np.asarray(mass, float)
    x = np.asarray(centres, float)
    tot = mass.sum(axis=-1)
    ok = tot > 0
    w = np.where(ok[..., None], mass / np.where(ok, tot, 1.0)[..., None], 0.0)
    mean = (w * x).sum(axis=-1)
    var = (w * (x - mean[..., None]) ** 2).sum(axis=-1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = ((w * (x - mean[..., None]) ** 3).sum(axis=-1)
                / np.maximum(sd, 1e-12) ** 3)
    nan = np.full_like(mean, np.nan)
    return (np.where(ok, mean, nan), np.where(ok, sd, nan),
            np.where(ok, skew, nan))


def leakage_fractions(mass, edges, lo, hi):
    """Fraction of a migration row's mass below ``lo``, inside, and above ``hi``.

    ``edges`` are the observed-bin edges (C+1,); a bin counts as "inside" when
    its whole extent lies within ``[lo, hi)`` (the reporting bins are exact
    unions of observed bins on this grid, so no bin is split).
    """
    mass = np.asarray(mass, float)
    e = np.asarray(edges, float)
    tot = mass.sum(axis=-1)
    ok = tot > 0
    below = e[1:] <= lo + 1e-9
    above = e[:-1] >= hi - 1e-9
    inside = ~below & ~above
    den = np.where(ok, tot, 1.0)
    out = np.stack([(mass * below).sum(axis=-1) / den,
                    (mass * inside).sum(axis=-1) / den,
                    (mass * above).sum(axis=-1) / den], axis=-1)
    return np.where(ok[..., None], out, np.nan)


def threshold_weights(ntrue_edges, thr, report_floor):
    """The committed threshold weight u_b = dex of latent bin b above ``thr``.

    Identical to ``cc_posterior_validation.perz_recovery``: open-topped, and
    zero on latent bins whose CENTRE is below the reporting floor.
    """
    nt = np.asarray(ntrue_edges, float)
    centres = 0.5 * (nt[:-1] + nt[1:])
    reported = centres >= float(report_floor) - 1e-9
    u = np.clip(nt[1:] - np.maximum(nt[:-1], float(thr)), 0.0, None)
    return np.where(reported, u, 0.0)
