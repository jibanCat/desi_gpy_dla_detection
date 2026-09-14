#!/usr/bin/env python
"""reduce_truthfree.py — the TRUTH-FREE half of the committed reductions.

WHY THIS EXISTS.  ``cc_posterior_validation.perz_recovery`` and the
``thresholds`` / ``reporting_bins`` blocks of ``validation/fp_ladder/run_ladder.py``
are the reduction of record for the frozen Paper-1 low-z model, but every one of
them takes the MOCK TRUTH surface ``ft`` as a required argument and reports
``truth`` / ``median_bias_pct`` / ``truth_in_68`` / ``truth_in_95``.  On real
data no such object exists.  This module reproduces the POSTERIOR half of those
reductions — and nothing else — by copying the weight construction and the
percentile arithmetic VERBATIM from:

  * ``CDDF_analysis/hbi_mcmc/cc_posterior_validation.perz_recovery``
    (lines 225-273 @ CODE_FREEZE_FINAL_3) for the per-z-cell / coarse-block /
    Paper-1-bin / all-z threshold reduction;
  * ``validation/fp_ladder/run_ladder.py`` lines 157-184 for the all-z
    threshold percentiles (via the committed ``model_a.reduce_f_posterior``)
    and the 0.2-dex reporting bins;
  * ``validation/fp_ladder/ladder_table.omega_allz_from_weights`` (lines
    ~325-350) for Omega[20.3, 21.6], whose weights are imported READ-ONLY from
    the paper repository's ``hbi_reduction.py``.

The float operations are byte-for-byte the same expressions in the same order,
so on any given ``f_draws`` this module's percentiles are bit-identical to the
mock runner's; ``tests/test_run_real_c1.py`` pins that against a stored mock
posterior to 1e-10 (in fact to exact equality).

NOTHING HERE READS, COMPUTES OR EMITS A TRUTH QUANTITY.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import (PAPER1_LOWZ_BINS,
                                                            _overlap_w)

#: the 0.2-dex reporting-bin edges, VERBATIM run_ladder.py line 159
REDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)

#: run_ladder.py line 161 — the two headline all-z estimands
THRESHOLD_KEYS = ((20.0, "dndx_dla_20p0_allz"), (20.3, "dndx_dla_20p3_allz"))

PAPER_FIGURES = "/home/mfho/Latex/gp_dla_desi_y3/paper_figures"   # READ-ONLY


def thresholds_allz(f_draws, pk, red=None):
    """dN/dX(>=20.0) and dN/dX(>=20.3), all-z, posterior percentiles only.

    VERBATIM run_ladder.py lines 155-167 with every truth term deleted.
    """
    red = reduce_f_posterior(f_draws, pk) if red is None else red
    rep = {}
    for thr, key in THRESHOLD_KEYS:
        dr = np.asarray(red[key])
        q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
        rep[f"ge{thr}"] = dict(
            key=key, n_draws=int(dr.size),
            post_p16_50_84=[float(x) for x in q[1:4]],
            post_p2p5_97p5=[float(q[0]), float(q[4])])
    return rep


def reporting_bins_0p2dex(f_draws, pk):
    """The locked 0.2-dex reporting bins, all-z, posterior percentiles only.

    VERBATIM run_ladder.py lines 157-159 + 175-184, truth terms deleted.
    """
    f_draws = np.asarray(f_draws)
    ntrue = np.asarray(pk.ntrue_edges, float)
    dN = np.diff(ntrue)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    out = []
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        dr = ((f_draws[:, m, :] * dN[None, m, None]).sum(axis=1)
              * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
        out.append(dict(bin=[round(e0, 1), round(e1, 1)],
                        post_p2p5_16_50_84_97p5=[float(x) for x in q]))
    return out


def perz_posterior(f_draws, pk, thresholds=(20.0, 20.3)):
    """perz_recovery's POSTERIOR half: per native z cell, per coarse block, per
    locked Paper-1 bin and all-z, for each threshold.

    Copied line-for-line from ``cc_posterior_validation.perz_recovery``; the
    ``ft`` argument, ``tr_k``, ``truth``, ``median_bias_pct`` and the two
    containment booleans are the ONLY things removed.  The weights ``u`` and
    ``w`` and the ``pd`` percentile expression are unchanged.
    """
    f_draws = np.asarray(f_draws)
    ntrue = np.asarray(pk.ntrue_edges, float)
    zf = np.asarray(pk.zf_edges, float)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    reported = 0.5 * (ntrue[:-1] + ntrue[1:]) >= \
        float(np.asarray(pk.nhat_edges, float)[0]) - 1e-9
    kz = np.asarray(pk.kz_to_K)
    zc = np.asarray(pk.zc_edges, float)
    out = {"z_cells": [[float(a), float(b)] for a, b in zip(zf[:-1], zf[1:])],
           "dX_k": [float(x) for x in dX_k], "estimand": {}}
    for thr in thresholds:
        u = np.where(reported, np.clip(ntrue[1:] - np.maximum(ntrue[:-1], thr),
                                       0.0, None), 0.0)            # (B,)
        per_k = np.einsum("dbk,b->dk", f_draws, u)                 # (D, Kf)

        def rec(w, lo, hi, name):
            if w.sum() <= 0:
                return dict(bin=name, z=[lo, hi], available=False)
            pd = (per_k * w[None, :]).sum(axis=1) / w.sum()
            q = np.percentile(pd, [2.5, 16, 50, 84, 97.5])
            return dict(bin=name, z=[float(lo), float(hi)], available=True,
                        dX=float(w.sum()),
                        post_p2p5_16_50_84_97p5=[float(x) for x in q])

        tag = f"ge{thr:.1f}"
        cells = [rec(np.where(np.arange(len(dX_k)) == k, dX_k, 0.0),
                     zf[k], zf[k + 1], f"k{k}") for k in range(len(dX_k))]
        coarse = [rec(np.where(kz == q, dX_k, 0.0), zc[q], zc[q + 1],
                      f"block{q}") for q in range(len(zc) - 1)]
        bins = []
        for name, lo, hi in PAPER1_LOWZ_BINS:
            w = _overlap_w(zf, dX_k, lo, hi)
            r = rec(w, lo, hi, name)
            r["coverage"] = float(np.clip(min(hi, zf[-1]) - max(lo, zf[0]),
                                          0, None) / (hi - lo))
            bins.append(r)
        allz = rec(dX_k, zf[0], zf[-1], "allz")
        out["estimand"][tag] = dict(native_cells=cells, coarse_blocks=coarse,
                                    paper1_bins=bins, allz=allz)
    return out


def omega_20p3_21p6_allz(f_draws, ntrue_edges, zf_edges, dX_k,
                         paper_figures=PAPER_FIGURES):
    """Omega_HI[20.3, 21.6] all-z, posterior percentiles only.

    The weights come from the PAPER'S OWN reduction module, imported READ-ONLY
    exactly as ``validation/fp_ladder/ladder_table.paper_omega_20p3_21p6`` and
    ``validation/long_chain/quantities.py`` do: a ``Posterior`` is built with
    ``__new__`` and only ``f / n_edges / z_edges / dX`` are set, so no
    definition is re-derived here.  The per-draw expression
    ``prefactor * einsum('dbk,b,k->d', f, ow, zw) / zw.sum()`` is VERBATIM
    ``ladder_table.omega_allz_from_weights`` with the truth line deleted.
    """
    if not os.path.isdir(paper_figures):
        return {"unavailable": f"paper_figures not found at {paper_figures}"}
    if paper_figures not in sys.path:
        sys.path.insert(0, paper_figures)
    try:
        import hbi_reduction as HR          # READ-ONLY import
    except Exception as e:                                  # pragma: no cover
        return {"unavailable": f"cannot import hbi_reduction: {type(e).__name__}: {e}"}
    f = np.asarray(f_draws, float)
    P = HR.Posterior.__new__(HR.Posterior)
    P.f = f
    P.n_edges = np.asarray(ntrue_edges, float)
    P.z_edges = np.asarray(zf_edges, float)
    P.dX = np.asarray(dX_k, float)
    ow = P._omega_weight(*HR.OMEGA_NHI)
    zw = P._z_weight(*HR.LOWZ_SUPPORT)
    zs = float(zw.sum())
    if zs <= 0.0:
        return {"unavailable": "the z weights sum to zero over LOWZ_SUPPORT"}
    post = float(HR.OMEGA_PREFACTOR_CM2) * np.einsum("dbk,b,k->d", f, ow, zw) / zs
    q = np.percentile(post, [2.5, 16, 50, 84, 97.5])
    return dict(post_p16_50_84=[float(q[1]), float(q[2]), float(q[3])],
                post_p2p5_97p5=[float(q[0]), float(q[4])],
                dX_total=zs,
                window_nhi=[float(x) for x in HR.OMEGA_NHI],
                window_z=[float(x) for x in HR.LOWZ_SUPPORT],
                prefactor_cm2=float(HR.OMEGA_PREFACTOR_CM2),
                h_reporting=float(getattr(HR, "H_REPORTING", float("nan"))),
                source=("paper_figures/hbi_reduction.py (read-only): "
                        "_omega_weight(*OMEGA_NHI), _z_weight(*LOWZ_SUPPORT), "
                        "OMEGA_PREFACTOR_CM2"),
                units="Omega_HI (dimensionless), paper reporting cosmology")


# --------------------------------------------------------------------------
# the fail-closed "no truth anywhere" scan
# --------------------------------------------------------------------------
#: keys that may legitimately contain the substring "truth": they are SUPPORT
#: CONTRACT FIELD NAMES (support_contract.SUPPORT_FIELDS), not truth values.
#: On the real pack ``truth_host_floor`` is the string "n/a" and
#: ``truth_catalogue_sha256`` is the declared "n/a (REAL DATA: ...)" string.
TRUTH_KEY_WHITELIST = frozenset({
    # support_contract.SUPPORT_FIELDS members (field NAMES, not truth values)
    "truth_host_floor", "truth_catalogue_sha256",
    # the REAL GATE's record that the pack carries the all-zero sentinel.
    # These say that NO truth plane exists; they carry no truth value.
    "truth_counts_sentinel", "truth_counts_all_zero", "truth_counts_shape",
})

#: keys the mock runner emits that must NEVER appear in a real-mode artifact.
FORBIDDEN_KEYS = frozenset({
    "truth", "truth_f", "truth_in_68", "truth_in_95", "median_bias_pct",
    "fp_truth", "hostless_block", "ratio_block", "ratio_nhat_group",
    "census", "diag_fix", "diag_ops", "perz_recovery",
})


#: JSON paths whose KEYS are support_contract PLANE NAMES (``pack.truth_counts``,
#: ``pack.truth_counts_bks``), i.e. names of arrays in the pack, mapped to a
#: support-stamp sha256 / a declared host floor.  They are a data dictionary, not
#: a truth value; the values are still type-checked (str / number / None only).
EXEMPT_SUBTREES = frozenset({"$.support_gate.planes",
                             "$.support_gate.truth_host_floor"})


def assert_no_truth(obj, path="$"):
    """Walk a JSON-able object and raise if it carries a truth-derived field."""
    if path in EXEMPT_SUBTREES:
        for k, v in (obj or {}).items():
            if not isinstance(v, (str, int, float, type(None))):
                raise SystemExit(f"PRIVACY/TRUTH GUARD: {path}.{k} is not a scalar stamp")
        return True
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if ks in FORBIDDEN_KEYS:
                raise SystemExit(f"PRIVACY/TRUTH GUARD: forbidden key {ks!r} at {path}")
            if "truth" in ks.lower() and ks not in TRUTH_KEY_WHITELIST:
                raise SystemExit(f"PRIVACY/TRUTH GUARD: truth-like key {ks!r} at {path}")
            assert_no_truth(v, f"{path}.{ks}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_truth(v, f"{path}[{i}]")
    return True
