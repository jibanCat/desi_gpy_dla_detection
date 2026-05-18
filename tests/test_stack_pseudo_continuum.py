"""Unit tests for the masked-spline pseudo-continuum fit in
examples/stack_real_loa_dlas.py — built on a synthetic composite with a
known truth pseudo-continuum and known injected absorption lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.stack_real_loa_dlas import (  # noqa: E402
    REST_LAMBDA_MIN, REST_LAMBDA_MAX, DLOG_LAMBDA, PCONT_LAMBDA_MIN,
    SIGMA_V, _C_KM_S, METAL_LINES, _continuum_mask,
)


def _rest_grid():
    return 10 ** np.arange(np.log10(REST_LAMBDA_MIN),
                           np.log10(REST_LAMBDA_MAX), DLOG_LAMBDA)


def make_mock_composite(rest_grid, *, inject=True, seed=0):
    """Synthetic composite with a known truth pseudo-continuum.

    Returns (curve, counts, P_true, lines) where `lines` is a list of
    (lambda0, depth, sigma) for each injected absorption line."""
    rng = np.random.default_rng(seed)
    lam = rest_grid
    # truth pseudo-continuum: gentle slope * smeared QSO Lya bump * forest decrement
    slope = 1.0 + 0.10 * (lam - 1200.0) / 900.0
    bump = 1.0 + 0.25 * np.exp(-0.5 * ((lam - 1280.0) / 60.0) ** 2)
    forest = 0.6 + 0.4 / (1.0 + np.exp(-(lam - 1180.0) / 25.0))
    P_true = slope * bump * forest
    # counts: ramp up from the blue edge, with one mid-band coverage hole
    counts = np.clip(50.0 + 0.9 * (lam - 700.0), 0.0, 800.0)
    counts[(lam > 1080.0) & (lam < 1090.0)] = 0.0
    # injected absorption
    absorption = np.zeros_like(lam)
    lines = []
    if inject:
        specs = [(1031.91, 0.20), (1063.18, 0.12), (1143.23, 0.10),
                 (1190.42, 0.18), (1260.42, 0.22), (1334.53, 0.15),
                 (1393.76, 0.25), (1548.20, 0.35),
                 (1117.0, 0.06), (1450.0, 0.05)]  # last 2: NOT in METAL_LINES
        for lam0, depth in specs:
            sig = lam0 * SIGMA_V / _C_KM_S
            absorption += depth * np.exp(-0.5 * ((lam - lam0) / sig) ** 2)
            lines.append((lam0, depth, sig))
    curve = P_true * (1.0 - absorption)
    noise = np.where(counts > 0,
                     0.02 / np.sqrt(np.maximum(counts, 1.0) / 400.0), 0.0)
    curve = curve + rng.normal(0.0, np.maximum(noise, 1e-6))
    curve[counts < 50] = np.nan
    return curve, counts, P_true, lines


def test_continuum_mask_excludes_lines_and_blue_end():
    rg = _rest_grid()
    curve, counts, _, lines = make_mock_composite(rg)
    fit_ok = _continuum_mask(rg, curve, counts)
    # nothing below PCONT_LAMBDA_MIN is kept
    assert not fit_ok[rg < PCONT_LAMBDA_MIN].any()
    # the CIV 1548 metal line centre is masked out
    civ = np.argmin(np.abs(rg - 1548.20))
    assert not fit_ok[civ]
    # a clean window (1500 A, no METAL_LINES within a few A) is kept
    clean = np.argmin(np.abs(rg - 1500.0))
    assert fit_ok[clean]
    # the coverage hole [1080,1090] is excluded
    assert not fit_ok[(rg > 1081.0) & (rg < 1089.0)].any()


from examples.stack_real_loa_dlas import fit_pseudo_continuum  # noqa: E402


def _offline_mask(rest_grid, lines, pad_sigma=6.0):
    """Pixels far from every injected line and away from the 945 Å edge."""
    ok = rest_grid >= 960.0
    for lam0, _depth, sig in lines:
        ok = ok & (np.abs(rest_grid - lam0) > pad_sigma * sig)
    return ok


def test_pcont_nan_below_945_finite_above():
    rg = _rest_grid()
    curve, counts, _, _ = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    assert np.all(np.isnan(P[rg < PCONT_LAMBDA_MIN]))
    mid = (rg > 1000.0) & (rg < 1550.0)
    assert np.isfinite(P[mid]).mean() > 0.99


def test_pcont_recovers_truth_off_lines():
    rg = _rest_grid()
    curve, counts, P_true, lines = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    off = _offline_mask(rg, lines) & np.isfinite(P) & (counts >= 50)
    rel = np.abs(P[off] / P_true[off] - 1.0)
    assert np.nanmedian(rel) < 0.02
    assert np.nanpercentile(rel, 95) < 0.04


def test_pcont_null_case_flat():
    rg = _rest_grid()
    curve, counts, _, _ = make_mock_composite(rg, inject=False)
    P = fit_pseudo_continuum(rg, curve, counts)
    norm = curve / P
    ok = np.isfinite(norm) & (rg > 960.0) & (rg < 1590.0)
    assert abs(np.nanmedian(norm[ok]) - 1.0) < 0.01
    assert np.nanstd(norm[ok]) < 0.05


def test_pcont_lines_survive_normalization():
    rg = _rest_grid()
    curve, counts, _, lines = make_mock_composite(rg)
    P = fit_pseudo_continuum(rg, curve, counts)
    norm = curve / P
    for lam0, depth, sig in lines:
        if lam0 < PCONT_LAMBDA_MIN:
            continue
        core = np.abs(rg - lam0) < 2.0 * sig
        measured = 1.0 - np.nanmin(norm[core])
        assert measured > 0.6 * depth, f"line {lam0} eaten: {measured} vs {depth}"
