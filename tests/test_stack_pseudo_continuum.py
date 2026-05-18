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
