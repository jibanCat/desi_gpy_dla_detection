"""B16 — the z-leaky truth reduction. FIXTURE-FIRST regression test.

THE DEFECT (both sites, pre-fix)
--------------------------------
The truth-side f(N) numerator counted EVERY truth absorber whose logN fell in the fine
grid, with NO redshift mask, and divided by

    X_sum = Σ_k ΔX_k   (built by total_DeltaX_in_zbins over cfg.zbins ONLY)

so the denominator carried pathlength for z ∈ [zbins[0], zbins[-1]) alone.  Rows with
Z_DLA OUTSIDE that range therefore entered the NUMERATOR against ZERO matching
pathlength in the DENOMINATOR.

    CDDF_analysis/hbi/cddf_tilt_closure.py  (pre-fix ~:144)   valid = (nidx >= 0)
    CDDF_analysis/hbi/cddf_catalog_hbi.py   (pre-fix ~:2141)  n = int((t_nidx == b).sum())

Consequence: f_truth is inflated bin-by-bin, and Ω = K·Σ N_b·f_truth·ΔN_b inherits the
inflation.  dN/dX was ALWAYS CLEAN — `dndx_total` already carried `& (t_zidx >= 0)` and
`dndx_z` reads the correctly-masked Wbk / per-z counts.

CORRECT SEMANTICS (stated before the change was made)
-----------------------------------------------------
The truth f(N) must be built from EXACTLY the same (N, z) support as the ΔX it is
divided by.  A truth row contributes to f_truth[b] iff

    (a) it survives the SNR cut  S2N_RED > cfg.snr_min                (already enforced)
    (b) its logN lands in a kept fine bin  -> _bin_index_logN >= 0    (already enforced)
    (c) its Z_DLA lands in a kept coarse z bin -> _zbin_index >= 0    (WAS MISSING)

Rows in the numerator but not the denominator (pre-fix) = rows satisfying (a) and (b)
but NOT (c): Z_DLA < zbins[0] or Z_DLA >= zbins[-1].

WHY THE FIXTURE NEEDS OUT-OF-WINDOW ROWS
----------------------------------------
Without truth rows that are IN the N window but OUT of the z window the two code paths
are numerically identical and the bug is INVISIBLE.  The table below deliberately
contains three such rows (two above/below the z range at low N, one below at high N),
one row that is OUT of the N window but IN the z window (must be dropped by BOTH), and
one row below the SNR cut (must be dropped by BOTH).

All values here are SYNTHETIC (no mock, no real data).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

Table = pytest.importorskip("astropy.table").Table

from CDDF_analysis.hbi.cddf_catalog_hbi import (  # noqa: E402
    HBIConfig, truth_reductions, omega_hi_prefactor,
)
from CDDF_analysis.hbi.cddf_tilt_closure import tilted_truth_reductions  # noqa: E402


# ---------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------
# coarse z grid: two bins, [2.0, 2.5) and [2.5, 3.0).  z >= 3.0 and z < 2.0 are OUT.
ZBINS = (2.0, 2.5, 3.0)
SNR_MIN = 2.0

# two fine N bins.  N_b / dN_b are chosen as round powers of ten so the Omega sum is
# exact in binary floating point and can be checked by hand.
LOGN_LO = np.array([20.0, 20.3])
LOGN_HI = np.array([20.3, 21.0])
N_B = np.array([1.0e20, 1.0e21])
DN_B = np.array([1.0e20, 1.0e21])

# pathlength per coarse z bin -> X_sum = 4.0
X_TOT = np.array([3.0, 1.0])
X_SUM = 4.0

# row |  NHI  | Z_DLA | S2N_RED | nidx | zidx | verdict
# ----+-------+-------+---------+------+------+-----------------------------------------
#  1  | 20.1  |  2.2  |  10     |  0   |  0   | IN  N, IN  z   -> counted (both paths)
#  2  | 20.1  |  2.7  |  10     |  0   |  1   | IN  N, IN  z   -> counted (both paths)
#  3  | 20.1  |  3.4  |  10     |  0   | -1   | IN  N, OUT z   -> *** THE LEAK ***
#  4  | 20.5  |  2.2  |  10     |  1   |  0   | IN  N, IN  z   -> counted (both paths)
#  5  | 20.5  |  1.5  |  10     |  1   | -1   | IN  N, OUT z   -> *** THE LEAK ***
#  6  | 19.0  |  2.2  |  10     | -1   |  0   | OUT N, IN  z   -> dropped by BOTH
#  7  | 20.1  |  1.0  |  10     |  0   | -1   | IN  N, OUT z   -> *** THE LEAK ***
#  8  | 20.1  |  2.2  |   1     |  -   |  -   | below SNR cut  -> dropped by BOTH
TRUTH_NHI = [20.1, 20.1, 20.1, 20.5, 20.5, 19.0, 20.1, 20.1]
TRUTH_Z = [2.2, 2.7, 3.4, 2.2, 1.5, 2.2, 1.0, 2.2]
TRUTH_SNR = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0]


def _truth_table():
    return Table({"NHI": np.array(TRUTH_NHI, float),
                  "Z_DLA": np.array(TRUTH_Z, float),
                  "S2N_RED": np.array(TRUTH_SNR, float)})


def _cfg():
    return HBIConfig(catalog_dir="<fixture>", truth_path="<fixture>",
                     bal_cat_path="<fixture>", molly_tsv="<fixture>",
                     out_dir="<fixture>",
                     zbins=ZBINS, report_logN_limits=(20.0, 20.3), snr_min=SNR_MIN)


# ---------------------------------------------------------------------------
# HAND-COMPUTED EXPECTATIONS  (arithmetic shown; no code was consulted)
# ---------------------------------------------------------------------------
# counts, SNR cut applied (row 8 gone):
#   bin 0 = [20.0, 20.3):  no z mask -> rows 1,2,3,7 = 4 ;  z-masked -> rows 1,2   = 2
#   bin 1 = [20.3, 21.0):  no z mask -> rows 4,5     = 2 ;  z-masked -> row  4     = 1
#
# CORRECT (post-fix) f_truth = count_zmasked / (X_sum * dN_b):
#   f[0] = 2 / (4.0 * 1e20) = 5.0e-21
#   f[1] = 1 / (4.0 * 1e21) = 2.5e-22
FIX_F_TRUTH = np.array([5.0e-21, 2.5e-22])

# BUGGY (pre-fix) f_truth = count_all_z / (X_sum * dN_b):
#   f[0] = 4 / (4.0 * 1e20) = 1.0e-20
#   f[1] = 2 / (4.0 * 1e21) = 5.0e-22
BUG_F_TRUTH = np.array([1.0e-20, 5.0e-22])

# Omega = K * Σ_{b in sel} N_b * f_truth[b] * dN_b[b],  K = omega_hi_prefactor(H0).
# We pin the K-free sum so the literal is exact and H0-independent.
#   lim = 20.0 -> sel = both bins
#     CORRECT: 1e20*5.0e-21*1e20 + 1e21*2.5e-22*1e21 = 0.5e20 + 2.5e20 = 3.0e20
#     BUGGY  : 1e20*1.0e-20*1e20 + 1e21*5.0e-22*1e21 = 1.0e20 + 5.0e20 = 6.0e20
#   lim = 20.3 -> sel = bin 1 only
#     CORRECT: 1e21*2.5e-22*1e21 = 2.5e20
#     BUGGY  : 1e21*5.0e-22*1e21 = 5.0e20
FIX_OMEGA_OVER_K = {20.0: 3.0e20, 20.3: 2.5e20}
BUG_OMEGA_OVER_K = {20.0: 6.0e20, 20.3: 5.0e20}
# on THIS fixture the leak factor is 2.0 at 20.0 and 2.0 at 20.3 -- deliberately huge.
# On the real 2LPT-0 cut bundle it is O(1.04-1.06). The leak is NOT a scalar: it is the
# ratio of two COUNTS and varies with the z grid, the N window and the cut bundle.

# dN/dX is CLEAN and must be IDENTICAL before and after the fix.
#   lim = 20.0: rows with NHI>=20.0 & <22.4 & zidx>=0 = rows 1,2,4 = 3 -> 3/4.0 = 0.75
#   lim = 20.3: rows with NHI>=20.3 & <22.4 & zidx>=0 = row  4     = 1 -> 1/4.0 = 0.25
DNDX_TOTAL = {20.0: 0.75, 20.3: 0.25}

# tilted_truth_reductions also returns dN/dX(z) = Σ_{N>=lim} W_bk[:,k] / X[k]; W_bk was
# ALREADY z-masked pre-fix, so these are unchanged too.
#   lim = 20.0: z bin 0 -> (row1 + row4)/3.0 = 2/3 ; z bin 1 -> (row2)/1.0 = 1.0
#   lim = 20.3: z bin 0 -> (row4)/3.0 = 1/3       ; z bin 1 -> 0/1.0       = 0.0
DNDX_Z = {20.0: np.array([2.0 / 3.0, 1.0]), 20.3: np.array([1.0 / 3.0, 0.0])}


# ===========================================================================
# site 2 -- cddf_catalog_hbi.truth_reductions
# ===========================================================================
def test_catalog_hbi_truth_reductions_f_truth_is_z_masked():
    """f_truth must equal the HAND-COMPUTED z-masked value, NOT the leaky one."""
    tr = truth_reductions(_cfg(), _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B, X_TOT)
    np.testing.assert_allclose(tr["f_truth"], FIX_F_TRUTH, rtol=1e-12)
    # and it must NOT be the pre-fix value (guards a silent revert)
    assert not np.allclose(tr["f_truth"], BUG_F_TRUTH, rtol=1e-6, atol=0.0)


def test_catalog_hbi_truth_reductions_omega_is_z_masked():
    cfg = _cfg()
    K = omega_hi_prefactor(cfg.H0)
    tr = truth_reductions(cfg, _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B, X_TOT)
    for lim, want in FIX_OMEGA_OVER_K.items():
        assert tr["omega"][lim] / K == pytest.approx(want, rel=1e-12)
        assert tr["omega"][lim] / K != pytest.approx(BUG_OMEGA_OVER_K[lim], rel=1e-9)


def test_catalog_hbi_truth_reductions_dndx_unchanged():
    """dN/dX was NEVER leaky. This pins that the fix did not perturb it."""
    tr = truth_reductions(_cfg(), _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B, X_TOT)
    for lim, want in DNDX_TOTAL.items():
        assert tr["dndx_total"][lim] == pytest.approx(want, rel=1e-12)


# ===========================================================================
# site 1 -- cddf_tilt_closure.tilted_truth_reductions (Delta-alpha = 0 => w == 1)
# ===========================================================================
def test_tilt_closure_truth_f_and_omega_are_z_masked():
    cfg = _cfg()
    K = omega_hi_prefactor(cfg.H0)
    t0 = tilted_truth_reductions(cfg, _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B,
                                 X_TOT, 0.0)
    np.testing.assert_allclose(t0["f_truth"], FIX_F_TRUTH, rtol=1e-12)
    assert not np.allclose(t0["f_truth"], BUG_F_TRUTH, rtol=1e-6, atol=0.0)
    for lim, want in FIX_OMEGA_OVER_K.items():
        assert t0["omega"][lim] / K == pytest.approx(want, rel=1e-12)


def test_tilt_closure_dndx_paths_unchanged():
    t0 = tilted_truth_reductions(_cfg(), _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B,
                                 X_TOT, 0.0)
    for lim, want in DNDX_TOTAL.items():
        assert t0["dndx_total"][lim] == pytest.approx(want, rel=1e-12)
    for lim, want in DNDX_Z.items():
        np.testing.assert_allclose(t0["dndx_z"][lim], want, rtol=1e-12)


def test_two_sites_agree_bitforbit():
    """The two independent implementations of the SAME truth reduction must agree.
    Pre-fix they also agreed (both were leaky) -- this is a permanent invariant, and it
    is what makes a one-site-only fix impossible to land silently."""
    cfg = _cfg()
    tr = truth_reductions(cfg, _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B, X_TOT)
    t0 = tilted_truth_reductions(cfg, _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B,
                                 X_TOT, 0.0)
    np.testing.assert_array_equal(tr["f_truth"], t0["f_truth"])
    for lim in cfg.report_logN_limits:
        assert tr["omega"][lim] == t0["omega"][lim]
        assert tr["dndx_total"][lim] == t0["dndx_total"][lim]


# ===========================================================================
# the leak is DIRECTLY exhibited: same table, wider z grid -> f_truth grows
# ===========================================================================
def test_widening_the_z_grid_recovers_the_leaked_rows():
    """A DIRECT demonstration that the numerator now tracks the denominator's support.
    Widen zbins to (1.0, 2.0, 2.5, 3.0, 3.5) and give the two new bins pathlength; the
    three previously-leaked rows (3, 5, 7) must now appear in f_truth, and the counts
    must be exactly the pre-fix all-z counts (4 and 2)."""
    cfg = HBIConfig(catalog_dir="<fixture>", truth_path="<fixture>",
                    bal_cat_path="<fixture>", molly_tsv="<fixture>", out_dir="<fixture>",
                    zbins=(1.0, 2.0, 2.5, 3.0, 3.5), report_logN_limits=(20.0, 20.3),
                    snr_min=SNR_MIN)
    # ALL 4 z bins now carry pathlength; keep X_sum = 4.0 so f_truth is comparable
    X_wide = np.array([0.5, 2.5, 0.5, 0.5])
    assert X_wide.sum() == X_SUM
    tr = truth_reductions(cfg, _truth_table(), LOGN_LO, LOGN_HI, N_B, DN_B, X_wide)
    # every surviving row is now inside the z support -> the pre-fix (leaky) numbers
    np.testing.assert_allclose(tr["f_truth"], BUG_F_TRUTH, rtol=1e-12)
