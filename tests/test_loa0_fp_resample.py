"""Acceptance tests for Loa0FP.resample (FIX 3c — grid-invariant AND Ω-safe FP band).

Evolution of the fix:
  * FIX 3 (defect): per-cell Gamma(n_i+1/2). Σ over a tier = (Σn + N_cells/2)/ell — the
    prior half-count is added PER CELL, so the spurious +N_cells/2 grows with grid
    fineness (~+49% FP above 19.5; ~5.2k phantom FP in the empty DLA tier).
  * FIX 3b: anchor the single 1/2 at the TOP logN row. Grid-invariant and gives every
    floor Gamma(Σn+1/2) — BUT places ~83 phantom FP at N~10^22.35: 0.16% of dN/dX
    (harmless) yet Ω-weighted ~100x a DLA, manufacturing a one-sided DOWNWARD Ω(>=20.3)
    bias (measured -6.6% on 2LPT-0) and a NEGATIVE top-bin f_b.
  * FIX 3c (this): anchor the single 1/2 at the LOWEST logN row, where the forest FP
    physically lives (loa-0 counts fall steeply; 0 above logN 20.3) and Ω-weight is
    minimal. The whole-grid total is still Gamma(Σn+1/2) and grid-invariant; every
    higher report floor gets Gamma(Σn_{>=f}) (the 1/2 is negligible for populated tiers;
    an empty high-N tier -> Gamma(0)=0 exactly, so NO Ω mass is manufactured where the
    FP is known negligible). Governing criterion: the band must not manufacture Ω mass
    at a column density where the FP background is negligible.
"""
import numpy as np
import pytest

from CDDF_analysis.hbi import cddf_catalog_hbi as H


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _make_loa0(n_fine, logN_lo, logN_hi, n_sl_loa0=2255.0, vol_scale=165.932,
               n_fp_molly=None):
    """Minimal Loa0FP with band_eta=0 (so mu_fp_grid = n * vol_scale exactly)."""
    n_fine = np.asarray(n_fine, float)
    n_nbins, n_zbins = n_fine.shape
    n_sl_prod = vol_scale * n_sl_loa0
    ell_eff = n_sl_loa0 * (n_sl_loa0 / n_sl_prod)
    if n_fp_molly is None:
        n_fp_molly = np.zeros((3, 4))
    n_fp_molly = np.asarray(n_fp_molly, float)
    return H.Loa0FP(
        n_fp_molly=n_fp_molly,
        b_fp_molly=np.zeros_like(n_fp_molly),
        snr_edges=np.linspace(0, 100, n_fp_molly.shape[0] + 1),
        nhi_edges=np.linspace(19.0, 22.5, n_fp_molly.shape[1] + 1),
        n_fp_fine=n_fine,
        logN_lo=np.asarray(logN_lo, float),
        logN_hi=np.asarray(logN_hi, float),
        band_eta_per_nbin=np.zeros(n_nbins),
        n_sl_loa0=n_sl_loa0, n_sl_prod=n_sl_prod, ell_eff=ell_eff,
    )


def _synthetic_grid():
    """12 logN bins (0.1 wide from 20.0..21.2), 3 z bins. Rows 0..7 populated (the LOW-N
    end, incl. the anchor row 0), rows 8..11 EMPTY (the DLA-like high-N tier). Tier
    boundaries at rows 4 and 8 are multiples of 4 so they stay block edges under x2/x4
    bottom-aligned rebinning; the anchor row 0 stays the lowest row under any merge."""
    logN_lo = np.round(20.0 + 0.1 * np.arange(12), 6)
    logN_hi = np.round(logN_lo + 0.1, 6)
    rng = np.random.default_rng(0)
    n_fine = np.zeros((12, 3))
    n_fine[:8, :] = rng.integers(0, 40, size=(8, 3)).astype(float)
    n_fine[0, 0] = 37.0            # guarantee the anchor row is populated
    n_fine[3, 1] = 0.0             # an interior-empty cell below the tier
    return n_fine, logN_lo, logN_hi


def _rebin_bottom(n_fine, logN_lo, logN_hi, factor):
    """Merge adjacent logN rows in bottom-aligned blocks of `factor`. Rows 4 and 8 stay
    block edges for factor in {1,2,4}; the lowest row (anchor) stays the lowest row."""
    K = n_fine.shape[0]
    rows, lo, hi = [], [], []
    for a in range(0, K, factor):
        b = min(a + factor, K)
        rows.append(n_fine[a:b].sum(axis=0))
        lo.append(logN_lo[a]); hi.append(logN_hi[b - 1])
    return np.array(rows), np.array(lo), np.array(hi)


def _tier_total_samples(loa0, floor, Nmc, seed):
    rng = np.random.default_rng(seed)
    mask = loa0.logN_lo >= floor - 1e-9
    n_nbins, n_zbins = loa0.n_fp_fine.shape
    out = np.empty(Nmc)
    for m in range(Nmc):
        grid = loa0.resample(rng).mu_fp_grid(None, None, n_nbins, n_zbins)
        out[m] = float(grid[mask].sum())
    return out


def _target(n_fine, logN_lo, floor, vol):
    """Grid-independent analytic target: mean/std of the tier total. The +1/2 rides ONLY
    with the lowest row, so a tier gets it iff its floor includes row 0 (the whole grid)."""
    mask = logN_lo >= floor - 1e-9
    n_sum = float(n_fine[mask].sum())
    includes_anchor = bool(mask[0])
    shape = n_sum + (0.5 if includes_anchor else 0.0)
    return shape * vol, np.sqrt(shape) * vol


# ---------------------------------------------------------------------------
# 1. grid-invariance: E and Var of any FIXED tier are invariant under x2/x4 rebin,
#    matched to the grid-independent analytic target (Σn preserved by rebinning).
#    floor_row 0 = whole grid (carries the 1/2); 4 = populated non-anchor tier;
#    8 = empty high-N tier (-> 0, the Ω-safety case).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("floor_row", [0, 4, 8])
def test_resample_grid_invariant_under_rebinning(floor_row):
    n0, lo0, hi0 = _synthetic_grid()
    vol = 165.932
    floor = lo0[floor_row]
    Nmc = 12000
    for factor in (1, 2, 4):
        nf, lo, hi = _rebin_bottom(n0, lo0, hi0, factor)
        tmean, tstd = _target(nf, lo, floor, vol)
        s = _tier_total_samples(_make_loa0(nf, lo, hi, vol_scale=vol),
                                floor, Nmc, seed=100 + factor)
        assert s.mean() == pytest.approx(tmean, rel=0.02, abs=0.05 * vol), (
            f"factor {factor} floor_row {floor_row}: mean {s.mean()} vs {tmean}")
        if tstd > 0:
            assert s.std() == pytest.approx(tstd, rel=0.12)
        else:
            assert s.std() == 0.0        # empty non-anchor tier: no FP variance at all


# ---------------------------------------------------------------------------
# 2. point byte-identity (unchanged path): every accessor == the raw formula EXACTLY.
# ---------------------------------------------------------------------------
def test_point_accessors_bit_identical_to_raw():
    n0, lo0, hi0 = _synthetic_grid()
    vol = 165.932
    n_cat = np.full((3, 4), 5.0)
    molly = np.array([[0.0, 2.0, 0.0, 1.0],
                      [3.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 4.0, 0.0]])
    loa0 = _make_loa0(n0, lo0, hi0, vol_scale=vol, n_fp_molly=molly)
    loa0 = loa0.with_n_cat_molly(n_cat)
    n_nbins, n_zbins = n0.shape
    grid = loa0.mu_fp_grid(None, None, n_nbins, n_zbins)
    assert np.array_equal(grid, n0 * vol)
    assert loa0.mu_fp_scalar() == float(np.sum(n0 * vol))
    keep = lo0 >= 20.4 - 1e-9
    assert loa0.mu_fp_scalar(logN_fit_floor=20.4) == float(np.sum((n0 * vol)[keep]))
    assert np.array_equal(loa0.mu_fp_cell(), molly * vol)
    xhat = np.array([21.0, 20.0, 22.0]); snr = np.array([5.0, 50.0, 90.0])
    got = loa0.lam_fp_per_obj(xhat, snr)
    i, j = loa0._cell_idx(xhat, snr)
    exp = (molly * vol)[i, j] / n_cat[i, j] * (1.0 - loa0._eta_at_nbin(j))
    assert np.array_equal(got, exp)
    assert loa0._gamma_draw is None


# ---------------------------------------------------------------------------
# 3. Ω-SAFETY: the empty HIGH-N tier draws EXACTLY 0 (no phantom ½·vol at high N), so
#    no Ω mass is manufactured and the top-bin mu_FP mean is ~0. The anchor (½·vol) sits
#    at the LOWEST row instead.
# ---------------------------------------------------------------------------
def test_empty_high_tier_is_zero_and_anchor_is_low_N():
    n0, lo0, hi0 = _synthetic_grid()
    vol = 165.932
    loa0 = _make_loa0(n0, lo0, hi0, vol_scale=vol)
    n_nbins, n_zbins = n0.shape
    floor_hi = lo0[8]                         # rows 8..11 all empty (the DLA-like tier)
    s_hi = _tier_total_samples(loa0, floor_hi, Nmc=8000, seed=7)
    assert s_hi.mean() == 0.0 and s_hi.std() == 0.0, (
        "empty high-N tier must manufacture ZERO FP (no phantom, no Ω mass)")

    # the highest logN ROW's resampled mu_FP is exactly 0 across draws (no top anchor).
    rng = np.random.default_rng(11)
    top_row = np.empty(200)
    low_row = np.empty(200)
    for m in range(200):
        g = loa0.resample(rng).mu_fp_grid(None, None, n_nbins, n_zbins)
        top_row[m] = g[-1, :].sum()           # highest logN row
        low_row[m] = g[0, :].sum()            # lowest logN row (the anchor)
    assert np.all(top_row == 0.0), "top logN row must be 0 (empty, un-anchored)"
    # the anchor's lowest row carries n_low + 1/2 -> mean ~ (Σn_row0 + 1/2)*vol
    n_row0 = float(n0[0, :].sum())
    assert low_row.mean() == pytest.approx((n_row0 + 0.5) * vol, rel=0.03)


# ---------------------------------------------------------------------------
# 4. aggregate posterior: whole grid ~ Gamma(Σn + 1/2, scale=vol); a populated non-anchor
#    tier ~ Gamma(Σn_tier, scale=vol); an empty tier -> 0.
# ---------------------------------------------------------------------------
def test_aggregate_posterior_gamma():
    from scipy import stats
    n0, lo0, hi0 = _synthetic_grid()
    vol = 165.932
    loa0 = _make_loa0(n0, lo0, hi0, vol_scale=vol)
    # whole grid: Gamma(Σn + 1/2)
    s = _tier_total_samples(loa0, lo0[0], Nmc=20000, seed=3)
    n_all = float(n0.sum())
    ks = stats.kstest(s, 'gamma', args=(n_all + 0.5, 0.0, vol))
    assert ks.pvalue > 0.01, (n_all, ks)
    assert s.mean() == pytest.approx((n_all + 0.5) * vol, rel=0.02)
    # populated non-anchor tier (rows>=4): Gamma(Σn_{>=4}) — NO extra 1/2
    s4 = _tier_total_samples(loa0, lo0[4], Nmc=20000, seed=4)
    n4 = float(n0[lo0 >= lo0[4] - 1e-9].sum())
    ks4 = stats.kstest(s4, 'gamma', args=(n4, 0.0, vol))
    assert ks4.pvalue > 0.01, (n4, ks4)
    assert s4.mean() == pytest.approx(n4 * vol, rel=0.02)


# ---------------------------------------------------------------------------
# 5. regression guard: no per-cell +1/2 inflation on the grand total (the 1/2 appears
#    ONCE, not N_cells times).
# ---------------------------------------------------------------------------
def test_no_per_cell_half_count_inflation_on_grand_total():
    n0, lo0, hi0 = _synthetic_grid()
    vol = 165.932
    loa0 = _make_loa0(n0, lo0, hi0, vol_scale=vol)
    n_sum = float(n0.sum()); n_cells = n0.size
    old_biased = (n_sum + 0.5 * n_cells) * vol
    correct = (n_sum + 0.5) * vol
    s = _tier_total_samples(loa0, lo0[0], Nmc=12000, seed=5)
    assert s.mean() == pytest.approx(correct, rel=0.02)
    assert s.mean() < 0.5 * (old_biased - correct) + correct
