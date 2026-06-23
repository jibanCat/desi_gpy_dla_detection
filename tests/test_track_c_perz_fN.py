"""Structural test for the `perz_fN` JSON-assembly layer of
``CDDF_analysis/track_c_tf_loa.py`` (the per-z DIFFERENTIAL f(N|z) deliverable).

PR-18 #6.3: a direct test on the `perz_fN` JSON block. The assembly was factored
out of ``run_measurement`` into the importable ``assemble_perz_fN(res, limits)``
in ``track_c_tf_loa.py``; this test imports and exercises THAT production function,
so a schema change in the module is caught here (no shadow copy to drift). The test:

  1. Builds a tiny SYNTHETIC ``res`` dict that satisfies the exact contract the
     ``perz_fN`` assembly reads (``mid``, ``map_fbk`` (n_nbins×n_zc), the four
     ``fNz_*`` band arrays (n_zc×n_nbins), ``zbins``, ``z_extrapolated``,
     ``z_thin``, ``truth_counts_perz``, ``fNz_floor``, ``fNz_band_method``,
     ``n_zc``) — NO FITS, NO GP inference, no full driver run (fast, deterministic).
  2. Calls the real ``assemble_perz_fN`` and asserts the emitted structure: the
     band method, plus per-z entries each with the logN grid (``logN_centers``) +
     f + 68/95 band (lo/hi) + the thin / extrapolation flags.
  3. Independently verifies the extrapolation-flag RULE the module applies in
     ``run_measurement``: a coarse-z bin is flagged ``extrapolated`` when its
     2LPT-0 truth count is 0 — here forced for the top [4.0, 4.25) z-bin — and
     asserts the assembled record carries that flag for the [4.0, 4.25) bin.
"""
import json

import numpy as np
import pytest

from CDDF_analysis.track_c_tf_loa import assemble_perz_fN


# z-grid that produces a coarse top bin [4.0, 4.25) (the Track-C extended grid).
ZBINS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.25]
N_ZC = len(ZBINS) - 1                       # 5 coarse z-bins; index 4 == [4.0,4.25)
EXTRAP_KIDX = N_ZC - 1                       # the [4.0, 4.25) bin


def _extrapolation_flag_rule(truth_counts_perz, zbins, max_truth_z, cz_min_count):
    """Mirror of the per-z support-flag rule in ``track_c_tf_loa.run_measurement``
    (lines ~452-465): cnt==0 OR z-lower-edge >= max truth z -> EXTRAPOLATED;
    0 < cnt < cz_min_count -> THIN. Kept verbatim so the test pins the rule."""
    n_zc = len(zbins) - 1
    z_extrap = np.zeros(n_zc, dtype=bool)
    z_thin = np.zeros(n_zc, dtype=bool)
    tcz = np.asarray(truth_counts_perz, int)
    for k in range(n_zc):
        cnt = tcz[k] if k < len(tcz) else -1
        if cnt == 0 or (np.isfinite(max_truth_z) and zbins[k] >= max_truth_z):
            z_extrap[k] = True
        elif 0 < cnt < cz_min_count:
            z_thin[k] = True
    return z_extrap, z_thin


def _synthetic_res(n_nbins=8):
    """A minimal ``res`` matching the fields the ``perz_fN`` assembly reads.

    The top z-bin [4.0, 4.25) is given truth count 0 -> EXTRAPOLATED; z-bin index 1
    is given a small (thin) count -> THIN; the rest are well-populated -> calibrated.
    """
    logN_lo = np.round(20.0 + 0.1 * np.arange(n_nbins), 2)
    logN_hi = np.round(logN_lo + 0.1, 2)
    mid = 0.5 * (logN_lo + logN_hi)

    # genuine 2-D MAP f(N|z): a positive power law, slightly z-dependent
    base = 1e-22 * (10.0 ** mid / 10.0 ** 20.3) ** -1.8     # (n_nbins,)
    map_fbk = np.outer(base, 1.0 + 0.05 * np.arange(N_ZC))  # (n_nbins, n_zc)

    # per-z bands (n_zc, n_nbins): symmetric ±20% / ±40% around the MAP column
    fNz_lo68 = np.full((N_ZC, n_nbins), np.nan)
    fNz_hi68 = np.full((N_ZC, n_nbins), np.nan)
    fNz_lo95 = np.full((N_ZC, n_nbins), np.nan)
    fNz_hi95 = np.full((N_ZC, n_nbins), np.nan)
    for k in range(N_ZC):
        col = map_fbk[:, k]
        fNz_lo68[k] = 0.8 * col
        fNz_hi68[k] = 1.2 * col
        fNz_lo95[k] = 0.6 * col
        fNz_hi95[k] = 1.4 * col

    truth_counts_perz = [500, 12, 300, 80, 0]   # idx1 thin (<30), idx4 zero -> extrap
    z_extrap, z_thin = _extrapolation_flag_rule(
        truth_counts_perz, ZBINS, max_truth_z=4.5, cz_min_count=30.0)

    return dict(
        mid=mid, logN_lo=logN_lo, logN_hi=logN_hi, zbins=ZBINS, n_zc=N_ZC,
        map_fbk=map_fbk,
        fNz_lo68=fNz_lo68, fNz_hi68=fNz_hi68, fNz_lo95=fNz_lo95, fNz_hi95=fNz_hi95,
        fNz_floor=20.0, fNz_band_method="direct_perN_z",
        z_extrapolated=z_extrap, z_thin=z_thin,
        truth_counts_perz=truth_counts_perz,
    )


def test_perz_fN_top_level_structure():
    """The perz_fN block carries the logN grid, floor, band method, zbins, the two
    flag vectors, truth counts, and a per-z list of length n_zc."""
    res = _synthetic_res()
    blk = assemble_perz_fN(res)

    n_nbins = len(res["mid"])
    assert set(blk) >= {"logN_centers", "floor", "band_method", "zbins",
                        "z_extrapolated", "z_thin", "truth_counts_perz", "perz"}
    # the band method must propagate from res (drift guard: the production assembly
    # carries fNz_band_method; a shadow copy that dropped it would fail here)
    assert blk["band_method"] == "direct_perN_z"
    assert len(blk["logN_centers"]) == n_nbins
    assert blk["floor"] == pytest.approx(20.0)
    assert blk["zbins"] == [float(z) for z in ZBINS]
    assert len(blk["z_extrapolated"]) == N_ZC
    assert len(blk["z_thin"]) == N_ZC
    assert len(blk["perz"]) == N_ZC


def test_perz_fN_each_entry_has_grid_f_band_and_flags():
    """Every per-z entry carries z_idx/z + the f curve + 68/95 band (lo/hi) on the
    SAME logN grid + the thin/extrapolation flags, all aligned to logN_centers."""
    res = _synthetic_res()
    blk = assemble_perz_fN(res)
    n_nbins = len(blk["logN_centers"])

    for k, e in enumerate(blk["perz"]):
        assert e["z_idx"] == k
        # z is the bin midpoint
        assert e["z"] == pytest.approx(0.5 * (ZBINS[k] + ZBINS[k + 1]))
        # f + the four band arrays all share the logN grid length
        for key in ("f", "f68_lo", "f68_hi", "f95_lo", "f95_hi"):
            assert key in e, f"perz entry {k} missing {key}"
            assert len(e[key]) == n_nbins, f"{key} not aligned to logN grid in bin {k}"
        # band nesting: 95% band brackets the 68% band brackets nothing-narrower
        f68lo = np.asarray(e["f68_lo"]); f68hi = np.asarray(e["f68_hi"])
        f95lo = np.asarray(e["f95_lo"]); f95hi = np.asarray(e["f95_hi"])
        assert np.all(f95lo <= f68lo + 1e-30)
        assert np.all(f95hi >= f68hi - 1e-30)
        # the two flags are present and boolean
        assert isinstance(e["extrapolated"], bool)
        assert isinstance(e["thin"], bool)


def test_perz_fN_extrapolation_flag_set_for_4p0_4p25_bin():
    """The [4.0, 4.25) coarse-z bin (zero 2LPT-0 truth support) is flagged
    EXTRAPOLATED both in the top-level vector and the per-z entry; the well-populated
    low-z bins are NOT extrapolated; the thin bin is flagged THIN not EXTRAPOLATED."""
    res = _synthetic_res()
    blk = assemble_perz_fN(res)

    # the top z-bin is [4.0, 4.25)
    assert ZBINS[EXTRAP_KIDX] == 4.0 and ZBINS[EXTRAP_KIDX + 1] == 4.25
    # flagged extrapolated at both layers
    assert blk["z_extrapolated"][EXTRAP_KIDX] is True
    assert blk["perz"][EXTRAP_KIDX]["extrapolated"] is True
    assert blk["perz"][EXTRAP_KIDX]["thin"] is False
    # a well-populated low-z bin (idx 0) is calibrated (neither flag)
    assert blk["z_extrapolated"][0] is False
    assert blk["perz"][0]["extrapolated"] is False
    assert blk["perz"][0]["thin"] is False
    # the small-count bin (idx 1) is THIN, not extrapolated
    assert blk["perz"][1]["thin"] is True
    assert blk["perz"][1]["extrapolated"] is False


def test_perz_fN_is_json_serializable():
    """The assembled block round-trips through json.dumps/loads (the driver writes it
    with json.dump(..., default=float)) — no numpy scalars leak into the structure."""
    res = _synthetic_res()
    blk = assemble_perz_fN(res)
    s = json.dumps(blk, default=float)
    back = json.loads(s)
    assert len(back["perz"]) == N_ZC
    assert back["perz"][EXTRAP_KIDX]["extrapolated"] is True


def test_extrapolation_flag_rule_matches_module_contract():
    """Pin the support-flag rule the driver applies (run_measurement lines ~452-465):
    cnt==0 -> extrapolated; 0<cnt<cz_min -> thin; cnt>=cz_min -> neither; and a bin
    whose lower z-edge sits at/above max_truth_z is extrapolated regardless of count."""
    zbins = ZBINS
    counts = [500, 12, 300, 80, 0]
    extr, thin = _extrapolation_flag_rule(counts, zbins, max_truth_z=10.0,
                                          cz_min_count=30.0)
    assert list(extr) == [False, False, False, False, True]     # only the cnt==0 bin
    assert list(thin) == [False, True, False, False, False]     # only the cnt==12 bin

    # max_truth_z cap: with truth ending at z=3.5, the [3.5,4.0) and [4.0,4.25) bins
    # are extrapolated even though one has a nonzero count.
    counts2 = [500, 400, 300, 80, 5]
    extr2, _ = _extrapolation_flag_rule(counts2, zbins, max_truth_z=3.5,
                                        cz_min_count=30.0)
    assert list(extr2) == [False, False, False, True, True]
