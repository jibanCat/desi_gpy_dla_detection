"""Tests for validation/absorber_ladder/fp_target_review — the FP CALIBRATION
TARGET DEFINITION REVIEW (PI ruling 2026-09-13c §7).

Two layers:
  * PURE layer — the event definitions of both estimands and the
    decomposition, exercised on a synthetic catalogue (no data needed);
  * PRODUCT layer — the built npz products: the decomposition must be a
    PARTITION, every product must carry a ``support_id``, and the mock
    products' support must equal the A0v2 census's (fail closed).  Skipped
    when the products are not on disk.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_FPTR = os.path.join(_REPO, "validation", "absorber_ladder", "fp_target_review")
_SUPPORT = os.path.join(_REPO, "validation", "absorber_ladder", "support")
for _p in (_FPTR, _SUPPORT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selection as SEL                                          # noqa: E402

PRODUCTS = ("/scratch/cavestru_root/cavestru0/mfho/"
            "fp_target_review_2026-09-13")
A0_SUPPORT_DIR = ("/scratch/cavestru_root/cavestru0/mfho/"
                  "absorber_ladder_2026-09-13/support")
FAMILIES = ("2lpt0", "london0", "saclay0")


# ---------------------------------------------------------------------------
# a synthetic loa-0-like catalogue, built so every cut has something to bite on
# ---------------------------------------------------------------------------
def synthetic_catalogue():
    #            snr   pdla   nhi    z_dla  z_qso
    rows = [
        (5.0, 0.995, 19.8, 2.50, 3.00),   # 0 passes everything
        (2.0, 0.995, 19.8, 2.50, 3.00),   # 1 snr EXACTLY at the floor -> out
        (2.001, 0.995, 19.8, 2.50, 3.00),  # 2 just above -> in
        (5.0, 0.99, 19.8, 2.50, 3.00),    # 3 p_dla EXACTLY at the floor -> out
        (5.0, 0.995, 19.8, 2.05, 3.00),   # 4 lam_rest = 1215.67*3.05/4 = 927 -> out
        (5.0, 0.995, 19.8, 1.99, 2.20),   # 5 z_dla below 2.0 -> out of the z window
        (5.0, 0.995, 19.8, 3.50, 3.90),   # 6 z_dla == 3.5 (half open) -> out
        (5.0, 0.995, 19.4, 2.50, 3.00),   # 7 below the 19.5 N-hat floor
        (5.0, 0.995, 22.5, 2.50, 3.00),   # 8 above the 22.4 grid top
        (0.5, 0.995, 19.8, 2.50, 3.00),   # 9 dead stratum s=0
    ]
    a = np.array(rows, float)
    return dict(snr=a[:, 0], pdla=a[:, 1], nhi=a[:, 2], z_dla=a[:, 3],
                z_qso=a[:, 4])


def test_lam_rest_and_the_lya_only_cut():
    c = synthetic_catalogue()
    lam = SEL.lam_rest(c["z_dla"], c["z_qso"])
    assert lam[0] == pytest.approx(1215.67 * 3.50 / 4.00, abs=0)
    assert lam[4] < 1025.0 and lam[0] >= 1025.0


def test_loa0_op_mask_is_strict_on_both_thresholds():
    c = synthetic_catalogue()
    m = SEL.loa0_op_mask(c["snr"], c["pdla"], c["z_dla"], c["z_qso"])
    assert not m[1], "S/N == snr_min must FAIL (the cut is strict >)"
    assert m[2], "S/N just above snr_min must pass"
    assert not m[3], "P_DLA == p_dla_min must FAIL (the cut is strict >)"
    assert not m[4], "the Lya-only lam_rest >= 1025 cut must bite"
    assert not m[5] and not m[6], "the [2.0, 3.5) z window must bite, half open"
    assert m[0] and m[7] and m[8]
    assert not m[9], "S/N 0.5 is below the op cut; no dead-stratum row survives"


def test_loa0_op_mask_layers_are_separable():
    c = synthetic_catalogue()
    base = SEL.loa0_op_mask(c["snr"], c["pdla"], c["z_dla"], c["z_qso"],
                            lam_rf_min=None, z_lo=None, z_hi=None)
    lya = SEL.loa0_op_mask(c["snr"], c["pdla"], c["z_dla"], c["z_qso"],
                           z_lo=None, z_hi=None)
    full = SEL.loa0_op_mask(c["snr"], c["pdla"], c["z_dla"], c["z_qso"])
    floored = SEL.loa0_op_mask(c["snr"], c["pdla"], c["z_dla"], c["z_qso"],
                               nhi=c["nhi"], nhat_lo=19.5)
    assert base.sum() >= lya.sum() >= full.sum() >= floored.sum()
    assert np.all(lya <= base) and np.all(full <= lya) and np.all(floored <= full)


def test_collar_window_narrows_monotonically_and_nests():
    zq = np.linspace(2.1, 4.2, 40)
    lo0, hi0 = SEL.collar_window(zq, collar_kms=0.0)
    lo3, hi3 = SEL.collar_window(zq, collar_kms=3000.0)
    lo33, hi33 = SEL.collar_window(zq, collar_kms=3300.0)
    assert np.all(lo33 >= lo3 - 1e-12) and np.all(hi33 <= hi3 + 1e-12)
    assert np.all(lo3 >= lo0 - 1e-12) and np.all(hi3 <= hi0 + 1e-12)


def test_grid_cs_drops_offgrid_rows_and_clips_snr():
    c = synthetic_catalogue()
    g = SEL.grid_cs(c["nhi"], c["snr"])
    assert g.shape == (29, 8)
    # rows 7 (N-hat 19.4) and 8 (22.5) must be OFF the grid
    assert g.sum() == int(((c["nhi"] >= 19.5) & (c["nhi"] < 22.4)).sum())
    # the dead stratum row lands in s = 0, it is not silently discarded
    assert g[:, 0].sum() == 1


def test_grid_cs_matches_the_committed_index_convention():
    ep = pytest.importorskip("CDDF_analysis.hbi_mcmc.extract_pack",
                             reason="needs the data-plane stack")
    rng = np.random.default_rng(7)
    nhat = rng.uniform(19.0, 22.8, 5000)
    snr = rng.uniform(0.5, 12.0, 5000)
    zobs = rng.uniform(2.0, 3.5 - 1e-9, 5000)
    mine = SEL.grid_cs(nhat, snr)
    theirs = ep.bin_counts_cks(nhat, zobs, snr)[0].sum(axis=1)
    assert np.array_equal(mine, theirs)


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------
def test_classify_host_association_each_mechanism():
    nhi_true = np.array([np.nan, np.nan, np.nan, 18.0, 19.2, 19.6, 20.5, 21.9])
    n_pool = np.array([0, 0, 2, 1, 1, 1, 1, 1])
    n_raw = np.array([0, 3, 2, 1, 1, 1, 1, 1])
    got = SEL.classify_host_association(nhi_true, n_pool, n_raw)
    assert list(got) == ["hostless_no_absorber", "hostless_unresolvable",
                         "hostless_taken", "host_17p2_19p0", "host_19p0_19p5",
                         "host_19p5_19p7", "host_19p7_21p6", "host_ge_21p6"]


def test_classify_host_association_is_a_partition():
    rng = np.random.default_rng(3)
    n = 4000
    host = rng.random(n) < 0.7
    nhi_true = np.where(host, rng.uniform(17.2, 22.2, n), np.nan)
    n_pool = np.where(host, rng.integers(1, 3, n), rng.integers(0, 3, n))
    n_raw = n_pool + rng.integers(0, 2, n)
    lab = SEL.classify_host_association(nhi_true, n_pool, n_raw)
    counts = {c: int((lab == c).sum()) for c in SEL.HOST_CLASSES}
    assert sum(counts.values()) == n
    assert set(np.unique(lab)) <= set(SEL.HOST_CLASSES)


def test_classify_rejects_inconsistent_pools_and_impossible_matches():
    with pytest.raises(ValueError):
        SEL.classify_host_association([np.nan], [2], [1])      # raw < pool
    with pytest.raises(ValueError):
        SEL.classify_host_association([20.0], [0], [0])        # matched, no cand


def test_class_grids_partition_the_total_and_fail_closed():
    rng = np.random.default_rng(11)
    n = 3000
    nhat = rng.uniform(19.5, 22.4, n)
    snr = rng.uniform(2.1, 9.0, n)
    lab = rng.choice(SEL.HOST_CLASSES, n)

    def binner(a, b):
        return SEL.grid_cs(a, b)

    g = SEL.class_grids(lab, binner, nhat, snr)
    tot = g.pop("_total")
    assert np.array_equal(sum(g.values()), tot)

    bad = lab.copy()
    bad[0] = "not_a_class"
    with pytest.raises(AssertionError):
        SEL.class_grids(bad, binner, nhat, snr)


# ---------------------------------------------------------------------------
# the template
# ---------------------------------------------------------------------------
def test_perks_log_share_matches_the_committed_implementation():
    fl = pytest.importorskip("CDDF_analysis.hbi_mcmc.fp_ladder",
                             reason="needs jax/numpyro")
    rng = np.random.default_rng(5)
    fpc = rng.poisson(0.5, size=(29, 8))
    live = np.array([False, False, True, True, True, True, True, True])
    fpc[:, ~live] = 0
    a = SEL.perks_log_share(fpc, live)
    b = fl.perks_log_share(fpc, live)
    assert np.allclose(a, b, rtol=0, atol=0)


def test_perks_shares_sum_to_one_on_live_cells_and_vanish_off_live():
    fpc = np.zeros((29, 8), int)
    fpc[0, 3] = 89
    live = np.array([False, False, True, True, True, True, True, True])
    m = SEL.perks_log_share(fpc, live)
    pi = np.where(live[None, :], np.exp(m), 0.0)
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert pi[:, ~live].sum() == 0.0


def test_template_mu_cs_total_and_empty_cell_mass():
    fpc = np.zeros((29, 8), int)
    fpc[0, 3] = 89
    live = np.array([False, False, True, True, True, True, True, True])
    mu = SEL.template_mu_cs(fpc, live, lam_total=6.5, fp_w_ell_eff=2255.0,
                            eta_c=np.zeros(29))
    assert mu.sum() == pytest.approx(2255.0 * 6.5, rel=1e-12)
    # EVERY empty live cell carries exactly the Perks pseudo-count share
    K = 29 * 6
    a0 = 1.0 / K
    assert mu[5, 4] == pytest.approx(2255.0 * 6.5 * a0 / (89.0 + K * a0),
                                     rel=1e-12)


# ---------------------------------------------------------------------------
# PRODUCT layer
# ---------------------------------------------------------------------------
prod_missing = not os.path.isdir(PRODUCTS)
pytestmark_products = pytest.mark.skipif(
    prod_missing, reason=f"products not built at {PRODUCTS}")


@pytestmark_products
def test_loa0_product_reproduces_the_pack_fp_counts_block():
    from support_contract import read_stamp
    L = np.load(os.path.join(PRODUCTS, "loa0_fp_target.npz"), allow_pickle=True)
    pack = np.load(os.path.join(A0_SUPPORT_DIR, "scanpack_2lpt0_b300_A0.npz"),
                   allow_pickle=True)
    assert np.array_equal(np.asarray(L["fp_counts_89"]),
                          np.asarray(pack["fp_counts"]))
    assert int(np.asarray(L["fp_counts_89"]).sum()) == 89
    assert read_stamp(os.path.join(PRODUCTS, "loa0_fp_target.npz")) is not None


@pytestmark_products
def test_the_2378_product_restricted_to_the_fp_counts_support_is_the_89():
    L = np.load(os.path.join(PRODUCTS, "loa0_fp_target.npz"), allow_pickle=True)
    nhi = np.asarray(L["nhi_2378"], float)
    z = np.asarray(L["z_2378"], float)
    snr = np.asarray(L["snr_2378"], float)
    m = (nhi >= 19.5) & (nhi < 22.4) & (z >= 2.0) & (z < 3.5)
    assert int(m.sum()) == int(np.asarray(L["fp_counts_89"]).sum())
    assert np.array_equal(SEL.grid_cs(nhi[m], snr[m]),
                          np.asarray(L["fp_counts_89"]))


@pytestmark_products
@pytest.mark.parametrize("fam", FAMILIES)
def test_decomposition_is_a_partition_of_the_census(fam):
    d = np.load(os.path.join(PRODUCTS, f"fp_target_decomposition_{fam}.npz"),
                allow_pickle=True)
    parts = [np.asarray(d[f"cks_{c}"]) for c in SEL.HOST_CLASSES]
    total = np.asarray(d["cks_total"])
    assert np.array_equal(sum(parts), total)
    hostless = (np.asarray(d["cks_hostless_no_absorber"])
                + np.asarray(d["cks_hostless_taken"])
                + np.asarray(d["cks_hostless_unresolvable"]))
    assert np.array_equal(hostless, np.asarray(d["cks_hostless"]))
    # and the (C,S) marginals must be the k-sum of the (C,K,S) planes
    for c in SEL.HOST_CLASSES + ("total", "hostless"):
        assert np.array_equal(np.asarray(d[f"cs_{c}"]),
                              np.asarray(d[f"cks_{c}"]).sum(axis=1))


@pytestmark_products
@pytest.mark.parametrize("fam", FAMILIES)
def test_decomposition_equals_the_A0v2_census_block_for_block(fam):
    d = np.load(os.path.join(PRODUCTS, f"fp_target_decomposition_{fam}.npz"),
                allow_pickle=True)
    cen = np.load(os.path.join(A0_SUPPORT_DIR, f"fp_census_{fam}_A0v2.npz"),
                  allow_pickle=True)
    for blk in ("counts_all", "hostless", "host_17p2_19p0", "host_19p0_19p5",
                "host_19p5_19p7", "host_19p7_21p6", "host_ge_21p6"):
        key = "cks_total" if blk == "counts_all" else f"cks_{blk}"
        assert np.array_equal(np.asarray(d[key]), np.asarray(cen[blk])), blk


@pytestmark_products
@pytest.mark.parametrize("fam", FAMILIES)
def test_every_mock_product_carries_the_A0v2_census_support_id(fam):
    from support_contract import assert_same_support, read_stamp
    p = os.path.join(PRODUCTS, f"fp_target_decomposition_{fam}.npz")
    cen = os.path.join(A0_SUPPORT_DIR, f"fp_census_{fam}_A0v2.npz")
    mine, theirs = read_stamp(p), read_stamp(cen)
    assert mine is not None and theirs is not None
    assert mine.sha256 == theirs.sha256
    assert_same_support({"fp_target_review": mine, "A0v2_census": theirs})
    assert str(np.load(p, allow_pickle=True)["support_id"]) == theirs.sha256


@pytestmark_products
def test_a_support_mismatch_fails_closed():
    from support_contract import (SupportMismatch, assert_same_support,
                                  read_stamp)
    a = read_stamp(os.path.join(PRODUCTS, "fp_target_decomposition_2lpt0.npz"))
    b = read_stamp(os.path.join(PRODUCTS, "loa0_fp_target.npz"))
    with pytest.raises(SupportMismatch):
        assert_same_support({"mock": a, "loa0_twin": b})


@pytestmark_products
def test_tables_report_a_partition_and_the_verdict_inputs():
    t = os.path.join(PRODUCTS, "FP_TARGET_REVIEW_TABLES.json")
    if not os.path.exists(t):
        pytest.skip("tables not built")
    T = json.load(open(t))
    for fam in FAMILIES:
        tot = T["families"][fam]["totals"]
        parts = sum(tot[c] for c in SEL.HOST_CLASSES)
        assert parts == tot["total"]
        assert (tot["hostless_no_absorber"] + tot["hostless_taken"]
                + tot["hostless_unresolvable"]) == tot["hostless"]
        assert T["families"][fam]["support_equals_A0v2_census"] is True
    assert T["product_89_vs_2378"]["identical_on_the_fp_counts_support"] is True
