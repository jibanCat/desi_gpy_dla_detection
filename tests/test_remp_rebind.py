"""test_remp_rebind.py — TDD GATE for the WALL-1 full-injection frozen-kernel re-bind.

The full-injection WALL-1 test (notes/2026-06-17_wall1_full_injection_design.md §5.3)
needs to apply the UNTILTED-calibrated R_emp response matrix R[s,jhat,jtru] to a
GENUINELY RE-INFERRED (tilted) catalog. The load-bearing new mechanism is
``assign_R_emp_to_catalog``: it bins each detection of an ARBITRARY catalog into its
(SNR-cell, x_hat-bin) and reads the frozen untilted response — exactly "freeze the
operator calibrated at the untilted slope, apply it to a re-inferred tilted population."

THE GATE (design §7, "that TDD check gates everything"):
  Re-binding the untilted R response onto the UNTILTED loa-124 catalog must reproduce
  the cached untilted R_emp kappa BYTE-FOR-BYTE. If it cannot, the test is invalid.

Two byte-for-byte assertions:
  1. refactor-equivalence: assign_R_emp_to_catalog(compute_R_response(...), <same cat>)
     == the original build_R_emp(...) kappa. (The re-bind IS the original build path
     decomposed; on the same cat it must be bit-identical — proves zero behaviour drift.)
  2. on-disk regression: the same re-bind reproduces the frozen
     <r_emp>/posterior_kernel_2lpt0.npz `kappa` already cached on scratch
     (the artifact the WALL-1 R_emp path consumes), to bit precision.

These run on the real loa-124 catalog + full-forest molly (the EXACT inputs the cached
r_emp build used: run_remp_kernel.py main() defaults — figures_molly, fit-floor 19.5,
lam_rf_min 911.0, smooth_bins 1.0, n_floor 20, host_col NHI_TILT_HOST). They are
skipped automatically when those large scratch inputs are absent (e.g. CI without the
GL filesystem), so the suite stays green off-cluster.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- the EXACT inputs the cached r_emp/posterior_kernel_2lpt0.npz was built from -----
# (run_remp_kernel.py main() defaults: full-forest figures_molly matrix, loa-124 truth,
#  the 2lpt-0 V1 combined catalog, fit-floor 19.5, lam_rf_min 911.0).
CAT_DIR = ("/scratch/cavestru_root/cavestru0/mfho/"
           "gl_prod_2lpt0_v1_20260526/combined_catalog/")
MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/"
         "gl_prod_2lpt0_v1_20260526/figures_molly/molly_matrix.tsv")
TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
         "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
       "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits")
CACHED_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
              "phase3d_experiments/r_emp/posterior_kernel_2lpt0.npz")

_INPUTS = [CAT_DIR, MOLLY, TRUTH, BAL]
_HAVE_INPUTS = all(os.path.exists(p) for p in _INPUTS)

pytestmark = pytest.mark.skipif(
    not _HAVE_INPUTS,
    reason=("WALL-1 R_emp re-bind gate needs the GL scratch inputs "
            "(loa-124 truth + 2lpt-0 V1 catalog + full-forest molly); "
            "skipped off-cluster."),
)


@pytest.fixture(scope="module")
def remp_ingredients():
    """Build the R_emp ingredients ONCE (the EXACT cached-build config)."""
    from CDDF_analysis.cddf_catalog_hbi import (
        HBIConfig, load_molly_matrix, load_and_cut_catalog, build_fine_grid,
        _build_qso_lookup, _fine_z_grid,
    )

    cfg = HBIConfig(
        catalog_dir=CAT_DIR, truth_path=TRUTH, bal_cat_path=BAL,
        molly_tsv=MOLLY, out_dir="/tmp/test_remp_rebind",
        mockdir=os.path.dirname(TRUTH),
        zbins=(2.0, 2.5, 3.0, 3.5),
        fp_estimator="purity_mixture", no_bal=True,
        v3_family="bspbody", v3_logN_fit_floor=19.5,
        lam_rf_min=911.0,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.1,
    )
    mm = load_molly_matrix(MOLLY)
    truth_floor = float(mm.nhi_edges[0])
    qso_lookup = _build_qso_lookup(cfg)
    # host_truth_floor=19.0 (run_remp_kernel default) so NHI_TILT_HOST exists
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(19.0, truth_floor))
    fine = build_fine_grid(cfg)
    return dict(cfg=cfg, mm=mm, cat_cut=cat_cut, good_mask=good_mask, fine=fine)


def test_rebind_reproduces_build_R_emp_byte_for_byte(remp_ingredients):
    """assign_R_emp_to_catalog(compute_R_response(...), <same cat>) == build_R_emp(...)."""
    from CDDF_analysis.run_remp_kernel import (
        build_R_emp, compute_R_response, assign_R_emp_to_catalog,
    )

    g = remp_ingredients
    # The original, full build path (the frozen reference).
    kappa_ref, ess_ref, info_ref = build_R_emp(
        g["cfg"], g["cat_cut"], g["good_mask"], g["fine"], g["mm"],
        smooth_bins=1.0, n_floor=20, host_col="NHI_TILT_HOST", verbose=False)

    # The decomposed path: measure the response, then re-bind it to the SAME cat.
    R_response = compute_R_response(
        g["cfg"], g["cat_cut"], g["good_mask"], g["fine"], g["mm"],
        smooth_bins=1.0, n_floor=20, host_col="NHI_TILT_HOST")
    kappa_rb, ess_rb, info_rb = assign_R_emp_to_catalog(
        R_response, g["cfg"], g["cat_cut"], g["good_mask"], g["fine"], g["mm"])

    assert kappa_rb.shape == kappa_ref.shape
    assert kappa_rb.dtype == kappa_ref.dtype
    # BYTE-FOR-BYTE: the re-bind on the same cat is the original build, decomposed.
    assert np.array_equal(kappa_rb, kappa_ref), (
        "R_emp re-bind does NOT reproduce build_R_emp byte-for-byte on the untilted "
        "cat — the frozen-kernel mechanism is broken; the full-injection test is invalid.")
    for tier in (20.3, 20.6, 21.0):
        assert np.array_equal(ess_rb[tier], ess_ref[tier]), f"ESS tier {tier} differs"


def test_rebind_reproduces_cached_npz_byte_for_byte(remp_ingredients):
    """The re-bind reproduces the on-disk frozen r_emp kappa to bit precision."""
    if not os.path.exists(CACHED_NPZ):
        pytest.skip(f"cached r_emp npz absent: {CACHED_NPZ}")
    from CDDF_analysis.run_remp_kernel import (
        compute_R_response, assign_R_emp_to_catalog,
    )

    g = remp_ingredients
    cached = np.load(CACHED_NPZ, allow_pickle=True)
    kappa_cached = cached["kappa"].astype(np.float32)

    R_response = compute_R_response(
        g["cfg"], g["cat_cut"], g["good_mask"], g["fine"], g["mm"],
        smooth_bins=float(cached["smooth_bins"]), n_floor=int(cached["n_floor"]),
        host_col=str(cached["host_col"]))
    kappa_rb, _, _ = assign_R_emp_to_catalog(
        R_response, g["cfg"], g["cat_cut"], g["good_mask"], g["fine"], g["mm"])

    assert kappa_rb.shape == kappa_cached.shape, (
        f"shape {kappa_rb.shape} != cached {kappa_cached.shape}")
    assert np.array_equal(kappa_rb, kappa_cached), (
        "R_emp re-bind does NOT reproduce the cached frozen kappa byte-for-byte.")
