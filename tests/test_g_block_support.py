"""g(N,z) support consistency (finding N1, 2026-08-20; PI ruling 2026-08-21
item 1-2: source calibration DEFECT, fixed at source).

The fold's truth support (``build_truth_counts``) and g's TP numerator are
both S2N_RED > snr_min. g's truth DENOMINATOR must live on the SAME support;
a denominator carrying the SNR <= snr_min systems is a numerator/denominator
support mismatch (the project's recurring bug class) that the per-row
normalisation turns into a spurious z-tilt. The decisive check is a counting
argument: g_occupancy must total the SAME systems as the fold's truth support.

MOCK-SYNTHETIC ONLY. Loads extract_pack.py file-directly (jax-free gpdla env).
"""
import importlib.util as _ilu
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_spec = _ilu.spec_from_file_location(
    "g_support_extract_pack",
    os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "extract_pack.py"))
EP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(EP)

from CDDF_analysis.hbi import cddf_catalog_hbi as H  # noqa: E402


def _cfg():
    return H.HBIConfig(
        catalog_dir="/dev/null", truth_path="/dev/null",
        bal_cat_path="/dev/null", molly_tsv="/dev/null", out_dir="/tmp",
        logN_lo=17.2, logN_hi=22.5, dlogN=0.1, drop_top_bin_above=22.4,
        zbins=(2.0, 2.5, 3.0, 3.5), report_logN_limits=(20.0, 20.3),
        H0=70.0, Omega_m=0.279, snr_min=2.0, p_dla_min=0.5,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.1,
        completeness_z_resolved=True)


def _bundle(seed=3, n=20000, frac_low_snr=0.5):
    """Synthetic truth-match: truth systems in one molly cell [20.3, 20.5),
    z in [2.0, 3.5); a fraction sits on LOW-SNR sightlines (S2N_RED = 1.0,
    below snr_min = 2) where NOTHING is detected; detections (TPs) come only
    from the SNR > snr_min sightlines, with detection probability rising in z
    (the 2LPT-0 signature) so the low-SNR fraction is z-dependent too."""
    from astropy.table import Table
    rng = np.random.default_rng(seed)
    t_nhi = rng.uniform(20.3, 20.5, n)
    t_z = rng.uniform(2.0, 3.5, n)
    # low-SNR share falls with z (as on the real mocks: SNR<=2 truth is
    # richer at low z_abs) -> the excess is z-dependent
    p_low = frac_low_snr * (1.3 - 0.4 * (t_z - 2.0) / 1.5)
    low = rng.random(n) < p_low
    s2n = np.where(low, 1.0, 5.0)
    # boundary rows: the production cut is STRICT (S2N_RED > 2, "strict" in
    # the op_mask contract) — rows exactly AT snr_min belong to the excluded
    # side on both supports
    at_edge = rng.random(n) < 0.05
    s2n = np.where(at_edge, 2.0, s2n)
    low = low | at_edge
    p_det = np.where(low, 0.0, 0.35 + 0.30 * (t_z - 2.0) / 1.5)
    det = rng.random(n) < p_det
    nd = int(det.sum())
    cat_cut = Table(dict(
        S2N_RED=s2n[det], P_DLA=np.full(nd, 0.9),
        NHI=t_nhi[det], NHI_TRUE=t_nhi[det],
        Z_DLA=t_z[det], Z_QSO=t_z[det] + 0.3))
    truth_cut = Table(dict(NHI=t_nhi, Z_DLA=t_z, S2N_RED=s2n))
    mm = H.MollyMatrix(snr_edges=np.array([0.0, np.inf]),
                       nhi_edges=np.array([20.3, 20.5, np.inf]),
                       purity=np.ones((1, 2)), completeness=np.full((1, 2), 0.5))
    return dict(cfg=_cfg(), mm=mm, cat_cut=cat_cut, truth_cut=truth_cut,
                good_mask=np.ones(nd, bool)), int((~low).sum()), n


def test_build_g_block_denominator_counts_only_the_fold_truth_support():
    """COUNTING IDENTITY (the decisive check for this bug class): g's
    occupancy must total exactly the systems build_truth_counts keeps —
    S2N_RED > snr_min — not every truth system in the z range."""
    bundle, n_kept, n_all = _bundle()
    g_grid, g_occ = EP.build_g_block(bundle)
    _, _, n_truth_support = EP.build_truth_counts(bundle)
    assert n_truth_support == n_kept            # the fold's support, by construction
    assert int(g_occ.sum()) == n_kept, (
        f"g denominator counts {int(g_occ.sum())} systems; the fold's truth "
        f"support has {n_kept} (all-SNR truth = {n_all}): support mismatch")
    assert int(g_occ.sum()) == n_truth_support


def test_build_g_block_per_z_occupancy_matches_truth_counts_per_z():
    """Per fine-z cell, not just in total: the excess is z-dependent, which is
    exactly what made it a tilt. truth_counts is (b, k) on the schema grid;
    sum over b gives the per-k support, which g_occupancy (rows x k) must
    reproduce cell by cell."""
    bundle, _, _ = _bundle(seed=5)
    _, g_occ = EP.build_g_block(bundle)
    tc_bk, _, _ = EP.build_truth_counts(bundle)
    np.testing.assert_array_equal(g_occ.sum(axis=0), tc_bk.sum(axis=0))


def test_g_truth_support_reports_total_and_kept_for_provenance():
    """The pack provenance must carry the support change (n_total -> n_kept)
    so the certification can check it without reloading the mock bundle."""
    bundle, n_kept, n_all = _bundle(seed=7)
    t_cons, n_total, n_keep = EP.g_truth_support(bundle)
    assert (n_total, n_keep) == (n_all, n_kept)
    assert len(t_cons) == n_kept
    assert np.all(np.asarray(t_cons["S2N_RED"], float) > bundle["cfg"].snr_min)
    # strictness: rows exactly at snr_min are on the EXCLUDED side
    assert np.any(np.asarray(bundle["truth_cut"]["S2N_RED"], float) == bundle["cfg"].snr_min)
