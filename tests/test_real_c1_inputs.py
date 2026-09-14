"""test_real_c1_inputs.py — gates on the PREPARED blind real-C1 inputs.

CLASSIFICATION: VALIDATION ONLY. No test here evaluates a likelihood, runs a
sampler, or asserts anything about a real-data RESULT. Real-data arrays are
touched only for shape / support / finiteness / mask structure, never for a
value that could be a science number.

Run:
  PYTHONPATH=/home/mfho/wt_abs_diag_2026-09 python -m pytest -q \
      tests/test_real_c1_inputs.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "validation", "absorber_ladder", "support"))
sys.path.insert(0, os.path.join(_REPO, "validation", "real_c1"))

L = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
INPUTS = os.path.join(L, "real_c1_inputs")
CAL = "2lpt0"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(INPUTS),
    reason="prepared real-C1 inputs absent (scratch not staged)")


@pytest.fixture(scope="module")
def real_pack():
    return np.load(os.path.join(INPUTS, "C1_pack.npz"), allow_pickle=True)


@pytest.fixture(scope="module")
def mg_real():
    return np.load(os.path.join(INPUTS, "Mg_B_real.npz"), allow_pickle=True)


@pytest.fixture(scope="module")
def mg_cal():
    return np.load(os.path.join(L, "response_review", "candidates",
                                f"Mg_B_{CAL}.npz"), allow_pickle=True)


# ---------------------------------------------------------------------------
# 1-2. Mg_B_real: the row-sum identity, through the REAL pack's kz map
# ---------------------------------------------------------------------------
def test_mg_real_row_sums_equal_phi_2lpt_through_the_real_kz_map(real_pack,
                                                                 mg_real):
    Mg = np.asarray(mg_real["Mg"], float)              # (S, Kf, C, B)
    phi = np.asarray(mg_real["phi_bsK"], float)        # (B, S, KK)
    kz = np.asarray(real_pack["kz_to_K"], int)         # the REAL pack's map
    phi_k = np.einsum("bsK->sKb", phi)[:, kz, :]       # (S, Kf, B)
    assert Mg.shape[:2] == phi_k.shape[:2]
    assert np.allclose(Mg.sum(axis=2), phi_k, rtol=0, atol=1e-12)


def test_mg_real_is_rows_unit_times_phi_and_rows_are_unit(real_pack, mg_real):
    ru = np.asarray(mg_real["rows_unit"], float)       # (B, S, KK, C)
    phi = np.asarray(mg_real["phi_bsK"], float)
    kz = np.asarray(real_pack["kz_to_K"], int)
    assert np.allclose(ru.sum(axis=3), 1.0, rtol=0, atol=1e-12)
    rebuilt = np.einsum("bsKc,bsK->sKcb", ru, phi)[:, kz, :, :]
    assert np.array_equal(rebuilt, np.asarray(mg_real["Mg"], float))


def test_mg_real_equals_the_frozen_calibration_tensor(mg_real, mg_cal):
    """The real pack's fine-z grid is the mocks': the tensor must be EXACT,
    not merely close. A drift here means the grids diverged."""
    assert np.array_equal(np.asarray(mg_real["Mg"], float),
                          np.asarray(mg_cal["Mg"], float))


# ---------------------------------------------------------------------------
# 3. strata / edges equality — the precondition for reusing the frozen tables
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["nhat_edges", "ntrue_edges", "zf_edges",
                                 "zc_edges", "snr_edges", "kz_to_K",
                                 "nhat_masked_bins"])
def test_real_pack_grid_is_identical_to_every_mock_v3_pack(real_pack, key):
    for fam in ("2lpt0", "london0", "saclay0"):
        p = os.path.join(L, "support_v3", f"scanpack_{fam}_b300_v3.npz")
        if not os.path.exists(p):
            pytest.skip(f"mock v3 pack absent: {fam}")
        zm = np.load(p, allow_pickle=True)
        assert np.array_equal(np.asarray(real_pack[key]),
                              np.asarray(zm[key])), f"{key} differs vs {fam}"


def test_c_table_strata_match_and_table_is_copied_unchanged(real_pack):
    zr = np.load(os.path.join(INPUTS, "C_C1nsadd_real.npz"), allow_pickle=True)
    zc = np.load(os.path.join(L, "completeness", f"C_C1nsadd_{CAL}.npz"),
                 allow_pickle=True)
    for k in ("ntrue_edges", "snr_edges", "kz_to_K"):
        assert np.array_equal(np.asarray(zr[k]), np.asarray(real_pack[k]))
    for k in ("C_fixed", "C_fixed_sd", "live_strata_mask", "b_to_cell"):
        assert np.array_equal(np.asarray(zr[k]), np.asarray(zc[k])), k
    C = np.asarray(zr["C_fixed"], float)
    assert C.shape == (len(np.asarray(real_pack["snr_edges"])) - 1,
                       len(np.asarray(real_pack["ntrue_edges"])) - 1)
    assert np.all(np.isfinite(C)) and C.min() >= 0.0 and C.max() <= 1.0


# ---------------------------------------------------------------------------
# 4. mu_extra: shape, units and the transport identity
# ---------------------------------------------------------------------------
def test_mu_extra_shape_units_and_transport_identity(real_pack):
    z = np.load(os.path.join(INPUTS, "mu_extra_P6bcal_real.npz"),
                allow_pickle=True)
    mu = np.asarray(z["mu_extra"], float)
    rate = np.asarray(z["rate_cKs"], float)
    dX = np.asarray(real_pack["dX"], float)
    kz = np.asarray(real_pack["kz_to_K"], int)
    assert mu.shape == tuple(np.asarray(real_pack["counts"]).shape)
    # units: rate is per unit dX, so mu = rate * dX reproduces it exactly
    assert np.array_equal(mu, rate[:, kz, :] * dX[None, :, :])
    assert np.all(np.isfinite(mu)) and mu.min() >= 0.0
    dead = ~(dX.sum(axis=0) > 0)
    assert mu[:, :, dead].sum() == 0.0
    # the rate itself is the 2LPT-0 calibration, NOT refitted on real data
    prov = json.loads(str(z["provenance"]))
    assert prov["fitted_to_real_data"] is False
    assert prov["calibration_family"] == CAL
    zc = np.load(os.path.join(L, "completeness", f"P6b_rate_{CAL}.npz"),
                 allow_pickle=True)
    assert np.array_equal(rate, np.asarray(zc["rate_cKs"], float))


def test_mu_extra_is_the_key_the_runner_reads_first():
    z = np.load(os.path.join(INPUTS, "mu_extra_P6bcal_real.npz"),
                allow_pickle=True)
    assert "mu_extra" in z.files


# ---------------------------------------------------------------------------
# 5. support_id recomputation + the mutation control
# ---------------------------------------------------------------------------
def test_support_id_recomputes_from_the_declared_fields():
    import support_contract as SC
    st = json.load(open(os.path.join(INPUTS, "C1_pack.support.json")))
    sid = SC.support_from_json(st)          # re-derives, does not trust
    assert sid.sha256 == st["support_id"]
    assert sid.row_sha256 == st["row_support_id"]
    # every one of the 12 fields is declared (no defaulting)
    assert set(st["fields"]) == set(SC.SUPPORT_FIELDS)


def test_support_gate_passes_on_the_staged_real_pack():
    import support_contract as SC
    row = tuple(f for f in SC.SUPPORT_FIELDS if f != "truth_host_floor")
    rec = SC.check_support_consistency(os.path.join(INPUTS, "C1_pack.npz"),
                                       None, None, fields=row)
    assert rec["status"] == "PASS"
    assert set(rec["planes"]) == {"pack.counts", "pack.dX", "pack.fp_E_alloc"}


def test_a_flipped_collar_is_refused_by_the_support_id():
    """MUTATION CONTROL: the support must be sensitive to the collar. A
    collar-3000 declaration may not collide with the collar-3300 one."""
    import support_contract as SC
    from audit_real_pack import real_support
    cache: dict = {}
    good = real_support(collar_kms=3300.0, cache=cache)
    bad = real_support(collar_kms=3000.0, cache=cache)
    assert good.sha256 != bad.sha256
    assert good.row_sha256 != bad.row_sha256
    st = json.load(open(os.path.join(INPUTS, "C1_pack.support.json")))
    assert st["support_id"] == good.sha256
    assert st["support_id"] != bad.sha256
    with pytest.raises(SC.SupportMismatch):
        SC.assert_same_support({"counts@3300": good, "dX@3000": bad})


def test_collar_mutation_also_changes_the_counts_selection():
    """The mutation is not merely a label: the same flip moves real rows."""
    from audit_real_pack import counts_ladder
    _, c3300, _ = counts_ladder(3300.0)
    _, c3000, _ = counts_ladder(3000.0)
    assert not np.array_equal(c3300, c3000)


# ---------------------------------------------------------------------------
# 6. the fp_counts calibration block + the blind-run invariants
# ---------------------------------------------------------------------------
def test_fp_counts_block_is_the_same_loa0_object_as_the_mocks(real_pack):
    fpc = np.asarray(real_pack["fp_counts"])
    assert int(fpc.sum()) == 89
    for fam in ("2lpt0", "london0", "saclay0"):
        p = os.path.join(L, "support_v3", f"fp_counts_{fam}_v3.npz")
        if not os.path.exists(p):
            pytest.skip(f"standalone fp_counts absent: {fam}")
        assert np.array_equal(np.asarray(np.load(p, allow_pickle=True)
                                         ["fp_counts"]), fpc)


def test_real_mode_gate_and_truth_sentinel(real_pack):
    prov = json.load(open(os.path.join(INPUTS, "C1_pack.provenance.json")))
    assert prov["real_data"] is True
    assert prov["truth_counts_sentinel"] == "ZEROS_NO_TRUTH"
    tc = np.asarray(real_pack["truth_counts"])
    assert tc.size > 0 and not np.any(tc != 0)


def test_fp_E_alloc_is_dX_renormalised_so_the_planes_share_one_collar(real_pack):
    dX = np.asarray(real_pack["dX"], float)
    fpE = np.asarray(real_pack["fp_E_alloc"], float)
    col = dX.sum(axis=0)
    exp = np.zeros_like(dX)
    nz = col > 0
    exp[:, nz] = dX[:, nz] / col[nz]
    assert np.array_equal(fpE, exp)


def test_dry_check_recorded_a_pass_and_never_observed_the_counts_site():
    p = os.path.join(INPUTS, "DRY_CHECK.json")
    if not os.path.exists(p):
        pytest.skip("dry check not run")
    rec = json.load(open(p))
    assert rec["VERDICT"] == "PASS"
    assert rec["trace"]["counts_site_is_observed"] is False
    assert rec["likelihood_evaluated"] is False and rec["mcmc_run"] is False


def test_the_sbatch_is_prepared_but_refuses_to_run_without_the_runner():
    p = os.path.join(_REPO, "validation", "real_c1", "run_real_c1.sbatch")
    txt = open(p).read()
    assert "DO NOT SUBMIT" in txt
    assert "--ladder M1CUT" in txt
    assert "--require-support" in txt
    assert "--lam-imputations $J" in txt and "J=8" in txt
    assert "20260811" in txt and "20260812" in txt
    assert "--fp-a0" not in txt.split("# NOTE ON --fp-a0")[0]
    assert "run_real_c1.py" in txt
