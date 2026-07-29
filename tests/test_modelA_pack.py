"""Tests for the Model A data-pack extractor (Queue 3 §1 data plane).

Contract: scratchpad modelA_pack_schema.md (BINDING) + notes/2026-07-11_q3_modelA_spec.md §1.

Two tiers:
  * pure tests (always run): schema constants, binning helpers, t_sigma derivation
    from the committed ff_fp_*.json artifacts, forward-model NPZ guards.
  * pack tests (run per generated pack; SKIP if the pack dir is absent): schema
    conformance (shapes/edges/finiteness/axis order), counts total == op-mask
    detection total, dX marginals vs the committed X_tot per coarse z, fp_E_alloc
    normalization, t_sigma floor, provenance sidecar, round-trip load.

MOCKS ONLY. No real-LOA path is ever touched here.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# INTERFACE NOTE: CDDF_analysis/hbi_mcmc/__init__.py imports jax at package-import
# time, but the extractor is a pure data-plane module (no jax) and runs in the
# `gpdla` env (no jax there — jax lives in `gpdla-hbi`). Load it file-directly so
# the package __init__ (and jax) is never triggered.
import importlib.util as _ilu  # noqa: E402

_EP_PATH = os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "extract_pack.py")
_spec = _ilu.spec_from_file_location("modelA_extract_pack", _EP_PATH)
EP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(EP)

PACK_DIR = os.environ.get(
    "MODELA_PACK_DIR",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs")

# dX marginal consistency: build_M_b PX vs build_pathlength X_tot are the SAME
# sightline set + the SAME analytic AbsorptionDistance.deltaX, partitioned on the
# fine grid, so the gap should be at float-accumulation level. The schema test
# tolerance is 1e-3; the MEASURED gap is pinned (and reported) per pack in
# provenance["dx_marginal_max_relgap"] — if a future rebuild pushes it above the
# pinned ceiling below, that is a real drift, not noise.
DX_RTOL_SCHEMA = 1e-3
DX_RELGAP_PIN = 1e-9   # measured 2026-07-11: <=3e-16 on all three mocks (see report)


def _packs():
    if not os.path.isdir(PACK_DIR):
        return []
    return sorted(glob.glob(os.path.join(PACK_DIR, "modelA_pack_*.npz")))


def _load(path):
    d = np.load(path, allow_pickle=False)
    prov_path = path[:-4] + ".provenance.json"
    with open(prov_path) as f:
        prov = json.load(f)
    return d, prov


packs = _packs()
needs_pack = pytest.mark.skipif(
    not packs, reason=f"no generated packs under {PACK_DIR}")


# ---------------------------------------------------------------------------
# pure tests — schema constants + helpers (no data needed)
# ---------------------------------------------------------------------------
def test_schema_grid_constants():
    assert len(EP.NHAT_EDGES) == 30 and EP.N_C == 29
    assert np.isclose(EP.NHAT_EDGES[0], 19.5) and np.isclose(EP.NHAT_EDGES[-1], 22.4)
    assert np.allclose(np.diff(EP.NHAT_EDGES), 0.1)
    assert len(EP.ZF_EDGES) == 16 and EP.N_K == 15
    assert np.isclose(EP.ZF_EDGES[0], 2.0) and np.isclose(EP.ZF_EDGES[-1], 3.5)
    assert np.allclose(EP.ZC_EDGES, [2.0, 2.5, 3.0, 3.5]) and EP.N_KC == 3
    assert len(EP.SNR_EDGES) == 9 and EP.N_S == 8
    assert EP.SNR_EDGES[0] == 0.0 and np.isinf(EP.SNR_EDGES[-1])
    # kz -> K map: 5 fine bins per coarse bin
    assert EP.KZ_TO_K.shape == (15,)
    assert np.array_equal(EP.KZ_TO_K, np.repeat([0, 1, 2], 5))


def test_bin_counts_halfopen_edges():
    # exactly-on-edge values follow the half-open [lo, hi) convention
    nhat = np.array([19.5, 22.4, 22.39, 19.49, 20.0])
    zhat = np.array([2.0, 2.5, 3.4999, 1.99, 3.5])
    snr = np.array([2.5, 3.0, 8.0, 5.0, 7.0])
    counts, n_in = EP.bin_counts_cks(nhat, zhat, snr)
    assert counts.dtype == np.int64
    assert counts.shape == (29, 15, 8)
    # in-window: (19.5, 2.0), (22.39, 3.4999); out: 22.4 (N edge), 19.49, z=3.5
    assert n_in == 2 and counts.sum() == 2
    assert counts[0, 0, 2] == 1          # 19.5 -> c=0; z=2.0 -> k=0; snr 2.5 -> s=2
    assert counts[28, 14, 7] == 1        # 22.39 -> c=28; 3.4999 -> k=14; snr 8 -> s=7


def test_t_sigma_from_committed_artifacts():
    t, detail = EP.compute_t_sigma()
    assert t.shape == (3,)
    assert np.all(t >= 0.10 - 1e-12)            # floor
    assert np.all(np.isfinite(t)) and np.all(t < 1.0)
    # re-derive independently from the committed jsons
    hbi = os.path.join(_REPO, "CDDF_analysis", "hbi")
    R = {}
    for m in ("2lpt0", "saclay0", "london0"):
        with open(os.path.join(hbi, f"ff_fp_{m}.json")) as f:
            d = json.load(f)
        R[m] = [d["strata"][z]["closure"]["subdla_195_203"]["R_point"]
                for z in ("z_2.0_2.5", "z_2.5_3.0", "z_3.0_3.5")]
    for K in range(3):
        expect = max(0.10, max(abs(np.log(R[m][K] / R["2lpt0"][K]))
                               for m in ("saclay0", "london0")))
        assert np.isclose(t[K], expect, rtol=0, atol=1e-12)


def test_forward_model_is_forward_not_kappa():
    fwd, meta = EP.load_forward_response_pack()
    assert "skewnormal" in meta["fwd_response_kind"]
    for key in fwd:
        assert "kappa" not in key.lower()
    assert fwd["resp_mu_coef"].shape == fwd["resp_sig_coef"].shape == \
        fwd["resp_skew_coef"].shape
    s_resp, z_resp, degp1 = fwd["resp_mu_coef"].shape
    assert len(fwd["resp_snr_edges"]) == s_resp + 1
    assert len(fwd["resp_z_edges"]) == z_resp + 1
    assert degp1 == meta["deg_N"] + 1
    assert fwd["resp_sig_floor"] > 0
    assert fwd["resp_skew_ramp"].shape == (2,)
    assert meta["z_covariate"] in ("zqso", "zdla")


def test_resp_N_fit_range_emitted_natively():
    """Schema v1.1 (finding D2): the extractor must emit the CALIBRATED
    covariate range directly, so a padded pack needs no upgrade_pack_v11 hop.
    The range is what makes the sub-floor pad's response a BRACKETED
    extrapolation instead of a silent one."""
    fwd, meta = EP.load_forward_response_pack()
    rr = fwd["resp_N_fit_range"]
    s_resp, z_resp, _ = fwd["resp_mu_coef"].shape
    assert rr.shape == (s_resp, z_resp, 2)
    assert np.all(rr[..., 1] > rr[..., 0]) and np.all(np.isfinite(rr))
    # the whole point: the fitted range does NOT reach the reporting floor, so
    # every padded bin below rr.min() evaluates a CLAMPED covariate.
    assert rr[..., 0].max() > 19.3
    assert rr[..., 1].max() < 22.4


# ---------------------------------------------------------------------------
# D1 — the schema-v1.1 downward BASIS PAD (2026-07-28)
# ---------------------------------------------------------------------------
def test_basis_pad_default_is_schema_v1_bit_for_bit():
    """No --basis-pad-floor => the true-N grid is IDENTICAL to schema v1.
    This is the regression lock on 'the default did not move'."""
    edges, n_pad = EP.basis_pad_edges(None)
    assert n_pad == 0
    assert edges.shape == EP.NHAT_EDGES.shape
    assert np.array_equal(edges, EP.NHAT_EDGES)
    # a floor AT or ABOVE the reporting floor is also a no-op (never shrink)
    for floor in (19.5, 19.6, 20.0):
        e, n = EP.basis_pad_edges(floor)
        assert n == 0 and np.array_equal(e, EP.NHAT_EDGES)


@pytest.mark.parametrize("floor,n_expect", [(19.4, 1), (19.3, 2), (19.0, 5),
                                            (18.5, 10), (18.0, 15)])
def test_basis_pad_edges_extend_down_only(floor, n_expect):
    edges, n_pad = EP.basis_pad_edges(floor)
    assert n_pad == n_expect
    assert len(edges) == len(EP.NHAT_EDGES) + n_expect
    assert np.isclose(edges[0], floor)
    # the OBSERVED window never moves: exact tail subset, same step, same top
    assert np.allclose(edges[n_expect:], EP.NHAT_EDGES)
    assert np.allclose(np.diff(edges), EP.N_STEP)
    assert np.isclose(edges[-1], EP.NHAT_EDGES[-1])


def test_basis_pad_off_grid_floor_refused():
    with pytest.raises(ValueError, match="not on the"):
        EP.basis_pad_edges(19.27)


class _FakeTable(dict):
    pass


class _FakeCfg:
    snr_min = 2.0


def _truth_fixture():
    """Truth rows with the OUT-OF-WINDOW rows FIRST (the repo's own fixture
    rule for the one-sided-support bug class): sub-floor N, over-top N,
    out-of-z, and sub-SNR rows precede the in-window ones."""
    nhi = np.array([
        18.05, 18.55, 19.05, 19.45,     # BELOW the reporting floor 19.5
        22.45,                          # above the top edge 22.4
        19.55,                          # in N, but out of the z window
        19.65,                          # in N and z, but SNR <= 2 (cut)
        19.55, 19.95, 20.35, 21.05,     # in window
    ])
    z = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 1.5, 2.5, 2.5, 3.4, 2.05])
    s2n = np.array([5., 5., 5., 5., 5., 5., 5., 1.5, 5., 5., 5.])
    return dict(cfg=_FakeCfg(),
                truth_cut=_FakeTable(NHI=nhi, Z_DLA=z, S2N_RED=s2n))


def test_build_truth_counts_unpadded_matches_schema_v1_grid():
    tc, tc_bks, n_in = _truth_fixture(), None, None
    b = _truth_fixture()
    tc, tc_bks, n_in = EP.build_truth_counts(b)
    assert tc.shape == (EP.N_C, EP.N_K)
    # only the 4 genuinely in-window rows survive (sub-floor / over-top /
    # out-of-z / sub-SNR all dropped)
    assert n_in == 4 and tc.sum() == 4
    assert np.allclose(tc_bks.sum(axis=2), tc)


def test_build_truth_counts_padded_tail_is_bit_identical():
    """The pad may only ADD rows below the reporting floor. If the >= 19.5 part
    of the padded histogram moved, the ladder would not be like-for-like."""
    b = _truth_fixture()
    tc0, tc0_bks, n0 = EP.build_truth_counts(b)
    edges, n_pad = EP.basis_pad_edges(18.0)
    tc1, tc1_bks, n1 = EP.build_truth_counts(b, edges)
    assert tc1.shape == (EP.N_C + n_pad, EP.N_K)
    assert np.array_equal(tc1[n_pad:], tc0)
    assert np.array_equal(tc1_bks[n_pad:], tc0_bks)
    # the 4 sub-floor rows are now CARRIED (they were silently dropped in v1)
    assert tc1[:n_pad].sum() == 4
    assert n1 == n0 + 4
    # ... and they land in the right bins: 18.05 -> [18.0,18.1), 19.45 -> [19.4,19.5)
    lo = np.round(edges[:-1], 3)
    assert tc1[np.isclose(lo, 18.0), :].sum() == 1
    assert tc1[np.isclose(lo, 19.4), :].sum() == 1
    assert tc1[np.isclose(lo, 18.5), :].sum() == 1
    assert tc1[np.isclose(lo, 19.0), :].sum() == 1


def test_completeness_convention_names():
    assert EP.COMPLETENESS_CONVENTIONS == ("const_extrap", "molly172")
    with pytest.raises(ValueError, match="convention"):
        EP.load_molly_counts_block(convention="nonsense")


def test_privacy_guard_rejects_real_loa():
    with pytest.raises(RuntimeError):
        EP.assert_mock_only("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
                            "loa_main_dark_v1/whatever")
    # london0 catalog lives under gpdla_catalogs but is a MOCK — allowed
    EP.assert_mock_only("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
                        "london0_jura124_v1")


# ---------------------------------------------------------------------------
# pack tests — per generated pack
# ---------------------------------------------------------------------------
@needs_pack
@pytest.mark.parametrize("path", packs, ids=[os.path.basename(p) for p in packs])
def test_pack_schema_conformance(path):
    d, prov = _load(path)
    # -- edges (values, not just shapes)
    assert np.allclose(d["nhat_edges"], EP.NHAT_EDGES)
    # schema v1.1 (finding D1): ntrue_edges may EXTEND DOWN; nhat_edges must
    # stay an exact TAIL subset (the reporting window never moves).
    ne = np.asarray(d["ntrue_edges"], float)
    assert len(ne) >= len(EP.NHAT_EDGES)
    assert np.allclose(ne[len(ne) - len(EP.NHAT_EDGES):], EP.NHAT_EDGES)
    assert np.allclose(np.diff(ne), EP.N_STEP)
    n_b = len(ne) - 1
    assert np.allclose(d["zf_edges"], EP.ZF_EDGES)
    assert np.allclose(d["zc_edges"], EP.ZC_EDGES)
    assert np.array_equal(d["kz_to_K"], EP.KZ_TO_K)
    assert np.allclose(d["snr_edges"][:-1], EP.SNR_EDGES[:-1]) \
        and np.isinf(d["snr_edges"][-1])
    # -- shapes / dtypes / axis order (c, b, k, K, s)
    assert d["counts"].shape == (29, 15, 8) and d["counts"].dtype == np.int64
    assert d["dX"].shape == (15, 8)
    n_nhi = len(d["molly_nhi_edges"]) - 1
    assert d["molly_n_det"].shape == (8, n_nhi)
    assert d["molly_n_tot"].shape == (8, n_nhi)
    assert d["g_grid"].shape == (n_nhi, 15)
    assert d["g_occupancy"].shape == (n_nhi, 15)
    assert d["fp_counts"].shape == (29, 8) and d["fp_counts"].dtype == np.int64
    assert d["fp_E_alloc"].shape == (15, 8)
    assert d["fp_ell_eff"].shape == () and d["fp_w_sightline_ratio"].shape == ()
    assert d["t_sigma"].shape == (3,)
    assert d["truth_counts"].shape == (n_b, 15)
    assert d["truth_counts_bks"].shape == (n_b, 15, 8)
    assert d["resp_mu_coef"].ndim == 3
    assert d["nhat_masked_bins"].shape == (29,) and d["nhat_masked_bins"].dtype == bool
    # [19.5, 19.7) mask carried
    assert d["nhat_masked_bins"][0] and d["nhat_masked_bins"][1] \
        and not d["nhat_masked_bins"][2:].any()
    # -- finiteness (inf permitted ONLY as documented top-edge sentinels: molly
    # SNR/NHI cell grids + the forward response's (SNR, z) cell edges)
    inf_ok_last_edge = {"snr_edges", "molly_snr_edges", "molly_nhi_edges",
                        "resp_snr_edges", "resp_z_edges"}
    for key in d.files:
        a = np.asarray(d[key])
        if not np.issubdtype(a.dtype, np.number):
            continue
        if key in inf_ok_last_edge:
            assert np.all(np.isfinite(a.ravel()[:-1])), key
        else:
            assert np.all(np.isfinite(a)), f"non-finite values in {key}"
    # -- counts non-negative
    for key in ("counts", "fp_counts", "truth_counts", "truth_counts_bks",
                "molly_n_det", "molly_n_tot", "g_occupancy", "dX"):
        assert np.all(np.asarray(d[key]) >= 0), key
    # molly numerator <= denominator
    assert np.all(d["molly_n_det"] <= d["molly_n_tot"])
    # truth b,k marginal of the (b,k,s) cube must match
    assert np.allclose(d["truth_counts_bks"].sum(axis=2), d["truth_counts"])
    # no kappa-derived object anywhere (fail-closed guard)
    assert not any("kappa" in k.lower() for k in d.files)
    assert prov["guards"]["forward_kernel_assert"] is True
    assert prov["guards"]["no_kappa_keys"] is True
    assert prov["guards"]["no_rho"] is True


@needs_pack
@pytest.mark.parametrize("path", packs, ids=[os.path.basename(p) for p in packs])
def test_counts_total_equals_opmask_total(path):
    d, prov = _load(path)
    assert int(d["counts"].sum()) == int(prov["op_mask"]["n_op_in_window"])
    # window subset of the full op set
    assert prov["op_mask"]["n_op_in_window"] <= prov["op_mask"]["n_op_total"]


@needs_pack
@pytest.mark.parametrize("path", packs, ids=[os.path.basename(p) for p in packs])
def test_dx_marginals_vs_committed_xtot(path):
    d, prov = _load(path)
    # fold the fine (k,s) pathlength onto coarse z and compare to the committed
    # build_pathlength X_tot (persisted in the pack for exactly this test)
    px_coarse = np.zeros(3)
    for k in range(15):
        px_coarse[EP.KZ_TO_K[k]] += d["dX"][k, :].sum()
    x_tot = np.asarray(d["dX_coarse_committed"], float)
    gap = np.max(np.abs(px_coarse - x_tot) / np.maximum(x_tot, 1e-30))
    assert gap < DX_RTOL_SCHEMA, f"dX marginal gap {gap:.3e} vs X_tot"
    # pinned measured gap (different builders SHOULD agree analytically here;
    # a gap above the pin means one of the two conventions drifted)
    assert gap <= max(DX_RELGAP_PIN, float(prov["checks"]["dx_marginal_max_relgap"]) * 1.5 + 1e-15)
    assert np.all(d["dX"] >= 0)
    # strata below the SNR>2 op cut carry zero pathlength
    assert d["dX"][:, 0].sum() == 0 and d["dX"][:, 1].sum() == 0


@needs_pack
@pytest.mark.parametrize("path", packs, ids=[os.path.basename(p) for p in packs])
def test_fp_E_alloc_normalization(path):
    d, _ = _load(path)
    dX = d["dX"]
    E = d["fp_E_alloc"]
    for s in range(8):
        tot = dX[:, s].sum()
        if tot > 0:
            assert np.isclose(E[:, s].sum(), 1.0, atol=1e-12)
            assert np.allclose(E[:, s], dX[:, s] / tot)
        else:
            assert np.all(E[:, s] == 0.0)   # empty stratum: explicit zeros, not NaN


@needs_pack
@pytest.mark.parametrize("path", packs, ids=[os.path.basename(p) for p in packs])
def test_fp_block(path):
    d, prov = _load(path)
    # fp counts must reconcile with the committed product's z-windowed fine grid
    assert int(d["fp_counts"].sum()) == int(prov["fp"]["n_fp_fine_ge195_total"])
    # SNR strata below the op cut are empty by construction
    assert d["fp_counts"][:, 0].sum() == 0 and d["fp_counts"][:, 1].sum() == 0
    # exposure scaling: w = n_sl_mock / n_sl_loa0, ell_eff = ns0^2 / n_sl_mock
    ns0 = float(prov["fp"]["n_sl_loa0"])
    nsm = float(prov["fp"]["n_sl_mock"])
    assert np.isclose(float(d["fp_w_sightline_ratio"]), nsm / ns0, rtol=1e-12)
    assert np.isclose(float(d["fp_ell_eff"]), ns0 * ns0 / nsm, rtol=1e-12)


@needs_pack
@pytest.mark.parametrize("path", packs, ids=[os.path.basename(p) for p in packs])
def test_t_sigma_floor_and_value(path):
    d, _ = _load(path)
    t = np.asarray(d["t_sigma"], float)
    assert np.all(t >= 0.10 - 1e-12)
    expect, _ = EP.compute_t_sigma()
    assert np.allclose(t, expect, atol=1e-12)


@needs_pack
@pytest.mark.parametrize("path", packs, ids=[os.path.basename(p) for p in packs])
def test_pack_roundtrip_and_provenance(path):
    d, prov = _load(path)
    for key in ("code_commit", "routines", "inputs", "date", "guards", "mock"):
        assert key in prov
    # frozen calibration is 2LPT-0 for every pack
    assert "2lpt0" in prov["inputs"]["forward_model"] \
        or "2lpt0" in os.path.basename(prov["inputs"]["forward_model"])
    # g availability is explicit
    assert prov["g_available"] in (True, False)
    if not prov["g_available"]:
        assert np.allclose(d["g_grid"], 1.0)
    # privacy: no real-LOA path anywhere in the recorded inputs
    assert "loa_main_dark_v1" not in json.dumps(prov["inputs"])
