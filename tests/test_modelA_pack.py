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
    """Stands in for the astropy/structured table `load_and_cut_catalog`
    returns: column access by key, and ``len()`` = number of ROWS (not
    columns) — the extractor reports ``len(cat_cut)`` / ``len(truth_cut)`` as
    row counts in the provenance sidecar."""

    def __len__(self):
        if dict.__len__(self) == 0:
            return 0
        return len(next(iter(self.values())))


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


# ---------------------------------------------------------------------------
# D1 GUARD COVERAGE (2026-07-29, referee defect 1)
#
# The two guards that make the ladder's central claims true — "like-for-like"
# (the padded truth histogram reproduces the unpadded one over the reporting
# window) and ">= 19.5 molly cells stay bit-identical" — only execute inside a
# REAL extraction. Nothing exercised them, so disabling either left the suite
# green (MUTATION-VERIFIED by the referee: `if False:` => 58 passed).
#
# The tests below drive the REAL `extract_pack` / `load_molly_counts_block`
# code paths on a tiny synthetic bundle. Every heavy ingredient (catalog cut,
# build_M_b pathlength, forward NPZ, loa-0 product) is injected, but the guard
# statements themselves are the committed ones.
# ---------------------------------------------------------------------------
def _tiny_bundle(rows, snr_min=2.0):
    """A minimal bundle with the exact keys `extract_pack` reads.

    ``rows`` = (nhi, z, s2n, is_detection) columns; detections are the op-mask
    rows (the observed side), every row is a truth row.
    """
    nhi, z, s2n = (np.asarray(a, float) for a in rows[:3])
    det = np.asarray(rows[3], bool)
    tab = _FakeTable(NHI=nhi, Z_DLA=z, S2N_RED=s2n)
    cfg = _FakeCfg()
    cfg.snr_min = snr_min
    return dict(cfg=cfg, cat_cut=tab, truth_cut=tab, op_mask=det,
                X_tot=np.array([10.0, 20.0, 30.0]), n_sl=1234,
                meta=dict(n_loaded=len(nhi), truth_nhi_floor=19.5,
                          n_cat_cut=len(nhi), n_truth_cut=len(nhi)))


def _truth_only_bundle(nhi, z, s2n, snr_min=2.0):
    cfg = _FakeCfg()
    cfg.snr_min = snr_min
    return dict(cfg=cfg,
                truth_cut=_FakeTable(NHI=np.asarray(nhi, float),
                                     Z_DLA=np.asarray(z, float),
                                     S2N_RED=np.asarray(s2n, float)),
                meta=dict(n_loaded=len(nhi), truth_nhi_floor=float(min(nhi)),
                          n_cat_cut=len(nhi), n_truth_cut=len(nhi)))


def _tiny_frozen():
    """The frozen calibration blocks, shaped exactly as the schema requires but
    filled with cheap synthetic values (no NERSC/GPFS input is touched)."""
    n_molly = 3
    return dict(
        molly=dict(molly_n_det=np.full((EP.N_S, n_molly), 5.0),
                   molly_n_tot=np.full((EP.N_S, n_molly), 10.0),
                   molly_nhi_edges=np.array([19.5, 20.0, 21.0, np.inf]),
                   molly_snr_edges=EP.SNR_EDGES.copy()),
        molly_prov=dict(path="<synthetic>", max_c_diff=0.0,
                        convention="const_extrap", below_floor="<synthetic>"),
        g_grid=np.ones((n_molly, EP.N_K)), g_occupancy=np.zeros((n_molly, EP.N_K)),
        g_available=False,
        fwd=dict(resp_mu_coef=np.zeros((2, 3, 3)), resp_sig_coef=np.zeros((2, 3, 3)),
                 resp_skew_coef=np.zeros((2, 3, 3)),
                 resp_snr_edges=np.array([0., 3., np.inf]),
                 resp_z_edges=np.array([2.0, 2.5, 3.0, np.inf]),
                 resp_sig_floor=np.float64(0.1),
                 resp_skew_ramp=np.array([21.0, 0.5]),
                 resp_N_ref=np.float64(20.3)),
        fwd_meta=dict(path="<synthetic>", z_covariate="zqso",
                      fwd_response_kind="skewnormal", deg_N=2),
        fp_counts=np.zeros((EP.N_C, EP.N_S), dtype=np.int64),
        fp_prov=dict(product="<synthetic>", loa0_out="<synthetic>",
                     n_sl_loa0=1000.0),
        t_sigma=np.full(EP.N_KC, 0.10), t_sigma_detail={},
    )


@pytest.fixture
def _no_heavy_dX(monkeypatch):
    """build_dX goes through build_M_b (the full PX machinery). The guards under
    test are downstream of it, so stub it — everything else in extract_pack is
    the committed code."""
    monkeypatch.setattr(EP, "build_dX",
                        lambda bundle: np.full((EP.N_K, EP.N_S), 4.0))


# --- rows shared by the M1 guard tests ------------------------------------
# in-window truth+detection rows (>= 19.5) and the sub-floor rows a pad adds
_IN_WINDOW = [(19.55, 2.5, 5.0), (20.35, 2.6, 5.0), (21.05, 3.1, 8.0)]
_SUB_FLOOR = [(18.05, 2.5, 5.0), (18.55, 2.5, 5.0), (19.45, 3.1, 5.0)]


def _cols(rows):
    return ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])


def _run_extract(tmp_path, monkeypatch, pad_floor, truth_rows_padded,
                 tag="_guardtest"):
    """Drive the REAL extract_pack with injected bundles."""
    frozen = _tiny_frozen()
    nhi, z, s2n = _cols(_IN_WINDOW)
    frozen["_bundles"] = {"2lpt0": _tiny_bundle(
        (nhi, z, s2n, [True] * len(nhi)))}
    if truth_rows_padded is not None:
        tn, tz, ts = _cols(truth_rows_padded)
        frozen["_truth_bundles"] = {
            ("2lpt0", round(float(pad_floor), 3)):
                _truth_only_bundle(tn, tz, ts)}
    return EP.extract_pack("2lpt0", str(tmp_path), frozen,
                           pad_floor=pad_floor, tag=tag)


def test_M1_basis_pad_guard_passes_on_a_consistent_padded_truth(
        tmp_path, monkeypatch, _no_heavy_dX):
    """POSITIVE leg: a real extraction with a pad whose padded truth reproduces
    the unpadded histogram over [19.5, 22.4) must SUCCEED, and the written pack
    must carry the sub-floor rows in the extra basis bins only."""
    r = _run_extract(tmp_path, monkeypatch, 18.0, _SUB_FLOOR + _IN_WINDOW)
    d = np.load(r["npz"], allow_pickle=False)
    n_pad = len(d["ntrue_edges"]) - 1 - EP.N_C
    assert n_pad == 15
    assert d["truth_counts"][:n_pad].sum() == len(_SUB_FLOOR)
    assert d["truth_counts"][n_pad:].sum() == len(_IN_WINDOW)
    with open(r["provenance"]) as f:
        prov = json.load(f)
    assert prov["basis_pad"]["n_truth_below_reporting_floor"] == len(_SUB_FLOOR)


def test_M1_basis_pad_guard_FIRES_when_the_padded_truth_moves_in_window(
        tmp_path, monkeypatch, _no_heavy_dX):
    """MUTATION TARGET (extract_pack.py `if not np.array_equal(tc_pad[n_pad:],
    truth_counts)`): if the deeper truth cut perturbs a row at or above the
    reporting floor, the ladder is no longer like-for-like and the extraction
    MUST refuse. Disabling the guard makes this test go red."""
    moved = list(_IN_WINDOW)
    moved[0] = (20.95, 2.5, 5.0)          # 19.55 -> 20.95: an IN-WINDOW move
    with pytest.raises(RuntimeError, match="reproduce the unpadded histogram"):
        _run_extract(tmp_path, monkeypatch, 18.0, _SUB_FLOOR + moved)


def test_M1_basis_pad_guard_FIRES_on_the_bks_leg_alone(
        tmp_path, monkeypatch, _no_heavy_dX):
    """The (b,k,s) leg has independent power: a row that keeps its (N, z) bin
    but changes SNR stratum leaves the (b,k) histogram bit-identical and must
    still be refused."""
    moved = list(_IN_WINDOW)
    moved[0] = (19.55, 2.5, 3.5)          # s=5 -> s=3, same (b, k)
    with pytest.raises(RuntimeError, match=r"\(b,k,s\) padded truth"):
        _run_extract(tmp_path, monkeypatch, 18.0, _SUB_FLOOR + moved)


def test_M1_unpadded_extraction_never_enters_the_guarded_branch(
        tmp_path, monkeypatch, _no_heavy_dX):
    """Schema v1 regression lock, through the REAL extractor: with no pad the
    truth bundle is never loaded and ntrue_edges == nhat_edges."""
    called = []
    monkeypatch.setattr(EP, "load_truth_bundle",
                        lambda *a, **k: called.append(a) or (_ for _ in ()).throw(
                            AssertionError("load_truth_bundle called with no pad")))
    r = _run_extract(tmp_path, monkeypatch, None, None, tag="_guardtest_nopad")
    d = np.load(r["npz"], allow_pickle=False)
    assert np.array_equal(np.asarray(d["ntrue_edges"]), EP.NHAT_EDGES)
    assert called == []
    with open(r["provenance"]) as f:
        prov = json.load(f)
    assert prov["basis_pad"]["n_pad_bins"] == 0


def test_padded_sidecar_truth_block_is_internally_coherent(
        tmp_path, monkeypatch, _no_heavy_dX):
    """Referee minor (2026-07-29): a padded pack used to report MORE truth
    systems 'in window' than were 'cut' (146792 cut vs 147188 in window on
    2LPT-0 at pad 18.0) because the two totals came from DIFFERENT bundles, and
    it carried two contradictory truth floors with no way to tell them apart.
    Both totals must now come from the bundle the saved histogram was built
    from, and each floor must be labelled."""
    # padded truth bundle deliberately LARGER than the detection bundle's
    r = _run_extract(tmp_path, monkeypatch, 18.0, _SUB_FLOOR + _IN_WINDOW)
    with open(r["provenance"]) as f:
        prov = json.load(f)
    t = prov["truth"]
    assert t["n_truth_in_window"] <= t["n_truth_cut"], (
        f"sidecar claims {t['n_truth_in_window']} in window but only "
        f"{t['n_truth_cut']} cut")
    assert t["n_truth_cut"] == len(_SUB_FLOOR) + len(_IN_WINDOW)
    assert t["n_truth_cut_detection_bundle"] == len(_IN_WINDOW)
    # the two floors are both present and each says which object it describes
    assert t["truth_nhi_floor"] == 18.0
    assert t["truth_nhi_floor_detection_bundle"] == 19.5
    assert prov["cut_meta"]["truth_nhi_floor"] == 19.5
    assert "detection bundle" in t["source_bundle"] or "padded" in t["source_bundle"]
    # the ~0.1% cat_cut discrepancy is stated in the SIDECAR, not only in the
    # module docstring (referee minor)
    assert "582855" in prov["basis_pad"]["cat_cut_discrepancy"]
    assert "88071" in prov["basis_pad"]["cat_cut_discrepancy"]
    assert prov["basis_pad"]["n_truth_cut_padded_bundle"] == \
        len(_SUB_FLOOR) + len(_IN_WINDOW)


def test_unpadded_sidecar_truth_block_names_one_floor(
        tmp_path, monkeypatch, _no_heavy_dX):
    r = _run_extract(tmp_path, monkeypatch, None, None, tag="_prov_nopad")
    with open(r["provenance"]) as f:
        prov = json.load(f)
    t = prov["truth"]
    assert t["truth_nhi_floor"] == t["truth_nhi_floor_detection_bundle"] == 19.5
    assert t["n_truth_cut"] == t["n_truth_cut_detection_bundle"]
    assert t["n_truth_in_window"] <= t["n_truth_cut"]
    assert "cat_cut_discrepancy" not in prov["basis_pad"]


# --- M2: the molly172 splice tail-subset check ------------------------------
_CANON_EDGES = np.array([19.5, 20.0, 21.0, np.inf])


def _fake_canonical_counts(path="<synthetic canonical>"):
    return dict(snr_edges=EP.SNR_EDGES.copy(), nhi_edges=_CANON_EDGES.copy(),
                cmp_nfound=np.full((EP.N_S, 3), 6.0),
                cmp_nfid=np.full((EP.N_S, 3), 10.0),
                max_c_diff=0.0, path=path)


def _write_counts172(tmp_path, edges, name="molly_counts_nhi172.npz"):
    p = str(tmp_path / name)
    n = len(edges) - 1
    np.savez(p, snr_edges=EP.SNR_EDGES, nhi_edges=np.asarray(edges, float),
             cmp_nfound=np.arange(EP.N_S * n, dtype=float).reshape(EP.N_S, n),
             cmp_nfid=np.full((EP.N_S, n), 100.0), max_c_diff=0.0)
    return p


def test_M2_molly172_splice_guard_passes_and_keeps_ge195_bit_identical(
        tmp_path, monkeypatch):
    """POSITIVE leg: the >= 19.5 cells of the spliced block must be BIT-identical
    to the canonical matrix — that is the claim the whole convention rests on."""
    monkeypatch.setattr(EP.FF, "load_molly_counts",
                        lambda *a, **k: _fake_canonical_counts())
    monkeypatch.setattr(
        "CDDF_analysis.hbi.cddf_catalog_hbi.load_molly_matrix",
        lambda *a, **k: "<mm_alt sentinel>")
    p172 = _write_counts172(tmp_path, [18.0, 18.5, 19.0, 19.5, 20.0, 21.0, np.inf])
    block, prov, mm_alt = EP.load_molly_counts_block(
        convention="molly172", counts172_path=p172)
    canon = _fake_canonical_counts()
    assert prov["n_cells_spliced_below_floor"] == 3
    assert np.array_equal(block["molly_n_det"][:, 3:], canon["cmp_nfound"])
    assert np.array_equal(block["molly_n_tot"][:, 3:], canon["cmp_nfid"])
    assert np.allclose(block["molly_nhi_edges"][3:-1], _CANON_EDGES[:-1])
    assert mm_alt == "<mm_alt sentinel>"


@pytest.mark.parametrize("edges,why", [
    ([18.0, 18.5, 19.0, 19.6, 20.0, 21.0, np.inf], "interior edge 19.5 -> 19.6"),
    ([18.0, 18.5, 19.0, 19.5, 20.0, 21.5, np.inf], "interior edge 21.0 -> 21.5"),
    ([18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0], "top edge not +inf"),
])
def test_M2_molly172_splice_guard_FIRES_when_edges_are_not_a_tail_subset(
        tmp_path, monkeypatch, edges, why):
    """MUTATION TARGET (extract_pack.py `if not (np.allclose(tail[:-1], ...)
    and np.isposinf(...))`): splicing a floor-17.2 matrix whose >= 19.5 cells do
    NOT coincide with the canonical ones would silently change the completeness
    INSIDE the reporting window. Disabling the check makes these go red."""
    monkeypatch.setattr(EP.FF, "load_molly_counts",
                        lambda *a, **k: _fake_canonical_counts())
    p172 = _write_counts172(tmp_path, edges)
    with pytest.raises(RuntimeError, match="not a tail"):
        EP.load_molly_counts_block(convention="molly172", counts172_path=p172)


def test_M2_molly172_splice_guard_FIRES_on_mismatched_snr_strata(
        tmp_path, monkeypatch):
    """Companion leg: the SNR strata must match too (a different stratification
    would make the two matrices non-spliceable in the s axis)."""
    monkeypatch.setattr(EP.FF, "load_molly_counts",
                        lambda *a, **k: _fake_canonical_counts())
    p = str(tmp_path / "bad_snr.npz")
    np.savez(p, snr_edges=np.arange(EP.N_S + 1, dtype=float),
             nhi_edges=np.array([18.0, 19.0, 19.5, 20.0, 21.0, np.inf]),
             cmp_nfound=np.zeros((EP.N_S, 5)), cmp_nfid=np.ones((EP.N_S, 5)),
             max_c_diff=0.0)
    with pytest.raises(RuntimeError, match="SNR strata differ"):
        EP.load_molly_counts_block(convention="molly172", counts172_path=p)


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
def test_production_packs_are_unpadded(path):
    """Referee minor (2026-07-29): ``test_pack_schema_conformance`` was relaxed
    from ``ntrue_edges == nhat_edges`` to a TAIL-SUBSET assertion so the D1
    ladder's padded packs would validate — which left NOTHING pinning that a
    PRODUCTION pack is unpadded. The pad is a research configuration; a pack in
    the production pack dir must still be schema v1 on the true-N axis, and its
    sidecar must say so.

    (The ladder's padded packs live in a scratch dir and are never in
    ``MODELA_PACK_DIR``; if a padded pack ever lands here, this is the tripwire.)
    """
    d, prov = _load(path)
    ne = np.asarray(d["ntrue_edges"], float)
    assert np.array_equal(ne, EP.NHAT_EDGES), (
        f"PRODUCTION pack {os.path.basename(path)} carries a basis pad "
        f"({len(ne) - len(EP.NHAT_EDGES)} extra bins down to {ne[0]}). Padded "
        "packs belong in a scratch ladder dir, not the production pack dir.")
    assert d["truth_counts"].shape[0] == EP.N_C
    # sidecars written before schema v1.1 carry no basis_pad block at all —
    # which is itself unambiguous evidence of "no pad". If the block IS there
    # it must agree with the arrays.
    if "basis_pad" in prov:
        assert prov["basis_pad"]["n_pad_bins"] == 0
        assert prov["basis_pad"]["pad_floor"] is None


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
