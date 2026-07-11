"""extract_pack.py — Model A data-pack EXTRACTOR (Queue 3 §1 data plane).

Contract: modelA_pack_schema.md (BINDING) + notes/2026-07-11_q3_modelA_spec.md §1.
One NPZ per mock (`modelA_pack_<mock>.npz`) + a `.provenance.json` sidecar; the
NumPyro module consumes ONLY the pack (no fitsio/h5py inside the model).

REUSE RULE (approved): every ingredient is READ through the committed legacy
machinery — never re-derived:

  * detections + op-mask ......... cddf_catalog_hbi.load_and_cut_catalog (+ the
        ab_loa0_fp_baseline op_mask line: S2N_RED>2 strict & P_DLA>0.99 strict &
        good_mask[DLAFLAG==0], BAL veto + z_qso window inside the cut bundle),
        with track_c_tf_saclay._snap_off_molly_edges applied first (the committed
        interior-edge tie-break).
  * dX (k=15, s=8) ............... cddf_catalog_hbi.build_M_b's PX machinery
        (per-SNR-cell, per-fine-z analytic pathlength over the SAME per-sightline
        windows build_pathlength carves: lam_rf [1025,1216], 3000 km/s collars,
        3600 A floor, SNR>2 sightlines only).
  * molly counts ................. ff_fp_estimator.build_molly_counts_cache /
        load_molly_counts (matched-real numerator; purity NEVER read).
  * g(N,z) ....................... cddf_catalog_hbi.build_cnz_resolved (the frozen
        2LPT-0 level-preserving z-shape) + znz_kernel.measure_c_nz occupancy.
  * forward response ............. the frozen ForwardResponseModel NPZ
        (track_c_tf_loa._DEF_FORWARD); ASSERTED forward (no kappa anywhere).
  * loa-0 FP ..................... Loa0FP.from_product for scalars/consistency +
        build_loa0_fp_product.load_loa0_fp_catalog for the raw loa-0 FP detections
        (re-binned onto the schema (c=29, s=8) grid with the product's OWN op cut;
        guarded by exact equality of the re-derived molly-cell counts against the
        committed product's n_fp_molly).
  * t_sigma (K=3) ................ committed Q2 artifacts CDDF_analysis/hbi/
        ff_fp_{saclay0,london0}.json per-coarse-z sub-DLA closure ratios relative
        to 2lpt0: t_sigma[K] = max(0.10, max_mock |ln(R_z_mock / R_z_2lpt0)|).
  * truth_counts ................. the truth table from the SAME cut bundle
        (windowed identically to the detections/pathlength), SNR>2 strict + the
        half-open binning conventions of calccddf_vs_hbi.truth_one_file.

AXIS ORDER (schema rule, "in this order everywhere"): c, b, k, K, s. In
particular `fp_counts` is (c=29, s=8) — the schema's prose lists "(s=8 ..., c=29
...)" but the global axis rule wins; pinned in provenance["fp"]["axes"].

MOCKS ONLY: 2lpt0 first, then saclay0/london0. This module HARD-REFUSES any
input path under loa_main_dark_v1 (real LOA — private).

CLI (script form, NOT `-m`: the hbi_mcmc package __init__ imports jax, which the
`gpdla` data-plane env deliberately does not carry — this module needs no jax):
  conda run -n gpdla python CDDF_analysis/hbi_mcmc/extract_pack.py \
      --mocks 2lpt0 [saclay0 london0] \
      --out-dir /scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi import ab_loa0_fp_baseline as AB                # noqa: E402
from CDDF_analysis.hbi import track_c_tf_loa as TF                     # noqa: E402
from CDDF_analysis.hbi import track_c_tf_saclay as TS                  # noqa: E402
from CDDF_analysis.hbi import track_c_tf_london0 as TL                 # noqa: E402
from CDDF_analysis.hbi import ff_fp_estimator as FF                    # noqa: E402
from CDDF_analysis.hbi.build_loa0_fp_product import (                  # noqa: E402
    load_loa0_fp_catalog, DEF_LOA0_OUT,
)
from CDDF_analysis.hbi.cddf_catalog_hbi import (                       # noqa: E402
    HBIConfig, LYA_REST, Loa0FP, _build_qso_lookup, _fine_z_grid,
    build_cnz_resolved, build_fine_grid, build_M_b, build_pathlength,
    load_and_cut_catalog, load_molly_matrix,
)
from CDDF_analysis.hbi.znz_kernel import measure_c_nz                  # noqa: E402

# ---------------------------------------------------------------------------
# schema grids (BINDING — modelA_pack_schema.md)
# ---------------------------------------------------------------------------
NHAT_EDGES = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 3)   # 30 edges -> 29 bins
N_C = len(NHAT_EDGES) - 1                                     # 29
ZF_EDGES = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 3)       # 16 edges -> 15 bins
N_K = len(ZF_EDGES) - 1                                       # 15
ZC_EDGES = np.array([2.0, 2.5, 3.0, 3.5])                     # 4 edges -> 3 bins
N_KC = len(ZC_EDGES) - 1                                      # 3
SNR_EDGES = np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf])  # molly cells
N_S = len(SNR_EDGES) - 1                                      # 8
_zmid = 0.5 * (ZF_EDGES[:-1] + ZF_EDGES[1:])
KZ_TO_K = (np.searchsorted(ZC_EDGES, _zmid, side="right") - 1).astype(np.int64)

T_SIGMA_FLOOR = 0.10
SUBDLA_TIER = "subdla_195_203"
Z_TAGS = ("z_2.0_2.5", "z_2.5_3.0", "z_3.0_3.5")

DEF_OUT_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "modelA_packs")

# per-mock inputs: the committed drivers' own defaults (read, not retyped where a
# module constant exists)
MOCKS = {
    "2lpt0": dict(
        catalog_dir=AB.DEF_CAT, truth_path=AB.DEF_TRUTH, bal_cat_path=AB.DEF_BAL,
        mockdir=os.path.dirname(AB.DEF_TRUTH)),
    "saclay0": dict(
        catalog_dir=TS._S0_CAT, truth_path=TS._S0_TRUTH, bal_cat_path=TS._S0_BAL,
        mockdir=TS._S0_MOCKDIR),
    "london0": dict(
        catalog_dir=TL._L0_CAT, truth_path=TL._L0_TRUTH, bal_cat_path=TL._L0_BAL,
        mockdir=TL._L0_MOCKDIR),
}


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def assert_mock_only(path: str) -> None:
    """PRIVACY: refuse ANY path under the real-LOA catalog (mocks only)."""
    if path and "loa_main_dark_v1" in str(path):
        raise RuntimeError(
            f"PRIVACY GUARD: refusing real-LOA path {path!r} — the Model A "
            "extractor is mock-only (loa_main_dark_v1 is private real data).")


def _git_commit() -> str:
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO,
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPO,
            stderr=subprocess.DEVNULL).decode().strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# binning helpers (schema half-open [lo, hi) conventions)
# ---------------------------------------------------------------------------
def _idx(edges: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, np.asarray(x, float), side="right") - 1


def bin_counts_cks(nhat, zhat, snr):
    """Histogram detections onto the (c=29, k=15, s=8) grid; returns
    (counts int64, n_in_window). Out-of-window rows (N̂ outside [19.5,22.4) or
    ẑ outside [2.0,3.5)) are dropped — the caller reports both totals."""
    nhat = np.asarray(nhat, float)
    zhat = np.asarray(zhat, float)
    snr = np.asarray(snr, float)
    c = _idx(NHAT_EDGES, nhat)
    k = _idx(ZF_EDGES, zhat)
    s = np.clip(_idx(SNR_EDGES, snr), 0, N_S - 1)
    ok = (c >= 0) & (c < N_C) & (k >= 0) & (k < N_K) & np.isfinite(nhat) \
        & np.isfinite(zhat)
    counts = np.zeros((N_C, N_K, N_S), dtype=np.int64)
    np.add.at(counts, (c[ok], k[ok], s[ok]), 1)
    return counts, int(ok.sum())


# ---------------------------------------------------------------------------
# frozen 2LPT-0 calibration blocks (identical in every pack)
# ---------------------------------------------------------------------------
def load_forward_response_pack(path: str = TF._DEF_FORWARD):
    """Dump the frozen ForwardResponseModel NPZ coefficient arrays per schema.
    FAIL-CLOSED: asserts the envelope is the skew-normal FORWARD response — no
    kappa object can pass (this is the sub-DLA/Track-C defect class guard)."""
    assert_mock_only(path)
    d = np.load(path, allow_pickle=True)
    kind = str(np.asarray(d["_fwd_response_kind"]))
    if "skewnormal" not in kind:
        raise RuntimeError(
            f"forward-kernel assert FAILED: {path} kind={kind!r} is not the "
            "skew-normal FORWARD response (kappa/posterior kernels are refused).")
    for key in d.files:
        if "kappa" in key.lower():
            raise RuntimeError(
                f"forward-kernel assert FAILED: kappa-derived key {key!r} in {path}")
    if os.path.basename(path).startswith("posterior_kernel"):
        raise RuntimeError(f"forward-kernel assert FAILED: posterior kernel {path}")
    # skew ramp: ForwardResponseModel.skew ramps gamma->0 linearly over a 0.5-dex
    # window above N_skew_collapse (znz_kernel.py:1362 `clip((N-collapse)/0.5,0,1)`)
    fwd = dict(
        resp_mu_coef=np.asarray(d["mu_coef"], float),
        resp_sig_coef=np.asarray(d["sig_coef"], float),
        resp_skew_coef=np.asarray(d["skew_coef"], float),
        resp_snr_edges=np.asarray(d["snr_edges"], float),
        resp_z_edges=np.asarray(d["z_edges"], float),
        resp_sig_floor=np.float64(d["sig_floor"]),
        resp_skew_ramp=np.array([float(d["N_skew_collapse"]), 0.5]),
        resp_N_ref=np.float64(d["N_ref"]),      # required to evaluate the coef polys
    )
    meta = dict(
        fwd_response_kind=kind,
        deg_N=int(d["deg_N"]),
        z_covariate=(str(np.asarray(d["z_covariate"])) if "z_covariate" in d.files
                     else "zqso"),
        path=path,
        has_empirical_block=bool("emp_rho" in d.files),
    )
    return fwd, meta


def compute_t_sigma(hbi_dir: str = None, floor: float = T_SIGMA_FLOOR):
    """t_sigma[K] = max(floor, max over held-out mocks of |ln(R_z / R_z^2lpt0)|)
    from the committed Q2 ff_fp_*.json per-coarse-z sub-DLA closure ratios
    (strata[z].closure.subdla_195_203.R_point — the combined FF+FP residual)."""
    hbi_dir = hbi_dir or os.path.join(_REPO, "CDDF_analysis", "hbi")
    R = {}
    for m in ("2lpt0", "saclay0", "london0"):
        with open(os.path.join(hbi_dir, f"ff_fp_{m}.json")) as f:
            d = json.load(f)
        R[m] = np.array([d["strata"][z]["closure"][SUBDLA_TIER]["R_point"]
                         for z in Z_TAGS], float)
    t = np.zeros(N_KC)
    detail = {}
    for K in range(N_KC):
        resid = {m: float(abs(np.log(R[m][K] / R["2lpt0"][K])))
                 for m in ("saclay0", "london0")}
        t[K] = max(floor, max(resid.values()))
        detail[Z_TAGS[K]] = dict(R_2lpt0=float(R["2lpt0"][K]),
                                 R_saclay0=float(R["saclay0"][K]),
                                 R_london0=float(R["london0"][K]),
                                 abs_ln_ratio=resid, floored=bool(t[K] == floor))
    return t, detail


def load_molly_counts_block():
    """Molly completeness counts (matched-real numerator) via the committed
    ff_fp_estimator cache; build it if absent. Purity is never read (RhoGuard)."""
    mc = FF.load_molly_counts()
    if mc is None:
        FF.build_molly_counts_cache()
        mc = FF.load_molly_counts()
    return dict(
        molly_n_det=np.asarray(mc["cmp_nfound"], float),   # (s=8, nhi_cells)
        molly_n_tot=np.asarray(mc["cmp_nfid"], float),
        molly_nhi_edges=np.asarray(mc["nhi_edges"], float),
        molly_snr_edges=np.asarray(mc["snr_edges"], float),
    ), dict(path=mc["path"], max_c_diff=float(mc["max_c_diff"]))


def build_fp_block(loa0_out: str = DEF_LOA0_OUT,
                   product_path: str = AB.DEF_LOA0_PRODUCT):
    """loa-0 forest-FP block. Scalars/consistency from the COMMITTED product
    (Loa0FP.from_product inputs); the (c=29, s=8) fp_counts from the raw loa-0
    dlacat re-binned with the product's OWN op cut (SNR>2 strict, P_DLA>0.99
    strict, lya_only lam_rest>=1025, Z_DLA in [2.0,3.5) as the fine mu_FP grid).

    GUARD: the re-derived molly-cell counts must equal the committed product's
    n_fp_molly EXACTLY (else the raw catalog drifted from the product)."""
    assert_mock_only(loa0_out); assert_mock_only(product_path)
    loa0 = Loa0FP.from_product(product_path)          # committed loader
    prod = np.load(product_path, allow_pickle=True)
    snr_min = float(prod["snr_min"]); p_dla_min = float(prod["p_dla_min"])
    lya_min = float(prod["lya_only_lam_rf_min"])

    cat = load_loa0_fp_catalog(loa0_out)              # committed raw loader
    snr = np.asarray(cat["SNR_REDSIDE"], float)
    pdla = np.asarray(cat["P_DLA"], float)
    nhi = np.asarray(cat["NHI"], float)
    z_dla = np.asarray(cat["Z_DLA"], float)
    op = (snr > snr_min) & (pdla > p_dla_min)
    if lya_min > 0:
        z_qso = np.asarray(cat["Z_QSO"], float)
        op &= (LYA_REST * (1.0 + z_dla) / (1.0 + z_qso)) >= lya_min
    nhi, snr, z_dla = nhi[op], snr[op], z_dla[op]

    # GUARD: exact replication of the committed product's molly-cell counts
    i, j = loa0._cell_idx(nhi, snr)
    n_molly_rederived = np.zeros_like(loa0.n_fp_molly)
    np.add.at(n_molly_rederived, (i, j), 1.0)
    if not np.array_equal(n_molly_rederived, loa0.n_fp_molly):
        raise RuntimeError(
            "loa-0 FP re-bin GUARD failed: raw dlacat op-binned molly counts != "
            f"committed product n_fp_molly ({product_path}) — inputs drifted.")

    # schema grid: c on nhat 0.1-dex bins, s on the molly SNR strata; z-windowed
    # to [2.0, 3.5) exactly like the product's fine mu_FP grid (n_fp_fine).
    c = _idx(NHAT_EDGES, nhi)
    s = np.clip(_idx(SNR_EDGES, snr), 0, N_S - 1)
    zok = (z_dla >= ZF_EDGES[0]) & (z_dla < ZF_EDGES[-1])
    ok = (c >= 0) & (c < N_C) & zok
    fp_counts = np.zeros((N_C, N_S), dtype=np.int64)
    np.add.at(fp_counts, (c[ok], s[ok]), 1)

    # cross-check vs the committed product's z-windowed fine grid over N>=19.5
    b195 = int(np.searchsorted(np.round(loa0.logN_lo, 3), 19.5))
    n_fine_ge195 = int(loa0.n_fp_fine[b195:, :].sum())
    if int(fp_counts.sum()) != n_fine_ge195:
        raise RuntimeError(
            f"loa-0 FP re-bin GUARD failed: fp_counts total {int(fp_counts.sum())} "
            f"!= committed n_fp_fine[N>=19.5] total {n_fine_ge195}.")

    prov = dict(
        product=product_path, loa0_out=loa0_out,
        op_cut=dict(snr_min=snr_min, p_dla_min=p_dla_min,
                    lya_only_lam_rf_min=lya_min,
                    z_window=[float(ZF_EDGES[0]), float(ZF_EDGES[-1])]),
        n_fp_op_total=int(len(nhi)),
        n_fp_in_c_window_all_z=int(((c >= 0) & (c < N_C)).sum()),
        n_fp_fine_ge195_total=n_fine_ge195,
        n_sl_loa0=float(loa0.n_sl_loa0),
        molly_rebin_guard="EXACT match to committed n_fp_molly",
        axes="fp_counts is (c=29, s=8) — schema global axis order c,b,k,K,s",
    )
    return fp_counts, loa0, prov


# ---------------------------------------------------------------------------
# per-mock extraction
# ---------------------------------------------------------------------------
def _make_cfg(mock: str, out_dir: str) -> HBIConfig:
    m = MOCKS[mock]
    for p in m.values():
        assert_mock_only(p)
    molly_tsv = FF.DEF_MOLLY_TSV                    # frozen 2LPT-0 lya_only-nhi195
    return HBIConfig(
        catalog_dir=m["catalog_dir"], truth_path=m["truth_path"],
        bal_cat_path=m["bal_cat_path"], molly_tsv=molly_tsv, out_dir=out_dir,
        mockdir=m["mockdir"],
        zbins=tuple(ZC_EDGES.tolist()),
        lam_rf_min=1025.0,                          # lya_only window (schema/dX)
        no_bal=True,
        v2_z_fit_lo=2.0, v2_z_fit_hi=3.5, v2_z_fit_step=0.1,
        completeness_z_resolved=True, completeness_z_min_count=30.0,
        rng_seed=0,
    )


def load_mock_bundle(mock: str, out_dir: str):
    """Load one mock's cut bundle + op mask + per-sightline pathlength through
    the committed machinery (ab_loa0_fp_baseline.build_ingredients path)."""
    cfg = _make_cfg(mock, out_dir)
    mm = load_molly_matrix(cfg.molly_tsv)
    truth_floor = float(mm.nhi_edges[0])            # 19.5 (nhi195 matrix)
    t0 = time.time()
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=truth_floor, qso_lookup=qso_lookup,
        host_truth_floor=min(19.0, truth_floor))
    # committed interior-edge tie-break (saclay has one NHI_TRUE==20.0 row;
    # NO-OP on 2lpt0/london0)
    TS._snap_off_molly_edges(cat_cut, truth_cut, mm)
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op_mask = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    print(f"  [{mock}] bundle: n_cat_cut={len(cat_cut)}, n_op={int(op_mask.sum())}, "
          f"n_sl={n_sl} ({time.time()-t0:.0f}s)")
    return dict(cfg=cfg, mm=mm, cat_cut=cat_cut, truth_cut=truth_cut,
                good_mask=good_mask, op_mask=op_mask, meta=meta,
                X_tot=np.asarray(X_tot, float), n_sl=int(n_sl),
                qzl=qzl, qzh=qzh, qsn=qsn, Xcalc=Xcalc)


def build_dX(bundle):
    """dX (k=15, s=8) via the committed build_M_b PX machinery (PX is (s, kz))."""
    cfg = bundle["cfg"]
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    z_edges_fine = _fine_z_grid(cfg)
    assert np.allclose(z_edges_fine, ZF_EDGES)
    M_meta = build_M_b(bundle["qzl"], bundle["qzh"], bundle["qsn"], bundle["mm"],
                       logN_lo, logN_hi, N_b, dN_b, z_edges_fine,
                       bundle["Xcalc"], cfg)
    PX = np.asarray(M_meta["PX"], float)            # (n_snr=8, n_zf=15)
    assert PX.shape == (N_S, N_K)
    return PX.T.copy()                              # (k, s)


def build_g_block(bundle):
    """Frozen 2LPT-0 g(N,z) + occupancy via the committed builders (build ONCE on
    the calibration mock; embedded identically in every pack)."""
    cfg, mm = bundle["cfg"], bundle["mm"]
    cnz = build_cnz_resolved(cfg, bundle["cat_cut"], bundle["truth_cut"],
                             bundle["good_mask"], mm)
    meas = measure_c_nz(bundle["cat_cut"], bundle["truth_cut"], cfg, mm,
                        _fine_z_grid(cfg), good_mask=bundle["good_mask"])
    g_grid = np.asarray(cnz.g_grid, float)                       # (n_nhi, 15)
    g_occ = np.asarray(meas["n_true"], float)                    # (n_nhi, 15)
    assert g_grid.shape == g_occ.shape == (len(mm.nhi_edges) - 1, N_K)
    return g_grid, g_occ


def build_truth_counts(bundle):
    """truth_counts (b=29, k=15) [+ (b,k,s)] — truth systems from the SAME cut
    bundle (identical windowing), SNR>2 strict (truth_one_file convention),
    half-open binning on the schema grids."""
    t = bundle["truth_cut"]
    cfg = bundle["cfg"]
    n = np.asarray(t["NHI"], float)
    z = np.asarray(t["Z_DLA"], float)
    s2n = np.asarray(t["S2N_RED"], float)
    keep = s2n > cfg.snr_min
    b = _idx(NHAT_EDGES, n[keep])
    k = _idx(ZF_EDGES, z[keep])
    s = np.clip(_idx(SNR_EDGES, s2n[keep]), 0, N_S - 1)
    ok = (b >= 0) & (b < N_C) & (k >= 0) & (k < N_K)
    tc_bks = np.zeros((N_C, N_K, N_S), float)
    np.add.at(tc_bks, (b[ok], k[ok], s[ok]), 1.0)
    return tc_bks.sum(axis=2), tc_bks, int(ok.sum())


# ---------------------------------------------------------------------------
# pack assembly
# ---------------------------------------------------------------------------
def extract_pack(mock: str, out_dir: str, frozen: dict) -> dict:
    """Build + save one mock's data pack. `frozen` carries the 2LPT-0 calibration
    blocks (forward / molly counts / g / fp / t_sigma) built once."""
    print(f"[pack] extracting {mock} ...")
    t0 = time.time()
    bundle = frozen.get("_bundle_2lpt0") if mock == "2lpt0" else None
    if bundle is None:
        bundle = load_mock_bundle(mock, out_dir)

    cat = bundle["cat_cut"]; op = bundle["op_mask"]
    nhat = np.asarray(cat["NHI"], float)[op]
    zhat = np.asarray(cat["Z_DLA"], float)[op]
    snr_op = np.asarray(cat["S2N_RED"], float)[op]
    counts, n_in_window = bin_counts_cks(nhat, zhat, snr_op)
    assert int(counts.sum()) == n_in_window

    dX = build_dX(bundle)
    # dX marginal vs the committed build_pathlength X_tot per coarse z
    px_coarse = np.array([dX[KZ_TO_K == K, :].sum() for K in range(N_KC)])
    x_tot = bundle["X_tot"]
    dx_gap = float(np.max(np.abs(px_coarse - x_tot) / np.maximum(x_tot, 1e-30)))

    # exposure allocation: E[k,s] = dX[k,s] / sum_k dX[k,s]; empty strata -> 0
    col = dX.sum(axis=0)
    fp_E_alloc = np.zeros_like(dX)
    nz = col > 0
    fp_E_alloc[:, nz] = dX[:, nz] / col[nz]

    truth_counts, truth_counts_bks, n_truth_in_window = build_truth_counts(bundle)

    # per-mock loa-0 exposure scalars (Loa0FP.from_product n_sl_prod semantics)
    ns0 = float(frozen["fp_prov"]["n_sl_loa0"])
    nsm = float(bundle["n_sl"])
    fp_w = nsm / ns0                                # vol_scale = N_prod/N_sl_loa0
    fp_ell = ns0 * (ns0 / nsm)                      # ell_eff at this mock's N_prod

    masked = np.zeros(N_C, dtype=bool)
    masked[(NHAT_EDGES[:-1] >= 19.5 - 1e-9) & (NHAT_EDGES[1:] <= 19.7 + 1e-9)] = True

    pack = dict(
        # axes
        nhat_edges=NHAT_EDGES, ntrue_edges=NHAT_EDGES.copy(),
        zf_edges=ZF_EDGES, zc_edges=ZC_EDGES, kz_to_K=KZ_TO_K,
        snr_edges=SNR_EDGES,
        nhat_masked_bins=masked,                    # explicit mask array (no NaN codes)
        # data plane
        counts=counts,                              # (c,k,s) int64
        dX=dX,                                      # (k,s)
        dX_coarse_committed=x_tot,                  # (K,) build_pathlength X_tot
        # molly completeness counts (frozen 2LPT-0)
        **frozen["molly"],
        # g(N,z) (frozen 2LPT-0)
        g_grid=frozen["g_grid"], g_occupancy=frozen["g_occupancy"],
        # forward response (frozen 2LPT-0)
        **frozen["fwd"],
        # loa-0 FP
        fp_counts=frozen["fp_counts"],              # (c,s) int64
        fp_ell_eff=np.float64(fp_ell),
        fp_w_sightline_ratio=np.float64(fp_w),
        fp_E_alloc=fp_E_alloc,                      # (k,s)
        # transfer prior widths
        t_sigma=frozen["t_sigma"],                  # (K,)
        # truth (mocks only, closure)
        truth_counts=truth_counts,                  # (b,k)
        truth_counts_bks=truth_counts_bks,          # (b,k,s)
    )

    # finiteness assert (inf permitted only as the documented top-edge sentinels:
    # the molly SNR/NHI cell grids and the forward response's (SNR, z) cell edges
    # all close with +inf in the committed calibration objects)
    inf_ok = {"snr_edges", "molly_snr_edges", "molly_nhi_edges",
              "resp_snr_edges", "resp_z_edges"}
    for key, a in pack.items():
        a = np.asarray(a)
        if not np.issubdtype(a.dtype, np.number):
            continue
        flat = a.ravel()
        chk = flat[:-1] if key in inf_ok else flat
        if not np.all(np.isfinite(chk)):
            raise RuntimeError(f"non-finite values in pack key {key!r}")
    if any("kappa" in k.lower() for k in pack):
        raise RuntimeError("kappa-derived key in pack — forward-only violated")

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"modelA_pack_{mock}.npz")
    np.savez(npz_path, **pack)

    m = MOCKS[mock]
    prov = dict(
        schema="modelA_pack_schema.md v1",
        mock=mock,
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        code_commit=_git_commit(),
        command=" ".join(sys.argv),
        inputs=dict(
            catalog_dir=m["catalog_dir"], truth=m["truth_path"],
            bal_cat=m["bal_cat_path"], mockdir=m["mockdir"],
            molly_tsv=FF.DEF_MOLLY_TSV,
            molly_counts_cache=frozen["molly_prov"]["path"],
            forward_model=frozen["fwd_meta"]["path"],
            loa0_product=frozen["fp_prov"]["product"],
            loa0_raw=frozen["fp_prov"]["loa0_out"],
            t_sigma_artifacts=[f"CDDF_analysis/hbi/ff_fp_{x}.json"
                               for x in ("2lpt0", "saclay0", "london0")],
        ),
        routines=dict(
            op_mask=("CDDF_analysis/hbi/cddf_catalog_hbi.py:load_and_cut_catalog "
                     "(:521) + ab_loa0_fp_baseline.py:146-148 op_mask "
                     "(S2N_RED>2 strict & P_DLA>0.99 strict & DLAFLAG==0; BAL veto "
                     "+ z_qso in (2,4.25) + lam_rf [1025,1216] inside the bundle) "
                     "+ track_c_tf_saclay.py:_snap_off_molly_edges (:138)"),
            pathlength=("CDDF_analysis/hbi/cddf_catalog_hbi.py:build_pathlength "
                        "(:823, return_per_sl) + build_M_b PX (:3730/3771-3784)"),
            molly=("CDDF_analysis/hbi/ff_fp_estimator.py:build_molly_counts_cache "
                   "(:319) / load_molly_counts (matched-real numerator; RhoGuard)"),
            g=("CDDF_analysis/hbi/cddf_catalog_hbi.py:build_cnz_resolved (:3869) + "
               "znz_kernel.py:measure_c_nz (:836) occupancy — frozen on 2LPT-0"),
            forward=("track_c_tf_loa.py:_DEF_FORWARD (:99); envelope "
                     "znz_kernel.py:save_forward_response (:2073); skew ramp = "
                     "linear over 0.5 dex above N_skew_collapse "
                     "(znz_kernel.py:1362) -> resp_skew_ramp=[collapse, 0.5]"),
            fp=("cddf_catalog_hbi.py:Loa0FP.from_product (:1059) scalars + "
                "build_loa0_fp_product.py:load_loa0_fp_catalog/:build_product op "
                "cut (:195-241) re-binned to (c=29,s=8); EXACT n_fp_molly guard"),
            t_sigma=("ff_fp_{mock}.json strata[z].closure.subdla_195_203.R_point; "
                     "t_sigma[K]=max(0.10, max_mock |ln(R_z/R_z_2lpt0)|)"),
            truth=("load_and_cut_catalog truth_cut (same window) + SNR>2 strict, "
                   "half-open bins (calccddf_vs_hbi.py:truth_one_file :218 "
                   "convention)"),
        ),
        op_mask=dict(
            n_cat_cut=int(len(cat)),
            n_op_total=int(op.sum()),
            n_op_in_window=int(n_in_window),
            snr_min=2.0, p_dla_min=0.99, strict=True,
            window=dict(nhat=[19.5, 22.4], zhat=[2.0, 3.5]),
        ),
        pathlength=dict(
            n_sl=int(bundle["n_sl"]),
            X_tot_coarse=[float(x) for x in x_tot],
            dX_total=float(dX.sum()),
        ),
        fp=dict(**frozen["fp_prov"], n_sl_mock=nsm,
                fp_w_sightline_ratio=fp_w, fp_ell_eff=fp_ell),
        molly_counts=frozen["molly_prov"],
        forward=frozen["fwd_meta"],
        resp_z_covariate=frozen["fwd_meta"]["z_covariate"],
        t_sigma=dict(values=[float(x) for x in frozen["t_sigma"]],
                     floor=T_SIGMA_FLOOR, detail=frozen["t_sigma_detail"]),
        truth=dict(n_truth_cut=int(len(bundle["truth_cut"])),
                   n_truth_in_window=int(n_truth_in_window),
                   truth_nhi_floor=19.5),
        g_available=bool(frozen["g_available"]),
        checks=dict(
            counts_total_equals_op_in_window=True,
            dx_marginal_max_relgap=dx_gap,
        ),
        guards=dict(
            forward_kernel_assert=True,
            no_kappa_keys=True,
            no_rho=True,                    # purity never read (RhoGuard cache)
            privacy_mock_only=True,
            fp_molly_rebin_exact=True,
        ),
        cut_meta={k: (int(v) if isinstance(v, (int, np.integer)) else
                      (float(v) if isinstance(v, (float, np.floating)) else v))
                  for k, v in bundle["meta"].items()},
    )
    prov_path = npz_path[:-4] + ".provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=1)

    # round-trip verification
    rt = np.load(npz_path, allow_pickle=False)
    assert int(rt["counts"].sum()) == n_in_window
    assert rt["counts"].shape == (N_C, N_K, N_S)

    print(f"[pack] {mock}: counts={int(counts.sum())} (op {int(op.sum())}, "
          f"window {n_in_window}), dX_tot={dX.sum():.1f}, dx_gap={dx_gap:.2e}, "
          f"truth_in_window={n_truth_in_window} -> {npz_path} "
          f"({time.time()-t0:.0f}s)")
    return dict(npz=npz_path, provenance=prov_path, counts_total=int(counts.sum()),
                dx_gap=dx_gap, bundle=bundle)


def build_frozen_calibration(out_dir: str) -> dict:
    """Build the frozen 2LPT-0 calibration blocks ONCE (shared by every pack).
    Returns the dict extract_pack consumes; stashes the 2LPT-0 bundle so the
    2lpt0 pack does not reload it."""
    print("[frozen] building 2LPT-0 calibration blocks ...")
    fwd, fwd_meta = load_forward_response_pack()
    t_sigma, t_detail = compute_t_sigma()
    molly, molly_prov = load_molly_counts_block()
    fp_counts, _loa0, fp_prov = build_fp_block()
    bundle0 = load_mock_bundle("2lpt0", out_dir)
    g_available = True
    try:
        g_grid, g_occ = build_g_block(bundle0)
    except Exception as e:          # committed builder needs heavy missing inputs
        print(f"[frozen] g(N,z) builder unavailable ({e}); emitting g=1")
        n_nhi = len(molly["molly_nhi_edges"]) - 1
        g_grid = np.ones((n_nhi, N_K))
        g_occ = np.zeros((n_nhi, N_K))
        g_available = False
    return dict(fwd=fwd, fwd_meta=fwd_meta, t_sigma=t_sigma,
                t_sigma_detail=t_detail, molly=molly, molly_prov=molly_prov,
                fp_counts=fp_counts, fp_prov=fp_prov,
                g_grid=g_grid, g_occupancy=g_occ, g_available=g_available,
                _bundle_2lpt0=bundle0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mocks", nargs="+", default=["2lpt0"],
                   choices=list(MOCKS))
    p.add_argument("--out-dir", default=DEF_OUT_DIR)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    frozen = build_frozen_calibration(args.out_dir)
    results = {}
    for mock in args.mocks:
        results[mock] = {k: v for k, v in
                         extract_pack(mock, args.out_dir, frozen).items()
                         if k != "bundle"}
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
