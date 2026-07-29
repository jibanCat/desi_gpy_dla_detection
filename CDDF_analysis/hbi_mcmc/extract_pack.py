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

SCHEMA v1.1 — the DOWNWARD BASIS PAD (finding D1, 2026-07-28/29)
----------------------------------------------------------------
``--basis-pad-floor FLOOR`` extends the TRUE-N basis (``ntrue_edges``) DOWN to
FLOOR on the same 0.1 dex step. The DETECTION / REPORTING window
(``nhat_edges``) does NOT move, so ``counts`` is bit-identical across the whole
pad ladder (MEASURED: 88071 / 87840 / 86763 on 2lpt0 / london0 / saclay0 at
every floor). Default = no pad = schema v1, bit-for-bit.

WHY: the forward response has a +0.2937 dex measured mean bias and a 0.276 dex
width at N = 19.503, so the lowest observed n-hat bins are fed overwhelmingly by
TRUE systems BELOW the reporting floor — MEASURED here, 83% of the predicted
counts in [19.5, 19.6) come from below 19.5. A basis truncated at the reporting
floor cannot carry them (the same one-sided-support class as B16), and the
counting argument settles it: 88071 observed > 73610 in-window truth on 2LPT-0
with completeness <= 1 and kernel row mass <= 1.

The truth histogram under a pad is built from a SEPARATE truth-only cut at the
pad floor (``load_truth_bundle``) and GUARDED to reproduce the unpadded
histogram exactly over the reporting window — lowering ``truth_nhi_floor``
perturbs a handful of ``cat_cut`` rows through the truth-z lambda cut, and the
detection side must not move.

TWO UNDECIDED CONVENTIONS the pad exposes (both BRACKETED, neither chosen):
  * the response below ~19.35 was never measured (``resp_N_fit_range`` spans
    19.336-21.503 to 21.05-21.216); ``resp_clamp='both'`` freezes it, ``'hi'``
    extrapolates it. They are identical unpadded and differ by up to 15% on the
    total under a pad.
  * the completeness below 19.5: ``--completeness-below-floor const_extrap``
    (constant extrapolation of molly cell 0, KNOWN TOO HIGH) vs ``molly172``
    (the measured sub-floor cells of the same production run's floor-17.2
    lya_only matrix, spliced under bit-identical >= 19.5 cells). ~8% on the
    total.

CLI (script form, NOT `-m`: the hbi_mcmc package __init__ imports jax, which the
`gpdla` data-plane env deliberately does not carry — this module needs no jax):
  conda run -n gpdla python CDDF_analysis/hbi_mcmc/extract_pack.py \
      --mocks 2lpt0 [saclay0 london0] \
      [--basis-pad-floor 18.0] [--completeness-below-floor molly172] \
      [--tag _pad18p0] \
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

# --- schema-v1.1 BASIS PAD (finding D1, 2026-07-28) -------------------------
# The TRUE-N basis may extend DOWN below the reporting floor; the OBSERVED
# (n-hat) grid never moves. See pack.validate_pack's tail-subset rule.
N_STEP = 0.1

# completeness convention BELOW the reporting floor (the pad's leading
# systematic; see `build_molly_counts_block`).
COMPLETENESS_CONVENTIONS = ("const_extrap", "molly172")
# the 2LPT-0 production molly matrix extracted down to the injector floor
# (lya_only, same production run + same P_DLA>0.99 cut as the canonical
# nhi195 matrix; molly_summary title "2lpt0_v1 floor17.2 lya_only").
MOLLY_TSV_NHI172 = ("/scratch/cavestru_root/cavestru0/mfho/"
                    "gl_prod_2lpt0_v1_20260526/figures_molly_nhi172/"
                    "molly_matrix.tsv")


def basis_pad_edges(pad_floor=None):
    """(ntrue_edges, n_pad_bins) for a true-N basis padded DOWN to ``pad_floor``.

    ``pad_floor=None`` (or >= the reporting floor) reproduces schema v1
    EXACTLY: ``ntrue_edges == NHAT_EDGES`` and ``n_pad_bins == 0``. The pad only
    ever goes DOWN — ``nhat_edges`` (the detection/reporting window) never
    moves, and stays an exact TAIL subset of ``ntrue_edges``.
    """
    if pad_floor is None:
        return NHAT_EDGES.copy(), 0
    pad_floor = float(pad_floor)
    n_pad = int(round((NHAT_EDGES[0] - pad_floor) / N_STEP))
    if n_pad <= 0:
        return NHAT_EDGES.copy(), 0
    if abs((NHAT_EDGES[0] - n_pad * N_STEP) - pad_floor) > 1e-8:
        raise ValueError(
            f"--basis-pad-floor {pad_floor} is not on the {N_STEP} dex grid "
            f"anchored at the reporting floor {NHAT_EDGES[0]}")
    lo = np.round(NHAT_EDGES[0] - N_STEP * np.arange(n_pad, 0, -1), 3)
    return np.round(np.concatenate([lo, NHAT_EDGES]), 3), n_pad

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


_PACK_MOD = None


def _pack_module():
    """Load ``pack.py`` file-directly (never through the hbi_mcmc package
    __init__, which imports jax — absent from the `gpdla` data-plane env)."""
    global _PACK_MOD
    if _PACK_MOD is None:
        import importlib.util
        p = os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "pack.py")
        spec = importlib.util.spec_from_file_location("_modelA_pack_nojax", p)
        m = importlib.util.module_from_spec(spec)
        # dataclasses resolves cls.__module__ through sys.modules
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        _PACK_MOD = m
    return _PACK_MOD


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
    # schema v1.1 (finding D2): the CALIBRATED covariate range of the per-cell
    # moment polynomials = the per-cell min/max of the empirical true-N anchors
    # they were weighted-least-squares fit at. Emitted NATIVELY here so padded
    # packs need no upgrade_pack_v11 round trip. Derived through the committed
    # routine pack.resp_fit_range_from_forward_npz (loaded file-directly: the
    # hbi_mcmc package __init__ imports jax, which the `gpdla` data-plane env
    # deliberately does not carry).
    fit_range = None
    if "emp_N_anchors" in d.files:
        fit_range = _pack_module().resp_fit_range_from_forward_npz(path)
        fwd["resp_N_fit_range"] = np.asarray(fit_range, float)
    meta = dict(
        fwd_response_kind=kind,
        deg_N=int(d["deg_N"]),
        resp_N_fit_range=(np.asarray(fit_range, float).tolist()
                          if fit_range is not None else None),
        resp_N_fit_range_source=("emp_N_anchors min/max per response cell "
                                 "(pack.py:resp_fit_range_from_forward_npz)"
                                 if fit_range is not None else
                                 "ABSENT — no emp_N_anchors in the frozen NPZ"),
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


def load_molly_counts_block(convention="const_extrap", counts172_path=None):
    """Molly completeness counts (matched-real numerator) via the committed
    ff_fp_estimator cache; build it if absent. Purity is never read (RhoGuard).

    ``convention`` selects what the completeness does BELOW the reporting floor
    (19.5) — the leading known systematic on the schema-v1.1 basis pad:

      "const_extrap" (default, schema v1 behaviour): the pack carries ONLY the
          canonical nhi195 cells, so ``forward.build_consts``'s
          ``clip(digitize(Nc, molly_nhi_edges) - 1, 0, M-2)`` makes every
          sub-floor true-N bin read cell 0 = [19.5, 20.0). That is a CONSTANT
          EXTRAPOLATION and it is KNOWN TOO HIGH (the measured completeness
          keeps falling below 19.5).
      "molly172": SPLICE the measured sub-floor cells of the SAME production
          run's floor-17.2 lya_only matrix underneath the canonical cells. The
          cells at and above 19.5 are left BIT-IDENTICAL to the canonical
          matrix, so the convention changes NOTHING inside the reporting
          window — it only replaces the constant extrapolation with the
          measured sub-floor completeness.

    Returns (block, prov, mm_alt) — ``mm_alt`` is the loaded floor-17.2
    MollyMatrix (or None) so the g(N,z) block can be spliced on the same cells.
    """
    if convention not in COMPLETENESS_CONVENTIONS:
        raise ValueError(f"completeness convention must be one of "
                         f"{COMPLETENESS_CONVENTIONS}, got {convention!r}")
    mc = FF.load_molly_counts()
    if mc is None:
        FF.build_molly_counts_cache()
        mc = FF.load_molly_counts()
    n_det = np.asarray(mc["cmp_nfound"], float)            # (s=8, nhi_cells)
    n_tot = np.asarray(mc["cmp_nfid"], float)
    nhi_edges = np.asarray(mc["nhi_edges"], float)
    prov = dict(path=mc["path"], max_c_diff=float(mc["max_c_diff"]),
                convention=convention,
                below_floor=("constant extrapolation of molly cell 0 "
                             "([19.5,20.0)) — forward.build_consts clips "
                             "b_to_cell to 0"))
    if convention == "const_extrap":
        return dict(molly_n_det=n_det, molly_n_tot=n_tot,
                    molly_nhi_edges=nhi_edges,
                    molly_snr_edges=np.asarray(mc["snr_edges"], float)), prov, None

    # --- molly172: splice the MEASURED sub-floor cells underneath ------------
    assert_mock_only(MOLLY_TSV_NHI172)
    counts172_path = counts172_path or os.path.join(
        os.path.dirname(mc["path"]), "molly_counts_nhi172.npz")
    if not os.path.exists(counts172_path):
        FF.build_molly_counts_cache(out_path=counts172_path,
                                    molly_tsv=MOLLY_TSV_NHI172)
    d = np.load(counts172_path, allow_pickle=True)
    e172 = np.asarray(d["nhi_edges"], float)
    if not np.allclose(np.asarray(d["snr_edges"], float),
                       np.asarray(mc["snr_edges"], float)):
        raise RuntimeError("molly172 splice: SNR strata differ from the "
                           "canonical matrix")
    # the canonical edges must be an exact TAIL subset of the floor-17.2 edges
    tail = e172[len(e172) - len(nhi_edges):]
    if not (np.allclose(tail[:-1], nhi_edges[:-1], atol=1e-8)
            and np.isposinf(tail[-1]) and np.isposinf(nhi_edges[-1])):
        raise RuntimeError(
            f"molly172 splice: canonical nhi edges {nhi_edges} are not a tail "
            f"subset of the floor-17.2 edges {e172}")
    n_sub = len(e172) - len(nhi_edges)                      # cells strictly < 19.5
    det = np.concatenate([np.asarray(d["cmp_nfound"], float)[:, :n_sub], n_det], 1)
    tot = np.concatenate([np.asarray(d["cmp_nfid"], float)[:, :n_sub], n_tot], 1)
    prov.update(
        path_below_floor=counts172_path, tsv_below_floor=MOLLY_TSV_NHI172,
        max_c_diff_below_floor=float(d["max_c_diff"]),
        n_cells_spliced_below_floor=int(n_sub),
        below_floor=("MEASURED sub-floor cells from the same production run's "
                     "floor-17.2 lya_only matrix; cells >= 19.5 are "
                     "bit-identical to the canonical nhi195 matrix"))
    from CDDF_analysis.hbi.cddf_catalog_hbi import load_molly_matrix as _lmm
    return dict(molly_n_det=det, molly_n_tot=tot, molly_nhi_edges=e172,
                molly_snr_edges=np.asarray(mc["snr_edges"], float)), \
        prov, _lmm(MOLLY_TSV_NHI172)


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
def _make_cfg(mock: str, out_dir: str, molly_tsv: str = None) -> HBIConfig:
    m = MOCKS[mock]
    for p in m.values():
        assert_mock_only(p)
    molly_tsv = molly_tsv or FF.DEF_MOLLY_TSV       # frozen 2LPT-0 lya_only-nhi195
    assert_mock_only(molly_tsv)
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


def load_mock_bundle(mock: str, out_dir: str, molly_tsv: str = None):
    """Load one mock's cut bundle + op mask + per-sightline pathlength through
    the committed machinery (ab_loa0_fp_baseline.build_ingredients path)."""
    cfg = _make_cfg(mock, out_dir, molly_tsv=molly_tsv)
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


def load_truth_bundle(mock: str, out_dir: str, truth_floor: float):
    """TRUTH-ONLY cut bundle at a LOWER NHI floor (schema-v1.1 basis pad).

    The detection side is deliberately NOT taken from here. Lowering
    ``truth_nhi_floor`` changes the PRIMARY truth match, hence ``Z_TRUE``, hence
    (through ``make_lambda_z_BAL_cuts(use_truth_z=True)``) a handful of rows in
    ``cat_cut`` — MEASURED on 2LPT-0: n_cat_cut 582855 -> 582078 and the binned
    detection total 88071 -> 88053 going from floor 19.5 to 17.2. The observed
    counts must be IDENTICAL across the whole pad ladder or the comparison is
    not like-for-like, so this loader is used for ``truth_cut`` ONLY and the
    caller GUARDS that its >= reporting-floor histogram reproduces the
    unpadded one exactly.
    """
    cfg = _make_cfg(mock, out_dir)
    mm = load_molly_matrix(cfg.molly_tsv)
    t0 = time.time()
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, _is_TP, _good, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=float(truth_floor), qso_lookup=qso_lookup,
        host_truth_floor=min(19.0, float(truth_floor)))
    TS._snap_off_molly_edges(cat_cut, truth_cut, mm)
    print(f"  [{mock}] truth-pad bundle @floor {truth_floor}: "
          f"n_truth_cut={len(truth_cut)} ({time.time()-t0:.0f}s)")
    return dict(cfg=cfg, truth_cut=truth_cut, meta=meta)


def build_truth_counts(bundle, ntrue_edges=None):
    """truth_counts (b, k) [+ (b,k,s)] — truth systems from the SAME cut
    bundle (identical windowing), SNR>2 strict (truth_one_file convention),
    half-open binning on the schema grids.

    ``ntrue_edges`` defaults to ``NHAT_EDGES`` (schema v1, b == c). Pass a
    DOWNWARD-padded grid (``basis_pad_edges``) for schema v1.1; the z / SNR
    axes and the SNR>2 cut are untouched, so the >= reporting-floor rows of the
    padded histogram are bit-identical to the unpadded one.
    """
    ntrue_edges = NHAT_EDGES if ntrue_edges is None else np.asarray(ntrue_edges, float)
    n_b = len(ntrue_edges) - 1
    t = bundle["truth_cut"]
    cfg = bundle["cfg"]
    n = np.asarray(t["NHI"], float)
    z = np.asarray(t["Z_DLA"], float)
    s2n = np.asarray(t["S2N_RED"], float)
    keep = s2n > cfg.snr_min
    b = _idx(ntrue_edges, n[keep])
    k = _idx(ZF_EDGES, z[keep])
    s = np.clip(_idx(SNR_EDGES, s2n[keep]), 0, N_S - 1)
    ok = (b >= 0) & (b < n_b) & (k >= 0) & (k < N_K)
    tc_bks = np.zeros((n_b, N_K, N_S), float)
    np.add.at(tc_bks, (b[ok], k[ok], s[ok]), 1.0)
    return tc_bks.sum(axis=2), tc_bks, int(ok.sum())


# ---------------------------------------------------------------------------
# pack assembly
# ---------------------------------------------------------------------------
def extract_pack(mock: str, out_dir: str, frozen: dict, pad_floor=None,
                 tag: str = "") -> dict:
    """Build + save one mock's data pack. `frozen` carries the 2LPT-0 calibration
    blocks (forward / molly counts / g / fp / t_sigma) built once.

    ``pad_floor`` (schema v1.1, finding D1): extend the TRUE-N basis DOWN to
    this log N_HI on the same 0.1 dex step. ``None`` = schema v1, bit-for-bit.
    The observed n-hat grid, ``counts``, ``dX`` and every calibration block are
    UNCHANGED by the pad — only ``ntrue_edges`` and the truth histogram grow.
    """
    print(f"[pack] extracting {mock} (pad_floor={pad_floor}) ...")
    t0 = time.time()
    # DETECTION-side bundle cache: identical for every pad floor by
    # construction (the pad touches the truth axis only), so a ladder sweep
    # reuses it instead of re-cutting the catalog.
    bundles = frozen.setdefault("_bundles", {})
    bundle = bundles.get(mock)
    if bundle is None:
        bundle = load_mock_bundle(mock, out_dir)
        bundles[mock] = bundle

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

    # --- truth histogram, optionally on a DOWNWARD-padded true-N basis (D1) ---
    ntrue_edges, n_pad_bins = basis_pad_edges(pad_floor)
    truth_counts, truth_counts_bks, n_truth_in_window = build_truth_counts(bundle)
    pad_prov = dict(n_pad_bins=int(n_pad_bins), pad_floor=None,
                    truth_nhi_floor_used=19.5)
    if n_pad_bins > 0:
        # TRUTH-side bundle cache: one cut per (mock, floor). The DEEPEST floor
        # already carries every shallower pad's rows, but the cut is re-run per
        # floor so each pack's provenance names the floor it was actually cut at.
        tkey = (mock, round(float(ntrue_edges[0]), 3))
        tcache = frozen.setdefault("_truth_bundles", {})
        tb = tcache.get(tkey)
        if tb is None:
            tb = load_truth_bundle(mock, out_dir, float(ntrue_edges[0]))
            tcache[tkey] = tb
        tc_pad, tc_pad_bks, n_pad_in_window = build_truth_counts(tb, ntrue_edges)
        # GUARD (like-for-like): the padded truth histogram must reproduce the
        # UNPADDED one EXACTLY over the reporting window. Anything else means
        # the lower truth floor perturbed the >= 19.5 truth rows and the ladder
        # would be comparing different data.
        if not np.array_equal(tc_pad[n_pad_bins:], truth_counts):
            raise RuntimeError(
                "basis-pad GUARD failed: the padded truth histogram does not "
                "reproduce the unpadded histogram over the reporting window "
                f"(max |diff| = {np.abs(tc_pad[n_pad_bins:] - truth_counts).max()})")
        if not np.array_equal(tc_pad_bks[n_pad_bins:], truth_counts_bks):
            raise RuntimeError(
                "basis-pad GUARD failed: (b,k,s) padded truth != unpadded over "
                "the reporting window")
        pad_prov.update(
            pad_floor=float(ntrue_edges[0]),
            truth_nhi_floor_used=float(ntrue_edges[0]),
            n_truth_in_window_unpadded=int(n_truth_in_window),
            n_truth_in_window_padded=int(n_pad_in_window),
            n_truth_below_reporting_floor=int(tc_pad[:n_pad_bins].sum()),
            tail_guard="EXACT match to the unpadded truth over [19.5, 22.4)",
            truth_bundle_note=(
                "truth_cut ONLY; the detection side stays on the 19.5-floor "
                "bundle so `counts` is identical across the pad ladder"))
        truth_counts, truth_counts_bks = tc_pad, tc_pad_bks
        n_truth_in_window = n_pad_in_window

    # per-mock loa-0 exposure scalars (Loa0FP.from_product n_sl_prod semantics)
    ns0 = float(frozen["fp_prov"]["n_sl_loa0"])
    nsm = float(bundle["n_sl"])
    fp_w = nsm / ns0                                # vol_scale = N_prod/N_sl_loa0
    fp_ell = ns0 * (ns0 / nsm)                      # ell_eff at this mock's N_prod

    masked = np.zeros(N_C, dtype=bool)
    masked[(NHAT_EDGES[:-1] >= 19.5 - 1e-9) & (NHAT_EDGES[1:] <= 19.7 + 1e-9)] = True

    pack = dict(
        # axes
        nhat_edges=NHAT_EDGES, ntrue_edges=ntrue_edges,
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
    npz_path = os.path.join(out_dir, f"modelA_pack_{mock}{tag}.npz")
    np.savez(npz_path, **pack)

    m = MOCKS[mock]
    prov = dict(
        schema=("modelA_pack_schema.md v1.1 (+resp_N_fit_range"
                + (", +basis pad)" if n_pad_bins else ")")),
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
                   truth_nhi_floor=pad_prov["truth_nhi_floor_used"]),
        basis_pad=dict(
            **pad_prov,
            ntrue_edges=[float(x) for x in ntrue_edges],
            nhat_edges_unchanged=True,
            routine="CDDF_analysis/hbi_mcmc/extract_pack.py:basis_pad_edges",
            rule=("schema v1.1: ntrue_edges extends DOWN only; nhat_edges is an "
                  "exact TAIL subset (pack.validate_pack enforces it). The "
                  "reporting/detection window never moves.")),
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


def build_frozen_calibration(out_dir: str, completeness="const_extrap") -> dict:
    """Build the frozen 2LPT-0 calibration blocks ONCE (shared by every pack).
    Returns the dict extract_pack consumes; stashes the 2LPT-0 bundle so the
    2lpt0 pack does not reload it.

    ``completeness`` (see ``load_molly_counts_block``) selects the convention
    BELOW the reporting floor. Under "molly172" the g(N,z) surface is spliced
    the SAME way — measured sub-floor rows from a floor-17.2 bundle, the
    >= 19.5 rows left bit-identical to the canonical 2LPT-0 surface — so the
    two conventions differ ONLY below the reporting floor.
    """
    print("[frozen] building 2LPT-0 calibration blocks ...")
    fwd, fwd_meta = load_forward_response_pack()
    t_sigma, t_detail = compute_t_sigma()
    molly, molly_prov, mm_alt = load_molly_counts_block(convention=completeness)
    fp_counts, _loa0, fp_prov = build_fp_block()
    bundle0 = load_mock_bundle("2lpt0", out_dir)
    g_available = True
    try:
        g_grid, g_occ = build_g_block(bundle0)
        if mm_alt is not None:
            bundle_alt = load_mock_bundle("2lpt0", out_dir,
                                          molly_tsv=MOLLY_TSV_NHI172)
            g_alt, occ_alt = build_g_block(bundle_alt)
            n_sub = g_alt.shape[0] - g_grid.shape[0]
            if n_sub <= 0:
                raise RuntimeError("molly172 g splice: alternate grid is not "
                                   "deeper than the canonical one")
            g_grid = np.concatenate([g_alt[:n_sub], g_grid], axis=0)
            g_occ = np.concatenate([occ_alt[:n_sub], g_occ], axis=0)
            molly_prov["g_below_floor"] = (
                f"{n_sub} sub-floor rows from build_cnz_resolved on the "
                "floor-17.2 molly matrix; >= 19.5 rows bit-identical to the "
                "canonical 2LPT-0 surface")
    except Exception as e:          # committed builder needs heavy missing inputs
        print(f"[frozen] g(N,z) builder unavailable ({e}); emitting g=1")
        n_nhi = len(molly["molly_nhi_edges"]) - 1
        g_grid = np.ones((n_nhi, N_K))
        g_occ = np.zeros((n_nhi, N_K))
        g_available = False
    if g_grid.shape[0] != len(molly["molly_nhi_edges"]) - 1:
        raise RuntimeError(
            f"g_grid has {g_grid.shape[0]} cells but the molly grid has "
            f"{len(molly['molly_nhi_edges']) - 1}")
    return dict(fwd=fwd, fwd_meta=fwd_meta, t_sigma=t_sigma,
                t_sigma_detail=t_detail, molly=molly, molly_prov=molly_prov,
                fp_counts=fp_counts, fp_prov=fp_prov,
                g_grid=g_grid, g_occupancy=g_occ, g_available=g_available,
                completeness_convention=completeness,
                _bundles={"2lpt0": bundle0})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mocks", nargs="+", default=["2lpt0"],
                   choices=list(MOCKS))
    p.add_argument("--out-dir", default=DEF_OUT_DIR)
    p.add_argument("--basis-pad-floor", type=float, default=None,
                   help="schema v1.1 (finding D1): extend the TRUE-N basis DOWN "
                        "to this log N_HI on the same 0.1 dex step. The "
                        "detection/reporting window (nhat_edges) does NOT move. "
                        "Default: no pad = schema v1, bit-for-bit.")
    p.add_argument("--completeness-below-floor", default="const_extrap",
                   choices=list(COMPLETENESS_CONVENTIONS),
                   help="what the completeness does BELOW 19.5 under a pad: "
                        "'const_extrap' = constant extrapolation of molly cell "
                        "0 (KNOWN TOO HIGH); 'molly172' = the measured "
                        "sub-floor cells of the same production run's "
                        "floor-17.2 matrix.")
    p.add_argument("--tag", default="",
                   help="filename suffix so a ladder of packs can share one "
                        "out-dir (modelA_pack_<mock><tag>.npz)")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    frozen = build_frozen_calibration(
        args.out_dir, completeness=args.completeness_below_floor)
    results = {}
    for mock in args.mocks:
        results[mock] = {k: v for k, v in
                         extract_pack(mock, args.out_dir, frozen,
                                      pad_floor=args.basis_pad_floor,
                                      tag=args.tag).items()
                         if k != "bundle"}
    print(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
