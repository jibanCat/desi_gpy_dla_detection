"""pack.py — Model A data-pack loader, schema validator, and synthetic generator.

Implements the Q3 Model A pack contract (schema v1, ``modelA_pack_schema.md``):
one NPZ per mock plus a ``.provenance.json`` sidecar. The NumPyro module
consumes ONLY the pack (no fitsio/h5py here or anywhere downstream).

Axes, fixed in this order everywhere (never reordered):
    c : observed-N-hat bins   (edges ``nhat_edges``; real grid 19.5..22.4 step 0.1 -> 29)
    b : true-N bins           (edges ``ntrue_edges`` == nhat grid for v1)
    k : fine z bins           (edges ``zf_edges``;  real grid 2.0..3.5 step 0.1 -> 15)
    K : coarse z bins         (edges ``zc_edges``;  real grid [2.0, 2.5, 3.0, 3.5] -> 3)
    s : SNR strata            (edges ``snr_edges``; real molly cells [0..7, inf] -> 8)

Interpretation notes pinned here (both sides of the contract):
  * ``fp_counts`` and the FP intensity live on (c, s) — the schema's global
    axis-order rule (c before s, as in ``counts`` (c, k, s)) wins over the
    prose order of the ``fp_counts`` line, which lists s first.
  * ``resp_fitcov_diag`` (2, s_resp, z_resp) is a MODULE EXTENSION to schema
    v1: the per-response-cell fit-covariance DIAGONAL (variances) for the
    leading (order-0) mu/sig coefficient terms, required by the model's
    psi_k_delta prior. The synthetic generator always emits it; ``load_pack``
    accepts packs without it (pack.resp_fitcov_diag is then None and the
    model must supply a documented default).
  * Polynomial coefficient convention: LOWEST order first
    (``coef[..., 0]`` = constant term = the "leading term" that psi_k_delta
    perturbs), covariate u = N_true - resp_N_ref, where ``resp_N_ref`` is the
    reference the coefficients were FIT at (carried in the pack; REQUIRED by
    the fold — no midpoint fallback; 2026-07-11 finding F1/F1b). The surfaces
    are MOMENTS with the committed ForwardResponseModel semantics: mean bias,
    sd = clip(poly, resp_sig_floor), moment skewness ramped to ZERO above
    resp_skew_ramp[0] over resp_skew_ramp[1] dex (znz_kernel.py:1362).

Validation is fail-closed: every schema rule (shapes, edges, axis order via
shape cross-checks, dtypes, finiteness, normalization) raises
``PackSchemaError`` with a precise message on mismatch. The REAL grid is
enforced by default; schema-consistent smaller grids (same 0.1-dex/0.1-z
steps, same structural rules) are allowed only via the explicit
``allow_nonstandard_grid=True`` override (tests use this).

PRIVACY: this module never touches survey data. ``synthetic_pack`` builds a
fully schema-conformant pack from a KNOWN synthetic truth (power-law x mild
z-evolution f, logistic completeness, skew-normal response with known
coefficients, falling FP intensity with known per-coarse-z transfer factors
t_K) and draws counts ~ Poisson(mu) with the EXACT Model A forward expression
(via the independent numpy oracle ``forward.fold_mu_reference``).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Optional

import numpy as np

__all__ = [
    "PackSchemaError",
    "ModelAPack",
    "load_pack",
    "save_pack",
    "synthetic_pack",
    "small_test_grid",
    "REAL_NHAT_EDGES",
    "REAL_ZF_EDGES",
    "REAL_ZC_EDGES",
    "REAL_SNR_EDGES",
]


class PackSchemaError(ValueError):
    """A Model A pack violated the schema contract (precise message attached)."""


# --- the real (default, enforced) grids ----------------------------------------
REAL_NHAT_EDGES = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 10)   # 29 bins
REAL_ZF_EDGES = np.round(np.arange(2.0, 3.5 + 1e-9, 0.1), 10)       # 15 bins
REAL_ZC_EDGES = np.array([2.0, 2.5, 3.0, 3.5])                       # 3 bins
REAL_SNR_EDGES = np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf])  # 8 strata

_N_STEP = 0.1
_Z_STEP = 0.1

# NPZ keys in schema order (kz_to_K stored explicitly; see schema "map kz_to_K[k]").
_REQUIRED_KEYS = (
    "nhat_edges", "ntrue_edges", "zf_edges", "zc_edges", "kz_to_K", "snr_edges",
    "counts", "dX",
    "molly_n_det", "molly_n_tot", "molly_nhi_edges",
    "g_grid", "g_occupancy",
    "resp_mu_coef", "resp_sig_coef", "resp_skew_coef",
    "resp_snr_edges", "resp_z_edges", "resp_sig_floor", "resp_skew_ramp",
    "fp_counts", "fp_ell_eff", "fp_w_sightline_ratio", "fp_E_alloc",
    "t_sigma",
    "truth_counts",
)
# schema-v1.1 extensions: emitted by the extractor (2026-07-11); each validated
# below where consumed. resp_N_ref is REQUIRED to evaluate the response coef
# polynomials on real packs (synthetic packs embed their own reference).
_OPTIONAL_KEYS = ("resp_fitcov_diag", "resp_N_ref", "truth_counts_bks",
                  "dX_coarse_committed", "molly_snr_edges", "nhat_masked_bins")


@dataclasses.dataclass(frozen=True)
class ModelAPack:
    """In-memory Model A data pack (schema v1 + the resp_fitcov_diag extension)."""

    # axes / edges
    nhat_edges: np.ndarray        # (C+1,)
    ntrue_edges: np.ndarray       # (B+1,) == nhat_edges for v1
    zf_edges: np.ndarray          # (Kf+1,)
    zc_edges: np.ndarray          # (KK+1,)
    kz_to_K: np.ndarray           # (Kf,) int64
    snr_edges: np.ndarray         # (S+1,), last = inf
    # data plane
    counts: np.ndarray            # (C, Kf, S) int64
    dX: np.ndarray                # (Kf, S)
    # completeness (molly)
    molly_n_det: np.ndarray       # (S, M)
    molly_n_tot: np.ndarray       # (S, M)
    molly_nhi_edges: np.ndarray   # (M+1,)
    # z-shape surface
    g_grid: np.ndarray            # (M, Kf)
    g_occupancy: np.ndarray       # (M, Kf)
    # forward response (skew-normal; NO kappa objects anywhere)
    resp_mu_coef: np.ndarray      # (SR, ZR, D)   lowest-order-first
    resp_sig_coef: np.ndarray     # (SR, ZR, D)
    resp_skew_coef: np.ndarray    # (SR, ZR, D)
    resp_snr_edges: np.ndarray    # (SR+1,), last = inf
    resp_z_edges: np.ndarray      # (ZR+1,)
    resp_sig_floor: float
    resp_skew_ramp: np.ndarray    # (2,) = (ramp_center, ramp_width)
    # loa-0 FP
    fp_counts: np.ndarray         # (C, S) int64
    fp_ell_eff: float
    fp_w_sightline_ratio: float
    fp_E_alloc: np.ndarray        # (Kf, S), sum over k == 1 per s
    t_sigma: np.ndarray           # (KK,)
    # truth closure (mocks only)
    truth_counts: Optional[np.ndarray]  # (B, Kf) int64 or None
    # module extension (see module docstring)
    resp_fitcov_diag: Optional[np.ndarray] = None  # (2, SR, ZR) variances
    # schema-v1.1 extractor extensions (validated where consumed; pass-through)
    resp_N_ref: Optional[float] = None            # reference N for resp coef polys
    truth_counts_bks: Optional[np.ndarray] = None  # (B, Kf, S)
    dX_coarse_committed: Optional[np.ndarray] = None
    molly_snr_edges: Optional[np.ndarray] = None
    nhat_masked_bins: Optional[np.ndarray] = None
    # non-schema carriers (never validated, never saved except provenance)
    provenance: Optional[dict] = None
    truth: Optional[dict] = None  # synthetic ground truth (in-memory only)

    # convenience dims
    @property
    def n_c(self):
        return len(self.nhat_edges) - 1

    @property
    def n_b(self):
        return len(self.ntrue_edges) - 1

    @property
    def n_k(self):
        return len(self.zf_edges) - 1

    @property
    def n_kk(self):
        return len(self.zc_edges) - 1

    @property
    def n_s(self):
        return len(self.snr_edges) - 1

    @property
    def n_molly(self):
        return len(self.molly_nhi_edges) - 1


# --- validation ----------------------------------------------------------------

def _fail(msg):
    raise PackSchemaError(msg)


def _check_finite(name, arr):
    a = np.asarray(arr, dtype=np.float64)
    # inf is legal ONLY as the last SNR-type edge; callers pass those separately.
    if not np.all(np.isfinite(a)):
        _fail(f"{name}: contains non-finite values (schema: every dataset must be finite)")


# The committed calibration objects close these edge arrays with a +inf
# sentinel (extractor allowlist, 2026-07-11); +inf is legal as the FINAL edge
# only, on exactly these arrays.
_INF_LAST_EDGE_OK = frozenset({
    "snr_edges", "molly_snr_edges", "molly_nhi_edges",
    "resp_snr_edges", "resp_z_edges"})


def _check_edges_uniform(name, edges, step, n_min=1):
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or len(e) < n_min + 1:
        _fail(f"{name}: expected 1-D edge array with >= {n_min + 1} entries, got shape {e.shape}")
    body = e
    if name in _INF_LAST_EDGE_OK and len(e) >= 2 and np.isposinf(e[-1]):
        body = e[:-1]
    _check_finite(name, body)
    d = np.diff(e)
    if np.any(d <= 0):
        _fail(f"{name}: edges must be strictly increasing")
    if step is not None and not np.allclose(d, step, atol=1e-8):
        _fail(f"{name}: schema requires uniform {step} steps, got steps {d}")


def _check_shape(name, arr, shape):
    a = np.asarray(arr)
    if a.shape != tuple(shape):
        _fail(f"{name}: expected shape {tuple(shape)}, got {a.shape} (axes are never reordered)")


def _check_int_counts(name, arr):
    a = np.asarray(arr)
    if a.dtype != np.int64:
        _fail(f"{name}: schema requires int64, got dtype {a.dtype}")
    if np.any(a < 0):
        _fail(f"{name}: counts must be non-negative")


def validate_pack(pack: ModelAPack, allow_nonstandard_grid: bool = False) -> None:
    """Validate every schema rule; raise PackSchemaError with a precise message."""
    # -- grids
    _check_edges_uniform("nhat_edges", pack.nhat_edges, _N_STEP)
    _check_edges_uniform("zf_edges", pack.zf_edges, _Z_STEP)
    if not np.array_equal(np.asarray(pack.ntrue_edges, float),
                          np.asarray(pack.nhat_edges, float)):
        _fail("ntrue_edges: v1 schema requires ntrue_edges == nhat_edges "
              "(basis-pad decision deferred to the module config)")
    if not allow_nonstandard_grid:
        for name, got, want in (("nhat_edges", pack.nhat_edges, REAL_NHAT_EDGES),
                                ("zf_edges", pack.zf_edges, REAL_ZF_EDGES),
                                ("zc_edges", pack.zc_edges, REAL_ZC_EDGES)):
            got_arr = np.asarray(got, float)
            if len(got_arr) != len(want) or not np.allclose(got_arr, want, atol=1e-8):
                _fail(f"{name}: does not match the REAL grid (schema default); "
                      f"pass allow_nonstandard_grid=True only for schema-consistent "
                      f"test grids. got {np.asarray(got)}")
        se = np.asarray(pack.snr_edges, float)
        if len(se) != len(REAL_SNR_EDGES) or not (
                np.allclose(se[:-1], REAL_SNR_EDGES[:-1]) and np.isinf(se[-1])):
            _fail(f"snr_edges: does not match the REAL molly strata {REAL_SNR_EDGES}; "
                  f"got {se}")
    # coarse z edges must be a subset of the fine grid with matching endpoints
    zc = np.asarray(pack.zc_edges, float)
    zf = np.asarray(pack.zf_edges, float)
    _check_edges_uniform("zc_edges", zc, None)
    for v in zc:
        if not np.any(np.isclose(zf, v, atol=1e-8)):
            _fail(f"zc_edges: coarse edge {v} is not on the fine grid zf_edges")
    if not (np.isclose(zc[0], zf[0]) and np.isclose(zc[-1], zf[-1])):
        _fail("zc_edges: coarse endpoints must equal the fine-grid endpoints")

    C, B, Kf, KK, S = pack.n_c, pack.n_b, pack.n_k, pack.n_kk, pack.n_s

    # kz_to_K consistency with the coarse edges
    kz = np.asarray(pack.kz_to_K)
    _check_shape("kz_to_K", kz, (Kf,))
    if kz.dtype != np.int64:
        _fail(f"kz_to_K: expected int64, got {kz.dtype}")
    zf_centers = 0.5 * (zf[:-1] + zf[1:])
    expect = np.digitize(zf_centers, zc) - 1
    expect = np.clip(expect, 0, KK - 1)
    if not np.array_equal(kz, expect):
        _fail(f"kz_to_K: inconsistent with zc_edges (expected {expect}, got {kz})")

    # SNR strata
    se = np.asarray(pack.snr_edges, float)
    if se.ndim != 1 or len(se) < 2:
        _fail(f"snr_edges: expected 1-D with >= 2 entries, got shape {se.shape}")
    if not np.isinf(se[-1]):
        _fail("snr_edges: last edge must be inf (open-ended top stratum)")
    if np.any(np.diff(se[:-1]) <= 0) or not np.all(np.isfinite(se[:-1])):
        _fail("snr_edges: interior edges must be finite and strictly increasing")

    # -- data plane
    _check_shape("counts", pack.counts, (C, Kf, S))
    _check_int_counts("counts", pack.counts)
    _check_shape("dX", pack.dX, (Kf, S))
    _check_finite("dX", pack.dX)
    dx = np.asarray(pack.dX)
    if np.any(dx < 0):
        _fail("dX: pathlengths must be non-negative")
    # dX == 0 is legal ONLY for genuinely-unsearched (k,s) strata: nothing
    # searched => nothing detected and no exposure allocation. Real packs have
    # empty low-SNR strata; such cells are masked out of the likelihood.
    zero = dx == 0
    if np.any(zero):
        if np.any(np.asarray(pack.counts).sum(axis=0)[zero] > 0):
            _fail("dX: zero-pathlength strata with nonzero counts — inconsistent pack")
        if np.any(np.abs(np.asarray(pack.fp_E_alloc)[zero]) > 0):
            _fail("dX: zero-pathlength strata with nonzero fp_E_alloc")

    # -- molly completeness
    M = pack.n_molly
    _check_edges_uniform("molly_nhi_edges", pack.molly_nhi_edges, None)
    ne = np.asarray(pack.ntrue_edges, float)
    me = np.asarray(pack.molly_nhi_edges, float)
    if me[0] > ne[0] + 1e-8 or me[-1] < ne[-1] - 1e-8:
        _fail(f"molly_nhi_edges: must cover the true-N range [{ne[0]}, {ne[-1]}], "
              f"got [{me[0]}, {me[-1]}]")
    _check_shape("molly_n_det", pack.molly_n_det, (S, M))
    _check_shape("molly_n_tot", pack.molly_n_tot, (S, M))
    _check_finite("molly_n_det", pack.molly_n_det)
    _check_finite("molly_n_tot", pack.molly_n_tot)
    if np.any(np.asarray(pack.molly_n_tot) <= 0):
        _fail("molly_n_tot: every completeness cell needs n_tot > 0")
    if np.any(np.asarray(pack.molly_n_det) > np.asarray(pack.molly_n_tot)):
        _fail("molly counts: n_det > n_tot in at least one cell")
    if np.any(np.asarray(pack.molly_n_det) < 0):
        _fail("molly_n_det: negative counts")

    # -- z-shape surface
    _check_shape("g_grid", pack.g_grid, (M, Kf))
    _check_shape("g_occupancy", pack.g_occupancy, (M, Kf))
    _check_finite("g_grid", pack.g_grid)
    _check_finite("g_occupancy", pack.g_occupancy)
    if np.any(np.asarray(pack.g_grid) <= 0):
        _fail("g_grid: the level-preserving z-shape must be strictly positive")

    # -- forward response
    mu_c = np.asarray(pack.resp_mu_coef, float)
    if mu_c.ndim != 3:
        _fail(f"resp_mu_coef: expected 3-D (s_resp, z_resp, deg+1), got shape {mu_c.shape}")
    SR, ZR, D = mu_c.shape
    for name, arr in (("resp_mu_coef", pack.resp_mu_coef),
                      ("resp_sig_coef", pack.resp_sig_coef),
                      ("resp_skew_coef", pack.resp_skew_coef)):
        _check_shape(name, arr, (SR, ZR, D))
        _check_finite(name, arr)
    rse = np.asarray(pack.resp_snr_edges, float)
    _check_shape("resp_snr_edges", rse, (SR + 1,))
    if not np.isinf(rse[-1]):
        _fail("resp_snr_edges: last edge must be inf")
    if np.any(np.diff(rse[:-1]) <= 0):
        _fail("resp_snr_edges: interior edges must be strictly increasing")
    # coverage is required only for strata that carry exposure: the op-mask
    # (S2N_RED > 2 strict) makes the sub-2 strata structurally empty (dX == 0,
    # counts == 0 — enforced above), and the committed response model is
    # calibrated for SNR > 2 only.
    dx_s = np.asarray(pack.dX).sum(axis=0)
    populated = np.where(dx_s > 0)[0]
    lo_needed = se[populated[0]] if len(populated) else se[0]
    if rse[0] > lo_needed + 1e-8:
        _fail("resp_snr_edges: must cover all POPULATED (dX > 0) SNR strata from below")
    rze = np.asarray(pack.resp_z_edges, float)
    _check_shape("resp_z_edges", rze, (ZR + 1,))
    _check_edges_uniform("resp_z_edges", rze, None)
    if rze[0] > zc[0] + 1e-8 or rze[-1] < zc[-1] - 1e-8:
        _fail(f"resp_z_edges: must cover the coarse-z range [{zc[0]}, {zc[-1]}]")
    sig_floor = float(np.asarray(pack.resp_sig_floor))
    if not (np.isfinite(sig_floor) and sig_floor > 0):
        _fail(f"resp_sig_floor: must be a finite positive scalar, got {sig_floor}")
    ramp = np.asarray(pack.resp_skew_ramp, float)
    _check_shape("resp_skew_ramp", ramp, (2,))
    _check_finite("resp_skew_ramp", ramp)
    if ramp[1] <= 0:
        _fail("resp_skew_ramp: ramp width (element 1) must be positive")
    if pack.resp_fitcov_diag is not None:
        _check_shape("resp_fitcov_diag", pack.resp_fitcov_diag, (2, SR, ZR))
        _check_finite("resp_fitcov_diag", pack.resp_fitcov_diag)
        if np.any(np.asarray(pack.resp_fitcov_diag) < 0):
            _fail("resp_fitcov_diag: variances must be non-negative")
    if pack.resp_N_ref is not None:
        nr = float(np.asarray(pack.resp_N_ref))
        if not np.isfinite(nr):
            _fail(f"resp_N_ref: must be a finite scalar, got {nr}")
    # every POPULATED stratum / coarse-z bin must map into a response cell
    # (structurally empty sub-op-mask strata are excluded from the likelihood).
    s_lo = se[:-1][populated] if len(populated) else se[:-1]
    if np.any(np.digitize(s_lo + 1e-9, rse) - 1 < 0) or \
            np.any(np.digitize(s_lo + 1e-9, rse) - 1 >= SR):
        _fail("resp_snr_edges: some populated SNR stratum does not map into any response cell")
    zc_centers = 0.5 * (zc[:-1] + zc[1:])
    zr_idx = np.digitize(zc_centers, rze) - 1
    if np.any(zr_idx < 0) or np.any(zr_idx >= ZR):
        _fail("resp_z_edges: some coarse-z bin does not map into any response cell")

    # -- FP
    _check_shape("fp_counts", pack.fp_counts, (C, S))
    _check_int_counts("fp_counts", pack.fp_counts)
    for name, v in (("fp_ell_eff", pack.fp_ell_eff),
                    ("fp_w_sightline_ratio", pack.fp_w_sightline_ratio)):
        vv = float(np.asarray(v))
        if not (np.isfinite(vv) and vv > 0):
            _fail(f"{name}: must be a finite positive scalar, got {vv}")
    _check_shape("fp_E_alloc", pack.fp_E_alloc, (Kf, S))
    _check_finite("fp_E_alloc", pack.fp_E_alloc)
    E = np.asarray(pack.fp_E_alloc, float)
    if np.any(E < 0):
        _fail("fp_E_alloc: allocation weights must be non-negative")
    colsum = E.sum(axis=0)
    # zero-dX (structurally empty) strata carry colsum == 0 exactly; populated
    # strata must normalize to 1.
    dx_pop = np.asarray(pack.dX).sum(axis=0) > 0
    if np.any(np.abs(colsum[~dx_pop]) > 0):
        _fail("fp_E_alloc: empty strata must have zero allocation")
    if not np.allclose(colsum[dx_pop], 1.0, atol=1e-8):
        _fail(f"fp_E_alloc: schema requires sum_k E[k,s] == 1 per populated stratum, got {colsum}")
    _check_shape("t_sigma", pack.t_sigma, (KK,))
    _check_finite("t_sigma", pack.t_sigma)
    if np.any(np.asarray(pack.t_sigma) <= 0):
        _fail("t_sigma: prior widths must be strictly positive")

    # -- truth closure
    if pack.truth_counts is not None:
        _check_shape("truth_counts", pack.truth_counts, (B, Kf))
        # schema mandates int64 only for counts/fp_counts; the extractor emits
        # integral float64 truth histograms -> accept both, require integral.
        tc = np.asarray(pack.truth_counts)
        _check_finite("truth_counts", tc)
        if np.any(tc < 0):
            _fail("truth_counts: counts must be non-negative")
        if tc.dtype != np.int64 and np.any(np.abs(tc - np.round(tc)) > 1e-9):
            _fail("truth_counts: must be integral-valued")


# --- IO --------------------------------------------------------------------------

def load_pack(npz_path, *, allow_nonstandard_grid: bool = False) -> ModelAPack:
    """Load + validate a Model A pack NPZ (fail-closed).

    The REAL grid is enforced by default; ``allow_nonstandard_grid=True`` admits
    schema-consistent smaller grids (tests only). The ``.provenance.json``
    sidecar is attached if present.
    """
    npz_path = pathlib.Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = [k for k in _REQUIRED_KEYS if k not in keys]
        if missing:
            _fail(f"pack {npz_path.name}: missing required schema keys {missing}")
        unknown = keys - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS)
        if unknown:
            _fail(f"pack {npz_path.name}: unknown keys {sorted(unknown)} "
                  f"(schema v1 is a closed contract)")
        data = {k: z[k] for k in z.files}
    prov_path = npz_path.parent / (npz_path.name[:-4] + ".provenance.json")
    provenance = None
    if prov_path.exists():
        provenance = json.loads(prov_path.read_text())
    pack = ModelAPack(
        nhat_edges=data["nhat_edges"], ntrue_edges=data["ntrue_edges"],
        zf_edges=data["zf_edges"], zc_edges=data["zc_edges"],
        kz_to_K=data["kz_to_K"], snr_edges=data["snr_edges"],
        counts=data["counts"], dX=data["dX"],
        molly_n_det=data["molly_n_det"], molly_n_tot=data["molly_n_tot"],
        molly_nhi_edges=data["molly_nhi_edges"],
        g_grid=data["g_grid"], g_occupancy=data["g_occupancy"],
        resp_mu_coef=data["resp_mu_coef"], resp_sig_coef=data["resp_sig_coef"],
        resp_skew_coef=data["resp_skew_coef"],
        resp_snr_edges=data["resp_snr_edges"], resp_z_edges=data["resp_z_edges"],
        resp_sig_floor=float(data["resp_sig_floor"]),
        resp_skew_ramp=data["resp_skew_ramp"],
        fp_counts=data["fp_counts"], fp_ell_eff=float(data["fp_ell_eff"]),
        fp_w_sightline_ratio=float(data["fp_w_sightline_ratio"]),
        fp_E_alloc=data["fp_E_alloc"], t_sigma=data["t_sigma"],
        truth_counts=data["truth_counts"],
        resp_fitcov_diag=data.get("resp_fitcov_diag"),
        # F1b fix (2026-07-11): the optional keys were silently DROPPED here
        # (resp_N_ref in particular — the response covariate reference the
        # fold REQUIRES). Carry every present optional key into the dataclass
        # so load/save round-trips are faithful.
        resp_N_ref=(float(data["resp_N_ref"]) if "resp_N_ref" in data else None),
        truth_counts_bks=data.get("truth_counts_bks"),
        dX_coarse_committed=data.get("dX_coarse_committed"),
        molly_snr_edges=data.get("molly_snr_edges"),
        nhat_masked_bins=data.get("nhat_masked_bins"),
        provenance=provenance,
    )
    validate_pack(pack, allow_nonstandard_grid=allow_nonstandard_grid)
    return pack


def save_pack(pack: ModelAPack, npz_path, *, allow_nonstandard_grid: bool = False):
    """Validate then write a pack NPZ + provenance sidecar (round-trip partner)."""
    validate_pack(pack, allow_nonstandard_grid=allow_nonstandard_grid)
    npz_path = pathlib.Path(npz_path)
    out = {}
    for k in _REQUIRED_KEYS + _OPTIONAL_KEYS:
        v = getattr(pack, {"kz_to_K": "kz_to_K"}.get(k, k))
        if v is None:
            if k in _OPTIONAL_KEYS:
                continue
            _fail(f"save_pack: required key {k} is None")
        out[k] = np.asarray(v)
    np.savez(npz_path, **out)
    prov = dict(pack.provenance or {})
    prov.setdefault("schema", "modelA_pack_schema v1 (+resp_fitcov_diag extension)")
    prov_path = npz_path.parent / (npz_path.name[:-4] + ".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2, default=str))
    return npz_path


# --- synthetic generator ----------------------------------------------------------

def small_test_grid() -> dict:
    """Schema-consistent SMALL grid kwargs for tests (10 N-bins, 6 z, 3 strata)."""
    return dict(
        nhat_edges=np.round(np.arange(19.5, 20.5 + 1e-9, 0.1), 10),
        zf_edges=np.round(np.arange(2.0, 2.6 + 1e-9, 0.1), 10),
        zc_edges=np.array([2.0, 2.2, 2.4, 2.6]),
        snr_edges=np.array([0., 2., 4., np.inf]),
        n_molly_cells=4,
    )


def synthetic_pack(
    seed=0,
    *,
    nhat_edges=None,
    zf_edges=None,
    zc_edges=None,
    snr_edges=None,
    n_molly_cells=6,
    molly_n_per_cell=6000,
    molly_scale=1.0,
    dx0=2000.0,
    f_amp=1.0,
    f_slope=2.0,
    f_curv=0.5,
    f_zslope=0.4,
    edge_push=0.0,
    response_mode="skewed",
    skew_strength=1.0,
    fp_frac=0.15,
    fp_ell_eff=4.0,
    fp_w=0.8,
    t_true=None,
    t_sigma=None,
) -> ModelAPack:
    """Generate a fully schema-conformant pack from a KNOWN synthetic truth.

    Truth pieces (all recorded in ``pack.truth``):
      * f_true(b, k): power law in N (slope ``f_slope`` per dex, curvature
        ``f_curv``) with mild z evolution ``f_zslope``; ``edge_push`` > 0 adds
        a hard bump against the TOP N edge (R8 stress).
      * C_true per molly cell (s, m): logistic in N with a stratum tilt; molly
        counts drawn Binomial(n_tot, C_true). The generator sets
        psi_c_true = eta_true - eta_hat(molly draws) so the exact forward
        expression evaluated at the truth parameters reproduces C_true.
      * Response: MOMENT coefficient surfaces (lowest-order-first) under the
        FIXED committed conventions (2026-07-11 F1-F4: covariate u relative to
        the emitted ``resp_N_ref``; sd = clip(poly, floor); moment skewness
        clamped to the attainable ceiling and ramped to ZERO above
        resp_skew_ramp[0]) — "skewed" = mean bias ~ +0.12 dex, sd ~ 0.12 dex,
        moment skewness ~ 0.2 x ``skew_strength`` (replicating the pre-fix
        generator's EFFECTIVE kernel moments so the rung statistics keep
        their calibrated regimes; well inside the ~0.995 ceiling), ramp
        center parked above the grid top so the skew stays fully active;
        "diagonal" = zero bias, sd ~ 0.021 (just above the 0.02 floor -> no
        clip kink), zero skew.
      * FP: lam_fp_true(c, s) falling in N-hat, scaled so the expected FP data
        counts are ``fp_frac`` of the expected signal counts; loa-0 counts
        ~ Poisson(ell_eff * lam); per-coarse-z transfer t_true applied in the
        data-side term. fp_frac=0 -> exact zero-FP pack.

    Counts are drawn Poisson(mu_true) with mu_true from the EXACT forward
    expression, evaluated by the independent numpy oracle
    ``forward.fold_mu_reference`` (not the jax path under test).
    """
    # lazy import: forward.py must not be imported at module load (no cycle)
    from CDDF_analysis.hbi_mcmc import forward as _fwd

    nhat_edges = REAL_NHAT_EDGES.copy() if nhat_edges is None else np.asarray(nhat_edges, float)
    zf_edges = REAL_ZF_EDGES.copy() if zf_edges is None else np.asarray(zf_edges, float)
    zc_edges = REAL_ZC_EDGES.copy() if zc_edges is None else np.asarray(zc_edges, float)
    snr_edges = REAL_SNR_EDGES.copy() if snr_edges is None else np.asarray(snr_edges, float)
    ntrue_edges = nhat_edges.copy()

    C = len(nhat_edges) - 1
    B = len(ntrue_edges) - 1
    Kf = len(zf_edges) - 1
    KK = len(zc_edges) - 1
    S = len(snr_edges) - 1

    ss = np.random.SeedSequence(seed)
    rng_molly, rng_counts, rng_fp, rng_truth = [
        np.random.default_rng(s) for s in ss.spawn(4)]

    Nc = 0.5 * (ntrue_edges[:-1] + ntrue_edges[1:])
    Nhat_c = 0.5 * (nhat_edges[:-1] + nhat_edges[1:])
    zk = 0.5 * (zf_edges[:-1] + zf_edges[1:])
    kz_to_K = (np.digitize(zk, zc_edges) - 1).clip(0, KK - 1).astype(np.int64)

    # --- pathlengths + FP exposure allocation
    w_s = 1.0 / (1.0 + 0.5 * np.arange(S))
    z_shape = 0.8 + 0.4 * (np.arange(Kf) / max(Kf - 1, 1))
    dX = dx0 * z_shape[:, None] * w_s[None, :]
    E_alloc = dX / dX.sum(axis=0, keepdims=True)

    # --- population truth
    dN_b = np.diff(ntrue_edges)
    dNc = Nc - Nc[0]
    zmid = zk.mean()
    ln_f = (np.log(f_amp) - f_slope * dNc[:, None] - f_curv * dNc[:, None] ** 2
            + f_zslope * (zk[None, :] - zmid))
    if edge_push > 0:
        from scipy.special import expit
        ln_f = ln_f + edge_push * expit((Nc[:, None] - (Nc[-1] - 0.1)) / 0.04)
    f_true = np.exp(ln_f)                                   # (B, Kf)
    theta_true = ln_f

    # --- molly completeness truth + counts
    M = int(n_molly_cells)
    molly_nhi_edges = np.linspace(ntrue_edges[0], ntrue_edges[-1], M + 1)
    Nm = 0.5 * (molly_nhi_edges[:-1] + molly_nhi_edges[1:])
    from scipy.special import expit as _expit, logit as _logit
    s_tilt = 0.3 * (np.arange(S) - (S - 1) / 2.0)
    C_true = 0.30 + 0.65 * _expit(2.0 * (Nm[None, :] - (ntrue_edges[0] + 0.35))
                                  + s_tilt[:, None])        # (S, M)
    eta_true = _logit(C_true)
    n_tot = np.full((S, M), max(int(round(molly_n_per_cell * molly_scale)), 2),
                    dtype=np.float64)
    n_det = rng_molly.binomial(n_tot.astype(np.int64), C_true).astype(np.float64)
    # keep cells away from the degenerate 0/n corners (Jeffreys logit stays finite anyway)
    eta_hat = np.log((n_det + 0.5) / (n_tot - n_det + 0.5))
    psi_c_true = eta_true - eta_hat                          # exact C_true through the fold

    # --- z-shape surface (level-preserving; flat truth for the synthetic scope)
    g_grid = np.ones((M, Kf))
    g_occupancy = np.broadcast_to(n_tot[:1, :].T, (M, Kf)).copy()

    # --- response MOMENT surfaces (lowest-order-first coefficients), FIXED
    # committed conventions (F1-F4): covariate u = N - resp_N_ref; the sd
    # polynomial IS the width in dex (clip(poly, floor), NOT floor+softplus);
    # the skew polynomial is the MOMENT skewness (attainable ceiling ~0.995),
    # multiplied by (1 - clip((N - ramp_c)/ramp_w, 0, 1)) — i.e. ramped to
    # ZERO going UP. The generator parks the ramp center ABOVE the grid top
    # so the synthetic skew stays fully active over the whole grid.
    SR = min(2, S)
    resp_snr_edges = np.array(
        [snr_edges[0], snr_edges[len(snr_edges) // 2], np.inf]) if SR == 2 else \
        np.array([snr_edges[0], np.inf])
    resp_z_edges = zc_edges.copy()
    ZR = KK
    D = 2
    resp_N_ref = 0.5 * (ntrue_edges[0] + ntrue_edges[-1])   # emitted explicitly
    si = np.arange(SR)[:, None, None]
    zi = np.arange(ZR)[None, :, None]
    if response_mode == "skewed":
        # MOMENT surfaces chosen to REPLICATE the pre-fix generator's
        # EFFECTIVE kernel (xi = N+0.03, omega ~ 0.147, alpha = 1.2 direct),
        # whose moments are mean-bias ~ 0.120, sd ~ 0.116, moment-skewness
        # ~ 0.200 — so the calibrated R4-R8 rung regimes (identifiability on
        # the small grid, the R6 bias contrast, the R5 width ordering) are
        # preserved. Verified 2026-07-11: emitting (0.03, 0.15, 0.8) instead
        # gave omega 0.238 / alpha 4.2, smeared the small-grid window and
        # broke R5/R6 while the fold<->generator agreement stayed 7e-15.
        resp_mu_coef = np.concatenate([
            0.12 + 0.01 * zi - 0.005 * si + 0 * zi,  # mean bias E[x-hat - N]
            0.02 + 0 * zi + 0 * si], axis=2).astype(float)
        resp_sig_coef = np.concatenate([
            0.12 + 0.01 * si + 0 * zi,      # sd in dex (clip semantics)
            0.005 + 0 * zi + 0 * si], axis=2).astype(float)
        resp_skew_coef = np.concatenate([
            0.2 * skew_strength + 0 * zi + 0 * si,   # moment skewness
            0 * zi + 0 * si], axis=2).astype(float)
        resp_sig_floor = 0.02
    elif response_mode == "diagonal":
        zeros = 0.0 * (si + zi)
        resp_mu_coef = np.concatenate([zeros, zeros], axis=2).astype(float)
        # sd just ABOVE the floor: near-diagonal, and no cell sits exactly on
        # the clip boundary (zero-gradient kink) at the truth point.
        resp_sig_coef = np.concatenate([0.021 + zeros, zeros], axis=2).astype(float)
        resp_skew_coef = np.concatenate([zeros, zeros], axis=2).astype(float)
        resp_sig_floor = 0.02
    else:
        raise ValueError(f"unknown response_mode {response_mode!r}")
    resp_skew_ramp = np.array([ntrue_edges[-1] + 0.5, 0.5])
    resp_fitcov_diag = np.stack([
        np.full((SR, ZR), 0.02 ** 2),   # var of the order-0 mu-coef perturbation
        # var of the order-0 sd-coef perturbation — now in DEX of width
        # directly (clip semantics); 0.02 dex sd keeps the perturbed width
        # well clear of the floor (0.15 - few sd >> 0.02: no clipped cells).
        np.full((SR, ZR), 0.02 ** 2),
    ], axis=0)

    # --- transfer factors + prior widths
    t_true = np.zeros(KK) if t_true is None else np.asarray(t_true, float)
    if t_true.shape != (KK,):
        raise ValueError(f"t_true must have shape ({KK},)")
    t_sigma = (np.linspace(0.45, 0.20, KK) if t_sigma is None
               else np.asarray(t_sigma, float))

    # --- assemble a provisional pack (zero counts) so the oracle can run on it
    zero_counts = np.zeros((C, Kf, S), dtype=np.int64)
    zero_fp = np.zeros((C, S), dtype=np.int64)
    pack = ModelAPack(
        nhat_edges=nhat_edges, ntrue_edges=ntrue_edges, zf_edges=zf_edges,
        zc_edges=zc_edges, kz_to_K=kz_to_K, snr_edges=snr_edges,
        counts=zero_counts, dX=dX,
        molly_n_det=n_det, molly_n_tot=n_tot, molly_nhi_edges=molly_nhi_edges,
        g_grid=g_grid, g_occupancy=g_occupancy,
        resp_mu_coef=resp_mu_coef, resp_sig_coef=resp_sig_coef,
        resp_skew_coef=resp_skew_coef, resp_snr_edges=resp_snr_edges,
        resp_z_edges=resp_z_edges, resp_sig_floor=resp_sig_floor,
        resp_skew_ramp=resp_skew_ramp,
        fp_counts=zero_fp, fp_ell_eff=fp_ell_eff,
        fp_w_sightline_ratio=fp_w, fp_E_alloc=E_alloc, t_sigma=t_sigma,
        truth_counts=np.zeros((B, Kf), dtype=np.int64),
        resp_fitcov_diag=resp_fitcov_diag,
        resp_N_ref=float(resp_N_ref),
    )

    psi_k_zero = np.zeros((2, SR, ZR))

    # --- FP intensity: scale so expected FP data counts = fp_frac * signal counts
    mu_signal = _fwd.fold_mu_reference(theta_true, psi_c_true, psi_k_zero,
                                       t_true, np.zeros((C, S)), pack)
    if fp_frac > 0:
        shape_cs = np.exp(-2.5 * (Nhat_c[:, None] - Nhat_c[0])) / (1.0 + 0.3 * np.arange(S))
        exp_t_alloc = (np.exp(t_true)[kz_to_K][:, None] * E_alloc).sum(axis=0)  # (S,)
        fp_data_per_unit = fp_w * (shape_cs * exp_t_alloc[None, :]).sum()
        L0 = fp_frac * mu_signal.sum() / fp_data_per_unit
        lam_fp_true = L0 * shape_cs
    else:
        lam_fp_true = np.zeros((C, S))
    fp_counts = rng_fp.poisson(fp_ell_eff * lam_fp_true).astype(np.int64)

    # --- data counts from the EXACT forward expression (numpy oracle)
    mu_true = _fwd.fold_mu_reference(theta_true, psi_c_true, psi_k_zero,
                                     t_true, lam_fp_true, pack)
    counts = rng_counts.poisson(mu_true).astype(np.int64)

    # --- truth closure counts (intrinsic systems, no selection)
    truth_counts = rng_truth.poisson(
        dX.sum(axis=1)[None, :] * f_true * dN_b[:, None]).astype(np.int64)

    pack = dataclasses.replace(
        pack, counts=counts, fp_counts=fp_counts, truth_counts=truth_counts,
        provenance={
            "generator": "CDDF_analysis.hbi_mcmc.pack.synthetic_pack",
            "synthetic": True, "seed": int(seed),
            "response_mode": response_mode, "fp_frac": float(fp_frac),
            "note": "fully synthetic; no survey data of any kind",
        },
        truth={
            "f_true": f_true, "theta_true": theta_true,
            "C_true": C_true, "eta_true": eta_true, "psi_c_true": psi_c_true,
            "lam_fp_true": lam_fp_true, "t_true": t_true,
            "mu_true": mu_true, "mu_signal": mu_signal,
            "dN_b": dN_b, "Nc": Nc, "zk": zk,
        },
    )
    validate_pack(pack, allow_nonstandard_grid=True)
    return pack
