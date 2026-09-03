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
    "upgrade_pack_v11",
    "coarsen_basis",
    "resp_fit_range_from_forward_npz",
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
                  "dX_coarse_committed", "molly_snr_edges", "nhat_masked_bins",
                  "resp_N_fit_range", "fp_eta_c")
# schema-v1.2 extension (PI ruling 2026-08-17, adoption + contract): the
# ADOPTED response representation + its sightline-bootstrap carrier + the
# count-conservation contract stamps. ALL-OR-NONE: a pack carrying any of
# these must carry every one (validated below); the frozen resp_* surfaces
# stay byte-identical and remain what fold_mu consumes — the adopted
# surfaces are folded ONLY through count_conserving_fold.cc_fold_adopted
# (renormalized, deployed phi_ref), never naively.
_ADOPTED_KEYS = ("tp_convention_id", "contract_id", "adopted_resp_version",
                 "adopted_resp_mu_coef", "adopted_resp_sig_coef",
                 "adopted_resp_skew_coef", "adopted_resp_fit_range",
                 "adopted_phi_ref", "adopted_carrier_mu",
                 "adopted_carrier_sig", "adopted_carrier_skew",
                 "adopted_carrier_shared3",
                 # 2026-09-02 (response-estimator rebuild, default-off): an EMPIRICAL bin-to-bin kernel (SR, ZR, C, B) that, when present,
                 # REPLACES the adopted skew-normal surfaces in build_cc_tensors; adopted_phi_ref must equal its column sums (G-CC analogue).
                 # Absent from every production pack; the low-z real pack never carries it.
                 "adopted_masses_override")
_OPTIONAL_KEYS = _OPTIONAL_KEYS + _ADOPTED_KEYS


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
    # (SR, ZR, 2) = [N_lo, N_hi] of the CALIBRATED covariate range per response
    # cell: the min/max true-N anchor the moment polynomials were actually FIT
    # at.  REQUIRED by the fold (2026-07-28 finding D2) -- see forward.py.
    resp_N_fit_range: Optional[np.ndarray] = None
    # (C,) per-observed-bin host-occlusion fraction of the loa-0 FP product
    # definition (mu_FP ∝ (1 - eta_band); eta_DLA == 0 forced). REQUIRED by
    # the fold (restoration 2026-08-06, PI ruling 8) -- see forward.py.
    fp_eta_c: Optional[np.ndarray] = None
    truth_counts_bks: Optional[np.ndarray] = None  # (B, Kf, S)
    dX_coarse_committed: Optional[np.ndarray] = None
    molly_snr_edges: Optional[np.ndarray] = None
    nhat_masked_bins: Optional[np.ndarray] = None
    # schema-v1.2 adopted-response/contract stamps (all-or-none; see
    # _ADOPTED_KEYS above and _validate_adopted_contract below)
    tp_convention_id: Optional[str] = None
    contract_id: Optional[str] = None
    adopted_resp_version: Optional[str] = None
    adopted_resp_mu_coef: Optional[np.ndarray] = None    # (SR, ZR, DA)
    adopted_resp_sig_coef: Optional[np.ndarray] = None   # (SR, ZR, DA)
    adopted_resp_skew_coef: Optional[np.ndarray] = None  # (SR, ZR, DA)
    adopted_resp_fit_range: Optional[np.ndarray] = None  # (SR, ZR, 2)
    adopted_phi_ref: Optional[np.ndarray] = None         # (SR, ZR, B)
    adopted_carrier_mu: Optional[np.ndarray] = None      # (Nd, SR, ZR, DA)
    adopted_carrier_sig: Optional[np.ndarray] = None     # (Nd, SR, ZR, DA)
    adopted_carrier_skew: Optional[np.ndarray] = None    # (Nd, SR, ZR, DA)
    adopted_carrier_shared3: Optional[np.ndarray] = None  # (Nd, 3)
    adopted_masses_override: Optional[np.ndarray] = None  # (SR, ZR, C, B) empirical kernel masses; default-off
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

    @property
    def n_pad_bins(self):
        """Schema-v1.1 DOWNWARD basis-pad depth: true-N bins BELOW the
        reporting floor (finding D1).  0 means the true-N basis is truncated at
        the reporting floor and the fold is arithmetically incapable of
        reproducing the lowest observed bins.  Exposed as a first-class number
        so a pre-flight can REFUSE an unpadded pack instead of re-deriving the
        edge arithmetic at every call site.

        COUNTED FROM THE EDGES, not from the array lengths (fixed 2026-07-29
        with PI decision 3).  ``len(ntrue) - len(nhat)`` was only ever right
        while the basis shared the observed 0.1-dex step; on the adopted 0.2-dex
        basis it goes NEGATIVE (18 - 30 = -12) and every downstream
        ``truth[:n_pad]`` slice silently means the wrong thing.  The definition
        below is identical on every 0.1-dex pack (pinned by test)."""
        ne = np.asarray(self.ntrue_edges, float)
        floor = float(np.asarray(self.nhat_edges, float)[0])
        return int(np.sum(ne[:-1] < floor - 1e-9))

    @property
    def basis_width(self):
        """NOMINAL latent true-N basis width in dex (PI decision 3): the MODAL
        bin width.  A non-uniform basis is legal and expected -- E4's merging
        convention absorbs the remainder into the last group of each segment, so
        the adopted 0.2-dex basis has a 0.3-dex top bin ([22.1, 22.4)) and, under
        pad 19.0, a 0.3-dex topmost PAD bin ([19.2, 19.5)).  Use
        ``basis_is_uniform`` to ask whether every bin has this width, and
        ``np.diff(ntrue_edges)`` when the exact widths matter (the fold always
        does)."""
        d = np.round(np.diff(np.asarray(self.ntrue_edges, float)), 8)
        vals, cnt = np.unique(d, return_counts=True)
        return float(vals[int(np.argmax(cnt))])

    @property
    def basis_is_uniform(self):
        """True iff every latent basis bin has the same width."""
        d = np.diff(np.asarray(self.ntrue_edges, float))
        return bool(np.allclose(d, d[0], atol=1e-8))


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
    # --- schema v1.1 BASIS PAD (2026-07-28, finding D1) ---------------------
    # v1 required ntrue_edges == nhat_edges and deferred the basis-pad decision.
    # That deferral IS the rung-9 forward-model failure: the observed n-hat bins
    # just above the reporting floor are fed overwhelmingly by TRUE systems
    # BELOW it (the forward response has a ~+0.27 dex up-bias and ~0.28 dex
    # width at 19.5), so a true-N basis truncated at the reporting floor cannot
    # reproduce the counts -- the same one-sided-support class as B16.  The
    # committed estimator already carries the fix (cddf_catalog_hbi.py
    # `basis_pad_floor`, 2026-06-17).  ntrue_edges may now EXTEND BELOW
    # nhat_edges on the SAME uniform step, sharing the top edge; nhat_edges must
    # be an exact tail-subset of ntrue_edges.  Padding only ever goes DOWN --
    # the reporting window never grows.
    # --- PI DECISION 3 (2026-07-29): a COARSER latent basis is legal ---------
    # The LATENT true-N basis may be coarser than the observed grid (adopted:
    # 0.2 dex) because the 0.1-dex basis is unidentifiable -- E4 measured up to
    # 45.7-62x noise amplification per bin and 17-19 of 27-29 basis directions
    # already prior-dominated.  What must NOT move is the OBSERVED / REPORTING
    # grid, so the rules are:
    #   * every ntrue edge lies on the observed 0.1-dex grid (no new sub-grid);
    #   * the TOP edge is shared with nhat_edges (the basis never extends UP);
    #   * the basis covers the observed grid from below (bottom <= nhat bottom);
    #   * the observed/reporting FLOOR is itself an ntrue edge, so no single
    #     basis bin straddles the pad/report boundary (a bin that did would mix
    #     convention-dependent sub-floor support into an in-window bin);
    #   * the basis need NOT be uniform: E4's merging convention absorbs the
    #     remainder into the last group of a segment (0.3-dex top bin at
    #     0.2 dex), and forward.build_consts uses diff(ntrue_edges) throughout.
    # When the basis IS on the 0.1-dex step the OLD, stricter tail-subset rule
    # is applied unchanged, so every v1 / v1.1 pack validates bit-identically.
    _ne = np.asarray(pack.ntrue_edges, float)
    _ce = np.asarray(pack.nhat_edges, float)
    if not np.array_equal(_ne, _ce):
        _check_finite("ntrue_edges", _ne)
        if _ne.ndim != 1 or len(_ne) < 2:
            _fail(f"ntrue_edges: expected 1-D with >= 2 entries, got {_ne.shape}")
        if np.any(np.diff(_ne) <= 0):
            _fail("ntrue_edges: edges must be strictly increasing")
        _uniform_fine = np.allclose(np.diff(_ne), _N_STEP, atol=1e-8)
        if _uniform_fine:
            _check_edges_uniform("ntrue_edges", _ne, _N_STEP)
            if len(_ne) < len(_ce):
                _fail("ntrue_edges: the true-N basis may only EXTEND the "
                      f"observed grid downward, never shrink it (got "
                      f"{len(_ne)-1} true bins vs {len(_ce)-1} observed bins)")
            if not np.allclose(_ne[len(_ne) - len(_ce):], _ce, atol=1e-8):
                _fail("ntrue_edges: nhat_edges must be an exact TAIL subset of "
                      "ntrue_edges (same step, same top edge) — the basis pad "
                      f"extends DOWN only. got ntrue tail "
                      f"{_ne[len(_ne)-len(_ce):]}")
        else:
            # coarser (or mixed-width) LATENT basis
            off = (_ne - _ce[0]) / _N_STEP
            if not np.allclose(off, np.round(off), atol=1e-6):
                _fail("ntrue_edges: every latent-basis edge must lie on the "
                      f"observed {_N_STEP} dex grid anchored at the reporting "
                      f"floor {_ce[0]}; got {_ne}")
            if not np.isclose(_ne[-1], _ce[-1], atol=1e-8):
                _fail("ntrue_edges: the latent basis must share the observed "
                      f"grid's TOP edge {_ce[-1]} (it never extends UP); got "
                      f"{_ne[-1]}")
            if _ne[0] > _ce[0] + 1e-8:
                _fail("ntrue_edges: the latent basis must cover the observed "
                      f"grid from below (basis floor {_ne[0]} > observed floor "
                      f"{_ce[0]})")
            if not np.any(np.isclose(_ne, _ce[0], atol=1e-8)):
                _fail("ntrue_edges: the observed/reporting FLOOR "
                      f"{_ce[0]} must itself be a latent-basis edge — otherwise "
                      "one basis bin straddles the pad/report boundary and "
                      "mixes convention-dependent sub-floor support into an "
                      f"in-window bin. got {_ne}")
            # NOTE (2026-07-29, found by mutation testing): there is NO
            # separate "finer than the observed grid" check, because it would be
            # DEAD CODE.  The on-grid rule above already forbids it: every ntrue
            # edge must sit on the observed 0.1-dex grid, and edges must be
            # strictly increasing, so the narrowest representable basis bin IS
            # 0.1 dex.  A 0.05-dex basis is refused by the on-grid rule, with
            # that rule's message.  An earlier version of this function carried
            # an extra max(diff) < _N_STEP check; it could never fire and a test
            # that "verified" it was in fact exercising the on-grid rule.
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
    # The molly must cover the REPORTED support [nhat_edges[0], ntrue_edges[-1]].
    # Below the reporting floor (the schema-v1.1 basis pad, finding D1) the
    # completeness is the CONSTANT EXTRAPOLATION of the molly's lowest cell —
    # the committed estimator's own convention (cddf_catalog_hbi.py:6117,
    # 2026-06-17 basis_pad_floor block), and automatic here because
    # forward.build_consts maps b -> molly cell with
    # clip(digitize(Nc, molly_nhi_edges) - 1, 0, M-2), so any sub-floor true-N
    # bin reads cell 0. This is an APPROXIMATION (completeness genuinely falls
    # below 19.5) and is the leading known systematic on the pad.
    ce_lo = float(np.asarray(pack.nhat_edges, float)[0])
    if me[0] > ce_lo + 1e-8 or me[-1] < ne[-1] - 1e-8:
        _fail(f"molly_nhi_edges: must cover the reported N range "
              f"[{ce_lo}, {ne[-1]}], got [{me[0]}, {me[-1]}]")
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
    # schema-v1.2 adopted-response/contract stamp group: ALL-OR-NONE,
    # fail-closed (PI ruling 2026-08-17). The frozen resp_* surfaces above
    # are untouched by this group; the adopted surfaces may only be folded
    # count-conservingly (count_conserving_fold.cc_fold_adopted).
    _adopted_fields = ("tp_convention_id", "contract_id",
                       "adopted_resp_version", "adopted_resp_mu_coef",
                       "adopted_resp_sig_coef", "adopted_resp_skew_coef",
                       "adopted_resp_fit_range", "adopted_phi_ref",
                       "adopted_carrier_mu", "adopted_carrier_sig",
                       "adopted_carrier_skew", "adopted_carrier_shared3")
    _present = [f for f in _adopted_fields if getattr(pack, f) is not None]
    if _present and len(_present) != len(_adopted_fields):
        _fail("adopted-contract stamp group is ALL-OR-NONE: present="
              f"{_present}, missing="
              f"{[f for f in _adopted_fields if f not in _present]}")
    if _present:
        B = len(np.asarray(pack.ntrue_edges)) - 1
        am = np.asarray(pack.adopted_resp_mu_coef, float)
        if am.ndim != 3 or am.shape[:2] != (SR, ZR):
            _fail(f"adopted_resp_mu_coef: bad shape {am.shape}")
        DA = am.shape[-1]
        for nm in ("adopted_resp_sig_coef", "adopted_resp_skew_coef"):
            _check_shape(nm, getattr(pack, nm), (SR, ZR, DA))
            _check_finite(nm, getattr(pack, nm))
        _check_finite("adopted_resp_mu_coef", am)
        _check_shape("adopted_resp_fit_range", pack.adopted_resp_fit_range,
                     (SR, ZR, 2))
        rr = np.asarray(pack.adopted_resp_fit_range, float)
        if np.any(rr[..., 0] >= rr[..., 1]):
            _fail("adopted_resp_fit_range: lo must be < hi in every cell")
        _check_shape("adopted_phi_ref", pack.adopted_phi_ref, (SR, ZR, B))
        if pack.adopted_masses_override is not None:
            _check_shape("adopted_masses_override", pack.adopted_masses_override, (SR, ZR, len(pack.nhat_edges) - 1, B))
            d_over = float(np.max(np.abs(np.asarray(pack.adopted_masses_override, float).sum(axis=2) - np.asarray(pack.adopted_phi_ref, float))))
            if d_over > 1e-9:
                raise ValueError(f"adopted_masses_override column sums deviate from adopted_phi_ref by {d_over:.2e} (G-CC analogue)")
        pr = np.asarray(pack.adopted_phi_ref, float)
        if np.any(~np.isfinite(pr)) or np.any(pr < 0) or np.any(pr > 1 + 1e-9):
            _fail("adopted_phi_ref: must be finite fractions in [0, 1]")
        cm = np.asarray(pack.adopted_carrier_mu, float)
        if cm.ndim != 4 or cm.shape[1:] != (SR, ZR, DA) or cm.shape[0] < 50:
            _fail(f"adopted_carrier_mu: bad shape {cm.shape} "
                  "(need >= 50 draws matching the adopted surfaces)")
        for nm in ("adopted_carrier_sig", "adopted_carrier_skew"):
            _check_shape(nm, getattr(pack, nm), cm.shape)
            _check_finite(nm, getattr(pack, nm))
        _check_shape("adopted_carrier_shared3", pack.adopted_carrier_shared3,
                     (cm.shape[0], 3))
        for nm in ("tp_convention_id", "contract_id",
                   "adopted_resp_version"):
            v = getattr(pack, nm)
            if not isinstance(v, str) or not v.strip():
                _fail(f"{nm}: must be a non-empty string, got {v!r}")
    if pack.resp_N_ref is not None:
        nr = float(np.asarray(pack.resp_N_ref))
        if not np.isfinite(nr):
            _fail(f"resp_N_ref: must be a finite scalar, got {nr}")
    if pack.resp_N_fit_range is not None:
        _check_shape("resp_N_fit_range", pack.resp_N_fit_range, (SR, ZR, 2))
        _check_finite("resp_N_fit_range", pack.resp_N_fit_range)
        rr = np.asarray(pack.resp_N_fit_range, float)
        if np.any(rr[..., 1] <= rr[..., 0]):
            _fail("resp_N_fit_range: every cell needs N_hi > N_lo (the "
                  "calibrated covariate range the moment polynomials were fit "
                  "over)")
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
    if pack.fp_eta_c is not None:
        _check_shape("fp_eta_c", pack.fp_eta_c, (C,))
        _check_finite("fp_eta_c", pack.fp_eta_c)
        eta = np.asarray(pack.fp_eta_c, float)
        if np.any(eta < 0) or np.any(eta >= 1):
            _fail("fp_eta_c: host-occlusion fractions must satisfy 0 <= eta < 1")
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
        resp_N_fit_range=data.get("resp_N_fit_range"),
        fp_eta_c=data.get("fp_eta_c"),
        truth_counts_bks=data.get("truth_counts_bks"),
        dX_coarse_committed=data.get("dX_coarse_committed"),
        molly_snr_edges=data.get("molly_snr_edges"),
        nhat_masked_bins=data.get("nhat_masked_bins"),
        tp_convention_id=(str(data["tp_convention_id"])
                          if "tp_convention_id" in data else None),
        contract_id=(str(data["contract_id"])
                     if "contract_id" in data else None),
        adopted_resp_version=(str(data["adopted_resp_version"])
                              if "adopted_resp_version" in data else None),
        adopted_resp_mu_coef=data.get("adopted_resp_mu_coef"),
        adopted_resp_sig_coef=data.get("adopted_resp_sig_coef"),
        adopted_resp_skew_coef=data.get("adopted_resp_skew_coef"),
        adopted_resp_fit_range=data.get("adopted_resp_fit_range"),
        adopted_phi_ref=data.get("adopted_phi_ref"),
        adopted_carrier_mu=data.get("adopted_carrier_mu"),
        adopted_carrier_sig=data.get("adopted_carrier_sig"),
        adopted_carrier_skew=data.get("adopted_carrier_skew"),
        adopted_carrier_shared3=data.get("adopted_carrier_shared3"),
        adopted_masses_override=data.get("adopted_masses_override"),
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


# --- schema v1.2 migration: fp_eta_c ------------------------------------------
#: the committed loa-0 product's host-occlusion band structure
#: (gl_loa0_fp_v1_20260615 band_eta_per_nbin; eta_DLA == 0 FORCED by the
#: product — see build_loa0_fp_product.py). Used ONLY by the explicit
#: legacy-pack migration below; the extractor reads the product arrays.
FP_ETA_BANDS_COMMITTED = (
    (17.2, 19.0, 0.0111869413715736),
    (19.0, 20.3, 0.005756532459300326),
    (20.3, np.inf, 0.0),
)


def eta_from_intervals(nhat_edges, seg_lo, seg_hi, seg_eta) -> np.ndarray:
    """(C,) per-observed-bin eta from piecewise interval values.

    THE one canonical interval lookup (used by the extractor on the product's
    fine grid and by ``attach_fp_eta_bands`` on the committed band table).
    FAIL-LOUD: every observed bin must be covered and must not straddle a
    boundary between different eta values — a band edge can never be silently
    averaged across (the one-sided-support bug class).
    """
    ne = np.asarray(nhat_edges, float)
    seg_lo = np.asarray(seg_lo, float)
    seg_hi = np.asarray(seg_hi, float)
    seg_eta = np.asarray(seg_eta, float)
    out = np.empty(len(ne) - 1, float)
    for c in range(len(ne) - 1):
        lo_c, hi_c = float(ne[c]), float(ne[c + 1])
        inside = (seg_hi > lo_c + 1e-9) & (seg_lo < hi_c - 1e-9)
        if not inside.any():
            raise ValueError(
                f"eta_from_intervals: observed bin [{lo_c}, {hi_c}) is not "
                "covered by the eta segments — cannot derive eta.")
        vals = np.unique(np.round(seg_eta[inside], 12))
        if vals.size != 1:
            raise ValueError(
                f"eta_from_intervals: observed bin [{lo_c}, {hi_c}) straddles "
                f"an eta band boundary (values {vals}) — refusing to average "
                "across bands.")
        out[c] = float(vals[0])
    return out


def attach_fp_eta_bands(pack: ModelAPack,
                        bands=FP_ETA_BANDS_COMMITTED) -> ModelAPack:
    """EXPLICIT schema-v1.2 migration for packs extracted before 2026-08-06.

    Derives ``fp_eta_c`` from the committed product's band structure and
    returns a new pack. Idempotent: a pack that already carries ``fp_eta_c``
    is returned unchanged (so it can wrap any loader). This is an explicit
    opt-in — ``build_consts`` stays fail-loud on unmigrated packs — mirroring
    the v1.1 ``resp_fit_range_from_forward_npz`` migration pattern.
    """
    if pack.fp_eta_c is not None:
        return pack
    lo = [b[0] for b in bands]
    hi = [b[1] for b in bands]
    eta = [b[2] for b in bands]
    return dataclasses.replace(
        pack, fp_eta_c=eta_from_intervals(pack.nhat_edges, lo, hi, eta))


# --- schema v1.1 migration ----------------------------------------------------
def resp_fit_range_from_forward_npz(forward_npz) -> np.ndarray:
    """(SR, ZR, 2) calibrated covariate range from a ForwardResponseModel NPZ.

    The committed ``save_forward_response`` NPZ carries ``emp_N_anchors``
    (SR, ZR, n_anchor) — the true-N sub-bin anchors the per-cell MOMENT
    polynomials were weighted-least-squares fitted at
    (``znz_kernel.fit_forward_response._fit_poly``). Their per-cell min/max IS
    the range outside which those polynomials are extrapolation (finding D2).
    """
    with np.load(forward_npz, allow_pickle=True) as d:
        if "emp_N_anchors" not in d.files:
            _fail(f"{forward_npz}: no emp_N_anchors — cannot recover the "
                  "calibrated covariate range (re-fit with build_empirical=True)")
        a = np.asarray(d["emp_N_anchors"], float)
    if a.ndim != 3:
        _fail(f"emp_N_anchors: expected (SR, ZR, n_anchor), got {a.shape}")
    return np.stack([a.min(axis=-1), a.max(axis=-1)], axis=-1)


def coarsen_basis(pack: ModelAPack, basis_width, pad_floor=None) -> ModelAPack:
    """Re-grid a pack's LATENT true-N basis to ``basis_width`` (PI decision 3).

    SYNTHETIC / TEST utility.  ``extract_pack`` builds real packs on the coarse
    basis directly from the catalogue; this exists so a SYNTHETIC pack can be put
    on the ADOPTED geometry (coarse basis, optional downward pad) for coverage
    work, using EXACTLY the adopted convention: ``reporting.basis_groups`` in two
    segments split at the reporting floor, truth COUNTS summed within a group,
    truth ``f`` dN-weighted-averaged (``reporting.merged_truth``).

    The observed axis (``nhat_edges``, ``counts``, ``dX``, every calibration
    block) is untouched.  New pad bins carry ZERO truth counts -- the pad's role
    here is to supply latent nuisance SUPPORT, and a synthetic study draws its own
    truth on that support from the prior.
    """
    from CDDF_analysis.hbi_mcmc import reporting as _RP
    fine = np.asarray(pack.ntrue_edges, float)
    if not np.allclose(np.diff(fine), _N_STEP, atol=1e-8):
        _fail("coarsen_basis: the input pack's basis must be on the observed "
              f"{_N_STEP} dex step (got widths {np.diff(fine)})")
    obs_lo = float(np.asarray(pack.nhat_edges, float)[0])
    n_pad_fine = 0
    if pad_floor is not None:
        n_pad_fine = int(round((fine[0] - float(pad_floor)) / _N_STEP))
        if n_pad_fine < 0:
            n_pad_fine = 0
        elif n_pad_fine and abs(fine[0] - n_pad_fine * _N_STEP
                                - float(pad_floor)) > 1e-8:
            _fail(f"coarsen_basis: pad_floor {pad_floor} is off the "
                  f"{_N_STEP} dex grid")
        if n_pad_fine:
            fine = np.round(np.concatenate(
                [fine[0] - _N_STEP * np.arange(n_pad_fine, 0, -1), fine]), 10)
    n_below = int(np.sum(fine[:-1] < obs_lo - 1e-9))
    n_above = len(fine) - 1 - n_below
    g = int(round(float(basis_width) / _N_STEP))
    if g < 1 or abs(g * _N_STEP - float(basis_width)) > 1e-8:
        _fail(f"coarsen_basis: basis_width {basis_width} must be a positive "
              f"integer multiple of {_N_STEP}")
    groups = []
    if n_below:
        groups += _RP.basis_groups(n_below, g)
    groups += [[b + n_below for b in gr] for gr in _RP.basis_groups(n_above, g)]
    edges = _RP.merged_edges(fine, groups)

    def _pad_sum(a):
        if a is None:
            return None
        a = np.asarray(a, float)
        if n_pad_fine:
            a = np.concatenate([np.zeros((n_pad_fine,) + a.shape[1:]), a], 0)
        return np.stack([a[gr].sum(axis=0) for gr in groups])

    truth = None
    if pack.truth is not None:
        truth = dict(pack.truth)
        if "f_true" in truth:
            f = np.asarray(truth["f_true"], float)
            if n_pad_fine:
                f = np.concatenate([np.zeros((n_pad_fine,) + f.shape[1:]), f], 0)
            dNf = np.diff(fine)
            truth["f_true"] = np.stack(
                [_RP.merged_truth(f[:, k], dNf, groups) for k in range(f.shape[1])],
                axis=1)
            truth["basis_coarsened_to_dex"] = float(basis_width)
    out = dataclasses.replace(
        pack, ntrue_edges=edges,
        truth_counts=_pad_sum(pack.truth_counts),
        truth_counts_bks=_pad_sum(pack.truth_counts_bks),
        truth=truth)
    validate_pack(out, allow_nonstandard_grid=True)
    return out


def upgrade_pack_v11(npz_in, npz_out, *, forward_npz=None,
                     resp_N_fit_range=None) -> str:
    """Copy a schema-v1 pack to v1.1, ADDING ``resp_N_fit_range``.

    Pure NPZ->NPZ (no survey IO, no heavy imports): every existing key is
    carried through byte-for-byte and NOTHING is deleted or overwritten. The
    provenance sidecar is copied with an ``upgrade`` note appended.
    """
    npz_in, npz_out = pathlib.Path(npz_in), pathlib.Path(npz_out)
    if resp_N_fit_range is None:
        if forward_npz is None:
            raise ValueError("upgrade_pack_v11: pass forward_npz or "
                             "resp_N_fit_range")
        resp_N_fit_range = resp_fit_range_from_forward_npz(forward_npz)
    rr = np.asarray(resp_N_fit_range, float)
    with np.load(npz_in, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if "resp_N_fit_range" in data:
        raise ValueError(f"{npz_in.name} already carries resp_N_fit_range")
    SR, ZR = np.asarray(data["resp_mu_coef"]).shape[:2]
    if rr.shape != (SR, ZR, 2):
        raise ValueError(f"resp_N_fit_range shape {rr.shape} != {(SR, ZR, 2)}")
    data["resp_N_fit_range"] = rr
    np.savez(npz_out, **data)
    prov_in = npz_in.parent / (npz_in.name[:-4] + ".provenance.json")
    if prov_in.exists():
        prov = json.loads(prov_in.read_text())
        prov["schema"] = "modelA_pack_schema v1.1 (+resp_N_fit_range)"
        prov.setdefault("upgrades", []).append({
            "from": str(npz_in), "key": "resp_N_fit_range",
            "source": str(forward_npz) if forward_npz else "explicit",
            "routine": "CDDF_analysis/hbi_mcmc/pack.py:upgrade_pack_v11",
            "reason": ("finding D2 (2026-07-28): the response MOMENT "
                       "polynomials were extrapolated ~1.2 dex past their top "
                       "empirical anchor, manufacturing the rung-9 high-N "
                       "excess. The fold now clamps the covariate to this "
                       "range."),
        })
        (npz_out.parent / (npz_out.name[:-4] + ".provenance.json")).write_text(
            json.dumps(prov, indent=1, default=str))
    return str(npz_out)


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


def _validate_basis_pad(nhat_edges, ntrue_edges):
    """The schema-v1.1 basis-pad rule, enforced at CONSTRUCTION.

    ``ntrue_edges`` may only EXTEND ``nhat_edges`` DOWNWARD on the same uniform
    step, sharing the top edge; ``nhat_edges`` must be an exact TAIL subset.
    Identical to the rule ``validate_pack`` enforces -- checked here too so a
    caller gets the error at the point of the mistake rather than deep inside
    the generator's array arithmetic.
    """
    ne = np.asarray(ntrue_edges, float)
    ce = np.asarray(nhat_edges, float)
    if np.array_equal(ne, ce):
        return
    _check_edges_uniform("ntrue_edges", ne, _N_STEP)
    if len(ne) < len(ce):
        _fail("ntrue_edges: the true-N basis may only EXTEND the observed grid "
              f"downward, never shrink it (got {len(ne)-1} true bins vs "
              f"{len(ce)-1} observed bins)")
    if not np.allclose(ne[len(ne) - len(ce):], ce, atol=1e-8):
        _fail("ntrue_edges: nhat_edges must be an exact TAIL subset of "
              "ntrue_edges (same step, same top edge) — the basis pad extends "
              f"DOWN only. got ntrue tail {ne[len(ne)-len(ce):]}, "
              f"nhat {ce}")


def synthetic_pack(
    seed=0,
    *,
    nhat_edges=None,
    ntrue_edges=None,
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
    fp_eta=None,
    t_true=None,
    t_sigma=None,
) -> ModelAPack:
    """Generate a fully schema-conformant pack from a KNOWN synthetic truth.

    ``ntrue_edges`` (schema v1.1 BASIS PAD, decisions 3 and 4) defaults to
    ``nhat_edges``, i.e. no pad.  Pass it to build a pack whose LATENT true-N
    basis extends BELOW the reporting floor: same uniform 0.1-dex step, same
    top edge, ``nhat_edges`` an exact tail subset, so ``n_pad_bins > 0``.  The
    molly and the response covariate range follow the padded basis
    automatically.  Padding is what makes a MATCHED SBC possible for the
    production geometry -- ``grid.ntrue_edges`` is a ``sbc.MATCH_KEYS`` entry
    and ``sbc.sbc_run`` builds its template pack here.

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
        The data-side normalisation carries ``fp_w * fp_ell_eff`` (2026-08-05
        repair), matching ``forward.fold_mu``: ``lam`` is an intensity per unit
        loa-0 exposure, so at fixed ``fp_frac`` a larger ``fp_ell_eff`` means a
        SMALLER lam_fp_true and correspondingly fewer loa-0 ``fp_counts`` --
        which is the production regime (89 loa-0 counts carrying mu_FP ~ 1.5e4).

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

    # --- the LATENT basis (schema v1.1 basis pad, decisions 3 and 4) --------
    # 🔴 This used to be hardcoded ``ntrue_edges = nhat_edges.copy()``, i.e. no
    # padded pack could be GENERATED at all -- the only padded packs in the
    # repository were hand-built ``SimpleNamespace``s and the
    # ``forward_selftest.extend_pack_truth`` diagnostic, which drops
    # ``truth_counts`` and never validates.  Since ``grid.ntrue_edges`` is a
    # matched-SBC MATCH_KEY (``sbc.MATCH_KEYS``) and ``sbc.sbc_run`` builds its
    # template pack through THIS function, the ratified matched-configuration
    # SBC requirement was UNSATISFIABLE AT ANY PRICE for a padded pack.  It is
    # constructible now.
    if ntrue_edges is None:
        ntrue_edges = nhat_edges.copy()
    else:
        ntrue_edges = np.asarray(ntrue_edges, float)
        _validate_basis_pad(nhat_edges, ntrue_edges)

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
    # calibrated covariate range: the synthetic moment surfaces are exact
    # polynomials with no measured anchors, so the "fit range" is the whole
    # true-N grid and the fold's response clamp is INERT here by construction
    # (the clamp only bites where a real fit is extrapolated -- finding D2).
    resp_N_fit_range = np.broadcast_to(
        np.array([ntrue_edges[0], ntrue_edges[-1]]), (SR, ZR, 2)).copy()

    # --- transfer factors + prior widths
    t_true = np.zeros(KK) if t_true is None else np.asarray(t_true, float)
    if t_true.shape != (KK,):
        raise ValueError(f"t_true must have shape ({KK},)")
    t_sigma = (np.linspace(0.45, 0.20, KK) if t_sigma is None
               else np.asarray(t_sigma, float))

    # --- host-occlusion vector (restoration 2026-08-06). Default 0 (no
    # occlusion) keeps legacy synthetic expectations; pass a vector or scalar
    # to exercise the (1 - eta_c) factor through generator AND fold.
    if fp_eta is None:
        fp_eta_c = np.zeros(C, float)
    else:
        fp_eta_c = np.broadcast_to(np.asarray(fp_eta, float), (C,)).copy()
    if np.any(fp_eta_c < 0) or np.any(fp_eta_c >= 1):
        raise ValueError("synthetic_pack: fp_eta must satisfy 0 <= eta < 1")

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
        fp_w_sightline_ratio=fp_w, fp_E_alloc=E_alloc,
        fp_eta_c=fp_eta_c, t_sigma=t_sigma,
        truth_counts=np.zeros((B, Kf), dtype=np.int64),
        resp_fitcov_diag=resp_fitcov_diag,
        resp_N_ref=float(resp_N_ref),
        resp_N_fit_range=resp_N_fit_range,
    )

    psi_k_zero = np.zeros((2, SR, ZR))

    # --- FP intensity: scale so expected FP data counts = fp_frac * signal counts
    mu_signal = _fwd.fold_mu_reference(theta_true, psi_c_true, psi_k_zero,
                                       t_true, np.zeros((C, S)), pack)
    if fp_frac > 0:
        shape_cs = np.exp(-2.5 * (Nhat_c[:, None] - Nhat_c[0])) / (1.0 + 0.3 * np.arange(S))
        exp_t_alloc = (np.exp(t_true)[kz_to_K][:, None] * E_alloc).sum(axis=0)  # (S,)
        # 🔴 fp_ell_eff belongs here.  The generator must invert the SAME fold
        # the model uses: data-side mu_FP = fp_w * fp_ell_eff * lam * exp(t) * E
        # while the loa-0 side is fp_counts ~ Poisson(fp_ell_eff * lam).  Until
        # 2026-08-05 this line omitted fp_ell_eff -- the identical omission as
        # forward.fold_mu -- so every synthetic pack, every rung and every SBC
        # replica was generated under the SAME convention as the defect and
        # could not possibly detect it.  Whatever the fold does, this line must
        # do; ``fp_frac`` is defined as a share of the DATA-side counts.
        # (1 - fp_eta_c) weighting: the generator must invert the SAME fold
        # the model uses (restoration 2026-08-06) — the data-side FP carries
        # the host-occlusion survival per observed bin; the loa-0 calibration
        # side (fp_counts below) does not.
        fp_data_per_unit = fp_w * fp_ell_eff * (
            (1.0 - fp_eta_c)[:, None] * shape_cs * exp_t_alloc[None, :]).sum()
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
