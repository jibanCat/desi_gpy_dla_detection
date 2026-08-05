"""legacy_oracle.py — LEGACY-side comparables for the Model A characterization gate.

Q3 spec §3 (the oracle rule): at FIXED calibration (Ψ_C = point, Ψ_K = point,
t = 0, FP off) the JAX fold μ(f) must reproduce the legacy ``A_ib``/``M_b``
expected-count construction on the same data pack to rtol 1e-6 — the LEGACY
builder is the validator, not the method. This module drives the SMALLEST
committed legacy entry points (``build_M_b``/``_apply_C_to_M`` for the M side,
``_build_A_ib_forward``/``_apply_C_to_A`` + ``ForwardResponseModel`` for the
A/kernel side) and maps their outputs onto the pack's (c, b, k, K, s) axes.

Convention mappings implemented here (each is a documented legacy⇄pack fact,
not a tunable):

  * f-units: the pack/fold population f_new[b,k] is a density per DEX
    (μ += f_new·ΔN_b, ΔN_b = 0.1). The legacy flat f is the CDDF per LINEAR N
    (M carries ΔN_lin = 10^hi − 10^lo per fine bin). Same physical bin mass ⇔
        f_leg[j,k] · ΔN_lin_j = f_new[b,k] · 0.1 .
  * dX placement: legacy ``build_M_b`` PX[s,kz] IS the pack's dX[k,s] (the pack
    was extracted from the same routine); the legacy A side carries dX/dz(ẑ)
    per detection instead (an f-independent factor in the marked-Poisson log
    term) — per-cell intensities divide it out and multiply dX[k,s].
  * response-cell registration: the fold conditions K on (stratum s, coarse
    z-bin K) via the stratum LOWER edge and the coarse-bin CENTER; the legacy
    A build conditions on each detection's OWN (SNR, z_QSO). Synthetic
    detections here pin (snr = stratum lower edge + 1e-9, zqso = coarse-bin
    center) so both sides address the SAME response cell; the residual
    within-stratum / within-bin mixing is a REPORTED convention difference.
  * REACH: ``_build_A_ib_forward`` truncates the response support at
    |x̂ − N_true| ≤ 2.0 dex; ``legacy_K_masses(reach=2.0)`` applies the same
    center-distance mask so mass- and density-level comparisons share it.

MOCKS ONLY. Nothing here is a fit path — read-only characterization.
"""
from __future__ import annotations

import dataclasses
import os
import types

import numpy as np
from scipy.special import expit
from scipy.stats import skewnorm as _skewnorm

from CDDF_analysis.hbi_mcmc.pack import ModelAPack
from CDDF_analysis.hbi_mcmc.forward import eta_hat_sigma_hat

__all__ = [
    "DEF_PACK", "DEF_FWD", "DEF_M_CACHE",
    "point_completeness", "f_new_to_f_leg", "legacy_K_masses",
    "K_from_pack_coeffs", "fold_with_K", "kernel_free_mu",
    "build_or_load_legacy_M", "legacy_M_expected_counts",
    "legacy_A_cell_intensity", "dxdz",
]

_SCRATCH = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata"
DEF_PACK = os.path.join(_SCRATCH, "modelA_packs", "modelA_pack_2lpt0.npz")
DEF_FWD = os.path.join(_SCRATCH, "track_c/stage0/forward_response_2lpt0.npz")
DEF_M_CACHE = os.path.join(_SCRATCH, "modelA_packs",
                           "legacy_oracle_M_2lpt0.npz")


# ---------------------------------------------------------------------------
# small shared pieces
# ---------------------------------------------------------------------------
def point_completeness(pack: ModelAPack) -> np.ndarray:
    """The FIXED-calibration completeness surface C(s, m) = expit(η̂) — the
    point (ψ_C = 0) of the model's Jeffreys-consistent logit surface. Fed to
    BOTH sides (the gate fixes the calibration; it does not compare calibration
    conventions)."""
    eta, _ = eta_hat_sigma_hat(pack.molly_n_det, pack.molly_n_tot)
    return expit(eta)


def _grids(pack: ModelAPack):
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    zf = np.asarray(pack.zf_edges, float)
    zc = np.asarray(pack.zc_edges, float)
    return ntrue, Nc, zf, zc


def dxdz(z, Omega_m: float = 0.279):
    """dX/dz — the EXACT expression of the legacy A builders (cddf_catalog_hbi
    ``Ez = sqrt(Ωm(1+z)^3 + (1−Ωm))``; ``dXdz = (1+z)^2/Ez``)."""
    z = np.asarray(z, float)
    Ez = np.sqrt(Omega_m * (1.0 + z) ** 3 + (1.0 - Omega_m))
    return (1.0 + z) ** 2 / Ez


def snr_rep(pack: ModelAPack) -> np.ndarray:
    """Per-stratum representative SNR = stratum lower edge + 1e-9 — the value
    whose response cell the fold's ``s_to_sresp`` map uses."""
    return np.asarray(pack.snr_edges, float)[:-1] + 1e-9


def zqso_rep(pack: ModelAPack) -> np.ndarray:
    """Per-coarse-bin representative z covariate = coarse-bin center — the
    value whose response cell the fold's ``K_to_zresp`` map uses. NOTE the
    covariate-convention finding: the frozen model was fit with
    z_covariate='zqso' (the QUASAR z); the fold conditions on the ABSORBER's
    coarse z bin."""
    _, _, _, zc = _grids(pack)
    return 0.5 * (zc[:-1] + zc[1:])


def f_new_to_f_leg(f_new: np.ndarray, pack: ModelAPack,
                   logN_lo: np.ndarray, logN_hi: np.ndarray,
                   n_zf: int) -> np.ndarray:
    """Map the pack-grid per-dex population f_new[b,k] to the legacy flat
    per-linear-N f (length n_nbins·n_zf), zero outside the pack window.

    Same bin mass: f_leg_j · (10^hi_j − 10^lo_j) = f_new_b · ΔN_b(dex)."""
    ntrue, _, zf, _ = _grids(pack)
    dN_dex = np.diff(ntrue)
    f_new = np.asarray(f_new, float)
    n_j = len(logN_lo)
    if f_new.shape != (len(dN_dex), n_zf):
        raise ValueError(f"f_new shape {f_new.shape} != ({len(dN_dex)}, {n_zf})")
    f_leg = np.zeros((n_j, n_zf))
    for j in range(n_j):
        hit = np.where(np.abs(np.asarray(logN_lo)[j] - ntrue[:-1]) < 1e-6)[0]
        if hit.size != 1:
            continue                       # legacy bin outside the pack window
        b = int(hit[0])
        if abs(logN_hi[j] - ntrue[b + 1]) > 1e-6:
            raise ValueError("legacy fine bin does not tile the pack bin")
        dN_lin = 10.0 ** logN_hi[j] - 10.0 ** logN_lo[j]
        f_leg[j, :] = f_new[b, :] * dN_dex[b] / dN_lin
    return f_leg.reshape(-1)


# ---------------------------------------------------------------------------
# kernel-side comparables
# ---------------------------------------------------------------------------
def legacy_K_masses(pack: ModelAPack, frm, reach: float | None = None
                    ) -> np.ndarray:
    """K_leg[s, K, c, b]: EXACT observed-bin masses of the legacy forward
    response — ``frm.response_skewnormal`` (the committed moment→(ξ,ω,a) map,
    σ-clip, skew clamp and (1−ramp) collapse) integrated with the exact scipy
    skew-normal CDF over the pack's N̂ bins, at the fold's representative
    (stratum, coarse-z) covariates.

    ``reach=2.0`` masks |N̂c_center − N_true,center| > reach (the committed
    ``_build_A_ib_forward`` REACH truncation) for like-for-like A comparisons.
    """
    nhat = np.asarray(pack.nhat_edges, float)
    _, Nc, _, _ = _grids(pack)
    S = len(pack.snr_edges) - 1
    KK = len(pack.zc_edges) - 1
    C = len(nhat) - 1
    B = len(Nc)
    srep, zrep = snr_rep(pack), zqso_rep(pack)
    out = np.zeros((S, KK, C, B))
    for s in range(S):
        for K in range(KK):
            xi, om, a = frm.response_skewnormal(
                Nc, np.full(B, srep[s]), np.full(B, zrep[K]))
            for b in range(B):
                cdf = _skewnorm.cdf(nhat, a[b], loc=xi[b], scale=om[b])
                out[s, K, :, b] = np.diff(cdf)
    if reach is not None:
        chat = 0.5 * (nhat[:-1] + nhat[1:])
        mask = np.abs(chat[:, None] - Nc[None, :]) <= float(reach)
        out = out * mask[None, None, :, :]
    return out


def legacy_K_density(pack: ModelAPack, frm, reach: float | None = 2.0
                     ) -> np.ndarray:
    """Kd[s, K, c, b]: the legacy forward response DENSITY p(x̂_c-center | N_b)
    via the committed ``frm.response_density``, at the fold's representative
    covariates, with the committed REACH mask. ``fold_with_K(pack, Kd, f)``
    then reproduces the legacy ``A_ib·f`` construction at bin-center
    detections (density-level; the f-independent dX/dz(ẑ) factor replaced by
    dX[k,s] as in ``legacy_A_cell_intensity``)."""
    nhat = np.asarray(pack.nhat_edges, float)
    _, Nc, _, _ = _grids(pack)
    chat = 0.5 * (nhat[:-1] + nhat[1:])
    S = len(pack.snr_edges) - 1
    KK = len(pack.zc_edges) - 1
    srep, zrep = snr_rep(pack), zqso_rep(pack)
    out = np.zeros((S, KK, len(chat), len(Nc)))
    for s in range(S):
        for K in range(KK):
            for b, Nb in enumerate(Nc):
                out[s, K, :, b] = frm.response_density(
                    chat, np.full(len(chat), Nb),
                    np.full(len(chat), srep[s]), np.full(len(chat), zrep[K]))
    if reach is not None:
        mask = np.abs(chat[:, None] - Nc[None, :]) <= float(reach)
        out = out * mask[None, None, :, :]
    return out


def K_from_pack_coeffs(pack: ModelAPack, n_ref: float, sig_mode: str,
                       ramp_mode: str, param_mode: str,
                       clamp_mode: str = "off") -> np.ndarray:
    """Kernel masses from the PACK coefficient block under selectable
    conventions — the finding-decomposition instrument.

    sig_mode  : 'softplus' (PRE-FIX: σ = floor + softplus(poly)) | 'clip'
                (committed: σ = clip(poly, floor); finding F2).
    ramp_mode : 'sigmoid' (PRE-FIX: γ·logistic((N−c)/w)) | 'legacy'
                (committed: γ clamped to ±0.995·skew_max,
                ·(1 − clip((N−c)/w, 0, 1)); finding F3).
    param_mode: 'direct' (PRE-FIX: (ξ,ω,a) = (N+μpoly, σ, γ)) | 'moment'
                (committed: moment-match via _moment_to_skewnormal_vec;
                finding F4).
    clamp_mode: 'off' (DEFAULT, and what this function did before 2026-08-05:
                the moment polynomials are evaluated at the UNCLAMPED bin
                centre) | 'both' | 'hi' (finding D2: the covariate is clamped
                to each response cell's calibrated ``resp_N_fit_range``, the
                same three modes ``forward.build_consts`` takes).

    🔴 CORRECTION (2026-08-05). This docstring used to say that
    ``(midpoint n_ref, 'softplus', 'sigmoid', 'direct')`` reproduces the fold's
    ``build_K`` and that the all-legacy tuple reproduces ``legacy_K_masses``.
    THE MAPPING WAS EXACTLY INVERTED, and had been since findings F1–F4 were
    FIXED — at which point ``build_K`` adopted the legacy conventions and the
    'softplus/sigmoid/direct' tuple became the PRE-FIX recipe it no longer runs.
    MEASURED on ``synthetic_pack(0, **small_test_grid())`` (n_ref = 20.0, the
    grid midpoint, which is also this pack's ``resp_N_ref``):

        ('softplus', 'sigmoid', 'direct') : max|diff vs build_K| = 2.811e-01
        ('clip',     'legacy',  'moment') : max|diff vs build_K| = 4.441e-16

    So: the COMMITTED tuple is ('clip', 'legacy', 'moment') at the pack's own
    ``resp_N_ref``, and it reproduces BOTH ``build_K`` and ``legacy_K_masses``,
    because after the F1–F4 fixes those two are the same recipe. The pinning
    test is ``tests/test_modelA_vs_legacy.py::test_B3_*``; the historical
    per-mechanism ladder is ``test_Tk2_decomposition_table``.

    D2, the second half of the same defect: until 2026-08-05 this function
    implemented NO covariate clamp at all, so the only cross-convention kernel
    instrument on the path evaluated the UNCLAMPED kernel while the fold it was
    being compared against clamps by default (``resp_clamp='both'``). The clamp
    is now a selectable convention like the other three. Default 'off'
    PRESERVES this function's historical behaviour — the ladder in
    ``test_Tk2_decomposition_table`` reads the same numbers as before — so a
    caller comparing against a clamped fold must ASK for the clamp.
    """
    from scipy.special import ndtr, owens_t
    from CDDF_analysis.hbi.znz_kernel import (_moment_to_skewnormal_vec,
                                              _SN_SKEW_MAX)
    if clamp_mode not in ("off", "both", "hi"):
        raise ValueError(f"clamp_mode must be 'off'|'both'|'hi', "
                         f"got {clamp_mode!r}")
    if clamp_mode != "off" and pack.resp_N_fit_range is None:
        raise ValueError(
            "K_from_pack_coeffs: clamp_mode=%r needs pack.resp_N_fit_range, "
            "which this pack does not carry (schema v1 / pre-2026-07-28). "
            "Re-extract, or pass clamp_mode='off' and READ the result as the "
            "unclamped kernel it is." % clamp_mode)
    nhat = np.asarray(pack.nhat_edges, float)
    ntrue, Nc, _, zc = _grids(pack)
    mu_c = np.asarray(pack.resp_mu_coef, float)
    sg_c = np.asarray(pack.resp_sig_coef, float)
    sk_c = np.asarray(pack.resp_skew_coef, float)
    D = mu_c.shape[-1]
    S = len(pack.snr_edges) - 1
    KK = len(zc) - 1
    Cn = len(nhat) - 1
    B = len(Nc)
    rse = np.asarray(pack.resp_snr_edges, float)
    rze = np.asarray(pack.resp_z_edges, float)
    s2sr = np.clip(np.searchsorted(rse, snr_rep(pack), side="right") - 1,
                   0, mu_c.shape[0] - 1)
    K2zr = np.clip(np.searchsorted(rze, zqso_rep(pack), side="right") - 1,
                   0, mu_c.shape[1] - 1)
    ramp_c, ramp_w = [float(v) for v in np.asarray(pack.resp_skew_ramp, float)]
    floor = float(pack.resp_sig_floor)
    u = Nc - float(n_ref)
    rr = (None if pack.resp_N_fit_range is None
          else np.asarray(pack.resp_N_fit_range, float))      # (SR, ZR, 2)
    out = np.zeros((S, KK, Cn, B))
    for s in range(S):
        for K in range(KK):
            sr, zr = int(s2sr[s]), int(K2zr[K])
            # finding D2: the fitted polynomials' COVARIATE is clamped to the
            # response cell's calibrated range; the ramp and the bin centre
            # stay on the UNCLAMPED Nc (exactly forward.build_K's split — the
            # clamp guards the polynomials' covariate, not the physical N of
            # the bin).
            if clamp_mode == "off":
                u_c = u
            elif clamp_mode == "both":
                u_c = np.clip(Nc, rr[sr, zr, 0], rr[sr, zr, 1]) - float(n_ref)
            else:                                             # "hi"
                u_c = np.minimum(Nc, rr[sr, zr, 1]) - float(n_ref)
            mpoly = sum(mu_c[sr, zr, d] * u_c ** d for d in range(D))
            spoly = sum(sg_c[sr, zr, d] * u_c ** d for d in range(D))
            kpoly = sum(sk_c[sr, zr, d] * u_c ** d for d in range(D))
            if sig_mode == "softplus":
                sig = floor + np.logaddexp(0.0, spoly)
            elif sig_mode == "clip":
                sig = np.clip(spoly, floor, None)
            else:
                raise ValueError(sig_mode)
            if ramp_mode == "sigmoid":
                gam = kpoly * expit((Nc - ramp_c) / ramp_w)
            elif ramp_mode == "legacy":
                gam = np.clip(kpoly, -0.995 * _SN_SKEW_MAX, 0.995 * _SN_SKEW_MAX)
                gam = gam * (1.0 - np.clip((Nc - ramp_c) / ramp_w, 0.0, 1.0))
            else:
                raise ValueError(ramp_mode)
            if param_mode == "direct":
                xi, om, a = Nc + mpoly, sig, gam
            elif param_mode == "moment":
                xi, om, a = _moment_to_skewnormal_vec(Nc + mpoly, sig, gam)
            else:
                raise ValueError(param_mode)
            for b in range(B):
                z_ = (nhat - xi[b]) / om[b]
                cdf = ndtr(z_) - 2.0 * owens_t(z_, a[b])
                out[s, K, :, b] = np.diff(cdf)
    return out


# ---------------------------------------------------------------------------
# fold-structure comparables (test-local numpy contractions, LABELED as such)
# ---------------------------------------------------------------------------
def _contrib(pack: ModelAPack, f_new: np.ndarray) -> np.ndarray:
    """contrib[b, k, s] = C_point[s, cell(b)] · g[cell(b), k] · f_new[b,k] · ΔN_b
    — the kernel-independent inner factor of the fold, at point calibration."""
    ntrue, Nc, zf, _ = _grids(pack)
    me = np.asarray(pack.molly_nhi_edges, float)
    b2cell = np.clip(np.searchsorted(me, Nc, side="right") - 1, 0, len(me) - 2)
    Cp = point_completeness(pack)                     # (S, M)
    g = np.asarray(pack.g_grid, float)                # (M, Kf)
    dN = np.diff(ntrue)
    f_new = np.asarray(f_new, float)
    return (Cp[:, b2cell].T[:, None, :] * g[b2cell, :][:, :, None]
            * f_new[:, :, None] * dN[:, None, None])  # (B, Kf, S)


def fold_with_K(pack: ModelAPack, K_skcb: np.ndarray, f_new: np.ndarray
                ) -> np.ndarray:
    """μ[c,k,s] = dX[k,s] · Σ_b K[s,K(k),c,b] · contrib[b,k,s] — the fold
    expression with an INJECTED kernel-mass table (FP off). Mirrors ``fold_mu``'s
    contraction exactly; used to isolate kernel ingestion from fold structure."""
    kz2K = np.asarray(pack.kz_to_K, int)
    contrib = _contrib(pack, f_new)                   # (B, Kf, S)
    dX = np.asarray(pack.dX, float)                   # (Kf, S)
    Kfull = K_skcb[:, kz2K]                           # (S, Kf, C, B)
    mu = np.einsum("skcb,bks->cks", Kfull, contrib)
    return mu * dX[None, :, :]


def kernel_free_mu(pack: ModelAPack, f_new: np.ndarray) -> np.ndarray:
    """tot[b,k,s] = dX[k,s]·contrib[b,k,s] — the fold at UNIT kernel mass
    (every detection counted wherever N̂ lands). The object the legacy M·f
    expected-count construction measures."""
    contrib = _contrib(pack, f_new)
    return contrib * np.asarray(pack.dX, float)[None, :, :]


# ---------------------------------------------------------------------------
# legacy M side (heavy: per-sightline pathlength through committed machinery)
# ---------------------------------------------------------------------------
def build_or_load_legacy_M(pack: ModelAPack, mock: str = "2lpt0",
                           cache_path: str = DEF_M_CACHE) -> dict:
    """Per-stratum legacy selection normalizer M_s[j, kz] via the COMMITTED
    ``build_M_b`` + ``_apply_C_to_M`` (3-D g-threaded path), at the fixed point
    calibration C3[s,m,kz] = expit(η̂)[s,m]·g[m,kz]. Heavy (the ~374k-sightline
    qso lookup + analytic per-fine-z pathlength), so cached to NPZ.

    Returns dict(M_s (S, n_j, n_zf), logN_lo, logN_hi, PX (S, n_zf)).
    The stratum decomposition zeroes all C rows but s — ``_apply_C_to_M`` is
    linear in C, so Σ_s M_s equals the committed full-C M identically.
    """
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    from CDDF_analysis.hbi_mcmc import extract_pack as EP
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        HBIConfig, _build_qso_lookup, _fine_z_grid, build_fine_grid,
        build_M_b, build_pathlength, load_molly_matrix, _apply_C_to_M)

    cfg = EP._make_cfg(mock, os.path.dirname(cache_path))
    mm = load_molly_matrix(cfg.molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)
    X_tot, n_sl, qzl, qzh, qsn, Xcalc = build_pathlength(
        cfg, qso_lookup=qso_lookup, return_per_sl=True)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    zf = _fine_z_grid(cfg)
    if not np.allclose(zf, np.asarray(pack.zf_edges, float)):
        raise RuntimeError("legacy fine z grid != pack zf_edges")
    M_meta = build_M_b(qzl, qzh, qsn, mm, logN_lo, logN_hi, N_b, dN_b,
                       zf, Xcalc, cfg)
    PX = np.asarray(M_meta["PX"], float)              # (S, n_zf)

    Cp = point_completeness(pack)                     # (S, M)
    g = np.asarray(pack.g_grid, float)                # (M, n_zf)
    n_j, n_zf = len(logN_lo), len(zf) - 1
    S = Cp.shape[0]
    M_s = np.zeros((S, n_j, n_zf))
    for s in range(S):
        C3 = np.zeros((S, Cp.shape[1], n_zf))
        C3[s] = Cp[s][:, None] * g                    # the _build_C_nz_3d product
        M_s[s] = _apply_C_to_M(M_meta, C3).reshape(n_j, n_zf)

    out = dict(M_s=M_s, logN_lo=np.asarray(logN_lo, float),
               logN_hi=np.asarray(logN_hi, float), PX=PX,
               n_sl=np.array(float(n_sl)))
    np.savez(cache_path, **out)
    return out


def legacy_M_expected_counts(pack: ModelAPack, Mres: dict, f_new: np.ndarray
                             ) -> np.ndarray:
    """Legacy TOTAL expected detected counts per (b, k, s) on the pack grid:
    M_s[j,k]·f_leg[j,k] restricted to the pack window (all N̂ — no window)."""
    ntrue, _, zf, _ = _grids(pack)
    n_zf = len(zf) - 1
    logN_lo = np.asarray(Mres["logN_lo"], float)
    logN_hi = np.asarray(Mres["logN_hi"], float)
    f_leg = f_new_to_f_leg(f_new, pack, logN_lo, logN_hi, n_zf
                           ).reshape(len(logN_lo), n_zf)
    M_s = np.asarray(Mres["M_s"], float)              # (S, n_j, n_zf)
    tot_full = M_s * f_leg[None, :, :]                # (S, n_j, n_zf)
    # gather the pack-window rows onto the b axis
    B = len(ntrue) - 1
    out = np.zeros((B, n_zf, M_s.shape[0]))
    for b in range(B):
        j = np.where(np.abs(logN_lo - ntrue[b]) < 1e-6)[0]
        if j.size != 1:
            raise RuntimeError(f"pack bin {b} not found on the legacy grid")
        out[b] = tot_full[:, int(j[0]), :].T
    return out                                        # (B, Kf, S)


# ---------------------------------------------------------------------------
# legacy A side (the committed forward A builder on synthetic detections)
# ---------------------------------------------------------------------------
def legacy_A_cell_intensity(pack: ModelAPack, f_new: np.ndarray,
                            fwd_path: str = DEF_FWD,
                            gl_nodes: int = 0) -> np.ndarray:
    """Per-(c,k,s) expected-count comparable from the COMMITTED legacy A
    builder ``_build_A_ib_forward`` + ``_apply_C_to_A`` on synthetic detections.

    gl_nodes == 0 : one detection per (c,k,s) at the N̂-bin CENTER; returns the
        legacy intensity DENSITY comparable
            D[c,k,s] = (A_i·f_leg / (dX/dz)(z_k)) · dX[k,s]
        (per unit x̂ — compare against the density-level fold contraction).
    gl_nodes >= 2 : Gauss-Legendre nodes per N̂ bin; returns the x̂-INTEGRATED
        μ_leg[c,k,s] (compare against mass-level fold_with_K(legacy_K(reach=2))
        where the response σ makes the quadrature valid).

    Strata with zero pathlength (dX column identically 0 — the op-mask SNR≤2
    strata) are returned as 0 without building rows.
    """
    from CDDF_analysis.hbi_mcmc import extract_pack as EP
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        _build_A_ib_forward, _apply_C_to_A, build_fine_grid, _fine_z_grid,
        load_molly_matrix)

    cfg = dataclasses.replace(EP._make_cfg("2lpt0", "/tmp"),
                              kernel_forward_model=fwd_path)
    mm = load_molly_matrix(cfg.molly_tsv)
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    zf = _fine_z_grid(cfg)
    n_zf = len(zf) - 1

    nhat = np.asarray(pack.nhat_edges, float)
    Cn = len(nhat) - 1
    zfc = 0.5 * (zf[:-1] + zf[1:])
    kz2K = np.asarray(pack.kz_to_K, int)
    srep, zrep = snr_rep(pack), zqso_rep(pack)
    dX = np.asarray(pack.dX, float)                   # (Kf, S)
    S = dX.shape[1]
    live_s = np.where(dX.sum(axis=0) > 0)[0]

    if gl_nodes and gl_nodes >= 2:
        gx, gw = np.polynomial.legendre.leggauss(int(gl_nodes))
    else:
        gx, gw = np.array([0.0]), np.array([np.nan])  # center point, no weight

    # synthetic detections: (c-node, k, s) triplets
    xs, zs, ss, iis, zq, cs, ks, sidx, wts = [], [], [], [], [], [], [], [], []
    for s in live_s:
        for k in range(n_zf):
            K = int(kz2K[k])
            for c in range(Cn):
                lo, hi = nhat[c], nhat[c + 1]
                nodes = 0.5 * (hi + lo) + 0.5 * (hi - lo) * gx
                w = 0.5 * (hi - lo) * gw
                for x, wi in zip(nodes, w):
                    xs.append(x); zs.append(zfc[k]); ss.append(srep[s])
                    iis.append(s); zq.append(zrep[K])
                    cs.append(c); ks.append(k); sidx.append(s); wts.append(wi)
    cat_op = dict(xhat=np.array(xs), zhat=np.array(zs),
                  sig_z=np.zeros(len(xs)), snr=np.array(ss),
                  i_snr=np.array(iis, int), zqso=np.array(zq))
    A_unit, meta = _build_A_ib_forward(cat_op, mm, logN_lo, logN_hi, zf, cfg)
    Cp = point_completeness(pack)
    g = np.asarray(pack.g_grid, float)
    C3 = Cp[:, :, None] * g[None, :, :]               # (S, M, n_zf)
    A = _apply_C_to_A(meta, C3)
    f_leg = f_new_to_f_leg(f_new, pack, logN_lo, logN_hi, n_zf)
    lam = np.asarray(A @ f_leg).ravel()               # per-detection intensity

    out = np.zeros((Cn, n_zf, S))
    cs = np.array(cs); ks = np.array(ks); sidx = np.array(sidx)
    wts = np.array(wts)
    conv = dX[ks, sidx] / dxdz(zfc[ks], cfg.Omega_m)  # dXdz(ẑ) → dX[k,s]
    if gl_nodes and gl_nodes >= 2:
        np.add.at(out, (cs, ks, sidx), lam * wts * conv)
    else:
        out[cs, ks, sidx] = lam * conv
    return out
