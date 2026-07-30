"""E4 — conditioning of the Model A deconvolution (DIAGNOSTIC ONLY).

The Model A fold ``forward.fold_mu`` is EXACTLY LINEAR in ``f = exp(theta_pop)``
at fixed nuisances::

    mu[c, k, s] = sum_b A[c, k, s, b] * f[b, k]      (+ the FP term)

with

    A[c, k, s, b] = K[s, K(k), c, b] * C[s, cell(b)] * g[b, k] * dN_b * dX[k, s]

and the fold is BLOCK-DIAGONAL in the fine-z index k (nothing in the sum over b
mixes k).  This module recovers ``A`` **by probing the committed fold with
one-hot basis vectors** — it does not reimplement any of the physics.  Every
number it reports therefore inherits whatever the production fold does,
including the D2 covariate clamp.

Scope limit (2026-07-29): this module is a DIAGNOSTIC.  It never writes a pack,
never changes the basis width, the prior or the estimator.  Choosing a basis
width changes the estimand's resolution and is a PI decision.

Contents
--------
build_fold_operator   probe the committed fold -> A[c, k, s, b]
operator_matrix       flatten A to the per-k design matrix (rows = live (c,s))
spectrum              singular values / condition number / effective ranks
null_directions       small-singular-value right vectors, described in log N_HI
merge_basis_columns   merge adjacent true-N basis bins (coarser basis)
truth_f               the pack's own truth f[b, k] on the pack's own grid
nnls_invert           non-negative least squares inversion of a folded signal
map_invert_rw2        MAP inversion under the model's own RW2 prior on log f
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

try:  # jax is present in the gpdla-hbi env; keep the import local-ish
    import jax
    import jax.numpy as jnp
except Exception as _e:  # pragma: no cover - env guard
    jax = None
    jnp = None

from .forward import build_consts, fold_mu
from .pack import ModelAPack
# The basis-MERGING convention (E4's own, unchanged) now lives in reporting.py so
# that the jax-free extractor can use the SAME implementation instead of a second
# copy (PI decision 3 made the 0.2-dex basis a first-class pack option).  These
# three names stay part of e4_probe's public surface.
from .reporting import basis_groups, merge_basis_columns, merged_truth  # noqa: F401

# theta value whose exp() underflows to EXACTLY 0.0 in float64, so a one-hot
# probe carries no leakage from the other basis bins (no subtraction needed).
_THETA_OFF = -1000.0


# --------------------------------------------------------------------------
# 1. recover the operator by probing the committed fold
# --------------------------------------------------------------------------

def build_fold_operator(pack: ModelAPack, *, resp_clamp: str = "both",
                        consts=None) -> np.ndarray:
    """Return A[c, k, s, b] by probing ``fold_mu`` with one-hot f.

    The nuisances are held at their prior-central values (psi_c = 0,
    psi_k_delta = 0, log t = 0) and the FP intensity at zero, so the returned
    operator is the signal fold alone.
    """
    if jnp is None:  # pragma: no cover
        raise RuntimeError("e4_probe needs jax (use the gpdla-hbi env)")
    if consts is None:
        consts = build_consts(pack, resp_clamp=resp_clamp)
    C, Kf, S, B = consts.n_c, consts.n_k, consts.n_s, consts.n_b
    psi_c = jnp.zeros((S, consts.n_molly))
    psi_k = jnp.zeros((2, consts.n_sr, consts.n_zr))
    log_t = jnp.zeros(consts.n_kk)
    lam_fp = jnp.zeros((C, S))

    # sanity: the all-off probe must fold to exactly zero
    off = jnp.full((B, Kf), _THETA_OFF)
    base = np.asarray(fold_mu(off, psi_c, psi_k, log_t, lam_fp, consts))
    if not np.all(base == 0.0):
        raise RuntimeError(
            "build_fold_operator: the all-off probe did not fold to exactly "
            f"zero (max {np.abs(base).max():.3e}) — the one-hot probe would "
            "carry leakage; refuse to build a contaminated operator.")

    A = np.empty((C, Kf, S, B), dtype=float)
    for b in range(B):
        th = np.full((B, Kf), _THETA_OFF)
        th[b, :] = 0.0                      # f[b, k] = 1 for every k
        A[..., b] = np.asarray(
            fold_mu(jnp.asarray(th), psi_c, psi_k, log_t, lam_fp, consts))
    return A


def check_linearity(pack: ModelAPack, A: np.ndarray, *, seed: int = 0,
                    resp_clamp: str = "both", consts=None) -> float:
    """Max relative deviation between ``A @ f`` and the committed fold.

    Returns the max relative error over the cells with nonzero mu.  A value at
    float64 round-off confirms the probe recovered the operator exactly.
    """
    if consts is None:
        consts = build_consts(pack, resp_clamp=resp_clamp)
    rng = np.random.default_rng(seed)
    B, Kf = consts.n_b, consts.n_k
    theta = rng.normal(-40.0, 2.0, size=(B, Kf))
    f = np.exp(theta)
    mu_probe = np.einsum("cksb,bk->cks", A, f)
    mu_fold = np.asarray(fold_mu(
        jnp.asarray(theta), jnp.zeros((consts.n_s, consts.n_molly)),
        jnp.zeros((2, consts.n_sr, consts.n_zr)), jnp.zeros(consts.n_kk),
        jnp.zeros((consts.n_c, consts.n_s)), consts))
    m = mu_fold > 0
    if not np.any(m):
        raise RuntimeError("check_linearity: fold produced no positive cells")
    return float(np.max(np.abs(mu_probe[m] - mu_fold[m]) / mu_fold[m]))


def live_rows(pack: ModelAPack, k: int) -> np.ndarray:
    """Boolean (C, S) mask of the cells the likelihood actually sees at fine-z k.

    ``model_a`` masks out every stratum with dX == 0 (structurally unobserved).
    """
    dX = np.asarray(pack.dX, float)
    C = pack.n_c
    return np.broadcast_to((dX[k] > 0)[None, :], (C, pack.n_s)).copy()


def operator_matrix(A: np.ndarray, pack: ModelAPack, k: int,
                    *, stratum: Optional[int] = None) -> np.ndarray:
    """Flatten A at fine-z bin ``k`` to a design matrix (rows, B).

    ``stratum=None`` stacks every LIVE (dX > 0) SNR stratum — the matrix the
    likelihood at that z actually inverts.  ``stratum=s`` returns the single
    stratum's (C, B) block.
    """
    if stratum is None:
        m = live_rows(pack, k)
        return A[:, k, :, :][m]                      # (n_live_rows, B)
    return A[:, k, stratum, :]


# --------------------------------------------------------------------------
# 2. spectra
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Spectrum:
    sv: np.ndarray            # singular values, descending
    cond: float               # sv[0] / sv[-1]  (inf if sv[-1] == 0)
    rank_thresholds: dict     # {threshold: effective rank}
    numerical_rank: int       # np.linalg.matrix_rank default tol
    n_rows: int
    n_cols: int

    def as_dict(self):
        return dict(singular_values=[float(v) for v in self.sv],
                    cond=float(self.cond),
                    effective_rank=self.rank_thresholds,
                    numerical_rank=int(self.numerical_rank),
                    n_rows=int(self.n_rows), n_cols=int(self.n_cols))


_RANK_TOLS = (1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)


def spectrum(M: np.ndarray, tols: Sequence[float] = _RANK_TOLS) -> Spectrum:
    """Singular value spectrum, condition number and effective ranks of M."""
    M = np.asarray(M, float)
    sv = np.linalg.svd(M, compute_uv=False)
    s0 = sv[0] if sv.size else 0.0
    cond = float(s0 / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")
    ranks = {f"{t:g}": int(np.sum(sv > t * s0)) for t in tols}
    return Spectrum(sv=sv, cond=cond, rank_thresholds=ranks,
                    numerical_rank=int(np.linalg.matrix_rank(M)),
                    n_rows=M.shape[0], n_cols=M.shape[1])


def null_directions(M: np.ndarray, centers: np.ndarray, *, n: int = 4) -> list:
    """Describe the ``n`` smallest-singular-value right vectors in log N_HI.

    Each entry reports the singular value, its ratio to sigma_0, the number of
    sign changes across the true-N basis (an oscillation count), the implied
    node spacing in dex, and the basis bins carrying the most weight.
    """
    M = np.asarray(M, float)
    _, sv, Vt = np.linalg.svd(M, full_matrices=False)
    centers = np.asarray(centers, float)
    out = []
    for j in range(1, n + 1):
        v = Vt[-j]
        s = float(sv[-j])
        sgn = np.sign(v)
        nz = sgn[sgn != 0]
        n_sign_changes = int(np.sum(nz[1:] != nz[:-1]))
        span = float(centers[-1] - centers[0]) if centers.size > 1 else 0.0
        node_dex = float(span / n_sign_changes) if n_sign_changes else float("inf")
        top = np.argsort(-np.abs(v))[:5]
        out.append(dict(
            index_from_smallest=j - 1,
            singular_value=s,
            ratio_to_sigma0=float(s / sv[0]) if sv[0] > 0 else float("inf"),
            n_sign_changes=n_sign_changes,
            node_spacing_dex=node_dex,
            top_bins_logNHI=[float(centers[i]) for i in top],
            top_bin_weights=[float(v[i]) for i in top],
            vector=[float(x) for x in v],
        ))
    return out


# --------------------------------------------------------------------------
# 3. basis-width sweep (merge adjacent true-N basis bins)
# --------------------------------------------------------------------------

# basis_groups / merge_basis_columns / merged_truth are IMPORTED from
# reporting.py (see the import block at the top of this module).  They were
# defined here originally; they moved so the jax-free extractor could build a
# 0.2-dex pack with the SAME merging convention rather than a second copy.  The
# behaviour is unchanged and `tests/test_e4_probe.py` still exercises them
# through this module's names.


# --------------------------------------------------------------------------
# 4. the pack's own truth, on the pack's own grid
# --------------------------------------------------------------------------

def truth_f(pack: ModelAPack, consts=None) -> np.ndarray:
    """f_true[b, k] implied by ``pack.truth_counts`` and the fold's own weights.

    The fold's intrinsic (pre-selection) expected count in (b, k, s) is
    ``f[b,k] * g[b,k] * dN_b * dX[k,s]``; summing over s and matching
    ``truth_counts[b,k]`` inverts to the expression below.  Cells with zero
    z-shape or zero exposure return 0.

    DIVERGENCE FROM ``forward_selftest.truth_f`` (documented 2026-07-29, after
    a referee flagged the silent difference).  The two functions are NOT the
    same quantity and neither is wrong:

      forward_selftest.truth_f  =  truth_counts / (dX_tot * dN)          [1]
      e4_probe.truth_f          =  truth_counts / (dX_tot * dN * g_bk)   [2]

    so ``e4_probe.truth_f * g_bk == forward_selftest.truth_f`` EXACTLY (pinned
    by ``test_truth_f_divergence_from_forward_selftest_is_exactly_g_bk``).
    [1] is the physical CDDF per dex per dX.  [2] is the fold's own population
    coordinate: the vector ``f`` for which ``A @ f`` reproduces the pack's
    counts, because the fold applies ``g_bk`` itself
    (``forward.fold_mu``: ``contrib = C_bs * g_bk * f * dN_b``).  Feeding [1]
    into the operator recovered by ``build_fold_operator`` would double-count
    the z-shape.  The two agree identically wherever ``g_grid == 1`` (which is
    the case for ``pack.synthetic_pack``), so a test built only on the
    synthetic pack cannot see the difference — the pinning test therefore
    overrides ``g_grid`` to a non-unit shape on purpose.
    """
    if consts is None:
        consts = build_consts(pack)
    tc = np.asarray(pack.truth_counts, float)               # (B, Kf)
    dN = np.asarray(consts.dN_b)                            # (B,)
    g = np.asarray(consts.g_bk)                             # (B, Kf)
    dXk = np.asarray(pack.dX, float).sum(axis=1)            # (Kf,)
    den = g * dN[:, None] * dXk[None, :]
    out = np.zeros_like(tc)
    m = den > 0
    out[m] = tc[m] / den[m]
    return out


# --------------------------------------------------------------------------
# 5. inversions
# --------------------------------------------------------------------------

def nnls_invert(M: np.ndarray, y: np.ndarray, *,
                weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Non-negative least squares  min_{x>=0} || W(Mx - y) ||_2 .

    The active-set solver can hit its iteration cap on exactly the
    ill-conditioned systems this module studies (the active set cycles when
    several columns are near-degenerate); the bounded trust-region solver is
    used as a fallback so a conditioning failure is never silently reported as
    a crash.
    """
    from scipy.optimize import nnls, lsq_linear
    M = np.asarray(M, float)
    y = np.asarray(y, float)
    if weights is not None:
        w = np.asarray(weights, float)[:, None]
        M = M * w
        y = y * w[:, 0]
    try:
        x, _ = nnls(M, y, maxiter=200 * M.shape[1])
        return x
    except RuntimeError:
        res = lsq_linear(M, y, bounds=(0.0, np.inf), method="trf",
                         tol=1e-14, max_iter=500)
        return np.asarray(res.x, float)


def d2_matrix(n: int) -> np.ndarray:
    """Second-difference operator D2 (n-2, n) — the model's RW2 curvature."""
    D = np.zeros((max(n - 2, 0), n))
    for i in range(max(n - 2, 0)):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    return D


def map_invert_rw2(M: np.ndarray, counts: np.ndarray, *, sigma_N: float,
                   theta_init: Optional[np.ndarray] = None,
                   maxiter: int = 3000):
    """Poisson MAP for log f under the model's own RW2 curvature prior.

    Minimises   sum_rows [ mu - y log mu ]  +  0.5 || D2 theta ||^2 / sigma_N^2
    with mu = M exp(theta).  This is exactly Model A's population block with
    every nuisance frozen at its prior centre and the level/slope left free
    (their production priors are deliberately weak: N(0, 4) / N(0, 2)).
    """
    if jnp is None:  # pragma: no cover
        raise RuntimeError("map_invert_rw2 needs jax (use the gpdla-hbi env)")
    from scipy.optimize import minimize
    M_j = jnp.asarray(np.asarray(M, float))
    y_j = jnp.asarray(np.asarray(counts, float))
    D2 = jnp.asarray(d2_matrix(M.shape[1]))
    inv_var = 1.0 / (sigma_N ** 2)

    def obj(theta):
        mu = M_j @ jnp.exp(theta)
        mu = jnp.clip(mu, 1e-300, None)
        nll = jnp.sum(mu - y_j * jnp.log(mu))
        pen = 0.5 * inv_var * jnp.sum((D2 @ theta) ** 2)
        return nll + pen

    val_grad = jax.jit(jax.value_and_grad(obj))

    def fun(x):
        v, g = val_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    if theta_init is None:
        theta_init = np.full(M.shape[1], np.log(max(counts.sum(), 1.0)
                                                / max(M.sum(), 1e-30)))
    res = minimize(fun, np.asarray(theta_init, float), jac=True,
                   method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=10 * maxiter,
                                ftol=1e-15, gtol=1e-12))
    return np.exp(np.asarray(res.x, float)), res


def ratio_profile(f_hat: np.ndarray, f_true: np.ndarray,
                  floor_frac: float = 1e-12) -> np.ndarray:
    """f_hat / f_true, with bins whose truth is negligible set to NaN."""
    f_true = np.asarray(f_true, float)
    ref = floor_frac * np.max(f_true) if np.max(f_true) > 0 else 0.0
    out = np.full_like(f_true, np.nan)
    m = f_true > ref
    out[m] = np.asarray(f_hat, float)[m] / f_true[m]
    return out
