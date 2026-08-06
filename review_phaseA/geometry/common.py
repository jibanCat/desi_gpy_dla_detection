"""REVIEW-ONLY (Phase A) — independent geometric-identifiability referee probe.

Independent implementation of the pad-vs-FP degeneracy geometry:

  * Jacobians of the PRODUCTION fold (forward.fold_mu) are obtained by
    AUTODIFF (jax.linearize / jax.jacfwd), never hand-coded.  This is
    independent of the archived probe's jac.py AND automatically settles the
    fp_operator convention question — autodiff returns the true d mu / d lam
    including the fp_ell_eff exposure factor.
  * Principal angles use an SVD-based orthonormalization (the archived probe
    used QR) — same mathematics, independent numerics.
  * The reference point is rebuilt from scratch: window truth from the
    production e4_probe.truth_f (permitted reuse), the sub-floor pad by a
    power-law continuation re-implemented here (continuity at the floor bin,
    NOT the archived probe's intercept convention), scaled through the actual
    production fold so the folded pad / FP totals hit the requested (T_A, T_B).

Coordinates: theta = log f (fold-native, scale-free); FP block in log lam
(columns d mu/d log lam = lam * d mu/d lam; span-identical to raw-lam columns
wherever lam > 0 — principal angles depend only on the span).

Do NOT import anything from the archived probe.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from CDDF_analysis.hbi_mcmc.pack import load_pack, coarsen_basis          # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu, fold_mu_fp  # noqa: E402
from CDDF_analysis.hbi_mcmc import e4_probe as E4                         # noqa: E402

PACKS = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
         "PRESERVED_2026-07-28_small_artifacts/modelA_packs")
MOCKS = ("2lpt0", "london0", "saclay0")


def get_pack(mock, basis_width=0.1, pad_floor=19.0):
    p = load_pack(f"{PACKS}/modelA_pack_{mock}_v11.npz")
    return coarsen_basis(p, basis_width, pad_floor=pad_floor)


# ---------------------------------------------------------------------------
# reference-point construction (own implementation)
# ---------------------------------------------------------------------------

def _fold_sig(theta, consts):
    """Signal-only fold (lam = 0) at nuisance-zero."""
    S, M = consts.n_s, consts.n_molly
    return fold_mu(jnp.asarray(theta), jnp.zeros((S, M)),
                   jnp.zeros((2, consts.n_sr, consts.n_zr)),
                   jnp.zeros(consts.n_kk), jnp.zeros((consts.n_c, consts.n_s)),
                   consts)


def live_mask(pack, consts):
    dX = np.asarray(pack.dX, float)
    return np.broadcast_to((dX > 0)[None, :, :],
                           (consts.n_c, consts.n_k, consts.n_s))


THETA_DEAD = -2000.0   # exp(-2000) == 0.0 exactly in float64; dead bins


def build_truth(pack, consts, T_A, T_B, *, pad_slope=None, fp_shape="n0",
                lam_floor=1e-4):
    """f (B, Kf) and lam (C, S) with PRODUCTION-fold totals (T_A, T_B).

    pad: log f continued below the floor as a power law fitted to the 5 lowest
    window bins of each z column; if pad_slope is given the slope is replaced
    and the continuation kept CONTINUOUS at the lowest window bin (this
    intentionally differs from the archived probe, which kept the fitted
    intercept).  The pad is then rescaled so the folded pad-only total over
    live cells equals T_A — the scaling is done through forward.fold_mu, not
    through any hand-built operator.

    fp_shape: "n0" (proportional to pack.fp_counts) or "flat" (uniform over
    the (c, s) cells the fold can respond to).  lam is scaled so the folded
    FP-only total over live cells equals T_B, then floored at
    lam_floor * mean(positive lam) (the archived probe's convention, kept so
    the all-columns variant is comparable; the floor's effect is measured in
    experiment 2).  Set lam_floor=None to skip flooring.
    """
    live = live_mask(pack, consts)
    npad = pack.n_pad_bins
    B, Kf = pack.n_b, pack.n_k
    C, S = pack.n_c, pack.n_s
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])

    f = np.asarray(E4.truth_f(pack, consts=consts), float).copy()   # (B, Kf)
    lo, hi = npad, npad + 5
    for k in range(Kf):
        y = f[lo:hi, k]
        m = y > 0
        if m.sum() >= 2:
            s, b0 = np.polyfit(Nc[lo:hi][m], np.log(y[m]), 1)
        else:
            s, b0 = -1.5, 0.0
        if pad_slope is not None:
            # keep the continuation continuous at the lowest window bin
            anchor = (np.log(y[0]) if y[0] > 0 else b0 + s * Nc[lo])
            s = float(pad_slope)
            b0 = anchor - s * Nc[lo]
        f[:npad, k] = np.exp(b0 + s * Nc[:npad])

    # scale the pad through the production fold
    theta_pad = np.full((B, Kf), THETA_DEAD)
    pos = f[:npad] > 0
    theta_pad[:npad][pos] = np.log(f[:npad][pos])
    tot_pad = float(np.asarray(_fold_sig(theta_pad, consts))[live].sum())
    if tot_pad > 0:
        f[:npad] *= T_A / tot_pad

    # FP shape and scale through the production FP fold
    if fp_shape == "n0":
        sh = np.asarray(pack.fp_counts, float).copy()               # (C, S)
    elif fp_shape == "flat":
        E = np.asarray(pack.fp_E_alloc, float)                      # (Kf, S)
        resp = (E.sum(axis=0) > 0)                                  # (S,)
        sh = np.repeat(resp[None, :].astype(float), C, axis=0)
    else:
        raise ValueError(fp_shape)
    if sh.sum() <= 0:
        raise ValueError("empty FP shape")
    lam = sh / sh.sum()
    tot_fp = float(np.asarray(
        fold_mu_fp(jnp.zeros(consts.n_kk), jnp.asarray(lam), consts))[live].sum())
    lam *= T_B / tot_fp
    if lam_floor is not None:
        lam = np.maximum(lam, lam_floor * lam[lam > 0].mean())
    return f, lam


def theta_from_f(f):
    th = np.full(f.shape, THETA_DEAD)
    pos = f > 0
    th[pos] = np.log(f[pos])
    return th


# ---------------------------------------------------------------------------
# autodiff Jacobians of the production fold
# ---------------------------------------------------------------------------

def _chunked_jac(fun, x0, chunk=256):
    """(n_out, n_par) Jacobian of fun at x0 by linearize + vmapped JVPs."""
    x0 = jnp.asarray(x0)
    _, lin = jax.linearize(fun, x0)
    n = int(x0.size)
    shp = x0.shape

    def jvp_flat(v):
        return lin(v.reshape(shp)).ravel()

    batched = jax.jit(jax.vmap(jvp_flat))
    cols = []
    for i in range(0, n, chunk):
        e = np.zeros((min(chunk, n - i), n))
        e[np.arange(e.shape[0]), i + np.arange(e.shape[0])] = 1.0
        cols.append(np.asarray(batched(jnp.asarray(e))))
    return np.concatenate(cols, axis=0).T                     # (n_out, n_par)


ALL_BLOCKS = ("theta", "psi_c", "psi_k", "log_t", "lam_raw")


def all_jacobians(pack, consts, theta0, lam0, *, psi_c0=None, psi_k0=None,
                  t0=None, blocks=ALL_BLOCKS):
    """mu (C,Kf,S) + autodiff Jacobians of fold_mu wrt every requested block.

    Blocks: theta (B*Kf), psi_c (S*M), psi_k (2*SR*ZR), log_t (KK),
    lam_raw = d mu/d lam (C*S).  When lam_raw is computed, loglam
    (= lam * lam_raw; d mu/d log lam) is derived too — zero columns at lam==0.
    """
    S, M = consts.n_s, consts.n_molly
    psi_c0 = jnp.zeros((S, M)) if psi_c0 is None else jnp.asarray(psi_c0)
    psi_k0 = (jnp.zeros((2, consts.n_sr, consts.n_zr)) if psi_k0 is None
              else jnp.asarray(psi_k0))
    t0 = jnp.zeros(consts.n_kk) if t0 is None else jnp.asarray(t0)
    th0 = jnp.asarray(theta0)
    lm0 = jnp.asarray(lam0)

    mu = np.asarray(fold_mu(th0, psi_c0, psi_k0, t0, lm0, consts))
    funs = {
        "theta": (lambda th: fold_mu(th, psi_c0, psi_k0, t0, lm0, consts),
                  th0, 256),
        "psi_c": (lambda pc: fold_mu(th0, pc, psi_k0, t0, lm0, consts),
                  psi_c0, 256),
        "psi_k": (lambda pk_: fold_mu(th0, psi_c0, pk_, t0, lm0, consts),
                  psi_k0, 6),
        "log_t": (lambda tt: fold_mu(th0, psi_c0, psi_k0, tt, lm0, consts),
                  t0, 256),
        "lam_raw": (lambda lm: fold_mu(th0, psi_c0, psi_k0, t0, lm, consts),
                    lm0, 256),
    }
    J = {}
    for name in blocks:
        fun, x0, chunk = funs[name]
        J[name] = _chunked_jac(fun, x0, chunk=chunk)
    if "lam_raw" in J:
        J["loglam"] = J["lam_raw"] * np.asarray(lam0).ravel()[None, :]
    return mu, J


def fd_check(pack, consts, theta0, lam0, J, mu, *, seed=0, rel=1e-6):
    """Central finite differences of fold_mu against every autodiff block."""
    rng = np.random.default_rng(seed)
    S, M = consts.n_s, consts.n_molly
    z_c = np.zeros((S, M))
    z_k = np.zeros((2, consts.n_sr, consts.n_zr))
    z_t = np.zeros(consts.n_kk)

    def fold(th, pc, pk_, tt, lm):
        return np.asarray(fold_mu(jnp.asarray(th), jnp.asarray(pc),
                                  jnp.asarray(pk_), jnp.asarray(tt),
                                  jnp.asarray(lm), consts))
    out = {}
    for name, base, which in [("theta", theta0, 0), ("psi_c", z_c, 1),
                              ("psi_k", z_k, 2), ("log_t", z_t, 3),
                              ("lam_raw", lam0, 4)]:
        v = rng.standard_normal(np.shape(base))
        scale = rel * max(1.0, float(np.abs(base).max()))
        args = [np.asarray(theta0, float).copy(), z_c.copy(), z_k.copy(),
                z_t.copy(), np.asarray(lam0, float).copy()]
        ap = [a.copy() for a in args]; am = [a.copy() for a in args]
        ap[which] = ap[which] + scale * v
        am[which] = am[which] - scale * v
        fd = (fold(*ap) - fold(*am)) / (2.0 * scale)
        ad = (J[name] @ v.ravel()).reshape(fd.shape)
        den = max(float(np.abs(fd).max()), 1e-30)
        out[name] = float(np.abs(ad - fd).max() / den)
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def orthonormal_basis(X, tol=1e-11):
    """Orthonormal basis of col-span(X) via SVD (rank-revealing)."""
    if X.shape[1] == 0:
        return X
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > tol * max(s[0], 1e-300)]


def principal_angles(X, Y, tol=1e-11):
    """Ascending principal angles (deg) between col-spans; also dims."""
    Qx = orthonormal_basis(X, tol)
    Qy = orthonormal_basis(Y, tol)
    if Qx.shape[1] == 0 or Qy.shape[1] == 0:
        return np.array([]), Qx.shape[1], Qy.shape[1]
    s = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s)), Qx.shape[1], Qy.shape[1]


def angle_summary(X, Y):
    ang, nx, ny = principal_angles(X, Y)
    if ang.size == 0:
        return {"n_x": nx, "n_y": ny, "empty": True}
    return {
        "n_x": int(nx), "n_y": int(ny),
        "min_deg": float(ang[0]),
        "smallest8": [float(a) for a in ang[:8]],
        "q25_deg": float(np.percentile(ang, 25)),
        "q50_deg": float(np.percentile(ang, 50)),
        "q75_deg": float(np.percentile(ang, 75)),
        "n_lt0p1": int((ang < 0.1).sum()),
        "n_lt1": int((ang < 1).sum()),
        "n_lt5": int((ang < 5).sum()),
        "n_lt10": int((ang < 10).sum()),
        "n_lt20": int((ang < 20).sum()),
    }


def informative(Dcols, tol=1e-10):
    n = np.linalg.norm(Dcols, axis=0)
    return n > tol * max(n.max(), 1e-300)


def whitened_blocks(pack, consts, mu, J, *, live=None):
    """Rows = live cells weighted 1/sqrt(mu); returns dict of design blocks."""
    if live is None:
        live = live_mask(pack, consts)
    lv = live.ravel()
    w = 1.0 / np.sqrt(np.clip(mu.ravel()[lv], 1e-300, None))
    out = {}
    for name in ("theta", "psi_c", "psi_k", "log_t", "loglam", "lam_raw"):
        if name in J:
            out[name] = J[name][lv] * w[:, None]
    npad, Kf = pack.n_pad_bins, pack.n_k
    padmask = np.zeros(out["theta"].shape[1], bool)
    padmask[:npad * Kf] = True     # theta stored (B, Kf) row-major: b*Kf + k
    out["pad"] = out["theta"][:, padmask]
    out["window"] = out["theta"][:, ~padmask]
    return out, w, lv
