"""REVIEW-ONLY (Phase A) — experiment 4: prior curvature (Laplace) in
PRODUCTION sampled coordinates (2lpt0).

Reparameterize exactly as model_a.model_a samples (non-centered RW theta via
level/slope/eps_N/eps_z with sigma_N = sigma_N_scale = 0.5 and sigma_z =
sigma_z_scale = 0.5 held at their prior scales; psi_c/psi_k/t standardized by
their prior sds; FP via (log lam_total, zero-sum shape u) with
v = fp_shape_sd * H u, H an orthonormal zero-sum basis).  In these
coordinates the prior Hessian is the identity on every standardized block,
plus the (negligible) Gamma(0.5, 1e-6) curvature b*lam_total on log
lam_total.

Posterior Hessian ~ J^T J (Poisson-Fisher-whitened survey likelihood)
[+ loa-0 anchor Fisher] [+ production prior].  Reported: marginal sds of the
ESTIMANDS log10 T_A (folded pad total) and log10 T_B (folded survey-FP
total), their correlation, and the null-space component of each estimand
gradient (a nonzero null component under a singular Hessian means the
projected sd is only a LOWER BOUND — the direction is structurally free).

Every Jacobian/gradient here is autodiff of the production fold composed with
the coordinate map — nothing hand-coded.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (get_pack, build_truth, theta_from_f, live_mask,      # noqa: E402
                    _chunked_jac, THETA_DEAD)
from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu, fold_mu_fp  # noqa: E402

import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402

T_A = 24000.0
FP_LEVELS = {"TB_1086p7_prerepair": 1086.7, "TB_14768_anchor": 14768.0}

# production prior scales (ModelAConfig defaults / model_a signature)
SIGMA_N = 0.5      # sigma_N held at its HalfNormal prior scale
SIGMA_Z = 0.5      # sigma_z held at its HalfNormal prior scale
LEVEL_SCALE = 4.0
SLOPE_SCALE = 2.0
FP_SHAPE_SD = 3.0
FP_EPS_RATE = 1e-6

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "out_exp4.json")
LN10 = float(np.log(10.0))


def make_maps(pack, consts):
    B, Kf = pack.n_b, pack.n_k
    S, M = consts.n_s, consts.n_molly
    C = consts.n_c
    SR, ZR, KK = consts.n_sr, consts.n_zr, consts.n_kk
    CS = C * S
    sizes = [1, 1, B - 2, B * (Kf - 1), S * M, 2 * SR * ZR, KK, 1, CS - 1]
    splits = np.cumsum(sizes)[:-1]
    sig_hat = jnp.asarray(consts.sigma_hat)
    fitcov = jnp.asarray(consts.fitcov_sd)
    t_sig = jnp.asarray(consts.t_sigma)
    # orthonormal basis of the zero-sum subspace of R^CS
    Pc = np.eye(CS) - np.ones((CS, CS)) / CS
    Uc, sc, _ = np.linalg.svd(Pc)
    H = jnp.asarray(Uc[:, :CS - 1])                        # (CS, CS-1)
    b_idx = jnp.arange(B) - 0.5 * (B - 1)

    def blocks(c):
        (lv_s, sl_s, eps_N, eps_z, pc_s, pk_s, t_s, llt,
         u) = jnp.split(c, splits)
        level = LEVEL_SCALE * lv_s[0]
        slope = SLOPE_SCALE * sl_s[0]
        # verbatim production construction (model_a.model_a), sigma at scale
        curv = jnp.cumsum(jnp.cumsum(
            jnp.concatenate([jnp.zeros(2), eps_N])))[:B]
        theta_col0 = level + slope * b_idx + SIGMA_N * curv
        theta = theta_col0[:, None] + jnp.concatenate(
            [jnp.zeros((B, 1)),
             SIGMA_Z * jnp.cumsum(eps_z.reshape(B, Kf - 1), axis=1)], axis=1)
        psi_c = sig_hat * pc_s.reshape(S, M)
        psi_k = fitcov * pk_s.reshape(2, SR, ZR)
        t = t_sig * t_s
        v = FP_SHAPE_SD * (H @ u)
        lam = (jnp.exp(llt[0]) * jax.nn.softmax(v)).reshape(C, S)
        return theta, psi_c, psi_k, t, lam

    def coords_from_ref(theta_ref, lam_ref):
        """Invert the (bijective) map at psi = 0, t = 0."""
        th = np.asarray(theta_ref, float)
        col0 = th[:, 0]
        bi = np.asarray(b_idx)
        # curv[0] = curv[1] = 0  ->  2x2 solve for level/slope
        Amat = np.array([[1.0, bi[0]], [1.0, bi[1]]])
        level, slope = np.linalg.solve(Amat, col0[:2])
        curv = (col0 - level - slope * bi) / SIGMA_N
        eps_N = np.diff(curv, 2)                            # (B-2,)
        eps_z = np.diff(th, axis=1) / SIGMA_Z               # (B, Kf-1)
        lam = np.asarray(lam_ref, float).ravel()
        llt = np.log(lam.sum())
        vv = np.log(lam / lam.sum())
        vv = vv - vv.mean()
        u = np.asarray(H).T @ vv / FP_SHAPE_SD
        c = np.concatenate([[level / LEVEL_SCALE], [slope / SLOPE_SCALE],
                            eps_N, eps_z.ravel(),
                            np.zeros(S * M), np.zeros(2 * SR * ZR),
                            np.zeros(KK), [llt], u])
        return c

    return blocks, coords_from_ref, sizes


def block_labels(sizes):
    names = ["level", "slope", "eps_N", "eps_z", "psi_c", "psi_k", "t",
             "log_lam_total", "fp_shape_u"]
    lab = []
    for n, s in zip(names, sizes):
        lab += [n] * s
    return np.array(lab)


def marginals(H, G, labels, tol=1e-12):
    """2x2 marginal covariance of (ln T_A, ln T_B) from Hessian H (may be
    singular): eigendecompose, invert the supported spectrum, report the
    null-space component of each gradient."""
    e, V = np.linalg.eigh(H)
    emax = float(e.max())
    keep = e > tol * emax
    Gt = V.T @ G                                            # (n, 2)
    C2 = (Gt[keep] / e[keep, None]).T @ Gt[keep]
    null_frac = []
    null_power = []
    for j in range(G.shape[1]):
        g = G[:, j]
        pn = V[:, ~keep] @ Gt[~keep, j]
        null_frac.append(float(np.linalg.norm(pn) / np.linalg.norm(g)))
        if np.linalg.norm(pn) > 1e-12 * np.linalg.norm(g):
            w = pn / np.linalg.norm(pn)
            null_power.append({lb: float((w[labels == lb] ** 2).sum())
                               for lb in np.unique(labels)
                               if (w[labels == lb] ** 2).sum() > 1e-3})
        else:
            null_power.append({})
    sd = np.sqrt(np.clip(np.diag(C2), 0, None))
    corr = float(C2[0, 1] / max(sd[0] * sd[1], 1e-300))
    return {
        "sd_lnTA_dex": float(sd[0] / LN10),
        "sd_lnTB_dex": float(sd[1] / LN10),
        "corr_TA_TB": corr,
        "null_frac_gA": null_frac[0], "null_frac_gB": null_frac[1],
        "null_power_gA": null_power[0], "null_power_gB": null_power[1],
        "n_null_dirs": int((~keep).sum()),
        "eig_min_kept_over_max": float(e[keep].min() / emax),
    }


def main():
    pack = get_pack("2lpt0")
    consts = build_consts(pack)
    live = jnp.asarray(live_mask(pack, consts))
    npad, Kf, B = pack.n_pad_bins, pack.n_k, pack.n_b
    padmask_bk = jnp.asarray(
        (np.arange(B) < npad)[:, None] * np.ones((1, Kf), bool))
    blocks, coords_from_ref, sizes = make_maps(pack, consts)
    labels = block_labels(sizes)
    ell = float(consts.fp_ell_eff)

    def mu_of(c):
        th, pc, pk_, t, lam = blocks(c)
        return fold_mu(th, pc, pk_, t, lam, consts)

    def logTA(c):
        th, pc, pk_, t, lam = blocks(c)
        th_pad = jnp.where(padmask_bk, th, THETA_DEAD)
        mu_pad = fold_mu(th_pad, pc, pk_, t, jnp.zeros_like(lam), consts)
        return jnp.log(jnp.sum(jnp.where(live, mu_pad, 0.0)))

    def logTB(c):
        th, pc, pk_, t, lam = blocks(c)
        mu_fp = fold_mu_fp(t, lam, consts)
        return jnp.log(jnp.sum(jnp.where(live, mu_fp, 0.0)))

    def anchor_mu(c):
        _, _, _, _, lam = blocks(c)
        return ell * lam.ravel()

    res = {"T_A": T_A, "coord_sizes": dict(zip(
        ["level", "slope", "eps_N", "eps_z", "psi_c", "psi_k", "t",
         "log_lam_total", "fp_shape_u"], [int(s) for s in sizes])),
        "prior_scales": {"sigma_N": SIGMA_N, "sigma_z": SIGMA_Z,
                         "level": LEVEL_SCALE, "slope": SLOPE_SCALE,
                         "fp_shape_sd": FP_SHAPE_SD,
                         "fp_eps_rate": FP_EPS_RATE},
        "note_sigma_fixed": "sigma_N/sigma_z held at their HalfNormal prior "
                            "scales (conditional Laplace)"}

    for tag, T_B in FP_LEVELS.items():
        t0 = time.time()
        f, lam_ref = build_truth(pack, consts, T_A, T_B)
        theta_ref = theta_from_f(f)
        c0 = jnp.asarray(coords_from_ref(theta_ref, lam_ref))
        n = int(c0.size)

        # reconstruction check
        th, pc, pk_, t, lam = blocks(c0)
        rec = {
            "theta_max_abs_err": float(jnp.abs(th - theta_ref).max()),
            "lam_max_rel_err": float(jnp.abs(
                lam - jnp.asarray(lam_ref)).max() / np.asarray(lam_ref).max()),
            "nuisance_max": float(max(jnp.abs(pc).max(), jnp.abs(pk_).max(),
                                      jnp.abs(t).max()))}

        # whitened likelihood design in sampled coordinates
        mu0 = np.asarray(mu_of(c0))
        lv = np.asarray(live).ravel()
        w = 1.0 / np.sqrt(np.clip(mu0.ravel()[lv], 1e-300, None))
        Jc = _chunked_jac(mu_of, c0, chunk=8)[lv] * w[:, None]
        H_lik = Jc.T @ Jc

        # loa-0 anchor Fisher
        m0 = np.asarray(anchor_mu(c0))
        Jm = _chunked_jac(anchor_mu, c0, chunk=256)
        H_anc = Jm.T @ (Jm / np.clip(m0, 1e-300, None)[:, None])

        # production prior curvature (identity on standardized blocks;
        # Gamma(1/2, eps) contributes b * lam_total on log lam_total)
        H_pri = np.eye(n)
        i_llt = int(np.flatnonzero(labels == "log_lam_total")[0])
        H_pri[i_llt, i_llt] = FP_EPS_RATE * float(np.asarray(lam_ref).sum())

        gA = np.asarray(jax.grad(logTA)(c0))
        gB = np.asarray(jax.grad(logTB)(c0))
        G = np.stack([gA, gB], axis=1)

        o = {"T_B": T_B, "reconstruction": rec,
             "n_coords": n, "n_live": int(lv.sum()),
             "anchor_expected_total": float(m0.sum()),
             "TA_check": float(np.exp(logTA(c0))),
             "TB_check": float(np.exp(logTB(c0)))}
        for case, H in [
                ("a_likelihood_only", H_lik),
                ("b_plus_anchor", H_lik + H_anc),
                ("c2_lik_plus_priors_no_anchor", H_lik + H_pri),
                ("c_full_anchor_plus_priors", H_lik + H_anc + H_pri)]:
            o[case] = marginals(H, G, labels)
            m = o[case]
            print("  %-30s sd(lgTA) %9.4f dex  sd(lgTB) %9.4f dex  "
                  "corr %+.3f  null gA %.2e gB %.2e (%d null dirs)" % (
                      case, m["sd_lnTA_dex"], m["sd_lnTB_dex"],
                      m["corr_TA_TB"], m["null_frac_gA"], m["null_frac_gB"],
                      m["n_null_dirs"]), flush=True)
        res[tag] = o
        print("== %s done (%.1fs)  TA %.0f TB %.1f anchor-total %.1f" % (
            tag, time.time() - t0, o["TA_check"], o["TB_check"],
            o["anchor_expected_total"]), flush=True)

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("EXP4 DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
