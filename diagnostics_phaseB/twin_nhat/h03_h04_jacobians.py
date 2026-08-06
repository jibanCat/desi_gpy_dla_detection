#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-B bounded diagnosis (exploratory) — H3 (response-kernel shape, psi_k
span) and H4 (completeness vs true N, psi_c span).

PRE-STATED PREDICTED SIGNATURES, written BEFORE running:
  H3: a kernel-shape error produces a SMOOTH tilt concentrated at high N-hat
      (the D2-clamp/moment-extrapolation region: anchors top out at
      21.04-21.22, so G3 = [21.0,21.6] sits where the kernel is weakly
      measured/clamped).  Discriminant: the per-bin window residual lies
      substantially inside the span of the 18 psi_k Jacobian columns, with
      required magnitudes not absurdly far outside the prior sd, and the
      fitted directions REPLICATE on London-0/Saclay-0.
  H4: a completeness-shape error produces excess concentrated below
      N-hat ~ 20.0, aligned with molly-cell boundaries (edges at 19.5, 20.0,
      20.3, 20.5, 21.0, 21.5 inside/near the window).  Discriminant: same
      projection onto the psi_c Jacobian; required offsets in sigma_hat units.

This is a DIAGNOSTIC PROJECTION, NOT a fit adopted into the model.  No new
model freedom: psi_k_delta and psi_c are existing calibrated nuisances of the
committed fold; the projection only asks whether the residual lies in their
span.  CAVEAT stated up front: with 18 (psi_k) / ~48 live (psi_c) columns
against a 19-bin marginal, an unrestricted LS fit can saturate; the
morphology-match evidence is carried by (a) the explained fraction from the
LEADING SVD modes, (b) the required magnitudes in prior-sd units, and (c)
twin -> held-out replication by REFOLDING (exact, not linearized).
CAVEAT (prior sd): resp_fitcov_diag is None in these packs, so the prior sd
is the documented hard-coded fallback (0.02 dex mu0, 0.10 dex sig0).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/mfho/wt_repair_phaseB")

import jax                                                            # noqa: E402
import jax.numpy as jnp                                               # noqa: E402

from CDDF_analysis.hbi_mcmc.pack import load_pack                     # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu      # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS             # noqa: E402

PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")
OUT = "/home/mfho/wt_repair_phaseB/diagnostics_phaseB/twin_nhat"
WIN = (19.7, 21.6)


def setup(m):
    pk = load_pack(PACK.format(m=m))
    consts = build_consts(pk, resp_clamp="both")
    f = FS.truth_f(pk)
    theta = jnp.asarray(np.log(np.clip(f, 1e-300, None)))
    lam = jnp.asarray(np.asarray(pk.fp_counts, float) / float(pk.fp_ell_eff))
    live = jnp.asarray((np.asarray(pk.dX, float) > 0)[None, :, :])
    ne = np.asarray(pk.nhat_edges, float)
    win = (ne[:-1] >= WIN[0] - 1e-9) & (ne[1:] <= WIN[1] + 1e-9)
    win_idx = jnp.asarray(np.where(win)[0])
    S, M = consts.n_s, consts.n_molly
    SR, ZR, KK = consts.n_sr, consts.n_zr, consts.n_kk

    def perbin(pk_flat, pc_flat):
        mu = fold_mu(theta, pc_flat.reshape(S, M), pk_flat.reshape(2, SR, ZR),
                     jnp.zeros(KK), lam, consts)
        mu_c = jnp.where(live, mu, 0.0).sum(axis=(1, 2))
        return mu_c[win_idx]

    base = np.load(f"{OUT}/base_{m}_both.npz")
    resid = base["resid_c"][win]                       # obs - mu at the point
    var = base["mu_c"][win] + base["var_cal_c"][win]   # survey + cal diagonal
    return dict(pk=pk, consts=consts, perbin=perbin, resid=resid, var=var,
                win=win, n_kp=2 * SR * ZR, n_cp=S * M, S=S, M=M, SR=SR, ZR=ZR,
                theta=theta, lam=lam, live=live, win_idx=win_idx, base=base)


def wls(J, resid, var):
    """Weighted LS of resid on columns of J; returns dict of diagnostics."""
    w = 1.0 / np.sqrt(var)
    Jw = J * w[:, None]
    rw = resid * w
    chi2_0 = float(rw @ rw)
    U, sv, Vt = np.linalg.svd(Jw, full_matrices=False)
    tol = sv.max() * 1e-10 if sv.size else 0.0
    r = int((sv > tol).sum())
    beta = Vt[:r].T @ ((U[:, :r].T @ rw) / sv[:r])
    chi2_fit = float(rw @ rw - (U[:, :r].T @ rw) @ (U[:, :r].T @ rw))
    # explained fraction by the leading modes (data-independent SVD of Jw)
    proj = U.T @ rw
    frac_modes = {}
    for nm in (1, 2, 3, 5, r):
        nm2 = min(nm, r)
        frac_modes[f"top{nm}"] = float((proj[:nm2] @ proj[:nm2]) / chi2_0)
    return dict(beta=beta, chi2_0=chi2_0, chi2_resid=chi2_fit,
                explained_fraction=1.0 - chi2_fit / chi2_0,
                rank=r, singvals=sv.tolist(),
                cond=float(sv.max() / max(sv[sv > tol].min(), 1e-300)),
                frac_modes=frac_modes)


ctx = {m: setup(m) for m in ("2lpt0", "london0", "saclay0")}

report = {"label": "exploratory (prespecified discriminants, spec s7.3-7.4)",
          "prior_sd_caveat": "resp_fitcov_diag is None in the packs; prior sd "
                             "= hard-coded fallback (0.02 mu0, 0.10 sig0) dex",
          "note": "diagnostic projection, NOT a fit adopted into the model"}

tw = ctx["2lpt0"]
zk = jnp.zeros(tw["n_kp"])
zc = jnp.zeros(tw["n_cp"])

# ---- Jacobians at the plug-in point (twin) --------------------------------
J_k = np.asarray(jax.jacfwd(tw["perbin"], argnums=0)(zk, zc))   # (19, 18)
J_c = np.asarray(jax.jacfwd(tw["perbin"], argnums=1)(zk, zc))   # (19, 96)
np.savez(f"{OUT}/jacobians_2lpt0.npz", J_k=J_k, J_c=J_c)

# live psi_c columns (strata with dX>0 and molly cells actually fed)
live_c = np.where(np.abs(J_c).max(axis=0) > 1e-9)[0]
J_c_live = J_c[:, live_c]

# ---- H3: psi_k projection -------------------------------------------------
fit_k = wls(J_k, tw["resid"], tw["var"])
consts_tw = tw["consts"]
fitcov_sd = np.asarray(consts_tw.fitcov_sd)                     # (2, SR, ZR)
beta_k = fit_k["beta"].reshape(2, tw["SR"], tw["ZR"])
beta_k_units = beta_k / fitcov_sd

# twin -> held-out replication by exact refold
repl_k = {}
for m in ("london0", "saclay0"):
    c = ctx[m]
    mu_new = np.asarray(c["perbin"](jnp.asarray(fit_k["beta"]),
                                    jnp.zeros(c["n_cp"])))
    resid_new = (c["base"]["obs_c"][c["win"]] - mu_new)
    chi2_0 = float(np.sum(c["resid"] ** 2 / c["var"]))
    chi2_new = float(np.sum(resid_new ** 2 / c["var"]))
    repl_k[m] = dict(chi2_before=chi2_0, chi2_after=chi2_new,
                     explained_fraction=1.0 - chi2_new / chi2_0)
# and the twin refold itself (linearity check on the LS solution)
mu_tw_new = np.asarray(tw["perbin"](jnp.asarray(fit_k["beta"]), zc))
resid_tw_new = tw["base"]["obs_c"][tw["win"]] - mu_tw_new
chi2_tw_refold = float(np.sum(resid_tw_new ** 2 / tw["var"]))

report["H3_psi_k"] = dict(
    predicted_signature="smooth high-N-hat tilt inside the psi_k span; "
                        "replicates cross-mock",
    n_columns=int(J_k.shape[1]),
    fit=dict(explained_fraction=fit_k["explained_fraction"],
             chi2_before=fit_k["chi2_0"], chi2_after=fit_k["chi2_resid"],
             rank=fit_k["rank"], cond=fit_k["cond"],
             explained_by_leading_modes=fit_k["frac_modes"]),
    beta_mu0_dex=beta_k[0].tolist(), beta_sig0_dex=beta_k[1].tolist(),
    beta_in_prior_sd_units_mu0=beta_k_units[0].tolist(),
    beta_in_prior_sd_units_sig0=beta_k_units[1].tolist(),
    max_abs_beta_prior_units=float(np.abs(beta_k_units).max()),
    twin_refold_chi2=chi2_tw_refold,
    replication_refold=repl_k)

# ---- H4: psi_c projection -------------------------------------------------
fit_c = wls(J_c_live, tw["resid"], tw["var"])
sigma_hat = np.asarray(consts_tw.sigma_hat).reshape(-1)         # (S*M,)
beta_c_full = np.zeros(tw["n_cp"])
beta_c_full[live_c] = fit_c["beta"]
beta_c_units = np.where(sigma_hat > 0, beta_c_full / sigma_hat, 0.0)

repl_c = {}
for m in ("london0", "saclay0"):
    c = ctx[m]
    mu_new = np.asarray(c["perbin"](jnp.zeros(c["n_kp"]),
                                    jnp.asarray(beta_c_full)))
    resid_new = (c["base"]["obs_c"][c["win"]] - mu_new)
    chi2_0 = float(np.sum(c["resid"] ** 2 / c["var"]))
    chi2_new = float(np.sum(resid_new ** 2 / c["var"]))
    repl_c[m] = dict(chi2_before=chi2_0, chi2_after=chi2_new,
                     explained_fraction=1.0 - chi2_new / chi2_0)

molly_edges = np.asarray(tw["pk"].molly_nhi_edges, float)
report["H4_psi_c"] = dict(
    predicted_signature="excess below N-hat ~ 20.0 aligned with molly-cell "
                        "boundaries",
    n_live_columns=int(len(live_c)),
    fit=dict(explained_fraction=fit_c["explained_fraction"],
             chi2_before=fit_c["chi2_0"], chi2_after=fit_c["chi2_resid"],
             rank=fit_c["rank"], cond=fit_c["cond"],
             explained_by_leading_modes=fit_c["frac_modes"]),
    max_abs_beta_sigma_hat_units=float(np.abs(beta_c_units).max()),
    beta_sigma_hat_units_by_cell={
        f"s{si}_m{mi}[{molly_edges[mi]:.1f},{molly_edges[mi+1]:.1f})":
            float(beta_c_units[si * tw['M'] + mi])
        for si in range(tw["S"]) for mi in range(tw["M"])
        if abs(beta_c_units[si * tw["M"] + mi]) > 0.5},
    replication_refold=repl_c)

# ---- joint span + degeneracy on the 19-bin marginal -----------------------
J_joint = np.hstack([J_k, J_c_live])
fit_j = wls(J_joint, tw["resid"], tw["var"])
w = 1.0 / np.sqrt(tw["var"])
Qk, _ = np.linalg.qr(J_k * w[:, None])
Qc, _ = np.linalg.qr(J_c_live * w[:, None])
sv_angles = np.linalg.svd(Qk.T @ Qc, compute_uv=False)
report["joint"] = dict(
    explained_fraction=fit_j["explained_fraction"], rank=fit_j["rank"],
    cond=fit_j["cond"],
    principal_angle_cosines_top10=np.sort(sv_angles)[::-1][:10].tolist(),
    n_cosines_above_0p99=int(np.sum(sv_angles > 0.99)),
    note="cosine ~ 1 means the two nuisance spans are degenerate on this "
         "19-bin marginal")

with open(f"{OUT}/h03_h04_jacobians.json", "w") as fh:
    json.dump(report, fh, indent=1)

print(json.dumps({k: v for k, v in report.items()
                  if k in ("H3_psi_k", "H4_psi_c", "joint")}, indent=1,
                 default=str)[:6000])
