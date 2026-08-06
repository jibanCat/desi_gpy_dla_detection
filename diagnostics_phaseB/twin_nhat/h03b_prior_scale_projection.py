#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-B bounded diagnosis (exploratory) — H3/H4 supplement: projection at
the CALIBRATED prior scale.

WHY THIS EXISTS, stated before the numbers: the unrestricted LS projections
(h03_h04_jacobians.py) saturate (18 and ~48 columns against 19 bins) and the
minimum-norm solutions demand psi_k up to ~320 prior-sd / psi_c up to ~3e5
sigma-hat — magnitudes at which the linearization is invalid (the twin refold
at the fitted psi_k gives chi2 7458 against a 178 baseline).  The question the
discriminant actually cares about is: can the EXISTING calibrated nuisances,
AT THEIR CALIBRATED SCALE, absorb a material fraction of the residual?  That
is the ridge solution with the model's OWN prior widths (fitcov_sd fallback
0.02/0.10 for psi_k; sigma_hat for psi_c) — no new prior is introduced; the
existing ones are used as the scale.  Every solution is verified by an EXACT
REFOLD (not the linear prediction), in the survey+calibration diagonal metric.

PRE-STATED PREDICTION: if the kernel-shape (or completeness-shape) mechanism
is the cause at a plausible magnitude, the ridge solution absorbs a material
fraction (>50%) of the with-cal window chi2 on the twin AND, refolded on
London-0/Saclay-0 unchanged, absorbs a comparable fraction there.
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
    theta = jnp.asarray(np.log(np.clip(FS.truth_f(pk), 1e-300, None)))
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
        return jnp.where(live, mu, 0.0).sum(axis=(1, 2))[win_idx]

    base = np.load(f"{OUT}/base_{m}_both.npz")
    resid = base["resid_c"][win]
    var = base["mu_c"][win] + base["var_cal_c"][win]     # survey + cal diag
    return dict(consts=consts, perbin=perbin, resid=resid, var=var, win=win,
                obs=base["obs_c"][win], n_kp=2 * SR * ZR, n_cp=S * M)


ctx = {m: setup(m) for m in ("2lpt0", "london0", "saclay0")}
tw = ctx["2lpt0"]
jac = np.load(f"{OUT}/jacobians_2lpt0.npz")
J_k, J_c = jac["J_k"], jac["J_c"]
live_c = np.where(np.abs(J_c).max(axis=0) > 1e-9)[0]

sd_k = np.asarray(tw["consts"].fitcov_sd).reshape(-1)                # (18,)
sd_c = np.asarray(tw["consts"].sigma_hat).reshape(-1)[live_c]        # live

W = 1.0 / tw["var"]
r = tw["resid"]
chi2_0 = float(np.sum(r * r * W))


def ridge(J, sd, tau=1.0):
    """argmin ||r - J b||^2_W + ||b/(tau*sd)||^2; returns b."""
    A = (J.T * W) @ J + np.diag(1.0 / (tau * sd) ** 2)
    return np.linalg.solve(A, (J.T * W) @ r)


def evaluate(name, b_k, b_c_full):
    """Exact refold on all three mocks at (b_k, b_c); explained fractions."""
    out = {}
    for m, c in ctx.items():
        mu_new = np.asarray(c["perbin"](jnp.asarray(b_k),
                                        jnp.asarray(b_c_full)))
        rn = c["obs"] - mu_new
        c2_0 = float(np.sum(c["resid"] ** 2 / c["var"]))
        c2_n = float(np.sum(rn ** 2 / c["var"]))
        out[m] = dict(chi2_before=c2_0, chi2_after=c2_n,
                      explained_fraction=1.0 - c2_n / c2_0)
    return out


report = {"label": "exploratory (supplement to spec s7.3-7.4 discriminants)",
          "metric": "survey + calibration diagonal (delta method, exact for "
                    "the linear FP fold)",
          "chi2_before_twin": chi2_0, "cases": {}}

zeros_c = np.zeros(tw["n_cp"])
zeros_k = np.zeros(tw["n_kp"])

# psi_k at 1x and 3x the calibrated prior scale
for tau in (1.0, 3.0):
    b = ridge(J_k, sd_k, tau)
    ev = evaluate(f"psi_k tau={tau}", b, zeros_c)
    report["cases"][f"psi_k_tau{tau:g}"] = dict(
        max_abs_b_prior_units=float(np.abs(b / sd_k).max()),
        rms_b_prior_units=float(np.sqrt(np.mean((b / sd_k) ** 2))),
        refold=ev)

# psi_c at 1x and 3x sigma_hat
for tau in (1.0, 3.0):
    bc = ridge(J_c[:, live_c], sd_c, tau)
    b_full = np.zeros(tw["n_cp"]); b_full[live_c] = bc
    ev = evaluate(f"psi_c tau={tau}", zeros_k, b_full)
    report["cases"][f"psi_c_tau{tau:g}"] = dict(
        max_abs_b_sigma_hat_units=float(np.abs(bc / sd_c).max()),
        rms_b_sigma_hat_units=float(np.sqrt(np.mean((bc / sd_c) ** 2))),
        refold=ev)

# joint at 1x
J_j = np.hstack([J_k, J_c[:, live_c]])
sd_j = np.concatenate([sd_k, sd_c])
bj = ridge(J_j, sd_j, 1.0)
b_full = np.zeros(tw["n_cp"]); b_full[live_c] = bj[len(sd_k):]
ev = evaluate("joint tau=1", bj[:len(sd_k)], b_full)
report["cases"]["joint_tau1"] = dict(
    max_abs_b_prior_units=float(np.abs(bj / sd_j).max()),
    refold=ev)

with open(f"{OUT}/h03b_prior_scale_projection.json", "w") as fh:
    json.dump(report, fh, indent=1)

print("chi2_before twin (with-cal metric): %.1f" % chi2_0)
for k, v in report["cases"].items():
    line = [k]
    for key in ("max_abs_b_prior_units", "max_abs_b_sigma_hat_units"):
        if key in v:
            line.append("max|b|/sd=%.2f" % v[key])
    for m, e in v["refold"].items():
        line.append("%s: %.0f%%" % (m, 100 * e["explained_fraction"]))
    print("  " + "  ".join(line))
