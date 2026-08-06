# REVIEW-ONLY (Phase A) — does not alter production behavior.
"""r05 — Double-count audit of the repaired convention, end to end.

If any of the four repaired sites applied ell_eff twice (or a fifth site
still omitted it), one of these round trips would break:

  (1) GENERATOR -> FOLD: synthetic_pack(fp_frac=0.15) scales lam_fp_true so
      the DATA-side FP share is 15%.  Folding the pack's own truth back
      through forward.fold_mu must therefore put mu_fp/(mu_sig+mu_fp) at
      ~0.15 — if the generator carried ell but the fold did not (or vice
      versa) the share would be 0.15*ell or 0.15/ell (~2.04 or ~0.011 at
      ell=13.6-like values).
  (2) GENERATOR -> CALIBRATION: the drawn fp_counts must be consistent with
      Poisson(ell * lam_fp_true) — mean-matched, not ell^2- or 1-scaled.
  (3) FOLD vs ORACLE: fold_mu and fold_mu_reference agree at 1e-10 (the
      committed tests assert this; re-checked here at a random point).
  (4) SELFTEST SPLIT: selftest's mu == mu_sig + mu_fp with mu_fp equal to
      fold_mu_fp at lam = fp_counts/ell — the re-typed copy C2 unified.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from CDDF_analysis.hbi_mcmc.pack import synthetic_pack, small_test_grid  # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS  # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import (  # noqa: E402
    build_consts, fold_mu, fold_mu_fp, fold_mu_reference)

out = {}

grid = small_test_grid()
FP_FRAC = 0.15
ELL = 13.589891949531905          # production-like exposure
pack = synthetic_pack(0, **grid, fp_frac=FP_FRAC, fp_ell_eff=ELL, fp_w=165.93)
tr = pack.truth

consts = build_consts(pack, resp_clamp="both")
theta = jnp.asarray(tr["theta_true"])
psi_c = jnp.asarray(tr["psi_c_true"])
psi_k = jnp.zeros((2, consts.n_sr, consts.n_zr))
log_t = jnp.asarray(tr["t_true"])
lam_true = jnp.asarray(tr["lam_fp_true"])

mu = np.asarray(fold_mu(theta, psi_c, psi_k, log_t, lam_true, consts))
mu_fp = np.asarray(fold_mu_fp(log_t, lam_true, consts))
share = float(mu_fp.sum() / mu.sum())
out["generator_fold_share"] = dict(
    requested_fp_frac=FP_FRAC, folded_fp_share=share,
    ratio_to_requested=share / FP_FRAC,
    would_be_if_fold_dropped_ell=share / FP_FRAC / 1.0 * (1.0 / ELL),
)

# (2) calibration side: fp_counts mean vs ell * lam_true
lam_np = np.asarray(tr["lam_fp_true"], float)
exp_counts = ELL * lam_np.sum()
obs_counts = float(np.asarray(pack.fp_counts, float).sum())
out["generator_calibration"] = dict(
    expected_total=exp_counts, drawn_total=obs_counts,
    z_score=float((obs_counts - exp_counts) / np.sqrt(max(exp_counts, 1e-12))),
)

# (3) fold vs independent oracle at a perturbed random point
rng = np.random.default_rng(7)
th2 = np.asarray(tr["theta_true"]) + 0.1 * rng.normal(size=np.shape(tr["theta_true"]))
lam2 = np.abs(lam_np + 0.3 * rng.normal(size=lam_np.shape) * (lam_np + 0.1))
t2 = np.asarray(tr["t_true"]) + 0.05 * rng.normal(size=np.shape(tr["t_true"]))
mu_j = np.asarray(fold_mu(jnp.asarray(th2), psi_c, psi_k, jnp.asarray(t2),
                          jnp.asarray(lam2), consts))
mu_r = fold_mu_reference(th2, np.asarray(tr["psi_c_true"]),
                         np.zeros((2, consts.n_sr, consts.n_zr)), t2, lam2, pack)
out["fold_vs_oracle_max_rel"] = float(
    np.max(np.abs(mu_j - mu_r) / np.maximum(np.abs(mu_r), 1e-300)))

# (4) selftest split
res = FS.selftest(pack, resp_clamp="both")
lam_hat = np.asarray(pack.fp_counts, float) / float(pack.fp_ell_eff)
mu_fp_direct = np.asarray(fold_mu_fp(jnp.zeros(consts.n_kk),
                                     jnp.asarray(lam_hat), consts))
out["selftest_split"] = dict(
    max_abs_mu_minus_sig_minus_fp=float(
        np.max(np.abs(res["mu"] - res["mu_sig"] - res["mu_fp"]))),
    max_abs_selftest_fp_vs_direct=float(
        np.max(np.abs(res["mu_fp"] - mu_fp_direct))),
)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "r05_out.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
