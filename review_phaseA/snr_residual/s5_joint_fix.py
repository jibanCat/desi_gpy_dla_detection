# REVIEW-ONLY (Phase A)
"""s5 — repaired JOINT cross-mock test + truth-stratification check.

s4's joint test drew cell-level Gamma(n0+0.5) over all 29 window cells per
stratum, adding ~14.5 phantom events/row (mean) — the same defect s3's b2 had.
DEFECTIVE-AS-IMPLEMENTED; superseded here.  Two repaired joint nulls, both with
the calibration draw SHARED across the three mocks (they share fp_counts):

  J1 (frequentist, unbiased): n0* ~ Poisson(n0) cellwise, shared; per mock
     obs* ~ Poisson(mu(n0)); replicate statistic uses mu(n0*).
  J2 (row-level Jeffreys posterior-predictive): lam_row* ~ Gamma(ncal_row+0.5)
     shared across mocks (ncal_row = window row sums, all > 0, so the +0.5 is
     0.5 of an event per row, not 14.5); obs* ~ Poisson(mu(lam*)); statistic
     at the point mu(n0).

Statistic: sum over mocks of window by_snr chi2/dof, and max over mocks+rows
of |z|.  Also: the truth-stratification check — does the fold's dX-
proportional SNR allocation of the truth match truth_counts_bks?
"""
import sys, os, json
import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax; jax.config.update("jax_enable_x64", True)

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc import forward_selftest as FS

PACKDIR = ("/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
           "b10b5e23-575d-487e-811d-479f51611f63/scratchpad/r04_packs")
WIN = (19.7, 21.6)
SEED = 20260807
N_DRAWS = 2000
mocks = ("2lpt0", "london0", "saclay0")
rng = np.random.default_rng(SEED)

state = {}
for m in mocks:
    pack = load_pack(os.path.join(
        PACKDIR, f"modelA_pack_{m}_bw0p2_pad19p0_molly172.npz"))
    res = FS.selftest(pack, resp_clamp="both")
    dxpos = np.asarray(pack.dX, float) > 0
    m3 = np.broadcast_to(dxpos[None, :, :], res["mu"].shape)
    nhat = np.asarray(pack.nhat_edges, float)
    csel = (nhat[:-1] >= WIN[0] - 1e-9) & (nhat[1:] <= WIN[1] + 1e-9)
    E = np.asarray(pack.fp_E_alloc, float)
    state[m] = dict(
        pack=pack,
        sig_r=np.where(m3, np.asarray(res["mu_sig"]), 0.0)[csel].sum(axis=(0, 1)),
        mu_r=np.where(m3, np.asarray(res["mu"]), 0.0)[csel].sum(axis=(0, 1)),
        o_r=np.where(m3, np.asarray(pack.counts, float), 0.0)[csel].sum(axis=(0, 1)),
        w=float(pack.fp_w_sightline_ratio),
        Esum=np.where(dxpos, E, 0.0).sum(axis=0),
        csel=csel)

n0 = np.asarray(state["2lpt0"]["pack"].fp_counts, float)      # shared
csel = state["2lpt0"]["csel"]                                  # same grid
ncal = n0[csel].sum(axis=0)                                    # (S,) shared

obs_stat = {}
for m in mocks:
    st = state[m]
    keep = st["o_r"] > 0
    z = np.where(keep, (st["o_r"] - st["mu_r"])
                 / np.sqrt(np.maximum(st["mu_r"], 1e-12)), 0.0)
    obs_stat[m] = dict(chi2=float((z[keep] ** 2).sum() / keep.sum()),
                       maxz=float(np.abs(z[keep]).max()), keep=keep)
sum_obs = sum(v["chi2"] for v in obs_stat.values())
maxz_obs = max(v["maxz"] for v in obs_stat.values())

results = {}
for tag in ("J1", "J2"):
    if tag == "J1":
        n0_star = rng.poisson(n0, size=(N_DRAWS,) + n0.shape).astype(float)
        row_star = n0_star[:, csel, :].sum(axis=1)             # (N, S)
    else:
        row_star = rng.gamma(ncal + 0.5, 1.0, size=(N_DRAWS, ncal.size))
    sum_star = np.zeros(N_DRAWS)
    maxz_star = np.zeros(N_DRAWS)
    for m in mocks:
        st = state[m]
        keep = obs_stat[m]["keep"]
        fp_row_point = st["w"] * ncal * st["Esum"]             # mu_fp(n0) rows
        fp_row_star = st["w"] * row_star * st["Esum"][None, :]
        if tag == "J1":
            # obs* under truth = n0; analyst folds n0*
            mu_truth = st["sig_r"] + fp_row_point
            obs_star = rng.poisson(np.broadcast_to(
                np.maximum(mu_truth, 0.0), (N_DRAWS, mu_truth.size)))
            mu_an = st["sig_r"][None, :] + fp_row_star
        else:
            # obs* under lam*; analyst stays at the point
            obs_star = rng.poisson(np.maximum(
                st["sig_r"][None, :] + fp_row_star, 0.0))
            mu_an = np.broadcast_to(st["sig_r"] + fp_row_point,
                                    (N_DRAWS, ncal.size))
        z_star = np.where(keep[None, :] & (mu_an > 0),
                          (obs_star - mu_an)
                          / np.sqrt(np.maximum(mu_an, 1e-12)), 0.0)
        sum_star += (z_star ** 2).sum(axis=1) / keep.sum()
        maxz_star = np.maximum(maxz_star, np.abs(z_star).max(axis=1))
    results[tag] = dict(
        p_sum_chi2=float((sum_star >= sum_obs).mean()),
        p_maxz=float((maxz_star >= maxz_obs).mean()),
        sum_chi2_null_q=[float(np.quantile(sum_star, q))
                         for q in (0.5, 0.9, 0.95, 0.99)],
        maxz_null_q=[float(np.quantile(maxz_star, q))
                     for q in (0.5, 0.9, 0.95, 0.99)])
    print(f"{tag}: sum chi2 obs={sum_obs:.1f} null q50/q95="
          f"{results[tag]['sum_chi2_null_q'][0]:.1f}/"
          f"{results[tag]['sum_chi2_null_q'][2]:.1f}  p={results[tag]['p_sum_chi2']:.4f}"
          f"   max|z| obs={maxz_obs:.2f} null q50/q95="
          f"{results[tag]['maxz_null_q'][0]:.2f}/{results[tag]['maxz_null_q'][2]:.2f}"
          f"  p={results[tag]['p_maxz']:.4f}")

# ---- truth stratification check -------------------------------------------
truthchk = {}
for m in mocks:
    pack = state[m]["pack"]
    tb = np.asarray(pack.truth_counts_bks, float)
    tc = np.asarray(pack.truth_counts, float)
    dX = np.asarray(pack.dX, float)
    share = dX / np.maximum(dX.sum(axis=1, keepdims=True), 1e-30)
    alloc = tc[:, :, None] * share[None, :, :]
    l1 = float(np.abs(alloc - tb).sum() / max(tb.sum(), 1e-30))
    per_s_true = tb.sum(axis=(0, 1)) / max(tb.sum(), 1e-30)
    per_s_dx = (alloc.sum(axis=(0, 1)) / max(tb.sum(), 1e-30))
    truthchk[m] = dict(L1_frac=l1,
                       truth_share_per_s=per_s_true.tolist(),
                       dx_alloc_share_per_s=per_s_dx.tolist(),
                       rel_err_per_s=((per_s_dx - per_s_true)
                                      / np.maximum(per_s_true, 1e-30)).tolist())
    print(f"{m}: truth-vs-dX-alloc L1 frac = {l1:.4f}; per-s rel err = "
          + " ".join(f"{v:+.3f}" for v in truthchk[m]["rel_err_per_s"][2:]))

out = dict(seed=SEED, n_draws=N_DRAWS, window=WIN,
           sum_chi2_obs=sum_obs, maxz_obs=maxz_obs,
           joint=results, truth_stratification=truthchk,
           supersedes=("s3 b2 (cell-level Jeffreys) and s4 joint_sharedcal — "
                       "both DEFECTIVE-AS-IMPLEMENTED: +0.5 phantom mean per "
                       "zero cell x 29 cells/row dominated the null"))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "s5_joint_fix.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote s5_joint_fix.json")
