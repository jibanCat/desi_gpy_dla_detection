# REVIEW-ONLY (Phase A)
"""s3 — propagate the loa-0 calibration Poisson uncertainty into the arm stats.

The committed z-scores use survey Poisson variance only (z=(obs-mu)/sqrt(mu)).
But mu_fp = fp_w * n0[c,s] * E[k,s] with n0 the RAW loa-0 counts (89 events,
fp_w ~ 166 survey counts per event), so each row's prediction carries
calibration variance fp_w^2 * sum(n0 in row) — up to ~50x the survey variance.

Route (a): delta method  var_row += fp_w^2 * sum_cells n0 * Esum_masked[s]^2.
Route (b): parametric bootstrap, TWO framings, 500 draws each:
  (b1) frequentist: treat observed n0 as the true rate.  n0* ~ Poisson(n0)
       [the analyst's fluctuated calibration], obs* ~ Poisson(mu(n0)) [survey
       under that truth]; the replicate statistic uses the replicate's own
       plug-in: z* = (obs* - mu(n0*)) / sqrt(mu(n0*)).  STATED CHOICE: the
       observed n0 is used as the rate (it is the only estimate available).
  (b2) Jeffreys posterior-predictive: lam* ~ Gamma(n0 + 0.5, 1) per cell
       (handles zero cells), obs* ~ Poisson(mu(lam*)); the analyst's statistic
       stays at the point: z* = (obs* - mu(n0)) / sqrt(mu(n0)).
Both give the null distribution of each arm statistic when the forward model
is RIGHT and the only mismatch is calibration sampling noise + survey noise.

Cross-mock coherence: the three mocks share the SAME n0 (bit-identical
fp_counts), so a single n0* draw is applied to all three and the induced
cross-mock correlation of the per-stratum z* is measured.
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
N_DRAWS = 500
SEED = 20260805
rng = np.random.default_rng(SEED)

mocks = ("2lpt0", "london0", "saclay0")
packs, sigs, masks, csels = {}, {}, {}, {}
for m in mocks:
    p = load_pack(os.path.join(
        PACKDIR, f"modelA_pack_{m}_bw0p2_pad19p0_molly172.npz"))
    res = FS.selftest(p, resp_clamp="both")
    dxpos = np.asarray(p.dX, float) > 0
    m3 = np.broadcast_to(dxpos[None, :, :], res["mu"].shape)
    packs[m] = p
    sigs[m] = np.where(m3, np.asarray(res["mu_sig"]), 0.0)   # (C,Kf,S) masked
    masks[m] = m3
    nhat = np.asarray(p.nhat_edges, float)
    csels[m] = {"window": (nhat[:-1] >= WIN[0] - 1e-9)
                & (nhat[1:] <= WIN[1] + 1e-9),
                "full": np.ones(p.n_c, bool)}

n0 = np.asarray(packs["2lpt0"].fp_counts, float)             # shared (C,S)
C, S = n0.shape


def mu_cube(m, n0_arr):
    """mu(n0) = mu_sig + fp_w * n0 x E, masked."""
    p = packs[m]
    E = np.asarray(p.fp_E_alloc, float)
    w = float(p.fp_w_sightline_ratio)
    fp = w * n0_arr[..., :, None, :] * E[None, :, :]
    return sigs[m] + np.where(masks[m], fp, 0.0)


def arm_rows(m, cube, scope, axis):
    """Marginalise cube over selected c-bins into rows. axis: 'snr'->(0,1) sums,
    'nhat'-> per-c rows, 'total'-> scalar."""
    csel = csels[m][scope]
    sub = cube[..., csel, :, :]
    if axis == "snr":
        return sub.sum(axis=(-3, -2))                        # (...,S)
    if axis == "nhat":
        return sub.sum(axis=(-2, -1))                        # (...,n_c_sel)
    return sub.sum(axis=(-3, -2, -1))[..., None]             # (...,1)


out = {"seed": SEED, "n_draws": N_DRAWS, "window": WIN,
       "choice_note": ("bootstrap b1 treats the OBSERVED n0 as the true rate; "
                       "b2 is the Jeffreys Gamma(n0+0.5) posterior-predictive "
                       "variant covering zero-cell uncertainty"),
       "mocks": {}}

# shared calibration draws (one set, applied to all three mocks)
n0_star = rng.poisson(n0, size=(N_DRAWS, C, S)).astype(float)       # b1
lam_star = rng.gamma(n0 + 0.5, 1.0, size=(N_DRAWS, C, S))           # b2

boot_z_snr = {}   # per mock: (N_DRAWS, S) b1 z*, for cross-mock coherence

for m in mocks:
    p = packs[m]
    obs_cube = np.where(masks[m], np.asarray(p.counts, float), 0.0)
    mu0 = mu_cube(m, n0)
    w = float(p.fp_w_sightline_ratio)
    E = np.asarray(p.fp_E_alloc, float)
    Esum = np.where(np.asarray(p.dX, float) > 0, E, 0.0).sum(axis=0)  # (S,)

    res_m = {}
    for scope in ("window", "full"):
        csel = csels[m][scope]
        entry = {}
        for axis in ("snr", "nhat", "total"):
            obs_r = arm_rows(m, obs_cube, scope, axis)
            mu_r = arm_rows(m, mu0, scope, axis)
            keep = obs_r > 0
            # ---- survey-only (the committed statistic)
            z_surv = np.where(keep, (obs_r - mu_r)
                              / np.sqrt(np.maximum(mu_r, 1e-12)), 0.0)
            chi2_surv = float((z_surv[keep] ** 2).sum() / max(keep.sum(), 1))
            # ---- (a) delta method
            if axis == "snr":
                ncal = n0[csel].sum(axis=0)                       # (S,)
                var_cal = w ** 2 * ncal * Esum ** 2
            elif axis == "nhat":
                var_cal = (w ** 2 * (n0[csel] * Esum[None, :] ** 2)).sum(axis=1)
            else:
                var_cal = np.array([w ** 2
                                    * (n0[csel] * Esum[None, :] ** 2).sum()])
            z_full = np.where(keep, (obs_r - mu_r)
                              / np.sqrt(np.maximum(mu_r + var_cal, 1e-12)), 0.0)
            chi2_full = float((z_full[keep] ** 2).sum() / max(keep.sum(), 1))
            entry[axis] = dict(
                n_rows=int(keep.sum()),
                obs=obs_r[keep].tolist(), mu=mu_r[keep].tolist(),
                var_cal_over_var_surv=(var_cal[keep]
                                       / np.maximum(mu_r[keep], 1e-12)).tolist(),
                z_surv=z_surv[keep].tolist(), z_delta=z_full[keep].tolist(),
                chi2_surv=chi2_surv, chi2_delta=chi2_full,
                maxz_surv=float(np.abs(z_surv[keep]).max()),
                maxz_delta=float(np.abs(z_full[keep]).max()))

            # ---- (b) bootstrap, both framings
            for tag, cal in (("b1", n0_star), ("b2", lam_star)):
                mu_star = mu_cube(m, cal)                     # (N,C,Kf,S)
                obs_star = rng.poisson(np.broadcast_to(
                    mu0[None], mu_star.shape)) if tag == "b1" else \
                    rng.poisson(mu_star)
                mu_ref = mu_star if tag == "b1" else \
                    np.broadcast_to(mu0[None], mu_star.shape)
                mu_r_star = arm_rows(m, mu_ref, scope, axis)
                obs_r_star = arm_rows(m, obs_star.astype(float), scope, axis)
                z_star = np.where(mu_r_star > 0,
                                  (obs_r_star - mu_r_star)
                                  / np.sqrt(np.maximum(mu_r_star, 1e-12)), 0.0)
                z_star = np.where(keep[None, :], z_star, 0.0)
                chi2_star = (z_star ** 2).sum(axis=1) / max(keep.sum(), 1)
                maxz_star = np.abs(z_star).max(axis=1)
                entry[axis][f"boot_{tag}"] = dict(
                    chi2_q=[float(np.quantile(chi2_star, q))
                            for q in (0.5, 0.9, 0.95, 0.99)],
                    maxz_q=[float(np.quantile(maxz_star, q))
                            for q in (0.5, 0.9, 0.95, 0.99)],
                    p_chi2=float((chi2_star >= chi2_surv).mean()),
                    p_maxz=float((maxz_star >= entry[axis]["maxz_surv"]).mean()))
                if axis == "snr" and scope == "window" and tag == "b1":
                    boot_z_snr[m] = z_star
        res_m[scope] = entry
    out["mocks"][m] = res_m
    wsnr = res_m["window"]["snr"]
    print(f"\n===== {m} (window by_snr) =====")
    print(f"  chi2/dof  surv={wsnr['chi2_surv']:.2f}  ->  "
          f"delta={wsnr['chi2_delta']:.2f}   "
          f"boot-b1 null q50/q95={wsnr['boot_b1']['chi2_q'][0]:.2f}/"
          f"{wsnr['boot_b1']['chi2_q'][2]:.2f}  p={wsnr['boot_b1']['p_chi2']:.3f}"
          f"   boot-b2 p={wsnr['boot_b2']['p_chi2']:.3f}")
    print(f"  max|z|    surv={wsnr['maxz_surv']:.2f}  ->  "
          f"delta={wsnr['maxz_delta']:.2f}   "
          f"boot-b1 p={wsnr['boot_b1']['p_maxz']:.3f}   "
          f"boot-b2 p={wsnr['boot_b2']['p_maxz']:.3f}")
    print("  z rows surv : " + " ".join(f"{v:+.2f}" for v in wsnr["z_surv"]))
    print("  z rows delta: " + " ".join(f"{v:+.2f}" for v in wsnr["z_delta"]))
    print("  var_cal/var_surv per row: "
          + " ".join(f"{v:.1f}" for v in wsnr["var_cal_over_var_surv"]))
    wn = res_m["window"]["nhat"]
    print(f"  window by_nhat chi2/dof surv={wn['chi2_surv']:.2f} -> "
          f"delta={wn['chi2_delta']:.2f}  (boot-b1 p={wn['boot_b1']['p_chi2']:.3f})")
    tt = res_m["full"]["total"]
    print(f"  FULL total: z_surv={tt['z_surv'][0]:+.2f} -> "
          f"z_delta={tt['z_delta'][0]:+.2f}  boot-b1 p_maxz={tt['boot_b1']['p_maxz']:.3f}")

# ---- cross-mock coherence under the SHARED calibration draw ----------------
coh = {}
zm = {m: boot_z_snr[m] for m in mocks}          # (N, S) each, same n0* draws
S_keep = [s for s in range(S) if n0[:, s].sum() > 0 or True]
pairs = [("2lpt0", "london0"), ("2lpt0", "saclay0"), ("london0", "saclay0")]
corr = {}
for a, b in pairs:
    cs = []
    for s in range(S):
        x, y = zm[a][:, s], zm[b][:, s]
        if x.std() > 0 and y.std() > 0:
            cs.append(float(np.corrcoef(x, y)[0, 1]))
        else:
            cs.append(None)
    corr[f"{a}|{b}"] = cs
# probability that all three mocks simultaneously show the committed SIGN
# pattern (z>0 in [2,3), z<0 in the three top strata) under the null
sgn = np.ones(N_DRAWS, bool)
for m in mocks:
    sgn &= (zm[m][:, 2] > 0) & (zm[m][:, 5] < 0) & (zm[m][:, 6] < 0) \
        & (zm[m][:, 7] < 0)
coh["corr_z_star_by_stratum"] = corr
coh["p_all3_committed_sign_pattern"] = float(sgn.mean())
out["cross_mock_coherence"] = coh
print("\ncross-mock corr of z*[s] (b1, shared n0*):")
for k, v in corr.items():
    print(f"  {k}: " + " ".join("--" if c is None else f"{c:+.2f}" for c in v))
print(f"P(all 3 mocks show the committed sign pattern | null) = "
      f"{coh['p_all3_committed_sign_pattern']:.3f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "s3_calnoise.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("\nwrote s3_calnoise.json")
