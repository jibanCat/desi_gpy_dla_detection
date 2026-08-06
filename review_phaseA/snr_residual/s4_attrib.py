# REVIEW-ONLY (Phase A)
"""s4 — attribute whatever residual survives the calibration-noise correction.

(a) use_fp=False fold: is there a signal-side per-stratum tilt already?
(b) correlate per-stratum residual with FP share / pad share / dX share.
(c) SNR support consistency: pack snr_edges vs resp_snr_edges vs
    molly_snr_edges vs the FP product's snr_min; s_to_sresp mapping.
(d) sensitivity: reallocate the FP's SNR profile to the loa-0 ALL-N measured
    profile (rank-1: keep the c-marginal, swap the s-marginal).  DIAGNOSIS
    ONLY — not a proposed model change.
Plus: row-level Jeffreys bootstrap (b2r, fixing s3's broken cell-level b2) and
the JOINT cross-mock p-value under the SHARED calibration draw.
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
FP_PRODUCT = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
              "outputs/loa0_fp_product_lyaonly1025.npz")
WIN = (19.7, 21.6)
SEED = 20260806
N_DRAWS = 500
mocks = ("2lpt0", "london0", "saclay0")
out = {"seed": SEED, "n_draws": N_DRAWS, "mocks": {}}

prod = np.load(FP_PRODUCT, allow_pickle=True)
allN_per_s = np.asarray(prod["n_fp_molly"], float).sum(axis=1)   # (8,)
out["loa0_allN_snr_profile"] = dict(
    counts=allN_per_s.tolist(), total=float(allN_per_s.sum()),
    nhi_lo=float(np.asarray(prod["nhi_edges"], float)[0]),
    nhi_hi=float(np.asarray(prod["nhi_edges"], float)[-1]),
    source=FP_PRODUCT)
print("loa-0 ALL-N FP SNR profile (n_fp_molly sum over NHI):",
      allN_per_s.astype(int).tolist(), " total", int(allN_per_s.sum()))

rng = np.random.default_rng(SEED)
chi2_obs, chi2_star_by_mock = {}, {}

for mock in mocks:
    pack = load_pack(os.path.join(
        PACKDIR, f"modelA_pack_{mock}_bw0p2_pad19p0_molly172.npz"))
    res = FS.selftest(pack, resp_clamp="both")
    res_nofp = FS.selftest(pack, resp_clamp="both", use_fp=False)
    dxpos = np.asarray(pack.dX, float) > 0
    m3 = np.broadcast_to(dxpos[None, :, :], res["mu"].shape)
    obs = np.where(m3, np.asarray(pack.counts, float), 0.0)
    mu = np.where(m3, np.asarray(res["mu"]), 0.0)
    mu_sig = np.where(m3, np.asarray(res["mu_sig"]), 0.0)
    mu_fp = np.where(m3, np.asarray(res["mu_fp"]), 0.0)
    nhat = np.asarray(pack.nhat_edges, float)
    csel = (nhat[:-1] >= WIN[0] - 1e-9) & (nhat[1:] <= WIN[1] + 1e-9)
    n0 = np.asarray(pack.fp_counts, float)
    w = float(pack.fp_w_sightline_ratio)
    E = np.asarray(pack.fp_E_alloc, float)
    Esum = np.where(dxpos, E, 0.0).sum(axis=0)

    def rows(a, sel=csel):
        return a[sel].sum(axis=(0, 1))
    o_r, mu_r, sig_r, fp_r = rows(obs), rows(mu), rows(mu_sig), rows(mu_fp)
    keep = o_r > 0
    ncal = n0[csel].sum(axis=0)
    var_cal = w ** 2 * ncal * Esum ** 2
    z_delta = np.where(keep, (o_r - mu_r)
                       / np.sqrt(np.maximum(mu_r + var_cal, 1e-12)), 0.0)

    # ---- (a) signal-only residual shape --------------------------------
    resid_nofp = o_r - sig_r                     # what FP must fill, per stratum
    z_nofp = np.where(keep, resid_nofp / np.sqrt(np.maximum(sig_r, 1e-12)), 0.0)
    # fractional signal-side tilt: (obs - mu_sig)/mu_sig vs the FP fill fp/mu_sig
    frac_gap = np.where(keep, resid_nofp / np.maximum(sig_r, 1e-12), 0.0)
    frac_fp = np.where(keep, fp_r / np.maximum(sig_r, 1e-12), 0.0)

    # ---- (b) correlates ------------------------------------------------
    dX_share = np.asarray(pack.dX, float).sum(axis=0)
    dX_share = dX_share / dX_share.sum()
    fp_share = np.where(keep, fp_r / np.maximum(mu_r, 1e-12), 0.0)
    pad_gone = None  # pad share comes from s2 (already computed there)
    resid = o_r - mu_r
    def corr(x, y, k=keep):
        x, y = np.asarray(x, float)[k], np.asarray(y, float)[k]
        if x.std() == 0 or y.std() == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    # ---- (d) rank-1 reallocation to the all-N SNR profile --------------
    c_marg = n0.sum(axis=1)                       # (C,) keep exactly
    s_prof = allN_per_s / allN_per_s.sum()        # all-N measured profile
    n0_realloc = np.outer(c_marg, s_prof)         # total preserved = 89
    mu_fp_re = w * n0_realloc[:, None, :] * E[None, :, :]
    mu_re = mu_sig + np.where(m3, mu_fp_re, 0.0)
    mu_re_r = rows(mu_re)
    z_re_surv = np.where(keep, (o_r - mu_re_r)
                         / np.sqrt(np.maximum(mu_re_r, 1e-12)), 0.0)
    chi2_re = float((z_re_surv[keep] ** 2).sum() / max(keep.sum(), 1))
    # also rank-1 with the pack's OWN s-marginal (isolates rank-1 flattening
    # from the profile swap)
    s_prof_own = n0.sum(axis=0) / n0.sum()
    n0_r1own = np.outer(c_marg, s_prof_own)
    mu_r1own_r = rows(mu_sig + np.where(
        m3, w * n0_r1own[:, None, :] * E[None, :, :], 0.0))
    z_r1own = np.where(keep, (o_r - mu_r1own_r)
                       / np.sqrt(np.maximum(mu_r1own_r, 1e-12)), 0.0)
    chi2_r1own = float((z_r1own[keep] ** 2).sum() / max(keep.sum(), 1))

    # ---- b2r: row-level Jeffreys posterior-predictive ------------------
    # lam_row* ~ Gamma(n_row + 0.5) per (stratum row), keeping the row's own
    # (c,k) shape; survey obs* ~ Poisson(mu(lam*)); statistic at the point.
    lam_row = rng.gamma(ncal + 0.5, 1.0, size=(N_DRAWS, ncal.size))
    scale = np.where(ncal > 0, lam_row / np.maximum(ncal, 1e-12), 0.0)
    mu_star_r = sig_r[None, :] + fp_r[None, :] * scale        # (N, S)
    obs_star_r = rng.poisson(np.maximum(mu_star_r, 0.0))
    z_star = np.where(keep[None, :] & (mu_r[None, :] > 0),
                      (obs_star_r - mu_r[None, :])
                      / np.sqrt(np.maximum(mu_r[None, :], 1e-12)), 0.0)
    chi2_star = (z_star ** 2).sum(axis=1) / max(keep.sum(), 1)
    z_surv = np.where(keep, (o_r - mu_r) / np.sqrt(np.maximum(mu_r, 1e-12)), 0)
    c_obs = float((z_surv[keep] ** 2).sum() / max(keep.sum(), 1))
    chi2_obs[mock] = c_obs
    chi2_star_by_mock[mock] = chi2_star
    p_b2r = float((chi2_star >= c_obs).mean())

    snr_lab = [f"[{pack.snr_edges[s]:.0f},{pack.snr_edges[s+1]:.0f})"
               for s in range(pack.n_s)]
    print(f"\n===== {mock} =====")
    print("  s      frac_gap(no-FP)  frac_FP_fill   z_nofp   z_delta  "
          "z_realloc(allN)")
    for s in range(pack.n_s):
        if not keep[s]:
            continue
        print(f"  {snr_lab[s]:8s} {frac_gap[s]:+9.4f} {frac_fp[s]:+9.4f} "
              f"{z_nofp[s]:+8.2f} {z_delta[s]:+8.2f} {z_re_surv[s]:+8.2f}")
    print(f"  window by_snr chi2/dof: surv={c_obs:.2f}  realloc-allN={chi2_re:.2f}"
          f"  rank1-own-profile={chi2_r1own:.2f}  b2r p={p_b2r:.3f}")
    print(f"  corr(resid, fp_share)={corr(resid, fp_share)}  "
          f"corr(resid, dX_share)={corr(resid, dX_share)}  "
          f"corr(frac_gap, frac_fp)={corr(frac_gap, frac_fp)}")

    out["mocks"][mock] = dict(
        snr_labels=[snr_lab[s] for s in range(pack.n_s) if keep[s]],
        frac_gap_nofp=frac_gap[keep].tolist(),
        frac_fp_fill=frac_fp[keep].tolist(),
        z_nofp=z_nofp[keep].tolist(), z_surv=z_surv[keep].tolist(),
        z_delta=z_delta[keep].tolist(), z_realloc_allN=z_re_surv[keep].tolist(),
        chi2_surv=c_obs, chi2_realloc_allN=chi2_re,
        chi2_rank1_own_profile=chi2_r1own, p_b2r_rowJeffreys=p_b2r,
        corr_resid_fp_share=corr(resid, fp_share),
        corr_resid_dX_share=corr(resid, dX_share),
        corr_gap_fpfill=corr(frac_gap, frac_fp),
        n0_win_per_s=ncal.tolist(),
        realloc_n0_win_per_s=(np.outer(c_marg, s_prof)[csel]
                              .sum(axis=0)).tolist())

    # ---- (c) support consistency, once (shared across mocks) -----------
    if mock == "2lpt0":
        from CDDF_analysis.hbi_mcmc.forward import build_consts
        consts = build_consts(pack, resp_clamp="both")
        out["support_check"] = dict(
            snr_edges=np.asarray(pack.snr_edges).tolist(),
            resp_snr_edges=np.asarray(pack.resp_snr_edges).tolist(),
            molly_snr_edges=np.asarray(pack.molly_snr_edges).tolist(),
            fp_product_snr_min=float(prod["snr_min"]),
            s_to_sresp=np.asarray(consts.s_to_sresp).tolist(),
            dX_zero_strata=[int(s) for s in range(pack.n_s)
                            if not dxpos[:, s].any()],
            counts_in_dX_zero_strata=float(
                np.asarray(pack.counts, float)[:, :, ~dxpos.any(axis=0)].sum()),
            fp_counts_in_dX_zero_strata=float(
                n0[:, ~dxpos.any(axis=0)].sum()),
            E_masked_colsum=Esum.tolist(),
            note=("strata [0,1) and [1,2) have dX==0, obs==0, fp==0 "
                  "consistently (product snr_min=2.0): no one-sided support "
                  "hole on the SNR axis at the bottom. Stratum [3,4) straddles "
                  "the response-cell edge at 3.5; molly grid matches the "
                  "stratum grid exactly."))

# ---- joint cross-mock p under the SHARED calibration draw -----------------
# NOTE: b2r above used INDEPENDENT lam_row* per mock (rng state advances), so
# rerun with one shared draw for the joint test.
rng2 = np.random.default_rng(SEED + 1)
packs2 = {m: load_pack(os.path.join(
    PACKDIR, f"modelA_pack_{m}_bw0p2_pad19p0_molly172.npz")) for m in mocks}
shared = {}
n0s = np.asarray(packs2["2lpt0"].fp_counts, float)
lam_full = rng2.gamma(n0s + 0.5, 1.0, size=(N_DRAWS,) + n0s.shape)  # cellwise
sum_obs = sum(chi2_obs.values())
sum_star = np.zeros(N_DRAWS)
for m in mocks:
    pack = packs2[m]
    res = FS.selftest(pack, resp_clamp="both")
    dxpos = np.asarray(pack.dX, float) > 0
    m3 = np.broadcast_to(dxpos[None, :, :], res["mu"].shape)
    mu_sig = np.where(m3, np.asarray(res["mu_sig"]), 0.0)
    mu = np.where(m3, np.asarray(res["mu"]), 0.0)
    obs = np.where(m3, np.asarray(pack.counts, float), 0.0)
    nhat = np.asarray(pack.nhat_edges, float)
    csel = (nhat[:-1] >= WIN[0] - 1e-9) & (nhat[1:] <= WIN[1] + 1e-9)
    w = float(pack.fp_w_sightline_ratio)
    E = np.asarray(pack.fp_E_alloc, float)
    sig_r = mu_sig[csel].sum(axis=(0, 1))
    mu_r = mu[csel].sum(axis=(0, 1))
    o_r = obs[csel].sum(axis=(0, 1))
    keep = o_r > 0
    # fold each shared lam* draw (row sums only; linear)
    Esum = np.where(dxpos, E, 0.0).sum(axis=0)
    lam_row = (lam_full[:, csel, :]).sum(axis=1)              # (N, S)
    mu_star_r = sig_r[None, :] + w * lam_row * Esum[None, :]
    obs_star_r = rng2.poisson(np.maximum(mu_star_r, 0.0))
    z_star = np.where(keep[None, :], (obs_star_r - mu_r[None, :])
                      / np.sqrt(np.maximum(mu_r[None, :], 1e-12)), 0.0)
    sum_star += (z_star ** 2).sum(axis=1) / max(keep.sum(), 1)
p_joint = float((sum_star >= sum_obs).mean())
out["joint_sharedcal"] = dict(
    sum_chi2_obs=sum_obs, p_joint=p_joint,
    sum_chi2_null_q=[float(np.quantile(sum_star, q))
                     for q in (0.5, 0.9, 0.95, 0.99)],
    note=("sum over the 3 mocks of the window by_snr chi2/dof; null = ONE "
          "shared cell-level Jeffreys calibration draw + independent survey "
          "Poisson per mock — the three mocks are ~one draw of the FP noise, "
          "so this is THE joint test"))
print(f"\nJOINT (shared cal draw): sum chi2 obs={sum_obs:.1f}  "
      f"null q50/q95={np.quantile(sum_star,0.5):.1f}/"
      f"{np.quantile(sum_star,0.95):.1f}  p={p_joint:.3f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "s4_attrib.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote s4_attrib.json")
