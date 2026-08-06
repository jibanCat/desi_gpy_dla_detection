# REVIEW-ONLY (Phase A)
"""s2 — independent reconstruction of the by_snr rows from pack arrays.

Own aggregation (numpy sums over the (C,Kf,S) cube), own FP formula
(mu_fp = fp_w * fp_counts[c,s] * E_alloc[k,s], the repaired normalisation),
decomposed per stratum into: signal-from-window-truth (N_true >= 19.5),
signal-from-pad-truth (19.0 <= N_true < 19.5), and FP.  Verified against
ratio_tables to machine precision.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc import forward_selftest as FS
from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu

PACKDIR = ("/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
           "b10b5e23-575d-487e-811d-479f51611f63/scratchpad/r04_packs")
WIN = (19.7, 21.6)
out = {"mocks": {}}


def fold_sig_only(pack, consts, f):
    """Signal-only fold (lam_fp = 0), for an arbitrary f on the basis grid."""
    theta = np.log(np.clip(f, 1e-300, None))
    psi_c = np.zeros((consts.n_s, consts.n_molly))
    return np.asarray(fold_mu(jnp.asarray(theta), jnp.asarray(psi_c),
                              jnp.zeros((2, consts.n_sr, consts.n_zr)),
                              jnp.zeros(consts.n_kk),
                              jnp.zeros((consts.n_c, consts.n_s)), consts))


for mock in ("2lpt0", "london0", "saclay0"):
    pth = os.path.join(PACKDIR, f"modelA_pack_{mock}_bw0p2_pad19p0_molly172.npz")
    pack = load_pack(pth)
    consts = build_consts(pack, resp_clamp="both")
    f = FS.truth_f(pack)                                    # (B, Kf)
    ntrue = np.asarray(pack.ntrue_edges, float)
    pad_bins = ntrue[1:] <= 19.5 + 1e-9                     # basis bins below floor
    f_pad = np.where(pad_bins[:, None], f, 0.0)
    f_win = np.where(pad_bins[:, None], 0.0, f)

    mu_sig_pad = fold_sig_only(pack, consts, f_pad)
    mu_sig_win = fold_sig_only(pack, consts, f_win)
    # own FP formula (repaired normalisation): fp_w*ell_eff*lam = fp_w*fp_counts
    n0 = np.asarray(pack.fp_counts, float)                  # (C, S)
    E = np.asarray(pack.fp_E_alloc, float)                  # (Kf, S)
    w = float(pack.fp_w_sightline_ratio)
    mu_fp_own = w * n0[:, None, :] * E[None, :, :]          # (C, Kf, S)
    mu_own = mu_sig_pad + mu_sig_win + mu_fp_own

    # production fold for verification
    res = FS.selftest(pack, resp_clamp="both")
    dxpos = np.asarray(pack.dX, float) > 0
    m3 = np.broadcast_to(dxpos[None, :, :], mu_own.shape)
    dev_mu = float(np.abs(np.where(m3, mu_own - np.asarray(res["mu"]), 0)).max())
    dev_fp = float(np.abs(np.where(m3, mu_fp_own - np.asarray(res["mu_fp"]), 0)).max())
    # linearity check: pad + win == full signal fold
    mu_sig_full = fold_sig_only(pack, consts, f)
    dev_lin = float(np.abs(mu_sig_pad + mu_sig_win - mu_sig_full).max())

    obs = np.asarray(pack.counts, float)
    nhat = np.asarray(pack.nhat_edges, float)
    rows = {}
    for tag, csel in (("window", (nhat[:-1] >= WIN[0] - 1e-9)
                       & (nhat[1:] <= WIN[1] + 1e-9)),
                      ("full", np.ones(pack.n_c, bool))):
        def rs(a):  # row-sum a component into strata over selected c, masked
            return np.where(m3, a, 0.0)[csel].sum(axis=(0, 1))
        o = np.where(m3, obs, 0.0)[csel].sum(axis=(0, 1))
        t = dict(obs=o, mu=rs(mu_own), sig_win=rs(mu_sig_win),
                 sig_pad=rs(mu_sig_pad), fp=rs(mu_fp_own))
        keep = o > 0
        z = (o - t["mu"]) / np.sqrt(np.maximum(t["mu"], 1e-12))
        t["z"] = np.where(keep, z, 0.0)
        t["chi2_dof"] = float((z[keep] ** 2).sum() / max(keep.sum(), 1))
        rows[tag] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in t.items()}
        if tag == "window":
            print(f"\n===== {mock} (window {WIN}) — chi2/dof "
                  f"{t['chi2_dof']:.2f} =====")
            print("  s  SNR       obs        mu    sig_win   sig_pad "
                  "      fp     resid       z")
            for s in range(pack.n_s):
                if o[s] <= 0:
                    continue
                print(f"  {s}  [{pack.snr_edges[s]:.0f},{pack.snr_edges[s+1]:.0f})"
                      f" {o[s]:9.0f} {t['mu'][s]:9.1f} {t['sig_win'][s]:9.1f}"
                      f" {t['sig_pad'][s]:9.1f} {t['fp'][s]:8.1f}"
                      f" {o[s]-t['mu'][s]:+9.1f} {z[s]:+7.2f}")
    out["mocks"][mock] = dict(
        dev_mu_vs_production=dev_mu, dev_fp_vs_production=dev_fp,
        dev_linearity=dev_lin, rows=rows,
        fp_w=w, ell_eff=float(pack.fp_ell_eff),
        n0_win_per_s=n0[(nhat[:-1] >= WIN[0] - 1e-9)
                        & (nhat[1:] <= WIN[1] + 1e-9)].sum(axis=0).tolist(),
        n0_full_per_s=n0.sum(axis=0).tolist(),
        E_masked_colsum=np.where(dxpos, E, 0.0).sum(axis=0).tolist())
    print(f"  verification: max|mu_own-mu_prod|={dev_mu:.2e}  "
          f"max|fp_own-fp_prod|={dev_fp:.2e}  linearity={dev_lin:.2e}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "s2_reconstruct.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("\nwrote s2_reconstruct.json")
