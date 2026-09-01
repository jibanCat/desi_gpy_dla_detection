#!/usr/bin/env python3
"""geometry.py — posterior-geometry audit of a validation run (2026-09-02 HBI identifiability
campaign): Task 1 (log Lambda, t_K, A_K = log Lambda + t_K, 7x7 corr/cov, ridge eigenvectors,
KL, per-chain means, per-site split-Rhat/ESS), the FP-share / t=0 forward counterfactual
(closed-form FP branch + model_cc's own fold for TP), Task 2 (effective upper-sub-DLA
completeness, leverage ranking of psi_c cells), likelihood-term accounting (per-site log
densities of the frozen model at chosen points), and low-dimensional mode diagnostics.

All numbers are REAL-DATA nuisance posteriors: outputs go to the private run directory only.

    python tools/hbi_validation/geometry.py --pack PACK --run-dir ROOT/R0 --pooled POOLED_ln_R0.json \
        --out ROOT/R0/analysis/geometry_R0.json [--extra-arm ROOT/R0/REAL_ln_deep_s20260826_allsites.npz:mirror]
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.pack import load_pack                                   # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors, model_cc  # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_real_ppc import cc_fold_mu                          # noqa: E402
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior                      # noqa: E402
from CDDF_analysis.hbi_mcmc.provenance_util import sha256                          # noqa: E402
from tools.hbi_validation.site_mapping import SITE_ORDER, build_mapping, flatten_draws  # noqa: E402

SUBDLA = (19.7, 20.3)
Z_BLOCKS = ((2.0, 2.5), (2.5, 3.0), (3.0, 3.5))
# Omega_HI prefactor at the paper's h = 0.70 (science-side diagnostic only; the manuscript's Omega is
# the paper-side reduction): H0 m_H / (c rho_crit) with f dN in per-unit-N units and N in cm^-2.
H0_KM_S_MPC = 70.0


def q(x, p=(2.5, 16, 50, 84, 97.5)):
    return [float(v) for v in np.percentile(np.asarray(x), p)]


def corr(x, y):
    """Pearson r, or None when either input is constant (a FIXED site in R1/R3/R4)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def fin(v):
    """float or None for non-finite (JSON-clean)."""
    try:
        v = float(v)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def omega_prefactor(H0=H0_KM_S_MPC):
    """H0 m_H / (c rho_crit)  [cm^2] such that Omega = pref * sum 10^N f dN  (N in cm^-2, f per cm^2 per dX)."""
    m_H = 1.6735575e-24        # g
    c = 2.99792458e10          # cm/s
    G = 6.674e-8
    H0 = H0 * 1e5 / 3.0856776e24   # s^-1
    rho_crit = 3 * H0**2 / (8 * np.pi * G)
    return H0 * m_H / (c * rho_crit)


def load_arm(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    return d


def split_rhat_ess(x):
    """x: (chains, draws) -> split-Rhat, bulk ESS, tail ESS via numpyro.diagnostics."""
    from numpyro.diagnostics import split_gelman_rubin, effective_sample_size
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None]
    if x.shape[0] * x.shape[1] < 4 or np.allclose(x, x.flat[0]):
        return float("nan"), float("nan"), float("nan")
    rh = float(split_gelman_rubin(x))
    ess = float(effective_sample_size(x))
    # tail ESS: ESS of the indicator of the 5/95 % tails (Vehtari+21 spirit)
    lo, hi = np.quantile(x, [0.05, 0.95])
    ess_t = min(float(effective_sample_size((x <= lo).astype(float))) if (x <= lo).any() else np.nan,
                float(effective_sample_size((x >= hi).astype(float))) if (x >= hi).any() else np.nan)
    return rh, ess, ess_t


def fp_mu(consts, t, lam_fp, dXm):
    return (consts.fp_w * consts.fp_ell_eff * (1.0 - np.asarray(consts.fp_eta_c))[:, None, None]
            * np.exp(np.asarray(t)[np.asarray(consts.kz_to_K)])[None, :, None]
            * np.asarray(lam_fp)[:, None, :] * np.asarray(consts.fp_E)[None, :, :]) * dXm[None]


def main(argv=None):
    import jax, jax.numpy as jnp
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--run-dir", default=None, help="(informational) run directory")
    ap.add_argument("--pooled", default=None, help="pooled JSON of the run (its selection block names the arms)")
    ap.add_argument("--arms", nargs="*", default=None, help="alternative to --pooled: explicit all-sites npz paths (testing / unpooled modes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-arm", action="append", default=[], help="path:label of an all-sites npz to summarise separately (e.g. the mirror chain)")
    ap.add_argument("--n-lev", type=int, default=8)
    ap.add_argument("--max-draws-fold", type=int, default=2000)
    a = ap.parse_args(argv)

    pk = load_pack(a.pack); consts, Mg = build_cc_tensors(pk)
    pack = np.load(a.pack)
    if a.pooled:
        P = json.load(open(a.pooled))
        arms = [(r["seed"], r["deep"], r["file"][:-5] + "_allsites.npz") for r in P["selection"]["included"]]
    elif a.arms:
        import re
        arms = [(int(re.search(r"_s(\d{8})_allsites", p).group(1)), "_deep_" in os.path.basename(p), p) for p in a.arms]
    else:
        raise SystemExit("give --pooled or --arms")
    for p in arms:
        if not os.path.isfile(p[2]):
            raise SystemExit(f"missing all-sites draws {p[2]}")
    res = dict(pack=a.pack, pack_sha256=sha256(a.pack), pooled=a.pooled, pooled_sha256=(sha256(a.pooled) if a.pooled else None),
               arms=[dict(seed=s, deep=d, allsites=p, sha256=sha256(p)) for s, d, p in arms])
    mapping = build_mapping(pack); names = [r["scalar_id"] for r in mapping]
    ts = np.asarray(consts.t_sigma); KK = len(ts)
    # ---- stack arms: X (n, 574) in mapping order; arm id; chain id; PE; f; theta; psi_c; lam_fp; t
    Xs, arm_id, chain_id, PE, F, TH, PS, LF, T = [], [], [], [], [], [], [], [], []
    per_arm = {}
    for ai, (s, d, p) in enumerate(arms):
        z = load_arm(p)
        X, ch = flatten_draws(z)
        Xs.append(X); arm_id.append(np.full(X.shape[0], ai)); chain_id.append(ch)
        PE.append(np.asarray(z["potential_energy"]).reshape(-1))
        F.append(np.asarray(z["f"]).reshape(-1, *z["f"].shape[2:])); TH.append(np.asarray(z["theta_pop"]).reshape(-1, *z["theta_pop"].shape[2:]))
        PS.append(np.asarray(z["psi_c"]).reshape(-1, *z["psi_c"].shape[2:])); LF.append(np.asarray(z["lam_fp"]).reshape(-1, *z["lam_fp"].shape[2:]))
        T.append(np.asarray(z["t"]).reshape(-1, KK))
        # per-site split-Rhat / ESS within this arm (chains, draws)
        nch, nd = z["t"].shape[:2]
        site_diag = {}
        for name, col in zip(names, X.T):
            rh, e, et = split_rhat_ess(col.reshape(nch, nd))
            site_diag[name] = (rh, e, et)
        rhs = np.array([v[0] for v in site_diag.values()]); ess = np.array([v[1] for v in site_diag.values()])
        per_arm[f"s{s}{'d' if d else ''}"] = dict(
            seed=s, deep=bool(d), n=int(X.shape[0]), pe_chain_mean=[float(np.mean(PE[-1].reshape(nch, nd)[c])) for c in range(nch)],
            t_mean=[float(v) for v in T[-1].mean(axis=0)], t_chain_mean=[[float(v) for v in T[-1].reshape(nch, nd, KK)[c].mean(axis=0)] for c in range(nch)],
            logL_mean=float(np.log(LF[-1].sum(axis=(1, 2))).mean()),
            site_rhat_max=float(np.nanmax(rhs)), site_rhat_n_gt_1p05=int(np.nansum(rhs > 1.05)), site_rhat_n_gt_1p10=int(np.nansum(rhs > 1.10)),
            site_ess_min=float(np.nanmin(ess)), site_ess_n_lt_100=int(np.nansum(ess < 100)),
            worst_sites_rhat=[(names[i], float(rhs[i])) for i in np.argsort(-np.nan_to_num(rhs, nan=0))[:8]],
            worst_sites_ess=[(names[i], float(ess[i])) for i in np.argsort(np.nan_to_num(ess, nan=1e9))[:8]],
            t_rhat=[float(site_diag[f"t[{K}]"][0]) for K in range(KK)], t_ess=[float(site_diag[f"t[{K}]"][1]) for K in range(KK)],
            lam_rhat=float(site_diag["fp_lam_total"][0]), lam_ess=float(site_diag["fp_lam_total"][1]))
    X = np.concatenate(Xs); arm_id = np.concatenate(arm_id); chain_id = np.concatenate(chain_id); PE = np.concatenate(PE)
    F = np.concatenate(F); TH = np.concatenate(TH); PS = np.concatenate(PS); LF = np.concatenate(LF); T = np.concatenate(T)
    n = X.shape[0]
    res["n_draws"] = int(n); res["per_arm"] = per_arm
    # ---- Task 1
    Lam = LF.sum(axis=(1, 2)); logL = np.log(Lam); A = logL[:, None] + T
    naive = float(np.asarray(pk.fp_counts, float).sum() / consts.fp_ell_eff)
    res["t_sigma"] = ts.tolist()
    res["t"] = {f"K{K}": dict(median=float(np.median(T[:, K])), mean=float(T[:, K].mean()), sd=float(T[:, K].std(ddof=1)), q=q(T[:, K]),
                             median_over_sigma=float(np.median(T[:, K]) / ts[K]), post_over_prior_width=float(T[:, K].std(ddof=1) / ts[K]),
                             exp_t_median=float(np.exp(np.median(T[:, K]))), exp_t_p16_84=q(np.exp(T[:, K]), (16, 84)),
                             kl_gauss_nats=fin(0.5 * ((T[:, K].std(ddof=1) / ts[K])**2 + (T[:, K].mean() / ts[K])**2 - 1 - 2 * np.log(T[:, K].std(ddof=1) / ts[K])) if T[:, K].std() > 0 else None),
                             arm_means=[float(T[arm_id == i, K].mean()) for i in range(len(arms))],
                             arm_sd_of_means=float(np.std([T[arm_id == i, K].mean() for i in range(len(arms))], ddof=1)))
                for K in range(KK)}
    res["log_Lambda"] = dict(median=float(np.median(logL)), sd=float(logL.std(ddof=1)), Lambda_median=float(np.median(Lam)), Lambda_mean=float(Lam.mean()), Lambda_sd=float(Lam.std(ddof=1)),
                             Lambda_over_naive=q(Lam / naive, (16, 50, 84)), naive=naive,
                             prior=dict(conc=float(np.asarray(pk.fp_counts).sum() + 0.5), rate=float(consts.fp_ell_eff),
                                        mean=float((np.asarray(pk.fp_counts).sum() + 0.5) / consts.fp_ell_eff), sd=float(np.sqrt(np.asarray(pk.fp_counts).sum() + 0.5) / consts.fp_ell_eff)))
    res["A_K"] = {f"K{K}": dict(median=float(np.median(A[:, K])), sd=float(A[:, K].std(ddof=1)), sd_A_over_sd_t=fin(A[:, K].std(ddof=1) / T[:, K].std(ddof=1)) if T[:, K].std() > 0 else None,
                               sd_A_over_sd_logL=float(A[:, K].std(ddof=1) / logL.std(ddof=1)), Lambda_exp_t_median=float(np.median(np.exp(A[:, K])))) for K in range(KK)}
    Y7 = np.column_stack([logL, T, A]); n7 = ["logLambda"] + [f"t{K}" for K in range(KK)] + [f"A{K}" for K in range(KK)]
    with np.errstate(invalid="ignore", divide="ignore"):
        c7 = np.corrcoef(Y7.T)
    res["corr7"] = dict(names=n7, corr=[[fin(v) for v in row] for row in c7.round(4)], cov=np.cov(Y7.T).tolist())
    Y4 = np.column_stack([logL, T]); w, v = np.linalg.eigh(np.cov(Y4.T))
    res["eig_logL_t"] = dict(names=n7[:1 + KK], eigvals=w[::-1].tolist(), eigvecs_rows=v[:, ::-1].T.round(4).tolist())
    res["ridge_pairs"] = {}
    for K in range(KK):
        c2 = np.cov(np.column_stack([logL, T[:, K]]).T); w2, v2 = np.linalg.eigh(c2)
        res["ridge_pairs"][f"K{K}"] = dict(corr=corr(logL, T[:, K]), eigvals=w2[::-1].tolist(), major_axis=v2[:, -1].round(4).tolist(),
                                          conserved_A_direction_share=float(abs(np.dot(v2[:, -1], np.array([-1, 1]) / np.sqrt(2)))**2))
    # ---- FP shares, t=0 forward counterfactual, mu_TP via model_cc's fold (subsample for the fold)
    nhat = np.asarray(pk.nhat_edges, float); cen = 0.5 * (nhat[:-1] + nhat[1:]); kz = np.asarray(consts.kz_to_K)
    dXm = np.asarray(consts.dX) > 0; counts = np.asarray(pk.counts, float) * dXm[None]
    idx = np.arange(n) if n <= a.max_draws_fold else np.sort(np.random.default_rng(0).choice(n, a.max_draws_fold, replace=False))
    fold = jax.vmap(cc_fold_mu, in_axes=(None, None, 0, 0, 0, 0))
    mu_cat = np.concatenate([np.asarray(fold(consts, Mg, jnp.asarray(TH[idx[i:i + 100]]), jnp.asarray(PS[idx[i:i + 100]]), jnp.asarray(T[idx[i:i + 100]]), jnp.asarray(LF[idx[i:i + 100]])))
                             for i in range(0, len(idx), 100)]) * dXm[None, None]
    muFP = np.stack([fp_mu(consts, T[i], LF[i], dXm) for i in idx]); muFP0 = np.stack([fp_mu(consts, np.zeros(KK), LF[i], dXm) for i in idx])
    muTP = mu_cat - muFP
    sets = {"all_19p5_22p4": np.ones(len(cen), bool), "subdla_19p7_20p3": (cen >= SUBDLA[0]) & (cen < SUBDLA[1]), "dla_ge20p3": cen >= 20.3, "sub_19p5_19p7": cen < 19.7}
    blocks = {f"z{K}_{lo}_{hi}": kz == K for K, (lo, hi) in enumerate(Z_BLOCKS)}; blocks["allz"] = np.ones(len(kz), bool)
    shares = {}
    for sn, sm in sets.items():
        shares[sn] = {}
        for bn, bm in blocks.items():
            sel = np.ix_(np.arange(len(idx)), np.where(sm)[0], np.where(bm)[0], np.arange(counts.shape[2]))
            fp = muFP[sel].sum(axis=(1, 2, 3)); fp0 = muFP0[sel].sum(axis=(1, 2, 3)); tp = muTP[sel].sum(axis=(1, 2, 3))
            ob = float(counts[np.ix_(np.where(sm)[0], np.where(bm)[0], np.arange(counts.shape[2]))].sum())
            shares[sn][bn] = dict(obs=ob, mu_cat=q(fp + tp, (16, 50, 84)), mu_TP=q(tp, (16, 50, 84)), mu_FP=q(fp, (16, 50, 84)),
                                  fp_share_of_mu_cat=q(fp / (fp + tp), (16, 50, 84)), fp_share_of_obs=q(fp / ob, (16, 50, 84)),
                                  fp_share_of_obs_at_t0_FORWARD_COUNTERFACTUAL=q(fp0 / ob, (16, 50, 84)),
                                  displacement_pct_obs_t_to_0_FORWARD_COUNTERFACTUAL=q(100 * (fp - fp0) / ob), mult_change=q(fp / fp0, (16, 50, 84)))
    res["fp_shares"] = shares
    # ---- the IDENTIFIED coordinate: log of the FP intensity actually leveraged by the catalogue (per N set × z block),
    # compared with A_K and log Lambda; plus the FP-shape pulls against the loa-0 prior and the pi mass in the leveraged cells
    fpc_np = np.asarray(pk.fp_counts, float); n_fp = fpc_np.sum(); Kc = fpc_np.size; a0 = 1.0 / Kc
    m_cs = np.log((fpc_np.reshape(-1) + a0) / (n_fp + Kc * a0)); s_cs = np.where(fpc_np.reshape(-1) > 0, 1.0 / np.sqrt(fpc_np.reshape(-1) + 1.0), 2.0)
    iv = [names.index(f"fp_shape_v[c={c},s={s_}]") for c in range(consts.n_c) for s_ in range(consts.n_s)]
    V = X[:, iv]; pull = (V - m_cs[None]) / s_cs[None]
    pi = LF / LF.sum(axis=(1, 2))[:, None, None]
    ident = {}
    for sn, sm in sets.items():
        ident[sn] = {}
        for bn, bm in blocks.items():
            if bn == "allz":
                continue
            K = int(np.where(bm)[0][0]) if bm.sum() else 0; K = int(kz[np.where(bm)[0][0]])
            mfp = np.stack([fp_mu(consts, T[i], LF[i], dXm)[np.ix_(np.where(sm)[0], np.where(bm)[0], np.arange(counts.shape[2]))].sum() for i in range(n)])
            logmu = np.log(mfp); pimass = pi[:, sm, :].sum(axis=(1, 2))
            ident[sn][bn] = dict(sd_log_muFP=float(logmu.std(ddof=1)), sd_A_K=float(A[:, K].std(ddof=1)), sd_logLambda=float(logL.std(ddof=1)), sd_log_pimass=float(np.log(pimass).std(ddof=1)),
                                 corr_logmu_AK=corr(logmu, A[:, K]), corr_logmu_logL=corr(logmu, logL), corr_logmu_t=corr(logmu, T[:, K]),
                                 corr_t_logpimass=corr(T[:, K], np.log(pimass)), corr_logL_logpimass=corr(logL, np.log(pimass)),
                                 pimass_median=float(np.median(pimass)), pimass_prior_centre=float(np.exp(m_cs).reshape(consts.n_c, consts.n_s)[sm].sum() / np.exp(m_cs).sum()))
    res["identified_coordinate"] = dict(note="log mu_FP(N set, z block) = A_K + log(sum_{c in set,s} pi_cs O_cks) + const: the coordinate the catalogue sees; compare its sd with sd(A_K), sd(log Lambda)", table=ident)
    pop = fpc_np.reshape(-1) > 0
    res["fp_shape_pulls"] = dict(prior="fp_shape_v ~ Normal(m_cs, s_cs); pull z = (v - m)/s", n_cells=int(Kc), n_populated=int(pop.sum()),
                                 sum_z2_median=float(np.median((pull**2).sum(axis=1))), sum_z2_expected_prior=float(Kc),
                                 sum_z2_populated_median=float(np.median((pull[:, pop]**2).sum(axis=1))), sum_z2_empty_median=float(np.median((pull[:, ~pop]**2).sum(axis=1))),
                                 mean_pull_populated=float(pull[:, pop].mean()), mean_pull_empty=float(pull[:, ~pop].mean()),
                                 top_cells_by_abs_mean_pull=[dict(cell=names[iv[i]], mean_pull=float(pull[:, i].mean()), sd_pull=float(pull[:, i].std(ddof=1)), n_loa0=int(fpc_np.reshape(-1)[i]))
                                                             for i in np.argsort(-np.abs(pull.mean(axis=0)))[:12]],
                                 log_prior_term_median=float(np.median((-0.5 * pull**2 - np.log(s_cs)[None] - 0.5 * np.log(2 * np.pi)).sum(axis=1))),
                                 log_prior_term_expected=float((-0.5 - np.log(s_cs) - 0.5 * np.log(2 * np.pi)).sum()))
    # ---- science outputs from the run's own draws (reduce_f_posterior + Omega[20.3,21.6])
    red = reduce_f_posterior(F, pk)
    ntrue = np.asarray(pk.ntrue_edges, float); Nc = 0.5 * (ntrue[:-1] + ntrue[1:]); dN = np.diff(ntrue); dXk = np.asarray(pk.dX, float).sum(axis=1)
    mO = (Nc >= 20.3 - 1e-9) & (Nc < 21.6 - 1e-9)
    omega_k = np.einsum("dbk,b->dk", F[:, mO, :], 10**Nc[mO] * dN[mO]) * omega_prefactor()
    omega_allz = (omega_k * dXk[None]).sum(axis=1) / dXk.sum()
    omega_blocks = [float(np.median((omega_k[:, kz == K] * dXk[kz == K][None]).sum(axis=1) / dXk[kz == K].sum())) for K in range(KK)]
    sci = dict(dndx_ge20p0_allz=q(red["dndx_dla_20p0_allz"]), dndx_ge20p3_allz=q(red["dndx_dla_20p3_allz"]),
               dndx_ge20p3_coarse_median=[float(np.median(np.asarray(red["dndx_dla_20p3_coarse"])[:, K])) for K in range(KK)],
               dndx_ge20p0_coarse_median=[float(np.median(np.asarray(red["dndx_dla_20p0_coarse"])[:, K])) for K in range(KK)],
               omega_20p3_21p6_allz_h0p70=q(omega_allz), omega_20p3_21p6_coarse_median_h0p70=omega_blocks,
               omega_note="science-side diagnostic: sum_b 10^N_b f dN over [20.3,21.6) x H0 m_H/(c rho_crit) at h=0.70; the manuscript's Omega is the paper-side reduction")
    REDGES = np.arange(19.7, 21.7 + 1e-9, 0.2); bins = []
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        dr = ((F[:, m, :] * dN[None, m, None]).sum(axis=1) * dXk[None, :]).sum(axis=1) / dXk.sum()
        bins.append(dict(bin=[round(float(e0), 1), round(float(e1), 1)], f_post=q(dr)))
    sci["reporting_bins"] = bins
    res["science"] = sci
    # latent upper-sub-DLA amplitude: dX-weighted f over latent bins in [19.7,20.3)
    mS = (Nc >= SUBDLA[0] - 1e-9) & (Nc < SUBDLA[1] - 1e-9)
    f_sub = (np.einsum("dbk,b->dk", F[:, mS, :], dN[mS]) * dXk[None]).sum(axis=1) / dXk.sum()
    f_sub_z0 = (np.einsum("dbk,b->dk", F[:, mS, :], dN[mS])[:, kz == 0] * dXk[kz == 0][None]).sum(axis=1) / dXk[kz == 0].sum()
    # ---- Task 2: effective upper-sub-DLA completeness (pre-declared definition, execution book Phase C)
    eta = np.asarray(consts.eta_hat); sig_hat = np.asarray(consts.sigma_hat); b2c = np.asarray(consts.b_to_cell); g = np.asarray(consts.g_bk); dX = np.asarray(consts.dX)
    fbar = F.mean(axis=0)                                                  # (B,Kf)
    w_bs = np.einsum("bk,bk,ks->bs", g[mS] * dN[mS][:, None], fbar[mS], dX)   # (Bsub, S) TP-fold weights, response marginalised
    cells = sorted(set(int(c) for c in b2c[mS]))
    def Cbar(psi):   # psi: (n,S,M)
        Cc = 1 / (1 + np.exp(-(eta[None] + psi)))                          # (n,S,M)
        Csb = Cc[:, :, b2c[mS]]                                            # (n,S,Bsub)
        return np.einsum("nsb,bs->n", Csb, w_bs) / w_bs.sum()
    Cb = Cbar(PS); C0 = float(Cbar(np.zeros((1,) + PS.shape[1:]))[0])
    dbar = np.einsum("nsb,bs->n", PS[:, :, b2c[mS]], w_bs) / w_bs.sum()
    res["completeness"] = dict(definition="C_bar = sum_{b in [19.7,20.3) latent, s} w[b,s] sigmoid(eta_hat[s,m(b)] + psi_c[s,m(b)]) / sum w ; w[b,s] = sum_k g[b,k] dN_b dX[k,s] fbar[b,k] (TP-fold weights at the posterior-mean f, response marginalised); d_bar = sum w psi_c / sum w ; z enters ONLY through dX[k,s]",
                               latent_bins=[[float(ntrue[b]), float(ntrue[b + 1])] for b in np.where(mS)[0]], molly_cells_m=cells,
                               molly_cell_edges=[[float(pack["molly_nhi_edges"][m]), float(pack["molly_nhi_edges"][m + 1])] for m in cells],
                               weights_by_stratum=(w_bs.sum(axis=0) / w_bs.sum()).round(4).tolist(),
                               C_bar_central=C0, C_bar=q(Cb), d_bar=q(dbar), corr_t_Cbar=[corr(T[:, K], Cb) for K in range(KK)],
                               corr_t_dbar=[corr(T[:, K], dbar) for K in range(KK)],
                               corr_logL_Cbar=corr(logL, Cb), corr_Cbar_fsub=corr(Cb, f_sub))
    # leverage of psi_c cells on the low-z sub-DLA catalogue prediction: |d mu_cat(19.7-20.3, z0) / d psi_c[s,m]| x sd(psi_c[s,m]) at the posterior mean
    th_m, ps_m, t_m, lf_m = jnp.asarray(TH.mean(axis=0)), jnp.asarray(PS.mean(axis=0)), jnp.asarray(T.mean(axis=0)), jnp.asarray(LF.mean(axis=0))
    smask = jnp.asarray(sets["subdla_19p7_20p3"]); 
    def mu_sub(psi, K):
        mu = cc_fold_mu(consts, Mg, th_m, psi, t_m, lf_m) * jnp.asarray(dXm)[None]
        return (mu * smask[:, None, None] * jnp.asarray(kz == K)[None, :, None]).sum()
    lev = {}
    for K in range(KK):
        gK = np.asarray(jax.grad(mu_sub)(ps_m, K))
        obsK = float(counts[np.ix_(np.where(sets["subdla_19p7_20p3"])[0], np.where(kz == K)[0], np.arange(counts.shape[2]))].sum())
        L = np.abs(gK) * PS.std(axis=0, ddof=1)
        order = np.argsort(-L.reshape(-1))[:a.n_lev]
        lev[f"z{K}"] = [dict(cell=f"psi_c[s={i // PS.shape[2]},m={i % PS.shape[2]}]", dmu_dpsi=float(gK.reshape(-1)[i]), sd_psi=float(PS.std(axis=0, ddof=1).reshape(-1)[i]),
                             leverage_counts=float(L.reshape(-1)[i]), leverage_frac_of_obs=float(L.reshape(-1)[i] / obsK),
                             corr_with_t=[corr(PS.reshape(n, -1)[:, i], T[:, KK2]) for KK2 in range(KK)]) for i in order]
    res["leverage_psi_c"] = lev
    # ---- correlations of t with the population and the completeness summaries
    prox = dict(dndx_ge20p0_allz=np.asarray(red["dndx_dla_20p0_allz"]), dndx_ge20p3_allz=np.asarray(red["dndx_dla_20p3_allz"]), omega_20p3_21p6=omega_allz,
                f_subdla_19p7_20p3_allz=f_sub, f_subdla_19p7_20p3_z0=f_sub_z0, C_bar_subDLA=Cb, d_bar_subDLA=dbar, logLambda=logL)
    for K in range(KK):
        prox[f"dndx_ge20p3_z{K}"] = np.asarray(red["dndx_dla_20p3_coarse"])[:, K]
    res["corr_t_vs"] = {k: [corr(T[:, K], v) for K in range(KK)] for k, v in prox.items()}
    # ---- low-dimensional mode diagnostics (A0,A1,A2,logL,Cbar,f_sub): GMM 1 vs 2 by BIC, dip statistic on the leading PC, chain/arm occupancy
    Z = np.column_stack([A, logL, Cb, f_sub]); Zsd = Z.std(axis=0); Zsd[Zsd == 0] = 1.0; Zs = (Z - Z.mean(axis=0)) / Zsd
    modes = dict(coords=["A0", "A1", "A2", "logLambda", "C_bar_subDLA", "f_subDLA"])
    try:
        from sklearn.mixture import GaussianMixture
        bic = {k: float(GaussianMixture(k, n_init=3, random_state=0).fit(Zs).bic(Zs)) for k in (1, 2, 3)}
        g2 = GaussianMixture(2, n_init=3, random_state=0).fit(Zs); lab = g2.predict(Zs)
        modes.update(bic=bic, gmm2_weights=g2.weights_.round(4).tolist(), gmm2_means_std_units=g2.means_.round(3).tolist(),
                     gmm2_component_by_arm=[[int((lab[arm_id == i] == c).sum()) for c in range(2)] for i in range(len(arms))],
                     gmm2_component_by_chain=[[int((lab[(arm_id == i) & (chain_id == c)] == 1).sum()) for c in range(2)] for i in range(len(arms))],
                     gmm2_mean_separation_mahalanobis=float(np.sqrt(np.sum((g2.means_[0] - g2.means_[1])**2))))
    except Exception as e:
        modes["gmm_error"] = str(e)
    pc = np.linalg.svd(Zs, full_matrices=False)[2][0]; proj = Zs @ pc
    hist, edges = np.histogram(proj, bins=40); modes["pc1_hist"] = dict(counts=hist.tolist(), edges=edges.round(3).tolist(), loading=pc.round(3).tolist())
    modes["pe_by_arm_chain"] = [[float(PE[(arm_id == i) & (chain_id == c)].mean()) for c in range(2)] for i in range(len(arms))]
    res["modes"] = modes
    # ---- extra arms (e.g. the mirror chain), summarised separately, never pooled
    res["extra_arms"] = {}
    for spec in a.extra_arm:
        p, lab = spec.split(":"); z = load_arm(p)
        Tx = np.asarray(z["t"]); LFx = np.asarray(z["lam_fp"]); PSx = np.asarray(z["psi_c"]); Fx = np.asarray(z["f"]); PEx = np.asarray(z["potential_energy"])
        nch = Tx.shape[0]; out = {}
        for c in range(nch):
            Lc = LFx[c].sum(axis=(1, 2)); Ac = np.log(Lc)[:, None] + Tx[c]
            redc = reduce_f_posterior(Fx[c], pk)
            out[f"chain{c}"] = dict(pe_mean=float(PEx[c].mean()), t_mean=Tx[c].mean(axis=0).tolist(), t_sd=Tx[c].std(axis=0, ddof=1).tolist(), logL_mean=float(np.log(Lc).mean()),
                                    A_mean=Ac.mean(axis=0).tolist(), C_bar=q(Cbar(PSx[c]), (16, 50, 84)), d_bar=q(np.einsum("nsb,bs->n", PSx[c][:, :, b2c[mS]], w_bs) / w_bs.sum(), (16, 50, 84)),
                                    dndx_ge20p3_allz=q(redc["dndx_dla_20p3_allz"]), dndx_ge20p0_allz=q(redc["dndx_dla_20p0_allz"]),
                                    f_subdla_allz=q((np.einsum("dbk,b->dk", Fx[c][:, mS, :], dN[mS]) * dXk[None]).sum(axis=1) / dXk.sum()))
        res["extra_arms"][lab] = dict(path=p, sha256=sha256(p), chains=out)
    # ---- likelihood-term accounting at three points: a representative posterior draw (median PE), the same with t=0, the mirror chain mean if given
    from numpyro import handlers
    counts_j = jnp.asarray(np.asarray(pk.counts, float)); fpc_j = jnp.asarray(np.asarray(pk.fp_counts, float))
    def terms_at(point):
        tr = handlers.trace(handlers.substitute(handlers.seed(model_cc, 0), data=point)).get_trace(consts, Mg, counts=counts_j, fp_counts=fpc_j, fp_mode="informative_ln")
        out = {}
        for k, v in tr.items():
            if v["type"] == "sample":
                lp = v["fn"].log_prob(v["value"])
                if v.get("mask") is not None:
                    lp = jnp.where(v["mask"], lp, 0.0)
                out[k] = float(jnp.sum(lp))
        out["prior_total"] = float(sum(v for k, v in out.items() if k != "counts")); out["total_log_joint"] = float(out["prior_total"] + out["counts"])
        return out
    i_rep = int(np.argsort(PE)[len(PE) // 2])
    def point_from_X(i):
        d = {}; off = 0
        for k in SITE_ORDER:
            shp = {"sigma_N": (), "sigma_z": (), "theta_level": (), "theta_slope": (), "eps_N": (consts.n_b - 2,), "eps_z": (consts.n_b, consts.n_k - 1),
                   "psi_c": tuple(np.asarray(consts.sigma_hat).shape), "fp_lam_total": (), "fp_shape_v": (consts.n_c * consts.n_s,), "t": (KK,)}[k]
            sz = int(np.prod(shp)) if shp else 1
            d[k] = jnp.asarray(X[i, off:off + sz].reshape(shp)); off += sz
        return d
    pt = point_from_X(i_rep); pt0 = dict(pt, t=jnp.zeros(KK))
    acct = dict(representative_draw=dict(index=i_rep, potential_energy=float(PE[i_rep]), terms=terms_at(pt)), same_draw_t0=dict(terms=terms_at(pt0)))
    acct["delta_t_to_0"] = {k: acct["same_draw_t0"]["terms"][k] - acct["representative_draw"]["terms"][k] for k in acct["representative_draw"]["terms"]}
    acct["note"] = "per-site log densities of the frozen model_cc (prior terms + the single counts likelihood, masked to dX>0); no fp_counts term exists under informative_ln"
    res["likelihood_terms"] = acct
    json.dump(res, open(a.out, "w"), indent=1)
    # ---- compact print
    print(json.dumps(dict(n=n, t={k: (round(v["median"], 4), round(v["median_over_sigma"], 2), round(v["post_over_prior_width"], 3)) for k, v in res["t"].items()},
                          logL=(round(res["log_Lambda"]["median"], 4), round(res["log_Lambda"]["sd"], 4)), A={k: (round(v["sd"], 4), round(v["sd_A_over_sd_t"], 3)) for k, v in res["A_K"].items()},
                          corr_logL_t=[res["ridge_pairs"][f"K{K}"]["corr"] for K in range(KK)], Cbar=[round(x, 4) for x in res["completeness"]["C_bar"]],
                          dbar=[round(x, 4) for x in res["completeness"]["d_bar"]], corr_t_Cbar=res["completeness"]["corr_t_Cbar"],
                          gmm_bic=res["modes"].get("bic"), sci20p3=[round(x, 5) for x in sci["dndx_ge20p3_allz"]], omega=[float(f"{x:.4e}") for x in sci["omega_20p3_21p6_allz_h0p70"]]), indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
