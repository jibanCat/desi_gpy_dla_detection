#!/usr/bin/env python
"""HZ2 coarse threshold-flow bookkeeping from the real posterior draws (PI ruling 2026-09-03 evening §6–§9).

Uses the deployed model's exact predictive expressions (cc_real_posterior.py G_A block): for each retained seed's draws
(theta_pop, psi_c, t, lam_fp) the expected observed rows by (observed bin c, latent bin b) are
    tp[c,b] = Σ_{k,s} Mg[s,k,c,b] · sigmoid(eta_hat + psi_c)[s, cell(b)] · g_bk[b,k] · exp(theta_pop[b,k]) · dN_b[b] · dX[k,s]
and the FP rows by observed bin
    fp[c]   = Σ_{k,s} fp_w · fp_ell_eff · (1 − fp_eta_c[c]) · exp(t[K(k)]) · lam_fp[c,s] · fp_E[k,s].
At a threshold T (20.3 and 20.0; observed bins c with lower edge ≥ T; latent bins b with lower edge ≥ T):
    N_stay = Σ_{c≥T,b≥T} tp,  N_up = Σ_{c≥T,b<T} tp,  N_down = Σ_{c<T,b≥T} tp,  N_FP = Σ_{c≥T} fp,
    N_obs^pred(≥T) = N_stay + N_up + N_FP  (compared with the observed catalogue rows ≥ T),
    latent incidence dN/dX(≥T) = Σ_{b≥T,k} f[b,k] dN_b dX_k / Σ_k dX_k (the pooled estimand; checked against the pooled JSON),
    and the 'naive' inversion  I_naive(≥T) = (N_obs(≥T) − N_FP) / ⟨C·g⟩_{b≥T} / X_tot with ⟨C·g⟩ the posterior's own completeness of latent ≥T
    systems — i.e. what a completeness-corrected count with NO migration bookkeeping would infer under the same C and FP.
Only the coarse flows are written; no fine-bin f(N) is reported.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from CDDF_analysis.hbi_mcmc.pack import load_pack  # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors  # noqa: E402


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--run-dir", required=True); ap.add_argument("--pooled", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--thin", type=int, default=5, help="use every k-th draw per chain")
    a = ap.parse_args(argv)
    pk = load_pack(a.pack, allow_nonstandard_grid=True); consts, Mg = build_cc_tensors(pk); Mg = np.asarray(Mg)   # (S, Kf, C, B)
    eta = np.asarray(consts.eta_hat); b2c = np.asarray(consts.b_to_cell); g = np.asarray(consts.g_bk); dN = np.asarray(consts.dN_b); dX = np.asarray(consts.dX)
    fp_w, fp_ell, fp_eta = float(consts.fp_w), float(consts.fp_ell_eff), np.asarray(consts.fp_eta_c); kz = np.asarray(consts.kz_to_K); fpE = np.asarray(consts.fp_E)
    ne = np.asarray(pk.nhat_edges, float); nt = np.asarray(pk.ntrue_edges, float); counts = np.asarray(pk.counts, float)
    obs_c = counts.sum(axis=(1, 2)); Xk = dX.sum(axis=1); Xtot = Xk.sum()
    P = json.load(open(a.pooled)); inc = [(e["seed"], e["deep"]) for e in P["selection"]["included"]]
    res = {}
    T_list = (20.3, 20.0); acc = {T: dict(stay=[], up=[], down=[], fp=[], pred=[], obs=float(obs_c[ne[:-1] >= T - 1e-9].sum()), inc=[], naive=[], Ceff=[]) for T in T_list}
    tot_pred = []; tot_fp = []
    for seed, deep in inc:
        z = np.load(os.path.join(a.run_dir, f"REAL_ln_{'deep_' if deep else ''}s{seed}_nuisance.npz"))
        th, pc, t, lf = z["theta_pop"], z["psi_c"], z["t"], z["lam_fp"]   # (chains, draws, ...)
        for ch in range(th.shape[0]):
            for d in range(0, th.shape[1], a.thin):
                f = np.exp(th[ch, d]); Cc = sig(eta + pc[ch, d])[:, b2c]          # (B,K), (S,B)
                w = g * f * dN[:, None]                                            # (B,K)
                tp_cb = np.einsum("skcb,sb,bk,ks->cb", Mg, Cc, w, dX)             # (C,B)
                fp_c = fp_w * fp_ell * (1.0 - fp_eta)[:, None, None] * np.exp(t[ch, d][kz])[None, :, None] * lf[ch, d][:, None, :] * fpE[None, :, :]
                fp_c = fp_c.sum(axis=(1, 2))                                       # (C,)
                tot_pred.append(tp_cb.sum() + fp_c.sum()); tot_fp.append(fp_c.sum())
                for T in T_list:
                    cm = ne[:-1] >= T - 1e-9; bm = nt[:-1] >= T - 1e-9
                    stay = tp_cb[np.ix_(cm, bm)].sum(); up = tp_cb[np.ix_(cm, ~bm)].sum(); down = tp_cb[np.ix_(~cm, bm)].sum(); fpT = fp_c[cm].sum()
                    inc_T = (f[bm] * dN[bm, None] * Xk[None, :]).sum() / Xtot
                    # posterior's own completeness of latent ≥T systems: detected-in-grid rows from b≥T (stay+down) / latent ≥T systems on the path
                    n_lat = (f[bm] * dN[bm, None] * Xk[None, :]).sum(); Ceff = (stay + down) / max(n_lat, 1e-12)
                    naive = (acc[T]["obs"] - fpT) / max(Ceff, 1e-12) / Xtot
                    A = acc[T]; A["stay"].append(stay); A["up"].append(up); A["down"].append(down); A["fp"].append(fpT); A["pred"].append(stay + up + fpT); A["inc"].append(inc_T); A["naive"].append(naive); A["Ceff"].append(Ceff)
    def q(x): x = np.asarray(x); return dict(median=float(np.median(x)), p16=float(np.percentile(x, 16)), p84=float(np.percentile(x, 84)), mean=float(x.mean()))
    res["n_draws_used"] = len(tot_pred); res["observed_rows_total"] = float(obs_c.sum()); res["predicted_rows_total"] = q(tot_pred); res["predicted_fp_total"] = q(tot_fp)
    res["level_posterior_mean_over_obs"] = float(np.mean(tot_pred) / obs_c.sum()); res["X_tot"] = float(Xtot)
    for T in T_list:
        A = acc[T]; r = {k: q(A[k]) for k in ("stay", "up", "down", "fp", "pred", "inc", "naive", "Ceff")}; r["obs_rows"] = A["obs"]
        pred = np.asarray(A["pred"]); r["fractions_of_pred_obs"] = dict(stay=float(np.mean(np.asarray(A["stay"]) / pred)), up=float(np.mean(np.asarray(A["up"]) / pred)), fp=float(np.mean(np.asarray(A["fp"]) / pred)))
        r["net_migration_rows"] = q(np.asarray(A["up"]) - np.asarray(A["down"])); r["pred_over_obs"] = q(pred / A["obs"])
        r["naive_over_inc"] = q(np.asarray(A["naive"]) / np.asarray(A["inc"]))
        res[f"T{T}"] = r
        print(f"T={T}: observed rows {A['obs']:.0f} | predicted {r['pred']['median']:.1f} = stay {r['stay']['median']:.1f} + up {r['up']['median']:.1f} + FP {r['fp']['median']:.1f} (fractions {r['fractions_of_pred_obs']}) | down {r['down']['median']:.1f} | net up−down {r['net_migration_rows']['median']:+.1f}")
        print(f"       latent dN/dX(≥{T}) {r['inc']['median']:.4f} [{r['inc']['p16']:.4f},{r['inc']['p84']:.4f}] | posterior-own completeness of latent ≥T {r['Ceff']['median']:.3f} | naive no-migration inversion {r['naive']['median']:.4f} (naive/inc {r['naive_over_inc']['median']:.3f})")
    print(f"total rows: observed {obs_c.sum():.0f}, predicted posterior mean {np.mean(tot_pred):.1f} (level {res['level_posterior_mean_over_obs']:.4f}); FP total {np.mean(tot_fp):.1f}")
    # per-observed-bin FP and TP expectations above 20.0 (coarse)
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
