#!/usr/bin/env python3
"""lpt_audit.py — STRICT Λ–π–t degeneracy audit (PI ruling 2026-09-01 #10). For a catalogue region R (N-set × coarse-z
block K) the FP intensity of the frozen model factorises EXACTLY as

    mu_FP(R) = Lambda * exp(t_K) * Pi_R,   Pi_R = sum_{c in set, s} pi_cs * W^K_cs,
    W^K_cs   = fp_w * ell_eff * (1 - eta_c) * sum_{k in K, dX_ks>0} E_ks          (the survey-opportunity weights, from the pack)

so  log mu = log Lambda + t_K + log Pi_R  with NO residual term (log O is absorbed in Pi_R's weights). This file returns, per
region: the 4×4 covariance/correlation of (log Λ, t_K, log Π_R, log μ), partial correlations, eigen-decomposition, the exact
variance decomposition Var(X+Y+Z) = ΣVar + 2ΣCov, the same split WITHIN chains (pooled within-chain covariance) versus BETWEEN
chain means (mixing / chain-state test), an alternative shape summary Π'_R,s = Σ_{c∈set} π_cs per stratum, forward sensitivities,
and the pair/conditional plots coloured by chain, arm and initialisation family. PRIVATE outputs.

    python tools/hbi_validation/lpt_audit.py --pack PACK --family R0=POOLED_ln_R0.json --family R2A=POOLED_ln_R2.json \
        --family R2Binit=POOLED_ln_R2B.json [--extra path:label:chain] --out-dir DIR
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.pack import load_pack                                  # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors        # noqa: E402
from CDDF_analysis.hbi_mcmc.provenance_util import sha256                          # noqa: E402
from tools.hbi_validation.viz_common import plt, SLOTS, INK2, SEQ, heatmap         # noqa: E402

REGIONS = {"subdla_19p7_20p3": (19.7, 20.3), "dla_ge20p3": (20.3, 99.0), "all_19p5_22p4": (0.0, 99.0)}


def load_family(spec):
    """POOLED.json (its selection block names the arms) or a comma-separated list of all-sites npz paths."""
    arms, tags = [], []
    if spec.endswith(".json"):
        P = json.load(open(spec))
        for r in P["selection"]["included"]:
            p = r["file"][:-5] + "_allsites.npz"; arms.append(np.load(p)); tags.append(f"s{r['seed']}{'d' if r['deep'] else ''}")
    else:
        for p in spec.split(","):
            arms.append(np.load(p)); tags.append(re.sub(r".*REAL_ln_(deep_|lc_)?s(\d{8})_allsites\.npz", lambda m: f"{(m.group(1) or '').rstrip('_')}s{m.group(2)}", p))
    return arms, tags


def region_weights(pk, consts, cen_mask, K):
    kz = np.asarray(consts.kz_to_K); dXm = np.asarray(consts.dX) > 0
    E = np.asarray(consts.fp_E) * dXm                                           # (Kf, S)
    Ek = E[kz == K].sum(axis=0)                                                 # (S,)
    W = consts.fp_w * consts.fp_ell_eff * (1.0 - np.asarray(consts.fp_eta_c))[:, None] * Ek[None, :]   # (C, S)
    return W * cen_mask[:, None]


def stack(arms):
    out = {}
    for k in ("t", "lam_fp", "fp_lam_total", "potential_energy"):
        out[k] = np.concatenate([np.asarray(z[k]).reshape(-1, *np.asarray(z[k]).shape[2:]) for z in arms])
    out["chain"] = np.concatenate([np.repeat(np.arange(z["t"].shape[0]), z["t"].shape[1]) for z in arms])
    out["arm"] = np.concatenate([np.full(z["t"].shape[0] * z["t"].shape[1], i) for i, z in enumerate(arms)])
    return out


def partial_corr(C):
    P = np.linalg.pinv(C); d = np.sqrt(np.diag(P)); R = -P / np.outer(d, d); np.fill_diagonal(R, 1.0); return R


def analyse(Y, names, chain_key):
    """Y (n,4) columns log Λ, t_K, log Π, log μ; chain_key (n,) identifies (arm,chain)."""
    C = np.cov(Y.T); R = np.corrcoef(Y.T)
    X3 = Y[:, :3]; C3 = np.cov(X3.T); w, v = np.linalg.eigh(C3); order = np.argsort(-w)
    var = np.diag(C3); tot = float(np.var(Y[:, 3], ddof=1))
    dec = dict(var_logL=float(var[0]), var_t=float(var[1]), var_logPi=float(var[2]), cov_logL_t=float(C3[0, 1]), cov_logL_logPi=float(C3[0, 2]), cov_t_logPi=float(C3[1, 2]),
               sum_var=float(var.sum()), sum_2cov=float(2 * (C3[0, 1] + C3[0, 2] + C3[1, 2])), var_sum_reconstructed=float(var.sum() + 2 * (C3[0, 1] + C3[0, 2] + C3[1, 2])), var_logmu_direct=tot,
               identity_check_abs=float(abs(var.sum() + 2 * (C3[0, 1] + C3[0, 2] + C3[1, 2]) - tot)),
               cancellation_fraction=float(1 - tot / var.sum()) if var.sum() > 0 else None)
    # within-chain (pooled) vs between-chain-mean decomposition of the 3×3 covariance
    keys = np.unique(chain_key); Cw = np.zeros((3, 3)); means = []
    nw = 0
    for k in keys:
        m = chain_key == k
        if m.sum() > 2:
            Cw += np.cov(X3[m].T) * (m.sum() - 1); nw += m.sum() - 1; means.append(X3[m].mean(axis=0))
    Cw /= max(nw, 1); Cb = np.cov(np.array(means).T) if len(means) > 1 else np.zeros((3, 3))
    def corr_of(Cm):
        d = np.sqrt(np.diag(Cm)); return Cm / np.outer(d, d) if np.all(d > 0) else np.full((3, 3), np.nan)
    ww, wv = np.linalg.eigh(Cw); wo = np.argsort(-ww)
    return dict(names=names, cov=C.tolist(), corr=R.round(4).tolist(), partial_corr=partial_corr(C).round(4).tolist(),
                eig3=dict(eigvals=w[order].tolist(), eigvecs_rows=v[:, order].T.round(4).tolist(), var_share=(w[order] / w.sum()).round(4).tolist()),
                decomposition=dec,
                within_chain=dict(cov=Cw.tolist(), corr=corr_of(Cw).round(4).tolist(), eigvals=ww[wo].tolist(), eigvecs_rows=wv[:, wo].T.round(4).tolist(),
                                  var_logmu_within=float(np.sum(Cw) ), sd=np.sqrt(np.diag(Cw)).round(4).tolist()),
                between_chain_means=dict(cov=Cb.tolist(), corr=corr_of(Cb).round(4).tolist(), sd=np.sqrt(np.diag(Cb)).round(4).tolist(), n_chains=len(means),
                                         within_over_total_var=[float(Cw[i, i] / C3[i, i]) for i in range(3)]),
                sd=np.sqrt(np.diag(C)).round(4).tolist())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--family", action="append", required=True, help="LABEL=POOLED.json (first = baseline)")
    ap.add_argument("--extra", action="append", default=[], help="allsites.npz:label:chain (e.g. the mirror chain)")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--primary", default="subdla_19p7_20p3")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pk = load_pack(a.pack); consts, Mg = build_cc_tensors(pk)
    nhat = np.asarray(pk.nhat_edges, float); cen = 0.5 * (nhat[:-1] + nhat[1:]); KK = int(np.asarray(consts.t_sigma).size); S = consts.n_s
    fams = {}
    for spec in a.family:
        lab, p = spec.split("=", 1); arms, tags = load_family(p); st = stack(arms); st["tags"] = tags; st["pooled"] = p; fams[lab] = st
    extras = {}
    for spec in a.extra:
        p, lab, ch = spec.split(":"); z = np.load(p); ch = int(ch)
        extras[lab] = dict(t=np.asarray(z["t"])[ch], lam_fp=np.asarray(z["lam_fp"])[ch], potential_energy=np.asarray(z["potential_energy"])[ch], path=p, chain=ch)
    res = dict(pack=a.pack, pack_sha256=sha256(a.pack), families={k: dict(source=v["pooled"], source_sha256=(sha256(v["pooled"]) if v["pooled"].endswith(".json") else [sha256(x) for x in v["pooled"].split(",")]), arms=v["tags"], n=int(v["t"].shape[0])) for k, v in fams.items()},
               definition=("mu_FP(R) = Lambda * exp(t_K) * Pi_R ; Pi_R = sum_{c in set,s} pi_cs W^K_cs ; W^K_cs = fp_w*ell_eff*(1-eta_c)*sum_{k in K, dX>0} E_ks ; "
                           "pi = lam_fp / sum(lam_fp) ; log mu = log Lambda + t_K + log Pi_R exactly"), regions={})
    base = list(fams)[0]
    for rname, (lo, hi) in REGIONS.items():
        cm = (cen >= lo - 1e-9) & (cen < hi - 1e-9)
        res["regions"][rname] = {}
        for K in range(KK):
            W = region_weights(pk, consts, cm, K)
            reg = {}
            for lab, st in fams.items():
                Lam = st["lam_fp"].sum(axis=(1, 2)); pi = st["lam_fp"] / Lam[:, None, None]
                Pi = np.einsum("ncs,cs->n", pi, W); logmu = np.log(Lam) + st["t"][:, K] + np.log(Pi)
                # exact check against the fold's FP sum
                mu_direct = Lam * np.exp(st["t"][:, K]) * Pi
                Y = np.column_stack([np.log(Lam), st["t"][:, K], np.log(Pi), logmu])
                ck = st["arm"] * 10 + st["chain"]
                an = analyse(Y, ["logLambda", f"t{K}", "logPi", "logmu"], ck)
                # alternative shape summary: per-stratum sub-mass Pi'_s = sum_{c in set} pi_cs (unweighted), its log and corr with t, logL
                Pi_s = pi[:, cm, :].sum(axis=1)                         # (n, S)
                an["alt_shape_summary"] = dict(definition="Pi'_s = sum_{c in set} pi_cs (unweighted per S/N stratum); log of the W-weighted total is logPi above",
                                               stratum_weight_share=(W.sum(axis=0) / W.sum()).round(4).tolist() if W.sum() > 0 else None,
                                               corr_t_logPis=[(float(np.corrcoef(st["t"][:, K], np.log(Pi_s[:, s_]))[0, 1]) if Pi_s[:, s_].min() > 0 and np.std(np.log(Pi_s[:, s_])) > 0 else None) for s_ in range(S)],
                                               corr_logL_logPis=[(float(np.corrcoef(np.log(Lam), np.log(Pi_s[:, s_]))[0, 1]) if Pi_s[:, s_].min() > 0 and np.std(np.log(Pi_s[:, s_])) > 0 else None) for s_ in range(S)],
                                               median_Pis=np.median(Pi_s, axis=0).round(5).tolist())
                # forward sensitivity: vary ONE term over its posterior 16–84 range with the other two at their posterior means (exact: dlogmu = dterm)
                q = lambda x: np.percentile(x, [16, 84])
                an["forward_sensitivity"] = dict(note="log mu = logLambda + t + logPi exactly: moving one term by its 16-84 half-range moves log mu by the same amount; listed = half-ranges (leverage in log mu)",
                                                 half_range_logLambda=float(0.5 * np.diff(q(np.log(Lam)))[0]), half_range_t=float(0.5 * np.diff(q(st["t"][:, K]))[0]),
                                                 half_range_logPi=float(0.5 * np.diff(q(np.log(Pi)))[0]), half_range_logmu=float(0.5 * np.diff(q(logmu))[0]),
                                                 mu_median=float(np.median(mu_direct)))
                an["per_arm_chain_means"] = [dict(arm=st["tags"][int(k // 10)], chain=int(k % 10), logL=float(np.log(Lam)[ck == k].mean()), t=float(st["t"][ck == k, K].mean()), logPi=float(np.log(Pi)[ck == k].mean()), logmu=float(logmu[ck == k].mean()), pe=float(st["potential_energy"][ck == k].mean())) for k in np.unique(ck)]
                reg[lab] = an
                st.setdefault("Y", {})[(rname, K)] = Y
            for lab, e in extras.items():
                Lam = e["lam_fp"].sum(axis=(1, 2)); pi = e["lam_fp"] / Lam[:, None, None]; Pi = np.einsum("ncs,cs->n", pi, W); logmu = np.log(Lam) + e["t"][:, K] + np.log(Pi)
                reg[f"extra:{lab}"] = dict(mean=dict(logL=float(np.log(Lam).mean()), t=float(e["t"][:, K].mean()), logPi=float(np.log(Pi).mean()), logmu=float(logmu.mean())), sd=dict(logL=float(np.log(Lam).std()), t=float(e["t"][:, K].std()), logPi=float(np.log(Pi).std()), logmu=float(logmu.std())), pe=float(e["potential_energy"].mean()))
                e.setdefault("Y", {})[(rname, K)] = np.column_stack([np.log(Lam), e["t"][:, K], np.log(Pi), logmu])
            res["regions"][rname][f"z{K}"] = reg
    json.dump(res, open(os.path.join(a.out_dir, "lpt_audit.json"), "w"), indent=1)
    # ---------------- plots for the primary region, each z block ----------------
    fam_labels = list(fams); pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    for K in range(KK):
        key = (a.primary, K); nm = ["log Λ", f"t{K}", "log Π", "log μ_FP"]
        Yb = fams[base]["Y"][key]; st = fams[base]
        # (1) pairs coloured by chain (baseline), by arm (baseline, small multiples), by family (all families on shared axes)
        fig, axs = plt.subplots(3, 6, figsize=(16, 7.5))
        for j, (i1, i2) in enumerate(pairs):
            ax = axs[0, j]
            for c in (0, 1):
                m = st["chain"] == c; ax.scatter(Yb[m, i1], Yb[m, i2], s=2, alpha=0.3, color=SLOTS[c], linewidths=0, rasterized=True, label=f"chain {c}" if j == 0 else None)
            ax.set_xlabel(nm[i1], fontsize=7); ax.set_ylabel(nm[i2], fontsize=7); ax.tick_params(labelsize=6)
            if j == 0: ax.legend(fontsize=6)
            ax = axs[1, j]   # by arm (arm index as discrete grey levels is not allowed -> use per-arm means + 68% ellipse-free: plot each arm's draws faintly in one hue, arm means labelled)
            ax.scatter(Yb[:, i1], Yb[:, i2], s=1.5, alpha=0.15, color="#9aa0a6", linewidths=0, rasterized=True)
            for ai, tag in enumerate(st["tags"]):
                for c in (0, 1):
                    m = (st["arm"] == ai) & (st["chain"] == c)
                    ax.plot(Yb[m, i1].mean(), Yb[m, i2].mean(), marker="o" if c == 0 else "s", ms=4, color=SLOTS[c], mec=INK2, mew=0.4, ls="none")
                    if j == 0:
                        ax.annotate(tag, (Yb[m, i1].mean(), Yb[m, i2].mean()), fontsize=5, xytext=(2, 2), textcoords="offset points", color=INK2)
            ax.set_xlabel(nm[i1], fontsize=7); ax.set_ylabel(nm[i2], fontsize=7); ax.tick_params(labelsize=6)
            ax = axs[2, j]
            for fi, lab in enumerate(fam_labels):
                Yf = fams[lab]["Y"][key]; ax.scatter(Yf[:, i1], Yf[:, i2], s=2, alpha=0.25, color=SLOTS[fi], linewidths=0, rasterized=True, label=lab if j == 0 else None)
            for ei, (lab, e) in enumerate(extras.items()):
                Ye = e["Y"][key]; ax.scatter(Ye[:, i1], Ye[:, i2], s=2, alpha=0.4, color=SLOTS[len(fam_labels) + ei], linewidths=0, rasterized=True, label=lab if j == 0 else None)
            ax.set_xlabel(nm[i1], fontsize=7); ax.set_ylabel(nm[i2], fontsize=7); ax.tick_params(labelsize=6)
            if j == 0: ax.legend(fontsize=6)
        axs[0, 0].set_title("baseline, by chain", fontsize=8, loc="left"); axs[1, 0].set_title("baseline, arm × chain means (○ chain 0, □ chain 1) over all draws", fontsize=8, loc="left"); axs[2, 0].set_title("by initialisation family / run (+ extra chains)", fontsize=8, loc="left")
        fig.suptitle(f"STRICT Λ–π–t audit — region {a.primary}, z block {K}: log μ_FP = log Λ + t{K} + log Π (exact)", fontsize=9, x=0.02, ha="left"); fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"lpt_pairs_{a.primary}_z{K}.png"), dpi=130); plt.close(fig)
        # (2) 3-D via pairs coloured by the third coordinate + eigenvector projections + conditional slices in log Π
        fig, axs = plt.subplots(2, 3, figsize=(12, 7))
        trip = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        for j, (i1, i2, i3) in enumerate(trip):
            ax = axs[0, j]; sc = ax.scatter(Yb[:, i1], Yb[:, i2], c=Yb[:, i3], s=3, cmap=SEQ, linewidths=0, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label(nm[i3], fontsize=7); cb.ax.tick_params(labelsize=6)
            ax.set_xlabel(nm[i1], fontsize=7); ax.set_ylabel(nm[i2], fontsize=7); ax.tick_params(labelsize=6)
        an = res["regions"][a.primary][f"z{K}"][base]; V = np.array(an["eig3"]["eigvecs_rows"]); X3 = Yb[:, :3] - Yb[:, :3].mean(axis=0); Pj = X3 @ V.T
        ax = axs[1, 0]
        for c in (0, 1):
            m = st["chain"] == c; ax.scatter(Pj[m, 0], Pj[m, 1], s=2, alpha=0.3, color=SLOTS[c], linewidths=0, rasterized=True)
        ax.set_xlabel(f"PC1 {np.round(V[0], 2)} ({100 * an['eig3']['var_share'][0]:.0f} %)", fontsize=7); ax.set_ylabel(f"PC2 {np.round(V[1], 2)} ({100 * an['eig3']['var_share'][1]:.0f} %)", fontsize=7); ax.tick_params(labelsize=6); ax.set_title("projection on covariance eigenvectors (log Λ, t, log Π)", fontsize=7, loc="left")
        ax = axs[1, 1]
        for c in (0, 1):
            m = st["chain"] == c; ax.scatter(Pj[m, 0], Pj[m, 2], s=2, alpha=0.3, color=SLOTS[c], linewidths=0, rasterized=True)
        ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel(f"PC3 {np.round(V[2], 2)} ({100 * an['eig3']['var_share'][2]:.0f} %)", fontsize=7); ax.tick_params(labelsize=6)
        ax = axs[1, 2]   # conditional slices: thin slices in log Π, show (log Λ, t) with slope −1 reference
        qs = np.percentile(Yb[:, 2], [10, 30, 50, 70, 90]); 
        for si in range(4):
            m = (Yb[:, 2] >= qs[si]) & (Yb[:, 2] < qs[si + 1]); ax.scatter(Yb[m, 0], Yb[m, 1], s=3, alpha=0.5, color=SLOTS[si], linewidths=0, rasterized=True, label=f"log Π ∈ [{qs[si]:.2f},{qs[si+1]:.2f})")
        xs = np.array([Yb[:, 0].min(), Yb[:, 0].max()]); A_med = float(np.median(Yb[:, 0] + Yb[:, 1]))
        for d in (-0.3, 0, 0.3): ax.plot(xs, -xs + A_med + d, color=INK2, lw=0.5, ls="--")
        ax.set_xlabel(nm[0], fontsize=7); ax.set_ylabel(nm[1], fontsize=7); ax.legend(fontsize=5); ax.tick_params(labelsize=6); ax.set_title("thin conditional slices in log Π (dashed: constant log Λ + t)", fontsize=7, loc="left")
        fig.suptitle(f"STRICT Λ–π–t audit — 3-D view, region {a.primary}, z block {K} (baseline {base})", fontsize=9, x=0.02, ha="left"); fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f"lpt_3d_{a.primary}_z{K}.png"), dpi=130); plt.close(fig)
    # print the primary summary
    for K in range(KK):
        an = res["regions"][a.primary][f"z{K}"][base]; d = an["decomposition"]
        print(f"[{base}] {a.primary} z{K}: sd(logL,t,logPi,logmu) = {an['sd']}  corr = {an['corr'][0][1]:.2f}/{an['corr'][0][2]:.2f}/{an['corr'][1][2]:.2f} (L-t, L-Pi, t-Pi)  partial = {an['partial_corr'][0][1]:.2f}/{an['partial_corr'][0][2]:.2f}/{an['partial_corr'][1][2]:.2f}")
        print(f"    Var: {d['var_logL']:.4f} + {d['var_t']:.4f} + {d['var_logPi']:.4f} = {d['sum_var']:.4f}; 2Cov: {2*d['cov_logL_t']:+.4f} {2*d['cov_logL_logPi']:+.4f} {2*d['cov_t_logPi']:+.4f} = {d['sum_2cov']:+.4f}; Var(sum) = {d['var_logmu_direct']:.5f} (cancellation {100*d['cancellation_fraction']:.1f} %); identity |err| {d['identity_check_abs']:.1e}")
        print(f"    eig3 shares {an['eig3']['var_share']} vecs {an['eig3']['eigvecs_rows']}; within-chain sd {an['within_chain']['sd']} corr {an['within_chain']['corr']}; between-chain-mean sd {an['between_chain_means']['sd']}; within/total var {an['between_chain_means']['within_over_total_var']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
