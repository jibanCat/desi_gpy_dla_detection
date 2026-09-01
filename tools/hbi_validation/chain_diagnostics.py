#!/usr/bin/env python3
"""chain_diagnostics.py — the MCMC diagnostics bundle required by the PI's transparency standard (ruling 2026-09-01 #8):
per chain, for the key nuisance and science coordinates (log Λ, t_K, log Π_subDLA,z0, A_K, C̄_subDLA, dN/dX ≥20.0 / ≥20.3) and the
potential energy: trace plots, rank plots (rank of each draw within the pooled chains), autocorrelation, split-R̂ / bulk ESS per chain
pair, and the per-site R̂ / ESS table for all sampled scalars. Works for a production arm (2 × 500) or the long chain (2 × 4000).
PRIVATE outputs.

    python tools/hbi_validation/chain_diagnostics.py --pack PACK --allsites REAL_ln_lc_s20260822_allsites.npz --label LC_s22 --out-dir DIR
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.pack import load_pack                                 # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors       # noqa: E402
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior                     # noqa: E402
from CDDF_analysis.hbi_mcmc.provenance_util import sha256                         # noqa: E402
from tools.hbi_validation.site_mapping import build_mapping, flatten_draws        # noqa: E402
from tools.hbi_validation.geometry import split_rhat_ess, SUBDLA                  # noqa: E402
from tools.hbi_validation.lpt_audit import region_weights                         # noqa: E402
from tools.hbi_validation.viz_common import plt, SLOTS, INK2                      # noqa: E402


def acf(x, maxlag=200):
    x = np.asarray(x, float) - np.mean(x); n = x.size
    f = np.fft.rfft(x, 2 * n); a = np.fft.irfft(f * np.conj(f))[:n] / (x.var() * np.arange(n, 0, -1))
    return a[:maxlag]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--allsites", required=True); ap.add_argument("--label", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pk = load_pack(a.pack); consts, _ = build_cc_tensors(pk); pack = np.load(a.pack)
    z = np.load(a.allsites); nch, nd = z["t"].shape[:2]; KK = z["t"].shape[2]
    nhat = np.asarray(pk.nhat_edges, float); cen = 0.5 * (nhat[:-1] + nhat[1:]); cm = (cen >= SUBDLA[0] - 1e-9) & (cen < SUBDLA[1] - 1e-9)
    W0 = region_weights(pk, consts, cm, 0)
    ntrue = np.asarray(pk.ntrue_edges, float); Nc = 0.5 * (ntrue[:-1] + ntrue[1:]); dN = np.diff(ntrue); dX = np.asarray(consts.dX)
    mS = (Nc >= SUBDLA[0] - 1e-9) & (Nc < SUBDLA[1] - 1e-9); eta = np.asarray(consts.eta_hat); b2c = np.asarray(consts.b_to_cell); g = np.asarray(consts.g_bk)
    series = {}
    lam = np.asarray(z["lam_fp"]); Lam = lam.sum(axis=(2, 3)); pi = lam / Lam[..., None, None]
    series["log Λ"] = np.log(Lam)
    for K in range(KK):
        series[f"t{K}"] = np.asarray(z["t"])[:, :, K]
    series["log Π subDLA z0"] = np.log(np.einsum("hncs,cs->hn", pi, W0))
    series["A0 = logΛ+t0"] = series["log Λ"] + series["t0"]
    series["log μ_FP subDLA z0"] = series["A0 = logΛ+t0"] + series["log Π subDLA z0"]
    F = np.asarray(z["f"]); PS = np.asarray(z["psi_c"])
    fbar = F.reshape(-1, *F.shape[2:]).mean(axis=0); w_bs = np.einsum("bk,bk,ks->bs", g[mS] * dN[mS][:, None], fbar[mS], dX)
    Cc = 1 / (1 + np.exp(-(eta[None, None] + PS))); series["C̄ subDLA"] = np.einsum("hnsb,bs->hn", Cc[:, :, :, b2c[mS]], w_bs) / w_bs.sum()
    for c in range(nch):
        red = reduce_f_posterior(F[c], pk)
        series.setdefault("dN/dX ≥20.0", np.zeros((nch, nd)))[c] = np.asarray(red["dndx_dla_20p0_allz"])
        series.setdefault("dN/dX ≥20.3", np.zeros((nch, nd)))[c] = np.asarray(red["dndx_dla_20p3_allz"])
    series["potential energy"] = np.asarray(z["potential_energy"]).reshape(nch, nd)
    names = list(series)
    # ---- diagnostics table
    diag = {}
    for n in names:
        x = series[n]; rh, e, et = split_rhat_ess(x)
        diag[n] = dict(split_rhat=rh, ess_bulk=e, ess_tail=et, chain_means=x.mean(axis=1).tolist(), chain_sds=x.std(axis=1, ddof=1).tolist(),
                       ess_per_chain=[float(split_rhat_ess(x[c:c + 1])[1]) for c in range(nch)] if nd >= 4 else None,
                       acf_lag1=[float(acf(x[c], 2)[1]) for c in range(nch)], acf_int_time=[float(1 + 2 * np.sum(np.clip(acf(x[c], min(500, nd // 2))[1:], 0, None))) for c in range(nch)])
    X, chain = flatten_draws(z); rows = build_mapping(pack); site_names = [r["scalar_id"] for r in rows]
    rh_all = np.array([split_rhat_ess(col.reshape(nch, nd))[0] for col in X.T]); ess_all = np.array([split_rhat_ess(col.reshape(nch, nd))[1] for col in X.T])
    sites = dict(n=int(X.shape[1]), rhat_gt_1p05=int(np.nansum(rh_all > 1.05)), rhat_gt_1p10=int(np.nansum(rh_all > 1.10)), rhat_max=float(np.nanmax(rh_all)),
                 ess_min=float(np.nanmin(ess_all)), ess_lt_100=int(np.nansum(ess_all < 100)), ess_lt_400=int(np.nansum(ess_all < 400)),
                 worst_rhat=[(site_names[i], float(rh_all[i])) for i in np.argsort(-np.nan_to_num(rh_all, nan=0))[:10]],
                 worst_ess=[(site_names[i], float(ess_all[i])) for i in np.argsort(np.nan_to_num(ess_all, nan=1e9))[:10]],
                 by_site={s: dict(rhat_max=float(np.nanmax(rh_all[[i for i, r in enumerate(rows) if r["site"] == s]])), ess_min=float(np.nanmin(ess_all[[i for i, r in enumerate(rows) if r["site"] == s]])),
                                  n_rhat_gt_1p10=int(np.nansum(rh_all[[i for i, r in enumerate(rows) if r["site"] == s]] > 1.10))) for s in sorted(set(r["site"] for r in rows))})
    out = dict(label=a.label, allsites=a.allsites, sha256=sha256(a.allsites), chains=int(nch), draws_per_chain=int(nd), key=diag, all_sites=sites)
    json.dump(out, open(os.path.join(a.out_dir, f"chain_diagnostics_{a.label}.json"), "w"), indent=1)
    with open(os.path.join(a.out_dir, f"site_rhat_ess_{a.label}.csv"), "w") as fh:
        fh.write("corr_index,scalar_id,split_rhat,ess_bulk\n")
        for i, s in enumerate(site_names):
            fh.write(f"{i},{s},{rh_all[i]:.4f},{ess_all[i]:.1f}\n")
    # ---- trace + rank + acf figure
    nrow = len(names); fig, axs = plt.subplots(nrow, 3, figsize=(13, 1.35 * nrow), gridspec_kw=dict(width_ratios=[3, 1.2, 1]))
    pooled_rank = {n: np.argsort(np.argsort(series[n].reshape(-1))).reshape(nch, nd) for n in names}
    for i, n in enumerate(names):
        x = series[n]
        for c in range(nch):
            axs[i, 0].plot(x[c], color=SLOTS[c], lw=0.4, alpha=0.8, rasterized=True)
            axs[i, 1].hist(pooled_rank[n][c], bins=20, histtype="step", color=SLOTS[c], lw=0.8)
            axs[i, 2].plot(acf(x[c], min(200, nd // 2)), color=SLOTS[c], lw=0.8)
        axs[i, 0].set_ylabel(n, fontsize=6); axs[i, 0].tick_params(labelsize=5); axs[i, 1].tick_params(labelsize=5); axs[i, 2].tick_params(labelsize=5)
        axs[i, 1].axhline(nd / 20, color=INK2, lw=0.5, ls="--"); axs[i, 2].axhline(0, color=INK2, lw=0.4)
        axs[i, 0].set_title(f"R̂ {diag[n]['split_rhat']:.3f}  ESS {diag[n]['ess_bulk']:.0f}  τ_int {[round(v) for v in diag[n]['acf_int_time']]}", fontsize=6, loc="right")
    axs[0, 0].set_title("trace per chain (blue chain 0, orange chain 1)", fontsize=7, loc="left"); axs[0, 1].set_title("rank plot (uniform = mixed)", fontsize=7, loc="left"); axs[0, 2].set_title("autocorrelation", fontsize=7, loc="left")
    axs[-1, 0].set_xlabel("draw", fontsize=6); axs[-1, 2].set_xlabel("lag", fontsize=6)
    fig.suptitle(f"{a.label}: MCMC diagnostics bundle — {nch} chains × {nd} draws; all-site R̂>1.10: {sites['rhat_gt_1p10']}/574, ESS<100: {sites['ess_lt_100']}/574", fontsize=8, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"chain_diagnostics_{a.label}.png"), dpi=120); plt.close(fig)
    print(json.dumps({n: (round(v["split_rhat"], 3), round(v["ess_bulk"], 1), [round(t) for t in v["acf_int_time"]]) for n, v in diag.items()}, indent=0))
    print("all sites:", {k: sites[k] for k in ("rhat_gt_1p05", "rhat_gt_1p10", "rhat_max", "ess_min", "ess_lt_100")}, "worst", sites["worst_rhat"][:4])
    return 0


if __name__ == "__main__":
    sys.exit(main())
