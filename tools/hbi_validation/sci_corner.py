#!/usr/bin/env python3
"""sci_corner.py — the SCIENTIFIC DEGENERACY CORNER of a validation run (kickoff §26 / request §22):
log Λ, t_K, A_K = log Λ + t_K, C̄_subDLA, d̄_subDLA, the top-leverage psi_c cells, the latent
upper-sub-DLA amplitude, dN/dX(≥20.0), dN/dX(≥20.3), Ω_HI[20.3,21.6] — coloured by chain, with
per-arm small multiples of the key pairs, optional overlay of an extra (e.g. mirror) chain, constant-A_K
reference lines in the (log Λ, t_K) panels, the curated correlation matrix and the eigenmode audit.
PRIVATE outputs.

    python tools/hbi_validation/sci_corner.py --pack PACK --geometry geometry_RUN.json --pooled POOLED.json|--arms ... \
        --out-dir DIR --run-id R0 [--extra-arm path:label] [--axes-from sci_axes_R0.json]
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.pack import load_pack                               # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors    # noqa: E402
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior                  # noqa: E402
from tools.hbi_validation.geometry import omega_prefactor, SUBDLA              # noqa: E402
from tools.hbi_validation.atlas import load_arms                                # noqa: E402
from tools.hbi_validation.viz_common import corner, heatmap, plt, SLOTS, INK2  # noqa: E402


def sci_coords(pk, consts, z_list, lev_cells):
    """Return (Y, names) for a list of all-sites dicts stacked."""
    ntrue = np.asarray(pk.ntrue_edges, float); Nc = 0.5 * (ntrue[:-1] + ntrue[1:]); dN = np.diff(ntrue); dXk = np.asarray(pk.dX, float).sum(axis=1)
    kz = np.asarray(consts.kz_to_K); eta = np.asarray(consts.eta_hat); b2c = np.asarray(consts.b_to_cell); g = np.asarray(consts.g_bk); dX = np.asarray(consts.dX)
    mS = (Nc >= SUBDLA[0] - 1e-9) & (Nc < SUBDLA[1] - 1e-9); mO = (Nc >= 20.3 - 1e-9) & (Nc < 21.6 - 1e-9)
    F = np.concatenate([np.asarray(z["f"]).reshape(-1, *z["f"].shape[2:]) for z in z_list]); PS = np.concatenate([np.asarray(z["psi_c"]).reshape(-1, *z["psi_c"].shape[2:]) for z in z_list])
    LF = np.concatenate([np.asarray(z["lam_fp"]).reshape(-1, *z["lam_fp"].shape[2:]) for z in z_list]); T = np.concatenate([np.asarray(z["t"]).reshape(-1, z["t"].shape[-1]) for z in z_list])
    fbar = F.mean(axis=0); w_bs = np.einsum("bk,bk,ks->bs", g[mS] * dN[mS][:, None], fbar[mS], dX)
    Cc = 1 / (1 + np.exp(-(eta[None] + PS))); Cb = np.einsum("nsb,bs->n", Cc[:, :, b2c[mS]], w_bs) / w_bs.sum(); db = np.einsum("nsb,bs->n", PS[:, :, b2c[mS]], w_bs) / w_bs.sum()
    logL = np.log(LF.sum(axis=(1, 2))); A = logL[:, None] + T
    red = reduce_f_posterior(F, pk)
    f_sub = (np.einsum("dbk,b->dk", F[:, mS, :], dN[mS]) * dXk[None]).sum(axis=1) / dXk.sum()
    om = (np.einsum("dbk,b->dk", F[:, mO, :], 10**Nc[mO] * dN[mO]) * omega_prefactor() * dXk[None]).sum(axis=1) / dXk.sum()
    cols = [logL] + [T[:, K] for K in range(T.shape[1])] + [A[:, K] for K in range(T.shape[1])] + [Cb, db]
    names = ["log Λ"] + [f"t{K}" for K in range(T.shape[1])] + [f"A{K}=logΛ+t{K}" for K in range(T.shape[1])] + ["C̄ subDLA", "d̄ subDLA"]
    for cell in lev_cells:
        s, m = [int(v) for v in re.findall(r"\d+", cell)]
        cols.append(PS[:, s, m]); names.append(f"ψc[s{s},m{m}]")
    cols += [f_sub, np.asarray(red["dndx_dla_20p0_allz"]), np.asarray(red["dndx_dla_20p3_allz"]), om * 1e3]
    names += ["f subDLA 19.7–20.3", "dN/dX ≥20.0", "dN/dX ≥20.3", "10³ Ω[20.3,21.6]"]
    return np.column_stack(cols), names, dict(fbar=fbar)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--geometry", required=True)
    ap.add_argument("--pooled", default=None); ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--run-id", default="R0")
    ap.add_argument("--extra-arm", action="append", default=[]); ap.add_argument("--n-lev", type=int, default=3)
    ap.add_argument("--axes-from", default=None, help="sci_axes JSON of a reference run: reuse its axis limits (cross-run comparability)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pk = load_pack(a.pack); consts, _ = build_cc_tensors(pk)
    G = json.load(open(a.geometry))
    lev_cells = [e["cell"] for e in G["leverage_psi_c"]["z0"][:a.n_lev]]
    _, chain, arm, tags, paths = load_arms(a.pooled, a.arms)
    z_list = [np.load(p) for p in paths]
    Y, names, _ = sci_coords(pk, consts, z_list, lev_cells)
    KK = int(np.asarray(consts.t_sigma).size)
    # reference lines of constant A_K in the (log Λ, t_K) panels: t = A - logΛ  (slope -1) at the posterior median A_K
    lines = {}
    for K in range(KK):
        i_t = 1 + K; j_L = 0; A_med = float(np.median(Y[:, 1 + KK + K]))
        lines[(i_t, j_L)] = [(-1.0, A_med + d) for d in (-0.3, 0.0, 0.3)]
    extras = []
    for spec in a.extra_arm:          # path:label[:chain]  — a single chain of an arm can be selected
        parts = spec.split(":"); p, lab = parts[0], parts[1]; ch_sel = int(parts[2]) if len(parts) > 2 else None
        z = dict(np.load(p))
        if ch_sel is not None:
            z = {k: (v[ch_sel:ch_sel + 1] if np.ndim(v) >= 2 and v.shape[0] == z["t"].shape[0] else v) for k, v in z.items()}
        Ye, _, _ = sci_coords(pk, consts, [z], lev_cells); extras.append((lab, Ye, z["t"].shape[0]))
    # main corner: chain colouring, per-arm contours (arms ≤ 6 → slot colours; the legend names them)
    groups = arm if len(tags) <= 6 else None
    fig = corner(Y, names, chain=chain, groups=groups, group_labels=(tags if groups is not None else None),
                 title=f"{a.run_id} scientific degeneracy corner — diagonal by chain (blue = chain 0, orange = chain 1); contours per arm; dashed = constant A_K", bins=32, panel=0.55, lines=lines)
    fig.savefig(os.path.join(a.out_dir, f"sci_corner_{a.run_id}.png"), dpi=120); fig.savefig(os.path.join(a.out_dir, f"sci_corner_{a.run_id}.pdf")); plt.close(fig)
    # key pairs, per-arm small multiples, with extra arms overlaid in slot 3+
    pairs = [("log Λ", f"t{K}") for K in range(KK)] + [(f"A{K}=logΛ+t{K}", f"t{K}") for K in range(KK)] + [(f"t0", "C̄ subDLA"), ("t0", "d̄ subDLA"), ("t0", "f subDLA 19.7–20.3"), ("log Λ", "C̄ subDLA"), ("t0", "dN/dX ≥20.3"), ("C̄ subDLA", "f subDLA 19.7–20.3")]
    ni = {n: i for i, n in enumerate(names)}
    axes_lim = {}
    if a.axes_from:
        axes_lim = json.load(open(a.axes_from))
    lim_out = {}
    for xn, yn in pairs:
        i, j = ni[xn], ni[yn]
        fig, axs = plt.subplots(1, len(tags), figsize=(2.2 * len(tags) + 0.5, 2.4), sharex=True, sharey=True)
        axs = np.atleast_1d(axs)
        xl = axes_lim.get(xn) or [float(np.percentile(Y[:, i], 0.1)), float(np.percentile(Y[:, i], 99.9))]
        yl = axes_lim.get(yn) or [float(np.percentile(Y[:, j], 0.1)), float(np.percentile(Y[:, j], 99.9))]
        for lab, Ye, _ in extras:
            xl = [min(xl[0], float(np.percentile(Ye[:, i], 1))), max(xl[1], float(np.percentile(Ye[:, i], 99)))]
            yl = [min(yl[0], float(np.percentile(Ye[:, j], 1))), max(yl[1], float(np.percentile(Ye[:, j], 99)))]
        lim_out[xn] = xl; lim_out[yn] = yl
        for k, ax in enumerate(axs):
            m = arm == k
            for c in (0, 1):
                mm = m & (chain == c)
                ax.scatter(Y[mm, i], Y[mm, j], s=2, alpha=0.35, color=SLOTS[c], linewidths=0, rasterized=True)
            for e_i, (lab, Ye, nch) in enumerate(extras):
                ax.scatter(Ye[:, i], Ye[:, j], s=2, alpha=0.35, color=SLOTS[2 + e_i], linewidths=0, rasterized=True, label=lab if k == 0 else None)
            if xn == "log Λ" and yn.startswith("t"):
                K = int(yn[1:]); A_med = float(np.median(Y[:, 1 + KK + K])); xs = np.array(xl)
                for d in (-0.3, 0.0, 0.3):
                    ax.plot(xs, -xs + A_med + d, color=INK2, lw=0.5, ls="--")
            ax.set_title(tags[k], fontsize=7); ax.set_xlim(xl); ax.set_ylim(yl); ax.tick_params(labelsize=6)
            if k == 0:
                ax.set_ylabel(yn, fontsize=7)
            ax.set_xlabel(xn, fontsize=7)
        if extras:
            axs[0].legend(fontsize=6, loc="best")
        fig.suptitle(f"{a.run_id}: {yn} vs {xn} per arm (blue chain 0, orange chain 1" + (", aqua/yellow = extra chains)" if extras else ")"), fontsize=8, x=0.02, ha="left")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{yn}_vs_{xn}")
        fig.savefig(os.path.join(a.out_dir, f"pair_{a.run_id}_{safe}.png"), dpi=130); plt.close(fig)
    json.dump(lim_out, open(os.path.join(a.out_dir, f"sci_axes_{a.run_id}.json"), "w"), indent=1)
    # curated correlation matrix + eigenmode audit
    keep = Y.std(axis=0, ddof=1) > 0                      # fixed sites (R1/R3/R4) give constant columns
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Y.T)
    C = np.nan_to_num(C, nan=0.0)
    fig = heatmap(C, names, names, f"{a.run_id}: correlation among the scientific quantities (constant = fixed site -> 0)", figsize=(7, 6)); fig.savefig(os.path.join(a.out_dir, f"sci_corr_{a.run_id}.png"), dpi=150); plt.close(fig)
    Yk = Y[:, keep]; nk = [n for n, k in zip(names, keep) if k]
    Ys = (Yk - Yk.mean(axis=0)) / Yk.std(axis=0, ddof=1); w, v = np.linalg.eigh(np.corrcoef(Ys.T)); order = np.argsort(-w)
    eig = [dict(rank=r + 1, eigval=float(w[k]), var_share=float(w[k] / w.sum()), loadings={n: round(float(v[i, k]), 3) for i, n in enumerate(nk)}) for r, k in enumerate(order[:4])]
    summ = dict(run_id=a.run_id, names=names, n=int(Y.shape[0]), arms=tags, corr=C.round(4).tolist(), eigenmodes_top4=eig, constant_columns=[n for n, k in zip(names, keep) if not k],
                sd={n: float(Y[:, i].std(ddof=1)) for i, n in enumerate(names)}, median={n: float(np.median(Y[:, i])) for i, n in enumerate(names)},
                extra_arms={lab: dict(median={n: float(np.median(Ye[:, i])) for i, n in enumerate(names)}) for lab, Ye, _ in extras})
    json.dump(summ, open(os.path.join(a.out_dir, f"sci_summary_{a.run_id}.json"), "w"), indent=1)
    print("eigenmodes:", [(e["rank"], round(e["var_share"], 3), sorted(e["loadings"].items(), key=lambda kv: -abs(kv[1]))[:4]) for e in eig])
    return 0


if __name__ == "__main__":
    sys.exit(main())
