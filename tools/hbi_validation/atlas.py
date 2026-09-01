#!/usr/bin/env python3
"""atlas.py — the COMPLETE paginated corner atlas of every sampled scalar (574) of the production
model, plus the full N×N posterior correlation matrix and block heatmaps (2026-09-02 HBI
identifiability campaign, kickoff §23–25 / request §21, §23). PRIVATE outputs (real-data draws).

    python tools/hbi_validation/atlas.py --pack PACK --pooled POOLED.json|--arms *.npz --mapping site_mapping_574.csv --out-dir DIR
"""
import argparse, csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.hbi_validation.site_mapping import flatten_draws, SITE_ORDER   # noqa: E402
from tools.hbi_validation.viz_common import corner, heatmap, short_label, plt, SLOTS  # noqa: E402
from CDDF_analysis.hbi_mcmc.provenance_util import sha256                   # noqa: E402


def load_arms(pooled, arms):
    if pooled:
        P = json.load(open(pooled))
        paths = [r["file"][:-5] + "_allsites.npz" for r in P["selection"]["included"]]
        tags = [f"s{r['seed']}{'d' if r['deep'] else ''}" for r in P["selection"]["included"]]
    else:
        paths = list(arms); tags = [re.sub(r".*REAL_ln_(deep_)?s(\d{8})_allsites\.npz", lambda m: f"s{m.group(2)}{'d' if m.group(1) else ''}", p) for p in paths]
    Xs, ch, ar = [], [], []
    for i, p in enumerate(paths):
        z = np.load(p); X, c = flatten_draws(z); Xs.append(X); ch.append(c); ar.append(np.full(X.shape[0], i))
    return np.concatenate(Xs), np.concatenate(ch), np.concatenate(ar), tags, paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--pooled", default=None); ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--mapping", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="R0"); ap.add_argument("--skip-pages", action="store_true"); ap.add_argument("--max-draws", type=int, default=12000)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out_dir, "atlas"), exist_ok=True)
    rows = list(csv.DictReader(open(a.mapping)))
    X, chain, arm, tags, paths = load_arms(a.pooled, a.arms)
    if X.shape[0] > a.max_draws:
        sel = np.sort(np.random.default_rng(0).choice(X.shape[0], a.max_draws, replace=False)); X, chain, arm = X[sel], chain[sel], arm[sel]
    assert X.shape[1] == len(rows) == 574, (X.shape, len(rows))
    names = [r["scalar_id"] for r in rows]
    # ---- full correlation matrix (machine-readable) + summary
    C = np.corrcoef(X.T)
    np.savez(os.path.join(a.out_dir, f"corr_full_{a.run_id}.npz"), corr=C, corr_index=np.arange(574), scalar_id=np.array(names), sd=X.std(axis=0, ddof=1), mean=X.mean(axis=0))
    with open(os.path.join(a.out_dir, f"corr_full_{a.run_id}.csv"), "w") as fh:
        fh.write("corr_index," + ",".join(names) + "\n")
        for i in range(574):
            fh.write(f"{i}," + ",".join(f"{v:.4f}" for v in C[i]) + "\n")
    iu = np.triu_indices(574, 1); absC = np.abs(C[iu])
    top = np.argsort(-absC)[:40]
    summary = dict(run_id=a.run_id, n_draws=int(X.shape[0]), arms=tags, arm_files={t: dict(path=p, sha256=sha256(p)) for t, p in zip(tags, paths)},
                   n_pairs_abs_corr_gt_0p5=int((absC > 0.5).sum()), n_pairs_abs_corr_gt_0p8=int((absC > 0.8).sum()),
                   top_abs_correlations=[dict(a=names[iu[0][k]], b=names[iu[1][k]], corr=float(C[iu[0][k], iu[1][k]])) for k in top])
    # eigen-spectrum of the full correlation
    w = np.linalg.eigvalsh(C)[::-1]; summary["corr_eigvals_top10"] = w[:10].round(3).tolist(); summary["corr_eigvals_min"] = float(w[-1])
    # ---- block heatmaps
    blocks = {}
    for r in rows:
        blocks.setdefault(r["atlas_block"], []).append(int(r["corr_index"]))
    lbl = [short_label(r).replace("\n", " ") for r in rows]
    for b, idx in blocks.items():
        if len(idx) > 120:   # C (224) and G (232): one heatmap per page group
            pages = {}
            for r in rows:
                if r["atlas_block"] == b:
                    pages.setdefault(int(r["atlas_page"]), []).append(int(r["corr_index"]))
            for pg, ii in pages.items():
                fig = heatmap(C[np.ix_(ii, ii)], [lbl[i] for i in ii], [lbl[i] for i in ii], f"{a.run_id} block {b} page {pg}: posterior correlation")
                fig.savefig(os.path.join(a.out_dir, "atlas", f"corr_block_{b}_p{pg:02d}.png"), dpi=130); plt.close(fig)
        else:
            fig = heatmap(C[np.ix_(idx, idx)], [lbl[i] for i in idx], [lbl[i] for i in idx], f"{a.run_id} block {b}: posterior correlation", tick_every=(1 if len(idx) <= 40 else 2))
            fig.savefig(os.path.join(a.out_dir, "atlas", f"corr_block_{b}.png"), dpi=130); plt.close(fig)
    # cross-block: F (Λ, t) against everything, as a 4×574 strip, and D (psi_c) vs F
    iF = blocks["F"]
    fig = heatmap(C[np.ix_(iF, range(574))], [str(i) for i in range(574)], [lbl[i] for i in iF], f"{a.run_id} block F (Λ, t) vs all 574 (columns = corr_index)", figsize=(14, 2.2), tick_every=20)
    fig.savefig(os.path.join(a.out_dir, "atlas", "corr_F_vs_all.png"), dpi=130); plt.close(fig)
    # block-level mean |corr| matrix
    bl = sorted(blocks); Mb = np.zeros((len(bl), len(bl)))
    for i, bi in enumerate(bl):
        for j, bj in enumerate(bl):
            sub = np.abs(C[np.ix_(blocks[bi], blocks[bj])]); 
            if bi == bj:
                n_ = len(blocks[bi]); sub = sub[np.triu_indices(n_, 1)] if n_ > 1 else np.array([0.0])
            Mb[i, j] = sub.mean()
    summary["block_mean_abs_corr"] = dict(blocks=bl, matrix=Mb.round(4).tolist())
    fig = heatmap(Mb, bl, bl, f"{a.run_id}: mean |corr| between atlas blocks", vmax=float(Mb.max()) if Mb.max() > 0 else 1, figsize=(4, 3.5))
    fig.savefig(os.path.join(a.out_dir, "atlas", "corr_block_means.png"), dpi=150); plt.close(fig)
    # ---- block E statement page
    fig = plt.figure(figsize=(8.5, 4)); fig.text(0.05, 0.9, "ATLAS BLOCK E — response nuisance: NO SAMPLED SITE EXISTS IN PRODUCTION", fontsize=11, weight="bold", va="top")
    fig.text(0.05, 0.78, ("The production posterior was sampled with model_cc (CDDF_analysis/hbi_mcmc/cc_posterior_validation.py at\n"
                          "prov/paper1-freeze-2026-08-26 = 1fd4828), run by cc_real_posterior.py with --fp-mode informative_ln.\n"
                          "Its response kernel enters as the precomputed constant tensor Mg (build_cc_tensors: adopted_resp_{mu,sig,skew}_coef,\n"
                          "count-conserving masses normalised to adopted_phi_ref) and is passed to the model as data. The site psi_k_delta\n"
                          "(model_a.py, the response-intercept perturbation) is NOT part of this model: a numpyro trace of the frozen model on the\n"
                          "frozen pack (2026-09-01) contains the ten sampled sites sigma_N, sigma_z, theta_level, theta_slope, eps_N, eps_z,\n"
                          "psi_c, fp_lam_total, fp_shape_v, t (574 scalars) and no response site. The kernel's fit uncertainty is carried\n"
                          "post hoc by the bootstrap carrier (not a likelihood parameter). Nothing is plotted here because nothing was sampled."),
             fontsize=8, va="top", family="monospace")
    fig.savefig(os.path.join(a.out_dir, "atlas", "block_E_absent.pdf")); plt.close(fig)
    json.dump(summary, open(os.path.join(a.out_dir, f"corr_summary_{a.run_id}.json"), "w"), indent=1)
    print("corr summary:", {k: summary[k] for k in ("n_draws", "n_pairs_abs_corr_gt_0p5", "n_pairs_abs_corr_gt_0p8", "corr_eigvals_top10")})
    if a.skip_pages:
        return 0
    # ---- corner pages
    pages = {}
    for r in rows:
        pages.setdefault((r["atlas_block"], int(r["atlas_page"])), []).append(r)
    index = []
    for (b, pg), rr in sorted(pages.items()):
        ii = [int(r["corr_index"]) for r in rr]
        labels = [short_label(r) for r in rr]
        fig = corner(X[:, ii], labels, chain=chain, groups=None, title=f"{a.run_id} atlas block {b} page {pg} — {len(ii)} sampled scalars; diagonal: chain 0 (blue) / chain 1 (orange); {X.shape[0]} draws, {len(tags)} arms",
                     bins=28, panel=0.36, contour=False)
        fn = f"atlas_{b}_p{pg:02d}.png"; fig.savefig(os.path.join(a.out_dir, "atlas", fn), dpi=110); plt.close(fig)
        index.append(dict(block=b, page=pg, file=fn, scalars=[r["scalar_id"] for r in rr]))
        print("page", b, pg, len(ii))
    json.dump(index, open(os.path.join(a.out_dir, "atlas", "ATLAS_INDEX.json"), "w"), indent=1)
    covered = sorted(set(s for e in index for s in e["scalars"])); assert len(covered) == 574
    print("atlas complete: 574/574 scalars on", len(index), "pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
