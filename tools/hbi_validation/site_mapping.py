#!/usr/bin/env python3
"""site_mapping.py — the machine-readable map of every sampled scalar of the production model
(model_cc, informative_ln) to its physical cell, atlas block and correlation index
(2026-09-02 HBI identifiability campaign, kickoff §24 / request §21.4). Built from the PACK
only (grids), so it carries no posterior value and is safe in the public tree.

    python tools/hbi_validation/site_mapping.py --pack PACK.npz --out mapping.csv [--json mapping.json]

Row schema: site, flat_index, array_index, scalar_id, physical_label, cell_edges, atlas_block,
atlas_page, corr_index.  The corr_index order is the site order of the numpyro trace
(sigma_N, sigma_z, theta_level, theta_slope, eps_N, eps_z, psi_c, fp_lam_total, fp_shape_v, t),
each site flattened C-order — the same order `flatten_draws` below produces from an all-sites npz.
"""
import argparse, csv, json, sys, os
import numpy as np

SITE_ORDER = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z",
              "psi_c", "fp_lam_total", "fp_shape_v", "t")


def _fmt(x):
    return "inf" if np.isinf(x) else f"{x:g}"


def build_mapping(pack):
    ntrue = np.asarray(pack["ntrue_edges"], float)
    nhat = np.asarray(pack["nhat_edges"], float)
    zf = np.asarray(pack["zf_edges"], float)
    zc = np.asarray(pack["zc_edges"], float)
    snr = np.asarray(pack["snr_edges"], float)
    molly = np.asarray(pack["molly_nhi_edges"], float)
    B, Kf, C, S, M, KK = len(ntrue) - 1, len(zf) - 1, len(nhat) - 1, len(snr) - 1, len(molly) - 1, len(zc) - 1
    rows = []
    idx = 0

    def add(site, flat, arr, sid, label, edges, block, page):
        nonlocal idx
        rows.append(dict(site=site, flat_index=flat, array_index=json.dumps(arr), scalar_id=sid,
                         physical_label=label, cell_edges=json.dumps(edges), atlas_block=block,
                         atlas_page=page, corr_index=idx))
        idx += 1

    add("sigma_N", 0, [], "sigma_N", "sigma_N: N-curvature innovation scale (HalfNormal 0.5)", None, "A", 1)
    add("sigma_z", 0, [], "sigma_z", "sigma_z: z random-walk innovation scale (HalfNormal 0.5)", None, "A", 1)
    add("theta_level", 0, [], "theta_level", "theta_level: log f level at the N-grid centre (Normal 0,4)", None, "A", 1)
    add("theta_slope", 0, [], "theta_slope", "theta_slope: log f slope per latent N bin (Normal 0,2)", None, "A", 1)
    # eps_N[j] drives the curvature entering theta at bin j+2 (double cumsum with two leading zeros)
    for j in range(B - 2):
        b = j + 2
        add("eps_N", j, [j], f"eps_N[{j}]",
            f"eps_N[{j}]: curvature innovation entering latent N bin b={b} [{ntrue[b]:g},{ntrue[b+1]:g})",
            [ntrue[b], ntrue[b + 1]], "B", 1)
    # eps_z[b,k] : z increment from fine bin k to k+1 for latent bin b (theta[b, k+1] = theta[b, k] + sigma_z eps_z[b,k])
    for b in range(B):
        for k in range(Kf - 1):
            add("eps_z", b * (Kf - 1) + k, [b, k], f"eps_z[b={b},k={k}]",
                f"eps_z: latent N bin [{ntrue[b]:g},{ntrue[b+1]:g}); z step {zf[k]:g}->{zf[k+1]:g} to [{zf[k+1]:g},{zf[k+2]:g})",
                [ntrue[b], ntrue[b + 1], zf[k + 1], zf[k + 2]], "C", b + 1)
    # psi_c[s, m] : completeness logit offset in S/N stratum s, molly N cell m
    for s in range(S):
        for m in range(M):
            add("psi_c", s * M + m, [s, m], f"psi_c[s={s},m={m}]",
                f"psi_c: S/N [{_fmt(snr[s])},{_fmt(snr[s+1])}); true-N_HI molly cell [{_fmt(molly[m])},{_fmt(molly[m+1])})",
                [snr[s], snr[s + 1], molly[m], molly[m + 1]], "D", s + 1)
    add("fp_lam_total", 0, [], "fp_lam_total", "Lambda: FP intensity per unit loa-0 exposure (Gamma prior)", None, "F", 1)
    # fp_shape_v flat index = c*S + s  (lam_fp = (lam_total*pi).reshape(C, S))
    for c in range(C):
        for s in range(S):
            add("fp_shape_v", c * S + s, [c, s], f"fp_shape_v[c={c},s={s}]",
                f"fp_shape_v: reported-N_hat cell [{nhat[c]:g},{nhat[c+1]:g}); S/N [{_fmt(snr[s])},{_fmt(snr[s+1])})",
                [nhat[c], nhat[c + 1], snr[s], snr[s + 1]], "G", s + 1)
    for K in range(KK):
        add("t", K, [K], f"t[{K}]", f"t_K: FP transfer, coarse z [{zc[K]:g},{zc[K+1]:g})", [zc[K], zc[K + 1]], "F", 1)
    expected = 4 + (B - 2) + B * (Kf - 1) + S * M + 1 + C * S + KK
    assert len(rows) == expected, (len(rows), expected)
    return rows


def flatten_draws(allsites, site_order=SITE_ORDER):
    """all-sites npz (chains, draws, ...) per site -> (chains*draws, N) in mapping order + chain id."""
    cols, names = [], []
    nch, nd = None, None
    for k in site_order:
        a = np.asarray(allsites[k], float)
        if nch is None:
            nch, nd = a.shape[0], a.shape[1]
        cols.append(a.reshape(nch * nd, -1))
    X = np.concatenate(cols, axis=1)
    chain = np.repeat(np.arange(nch), nd)
    return X, chain


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--out", required=True); ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rows = build_mapping(np.load(a.pack))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=0)
    print(f"{len(rows)} sampled scalars mapped -> {a.out}")
    from collections import Counter
    print(Counter(r["atlas_block"] for r in rows), Counter((r["atlas_block"], r["atlas_page"]) for r in rows).__len__(), "pages")


if __name__ == "__main__":
    sys.exit(main())
