"""wall1_is_faithfulness.py — WALL-1 full-injection IS-faithfulness check (reduce-only).

notes/2026-06-17_wall1_full_injection_design.md §3.4 A (the proxy check). Tests the
load-bearing claim of the reweighting WALL-1 gate: that reweighting the existing
untilted GP detections by w(logN_host)=10^(Δα·(N−20.3)) is a FAITHFUL importance-
sampling proxy for genuinely re-inferring a tilted-slope injected population.

For each arm (Δα = ±0.5):
  * RE-INFERENCE N̂-distribution  = histogram of the arm's GENUINELY re-inferred op
    detections' predicted NHI (the dlacat in <arm>/gp_out/), on op rows ≥ floor.
  * REWEIGHTING-PREDICTED N̂-dist = histogram of the UNTILTED loa-124 production op
    detections' predicted NHI, each weighted by w(NHI_TILT_HOST) (its truth host's
    column), hostless detection (no truth host) → weight 1.0 (forest FP, slope-blind)
    — EXACTLY the gate's detection_tilt_weights. Normalised to the same total.

Compare the two predicted-NHI distributions with a weighted two-sample KS and a
binned χ². PASS (proxy holds) = the shapes agree within MC error. FAIL (proxy
artifact) = the re-inference produces a detection N̂-distribution the reweighting
cannot mimic (a selection effect reweighting misses).

CAVEAT (substrate): the re-inference arm is injected into loa-0 (1 absorber per clean
sightline); the reweighting predicts off the loa-124 NATURAL population. The N̂-SHAPE
comparison is what tests the proxy — the absolute number density differs (injection
density ≠ natural density) and is irrelevant here (both are area-normalised). This is
the SHAPE test the design specifies; it is independent of the §3.4-B closure read.

DISCIPLINE: reduce-only, no inference, no SLURM. Reuses cddf_catalog_hbi loaders +
cddf_tilt_closure.tilt_weight unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, _build_qso_lookup,
)
from CDDF_analysis.hbi.cddf_tilt_closure import tilt_weight, LOGN_PIVOT


# defaults mirror wall1_full_injection.py
DEF_UNTILTED_CAT = ("/scratch/cavestru_root/cavestru0/mfho/"
                    "gl_prod_2lpt0_v1_20260526/combined_catalog/")
DEF_UNTILTED_TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                      "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
DEF_UNTILTED_BAL = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                    "qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits")
DEF_MOLLY = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
             "figures_molly_nhi195/lya_only/molly_matrix.tsv")
DEF_LOA0 = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
            "qq_desi_y3/v2.8.5/mock-0/loa-0")


def _cfg(catalog_dir, truth_path, bal_cat_path, molly_tsv, mockdir, args):
    return HBIConfig(
        catalog_dir=catalog_dir, truth_path=truth_path, bal_cat_path=bal_cat_path,
        molly_tsv=molly_tsv, out_dir="/tmp/wall1_is", mockdir=mockdir,
        zbins=(2.0, 2.5, 3.0, 3.5), report_logN_limits=(20.0, 20.3, 20.6),
        fp_estimator="purity_mixture", no_bal=True,
        v3_family="bspbody", v3_logN_fit_floor=args.fit_floor,
        lam_rf_min=1025.0, rng_seed=0,
    )


def _op_NHI_weights(cfg, cat_cut, good_mask, dalpha, floor):
    """op detections' predicted NHI + per-row tilt weight (truth-host driven)."""
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    nhi = np.asarray(cat_cut["NHI"], float)
    op = op & np.isfinite(nhi) & (nhi >= floor)
    nhi_hat = nhi[op]
    if dalpha is None:
        w = np.ones_like(nhi_hat)
    else:
        host = np.asarray(cat_cut["NHI_TILT_HOST"], float)[op]
        w = tilt_weight(host, dalpha, LOGN_PIVOT)  # hostless (NaN) -> 1.0
    return nhi_hat, w


def _weighted_ecdf(x, w, grid):
    """weighted ECDF of x evaluated on `grid` (sorted)."""
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cum = np.cumsum(ws) / ws.sum()
    # value of the step function at each grid point
    idx = np.searchsorted(xs, grid, side="right") - 1
    out = np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)
    return out


def weighted_ks(x1, w1, x2, w2):
    """Two-sample weighted KS statistic D + asymptotic p-value (effective n)."""
    grid = np.unique(np.concatenate([x1, x2]))
    F1 = _weighted_ecdf(x1, w1, grid)
    F2 = _weighted_ecdf(x2, w2, grid)
    D = float(np.max(np.abs(F1 - F2)))
    # effective sample sizes (Kish) for weighted samples
    n1 = (w1.sum() ** 2) / np.sum(w1 ** 2)
    n2 = (w2.sum() ** 2) / np.sum(w2 ** 2)
    en = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(en) + 0.12 + 0.11 / np.sqrt(en)) * D
    # Kolmogorov asymptotic Q(lam)
    j = np.arange(1, 101)
    p = 2.0 * np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * (j ** 2) * lam ** 2))
    p = float(min(max(p, 0.0), 1.0))
    return D, p, float(n1), float(n2)


def binned_chi2(x1, w1, x2, w2, edges):
    """Area-normalised binned χ² between the two weighted N̂ histograms.

    Each histogram is normalised to unit area; χ² compares the two normalised
    densities with a combined Poisson-like per-bin variance (Kish-effective counts).
    Returns chi2, dof, p, and the two normalised densities.
    """
    from scipy.stats import chi2 as chi2dist
    h1, _ = np.histogram(x1, bins=edges, weights=w1)
    h2, _ = np.histogram(x2, bins=edges, weights=w2)
    # Kish-effective per-bin counts (for the variance) and area-normalised densities
    s1 = w1.sum(); s2 = w2.sum()
    widths = np.diff(edges)
    d1 = h1 / s1 / widths
    d2 = h2 / s2 / widths
    # effective per-bin n: sum(w)^2/sum(w^2) per bin
    e1 = np.histogram(x1, bins=edges, weights=w1)[0] ** 2 / np.clip(
        np.histogram(x1, bins=edges, weights=w1 ** 2)[0], 1e-30, None)
    e2 = np.histogram(x2, bins=edges, weights=w2)[0] ** 2 / np.clip(
        np.histogram(x2, bins=edges, weights=w2 ** 2)[0], 1e-30, None)
    # variance of each density estimate (multinomial fraction / area)
    p1 = h1 / s1; p2 = h2 / s2
    var1 = np.where(e1 > 0, p1 * (1 - p1) / np.clip(e1, 1, None), np.inf) / widths ** 2
    var2 = np.where(e2 > 0, p2 * (1 - p2) / np.clip(e2, 1, None), np.inf) / widths ** 2
    use = (h1 > 0) | (h2 > 0)
    denom = var1 + var2
    terms = np.where(use & (denom > 0), (d1 - d2) ** 2 / denom, 0.0)
    chi2 = float(np.sum(terms))
    dof = int(np.sum(use & (denom > 0)) - 1)
    p = float(chi2dist.sf(chi2, max(dof, 1)))
    return chi2, dof, p, d1, d2


def run(args):
    mm = load_molly_matrix(args.molly)
    floor = max(float(mm.nhi_edges[0]), args.fit_floor)

    # untilted loa-124 production op detections (the reweighting substrate)
    cfg_u = _cfg(args.untilted_cat, args.untilted_truth, args.untilted_bal,
                 args.molly, os.path.dirname(args.untilted_truth), args)
    ql_u = _build_qso_lookup(cfg_u)
    cat_u, _t, _tp, gm_u, _m = load_and_cut_catalog(
        cfg_u, truth_nhi_floor=float(mm.nhi_edges[0]), qso_lookup=ql_u,
        host_truth_floor=min(args.host_truth_floor, float(mm.nhi_edges[0])))

    edges = np.arange(floor, args.nhi_max + 1e-9, args.nhi_step)
    results = {}
    for dalpha, label in ((args.dalpha, f"dalpha{args.dalpha:+g}"),
                          (-args.dalpha, f"dalpha{-args.dalpha:+g}")):
        arm = (args.arm_plus if dalpha > 0 else args.arm_minus)
        # re-inference N̂ (genuine, injected arm)
        cfg_a = _cfg(os.path.join(arm, "gp_out"),
                     os.path.join(arm, "injected_truth_cat.fits"),
                     args.untilted_bal, args.molly, args.loa0, args)
        ql_a = _build_qso_lookup(cfg_a)
        cat_a, _t2, _tp2, gm_a, _m2 = load_and_cut_catalog(
            cfg_a, truth_nhi_floor=float(mm.nhi_edges[0]), qso_lookup=ql_a,
            host_truth_floor=min(args.host_truth_floor, float(mm.nhi_edges[0])))
        nhat_re, w_re = _op_NHI_weights(cfg_a, cat_a, gm_a, None, floor)

        # reweighting-predicted N̂ (untilted loa-124 reweighted by w(host) at THIS Δα)
        nhat_rw, w_rw = _op_NHI_weights(cfg_u, cat_u, gm_u, dalpha, floor)

        D, p_ks, n1, n2 = weighted_ks(nhat_re, w_re, nhat_rw, w_rw)
        chi2, dof, p_chi2, d_re, d_rw = binned_chi2(
            nhat_re, w_re, nhat_rw, w_rw, edges)
        verdict = "FAITHFUL" if (p_ks > 0.05 and p_chi2 > 0.05) else "NOT-FAITHFUL"
        results[label] = dict(
            dalpha=dalpha, n_re=int(len(nhat_re)), n_rw=int(len(nhat_rw)),
            ks_D=D, ks_p=p_ks, ks_neff_re=n1, ks_neff_rw=n2,
            chi2=chi2, dof=dof, chi2_p=p_chi2, verdict=verdict,
            edges=edges, d_re=d_re, d_rw=d_rw,
            mean_re=float(np.average(nhat_re, weights=w_re)),
            mean_rw=float(np.average(nhat_rw, weights=w_rw)))
        print(f"\n[IS-faithfulness] {label}  (floor {floor})")
        print(f"  re-inference op N̂: n={len(nhat_re)}  mean N̂={results[label]['mean_re']:.4f}")
        print(f"  reweight-pred  N̂: n_eff={n2:.0f} (from {len(nhat_rw)} loa-124 op)  "
              f"mean N̂={results[label]['mean_rw']:.4f}")
        print(f"  weighted-KS  D={D:.4f}  p={p_ks:.4g}")
        print(f"  binned-χ²    χ²={chi2:.2f}  dof={dof}  p={p_chi2:.4g}")
        print(f"  → {verdict} (proxy {'holds' if verdict=='FAITHFUL' else 'is an artifact'})")

    # TSV
    os.makedirs(args.out, exist_ok=True)
    out_tsv = os.path.join(args.out, "wall1_is_faithfulness.tsv")
    with open(out_tsv, "w") as fh:
        fh.write("arm\tn_re\tn_rw\tmean_re\tmean_rw\tks_D\tks_p\tchi2\tdof\tchi2_p\tverdict\n")
        for label, r in results.items():
            fh.write(f"{label}\t{r['n_re']}\t{r['n_rw']}\t{r['mean_re']:.5f}\t"
                     f"{r['mean_rw']:.5f}\t{r['ks_D']:.5f}\t{r['ks_p']:.5g}\t"
                     f"{r['chi2']:.4f}\t{r['dof']}\t{r['chi2_p']:.5g}\t{r['verdict']}\n")
    # per-bin density TSV (for an overlay figure)
    out_dens = os.path.join(args.out, "wall1_is_faithfulness_density.tsv")
    with open(out_dens, "w") as fh:
        fh.write("arm\tlogN_lo\tlogN_hi\td_reinference\td_reweight\n")
        for label, r in results.items():
            e = r["edges"]
            for i in range(len(e) - 1):
                fh.write(f"{label}\t{e[i]:.3f}\t{e[i+1]:.3f}\t"
                         f"{r['d_re'][i]:.6g}\t{r['d_rw'][i]:.6g}\n")
    print(f"\n[IS-faithfulness] saved -> {out_tsv}")
    print(f"[IS-faithfulness] saved -> {out_dens}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-plus", required=True, help="Δα=+ arm tree root")
    p.add_argument("--arm-minus", required=True, help="Δα=− arm tree root")
    p.add_argument("--dalpha", type=float, default=0.5)
    p.add_argument("--out", default="/tmp/wall1_full_injection")
    p.add_argument("--untilted-cat", default=DEF_UNTILTED_CAT)
    p.add_argument("--untilted-truth", default=DEF_UNTILTED_TRUTH)
    p.add_argument("--untilted-bal", default=DEF_UNTILTED_BAL)
    p.add_argument("--molly", default=DEF_MOLLY)
    p.add_argument("--loa0", default=DEF_LOA0)
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    p.add_argument("--nhi-max", type=float, default=22.5)
    p.add_argument("--nhi-step", type=float, default=0.2)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
