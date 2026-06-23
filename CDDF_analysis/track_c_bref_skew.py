"""track_c_bref_skew.py — NON-CIRCULAR diagnosis of the kernel re-center statistic
for Track-C (DLA-detection CDDF).

Resolves the b_ref contradiction from the truth-match ONLY (never from R0):
 1. conditional dx=xhat-xtrue distribution per (xhat,z) cell: mean vs median vs
    mode + skew, in the DLA tier (xhat>=20.3) and the sub-DLA bulk (xhat~20.0).
 2. broaden012's built-in shift b_base(xhat)=xhat-mu_col per xhat bin.
 3. (Part 4, separate script) after-the-fact R0(z) CHECK under each recipe.

Reduce-only. Cached kernel. NO inference. Uses ab_loa0_fp_baseline.build_ingredients.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.ab_loa0_fp_baseline import (
    build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL, DEF_KERNEL, DEF_LOA0_PRODUCT,
)
from CDDF_analysis.cddf_catalog_hbi import _zbin_index, _fine_z_grid


def _mode_kde(x, bw=None):
    """Crude mode via histogram peak (robust, no scipy KDE dependency)."""
    x = np.asarray(x, float)
    if len(x) < 5:
        return float(np.median(x))
    nb = max(8, int(np.sqrt(len(x))))
    h, e = np.histogram(x, bins=nb)
    k = int(np.argmax(h))
    return float(0.5 * (e[k] + e[k + 1]))


def _skew(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan
    m = x.mean(); s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--catalog-dir", default=DEF_CAT)
    p.add_argument("--truth", default=DEF_TRUTH)
    p.add_argument("--bal-cat", default=DEF_BAL)
    p.add_argument("--molly-tsv", default=None)
    p.add_argument("--kernel", default=DEF_KERNEL)
    p.add_argument("--loa0-product", default=DEF_LOA0_PRODUCT)
    p.add_argument("--out", default="/tmp/track_c_bref")
    p.add_argument("--mockdir", default=None)
    p.add_argument("--zbins", default="2.0,2.5,3.0,3.5")
    p.add_argument("--report-limits", default="20.0,20.3,20.6")
    p.add_argument("--family", default="bspbody")
    p.add_argument("--fit-floor", type=float, default=19.5)
    p.add_argument("--fit-ceil", type=float, default=99.0)
    p.add_argument("--lambda-bspbody", type=float, default=30.0)
    p.add_argument("--lam-rf-min", type=float, default=1025.0)
    p.add_argument("--edge-slope-lam", type=float, default=40.0)
    p.add_argument("--gl-nodes", type=int, default=1)
    p.add_argument("--host-truth-floor", type=float, default=19.0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    ing = build_ingredients(args, "loa0", loa0_product=args.loa0_product)
    cfg = ing["cfg"]; cat_cut = ing["cat_cut"]
    good_mask = ing["good_mask"]
    zbins = np.asarray(cfg.zbins, float)
    z_edges_fine = _fine_z_grid(cfg)

    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    xhat = np.asarray(cat_cut["NHI"], float)[op]
    # truth host: use NHI_TILT_HOST if present (==NHI_TRUE at floor), else NHI_TRUE
    host_col = "NHI_TILT_HOST" if "NHI_TILT_HOST" in cat_cut.colnames else "NHI_TRUE"
    xtrue = np.asarray(cat_cut[host_col], float)[op]
    z = np.asarray(cat_cut["Z_DLA"], float)[op]
    tp = np.isfinite(xtrue)
    xhat, xtrue, z = xhat[tp], xtrue[tp], z[tp]
    dx = xhat - xtrue
    print(f"\nTRUTH-MATCHED OP DETECTIONS (TP): n={len(dx)}")
    print(f"  global dx: mean={dx.mean():+.4f} median={np.median(dx):+.4f} "
          f"mode={_mode_kde(dx):+.4f} skew={_skew(dx):+.3f} sig={dx.std():.4f}")

    # =====================================================================
    # PART 1 — skew of the conditional dx per (xhat, z) cell
    # =====================================================================
    # xhat bins spanning sub-DLA bulk -> DLA tier
    xedges = np.array([19.5, 19.7, 19.9, 20.0, 20.1, 20.3, 20.6, 21.0, 23.0])
    print("\n" + "=" * 100)
    print("PART 1a — conditional dx by xhat bin (z marginalized), TP only")
    print("=" * 100)
    print(f"  {'xhat_bin':>14} {'n':>6} {'mean':>8} {'median':>8} {'mode':>8} "
          f"{'mean-med':>9} {'skew':>7} {'sig':>7}")
    rows_x = []
    for a, b in zip(xedges[:-1], xedges[1:]):
        m = (xhat >= a) & (xhat < b)
        if m.sum() < 5:
            continue
        d = dx[m]
        mn, md, mo = d.mean(), np.median(d), _mode_kde(d)
        rows_x.append((a, b, m.sum(), mn, md, mo, mn - md, _skew(d), d.std()))
        print(f"  [{a:>5.2f},{b:>5.2f}) {m.sum():>6d} {mn:>+8.4f} {md:>+8.4f} "
              f"{mo:>+8.4f} {mn-md:>+9.4f} {_skew(d):>+7.3f} {d.std():>7.4f}")

    # z-resolved in the DLA tier and the bulk
    zc = 0.5 * (zbins[:-1] + zbins[1:])
    zidx = _zbin_index(z, zbins)
    print("\n" + "=" * 100)
    print("PART 1b — conditional dx in DLA tier (xhat>=20.3) and bulk (20.0<=xhat<20.1), by z")
    print("=" * 100)
    rows_xz = []
    for label, xsel in (("DLA_tier(xhat>=20.3)", xhat >= 20.3),
                        ("bulk(20.0<=xhat<20.1)", (xhat >= 20.0) & (xhat < 20.1)),
                        ("subdla(19.5<=xhat<20.0)", (xhat >= 19.5) & (xhat < 20.0))):
        print(f"\n  --- {label} ---")
        print(f"  {'z':>5} {'n':>6} {'mean':>8} {'median':>8} {'mode':>8} "
              f"{'mean-med':>9} {'skew':>7}")
        for k in range(len(zc)):
            m = xsel & (zidx == k)
            if m.sum() < 5:
                print(f"  {zc[k]:>5.2f} {m.sum():>6d}   (too few)")
                continue
            d = dx[m]
            mn, md, mo = d.mean(), np.median(d), _mode_kde(d)
            rows_xz.append((label, zc[k], m.sum(), mn, md, mo, mn - md, _skew(d)))
            print(f"  {zc[k]:>5.2f} {m.sum():>6d} {mn:>+8.4f} {md:>+8.4f} "
                  f"{mo:>+8.4f} {mn-md:>+9.4f} {_skew(d):>+7.3f}")

    # =====================================================================
    # PART 3 — broaden012 built-in shift b_base(xhat) = xhat - mu_col
    # =====================================================================
    # load the cached kernel, row-aligned to the op set; mu_col is the
    # mass-weighted mean of the N-response (summed over z) per op detection.
    d = np.load(args.kernel, allow_pickle=True)
    kappa = d["kappa"].astype(np.float64)           # (n_obs, n_N, n_z)
    from CDDF_analysis.cddf_catalog_hbi import build_fine_grid
    logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
    mids = 0.5 * (np.asarray(logN_lo, float) + np.asarray(logN_hi, float))
    n_obs = kappa.shape[0]
    print(f"\n  kernel shape {kappa.shape}; op-set xhat len {len(xhat)} "
          f"(NOTE: kernel rows align to the FULL floored op set, not just TP)")
    # We need the kernel's op-set xhat to align rows. Rebuild the full op xhat (TP+FP).
    xhat_op_full = np.asarray(cat_cut["NHI"], float)[op]
    if n_obs != len(xhat_op_full):
        print(f"  WARNING: kernel rows {n_obs} != op rows {len(xhat_op_full)}; "
              "b_base computed on min-length overlap")
    nlap = min(n_obs, len(xhat_op_full))
    col_mass = kappa[:nlap].sum(axis=2)                      # (nlap, n_N)
    tot = col_mass.sum(axis=1)
    good = tot > 0
    mu_col = np.full(nlap, np.nan)
    mu_col[good] = (col_mass[good] * mids[None, :]).sum(axis=1) / tot[good]
    b_base = xhat_op_full[:nlap] - mu_col                    # xhat - posterior mean
    print("\n" + "=" * 100)
    print("PART 3 — broaden012 built-in shift b_base(xhat) = xhat - mu_col(posterior mean)")
    print("=" * 100)
    print(f"  {'xhat_bin':>14} {'n':>6} {'mean_b_base':>12} {'med_b_base':>12} "
          f"{'mean_mu_col':>12}")
    rows_bb = []
    for a, b in zip(xedges[:-1], xedges[1:]):
        m = (xhat_op_full[:nlap] >= a) & (xhat_op_full[:nlap] < b) & good
        if m.sum() < 5:
            continue
        bb = b_base[m]
        rows_bb.append((a, b, m.sum(), np.mean(bb), np.median(bb), np.mean(mu_col[m])))
        print(f"  [{a:>5.2f},{b:>5.2f}) {m.sum():>6d} {np.mean(bb):>+12.4f} "
              f"{np.median(bb):>+12.4f} {np.mean(mu_col[m]):>12.4f}")

    # save tables
    with open(os.path.join(args.out, "part1a_dx_by_xhat.tsv"), "w") as fh:
        fh.write("xlo\txhi\tn\tmean\tmedian\tmode\tmean_minus_med\tskew\tsig\n")
        for r in rows_x:
            fh.write("\t".join(f"{v:.6g}" for v in r) + "\n")
    with open(os.path.join(args.out, "part1b_dx_by_xhat_z.tsv"), "w") as fh:
        fh.write("sel\tz\tn\tmean\tmedian\tmode\tmean_minus_med\tskew\n")
        for r in rows_xz:
            fh.write(f"{r[0]}\t" + "\t".join(f"{v:.6g}" for v in r[1:]) + "\n")
    with open(os.path.join(args.out, "part3_bbase.tsv"), "w") as fh:
        fh.write("xlo\txhi\tn\tmean_b_base\tmed_b_base\tmean_mu_col\n")
        for r in rows_bb:
            fh.write("\t".join(f"{v:.6g}" for v in r) + "\n")
    print(f"\n[done] tables -> {args.out}")
    return dict(rows_x=rows_x, rows_xz=rows_xz, rows_bb=rows_bb,
                dx=dx, xhat=xhat, z=z)


if __name__ == "__main__":
    main()
