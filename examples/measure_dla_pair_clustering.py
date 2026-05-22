"""
examples/measure_dla_pair_clustering.py
=======================================
Smoke-test the clustering assumption behind the velocity-separation prior
(docs/notes/2026-05-03_multidla_velocity_separation_prior_design.md, Option A):

  "DLAs cluster (b_DLA ≈ 2) and pairs at small Δv are MORE PROBABLE than uniform."

Measures the line-of-sight DLA pair correlation 1+ξ(Δv) from a TRUTH catalog:
  - n_pairs(Δv): for each sightline (TARGETID) with ≥2 DLAs above nhi_min, all
    pairwise Δv = c·|z_i − z_j| / (1 + z̄), histogrammed in dv_bin km/s bins.
  - n_random(Δv): the SAME per-sightline DLA counts, but z_DLA permuted across
    all sightlines (breaks line-of-sight correlation while preserving the global
    z distribution + counts), averaged over n_boot permutations.
  - 1 + ξ(Δv) = n_pairs / n_random.  >1 at small Δv  ⇒  real clustering excess
    over the GP's implicit uniform/independent z_DLA prior; ≈1 ⇒ no clustering.

Outputs: <prefix>.npz (dv_mid, n_pairs, n_random, one_plus_xi) + <prefix>.png.

Usage:
    python examples/measure_dla_pair_clustering.py \
        --truth <mock>/dla_cat.fits --nhi-min 20.3 --out-prefix dla_clustering
"""
from __future__ import annotations
import argparse
import numpy as np
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_KMS = 299792.458


def pair_hist(tids, zs, groups, bins):
    """Histogram of within-sightline pairwise Δv for the given z array."""
    dv = []
    for idx in groups:
        zg = zs[idx]
        for a in range(zg.size):
            for b in range(a + 1, zg.size):
                zm = 0.5 * (zg[a] + zg[b])
                dv.append(C_KMS * abs(zg[a] - zg[b]) / (1.0 + zm))
    h, _ = np.histogram(dv, bins=bins)
    return h, len(dv)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--nhi-min", type=float, default=20.3)
    ap.add_argument("--vmax", type=float, default=20000.0)
    ap.add_argument("--dv-bin", type=float, default=500.0)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--out-prefix", default="dla_clustering")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = fitsio.read(args.truth)
    names = {n.upper(): n for n in t.dtype.names}
    zcol = names.get("Z_DLA") or names.get("Z_DLA_NO_RSD") or names.get("Z")
    z = np.asarray(t[zcol], float)
    nhi = np.asarray(t[names["NHI"]], float)
    tid = np.asarray(t[names["TARGETID"]]).astype(np.int64)
    keep = nhi >= args.nhi_min
    z, tid = z[keep], tid[keep]
    print(f"[truth] {z.size} DLAs (NHI≥{args.nhi_min}) over {np.unique(tid).size} sightlines")

    # group indices for sightlines with >=2 DLAs
    order = np.argsort(tid, kind="stable")
    z, tid = z[order], tid[order]
    groups = []
    i, n = 0, len(tid)
    while i < n:
        j = i
        while j < n and tid[j] == tid[i]:
            j += 1
        if j - i >= 2:
            groups.append(np.arange(i, j))
        i = j
    print(f"[truth] {len(groups)} multi-DLA sightlines")

    bins = np.arange(0, args.vmax + args.dv_bin, args.dv_bin)
    dv_mid = 0.5 * (bins[:-1] + bins[1:])

    n_pairs, npair_tot = pair_hist(tid, z, groups, bins)
    print(f"[truth] {npair_tot} pairs (Δv<{args.vmax:.0f}: {int(n_pairs.sum())})")

    # random: permute z across ALL sightlines (preserve counts), recompute pairs
    n_rand = np.zeros(len(dv_mid))
    for _ in range(args.n_boot):
        zr = rng.permutation(z)
        h, _ = pair_hist(tid, zr, groups, bins)
        n_rand += h
    n_rand /= args.n_boot

    with np.errstate(divide="ignore", invalid="ignore"):
        one_plus_xi = np.where(n_rand > 0, n_pairs / n_rand, np.nan)

    np.savez(args.out_prefix + ".npz", dv_mid=dv_mid, n_pairs=n_pairs,
             n_random=n_rand, one_plus_xi=one_plus_xi, nhi_min=args.nhi_min)

    # report small-Δv bins
    print(f"\n{'Δv_bin[km/s]':>14} {'n_pairs':>8} {'n_random':>9} {'1+ξ':>7}")
    for k in range(min(8, len(dv_mid))):
        print(f"{bins[k]:>6.0f}-{bins[k+1]:<6.0f} {int(n_pairs[k]):>8} "
              f"{n_rand[k]:>9.1f} {one_plus_xi[k]:>7.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.step(dv_mid, n_pairs, where="mid", lw=2, color="k", label="truth pairs")
    a1.step(dv_mid, n_rand, where="mid", lw=2, color="C1", ls="--", label="random (z permuted)")
    a1.set_xlabel(r"$\Delta v$ [km/s]"); a1.set_ylabel("n_pairs"); a1.legend()
    a1.set_title(f"DLA pair Δv: truth vs random (NHI≥{args.nhi_min})")
    a2.step(dv_mid, one_plus_xi, where="mid", lw=2, color="C0")
    a2.axhline(1.0, color="k", lw=1, ls=":")
    a2.set_xlabel(r"$\Delta v$ [km/s]"); a2.set_ylabel(r"$1+\xi(\Delta v)$")
    a2.set_title("clustering excess over uniform/independent prior")
    fig.tight_layout(); fig.savefig(args.out_prefix + ".png", dpi=130)
    print(f"\n[out] {args.out_prefix}.npz + {args.out_prefix}.png")


if __name__ == "__main__":
    main()
