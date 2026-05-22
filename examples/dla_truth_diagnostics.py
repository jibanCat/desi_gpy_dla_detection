"""
examples/dla_truth_diagnostics.py
=================================
GP-vs-truth diagnostic figures for a mock run, complementing
``molly_faithful_pc_plots.py`` (reuses its loader + nhi-desc matcher from
``gp_native_pc_plots`` so the matched set is consistent with the P/C numbers).

Figures
-------
1. ``diag_dNHI_hist.png``  — histogram of (logNHI_GP − logNHI_true) on matched
   (true-positive) DLAs. Reveals column-density bias.
2. ``diag_dz_hist.png``    — histogram of (z_DLA_GP − z_DLA_true) on matched
   DLAs. Reveals redshift bias.
3. ``diag_pair_dv.png``    — number of DLA *pairs* (same sightline) vs velocity
   separation Δv, GP catalog vs truth catalog, on the SAME processed sightlines
   and cuts. The ``MIN_Z_SEPARATION`` (default 3000 km/s) floor is marked: the
   GP cannot resolve pairs below it. A truth excess of small-Δv pairs that the
   GP does not recover = evidence the uniform/independent z_DLA prior (and the
   3000 km/s floor) under-represents DLA clustering.

Cuts mirror the molly recipe: restrict to processed sightlines (via the
``snr_cat``), SNR_REDSIDE>snr_min, P_DLA>gp_conf, DLAFLAG==0, NHI>nhi_min, BAL
excluded; truth restricted to the same processed sightlines + NHI>nhi_min.

Usage
-----
    python examples/dla_truth_diagnostics.py \
        --catalog-dir <OUTDIR> --truth <mock>/dla_cat.fits \
        --snr-cat <OUTDIR>/../snr_cat.fits --bal-cat <mock>/bal_cat.fits \
        --out-dir <OUTDIR>/../diagnostics
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gp_native_pc_plots import (  # noqa: E402
    load_catalog_dir, load_truth, apply_bal_cut, match_truth_to_cat,
)

C_KMS = 299792.458


def pair_dv(tids: np.ndarray, zs: np.ndarray):
    """All within-sightline pairwise velocity separations |Δv| (km/s) and the
    pair mean redshift. Δv = c·|Δz|/(1+z_mean)."""
    order = np.argsort(tids, kind="stable")
    tids, zs = tids[order], zs[order]
    dvs, zms = [], []
    i = 0
    n = len(tids)
    while i < n:
        j = i
        while j < n and tids[j] == tids[i]:
            j += 1
        zg = zs[i:j]
        if zg.size >= 2:
            for a in range(zg.size):
                for b in range(a + 1, zg.size):
                    zm = 0.5 * (zg[a] + zg[b])
                    dvs.append(C_KMS * abs(zg[a] - zg[b]) / (1.0 + zm))
                    zms.append(zm)
        i = j
    return np.asarray(dvs), np.asarray(zms)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog-dir", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--snr-cat", required=True,
                    help="snr_cat.fits (TARGETID, SNR_REDSIDE) from make_snr_cat_from_processed.py "
                         "— defines processed sightlines + SNR.")
    ap.add_argument("--bal-cat", default=None)
    ap.add_argument("--no-bal", action="store_true")
    ap.add_argument("--nhi-min", type=float, default=20.3)
    ap.add_argument("--snr-min", type=float, default=2.0)
    ap.add_argument("--gp-conf", type=float, default=0.99)
    ap.add_argument("--dz-rel", type=float, default=0.01)
    ap.add_argument("--min-z-sep-kms", type=float, default=3000.0,
                    help="MIN_Z_SEPARATION used at inference (marked on the pair plot).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cat = load_catalog_dir(args.catalog_dir)
    truth = load_truth(args.truth, args.nhi_min)
    if args.no_bal and args.bal_cat:
        cat, truth = apply_bal_cut(cat, truth, args.bal_cat)

    # processed sightlines + SNR from snr_cat
    snr = fitsio.read(args.snr_cat, ext=1)
    proc = {int(t): float(s) for t, s in zip(snr["TARGETID"], snr["SNR_REDSIDE"])}
    keep_snr = {t for t, s in proc.items() if s > args.snr_min}

    def restrict(tab):
        tids = np.asarray(tab["TARGETID"]).astype(int)
        return tab[np.isin(tids, list(keep_snr))]

    cat, truth = restrict(cat), restrict(truth)

    # GP detection cuts
    m = ((np.asarray(cat["P_DLA"], float) > args.gp_conf)
         & (np.asarray(cat["DLAFLAG"], int) == 0)
         & (np.asarray(cat["NHI"], float) > args.nhi_min))
    cat = cat[m]
    print(f"[cuts] cat→{len(cat)} (P_DLA>{args.gp_conf}, DLAFLAG==0, NHI>{args.nhi_min}, "
          f"SNR>{args.snr_min}, processed); truth→{len(truth)}")

    # match
    cat_is_TP, cat_NHI_TR, cat_Z_TR, truth_matched = match_truth_to_cat(cat, truth, args.dz_rel)
    tp = cat_is_TP
    dNHI = np.asarray(cat["NHI"], float)[tp] - cat_NHI_TR[tp]
    dz = np.asarray(cat["Z_DLA"], float)[tp] - cat_Z_TR[tp]
    print(f"[match] {tp.sum()} TP DLAs;  ΔNHI median={np.median(dNHI):+.3f} std={np.std(dNHI):.3f};  "
          f"Δz median={np.median(dz):+.5f} std={np.std(dz):.5f}")

    # ---- Fig 1: ΔNHI ----
    plt.figure(figsize=(6, 4))
    plt.hist(dNHI, bins=np.linspace(-1.5, 1.5, 61), color="C0", alpha=0.85)
    plt.axvline(0, color="k", lw=1)
    plt.axvline(np.median(dNHI), color="C3", ls="--", lw=1.5,
                label=f"median {np.median(dNHI):+.3f}")
    plt.xlabel(r"$\log N_{\rm HI}^{\rm GP} - \log N_{\rm HI}^{\rm true}$")
    plt.ylabel("matched DLAs")
    plt.title(f"NHI residual (n={tp.sum()})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "diag_dNHI_hist.png"), dpi=130)
    plt.close()

    # ---- Fig 2: Δz ----
    plt.figure(figsize=(6, 4))
    plt.hist(dz, bins=np.linspace(-0.05, 0.05, 61), color="C2", alpha=0.85)
    plt.axvline(0, color="k", lw=1)
    plt.axvline(np.median(dz), color="C3", ls="--", lw=1.5,
                label=f"median {np.median(dz):+.5f}")
    plt.xlabel(r"$z_{\rm DLA}^{\rm GP} - z_{\rm DLA}^{\rm true}$")
    plt.ylabel("matched DLAs")
    plt.title(f"z residual (n={tp.sum()})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "diag_dz_hist.png"), dpi=130)
    plt.close()

    # ---- Fig 3: pair Δv (clustering test) ----
    gp_dv, gp_zm = pair_dv(np.asarray(cat["TARGETID"]).astype(int),
                           np.asarray(cat["Z_DLA"], float))
    tr_dv, tr_zm = pair_dv(np.asarray(truth["TARGETID"]).astype(int),
                           np.asarray(truth["Z_TRUTH"], float))
    # The inference floor is Δz = kms_to_z(min_z_sep) = min_z_sep_kms/c, so in
    # proper Δv = c·Δz/(1+z) the floor is min_z_sep_kms/(1+z) — z-dependent.
    z_med = float(np.median(np.concatenate([gp_zm, tr_zm]))) if (gp_zm.size + tr_zm.size) else 2.5
    floor_dv = args.min_z_sep_kms / (1.0 + z_med)
    bins = np.linspace(0, 15000, 31)  # 500 km/s bins to 15,000
    plt.figure(figsize=(7, 4.5))
    plt.hist(tr_dv, bins=bins, histtype="step", lw=2, color="k",
             label=f"truth pairs (n={tr_dv.size})")
    plt.hist(gp_dv, bins=bins, histtype="stepfilled", alpha=0.55, color="C0",
             label=f"GP pairs (n={gp_dv.size})")
    plt.axvline(floor_dv, color="C3", ls="--", lw=1.5,
                label=f"GP floor Δz=0.01 ≈ {floor_dv:.0f} km/s (z≈{z_med:.1f})")
    plt.xlabel(r"absorber pair velocity separation $\Delta v$ [km/s]")
    plt.ylabel("number of pairs")
    plt.title(f"Absorber-pair Δv: GP vs truth (NHI≥{args.nhi_min}, same sightlines/cuts)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "diag_pair_dv.png"), dpi=130)
    plt.close()

    n_close_tr = int((tr_dv < floor_dv).sum())
    n_close_gp = int((gp_dv < floor_dv).sum())
    print(f"[pairs] truth={tr_dv.size} (<floor {floor_dv:.0f} km/s: {n_close_tr}); "
          f"GP={gp_dv.size} (<floor: {n_close_gp}); z_med={z_med:.2f}")
    print(f"[out] 3 figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
