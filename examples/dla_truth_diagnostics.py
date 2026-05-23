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

Pair-purity mode (--pair-purity)
---------------------------------
Takes two catalog directories (clustering-ON via ``--cat-on`` and OFF via
``--cat-off``) plus a truth catalog (``--truth``).  Reports the purity of
*newly-detected* close DLA pairs: pairs present in the ON catalog for a
sightline that have no corresponding close pair in the OFF catalog.

    purity = (# new pairs that match a true truth pair) / (# new pairs)

A new pair matches truth when *both* members each match a distinct truth DLA on
the same sightline within ``--match-dz`` and the truth pair itself has Δv <
``--dv-max``.

Usage
-----
    # Standard diagnostic figures:
    python examples/dla_truth_diagnostics.py \
        --catalog-dir <OUTDIR> --truth <mock>/dla_cat.fits \
        --snr-cat <OUTDIR>/../snr_cat.fits --bal-cat <mock>/bal_cat.fits \
        --out-dir <OUTDIR>/../diagnostics

    # Pair-purity gate (clustering ON vs OFF):
    python examples/dla_truth_diagnostics.py \
        --pair-purity \
        --cat-on <OUTDIR_ON> --cat-off <OUTDIR_OFF> \
        --truth <mock>/dla_cat.fits
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


# ---------------------------------------------------------------------------
# Pair-purity pure logic — operates on plain dicts; no FITS I/O here
# ---------------------------------------------------------------------------

def close_pairs(
    catalog_by_tid: dict,
    dv_max: float,
) -> list:
    """Find all within-sightline DLA close pairs in a catalog.

    Parameters
    ----------
    catalog_by_tid : dict
        Mapping TARGETID (int) -> list of (z_dla, log_nhi) tuples.
    dv_max : float
        Maximum velocity separation in km/s.

    Returns
    -------
    list of (tid, z_a, z_b) triples (z_a <= z_b by convention) for every pair
    with Δv = c·|Δz|/(1+z_mean) < dv_max.
    """
    pairs = []
    for tid, members in catalog_by_tid.items():
        zs = [m[0] for m in members]
        n = len(zs)
        for a in range(n):
            for b in range(a + 1, n):
                zm = 0.5 * (zs[a] + zs[b])
                dv = C_KMS * abs(zs[a] - zs[b]) / (1.0 + zm)
                if dv < dv_max:
                    za, zb = (zs[a], zs[b]) if zs[a] <= zs[b] else (zs[b], zs[a])
                    pairs.append((int(tid), za, zb))
    return pairs


def matches_truth_pair(
    tid: int,
    z_a: float,
    z_b: float,
    truth_by_tid: dict,
    match_dz: float,
    dv_max: float,
) -> bool:
    """Check whether a GP close pair (tid, z_a, z_b) has a matching truth pair.

    Both pair members must each individually match a *distinct* truth DLA on the
    same sightline within ``match_dz`` in Δz/(1+z_truth), AND the matched truth
    pair must itself have Δv < dv_max.

    Parameters
    ----------
    tid : int
        TARGETID of the sightline.
    z_a, z_b : float
        Redshifts of the two GP DLAs (order does not matter).
    truth_by_tid : dict
        Mapping TARGETID (int) -> list of z_truth floats for that sightline.
    match_dz : float
        Absolute |Δz|/(1+z_truth) tolerance for member-level matching.
    dv_max : float
        Maximum Δv (km/s) the matched truth pair must satisfy.

    Returns
    -------
    bool : True if a qualifying truth-pair match exists.
    """
    truth_zs = truth_by_tid.get(int(tid), [])
    if len(truth_zs) < 2:
        return False

    # Try every assignment: GP_a → truth_i, GP_b → truth_j  (i ≠ j)
    for i, tz_i in enumerate(truth_zs):
        if abs(z_a - tz_i) / (1.0 + tz_i) > match_dz:
            continue
        for j, tz_j in enumerate(truth_zs):
            if j == i:
                continue
            if abs(z_b - tz_j) / (1.0 + tz_j) > match_dz:
                continue
            # Both members matched distinct truth DLAs — check truth-pair Δv
            zm = 0.5 * (tz_i + tz_j)
            dv_truth = C_KMS * abs(tz_i - tz_j) / (1.0 + zm)
            if dv_truth < dv_max:
                return True
    return False


def pair_purity(
    on_by_tid: dict,
    off_by_tid: dict,
    truth_by_tid: dict,
    dv_max: float = 2000.0,
    match_dz: float = 0.005,
    dv_bin_edges: "np.ndarray | None" = None,
) -> tuple:
    """Purity of newly-detected close DLA pairs (clustering gate metric).

    A pair is "new" if it is present in the ON catalog for a sightline but the
    same sightline has no close pair (within a Δv tolerance defined by dv_max)
    in the OFF catalog.  Specifically, an ON pair (tid, za, zb) is "new" if no
    pair in the OFF catalog for the same tid has both members within match_dz of
    the ON pair members.

    Parameters
    ----------
    on_by_tid : dict
        TARGETID -> list of (z_dla, log_nhi) for the clustering-ON catalog.
    off_by_tid : dict
        TARGETID -> list of (z_dla, log_nhi) for the clustering-OFF catalog.
    truth_by_tid : dict
        TARGETID -> list of z_truth floats (truth DLAs, already NHI-filtered).
    dv_max : float
        Velocity separation threshold for "close" pairs (km/s).
    match_dz : float
        |Δz|/(1+z_truth) tolerance for member-level truth matching.
    dv_bin_edges : array-like or None
        Bin edges (km/s) for the per-Δv breakdown.  If None, uses
        10 equal bins from 0 to dv_max.

    Returns
    -------
    purity : float  — n_true_new / n_new  (NaN if n_new == 0)
    n_true_new : int — new pairs that matched a truth pair
    n_new : int — total new pairs
    per_bin : list of dicts with keys 'dv_lo', 'dv_hi', 'n_new', 'n_true_new',
              'purity' — purity broken down by the ON-pair Δv.
    """
    if dv_bin_edges is None:
        dv_bin_edges = np.linspace(0.0, dv_max, 11)

    on_pairs = close_pairs(on_by_tid, dv_max)
    off_pairs_by_tid: dict = {}
    for tid, za, zb in close_pairs(off_by_tid, dv_max):
        off_pairs_by_tid.setdefault(tid, []).append((za, zb))

    n_new = 0
    n_true_new = 0
    bin_counts = np.zeros((len(dv_bin_edges) - 1, 2), dtype=int)  # [:,0]=n_new, [:,1]=n_true

    for tid, za, zb in on_pairs:
        # Is this ON pair "new"?  Check whether OFF has a corresponding pair.
        off_for_tid = off_pairs_by_tid.get(int(tid), [])
        is_new = True
        for (oza, ozb) in off_for_tid:
            # Same pair ≈ both members match within match_dz
            match_a = abs(za - oza) / (1.0 + oza) <= match_dz
            match_b = abs(zb - ozb) / (1.0 + ozb) <= match_dz
            match_ab = abs(za - ozb) / (1.0 + ozb) <= match_dz
            match_ba = abs(zb - oza) / (1.0 + oza) <= match_dz
            if (match_a and match_b) or (match_ab and match_ba):
                is_new = False
                break

        if not is_new:
            continue

        n_new += 1
        zm = 0.5 * (za + zb)
        dv_on = C_KMS * abs(za - zb) / (1.0 + zm)

        is_true = matches_truth_pair(tid, za, zb, truth_by_tid, match_dz, dv_max)
        if is_true:
            n_true_new += 1

        # Bin by the ON-pair Δv
        bi = np.searchsorted(dv_bin_edges, dv_on, side="right") - 1
        bi = max(0, min(bi, len(bin_counts) - 1))
        bin_counts[bi, 0] += 1
        bin_counts[bi, 1] += int(is_true)

    purity = n_true_new / n_new if n_new > 0 else float("nan")

    per_bin = []
    for i in range(len(dv_bin_edges) - 1):
        nn = int(bin_counts[i, 0])
        nt = int(bin_counts[i, 1])
        per_bin.append({
            "dv_lo": float(dv_bin_edges[i]),
            "dv_hi": float(dv_bin_edges[i + 1]),
            "n_new": nn,
            "n_true_new": nt,
            "purity": nt / nn if nn > 0 else float("nan"),
        })

    return purity, n_true_new, n_new, per_bin


# ---------------------------------------------------------------------------
# Adapter: convert a loaded astropy Table catalog into the by_tid dict form
# ---------------------------------------------------------------------------

def _cat_table_to_by_tid(cat) -> dict:
    """Convert an astropy Table with TARGETID/Z_DLA/NHI cols to by_tid dict."""
    import numpy as np
    tids = np.asarray(cat["TARGETID"]).astype(int)
    zs = np.asarray(cat["Z_DLA"], dtype=float)
    nhis = np.asarray(cat["NHI"], dtype=float)
    by_tid: dict = {}
    for tid, z, nhi in zip(tids, zs, nhis):
        by_tid.setdefault(int(tid), []).append((float(z), float(nhi)))
    return by_tid


def _truth_table_to_by_tid(truth) -> dict:
    """Convert an astropy truth Table (with Z_TRUTH col) to by_tid dict of z-lists."""
    import numpy as np
    tids = np.asarray(truth["TARGETID"]).astype(int)
    zs = np.asarray(truth["Z_TRUTH"], dtype=float)
    by_tid: dict = {}
    for tid, z in zip(tids, zs):
        by_tid.setdefault(int(tid), []).append(float(z))
    return by_tid


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


def _run_pair_purity_mode(args):
    """Execute --pair-purity mode: load ON/OFF catalogs + truth, compute purity."""
    cat_on = load_catalog_dir(args.cat_on)
    cat_off = load_catalog_dir(args.cat_off)
    truth = load_truth(args.truth, args.nhi_min)

    # Apply GP detection cuts to ON and OFF catalogs
    def apply_gp_cuts(cat):
        m = (
            (np.asarray(cat["P_DLA"], float) > args.gp_conf)
            & (np.asarray(cat["DLAFLAG"], int) == 0)
            & (np.asarray(cat["NHI"], float) > args.nhi_min)
        )
        return cat[m]

    cat_on = apply_gp_cuts(cat_on)
    cat_off = apply_gp_cuts(cat_off)
    print(f"[pair-purity] ON={len(cat_on)} DLAs, OFF={len(cat_off)} DLAs, "
          f"truth={len(truth)} DLAs")

    on_by_tid = _cat_table_to_by_tid(cat_on)
    off_by_tid = _cat_table_to_by_tid(cat_off)
    truth_by_tid = _truth_table_to_by_tid(truth)

    purity, n_true_new, n_new, per_bin = pair_purity(
        on_by_tid, off_by_tid, truth_by_tid,
        dv_max=args.dv_max, match_dz=args.match_dz,
    )

    print(f"\n[pair-purity] dv_max={args.dv_max:.0f} km/s  match_dz={args.match_dz}")
    print(f"  new pairs   : {n_new}")
    print(f"  true-new    : {n_true_new}")
    print(f"  purity      : {purity:.3f}" if not (isinstance(purity, float) and
                                                    purity != purity) else "  purity : N/A (0 new pairs)")
    print("\n  per-Δv-bin breakdown:")
    print(f"  {'dv_lo':>8}  {'dv_hi':>8}  {'n_new':>6}  {'n_true':>6}  {'purity':>7}")
    for b in per_bin:
        p_str = f"{b['purity']:.3f}" if b['n_new'] > 0 else "  ---  "
        print(f"  {b['dv_lo']:>8.0f}  {b['dv_hi']:>8.0f}  {b['n_new']:>6}  "
              f"{b['n_true_new']:>6}  {p_str:>7}")

    if n_new == 0:
        print("\nVERDICT: no newly-detected close pairs — prior adds no close pairs at this cut.")
    elif purity >= 0.8:
        print(f"\nVERDICT: PASS — purity {purity:.3f} >= 0.80: prior adds clean completeness.")
    elif purity >= 0.5:
        print(f"\nVERDICT: MARGINAL — purity {purity:.3f} in [0.50, 0.80): inspect per-bin.")
    else:
        print(f"\nVERDICT: FAIL — purity {purity:.3f} < 0.50: prior inflates false pairs; retune.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- pair-purity mode ---
    ap.add_argument("--pair-purity", action="store_true",
                    help="Run pair-purity gate: requires --cat-on, --cat-off, --truth.")
    ap.add_argument("--cat-on", default=None,
                    help="Catalog directory for clustering-ON run.")
    ap.add_argument("--cat-off", default=None,
                    help="Catalog directory for clustering-OFF run.")
    ap.add_argument("--dv-max", type=float, default=2000.0,
                    help="Max Δv (km/s) for 'close' pairs (default 2000).")
    ap.add_argument("--match-dz", type=float, default=0.005,
                    help="|Δz|/(1+z_truth) tolerance for member matching (default 0.005).")
    # --- standard diagnostic figures mode ---
    ap.add_argument("--catalog-dir", default=None)
    ap.add_argument("--snr-cat", default=None,
                    help="snr_cat.fits (TARGETID, SNR_REDSIDE) from make_snr_cat_from_processed.py "
                         "— defines processed sightlines + SNR.")
    ap.add_argument("--out-dir", default=None)
    # --- shared args ---
    ap.add_argument("--truth", default=None)
    ap.add_argument("--bal-cat", default=None)
    ap.add_argument("--no-bal", action="store_true")
    ap.add_argument("--nhi-min", type=float, default=20.3)
    ap.add_argument("--snr-min", type=float, default=2.0)
    ap.add_argument("--gp-conf", type=float, default=0.99)
    ap.add_argument("--dz-rel", type=float, default=0.01)
    ap.add_argument("--min-z-sep-kms", type=float, default=3000.0,
                    help="MIN_Z_SEPARATION used at inference (marked on the pair plot).")
    args = ap.parse_args()

    if args.pair_purity:
        missing = [f for f, v in [("--cat-on", args.cat_on),
                                   ("--cat-off", args.cat_off),
                                   ("--truth", args.truth)] if v is None]
        if missing:
            ap.error(f"--pair-purity requires: {', '.join(missing)}")
        _run_pair_purity_mode(args)
        return

    # --- Standard diagnostic figures mode ---
    if args.catalog_dir is None:
        ap.error("--catalog-dir is required for standard diagnostic figures mode.")
    if args.truth is None:
        ap.error("--truth is required.")
    if args.snr_cat is None:
        ap.error("--snr-cat is required for standard diagnostic figures mode.")
    if args.out_dir is None:
        ap.error("--out-dir is required for standard diagnostic figures mode.")
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
