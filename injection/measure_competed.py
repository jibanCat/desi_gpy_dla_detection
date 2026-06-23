#!/usr/bin/env python
"""COMPETED completeness measurement for the natural-host sub-DLA campaign (campaign 'S').

Critical fix from the scientific-validity review: on a natural HCD-bearing host running
max_dlas=1, the single reported absorber may be the HOST's real HCD, not the injected one.
A bare p_DLA>thresh count would mis-attribute that to the injection and inflate competed
completeness to ~1. So a recovery is credited to the injected absorber ONLY when the
recovered redshift z_rec matches the injected z_true within the SAME tolerance the
production purity/completeness matcher uses: |z_rec − z_true|/(1+z_true) < DZ_REL (0.01 ≈
3000 km/s = MIN_Z_SEPARATION). This mirrors examples/{gp_native,molly_faithful}_pc_plots.py.

Reports competed completeness C(logN_true) (fine + the [19.5,20.3) headline), the recovered-N
migration, and stratification by the host's natural competitor (comp_logN_max, comp_dz_near).
Excludes forest_blend rows (injected z landed on a pre-existing near-black host feature).
"""
import argparse, glob, os
import numpy as np
from astropy.table import Table, vstack

DZ_REL = 0.01                       # = production matcher tolerance (Δv/c ≈ 3000 km/s)
P_THRESH = 0.5
SUB = (19.5, 20.3)                  # headline competed sub-DLA bin


def _recovered_by_tid(gp_out):
    """{TARGETID -> [(P_DLA, NHI, Z_DLA), ...]} over all dlacat chunks (P>thresh kept)."""
    fs = sorted(glob.glob(f"{gp_out}/dlacat-*.fits"))
    if not fs:
        raise SystemExit(f"[measure] no dlacat-*.fits under {gp_out} (GP run not finished?)")
    dc = vstack([Table.read(f) for f in fs])
    out = {}
    P = np.asarray(dc["P_DLA"], float); N = np.asarray(dc["NHI"], float)
    Z = np.asarray(dc["Z_DLA"], float); T = np.asarray(dc["TARGETID"], np.int64)
    for i in range(len(dc)):
        if P[i] > P_THRESH:
            out.setdefault(int(T[i]), []).append((P[i], N[i], Z[i]))
    return out


def _match(rec_list, z_true, logN_true):
    """Return (recovered_bool, recN) for the INJECTED absorber: a recovered row whose
    z matches z_true within DZ_REL; tie-break (rare, max_dlas=1) by |NHI−logN_true|."""
    best = None
    for P, N, Z in rec_list:
        if abs(Z - z_true) / (1.0 + z_true) < DZ_REL:
            key = abs(N - logN_true)
            if best is None or key < best[0]:
                best = (key, N)
    return (best is not None), (best[1] if best else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True, help="campaignS tree root (has injection_truth.fits + gp_out/)")
    ap.add_argument("--gp_out", default=None)
    ap.add_argument("--keep_blends", action="store_true", help="do NOT exclude forest_blend rows")
    a = ap.parse_args()
    gp_out = a.gp_out or f"{a.campaign}/gp_out"

    man = Table.read(f"{a.campaign}/injection_truth.fits")
    man = man[~np.asarray(man["control"], bool)] if "control" in man.colnames else man
    rec = _recovered_by_tid(gp_out)

    tid = np.asarray(man["target_id"], np.int64); zt = np.asarray(man["z_true"], float)
    lNt = np.asarray(man["logN_true"], float)
    blend = np.asarray(man["forest_blend"], bool) if "forest_blend" in man.colnames else np.zeros(len(man), bool)
    clx = np.asarray(man["comp_logN_max"], float) if "comp_logN_max" in man.colnames else np.full(len(man), np.nan)
    cdz = np.asarray(man["comp_dz_near"], float) if "comp_dz_near" in man.colnames else np.full(len(man), np.nan)

    det = np.zeros(len(man), bool); recN = np.full(len(man), np.nan)
    for i in range(len(man)):
        d, n = _match(rec.get(int(tid[i]), []), zt[i], lNt[i])
        det[i] = d; recN[i] = n

    use = np.ones(len(man), bool)
    if not a.keep_blends:
        use &= ~blend
    print(f"[measure] {use.sum()}/{len(man)} injections used "
          f"({(~use).sum()} forest_blend excluded), DZ_REL={DZ_REL}")

    # competed completeness per fine logN_true bin
    print(f"\n=== COMPETED completeness C(logN_true) [z-matched, |dz|/(1+z)<{DZ_REL}] ===")
    print(f"  {'logN':>6}{'n':>6}{'C_competed':>12}{'<recN|det>':>11}")
    for lv in sorted(set(np.round(lNt[use], 2))):
        m = use & (np.round(lNt, 2) == lv); n = int(m.sum())
        if n == 0:
            continue
        C = det[m].mean(); rN = np.nanmean(recN[m & det]) if (m & det).any() else np.nan
        print(f"  {lv:6.2f}{n:6d}{C:12.3f}{rN:11.2f}")

    # headline sub-DLA [19.5,20.3) + stratification by competitor
    sub = use & (lNt >= SUB[0]) & (lNt < SUB[1])
    C_sub = det[sub].mean()
    has_dla = sub & (clx >= 20.3)            # host has a DLA competitor
    no_comp = sub & ~(clx >= 17.2)           # host has no HCD competitor (NaN comp)
    print(f"\n=== HEADLINE competed sub-DLA [{SUB[0]},{SUB[1]}) ===")
    print(f"  C_competed (all hosts)       = {C_sub:.3f}  (n={int(sub.sum())})")
    if has_dla.any():
        print(f"  C | host has DLA competitor  = {det[has_dla].mean():.3f}  (n={int(has_dla.sum())})")
    if no_comp.any():
        print(f"  C | host has NO HCD compet.  = {det[no_comp].mean():.3f}  (n={int(no_comp.sum())})")
    print(f"  vs truth-based f_raw/f_truth competed completeness ~0.76 (BAL-free, snr>2 selection)")

    # LLS migration into recN>=19.5 (the upper-limit-relevant channel)
    lls = use & (lNt < 19.0) & det
    if lls.any():
        frac_up = np.mean(recN[lls] >= 19.5)
        print(f"\n=== LLS<19.0 migration: of detected LLS, frac recovered at recN>=19.5 = {frac_up:.3f} "
              f"(n={int(lls.sum())}) ===")

    np.savez(f"{a.campaign}/competed_completeness.npz",
             logN_true=lNt[use], detected=det[use], recN=recN[use],
             comp_logN_max=clx[use], comp_dz_near=cdz[use], dz_rel=DZ_REL, C_sub=C_sub)
    print(f"\n[save] {a.campaign}/competed_completeness.npz")


if __name__ == "__main__":
    main()
