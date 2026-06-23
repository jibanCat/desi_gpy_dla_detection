#!/usr/bin/env python
"""gen_wall1_inject.py — WALL-1 FULL-INJECTION arm generator (loa-0 substrate).

Build one tilted-f(N) injection arm into the HCD-free, BAL-free loa-0 mock:
  clean-select (loa-0 = identity, no hcd/bal cut) -> tilted manifest
  (f(N)_2LPT × 10^(Δα(logN−20.3)), z uniform-in-window, one per sightline)
  -> inject Voigt into the loa-0 coadds -> write the GP-readable injectable tree
  + the injected truth catalog (hcd_truth_cat schema: NHI/Z/TARGETID/DLAID/SNR)
  + a restricted pilot_qsocat.fits (only injected targets, keeps the GP run cheap).

The re-inference on the resulting tree uses the UNMODIFIED production GP via the
production-config sbatch (wall1_inject_gl_v1.env -> 2lpt0_gl_v1.env). NOTHING here
touches dla_gp.py / run_bayes_select.py / inference (design discipline).

loa-0 substrate (design §1): the HCD-free / BAL-free byte-identical twin of
production loa-124 — same QSOs, same z range, same 1150 healpix coadds, NO
hcd_truth_cat / dla_cat / bal_cat. So clean = every SNR>2 zcat sightline (the
clean cut is the identity — there is nothing to subtract).

Usage (cheap, on-node, NO sbatch — minutes/arm):
  python injection/gen_wall1_inject.py --out <arm_tree> --dalpha 0.5 --n_inj 200 \
      --n_healpix 4   # PILOT
  python injection/gen_wall1_inject.py --out <arm_tree> --dalpha 0.5 --n_inj 4000
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.table import Table

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # repo root (gpy_dla_detection)
sys.path.insert(0, _HERE)                        # injection modules

from coadd_injection import write_campaign, verify_coadd_consistency  # noqa: E402
from campaign_grid import (  # noqa: E402
    build_tilted_manifest, _empirical_logn_pdf, validate_manifest, LOGN_MAX,
)

# loa-0 = HCD-free / BAL-free twin of loa-124 (design §1). loa-124 truth supplies the
# 2LPT f(N) SHAPE the tilted draw multiplies (the natural population), NOT a parametric guess.
DEFAULT_LOA0 = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                "qq_desi_y3/v2.8.5/mock-0/loa-0")
DEFAULT_LOA124_TRUTH = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                        "qq_desi_y3/v2.8.5/mock-0/loa-124/hcd_truth_cat.fits")
_NSIDE = 16


def build_loa0_clean_table(mockdir, snr_cut=2.0):
    """Clean-sightline table on the loa-0 substrate (clean = identity; no hcd/bal cut).

    loa-0 has NO hcd_truth_cat / bal_cat, so EVERY zcat sightline is clean by
    construction (no pre-existing absorber). Returns the same column set
    build_clean_table emits: TARGETID(int64), Z, TARGET_RA, TARGET_DEC, HEALPIX(int64),
    + the snr_cat SNR columns — but without the set-difference (there is nothing to
    subtract). Restricted to SNR_REDSIDE > snr_cut (the DLA-uncorrelated red-side cut).
    """
    import healpy as hp

    zcat = Table.read(f"{mockdir}/zcat.fits")
    snr_cat = Table.read(f"{mockdir}/snr_cat.fits")

    tid = np.asarray(zcat["TARGETID"], dtype=np.int64)
    ra_col = "TARGET_RA" if "TARGET_RA" in zcat.colnames else "RA"
    dec_col = "TARGET_DEC" if "TARGET_DEC" in zcat.colnames else "DEC"
    ra = np.asarray(zcat[ra_col], dtype=np.float64)
    dec = np.asarray(zcat[dec_col], dtype=np.float64)
    z = np.asarray(zcat["Z"], dtype=np.float64)
    healpix = hp.ang2pix(_NSIDE, ra, dec, nest=True, lonlat=True).astype(np.int64)

    out = Table()
    out["TARGETID"] = tid
    out["Z"] = z
    out["TARGET_RA"] = ra
    out["TARGET_DEC"] = dec
    out["HEALPIX"] = healpix

    # left-join SNR columns by TARGETID (same machinery as build_clean_table)
    snr_tid = np.asarray(snr_cat["TARGETID"], dtype=np.int64)
    order = np.argsort(snr_tid, kind="stable")
    snr_tid_sorted = snr_tid[order]
    pos = np.clip(np.searchsorted(snr_tid_sorted, tid), 0, snr_tid_sorted.size - 1)
    found = snr_tid_sorted[pos] == tid
    src_idx = order[pos]
    for name in snr_cat.colnames:
        if name == "TARGETID":
            continue
        col = np.asarray(snr_cat[name])
        joined = np.full(tid.shape, np.nan, dtype=np.float64)
        joined[found] = col[src_idx[found]].astype(np.float64)
        out[name] = joined

    rs = np.asarray(out["SNR_REDSIDE"], float)
    return out[np.isfinite(rs) & (rs > snr_cut)]


def write_injected_truth(manifest_rows, out_root):
    """Write the injected truth catalog in the hcd_truth_cat schema the HBI reads.

    Columns: NHI (logN injected), Z (z injected), TARGETID, DLAID, SNR (native red-side).
    load_and_cut_catalog reads TARGETID/NHI/Z(→Z_DLA) from cfg.truth_path ext=1, so this
    file IS the n_true^tilt the closure compares against. DLAID = TARGETID*1000 + slot
    matches the real hcd_truth_cat convention (3-digit slot suffix); slot 0 = the single
    injected absorber per sightline.
    """
    tid = np.array([int(r["target_id"]) for r in manifest_rows], dtype=np.int64)
    nhi = np.array([float(r["logN_true"]) for r in manifest_rows], dtype=np.float64)
    z = np.array([float(r["z_true"]) for r in manifest_rows], dtype=np.float64)
    snr = np.array([float(r["native_snr"]) for r in manifest_rows], dtype=np.float64)
    dlaid = (tid.astype(np.int64) * 1000)  # slot 0, real hcd_truth_cat convention
    t = Table()
    t["NHI"] = nhi
    t["Z"] = z
    t["TARGETID"] = tid
    t["DLAID"] = dlaid
    t["SNR"] = snr
    path = os.path.join(out_root, "injected_truth_cat.fits")
    t.write(path, overwrite=True, format="fits")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="injectable-tree root for this arm")
    ap.add_argument("--mockdir", default=DEFAULT_LOA0, help="loa-0 substrate")
    ap.add_argument("--fN-truth", default=DEFAULT_LOA124_TRUTH,
                    help="loa-124 truth catalog supplying the 2LPT f(N) shape")
    ap.add_argument("--dalpha", type=float, required=True, help="tilt slope Δα")
    ap.add_argument("--n_inj", type=int, default=4000, help="target injections (arm)")
    ap.add_argument("--n_healpix", type=int, default=0,
                    help="0 = all clean healpix; pilot uses a small subset")
    ap.add_argument("--fit-floor", type=float, default=19.5)
    ap.add_argument("--logN-ceil", type=float, default=LOGN_MAX)
    ap.add_argument("--pivot", type=float, default=20.3)
    ap.add_argument("--snr_cut", type=float, default=2.0)
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260617)
    a = ap.parse_args()

    print(f"[wall1] loa-0 substrate: {a.mockdir}", flush=True)
    clean = build_loa0_clean_table(a.mockdir, snr_cut=a.snr_cut)
    if a.n_healpix:
        # span the host z_QSO range (low->high median) so a pilot subset still samples
        # the full rest-frame forest position (same logic as gen_injectables.py).
        hpx = np.asarray(clean["HEALPIX"], np.int64)
        zq = np.asarray(clean["Z"], float)
        u = np.unique(hpx)
        cnt = np.array([(hpx == h).sum() for h in u])
        floor = max(20, int(np.median(cnt) * 0.25))
        cand = u[cnt >= floor]
        if cand.size < a.n_healpix:
            cand = u
        med = np.array([np.median(zq[hpx == h]) for h in cand])
        order = np.argsort(med)
        pick = np.unique(np.linspace(0, order.size - 1, a.n_healpix).round().astype(int))
        top = cand[order[pick]]
        clean = clean[np.isin(hpx, top)]
    zqa = np.asarray(clean["Z"], float)
    print(f"[clean] {len(clean)} clean loa-0 sightlines on "
          f"{len(set(clean['HEALPIX'].tolist()))} healpix; z_QSO "
          f"[{zqa.min():.2f},{zqa.max():.2f}] median {np.median(zqa):.2f}", flush=True)

    # 2LPT f(N) SHAPE from the loa-124 truth (the natural population the tilt multiplies).
    fN_truth = Table.read(a.fN_truth)
    pdf_2lpt = _empirical_logn_pdf(
        np.asarray(fN_truth["NHI"], float),
        logN_range=(a.fit_floor, a.logN_ceil))

    clean_sl = dict(
        target_id=np.asarray(clean["TARGETID"], np.int64),
        healpix=np.asarray(clean["HEALPIX"], np.int64),
        z_qso=np.asarray(clean["Z"], float),
        native_snr=np.asarray(clean["SNR_REDSIDE"], float))

    manifest = build_tilted_manifest(
        clean_sl, dalpha=a.dalpha, n_inj=a.n_inj, logn_pdf_2lpt=pdf_2lpt,
        fit_floor=a.fit_floor, logN_ceil=a.logN_ceil, pivot=a.pivot,
        seed=a.seed, num_lines=a.num_lines)
    validate_manifest(manifest)
    if not manifest:
        raise SystemExit("[wall1] ERROR: zero injections — clean pool empty or all "
                         "GP windows empty. Add --n_healpix / lower --snr_cut.")
    nlt = np.array([r["logN_true"] for r in manifest])
    print(f"[manifest] Δα={a.dalpha:+.2f}  {len(manifest)} injections; logN "
          f"[{nlt.min():.2f},{nlt.max():.2f}] median {np.median(nlt):.2f}; "
          f"frac>=20.3 = {np.mean(nlt >= 20.3):.2f}", flush=True)
    if len(manifest) < int(0.95 * a.n_inj):
        print(f"[manifest] WARNING: requested {a.n_inj} but the clean pool only "
              f"supports {len(manifest)} ({100*len(manifest)/a.n_inj:.0f}%). Add "
              f"--n_healpix (0 = all) or lower --snr_cut.", flush=True)

    truth_path = write_campaign(manifest, clean, out_root=a.out,
                                mockdir=a.mockdir, num_lines=a.num_lines)
    n = len(glob.glob(f"{a.out}/spectra-16/*/*/spectra-16-*.fits"))
    print(f"[write] {n} injected coadds -> {a.out}/spectra-16/ ; campaign truth "
          f"-> {truth_path}", flush=True)

    # the HBI-readable injected truth catalog (hcd_truth_cat schema).
    inj_truth = write_injected_truth(manifest, a.out)
    print(f"[truth] HBI truth (NHI/Z/TARGETID/DLAID/SNR) -> {inj_truth}", flush=True)

    # restricted qsocat = ONLY injected targets (keeps the GP run cheap).
    zc = Table.read(f"{a.mockdir}/zcat.fits")
    want = np.array(sorted(int(r["target_id"]) for r in manifest), np.int64)
    keep = np.isin(np.asarray(zc["TARGETID"], np.int64), want)
    qpath = os.path.join(a.out, "pilot_qsocat.fits")
    zc[keep].write(qpath, overwrite=True)
    print(f"[qsocat] {keep.sum()} injected targets -> {qpath}", flush=True)

    # sample coadd-consistency check (per-camera injection survives coadd_cameras).
    inj_by_tid = {}
    for r in manifest:
        inj_by_tid.setdefault(int(r["target_id"]), []).append(
            (10.0 ** float(r["logN_true"]), float(r["z_true"])))
    srcs = sorted(glob.glob(f"{a.out}/spectra-16/*/*/spectra-16-*.fits"))
    n_verify = min(3, len(srcs))
    worst_all = 0.0
    for src in srcs[:n_verify]:
        hpx = int(src.rsplit("-", 1)[1].split(".")[0])
        orig = f"{a.mockdir}/spectra-16/{hpx // 100}/{hpx}/spectra-16-{hpx}.fits"
        try:
            worst = verify_coadd_consistency(orig, src, inj_by_tid, num_lines=a.num_lines)
            worst_all = max(worst_all, worst)
            print(f"[verify] hp {hpx}: coadd_cameras(injected)==T*coadd(original) "
                  f"max dev {worst:.2e} (<1e-2 OK)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[verify] hp {hpx}: WARNING: {e!r}", flush=True)
    print(f"[verify] worst dev over {n_verify} healpix: {worst_all:.2e}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
