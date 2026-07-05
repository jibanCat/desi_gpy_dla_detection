"""lyc_break_feasibility.py — is the Lyman-limit BREAK detectable per-sightline at DESI S/N?

The joint-HBI review's core problem (the LLS band is prior-driven) exists because in-band
counting is ~5% pure. If a break-aware GP finder can detect individual LLS by their continuum
break, counting is rescued and a rigid-form joint HBI becomes data-driven. The make-or-break:
is the break a usable PER-SIGHTLINE signal at DESI blue-edge S/N? This measures the matched-
filter detection S/N of the injected LLS break vs N_HI (best case: true continuum + known z,N).

Uses CDDF_analysis.lyc.break_matched_filter_snr. Reads only --limit-healpix files (bounded I/O).
Run: python examples/lyc_break_feasibility.py --limit-healpix 15 --out /tmp/lyc_break
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
from astropy.table import Table
import fitsio

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from preload_spectra.preload_2lpt_simple import _read_one_healpix_file, _spec_path, _healpix_for_radec
from CDDF_analysis.lyc import break_matched_filter_snr, LYMAN_LIMIT

DEF_MOCK = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
CWAVE = 3500.0 + 2.0 * np.arange(3251)   # TRUE_CONT observed grid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mockdir", default=DEF_MOCK)
    ap.add_argument("--limit-healpix", type=int, default=15)
    ap.add_argument("--out", default="/tmp/lyc_break")
    args = ap.parse_args()
    M = Path(args.mockdir); os.makedirs(args.out, exist_ok=True)

    zc = Table.read(M / "zcat.fits")
    z = np.asarray(zc["Z"], float)
    zw = np.asarray(zc["ZWARN"], float) if "ZWARN" in zc.colnames else np.zeros(len(zc))
    zc = zc[(z > 2.9) & (zw == 0)]
    tid_all = np.asarray(zc["TARGETID"])
    zq = dict(zip(tid_all.tolist(), np.asarray(zc["Z"], float).tolist()))
    hp_all = _healpix_for_radec(np.asarray(zc["TARGET_RA"], float), np.asarray(zc["TARGET_DEC"], float))
    tid_hp = dict(zip(tid_all.tolist(), hp_all.tolist()))

    hcd = Table.read(M / "hcd_truth_cat.fits")
    N = np.asarray(hcd["NHI"], float); za = np.asarray(hcd["Z"], float); tid = np.asarray(hcd["TARGETID"])
    sel = (N >= 17.2) & (N < 19.0) & (za > 2.95)
    N, za, tid = N[sel], za[sel], tid[sel]
    keep = np.array([int(t) in zq and za[i] < zq[int(t)] for i, t in enumerate(tid)])
    N, za, tid = N[keep], za[keep], tid[keep]
    hp = np.array([tid_hp.get(int(t), -1) for t in tid])
    uniq = [h for h in np.unique(hp) if h >= 0][:args.limit_healpix]
    m = np.isin(hp, uniq); N, za, tid, hp = N[m], za[m], tid[m], hp[m]
    print(f"[sample] {len(N)} LLS sightlines over {len(uniq)} healpix")

    snr, Ns = [], []
    for h in uniq:
        sf = _spec_path(M, int(h))
        if not sf.exists():
            continue
        tf = str(sf).replace("spectra-16-", "truth-16-")
        try:
            tc = fitsio.read(tf, ext="TRUE_CONT")
        except Exception:
            continue
        tcm = {int(t): np.asarray(tc["TRUE_CONT"])[i] for i, t in enumerate(np.asarray(tc["TARGETID"]))}
        rows = {int(t): (w, iv, mk) for (t, w, f, iv, mk) in
                _read_one_healpix_file(sf, tid[hp == h].tolist())}
        for i in np.where(hp == h)[0]:
            t = int(tid[i])
            if t not in rows or t not in tcm:
                continue
            w, iv, mk = rows[t]
            cont = np.interp(w, CWAVE, tcm[t], left=np.nan, right=np.nan)
            s = break_matched_filter_snr(w, iv, cont, mk, N[i], za[i])
            if np.isfinite(s):
                snr.append(s); Ns.append(N[i])
    snr, Ns = np.array(snr), np.array(Ns)
    print(f"[measured] {len(snr)} LLS with in-band break coverage\n")
    print(f"{'logN bin':>12} | {'median S/N':>10} {'frac>5':>7} {'frac>3':>7} {'N':>5}")
    for lo, hi in [(17.2, 17.5), (17.5, 18.0), (18.0, 18.5), (18.5, 19.0)]:
        b = (Ns >= lo) & (Ns < hi)
        if b.sum():
            print(f"  {lo:.1f}-{hi:.1f}   | {np.median(snr[b]):10.1f} {np.mean(snr[b] > 5):7.2f} "
                  f"{np.mean(snr[b] > 3):7.2f} {b.sum():5d}")
    if len(snr):
        print(f"\n[overall] median break S/N = {np.median(snr):.1f}; frac>5 = {np.mean(snr > 5):.2f}")
        np.savez(os.path.join(args.out, "break_snr.npz"), snr=snr, nhi=Ns)
    print("\nCAVEAT: TRUE_CONT (best case; real needs a continuum fit) + matched filter (knows z,N) "
          "=> this is an UPPER BOUND on per-sightline break detectability.")


if __name__ == "__main__":
    main()
