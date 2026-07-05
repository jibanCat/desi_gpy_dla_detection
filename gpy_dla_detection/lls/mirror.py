"""lyc_make_mirror_mock.py — M1: a MIRROR 2LPT mock with the Lyman-limit drop added.

The 2LPT spectra carry quickquasars' HCD absorption as Lyman-SERIES lines only (no bound-free
912 A break; source-verified). Since optical depth is additive (tau_total = tau_lines + tau_LL),
re-doing the HCD absorption with the improved (break-aware) Voigt is IDENTICAL to just
multiplying each existing spectrum by exp(-tau_LL) for its truth HCDs. So we do exactly that —
NO quickquasars re-run — and write a mirror spectra-16 tree the GP finder reads unchanged. Then
the break-aware finder (SubDLAGPMATLymanBreak) can be re-inferred on the mirror (M3) and its LLS
purity compared to the line-only control (M4).

Caveat: flux is scaled by exp(-tau_LL) and ivar is kept (the noise is not re-generated from the
sky model) -> the break region's S/N is approximate; fine for the detectability test, flag for a
rigorous run.

Run (subset): python examples/lyc_make_mirror_mock.py --limit-healpix 2 --out /scratch/.../mirror
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from preload_spectra.preload_2lpt_simple import _spec_path
from CDDF_analysis.lyc import lyc_transmission

DEF_MOCK = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"


def load_hcd_by_tid(mockdir: Path, nhi_min=17.2):
    hcd = Table.read(mockdir / "hcd_truth_cat.fits")
    N = np.asarray(hcd["NHI"], float); z = np.asarray(hcd["Z"], float); tid = np.asarray(hcd["TARGETID"])
    keep = N >= nhi_min
    N, z, tid = N[keep], z[keep], tid[keep]
    order = np.argsort(tid); tid, z, N = tid[order], z[order], N[order]
    uniq, start = np.unique(tid, return_index=True); end = np.r_[start[1:], len(tid)]
    return {int(t): (z[s:e], N[s:e]) for t, s, e in zip(uniq, start, end)}


def _healpix_list(mockdir: Path, limit):
    base = mockdir / "spectra-16"
    hp = []
    for grp in sorted(base.iterdir()):
        if grp.is_dir():
            for hd in sorted(grp.iterdir()):
                if hd.is_dir():
                    hp.append(int(hd.name))
    hp = sorted(hp)
    return hp[:limit] if limit else hp


def inject_file(specfile: Path, out_file: Path, hcd_by_tid, zq_of):
    """Copy spectra-16 file and multiply B/R/Z_FLUX by exp(-tau_LL) per target's truth HCDs."""
    os.makedirs(out_file.parent, exist_ok=True)
    shutil.copy2(specfile, out_file)
    n_inj = 0
    with fits.open(out_file, mode="update") as hdul:
        tids = np.asarray(hdul["FIBERMAP"].data["TARGETID"])
        for cam in ("B", "R", "Z"):
            fk, wk = f"{cam}_FLUX", f"{cam}_WAVELENGTH"
            if fk not in hdul or wk not in hdul:
                continue
            wave = np.asarray(hdul[wk].data, float)
            flux = hdul[fk].data
            for row, t in enumerate(tids):
                t = int(t)
                if t not in hcd_by_tid:
                    continue
                zk, nk = hcd_by_tid[t]
                zqso = zq_of.get(t, None)
                if zqso is not None:
                    m = zk < zqso
                    zk, nk = zk[m], nk[m]
                if zk.size == 0:
                    continue
                flux[row] = flux[row] * lyc_transmission(wave, zk, nk)
                if cam == "B":
                    n_inj += 1
            hdul[fk].data = flux
        hdul.flush()
    return n_inj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mockdir", default=DEF_MOCK)
    ap.add_argument("--out", required=True, help="mirror mock root (a new dir; spectra-16 written under it)")
    ap.add_argument("--limit-healpix", type=int, default=2)
    args = ap.parse_args()
    M = Path(args.mockdir); OUT = Path(args.out)
    os.makedirs(OUT, exist_ok=True)
    # symlink the non-spectra products (zcat, truth cats) so the mirror is a usable mock dir
    for name in ("zcat.fits", "hcd_truth_cat.fits", "bal_cat.fits", "snr_cat.fits", "zcat_gauss_400.fits"):
        src = M / name; dst = OUT / name
        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                pass
    zc = Table.read(M / "zcat.fits")
    zq_of = dict(zip(np.asarray(zc["TARGETID"]).tolist(), np.asarray(zc["Z"], float).tolist()))
    hcd_by_tid = load_hcd_by_tid(M)
    print(f"[hcd] {len(hcd_by_tid)} sightlines carry >=1 HCD (logN>=17.2)")
    hplist = _healpix_list(M, args.limit_healpix)
    print(f"[mirror] injecting LyC drop into {len(hplist)} healpix -> {OUT}")
    tot = 0
    for i, hp in enumerate(hplist):
        sf = _spec_path(M, int(hp))
        of = _spec_path(OUT, int(hp))
        if not sf.exists():
            continue
        n = inject_file(sf, of, hcd_by_tid, zq_of)
        tot += n
        # copy the sibling truth-16 (needed for camera resolution + TRUE_CONT on read)
        tsrc = Path(str(sf).replace("spectra-16-", "truth-16-"))
        tdst = Path(str(of).replace("spectra-16-", "truth-16-"))
        if tsrc.exists() and not tdst.exists():
            try:
                os.symlink(tsrc, tdst)
            except OSError:
                shutil.copy2(tsrc, tdst)
        print(f"  [{i+1}/{len(hplist)}] healpix {hp}: injected break into {n} sightlines")
    print(f"[done] mirror mock at {OUT}; {tot} sightlines given a Lyman-limit break")


if __name__ == "__main__":
    main()
