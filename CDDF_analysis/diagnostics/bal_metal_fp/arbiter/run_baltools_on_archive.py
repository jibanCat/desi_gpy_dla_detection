#!/usr/bin/env python
"""run_baltools_on_archive.py — run OUR own DESI balfinder (Paul Martini's baltools) on the
compressed LOA spectra archive (`loa_full_z2_*.h5`), for FULL CONTROL of the BAL VAC used in
the DLA high-N false-positive arbiter (no dependence on Paul's v2/v3 altbal version).

Why the archive (not the healpix tree): it is COMPLETE (928,920 z>2 QSOs vs the partial 856-hp
tree), PRE-COADDED on a clean common grid (3600-9824 Å, 0.8 Å) — so it covers the CIV region AND
sidesteps the camera-alignment warning that degraded the mock run. VALIDATED: our BI/AI reproduce
Paul's v2 VAC to the integer (strong-BAL cross-check).

Bridge: the archive gives (wave, flux, ivar, z) per TARGETID; baltools' core fitter
`fitbal.calcbalparams(qsospec, pcaeigen, z)` takes exactly a structured array with wave/flux/ivar
(no desispec Spectra needed). We collect BI_CIV, AI_CIV, the trough velocities (VMIN/VMAX_CIV_450/2000
→ native broad-trough), and the finder's own SNR_CIV/SNR_REDSIDE.

Env: source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate gpdla
     export PYTHONPATH=/home/mfho/baltools/py:$PYTHONPATH
Run: python run_baltools_on_archive.py --out our_loa_bal_vac.fits [--nmax N] [--nproc 16]
Aggregate/real-LOA privacy: output is a per-QSO BAL VAC (BI/AI/troughs) — treat as the derived
BAL catalog, not raw spectra.
"""
import os, argparse, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import h5py, fitsio
from baltools import fitbal

ARCHIVE = "/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5"
PCA = "/home/mfho/baltools/data/PCA_Eigenvectors_Brodzeller.fits"
_DT = np.dtype([("wave", ">f8"), ("flux", ">f8"), ("ivar", ">f8"), ("model", ">f8")])

# scalar fields we pull from the calcbalparams `info` dict (arrays flattened to first-trough where needed)
SCALARS = ["BI_CIV", "BI_CIV_ERR", "AI_CIV", "AI_CIV_ERR", "NCIV_2000", "NCIV_450",
           "SNR_CIV", "SNR_REDSIDE", "SNR_FOREST"]

_WAVE = None
_PCA = None


def _init(wave, pca):
    global _WAVE, _PCA
    _WAVE, _PCA = wave, pca


def _fit_one(args):
    tid, z, flux, ivar = args
    qs = np.zeros(len(_WAVE), dtype=_DT)
    qs["wave"] = _WAVE; qs["flux"] = flux; qs["ivar"] = ivar
    try:
        info, _, _ = fitbal.calcbalparams(qs, _PCA, float(z))
    except Exception:
        return (int(tid), float(z)) + tuple([np.nan] * len(SCALARS)) + (0.0, 0.0)
    out = [int(tid), float(z)]
    for k in SCALARS:
        v = info.get(k, np.nan)
        out.append(float(np.atleast_1d(v)[0]) if v is not None else np.nan)
    # widest CIV trough (km/s) from the AI-450 velocity arrays -> native broad-trough flag
    vmn = np.atleast_1d(info.get("VMIN_CIV_450", [0])); vmx = np.atleast_1d(info.get("VMAX_CIV_450", [0]))
    w = np.abs(np.asarray(vmx, float) - np.asarray(vmn, float))
    w[(np.asarray(vmn, float) == 0) & (np.asarray(vmx, float) == 0)] = 0
    widest450 = float(w.max()) if w.size else 0.0
    # widest BI (2000) trough
    vmn2 = np.atleast_1d(info.get("VMIN_CIV_2000", [0])); vmx2 = np.atleast_1d(info.get("VMAX_CIV_2000", [0]))
    w2 = np.abs(np.asarray(vmx2, float) - np.asarray(vmn2, float))
    w2[(np.asarray(vmn2, float) == 0) & (np.asarray(vmx2, float) == 0)] = 0
    widest2000 = float(w2.max()) if w2.size else 0.0
    return tuple(out) + (widest450, widest2000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE); ap.add_argument("--pca", default=PCA)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nmax", type=int, default=None, help="cap #QSOs (smoke/timing)")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--z-min", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=4000, help="QSOs per I/O chunk")
    a = ap.parse_args()
    import multiprocessing as mp

    pca = fitsio.read(a.pca)
    with h5py.File(a.archive, "r") as h:
        wave = h["wavelength"][:]
        cat = h["catalog"][:]
        tid = np.asarray(cat["TARGETID"], np.int64); z = np.asarray(cat["Z"], float)
        sel = np.where(z > a.z_min)[0]
        if a.nmax:
            sel = sel[: a.nmax]
        n = len(sel)
        print(f"[baltools-on-archive] {n} QSOs (z>{a.z_min}) of {len(tid)}; nproc={a.nproc}; chunk={a.chunk}")
        cols = ["TARGETID", "Z"] + SCALARS + ["WIDEST_CIV_450", "WIDEST_CIV_2000"]
        rows = []
        t0 = time.time()
        pool = mp.Pool(a.nproc, initializer=_init, initargs=(wave, pca))
        for c0 in range(0, n, a.chunk):
            ci = sel[c0: c0 + a.chunk]
            ci_sorted = np.sort(ci)                      # h5 fancy-index needs increasing order
            flux = h["flux"][ci_sorted]; ivar = h["ivar"][ci_sorted]
            args = [(tid[j], z[j], flux[m], ivar[m]) for m, j in enumerate(ci_sorted)]
            rows.extend(pool.map(_fit_one, args, chunksize=32))
            done = c0 + len(ci); rate = done / (time.time() - t0)
            print(f"  {done}/{n}  ({rate:.0f} QSO/s, eta {(n-done)/max(rate,1)/60:.0f} min)", flush=True)
        pool.close(); pool.join()

    arr = np.array(rows, dtype=[(c, "f8" if c != "TARGETID" else "i8") for c in cols])
    fitsio.write(a.out, arr, clobber=True)
    print(f"\n-> {a.out}  ({len(arr)} QSOs, {time.time()-t0:.0f}s)")
    bi = arr["BI_CIV"]; ai = arr["AI_CIV"]
    print(f"   BI>0: {(bi>0).sum()}  AI>0: {(ai>0).sum()}  broad-trough(>2000, our native): {((arr['WIDEST_CIV_450']>2000)&(ai>0)).sum()}")


if __name__ == "__main__":
    main()
