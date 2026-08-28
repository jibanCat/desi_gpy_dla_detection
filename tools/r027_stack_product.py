#!/usr/bin/env python
"""r027_stack_product.py — the backing product for the stacked-spectrum figure (FIG-03 / R-027):
absorber-frame stacks of accepted production detections in bins of the reported log N-hat,
built from the real-LOA healpix coadds. A data product (velocity grid, per-bin mean / median /
16-84 % / counts, normalisation definitions, selection contract, provenance), not a PNG edit.

Selection (the catalogue's canonical contract; stated in the output):
  P_DLA > 0.99, DLAFLAG == 0, SNR_REDSIDE > 2, BAL_FLAG == 0 (when present), LYBETA_FLAG == 0
  (when present), absorber inside the sightline's Lyα window (rest 1025-1216 Å, 3600 Å floor,
  3000 km/s collar). Sightlines are drawn from a seeded random subset of healpix files (I/O
  bound); every accepted detection in those files is used, so no bin is cherry-picked.
Bins of N-hat: [19.5,19.7) (below the reporting floor; masked-interval DIAGNOSTIC only),
  [19.7,20.0), [20.0,20.3), [20.3,20.5), [20.5,21.0), [21.0,22.4]. No LLS (< 19.5) line.
Normalisation (both stored):
  local  : flux / median(flux) over 6000 < |v| < 12000 km/s (unmasked, ivar > 0) — a local
           pseudo-continuum that includes the mean forest transmission around the absorber;
  redside: flux / median(flux) over QSO rest 1275-1290 Å (absorption-free; the raw forest
           level is then visible in the wings).
Stack statistics per velocity bin: mean, median, p16, p84 of the normalised flux; N objects.
Velocity grid: |v| <= 15000 km/s in 100 km/s bins (each spectrum's pixels averaged per bin).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
from astropy.io import fits

LYA = 1215.67
C_KMS = 299792.458
NBINS = [19.5, 19.7, 20.0, 20.3, 20.5, 21.0, 22.4]
COLLAR = 3000.0 / C_KMS


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--healpix-root", default="/nfs/turbo/lsa-cavestru/mfho/DESI/loa/healpix/main/dark")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--n-healpix", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--vmax", type=float, default=15000.0)
    ap.add_argument("--dv", type=float, default=100.0)
    ap.add_argument("--archive", default=None, help="LoaArchive h5 (catalog/flux/ivar/mask/wavelength); if given, spectra come from it instead of healpix coadds")
    ap.add_argument("--n-per-bin", type=int, default=4000, help="archive mode: seeded cap of detections per N-hat bin")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import healpy as hp
    d = fits.open(a.catalog)[1].data
    cols = d.columns.names
    sel = (d["P_DLA"] > 0.99) & (d["DLAFLAG"] == 0) & (d["SNR_REDSIDE"] > 2.0)
    if "BAL_FLAG" in cols:
        sel &= (d["BAL_FLAG"] == 0)
    if "LYBETA_FLAG" in cols:
        sel &= (d["LYBETA_FLAG"] == 0)
    zq = d["Z_QSO"].astype(float); za = d["Z_DLA"].astype(float)
    zlo = np.maximum(3600.0 / LYA - 1.0, 1025.0 * (1 + zq) / LYA - 1.0 + COLLAR); zhi = np.minimum(zq - COLLAR, 1216.0 * (1 + zq) / LYA - 1.0 - COLLAR)
    sel &= (za > zlo) & (za < zhi) & (d["NHI"] >= NBINS[0]) & (d["NHI"] < NBINS[-1])
    d = d[sel]
    vedges = np.arange(-a.vmax, a.vmax + a.dv, a.dv); vc = 0.5 * (vedges[:-1] + vedges[1:])
    nb = len(NBINS) - 1
    store = {k: [[] for _ in range(nb)] for k in ("local", "redside")}
    meta = [[] for _ in range(nb)]
    n_missing = 0
    rng = np.random.default_rng(a.seed)

    def accumulate(r, w, fl, iv, mk, red_fn):
        lam0 = LYA * (1 + float(r["Z_DLA"]))
        v = (w / lam0 - 1.0) * C_KMS
        inwin = np.abs(v) <= a.vmax + a.dv
        w, fl, iv, mk, v = w[inwin], fl[inwin], iv[inwin], mk[inwin], v[inwin]
        good = (mk == 0) & np.isfinite(iv) & (iv > 0) & np.isfinite(fl)
        wing = good & (np.abs(v) > 6000.0) & (np.abs(v) < 12000.0)
        if wing.sum() < 20:
            return
        norm_local = float(np.median(fl[wing])); norm_red = red_fn()
        idx = np.digitize(v[good], vedges) - 1
        okb = (idx >= 0) & (idx < vc.size)
        prof = np.full(vc.size, np.nan); cnt = np.bincount(idx[okb], minlength=vc.size); sm = np.bincount(idx[okb], weights=fl[good][okb], minlength=vc.size)
        prof[cnt > 0] = sm[cnt > 0] / cnt[cnt > 0]
        b = int(np.digitize(float(r["NHI"]), NBINS) - 1)
        if norm_local > 0:
            store["local"][b].append(prof / norm_local)
        if np.isfinite(norm_red) and norm_red > 0:
            store["redside"][b].append(prof / norm_red)
        meta[b].append((int(r["TARGETID"]), float(r["Z_DLA"]), float(r["NHI"]), float(r["SNR_REDSIDE"]), float(r["Z_QSO"]), norm_local, norm_red))

    pick = np.zeros(0)
    if a.archive:
        import h5py
        bins_of = np.digitize(d["NHI"].astype(float), NBINS) - 1
        keep = np.zeros(len(d), bool)
        for b in range(nb):
            ii = np.where(bins_of == b)[0]
            keep[rng.choice(ii, size=min(a.n_per_bin, ii.size), replace=False)] = True
        d = d[keep]
        with h5py.File(a.archive, "r") as h:
            cat = h["catalog"][:]; tid2row = {int(t): i for i, t in enumerate(cat["TARGETID"])}
            w_all = h["wavelength"][:].astype(float)
            rows = np.array(sorted({tid2row[int(t)] for t in d["TARGETID"] if int(t) in tid2row}), int)
            n_missing = int(sum(1 for t in d["TARGETID"] if int(t) not in tid2row))
            row2cat = {}
            for r in d:
                if int(r["TARGETID"]) in tid2row:
                    row2cat.setdefault(tid2row[int(r["TARGETID"])], []).append(r)
            for start in range(0, rows.size, 2000):
                sl = rows[start:start + 2000]
                F = h["flux"][sl].astype(float); IV = h["ivar"][sl].astype(float); M = h["mask"][sl]
                for j, row in enumerate(sl):
                    for r in row2cat[row]:
                        zqv = float(r["Z_QSO"])
                        def red_fn(fl=F[j], mk=M[j], zqv=zqv):
                            rest = w_all / (1 + zqv); m = (rest > 1275.0) & (rest < 1290.0) & (mk == 0)
                            return float(np.median(fl[m])) if m.sum() > 5 else np.nan
                        accumulate(r, w_all, F[j], IV[j], M[j], red_fn)
        pick = rows
    hpx = hp.ang2pix(a.nside, d["RA"].astype(float), d["DEC"].astype(float), nest=True, lonlat=True)
    uhp = np.unique(hpx)
    if not a.archive:
        pick = np.sort(rng.choice(uhp, size=min(a.n_healpix, uhp.size), replace=False))
    for h in (pick if not a.archive else []):
        f = os.path.join(a.healpix_root, str(int(h) // 100), str(int(h)), f"coadd-main-dark-{int(h)}.fits")
        if not os.path.exists(f):
            n_missing += 1; continue
        with fits.open(f, memmap=True) as co:
            fm = co["FIBERMAP"].data; tids = np.asarray(fm["TARGETID"], dtype=np.int64)
            cams = {}
            for cam in ("B", "R", "Z"):
                if f"{cam}_WAVELENGTH" in co:
                    cams[cam] = (co[f"{cam}_WAVELENGTH"].data.astype(float), co[f"{cam}_FLUX"].data, co[f"{cam}_IVAR"].data, co[f"{cam}_MASK"].data)
            rows = d[hpx == h]
            t2i = {int(t): i for i, t in enumerate(tids)}
            for r in rows:
                i = t2i.get(int(r["TARGETID"]))
                if i is None:
                    continue
                W, F, IV, M = [], [], [], []
                for cam, (w, fl, iv, mk) in cams.items():
                    W.append(w); F.append(np.asarray(fl[i], float)); IV.append(np.asarray(iv[i], float)); M.append(np.asarray(mk[i]))
                w = np.concatenate(W); fl = np.concatenate(F); iv = np.concatenate(IV); mk = np.concatenate(M)
                zqv = float(r["Z_QSO"])
                def red_fn(fl=fl, mk=mk, w=w, zqv=zqv):
                    rest = w / (1 + zqv); m = (rest > 1275.0) & (rest < 1290.0) & (mk == 0)
                    return float(np.median(fl[m])) if m.sum() > 5 else np.nan
                accumulate(r, w, fl, iv, mk, red_fn)
    out = dict(v_center=vc, v_edges=vedges, nhi_edges=np.array(NBINS))
    summ = dict(bins=[], source=("archive" if a.archive else "healpix coadds"), n_units_used=int(len(pick)), n_missing=n_missing, n_detections_in_subset=int(sum(len(m) for m in meta)))
    for b in range(nb):
        for norm in ("local", "redside"):
            arr = np.array(store[norm][b]) if store[norm][b] else np.zeros((0, vc.size))
            with np.errstate(all="ignore"):
                out[f"{norm}_mean_bin{b}"] = np.nanmean(arr, 0) if arr.size else np.full(vc.size, np.nan)
                out[f"{norm}_median_bin{b}"] = np.nanmedian(arr, 0) if arr.size else np.full(vc.size, np.nan)
                out[f"{norm}_p16_bin{b}"] = np.nanpercentile(arr, 16, axis=0) if arr.size else np.full(vc.size, np.nan)
                out[f"{norm}_p84_bin{b}"] = np.nanpercentile(arr, 84, axis=0) if arr.size else np.full(vc.size, np.nan)
                out[f"{norm}_count_bin{b}"] = np.sum(np.isfinite(arr), 0) if arr.size else np.zeros(vc.size, int)
            out[f"{norm}_n_bin{b}"] = np.array(arr.shape[0])
        m = np.array(meta[b], dtype=float) if meta[b] else np.zeros((0, 7))
        out[f"objects_bin{b}"] = m
        summ["bins"].append(dict(bin=b, nhi=[NBINS[b], NBINS[b + 1]], n_local=int(len(store["local"][b])), n_redside=int(len(store["redside"][b])),
                                  below_reporting_floor=bool(NBINS[b + 1] <= 19.7 + 1e-9),
                                  z_abs_mean=(float(m[:, 1].mean()) if m.size else None), nhat_mean=(float(m[:, 2].mean()) if m.size else None)))
    np.savez(a.out, **out)
    prov = dict(catalog=a.catalog, catalog_sha256=_sha(a.catalog), spectra_source=(a.archive or a.healpix_root), archive_sha256_note=("archive file too large to hash here; identity = path + h5 attrs produced_utc" if a.archive else None), nside=a.nside, nest=True, seed=a.seed, n_per_bin=a.n_per_bin,
                contract="P_DLA>0.99, DLAFLAG==0, SNR_REDSIDE>2, BAL_FLAG==0, LYBETA_FLAG==0, absorber inside the Lya window (rest 1025-1216 A, 3600 A floor, 3000 km/s collar)",
                nhi_bins=NBINS, normalisations=dict(local="median flux over 6000<|v|<12000 km/s (unmasked, ivar>0)", redside="median flux over QSO rest 1275-1290 A (unmasked)"),
                velocity_grid=dict(vmax=a.vmax, dv=a.dv), summary=summ, out=a.out, out_sha256=_sha(a.out),
                generator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip())
    json.dump(prov, open(a.out.replace(".npz", ".provenance.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
