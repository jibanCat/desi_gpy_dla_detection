#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""High-z real-LOA candidate review package (PI 2026-08-12 items 16-17).

Unfiltered: EVERY clean candidate at z_DLA >= 4.0 gets a diagnostic page;
manifest additionally covers 3.8 <= z_DLA < 4.0 and z>4 rows failing the
clean cuts (labeled, no pages unless --all-pages). No visual preselection.

Outputs (PRIVATE notes repo):
  notes/figures/highz_review_2026-08-12/
    manifest_highz.csv              all rows 3.8+, with flags + clean bit
    rank_by_{z,nhi,p,omega}.csv     ranked views (clean z>4 set)
    omega_dominance.txt             cumulative shares + leave-one-out
    pages/hz_<TARGETID>_<zdla>.png  one page per clean z>4 candidate
    contact_sheet_<n>.png           Lya-cutout thumbnails, 8x6 per sheet
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import wofz

# Historical GL defaults; override via env or CLI (--cddf-cat/--qso-cat/
# --hpx-root/--archive). Missing inputs fail loudly — no silent fallback.
CDDF_CAT = os.environ.get(
    "HZ_CDDF_CAT",
    "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
    "loa_cddf_main_dark_v1/dlacat-loa-cddf-main-dark-v1.fits")
QSO_CAT = os.environ.get(
    "HZ_QSO_CAT",
    "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/"
    "QSO_cat_loa_main_dark_healpix_v2-altbal.fits")
HPX_ROOT = os.environ.get(
    "HZ_HPX_ROOT", "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/healpix/main/dark")
OUTDIR = os.environ.get("HZ_REVIEW_OUT", "highz_review_out")

LYA, LYB, LYG, LYD, LYLIM = 1215.67, 1025.7223, 972.5368, 949.7431, 911.76
LY_F = {"Lya": (LYA, 0.41641, 6.2648e8), "Lyb": (LYB, 0.079142, 1.8971e8),
        "Lyg": (LYG, 0.029006, 8.1272e7)}
METALS = [("SiII 1260", 1260.4221), ("OI 1302", 1302.1685),
          ("SiII 1304", 1304.3702), ("CII 1334", 1334.5323),
          ("SiIV 1393", 1393.7546), ("SiIV 1402", 1402.7697),
          ("SiII 1526", 1526.7070), ("CIV 1548", 1548.2049),
          ("CIV 1550", 1550.7785), ("FeII 1608", 1608.4509),
          ("AlII 1670", 1670.7886)]
CKMS = 299792.458


def voigt_tau(lam_obs, z, logN, line, b_kms=15.0):
    lam0, f, gam = LY_F[line]
    N = 10 ** logN
    lam_r = lam_obs / (1 + z)
    nu = CKMS * 1e13 / lam_r          # crude; work in velocity instead
    v = CKMS * (lam_r - lam0) / lam0
    b = b_kms
    a = gam * lam0 * 1e-13 / (4 * np.pi * b * 1e5 / 1e-8) / 1e5
    # standard: a = Gamma * lam0(A->cm) / (4 pi b)
    a = gam * (lam0 * 1e-8) / (4 * np.pi * (b * 1e5))
    x = v / b
    H = np.real(wofz(x + 1j * a))
    # tau0 = 1.497e-15 * N * f * lam0(A) / b(km/s)
    tau0 = 1.497e-15 * N * f * lam0 / b
    return tau0 * H


def read_coadd_rows(pix, tids, hpx_root=None):
    path = os.path.join(hpx_root if hpx_root is not None else HPX_ROOT,
                        str(pix // 100), str(pix),
                        f"coadd-main-dark-{pix}.fits")
    out = {}
    with fits.open(path, memmap=True) as h:
        fm = h["FIBERMAP"].data
        idx = {int(t): i for i, t in enumerate(fm["TARGETID"])}
        for t in tids:
            if int(t) not in idx:
                continue
            i = idx[int(t)]
            spec = {}
            for cam in ("B", "R", "Z"):
                w = h[f"{cam}_WAVELENGTH"].data
                fl = h[f"{cam}_FLUX"].data[i]
                iv = h[f"{cam}_IVAR"].data[i]
                spec[cam] = (w, fl, iv)
            out[int(t)] = spec
    return out


def read_archive_rows(archive, tids):
    """Spectrum source from a LoaArchive HDF5 (I/O substitute for
    read_coadd_rows when the raw healpix coadds are not local).

    Returns the same {tid: {cam: (wave, flux, ivar)}} shape; the archive's
    single stitched grid is placed under "B" with empty "R"/"Z" so
    stitch()/page() are unchanged. ivar is used as stored (the coadd path
    reads *_IVAR only and never applies MASK; mirrored here).
    """
    out = {}
    empty = (np.array([]), np.array([]), np.array([]))
    for t in tids:
        try:
            s = archive.get_spectrum(int(t))
        except KeyError:
            continue
        out[int(t)] = {"B": (s.wavelength, s.flux, s.ivar),
                       "R": empty, "Z": empty}
    return out


def stitch(spec):
    w = np.concatenate([spec[c][0] for c in ("B", "R", "Z")])
    f = np.concatenate([spec[c][1] for c in ("B", "R", "Z")])
    iv = np.concatenate([spec[c][2] for c in ("B", "R", "Z")])
    o = np.argsort(w)
    return w[o], f[o], iv[o]


def smooth(y, n=7):
    k = np.ones(n) / n
    return np.convolve(np.nan_to_num(y), k, mode="same")


def local_continuum(w, f, iv, lam_c, dv_in=2500., dv_out=6000.):
    v = CKMS * (w - lam_c) / lam_c
    m = (np.abs(v) > dv_in) & (np.abs(v) < dv_out) & (iv > 0)
    if m.sum() < 10:
        return np.nan
    return float(np.median(f[m]))


def page(row, spec, outpath):
    w, f, iv = stitch(spec)
    ns = np.where(iv > 0, 1 / np.sqrt(np.maximum(iv, 1e-30)), np.nan)
    zq, zd, nhi = row["Z_QSO"], row["Z_DLA"], row["NHI"]
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(5, 4, height_ratios=[1.6, 1, 1, 1, 1],
                          hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    m = (w > 3600) & (w < min((1 + zq) * 1400, 9800))
    ax.plot(w[m], smooth(f[m]), lw=0.5, color="k")
    ax.plot(w[m], ns[m], lw=0.4, color="tab:red", alpha=0.6)
    for lam0, lab, c in ((LYA, "Lyα_em", "tab:blue"),
                         (LYB, "Lyβ_em", "tab:cyan")):
        ax.axvline((1 + zq) * lam0, color=c, ls=":", lw=1)
    for lam0, lab in (("Lya", "Lyα_abs"), ("Lyb", "Lyβ_abs"),
                      ("Lyg", "Lyγ_abs")):
        ax.axvline((1 + zd) * LY_F[lam0][0], color="tab:orange", ls="--",
                   lw=1)
    ax.axvline((1 + zd) * LYLIM, color="tab:purple", ls="-.", lw=1)
    lo = np.nanpercentile(f[m], 1)
    hi = np.nanpercentile(smooth(f[m]), 99.5) * 1.4
    ax.set_ylim(min(lo, -0.5), hi)
    ax.set_title(
        f"TARGETID {row['TARGETID']}  RA {row['RA']:.4f} DEC {row['DEC']:.4f}"
        f"  z_qso {zq:.3f}  |  z_DLA {zd:.4f}  logN {nhi:.2f}"
        f"±{row['NHI_ERR']:.2f}  P_DLA {row['P_DLA']:.4f}  "
        f"SNR {row['SNR_REDSIDE']:.1f}  DLAFLAG {row['DLAFLAG']}"
        f"  BAL {bool(row['BAL_FLAG'])}  [loa_cddf_main_dark_v1]",
        fontsize=9)
    ax.set_xlabel("obs wavelength [Å]")

    # Lyman-series cutouts with Voigt overlay
    for j, line in enumerate(("Lya", "Lyb", "Lyg")):
        lam_c = (1 + zd) * LY_F[line][0]
        axz = fig.add_subplot(gs[1, j])
        v = CKMS * (w - lam_c) / lam_c
        mm = (np.abs(v) < 6000) & (w > 3600)
        if mm.sum() > 5:
            axz.plot(v[mm], f[mm], lw=0.6, color="k", drawstyle="steps-mid")
            axz.plot(v[mm], ns[mm], lw=0.4, color="tab:red", alpha=0.6)
            cont = local_continuum(w, f, iv, lam_c)
            if np.isfinite(cont) and cont > 0:
                tau = voigt_tau(w[mm], zd, nhi, line)
                axz.plot(v[mm], cont * np.exp(-tau), color="tab:orange",
                         lw=1.2, alpha=0.9)
                axz.axhline(cont, color="tab:green", lw=0.6, ls=":")
            axz.axvline(0, color="tab:orange", ls="--", lw=0.7)
            axz.set_ylim(min(-0.5, np.nanpercentile(f[mm], 2)),
                         np.nanpercentile(f[mm], 99) * 1.3)
        else:
            axz.text(0.5, 0.5, "not covered", ha="center",
                     transform=axz.transAxes)
        axz.set_title(f"{line} @ {lam_c:.0f} Å", fontsize=8)
    axl = fig.add_subplot(gs[1, 3])
    lam_ll = (1 + zd) * LYLIM
    mll = (w > lam_ll - 120) & (w < lam_ll + 200)
    if mll.sum() > 5 and lam_ll > 3600:
        axl.plot(w[mll], smooth(f[mll], 5), lw=0.6, color="k")
        axl.axvline(lam_ll, color="tab:purple", ls="-.", lw=1)
    else:
        axl.text(0.5, 0.5, "LL not covered", ha="center",
                 transform=axl.transAxes)
    axl.set_title(f"Ly-limit @ {lam_ll:.0f} Å", fontsize=8)

    # metal stamps
    for j, (lab, lam0) in enumerate(METALS[:12]):
        r, cix = 2 + j // 4, j % 4
        axm = fig.add_subplot(gs[r, cix])
        lam_c = (1 + zd) * lam0
        v = CKMS * (w - lam_c) / lam_c
        mm = (np.abs(v) < 1500) & (w > 3600) & (w < 9824)
        if lam_c < 3600 or lam_c > 9824:
            axm.text(0.5, 0.5, "outside coverage", ha="center",
                     fontsize=8, transform=axm.transAxes)
        elif (1 + zq) * LYA > lam_c:
            axm.text(0.5, 0.85, "in forest", ha="center", fontsize=7,
                     color="tab:red", transform=axm.transAxes)
            if mm.sum() > 5:
                axm.plot(v[mm], f[mm], lw=0.6, color="k",
                         drawstyle="steps-mid")
                axm.axvline(0, color="tab:orange", ls="--", lw=0.7)
        elif mm.sum() > 5:
            axm.plot(v[mm], f[mm], lw=0.6, color="k", drawstyle="steps-mid")
            axm.plot(v[mm], ns[mm], lw=0.4, color="tab:red", alpha=0.5)
            axm.axvline(0, color="tab:orange", ls="--", lw=0.7)
        axm.set_title(f"{lab} @ {lam_c:.0f} Å", fontsize=7)
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    # thumbnail data: Lya cutout
    lam_c = (1 + zd) * LYA
    v = CKMS * (w - lam_c) / lam_c
    mm = np.abs(v) < 5000
    return (v[mm], f[mm])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cddf-cat", default=CDDF_CAT,
                    help="DLA/CDDF catalog FITS (candidate selection source)")
    ap.add_argument("--qso-cat", default=QSO_CAT,
                    help="QSO parent catalog FITS (TARGETID->HPXPIXEL map)")
    ap.add_argument("--hpx-root", default=HPX_ROOT,
                    help="healpix coadd tree root (spectrum source)")
    ap.add_argument("--archive", default=os.environ.get("HZ_ARCHIVE") or None,
                    help="LoaArchive HDF5 spectrum source; replaces coadd "
                         "reads under --hpx-root (I/O substitution only)")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip candidates whose page PNG already exists "
                         "in OUTDIR/pages")
    args = ap.parse_args()

    # fail loudly on missing inputs — never fall back to an unintended path
    for label, p in (("CDDF cat", args.cddf_cat), ("QSO cat", args.qso_cat)):
        if not os.path.isfile(p):
            sys.exit(f"FATAL: {label} not found: {p}")
    if args.archive is not None:
        if not os.path.isfile(args.archive):
            sys.exit(f"FATAL: archive not found: {args.archive}")
    elif not os.path.isdir(args.hpx_root):
        sys.exit(f"FATAL: healpix root not found: {args.hpx_root} "
                 "(pass --hpx-root or --archive)")

    os.makedirs(os.path.join(OUTDIR, "pages"), exist_ok=True)

    c = fits.open(args.cddf_cat)[1].data
    q = fits.open(args.qso_cat)[1].data
    hpx = {int(t): int(p) for t, p in zip(q["TARGETID"], q["HPXPIXEL"])}

    m_all = c["Z_DLA"] >= 3.8
    clean = ((c["DLAFLAG"] == 0) & (c["P_DLA"] > 0.99)
             & (c["SNR_REDSIDE"] > 2.0))
    rows = c[m_all]
    cl = clean[m_all]
    # manifest
    import csv
    with open(os.path.join(OUTDIR, "manifest_highz.csv"), "w",
              newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["TARGETID", "RA", "DEC", "Z_QSO", "Z_DLA", "NHI",
                       "NHI_ERR", "P_DLA", "SNR_REDSIDE", "DLAFLAG",
                       "BAL_FLAG", "LYBETA_FLAG", "clean", "page",
                       "HPXPIXEL"])
        for r, is_cl in zip(rows, cl):
            has_page = bool(is_cl and r["Z_DLA"] >= 4.0)
            wcsv.writerow([r["TARGETID"], f"{r['RA']:.5f}",
                           f"{r['DEC']:.5f}", f"{r['Z_QSO']:.4f}",
                           f"{r['Z_DLA']:.4f}", f"{r['NHI']:.3f}",
                           f"{r['NHI_ERR']:.3f}", f"{r['P_DLA']:.5f}",
                           f"{r['SNR_REDSIDE']:.2f}", r["DLAFLAG"],
                           int(bool(r["BAL_FLAG"])),
                           int(bool(r["LYBETA_FLAG"])), int(bool(is_cl)),
                           int(has_page),
                           hpx.get(int(r["TARGETID"]), -1)])

    sel = m_all & clean & (c["Z_DLA"] >= 4.0)
    cand = c[sel]
    print(f"clean z>=4 candidates with pages: {len(cand)}")
    if args.limit:
        cand = cand[:args.limit]

    # rankings + omega dominance (clean z>4 set, band-limited weight 10^NHI)
    wgt = 10.0 ** (cand["NHI"] - 20.3)
    order_om = np.argsort(wgt)[::-1]
    share = wgt[order_om] / wgt.sum()
    with open(os.path.join(OUTDIR, "omega_dominance.txt"), "w") as fh:
        fh.write("clean z>=4 set, band-limited Omega weight 10^(NHI-20.3)\n")
        cum = 0.0
        for rank, i in enumerate(order_om[:25]):
            cum += share[rank]
            r = cand[i]
            loo = wgt.sum() - wgt[i]
            fh.write(f"#{rank+1} TID {r['TARGETID']} z {r['Z_DLA']:.3f} "
                     f"logN {r['NHI']:.2f} P {r['P_DLA']:.3f} "
                     f"share {share[rank]:.3%} cum {cum:.3%} "
                     f"LOO Omega drop {wgt[i]/wgt.sum():.3%}\n")
        fh.write(f"\nn objects for 50% of Omega: "
                 f"{int(np.searchsorted(np.cumsum(share), 0.5) + 1)}\n")
        fh.write(f"n objects for 90%: "
                 f"{int(np.searchsorted(np.cumsum(share), 0.9) + 1)}\n")
        fh.write("dN/dX dominance: uniform (each object = 1 count); "
                 "no single-object dominance possible by construction.\n")
    for key, arr in (("z", cand["Z_DLA"]), ("nhi", cand["NHI"]),
                     ("p", cand["P_DLA"]),
                     ("omega", wgt)):
        o = np.argsort(arr)[::-1]
        with open(os.path.join(OUTDIR, f"rank_by_{key}.csv"), "w",
                  newline="") as fh:
            wcsv = csv.writer(fh)
            wcsv.writerow(["rank", "TARGETID", "Z_DLA", "NHI", "P_DLA",
                           "SNR_REDSIDE", "page"])
            for rank, i in enumerate(o):
                r = cand[i]
                wcsv.writerow([rank + 1, r["TARGETID"],
                               f"{r['Z_DLA']:.4f}", f"{r['NHI']:.3f}",
                               f"{r['P_DLA']:.5f}",
                               f"{r['SNR_REDSIDE']:.2f}",
                               f"pages/hz_{r['TARGETID']}_"
                               f"{r['Z_DLA']:.3f}.png"])

    # group by healpix, generate pages
    archive = None
    if args.archive is not None:
        try:
            from gpy_dla_detection.loa_archive import LoaArchive
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            from gpy_dla_detection.loa_archive import LoaArchive
        archive = LoaArchive(args.archive)
        archive.open()
    bypix = defaultdict(list)
    for i, r in enumerate(cand):
        p = hpx.get(int(r["TARGETID"]), -1)
        bypix[p].append(i)
    thumbs = []
    ndone = 0
    for p, idxs in sorted(bypix.items()):
        if p < 0:
            continue
        tids = [int(cand[i]["TARGETID"]) for i in idxs]
        try:
            if archive is not None:
                specs = read_archive_rows(archive, tids)
            else:
                specs = read_coadd_rows(p, tids, args.hpx_root)
        except Exception as e:
            print(f"[pix {p}] read failed: {e}")
            continue
        for i in idxs:
            r = cand[i]
            t = int(r["TARGETID"])
            if t not in specs:
                print(f"[pix {p}] TID {t} not in "
                      f"{'archive' if archive is not None else 'coadd'}")
                continue
            out = os.path.join(OUTDIR, "pages",
                               f"hz_{t}_{r['Z_DLA']:.3f}.png")
            if args.only_missing and os.path.exists(out):
                continue
            try:
                th = page(r, specs[t], out)
                thumbs.append((r, th))
                ndone += 1
                if ndone % 25 == 0:
                    print(f"  {ndone} pages done")
            except Exception as e:
                print(f"[pix {p}] TID {t} page failed: {e}")
    if archive is not None:
        archive.close()

    # contact sheets: 48 per sheet, ordered by z_DLA desc
    thumbs.sort(key=lambda x: -x[0]["Z_DLA"])
    per = 48
    for s in range(0, len(thumbs), per):
        chunk = thumbs[s:s + per]
        fig, axes = plt.subplots(6, 8, figsize=(22, 15))
        for ax, (r, (v, fl)) in zip(axes.ravel(), chunk):
            ax.plot(v, fl, lw=0.4, color="k")
            ax.axvline(0, color="tab:orange", lw=0.5, ls="--")
            ax.set_title(f"{r['TARGETID']}\nz{r['Z_DLA']:.2f} "
                         f"N{r['NHI']:.1f} P{r['P_DLA']:.2f}", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.suptitle(f"high-z review contact sheet {s//per + 1} "
                     f"(z-ordered, unfiltered)", fontsize=12)
        fig.savefig(os.path.join(OUTDIR,
                                 f"contact_sheet_{s//per + 1}.png"),
                    dpi=90, bbox_inches="tight")
        plt.close(fig)
    print(f"DONE: {ndone} pages, {int(np.ceil(len(thumbs)/per))} sheets")


if __name__ == "__main__":
    main()
