"""For 4 high-BI_CIV TIDs from loa-124's bal_cat:
  - Plot the loa-0 spectrum (uncontaminated realization, no BAL injection)
  - Plot the loa-124 spectrum (BAL-injected realization)
  - Side by side, same y-axis, with CIV/SiIV/Lyα marked

Goal: visually confirm that loa-0 is truly BAL-free even at TIDs that
loa-124 marks as BAL.
"""
import sys
sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
import numpy as np
import h5py
import fitsio
import matplotlib.pyplot as plt
import desispec.io
import warnings
import os

BAL_CAT = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124/bal_cat.fits"
LOA0_BASE = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-0"
LOA124_BASE = "/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"
ZCAT_LOA0 = f"{LOA0_BASE}/zcat.fits"
ZCAT_LOA124 = f"{LOA124_BASE}/zcat.fits"

# Pick 4 high-BI_CIV TIDs (vary z to test redshift coverage)
bal = fitsio.read(BAL_CAT, columns=["TARGETID", "Z", "BI_CIV"])
mask = np.isfinite(bal["BI_CIV"]) & (bal["BI_CIV"] > 0)
bal = bal[mask]
# pick 4 by z bins
picks = []
for zlo, zhi in [(2.0, 2.4), (2.4, 2.8), (2.8, 3.2), (3.2, 3.8)]:
    in_bin = (bal["Z"] >= zlo) & (bal["Z"] < zhi)
    candidates = bal[in_bin]
    if len(candidates) > 0:
        # pick highest BI_CIV in this bin
        idx = int(np.argmax(candidates["BI_CIV"]))
        picks.append(candidates[idx])

print(f"Picked {len(picks)} BAL TIDs:")
for p in picks:
    print(f"  TID {p['TARGETID']}  z={p['Z']:.3f}  BI_CIV={p['BI_CIV']:.0f} km/s")


def find_and_load(tid, base, healpix_for_tid):
    """Reuse examples.smoke_one_spectrum.load_one_desi_spectrum which
    already handles the mock camera-coadd + truth-file resolution
    fallback. Don't reinvent IO."""
    from examples.smoke_one_spectrum import load_one_desi_spectrum
    hpx = healpix_for_tid
    spec_path = f"{base}/spectra-16/{hpx // 100}/{hpx}/spectra-16-{hpx}.fits"
    if not os.path.exists(spec_path):
        return None
    try:
        wave, flux, nv, mask = load_one_desi_spectrum(spec_path, int(tid))
        return dict(wave=wave, flux=flux, nv=nv, mask=mask)
    except Exception as e:
        print(f"    [warn] tid {tid} ({spec_path}): {e}")
        return None


def healpix_for_tid_via_zcat(tid, zcat_path, nside=16):
    """Compute NSIDE=16 HPXPIXEL for TID from zcat's RA/DEC.
    spectra-16-N.fits files are organized at NSIDE=16 (DESI's
    'spectra-16' = nside-16 healpix grouping) — confirmed by
    listing spectra-16/ subdirs (max hpx ≈ 998 = 12*16²-1)."""
    cols = fitsio.FITS(zcat_path)[1].get_colnames()
    if "HPXPIXEL" in cols:
        d = fitsio.read(zcat_path, columns=["TARGETID", "HPXPIXEL"])
        idx = np.where(d["TARGETID"] == int(tid))[0]
        if len(idx) == 0:
            return None
        return int(d["HPXPIXEL"][idx[0]])
    if "TARGET_RA" in cols and "TARGET_DEC" in cols:
        d = fitsio.read(zcat_path, columns=["TARGETID", "TARGET_RA", "TARGET_DEC"])
        idx = np.where(d["TARGETID"] == int(tid))[0]
        if len(idx) == 0:
            return None
        import healpy as hp
        ra = float(d["TARGET_RA"][idx[0]])
        dec = float(d["TARGET_DEC"][idx[0]])
        # DESI standard for spectra-16 is NESTED ordering at nside=16.
        return int(hp.ang2pix(nside, ra, dec, lonlat=True, nest=True))
    return None


print(f"\n=== loading + plotting ===")
n_picks = len(picks)
fig, axes = plt.subplots(n_picks, 2, figsize=(15, 2.5 * n_picks),
                         sharex=False, gridspec_kw=dict(hspace=0.30, wspace=0.05))
if n_picks == 1:
    axes = axes[None, :]

LINES = {"Lyα": 1215.67, "NV": 1240.0, "SiIV": 1394.0, "CIV": 1548.2}

for row, pick in enumerate(picks):
    tid = int(pick["TARGETID"])
    z_qso = float(pick["Z"])
    bi = float(pick["BI_CIV"])

    # find healpix for this TID in each zcat
    hpx_loa0 = healpix_for_tid_via_zcat(tid, ZCAT_LOA0)
    hpx_loa124 = healpix_for_tid_via_zcat(tid, ZCAT_LOA124)
    print(f"  TID {tid}: hpx loa-0={hpx_loa0} loa-124={hpx_loa124}")

    spec_loa0 = find_and_load(tid, LOA0_BASE, hpx_loa0) if hpx_loa0 is not None else None
    spec_loa124 = find_and_load(tid, LOA124_BASE, hpx_loa124) if hpx_loa124 is not None else None

    for col, (label, spec) in enumerate([("loa-0 (uncontaminated)", spec_loa0),
                                          ("loa-124 (BAL-injected)", spec_loa124)]):
        ax = axes[row, col]
        if spec is None:
            ax.text(0.5, 0.5, "spectrum not loaded", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="0.5")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        rest = spec["wave"] / (1.0 + z_qso)
        ax.plot(rest, spec["flux"], color="0.3", lw=0.4)
        ax.axhline(0, color="0.7", lw=0.4, ls="--")
        # mark lines + BAL outflow zone
        for lname, lwave in LINES.items():
            if rest[0] < lwave < rest[-1]:
                ax.axvline(lwave, color="C2", ls=":", lw=0.6, alpha=0.6)
                ax.text(lwave, ax.get_ylim()[1] * 0.92, lname,
                        fontsize=7, ha="center", color="C2",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="C2", alpha=0.7))
        # BAL outflow zone (CIV blueward by 0–30,000 km/s)
        civ = 1548.2
        bal_blue = civ * (1 - 30000.0 / 299792.458)
        ax.axvspan(bal_blue, civ, color="C3", alpha=0.10)
        ax.set_xlim(900, 1700)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel(f"flux\nTID {tid}\nz={z_qso:.3f}\nBI_CIV={bi:.0f}",
                          fontsize=8)
        ax.set_title(label, fontsize=9)
    if row == n_picks - 1:
        axes[row, 0].set_xlabel("rest wavelength [Å]")
        axes[row, 1].set_xlabel("rest wavelength [Å]")

fig.suptitle("Same TID in 2lpt loa-0 (uncontaminated) vs loa-124 (BAL-injected)\n"
             "Pink shading = typical BAL outflow zone (0-30,000 km/s blueward of CIV)",
             fontsize=10, y=0.998)
fig.tight_layout()
out = "/home/mfho/desi_gpy_dla_detection/docs/notes/2026-05-03_bal_loa0_vs_loa124.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nwrote {out}")
