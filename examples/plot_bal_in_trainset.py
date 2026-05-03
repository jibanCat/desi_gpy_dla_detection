"""Verify loa_no_hcd_with_bal_52198070/trainset.h5 actually contains
BAL spectra by cross-referencing with the QSO catalog and plotting
the highest-BI_CIV TIDs in the trainset.
"""
import sys, os
sys.path.insert(0, "/home/mfho/desi_gpy_dla_detection")
import numpy as np
import h5py
import fitsio
import matplotlib.pyplot as plt

TRAINSET = "/nfs/turbo/lsa-cavestru/mfho/DESI/GP_trained/loa_no_hcd_with_bal_52198070/trainset.h5"
QSOCAT = "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v3-altbal.fits"

# 1. Read TIDs in trainset
with h5py.File(TRAINSET, "r") as f:
    tids_train = f["tids"][:]
    print(f"trainset n_spectra: {len(tids_train)}")

# 2. Read QSO catalog
qso = fitsio.read(QSOCAT, columns=["TARGETID", "Z", "BI_CIV", "AI_CIV"])
print(f"QSO catalog rows: {len(qso)}")
# Filter to TIDs present in trainset AND with BI_CIV available
mask_in = np.isin(qso["TARGETID"], tids_train)
qso_in = qso[mask_in]
print(f"QSO rows in trainset: {len(qso_in)}")

# 3. BAL classification
bi = qso_in["BI_CIV"]
finite = np.isfinite(bi)
n_bal = int((finite & (bi > 0)).sum())
n_nonbal = int((finite & (bi == 0)).sum())
n_unknown = int((~finite).sum())
print(f"\nBAL fraction in trainset (BI_CIV > 0): {n_bal} of {len(qso_in)} ({100*n_bal/len(qso_in):.1f}%)")
print(f"  non-BAL (BI_CIV == 0):  {n_nonbal} ({100*n_nonbal/len(qso_in):.1f}%)")
print(f"  no measurement (NaN):   {n_unknown} ({100*n_unknown/len(qso_in):.1f}%)")
if n_bal > 0:
    bi_pos = bi[finite & (bi > 0)]
    print(f"  BI_CIV among BALs: median={np.median(bi_pos):.0f}  p90={np.percentile(bi_pos, 90):.0f}  max={bi_pos.max():.0f}  km/s")

# 4. Pick the 6 highest-BI_CIV TIDs that are in the trainset
qso_bal = qso_in[finite & (bi > 0)]
top_idx = np.argsort(qso_bal["BI_CIV"])[::-1][:6]
top_tids = qso_bal["TARGETID"][top_idx]
top_bi = qso_bal["BI_CIV"][top_idx]
top_z = qso_bal["Z"][top_idx]
print(f"\nTop 6 BAL TIDs in trainset:")
for t, b, z in zip(top_tids, top_bi, top_z):
    print(f"  TID {t}  BI_CIV={b:.0f} km/s  z={z:.3f}")

# 5. Read those spectra from trainset
with h5py.File(TRAINSET, "r") as f:
    rw = f["rest_wavelengths"][0]   # shared rest grid
    fluxes_all = f["fluxes"]        # (N, n_pix) — DON'T load all
    # find indices of top_tids in tids_train
    indices = []
    for tid in top_tids:
        i = int(np.where(tids_train == tid)[0][0])
        indices.append(i)
    indices = sorted(indices)
    fluxes_top = fluxes_all[indices]   # only read those rows
    rest_w = rw

# 6. Plot
fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
fig.suptitle(
    "BAL spectra in loa_no_hcd_with_bal_52198070/trainset.h5\n"
    "(top 6 by BI_CIV, AS-LOADED from trainset.h5 — pre-normalize, pre-deforest)",
    fontsize=10, y=0.995,
)
# Mark canonical BAL/emission lines (rest frame)
LINES = {"Lyα": 1215.67, "NV": 1240.0, "SiIV": 1394.0, "CIV": 1548.2, "CIII]": 1908.7}
for ax, tid, bi_val, z_val, idx in zip(axes, top_tids, top_bi, top_z,
                                       [int(np.where(tids_train == t)[0][0]) for t in top_tids]):
    flux_row = fluxes_top[sorted([int(np.where(tids_train == t)[0][0]) for t in top_tids]).index(idx)]
    ax.plot(rest_w, flux_row, color="0.3", lw=0.5, label=f"TID {tid}")
    ax.axhline(0, color="0.7", lw=0.4, ls="--")
    for lname, lwave in LINES.items():
        if rest_w[0] < lwave < rest_w[-1]:
            ax.axvline(lwave, color="C2", ls=":", lw=0.5, alpha=0.6)
            ax.text(lwave, ax.get_ylim()[1] * 0.92, lname,
                    fontsize=7, ha="center", color="C2",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="C2", alpha=0.7))
    # BAL-trough marker: blueward of CIV by typical 0-30,000 km/s outflow
    civ = 1548.2
    bal_blueward_max = civ * (1 - 30000.0 / 299792.458)  # 30k km/s blueward
    ax.axvspan(bal_blueward_max, civ, color="C3", alpha=0.08,
               label="typical BAL outflow blueward of CIV")
    ax.set_ylabel(f"flux\n(BI={bi_val:.0f} km/s,\nz={z_val:.2f})", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(rest_w[0], rest_w[-1])
axes[-1].set_xlabel("rest wavelength [Å]")
fig.tight_layout()
out = "/home/mfho/desi_gpy_dla_detection/docs/notes/2026-05-03_bal_spectra_in_trainset.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nwrote {out}")
