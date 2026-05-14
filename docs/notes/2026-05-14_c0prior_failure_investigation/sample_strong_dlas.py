"""Sample 10 strong 2lpt loa-124 mock-0 DLAs with their healpix-spectra paths.

Picks DLAs at logNHI in [20.6, 21.5] with truth SNR > 3, and confirms the
parent spectra-16-<pix>.fits exists on /nfs/turbo. Output JSON is consumed
by the multi-target inference script.
"""
from astropy.io import fits
import numpy as np
import os
import json

MOCK_BASE = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124'
TRUTH = f'{MOCK_BASE}/hcd_truth_cat.fits'
ZCAT = f'{MOCK_BASE}/zcat.fits'
OUT = '/home/mfho/desi_gpy_dla_detection/docs/notes/2026-05-14_c0prior_failure_investigation/sampled_dlas.json'

with fits.open(ZCAT) as zcat_h:
    zcat = zcat_h[1].data
    print('zcat cols:', zcat.columns.names[:30])

with fits.open(TRUTH) as th:
    truth = th[1].data
    strong = truth[(truth['NHI'] >= 20.6) & (truth['NHI'] <= 21.5) & (truth['SNR'] > 3.0)]
    print(f'strong DLA candidates (NHI 20.6-21.5, SNR>3): {len(strong)}')

zcat_idx = {int(row['TARGETID']): i for i, row in enumerate(zcat)}
print(f'zcat has {len(zcat_idx)} TIDs')

# zcat has no HEALPIX column; compute from RA/DEC (nside=16 nested per DESI mock).
import healpy as healpy_mod

np.random.seed(13)
candidates = []
order = np.random.permutation(len(strong))
for i in order:
    row = strong[i]
    tid = int(row['TARGETID'])
    if tid not in zcat_idx:
        continue
    z_row = zcat[zcat_idx[tid]]
    ra = float(z_row['TARGET_RA']); dec = float(z_row['TARGET_DEC'])
    theta = np.deg2rad(90.0 - dec); phi = np.deg2rad(ra)
    hp = int(healpy_mod.ang2pix(16, theta, phi, nest=True))
    group = hp // 100
    spec = os.path.join(MOCK_BASE, f'spectra-16/{group}/{hp}/spectra-16-{hp}.fits')
    if not os.path.exists(spec):
        continue
    z_qso = float(z_row['Z'])
    candidates.append(dict(tid=tid, nhi=float(row['NHI']), z_dla=float(row['Z']),
                           snr=float(row['SNR']), z_qso=z_qso, hp=hp, spec=spec))
    if len(candidates) >= 10:
        break

for c in candidates:
    print(f"  TID={c['tid']:>11d}  NHI={c['nhi']:.3f}  z_DLA={c['z_dla']:.4f}  z_QSO={c['z_qso']:.4f}  SNR={c['snr']:.2f}  HP={c['hp']}")

json.dump(candidates, open(OUT, 'w'), indent=2)
print(f'wrote {OUT}')
