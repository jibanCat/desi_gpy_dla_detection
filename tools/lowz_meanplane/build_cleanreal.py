"""WS2A-ii: build the CLEAN-substrate real injection campaign (checkpoint
10.5 ruling item 2A) — same design/injection/analysis contract as H2-M, on
real Paper-1 sightlines with NO pre-existing accepted detection (tier A1:
zero P_DLA>0.99 rows at any NHI in loa_main_dark_v1) and not in the H2-M
540. Substrate-only change; everything else replicates H2-M exactly.

Outputs under /scratch/.../h2m_ckpt10p5_20260817/cleanreal/:
  h2mc_realized_plan.csv, h2mc_sightlines.csv, h2mc_injected.h5 (+truth csv,
  build summary), qsocat_h2mc.fits, h2mc_hpx_list.txt, h2mc_summary.json
Real TARGETIDs stay on scratch (never in git).

MEAN-S/N REPAIR (2026-09-11, contract C1) -- ADDITIVE, DEFAULT-OFF.
``--snr-source archive-median`` (DEFAULT) is the frozen behaviour, bit for bit.
``--snr-source finder-mean`` replaces the three ``RED_SNR`` reads (selection
threshold at L42/61, the parent S/N at L70, and hence the planner's
stratification) with the canonical finder mean ``SNR_REDSIDE`` joined by int64
TARGETID from ``pack/searched_population.npz``.  The 75 GB archive is never
rewritten; the archive schema has no mean column
(``gpy_dla_detection/loa_archive.py:122-123``).
Under ``finder-mean`` a sightline with no finder mean is INELIGIBLE.
``--run-dir`` re-points the outputs; ``--plan-only`` stops before injection
(used by the CODE-R1 gate).
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import h5py
import fitsio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import campaign_planner as cp
import snr_source as ss
from gpy_dla_detection.inject_absorber import voigt_transmission

H2M = '/scratch/cavestru_root/cavestru0/mfho/h2m_20260817'
RUN = '/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/cleanreal'
ANA = '/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis'
SRC = '/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5'
QSOCAT = '/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits'
REALCAT = ('/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/'
           'dlacat-loa-main-dark-v1.fits')
SEED = 20260817
NUM_LINES = 3

ap = argparse.ArgumentParser()
ap.add_argument('--snr-source', default=ss.ARCHIVE_MEDIAN,
                choices=list(ss.SNR_SOURCE_CHOICES),
                help='S/N variable for the >2 cut AND the cell stratification. '
                     'Default reproduces the frozen (defective) plan exactly.')
ap.add_argument('--run-dir', default=RUN)
ap.add_argument('--h2m-dir', default=H2M,
                help='campaign dir holding qsocat_h2m.fits, whose sightlines '
                     'are excluded from the clean substrate. Must be the H2-M '
                     'campaign built on the SAME S/N plane.')
ap.add_argument('--seed', type=int, default=SEED,
                help='planner RNG seed.  DEFAULT = the declared/frozen seed, '
                     'so the single-realisation behaviour is unchanged.  The '
                     '8-realisation L2 campaign (2026-09-12) passes one of the '
                     'seven predeclared seeds 20260901..20260907; nothing else '
                     'about the protocol varies.')
ap.add_argument('--plan-only', action='store_true')
args = ap.parse_args()
SEED = args.seed
RUN = args.run_dir
H2M = args.h2m_dir
SNR_FIELD = 'RED_SNR' if args.snr_source == ss.ARCHIVE_MEDIAN else 'SNR_MEAN'

os.makedirs(RUN, exist_ok=True)
t0 = time.time()

# ---- parent pool ---------------------------------------------------------
cat = np.load(ANA + '/src_archive_catalog.npy')
tid_arch = cat['TARGETID'].astype(np.int64)
print('archive catalog rows', len(cat), time.time() - t0)

qso = fitsio.read(QSOCAT, ext=1,
                  columns=['TARGETID', 'BI_CIV', 'BAL_PROB', 'BALMASK'])
qtid = qso['TARGETID'].astype(np.int64)
bal_bad = qtid[(qso['BI_CIV'] > 0) | (qso['BAL_PROB'] > 0)
               | (qso['BALMASK'] != 0)]
print('BAL-excluded TIDs', len(bal_bad))

rc = fitsio.read(REALCAT, ext=1, columns=['TARGETID', 'P_DLA', 'Z_DLA'])
rtid = rc['TARGETID'].astype(np.int64)
p99_tids = np.unique(rtid[rc['P_DLA'] > 0.99])
print('TIDs with >=1 P>0.99 row', len(p99_tids))

h2m_tids = fitsio.read(H2M + '/qsocat_h2m.fits', ext=1,
                       columns=['TARGETID'])['TARGETID'].astype(np.int64)

# S/N variable: archive median (frozen default) or the canonical finder mean
snr_used, eligible, snr_meta = ss.resolve_snr(
    tid_arch, cat['RED_SNR'].astype(float), snr_source=args.snr_source)
print('snr_source', json.dumps(snr_meta))

sel = ((cat['Z'] >= 2.1) & (cat['Z'] < 3.79) & eligible & (snr_used > 2.0)
       & ~np.isin(tid_arch, bal_bad) & ~np.isin(tid_arch, p99_tids)
       & ~np.isin(tid_arch, h2m_tids) & (cat['ZWARN'] == 0))
parent_raw = cat[sel]
parent = np.zeros(len(parent_raw), dtype=[('TARGETID', np.int64),
                                          ('Z_QSO', float), ('RED_SNR', float),
                                          ('SNR_MEAN', float),
                                          ('HPXPIXEL', np.int64)])
parent['TARGETID'] = parent_raw['TARGETID'].astype(np.int64)
parent['Z_QSO'] = parent_raw['Z'].astype(float)
parent['RED_SNR'] = parent_raw['RED_SNR'].astype(float)
parent['SNR_MEAN'] = snr_used[sel]
parent['HPXPIXEL'] = parent_raw['HEALPIX'].astype(np.int64)
print('clean parent pool', len(parent), time.time() - t0)

# collision candidates: ANY catalog row at ANY P on clean-pool TIDs
sel_rc = np.isin(rtid, parent['TARGETID'])
cand = {}
for t, z in zip(rtid[sel_rc], rc['Z_DLA'][sel_rc].astype(float)):
    cand.setdefault(int(t), []).append(z)
print('pool TIDs with any-P candidates', len(cand))

# ---- plan ----------------------------------------------------------------
sightlines, plan_rows, dropped = cp.plan_campaign(parent, cand, SEED,
                                                 snr_field=SNR_FIELD)
print('planned', len(sightlines), 'sightlines,', len(plan_rows),
      'injections,', len(dropped), 'dropped')

# On the frozen plane the header/columns are untouched so the recorded CSV is
# bit-reproducible.  On the mean plane two provenance columns are appended:
# 'RED_SNR' always carries the S/N the stratification actually used (the FROZEN
# reducers read that key), 'SNR_STAT' names the statistic, and
# 'RED_SNR_ARCHIVE_MEDIAN' keeps the superseded value for the record.
med_by_tid = dict(zip(parent['TARGETID'].tolist(), parent['RED_SNR'].tolist()))
if args.snr_source == ss.ARCHIVE_MEDIAN:
    sl_fields = ['TARGETID', 'Z_QSO', 'RED_SNR', 'HPXPIXEL', 'cell', 'n_inj']
    sl_rows = sightlines
else:
    sl_fields = ['TARGETID', 'Z_QSO', 'RED_SNR', 'HPXPIXEL', 'cell', 'n_inj',
                 'SNR_STAT', 'RED_SNR_ARCHIVE_MEDIAN']
    sl_rows = [dict(r, SNR_STAT='finder_mean',
                    RED_SNR_ARCHIVE_MEDIAN=med_by_tid[int(r['TARGETID'])])
               for r in sightlines]
with open(RUN + '/h2mc_sightlines.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, sl_fields)
    w.writeheader()
    [w.writerow(s) for s in sl_rows]
with open(RUN + '/h2mc_realized_plan.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['TARGETID', 'inj_idx', 'cell', 'Z_QSO', 'HPXPIXEL',
                           'z_inj', 'logN', 'z_segment', 'attempts'])
    w.writeheader()
    [w.writerow(r) for r in plan_rows]

if args.plan_only:
    print('PLAN-ONLY: stopping before injection', time.time() - t0)
    sys.exit(0)

# ---- injected archive ----------------------------------------------------
tids_sel = np.array([s['TARGETID'] for s in sightlines], np.int64)
arch_idx = {int(t): i for i, t in enumerate(tid_arch)}
rows_sorted = sorted(int(arch_idx[int(t)]) for t in tids_sel)
row_of_tid = {int(tid_arch[r]): k for k, r in enumerate(rows_sorted)}
from collections import defaultdict
inj_by_tid = defaultdict(list)
for r in plan_rows:
    inj_by_tid[int(r['TARGETID'])].append((float(r['z_inj']),
                                           float(r['logN'])))

with h5py.File(SRC, 'r') as hs, h5py.File(RUN + '/h2mc_injected.h5', 'w') as ho:
    wave = hs['wavelength'][:]
    n = len(rows_sorted)
    ho.create_dataset('wavelength', data=wave)
    ho.create_dataset('catalog', data=hs['catalog'][:][rows_sorted])
    for name in ['flux', 'ivar', 'mask', 'fwhm_pix']:
        shape = (n, hs[name].shape[1])
        ho.create_dataset(name, shape=shape, dtype=hs[name].dtype)
    for k, r in enumerate(rows_sorted):
        tid = int(tid_arch[r])
        f64 = hs['flux'][r, :].astype(np.float64)
        for z, logN in inj_by_tid[tid]:
            f64 = f64 * voigt_transmission(wave.astype(np.float64),
                                           10.0 ** logN, z,
                                           num_lines=NUM_LINES)
        ho['flux'][k, :] = f64.astype(np.float32)
        for name in ['ivar', 'mask', 'fwhm_pix']:
            ho[name][k, :] = hs[name][r, :]
        if k % 100 == 0:
            print('  row', k, time.time() - t0, flush=True)
    for a, v in hs.attrs.items():
        ho.attrs[a] = v
    ho.attrs['h2_injected'] = 1
    ho.attrs['h2_source_archive'] = SRC
    ho.attrs['ckpt10p5_campaign'] = 'cleanreal_A1_noP99'

# truth manifest
with open(RUN + '/h2mc_injected.h5.truth.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['TARGETID', 'inj_idx', 'cell', 'z_true', 'logN_true',
                'num_lines'])
    for r in plan_rows:
        w.writerow([r['TARGETID'], r['inj_idx'], r['cell'], r['z_inj'],
                    r['logN'], NUM_LINES])

# qsocat subset + hpx list (row-subset read: the full-table read of the
# 2.7M-row wide catalog gets OOM-killed on the login node)
qt = fitsio.read(QSOCAT, ext=1, columns=['TARGETID'])['TARGETID'].astype(np.int64)
ridx = np.where(np.isin(qt, tids_sel))[0]
sub = fitsio.read(QSOCAT, ext=1, rows=ridx)
fitsio.write(RUN + '/qsocat_h2mc.fits', sub, clobber=True)
del qt, sub
hpx = sorted({s['HPXPIXEL'] for s in sightlines})
with open(RUN + '/h2mc_hpx_list.txt', 'w') as f:
    f.write('\n'.join(str(h) for h in hpx) + '\n')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()

summary = dict(
    campaign='ckpt10p5 cleanreal (A1: zero P>0.99 rows, not in H2-M 540)',
    snr_source=args.snr_source, snr_field=SNR_FIELD, snr_meta=snr_meta,
    seed=SEED, n_sightlines=len(sightlines), n_injections=len(plan_rows),
    n_dropped=len(dropped), dropped=dropped, n_hpx=len(hpx),
    parent_pool=int(len(parent)),
    config=dict(salt='h2mc1', logN_grid=list(cp.LOGN_GRID),
                logN_weights=list(cp.LOGN_W), sl_per_cell=cp.SL_PER_CELL,
                dbl_per_cell=cp.DBL_PER_CELL, z_inj_cap=list(cp.Z_CAP),
                collision_kms=cp.COLLISION_KMS, collar_kms=cp.COLLAR_KMS,
                window='lya_only 1025-1216', num_lines=NUM_LINES,
                sibling_avoidance=False),
    source_archive=SRC, source_archive_note='sha inherited from attrs',
    injected_archive_sha256=sha(RUN + '/h2mc_injected.h5'),
    plan_sha256=sha(RUN + '/h2mc_realized_plan.csv'),
    truth_sha256=sha(RUN + '/h2mc_injected.h5.truth.csv'),
)
json.dump(summary, open(RUN + '/h2mc_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ('dropped',)}, indent=1))
print('DONE', time.time() - t0)
