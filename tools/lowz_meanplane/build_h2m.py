"""RECONSTRUCTED H2-M main-range real-spectrum injection builder.

*** THIS IS A RECONSTRUCTION, NOT THE EXECUTED PRODUCER. ***
The original 540-sightline H2-M builder ran from an ephemeral agent scratchpad
in 2026-08-16 and is UNRECOVERED (stated on the record at
paper_figures/reductions/h2m_reduce.py:1-25).  Its numeric seed was never
preserved -- h2m_summary.json records only the salt string "h2m1" -- and a
read-only search over 13 candidate seeds on the exactly-reconstructed
median-plane parent pool (362,532 rows, matching the frozen record) reproduced
at most 3 / 540 of the frozen sightlines.  The frozen H2-M draw therefore
CANNOT be reproduced, and there is no byte-for-byte gate for this arm.

What IS verified: the substrate selection reconstructed here reproduces the
frozen H2-M parent pool exactly under --snr-source archive-median, and every
one of the frozen 540 sightlines satisfies it (BAL-excluded, RED_SNR > 2,
z_QSO in [2.1, 3.79), ZWARN == 0, NO A1 / P>0.99 restriction -- 372 of the
frozen 540 carry a pre-existing P>0.99 row, which is why this builder is a
sibling of build_cleanreal.py and not build_cleanreal.py itself).

Design/injection/analysis contract identical to build_cleanreal.py.
Outputs under <run-dir>: h2m_realized_plan.csv, h2m_sightlines.csv,
h2m_injected.h5 (+ truth csv), qsocat_h2m.fits, h2m_hpx_list.txt,
h2m_summary.json.  Real TARGETIDs stay on scratch (never in git).

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

RUN = '/scratch/cavestru_root/cavestru0/mfho/h2m_meanplane_20260911/h2m'
ANA = '/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis'
SRC = '/scratch/cavestru_root/cavestru0/mfho/nersc/loa_archives/loa_full_z2_noR_v2.h5'
QSOCAT = '/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits'
REALCAT = ('/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/'
           'dlacat-loa-main-dark-v1.fits')
# DECLARED seed (R2_PREDECLARATION.md section 4): the campaign's own
# date-of-build convention (clean arm 2026-08-17 -> 20260817; mock closure
# 2026-08-19 -> 20260819; H2-M was built 2026-08-16).  NOT the frozen seed,
# which is unrecovered.  Fixed before any draw; never re-tuned.
SEED = 20260816
NUM_LINES = 3

ap = argparse.ArgumentParser()
ap.add_argument('--snr-source', default=ss.ARCHIVE_MEDIAN,
                choices=list(ss.SNR_SOURCE_CHOICES),
                help='S/N variable for the >2 cut AND the cell stratification. '
                     'Default reproduces the frozen (defective) plan exactly.')
ap.add_argument('--run-dir', default=RUN)
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
# H2-M does NOT apply the clean arm's A1 tier: pre-existing accepted detections
# stay in the substrate (372 / 540 frozen sightlines carry one).  The catalogue
# is read only for the ANY-P collision candidates below.

# S/N variable: archive median (frozen default) or the canonical finder mean
snr_used, eligible, snr_meta = ss.resolve_snr(
    tid_arch, cat['RED_SNR'].astype(float), snr_source=args.snr_source)
print('snr_source', json.dumps(snr_meta))

sel = ((cat['Z'] >= 2.1) & (cat['Z'] < 3.79) & eligible & (snr_used > 2.0)
       & ~np.isin(tid_arch, bal_bad) & (cat['ZWARN'] == 0))
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
print('H2-M parent pool', len(parent), time.time() - t0,
      '(frozen median-plane record: 362532)')

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
with open(RUN + '/h2m_sightlines.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, sl_fields)
    w.writeheader()
    [w.writerow(s) for s in sl_rows]
with open(RUN + '/h2m_realized_plan.csv', 'w', newline='') as f:
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

with h5py.File(SRC, 'r') as hs, h5py.File(RUN + '/h2m_injected.h5', 'w') as ho:
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
    ho.attrs['ckpt10p5_campaign'] = 'h2m_main_reconstructed'

# truth manifest
with open(RUN + '/h2m_injected.h5.truth.csv', 'w', newline='') as f:
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
fitsio.write(RUN + '/qsocat_h2m.fits', sub, clobber=True)
del qt, sub
hpx = sorted({s['HPXPIXEL'] for s in sightlines})
with open(RUN + '/h2m_hpx_list.txt', 'w') as f:
    f.write('\n'.join(str(h) for h in hpx) + '\n')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()

summary = dict(
    campaign='H2-M main-range real injection (RECONSTRUCTED builder; frozen producer unrecovered)',
    snr_source=args.snr_source, snr_field=SNR_FIELD, snr_meta=snr_meta,
    seed=SEED, n_sightlines=len(sightlines), n_injections=len(plan_rows),
    n_dropped=len(dropped), dropped=dropped, n_hpx=len(hpx),
    parent_pool=int(len(parent)),
    config=dict(salt='h2m1', logN_grid=list(cp.LOGN_GRID),
                logN_weights=list(cp.LOGN_W), sl_per_cell=cp.SL_PER_CELL,
                dbl_per_cell=cp.DBL_PER_CELL, z_inj_cap=list(cp.Z_CAP),
                collision_kms=cp.COLLISION_KMS, collar_kms=cp.COLLAR_KMS,
                window='lya_only 1025-1216', num_lines=NUM_LINES,
                sibling_avoidance=False),
    source_archive=SRC, source_archive_note='sha inherited from attrs',
    injected_archive_sha256=sha(RUN + '/h2m_injected.h5'),
    plan_sha256=sha(RUN + '/h2m_realized_plan.csv'),
    truth_sha256=sha(RUN + '/h2m_injected.h5.truth.csv'),
)
json.dump(summary, open(RUN + '/h2m_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ('dropped',)}, indent=1))
print('DONE', time.time() - t0)
