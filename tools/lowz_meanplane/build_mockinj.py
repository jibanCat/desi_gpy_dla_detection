"""WS2B: build the MOCK injection-closure campaign (checkpoint 10.5 ruling
item 2B) — the SAME H2-M injection protocol run on the 2LPT-0 calibration
mock substrate, so that R_inj = C_mock,injected / C_mock,natural isolates the
injection-protocol completeness effect.

Protocol mirrored exactly (validated planner, same primitive):
  - substrate selection mirrors H2-M's real contract: BAL-excluded (mock
    bal_cat, ALL rows — the molly mock convention), SNR_REDSIDE > 2,
    z_QSO in [2.1, 3.79); mock truth HCDs are NOT removed (the real
    substrate keeps its real absorption too);
  - collision avoidance vs ANY 2LPT-0 catalog candidate at ANY P (the
    protocol is defined on DETECTED candidates, as on real data);
  - same cells / 60 per cell / 40 doubled / logN grid+weights / window +
    3,000 km/s collars / cap / NO sibling avoidance;
  - injection via the committed coadd_injection.write_campaign (same
    voigt_transmission primitive, num_lines=3) into a same-layout tree;
  - finder config = certified calibration config (100k/MAX_DLAS=4/FILTER=1).

Outputs under /scratch/.../h2m_ckpt10p5_20260817/mockinj/.
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import fitsio

sys.path.insert(0, '/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/38fff699-e218-48a5-b1c3-ceb424407bd5/scratchpad')
sys.path.insert(0, '/home/mfho/wt_forward_2026_08')
import campaign_planner as cp
from injection.coadd_injection import write_campaign

M = '/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124'
RUN = '/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/mockinj'
MOCKCAT = ('/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/2lpt0_loa124_v1/'
           'dlacat-v2.8.5-mockcat.fits')
SEED = 20260819
NUM_LINES = 3

os.makedirs(RUN, exist_ok=True)
t0 = time.time()

zc = fitsio.read(M + '/zcat.fits', ext=1,
                 columns=['TARGETID', 'Z', 'TARGET_RA', 'TARGET_DEC'])
sn = fitsio.read(M + '/snr_cat.fits', ext=1)
bal = fitsio.read(M + '/bal_cat.fits', ext=1,
                  columns=['TARGETID'])['TARGETID'].astype(np.int64)
snr_map = dict(zip(sn['TARGETID'].astype(np.int64),
                   sn['SNR_REDSIDE'].astype(float)))

import healpy as hp
tid = zc['TARGETID'].astype(np.int64)
zq = zc['Z'].astype(float)
snr = np.array([snr_map.get(int(t), np.nan) for t in tid])
hpx = hp.ang2pix(16, zc['TARGET_RA'].astype(float),
                 zc['TARGET_DEC'].astype(float), nest=True,
                 lonlat=True).astype(np.int64)
sel = ((zq >= 2.1) & (zq < 3.79) & (snr > 2.0) & ~np.isin(tid, bal))
parent = np.zeros(int(sel.sum()), dtype=[('TARGETID', np.int64),
                                         ('Z_QSO', float), ('RED_SNR', float),
                                         ('HPXPIXEL', np.int64)])
parent['TARGETID'] = tid[sel]
parent['Z_QSO'] = zq[sel]
parent['RED_SNR'] = snr[sel]
parent['HPXPIXEL'] = hpx[sel]
print('mock parent pool', len(parent), time.time() - t0, flush=True)

cat = fitsio.read(MOCKCAT, ext=1, columns=['TARGETID', 'Z_DLA'])
ctid = cat['TARGETID'].astype(np.int64)
sel_c = np.isin(ctid, parent['TARGETID'])
cand = {}
for t, z in zip(ctid[sel_c], cat['Z_DLA'][sel_c].astype(float)):
    cand.setdefault(int(t), []).append(z)
print('pool TIDs with any-P candidates', len(cand), flush=True)

sightlines, plan_rows, dropped = cp.plan_campaign(parent, cand, SEED)
print('planned', len(sightlines), 'sightlines,', len(plan_rows),
      'injections,', len(dropped), 'dropped', flush=True)

with open(RUN + '/h2mm_sightlines.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['TARGETID', 'Z_QSO', 'RED_SNR', 'HPXPIXEL',
                           'cell', 'n_inj'])
    w.writeheader()
    [w.writerow(s) for s in sightlines]
with open(RUN + '/h2mm_realized_plan.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['TARGETID', 'inj_idx', 'cell', 'Z_QSO', 'HPXPIXEL',
                           'z_inj', 'logN', 'z_segment', 'attempts'])
    w.writeheader()
    [w.writerow(r) for r in plan_rows]

# manifest: one row per sightline, doubles via logN_true2/z_true2
from collections import defaultdict
by = defaultdict(list)
for r in plan_rows:
    by[int(r['TARGETID'])].append(r)
manifest = []
sl_info = {int(s['TARGETID']): s for s in sightlines}
for i, (t, rs) in enumerate(sorted(by.items())):
    s = sl_info[t]
    row = dict(inj_id=i, campaign='h2mm_ckpt10p5', method='coadd',
               target_id=t, healpix=int(s['HPXPIXEL']),
               z_qso=float(s['Z_QSO']), snr_bin=s['cell'],
               native_snr=float(s['RED_SNR']),
               logN_true=float(rs[0]['logN']), z_true=float(rs[0]['z_inj']),
               num_lines=NUM_LINES)
    if len(rs) == 2:
        row['logN_true2'] = float(rs[1]['logN'])
        row['z_true2'] = float(rs[1]['z_inj'])
    manifest.append(row)

out_tree = RUN + '/injected_tree'
tm = write_campaign(manifest, None, out_root=out_tree, mockdir=M,
                    num_lines=NUM_LINES,
                    truth_manifest_name='h2mm_injection_truth.fits')
print('tree written; truth manifest:', tm, time.time() - t0, flush=True)

# H2-M-format truth CSV (for the shared join analysis)
with open(RUN + '/h2mm_truth.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['TARGETID', 'inj_idx', 'cell', 'z_true', 'logN_true',
                'num_lines'])
    for r in plan_rows:
        w.writerow([r['TARGETID'], r['inj_idx'], r['cell'], r['z_inj'],
                    r['logN'], NUM_LINES])

# restricted zcat for the finder run (full zcat schema, campaign rows only)
full = fitsio.read(M + '/zcat.fits', ext=1)
fsel = np.isin(full['TARGETID'].astype(np.int64),
               np.array(sorted(by.keys()), np.int64))
fitsio.write(RUN + '/zcat_h2mm.fits', full[fsel], clobber=True)
hpxs = sorted({int(s['HPXPIXEL']) for s in sightlines})
with open(RUN + '/h2mm_hpx_list.txt', 'w') as f:
    f.write('\n'.join(str(h) for h in hpxs) + '\n')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()

summary = dict(
    campaign='ckpt10p5 mock injection closure (2LPT-0 substrate)',
    seed=SEED, n_sightlines=len(sightlines), n_injections=len(plan_rows),
    n_dropped=len(dropped), dropped=dropped, n_hpx=len(hpxs),
    parent_pool=int(len(parent)),
    config=dict(salt='h2mm1', logN_grid=list(cp.LOGN_GRID),
                logN_weights=list(cp.LOGN_W), sl_per_cell=cp.SL_PER_CELL,
                dbl_per_cell=cp.DBL_PER_CELL, z_inj_cap=list(cp.Z_CAP),
                collision_kms=cp.COLLISION_KMS, collar_kms=cp.COLLAR_KMS,
                window='lya_only 1025-1216', num_lines=NUM_LINES,
                sibling_avoidance=False,
                bal_policy='mock bal_cat ALL rows excluded',
                collision_source='2lpt0 catalog rows at ANY P'),
    mockdir=M, plan_sha256=sha(RUN + '/h2mm_realized_plan.csv'),
    truth_sha256=sha(RUN + '/h2mm_truth.csv'),
)
json.dump(summary, open(RUN + '/h2mm_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'dropped'},
                 indent=1))
print('DONE', time.time() - t0)
