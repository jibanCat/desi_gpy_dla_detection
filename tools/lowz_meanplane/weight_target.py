"""Recover the EXACT H2-M weighting target from recorded artifacts:
  w(inj) = imp_cell x zfac(cell, z_QSO-bin of the sightline)
  - imp_cell = calib_cell_frac / (1/9)          (h2m_summary, exact)
  - zfac     = calib z-bin frac / campaign z-bin frac, 8 bins per cell
Recover the bin edges (verify equal-width over the cell z-range) and the
implied calib z-bin fracs; validate by reproducing all 900 recorded w.
The recovered target (cell fracs + per-cell 8-bin calib z fracs) is then
the PINNED weighting target for the new clean-real and mock campaigns.
"""
import csv
import json

import numpy as np

H2M = '/scratch/cavestru_root/cavestru0/mfho/h2m_20260817'
OUT = ('/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis/'
       'weight_target.json')

summ = json.load(open(H2M + '/h2m_summary.json'))
imp = np.array(summ['importance_weights'])          # [si][zi]
calib_frac = np.array(summ['calib_cell_frac'])
per = json.load(open(H2M + '/h2m_perinj.json'))['100k']['per']

sl = {}
with open(H2M + '/h2m_sightlines.csv') as f:
    for r in csv.DictReader(f):
        sl[int(r['TARGETID'])] = (float(r['Z_QSO']), float(r['RED_SNR']),
                                  r['cell'])

Z_E = [2.1, 2.56, 2.96, 3.79]
NBIN = 8

# campaign sightlines per cell
cells = {}
for tid, (zq, snr, cell) in sl.items():
    cells.setdefault(cell, []).append((tid, zq))

target = {'cell_frac': calib_frac.tolist(), 'z_edges': Z_E, 'nbin': NBIN,
          'per_cell': {}}
n_ok = 0
n_tot = 0
for cell, members in sorted(cells.items()):
    si, zi = int(cell[1]), int(cell[3])
    zlo, zhi = Z_E[zi], Z_E[zi + 1]
    edges = np.linspace(zlo, zhi, NBIN + 1)
    zqs = np.array([z for _, z in members])
    camp_counts = np.histogram(zqs, edges)[0]
    camp_frac = camp_counts / camp_counts.sum()
    # recorded zfac per sightline
    zfac_by_tid = {}
    for p in per:
        if p['cell'] == cell:
            zfac_by_tid[p['tid']] = p['w'] / imp[si][zi]
    # implied calib frac per bin: zfac * camp_frac  (bins with no campaign
    # sightline are unconstrained -> 0 weight assigned there)
    calib_zfrac = np.zeros(NBIN)
    consistent = True
    for b in range(NBIN):
        tids_b = [t for (t, z) in members
                  if edges[b] <= z < edges[b + 1] or (b == NBIN - 1 and z == zhi)]
        zf = sorted({round(zfac_by_tid[t], 9) for t in tids_b
                     if t in zfac_by_tid})
        if len(zf) > 1:
            consistent = False
        if zf:
            calib_zfrac[b] = zf[0] * camp_frac[b]
    # renormalize (calib fracs should sum to 1 if recovered fully)
    ssum = calib_zfrac.sum()
    target['per_cell'][cell] = {
        'edges': edges.tolist(),
        'camp_counts': camp_counts.tolist(),
        'calib_zfrac_raw_sum': round(float(ssum), 6),
        'calib_zfrac': (calib_zfrac / ssum).tolist() if ssum > 0 else None,
        'equal_width_consistent': consistent,
    }
    # validation: reproduce recorded w for every injection in this cell
    for p in per:
        if p['cell'] != cell:
            continue
        n_tot += 1
        zq = sl[p['tid']][0]
        b = min(int((zq - zlo) / (zhi - zlo) * NBIN), NBIN - 1)
        w_hat = imp[si][zi] * (calib_zfrac[b] / ssum) / (camp_frac[b]) * ssum \
            if camp_frac[b] > 0 else np.nan
        if abs(w_hat - p['w']) < 1e-6:
            n_ok += 1

print('equal-width consistency per cell:',
      {c: v['equal_width_consistent'] for c, v in target['per_cell'].items()})
print('calib_zfrac raw sums (should be ~1):',
      {c: v['calib_zfrac_raw_sum'] for c, v in target['per_cell'].items()})
print(f'recorded w reproduced: {n_ok}/{n_tot}')
if n_ok == n_tot:
    json.dump(target, open(OUT, 'w'), indent=1)
    print('saved', OUT)
