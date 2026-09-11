"""Shared analysis for the checkpoint-10.5 campaigns: molly-exact join
(h2mlib, validated 900/900 against H2-M) + the PINNED H2-M weighting target
(weight_target.json: calibration cell fracs + per-cell 8-bin z_QSO fracs,
recovered exactly from the recorded weights) + the certified natpair mockC.

Usage: python analyze_campaign.py <outputs_dir> <truth_csv> <sightlines_csv>
       <out_json> [--placebo-cat CAT --collision-src real|mock]
Produces the same summary structure as h2m_results.json (one arm).
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, '/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/38fff699-e218-48a5-b1c3-ceb424407bd5/scratchpad')
import h2mlib

NAT = ('/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/'
       'stage0/p1_natpair_ck_v1.npz')
WTGT = ('/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/'
        'analysis/weight_target.json')

def main(outputs_dir, truth_csv, sightlines_csv, out_json):
    with open(truth_csv) as f:
        trows = list(csv.DictReader(f))
    truth = {
        'TARGETID': np.array([int(r['TARGETID']) for r in trows], np.int64),
        'z_inj': np.array([float(r['z_true']) for r in trows]),
        'logN': np.array([float(r['logN_true']) for r in trows]),
    }
    cells = [r['cell'] for r in trows]
    sl = {}
    with open(sightlines_csv) as f:
        for r in csv.DictReader(f):
            sl[int(r['TARGETID'])] = (float(r['Z_QSO']), float(r['RED_SNR']),
                                      r['cell'])
    rows = h2mlib.load_dlacat_rows(outputs_dir)
    rec, dlogN, dv = h2mlib.join_injections(rows, truth)

    # ---- weights: pinned target / this campaign's occupancy ---------------
    tgt = json.load(open(WTGT))
    cell_frac = np.array(tgt['cell_frac'])
    n_by_cell = {}
    for tid, (zq, snr, cn) in sl.items():
        n_by_cell[cn] = n_by_cell.get(cn, 0) + 1
    n_sl = sum(n_by_cell.values())
    w = np.empty(len(trows))
    diag_cells = {}
    for i, r in enumerate(trows):
        cn = r['cell']
        si, zi = int(cn[1]), int(cn[3])
        imp = cell_frac[si][zi] / (n_by_cell[cn] / n_sl)
        pc = tgt['per_cell'][cn]
        edges = np.array(pc['edges'])
        calib_zfrac = np.array(pc['calib_zfrac'])
        # campaign z-bin fracs for THIS campaign
        zqs = np.array([v[0] for v in sl.values()
                        if v[2] == cn])
        camp_counts = np.histogram(zqs, edges)[0].astype(float)
        camp_frac = camp_counts / camp_counts.sum()
        zq = sl[int(r['TARGETID'])][0]
        b = min(int((zq - edges[0]) / (edges[-1] - edges[0]) * tgt['nbin']),
                tgt['nbin'] - 1)
        if camp_frac[b] > 0:
            # renormalize the calib target to bins the campaign populates
            live = camp_frac > 0
            czf = calib_zfrac.copy()
            czf[~live] = 0.0
            czf = czf / czf.sum()
            zfac = czf[b] / camp_frac[b]
        else:
            zfac = 1.0
        w[i] = imp * zfac
        diag_cells.setdefault(cn, {'ess_num': 0.0, 'ess_den': 0.0})
        diag_cells[cn]['ess_num'] += w[i]
        diag_cells[cn]['ess_den'] += w[i] ** 2

    # ---- mockC lookup (certified natpair matrix) --------------------------
    p = np.load(NAT, allow_pickle=True)
    C = (p['C_molly_n_det'].astype(float)
         / np.maximum(p['C_molly_n_tot'].astype(float), 1))
    SE = np.asarray(p['C_snr_edges'], float)
    NE = np.asarray(p['C_nhi_edges'], float)
    mockC = np.empty(len(trows))
    for i, r in enumerate(trows):
        snr = sl[int(r['TARGETID'])][1]
        s = min(int(np.searchsorted(SE, snr, side='right') - 1), len(SE) - 2)
        n = int(np.searchsorted(NE, truth['logN'][i], side='right') - 1)
        mockC[i] = C[s, n]

    logn = truth['logN']
    out = {'n_injections': len(trows), 'n_accepted_rows': int(len(rows)),
           'ess_total': round(float(w.sum() ** 2 / np.sum(w ** 2)), 1)}
    for name, s in [('all', np.ones(len(logn), bool)),
                    ('ge20.3', logn >= 20.3),
                    ('20.0-20.3', (logn >= 20.0) & (logn < 20.3)),
                    ('19.5-20.0', (logn >= 19.5) & (logn < 20.0))]:
        out[name] = h2mlib.ratio_summary(w, rec, mockC, s)
    out['by_logN'] = {}
    for g in sorted(set(logn)):
        out['by_logN'][f'{g:.2f}'] = h2mlib.ratio_summary(
            w, rec, mockC, logn == g)
    out['by_cell_ge20.3'] = {}
    carr = np.array(cells)
    for cn in sorted(set(cells)):
        m = (carr == cn) & (logn >= 20.3)
        if m.sum() >= 5:
            out['by_cell_ge20.3'][cn] = h2mlib.ratio_summary(w, rec, mockC, m)
    m = rec
    out['landing'] = dict(
        n_matched=int(m.sum()),
        dlogN_p16_50_84=[round(float(x), 3) for x in
                         np.percentile(dlogN[m], [16, 50, 84])],
        dlogN_mean=round(float(np.mean(dlogN[m])), 4),
        dv_p16_50_84_kms=[round(float(x), 1) for x in
                          np.percentile(dv[m], [16, 50, 84])],
        big_misses=int((np.abs(dlogN[m]) > 0.5).sum()))
    # per-injection records for downstream decomposition
    out['per'] = [dict(tid=int(truth['TARGETID'][i]), cell=cells[i],
                       logN=float(logn[i]), z=float(truth['z_inj'][i]),
                       snr=float(sl[int(truth['TARGETID'][i])][1]),
                       w=float(w[i]), rec=bool(rec[i]),
                       dlogN=(None if not rec[i] else float(dlogN[i])),
                       mockC=float(mockC[i]))
                  for i in range(len(trows))]
    json.dump(out, open(out_json, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('per', 'by_logN')}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
