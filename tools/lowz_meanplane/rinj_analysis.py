"""Post-mock-closure analysis: R_inj, iso/blend split, protocol-corrected
transport, survey-weighted corrected factor, fig22."""
import csv
import json
import sys

import numpy as np
from collections import defaultdict

sys.path.insert(0, '/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/38fff699-e218-48a5-b1c3-ceb424407bd5/scratchpad')

A = '/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817'
H2M = '/scratch/cavestru_root/cavestru0/mfho/h2m_20260817'
C_KMS = 2.998e5

mock = json.load(open(A + '/analysis/mockinj_results.json'))
real = json.load(open(H2M + '/h2m_perinj.json'))['100k']['per']

def iso_map(planf):
    plan = list(csv.DictReader(open(planf)))
    by = defaultdict(list)
    for r in plan:
        by[int(r['TARGETID'])].append(float(r['z_inj']))
    return {t: (len(v) == 1 or abs(v[0] - v[1]) / (1 + min(v)) * C_KMS >= 6000)
            for t, v in by.items()}

iso_m = iso_map(A + '/mockinj/h2mm_realized_plan.csv')
iso_r = iso_map(H2M + '/h2m_realized_plan.csv')

def arrays(per, isod):
    return dict(
        w=np.array([p['w'] for p in per]),
        rec=np.array([p['rec'] for p in per], float),
        logn=np.array([p['logN'] for p in per]),
        mock=np.array([p['mockC'] for p in per]),
        cell=np.array([p['cell'] for p in per]),
        iso=np.array([isod[p['tid']] for p in per]))

M = arrays(mock['per'], iso_m)
R = arrays(real, iso_r)

def wratio(a, m):
    Cw = float(np.sum(a['w'][m] * a['rec'][m]) / np.sum(a['w'][m]))
    mCw = float(np.sum(a['w'][m] * a['mock'][m]) / np.sum(a['w'][m]))
    neff = a['w'][m].sum() ** 2 / np.sum(a['w'][m] ** 2)
    se = np.sqrt(max(Cw * (1 - Cw), 1e-9) / neff) / mCw
    return Cw, mCw, Cw / mCw, se, int(m.sum())

NB = [('ge20.3', lambda l: l >= 20.3),
      ('20.0-20.3', lambda l: (l >= 20.0) & (l < 20.3)),
      ('19.5-20.0', lambda l: (l >= 19.5) & (l < 20.0))]

out = {'R_inj': {}, 'decomposition': {}, 'R_inj_by_class': {}}
for nm, f in NB:
    mm = f(M['logn'])
    mr = f(R['logn'])
    Ci, Cn, rinj, se_i, n_i = wratio(M, mm)
    Cr, Cn_r, rreal, se_r, n_r = wratio(R, mr)
    tcorr = rreal / rinj
    se_t = tcorr * np.hypot(se_i / rinj, se_r / rreal)
    # direct cross-ratio with each campaign's own weights
    cross = Cr / Ci
    out['R_inj'][nm] = dict(C_mock_inj=round(Ci, 4), C_mock_nat=round(Cn, 4),
                            R_inj=round(rinj, 4), se=round(se_i, 4), n=n_i)
    out['decomposition'][nm] = dict(
        real_ratio=round(rreal, 4), T_corr_realratio_over_Rinj=round(tcorr, 4),
        se=round(float(se_t), 4), direct_Creal_over_Cmockinj=round(cross, 4))
    # iso/blend split
    for lab, sel in [('iso', M['iso']), ('blend', ~M['iso'])]:
        _, _, r, se, n = wratio(M, mm & sel)
        out['R_inj_by_class'][f'{nm}_{lab}'] = f"{r:.4f}+-{se:.4f} (n={n})"

# survey-weighted protocol-corrected factor: per (cell, Nbin) stratum,
# r_corr_s = C_real,s / C_mockinj,s ; F = sum W_s / r_corr_s / sum W_s
# with the same real-survey weights as WS4; plus the single-ratio version.
import fitsio
REALCAT = ('/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/'
           'dlacat-loa-main-dark-v1.fits')
QSOCAT = '/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits'
SNR_E = [2.0, 3.5, 6.5, np.inf]
Z_E = [2.1, 2.56, 2.96, 3.79]
NBINS = [(19.5, 20.0), (20.0, 20.3), (20.3, 20.75), (20.75, 21.25),
         (21.25, 22.5)]
LYA = 1215.67
coll = 3000.0 / C_KMS
cat = fitsio.read(REALCAT, ext=1,
                  columns=['TARGETID', 'Z_QSO', 'SNR_REDSIDE', 'Z_DLA',
                           'NHI', 'P_DLA', 'DLAFLAG'])
qso = fitsio.read(QSOCAT, ext=1, columns=['TARGETID', 'BI_CIV'])
bal = qso['TARGETID'][qso['BI_CIV'] > 0].astype(np.int64)
zq = cat['Z_QSO'].astype(float)
snr = cat['SNR_REDSIDE'].astype(float)
zd = cat['Z_DLA'].astype(float)
nhi = cat['NHI'].astype(float)
zlo_w = (1025.0 * (1 + zq) / LYA - 1.0)
zhi_w = (1216.0 * (1 + zq) / LYA - 1.0)
z_lo = np.maximum(3600.0 / LYA - 1.0, zlo_w + (1 + zlo_w) * coll)
z_hi = np.minimum(zq - (1 + zq) * coll, zhi_w - (1 + zhi_w) * coll)
acc = ((cat['P_DLA'] > 0.99) & (cat['DLAFLAG'] == 0)
       & ~np.isin(cat['TARGETID'].astype(np.int64), bal)
       & (snr > 2.0) & (zq > 2.0) & (zq < 4.25)
       & (zd > z_lo) & (zd < z_hi) & (nhi >= 19.5))
D = cat[acc]
si_d = np.digitize(D['SNR_REDSIDE'].astype(float), SNR_E[1:3])
zi_d = np.clip(np.digitize(D['Z_QSO'].astype(float), Z_E[1:3]), 0, 2)
nhi_d = D['NHI'].astype(float)

def nb_of(x):
    for i, (a, b) in enumerate(NBINS):
        if a <= x < b or (i == len(NBINS) - 1 and x >= b):
            return i

nb_d = np.array([nb_of(x) for x in nhi_d])
Mnb = np.array([nb_of(x) for x in M['logn']])
Rnb = np.array([nb_of(x) for x in R['logn']])

def survey_corrected(thr, idxR=None, idxM=None):
    if idxR is None:
        idxR = np.arange(len(R['w']))
    if idxM is None:
        idxM = np.arange(len(M['w']))
    keep = nhi_d >= thr
    num = den = 0.0
    for s in range(3):
        for z in range(3):
            cn = f's{s}z{z}'
            for nb in range(len(NBINS)):
                dc = int(((si_d[keep] == s) & (zi_d[keep] == z)
                          & (nb_d[keep] == nb)).sum())
                if dc == 0:
                    continue
                mr = (R['cell'][idxR] == cn) & (Rnb[idxR] == nb)
                mi = (M['cell'][idxM] == cn) & (Mnb[idxM] == nb)
                if mr.sum() >= 3:
                    cr = float(np.mean(R['rec'][idxR][mr]))
                else:
                    mr2 = Rnb[idxR] == nb
                    cr = float(np.mean(R['rec'][idxR][mr2]))
                if mi.sum() >= 3:
                    ci = float(np.mean(M['rec'][idxM][mi]))
                else:
                    mi2 = Mnb[idxM] == nb
                    ci = float(np.mean(M['rec'][idxM][mi2]))
                mcm = (float(np.mean(M['mock'][idxM][mi])) if mi.sum() >= 3
                       else float(np.mean(M['mock'][idxM][Mnb[idxM] == nb])))
                W = dc / max(mcm, 0.05)
                r_corr = max(cr, 0.05) / max(ci, 0.05)
                num += W / max(r_corr, 0.05)
                den += W
    return num / den

rng = np.random.default_rng(20260823)
CELLS = [f's{s}z{z}' for s in range(3) for z in range(3)]
for thr in (20.3, 20.0):
    f0 = survey_corrected(thr)
    boots = []
    for _ in range(200):
        iR = np.concatenate([rng.choice(np.where(R['cell'] == c)[0],
                                        size=int((R['cell'] == c).sum()),
                                        replace=True) for c in CELLS])
        iM = np.concatenate([rng.choice(np.where(M['cell'] == c)[0],
                                        size=int((M['cell'] == c).sum()),
                                        replace=True) for c in CELLS])
        boots.append(survey_corrected(thr, iR, iM))
    out[f'survey_corrected_ge{thr}'] = dict(
        F=round(f0, 4),
        p16_84=[round(float(x), 4) for x in np.percentile(boots, [16, 84])])

json.dump(out, open(A + '/analysis/rinj_decomposition.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# ---- fig22 ---------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIGDIR = '/home/mfho/desi_gpy_dla_notes/figures/2026-08-17_stilt_diag/'
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
nbs = ['ge20.3', '20.0-20.3', '19.5-20.0']
x = np.arange(3)
ri = [out['R_inj'][n]['R_inj'] for n in nbs]
re = [out['R_inj'][n]['se'] for n in nbs]
rr = [out['decomposition'][n]['real_ratio'] for n in nbs]
tc = [out['decomposition'][n]['T_corr_realratio_over_Rinj'] for n in nbs]
te = [out['decomposition'][n]['se'] for n in nbs]
ax.errorbar(x - 0.15, rr, fmt='s', label='real ratio (H2-M, recorded conv.)',
            color='C3')
ax.errorbar(x, ri, yerr=re, fmt='o', label='R_inj (mock closure)', color='C0',
            capsize=3)
ax.errorbar(x + 0.15, tc, yerr=te, fmt='^',
            label='protocol-corrected transport', color='C2', capsize=3)
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set_xticks(x, ['≥20.3', '20.0–20.3', '19.5–20.0'])
ax.set_ylabel('completeness ratio')
ax.set_title('Mock injection closure and decomposition')
ax.legend(fontsize=8)
ax = axes[1]
labs = []
vals = []
for nm in nbs:
    for lab in ['iso', 'blend']:
        v = out['R_inj_by_class'][f'{nm}_{lab}']
        vals.append(float(v.split('+-')[0]))
        labs.append(f'{nm}\n{lab}')
ax.bar(range(len(vals)), vals,
       color=['C0' if 'iso' in l else 'C3' for l in labs], alpha=0.85)
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set_xticks(range(len(vals)), labs, fontsize=7)
ax.set_ylabel('R_inj')
ax.set_title('R_inj by sibling class (mock substrate)')
fig.suptitle('fig22 — injection-protocol closure', y=1.00)
fig.tight_layout()
fig.savefig(FIGDIR + 'fig22_rinj_closure.png', dpi=150)
print('wrote fig22')
