"""WS4 (ruling item 4): the survey-weighted mock-to-real completeness
estimand.

Estimand: the multiplicative factor F on dN/dX(>=N_thr) if the Paper-1
completeness division switched from C_mock to C_real = r x C_mock:

    F = sum_s [D_s / (C_mock,s r_s)] / sum_s [D_s / C_mock,s]

with strata s = (SNR cell 3) x (z_QSO cell 3) x (N bin), D_s = the ACTUAL
real Paper-1 accepted detections (P>0.99, DLAFLAG==0, BI_CIV-BAL-excluded,
SNR_REDSIDE>2, z_QSO in (2.0,4.25), absorber in the lya window with
collars+3600A floor) — i.e. the real survey population weights, and
r_s = C_real/C_mock measured by H2-M in that stratum (100k arm).

Fallbacks (reported): strata with no injection support inherit the N-bin
global ratio; detections outside the H2-M z_QSO cells (2.1-3.79) are
assigned to the nearest cell (extrapolation share reported).

Uncertainty: bootstrap over injections (per-cell resampling).
Also reports the calibration-occupancy-weighted factor (checkpoint-10
convention) for the same N bins, plus occupancy-overlap / ESS diagnostics.
"""
import json

import numpy as np
import fitsio

H2M = '/scratch/cavestru_root/cavestru0/mfho/h2m_20260817'
REALCAT = ('/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/loa_main_dark_v1/'
           'dlacat-loa-main-dark-v1.fits')
QSOCAT = '/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits'
OUT = ('/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis/'
       'ws4_survey_weighted.json')

SNR_E = [2.0, 3.5, 6.5, np.inf]
Z_E = [2.1, 2.56, 2.96, 3.79]
NBINS = [(19.5, 20.0), (20.0, 20.3), (20.3, 20.75), (20.75, 21.25),
         (21.25, 22.5)]
LYA = 1215.67
C_KMS = 2.998e5

# ---- real Paper-1 detected population ------------------------------------
cat = fitsio.read(REALCAT, ext=1,
                  columns=['TARGETID', 'Z_QSO', 'SNR_REDSIDE', 'Z_DLA',
                           'NHI', 'P_DLA', 'DLAFLAG'])
qso = fitsio.read(QSOCAT, ext=1, columns=['TARGETID', 'BI_CIV'])
bal = qso['TARGETID'][qso['BI_CIV'] > 0].astype(np.int64)

zq = cat['Z_QSO'].astype(float)
snr = cat['SNR_REDSIDE'].astype(float)
zd = cat['Z_DLA'].astype(float)
nhi = cat['NHI'].astype(float)
coll = 3000.0 / C_KMS
z_lo = np.maximum(3600.0 / LYA - 1.0,
                  1025.0 * (1 + zq) / LYA - 1.0 + coll * (1 + 1025.0 * (1 + zq) / LYA - 1.0) * 0)
# molly convention: z_lo = max(3600/lya - 1, lam_min(1+zq)/lya - 1 + collar)
z_lo = np.maximum(3600.0 / LYA - 1.0,
                  (1025.0 * (1 + zq) / LYA - 1.0)
                  + (1 + (1025.0 * (1 + zq) / LYA - 1.0)) * coll)
z_hi = np.minimum(zq - (1 + zq) * coll,
                  (1216.0 * (1 + zq) / LYA - 1.0)
                  - (1 + (1216.0 * (1 + zq) / LYA - 1.0)) * coll)
acc = ((cat['P_DLA'] > 0.99) & (cat['DLAFLAG'] == 0)
       & ~np.isin(cat['TARGETID'].astype(np.int64), bal)
       & (snr > 2.0) & (zq > 2.0) & (zq < 4.25)
       & (zd > z_lo) & (zd < z_hi) & (nhi >= 19.5))
D = cat[acc]
print('real accepted detections (>=19.5, contract cuts):', len(D))

zq_d = D['Z_QSO'].astype(float)
snr_d = D['SNR_REDSIDE'].astype(float)
nhi_d = D['NHI'].astype(float)
si_d = np.digitize(snr_d, SNR_E[1:3])
zi_d = np.clip(np.digitize(zq_d, Z_E[1:3]), 0, 2)
outside = (zq_d < Z_E[0]) | (zq_d >= Z_E[3])
print('share of detections with z_QSO outside [2.1,3.79):',
      round(float(outside.mean()), 4))

# ---- H2-M stratum ratios --------------------------------------------------
per = json.load(open(H2M + '/h2m_perinj.json'))['100k']['per']
p_cell = np.array([p['cell'] for p in per])
p_logn = np.array([p['logN'] for p in per])
p_rec = np.array([p['rec'] for p in per], float)
p_mock = np.array([p['mockC'] for p in per])
p_w = np.array([p['w'] for p in per])

def nbin_of(x):
    for i, (a, b) in enumerate(NBINS):
        if a <= x < b or (i == len(NBINS) - 1 and x >= b):
            return i
    return None

def stratum_ratios(idx):
    """r_s per (cell, N bin) from injections idx; None where empty."""
    r = {}
    for s in range(3):
        for z in range(3):
            cn = f's{s}z{z}'
            for nb in range(len(NBINS)):
                m = (p_cell[idx] == cn) & np.array(
                    [nbin_of(x) == nb for x in p_logn[idx]])
                if m.sum() >= 3:
                    Cw = float(np.mean(p_rec[idx][m]))
                    mCw = float(np.mean(p_mock[idx][m]))
                    r[(cn, nb)] = Cw / mCw if mCw > 0 else None
    # global N-bin fallbacks
    g = {}
    for nb in range(len(NBINS)):
        m = np.array([nbin_of(x) == nb for x in p_logn[idx]])
        g[nb] = (float(np.mean(p_rec[idx][m]))
                 / float(np.mean(p_mock[idx][m])))
    return r, g

def survey_factor(idx, thr):
    r, g = stratum_ratios(idx)
    keep = nhi_d >= thr
    num = den = 0.0
    n_fb = 0
    for s, z, x, mc_snr in zip(si_d[keep], zi_d[keep], nhi_d[keep],
                               snr_d[keep]):
        nb = nbin_of(x)
        cn = f's{s}z{z}'
        rs = r.get((cn, nb))
        if rs is None:
            rs = g[nb]
            n_fb += 1
        rs = max(rs, 0.05)
        den += 1.0
        num += 1.0 / rs
    return num / den, n_fb / max(keep.sum(), 1)

rng = np.random.default_rng(20260820)
res = {}
for thr in (20.3, 20.0):
    f0, fbshare = survey_factor(np.arange(len(per)), thr)
    boots = []
    cells = np.unique(p_cell)
    for _ in range(400):
        idx = np.concatenate([
            rng.choice(np.where(p_cell == c)[0],
                       size=(p_cell == c).sum(), replace=True)
            for c in cells])
        boots.append(survey_factor(idx, thr)[0])
    lo, hi = np.percentile(boots, [16, 84])
    res[f'ge{thr}'] = dict(
        survey_weighted_factor=round(f0, 4),
        boot_p16_84=[round(float(lo), 4), round(float(hi), 4)],
        fallback_share=round(float(fbshare), 4))

# calibration-occupancy convention (checkpoint-10) recomputed for reference
for thr, key in ((20.3, 'ge20.3'), (20.0, 'ge20.0')):
    m = p_logn >= thr
    Cw = float(np.sum(p_w[m] * p_rec[m]) / np.sum(p_w[m]))
    mCw = float(np.sum(p_w[m] * p_mock[m]) / np.sum(p_w[m]))
    res.setdefault(f'ge{thr}', {})
    res[f'ge{thr}']['calib_occupancy_ratio'] = round(Cw / mCw, 4)
    res[f'ge{thr}']['calib_occupancy_implied_factor_naive'] = round(mCw / Cw, 4)

# occupancy diagnostics: real detected population vs calibration occupancy
calib_frac = np.array(json.load(open(H2M + '/h2m_summary.json'))
                      ['calib_cell_frac'])
D_frac = np.zeros((3, 3))
for s in range(3):
    for z in range(3):
        D_frac[s, z] = float(((si_d == s) & (zi_d == z)).sum()) / len(D)
res['occupancy'] = dict(
    real_detected_cell_frac=np.round(D_frac, 4).tolist(),
    calib_cell_frac=calib_frac.tolist(),
    overlap_l1=round(float(1 - 0.5 * np.abs(D_frac - calib_frac).sum()), 4))
res['n_real_detections_ge19.5'] = int(len(D))
res['estimand_note'] = (
    'F = sum D_s/(C_mock r_s) / sum D_s/C_mock over (SNRcell x zcell x Nbin) '
    'strata; D_s = real Paper-1 accepted detections under the molly contract; '
    'r_s from H2-M 100k arm; empty strata inherit the N-bin global ratio.')
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
