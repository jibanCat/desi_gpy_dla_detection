#!/usr/bin/env python
"""decompose_highn_fp.py — truth-anchored high-N false-positive decomposition on the
2LPT-0 loa-124 mock, and the effective real-CDDF residual after BAL-finder completeness.

REDUCE-ONLY (no inference). Uses the CANONICAL production GP catalog on loa-124 + the mock
truth/BAL catalogs. Every high-N detection in an HCD-FREE sightline (no injected DLA per
hcd_truth_cat) is a definitive false positive; the dlacat's postprocess flags (BAL_FLAG,
LYBETA_FLAG, DLAFLAG) split it by source. We apply the CDDF forest op-cut (SNR_forest>2),
Omega-weight (sum 10^NHI) vs the SNR>2 truth, bin the BAL FPs by strength (BI_CIV) and by
SNR_REDSIDE, and fold the DESI BAL-finder completeness(SNR) (Filbert/Martini 2024:
~95% asymptote at higher SNR, dropping only for SNR<1) to get the effective residual that
leaks into the real clean set.  Design + numbers: notes/2026-07-02_bal_metal_lyb_fp_plan.md.
"""
from __future__ import annotations
import argparse, numpy as np, fitsio

DEF_DLACAT="/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/combined_catalog/dlacat-v2.8.5-mockcat.fits"
DEF_MOCK="/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124"

def comp_snr(snr, asym=0.95, scale=1.0):
    """BAL-finder completeness vs SNR (rough, paper-consistent: ~95% asymptote, drop <1)."""
    return asym*(1.0-np.exp(-np.clip(snr,0,None)/scale))

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--dlacat", default=DEF_DLACAT)
    p.add_argument("--mockdir", default=DEF_MOCK)
    p.add_argument("--snr-forest-min", type=float, default=2.0)
    p.add_argument("--comp-asym", type=float, default=0.95)
    p.add_argument("--comp-scale", type=float, default=1.0)
    a=p.parse_args(argv)
    d=fitsio.read(a.dlacat)
    tid=d["TARGETID"]; nhi=d["NHI"].astype(float)
    balf=d["BAL_FLAG"].astype(bool); lybf=d["LYBETA_FLAG"].astype(bool); dflag=d["DLAFLAG"]
    snrF=d["SNR_FOREST"].astype(float); snrR=d["SNR_REDSIDE"].astype(float)
    tr=fitsio.read(f"{a.mockdir}/hcd_truth_cat.fits")
    truth=set(np.unique(tr["TARGETID"]).tolist()); tr_nhi=tr["NHI"].astype(float); tr_tid=tr["TARGETID"]
    bal=fitsio.read(f"{a.mockdir}/bal_cat.fits")
    bi=dict(zip(bal["TARGETID"].tolist(), np.asarray(bal["BI_CIV"],float).tolist()))
    sc=fitsio.read(f"{a.mockdir}/snr_cat.fits"); scc=[c for c in sc.dtype.names if 'SNR' in c.upper()][0]
    snr_of=dict(zip(sc["TARGETID"].tolist(), np.asarray(sc[scc],float).tolist()))
    tr_snr=np.array([snr_of.get(t,np.nan) for t in tr_tid])
    hcdfree=np.fromiter((t not in truth for t in tid),bool,len(tid))
    op=snrF>a.snr_forest_min
    def tw(lim): m=(tr_nhi>=lim)&(tr_snr>a.snr_forest_min); return np.sum(10.0**tr_nhi[m])
    ow=lambda mask: float(np.sum(10.0**nhi[mask]))
    print(f"# dlacat={a.dlacat}\n# SNR_forest>{a.snr_forest_min}: {op.sum()}/{len(d)} detections; "
          f"HCD-free-sightline detections: {(hcdfree&op).sum()}")
    print("\n## FP Omega over-count vs truth (SNR>%g), by source & N_HI bin" % a.snr_forest_min)
    print(f"{'N_HI':12s} {'BAL%':>7s} {'metal%':>8s} {'Lyb%':>6s} {'nBAL':>6s} {'nMetal':>7s}")
    for lo,hi in [(20.3,21.0),(21.0,21.6),(21.6,99),(20.3,99),(21.6,99)]:
        t=tw(lo)
        selB=hcdfree&balf&op&(nhi>=lo)&(nhi<hi); selM=hcdfree&~balf&~lybf&(dflag==0)&op&(nhi>=lo)&(nhi<hi)
        selL=hcdfree&lybf&~balf&op&(nhi>=lo)&(nhi<hi)
        tag=f"[{lo},{hi if hi<90 else 'inf'})"
        print(f"{tag:12s} {100*ow(selB)/t:7.1f} {100*ow(selM)/t:8.2f} {100*ow(selL)/t:6.2f} {selB.sum():6d} {selM.sum():7d}")
    print("\n## Effective residual BAL contamination after finder completeness(SNR_red)")
    print(f"   (completeness model: {a.comp_asym}*(1-exp(-SNR/{a.comp_scale})))")
    for lim in (20.3,21.6):
        fp=hcdfree&balf&op&(nhi>=lim); w=10.0**nhi[fp]; s=snrR[fp]; t=tw(lim)
        over=100*w.sum()/t; c=comp_snr(s,a.comp_asym,a.comp_scale)
        eff=100*np.sum(w*(1-c))/t; f95=100*np.sum(w*0.05)/t
        print(f"   >= {lim}: over-count(unflagged)={over:5.1f}%  effective(SNR-model)={eff:4.1f}%  "
              f"flat95={f95:4.1f}%  median SNR_red={np.median(s):.1f}")
    # BAL FP Omega by BI strength (which BALs cause it)
    fp=hcdfree&balf&op&(nhi>=20.3); w=10.0**nhi[fp]; b=np.array([bi.get(t,np.nan) for t in tid[fp]])
    tot=w.sum()
    print("\n## BAL-FP Omega(>=20.3) by BI_CIV strength")
    for lo,hi,nm in [(-1e9,0,"BI<=0"),(0,500,"0-500"),(500,2000,"500-2000"),(2000,1e9,"BI>2000 strong")]:
        m=(b>lo)&(b<=hi); print(f"   {nm:16s}: {100*w[m].sum()/tot:5.1f}%  (n={m.sum()})")

if __name__=="__main__":
    main()
