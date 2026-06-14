#!/usr/bin/env python
"""Targeted COMPETED sub-DLA selection-function campaign (campaign 'S').

Injects ONE known absorber (fine sub-DLA grid [19.5,20.3) + LLS [17.5,19.5) for the
migration tail) onto NATURAL, HCD-BEARING host sightlines (load_natural_sightlines:
BAL-free + SNR>cut but HCD KEPT), so the host's real absorbers provide realistic
single-slot COMPETITION.  Running the production max_dlas=1 GP on this tree measures:

  * the COMPETED completeness C(logN_true) — the 0.83(clean)→0.76(competed) gap that the
    HCD-free campaigns A/D cannot see — as injection-recovery (the method transferable to
    REAL DESI sightlines, where there is no truth);
  * the recovered-N migration (LLS→recN≥19.5 in particular) and recN distribution;

validating that injection-recovery reproduces the truth-based 0.76 on the mock.  b_FP is
NOT measured here (a natural host's detection may be a real HCD, not a false positive) —
it reuses the CLEAN campaignControls.

Per-row COMPETITOR columns (additive; validate_manifest ignores extras) record the host's
natural HCD so completeness can be stratified by competitor strength / proximity:
  comp_n (HCD count z<z_qso), comp_logN_max (strongest), comp_dz_near (|z_true−z_nearest|),
  comp_logN_near (logN of the nearest-in-z competitor).
"""
import argparse, os, sys
import numpy as np
from astropy.table import Table

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from _gen_common import load_natural_sightlines, finalize_tree, report_restframe
from campaign_grid import build_injection_grid, validate_manifest, default_zqso_bins

DEFAULT_MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                "qq_desi_y3/v2.8.5/mock-0/loa-124")
# Fine sub-DLA grid (the measurable [19.5,20.3) bin) + LLS tail for the recN-migration.
DEFAULT_LOGN = [17.5, 18.0, 18.5, 19.0, 19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2]


def _attach_competitors(inj, mockdir):
    """Add comp_* columns from the host's NATURAL HCD (hcd_truth_cat), per injected row."""
    hcd = Table.read(f"{mockdir}/hcd_truth_cat.fits")
    htid = np.asarray(hcd["TARGETID"], np.int64)
    hN = np.asarray(hcd["NHI"], float); hz = np.asarray(hcd["Z"], float)
    order = np.argsort(htid); htid, hN, hz = htid[order], hN[order], hz[order]
    uq, st = np.unique(htid, return_index=True); en = np.append(st[1:], len(htid))
    sl = {int(uq[i]): (st[i], en[i]) for i in range(len(uq))}
    for r in inj:
        tid = int(r["target_id"]); zq = float(r["z_qso"]); zt = float(r["z_true"])
        comp_n = 0; comp_lNmax = np.nan; comp_dz = np.nan; comp_lNnear = np.nan
        if tid in sl:
            s, e = sl[tid]; zz = hz[s:e]; NN = hN[s:e]
            m = zz < zq                                  # physical: competitor blueward of QSO
            if m.any():
                zz, NN = zz[m], NN[m]
                comp_n = int(m.sum()); comp_lNmax = float(NN.max())
                k = int(np.argmin(np.abs(zz - zt)))
                comp_dz = float(abs(zz[k] - zt)); comp_lNnear = float(NN[k])
        r["comp_n"] = comp_n; r["comp_logN_max"] = comp_lNmax
        r["comp_dz_near"] = comp_dz; r["comp_logN_near"] = comp_lNnear
        # Distinguish competed (natural-host) rows from the clean Campaign-A R-build:
        # both carry campaign='A' (validate_manifest only accepts A/B/D), so an explicit
        # flag is the only safe separator if clean-A and competed rows ever co-locate.
        r["competed"] = True
    return inj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mockdir", default=DEFAULT_MOCK)
    ap.add_argument("--target_injections", type=int, default=7000)
    ap.add_argument("--n_per_cell", type=int, default=None)
    ap.add_argument("--n_healpix", type=int, default=0, help="0 = all natural healpix")
    ap.add_argument("--snr_cut", type=float, default=2.0)
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--logN_grid", type=float, nargs="+", default=DEFAULT_LOGN)
    ap.add_argument("--snr_bins", type=float, nargs="+", default=[2.0, 4.0, 8.0, 1e9])
    ap.add_argument("--zqso_bins", type=float, nargs="+", default=None)
    a = ap.parse_args()
    if a.zqso_bins is None:
        zqso_bins = list(default_zqso_bins())
    elif len(a.zqso_bins) == 1 and a.zqso_bins[0] == 0:
        zqso_bins = None
    else:
        zqso_bins = list(a.zqso_bins)

    nat, nat_sl = load_natural_sightlines(a.mockdir, snr_cut=a.snr_cut, n_healpix=a.n_healpix)
    _npc = a.n_per_cell
    _tgt = None if _npc is not None else a.target_injections
    inj = build_injection_grid(nat_sl, logN_grid=a.logN_grid, snr_bins=a.snr_bins,
                               zqso_bins=zqso_bins, n_per_cell=_npc, target_injections=_tgt,
                               seed=a.seed, campaign="A", method="coadd", num_lines=a.num_lines)
    if not inj:
        raise SystemExit("[manifest] ERROR: zero injections — empty natural∩hostable∩SNR pool.")
    # Under-delivery is bounded by the one-injection-per-host cap; warn loudly (mirrors
    # gen_injectables) so a restricted --n_healpix can't silently ship a thin campaign.
    if _npc is None and len(inj) < int(0.95 * a.target_injections):
        print(f"[manifest] WARNING: requested {a.target_injections} injections but the "
              f"natural pool only supports {len(inj)} distinct hosts "
              f"({100 * len(inj) / a.target_injections:.0f}%). Use --n_per_cell, add "
              f"healpix (--n_healpix 0 = all), or lower --snr_cut.", flush=True)
    inj = _attach_competitors(inj, a.mockdir)
    validate_manifest(inj)
    nlt = np.array([r["logN_true"] for r in inj])
    cn = np.array([r["comp_n"] for r in inj]); clx = np.array([r["comp_logN_max"] for r in inj])
    has_dla_comp = np.mean((clx >= 20.3))
    print(f"[manifest] {len(inj)} injections on NATURAL hosts; logN [{nlt.min():.2f},{nlt.max():.2f}] "
          f"frac sub-DLA[19.5,20.3)={np.mean((nlt>=19.5)&(nlt<20.3)):.2f}", flush=True)
    print(f"[competition] hosts with >=1 natural HCD: {np.mean(cn>0):.2f}; "
          f"with a DLA competitor (logN>=20.3): {has_dla_comp:.2f}; "
          f"median comp_n={np.median(cn):.0f}", flush=True)
    report_restframe(inj, zqso_bins)
    finalize_tree(inj, nat, out_root=a.out, mockdir=a.mockdir, num_lines=a.num_lines)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
