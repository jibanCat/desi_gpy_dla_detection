"""Shared orchestration for the injection-campaign generators (A / B / D).

Factors the campaign-INDEPENDENT steps out of ``gen_injectables.py`` so the
close-pair (Campaign B) and non-PW100 (Campaign D) generators reuse the SAME
validated clean-sightline selection, z_QSO-spanning host selection, and
tree-write + restricted-qsocat + coadd-consistency-verify path.  Only the GRID
builder differs between campaigns.
"""
import glob
import os

import numpy as np
from astropy.table import Table

from coadd_injection import build_clean_table, write_campaign, verify_coadd_consistency

LYA = 1215.67


def load_clean_sightlines(mockdir, *, snr_cut=2.0, n_healpix=0):
    """Clean (HCD-free ∩ BAL-free) sightlines with SNR_REDSIDE>cut, optionally
    restricted to ``n_healpix`` healpix that SPAN the host z_QSO range.

    Returns ``(clean_table, clean_sl_dict)`` where ``clean_sl_dict`` is the
    ``{target_id, healpix, z_qso, native_snr}`` form ``campaign_grid`` consumes
    (native_snr = SNR_REDSIDE, the DLA-uncorrelated red-side SNR).
    """
    D = mockdir
    clean = build_clean_table(Table.read(f"{D}/zcat.fits"),
                              Table.read(f"{D}/hcd_truth_cat.fits"),
                              Table.read(f"{D}/bal_cat.fits"),
                              Table.read(f"{D}/snr_cat.fits"))
    rs = np.asarray(clean["SNR_REDSIDE"], float)
    clean = clean[np.isfinite(rs) & (rs > snr_cut)]   # red-side SNR cut
    if n_healpix:
        hp = np.asarray(clean["HEALPIX"], np.int64)
        zq = np.asarray(clean["Z"], float)
        u = np.unique(hp)
        cnt = np.array([(hp == h).sum() for h in u])
        floor = max(20, int(np.median(cnt) * 0.25))
        cand = u[cnt >= floor]
        if cand.size < n_healpix:
            cand = u
        med = np.array([np.median(zq[hp == h]) for h in cand])
        order = np.argsort(med)
        pick = np.unique(np.linspace(0, order.size - 1, n_healpix).round().astype(int))
        clean = clean[np.isin(hp, cand[order[pick]])]
    zqa = np.asarray(clean["Z"], float)
    print(f"[clean] {len(clean)} sightlines on {len(set(clean['HEALPIX'].tolist()))} "
          f"healpix; z_QSO [{zqa.min():.2f},{zqa.max():.2f}] median {np.median(zqa):.2f}",
          flush=True)
    clean_sl = dict(target_id=np.asarray(clean["TARGETID"], np.int64),
                    healpix=np.asarray(clean["HEALPIX"], np.int64),
                    z_qso=np.asarray(clean["Z"], float),
                    native_snr=np.asarray(clean["SNR_REDSIDE"], float))
    return clean, clean_sl


def report_restframe(inj, zqso_bins):
    """Print the rest-frame forest-position (λ_rest) coverage of an injection list."""
    zt = np.array([r["z_true"] for r in inj]); zq = np.array([r["z_qso"] for r in inj])
    lam = (1 + zt) / (1 + zq) * LYA
    if zqso_bins is not None:
        zb = np.asarray([r["zqso_bin"] for r in inj])
        nb = [int((zb == k).sum()) for k in range(len(zqso_bins) - 1)]
        print(f"[restframe] z_QSO bins {list(zqso_bins)} -> per-bin n {nb}; "
              f"λ_rest [{lam.min():.0f},{lam.max():.0f}] Å "
              f"(pct10/50/90={np.percentile(lam,[10,50,90]).round(0).astype(int).tolist()})",
              flush=True)
    else:
        print(f"[restframe] z_QSO strat OFF; λ_rest [{lam.min():.0f},{lam.max():.0f}] Å",
              flush=True)


def finalize_tree(manifest, clean, *, out_root, mockdir, num_lines, n_verify=3):
    """write_campaign → restricted qsocat (only injected/control targets) →
    coadd-consistency verify on a few healpix.  Returns the truth-manifest path."""
    D = mockdir
    truth_path = write_campaign(manifest, clean, out_root=out_root, mockdir=D,
                                num_lines=num_lines)
    n = len(glob.glob(f"{out_root}/spectra-16/*/*/spectra-16-*.fits"))
    print(f"[write] {n} injected coadds -> {out_root}/spectra-16/ ; truth -> {truth_path}",
          flush=True)

    zc = Table.read(f"{D}/zcat.fits")
    want = set(int(r["target_id"]) for r in manifest)
    keep = np.isin(np.asarray(zc["TARGETID"], np.int64), np.array(sorted(want), np.int64))
    qpath = os.path.join(out_root, "pilot_qsocat.fits")
    zc[keep].write(qpath, overwrite=True)
    print(f"[qsocat] {int(keep.sum())} targets -> {qpath}", flush=True)

    # coadd-consistency smoke on a few healpix (per-injection absorber list per tid)
    inj_by_tid = {}
    for r in manifest:
        if r.get("control", False):
            continue
        inj_by_tid.setdefault(int(r["target_id"]), []).append(
            (10.0 ** float(r["logN_true"]), float(r["z_true"])))
        # close-pair second absorber rides the same fiber
        if r.get("logN_true2") is not None and np.isfinite(r.get("logN_true2", np.nan)):
            inj_by_tid[int(r["target_id"])].append(
                (10.0 ** float(r["logN_true2"]), float(r["z_true2"])))
    srcs = sorted(glob.glob(f"{out_root}/spectra-16/*/*/spectra-16-*.fits"))
    worst = 0.0
    for src in srcs[:n_verify]:
        hp = int(src.rsplit("-", 1)[1].split(".")[0])
        orig = f"{D}/spectra-16/{hp // 100}/{hp}/spectra-16-{hp}.fits"
        try:
            w = verify_coadd_consistency(orig, src, inj_by_tid, num_lines=num_lines)
            worst = max(worst, w)
            print(f"[verify] hp {hp}: max dev {w:.2e} (<1e-2 OK)", flush=True)
        except Exception as e:
            print(f"[verify] hp {hp}: WARNING {e!r}", flush=True)
    print(f"[verify] worst dev over {min(n_verify, len(srcs))} healpix: {worst:.2e}",
          flush=True)
    return truth_path
