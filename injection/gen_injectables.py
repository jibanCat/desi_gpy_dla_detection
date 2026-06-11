#!/usr/bin/env python
"""Generate an injectable tree: clean-select -> grid (dense NHI<19) -> inject -> write.

Reproduce step 1 of the injection campaign (see injection/README.md). Also writes a
``pilot_qsocat.fits`` restricted to ONLY the injected/control TARGETIDs (so the GP run
processes the injections, not all ~8600 fibers/healpix), and a sample
coadd-consistency check.
"""
import argparse, os, sys, glob
import numpy as np
from astropy.table import Table

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # repo root (gpy_dla_detection)
sys.path.insert(0, _HERE)                        # injection modules
from coadd_injection import build_clean_table, write_campaign, verify_coadd_consistency
from campaign_grid import (build_injection_grid, build_control_rows, validate_manifest)

DEFAULT_MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                "qq_desi_y3/v2.8.5/mock-0/loa-124")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="injectable-tree root")
    ap.add_argument("--mockdir", default=DEFAULT_MOCK)
    ap.add_argument("--target_injections", type=int, default=150)
    ap.add_argument("--n_controls", type=int, default=50)
    ap.add_argument("--n_healpix", type=int, default=0, help="0 = all clean healpix")
    ap.add_argument("--snr_cut", type=float, default=2.0, help="SNR_REDSIDE > cut")
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--snr_bins", type=float, nargs="+", default=[2.0, 4.0, 8.0, 1e9])
    a = ap.parse_args()

    D = a.mockdir
    print("[load] catalogs ...", flush=True)
    clean = build_clean_table(Table.read(f"{D}/zcat.fits"),
                              Table.read(f"{D}/hcd_truth_cat.fits"),
                              Table.read(f"{D}/bal_cat.fits"),
                              Table.read(f"{D}/snr_cat.fits"))
    rs = np.asarray(clean["SNR_REDSIDE"], float)
    clean = clean[np.isfinite(rs) & (rs > a.snr_cut)]   # red-side SNR cut (DLA-uncorrelated)
    if a.n_healpix:
        hp = np.asarray(clean["HEALPIX"], np.int64)
        u, c = np.unique(hp, return_counts=True)
        top = u[np.argsort(c)[::-1][:a.n_healpix]]
        clean = clean[np.isin(hp, top)]
    print(f"[clean] {len(clean)} sightlines on {len(set(clean['HEALPIX'].tolist()))} healpix", flush=True)

    clean_sl = dict(target_id=np.asarray(clean["TARGETID"], np.int64),
                    healpix=np.asarray(clean["HEALPIX"], np.int64),
                    z_qso=np.asarray(clean["Z"], float),
                    native_snr=np.asarray(clean["SNR_REDSIDE"], float))  # RED-SIDE
    inj = build_injection_grid(clean_sl, snr_bins=a.snr_bins,
                               target_injections=a.target_injections, seed=a.seed,
                               campaign="A", method="coadd", num_lines=a.num_lines)
    ctrl = build_control_rows(clean_sl, snr_bins=a.snr_bins, target_controls=a.n_controls,
                              seed=a.seed + 1, inj_id_start=len(inj))
    manifest = list(inj) + list(ctrl)
    validate_manifest(manifest)
    nlt = np.array([r["logN_true"] for r in inj])
    print(f"[manifest] {len(inj)} inj + {len(ctrl)} ctrl; logN [{nlt.min():.2f},{nlt.max():.2f}], "
          f"frac<19={np.mean(nlt<19):.2f}", flush=True)

    truth_path = write_campaign(manifest, clean, out_root=a.out, mockdir=D, num_lines=a.num_lines)
    n = len(glob.glob(f"{a.out}/spectra-16/*/*/spectra-16-*.fits"))
    print(f"[write] {n} injected coadds -> {a.out}/spectra-16/ ; truth -> {truth_path}", flush=True)

    # restricted qsocat = ONLY injected/control targets (keeps GP run cheap)
    zc = Table.read(f"{D}/zcat.fits")
    want = set(int(r["target_id"]) for r in manifest)
    keep = np.isin(np.asarray(zc["TARGETID"], np.int64),
                   np.array(sorted(want), np.int64))
    qpath = os.path.join(a.out, "pilot_qsocat.fits")
    zc[keep].write(qpath, overwrite=True)
    print(f"[qsocat] {keep.sum()} injected targets -> {qpath}", flush=True)

    # sample coadd-consistency check (per-camera injection survives coadd_cameras)
    inj_by_tid = {}
    for r in inj:
        inj_by_tid.setdefault(int(r["target_id"]), []).append(
            (10.0 ** float(r["logN_true"]), float(r["z_true"])))
    src = sorted(glob.glob(f"{a.out}/spectra-16/*/*/spectra-16-*.fits"))[0]
    hp = int(src.rsplit("-", 1)[1].split(".")[0])
    orig = f"{D}/spectra-16/{hp // 100}/{hp}/spectra-16-{hp}.fits"
    sub = {t: v for t, v in inj_by_tid.items()}
    try:
        worst = verify_coadd_consistency(orig, src, sub, num_lines=a.num_lines)
        print(f"[verify] coadd_cameras(injected)==T*coadd(original): max dev {worst:.2e} (<1e-2 OK)", flush=True)
    except Exception as e:
        print(f"[verify] WARNING: {e!r}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
