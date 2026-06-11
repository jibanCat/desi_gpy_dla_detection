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
from campaign_grid import (build_injection_grid, build_control_rows, validate_manifest,
                           default_zqso_bins)

DEFAULT_MOCK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                "qq_desi_y3/v2.8.5/mock-0/loa-124")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="injectable-tree root")
    ap.add_argument("--mockdir", default=DEFAULT_MOCK)
    ap.add_argument("--target_injections", type=int, default=150)
    ap.add_argument("--n_per_cell", type=int, default=None,
                    help="sightlines per (logN×z×zqso×snr) cell. Takes precedence over "
                         "--target_injections — the predictable R-density knob (the "
                         "z_QSO×window feasibility makes target_injections under-deliver).")
    ap.add_argument("--n_controls", type=int, default=50)
    ap.add_argument("--n_healpix", type=int, default=0, help="0 = all clean healpix")
    ap.add_argument("--snr_cut", type=float, default=2.0, help="SNR_REDSIDE > cut")
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--snr_bins", type=float, nargs="+", default=[2.0, 4.0, 8.0, 1e9])
    ap.add_argument("--zqso_bins", type=float, nargs="+", default=None,
                    help="host z_QSO stratification edges (default: campaign_grid."
                         "default_zqso_bins(); pass a single 0 to DISABLE).")
    a = ap.parse_args()
    # z_QSO stratification: default-on (spans the forest rest-frame position so the
    # N_HI bias is resolved across λ_rest, not confounded). `--zqso_bins 0` disables.
    if a.zqso_bins is None:
        zqso_bins = list(default_zqso_bins())
    elif len(a.zqso_bins) == 1 and a.zqso_bins[0] == 0:
        zqso_bins = None
    else:
        zqso_bins = list(a.zqso_bins)

    D = a.mockdir
    print("[load] catalogs ...", flush=True)
    clean = build_clean_table(Table.read(f"{D}/zcat.fits"),
                              Table.read(f"{D}/hcd_truth_cat.fits"),
                              Table.read(f"{D}/bal_cat.fits"),
                              Table.read(f"{D}/snr_cat.fits"))
    rs = np.asarray(clean["SNR_REDSIDE"], float)
    clean = clean[np.isfinite(rs) & (rs > a.snr_cut)]   # red-side SNR cut (DLA-uncorrelated)
    if a.n_healpix:
        # Select healpix that SPAN the host z_QSO range (not just the most populous),
        # so a pilot subset still samples the full rest-frame forest position. Rank
        # healpix by median clean z_QSO and pick `n_healpix` evenly across that order;
        # require a floor count so a picked healpix has enough sightlines to draw from.
        hp = np.asarray(clean["HEALPIX"], np.int64)
        zq = np.asarray(clean["Z"], float)
        u = np.unique(hp)
        cnt = np.array([(hp == h).sum() for h in u])
        floor = max(20, int(np.median(cnt) * 0.25))
        cand = u[cnt >= floor]
        if cand.size < a.n_healpix:           # not enough well-populated healpix → use all
            cand = u
        med = np.array([np.median(zq[hp == h]) for h in cand])
        order = np.argsort(med)               # low → high median z_QSO
        pick = np.unique(np.linspace(0, order.size - 1, a.n_healpix).round().astype(int))
        top = cand[order[pick]]
        clean = clean[np.isin(hp, top)]
    zqa = np.asarray(clean["Z"], float)
    print(f"[clean] {len(clean)} sightlines on {len(set(clean['HEALPIX'].tolist()))} healpix; "
          f"z_QSO [{zqa.min():.2f}, {zqa.max():.2f}] median {np.median(zqa):.2f}", flush=True)

    clean_sl = dict(target_id=np.asarray(clean["TARGETID"], np.int64),
                    healpix=np.asarray(clean["HEALPIX"], np.int64),
                    z_qso=np.asarray(clean["Z"], float),
                    native_snr=np.asarray(clean["SNR_REDSIDE"], float))  # RED-SIDE
    # n_per_cell (if given) controls R density directly; else size to target_injections.
    _npc = a.n_per_cell
    _tgt = None if _npc is not None else a.target_injections
    inj = build_injection_grid(clean_sl, snr_bins=a.snr_bins, zqso_bins=zqso_bins,
                               n_per_cell=_npc, target_injections=_tgt, seed=a.seed,
                               campaign="A", method="coadd", num_lines=a.num_lines)
    ctrl = build_control_rows(clean_sl, snr_bins=a.snr_bins, target_controls=a.n_controls,
                              seed=a.seed + 1, inj_id_start=len(inj),
                              exclude_target_ids={int(r["target_id"]) for r in inj},
                              zqso_bins=zqso_bins)
    manifest = list(inj) + list(ctrl)
    validate_manifest(manifest)
    if not inj:
        raise SystemExit("[manifest] ERROR: zero injections built — the clean ∩ "
                         "hostable ∩ SNR-bin pool is empty. Lower --snr_cut, add "
                         "--n_healpix, or widen the grid.")
    nlt = np.array([r["logN_true"] for r in inj])
    print(f"[manifest] {len(inj)} inj + {len(ctrl)} ctrl; logN [{nlt.min():.2f},{nlt.max():.2f}], "
          f"frac<19={np.mean(nlt<19):.2f}", flush=True)
    # Rest-frame forest-position coverage (the confound z_QSO stratification fixes):
    # lam_rest = (1+z_DLA)/(1+z_QSO)*1215.67, spanning Lyman-limit 912 .. Lyα 1216.
    zt = np.array([r["z_true"] for r in inj]); zq = np.array([r["z_qso"] for r in inj])
    lam = (1 + zt) / (1 + zq) * 1215.67
    if zqso_bins is not None:
        nb = np.array([(np.asarray([r["zqso_bin"] for r in inj]) == k).sum()
                       for k in range(len(zqso_bins) - 1)])
        print(f"[restframe] z_QSO bins {zqso_bins} -> per-bin n {nb.tolist()}; "
              f"λ_rest [{lam.min():.0f},{lam.max():.0f}] Å "
              f"(pct10/50/90 = {np.percentile(lam,[10,50,90]).round(0).astype(int).tolist()})",
              flush=True)
    else:
        print(f"[restframe] z_QSO stratification OFF; λ_rest [{lam.min():.0f},{lam.max():.0f}] Å",
              flush=True)
    # Global one-injection-per-target means the achievable total is bounded by the
    # distinct clean (hostable ∩ SNR-bin) pool — never silently undercount the
    # requested budget.  Warn loudly so the driver can widen the clean pool
    # (more healpix / lower SNR cut) rather than ship a thin campaign.
    if a.n_per_cell is None and len(inj) < int(0.95 * a.target_injections):
        print(f"[manifest] WARNING: requested {a.target_injections} injections but the "
              f"clean pool only supports {len(inj)} distinct sightlines "
              f"({100 * len(inj) / a.target_injections:.0f}%). Add healpix (--n_healpix 0 "
              f"= all), lower --snr_cut, or use --n_per_cell (z_QSO×window feasibility "
              f"makes target_injections under-deliver).", flush=True)

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

    # sample coadd-consistency check (per-camera injection survives coadd_cameras).
    # Verify a FEW healpix (not just the first) so a per-healpix grid/resolution
    # quirk can't slip through; the mechanism is identical per file, so a handful
    # is a sufficient smoke without re-reading the whole tree.
    inj_by_tid = {}
    for r in inj:
        inj_by_tid.setdefault(int(r["target_id"]), []).append(
            (10.0 ** float(r["logN_true"]), float(r["z_true"])))
    srcs = sorted(glob.glob(f"{a.out}/spectra-16/*/*/spectra-16-*.fits"))
    n_verify = min(3, len(srcs))
    worst_all = 0.0
    for src in srcs[:n_verify]:
        hp = int(src.rsplit("-", 1)[1].split(".")[0])
        orig = f"{D}/spectra-16/{hp // 100}/{hp}/spectra-16-{hp}.fits"
        try:
            worst = verify_coadd_consistency(orig, src, inj_by_tid, num_lines=a.num_lines)
            worst_all = max(worst_all, worst)
            print(f"[verify] hp {hp}: coadd_cameras(injected)==T*coadd(original) "
                  f"max dev {worst:.2e} (<1e-2 OK)", flush=True)
        except Exception as e:
            print(f"[verify] hp {hp}: WARNING: {e!r}", flush=True)
    print(f"[verify] worst dev over {n_verify} healpix: {worst_all:.2e}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
