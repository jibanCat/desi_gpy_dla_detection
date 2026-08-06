#!/usr/bin/env python
"""gen_phaseC_resp.py — Phase-C high-N response-calibration arm generator.

Builds an ANCHORED injection arm on the 2LPT-0 (loa-124) substrate per the
FROZEN design `docs/PHASEC_CALIB_DESIGN.md` (+ the bridge doc): injections at
fixed true-N anchors × z anchors × native-SNR strata, one per sightline,
through the committed, round-trip-validated coadd injection path
(`injection/coadd_injection.py`). The finder is NEVER modified; the GP re-run
uses the UNMODIFIED production config (phaseC_resp_gl_v1.env).

SUBSTRATE MODES (design §4)
---------------------------
--substrate prodlike (production default): BAL-veto only (zcat − bal_cat);
  sightlines MAY carry truth HCDs elsewhere in the forest; a draw whose
  injected z falls within ±--dv-excl km/s of ANY truth HCD on that sightline
  is DROPPED (reported; the production generator will redraw instead — the
  drop rate is a pilot measurable). Unambiguous truth ownership inside the
  matching window is preserved; environments stay production-like.
--substrate clean: the M3 fully-clean table (zcat − hcd − bal) — the
  environment-consistency sub-arm.

ROLES: the manifest schema is the FROZEN M3 contract (no role field), so
roles travel in a `roles.json` SIDECAR keyed by inj_id, written next to the
truth manifest. This arm stamps every row with --role (default
pilot-validation).

PAD–FP BOUNDARY (§5 of the rulings): anchors below logN 19.5 are REFUSED at
argument-parse time. This generator measures the high-N response; it is not
the prohibited pad–FP campaign.

Usage (pilot, on-node, minutes):
  python injection/gen_phaseC_resp.py --out <arm> --substrate prodlike \
      --n-per-cell 4 --n-healpix 6 --seed 20260806
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from astropy.table import Table

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from coadd_injection import (  # noqa: E402
    build_clean_table, write_campaign, verify_coadd_consistency)
from campaign_grid import build_injection_grid, validate_manifest  # noqa: E402
from gen_wall1_inject import write_injected_truth  # noqa: E402

DEFAULT_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                   "qq_desi_y3/v2.8.5/mock-0/loa-124")

#: Phase-C pilot anchor sets (design §3; production uses every 0.2-dex bin —
#: the pilot samples the support: 2 bridge, 3 boundary/clamped, 1 ceiling,
#: 1 above-ceiling anchors).
PILOT_LOGN_ANCHORS = (19.6, 20.0, 20.6, 21.0, 21.2, 21.6, 22.0)
#: one z anchor per response z cell (edges [0, 2.56, 2.96, inf))
PILOT_Z_ANCHORS = (2.30, 2.75, 3.20)
#: the response SNR strata edges (native red-side SNR)
RESP_SNR_BINS = (2.0, 3.5, 6.5, np.inf)

_C_KMS = 299792.458


def build_prodlike_table(mockdir, snr_cut=2.0):
    """BAL-veto-only sightline table (truth HCDs allowed elsewhere)."""
    zcat = Table.read(f"{mockdir}/zcat.fits")
    bal = Table.read(f"{mockdir}/bal_cat.fits")
    snr_cat = Table.read(f"{mockdir}/snr_cat.fits")
    hcd = Table.read(f"{mockdir}/hcd_truth_cat.fits")
    # reuse the M3 builder with an EMPTY hcd subtraction (BAL veto only):
    empty_hcd = hcd[:0]
    return build_clean_table(zcat, empty_hcd, bal, snr_cat), hcd


def veto_hcd_neighbors(manifest, hcd, dv_excl_kms):
    """Drop rows whose injected z sits within dv_excl of a truth HCD on the
    SAME sightline. Returns (kept_rows, n_dropped)."""
    tid_arr = np.asarray(hcd["TARGETID"], np.int64)
    z_arr = np.asarray(hcd["Z"], float)
    order = np.argsort(tid_arr, kind="stable")
    tid_s, z_s = tid_arr[order], z_arr[order]
    starts = np.searchsorted(tid_s, np.unique(tid_s))
    uniq = np.unique(tid_s)
    bounds = dict(zip(uniq.tolist(),
                      zip(starts.tolist(),
                          np.append(starts[1:], len(tid_s)).tolist())))
    kept, dropped = [], 0
    for r in manifest:
        tid = int(r["target_id"])
        b = bounds.get(tid)
        if b is None:
            kept.append(r)
            continue
        zh = z_s[b[0]:b[1]]
        dv = np.abs(zh - float(r["z_true"])) / (1.0 + float(r["z_true"])) * _C_KMS
        if np.any(dv < dv_excl_kms):
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mockdir", default=DEFAULT_MOCKDIR)
    ap.add_argument("--substrate", choices=("prodlike", "clean"),
                    default="prodlike")
    ap.add_argument("--anchors", type=float, nargs="+",
                    default=list(PILOT_LOGN_ANCHORS))
    ap.add_argument("--z-anchors", type=float, nargs="+",
                    default=list(PILOT_Z_ANCHORS))
    ap.add_argument("--n-per-cell", type=int, default=4)
    ap.add_argument("--n-healpix", type=int, default=0,
                    help="0 = all; pilot restricts for cheap trees")
    ap.add_argument("--dv-excl", type=float, default=5000.0,
                    help="km/s truth-HCD exclusion around z_inj (prodlike)")
    ap.add_argument("--snr_cut", type=float, default=2.0)
    ap.add_argument("--num_lines", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--role", default="pilot-validation")
    a = ap.parse_args()

    if min(a.anchors) < 19.5:
        raise SystemExit(
            f"REFUSED: anchor {min(a.anchors)} < 19.5. Injections below 19.5 "
            "are outside the Phase-C authorization (PI §5: the pad–FP "
            "campaign stays prohibited; a new PI ruling is required).")

    print(f"[phaseC] substrate={a.substrate} mockdir={a.mockdir}", flush=True)
    if a.substrate == "prodlike":
        clean, hcd = build_prodlike_table(a.mockdir, snr_cut=a.snr_cut)
    else:
        zcat = Table.read(f"{a.mockdir}/zcat.fits")
        hcd = Table.read(f"{a.mockdir}/hcd_truth_cat.fits")
        bal = Table.read(f"{a.mockdir}/bal_cat.fits")
        snr_cat = Table.read(f"{a.mockdir}/snr_cat.fits")
        clean = build_clean_table(zcat, hcd, bal, snr_cat)
    rs = np.asarray(clean["SNR_REDSIDE"], float)
    clean = clean[np.isfinite(rs) & (rs > a.snr_cut)]

    if a.n_healpix:
        hpx = np.asarray(clean["HEALPIX"], np.int64)
        zq = np.asarray(clean["Z"], float)
        u = np.unique(hpx)
        cnt = np.array([(hpx == h).sum() for h in u])
        floor = max(20, int(np.median(cnt) * 0.25))
        cand = u[cnt >= floor]
        if cand.size < a.n_healpix:
            cand = u
        med = np.array([np.median(zq[hpx == h]) for h in cand])
        order = np.argsort(med)
        pick = np.unique(np.linspace(0, order.size - 1,
                                     a.n_healpix).round().astype(int))
        clean = clean[np.isin(hpx, cand[order[pick]])]
    print(f"[substrate] {len(clean)} eligible sightlines on "
          f"{len(set(clean['HEALPIX'].tolist()))} healpix", flush=True)

    clean_sl = dict(
        target_id=np.asarray(clean["TARGETID"], np.int64),
        healpix=np.asarray(clean["HEALPIX"], np.int64),
        z_qso=np.asarray(clean["Z"], float),
        native_snr=np.asarray(clean["SNR_REDSIDE"], float))

    manifest = build_injection_grid(
        clean_sl,
        logN_grid=list(a.anchors),
        z_grid=list(a.z_anchors),
        snr_bins=list(RESP_SNR_BINS),
        n_per_cell=a.n_per_cell,
        seed=a.seed, campaign="A", method="coadd",
        num_lines=a.num_lines)
    validate_manifest(manifest)
    n0 = len(manifest)
    n_veto = 0
    if a.substrate == "prodlike":
        manifest, n_veto = veto_hcd_neighbors(manifest, hcd, a.dv_excl)
        print(f"[veto] {n_veto}/{n0} draws within {a.dv_excl:.0f} km/s of a "
              f"truth HCD -> dropped (pilot measurable; production redraws)",
              flush=True)
    if not manifest:
        raise SystemExit("[phaseC] ERROR: zero injections after veto")
    nlt = np.array([r["logN_true"] for r in manifest])
    print(f"[manifest] {len(manifest)} injections; anchors "
          f"{sorted(set(np.round(nlt, 2).tolist()))}", flush=True)

    truth_path = write_campaign(manifest, clean, out_root=a.out,
                                mockdir=a.mockdir, num_lines=a.num_lines)
    n = len(glob.glob(f"{a.out}/spectra-16/*/*/spectra-16-*.fits"))
    print(f"[write] {n} injected coadds -> {a.out}/spectra-16/", flush=True)

    inj_truth = write_injected_truth(manifest, a.out)
    print(f"[truth] {inj_truth}", flush=True)

    # roles sidecar (manifest schema is frozen; roles must not mutate it)
    roles = {int(r["inj_id"]): {"role": a.role, "substrate": a.substrate,
                                "seed": a.seed} for r in manifest}
    roles_path = os.path.join(a.out, "roles.json")
    with open(roles_path, "w") as fh:
        json.dump({"schema": "phaseC_roles/v1", "dv_excl_kms": a.dv_excl,
                   "n_vetoed": n_veto, "roles": roles}, fh, indent=0)
    print(f"[roles] {roles_path}", flush=True)

    zc = Table.read(f"{a.mockdir}/zcat.fits")
    want = np.array(sorted(int(r["target_id"]) for r in manifest), np.int64)
    keep = np.isin(np.asarray(zc["TARGETID"], np.int64), want)
    qpath = os.path.join(a.out, "pilot_qsocat.fits")
    zc[keep].write(qpath, overwrite=True)
    print(f"[qsocat] {keep.sum()} targets -> {qpath}", flush=True)

    inj_by_tid = {}
    for r in manifest:
        inj_by_tid.setdefault(int(r["target_id"]), []).append(
            (10.0 ** float(r["logN_true"]), float(r["z_true"])))
    srcs = sorted(glob.glob(f"{a.out}/spectra-16/*/*/spectra-16-*.fits"))
    worst = 0.0
    for src in srcs[:3]:
        hp = int(src.rsplit("-", 1)[1].split(".")[0])
        orig = f"{a.mockdir}/spectra-16/{hp // 100}/{hp}/spectra-16-{hp}.fits"
        try:
            w = verify_coadd_consistency(orig, src, inj_by_tid,
                                         num_lines=a.num_lines)
            worst = max(worst, w)
            print(f"[verify] hp {hp}: max dev {w:.2e} (<1e-2 OK)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[verify] hp {hp}: WARNING: {e!r}", flush=True)
    print(f"[verify] worst dev: {worst:.2e}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
