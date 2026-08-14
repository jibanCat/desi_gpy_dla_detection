#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Freeze the H2 execution-time collision catalogue and realize the plan.

PI ruling 2026-08-13 §§20-22 (H2-v2 APPROVED; this is the execution-time
input freeze, not a design step). Applies the PRE-DECLARED collision rule
exactly as frozen in the decision sheets:

  collision := any absorber row for the sightline in the frozen candidate
  list within 5,000 km/s of z_inj (ANY P_DLA — probability never enters);
  on collision: deterministic redraw from the same per-injection seeded
  stream (same two-segment/uniform design density), up to 100 attempts;
  exhaustion -> DROP that injection (logged); logN is never redrawn.

Inputs: the frozen H2-v2 plan + sightlines, and the candidate catalogues
(the full provisional high-z production dlacat FITS tree + the low-z
production CDDF catalogue for arm A). Outputs (all sha-recorded):
  h2_collision_catalog.csv   frozen per-sightline candidate z list
  h2_realized_plan.csv       final accepted injections (+ segment, attempts)
  h2_collision_log.csv       every rejected draw
  h2_freeze_summary.json     shas + counts + per-stratum attrition
"""
import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np

CKMS = 299792.458
COLLISION_KMS = 5000.0
Z_INJ_RANGE = (3.8, 5.0)
COLLAR_KMS = 3000.0
MAX_ATTEMPTS = 100


def _seed_for(targetid, k, salt):
    h = hashlib.sha256(f"{salt}:{int(targetid)}:{int(k)}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def collides(z, cands):
    for zc in cands:
        if abs(CKMS * (z - zc) / (1.0 + zc)) < COLLISION_KMS:
            return zc
    return None


def realize_one(tid, k, z_qso, cands, seg_frac, seg_edge, seg_minw,
                logn_grid, logn_weights, salt, log_rows):
    """Deterministic realization: same seeded stream as the base draw, with
    rejected z draws logged and redrawn. Returns (z_inj, logN, segment,
    attempts) or None if dropped."""
    rng = np.random.default_rng(_seed_for(tid, k, salt))
    zlo = Z_INJ_RANGE[0]
    zhi = min(float(z_qso) - (COLLAR_KMS / CKMS) * (1.0 + float(z_qso)),
              Z_INJ_RANGE[1])
    if zhi <= zlo:
        return None
    has_seg = seg_frac is not None and (zhi - seg_edge) >= seg_minw
    # logN drawn ONCE from the stream position matching the frozen selector:
    # selector order was [segment-choice?] z-draw, logN-draw. To keep the
    # accepted (z, logN) of collision-free injections IDENTICAL to the frozen
    # plan, replay draws in the same order; on collision, continue the stream.
    z_inj = segment = None
    attempts = 0
    logn = None
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        if has_seg:
            if rng.uniform() < seg_frac:
                z_try, seg = float(rng.uniform(seg_edge, zhi)), "hi"
            else:
                z_try, seg = float(rng.uniform(zlo, seg_edge)), "lo"
        else:
            z_try, seg = float(rng.uniform(zlo, zhi)), "uniform"
        if logn is None:
            p = np.asarray(logn_weights, float)
            logn = float(rng.choice(logn_grid, p=p / p.sum()))
        zc = collides(z_try, cands)
        if zc is None:
            z_inj, segment = z_try, seg
            break
        log_rows.append(dict(TARGETID=tid, inj_idx=k, attempt=attempts,
                             z_rejected=round(z_try, 6),
                             colliding_z=round(zc, 6)))
    if z_inj is None:
        return None
    return z_inj, logn, segment, attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="frozen h2_injection_plan.csv")
    ap.add_argument("--sightlines", required=True)
    ap.add_argument("--config", required=True, help="h2 campaign config json")
    ap.add_argument("--hz-dlacat-glob", required=True,
                    help="glob of the full provisional high-z production "
                         "dlacat FITS (arm-B candidate source)")
    ap.add_argument("--lowz-dlacat", required=True,
                    help="production CDDF catalogue FITS (arm-A candidates)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from astropy.io import fits

    cfg = json.load(open(args.config))
    zinj = cfg.get("zinj", {})
    seg_frac = zinj.get("segment_frac")
    seg_edge = float(zinj.get("segment_edge", 4.25))
    seg_minw = float(zinj.get("segment_min_width", 0.03))
    salt = cfg.get("seed_salt", "h2v1")
    logn_grid = np.asarray(cfg["logN_grid"], float)
    logn_weights = np.asarray(cfg["logN_weights"], float)

    sl = {int(r["TARGETID"]): r
          for r in csv.DictReader(open(args.sightlines))}
    plan = list(csv.DictReader(open(args.plan)))

    # ---- frozen collision catalogue -------------------------------------
    cand = {}
    hz_files = sorted(glob.glob(args.hz_dlacat_glob))
    if not hz_files:
        raise SystemExit(f"FATAL: no hz dlacat files match {args.hz_dlacat_glob}")
    for f in hz_files:
        d = fits.open(f)[1].data
        for r in d:
            cand.setdefault(int(r["TARGETID"]), set()).add(
                round(float(r["Z_DLA"]), 6))
    lz = fits.open(args.lowz_dlacat)[1].data
    for r in lz:
        cand.setdefault(int(r["TARGETID"]), set()).add(
            round(float(r["Z_DLA"]), 6))

    os.makedirs(args.outdir, exist_ok=True)
    cat_path = os.path.join(args.outdir, "h2_collision_catalog.csv")
    with open(cat_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["TARGETID", "z_cand"])
        for t in sorted(sl):
            for z in sorted(cand.get(t, [])):
                w.writerow([t, f"{z:.6f}"])

    # ---- deterministic realization --------------------------------------
    log_rows, realized, dropped = [], [], []
    for row in plan:
        t, k = int(row["TARGETID"]), int(row["inj_idx"])
        zq = float(row["Z_QSO"])
        out = realize_one(t, k, zq, sorted(cand.get(t, [])), seg_frac,
                          seg_edge, seg_minw, logn_grid, logn_weights,
                          salt, log_rows)
        if out is None:
            dropped.append(dict(TARGETID=t, inj_idx=k))
            continue
        z_inj, logn, segment, attempts = out
        assert abs(logn - float(row["logN"])) < 1e-9, \
            f"logN drift for {t}:{k} — stream misalignment"
        if attempts == 1:
            assert abs(z_inj - float(row["z_inj"])) < 1e-6, \
                f"collision-free draw differs from frozen plan for {t}:{k}"
        realized.append(dict(TARGETID=t, inj_idx=k, cell=row["cell"],
                             Z_QSO=zq, HPXPIXEL=int(row["HPXPIXEL"]),
                             z_inj=round(z_inj, 6), logN=logn,
                             z_segment=segment, attempts=attempts))

    rp = os.path.join(args.outdir, "h2_realized_plan.csv")
    with open(rp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(realized[0]))
        w.writeheader()
        for r in sorted(realized, key=lambda r: (r["TARGETID"], r["inj_idx"])):
            w.writerow(r)
    lp = os.path.join(args.outdir, "h2_collision_log.csv")
    with open(lp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["TARGETID", "inj_idx", "attempt",
                                           "z_rejected", "colliding_z"])
        w.writeheader()
        for r in log_rows:
            w.writerow(r)

    canon = "\n".join(f"{r['TARGETID']}:{r['inj_idx']}:{r['z_inj']:.6f}:"
                      f"{r['logN']:.2f}"
                      for r in sorted(realized,
                                      key=lambda r: (r["TARGETID"],
                                                     r["inj_idx"])))
    n_coll = len({(r["TARGETID"], r["inj_idx"]) for r in log_rows})
    summary = dict(
        n_plan=len(plan), n_realized=len(realized), n_dropped=len(dropped),
        dropped=dropped,
        n_injections_with_any_collision=n_coll,
        n_redraw_events=len(log_rows),
        collision_catalog_sha256=_sha_file(cat_path),
        realized_plan_sha256=hashlib.sha256(canon.encode()).hexdigest(),
        collision_log_sha256=_sha_file(lp),
        inputs=dict(plan=args.plan, config=args.config,
                    hz_dlacat_glob=args.hz_dlacat_glob,
                    lowz_dlacat=args.lowz_dlacat,
                    lowz_dlacat_sha256=_sha_file(args.lowz_dlacat)),
        rule="5000 km/s any-P collision; deterministic redraw <=100; "
             "drop+log on exhaustion; logN never redrawn",
    )
    json.dump(summary, open(os.path.join(args.outdir,
                                         "h2_freeze_summary.json"), "w"),
              indent=1)
    print(f"H2 FREEZE: {len(realized)}/{len(plan)} realized, "
          f"{len(dropped)} dropped, {n_coll} injections collided "
          f"({len(log_rows)} redraw events); realized sha "
          f"{summary['realized_plan_sha256'][:16]}…")


if __name__ == "__main__":
    main()
