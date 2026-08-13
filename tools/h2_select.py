#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic, candidate-blind H2 injection-campaign selector (PREP ONLY).

Implements the bounded H2 protocol frozen in the notes repo
(2026-08-12_fable_floor_optimum_and_highz.md §7, "Bounded H2 protocol
(prepared; NOT launched)"). This script prepares the sightline manifest and
the base injection plan; it does NOT inject anything and does NOT run the GP.
H2 SCIENTIFIC EXECUTION REMAINS PI-GATED.

Blindness (protocol requirement, frozen BEFORE any manual review of the
high-z contact sheets): the ONLY input is the QSO parent catalog. No absorber
catalog, no candidate positions/quality, no metal confirmations, no visual
inspection, no pilot posteriors enter the selection. TARGETID order is
arbitrary w.r.t. every scientific property (pilot precedent,
tools/hz_pilot_select.py), so lowest-TARGETID-per-cell is blind by
construction.

Population and strata (protocol):
  arm A: 4.00 <= z_qso <= 4.25  (the searched population)
  arm B: 4.25 <  z_qso <= 7.00  (the post-recovery high-z population)
  x TSNR2_LYA terciles, edges computed per arm on that arm's population
  (arm B reproduces the pilot's tercile edges; asserted against the pilot
  summary when provided).
  => 6 (arm, tercile) macro-cells; per-cell counts from the campaign config
  (~150 injections per macro-cell, <= 2 per sightline, ~900 total).

Per-injection base draw (deterministic; seeded from TARGETID so the plan is
reproducible byte-for-byte):
  z_inj  ~ uniform in [max(3.8, z_forest_min), z_qso - 3000 km/s collar]
           intersected with the protocol range [3.8, 5.0]
  logN   ~ campaign-config grid 19.5..21.5 step 0.25, weights from the
           config (concentration 19.8-20.6 per protocol; exact weights are a
           PI declaration — see config "status" field)
The pre-declared existing-absorber rule (re-draw z_inj if within 5,000 km/s
of ANY catalog candidate, collisions logged) is applied AT EXECUTION against
the then-current candidate catalogs via --collision-cat; the base plan
records collision_status=PENDING.

Outputs (real-survey TARGETIDs -> NOT committed to Git; write to the GL prep
tree): h2_sightlines.csv, h2_injection_plan.csv, h2_summary.json (with the
canonical sha256 of the sorted plan).
"""
import argparse
import hashlib
import json
import os

import numpy as np

Z_ARM_A = (4.00, 4.25)
Z_ARM_B = (4.25, 7.00)
Z_INJ_RANGE = (3.8, 5.0)
COLLAR_KMS = 3000.0          # production Lya search collar (frozen convention)
COLLISION_KMS = 5000.0       # pre-declared existing-absorber re-draw radius
CKMS = 299792.458


def _seed_for(targetid, k, salt="h2v1"):
    h = hashlib.sha256(f"{salt}:{int(targetid)}:{int(k)}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _draw_one(targetid, k, z_qso, logn_grid, logn_weights,
              seg_frac=None, seg_edge=4.25, seg_min_width=0.03,
              salt="h2v1"):
    """Deterministic per-(TARGETID, k) draw.

    seg_frac=None reproduces the H2-v1 design exactly: z_inj uniform in the
    sightline window. With seg_frac=q (H2-v2, design-only variance
    allocation, recorded in the plan): when the window has a usable segment
    above seg_edge (width >= seg_min_width), draw the segment with
    probability q for [seg_edge, zhi] and 1-q for [zlo, seg_edge], uniform
    within the chosen segment. Sightlines without a usable high segment
    keep the uniform draw (physical path is never manufactured).
    """
    rng = np.random.default_rng(_seed_for(targetid, k, salt=salt))
    zlo = Z_INJ_RANGE[0]
    zhi = min(float(z_qso) - (COLLAR_KMS / CKMS) * (1.0 + float(z_qso)),
              Z_INJ_RANGE[1])
    if zhi <= zlo:
        return None
    segment = "uniform"
    if seg_frac is not None and (zhi - seg_edge) >= seg_min_width:
        if rng.uniform() < seg_frac:
            z_inj = float(rng.uniform(seg_edge, zhi))
            segment = "hi"
        else:
            z_inj = float(rng.uniform(zlo, seg_edge))
            segment = "lo"
    else:
        z_inj = float(rng.uniform(zlo, zhi))
    p = np.asarray(logn_weights, float)
    logn = float(rng.choice(logn_grid, p=p / p.sum()))
    return z_inj, logn, segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qsocat", required=True,
                    help="QSO parent catalog FITS (the ONLY selection input)")
    ap.add_argument("--config", required=True,
                    help="h2 campaign config JSON (grid/weights/counts)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pilot-summary", default=None,
                    help="pilot_summary.json; if given, assert arm-B tercile "
                         "edges reproduce the pilot's (provenance cross-check)")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)
    logn_grid = np.asarray(cfg["logN_grid"], float)
    logn_weights = np.asarray(cfg["logN_weights"], float)
    if len(logn_weights) != len(logn_grid):
        raise SystemExit("FATAL: logN_weights length != logN_grid length")
    # v2 extensions (all optional; absent keys reproduce the v1 design):
    #   arm_overrides: {"A": {"sightlines_per_cell", "double_..."},
    #                   "B": {..., "z_substrata": [edges]}}
    #   zinj: {"segment_frac": q, "segment_edge": 4.25,
    #          "segment_min_width": 0.03}
    #   seed_salt: distinct salt per campaign version ("h2v1" default)
    overrides = cfg.get("arm_overrides", {})
    zinj = cfg.get("zinj", {})
    seg_frac = zinj.get("segment_frac")
    seg_edge = float(zinj.get("segment_edge", 4.25))
    seg_minw = float(zinj.get("segment_min_width", 0.03))
    salt = cfg.get("seed_salt", "h2v1")

    def cell_counts(arm):
        o = overrides.get(arm, {})
        ns = int(o.get("sightlines_per_cell", cfg["sightlines_per_cell"]))
        nd = int(o.get("double_injection_sightlines_per_cell",
                       cfg["double_injection_sightlines_per_cell"]))
        if nd > ns:
            raise SystemExit(f"FATAL: arm {arm} double count exceeds cell size")
        return ns, nd

    from astropy.io import fits
    if not os.path.isfile(args.qsocat):
        raise SystemExit(f"FATAL: QSO catalog not found: {args.qsocat}")
    q = fits.open(args.qsocat)[1].data

    arms = {"A": Z_ARM_A, "B": Z_ARM_B}
    sight_rows, plan_rows = [], []
    summary_cells = {}
    tercile_edges = {}
    for arm, (zlo, zhi) in arms.items():
        if arm == "A":
            m = (q["Z"] >= zlo) & (q["Z"] <= zhi)
        else:
            m = (q["Z"] > zlo) & (q["Z"] <= zhi)
        z = np.asarray(q["Z"][m], float)
        ts = np.asarray(q["TSNR2_LYA"][m], float)
        tid = np.asarray(q["TARGETID"][m], np.int64)
        pix = np.asarray(q["HPXPIXEL"][m], np.int64)
        edges = np.percentile(ts, [100 / 3, 200 / 3])
        tercile_edges[arm] = [float(x) for x in edges]
        n_sight, n_double = cell_counts(arm)
        substrata = overrides.get(arm, {}).get("z_substrata")
        for ti in range(3):
            if ti == 0:
                mm = ts < edges[0]
            elif ti == 1:
                mm = (ts >= edges[0]) & (ts < edges[1])
            else:
                mm = ts >= edges[1]
            if substrata:
                # lowest-TARGETID per (z-substratum x tercile): blind,
                # deterministic; guarantees z_qso coverage inside the cell.
                nz = len(substrata) - 1
                per = n_sight // nz
                if per * nz != n_sight:
                    raise SystemExit(
                        f"FATAL: arm {arm} sightlines_per_cell {n_sight} "
                        f"not divisible by {nz} z-substrata")
                parts = []
                for zi in range(nz):
                    zm = mm & (z >= substrata[zi]) & (z < substrata[zi + 1])
                    avail = np.sort(tid[zm])
                    if len(avail) < per:
                        raise SystemExit(
                            f"FATAL: arm {arm} t{ti} z-substratum "
                            f"[{substrata[zi]},{substrata[zi+1]}) has only "
                            f"{len(avail)} parents < {per}")
                    parts.append(avail[:per])
                picks = np.sort(np.concatenate(parts))
            else:
                picks = np.sort(tid[mm])[:n_sight]
            cell = f"{arm}_t{ti}"
            n_inj_cell = 0
            for rank, t in enumerate(picks):
                j = int(np.where(tid == t)[0][0])
                n_inj = 2 if rank < n_double else 1
                sight_rows.append(dict(
                    TARGETID=int(t), Z_QSO=float(z[j]),
                    TSNR2_LYA=float(ts[j]), HPXPIXEL=int(pix[j]),
                    arm=arm, tercile=ti, cell=cell, n_inj=n_inj))
                for k in range(n_inj):
                    d = _draw_one(t, k, z[j], logn_grid, logn_weights,
                                  seg_frac=seg_frac, seg_edge=seg_edge,
                                  seg_min_width=seg_minw, salt=salt)
                    if d is None:
                        continue
                    z_inj, logn, segment = d
                    plan_rows.append(dict(
                        TARGETID=int(t), inj_idx=k, cell=cell,
                        Z_QSO=float(z[j]), HPXPIXEL=int(pix[j]),
                        z_inj=round(z_inj, 6), logN=logn,
                        z_segment=segment,
                        collision_status="PENDING"))
                    n_inj_cell += 1
            summary_cells[cell] = dict(n_sightlines=len(picks),
                                       n_injections=n_inj_cell)

    if args.pilot_summary:
        with open(args.pilot_summary) as fh:
            ps = json.load(fh)
        got = np.asarray(tercile_edges["B"], float)
        want = np.asarray(ps["tsnr_tercile_edges"], float)
        if not np.allclose(got, want, rtol=0, atol=1e-6):
            raise SystemExit(
                f"FATAL: arm-B tercile edges {got} do not reproduce the "
                f"pilot's {want} — population or catalog drift; STOP")

    os.makedirs(args.outdir, exist_ok=True)
    import csv
    with open(os.path.join(args.outdir, "h2_sightlines.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sight_rows[0]))
        w.writeheader()
        for r in sorted(sight_rows, key=lambda r: r["TARGETID"]):
            w.writerow(r)
    with open(os.path.join(args.outdir, "h2_injection_plan.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(plan_rows[0]))
        w.writeheader()
        for r in sorted(plan_rows,
                        key=lambda r: (r["TARGETID"], r["inj_idx"])):
            w.writerow(r)

    canon = "\n".join(
        f"{r['TARGETID']}:{r['inj_idx']}:{r['z_inj']:.6f}:{r['logN']:.2f}"
        for r in sorted(plan_rows,
                        key=lambda r: (r["TARGETID"], r["inj_idx"])))
    plan_sha = hashlib.sha256(canon.encode()).hexdigest()
    summary = dict(
        protocol="notes 2026-08-12_fable_floor_optimum_and_highz.md §7 "
                 "(bounded H2; prepared, NOT launched)",
        config=cfg,
        n_sightlines=len(sight_rows),
        n_injections=len(plan_rows),
        cells=summary_cells,
        tercile_edges=tercile_edges,
        collar_kms=COLLAR_KMS,
        collision_kms=COLLISION_KMS,
        z_inj_range=list(Z_INJ_RANGE),
        plan_sha256=plan_sha,
        execution_gate="H2 scientific execution is PI-gated; this output is "
                       "preparation only",
    )
    with open(os.path.join(args.outdir, "h2_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"H2 PREP: {len(sight_rows)} sightlines, {len(plan_rows)} "
          f"injections, plan sha256 {plan_sha[:16]}…  -> {args.outdir}")


if __name__ == "__main__":
    main()
