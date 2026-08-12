#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic, candidate-blind high-z pilot selector (PI §26).

Selection rule (pre-declared, no randomness, no candidate information):
stratify the never-searched z_qso > 4.25 population into
  z_qso bins  [4.25,4.4) [4.4,4.6) [4.6,4.9) [4.9,7.0)
  × TSNR2_LYA terciles (tercile edges computed on the z>4.25 set itself)
and take, in each of the 12 cells, the N (default 4) LOWEST-TARGETID QSOs.
TARGETID order is arbitrary with respect to every scientific property, so
the selection is blind by construction; it uses only the QSO parent
catalog (never the absorber catalog, never visual inspection).

Outputs (NOT committed to Git — real-survey identifiers): the pilot target
list, its healpix list (for --external_hpx_list), and a JSON summary with
the canonical sha256 of the sorted TARGETID list.
"""
import argparse
import hashlib
import json
import os

import numpy as np

Z_EDGES = [4.25, 4.4, 4.6, 4.9, 7.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qsocat", required=True)
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from astropy.io import fits
    q = fits.open(args.qsocat)[1].data
    m = (q["Z"] > 4.25) & (q["Z"] <= 7.0)
    z = np.asarray(q["Z"][m], float)
    ts = np.asarray(q["TSNR2_LYA"][m], float)
    tid = np.asarray(q["TARGETID"][m], np.int64)
    pix = np.asarray(q["HPXPIXEL"][m], np.int64)

    t_edges = np.percentile(ts, [100 / 3, 200 / 3])
    picks = []
    for zi in range(4):
        for ti in range(3):
            mm = (z >= Z_EDGES[zi]) & (z < Z_EDGES[zi + 1])
            if ti == 0:
                mm &= ts < t_edges[0]
            elif ti == 1:
                mm &= (ts >= t_edges[0]) & (ts < t_edges[1])
            else:
                mm &= ts >= t_edges[1]
            ids = np.sort(tid[mm])[: args.per_cell]
            for t in ids:
                j = int(np.where(tid == t)[0][0])
                picks.append(dict(TARGETID=int(t), Z_QSO=float(z[j]),
                                  TSNR2_LYA=float(ts[j]),
                                  HPXPIXEL=int(pix[j]), zcell=zi, tcell=ti))
    picks.sort(key=lambda r: r["TARGETID"])

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "pilot_targets.txt"), "w") as fh:
        for r in picks:
            fh.write(f"{r['TARGETID']}\n")
    upix = sorted({r["HPXPIXEL"] for r in picks})
    with open(os.path.join(args.outdir, "pilot_hpx_list.txt"), "w") as fh:
        for p in upix:
            fh.write(f"{p}\n")
    ident = hashlib.sha256("\n".join(str(r["TARGETID"])
                                     for r in picks).encode()).hexdigest()
    summ = dict(n_pilot=len(picks), n_healpix=len(upix),
                tsnr_tercile_edges=[float(x) for x in t_edges],
                z_edges=Z_EDGES, per_cell=args.per_cell,
                pilot_list_sha256=ident,
                cells={f"z{zi}_t{ti}": sum(1 for r in picks
                                           if r["zcell"] == zi
                                           and r["tcell"] == ti)
                       for zi in range(4) for ti in range(3)})
    json.dump(summ, open(os.path.join(args.outdir,
                                      "pilot_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
