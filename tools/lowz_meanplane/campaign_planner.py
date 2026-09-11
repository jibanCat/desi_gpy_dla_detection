"""Shared campaign planner for checkpoint-10.5 diagnostics — replicates the
H2-M design semantics EXACTLY (validated against the recorded realized plan):
  - 9 kernel cells (SNR_REDSIDE edges 2/3.5/6.5/inf x z_QSO edges
    2.1/2.56/2.96/3.79), 60 sightlines/cell, 40 of them doubled -> 100
    injections/cell, 900 total;
  - logN grid 19.5..21.5 step 0.25 with draw weights [1,1.5,3,3,3,1.5,1,1,1];
  - z placement uniform in the lya-only window lambda_rf in [1025,1216] with
    3,000 km/s collars on both edges, capped to z_inj in [1.96, 4.2];
  - collision rule: reject a z within 5,000 km/s of ANY catalog candidate on
    that sightline at ANY P; NO sibling (same-sightline) avoidance —
    deliberately replicated (part of the protocol under test);
  - a sightline whose slot cannot place after MAX_ATTEMPTS draws is dropped
    and replaced from the remaining cell pool (drops recorded).
"""
from __future__ import annotations

import numpy as np

SNR_EDGES = (2.0, 3.5, 6.5, np.inf)
Z_EDGES = (2.1, 2.56, 2.96, 3.79)
LOGN_GRID = (19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5)
LOGN_W = (1.0, 1.5, 3.0, 3.0, 3.0, 1.5, 1.0, 1.0, 1.0)
SL_PER_CELL = 60
DBL_PER_CELL = 40
COLLISION_KMS = 5000.0
COLLAR_KMS = 3000.0
Z_CAP = (1.96, 4.2)
LAM_RF = (1025.0, 1216.0)
LYA = 1215.67
C_KMS = 2.998e5
MAX_ATTEMPTS = 30


def cell_name(snr, zq):
    si = int(np.digitize(snr, SNR_EDGES[1:3]))
    zi = int(np.digitize(zq, Z_EDGES[1:3]))
    return f"s{si}z{zi}"


def z_bounds(zq):
    z_lo = LAM_RF[0] * (1.0 + zq) / LYA - 1.0
    z_hi = LAM_RF[1] * (1.0 + zq) / LYA - 1.0
    coll = COLLAR_KMS / C_KMS
    z_lo = z_lo + (1.0 + z_lo) * coll
    z_hi = z_hi - (1.0 + z_hi) * coll
    return max(z_lo, Z_CAP[0]), min(z_hi, Z_CAP[1])


def collides(z, cand_z):
    for zz in cand_z:
        if abs(z - zz) / (1.0 + min(z, zz)) * C_KMS < COLLISION_KMS:
            return True
    return False


def plan_campaign(parent, cand_z_by_tid, seed, *,
                  sl_per_cell=None, dbl_per_cell=None,
                  logn_grid=None, logn_w=None):
    """Overrides (L8 extension, PI checkpoint 10.7): sl_per_cell,
    dbl_per_cell=0 => singles-only (the ruled threshold-focused design;
    everything else identical to the validated H2-M semantics)."""
    """parent: structured array w/ TARGETID(int64), Z_QSO, RED_SNR, HPXPIXEL.
    cand_z_by_tid: dict tid -> list of candidate z (ANY P) for collisions.
    Returns (sightlines, plan_rows, dropped): lists of dicts."""
    global SL_PER_CELL, DBL_PER_CELL
    _sl = SL_PER_CELL if sl_per_cell is None else sl_per_cell
    _dbl = DBL_PER_CELL if dbl_per_cell is None else dbl_per_cell
    _grid = LOGN_GRID if logn_grid is None else tuple(logn_grid)
    _w = LOGN_W if logn_w is None else tuple(logn_w)
    rng = np.random.default_rng(seed)
    logn_p = np.asarray(_w) / np.sum(_w)
    tids = parent["TARGETID"].astype(np.int64)
    assert tids.dtype == np.int64
    cells = np.array([cell_name(s, z) for s, z in
                      zip(parent["RED_SNR"], parent["Z_QSO"])])
    sightlines, plan_rows, dropped = [], [], []
    for si in range(3):
        for zi in range(3):
            cname = f"s{si}z{zi}"
            pool = np.where(cells == cname)[0]
            rng.shuffle(pool)
            pool = list(pool)
            # shortfall cells: take everything available, keep the 40/60
            # double ratio (recorded; weights handle occupancy at analysis)
            n_target = min(_sl, len(pool))
            n_dbl = _dbl if n_target == _sl else \
                int(round(n_target * _dbl / max(_sl, 1)))
            if n_target < _sl:
                print(f"  [plan] cell {cname}: pool {len(pool)} < "
                      f"{_sl} -> taking {n_target} ({n_dbl} doubled)")
            picked = 0
            while picked < n_target and pool:
                row = parent[pool.pop(0)]
                tid = int(row["TARGETID"])
                zq = float(row["Z_QSO"])
                n_inj = 2 if picked < n_dbl else 1
                zlo, zhi = z_bounds(zq)
                if not (zhi > zlo):
                    dropped.append(dict(TARGETID=tid, reason="empty_window"))
                    continue
                cand = cand_z_by_tid.get(tid, [])
                injs = []
                failed = False
                for k in range(n_inj):
                    placed = False
                    for att in range(1, MAX_ATTEMPTS + 1):
                        logN = float(rng.choice(_grid, p=logn_p))
                        z = float(rng.uniform(zlo, zhi))
                        if not collides(z, cand):
                            injs.append(dict(TARGETID=tid, inj_idx=k,
                                             cell=cname, Z_QSO=zq,
                                             HPXPIXEL=int(row["HPXPIXEL"]),
                                             z_inj=round(z, 6), logN=logN,
                                             z_segment="uniform",
                                             attempts=att))
                            placed = True
                            break
                    if not placed:
                        dropped.append(dict(TARGETID=tid, inj_idx=k,
                                            reason="collision_exhausted"))
                        failed = True
                        break
                if failed:
                    continue
                plan_rows.extend(injs)
                sightlines.append(dict(TARGETID=tid, Z_QSO=zq,
                                       RED_SNR=float(row["RED_SNR"]),
                                       HPXPIXEL=int(row["HPXPIXEL"]),
                                       cell=cname, n_inj=n_inj))
                picked += 1
            if picked < n_target:
                print(f"  [plan] cell {cname}: exhausted pool at {picked}")
    return sightlines, plan_rows, dropped
