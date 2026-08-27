#!/usr/bin/env python
"""Sightline-bootstrap covariance CARRIER for the ADOPTED estimator
(per-cell deg-2 ML + shared cubic), PI ruling 2026-08-17 item 1.
Env gpdla; multiprocessing over draws. Unit-weight gate: the w==1 pipeline
must reproduce the adopted full-data surfaces.
Output: adopted_carrier_ensemble.npz (draws of the (3,3,4) moment surfaces
+ shared coefficients + gate record).
"""
import os, sys
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fitlib
from run_d2b_lib import shared_surfaces

N_DRAWS = 96
N_PROC = 7
SEED0 = 20260818

EV = np.load(os.path.join(HERE, "events_full.npz"))
N_REF = float(EV["N_ref"])
EDGES = np.arange(19.0, 21.4 + 1e-9, 0.1)

xhat = EV["xhat"]; snr = EV["snr"]; zq = EV["zqso"]; tid = EV["tid"]
Nhost = EV["nhi_tilt_host"]
keep = np.isfinite(Nhost) & (xhat >= 19.5)
N = Nhost[keep]; dx = (xhat - Nhost)[keep]
isr = np.clip(np.digitize(snr[keep], EV["snr_edges"]) - 1, 0, 2)
izr = np.clip(np.digitize(zq[keep], EV["z_edges"]) - 1, 0, 2)
tids = tid[keep]
uniq, tid_idx = np.unique(tids, return_inverse=True)
n_u = len(uniq)

# frozen anchor set: bins each cell fits at min_n=50 on the full data
ANCH = {}
ib_all = np.digitize(N, EDGES) - 1
for i in range(3):
    for j in range(3):
        m = (isr == i) & (izr == j)
        counts = np.bincount(ib_all[m][(ib_all[m] >= 0)
                                       & (ib_all[m] < len(EDGES) - 1)],
                             minlength=len(EDGES) - 1)
        ANCH[(i, j)] = [b for b in range(len(EDGES) - 1) if counts[b] >= 50]


def one_draw(seed):
    if seed is None:                   # unit-weight gate draw
        w = np.ones(len(N))
    else:
        rng = np.random.default_rng(seed)
        mult = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u)).astype(float)
        w = mult[tid_idx]
    rows = [[fitlib.subbin_moments(
        N[(isr == i) & (izr == j)], dx[(isr == i) & (izr == j)], EDGES,
        50, "ml", weights=w[(isr == i) & (izr == j)],
        fixed_bins=ANCH[(i, j)]) for j in range(3)] for i in range(3)]
    surf, rng_fit, shared = shared_surfaces(rows, 3, N_REF)
    return (surf["mu"], surf["sig"], surf["skew"], shared[:, 3], rng_fit)


if __name__ == "__main__":
    print(f"[carrier] events {len(N)}, uniq TIDs {n_u}, "
          f"anchors/cell {[len(v) for v in ANCH.values()]}", flush=True)
    unit = one_draw(None)
    vz = np.load(os.path.join(HERE, "d2b_variants.npz"))
    gate = {}
    for nm, arr, ref in (("mu", unit[0], vz["ml_shared3__mu"]),
                         ("sig", unit[1], vz["ml_shared3__sig"]),
                         ("skew", unit[2], vz["ml_shared3__skew"])):
        gate[nm] = float(np.max(np.abs(arr - ref)))
    print(f"[carrier] unit-weight gate max|delta| vs adopted: {gate}",
          flush=True)
    if max(gate.values()) > 5e-3:
        raise SystemExit(f"unit-weight gate FAILED: {gate}")

    seeds = [SEED0 + i for i in range(N_DRAWS)]
    with Pool(N_PROC) as pool:
        out = []
        for r, res in enumerate(pool.imap(one_draw, seeds)):
            out.append(res)
            if (r + 1) % 10 == 0:
                print(f"[carrier] {r+1}/{N_DRAWS}", flush=True)
    mu_e = np.stack([o[0] for o in out])
    sig_e = np.stack([o[1] for o in out])
    skw_e = np.stack([o[2] for o in out])
    sh_e = np.stack([o[3] for o in out])
    np.savez_compressed(
        os.path.join(HERE, "adopted_carrier_ensemble.npz"),
        mu=mu_e, sig=sig_e, skew=skw_e, shared3=sh_e,
        unit_gate=np.array([gate[k] for k in ("mu", "sig", "skew")]),
        point_mu=vz["ml_shared3__mu"], point_sig=vz["ml_shared3__sig"],
        point_skew=vz["ml_shared3__skew"], rng=vz["ml_shared3__rng"],
        seed0=np.array(SEED0), n_events=np.array(len(N)),
        n_uniq=np.array(n_u))
    print("[carrier] order-0 sd mu:",
          np.round(mu_e[..., 0].std(axis=0, ddof=1), 5).tolist(), flush=True)
    print("[carrier] order-0 sd sig:",
          np.round(sig_e[..., 0].std(axis=0, ddof=1), 5).tolist(), flush=True)
    print("[carrier] shared3 mean/sd:",
          np.round(sh_e.mean(axis=0), 5).tolist(),
          np.round(sh_e.std(axis=0, ddof=1), 5).tolist(), flush=True)
    print("[carrier] wrote adopted_carrier_ensemble.npz")
