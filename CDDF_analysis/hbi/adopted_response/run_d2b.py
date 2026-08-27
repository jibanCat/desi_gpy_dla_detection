#!/usr/bin/env python
"""D2b (env gpdla): calibration-side selection, part 2.
(1) Estimator-class CV: sample-moment vs untruncated-ML deg-2 surfaces,
    scored by held-out CONDITIONAL (floor-normalized) per-event loglik,
    sightline (TID) 2-fold.
(2) Shared-across-cells higher-order correction: per-cell deg-2 ML + pooled
    shared u^3 (and u^3+u^4) moment corrections, 2 refit iterations; same CV.
(3) Save the full-data winner surfaces for folding.
DIAGNOSTIC ONLY.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fitlib

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

rngseed = np.random.default_rng(20260817)
uniq = np.unique(tids)
half = rngseed.permutation(len(uniq)) < len(uniq) // 2
inA = np.isin(tids, uniq[half])

def rows_all(mask, mode, min_n):
    return [[fitlib.subbin_moments(
        N[mask & (isr == i) & (izr == j)],
        dx[mask & (isr == i) & (izr == j)], EDGES, min_n, mode)
        for j in range(3)] for i in range(3)]

def shared_surfaces(rows, deg_shared, n_iter=2):
    """Per-cell deg-2 + shared orders 3..deg_shared (pooled WLS)."""
    Ds = deg_shared + 1
    shared = np.zeros((3, Ds))          # per moment (mu,sig,skew) x orders
    keys = ("mu", "sig", "skew")
    for _ in range(n_iter):
        # per-cell deg-2 on y - shared
        cell2 = {k: np.zeros((3, 3, 3)) for k in keys}
        rng = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                rr = [r for r in rows[i][j] if r["ok"]]
                c = np.array([r["c"] for r in rr]); u = c - N_REF
                w = np.sqrt([r["n"] for r in rr])
                rng[i, j] = (c.min(), c.max())
                up = u[:, None] ** np.arange(Ds)[None, :]
                for ki, k in enumerate(keys):
                    y = np.array([r[k] for r in rr]) - up[:, 3:] @ shared[ki, 3:]
                    cell2[k][i, j] = np.polynomial.polynomial.polyfit(
                        u, y, 2, w=w)
        # pooled shared orders 3..deg on residuals vs the cell deg-2
        for ki, k in enumerate(keys):
            uu, rr_, ww = [], [], []
            for i in range(3):
                for j in range(3):
                    rrows = [r for r in rows[i][j] if r["ok"]]
                    c = np.array([r["c"] for r in rrows]); u = c - N_REF
                    up2 = u[:, None] ** np.arange(3)[None, :]
                    y = np.array([r[k] for r in rrows]) - up2 @ cell2[k][i, j]
                    uu.append(u); rr_.append(y)
                    ww.append(np.sqrt([r["n"] for r in rrows]))
            uu = np.concatenate(uu); rr_ = np.concatenate(rr_)
            ww = np.concatenate(ww)
            X = uu[:, None] ** np.arange(3, Ds)[None, :]
            W = np.diag(ww)
            beta, *_ = np.linalg.lstsq(W @ X, ww * rr_, rcond=None)
            shared[ki, 3:] = beta
    surf = {}
    for ki, k in enumerate(keys):
        arr = np.zeros((3, 3, Ds))
        arr[..., :3] = cell2[k]
        arr[..., 3:] = shared[ki, 3:][None, None, :]
        surf[k] = arr
    return surf, rng, shared

res = {}
# ---- (1) estimator-class CV ---------------------------------------------
cv = {}
half_rows = {}
for tag, mask in (("A", inA), ("B", ~inA)):
    half_rows[tag] = {m: rows_all(mask, m, 25) for m in ("sample", "ml")}
    print(f"[d2b] half-{tag} rows done", flush=True)
for mode in ("sample", "ml"):
    tot, ntot = 0.0, 0
    for fit_tag, ev_mask in (("A", ~inA), ("B", inA)):
        surf, rng = fitlib.surfaces_from_rows(half_rows[fit_tag][mode],
                                              N_REF, 2)
        ll, n = fitlib.eval_loglik(N[ev_mask], dx[ev_mask], isr[ev_mask],
                                   izr[ev_mask], surf, rng, N_REF,
                                   truncated=True)
        tot += ll; ntot += n
    cv[f"{mode}_deg2"] = round(tot / ntot, 5)
    print(f"[d2b] CV {mode}_deg2: {tot/ntot:.5f}/event", flush=True)

# ---- (2) shared higher-order CV -----------------------------------------
for degs in (3, 4):
    tot, ntot = 0.0, 0
    for fit_tag, ev_mask in (("A", ~inA), ("B", inA)):
        surf, rng, _ = shared_surfaces(half_rows[fit_tag]["ml"], degs)
        ll, n = fitlib.eval_loglik(N[ev_mask], dx[ev_mask], isr[ev_mask],
                                   izr[ev_mask], surf, rng, N_REF,
                                   truncated=True)
        tot += ll; ntot += n
    cv[f"ml_deg2_shared{degs}"] = round(tot / ntot, 5)
    print(f"[d2b] CV ml_deg2_shared{degs}: {tot/ntot:.5f}/event", flush=True)
res["cv_per_event_loglik"] = cv

# ---- (3) full-data surfaces for the shared variants ----------------------
full_rows = rows_all(np.ones(len(N), bool), "ml", 50)
sv = {"N_ref": EV["N_ref"]}
for degs in (3, 4):
    surf, rng, shared = shared_surfaces(full_rows, degs)
    key = f"ml_shared{degs}"
    for k in ("mu", "sig", "skew"):
        sv[f"{key}__{k}"] = surf[k]
    sv[f"{key}__rng"] = rng
    res[f"shared{degs}_coeffs"] = {k: np.round(shared[ki, 3:], 5).tolist()
                                   for ki, k in enumerate(("mu", "sig",
                                                           "skew"))}
np.savez_compressed(os.path.join(HERE, "d2b_variants.npz"), **sv)
json.dump(res, open(os.path.join(HERE, "d2b_results.json"), "w"), indent=1)
print("wrote d2b_variants.npz, d2b_results.json")
