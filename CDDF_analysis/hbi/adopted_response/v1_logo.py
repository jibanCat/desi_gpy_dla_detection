#!/usr/bin/env python
"""V1 (PI ruling item 1): strict leave-one-group-out predictive validation of
the SHARED CUBIC response mode. Env gpdla. Closure never enters.

Design: for each held-out group G (each of the 9 cells; each z column; each
SNR row), the shared cubic is estimated from the OTHER cells only (iterated
per-cell deg-2 + pooled orders-3 WLS, exactly the selected procedure). For
every held-out cell g in G, its events are sightline-split A/B; per half:
fit cell-g's OWN deg-2 (on that half's ML sub-bin rows) under two arms:
  (i) transfer: y - shared(u) for the deg-2 fit, surfaces = deg2 + shared
  (ii) null:    plain deg-2, no shared term
then evaluate the held-out HALF's per-event conditional loglik under each.
Metric: delta = (i)-(ii) per event, summed over both half-directions.
The shared mode SURVIVES if delta > 0 consistently across held-out groups.
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

# ---- ML sub-bin rows: full data (for shared estimation) + per half -------
def cell_rows(mask, i, j, min_n):
    m = mask & (isr == i) & (izr == j)
    return fitlib.subbin_moments(N[m], dx[m], EDGES, min_n, "ml")

print("[v1] full-data ml rows", flush=True)
rows_full = [[cell_rows(np.ones(len(N), bool), i, j, 50) for j in range(3)]
             for i in range(3)]
rows_half = {}
for tag, hm in (("A", inA), ("B", ~inA)):
    print(f"[v1] half-{tag} ml rows", flush=True)
    rows_half[tag] = [[cell_rows(hm, i, j, 25) for j in range(3)]
                      for i in range(3)]

def shared_from_cells(rows, train_cells, n_iter=2):
    """Iterated per-cell deg-2 + pooled shared u^3, training cells only."""
    shared = np.zeros(3)                      # (mu, sig, skew) u^3 coefs
    keys = ("mu", "sig", "skew")
    for _ in range(n_iter):
        cell2 = {}
        for (i, j) in train_cells:
            rr = [r for r in rows[i][j] if r["ok"]]
            c = np.array([r["c"] for r in rr]); u = c - N_REF
            w = np.sqrt([r["n"] for r in rr])
            cell2[(i, j)] = {}
            for ki, k in enumerate(keys):
                y = np.array([r[k] for r in rr]) - shared[ki] * u ** 3
                cell2[(i, j)][k] = np.polynomial.polynomial.polyfit(
                    u, y, 2, w=w)
        for ki, k in enumerate(keys):
            uu, rr_, ww = [], [], []
            for (i, j) in train_cells:
                rrows = [r for r in rows[i][j] if r["ok"]]
                c = np.array([r["c"] for r in rrows]); u = c - N_REF
                up2 = u[:, None] ** np.arange(3)[None, :]
                y = np.array([r[k] for r in rrows]) - up2 @ cell2[(i, j)][k]
                uu.append(u); rr_.append(y)
                ww.append(np.sqrt([r["n"] for r in rrows]))
            uu = np.concatenate(uu); rr_ = np.concatenate(rr_)
            ww = np.concatenate(ww)
            shared[ki] = float(np.sum(ww ** 2 * uu ** 3 * rr_)
                               / np.sum(ww ** 2 * uu ** 6))
    return shared

def eval_cell(i, j, ev_mask, coefs2, shared):
    """Conditional per-event loglik of cell (i,j) events under
    deg2(+shared u^3) surfaces; returns (sum, n)."""
    surf = {k: np.zeros((3, 3, 4)) for k in ("mu", "sig", "skew")}
    rng = np.zeros((3, 3, 2))
    rr = [r for r in rows_full[i][j] if r["ok"]]
    rng[i, j] = (min(r["c"] for r in rr), max(r["c"] for r in rr))
    for ki, k in enumerate(("mu", "sig", "skew")):
        surf[k][i, j, :3] = coefs2[k]
        surf[k][i, j, 3] = shared[ki]
    m = ev_mask & (isr == i) & (izr == j)
    return fitlib.eval_loglik(N[m], dx[m], isr[m], izr[m], surf, rng,
                              N_REF, truncated=True)

def fit_deg2(rows_ij, shared):
    """Per-cell deg-2 with a FIXED shared u^3 subtracted."""
    rr = [r for r in rows_ij if r["ok"]]
    c = np.array([r["c"] for r in rr]); u = c - N_REF
    w = np.sqrt([r["n"] for r in rr])
    out = {}
    for ki, k in enumerate(("mu", "sig", "skew")):
        y = np.array([r[k] for r in rr]) - shared[ki] * u ** 3
        out[k] = np.polynomial.polynomial.polyfit(u, y, 2, w=w)
    return out

ALL = [(i, j) for i in range(3) for j in range(3)]
folds = {f"cell_{i}{j}": [(i, j)] for i, j in ALL}
folds.update({f"zcol_{j}": [(i, j) for i in range(3)] for j in range(3)})
folds.update({f"snrrow_{i}": [(i, j) for j in range(3)] for i in range(3)})

ZERO = np.zeros(3)
res = {}
for fname, held in folds.items():
    train = [c for c in ALL if c not in held]
    sh = shared_from_cells(rows_full, train)
    d_tot, n_tot = 0.0, 0
    for (i, j) in held:
        for fit_tag, ev_mask in (("A", ~inA), ("B", inA)):
            rows_ij = rows_half[fit_tag][i][j]
            c_tr = fit_deg2(rows_ij, sh)
            c_nu = fit_deg2(rows_ij, ZERO)
            ll_tr, n1 = eval_cell(i, j, ev_mask, c_tr, sh)
            ll_nu, n2 = eval_cell(i, j, ev_mask, c_nu, ZERO)
            assert n1 == n2
            d_tot += ll_tr - ll_nu; n_tot += n1
    res[fname] = dict(shared_mu3=round(float(sh[0]), 5),
                      shared_sig3=round(float(sh[1]), 5),
                      shared_skew3=round(float(sh[2]), 5),
                      delta_loglik=round(d_tot, 2), n=n_tot,
                      delta_per_1k_events=round(1000 * d_tot / n_tot, 3))
    print(f"[v1] {fname:10s} mu3 {sh[0]:+.4f}  dLL {d_tot:+9.2f} "
          f"({1000*d_tot/n_tot:+.3f}/1k ev)", flush=True)

n_pos = sum(1 for v in res.values() if v["delta_loglik"] > 0)
res["_summary"] = dict(
    n_folds=len(folds), n_positive=n_pos,
    mu3_range=[min(v["shared_mu3"] for k, v in res.items()
                   if not k.startswith("_")),
               max(v["shared_mu3"] for k, v in res.items()
                   if not k.startswith("_"))])
json.dump(res, open(os.path.join(HERE, "v1_logo_results.json"), "w"),
          indent=1)
print(f"[v1] positive folds: {n_pos}/{len(folds)}; wrote v1_logo_results.json")
