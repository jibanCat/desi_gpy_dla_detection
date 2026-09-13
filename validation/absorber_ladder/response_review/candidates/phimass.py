#!/usr/bin/env python
"""phimass.py -- the row HAD-MASS phi(b, s, K), the in-grid fraction of the
response row.

POST-SEAL ADDITION (Science-lane instruction of 2026-09-13, received AFTER the
opening rule and the implementation predeclaration were sealed).  It changes
what the delivered Mg tensor CARRIES; it does NOT change the sealed section 5
criterion, which is scored on the in-grid row SHAPE only.

    phi(b, s, K) = P( N_hat lands on the >= 19.5 observed grid | b, s, K,
                      detected )

``calib_events_2lpt0.npz`` contains ONLY in-grid detections, so the row shape
fitted by every candidate is P(c | b, s, K, in-grid) with rows summing to one;
phi is the separate factor the fold needs, and the frozen parametric
``adopted_phi_ref`` of the pack is defective below N_true ~ 19.7.

Estimators
----------
* ``phi_percell``  : the MEASURED per-cell conditional count ratio with a
  Jeffreys +1/2, (k + 1/2) / (n + 1).  No smoothing across b, s or K -- an
  unsmoothed conditional count ratio, which the archaeology review classifies
  as compliant (the denominator is the SAME cell's detected count, so no
  calibration population slope enters).  THE DEFAULT for every candidate.
* ``fit_phi_smooth`` : logistic( poly_2(N_true) + poly_1(log10 S/N)
  + K offsets ), 6 coefficients, binomial ML on the same cell counts.

Cross-validation of phi
-----------------------
No event-level off-grid table exists at matched support (the 73,845-event
matched table is in-grid only, and ``cal_table``'s E/O split covers the
ALL-detected count but not the in-grid numerator, so mixing them would put the
numerator and denominator on different supports -- the recurring bug class).
phi is therefore cross-validated by a seeded TRIAL-level split of the Bernoulli
trials in each cell (each detected absorber is one in-grid/off-grid trial and
the trials within a cell are exchangeable).  This is a valid held-out binomial
score but it is NOT a sightline-level split, so it does not carry any
sightline-correlated component: disclosed in the variants table.

ENV: gpdla-hbi.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import candlib as CL


def load_phi_counts(ops_path, geom):
    """(k_in, n_all) per (b, s, K) from the A0 empirical-operator pack.

    ``N_det_bks_true_z``      = detections whose N_hat is ON the observed grid
    ``N_det_all_bks_true_z``  = all detections of the same truth rows
    Both are (B, kf, S) on the SAME support and the SAME truth-z convention;
    the fine-z axis is summed into the 3 coarse blocks by ``kz_to_K``.
    """
    z = np.load(ops_path, allow_pickle=True)
    din = np.asarray(z["N_det_bks_true_z"], float)          # (B, kf, S)
    dall = np.asarray(z["N_det_all_bks_true_z"], float)
    kz = np.asarray(z["kz_to_K"], int)
    B, Kf, S = din.shape
    K = int(kz.max()) + 1
    k_in = np.zeros((B, S, K)); n_all = np.zeros((B, S, K))
    for kf in range(Kf):
        k_in[:, :, kz[kf]] += din[:, kf, :]
        n_all[:, :, kz[kf]] += dall[:, kf, :]
    if np.any(k_in > n_all + 1e-9):
        raise SystemExit("phi counts: in-grid exceeds all-detected -- the "
                         "numerator and denominator are on different supports")
    return k_in, n_all


def phi_percell(k_in, n_all, alpha=0.5):
    """Measured per-cell phi with a Jeffreys +1/2 (the DEFAULT)."""
    return (np.asarray(k_in, float) + alpha) / \
        (np.asarray(n_all, float) + 2.0 * alpha)


def phi_design(geom):
    """(B, S, K, 6) cell-level design: [1, N-20.5, (N-20.5)^2, v, K1, K2].

    The S/N covariate is the MIDPOINT of log10(S/N) across the stratum -- the
    same flat convention as the row quadrature (predeclaration I2b), NOT the
    stratum's occupancy median.
    """
    _, vn = CL.quad_grid(geom)
    vmid = vn.mean(axis=1)                                   # (S,)
    B, S, K = geom["B"], geom["S"], geom["K"]
    X = np.zeros((B, S, K, 6))
    for b in range(B):
        u = geom["bcen"][b] - CL.N_REF_U
        for s in range(S):
            for k in range(K):
                X[b, s, k] = [1.0, u, u * u, vmid[s],
                              1.0 * (k == 1), 1.0 * (k == 2)]
    return X


def fit_phi_smooth(k_in, n_all, geom, ridge=1e-6):
    """Binomial ML of logistic(poly2(N) + poly1(log S/N) + K offsets)."""
    X = phi_design(geom).reshape(-1, 6)
    k = np.asarray(k_in, float).ravel()
    n = np.asarray(n_all, float).ravel()
    m = n > 0

    def nll(t):
        e = X[m] @ t
        # -sum[ k*log sigma(e) + (n-k)*log(1-sigma(e)) ]
        ll = k[m] * (-np.logaddexp(0.0, -e)) + \
            (n[m] - k[m]) * (-np.logaddexp(0.0, e))
        return -float(ll.sum()) / float(n[m].sum()) + ridge * float(t @ t)

    def grad(t):
        e = X[m] @ t
        p = 1.0 / (1.0 + np.exp(-e))
        g = -(X[m].T @ (k[m] - n[m] * p)) / float(n[m].sum())
        return g + 2.0 * ridge * t

    best = None
    for x0 in (np.zeros(6), np.array([2.0, 1.0, 0.0, 0.5, 0.0, 0.0])):
        r = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                     options=dict(maxiter=5000, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    t = best.x
    phi = 1.0 / (1.0 + np.exp(-(X @ t)))
    return phi.reshape(geom["B"], geom["S"], geom["K"]), dict(
        coef=t.tolist(), nominal_dof=6, train_obj=float(best.fun),
        form="logistic(a0 + a1 u + a2 u^2 + a3 v + a4 1[K=1] + a5 1[K=2])")


def binomial_ll_per_trial(phi, k, n):
    """Mean per-TRIAL binomial log-likelihood of ``phi`` on (k, n)."""
    phi = np.clip(np.asarray(phi, float), 1e-12, 1.0 - 1e-12)
    k = np.asarray(k, float); n = np.asarray(n, float)
    ll = k * np.log(phi) + (n - k) * np.log1p(-phi)
    tot = float(n.sum())
    return float(ll.sum() / tot) if tot > 0 else float("nan")


def phi_cv(k_in, n_all, geom, seed=20260913):
    """2-fold TRIAL-level split CV of the two phi estimators.

    Each cell's ``n`` Bernoulli trials are split at random into two halves
    (hypergeometric on the successes), each estimator is built on one half and
    scored on the other, and the two directions are averaged.
    """
    rs = np.random.RandomState(seed)
    k = np.rint(np.asarray(k_in, float)).astype(np.int64)
    n = np.rint(np.asarray(n_all, float)).astype(np.int64)
    nA = n // 2
    kA = np.zeros_like(k)
    it = np.nditer(n, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        if n[i] > 0:
            kA[i] = rs.hypergeometric(max(k[i], 0), max(n[i] - k[i], 0),
                                      max(int(nA[i]), 0)) if nA[i] > 0 else 0
    kB = k - kA
    nB = n - nA
    out = {}
    for tag, (kf, nf, ke, ne) in (("A->B", (kA, nA, kB, nB)),
                                  ("B->A", (kB, nB, kA, nA))):
        pc = phi_percell(kf, nf)
        ps, _ = fit_phi_smooth(kf, nf, geom)
        # empty training cells fall back to the smooth surface
        pc = np.where(nf > 0, pc, ps)
        out[tag] = dict(percell=binomial_ll_per_trial(pc, ke, ne),
                        smooth=binomial_ll_per_trial(ps, ke, ne))
    mean = {est: float(np.mean([out[t][est] for t in out]))
            for est in ("percell", "smooth")}
    return dict(folds=out, heldout_ll_per_trial=mean,
                protocol=("seeded TRIAL-level (not sightline-level) split of "
                          "each cell's Bernoulli in-grid trials; no "
                          "event-level off-grid table exists at matched "
                          "support"),
                seed=int(seed))


def phi_summary(phi, phi_ref_gathered, k_in, n_all, geom):
    """Diagnostics the variants table quotes."""
    n = np.asarray(n_all, float)
    pop = n > 0
    ratio = np.where(phi_ref_gathered > 0, phi / np.maximum(
        phi_ref_gathered, 1e-12), np.nan)
    lo = geom["ntrue"][:-1] < 19.7
    return dict(
        n_cells=int(phi.size), n_cells_populated=int(pop.sum()),
        n_cells_phi_lt_0p98=int(((phi < 0.98) & pop).sum()),
        effective_dof_percell=int(((phi < 0.98) & pop).sum()),
        ratio_to_phi_ref_min=float(np.nanmin(ratio[pop])),
        ratio_to_phi_ref_max=float(np.nanmax(ratio[pop])),
        ratio_below_19p7_min=float(np.nanmin(ratio[lo][pop[lo]])),
        ratio_below_19p7_max=float(np.nanmax(ratio[lo][pop[lo]])),
        phi_b0_s2=float(phi[0, 2, 0]), phi_b0_s7=float(phi[0, 7, 0]),
        phi_ref_b0_s2=float(phi_ref_gathered[0, 2, 0]),
        phi_ref_b0_s7=float(phi_ref_gathered[0, 7, 0]),
        n_trials=float(n.sum()))
