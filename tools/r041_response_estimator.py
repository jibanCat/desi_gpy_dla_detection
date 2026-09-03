#!/usr/bin/env python
"""R-041 response-estimator rebuild: Candidates E (empirical kernel) and P (balanced fixed-bin parametric), estimator-level closure tests
T1–T5 and the calibration-arm comparison (mock-only; PI continuation ruling 2026-09-02; frozen gate
MAX4_RESPONSE_ESTIMATOR_CLOSURE_GATE_2026-09-02.md §1–§3, §5).

Inputs: the full-support matched tables of the response study (matches_{N2,NL}_native_full.csv) and the injection per-injection tables, read through
r041_response_population_study.load_events. Outputs: JSON under --out.
Stages: build | closure | arms
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r041_response_population_study import load_events, ROOT, HBI_REPO, FIT_SNR_EDGES, ZB_NATIVE, ZB_EMUL  # noqa: E402

sys.path.insert(0, os.path.join(HBI_REPO, "CDDF_analysis", "hbi", "adopted_response"))
sys.path.insert(0, HBI_REPO)

TB = np.array([19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7, 20.9, 21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4])   # pack latent basis (16 bins)
OB_FINE = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 3)                                                              # pack observed grid (29)
OB = np.round(np.arange(19.5, 22.3 + 1e-9, 0.2), 2)                                                                    # estimator-level observed grid (14)
SUB = np.round(np.arange(19.0, 22.4 + 1e-9, 0.1), 3)                                                                   # Candidate P sub-bins
N_REF_P = 20.5
MIN_E, MIN_P = 15, 20
XHAT_FLOOR = 19.5
SEED = 20260902


def cells_of(ev):
    isr = np.clip(np.searchsorted(np.asarray(FIT_SNR_EDGES), ev["snr"], side="right") - 1, 0, len(FIT_SNR_EDGES) - 2)
    return isr, ev["zblock"].astype(int)


def bidx(edges, x):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, -1, len(edges) - 2)


# ----------------------------------------------------------------------------------------------------------------------- Candidate E
def build_E(ev, w=None, ob=OB_FINE, tb=TB, min_n=MIN_E):
    """Empirical kernel M[sr, zr, c, b] = P(x̂ in c | detected, truth in b, sr, zr) with the frozen sparsity fallback; completeness C[sr, zr, b]."""
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    isr, izr = cells_of(ev); b = bidx(tb, ev["logN"]); det = ev["matched"] & np.isfinite(ev["Nhat"])
    c = np.where(det, bidx(ob, np.nan_to_num(ev["Nhat"], nan=-1.0)), -1)
    n_sr, n_zr, C_n, B_n = len(FIT_SNR_EDGES) - 1, 3, len(ob) - 1, len(tb) - 1
    M = np.zeros((n_sr, n_zr, C_n, B_n)); C = np.zeros((n_sr, n_zr, B_n)); n_det = np.zeros((n_sr, n_zr, B_n)); fallback = {}
    inb = b >= 0
    def row(mask):
        m = mask & det; tot = w[m].sum(); r = np.zeros(C_n)
        for cc in range(C_n):
            r[cc] = w[m & (c == cc)].sum()
        return r / tot if tot > 0 else r, tot
    for bb in range(B_n):
        g_row, g_n = row(inb & (b == bb))
        for a in range(n_sr):
            p_row, p_n = row(inb & (b == bb) & (isr == a))
            for z in range(n_zr):
                m = inb & (b == bb) & (isr == a) & (izr == z)
                tt = w[m].sum(); C[a, z, bb] = (w[m & det].sum() / tt) if tt > 0 else np.nan
                r, n = row(m); n_det[a, z, bb] = n
                if n >= min_n:
                    M[a, z, :, bb] = r
                elif p_n >= min_n:
                    M[a, z, :, bb] = p_row; fallback[f"{a},{z},{bb}"] = "SR-pooled"
                elif g_n >= min_n:
                    M[a, z, :, bb] = g_row; fallback[f"{a},{z},{bb}"] = "global"
                else:
                    M[a, z, :, bb] = np.nan; fallback[f"{a},{z},{bb}"] = "nearest-bin"
    # nearest populated bin substitution
    for a in range(n_sr):
        for z in range(n_zr):
            ok = np.array([np.all(np.isfinite(M[a, z, :, bb])) for bb in range(B_n)])
            for bb in np.where(~ok)[0]:
                near = np.where(ok)[0]; j = near[np.argmin(np.abs(near - bb))]; M[a, z, :, bb] = M[a, z, :, j]
    return dict(M=M, C=C, n_det=n_det, fallback=fallback, ob=ob, tb=tb)


# ----------------------------------------------------------------------------------------------------------------------- Candidate P
def build_P(ev, w=None, ob=OB_FINE, tb=TB, min_anchor=MIN_P):
    """Balanced fixed-bin parametric kernel: 0.1-dex sub-bins, truncated-ML skew-normal moments (fitlib), unweighted deg-2 through the anchors, N_ref 20.5 fixed,
    clamp to the anchor range, skew ramp above 21.0; pooled ML constants for cells with < 4 anchors."""
    import fitlib
    from scipy.stats import skewnorm
    from CDDF_analysis.hbi.znz_kernel import _moment_to_skewnormal_vec
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    isr, izr = cells_of(ev); det = ev["matched"] & np.isfinite(ev["Nhat"]) & (ev["Nhat"] >= XHAT_FLOOR)
    N = ev["logN"]; dx = ev["Nhat"] - ev["logN"]
    n_sr, n_zr = len(FIT_SNR_EDGES) - 1, 3
    coef = {k: np.zeros((n_sr, n_zr, 3)) for k in ("mu", "sig", "skew")}; rng = np.zeros((n_sr, n_zr, 2)); anchors = {}
    def ml_rows(mask):
        m = mask & det
        return fitlib.subbin_moments(N[m], dx[m], SUB, min_anchor, "ml_trunc", weights=w[m])
    glob_rows = [r for r in ml_rows(np.ones(len(N), bool)) if r["ok"]]
    g_rng = (min(r["c"] for r in glob_rows), max(r["c"] for r in glob_rows))
    def pooled_const(mask):
        m = mask & det
        if m.sum() < 20:
            m = det
        p, ok, _ = fitlib.fit_subbin(dx[m], 19.5 - N[m], truncated=True, w=w[m])
        return p
    for a in range(n_sr):
        for z in range(n_zr):
            rows = [r for r in ml_rows((isr == a) & (izr == z)) if r["ok"]]
            anchors[f"{a},{z}"] = [round(r["c"], 2) for r in rows]
            if len(rows) >= 4:
                cc = np.array([r["c"] for r in rows]); u = cc - N_REF_P
                for k in ("mu", "sig", "skew"):
                    y = np.array([r[k] for r in rows]); coef[k][a, z] = np.polynomial.polynomial.polyfit(u, y, 2)   # UNWEIGHTED (flat-in-N)
                rng[a, z] = (cc.min(), cc.max())
            else:
                p = pooled_const((isr == a) & (izr == z)); coef["mu"][a, z, 0], coef["sig"][a, z, 0], coef["skew"][a, z, 0] = p; rng[a, z] = g_rng
    # masses (mirror surface_masses)
    Nc = 0.5 * (tb[:-1] + tb[1:]); C_n, B_n = len(ob) - 1, len(tb) - 1
    M = np.zeros((n_sr, n_zr, C_n, B_n)); mu_grid = np.zeros((n_sr, n_zr, B_n))
    for a in range(n_sr):
        for z in range(n_zr):
            Ncl = np.clip(Nc, rng[a, z, 0], rng[a, z, 1]); u = Ncl - N_REF_P; up = u[:, None] ** np.arange(3)[None, :]
            mean = Nc + up @ coef["mu"][a, z]; sd = np.maximum(up @ coef["sig"][a, z], 1e-3)
            ramp = np.clip((Nc - 21.0) / 0.5, 0, 1); sk = np.clip(up @ coef["skew"][a, z], -0.995 * 0.95, 0.995 * 0.95) * (1 - ramp)
            xi, om, al = _moment_to_skewnormal_vec(mean, sd, sk)
            cdf = np.stack([skewnorm.cdf(e, al, loc=xi, scale=om) for e in ob]); M[a, z] = np.clip(np.diff(cdf, axis=0), 0, 1); mu_grid[a, z] = mean - Nc
    E = build_E(ev, w, ob, tb)
    return dict(M=M, C=E["C"], coef={k: v.tolist() for k, v in coef.items()}, rng=rng.tolist(), anchors=anchors, mu_grid=mu_grid.tolist(), ob=ob, tb=tb, N_ref=N_REF_P)


# ----------------------------------------------------------------------------------------------------------------------- weights
def scheme_w(N, scheme, ref_N=None):
    N = np.asarray(N, float)
    if scheme == "W1_unit":
        w = np.ones(len(N))
    elif scheme == "W2_flat":
        h, _ = np.histogram(N, TB); d = h[np.clip(bidx(TB, N), 0, len(TB) - 2)] / np.diff(TB)[np.clip(bidx(TB, N), 0, len(TB) - 2)]
        w = np.minimum(1.0 / np.maximum(d / d.max(), 1e-3), 20.0)
    elif scheme == "W5_tilt_minus":
        w = 10 ** (-0.4 * (N - 20.5))
    elif scheme == "W5_tilt_plus":
        w = 10 ** (+0.4 * (N - 20.5))
    else:
        raise ValueError(scheme)
    return w / w.mean()


# ----------------------------------------------------------------------------------------------------------------------- evaluation
def coarsen(M, ob_fine=OB_FINE, ob=OB):
    """0.1-dex observed masses -> 0.2-dex observed bins (sum pairs); the last fine bin [22.3,22.4) is dropped (outside the 0.2 grid)."""
    idx = bidx(ob, 0.5 * (ob_fine[:-1] + ob_fine[1:])); out = np.zeros(M.shape[:2] + (len(ob) - 1,) + M.shape[3:])
    for i, j in enumerate(idx):
        if j >= 0:
            out[:, :, j] += M[:, :, i]
    return out


def population(ev, w=None, tb=TB, ob=OB):
    """Evaluation population: truth histogram T[sr, zr, b], TP observed histogram n_obs[c] (0.2 dex), S/N mix pi[sr, zr]."""
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    isr, izr = cells_of(ev); b = bidx(tb, ev["logN"]); det = ev["matched"] & np.isfinite(ev["Nhat"])
    c = np.where(det, bidx(ob, np.nan_to_num(ev["Nhat"], nan=-1.0)), -1)
    T = np.zeros((3, 3, len(tb) - 1)); n_obs = np.zeros(len(ob) - 1); n_obs_cell = np.zeros((3, 3, len(ob) - 1))
    for a in range(3):
        for z in range(3):
            m = (isr == a) & (izr == z)
            for bb in range(len(tb) - 1):
                T[a, z, bb] = w[m & (b == bb)].sum()
            for cc in range(len(ob) - 1):
                n_obs_cell[a, z, cc] = w[m & (c == cc)].sum()
    n_obs = n_obs_cell.sum(axis=(0, 1))
    return dict(T=T, n_obs=n_obs, n_obs_cell=n_obs_cell)


def forward(A, T):
    """mu[c] = sum_{sr,zr,b} A[sr,zr,c,b] T[sr,zr,b]."""
    return np.einsum("azcb,azb->c", A, T)


def share_above(edges, thr):
    return np.clip((edges[1:] - thr) / np.diff(edges), 0, 1)


def smooth_unfold(A, n_obs, pi, tb=TB, n_iter=None):
    """log f_z(N) = a_z + cubic(N - 20.5) with ONE global cubic shape and a per-z-block log-amplitude a_z (gate Amendment 1: the per-block cubic is
    unidentifiable where a z block is sparse). Maximum Poisson likelihood of n_obs through A with the S/N mix pi[sr, zr]; parameters bounded to [-25, 25].
    Returns f[zr, b] (per truth bin, summed over sr via pi) and the fitted theta."""
    from scipy.optimize import minimize
    Nc = 0.5 * (tb[:-1] + tb[1:]); u = Nc - 20.5; V = np.vander(u, 4, increasing=True)          # (B, 4)
    def f_of(theta):
        shape = V[:, 1:] @ theta[3:6]; return np.exp(theta[:3][:, None] + shape[None, :])       # (zr, B)
    def mu_of(theta):
        f = f_of(theta); T = pi[:, :, None] * f[None, :, :]
        return forward(A, T), f
    def nll(theta):
        mu, _ = mu_of(theta); mu = np.maximum(mu, 1e-9); return float((mu - n_obs * np.log(mu)).sum())
    tot = max(n_obs.sum(), 1.0); th0 = np.zeros(6); th0[:3] = np.log(tot / 3.0 / len(Nc) + 1e-9)
    res = minimize(nll, th0, method="L-BFGS-B", bounds=[(-25, 25)] * 6, options=dict(maxiter=5000))
    if not res.success:
        res = minimize(nll, res.x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7))
    mu, f = mu_of(res.x)
    return dict(f=f, mu=mu, theta=res.x, success=bool(res.success), nll=float(res.fun))


def sums(f_zb, tb=TB):
    """Integrated sums over truth bins with partial-bin shares at 20.0 / 20.3 and the tail >= 21.1 (f_zb: (zr, B) or (B,))."""
    f = np.asarray(f_zb); tot = f.sum(axis=0) if f.ndim == 2 else f
    return dict(ge20p0=float((tot * share_above(tb, 20.0)).sum()), ge20p3=float((tot * share_above(tb, 20.3)).sum()), ge21p1=float((tot * share_above(tb, 21.1)).sum()))


def crossing_from_A(M, C, ob=OB, tb=TB):
    """Kernel U (truth [20.1,20.3) -> x̂ >= 20.3) and D (truth [20.3,20.5) -> x̂ < 20.3) per (sr, zr), detected-conditional (masses / phi)."""
    bU = int(np.argmin(np.abs(0.5 * (tb[:-1] + tb[1:]) - 20.2))); bD = int(np.argmin(np.abs(0.5 * (tb[:-1] + tb[1:]) - 20.4)))
    ge = ob[:-1] >= 20.3 - 1e-9
    phiU = M[:, :, :, bU].sum(axis=2); phiD = M[:, :, :, bD].sum(axis=2)
    U = np.where(phiU > 0, M[:, :, ge, bU].sum(axis=2) / np.maximum(phiU, 1e-12), np.nan)
    D = np.where(phiD > 0, M[:, :, ~ge, bD].sum(axis=2) / np.maximum(phiD, 1e-12), np.nan)
    return U, D


def crossing_aggregate_A(M, C, T, ob=OB, tb=TB):
    """Population-level U, D implied by the operator on the evaluation truth T[sr, zr, b]: detected-weighted average over cells (gate Amendment 1)."""
    U, D = crossing_from_A(M, C, ob, tb)
    bU = int(np.argmin(np.abs(0.5 * (tb[:-1] + tb[1:]) - 20.2))); bD = int(np.argmin(np.abs(0.5 * (tb[:-1] + tb[1:]) - 20.4)))
    Cn = np.nan_to_num(C)
    wU = T[:, :, bU] * Cn[:, :, bU]; wD = T[:, :, bD] * Cn[:, :, bD]
    Ua = float(np.nansum(np.nan_to_num(U) * wU) / max(wU.sum(), 1e-12)); Da = float(np.nansum(np.nan_to_num(D) * wD) / max(wD.sum(), 1e-12))
    return Ua, Da


def crossing_from_events(ev, w=None):
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    isr, izr = cells_of(ev); det = ev["matched"] & np.isfinite(ev["Nhat"]); U = np.full((3, 3), np.nan); D = np.full((3, 3), np.nan)
    for a in range(3):
        for z in range(3):
            m = (isr == a) & (izr == z) & det
            mU = m & (ev["logN"] >= 20.1) & (ev["logN"] < 20.3); mD = m & (ev["logN"] >= 20.3) & (ev["logN"] < 20.5)
            if w[mU].sum() >= 10: U[a, z] = w[mU & (ev["Nhat"] >= 20.3)].sum() / w[mU].sum()
            if w[mD].sum() >= 10: D[a, z] = w[mD & (ev["Nhat"] < 20.3)].sum() / w[mD].sum()
    return U, D


def crossing_aggregate_events(ev, w=None):
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    det = ev["matched"] & np.isfinite(ev["Nhat"])
    mU = det & (ev["logN"] >= 20.1) & (ev["logN"] < 20.3); mD = det & (ev["logN"] >= 20.3) & (ev["logN"] < 20.5)
    return float(w[mU & (ev["Nhat"] >= 20.3)].sum() / max(w[mU].sum(), 1e-12)), float(w[mD & (ev["Nhat"] < 20.3)].sum() / max(w[mD].sum(), 1e-12))


def template_fit(A, T, n_obs):
    """Amplitude-only unfold (gate Amendment 2): f = a * T_shape (the evaluation truth SHAPE); maximum Poisson likelihood for a; returns a."""
    mu1 = forward(A, T); mu1 = np.maximum(mu1, 1e-12)
    # closed form: a = sum(n_obs) / sum(mu1) maximises the Poisson likelihood for a pure scale factor
    return float(n_obs.sum() / mu1.sum())


def evaluate(op, ev_eval, w_eval=None, n_boot=200, seed=SEED, label="", C_override=None):
    """T1 forward residuals (HZ1 reporting range [20.0, 21.7); top bins reported separately), T2' amplitude-only recovery with sightline bootstrap
    (the cubic unfold kept as a superseded diagnostic), T3 migration; C_override: use another arm's completeness (M-only transfer)."""
    from scipy.stats import poisson
    Cuse = np.nan_to_num(op["C"]) if C_override is None else np.nan_to_num(C_override)
    A_fine = op["M"] * Cuse[:, :, None, :]; A = coarsen(A_fine)
    pop = population(ev_eval, w_eval); T = pop["T"]; n_obs = pop["n_obs"]
    mu = forward(A, T); ok = (OB[:-1] >= 20.0 - 1e-9) & (OB[:-1] < 21.7 - 1e-9); top = OB[:-1] >= 21.7 - 1e-9
    r = np.where(mu > 0, (n_obs - mu) / np.sqrt(np.maximum(mu, 1e-9)), 0.0)
    pv = np.array([min(1.0, 2 * min(poisson.cdf(k, m), poisson.sf(k - 1, m))) if m > 0 else (1.0 if k == 0 else 0.0) for k, m in zip(n_obs, np.maximum(mu, 1e-12))])
    rr = r[ok]; pp = pv[ok]; run = 0; worst = 0; prev = 0.0
    for x, p_ in zip(rr, pp):
        sig1 = p_ < 0.32
        run = run + 1 if sig1 and (run == 0 or np.sign(x) == np.sign(prev)) else (1 if sig1 else 0); prev = x; worst = max(worst, run)
    T1 = dict(resid=np.round(r, 2).tolist(), p_value=np.round(pv, 4).tolist(), min_p=float(pp.min()), max_abs=float(np.max(np.abs(rr))), longest_same_sign_run_gt1sigma=int(worst),
              PASS=bool(pp.min() >= 0.05 and worst < 3), n_obs=n_obs.tolist(), mu=np.round(mu, 1).tolist(),
              top_bins=dict(bins=OB[:-1][top].tolist(), n_obs=n_obs[top].tolist(), mu=np.round(mu[top], 2).tolist(), p_value=np.round(pv[top], 4).tolist()))
    a_hat = template_fit(A, T, n_obs)
    # superseded cubic unfold (diagnostic only)
    pi = T.sum(axis=2); pi = pi / np.maximum(pi.sum(axis=0, keepdims=True), 1e-12)
    try:
        uf = smooth_unfold(A, n_obs, pi); tr = sums(T.sum(axis=0)); est = sums(uf["f"])
        cubic = dict(bias_ge20p0_pct=round(100 * (est["ge20p0"] / tr["ge20p0"] - 1), 2), bias_ge20p3_pct=round(100 * (est["ge20p3"] / tr["ge20p3"] - 1), 2), success=uf["success"])
    except Exception as e:
        cubic = dict(error=str(e))
    rng = np.random.default_rng(seed); tids = np.unique(ev_eval["TARGETID"]); tid_idx = np.searchsorted(tids, ev_eval["TARGETID"]); boots = []
    w0 = np.ones(len(ev_eval["logN"])) if w_eval is None else np.asarray(w_eval, float)
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(tids), len(tids)), minlength=len(tids)).astype(float)
        wb = w0 * mult[tid_idx]; pb = population(ev_eval, wb); boots.append(template_fit(A, pb["T"], pb["n_obs"]))
    boots = np.array(boots) if boots else np.array([np.nan]); q = np.nanpercentile(boots, [2.5, 50, 97.5])
    T2 = dict(amplitude=round(a_hat, 4), bias_pct=round(100 * (a_hat - 1), 2), boot95_amplitude=[round(float(q[0]), 4), round(float(q[2]), 4)],
              truth_in_95=bool(q[0] <= 1.0 <= q[2]) if n_boot > 0 else None, superseded_cubic_unfold=cubic)
    T2["PASS"] = bool((n_boot == 0 or T2["truth_in_95"]) and abs(T2["bias_pct"]) <= 5.4)
    UA, DA = crossing_from_A(coarsen(op["M"]), Cuse); UE, DE = crossing_from_events(ev_eval, w_eval)
    Ua, Da = crossing_aggregate_A(coarsen(op["M"]), Cuse, T); Ue, De = crossing_aggregate_events(ev_eval, w_eval)
    dU = abs(Ua - Ue); dD = abs(Da - De)
    T3 = dict(U_operator_cells=np.round(UA, 3).tolist(), U_population_cells=np.round(UE, 3).tolist(), D_operator_cells=np.round(DA, 3).tolist(), D_population_cells=np.round(DE, 3).tolist(),
              U_operator=round(Ua, 4), U_population=round(Ue, 4), D_operator=round(Da, 4), D_population=round(De, 4),
              max_dU=round(float(dU), 3), max_dD=round(float(dD), 3), PASS=bool(dU <= 0.05 and dD <= 0.05))
    return dict(label=label, T1=T1, T2=T2, T3=T3, PASS=bool(T1["PASS"] and T2["PASS"] and T3["PASS"]))


def dn_stats(ev, w=None):
    w = np.ones(len(ev["logN"])) if w is None else np.asarray(w, float)
    isr, _ = cells_of(ev); det = ev["matched"] & np.isfinite(ev["Nhat"]); out = {}
    for a in range(3):
        for lo, hi in ((20.0, 20.3), (20.3, 20.5), (20.5, 21.0), (21.0, 21.5)):
            m = det & (isr == a) & (ev["logN"] >= lo) & (ev["logN"] < hi)
            if w[m].sum() < 10:
                continue
            d = ev["Nhat"][m] - ev["logN"][m]; ww = w[m] / w[m].sum(); mean = float((ww * d).sum())
            out[f"sr{a}|[{lo},{hi})"] = dict(n=int(m.sum()), mean=round(mean, 4), sd=round(float(np.sqrt((ww * (d - mean) ** 2).sum())), 4),
                                            p_gt_0p1=round(float((ww * (d > 0.1)).sum()), 3), p_gt_0p2=round(float((ww * (d > 0.2)).sum()), 3))
    return out


def stage_build(out_dir, study_dir):
    res = {}
    for s in ("N2", "NL"):
        ev = load_events(s, study_dir); E = build_E(ev); P = build_P(ev)
        res[s] = dict(E=dict(fallback_cells=len(E["fallback"]), fallback=E["fallback"], n_det_by_cell=np.round(E["n_det"], 0).tolist(), C=np.round(np.nan_to_num(E["C"]), 4).tolist()),
                      P=dict(coef=P["coef"], rng=P["rng"], anchors=P["anchors"], mu_grid=np.round(np.array(P["mu_grid"]), 4).tolist()))
        np.savez(os.path.join(out_dir, f"operator_E_{s}.npz"), M=E["M"], C=np.nan_to_num(E["C"]), ob=OB_FINE, tb=TB)
        np.savez(os.path.join(out_dir, f"operator_P_{s}.npz"), M=P["M"], C=np.nan_to_num(P["C"]), ob=OB_FINE, tb=TB)
        print(f"{s}: E fallback cells {len(E['fallback'])}/{E['M'].shape[0]*E['M'].shape[1]*E['M'].shape[3]}; P anchors per cell", {k: len(v) for k, v in P['anchors'].items()})
    json.dump(res, open(os.path.join(out_dir, "operators_build.json"), "w"), indent=1)


def stage_closure(out_dir, study_dir):
    evs = {s: load_events(s, study_dir) for s in ("N2", "NL")}
    res = {}
    for cand, builder in (("E", build_E), ("P", build_P)):
        res[cand] = {}
        ops = {s: builder(evs[s]) for s in evs}
        for cal in evs:
            for ev_ in evs:
                r = evaluate(ops[cal], evs[ev_], label=f"{cand}:{cal}->{ev_}"); res[cand][f"{cal}->{ev_}"] = r
                print(f"{cand} {cal}->{ev_}: T1 min p {r['T1']['min_p']:.3f} run {r['T1']['longest_same_sign_run_gt1sigma']} {'PASS' if r['T1']['PASS'] else 'FAIL'} | "
                      f"T2' amp bias {r['T2']['bias_pct']:+.2f} % 95% {r['T2']['boot95_amplitude']} {'PASS' if r['T2']['PASS'] else 'FAIL'} (cubic sup. {r['T2']['superseded_cubic_unfold'].get('bias_ge20p3_pct')}) | "
                      f"T3 U {r['T3']['U_operator']:.3f}/{r['T3']['U_population']:.3f} D {r['T3']['D_operator']:.3f}/{r['T3']['D_population']:.3f} {'PASS' if r['T3']['PASS'] else 'FAIL'} -> {'PASS' if r['PASS'] else 'FAIL'}")
                if cal != ev_:
                    rm = evaluate(ops[cal], evs[ev_], label=f"{cand}:{cal}->{ev_} M-only", C_override=ops[ev_]["C"]); res[cand][f"{cal}->{ev_}|M-only"] = rm
                    print(f"   M-only (C from {ev_}): T1 min p {rm['T1']['min_p']:.3f} run {rm['T1']['longest_same_sign_run_gt1sigma']} {'PASS' if rm['T1']['PASS'] else 'FAIL'} | T2' {rm['T2']['bias_pct']:+.2f} % {rm['T2']['boot95_amplitude']} {'PASS' if rm['T2']['PASS'] else 'FAIL'} | T3 dU {rm['T3']['max_dU']:.3f} dD {rm['T3']['max_dD']:.3f} {'PASS' if rm['T3']['PASS'] else 'FAIL'} -> {'PASS' if rm['PASS'] else 'FAIL'}")
        for cal in evs:
            base = res[cand][f"{cal}->{cal}"]; t4 = {}
            for sc in ("W2_flat", "W5_tilt_minus", "W5_tilt_plus"):
                opw = builder(evs[cal], scheme_w(evs[cal]["logN"], sc)); r = evaluate(opw, evs[cal], n_boot=0, label=f"{cand}:{cal}[{sc}]->{cal}")
                T_eval = population(evs[cal])["T"]
                U0, D0 = crossing_aggregate_A(coarsen(ops[cal]["M"]), ops[cal]["C"], T_eval); U1, D1 = crossing_aggregate_A(coarsen(opw["M"]), opw["C"], T_eval)
                t4[sc] = dict(d_bias_pct=round(r["T2"]["bias_pct"] - base["T2"]["bias_pct"], 2), dU=round(float(abs(U1 - U0)), 3), dD=round(float(abs(D1 - D0)), 3))
                t4[sc]["PASS"] = bool(abs(t4[sc]["d_bias_pct"]) <= 2 and t4[sc]["dU"] <= 0.03 and t4[sc]["dD"] <= 0.03)
                print(f"{cand} T4 {cal} {sc}: dbias {t4[sc]['d_bias_pct']:+.2f} % dU {t4[sc]['dU']:.3f} dD {t4[sc]['dD']:.3f} {'PASS' if t4[sc]['PASS'] else 'FAIL'}")
            res[cand][f"T4:{cal}"] = t4
        E2 = dn_stats(evs["N2"]); EL = dn_stats(evs["NL"]); diff = {k: round(E2[k]["mean"] - EL[k]["mean"], 4) for k in E2 if k in EL}
        res[cand]["T5_kernel_diff_mean_dN"] = diff
    json.dump(res, open(os.path.join(out_dir, "closure_estimator_level.json"), "w"), indent=1)
    return res


def stage_arms(out_dir, study_dir):
    """Calibration-arm comparison with Candidate E (and P for reference): native 2LPT, native London, mock injections, real-spectrum injections."""
    arms = {"N2": load_events("N2", study_dir), "NL": load_events("NL", study_dir), "I2": load_events("I2", study_dir), "IL": load_events("IL", study_dir), "IR": load_events("IR", study_dir)}
    res = {}
    for cand, builder in (("E", build_E), ("P", build_P)):
        ops = {a: builder(arms[a]) for a in arms}; res[cand] = dict(kernels={}, transfer={})
        for a in arms:
            UA, DA = crossing_from_A(coarsen(ops[a]["M"]), ops[a]["C"]); Ua, Da = crossing_aggregate_A(coarsen(ops[a]["M"]), ops[a]["C"], population(arms["N2"])["T"])
            res[cand]["kernels"][a] = dict(dN=dn_stats(arms[a]), U_cells=np.round(UA, 3).tolist(), D_cells=np.round(DA, 3).tolist(), U_on_N2_truth=round(Ua, 4), D_on_N2_truth=round(Da, 4))
        for target in ("N2", "NL"):
            pop = population(arms[target]); pi = pop["T"].sum(axis=2); pi = pi / np.maximum(pi.sum(axis=0, keepdims=True), 1e-12); truth = sums(pop["T"].sum(axis=0))
            row = {}
            for a in arms:
                A = coarsen(ops[a]["M"] * np.nan_to_num(ops[a]["C"])[:, :, None, :]); uf = smooth_unfold(A, pop["n_obs"], pi); est = sums(uf["f"])
                row[a] = dict(ratio_to_truth_ge20p3=round(est["ge20p3"] / truth["ge20p3"], 4), ratio_to_truth_ge20p0=round(est["ge20p0"] / truth["ge20p0"], 4))
            selfv = row[target]["ratio_to_truth_ge20p3"]
            for a in arms:
                row[a]["relative_to_native_self_ge20p3_pct"] = round(100 * (row[a]["ratio_to_truth_ge20p3"] / selfv - 1), 2)
            res[cand]["transfer"][f"catalogue={target}"] = row
            print(f"{cand} catalogue {target}: " + " | ".join(f"{a} {row[a]['ratio_to_truth_ge20p3']:.3f} ({row[a]['relative_to_native_self_ge20p3_pct']:+.1f} %)" for a in arms))
    json.dump(res, open(os.path.join(out_dir, "calibration_arms.json"), "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "closure", "arms"])
    ap.add_argument("--out", default=f"{ROOT}/response_estimator"); ap.add_argument("--study", default=f"{ROOT}/response_study")
    a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    {"build": stage_build, "closure": stage_closure, "arms": stage_arms}[a.stage](a.out, a.study)


if __name__ == "__main__":
    main()
