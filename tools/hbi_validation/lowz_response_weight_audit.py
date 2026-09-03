#!/usr/bin/env python
"""Low-z response weight-sensitivity audit (PI continuation ruling 2026-09-02 §13; gate MAX4_RESPONSE_ESTIMATOR_CLOSURE_GATE_2026-09-02.md §6).

Mock-only, reduce-only. Reproduces the production ml_shared3 estimator (fitlib "ml" — UNTRUNCATED ML per the adopted_response README — on the recovered stage1b events, EDGES 19.0–21.4 by 0.1,
min_n 50, √n row weights, deg-2 per cell + shared cubic, N_ref from the events file) at unit weights — must match d2b_variants.npz ml_shared3__* —
then refits under importance weights on the truth-host N (flat-in-N, tilt ±0.4) and reports the change of the response surfaces, of the kernel's
threshold crossing on the frozen low-z real pack grids, and of the predicted TP counts when the pooled real posterior median f is folded through the
reweighted surfaces (implied population shift = −Δμ/μ). Nothing is written except the audit JSON. No low-z product is touched.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "CDDF_analysis", "hbi", "adopted_response"))
import fitlib  # noqa: E402
from run_d2b_lib import shared_surfaces  # noqa: E402
sys.path[:] = [p for p in sys.path if "wt_forward_2026_08" not in p]; sys.path.insert(0, REPO)   # fitlib prepends a stale worktree path; CDDF_analysis must resolve to THIS repo

CHAIN = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/code_review_20260826/response_chain/adopted"
PACK = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172.npz"
FDRAWS = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/cp3_real/POOLED_ln_real_v2_20260821_fdraws.npz"
EDGES = np.arange(19.0, 21.4 + 1e-9, 0.1)
MIN_N = 50


def scheme_w(N, scheme):
    if scheme == "unit":
        w = np.ones(len(N))
    elif scheme == "flat":
        h, e = np.histogram(N, np.arange(19.0, 21.4 + 1e-9, 0.1)); i = np.clip(np.searchsorted(e, N, side="right") - 1, 0, len(h) - 1)
        w = np.minimum(1.0 / np.maximum(h[i] / h.max(), 1e-3), 20.0)
    elif scheme == "tilt_minus":
        w = 10 ** (-0.4 * (N - 20.5))
    elif scheme == "tilt_plus":
        w = 10 ** (+0.4 * (N - 20.5))
    else:
        raise ValueError(scheme)
    return w / w.mean()


def refit(N, dx, isr, izr, w, N_ref):
    rows = [[fitlib.subbin_moments(N[(isr == i) & (izr == j)], dx[(isr == i) & (izr == j)], EDGES, MIN_N, "ml",
                                   weights=(None if w is None else w[(isr == i) & (izr == j)])) for j in range(3)] for i in range(3)]
    surf, rng, shared = shared_surfaces(rows, 3, N_ref, n_iter=2)
    return surf, rng, rows


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.count_conserving_fold import surface_masses, phi_from_surfaces, cc_fold_cmarginal
    EV = np.load(os.path.join(CHAIN, "events_full.npz")); V = np.load(os.path.join(CHAIN, "d2b_variants.npz"))
    N_ref = float(EV["N_ref"]); xhat = EV["xhat"]; Nhost = EV["nhi_tilt_host"]
    keep = np.isfinite(Nhost) & (xhat >= 19.5)
    N = Nhost[keep]; dx = (xhat - Nhost)[keep]
    isr = np.clip(np.digitize(EV["snr"][keep], EV["snr_edges"]) - 1, 0, 2); izr = np.clip(np.digitize(EV["zqso"][keep], EV["z_edges"]) - 1, 0, 2)
    out = dict(n_events=int(keep.sum()), N_ref=N_ref, edges=EDGES.tolist(), min_n=MIN_N, schemes={})
    # unit-weight reproduction (gate §6: must match d2b_variants ml_shared3 to <= 1e-6, else STOP)
    surf0, rng0, _ = refit(N, dx, isr, izr, None, N_ref)
    dev = max(float(np.max(np.abs(surf0[k] - V[f"ml_shared3__{k}"]))) for k in ("mu", "sig", "skew"))
    dev_rng = float(np.max(np.abs(rng0 - V["ml_shared3__rng"])))
    out["reproduction"] = dict(max_abs_coef_dev=dev, max_abs_rng_dev=dev_rng, PASS=bool(dev <= 1e-6 and dev_rng <= 1e-9))
    print(f"unit-weight reproduction: max |Δcoef| {dev:.2e}, max |Δrng| {dev_rng:.2e} -> {'PASS' if out['reproduction']['PASS'] else 'STOP'}")
    if not out["reproduction"]["PASS"]:
        json.dump(out, open(a.out, "w"), indent=1); return
    pk = load_pack(PACK)
    ne = np.asarray(pk.nhat_edges, float); nt = np.asarray(pk.ntrue_edges, float); Nc = 0.5 * (nt[:-1] + nt[1:])
    f_draws = np.load(FDRAWS)["f"]; f_med = np.median(f_draws, axis=0)                       # (B, Kf)
    theta = np.log(np.maximum(f_med, 1e-30)); lam0 = np.zeros((len(ne) - 1, len(np.asarray(pk.snr_edges)) - 1))
    phi_dep = phi_from_surfaces(pk)
    def fold(surf, rng):
        mu, parts = cc_fold_cmarginal(pk, theta, lam0, mu_coef=surf["mu"], sig_coef=surf["sig"], skew_coef=surf["skew"], fit_rng=rng, renormalize=True, phi_ref=phi_dep)
        return parts["tp"]
    def kern(surf, rng):
        m, phi = surface_masses(pk, surf["mu"], surf["sig"], surf["skew"], rng, ne); bU = int(np.argmin(np.abs(Nc - 20.2))); bD = int(np.argmin(np.abs(Nc - 20.4)))
        ge = ne[:-1] >= 20.3 - 1e-9
        U = m[:, :, ge, bU].sum(axis=2) / np.maximum(phi[:, :, bU], 1e-12); D = m[:, :, ~ge, bD].sum(axis=2) / np.maximum(phi[:, :, bD], 1e-12)
        return U, D
    tp0 = fold(surf0, rng0); U0, D0 = kern(surf0, rng0); ge3 = ne[:-1] >= 20.3 - 1e-9; ge0 = ne[:-1] >= 20.0 - 1e-9
    Ng = np.round(np.arange(19.6, 21.3 + 1e-9, 0.1), 2)
    def mu_grid(surf, rng):
        g = np.zeros((3, 3, len(Ng)))
        for i in range(3):
            for j in range(3):
                u = np.clip(Ng, rng[i, j, 0], rng[i, j, 1]) - N_ref; g[i, j] = (u[:, None] ** np.arange(4)[None, :]) @ surf["mu"][i, j]
        return g
    mg0 = mu_grid(surf0, rng0)
    out["unit"] = dict(tp_ge20p3=float(tp0[ge3].sum()), tp_ge20p0=float(tp0[ge0].sum()), U=np.round(U0, 4).tolist(), D=np.round(D0, 4).tolist())
    for sc in ("flat", "tilt_minus", "tilt_plus"):
        w = scheme_w(N, sc); surf, rng, rows = refit(N, dx, isr, izr, w, N_ref)
        tp = fold(surf, rng); U, D = kern(surf, rng); mg = mu_grid(surf, rng)
        d3 = tp[ge3].sum() / tp0[ge3].sum() - 1; d0 = tp[ge0].sum() / tp0[ge0].sum() - 1
        rec = dict(n_eff=round(float(w.sum() ** 2 / (w ** 2).sum()), 1), anchors_per_cell=[[len([r for r in rows[i][j] if r["ok"]]) for j in range(3)] for i in range(3)],
                   fit_range=np.round(rng, 3).tolist(), d_mu_max_dex=round(float(np.max(np.abs(mg - mg0))), 4),
                   d_mu_by_cell=np.round(np.max(np.abs(mg - mg0), axis=2), 4).tolist(), d_sig_coef=np.round(surf["sig"] - surf0["sig"], 4).tolist(),
                   dU_max=round(float(np.max(np.abs(U - U0))), 4), dD_max=round(float(np.max(np.abs(D - D0))), 4),
                   d_tp_ge20p3_pct=round(100 * d3, 3), d_tp_ge20p0_pct=round(100 * d0, 3), implied_pop_shift_ge20p3_pct=round(-100 * d3, 3), implied_pop_shift_ge20p0_pct=round(-100 * d0, 3))
        rec["PASS"] = bool(abs(rec["implied_pop_shift_ge20p3_pct"]) <= 2 and abs(rec["implied_pop_shift_ge20p0_pct"]) <= 2 and rec["dU_max"] <= 0.03 and rec["dD_max"] <= 0.03)
        out["schemes"][sc] = rec
        print(f"{sc}: n_eff {rec['n_eff']} d_mu max {rec['d_mu_max_dex']:.4f} dU {rec['dU_max']:.4f} dD {rec['dD_max']:.4f} implied shift ≥20.3 {rec['implied_pop_shift_ge20p3_pct']:+.3f} % ≥20.0 {rec['implied_pop_shift_ge20p0_pct']:+.3f} % -> {'PASS' if rec['PASS'] else 'MATERIAL'}")
    out["verdict"] = "PASS / NO ACTION" if all(r["PASS"] for r in out["schemes"].values()) else "MATERIAL"
    json.dump(out, open(a.out, "w"), indent=1); print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
