#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-C response-preimage analysis (PI ruling 2026-08-06 §6).

PURPOSE (stated before the run): compute, from the CURRENT response, the
true-N preimage of the observed groups — for every true-N basis bin b,
its folded contribution to observed G1 [19.7,20.3), G2 [20.3,21.0),
G3 [21.0,21.6], the below-window bins [19.5,19.7), the above-ceiling
in-grid bins [21.6,22.4), and the off-grid mass on both sides — plus the
conditioning decomposition (response cell, SNR stratum, coarse z) and a
per-true-bin SENSITIVITY map (dG3 per kernel mean-shift / width-scale)
that defines the effect-size scale for the §9 power criterion.

STATUS: planning map ONLY.  The preimage is computed THROUGH the current
kernel, so it cannot prove that its own preimage is correct (PI §6).  It
informs anchor placement and the conservative support margin; it is not
evidence about where the failure physically originates.

NO NEW MODEL FREEDOM: the fold is the committed forward.build_K /
forward.build_consts at the truth-equivalent point, exactly as the
closure table calls it (sanity-gated below).  The sensitivity map
perturbs a DIAGNOSTIC COPY of the committed numpy-oracle kernel block
(fold_mu_reference, forward.py) — nothing is fitted, nothing is adopted.

Usage
-----
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python diagnostics_phaseC/preimage/run_preimage.py \
        [--packs-dir DIR] [--out DIR]

MOCKS ONLY.  No real-survey values anywhere in the inputs or outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from CDDF_analysis.hbi_mcmc.pack import load_pack                    # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS            # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K     # noqa: E402
from CDDF_analysis.hbi_mcmc.gate_covariance import (                 # noqa: E402
    group_aggregator, PRIMARY_GROUP_EDGES)

DEFAULT_PACKS = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                 "phaseB_packs")
PACK_FMT = "modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz"
MOCKS = ("2lpt0", "london0", "saclay0")
WIN = (19.7, 21.6)
CEIL = 21.6

#: sensitivity-probe sizes (finite differences on the DIAGNOSTIC oracle copy)
DELTA_MEAN = 0.02      # dex, per-true-bin kernel mean shift
DELTA_WIDTH = 0.10     # fractional per-true-bin kernel sd scale


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_REPO, text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# DIAGNOSTIC COPY of the committed numpy-oracle kernel block
# (forward.fold_mu_reference, lines building mean/sd/skew -> skew-normal ->
# bin mass).  Copied, not shared, so a bug here cannot touch production; the
# copy is verified against build_K to <1e-9 before use (see _verify_oracle).
# Extended with two DIAGNOSTIC perturbations: delta_mean[b] (dex) and
# width_scale[b] (multiplies sd) — zero/one reproduces the committed kernel.
# ---------------------------------------------------------------------------
from scipy.special import ndtr, owens_t                              # noqa: E402

_SKEW_MAX = 0.5 * (4.0 - np.pi) * (np.sqrt(2.0 / np.pi) ** 3) / \
    (1.0 - 2.0 / np.pi) ** 1.5


def _m2sn_vec(mean, sd, skew):
    bb = np.sqrt(2.0 / np.pi)
    s_ = np.clip(skew, -0.995 * _SKEW_MAX, 0.995 * _SKEW_MAX)
    sd = np.maximum(sd, 1e-9)
    cc = 0.5 * (4.0 - np.pi)
    sym = np.abs(s_) < 1e-9
    s_safe = np.where(sym, 1.0, np.abs(s_))
    r = (s_safe / cc) ** (2.0 / 3.0)
    gg = r / (1.0 + r)
    delta = np.clip(np.sign(s_) * np.sqrt(gg) / bb, -0.999, 0.999)
    delta = np.where(sym, 0.0, delta)
    al = delta / np.sqrt(np.maximum(1.0 - delta * delta, 1e-12))
    om = sd / np.sqrt(np.maximum(1.0 - (bb * delta) ** 2, 1e-12))
    xi = mean - om * bb * delta
    return (np.where(sym, mean, xi), np.where(sym, sd, om),
            np.where(sym, 0.0, al))


def _sn_cdf_vec(x, xi, om, al):
    z = (x - xi) / om
    return ndtr(z) - 2.0 * owens_t(z, al)


def oracle_K_edges(pack, consts, edges, delta_mean=None, width_scale=None):
    """K at arbitrary observed EDGES: returns CDF F[s, kk, e, b] at each edge.

    Mirrors fold_mu_reference's per-cell kernel math, vectorized over
    (s, kk, b, e); the D2 covariate clamp, F2 sd floor, F3 skew ramp and F4
    moment match are reproduced exactly.  delta_mean (B,) adds to the kernel
    MEAN (dex); width_scale (B,) multiplies the kernel sd.  Both default to
    identity (committed kernel).
    """
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    B = len(Nc)
    n_ref = float(pack.resp_N_ref)
    rr = np.asarray(pack.resp_N_fit_range, float)          # (SR, ZR, 2)
    mu_co = np.asarray(pack.resp_mu_coef, float)           # (SR, ZR, D)
    sg_co = np.asarray(pack.resp_sig_coef, float)
    sk_co = np.asarray(pack.resp_skew_coef, float)
    D = mu_co.shape[-1]
    sig_floor = float(pack.resp_sig_floor)
    ramp_c, ramp_w = [float(v) for v in np.asarray(pack.resp_skew_ramp, float)]
    s2sr = np.asarray(consts.s_to_sresp)
    K2zr = np.asarray(consts.K_to_zresp)
    S, KK = len(s2sr), len(K2zr)
    if delta_mean is None:
        delta_mean = np.zeros(B)
    if width_scale is None:
        width_scale = np.ones(B)

    # covariate clamp "both" per response cell, gathered to (S, KK, B)
    lo = rr[s2sr][:, K2zr][..., 0]                         # (S, KK)
    hi = rr[s2sr][:, K2zr][..., 1]
    Ncl = np.clip(Nc[None, None, :], lo[..., None], hi[..., None])
    u = Ncl - n_ref                                        # (S, KK, B)
    upow = u[..., None] ** np.arange(D)                    # (S, KK, B, D)
    mu_cell = mu_co[s2sr][:, K2zr]                         # (S, KK, D)
    sg_cell = sg_co[s2sr][:, K2zr]
    sk_cell = sk_co[s2sr][:, K2zr]
    mean = Nc[None, None, :] + np.einsum("skd,skbd->skb", mu_cell, upow) \
        + delta_mean[None, None, :]
    sd = np.maximum(np.einsum("skd,skbd->skb", sg_cell, upow), sig_floor) \
        * width_scale[None, None, :]
    ramp = np.clip((Nc - ramp_c) / ramp_w, 0.0, 1.0)
    skw = np.clip(np.einsum("skd,skbd->skb", sk_cell, upow),
                  -0.995 * _SKEW_MAX, 0.995 * _SKEW_MAX) * (1.0 - ramp)
    xi, om, al = _m2sn_vec(mean, sd, skw)
    edges = np.asarray(edges, float)
    F = _sn_cdf_vec(edges[None, None, :, None], xi[:, :, None, :],
                    om[:, :, None, :], al[:, :, None, :])  # (S, KK, E, B)
    return F


def _verify_oracle(pack, consts, K_committed):
    """The oracle copy must reproduce build_K's bin masses to <1e-9."""
    F = oracle_K_edges(pack, consts, np.asarray(pack.nhat_edges, float))
    Ko = np.clip(F[:, :, 1:, :] - F[:, :, :-1, :], 0.0, 1.0)
    err = float(np.max(np.abs(Ko - K_committed)))
    if err > 1e-9:
        raise RuntimeError(f"oracle copy disagrees with build_K: max|d|={err}")
    return err


def preimage_one(mock, pack_path):
    import jax.numpy as jnp
    pk = load_pack(pack_path)
    consts = build_consts(pk, resp_clamp="both")
    ne = np.asarray(pk.nhat_edges, float)                  # (C+1,)
    ntrue = np.asarray(pk.ntrue_edges, float)              # (B+1,)
    Nc_b = np.asarray(consts.Nc_b)
    B, C = len(Nc_b), pk.n_c
    Kf, S = pk.n_k, pk.n_s
    kz = np.asarray(consts.kz_to_K)
    s2sr = np.asarray(consts.s_to_sresp)
    K2zr = np.asarray(consts.K_to_zresp)
    rr = np.asarray(pk.resp_N_fit_range, float)

    # --- the truth-equivalent contribution (B, Kf, S), exactly as fold_mu ---
    f = FS.truth_f(pk)                                     # (B, Kf)
    C_cells = 1.0 / (1.0 + np.exp(-np.asarray(consts.eta_hat)))   # (S, M)
    C_bs = C_cells[:, np.asarray(consts.b_to_cell)]        # (S, B)
    g_bk = np.asarray(consts.g_bk)                         # (B, Kf)
    dN_b = np.asarray(consts.dN_b)
    dX = np.asarray(pk.dX, float)                          # (Kf, S)
    contrib = (C_bs.T[:, None, :] * g_bk[:, :, None] * f[:, :, None]
               * dN_b[:, None, None]) * dX[None, :, :]     # (B, Kf, S)

    # --- committed kernel ---
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    _verify_oracle(pk, consts, K)                          # (S, KK, C, B)
    K_full = K[:, kz]                                      # (S, Kf, C, B)

    # M[b, c] and strata-resolved M[b, c, s], M[b, c, kk]
    M_bcks = np.einsum("skcb,bks->bcks", K_full, contrib)  # (B, C, Kf, S)
    M_bc = M_bcks.sum(axis=(2, 3))

    # sanity: sum_b M[b,c] must equal the closure fold's signal mu
    st = FS.selftest(pk, resp_clamp="both")
    mu_sig_c = np.asarray(st["mu_sig"]).sum(axis=(1, 2))
    rec_err = float(np.max(np.abs(M_bc.sum(axis=0) - mu_sig_c)
                           / np.maximum(mu_sig_c, 1.0)))
    if rec_err > 1e-8:
        raise RuntimeError(f"preimage does not reconstruct mu_sig: {rec_err}")

    # --- observed-side group columns ---
    A = group_aggregator(pk, PRIMARY_GROUP_EDGES)          # (3, C)
    below = ((ne[:-1] >= 19.5 - 1e-9) & (ne[1:] <= WIN[0] + 1e-9))
    above = (ne[:-1] >= CEIL - 1e-9)                       # in-grid > ceiling
    grp = {"G1": A[0] > 0, "G2": A[1] > 0, "G3": A[2] > 0,
           "below_window": below, "above_ceiling_ingrid": above}

    # off-grid split via the oracle CDF at the two grid ends
    Fends = oracle_K_edges(pk, consts, np.array([ne[0], ne[-1]]))
    off_lo_kf = Fends[:, :, 0, :][:, kz]                   # (S, Kf, B)
    off_hi_kf = (1.0 - Fends[:, :, 1, :])[:, kz]
    off_lo_b = np.einsum("bks,skb->b", contrib, off_lo_kf)
    off_hi_b = np.einsum("bks,skb->b", contrib, off_hi_kf)

    tot_b = contrib.sum(axis=(1, 2))                       # (B,) live truth mass
    tab = {"ntrue_edges": ntrue.tolist(), "true_bin_center": Nc_b.tolist(),
           "truth_counts_total": np.asarray(pk.truth_counts, float)
                                   .sum(axis=1).tolist(),
           "folded_live_total": tot_b.tolist(),
           "off_grid_low": off_lo_b.tolist(),
           "off_grid_high": off_hi_b.tolist()}
    for gname, msk in grp.items():
        tab[gname] = M_bc[:, msk].sum(axis=1).tolist()

    # fractions of each group's signal mu
    frac = {g: (np.array(tab[g]) / max(np.array(tab[g]).sum(), 1e-30)).tolist()
            for g in ("G1", "G2", "G3", "above_ceiling_ingrid")}

    # cumulative G3 coverage from the top contributor down / support margins
    g3_b = np.array(tab["G3"])
    order = np.argsort(g3_b)[::-1]
    csum = np.cumsum(g3_b[order]) / max(g3_b.sum(), 1e-30)
    cover = {}
    for q in (0.95, 0.99, 0.999):
        sel = order[: int(np.searchsorted(csum, q) + 1)]
        cover[str(q)] = {"bins": sorted(int(i) for i in sel),
                         "true_lo": float(ntrue[min(sel)]),
                         "true_hi": float(ntrue[max(sel) + 1])}

    # migration across the ceiling, true-side view
    true_in_g3 = (Nc_b >= 21.0) & (Nc_b < 21.6)
    true_above = Nc_b >= 21.6
    mig = {
        "G3_mu_from_true_below_21.0":
            float(M_bc[Nc_b < 21.0][:, grp["G3"]].sum()),
        "G3_mu_from_true_in_21.0_21.6":
            float(M_bc[true_in_g3][:, grp["G3"]].sum()),
        "G3_mu_from_true_above_21.6":
            float(M_bc[true_above][:, grp["G3"]].sum()),
        "true_21.0_21.6_mass_landing_above_ceiling":
            float(M_bc[true_in_g3][:, grp["above_ceiling_ingrid"]].sum()
                  + off_hi_b[true_in_g3].sum()),
        "true_above_21.6_live_total": float(tot_b[true_above].sum()),
    }

    # conditioning decomposition of G3: by response cell / SNR stratum / kk
    g3_mask_c = grp["G3"]
    M_b_g3_ks = M_bcks[:, g3_mask_c].sum(axis=1)           # (B, Kf, S)
    by_snr = M_b_g3_ks.sum(axis=(0, 1))                    # (S,)
    by_kk = np.zeros(consts.n_kk)
    for k in range(Kf):
        by_kk[kz[k]] += M_b_g3_ks[:, k, :].sum()
    by_cell = np.zeros((consts.n_sr, consts.n_zr))
    clamped_g3 = np.zeros((consts.n_sr, consts.n_zr))
    for s in range(S):
        for k in range(Kf):
            sr, zr = int(s2sr[s]), int(K2zr[kz[k]])
            v = M_b_g3_ks[:, k, s]
            by_cell[sr, zr] += v.sum()
            hi_cl = rr[sr, zr, 1]
            clamped_g3[sr, zr] += v[Nc_b > hi_cl + 1e-9].sum()

    # per-bin clamped status per cell (for the support-label map)
    clamped_share_b = np.zeros(B)
    for b in range(B):
        num = 0.0
        for s in range(S):
            for k in range(Kf):
                sr, zr = int(s2sr[s]), int(K2zr[kz[k]])
                if Nc_b[b] > rr[sr, zr, 1] + 1e-9:
                    num += M_b_g3_ks[b, k, s]
        clamped_share_b[b] = num

    # --- sensitivity map (the §9 effect-size scale) -----------------------
    # dG3[b]: counts moved into G3 by a +DELTA_MEAN dex mean shift of bin b's
    # kernel alone; dG3w[b]: by a +DELTA_WIDTH fractional sd increase.
    g3_lo, g3_hi = 21.0, CEIL
    dG3_mean = np.zeros(B)
    dG3_width = np.zeros(B)
    F0 = oracle_K_edges(pk, consts, np.array([g3_lo, g3_hi]))
    mass0 = (F0[:, :, 1, :] - F0[:, :, 0, :])[:, kz]       # (S, Kf, B)
    base_g3_b = np.einsum("skb,bks->b", mass0, contrib)
    for b in range(B):
        dm = np.zeros(B); dm[b] = DELTA_MEAN
        Fm = oracle_K_edges(pk, consts, np.array([g3_lo, g3_hi]),
                            delta_mean=dm)
        mm = (Fm[:, :, 1, :] - Fm[:, :, 0, :])[:, kz]
        dG3_mean[b] = np.einsum("skb,bks->b", mm, contrib)[b] - base_g3_b[b]
        ws = np.ones(B); ws[b] = 1.0 + DELTA_WIDTH
        Fw = oracle_K_edges(pk, consts, np.array([g3_lo, g3_hi]),
                            width_scale=ws)
        mw = (Fw[:, :, 1, :] - Fw[:, :, 0, :])[:, kz]
        dG3_width[b] = np.einsum("skb,bks->b", mw, contrib)[b] - base_g3_b[b]

    # sanity: reproduce the closure table's 3-group residual for this mock
    mu_all_c = np.asarray(st["mu"])
    live = (dX > 0)[None, :, :]
    mu_c = np.where(live, mu_all_c, 0.0).sum(axis=(1, 2))
    obs_c = np.where(live, np.asarray(st["counts"], float), 0.0).sum(axis=(1, 2))
    d_grp = (A @ obs_c) - (A @ mu_c)
    ct_path = os.path.join(_REPO, "CDDF_analysis/hbi_mcmc/closure_table_phaseB.json")
    ct = json.load(open(ct_path))
    row = next(r for r in ct["rows"] if f"_{mock}_" in r["pack"])
    want = np.asarray(row["predictive"]["residual"], float)
    if not np.allclose(d_grp, want, rtol=0, atol=1e-6):
        raise RuntimeError(
            f"[{mock}] closure-table sanity FAILED: recomputed {d_grp} "
            f"vs table {want}")

    return {
        "pack": pack_path,
        "closure_sanity": {"recomputed_group_residual": d_grp.tolist(),
                           "closure_table_residual": want.tolist(),
                           "reproduced": True},
        "oracle_vs_buildK_max_abs": _verify_oracle(pk, consts, K),
        "reconstruction_rel_err": rec_err,
        "resp_N_fit_range": rr.tolist(),
        "resp_N_ref": float(pk.resp_N_ref),
        "table": tab,
        "group_fractions": frac,
        "g3_coverage": cover,
        "migration": mig,
        "g3_by_snr_stratum": by_snr.tolist(),
        "g3_by_coarse_z": by_kk.tolist(),
        "g3_by_response_cell": by_cell.tolist(),
        "g3_clamped_by_response_cell": clamped_g3.tolist(),
        "g3_clamped_share_per_bin": clamped_share_b.tolist(),
        "g3_total_signal_mu": float(g3_b.sum()),
        "sensitivity": {
            "delta_mean_dex": DELTA_MEAN,
            "delta_width_frac": DELTA_WIDTH,
            "dG3_per_bin_mean_shift": dG3_mean.tolist(),
            "dG3_per_bin_width_scale": dG3_width.tolist(),
        },
        "M_bc_file": None,   # filled by caller
        "_M_bc": M_bc.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs-dir", default=DEFAULT_PACKS)
    ap.add_argument("--out", default=_HERE)
    args = ap.parse_args()

    out = {"schema": "phaseC_preimage/v1",
           "label": ("PLANNING MAP ONLY — computed through the current "
                     "response; cannot validate its own preimage (PI §6)"),
           "window": list(WIN), "ceiling": CEIL,
           "group_edges": [list(g) for g in PRIMARY_GROUP_EDGES],
           "routine": "diagnostics_phaseC/preimage/run_preimage.py",
           "date": time.strftime("%Y-%m-%d"),
           "code_commit": _git(),
           "scope": "MOCK ONLY (2lpt0 / london0 / saclay0). No real-survey values.",
           "mocks": {}}
    for m in MOCKS:
        p = os.path.join(args.packs_dir, PACK_FMT.format(m=m))
        r = preimage_one(m, p)
        mbc = np.asarray(r.pop("_M_bc"))
        npz_path = os.path.join(args.out, f"preimage_M_{m}.npz")
        np.savez(npz_path, M_bc=mbc,
                 ntrue_edges=np.asarray(r["table"]["ntrue_edges"]),
                 note="M_bc[b,c] = folded signal counts from true bin b into "
                      "observed bin c (current kernel; planning map only)")
        r["M_bc_file"] = os.path.basename(npz_path)
        out["mocks"][m] = r
        print(f"[{m}] G3 signal mu = {r['g3_total_signal_mu']:.1f}; "
              f"clamped-cell G3 share = "
              f"{np.sum(r['g3_clamped_by_response_cell']) / max(r['g3_total_signal_mu'], 1e-30):.3f}; "
              f"oracle err = {r['oracle_vs_buildK_max_abs']:.2e}")
    with open(os.path.join(args.out, "preimage.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(args.out, "preimage.json"))


if __name__ == "__main__":
    main()
