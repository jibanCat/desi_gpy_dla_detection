#!/usr/bin/env python
"""omega_anatomy.py — Omega_DLA systematic anatomy from FROZEN Paper-1 inputs (R-037).

Paper-1 request package 2026-08-28, R-037 (PI, via the paper lane): how sensitive is the
inferred Omega_DLA(z) to the selection / response / transport / high-N_HI assumptions
that affect dN/dX, given the mass weighting of the high-column-density tail?

Everything here is a REDUCTION of frozen products; nothing is fitted, tuned or written
into any frozen artifact. The reduction is the paper-facing one (gp_dla_desi_y3
paper_figures/hbi_reduction.py @ the paper tag paper1-figures-2026-08-26):

    Omega(z_lo, z_hi; n_min, n_max) per draw d
        = PREF * sum_b w_b sum_k zw_k f[d, b, k] / sum_k zw_k
    w_b  = (10^b2 - 10^a2) / ln 10  on the latent bin [a, b) clipped to [n_min, n_max]
           (f is per dex and taken FLAT IN log N inside a latent bin)
    zw_k = dX_k * |cell_k ∩ [z_lo, z_hi)| / |cell_k|
    PREF = H0 m_p / (c rho_crit) in cm^2, h = 0.70, Omega_m = 0.279 (PI ruling 2026-08-26 #48)

and the closure of the adopted numbers against the paper-facing npz is asserted before
anything else is written. Quantiles are the frozen 2.5 / 16 / 50 / 84 / 97.5 levels.

Named lines (systematics ledger v2.3 r5) are propagated to Omega ONLY where a
draw-level carrier exists in the frozen record:
  * L15 (configuration ambiguity)  — the s26 mirror chain of the deep s20260826 run
                                     (the same carrier the frozen ledger / audit use);
  * mock-recovery envelope + L1     — the 16 certified corrected-g validation runs
                                     (Battery 2+3 DIAGPACK_gcons + CP-2 production packs):
                                     Omega_post / Omega_truth - 1 per run, family min..max;
  * L13 and the other predeclared nuisance arms — the 2LPT-0 s20260818 sensitivity runs
                                     (fp_s_empty 1.5 / 3.0, t_sigma x2, total-width x2)
                                     against the same-seed primary;
  * L2 (mock-to-real transport)    — a GLOBAL scalar band on the >=20.3 counting statistic
                                     in the frozen record; no N-resolved transport above
                                     20.3 exists, so it is carried as the same scalar band
                                     and flagged (see the JSON 'lines' block).
Migration (items 11, 13) is read from the pack's ADOPTED response surfaces exactly as the
production count-conserving fold applies them (count_conserving_fold.cc_fold_cmarginal
with renormalize=True and the stored adopted_phi_ref), at the posterior-median f.

Status of every output: DIAGNOSTIC (R-037 return contract item 7) unless the JSON says
otherwise. No quadrature anywhere. Real-data VALUES never enter this file or its tests.

Usage (all inputs explicit; nothing defaults to a frozen path):
  python -m CDDF_analysis.hbi_mcmc.omega_anatomy \
      --draws <POOLED fdraws.npz> --pack <real pack v2 npz> --summary <POOLED json> \
      --paper-dndx-npz <fig_hbi_dndx.data.npz> --paper-cddf-npz <fig_hbi_cddf.data.npz> \
      --mirror <deep fdraws.npz> --mirror-chain 0 --mirror-chains 2 \
      --mock-runs <16 validation fdraws.npz ...> \
      --sens-primary <perz_gcons ... s20260818 fdraws.npz> \
      --sens-arms name=path ... \
      --audit <cddf_recovery_audit.json> --ledger <ledger_v2p3_cp3.json> \
      --bh-artifact <RATIFIED BH json> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

# --- the paper-facing reporting constants (hbi_reduction.py @ paper tag) -------------
LOWZ_BINS = [("B1", 2.15, 2.35), ("B2", 2.35, 2.56), ("B3", 2.56, 2.96),
             ("B4", 2.96, 3.40), ("B5", 3.40, 3.80)]
Z_SLICES = [("z2.0-2.5", 2.0, 2.5), ("z2.5-3.0", 2.5, 3.0), ("z3.0-3.5", 3.0, 3.5)]
LOWZ_SUPPORT = (2.0, 3.5)
QUANTILES = [2.5, 16.0, 50.0, 84.0, 97.5]
OMEGA_NHI = (20.3, 21.6)
RESPONSE_CALIBRATED_TOP = 21.35
H_REPORTING = 0.70
OMEGA_M_REPORTING = 0.279
_MPC_M = 3.0856775814913673e22
_H0 = H_REPORTING * 100.0e3 / _MPC_M
_M_P = 1.67262192369e-27
_C = 2.99792458e8
_G = 6.67430e-11
_RHO_CRIT = 3.0 * _H0 ** 2 / (8.0 * math.pi * _G)
OMEGA_PREFACTOR_CM2 = (_H0 * _M_P / (_C * _RHO_CRIT)) * 1.0e4
UPPER_LIMIT_SCAN = [21.1, 21.3, 21.5, 21.6, 21.7, 21.9, 22.1, 22.4]
FAMILY_OF = ("2lpt0", "london0", "saclay0")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here).decode().strip()
    except Exception:
        return "unknown"


def _q(x, axis=0):
    return np.percentile(np.asarray(x, float), QUANTILES, axis=axis)


def _overlap(a_lo, a_hi, b_lo, b_hi):
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


# --- weights (identical to hbi_reduction.Posterior) ------------------------------------
def omega_weight(n_edges, n_min, n_max):
    lo, hi = n_edges[:-1], n_edges[1:]
    w = np.zeros_like(lo)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if _overlap(a, b, n_min, n_max) <= 0.0:
            continue
        a2, b2 = max(a, n_min), min(b, n_max)
        w[i] = (10.0 ** b2 - 10.0 ** a2) / math.log(10.0)
    return w


def nhi_weight(n_edges, threshold):
    lo, hi = n_edges[:-1], n_edges[1:]
    return np.maximum(0.0, hi - np.maximum(lo, threshold))


def z_weight(z_edges, dX_k, z_lo, z_hi):
    ze = np.asarray(z_edges, float)
    return np.array([dX_k[k] * _overlap(ze[k], ze[k + 1], z_lo, z_hi) / (ze[k + 1] - ze[k])
                     for k in range(len(dX_k))])


def reduce_draws(f, nw, zw):
    """per-draw sum_b nw_b sum_k zw_k f[d,b,k] / sum_k zw_k  (f: (D,B,K) or (B,K))."""
    f = np.asarray(f, float)
    if f.ndim == 2:
        return float(np.einsum("bk,b,k->", f, nw, zw) / zw.sum())
    return np.einsum("dbk,b,k->d", f, nw, zw) / zw.sum()


def omega_per_draw(f, n_edges, zw, n_min=OMEGA_NHI[0], n_max=OMEGA_NHI[1]):
    return OMEGA_PREFACTOR_CM2 * reduce_draws(f, omega_weight(n_edges, n_min, n_max), zw)


def domains(z_edges):
    """The reporting domains: all-z, the five Paper-1 bins clipped to the support, three
    z slices. Returns [(name, z_lo, z_hi, coverage)]."""
    out = [("allz", LOWZ_SUPPORT[0], LOWZ_SUPPORT[1], 1.0)]
    for name, lo, hi in LOWZ_BINS:
        cov = _overlap(lo, hi, *LOWZ_SUPPORT) / (hi - lo)
        out.append((name, lo, hi, cov))
    for name, lo, hi in Z_SLICES:
        out.append((name, lo, hi, 1.0))
    return out


# --- per-bin contribution machinery ----------------------------------------------------
def window_bins(n_edges, n_min=OMEGA_NHI[0], n_max=OMEGA_NHI[1]):
    """Latent bins overlapping the Omega window, with their clipped edges."""
    lo, hi = n_edges[:-1], n_edges[1:]
    out = []
    for i, (a, b) in enumerate(zip(lo, hi)):
        if _overlap(a, b, n_min, n_max) > 0.0:
            out.append((i, float(max(a, n_min)), float(min(b, n_max)), float(a), float(b)))
    return out


def contributions(f, n_edges, zw, n_min=OMEGA_NHI[0], n_max=OMEGA_NHI[1]):
    """(D, nb) per-draw Omega contribution of each window bin (sums to omega_per_draw)."""
    wb = window_bins(n_edges, n_min, n_max)
    full = omega_weight(n_edges, n_min, n_max)
    cols = []
    for (i, a2, b2, a, b) in wb:
        w = np.zeros_like(full)
        w[i] = full[i]
        cols.append(OMEGA_PREFACTOR_CM2 * reduce_draws(f, w, zw))
    return np.stack(cols, axis=1), wb


def powerlaw_intrabin_omega(f, n_edges, zw, n_min=OMEGA_NHI[0], n_max=OMEGA_NHI[1]):
    """Alternative intra-bin shape: inside each latent bin f_dex follows a local power law
    10^(beta (logN - logN_c)) with beta from the centred difference of ln f_dex over the
    neighbouring bins (per draw, per z cell), normalised so the bin's dex-average is
    unchanged (the count in the bin is conserved; only its N-weighting moves). The
    adopted reduction is beta = 0 (flat in log N). Returns per-draw Omega."""
    f = np.asarray(f, float)
    D, B, K = f.shape
    lo, hi = n_edges[:-1], n_edges[1:]
    logf = np.log(np.maximum(f, 1e-300))
    beta = np.zeros_like(f)
    for i in range(B):
        i0, i1 = max(i - 1, 0), min(i + 1, B - 1)
        dl = 0.5 * (lo[i1] + hi[i1]) - 0.5 * (lo[i0] + hi[i0])
        beta[:, i, :] = (logf[:, i1, :] - logf[:, i0, :]) / dl / math.log(10.0)
    out = np.zeros(D)
    for (i, a2, b2, a, b) in window_bins(n_edges, n_min, n_max):
        c = 0.5 * (a + b)
        # dex-average normaliser over the FULL bin [a,b): mean of 10^(beta (x-c)) over x
        g = beta[:, i, :] * math.log(10.0)                       # (D,K)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(np.abs(g) < 1e-9, (b - a),
                            (np.exp(g * (b - c)) - np.exp(g * (a - c))) / g)   # ∫_a^b e^{g(x-c)} dx
            # ∫_{a2}^{b2} 10^x e^{g (x-c)} dx  = ∫ e^{(ln10 + g)(x - c)} 10^c dx
            gg = math.log(10.0) + g
            num = 10.0 ** c * (np.exp(gg * (b2 - c)) - np.exp(gg * (a2 - c))) / gg
        wbin = (b - a) * num / norm                                # (D,K) effective weight
        out += np.einsum("dk,dk,k->d", f[:, i, :], wbin, zw) / zw.sum()
    return OMEGA_PREFACTOR_CM2 * out


def treatment_record(per_draw, ref_median):
    q = _q(per_draw)
    return {"q_p2p5_16_50_84_97p5": q.tolist(),
            "median_over_adopted": float(q[2] / ref_median),
            "median_shift_pct": float(100.0 * (q[2] / ref_median - 1.0)),
            "halfwidth68_pct": float(50.0 * (q[3] - q[1]) / q[2]),
            "halfwidth95_pct": float(50.0 * (q[4] - q[0]) / q[2])}


# --- migration from the adopted response surfaces ---------------------------------------
def migration_block(pack_npz, f_median, n_edges, z_lo, z_hi):
    """Expected TP detections by (N-hat bin, true bin) under the production fold at the
    posterior-median f, path-restricted to [z_lo, z_hi) through the pack's dX. Uses the
    same masses (renormalised x adopted_phi_ref) as cc_fold_adopted."""
    from CDDF_analysis.hbi_mcmc.count_conserving_fold import cc_fold_cmarginal, phi_from_surfaces
    d = {k: pack_npz[k] for k in pack_npz.files}
    pk = SimpleNamespace(**d)
    phi_stored = np.asarray(pk.adopted_phi_ref, float)
    phi_fresh = phi_from_surfaces(pk)
    assert float(np.max(np.abs(phi_stored - phi_fresh))) <= 1e-9, "adopted_phi_ref corrupt (G-CC)"
    zf = np.asarray(pk.zf_edges, float)
    ov = np.array([_overlap(zf[k], zf[k + 1], z_lo, z_hi) / (zf[k + 1] - zf[k]) for k in range(len(zf) - 1)])
    pk.dX = np.asarray(pk.dX, float) * ov[:, None]
    pk.fp_E_alloc = np.asarray(pk.fp_E_alloc, float)
    lam_fp = np.zeros((len(pk.nhat_edges) - 1, len(pk.snr_edges) - 1))
    _, parts = cc_fold_cmarginal(pk, np.log(np.maximum(f_median, 1e-300)), lam_fp,
                                 mu_coef=pk.adopted_resp_mu_coef, sig_coef=pk.adopted_resp_sig_coef,
                                 skew_coef=pk.adopted_resp_skew_coef,
                                 fit_rng=np.asarray(pk.adopted_resp_fit_range, float),
                                 renormalize=True, phi_ref=phi_stored, return_contrib=True)
    return parts["contrib_cb"], np.asarray(pk.nhat_edges, float)


def migration_summary(contrib, nhat_edges, n_edges):
    """Row/column-normalised flows across the window edges, in counts and in true-N mass."""
    ne, te = nhat_edges, n_edges
    nc = 0.5 * (ne[:-1] + ne[1:])
    tc = 0.5 * (te[:-1] + te[1:])
    Ntrue = 10.0 ** tc
    classes_obs = {"nhat<20.3": nc < 20.3, "20.3<=nhat<21.3": (nc >= 20.3) & (nc < 21.3),
                   "21.3<=nhat<21.6": (nc >= 21.3) & (nc < 21.6), "nhat>=21.6": nc >= 21.6}
    classes_true = {"true<20.3": tc < 20.3, "20.3<=true<21.3": (tc >= 20.3) & (tc < 21.3),
                    "21.3<=true<21.7": (tc >= 21.3) & (tc < 21.7), "true>=21.7": tc >= 21.7}
    out = {"per_true_bin_outflow": [], "per_observed_class_composition": {},
           "per_observed_class_composition_mass_weighted": {}}
    tot_in = contrib.sum(axis=0)                    # expected in-grid detections per true bin
    for b in range(len(tc)):
        if te[b] < 20.3 - 1e-9:
            continue
        row = {"true_bin": [float(te[b]), float(te[b + 1])], "expected_in_grid_detections": float(tot_in[b])}
        for name, m in classes_obs.items():
            row["P(" + name + " | true bin)"] = float(contrib[m, b].sum() / tot_in[b]) if tot_in[b] > 0 else None
        out["per_true_bin_outflow"].append(row)
    for name, m in classes_obs.items():
        col = contrib[m, :].sum(axis=0)             # by true bin
        s = col.sum()
        out["per_observed_class_composition"][name] = {
            "expected_detections": float(s),
            **{"P(" + tn + " | " + name + ")": float(col[tm].sum() / s) if s > 0 else None
               for tn, tm in classes_true.items()}}
        colm = col * Ntrue
        sm = colm.sum()
        out["per_observed_class_composition_mass_weighted"][name] = {
            **{"mass_frac(" + tn + " | " + name + ")": float(colm[tm].sum() / sm) if sm > 0 else None
               for tn, tm in classes_true.items()}}
    return out


def kernel_cell_table(pack_npz, n_edges):
    """Raw adopted kernel per response cell (SNR cell x z cell): for the true bins at the
    top of the window, P(N-hat >= 21.6), P(N-hat < 21.3), P(21.3 <= N-hat < 21.6) and the
    off-grid fraction 1 - phi (counts leaving the observed grid, ceiling behaviour)."""
    from CDDF_analysis.hbi_mcmc.count_conserving_fold import surface_masses
    d = {k: pack_npz[k] for k in pack_npz.files}
    pk = SimpleNamespace(**d)
    ne = np.asarray(pk.nhat_edges, float)
    masses, phi = surface_masses(pk, pk.adopted_resp_mu_coef, pk.adopted_resp_sig_coef,
                                 pk.adopted_resp_skew_coef, np.asarray(pk.adopted_resp_fit_range, float), ne)
    nc = 0.5 * (ne[:-1] + ne[1:])
    rows = []
    for b in range(len(n_edges) - 1):
        if n_edges[b] < 21.1 - 1e-9:
            continue
        rec = {"true_bin": [float(n_edges[b]), float(n_edges[b + 1])], "cells": []}
        for sr in range(masses.shape[0]):
            for zr in range(masses.shape[1]):
                m = masses[sr, zr, :, b]
                rec["cells"].append({"snr_cell": sr, "z_cell": zr, "in_grid_phi": float(phi[sr, zr, b]),
                                     "P_nhat_ge_21p6": float(m[nc >= 21.6].sum()),
                                     "P_nhat_in_21p3_21p6": float(m[(nc >= 21.3) & (nc < 21.6)].sum()),
                                     "P_nhat_lt_21p3": float(m[nc < 21.3].sum()),
                                     "P_nhat_lt_20p3": float(m[nc < 20.3].sum())})
        rows.append(rec)
    return {"resp_snr_edges": np.asarray(pk.resp_snr_edges, float).tolist(),
            "resp_z_edges": np.asarray(pk.resp_z_edges, float).tolist(), "rows": rows}


# --- the anatomy for one draw set on one domain ---------------------------------------
def anatomy(f, n_edges, zw, envelope=None, mirror_f=None, label=""):
    """All R-037 items that are pure functions of a draw set (and optional carriers)."""
    om = omega_per_draw(f, n_edges, zw)
    q = _q(om)
    med = float(q[2])
    dn = reduce_draws(f, nhi_weight(n_edges, 20.3), zw)
    rec = {"omega_adopted": {"q_p2p5_16_50_84_97p5": q.tolist(), "n_min": OMEGA_NHI[0], "n_max": OMEGA_NHI[1],
                             "halfwidth68_pct": float(50.0 * (q[3] - q[1]) / q[2]),
                             "halfwidth95_pct": float(50.0 * (q[4] - q[0]) / q[2])},
           "dndx_ge20p3": {"q_p2p5_16_50_84_97p5": _q(dn).tolist()}}
    # 9 / 10: contributions per latent bin, top-one/two share, calibrated vs response-held
    C, wb = contributions(f, n_edges, zw)
    frac = C / om[:, None]
    bins = [[a2, b2] for (_, a2, b2, _, _) in wb]
    tiers = ["calibrated" if b2 <= RESPONSE_CALIBRATED_TOP + 1e-9 else "response_held" for (_, a2, b2, _, _) in wb]
    rec["contribution_bins"] = bins
    rec["contribution_tier"] = tiers
    rec["contribution_q"] = _q(C).T.tolist()                    # (nb, 5)
    rec["contribution_fraction_q"] = _q(frac).T.tolist()
    top1 = frac[:, -1]
    top2 = frac[:, -2:].sum(axis=1)
    held = frac[:, [i for i, t in enumerate(tiers) if t == "response_held"]].sum(axis=1)
    rec["fraction_top1_bin_q"] = _q(top1).tolist()
    rec["fraction_top2_bins_q"] = _q(top2).tolist()
    rec["fraction_response_held_q"] = _q(held).tolist()
    # 14: variance decomposition (per-draw covariance of the contributions)
    cov = np.cov(C, rowvar=False)
    var = float(np.var(om))
    share = cov.sum(axis=1) / var
    rec["variance_share_by_bin"] = share.tolist()
    rec["variance_share_top2"] = float(share[-2:].sum())
    rec["variance_share_response_held"] = float(sum(share[i] for i, t in enumerate(tiers) if t == "response_held"))
    sd = np.sqrt(np.diag(cov))
    rec["contribution_correlation"] = (cov / np.outer(sd, sd)).tolist()
    rec["contribution_rel_halfwidth68_pct"] = [float(50.0 * (qq[3] - qq[1]) / qq[2]) for qq in _q(C).T]
    # 12: upper integration limit scan
    rec["upper_limit_scan"] = {}
    for X in UPPER_LIMIT_SCAN:
        rec["upper_limit_scan"][str(X)] = treatment_record(omega_per_draw(f, n_edges, zw, OMEGA_NHI[0], X), med)
    # 15 / 16: altered treatments
    T = {"adopted": treatment_record(om, med)}
    for i, (bidx, a2, b2, a, b) in enumerate(wb):
        T[f"leave_out_[{a2},{b2}]"] = treatment_record(om - C[:, i], med)
    T["calibrated_only_top_21p3"] = rec["upper_limit_scan"]["21.3"]
    T["top_21p7"] = rec["upper_limit_scan"]["21.7"]
    T["top_22p4_open"] = rec["upper_limit_scan"]["22.4"]
    T["intrabin_local_powerlaw"] = treatment_record(powerlaw_intrabin_omega(f, n_edges, zw), med)
    # the same alternative bin by bin at the posterior median: the net is a CANCELLATION between
    # the fully-contained bins (a falling slope moves mass to the lower edge) and the clipped
    # ceiling bin (the same slope puts more of [21.5,21.7)'s mass into its reported lower part)
    fmed = np.median(f, axis=0)[None]
    per = []
    for (bidx, a2, b2, a, b) in wb:
        w = np.zeros(len(n_edges) - 1)
        w[bidx] = omega_weight(n_edges, OMEGA_NHI[0], OMEGA_NHI[1])[bidx]
        flat = OMEGA_PREFACTOR_CM2 * reduce_draws(fmed, w, zw)[0]
        per.append(float(powerlaw_intrabin_omega(fmed, n_edges, zw, a2, b2)[0] / flat))
    rec["intrabin_powerlaw_per_bin_ratio_at_median"] = per
    if envelope is not None:
        # coherent truth-side correction of the response-held bins by the audit envelope
        # (f_truth = f_post / (1 + bias)); both held bins moved together to the family
        # extreme; a DIAGNOSTIC bracket, not a CI (sys_layers.py semantics)
        for tag in ("bias_min", "bias_max"):
            fa = np.array(f, float, copy=True)
            for (bidx, a2, b2, a, b) in wb:
                key = (round(a, 1), round(b, 1))
                if tiers[[w[0] for w in wb].index(bidx)] == "response_held" and key in envelope:
                    fa[:, bidx, :] = fa[:, bidx, :] / (1.0 + envelope[key][tag])
            T[f"response_held_bins_truth_side_{tag}"] = treatment_record(omega_per_draw(fa, n_edges, zw), med)
        fa = np.array(f, float, copy=True)
        for (bidx, a2, b2, a, b) in wb:
            key = (round(a, 1), round(b, 1))
            if key in envelope:
                fa[:, bidx, :] = fa[:, bidx, :] / (1.0 + envelope[key]["bias_max"])
        T["all_window_bins_truth_side_bias_max"] = treatment_record(omega_per_draw(fa, n_edges, zw), med)
        fa = np.array(f, float, copy=True)
        for (bidx, a2, b2, a, b) in wb:
            key = (round(a, 1), round(b, 1))
            if key in envelope:
                fa[:, bidx, :] = fa[:, bidx, :] / (1.0 + envelope[key]["bias_min"])
        T["all_window_bins_truth_side_bias_min"] = treatment_record(omega_per_draw(fa, n_edges, zw), med)
    if mirror_f is not None:
        T["L15_mirror_configuration"] = treatment_record(omega_per_draw(mirror_f, n_edges, zw), med)
        dm = reduce_draws(mirror_f, nhi_weight(n_edges, 20.3), zw)
        rec["L15_mirror_dndx_ge20p3_over_pooled_minus1_pct"] = float(100.0 * (np.median(dm) / np.median(dn) - 1.0))
        Cm, _ = contributions(mirror_f, n_edges, zw)
        rec["L15_mirror_contribution_ratio_to_pooled_median"] = (np.median(Cm, axis=0) / np.median(C, axis=0)).tolist()
    rec["treatments"] = T
    return rec


def envelope_from_audit(audit, slice_key):
    """Per-0.2-dex family min..max of the posterior-median recovery bias (fractions)."""
    fs = audit["mock"][slice_key]["family_summary"]
    out = {}
    for key, rec in fs.items():
        a, b = [float(x) for x in key.strip("()").split(",")]
        fams = [f for f in FAMILY_OF if f in rec]
        out[(round(a, 1), round(b, 1))] = {"bias_min": min(rec[f]["min"] for f in fams) / 100.0,
                                           "bias_max": max(rec[f]["max"] for f in fams) / 100.0}
    return out


def mock_recovery(run_paths):
    """Omega and dN/dX(>=20.3) posterior recovery vs the pack's own truth, per run, all
    domains; plus the recovery of the response-held contribution and the truth-side top
    share. run_paths: {stem: path}."""
    per_run = {}
    for stem, p in run_paths.items():
        z = np.load(p)
        f, tf, ne, ze, dXk = z["f"], z["truth_f"], np.asarray(z["ntrue_edges"], float), np.asarray(z["zf_edges"], float), np.asarray(z["dX_k"], float)
        fam = next((x for x in FAMILY_OF if x in stem), "unknown")
        rec = {"family": fam, "file": p, "sha256": _sha(p), "domains": {}}
        for name, lo, hi, cov in domains(ze):
            zw = z_weight(ze, dXk, lo, hi)
            if zw.sum() <= 0:
                continue
            om = omega_per_draw(f, ne, zw)
            omt = omega_per_draw(tf, ne, zw)
            dn = reduce_draws(f, nhi_weight(ne, 20.3), zw)
            dnt = reduce_draws(tf, nhi_weight(ne, 20.3), zw)
            C, wb = contributions(f, ne, zw)
            Ct, _ = contributions(tf[None], ne, zw)          # truth as a one-draw set
            tiers = ["calibrated" if b2 <= RESPONSE_CALIBRATED_TOP + 1e-9 else "response_held" for (_, a2, b2, _, _) in wb]
            held = [i for i, t in enumerate(tiers) if t == "response_held"]
            qo = _q(om)
            rec["domains"][name] = {
                "omega_post_q": qo.tolist(), "omega_truth": float(omt),
                "omega_median_bias_pct": float(100.0 * (qo[2] / omt - 1.0)),
                "omega_truth_in_68": bool(qo[1] <= omt <= qo[3]), "omega_truth_in_95": bool(qo[0] <= omt <= qo[4]),
                "dndx_ge20p3_median_bias_pct": float(100.0 * (np.median(dn) / dnt - 1.0)),
                "response_held_contribution_median_bias_pct": float(100.0 * (np.median(C[:, held].sum(axis=1)) / Ct[:, held].sum() - 1.0)),
                "top2_contribution_median_bias_pct": float(100.0 * (np.median(C[:, -2:].sum(axis=1)) / Ct[:, -2:].sum() - 1.0)),
                "truth_fraction_top2_bins": float(Ct[:, -2:].sum() / Ct.sum()),
                "post_fraction_top2_bins_median": float(np.median(C[:, -2:].sum(axis=1) / om)),
                "per_bin_contribution_median_bias_pct": (100.0 * (np.median(C, axis=0) / Ct[0] - 1.0)).tolist(),
                "contribution_bins": [[a2, b2] for (_, a2, b2, _, _) in wb]}
        per_run[stem] = rec
    # family summaries per domain
    summary = {}
    for name in {d for r in per_run.values() for d in r["domains"]}:
        summary[name] = {}
        for key in ("omega_median_bias_pct", "dndx_ge20p3_median_bias_pct",
                    "response_held_contribution_median_bias_pct", "top2_contribution_median_bias_pct"):
            fam_rec = {}
            allv = []
            for fam in FAMILY_OF:
                v = [r["domains"][name][key] for r in per_run.values() if r["family"] == fam and name in r["domains"]]
                if v:
                    fam_rec[fam] = {"n_runs": len(v), "min": float(min(v)), "max": float(max(v)), "mean": float(np.mean(v))}
                    allv += v
            fam_rec["all_families"] = {"n_runs": len(allv), "min": float(min(allv)), "max": float(max(allv))}
            summary[name][key] = fam_rec
    return per_run, summary


def sensitivity_arms(primary, arms):
    P = np.load(primary)
    ne, ze, dXk = np.asarray(P["ntrue_edges"], float), np.asarray(P["zf_edges"], float), np.asarray(P["dX_k"], float)
    out = {"primary": {"file": primary, "sha256": _sha(primary)}, "arms": {}}
    for name, p in arms.items():
        A = np.load(p)
        rec = {"file": p, "sha256": _sha(p), "domains": {}}
        for dname, lo, hi, cov in domains(ze):
            zw = z_weight(ze, dXk, lo, hi)
            if zw.sum() <= 0:
                continue
            om0, om1 = omega_per_draw(P["f"], ne, zw), omega_per_draw(A["f"], ne, zw)
            dn0, dn1 = reduce_draws(P["f"], nhi_weight(ne, 20.3), zw), reduce_draws(A["f"], nhi_weight(ne, 20.3), zw)
            rec["domains"][dname] = {"omega_median_shift_pct": float(100.0 * (np.median(om1) / np.median(om0) - 1.0)),
                                     "dndx_ge20p3_median_shift_pct": float(100.0 * (np.median(dn1) / np.median(dn0) - 1.0)),
                                     "omega_halfwidth68_pct_primary": float(50.0 * np.diff(np.percentile(om0, [16, 84]))[0] / np.median(om0)),
                                     "omega_halfwidth68_pct_arm": float(50.0 * np.diff(np.percentile(om1, [16, 84]))[0] / np.median(om1))}
        out["arms"][name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--paper-dndx-npz", required=True)
    ap.add_argument("--paper-cddf-npz", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--mirror-chain", type=int, default=0)
    ap.add_argument("--mirror-chains", type=int, default=2)
    ap.add_argument("--mock-runs", nargs="+", required=True)
    ap.add_argument("--sens-primary", required=True)
    ap.add_argument("--sens-arms", nargs="+", required=True, help="name=path")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--bh-artifact", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    D = np.load(a.draws)
    P = np.load(a.pack, allow_pickle=True)
    f = np.asarray(D["f"], float)
    ne = np.asarray(D["ntrue_edges"], float)
    ze = np.asarray(D["zf_edges"], float)
    assert np.allclose(ze, np.asarray(P["zf_edges"], float)) and np.allclose(ne, np.asarray(P["ntrue_edges"], float))
    dXk = np.asarray(P["dX"], float).sum(axis=1)
    M = np.load(a.mirror)
    nchain = M["f"].shape[0] // a.mirror_chains
    fm = np.asarray(M["f"][a.mirror_chain * nchain:(a.mirror_chain + 1) * nchain], float)
    audit = json.load(open(a.audit))
    ledger = json.load(open(a.ledger))
    paper = np.load(a.paper_dndx_npz, allow_pickle=True)
    cddf = np.load(a.paper_cddf_npz, allow_pickle=True)
    summary = json.load(open(a.summary))

    # ---- closure against the paper-facing products (must hold before anything else) ----
    zw_all = z_weight(ze, dXk, *LOWZ_SUPPORT)
    om_all = _q(omega_per_draw(f, ne, zw_all))
    closure = {"allz_omega_20p3_max_rel": float(np.max(np.abs(om_all / np.asarray(paper["allz_omega_20p3"], float) - 1.0)))}
    per_bin = []
    for (name, lo, hi), row in zip(LOWZ_BINS, np.asarray(paper["lowz_omega_20p3"], float)):
        zw = z_weight(ze, dXk, lo, hi)
        per_bin.append(float(np.max(np.abs(_q(omega_per_draw(f, ne, zw)) / row - 1.0))))
    closure["lowz_omega_20p3_max_rel_per_bin"] = per_bin
    closure["dndx_20p3_allz_max_rel"] = float(np.max(np.abs(_q(reduce_draws(f, nhi_weight(ne, 20.3), zw_all)) / np.asarray(paper["allz_dndx_20p3"], float) - 1.0)))
    # item 17: what omega_20p0 / omega_20p3 in fig_hbi_cddf.data.npz integrate
    om_20p0 = _q(omega_per_draw(f, ne, zw_all, 20.0, 21.6))
    closure["cddf_npz_omega_20p0_is_[20.0,21.6]_max_rel"] = float(np.max(np.abs(om_20p0 / np.asarray(cddf["omega_20p0"], float) - 1.0)))
    closure["cddf_npz_omega_20p3_is_[20.3,21.6]_max_rel"] = float(np.max(np.abs(om_all / np.asarray(cddf["omega_20p3"], float) - 1.0)))
    worst = max(closure["allz_omega_20p3_max_rel"], max(per_bin), closure["dndx_20p3_allz_max_rel"],
                closure["cddf_npz_omega_20p0_is_[20.0,21.6]_max_rel"], closure["cddf_npz_omega_20p3_is_[20.3,21.6]_max_rel"])
    if worst > 1e-9:
        raise SystemExit(f"BLOCKED: reduction does not close against the paper-facing npz (max rel {worst:.3e})")
    print(f"closure OK (max rel {worst:.2e}); omega_20p0 in fig_hbi_cddf.data.npz integrates [20.0, 21.6]")

    # ---- real-data anatomy on every domain ----
    env_by_slice = {"allz": envelope_from_audit(audit, "allz")}
    for name, lo, hi in Z_SLICES:
        env_by_slice[name] = envelope_from_audit(audit, name)
    real = {}
    for name, lo, hi, cov in domains(ze):
        zw = z_weight(ze, dXk, lo, hi)
        if zw.sum() <= 0:
            continue
        env = env_by_slice.get(name, env_by_slice["allz"])
        real[name] = {"z": [lo, hi], "coverage": cov, "dX": float(zw.sum()),
                      "envelope_source_slice": name if name in env_by_slice else "allz",
                      **anatomy(f, ne, zw, envelope=env, mirror_f=fm, label=name)}
    real["allz"]["omega_[20.0,21.6]_q"] = om_20p0.tolist()

    # ---- migration (adopted response surfaces at the posterior-median f) ----
    fmed = np.median(f, axis=0)
    mig = {}
    for name, lo, hi in [("allz", *LOWZ_SUPPORT)] + [(n, l, h) for n, l, h in Z_SLICES]:
        contrib, nhat_edges = migration_block(P, fmed, ne, lo, hi)
        mig[name] = migration_summary(contrib, nhat_edges, ne)
    kernel = kernel_cell_table(P, ne)

    # ---- observed support of the top bins (integer detections on the pack grid) ----
    counts = np.asarray(P["counts"], int)
    nh = np.asarray(P["nhat_edges"], float)
    support = {"nhat_edges": nh.tolist(),
               "detections_nhat_in_[21.3,21.6)_allz": int(counts[(nh[:-1] >= 21.3 - 1e-9) & (nh[1:] <= 21.6 + 1e-9)].sum()),
               "detections_nhat_ge_21.6_allz": int(counts[nh[:-1] >= 21.6 - 1e-9].sum()),
               "detections_nhat_ge_20.3_allz": int(counts[nh[:-1] >= 20.3 - 1e-9].sum()),
               "detections_nhat_in_[21.3,21.6)_per_z_cell": counts[(nh[:-1] >= 21.3 - 1e-9) & (nh[1:] <= 21.6 + 1e-9)].sum(axis=(0, 2)).tolist(),
               "zf_edges": ze.tolist()}

    # ---- mocks: Omega recovery envelope + L1 twin; sensitivity arms ----
    run_paths = {os.path.basename(p).replace("_fdraws.npz", ""): p for p in a.mock_runs}
    per_run, fam_summary = mock_recovery(run_paths)
    arms = dict(kv.split("=", 1) for kv in a.sens_arms)
    sens = sensitivity_arms(a.sens_primary, arms)

    # ---- the high-z arm (diagnostic-only, ruling #45): read, never recomputed ----
    bh = json.load(open(a.bh_artifact))
    bh_rec = {"file": a.bh_artifact, "sha256": _sha(a.bh_artifact),
              "omega_integrated_MAP_q16_q84_q025_q975": [bh["measurement"]["20.3"]["omega"]["integrated"][k] for k in ("MAP", "q16", "q84", "q025", "q975")],
              "named_line_omega_BH": bh["metadata"]["ratification"]["named_lines"].get("omega_BH"),
              "status": "DIAGNOSTIC ONLY (PI 2026-08-26 #45); see RESPONSE_R-037.md for why it cannot leave that status here"}

    lines = {
        "L1_RESPONSE_BIAS": {"ledger": ledger["lines"]["L1_RESPONSE_BIAS"],
                             "omega_twin": "per-family all-z Omega posterior recovery bias on the same 16 corrected-g runs (mock_recovery.family_summary.allz.omega_median_bias_pct); side by side with dndx_ge20p3_median_bias_pct from the same draws"},
        "L2_MOCK2REAL_C_TRANSPORT": {"ledger": ledger["lines"]["L2_MOCK2REAL_C_TRANSPORT"],
                                     "omega_twin": "NOT N-resolved in the frozen record above 20.3 (rinj_decomposition.json / ws4_survey_weighted_refined.json carry strata >=20.3, 20.0-20.3, 19.5-20.0 only). As a global scalar on the >=20.3 class the fractional band on Omega[20.3,21.6] is identical: x1.00 -> x1.10. The H2-M record states the completeness deficit is concentrated at low N x low z and absent at high N (2026-08-17 ckpt 10.5 note section 4), so for the mass-weighted integral x1.10 is an UPPER bracket; a smaller value cannot be claimed without an N-resolved transport measurement (not in scope).",
                                     "band_on_omega_ge20p3_window": [1.00, 1.10], "applied": False},
        "L15_CONFIG_AMBIGUITY": {"ledger_mirror_vs_pooled_pct": ledger["lines"]["L15_CONFIG_AMBIGUITY"]["mirror_vs_pooled_pct"],
                                 "omega_twin": "treatments.L15_mirror_configuration per domain (same carrier: deep s20260826 chain 0)"},
        "L16_FP_TRANSFER_EXCURSION": {"omega_twin": "= L15 (one object, PI #32); no separate number"},
        "L13_FP_ANCHOR_S_EMPTY": {"ledger": ledger["lines"]["L13_FP_ANCHOR_S_EMPTY"], "omega_twin": "sensitivity_arms (semp1p5 / semp3, 2LPT-0 s20260818 corrected-g, same seed as the primary)"},
        "L3_NUISANCE_TREATMENT": {"ledger": ledger["lines"]["L3_NUISANCE_TREATMENT"], "omega_twin": "NOT propagated: the anchored/amplitude family runs predate the corrected-g packs and no draw-level carrier is in the frozen record; a one-sided +1.5..+1.9 % counting-statistic line whose Omega analogue would require a rerun (not in scope)"},
        "L4_XFAMILY_TRANSPORT_SCATTER": {"ledger": ledger["lines"]["L4_XFAMILY_TRANSPORT_SCATTER"], "omega_twin": "the family SPREAD of response_held_contribution_median_bias_pct / top2_contribution_median_bias_pct in mock_recovery.family_summary is the Omega-relevant view of the high-tail cross-family scatter (same runs, not the ledger's masked-named predictive number)"},
        "L5_CEILING_VALIDITY": {"ledger": ledger["lines"]["L5_CEILING_VALIDITY"], "omega_twin": "kernel_cell_table in_grid_phi for the true bins >= 21.1 (off-grid mass) and migration.per_observed_class_composition for nhat >= 21.6"},
        "no_quadrature": "statistical intervals and named lines are reported separately; nothing is summed"}

    env_json = {str(k): v for k, v in env_by_slice["allz"].items()}
    out = {
        "role": "R-037 Omega_DLA systematic anatomy — DIAGNOSTIC reduction of frozen Paper-1 products (paper1_requests_2026-08-28); sits BESIDE the frozen products, supersedes nothing",
        "status": "diagnostic",
        "written_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"module": "CDDF_analysis/hbi_mcmc/omega_anatomy.py", "commit": _git_commit(), "argv": sys.argv,
                      "python": sys.version.split()[0], "numpy": np.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV")},
        "inputs": {k: {"path": p, "sha256": _sha(p)} for k, p in
                   [("draws", a.draws), ("pack", a.pack), ("summary", a.summary), ("paper_dndx_npz", a.paper_dndx_npz),
                    ("paper_cddf_npz", a.paper_cddf_npz), ("mirror", a.mirror), ("audit", a.audit), ("ledger", a.ledger),
                    ("bh_artifact", a.bh_artifact)]},
        "mirror_chain": {"chain": a.mirror_chain, "chains": a.mirror_chains, "n_draws": int(nchain)},
        "conventions": {"omega_window": list(OMEGA_NHI), "h": H_REPORTING, "Omega_m": OMEGA_M_REPORTING, "mass": "proton (m_p = 1.67262192369e-27 kg); no helium/neutral correction",
                        "omega_prefactor_cm2": OMEGA_PREFACTOR_CM2, "quantile_levels": QUANTILES,
                        "interval_type": "posterior interval over the frozen pooled draws (6000); mock and mirror sets are posterior intervals over their own draws",
                        "intra_bin_shape_adopted": "f per dex flat in log N inside each 0.2-dex latent bin; the partial top bin [21.5,21.6] takes the linear-N share of [21.5,21.7)",
                        "response_calibrated_top": RESPONSE_CALIBRATED_TOP, "no_quadrature": True,
                        "selection_calibration_contract": {"pack_contract_id": str(P["contract_id"]), "tp_convention_id": str(P["tp_convention_id"]),
                                                           "adopted_resp_version": str(P["adopted_resp_version"]),
                                                           "selection_contract_sidecar": a.pack[:-4] + ".selection_contract.json",
                                                           "collar_kms": 3300, "P_DLA_min": 0.99, "SNR_REDSIDE_min": 2.0, "BAL_policy": "BI_CIV>0 dropped",
                                                           "z_support": list(LOWZ_SUPPORT), "latent_edges": ne.tolist(), "nhat_edges": nh.tolist()}},
        "closure": closure,
        "naming_defect_item17": {"fig_hbi_cddf.data.npz": {"omega_20p0": "Omega over [20.0, 21.6] (verified by recomputation, rel 1e-16 level)",
                                                            "omega_20p3": "Omega over [20.3, 21.6] (the adopted interval)",
                                                            "omega_limits": "[20.3, 21.6] — describes omega_20p3 only; DEFECT: the 20p0 key has no limits carrier"},
                                 "resolution": "emit omega_limits_20p0=[20.0,21.6] and omega_limits_20p3=[20.3,21.6] in fig_hbi_cddf.py; Omega over [20.0,21.6] is NOT the ratified reporting interval (D1 2026-07-29; #47) and stays a companion, never a headline"},
        "real": real,
        "envelope_allz_per_0p2dex": env_json,
        "migration": mig, "kernel_cells": kernel, "observed_support_top_bins": support,
        "mock_recovery": {"per_run": per_run, "family_summary": fam_summary},
        "sensitivity_arms": sens,
        "high_z_arm": bh_rec,
        "lines": lines,
    }
    jp = os.path.join(a.out_dir, "R037_omega_anatomy.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1)
    # arrays (axis order stated in the file)
    npz = {"axis_note": np.array(["contribution arrays: (domain order as 'domains', bin, quantile 2.5/16/50/84/97.5)"]),
           "domains": np.array(list(real.keys())),
           "contribution_bins": np.array(real["allz"]["contribution_bins"], float),
           "quantile_levels": np.array(QUANTILES)}
    npz["omega_adopted_q"] = np.array([real[d]["omega_adopted"]["q_p2p5_16_50_84_97p5"] for d in real])
    npz["contribution_fraction_q"] = np.array([real[d]["contribution_fraction_q"] for d in real])
    npz["variance_share_by_bin"] = np.array([real[d]["variance_share_by_bin"] for d in real])
    npz["upper_limit_scan_X"] = np.array(UPPER_LIMIT_SCAN)
    npz["upper_limit_scan_q"] = np.array([[real[d]["upper_limit_scan"][str(X)]["q_p2p5_16_50_84_97p5"] for X in UPPER_LIMIT_SCAN] for d in real])
    np.savez(os.path.join(a.out_dir, "R037_omega_anatomy.npz"), **npz)
    shas = {os.path.basename(p): _sha(p) for p in (jp, os.path.join(a.out_dir, "R037_omega_anatomy.npz"))}
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for k, v in shas.items():
            fh.write(f"{v}  {k}\n")
    print(json.dumps(shas, indent=1))
    # a compact console table (all-z + B1..B5)
    for d in real:
        r = real[d]
        q = r["omega_adopted"]["q_p2p5_16_50_84_97p5"]
        print(f"{d:9s} Omega {q[2]:.4e} [{q[1]:.4e},{q[3]:.4e}] hw68 {r['omega_adopted']['halfwidth68_pct']:.2f}% | top2 frac {r['fraction_top2_bins_q'][2]:.3f} "
              f"held frac {r['fraction_response_held_q'][2]:.3f} var-share top2 {r['variance_share_top2']:.3f} | top 21.3 {r['upper_limit_scan']['21.3']['median_shift_pct']:+.1f}% "
              f"21.7 {r['upper_limit_scan']['21.7']['median_shift_pct']:+.1f}% 22.4 {r['upper_limit_scan']['22.4']['median_shift_pct']:+.1f}% | L15 {r['treatments']['L15_mirror_configuration']['median_shift_pct']:+.2f}% "
              f"(dN/dX {r['L15_mirror_dndx_ge20p3_over_pooled_minus1_pct']:+.2f}%) | held-env [{r['treatments']['response_held_bins_truth_side_bias_max']['median_shift_pct']:+.1f},{r['treatments']['response_held_bins_truth_side_bias_min']['median_shift_pct']:+.1f}]% | plaw {r['treatments']['intrabin_local_powerlaw']['median_shift_pct']:+.2f}%")


if __name__ == "__main__":
    main()
