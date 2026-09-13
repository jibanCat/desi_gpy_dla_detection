#!/usr/bin/env python
"""analyze_fold.py — OPERATOR-LEVEL forensics of the absorber side of the
frozen forward fold (VALIDATION-ONLY; no sampler is run; nothing under
``CDDF_analysis/`` is modified).

The frozen fold (``cc_posterior_validation.model_cc`` :195-208, tensors from
``build_cc_tensors`` :42-70):

    mu_TP[c,k,s] = dX[k,s] * sum_b Mg[s,k,c,b] * C[s,b] * g[b,k] * f[b,k] * dN_b

This script folds the mock's OWN truth (``forward_selftest.truth_f`` :64-80)
through that operator with psi_c = 0 (the calibrated completeness) and compares
it, cell by cell, with the matched detections rebuilt by
``build_matched_ops.py``.  Every comparison is a RATIO of two counts on the
same grid; the Poisson z convention is ``forward_selftest.poisson_z`` :185.

Env: gpdla-hbi (jax is needed only for ``build_consts`` / ``build_cc_tensors``).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
from binning import (coarse_block_sum, safe_ratio, migration_moments,   # noqa
                     leakage_fractions, threshold_weights)

FAMILIES = ("2lpt0", "london0", "saclay0")
ADOPTED_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "adopted_packs_v2p2_20260821")
SCANPACK_DIR = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
                "packs")
ORACLE_DIR = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
              "diag_oracle")
#: the 0.2-dex reporting bins that alternate in the ORACLE runs
ALT_BINS = ((19.7, 19.9, "low"), (19.9, 20.1, "high"), (20.1, 20.3, "low"),
            (21.1, 21.3, "high"), (21.5, 21.7, "low"))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                        # pragma: no cover
        return "unknown"


def poisson_z(mu, obs):
    """``forward_selftest.poisson_z`` :185 — (obs - mu)/sqrt(max(mu, 1e-12))."""
    return (np.asarray(obs, float) - np.asarray(mu, float)) / np.sqrt(
        np.maximum(np.asarray(mu, float), 1e-12))


def fold_truth(pack_path):
    """mu_TP decomposed as (c,k,s,b) from the pack's OWN truth, psi_c = 0."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f

    pk = load_pack(pack_path)
    consts, Mg = build_cc_tensors(pk)              # Mg (S, Kf, C, B)
    Mg = np.asarray(Mg, float)
    from scipy.special import expit
    eta = np.asarray(consts.eta_hat, float)        # (S, M)
    b2c = np.asarray(consts.b_to_cell, int)
    Cc = expit(eta)[:, b2c]                        # (S, B)  psi_c = 0
    g = np.asarray(consts.g_bk, float)             # (B, Kf)
    dN = np.asarray(consts.dN_b, float)            # (B,)
    dX = np.asarray(consts.dX, float)              # (Kf, S)
    f = np.asarray(truth_f(pk), float)             # (B, Kf)
    w = g * f * dN[:, None]                        # (B, Kf)
    # tp[c,k,s,b]
    tp = np.einsum("skcb,sb,bk,ks->cksb", Mg, Cc, w, dX, optimize=True)
    # the UNGATHERED count-conserving surfaces, so the z-cell gather can be
    # varied (build_cc_tensors :60-69 gathers the response z cell through the
    # COARSE block: K_to_zresp[kz_to_K[k]])
    from CDDF_analysis.hbi_mcmc.count_conserving_fold import (surface_masses,
                                                              phi_from_surfaces)
    ne = np.asarray(pk.nhat_edges, float)
    masses, phi = surface_masses(pk, pk.adopted_resp_mu_coef,
                                 pk.adopted_resp_sig_coef,
                                 pk.adopted_resp_skew_coef,
                                 np.asarray(pk.adopted_resp_fit_range, float),
                                 ne)
    phi_ref = np.asarray(pk.adopted_phi_ref, float)
    masses = masses / np.maximum(phi, 1e-12)[:, :, None, :] \
        * phi_ref[:, :, None, :]
    zfc = 0.5 * (np.asarray(pk.zf_edges, float)[:-1]
                 + np.asarray(pk.zf_edges, float)[1:])
    zr_fine = (np.digitize(zfc, np.asarray(pk.resp_z_edges, float))
               - 1).astype(int)
    s2sr = np.asarray(consts.s_to_sresp, int)
    Mg_finez = masses[s2sr[:, None], zr_fine[None, :], :, :]     # (S,Kf,C,B)
    return dict(pk=pk, consts=consts, Mg=Mg, Cc=Cc, g=g, dN=dN, dX=dX, f=f,
                tp=tp, Mg_finez=Mg_finez, zr_fine=zr_fine,
                zr_coarse=np.asarray(consts.K_to_zresp,
                                     int)[np.asarray(consts.kz_to_K, int)])


def ratio_row(mu, obs):
    mu = float(np.sum(mu))
    obs = float(np.sum(obs))
    return dict(mu=mu, obs=obs, ratio=(mu / obs if obs > 0 else float("nan")),
                z=float(poisson_z(mu, obs)))


def analyse(family, out_dir, fig_dir):
    ap = os.path.join(ADOPTED_DIR,
                      f"modelA_pack_{family}_bw0p2_pad19p0_molly172_v2.npz")
    sp = os.path.join(SCANPACK_DIR, f"scanpack_{family}_b300.npz")
    ops_path = os.path.join(_HERE, f"empirical_ops_{family}.npz")
    ops = np.load(ops_path, allow_pickle=True)

    F = fold_truth(ap)
    pk, tp = F["pk"], F["tp"]
    ntrue = np.asarray(pk.ntrue_edges, float)
    nhat = np.asarray(pk.nhat_edges, float)
    zf = np.asarray(pk.zf_edges, float)
    kz = np.asarray(pk.kz_to_K, int)
    C, Kf, S, B = tp.shape
    KK = int(kz.max()) + 1
    Nc_b = 0.5 * (ntrue[:-1] + ntrue[1:])
    Cc_c = 0.5 * (nhat[:-1] + nhat[1:])

    N_match = np.asarray(ops["N_match_cksb"], float)          # (C,Kf,S,B)
    N_match_tz = np.asarray(ops["N_match_true_z_cksb"], float)
    P6b = np.asarray(ops["P6b_cks"], float)
    hostless = np.asarray(ops["hostless_cks"], float)
    counts_obs = np.asarray(ops["counts_obs_cks"], float)
    tc_bks = np.asarray(ops["truth_counts_bks"], float)
    tc_bks_3300 = np.asarray(ops["truth_counts_bks_collar3300"], float)
    lo_host = np.asarray(ops["N_match_host_19p0_19p5_cks"], float)
    hi_host = np.asarray(ops["N_match_host_ge19p5_cks"], float)

    mu_cks = tp.sum(axis=3)
    obs_cks = N_match.sum(axis=3)
    dxpos = np.asarray(F["dX"], float) > 0                     # (Kf,S)
    m3 = np.broadcast_to(dxpos[None, :, :], mu_cks.shape)

    R = {"family": family}
    R["totals"] = dict(
        fold_mu_TP=float(mu_cks[m3].sum()),
        matched_in_basis=float(obs_cks[m3].sum()),
        ratio=float(mu_cks[m3].sum() / obs_cks[m3].sum()),
        P6b=float(P6b[m3].sum()), hostless=float(hostless[m3].sum()),
        realised_all_detections=float(counts_obs[m3].sum()),
        host_19p0_19p5=float(lo_host[m3].sum()),
        host_ge_19p5=float(hi_host[m3].sum()))

    # ---- 1. ratio by observed N-hat bin, per coarse block K --------------
    mu_cK = coarse_block_sum(np.where(m3, mu_cks, 0.0), kz, axis=1).sum(axis=2)
    ob_cK = coarse_block_sum(np.where(m3, obs_cks, 0.0), kz, axis=1).sum(axis=2)
    R["by_nhat_by_K"] = [
        dict(lo=float(nhat[c]), hi=float(nhat[c + 1]),
             mu=[float(mu_cK[c, K]) for K in range(KK)],
             obs=[float(ob_cK[c, K]) for K in range(KK)],
             ratio=[float(mu_cK[c, K] / ob_cK[c, K]) if ob_cK[c, K] > 0
                    else None for K in range(KK)],
             z=[float(poisson_z(mu_cK[c, K], ob_cK[c, K])) for K in range(KK)])
        for c in range(C)]
    R["by_K"] = [ratio_row(mu_cK[:, K], ob_cK[:, K]) for K in range(KK)]
    R["by_nhat"] = [ratio_row(mu_cK[c], ob_cK[c]) for c in range(C)]

    # ---- by S/N stratum, and by (c, s) ----------------------------------
    mu_s = np.where(m3, mu_cks, 0.0).sum(axis=(0, 1))
    ob_s = np.where(m3, obs_cks, 0.0).sum(axis=(0, 1))
    R["by_snr"] = [dict(s=s, snr=[float(pk.snr_edges[s]),
                                  float(pk.snr_edges[s + 1])],
                        **ratio_row(mu_s[s], ob_s[s])) for s in range(S)]
    mu_cs = np.where(m3, mu_cks, 0.0).sum(axis=1)
    ob_cs = np.where(m3, obs_cks, 0.0).sum(axis=1)
    R["by_nhat_by_snr"] = [
        dict(lo=float(nhat[c]), hi=float(nhat[c + 1]),
             ratio=[float(mu_cs[c, s] / ob_cs[c, s]) if ob_cs[c, s] > 0
                    else None for s in range(S)],
             obs=[float(ob_cs[c, s]) for s in range(S)]) for c in range(C)]
    # S/N x K (is the 2-3 shortfall z-dependent?)
    mu_sK = coarse_block_sum(np.where(m3, mu_cks, 0.0), kz, axis=1).sum(axis=0)
    ob_sK = coarse_block_sum(np.where(m3, obs_cks, 0.0), kz, axis=1).sum(axis=0)
    R["by_snr_by_K"] = [dict(s=s, ratio=[float(mu_sK[K, s] / ob_sK[K, s])
                                         if ob_sK[K, s] > 0 else None
                                         for K in range(KK)],
                             obs=[float(ob_sK[K, s]) for K in range(KK)])
                        for s in range(S)]

    # ---- by TRUE-N bin b: the fold's implied detections per b ------------
    mu_b = np.where(m3[..., None], tp, 0.0).sum(axis=(0, 1, 2))
    ob_b = np.where(m3[..., None], N_match, 0.0).sum(axis=(0, 1, 2))
    R["by_true_b"] = [
        dict(b=b, lo=float(ntrue[b]), hi=float(ntrue[b + 1]),
             mu=float(mu_b[b]), obs=float(ob_b[b]),
             ratio=(float(mu_b[b] / ob_b[b]) if ob_b[b] > 0 else None),
             z=float(poisson_z(mu_b[b], ob_b[b]))) for b in range(B)]
    mu_bK = coarse_block_sum(np.where(m3[..., None], tp, 0.0).sum(axis=(0, 2)),
                             kz, axis=0)                        # (KK,B)
    ob_bK = coarse_block_sum(np.where(m3[..., None], N_match, 0.0)
                             .sum(axis=(0, 2)), kz, axis=0)
    R["by_true_b_by_K"] = [
        dict(b=b, lo=float(ntrue[b]), hi=float(ntrue[b + 1]),
             ratio=[float(mu_bK[K, b] / ob_bK[K, b]) if ob_bK[K, b] > 0
                    else None for K in range(KK)],
             obs=[float(ob_bK[K, b]) for K in range(KK)]) for b in range(B)]

    # ---- the class with no term: P6b, hostless, above-basis --------------
    R["no_term_classes"] = dict(
        P6b_by_nhat=[float(np.where(m3, P6b, 0.0)[c].sum()) for c in range(C)],
        P6b_by_K=[float(coarse_block_sum(np.where(m3, P6b, 0.0), kz, axis=1)
                        [:, K, :].sum()) for K in range(KK)],
        P6b_by_snr=[float(np.where(m3, P6b, 0.0)[:, :, s].sum())
                    for s in range(S)],
        hostless_by_nhat=[float(np.where(m3, hostless, 0.0)[c].sum())
                          for c in range(C)],
        P6b_fraction_of_all=float(P6b[m3].sum() / counts_obs[m3].sum()),
        P6b_fraction_above_20p3=float(
            P6b[8:][m3[8:]].sum() / max(counts_obs[8:][m3[8:]].sum(), 1)),
        note=("P6b = matched host in [17.2, 19.0): BELOW the latent basis "
              "floor; the fold has NO term for it (matching_contract "
              "P6_RESIDUAL sub-slot b). hostless = the census P4+P6c, which "
              "the ORACLE FP term is pinned to."))

    # ---- 2. completeness: truth vs calibration --------------------------
    C_true_bKs = np.asarray(ops["C_true_bKs"], float)          # (B,KK,S)
    C_true_bs = np.asarray(ops["C_true_bs"], float)
    tc_bKs = coarse_block_sum(tc_bks, kz, axis=1)              # (B,KK,S)
    Cmod_sb = np.asarray(F["Cc"], float)                       # (S,B)
    g_bk = np.asarray(F["g"], float)                           # (B,Kf)
    # the model's implied z-trend: C[s,b] * <g[b,k]>_K, occupancy-weighted
    occ = tc_bks.sum(axis=2)                                   # (B,Kf)
    gK = np.zeros((B, KK))
    for K in range(KK):
        sel = kz == K
        wgt = occ[:, sel]
        gK[:, K] = np.where(wgt.sum(axis=1) > 0,
                            (g_bk[:, sel] * wgt).sum(axis=1)
                            / np.maximum(wgt.sum(axis=1), 1e-30),
                            g_bk[:, sel].mean(axis=1))
    Cmod_bKs = Cmod_sb.T[:, None, :] * gK[:, :, None]          # (B,KK,S)
    R["completeness"] = dict(
        b_edges=[[float(ntrue[b]), float(ntrue[b + 1])] for b in range(B)],
        C_model_sb=[[float(x) for x in Cmod_sb[s]] for s in range(S)],
        C_true_bs=[[float(x) for x in C_true_bs[b]] for b in range(B)],
        C_true_bKs=[[[float(C_true_bKs[b, K, s]) for s in range(S)]
                     for K in range(KK)] for b in range(B)],
        C_model_bKs=[[[float(Cmod_bKs[b, K, s]) for s in range(S)]
                      for K in range(KK)] for b in range(B)],
        truth_counts_bKs=[[[float(tc_bKs[b, K, s]) for s in range(S)]
                           for K in range(KK)] for b in range(B)],
        gK=[[float(gK[b, K]) for K in range(KK)] for b in range(B)],
        resid_model_over_true_bKs=[
            [[float(Cmod_bKs[b, K, s] / C_true_bKs[b, K, s])
              if C_true_bKs[b, K, s] > 0 else None for s in range(S)]
             for K in range(KK)] for b in range(B)])

    # ---- the fold's STRATUM ALLOCATION of the truth ----------------------
    # the fold uses f[b,k] (all-stratum) x dX[k,s]: truth is re-allocated
    # across strata IN PROPORTION TO PATHLENGTH.  Compare with the measured
    # truth_counts_bks.
    dX = np.asarray(F["dX"], float)
    dX_tot = dX.sum(axis=1)
    alloc = (tc_bks.sum(axis=2)[:, :, None]
             * np.where(dX_tot[None, :, None] > 0,
                        dX[None, :, :] / np.maximum(dX_tot[None, :, None],
                                                    1e-30), 0.0))
    alloc_K = coarse_block_sum(alloc, kz, axis=1)
    R["stratum_allocation"] = dict(
        note=("fold-implied truth per (b,k,s) = truth_counts[b,k] * dX[k,s] / "
              "dX_tot[k] (f is not stratified) vs the MEASURED "
              "truth_counts_bks[b,k,s]"),
        by_snr_ratio=[float(alloc[:, :, s].sum() / tc_bks[:, :, s].sum())
                      if tc_bks[:, :, s].sum() > 0 else None
                      for s in range(S)],
        by_snr_truth=[float(tc_bks[:, :, s].sum()) for s in range(S)],
        by_snr_by_K_ratio=[[float(alloc_K[:, K, s].sum()
                                  / tc_bKs[:, K, s].sum())
                            if tc_bKs[:, K, s].sum() > 0 else None
                            for K in range(KK)] for s in range(S)],
        weighted_completeness_error_pct=None)
    # what the mis-allocation costs: sum_s C[s,b]*alloc vs sum_s C[s,b]*truth
    num_alloc = np.einsum("sb,bks->bk", Cmod_sb, alloc)
    num_true = np.einsum("sb,bks->bk", Cmod_sb, tc_bks)
    err_b = safe_ratio(num_alloc.sum(axis=1), num_true.sum(axis=1), fill=np.nan)
    R["stratum_allocation"]["weighted_completeness_error_pct"] = [
        (float(100 * (err_b[b] - 1)) if np.isfinite(err_b[b]) else None)
        for b in range(B)]
    errK = safe_ratio(coarse_block_sum(num_alloc, kz, axis=1),
                      coarse_block_sum(num_true, kz, axis=1), fill=np.nan)
    R["stratum_allocation"]["weighted_completeness_error_pct_by_K"] = [
        [float(100 * (errK[b, K] - 1)) if np.isfinite(errK[b, K]) else None
         for K in range(KK)] for b in range(B)]

    # ---- 3. migration: adopted skew-normal Mg vs M_true ------------------
    Mg = F["Mg"]                                                # (S,Kf,C,B)
    MgK = np.zeros((S, KK, C, B))
    for K in range(KK):
        sel = np.where(kz == K)[0]
        MgK[:, K] = Mg[:, sel[0], :, :]      # Mg is constant within K by gather
        for kk in sel[1:]:
            if not np.allclose(Mg[:, kk], Mg[:, sel[0]]):
                raise AssertionError("Mg varies inside a coarse block")
    Mg_norm = MgK / np.maximum(MgK.sum(axis=2, keepdims=True), 1e-300)
    M_true = np.asarray(ops["M_true_sKcb"], float)
    M_counts = np.asarray(ops["M_counts_sKcb"], float)
    mm_mean, mm_sd, mm_skew = migration_moments(np.moveaxis(Mg_norm, 2, -1),
                                                Cc_c)
    mt_mean, mt_sd, mt_skew = migration_moments(np.moveaxis(M_true, 2, -1),
                                                Cc_c)
    mig = []
    for b in range(B):
        row = dict(b=b, lo=float(ntrue[b]), hi=float(ntrue[b + 1]),
                   in_grid_mass_model=[[float(MgK[s, K, :, b].sum())
                                        for K in range(KK)] for s in range(S)],
                   n_matched=[[float(M_counts[s, K, :, b].sum())
                               for K in range(KK)] for s in range(S)],
                   mean_model=[[float(mm_mean[s, K, b]) for K in range(KK)]
                               for s in range(S)],
                   mean_true=[[float(mt_mean[s, K, b]) for K in range(KK)]
                              for s in range(S)],
                   sd_model=[[float(mm_sd[s, K, b]) for K in range(KK)]
                             for s in range(S)],
                   sd_true=[[float(mt_sd[s, K, b]) for K in range(KK)]
                            for s in range(S)],
                   skew_model=[[float(mm_skew[s, K, b]) for K in range(KK)]
                               for s in range(S)],
                   skew_true=[[float(mt_skew[s, K, b]) for K in range(KK)]
                              for s in range(S)])
        lo, hi = float(ntrue[b]), float(ntrue[b + 1])
        lk_m = leakage_fractions(np.moveaxis(Mg_norm, 2, -1)[:, :, b], nhat,
                                 lo, hi)
        lk_t = leakage_fractions(np.moveaxis(M_true, 2, -1)[:, :, b], nhat,
                                 lo, hi)
        row["leak_model_down_in_up"] = [[[float(x) for x in lk_m[s, K]]
                                         for K in range(KK)] for s in range(S)]
        row["leak_true_down_in_up"] = [[[float(x) for x in lk_t[s, K]]
                                        for K in range(KK)] for s in range(S)]
        mig.append(row)
    R["migration"] = mig

    # ---- 3b. the model's DETECTION-AND-COUNTING operator vs the truth ----
    # A_mod[b,K,s] = C_molly[s,b] * <g[b,k]>_K * phi[s,K,b]  (the fold's own
    # probability that a truth system in (b, K, s) becomes an ON-GRID
    # detection); A_true = C_true_bKs, which is measured the same way
    # (matched ON-GRID detections / truth).  The count-conserving split
    # (count_conserving_fold.py docstring) names phi the counting part of C.
    phi_sKb = MgK.sum(axis=2)                                    # (S,KK,B)
    A_mod_bKs = (Cmod_sb.T[:, None, :] * gK[:, :, None]
                 * np.transpose(phi_sKb, (2, 1, 0)))             # (B,KK,S)
    R["operator_efficiency"] = dict(
        note=("A_mod[b,K,s] = C_molly[s,b]*<g>_K*phi[s,K,b] vs the measured "
              "C_true_bKs (both = P(on-grid detection | truth in b,K,s))"),
        A_mod_bKs=[[[float(A_mod_bKs[b, K, s]) for s in range(S)]
                    for K in range(KK)] for b in range(B)],
        phi_sKb=[[[float(phi_sKb[s, K, b]) for b in range(B)]
                  for K in range(KK)] for s in range(S)],
        ratio_bKs=[[[float(A_mod_bKs[b, K, s] / C_true_bKs[b, K, s])
                     if C_true_bKs[b, K, s] > 0 else None for s in range(S)]
                    for K in range(KK)] for b in range(B)],
        truth_weighted_ratio_bs=[
            [float((A_mod_bKs[b, :, s] * tc_bKs[b, :, s]).sum()
                   / max((C_true_bKs[b, :, s] * tc_bKs[b, :, s]).sum(), 1e-12))
             if tc_bKs[b, :, s].sum() > 0 else None for s in range(S)]
            for b in range(B)],
        truth_weighted_ratio_bK=[
            [float((A_mod_bKs[b, K, :] * tc_bKs[b, K, :]).sum()
                   / max((C_true_bKs[b, K, :] * tc_bKs[b, K, :]).sum(), 1e-12))
             if tc_bKs[b, K, :].sum() > 0 else None for K in range(KK)]
            for b in range(B)])

    # ---- 3c. SEPARATE detection probability from the counting fraction ---
    # C_true carries BOTH (detected AND landing on the observed grid).  With
    # the unrestricted matched histogram the two factor exactly:
    #   C_det_true[b,K,s] = all matched detections / truth   (DETECTION)
    #   phi_true[b,K,s]   = on-grid / all matched            (COUNTING)
    # and the model's counterparts are C_molly*<g>_K and phi[s,K,b].
    if "N_det_all_bks_true_z" in ops.files:
        nd_all = coarse_block_sum(np.asarray(ops["N_det_all_bks_true_z"],
                                             float), kz, axis=1)   # (B,KK,S)
        nd_grid = coarse_block_sum(np.asarray(ops["N_det_bks_true_z"], float),
                                   kz, axis=1)
        C_det_true = safe_ratio(nd_all, tc_bKs)
        phi_true = safe_ratio(nd_grid, nd_all)
        C_det_mod = Cmod_sb.T[:, None, :] * gK[:, :, None]          # (B,KK,S)
        phi_mod = np.transpose(phi_sKb, (2, 1, 0))                  # (B,KK,S)
        R["completeness_vs_counting"] = dict(
            note=("C_det = P(detected | truth in b,K,s) regardless of the "
                  "observed N-hat bin; phi = P(the detection lands on the "
                  "[19.5,22.4) observed grid | detected).  C_true = C_det*phi."),
            C_det_true=[[[float(C_det_true[b, K, s]) for s in range(S)]
                         for K in range(KK)] for b in range(B)],
            C_det_model=[[[float(C_det_mod[b, K, s]) for s in range(S)]
                          for K in range(KK)] for b in range(B)],
            phi_true=[[[float(phi_true[b, K, s]) for s in range(S)]
                       for K in range(KK)] for b in range(B)],
            phi_model=[[[float(phi_mod[b, K, s]) for s in range(S)]
                        for K in range(KK)] for b in range(B)],
            C_det_ratio_bs=[[float((C_det_mod[b, :, s] * tc_bKs[b, :, s]).sum()
                                   / max((C_det_true[b, :, s]
                                          * tc_bKs[b, :, s]).sum(), 1e-12))
                             if tc_bKs[b, :, s].sum() > 0 else None
                             for s in range(S)] for b in range(B)],
            phi_ratio_bs=[[float((phi_mod[b, :, s] * nd_all[b, :, s]).sum()
                                 / max((phi_true[b, :, s]
                                        * nd_all[b, :, s]).sum(), 1e-12))
                           if nd_all[b, :, s].sum() > 0 else None
                           for s in range(S)] for b in range(B)],
            C_det_ratio_bK=[[float((C_det_mod[b, K, :] * tc_bKs[b, K, :]).sum()
                                   / max((C_det_true[b, K, :]
                                          * tc_bKs[b, K, :]).sum(), 1e-12))
                             if tc_bKs[b, K, :].sum() > 0 else None
                             for K in range(KK)] for b in range(B)],
            phi_ratio_bK=[[float((phi_mod[b, K, :] * nd_all[b, K, :]).sum()
                                 / max((phi_true[b, K, :]
                                        * nd_all[b, K, :]).sum(), 1e-12))
                           if nd_all[b, K, :].sum() > 0 else None
                           for K in range(KK)] for b in range(B)])

    # ---- 4. the fold with the EMPIRICAL operators swapped in -------------
    # EXACT component replacement.  The count-conserving convention
    # (count_conserving_fold.py) splits the kernel into a UNIT-MASS
    # redistribution Mg_norm and a counting probability phi that belongs to C.
    # ``M_true_sKcb`` is unit-mass by construction and ``C_true_bKs`` already
    # carries the on-grid factor, so (C_true, M_true) compose exactly; neither
    # may be swapped in alone without restoring the other's mass.
    Cc_mod = np.asarray(F["Cc"], float)
    w_bk = F["g"] * F["f"] * F["dN"][:, None]
    base = np.einsum("skcb,sb,bk,ks->cks", Mg, Cc_mod, w_bk, dX, optimize=True)
    Mn = Mg_norm[:, kz, :, :]                                   # (S,Kf,C,B)
    Mt = M_true[:, kz, :, :]
    fb = (M_counts.sum(axis=2) <= 0)                            # (S,KK,B)
    Mt = np.where(fb[:, kz, None, :], Mn, Mt)
    # the FROZEN operator uses the FINE-z g (not the K-average): this makes
    # A_mod_k * Mn * dX * dN reproduce the fold's own tp exactly
    A_mod_k = (Cmod_sb.T[:, None, :] * g_bk[:, :, None]
               * np.transpose(phi_sKb, (2, 1, 0))[:, kz, :])     # (B,Kf,S)
    A_true_k = np.where(C_true_bKs > 0, C_true_bKs, A_mod_bKs)[:, kz, :]
    fdN = (F["f"] * F["dN"][:, None])                           # (B,Kf)
    swaps = {}
    swaps["migration_only"] = np.einsum("skcb,bks,bk,ks->cks", Mt, A_mod_k,
                                        fdN, dX, optimize=True)
    swaps["completeness_only"] = np.einsum("skcb,bks,bk,ks->cks", Mn, A_true_k,
                                           fdN, dX, optimize=True)
    swaps["both"] = np.einsum("skcb,bks,bk,ks->cks", Mt, A_true_k, fdN, dX,
                              optimize=True)
    # truth stratification: replace f[b,k]*dX[k,s] by the MEASURED
    # truth_counts_bks (the fold re-allocates truth across strata by dX)
    swaps["truth_stratified"] = np.einsum("skcb,bks,bks->cks", Mn, A_mod_k,
                                          tc_bks, optimize=True)
    swaps["all_three"] = np.einsum("skcb,bks,bks->cks", Mt, A_true_k, tc_bks,
                                   optimize=True)
    obs_m = np.where(m3, obs_cks, 0.0)
    obs_K = coarse_block_sum(obs_m, kz, axis=1)
    R["component_swaps"] = {}
    for nm, arr in [("frozen", base)] + list(swaps.items()):
        a = np.where(m3, arr, 0.0)
        aK = coarse_block_sum(a, kz, axis=1)
        R["component_swaps"][nm] = dict(
            total_ratio=float(a.sum() / obs_m.sum()),
            by_K=[float(aK[:, K, :].sum() / obs_K[:, K, :].sum())
                  for K in range(KK)],
            by_snr=[float(a[:, :, s].sum() / max(obs_m[:, :, s].sum(), 1e-9))
                    for s in range(S)],
            by_nhat=[float(a[c].sum() / obs_m[c].sum()) if obs_m[c].sum() > 0
                     else None for c in range(C)],
            rms_log_nhat_resid=float(np.sqrt(np.mean(
                np.log(np.clip([a[c].sum() / obs_m[c].sum()
                                for c in range(C) if obs_m[c].sum() >= 50],
                               1e-6, None)) ** 2))))
    R["component_swaps"]["_migration_fallback_sKb_rows"] = int(fb.sum())

    # ---- 4b. the COUNTS BUDGET the fitted f has to close -----------------
    # Under ORACLE the FP arm is PINNED to the census ``hostless`` (run_ladder
    # :68).  Everything else in ``counts`` must come out of the TP arm, so the
    # level the sampler drives f to is
    #     f_hat / f_true ~= (counts - hostless) / mu_TP(f_true)
    # and its two factors are (i) the classes with no term (P6b, above-basis)
    # and (ii) the operator's own mis-prediction of the matched set.
    ob_all = np.where(m3, counts_obs, 0.0)
    hl = np.where(m3, hostless, 0.0)
    p6 = np.where(m3, P6b, 0.0)
    mu_m = np.where(m3, mu_cks, 0.0)
    need = ob_all - hl
    R["counts_budget"] = dict(
        note=("ORACLE pins mu_FP to the census hostless array; the TP arm must "
              "then carry counts - hostless, i.e. the matched set PLUS every "
              "class the fold has no term for"),
        counts_on_grid=float(ob_all.sum()), hostless=float(hl.sum()),
        P6b=float(p6.sum()), matched_in_basis=float(obs_cks[m3].sum()),
        above_basis_top=float(np.where(m3, np.asarray(
            ops["host_above_top_cks"], float), 0.0).sum()),
        residual_unaccounted=float(ob_all.sum() - hl.sum() - p6.sum()
                                   - obs_cks[m3].sum()),
        mu_TP_truth=float(mu_m.sum()),
        required_over_truthfold=float(need.sum() / mu_m.sum()),
        operator_over_prediction=float(mu_m.sum() / obs_cks[m3].sum()),
        no_term_inflation=float((obs_cks[m3].sum() + p6.sum())
                                / obs_cks[m3].sum()),
        by_K_required=[float(coarse_block_sum(need, kz, axis=1)[:, K, :].sum()
                             / coarse_block_sum(mu_m, kz, axis=1)[:, K, :].sum())
                       for K in range(KK)],
        by_snr_required=[float(need[:, :, s].sum() / mu_m[:, :, s].sum())
                         if mu_m[:, :, s].sum() > 0 else None
                         for s in range(S)],
        by_nhat_required=[float(need[c].sum() / mu_m[c].sum())
                          if mu_m[c].sum() > 0 else None for c in range(C)],
        by_nhat_P6b_share=[float(p6[c].sum() / ob_all[c].sum())
                           if ob_all[c].sum() > 0 else None for c in range(C)],
        by_nhat_by_K_required=[
            [float(coarse_block_sum(need, kz, axis=1)[c, K, :].sum()
                   / max(coarse_block_sum(mu_m, kz, axis=1)[c, K, :].sum(),
                         1e-12)) for K in range(KK)] for c in range(C)])
    if "hostless_cks_collar3300" in ops.files:
        hl33 = np.asarray(ops["hostless_cks_collar3300"], float)
        R["counts_budget"]["hostless_collar3300"] = float(hl33[m3].sum())
        R["counts_budget"]["oracle_FP_pin_collar_excess_counts"] = float(
            hostless.sum() - hl33.sum())

    # ---- 4c. DETERMINISTIC Poisson-MLE inversion (no sampler) ------------
    # The fold is LINEAR in f and does not mix fine-z cells, so for each k the
    # latent bias is the solution of a 16-parameter non-negative Poisson MLE,
    # obtainable by EM (multiplicative / Richardson-Lucy) updates.  This is an
    # OPERATOR-LEVEL prediction of what the sampler is driven to; it carries NO
    # population prior, so it is the UNPENALISED limit of the posterior (the
    # committed model adds the 2-D RW, which damps the oscillating mode).
    def build_A(Mk, Ak):
        a = np.einsum("skcb,bks,ks->cksb", Mk, Ak, dX, optimize=True) \
            * F["dN"][None, None, None, :]                  # (C,Kf,S,B)
        return np.where(m3[..., None], a, 0.0)

    A = build_A(Mn, A_mod_k)                                # the FROZEN fold

    def em_solve(data, A=A, n_iter=4000):
        f_hat = np.where(F["f"] > 0, F["f"], 1e-12).copy()
        col = A.sum(axis=(0, 2))                            # (Kf,B)
        d = np.where(m3, data, 0.0)
        for _ in range(n_iter):
            mu = np.einsum("cksb,bk->cks", A, f_hat, optimize=True)
            r = np.where(mu > 0, d / np.maximum(mu, 1e-300), 0.0)
            upd = np.einsum("cksb,cks->bk", A, r, optimize=True)
            f_hat = np.where(col.T > 0, f_hat * upd / np.maximum(col.T, 1e-300),
                             f_hat)
        return f_hat

    dXk = dX.sum(axis=1)
    u_by_thr = {thr: threshold_weights(ntrue, thr, nhat[0])
                for thr in (20.0, 20.3)}

    def report_f(f_hat):
        out = {}
        for thr, u in u_by_thr.items():
            num = (np.einsum("bk,b->k", f_hat, u) * dXk).sum() / dXk.sum()
            den = (np.einsum("bk,b->k", F["f"], u) * dXk).sum() / dXk.sum()
            out[f"ge{thr}_bias_pct"] = float(100 * (num / den - 1))
            perK = []
            for K in range(KK):
                w = np.where(kz == K, dXk, 0.0)
                n2 = (np.einsum("bk,b->k", f_hat, u) * w).sum() / w.sum()
                d2 = (np.einsum("bk,b->k", F["f"], u) * w).sum() / w.sum()
                perK.append(float(100 * (n2 / d2 - 1)))
            out[f"ge{thr}_bias_pct_by_K"] = perK
        bins = []
        for b in range(B):
            if 0.5 * (ntrue[b] + ntrue[b + 1]) < nhat[0]:
                continue
            n2 = (f_hat[b] * dXk).sum() / dXk.sum()
            d2 = (F["f"][b] * dXk).sum() / dXk.sum()
            bins.append(dict(bin=[float(ntrue[b]), float(ntrue[b + 1])],
                             bias_pct=float(100 * (n2 / d2 - 1))))
        out["reporting_bins"] = bins
        return out

    inv = {}
    inv["matched_only"] = report_f(em_solve(obs_cks))
    inv["oracle_counts_minus_hostless"] = report_f(em_solve(counts_obs
                                                            - hostless))
    inv["oracle_minus_hostless_minus_P6b"] = report_f(
        em_solve(counts_obs - hostless - P6b))
    # BEFORE/AFTER for each exact component replacement, on the ORACLE data
    dat = counts_obs - hostless
    inv["fix_completeness"] = report_f(em_solve(dat, build_A(Mn, A_true_k)))
    inv["fix_migration"] = report_f(em_solve(dat, build_A(Mt, A_mod_k)))
    inv["fix_both"] = report_f(em_solve(dat, build_A(Mt, A_true_k)))
    inv["fix_both_and_P6b"] = report_f(
        em_solve(dat - P6b, build_A(Mt, A_true_k)))
    # the response-z GATHER: fine-z cells -> response cells directly, instead
    # of through the coarse block (the only fine cell that moves is k=5,
    # z [2.5,2.6), whose centre 2.55 lies BELOW the response edge 2.56)
    Mfz = F["Mg_finez"]
    phi_fz = Mfz.sum(axis=2)                                    # (S,Kf,B)
    Mn_fz = Mfz / np.maximum(phi_fz[:, :, None, :], 1e-300)
    A_mod_fz = (Cmod_sb.T[:, None, :] * g_bk[:, :, None]
                * np.transpose(phi_fz, (2, 1, 0)))
    inv["fix_respz_gather"] = report_f(em_solve(dat,
                                                build_A(Mn_fz, A_mod_fz)))
    R["respz_gather"] = dict(
        fine=[int(x) for x in F["zr_fine"]],
        coarse=[int(x) for x in F["zr_coarse"]],
        n_fine_cells_moved=int(np.sum(np.asarray(F["zr_fine"])
                                      != np.asarray(F["zr_coarse"]))),
        moved_fine_cells=[int(x) for x in
                          np.where(np.asarray(F["zr_fine"])
                                   != np.asarray(F["zr_coarse"]))[0]],
        resp_z_edges=[float(x) for x in np.asarray(pk.resp_z_edges, float)],
        zc_edges=[float(x) for x in np.asarray(pk.zc_edges, float)])
    inv["_selftest_exact_operator_on_matched"] = report_f(
        em_solve(obs_cks, build_A(Mt, A_true_k)))
    inv["_note"] = ("unpenalised non-negative Poisson MLE by EM (4000 iters, "
                    "started at f_true); the committed model adds a 2-D RW "
                    "prior, so these are the UNDAMPED operator-level "
                    "predictions, not a posterior.  "
                    "_selftest_exact_operator_on_matched folds the EMPIRICAL "
                    "operator against the MATCHED counts: it must return ~0 "
                    "(residual = z-migration + stratum allocation only).")
    R["deterministic_inversion"] = inv

    # ---- 5. z-migration the fold has no term for ------------------------
    dz = np.where(m3[..., None], N_match, 0.0).sum(axis=(0, 2, 3))
    dzt = np.where(m3[..., None], N_match_tz, 0.0).sum(axis=(0, 2, 3))
    R["z_migration"] = dict(
        by_fine_z_obs=[float(x) for x in dz],
        by_fine_z_true=[float(x) for x in dzt],
        by_fine_z_ratio=[float(a / b) if b > 0 else None
                         for a, b in zip(dz, dzt)],
        by_K_obs=[float(coarse_block_sum(dz, kz, axis=0)[K])
                  for K in range(KK)],
        by_K_true=[float(coarse_block_sum(dzt, kz, axis=0)[K])
                   for K in range(KK)],
        n_cross_fine_bin=int(np.sum(
            np.asarray(ops["N_det_bks_true_z"]).sum()
            - np.minimum(np.asarray(ops["N_det_bks_true_z"]),
                         np.asarray(ops["N_det_bks_obs_z"])).sum())))

    # ---- 6. collar mismatch on the SCAN packs ---------------------------
    dN = np.diff(ntrue)
    spz = np.load(sp, allow_pickle=True)
    dX_scan = np.asarray(spz["dX"], float)
    tc_bk = tc_bks.sum(axis=2)
    tc_bk_33 = tc_bks_3300.sum(axis=2)
    f_used = safe_ratio(tc_bk, dX_scan.sum(axis=1)[None, :] * dN[:, None])
    f_corr = safe_ratio(tc_bk_33, dX_scan.sum(axis=1)[None, :] * dN[:, None])
    coll = dict(
        note=("the posterior runs consume scanpack_*_b300 whose counts/dX are "
              "at collar 3300 km/s while truth_counts/truth_counts_bks are "
              "the collar-3000 arrays copied byte-identically by "
              "build_scan_packs.py — so truth_f is built from a truth "
              "histogram on a WIDER selection than its own dX"),
        truth_total_ratio_3300_over_3000=float(tc_bk_33.sum() / tc_bk.sum()))
    for thr in (20.0, 20.3):
        u = threshold_weights(ntrue, thr, nhat[0])
        dXk = dX_scan.sum(axis=1)
        used = float((np.einsum("bk,b->k", f_used, u) * dXk).sum() / dXk.sum())
        corr = float((np.einsum("bk,b->k", f_corr, u) * dXk).sum() / dXk.sum())
        coll[f"truth_ge{thr}_as_used"] = used
        coll[f"truth_ge{thr}_collar_matched"] = corr
        coll[f"bias_understatement_pct_ge{thr}"] = float(100 * (used / corr - 1))
    coll["per_b_ratio_3300_over_3000"] = [
        float(tc_bk_33[b].sum() / tc_bk[b].sum()) if tc_bk[b].sum() > 0
        else None for b in range(B)]
    coll["per_K_ratio_3300_over_3000"] = [
        float(coarse_block_sum(tc_bk_33, kz, axis=1)[:, K].sum()
              / coarse_block_sum(tc_bk, kz, axis=1)[:, K].sum())
        for K in range(KK)]
    R["collar_mismatch"] = coll

    # ---- 7. the ORACLE posterior psi_c (reported, not used above) --------
    bych = os.path.join(ORACLE_DIR, f"RUN_ORACLE_{family}_s20260811_bychain.npz")
    if os.path.exists(bych):
        z = np.load(bych, allow_pickle=True)
        key = next((k for k in z.files if k.endswith("psi_c")), None)
        if key is not None:
            psi = np.asarray(z[key], float).reshape(-1, S, Cmod_sb.shape[1]
                                                    if False else
                                                    np.asarray(pk.molly_n_det).shape[1])
            pm = psi.mean(axis=0)
            from scipy.special import expit
            Cpsi = expit(np.asarray(F["consts"].eta_hat, float) + pm)[
                :, np.asarray(F["consts"].b_to_cell, int)]
            tp_psi = np.einsum("skcb,sb,bk,ks->cks", Mg, Cpsi, w_bk, dX,
                               optimize=True)
            R["oracle_psi_c"] = dict(
                file=bych, mean_psi_over_cells=float(pm.mean()),
                mean_psi_in_prior_sd=float(
                    (pm / np.asarray(F["consts"].sigma_hat)).mean()),
                total_ratio_with_psi=float(np.where(m3, tp_psi, 0).sum()
                                           / obs_cks[m3].sum()),
                by_snr_ratio_with_psi=[
                    float(np.where(m3, tp_psi, 0)[:, :, s].sum()
                          / max(np.where(m3, obs_cks, 0)[:, :, s].sum(), 1e-9))
                    for s in range(S)],
                by_K_ratio_with_psi=[
                    float(coarse_block_sum(np.where(m3, tp_psi, 0), kz,
                                           axis=1)[:, K, :].sum()
                          / coarse_block_sum(np.where(m3, obs_cks, 0), kz,
                                             axis=1)[:, K, :].sum())
                    for K in range(KK)])

    R["provenance"] = dict(
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        recipe="validation/absorber_diag/analyze_fold.py",
        git_commit=_git(), adopted_pack=ap, adopted_pack_sha256=_sha256(ap),
        scanpack=sp, scanpack_sha256=_sha256(sp),
        ops=ops_path, ops_sha256=_sha256(ops_path),
        fold_lines=("cc_posterior_validation.py:195-208 (model_cc), :42-70 "
                    "(build_cc_tensors); forward.py:265-458 (build_consts); "
                    "count_conserving_fold.py:surface_masses/phi_from_surfaces"),
        truth_f="forward_selftest.py:64-80", poisson_z="forward_selftest.py:185",
        psi_c="0 (calibrated completeness) for every operator comparison")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"fold_forensics_{family}.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    np.savez_compressed(
        os.path.join(out_dir, f"fold_forensics_{family}.npz"),
        tp_cksb=tp, mu_cks=mu_cks, obs_cks=obs_cks, Mg=Mg, MgK=MgK,
        Mg_norm=Mg_norm, Cmod_sb=Cmod_sb, Cmod_bKs=Cmod_bKs, gK=gK,
        alloc_bks=alloc, swaps_migration=swaps["migration_only"],
        swaps_completeness=swaps["completeness_only"], swaps_both=swaps["both"],
        swaps_truth_stratified=swaps["truth_stratified"], base_cks=base)
    return R


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--family", nargs="+", default=list(FAMILIES))
    a.add_argument("--out", default=_HERE)
    a.add_argument("--figdir", default=("/home/mfho/desi_gpy_dla_notes/figures/"
                                        "2026-09-13_absorber_diag"))
    ns = a.parse_args(argv)
    for fam in ns.family:
        R = analyse(fam, ns.out, ns.figdir)
        print(f"[{fam}] total mu/obs = {R['totals']['ratio']:.4f}  "
              f"byK = {[round(r['ratio'], 4) for r in R['by_K']]}", flush=True)


if __name__ == "__main__":
    main()
