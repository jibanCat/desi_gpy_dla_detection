#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FROZEN (C, K, M_<19.5) shared-resampling calibration covariance
(PI ruling 2026-08-08 §5.1 — the deterministic prerequisite of the gated
refold).

Extends the frozen common-healpix resampling (`p1_joint_cov.py`, whole
healpix, nside 16 nested — the same unit as the stability jackknife and
the holdout blocking) to the FULL calibration block: the SAME delete-one
healpix realization simultaneously perturbs

  * C — the deployed completeness counts (kernel-event numerators /
    truth denominators per molly cell, pushed through the deployed
    eta_hat -> sigmoid path);
  * K — the natural-pair kernel (battery-bin means AND the full
    empirical landing distributions used by the refold);
  * M_<19.5 — the below-floor net-migration counts (frozen
    `p1_migration` definition; jackknife pseudo-total scaling
    g/(g-1) for the count components);

and propagates each replicate through the P1 fold (fixed truth
allocation, deployed g_bk, fixed FP term) to the predicted G1/G2/G3
group means.  NO independent-error assumption is made anywhere: every
component's variation comes from the same shared healpix deletions, so
all C-K, C-M, K-M and G-level cross-covariances are carried.

NO OBSERVED COUNT enters this builder or its outputs: the pack's
`counts` array is never referenced.  This is calibration-side only.

Frozen estimator vector theta (ordering pinned in the artifact):
  [ C_paf(7 battery bins) | K_mean(7 battery bins) |
    M_G1, M_G2, M_G3 (pseudo-totals) | Gpred_1, Gpred_2, Gpred_3 ]

Gates (fail-loud, all run before the artifact is written):
  G1  full-sample kernel/truth cell counts == pack molly counts,
      integer-exact on every >=19.5 cell (the load identity, again);
  G2  full-sample migration group totals == committed p1_migration.json
      (4088 / 144 / 0), exact;
  G3  full-sample landing rows: in-grid mass + out-of-grid mass == 1
      per measured cell (1e-12);
  G4  deployed-fold rebuild <= 1e-8 (the committed truth-by-SNR guard,
      executed inside build_fold);
  G5  the builder's replicate contraction reproduces the einsum fold at
      the full sample to <= 1e-9 relative;
  G6  no unmapped healpix;
  G7  covariance finite; symmetric; PSD within -1e-10 * max eigenvalue;
  G8  battery-bin sigma_C / sigma_K within 2x of the committed
      p1_joint_cov.json values (block-universe consistency check).

Artifact: `p1_ckm_cov_v1.npz` (scratch, hash-recorded) + committed JSON
summary.  `load_p1_ckm_cov` is the fail-loud loader (schema, ordering,
shape, finiteness, PSD re-checked at load; arrays read-only).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from p1_refold_fold import (                                   # noqa: E402
    BATTERY_BINS, FLOOR, SPARSE_N_MIN, P1RefoldGuardError,
    build_fold, build_p1_kernel, c_marginal, healpix_of,
    load_kernel_events, load_migration, mu_sig_p1, provenance, sha256,
)

OUT_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
           "track_c/stage0/p1_ckm_cov_v1.npz")
OUT_JSON = os.path.join(_HERE, "p1_ckm_cov.json")
SCHEMA = "p1_ckm_cov/v1"
PSD_TOL = 1e-10

LABELS = ([f"C_paf[{lo},{hi})" for lo, hi in BATTERY_BINS]
          + [f"K_mean[{lo},{hi})" for lo, hi in BATTERY_BINS]
          + ["M_G1", "M_G2", "M_G3"]
          + ["Gpred_G1", "Gpred_G2", "Gpred_G3"])


def _battery_index(N):
    out = np.full(len(N), -1, np.int64)
    for i, (lo, hi) in enumerate(BATTERY_BINS):
        out[(N >= lo) & (N < hi)] = i
    return out


def main():
    t0 = time.time()
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.gate_covariance import PRIMARY_GROUP_EDGES
    from p1_refold_fold import PACK

    pk = load_pack(PACK)
    fold = build_fold(pk)                                  # G4 inside
    E, truth, sparse, art, cache = load_kernel_events()
    ne = fold["nhat_edges"]
    n_c = ne.size - 1
    mig = load_migration(ne)                               # G2 inside

    # ---- healpix assignment (G6 inside healpix_of) ---------------------
    hE = healpix_of(E["TID"])
    t_live = truth["S2N"] > 2.0
    # C denominators: ALL truth rows (the molly_n_tot population)
    hT = healpix_of(truth["TID"])
    hM = healpix_of(mig["TID"])
    universe = np.unique(np.concatenate([hE, hT, hM]))
    g = len(universe)
    h_of = {int(h): i for i, h in enumerate(universe)}
    iE = np.asarray([h_of[int(h)] for h in hE])
    iT = np.asarray([h_of[int(h)] for h in hT])
    iM = np.asarray([h_of[int(h)] for h in hM])

    # ---- per-healpix sufficient statistics -----------------------------
    snr_e = np.asarray(art["C_snr_edges"], float)
    nhi_e = np.asarray(art["C_nhi_edges"], float)
    j195 = int(np.asarray(art["K_id_j195"]).ravel()[0])
    n_sr8, n_col = len(snr_e) - 1, len(nhi_e) - 1

    def molly_cell(s2n, N, what):
        """Strict-bound cell assignment, the deployed `_cell_counts`
        convention: a value exactly ON an edge belongs to NO cell (it is
        excluded from the deployed numerators/denominators identically —
        the integer identity gate below verifies this)."""
        i = (np.searchsorted(snr_e, s2n, side="left") - 1).astype(np.int64)
        j = (np.searchsorted(nhi_e, N, side="left") - 1).astype(np.int64)
        i = np.clip(i, 0, n_sr8 - 1)
        j = np.clip(j, 0, n_col - 1)
        strict = ((s2n > snr_e[i]) & (s2n < snr_e[i + 1])
                  & (N > nhi_e[j]) & (N < nhi_e[j + 1]))
        n_edge = int(np.sum(~strict))
        if n_edge:
            print(f"[molly_cell] {what}: {n_edge} edge-exact rows excluded "
                  "from every strict cell (deployed convention)")
        return i, j, strict

    iEc, jEc, okE = molly_cell(E["S2N"], E["N"], "kernel events")
    iTc, jTc, okT = molly_cell(truth["S2N"], truth["N"], "truth rows")
    if not np.all(jEc[okE] >= j195):
        raise P1RefoldGuardError("kernel event below the 19.5 molly column")

    det_h = np.zeros((g, n_sr8, n_col))
    np.add.at(det_h, (iE[okE], iEc[okE], jEc[okE]), 1.0)
    tot_h = np.zeros((g, n_sr8, n_col))
    np.add.at(tot_h, (iT[okT], iTc[okT], jTc[okT]), 1.0)

    # G1: integer identity at the full sample (>=19.5 cells).  The event
    # set is LIVE (S2N > 2) so the det identity holds on live rows; dead
    # rows never enter the fold (their pathlength share is zero).
    live_row = np.asarray(art["C_live_row"], bool)
    det_full = det_h.sum(axis=0)
    tot_full = tot_h.sum(axis=0)
    det_pack = np.asarray(art["C_molly_n_det"], np.int64)
    tot_pack = np.asarray(art["C_molly_n_tot"], np.int64)
    if not np.array_equal(det_full[live_row][:, j195:].astype(np.int64),
                          det_pack[live_row][:, j195:]):
        raise P1RefoldGuardError("G1: kernel cell counts != pack n_det")
    if not np.array_equal(tot_full[:, j195:].astype(np.int64),
                          tot_pack[:, j195:]):
        raise P1RefoldGuardError("G1: truth cell counts != pack n_tot")

    # landing tables per healpix (merged top rep bin 13 <- 13+14)
    ci = np.digitize(E["NHAT"], ne) - 1
    in_grid = (ci >= 0) & (ci < n_c) & (E["NHAT"] < ne[-1])
    brm = np.minimum(E["BREP"], 13)                        # merged row index
    Lk_h = np.zeros((g, 14, 3, 3, n_c))
    np.add.at(Lk_h, (iE[in_grid], brm[in_grid], E["ZR"][in_grid],
                     E["SR"][in_grid], ci[in_grid]), 1.0)
    ncell_h = np.zeros((g, 14, 3, 3))
    np.add.at(ncell_h, (iE, brm, E["ZR"], E["SR"]), 1.0)
    Lm_h = np.zeros((g, 14, n_c))
    np.add.at(Lm_h, (iE[in_grid], brm[in_grid], ci[in_grid]), 1.0)
    nmarg_h = np.zeros((g, 14))
    np.add.at(nmarg_h, (iE, brm), 1.0)

    # battery-bin sufficient statistics
    bE = _battery_index(E["N"])
    bT = _battery_index(truth["N"][t_live])
    iT_live = iT[t_live]
    kcnt_h = np.zeros((g, 7))
    ksum_h = np.zeros((g, 7))
    ok = bE >= 0
    np.add.at(kcnt_h, (iE[ok], bE[ok]), 1.0)
    np.add.at(ksum_h, (iE[ok], bE[ok]), E["DX"][ok])
    tcnt_h = np.zeros((g, 7))
    okT = bT >= 0
    np.add.at(tcnt_h, (iT_live[okT], bT[okT]), 1.0)

    # migration per healpix (grid bins + groups)
    Mc_h = np.zeros((g, n_c))
    mi = mig["CI"][mig["in_grid"]]
    np.add.at(Mc_h, (iM[mig["in_grid"]], mi), 1.0)
    Mg_h = np.zeros((g, 3))
    for gi, (glo, ghi) in enumerate(PRIMARY_GROUP_EDGES):
        mg = (mig["NHAT"] >= glo) & (mig["NHAT"] < ghi)
        np.add.at(Mg_h, (iM[mg], np.full(int(mg.sum()), gi)), 1.0)

    # ---- frozen sparse structure (full-sample) -------------------------
    ncell_full = ncell_h.sum(axis=0)
    nmarg_full = nmarg_h.sum(axis=0)
    sparse_m = sparse.copy()
    sparse_m[13] = sparse[13] & sparse[14]
    use_marg = sparse_m[:14] | (ncell_full < SPARSE_N_MIN)    # (14,3,3) frozen

    # G3: landing-mass closure per measured cell at full sample
    Lk_full = Lk_h.sum(axis=0)
    Lm_full = Lm_h.sum(axis=0)
    n_out_cell = ncell_full - Lk_full.sum(axis=3)
    if np.any(n_out_cell < -1e-9):
        raise P1RefoldGuardError("G3: negative out-of-grid mass")
    mass = np.where(ncell_full > 0,
                    (Lk_full.sum(axis=3) + n_out_cell)
                    / np.maximum(ncell_full, 1), 1.0)
    if np.max(np.abs(mass - 1.0)) > 1e-12:
        raise P1RefoldGuardError("G3: landing mass closure failed")

    # ---- replicate contraction pieces ----------------------------------
    from CDDF_analysis.hbi_mcmc.forward import eta_hat_sigma_hat
    T_bks = fold["g_bk"][:, :, None] * fold["alloc"]           # (B, Kf, S)
    pad = fold["ntrue_edges"][:-1] < FLOOR - 1e-9
    T_bks[pad] = 0.0
    B, Kf, S = T_bks.shape
    r_of_b = np.full(B, -1, np.int64)
    j_of_b = np.full(B, -1, np.int64)
    for b in fold["b_used"]:
        r_of_b[b] = min(fold["b_rep"][int(b)][0], 13)
        j_of_b[b] = fold["b_to_cell"][b]
    # U[j, r, zr, s] = sum_{b: j,r} sum_{k: zr} T[b, k, s]
    U = np.zeros((n_col, 14, 3, S))
    for b in fold["b_used"]:
        for k in range(Kf):
            U[j_of_b[b], r_of_b[b], fold["k_to_zr"][k], :] += T_bks[b, k, :]

    s_to_sr = fold["s_to_sr"]

    def predict_c(det_cells, tot_cells, Lk, ncell, Lm, nmarg):
        """mu_sig c-marginal from replicate (C, K) via the W contraction."""
        eta, _ = eta_hat_sigma_hat(det_cells, tot_cells)
        C_cells = 1.0 / (1.0 + np.exp(-eta))                   # (8, 12)
        # W[r, zr, sr] = sum_{j, s} C[s, j] U[j, r, zr, s] restricted sr(s)
        W = np.zeros((14, 3, 3))
        for s in range(S):
            W[:, :, s_to_sr[s]] += np.einsum(
                "j,jrz->rz", C_cells[s, :], U[:, :, :, s])
        # landing probabilities with the FROZEN structure
        Pm = np.where(nmarg[:, None] > 0, Lm / np.maximum(nmarg[:, None], 1),
                      0.0)                                     # (14, C)
        P = np.where(use_marg[..., None], Pm[:, None, None, :],
                     Lk / np.maximum(ncell[..., None], 1))     # (14,3,3,C)
        return np.einsum("rzs,rzsc->c", W, P)

    # G5: full-sample contraction == einsum fold
    K_P1, kinfo = build_p1_kernel(E, fold, sparse)
    mu_ref_c = c_marginal(mu_sig_p1(K_P1, fold))
    mu_chk_c = predict_c(det_full, tot_full, Lk_full, ncell_full,
                         Lm_full, nmarg_full)
    rel = float(np.max(np.abs(mu_chk_c - mu_ref_c))
                / max(mu_ref_c.max(), 1e-30))
    if rel > 1e-9:
        raise P1RefoldGuardError(f"G5: contraction mismatch {rel}")

    fp_c = c_marginal(fold["mu_fp"])
    A = fold["A"]
    scale = g / (g - 1.0)

    # ---- delete-one replicates -----------------------------------------
    KC = kcnt_h.sum(axis=0)
    KS = ksum_h.sum(axis=0)
    TC = tcnt_h.sum(axis=0)
    Mg_full = Mg_h.sum(axis=0)
    Mc_full = Mc_h.sum(axis=0)

    theta = np.empty((g, 20))
    for i in range(g):
        kc = KC - kcnt_h[i]
        tc = TC - tcnt_h[i]
        theta[i, 0:7] = np.where(tc > 0, kc / np.maximum(tc, 1), np.nan)
        theta[i, 7:14] = np.where(kc > 0,
                                  (KS - ksum_h[i]) / np.maximum(kc, 1),
                                  np.nan)
        Mg_i = (Mg_full - Mg_h[i]) * scale
        theta[i, 14:17] = Mg_i
        mu_c_i = predict_c(det_full - det_h[i], tot_full - tot_h[i],
                           Lk_full - Lk_h[i], ncell_full - ncell_h[i],
                           Lm_full - Lm_h[i], nmarg_full - nmarg_h[i])
        Mc_i = (Mc_full - Mc_h[i]) * scale
        theta[i, 17:20] = A @ (mu_c_i + Mc_i + fp_c)
    if not np.all(np.isfinite(theta)):
        bad = np.where(~np.isfinite(theta))
        raise P1RefoldGuardError(
            f"non-finite replicate components at {bad[1][:10].tolist()}")

    tb = theta.mean(axis=0)
    dev = theta - tb
    cov = (g - 1.0) / g * dev.T @ dev

    # ---- G7: numerical validation --------------------------------------
    if not np.all(np.isfinite(cov)):
        raise P1RefoldGuardError("G7: non-finite covariance")
    asym = float(np.max(np.abs(cov - cov.T)))
    if asym > 0:
        cov = 0.5 * (cov + cov.T)
    evals = np.linalg.eigvalsh(cov)
    if evals[0] < -PSD_TOL * max(evals[-1], 1e-300):
        raise P1RefoldGuardError(
            f"G7: covariance not PSD within tolerance (min eig {evals[0]})")

    # ---- G8: battery-bin sigma sanity vs committed p1_joint_cov --------
    jc = json.load(open(os.path.join(_HERE, "p1_joint_cov.json")))
    sig = np.sqrt(np.diag(cov))
    g8 = []
    for i, b in enumerate(jc["bins"]):
        rC = float(sig[i] / b["sigma_C_jk"])
        rK = float(sig[7 + i] / b["sigma_K_jk"])
        g8.append({"N": b["N"], "sigmaC_ratio": rC, "sigmaK_ratio": rK})
        if not (0.5 < rC < 2.0 and 0.5 < rK < 2.0):
            raise P1RefoldGuardError(
                f"G8: sigma ratio out of range at bin {b['N']}: "
                f"C {rC:.2f}, K {rK:.2f}")

    # ---- block diagnostics ---------------------------------------------
    cnts = tot_h.sum(axis=(1, 2))
    ess = float(cnts.sum() ** 2 / np.sum(cnts ** 2))
    max_share = float(cnts.max() / cnts.sum())
    D = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(D, D)
    Sigma_G = cov[17:20, 17:20]
    sigma_G = np.sqrt(np.diag(Sigma_G))
    # dominant C/K/M cross-correlations with the predicted groups + M
    dom = []
    for a in range(14):
        for bcol in range(14, 20):
            r = float(corr[a, bcol])
            if abs(r) > 0.3:
                dom.append({"pair": [LABELS[a], LABELS[bcol]], "corr": r})
    corr_MG = {f"corr(M_{gn}, Gpred_{gn})":
               float(corr[14 + i, 17 + i]) for i, gn in
               enumerate(("G1", "G2", "G3"))}

    prov = provenance()
    np.savez(
        OUT_NPZ,
        schema=np.array(SCHEMA), version=np.array([1]),
        labels=np.array(LABELS),
        matrix=cov, eigenvalues=evals,
        Sigma_G=Sigma_G,
        battery_bins=np.array(BATTERY_BINS, float),
        group_edges=np.array(PRIMARY_GROUP_EDGES, float),
        n_blocks=np.array([g]), ess_blocks=np.array([ess]),
        max_block_truth_share=np.array([max_share]),
        theta_mean=tb,
        provenance_json=np.array(json.dumps(prov)))
    art_sha = sha256(OUT_NPZ)

    summary = dict(
        schema=SCHEMA, date=time.strftime("%Y-%m-%d"),
        artifact=OUT_NPZ, artifact_sha256=art_sha,
        resampling=("whole-healpix delete-one jackknife, nside 16 nested, "
                    "ONE shared block universe for C, K, M and the "
                    "predicted groups; count components use the g/(g-1) "
                    "pseudo-total scaling"),
        n_blocks=g, ess_blocks=ess, max_block_truth_share=max_share,
        component_ordering=LABELS,
        gates=dict(
            identity_integer_exact=True,
            migration_groups_committed=True,
            landing_mass_closure=True,
            deployed_rebuild_rel_err=fold["rebuild_rel_err"],
            contraction_rel_err=rel,
            psd_min_eig=float(evals[0]), psd_max_eig=float(evals[-1]),
            asymmetry_before_symmetrize=asym,
            sigma_vs_p1_joint_cov=g8),
        theta_full_sample=dict(
            C_paf=[float(v) for v in KC / TC],
            K_mean=[float(v) for v in KS / KC],
            M_groups=[float(v) for v in Mg_full],
            Gpred_calibration_side=[float(v) for v in
                                    A @ (mu_ref_c + Mc_full + fp_c)]),
        sigma=dict(zip(LABELS, [float(v) for v in sig])),
        Sigma_G=[[float(v) for v in row] for row in Sigma_G],
        sigma_G=dict(zip(("G1", "G2", "G3"),
                         [float(v) for v in sigma_G])),
        dominant_cross_correlations_gt_0p3=dom,
        corr_M_vs_Gpred=corr_MG,
        migration=dict(n_net_total=mig["n_net_total"],
                       n_out_of_grid=mig["n_out_of_grid"]),
        kernel=dict(n_events_live=int(len(E["N"])),
                    n_out_of_grid_landing=kinfo["n_out_of_grid"],
                    n_cells_marginal_inherited=int(np.sum(use_marg))),
        independence_note=("NO independent-error assumption: all "
                           "components share the same deleted healpix; "
                           "cross-covariances carried in `matrix`."),
        no_observed_counts=("pack `counts` never referenced; "
                            "calibration-side only"),
        provenance=prov, wall_s=round(time.time() - t0, 1))
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=1)

    print(f"blocks g={g}  ess={ess:.0f}  max_share={max_share:.4f}")
    print("sigma_G:", np.round(sigma_G, 2).tolist())
    print("Gpred (cal side):",
          np.round(A @ (mu_ref_c + Mc_full + fp_c), 1).tolist())
    print("corr(Gpred):", np.round(corr[17:20, 17:20], 3).tolist())
    print("dominant |corr|>0.3 pairs:", len(dom))
    print("PSD eigs:", [f"{v:.3g}" for v in evals])
    print("wrote", OUT_JSON, "and", OUT_NPZ, f"({art_sha[:12]}…)")


def load_p1_ckm_cov(path=OUT_NPZ, expect_schema=SCHEMA, expect_version=1):
    """Fail-loud loader: schema/version/ordering/shape/finite/PSD re-checked
    at every load; arrays returned read-only.  No renormalization helper
    exists in this module and none may be added."""
    z = np.load(path, allow_pickle=False)
    need = ["schema", "version", "labels", "matrix", "eigenvalues",
            "Sigma_G", "battery_bins", "group_edges", "n_blocks",
            "ess_blocks", "theta_mean", "provenance_json"]
    missing = [k for k in need if k not in z]
    if missing:
        raise P1RefoldGuardError(f"ckm-cov artifact missing: {missing}")
    if str(z["schema"]) != expect_schema \
            or int(np.asarray(z["version"]).ravel()[0]) != expect_version:
        raise P1RefoldGuardError(
            f"ckm-cov schema mismatch: {z['schema']} v{z['version']}")
    labels = [str(v) for v in z["labels"]]
    if labels != LABELS:
        raise P1RefoldGuardError("ckm-cov component ordering mismatch")
    cov = np.asarray(z["matrix"], float)
    if cov.shape != (20, 20) or not np.all(np.isfinite(cov)):
        raise P1RefoldGuardError("ckm-cov matrix bad shape / non-finite")
    if np.max(np.abs(cov - cov.T)) > 0:
        raise P1RefoldGuardError("ckm-cov matrix not symmetric")
    ev = np.linalg.eigvalsh(cov)
    if ev[0] < -PSD_TOL * max(ev[-1], 1e-300):
        raise P1RefoldGuardError("ckm-cov matrix not PSD at load")
    SG = np.asarray(z["Sigma_G"], float)
    if not np.allclose(SG, cov[17:20, 17:20], rtol=0, atol=0):
        raise P1RefoldGuardError("ckm-cov Sigma_G != matrix block")
    out = {k: np.asarray(z[k]) for k in need if k != "provenance_json"}
    out["provenance"] = json.loads(str(z["provenance_json"]))
    for v in out.values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return out


if __name__ == "__main__":
    main()
