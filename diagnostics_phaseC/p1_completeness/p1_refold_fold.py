#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared fold library for the GATED P1 refold (PI ruling 2026-08-08 §5).

This module holds the DETERMINISTIC building blocks used by BOTH the
frozen (C, K, M_<19.5) covariance builder (`p1_ckm_cov.py`) and the
two-phase gated refold runner (`p1_refold.py`).  Nothing here reads an
observed count into any output — the observed side enters only in the
refold runner's `--phase close`.

DESIGN (stated before any run; every choice is a committed/frozen object)
------------------------------------------------------------------------
* Fold plumbing = the committed Phase-B fold, rebuilt exactly as the
  committed truth-by-SNR refold does (`run_truth_by_snr.py`), with the
  bit-level rebuild guard against `forward_selftest.selftest` executed
  as a PRECONDITION: einsum("skcb,sb,bk,bks->cks", K, C_bs, g_bk,
  alloc) must reproduce the deployed mu_sig to <=1e-8 relative before
  any P1 quantity is computed.  C path (eta_hat -> sigmoid), g_bk,
  pathlength truth allocation and the FP fold are the deployed objects,
  byte-unchanged.
* K (response) = the CERTIFIED P1 natural-pair kernel: the EMPIRICAL
  landing distributions P(N-hat bin | truth 0.2-dex bin, z-cell, SNR
  stratum) of exactly the frozen kernel event set (`p1_natpair_ck/v1`;
  the deployed completeness-numerator events, loaded through the
  fail-loud loader and re-verified against `K_rep` cell-by-cell at
  integer/1e-9 precision).  This is the battery v2 "joint operator"
  construction (per-group landing probabilities), extended from the
  validation overlap to every observed bin: pairs-faithful,
  non-parametric, ZERO added parameters, no clamp, no polynomial, no
  extrapolation.  Sparse cells (the artifact's FROZEN `K_sparse_flag`)
  inherit the same-N-bin live marginal (the frozen inheritance rule).
* Truth support of the (C,K) fold = N_true >= 19.5 (the ratified
  primary representation).  The pack's pad truth bins [19.0,19.5) are
  EXCLUDED from the fold and replaced by the EXPLICIT below-floor
  migration source term M_<19.5 — the committed `p1_migration.json`
  definition (net 17.2-chain attribution, competition-reassignment
  excluded), re-binned deterministically onto the observed 0.1-dex
  grid and gate-checked against the committed group totals
  (G1=4088 / G2=144 / G3=0).  K is never renormalized.
* z / SNR cell conventions: events are binned by the ARTIFACT
  convention (z vs (2.56, 2.96); S2N vs (3.5, 6.5)); fold z-bins and
  SNR strata map to those cells through the fold's own composite maps
  (kz_to_K -> K_to_zresp; s_to_sresp) — exactly how the deployed fold
  consumes its response cells.
* Mass above the observed grid top (N-hat >= 22.4) is NOT renormalized
  away: it leaves the observed support exactly as it does in the
  deployed fold (and in the observed counts).  It is counted and
  reported.

Support identifiers: fold truth support N_true >= 19.5 (primary
certification is truth >= 20.3; the [19.5, 20.3) truth rows carry the
low-boundary RESTRICTED status and are labeled in every output);
observed support = the pack N-hat grid [19.5, 22.4); groups G1/G2/G3 =
the frozen PRIMARY_GROUP_EDGES.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "injection"))

PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz")
CK_ARTIFACT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/stage0/p1_natpair_ck_v1.npz")
CACHE195 = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
            "p1_completeness_cache.npz")
CACHE172 = ("/scratch/cavestru_root/cavestru0/mfho/phaseC_resp/"
            "p1_completeness_cache_172.npz")
ZCAT = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
        "v2.8.5/mock-0/loa-124/zcat.fits")

NSIDE = 16
FLOOR = 19.5
P_DLA_MIN = 0.99
REPORT_N_EDGES = np.round(19.5 + 0.2 * np.arange(16), 10)      # 19.5 .. 22.5
ZR_EDGES = (2.56, 2.96)
SR_EDGES = (3.5, 6.5)
SPARSE_N_MIN = 25
BATTERY_BINS = [(19.5, 20.0), (20.0, 20.4), (20.4, 20.7), (20.7, 21.0),
                (21.0, 21.3), (21.3, 21.7), (21.7, 22.4)]
# committed migration group totals (p1_migration.json) — gate reference
MIGRATION_GROUPS_REF = {"G1": 4088, "G2": 144, "G3": 0}

REBUILD_TOL = 1e-8


class P1RefoldGuardError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


HEALPIX_MAP = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/stage0/p1_healpix_map_nside16.npz")

_HPX_CACHE = {}


def healpix_of(tids):
    """TARGETID -> nside-16 nested healpix (frozen convention, the same map
    `p1_joint_cov.py` and the stability jackknife used), served from the
    materialized sidecar written by `p1_healpix_map.py`."""
    if "map" not in _HPX_CACHE:
        z = np.load(HEALPIX_MAP, allow_pickle=False)
        if str(z["schema"]) != "p1_healpix_map/v1" \
                or int(np.asarray(z["nside"]).ravel()[0]) != NSIDE \
                or not bool(np.asarray(z["nested"]).ravel()[0]) \
                or str(z["zcat"]) != ZCAT:
            raise P1RefoldGuardError("healpix sidecar convention mismatch")
        _HPX_CACHE["map"] = (np.asarray(z["targetid"], np.int64),
                             np.asarray(z["healpix"], np.int64))
    tid_s, hpx_s = _HPX_CACHE["map"]
    t = np.asarray(tids, np.int64)
    idx = np.searchsorted(tid_s, t)
    bad = (idx >= len(tid_s)) | (tid_s[np.minimum(idx, len(tid_s) - 1)] != t)
    if np.any(bad):
        raise P1RefoldGuardError(
            f"unmapped healpix for {int(np.sum(bad))} TARGETIDs")
    return hpx_s[idx]


# ---------------------------------------------------------------------------
# frozen event sets
# ---------------------------------------------------------------------------

def load_kernel_events():
    """The frozen kernel event set + truth rows, with cell assignments.

    Loads the atomic artifact through the fail-loud loader FIRST (estimand
    id + identity + miss closure re-verified at load), then rebuilds the
    event-level set from the hash-pinned cache and re-verifies the per-cell
    integer identity against the loaded artifact.
    """
    from p1_ck_loader import load_p1_ck
    from build_p1_natpair_ck import extract_kernel_events

    art = load_p1_ck(CK_ARTIFACT)
    ev, d = extract_kernel_events()

    kin = ev["IN_KERNEL"]
    live = ev["S2N"] > 2.0
    m = kin & live
    E = dict(N=ev["N"][m], NHAT=ev["NHAT"][m], DX=ev["DX"][m],
             Z=ev["Z"][m], S2N=ev["S2N"][m], TID=ev["TID"][m])

    # artifact-convention cell assignment
    E["BREP"] = np.digitize(E["N"], REPORT_N_EDGES) - 1        # 0..14
    if np.any((E["BREP"] < 0) | (E["BREP"] > 14)):
        raise P1RefoldGuardError("kernel event outside [19.5, 22.5) truth")
    E["ZR"] = np.digitize(E["Z"], ZR_EDGES)                    # 0,1,2
    E["SR"] = np.digitize(E["S2N"], SR_EDGES)                  # 0,1,2

    # per-cell coherence gate vs the loaded artifact's K_rep (same events)
    rep = np.asarray(art["K_rep"], float)                      # (15,3,3,5)
    marg = np.asarray(art["K_marginal"], float)                # (15,5)
    for b in range(15):
        mb = E["BREP"] == b
        n_m = int(np.sum(mb))
        if int(marg[b, 0]) != n_m:
            raise P1RefoldGuardError(
                f"K_marginal n mismatch at bin {b}: {marg[b,0]} != {n_m}")
        if n_m and abs(float(E["DX"][mb].mean()) - marg[b, 1]) > 1e-9:
            raise P1RefoldGuardError(f"K_marginal mean mismatch at bin {b}")
        for zi in range(3):
            for si in range(3):
                mc = mb & (E["ZR"] == zi) & (E["SR"] == si)
                if int(rep[b, zi, si, 0]) != int(np.sum(mc)):
                    raise P1RefoldGuardError(
                        f"K_rep n mismatch at ({b},{zi},{si})")
    sparse = np.asarray(art["K_sparse_flag"], bool)            # (15,3,3)

    truth = dict(N=d["tr_NHI"], S2N=d["tr_S2N"], TID=d["tr_TARGETID"])
    return E, truth, sparse, art, d


def load_migration(nhat_edges):
    """Below-floor net migration on the observed grid (frozen definition).

    Identical selection/attribution to the committed `p1_migration.py`
    (net = 17.2-chain subfloor TP, selected, NOT a >=19.5-truth TP of the
    nhi195 chain); only the reporting binning is the observed 0.1-dex
    grid.  Gate: group totals must equal the committed record exactly.
    Returns per-bin counts, the migrant row TID/N-hat arrays, and the
    out-of-grid count.
    """
    d = np.load(CACHE172)
    d5 = np.load(CACHE195)
    sel = ((d["cat_P_DLA"] > P_DLA_MIN) & d["cat_good"]
           & (d["cat_S2N"] > 2.0) & (d["cat_NHI"] > FLOOR))
    sel5 = ((d5["cat_P_DLA"] > P_DLA_MIN) & d5["cat_good"]
            & (d5["cat_S2N"] > 2.0) & (d5["cat_NHI"] > FLOOR))
    tp195keys = set(zip(d5["cat_TARGETID"][d5["cat_is_TP"] & sel5].tolist(),
                        np.round(d5["cat_Z_DLA"][d5["cat_is_TP"] & sel5],
                                 6).tolist()))
    rowkeys = list(zip(d["cat_TARGETID"].tolist(),
                       np.round(d["cat_Z_DLA"], 6).tolist()))
    in195 = np.array([k in tp195keys for k in rowkeys])
    net = sel & d["cat_is_TP"] & (d["cat_NHI_TRUE"] < FLOOR) & ~in195

    nhat = d["cat_NHI"][net]
    tid = d["cat_TARGETID"][net]
    ne = np.asarray(nhat_edges, float)
    ci = np.digitize(nhat, ne) - 1
    in_grid = (ci >= 0) & (ci < len(ne) - 1) & (nhat < ne[-1])
    M_c = np.bincount(ci[in_grid], minlength=len(ne) - 1).astype(float)
    n_out = int(np.sum(~in_grid))

    # gate: committed group totals reproduced exactly
    from CDDF_analysis.hbi_mcmc.gate_covariance import PRIMARY_GROUP_EDGES
    for (glo, ghi), gname in zip(PRIMARY_GROUP_EDGES, ("G1", "G2", "G3")):
        n_g = int(np.sum((nhat >= glo) & (nhat < ghi)))
        if n_g != MIGRATION_GROUPS_REF[gname]:
            raise P1RefoldGuardError(
                f"migration {gname} = {n_g} != committed "
                f"{MIGRATION_GROUPS_REF[gname]}")
    return dict(M_c=M_c, TID=np.asarray(tid, np.int64),
                NHAT=np.asarray(nhat, float), CI=ci, in_grid=in_grid,
                n_out_of_grid=n_out, n_net_total=int(np.sum(net)))


# ---------------------------------------------------------------------------
# fold plumbing (deployed, rebuilt with the committed guard)
# ---------------------------------------------------------------------------

def build_fold(pk):
    """Deployed fold pieces + the committed bit-level rebuild guard."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K
    from CDDF_analysis.hbi_mcmc.gate_covariance import (
        group_aggregator, PRIMARY_GROUP_EDGES)

    consts = build_consts(pk, resp_clamp="both")
    st = FS.selftest(pk, resp_clamp="both")
    dX = np.asarray(pk.dX, float)
    live = (dX > 0)                                            # (Kf, S)
    kz = np.asarray(consts.kz_to_K)
    K_dep = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)),
                               consts))
    K_dep_full = K_dep[:, kz]                                  # (S,Kf,C,B)
    C_cells = 1.0 / (1.0 + np.exp(-np.asarray(consts.eta_hat)))  # (S, 12)
    b_to_cell = np.asarray(consts.b_to_cell)
    C_bs = C_cells[:, b_to_cell]                               # (S, B)
    g_bk = np.asarray(consts.g_bk)                             # (B, Kf)
    tc = np.asarray(pk.truth_counts, float)                    # (B, Kf)
    dX_tot = dX.sum(axis=1)
    share = dX / np.maximum(dX_tot[:, None], 1e-30)            # (Kf, S)
    alloc = tc[:, :, None] * share[None, :, :]                 # (B, Kf, S)

    # the committed rebuild guard (run_truth_by_snr.py)
    mu_sig_rebuilt = np.einsum("skcb,sb,bk,bks->cks",
                               K_dep_full, C_bs, g_bk, alloc)
    live3 = live[None, :, :]
    mu_sig_rebuilt = np.where(live3, mu_sig_rebuilt, 0.0)
    mu_sig_ref = np.where(live3, np.asarray(st["mu_sig"]), 0.0)
    err = float(np.max(np.abs(mu_sig_rebuilt - mu_sig_ref))
                / max(mu_sig_ref.max(), 1e-30))
    if err > REBUILD_TOL:
        raise P1RefoldGuardError(f"deployed-fold rebuild failed: {err}")

    ne = np.asarray(pk.nhat_edges, float)
    ntrue = np.asarray(pk.ntrue_edges, float)
    A = group_aggregator(pk, PRIMARY_GROUP_EDGES)

    # composite response-cell maps (exactly how the fold consumes cells)
    s_to_sr = np.asarray(consts.s_to_sresp)                    # (S,) -> 0..2
    k_to_zr = np.asarray(consts.K_to_zresp)[kz]                # (Kf,) -> 0..2

    # pack truth bin -> P1 reporting bin (>=19.5 only; top bin merged)
    b_used = np.where(ntrue[:-1] >= FLOOR - 1e-9)[0]
    b_rep = {}
    for b in b_used:
        lo, hi = ntrue[b], ntrue[b + 1]
        i0 = int(np.searchsorted(REPORT_N_EDGES, lo + 1e-9) - 1)
        i1 = int(np.searchsorted(REPORT_N_EDGES, hi - 1e-9) - 1)
        b_rep[int(b)] = list(range(i0, i1 + 1))                # 1 or 2 bins
    return dict(consts=consts, st=st, live=live, live3=live3,
                K_dep_full=K_dep_full, C_cells=C_cells, C_bs=C_bs,
                b_to_cell=b_to_cell, g_bk=g_bk, tc=tc, share=share,
                alloc=alloc, A=A, nhat_edges=ne, ntrue_edges=ntrue,
                s_to_sr=s_to_sr, k_to_zr=k_to_zr, b_used=b_used,
                b_rep=b_rep, rebuild_rel_err=err,
                mu_fp=np.where(live3, np.asarray(st["mu_fp"]), 0.0),
                obs_counts=np.where(live3, np.asarray(st["counts"]), 0.0))


def landing_tables(E, nhat_edges):
    """Total landing-count tables from the frozen event set.

    Returns L (15, 3, 3, n_c) cell counts, Lm (15, n_c) marginals, and the
    out-of-grid count (N-hat >= grid top).
    """
    ne = np.asarray(nhat_edges, float)
    n_c = ne.size - 1
    ci = np.digitize(E["NHAT"], ne) - 1
    in_grid = (ci >= 0) & (ci < n_c) & (E["NHAT"] < ne[-1])
    L = np.zeros((15, 3, 3, n_c))
    Lm = np.zeros((15, n_c))
    np.add.at(L, (E["BREP"][in_grid], E["ZR"][in_grid], E["SR"][in_grid],
                  ci[in_grid]), 1.0)
    np.add.at(Lm, (E["BREP"][in_grid], ci[in_grid]), 1.0)
    return L, Lm, int(np.sum(~in_grid))


def landing_probability(L, Lm, n_ev_cell, n_ev_marg, sparse):
    """P(land in c | rep bin, zr, sr) with the FROZEN sparse rule.

    Denominators are the cells' TOTAL event counts (in-grid + out-of-grid)
    so the out-of-grid mass leaves the observed support, exactly as in the
    deployed fold.  Sparse cells (frozen artifact flags, or empty) inherit
    the same-bin live marginal.
    """
    P = np.zeros_like(L)                                        # (15,3,3,C)
    Pm = np.zeros_like(Lm)                                      # (15,C)
    for b in range(15):
        if n_ev_marg[b] > 0:
            Pm[b] = Lm[b] / n_ev_marg[b]
    for b in range(15):
        for zi in range(3):
            for si in range(3):
                n = n_ev_cell[b, zi, si]
                if sparse[b, zi, si] or n < SPARSE_N_MIN:
                    P[b, zi, si] = Pm[b]
                else:
                    P[b, zi, si] = L[b, zi, si] / n
    return P, Pm


def cell_event_counts(E):
    """Total event counts per (rep bin, zr, sr) and per rep bin (incl.
    out-of-grid landers — the landing denominators)."""
    n_cell = np.zeros((15, 3, 3))
    np.add.at(n_cell, (E["BREP"], E["ZR"], E["SR"]), 1.0)
    n_marg = np.zeros(15)
    np.add.at(n_marg, E["BREP"], 1.0)
    return n_cell, n_marg


def build_p1_kernel(E, fold, sparse):
    """K_P1_full (S, Kf, C, B): the P1 empirical landing kernel on the fold
    grids.  Pack pad bins (< 19.5) are zero.  The top pack truth bin
    [22.1, 22.4) uses the event-merged [22.1, 22.5) landing row."""
    n_c = fold["nhat_edges"].size - 1
    L, Lm, n_out = landing_tables(E, fold["nhat_edges"])
    n_cell, n_marg = cell_event_counts(E)

    # merge rep bins 13, 14 (pack top truth bin [22.1, 22.4)) by COUNTS
    Lmrg = L.copy()
    Lmmrg = Lm.copy()
    n_cell_m = n_cell.copy()
    n_marg_m = n_marg.copy()
    Lmrg[13] = L[13] + L[14]
    Lmmrg[13] = Lm[13] + Lm[14]
    n_cell_m[13] = n_cell[13] + n_cell[14]
    n_marg_m[13] = n_marg[13] + n_marg[14]
    sparse_m = sparse.copy()
    sparse_m[13] = sparse[13] & sparse[14]

    P, Pm = landing_probability(Lmrg, Lmmrg, n_cell_m, n_marg_m, sparse_m)

    S = fold["C_bs"].shape[0]
    Kf = fold["g_bk"].shape[1]
    B = fold["tc"].shape[0]
    K_P1 = np.zeros((S, Kf, n_c, B))
    for b in fold["b_used"]:
        reps = fold["b_rep"][int(b)]
        r = min(reps[0], 13)                    # top bin -> merged row 13
        for k in range(Kf):
            zr = fold["k_to_zr"][k]
            for s in range(S):
                sr = fold["s_to_sr"][s]
                K_P1[s, k, :, b] = P[r, zr, sr]
    return K_P1, dict(P=P, Pm=Pm, L=Lmrg, Lm=Lmmrg, n_cell=n_cell_m,
                      n_marg=n_marg_m, sparse=sparse_m,
                      n_out_of_grid=n_out)


def mu_sig_p1(K_P1, fold):
    """(C, Kf, S) predicted signal-mean of the P1 fold (truth >= 19.5)."""
    alloc = fold["alloc"].copy()
    pad = fold["ntrue_edges"][:-1] < FLOOR - 1e-9
    alloc[pad] = 0.0
    mu = np.einsum("skcb,sb,bk,bks->cks",
                   K_P1, fold["C_bs"], fold["g_bk"], alloc)
    return np.where(fold["live3"], mu, 0.0)


def c_marginal(x):
    return x.sum(axis=(1, 2))


def provenance():
    return dict(
        git_commit=git_head(),
        pack=PACK, pack_sha256=sha256(PACK),
        ck_artifact=CK_ARTIFACT, ck_artifact_sha256=sha256(CK_ARTIFACT),
        cache195=CACHE195, cache195_sha256=sha256(CACHE195),
        cache172=CACHE172, cache172_sha256=sha256(CACHE172),
        zcat=ZCAT,
        estimand_id="p1_natpair_ck/v1",
        migration_schema="p1_migration/v1 (frozen definition, 0.1-dex "
                         "re-binning; committed group totals gate-checked)",
        fold_support="truth N_true >= 19.5 (primary certified >= 20.3; "
                     "[19.5, 20.3) truth carries RESTRICTED low-boundary "
                     "status); observed support = pack N-hat grid "
                     "[19.5, 22.4); groups = frozen G1/G2/G3",
    )
