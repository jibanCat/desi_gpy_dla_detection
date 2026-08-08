#!/usr/bin/env python
"""measure_phaseC_pairs.py — matched-pair measurement for a Phase-C response arm.

Scores a GP re-inference run on an anchored injection arm
(`gen_phaseC_resp.py`) against the arm's injection manifest, using THE
production matching object — `match_truth_to_cat_molly` imported from
`examples.molly_faithful_pc_plots` (the same function `load_and_cut_catalog`
uses, at the same window `dz_rel = 0.01`, same NHI-descending greedy order,
same min-|ΔNHI| tie-break). Production ORDER is reproduced: the sentinel
filter (`NHI_ERR == −1 | Z_DLA_ERR == −1` rows dropped) runs BEFORE
matching (load_and_cut_catalog step 3), matching runs before cuts (step
5), then the op-mask (P_DLA > 0.99 strict, native red-side SNR > 2
strict, DLAFLAG == 0 where the column exists) applies to the matched rows
as in `measure_znz_response`. REMAINING KNOWN DIFFERENCE (review finding
F3, open pre-production item): the production λ_rf analysis-window /
z_QSO / BAL cuts are NOT applied here — required before any Stage-2
scoring, immaterial for the pilot's per-anchor pair moments. No new
matching convention is introduced.

Outputs per (logN anchor × z anchor × SNR stratum) and per response cell:
  n_injected, n_matched_op (completeness w/ Jeffreys CI), dx = N̂ − N_true
  moments (mean, sd, skew) + raw dx lists, multi-candidate statistics,
  unmatched op-detections on injected sightlines split by proximity to
  NATURAL truth HCDs (prodlike substrate context), and the pilot-precision
  comparison of measured anchor moments against the OLD response envelope's
  moment surfaces (bias + sd at the clamped covariate) per cell.

ENGINEERING VALIDATION ONLY at pilot scale (design §10): pilot numbers never
enter a production artifact.

Usage:
  python injection/measure_phaseC_pairs.py \
      --arm /scratch/.../phaseC_resp/pilot_prodlike [--out pairs.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from astropy.table import Table, vstack

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from examples.molly_faithful_pc_plots import match_truth_to_cat_molly  # noqa: E402

DEFAULT_ENVELOPE = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                    "track_c/stage0/forward_response_2lpt0.npz")
DEFAULT_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                   "qq_desi_y3/v2.8.5/mock-0/loa-124")
DZ_REL = 0.01           # HBIConfig.dz_rel default — THE production window
P_DLA_MIN = 0.99        # production op-mask (strict >)
SNR_MIN = 2.0           # production op-mask (strict >)
RESP_SNR_EDGES = (2.0, 3.5, 6.5, np.inf)
RESP_Z_EDGES = (0.0, 2.56, 2.96, np.inf)


def _load_dlacat(gp_out):
    paths = sorted(glob.glob(os.path.join(gp_out, "**", "dlacat-*.fits"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no dlacat-*.fits under {gp_out} (GP not finished?)")
    tabs = [Table.read(p) for p in paths]
    t = vstack(tabs, metadata_conflicts="silent") if len(tabs) > 1 else tabs[0]
    return t, paths


def _cell(x, edges):
    return int(np.clip(np.digitize([x], edges)[0] - 1, 0, len(edges) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--envelope", default=DEFAULT_ENVELOPE)
    ap.add_argument("--mockdir", default=DEFAULT_MOCKDIR,
                    help="natural-truth source for unmatched-row attribution")
    ap.add_argument("--out", default=None)
    ap.add_argument("--role", default="pilot-validation",
                    help="ONLY manifest rows with this sidecar role are "
                         "scored (F7 role enforcement; fail-loud)")
    ap.add_argument("--evaluation-step", action="store_true",
                    help="required to score role=held-out-evaluation")
    a = ap.parse_args()

    if a.role == "held-out-evaluation" and not a.evaluation_step:
        raise SystemExit(
            "REFUSED: scoring held-out-evaluation rows requires "
            "--evaluation-step (the frozen-statistic evaluation only; "
            "rulings §5/§12 — holdout must never enter calibration).")

    man = Table.read(os.path.join(a.arm, "injection_truth.fits"))
    roles = json.load(open(os.path.join(a.arm, "roles.json")))
    role_map = {int(k): v["role"] for k, v in roles["roles"].items()}
    if "inj_id" not in man.colnames:
        raise SystemExit("injection_truth.fits lacks inj_id — regenerate")
    man_roles = np.array([role_map.get(int(i), "MISSING")
                          for i in man["inj_id"]])
    if np.any(man_roles == "MISSING"):
        raise SystemExit("roles.json does not cover every manifest row "
                         "(fail-loud; F7)")
    keep = man_roles == a.role
    n_role_excluded = int((~keep).sum())
    hold_hpx = set(roles.get("holdout_healpix") or [])
    if a.role != "held-out-evaluation" and hold_hpx:
        bad = np.isin(np.asarray(man["healpix"], np.int64)[keep],
                      sorted(hold_hpx))
        if bad.any():
            raise SystemExit("role/healpix inconsistency: non-holdout role "
                             "selected rows on holdout healpix (fail-loud)")
    man = man[keep]
    if len(man) == 0:
        raise SystemExit(f"no manifest rows with role {a.role!r}")

    # F3 truth-side estimand support (BEFORE matching): only injections
    # whose z_true sits inside the sightline's ANALYSIS window and whose
    # z_qso is in (2, 4.25) belong to the response estimand. Production
    # generation guarantees this by construction; pilot arms have a
    # reported out-of-window fraction.
    from gen_phaseC_resp import analysis_window, ZQSO_MIN, ZQSO_MAX
    man_win = np.array([analysis_window(float(z)) for z in man["z_qso"]])
    man_in_win = ((np.asarray(man["z_qso"], float) > ZQSO_MIN)
                  & (np.asarray(man["z_qso"], float) < ZQSO_MAX)
                  & (np.asarray(man["z_true"], float) > man_win[:, 0])
                  & (np.asarray(man["z_true"], float) < man_win[:, 1]))
    n_inj_out_of_window = int((~man_in_win).sum())
    man = man[man_in_win]
    if len(man) == 0:
        raise SystemExit("no in-window manifest rows")
    dla, dlacat_paths = _load_dlacat(os.path.join(a.arm, "gp_out"))

    # production step 3: sentinel rows dropped BEFORE matching (F3 fix)
    n_sentinel = 0
    if "NHI_ERR" in dla.colnames and "Z_DLA_ERR" in dla.colnames:
        sent = ((np.asarray(dla["NHI_ERR"], float) == -1)
                | (np.asarray(dla["Z_DLA_ERR"], float) == -1))
        n_sentinel = int(sent.sum())
        dla = dla[~sent]

    # the matcher's cat/truth column contract
    cat = Table()
    cat["TARGETID"] = np.asarray(dla["TARGETID"], np.int64)
    cat["Z_DLA"] = np.asarray(dla["Z_DLA"], float)
    cat["NHI"] = np.asarray(dla["NHI"], float)
    cat["P_DLA"] = np.asarray(dla["P_DLA"], float)
    dlaflag_ok = (np.asarray(dla["DLAFLAG"], float) == 0
                  if "DLAFLAG" in dla.colnames
                  else np.ones(len(dla), bool))
    truth = Table()
    truth["TARGETID"] = np.asarray(man["target_id"], np.int64)
    truth["Z_TRUTH"] = np.asarray(man["z_true"], float)
    truth["NHI"] = np.asarray(man["logN_true"], float)

    is_tp, nhi_tr, z_tr, truth_matched = match_truth_to_cat_molly(
        cat, truth, DZ_REL, cat_iter_order="nhi_desc")

    # native SNR per injected sightline (from the manifest)
    tid2snr = {int(t): float(s) for t, s in zip(man["target_id"],
                                                man["native_snr"])}
    cat_snr = np.array([tid2snr.get(int(t), np.nan)
                        for t in cat["TARGETID"]])
    # F3 cat-side: the ANALYSIS-window geometry per sightline (=
    # make_lambda_z_BAL_cuts direct form) + z_qso in (2, 4.25) on the
    # op-mask. BAL veto holds by substrate construction.
    tid2zq = {int(t): float(z) for t, z in zip(man["target_id"],
                                               man["z_qso"])}
    cat_zq = np.array([tid2zq.get(int(t), np.nan) for t in cat["TARGETID"]])
    win = np.array([analysis_window(z) if np.isfinite(z) else (np.inf, -np.inf)
                    for z in cat_zq])
    cat_zd = np.asarray(cat["Z_DLA"], float)
    in_window = ((cat_zq > ZQSO_MIN) & (cat_zq < ZQSO_MAX)
                 & (cat_zd > win[:, 0]) & (cat_zd < win[:, 1]))
    op = (np.asarray(cat["P_DLA"], float) > P_DLA_MIN) & (cat_snr > SNR_MIN) \
        & dlaflag_ok & in_window

    # per-injection records: matched op-row (via the 1-to-1 match), moments
    # match_truth_to_cat_molly gives cat-side NHI_TRUE; invert to truth-side:
    # a truth row j is recovered iff some cat row ci matched it. Rebuild the
    # (truth j -> cat ci) map by re-walking the match outputs.
    t_tid = np.asarray(truth["TARGETID"], np.int64)
    t_z = np.asarray(truth["Z_TRUTH"], float)
    t_n = np.asarray(truth["NHI"], float)
    c_tid = np.asarray(cat["TARGETID"], np.int64)
    j_by_key = {}
    for j in range(len(truth)):
        j_by_key.setdefault((int(t_tid[j]), round(float(t_n[j]), 6),
                             round(float(t_z[j]), 6)), j)
    truth_cat_row = np.full(len(truth), -1, int)
    for ci in np.where(is_tp)[0]:
        key = (int(c_tid[ci]), round(float(nhi_tr[ci]), 6),
               round(float(z_tr[ci]), 6))
        j = j_by_key.get(key, None)
        if j is not None and truth_cat_row[j] < 0:
            truth_cat_row[j] = ci

    n_match_any = int((truth_cat_row >= 0).sum())
    matched_op_mask = np.array([truth_cat_row[j] >= 0
                                and bool(op[truth_cat_row[j]])
                                for j in range(len(truth))])

    # natural truth for unmatched-row attribution (prodlike context)
    nat = Table.read(os.path.join(a.mockdir, "hcd_truth_cat.fits"))
    nat_by_tid = {}
    for t, z in zip(np.asarray(nat["TARGETID"], np.int64),
                    np.asarray(nat["Z"], float)):
        nat_by_tid.setdefault(int(t), []).append(float(z))
    unmatched_op = np.where(op & ~is_tp)[0]
    near_nat = 0
    for ci in unmatched_op:
        zs = nat_by_tid.get(int(c_tid[ci]), [])
        zc = float(cat["Z_DLA"][ci])
        if any(abs(zc - zn) / (1.0 + zn) < DZ_REL for zn in zs):
            near_nat += 1

    # multi-candidate statistic: op-rows within the window of each injection
    rows_by_tid = {}
    for ci in range(len(cat)):
        rows_by_tid.setdefault(int(c_tid[ci]), []).append(ci)
    n_cand = np.zeros(len(truth), int)
    for j in range(len(truth)):
        for ci in rows_by_tid.get(int(t_tid[j]), []):
            if op[ci] and abs(float(cat["Z_DLA"][ci]) - t_z[j]) \
                    / (1.0 + t_z[j]) < DZ_REL:
                n_cand[j] += 1

    # old-envelope moment surfaces
    env = np.load(a.envelope)
    mu_co, sg_co = env["mu_coef"], env["sig_coef"]
    nref = float(env["N_ref"])
    sig_floor = float(env["sig_floor"])
    # per-cell anchor range for the clamp
    rr = np.stack([np.stack([
        (env["emp_N_anchors"][sr, zr].min(), env["emp_N_anchors"][sr, zr].max())
        for zr in range(3)]) for sr in range(3)])

    # per-anchor-cell aggregation: (logN anchor × response z-cell × SNR
    # stratum) — the z key is the RESPONSE CELL, not the exact z value
    # (production draws z continuously ∝ dX within the cell; the pilot's
    # fixed z-anchors map into the same cells)
    man_z = np.asarray(man["z_true"], float)
    man_n = np.asarray(man["logN_true"], float)
    man_snr = np.asarray(man["native_snr"], float)
    anchors = sorted(set(np.round(man_n, 4).tolist()))
    man_zr = np.array([_cell(z, RESP_Z_EDGES) for z in man_z])
    per_anchor = []
    for A in anchors:
        for zr_i in range(len(RESP_Z_EDGES) - 1):
            for s_i in range(len(RESP_SNR_EDGES) - 1):
                sel = (np.abs(man_n - A) < 1e-6) & (man_zr == zr_i)
                sel &= ((man_snr > RESP_SNR_EDGES[s_i])
                        & (man_snr <= RESP_SNR_EDGES[s_i + 1] if
                           np.isfinite(RESP_SNR_EDGES[s_i + 1]) else True))
                idx = np.where(sel)[0]
                if idx.size == 0:
                    continue
                mo = matched_op_mask[idx]
                dxs, hlist, ztl, zql = [], [], [], []
                for j in idx[mo]:
                    ci = truth_cat_row[j]
                    dxs.append(float(cat["NHI"][ci]) - float(t_n[j]))
                    hlist.append(int(man["healpix"][j]))
                    # additive per-pair DESIGN covariates (2026-08-07,
                    # pre-holdout protocol amendment): needed by the
                    # ratified battery-v2 region/z_qso tests; no
                    # matching/selection semantics change
                    ztl.append(float(man["z_true"][j]))
                    zql.append(float(man["z_qso"][j]))
                dxs = np.array(dxs)
                Z = float(np.median(man_z[idx]))     # descriptive center
                sr, zr = s_i, zr_i
                Ncl = float(np.clip(A, rr[sr, zr, 0], rr[sr, zr, 1]))
                u = np.array([1.0, Ncl - nref, (Ncl - nref) ** 2])
                pred_b = float(mu_co[sr, zr] @ u)
                pred_s = float(max(sg_co[sr, zr] @ u, sig_floor))
                rec = {"logN": A, "z": Z, "snr_stratum": s_i,
                       "resp_cell": [sr, zr],
                       "n_inj": int(idx.size),
                       "n_matched_any": int((truth_cat_row[idx] >= 0).sum()),
                       "n_matched_op": int(mo.sum()),
                       "n_multi_candidate": int((n_cand[idx] > 1).sum()),
                       "dx_mean": float(dxs.mean()) if dxs.size else None,
                       "dx_sd": float(dxs.std(ddof=1)) if dxs.size > 1 else None,
                       "dx": dxs.tolist(),
                       "pair_healpix": hlist,
                       "pair_z_true": ztl, "pair_z_qso": zql,
                       "old_pred_bias": pred_b, "old_pred_sd": pred_s,
                       "old_covariate_clamped": bool(Ncl != A)}
                if dxs.size > 1:
                    se = rec["dx_sd"] / np.sqrt(dxs.size)
                    rec["z_vs_old_mean"] = float((rec["dx_mean"] - pred_b) / se)
                per_anchor.append(rec)

    out = {
        "schema": "phaseC_pairs/v2",
        "label": "PILOT ENGINEERING VALIDATION (design §10) — not production",
        "arm": a.arm, "dlacats": dlacat_paths,
        "matcher": "examples.molly_faithful_pc_plots.match_truth_to_cat_molly "
                   f"dz_rel={DZ_REL} nhi_desc (the production object)",
        "op_mask": {"p_dla_min": P_DLA_MIN, "snr_min": SNR_MIN},
        "n_injected": int(len(truth)),
        "role_scored": a.role,
        "n_role_excluded": n_role_excluded,
        "n_injections_out_of_analysis_window": n_inj_out_of_window,
        "n_sentinel_rows_dropped_prematch": n_sentinel,
        "n_matched_any_p": n_match_any,
        "n_matched_op": int(matched_op_mask.sum()),
        "n_gp_rows": int(len(cat)),
        "n_op_rows": int(op.sum()),
        "n_unmatched_op_rows": int(unmatched_op.size),
        "n_unmatched_op_near_natural_truth": int(near_nat),
        "n_injections_with_multi_candidates": int((n_cand > 1).sum()),
        "substrate": roles.get("roles", {}).get(
            next(iter(roles.get("roles", {})), None), {}).get("substrate"),
        "per_anchor": per_anchor,
    }
    path = a.out or os.path.join(a.arm, "pairs.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_anchor", "dlacats")}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
