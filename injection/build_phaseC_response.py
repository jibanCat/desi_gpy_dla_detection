#!/usr/bin/env python
"""build_phaseC_response.py — Stage-2A bridge evaluation + response artifact.

Consumes the role-scored pair measurements of the production arm
(`measure_phaseC_pairs.py --role bridge` and `--role
production-calibration`; holdout and environment-probe rows are excluded
by the scorer's role enforcement) and executes, IN ORDER (rulings §3.1):

  1. the PRODUCTION BRIDGE STATISTIC against the old envelope, under the
     FROZEN covariance-aware acceptance criterion
     (`docs/PHASEC_BRIDGE_DESIGN.md` §4, incl. the RATIFIED dispersion
     guard) — evaluated BEFORE any artifact is frozen;
  2. on PASS: the degree-2 UNION refit per response cell (old anchors
     re-derived from the stored rho distributions, weight n_eff = 40;
     new anchors weighted by pair count), the per-cell lack-of-fit test
     (p < 0.01 in ANY cell ⇒ STOP for PI — "further response structure"
     is a checkpoint decision), the extended fit ranges, the support-
     label map, and the transition/continuity report;
  3. on FAIL: a QUARANTINED diagnostic artifact — no splice, no
     smoothing, PI review (rulings §3.1 no-go).

Comparison convention (documented design choice): old and new are
compared AT THE NEW ANCHOR N (bin centers), the old side evaluated from
its fitted moment surface at the deployed clamped covariate — i.e. the
comparison tests the response THE DEPLOYED MODEL ACTUALLY USES in the
bridge region. σ_old(mean at N) = max( rms of the cell's old-anchor
means about its own surface, sd_old(N)/√40 ) — conservative. Ĉ_shared is
bounded by the healpix split-half ratio of the new-side means (bridge
doc §3): between/within ratio > 1.5 ⇒ inflate C_bridge by the measured
ratio.

Multi-candidate note (implementation clarification, recorded in the
artifact): the old envelope stores NO per-anchor multi-candidate record,
so the frozen criterion's "difference per cell" is operationalized as
no new-side cell deviating > 3σ (binomial) from the pooled new-side
rate; the raw rates are reported either way.

Skew surfaces: copied from the old envelope unchanged — the frozen skew
ramp (`N_skew_collapse = 21.0`) zeroes skew across the newly measured
region, so no new skew freedom is introduced (and none is measurable at
the per-cell pair counts).

MOCKS ONLY. Usage:
  python injection/build_phaseC_response.py \
      --pairs-bridge <arm>/pairs_bridge.json \
      --pairs-production <arm>/pairs_production.json \
      --out /scratch/.../forward_response_2lpt0_phaseC.npz
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import chi2 as _chi2, norm as _norm, beta as _beta

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

DEFAULT_ENVELOPE = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                    "track_c/stage0/forward_response_2lpt0.npz")
PREIMAGE = os.path.join(_REPO, "diagnostics_phaseC/preimage/preimage.json")
PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz")

#: FROZEN acceptance numbers (PHASEC_BRIDGE_DESIGN §4, amendment ratified)
G3_PROJ_MAX = 75.0
G3_PROJ_CI_UPPER_MAX = 116.7
GLOBAL_Z_MAX = 3.0
LOCAL_Z_MAX = 4.0
COMPLETENESS_Z_MAX = 3.0
LOF_P_MIN = 0.01
N_EFF_OLD = 40.0
SHARED_INFLATE_RATIO = 1.5

BRIDGE_BINS_N = (19.6, 19.8, 20.0, 20.2, 20.4)   # b2–b6 centers
SHARED_TOP_N = 21.0                              # b9 (dual-role)


def _old_surface(env, sr, zr, N, kind="mu"):
    co = env[f"{kind}_coef"][sr, zr]
    lo = float(env["emp_N_anchors"][sr, zr].min())
    hi = float(env["emp_N_anchors"][sr, zr].max())
    Ncl = min(max(N, lo), hi)
    u = Ncl - float(env["N_ref"])
    v = float(co @ np.array([1.0, u, u * u]))
    if kind == "sig":
        v = max(v, float(env["sig_floor"]))
    return v


def _old_anchor_moments(env, sr, zr):
    """(N_a, mean_a, sd_a) per stored old anchor, from the rho histograms."""
    A = np.asarray(env["emp_N_anchors"][sr, zr], float)
    rho = np.asarray(env["emp_rho"][sr, zr], float)      # (7, 121)
    rg = np.asarray(env["emp_r_grid"], float)
    w = rho / np.maximum(rho.sum(axis=1, keepdims=True), 1e-30)
    mean = (w * rg[None, :]).sum(axis=1)
    var = (w * (rg[None, :] - mean[:, None]) ** 2).sum(axis=1)
    return A, mean, np.sqrt(np.maximum(var, 1e-12))


def load_cells(pairs):
    """{(logN, sr, zr): rec} from a pairs JSON."""
    out = {}
    for r in pairs["per_anchor"]:
        out[(round(float(r["logN"]), 2), int(r["resp_cell"][0]),
             int(r["resp_cell"][1]))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-bridge", required=True)
    ap.add_argument("--pairs-production", required=True)
    ap.add_argument("--envelope", default=DEFAULT_ENVELOPE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    env = np.load(a.envelope)
    pre = json.load(open(PREIMAGE))
    tw = pre["mocks"]["2lpt0"]
    S_per_dex = np.abs(np.array(tw["sensitivity"]["dG3_per_bin_mean_shift"])) \
        / 0.02
    ed = np.array(tw["table"]["ntrue_edges"])
    Nc_bins = 0.5 * (ed[:-1] + ed[1:])
    cell_share = np.array(tw["g3_by_response_cell"], float)
    cell_share = cell_share / cell_share.sum()

    pb = json.load(open(a.pairs_bridge))
    pp = json.load(open(a.pairs_production))
    cells_b = load_cells(pb)
    cells_p = load_cells(pp)

    # ---------------- 1. the bridge statistic ----------------
    rows = []
    for (A_n, sr, zr), rec in sorted(cells_b.items()) \
            + [(k, v) for k, v in sorted(cells_p.items())
               if abs(k[0] - SHARED_TOP_N) < 1e-6]:
        n = rec["n_matched_op"]
        if n < 5 or rec["dx_sd"] is None:
            continue
        old_b = _old_surface(env, sr, zr, A_n, "mu")
        old_s = _old_surface(env, sr, zr, A_n, "sig")
        Ao, mo, so = _old_anchor_moments(env, sr, zr)
        surf = np.array([_old_surface(env, sr, zr, x, "mu") for x in Ao])
        rms_about = float(np.sqrt(np.mean((mo - surf) ** 2)))
        sig_old = max(rms_about, old_s / np.sqrt(N_EFF_OLD))
        sig_new = rec["dx_sd"] / np.sqrt(n)
        d = rec["dx_mean"] - old_b
        omega = rec["dx_sd"] / old_s - 1.0
        sig_om = 1.0 / np.sqrt(2.0 * n)     # fractional width sd
        rows.append(dict(logN=A_n, sr=sr, zr=zr, n=n, delta=d,
                         sig_old=sig_old, sig_new=sig_new,
                         sig_d=float(np.hypot(sig_old, sig_new)),
                         omega=omega, sig_omega=sig_om,
                         healpix=rec.get("pair_healpix", []),
                         dx=rec["dx"],
                         n_multi=rec["n_multi_candidate"]))
    if not rows:
        raise SystemExit("no bridge anchor-cells with n >= 5")

    # C_shared bound: healpix split-half on the new side, pooled over rows
    within, between = [], []
    for r in rows:
        hp = np.asarray(r["healpix"])
        dx = np.asarray(r["dx"], float)
        if len(set(hp.tolist())) < 2 or dx.size < 8:
            continue
        us = sorted(set(hp.tolist()))
        g1 = np.isin(hp, us[::2])
        if g1.sum() < 3 or (~g1).sum() < 3:
            continue
        m1, m2 = dx[g1].mean(), dx[~g1].mean()
        nominal = r["dx_sd"] * np.sqrt(1.0 / g1.sum() + 1.0 / (~g1).sum())
        between.append((m1 - m2) / max(nominal, 1e-9))
    shared_ratio = float(np.std(between, ddof=1)) if len(between) >= 4 else 1.0
    inflate = max(1.0, shared_ratio) if shared_ratio > SHARED_INFLATE_RATIO \
        else 1.0
    for r in rows:
        r["sig_d_infl"] = r["sig_d"] * inflate
        r["z"] = r["delta"] / r["sig_d_infl"]
        r["z_omega"] = r["omega"] / (r["sig_omega"] * inflate)

    # criterion 1: G3-projected difference (per-bin pooled over cells,
    # weighted by 1/sig^2 within the bin; S_b from the preimage)
    D = 0.0
    varD = 0.0
    for A_n in sorted({r["logN"] for r in rows}):
        sub = [r for r in rows if r["logN"] == A_n]
        w = np.array([1.0 / r["sig_d_infl"] ** 2 for r in sub])
        dbar = float(np.sum([r["delta"] for r in sub] * w) / w.sum())
        var_dbar = float(1.0 / w.sum())
        b = int(np.argmin(np.abs(Nc_bins - A_n)))
        D += S_per_dex[b] * dbar
        varD += (S_per_dex[b] ** 2) * var_dbar
    sigD = float(np.sqrt(varD))
    ci_upper = abs(D) + 1.96 * sigD
    c1 = (abs(D) < G3_PROJ_MAX) and (ci_upper < G3_PROJ_CI_UPPER_MAX)

    # criterion 2: global coherence (means, widths)
    wz = np.array([1.0 / r["sig_d_infl"] ** 2 for r in rows])
    gz_mean = float(abs(np.sum(np.array([r["delta"] for r in rows]) * wz)
                        / np.sqrt(wz.sum())))
    ww = np.array([1.0 / (r["sig_omega"] * inflate) ** 2 for r in rows])
    gz_w = float(abs(np.sum(np.array([r["omega"] for r in rows]) * ww)
                     / np.sqrt(ww.sum())))
    c2 = (gz_mean < GLOBAL_Z_MAX) and (gz_w < GLOBAL_Z_MAX)

    # criterion 3: no localized break + multi-candidate consistency
    maxz = float(np.max(np.abs([r["z"] for r in rows])))
    n_all = sum(r["n"] for r in rows)
    k_all = sum(r["n_multi"] for r in rows)
    p_pool = k_all / max(n_all, 1)
    mc_z = []
    for r in rows:
        sd = np.sqrt(max(p_pool * (1 - p_pool) / r["n"], 1e-12))
        mc_z.append((r["n_multi"] / r["n"] - p_pool) / sd)
    c3 = (maxz < LOCAL_Z_MAX) and (float(np.max(np.abs(mc_z))) < 3.0)

    # criterion 4: completeness vs the molly surface (overlap cells)
    pkz = np.load(PACK)
    nd, nt = np.asarray(pkz["molly_n_det"], float), \
        np.asarray(pkz["molly_n_tot"], float)
    me = np.asarray(pkz["molly_nhi_edges"], float)
    comp_z = []
    for A_n in sorted({r["logN"] for r in rows}):
        recs = [(k, v) for k, v in cells_b.items() if abs(k[0] - A_n) < 1e-6]
        n_inj = sum(v["n_inj"] for _, v in recs)
        n_det = sum(v["n_matched_op"] for _, v in recs)
        m = int(np.clip(np.digitize([A_n], me)[0] - 1, 0, len(me) - 2))
        pmolly = float((nd[:, m].sum() + 0.5) / (nt[:, m].sum() + 1.0))
        lo = _beta.ppf(0.0013, n_det + 0.5, n_inj - n_det + 0.5)
        hi = _beta.ppf(0.9987, n_det + 0.5, n_inj - n_det + 0.5)
        comp_z.append({"logN": A_n, "C_inj": n_det / max(n_inj, 1),
                       "C_molly": pmolly,
                       "inside_3sig": bool(lo <= pmolly <= hi)})
    c4 = all(c["inside_3sig"] for c in comp_z)

    bridge_pass = bool(c1 and c2 and c3 and c4)
    verdict = {
        "criterion1_G3_projection": {"D_counts": D, "sigma_D": sigD,
                                     "ci_upper": ci_upper,
                                     "max": G3_PROJ_MAX,
                                     "ci_upper_max": G3_PROJ_CI_UPPER_MAX,
                                     "pass": bool(c1)},
        "criterion2_global_coherence": {"z_mean": gz_mean, "z_width": gz_w,
                                        "max": GLOBAL_Z_MAX, "pass": bool(c2)},
        "criterion3_local": {"max_abs_z": maxz, "max": LOCAL_Z_MAX,
                             "multi_candidate_max_z":
                                 float(np.max(np.abs(mc_z))),
                             "pooled_rate": p_pool, "pass": bool(c3)},
        "criterion4_completeness": {"cells": comp_z, "pass": bool(c4)},
        "shared_covariance": {"split_half_ratio": shared_ratio,
                              "inflation_applied": inflate,
                              "n_split_rows": len(between)},
        "n_anchor_cells": len(rows),
        "BRIDGE_PASS": bridge_pass,
    }

    # ---------------- 2/3. artifact (adopted or quarantined) -------------
    new_anchor_N = sorted({round(float(r["logN"]), 2)
                           for r in pp["per_anchor"]})
    B_new = len(new_anchor_N)
    ph_mean = np.full((3, 3, B_new), np.nan)
    ph_sd = np.full((3, 3, B_new), np.nan)
    ph_n = np.zeros((3, 3, B_new), int)
    for (A_n, sr, zr), rec in {**cells_b, **cells_p}.items():
        if rec["dx_mean"] is None:
            continue
        bi = new_anchor_N.index(round(A_n, 2))
        ph_mean[sr, zr, bi] = rec["dx_mean"]
        ph_sd[sr, zr, bi] = rec["dx_sd"] if rec["dx_sd"] else np.nan
        ph_n[sr, zr, bi] = rec["n_matched_op"]

    lof = np.ones((3, 3))
    mu_new = np.array(env["mu_coef"], float)
    sg_new = np.array(env["sig_coef"], float)
    rr_new = np.zeros((3, 3, 2))
    cont = np.zeros((3, 3, 2))
    nref = float(env["N_ref"])
    if bridge_pass:
        for sr in range(3):
            for zr in range(3):
                Ao, mo, so = _old_anchor_moments(env, sr, zr)
                Nn = np.array(new_anchor_N, float)
                ok = np.isfinite(ph_mean[sr, zr])
                xs = np.concatenate([Ao, Nn[ok]])
                ys = np.concatenate([mo, ph_mean[sr, zr][ok]])
                ss = np.concatenate([so, ph_sd[sr, zr][ok]])
                ns = np.concatenate([np.full(len(Ao), N_EFF_OLD),
                                     ph_n[sr, zr][ok]])
                sig_mean = ss / np.sqrt(np.maximum(ns, 1))
                u = xs - nref
                X = np.vstack([np.ones_like(u), u, u * u]).T
                W = 1.0 / sig_mean ** 2
                co, *_ = np.linalg.lstsq(X * W[:, None] ** 0.5,
                                         ys * W ** 0.5, rcond=None)
                mu_new[sr, zr] = co
                resid = (ys - X @ co) / sig_mean
                dof = len(ys) - 3
                lof[sr, zr] = float(_chi2.sf(np.sum(resid ** 2), dof))
                # sd surface refit (weights n; sd of sd = s/sqrt(2n))
                ys2 = np.concatenate([so, ph_sd[sr, zr][ok]])
                sig_s = ys2 / np.sqrt(2.0 * np.maximum(ns, 1))
                co2, *_ = np.linalg.lstsq(X / sig_s[:, None],
                                          ys2 / sig_s, rcond=None)
                sg_new[sr, zr] = co2
                rr_new[sr, zr] = (min(Ao.min(), Nn[ok].min()),
                                  max(Ao.max(), Nn[ok].max()))
                # continuity over the OLD fit range
                grid = np.linspace(Ao.min(), Ao.max(), 40)
                dmu = [abs(float(np.array([1, g - nref, (g - nref) ** 2]) @ co)
                           - _old_surface(env, sr, zr, g, "mu"))
                       for g in grid]
                dsd = [abs(float(np.array([1, g - nref, (g - nref) ** 2]) @ co2)
                           / _old_surface(env, sr, zr, g, "sig") - 1.0)
                       for g in grid]
                cont[sr, zr] = (max(dmu), max(dsd))
        lof_fail = bool((lof < LOF_P_MIN).any())
    else:
        lof_fail = False

    support = {
        "vocabulary": ["directly-measured", "bridge-validated",
                       "interpolated-within-support", "pooled", "transferred",
                       "clamped", "extrapolated", "unsupported",
                       "quarantined"],
        "joint_conditioning": {
            "z_cells": [[0.0, 2.56], [2.56, 2.96], [2.96, 3.5]],
            "snr_cells": [[2.0, 3.5], [3.5, 6.5], [6.5, None]],
            "note": ("labels apply ONLY inside these covered (z, SNR) boxes "
                     "(effective z support [2.15, 3.5]); outside them the "
                     "response remains transferred/extrapolated regardless "
                     "of the true-N coordinate (rulings §7)")},
        "intervals": ([
            {"true_N": [19.34, 19.5], "label": "old-support (unchanged)"},
            {"true_N": [19.5, 20.5], "label": "bridge-validated"},
            {"true_N": [20.5, 22.35], "label": "directly-measured"},
            {"true_N": [22.35, None], "label": "clamped"},
            {"true_N": [None, 19.34], "label": "unsupported (pad region "
                                               "conventions unchanged)"}]
            if bridge_pass and not lof_fail else
            [{"true_N": None, "label": "quarantined"}]),
    }

    status = ("ADOPTED" if bridge_pass and not lof_fail else
              "STOP-LACK-OF-FIT" if bridge_pass else "QUARANTINED")
    out_path = a.out if status == "ADOPTED" else \
        os.path.join(os.path.dirname(a.out),
                     "quarantined_" + os.path.basename(a.out))
    np.savez(
        out_path,
        # deployed-schema keys (adopted: refit; else old copied for record)
        mu_coef=mu_new if status == "ADOPTED" else np.array(env["mu_coef"]),
        sig_coef=sg_new if status == "ADOPTED" else np.array(env["sig_coef"]),
        skew_coef=np.array(env["skew_coef"]),
        N_ref=np.array(env["N_ref"]),
        sig_floor=np.array(env["sig_floor"]),
        N_skew_collapse=np.array(env["N_skew_collapse"]),
        z_edges=np.array(env["z_edges"]),
        snr_edges=np.array(env["snr_edges"]),
        emp_N_anchors=np.array(env["emp_N_anchors"]),
        emp_rho=np.array(env["emp_rho"]),
        emp_r_grid=np.array(env["emp_r_grid"]),
        emp_z_edges=np.array(env["emp_z_edges"]),
        emp_snr_edges=np.array(env["emp_snr_edges"]),
        _fwd_response_kind=np.array(str(env["_fwd_response_kind"])),
        deg_N=np.array(env["deg_N"]),
        # phaseC extensions
        phaseC_status=np.array(status),
        phaseC_anchor_N=np.array(new_anchor_N),
        phaseC_mean=ph_mean, phaseC_sd=ph_sd, phaseC_n=ph_n,
        phaseC_fit_range=rr_new,
        phaseC_lack_of_fit_p=lof,
        phaseC_continuity_max_dmu_dsdfrac=cont,
        phaseC_support_json=np.array(json.dumps(support)),
        phaseC_bridge_json=np.array(json.dumps(verdict)),
    )
    report = {
        "schema": "phaseC_stage2A_bridge/v1",
        "date": time.strftime("%Y-%m-%d"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
        "status": status, "artifact": out_path,
        "bridge": verdict,
        "lack_of_fit_p_per_cell": lof.tolist(),
        "continuity": cont.tolist(),
        "fit_range_new": rr_new.tolist(),
        "anchor_rows": [{k: v for k, v in r.items()
                         if k not in ("dx", "healpix")} for r in rows],
        "support": support,
    }
    rp = a.report or (os.path.splitext(a.out)[0] + "_bridge_report.json")
    with open(rp, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"status": status, "BRIDGE_PASS": bridge_pass,
                      "criteria": {k: v.get("pass") for k, v in
                                   verdict.items() if isinstance(v, dict)
                                   and "pass" in v},
                      "D_counts": D, "sigma_D": sigD,
                      "max_abs_z": maxz,
                      "lack_of_fit_min_p": float(lof.min())}, indent=1))
    print("wrote", out_path, "and", rp)


if __name__ == "__main__":
    main()
