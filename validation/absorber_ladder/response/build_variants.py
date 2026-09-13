#!/usr/bin/env python
"""build_variants.py — fit, cross-validate and deliver the nested family of
FIXED response calibration objects R0 / R1a / R1b / R1c (/ R1d) for the first
absorber-side ladder (PI ruling 2026-09-13b §4, §5, §15).

VALIDATION-ONLY.  No sampler is run; nothing under ``CDDF_analysis/`` is
modified.  Every variant is a fixed calibration object of the production form
(mu/sig/skew moment polynomials + covariate clamp range) that is frozen BEFORE
any HBI run, fit on the 2LPT-0 natural-pair matched calibration events ONLY,
with model complexity chosen by 2-fold cross-validation over sightlines
(TARGETID parity).  No mock-closure number is read, computed or used anywhere
in this program.

Products (per variant):
  * ``response_<variant>.npz`` — the coefficient object (adopted_response
    schema + the variant's skew ramp and provenance);
  * ``Mg_<variant>_<fam>.npz`` — the (S=8, Kf=15, C=29, B=16) gathered mass
    tensor for the runner's ``--mg-fixed`` hook, built through the COMMITTED
    ``count_conserving_fold.surface_masses`` under the ratified
    count-conservation rule (unit in-grid mass x the pack's frozen
    ``adopted_phi_ref``), exactly as ``cc_posterior_validation.build_cc_tensors``
    does;
  * ``variants_report.json`` — every CV/held-out number in the report.

ENV: gpdla.  Usage:
  python validation/absorber_ladder/response/build_variants.py \
      --events .../calib_events_2lpt0.npz --out-dir .../response
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import respfit as R                                             # noqa: E402
import opmetrics as M                                           # noqa: E402
import r1d_empirical as R1D                                     # noqa: E402

FAMILIES = ("2lpt0", "london0", "saclay0")
SCANPACK_DIR = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
                "packs")
ADOPTED_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/stage0/adopted_response_v1p1.npz")
EMP_OPS_DIR = os.path.join(_REPO, "validation", "absorber_diag")


# ---------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git():
    try:
        return dict(commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip(),
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=_REPO).decode().strip())
    except Exception as e:                                   # pragma: no cover
        return dict(commit="unknown", error=str(e))


def scanpack(fam):
    return os.path.join(SCANPACK_DIR, f"scanpack_{fam}_b300.npz")


# ---------------------------------------------------------------------------
# variant specifications — the nested ladder
# ---------------------------------------------------------------------------
def specs(marg_deg=4, r1c_estimator="sample"):
    s = {}
    s["R0"] = R.VariantSpec(
        name="R0", edges="r0_fixed", lo=19.0, hi=21.4, step=0.1, min_n=50,
        estimator="ml", deg_cell=2, deg_shared=3, fit_rng="data",
        ramp=(21.0, 0.5), marginalise=None,
        defect="(baseline: the frozen adopted_response v1.1 estimator)")
    s["R1a"] = R.VariantSpec(
        name="R1a", edges="adaptive", lo=19.0, hi=22.4, step=0.1, min_n=50,
        hard_min_n=25, max_width=0.3,
        estimator="ml", deg_cell=2, deg_shared=3, fit_rng="full",
        ramp=(21.0, 0.5), marginalise=None,
        defect=("covariate clamp: adopted_resp_fit_range[...,1]=21.35 freezes "
                "every moment polynomial for the 4 latent bins above it "
                "(forensics 9.3); mean +0.03..+0.08 dex too high at b>=21.3"))
    s["R1b"] = R.VariantSpec(
        name="R1b", edges="adaptive", lo=19.0, hi=22.4, step=0.1, min_n=50,
        hard_min_n=25, max_width=0.3,
        estimator="ml", deg_cell=2, deg_shared=3, fit_rng="full",
        ramp=None, marginalise=None,
        defect=("resp_skew_ramp=(21.0,0.5) forces the modelled skew to "
                "EXACTLY 0 for every latent bin above 21.5 while the measured "
                "skew there is +0.3..+1.3 (forensics 9.4)"))
    s["R1c"] = R.VariantSpec(
        name="R1c", edges="adaptive", lo=19.0, hi=22.4, step=0.1, min_n=50,
        hard_min_n=25, max_width=0.3,
        estimator=r1c_estimator, deg_cell=2, deg_shared=3, fit_rng="full",
        ramp=None,
        marginalise=dict(n_quad=17, deg=marg_deg, weight="flat"),
        defect=("kernel rows 8-25%% too narrow at high S/N (forensics 9.1): "
                "surface_masses evaluates the response at the SINGLE point "
                "N=centre(b), so the row misses Var_{N in b}[N+mu(N)] "
                "(midpoint-quadrature error), and the ML sub-bin estimator is "
                "a further ~2-3%% narrow vs the second moment"))
    return s


# ---------------------------------------------------------------------------
def fit_one(ev, mask, spec, N_ref, ntrue_edges):
    sp = dict(spec)
    return R.fit_variant(ev["N_true"][mask], ev["dx"][mask],
                         ev["isr"][mask], ev["izr"][mask], N_ref, sp,
                         ntrue_edges=ntrue_edges)


def fit_obj(ev, mask, spec, N_ref, geom):
    """Fit one variant on ``mask`` and return (object, (SR,ZR,C,B) masses).

    R0-R1c are moment-polynomial objects evaluated by the COMMITTED
    ``surface_masses``; R1d carries its mass tensor directly.
    """
    if spec.get("representation") == "empirical":
        obj = R1D.fit_r1d(ev["N_true"][mask], ev["xhat"][mask],
                          ev["isr"][mask], ev["izr"][mask], ev["b_i"][mask],
                          geom["ntrue"], geom["nhat"], N_ref,
                          deg=spec["deg"], lam=spec["ridge_lambda"],
                          J=spec["J"])
        obj["rng"] = np.zeros((3, 3, 2))
        obj["rng_data"] = np.zeros((3, 3, 2))
        obj["shared"] = np.zeros((3, 1))
        obj["ramp"] = R.NO_RAMP
        obj["spec"] = dict(spec)
        return obj, obj["masses"]
    obj = fit_one(ev, mask, spec, N_ref, geom["ntrue"])
    masses, _ = M.model_masses(obj, geom["ntrue"], geom["nhat"],
                               sig_floor=geom["sig_floor"])
    return obj, masses


def cv_variant(ev, spec, N_ref, geom, min_events=40):
    """2-fold CV over sightlines (TARGETID parity): fit on one half, score the
    held-out half.  Returns (records, per-event loglik, objects)."""
    A = ev["fold"]
    recs = []
    ll_tot, ll_n = 0.0, 0
    objs = {}
    for tag, fitm, evm in (("A", A, ~A), ("B", ~A, A)):
        obj, masses = fit_obj(ev, fitm, spec, N_ref, geom)
        objs[tag] = obj
        emp = M.empirical_rows(ev["xhat"][evm], ev["b_i"][evm],
                               ev["s_i"][evm], ev["K_i"][evm], geom["nhat"],
                               geom["B"], geom["S"], geom["KK"])
        recs += M.compare_rows(masses, emp, geom["s2sr"], geom["K2zr"],
                               geom["ntrue"], geom["nhat"],
                               min_events=min_events)
        if "surf" in obj:
            ll, n = R.per_event_loglik(
                obj["surf"], obj["rng"], N_ref, ev["N_true"][evm],
                ev["dx"][evm], ev["isr"][evm], ev["izr"][evm],
                truncated=True, skew_ramp=obj["ramp"])
            ll_tot += ll; ll_n += n
    return recs, (ll_tot / max(ll_n, 1)), objs


def insample_vs_empirical(masses, geom, emp_ops_path, min_events=40):
    """In-sample comparison against the forensics' empirical M_true operator
    (normalised over c), on the SAME (s, K, c, b) grid."""
    z = np.load(emp_ops_path, allow_pickle=True)
    Mt = np.asarray(z["M_true_sKcb"], float)                # (S, KK, C, B)
    Nm = np.asarray(z["N_match_cksb"], float)               # (C, Kf, S, B)
    kz = np.asarray(geom["kz2K"], int)
    nK = np.zeros((geom["S"], geom["KK"], geom["B"]))
    for k in range(geom["Kf"]):
        nK[:, kz[k], :] += Nm[:, k, :, :].sum(axis=0)      # (S, B)
    ne = np.asarray(geom["nhat"], float)
    cen = 0.5 * (ne[:-1] + ne[1:])
    nt = np.asarray(geom["ntrue"], float)
    recs = []
    for b in range(geom["B"]):
        lo, hi = nt[b], nt[b + 1]
        for s in range(geom["S"]):
            for k in range(geom["KK"]):
                n = float(nK[s, k, b])
                if n < min_events or Mt[s, k, :, b].sum() <= 0:
                    continue
                em = M.row_stats(Mt[s, k, :, b], cen, lo, hi)
                mm = M.row_stats(masses[geom["s2sr"][s], geom["K2zr"][k], :, b],
                                 cen, lo, hi)
                recs.append(dict(
                    b=b, s=s, k=k, n=n, b_lo=float(lo), b_hi=float(hi),
                    d_mean=mm[0] - em[0],
                    r_sd=(mm[1] / em[1] - 1.0) if em[1] > 0 else np.nan,
                    d_skew=mm[2] - em[2], d_down=mm[3] - em[3],
                    d_in=mm[4] - em[4], d_up=mm[5] - em[5]))
    return recs


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--min-events", type=int, default=40)
    ap.add_argument("--marg-deg-grid", default="3,4,5")
    ap.add_argument("--skip-r1d", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    fams = [f for f in a.families.split(",") if f]

    ev = R.load_events(a.events)
    ev["isr"], ev["izr"] = R.cell_index(ev["snr"], ev["zqso"])
    ev["fold"] = R.parity_fold(ev["tid"])
    print(f"[var] events={len(ev['dx'])} foldA={int(ev['fold'].sum())} "
          f"foldB={int((~ev['fold']).sum())} "
          f"uniq_tid={len(np.unique(ev['tid']))}")

    ad = np.load(ADOPTED_NPZ, allow_pickle=True)
    N_ref = float(ad["N_ref"])

    pk = np.load(scanpack(fams[0]), allow_pickle=True)
    geom = dict(
        ntrue=np.asarray(pk["ntrue_edges"], float),
        nhat=np.asarray(pk["nhat_edges"], float),
        snr=np.asarray(pk["snr_edges"], float),
        zc=np.asarray(pk["zc_edges"], float),
        sig_floor=float(pk["resp_sig_floor"]))
    rse = np.asarray(pk["resp_snr_edges"], float)
    rze = np.asarray(pk["resp_z_edges"], float)
    geom["s2sr"] = np.clip(np.searchsorted(rse, geom["snr"][:-1] + 1e-9,
                                           "right") - 1, 0, 2)
    geom["K2zr"] = np.searchsorted(rze, 0.5 * (geom["zc"][:-1] + geom["zc"][1:]),
                                   "right") - 1
    geom["kz2K"] = np.asarray(pk["kz_to_K"], int)
    geom["B"] = len(geom["ntrue"]) - 1
    geom["C"] = len(geom["nhat"]) - 1
    geom["S"] = len(geom["snr"]) - 1
    geom["Kf"] = len(geom["kz2K"])
    geom["KK"] = int(geom["kz2K"].max()) + 1
    print(f"[var] geometry B={geom['B']} C={geom['C']} S={geom['S']} "
          f"Kf={geom['Kf']} KK={geom['KK']} s2sr={geom['s2sr']} "
          f"K2zr={geom['K2zr']}")

    # pack-side event indices (the strata the metrics are resolved on)
    ev["b_i"] = np.clip(np.digitize(ev["N_true"], geom["ntrue"]) - 1, 0,
                        geom["B"] - 1)
    ev["s_i"] = np.clip(np.digitize(ev["snr"], geom["snr"]) - 1, 0,
                        geom["S"] - 1)
    ev["K_i"] = np.clip(np.digitize(ev["zdla"], geom["zc"]) - 1, 0,
                        geom["KK"] - 1)

    report = dict(
        schema="absorber_ladder/response_variants/v1",
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        host=platform.node(), git=_git(),
        events=dict(path=a.events, sha256=_sha256(a.events),
                    n=int(len(ev["dx"])),
                    n_uniq_tid=int(len(np.unique(ev["tid"]))),
                    provenance=json.loads(ev["provenance"])),
        adopted=dict(path=ADOPTED_NPZ, sha256=_sha256(ADOPTED_NPZ),
                     N_ref=N_ref),
        cv_protocol=("2-fold over SIGHTLINES: TARGETID parity; fit on one "
                     "half, score the held-out half; both directions pooled"))

    # ---- GATE 1: R0 reproduces the frozen coefficients bit-for-bit --------
    r0_full = fit_one(ev, np.ones(len(ev["dx"]), bool), specs()["R0"], N_ref,
                      geom["ntrue"])
    gate = {}
    for k, key in (("mu", "mu_coef"), ("sig", "sig_coef"),
                   ("skew", "skew_coef")):
        gate[key] = float(np.max(np.abs(r0_full["surf"][k]
                                        - np.asarray(ad[key], float))))
    gate["fit_rng"] = float(np.max(np.abs(r0_full["rng_data"]
                                          - np.asarray(ad["fit_rng"], float))))
    gate["BITWISE"] = bool(max(gate.values()) == 0.0)
    print(f"[gate1] R0 refit vs frozen adopted_response_v1p1: {gate}")
    if not gate["BITWISE"]:
        raise SystemExit("GATE 1 FAILED — the rebuilt calibration events do "
                         "not reproduce the frozen response of record; "
                         "refusing to build variants.")
    report["gate_R0_coefficient_reproduction"] = gate

    # ---- GATE 2: R0 bin masses reproduce the pack's own adopted masses -----
    pk0 = np.load(scanpack(fams[0]), allow_pickle=True)
    ccf = M._load_ccf()
    shim = M._Shim(geom["ntrue"], float(pk0["resp_N_ref"]),
                   np.asarray(pk0["resp_skew_ramp"], float),
                   float(pk0["resp_sig_floor"]))
    ref_masses, ref_phi = ccf.surface_masses(
        shim, np.asarray(pk0["adopted_resp_mu_coef"], float),
        np.asarray(pk0["adopted_resp_sig_coef"], float),
        np.asarray(pk0["adopted_resp_skew_coef"], float),
        np.asarray(pk0["adopted_resp_fit_range"], float), geom["nhat"])
    r0_masses, r0_phi = M.model_masses(r0_full, geom["ntrue"], geom["nhat"],
                                       sig_floor=geom["sig_floor"])
    g2 = dict(max_abs_mass_difference=float(np.max(np.abs(r0_masses
                                                          - ref_masses))),
              max_abs_phi_difference=float(np.max(np.abs(r0_phi - ref_phi))),
              phi_ref_stored_max_diff=float(np.max(np.abs(
                  ref_phi - np.asarray(pk0["adopted_phi_ref"], float)))))
    g2["BITWISE"] = bool(g2["max_abs_mass_difference"] == 0.0)
    print(f"[gate2] R0 bin masses vs the pack's deployed kernel: {g2}")
    if not g2["BITWISE"]:
        raise SystemExit("GATE 2 FAILED — R0 does not reproduce the deployed "
                         "bin masses bit-for-bit.")
    report["gate_R0_bin_mass_reproduction"] = g2

    # ---- GATE 3: adopted_phi_ref is the DEPLOYED kernel's phi -------------
    shim_d = M._Shim(geom["ntrue"], float(pk0["resp_N_ref"]),
                     np.asarray(pk0["resp_skew_ramp"], float),
                     float(pk0["resp_sig_floor"]))
    _, phi_dep = ccf.surface_masses(
        shim_d, np.asarray(pk0["resp_mu_coef"], float),
        np.asarray(pk0["resp_sig_coef"], float),
        np.asarray(pk0["resp_skew_coef"], float),
        np.asarray(pk0["resp_N_fit_range"], float), geom["nhat"])
    g3 = dict(
        max_abs_diff_vs_stored_phi_ref=float(np.max(np.abs(
            phi_dep - np.asarray(pk0["adopted_phi_ref"], float)))),
        max_abs_diff_adopted_kernel_phi_vs_stored=(
            g2["phi_ref_stored_max_diff"]),
        note=("adopted_phi_ref is frozen from the DEPLOYED (resp_*) kernel, "
              "not from the adopted surfaces -- the ratified rule, because "
              "C_molly was calibrated jointly with the deployed kernel; every "
              "variant therefore inherits the SAME counting probability and "
              "can only redistribute detections"))
    g3["PASS"] = bool(g3["max_abs_diff_vs_stored_phi_ref"] < 1e-9)
    print(f"[gate3] adopted_phi_ref vs deployed-kernel phi: {g3}")
    if not g3["PASS"]:
        raise SystemExit("GATE 3 FAILED — adopted_phi_ref is not the deployed "
                         "kernel's in-grid fraction; the count-conservation "
                         "arithmetic cannot be reproduced.")
    report["gate_phi_ref_provenance"] = g3

    # ---- CV model selection for R1c --------------------------------------
    sel = {}
    for est in ("ml", "sample"):
        for dg in [int(x) for x in a.marg_deg_grid.split(",")]:
            sp = specs(marg_deg=dg, r1c_estimator=est)["R1c"]
            recs, ll, _ = cv_variant(ev, sp, N_ref, geom, a.min_events)
            agg = M.aggregate(recs, geom["ntrue"], geom["snr"])
            score = dict(
                estimator=est, marg_deg=dg, per_event_loglik=ll,
                held_out_abs_wmean_r_sd=abs(agg["overall"]["r_sd"]["wmean"]),
                held_out_rms_r_sd=float(np.sqrt(np.mean(
                    [r["r_sd"] ** 2 for r in recs if np.isfinite(r["r_sd"])]))),
                held_out_rms_d_mean=float(np.sqrt(np.mean(
                    [r["d_mean"] ** 2 for r in recs]))),
                held_out_rms_d_up=float(np.sqrt(np.mean(
                    [r["d_up"] ** 2 for r in recs]))))
            sel[f"{est}_deg{dg}"] = score
            print(f"[cv] R1c {est} deg{dg}: rms(sd resid)="
                  f"{score['held_out_rms_r_sd']:.4f} "
                  f"rms(mean resid)={score['held_out_rms_d_mean']:.4f} "
                  f"rms(up resid)={score['held_out_rms_d_up']:.4f} "
                  f"loglik/event={ll:.4f}", flush=True)
    # SELECTION RULE (causal and stated in the report: the defect R1c exists
    # to repair is a width BIAS, not width scatter -- the per-cell RMS is
    # dominated by held-out counting noise and barely discriminates -- so the
    # primary score is the held-out count-weighted MEAN width residual, with
    # the held-out RMS mean residual as the tie-break.  No closure quantity
    # enters.
    best = min(sel.values(),
               key=lambda d: (round(d["held_out_abs_wmean_r_sd"], 4),
                              d["held_out_rms_d_mean"]))
    print(f"[cv] R1c selected: {best['estimator']} deg{best['marg_deg']}")
    report["R1c_model_selection"] = dict(
        grid=sel, chosen=dict(estimator=best["estimator"],
                              marg_deg=best["marg_deg"]),
        rule=("held-out |count-weighted MEAN width residual| (primary, "
              "rounded to 1e-4), held-out RMS mean residual (tie-break); "
              "NO closure quantity enters"))

    SP = specs(marg_deg=best["marg_deg"], r1c_estimator=best["estimator"])
    order = ["R0", "R1a", "R1b", "R1c"]

    # ---- R1d: opened ONLY on a diagnosed R1c residual ---------------------
    r1c_recs, _, _ = cv_variant(ev, SP["R1c"], N_ref, geom, a.min_events)
    r1c_agg = M.aggregate(r1c_recs, geom["ntrue"], geom["snr"])
    resid = dict(
        skew_overall=r1c_agg["overall"]["d_skew"]["wmean"],
        skew_low_boundary=r1c_agg["boundary_low_b_lt_19p7"]["d_skew"]["wmean"],
        in_high_boundary=r1c_agg["boundary_high_b_ge_21p3"]["d_in"]["wmean"],
        down_high_boundary=(
            r1c_agg["boundary_high_b_ge_21p3"]["d_down"]["wmean"]))
    open_r1d = (abs(resid["skew_overall"]) > 0.05
                or abs(resid["in_high_boundary"]) > 0.05)
    report["R1d_opening_test"] = dict(
        rule=("R1d is opened ONLY if R1c leaves a diagnosed residual the "
              "3-moment skew-normal family cannot carry: |held-out row-skew "
              "residual| > 0.05 overall, or |in-bin fraction residual| > 0.05 "
              "at b >= 21.3"),
        r1c_residual=resid, opened=bool(open_r1d))
    print(f"[r1d] opening test: {resid} -> opened={open_r1d}")
    if open_r1d and not a.skip_r1d:
        grid = {}
        for dg in (1, 2):
            for lam in (0.0, 1e2, 1e3, 1e4):
                sp = dict(name="R1d", representation="empirical", deg=dg,
                          ridge_lambda=lam, J=15,
                          defect=("R1c residual: the skew-normal 3-moment "
                                  "family cannot carry a narrow core plus a "
                                  "broad tail simultaneously (row skew and "
                                  "the down/in split at b >= 21.3)"))
                recs, _, _ = cv_variant(ev, sp, N_ref, geom, a.min_events)
                ag = M.aggregate(recs, geom["ntrue"], geom["snr"])
                grid[f"deg{dg}_lam{lam:g}"] = dict(
                    deg=dg, ridge_lambda=lam,
                    held_out_abs_wmean_r_sd=abs(ag["overall"]["r_sd"]["wmean"]),
                    held_out_rms_r_sd=float(np.sqrt(np.mean(
                        [r["r_sd"] ** 2 for r in recs
                         if np.isfinite(r["r_sd"])]))),
                    held_out_rms_d_mean=float(np.sqrt(np.mean(
                        [r["d_mean"] ** 2 for r in recs]))),
                    held_out_abs_wmean_d_skew=abs(
                        ag["overall"]["d_skew"]["wmean"]),
                    held_out_rms_d_in=float(np.sqrt(np.mean(
                        [r["d_in"] ** 2 for r in recs]))))
                print(f"[cv] R1d deg{dg} lam{lam:g}: "
                      f"|wmean sd|={grid[f'deg{dg}_lam{lam:g}']['held_out_abs_wmean_r_sd']:.4f} "
                      f"rms_in={grid[f'deg{dg}_lam{lam:g}']['held_out_rms_d_in']:.4f} "
                      f"|wmean skew|={grid[f'deg{dg}_lam{lam:g}']['held_out_abs_wmean_d_skew']:.4f}",
                      flush=True)
        bd = min(grid.values(), key=lambda d: (d["held_out_rms_d_in"],
                                               d["held_out_rms_d_mean"]))
        report["R1d_model_selection"] = dict(
            grid=grid, chosen=dict(deg=bd["deg"],
                                   ridge_lambda=bd["ridge_lambda"]),
            rule=("held-out RMS in-bin-fraction residual (primary — the "
                  "residual R1d was opened on), held-out RMS mean residual "
                  "(tie-break); NO closure quantity enters"))
        SP["R1d"] = dict(name="R1d", representation="empirical",
                         deg=bd["deg"], ridge_lambda=bd["ridge_lambda"], J=15,
                         defect=("R1c residual: the skew-normal 3-moment "
                                 "family cannot carry a narrow core plus a "
                                 "broad tail simultaneously (row skew and the "
                                 "down/in split at b >= 21.3)"))
        order.append("R1d")

    variants = {}
    for name in order:
        sp = SP[name]
        recs, ll, _ = cv_variant(ev, sp, N_ref, geom, a.min_events)
        agg = M.aggregate(recs, geom["ntrue"], geom["snr"])
        full, full_masses = fit_obj(ev, np.ones(len(ev["dx"]), bool), sp,
                                    N_ref, geom)
        ins_recs = M.compare_rows(
            full_masses,
            M.empirical_rows(ev["xhat"], ev["b_i"], ev["s_i"], ev["K_i"],
                             geom["nhat"], geom["B"], geom["S"], geom["KK"]),
            geom["s2sr"], geom["K2zr"], geom["ntrue"], geom["nhat"],
            min_events=a.min_events)
        ins_agg = M.aggregate(ins_recs, geom["ntrue"], geom["snr"])
        emp_cmp = {}
        for fam in fams:
            p = os.path.join(EMP_OPS_DIR, f"empirical_ops_{fam}.npz")
            if os.path.exists(p):
                er = insample_vs_empirical(full_masses, geom, p,
                                           a.min_events)
                emp_cmp[fam] = M.aggregate(er, geom["ntrue"], geom["snr"])
        rms = {k: float(np.sqrt(np.mean([r[k] ** 2 for r in recs
                                         if np.isfinite(r[k])])))
               for k in M.KEYS}
        variants[name] = dict(obj=full, masses=full_masses, spec=dict(sp),
                              held_out_rms=rms, cv_records=recs,
                              cv=agg, insample=ins_agg,
                              per_event_loglik=ll, vs_empirical_Mtrue=emp_cmp)
        print(f"[var] {name}: n_coef={full['n_coef']} "
              f"heldout wmean d_mean={agg['overall']['d_mean']['wmean']:+.4f} "
              f"r_sd={agg['overall']['r_sd']['wmean']:+.4f} "
              f"d_up={agg['overall']['d_up']['wmean']:+.4f}", flush=True)

    # ---- write the products ----------------------------------------------
    prod = {}
    for name in order:
        v = variants[name]
        obj = v["obj"]
        cpath = os.path.join(out, f"response_{name}.npz")
        if "surf" in obj:
            coef_kw = dict(
                mu_coef=obj["surf"]["mu"], sig_coef=obj["surf"]["sig"],
                skew_coef=obj["surf"]["skew"],
                fit_rng=np.asarray(obj["rng"]),
                fit_rng_data=np.asarray(obj["rng_data"]),
                shared=np.asarray(obj["shared"]))
        else:
            coef_kw = dict(p_offset=obj["p_offset"], raw_counts=obj["raw"],
                           row_totals=obj["tot"], c0=obj["c0"],
                           J=np.array(obj["J"]), deg=np.array(obj["deg"]),
                           ridge_lambda=np.array(obj["lam"]))
        np.savez_compressed(
            cpath, **coef_kw,
            N_ref=np.array(obj["N_ref"]),
            skew_ramp=np.asarray(obj["ramp"], float),
            provenance=np.array(json.dumps(dict(
                schema="absorber_ladder/response_variant/v1",
                variant=name, spec=_jsonable(v["spec"]),
                n_coef=obj["n_coef"],
                fixed_before_hbi=True,
                calibration=("2LPT-0 natural-pair matched events "
                             f"(n={len(ev['dx'])}), tp_natpair_tilthost_op/v1"),
                git=_git()))))
        prod[os.path.basename(cpath)] = cpath
        for fam in fams:
            pkf = np.load(scanpack(fam), allow_pickle=True)
            mg, extra = build_mg(obj, pkf, geom,
                                 masses=(None if "surf" in obj
                                         else obj["masses"]))
            mpath = os.path.join(out, f"Mg_{name}_{fam}.npz")
            np.savez_compressed(
                mpath, Mg=mg, phi_ref_used=extra["phi_ref"],
                variant=np.array(name),
                coefficients=np.array(json.dumps(
                    dict(mu_coef=obj["surf"]["mu"].tolist(),
                         sig_coef=obj["surf"]["sig"].tolist(),
                         skew_coef=obj["surf"]["skew"].tolist(),
                         fit_rng=np.asarray(obj["rng"]).tolist(),
                         N_ref=obj["N_ref"],
                         skew_ramp=list(map(float, obj["ramp"])))
                    if "surf" in obj else
                    dict(representation="smoothed empirical masses",
                         p_offset_shape=list(obj["p_offset"].shape),
                         J=obj["J"], deg=obj["deg"],
                         ridge_lambda=obj["lam"],
                         N_ref=obj["N_ref"],
                         product_file=f"response_{name}.npz"))),
                provenance=np.array(json.dumps(dict(
                    schema="absorber_ladder/Mg_fixed/v1",
                    variant=name, family=fam, pack=scanpack(fam),
                    pack_sha256=_sha256(scanpack(fam)),
                    shape=list(mg.shape),
                    consumer=("validation/fp_ladder/run_ladder.py --fix M "
                              "(Mg_fixed hook), shape (S,Kf,C,B)"),
                    count_conservation=("masses / phi * adopted_phi_ref, the "
                                        "ratified rule; identical arithmetic "
                                        "to cc_posterior_validation."
                                        "build_cc_tensors"),
                    builder=("count_conserving_fold.surface_masses "
                             "(COMMITTED)" if "surf" in obj else
                             "r1d_empirical.fit_r1d (non-parametric rows; "
                             "surface_masses does not apply)"),
                    n_coef=obj["n_coef"], git=_git()))), )
            prod[os.path.basename(mpath)] = mpath

    report["variants"] = {
        n: dict(spec=_jsonable(variants[n]["spec"]),
                n_coef=variants[n]["obj"]["n_coef"],
                fit_range=np.asarray(variants[n]["obj"]["rng"]).tolist(),
                fit_range_data=np.asarray(
                    variants[n]["obj"]["rng_data"]).tolist(),
                skew_ramp=list(map(float, variants[n]["obj"]["ramp"])),
                marginal_resid=variants[n]["obj"].get("marginal_resid"),
                per_event_loglik=variants[n]["per_event_loglik"],
                held_out_rms=variants[n]["held_out_rms"],
                held_out=variants[n]["cv"], in_sample=variants[n]["insample"],
                vs_empirical_Mtrue=variants[n]["vs_empirical_Mtrue"])
        for n in order}

    rpath = os.path.join(out, "variants_report.json")
    with open(rpath, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    prod[os.path.basename(rpath)] = rpath
    # per-record CV tables (for the figures)
    for n in order:
        p = os.path.join(out, f"cv_records_{n}.npz")
        recs = variants[n]["cv_records"]
        np.savez_compressed(p, **{k: np.array([r[k] for r in recs], float)
                                  for k in recs[0]})
        prod[os.path.basename(p)] = p

    with open(os.path.join(out, "SHA256SUMS"), "w") as fh:
        for k in sorted(prod):
            fh.write(f"{_sha256(prod[k])}  {k}\n")
    print(f"[var] wrote {len(prod)} products to {out}")
    return report


def _jsonable(d):
    o = {}
    for k, v in d.items():
        if isinstance(v, (np.ndarray,)):
            o[k] = v.tolist()
        elif isinstance(v, dict):
            o[k] = _jsonable(v)
        elif isinstance(v, tuple):
            o[k] = list(v)
        else:
            o[k] = v
    return o


def build_mg(obj, pkf, geom, masses=None):
    """(S, Kf, C, B) tensor under the ratified count-conservation rule."""
    ne = np.asarray(pkf["nhat_edges"], float)
    nt = np.asarray(pkf["ntrue_edges"], float)
    if not (np.array_equal(ne, geom["nhat"]) and
            np.array_equal(nt, geom["ntrue"])):
        raise SystemExit("pack geometry differs from the reference pack")
    if masses is None:
        masses, phi = M.model_masses(obj, nt, ne,
                                     sig_floor=float(pkf["resp_sig_floor"]))
    else:
        masses = np.asarray(masses, float)
        phi = masses.sum(axis=2)
    phi_ref = np.asarray(pkf["adopted_phi_ref"], float)
    masses = masses / np.maximum(phi, 1e-12)[:, :, None, :] \
        * phi_ref[:, :, None, :]
    rse = np.asarray(pkf["resp_snr_edges"], float)
    rze = np.asarray(pkf["resp_z_edges"], float)
    snr = np.asarray(pkf["snr_edges"], float)
    zc = np.asarray(pkf["zc_edges"], float)
    s2sr = np.clip(np.searchsorted(rse, snr[:-1] + 1e-9, "right") - 1, 0,
                   masses.shape[0] - 1)
    K2zr = np.searchsorted(rze, 0.5 * (zc[:-1] + zc[1:]), "right") - 1
    kz2K = np.asarray(pkf["kz_to_K"], int)
    mg = M.gather_Mg(masses, s2sr, kz2K, K2zr)
    return mg, dict(phi_ref=phi_ref)


if __name__ == "__main__":
    main()
