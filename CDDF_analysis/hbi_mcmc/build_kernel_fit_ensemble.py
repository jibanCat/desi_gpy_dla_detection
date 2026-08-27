#!/usr/bin/env python
"""build_kernel_fit_ensemble.py — PI ruling 2026-08-16 item 1, phase 1:
the FULL legitimate kernel-fit covariance, as a coefficient ENSEMBLE from
the COMMITTED Track-C T-D resample-refit machinery.

Per draw, a sightline-level bootstrap multiplicity over the unique TARGETID
basis of the 2LPT-0 forward-fit population re-weights the SAME committed
fit (`znz_kernel.fit_forward_response` via
`refit_forward_response_from_resample`) that produced the frozen
`forward_response_2lpt0.npz`. This carries ALL polynomial orders, all
moments (mu/sig/skew), their within- and cross-moment correlations, and
the sightline dependence unit — with NO re-specification of the fit and
NOTHING tuned to any data under test.

Built-in gate (fail-loud): the unit-weight refit must reproduce the frozen
point surfaces (the committed unit-weight invariance).

Env: gpdla (numpy-only). Output: an NPZ with (n_draws, 3,3,deg+1) coef
ensembles + the point coefficients + the gate record. Phase 2
(kernel_uncertainty_closure --ensemble) folds these through the packs.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FROZEN_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
              "track_c/stage0/forward_response_2lpt0.npz")
# (no default output: an explicit --out is required so that an accidental run can
#  never overwrite the frozen kernel_fit_ensemble_v1.npz of record — pre-push hardening 2026-08-26)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=400)
    ap.add_argument("--out", required=True,
                    help="output npz (REQUIRED; never defaults to the frozen artifact of record)")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--frozen-npz", default=FROZEN_NPZ,
                    help="forward-response point model to resample around (default: the frozen "
                         "forward_response_2lpt0.npz of record; pre-tag review 2026-08-26 adds the option "
                         "for current-contract reconciliation runs into a work dir)")
    a = ap.parse_args()

    from CDDF_analysis.hbi.znz_kernel import (
        load_forward_response, measure_forward_response,
        build_forward_response_fit_resample,
        refit_forward_response_from_resample)
    import CDDF_analysis.hbi.track_c_tf_loa as TF
    import CDDF_analysis.hbi.ab_loa0_fp_baseline as AB
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        HBIConfig, load_molly_matrix, load_and_cut_catalog, _build_qso_lookup)

    frm_point = load_forward_response(a.frozen_npz)

    # the 2LPT-0 calibration bundle, exactly as build_frozen_calibration loads it
    class _A:  # minimal arg shim for _resolve_molly
        molly_tsv = None
    molly_tsv = AB._resolve_molly(_A)
    cfg = HBIConfig(
        catalog_dir=TF._C0_CAT, truth_path=TF._C0_TRUTH,
        bal_cat_path=TF._C0_BAL, molly_tsv=molly_tsv,
        out_dir="/tmp", mockdir=os.path.dirname(TF._C0_TRUTH),
        fp_estimator="purity_mixture", no_bal=True, lam_rf_min=1025.0)
    mm = load_molly_matrix(molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=float(mm.nhi_edges[0]), qso_lookup=qso_lookup,
        host_truth_floor=19.0)

    host_col = "NHI_TILT_HOST"
    xhat_floor = 19.5
    meas = measure_forward_response(
        cat_cut, good_mask, cfg, host_col=host_col, xhat_floor=xhat_floor,
        z_covariate=str(getattr(frm_point, "z_covariate", "zqso")))
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    nhi_op = np.asarray(cat_cut["NHI"], float)[op]
    true_col = host_col if host_col in cat_cut.colnames else "NHI_TRUE"
    xtrue_op = np.asarray(cat_cut[true_col], float)[op]
    tp = np.isfinite(xtrue_op)
    keep = nhi_op[tp] >= xhat_floor
    det_tids = np.asarray(cat_cut["TARGETID"], np.int64)[op][tp][keep]
    assert len(det_tids) == len(meas["dx"]), "det_tids/meas row mismatch"

    uniq_tids = np.unique(det_tids)     # kernel-fit resample basis (sightline unit)
    rfr = build_forward_response_fit_resample(
        meas, det_tids, uniq_tids, frm_point,
        n_N_cells=7, min_count=60, build_empirical=False)

    # GATE: unit-weight refit reproduces the frozen point surfaces
    unit = refit_forward_response_from_resample(rfr, np.ones(len(uniq_tids)))
    gate = {}
    for nm in ("mu_coef", "sig_coef", "skew_coef"):
        d = float(np.max(np.abs(np.asarray(getattr(unit, nm))
                                - np.asarray(getattr(frm_point, nm)))))
        gate[nm] = d
    print(f"[kfe] unit-weight gate max|delta|: {gate}")
    if max(gate.values()) > 1e-8:
        raise SystemExit(f"unit-weight invariance FAILED: {gate} — the "
                         "resample refit does not reproduce the frozen "
                         "point model; refusing to build the ensemble.")

    rng = np.random.default_rng(a.seed)
    n_u = len(uniq_tids)
    mu_e, sig_e, skew_e = [], [], []
    for i in range(a.n_draws):
        bm = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u)).astype(float)
        frm_i = refit_forward_response_from_resample(rfr, bm)
        mu_e.append(np.asarray(frm_i.mu_coef))
        sig_e.append(np.asarray(frm_i.sig_coef))
        skew_e.append(np.asarray(frm_i.skew_coef))
        if (i + 1) % 50 == 0:
            print(f"[kfe] {i+1}/{a.n_draws}", flush=True)

    def _git():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=_REPO).decode().strip()
        except Exception:
            return "unknown"

    np.savez_compressed(
        a.out,
        mu_coef=np.stack(mu_e), sig_coef=np.stack(sig_e),
        skew_coef=np.stack(skew_e),
        point_mu=np.asarray(frm_point.mu_coef),
        point_sig=np.asarray(frm_point.sig_coef),
        point_skew=np.asarray(frm_point.skew_coef),
        n_events=np.array(len(det_tids)), n_uniq_tids=np.array(n_u),
        seed=np.array(a.seed),
        provenance=np.array(json.dumps(dict(
            schema="kernel_fit_ensemble/v1",
            method=("committed Track-C T-D resample refit "
                    "(build_forward_response_fit_resample + "
                    "refit_forward_response_from_resample); sightline "
                    "multinomial bootstrap over the unique forward-"
                    "population TIDs; unit-weight gate PASSED"),
            frozen_npz=a.frozen_npz, unit_gate=gate, code_commit=_git()))))
    print(f"[kfe] wrote {a.out}  (n_events={len(det_tids)}, n_uniq={n_u})")


if __name__ == "__main__":
    main()
