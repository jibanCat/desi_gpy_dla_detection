#!/usr/bin/env python
"""extract_calib_events.py — dump the 2LPT-0 natural-pair matched CALIBRATION
events that ``znz_kernel.fit_forward_response`` regresses, together with the
per-detection TARGETID (the CV split unit).

VALIDATION-ONLY.  Nothing under ``CDDF_analysis/`` is modified.  The loader
block is a VERBATIM copy of ``CDDF_analysis/hbi_mcmc/build_kernel_fit_ensemble.py``
(the producer of ``adopted_response_v1p1.npz``), so the event set is the SAME
population the frozen response was fit on — the raw pair table is not on disk
(``forward_response_2lpt0.npz`` stores only coefficients + the empirical
density), so it is rebuilt here through the committed machinery and gated on
the event count.

ENV: gpdla (numpy-only; znz_kernel is jax-free).

Usage:
    python validation/absorber_ladder/response/extract_calib_events.py \
        --out /scratch/.../response/calib_events_2lpt0.npz
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FROZEN_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
              "track_c/stage0/forward_response_2lpt0.npz")


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_REPO).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(frozen_npz=FROZEN_NPZ):
    from CDDF_analysis.hbi.znz_kernel import (load_forward_response,
                                              measure_forward_response)
    import CDDF_analysis.hbi.track_c_tf_loa as TF
    import CDDF_analysis.hbi.ab_loa0_fp_baseline as AB
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        HBIConfig, load_molly_matrix, load_and_cut_catalog, _build_qso_lookup)

    frm_point = load_forward_response(frozen_npz)

    class _A:
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
    # the absorber redshift is carried too (stratification axis only; the
    # frozen model's cell axis is Z_QSO)
    zdla = np.asarray(cat_cut["Z_DLA"], float)[op][tp][keep]
    return meas, det_tids, zdla, dict(
        snr_min=float(cfg.snr_min), p_dla_min=float(cfg.p_dla_min),
        host_col=host_col, xhat_floor=float(xhat_floor),
        z_covariate=str(meas["z_covariate"]),
        snr_edges=np.asarray(frm_point.snr_edges, float).tolist(),
        z_edges=np.asarray(frm_point.z_edges, float).tolist(),
        N_ref_frozen=float(frm_point.N_ref),
        deg_N_frozen=int(frm_point.deg_N),
        catalog_dir=str(TF._C0_CAT), truth_path=str(TF._C0_TRUTH))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen-npz", default=FROZEN_NPZ)
    ap.add_argument("--expect-events", type=int, default=0,
                    help="fail-closed gate on the event count (0 = report only)")
    a = ap.parse_args(argv)

    meas, tids, zdla, prov = extract(a.frozen_npz)
    n = len(meas["dx"])
    print(f"[calib] n_events={n}  n_uniq_tids={len(np.unique(tids))}")
    if a.expect_events and n != a.expect_events:
        raise SystemExit(f"EVENT-COUNT GATE FAILED: {n} != {a.expect_events}")
    prov.update(n_events=int(n), n_uniq_tids=int(len(np.unique(tids))),
                code_commit=_git_head(),
                frozen_npz=a.frozen_npz,
                frozen_npz_sha256=_sha256(a.frozen_npz),
                schema="absorber_ladder/calib_events/v1")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out,
        N_true=np.asarray(meas["N_true"], float),
        snr=np.asarray(meas["snr"], float),
        zqso=np.asarray(meas["zqso"], float),
        zdla=np.asarray(zdla, float),
        dx=np.asarray(meas["dx"], float),
        xhat=np.asarray(meas["xhat"], float),
        tid=np.asarray(tids, np.int64),
        provenance=np.array(json.dumps(prov)))
    print(f"[calib] wrote {a.out}")


if __name__ == "__main__":
    main()
