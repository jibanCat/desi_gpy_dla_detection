#!/usr/bin/env python
"""PI-ruled diagnostics, stage 1b (env gpdla): one catalog load, full event
arrays for D1-D4. Saves per-detection (op-cut applied, NO tp/floor cut yet):
xhat, snr, zqso, zdla, tid, truth columns (tilt host + plain true if present).
"""
import os, sys
import numpy as np

REPO = "/home/mfho/wt_forward_2026_08"
sys.path.insert(0, REPO)
os.chdir(REPO)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "events_full.npz")

from CDDF_analysis.hbi.znz_kernel import load_forward_response
import CDDF_analysis.hbi.track_c_tf_loa as TF
import CDDF_analysis.hbi.ab_loa0_fp_baseline as AB
from CDDF_analysis.hbi.cddf_catalog_hbi import (
    HBIConfig, load_molly_matrix, load_and_cut_catalog, _build_qso_lookup)

FROZEN_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
              "track_c/stage0/forward_response_2lpt0.npz")
frm = load_forward_response(FROZEN_NPZ)

class _A:
    molly_tsv = None
molly_tsv = AB._resolve_molly(_A)
cfg = HBIConfig(catalog_dir=TF._C0_CAT, truth_path=TF._C0_TRUTH,
                bal_cat_path=TF._C0_BAL, molly_tsv=molly_tsv,
                out_dir="/tmp", mockdir=os.path.dirname(TF._C0_TRUTH),
                fp_estimator="purity_mixture", no_bal=True, lam_rf_min=1025.0)
mm = load_molly_matrix(molly_tsv)
qso_lookup = _build_qso_lookup(cfg)
cat_cut, truth_cut, is_TP, good_mask, meta = load_and_cut_catalog(
    cfg, truth_nhi_floor=float(mm.nhi_edges[0]), qso_lookup=qso_lookup,
    host_truth_floor=19.0)
print("columns:", cat_cut.colnames)

s2n = np.asarray(cat_cut["S2N_RED"], float)
pdla = np.asarray(cat_cut["P_DLA"], float)
op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
print("op-cut detections:", int(op.sum()))

def col(name, dtype=float):
    if name in cat_cut.colnames:
        return np.asarray(cat_cut[name], dtype)[op]
    print(f"  [absent] {name}")
    return None

save = dict(
    xhat=col("NHI"),
    snr=s2n[op],
    zqso=col("Z_QSO"),
    zdla=col("Z_DLA"),
    tid=col("TARGETID", np.int64),
    nhi_tilt_host=col("NHI_TILT_HOST"),
    nhi_true=col("NHI_TRUE"),
    snr_edges=np.asarray(frm.snr_edges, float),
    z_edges=np.asarray(frm.z_edges, float),
    N_ref=np.array(float(frm.N_ref)),
    point_mu=np.asarray(frm.mu_coef, float),
    point_sig=np.asarray(frm.sig_coef, float),
    point_skew=np.asarray(frm.skew_coef, float),
)
save = {k: v for k, v in save.items() if v is not None}
np.savez_compressed(OUT, **save)
print("wrote", OUT, "keys:", sorted(save))
