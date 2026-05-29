"""Convert phase2_desi .pt checkpoints to NullGPMAT-compatible h5.

Grid: linspace(850.75, 850.75 + (n_pix-1)*0.15, n_pix). For n_pix=5662 this
is [850.75, 1699.90] with dλ=0.15 — same dλ as production but wider rest
range. Validated by matching mu peaks to known emission lines (Lyα, NV,
SiIV, CIV, HeII).
"""
import os
import sys
import numpy as np
import h5py
import torch

PT_FILES = {
    "2lpt_loa0_wide": "/global/cfs/cdirs/desicollab/users/jibancat/DLA/learned/phase2_desi/2lpt_loa0_wide/checkpoints/phase2_desi_checkpoint_final_iter1499.pt",
    "2lpt_loa124_nohcd_nobal_wide": "/global/cfs/cdirs/desicollab/users/jibancat/DLA/learned/phase2_desi/2lpt_loa124_nohcd_nobal_wide/checkpoints/phase2_desi_checkpoint_final_iter1499.pt",
}

OUT_DIR = "/pscratch/sd/j/jibancat/prod533_5k_20260511/null_gp_test/converted"
os.makedirs(OUT_DIR, exist_ok=True)

REST_MIN = 850.75
REST_DLAMBDA = 0.15
MAX_NOISE_VARIANCE = 9.0
NORMALIZATION_MIN_LAMBDA = 1310.0
NORMALIZATION_MAX_LAMBDA = 1325.0


def _np(t):
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


for tag, pt_path in PT_FILES.items():
    print(f"\n=== converting {tag} ===")
    print(f"  source: {pt_path}")
    ck = torch.load(pt_path, weights_only=False, map_location="cpu")
    M = _np(ck["M"]).astype(np.float64)
    log_omega = _np(ck["log_omega"]).astype(np.float64)
    log_c_0 = float(_np(ck["log_c_0"]))
    log_tau_0 = float(_np(ck["log_tau_0"]))
    log_beta = float(_np(ck["log_beta"]))
    mu = _np(ck["mu"]).astype(np.float64)

    n_pix = mu.shape[0]
    rest_wavelengths = np.linspace(REST_MIN, REST_MIN + (n_pix - 1) * REST_DLAMBDA, n_pix)
    print(f"  n_pix={n_pix}, rest_wl=[{rest_wavelengths[0]:.2f}, {rest_wavelengths[-1]:.2f}]")
    print(f"  log_c_0={log_c_0:.4f}  log_tau_0={log_tau_0:.4f}  log_beta={log_beta:.4f}")
    print(f"  M.shape={M.shape}  log_omega.shape={log_omega.shape}  mu.shape={mu.shape}")
    print(f"  mu peak idx={int(np.argmax(mu))} → λ={rest_wavelengths[int(np.argmax(mu))]:.3f} (Lyα expected ≈ 1215.67)")

    out_h5 = os.path.join(OUT_DIR, f"{tag}.h5")
    with h5py.File(out_h5, "w") as f:
        # v2-style scalar datasets (consistent with how NullGPMAT detects DESI format)
        f.create_dataset("M",                          data=M.astype(np.float32))
        f.create_dataset("mu",                         data=mu.astype(np.float32))
        f.create_dataset("log_omega",                  data=log_omega.astype(np.float32))
        f.create_dataset("log_c_0",                    data=np.float64(log_c_0))
        f.create_dataset("log_tau_0",                  data=np.float64(log_tau_0))
        f.create_dataset("log_beta",                   data=np.float64(log_beta))
        f.create_dataset("rest_wavelengths",           data=rest_wavelengths.astype(np.float32))
        f.create_dataset("max_noise_variance",         data=np.float64(MAX_NOISE_VARIANCE))
        f.create_dataset("normalization_min_lambda",   data=np.float64(NORMALIZATION_MIN_LAMBDA))
        f.create_dataset("normalization_max_lambda",   data=np.float64(NORMALIZATION_MAX_LAMBDA))
    print(f"  wrote: {out_h5}  ({os.path.getsize(out_h5)/1e6:.2f} MB)")

print("\n=== validating load ===")
import sys
sys.path.insert(0, "/pscratch/sd/j/jibancat/desi_gpy_dla_detection")
from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.null_gp import NullGPMAT

REPO = "/pscratch/sd/j/jibancat/desi_gpy_dla_detection"
PRIOR_CAT = f"{REPO}/data/dr12q/processed/catalog.mat"
LOS_CAT = f"{REPO}/data/dla_catalogs/dr9q_concordance/processed/los_catalog"
DLA_CAT = f"{REPO}/data/dla_catalogs/dr9q_concordance/processed/dla_catalog"

params = Parameters(max_lambda=1216.75, dlambda=0.15, k=30, num_lines=3, num_forest_lines=3)
prior = PriorCatalog(params, PRIOR_CAT, LOS_CAT, DLA_CAT)

for tag in PT_FILES:
    h5_path = os.path.join(OUT_DIR, f"{tag}.h5")
    print(f"\nLoading {h5_path} ...")
    try:
        null_gp = NullGPMAT(params, prior, h5_path, prev_tau_0=0.00246, prev_beta=3.62)
        print(f"  ✓ loaded ({type(null_gp).__name__})")
        print(f"  mu shape = {null_gp.mu.shape}, rest_wavelengths range = [{null_gp.rest_wavelengths[0]:.2f}, {null_gp.rest_wavelengths[-1]:.2f}]")
        print(f"  log_c_0={null_gp.log_c_0:.4f} log_tau_0={null_gp.log_tau_0:.4f} log_beta={null_gp.log_beta:.4f}")
    except Exception as e:
        import traceback
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
print("\nDONE")
