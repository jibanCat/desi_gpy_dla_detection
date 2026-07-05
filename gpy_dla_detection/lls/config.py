"""config.py — the PRODUCTION single-absorber LLS finder config, reproduced in-submodule.

Every knob here is transcribed from the config files that generated the LoA + 2LPT LLS/CDDF
results:
  * scientific base : slurm/configs/_base.env
  * LLS flavour     : slurm/configs/2lpt0_y3_lls172.env  (2LPT) / loa_y3_lls172.env (LoA)
  * param contract  : desi-DLAGP.py::main  params_dict + model_params (lines ~573-636)

So `build_lls_holder()` reconstructs the EXACT production DLAHolder (single-absorber, NHI-floor
17.2 PW samples), and `build_lls_holder(break_aware=True)` makes the ONE change under test — the
absorber model becomes break-aware (DLAGPMATLymanBreak) and the window is extended blueward so
the 912 A break is in-window. Referee discipline: reproduce production first, then diff one knob.

Retired: the 3-way null/subDLA/DLA model — production is SINGLE_ABSORBER_MODEL=1 (2-way).
"""
from __future__ import annotations
import os

from ..set_parameters import Parameters

# --- scientific base: slurm/configs/_base.env ---------------------------------
PROD_PARAMS = dict(
    loading_min_lambda=910.0, loading_max_lambda=1550.0,
    normalization_min_lambda=1425.0, normalization_max_lambda=1475.0,
    min_lambda=911.75, max_lambda=1216.75, dlambda=0.15, k=30,
    max_noise_variance=9.0, max_z_cut=3000.0, min_z_cut=3000.0,
    num_forest_lines=3, num_lines=3,
)
# --- LLS flavour overrides: slurm/configs/2lpt0_y3_lls172.env -----------------
PROD_MODEL = dict(
    single_absorber_model=True,   # 2-way null vs one absorber (SINGLE_ABSORBER_MODEL=1)
    max_dlas=1,                   # MAX_DLAS=1
    filter_low_likelihood=False,  # FILTER_LOW_LIKELIHOOD=0
    prev_tau_0=0.00246, prev_beta=3.62,   # Turner+2024 mean-flux prior
    min_z_separation=3000.0,
    num_dla_samples=50000,        # matches pw_samples_a3_172_220_50000.mat row count
    num_subdla_samples=100000,    # subdla samples (UNUSED under single-absorber, set for parity)
)
# --- data files, relative to the data root ------------------------------------
REL_FILES = dict(
    learned_file="learnlogs/model_epoch_920.h5",
    catalog_name="data/dr12q/processed/catalog.mat",
    los_catalog="data/dla_catalogs/dr9q_concordance/processed/los_catalog",
    dla_catalog="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    # single-absorber samples: PW alpha=3, NHI [17.2, 22.0], 50k
    dla_samples_file="data/dr12q/processed/pw_samples_a3_172_220_50000.mat",
    sub_dla_samples_file="data/dr12q/processed/subdla_samples_a03_191_200_100000.mat",
)
DEFAULT_DATA_ROOT = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"
# model_epoch_920 rest grid floor is 850.90; extend the LLS window to 851 (safe margin) so the
# 912 A break of a z_abs~z_qso absorber is inside the modelled range (Tier-1, no retrain).
LLS_WINDOW_MIN = 851.0


def _params(num_dla_samples: int, break_aware: bool, lls_window_min: float) -> Parameters:
    pd = dict(PROD_PARAMS)
    pd["num_dla_samples"] = num_dla_samples
    if break_aware:
        pd["min_lambda"] = float(lls_window_min)
        pd["loading_min_lambda"] = min(pd["loading_min_lambda"], float(lls_window_min))
    return Parameters(**pd)


def build_lls_holder(
    data_root: str = DEFAULT_DATA_ROOT,
    break_aware: bool = False,
    lls_window_min: float = LLS_WINDOW_MIN,
    learned_file: str | None = None,
    max_workers: int = 1,
    batch_size: int = 100,
    figure_dir: str = "/tmp/lls_figs",
):
    """Reconstruct the production single-absorber LLS DLAHolder.

    break_aware=False -> exact production finder (line-only DLAGPMAT, window 911.75).
    break_aware=True  -> LLSHolder (absorber model = DLAGPMATLymanBreak) with the window
                         extended to lls_window_min so the 912 A break is in-window.
    """
    params = _params(PROD_MODEL["num_dla_samples"], break_aware, lls_window_min)
    params_subdla = _params(PROD_MODEL["num_subdla_samples"], break_aware, lls_window_min)
    files = {k: os.path.join(data_root, v) for k, v in REL_FILES.items()}
    if learned_file is not None:
        files["learned_file"] = learned_file

    if break_aware:
        from .holder import LLSHolder as Holder
    else:
        from run_bayes_select import DLAHolder as Holder

    return Holder(
        learned_file=files["learned_file"],
        catalog_name=files["catalog_name"],
        los_catalog=files["los_catalog"],
        dla_catalog=files["dla_catalog"],
        dla_samples_file=files["dla_samples_file"],
        sub_dla_samples_file=files["sub_dla_samples_file"],
        params=params,
        params_subdla=params_subdla,
        min_z_separation=PROD_MODEL["min_z_separation"],
        prev_tau_0=PROD_MODEL["prev_tau_0"],
        prev_beta=PROD_MODEL["prev_beta"],
        max_dlas=PROD_MODEL["max_dlas"],
        single_absorber_model=PROD_MODEL["single_absorber_model"],
        filter_low_likelihood=PROD_MODEL["filter_low_likelihood"],
        max_workers=max_workers,
        batch_size=batch_size,
        figure_dir=figure_dir,
    )
