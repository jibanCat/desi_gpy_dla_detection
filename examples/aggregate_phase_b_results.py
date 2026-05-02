"""Aggregate ALL Phase B Phase B results across mocks + LOA into a
single summary CSV + per-mock JSON for embedding in the story docs.

Intended to be re-run as new validations land. Reads the chunk_*.tsv
output of each Phase B array, plus the τ_factor from each array's
SLURM logs (the runner doesn't write τ_factor into the chunk TSV;
this script reconstructs it from log-line ordering).

Output:
  tests/profile/results/phase_b_aggregate.csv  — one row per dataset
  tests/profile/results/phase_b_tau_factor_per_mock.csv  — τ histogram
"""
from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROFILE_DIR = Path(__file__).resolve().parent.parent / "tests" / "profile" / "results"
LOG_DIR = Path(__file__).resolve().parent.parent / "slurm" / "greatlakes"
DATA_BASE = "/nfs/turbo/lsa-cavestru/mfho/DESI/pscratch/desi_gpy_dla_detection"


# (label, slurm-array-job-id, output-dir, base-or-real)
DATASETS = [
    ("2lpt_5k_F0M3", 49040725, f"{DATA_BASE}/phase_b_49040725", "mock"),
    ("2lpt_5k_F0M3_grid6", 49062626, f"{DATA_BASE}/phase_b_2lpt", "mock"),
    ("london_5k_F0M3", 49062627, f"{DATA_BASE}/phase_b_london", "mock"),
    ("saclay_5k_F0M3", 49062628, f"{DATA_BASE}/phase_b_saclay", "mock"),
    ("2lpt_5k_F1M4_nobal", 49063779, f"{DATA_BASE}/phase_b_2lpt_filter1_max4_nobal", "mock"),
    ("2lpt_50k_F1M4_nobal", 49065622, f"{DATA_BASE}/phase_b_2lpt_50k_filter1_max4", "mock"),
    ("london_50k_F1M4_nobal", 49071204, f"{DATA_BASE}/phase_b_london_50k_filter1_max4", "mock"),
    ("saclay_50k_F1M4_nobal", 49071205, f"{DATA_BASE}/phase_b_saclay_50k_filter1_max4", "mock"),
    ("loa_5k_F1M4", 49071304, f"{DATA_BASE}/phase_b_loa_5k_filter1_max4", "real"),
]


def _aggregate_chunks(base: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(base, "chunk_*.tsv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)


def _tau_factor_from_logs(jid: int) -> Dict[int, float]:
    """Reconstruct the τ_factor chosen for each TID from the array logs.
    The runner prints `Processing spectrum (ID: <tid>)` followed by
    `τ-EB[...] factor_best=X.XX` when enable_tau_eb=True.
    """
    pat_tid = re.compile(r"Processing spectrum \d+/\d+ \(ID: (\d+)\)")
    pat_tau = re.compile(r"factor_best=(\d+\.\d+)")
    out = {}
    for f in sorted(glob.glob(str(LOG_DIR / f"phase_b_{jid}_*.log"))):
        cur = None
        with open(f) as fh:
            for line in fh:
                mt = pat_tid.search(line)
                if mt:
                    cur = int(mt.group(1))
                    continue
                mf = pat_tau.search(line)
                if mf and cur is not None:
                    # Only ENABLED prints τ-EB lines; for spectra processed
                    # twice (BASELINE then ENABLED), the τ-EB line precedes
                    # the second "Processing spectrum" line — but the cur
                    # TID was already set on the FIRST "Processing" line.
                    # Each TID gets one log line so this is correct.
                    out[cur] = float(mf.group(1))
    return out


def _bias_at_cut(df: pd.DataFrame, p_cut: float):
    dla_truth = df[df.nhi_regime == "DLA"]
    both = dla_truth[(dla_truth.baseline_p_dla >= p_cut)
                     & (dla_truth.enabled_p_dla >= p_cut)]
    if len(both) == 0:
        return None
    b = both.baseline_map_log_nhi - both.truth_log_nhi
    e = both.enabled_map_log_nhi - both.truth_log_nhi
    n_none = (df.nhi_regime == "none").sum()
    n_b_fp = ((df.nhi_regime == "none") & (df.baseline_p_dla >= p_cut)).sum()
    n_e_fp = ((df.nhi_regime == "none") & (df.enabled_p_dla >= p_cut)).sum()
    n_dla = len(dla_truth)
    n_b_dla = (dla_truth.baseline_p_dla >= p_cut).sum()
    n_e_dla = (dla_truth.enabled_p_dla >= p_cut).sum()
    return dict(
        p_cut=p_cut, n_both_detect=len(both),
        b_med=float(b.median()), b_mean=float(b.mean()),
        b_rms=float(np.sqrt((b**2).mean())), b_std=float(b.std()),
        e_med=float(e.median()), e_mean=float(e.mean()),
        e_rms=float(np.sqrt((e**2).mean())), e_std=float(e.std()),
        b_wilcoxon_p=float(wilcoxon(b).pvalue) if len(b) > 0 else float("nan"),
        e_wilcoxon_p=float(wilcoxon(e).pvalue) if len(e) > 0 else float("nan"),
        n_dla_truth=n_dla, b_dla_det=int(n_b_dla), e_dla_det=int(n_e_dla),
        b_compl=float(n_b_dla / n_dla * 100) if n_dla > 0 else float("nan"),
        e_compl=float(n_e_dla / n_dla * 100) if n_dla > 0 else float("nan"),
        n_no_truth=int(n_none), b_fp=int(n_b_fp), e_fp=int(n_e_fp),
        b_fpr=float(n_b_fp / n_none * 100) if n_none > 0 else float("nan"),
        e_fpr=float(n_e_fp / n_none * 100) if n_none > 0 else float("nan"),
    )


def main():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    tau_rows = []
    for label, jid, base, kind in DATASETS:
        df = _aggregate_chunks(base)
        if df is None:
            print(f"  ! {label}: no chunks at {base}")
            continue
        df = df[df.status == "ok"].copy()
        n = len(df)

        # τ_factor distribution from logs
        tau_map = _tau_factor_from_logs(jid)
        df["tau_factor"] = df["target_id"].map(tau_map)
        tau_arr = df["tau_factor"].dropna().to_numpy()
        if len(tau_arr) > 0:
            for tau in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
                tau_rows.append(dict(label=label, kind=kind, tau_factor=tau,
                                     count=int((tau_arr == tau).sum()),
                                     frac=float((tau_arr == tau).mean())))

        # Bias + detection per cut (only mocks have meaningful bias —
        # LOA has no truth so b_med etc. will be NaN)
        if kind == "mock":
            for cut in [0.5, 0.9, 0.97, 0.99]:
                stats = _bias_at_cut(df, cut)
                if stats is None:
                    continue
                row = dict(label=label, jid=jid, kind=kind, n=n)
                row.update(stats)
                row["tau_median"] = float(np.median(tau_arr)) if len(tau_arr) > 0 else float("nan")
                row["tau_mean"] = float(np.mean(tau_arr)) if len(tau_arr) > 0 else float("nan")
                row["tau_frac_ge_2"] = float((tau_arr >= 2.0).mean()) if len(tau_arr) > 0 else float("nan")
                rows.append(row)
        else:
            # LOA — store τ stats + p_DLA distribution only, no bias
            for cut in [0.5, 0.9, 0.97, 0.99]:
                row = dict(label=label, jid=jid, kind=kind, n=n, p_cut=cut,
                           b_dla_det=int((df.baseline_p_dla >= cut).sum()),
                           e_dla_det=int((df.enabled_p_dla >= cut).sum()),
                           tau_median=float(np.median(tau_arr)) if len(tau_arr) > 0 else float("nan"),
                           tau_mean=float(np.mean(tau_arr)) if len(tau_arr) > 0 else float("nan"),
                           tau_frac_ge_2=float((tau_arr >= 2.0).mean()) if len(tau_arr) > 0 else float("nan"))
                rows.append(row)

    out_csv = PROFILE_DIR / "phase_b_aggregate.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[wrote] {out_csv}  ({len(rows)} rows)")
    out_tau = PROFILE_DIR / "phase_b_tau_factor_per_mock.csv"
    pd.DataFrame(tau_rows).to_csv(out_tau, index=False)
    print(f"[wrote] {out_tau}  ({len(tau_rows)} rows)")


if __name__ == "__main__":
    main()
