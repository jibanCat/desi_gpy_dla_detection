# -*- coding: utf-8 -*-
"""run_rung9.py — Model A on a REAL mock pack (validation-ladder rung 9+).

Loads a stamped Model A pack, runs the NUTS posterior, writes a compact result
JSON (reductions, closure vs pack truth, diagnostics, provenance). MOCK ONLY —
refuses any pack whose provenance mentions the real survey.

Usage (SLURM or interactive smoke):
    conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.run_rung9 \
        --pack /scratch/.../modelA_pack_2lpt0.npz --out <json> \
        [--warmup 1000 --samples 1000 --chains 4 --seed 0] [--smoke]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.model_a import ModelAConfig, run_model_a


def _git():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=here, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--",
             os.path.join(here, "run_rung9.py"), os.path.join(here, "model_a.py"),
             os.path.join(here, "forward.py"), os.path.join(here, "pack.py")],
            cwd=here, text=True).strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sampler settings (100/100/2) for a wiring check")
    ap.add_argument("--allow-low-farr", metavar="REASON", default=None,
                    help="override the fail-closed Farr N_eff gate with a stamped "
                         "justification (required for on-mock SELF-calibration, "
                         "where the calibration set is the same mock as the data "
                         "so the 4x headroom is unattainable; the finite-"
                         "calibration variance is instead SAMPLED via psi_C)")
    a = ap.parse_args()

    assert "main_dark" not in a.pack, "REAL-LOA guard: mock packs only"
    pack = load_pack(a.pack)
    prov = pack.provenance or {}
    assert "loa_main_dark" not in json.dumps(prov), "REAL-LOA guard (provenance)"

    if a.smoke:
        a.warmup, a.samples, a.chains = 100, 100, 2

    # Capture HEAD at PROCESS START, not at write time. A multi-hour run whose
    # repo advances mid-flight would otherwise stamp a commit it never used --
    # this mis-stamped the 2026-07-11 broken-kernel ablation with a commit that
    # contains the very kernel fix that run predates.
    code_commit = _git()

    cfg = ModelAConfig(num_warmup=a.warmup, num_samples=a.samples,
                       num_chains=a.chains, seed=a.seed,
                       target_accept=a.target_accept,
                       enforce_farr_gate=(a.allow_low_farr is None))
    t0 = time.time()
    mcmc, red = run_model_a(pack, cfg)
    wall = time.time() - t0

    # closure vs pack truth (truth-space): posterior expected counts per tier
    # vs truth_counts, on the coarse report tiers.
    out = dict(
        pack=os.path.basename(a.pack),
        sampler=dict(warmup=a.warmup, samples=a.samples, chains=a.chains,
                     seed=a.seed, wallclock_s=wall),
        reductions={k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in red.items() if k != "diagnostics"},
        diagnostics=red.get("diagnostics"),
        provenance=dict(
            routine="CDDF_analysis/hbi_mcmc/run_rung9.py",
            code_commit=code_commit,
            pack_provenance_commit=prov.get("code_commit"),
            farr_gate_override=a.allow_low_farr,
            # A bypass must be visible AS a bypass downstream, not merely as a
            # free-text reason field a reader has to notice: every prepared
            # rung-9/10 sbatch passes --allow-low-farr, so without this every
            # such artifact was indistinguishable from a clean one.
            bypasses=({"allow_low_farr": a.allow_low_farr}
                      if a.allow_low_farr is not None else {}),
            paper_facing=False if a.allow_low_farr is not None else None,
            paper_facing_note=(
                "a run with the Farr headroom gate bypassed can never be "
                "paper-facing" if a.allow_low_farr is not None else
                "rung 9 is a VALIDATION rung; paper-facing status is decided "
                "by run_posterior / run_evidence, not here"),
            date=time.strftime("%Y-%m-%d"),
            rederive=(f"conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc."
                      f"run_rung9 --pack {a.pack} --out <out> --warmup {a.warmup} "
                      f"--samples {a.samples} --chains {a.chains} --seed {a.seed}"),
        ),
    )

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=_default)
    d = red.get("diagnostics") or {}
    print(f"[rung9] wrote {a.out}  wall={wall:.0f}s  "
          f"r_hat_max={d.get('r_hat_max')}  ess_bulk_min={d.get('ess_bulk_min')}  "
          f"divergences={d.get('n_divergent')}  policy_pass={d.get('policy_pass')}")


if __name__ == "__main__":
    main()
