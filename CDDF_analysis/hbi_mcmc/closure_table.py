"""closure_table.py — the post-review zero-sampling closure table (Phase B).

Thin CLI over the frozen statistical machinery. Produces, per mock, the three
diagnostic layers of ``docs/PHASEB_STATS_SPEC_2026-08-06.md``:

  A. CONDITIONAL implementation diagnostics — the legacy
     ``forward_selftest.ratio_tables`` arms (variance = predicted mean;
     historical chi2/dof <= 3 threshold), labeled conditional-only.
  B. CALIBRATION-PREDICTIVE diagnostics — ``gate_covariance.predictive_gate``
     (frozen 3-group observed-N-hat Mahalanobis; simulation-calibrated
     p-value; NO ratified threshold) plus descriptive secondary axes.
  C. TRANSPORT stress — ``gate_covariance.transport_stress_stats``
     (uncalibrated systematic; reported separately, never absorbed).

Also reports before/after the (1 - eta) restoration, component totals, FP
calibration event counts, and support/clamp fractions. Deterministic: all
seeds and ensemble sizes come from the frozen spec constants.

Usage:
    python -m CDDF_analysis.hbi_mcmc.closure_table \
        --packs modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz ... \
        --out closure_table_phaseB.json

MOCKS ONLY. Fails loud on packs without ``fp_eta_c`` (re-extract; the
``--allow-legacy-eta`` escape attaches the committed band table with a logged
note, for before/after comparison only).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np

from CDDF_analysis.hbi_mcmc import gate_covariance as GC
from CDDF_analysis.hbi_mcmc import forward_selftest as FS
from CDDF_analysis.hbi_mcmc import reporting as REP
from CDDF_analysis.hbi_mcmc.pack import (
    load_pack, attach_fp_eta_bands, ModelAPack)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True).strip()
    except Exception:
        return "<unavailable>"


def conditional_layer(pack: ModelAPack) -> dict:
    """Layer A: the legacy conditional gate, unchanged, labeled."""
    res = FS.selftest(pack)
    tab = FS.ratio_tables(res, pack)
    full = REP.window_closure_metrics(tab["by_nhat"], label="full")
    win = REP.window_closure_metrics(tab["by_nhat"], *REP.REPORTING_WINDOW,
                                     label="win")
    out = dict(
        layer=("A (conditional implementation; variance = predicted mean; "
               "historical chi2/dof <= 3 is a CONDITIONAL diagnostic, not a "
               "predictive science gate)"),
        window=dict(win), full=dict(full),
        mu_total=float(np.asarray(res["mu"]).sum()),
        mu_fp_total=float(np.asarray(res["mu_fp"]).sum()),
        mu_sig_total=float(np.asarray(res["mu"] - res["mu_fp"]).sum()),
        obs_total=float(np.asarray(res["counts"]).sum()),
    )
    # per-arm chi2/dof + max|z| for the record (window-restriction stated)
    for arm in ("by_z", "by_snr"):
        rows = tab[arm]
        z = np.asarray([r["z"] for r in rows if r.get("obs", 0) > 0], float)
        out[arm] = dict(chi2_dof=float(np.mean(z ** 2)) if z.size else None,
                        max_abs_z=float(np.max(np.abs(z))) if z.size else None,
                        n_rows=int(z.size),
                        note="FULL grid (not window-restricted)")
    return out


def build_row(path: str, *, allow_legacy_eta: bool) -> dict:
    pack = load_pack(path)
    migrated = False
    if pack.fp_eta_c is None:
        if not allow_legacy_eta:
            raise SystemExit(
                f"{os.path.basename(path)}: pack has no fp_eta_c — re-extract "
                "at the Phase-B tip, or pass --allow-legacy-eta to attach the "
                "committed band table (logged).")
        print(f"[closure_table] {os.path.basename(path)}: legacy pack — "
              "attaching fp_eta_c from the committed band table",
              file=sys.stderr)
        pack = attach_fp_eta_bands(pack)
        migrated = True

    # before/after the (1 - eta) restoration (diagnostic convention only)
    pack_eta_off = dataclasses.replace(
        pack, fp_eta_c=np.zeros(pack.n_c))

    row = dict(
        pack=os.path.basename(path),
        fp_eta_migrated_at_load=migrated,
        fp_calibration_events=int(np.asarray(pack.fp_counts).sum()),
        conditional=conditional_layer(pack),
        conditional_eta_off=conditional_layer(pack_eta_off),
    )
    cov = GC.estimate_covariance(pack)
    gate = GC.predictive_gate(pack, covariance=cov)
    row["predictive"] = gate.report()
    row["transport"] = GC.transport_stress_stats(pack)
    # effective calibration sample size per group + support fractions
    row["support"] = dict(
        live_cells=int(np.broadcast_to(
            (np.asarray(pack.dX) > 0)[None, :, :],
            np.asarray(pack.counts, float).shape).sum()),
        n_pad_bins=int(pack.n_pad_bins),
        fp_events_per_group=list(cov.calibration_events_per_group),
    )
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-legacy-eta", action="store_true")
    a = ap.parse_args(argv)

    rows = [build_row(p, allow_legacy_eta=a.allow_legacy_eta)
            for p in a.packs]
    out = dict(
        spec="docs/PHASEB_STATS_SPEC_2026-08-06.md (frozen)",
        seeds=dict(cov=GC.SEED_COV, null=GC.SEED_NULL),
        ensembles=dict(cov=GC.N_COV_DRAWS, null=GC.N_NULL_DRAWS),
        group_edges=[list(e) for e in GC.PRIMARY_GROUP_EDGES],
        code_commit=_git_commit(),
        rows=rows,
        metadata=dict(rederive=(
            "python -m CDDF_analysis.hbi_mcmc.closure_table --packs "
            + " ".join(a.packs) + " --out " + a.out)),
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[closure_table] wrote {a.out} ({len(rows)} mocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
