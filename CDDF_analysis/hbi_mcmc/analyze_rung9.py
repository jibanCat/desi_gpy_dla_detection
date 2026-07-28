# -*- coding: utf-8 -*-
"""analyze_rung9.py — closure/diagnostics reader for run_rung9 result JSONs.

Reads a rung-9 (or rung-10 cross-mock) result JSON + its data pack and writes a
compact stamped closure report:

  1. DIAGNOSTICS FIRST: the spec section-5 policy gate (policy_pass, r_hat,
     ESS, divergences) leads the report; closure is still computed on a failed
     gate (report honestly), never silently suppressed.
  2. Tier closure: posterior dN/dX and Omega per coarse-z bin (dX-weighted,
     the identical construction to reduce_f_posterior's *_coarse) vs the pack
     truth_counts-implied values, with posterior quantiles and the truth-count
     Poisson relative error.
  3. CDDF cell closure on the (b, K) coarse grid; [19.5, 19.7) masked out,
     zero-truth cells reported separately (no fake ratios).
  4. t_K posteriors vs their N(0, t_sigma) priors: shrinkage sd/sigma and the
     prior z-score mean/sigma (on the SELF mock t should sit near 0).

MOCK ONLY (requires pack.truth_counts). Omega values stay in the module's
ARBITRARY-CONSTANT units (see reduce_f_posterior) — closure is a ratio, the
physical constant cancels.

Usage:
    conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.analyze_rung9 \
        --result <rung9_*.json> --pack <modelA_pack_*.npz> --out <closure.json>
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.model_a import _THRESHOLDS, _MASK_LO, _MASK_HI

__all__ = ["truth_tier_table", "closure_report", "main"]


def _tag(thr):
    return f"{thr:.1f}".replace(".", "p")


def _grids(pack):
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    dN = np.diff(ntrue)
    dX_k = np.asarray(pack.dX, float).sum(axis=1)          # (Kf,) over strata
    kz = np.asarray(pack.kz_to_K)
    dX_K = np.array([dX_k[kz == K].sum() for K in range(pack.n_kk)])
    return Nc, dN, dX_k, kz, dX_K


def truth_tier_table(pack):
    """truth_counts -> the closure estimand on the report grain.

    Returns dict with, per threshold tag ('20p0', '20p3'):
      n_truth / dndx_truth / omega_truth : (KK,) per coarse-z bin,
    plus f_truth_coarse (B, KK) and n_truth_coarse_cells (B, KK) for the
    per-cell CDDF closure, and dX_coarse (KK,).
    """
    if pack.truth_counts is None:
        raise ValueError("pack has no truth_counts (mock packs only)")
    Nc, dN, dX_k, kz, dX_K = _grids(pack)
    KK = pack.n_kk
    tc = np.asarray(pack.truth_counts, float)

    out = {"dX_coarse": dX_K}
    for thr in _THRESHOLDS:
        sel = Nc >= thr - 1e-9
        n_K = np.array([tc[np.ix_(sel, kz == K)].sum() for K in range(KK)])
        om_K = np.array([
            (tc[np.ix_(sel, kz == K)].sum(axis=1)
             * 10.0 ** (Nc[sel] - 21.0)).sum() for K in range(KK)])
        out[_tag(thr)] = {"n_truth": n_K, "dndx_truth": n_K / dX_K,
                          "omega_truth": om_K / dX_K}
    n_cells = np.stack([tc[:, kz == K].sum(axis=1) for K in range(KK)], axis=1)
    out["n_truth_coarse_cells"] = n_cells                          # (B, KK)
    out["f_truth_coarse"] = n_cells / (dX_K[None, :] * dN[:, None])
    return out


def _coarse_avg(draws_fine, kz, dX_k, KK):
    """(n, Kf) fine-z draws -> (n, KK) pathlength-weighted coarse means
    (the identical construction to reduce_f_posterior's *_coarse)."""
    return np.stack(
        [(draws_fine[:, kz == K] * dX_k[kz == K][None, :]).sum(axis=1)
         / dX_k[kz == K].sum() for K in range(KK)], axis=1)


def _stat_entry(draws, truth, K, stat, n_truth):
    q = np.quantile(draws, [0.025, 0.16, 0.5, 0.84, 0.975])
    mean, sd = float(draws.mean()), float(draws.std(ddof=1))
    return {
        "coarse_z": int(K), "stat": stat,
        "post_mean": mean, "post_sd": sd,
        "q025": float(q[0]), "q16": float(q[1]), "q50": float(q[2]),
        "q84": float(q[3]), "q975": float(q[4]),
        "truth": float(truth),
        "ratio": (mean / truth) if truth > 0 else None,
        "z": ((mean - truth) / sd) if sd > 0 else None,
        "in68": bool(q[1] <= truth <= q[3]),
        "in95": bool(q[0] <= truth <= q[4]),
        "truth_rel_poisson_err": (float(np.sqrt(n_truth) / n_truth)
                                  if n_truth > 0 else None),
    }


def closure_report(result, pack):
    """result (the loaded run_rung9 JSON dict) + pack -> the closure report.

    Diagnostics gate first; then tier closure, CDDF cell closure, t-vs-prior,
    FP passthrough. All values are plain python types (JSON-ready).
    """
    red = result["reductions"]
    diags = dict(result.get("diagnostics") or {})
    Nc, dN, dX_k, kz, dX_K = _grids(pack)
    KK = pack.n_kk
    tab = truth_tier_table(pack)

    # --- 1. diagnostics gate (leads the report; never blocks closure)
    report = {
        "policy_pass": bool(diags.get("policy_pass", False)),
        "diagnostics": diags,
        "farr_ratio": red.get("farr_ratio"),
        "sampler": result.get("sampler"),
        "pack": result.get("pack"),
    }

    # --- 2. tier closure (dN/dX + Omega per coarse z, both thresholds)
    tiers = {}
    for thr in _THRESHOLDS:
        tag = _tag(thr)
        dndx_coarse = np.asarray(red[f"dndx_{tag}_coarse"], float)   # (n, KK)
        omega_coarse = _coarse_avg(
            np.asarray(red[f"omega_{tag}"], float), kz, dX_k, KK)
        entries = []
        for K in range(KK):
            n_truth = float(tab[tag]["n_truth"][K])
            entries.append(_stat_entry(dndx_coarse[:, K],
                                       tab[tag]["dndx_truth"][K],
                                       K, "dndx", n_truth))
            entries.append(_stat_entry(omega_coarse[:, K],
                                       tab[tag]["omega_truth"][K],
                                       K, "omega", n_truth))
        tiers[tag] = entries
    report["tiers"] = tiers

    # --- 3. CDDF cell closure on the (b, K) coarse grid
    f_draws = np.asarray(red["f"], float)                        # (n, B, Kf)
    f_coarse = np.stack(
        [(f_draws[:, :, kz == K] * dX_k[kz == K][None, None, :]).sum(axis=2)
         / dX_k[kz == K].sum() for K in range(KK)], axis=2)      # (n, B, KK)
    mask_b = (Nc >= _MASK_LO - 1e-9) & (Nc < _MASK_HI - 1e-9)
    # UNREPORTED BASIS PAD (schema v1.1, finding D1): true-N bins below the
    # reporting floor are inferred against the constant-extrapolation
    # completeness convention -- a stated systematic, not a measurement. They
    # must not appear in the per-cell CDDF closure table, which is a headline
    # diagnostic. reduce_f_posterior publishes the reported support; consume it
    # rather than re-deriving it. No-op on unpadded packs.
    _rep = red.get("reported_mask")
    if _rep is not None:
        mask_b = mask_b | (~np.asarray(_rep, bool))
    f_truth_c = np.asarray(tab["f_truth_coarse"])
    q = np.quantile(f_coarse, [0.025, 0.16, 0.84, 0.975], axis=0)  # (4, B, KK)
    mean_c = f_coarse.mean(axis=0)
    sd_c = f_coarse.std(axis=0, ddof=1)
    cddf = []
    for K in range(KK):
        use = (~mask_b) & (f_truth_c[:, K] > 0)
        t = f_truth_c[use, K]
        in68 = (q[1, use, K] <= t) & (t <= q[2, use, K])
        in95 = (q[0, use, K] <= t) & (t <= q[3, use, K])
        with np.errstate(divide="ignore", invalid="ignore"):
            zsc = (mean_c[use, K] - t) / sd_c[use, K]
        cddf.append({
            "coarse_z": int(K),
            "n_cells": int(use.sum()),
            "n_cells_zero_truth": int(((~mask_b) & (f_truth_c[:, K] <= 0)).sum()),
            "frac_in68": float(in68.mean()) if use.any() else None,
            "frac_in95": float(in95.mean()) if use.any() else None,
            "ratio_median": (float(np.median(mean_c[use, K] / t))
                             if use.any() else None),
            "max_abs_z": (float(np.nanmax(np.abs(zsc))) if use.any() else None),
        })
    report["cddf"] = cddf
    report["n_cddf_cells_masked_out"] = int(mask_b.sum()) * KK

    # --- 4. t_K posterior vs its N(0, t_sigma) prior
    t_mean = np.asarray(red["t_mean"], float)
    t_sd = np.asarray(red["t_sd"], float)
    t_sigma = np.asarray(pack.t_sigma, float)
    report["t"] = [
        {"K": int(K), "mean": float(t_mean[K]), "sd": float(t_sd[K]),
         "prior_sigma": float(t_sigma[K]),
         "shrinkage": float(t_sd[K] / t_sigma[K]),
         "z0": float(t_mean[K] / t_sigma[K])}
        for K in range(KK)]

    # --- 5. FP passthrough (joint mode only)
    if "fp_lam_total_mean" in red:
        report["fp"] = {"lam_total_mean": red["fp_lam_total_mean"],
                        "lam_total_sd": red.get("fp_lam_total_sd")}
    else:
        report["fp"] = None
    return report


def _git():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=here, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--",
             os.path.join(here, "analyze_rung9.py"),
             os.path.join(here, "model_a.py"), os.path.join(here, "pack.py")],
            cwd=here, text=True).strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="run_rung9 output JSON")
    ap.add_argument("--pack", required=True, help="the matching data pack npz")
    ap.add_argument("--out", required=True, help="closure report JSON")
    ap.add_argument("--allow-nonstandard-grid", action="store_true")
    a = ap.parse_args(argv)

    with open(a.result) as fh:
        result = json.load(fh)
    pack = load_pack(a.pack, allow_nonstandard_grid=a.allow_nonstandard_grid)

    report = closure_report(result, pack)
    report["provenance"] = {
        "routine": "CDDF_analysis/hbi_mcmc/analyze_rung9.py",
        "code_commit": _git(),
        "result_file": os.path.basename(a.result),
        "result_provenance": result.get("provenance"),
        "pack_file": os.path.basename(a.pack),
        "pack_provenance_commit": (pack.provenance or {}).get("code_commit"),
        "date": time.strftime("%Y-%m-%d"),
        "rederive": (f"conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc."
                     f"analyze_rung9 --result {a.result} --pack {a.pack} "
                     f"--out <out>"),
    }

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1, default=_default)

    d = report["diagnostics"]
    dndx = [e for e in report["tiers"]["20p3"] if e["stat"] == "dndx"]
    print(f"[analyze_rung9] wrote {a.out}")
    print(f"  GATE policy_pass={report['policy_pass']} "
          f"flags={d.get('flags_fired')} rhat_max={d.get('r_hat_max')} "
          f"ess_bulk_min={d.get('ess_bulk_min')} "
          f"divergences={d.get('n_divergent')}")
    print("  dN/dX(>=20.3) closure per coarse z: "
          + "  ".join(f"K{e['coarse_z']}: R={e['ratio']:.3f} in95={e['in95']}"
                      if e["ratio"] is not None else f"K{e['coarse_z']}: truth=0"
                      for e in dndx))
    print("  t_K vs prior: "
          + "  ".join(f"K{tb['K']}: z0={tb['z0']:+.2f} "
                      f"shrink={tb['shrinkage']:.2f}" for tb in report["t"]))


if __name__ == "__main__":
    main()
