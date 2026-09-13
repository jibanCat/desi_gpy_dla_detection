#!/usr/bin/env python
"""mock_ppc.py — VALIDATION-ONLY posterior-predictive check of an FP-ladder MOCK
run, through the model_cc fold, with the COMMITTED ``evidence.ppc_block``
statistics (predeclaration §5.3, PPC soft flag).

This is `cc_real_ppc.main` with exactly two differences, both required and both
stated here:
  1. the real-mode gate (`cc_real_posterior._real_mode_gate`) is NOT applied —
     the ladder is MOCK-ONLY, so a mock pack must be admitted; a MOCK gate
     (truth_counts present and nonzero) is applied instead, which is the
     complement of the refused condition, not a relaxation of it;
  2. the nuisance draws come from the ladder runner's ``RUN_*_bychain.npz``,
     which saves the POPULATION sites (sigma_N, sigma_z, theta_level,
     theta_slope, eps_N, eps_z, psi_c) rather than the derived ``theta_pop``.
     ``theta_pop`` is therefore reconstructed here EXACTLY as
     ``fp_ladder.model_cc_ladder`` builds it (2-D random walk, non-centred), and
     — when the run's ``_fdraws.npz`` is present — the reconstruction is VERIFIED
     draw-for-draw against ``f = exp(theta_pop)`` saved by the runner. The max
     absolute log-deviation is reported in the output as
     ``theta_reconstruction.max_abs_dev``; the check is fail-closed unless
     ``--allow-theta-mismatch`` is given.

No statistic is re-derived: `evidence.ppc_block` and `cc_real_ppc.report_grain_ppc`
are called as-is. The one additive number is the predeclared S/N-stratum ramp
amplitude, defined as it is read off the frozen real-C1 PPC record:

    ramp amplitude = max_s (mu_median_s / obs_s) - min_s (mu_median_s / obs_s)

over ``ppc_block["marginal_by_snr"]`` (frozen real run = 0.1123, C1 = 0.1063 ->
the predeclaration's flag line 0.106). Flagged if > 0.106.

Usage:
  python validation/fp_ladder/mock_ppc.py --run RUN_M2_2lpt0_s20260811.json \
      [--pack PACK.npz] [--n-rep-draws 300] [--ppc-seed 0] [--out OUT.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

RAMP_FLAG = 0.106                # predeclaration §5.3 (the real-C1 PPC value)
POP_SITES = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z")


def theta_pop_from_population_sites(sigma_N, sigma_z, level, slope, eps_N, eps_z):
    """model_cc / model_cc_ladder's population construction, verbatim, vectorised
    over a leading draw axis.

        b_idx      = arange(B) - (B-1)/2
        curv       = cumsum(cumsum([0, 0, eps_N]))[:B]
        theta[:,0] = level + slope*b_idx + sigma_N*curv
        theta      = theta[:,0][:,None] + [0 | sigma_z*cumsum(eps_z, axis=1)]

    Shapes: sigma_N/sigma_z/level/slope (n,), eps_N (n, B-2), eps_z (n, B, Kf-1).
    Returns theta (n, B, Kf).
    """
    sigma_N = np.asarray(sigma_N, float).reshape(-1)
    sigma_z = np.asarray(sigma_z, float).reshape(-1)
    level = np.asarray(level, float).reshape(-1)
    slope = np.asarray(slope, float).reshape(-1)
    eps_N = np.asarray(eps_N, float)
    eps_z = np.asarray(eps_z, float)
    n, B = eps_z.shape[0], eps_z.shape[1]
    Kf = eps_z.shape[2] + 1
    pad = np.concatenate([np.zeros((n, 2)), eps_N], axis=1)          # (n, B)
    curv = np.cumsum(np.cumsum(pad, axis=1), axis=1)[:, :B]
    b_idx = np.arange(B) - 0.5 * (B - 1)
    col0 = level[:, None] + slope[:, None] * b_idx[None, :] + sigma_N[:, None] * curv
    walk = sigma_z[:, None, None] * np.cumsum(eps_z, axis=2)          # (n, B, Kf-1)
    return np.concatenate([col0[:, :, None],
                           col0[:, :, None] + walk], axis=2)          # (n, B, Kf)


def snr_ramp(ppc_block, flag_at=RAMP_FLAG):
    """The S/N-stratum marginal ramp of the committed PPC: mu_median / obs per
    stratum, and its peak-to-peak amplitude."""
    rows = ppc_block.get("marginal_by_snr") or []
    ratio = [(int(r["i"]), float(r["mu_median"]) / float(r["obs"]))
             for r in rows if float(r.get("obs", 0)) > 0]
    if not ratio:
        return dict(ratios=None, amplitude=None, flag=False,
                    note="no S/N stratum with obs > 0")
    vals = [v for _, v in ratio]
    amp = float(max(vals) - min(vals))
    return dict(strata=[i for i, _ in ratio],
                ratios=[float(v) for v in vals],
                amplitude=amp, flag_threshold=float(flag_at),
                flag=bool(amp > flag_at),
                definition=("max_s(mu_median/obs) - min_s(mu_median/obs) over "
                            "ppc_block.marginal_by_snr; frozen real run 0.1123, "
                            "C1 0.1063"))


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="RUN_<M>_<fam>_s<seed>.json")
    ap.add_argument("--pack", default=None, help="override the pack named in the run JSON")
    ap.add_argument("--n-rep-draws", type=int, default=300)
    ap.add_argument("--ppc-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <run base>_ppc.json")
    ap.add_argument("--theta-tol", type=float, default=1e-6)
    ap.add_argument("--allow-theta-mismatch", action="store_true")
    a = ap.parse_args(argv)

    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    from CDDF_analysis.hbi_mcmc import evidence as EV
    from CDDF_analysis.hbi_mcmc import cc_real_ppc as CRP

    run = json.load(open(a.run))
    base = a.run[:-5]
    byc, fdr = base + "_bychain.npz", base + "_fdraws.npz"
    if not os.path.exists(byc):
        raise SystemExit(f"missing by-chain draws: {byc}")
    packp = a.pack or run.get("pack")
    if not packp or not os.path.exists(packp):
        raise SystemExit(f"pack not found ({packp!r}); pass --pack")

    pk = load_pack(packp)
    tc = np.asarray(pk.truth_counts) if pk.truth_counts is not None else np.zeros(0)
    if tc.size == 0 or tc.sum() <= 0:
        raise SystemExit("MOCK GATE: pack carries no truth_counts — this PPC is "
                         "MOCK-ONLY (the real-mode gate of cc_real_ppc is replaced "
                         "by its complement, not removed)")
    consts, Mg = build_cc_tensors(pk)

    z = np.load(byc, allow_pickle=True)
    by = {k: z[k] for k in z.files}
    nch, ndr = np.asarray(by["potential_energy"]).shape[:2]

    def _flat(k):
        v = np.asarray(by[k], float)
        return v.reshape((nch * ndr,) + v.shape[2:])

    # ---- theta_pop: use it if the runner saved it, else rebuild it verbatim --
    if "theta_pop" in by:
        theta = _flat("theta_pop")
        theta_src = "read from the by-chain file"
        dev = 0.0
    else:
        missing = [s for s in POP_SITES if s not in by]
        if missing:
            raise SystemExit(f"by-chain file lacks {missing}; cannot rebuild theta_pop")
        theta = theta_pop_from_population_sites(
            _flat("sigma_N"), _flat("sigma_z"), _flat("theta_level"),
            _flat("theta_slope"), _flat("eps_N"), _flat("eps_z"))
        theta_src = ("reconstructed from the population sites exactly as "
                     "fp_ladder.model_cc_ladder builds theta_pop")
        dev = None
    if os.path.exists(fdr):
        f_saved = np.asarray(np.load(fdr)["f"], float)
        if f_saved.shape == theta.shape:
            dev = float(np.max(np.abs(np.log(np.clip(f_saved, 1e-300, None)) - theta)))
        else:
            dev = None
    if dev is None:
        if not a.allow_theta_mismatch:
            raise SystemExit("theta_pop reconstruction could not be verified against "
                             "the saved f draws (shape mismatch or no _fdraws.npz); "
                             "pass --allow-theta-mismatch to proceed anyway")
    elif dev > a.theta_tol and not a.allow_theta_mismatch:
        raise SystemExit(f"theta_pop reconstruction deviates from log f by {dev:.3e} "
                         f"> {a.theta_tol:g} — refusing (fail-closed)")

    n_kk = int(consts.n_kk)
    t = _flat("t") if "t" in by else np.zeros((nch * ndr, n_kk))
    flat = dict(theta_pop=theta, psi_c=_flat("psi_c"), t=t, lam_fp=_flat("lam_fp"))

    mu, idx = CRP.mu_draws_cc(consts, Mg, flat, n_max=a.n_rep_draws, seed=a.ppc_seed)
    # ORACLE diagnostic: lam_fp is identically zero in the saved draws because mu_FP was
    # PINNED to the mock FP-truth census inside the model; add that fixed term back so the
    # PPC sees the same mu the likelihood saw (otherwise the ratios are TP-only / obs).
    oracle_note = None
    if run.get("ladder") == "ORACLE":
        argv_run = list((run.get("run_config") or {}).get("argv") or [])
        cpath = None
        for i_, tok in enumerate(argv_run):
            if tok == "--census" and i_ + 1 < len(argv_run):
                cpath = argv_run[i_ + 1]
        if cpath is None or not os.path.exists(cpath):
            raise SystemExit("ORACLE PPC: census path not recoverable from run_config.argv")
        host = np.asarray(np.load(cpath, allow_pickle=True)["hostless"], float)
        mu = mu + host[None, ...]
        oracle_note = f"ORACLE: fixed mu_FP = hostless census added from {cpath}"
    blk = EV.ppc_block({"samples_by_chain": None}, pk, consts,
                       n_rep_draws=a.n_rep_draws, seed=a.ppc_seed, mu_draws=(mu, idx))
    grain = CRP.report_grain_ppc(mu, np.asarray(pk.counts, float),
                                 np.asarray(pk.dX, float),
                                 np.asarray(pk.nhat_edges, float),
                                 np.asarray(pk.zf_edges, float),
                                 seed=a.ppc_seed + 1)
    ramp = snr_ramp(blk)
    checks = blk.get("checks", {})
    flags = []
    for k, v in checks.items():
        if v is False:
            flags.append(f"committed check {k} FAILED")
    if ramp.get("flag"):
        flags.append(f"S/N ramp amplitude {ramp['amplitude']:.4f} > {RAMP_FLAG}")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        commit = "UNKNOWN"

    out = dict(
        role=("MOCK posterior-predictive check of an FP-ladder candidate run — "
              "DIAGNOSTIC / soft flag (sealed predeclaration §5.3); not a gate; "
              "the posterior, model, prior and calibration are unchanged"),
        run=os.path.abspath(a.run), run_sha256=_sha(a.run),
        draws=os.path.abspath(byc), draws_sha256=_sha(byc),
        pack=os.path.abspath(packp), pack_sha256=_sha(packp),
        ladder=run.get("ladder"), run_config=run.get("run_config"), oracle_note=oracle_note,
        run_thresholds=run.get("thresholds"), run_diagnostics=run.get("diagnostics"),
        n_draws_total=int(nch * ndr), n_rep_draws=int(len(idx)), ppc_seed=a.ppc_seed,
        theta_reconstruction=dict(source=theta_src, max_abs_dev=dev,
                                  tol=a.theta_tol,
                                  verified_against="f = exp(theta_pop) in the _fdraws.npz"),
        policy={"PPC_PVAL_MIN": EV.PPC_PVAL_MIN,
                "PPC_MAX_FAILED_CELL_FRAC": EV.PPC_MAX_FAILED_CELL_FRAC,
                "PPC_OMNIBUS_MIN": EV.PPC_OMNIBUS_MIN,
                "SNR_RAMP_FLAG": RAMP_FLAG},
        ppc_block=blk, report_grain=grain, snr_ramp=ramp,
        flags=flags, flag_any=bool(flags), code_commit=commit)
    outp = a.out or (base + "_ppc.json")
    os.makedirs(os.path.dirname(os.path.abspath(outp)) or ".", exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1, default=str)

    print(f"PPC written {outp}: cells {blk['n_cells']} failed {blk['n_cells_failed']} "
          f"omnibus p {blk['omnibus_chi2_discrepancy']['posterior_predictive_p']:.4f} "
          f"checks {checks}")
    print(f"  S/N ramp ratios {['%.4f' % v for v in (ramp.get('ratios') or [])]} "
          f"amplitude {ramp.get('amplitude')} flag {ramp.get('flag')} (> {RAMP_FLAG})")
    print(f"  theta_pop {theta_src}; max|log f - theta| = {dev}")
    for b in grain["blocks"]:
        print(f"  report grain z[{b['z_lo']},{b['z_hi']}): "
              f"T_obs/n {b['T_over_n']['T_obs_over_n']:.2f} "
              f"p {b['T_over_n']['posterior_predictive_p']:.3f} "
              f"bins failed {b['n_bins_failed']}/{b['n_bins']}")
    if flags:
        print("  FLAGS:", "; ".join(flags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
