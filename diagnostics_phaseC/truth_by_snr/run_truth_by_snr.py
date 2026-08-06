#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-C truth-by-SNR refold (PI ruling 2026-08-06 §16 — AUTHORIZED).

ONE bounded diagnostic; ZERO added model freedom; no parameter fitting; no
modification of the production response or FP model.  This is the
"cheapest discriminating next test" named (not run) at the end of the
Phase-B twin diagnosis, now authorized.

PRESPECIFICATION (stated before the run)
----------------------------------------
* SNR definition: the pack's own SNR axis (`snr_edges`, 8 strata; the
  op-mask S2N_RED definition inherited from the catalogue build).
* Truth population: the pack's own `truth_counts_bks` (B, Kf, S) — the
  truth histogram stratified by the TRUE systems' host-sightline SNR.
  This is an EXISTING pack array; nothing is re-measured.
* Fold inputs: the committed kernel/completeness/g at the truth-equivalent
  point, exactly as `forward_selftest.selftest` uses them.  The ONLY change
  is the truth's SNR allocation:

      baseline  contrib·dX = C·g·tc[b,k]·dX[k,s]/dX_tot[k]   (pathlength ∝)
      refold    contrib·dX = C·g·tc_bks[b,k,s]               (truth's own)

  The FP term is untouched.  Support: the same live (dX > 0) cells.
* PREDICTED DISCRIMINATING SIGNATURE (before the run): the kernel K and
  completeness C are SNR-dependent, so if the true absorbers are NOT
  distributed across strata in proportion to pathlength, the baseline fold
  mis-weights the per-stratum kernels; that mis-weighting can generate BOTH
  the smooth observed-N̂ tilt and the H10 G1 SNR tilt with no calibration
  error at all.  If instead the refold leaves the 3-group residual and the
  smooth tilt essentially unchanged, allocation/composition is REFUTED as
  the driver and the response-shape explanation stands unchanged.

Reported: per-stratum G1/G2/G3 contributions under both allocations; the
3-group residual change; window χ²/dof change; per-bin z profile change;
replication across all three mocks.  EXPLORATORY: not part of any frozen
confirmatory specification; no SNR nuisance function is introduced (§16).

Usage:  OMP_NUM_THREADS=1 ... python diagnostics_phaseC/truth_by_snr/run_truth_by_snr.py
MOCKS ONLY.  No real-survey values.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from CDDF_analysis.hbi_mcmc.pack import load_pack                    # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS            # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K     # noqa: E402
from CDDF_analysis.hbi_mcmc.gate_covariance import (                 # noqa: E402
    group_aggregator, PRIMARY_GROUP_EDGES)

PACKS = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
         "modelA_pack_{m}_winlya_only_pad19p0_molly172_bw0p2.npz")
MOCKS = ("2lpt0", "london0", "saclay0")
WIN = (19.7, 21.6)


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_REPO, text=True).strip()
    except Exception:
        return "unknown"


def run_one(mock):
    import jax.numpy as jnp
    pk = load_pack(PACKS.format(m=mock))
    consts = build_consts(pk, resp_clamp="both")
    ne = np.asarray(pk.nhat_edges, float)
    dX = np.asarray(pk.dX, float)
    live = (dX > 0)[None, :, :]
    kz = np.asarray(consts.kz_to_K)
    A = group_aggregator(pk, PRIMARY_GROUP_EDGES)
    win_mask = (ne[:-1] >= WIN[0] - 1e-9) & (ne[1:] <= WIN[1] + 1e-9)

    # committed baseline (pathlength-proportional truth allocation)
    st = FS.selftest(pk, resp_clamp="both")
    mu_base = np.where(live, st["mu"], 0.0)
    mu_fp = np.where(live, st["mu_fp"], 0.0)
    obs = np.where(live, st["counts"], 0.0)

    # the SAME kernel/completeness plumbing, truth allocated by its own strata
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    K_full = K[:, kz]                                          # (S, Kf, C, B)
    C_cells = 1.0 / (1.0 + np.exp(-np.asarray(consts.eta_hat)))
    C_bs = C_cells[:, np.asarray(consts.b_to_cell)]            # (S, B)
    g_bk = np.asarray(consts.g_bk)                             # (B, Kf)
    tc = np.asarray(pk.truth_counts, float)                    # (B, Kf)
    tcb = np.asarray(pk.truth_counts_bks, float)               # (B, Kf, S)

    # sanity 1: strata sum reproduces the marginal truth histogram
    marg_err = float(np.max(np.abs(tcb.sum(axis=2) - tc)
                            / np.maximum(tc, 1.0)))
    # sanity 2: rebuilding the BASELINE from the same plumbing reproduces
    # selftest's mu_sig (guards this script's einsum against the fold)
    dX_tot = dX.sum(axis=1)
    share = dX / np.maximum(dX_tot[:, None], 1e-30)            # (Kf, S)
    alloc_base = tc[:, :, None] * share[None, :, :]            # (B, Kf, S)
    mu_sig_rebuilt = np.einsum("skcb,sb,bk,bks->cks",
                               K_full, C_bs, g_bk, alloc_base)
    mu_sig_rebuilt = np.where(live, mu_sig_rebuilt, 0.0)
    mu_sig_ref = np.where(live, np.asarray(st["mu_sig"]), 0.0)
    rebuild_err = float(np.max(np.abs(mu_sig_rebuilt - mu_sig_ref))
                        / max(mu_sig_ref.max(), 1e-30))
    if rebuild_err > 1e-8:
        raise RuntimeError(f"[{mock}] baseline rebuild failed: {rebuild_err}")

    # the refold
    mu_sig_snr = np.einsum("skcb,sb,bk,bks->cks", K_full, C_bs, g_bk, tcb)
    mu_snr = np.where(live, mu_sig_snr, 0.0) + mu_fp

    def _summ(mu):
        mu_c = mu.sum(axis=(1, 2))
        obs_c = obs.sum(axis=(1, 2))
        d_grp = (A @ obs_c) - (A @ mu_c)
        z_c = (obs_c - mu_c) / np.sqrt(np.maximum(mu_c, 1e-12))
        wi = np.where(win_mask)[0]
        chi2 = float(np.sum(z_c[wi] ** 2) / len(wi))
        # per-stratum group table
        mu_cs = mu.sum(axis=1)                                 # (C, S)
        obs_cs = obs.sum(axis=1)
        grp_s = {"mu": (A @ mu_cs).tolist(), "obs": (A @ obs_cs).tolist()}
        return {"group_residual": d_grp.tolist(), "chi2_dof_window": chi2,
                "z_per_bin": z_c.tolist(), "per_stratum_groups": grp_s,
                "total_mu": float(mu_c.sum())}

    base, snr = _summ(mu_base), _summ(mu_snr)
    # allocation distance (context; structural_probes reports the same ratio)
    L1 = float(np.abs(alloc_base - tcb).sum() / max(tcb.sum(), 1e-30))
    return {
        "pack": PACKS.format(m=mock),
        "sanity": {"strata_marginal_max_rel_err": marg_err,
                   "baseline_rebuild_rel_err": rebuild_err},
        "truth_alloc_L1_frac": L1,
        "baseline_pathlength_alloc": base,
        "refold_truth_by_snr": snr,
        "delta_group_residual": (np.array(snr["group_residual"])
                                 - np.array(base["group_residual"])).tolist(),
    }


def main():
    out = {"schema": "phaseC_truth_by_snr/v1",
           "label": ("EXPLORATORY bounded diagnostic (PI §16); zero model "
                     "freedom; predicted signature in the module docstring"),
           "routine": "diagnostics_phaseC/truth_by_snr/run_truth_by_snr.py",
           "date": time.strftime("%Y-%m-%d"), "code_commit": _git(),
           "scope": "MOCK ONLY. No real-survey values.",
           "mocks": {}}
    for m in MOCKS:
        r = run_one(m)
        out["mocks"][m] = r
        print(f"[{m}] L1(alloc)={r['truth_alloc_L1_frac']:.4f}  "
              f"d_grp base={np.round(r['baseline_pathlength_alloc']['group_residual'], 1)}"
              f" -> refold={np.round(r['refold_truth_by_snr']['group_residual'], 1)}  "
              f"chi2 {r['baseline_pathlength_alloc']['chi2_dof_window']:.2f}"
              f" -> {r['refold_truth_by_snr']['chi2_dof_window']:.2f}")
    with open(os.path.join(_HERE, "truth_by_snr.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(_HERE, "truth_by_snr.json"))


if __name__ == "__main__":
    main()
