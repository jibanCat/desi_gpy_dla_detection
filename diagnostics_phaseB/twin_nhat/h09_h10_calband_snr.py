#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-B bounded diagnosis (exploratory) — H9 (finite calibration-shape
noise) and H10 (N-SNR interaction, descriptive).

PRE-STATED PREDICTED SIGNATURES, written BEFORE the numbers:
  H9: the loa-0 FP calibration has 89 in-support events, ALL below N-hat 20.3
      (per-group [29, 0, 0] in the window).  Finite-sample shape noise in
      n0[c,s] produces per-bin mu wiggles ONLY where FP mass sits (the G1
      bins), coherent ACROSS mocks (the same n0 enters every pack).  If the
      low-end per-bin wiggle is calibration-shape noise, then dividing the
      residual by sqrt(mu + var_cal) instead of sqrt(mu) collapses the
      low-end z's toward O(1) while leaving the G3 tilt untouched (G3 has no
      FP mass, var_cal ~ 0).  Layer B already answered this at group level
      (G1 z: -8.83 survey-only -> -1.99 with calibration); this section
      reports the PER-BIN version and the chi2 fraction it covers.
      var_cal per bin is the delta method, which is EXACT here: mu_FP is
      linear in n0 and the E_cov resampling unit is n0* ~ Poisson(n0).
  H10: if the G3 under-prediction is a kernel/shape error in N alone it
      should be roughly SNR-uniform (the response cells share the high-N
      extrapolation regime); if it is selection-like it concentrates in
      specific strata.  Descriptive only.
"""
import json

import numpy as np

OUT = "/home/mfho/wt_repair_phaseB/diagnostics_phaseB/twin_nhat"

res = {"label": "exploratory (prespecified discriminants, spec s7.9-7.10)",
       "H9": {}, "H10": {}}

for m in ("2lpt0", "london0", "saclay0"):
    d = np.load(f"{OUT}/base_{m}_both.npz")
    win = d["win_mask"].astype(bool)
    lo = d["nhat_edges"][:-1][win]
    resid = d["resid_c"][win]
    mu = d["mu_c"][win]
    mu_fp = d["mu_fp_c"][win]
    vcal = d["var_cal_c"][win]
    z_surv = resid / np.sqrt(mu)
    z_pred = resid / np.sqrt(mu + vcal)
    chi2_surv = float(np.sum(z_surv ** 2))
    chi2_pred = float(np.sum(z_pred ** 2))
    res["H9"][m] = {
        "per_bin": [
            {"lo": float(l), "mu_fp_share": float(fp / mm),
             "sd_cal_over_sd_surv": float(np.sqrt(vc / mm)),
             "z_survey_only": float(zs), "z_with_cal": float(zp)}
            for l, mm, fp, vc, zs, zp
            in zip(lo, mu, mu_fp, vcal, z_surv, z_pred)],
        "chi2_dof_survey_only": chi2_surv / len(lo),
        "chi2_dof_with_cal": chi2_pred / len(lo),
        "chi2_fraction_absorbed_by_cal_band": 1.0 - chi2_pred / chi2_surv,
        "n_bins_abs_z_above_3_survey": int(np.sum(np.abs(z_surv) > 3)),
        "n_bins_abs_z_above_3_with_cal": int(np.sum(np.abs(z_pred) > 3)),
        "chi2_G3_bins_survey": float(np.sum(z_surv[lo >= 21.0 - 1e-9] ** 2)),
        "chi2_G3_bins_with_cal": float(np.sum(z_pred[lo >= 21.0 - 1e-9] ** 2)),
    }
    # H10: 3-group residual per SNR stratum
    A = d["A"]
    mu_cs, obs_cs = d["mu_cs"], d["obs_cs"]
    snr_edges = d["snr_edges"]
    tab = []
    for s in range(mu_cs.shape[1]):
        gmu = A @ mu_cs[:, s]
        gobs = A @ obs_cs[:, s]
        if gmu.sum() == 0 and gobs.sum() == 0:
            continue
        z = (gobs - gmu) / np.sqrt(np.maximum(gmu, 1e-12))
        tab.append({"snr_lo": float(snr_edges[s]),
                    "snr_hi": float(snr_edges[s + 1]),
                    "obs": gobs.tolist(), "mu": gmu.tolist(),
                    "residual": (gobs - gmu).tolist(),
                    "z_survey_only": z.tolist()})
    res["H10"][m] = tab

with open(f"{OUT}/h09_h10_calband_snr.json", "w") as fh:
    json.dump(res, fh, indent=1)

for m in ("2lpt0", "london0", "saclay0"):
    r = res["H9"][m]
    print(m, "chi2/dof survey %.2f -> with-cal %.2f (absorbed %.0f%%); "
          "G3 chi2 %.1f -> %.1f"
          % (r["chi2_dof_survey_only"], r["chi2_dof_with_cal"],
             100 * r["chi2_fraction_absorbed_by_cal_band"],
             r["chi2_G3_bins_survey"], r["chi2_G3_bins_with_cal"]))
    print("  z(with cal):", " ".join(
        "%+.1f" % b["z_with_cal"] for b in r["per_bin"]))
    print("  SNR strata (G1/G2/G3 z):")
    for t in res["H10"][m]:
        print("   [%.0f,%s): %s" % (
            t["snr_lo"],
            "inf" if np.isinf(t["snr_hi"]) else "%.0f" % t["snr_hi"],
            " ".join("%+.2f" % z for z in t["z_survey_only"])))
