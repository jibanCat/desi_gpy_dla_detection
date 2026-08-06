#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-C calibration sizing: anchor counts for the §9 precision/power criteria.

Turns the committed preimage sensitivity map (diagnostics_phaseC/preimage/)
plus the current response's per-cell kernel widths into the production
sample size: pairs per true-N bin (pooled and per response cell), implied
injections, CPU-hours, and the achieved sigma(G3 prediction) / power.

SIZING MODEL (all assumptions stated; every number here is PILOT-ADJUSTABLE
except the two frozen criteria):
  * The new calibration measures, per true-N production bin b and response
    cell r, the kernel mean (sigma = sd_r(N_b)/sqrt(n_br)) and fractional
    width (sigma = 1/sqrt(2 n_br)) from n_br matched injection pairs.
  * Bins are treated as independently measured (conservative vs a smooth
    fit across anchors, which shares information).
  * G3-projection: sigma^2(G3) = sum_b [ S_b^2 sd_b^2 / n_b
                                        + W_b^2 / (2 n_b) ],
    with S_b = dG3/d(mean shift) [counts/dex] and W_b = dG3/d(unit
    fractional width) [counts] from the preimage sensitivity map, and
    sd_b = the G3-share-weighted per-cell kernel sd at N_b.
  * FROZEN CRITERIA (PI §9; may not be weakened):
      (1) sigma(G3 pred) <= (1/3) * |G3 residual| = 150.1 counts;
      (2) power >= 0.90 against a perturbation explaining the full G3
          residual, at two-sided alpha = 0.01 on the G3-projected
          difference test  =>  sigma(G3) <= |G3|/(z_{0.995}+z_{0.90}).
    Criterion (2) is the binding one.
  * Pairs -> injections via the pack's own molly completeness
    (G3-share-weighted across SNR strata), + a 15% retry/failure allowance.
  * CPU cost: 167 CPU-s per injected spectrum (loa-124 production log,
    re-confirmed by the wall1 pilot gating note) — the GP finder dominates;
    generation is minutes per arm.

Outputs sizing.json next to this script.  MOCKS ONLY.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

PREIMAGE = os.path.join(_REPO, "diagnostics_phaseC/preimage/preimage.json")
ENVELOPE = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/track_c/"
            "stage0/forward_response_2lpt0.npz")
PACK = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
        "modelA_pack_2lpt0_winlya_only_pad19p0_molly172_bw0p2.npz")

G3_RESIDUAL = 450.25          # closure_table_phaseB.json, twin (planning value)
ALPHA = 0.01                  # two-sided, frozen with criterion 2
POWER = 0.90
CPU_S_PER_SPEC = 167.0        # measured (loa-124 production; wall1 pilot)
RETRY_ALLOWANCE = 0.15
#: production true-N bins (0.2-dex basis indices) — the G3 feed + margin
#: from the preimage: [20.5, 21.9) dense + [21.9, 22.2) thin tail coverage
PROD_BINS = list(range(7, 14))            # b7..b13 = [20.5, 21.9)
TAIL_BINS = [14, 15]                      # [21.9, 22.2] thin (fixed floor)
BRIDGE_BINS = list(range(2, 7))           # b2..b6 = [19.5, 20.5) overlap
TAIL_FLOOR = 60                           # pairs per tail bin (fixed, thin)
BRIDGE_PAIRS_PER_BIN = 150                # pooled; sized for the bridge test
CELL_FLOOR_PAIRS = 12                     # min pairs per (bin, cell)


def main():
    pre = json.load(open(PREIMAGE))
    tw = pre["mocks"]["2lpt0"]
    S_raw = np.array(tw["sensitivity"]["dG3_per_bin_mean_shift"])  # per +0.02dex
    W_raw = np.array(tw["sensitivity"]["dG3_per_bin_width_scale"])  # per +10%
    S = np.abs(S_raw) / 0.02                                   # counts / dex
    W = np.abs(W_raw) / 0.10                                   # counts / unit frac
    ed = np.array(tw["table"]["ntrue_edges"])
    Nc = 0.5 * (ed[:-1] + ed[1:])
    cell_share = np.array(tw["g3_by_response_cell"])           # (3, 3)
    cell_share = cell_share / cell_share.sum()

    env = np.load(ENVELOPE)
    mu_co, sg_co = env["mu_coef"], env["sig_coef"]
    nref = float(env["N_ref"])

    def sd_cell(N, sr, zr):
        u = np.array([1.0, N - nref, (N - nref) ** 2])
        return max(float(sg_co[sr, zr] @ u), 0.02)

    # G3-share-weighted kernel sd per bin (covariate clamped at each cell's
    # top anchor, matching the deployed clamp)
    rr = np.array(tw["resp_N_fit_range"])                      # (3, 3, 2)
    sd_b = np.zeros(len(Nc))
    for b, N in enumerate(Nc):
        acc = 0.0
        for sr in range(3):
            for zr in range(3):
                Ncl = min(max(N, rr[sr, zr, 0]), rr[sr, zr, 1])
                acc += cell_share[sr, zr] * sd_cell(Ncl, sr, zr)
        sd_b[b] = acc

    # completeness (pairs -> injections), G3-SNR-share-weighted
    import sys
    sys.path.insert(0, _REPO)
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    pk = load_pack(PACK)
    nd = np.asarray(pk.molly_n_det, float)                     # (S, M)
    nt = np.asarray(pk.molly_n_tot, float)
    comp = (nd + 0.5) / (nt + 1.0)
    me = np.asarray(pk.molly_nhi_edges, float)
    snr_share = np.array(tw["g3_by_snr_stratum"])
    snr_share = snr_share / snr_share.sum()
    b2m = np.clip(np.digitize(Nc, me) - 1, 0, len(me) - 2)

    def comp_b(b):
        return float(np.sum(snr_share * comp[:, b2m[b]]))

    # ---- solve for uniform n over PROD_BINS, then Neyman allocation ----
    z_need = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
    sig_target_power = G3_RESIDUAL / z_need                    # binding
    sig_target_prec = G3_RESIDUAL / 3.0
    sig_target = min(sig_target_power, sig_target_prec)

    def var_G3(n_by_bin):
        v = 0.0
        for b, n in n_by_bin.items():
            v += (S[b] ** 2 * sd_b[b] ** 2 + W[b] ** 2 / 2.0) / n
        return v

    # per-bin variance load
    load = {b: S[b] ** 2 * sd_b[b] ** 2 + W[b] ** 2 / 2.0 for b in PROD_BINS}
    # uniform n
    n_uni = int(np.ceil(sum(load.values()) / sig_target ** 2))
    # Neyman: n_b proportional to sqrt(load_b); total minimized
    tot_neyman = (sum(np.sqrt(v) for v in load.values()) ** 2) / sig_target ** 2
    n_ney = {b: int(np.ceil(np.sqrt(load[b]) / sum(np.sqrt(v) for v in load.values())
                            * tot_neyman)) for b in PROD_BINS}
    # enforce per-(bin, cell) floor so every response cell is measured
    n_final = {b: max(n_ney[b], CELL_FLOOR_PAIRS * 9) for b in PROD_BINS}
    for b in TAIL_BINS:
        n_final[b] = TAIL_FLOOR
    sig_achieved = float(np.sqrt(var_G3({b: n for b, n in n_final.items()
                                         if b in PROD_BINS})))
    power_achieved = float(
        norm.cdf(G3_RESIDUAL / sig_achieved - norm.ppf(1 - ALPHA / 2)))

    # implied G1/G2 precision (PI §9: report, do not let it hide G3)
    implied = {}
    sens = pre["mocks"]["2lpt0"]["sensitivity"]
    for g in ("G1", "G2"):
        kM, kW = f"d{g}_per_bin_mean_shift", f"d{g}_per_bin_width_scale"
        if kM not in sens:
            implied[g] = "preimage sensitivity columns absent (rerun preimage)"
            continue
        Sg = np.abs(np.array(sens[kM])) / 0.02
        Wg = np.abs(np.array(sens[kW])) / 0.10
        v = 0.0
        for b, n in n_final.items():
            v += (Sg[b] ** 2 * sd_b[b] ** 2 + Wg[b] ** 2 / 2.0) / n
        for b in BRIDGE_BINS:
            v += (Sg[b] ** 2 * sd_b[b] ** 2 + Wg[b] ** 2 / 2.0) \
                / BRIDGE_PAIRS_PER_BIN
        implied[g] = round(float(np.sqrt(v)), 1)

    # injections and cost
    rows = []
    tot_pairs = tot_inj = 0
    for b in sorted(set(PROD_BINS + TAIL_BINS + BRIDGE_BINS)):
        role = ("production" if b in PROD_BINS else
                "production-tail" if b in TAIL_BINS else "bridge")
        n_pairs = (n_final.get(b) if role.startswith("production")
                   else BRIDGE_PAIRS_PER_BIN)
        cb = comp_b(b)
        n_inj = int(np.ceil(n_pairs / cb * (1 + RETRY_ALLOWANCE)))
        rows.append({"bin": int(b), "true_lo": float(ed[b]),
                     "true_hi": float(ed[b + 1]), "role": role,
                     "pairs": int(n_pairs), "completeness": round(cb, 3),
                     "injections": n_inj,
                     "sd_kernel_dex": round(float(sd_b[b]), 3),
                     "S_counts_per_dex": round(float(S[b]), 1),
                     "W_counts_per_unitfrac": round(float(W[b]), 1)})
        tot_pairs += n_pairs
        tot_inj += n_inj
    cpu_h = tot_inj * CPU_S_PER_SPEC / 3600.0

    out = {
        "schema": "phaseC_calib_sizing/v1",
        "date": time.strftime("%Y-%m-%d"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
        "frozen_criteria": {
            "sigma_G3_precision_max": sig_target_prec,
            "sigma_G3_power_max": sig_target_power,
            "binding": "power",
            "alpha_two_sided": ALPHA, "power_target": POWER,
            "effect_size_counts": G3_RESIDUAL,
            "effect_size_equivalents": {
                "mean_shift_dex_at_20.7_21.1": 0.031,
                "mean_shift_dex_at_21.1_21.5_negative": -0.13,
                "note": "from the preimage sensitivity map (planning)"},
        },
        "sizing_model_assumptions": [
            "per-bin independent measurement (conservative vs smooth fit)",
            "sigma(mean) = sd_kernel/sqrt(n); sigma(width)/width = 1/sqrt(2n)",
            "sd_kernel from the CURRENT response (pilot re-measures)",
            "completeness from the pack molly surface (pilot re-measures)",
            "167 CPU-s/spec (pilot re-measures)",
        ],
        "n_uniform_per_bin": n_uni,
        "sigma_G3_achieved": sig_achieved,
        "power_achieved": power_achieved,
        "implied_sigma_G1_G2": implied,
        "table": rows,
        "totals": {"pairs": int(tot_pairs), "injections": int(tot_inj),
                   "cpu_hours_finder": round(cpu_h, 1),
                   "cpu_hours_with_pilot_and_margin":
                       round(cpu_h + 30.0, 1),
                   "storage_gb_estimate": round(tot_inj * 0.001 + 5, 1)},
        "per_cell_note": ("pairs allocated across the 9 response cells "
                          "proportional to the cell G3 share "
                          f"{np.round(cell_share, 3).tolist()} with a floor "
                          f"of {CELL_FLOOR_PAIRS} pairs/(bin,cell)"),
    }
    with open(os.path.join(_HERE, "sizing.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("n_uniform_per_bin", "sigma_G3_achieved",
                       "power_achieved", "totals")}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
