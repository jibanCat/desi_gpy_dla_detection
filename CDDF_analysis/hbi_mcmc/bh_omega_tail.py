#!/usr/bin/env python
"""bh_omega_tail.py — the BH / high-z arm's Omega_DLA under explicit upper-integration
conventions and high-N_HI tail continuations (R-037 extension, PI 2026-08-28), plus the
convention comparison with the Qz5 z ~ 5 HI mass density.

Referee-facing DIAGNOSTIC. The headline Omega_DLA of Paper 1 stays the low-z HBI value over
20.3 <= log N_HI <= 21.6; the high-z route's Omega is diagnostic-only (PI 2026-08-26 #45) and
nothing here changes that. No frozen artifact is modified.

What the BH route actually integrates (read from the producer, track_c_tf_loa.run_measurement /
track_c_perz_band.perz_omega_from_fbk and cddf_catalog_hbi.joint_mc_errors): Omega(>= lim) =
K * sum_{logN_lo >= lim} N_b f_b dN_b over the fine 0.1-dex grid up to drop_top_bin_above
(22.4) -- i.e. OPEN-TOPPED to 22.4, not closed at 21.6. The MC band is the recentred
percentile band of the same reduction over the joint-MC f(N) samples (band_recenter). This
module re-implements exactly that reduction on the samples dumped by
`track_c_tf_hz.py --dump-npz` and asserts closure against the RATIFIED artifact before
writing anything.

Tail continuations (f(N) below the cut is always the in-data BH f; above it, one of):
  M0  hard cut (the adopted convention, [20.3, 21.6]);
  M1  the BH model's own continuation to 22.4 (= the artifact's open-topped value);
  M2  power law anchored just below the cut, slope fitted over the 0.6 dex below it, with the
      repo's predeclared +-sigma_slope bracket (HBIConfig.omega_slope_extrap_sigma = 0.5 per
      dex; FIX 2) as the conservative bracket allowed by the calibrated bins;
  M3  the PHW05 Gamma-function reference tail (the paper's reference spine; params from the
      frozen fig_hbi_cddf.data.npz), amplitude-matched to the in-data f over the last
      0.5 dex below the cut;
  M4  Paper 1's own frozen LOW-z tail shape (pooled draws, path-weighted, per draw),
      amplitude-matched at [21.3, 21.5).
Nothing else; no invented forms. Every added N_HI region carries its calibration status.

Real-data VALUES never enter this file; literature values carry their provenance strings.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

QUANTILES = [2.5, 16.0, 50.0, 84.0, 97.5]
ADOPTED = (20.3, 21.6)
NMAX_SCAN = [21.1, 21.2, 21.3, 21.5, 21.6, 21.7, 21.9, 22.1, 22.4]
SIGMA_SLOPE = 0.5          # HBIConfig.omega_slope_extrap_sigma (predeclared FIX-2 bracket)
FIT_DEX = 0.6              # HBIConfig.omega_slope_extrap_fit_dex
# cosmology of the reporting convention (PI #48) and Qz5's
H0_KMS = 70.0
_MPC_M = 3.0856775814913673e22
_G = 6.67430e-11
_MSUN = 1.98847e30
RHO_CRIT_MSUN_MPC3 = 3.0 * (H0_KMS * 1e3 / _MPC_M) ** 2 / (8.0 * math.pi * _G) * _MPC_M ** 3 / _MSUN

# Qz5 (Oyarzun et al. 2025, ApJ 983, 10; arXiv:2502.05261), read 2026-08-28 from the arXiv
# HTML (Table 3, Table 2, Sec. III.2 eq. 9, Sec. IV). Bounds are the paper's 1-sigma Monte-Carlo
# BOUNDS (typeset as sub/superscripts), NOT +/- errors -- the same convention as the
# transcribed l_DLA(X) row in DLA_data/qz5_crighton15_dndx.txt.
QZ5 = {
    "citation": "Oyarzun et al. 2025, ApJ 983, 10 (arXiv:2502.05261) -- The Qz5 Survey (I)",
    "rho_DLA_1e8_Msun_Mpc3": {"value": 0.39, "lo": 0.18, "hi": 0.68,
                              "what": "Table 3, 'Qz5 DLAs (full sample)': rho_DLA from the five intervening DLAs (log N_HI >= 20.3), z = 4.5-5.6; consistent with rho_HI 0.56 = 1.44 x rho_DLA and the subDLA row 0.169"},
    "rho_HI_1e8_Msun_Mpc3": {"value": 0.56, "lo": 0.31, "hi": 0.82,
                             "what": "Table 3 / abstract: DLAs + subDLAs (delta_HI = 1.44 measured by Qz5; 1.2 in the literature)"},
    "definition": "Sec. III.2 eq. (9): rho_HI(X) dX = (8 pi G / 3 H0)(m_H / c) delta_HI int N_HI f(N_HI, X) dN_HI dX -- the first moment of f, 'computed by discretely sampling the integrand in narrow N_HI bins (dN_HI = 1e20) and integrating over redshift intervals'; with five DLAs this is a sum over the observed systems: no fitted f(N), no tail beyond the highest observed column",
    "upper_limit": "none stated; the sum ends at the highest observed DLA",
    "observed_DLA_logN": [20.3, 20.9, 21.1, 20.4, 20.8],
    "observed_DLA_logN_err": [0.15, 0.15, 0.25, 0.2, 0.15],
    "uncertainty": "Sec. IV: Monte Carlo that (1) bootstraps the detected DLAs/subDLAs with Poisson statistics and (2) draws each N_HI from a normal with its Delta N_HI (continuum-dominated, Sec. III.1). No systematic, model or tail/extrapolation term.",
    "high_N_sensitivity_tests": "none reported (no leave-one-out, no upper-limit variation)",
    "cosmology": "flat, Omega_Lambda = 0.7, Omega_m = 0.3, H0 = 70 km/s/Mpc; comoving densities",
    "z_range": [4.5, 5.6], "n_DLA": 5, "n_QSO": 63,
}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here).decode().strip()
    except Exception:
        return "unknown"


def band(samp, point, recenter=True):
    """The producer's band: percentile band of the samples, recentred on the point."""
    s = np.asarray(samp, float)
    s = s[np.isfinite(s)]
    q = np.percentile(s, QUANTILES)
    if recenter:
        q = point + (q - np.median(s))
    return {"MAP": float(point), "q025": float(q[0]), "q16": float(q[1]), "q50_recentred": float(q[2]),
            "q84": float(q[3]), "q975": float(q[4]),
            "halfwidth68_pct": float(50.0 * (q[3] - q[1]) / point) if point > 0 else None,
            "halfwidth95_pct": float(50.0 * (q[4] - q[0]) / point) if point > 0 else None}


def omega_grid(fb, logN_lo, N_b, dN_b, K, lo, hi):
    """K * sum N_b f_b dN_b over fine bins with lo <= logN_lo < hi (fb: (..., n_nbins))."""
    sel = (logN_lo >= lo - 1e-9) & (logN_lo < hi - 1e-9)
    return K * np.nansum(np.asarray(fb, float)[..., sel] * (N_b * dN_b)[sel], axis=-1)


def fit_slope(fb, logN_lo, logN_hi, lo, hi):
    """d log10 f_N / d log N over [lo, hi) per sample (least squares on the fine bins)."""
    mid = 0.5 * (logN_lo + logN_hi)
    sel = (logN_lo >= lo - 1e-9) & (logN_hi <= hi + 1e-9)
    x = mid[sel]
    y = np.log10(np.clip(np.asarray(fb, float)[..., sel], 1e-300, None))
    xm = x - x.mean()
    return (y * xm).sum(axis=-1) / (xm ** 2).sum(), x, sel


def powerlaw_tail(fb, logN_lo, logN_hi, N_b, dN_b, K, cut, top, dslope=0.0):
    """Omega of a power-law continuation above `cut` to `top`, anchored at the in-data f_N in the
    last fine bin below the cut, slope fitted over [cut - FIT_DEX, cut) (+ dslope)."""
    slope, x, sel = fit_slope(fb, logN_lo, logN_hi, cut - FIT_DEX, cut)
    slope = slope + dslope
    mid = 0.5 * (logN_lo + logN_hi)
    ianc = int(np.where(np.isclose(logN_hi, cut))[0][0])          # last bin below the cut
    f_anc = np.asarray(fb, float)[..., ianc]
    tail = (logN_lo >= cut - 1e-9) & (logN_lo < top - 1e-9)
    f_tail = f_anc[..., None] * 10.0 ** (slope[..., None] * (mid[tail] - mid[ianc]))
    om = K * (f_tail * (N_b * dN_b)[tail]).sum(axis=-1)
    # analytic continuation to infinity (converges iff slope < -2 in f_N):  int N^{1+s} dN
    s = slope
    Ntop = 10.0 ** top
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inf_extra = np.where(s < -2.0, K * f_anc * (10.0 ** mid[ianc]) ** (-s) * Ntop ** (s + 2.0) / (-(s + 2.0)), np.inf)
    return om, slope, inf_extra


def phw05_tail(fb, logN_lo, logN_hi, N_b, dN_b, K, cut, top, params):
    """PHW05 Gamma-function tail f_N = k2 (N/Ng)^a exp(-N/Ng), amplitude-matched to the in-data
    f_N over [cut-0.5, cut) (ratio of Omega-weighted sums), integrated over [cut, top) on the
    fine grid and to infinity analytically."""
    from scipy.special import gammaincc, gamma as _gamma
    k2, Ng, a = params
    mid = 0.5 * (logN_lo + logN_hi)
    N = 10.0 ** mid
    ref = k2 * (N / Ng) ** a * np.exp(-N / Ng)
    m = (logN_lo >= cut - 0.5 - 1e-9) & (logN_hi <= cut + 1e-9)
    amp = (np.asarray(fb, float)[..., m] * (N_b * dN_b)[m]).sum(axis=-1) / (ref[m] * (N_b * dN_b)[m]).sum()
    tail = (logN_lo >= cut - 1e-9) & (logN_lo < top - 1e-9)
    om = K * amp * (ref[tail] * (N_b * dN_b)[tail]).sum()
    # int_{Ntop}^inf N f_N dN = k2 Ng^2 Gamma(a+2, Ntop/Ng)
    inf_extra = K * amp * k2 * Ng ** 2 * _gamma(a + 2.0) * gammaincc(a + 2.0, 10.0 ** top / Ng)
    return om, amp, inf_extra


def lowz_tail_shape(draws_path, pack_path):
    """Paper-1 frozen low-z f per dex on the latent bins, path-weighted all-z, per draw, as
    ratios to the [21.3, 21.5) bin; returns (edges, per-draw f_dex (D, B), ratios (D, B))."""
    D = np.load(draws_path)
    P = np.load(pack_path, allow_pickle=True)
    f = np.asarray(D["f"], float)
    ne = np.asarray(D["ntrue_edges"], float)
    dXk = np.asarray(P["dX"], float).sum(axis=1)
    fz = (f * dXk[None, None, :]).sum(axis=2) / dXk.sum()          # (D, B) f per dex, path-weighted
    i_ref = int(np.where(np.isclose(ne[:-1], 21.3))[0][0])
    return ne, fz, fz / fz[:, i_ref:i_ref + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="npz from track_c_tf_hz.py --dump-npz (run of record)")
    ap.add_argument("--dump-finite-snr", default=None, help="npz from the --finite-snr-only variant (NaN-SNR closure)")
    ap.add_argument("--bh-artifact", required=True)
    ap.add_argument("--bh-finite-json", default=None, help="the --finite-snr-only variant's output JSON")
    ap.add_argument("--lowz-draws", required=True)
    ap.add_argument("--lowz-pack", required=True)
    ap.add_argument("--paper-cddf-npz", required=True, help="carries the PHW05 reference-spine parameters")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    Z = np.load(a.dump, allow_pickle=True)
    fb = np.asarray(Z["fb_samp"], float)
    fmap = np.asarray(Z["map_fb"], float)
    lo, hi = np.asarray(Z["logN_lo"], float), np.asarray(Z["logN_hi"], float)
    N_b, dN_b, K = np.asarray(Z["N_b"], float), np.asarray(Z["dN_b"], float), float(Z["K"])
    recenter = bool(Z["band_recenter"])
    top_grid = float(hi.max())
    bh = json.load(open(a.bh_artifact))
    art = bh["measurement"]["20.3"]["omega"]["integrated"]
    art_d = bh["measurement"]["20.3"]["dndx"]["integrated"]

    # ---- closure: the open-topped reduction must reproduce the artifact (MAP + band) ----
    om_open_map = float(omega_grid(fmap, lo, N_b, dN_b, K, 20.3, top_grid + 1e-6))
    om_open = omega_grid(fb, lo, N_b, dN_b, K, 20.3, top_grid + 1e-6)
    b_open = band(om_open, om_open_map, recenter)
    closure = {"artifact_integrated_omega": art, "recomputed_open_top_[20.3,%.1f)" % top_grid: b_open,
               "max_rel_diff": max(abs(b_open[k] / art[k] - 1.0) for k in ("MAP", "q16", "q84", "q025", "q975"))}
    if closure["max_rel_diff"] > 1e-9:
        raise SystemExit(f"BLOCKED: open-topped reduction does not reproduce the artifact ({closure['max_rel_diff']:.3e})")
    print(f"closure OK: the artifact's Omega is the OPEN-TOPPED integral [20.3, {top_grid:.1f}) (max rel {closure['max_rel_diff']:.1e})")

    # ---- calibration status of each fine region (from the artifact's own H2 patch record) ----
    h2 = bh["metadata"]["calibration"]["h2_patch"]
    status = [
        {"region": [20.3, 20.5], "status": "gap cell: C_gap = 0.496 [0.407, 0.593] bracket (no H2 injection point); kernel = 2LPT-0 forward response (transported)"},
        {"region": [20.5, 21.0], "status": "H2-calibrated completeness (k/n = 77/117); kernel transported"},
        {"region": [21.0, 21.5], "status": "H2-calibrated completeness (k/n = 35/43); kernel transported"},
        {"region": [21.5, 22.0], "status": "H2 completeness cell k/n = 22/26 -- ABOVE the nominal injected logN grid top (21.5): response-held / thinly calibrated; kernel transported"},
        {"region": [22.0, top_grid], "status": "completeness kept FROZEN from the 2LPT-0 mock (no H2 injections): EXTRAPOLATED; [22.1, 22.4) is ceiling-adjacent on the observed grid"},
        {"h2_patch_record": h2},
    ]

    # ---- upper-limit scan ----
    om_ad_map = float(omega_grid(fmap, lo, N_b, dN_b, K, *ADOPTED))
    om_ad = omega_grid(fb, lo, N_b, dN_b, K, *ADOPTED)
    b_ad = band(om_ad, om_ad_map, recenter)
    scan = {}
    for X in NMAX_SCAN:
        om_map = float(omega_grid(fmap, lo, N_b, dN_b, K, 20.3, X))
        om = omega_grid(fb, lo, N_b, dN_b, K, 20.3, X)
        b = band(om, om_map, recenter)
        above = omega_grid(fmap, lo, N_b, dN_b, K, 21.6, X) / om_map if X > 21.6 else 0.0
        scan[str(X)] = {**b, "median_shift_vs_21p6_pct": float(100.0 * (om_map / om_ad_map - 1.0)),
                        "fraction_of_omega_above_21p6": float(above),
                        "fraction_above_21p6_samples_q16_50_84": (np.percentile(omega_grid(fb, lo, N_b, dN_b, K, 21.6, X) / om, [16, 50, 84]).tolist() if X > 21.6 else [0.0, 0.0, 0.0]),
                        "width68_ratio_vs_21p6": float((b["q84"] - b["q16"]) / (b_ad["q84"] - b_ad["q16"])),
                        "added_region_status": [s["status"] for s in status[:5] if "region" in s and s["region"][0] >= 21.5 and s["region"][0] < X] if X > 21.6 else []}
    # contribution profile of the MAP by 0.1-dex bin over [20.3, top)
    prof = {"logN_lo": lo[(lo >= 20.3 - 1e-9)].tolist(),
            "omega_fraction_MAP": (omega_grid(fmap[None].repeat(1, 0), lo, N_b, dN_b, K, 20.3, top_grid + 1e-6)[0] and
                                   ((fmap * N_b * dN_b)[lo >= 20.3 - 1e-9] * K / om_open_map).tolist()),
            "dndx_fraction_MAP": ((fmap * dN_b)[lo >= 20.3 - 1e-9] / (fmap * dN_b)[lo >= 20.3 - 1e-9].sum()).tolist()}

    # ---- tail continuations above the adopted cut ----
    cut, top = ADOPTED[1], top_grid
    models = {}
    models["M0_hard_cut_adopted"] = {"omega_tail_MAP": 0.0, "omega_total": b_ad, "status": "adopted convention [20.3, 21.6]"}
    t1_map = float(omega_grid(fmap, lo, N_b, dN_b, K, cut, top + 1e-6))
    t1 = omega_grid(fb, lo, N_b, dN_b, K, cut, top + 1e-6)
    models["M1_bh_model_continuation_to_%.1f" % top] = {
        "omega_tail_MAP": t1_map, "omega_tail_band": band(t1, t1_map, recenter),
        "omega_total": band(om_ad + t1, om_ad_map + t1_map, recenter),
        "total_shift_vs_adopted_pct": float(100.0 * t1_map / om_ad_map),
        "status": "the artifact's own open-topped value; added region [21.6, 22.0) thinly H2-calibrated (k/n = 22/26), [22.0, 22.4) frozen-mock completeness (extrapolated)"}
    for name, ds in (("M2_powerlaw_fitted_slope", 0.0), ("M2_powerlaw_slope_minus_sigma", -SIGMA_SLOPE), ("M2_powerlaw_slope_plus_sigma", +SIGMA_SLOPE)):
        tm, sm, im = powerlaw_tail(fmap, lo, hi, N_b, dN_b, K, cut, top, ds)
        ts, ss, isamp = powerlaw_tail(fb, lo, hi, N_b, dN_b, K, cut, top, ds)
        models[name] = {"omega_tail_MAP": float(tm), "slope_MAP_dlog10fN_dlogN": float(sm),
                        "slope_samples_q16_50_84": np.percentile(ss, [16, 50, 84]).tolist(),
                        "omega_tail_band": band(ts, float(tm), recenter),
                        "omega_total": band(om_ad + ts, om_ad_map + float(tm), recenter),
                        "total_shift_vs_adopted_pct": float(100.0 * tm / om_ad_map),
                        "extra_beyond_%.1f_to_infinity_MAP" % top: (float(im) if np.isfinite(im) else "diverges (slope >= -2)"),
                        "status": f"power law anchored at the last in-data bin below {cut}, slope fitted over [{cut - FIT_DEX:.1f}, {cut}) {'+' if ds > 0 else ''}{ds:g}; the +-{SIGMA_SLOPE} bracket is the repo's predeclared FIX-2 slope prior (HBIConfig.omega_slope_extrap_sigma)"}
    cddf = np.load(a.paper_cddf_npz, allow_pickle=True)
    pp = {s.split("=")[0]: s.split("=")[1] for s in cddf["ref_spine_params"].tolist() if "=" in s and "err" not in s and "cosmology" not in s}
    params = (10.0 ** float(pp["log_k2"]), 10.0 ** float(pp["log_Ngamma"]), float(pp["alpha2"]))
    tm, am, im = phw05_tail(fmap, lo, hi, N_b, dN_b, K, cut, top, params)
    ts, asamp, isamp = phw05_tail(fb, lo, hi, N_b, dN_b, K, cut, top, params)
    models["M3_phw05_gamma_reference_tail"] = {
        "omega_tail_MAP": float(tm), "amplitude_match_MAP": float(am), "omega_tail_band": band(ts, float(tm), recenter),
        "omega_total": band(om_ad + ts, om_ad_map + float(tm), recenter),
        "total_shift_vs_adopted_pct": float(100.0 * tm / om_ad_map),
        "extra_beyond_%.1f_to_infinity_MAP" % top: float(im),
        "params": {"log_k2": pp["log_k2"], "log_Ngamma": pp["log_Ngamma"], "alpha2": pp["alpha2"], "citation": str(cddf["ref_spine_citation"][0])},
        "status": "analytic reference tail (PHW05 Gamma function, the paper's reference spine), amplitude-matched over the last 0.5 dex below the cut; literature shape, not our measurement"}
    ne, fz, ratio = lowz_tail_shape(a.lowz_draws, a.lowz_pack)
    # BH amplitude at [21.3, 21.5) in f per dex: sum f_N dN over the fine bins / 0.2
    m_amp = (lo >= 21.3 - 1e-9) & (hi <= 21.5 + 1e-9)
    amp_map = (fmap[m_amp] * dN_b[m_amp]).sum() / 0.2
    amp_s = (fb[:, m_amp] * dN_b[m_amp]).sum(axis=1) / 0.2
    # latent-bin Omega weights of the low-z grid over [21.6, top]
    wb = np.zeros(len(ne) - 1)
    for i, (a0, b0) in enumerate(zip(ne[:-1], ne[1:])):
        o = max(0.0, min(b0, top) - max(a0, cut))
        if o > 0:
            a2, b2 = max(a0, cut), min(b0, top)
            wb[i] = (10.0 ** b2 - 10.0 ** a2) / math.log(10.0)
    tail_shape = ratio @ wb                                          # (D_lowz,) sum_b S_b w_b
    om_tail_lowz_map = K * amp_map * np.median(tail_shape)
    # combine BH amplitude samples with low-z shape draws (independent; pair by index modulo)
    Dl = tail_shape.size
    ts = K * amp_s * tail_shape[np.arange(fb.shape[0]) % Dl]
    models["M4_paper1_lowz_frozen_tail_shape"] = {
        "omega_tail_MAP": float(om_tail_lowz_map), "omega_tail_band": band(ts, float(om_tail_lowz_map), recenter),
        "omega_total": band(om_ad + ts, om_ad_map + float(om_tail_lowz_map), recenter),
        "total_shift_vs_adopted_pct": float(100.0 * om_tail_lowz_map / om_ad_map),
        "lowz_shape_ratio_to_21p3_21p5_median": {f"[{ne[i]:.1f},{ne[i+1]:.1f})": float(np.median(ratio[:, i])) for i in range(len(ne) - 1) if ne[i] >= 21.3 - 1e-9},
        "status": "Paper-1's own frozen low-z f(N) tail (tiers: response-held to 22.1, ceiling-adjacent [22.1,22.4); mock-recovery envelope supported only to 21.9), transported in SHAPE to the BH amplitude at [21.3,21.5); assumes no shape evolution 2 < z < 5"}

    # ---- NaN-SNR closure (A0 vs A1) ----
    nan_closure = None
    if a.dump_finite_snr and a.bh_finite_json:
        Zf = np.load(a.dump_finite_snr, allow_pickle=True)
        bf = json.load(open(a.bh_finite_json))
        fmf = np.asarray(Zf["map_fb"], float)
        fbf = np.asarray(Zf["fb_samp"], float)
        rec = {}
        for label, (l0, l1) in {"omega_adopted_[20.3,21.6]": ADOPTED, "omega_open_[20.3,%.1f)" % top: (20.3, top + 1e-6)}.items():
            m0, m1 = float(omega_grid(fmap, lo, N_b, dN_b, K, l0, l1)), float(omega_grid(fmf, lo, N_b, dN_b, K, l0, l1))
            b0, b1 = band(omega_grid(fb, lo, N_b, dN_b, K, l0, l1), m0, recenter), band(omega_grid(fbf, lo, N_b, dN_b, K, l0, l1), m1, recenter)
            rec[label] = {"MAP_ofrecord": m0, "MAP_finite_snr": m1, "rel_shift_pct": float(100.0 * (m1 / m0 - 1.0)),
                          "hw68_ofrecord_pct": b0["halfwidth68_pct"], "hw68_finite_snr_pct": b1["halfwidth68_pct"]}
        for thr in ("20.3", "20.0"):
            d0, d1 = bh["measurement"][thr]["dndx"]["integrated"], bf["measurement"][thr]["dndx"]["integrated"]
            rec[f"dndx_ge{thr}"] = {k: [d0[k], d1[k]] for k in ("MAP", "q16", "q84", "q025", "q975")}
            rec[f"dndx_ge{thr}"]["MAP_rel_shift_pct"] = float(100.0 * (d1["MAP"] / d0["MAP"] - 1.0))
            o0, o1 = bh["measurement"][thr]["omega"]["integrated"], bf["measurement"][thr]["omega"]["integrated"]
            rec[f"omega_open_ge{thr}_artifact_vs_finite"] = {k: [o0[k], o1[k]] for k in ("MAP", "q16", "q84")}
        l0, l1 = bh["metadata"]["calibration"]["loa0"], bf["metadata"]["calibration"]["loa0"]
        rec["loa0_fp_volume_scale"] = {"ofrecord": l0, "finite_snr": l1, "rel_change_pct": float(100.0 * (l1["vol_scale"] / l0["vol_scale"] - 1.0))}
        rec["n_op_sl"] = [bh["metadata"]["n_op_sl"], bf["metadata"]["n_op_sl"]]
        rec["n_op_detections"] = [bh["metadata"]["n_op_detections"], bf["metadata"]["n_op_detections"]]
        rec["X_tot_per_subbin"] = [np.asarray(Z["X_tot"], float).tolist(), np.asarray(Zf["X_tot"], float).tolist()]
        rec["max_abs_rel_diff_map_fN_over_[20.3,22.4)"] = float(np.nanmax(np.abs(fmf[lo >= 20.3 - 1e-9] / fmap[lo >= 20.3 - 1e-9] - 1.0)))
        rec["inputs"] = {"dump_finite_snr": {"path": a.dump_finite_snr, "sha256": _sha(a.dump_finite_snr)}, "bh_finite_json": {"path": a.bh_finite_json, "sha256": _sha(a.bh_finite_json)}}
        nan_closure = rec

    # ---- Qz5 convention comparison ----
    rho = QZ5["rho_DLA_1e8_Msun_Mpc3"]
    qz5_omega = {k: rho[k] * 1e8 / RHO_CRIT_MSUN_MPC3 for k in ("value", "lo", "hi")}
    rhoHI = QZ5["rho_HI_1e8_Msun_Mpc3"]
    qz5_omega_HI = {k: rhoHI[k] * 1e8 / RHO_CRIT_MSUN_MPC3 for k in ("value", "lo", "hi")}
    Nq = 10.0 ** np.asarray(QZ5["observed_DLA_logN"], float)
    share = (Nq / Nq.sum()).tolist()
    qmax = float(max(QZ5["observed_DLA_logN"]))
    matched = {}
    for X in (21.1, 21.2, 21.3, 21.6, top):
        om_map = float(omega_grid(fmap, lo, N_b, dN_b, K, 20.3, X + (1e-6 if X == top else 0.0)))
        b = band(omega_grid(fb, lo, N_b, dN_b, K, 20.3, X + (1e-6 if X == top else 0.0)), om_map, recenter)
        matched["bh_[20.3,%s)" % ("%.1f" % X)] = {"MAP": om_map, "q16": b["q16"], "q84": b["q84"],
                                                   "ratio_to_qz5_omega_DLA": float(om_map / qz5_omega["value"]),
                                                   "ratio_range_using_qz5_bounds": [float(om_map / qz5_omega["hi"]), float(om_map / qz5_omega["lo"])]}
    qz5_block = {
        "source": QZ5,
        "rho_crit_Msun_Mpc3_h0p7": RHO_CRIT_MSUN_MPC3,
        "qz5_omega_DLA_(rho_DLA/rho_crit, h=0.7)": qz5_omega,
        "qz5_omega_HI_incl_subDLA": qz5_omega_HI,
        "qz5_single_system_mass_shares": dict(zip([str(x) for x in QZ5["observed_DLA_logN"]], share)),
        "qz5_highest_system_share_of_rho_DLA": float(max(share)),
        "qz5_effective_upper_limit_logN": qmax,
        "bh_under_matched_conventions": matched,
        "dndx_ratio_for_reference": {"bh_MAP": art_d["MAP"], "qz5_lX": 0.034, "ratio": float(art_d["MAP"] / 0.034)},
        "convention_verdict": ("Qz5's rho_DLA is a Poisson-bootstrapped sum over five observed systems (max log N = %.1f) with N_HI errors only: it contains NO tail above its highest observed column and NO model/extrapolation term. "
                               "Our BH Omega is an f(N)-integral over a fitted model with completeness/response corrections; its published open-topped form reaches %.1f and the adopted convention closes at 21.6. "
                               "A direct sigma-level tension statement between the two Omega values is therefore NOT defined under a common high-N_HI convention; the comparable object is our Omega closed at the top of their observed range (bh_under_matched_conventions), and even that compares a corrected f(N)-integral with a five-system sum." % (qmax, top)),
        "omega_m_note": "Qz5 uses Omega_m = 0.3; Paper 1 uses 0.279 -- dX/dz differs by ~1-2 % at z ~ 5; not applied (below every other term here)",
    }

    out = {
        "role": "R-037 extension (PI 2026-08-28): BH / high-z arm Omega_DLA under explicit upper-integration conventions and tail continuations, NaN-SNR closure, and the Qz5 Omega convention comparison -- referee-facing DIAGNOSTIC; sits BESIDE the frozen products; the adopted Paper-1 Omega convention [20.3, 21.6] and the diagnostic-only status of the high-z Omega (PI #45) are unchanged",
        "status": "diagnostic",
        "written_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"module": "CDDF_analysis/hbi_mcmc/bh_omega_tail.py", "commit": _git_commit(), "argv": sys.argv, "python": sys.version.split()[0], "numpy": np.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV")},
        "inputs": {k: {"path": p, "sha256": _sha(p)} for k, p in [("dump", a.dump), ("bh_artifact", a.bh_artifact), ("lowz_draws", a.lowz_draws), ("lowz_pack", a.lowz_pack), ("paper_cddf_npz", a.paper_cddf_npz)]},
        "conventions": {"omega_definition": "K * sum N_b f_b dN_b on the BH fine 0.1-dex grid (K = H0 m_p / (c rho_crit), H0 = 70, proton mass, no helium), MAP = forward estimator, band = recentred percentile band of the joint-MC f(N) samples (band_recenter=%s), n_mc = %d, seed = %d" % (recenter, int(Z["n_mc"]), int(Z["seed"])),
                        "adopted_interval": list(ADOPTED), "fine_grid_top": top_grid, "quantile_levels": QUANTILES, "no_quadrature": True,
                        "selection_calibration_contract": "P1_PRIMARY_LYA; CANONICAL_PURITY_COMPLETENESS_CONTRACT v1; h2cal loa0 lya gap_c 0.496; z_QSO (4.25, 7.0); collar 3000 km/s; reported bin [3.8, 5.0)",
                        "what_the_artifact_integrates": "OPEN-TOPPED [20.3, %.1f) -- the paper-facing npz key highz_omega_20p3 carries this value although it sits beside omega_limits = [20.3, 21.6]" % top_grid},
        "closure": closure,
        "calibration_status_by_region": status,
        "adopted_bh_omega_[20.3,21.6]": b_ad,
        "upper_limit_scan": scan,
        "map_contribution_profile": prof,
        "tail_continuations_above_21p6": models,
        "nan_snr_closure": nan_closure,
        "qz5_comparison": qz5_block,
    }
    jp = os.path.join(a.out_dir, "R037ext_bh_omega_tail.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    np.savez(os.path.join(a.out_dir, "R037ext_bh_omega_tail.npz"),
             axis_note=np.array(["upper_limit_scan_q: (N_max order as nmax, quantile 2.5/16/50/84/97.5 recentred on MAP); tail_model_total_q: (model order as tail_models, same quantiles)"]),
             nmax=np.array(NMAX_SCAN), upper_limit_scan_MAP=np.array([scan[str(X)]["MAP"] for X in NMAX_SCAN]),
             upper_limit_scan_q=np.array([[scan[str(X)][k] for k in ("q025", "q16", "q50_recentred", "q84", "q975")] for X in NMAX_SCAN]),
             tail_models=np.array(list(models.keys())),
             tail_model_total_MAP=np.array([models[m]["omega_total"]["MAP"] for m in models]),
             tail_model_total_q=np.array([[models[m]["omega_total"][k] for k in ("q025", "q16", "q50_recentred", "q84", "q975")] for m in models]),
             profile_logN_lo=np.array(prof["logN_lo"]), profile_omega_fraction_MAP=np.array(prof["omega_fraction_MAP"]))
    shas = {os.path.basename(p): _sha(p) for p in (jp, os.path.join(a.out_dir, "R037ext_bh_omega_tail.npz"))}
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for k, v in shas.items():
            fh.write(f"{v}  {k}\n")
    print(json.dumps(shas, indent=1))
    print(f"adopted [20.3,21.6] MAP {b_ad['MAP']:.4e} [{b_ad['q16']:.4e},{b_ad['q84']:.4e}] hw68 {b_ad['halfwidth68_pct']:.2f}%")
    for X in NMAX_SCAN:
        s = scan[str(X)]
        print(f"  Nmax {X}: MAP {s['MAP']:.4e} shift {s['median_shift_vs_21p6_pct']:+6.1f}%  frac>21.6 {s['fraction_of_omega_above_21p6']:.3f}  hw68 {s['halfwidth68_pct']:.2f}%  width68 ratio {s['width68_ratio_vs_21p6']:.2f}")
    for m, r in models.items():
        print(f"  {m:40s} tail MAP {r['omega_tail_MAP']:.3e}  total shift {r.get('total_shift_vs_adopted_pct', 0):+6.1f}%  hw68 {r['omega_total']['halfwidth68_pct']:.2f}%")
    if nan_closure:
        print("  NaN-SNR:", json.dumps({k: v for k, v in nan_closure.items() if 'rel' in k or k in ('n_op_sl',)}, indent=None)[:600])
    print("  Qz5:", json.dumps(qz5_block["qz5_omega_DLA_(rho_DLA/rho_crit, h=0.7)"]), {k: round(v["ratio_to_qz5_omega_DLA"], 2) for k, v in matched.items()})


if __name__ == "__main__":
    main()
