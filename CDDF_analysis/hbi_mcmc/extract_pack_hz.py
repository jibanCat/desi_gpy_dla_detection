#!/usr/bin/env python
"""extract_pack_hz.py — HIGH-z CALIBRATION-ARM pack for the production HBI architecture (validation branch
validation/hz-hbi-extension-2026-09; PI ruling 2026-09-02 §15-§20; predeclaration
MAX4_HZ_HBI_EXTENSION_PREDECLARATION_2026-09-02.md + Amendment 1). DIAGNOSTIC / CLOSURE EXPERIMENT ONLY.

Every calibration array is sourced from the HIGH-z arm (never from a low-z injection product):
  counts, dX      : the MAX4 P0 real catalogue on the frozen 2356-sightline population (op cut DLAFLAG 0, P_DLA > 0.99,
                    SNR > 2, inside the Lya-only windows), fine z 3.8-5.0 (0.1), coarse blocks 3.8/4.25/4.5/5.0;
  completeness    : the MAX4 fiducial calibration (A_shared real-spectrum injections; analysis_fid_MAX4.json) on the
                    analyzer's 7 molly cells x 5 S/N strata -> the 8 molly SNR rows;
  g(N, z)         : the same calibration per injection z bin, occupancy-weighted mean 1 per row;
  response        : per-cell skew-normal fit (fit_forward_response) to the injection truth-matches (N_true, S/N, z_dla, N-hat - N)
                    + its own phi_ref + 96 sightline-bootstrap carriers (Amendment 1);
  false positives : the HCD-free 2LPT loa-0 random arm under MAX4 at the emulated high z (extra accepted rows in window);
  t_sigma         : high-z-specific (P1 recipe test |ln 1.024| floored at 0.10 in every block).
--mode real  : the real high-z pack (truth_counts = ZEROS sentinel, sidecar real_data true).
--mode mock  : the OPERATOR-CLOSURE pack on the 2LPT random arm itself (counts from its MAX4 outputs, truth_counts from its injections,
               C / g / response from the SAME arm's analysis; FP from the same arm) for cc_posterior_validation.
Real-LOA values land on scratch only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

ROOT_MAX4 = "/scratch/cavestru_root/cavestru0/mfho/r041_max4_highz_2026-09"
ROOT_R041 = "/scratch/cavestru_root/cavestru0/mfho/r041_highz_repair_2026-08-28"
LYA = 1215.67
C_KMS = 299792.458
NHAT_EDGES = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 3)          # 29 bins (as low z)
ZF_EDGES = np.round(np.arange(3.8, 5.0 + 1e-9, 0.1), 3)              # 12 fine bins (high z)
ZC_EDGES = np.array([3.8, 4.2, 4.5, 5.0])                            # 3 coarse blocks ON the fine grid (schema rule; 4.25 is not a fine edge -> 4.2; predeclaration Amendment 2)
SNR_EDGES = np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf])       # molly rows (as low z)
MOLLY_EDGES = np.array([19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf])
STRATA = [2.0, 3.0, 4.0, 5.0, 7.0, np.inf]
ZBINS_INJ = [3.8, 4.25, 4.5, 5.0]
RESP_SNR_EDGES = np.array([2.0, 3.5, 6.5, np.inf])
T_SIGMA_FLOOR = 0.10


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _pack_mod():
    """pack.py loaded file-directly (jax-free), as extract_pack does."""
    spec = importlib.util.spec_from_file_location("_modelA_pack_hz", os.path.join(_HERE, "pack.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m


FROZEN_LOWZ_PACK = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz"


def basis_pad_edges_19p0_0p2():
    """The low-z latent basis (extract_pack.basis_pad_edges(19.0, 0.2)) taken VERBATIM from the frozen low-z real pack's ntrue_edges
    (the shared architecture's latent support; read-only; no low-z calibration array is read)."""
    return np.asarray(np.load(FROZEN_LOWZ_PACK)["ntrue_edges"], float)


def load_forward_response_pack_local(path):
    """The extract_pack.load_forward_response_pack key mapping, re-stated locally (same keys, same ramp-width rule)."""
    d = np.load(path, allow_pickle=True)
    kind = str(np.asarray(d["_fwd_response_kind"]))
    assert "skewnormal" in kind, kind
    ramp_width = float(d["N_skew_ramp_width"]) if "N_skew_ramp_width" in d.files else 0.5
    fwd = dict(resp_mu_coef=np.asarray(d["mu_coef"], float), resp_sig_coef=np.asarray(d["sig_coef"], float), resp_skew_coef=np.asarray(d["skew_coef"], float),
               resp_snr_edges=np.asarray(d["snr_edges"], float), resp_z_edges=np.asarray(d["z_edges"], float), resp_sig_floor=np.float64(d["sig_floor"]),
               resp_skew_ramp=np.array([float(d["N_skew_collapse"]), ramp_width]), resp_N_ref=np.float64(d["N_ref"]))
    fwd["resp_N_fit_range"] = np.asarray(_pack_mod().resp_fit_range_from_forward_npz(path), float)
    meta = dict(fwd_response_kind=kind, deg_N=int(d["deg_N"]), resp_skew_ramp_width=ramp_width, z_covariate=str(np.asarray(d["z_covariate"])) if "z_covariate" in d.files else "zqso", path=path,
                has_empirical_block=bool("emp_rho" in d.files))
    return fwd, meta


def _idx(edges, x):
    return np.searchsorted(edges, x, side="right") - 1


def stratum_of_row(r):
    """molly SNR row (0..7) -> R-041 stratum index (rows below SNR 2 -> stratum 0, inert: zero exposure)."""
    centre = 0.5 * (SNR_EDGES[r] + (SNR_EDGES[r + 1] if np.isfinite(SNR_EDGES[r + 1]) else SNR_EDGES[r] + 2.0))
    return max(0, int(np.digitize(centre, STRATA) - 1))


def dX_from_windows(pop_rows, snr_of, Om=0.279):
    def dXdz(z): return (1 + z) ** 2 / np.sqrt(Om * (1 + z) ** 3 + 1 - Om)
    def X(a, b):
        g = np.linspace(a, b, 200); return float(np.trapezoid(dXdz(g), g))
    dX = np.zeros((len(ZF_EDGES) - 1, len(SNR_EDGES) - 1)); n_sl = 0; xc = np.zeros(len(ZC_EDGES) - 1)
    for r in pop_rows:
        s = snr_of(r)
        if not (np.isfinite(s) and s > 2.0):
            continue
        n_sl += 1; row = int(np.clip(_idx(SNR_EDGES, s), 0, len(SNR_EDGES) - 2)); lo, hi = float(r["zlo"]), float(r["zhi"])
        for k in range(len(ZF_EDGES) - 1):
            a, b = max(lo, ZF_EDGES[k]), min(hi, ZF_EDGES[k + 1])
            if b > a: dX[k, row] += X(a, b)
        for K in range(len(ZC_EDGES) - 1):
            a, b = max(lo, ZC_EDGES[K]), min(hi, ZC_EDGES[K + 1])
            if b > a: xc[K] += X(a, b)
    return dX, xc, n_sl


def counts_from_rows(tid, z, N, P, flag, snr, win):
    ok = np.array([win.get(int(t), (np.inf, -np.inf))[0] <= zz <= win.get(int(t), (np.inf, -np.inf))[1] for t, zz in zip(tid, z)])
    op = ok & (flag == 0) & (P > 0.99) & (snr > 2.0) & np.isfinite(N)
    c = _idx(NHAT_EDGES, N); k = _idx(ZF_EDGES, z); s = np.clip(_idx(SNR_EDGES, snr), 0, len(SNR_EDGES) - 2)
    keep = op & (c >= 0) & (c < len(NHAT_EDGES) - 1) & (k >= 0) & (k < len(ZF_EDGES) - 1)
    counts = np.zeros((len(NHAT_EDGES) - 1, len(ZF_EDGES) - 1, len(SNR_EDGES) - 1), np.int64)
    np.add.at(counts, (c[keep], k[keep], s[keep]), 1)
    return counts, int(op.sum()), int(keep.sum())


def molly_blocks(analysis_json):
    """(molly_n_det, molly_n_tot) (8 rows x 7 cells) and g_grid / occupancy (7 x 12) from the analyzer tables."""
    A = json.load(open(analysis_json))["tables"]
    cells = sorted({(float(c["key"]["n_lo"]), c["key"]["n_hi"]) for c in A["per_molly_cell_x_stratum"]}, key=lambda x: x[0])
    M = len(cells); cell_idx = {lo: i for i, (lo, hi) in enumerate(cells)}
    kn = np.zeros((5, M)); nn = np.zeros((5, M))
    for c in A["per_molly_cell_x_stratum"]:
        kn[int(c["key"]["stratum"]), cell_idx[float(c["key"]["n_lo"])]] += c["k"]; nn[int(c["key"]["stratum"]), cell_idx[float(c["key"]["n_lo"])]] += c["n"]
    n_det = np.zeros((8, M)); n_tot = np.zeros((8, M))
    for r in range(8):
        s = stratum_of_row(r); n_det[r] = kn[s]; n_tot[r] = nn[s]
    # g(N, z): per cell, C(zbin) / occupancy-weighted mean over zbins; held over the fine bins of each zbin
    kz = np.zeros((M, 3)); nz = np.zeros((M, 3))
    for c in A["per_molly_cell_x_zbin_x_stratum"]:
        i = cell_idx[float(c["key"]["n_lo"])]; zb = int(c["key"]["zbin"]); kz[i, zb] += c["k"]; nz[i, zb] += c["n"]
    Cz = np.where(nz > 0, kz / np.maximum(nz, 1), np.nan)
    Kf = len(ZF_EDGES) - 1; g = np.ones((M, Kf)); occ = np.zeros((M, Kf))
    zmid = 0.5 * (ZF_EDGES[:-1] + ZF_EDGES[1:]); kzb = np.clip(np.digitize(zmid, ZBINS_INJ) - 1, 0, 2)
    for i in range(M):
        w = nz[i] / max(nz[i].sum(), 1.0); cbar = np.nansum(np.where(np.isnan(Cz[i]), 0.0, Cz[i]) * w)
        for k in range(Kf):
            zb = kzb[k]; g[i, k] = (Cz[i, zb] / cbar) if (cbar > 0 and np.isfinite(Cz[i, zb])) else 1.0
            occ[i, k] = nz[i, zb] / max(int((kzb == zb).sum()), 1)
    return n_det, n_tot, np.array([c[0] for c in cells] + [np.inf]), g, occ


def response_fit(inj_rows, out_npz, n_boot=96, seed=20260907):
    """Per-cell skew-normal forward response from the injection truth-matches (+ bootstrap carriers)."""
    from CDDF_analysis.hbi.znz_kernel import (fit_forward_response, save_forward_response,
                                              build_forward_response_fit_resample, refit_forward_response_from_resample)
    det = [r for r in inj_rows if r["detected"] == "True" and r["nhat"] not in ("", "nan")]
    N_true = np.array([float(r["logN"]) for r in det]); nhat = np.array([float(r["nhat"]) for r in det])
    snr = np.array([float(r["snr"]) for r in det]); zd = np.array([float(r["z_inj"]) for r in det]); tids = np.array([int(r["TARGETID"]) for r in det])
    meas = dict(N_true=N_true, snr=snr, zqso=zd, dx=nhat - N_true, xhat=nhat, z_covariate="zdla")
    frm = fit_forward_response(meas, snr_edges=RESP_SNR_EDGES, z_edges=np.array(ZC_EDGES), deg_N=2, n_N_cells=7, min_count=40,
                               N_skew_collapse=21.0, N_skew_ramp_width=0.5, N_ref=None, build_empirical=True)
    frm.z_covariate = "zdla"
    save_forward_response(out_npz, frm)
    uniq = np.unique(tids)
    rfr = build_forward_response_fit_resample(meas, tids, uniq, frm, n_N_cells=7, min_count=40, build_empirical=False)
    rng = np.random.default_rng(seed); mus, sigs, sks = [], [], []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        fr = refit_forward_response_from_resample(rfr, mult); mus.append(fr.mu_coef); sigs.append(fr.sig_coef); sks.append(fr.skew_coef)
    return frm, np.stack(mus), np.stack(sigs), np.stack(sks), len(det), len(uniq)


def fp_block(mock_pi_csv, mock_pop_csv):
    pop = {int(r["TARGETID"]): r for r in csv.DictReader(open(mock_pop_csv))}
    fp = np.zeros((len(NHAT_EDGES) - 1, len(SNR_EDGES) - 1), np.int64); n_sl = 0; n_extra = 0
    for r in csv.DictReader(open(mock_pi_csv)):
        n_sl += 1; p = pop[int(r["TARGETID"])]; zlo, zhi = float(p["zlo"]), float(p["zhi"]); zi = float(r["z_inj"]); s = float(r["snr"])
        zs = [float(x) for x in r["accepted_zs"].split(";") if x] if r["accepted_zs"] else []; ns = [float(x) for x in r["accepted_nhats"].split(";") if x] if r["accepted_nhats"] else []
        for z, n in zip(zs, ns):
            if abs(z - zi) / (1 + zi) <= 0.01 or not (zlo <= z <= zhi):
                continue
            n_extra += 1; c = _idx(NHAT_EDGES, n); srow = int(np.clip(_idx(SNR_EDGES, s), 0, len(SNR_EDGES) - 2))
            if 0 <= c < len(NHAT_EDGES) - 1: fp[c, srow] += 1
    return fp, n_sl, n_extra


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "mock", "mock_native"], required=True,
                    help="mock = operator closure on the injection arm itself (comb truth; diagnostic); mock_native = end-to-end closure: population = the 2LPT "
                         "native loa-124 arm (smooth mock truth), calibration = the random injection arm on the same substrate (predeclaration Amendment 3)")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--n-boot", type=int, default=96)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    fid_json = f"{ROOT_MAX4}/fid_max4/analysis/analysis_fid_MAX4.json"; fid_pi = f"{ROOT_MAX4}/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv"
    cmp_pi = f"{ROOT_MAX4}/gate_prescription/analysis_cmpA_MAX4_per_injection.csv"
    mock_pi = f"{ROOT_MAX4}/p1/reductions/analysis_mock_2lpt_random_MAX4_per_injection.csv"; mock_pop = f"{ROOT_MAX4}/p1/mock/truth_r041/2lpt_random_population.csv"
    mock_json = f"{ROOT_MAX4}/p1/reductions/analysis_mock_2lpt_random_MAX4.json"
    inputs = {}
    if a.mode == "real":
        from astropy.io import fits
        pop_csv = f"{ROOT_R041}/population/r041_population.csv"; cat_path = f"{ROOT_MAX4}/real/combined/dlacat-loa-hz-MAX4-v1.fits"
        pop = list(csv.DictReader(open(pop_csv))); win = {int(r["TARGETID"]): (float(r["zlo"]), float(r["zhi"])) for r in pop}
        d = fits.open(cat_path)[1].data
        counts, n_op, n_binned = counts_from_rows(d["TARGETID"].astype(np.int64), np.asarray(d["Z_DLA"], float), np.asarray(d["NHI"], float),
                                                  np.asarray(d["P_DLA"], float), np.asarray(d["DLAFLAG"], int), np.asarray(d["SNR_REDSIDE"], float), win)
        dX, xc, n_sl = dX_from_windows(pop, lambda r: float(r["snr"]))
        n_det, n_tot, medges, g, occ = molly_blocks(fid_json)
        inj = list(csv.DictReader(open(fid_pi))) + list(csv.DictReader(open(cmp_pi)))
        frm, cm, cs, ck, n_tp, n_uniq = response_fit(inj, os.path.join(a.out_dir, "forward_response_hz_real_arm.npz"), a.n_boot)
        fp, n_sl_mock, n_extra = fp_block(mock_pi, mock_pop)
        truth_counts = None
        inputs = dict(population=pop_csv, population_sha256=_sha(pop_csv), catalogue=cat_path, catalogue_sha256=_sha(cat_path), fid_analysis=fid_json, fid_analysis_sha256=_sha(fid_json),
                      fid_per_injection=fid_pi, fid_per_injection_sha256=_sha(fid_pi), cmp_per_injection=cmp_pi, cmp_per_injection_sha256=_sha(cmp_pi),
                      fp_mock_per_injection=mock_pi, fp_mock_per_injection_sha256=_sha(mock_pi), fp_mock_population=mock_pop)
    elif a.mode == "mock_native":
        # END-TO-END closure (Amendment 3): population = the P1 native loa-124 arm (1,028 sightlines: 513 multi + 515 single, native truth
        # 1,594 absorbers, emulated z = z + 1); counts from its MAX4 outputs; calibration (C, g, response) and FP from the RANDOM injection arm on the
        # same substrate (the injection calibration arm) — the same separation between population and calibration as the real analysis.
        import glob
        from astropy.io import fits
        nat_root = f"{ROOT_MAX4}/p1/mock_native/2lpt"
        pop = list(csv.DictReader(open(f"{nat_root}/population_native.csv")))
        for r in pop: r["zlo"] = str(float(r["zlo"]) + 1.0); r["zhi"] = str(float(r["zhi"]) + 1.0)
        win = {int(r["TARGETID"]): (float(r["zlo"]), float(r["zhi"])) for r in pop}
        rows = [fits.open(f)[1].data for f in sorted(glob.glob(f"{nat_root}/native_outputs/dlacat-*.fits"))]
        tid = np.concatenate([np.asarray(r["TARGETID"], np.int64) for r in rows]); z = np.concatenate([np.asarray(r["Z_DLA"], float) for r in rows]) + 1.0
        N = np.concatenate([np.asarray(r["NHI"], float) for r in rows]); P = np.concatenate([np.asarray(r["P_DLA"], float) for r in rows])
        fl = np.concatenate([np.asarray(r["DLAFLAG"], int) for r in rows]); snr = np.concatenate([np.asarray(r["SNR_REDSIDE"], float) for r in rows])
        counts, n_op, n_binned = counts_from_rows(tid, z, N, P, fl, snr, win)
        dX, xc, n_sl = dX_from_windows(pop, lambda r: float(r["snr"]))
        n_det, n_tot, medges, g, occ = molly_blocks(mock_json)                     # calibration arm = the random injection arm
        inj = list(csv.DictReader(open(mock_pi)))
        for r in inj: r["z_inj"] = str(float(r["z_inj"]) + 1.0)
        frm, cm, cs, ck, n_tp, n_uniq = response_fit(inj, os.path.join(a.out_dir, "forward_response_hz_mocknative.npz"), a.n_boot)
        fp, n_sl_mock, n_extra = fp_block(mock_pi, mock_pop)
        ntrue_edges_tmp = basis_pad_edges_19p0_0p2()
        truth_counts = np.zeros((len(ntrue_edges_tmp) - 1, len(ZF_EDGES) - 1), np.int64)
        for r in csv.DictReader(open(f"{nat_root}/native/native_truth.csv")):
            zt = float(r["z_inj"]) + 1.0; b = _idx(ntrue_edges_tmp, float(r["logN"])); k = _idx(ZF_EDGES, zt)
            if 0 <= b < truth_counts.shape[0] and 0 <= k < truth_counts.shape[1]: truth_counts[b, k] += 1
        inputs = dict(native_outputs=f"{nat_root}/native_outputs", native_truth=f"{nat_root}/native/native_truth.csv", native_population=f"{nat_root}/population_native.csv",
                      calibration_arm_analysis=mock_json, calibration_arm_per_injection=mock_pi, fp_mock_population=mock_pop)
    else:
        # operator closure on the 2LPT random arm: counts from its own MAX4 outputs; truth = its injections (per fine z bin: emulated z = z_inj + 1)
        import glob
        from astropy.io import fits
        pop = list(csv.DictReader(open(mock_pop))); win = {int(r["TARGETID"]): (float(r["zlo"]) + 1.0, float(r["zhi"]) + 1.0) for r in pop}
        for r in pop: r["zlo"] = str(float(r["zlo"]) + 1.0); r["zhi"] = str(float(r["zhi"]) + 1.0)     # emulated high-z windows (delta_z = 1)
        rows = []
        for f in sorted(glob.glob(f"{ROOT_MAX4}/p1/mock/2lpt_random_MAX4_outputs/dlacat-*.fits")):
            rows.append(fits.open(f)[1].data)
        tid = np.concatenate([np.asarray(r["TARGETID"], np.int64) for r in rows]); z = np.concatenate([np.asarray(r["Z_DLA"], float) for r in rows]) + 1.0
        N = np.concatenate([np.asarray(r["NHI"], float) for r in rows]); P = np.concatenate([np.asarray(r["P_DLA"], float) for r in rows])
        fl = np.concatenate([np.asarray(r["DLAFLAG"], int) for r in rows]); snr = np.concatenate([np.asarray(r["SNR_REDSIDE"], float) for r in rows])
        counts, n_op, n_binned = counts_from_rows(tid, z, N, P, fl, snr, win)
        dX, xc, n_sl = dX_from_windows(pop, lambda r: float(r["snr"]))
        n_det, n_tot, medges, g, occ = molly_blocks(mock_json)
        inj = list(csv.DictReader(open(mock_pi)))
        for r in inj: r["z_inj"] = str(float(r["z_inj"]) + 1.0)
        frm, cm, cs, ck, n_tp, n_uniq = response_fit(inj, os.path.join(a.out_dir, "forward_response_hz_mockclosure.npz"), a.n_boot)
        fp, n_sl_mock, n_extra = fp_block(mock_pi, mock_pop)
        ntrue_edges_tmp = basis_pad_edges_19p0_0p2()
        truth_counts = np.zeros((len(ntrue_edges_tmp) - 1, len(ZF_EDGES) - 1), np.int64)
        for r in inj:
            b = _idx(ntrue_edges_tmp, float(r["logN"])); k = _idx(ZF_EDGES, float(r["z_inj"]))
            if 0 <= b < truth_counts.shape[0] and 0 <= k < truth_counts.shape[1]: truth_counts[b, k] += 1
        inputs = dict(mock_outputs=f"{ROOT_MAX4}/p1/mock/2lpt_random_MAX4_outputs", mock_analysis=mock_json, mock_analysis_sha256=_sha(mock_json), mock_per_injection=mock_pi, mock_population=mock_pop)
    # ---- assemble the pack --------------------------------------------------------------------------------------------
    ntrue_edges = basis_pad_edges_19p0_0p2(); B = len(ntrue_edges) - 1
    assert np.allclose(np.load(FROZEN_LOWZ_PACK)['nhat_edges'], NHAT_EDGES), 'nhat grid differs from the frozen low-z pack'
    print('latent basis (from the frozen low-z pack):', ntrue_edges.tolist())
    masked = np.zeros(len(NHAT_EDGES) - 1, bool); masked[(NHAT_EDGES[:-1] >= 19.5 - 1e-9) & (NHAT_EDGES[1:] <= 19.7 + 1e-9)] = True
    zmid = 0.5 * (ZF_EDGES[:-1] + ZF_EDGES[1:]); kz_to_K = (np.searchsorted(ZC_EDGES, zmid, side="right") - 1).astype(np.int64)
    col = dX.sum(axis=0); fp_E = np.zeros_like(dX); nz = col > 0; fp_E[:, nz] = dX[:, nz] / col[nz]
    fp_w = n_sl / n_sl_mock; fp_ell = n_sl_mock * (n_sl_mock / n_sl)
    fwd, fwd_meta = load_forward_response_pack_local(os.path.join(a.out_dir, {"real": "forward_response_hz_real_arm.npz", "mock": "forward_response_hz_mockclosure.npz", "mock_native": "forward_response_hz_mocknative.npz"}[a.mode]))
    pack = dict(nhat_edges=NHAT_EDGES, ntrue_edges=ntrue_edges, zf_edges=ZF_EDGES, zc_edges=ZC_EDGES, kz_to_K=kz_to_K, snr_edges=SNR_EDGES, nhat_masked_bins=masked,
                counts=counts, dX=dX, dX_coarse_committed=xc, molly_n_det=n_det, molly_n_tot=n_tot, molly_nhi_edges=medges, molly_snr_edges=SNR_EDGES,
                g_grid=g, g_occupancy=occ, **fwd, fp_counts=fp, fp_eta_c=np.zeros(len(NHAT_EDGES) - 1), fp_ell_eff=np.float64(fp_ell), fp_w_sightline_ratio=np.float64(fp_w),
                fp_E_alloc=fp_E, t_sigma=np.full(len(ZC_EDGES) - 1, T_SIGMA_FLOOR), truth_counts=(np.zeros((B, len(ZF_EDGES) - 1), np.int64) if truth_counts is None else truth_counts))
    # adopted-contract stamps (Amendment 1): adopted := the direct fit; phi_ref from the same surfaces; carriers = bootstrap refits
    from CDDF_analysis.hbi_mcmc.pack import ModelAPack, save_pack, load_pack
    from CDDF_analysis.hbi_mcmc.count_conserving_fold import surface_masses
    tmp = ModelAPack(**{k: v for k, v in pack.items()})
    _, phi = surface_masses(tmp, tmp.resp_mu_coef, tmp.resp_sig_coef, tmp.resp_skew_coef, np.asarray(tmp.resp_N_fit_range, float), NHAT_EDGES)
    pack.update(tp_convention_id="hz_injection_nearest_dz_0.01", contract_id="hz-trial-2026-09-02", adopted_resp_version="hz_direct_fit_v0",
                adopted_resp_mu_coef=fwd["resp_mu_coef"], adopted_resp_sig_coef=fwd["resp_sig_coef"], adopted_resp_skew_coef=fwd["resp_skew_coef"],
                adopted_resp_fit_range=fwd["resp_N_fit_range"], adopted_phi_ref=phi, adopted_carrier_mu=cm, adopted_carrier_sig=cs, adopted_carrier_skew=ck,
                adopted_carrier_shared3=np.zeros((cm.shape[0], 3)))
    name = {"real": "modelA_pack_HZ_MAX4_real_arm.npz", "mock": "modelA_pack_HZ_mockclosure_2lpt_random.npz", "mock_native": "modelA_pack_HZ_mockclosure_2lpt_native.npz"}[a.mode]
    npz = os.path.join(a.out_dir, name); np.savez(npz, **pack)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
    prov = dict(real_data=(a.mode == "real"), truth_counts_sentinel=("ZEROS_NO_TRUTH" if a.mode == "real" else None), mode=a.mode, arm="HIGH-z CALIBRATION ARM (MAX4 real-spectrum injections)",
                predeclaration="MAX4_HZ_HBI_EXTENSION_PREDECLARATION_2026-09-02.md + Amendment 1", extractor="CDDF_analysis/hbi_mcmc/extract_pack_hz.py", code_commit=commit,
                built=_dt.datetime.now().astimezone().isoformat(), grids=dict(nhat=NHAT_EDGES.tolist(), zf=ZF_EDGES.tolist(), zc=ZC_EDGES.tolist(), molly=medges.tolist(), snr=SNR_EDGES.tolist()),
                sizes=dict(n_op=n_op, n_binned=n_binned, counts_total=int(counts.sum()), n_sl=n_sl, dX_total=float(dX.sum()), n_tp_response=n_tp, n_uniq_sightlines_response=n_uniq,
                           fp_rows=int(fp.sum()), fp_extra_rows_all=n_extra, n_sl_mock=n_sl_mock, fp_w=fp_w, fp_ell_eff=fp_ell),
                conventions=dict(completeness="const_extrap below 19.5 (no sub-floor injections at high z); SNR rows <2 carry stratum-0 values (zero exposure)",
                                 g="C(cell, zbin)/occupancy-weighted mean over the 3 injection z bins; constant across the fine bins of a z bin",
                                 response="fit_forward_response deg_N 2, SNR cells (2,3.5,6.5,inf), z cells = coarse blocks on z_dla ('zdla'); adopted := direct fit; phi_ref from the same surfaces; 96 bootstrap carriers; shared3 = zeros (n/a)",
                                 fp="2LPT loa-0 random arm under MAX4 (HCD-free, emulated high z): extra accepted rows in window; eta_c = 0", t_sigma=f"{T_SIGMA_FLOOR} floor in every block (P1 recipe test |ln 1.024| below the floor)",
                                 path_length="population windows (collar 3000 km/s), Omega_m 0.279"),
                inputs=inputs, forward_meta={k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in fwd_meta.items()}, distinct_from_low_z="no low-z injection product read; the low-z pack is untouched")
    json.dump(prov, open(npz[:-4] + ".provenance.json", "w"), indent=1)
    pk = load_pack(npz, allow_nonstandard_grid=True)
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    consts, Mg = build_cc_tensors(pk)
    print(json.dumps(dict(pack=npz, sha256=_sha(npz), counts_total=int(counts.sum()), n_op=n_op, dX_total=round(float(dX.sum()), 1), n_tp_response=n_tp, fp_rows=int(fp.sum()),
                          Mg_shape=list(np.asarray(Mg).shape), B=B, Kf=len(ZF_EDGES) - 1, phi_ref_min=float(phi.min()), phi_ref_max=float(phi.max())), indent=1))


if __name__ == "__main__":
    main()
