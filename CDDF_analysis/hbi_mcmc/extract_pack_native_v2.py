#!/usr/bin/env python
"""High-z END-TO-END closure packs for the response-estimator rebuild (gate MAX4_RESPONSE_ESTIMATOR_CLOSURE_GATE_2026-09-02.md §4; mock-only).

Population = a native arm (N2 = 2LPT loa-124, NL = London jura-124) with counts from its MAX4 outputs and truth_counts from the FULL >= 19.0 truth
(emulated z = z + 1.0). Calibration = a chosen arm (N2 / NL / I2 / IL / IR): completeness (molly block) and the response candidate — E (empirical
bin-to-bin kernel via ``adopted_masses_override``) or P (balanced fixed-bin parametric surfaces as the adopted stamps) — built by the science lane's
estimator tool (tools/r041_response_estimator.py on the repair worktree). g = 1 (no z resolution at high z). FP block = the calibration arm's unmatched
accepted rows (native arms) or the HCD-free twin's extra rows (injection arms; also used for IR because a real-spectrum FP block would be real data).
Writes modelA_pack_HZ_v2_<pop>_cal<calib>_<kernel>.npz + provenance sidecar. No real-data value is read.
"""
import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
from CDDF_analysis.hbi_mcmc.extract_pack_hz import (NHAT_EDGES, ZF_EDGES, ZC_EDGES, SNR_EDGES, RESP_SNR_EDGES, T_SIGMA_FLOOR, ROOT_MAX4,  # noqa: E402
                                                    basis_pad_edges_19p0_0p2, dX_from_windows, counts_from_rows, fp_block, stratum_of_row, _idx, _sha)
REPAIR = os.environ.get("REPAIR_REPO", "/home/mfho/wt_highz_repair")
sys.path.insert(0, os.path.join(REPAIR, "tools"))
from r041_response_population_study import load_events, NATIVE as NATIVE_ARMS  # noqa: E402
from r041_response_estimator import build_E, build_P, TB, OB_FINE, cells_of  # noqa: E402

MOLLY = np.array([19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, np.inf])


def molly_from_events(ev):
    """(molly_n_det, molly_n_tot) (8 rows x 7 cells) from a calibration arm's events: per stratum, detected fraction per molly cell on the truth N."""
    kn = np.zeros((5, 7)); nn = np.zeros((5, 7))
    cell = np.clip(np.searchsorted(MOLLY, ev["logN"], side="right") - 1, -1, 6); det = ev["matched"] & np.isfinite(ev["Nhat"])
    for s in range(5):
        for c in range(7):
            m = (ev["stratum"] == s) & (cell == c); nn[s, c] = m.sum(); kn[s, c] = (m & det).sum()
    n_det = np.zeros((8, 7)); n_tot = np.zeros((8, 7))
    for r in range(8):
        s = stratum_of_row(r); n_det[r] = kn[s]; n_tot[r] = nn[s]
    # cells with no truth (e.g. [19.5,20.0) for the native selection) take the nearest populated cell's numbers; (row, cell) entries with n_tot = 0
    # take the cell's stratum-pooled numbers (the schema requires n_tot > 0 everywhere). Both substitutions are recorded.
    fill = []
    for c in range(7):
        if n_tot[:, c].sum() == 0:
            near = [j for j in range(7) if n_tot[:, j].sum() > 0]; j = near[int(np.argmin(np.abs(np.array(near) - c)))]
            n_det[:, c] = n_det[:, j]; n_tot[:, c] = n_tot[:, j]; fill.append(("cell", c, j))
    for c in range(7):
        pooled_det, pooled_tot = n_det[:, c].sum(), n_tot[:, c].sum()
        for r in range(8):
            if n_tot[r, c] == 0:
                n_det[r, c] = pooled_det; n_tot[r, c] = pooled_tot; fill.append(("row", r, c))
    return n_det, n_tot, fill


def fp_from_native(arm):
    """Unmatched accepted rows of a native calibration arm (two-pass one-to-one matching against the FULL truth >= 17.2), binned (nhat, SNR row)."""
    from astropy.io import fits
    sys.path.insert(0, os.path.join(REPAIR, "tools"))
    from r041_response_population_study import match, P_MIN, SNR_MIN
    cfg = NATIVE_ARMS[arm]; pop = {int(r["TARGETID"]): r for r in csv.DictReader(open(cfg["pop"]))}
    t = fits.open(cfg["truth"])[1].data; tid = np.asarray(t["TARGETID"]).astype(np.int64); N = np.asarray(t[cfg["ncol"]], float)
    N = np.log10(N) if np.nanmax(N) > 100 else N; Z = np.asarray(t[cfg["zcol"]], float)
    sel = np.isin(tid, np.fromiter(pop.keys(), dtype=np.int64)); truth_by = {}
    for ti, n, z in zip(tid[sel], N[sel], Z[sel]):
        p = pop[int(ti)]
        if float(p["zlo"]) <= z <= float(p["zhi"]):
            truth_by.setdefault(int(ti), []).append((float(z), float(n)))
    acc = {}
    for f in sorted(glob.glob(os.path.join(cfg["outputs"], "dlacat-*.fits"))):
        d = fits.open(f)[1].data
        for r in d:
            ti = int(r["TARGETID"]); p = pop.get(ti)
            if p is None:
                continue
            if float(r["P_DLA"]) > P_MIN and int(r["DLAFLAG"]) == 0 and float(r["SNR_REDSIDE"]) > SNR_MIN and float(p["zlo"]) < float(r["Z_DLA"]) < float(p["zhi"]):
                acc.setdefault(ti, []).append((float(r["Z_DLA"]), float(r["NHI"]), float(r["P_DLA"])))
    fp = np.zeros((len(NHAT_EDGES) - 1, len(SNR_EDGES) - 1), np.int64); n_extra = 0
    for ti, p in pop.items():
        tl = sorted(truth_by.get(ti, [])); rws = acc.get(ti, [])
        m1, unused = match([dict(z=z) for z, n in tl if n >= 19.0], rws); rest_idx = sorted(unused)
        m2, unused2 = match([dict(z=z) for z, n in tl if n < 19.0], [rws[j] for j in rest_idx])
        left = [rws[rest_idx[j]] for j in sorted(unused2)]
        srow = int(np.clip(_idx(SNR_EDGES, float(p["snr"])), 0, len(SNR_EDGES) - 2))
        for z, n, P in left:
            c = _idx(NHAT_EDGES, n); n_extra += 1
            if 0 <= c < len(NHAT_EDGES) - 1:
                fp[c, srow] += 1
    return fp, len(pop), n_extra


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=["N2", "NL"], required=True); ap.add_argument("--calib", choices=["N2", "NL", "I2", "IL", "IR"], required=True)
    ap.add_argument("--kernel", choices=["E", "P"], required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--study-dir", default=f"{ROOT_MAX4}/response_study")
    a = ap.parse_args(argv); os.makedirs(a.out_dir, exist_ok=True)
    from astropy.io import fits
    # ---- population: counts, dX, truth_counts (full >= 19.0 support; emulated z + 1.0)
    cfg = NATIVE_ARMS[a.population]
    pop = list(csv.DictReader(open(cfg["pop"])))
    for r in pop:
        r["zlo"] = str(float(r["zlo"]) + 1.0); r["zhi"] = str(float(r["zhi"]) + 1.0)
    win = {int(r["TARGETID"]): (float(r["zlo"]), float(r["zhi"])) for r in pop}
    rows = [fits.open(f)[1].data for f in sorted(glob.glob(f"{cfg['outputs']}/dlacat-*.fits"))]
    tid = np.concatenate([np.asarray(r["TARGETID"], np.int64) for r in rows]); z = np.concatenate([np.asarray(r["Z_DLA"], float) for r in rows]) + 1.0
    N = np.concatenate([np.asarray(r["NHI"], float) for r in rows]); P = np.concatenate([np.asarray(r["P_DLA"], float) for r in rows])
    fl = np.concatenate([np.asarray(r["DLAFLAG"], int) for r in rows]); snr = np.concatenate([np.asarray(r["SNR_REDSIDE"], float) for r in rows])
    counts, n_op, n_binned = counts_from_rows(tid, z, N, P, fl, snr, win)
    dX, xc, n_sl = dX_from_windows(pop, lambda r: float(r["snr"]))
    ntrue_edges = basis_pad_edges_19p0_0p2(); assert np.allclose(ntrue_edges, TB), "latent basis differs from the estimator's TB"
    ev_pop = load_events(a.population, a.study_dir)
    truth_counts = np.zeros((len(ntrue_edges) - 1, len(ZF_EDGES) - 1), np.int64)
    for lN, zz in zip(ev_pop["logN"], ev_pop["z"] + 1.0):
        b = _idx(ntrue_edges, lN); k = _idx(ZF_EDGES, zz)
        if 0 <= b < truth_counts.shape[0] and 0 <= k < truth_counts.shape[1]:
            truth_counts[b, k] += 1
    # ---- calibration arm: completeness block, response candidate, FP
    ev_cal = load_events(a.calib, a.study_dir)
    n_det, n_tot, fill = molly_from_events(ev_cal)
    g = np.ones((7, len(ZF_EDGES) - 1)); occ = np.zeros((7, len(ZF_EDGES) - 1))
    for c in range(7):
        occ[c] = n_tot[:, c].sum() / (len(ZF_EDGES) - 1)
    E = build_E(ev_cal); Pc = build_P(ev_cal)
    # deployed + adopted surfaces = Candidate P's surfaces (deg 2, N_ref 20.5, anchor range); for kernel E the override supersedes them in the fold
    mu_c = np.asarray(Pc["coef"]["mu"], float); sig_c = np.asarray(Pc["coef"]["sig"], float); sk_c = np.asarray(Pc["coef"]["skew"], float); rng = np.asarray(Pc["rng"], float)
    if a.calib == "IR":
        fp, n_sl_mock, n_extra = fp_block(f"{ROOT_MAX4}/p1/reductions/analysis_mock_2lpt_random_MAX4_per_injection.csv", f"{ROOT_MAX4}/p1/mock/truth_r041/2lpt_random_population.csv")
        fp_source = "2LPT loa-0 random arm extra rows (mock FP block; a real-spectrum FP block would be real data)"
    elif a.calib in ("I2", "IL"):
        fam = "2lpt" if a.calib == "I2" else "london"
        fp, n_sl_mock, n_extra = fp_block(f"{ROOT_MAX4}/p1/reductions/analysis_mock_{fam}_random_MAX4_per_injection.csv", f"{ROOT_MAX4}/p1/mock/truth_r041/{fam}_random_population.csv")
        fp_source = f"{fam} HCD-free twin random arm extra rows"
    else:
        fp, n_sl_mock, n_extra = fp_from_native(a.calib); fp_source = f"native {a.calib} arm: accepted rows unmatched to any truth >= 17.2"
    col = dX.sum(axis=0); fp_E = np.zeros_like(dX); nz = col > 0; fp_E[:, nz] = dX[:, nz] / col[nz]
    fp_w = n_sl / n_sl_mock; fp_ell = n_sl_mock * (n_sl_mock / n_sl)
    zmid = 0.5 * (ZF_EDGES[:-1] + ZF_EDGES[1:]); kz_to_K = (np.searchsorted(ZC_EDGES, zmid, side="right") - 1).astype(np.int64)
    masked = np.zeros(len(NHAT_EDGES) - 1, bool); masked[(NHAT_EDGES[:-1] >= 19.5 - 1e-9) & (NHAT_EDGES[1:] <= 19.7 + 1e-9)] = True
    B = len(ntrue_edges) - 1; nd = 50   # carriers are provenance-only here (zeros); the schema requires >= 50 draws
    pack = dict(nhat_edges=NHAT_EDGES, ntrue_edges=ntrue_edges, zf_edges=ZF_EDGES, zc_edges=ZC_EDGES, kz_to_K=kz_to_K, snr_edges=SNR_EDGES, nhat_masked_bins=masked,
                counts=counts, dX=dX, dX_coarse_committed=xc, molly_n_det=n_det, molly_n_tot=n_tot, molly_nhi_edges=MOLLY, molly_snr_edges=SNR_EDGES, g_grid=g, g_occupancy=occ,
                resp_mu_coef=mu_c, resp_sig_coef=sig_c, resp_skew_coef=sk_c, resp_snr_edges=np.asarray(RESP_SNR_EDGES, float), resp_z_edges=np.asarray(ZC_EDGES, float),
                resp_sig_floor=np.float64(1e-3), resp_skew_ramp=np.array([21.0, 0.5]), resp_N_ref=np.float64(Pc["N_ref"]), resp_N_fit_range=rng,
                fp_counts=fp, fp_eta_c=np.zeros(len(NHAT_EDGES) - 1), fp_ell_eff=np.float64(fp_ell), fp_w_sightline_ratio=np.float64(fp_w), fp_E_alloc=fp_E,
                t_sigma=np.full(len(ZC_EDGES) - 1, T_SIGMA_FLOOR), truth_counts=truth_counts)
    from CDDF_analysis.hbi_mcmc.pack import ModelAPack, load_pack
    from CDDF_analysis.hbi_mcmc.count_conserving_fold import surface_masses
    tmp = ModelAPack(**pack)
    _, phi_P = surface_masses(tmp, mu_c, sig_c, sk_c, rng, NHAT_EDGES)
    pack.update(tp_convention_id="hz_native_full_support_dz_0.01", contract_id="hz-estimator-rebuild-2026-09-02", adopted_resp_version=f"candidate_{a.kernel}_v1",
                adopted_resp_mu_coef=mu_c, adopted_resp_sig_coef=sig_c, adopted_resp_skew_coef=sk_c, adopted_resp_fit_range=rng,
                adopted_carrier_mu=np.zeros((nd,) + mu_c.shape), adopted_carrier_sig=np.zeros((nd,) + mu_c.shape), adopted_carrier_skew=np.zeros((nd,) + mu_c.shape),
                adopted_carrier_shared3=np.zeros((nd, 3)))
    if a.kernel == "E":
        M = np.asarray(E["M"], float); assert M.shape == (3, 3, len(NHAT_EDGES) - 1, B), M.shape
        pack.update(adopted_masses_override=M, adopted_phi_ref=M.sum(axis=2))
    else:
        pack.update(adopted_phi_ref=phi_P)
    name = f"modelA_pack_HZ_v2_{a.population}_cal{a.calib}_{a.kernel}.npz"; npz = os.path.join(a.out_dir, name); np.savez(npz, **pack)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
    commit_repair = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPAIR).decode().strip()
    prov = dict(real_data=False, mode="native_v2", population=a.population, calibration=a.calib, kernel=a.kernel, gate="MAX4_RESPONSE_ESTIMATOR_CLOSURE_GATE_2026-09-02.md §4",
                code_commit=commit, repair_commit=commit_repair, built=_dt.datetime.now().astimezone().isoformat(),
                sizes=dict(n_op=n_op, n_binned=n_binned, counts_total=int(counts.sum()), n_sl=n_sl, dX_total=float(dX.sum()), truth_total=int(truth_counts.sum()),
                           truth_ge20=int(truth_counts[ntrue_edges[:-1] >= 20.0 - 1e-9].sum()), fp_rows=int(fp.sum()), n_sl_calib=n_sl_mock, fp_w=fp_w, fp_ell_eff=fp_ell,
                           E_fallback_cells=len(E["fallback"]), P_anchors={k: len(v) for k, v in Pc["anchors"].items()}),
                conventions=dict(truth="FULL native truth >= 19.0 inside the population windows (response-study matched table), emulated z + 1.0", g="1 (no z resolution)",
                                 completeness=f"molly block from the {a.calib} arm's matched events (detected fraction per stratum x molly cell); filled cells {fill}",
                                 response=("Candidate E: empirical bin-to-bin kernel (adopted_masses_override; phi_ref = column sums)" if a.kernel == "E" else
                                           "Candidate P: balanced fixed-bin parametric surfaces (deg 2, N_ref 20.5, anchor-range clamp) as the adopted stamps"),
                                 fp=fp_source, t_sigma=f"{T_SIGMA_FLOOR} floor"),
                inputs=dict(population_csv=cfg["pop"], population_csv_sha256=_sha(cfg["pop"]), outputs=cfg["outputs"],
                            matches_population=f"{a.study_dir}/matches_{a.population}_native_full.csv",
                            matches_population_sha256=_sha(f"{a.study_dir}/matches_{a.population}_native_full.csv"),
                            calibration_events=(f"{a.study_dir}/matches_{a.calib}_native_full.csv" if a.calib in ("N2", "NL") else "injection per-injection table")))
    json.dump(prov, open(npz[:-4] + ".provenance.json", "w"), indent=1)
    pk = load_pack(npz, allow_nonstandard_grid=True)
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    consts, Mg = build_cc_tensors(pk)
    print(f"pack {npz} sha256 {_sha(npz)[:16]}… counts {int(counts.sum())} truth {int(truth_counts.sum())} (>=20: {prov['sizes']['truth_ge20']}) fp {int(fp.sum())} Mg {tuple(np.asarray(Mg).shape)} fallback_E {len(E['fallback'])}")


if __name__ == "__main__":
    main()
