#!/usr/bin/env python
"""subdla_catalog_products.py — the sub-DLA CATALOG products of Paper 1 (R-035, request package
2026-08-28, priority 3): raw uncorrected distributions, reliability under the stated denominator,
and the downstream response / correction set. NOT a population measurement: no abundance, no
CDDF, no Omega (D3, PI #52).

Three things are kept separate throughout and carried as metadata on every array:
  * the catalogue's REACH: the finder reports log N-hat down to 17.2 (LLS mode) — the raw
    distributions are shown over the full reach so the reader sees where the reported range sits;
  * the REPORTING FLOOR 19.7 (D3): the reported sub-DLA range is [19.7, 20.3);
  * the MASKED interval [19.5, 19.7): inside the frozen data plane as latent nuisance support,
    never a reported number.
No rebinning across 19.7 (the histogram edges are the 0.1-dex N-hat grid of the frozen data
plane, on which 19.5, 19.7 and 20.3 are edges).

Products:
  1. raw reported-N-hat distribution of accepted candidates under the Paper-1 selection contract
     (contract row mask at the 3300 km/s collar, non-BAL, DLAFLAG 0, P_DLA > 0.99, SNR_REDSIDE > 2),
     per SNR cell, for z_DLA inside the data-plane support [2.0, 3.5) and for all z in the
     window — RE-DERIVED from the catalogue and CLOSED against the frozen pack's counts grid;
  2. raw absorber-redshift distribution of the reported sub-DLA candidates [19.7, 20.3)
     (and, separately, of the masked [19.5, 19.7) candidates), per SNR cell;
  3. reliability over the reported sub-DLA range, read from the R-033 product (same denominator
     conventions; 2LPT-0 calibration substrate): absorber-level completeness and purity, class-level
     purity/completeness, per SNR cell;
  4. the downstream propagation set: the adopted response kernel p(N-hat bin | N_true bin, SNR
     cell, z cell) exactly as the production count-conserving fold applies it (renormalised x
     adopted_phi_ref), the frozen completeness matrix (molly n_det / n_tot on its own grid), the
     FP block (fp_counts, fp_eta_c, fp_w, fp_ell_eff, fp_E_alloc) with their semantics.

Real-data VALUES never enter this file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

REACH_EDGES = np.round(np.arange(17.2, 22.5 + 1e-9, 0.1), 3)
REPORT_FLOOR, CLASS_CUT, PLANE_FLOOR = 19.7, 20.3, 19.5
Z_SUPPORT = (2.0, 3.5)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="dlacat-loa-main-dark-v1.fits")
    ap.add_argument("--real-qsocat", required=True, help="QSO catalogue with BI_CIV (BAL policy)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--pack-provenance", required=True)
    ap.add_argument("--pc-json", required=True, help="R-033 2LPT-0 product (reliability cells)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)
    import fitsio
    from CDDF_analysis.hbi_mcmc import extract_pack as EP
    from CDDF_analysis.hbi_mcmc.extract_pack_real import contract_row_mask, make_cfg, COLLAR_KMS
    from CDDF_analysis.hbi_mcmc.count_conserving_fold import surface_masses, phi_from_surfaces

    P = np.load(a.pack, allow_pickle=True)
    prov = json.load(open(a.pack_provenance))
    cat = fitsio.read(a.catalog, ext=1)
    qso = fitsio.read(a.real_qsocat, ext=1, columns=["TARGETID", "BI_CIV"])
    bal_tids = np.unique(qso["TARGETID"][qso["BI_CIV"] > 0].astype(np.int64))
    cfg = make_cfg(os.path.dirname(a.catalog), "(bal list passed directly)", str(P["molly_nhi_edges"]), a.out_dir)
    tid = cat["TARGETID"].astype(np.int64)
    zq, zd, nhi, snr = (np.asarray(cat[k], float) for k in ("Z_QSO", "Z_DLA", "NHI", "SNR_REDSIDE"))
    keep = contract_row_mask(zq, zd, tid, bal_tids, cfg, collar_kms=COLLAR_KMS)
    op = keep & (np.asarray(cat["DLAFLAG"], int) == 0) & (np.asarray(cat["P_DLA"], float) > cfg.p_dla_min) & (snr > cfg.snr_min)
    counts, n_in = EP.bin_counts_cks(nhi[op], zd[op], snr[op])
    closure = {"n_op_rows_rederived": int(op.sum()), "n_op_rows_provenance": prov["n_op_rows"],
               "counts_in_window_rederived": int(n_in), "counts_in_window_provenance": prov["counts_in_window"],
               "counts_grid_identical_to_pack": bool(np.array_equal(counts, P["counts"]))}
    if not (closure["counts_grid_identical_to_pack"] and closure["n_op_rows_rederived"] == prov["n_op_rows"]):
        raise SystemExit(f"BLOCKED: contract re-derivation does not close against the frozen pack: {closure}")
    print("closure OK: the frozen counts grid and n_op_rows are reproduced from the catalogue under the contract")

    snr_e = np.asarray(P["snr_edges"], float)
    zf = np.asarray(P["zf_edges"], float)
    si = np.clip(np.searchsorted(snr_e, snr[op], side="right") - 1, 0, len(snr_e) - 2)
    inz = (zd[op] >= Z_SUPPORT[0]) & (zd[op] < Z_SUPPORT[1])
    # 1. raw N-hat distribution over the full reach, per SNR cell
    def hist2(x, sel):
        H = np.zeros((len(REACH_EDGES) - 1, len(snr_e) - 1))
        i = np.searchsorted(REACH_EDGES, x[sel], side="right") - 1
        ok = (i >= 0) & (i < len(REACH_EDGES) - 1)
        np.add.at(H, (i[ok], si[sel][ok]), 1)
        return H
    H_nhat_inz = hist2(nhi[op], inz)
    H_nhat_allz = hist2(nhi[op], np.ones(op.sum(), bool))
    region = np.where(REACH_EDGES[:-1] < PLANE_FLOOR - 1e-9, "catalogue_reach_below_data_plane",
                      np.where(REACH_EDGES[:-1] < REPORT_FLOOR - 1e-9, "masked_latent_support_[19.5,19.7)",
                               np.where(REACH_EDGES[:-1] < CLASS_CUT - 1e-9, "reported_subDLA_[19.7,20.3)", "reported_DLA_>=20.3")))
    # 2. raw z_DLA distribution of the reported sub-DLA candidates and of the masked ones, per SNR cell
    zedges = np.concatenate([[1.9], zf, [3.6, 4.3]])
    def histz(sel):
        H = np.zeros((len(zedges) - 1, len(snr_e) - 1))
        i = np.searchsorted(zedges, zd[op][sel], side="right") - 1
        ok = (i >= 0) & (i < len(zedges) - 1)
        np.add.at(H, (i[ok], si[sel][ok]), 1)
        return H
    sub = (nhi[op] >= REPORT_FLOOR - 1e-9) & (nhi[op] < CLASS_CUT - 1e-9)
    msk = (nhi[op] >= PLANE_FLOOR - 1e-9) & (nhi[op] < REPORT_FLOOR - 1e-9)
    H_z_sub, H_z_msk = histz(sub), histz(msk)
    # closure of the sub-DLA counts with the pack grid inside the support
    nh = np.asarray(P["nhat_edges"], float)
    sub_cells = (nh[:-1] >= REPORT_FLOOR - 1e-9) & (nh[1:] <= CLASS_CUT + 1e-9)
    msk_cells = (nh[:-1] >= PLANE_FLOOR - 1e-9) & (nh[1:] <= REPORT_FLOOR + 1e-9)
    n_sub_plane = int(P["counts"][sub_cells].sum()); n_msk_plane = int(P["counts"][msk_cells].sum())
    assert n_sub_plane == int(H_z_sub[1:-2].sum()) and n_msk_plane == int(H_z_msk[1:-2].sum())
    # 3. reliability over the reported sub-DLA range (from R-033, 2LPT-0)
    pc = json.load(open(a.pc_json))
    ne = np.array(pc["contract"]["n_edges"])
    sub_rows = [i for i in range(len(ne) - 1) if ne[i] >= REPORT_FLOOR - 1e-9 and ne[i + 1] <= CLASS_CUT + 1e-9]
    msk_rows = [i for i in range(len(ne) - 1) if ne[i + 1] <= REPORT_FLOOR + 1e-9]
    rel = {}
    for grid in ("molly_snr_cells", "response_snr_cells"):
        T = pc["tables"][grid]
        def agg(num, den, rows):
            n = np.array(T[num], float)[rows].sum(axis=0); d = np.array(T[den], float)[rows].sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                return {"value": (n / d).tolist(), "numerator": n.tolist(), "denominator": d.tolist()}
        rel[grid] = {
            "snr_edges": pc["contract"].get(grid, None) or (pc["tables"] and None),
            "per_cell_rows_[19.7,19.9),[19.9,20.1),[20.1,20.3)": {k: np.array(T[k], float)[sub_rows].tolist() for k in ("C_abs_any", "C_abs_reported", "P_abs", "P_abs_incl_subfloor_host", "C_cls", "P_cls", "n_true", "n_det", "n_tp_by_true", "n_tp_by_nhat")},
            "aggregated_over_[19.7,20.3)": {"C_abs_any": agg("n_found_any", "n_true", sub_rows), "C_abs_reported": agg("n_found_reported", "n_true", sub_rows),
                                            "P_abs": agg("n_tp_by_nhat", "n_det", sub_rows), "P_abs_incl_subfloor_host": {"value": ((np.array(T["n_tp_by_nhat"], float)[sub_rows].sum(0) + np.array(T["n_subfloor_host"], float)[sub_rows].sum(0)) / np.array(T["n_det"], float)[sub_rows].sum(0)).tolist()},
                                            "C_cls": agg("n_cls_ok_by_true", "n_tp_by_true", sub_rows), "P_cls": agg("n_cls_ok_by_nhat", "n_tp_by_nhat", sub_rows)},
            "masked_[19.5,19.7)_for_reference_only": {"C_abs_any": agg("n_found_any", "n_true", msk_rows), "P_abs": agg("n_tp_by_nhat", "n_det", msk_rows)},
        }
    # 4. the propagation set
    pk = SimpleNamespace(**{k: P[k] for k in P.files})
    phi_stored = np.asarray(pk.adopted_phi_ref, float)
    assert float(np.max(np.abs(phi_stored - phi_from_surfaces(pk)))) <= 1e-9
    masses, phi = surface_masses(pk, pk.adopted_resp_mu_coef, pk.adopted_resp_sig_coef, pk.adopted_resp_skew_coef,
                                 np.asarray(pk.adopted_resp_fit_range, float), np.asarray(pk.nhat_edges, float))
    kernel = masses / np.maximum(phi, 1e-12)[:, :, None, :] * phi_stored[:, :, None, :]     # as the production fold applies it
    with np.errstate(invalid="ignore", divide="ignore"):
        C_molly = np.asarray(P["molly_n_det"], float) / np.asarray(P["molly_n_tot"], float)
    propagation = {
        "response_kernel": "kernel[sr, zr, c, b] = P(N-hat in observed bin c | true N in latent bin b, SNR cell sr, z cell zr), the adopted v1.1 skew-normal response surfaces evaluated on the pack grids, renormalised to unit in-grid mass and multiplied by the deployed in-grid fraction adopted_phi_ref (the production count-conserving fold; sum over c = phi[sr, zr, b] <= 1, the remainder leaves the observed grid)",
        "resp_snr_edges": np.asarray(P["resp_snr_edges"], float).tolist(), "resp_z_edges": np.asarray(P["resp_z_edges"], float).tolist(),
        "nhat_edges": nh.tolist(), "ntrue_edges": np.asarray(P["ntrue_edges"], float).tolist(),
        "completeness": "C_molly[s, j] = molly_n_det / molly_n_tot on molly_snr_edges x molly_nhi_edges (TRUE N; the frozen calibration completeness the likelihood consumes); note its lowest cells [19.5, 20.0) straddle the 19.7 floor — use the R-033 tables for a 19.7-aligned view",
        "molly_snr_edges": np.asarray(P["molly_snr_edges"], float).tolist(), "molly_nhi_edges": [float(x) if np.isfinite(x) else "inf" for x in np.asarray(P["molly_nhi_edges"], float)],
        "false_positives": "fp_counts[c, s] = loa-0 (HCD-free twin mock) FP detections per (N-hat bin, SNR cell) with fp_eta_c the per-N-hat-bin sub-floor-host fraction; the real-data FP expectation is fp_w * fp_ell_eff * (1 - fp_eta_c) * lambda_fp * fp_E_alloc[k, s] (model_a / count_conserving_fold); fp_w = n_sl_real / n_sl_loa0, fp_ell_eff the effective loa-0 path scale",
        "fp_w_sightline_ratio": float(P["fp_w_sightline_ratio"]), "fp_ell_eff": float(P["fp_ell_eff"]),
        "usage": "expected observed counts in (c, k, s) = sum_b dX[k, s] * C(N_b, s) * g[b, k] * f_true[b, k] * dN_b * kernel[sr(s), zr(k), c, b] + FP term — i.e. the catalogue counts are to be used WITH these operators (a forward model), never divided by a completeness alone",
    }
    npz = {"reach_edges": REACH_EDGES, "reach_region": region, "snr_edges": snr_e, "z_edges_for_zhist": zedges,
           "raw_nhat_hist_inplane_z": H_nhat_inz, "raw_nhat_hist_all_window_z": H_nhat_allz,
           "raw_zdla_hist_reported_subDLA_19p7_20p3": H_z_sub, "raw_zdla_hist_masked_19p5_19p7": H_z_msk,
           "kernel_adopted": kernel, "kernel_phi": phi_stored, "completeness_molly": C_molly,
           "molly_n_det": np.asarray(P["molly_n_det"], float), "molly_n_tot": np.asarray(P["molly_n_tot"], float),
           "molly_snr_edges": np.asarray(P["molly_snr_edges"], float), "molly_nhi_edges": np.asarray(P["molly_nhi_edges"], float),
           "fp_counts": np.asarray(P["fp_counts"], float), "fp_eta_c": np.asarray(P["fp_eta_c"], float), "fp_E_alloc": np.asarray(P["fp_E_alloc"], float),
           "resp_snr_edges": np.asarray(P["resp_snr_edges"], float), "resp_z_edges": np.asarray(P["resp_z_edges"], float),
           "nhat_edges": nh, "ntrue_edges": np.asarray(P["ntrue_edges"], float), "zf_edges": zf,
           "axis_note": np.array(["raw_nhat_hist_*: (reach bin in reach_edges order, SNR cell in snr_edges order) — UNCORRECTED accepted-candidate counts; raw_zdla_hist_*: (z bin in z_edges_for_zhist order, SNR cell); kernel_adopted: (resp SNR cell, resp z cell, N-hat bin, N_true bin); completeness_molly: (molly SNR cell, molly N cell)"])}
    out = {
        "role": "R-035 sub-DLA CATALOG products — raw UNCORRECTED distributions of accepted candidates under the Paper-1 selection contract, reliability of the reported sub-DLA range under the R-033 denominators, and the downstream response / correction set; NOT a population measurement (D3, PI #52); sits BESIDE the frozen products",
        "status": "publication-ready as catalogue accounting (distributions are read-throughs of the frozen data plane, closed against it) and as calibration tables (reliability from the R-033 product); the response set is the frozen operator re-emitted with its semantics",
        "written_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"module": "CDDF_analysis/hbi_mcmc/subdla_catalog_products.py", "commit": _git(repo), "argv": sys.argv, "python": sys.version.split()[0], "numpy": np.__version__, "conda_env": os.environ.get("CONDA_DEFAULT_ENV")},
        "inputs": {k: {"path": p, "sha256": _sha(p)} for k, p in [("catalog", a.catalog), ("real_qsocat", a.real_qsocat), ("pack", a.pack), ("pack_provenance", a.pack_provenance), ("pc_json", a.pc_json)]},
        "contract": {"row_mask": f"z_QSO in ({cfg.z_qso_min}, {cfg.z_qso_max}), Lyα window lam_rf [{cfg.lam_rf_min}, {cfg.lam_rf_max}] with 3600 A floor and {COLLAR_KMS} km/s collar, TARGETID not in the BI_CIV > 0 list; DLAFLAG == 0; P_DLA > {cfg.p_dla_min}; SNR_REDSIDE > {cfg.snr_min}",
                     "three_ranges": {"catalogue_reach": [float(REACH_EDGES[0]), float(REACH_EDGES[-1])], "data_plane": [PLANE_FLOOR, 22.4], "masked_latent_support": [PLANE_FLOOR, REPORT_FLOOR], "reported_subDLA": [REPORT_FLOOR, CLASS_CUT], "reported_DLA": [CLASS_CUT, 22.4]},
                     "z_support_of_the_data_plane": list(Z_SUPPORT), "uncorrected": "every raw_* array is an UNCORRECTED count of accepted candidates (no completeness, purity or response correction)"},
        "closure": closure,
        "raw_summary": {"n_accepted_rows_in_window_any_nhat": int(op.sum()), "n_in_data_plane_[19.5,22.4)x[2.0,3.5)": int(n_in),
                        "n_reported_subDLA_[19.7,20.3)_in_plane": n_sub_plane, "n_masked_[19.5,19.7)_in_plane": n_msk_plane,
                        "n_below_19p5_in_plane_z": int(H_nhat_inz[REACH_EDGES[:-1] < PLANE_FLOOR - 1e-9].sum()),
                        "n_reported_DLA_ge20p3_in_plane": int(P["counts"][nh[:-1] >= CLASS_CUT - 1e-9].sum())},
        "reliability_reported_subDLA_range": rel,
        "propagation_set": propagation,
    }
    jp = os.path.join(a.out_dir, "R035_subdla_catalog_products.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else float(o))
    np.savez(os.path.join(a.out_dir, "R035_subdla_catalog_products.npz"), **npz)
    files = [jp, jp[:-5] + ".npz"]
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for p in files:
            fh.write(f"{_sha(p)}  {os.path.basename(p)}\n")
    print(open(os.path.join(a.out_dir, "SHA256SUMS")).read())
    print(json.dumps(out["raw_summary"], indent=1))
    print("reliability [19.7,20.3) (response SNR cells):", json.dumps({k: [round(x, 3) if x is not None else None for x in v["value"]] for k, v in rel["response_snr_cells"]["aggregated_over_[19.7,20.3)"].items()}))


if __name__ == "__main__":
    main()
