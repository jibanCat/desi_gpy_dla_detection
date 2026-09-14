#!/usr/bin/env python
"""build_fixed_objects.py — the FROZEN fixed calibration objects, mapped onto
the REAL low-z C1 pack.

CLASSIFICATION: **VALIDATION / PREPARATION ONLY.**  No likelihood, no sampler,
no posterior.  Every object written here is a calibration product fitted on the
MOCK families before the freeze (PI ruling 2026-09-14 §16); this script only
re-expresses them on the real pack's grid.  It reads the real pack for its
grids / dX only.

Products (into ``--out-dir``):

  ``Mg_B_real.npz``            Mg[s, kf, c, b] = rows_unit(B) x phi_2LPT,measured
                               gathered through the REAL pack's ``kz_to_K``.
                               Schema ``absorber_ladder/Mg_fixed/v1``.
  ``C_C1nsadd_real.npz``       the frozen C1nsadd completeness table (S, B),
                               copied unchanged after the strata/edge equality
                               gate.
  ``mu_extra_P6bcal_real.npz`` the TRANSPORTED sub-floor-host term
                               mu_extra[c, k, s] = rate_2LPT[c, K(k), s]
                                                   * dX_real[k, s].
                               Key ``mu_extra`` (what ``run_ladder.py
                               --extra-fixed-file`` reads first).

Every gate is FAIL-CLOSED: a grid mismatch raises, nothing is written.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

L = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
REAL_PACK = ("/home/mfho/lowz_clean_work_2026-09-12/posterior_campaign/"
             "packs/C1_pack.npz")
CAL_FAMILY = "2lpt0"          # the calibration family of record for B / C / P6b


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _eq(a, b, name):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise SystemExit(
            f"FAIL-CLOSED: {name} differs between the real pack and the frozen "
            f"calibration object (shapes {a.shape} vs {b.shape}). The frozen "
            "objects may NOT be applied to this pack; STOP and report.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=REAL_PACK)
    ap.add_argument("--ladder-root", default=L)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    LR = a.ladder_root

    z = np.load(a.pack, allow_pickle=True)
    kz = np.asarray(z["kz_to_K"], int)
    dX_real = np.asarray(z["dX"], float)
    common = dict(
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        builder="validation/real_c1/build_fixed_objects.py",
        classification=("VALIDATION/PREPARATION ONLY — the fixed object is a "
                        "MOCK-fitted calibration product frozen before the "
                        "blind real run (PI ruling 2026-09-14 §16, §17); "
                        "nothing here is fitted to real data"),
        fitted_to_real_data=False,
        real_pack=os.path.abspath(a.pack),
        real_pack_sha256=sha256_file(a.pack),
        calibration_family=CAL_FAMILY)
    written = {}

    # =====================================================================
    # (a) Mg_B_real.npz
    # =====================================================================
    mg_src = os.path.join(LR, "response_review", "candidates",
                          f"Mg_B_{CAL_FAMILY}.npz")
    zm = np.load(mg_src, allow_pickle=True)
    rows_unit = np.asarray(zm["rows_unit"], float)          # (B, S, KK, C)
    phi = np.asarray(zm["phi_bsK"], float)                  # (B, S, KK)
    if rows_unit.shape[:3] != phi.shape:
        raise SystemExit("rows_unit / phi_bsK shape mismatch (fail-closed)")
    rs = rows_unit.sum(axis=3)
    if not np.allclose(rs, 1.0, rtol=0, atol=1e-12):
        raise SystemExit(f"rows_unit does not sum to 1 over c "
                         f"(max dev {np.abs(rs - 1).max():.3e}) — fail-closed")
    if int(kz.max()) + 1 != phi.shape[2]:
        raise SystemExit("the real pack's kz_to_K does not address the "
                         "calibration object's coarse-K axis — fail-closed")
    Mg = np.einsum("bsKc,bsK->sKcb", rows_unit, phi)[:, kz, :, :]  # (S,Kf,C,B)
    B, S, KK, C = rows_unit.shape
    if Mg.shape != (S, len(kz), C, B):
        raise SystemExit(f"Mg shape {Mg.shape} unexpected — fail-closed")
    # the row-sum identity the fold depends on
    phi_k = np.einsum("bsK->sKb", phi)[:, kz, :]
    dev = float(np.abs(Mg.sum(axis=2) - phi_k).max())
    if dev > 1e-12:
        raise SystemExit(f"sum_c Mg != phi (max dev {dev:.3e}) — fail-closed")
    # equality with the calibration family's stored tensor (the real pack's
    # kz_to_K is byte-identical to the mocks', so this must be EXACT)
    mg_ref = np.asarray(zm["Mg"], float)
    exact_vs_mock = bool(np.array_equal(Mg, mg_ref))
    if not exact_vs_mock:
        raise SystemExit("Mg rebuilt on the real pack's kz map differs from "
                         "the frozen mock tensor — the fine-z grid is NOT the "
                         "same; STOP and report (fail-closed)")
    src_prov = json.loads(str(zm["provenance"]))
    p = dict(common)
    p.update(schema="absorber_ladder/Mg_fixed/v1", variant="B", family="real",
             shape=list(Mg.shape),
             source=mg_src, source_sha256=sha256_file(mg_src),
             construction=("Mg[s, kf, c, b] = phi_bsK[b, s, K(kf)] * "
                           "rows_unit[b, s, K(kf), c], K(kf) = the REAL pack's "
                           "kz_to_K; rows_unit sums to 1 over c EXACTLY, so "
                           "sum_c Mg = phi (verified to "
                           f"{dev:.2e})"),
             phi=("2LPT-0 MEASURED per-cell conditional in-grid fraction "
                  "(Jeffreys +1/2), the model of record (PI ruling "
                  "2026-09-14 §1)"),
             identical_to_frozen_mock_tensor=exact_vs_mock,
             row_sum_max_abs_dev_vs_phi=dev,
             consumer=("CDDF_analysis/hbi_mcmc/fp_ladder.model_cc_ladder "
                       "einsum('skcb,sb,bk->cks', Mg, C, w); "
                       "--mg-fixed-file <this> --mg-fixed-key Mg"),
             source_provenance=src_prov)
    out = os.path.join(a.out_dir, "Mg_B_real.npz")
    np.savez_compressed(out, Mg=Mg, rows_unit=rows_unit, phi_bsK=phi,
                        kz_to_K=kz,
                        provenance=np.array(json.dumps(p, indent=1, default=str),
                                            dtype=object))
    written["Mg_B_real.npz"] = out

    # =====================================================================
    # (b) C_C1nsadd_real.npz
    # =====================================================================
    c_src = os.path.join(LR, "completeness", f"C_C1nsadd_{CAL_FAMILY}.npz")
    zc = np.load(c_src, allow_pickle=True)
    _eq(zc["ntrue_edges"], z["ntrue_edges"], "ntrue_edges")
    _eq(zc["snr_edges"], z["snr_edges"], "snr_edges")
    _eq(zc["kz_to_K"], z["kz_to_K"], "kz_to_K")
    # b_to_cell is derived inside the fold; re-derive it from the pack and
    # compare with the table's stored map (this is the object the completeness
    # actually indexes)
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.forward import build_consts
    pk = load_pack(a.pack)
    consts = build_consts(pk, resp_clamp="both")
    _eq(zc["b_to_cell"], np.asarray(consts.b_to_cell), "b_to_cell")
    C_fixed = np.asarray(zc["C_fixed"], float)
    if C_fixed.shape != (consts.n_s, consts.n_b):
        raise SystemExit(f"C_fixed shape {C_fixed.shape} != (S, B) "
                         f"({consts.n_s}, {consts.n_b}) — fail-closed")
    if not np.all(np.isfinite(C_fixed)) or C_fixed.min() < 0 or C_fixed.max() > 1:
        raise SystemExit("C_fixed is not a finite probability table")
    live = np.asarray(zc["live_strata_mask"], bool)
    dX_live = dX_real.sum(axis=0) > 0
    p = dict(common)
    p.update(schema="absorber_ladder/C_fixed/v1", variant="C1nsadd",
             family="real", shape=list(C_fixed.shape),
             source=c_src, source_sha256=sha256_file(c_src),
             note=("the C1nsadd table is a function of (true N bin, S/N "
                   "stratum) ONLY — it is byte-identical across the three mock "
                   "families — so on a pack sharing ntrue_edges, snr_edges, "
                   "kz_to_K and b_to_cell it applies UNCHANGED. Copied, not "
                   "refitted."),
             strata_gate=dict(
                 ntrue_edges_EQUAL=True, snr_edges_EQUAL=True,
                 kz_to_K_EQUAL=True, b_to_cell_EQUAL=True,
                 table_live_strata=[int(i) for i in np.where(live)[0]],
                 real_pack_live_strata=[int(i) for i in np.where(dX_live)[0]],
                 live_strata_EQUAL=bool(np.array_equal(live, dX_live))),
             consumer="--c-fixed-file <this> (2-D (S,B) branch: keeps g_bk)",
             source_provenance=json.loads(str(zc["provenance"])))
    out = os.path.join(a.out_dir, "C_C1nsadd_real.npz")
    np.savez_compressed(
        out, C_fixed=C_fixed, C_fixed_sd=np.asarray(zc["C_fixed_sd"], float),
        live_strata_mask=live, ntrue_edges=np.asarray(zc["ntrue_edges"]),
        snr_edges=np.asarray(zc["snr_edges"]), kz_to_K=np.asarray(zc["kz_to_K"]),
        b_to_cell=np.asarray(zc["b_to_cell"]),
        provenance=np.array(json.dumps(p, indent=1, default=str), dtype=object))
    written["C_C1nsadd_real.npz"] = out

    # =====================================================================
    # (c) the TRANSPORTED sub-floor-host term
    # =====================================================================
    p6 = os.path.join(LR, "completeness", f"P6b_rate_{CAL_FAMILY}.npz")
    z6 = np.load(p6, allow_pickle=True)
    _eq(z6["nhat_edges"], z["nhat_edges"], "nhat_edges (P6b)")
    _eq(z6["zf_edges"], z["zf_edges"], "zf_edges (P6b)")
    _eq(z6["snr_edges"], z["snr_edges"], "snr_edges (P6b)")
    _eq(z6["kz_to_K"], z["kz_to_K"], "kz_to_K (P6b)")
    rate = np.asarray(z6["rate_cKs"], float)                 # (C, KK, S) per dX
    if rate.shape[1] != int(kz.max()) + 1:
        raise SystemExit("P6b rate coarse-K axis does not match kz_to_K")
    mu_extra = rate[:, kz, :] * dX_real[None, :, :]          # (C, Kf, S)
    if mu_extra.shape != tuple(np.asarray(z["counts"]).shape):
        raise SystemExit(f"mu_extra shape {mu_extra.shape} != counts shape "
                         f"{np.asarray(z['counts']).shape} — fail-closed")
    if not np.all(np.isfinite(mu_extra)) or mu_extra.min() < 0:
        raise SystemExit("mu_extra is not a finite non-negative intensity")
    dead = ~(dX_real.sum(axis=0) > 0)
    if mu_extra[:, :, dead].sum() != 0:
        raise SystemExit("mu_extra is nonzero on a dead stratum — fail-closed")
    p = dict(common)
    p.update(schema="absorber_ladder/mu_extra_fixed/v1", variant="P6b-cal",
             family="real", shape=list(mu_extra.shape),
             source=p6, source_sha256=sha256_file(p6),
             representation=("rate[c, K, s] = P6b_counts(2LPT-0) / dX(2LPT-0) "
                             "[events per unit absorption distance]; "
                             "mu_extra[c, k, s] = rate[c, kz_to_K[k], s] * "
                             "dX_real[k, s] [expected COUNTS], added to mu_FP "
                             "inside model_cc_ladder"),
             units="expected counts per (Nhat cell, fine-z bin, S/N stratum)",
             prescription=("survey-facing sub-floor-host term = the "
                           "TRANSPORTED 2LPT-0 rate (PI ruling 2026-09-14 §1: "
                           "'sub-floor-host = fixed TRANSPORTED calibration "
                           "term for survey use'); the truth-pinned P6b "
                           "(--fix P) is MOCK-CLOSURE ONLY and is NOT "
                           "available on real data"),
             consumer="--extra-fixed-file <this> (key 'mu_extra')",
             source_provenance=json.loads(str(z6["provenance"])))
    out = os.path.join(a.out_dir, "mu_extra_P6bcal_real.npz")
    np.savez_compressed(
        out, mu_extra=mu_extra, rate_cKs=rate,
        rate_cKs_halfE=np.asarray(z6["rate_cKs_halfE"], float),
        rate_cKs_halfO=np.asarray(z6["rate_cKs_halfO"], float),
        dX=dX_real, kz_to_K=kz,
        nhat_edges=np.asarray(z["nhat_edges"], float),
        zf_edges=np.asarray(z["zf_edges"], float),
        snr_edges=np.asarray(z["snr_edges"], float),
        provenance=np.array(json.dumps(p, indent=1, default=str), dtype=object))
    written["mu_extra_P6bcal_real.npz"] = out

    man = dict(schema="absorber_ladder/real_c1_fixed_objects/v1",
               built_utc=common["built_utc"],
               real_pack=common["real_pack"],
               real_pack_sha256=common["real_pack_sha256"],
               products={k: dict(path=v, sha256=sha256_file(v))
                         for k, v in written.items()})
    with open(os.path.join(a.out_dir, "FIXED_OBJECTS_MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    for k, v in man["products"].items():
        print(f"{k:28s} {v['sha256'][:16]}  {v['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
