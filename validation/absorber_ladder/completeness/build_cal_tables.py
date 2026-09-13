#!/usr/bin/env python
"""build_cal_tables.py — completeness calibration tables WITH sightline halves.

VALIDATION-ONLY.  No sampler.  Nothing under ``CDDF_analysis/`` is modified.
This is a NEW file; ``validation/absorber_diag/build_matched_ops.py`` is
imported and re-used verbatim for the cut + match, and every array this builder
produces is GATED elementwise against the objects that builder already wrote
(``empirical_ops_<fam>.npz``) and against the adopted pack.  The ONE thing this
builder adds is the per-detection / per-truth ``TARGETID``, which the empirical
operators do not carry and which the sightline-half cross-validation needs.

Outputs ``cal_table_<fam>.npz`` with

    truth_bks      (B, Kf, S)  truth systems  (== pack truth_counts_bks, GATED)
    det_bks        (B, Kf, S)  matched detections, NO observed-grid restriction
                               (== ops N_det_all_bks_true_z, GATED)
    truth_bks_E/O, det_bks_E/O  the same, split by TARGETID parity
    P6b_cks        (C, Kf, S)  host in [17.2, 19.0)  (== ops P6b_cks, GATED)
    P6b_cks_E/O    the parity halves
    snr_med_s, snr_logmed_s, snr_mean_s  per-stratum S/N summaries of the TRUTH
                               population (the continuous S/N covariate)
    dX             (Kf, S) from the adopted pack

Detection probability convention.  ``det_bks`` counts a matched detection
irrespective of where its observed N-hat lands, so ``det/truth`` is the pure
DETECTION probability ``C_det`` — the same object the frozen molly matrix
measures (``molly_n_det / molly_n_tot``) and the same object the fold's
``C_bs = sigmoid(eta_hat + psi_c)[b_to_cell]`` occupies.  The in-grid counting
fraction ``phi`` stays with the response kernel, where it belongs
(``OPERATOR_FORENSICS_REPORT.md`` §5c: ``phi`` is exact above 19.9).

ENV: ``gpdla`` (jax-free).

    python validation/absorber_ladder/completeness/build_cal_tables.py \
        --family 2lpt0 london0 saclay0 \
        --out /scratch/.../absorber_ladder_2026-09-13/completeness
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_DIAG = os.path.join(_REPO, "validation", "absorber_diag")
sys.path.insert(0, _DIAG)
sys.path.insert(0, _HERE)

import build_matched_ops as BMO                                   # noqa: E402
from binning import bin_index, coarse_block_sum                   # noqa: E402

FAMILIES = ("2lpt0", "london0", "saclay0")
OPS_DIR = _DIAG


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head():
    try:
        return dict(
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO,
                stderr=subprocess.DEVNULL).decode().strip(),
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_REPO,
                stderr=subprocess.DEVNULL).decode().strip(),
            dirty=bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_REPO, stderr=subprocess.DEVNULL).decode().strip()))
    except Exception as exc:                                # pragma: no cover
        return dict(commit="unknown", error=str(exc))


def matching_pass_with_tid(ep, family, work_dir, floor):
    """``build_matched_ops.matching_pass`` with TARGETID retained.

    The cut / match / op-mask lines are the committed ones, taken from the
    imported module, so nothing is re-implemented: the only addition is that
    the detection and truth TARGETID columns are carried out.
    """
    from CDDF_analysis.hbi.cddf_catalog_hbi import (
        load_and_cut_catalog, load_molly_matrix, _build_qso_lookup)
    from CDDF_analysis.hbi import track_c_tf_saclay as TS

    t0 = time.time()
    cfg = ep._make_cfg(family, work_dir)
    mm = load_molly_matrix(cfg.molly_tsv)
    qso_lookup = _build_qso_lookup(cfg)
    cat_cut, truth_cut, _is_TP, good_mask, meta = load_and_cut_catalog(
        cfg, truth_nhi_floor=float(floor), qso_lookup=qso_lookup,
        host_truth_floor=float(floor))
    TS._snap_off_molly_edges(cat_cut, truth_cut, mm)
    s2n = np.asarray(cat_cut["S2N_RED"], float)
    pdla = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n > cfg.snr_min) & (pdla > cfg.p_dla_min) & good_mask
    det = dict(
        nhat=np.asarray(cat_cut["NHI"], float)[op],
        zobs=np.asarray(cat_cut["Z_DLA"], float)[op],
        snr=np.asarray(cat_cut["S2N_RED"], float)[op],
        nhi_true=np.asarray(cat_cut["NHI_TRUE"], float)[op],
        z_true=np.asarray(cat_cut["Z_TRUE"], float)[op],
        tid=np.asarray(cat_cut["TARGETID"], np.int64)[op])
    BMO._assert_index_convention(ep, det["nhat"], det["zobs"], det["snr"])
    tru = dict(
        nhi=np.asarray(truth_cut["NHI"], float),
        z=np.asarray(truth_cut["Z_DLA"], float),
        snr=np.asarray(truth_cut["S2N_RED"], float),
        tid=np.asarray(truth_cut["TARGETID"], np.int64))
    return dict(det=det, truth=tru, cfg=cfg, meta=meta,
                n_op=int(op.sum()), seconds=time.time() - t0)


def _hist_bks(nhi, z, snr, tid_keep, ntrue, zf, snr_e, extra_mask=None):
    """(B, Kf, S) histogram on the pack's own axes; half-open, S clipped."""
    B, Kf, S = len(ntrue) - 1, len(zf) - 1, len(snr_e) - 1
    b = bin_index(ntrue, nhi)
    k = bin_index(zf, z)
    s = np.clip(bin_index(snr_e, snr), 0, S - 1)
    ok = (b >= 0) & (b < B) & (k >= 0) & (k < Kf) & np.isfinite(nhi)
    if tid_keep is not None:
        ok = ok & tid_keep
    if extra_mask is not None:
        ok = ok & extra_mask
    out = np.zeros((B, Kf, S), float)
    np.add.at(out, (b[ok], k[ok], s[ok]), 1.0)
    return out


def _hist_cks(nhat, z, snr, nhat_e, zf, snr_e, keep):
    C, Kf, S = len(nhat_e) - 1, len(zf) - 1, len(snr_e) - 1
    c = bin_index(nhat_e, nhat)
    k = bin_index(zf, z)
    s = np.clip(bin_index(snr_e, snr), 0, S - 1)
    ok = (c >= 0) & (c < C) & (k >= 0) & (k < Kf) & keep
    out = np.zeros((C, Kf, S), float)
    np.add.at(out, (c[ok], k[ok], s[ok]), 1.0)
    return out


def build(family, out_dir, work_dir=None):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    ep = BMO._load_ep()
    if not np.array_equal(np.asarray(ep.KZ_TO_K), np.repeat([0, 1, 2], 5)):
        raise AssertionError("KZ_TO_K changed — the coarse-K slicing is stale")
    work_dir = work_dir or os.path.join(
        "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13",
        "_work")
    os.makedirs(work_dir, exist_ok=True)

    pk_path = BMO.adopted_pack(family)
    pk = np.load(pk_path, allow_pickle=True)
    ntrue = np.asarray(pk["ntrue_edges"], float)
    nhat = np.asarray(pk["nhat_edges"], float)
    zf = np.asarray(pk["zf_edges"], float)
    snr_e = np.asarray(pk["snr_edges"], float)
    kz = np.asarray(pk["kz_to_K"], int)
    dX = np.asarray(pk["dX"], float)
    tc_pack = np.asarray(pk["truth_counts_bks"], float)

    ops_path = os.path.join(OPS_DIR, f"empirical_ops_{family}.npz")
    ops = np.load(ops_path, allow_pickle=True)

    # ---------------- pass at the basis floor 19.0 -------------------------
    p19 = matching_pass_with_tid(ep, family, work_dir, BMO.BASIS_FLOOR)
    snr_min = float(p19["cfg"].snr_min)
    det, tru = p19["det"], p19["truth"]

    keep_t = np.asarray(tru["snr"], float) > snr_min       # build_truth_counts
    truth_bks = _hist_bks(tru["nhi"], tru["z"], tru["snr"], None,
                          ntrue, zf, snr_e, extra_mask=keep_t)
    det_bks = _hist_bks(det["nhi_true"], det["z_true"], det["snr"], None,
                        ntrue, zf, snr_e)

    gates = {}
    gates["truth_vs_pack_EQUAL"] = bool(np.array_equal(truth_bks, tc_pack))
    gates["truth_max_abs_diff"] = float(np.abs(truth_bks - tc_pack).max())
    ref_det = np.asarray(ops["N_det_all_bks_true_z"], float)
    gates["det_vs_ops_EQUAL"] = bool(np.array_equal(det_bks, ref_det))
    gates["det_max_abs_diff"] = float(np.abs(det_bks - ref_det).max())
    if not (gates["truth_vs_pack_EQUAL"] and gates["det_vs_ops_EQUAL"]):
        print(json.dumps(gates, indent=1), file=sys.stderr)
        raise SystemExit(f"[{family}] REBUILD GATE FAILED — nothing written")

    te_d, to_d = _parity(det["tid"])
    te_t, to_t = _parity(tru["tid"])
    det_E = _hist_bks(det["nhi_true"], det["z_true"], det["snr"], te_d,
                      ntrue, zf, snr_e)
    det_O = _hist_bks(det["nhi_true"], det["z_true"], det["snr"], to_d,
                      ntrue, zf, snr_e)
    tru_E = _hist_bks(tru["nhi"], tru["z"], tru["snr"], te_t,
                      ntrue, zf, snr_e, extra_mask=keep_t)
    tru_O = _hist_bks(tru["nhi"], tru["z"], tru["snr"], to_t,
                      ntrue, zf, snr_e, extra_mask=keep_t)
    gates["parity_partition_det_EQUAL"] = bool(
        np.array_equal(det_E + det_O, det_bks))
    gates["parity_partition_truth_EQUAL"] = bool(
        np.array_equal(tru_E + tru_O, truth_bks))
    if not (gates["parity_partition_det_EQUAL"]
            and gates["parity_partition_truth_EQUAL"]):
        raise SystemExit(f"[{family}] PARITY PARTITION GATE FAILED")

    # per-stratum continuous S/N covariate, from the TRUTH population
    S = len(snr_e) - 1
    s_of_truth = np.clip(bin_index(snr_e, tru["snr"]), 0, S - 1)[keep_t]
    v = np.asarray(tru["snr"], float)[keep_t]
    snr_med = np.full(S, np.nan)
    snr_mean = np.full(S, np.nan)
    snr_logmed = np.full(S, np.nan)
    for s in range(S):
        m = s_of_truth == s
        if m.sum() > 0:
            snr_med[s] = float(np.median(v[m]))
            snr_mean[s] = float(v[m].mean())
            snr_logmed[s] = float(np.median(np.log10(np.maximum(v[m], 1e-3))))

    # ---------------- pass at the census floor 17.2 (P6b) ------------------
    p172 = matching_pass_with_tid(ep, family, work_dir, BMO.CENSUS_FLOOR)
    d2 = p172["det"]
    n2 = d2["nhi_true"]
    h2 = np.isfinite(n2)
    m_p6b = h2 & (n2 >= BMO.CENSUS_FLOOR - BMO.SLOT_EPS) \
        & (n2 < BMO.BASIS_FLOOR - BMO.SLOT_EPS)
    P6b = _hist_cks(d2["nhat"], d2["zobs"], d2["snr"], nhat, zf, snr_e, m_p6b)
    ref_p6b = np.asarray(ops["P6b_cks"], float)
    gates["P6b_vs_ops_EQUAL"] = bool(np.array_equal(P6b, ref_p6b))
    gates["P6b_max_abs_diff"] = float(np.abs(P6b - ref_p6b).max())
    if not gates["P6b_vs_ops_EQUAL"]:
        raise SystemExit(f"[{family}] P6b GATE FAILED")
    pe, po = _parity(d2["tid"])
    P6b_E = _hist_cks(d2["nhat"], d2["zobs"], d2["snr"], nhat, zf, snr_e,
                      m_p6b & pe)
    P6b_O = _hist_cks(d2["nhat"], d2["zobs"], d2["snr"], nhat, zf, snr_e,
                      m_p6b & po)

    n_tid = np.unique(tru["tid"]).size
    n_tid_even = np.unique(tru["tid"][te_t]).size
    prov = dict(
        role=("completeness calibration tables with sightline halves "
              "(TARGETID parity); VALIDATION-ONLY, no sampler, "
              "CDDF_analysis untouched"),
        family=family, built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        recipe="validation/absorber_ladder/completeness/build_cal_tables.py",
        reuses="validation/absorber_diag/build_matched_ops.py (imported)",
        git=_git_head(), python=platform.python_version(),
        numpy=np.__version__,
        pack=pk_path, pack_sha256=_sha256(pk_path),
        ops=ops_path, ops_sha256=_sha256(ops_path),
        molly_tsv=str(p19["cfg"].molly_tsv),
        snr_min=snr_min, p_dla_min=float(p19["cfg"].p_dla_min),
        matching_floor_in_basis=BMO.BASIS_FLOOR,
        matching_floor_census=BMO.CENSUS_FLOOR,
        detection_probability_convention=(
            "det_bks counts matched detections with NO observed-grid "
            "restriction, so det/truth is the pure detection probability "
            "C_det; the in-grid counting fraction phi stays with the kernel"),
        n_truth=float(truth_bks.sum()), n_det=float(det_bks.sum()),
        n_P6b=float(P6b.sum()),
        n_sightlines_truth=int(n_tid), n_sightlines_truth_even=int(n_tid_even),
        parity_even_truth_frac=float(tru_E.sum() / max(truth_bks.sum(), 1)),
        parity_even_det_frac=float(det_E.sum() / max(det_bks.sum(), 1)),
        gates=gates, wall_s=round(time.time() - t_start, 1))

    out = os.path.join(out_dir, f"cal_table_{family}.npz")
    np.savez_compressed(
        out,
        truth_bks=truth_bks, det_bks=det_bks,
        truth_bks_E=tru_E, truth_bks_O=tru_O,
        det_bks_E=det_E, det_bks_O=det_O,
        P6b_cks=P6b, P6b_cks_E=P6b_E, P6b_cks_O=P6b_O,
        snr_med_s=snr_med, snr_mean_s=snr_mean, snr_logmed_s=snr_logmed,
        dX=dX, ntrue_edges=ntrue, nhat_edges=nhat, zf_edges=zf,
        snr_edges=snr_e, kz_to_K=kz,
        molly_n_det=np.asarray(pk["molly_n_det"], float),
        molly_n_tot=np.asarray(pk["molly_n_tot"], float),
        molly_nhi_edges=np.asarray(pk["molly_nhi_edges"], float),
        g_grid=np.asarray(pk["g_grid"], float),
        provenance=np.array(json.dumps(prov, indent=1), dtype=object))
    with open(out.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    print(f"[{family}] GATES PASSED; wrote {out} "
          f"(truth={truth_bks.sum():.0f} det={det_bks.sum():.0f} "
          f"P6b={P6b.sum():.0f}, {prov['wall_s']:.0f}s)", flush=True)
    return out


def _parity(tid):
    t = np.asarray(tid, np.int64)
    even = (t % 2) == 0
    return even, ~even


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", nargs="+", default=list(FAMILIES))
    ap.add_argument("--out", default=("/scratch/cavestru_root/cavestru0/mfho/"
                                      "absorber_ladder_2026-09-13/"
                                      "completeness"))
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    for fam in a.family:
        build(fam, a.out, a.work)


if __name__ == "__main__":
    main()
