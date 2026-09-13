#!/usr/bin/env python
"""build_a0_support.py — A0: rebuild the MOCK truth histogram and the MOCK
FP-truth census on the SAME support as the scan packs' ``counts`` / ``dX``
(collar 3300 km/s), and stamp every product with a ``support_id``.

PI ruling 2026-09-13b §3 (A0, mandatory):
  "Rebuild mock truth on the same collar support as counts/path; rebuild
   ORACLE/census on that support ... Support consistency becomes a
   machine-enforced invariant."

THE DEFECT (``validation/absorber_diag/OPERATOR_FORENSICS_REPORT.md`` §7).
``CDDF_analysis/hbi_mcmc/build_scan_packs.py`` rebuilds ``counts``, ``dX``,
``dX_coarse_committed`` and ``fp_E_alloc`` at collar 3000 + b km/s and copies
EVERY OTHER ARRAY byte-identically from the adopted (collar-3000) pack — so the
packs of record carry a 3000 km/s ``truth_counts`` / ``truth_counts_bks``
against a 3300 km/s data/exposure pair.  Every mock bias is understated by
0.45-0.49 pp.  ``validation/fp_ladder/build_fp_census.py`` was likewise gated
against the collar-3000 adopted pack, so the ORACLE FP pin is over-pinned too.

WHAT THIS SCRIPT DOES, per family:
  1. matching pass at the BASIS floor 19.0 (the floor the pack's own
     ``truth_counts`` was built at), through the committed
     ``cddf_catalog_hbi.load_and_cut_catalog``;
  2. GATE: the rebuilt collar-3000 ``truth_counts`` / ``truth_counts_bks`` must
     equal the ADOPTED pack's arrays BIT-EXACTLY (the same gate
     ``validation/absorber_diag/build_matched_ops.py`` passes), and the scan
     pack's copies must be byte-identical to them;
  3. rebuild the truth histogram at collar 3300 with the collar geometry of
     ``make_lambda_z_BAL_cuts`` (``use_truth_z=False`` on the truth side, which
     is what ``build_scan_packs`` uses for ``counts`` as well);
  4. matching pass at the CENSUS floor 17.2, GATED bit-exactly against the
     census NPZ on disk, then rebuilt at collar 3300 for every host slot;
  5. write ``scanpack_<fam>_b300_A0.npz`` (truth planes replaced; NOTHING else
     touched) and ``fp_census_<fam>_A0.npz`` + ``.json``, each with a
     ``.support.json`` stamp and a ``.provenance.json`` sidecar, + SHA256SUMS.

The packs of record are NEVER overwritten (they are read-only anyway); the A0
products go to a separate directory.

NO SAMPLER IS RUN.  Nothing under ``CDDF_analysis/`` is modified.
ENV: ``gpdla`` (jax-free; ``extract_pack.py`` / ``pack.py`` loaded FILE-DIRECTLY).

Usage
-----
    python validation/absorber_ladder/support/build_a0_support.py \
        --family 2lpt0 london0 saclay0 \
        --out /scratch/.../absorber_ladder_2026-09-13/support
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import importlib.util as ilu
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_ABSDIAG = os.path.join(_REPO, "validation", "absorber_diag")
sys.path.insert(0, _HERE)
sys.path.insert(0, _ABSDIAG)

from support_contract import (                                   # noqa: E402
    NO_TRUTH_SIDE, ROW_SELECTION_FIELDS, SUPPORT_FIELDS, Z_CUT_ZDLA_ONLY,
    Z_CUT_ZDLA_OR_ZTRUE, catalogue_identity, check_support_consistency,
    file_sha256, stamp, stamp_array, support_id,
)

FAMILIES = ("2lpt0", "london0", "saclay0")

PACK_OF_RECORD_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                      "real_pack_v2_20260821")
ADOPTED_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "adopted_packs_v2p2_20260821")
CENSUS_DIR = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
              "census")

ADOPTED_COLLAR = 3000.0          # the collar every 'other array' was built at
SCANPACK_B = 300                 # the scan pack of record
SCANPACK_COLLAR = 3000.0 + SCANPACK_B

BASIS_FLOOR = 19.0               # ntrue_edges[0]; the pack's truth_counts floor
CENSUS_FLOOR = 17.2
SLOT_EPS = 1e-9
HOST_SLOT_NAMES = ("host_17p2_19p0", "host_19p0_19p5", "host_19p5_19p7",
                   "host_19p7_21p6", "host_ge_21p6")
CENSUS_BLOCKS = ("counts_all", "hostless") + HOST_SLOT_NAMES

LYA = 1215.67
C_KMS = 299792.458

_SHA_CACHE: dict = {}


def pack_of_record(fam):
    return os.path.join(PACK_OF_RECORD_DIR, f"scanpack_{fam}_b{SCANPACK_B}.npz")


def adopted_pack(fam):
    return os.path.join(ADOPTED_DIR,
                        f"modelA_pack_{fam}_bw0p2_pad19p0_molly172_v2.npz")


def census_of_record(fam):
    return os.path.join(CENSUS_DIR, f"fp_census_{fam}.npz")


# ---------------------------------------------------------------------------
def _git_head():
    try:
        return dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=_REPO, stderr=subprocess.DEVNULL
                                           ).decode().strip(),
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_REPO,
                stderr=subprocess.DEVNULL).decode().strip(),
            dirty=bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_REPO, stderr=subprocess.DEVNULL).decode().strip()))
    except Exception as exc:                                # pragma: no cover
        return dict(commit="unknown", error=str(exc))


def _load_module(path, name):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bmo():
    """``validation/absorber_diag/build_matched_ops.py`` — the machinery that
    already reproduces ``truth_counts_bks`` bit-exactly at collar 3000."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    return _load_module(os.path.join(_ABSDIAG, "build_matched_ops.py"),
                        "_a0_build_matched_ops")


# ---------------------------------------------------------------------------
# the support declarations
# ---------------------------------------------------------------------------
def family_support(cfg, catalog_dir, truth_path, bal_cat_path, *, collar_kms,
                   truth_host_floor, z_cut_columns):
    """One support declaration for one object of one family."""
    return support_id(
        collar_kms=float(collar_kms),
        snr_min=float(cfg.snr_min),
        z_window=(float(cfg.z_qso_min), float(cfg.z_qso_max)),
        p_dla_min=float(cfg.p_dla_min),
        lya_only_lam_min=float(cfg.lam_rf_min),
        lam_rf_max=float(cfg.lam_rf_max),
        z_cut_columns=z_cut_columns,
        quality_cut="DLAFLAG==0",
        truth_host_floor=truth_host_floor,
        bal_policy=("mock:drop_all_TARGETID_in_bal_cat(no_bal=True)+DLAFLAG==0;"
                    "bal_cat_sha12=" + file_sha256(bal_cat_path,
                                                   _SHA_CACHE)[:12]),
        catalogue_id=catalogue_identity(catalog_dir, cache=_SHA_CACHE),
        truth_catalogue_sha256=file_sha256(truth_path, _SHA_CACHE),
    )


# ---------------------------------------------------------------------------
# the Z_DLA-only counts recipe of ``build_scan_packs.py`` (diagnostic)
# ---------------------------------------------------------------------------
def scanpack_counts_recipe(ep, fam, collars):
    """``build_scan_packs.build_family``'s counts block, verbatim, per collar.

    Reproduced here ONLY to (a) prove the scan pack's counts selection is what
    this script says it is (bit-exact regression at collar 3300), and (b)
    isolate the z-column convention from the collar at fixed collar 3000.
    """
    import fitsio
    m = ep.MOCKS[fam]
    catf = sorted(glob.glob(os.path.join(m["catalog_dir"], "dlacat*.fits")))
    cat = (np.concatenate([fitsio.read(f, ext=1) for f in catf])
           if len(catf) > 1 else fitsio.read(catf[0], ext=1))
    bal = np.unique(fitsio.read(m["bal_cat_path"], ext=1,
                                columns=["TARGETID"])["TARGETID"]
                    .astype(np.int64))
    tid = cat["TARGETID"].astype(np.int64)
    zqc = np.asarray(cat["Z_QSO"], float)
    zd = np.asarray(cat["Z_DLA"], float)
    nhi = np.asarray(cat["NHI"], float)
    snr = np.asarray(cat["SNR_REDSIDE"], float)
    base = ((np.asarray(cat["DLAFLAG"], int) == 0)
            & (np.asarray(cat["P_DLA"], float) > 0.99)
            & (snr > 2.0) & ~np.isin(tid, bal)
            & (zqc > 2.0) & (zqc < 4.25))
    out = {}
    for coll_kms in collars:
        coll = float(coll_kms) / C_KMS
        z_lo = np.maximum(3600.0 / LYA - 1.0,
                          1025.0 * (1 + zqc) / LYA - 1.0 + coll)
        z_hi = np.minimum(zqc - coll, 1216.0 * (1 + zqc) / LYA - 1.0 - coll)
        op = base & (zd > z_lo) & (zd < z_hi)
        counts, _ = ep.bin_counts_cks(nhi[op], zd[op], snr[op])
        out[float(coll_kms)] = counts.astype(np.int64)
    return out


# ---------------------------------------------------------------------------
def build(family, out_dir, work_dir=None, zcol_diagnostic=True):
    t0 = time.time()
    bmo = _bmo()
    ep = bmo._load_ep()
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = work_dir or os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)
    rec: dict = dict(family=family)

    if not np.array_equal(np.asarray(ep.KZ_TO_K), np.repeat([0, 1, 2], 5)):
        raise AssertionError("KZ_TO_K changed — the coarse-K slicing is stale")

    sp_path, ad_path, cen_path = (pack_of_record(family), adopted_pack(family),
                                  census_of_record(family))
    sp = np.load(sp_path, allow_pickle=False)
    ad = np.load(ad_path, allow_pickle=False)
    ntrue = np.asarray(ad["ntrue_edges"], float)
    zf = np.asarray(ad["zf_edges"], float)
    snr_e = np.asarray(ad["snr_edges"], float)
    kz = np.asarray(ad["kz_to_K"], int)
    tc_pack = np.asarray(ad["truth_counts"], float)
    tcb_pack = np.asarray(ad["truth_counts_bks"], float)

    # the scan pack's truth planes MUST be the byte-identical copies the
    # forensics reported — otherwise the premise of this repair is wrong
    rec["scanpack_truth_is_byte_identical_to_adopted"] = dict(
        truth_counts=bool(np.array_equal(np.asarray(sp["truth_counts"], float),
                                         tc_pack)),
        truth_counts_bks=bool(np.array_equal(
            np.asarray(sp["truth_counts_bks"], float), tcb_pack)))
    if not all(rec["scanpack_truth_is_byte_identical_to_adopted"].values()):
        raise SystemExit("PREMISE FAILED: the scan pack's truth planes are NOT "
                         "byte-identical copies of the adopted pack's")

    # ---- 1/2/3. the BASIS-floor (19.0) truth pass --------------------------
    p19 = bmo.matching_pass(ep, family, work_dir, BASIS_FLOOR)
    cfg = p19["cfg"]
    snr_min = float(cfg.snr_min)
    lam_lo, lam_hi = float(cfg.lam_rf_min), float(cfg.lam_rf_max)
    tru = p19["truth"]

    tcb_3000, _, _ = bmo.truth_hist_bks(tru, ntrue, zf, snr_e, snr_min)
    tc_3000 = tcb_3000.sum(axis=2)
    gate = dict(
        adopted_pack=ad_path, adopted_pack_sha256=file_sha256(ad_path, _SHA_CACHE),
        truth_counts_bks_EQUAL=bool(np.array_equal(tcb_3000, tcb_pack)),
        truth_counts_EQUAL=bool(np.array_equal(tc_3000, tc_pack)),
        rebuilt_total=float(tcb_3000.sum()), pack_total=float(tcb_pack.sum()),
        max_abs_cell_difference=float(np.abs(tcb_3000 - tcb_pack).max()))
    if not (gate["truth_counts_bks_EQUAL"] and gate["truth_counts_EQUAL"]):
        print(json.dumps(gate, indent=1), file=sys.stderr)
        raise SystemExit("COLLAR-3000 TRUTH GATE FAILED — nothing written")
    rec["truth_gate_collar3000"] = gate
    print(f"[{family}] TRUTH GATE @3000 PASSED ({tcb_3000.sum():.0f} systems, "
          f"bit-exact vs the adopted pack)", flush=True)

    # idempotence self-test: the 3000 km/s mask is a NO-OP on rows already cut
    # at 3000 km/s — so applying a 3300 km/s mask on top IS the 3300 selection
    keep30 = bmo.collar_keep(tru["z"], tru["zqso"], ADOPTED_COLLAR, lam_lo, lam_hi)
    rec["collar3000_mask_is_a_noop_on_the_cut_truth"] = dict(
        n_truth_rows=int(keep30.size), n_dropped=int((~keep30).sum()),
        NOOP=bool(keep30.all()))
    if not keep30.all():
        raise SystemExit("SELF-TEST FAILED: the collar-3000 mask drops rows "
                         "from an already-collar-3000 truth table")

    keep33 = bmo.collar_keep(tru["z"], tru["zqso"], SCANPACK_COLLAR, lam_lo, lam_hi)
    tcb_3300, _, _ = bmo.truth_hist_bks(tru, ntrue, zf, snr_e, snr_min,
                                        keep=keep33)
    tc_3300 = tcb_3300.sum(axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_b = np.where(tc_3000.sum(axis=1) > 0,
                       tc_3300.sum(axis=1) / np.maximum(tc_3000.sum(axis=1), 1e-30),
                       np.nan)
        tcK_3000 = np.stack([tc_3000[:, kz == K].sum() for K in range(kz.max() + 1)])
        tcK_3300 = np.stack([tc_3300[:, kz == K].sum() for K in range(kz.max() + 1)])
        r_k_fine = np.where(tc_3000.sum(axis=0) > 0,
                            tc_3300.sum(axis=0) / np.maximum(tc_3000.sum(axis=0), 1e-30),
                            np.nan)
    rec["truth_collar"] = dict(
        collar_3000_total=float(tcb_3000.sum()),
        collar_3300_total=float(tcb_3300.sum()),
        ratio_total=float(tcb_3300.sum() / tcb_3000.sum()),
        ratio_by_b=[None if not np.isfinite(x) else round(float(x), 6) for x in r_b],
        ntrue_edges=[float(x) for x in ntrue],
        ratio_by_K=[round(float(a / b), 6) for a, b in zip(tcK_3300, tcK_3000)],
        counts_by_K_3000=[float(x) for x in tcK_3000],
        counts_by_K_3300=[float(x) for x in tcK_3300],
        ratio_by_fine_k=[None if not np.isfinite(x) else round(float(x), 6)
                         for x in r_k_fine],
        n_truth_systems_removed=float(tcb_3000.sum() - tcb_3300.sum()))
    print(f"[{family}] truth @3300 = {tcb_3300.sum():.0f} "
          f"(ratio {tcb_3300.sum()/tcb_3000.sum():.6f})", flush=True)

    # ---- 4. the CENSUS-floor (17.2) pass ----------------------------------
    p172 = bmo.matching_pass(ep, family, work_dir, CENSUS_FLOOR)
    d = p172["det"]
    n_true = d["nhi_true"]
    host = np.isfinite(n_true)
    masks = {"hostless": ~host}
    for name in HOST_SLOT_NAMES:
        lo, hi = _slot_bounds(name)
        masks[name] = (host & (n_true >= lo - SLOT_EPS)
                       & ((n_true < hi - SLOT_EPS) if np.isfinite(hi)
                          else np.ones_like(host)))
    stacked = np.vstack([masks[k] for k in ("hostless",) + HOST_SLOT_NAMES])
    if not np.all(stacked.sum(axis=0) == 1):
        raise SystemExit("SLOT PARTITION FAILED (census floor 17.2)")

    def _blocks(sel):
        out = {}
        out["counts_all"], _ = ep.bin_counts_cks(d["nhat"][sel], d["zobs"][sel],
                                                 d["snr"][sel])
        for name in ("hostless",) + HOST_SLOT_NAMES:
            m = masks[name] & sel
            out[name], _ = ep.bin_counts_cks(d["nhat"][m], d["zobs"][m],
                                             d["snr"][m])
        return out

    all_rows = np.ones(len(n_true), bool)
    cen_3000 = _blocks(all_rows)
    cz = np.load(cen_path, allow_pickle=True)
    cgate = {k: bool(np.array_equal(cen_3000[k].astype(float),
                                    np.asarray(cz[k], float)))
             for k in CENSUS_BLOCKS}
    cgate["census"] = cen_path
    cgate["census_sha256"] = file_sha256(cen_path, _SHA_CACHE)
    if not all(v for k, v in cgate.items() if k in CENSUS_BLOCKS):
        print(json.dumps(cgate, indent=1), file=sys.stderr)
        raise SystemExit("COLLAR-3000 CENSUS GATE FAILED — nothing written")
    rec["census_gate_collar3000"] = cgate
    print(f"[{family}] CENSUS GATE @3000 PASSED (hostless="
          f"{cen_3000['hostless'].sum():.0f})", flush=True)

    # ``make_lambda_z_BAL_cuts(use_truth_z=True)`` on the detection side
    z_for_cut = np.where(np.isfinite(d["z_true"]), d["z_true"], d["zobs"])
    sel33 = (bmo.collar_keep(d["zobs"], d["zqso"], SCANPACK_COLLAR, lam_lo, lam_hi)
             & bmo.collar_keep(z_for_cut, d["zqso"], SCANPACK_COLLAR, lam_lo,
                               lam_hi))
    cen_3300 = _blocks(sel33)

    # the hostless class is z-column INVARIANT iff no hostless row has a finite
    # Z_TRUE (then min/max(Z_DLA, Z_TRUE) == Z_DLA for it)
    rec["hostless_is_z_column_invariant"] = dict(
        n_hostless_rows=int((~host).sum()),
        n_hostless_with_finite_Z_TRUE=int(np.isfinite(d["z_true"][~host]).sum()),
        INVARIANT=bool(not np.isfinite(d["z_true"][~host]).any()))
    rec["census_collar"] = {
        k: dict(collar_3000=int(cen_3000[k].sum()),
                collar_3300=int(cen_3300[k].sum()),
                delta=int(cen_3000[k].sum() - cen_3300[k].sum()),
                ratio=round(float(cen_3300[k].sum()
                                  / max(cen_3000[k].sum(), 1)), 6))
        for k in CENSUS_BLOCKS}
    print(f"[{family}] census @3300: hostless "
          f"{cen_3000['hostless'].sum():.0f} -> {cen_3300['hostless'].sum():.0f} "
          f"(-{cen_3000['hostless'].sum()-cen_3300['hostless'].sum():.0f})",
          flush=True)

    # ---- the z-column diagnostic (a SECOND support difference) ------------
    if zcol_diagnostic:
        zc = scanpack_counts_recipe(ep, family, (ADOPTED_COLLAR, SCANPACK_COLLAR))
        sp_counts = np.asarray(sp["counts"], np.int64)
        rec["z_column_diagnostic"] = dict(
            recipe="CDDF_analysis/hbi_mcmc/build_scan_packs.py counts block "
                   "(Z_DLA-only window, truth never enters the selection)",
            zdla_only_counts_collar3000=int(zc[ADOPTED_COLLAR].sum()),
            zdla_only_counts_collar3300=int(zc[SCANPACK_COLLAR].sum()),
            scanpack_counts=int(sp_counts.sum()),
            REGRESSION_scanpack_reproduced_bit_exactly=bool(
                np.array_equal(zc[SCANPACK_COLLAR], sp_counts)),
            truth_aware_counts_collar3000_adopted_pack=int(
                np.asarray(ad["counts"]).sum()),
            z_column_effect_rows_at_collar3000=int(
                zc[ADOPTED_COLLAR].sum() - np.asarray(ad["counts"]).sum()),
            z_column_effect_pct_at_collar3000=round(
                100.0 * (zc[ADOPTED_COLLAR].sum()
                         / np.asarray(ad["counts"]).sum() - 1), 4),
            collar_effect_rows_zdla_only=int(zc[ADOPTED_COLLAR].sum()
                                             - zc[SCANPACK_COLLAR].sum()))
        if not rec["z_column_diagnostic"][
                "REGRESSION_scanpack_reproduced_bit_exactly"]:
            raise SystemExit("Z-COLUMN DIAGNOSTIC FAILED: the build_scan_packs "
                             "counts recipe does not reproduce the pack of "
                             "record — the selection premise is wrong")

    # ---- supports ---------------------------------------------------------
    m = ep.MOCKS[family]
    def _sup(collar, floor, zcol):
        return family_support(cfg, m["catalog_dir"], str(cfg.truth_path),
                              m["bal_cat_path"], collar_kms=collar,
                              truth_host_floor=floor, z_cut_columns=zcol)

    sup_data_3300 = _sup(SCANPACK_COLLAR, NO_TRUTH_SIDE, Z_CUT_ZDLA_ONLY)
    sup_truth_3300 = _sup(SCANPACK_COLLAR, BASIS_FLOOR, Z_CUT_ZDLA_ONLY)
    sup_truth_3000 = _sup(ADOPTED_COLLAR, BASIS_FLOOR, Z_CUT_ZDLA_ONLY)
    sup_census_3300 = _sup(SCANPACK_COLLAR, CENSUS_FLOOR, Z_CUT_ZDLA_OR_ZTRUE)
    sup_census_3000 = _sup(ADOPTED_COLLAR, CENSUS_FLOOR, Z_CUT_ZDLA_OR_ZTRUE)
    sup_hostless_3300 = _sup(SCANPACK_COLLAR, CENSUS_FLOOR, Z_CUT_ZDLA_ONLY)
    rec["support_ids"] = {
        "scanpack.counts/dX/fp_E_alloc @3300 (Z_DLA-only)": sup_data_3300.sha256,
        "A0 truth_counts/_bks @3300": sup_truth_3300.sha256,
        "OF-RECORD truth_counts/_bks @3000 (the defect)": sup_truth_3000.sha256,
        "A0 census host slots @3300 (truth-aware z)": sup_census_3300.sha256,
        "OF-RECORD census @3000": sup_census_3000.sha256,
        "A0 census hostless @3300 (z-column invariant)": sup_hostless_3300.sha256,
    }
    rec["support_fields_A0_data_plane"] = {k: sup_data_3300.fields[k]
                                           for k in SUPPORT_FIELDS}

    # ---- 5. write ---------------------------------------------------------
    prov_common = dict(
        role="A0 support repair (PI ruling 2026-09-13b §3): mock truth and "
             "FP-truth census rebuilt on the scan pack's own collar-3300 "
             "support; support_id stamped; VALIDATION-ONLY, no sampler",
        family=family, recipe="validation/absorber_ladder/support/"
                              "build_a0_support.py",
        machinery="validation/absorber_diag/build_matched_ops.py "
                  "(matching_pass / truth_hist_bks / collar_keep) + "
                  "CDDF_analysis/hbi_mcmc/extract_pack.py (bin_counts_cks) — "
                  "both loaded, not reimplemented",
        code=_git_head(),
        env=dict(python=platform.python_version(), numpy=np.__version__,
                 conda_prefix=os.environ.get("CONDA_PREFIX"),
                 host=platform.node()),
        catalog_dir=str(m["catalog_dir"]), truth_path=str(cfg.truth_path),
        bal_cat_path=str(m["bal_cat_path"]), molly_tsv=str(cfg.molly_tsv),
        snr_min=snr_min, p_dla_min=float(cfg.p_dla_min),
        lam_rf_min=lam_lo, lam_rf_max=lam_hi,
        z_qso_window=[float(cfg.z_qso_min), float(cfg.z_qso_max)],
        adopted_collar_kms=ADOPTED_COLLAR, a0_collar_kms=SCANPACK_COLLAR,
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        findings=rec)

    written = []

    # (a) the A0 pack: the pack of record with ONLY the truth planes replaced
    raw = dict(np.load(sp_path, allow_pickle=False))
    raw["truth_counts"] = tc_3300.astype(np.asarray(sp["truth_counts"]).dtype)
    raw["truth_counts_bks"] = tcb_3300.astype(
        np.asarray(sp["truth_counts_bks"]).dtype)
    changed = [k for k in raw
               if not np.array_equal(np.asarray(raw[k]), np.asarray(sp[k]))]
    if sorted(changed) != ["truth_counts", "truth_counts_bks"]:
        raise SystemExit(f"A0 pack changed unexpected keys: {sorted(changed)}")
    a0_pack = os.path.join(out_dir, f"scanpack_{family}_b{SCANPACK_B}_A0.npz")
    np.savez_compressed(a0_pack, **raw)
    # the support_id key is NOT embedded: pack.load_pack's schema is a CLOSED
    # contract and rejects unknown keys (verified; see A0_SUPPORT_REPORT.md §7)
    pprov = dict(prov_common)
    pprov.update(
        product=a0_pack, source_pack=sp_path,
        source_pack_sha256=file_sha256(sp_path, _SHA_CACHE),
        change="truth_counts and truth_counts_bks REPLACED by the collar-3300 "
               "rebuild; every other array byte-identical to the pack of "
               "record (asserted key-by-key)",
        keys_changed=sorted(changed),
        support_id_storage="sidecar .support.json ONLY — pack.load_pack "
                           "(_REQUIRED_KEYS/_OPTIONAL_KEYS) is a closed schema "
                           "and raises PackSchemaError on an extra NPZ key",
        truth_counts_total_before=float(tc_pack.sum()),
        truth_counts_total_after=float(tc_3300.sum()))
    with open(a0_pack.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(pprov, fh, indent=1, default=str)
    stamp(a0_pack, sup_data_3300, extra=dict(
        note="A0 pack: counts/dX/fp_E_alloc and truth_counts/_bks now share "
             "the collar-3300 Z_DLA-only row support",
        planes={"counts": dict(sup_data_3300.fields),
                "dX": dict(sup_data_3300.fields),
                "fp_E_alloc": dict(sup_data_3300.fields),
                "truth_counts": dict(sup_truth_3300.fields),
                "truth_counts_bks": dict(sup_truth_3300.fields)}))
    written.append(a0_pack)

    # (b) the A0 census
    a0_cen = os.path.join(out_dir, f"fp_census_{family}_A0.npz")
    np.savez(
        a0_cen,
        counts_19p5=np.asarray(cz["counts_19p5"]),
        counts_tp_19p5=np.asarray(cz["counts_tp_19p5"]),
        counts_unmatched_19p5=np.asarray(cz["counts_unmatched_19p5"]),
        **{k: cen_3300[k] for k in CENSUS_BLOCKS},
        **{k + "_collar3000": cen_3000[k] for k in CENSUS_BLOCKS},
        nhat_edges=ep.NHAT_EDGES, zf_edges=ep.ZF_EDGES, snr_edges=ep.SNR_EDGES,
        zc_edges=ep.ZC_EDGES, kz_to_K=ep.KZ_TO_K,
        host_slot_names=np.array(HOST_SLOT_NAMES),
        support_id=stamp_array(sup_census_3300),
        provenance=np.array(json.dumps(prov_common, default=str)))
    cprov = dict(prov_common)
    cprov.update(
        product=a0_cen, source_census=cen_path,
        source_census_sha256=file_sha256(cen_path, _SHA_CACHE),
        change="every 17.2-floor block (counts_all / hostless / the five host "
               "slots) REBUILT at collar 3300; the collar-3000 arrays are kept "
               "alongside under the '_collar3000' suffix; the 19.5-floor "
               "accounting arrays are copied unchanged (they belong to the "
               "adopted collar-3000 pack and are diagnostic only)",
        census_of_record_totals={k: int(cen_3000[k].sum()) for k in CENSUS_BLOCKS},
        a0_totals={k: int(cen_3300[k].sum()) for k in CENSUS_BLOCKS},
        support_id_storage="embedded NPZ key 'support_id' + sidecar "
                           ".support.json (the census has no closed schema)")
    with open(a0_cen.replace(".npz", ".provenance.json"), "w") as fh:
        json.dump(cprov, fh, indent=1, default=str)
    with open(a0_cen.replace(".npz", ".json"), "w") as fh:
        json.dump(dict(family=family,
                       collar_kms=SCANPACK_COLLAR,
                       totals={k: int(cen_3300[k].sum()) for k in CENSUS_BLOCKS},
                       totals_collar3000={k: int(cen_3000[k].sum())
                                          for k in CENSUS_BLOCKS},
                       deltas=rec["census_collar"],
                       hostless_per_coarse_z_K=[
                           int(cen_3300["hostless"][:, K * 5:(K + 1) * 5, :].sum())
                           for K in range(3)],
                       hostless_per_snr_s=[
                           int(cen_3300["hostless"][:, :, s].sum())
                           for s in range(cen_3300["hostless"].shape[2])],
                       support_id=sup_census_3300.sha256,
                       provenance=cprov), fh, indent=1, default=str)
    stamp(a0_cen, sup_census_3300, extra=dict(
        note="A0 census at collar 3300; the 'hostless' block (the ORACLE FP "
             "pin) is additionally z-column invariant, so it also carries "
             f"support {sup_hostless_3300.short}",
        planes={"hostless": dict(sup_hostless_3300.fields),
                **{k: dict(sup_census_3300.fields) for k in HOST_SLOT_NAMES},
                "counts_all": dict(sup_census_3300.fields)}))
    written.append(a0_cen)

    rec["wall_s"] = round(time.time() - t0, 1)
    print(f"[{family}] wrote {len(written)} products in {rec['wall_s']:.0f}s",
          flush=True)
    return rec, written


def _slot_bounds(name):
    return {"host_17p2_19p0": (17.2, 19.0), "host_19p0_19p5": (19.0, 19.5),
            "host_19p5_19p7": (19.5, 19.7), "host_19p7_21p6": (19.7, 21.6),
            "host_ge_21p6": (21.6, np.inf)}[name]


# ---------------------------------------------------------------------------
def write_sha256sums(out_dir):
    lines = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*"))):
        if os.path.isfile(p) and not p.endswith("SHA256SUMS"):
            lines.append(f"{file_sha256(p)}  {os.path.basename(p)}")
    path = os.path.join(out_dir, "SHA256SUMS")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--no-zcol-diagnostic", action="store_true")
    a = ap.parse_args(argv)
    summary = {}
    for fam in a.family:
        summary[fam], _ = build(fam, a.out, a.work,
                                zcol_diagnostic=not a.no_zcol_diagnostic)
    with open(os.path.join(a.out, "A0_BUILD_SUMMARY.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    # the gate, run on our own products at BOTH levels, recorded not asserted:
    # the pack's truth floor (19.0) and the census floor (17.2) are an INTENDED
    # per-object difference, so the full 12-field level is expected to flag it
    # while the row-selection level — the invariant the PI states — must PASS.
    from support_contract import SupportContractError
    gates = {}
    for fam in a.family:
        pk = os.path.join(a.out, f"scanpack_{fam}_b{SCANPACK_B}_A0.npz")
        cen = os.path.join(a.out, f"fp_census_{fam}_A0.npz")
        for label, args, lvl in (
                ("pack_only.full", (pk, None), SUPPORT_FIELDS),
                ("pack_only.row", (pk, None), ROW_SELECTION_FIELDS),
                ("pack+census.full", (pk, cen), SUPPORT_FIELDS),
                ("pack+census.row", (pk, cen), ROW_SELECTION_FIELDS)):
            key = f"{fam}:{label}"
            try:
                gates[key] = check_support_consistency(args[0], args[1],
                                                       fields=lvl)
            except SupportContractError as exc:
                gates[key] = dict(status="FAIL", error=str(exc))
            print(f"\n=== {key} -> {gates[key]['status']} ===")
            if gates[key]["status"] == "FAIL":
                print(gates[key]["error"])
            else:
                print(" support_id", gates[key]["support_id_short"],
                      "planes", len(gates[key]["planes"]))
    with open(os.path.join(a.out, "A0_SUPPORT_GATES.json"), "w") as fh:
        json.dump(gates, fh, indent=1, default=str)
    print("\nSHA256SUMS ->", write_sha256sums(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
