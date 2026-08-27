#!/usr/bin/env python3
"""frozen_manifest.py — ONE manifest of every frozen/ratified input and output of the
Paper-1 science path, with sha256 and size, and a --verify mode that fails closed on any
change.  Written for the Paper-1 code review (2026-08-26): several frozen inputs (raw
mock catalogues, the archive catalogue, the H2 tables, the scan packs, the Battery-2/3
reference records) were hashed nowhere, and the pooled posterior's own `inputs` block
records the _fdraws hashes under the .json keys.

    python tools/paper1/frozen_manifest.py --write docs/PAPER1_FROZEN_MANIFEST.json
    python tools/paper1/frozen_manifest.py --verify docs/PAPER1_FROZEN_MANIFEST.json

Paths are absolute on GreatLakes; the manifest is the portable record of WHAT was used.
Role tags: input | frozen-output | ratified | record | reference | superseded.
"""
import argparse, hashlib, json, os, pathlib, sys, time

BASE = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata"
NEW = f"{BASE}/real_pack_v2_20260821"
TFHZ = f"{BASE}/track_c/tf_hz"
H2 = "/scratch/cavestru_root/cavestru0/mfho/loa_hz_production/h2_exec"
NOTES = "/home/mfho/desi_gpy_dla_notes"
CAT = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs"

ENTRIES = [
    # ---- product 1: calibration / certified pack
    ("input", "real catalogue (dlacat loa main dark v1)", f"{CAT}/loa_main_dark_v1/dlacat-loa-main-dark-v1.fits"),
    ("input", "QSO catalogue (BI_CIV)", "/nfs/turbo/lsa-cavestru/mfho/DESI/loa/QSO_cat_loa_main_dark_healpix_v2-altbal.fits"),
    ("input", "archive sightline catalogue", "/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis/src_archive_catalog.npy"),
    ("input", "adopted response operator v1.1", f"{BASE}/track_c/stage0/adopted_response_v1p1.npz"),
    ("input", "forward response 2LPT-0 (BH kernel)", f"{BASE}/track_c/stage0/forward_response_2lpt0.npz"),
    ("input", "molly matrix nhi172 (pack completeness source)", "/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi172/molly_matrix.tsv"),
    ("input", "molly matrix nhi195 lya_only (BH)", "/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/figures_molly_nhi195/lya_only/molly_matrix.tsv"),
    ("input", "pc counts nhi172", f"{BASE}/ff_fp_cache/molly_counts_nhi172.npz"),
    ("frozen-output", "certified corrected-g real pack (v2, PRODUCTION)", f"{NEW}/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz"),
    ("frozen-output", "real pack v1.1 (pre-stamp intermediate)", f"{NEW}/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172.npz"),
    ("record", "real pack provenance sidecar", f"{NEW}/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.provenance.json"),
    ("record", "real pack contract guards", f"{NEW}/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.contract_guards.json"),
    ("record", "selection-contract sidecar v1.2", f"{NEW}/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.selection_contract.json"),
    ("frozen-output", "scan pack 2lpt0 b300", f"{NEW}/scanpack_2lpt0_b300.npz"),
    ("frozen-output", "scan pack london0 b300", f"{NEW}/scanpack_london0_b300.npz"),
    ("frozen-output", "scan pack saclay0 b300", f"{NEW}/scanpack_saclay0_b300.npz"),
    ("record", "CERT_G_SUPPORT_CP1_CP2", f"{NEW}/CERT_G_SUPPORT_CP1_CP2.json"),
    ("record", "CERT_G_SUPPORT_CP1", f"{NEW}/CERT_G_SUPPORT_CP1.json"),
    ("record", "CP-1 regeneration sbatch (as archived)", f"{NEW}/run_cp1_regeneration.sbatch"),
    # ---- product 2: mock validation
    ("frozen-output", "CP-2 gate v2 on production packs", f"{NEW}/cp2_validation/perz_gate_v2_cp2_production.json"),
    ("record", "CP-2 bit-reproduction vs Battery 2+3", f"{NEW}/cp2_validation/bitrepro_cp2_vs_battery23.json"),
    ("reference", "Battery 2+3 per-z gate (DIAGPACK_gcons)", f"{NOTES}/figures/2026-08-20_perz_fold_closure/perz_gate_v2_battery2plus3_gcons.json"),
] + [("frozen-output", f"CP-2 validation run {f}", f"{NEW}/cp2_validation/cp2_ln_w1500_{f}_s20260811.json") for f in ("2lpt0", "london0", "saclay0")] \
  + [("reference", f"Battery-3 reference record {f}", f"/gpfs/accounts/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/analysis/perz_20260820/perz_gcons_ln_w1500_{f}_s20260811.json") for f in ("2lpt0", "london0", "saclay0")] + [
    # ---- product 3: frozen real posterior
    ("frozen-output", "FROZEN pooled posterior summary", f"{NEW}/cp3_real/POOLED_ln_real_v2_20260821.json"),
    ("frozen-output", "FROZEN pooled posterior draws", f"{NEW}/cp3_real/POOLED_ln_real_v2_20260821_fdraws.npz"),
    ("record", "FROZEN_STATUS sidecar", f"{NEW}/cp3_real/FROZEN_STATUS.json"),
    ("record", "CP-3 selection stage 1", f"{NEW}/cp3_real/selection_stage1.txt"),
    ("record", "CP-3 selection stage 2", f"{NEW}/cp3_real/selection_stage2.txt"),
] + [("frozen-output", f"CP-3 run {n}", f"{NEW}/cp3_real/{n}.json") for n in ("REAL_ln_deep_s20260821", "REAL_ln_s20260822", "REAL_ln_s20260824", "REAL_ln_s20260825", "REAL_ln_s20260827", "REAL_ln_deep_s20260828")] \
  + [("frozen-output", f"CP-3 run draws {n}", f"{NEW}/cp3_real/{n}_fdraws.npz") for n in ("REAL_ln_deep_s20260821", "REAL_ln_s20260822", "REAL_ln_s20260824", "REAL_ln_s20260825", "REAL_ln_s20260827", "REAL_ln_deep_s20260828")] \
  + [("record", f"CP-3 excluded-and-disclosed run {n}", f"{NEW}/cp3_real/{n}.json") for n in ("REAL_ln_deep_s20260823", "REAL_ln_deep_s20260826")] + [
    ("record", "CONFIG_AMBIGUITY s26 mirror vs pooled", f"{NEW}/cp3_real/CONFIG_AMBIGUITY_s26mirror_vs_pooled.json"),
    ("record", "ZDOMAIN estimands pooled", f"{NEW}/cp3_real/ZDOMAIN_estimands_pooled.json"),
    ("superseded", "real_pack_v1 pooled candidate (defective g)", f"{BASE}/real_pack_v1/cp3_reference/POOLED_ln_real_v1_SUPERSEDED.json"),
    # ---- product 4: BH / H2
    ("ratified", "BH artifact of record (RATIFIED)", f"{TFHZ}/track_c_tf_hz_h2cal_loa0_lya_gapc0.496_RATIFIED_20260826.json"),
    ("frozen-output", "BH source artifact gapc0.496", f"{TFHZ}/track_c_tf_hz_h2cal_loa0_lya_gapc0.496.json"),
    ("record", "H2 C_gap inference", f"{TFHZ}/H2_CGAP_INFERENCE.json"),
    ("record", "architecture lock", f"{TFHZ}/PAPER1_ARCHITECTURE_LOCK.json"),
    ("superseded", "architecture lock superseded 2026-08-15 (candidate value withheld: public repo)", f"{TFHZ}/PAPER1_ARCHITECTURE_LOCK_SUPERSEDED_20260815.json"),
    ("record", "BH anchor [3.5,3.8)", f"{TFHZ}/diag_20260819/tf_hz_gapc0.496_zext35.json"),
    ("record", "BH transport envelope plus", f"{TFHZ}/diag_20260819/tf_hz_gapc0.496_envplus.json"),
    ("record", "BH transport envelope minus", f"{TFHZ}/diag_20260819/tf_hz_gapc0.496_envminus.json"),
    ("record", "BH z>=5 tail audit", f"{TFHZ}/tail_ge5_audit.json"),
    ("record", "BH near-QSO strip sensitivity", f"{TFHZ}/NEARQSO_STRIP_SENSITIVITY.json"),
    ("record", "collar convention audit", f"{TFHZ}/COLLAR_CONVENTION_AUDIT.json"),
] + [("input", f"H2 canonical table {n}", f"{H2}/h2_canonical_{n}.json") for n in ("armA_lya", "armA_lya_nobal", "armA_lyab", "armA_lyab_nobal", "armB_lya", "armB_lya_nobal", "armB_lyab", "armB_lyab_nobal", "ab_transport")] + [
    ("input", "H2 injected archive arm A", f"{H2}/h2_injected_armA.h5"),
    ("input", "H2 injected archive arm B", f"{H2}/h2_injected_armB.h5"),
    ("input", "H2 realized plan arm A", f"{H2}/h2_realized_plan_armA.csv"),
    ("input", "H2 realized plan arm B", f"{H2}/h2_realized_plan_armB.csv"),
    ("input", "H2 BAL TID drop list", f"{H2}/bal_bi_civ_tids.txt"),
    ("input", "loa0 FP companion catalogue file 0-2", "/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/dlacat-v2.8.5-mockcat-0-2.fits"),
    ("input", "loa0 FP companion catalogue file 2-4", "/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/dlacat-v2.8.5-mockcat-2-4.fits"),
    ("input", "loa0 FP companion catalogue file 4-6", "/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/outputs/dlacat-v2.8.5-mockcat-4-6.fits"),
    # ---- products 5-7: ledger, audit product, comparison data
    ("record", "systematics ledger v2.3 r5", f"{NOTES}/figures/2026-08-21_freeze_pathB/ledger_v2p3_cp3.json"),
    ("record", "CDDF recovery audit product", f"{NOTES}/figures/2026-08-26_sys_viz_preview/cddf_recovery_audit.json"),
    ("record", "artifact manifest cp3", f"{NOTES}/figures/2026-08-21_freeze_pathB/ARTIFACT_MANIFEST_cp3.md"),
]
# Directory expansions (PI requirement 2026-08-26 §8: the CHAINS are first-class frozen
# artifacts -- every CP-3 run JSON and its draws, deep reruns, the excluded-and-disclosed
# s23/s26 chains, stdout/logs, selection and pooling records, nuisance/PPC draws; and the
# CP-2 validation runs with their draws; plus the high-z catalogue files).
def expand_dirs():
    out = []
    d = pathlib.Path(f"{CAT}/gl_cddf_loa_hz_v1_20260813/outputs")
    if d.is_dir():
        for f in sorted(d.glob("dlacat-*.fits")):
            out.append(("input", f"high-z catalogue {f.name}", str(f)))
    cp3 = pathlib.Path(f"{NEW}/cp3_real")
    seen = {e[2] for e in ENTRIES}
    for f in sorted(cp3.rglob("*")):
        if f.is_file() and str(f) not in seen:
            role = "frozen-output" if (f.suffix in (".json", ".npz") and f.name.startswith(("REAL_ln", "POOLED"))) else "record"
            out.append((role, f"CP-3 chain set: {f.relative_to(cp3)}", str(f)))
    cp2 = pathlib.Path(f"{NEW}/cp2_validation")
    for f in sorted(cp2.glob("*")):
        if f.is_file() and str(f) not in seen:
            out.append(("record", f"CP-2 validation: {f.name}", str(f)))
    for f in sorted(pathlib.Path(f"{NEW}/logs").glob("*")):
        if f.is_file():
            out.append(("record", f"CP-1/CP-2 slurm log: {f.name}", str(f)))
    return out


def sha256(p, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    rows = []
    for role, desc, path in ENTRIES + expand_dirs():
        p = pathlib.Path(path)
        if not p.is_file():
            rows.append({"role": role, "description": desc, "path": path, "status": "MISSING"})
            continue
        rows.append({"role": role, "description": desc, "path": path, "bytes": p.stat().st_size,
                     "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)),
                     "sha256": sha256(p)})
    return {"role": "Paper-1 frozen manifest (inputs, frozen outputs, ratified artifacts, records); "
                    "written by tools/paper1/frozen_manifest.py",
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "entries": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write"); g.add_argument("--verify")
    ap.add_argument("--prefix", default="", help="verify the copy under <prefix>/<absolute path> (the Turbo archive layout)")
    a = ap.parse_args(argv)
    if a.write:
        m = build()
        pathlib.Path(a.write).write_text(json.dumps(m, indent=1) + "\n")
        miss = [r for r in m["entries"] if r.get("status") == "MISSING"]
        print(f"wrote {a.write}: {len(m['entries'])} entries, {len(miss)} MISSING")
        for r in miss: print("  MISSING", r["path"])
        return 1 if miss else 0
    m = json.loads(pathlib.Path(a.verify).read_text())
    bad = []
    for r in m["entries"]:
        if r.get("status") == "MISSING": continue
        p = pathlib.Path(a.prefix.rstrip("/") + r["path"]) if a.prefix else pathlib.Path(r["path"])
        if not p.is_file(): bad.append((str(p), "now missing")); continue
        if sha256(p) != r["sha256"]: bad.append((r["path"], "sha256 CHANGED"))
    print(f"verified {len(m['entries'])} entries under prefix '{a.prefix or '/'}': {len(bad)} problem(s); manifest sha256 {sha256(pathlib.Path(a.verify))[:16]}")
    for p, why in bad: print("  ", why, p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
