#!/usr/bin/env python
"""Re-derive the deterministic frozen intermediates of Paper 1 from their recorded inputs
and classify each against the product of record: BITWISE / NUMERICALLY(max|d|) / MISMATCH.

Pre-tag review 2026-08-26, requirement 7 (intermediate reproducibility matrix).  Runs in a
scratch work dir; NEVER writes into the frozen directories.  Stages (each independent):

  real-pack   extract_pack_real --cert-2lpt0 / --real (gpdla) / --stamp-v12 (gpdla-hbi)
  pool        cc_pool_posterior on the frozen chains (stage-2 collector command of record)
  zdomain     cc_zdomain_estimand on the frozen pooled posterior
  config      cc_config_ambiguity (s26 mirror vs pooled)
  recovery    cddf_recovery_audit (argv reconstructed from the product's provenance block)
  compare     compare every work-dir product with its product of record and print the matrix

Usage: python tools/paper1/reproduce_intermediates.py --work DIR [--py-gpdla P] [--py-hbi P] STAGE...
"""
import argparse, hashlib, json, os, subprocess, sys
import numpy as np

D = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata"
V2 = f"{D}/real_pack_v2_20260821"
CP3 = f"{V2}/cp3_real"
PACK_V1 = f"{V2}/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172.npz"
PACK_V2 = f"{V2}/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz"
PACK_V2_SHA = "219c43aaa59aeb7f070e223c3e652190b4cabe55683c45679a88522d88b920bb"
REF_2LPT0 = f"{D}/adopted_packs_v2p2_20260821/modelA_pack_2lpt0_bw0p2_pad19p0_molly172_v2.npz"
POOLED = f"{CP3}/POOLED_ln_real_v2_20260821.json"
POOLED_F = f"{CP3}/POOLED_ln_real_v2_20260821_fdraws.npz"
ZDOMAIN = f"{CP3}/ZDOMAIN_estimands_pooled.json"
CONFIG = f"{CP3}/CONFIG_AMBIGUITY_s26mirror_vs_pooled.json"
RECOVERY = "/home/mfho/desi_gpy_dla_notes/figures/2026-08-26_sys_viz_preview/cddf_recovery_audit.json"
PROV_KEYS = {"code_commit", "run_config", "inputs_note", "inputs_fdraws_sha256",
             "inputs_json_sha256", "date", "argv", "provenance", "out"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, py, log):
    print(f"$ {py} {' '.join(cmd)}", flush=True)
    with open(log, "a") as f:
        f.write(f"$ {py} {' '.join(cmd)}\n"); f.flush()
        r = subprocess.run([py] + cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        print(f"  FAILED rc={r.returncode} (see {log})", flush=True)
    return r.returncode


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in PROV_KEYS}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def num_diff(a, b, path="", acc=None):
    """Recursive compare of two JSON trees; returns (max_abs_diff, [structural mismatches])."""
    acc = acc if acc is not None else [0.0, []]
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            acc[1].append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in a.keys() & b.keys():
            num_diff(a[k], b[k], f"{path}/{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc[1].append(f"{path}: len {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            num_diff(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != b:
            acc[0] = max(acc[0], abs(float(a) - float(b)))
    elif a != b:
        if isinstance(a, str) and isinstance(b, str) and (a.startswith("/") or b.startswith("/")):
            pass  # path strings (work dir vs frozen dir) are not science
        else:
            acc[1].append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")
    return acc


def npz_compare(new, ref, ignore=("provenance",)):
    a, b = np.load(new, allow_pickle=True), np.load(ref, allow_pickle=True)
    ka, kb = set(a.files) - set(ignore), set(b.files) - set(ignore)
    notes, maxd, bit = [], 0.0, True
    if ka != kb:
        notes.append(f"keys differ: {sorted(ka ^ kb)}")
    for k in sorted(ka & kb):
        x, y = a[k], b[k]
        if x.dtype.kind in "OUS" or y.dtype.kind in "OUS":
            if not np.array_equal(x, y):
                notes.append(f"{k}: object/str differs"); bit = False
            continue
        if x.shape != y.shape:
            notes.append(f"{k}: shape {x.shape} vs {y.shape}"); bit = False; continue
        if not np.array_equal(x, y):
            bit = False
            d = float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
            maxd = max(maxd, d); notes.append(f"{k}: max|d|={d:.3e}")
    return bit, maxd, notes


def classify(name, bitwise, maxd, notes, sha_new=None, sha_ref=None):
    if sha_new and sha_ref and sha_new == sha_ref:
        cls = "BITWISE (sha256 identical)"
    elif bitwise:
        cls = "BITWISE (all science arrays/values identical; provenance keys differ)"
    elif not [n for n in notes if "differs" in n or "keys" in n or "shape" in n or "len" in n or "vs" in n] and maxd <= 1e-9:
        cls = f"NUMERICALLY (max|d| = {maxd:.3e})"
    else:
        cls = f"MISMATCH (max|d| = {maxd:.3e})"
    print(f"MATRIX {name:32s} {cls}")
    for n in notes[:12]:
        print(f"        {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--py-gpdla", default=os.path.expanduser("~/.conda/envs/gpdla/bin/python"))
    ap.add_argument("--py-hbi", default=os.path.expanduser("~/.conda/envs/gpdla-hbi/bin/python"))
    ap.add_argument("stages", nargs="+")
    a = ap.parse_args()
    W = os.path.abspath(a.work); os.makedirs(W, exist_ok=True)
    log = f"{W}/reproduce_intermediates.log"
    for st in a.stages:
        if st == "real-pack":
            run(["CDDF_analysis/hbi_mcmc/extract_pack_real.py", "--cert-2lpt0", "--out-dir", W, "--ref-pack", REF_2LPT0], a.py_gpdla, log)
            run(["CDDF_analysis/hbi_mcmc/extract_pack_real.py", "--real", "--out-dir", W, "--ref-pack", REF_2LPT0], a.py_gpdla, log)
            run(["CDDF_analysis/hbi_mcmc/extract_pack_real.py", "--stamp-v12", "--out-dir", W], a.py_hbi, log)
        elif st == "pool":
            base = sorted(f"{CP3}/REAL_ln_s2026082{i}.json" for i in range(1, 9))
            deep = sorted(f"{CP3}/REAL_ln_deep_s2026082{i}.json" for i in (1, 3, 6, 8))
            run(["-m", "CDDF_analysis.hbi_mcmc.cc_pool_posterior", "--runs", *base, "--deep", *deep,
                 "--rhat-max", "1.10", "--div-max", "10", "--expect-pack-sha256", PACK_V2_SHA,
                 "--out", f"{W}/POOLED_ln_real_v2_20260821.json"], a.py_hbi, log)
        elif st == "zdomain":
            z = json.load(open(ZDOMAIN))
            cmd = ["-m", "CDDF_analysis.hbi_mcmc.cc_zdomain_estimand", "--pooled", POOLED, "--pack", PACK_V2,
                   "--z-los", *[str(v) for v in z["z_los"]], "--out", f"{W}/ZDOMAIN_estimands_pooled.json"]
            lev = z.get("config_leverage_pct") or {}
            if lev.get("config_run"):   # the recorded alternative-configuration run (s26 mirror) and chain
                cmd += ["--config-run", lev["config_run"], "--chain", str(lev.get("chain", 0))]
            run(cmd, a.py_hbi, log)
        elif st == "config":
            c = json.load(open(CONFIG))
            run(["-m", "CDDF_analysis.hbi_mcmc.cc_config_ambiguity", "--run", c["run"], "--chain", str(c["chain"]),
                 "--n-chains", str(c["n_chains"]), "--pooled", POOLED, "--pack", PACK_V2,
                 "--out", f"{W}/CONFIG_AMBIGUITY_s26mirror_vs_pooled.json"], a.py_hbi, log)
        elif st == "recovery":
            r = json.load(open(RECOVERY)); c = json.load(open(CONFIG))
            prov = r["mock"]["allz"]["provenance"]
            runs = [v["fdraws"] for v in prov.values()]; rj = [v["run_json"] for v in prov.values()]
            slices = ",".join(f"{s[0]}" for s in r["slices"] if s[0] is not None) + "," + f"{r['slices'][-1][1]}"
            rp = r["real"]["provenance"]
            cmd = ["-m", "CDDF_analysis.hbi_mcmc.cddf_recovery_audit", "--runs", *runs, "--run-json", *rj,
                   "--slices", slices, "--real", rp["pooled"], "--real-pack", rp["pack"],
                   "--redges", ",".join(str(x) for x in r["redges"]),
                   "--out", f"{W}/cddf_recovery_audit.json"]
            if rp.get("mirror"):
                cmd += ["--mirror", rp["mirror"], "--mirror-chain", str(rp.get("mirror_chain", c["chain"])),
                        "--mirror-chains", str(rp.get("mirror_chains", c["n_chains"]))]
            run(cmd, a.py_hbi, log)
        elif st == "compare":
            print("=== INTERMEDIATE REPRODUCIBILITY MATRIX (work dir vs product of record)")
            pairs = [("real pack v1 (unstamped)", f"{W}/{os.path.basename(PACK_V1)}", PACK_V1),
                     ("real pack v2 (stamped)", f"{W}/{os.path.basename(PACK_V2)}", PACK_V2),
                     ("pooled draws", f"{W}/POOLED_ln_real_v2_20260821_fdraws.npz", POOLED_F)]
            for name, new, ref in pairs:
                if not os.path.exists(new):
                    print(f"MATRIX {name:32s} NOT RUN ({new} absent)"); continue
                bit, maxd, notes = npz_compare(new, ref)
                classify(name, bit, maxd, notes, sha(new), sha(ref))
            jpairs = [("pooled summary", f"{W}/POOLED_ln_real_v2_20260821.json", POOLED),
                      ("z-domain estimands", f"{W}/ZDOMAIN_estimands_pooled.json", ZDOMAIN),
                      ("config ambiguity (L15)", f"{W}/CONFIG_AMBIGUITY_s26mirror_vs_pooled.json", CONFIG),
                      ("cddf recovery audit", f"{W}/cddf_recovery_audit.json", RECOVERY)]
            for name, new, ref in jpairs:
                if not os.path.exists(new):
                    print(f"MATRIX {name:32s} NOT RUN ({new} absent)"); continue
                s_new, s_ref = sha(new), sha(ref)
                acc = num_diff(strip(json.load(open(new))), strip(json.load(open(ref))))
                classify(name, s_new == s_ref or (acc[0] == 0.0 and not acc[1]), acc[0], acc[1], s_new, s_ref)
        else:
            sys.exit(f"unknown stage {st}")


if __name__ == "__main__":
    main()
