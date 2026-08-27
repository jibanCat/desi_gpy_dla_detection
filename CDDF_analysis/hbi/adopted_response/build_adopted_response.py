#!/usr/bin/env python
"""build_adopted_response.py — committed driver of record for the adopted response operator
(`adopted_response/v1.1`), PI ruling 4 of 2026-08-26. See README.md in this directory.

Runs the recovered chain verbatim (the scripts are copied unchanged into --work-dir so their
HERE-relative inputs/outputs stay together), then assembles the operator exactly as the
2026-08-16 assembly heredoc did (same keys, same provenance text; `code_commit` = current HEAD,
plus `builder`/`rebuild_of` fields), and optionally compares every array with a reference file.

    python CDDF_analysis/hbi/adopted_response/build_adopted_response.py \
        --work-dir <scratch dir> --out <adopted_response_v1p1_rebuild.npz> \
        [--compare /scratch/.../track_c/stage0/adopted_response_v1p1.npz] [--skip-logo] [--skip-gb]

Env: gpdla for the chain (numpy, fitsio, multiprocessing); gb_audit needs gpdla-hbi (--py-hbi).
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SCRIPTS = ("stage1b_events_full.py", "fitlib.py", "run_d2b.py", "run_d2b_lib.py",
           "v1_logo.py", "boot_carrier.py", "gb_audit.py")
CHANGE_TEXT = ("v1 -> v1.1: kernel population rebuilt under the HIERARCHICAL "
               "tilt match (PI ruling 2026-08-17 G-B fix; cddf_catalog_hbi.py "
               "hierarchical_v2_20260817) - primary >=matrix-floor assignments "
               "preserved; low-floor pool assigns hostless detections only. "
               "G-B: B1 integer-exact, B2 ZERO mismatches (66,481 events). "
               "LOGO 15/15 PASS; carrier 96 draws unit-gate <=4.4e-6; "
               "shared cubic -0.0364+/-0.0081.")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _run(py, script, work, log):
    cmd = [py, os.path.join(work, script)]
    print(f"=== {script}", flush=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n"); f.flush()
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                    MKL_NUM_THREADS="1"))
    if r.returncode:
        sys.exit(f"{script} FAILED rc={r.returncode} (see {log})")


def assemble(work, out, commit):
    vz = np.load(os.path.join(work, "d2b_variants.npz"))
    car = np.load(os.path.join(work, "adopted_carrier_ensemble.npz"))
    np.savez_compressed(
        out,
        mu_coef=vz["ml_shared3__mu"], sig_coef=vz["ml_shared3__sig"],
        skew_coef=vz["ml_shared3__skew"], fit_rng=vz["ml_shared3__rng"],
        N_ref=vz["N_ref"],
        carrier_mu=car["mu"], carrier_sig=car["sig"], carrier_skew=car["skew"],
        carrier_shared3=car["shared3"], carrier_unit_gate=car["unit_gate"],
        provenance=np.array(json.dumps(dict(
            schema="adopted_response/v1.1", change=CHANGE_TEXT, code_commit=commit,
            seed0=20260818,
            builder="CDDF_analysis/hbi/adopted_response/build_adopted_response.py",
            rebuild_of="track_c/stage0/adopted_response_v1p1.npz (2026-08-16 17:24 EDT, sha256 8fb580b5...)"))))
    print("wrote", out, "commit", commit[:8])


def compare(new, ref):
    a, b = np.load(new, allow_pickle=True), np.load(ref, allow_pickle=True)
    bit, maxd, notes = True, 0.0, []
    for k in sorted(set(a.files) | set(b.files)):
        if k == "provenance":
            continue
        if k not in a.files or k not in b.files:
            notes.append(f"{k}: only in {'new' if k in a.files else 'ref'}"); bit = False; continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            notes.append(f"{k}: shape {x.shape} vs {y.shape}"); bit = False; continue
        if not np.array_equal(x, y):
            bit = False
            d = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
            maxd = max(maxd, d)
            notes.append(f"{k}: max|d|={d:.3e} rel={d / max(float(np.nanmax(np.abs(y))), 1e-300):.3e}")
    print("MATRIX adopted_response_v1p1 rebuild vs reference:",
          "BITWISE-ARRAYS" if bit else f"DIFFERS max|d|={maxd:.3e}",
          f"| sha_new {_sha(new)[:16]} sha_ref {_sha(ref)[:16]}")
    for n in notes:
        print("   ", n)
    return bit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--skip-logo", action="store_true")
    ap.add_argument("--skip-gb", action="store_true")
    ap.add_argument("--py", default=sys.executable, help="gpdla python for the chain")
    ap.add_argument("--py-hbi", default=os.path.expanduser("~/.conda/envs/gpdla-hbi/bin/python"))
    a = ap.parse_args()
    work = os.path.abspath(a.work_dir); os.makedirs(work, exist_ok=True)
    log = os.path.join(work, "build_adopted_response.log")
    for s in SCRIPTS:
        shutil.copy2(os.path.join(HERE, s), os.path.join(work, s))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    print(f"code={commit} work={work}", flush=True)
    _run(a.py, "stage1b_events_full.py", work, log)
    _run(a.py, "run_d2b.py", work, log)
    if not a.skip_logo:
        _run(a.py, "v1_logo.py", work, log)
    _run(a.py, "boot_carrier.py", work, log)
    if not a.skip_gb:
        _run(a.py_hbi, "gb_audit.py", work, log)
    assemble(work, a.out, commit)
    if a.compare:
        ok = compare(a.out, a.compare)
        sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
