#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-loud preflight for the canonical C Voigt extension (_voigt.so).

Production inference must run the compiled C Voigt path. The import sites
(e.g. dla_gp.py) fall back to the ~100x slower — and numerically different —
pure-Python voigt_absorption with only a RuntimeWarning, which a batch log
can silently swallow (observed at NERSC, 2026-08-12 execution report §3.2).

This preflight replicates the exact production import condition and exits
nonzero, loudly, if the C extension would not be used:

    from gpy_dla_detection.voigt_fast import VoigtProfile
    VoigtProfile().compute_voigt_profile(...)

It does NOT modify any science module. On success it prints a one-line PASS
plus a provenance fingerprint (paths + sha256 of _voigt.so / ctypes_voigt.c,
resolved libcerf); on failure it prints the reason and exits 1.

Usage:
    python tools/voigt_preflight.py [--repo-root PATH] [--json OUT.json]
Exit codes: 0 = C extension loads and evaluates; 1 = it would fall back.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolved_libcerf(so_path):
    """Return the libcerf line(s) from ldd, or a note if ldd unavailable."""
    try:
        out = subprocess.run(["ldd", so_path], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "ldd unavailable"
    lines = [ln.strip() for ln in out.splitlines() if "cerf" in ln.lower()]
    return "; ".join(lines) if lines else "no libcerf line in ldd output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--json", default=None,
                    help="also write the fingerprint to this JSON file")
    args = ap.parse_args()

    root = os.path.abspath(args.repo_root)
    pkg_dir = os.path.join(root, "gpy_dla_detection")
    so_path = os.path.join(pkg_dir, "_voigt.so")
    c_path = os.path.join(pkg_dir, "ctypes_voigt.c")

    def fail(reason):
        print(f"VOIGT PREFLIGHT FAIL: {reason}", file=sys.stderr)
        print("Production inference would silently fall back to the "
              "pure-Python Voigt (~100x slower, different numerical path). "
              "Build the C extension first: bash tools/build_voigt.sh "
              "(see README 'Compiling the C Voigt function').",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(so_path):
        fail(f"_voigt.so not found at {so_path} "
             "(gitignored local build artifact; a fresh checkout does not "
             "contain it)")

    # Replicate the production import condition exactly (dla_gp.py:67-70).
    sys.path.insert(0, root)
    try:
        from gpy_dla_detection.voigt_fast import VoigtProfile
        profile_fn = VoigtProfile().compute_voigt_profile
    except (OSError, ImportError) as e:
        fail(f"C extension failed to load/bind: {e!r}")

    # Evaluate once on a small grid: catches an .so that loads but cannot
    # execute (missing symbol resolution is deferred until call on some
    # platforms / broken ABI).
    import numpy as np
    lam = np.linspace(4000.0, 4100.0, 64)
    try:
        prof = profile_fn(lam, nhi=20.3, z_dla=2.35)
    except Exception as e:  # any failure here means production would crash
        fail(f"C extension loaded but evaluation failed: {e!r}")
    if not (np.all(np.isfinite(prof)) and prof.min() >= 0.0
            and prof.max() <= 1.0 + 1e-12):
        fail(f"C extension produced non-physical output "
             f"(min {prof.min()!r}, max {prof.max()!r})")

    fp = {
        "verdict": "PASS",
        "so_path": so_path,
        "so_sha256": sha256(so_path),
        "ctypes_voigt_c_sha256": sha256(c_path) if os.path.isfile(c_path)
        else "MISSING",
        "libcerf": resolved_libcerf(so_path),
        "python": sys.version.split()[0],
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(fp, fh, indent=1)
    print("VOIGT PREFLIGHT PASS: canonical C extension loads and evaluates "
          "(no pure-Python fallback).")
    for k in ("so_path", "so_sha256", "ctypes_voigt_c_sha256", "libcerf"):
        print(f"  {k}: {fp[k]}")
    sys.exit(0)


if __name__ == "__main__":
    main()
