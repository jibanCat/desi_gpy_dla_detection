#!/usr/bin/env python
"""identity_test.py — THE DECISIVE TEST for the real-mode runner.

Runs ``validation/real_c1/run_real_c1.py`` (under --allow-mock-for-identity-test)
and the FROZEN ``validation/fp_ladder/run_ladder.py`` on the SAME mock pack with
the SAME frozen fixed objects, the same seed and the same NUTS configuration,
and asserts

  1. ``np.array_equal`` on the f draws (bit-identical posterior), and
  2. agreement of the headline percentiles the two JSONs share, to 1e-10.

Both runners receive the TRANSPORTED sub-floor term through --extra-fixed-file
(run_ladder's ``--fix P`` truth-pinned census term is NOT used), so the model
inputs are argument-for-argument the same object.

MOCK ONLY. Nothing here reads the real pack or the real likelihood.

Usage:
  python validation/real_c1/identity_test.py --out-dir DIR \
      [--chains 4 --warmup 1500 --samples 1000] [--seed 20260811] [-J 8 -j 3]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

L = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
PACK = os.path.join(L, "support_v3", "scanpack_2lpt0_b300_v3.npz")
CENSUS = os.path.join(L, "support_v3", "fp_census_2lpt0_v3.npz")
OPS = os.path.join(L, "support_v3", "empirical_ops_2lpt0_v3.npz")
MG = os.path.join(L, "response_review", "candidates", "Mg_B_2lpt0.npz")
CF = os.path.join(L, "completeness", "C_C1nsadd_2lpt0.npz")
MU = os.path.join(L, "real_c1", "identity_inputs", "mu_extra_P6bcal_2lpt0.npz")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("-J", "--lam-imputations", type=int, default=8)
    ap.add_argument("-j", "--lam-imputation", type=int, default=3)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=REPO, JAX_PLATFORMS="cpu",
               HDF5_USE_FILE_LOCKING="FALSE", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    common = ["--pack", PACK, "--ladder", "M1CUT", "--seed", str(a.seed),
              "--chains", str(a.chains), "--warmup", str(a.warmup),
              "--samples", str(a.samples), "--target-accept", "0.95",
              "--require-support",
              "--mg-fixed-file", MG, "--mg-fixed-key", "Mg",
              "--c-fixed-file", CF, "--extra-fixed-file", MU,
              "--lam-imputations", str(a.lam_imputations),
              "--lam-imputation", str(a.lam_imputation)]
    new_out = os.path.join(a.out_dir, "IDENT_real_runner.json")
    old_out = os.path.join(a.out_dir, "IDENT_run_ladder.json")

    cmds = [
        ([sys.executable, os.path.join(REPO, "validation", "real_c1", "run_real_c1.py")]
         + common + ["--allow-mock-for-identity-test", "--skip-contract-guards",
                     "--stage", "IDENTITY_TEST", "--out", new_out]),
        # the FROZEN mock runner, same inputs; --census/--ops feed only the
        # support gate and the (unused here) FP-truth flag, never the model.
        ([sys.executable, os.path.join(REPO, "validation", "fp_ladder", "run_ladder.py")]
         + common + ["--census", CENSUS, "--ops", OPS,
                     "--stage", "IDENTITY_TEST", "--out", old_out]),
    ]
    for c in cmds:
        print("\n$", " ".join(c), flush=True)
        r = subprocess.run(c, env=env, cwd=REPO)
        if r.returncode != 0:
            raise SystemExit(f"runner failed rc={r.returncode}: {c[1]}")

    zn = np.load(new_out[:-5] + "_fdraws.npz")
    zo = np.load(old_out[:-5] + "_fdraws.npz")
    fn, fo = np.asarray(zn["f"]), np.asarray(zo["f"])
    bit_identical = bool(fn.shape == fo.shape and np.array_equal(fn, fo))
    maxdev = float(np.abs(fn - fo).max()) if fn.shape == fo.shape else float("nan")

    dn, do = json.load(open(new_out)), json.load(open(old_out))
    agree = {}
    worst = 0.0
    for thr in ("20.0", "20.3"):
        ref = do["thresholds"][f"ge{thr}"]
        got = dn["estimands"]["thresholds_allz"][f"ge{float(thr)}"]
        for k in ("post_p16_50_84", "post_p2p5_97p5"):
            d = float(np.abs(np.asarray(ref[k]) - np.asarray(got[k])).max())
            worst = max(worst, d)
            agree[f"ge{thr}.{k}.max_abs_diff"] = d
    # the Paper-1 z bins, both thresholds, against run_ladder's perz_recovery
    n_bins = 0
    for tag in do["perz_recovery"]["estimand"]:
        for r_ref, r_got in zip(do["perz_recovery"]["estimand"][tag]["paper1_bins"],
                                dn["estimands"]["perz_posterior"]["estimand"][tag]["paper1_bins"]):
            if not r_ref.get("available"):
                continue
            d = float(np.abs(np.asarray(r_ref["post_p2p5_16_50_84_97p5"])
                             - np.asarray(r_got["post_p2p5_16_50_84_97p5"])).max())
            worst = max(worst, d)
            n_bins += 1
    agree["paper1_bins_compared"] = n_bins
    agree["worst_max_abs_diff"] = worst
    agree["headline_agree_1e-10"] = bool(worst <= 1e-10)

    verdict = dict(
        f_draws_shape=[int(x) for x in fn.shape],
        f_draws_bit_identical=bit_identical, f_draws_max_abs_dev=maxdev,
        lam_fixed_new=dn["lam_cut"]["lam_fixed"],
        lam_fixed_old=do["diagnostics"]["lam_cut"]["lam_fixed"],
        lam_fixed_equal=bool(dn["lam_cut"]["lam_fixed"]
                             == do["diagnostics"]["lam_cut"]["lam_fixed"]),
        divergences_new=dn["sampler_health"]["divergences"],
        divergences_old=do["diagnostics"]["divergences"],
        headline=agree,
        seed=a.seed, chains=a.chains, warmup=a.warmup, samples=a.samples,
        J=a.lam_imputations, j=a.lam_imputation,
        PASS=bool(bit_identical and agree["headline_agree_1e-10"]))
    p = os.path.join(a.out_dir, "IDENTITY_VERDICT.json")
    json.dump(verdict, open(p, "w"), indent=1)
    print("\n" + json.dumps(verdict, indent=1))
    print("verdict written to", p)
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
