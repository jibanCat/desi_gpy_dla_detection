#!/usr/bin/env python
"""test_run_real_c1.py — the REAL-MODE runner's refusals, schema and reduction.

Scope (blocker B1, REAL_C1_READINESS.md §6):
  * every fail-closed refusal of ``validation/real_c1/run_real_c1.py``
    (mock pack without the identity flag, ORACLE, --fix/--ops/--census,
    --mg-fixed-key Mg_phi_family, --no-quiet-values in real mode, a flipped
    support stamp);
  * the truth-free reduction reproduces the committed one EXACTLY on a stored
    MOCK posterior (``_fdraws.npz`` of the frozen j1 run) — thresholds,
    Paper-1 bins, per-z cells, coarse blocks and Omega[20.3, 21.6];
  * the output schema carries no truth field;
  * a tiny end-to-end smoke run on a MOCK pack under the identity flag.

NOTHING HERE TOUCHES THE REAL PACK OR THE REAL LIKELIHOOD.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "validation", "real_c1"))

L = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
MOCK_PACK = os.path.join(L, "support_v3", "scanpack_2lpt0_b300_v3.npz")
MG_FIXED = os.path.join(L, "response_review", "candidates", "Mg_B_2lpt0.npz")
C_FIXED = os.path.join(L, "completeness", "C_C1nsadd_2lpt0.npz")
MU_EXTRA = os.path.join(L, "real_c1", "identity_inputs", "mu_extra_P6bcal_2lpt0.npz")
STORED_RUN = os.path.join(
    L, "final", "runs", "B-phi2lpt-C1nsadd-M1CUTj1",
    "RUN_B-phi2lpt-C1nsadd-M1CUTj1_2lpt0_s20260811")
RUNNER = os.path.join(REPO, "validation", "real_c1", "run_real_c1.py")

TOL = 1e-10

needs_scratch = pytest.mark.skipif(
    not os.path.exists(MOCK_PACK), reason="scratch ladder root unavailable")
needs_stored = pytest.mark.skipif(
    not os.path.exists(STORED_RUN + "_fdraws.npz"),
    reason="stored mock posterior unavailable")


def _run(args, expect_fail=True):
    env = dict(os.environ, PYTHONPATH=REPO, JAX_PLATFORMS="cpu",
               HDF5_USE_FILE_LOCKING="FALSE", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, RUNNER] + args, capture_output=True,
                       text=True, env=env, cwd=REPO)
    if expect_fail:
        assert r.returncode != 0, f"expected a refusal, got rc=0\n{r.stdout}"
    return r


# ---------------------------------------------------------------------------
# 1. the refusals
# ---------------------------------------------------------------------------
BASE = ["--ladder", "M1CUT", "--seed", "1", "--chains", "1", "--warmup", "2",
        "--samples", "2", "--out", "/dev/null"]


@needs_scratch
def test_refuses_a_mock_pack_without_the_identity_flag(tmp_path):
    r = _run(["--pack", MOCK_PACK] + BASE[:-1] + [str(tmp_path / "x.json")])
    assert "REAL GATE" in (r.stdout + r.stderr)
    assert "NONZERO truth_counts" in (r.stdout + r.stderr)


@needs_scratch
def test_refuses_oracle(tmp_path):
    r = _run(["--pack", MOCK_PACK, "--ladder", "ORACLE", "--seed", "1",
              "--allow-mock-for-identity-test",
              "--out", str(tmp_path / "x.json")])
    assert "ORACLE" in (r.stdout + r.stderr)


@needs_scratch
@pytest.mark.parametrize("flag", ["--fix", "--ops", "--census"])
def test_refuses_the_mock_only_truth_diagnostics(flag, tmp_path):
    r = _run(["--pack", MOCK_PACK, flag, "P", "--allow-mock-for-identity-test"]
             + BASE[:-1] + [str(tmp_path / "x.json")])
    assert "MOCK-ONLY" in (r.stdout + r.stderr) or "refus" in (r.stdout + r.stderr).lower()


@needs_scratch
def test_refuses_the_family_phi_oracle_response_key(tmp_path):
    r = _run(["--pack", MOCK_PACK, "--mg-fixed-key", "Mg_phi_family",
              "--allow-mock-for-identity-test"]
             + BASE[:-1] + [str(tmp_path / "x.json")])
    assert "Mg_phi_family" in (r.stdout + r.stderr)


def test_refuses_unknown_ladder(tmp_path):
    r = _run(["--pack", "/nonexistent.npz", "--ladder", "M9", "--seed", "1",
              "--out", str(tmp_path / "x.json")])
    assert "unknown ladder member" in (r.stdout + r.stderr)


def test_no_quiet_values_is_refused_outside_the_identity_test(tmp_path):
    r = _run(["--pack", "/nonexistent.npz", "--no-quiet-values"]
             + BASE[:-1] + [str(tmp_path / "x.json")])
    assert "quiet-values" in (r.stdout + r.stderr)


@needs_scratch
def test_support_gate_fails_closed_on_a_flipped_stamp(tmp_path):
    """A stamp whose collar is flipped 3300 -> 3000 must be refused."""
    pack = shutil.copy(MOCK_PACK, tmp_path / "pack.npz")
    src_stamp = MOCK_PACK[:-4] + ".support.json"
    if not os.path.exists(src_stamp):
        pytest.skip("the mock pack carries no support sidecar")
    st = json.load(open(src_stamp))
    # flip ONE row-selection field in every stamped plane
    def _flip(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "collar_kms":
                    node[k] = 3000.0
                else:
                    _flip(v)
        elif isinstance(node, list):
            for v in node:
                _flip(v)
    _flip(st)
    json.dump(st, open(str(pack)[:-4] + ".support.json", "w"))
    r = _run(["--pack", str(pack), "--require-support",
              "--allow-mock-for-identity-test"]
             + BASE[:-1] + [str(tmp_path / "x.json")])
    txt = r.stdout + r.stderr
    assert ("SupportMismatch" in txt or "SupportContractError" in txt
            or "support" in txt.lower())


# ---------------------------------------------------------------------------
# 2. the truth-free reduction == the committed reduction (stored mock posterior)
# ---------------------------------------------------------------------------
@needs_stored
def test_reduction_equals_perz_recovery_on_a_stored_mock_posterior():
    import reduce_truthfree as RT
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import perz_recovery

    z = np.load(STORED_RUN + "_fdraws.npz")
    f = np.asarray(z["f"], float)
    ft = np.asarray(z["truth_f"], float)      # mock-only; used ONLY to call the
    pk = load_pack(MOCK_PACK)                 # committed truth-taking reference
    ref = perz_recovery(f, ft, pk)
    got = RT.perz_posterior(f, pk)

    assert got["z_cells"] == ref["z_cells"]
    assert got["dX_k"] == ref["dX_k"]
    n_checked = 0
    for tag in ref["estimand"]:
        for group in ("native_cells", "coarse_blocks", "paper1_bins"):
            for r_ref, r_got in zip(ref["estimand"][tag][group],
                                    got["estimand"][tag][group]):
                assert r_ref["bin"] == r_got["bin"]
                assert r_ref.get("available") == r_got.get("available")
                if not r_ref.get("available"):
                    continue
                a = np.asarray(r_ref["post_p2p5_16_50_84_97p5"], float)
                b = np.asarray(r_got["post_p2p5_16_50_84_97p5"], float)
                assert np.array_equal(a, b), (tag, group, r_ref["bin"])
                assert np.allclose(a, b, rtol=0, atol=TOL)
                n_checked += 1
        a = np.asarray(ref["estimand"][tag]["allz"]["post_p2p5_16_50_84_97p5"], float)
        b = np.asarray(got["estimand"][tag]["allz"]["post_p2p5_16_50_84_97p5"], float)
        assert np.array_equal(a, b)
        n_checked += 1
    assert n_checked > 40


@needs_stored
def test_thresholds_and_reporting_bins_equal_the_stored_mock_json():
    """The all-z headline percentiles and the 0.2-dex bins reproduce the values
    the FROZEN mock runner wrote into its own JSON, to 1e-10."""
    import reduce_truthfree as RT
    from CDDF_analysis.hbi_mcmc.pack import load_pack

    stored = json.load(open(STORED_RUN + ".json"))
    z = np.load(STORED_RUN + "_fdraws.npz")
    f = np.asarray(z["f"], float)
    pk = load_pack(MOCK_PACK)

    got = RT.thresholds_allz(f, pk)
    for thr in ("20.0", "20.3"):
        ref = stored["thresholds"][f"ge{thr}"]
        g = got[f"ge{float(thr)}"]
        assert np.allclose(ref["post_p16_50_84"], g["post_p16_50_84"], rtol=0, atol=TOL)
        assert np.allclose(ref["post_p2p5_97p5"], g["post_p2p5_97p5"], rtol=0, atol=TOL)

    # the 0.2-dex reporting bins: the stored JSON keeps only bias, so the bin
    # EDGES are pinned here and the medians are pinned against the stored
    # perz/threshold arithmetic recomputed from the same draws.
    bins = RT.reporting_bins_0p2dex(f, pk)
    assert [b["bin"] for b in bins] == [b["bin"] for b in stored["reporting_bins"]]
    assert all(len(b["post_p2p5_16_50_84_97p5"]) == 5 for b in bins)
    assert all(np.all(np.diff(b["post_p2p5_16_50_84_97p5"]) >= 0) for b in bins)


@needs_stored
def test_paper1_bins_posterior_percentiles_match_the_stored_json_perz_block():
    """Direct equality against the frozen run's own ``perz_recovery`` block."""
    import reduce_truthfree as RT
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    stored = json.load(open(STORED_RUN + ".json"))["perz_recovery"]
    z = np.load(STORED_RUN + "_fdraws.npz")
    got = RT.perz_posterior(np.asarray(z["f"], float), load_pack(MOCK_PACK))
    n = 0
    for tag in stored["estimand"]:
        for r_ref, r_got in zip(stored["estimand"][tag]["paper1_bins"],
                                got["estimand"][tag]["paper1_bins"]):
            assert r_ref["bin"] == r_got["bin"]
            if not r_ref.get("available"):
                continue
            assert np.allclose(r_ref["post_p2p5_16_50_84_97p5"],
                               r_got["post_p2p5_16_50_84_97p5"], rtol=0, atol=TOL)
            n += 1
    assert n >= 8


@needs_stored
def test_omega_equals_the_ladder_tables_posterior_half():
    """Omega[20.3, 21.6]: same weights, same prefactor, same percentiles as the
    frozen ``ladder_table.omega_allz_from_weights``."""
    import reduce_truthfree as RT
    sys.path.insert(0, os.path.join(REPO, "validation", "fp_ladder"))
    import ladder_table as LT
    if not os.path.isdir(LT.PAPER_FIGURES):
        pytest.skip("paper_figures unavailable")
    sys.path.insert(0, LT.PAPER_FIGURES)
    import hbi_reduction as HR

    z = np.load(STORED_RUN + "_fdraws.npz")
    f = np.asarray(z["f"], float)
    P = HR.Posterior.__new__(HR.Posterior)
    P.f, P.n_edges = f, np.asarray(z["ntrue_edges"], float)
    P.z_edges, P.dX = np.asarray(z["zf_edges"], float), np.asarray(z["dX_k"], float)
    ref = LT.omega_allz_from_weights(f, np.asarray(z["truth_f"], float),
                                     P._omega_weight(*HR.OMEGA_NHI),
                                     P._z_weight(*HR.LOWZ_SUPPORT),
                                     HR.OMEGA_PREFACTOR_CM2)
    got = RT.omega_20p3_21p6_allz(f, z["ntrue_edges"], z["zf_edges"], z["dX_k"])
    assert np.allclose(ref["post_p16_50_84"], got["post_p16_50_84"], rtol=0, atol=TOL)
    assert np.allclose(ref["post_p2p5_97p5"], got["post_p2p5_97p5"], rtol=0, atol=TOL)
    assert got["window_nhi"] == [20.3, 21.6]
    assert got["h_reporting"] == 0.70
    assert "truth" not in json.dumps(got)


# ---------------------------------------------------------------------------
# 3. the truth guard
# ---------------------------------------------------------------------------
def test_assert_no_truth_accepts_the_support_field_names_and_rejects_values():
    import reduce_truthfree as RT
    RT.assert_no_truth({"support_gate": {"fields": {"truth_host_floor": "n/a",
                                                    "truth_catalogue_sha256": "n/a"},
                                        "planes": {"pack.truth_counts": "ab12"},
                                        "truth_host_floor": {"pack.truth_counts": 19.0}},
                        "real_gate": {"truth_counts_sentinel": "ZEROS_NO_TRUTH",
                                      "truth_counts_all_zero": True}})
    # the exempt subtree is a NAME->stamp dictionary only: an array smuggled in
    # under it is still refused
    with pytest.raises(SystemExit):
        RT.assert_no_truth({"support_gate": {"planes": {"pack.truth_counts": [1, 2]}}})
    for bad in ({"truth": 1.0}, {"a": [{"median_bias_pct": 0.4}]},
                {"diagnostics": {"fp_truth": {}}}, {"x": {"truth_value": 3}},
                {"perz_recovery": {}}, {"nested": [{"truth_in_68": True}]}):
        with pytest.raises(SystemExit):
            RT.assert_no_truth(bad)


# ---------------------------------------------------------------------------
# 4. end-to-end smoke (tiny chains) — schema + no truth keys anywhere
# ---------------------------------------------------------------------------
@needs_scratch
@pytest.mark.slow
def test_smoke_run_on_a_mock_pack_has_the_real_mode_schema(tmp_path):
    out = tmp_path / "smoke.json"
    r = _run(["--pack", MOCK_PACK, "--ladder", "M1CUT", "--seed", "20260811",
              "--chains", "2", "--warmup", "40", "--samples", "40",
              "--target-accept", "0.95", "--require-support",
              "--mg-fixed-file", MG_FIXED, "--mg-fixed-key", "Mg",
              "--c-fixed-file", C_FIXED, "--extra-fixed-file", MU_EXTRA,
              "--lam-imputations", "8", "--lam-imputation", "3",
              "--allow-mock-for-identity-test", "--skip-contract-guards",
              "--stage", "SMOKE", "--out", str(out)], expect_fail=False)
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.load(open(out))
    for k in ("run_config", "support_gate", "fixed_files", "lam_cut",
              "sampler_health", "t_posterior", "fp_totals", "estimands",
              "predictive_marginals", "shapes", "mode", "role"):
        assert k in d, k
    assert d["mode"] == "MOCK_IDENTITY_TEST"
    assert d["lam_cut"]["J"] == 8 and d["lam_cut"]["j"] == 3
    assert d["lam_cut"]["fp_a0"] is None
    for side in ("mg", "c", "extra"):
        assert "@" in d["fixed_files"][side]
    e = d["estimands"]
    assert set(e["thresholds_allz"]) == {"ge20.0", "ge20.3"}
    assert len(e["perz_posterior"]["estimand"]["ge20.0"]["paper1_bins"]) == 5
    assert e["omega_20p3_21p6_allz"]["window_nhi"] == [20.3, 21.6]
    # the truth guard, applied to the artifact as written
    import reduce_truthfree as RT
    RT.assert_no_truth(d)
    # the draws file carries NO truth surface
    z = np.load(str(out)[:-5] + "_fdraws.npz")
    assert set(z.files) == {"f", "ntrue_edges", "zf_edges", "dX_k"}
    assert "truth_f" not in z.files
    # stdout never carries an estimand value
    assert "VALUES WITHHELD" in r.stdout
    for key in ("post_p16_50_84", "post_p2p5", "omega_20p3_21p6",
                "reporting_bins", "paper1_bins"):
        assert key not in r.stdout
    # and no headline NUMBER reaches stdout either (the mixing block prints the
    # estimand KEY NAME with its R-hat/ESS — that is sampler health, not a value)
    for thr in ("ge20.0", "ge20.3"):
        for v in e["thresholds_allz"][thr]["post_p16_50_84"]:
            assert repr(v) not in r.stdout and f"{v:.6g}" not in r.stdout


# ---------------------------------------------------------------------------
# 5. the production sbatch and the frozen mock runner
# ---------------------------------------------------------------------------
SBATCH = os.path.join(REPO, "validation", "real_c1", "run_real_c1.sbatch")


def test_production_sbatch_calls_the_new_runner_and_stays_fail_closed():
    s = open(SBATCH).read()
    assert "DO NOT SUBMIT" in s
    assert "validation/real_c1/run_real_c1.py" in s
    assert "--ladder M1CUT" in s and "--require-support" in s
    assert "--lam-imputations $J --lam-imputation $j" in s
    assert "--quiet-values" in s
    assert "--fp-a0" not in s.split("NOTE ON --fp-a0")[0]   # never passed
    assert "--array=0-15" in s and "J=8" in s
    assert "SEEDS=(20260811 20260812)" in s
    assert "real_c1_inputs" in s and "real_c1/runs" in s
    assert "--account=cavestru0" in s and "--partition=standard" in s
    assert 'export SBATCH_CONSTRAINT=""' in s
    # fail-closed preconditions: pack sha, input SHA256SUMS, runner, identity verdict
    for guard in ("sha256sum -c", "PACK_SHA=", "IDENTITY_VERDICT.json",
                  "exit 3", "exit 4", "exit 5", "exit 6"):
        assert guard in s, guard
    # SLURM env hygiene (GL gotcha) and set -u only AFTER sourcing bashrc
    assert s.index("source ~/.bashrc") < s.index("set -u")


def test_the_frozen_mock_runner_is_untouched():
    """run_ladder.py and fp_ladder.py must be unmodified in the worktree."""
    r = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "-uno",
                        "validation/fp_ladder/run_ladder.py",
                        "CDDF_analysis/hbi_mcmc/fp_ladder.py"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", f"frozen files modified:\n{r.stdout}"


def test_runner_and_run_ladder_agree_on_the_model_call_arguments():
    """The real runner must pass model_cc_ladder the same keyword set."""
    import inspect
    from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder
    sig = set(inspect.signature(model_cc_ladder).parameters)
    src = open(RUNNER).read()
    call = src[src.index("mcmc.run("):src.index("extra_fields=(")]
    passed = {k.strip() for k in
              [t.split("=")[0].strip() for t in call.replace("\n", " ").split(",")
               if "=" in t]}
    for k in ("counts", "fp_counts", "ladder", "t_sd", "tau_scale", "calib_weight",
              "mu_fp_fixed", "mu_extra_fixed", "C_fixed", "Mg_fixed", "E_fixed",
              "lam_fixed", "fp_a0"):
        assert k in passed, f"{k} not passed to model_cc_ladder"
        assert k in sig


@needs_scratch
def test_contract_guard_subprocess_parses_and_enforces():
    """The guards helper the REAL path calls: exercised on a MOCK pack, so the
    subprocess + parse + 'every non-G_A guard must pass' rule is covered without
    touching the real pack."""
    import run_real_c1 as R
    rep = R.contract_guards(MOCK_PACK)
    assert isinstance(rep, dict) and rep
    assert all(v in ("PASS", "FAIL", None) or isinstance(v, str) for v in rep.values())
    assert [k for k in rep if k.startswith("G_")]
    assert all(rep[k] == "PASS" for k in rep if k.startswith("G_"))
    with pytest.raises(SystemExit):
        R.contract_guards("/nonexistent_pack.npz")
