"""cc_pool_posterior — PI ruling 2026-08-21 #19: the final inference artifact
is the PREDECLARED pooled posterior over converged runs. The selection rule,
deep-rerun replacement and pooling are pure functions tested here on
synthetic records (no real data). Runs in gpdla-hbi? No — jax-free by
design; loaded file-directly."""
import importlib.util as _ilu
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "cc_pool_posterior_mod",
    os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "cc_pool_posterior.py"))
PP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(PP)

RULE = dict(rhat_max=1.10, div_max=10)


def _rec(seed, rhat0, rhat3, div=0, ga="PASS", deep=False):
    return dict(seed=seed, deep=deep, file=f"REAL_{'deep_' if deep else ''}s{seed}.json",
                diagnostics=dict(split_rhat={"dndx_dla_20p0_allz": rhat0,
                                             "dndx_dla_20p3_allz": rhat3},
                                 divergences=div),
                guards=dict(G_A_real_mode=dict(status=ga)))


def test_select_runs_keeps_converged_base_runs_and_flags_strays():
    recs = [_rec(21, 1.02, 1.01), _rec(22, 1.30, 1.02), _rec(23, 1.01, 1.01, div=12)]
    sel = PP.select_runs(recs, RULE)
    assert [r["seed"] for r in sel["included"]] == [21]
    assert {r["seed"]: r["reason"] for r in sel["excluded"]} == {
        22: "split_rhat dndx_dla_20p0_allz 1.3 > 1.1",
        23: "divergences 12 > 10"}
    assert sel["needs_deep_rerun"] == [22, 23]


def test_select_runs_deep_rerun_replaces_its_base_never_both():
    recs = [_rec(21, 1.02, 1.01), _rec(22, 1.30, 1.02),
            _rec(22, 1.03, 1.02, deep=True)]
    sel = PP.select_runs(recs, RULE)
    inc = [(r["seed"], r["deep"]) for r in sel["included"]]
    assert inc == [(21, False), (22, True)]
    assert sel["needs_deep_rerun"] == []
    assert any(r["seed"] == 22 and not r["deep"] for r in sel["excluded"])


def test_select_runs_excludes_a_seed_that_still_fails_after_deep_rerun_with_disclosure():
    recs = [_rec(21, 1.02, 1.01), _rec(26, 11.7, 5.5), _rec(26, 12.7, 5.6, deep=True)]
    sel = PP.select_runs(recs, RULE)
    assert [r["seed"] for r in sel["included"]] == [21]
    ex = [r for r in sel["excluded"] if r["seed"] == 26 and r["deep"]]
    assert ex and ex[0]["disclosed"] is True
    assert sel["needs_deep_rerun"] == []          # already rerun once; not again


def test_select_runs_real_mode_guard_fail_excludes():
    recs = [_rec(21, 1.02, 1.01, ga="FAIL")]
    sel = PP.select_runs(recs, RULE)
    assert sel["included"] == [] and "G_A_real_mode" in sel["excluded"][0]["reason"]


def test_pool_draws_is_equal_weight_concatenation_in_seed_order():
    a = np.full((3, 2, 2), 1.0); b = np.full((5, 2, 2), 2.0)
    pooled, index = PP.pool_draws([(22, b), (21, a)])
    assert pooled.shape == (8, 2, 2)
    assert np.all(pooled[:3] == 1.0) and np.all(pooled[3:] == 2.0)
    assert index == [dict(seed=21, n_draws=3, start=0), dict(seed=22, n_draws=5, start=3)]


def test_quantile_block_matches_the_real_runner_convention():
    dr = np.arange(1000.0)
    q = PP.q5(dr)
    assert q == [float(x) for x in np.percentile(dr, [2.5, 16, 50, 84, 97.5])]
