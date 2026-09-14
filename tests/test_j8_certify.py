import numpy as np
from validation.fp_ladder.j8_certify import rank_rhat_ess


def test_rank_rhat_near_one_for_iid_chains():
    rng = np.random.default_rng(0); x = rng.normal(size=(4, 1000))
    rhat, bulk, tail = rank_rhat_ess(x)
    assert abs(rhat - 1) < 0.02 and bulk > 2500 and tail > 800


def test_rank_rhat_detects_separated_chains():
    rng = np.random.default_rng(1); x = rng.normal(size=(4, 1000)); x[0] += 5.0
    rhat, bulk, tail = rank_rhat_ess(x)
    assert rhat > 1.5 and bulk < 200


def test_ess_drops_for_autocorrelated_chains():
    rng = np.random.default_rng(2); n = 2000; x = np.zeros((4, n))
    for c in range(4):
        e = rng.normal(size=n)
        for i in range(1, n): x[c, i] = 0.95 * x[c, i - 1] + e[i]
    rhat, bulk, tail = rank_rhat_ess(x)
    assert bulk < 0.2 * 4 * n


def test_pooled_headline_reduction_matches_runner_on_a_stored_run():
    import os, glob, json
    F = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/final/runs/B-phi2lpt-C1nsadd-M1CUTJ8"
    fs = sorted(glob.glob(os.path.join(F, "RUN_*_2lpt0_*J8j3.json")))
    if not fs:
        import pytest; pytest.skip("production runs not present")
    from validation.fp_ladder.j8_certify import pooled_headlines
    j = json.load(open(fs[0])); r = pooled_headlines([fs[0].replace(".json", "_fdraws.npz")], {t: j["thresholds"][t]["truth"] for t in ("ge20.0", "ge20.3")})
    for t in ("ge20.0", "ge20.3"):
        assert abs(j["thresholds"][t]["post_p16_50_84"][1] - r[t]["median"]) < 1e-12
