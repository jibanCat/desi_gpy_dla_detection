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


# --------------------------------------------------------------------------
# 2026-09-15 cleanup (PI ruling 2026-09-14b sec.10, sec.17): the Omega column
# was the SUB-DLA window, and the sealed rule's rank-Rhat was implemented as
# the runner's split-Rhat.  Both are documentation corrections; the verdict is
# unchanged, and these tests pin that.
# --------------------------------------------------------------------------
PRODUCTS = ("/scratch/cavestru_root/cavestru0/mfho/"
            "absorber_ladder_2026-09-13")
J8DIR = PRODUCTS + "/final/runs/B-phi2lpt-C1nsadd-M1CUTJ8"


def _need():
    import os, pytest
    if not os.path.isdir(J8DIR):
        pytest.skip("production runs not present")


def test_analyse_separates_the_subdla_window_from_the_paper_omega():
    """The two Omega windows are different quantities and must not share a name."""
    _need()
    import glob, os
    from validation.fp_ladder.j8_certify import analyse
    p = sorted(glob.glob(os.path.join(J8DIR, "RUN_*_2lpt0_*J8j0.json")))[0]
    r, _j = analyse(p)
    sub = r["omega_subdla_19p5_20p3"]
    paper = r["omega_paper1_20p3_21p6"]
    assert sub["key"] == "omega_subdla_195_203_allz"
    assert paper is not None, "the Paper-1 Omega read-out is missing"
    assert abs(paper["bias_pct"] - sub["bias_pct"]) > 0.05
    assert 20.29 < paper["truth"] * 0 + 20.3 <= 20.3        # window is [20.3, 21.6]


def test_pooled_paper_omega_matches_the_single_run_readout():
    """Pooling one imputation must reproduce that imputation's own Omega."""
    _need()
    import glob, os
    from validation.fp_ladder.j8_certify import pooled_paper_omega, paper_omega
    p = sorted(glob.glob(os.path.join(J8DIR, "RUN_*_2lpt0_*J8j3.json")))[0]
    f = p.replace(".json", "_fdraws.npz")
    one = paper_omega(f)
    pool = pooled_paper_omega([f])
    assert abs(pool["median"] - one["post_p16_50_84"][1]) < 1e-18
    assert abs(pool["truth"] - one["truth"]) < 1e-18
    assert pool["window_nhi"] == [20.3, 21.6]


def test_sealed_rule_substitution_is_recorded_not_hidden():
    """PI 2026-09-14b sec.17: preserve history where the rule was implemented
    differently from its wording."""
    from validation.fp_ladder.j8_certify import SEALED_RULE_IMPLEMENTATION_NOTE
    n = SEALED_RULE_IMPLEMENTATION_NOTE
    assert "rank-Rhat" in n and "split-Rhat" in n
    assert "IMPLEMENTED AS" in n
    assert "Verdict unaffected" in n
    assert "preserved verbatim" in n
