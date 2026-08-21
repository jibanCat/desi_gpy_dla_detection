"""cp3_envelope_check — measures a new pooled posterior against the
PREDECLARED envelope of expected calibration-induced change (notes
2026-08-21_CP3_PREDECLARATION.md §5). Reports in/out per line; decides
nothing. Synthetic numbers only."""
import importlib.util as _ilu
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = _ilu.spec_from_file_location(
    "cp3_env_mod", os.path.join(_REPO, "CDDF_analysis", "hbi_mcmc", "cp3_envelope_check.py"))
EC = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(EC)


def _pooled(allz0, allz3, bins0, bins3):
    q = lambda m: [m * 0.97, m * 0.985, m, m * 1.015, m * 1.03]
    return dict(thresholds={"dndx_dla_20p0_allz": dict(post_p2p5_16_50_84_97p5=q(allz0)),
                            "dndx_dla_20p3_allz": dict(post_p2p5_16_50_84_97p5=q(allz3))},
                perz_paper1={"ge20.0": dict(paper1_bins=[dict(bin=b, available=True, post_p2p5_16_50_84_97p5=q(v)) for b, v in bins0.items()]),
                             "ge20.3": dict(paper1_bins=[dict(bin=b, available=True, post_p2p5_16_50_84_97p5=q(v)) for b, v in bins3.items()])})


ENV = dict(allz_pct=1.0,
           bins_pts={"ge20.0": {"B1": (-6, -3), "B2": (-2, 2), "B3": (-2, 2), "B4": (11, 16), "B5": (19, 28)},
                     "ge20.3": {"B1": (-7, -3), "B2": (-2, 2), "B3": (1, 4), "B4": (10, 17), "B5": (13, 33)}})


def test_all_inside_when_shifts_match_the_expected_pattern():
    old = _pooled(0.10, 0.07, dict(B1=0.10, B2=0.10, B3=0.10, B4=0.10, B5=0.10), dict(B1=0.07, B2=0.07, B3=0.07, B4=0.07, B5=0.07))
    new = _pooled(0.1005, 0.0703, dict(B1=0.095, B2=0.101, B3=0.10, B4=0.113, B5=0.122), dict(B1=0.0665, B2=0.07, B3=0.0717, B4=0.079, B5=0.084))
    rep = EC.envelope_report(new, old, ENV)
    assert rep["all_inside"] is True
    assert rep["lines"]["allz"]["ge20.0"]["inside"] and abs(rep["lines"]["allz"]["ge20.0"]["shift_pct"] - 0.5) < 1e-9


def test_allz_outside_is_flagged_with_the_measured_shift():
    old = _pooled(0.10, 0.07, dict(B1=0.1), dict(B1=0.07))
    new = _pooled(0.102, 0.07, dict(B1=0.1), dict(B1=0.07))
    rep = EC.envelope_report(new, old, dict(allz_pct=1.0, bins_pts={"ge20.0": {}, "ge20.3": {}}))
    assert rep["all_inside"] is False
    assert rep["lines"]["allz"]["ge20.0"]["inside"] is False
    assert abs(rep["lines"]["allz"]["ge20.0"]["shift_pct"] - 2.0) < 1e-9


def test_bin_shift_is_in_percentage_points_of_the_superseded_value():
    old = _pooled(0.1, 0.07, dict(B4=0.10), dict(B4=0.07))
    new = _pooled(0.1, 0.07, dict(B4=0.12), dict(B4=0.07 * 1.3))
    rep = EC.envelope_report(new, old, dict(allz_pct=5, bins_pts={"ge20.0": {"B4": (11, 16)}, "ge20.3": {"B4": (10, 17)}}))
    b0 = rep["lines"]["bins"]["ge20.0"]["B4"]; b3 = rep["lines"]["bins"]["ge20.3"]["B4"]
    assert abs(b0["shift_pts"] - 20.0) < 1e-9 and b0["inside"] is False
    assert abs(b3["shift_pts"] - 30.0) < 1e-9 and b3["inside"] is False
