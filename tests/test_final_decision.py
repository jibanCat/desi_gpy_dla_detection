import numpy as np
from validation.fp_ladder.final_decision import decide, envelope, THR, FAMS


def S(bias0, bias3, gate, zz=20.0, k1=3.0, hw=0.6, div=0, sd=0.02):
    return {f: dict(n_seeds=2, bias={"ge20.0": bias0[i], "ge20.3": bias3[i]}, bias_sd={"ge20.0": sd, "ge20.3": sd},
                    hw68={"ge20.0": hw, "ge20.3": hw}, zigzag=zz, k1_ge20p3=k1, div_max=div, ebfmi_min=0.9,
                    gate_pass=[gate[i], gate[i]]) for i, f in enumerate(FAMS)}


def test_outcome_A_clear_winner_carried_alone():
    SE = S([0.1, 0.2, 0.3], [1.0, 1.0, 1.0], ["PASS", "PASS", "PASS"])
    SB = S([0.1, 0.9, 0.3], [1.0, 1.0, 1.0], ["PASS", "FAIL", "PASS"])
    o, carried, base, alt, trail = decide(SE, SB)
    assert o == "A" and carried == ["E"] and base == "E" and alt == "B"


def test_outcome_A_blocked_by_worse_structured_residual_falls_through():
    SE = S([0.1, 0.2, 0.3], [1.0, 1.0, 1.0], ["PASS", "PASS", "PASS"], zz=30.0)   # passes more but much worse zigzag
    SB = S([0.1, 0.9, 0.3], [1.0, 1.0, 1.0], ["PASS", "FAIL", "PASS"], zz=20.0)
    o, carried, base, alt, trail = decide(SE, SB)
    assert o == "C" and any("not Outcome A" in t for t in trail)


def test_outcome_B_both_pass_and_agree():
    SE = S([0.1, 0.2, 0.3], [1.0, 1.1, 1.2], ["PASS"] * 3, zz=22.0, k1=3.0)
    SB = S([0.2, 0.3, 0.2], [1.2, 1.0, 1.3], ["PASS"] * 3, zz=21.0, k1=2.9)
    o, carried, base, alt, trail = decide(SE, SB)
    assert o == "B" and sorted(carried) == ["B", "E"] and {base, alt} == {"B", "E"}


def test_outcome_C_when_headlines_disagree():
    SE = S([0.1, 0.2, 0.3], [1.0, 1.0, 1.0], ["FAIL"] * 3)
    SB = S([0.9, 1.2, 1.5], [3.0, 3.0, 3.0], ["FAIL"] * 3)
    o, carried, base, alt, trail = decide(SE, SB)
    assert o == "C" and carried == ["E"] and base == "E"


def test_envelope_signed_and_units():
    SE = S([0.1, 0.2, 0.3], [1.0, 1.0, 1.0], ["PASS"] * 3, hw=0.5)
    SB = S([0.4, 0.2, 0.0], [1.0, 1.5, 1.0], ["PASS"] * 3, hw=0.5)
    SR = S([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], ["PASS"] * 3, hw=1.0)
    env = envelope(SE, SB, SR)
    d = {(r["family"], r["threshold"]): r for r in env["rows"]}
    assert np.isclose(d[("2lpt0", "ge20.0")]["signed_diff_pp"], 0.3) and np.isclose(d[("saclay0", "ge20.0")]["signed_diff_pp"], -0.3)
    assert np.isclose(env["max_abs_shift_pp"], 0.5) and np.isclose(env["max_abs_shift_old_hw68"], 0.5) and np.isclose(env["max_abs_shift_new_hw68"], 1.0)
