"""cc_posterior_validation must stamp EVERY predeclared sensitivity knob in
its diagnostics block. Found 2026-08-21: fp_total_scale and t_scale were
stamped, fp_s_empty was not (the Battery-4 arm identity had to be recovered
from the sbatch/log lines). Runs in gpdla-hbi (module imports jax)."""
from types import SimpleNamespace

from CDDF_analysis.hbi_mcmc.cc_posterior_validation import sensitivity_stamp


def test_sensitivity_stamp_carries_fp_s_empty():
    a = SimpleNamespace(fp_mode="informative_ln", target_accept=0.95,
                        fp_alpha0=None, fp_total_scale=0.25, t_scale=1.0,
                        fp_s_empty=1.5)
    d = sensitivity_stamp(a)
    assert d["fp_s_empty"] == 1.5
    assert d["fp_total_scale"] == 0.25 and d["t_scale"] == 1.0
    assert d["fp_mode"] == "informative_ln" and d["target_accept"] == 0.95


def test_sensitivity_stamp_records_the_default_explicitly():
    a = SimpleNamespace(fp_mode="informative_ln", target_accept=0.95,
                        fp_alpha0=None, fp_total_scale=1.0, t_scale=1.0,
                        fp_s_empty=None)
    d = sensitivity_stamp(a)
    assert "fp_s_empty" in d and d["fp_s_empty"] is None
    assert d["fp_s_empty_effective"] == 2.0          # the predeclared default
