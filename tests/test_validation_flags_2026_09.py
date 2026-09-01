"""Validation-only flags for the 2026-09-02 HBI identifiability campaign
(science-lane validation worktree, branch validation/hbi-identifiability-2026-09;
NEVER merged to a production branch).

  model_cc(fix_t=True)      -> `t`     becomes a deterministic all-zero site (R1/R4)
  model_cc(fix_psi_c=True)  -> `psi_c` becomes a deterministic all-zero site (R3/R4)
  cc_real_posterior --fix-t / --fix-psi-c / --init-from / --save-all-sites

Default-off contract: with the flags off the trace (site names, types, shapes) and the
`counts` rate are IDENTICAL to the frozen model_cc at prov/paper1-freeze-2026-08-26;
R0 proves the same thing on the real chains (bit-identical draws)."""
import inspect
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from CDDF_analysis.hbi_mcmc.cc_posterior_validation import model_cc   # noqa: E402


def _consts(seed=0, B=6, Kf=4, C=5, S=3, M=2, KK=2):
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    c = types.SimpleNamespace(
        n_b=B, n_k=Kf, n_c=C, n_s=S, n_molly=M,
        sigma_hat=jnp.asarray(rng.uniform(0.2, 0.5, (S, M))),
        eta_hat=jnp.asarray(rng.normal(0.0, 1.0, (S, M))),
        t_sigma=jnp.asarray(rng.uniform(0.1, 0.2, KK)),
        b_to_cell=np.asarray(rng.integers(0, M, B)),
        g_bk=jnp.asarray(rng.uniform(0.5, 1.5, (B, Kf))),
        dN_b=jnp.asarray(np.full(B, 0.2)),
        dX=jnp.asarray(rng.uniform(100.0, 300.0, (Kf, S))),
        fp_w=3.0, fp_ell_eff=7.0,
        fp_eta_c=jnp.asarray(rng.uniform(0.0, 0.3, C)),
        kz_to_K=np.asarray(rng.integers(0, KK, Kf)),
        fp_E=jnp.asarray(rng.uniform(0.5, 1.5, (Kf, S))),
    )
    Mg = jnp.asarray(rng.uniform(0.0, 1.0, (S, Kf, C, B)) * 1e-3)
    fpc = rng.poisson(4.0, (C, S)).astype(float)
    return c, Mg, fpc


def _params(c, rng):
    B, Kf, C, S, KK = c.n_b, c.n_k, c.n_c, c.n_s, int(np.asarray(c.t_sigma).size)
    M = int(np.asarray(c.sigma_hat).shape[1])
    # theta_level=3.0 makes the TP term comparable to the FP term on the synthetic
    # tensors, so a psi_c change is visible in the rate (the -49.5 of the PPC test
    # pins the FP fold alone and would make the psi_c non-vacuity check vacuous)
    return dict(sigma_N=0.3, sigma_z=0.2, theta_level=3.0, theta_slope=-0.3,
                eps_N=rng.normal(size=max(B - 2, 0)),
                eps_z=rng.normal(size=(B, max(Kf - 1, 0))),
                psi_c=rng.normal(size=(S, M)) * 0.3,
                fp_lam_total=2.0, fp_shape_v=rng.normal(size=C * S),
                t=rng.normal(size=KK) * 0.1)


def _trace(c, Mg, fpc, params, **kw):
    import jax.numpy as jnp
    import numpyro
    m = numpyro.handlers.seed(numpyro.handlers.substitute(model_cc, data=params), rng_seed=0)
    return numpyro.handlers.trace(m).get_trace(
        c, Mg, counts=jnp.zeros((c.n_c, c.n_k, c.n_s)), fp_counts=jnp.asarray(fpc),
        fp_mode="informative_ln", **kw)


def _rate(tr):
    fn = tr["counts"]["fn"]
    return np.asarray(fn.rate if hasattr(fn, "rate") else fn.base_dist.rate)


def _sites(tr):
    return {k: (v["type"], tuple(np.shape(v["value"])), bool(v.get("is_observed", False)))
            for k, v in tr.items()}


PRODUCTION_SAMPLED = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z",
                      "psi_c", "fp_lam_total", "fp_shape_v", "t")


def test_flags_exist_and_default_off():
    sig = inspect.signature(model_cc)
    assert sig.parameters["fix_t"].default is False
    assert sig.parameters["fix_psi_c"].default is False


def test_default_off_trace_identical_to_production_call():
    """Calling with the flags explicitly False == calling without them (the R0 code path)."""
    c, Mg, fpc = _consts()
    p = _params(c, np.random.default_rng(1))
    tr0 = _trace(c, Mg, fpc, p)
    tr1 = _trace(c, Mg, fpc, p, fix_t=False, fix_psi_c=False)
    assert _sites(tr0) == _sites(tr1)
    assert np.array_equal(_rate(tr0), _rate(tr1))
    sampled = [k for k, v in tr0.items() if v["type"] == "sample" and not v.get("is_observed")]
    assert tuple(sampled) == PRODUCTION_SAMPLED
    n = sum(int(np.prod(np.shape(tr0[k]["value"]))) for k in sampled)
    B, Kf, C, S, M, KK = c.n_b, c.n_k, c.n_c, c.n_s, c.n_molly, int(np.asarray(c.t_sigma).size)
    assert n == 4 + (B - 2) + B * (Kf - 1) + S * M + 1 + C * S + KK
    assert "psi_k_delta" not in tr0 and "fp_counts" not in tr0


def test_fix_t_makes_t_deterministic_zero_and_changes_nothing_else():
    c, Mg, fpc = _consts()
    p = _params(c, np.random.default_rng(2))
    # numpyro.substitute overrides deterministic sites too: do not substitute `t`
    tr = _trace(c, Mg, fpc, {k: v for k, v in p.items() if k != "t"}, fix_t=True)
    assert tr["t"]["type"] == "deterministic"
    assert np.array_equal(np.asarray(tr["t"]["value"]), np.zeros(int(np.asarray(c.t_sigma).size)))
    sampled = [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]
    assert tuple(sampled) == tuple(k for k in PRODUCTION_SAMPLED if k != "t")
    # the rate equals the production model evaluated at t == 0 (nothing else moved)
    p0 = dict(p, t=np.zeros_like(np.asarray(p["t"])))
    assert np.allclose(_rate(tr), _rate(_trace(c, Mg, fpc, p0)), rtol=1e-12, atol=0.0)
    # and differs from the production rate at the drawn t (the flag is not vacuous)
    assert not np.allclose(_rate(tr), _rate(_trace(c, Mg, fpc, p)), rtol=1e-6, atol=0.0)
    # lam_fp untouched by the flag (the `anchored` mode would have replaced it)
    assert np.allclose(np.asarray(tr["lam_fp"]["value"]),
                       np.asarray(_trace(c, Mg, fpc, p)["lam_fp"]["value"]), rtol=1e-12, atol=0.0)


def test_fix_psi_c_makes_psi_c_deterministic_zero_and_changes_nothing_else():
    c, Mg, fpc = _consts()
    p = _params(c, np.random.default_rng(3))
    tr = _trace(c, Mg, fpc, {k: v for k, v in p.items() if k != "psi_c"}, fix_psi_c=True)
    assert tr["psi_c"]["type"] == "deterministic"
    assert np.array_equal(np.asarray(tr["psi_c"]["value"]), np.zeros(np.asarray(c.sigma_hat).shape))
    sampled = [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]
    assert tuple(sampled) == tuple(k for k in PRODUCTION_SAMPLED if k != "psi_c")
    p0 = dict(p, psi_c=np.zeros_like(np.asarray(p["psi_c"])))
    assert np.allclose(_rate(tr), _rate(_trace(c, Mg, fpc, p0)), rtol=1e-12, atol=0.0)
    assert not np.allclose(_rate(tr), _rate(_trace(c, Mg, fpc, p)), rtol=1e-6, atol=0.0)


def test_both_flags():
    c, Mg, fpc = _consts()
    p = _params(c, np.random.default_rng(4))
    tr = _trace(c, Mg, fpc, {k: v for k, v in p.items() if k not in ("t", "psi_c")},
                fix_t=True, fix_psi_c=True)
    sampled = [k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")]
    assert tuple(sampled) == tuple(k for k in PRODUCTION_SAMPLED if k not in ("t", "psi_c"))
    p0 = dict(p, t=np.zeros_like(np.asarray(p["t"])), psi_c=np.zeros_like(np.asarray(p["psi_c"])))
    assert np.allclose(_rate(tr), _rate(_trace(c, Mg, fpc, p0)), rtol=1e-12, atol=0.0)


def test_fix_t_refused_outside_informative_ln():
    """Fail closed: the flag is defined for the production branch only."""
    import jax.numpy as jnp
    import numpyro
    c, Mg, fpc = _consts()
    p = _params(c, np.random.default_rng(5))
    m = numpyro.handlers.seed(numpyro.handlers.substitute(model_cc, data=p), rng_seed=0)
    with pytest.raises(ValueError):
        numpyro.handlers.trace(m).get_trace(
            c, Mg, counts=jnp.zeros((c.n_c, c.n_k, c.n_s)), fp_counts=jnp.asarray(fpc),
            fp_mode="joint", fix_t=True)


def test_runner_flags_default_off_and_init_values_roundtrip(tmp_path):
    from CDDF_analysis.hbi_mcmc import cc_real_posterior as R
    ap = R.build_parser()
    a = ap.parse_args(["--pack", "x.npz", "--out", "y.json"])
    assert a.fix_t is False and a.fix_psi_c is False
    assert a.init_from is None and a.save_all_sites is None
    a2 = ap.parse_args(["--pack", "x.npz", "--out", "y.json", "--fix-t", "--fix-psi-c"])
    assert a2.fix_t is True and a2.fix_psi_c is True
    vals = {"t": [0.0, 0.0, 0.0], "fp_lam_total": 7.19, "psi_c": [[0.0, 0.1], [0.2, 0.3]]}
    pj = tmp_path / "init.json"
    pj.write_text(json.dumps(vals))
    got = R.load_init_values(str(pj))
    assert set(got) == set(vals)
    for k, v in vals.items():
        assert np.allclose(np.asarray(got[k]), np.asarray(v), rtol=0, atol=0)
    pn = tmp_path / "init.npz"
    np.savez(pn, **{k: np.asarray(v) for k, v in vals.items()})
    got2 = R.load_init_values(str(pn))
    for k, v in vals.items():
        assert np.allclose(np.asarray(got2[k]), np.asarray(v), rtol=0, atol=0)


def test_runner_help_lists_the_flags():
    r = subprocess.run([sys.executable, "-m", "CDDF_analysis.hbi_mcmc.cc_real_posterior", "--help"],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0
    for f in ("--fix-t", "--fix-psi-c", "--init-from", "--save-all-sites"):
        assert f in r.stdout
