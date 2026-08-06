"""test_modelA_forward.py — Q3 Model A: pack schema validation + forward-fold oracle.

Fast, sampler-free tests:
  * pack round-trip (real grid + small test grid), default-grid enforcement,
    precise refusal on every schema-rule violation class.
  * skew-normal CDF (jnp, Gauss-Legendre Owen's T) vs scipy exact:
    zero skew and |skew| <= 1.4, tolerance 1e-4 on bin masses (measured ~1e-13;
    the brief's Azzalini fallback was NOT needed — the GL rule is exact to
    quadrature accuracy and fully differentiable).
  * fold_mu (pure jnp) vs fold_mu_reference (independent numpy/scipy oracle)
    at multiple random parameter points: rtol 1e-10 (+ tiny scaled atol for
    far-tail cells), small grid and one full-grid point; jit-compilability.

Run: conda run -n gpdla-hbi python -m pytest tests/test_modelA_forward.py -v
"""
import dataclasses
import glob
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from CDDF_analysis.hbi_mcmc import pack as packmod  # noqa: E402  (enables x64)
from CDDF_analysis.hbi_mcmc import forward as fwd  # noqa: E402
from CDDF_analysis.hbi_mcmc.pack import (  # noqa: E402
    ModelAPack, PackSchemaError, load_pack, save_pack, small_test_grid,
    synthetic_pack)

# 2026-08-06 (fp_eta_c restoration): legacy fixture packs predate the schema
# field; migrate them EXPLICITLY at the test boundary (idempotent; values
# identical to a fresh extraction — pack.FP_ETA_BANDS_COMMITTED).
from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _attach_fp_eta
load_pack = (lambda _f: (lambda *a, **k: _attach_fp_eta(_f(*a, **k))))(load_pack)

SEED = 0


@pytest.fixture(scope="module")
def small_pack():
    return synthetic_pack(SEED, **small_test_grid())


# --- pack: round trip + validation ------------------------------------------------

def test_small_pack_roundtrip(small_pack, tmp_path):
    p = tmp_path / "modelA_pack_test.npz"
    save_pack(small_pack, p, allow_nonstandard_grid=True)
    assert (tmp_path / "modelA_pack_test.provenance.json").exists()
    # small grid REFUSED by default (real grid enforced) ...
    with pytest.raises(PackSchemaError, match="REAL grid"):
        load_pack(p)
    # ... allowed via the explicit override
    pk = load_pack(p, allow_nonstandard_grid=True)
    assert np.array_equal(pk.counts, small_pack.counts)
    assert np.array_equal(pk.fp_counts, small_pack.fp_counts)
    assert pk.provenance is not None and pk.provenance.get("synthetic") in (True, "True")


def test_full_grid_pack_conforms(tmp_path):
    pk = synthetic_pack(1)  # default = the REAL grid
    assert pk.counts.shape == (29, 15, 8)
    assert pk.dX.shape == (15, 8)
    assert pk.fp_counts.shape == (29, 8)
    assert pk.fp_E_alloc.shape == (15, 8)
    assert pk.t_sigma.shape == (3,)
    p = tmp_path / "modelA_pack_full.npz"
    save_pack(pk, p)
    pk2 = load_pack(p)  # real grid loads WITHOUT the override
    assert np.array_equal(pk2.counts, pk.counts)


@pytest.mark.parametrize("mutation, match", [
    (dict(counts=lambda pk: pk.counts[:, :, :-1]), "counts.*shape"),
    (dict(counts=lambda pk: pk.counts.astype(np.float64)), "int64"),
    (dict(counts=lambda pk: np.abs(pk.counts) * -1 - 1), "non-negative"),
    (dict(dX=lambda pk: pk.dX * 0.0), "zero-pathlength"),  # dX==0 legal only with zero counts
    (dict(dX=lambda pk: pk.dX * -1.0), "non-negative"),
    (dict(dX=lambda pk: np.where(np.arange(pk.dX.size).reshape(pk.dX.shape) == 0,
                                 np.nan, pk.dX)), "finite"),
    (dict(ntrue_edges=lambda pk: pk.ntrue_edges + 0.05), "ntrue_edges"),
    (dict(zc_edges=lambda pk: pk.zc_edges + 0.05), "zc_edges"),
    (dict(kz_to_K=lambda pk: pk.kz_to_K[::-1].copy()), "kz_to_K"),
    (dict(fp_E_alloc=lambda pk: pk.fp_E_alloc * 2.0), "sum_k"),
    (dict(molly_n_det=lambda pk: pk.molly_n_det + pk.molly_n_tot), "n_det > n_tot"),
    (dict(t_sigma=lambda pk: pk.t_sigma * 0.0), "positive"),
    (dict(snr_edges=lambda pk: np.linspace(0, 7, len(pk.snr_edges))), "inf"),
    (dict(resp_sig_floor=lambda pk: -1.0), "resp_sig_floor"),
])
def test_validation_refuses_each_break(small_pack, mutation, match):
    (field, fn), = mutation.items()
    broken = dataclasses.replace(small_pack, **{field: fn(small_pack)})
    with pytest.raises(PackSchemaError, match=match):
        packmod.validate_pack(broken, allow_nonstandard_grid=True)


def test_loader_refuses_missing_key(small_pack, tmp_path):
    p = tmp_path / "modelA_pack_missing.npz"
    save_pack(small_pack, p, allow_nonstandard_grid=True)
    with np.load(p) as z:
        data = {k: z[k] for k in z.files if k != "t_sigma"}
    p2 = tmp_path / "modelA_pack_missing2.npz"
    np.savez(p2, **data)
    with pytest.raises(PackSchemaError, match="missing required schema keys.*t_sigma"):
        load_pack(p2, allow_nonstandard_grid=True)


def test_loader_refuses_unknown_key(small_pack, tmp_path):
    p = tmp_path / "modelA_pack_extra.npz"
    save_pack(small_pack, p, allow_nonstandard_grid=True)
    with np.load(p) as z:
        data = {k: z[k] for k in z.files}
    data["kappa_matrix"] = np.eye(3)  # e.g. a smuggled kappa object
    p2 = tmp_path / "modelA_pack_extra2.npz"
    np.savez(p2, **data)
    with pytest.raises(PackSchemaError, match="unknown keys.*kappa_matrix"):
        load_pack(p2, allow_nonstandard_grid=True)


# --- skew-normal CDF accuracy -------------------------------------------------------

def test_skewnorm_cdf_vs_scipy_bin_masses():
    """GL-quadrature Owen's T path vs scipy exact: bin masses to <= 1e-4 (brief),
    measured ~1e-13. Covers zero skew and |skew| <= 1.4."""
    from scipy.stats import skewnorm
    edges = np.arange(19.5, 22.4 + 1e-9, 0.1)
    for alpha in (0.0, 0.7, -0.7, 1.4, -1.4):
        for xi, om in ((20.0, 0.15), (21.3, 0.35), (19.6, 0.08)):
            F_j = np.asarray(fwd.skewnorm_cdf_jnp(jnp.asarray(edges), xi, om, alpha))
            m_j = np.diff(F_j)
            m_s = np.diff(skewnorm.cdf(edges, alpha, loc=xi, scale=om))
            assert np.max(np.abs(m_j - m_s)) < 1e-4, (alpha, xi, om)
            assert np.max(np.abs(m_j - m_s)) < 1e-10  # actual accuracy


def test_owens_t_symmetries():
    h = np.array([0.0, 0.5, 2.0, -1.3])
    a = np.array([0.3, 1.0, 1.4, 0.8])
    T = np.asarray(fwd.owens_t_jnp(h, a))
    T_neg_a = np.asarray(fwd.owens_t_jnp(h, -a))
    T_neg_h = np.asarray(fwd.owens_t_jnp(-h, a))
    assert np.allclose(T_neg_a, -T, atol=1e-14)   # odd in a
    assert np.allclose(T_neg_h, T, atol=1e-14)    # even in h
    # T(0, a) = arctan(a) / 2pi
    T0 = np.asarray(fwd.owens_t_jnp(0.0, a))
    assert np.allclose(T0, np.arctan(a) / (2 * np.pi), atol=1e-13)


# --- fold vs independent oracle -------------------------------------------------------

def _random_params(pk, rng, scale=1.0):
    tr = pk.truth
    th = tr["theta_true"] + 0.3 * scale * rng.standard_normal(tr["theta_true"].shape)
    pc = 0.2 * scale * rng.standard_normal((pk.n_s, pk.n_molly))
    pkd = 0.05 * scale * rng.standard_normal((2,) + pk.resp_mu_coef.shape[:2])
    lt = 0.3 * scale * rng.standard_normal(pk.n_kk)
    lam = np.abs(rng.standard_normal((pk.n_c, pk.n_s))) * 3.0
    return th, pc, pkd, lt, lam


def _assert_fold_agrees(pk, params):
    consts = fwd.build_consts(pk)
    th, pc, pkd, lt, lam = params
    mu_j = np.asarray(fwd.fold_mu(jnp.asarray(th), jnp.asarray(pc),
                                  jnp.asarray(pkd), jnp.asarray(lt),
                                  jnp.asarray(lam), consts))
    mu_r = fwd.fold_mu_reference(th, pc, pkd, lt, lam, pk)
    assert mu_j.shape == (pk.n_c, pk.n_k, pk.n_s)
    assert np.all(np.isfinite(mu_j)) and np.all(mu_r >= 0.0)
    # rtol 1e-10 with a tiny scale-relative atol for far-tail cells
    np.testing.assert_allclose(mu_j, mu_r, rtol=1e-10, atol=1e-12 * mu_r.max())


def test_fold_vs_reference_small_grid(small_pack):
    rng = np.random.default_rng(7)
    for i in range(5):  # multiple random parameter points
        _assert_fold_agrees(small_pack, _random_params(small_pack, rng))


def test_fold_vs_reference_zero_fp_and_truth_point(small_pack):
    tr = small_pack.truth
    zeros_lam = np.zeros((small_pack.n_c, small_pack.n_s))
    _assert_fold_agrees(small_pack, (tr["theta_true"], tr["psi_c_true"],
                                     np.zeros((2,) + small_pack.resp_mu_coef.shape[:2]),
                                     tr["t_true"], zeros_lam))
    # and: the generator's mu_true is reproduced by the jnp fold at the truth
    consts = fwd.build_consts(small_pack)
    mu_j = np.asarray(fwd.fold_mu(
        jnp.asarray(tr["theta_true"]), jnp.asarray(tr["psi_c_true"]),
        jnp.zeros((2,) + small_pack.resp_mu_coef.shape[:2]),
        jnp.asarray(tr["t_true"]), jnp.asarray(tr["lam_fp_true"]), consts))
    np.testing.assert_allclose(mu_j, tr["mu_true"], rtol=1e-10,
                               atol=1e-12 * tr["mu_true"].max())


def test_fold_vs_reference_full_grid():
    pk = synthetic_pack(3)  # real 29 x 15 x 8 grid
    rng = np.random.default_rng(11)
    _assert_fold_agrees(pk, _random_params(pk, rng))


def test_fold_jits_and_differentiates(small_pack):
    consts = fwd.build_consts(small_pack)
    tr = small_pack.truth
    pkd0 = jnp.zeros((2,) + small_pack.resp_mu_coef.shape[:2])
    lam0 = jnp.asarray(np.abs(tr["lam_fp_true"]) + 0.1)

    def total_mu(theta):
        return fwd.fold_mu(theta, jnp.asarray(tr["psi_c_true"]), pkd0,
                           jnp.asarray(tr["t_true"]), lam0, consts).sum()

    jitted = jax.jit(total_mu)
    v = float(jitted(jnp.asarray(tr["theta_true"])))
    assert np.isfinite(v) and v > 0
    g = np.asarray(jax.grad(total_mu)(jnp.asarray(tr["theta_true"])))
    assert g.shape == tr["theta_true"].shape and np.all(np.isfinite(g))
    assert np.all(g >= 0)  # d(total mu)/d(log f) >= 0 everywhere


def _fp_only_total(pk):
    """Total folded FP: the fold at f == 0 (theta = -inf), log_t = 0, and the
    loa-0 point intensity lam_fp = fp_counts / fp_ell_eff."""
    consts = fwd.build_consts(
        pk, allow_unclamped_response=(pk.resp_N_fit_range is None))
    theta = jnp.full((pk.n_b, pk.n_k), -jnp.inf)
    lam = jnp.asarray(np.asarray(pk.fp_counts, float) / float(pk.fp_ell_eff))
    mu = np.asarray(fwd.fold_mu(
        theta, jnp.zeros((pk.n_s, pk.n_molly)),
        jnp.zeros((2, consts.n_sr, consts.n_zr)),
        jnp.zeros(consts.n_kk), lam, consts))
    assert np.all(np.isfinite(mu))
    return float(mu.sum())


def test_fold_mu_fp_is_exactly_the_folds_own_FP_term(small_pack):
    """``fold_mu_fp`` IS the fold's FP term, at a NON-zero log_t.

    The extraction exists because ``forward_selftest.selftest`` used to re-type
    this expression and its copy had dropped ``exp(log_t)`` -- inert there only
    because that caller passes log_t = 0.  So the pin has to be stated where
    the dropped factor is alive: a random log_t, and bit-for-bit equality
    against ``fold_mu(lam) - fold_mu(0)``.

    MUTATION A: drop ``exp_t_k`` from ``fold_mu_fp`` (i.e. reproduce the copy
    forward_selftest used to carry).  MEASURED: the identity below breaks by up
    to a factor exp(max|log_t|) and both assertions go red.
    MUTATION B: drop ``consts.fp_ell_eff`` from ``fold_mu_fp``.  Both terms
    move together so the difference identity SURVIVES -- which is why the
    separate loa-0-product total test is the one that pins the normalisation,
    and this one pins only that there is a single expression.
    """
    consts = fwd.build_consts(small_pack)
    tr = small_pack.truth
    rng = np.random.default_rng(23)
    log_t = 0.4 * rng.standard_normal(small_pack.n_kk)
    assert np.max(np.abs(log_t)) > 0.05           # the factor must be alive
    lam = np.abs(rng.standard_normal((small_pack.n_c, small_pack.n_s))) * 2.0
    args = (jnp.asarray(tr["theta_true"]), jnp.asarray(tr["psi_c_true"]),
            jnp.zeros((2, consts.n_sr, consts.n_zr)), jnp.asarray(log_t))
    on = np.asarray(fwd.fold_mu(*args, jnp.asarray(lam), consts))
    off = np.asarray(fwd.fold_mu(
        *args, jnp.zeros((small_pack.n_c, small_pack.n_s)), consts))
    direct = np.asarray(fwd.fold_mu_fp(jnp.asarray(log_t), jnp.asarray(lam),
                                       consts))
    np.testing.assert_allclose(direct, on - off, rtol=1e-13,
                               atol=1e-13 * direct.max())
    # and it really does carry exp(log_t): the k-profile ratio against log_t=0
    flat = np.asarray(fwd.fold_mu_fp(jnp.zeros(small_pack.n_kk),
                                     jnp.asarray(lam), consts))
    live = flat.sum(axis=(0, 2)) > 0
    ratio = direct.sum(axis=(0, 2))[live] / flat.sum(axis=(0, 2))[live]
    want = np.exp(log_t)[np.asarray(consts.kz_to_K)][live]
    np.testing.assert_allclose(ratio, want, rtol=1e-12)


@pytest.mark.parametrize("mkpack", [
    pytest.param(lambda: synthetic_pack(5, **small_test_grid()), id="small"),
    pytest.param(lambda: synthetic_pack(5), id="real_grid"),
])
def test_folded_fp_total_equals_the_loa0_product_definition(mkpack):
    """The folded FP total must equal the loa-0 product's OWN mu_FP definition.

    ``build_loa0_fp_product.py:34-39`` defines
        mu_FP = (N_prod / N_sl_loa0) * N_FP_loa0 * (1 - eta_bar)
    i.e. in pack scalars ``fp_w * fp_counts.sum()`` (eta_DLA is forced to 0, so
    (1 - eta_bar) == 1 on this grid).  The fold reaches it through
    ``fp_w * fp_ell_eff * lam_fp`` with ``lam_fp = fp_counts / fp_ell_eff``,
    and the loa-0 exposure cancels EXACTLY -- which is the whole content of the
    2026-08-05 repair.  Summing over k uses ``sum_k fp_E_alloc[k,s] == 1`` per
    populated stratum (schema), so empty strata must carry no loa-0 counts.

    MUTATION: drop ``consts.fp_ell_eff`` from ``forward.fold_mu``'s mu_fp
    expression (the pre-2026-08-05 code).  MEASURED baseline on the adopted
    2LPT-0 pack: the folded total goes 14767.961419068737 -> 1086.6871844096897,
    i.e. short by exactly fp_ell_eff = 13.589891949531907, and this test goes
    red with a ratio of 1/13.59.
    """
    pk = mkpack()
    E = np.asarray(pk.fp_E_alloc, float)
    fp = np.asarray(pk.fp_counts, float)
    populated = E.sum(axis=0) > 0
    assert fp[:, ~populated].sum() == 0.0, \
        "loa-0 counts in a stratum with no exposure allocation"
    np.testing.assert_allclose(E[:, populated].sum(axis=0), 1.0, rtol=1e-12)

    got = _fp_only_total(pk)
    want = float(pk.fp_w_sightline_ratio) * float(fp.sum())
    assert want > 0.0
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("fp_frac", [0.15, 0.4])
@pytest.mark.parametrize("fp_ell_eff", [4.0, 13.589891949531907])
def test_synthetic_generator_uses_the_same_fp_normalisation_as_the_fold(
        fp_frac, fp_ell_eff):
    """``pack.synthetic_pack`` must invert the SAME fold the model uses.

    ``fp_frac`` is documented as the share of the expected DATA counts that is
    false positive, so by construction

        (mu_true.sum() - mu_signal.sum()) / mu_signal.sum() == fp_frac

    exactly.  That identity holds only if the generator's ``fp_data_per_unit``
    carries every factor the fold's FP term carries -- including
    ``fp_ell_eff``.  Until 2026-08-05 it did NOT, with the identical omission
    as ``forward.fold_mu``, which is precisely why no synthetic rung and no SBC
    replica could ever detect the defect: generator and fold were wrong the
    same way.  ``fp_ell_eff`` is varied here so the test cannot pass by the two
    errors cancelling at one particular exposure.

    MUTATION: drop ``fp_ell_eff`` from ``fp_data_per_unit`` in
    ``pack.synthetic_pack`` (the pre-fix line).  MEASURED: the FP share becomes
    fp_frac * fp_ell_eff -- 0.60 instead of 0.15 at ell_eff = 4, and 2.038
    instead of 0.15 at the production ell_eff -- and this goes red.
    """
    pk = synthetic_pack(9, fp_frac=fp_frac, fp_ell_eff=fp_ell_eff,
                        **small_test_grid())
    tr = pk.truth
    sig = float(np.asarray(tr["mu_signal"]).sum())
    fp = float(np.asarray(tr["mu_true"]).sum()) - sig
    assert sig > 0
    np.testing.assert_allclose(fp / sig, fp_frac, rtol=1e-12)


_REAL_PACK_DIR = os.environ.get(
    "MODELA_PACK_DIR",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs")
_REAL_PACKS = sorted(glob.glob(os.path.join(_REAL_PACK_DIR,
                                            "modelA_pack_*.npz")))


@pytest.mark.skipif(not _REAL_PACKS, reason=f"no packs under {_REAL_PACK_DIR}")
@pytest.mark.parametrize("path", _REAL_PACKS,
                         ids=[os.path.basename(p) for p in _REAL_PACKS])
def test_folded_fp_total_on_extracted_packs(path):
    """Same identity as above, on the EXTRACTED packs (real scalars, real grid).

    MEASURED 2026-08-05, adopted 2LPT-0 pack (fp_w = 165.9321507761,
    fp_ell_eff = 13.5898919495, 89 loa-0 counts): folded FP total
    14767.961419068737 == fp_w * 89.

    MUTATION: as above -- drop ``consts.fp_ell_eff`` from ``fold_mu``; the
    folded total collapses to 1086.6871844096897 and this goes red.
    """
    pk = load_pack(path)
    E = np.asarray(pk.fp_E_alloc, float)
    fp = np.asarray(pk.fp_counts, float)
    populated = E.sum(axis=0) > 0
    assert fp[:, ~populated].sum() == 0.0
    got = _fp_only_total(pk)
    # (1 - eta) restoration (2026-08-06, PI ruling 8): the folded total now
    # carries the host-occlusion survival per observed bin. On the committed
    # calibration all 89 events sit below N-hat 20.3, so the total is
    # fp_w * (1 - eta_subdla) * 89 = 14682.949... (was 14767.961 before the
    # restoration; the pre-2026-08-05 defect value 1086.687 still goes red).
    eta_c = np.asarray(pk.fp_eta_c, float)
    want = float(pk.fp_w_sightline_ratio) * float(
        ((1.0 - eta_c)[:, None] * fp).sum())
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_kernel_columns_are_probability_masses(small_pack):
    consts = fwd.build_consts(small_pack)
    K = np.asarray(fwd.build_K(jnp.zeros((2,) + small_pack.resp_mu_coef.shape[:2]),
                               consts))
    assert K.shape == (small_pack.n_s, small_pack.n_kk,
                       small_pack.n_c, small_pack.n_b)
    assert np.all(K >= 0.0) and np.all(K <= 1.0)
    colsum = K.sum(axis=2)  # mass of each true bin landing anywhere observed
    assert np.all(colsum <= 1.0 + 1e-12)
    assert np.all(colsum > 0.05)  # nothing pathologically lost on this grid


# ---------------------------------------------------------------------------
# B2 (2026-08-05) — the two _SN_SKEW_MAX copies, pinned rather than commented
# ---------------------------------------------------------------------------
def test_SN_SKEW_MAX_equals_the_znz_kernel_constant():
    """``forward._SN_SKEW_MAX`` re-types ``znz_kernel._SN_SKEW_MAX``. Until
    2026-08-05 the only thing asserting they agreed was a comment.

    An ASSERTION, not an import, because ``forward.py`` must stay importable
    without the heavy ``CDDF_analysis.hbi`` chain — see the companion test."""
    from CDDF_analysis.hbi import znz_kernel as ZK
    from CDDF_analysis.hbi_mcmc import forward as _F
    assert _F._SN_SKEW_MAX == ZK._SN_SKEW_MAX
    # both are the attainable moment-skewness ceiling of the skew-normal
    # family, i.e. the a -> inf limit; stated numerically so a change to EITHER
    # copy has to argue with a number
    import numpy as _np
    closed_form = 0.5 * (4.0 - _np.pi) * (_np.sqrt(2.0 / _np.pi) ** 3) \
        / (1.0 - 2.0 / _np.pi) ** 1.5
    assert _F._SN_SKEW_MAX == closed_form
    assert repr(float(_F._SN_SKEW_MAX)) == "0.9952717464311566"
    # and the clamp both modules apply is the same fraction of it
    assert 0.995 * _F._SN_SKEW_MAX == 0.995 * ZK._SN_SKEW_MAX


def test_forward_imports_without_the_hbi_chain():
    """THE constraint that makes the duplication the right call: importing
    ``CDDF_analysis.hbi_mcmc.forward`` must not drag in ``CDDF_analysis.hbi``
    (or ``znz_kernel``). Run in a FRESH interpreter, since this test process has
    almost certainly imported them already."""
    import os
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import CDDF_analysis.hbi_mcmc.forward as F\n"
        "bad = [m for m in ('CDDF_analysis.hbi', 'CDDF_analysis.hbi.znz_kernel')\n"
        "       if m in sys.modules]\n"
        "print('LEAKED=' + ','.join(bad))\n"
        "print('VALUE=' + repr(float(F._SN_SKEW_MAX)))\n" % repo)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=repo)
    assert out.returncode == 0, out.stderr
    assert "LEAKED=\n" in out.stdout or "LEAKED=" + "\n" in out.stdout, out.stdout
    assert "VALUE=0.9952717464311566" in out.stdout, out.stdout
