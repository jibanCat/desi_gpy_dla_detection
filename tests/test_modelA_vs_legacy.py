"""Q3 §3 characterization gate: the JAX Model A fold vs the LEGACY HBI
expected-count construction, at FIXED calibration (the oracle rule).

    "At fixed calibration (Ψ_C = point, Ψ_K = point, t = 0), the JAX fold μ(f)
     must reproduce the legacy A_ib/M_b construction's expected counts on the
     same data pack to rtol 1e-6 — the legacy builder is the validator, not
     the method."

Comparable objects (driven through the COMMITTED legacy entry points via
``CDDF_analysis/hbi_mcmc/legacy_oracle.py``):

  (a) TOTAL expected detected counts — legacy ``build_M_b``+``_apply_C_to_M``
      (g-threaded 3-D path, per-stratum) vs the fold at UNIT kernel mass.
      CONVENTION: legacy M integrates TRUE-N space without the response, so it
      counts every detection regardless of where N̂ lands; the fold's K-mass is
      truncated to the observed window [19.5, 22.4). The out-of-window (OOW)
      migration mass is quantified and the identity
          legacy_total − fold_total(K_leg) = Σ_b (1 − Σ_c K_leg)·contrib
      is asserted — the residual AFTER that correction is the fidelity metric.
  (b) per-(c,k,s) expected intensities — the committed ``_build_A_ib_forward``
      + ``_apply_C_to_A`` on synthetic detections (density level at bin
      centers, and Gauss-Legendre x̂-integrated) vs the fold contraction.

VERDICT (2026-07-11, real 2LPT-0 pack; all values mock-only). The original
characterization found the fold STRUCTURE exact (~1e-13) but kernel INGESTION
broken by five numbered findings; the fixes landed the same day
(forward.py/pack.py) and this suite now ASSERTS the fixed state — the gate
test test_Tk1 requires build_K == the frozen ForwardResponseModel masses to
1e-6 (observed ~1e-14):
  F1  covariate reference: build_consts used n_ref = mean(ntrue ends) = 20.95;
      the coefficients were fit at N_ref = 20.104. FIXED: build_consts and
      fold_mu_reference REQUIRE pack.resp_N_ref (fail-loud, no fallback).
  F1b load_pack silently dropped resp_N_ref (and every other optional key).
      FIXED: carried into the dataclass (test_Tk3).
  F2  width transform was floor + softplus(poly); legacy is clip(poly, floor).
      FIXED (softplus(~0.1) ≈ 0.74 dex had smeared the kernel: diag mass
      ~0.03–0.05 vs legacy ~0.18–0.31).
  F3  skew ramp was logistic((N−21)/0.5) (≈0 below 21, →1 above); legacy is
      (1 − clip((N−21)/0.5, 0, 1)) (full skew below 21, ZERO above 21.5).
      FIXED to the legacy direction and shape.
  F4  (N+μpoly, σ, γ) were used as skew-normal (ξ, ω, a) directly; legacy
      moment-matches via _moment_to_skewnormal_vec with the attainable-
      skewness clamp. FIXED (forward.moment_to_skewnormal_jnp).
  F5  s_to_sresp = digitize(lower_edge)−1 gave −1 for the SNR<2 strata → the
      negative-index gather WRAPPED to the highest response cell (masked only
      by dX ≡ 0 there). FIXED: fail-loud if such strata carry exposure, else
      clamp (the legacy _i_snr clip convention) — test_Tk2.
Registration conventions (documented MODEL SYSTEMATICS, deliberately NOT
code-changed): the fold conditions the response cell on (stratum LOWER edge,
coarse-z-bin CENTER) where legacy conditions on each detection's own
(SNR, z_QSO=zqso covariate); strata [3,4) and [6,7) straddle response SNR
edges 3.5/6.5, and the z-covariate axis itself differs (absorber coarse-z vs
quasar z — the frozen model was fit on z_QSO).

ENV: needs jax + fitsio + astropy (the gpdla-hbi env); heavy inputs on
/scratch + /nfs/turbo mock paths — every test skips when absent.
Run:  conda run -n gpdla-hbi python -m pytest tests/test_modelA_vs_legacy.py -v -s
"""
from __future__ import annotations

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("fitsio")

import jax.numpy as jnp  # noqa: E402

from CDDF_analysis.hbi_mcmc.pack import load_pack  # noqa: E402

# 2026-08-06 (fp_eta_c restoration): legacy fixture packs predate the schema
# field; migrate them EXPLICITLY at the test boundary (idempotent; values
# identical to a fresh extraction — pack.FP_ETA_BANDS_COMMITTED).
from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _attach_fp_eta
load_pack = (lambda _f: (lambda *a, **k: _attach_fp_eta(_f(*a, **k))))(load_pack)
from CDDF_analysis.hbi_mcmc import forward as F  # noqa: E402
from CDDF_analysis.hbi_mcmc import legacy_oracle as LO  # noqa: E402

_MOLLY_TSV = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
              "figures_molly_nhi195/lya_only/molly_matrix.tsv")
_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/"
            "v2.8.5/mock-0/loa-124")

_HAVE_PACK = os.path.exists(LO.DEF_PACK) and os.path.exists(LO.DEF_FWD)
_HAVE_A_INPUTS = _HAVE_PACK and os.path.exists(_MOLLY_TSV)
_HAVE_M_INPUTS = _HAVE_A_INPUTS and (
    os.path.exists(LO.DEF_M_CACHE) or os.path.exists(_MOCKDIR))

pytestmark = pytest.mark.skipif(
    not _HAVE_PACK, reason="real 2LPT-0 Model A pack / frozen forward NPZ absent")

# atol guard for ratio metrics: cells at ~1e-9 of the array max are compared
# absolutely (REACH-boundary straddle produces exact-zero legacy cells).
_ATOL_FRAC = 1e-9


def _relmax(a, b):
    """max |a−b| / (|b| + atol_frac·max|b|) — rtol with a scale-aware atol."""
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b) / (np.abs(b) + _ATOL_FRAC * scale)))


# ---------------------------------------------------------------------------
# session fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pack():
    return load_pack(LO.DEF_PACK)


@pytest.fixture(scope="session")
def frm():
    from CDDF_analysis.hbi.znz_kernel import load_forward_response
    return load_forward_response(LO.DEF_FWD)


@pytest.fixture(scope="session")
def consts(pack):
    # resp_clamp="off": this whole module CHARACTERIZES the fold against the
    # committed legacy ForwardResponseModel, whose _eval_surface has no
    # covariate-range guard. Comparing like-for-like therefore REQUIRES the
    # unclamped fold. The clamp (finding D2, 2026-07-28) is a deliberate
    # DIVERGENCE from the legacy object and is tested in
    # tests/test_modelA_forward_selftest.py, not here.
    return F.build_consts(pack, resp_clamp="off", allow_unclamped_response=True)


@pytest.fixture(scope="session")
def f_battery(pack):
    """Test population vectors f_new[b,k] (per-dex density on the pack grid)."""
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    zf = np.asarray(pack.zf_edges, float)
    zfc = 0.5 * (zf[:-1] + zf[1:])
    B, Kf = len(Nc), len(zfc)
    out = {"flat": np.ones((B, Kf))}
    for sl in (-1.5, -2.5):
        # per-dex density of an f(N) ∝ N^sl linear-N power law
        fl = (10.0 ** (Nc - 20.3)) ** (sl + 1.0)
        out[f"powerlaw{sl}"] = np.repeat(fl[:, None], Kf, axis=1)
    out["ztilt"] = (np.repeat(((1.0 + zfc) / 3.25)[None, :] ** 3.0, B, axis=0)
                    * 10.0 ** (-1.0 * (Nc - 20.3))[:, None])
    for b in (0, 10, 28):
        d = np.zeros((B, Kf))
        d[b, :] = 1.0
        out[f"delta_b{b}"] = d
    return out


@pytest.fixture(scope="session")
def K_leg(pack, frm):
    return LO.legacy_K_masses(pack, frm)


@pytest.fixture(scope="session")
def M_legacy(pack):
    if not _HAVE_M_INPUTS:
        pytest.skip("legacy M inputs (molly tsv / mock qso cats / cache) absent")
    return LO.build_or_load_legacy_M(pack)


def _fold_mu_point(pack, consts, f_new):
    """fold_mu at the FIXED calibration point: ψ_C=0, ψ_K=0, t=0, FP OFF
    (λ_FP ≡ 0 — equivalently exp(t)·λ with t → log 0)."""
    S = len(np.asarray(pack.snr_edges)) - 1
    C = len(np.asarray(pack.nhat_edges)) - 1
    M = np.asarray(pack.molly_n_det).shape[1]
    SR, ZR = np.asarray(pack.resp_mu_coef).shape[:2]
    KK = len(np.asarray(pack.zc_edges)) - 1
    with np.errstate(divide="ignore"):
        theta = jnp.asarray(np.log(np.asarray(f_new, float)))
    return np.asarray(F.fold_mu(theta, jnp.zeros((S, M)), jnp.zeros((2, SR, ZR)),
                                jnp.zeros(KK), jnp.zeros((C, S)), consts))


# ---------------------------------------------------------------------------
# T0 — module-internal oracle sanity (grounds fold_mu == fold_mu_reference
# on the REAL pack before anything is compared against legacy)
# ---------------------------------------------------------------------------
def test_T0_fold_matches_inmodule_oracle(pack, consts, f_battery):
    f = f_battery["powerlaw-1.5"]
    mu = _fold_mu_point(pack, consts, f)
    with np.errstate(divide="ignore"):
        theta = np.log(f)
    S = mu.shape[2]
    M = np.asarray(pack.molly_n_det).shape[1]
    SR, ZR = np.asarray(pack.resp_mu_coef).shape[:2]
    KK = len(np.asarray(pack.zc_edges)) - 1
    mu_ref = F.fold_mu_reference(theta, np.zeros((S, M)), np.zeros((2, SR, ZR)),
                                 np.zeros(KK), np.zeros((mu.shape[0], S)), pack,
                                 resp_clamp="off",
                                 allow_unclamped_response=True)
    assert _relmax(mu, mu_ref) < 1e-10


# ---------------------------------------------------------------------------
# (a) TOTAL expected detected counts vs the legacy M_b construction
# ---------------------------------------------------------------------------
def test_Ta1_pathlength_is_the_committed_PX(pack, M_legacy):
    """The pack dX must BE build_M_b's PX (same committed routine) — bitwise."""
    assert np.array_equal(np.asarray(M_legacy["PX"]).T, np.asarray(pack.dX))


def test_Ta2_total_counts_match_legacy_M(pack, M_legacy, f_battery):
    """Legacy M_s·f (per (b,k,s), g-threaded, fixed C) == the fold at unit
    kernel mass, after the documented f-units mapping. Target rtol 1e-6;
    observed ~4e-16."""
    for name, f in f_battery.items():
        leg = LO.legacy_M_expected_counts(pack, M_legacy, f)
        new = LO.kernel_free_mu(pack, f)
        r = _relmax(new, leg)
        print(f"[Ta2] {name:14s} max rel (fold@K=1 vs legacy M·f) = {r:.3e}")
        assert r < 1e-6, f"{name}: fold(K=1) vs legacy M·f residual {r}"


def test_Ta3_out_of_window_migration_identity(pack, M_legacy, K_leg, f_battery):
    """legacy_total − fold_total(K_leg) == Σ_b (1 − Σ_c K_leg)·dX·contrib.
    The OOW migration mass is the WHOLE difference between the two counting
    conventions; the residual after correcting for it is ~1e-16."""
    kz2K = np.asarray(pack.kz_to_K, int)
    colmass = K_leg.sum(axis=2)                       # (S, KK, B)
    W = (1.0 - colmass[:, kz2K, :]).transpose(2, 1, 0)  # (B, Kf, S)
    print(f"[Ta3] {'f-case':14s} {'legacy_tot':>12s} {'OOW_mass':>11s} "
          f"{'OOW_%':>7s} {'identity_rel':>12s}")
    for name, f in f_battery.items():
        leg_tot = float(LO.legacy_M_expected_counts(pack, M_legacy, f).sum())
        mu_kleg_tot = float(LO.fold_with_K(pack, K_leg, f).sum())
        oow = float(np.sum(W * LO.kernel_free_mu(pack, f)))
        resid = abs((leg_tot - mu_kleg_tot) - oow) / max(leg_tot, 1e-300)
        print(f"[Ta3] {name:14s} {leg_tot:12.4e} {oow:11.4e} "
              f"{100 * oow / leg_tot:7.3f} {resid:12.3e}")
        assert resid < 1e-12, f"{name}: OOW identity residual {resid}"


# ---------------------------------------------------------------------------
# (b) per-(c,k,s) intensities vs the committed A_ib builder
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_A_INPUTS, reason="molly tsv absent")
def test_Tb1_density_level_A_fold_threading(pack, frm, f_battery):
    """Committed _build_A_ib_forward + _apply_C_to_A row·f at N̂-bin-center
    synthetic detections == the fold contraction with the response DENSITY
    injected (same C/g/f/ΔN threading, dX/dz(ẑ)→dX[k,s] conversion, REACH).
    This validates the fold's contraction STRUCTURE against the legacy A
    construction independently of kernel ingestion. Observed ~1e-13–1e-9."""
    Kd = LO.legacy_K_density(pack, frm, reach=2.0)
    for name in ("flat", "powerlaw-2.5", "ztilt", "delta_b28"):
        f = f_battery[name]
        legA = LO.legacy_A_cell_intensity(pack, f, gl_nodes=0)
        Dnew = LO.fold_with_K(pack, Kd, f)
        r = _relmax(Dnew, legA)
        print(f"[Tb1] {name:14s} max rel (fold-contraction vs A_ib·f) = {r:.3e}")
        assert r < 1e-6, f"{name}: A-side threading residual {r}"


@pytest.mark.skipif(not _HAVE_A_INPUTS, reason="molly tsv absent")
def test_Tb2_integrated_A_quadrature_vs_K_masses(pack, frm, f_battery):
    """x̂-integrated committed A rows (Gauss-Legendre over each N̂ bin) vs the
    fold with the EXACT legacy CDF masses (REACH-masked). GL on the clamped-
    skew cells integrates a kinked pdf (a at the attainable-skewness ceiling
    ⇒ near-half-normal corner inside the bin), which limits the quadrature —
    NOT the construction — so the smooth flat case is asserted tight and the
    steep case loose (the mass-level equality is already implied analytically
    by Tb1 + the exact-CDF kernel identity in test_Tk2_decomposition_table)."""
    K_leg_reach = LO.legacy_K_masses(pack, frm, reach=2.0)
    for name, tol in (("flat", 2e-6), ("powerlaw-2.5", 1e-3)):
        f = f_battery[name]
        legQ = LO.legacy_A_cell_intensity(pack, f, gl_nodes=32)
        muK = LO.fold_with_K(pack, K_leg_reach, f)
        r = _relmax(muK, legQ)
        print(f"[Tb2] {name:14s} max rel (fold(K_leg) vs GL32-integrated A) = {r:.3e}")
        assert r < tol, f"{name}: integrated A residual {r} (tol {tol})"


# ---------------------------------------------------------------------------
# kernel ingestion: THE gate assertion + the finding decomposition
# ---------------------------------------------------------------------------
def test_Tk1_GATE_kernel_masses_match_legacy(pack, consts, K_leg):
    """THE §3 gate: build_K must reproduce the frozen ForwardResponseModel's
    exact CDF bin masses on every exposure-carrying stratum to 1e-6.
    (Was a strict xfail while findings F1–F4 were open; flipped 2026-07-11
    with the fixes. Observed ~1e-14: the only remaining difference is the
    GL-64 Owen's T vs scipy's exact owens_t.)"""
    SR, ZR = np.asarray(pack.resp_mu_coef).shape[:2]
    K_new = np.asarray(F.build_K(jnp.zeros((2, SR, ZR)), consts))
    live = np.asarray(pack.dX).sum(axis=0) > 0
    gap = np.max(np.abs(K_new - K_leg)[live])
    print(f"[Tk1] max|build_K − K_leg| (live strata) = {gap:.3e}")
    assert gap < 1e-6


def test_Tk2_decomposition_table(pack, frm, consts, K_leg):
    """Historical per-mechanism attribution of F1–F4 (kept as the record of
    WHY each convention matters), plus the fixed-state assertions:
    build_K now equals the all-legacy-conventions variant AND the exact CDF
    masses; the F5 sub-range strata are clamped (never a wrapping gather)
    and verified structurally empty."""
    live = np.asarray(pack.dX).sum(axis=0) > 0
    variants = [
        ("pre-fix (nref=20.95, softplus, sigmoid, direct)",
         LO.K_from_pack_coeffs(pack, 20.95, "softplus", "sigmoid", "direct")),
        ("+F1 N_ref=NPZ (20.104)",
         LO.K_from_pack_coeffs(pack, frm.N_ref, "softplus", "sigmoid", "direct")),
        ("+F2 sigma clip",
         LO.K_from_pack_coeffs(pack, frm.N_ref, "clip", "sigmoid", "direct")),
        ("+F3 legacy skew ramp",
         LO.K_from_pack_coeffs(pack, frm.N_ref, "clip", "legacy", "direct")),
        ("+F4 moment-matched (xi,omega,a)  [= all legacy]",
         LO.K_from_pack_coeffs(pack, frm.N_ref, "clip", "legacy", "moment")),
    ]
    print(f"[Tk2] {'variant':55s} {'max|K − K_leg|':>15s}")
    for name, Kv in variants:
        print(f"[Tk2] {name:55s} {np.abs(Kv - K_leg).max():15.3e}")
    # end point: all-legacy conventions reproduce the committed model exactly
    assert np.abs(variants[-1][1] - K_leg).max() < 1e-12
    # FIXED state: build_K == the all-legacy variant on the live strata
    SR, ZR = np.asarray(pack.resp_mu_coef).shape[:2]
    K_new = np.asarray(F.build_K(jnp.zeros((2, SR, ZR)), consts))
    assert np.abs(K_new - variants[-1][1])[live].max() < 1e-12
    # F5 fixed: sub-range strata are CLAMPED into the response-cell range
    # (legacy _i_snr convention), never negative, and structurally empty.
    assert consts.s_to_sresp.min() >= 0
    assert consts.s_to_sresp.max() < SR
    assert np.asarray(pack.counts)[:, :, ~live].sum() == 0


def test_Tk3_resp_N_ref_carried_and_required(pack):
    """F1/F1b fixed state: the NPZ's resp_N_ref is carried by load_pack, and
    build_consts / fold_mu_reference REFUSE a pack without it (fail-loud —
    the silent 20.95-midpoint fallback was the F1 defect)."""
    import dataclasses as _dc
    with np.load(LO.DEF_PACK) as z:
        assert "resp_N_ref" in z.files
        npz_ref = float(z["resp_N_ref"])
    assert abs(npz_ref - 20.104069697852808) < 1e-9
    assert pack.resp_N_ref is not None
    assert abs(float(pack.resp_N_ref) - npz_ref) < 1e-12
    stripped = _dc.replace(pack, resp_N_ref=None)
    with pytest.raises(ValueError, match="resp_N_ref"):
        F.build_consts(stripped)
    with pytest.raises(ValueError, match="resp_N_ref"):
        F.fold_mu_reference(
            np.zeros((pack.n_b, pack.n_k)), np.zeros((pack.n_s, pack.n_molly)),
            np.zeros((2,) + np.asarray(pack.resp_mu_coef).shape[:2]),
            np.zeros(pack.n_kk), np.zeros((pack.n_c, pack.n_s)), stripped)


# ---------------------------------------------------------------------------
# headline characterization row: the as-implemented fold vs legacy, per f
# ---------------------------------------------------------------------------
def test_Tc1_report_fold_vs_legacy_at_fixed_calibration(
        pack, consts, M_legacy, K_leg, f_battery):
    """The §3 verdict table. FIXED state: the fold total equals fold(K_leg)
    (same kernel now) and reproduces legacy after the documented OOW-migration
    correction — both asserted at the gate tolerance per test-f."""
    print(f"[Tc1] {'f-case':14s} {'legacy_tot':>12s} {'fold_tot':>12s} "
          f"{'fold/legacy':>11s} {'fold(Kleg)/legacy':>18s} {'OOW_%':>7s}")
    for name, f in f_battery.items():
        leg = float(LO.legacy_M_expected_counts(pack, M_legacy, f).sum())
        mu_new = float(_fold_mu_point(pack, consts, f).sum())
        mu_kleg = float(LO.fold_with_K(pack, K_leg, f).sum())
        print(f"[Tc1] {name:14s} {leg:12.4e} {mu_new:12.4e} "
              f"{mu_new / leg:11.4f} {mu_kleg / leg:18.4f} "
              f"{100 * (leg - mu_kleg) / leg:7.3f}")
        # the fold now carries the legacy kernel: totals agree at gate tol
        assert abs(mu_new - mu_kleg) / max(leg, 1e-300) < 1e-6
        # and reproduces legacy exactly after the OOW-migration correction
        kz2K = np.asarray(pack.kz_to_K, int)
        W = (1.0 - K_leg.sum(axis=2)[:, kz2K, :]).transpose(2, 1, 0)
        oow = float(np.sum(W * LO.kernel_free_mu(pack, f)))
        assert abs(mu_kleg + oow - leg) / leg < 1e-6
        assert abs(mu_new + oow - leg) / leg < 1e-6


# ---------------------------------------------------------------------------
# B3 (2026-08-05) — the convention mapping was INVERTED, and there was no clamp
# ---------------------------------------------------------------------------
def _synth_pack():
    from CDDF_analysis.hbi_mcmc.pack import synthetic_pack, small_test_grid
    return synthetic_pack(0, **small_test_grid())


def test_B3_the_documented_convention_mapping_is_the_measured_one():
    """``K_from_pack_coeffs``'s docstring claimed
    ``(midpoint, 'softplus', 'sigmoid', 'direct')`` reproduces
    ``forward.build_K``. It is the PRE-FIX recipe; the mapping was exactly
    inverted. Both numbers pinned on synthetic_pack(0, **small_test_grid()),
    whose ``resp_N_ref`` IS the grid midpoint 20.0."""
    pk = _synth_pack()
    assert float(pk.resp_N_ref) == 20.0
    assert float(np.asarray(pk.ntrue_edges, float).mean()) == pytest.approx(20.0)
    SR, ZR = np.asarray(pk.resp_mu_coef).shape[:2]
    c = F.build_consts(pk, resp_clamp="both")
    K = np.asarray(F.build_K(jnp.zeros((2, SR, ZR)), c))

    pre = LO.K_from_pack_coeffs(pk, 20.0, "softplus", "sigmoid", "direct")
    com = LO.K_from_pack_coeffs(pk, 20.0, "clip", "legacy", "moment")
    d_pre = float(np.abs(pre - K).max())
    d_com = float(np.abs(com - K).max())
    # the two MEASURED numbers, pinned
    assert d_pre == pytest.approx(2.811e-01, rel=1e-3), d_pre
    assert d_com == pytest.approx(4.441e-16, rel=1e-3), d_com
    # ...and the ORDERING, which is the whole claim: the committed tuple is the
    # one that reproduces build_K, by ~15 orders of magnitude
    assert d_com < 1e-12 < 1e-3 < d_pre
    # the retracted sentence must not be back in the source
    import inspect
    src = inspect.getsource(LO.K_from_pack_coeffs)
    assert "CORRECTION (2026-08-05)" in src
    assert "2.811e-01" in src and "4.441e-16" in src


def test_B3_the_D2_clamp_is_selectable_and_off_by_default(pack):
    """Before B3 this function implemented NO covariate clamp, so the only
    cross-convention kernel instrument on the path evaluated the UNCLAMPED
    kernel while the fold it is compared against clamps by default.

    MEASURED on the committed v1.1 2LPT-0 pack, over the live strata:
      oracle clamp_mode='off'  vs fold resp_clamp='both' : max|diff| = 3.55e-01
      oracle clamp_mode='both' vs fold resp_clamp='both' : max|diff| = 6.38e-15
    i.e. the instrument was a THIRD of a unit of kernel bin mass away from
    production, and now is not — if you ask for the clamp."""
    import os
    from CDDF_analysis.hbi_mcmc.pack import load_pack as _lp
    from CDDF_analysis.hbi_mcmc.pack import attach_fp_eta_bands as _aeta
    _lp = (lambda _f: (lambda *a, **k: _aeta(_f(*a, **k))))(_lp)
    v11 = LO.DEF_PACK.replace(".npz", "_v11.npz")
    if not os.path.exists(v11):
        pytest.skip("v1.1 pack absent")
    rp = _lp(v11)
    assert rp.resp_N_fit_range is not None
    SR, ZR = np.asarray(rp.resp_mu_coef).shape[:2]
    nref = float(rp.resp_N_ref)
    live = np.asarray(rp.dX).sum(axis=0) > 0

    Kf = np.asarray(F.build_K(jnp.zeros((2, SR, ZR)),
                              F.build_consts(rp, resp_clamp="both")))
    off = LO.K_from_pack_coeffs(rp, nref, "clip", "legacy", "moment")
    both = LO.K_from_pack_coeffs(rp, nref, "clip", "legacy", "moment",
                                 clamp_mode="both")
    d_off = float(np.abs(off - Kf)[live].max())
    d_both = float(np.abs(both - Kf)[live].max())
    assert d_off == pytest.approx(3.5505e-01, rel=1e-3), d_off
    assert d_both < 1e-13, d_both

    # DEFAULT PRESERVES THE OLD BEHAVIOUR: no clamp_mode == clamp_mode='off'
    np.testing.assert_array_equal(
        off, LO.K_from_pack_coeffs(rp, nref, "clip", "legacy", "moment",
                                   clamp_mode="off"))
    # ...and 'off' still reproduces the UNCLAMPED fold exactly, which is what
    # this module's `consts` fixture builds
    Ku = np.asarray(F.build_K(jnp.zeros((2, SR, ZR)),
                              F.build_consts(rp, resp_clamp="off")))
    assert float(np.abs(off - Ku)[live].max()) < 1e-13

    # 'hi' is the third mode forward.build_consts takes and it agrees with the
    # fold's 'hi' too
    Kh = np.asarray(F.build_K(jnp.zeros((2, SR, ZR)),
                              F.build_consts(rp, resp_clamp="hi")))
    hi = LO.K_from_pack_coeffs(rp, nref, "clip", "legacy", "moment",
                               clamp_mode="hi")
    assert float(np.abs(hi - Kh)[live].max()) < 1e-13


def test_B3_clamp_mode_fails_closed():
    """A bad mode, and a clamp asked of a pack that cannot supply the range,
    must RAISE — never silently fall back to the unclamped kernel, which is the
    failure this whole finding is about."""
    import dataclasses
    pk = _synth_pack()
    with pytest.raises(ValueError, match="clamp_mode"):
        LO.K_from_pack_coeffs(pk, 20.0, "clip", "legacy", "moment",
                              clamp_mode="yes")
    stripped = dataclasses.replace(pk, resp_N_fit_range=None)
    with pytest.raises(ValueError, match="resp_N_fit_range"):
        LO.K_from_pack_coeffs(stripped, 20.0, "clip", "legacy", "moment",
                              clamp_mode="both")
    # ...but clamp_mode='off' on that same pack still works, and is documented
    # as the unclamped kernel
    assert LO.K_from_pack_coeffs(stripped, 20.0, "clip", "legacy",
                                 "moment").shape[0] > 0
