"""Unit tests for the low-DOF response-representation candidates
(validation/absorber_ladder/response_review/candidates/).

Every check is either an exact identity (row normalisation, Jeffreys formula,
count-conservation of the delivered Mg tensor), a closed form derived
independently of the implementation (paired SE, row KL, PIT), a recovery test
on a synthetic row set whose truth is inside the family, or a MUTATION test
that a deliberately wrong normalisation is rejected.  Tolerances are explicit;
``np.allclose``'s default atol is never relied on
(feedback_allclose_atol_tiny_values).

ENV: gpdla-hbi.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_CAND = os.path.join(_REPO, "validation", "absorber_ladder", "response_review",
                     "candidates")
_RESP = os.path.join(_REPO, "validation", "absorber_ladder", "response")
for _p in (_REPO, _RESP, _CAND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import candlib as CL                                            # noqa: E402
import candmetrics as CM                                        # noqa: E402
import families as F                                            # noqa: E402


# ---------------------------------------------------------------------------
# a small synthetic geometry (no data files needed)
# ---------------------------------------------------------------------------
def toy_geom(B=4, C=9, S=3, K=2):
    nt = np.linspace(20.0, 21.6, B + 1)
    nh = np.linspace(19.9, 21.7, C + 1)
    se = np.array([0.0, 2.0, 5.0, np.inf])[: S + 1]
    zc = np.linspace(2.0, 3.0, K + 1)
    g = dict(ntrue=nt, nhat=nh, snr=se, zc=zc, B=B, C=C, S=S, K=K,
             ccen=0.5 * (nh[:-1] + nh[1:]), bcen=0.5 * (nt[:-1] + nt[1:]))
    return g


def toy_events(geom, n=4000, seed=3):
    rs = np.random.RandomState(seed)
    N = rs.uniform(geom["ntrue"][0], geom["ntrue"][-1], n)
    snr = 10 ** rs.uniform(0.1, 1.2, n)
    z = rs.uniform(geom["zc"][0], geom["zc"][-1], n)
    xh = N + rs.normal(0.0, 0.12, n)
    ev = dict(N_true=N, snr=snr, zqso=z, zdla=z, xhat=xh, dx=xh - N,
              tid=np.arange(n) * 2 + rs.randint(0, 2, n), n=n)
    ev["b_i"] = np.clip(np.digitize(N, geom["ntrue"]) - 1, 0, geom["B"] - 1)
    ev["s_i"] = np.clip(np.digitize(snr, geom["snr"]) - 1, 0, geom["S"] - 1)
    ev["K_i"] = np.clip(np.digitize(z, geom["zc"]) - 1, 0, geom["K"] - 1)
    ev["c_i"] = np.clip(np.digitize(xh, geom["nhat"]) - 1, 0, geom["C"] - 1)
    ev["u"] = N - CL.N_REF_U
    ev["v"] = np.log10(snr) - CL.V_REF
    ev["fold"] = (ev["tid"] % 2) == 0
    return ev


# ===========================================================================
# 1-3.  exact row normalisation of EVERY delivered object
# ===========================================================================
def test_lowrank_rows_sum_to_one_exactly():
    g = toy_geom()
    fam = F.LowRank("A-small", "asmall", R=2)
    rs = np.random.RandomState(0)
    par = dict(alpha=rs.normal(size=(2, 6)), phi=rs.normal(size=(2, g["C"])),
               P=6, C=g["C"], R=2, kind="asmall")
    X = CL.design(np.linspace(-1, 1, 11), np.linspace(-0.5, 0.5, 11),
                  np.zeros(11, int), "asmall")
    p = np.exp(fam.logp_rows(par, X, None))
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-12


def test_lowrank_with_base_rows_sum_to_one_exactly():
    g = toy_geom()
    fam = F.LowRank("E", "e", R=2, use_base=True)
    rs = np.random.RandomState(1)
    base = rs.dirichlet(np.ones(g["C"]), size=13)
    par = dict(alpha=rs.normal(size=(2, 5)), phi=rs.normal(size=(2, g["C"])),
               P=5, C=g["C"], R=2, kind="e")
    X = CL.design(np.linspace(-1, 1, 13), np.zeros(13), np.zeros(13, int), "e")
    p = np.exp(fam.logp_rows(par, X, np.log(base)))
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-12


def test_quantile_spline_is_monotone_and_normalised():
    g = toy_geom()
    fam = F.QuantileSpline(n_knot=4)
    rs = np.random.RandomState(2)
    par = dict(beta=rs.normal(scale=0.3, size=(fam.NPAR, 6)), P=6,
               loc_scale=8.0, n_knot=4, kind="full")
    X = CL.design(np.linspace(-1, 1, 9), np.zeros(9), np.zeros(9, int), "full")
    p = np.exp(fam.logp_rows(par, X, np.linspace(20.0, 21.5, 9), g))
    assert np.min(p) >= 0.0
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-12
    # the implied CDF is non-decreasing -- monotonicity by construction
    assert np.min(np.diff(np.cumsum(p, axis=1), axis=1)) >= -1e-15


def test_mixture_rows_sum_to_one_exactly():
    g = toy_geom()
    fam = F.Mixture()
    rs = np.random.RandomState(4)
    beta = np.zeros((fam.NPAR, 6)); beta[0, 0] = 0.5
    beta[2, 0] = np.log(0.12); beta[4, 0] = 0.1; beta[5, 0] = np.log(0.3)
    beta += 0.02 * rs.normal(size=beta.shape)
    par = dict(beta=beta, P=6, kind="full")
    X = CL.design(np.linspace(-1, 1, 7), np.zeros(7), np.zeros(7, int),
                  "full")
    p = np.exp(fam.logp_rows(par, X, np.linspace(20.1, 21.4, 7), g))
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-12


# ===========================================================================
# 4.  MUTATION: a wrong normalisation must be caught
# ===========================================================================
def test_mutation_wrong_normalisation_is_detected():
    """Normalising over the LATENT axis instead of the observed axis (the
    recurring one-sided-support bug class) must break both the exact row-sum
    identity and the row-KL metric."""
    g = toy_geom()
    rs = np.random.RandomState(5)
    rows = rs.dirichlet(np.ones(g["C"]), size=(g["B"], g["S"], g["K"]))
    assert np.max(np.abs(rows.sum(axis=-1) - 1.0)) < 1e-12
    bad = rows / rows.sum(axis=0, keepdims=True)          # WRONG axis
    assert np.max(np.abs(bad.sum(axis=-1) - 1.0)) > 1e-3
    counts = np.zeros_like(rows)
    counts[..., :] = np.round(rows * 1000)
    good_kl = CM.row_table(rows, counts, g, min_events=10)
    bad_kl = CM.row_table(bad, counts, g, min_events=10)
    assert CM.wmean(bad_kl, "kl") > 10.0 * CM.wmean(good_kl, "kl")


# ===========================================================================
# 5.  identifiability / canonical gauge
# ===========================================================================
def test_canonicalise_preserves_the_likelihood_and_fixes_the_gauge():
    rs = np.random.RandomState(6)
    C, P, R = 29, 6, 2
    ccen = np.linspace(19.55, 22.35, C)
    alpha = rs.normal(size=(R, P)); phi = rs.normal(size=(R, C))
    X = rs.normal(size=(50, P))
    eta = (X @ alpha.T) @ phi
    lp = eta - np.log(np.exp(eta - eta.max(1, keepdims=True)).sum(
        1, keepdims=True)) - eta.max(1, keepdims=True)
    a2, p2, sv = CL.canonicalise(alpha, phi, ccen)
    eta2 = (X @ a2.T) @ p2
    lp2 = eta2 - np.log(np.exp(eta2 - eta2.max(1, keepdims=True)).sum(
        1, keepdims=True)) - eta2.max(1, keepdims=True)
    assert np.max(np.abs(lp - lp2)) < 1e-9
    assert np.max(np.abs(p2.sum(axis=1))) < 1e-12          # sum-zero
    gram = p2 @ p2.T
    assert np.max(np.abs(gram - np.eye(R))) < 1e-12        # orthonormal
    assert sv[0] >= sv[1] >= 0.0


def test_nominal_dof_formula():
    assert CL.nominal_dof_lowrank(6, 29, 2) == 2 * (6 + 28 - 2)
    assert CL.nominal_dof_lowrank(5, 29, 2) == 2 * (5 + 28 - 2)


# ===========================================================================
# 6.  weights enter the likelihood
# ===========================================================================
def test_weights_equal_duplication_in_the_fitted_likelihood():
    """Weighting an event by 2 must be identical to listing it twice."""
    g = toy_geom(B=3, C=7, S=2, K=1)
    rs = np.random.RandomState(7)
    n = 300
    X = CL.design(rs.uniform(-1, 1, n), rs.uniform(-0.4, 0.4, n),
                  np.zeros(n, int), "asmall")
    c = rs.randint(0, g["C"], n)
    fam = F.LowRank("A", "asmall", R=2)
    par = dict(alpha=rs.normal(scale=0.4, size=(2, 6)),
               phi=rs.normal(scale=0.4, size=(2, g["C"])), P=6, C=g["C"],
               R=2, kind="asmall")
    lp = fam.logp_rows(par, X, None)
    w = np.ones(n); w[:50] = 2.0
    ll_w = float(np.sum(w * lp[np.arange(n), c]))
    Xd = np.concatenate([X, X[:50]]); cd = np.concatenate([c, c[:50]])
    lpd = fam.logp_rows(par, Xd, None)
    ll_d = float(np.sum(lpd[np.arange(len(cd)), cd]))
    assert abs(ll_w - ll_d) < 1e-10


def test_slope_weights_are_normalised_and_monotone():
    g = toy_geom()
    ev = toy_events(g, n=500)
    w = None
    import run_candidates as RC
    for kind in ("native", "steep", "flat", "equalised"):
        w = RC.slope_weights(ev, g, kind)
        assert abs(w.sum() - ev["n"]) < 1e-8
    ws = RC.slope_weights(ev, g, "steep")
    o = np.argsort(ev["u"])
    assert np.all(np.diff(ws[o]) >= -1e-12)                # increasing in N


# ===========================================================================
# 7.  A-small recovers a toy row set drawn from its own family
# ===========================================================================
@pytest.mark.slow
def test_asmall_recovers_a_toy_rank2_row_set():
    pytest.importorskip("jax")
    C, P = 15, 6
    rs = np.random.RandomState(11)
    cc = (np.arange(C) - (C - 1) / 2.0) / C
    phi_t = np.stack([cc, cc ** 2 - (cc ** 2).mean()])
    phi_t /= np.linalg.norm(phi_t, axis=1, keepdims=True)
    alpha_t = np.array([[0.0, 22.0, 0.0, 1.5, 0.0, 0.0],
                        [-9.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    n = 60000
    u = rs.uniform(-1.0, 1.0, n); v = rs.uniform(-0.5, 0.5, n)
    K = np.zeros(n, int)
    X = CL.design(u, v, K, "asmall")
    eta = (X @ alpha_t.T) @ phi_t
    p = np.exp(eta - eta.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    c = (rs.uniform(size=(n, 1)) > cdf).sum(axis=1)
    fam = F.LowRank("A", "asmall", R=2)
    par = fam.fit(X, c, None, np.ones(n), C)
    lp = fam.logp_rows(par, X, None)
    ph = np.exp(lp)
    # the FITTED row probabilities must match the truth rows, not merely the
    # parameters (which are only identified up to the canonical gauge)
    assert np.max(np.abs(ph - p)) < 0.02
    kl = np.sum(np.where(p > 0, p * np.log(p / np.maximum(ph, 1e-300)), 0.0),
                axis=1)
    assert float(kl.mean()) < 2e-3


# ===========================================================================
# 8.  metric functions on known distributions
# ===========================================================================
def test_row_kl_and_deviance_against_closed_form():
    g = toy_geom(B=1, C=4, S=1, K=1)
    counts = np.zeros((1, 1, 1, 4)); counts[0, 0, 0] = [40, 30, 20, 10]
    rows = np.zeros((1, 1, 1, 4)); rows[0, 0, 0] = [0.25, 0.25, 0.25, 0.25]
    rec = CM.row_table(rows, counts, g, min_events=10)[0]
    ph = np.array([0.4, 0.3, 0.2, 0.1])
    kl = float(np.sum(ph * np.log(ph / 0.25)))
    assert abs(rec["kl"] - kl) < 1e-12
    assert abs(rec["deviance"] - 2 * 100 * kl) < 1e-10


def test_paired_diff_se_matches_the_closed_form():
    rs = np.random.RandomState(12)
    a = rs.normal(size=500); b = rs.normal(size=500)
    pd = CM.paired_diff(a, b)
    d = a - b
    assert abs(pd["mean"] - d.mean()) < 1e-14
    assert abs(pd["se"] - d.std(ddof=1) / np.sqrt(500)) < 1e-14


def test_pit_is_uniform_for_a_correct_model():
    g = toy_geom(B=1, C=20, S=1, K=1)
    rs = np.random.RandomState(13)
    p = rs.dirichlet(np.ones(20) * 3.0)
    rows = np.zeros((1, 1, 1, 20)); rows[0, 0, 0] = p
    n = 40000
    c = (rs.uniform(size=(n, 1)) > np.cumsum(p)[None, :]).sum(axis=1)
    ev = dict(b_i=np.zeros(n, int), s_i=np.zeros(n, int),
              K_i=np.zeros(n, int), c_i=c)
    h = CM.pit_hist(rows, ev, np.ones(n, bool), nbin=20)
    assert h["chi2"] < 3.0 * h["dof"]
    # and NON-uniform for a wrong model (power check, not just containment)
    q = np.roll(p, 4); q /= q.sum()
    rows2 = rows.copy(); rows2[0, 0, 0] = q
    h2 = CM.pit_hist(rows2, ev, np.ones(n, bool), nbin=20)
    assert h2["chi2"] > 10.0 * h["chi2"]


def test_boundary_table_counts_the_right_side():
    g = toy_geom(B=2, C=6, S=1, K=1)
    # bin 0 centre below 20.9, bin 1 centre above it
    beta = float(g["bcen"].mean())
    counts = np.zeros((2, 1, 1, 6)); counts[..., :] = 50.0
    rows = np.full((2, 1, 1, 6), 1.0 / 6.0)
    bt = CM.boundary_table(rows, counts, g, boundaries=(beta,),
                           min_events=100)
    recs = bt[f"{beta:.1f}"]["records"]
    assert len(recs) == 2
    up = float((g["ccen"] >= beta).sum()) / 6.0
    got = {r["b"]: r["p_mod"] for r in recs}
    assert abs(got[0] - up) < 1e-12
    assert abs(got[1] - (1.0 - up)) < 1e-12


def test_jeffreys_rows_exact_formula_and_fallback():
    counts = np.zeros((1, 1, 2, 4))
    counts[0, 0, 0] = [3, 0, 1, 0]
    p = F.jeffreys_rows(counts, 0.5,
                        fallback=np.full((1, 1, 2, 4), 0.25))
    assert abs(p[0, 0, 0, 0] - 3.5 / (4 + 2.0)) < 1e-14
    assert abs(p[0, 0, 0, 1] - 0.5 / (4 + 2.0)) < 1e-14
    assert abs(p[0, 0, 1, 0] - 0.25) < 1e-14              # empty row fallback
    assert np.max(np.abs(p.sum(axis=-1) - 1.0)) < 1e-14


def test_transfer_kl_is_zero_for_the_measured_operator_itself(tmp_path):
    rs = np.random.RandomState(14)
    S, K, C, B, kf = 2, 2, 5, 3, 4
    Mt = rs.dirichlet(np.ones(C), size=(S, K, B)).transpose(0, 1, 3, 2)
    Nm = rs.uniform(20, 60, size=(C, kf, S, B))
    p = tmp_path / "ops.npz"
    np.savez(p, M_true_sKcb=Mt, N_match_cksb=Nm,
             kz_to_K=np.array([0, 0, 1, 1]))
    rows = np.transpose(Mt, (3, 0, 1, 2))
    g = dict(B=B, C=C, S=S, K=K)
    t = CM.transfer_kl(rows, str(p), g, min_events=1)
    assert t["wmean_kl"] < 1e-12
    rows_bad = np.full_like(rows, 1.0 / C)
    t2 = CM.transfer_kl(rows_bad, str(p), g, min_events=1)
    assert t2["wmean_kl"] > 0.05


# ===========================================================================
# 9.  the section 5 criterion on CONSTRUCTED inputs
# ===========================================================================
def _draw(g, model_rows, n_per_row, rs):
    counts = np.zeros((g["B"], g["S"], g["K"], g["C"]))
    b_i, s_i, K_i, c_i = [], [], [], []
    for b in range(g["B"]):
        for s in range(g["S"]):
            for k in range(g["K"]):
                c = rs.choice(g["C"], size=n_per_row,
                              p=model_rows[b, s, k])
                np.add.at(counts[b, s, k], c, 1.0)
                b_i += [b] * n_per_row; s_i += [s] * n_per_row
                K_i += [k] * n_per_row; c_i += list(c)
    ev = dict(b_i=np.array(b_i), s_i=np.array(s_i), K_i=np.array(K_i),
              c_i=np.array(c_i), n=len(c_i))
    return ev, counts


def _c5_inputs(g, model_rows, n_per_row=400, seed=15):
    """TRAIN and HELD-OUT draws from the same truth.

    The R1d-raw reference is built on the TRAIN draw and scored on the
    HELD-OUT draw, exactly as the sealed 2-fold protocol does: scoring a
    saturated in-sample estimate against a held-out model would make the
    reference unbeatable by construction (it is the one thing the sealed
    criterion is NOT).
    """
    rs = np.random.RandomState(seed)
    nrows = g["B"] * g["S"] * g["K"]
    _, train_counts = _draw(g, model_rows, n_per_row, rs)
    ev, counts = _draw(g, model_rows, n_per_row, rs)
    rid = (ev["b_i"] * g["S"] + ev["s_i"]) * g["K"] + ev["K_i"]
    return ev, counts, rid, nrows, train_counts


def test_section5_passes_when_the_model_is_the_truth():
    g = toy_geom(B=2, C=9, S=2, K=1)
    rs = np.random.RandomState(16)
    rows = rs.dirichlet(np.ones(9) * 4.0, size=(2, 2, 1))
    ev, counts, rid, nr, tr = _c5_inputs(g, rows, n_per_row=800)
    mask = np.ones(ev["n"], bool)
    ll = CM.event_logp_from_rows(rows, ev, mask)
    ref = F.jeffreys_rows(tr, 0.5)
    llr = CM.event_logp_from_rows(ref, ev, mask)
    out = CM.criterion_section5(ll, llr, rid, nr, [(rows, counts)], g,
                                min_big=200, skew_tol=10.0)
    assert out["detail"]["i_fail"] is False
    assert out["detail"]["ii_fail"] is False


def test_section5_i_fires_on_a_concentrated_loglik_deficit():
    g = toy_geom(B=2, C=9, S=2, K=1)
    rs = np.random.RandomState(17)
    truth = rs.dirichlet(np.ones(9) * 4.0, size=(2, 2, 1))
    ev, counts, rid, nr, tr = _c5_inputs(g, truth, n_per_row=800)
    mask = np.ones(ev["n"], bool)
    bad = np.full_like(truth, 1.0 / 9.0)
    ll = CM.event_logp_from_rows(bad, ev, mask)
    ref = F.jeffreys_rows(tr, 0.5)
    llr = CM.event_logp_from_rows(ref, ev, mask)
    out = CM.criterion_section5(ll, llr, rid, nr, [(bad, counts)], g,
                               min_big=200, skew_tol=10.0)
    assert out["detail"]["i_fail"] is True
    assert out["detail"]["deficit_frac_big"] > 0.99
    assert out["fail"] is True


def test_section5_ii_fires_on_a_boundary_mass_error():
    g = toy_geom(B=2, C=9, S=2, K=1)
    rs = np.random.RandomState(18)
    truth = rs.dirichlet(np.ones(9) * 4.0, size=(2, 2, 1))
    ev, counts, rid, nr, tr = _c5_inputs(g, truth, n_per_row=2000)
    mask = np.ones(ev["n"], bool)
    shifted = np.roll(truth, 2, axis=-1)
    shifted /= shifted.sum(axis=-1, keepdims=True)
    ll = CM.event_logp_from_rows(shifted, ev, mask)
    llr = CM.event_logp_from_rows(F.jeffreys_rows(tr, 0.5), ev, mask)
    out = CM.criterion_section5(ll, llr, rid, nr, [(shifted, counts)], g,
                               min_big=200, skew_tol=10.0)
    assert out["detail"]["ii_fail"] is True


def test_section5_iii_fires_on_a_high_N_skew_residual():
    g = toy_geom(B=6, C=11, S=1, K=1)
    assert g["ntrue"][-2] >= 21.3                 # the top bins are b >= 21.3
    rs = np.random.RandomState(19)
    truth = rs.dirichlet(np.ones(11) * 3.0, size=(6, 1, 1))
    ev, counts, rid, nr, tr = _c5_inputs(g, truth, n_per_row=600)
    mask = np.ones(ev["n"], bool)
    ll = CM.event_logp_from_rows(truth, ev, mask)
    llr = CM.event_logp_from_rows(F.jeffreys_rows(tr, 0.5), ev, mask)
    ok = CM.criterion_section5(ll, llr, rid, nr, [(truth, counts)], g,
                               min_big=200, skew_tol=0.10)
    assert ok["detail"]["iii_fail"] is False
    # now a model whose HIGH-N rows are strongly skewed the wrong way
    bad = truth.copy()
    hi = g["ntrue"][:-1] >= 21.3 - 1e-9
    w = np.exp(3.0 * np.linspace(0, 1, g["C"]))
    bad[hi] = bad[hi] * w
    bad = bad / bad.sum(axis=-1, keepdims=True)
    llb = CM.event_logp_from_rows(bad, ev, mask)
    out = CM.criterion_section5(llb, llr, rid, nr, [(bad, counts)], g,
                                min_big=200, skew_tol=0.10)
    assert out["detail"]["iii_fail"] is True
    assert "(iii)" in " ".join(out["which"])


# ===========================================================================
# 10.  D: offset alignment round-trip, and the shrinkage limits
# ===========================================================================
def test_offset_alignment_round_trip_is_exact():
    g = toy_geom(B=4, C=9, S=2, K=1)
    rs = np.random.RandomState(20)
    counts = rs.poisson(5.0, size=(4, 2, 1, 9)).astype(float)
    off, c0 = F._offset_align(counts, g)
    back = F._offset_to_c(off, c0, 9)
    assert np.max(np.abs(back - counts)) == 0.0


def test_D_limits_are_the_empirical_row_and_the_global_row():
    g = toy_geom(B=3, C=7, S=2, K=1)
    rs = np.random.RandomState(21)
    counts = rs.poisson(30.0, size=(3, 2, 1, 7)).astype(float)
    p0, w0 = F.fit_D(counts, g, 0.0, 0.0, 0.0, eps=1e-12)
    emp = counts / counts.sum(axis=-1, keepdims=True)
    assert np.max(np.abs(p0 - emp)) < 1e-9
    p1, w1 = F.fit_D(counts, g, 0.0, 0.0, 1e9, eps=1e-12)
    glob = counts.sum(axis=(1, 2), keepdims=True)
    glob = glob / glob.sum(axis=-1, keepdims=True)
    assert np.max(np.abs(p1 - np.broadcast_to(glob, p1.shape))) < 1e-6
    assert np.max(np.abs(p0.sum(axis=-1) - 1.0)) < 1e-14
    assert float(w1.max()) < 1e-4                       # fully shrunk


# ===========================================================================
# 11.  the delivered Mg tensor obeys the count-conservation convention
# ===========================================================================
def test_build_mg_rows_carry_phi_not_the_packs_phi_ref():
    """POST-SEAL requirement: Mg rows must sum to the MEASURED in-grid had
    mass phi(b,s,K), and phi_ref_pack_gathered must be returned separately."""
    pack = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
            "packs/scanpack_2lpt0_b300.npz")
    ops = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
           "support/empirical_ops_2lpt0_A0.npz")
    if not (os.path.exists(pack) and os.path.exists(ops)):
        pytest.skip("scratch packs not available")
    import run_candidates as RC
    geom = CL.load_geom(ops, pack)
    rs = np.random.RandomState(22)
    rows = rs.dirichlet(np.ones(geom["C"]),
                        size=(geom["B"], geom["S"], geom["K"]))
    phi = rs.uniform(0.1, 1.0, size=(geom["B"], geom["S"], geom["K"]))
    Mg, phi_ref_g = RC.build_mg(rows, "2lpt0", geom, phi)
    kz = np.asarray(np.load(pack)["kz_to_K"], int)
    assert Mg.shape == (geom["S"], len(kz), geom["C"], geom["B"])
    for s, kf in ((3, 7), (5, 0), (2, 14)):
        got = Mg[s, kf].sum(axis=0)                    # over c -> (B,)
        assert np.max(np.abs(got - phi[:, s, kz[kf]])) < 1e-12
    # and the shape is preserved exactly
    assert np.max(np.abs(Mg[3, 7][:, 5] / phi[5, 3, kz[7]] -
                         rows[5, 3, kz[7]])) < 1e-12
    assert phi_ref_g.shape == (geom["B"], geom["S"], geom["K"])


# ===========================================================================
# 11b.  phi -- the row HAD MASS (post-seal addition)
# ===========================================================================
def test_phi_percell_is_the_jeffreys_conditional_count_ratio():
    import phimass as PH
    k = np.array([[[3.0]], [[0.0]]])
    n = np.array([[[10.0]], [[4.0]]])
    p = PH.phi_percell(k, n)
    assert abs(p[0, 0, 0] - 3.5 / 11.0) < 1e-14
    assert abs(p[1, 0, 0] - 0.5 / 5.0) < 1e-14


def test_phi_counts_share_one_support():
    """The in-grid numerator must never exceed the all-detected denominator:
    the counting check that catches the recurring one-sided-support bug."""
    import phimass as PH
    ops = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
           "support/empirical_ops_2lpt0_A0.npz")
    pack = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
            "packs/scanpack_2lpt0_b300.npz")
    if not (os.path.exists(pack) and os.path.exists(ops)):
        pytest.skip("scratch packs not available")
    geom = CL.load_geom(ops, pack)
    k, n = PH.load_phi_counts(ops, geom)
    assert k.shape == (geom["B"], geom["S"], geom["K"])
    assert float((k - n).max()) <= 0.0
    z = np.load(ops, allow_pickle=True)
    assert abs(k.sum() - np.asarray(z["N_det_bks_true_z"], float).sum()) < 1e-9
    assert abs(n.sum() -
               np.asarray(z["N_det_all_bks_true_z"], float).sum()) < 1e-9


def test_phi_smooth_recovers_a_logistic_truth():
    import phimass as PH
    ops = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
           "support/empirical_ops_2lpt0_A0.npz")
    pack = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
            "packs/scanpack_2lpt0_b300.npz")
    if not (os.path.exists(pack) and os.path.exists(ops)):
        pytest.skip("scratch packs not available")
    geom = CL.load_geom(ops, pack)
    X = PH.phi_design(geom)
    t = np.array([1.5, 2.0, -0.3, 0.4, 0.1, -0.2])
    p = 1.0 / (1.0 + np.exp(-(X @ t)))
    n = np.full(p.shape, 4000.0)
    rs = np.random.RandomState(30)
    k = rs.binomial(4000, p).astype(float)
    ph, meta = PH.fit_phi_smooth(k, n, geom)
    assert meta["nominal_dof"] == 6
    assert np.max(np.abs(ph - p)) < 0.02
    assert np.max(np.abs(np.array(meta["coef"]) - t)) < 0.08


def test_phi_cv_split_conserves_the_trials():
    import phimass as PH
    g = toy_geom(B=3, C=5, S=2, K=1)
    rs = np.random.RandomState(31)
    n = rs.randint(50, 400, size=(3, 2, 1)).astype(float)
    k = np.floor(n * rs.uniform(0.2, 0.9, size=n.shape))
    out = PH.phi_cv(k, n, g, seed=7)
    assert set(out["folds"]) == {"A->B", "B->A"}
    for est in ("percell", "smooth"):
        assert np.isfinite(out["heldout_ll_per_trial"][est])
        assert out["heldout_ll_per_trial"][est] < 0.0
    # a deliberately wrong phi must score worse than the measured one
    good = PH.binomial_ll_per_trial(PH.phi_percell(k, n), k, n)
    bad = PH.binomial_ll_per_trial(np.full_like(n, 0.5), k, n)
    assert good > bad


# ===========================================================================
# 12.  the flat quadrature really is flat and interior
# ===========================================================================
def test_quadrature_grid_is_flat_and_strictly_inside_the_bin():
    pack = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
            "packs/scanpack_2lpt0_b300.npz")
    ops = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
           "support/empirical_ops_2lpt0_A0.npz")
    if not (os.path.exists(pack) and os.path.exists(ops)):
        pytest.skip("scratch packs not available")
    geom = CL.load_geom(ops, pack)
    un, vn = CL.quad_grid(geom)
    assert un.shape == (geom["B"], CL.N_QUAD_U)
    assert vn.shape == (geom["S"], CL.N_QUAD_V)
    for b in range(geom["B"]):
        N = un[b] + CL.N_REF_U
        assert N.min() > geom["ntrue"][b]
        assert N.max() < geom["ntrue"][b + 1]
        d = np.diff(N)
        assert float(d.max() - d.min()) < 1e-12          # equally spaced
    assert np.all(np.isfinite(vn))


def test_design_columns_are_the_declared_ones():
    u = np.array([0.5, -1.0]); v = np.array([0.2, -0.3])
    K = np.array([1, 2])
    Xa = CL.design(u, v, K, "asmall")
    assert Xa.shape == (2, 6)
    assert np.max(np.abs(Xa[:, 0] - 1.0)) == 0.0
    assert np.max(np.abs(Xa[:, 1] - u)) == 0.0
    assert np.max(np.abs(Xa[:, 2] - u * u)) == 0.0
    assert np.max(np.abs(Xa[:, 3] - v)) == 0.0
    assert Xa[0, 4] == 1.0 and Xa[0, 5] == 0.0
    assert Xa[1, 4] == 0.0 and Xa[1, 5] == 1.0
    Xe = CL.design(u, v, K, "e")
    assert Xe.shape == (2, 5)
    assert np.max(np.abs(Xe[:, 1] - u)) == 0.0
    assert np.max(np.abs(Xe[:, 2] - v)) == 0.0


def test_split_normal_cdf_is_a_proper_cdf_and_reduces_to_the_normal():
    """The mixture's skewed kernel: monotone, 0->1, continuous at the split,
    and exactly Gaussian when the single skew parameter is zero."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from scipy.stats import norm
    x = np.linspace(-8.0, 8.0, 2001)
    m, w, ka = 0.3, 0.7, 0.9
    wm, wp = w * np.exp(-0.5 * ka), w * np.exp(0.5 * ka)
    Fc = np.asarray(F.Mixture._split_normal_cdf(jnp.asarray(x), m, wm, wp),
                    float)
    assert np.min(np.diff(Fc)) >= -1e-14
    assert Fc[0] < 1e-8 and abs(Fc[-1] - 1.0) < 1e-8
    i = int(np.searchsorted(x, m))
    assert abs(Fc[i] - Fc[i - 1]) < 5e-3                # continuous at m
    F0 = np.asarray(F.Mixture._split_normal_cdf(jnp.asarray(x), m, w, w),
                    float)
    assert np.max(np.abs(F0 - norm.cdf((x - m) / w))) < 1e-12
    # the density is right-skewed for kappa > 0
    p = np.diff(Fc); c = 0.5 * (x[1:] + x[:-1])
    mu = float(np.sum(p * c) / p.sum())
    sd = np.sqrt(float(np.sum(p * (c - mu) ** 2) / p.sum()))
    sk = float(np.sum(p * (c - mu) ** 3) / p.sum()) / sd ** 3
    assert sk > 0.2
