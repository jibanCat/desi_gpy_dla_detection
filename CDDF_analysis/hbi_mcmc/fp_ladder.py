"""fp_ladder.py — the nested, loa-0-anchored FP observation-model ladder (M0–M5).

CLASSIFICATION: SCIENCE-CANDIDATE models evaluated by VALIDATION-ONLY machinery
(PI instruction 2026-09-12; sealed predeclaration
notes/governance/FP_REGULARIZATION_MODEL_LADDER_PREDECLARATION.md, sha256
3112022a0f673ec83f672b825f98e90a4abf678dd2f1f951ab5475a5c0f2bb5a).

Nothing in the frozen production files is modified. ``model_cc_ladder`` copies
``cc_posterior_validation.model_cc`` VERBATIM for the population prior, the
completeness offsets, the count-conserving fold and the masked Poisson
likelihood, and replaces ONLY the FP block:

  * FP intensity lambda[c,s] (per unit loa-0 exposure) is defined on the LIVE
    cells only (strata with dX > 0, i.e. S/N > 2: 29 x 6 = 174 cells) and is
    exactly 0 on the 58 dead cells (S/N < 2: no path, no counts, no FP
    allocation, no loa-0 events — all by construction of the strict SNR > 2
    cuts in build_loa0_fp_product.py and extract_pack.py). No softmax, no
    simplex, hence no constant-shift null direction and no dead-cell leak.
  * the loa-0 calibration enters ONCE, as a Poisson LIKELIHOOD on lambda:
        fp_counts[c,s] ~ Poisson(fp_ell_eff * lambda[c,s])   over live cells,
    with generic weak priors on the low-dimensional shape parameters. Zero
    loa-0 counts therefore act as calibrated upper limits.
  * Lambda = sum_live lambda is DERIVED (deterministic "fp_lam_total").

Ladder (predeclaration §4; x_c = Nhat-bin centre - 20.0 dex; j = s - 2 the live
stratum index, reference j = 0; m_cs = the loa-0 Perks log-share centre of
record, used only as a FIXED template in M0/M1):

    M0  log lam = l0 + m_cs                          t == 0
    M1  log lam = l0 + m_cs                          t_K ~ N(0, t_sd)
    M2  log lam = l0 + b1 x_c + h_j                  t_K
    M3  M2 + b2 x_c^2
    M4  M3 + r_c,  r_c = cumsum(delta), delta_c ~ N(0, tau_N)  (28 increments)
    M5  M4 + tau_I * eps_cs on live cells            (NOT run without PI GO)

Priors (fixed by the predeclaration, not tuned): l0 ~ N(log(89/(ell_eff*174)), 3);
b1, b2 ~ N(0, 5); h_j ~ N(0, 3) (h_0 == 0); t_K ~ N(0, t_sd = 1.0);
tau_N, tau_I ~ HalfNormal(0.5). Hierarchical terms are sampled non-centred.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

LADDER_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")

# nominal FP degrees of freedom per model (predeclaration §4)
NOMINAL_FP_DOF = {"M0": 1, "M1": 4, "M2": 10, "M3": 11, "M4": 40, "M5": 215, "ORACLE": 0}


def live_mask(consts):
    """(S,) boolean: strata with any path length. Dead strata have dX == 0 in
    every fine-z bin (validator guarantees zero counts and zero fp_E there)."""
    return np.asarray(consts.dX, float).sum(axis=0) > 0


def perks_log_share(fp_counts, live, a0=None):
    """The loa-0 Perks-smoothed log-share centre of record, restricted to live
    cells: m_cs = log((n_cs + a0) / (N + K a0)) with a0 = 1/K, K = number of
    LIVE cells. Off-live entries are set to 0 and never used."""
    fpc = np.asarray(fp_counts, float)
    C, S = fpc.shape
    K = int(C * live.sum())
    a0 = (1.0 / K) if a0 is None else float(a0)
    n_fp = float(fpc[:, live].sum())
    m = np.zeros((C, S), float)
    m[:, live] = np.log((fpc[:, live] + a0) / (n_fp + K * a0))
    return m


def _fp_block(consts, fp_counts, ladder, *, t_sd, tau_scale, calib_weight):
    """Sample the FP block; return (lam_fp (C,S) with zeros off-live, t (KK,))."""
    C, S, KK = consts.n_c, consts.n_s, consts.n_kk
    live = live_mask(consts)                              # (S,)
    live_j = jnp.asarray(live, dtype=jnp.float32)
    n_live_s = int(live.sum())
    fpc = np.asarray(fp_counts, float)
    ell = float(consts.fp_ell_eff)
    nhat = np.asarray(consts.nhat_edges, float)
    x_c = jnp.asarray(0.5 * (nhat[:-1] + nhat[1:]) - 20.0)  # (C,)

    l0_centre = float(np.log(max(fpc[:, live].sum(), 1.0) / (ell * C * n_live_s)))
    l0 = numpyro.sample("fp_l0", dist.Normal(l0_centre, 3.0))

    if ladder in ("M0", "M1"):
        m = jnp.asarray(perks_log_share(fpc, live))       # (C,S), 0 off-live
        log_lam = l0 + m
    else:
        b1 = numpyro.sample("fp_b1", dist.Normal(0.0, 5.0))
        h_free = numpyro.sample("fp_h", dist.Normal(0.0, 3.0)
                                .expand([n_live_s - 1]).to_event(1))
        # stratum effects on the live strata, reference (first live) == 0;
        # placed back onto the full S axis (zeros on dead strata; unused there)
        h_live = jnp.concatenate([jnp.zeros(1), h_free])            # (n_live,)
        live_idx = np.where(live)[0]
        h_full = jnp.zeros(S).at[live_idx].set(h_live)              # (S,)
        log_lam = l0 + b1 * x_c[:, None] + h_full[None, :]
        if ladder in ("M3", "M4", "M5"):
            b2 = numpyro.sample("fp_b2", dist.Normal(0.0, 5.0))
            log_lam = log_lam + b2 * (x_c ** 2)[:, None]
        if ladder in ("M4", "M5"):
            tau_N = numpyro.sample("fp_tau_N", dist.HalfNormal(tau_scale))
            dz = numpyro.sample("fp_delta_z", dist.Normal(0.0, 1.0)
                                .expand([C - 1]).to_event(1))
            r_c = jnp.concatenate([jnp.zeros(1), tau_N * jnp.cumsum(dz)])  # (C,)
            numpyro.deterministic("fp_r_c", r_c)
            log_lam = log_lam + r_c[:, None]
        if ladder == "M5":
            tau_I = numpyro.sample("fp_tau_I", dist.HalfNormal(tau_scale))
            # interaction innovations on the LIVE cells only (C x n_live), placed
            # back onto the full S axis (zeros on dead strata; inert there)
            eps_live = numpyro.sample("fp_eps_z", dist.Normal(0.0, 1.0)
                                      .expand([C, n_live_s]).to_event(2))
            eps = jnp.zeros((C, S)).at[:, live_idx].set(eps_live)
            log_lam = log_lam + tau_I * eps
    lam_fp = numpyro.deterministic("lam_fp", jnp.exp(log_lam) * live_j[None, :])
    numpyro.deterministic("fp_lam_total", lam_fp.sum())

    # the loa-0 calibration LIKELIHOOD, live cells only, entered once
    calib_mask = jnp.broadcast_to(live_j[None, :] > 0, (C, S))
    with numpyro.handlers.mask(mask=calib_mask):
        with numpyro.handlers.scale(scale=float(calib_weight)):
            numpyro.sample("fp_counts",
                           dist.Poisson(jnp.clip(ell * lam_fp, 1e-300, None)),
                           obs=jnp.asarray(fpc))

    if ladder == "M0":
        t = numpyro.deterministic("t", jnp.zeros(KK))
    else:
        t = numpyro.sample("t", dist.Normal(0.0, float(t_sd)).expand([KK]).to_event(1))
    return lam_fp, t


def model_cc_ladder(consts, Mg, counts=None, fp_counts=None, *, ladder="M2",
                    t_sd=1.0, tau_scale=0.5, calib_weight=1.0,
                    mu_fp_fixed=None,
                    sigma_N_scale=0.5, sigma_z_scale=0.5,
                    level_scale=4.0, slope_scale=2.0):
    """model_cc with the FP block replaced by ladder member ``ladder``.

    Population prior, completeness offsets, fold and likelihood are copied
    verbatim from cc_posterior_validation.model_cc (tests/test_fp_ladder.py
    asserts fold identity at equal (lam_fp, t) to 1e-12)."""
    if ladder not in LADDER_MODELS and ladder != "ORACLE":
        raise ValueError(f"unknown ladder member {ladder!r}")
    if ladder == "ORACLE" and mu_fp_fixed is None:
        raise ValueError("ORACLE diagnostic needs mu_fp_fixed (C,Kf,S) = the mock FP truth census")
    B, Kf = consts.n_b, consts.n_k
    # ---- population prior (VERBATIM model_cc) ------------------------------
    sigma_N = numpyro.sample("sigma_N", dist.HalfNormal(sigma_N_scale))
    sigma_z = numpyro.sample("sigma_z", dist.HalfNormal(sigma_z_scale))
    level = numpyro.sample("theta_level", dist.Normal(0.0, level_scale))
    slope = numpyro.sample("theta_slope", dist.Normal(0.0, slope_scale))
    eps_N = numpyro.sample(
        "eps_N", dist.Normal(0.0, 1.0).expand([max(B - 2, 0)]).to_event(1))
    eps_z = numpyro.sample(
        "eps_z", dist.Normal(0.0, 1.0).expand([B, max(Kf - 1, 0)]).to_event(2))
    b_idx = jnp.arange(B) - 0.5 * (B - 1)
    curv = jnp.cumsum(jnp.cumsum(jnp.concatenate([jnp.zeros(2), eps_N])))[:B]
    theta_col0 = level + slope * b_idx + sigma_N * curv
    theta = theta_col0[:, None] + jnp.concatenate(
        [jnp.zeros((B, 1)), sigma_z * jnp.cumsum(eps_z, axis=1)], axis=1)
    theta = numpyro.deterministic("theta_pop", theta)
    numpyro.deterministic("f", jnp.exp(theta))

    psi_c = numpyro.sample(
        "psi_c", dist.Normal(0.0, consts.sigma_hat).to_event(2))

    # ---- the FP block (the ONLY change) ------------------------------------
    if ladder == "ORACLE":
        # DIAGNOSTIC ONLY (outside the sealed ladder; uses the mock's own FP truth):
        # mu_FP pinned per (c,k,s) to the hostless@17.2 census; no FP parameter,
        # no calibration term. Answers "does a TRUE FP field close the gate?"
        C, S, KK = consts.n_c, consts.n_s, consts.n_kk
        lam_fp = numpyro.deterministic("lam_fp", jnp.zeros((C, S)))
        numpyro.deterministic("fp_lam_total", jnp.asarray(0.0))
        t = numpyro.deterministic("t", jnp.zeros(KK))
    else:
        lam_fp, t = _fp_block(consts, fp_counts, ladder, t_sd=t_sd,
                              tau_scale=tau_scale, calib_weight=calib_weight)

    # ---- fold + likelihood (VERBATIM model_cc) -----------------------------
    Cc = jax.nn.sigmoid(consts.eta_hat + psi_c)[:, consts.b_to_cell]  # (S,B)
    f = jnp.exp(theta)                                                # (B,Kf)
    w = consts.g_bk * f * consts.dN_b[:, None]                        # (B,Kf)
    tp = jnp.einsum("skcb,sb,bk->cks", Mg, Cc, w) * consts.dX[None, :, :]
    if ladder == "ORACLE":
        fp = jnp.asarray(np.asarray(mu_fp_fixed, float))
    else:
        fp = (consts.fp_w * consts.fp_ell_eff
              * (1.0 - consts.fp_eta_c)[:, None, None]
              * jnp.exp(t[consts.kz_to_K])[None, :, None]
              * lam_fp[:, None, :] * consts.fp_E[None, :, :])
    mu = tp + fp
    obs_mask = jnp.broadcast_to(jnp.asarray(consts.dX > 0)[None, :, :],
                                mu.shape)
    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("counts", dist.Poisson(jnp.clip(mu, 1e-300, None)),
                       obs=counts)


# sampled FP-block site names per model (for by-chain retention + whitening)
FP_SITES = {
    "ORACLE": (),
    "M0": ("fp_l0",),
    "M1": ("fp_l0", "t"),
    "M2": ("fp_l0", "fp_b1", "fp_h", "t"),
    "M3": ("fp_l0", "fp_b1", "fp_h", "fp_b2", "t"),
    "M4": ("fp_l0", "fp_b1", "fp_h", "fp_b2", "fp_tau_N", "fp_delta_z", "t"),
    "M5": ("fp_l0", "fp_b1", "fp_h", "fp_b2", "fp_tau_N", "fp_delta_z",
           "fp_tau_I", "fp_eps_z", "t"),
}


def fp_prior_moments(consts, fp_counts, ladder, *, t_sd=1.0, tau_scale=0.5):
    """Prior mean/sd of every SCALAR FP-block coordinate, in sampling order,
    for prior-whitening (u = (x - m)/s). Hierarchical non-centred innovations
    are N(0,1); scales are HalfNormal (mean/sd given for reporting only)."""
    C, S = consts.n_c, consts.n_s
    live = live_mask(consts)
    n_live_s = int(live.sum())
    fpc = np.asarray(fp_counts, float)
    ell = float(consts.fp_ell_eff)
    l0c = float(np.log(max(fpc[:, live].sum(), 1.0) / (ell * C * n_live_s)))
    hn_mean = float(tau_scale * np.sqrt(2 / np.pi))
    hn_sd = float(tau_scale * np.sqrt(1 - 2 / np.pi))
    names, mean, sd = [], [], []

    def add(n, m, s):
        names.append(n); mean.append(m); sd.append(s)
    add("fp_l0", l0c, 3.0)
    if ladder != "M0":
        pass
    if ladder in ("M2", "M3", "M4", "M5"):
        add("fp_b1", 0.0, 5.0)
        for j in range(n_live_s - 1):
            add(f"fp_h[{j}]", 0.0, 3.0)
    if ladder in ("M3", "M4", "M5"):
        add("fp_b2", 0.0, 5.0)
    if ladder in ("M4", "M5"):
        add("fp_tau_N", hn_mean, hn_sd)
        for c in range(C - 1):
            add(f"fp_delta_z[{c}]", 0.0, 1.0)
    if ladder == "M5":
        add("fp_tau_I", hn_mean, hn_sd)
        for c in range(C):
            for j in range(n_live_s):
                add(f"fp_eps_z[{c},{j}]", 0.0, 1.0)
    if ladder != "M0":
        for k in range(consts.n_kk):
            add(f"t[{k}]", 0.0, float(t_sd))
    return names, np.asarray(mean), np.asarray(sd)
