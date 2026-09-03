"""count_conserving_fold.py — count-conserving evaluation of the Model A fold
(PI ruling 2026-08-17, adoption item 1).

The production fold applies the kernel K as skew-normal mass on the observed
grid [19.5, 22.4]; mass outside the grid is silently dropped, so the IN-GRID
FRACTION phi(N, cell) = sum_c K[c<-b] acts as a hidden detection/counting
probability attached to the kernel REPRESENTATION. Under the ratified joint
C/K/FP contract that probability belongs to C, not K. This module makes the
split explicit:

    K~[c<-b]  = K[c<-b] / phi(b, cell)         (unit mass on the counting
                                                domain: pure redistribution)
    C_op(b,s) = C_molly(b,s) * phi_ref(b,cell) (detection-AND-counting
                                                probability of the DEPLOYED
                                                operator; phi_ref frozen from
                                                the deployed kernel, since
                                                C_molly was calibrated jointly
                                                with it)
    mu        = dX * K~ * C_op * g * f * dN  +  FP        (unchanged FP arm)

With the frozen surfaces this is BYTE-EQUIVALENT to the production fold
(K = phi_ref * K~ and C_op/phi_ref = C_molly). For any OTHER admissible
representation the count-conservation rule is: renormalize to unit mass and
keep phi_ref frozen — a kernel change then redistributes detected objects in
measured N but cannot create or remove counting probability.

Also supports a latent lower-N support floor (bins with lower edge < floor
excluded from the truth sum) for the low-N convergence study (ruling item 4).

Pure numpy/scipy; adapted from the committed ``fold_mu_reference`` oracle
(same clamp/ramp/moment->skew-normal semantics), vectorized over (b, c).
DIAGNOSTIC/ADOPTION-AUDIT module: production fold_mu is untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, owens_t, expit

__all__ = ["surface_masses", "cc_fold_cmarginal", "phi_from_surfaces",
           "cc_fold_adopted", "outcome_probabilities"]

_SKEW_MAX = 0.5 * (4.0 - np.pi) * (np.sqrt(2.0 / np.pi) ** 3) / \
    (1.0 - 2.0 / np.pi) ** 1.5


def _m2sn_vec(mean, sd, skew):
    """Vector moment -> skew-normal (xi, omega, alpha); committed semantics."""
    bb = np.sqrt(2.0 / np.pi)
    s_ = np.clip(skew, -0.995 * _SKEW_MAX, 0.995 * _SKEW_MAX)
    sd = np.maximum(sd, 1e-9)
    cc = 0.5 * (4.0 - np.pi)
    r = (np.abs(s_) / cc) ** (2.0 / 3.0)
    gg = r / (1.0 + r)
    delta = np.clip(np.sign(s_) * np.sqrt(gg) / bb, -0.999, 0.999)
    small = np.abs(s_) < 1e-9
    delta = np.where(small, 0.0, delta)
    al = delta / np.sqrt(np.maximum(1.0 - delta * delta, 1e-12))
    om = sd / np.sqrt(np.maximum(1.0 - (bb * delta) ** 2, 1e-12))
    xi = mean - om * bb * delta
    return xi, om, al


def _sn_cdf(x, xi, om, al):
    z = (x - xi) / om
    return ndtr(z) - 2.0 * owens_t(z, al)


def surface_masses(pack, mu_coef, sig_coef, skew_coef, fit_rng, nhat_edges):
    """Per-(sr, zr) kernel mass matrices on the observed grid.

    mu_coef/sig_coef/skew_coef: (SR, ZR, D) moment polynomials in
    u = N - resp_N_ref; fit_rng: (SR, ZR, 2) covariate clamp range.
    Returns masses (SR, ZR, C, B) and the in-grid fraction phi (SR, ZR, B).
    """
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    n_ref = float(pack.resp_N_ref)
    SR, ZR, D = np.asarray(mu_coef).shape
    C_n = len(nhat_edges) - 1
    B_n = len(Nc)
    ramp_c, ramp_w = [float(v) for v in np.asarray(pack.resp_skew_ramp, float)]
    sig_floor = float(pack.resp_sig_floor)
    masses = np.zeros((SR, ZR, C_n, B_n))
    phi = np.zeros((SR, ZR, B_n))
    for sr in range(SR):
        for zr in range(ZR):
            Ncl = np.clip(Nc, fit_rng[sr, zr, 0], fit_rng[sr, zr, 1])
            u = Ncl - n_ref
            up = u[:, None] ** np.arange(D)[None, :]
            mean = Nc + up @ np.asarray(mu_coef, float)[sr, zr]
            sd = np.maximum(up @ np.asarray(sig_coef, float)[sr, zr],
                            sig_floor)
            ramp = np.clip((Nc - ramp_c) / ramp_w, 0.0, 1.0)
            skw = np.clip(up @ np.asarray(skew_coef, float)[sr, zr],
                          -0.995 * _SKEW_MAX, 0.995 * _SKEW_MAX) * (1 - ramp)
            xi, om, al = _m2sn_vec(mean, sd, skw)
            cdf = np.stack([_sn_cdf(e, xi, om, al) for e in nhat_edges])
            m = np.clip(np.diff(cdf, axis=0), 0.0, 1.0)       # (C, B)
            masses[sr, zr] = m
            phi[sr, zr] = m.sum(axis=0)
    return masses, phi


def phi_from_surfaces(pack, nhat_edges=None):
    """In-grid fraction of the pack's own (deployed) surfaces."""
    ne = (np.asarray(pack.nhat_edges, float) if nhat_edges is None
          else np.asarray(nhat_edges, float))
    _, phi = surface_masses(pack, pack.resp_mu_coef, pack.resp_sig_coef,
                            pack.resp_skew_coef,
                            np.asarray(pack.resp_N_fit_range, float), ne)
    return phi


def cc_fold_cmarginal(pack, theta_pop, lam_fp, *, mu_coef=None, sig_coef=None,
                      skew_coef=None, fit_rng=None, renormalize=False,
                      phi_ref=None, n_lat_floor=None, masses_override=None,
                      return_contrib=False):
    """c-marginal expected counts under the (optionally count-conserving) fold.

    Defaults (all surface args None, renormalize False) reproduce the
    production fold with the pack's frozen surfaces. With
    ``renormalize=True`` the kernel is normalized to unit in-grid mass and
    multiplied by ``phi_ref`` (SR, ZR, B); pass the DEPLOYED kernel's phi
    (``phi_from_surfaces``) to enforce the count-conservation rule.
    ``n_lat_floor``: exclude truth bins whose LOWER EDGE is below the floor.
    Returns (mu_c, parts) with parts = dict(tp=..., fp=...) c-marginals.
    """
    nhat = np.asarray(pack.nhat_edges, float)
    ntrue = np.asarray(pack.ntrue_edges, float)
    zf = np.asarray(pack.zf_edges, float)
    zc = np.asarray(pack.zc_edges, float)
    snr = np.asarray(pack.snr_edges, float)
    C_n, B_n, K_n, S_n = (len(nhat) - 1, len(ntrue) - 1, len(zf) - 1,
                          len(snr) - 1)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    dN = np.diff(ntrue)

    mu_c = pack.resp_mu_coef if mu_coef is None else mu_coef
    sig_c = pack.resp_sig_coef if sig_coef is None else sig_coef
    skw_c = pack.resp_skew_coef if skew_coef is None else skew_coef
    rng = (np.asarray(pack.resp_N_fit_range, float) if fit_rng is None
           else np.asarray(fit_rng, float))

    if masses_override is not None:
        masses = np.asarray(masses_override, float)
        phi = masses.sum(axis=2)
    else:
        masses, phi = surface_masses(pack, mu_c, sig_c, skw_c, rng, nhat)
    if renormalize:
        masses = masses / np.maximum(phi, 1e-12)[:, :, None, :]
        if phi_ref is not None:
            masses = masses * np.asarray(phi_ref, float)[:, :, None, :]

    # index maps (fold_mu_reference logic)
    zf_centers = 0.5 * (zf[:-1] + zf[1:])
    kz2K = np.minimum(np.searchsorted(zc, zf_centers, side="right") - 1,
                      len(zc) - 2)
    rse = np.asarray(pack.resp_snr_edges, float)
    s2sr = np.searchsorted(rse, snr[:-1] + 1e-9, side="right") - 1
    oob = (s2sr < 0) | (s2sr >= np.asarray(mu_c).shape[0])
    if np.any(oob):
        if np.any(np.asarray(pack.dX, float)[:, oob] > 0):
            raise ValueError("cc_fold: sub-range SNR strata carry exposure")
        s2sr = np.clip(s2sr, 0, np.asarray(mu_c).shape[0] - 1)
    rze = np.asarray(pack.resp_z_edges, float)
    zc_centers = 0.5 * (zc[:-1] + zc[1:])
    K2zr = np.searchsorted(rze, zc_centers, side="right") - 1
    me = np.asarray(pack.molly_nhi_edges, float)
    b2cell = np.clip(np.searchsorted(me, Nc, side="right") - 1, 0,
                     len(me) - 2)

    nd = np.asarray(pack.molly_n_det, float)
    nt = np.asarray(pack.molly_n_tot, float)
    C_cells = expit(np.log(nd + 0.5) - np.log(nt - nd + 0.5))   # (S, M)
    g = np.asarray(pack.g_grid, float)
    dX = np.asarray(pack.dX, float)
    E = np.asarray(pack.fp_E_alloc, float)
    w = float(pack.fp_w_sightline_ratio)
    ell = float(pack.fp_ell_eff)
    eta_c = np.asarray(pack.fp_eta_c, float)
    lam_fp = np.asarray(lam_fp, float)

    f = np.exp(np.asarray(theta_pop, float))                    # (B, Kf)
    if n_lat_floor is not None:
        f = f * (ntrue[:-1][:, None] >= float(n_lat_floor) - 1e-9)

    tp = np.zeros(C_n)
    fp = np.zeros(C_n)
    contrib = np.zeros((C_n, B_n)) if return_contrib else None
    for s in range(S_n):
        sr = int(s2sr[s])
        Cb = C_cells[s, b2cell]                                # (B,)
        for k in range(K_n):
            Kc = int(kz2K[k])
            zr = int(K2zr[Kc])
            weight = Cb * g[b2cell, k] * f[:, k] * dN          # (B,)
            tp += dX[k, s] * (masses[sr, zr] @ weight)
            if return_contrib:
                contrib += dX[k, s] * (masses[sr, zr] * weight[None, :])
            fp += w * ell * (1.0 - eta_c) * lam_fp[:, s] * E[k, s]
    parts = dict(tp=tp, fp=fp)
    if return_contrib:
        parts["contrib_cb"] = contrib
    return tp + fp, parts


def outcome_probabilities(pack):
    """Per-system outcome probabilities of the DEPLOYED adopted observation model, P[c, b, k, s] = P(detected AND x̂ in observed bin c |
    truth in latent bin b, fine z bin k, S/N row s) = masses[sr(s), zr(k), c, b] · C_cells[s, cell(b)] · g[cell(b), k], using the same
    index maps as the fold, so that  tp[c] = Σ_{b,k,s} dX[k,s] f[b,k] dN[b] P[c,b,k,s]  reproduces cc_fold_adopted exactly (checked
    by the caller). Used by the HZ2 fiducial GENERATIVE closure to simulate catalogues from the model itself (gate Amendment 1,
    2026-09-03). Requires the adopted stamp group; uses adopted_masses_override when present, else the adopted surfaces renormalized
    to the stored phi_ref (identical to cc_fold_adopted)."""
    nhat = np.asarray(pack.nhat_edges, float); ntrue = np.asarray(pack.ntrue_edges, float)
    zf = np.asarray(pack.zf_edges, float); zc = np.asarray(pack.zc_edges, float); snr = np.asarray(pack.snr_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    override = getattr(pack, "adopted_masses_override", None)
    if override is not None:
        masses = np.asarray(override, float)
    else:
        masses, phi = surface_masses(pack, pack.adopted_resp_mu_coef, pack.adopted_resp_sig_coef, pack.adopted_resp_skew_coef,
                                     np.asarray(pack.adopted_resp_fit_range, float), nhat)
        masses = masses / np.maximum(phi, 1e-12)[:, :, None, :] * np.asarray(pack.adopted_phi_ref, float)[:, :, None, :]
    zf_centers = 0.5 * (zf[:-1] + zf[1:])
    kz2K = np.minimum(np.searchsorted(zc, zf_centers, side="right") - 1, len(zc) - 2)
    rse = np.asarray(pack.resp_snr_edges, float)
    s2sr = np.clip(np.searchsorted(rse, snr[:-1] + 1e-9, side="right") - 1, 0, masses.shape[0] - 1)
    rze = np.asarray(pack.resp_z_edges, float); zc_centers = 0.5 * (zc[:-1] + zc[1:])
    K2zr = np.searchsorted(rze, zc_centers, side="right") - 1
    me = np.asarray(pack.molly_nhi_edges, float)
    b2cell = np.clip(np.searchsorted(me, Nc, side="right") - 1, 0, len(me) - 2)
    nd = np.asarray(pack.molly_n_det, float); nt = np.asarray(pack.molly_n_tot, float)
    C_cells = expit(np.log(nd + 0.5) - np.log(nt - nd + 0.5)); g = np.asarray(pack.g_grid, float)
    C_n, B_n, K_n, S_n = len(nhat) - 1, len(Nc), len(zf) - 1, len(snr) - 1
    P = np.zeros((C_n, B_n, K_n, S_n))
    for s in range(S_n):
        Cb = C_cells[s, b2cell]
        for k in range(K_n):
            zr = int(K2zr[int(kz2K[k])])
            P[:, :, k, s] = masses[int(s2sr[s]), zr] * (Cb * g[b2cell, k])[None, :]
    return P


def cc_fold_adopted(pack, theta_pop, lam_fp, *, n_lat_floor=None,
                    mu_coef=None, sig_coef=None, skew_coef=None,
                    phi_ref_tol=1e-9):
    """FAIL-CLOSED fold of a v1.2 pack's ADOPTED response representation.

    Requires the complete adopted-contract stamp group (schema v1.2) and
    ALWAYS folds count-conservingly: the adopted kernel renormalized to unit
    in-grid mass, multiplied by the pack's stored deployed ``adopted_phi_ref``
    — which is verified against a fresh recomputation from the pack's own
    frozen surfaces (guard G-CC's stored-reference identity) before use.
    Optional mu/sig/skew coefficient overrides (same shape as the adopted
    surfaces) exist so the CARRIER ensemble can be propagated draw-by-draw
    under the identical contract; the stamps and phi_ref stay mandatory.
    """
    for f in ("tp_convention_id", "contract_id", "adopted_resp_version",
              "adopted_resp_mu_coef", "adopted_phi_ref"):
        if getattr(pack, f, None) is None:
            raise ValueError(
                "cc_fold_adopted: pack lacks the adopted-contract stamp "
                f"group (missing {f}) — refuse to fold; rebuild the pack "
                "with upgrade_packs_v2 (PI ruling 2026-08-17).")
    phi_stored = np.asarray(pack.adopted_phi_ref, float)
    override = getattr(pack, "adopted_masses_override", None)
    if override is not None:
        # 2026-09-03 HZ2 (default-off extension): the adopted representation is the EMPIRICAL bin-to-bin kernel
        # (Candidate E); its column sums ARE the deployed in-grid fractions, so the G-CC stored-reference identity is
        # checked against them and the fold uses the masses directly (no surface renormalisation). Mirrors
        # cc_posterior_validation.build_cc_tensors.
        if mu_coef is not None or sig_coef is not None or skew_coef is not None:
            raise ValueError("cc_fold_adopted: coefficient overrides are undefined for an adopted_masses_override pack")
        masses = np.asarray(override, float)
        d = float(np.max(np.abs(phi_stored - masses.sum(axis=2))))
        if d > phi_ref_tol:
            raise ValueError(
                f"cc_fold_adopted: stored adopted_phi_ref differs from the "
                f"override kernel's column sums by {d:.3e} > {phi_ref_tol} "
                "— the count-conservation reference is corrupt (G-CC).")
        return cc_fold_cmarginal(pack, theta_pop, lam_fp, masses_override=masses,
                                 renormalize=False, n_lat_floor=n_lat_floor)
    phi_fresh = phi_from_surfaces(pack)
    d = float(np.max(np.abs(phi_stored - phi_fresh)))
    if d > phi_ref_tol:
        raise ValueError(
            f"cc_fold_adopted: stored adopted_phi_ref differs from the "
            f"deployed kernel's in-grid fraction by {d:.3e} > {phi_ref_tol} "
            "— the count-conservation reference is corrupt (G-CC).")
    return cc_fold_cmarginal(
        pack, theta_pop, lam_fp,
        mu_coef=(pack.adopted_resp_mu_coef if mu_coef is None else mu_coef),
        sig_coef=(pack.adopted_resp_sig_coef if sig_coef is None
                  else sig_coef),
        skew_coef=(pack.adopted_resp_skew_coef if skew_coef is None
                   else skew_coef),
        fit_rng=np.asarray(pack.adopted_resp_fit_range, float),
        renormalize=True, phi_ref=phi_stored, n_lat_floor=n_lat_floor)
