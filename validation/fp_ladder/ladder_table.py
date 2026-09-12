#!/usr/bin/env python
"""ladder_table.py — VALIDATION-ONLY reader that turns the FP-model-ladder run
artifacts into the predeclared model table.

Sealed predeclaration: notes/governance/FP_REGULARIZATION_MODEL_LADDER_PREDECLARATION.md
(sha256 3112022a…). This module DECIDES NOTHING that the predeclaration did not
already fix: it applies ``perz_gate.gate_one`` UNCHANGED (§5.1), evaluates the
frozen hard flags (§5.2) and soft flags (§5.3), assembles the §5.4 reporting
list, and prints the §6 opening rule ("M_{k+1} is run iff M_k fails"). Nothing
here re-derives a criterion, re-fits, or samples.

What it reads, per run ``RUN_<M>_<fam>_s<seed>.json`` written by run_ladder.py:
  RUN_*.json          the cc_posterior_validation-schema summary (+ additive
                      diagnostics blocks)
  RUN_*_fdraws.npz    f (D, B, Kf), truth_f (B, Kf), grids  -> coverage power
  RUN_*_bychain.npz   by-chain draws of the population + FP sites, lam_fp, t,
                      potential_energy, energy, diverging -> effective complexity
  RUN_*_ppc.json      (optional; written by mock_ppc.py) -> the PPC column

Usage:
  python validation/fp_ladder/ladder_table.py --runs DIR [DIR ...] \
      --out-json LADDER_TABLE.json --out-md LADDER_TABLE.md [--pack-dir P]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# frozen constants of the predeclaration (NOT tunable here)
# ---------------------------------------------------------------------------
EBFMI_MIN = 0.2                         # §5.2 sampler hard flag
OMEGA_BIAS_PCT_MAX = 3.0                # §5.3 soft flag: |bias| on Omega[20.3, 21.6]
PAPER_FIGURES = "/home/mfho/Latex/gp_dla_desi_y3/paper_figures"   # READ-ONLY import
FP_TRUTH_RATIO_BAND = (0.5, 2.0)        # §5.3 soft flag (evaluated by the runner)
COVERAGE_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)   # §5.3 detection curve
PE_EXCESS_NATS = 50.0                   # stuck-chain exclusion for the FP eigen block
HEADLINES = ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz")
SCALE_SITES = ("fp_tau_N", "fp_tau_I")  # HalfNormal scales: reported, not whitened

# §4 of the predeclaration / §4 of the Stage-0 design, verbatim in substance.
WHY_INTRODUCED = {
    "M0": ("calibrated total only — the loa-0 template with t == 0; the reference "
           "row (anchored / amplitude analogue), expected to fail"),
    "M1": ("+ redshift transfer t_K ~ N(0,1): mock truth shows the loa-0 z-transfer "
           "is off by 2-4x and the old sigma_t is 5-10 sigma too tight"),
    "M2": ("+ one N-hat slope beta_1 and calibration-identified stratum effects h_j: "
           "replaces the noisy 25-cell template by the calibrated log-linear tail "
           "and the six stratum totals (6-31 loa-0 events each)"),
    "M3": ("+ N-hat curvature beta_2 — run only if M2 fails; the loa-0 tail steepens "
           "near 20.0"),
    "M4": ("+ r_c, a hierarchically shrunk random walk in N-hat shared across strata "
           "(tau_N ~ HalfNormal(0.5)) — run only if M3 fails; the local-residual "
           "freedom block, effective size set by tau_N"),
    "M5": ("+ tau_I eps_cs on live cells — the hierarchical ceiling; NOT run without "
           "a PI ruling (maximum complexity without returning to the PI is M4)"),
}
LADDER_ORDER = ("M0", "M1", "M2", "M3", "M4", "M5")


# ---------------------------------------------------------------------------
# (e) effective complexity of the FP block — pure helpers (unit-tested)
# ---------------------------------------------------------------------------
_NAME_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([0-9,\s]+)\])?$")


def parse_coord_name(name):
    """'fp_h[3]' -> ('fp_h', (3,)); 'fp_l0' -> ('fp_l0', ());
    'fp_eps_z[2,5]' -> ('fp_eps_z', (2, 5))."""
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"unparseable FP coordinate name {name!r}")
    base, idx = m.group(1), m.group(2)
    if idx is None:
        return base, ()
    return base, tuple(int(t) for t in idx.split(","))


def assemble_fp_matrix(by, names):
    """Gather the scalar FP coordinates named by ``fp_prior_moments`` out of the
    by-chain arrays.

    by    : mapping site -> array (chains, draws, *event_shape) (the _bychain npz)
    names : the sampling-order coordinate names from ``fp_ladder.fp_prior_moments``

    Returns (X, kept_names, missing) with X of shape (chains, draws, len(kept)).
    A name whose site is absent from the npz is skipped and listed in ``missing``
    (M0's ``t`` is deterministic-zero, for instance).
    """
    cols, kept, missing = [], [], []
    for nm in names:
        base, idx = parse_coord_name(nm)
        if base not in by:
            missing.append(nm)
            continue
        arr = np.asarray(by[base])
        if arr.ndim < 2:
            missing.append(nm)
            continue
        if arr.ndim != 2 + len(idx):
            missing.append(nm)          # index depth does not match the saved shape
            continue
        sub = arr
        ok = True
        for i in idx:
            if i >= sub.shape[2]:
                ok = False
                break
            sub = sub[:, :, i]
        if not ok or sub.ndim != 2:
            missing.append(nm)
            continue
        cols.append(sub)
        kept.append(nm)
    if not cols:
        return np.zeros((0, 0, 0)), [], missing
    X = np.stack(cols, axis=-1)          # (chains, draws, P)
    return X, kept, missing


def stuck_chains(potential_energy, excess_nats=PE_EXCESS_NATS):
    """Chains whose MEAN potential energy exceeds the chain minimum by more than
    ``excess_nats``. (ckpt 10.9 pathology: a chain parked in a distant basin
    contaminates any pooled covariance.) Returns (kept_idx, excluded_idx, means)."""
    pe = np.asarray(potential_energy, float)
    means = pe.mean(axis=1)
    lo = float(np.min(means))
    excl = [int(i) for i in np.flatnonzero(means > lo + float(excess_nats))]
    kept = [int(i) for i in range(len(means)) if i not in excl]
    return kept, excl, [float(m) for m in means]


def whiten(X, mean, sd):
    """u = (x - prior_mean) / prior_sd, elementwise on the last axis."""
    X = np.asarray(X, float)
    return (X - np.asarray(mean, float)[None, :]) / np.asarray(sd, float)[None, :]


def eigen_report(u):
    """Effective complexity of a prior-whitened block u (n_draws, P).

    p_eff             = sum_i (1 - Var u_i)  == tr(I - diag Sigma_post)
                        (the predeclaration's tr(I - Sigma_post) read on the
                        whitened coordinates; the diagonal read is the one that
                        is defined for P == 1 and P == 2 as well)
    eigenvalues       = eig(Cov(u)), descending
    n_informed        = #{i : 1 - lambda_i > 0.5}
    participation_ratio = (sum nu)^2 / sum nu^2 on nu_i = max(1 - lambda_i, 0)
                        — how many directions carry the information (1 = one
                        direction does all of it; P = spread evenly)
    """
    u = np.asarray(u, float)
    n, P = u.shape
    var = u.var(axis=0, ddof=1) if n > 1 else np.zeros(P)
    p_eff = float(np.sum(1.0 - var))
    out = dict(n_draws=int(n), n_coords=int(P),
               posterior_var_whitened=[float(v) for v in var],
               p_eff=p_eff)
    if P >= 3 and n > P:
        C = np.cov(u, rowvar=False)
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        nu = 1.0 - ev
        nu_pos = np.clip(nu, 0.0, None)
        pr = float((nu_pos.sum() ** 2) / np.sum(nu_pos ** 2)) if np.sum(nu_pos ** 2) > 0 else 0.0
        out.update(eigenvalues=[float(x) for x in ev],
                   n_informed_1m_lambda_gt_0p5=int(np.sum(nu > 0.5)),
                   participation_ratio=pr,
                   p_eff_trace=float(np.sum(nu)))
    else:
        out.update(eigenvalues=None, n_informed_1m_lambda_gt_0p5=None,
                   participation_ratio=None, p_eff_trace=None,
                   note=("eigen analysis requires >= 3 whitened coordinates and "
                         "n_draws > n_coords"))
    return out


def r2_on_block(y, u):
    """R^2 of the OLS regression of y (n,) on [1, u] (n, P). The 'science-loading
    direction' figure of §5.4: how much of the headline's posterior variance the
    FP block explains."""
    y = np.asarray(y, float).ravel()
    u = np.asarray(u, float)
    if u.size == 0 or y.size != u.shape[0] or y.size < u.shape[1] + 2:
        return None
    A = np.concatenate([np.ones((u.shape[0], 1)), u], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        return None
    return float(1.0 - float(np.sum(resid ** 2)) / sst)


# ---------------------------------------------------------------------------
# rollup logic (unit-tested on synthetic dicts)
# ---------------------------------------------------------------------------
def rollup(runs):
    """(model, family) passes iff EVERY run of that pair passes the gate AND
    raises no hard flag (the predeclaration's "a family passes when both seeds
    pass"); a model passes iff all three families pass.

    ``runs`` is a list of dicts carrying at least: model, family, seed,
    gate_status ('PASS'/'FAIL'), hard_flags (list).
    """
    fam = {}
    for r in runs:
        key = (r["model"], r["family"])
        fam.setdefault(key, []).append(r)
    by_family = {}
    for (m, f), rs in sorted(fam.items()):
        ok = all(x["gate_status"] == "PASS" and not x["hard_flags"] for x in rs)
        by_family[f"{m}|{f}"] = dict(
            model=m, family=f, n_runs=len(rs),
            seeds=sorted(x.get("seed") for x in rs),
            status="PASS" if ok else "FAIL",
            gate_fails=sorted({g for x in rs for g in x.get("gate_fails", [])}),
            hard_flags=sorted({g for x in rs for g in x["hard_flags"]}),
            soft_flags=sorted({g for x in rs for g in x.get("soft_flags", [])}))
    by_model = {}
    for m in sorted({r["model"] for r in runs}, key=lambda x: LADDER_ORDER.index(x)
                    if x in LADDER_ORDER else 99):
        fams = [v for v in by_family.values() if v["model"] == m]
        by_model[m] = dict(
            model=m, n_families=len(fams),
            families={v["family"]: v["status"] for v in fams},
            status="PASS" if fams and all(v["status"] == "PASS" for v in fams) else "FAIL",
            gate_fails=sorted({g for v in fams for g in v["gate_fails"]}),
            hard_flags=sorted({g for v in fams for g in v["hard_flags"]}),
            soft_flags=sorted({g for v in fams for g in v["soft_flags"]}))
    return dict(by_family=by_family, by_model=by_model)


def opening_rule(by_model):
    """§6: wave 1 is M0, M1, M2; M_{k+1} is run iff M_k fails (k = 2, 3); stop at
    the first passing M_k (k <= 4); if M4 fails, STOP and return to the PI."""
    lines, decisions = [], {}
    for k, nxt in (("M2", "M3"), ("M3", "M4")):
        if k not in by_model:
            decisions[nxt] = "UNDETERMINED (no %s result present)" % k
        elif by_model[k]["status"] == "PASS":
            decisions[nxt] = f"NOT RUN — {k} PASSES (selection stops at {k})"
        else:
            decisions[nxt] = f"RUN — {k} FAILS ({'; '.join(by_model[k]['gate_fails'] + by_model[k]['hard_flags']) or 'flagged'})"
    passing = [m for m in LADDER_ORDER
               if m in by_model and by_model[m]["status"] == "PASS"]
    selected = passing[0] if passing else None
    if "M4" in by_model and by_model["M4"]["status"] != "PASS" and not passing:
        decisions["M5"] = ("NOT RUN — M4 FAILED: STOP and return to the PI "
                           "(calibration-insufficient conclusion goes on the table)")
    else:
        decisions["M5"] = "NOT RUN (never run without a PI ruling)"
    for k, v in decisions.items():
        lines.append(f"{k}: {v}")
    return dict(decisions=decisions, selected_model=selected,
                selection_rule=("the passing model of lowest k; if M_k passes, "
                                "M_{k+1} is not run"),
                lines=lines)


# ---------------------------------------------------------------------------
# per-run analysis
# ---------------------------------------------------------------------------
def _pack_path(d, pack_dir):
    p = d.get("pack", "")
    if pack_dir:
        cand = os.path.join(pack_dir, os.path.basename(p))
        if os.path.exists(cand):
            return cand
    return p if os.path.exists(p) else None


def coverage_power(fdraws_path, pack, scales=COVERAGE_SCALES):
    """(d) the rescale_dispersion detection curve on the two all-z headlines.

    ``sbc.rescale_dispersion`` moves the posterior WIDTH about the per-bin
    median in log f and leaves the median invariant (s == 1 returns the input
    object bit-identically, so the curve's baseline IS the certified result).
    We report truth_in_68 / truth_in_95 at every s: a containment claim is only
    supported if the check actually FAILS where the dispersion is wrong."""
    from CDDF_analysis.hbi_mcmc.sbc import rescale_dispersion
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
    z = np.load(fdraws_path)
    f = np.asarray(z["f"], float)
    ft = np.asarray(z["truth_f"], float)
    red_t = reduce_f_posterior(ft[None, :, :], pack)
    truth = {k: float(np.asarray(red_t[k]).reshape(-1)[0]) for k in HEADLINES}
    curve = {k: [] for k in HEADLINES}
    for s in scales:
        fs = rescale_dispersion(f, s)
        red = reduce_f_posterior(fs, pack)
        for k in HEADLINES:
            dr = np.asarray(red[k]).ravel()
            q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
            curve[k].append(dict(
                scale=float(s), median=float(q[2]),
                p16_84=[float(q[1]), float(q[3])],
                p2p5_97p5=[float(q[0]), float(q[4])],
                truth_in_68=bool(q[1] <= truth[k] <= q[3]),
                truth_in_95=bool(q[0] <= truth[k] <= q[4]),
                median_bias_pct=(round(100 * (q[2] / truth[k] - 1), 3)
                                 if truth[k] > 0 else None)))
    out = dict(scales=[float(s) for s in scales], truth=truth, curve=curve)
    for k in HEADLINES:
        rows = {r["scale"]: r for r in curve[k]}
        flagged_lo = not rows[0.5]["truth_in_68"] if 0.5 in rows else None
        flagged_hi = not rows[2.0]["truth_in_68"] if 2.0 in rows else None
        out.setdefault("power", {})[k] = dict(
            flagged_at_s0p5=flagged_lo, flagged_at_s2p0=flagged_hi,
            containment_claimable=bool(flagged_lo and flagged_hi),
            note=("a containment claim is NOT made unless s = 2.0 is also flagged "
                  "(a containment test cannot fail an over-wide band)"))
    return out


# ---------------------------------------------------------------------------
# Omega[20.3, 21.6] all-z, post-hoc, on the PAPER's own reduction weights
# ---------------------------------------------------------------------------
def omega_allz_from_weights(f, truth_f, omega_w, z_w, prefactor):
    """The path-weighted all-z Omega reduction, given the weights.

    Pure arithmetic, identical in form to hbi_reduction.Posterior._reduce:

        Omega = prefactor * einsum('dbk,b,k->d', f, omega_w, z_w) / sum(z_w)

    Separated from the weight construction so it can be pinned by a unit test
    without importing the paper repository.
    """
    f = np.asarray(f, float)
    omega_w = np.asarray(omega_w, float)
    z_w = np.asarray(z_w, float)
    zs = float(z_w.sum())
    if zs <= 0.0:
        return None
    post = float(prefactor) * np.einsum("dbk,b,k->d", f, omega_w, z_w) / zs
    truth = float(float(prefactor)
                  * np.einsum("bk,b,k->", np.asarray(truth_f, float), omega_w, z_w) / zs)
    q = np.percentile(post, [2.5, 16, 50, 84, 97.5])
    return dict(truth=truth,
                post_p16_50_84=[float(q[1]), float(q[2]), float(q[3])],
                post_p2p5_97p5=[float(q[0]), float(q[4])],
                median_bias_pct=(round(100 * (float(q[2]) / truth - 1), 3)
                                 if truth > 0 else None),
                truth_in_68=bool(q[1] <= truth <= q[3]),
                truth_in_95=bool(q[0] <= truth <= q[4]),
                dX_total=zs)


def paper_omega_20p3_21p6(fdraws_path, pack=None, paper_figures=PAPER_FIGURES):
    """Omega[20.3, 21.6] all-z bias, built with the PAPER'S OWN reduction weights.

    Why this exists: ``reduce_f_posterior`` has no [20.3, 21.6] tier, so the
    run JSON's ``thresholds.omega_allz`` is the sub-DLA window [19.5, 20.3) —
    NOT the quantity the predeclaration §5.3 soft flag names. The paper-side
    weights are imported READ-ONLY from ``paper_figures/hbi_reduction.py``
    (``_omega_weight(*OMEGA_NHI)``, ``_z_weight`` over ``LOWZ_SUPPORT``,
    ``OMEGA_PREFACTOR_CM2``), exactly as the long-chain campaign's
    ``quantities.py`` does: a ``Posterior`` is built with ``__new__`` and only
    its ``f / n_edges / z_edges / dX`` attributes are set, so nothing is
    recomputed and no definition is re-derived here.

    ``dX`` comes from the run's own ``_fdraws.npz`` (``dX_k``, which run_ladder
    writes as ``pack.dX.sum(axis=1)``); when the pack is available the redshift
    grids are cross-checked and a disagreement BLOCKS, as quantities.py does.
    """
    if not os.path.isdir(paper_figures):
        return {"unavailable": f"paper_figures not found at {paper_figures}"}
    if paper_figures not in sys.path:
        sys.path.insert(0, paper_figures)
    try:
        import hbi_reduction as HR          # READ-ONLY import
    except Exception as e:
        return {"unavailable": f"cannot import hbi_reduction: {type(e).__name__}: {e}"}

    z = np.load(fdraws_path)
    need = ("f", "truth_f", "ntrue_edges", "zf_edges", "dX_k")
    miss = [k for k in need if k not in z.files]
    if miss:
        return {"unavailable": f"_fdraws.npz lacks {miss}"}
    f = np.asarray(z["f"], float)
    ft = np.asarray(z["truth_f"], float)
    n_edges = np.asarray(z["ntrue_edges"], float)
    z_edges = np.asarray(z["zf_edges"], float)
    dX = np.asarray(z["dX_k"], float)
    if pack is not None:
        pz = np.asarray(pack.zf_edges, float)
        if not np.allclose(z_edges, pz):
            return {"blocked": "the draws and the pack disagree about the redshift grid"}
        pdx = np.asarray(pack.dX, float).sum(axis=1)
        if not np.allclose(dX, pdx):
            return {"blocked": "the draws' dX_k and the pack's dX.sum(axis=1) disagree"}

    P = HR.Posterior.__new__(HR.Posterior)      # borrow the weight functions only
    P.f, P.n_edges, P.z_edges, P.dX = f, n_edges, z_edges, dX
    ow = P._omega_weight(*HR.OMEGA_NHI)
    zw = P._z_weight(*HR.LOWZ_SUPPORT)
    out = omega_allz_from_weights(f, ft, ow, zw, HR.OMEGA_PREFACTOR_CM2)
    if out is None:
        return {"unavailable": "the z weights sum to zero over LOWZ_SUPPORT"}
    out.update(window_nhi=[float(x) for x in HR.OMEGA_NHI],
               window_z=[float(x) for x in HR.LOWZ_SUPPORT],
               prefactor_cm2=float(HR.OMEGA_PREFACTOR_CM2),
               h_reporting=float(getattr(HR, "H_REPORTING", float("nan"))),
               source=("paper_figures/hbi_reduction.py (read-only): "
                       "_omega_weight(*OMEGA_NHI), _z_weight(*LOWZ_SUPPORT), "
                       "OMEGA_PREFACTOR_CM2"),
               units="Omega_HI (dimensionless), paper reporting cosmology")
    return out


def complexity_block(bychain_path, fdraws_path, pack, ladder, t_sd, tau_scale):
    """(e) effective complexity of the FP block: prior-whitened p_eff, eigen
    spectrum, informed directions, participation ratio, and the R^2 of each all-z
    headline on the whitened FP coordinates."""
    from CDDF_analysis.hbi_mcmc.fp_ladder import fp_prior_moments, NOMINAL_FP_DOF
    from CDDF_analysis.hbi_mcmc.forward import build_consts
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior

    z = np.load(bychain_path, allow_pickle=True)
    by = {k: z[k] for k in z.files}
    consts = build_consts(pack, resp_clamp="both")
    names, mean, sd = fp_prior_moments(consts, np.asarray(pack.fp_counts, float),
                                       ladder, t_sd=t_sd, tau_scale=tau_scale)
    X, kept, missing = assemble_fp_matrix(by, names)
    keep_c, excl_c, pe_means = stuck_chains(by["potential_energy"])
    idx = {n: i for i, n in enumerate(names)}

    # HalfNormal scale coordinates: reported (posterior vs prior sd), not whitened
    scale_report = {}
    for s in SCALE_SITES:
        if s in by:
            v = np.asarray(by[s], float)[keep_c].ravel()
            scale_report[s] = dict(
                posterior_sd=float(v.std(ddof=1)) if v.size > 1 else None,
                posterior_p16_50_84=[float(x) for x in np.percentile(v, [16, 50, 84])],
                prior_sd=float(sd[idx[s]]) if s in idx else None,
                prior_mean=float(mean[idx[s]]) if s in idx else None)
    eig_names = [n for n in kept if parse_coord_name(n)[0] not in SCALE_SITES]
    sel = [kept.index(n) for n in eig_names]
    out = dict(ladder=ladder, nominal_fp_dof=int(NOMINAL_FP_DOF[ladder]),
               coordinates_expected=len(names), coordinates_found=len(kept),
               coordinates_missing=missing, coordinates_whitened=eig_names,
               scale_coordinates=scale_report,
               chains_total=int(np.asarray(by["potential_energy"]).shape[0]),
               chains_excluded_pe=excl_c, mean_potential_energy_per_chain=pe_means,
               pe_exclusion_excess_nats=PE_EXCESS_NATS)
    if not eig_names:
        out["note"] = "no non-scale FP coordinate found in the by-chain file"
        return out
    m = np.asarray([mean[idx[n]] for n in eig_names], float)
    s_ = np.asarray([sd[idx[n]] for n in eig_names], float)
    Xk = X[keep_c][:, :, sel]                              # (kept_chains, draws, P)
    u = whiten(Xk.reshape(-1, Xk.shape[-1]), m, s_)        # (n, P)
    out.update(eigen_report(u))

    # science loading: R^2 of each headline on the whitened FP block. The f draws
    # are chain-major flattened by numpyro, so chain c draw d is row c*draws + d.
    if fdraws_path and os.path.exists(fdraws_path):
        fz = np.load(fdraws_path)
        f = np.asarray(fz["f"], float)
        nch, ndr = Xk.shape[0], Xk.shape[1]
        allch = int(np.asarray(by["potential_energy"]).shape[0])
        if f.shape[0] == allch * ndr:
            rows = np.concatenate([np.arange(c * ndr, (c + 1) * ndr) for c in keep_c])
            red = reduce_f_posterior(f[rows], pack)
            out["headline_r2_on_fp_block"] = {
                k: r2_on_block(np.asarray(red[k]).ravel(), u) for k in HEADLINES}
        else:
            out["headline_r2_on_fp_block"] = None
            out["headline_r2_note"] = (
                f"f draws ({f.shape[0]}) do not match chains*draws ({allch * ndr}); "
                "the draw-for-draw join is refused rather than guessed")
    return out


def reporting_list(d):
    """(f) the §5.4 always-reported items, copied out of the run JSON."""
    dg = d.get("diagnostics", {})
    perz = d.get("perz_recovery", {}).get("estimand", {})

    def _bins(thr, key):
        blk = perz.get(thr, {})
        return [dict(bin=b.get("bin"), z=b.get("z"),
                     median_bias_pct=b.get("median_bias_pct"),
                     truth_in_68=b.get("truth_in_68"), truth_in_95=b.get("truth_in_95"))
                for b in blk.get(key, []) if b.get("available")]

    return dict(
        thresholds=d.get("thresholds"),
        reporting_bins_zigzag=d.get("reporting_bins"),
        perz_coarse_blocks={t: _bins(t, "coarse_blocks") for t in ("ge20.0", "ge20.3")},
        perz_paper1_bins={t: _bins(t, "paper1_bins") for t in ("ge20.0", "ge20.3")},
        lam_over_naive=dg.get("fp_lam_total_over_naive"),
        t_post_mean=dg.get("t_post_mean"), t_post_sd=dg.get("t_post_sd"),
        e_t_post_median=dg.get("e_t_post_median"),
        t_post_in_record_prior_sd=dg.get("t_post_in_record_prior_sd"),
        fp_scalar_posts=dg.get("fp_scalar_posts"),
        predictive_fp_share=dg.get("predictive_fp_share"),
        predictive_total_ratio=dg.get("predictive_total_ratio"),
        fp_by_block=dg.get("fp_by_block"),
        calibration_predictive=dg.get("calibration_predictive"),
        fp_truth=dg.get("fp_truth"),
        divergences=d.get("divergences"),
        divergences_per_chain=dg.get("divergences_per_chain"),
        ebfmi_per_chain=dg.get("ebfmi_per_chain"),
        estimand_mixing=dg.get("estimand_mixing"),
        nominal_fp_dof=dg.get("nominal_fp_dof"),
        run_config=d.get("run_config"))


def analyse_run(path, pack_dir=None):
    from CDDF_analysis.hbi_mcmc.perz_gate import gate_one
    from CDDF_analysis.hbi_mcmc.pack import load_pack

    d = json.load(open(path))
    base = path[:-5]
    dg = d.get("diagnostics", {})
    gate = gate_one(d)                                   # §5.1 UNCHANGED

    # ---- §5.2 hard flags ---------------------------------------------------
    hard = []
    cal = dg.get("calibration_predictive") or {}
    if cal.get("flag"):
        hard.append("calibration_predictive (p_dev %s, P0>=20.2 %s)"
                    % (cal.get("deviance_pvalue"), cal.get("prob_zero_ge20p2")))
    eb = dg.get("ebfmi_per_chain") or []
    eb_min = float(min(eb)) if eb else None
    if eb_min is not None and eb_min < EBFMI_MIN:
        hard.append(f"E-BFMI min {eb_min:.4f} < {EBFMI_MIN}")

    # ---- §5.3 soft flags ---------------------------------------------------
    soft = []
    notes = []
    fpt = dg.get("fp_truth")
    if isinstance(fpt, dict) and fpt.get("flag"):
        soft.append("fp_truth ratio outside [0.5, 2.0]")
    # The runner's own thresholds.omega_allz is NOT the predeclared quantity:
    # reduce_f_posterior has no [20.3, 21.6] tier (its only emittable Omega
    # window is report_197_216 = [19.7, 21.6]), so run_ladder stores the first
    # 'omega*allz' key it finds — the sub-DLA window [19.5, 20.3). It is carried
    # below as a clearly-labelled extra column and NEVER gates the soft flag.
    om_runner = (d.get("thresholds") or {}).get("omega_allz")
    if isinstance(om_runner, dict) and "20p3" not in str(om_runner.get("key")):
        notes.append(f"the runner-carried thresholds.omega_allz is "
                     f"'{om_runner.get('key')}' (NOT Omega[20.3, 21.6]); it is "
                     "reported as an extra column only and gates nothing")

    rec = dict(file=os.path.basename(path), path=os.path.abspath(path),
               model=d.get("ladder"), family=gate["family"],
               seed=(d.get("run_config") or {}).get("seed"),
               pack=d.get("pack"), n_draws=d.get("n_draws"),
               gate=gate, gate_status=gate["status"], gate_fails=gate["fails"],
               hard_flags=hard, soft_flags=soft,
               ebfmi_min=eb_min, schema_notes=notes,
               reporting=reporting_list(d))

    # ---- (d) + (e): need the pack and the sidecars -------------------------
    pkp = _pack_path(d, pack_dir)
    fdr, byc = base + "_fdraws.npz", base + "_bychain.npz"
    rec["pack_resolved"] = pkp
    pack = load_pack(pkp) if pkp else None

    # ---- §5.3 Omega[20.3, 21.6] soft flag, on the PAPER's own weights -------
    # Needs only the _fdraws.npz (f, truth_f, grids, dX_k); the pack is used to
    # cross-check the grids when it is available.
    if os.path.exists(fdr):
        try:
            om = paper_omega_20p3_21p6(fdr, pack)
        except Exception as e:
            om = {"error": f"{type(e).__name__}: {e}"}
    else:
        om = {"unavailable": "no _fdraws.npz next to the run"}
    rec["reporting"]["omega_20p3_21p6_allz"] = om
    rec["reporting"]["omega_runner_allz_other_window"] = om_runner
    if om.get("median_bias_pct") is not None:
        if abs(float(om["median_bias_pct"])) > OMEGA_BIAS_PCT_MAX:
            soft.append(f"omega[20.3,21.6] all-z bias {om['median_bias_pct']:+.2f}% "
                        f"(|bias| > {OMEGA_BIAS_PCT_MAX}%)")
    else:
        notes.append("the Omega[20.3, 21.6] soft flag could NOT be evaluated: "
                     + str(om.get("unavailable") or om.get("blocked")
                           or om.get("error") or "unknown reason"))

    if pkp is None:
        rec["coverage_power"] = {"unavailable": "pack not found (pass --pack-dir)"}
        rec["complexity"] = {"unavailable": "pack not found (pass --pack-dir)"}
    else:
        try:
            rec["coverage_power"] = (coverage_power(fdr, pack)
                                     if os.path.exists(fdr)
                                     else {"unavailable": "no _fdraws.npz next to the run"})
        except Exception as e:                                  # reported, never silent
            rec["coverage_power"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            rec["complexity"] = (
                complexity_block(byc, fdr, pack, d.get("ladder"),
                                 float(dg.get("t_sd", 1.0)),
                                 float(dg.get("tau_scale", 0.5)))
                if os.path.exists(byc)
                else {"unavailable": "no _bychain.npz next to the run"})
        except Exception as e:
            rec["complexity"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- PPC column (only if mock_ppc.py has been run next to this run) -----
    ppcp = base + "_ppc.json"
    if os.path.exists(ppcp):
        p = json.load(open(ppcp))
        rec["ppc"] = dict(file=os.path.basename(ppcp), checks=p.get("ppc_block", {}).get("checks"),
                          snr_ramp=p.get("snr_ramp"),
                          omnibus_p=(p.get("ppc_block", {})
                                     .get("omnibus_chi2_discrepancy", {})
                                     .get("posterior_predictive_p")),
                          n_cells_failed=p.get("ppc_block", {}).get("n_cells_failed"),
                          flag=p.get("flag_any"))
        if p.get("flag_any"):
            soft.append("PPC: " + "; ".join(p.get("flags", []) or ["flagged"]))
    else:
        rec["ppc"] = None
    return rec


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------
def _rng(vals, fmt="{:+.2f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)} .. {fmt.format(hi)}"


def render_md(runs, roll, opening):
    L = []
    L.append("# FP-model ladder — model table (predeclaration §5.4)")
    L.append("")
    L.append("Generated by `validation/fp_ladder/ladder_table.py`. VALIDATION-ONLY read-out "
             "of MOCK runs; the gate is `perz_gate.gate_one` CRIT v2 UNCHANGED and every "
             "flag threshold is the sealed one. This table decides nothing beyond the "
             "predeclared rules.")
    L.append("")
    L.append("## Model table")
    L.append("")
    L.append("| model | nominal FP DOF | why introduced | mock bias ≥20.0 / ≥20.3 (range over runs, %) | "
             "Ω[20.3,21.6] all-z bias % (§5.3) | Ω runner window (extra, gates nothing) | "
             "coverage (s=1 truth_in_68/95; power) | PPC | calibration check | FP-truth ratios | p_eff | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in LADDER_ORDER:
        rs = [r for r in runs if r["model"] == m]
        if not rs:
            continue
        dof = next((r["reporting"].get("nominal_fp_dof") for r in rs
                    if r["reporting"].get("nominal_fp_dof") is not None), "?")
        b0 = _rng([r["gate"]["allz"]["ge20p0"] for r in rs])
        b3 = _rng([r["gate"]["allz"]["ge20p3"] for r in rs])
        omp = _rng([(r["reporting"].get("omega_20p3_21p6_allz") or {}).get("median_bias_pct")
                    for r in rs])
        omr_key = next((str((r["reporting"].get("omega_runner_allz_other_window") or {})
                            .get("key")) for r in rs
                        if r["reporting"].get("omega_runner_allz_other_window")), "n/a")
        omr = (_rng([(r["reporting"].get("omega_runner_allz_other_window") or {})
                     .get("median_bias_pct") for r in rs]) + f" [{omr_key}]")
        # coverage at s = 1
        c68 = c95 = ntot = 0
        claim = []
        for r in rs:
            cp = r.get("coverage_power") or {}
            cur = cp.get("curve") or {}
            for k in HEADLINES:
                row = next((x for x in cur.get(k, []) if x["scale"] == 1.0), None)
                if row:
                    ntot += 1
                    c68 += int(row["truth_in_68"]); c95 += int(row["truth_in_95"])
            for k, v in (cp.get("power") or {}).items():
                claim.append(bool(v.get("containment_claimable")))
        cov = (f"{c68}/{ntot} in 68, {c95}/{ntot} in 95; "
               f"power {sum(claim)}/{len(claim)}" if ntot else "n/a")
        ppc = "n/a"
        pp = [r["ppc"] for r in rs if r.get("ppc")]
        if pp:
            amps = [p["snr_ramp"]["amplitude"] for p in pp if p.get("snr_ramp")]
            nf = sum(1 for p in pp if p.get("flag"))
            ppc = (f"{len(pp)} run(s); ramp {_rng(amps, '{:.4f}')}; flagged {nf}")
        pdev = _rng([(r["reporting"].get("calibration_predictive") or {}).get("deviance_pvalue")
                     for r in rs], "{:.3f}")
        p0 = _rng([(r["reporting"].get("calibration_predictive") or {}).get("prob_zero_ge20p2")
                   for r in rs], "{:.3f}")
        ncf = sum(1 for r in rs if any(f.startswith("calibration") for f in r["hard_flags"]))
        cal = f"p_dev {pdev}; P(0 ≥20.2) {p0}; flagged {ncf}/{len(rs)}"
        fptr = []
        for r in rs:
            ft = r["reporting"].get("fp_truth")
            if isinstance(ft, dict):
                fptr += list(ft.get("ratio_block") or []) + list((ft.get("ratio_nhat_group") or {}).values())
        fpt = _rng(fptr, "{:.2f}") if fptr else "n/a (no census)"
        pe = _rng([(r.get("complexity") or {}).get("p_eff") for r in rs], "{:.2f}")
        v = roll["by_model"].get(m, {})
        verdict = v.get("status", "?")
        fams = v.get("families", {})
        verdict += " (" + ", ".join(f"{k}:{s}" for k, s in sorted(fams.items())) + ")" if fams else ""
        L.append(f"| {m} | {dof} | {WHY_INTRODUCED.get(m, '')} | {b0} / {b3} | {omp} | {omr} | "
                 f"{cov} | {ppc} | {cal} | {fpt} | {pe} | **{verdict}** |")
    L.append("")
    L.append("## Per-run detail")
    L.append("")
    L.append("| run | model | family | seed | gate | allz ≥20.0 / ≥20.3 (%) | B3 residual | "
             "Ω[20.3,21.6] median (truth) | Ω[20.3,21.6] bias % | Ω in 68/95 | "
             "Ω runner window bias % | div | "
             "E-BFMI min | p_dev | P(0 ≥20.2) | λ/naive (median) | e^t median | FP share | "
             "p_eff / nominal | informed dirs | PR | R² ≥20.0 / ≥20.3 | hard flags | soft flags |")
    L.append("|" + "---|" * 24)
    for r in sorted(runs, key=lambda x: (LADDER_ORDER.index(x["model"])
                                         if x["model"] in LADDER_ORDER else 99,
                                         str(x["family"]), str(x["seed"]))):
        rep, cx = r["reporting"], (r.get("complexity") or {})
        cal = rep.get("calibration_predictive") or {}
        lam = rep.get("lam_over_naive") or []
        et = rep.get("e_t_post_median") or []
        r2 = cx.get("headline_r2_on_fp_block") or {}
        def _f(x, f="{:.3f}"):
            return f.format(x) if isinstance(x, (int, float)) else "n/a"
        L.append(
            f"| {r['file']} | {r['model']} | {r['family']} | {r['seed']} | "
            f"{r['gate_status']} | {_f(r['gate']['allz']['ge20p0'], '{:+.2f}')} / "
            f"{_f(r['gate']['allz']['ge20p3'], '{:+.2f}')} | "
            f"{_f(r['gate'].get('named_residual_B3_ge20p3'), '{:+.2f}')} | "
            f"{_f((rep.get('omega_20p3_21p6_allz') or {}).get('post_p16_50_84', [None, None])[1], '{:.4e}')} "
            f"({_f((rep.get('omega_20p3_21p6_allz') or {}).get('truth'), '{:.4e}')}) | "
            f"{_f((rep.get('omega_20p3_21p6_allz') or {}).get('median_bias_pct'), '{:+.2f}')} | "
            f"{(rep.get('omega_20p3_21p6_allz') or {}).get('truth_in_68')}"
            f"/{(rep.get('omega_20p3_21p6_allz') or {}).get('truth_in_95')} | "
            f"{_f((rep.get('omega_runner_allz_other_window') or {}).get('median_bias_pct'), '{:+.2f}')} | "
            f"{r['reporting'].get('divergences')} | {_f(r.get('ebfmi_min'), '{:.4f}')} | "
            f"{_f(cal.get('deviance_pvalue'))} | {_f(cal.get('prob_zero_ge20p2'))} | "
            f"{_f(lam[1] if len(lam) > 1 else None)} | "
            f"{'/'.join(_f(x, '{:.2f}') for x in et) if et else 'n/a'} | "
            f"{_f(rep.get('predictive_fp_share'))} | "
            f"{_f(cx.get('p_eff'), '{:.2f}')} / {cx.get('nominal_fp_dof', 'n/a')} | "
            f"{cx.get('n_informed_1m_lambda_gt_0p5', 'n/a')} | "
            f"{_f(cx.get('participation_ratio'), '{:.2f}')} | "
            f"{_f(r2.get(HEADLINES[0]))} / {_f(r2.get(HEADLINES[1]))} | "
            f"{'; '.join(r['hard_flags']) or '—'} | {'; '.join(r['soft_flags']) or '—'} |")
    L.append("")
    L.append("## Coverage detection curve (rescale_dispersion, §5.3)")
    L.append("")
    L.append("| run | estimand | " + " | ".join(f"s={s}" for s in COVERAGE_SCALES)
             + " | containment claimable |")
    L.append("|---|---|" + "---|" * (len(COVERAGE_SCALES) + 1))
    for r in runs:
        cp = r.get("coverage_power") or {}
        for k in HEADLINES:
            rows = (cp.get("curve") or {}).get(k)
            if not rows:
                continue
            cells = []
            for s in COVERAGE_SCALES:
                row = next((x for x in rows if x["scale"] == s), None)
                cells.append("—" if row is None else
                             ("68✓" if row["truth_in_68"] else
                              ("95✓" if row["truth_in_95"] else "OUT")))
            cl = (cp.get("power") or {}).get(k, {}).get("containment_claimable")
            L.append(f"| {r['file']} | {k} | " + " | ".join(cells) + f" | {cl} |")
    L.append("")
    L.append("## §6 opening rule")
    L.append("")
    L.append("Rule as sealed: wave 1 = M0, M1, M2 on all three families × both seeds; "
             "**M_{k+1} is run iff M_k fails** 5.1 or raises a 5.2 flag on any family or "
             "seed (k = 2, 3); stop at the first M_k (k ≤ 4) passing 5.1 and 5.2 "
             "everywhere; if M4 fails, STOP and return to the PI (no M5).")
    L.append("")
    for line in opening["lines"]:
        L.append(f"* {line}")
    L.append("")
    L.append(f"* **Selected model (lowest passing k): {opening['selected_model'] or 'NONE YET'}**")
    notes = sorted({n for r in runs for n in (r.get("schema_notes") or [])})
    miss = sorted({m for r in runs
                   for m in ((r.get("complexity") or {}).get("coordinates_missing") or [])})
    unav = sorted({f"{k}: {v['unavailable']}" for r in runs
                   for k, v in (("coverage_power", r.get("coverage_power") or {}),
                                ("complexity", r.get("complexity") or {}))
                   if isinstance(v, dict) and v.get("unavailable")})
    if notes or miss or unav:
        L.append("")
        L.append("## Schema notes (disclosed, not corrected here)")
        L.append("")
        for n in notes:
            L.append(f"* {n}")
        if miss:
            L.append(f"* FP coordinates named by `fp_prior_moments` but absent from the "
                     f"by-chain files (deterministic or not retained): {', '.join(miss)}")
        for u in unav:
            L.append(f"* {u}")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="directories holding RUN_<M>_<fam>_s<seed>.json (+ sidecars)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--pack-dir", default=None,
                    help="fallback directory for the packs named in the run JSONs")
    a = ap.parse_args(argv)

    paths = []
    for d in a.runs:
        if os.path.isfile(d):
            paths.append(d)
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            if p.endswith("_ppc.json"):
                continue
            try:
                j = json.load(open(p))
            except Exception:
                continue
            if isinstance(j, dict) and {"ladder", "thresholds", "perz_recovery"} <= set(j):
                paths.append(p)
    if not paths:
        raise SystemExit("no ladder run JSONs found under: " + " ".join(a.runs))

    runs = [analyse_run(p, a.pack_dir) for p in paths]
    roll = rollup(runs)
    opening = opening_rule(roll["by_model"])
    out = dict(
        role=("VALIDATION-ONLY model table for the sealed FP-model ladder "
              "(predeclaration 3112022a); gate = perz_gate CRIT v2 unchanged; "
              "decides nothing beyond the predeclared rules"),
        criteria=dict(ebfmi_min=EBFMI_MIN, omega_bias_pct_max=OMEGA_BIAS_PCT_MAX,
                      fp_truth_ratio_band=list(FP_TRUTH_RATIO_BAND),
                      coverage_scales=list(COVERAGE_SCALES),
                      pe_exclusion_excess_nats=PE_EXCESS_NATS),
        n_runs=len(runs), runs=runs, rollup=roll, opening_rule=opening)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)) or ".", exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=1, default=str)
    open(a.out_md, "w").write(render_md(runs, roll, opening))

    for r in runs:
        print(f"{r['gate_status']:4s} {r['file']:44s} {r['model']} {r['family']} s{r['seed']} "
              f"hard={r['hard_flags']} soft={r['soft_flags']}")
    print("\nby model:", {m: v["status"] for m, v in roll["by_model"].items()})
    print("OPENING RULE (§6): M_{k+1} is run iff M_k fails")
    for line in opening["lines"]:
        print("  " + line)
    print("  selected:", opening["selected_model"] or "NONE YET")
    print(f"\nwrote {a.out_json} and {a.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
