"""wg_predictive_breakdown.py — DETERMINISTIC read-out of the FROZEN Paper-1 low-z model
(B + phi_2LPT + C1nsadd + M1CUT) for the Lya-WG-facing S/N predictive diagnostic.

PI ruling 2026-09-14b §5 / §14.  NOTHING here fits, samples or corrects anything.  The
only operation performed is the SAME deterministic evaluation of the frozen fold at the
stored posterior-median draw that the production runners already perform for their
`predictive_marginals` block (validation/real_c1/run_real_c1.py lines 368-397 and
validation/fp_ladder/run_ladder.py lines 254-282).  That code path is COPIED VERBATIM
below (marked) so that the finer (S/N x coarse-K) and (S/N x N-hat group) breakdowns are
by construction the same object as the stored all-K S/N marginal; `verify_stored_snr`
asserts the equality to 1e-10 before any finer number is trusted.

Blinding: this module returns RATIOS mu/obs and FRACTIONS only.  No absolute counts and
no dN/dX ever leave it (`assert_ratios_only`).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

SCRATCH = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
REAL_RUNS = os.path.join(SCRATCH, "real_c1", "runs", "RUN_REALC1_*.json")
MOCK_DIR = os.path.join(SCRATCH, "final", "runs", "B-phi2lpt-C1nsadd-M1CUTJ8")
REAL_INPUTS = os.path.join(SCRATCH, "real_c1_inputs")

MOCK_FAMILIES = ("2lpt0", "london0", "saclay0")

# VERBATIM from validation/real_c1/run_real_c1.py (NHAT_GROUPS, line 57)
NHAT_GROUPS = (("19.5_20.0", 19.5, 20.0), ("20.0_20.3", 20.0, 20.3), ("20.3_22.4", 20.3, 22.4))


# ----------------------------------------------------------------------------------
# run discovery
# ----------------------------------------------------------------------------------
def real_run_jsons():
    return sorted(glob.glob(REAL_RUNS))


def mock_run_jsons(family):
    return sorted(glob.glob(os.path.join(MOCK_DIR, f"RUN_*_{family}_*J8j*.json")))


def _sidecar(run_json, suffix):
    return run_json[:-5] + suffix


def _pred_marg(run_json):
    j = json.load(open(run_json))
    return j.get("predictive_marginals") or j["diagnostics"]["predictive_marginals"]


# ----------------------------------------------------------------------------------
# the frozen fold, evaluated at the stored posterior-median draw
# ----------------------------------------------------------------------------------
def _median_draw_index(bychain):
    """`idx_med` of the runners: argsort(theta_level)[n//2] on the CHAIN-FLATTENED samples.

    numpyro's get_samples(group_by_chain=False) is the (chains, draws) array flattened in
    C order, i.e. chains concatenated in order — hence reshape(-1) of the by-chain array.
    """
    tl = np.asarray(bychain["theta_level"], float).reshape(-1)
    return int(np.argsort(tl)[len(tl) // 2])


def fold_at_median(run_json, fixed):
    """Return (mu3, obs3, kz_to_K, nhat_centres) for one run.

    `fixed` = dict(pack=..., mg=..., mg_key=..., c=..., mu_extra=array|None).
    mu3[c, k, s] = the frozen model's posterior-median predictive expectation; obs3 = the
    pack's observed counts.  Both stay INSIDE this process: only ratios are returned by the
    public table builders.
    """
    import jax  # noqa: F401  (kept so the einsum path is bit-identical to the runners')
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors

    pk = load_pack(fixed["pack"])
    consts, Mg = build_cc_tensors(pk)

    Mg_fixed = None
    if fixed.get("mg"):
        Mg_fixed = np.asarray(np.load(fixed["mg"], allow_pickle=True)[fixed.get("mg_key", "Mg")], float)
    C_fixed = None
    if fixed.get("c"):
        cf = np.load(fixed["c"], allow_pickle=True)
        C_fixed = np.asarray(cf["C_fixed_bks"], float) if "C_fixed_bks" in cf.files else np.asarray(cf["C_fixed"], float)
    mu_extra = fixed.get("mu_extra")

    bych = np.load(_sidecar(run_json, "_bychain.npz"), allow_pickle=True)
    fdr = np.load(_sidecar(run_json, "_fdraws.npz"), allow_pickle=True)
    idx_med = _median_draw_index(bych)

    f_draws = np.asarray(fdr["f"], float)                       # (D, B, Kf) = exp(theta_pop)
    t_draws = np.asarray(bych["t"], float).reshape(-1, np.asarray(bych["t"]).shape[-1])
    lam = np.asarray(bych["lam_fp"], float)
    lam_draws = lam.reshape(-1, lam.shape[-2], lam.shape[-1])

    # --- VERBATIM run_real_c1.py lines 370-386 / run_ladder.py lines 255-273 ---------
    # (`f_med = jnp.exp(th_med)` with th_med = theta_pop[idx_med] is the stored site
    #  `f` = deterministic("f", exp(theta_pop)) — fp_ladder.py lines 185-186 — so the
    #  saved f-draws ARE exp(theta_pop) and are used directly here.)
    f_med = jnp.asarray(f_draws[idx_med])
    t_med = jnp.asarray(t_draws[idx_med])
    lf_med = jnp.asarray(lam_draws[idx_med])
    Mg_use = Mg if Mg_fixed is None else jnp.asarray(Mg_fixed)
    if C_fixed is None:
        raise SystemExit("the frozen model of record always carries a fixed C (C1nsadd); refusing the psi_c branch")
    if np.asarray(C_fixed).ndim == 2:
        tpx = jnp.einsum("skcb,sb,bk->cks", Mg_use, jnp.asarray(C_fixed),
                         consts.g_bk * f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
    else:
        tpx = jnp.einsum("skcb,bks,bk->cks", Mg_use, jnp.asarray(C_fixed),
                         f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
    fpx = (consts.fp_w * consts.fp_ell_eff * (1.0 - consts.fp_eta_c)[:, None, None]
           * jnp.exp(t_med[consts.kz_to_K])[None, :, None] * lf_med[:, None, :] * consts.fp_E[None, :, :])
    if mu_extra is not None:
        fpx = fpx + jnp.asarray(mu_extra)
    obs3 = np.asarray(pk.counts, float)
    mu3 = np.asarray(tpx) + np.asarray(fpx)
    # --- end verbatim ---------------------------------------------------------------

    nh = np.asarray(pk.nhat_edges, float)
    return mu3, obs3, np.asarray(consts.kz_to_K, int), 0.5 * (nh[:-1] + nh[1:])


# ----------------------------------------------------------------------------------
# frozen-input resolution (real + each mock family), reproduced from the run JSONs
# ----------------------------------------------------------------------------------
def real_fixed():
    return dict(pack=os.path.join(REAL_INPUTS, "C1_pack.npz"),
                mg=os.path.join(REAL_INPUTS, "Mg_B_real.npz"), mg_key="Mg",
                c=os.path.join(REAL_INPUTS, "C_C1nsadd_real.npz"),
                mu_extra=np.asarray(np.load(os.path.join(REAL_INPUTS, "mu_extra_P6bcal_real.npz"),
                                            allow_pickle=True)["mu_extra"], float))


def mock_fixed(run_json):
    """Rebuild the mock run's fixed objects EXACTLY as the run's own argv recorded them,
    including the truth-pinned P6b term taken from the census (`--fix P`, run_ladder.py
    lines 120-123: mu_extra = census['host_17p2_19p0'])."""
    j = json.load(open(run_json))
    argv = list(j["run_config"]["argv"])

    def opt(name):
        return argv[argv.index(name) + 1] if name in argv else None

    fix = [x for x in (opt("--fix") or "").split(",") if x]
    if fix != ["P"]:
        raise SystemExit(f"unexpected --fix {fix!r} in {run_json} (expected ['P'])")
    census = opt("--census")
    mu_extra = np.asarray(np.load(census, allow_pickle=True)["host_17p2_19p0"], float)
    return dict(pack=opt("--pack"), mg=opt("--mg-fixed-file"), mg_key=opt("--mg-fixed-key") or "Mg",
                c=opt("--c-fixed-file"), mu_extra=mu_extra, census=census)


# ----------------------------------------------------------------------------------
# the breakdowns — RATIOS ONLY
# ----------------------------------------------------------------------------------
def breakdown_one(run_json, fixed):
    """All ratio tables for one run.  mu/obs by S/N; by (S/N x coarse K); by (S/N x N-hat
    group); by N-hat.  Cells with zero observed counts are returned as NaN (never 0/0)."""
    mu3, obs3, kz, cc = fold_at_median(run_json, fixed)

    def ratio(m, o):
        m = np.asarray(m, float); o = np.asarray(o, float)
        return np.where(o > 0, m / np.where(o > 0, o, 1.0), np.nan)

    KK = int(kz.max()) + 1
    by_snr = ratio(mu3.sum((0, 1)), obs3.sum((0, 1)))
    by_snr_K = np.stack([ratio(mu3[:, kz == K, :].sum((0, 1)), obs3[:, kz == K, :].sum((0, 1)))
                         for K in range(KK)])                                  # (KK, S)
    grp = {}
    for name, lo, hi in NHAT_GROUPS:
        m = (cc >= lo) & (cc < hi)
        grp[name] = ratio(mu3[m].sum((0, 1)), obs3[m].sum((0, 1)))             # (S,)
    by_nhat = ratio(mu3.sum((1, 2)), obs3.sum((1, 2)))
    live_s = obs3.sum((0, 1)) > 0
    return dict(by_snr=by_snr, by_snr_K=by_snr_K, by_snr_nhat_group=grp,
                by_nhat=by_nhat, live_snr=live_s, nhat_centres=cc, n_kk=KK)


def verify_stored_snr(run_json, fixed, atol=1e-10):
    """The gate: our recomputed ALL-K S/N marginal must equal the run's STORED
    `mu_over_obs_by_snr` to `atol`.  Returns (max_abs_diff, n_compared)."""
    b = breakdown_one(run_json, fixed)
    stored = np.asarray(_pred_marg(run_json)["mu_over_obs_by_snr"], float)
    ours = np.where(np.isnan(b["by_snr"]), 0.0, b["by_snr"])   # the runners emit m/max(o,1) -> 0 on empty
    d = np.abs(ours - stored)
    if not np.all(d <= atol):
        raise AssertionError(f"recomputed S/N marginal differs from stored by {d.max():.3e} ({run_json})")
    return float(d.max()), int(stored.size)


def median_over_runs(run_jsons, fixed_for):
    """Element-wise median over runs of every ratio table (the SEC-1b convention)."""
    bs = [breakdown_one(p, fixed_for(p)) for p in run_jsons]
    out = dict(n_runs=len(bs), nhat_centres=bs[0]["nhat_centres"], n_kk=bs[0]["n_kk"],
               live_snr=np.all([b["live_snr"] for b in bs], axis=0))
    for k in ("by_snr", "by_nhat"):
        out[k] = np.nanmedian(np.stack([b[k] for b in bs]), axis=0)
    out["by_snr_K"] = np.nanmedian(np.stack([b["by_snr_K"] for b in bs]), axis=0)
    out["by_snr_nhat_group"] = {g: np.nanmedian(np.stack([b["by_snr_nhat_group"][g] for b in bs]), axis=0)
                                for g, _, _ in NHAT_GROUPS}
    return out


def assert_ratios_only(payload, lo=0.0, hi=10.0):
    """Blinding guard: every number in the delivered payload must be a ratio/fraction in
    (lo, hi) or NaN — an absolute count or a dN/dX value would fail."""
    bad = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, (list, tuple, np.ndarray)):
            a = np.asarray(x, float)
            for i, v in np.ndenumerate(a):
                if np.isfinite(v) and not (lo <= v <= hi):
                    bad.append((path + str(list(i)), float(v)))
        elif isinstance(x, (int, float, np.floating, np.integer)):
            v = float(x)
            if np.isfinite(v) and not (lo <= v <= hi):
                bad.append((path, v))

    walk(payload)
    if bad:
        raise AssertionError(f"non-ratio values in the delivered payload: {bad[:5]}")
    return True


def coarse_K_labels(pack_path):
    """Human labels for the coarse-z blocks, from the pack's fine-z edges and kz_to_K."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    pk = load_pack(pack_path)
    consts, _ = build_cc_tensors(pk)
    kz = np.asarray(consts.kz_to_K, int)
    ze = np.asarray(pk.zf_edges, float)
    return [f"K{K}: {ze[np.where(kz == K)[0][0]]:.2f}–{ze[np.where(kz == K)[0][-1] + 1]:.2f}"
            for K in range(int(kz.max()) + 1)]
