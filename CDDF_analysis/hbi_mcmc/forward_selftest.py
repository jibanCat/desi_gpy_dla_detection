# -*- coding: utf-8 -*-
"""forward_selftest.py — the pure forward-model TRUTH-FOLD self-test (no sampling).

The decisive, cheap gate on the Model A / NUTS route.  Take a pack's OWN truth
f(N, z) (from ``truth_counts``), fold it through that pack's OWN
kernel / completeness / g / dX / FP machinery (``forward.build_consts`` +
``forward.fold_mu``) at the truth-equivalent parameter point, and compare the
predicted expected counts ``mu`` against the pack's ACTUAL observed ``counts``.

If the forward model is faithful this must reproduce the counts to within
Poisson noise.  It needs NO MCMC: any failure here is upstream of NUTS and no
amount of sampling can fix it.

Parameter point used for the fold ("truth-equivalent"):

    theta_pop  = log f_truth,  f_truth[b,k] = truth_counts[b,k] / (dX_tot[k] dN_b)
    psi_c      = 0             (completeness at the Jeffreys molly point surface)
    psi_k_delta= 0             (response coefficients at their fitted point)
    log_t      = 0             (transfer factors at their prior centre)
    lam_fp     = fp_counts / ell_eff   (the loa-0 FP point estimate)

TRUTH-SUPPORT EXTENSION (``--truth-floor``)
-------------------------------------------
The pack's ``truth_counts`` is truncated at the bottom of the reporting grid
(N_true >= nhat_edges[0] = 19.5); the extractor's truth cut carries the SAME
19.5 floor as the reporting grid.  But the forward response has a POSITIVE
mean bias (~+0.1 dex) and a ~0.2 dex width, so the observed n-hat bins just
above 19.5 are fed overwhelmingly by TRUE systems BELOW 19.5 that the pack
simply does not carry.  Folding the truncated truth therefore under-predicts
the bottom n-hat bins by construction -- the same CLASS of one-sided-support
bug as B16.  ``--truth-floor 18.5`` (say) extends the true-N grid downward with
a power law fitted to the pack's own truth over a clean, un-truncated window
and re-folds; the difference between the two runs IS the size of the
truncation defect.

Usage
-----
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m CDDF_analysis.hbi_mcmc.forward_selftest \
        --pack /path/modelA_pack_2lpt0.npz [--out out.json] \
        [--truth-floor 18.0] [--fit-lo 19.8 --fit-hi 20.8] [--no-fp]

MOCKS ONLY (truth_counts exists only for mocks); refuses real-LOA packs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np

__all__ = ["truth_f", "extend_pack_truth", "selftest", "ratio_tables"]


# --------------------------------------------------------------------------
# truth -> f(N, z)
# --------------------------------------------------------------------------
def truth_f(pack):
    """f_truth[b, k] = truth_counts[b, k] / (dX_tot[k] * dN_b)  (per dex, per dX).

    dX_tot[k] = sum_s dX[k, s]: the truth histogram is NOT stratified by the
    fold's SNR axis, so the fold's per-stratum dX[k, s] re-allocates the truth
    across strata in proportion to pathlength (checked separately against
    ``truth_counts_bks``).
    """
    tc = np.asarray(pack.truth_counts, float)             # (B, Kf)
    dX_tot = np.asarray(pack.dX, float).sum(axis=1)       # (Kf,)
    dN = np.diff(np.asarray(pack.ntrue_edges, float))     # (B,)
    denom = dX_tot[None, :] * dN[:, None]
    f = np.zeros_like(tc)
    ok = denom > 0
    f[ok] = tc[ok] / denom[ok]
    return f


def _fit_truth_powerlaw(pack, fit_lo, fit_hi):
    """Least-squares log10 f = a + s * (N - N0) over the clean window, per z-bin.

    Returns (slope_per_dex, log_f_at_grid_bottom) using the POOLED-in-N,
    per-z-bin fit so the downward extension keeps the truth's own z shape.
    """
    f = truth_f(pack)
    ntrue = np.asarray(pack.ntrue_edges, float)
    Nc = 0.5 * (ntrue[:-1] + ntrue[1:])
    sel = (Nc >= fit_lo - 1e-9) & (Nc < fit_hi - 1e-9)
    if sel.sum() < 3:
        raise ValueError("truth power-law fit window has < 3 bins")
    x = Nc[sel]
    slopes, intercepts = [], []
    for k in range(f.shape[1]):
        y = f[sel, k]
        good = y > 0
        if good.sum() < 3:
            slopes.append(np.nan)
            intercepts.append(np.nan)
            continue
        A = np.vstack([np.ones(good.sum()), x[good] - Nc[0]]).T
        coef, *_ = np.linalg.lstsq(A, np.log(y[good]), rcond=None)
        intercepts.append(coef[0])
        slopes.append(coef[1])
    return np.asarray(slopes), np.asarray(intercepts), Nc[0]


def extend_pack_truth(pack, truth_floor, fit_lo, fit_hi):
    """Return (pack2, f2) whose TRUE-N grid runs down to ``truth_floor``.

    Only the ntrue axis is extended -- the OBSERVED n-hat axis (and therefore
    ``counts``) is untouched, so the comparison stays like-for-like.  The
    extension's f is the power law fitted to the pack's own truth over
    [fit_lo, fit_hi) per z bin.

    NOTE the pack schema requires ntrue_edges == nhat_edges (v1); this function
    deliberately breaks that for the DIAGNOSTIC only and never saves the pack.
    """
    ntrue = np.asarray(pack.ntrue_edges, float)
    step = float(np.diff(ntrue)[0])
    n_extra = int(round((ntrue[0] - truth_floor) / step))
    if n_extra <= 0:
        return pack, truth_f(pack)
    lo_edges = ntrue[0] - step * np.arange(n_extra, 0, -1)
    new_edges = np.concatenate([lo_edges, ntrue])
    slopes, intercepts, N0 = _fit_truth_powerlaw(pack, fit_lo, fit_hi)
    new_Nc = 0.5 * (new_edges[:-1] + new_edges[1:])
    f_old = truth_f(pack)
    Kf = f_old.shape[1]
    f_new = np.zeros((len(new_Nc), Kf))
    f_new[n_extra:, :] = f_old
    for k in range(Kf):
        if not np.isfinite(slopes[k]):
            continue
        f_new[:n_extra, k] = np.exp(
            intercepts[k] + slopes[k] * (new_Nc[:n_extra] - N0))
    pack2 = dataclasses.replace(pack, ntrue_edges=new_edges,
                                truth_counts=None, truth_counts_bks=None)
    return pack2, f_new


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------
def selftest(pack, f=None, *, use_fp=True, psi_c=None, resp_clamp="both"):
    """Fold the truth through the pack's own machinery; return mu and counts."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu

    consts = build_consts(pack, resp_clamp=resp_clamp,
                          allow_unclamped_response=(resp_clamp == "off"))
    if f is None:
        f = truth_f(pack)
    f = np.asarray(f, float)
    theta = np.log(np.clip(f, 1e-300, None))
    C, S = consts.n_c, consts.n_s
    if psi_c is None:
        psi_c = np.zeros((S, consts.n_molly))
    lam_fp = (np.asarray(pack.fp_counts, float) / float(pack.fp_ell_eff)
              if use_fp else np.zeros((C, S)))
    mu = np.asarray(fold_mu(jnp.asarray(theta), jnp.asarray(psi_c),
                            jnp.zeros((2, consts.n_sr, consts.n_zr)),
                            jnp.zeros(consts.n_kk), jnp.asarray(lam_fp),
                            consts))
    mu_fp = np.asarray(
        float(pack.fp_w_sightline_ratio) * lam_fp[:, None, :]
        * np.asarray(pack.fp_E_alloc, float)[None, :, :])
    return dict(mu=mu, mu_fp=mu_fp, mu_sig=mu - mu_fp,
                counts=np.asarray(pack.counts, float), consts=consts, f=f)


def ratio_tables(res, pack):
    """Per-cell / marginal mu-over-counts ratios + Poisson z-scores."""
    mu = res["mu"]
    obs = res["counts"]
    nhat = np.asarray(pack.nhat_edges, float)
    zf = np.asarray(pack.zf_edges, float)
    dxpos = np.asarray(pack.dX, float) > 0

    def _z(m, o):
        return (o - m) / np.sqrt(np.maximum(m, 1e-12))

    mask3 = np.broadcast_to(dxpos[None, :, :], mu.shape)
    mu_m = np.where(mask3, mu, 0.0)
    obs_m = np.where(mask3, obs, 0.0)

    out = {
        "total": dict(mu=float(mu_m.sum()), obs=float(obs_m.sum()),
                      ratio=float(mu_m.sum() / max(obs_m.sum(), 1e-30)),
                      z=float(_z(mu_m.sum(), obs_m.sum()))),
        "by_nhat": [], "by_z": [], "by_snr": [],
    }
    for c in range(mu.shape[0]):
        m, o = float(mu_m[c].sum()), float(obs_m[c].sum())
        out["by_nhat"].append(dict(
            lo=float(nhat[c]), hi=float(nhat[c + 1]), mu=m, obs=o,
            ratio=(m / o if o > 0 else float("nan")), z=float(_z(m, o))))
    for k in range(mu.shape[1]):
        m, o = float(mu_m[:, k].sum()), float(obs_m[:, k].sum())
        out["by_z"].append(dict(
            lo=float(zf[k]), hi=float(zf[k + 1]), mu=m, obs=o,
            ratio=(m / o if o > 0 else float("nan")), z=float(_z(m, o))))
    for s in range(mu.shape[2]):
        m, o = float(mu_m[:, :, s].sum()), float(obs_m[:, :, s].sum())
        out["by_snr"].append(dict(
            s=s, mu=m, obs=o,
            ratio=(m / o if o > 0 else float("nan")), z=float(_z(m, o))))
    return out


def print_tables(tab, title=""):
    print(f"\n=== {title} ===")
    t = tab["total"]
    print(f"TOTAL   mu={t['mu']:12.1f}  obs={t['obs']:12.1f}  "
          f"ratio={t['ratio']:.4f}  z={t['z']:+.1f}")
    print(" n-hat bin        mu         obs      mu/obs      z")
    for r in tab["by_nhat"]:
        print(f" [{r['lo']:.1f},{r['hi']:.1f})  {r['mu']:10.2f} {r['obs']:10.0f} "
              f"  {r['ratio']:8.4f}  {r['z']:+8.1f}")
    print("   z bin          mu         obs      mu/obs      z")
    for r in tab["by_z"]:
        print(f" [{r['lo']:.1f},{r['hi']:.1f})  {r['mu']:10.2f} {r['obs']:10.0f} "
              f"  {r['ratio']:8.4f}  {r['z']:+8.1f}")


# --------------------------------------------------------------------------
# structural probes (the hypothesis battery)
# --------------------------------------------------------------------------
def structural_probes(pack):
    """Cheap, decisive structural checks on the fold's ingredients."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K

    consts = build_consts(pack, allow_unclamped_response=True,
                          resp_clamp=("both" if pack.resp_N_fit_range is not None
                                      else "off"))
    probes = {}
    probes["resp_clamp"] = consts.resp_clamp
    probes["has_resp_N_fit_range"] = pack.resp_N_fit_range is not None

    # (c) kernel row mass: sum_c K[s,K,c,b] -- how much probability the observed
    #     n-hat window RETAINS per true-N bin (1.0 = nothing lost off the grid).
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    rowmass = K.sum(axis=2)                       # (S, KK, B)
    probes["kernel_rowmass_min"] = float(rowmass.min())
    probes["kernel_rowmass_max"] = float(rowmass.max())
    probes["kernel_rowmass_by_b"] = rowmass.mean(axis=(0, 1)).tolist()

    # (a) Jacobian / bin-width: dN_b uniform?
    dN = np.diff(np.asarray(pack.ntrue_edges, float))
    probes["dN_b_unique"] = np.unique(np.round(dN, 12)).tolist()

    # (d) g surface: occupancy-weighted mean over z per molly cell (1.0 if the
    #     z-shape is level-preserving and hence NOT double-counting completeness)
    g = np.asarray(pack.g_grid, float)
    occ = np.asarray(pack.g_occupancy, float)
    w = occ / np.maximum(occ.sum(axis=1, keepdims=True), 1e-30)
    probes["g_occ_weighted_mean_by_cell"] = (g * w).sum(axis=1).tolist()

    # (e) truth support: is truth_counts truncated at the grid bottom?
    ntrue = np.asarray(pack.ntrue_edges, float)
    probes["ntrue_lo"] = float(ntrue[0])
    probes["nhat_lo"] = float(np.asarray(pack.nhat_edges, float)[0])
    probes["truth_floor_equals_grid_floor"] = bool(
        abs(float(ntrue[0]) - float(np.asarray(pack.nhat_edges, float)[0])) < 1e-9)

    # (f) index maps
    probes["kz_to_K"] = np.asarray(pack.kz_to_K).tolist()
    probes["s_to_sresp"] = np.asarray(consts.s_to_sresp).tolist()
    probes["K_to_zresp"] = np.asarray(consts.K_to_zresp).tolist()
    probes["b_to_cell"] = np.asarray(consts.b_to_cell).tolist()

    # (g) dX zero pattern vs counts
    dX = np.asarray(pack.dX, float)
    cnt = np.asarray(pack.counts, float).sum(axis=0)
    probes["n_zero_dX_cells"] = int((dX == 0).sum())
    probes["counts_in_zero_dX"] = float(cnt[dX == 0].sum())

    # truth vs counts totals (support asymmetry smoking gun)
    probes["truth_total"] = float(np.asarray(pack.truth_counts, float).sum()) \
        if pack.truth_counts is not None else None
    probes["counts_total"] = float(np.asarray(pack.counts, float).sum())

    # truth_counts_bks vs the dX-proportional re-allocation the fold implies
    if pack.truth_counts_bks is not None:
        tb = np.asarray(pack.truth_counts_bks, float)
        tc = np.asarray(pack.truth_counts, float)
        share = dX / np.maximum(dX.sum(axis=1, keepdims=True), 1e-30)   # (k,s)
        alloc = tc[:, :, None] * share[None, :, :]
        num = np.abs(alloc - tb).sum()
        probes["truth_strat_realloc_L1_frac"] = float(num / max(tb.sum(), 1e-30))
    return probes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _git():
    """FULL 40-char HEAD SHA (+ a dirty flag).

    This used to be ``rev-parse --short HEAD``, and the artifact it stamped
    (``rung9_forward_selftest.json``) carried ``code_commit: 'b76ded7'``.
    Abbreviated SHAs are a known defect class in this repo -- the provenance
    audit's ORPHANED class -- and that stamp was an instance of it: at b76ded7
    ``forward_selftest.py`` did not yet exist (it was added at 85ddd95), so the
    stamp named a commit at which the routine could not have run.  A 40-char
    SHA is checkable with ``git cat-file -e <sha>:<routine>``; a 7-char one
    invites exactly the mis-resolution that happened.

    NOTE the split from the dirty probe (2026-07-29).  A ``-dirty`` SUFFIX
    makes ``code_commit`` unusable with ``git cat-file -e <sha>:<routine>``,
    which is the entire reason the 40-char SHA is stamped.  Dirt is a separate
    BOOLEAN FIELD now, reported alongside the SCOPE it was measured over.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=here, text=True).strip()
    except Exception:
        return "unknown"


# the dirty probe is PATH-SCOPED; this string travels WITH the flag so no
# reader can upgrade it into a claim about the whole working tree.
_DIRTY_SCOPE = ("uncommitted changes under CDDF_analysis/hbi_mcmc/ ONLY -- "
                "this is NOT a whole-tree cleanliness claim")


def _git_dirty():
    """True if CDDF_analysis/hbi_mcmc/ has uncommitted changes.

    Path-scoped by design (the rest of the repo does not affect this routine's
    result), and therefore NOT evidence of a clean tree.  Unknown -> True:
    fail closed, an unprobeable tree is treated as dirty.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", here],
            cwd=here, text=True).strip())
    except Exception:
        return True


def _stamp_fields(mocks):
    """The provenance fields of the aggregate artifact.  Touches no pack, so
    it is directly testable."""
    return {
        "routine": "CDDF_analysis/hbi_mcmc/forward_selftest.py",
        "entry_point": "aggregate_report / --mock NAME=PATH",
        "date": time.strftime("%Y-%m-%d"),
        "code_commit": _git(),
        "code_dirty": bool(_git_dirty()),
        "code_dirty_scope": _DIRTY_SCOPE,
        "scope": (f"MOCK ONLY ({' / '.join(n for n, _ in mocks)}). "
                  f"No real-survey values."),
        "rederive": ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                     "MKL_NUM_THREADS=1 python -m "
                     "CDDF_analysis.hbi_mcmc.forward_selftest "
                     + " ".join(f"--mock {n}={p}" for n, p in mocks)
                     + " --out CDDF_analysis/hbi_mcmc/"
                       "rung9_forward_selftest.json"),
    }


def aggregate_report(mocks, *, clamps=("off", "both", "hi"), use_fp=True):
    """The MULTI-MOCK, MULTI-CLAMP report -- the committed routine behind
    ``rung9_forward_selftest.json``.

    ``mocks`` is a list of ``(name, pack_path)``.  This existed only as an
    uncommitted scratch driver, which is how the artifact came to carry a
    hand-written 7-char ``code_commit`` naming a commit at which this file did
    not exist (the ORPHANED provenance class).  It is committed now so the
    artifact has a `rederive` line that actually runs.

    MOCKS ONLY.
    """
    from CDDF_analysis.hbi_mcmc.pack import load_pack

    out_mocks, closes, pads = {}, {}, {}
    for name, path in mocks:
        assert "main_dark" not in path, "REAL-LOA guard: mock packs only"
        pack = load_pack(path)
        assert "loa_main_dark" not in json.dumps(pack.provenance or {}), \
            "REAL-LOA guard (provenance)"
        if pack.truth_counts is None:
            raise SystemExit(f"{name}: pack carries no truth_counts")
        entry = {"pack": os.path.basename(path),
                 "n_pad_bins": int(pack.n_pad_bins),
                 "probes": structural_probes(pack)}
        for clamp in clamps:
            tab = ratio_tables(selftest(pack, use_fp=use_fp, resp_clamp=clamp),
                               pack)
            entry[f"clamp_{clamp}"] = tab
            if clamp == "both":
                closes[name] = _closure_verdict(tab, 5.0, 5.0, 3.0)
        pads[name] = int(pack.n_pad_bins)
        out_mocks[name] = entry

    return {
        **_stamp_fields(mocks),
        "what": ("pure forward-model truth-fold self-test: the pack's own "
                 "truth f(N,z) folded through the pack's own kernel/"
                 "completeness/g/dX/FP machinery at the truth-equivalent "
                 "parameter point, vs the pack's own observed counts. NO "
                 "SAMPLING."),
        "mocks": out_mocks,
        "closure_verdicts": closes,
        "n_pad_bins": pads,
        "verdict": {
            "D1_basis_pad_low_N": (
                "OPEN — needs a re-extracted basis-padded pack"
                if any(v == 0 for v in pads.values())
                else "a basis-padded pack is in use"),
            "D2_response_extrapolation_high_N":
                "FIXED in-code (resp_clamp, default 'both')",
            "forward_model_closes": bool(
                closes and all(v["closes"] for v in closes.values())),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack")
    ap.add_argument("--mock", action="append", metavar="NAME=PATH",
                    help="AGGREGATE mode: repeat once per mock. Emits the "
                         "multi-mock / multi-clamp report "
                         "(rung9_forward_selftest.json). Mutually exclusive "
                         "with --pack.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--truth-floor", type=float, default=None,
                    help="extend the TRUE-N grid down to this log N_HI using a "
                         "power law fitted to the pack's own truth")
    ap.add_argument("--fit-lo", type=float, default=19.8)
    ap.add_argument("--fit-hi", type=float, default=20.8)
    ap.add_argument("--no-fp", action="store_true")
    ap.add_argument("--resp-clamp", default=None, choices=["both", "hi", "off"],
                    help="response covariate-range guard (finding D2); default "
                         "= 'both' when the pack carries resp_N_fit_range, "
                         "'off' otherwise (pre-fix reproduction)")
    ap.add_argument("--require-closure", action="store_true",
                    help="EXIT NONZERO unless the truth-fold closes within the "
                         "tolerances below. Without this the command is a REPORT "
                         "and always exits 0 -- so `selftest ... || exit 1` in a "
                         "batch script is NOT a gate. Any script that spends "
                         "sampler time must pass this.")
    ap.add_argument("--require-basis-pad", action="store_true",
                    help="EXIT NONZERO unless the pack's true-N basis is padded "
                         "BELOW the reporting floor (n_pad_bins > 0). Finding "
                         "D1: an unpadded pack cannot arithmetically reproduce "
                         "its own lowest observed bins, so no batch script that "
                         "spends sampler time may run on one.")
    ap.add_argument("--min-pad-bins", type=int, default=1)
    ap.add_argument("--max-abs-z-total", type=float, default=5.0)
    ap.add_argument("--max-abs-z-bin", type=float, default=5.0)
    ap.add_argument("--max-chi2-dof", type=float, default=3.0)
    a = ap.parse_args(argv)

    if a.mock:
        if a.pack:
            raise SystemExit("--mock and --pack are mutually exclusive")
        pairs = []
        for spec in a.mock:
            if "=" not in spec:
                raise SystemExit(f"--mock expects NAME=PATH, got {spec!r}")
            n, _, p = spec.partition("=")
            pairs.append((n, p))
        rep = aggregate_report(pairs)
        for n, e in rep["mocks"].items():
            print_tables(e["clamp_both"], f"{n} (resp_clamp=both, "
                                          f"n_pad_bins={e['n_pad_bins']})")
        print(f"\n[selftest] verdict: {json.dumps(rep['verdict'])}")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(rep, fh, indent=1)
            print(f"[selftest] wrote {a.out}")
        if a.require_closure and not rep["verdict"]["forward_model_closes"]:
            print("\n[selftest] FORWARD MODEL DOES NOT CLOSE -- refusing.",
                  file=sys.stderr)
            raise SystemExit(3)
        return rep

    if not a.pack:
        raise SystemExit("--pack is required (or use --mock NAME=PATH)")

    from CDDF_analysis.hbi_mcmc.pack import load_pack
    assert "main_dark" not in a.pack, "REAL-LOA guard: mock packs only"
    pack = load_pack(a.pack)
    assert "loa_main_dark" not in json.dumps(pack.provenance or {}), \
        "REAL-LOA guard (provenance)"
    if pack.truth_counts is None:
        raise SystemExit("pack carries no truth_counts — self-test needs a mock")

    # --- the BASIS-PAD gate (finding D1), before any other work -----------
    n_pad = int(getattr(pack, "n_pad_bins", 0))
    if a.require_basis_pad and n_pad < a.min_pad_bins:
        raise SystemExit(
            f"[selftest] REFUSING: pack has n_pad_bins={n_pad} "
            f"(< {a.min_pad_bins}). The true-N basis stops at the reporting "
            f"floor {float(np.asarray(pack.ntrue_edges, float)[0]):.2f}, so "
            f"the truth cannot feed the lowest observed bins and the fold "
            f"cannot close at ANY parameter value (finding D1). Re-extract a "
            f"basis-padded pack (schema v1.1 permits a DOWNWARD pad).")

    t0 = time.time()
    probes = structural_probes(pack)
    clamp = a.resp_clamp or ("both" if pack.resp_N_fit_range is not None else "off")
    res = selftest(pack, use_fp=not a.no_fp, resp_clamp=clamp)
    tab = ratio_tables(res, pack)
    print_tables(tab, f"baseline truth-fold (truth floor {probes['ntrue_lo']:.1f}, "
                      f"resp_clamp={clamp})")

    out = dict(pack=os.path.basename(a.pack), probes=probes, baseline=tab,
               resp_clamp=clamp, n_pad_bins=n_pad,
               provenance=dict(routine="CDDF_analysis/hbi_mcmc/forward_selftest.py",
                               code_commit=_git(),
                               code_dirty=bool(_git_dirty()),
                               code_dirty_scope=_DIRTY_SCOPE,
                               date=time.strftime("%Y-%m-%d"),
                               rederive=("python -m CDDF_analysis.hbi_mcmc."
                                         f"forward_selftest --pack {a.pack}")))

    if a.truth_floor is not None:
        pack2, f2 = extend_pack_truth(pack, a.truth_floor, a.fit_lo, a.fit_hi)
        res2 = selftest(pack2, f=f2, use_fp=not a.no_fp, resp_clamp=clamp)
        tab2 = ratio_tables(res2, pack2)
        print_tables(tab2, f"truth-fold, true-N extended to {a.truth_floor:.1f}")
        out["extended"] = tab2
        out["extended_floor"] = a.truth_floor

    print(f"\n[selftest] {time.time() - t0:.1f}s")

    # --- the actual gate -------------------------------------------------
    # main() is a REPORT by default: it prints the ratio table and returns.  That
    # made `forward_selftest --pack $PACK || exit 1` in rung9v3_2lpt0.sbatch a
    # no-op -- it exited 0 on the very pack this module's own commit message
    # declares broken (total mu/obs 0.7312, chi2/dof 2216), so the "fail-closed
    # pre-flight" would have let ~36 h of sampler time run on a forward model
    # known not to close.  --require-closure makes it a gate for real.
    verdict = _closure_verdict(tab, a.max_abs_z_total, a.max_abs_z_bin,
                               a.max_chi2_dof)
    out["closure_verdict"] = verdict
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[selftest] wrote {a.out}")
    if a.require_closure and not verdict["closes"]:
        print("\n[selftest] FORWARD MODEL DOES NOT CLOSE -- refusing.\n  "
              + "\n  ".join(verdict["reasons"]), file=sys.stderr)
        raise SystemExit(3)
    return out


def _closure_verdict(tab, max_abs_z_total, max_abs_z_bin, max_chi2_dof):
    """PASS/FAIL on the truth-fold, from the same table the report prints.

    Deliberately reads 'by_z' as well as 'by_nhat': ratio_tables already computes
    the z-marginal and the production gate discarded it, so a pack carrying the
    full ~+22% z-marginal swing would otherwise sail through a check that only
    looked at the total and the N-marginal.
    """
    tot = tab.get("total", {})
    reasons = []
    zt = abs(float(tot.get("z", 0.0)))
    if zt > max_abs_z_total:
        reasons.append(f"|z_total| {zt:.2f} > {max_abs_z_total}")
    # chi2/dof is COMPUTED here from the n-hat rows.  It used to be read as
    # ``tot.get("chi2_dof", 0.0)`` -- but ``ratio_tables``'s ``total`` has only
    # mu/obs/ratio/z and has NEVER carried a ``chi2_dof`` key, so this arm read
    # 0.0 unconditionally and could not fire.  A table of many mildly-off bins,
    # each individually under the per-bin |z| limit, therefore "closed".
    _rows = [r for r in (tab.get("by_nhat") or [])
             if isinstance(r, dict) and r.get("obs", 0) > 0
             and r.get("z") is not None and np.isfinite(float(r["z"]))]
    _z = np.array([float(r["z"]) for r in _rows], float)
    c2 = float((_z ** 2).sum() / len(_z)) if len(_z) else float("nan")
    if np.isfinite(c2) and c2 > max_chi2_dof:
        reasons.append(f"chi2/dof {c2:.2f} > {max_chi2_dof} "
                       f"over {len(_z)} n-hat bins")
    for key in ("by_nhat", "by_z"):
        rows = tab.get(key) or {}
        zs = [abs(float(r.get("z", 0.0)))
              for r in (rows.values() if isinstance(rows, dict) else rows)
              if isinstance(r, dict) and r.get("z") is not None]
        if zs and max(zs) > max_abs_z_bin:
            reasons.append(f"max|z| in {key} = {max(zs):.2f} > {max_abs_z_bin}")
    return dict(closes=not reasons, reasons=reasons,
                chi2_dof=c2, n_bins=int(len(_z)),
                tolerances=dict(max_abs_z_total=max_abs_z_total,
                                max_abs_z_bin=max_abs_z_bin,
                                max_chi2_dof=max_chi2_dof))


if __name__ == "__main__":
    main()
