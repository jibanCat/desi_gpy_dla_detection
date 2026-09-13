#!/usr/bin/env python
"""variant_table.py — VALIDATION-ONLY reader that turns the ABSORBER-SIDE ladder
run artifacts into the PI's variant table.

Sealed predeclaration: /home/mfho/lowz_clean_work_2026-09-12/absorber_ladder/
ABSORBER_LADDER_PREDECLARATION.md (§2 variant list, §4 gates and readouts);
PI ruling 2026-09-13 (second) §15-§16 ("structured mock diagnostics are
first-class gates; a repair that moves the defect elsewhere is not a PASS").

This module DECIDES NOTHING. It applies ``perz_gate.gate_one`` UNCHANGED (the
frozen aggregate gate), reads the structured residuals the predeclaration
already named, and lays them out variant by variant against a baseline. It
re-fits nothing, samples nothing, and applies no new threshold: the only
"verdict" it prints is the frozen gate's own PASS/FAIL plus the purely
descriptive ``DEFECT MOVED`` annotation defined in §4 (a structured residual
worsening by more than 2x its seed spread while a headline improves).

What it reads, per run
``RUN_<VARIANT>_<fam>_s<seed>[_j<k>].json`` written by ``run_ladder.py``:
  RUN_*.json          the cc_posterior_validation-schema summary + the additive
                      diagnostics blocks (predictive_marginals, fp_by_block,
                      support_gate, fixed_files, lam_cut)
  RUN_*_fdraws.npz    f (D, B, Kf), truth_f (B, Kf), ntrue_edges, zf_edges,
                      dX_k -> Omega[20.3, 21.6] and the 13 CDDF bins
  RUN_*_bychain.npz   (not needed here; ladder_table.py reads it)

Grouping
  VARIANT = ``stage`` with the ``LADDER_`` / ``DIAG_`` prefix stripped; when
  ``stage`` is empty the ``ladder`` field (``ORACLE`` / ``M0``..``M5``) is used.
  Then by family (``perz_gate._fam(pack)``) and seed. The M1CUT runs of one
  variant/family/seed are ONE unit, pooled over the imputations j with EQUAL
  weights (the equal-weight mixture of the J posteriors = the concatenation of
  equal numbers of draws); the per-j medians are listed alongside.

Usage:
  python validation/fp_ladder/variant_table.py \
      --runs DIR [DIR ...] --out-json VARIANT_TABLE.json \
      --out-md VARIANT_TABLE.md [--pack-dir P] [--baseline A0]

Example (the diagnostic runs already on disk; baseline ORACLE):
  python validation/fp_ladder/variant_table.py \
      --runs /scratch/.../absorber_diag_2026-09-13/{OP,OC,OCz,OM,OCM,OE} \
             /scratch/.../fp_ladder_2026-09-12/diag_oracle \
      --pack-dir /scratch/.../absorber_ladder_2026-09-13/support \
      --baseline ORACLE --out-json /tmp/VT.json --out-md /tmp/VT.md
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name, filename):
    """Import a sibling tool by path (validation/fp_ladder is not a package)."""
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LT = _load_sibling("_fp_ladder_ladder_table", "ladder_table.py")

# ---------------------------------------------------------------------------
# frozen constants (NOT tunable here)
# ---------------------------------------------------------------------------
THRESH_KEYS = ("ge20.0", "ge20.3")
KBLOCK_NAMES = ("K0", "K1", "K2")          # coarse_blocks block0/1/2
CDDF_N_LO = 19.7                            # the 13 reported CDDF bins start here
ZIGZAG_SOURCE = "reporting_bins"            # the 0.2-dex reporting bins
DEFECT_MOVED_FACTOR = 2.0                   # §4: "beyond its CV/MC noise" -> 2x seed spread
STRUCTURED_METRICS = ("zigzag_range", "K1_bias_ge20p3", "snr_max_abs_dev")
REDGES = np.arange(19.7, 21.7 + 1e-9, 0.2)  # run_ladder's reporting-bin edges (verbatim)


# ---------------------------------------------------------------------------
# pure helpers (unit-tested; no I/O)
# ---------------------------------------------------------------------------
def variant_of(d):
    """VARIANT id of a run dict: ``stage`` minus the LADDER_/DIAG_ prefix, else
    the ``ladder`` field (an empty stage is how some pre-ladder runs were
    written).

    Two disclosed normalisations, both provenance-preserving:
      * the prefix match is case-insensitive (the 2026-09-12 oracle runs carry
        ``stage = 'diag_oracle'``, the 2026-09-13 ones ``'DIAG_OP'``);
      * a stage that merely restates the ladder id (``diag_oracle`` on an
        ``ORACLE`` run) collapses to the ladder id itself, so those runs group
        as ``ORACLE`` rather than as a separate one-run variant.
    """
    lad = str(d.get("ladder") or "UNKNOWN")
    st = str(d.get("stage") or "").strip()
    if not st:
        return lad
    for pre in ("LADDER_", "DIAG_"):
        if st.upper().startswith(pre):
            st = st[len(pre):]
            break
    if not st:
        return lad
    return lad if st.upper() == lad.upper() else st


def imputation_of(d):
    """(J, j) for an M1CUT run, else (None, None)."""
    lc = (d.get("diagnostics") or {}).get("lam_cut")
    if isinstance(lc, dict):
        return lc.get("J"), lc.get("j")
    return None, None


def zigzag_stats(reporting_bins):
    """The 0.2-dex reporting-bin zigzag: the list of per-bin median biases (%),
    their range (max - min), RMS and max |bias|.

    ``range`` is the predeclaration's "zigzag amplitude": a f(N) recovery that
    alternates bin to bin has a large peak-to-peak spread even when the
    integrated headline is unbiased."""
    rows = [r for r in (reporting_bins or []) if r.get("median_bias_pct") is not None]
    vals = [float(r["median_bias_pct"]) for r in rows]
    if not vals:
        return dict(n_bins=0, bins=[], biases=[], range=None, rms=None, max_abs=None)
    a = np.asarray(vals, float)
    return dict(n_bins=len(vals),
                bins=[r.get("bin") for r in rows],
                biases=[float(v) for v in a],
                range=float(a.max() - a.min()),
                rms=float(np.sqrt(np.mean(a ** 2))),
                max_abs=float(np.max(np.abs(a))))


def marginal_stats(ratios, live_eps=0.0):
    """Summary of a mu/obs marginal (``diagnostics.predictive_marginals``).

    A stratum with no observed counts is written by the runner as exactly 0.0
    (``mu / max(obs, 1)`` with obs == 0 and mu == 0); those are DEAD strata and
    are excluded, never counted as a 100 % deviation. Returns the live indices,
    max |ratio - 1|, and the OLS slope of ratio on the live stratum INDEX
    (per stratum), which is the "ramp" the predeclaration asks for."""
    r = np.asarray(ratios, float) if ratios is not None else np.zeros(0)
    idx = np.flatnonzero(np.isfinite(r) & (r > live_eps))
    if idx.size == 0:
        return dict(n_strata=int(r.size), n_live=0, live_idx=[], ratios_live=[],
                    max_abs_dev=None, argmax_idx=None, slope=None, ptp=None)
    v = r[idx]
    dev = np.abs(v - 1.0)
    slope = None
    if idx.size >= 2:
        x = idx.astype(float)
        xm, ym = x.mean(), v.mean()
        den = float(np.sum((x - xm) ** 2))
        slope = float(np.sum((x - xm) * (v - ym)) / den) if den > 0 else None
    return dict(n_strata=int(r.size), n_live=int(idx.size),
                live_idx=[int(i) for i in idx],
                ratios_live=[float(x) for x in v],
                max_abs_dev=float(dev.max()),
                argmax_idx=int(idx[int(np.argmax(dev))]),
                slope=slope, ptp=float(v.max() - v.min()))


def cddf_bin_biases(f, truth_f, ntrue_edges, dX_k, n_lo=CDDF_N_LO):
    """The 13 reported CDDF bins: path-weighted all-z f per true-N bin, median
    bias vs truth and 68/95 containment.

    Pure arithmetic, the same path weighting as run_ladder's reporting bins:
        f_allz[b] = sum_k f[.,b,k] dX_k / sum_k dX_k
    (the dN factor of a dN/dX-style reduction cancels in a per-bin ratio, so the
    CDDF bin bias and the per-bin dN/dX bias are the same number).
    """
    f = np.asarray(f, float)
    ft = np.asarray(truth_f, float)
    e = np.asarray(ntrue_edges, float)
    w = np.asarray(dX_k, float)
    tot = float(w.sum())
    if tot <= 0:
        return []
    post = np.einsum("dbk,k->db", f, w) / tot
    truth = np.einsum("bk,k->b", ft, w) / tot
    out = []
    for b in range(len(e) - 1):
        if e[b] < n_lo - 1e-9:
            continue
        tv = float(truth[b])
        q = np.percentile(post[:, b], [2.5, 16, 50, 84, 97.5])
        out.append(dict(bin=[round(float(e[b]), 2), round(float(e[b + 1]), 2)],
                        truth=tv,
                        post_p16_50_84=[float(q[1]), float(q[2]), float(q[3])],
                        median_bias_pct=(round(100.0 * (float(q[2]) / tv - 1.0), 3)
                                         if tv > 0 else None),
                        truth_in_68=bool(q[1] <= tv <= q[3]),
                        truth_in_95=bool(q[0] <= tv <= q[4])))
    return out


def seed_rollup(values):
    """mean / spread over seeds. ``spread`` = sample sd (ddof=1) when more than
    one seed is present, else None (NEVER 0, so a single-seed variant cannot
    silently acquire a zero noise scale)."""
    v = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    if not v:
        return dict(n=0, mean=None, spread=None, min=None, max=None, values=[])
    a = np.asarray(v, float)
    return dict(n=len(v), mean=float(a.mean()),
                spread=(float(a.std(ddof=1)) if len(v) > 1 else None),
                min=float(a.min()), max=float(a.max()),
                values=[float(x) for x in a])


def movement_flag(delta_struct, spreads, headline_improved,
                  factor=DEFECT_MOVED_FACTOR):
    """§4 annotation. ``delta_struct`` maps metric -> Delta|metric| vs baseline
    (positive == the structured residual got WORSE); ``spreads`` maps metric ->
    the noise scale to compare against (None when no seed spread is available).

    Returns (flag, reasons, undecidable) where ``flag`` is True only when the
    headline improved AND at least one structured residual worsened by more
    than ``factor`` x its noise scale. Metrics whose noise scale is unknown are
    listed in ``undecidable`` and NEVER silently flagged or silently cleared."""
    reasons, undec = [], []
    for m, dv in sorted(delta_struct.items()):
        if dv is None:
            continue
        s = spreads.get(m)
        if s is None or not np.isfinite(s) or s <= 0:
            if dv > 0:
                undec.append(f"{m} worsened by {dv:+.4g} (no seed spread available)")
            continue
        if dv > factor * s:
            reasons.append(f"{m} {dv:+.4g} > {factor:g}x spread {s:.4g}")
    return (bool(headline_improved and reasons), reasons, undec)


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""):
                h.update(ch)
        return h.hexdigest()
    except Exception as e:                               # reported, never silent
        return f"unavailable: {type(e).__name__}: {e}"


_SHA_CACHE = {}


def sha_of(path):
    if not path:
        return None
    if path not in _SHA_CACHE:
        _SHA_CACHE[path] = _sha256(path) if os.path.exists(path) else "missing"
    return _SHA_CACHE[path]


# ---------------------------------------------------------------------------
# verbatim copies of run_ladder's own reductions (used ONLY when pooling M1CUT
# imputations, where the runner's per-j JSON blocks must be recomputed on the
# equal-weight mixture). Copied, not re-derived.
# ---------------------------------------------------------------------------
def recompute_thresholds_and_bins(f_draws, ft, pk):
    """run_ladder.py lines 139-170, verbatim in substance: the committed
    ``reduce_f_posterior`` headlines and the 0.2-dex reporting bins."""
    from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior
    red = reduce_f_posterior(f_draws, pk)
    red_t = reduce_f_posterior(np.asarray(ft, float)[None, :, :], pk)
    ntrue = np.asarray(pk.ntrue_edges, float)
    dN = np.diff(ntrue)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    rep = {}
    for thr, key in ((20.0, "dndx_dla_20p0_allz"), (20.3, "dndx_dla_20p3_allz")):
        dr = np.asarray(red[key])
        tv = float(np.asarray(red_t[key])[0])
        q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
        rep[f"ge{thr}"] = dict(truth=tv, post_p16_50_84=[float(x) for x in q[1:4]],
                               post_p2p5_97p5=[float(q[0]), float(q[4])],
                               median_bias_pct=round(100 * (q[2] / tv - 1), 2),
                               truth_in_68=bool(q[1] <= tv <= q[3]),
                               truth_in_95=bool(q[0] <= tv <= q[4]))
    binrep = []
    for e0, e1 in zip(REDGES[:-1], REDGES[1:]):
        m = (ntrue[:-1] >= e0 - 1e-9) & (ntrue[1:] <= e1 + 1e-9)
        if not m.any():
            continue
        dr = ((f_draws[:, m, :] * dN[None, m, None]).sum(axis=1)
              * dX_k[None, :]).sum(axis=1) / dX_k.sum()
        tv = float(((np.asarray(ft, float)[m, :] * dN[m, None]).sum(axis=0) * dX_k).sum()
                   / dX_k.sum())
        q = np.percentile(dr, [2.5, 16, 50, 84, 97.5])
        binrep.append(dict(bin=[round(float(e0), 1), round(float(e1), 1)],
                           median_bias_pct=round(100 * (q[2] / tv - 1), 2),
                           truth_in_68=bool(q[1] <= tv <= q[3]),
                           truth_in_95=bool(q[0] <= tv <= q[4])))
    return rep, binrep


def pool_f_draws(paths):
    """Equal-weight mixture of the J imputation posteriors: take the SAME number
    of draws (the minimum over j, contiguous from the front of each chain-major
    array) from every imputation and concatenate. Returns (f, truth_f, extras)."""
    arrs, truths, meta = [], [], []
    for p in paths:
        z = np.load(p)
        arrs.append(np.asarray(z["f"], float))
        truths.append(np.asarray(z["truth_f"], float))
        meta.append(dict(path=p, n_draws=int(arrs[-1].shape[0])))
    n = min(a.shape[0] for a in arrs)
    for t in truths[1:]:
        if not np.allclose(t, truths[0], atol=0):
            raise ValueError("imputations disagree about truth_f — refusing to pool")
    f = np.concatenate([a[:n] for a in arrs], axis=0)
    return f, truths[0], dict(n_per_imputation=int(n), n_imputations=len(arrs),
                              members=meta)


# ---------------------------------------------------------------------------
# per-run read-out
# ---------------------------------------------------------------------------
def _bins_map(perz, thr, key):
    blk = (perz.get(thr) or {}).get(key, [])
    return {b.get("bin"): dict(bias=b.get("median_bias_pct"),
                               in68=b.get("truth_in_68"),
                               in95=b.get("truth_in_95"),
                               z=b.get("z"), dX=b.get("dX"))
            for b in blk if b.get("available")}


def _kblocks(perz, thr):
    m = _bins_map(perz, thr, "coarse_blocks")
    out = {}
    for i, name in enumerate(KBLOCK_NAMES):
        src = m.get(f"block{i}")
        if src is not None:
            out[name] = src
    return out


def _structured(d, cddf13):
    """The structured residuals of §4 that live in the run JSON."""
    dg = d.get("diagnostics") or {}
    pm = dg.get("predictive_marginals") or {}
    zz = zigzag_stats(d.get("reporting_bins"))
    snr = marginal_stats(pm.get("mu_over_obs_by_snr"))
    nhat = marginal_stats(pm.get("mu_over_obs_by_nhat"))
    zmg = marginal_stats(pm.get("mu_over_obs_by_z"))
    byk = [marginal_stats(x) for x in (pm.get("mu_over_obs_by_nhat_K") or [])]
    cb = [c.get("median_bias_pct") for c in cddf13 if c.get("median_bias_pct") is not None]
    return dict(
        zigzag=zz, snr=snr, nhat=nhat, z=zmg,
        nhat_by_K=[dict(K=i, max_abs_dev=b["max_abs_dev"], slope=b["slope"],
                        n_live=b["n_live"]) for i, b in enumerate(byk)],
        nhat_by_K_max_abs_dev=(max([b["max_abs_dev"] for b in byk
                                    if b["max_abs_dev"] is not None], default=None)),
        cddf13=cddf13,
        cddf13_max_abs_bias=(float(np.max(np.abs(cb))) if cb else None),
        cddf13_rms_bias=(float(np.sqrt(np.mean(np.square(cb)))) if cb else None))


def _sampler(d):
    dg = d.get("diagnostics") or {}
    mx = dg.get("estimand_mixing") or {}
    rh = [v.get("split_rhat") for v in mx.values() if v.get("split_rhat") is not None]
    ess = [v.get("ess") for v in mx.values() if v.get("ess") is not None]
    eb = dg.get("ebfmi_per_chain") or []
    return dict(divergences=d.get("divergences"),
                divergences_per_chain=dg.get("divergences_per_chain"),
                rhat_max=(float(max(rh)) if rh else None),
                ess_min=(float(min(ess)) if ess else None),
                ebfmi_min=(float(min(eb)) if eb else None),
                ebfmi_per_chain=list(eb),
                estimand_mixing=mx)


def _fixed_files(d):
    dg = d.get("diagnostics") or {}
    ff = dg.get("fixed_files") or {}
    out = {}
    for k in ("mg", "c", "extra"):
        p = ff.get(k)
        out[k] = dict(path=p, basename=(os.path.basename(p) if p else None),
                      sha256=sha_of(p)) if p else None
    return out


def _cddf_and_omega(fdraws_path, pack):
    """The two _fdraws.npz-derived readouts: Omega[20.3, 21.6] on the PAPER's own
    weights (reused verbatim from ladder_table) and the 13 CDDF bins."""
    om = {"unavailable": "no _fdraws.npz next to the run"}
    cd = []
    if fdraws_path and os.path.exists(fdraws_path):
        try:
            om = LT.paper_omega_20p3_21p6(fdraws_path, pack)
        except Exception as e:
            om = {"error": f"{type(e).__name__}: {e}"}
        try:
            z = np.load(fdraws_path)
            cd = cddf_bin_biases(z["f"], z["truth_f"], z["ntrue_edges"], z["dX_k"])
        except Exception as e:
            cd = [{"error": f"{type(e).__name__}: {e}"}]
    return om, cd


def analyse_run(path, pack_dir=None, pack_cache=None):
    """One RUN_*.json -> the full per-run readout."""
    from CDDF_analysis.hbi_mcmc.perz_gate import gate_one

    d = json.load(open(path))
    base = path[:-5]
    gate = gate_one(d)                                   # FROZEN, unchanged
    pkp = LT._pack_path(d, pack_dir)
    pack = None
    if pkp:
        if pack_cache is not None and pkp in pack_cache:
            pack = pack_cache[pkp]
        else:
            try:
                from CDDF_analysis.hbi_mcmc.pack import load_pack
                pack = load_pack(pkp)
            except Exception:
                pack = None
            if pack_cache is not None:
                pack_cache[pkp] = pack
    om, cd = _cddf_and_omega(base + "_fdraws.npz", pack)
    perz = (d.get("perz_recovery") or {}).get("estimand", {})
    J, j = imputation_of(d)
    thr = d.get("thresholds") or {}
    return dict(
        file=os.path.basename(path), path=os.path.abspath(path),
        variant=variant_of(d), ladder=d.get("ladder"), stage=d.get("stage"),
        family=gate["family"], seed=(d.get("run_config") or {}).get("seed"),
        imputation_J=J, imputation_j=j, pack=d.get("pack"), pack_resolved=pkp,
        n_draws=d.get("n_draws"),
        gate_status=gate["status"], gate_fails=gate["fails"], gate=gate,
        allz={k: dict(bias=(thr.get(k) or {}).get("median_bias_pct"),
                      in68=(thr.get(k) or {}).get("truth_in_68"),
                      in95=(thr.get(k) or {}).get("truth_in_95"),
                      truth=(thr.get(k) or {}).get("truth")) for k in THRESH_KEYS},
        bins={k: _bins_map(perz, k, "paper1_bins") for k in THRESH_KEYS},
        kblocks={k: _kblocks(perz, k) for k in THRESH_KEYS},
        structured=_structured(d, cd),
        omega_20p3_21p6=om,
        sampler=_sampler(d),
        support_gate=d.get("support_gate"),
        fixed_files=_fixed_files(d),
        lam_cut=(d.get("diagnostics") or {}).get("lam_cut"),
        fdraws=(base + "_fdraws.npz" if os.path.exists(base + "_fdraws.npz") else None),
        role=d.get("role"))


# ---------------------------------------------------------------------------
# units: single runs, or one pooled M1CUT unit per (variant, family, seed)
# ---------------------------------------------------------------------------
def _mean_ratio_lists(lists):
    ok = [np.asarray(x, float) for x in lists if x]
    if not ok or len({a.shape for a in ok}) != 1:
        return None
    return np.mean(np.stack(ok), axis=0)


def pool_m1cut_unit(runs, pack, pack_path):
    """Pool the imputations of ONE (variant, family, seed) M1CUT group with equal
    weights and rebuild the readouts on the mixture.

    Requires the pack (the committed reductions are pack-driven). Without it the
    unit degrades to the per-j medians with an explicit note — it is never
    silently approximated."""
    runs = sorted(runs, key=lambda r: (r.get("imputation_j") if r.get("imputation_j")
                                       is not None else 0))
    per_j = [dict(j=r.get("imputation_j"), file=r["file"],
                  allz={k: r["allz"][k]["bias"] for k in THRESH_KEYS},
                  zigzag_range=r["structured"]["zigzag"]["range"],
                  snr_max_abs_dev=r["structured"]["snr"]["max_abs_dev"],
                  omega_bias=(r["omega_20p3_21p6"] or {}).get("median_bias_pct"),
                  gate_status=r["gate_status"], gate_fails=r["gate_fails"],
                  divergences=r["sampler"]["divergences"],
                  rhat_max=r["sampler"]["rhat_max"],
                  ebfmi_min=r["sampler"]["ebfmi_min"]) for r in runs]
    fdr = [r["fdraws"] for r in runs if r.get("fdraws")]
    unit = dict(
        kind="M1CUT_POOLED", variant=runs[0]["variant"], ladder=runs[0]["ladder"],
        family=runs[0]["family"], seed=runs[0]["seed"],
        n_imputations=len(runs), imputation_J=runs[0].get("imputation_J"),
        members=[r["file"] for r in runs], per_j=per_j,
        pack=runs[0].get("pack"), pack_resolved=pack_path,
        support_gate=runs[0].get("support_gate"), fixed_files=runs[0]["fixed_files"],
        lam_cut=[r.get("lam_cut") for r in runs],
        file="+".join(r["file"] for r in runs)[:120])
    # sampler health: the WORST over the imputations (each j is its own 4-chain run,
    # so the frozen per-run limits stay per-run; summing them would fail the group
    # for having been imputed at all).
    def _worst(key, how):
        vals = [r["sampler"][key] for r in runs if r["sampler"][key] is not None]
        return (how(vals) if vals else None)
    unit["sampler"] = dict(divergences=_worst("divergences", max),
                           rhat_max=_worst("rhat_max", max),
                           ess_min=_worst("ess_min", min),
                           ebfmi_min=_worst("ebfmi_min", min),
                           per_j=[dict(j=p["j"], divergences=p["divergences"],
                                       rhat_max=p["rhat_max"], ebfmi_min=p["ebfmi_min"])
                                  for p in per_j],
                           note=("worst over the J imputations; each j is its own "
                                 "4-chain run and keeps the frozen per-run limits"))
    if pack is None or len(fdr) != len(runs):
        unit.update(pooled="unavailable",
                    pooled_note=("pooling needs the pack and every imputation's "
                                 "_fdraws.npz; the per-j readouts are listed instead"),
                    allz={k: dict(bias=float(np.median([p["allz"][k] for p in per_j
                                                        if p["allz"][k] is not None]))
                                  if any(p["allz"][k] is not None for p in per_j) else None,
                                  in68=None, in95=None) for k in THRESH_KEYS},
                    bins={k: {} for k in THRESH_KEYS}, kblocks={k: {} for k in THRESH_KEYS},
                    structured=_structured({}, []),
                    omega_20p3_21p6={"unavailable": "not pooled"},
                    gate_status=("PASS" if all(p["gate_status"] == "PASS" for p in per_j)
                                 else "FAIL"),
                    gate_fails=sorted({g for p in per_j for g in p["gate_fails"]}))
        return unit

    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import perz_recovery
    f, ft, meta = pool_f_draws(fdr)
    rep, binrep = recompute_thresholds_and_bins(f, ft, pack)
    perz = perz_recovery(f, ft, pack)
    z0 = np.load(fdr[0])
    cd = cddf_bin_biases(f, ft, z0["ntrue_edges"], z0["dX_k"])
    pooled_d = dict(
        pack=runs[0].get("pack"), ladder=runs[0]["ladder"], stage=runs[0]["stage"],
        n_draws=int(f.shape[0]), divergences=unit["sampler"]["divergences"],
        thresholds=rep, reporting_bins=binrep, perz_recovery=perz,
        diagnostics=dict(estimand_mixing={
            k: dict(split_rhat=unit["sampler"]["rhat_max"],
                    ess=unit["sampler"]["ess_min"])
            for k in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz")},
            predictive_marginals=_mean_marginals([r.get("_pm") or {} for r in runs])))
    from CDDF_analysis.hbi_mcmc.perz_gate import gate_one
    gate = gate_one(pooled_d)
    pest = perz.get("estimand", {})
    # Omega on the pooled mixture, with the paper's own weights
    try:
        zg = np.asarray(z0["zf_edges"], float)
        ow, zw, pref = _paper_omega_weights(f, z0["ntrue_edges"], zg, z0["dX_k"])
        omp = (LT.omega_allz_from_weights(f, ft, ow, zw, pref)
               if ow is not None else {"unavailable": "paper weights unavailable"})
    except Exception as e:
        omp = {"error": f"{type(e).__name__}: {e}"}
    unit.update(pooled="ok", pooled_meta=meta,
                n_draws=int(f.shape[0]),
                gate_status=gate["status"], gate_fails=gate["fails"], gate=gate,
                allz={k: dict(bias=(rep.get(k) or {}).get("median_bias_pct"),
                              in68=(rep.get(k) or {}).get("truth_in_68"),
                              in95=(rep.get(k) or {}).get("truth_in_95"),
                              truth=(rep.get(k) or {}).get("truth")) for k in THRESH_KEYS},
                bins={k: _bins_map(pest, k, "paper1_bins") for k in THRESH_KEYS},
                kblocks={k: _kblocks(pest, k) for k in THRESH_KEYS},
                structured=_structured(pooled_d, cd),
                omega_20p3_21p6=omp)
    return unit


def _lst(a):
    return None if a is None else [float(x) for x in np.asarray(a, float).ravel()]


def _mean_marginals(pms):
    """Equal-weight mean of the per-imputation posterior-median predictive
    marginals. A marginal missing from ANY imputation (older runs do not carry
    ``predictive_marginals`` at all) yields None rather than a partial mean."""
    out = {}
    for k in ("mu_over_obs_by_nhat", "mu_over_obs_by_z", "mu_over_obs_by_snr",
              "tp_over_obs_by_nhat"):
        vals = [p.get(k) for p in pms]
        out[k] = (_lst(_mean_ratio_lists(vals)) if all(v for v in vals) else None)
    nk = [len(p.get("mu_over_obs_by_nhat_K") or []) for p in pms]
    out["mu_over_obs_by_nhat_K"] = (
        [_lst(_mean_ratio_lists([p["mu_over_obs_by_nhat_K"][i] for p in pms]))
         for i in range(nk[0])] if nk and len(set(nk)) == 1 and nk[0] > 0 else [])
    out["note"] = ("equal-weight mean of the per-imputation posterior-median "
                   "marginals (None where any imputation lacks the block)")
    return out


def _paper_omega_weights(f, n_edges, z_edges, dX):
    """The paper's Omega[20.3, 21.6] weights, imported READ-ONLY exactly as
    ladder_table.paper_omega_20p3_21p6 does (shared code path, no re-derivation)."""
    pf = LT.PAPER_FIGURES
    if not os.path.isdir(pf):
        return None, None, None
    if pf not in sys.path:
        sys.path.insert(0, pf)
    import hbi_reduction as HR                              # READ-ONLY
    P = HR.Posterior.__new__(HR.Posterior)
    P.f, P.n_edges, P.z_edges, P.dX = (np.asarray(f, float), np.asarray(n_edges, float),
                                       np.asarray(z_edges, float), np.asarray(dX, float))
    return (P._omega_weight(*HR.OMEGA_NHI), P._z_weight(*HR.LOWZ_SUPPORT),
            HR.OMEGA_PREFACTOR_CM2)


def build_units(runs):
    """Single runs, except that the M1CUT runs of one (variant, family, seed) are
    collapsed into one pooled unit."""
    groups, singles = {}, []
    for r in runs:
        if str(r.get("ladder")) == "M1CUT" and r.get("imputation_j") is not None:
            groups.setdefault((r["variant"], r["family"], r["seed"]), []).append(r)
        else:
            singles.append(r)
    return singles, groups


# ---------------------------------------------------------------------------
# metric extraction from a unit (single run or pooled)
# ---------------------------------------------------------------------------
def unit_metrics(u):
    """The flat metric dictionary the aggregation and the movement table use."""
    st = u.get("structured") or {}
    m = dict(
        bias_ge20p0=(u["allz"].get("ge20.0") or {}).get("bias"),
        bias_ge20p3=(u["allz"].get("ge20.3") or {}).get("bias"),
        K0_bias_ge20p0=((u["kblocks"].get("ge20.0") or {}).get("K0") or {}).get("bias"),
        K1_bias_ge20p0=((u["kblocks"].get("ge20.0") or {}).get("K1") or {}).get("bias"),
        K2_bias_ge20p0=((u["kblocks"].get("ge20.0") or {}).get("K2") or {}).get("bias"),
        K0_bias_ge20p3=((u["kblocks"].get("ge20.3") or {}).get("K0") or {}).get("bias"),
        K1_bias_ge20p3=((u["kblocks"].get("ge20.3") or {}).get("K1") or {}).get("bias"),
        K2_bias_ge20p3=((u["kblocks"].get("ge20.3") or {}).get("K2") or {}).get("bias"),
        zigzag_range=(st.get("zigzag") or {}).get("range"),
        zigzag_rms=(st.get("zigzag") or {}).get("rms"),
        zigzag_max_abs=(st.get("zigzag") or {}).get("max_abs"),
        snr_max_abs_dev=(st.get("snr") or {}).get("max_abs_dev"),
        snr_slope=(st.get("snr") or {}).get("slope"),
        nhat_max_abs_dev=(st.get("nhat") or {}).get("max_abs_dev"),
        z_max_abs_dev=(st.get("z") or {}).get("max_abs_dev"),
        cddf13_max_abs_bias=st.get("cddf13_max_abs_bias"),
        cddf13_rms_bias=st.get("cddf13_rms_bias"),
        omega_bias=(u.get("omega_20p3_21p6") or {}).get("median_bias_pct"),
        divergences=(u.get("sampler") or {}).get("divergences"),
        rhat_max=(u.get("sampler") or {}).get("rhat_max"),
        ess_min=(u.get("sampler") or {}).get("ess_min"),
        ebfmi_min=(u.get("sampler") or {}).get("ebfmi_min"))
    for k in THRESH_KEYS:
        for b, v in (u.get("bins") or {}).get(k, {}).items():
            m[f"{b}_bias_{k.replace('.', 'p')}"] = v.get("bias")
    return m


def aggregate(units):
    """variant x family -> mean over seeds + seed spread of every metric, plus
    the containment tallies and the frozen gate rollup."""
    by = {}
    for u in units:
        by.setdefault((u["variant"], u["family"]), []).append(u)
    out = {}
    for (v, fam), us in sorted(by.items()):
        mets = [unit_metrics(x) for x in us]
        keys = sorted({k for m in mets for k in m})
        roll = {k: seed_rollup([m.get(k) for m in mets]) for k in keys}
        in68 = sum(int(bool((x["allz"].get(k) or {}).get("in68"))) for x in us
                   for k in THRESH_KEYS)
        in95 = sum(int(bool((x["allz"].get(k) or {}).get("in95"))) for x in us
                   for k in THRESH_KEYS)
        ntot = len(us) * len(THRESH_KEYS)
        bins_in95 = sum(int(bool(b.get("in95"))) for x in us for k in THRESH_KEYS
                        for b in (x.get("bins") or {}).get(k, {}).values())
        bins_tot = sum(1 for x in us for k in THRESH_KEYS
                       for _ in (x.get("bins") or {}).get(k, {}).values())
        out[f"{v}|{fam}"] = dict(
            variant=v, family=fam, n_units=len(us),
            seeds=sorted({x.get("seed") for x in us}, key=str),
            kinds=sorted({x.get("kind", "RUN") for x in us}),
            status=("PASS" if all(x["gate_status"] == "PASS" for x in us) else "FAIL"),
            gate_fails=sorted({g for x in us for g in x.get("gate_fails", [])}),
            metrics=roll,
            allz_in68=f"{in68}/{ntot}", allz_in95=f"{in95}/{ntot}",
            perbin_in95=f"{bins_in95}/{bins_tot}",
            units=[x["file"] for x in us])
    return out


def by_variant(agg):
    out = {}
    for rec in agg.values():
        v = rec["variant"]
        d = out.setdefault(v, dict(variant=v, families={}, status=None,
                                   gate_fails=[], n_units=0))
        d["families"][rec["family"]] = rec["status"]
        d["gate_fails"] = sorted(set(d["gate_fails"]) | set(rec["gate_fails"]))
        d["n_units"] += rec["n_units"]
    for v, d in out.items():
        fams = d["families"]
        d["n_families_pass"] = sum(1 for s in fams.values() if s == "PASS")
        d["n_families"] = len(fams)
        d["status"] = ("PASS" if fams and all(s == "PASS" for s in fams.values())
                       else "FAIL")
    return out


# ---------------------------------------------------------------------------
# structured residual movement vs the baseline
# ---------------------------------------------------------------------------
def _global_spread(agg, metric):
    """Median of the AVAILABLE nonzero seed spreads of ``metric`` across the whole
    table — the fallback noise scale for a single-seed variant."""
    sp = [rec["metrics"][metric]["spread"] for rec in agg.values()
          if metric in rec["metrics"] and rec["metrics"][metric]["spread"]]
    sp = [s for s in sp if s and np.isfinite(s) and s > 0]
    return float(np.median(sp)) if sp else None


def movement_table(agg, baseline):
    """§4 / PI §16: for every variant x family, the movement of the structured
    residuals against the baseline variant, with the DEFECT MOVED annotation."""
    rows, fallback = [], {m: _global_spread(agg, m) for m in STRUCTURED_METRICS}
    for key, rec in sorted(agg.items()):
        v, fam = rec["variant"], rec["family"]
        if v == baseline:
            continue
        base = agg.get(f"{baseline}|{fam}")
        if base is None:
            rows.append(dict(variant=v, family=fam, comparable=False,
                             note=f"no {baseline} run for family {fam}"))
            continue

        def g(r, m, field="mean"):
            return (r["metrics"].get(m) or {}).get(field)

        d_abs, d_signed, spreads = {}, {}, {}
        for m in STRUCTURED_METRICS:
            a, b = g(rec, m), g(base, m)
            if a is None or b is None:
                d_abs[m] = d_signed[m] = None
            else:
                d_signed[m] = float(a) - float(b)
                d_abs[m] = abs(float(a)) - abs(float(b))      # >0 == worse
            s = [x for x in (g(rec, m, "spread"), g(base, m, "spread"))
                 if x is not None and np.isfinite(x) and x > 0]
            spreads[m] = max(s) if s else fallback.get(m)
        h0a, h0b = g(rec, "bias_ge20p0"), g(base, "bias_ge20p0")
        h3a, h3b = g(rec, "bias_ge20p3"), g(base, "bias_ge20p3")
        d_h0 = (None if h0a is None or h0b is None else float(h0a) - float(h0b))
        d_h3 = (None if h3a is None or h3b is None else float(h3a) - float(h3b))
        imp = []
        if h0a is not None and h0b is not None and abs(h0a) < abs(h0b):
            imp.append("ge20.0")
        if h3a is not None and h3b is not None and abs(h3a) < abs(h3b):
            imp.append("ge20.3")
        flag, reasons, undec = movement_flag(d_abs, spreads, bool(imp))
        rows.append(dict(
            variant=v, family=fam, comparable=True, baseline=baseline,
            delta_headline_pp=dict(ge20p0=d_h0, ge20p3=d_h3),
            headline_improved_on=imp,
            delta_abs=d_abs, delta_signed=d_signed,
            spread_used=spreads, spread_fallback_used={
                m: bool(spreads[m] is not None
                        and (g(rec, m, "spread") in (None, 0)
                             and g(base, m, "spread") in (None, 0)))
                for m in STRUCTURED_METRICS},
            defect_moved=flag, defect_moved_reasons=reasons,
            undecidable=undec,
            rule=(f"DEFECT MOVED iff a headline |bias| improves AND some structured "
                  f"residual |value| worsens by more than {DEFECT_MOVED_FACTOR:g}x its "
                  f"seed spread (fallback: the table's median nonzero spread)")))
    return rows


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
FAM_ORDER = ("2lpt0", "london0", "saclay0")


def _fmt(x, f="{:+.2f}"):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f.format(x)
    except Exception:
        return "n/a"


def _fam_cells(agg, v, metric, f="{:+.2f}", fams=None):
    cells = []
    for fam in (fams or FAM_ORDER):
        rec = agg.get(f"{v}|{fam}")
        if rec is None:
            cells.append("—")
            continue
        r = rec["metrics"].get(metric) or {}
        s = _fmt(r.get("mean"), f)
        if r.get("spread") is not None:
            s += f"±{r['spread']:.2f}"
        cells.append(s)
    return " / ".join(cells)


def render_md(units, agg, variants, moves, baseline, families):
    L = []
    L.append("# Absorber-side ladder — variant table")
    L.append("")
    L.append("Generated by `validation/fp_ladder/variant_table.py`. VALIDATION-ONLY "
             "read-out of MOCK runs. The aggregate gate is `perz_gate.gate_one` "
             "CRIT v2 **unchanged**; every structured residual below is one the "
             "sealed predeclaration §4 / PI ruling 2026-09-13b §16 already named. "
             "This table applies no new threshold and selects nothing.")
    L.append("")
    L.append(f"Baseline for all Δ columns: **{baseline}**. Families in column order: "
             + ", ".join(families) + ". Cells are mean over seeds ± seed sd "
             "(sd shown only where more than one seed is present).")
    L.append("")

    # ---------------- (1) the PI's variant result table ---------------------
    L.append("## 1. Variant result table")
    L.append("")
    L.append("| variant | families passing frozen gate | ≥20.0 bias % (" + "/".join(families)
             + ") | ≥20.3 bias % | K1 ≥20.3 bias % | zigzag range (pp) | S/N max dev | "
             "Ω[20.3,21.6] bias % | coverage (all-z in68/in95; per-bin in95) | "
             "sampler health (div / R̂ / ESS / E-BFMI) | Δ vs " + baseline + " (pp, ≥20.0 / ≥20.3) |")
    L.append("|" + "---|" * 11)
    for v in sorted(variants):
        d = variants[v]
        fams_s = ", ".join(f"{k}:{s}" for k, s in sorted(d["families"].items()))
        recs = [agg[k] for k in agg if agg[k]["variant"] == v]
        i68 = sum(int(r["allz_in68"].split("/")[0]) for r in recs)
        n68 = sum(int(r["allz_in68"].split("/")[1]) for r in recs)
        i95 = sum(int(r["allz_in95"].split("/")[0]) for r in recs)
        pb = sum(int(r["perbin_in95"].split("/")[0]) for r in recs)
        pbn = sum(int(r["perbin_in95"].split("/")[1]) for r in recs)
        dv = LT._rng([r["metrics"]["divergences"]["max"] for r in recs], "{:.0f}")
        rh = LT._rng([r["metrics"]["rhat_max"]["max"] for r in recs], "{:.4f}")
        es = LT._rng([r["metrics"]["ess_min"]["min"] for r in recs], "{:.0f}")
        eb = LT._rng([r["metrics"]["ebfmi_min"]["min"] for r in recs], "{:.3f}")
        if v == baseline:
            dh = "— (baseline)"
        else:
            mv = [m for m in moves if m["variant"] == v and m.get("comparable")]
            dh = (" / ".join(
                LT._rng([m["delta_headline_pp"][k] for m in mv], "{:+.2f}")
                for k in ("ge20p0", "ge20p3")) if mv else "n/a")
        L.append(
            f"| **{v}** | {d['n_families_pass']}/{d['n_families']} ({fams_s}) | "
            f"{_fam_cells(agg, v, 'bias_ge20p0', fams=families)} | "
            f"{_fam_cells(agg, v, 'bias_ge20p3', fams=families)} | "
            f"{_fam_cells(agg, v, 'K1_bias_ge20p3', fams=families)} | "
            f"{_fam_cells(agg, v, 'zigzag_range', '{:.2f}', fams=families)} | "
            f"{_fam_cells(agg, v, 'snr_max_abs_dev', '{:.4f}', fams=families)} | "
            f"{_fam_cells(agg, v, 'omega_bias', fams=families)} | "
            f"{i68}/{n68}; {i95}/{n68}; {pb}/{pbn} | "
            f"{dv} / {rh} / {es} / {eb} | {dh} |")
    L.append("")

    # ---------------- (2) per-run detail ------------------------------------
    L.append("## 2. Per-run (per-unit) detail")
    L.append("")
    L.append("| unit | variant | ladder | family | seed | j | gate | ≥20.0 % | ≥20.3 % | "
             "in68/95 (20.0, 20.3) | K0/K1/K2 ≥20.3 % | zigzag range / RMS | "
             "S/N maxdev (slope) | N̂ maxdev | z maxdev | CDDF13 max\\|bias\\| | Ω bias % | "
             "div | R̂ | ESS | E-BFMI | support id | fixed files | gate fails |")
    L.append("|" + "---|" * 24)
    for u in sorted(units, key=lambda x: (str(x["variant"]), str(x["family"]),
                                          str(x["seed"]))):
        m = unit_metrics(u)
        st = u["structured"]
        sg = u.get("support_gate") or {}
        sid = (sg.get("support_id_short") if isinstance(sg, dict) else None) or "—"
        ff = u.get("fixed_files") or {}
        ffs = ", ".join(f"{k}={ff[k]['basename']}@{str(ff[k]['sha256'])[:8]}"
                        for k in ("mg", "c", "extra") if ff.get(k)) or "—"
        a0, a3 = u["allz"].get("ge20.0") or {}, u["allz"].get("ge20.3") or {}
        L.append(
            f"| {u['file']} | {u['variant']} | {u.get('ladder')} | {u['family']} | "
            f"{u.get('seed')} | {u.get('imputation_j') if u.get('kind') != 'M1CUT_POOLED' else 'pooled'} | "
            f"{u['gate_status']} | {_fmt(m['bias_ge20p0'])} | {_fmt(m['bias_ge20p3'])} | "
            f"{a0.get('in68')}/{a0.get('in95')}, {a3.get('in68')}/{a3.get('in95')} | "
            f"{_fmt(m['K0_bias_ge20p3'])} / {_fmt(m['K1_bias_ge20p3'])} / {_fmt(m['K2_bias_ge20p3'])} | "
            f"{_fmt(m['zigzag_range'], '{:.2f}')} / {_fmt(m['zigzag_rms'], '{:.2f}')} | "
            f"{_fmt(m['snr_max_abs_dev'], '{:.4f}')} ({_fmt(m['snr_slope'], '{:+.4f}')}) | "
            f"{_fmt(m['nhat_max_abs_dev'], '{:.4f}')} | {_fmt(m['z_max_abs_dev'], '{:.4f}')} | "
            f"{_fmt(m['cddf13_max_abs_bias'], '{:.2f}')} | {_fmt(m['omega_bias'])} | "
            f"{m['divergences']} | {_fmt(m['rhat_max'], '{:.4f}')} | "
            f"{_fmt(m['ess_min'], '{:.0f}')} | {_fmt(m['ebfmi_min'], '{:.3f}')} | "
            f"{sid} | {ffs} | {'; '.join(u.get('gate_fails') or []) or '—'} |")
    L.append("")
    pooled = [u for u in units if u.get("kind") == "M1CUT_POOLED"]
    if pooled:
        L.append("### M1CUT imputations (per-j medians; the unit above is the "
                 "equal-weight pool)")
        L.append("")
        L.append("| unit | j | gate | ≥20.0 % | ≥20.3 % | zigzag range | S/N maxdev | "
                 "Ω bias % | div | R̂ | E-BFMI |")
        L.append("|" + "---|" * 11)
        for u in pooled:
            for p in u["per_j"]:
                L.append(f"| {u['variant']}/{u['family']}/s{u['seed']} | {p['j']} | "
                         f"{p['gate_status']} | {_fmt(p['allz']['ge20.0'])} | "
                         f"{_fmt(p['allz']['ge20.3'])} | "
                         f"{_fmt(p['zigzag_range'], '{:.2f}')} | "
                         f"{_fmt(p['snr_max_abs_dev'], '{:.4f}')} | "
                         f"{_fmt(p['omega_bias'])} | {p['divergences']} | "
                         f"{_fmt(p['rhat_max'], '{:.4f}')} | "
                         f"{_fmt(p['ebfmi_min'], '{:.3f}')} |")
        L.append("")

    # ---------------- (3) structured residual movement ----------------------
    L.append("## 3. Structured residual movement vs " + baseline)
    L.append("")
    L.append("Δ|x| > 0 means the structured residual got WORSE. `DEFECT MOVED` is "
             "raised when a headline |bias| improves while some structured residual "
             f"worsens by more than {DEFECT_MOVED_FACTOR:g}× its seed spread "
             "(predeclaration §4: such a repair is **not** a PASS). Where no seed "
             "spread exists the worsening is listed as *undecidable*, never cleared.")
    L.append("")
    L.append("| variant | family | Δ≥20.0 (pp) | Δ≥20.3 (pp) | headline improved on | "
             "Δ(zigzag range) | Δ(K1 ≥20.3 bias) | Δ(S/N max dev) | flag |")
    L.append("|" + "---|" * 9)
    for m in moves:
        if not m.get("comparable"):
            L.append(f"| {m['variant']} | {m['family']} | — | — | — | — | — | — | "
                     f"{m.get('note')} |")
            continue
        da = m["delta_abs"]
        tag = ("**DEFECT MOVED**" if m["defect_moved"] else
               ("undecidable: " + "; ".join(m["undecidable"]) if m["undecidable"] else "—"))
        L.append(
            f"| {m['variant']} | {m['family']} | "
            f"{_fmt(m['delta_headline_pp']['ge20p0'])} | "
            f"{_fmt(m['delta_headline_pp']['ge20p3'])} | "
            f"{', '.join(m['headline_improved_on']) or 'neither'} | "
            f"{_fmt(da.get('zigzag_range'), '{:+.2f}')} | "
            f"{_fmt(da.get('K1_bias_ge20p3'), '{:+.2f}')} | "
            f"{_fmt(da.get('snr_max_abs_dev'), '{:+.4f}')} | {tag} |")
        if m["defect_moved"]:
            for r in m["defect_moved_reasons"]:
                L.append(f"| | | | | | | | | ↳ {r} |")
    L.append("")

    # ---------------- (4) closure / coverage --------------------------------
    L.append("## 4. Closure and coverage across families")
    L.append("")
    L.append("| variant | family | gate | ≥20.0 bias % | ≥20.3 bias % | all-z in68 | "
             "all-z in95 | per-bin in95 | K0/K1/K2 ≥20.0 % | K0/K1/K2 ≥20.3 % | "
             "B1..B5 ≥20.3 % | Ω bias % | CDDF13 max\\|bias\\| / RMS | gate fails |")
    L.append("|" + "---|" * 14)
    for key in sorted(agg):
        rec = agg[key]
        mm = rec["metrics"]

        def gm(k, f="{:+.2f}"):
            return _fmt((mm.get(k) or {}).get("mean"), f)
        bs = " / ".join(gm(f"B{i}_bias_ge20p3") for i in range(1, 6))
        L.append(
            f"| {rec['variant']} | {rec['family']} | **{rec['status']}** | "
            f"{gm('bias_ge20p0')} | {gm('bias_ge20p3')} | {rec['allz_in68']} | "
            f"{rec['allz_in95']} | {rec['perbin_in95']} | "
            f"{gm('K0_bias_ge20p0')} / {gm('K1_bias_ge20p0')} / {gm('K2_bias_ge20p0')} | "
            f"{gm('K0_bias_ge20p3')} / {gm('K1_bias_ge20p3')} / {gm('K2_bias_ge20p3')} | "
            f"{bs} | {gm('omega_bias')} | "
            f"{gm('cddf13_max_abs_bias', '{:.2f}')} / {gm('cddf13_rms_bias', '{:.2f}')} | "
            f"{'; '.join(rec['gate_fails']) or '—'} |")
    L.append("")

    # ---------------- provenance -------------------------------------------
    L.append("## Provenance of the fixed calibration objects")
    L.append("")
    L.append("| variant | family | seed | support id | Mg fixed | C fixed | extra fixed |")
    L.append("|" + "---|" * 7)
    seen = set()
    for u in sorted(units, key=lambda x: (str(x["variant"]), str(x["family"]))):
        ff = u.get("fixed_files") or {}
        sg = u.get("support_gate") or {}
        row = (u["variant"], u["family"], str(u.get("seed")))
        if row in seen:
            continue
        seen.add(row)

        def _c(k):
            e = ff.get(k)
            return f"{e['basename']} `{str(e['sha256'])[:12]}`" if e else "—"
        L.append(f"| {u['variant']} | {u['family']} | {u.get('seed')} | "
                 f"{(sg.get('support_id_short') if isinstance(sg, dict) else None) or '—'} | "
                 f"{_c('mg')} | {_c('c')} | {_c('extra')} |")
    L.append("")
    notes = sorted({n for u in units for n in (u.get("notes") or [])})
    unav = sorted({str((u.get('omega_20p3_21p6') or {}).get('unavailable'))
                   for u in units
                   if (u.get('omega_20p3_21p6') or {}).get('unavailable')})
    if notes or unav:
        L.append("## Schema notes (disclosed, not corrected here)")
        L.append("")
        for n in notes:
            L.append(f"* {n}")
        for n in unav:
            L.append(f"* Ω[20.3,21.6] unavailable for some units: {n}")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def collect_paths(dirs):
    paths = []
    for d in dirs:
        if os.path.isfile(d):
            paths.append(d)
            continue
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            if p.endswith(("_ppc.json",)) or os.path.basename(p).startswith(("LADDER_TABLE",
                                                                             "VARIANT_TABLE")):
                continue
            try:
                j = json.load(open(p))
            except Exception:
                continue
            if isinstance(j, dict) and {"ladder", "thresholds", "perz_recovery"} <= set(j):
                paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="directories holding RUN_<VARIANT>_<fam>_s<seed>[_j<k>].json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--pack-dir", default=None,
                    help="fallback directory for the packs named in the run JSONs")
    ap.add_argument("--baseline", default="A0",
                    help="variant every Δ column is measured against (default A0)")
    a = ap.parse_args(argv)

    paths = collect_paths(a.runs)
    if not paths:
        raise SystemExit("no ladder run JSONs found under: " + " ".join(a.runs))

    pack_cache = {}
    runs = [analyse_run(p, a.pack_dir, pack_cache) for p in paths]
    # keep the raw predictive marginals around for the pooled units
    for r, p in zip(runs, paths):
        r["_pm"] = (json.load(open(p)).get("diagnostics") or {}).get("predictive_marginals") or {}

    singles, groups = build_units(runs)
    units = list(singles)
    for key, grp in sorted(groups.items()):
        pk = grp[0].get("pack_resolved")
        units.append(pool_m1cut_unit(grp, pack_cache.get(pk), pk))
    for u in units:
        u.pop("_pm", None)
        u.setdefault("kind", "RUN")

    agg = aggregate(units)
    variants = by_variant(agg)
    if a.baseline not in variants:
        print(f"WARNING: baseline variant {a.baseline!r} is not present "
              f"(present: {sorted(variants)}); Δ columns will be empty")
    moves = movement_table(agg, a.baseline)
    families = [f for f in FAM_ORDER if any(r["family"] == f for r in agg.values())]
    families += sorted({r["family"] for r in agg.values()} - set(families))

    out = dict(
        role=("VALIDATION-ONLY variant table for the sealed absorber-side ladder "
              "(ABSORBER_LADDER_PREDECLARATION §2/§4; PI ruling 2026-09-13b §15-16). "
              "The aggregate gate is perz_gate CRIT v2 unchanged; the structured "
              "residuals are the predeclared ones; this table selects nothing."),
        baseline=a.baseline, families=families,
        conventions=dict(
            variant_id="stage minus LADDER_/DIAG_ prefix, else the ladder field",
            m1cut_pooling=("equal-weight mixture of the J imputation posteriors "
                           "(equal draw counts concatenated); per-j medians listed"),
            cddf_bins=f"native true-N bins with lower edge >= {CDDF_N_LO} (13 bins)",
            zigzag=f"{ZIGZAG_SOURCE} (0.2-dex): range = max-min of the per-bin median bias",
            snr_marginal=("diagnostics.predictive_marginals.mu_over_obs_by_snr; "
                          "strata written as exactly 0.0 are DEAD (obs == 0) and "
                          "are excluded from max-dev and slope"),
            defect_moved=(f"headline |bias| improves AND a structured residual "
                          f"worsens by > {DEFECT_MOVED_FACTOR:g}x its seed spread"),
            structured_metrics=list(STRUCTURED_METRICS)),
        n_units=len(units), n_runs=len(runs),
        units=units, by_variant_family=agg, by_variant=variants,
        movement=moves)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)) or ".", exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=1, default=str)
    open(a.out_md, "w").write(render_md(units, agg, variants, moves, a.baseline, families))

    for u in sorted(units, key=lambda x: (str(x["variant"]), str(x["family"]))):
        m = unit_metrics(u)
        print(f"{u['gate_status']:4s} {u['variant']:14s} {str(u['family']):9s} "
              f"s{u.get('seed')} ge20.0={_fmt(m['bias_ge20p0'])} "
              f"ge20.3={_fmt(m['bias_ge20p3'])} zig={_fmt(m['zigzag_range'], '{:.2f}')} "
              f"snr={_fmt(m['snr_max_abs_dev'], '{:.4f}')} "
              f"omega={_fmt(m['omega_bias'])} fails={u.get('gate_fails')}")
    print("\nby variant:", {v: d["status"] for v, d in sorted(variants.items())})
    dm = [f"{m['variant']}/{m['family']}" for m in moves if m.get("defect_moved")]
    print("DEFECT MOVED:", dm or "none")
    print(f"\nwrote {a.out_json} and {a.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
