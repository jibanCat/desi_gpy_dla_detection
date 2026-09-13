#!/usr/bin/env python
"""run_antitautology.py — the anti-tautology stress battery A-D.

Authority: PI ruling 2026-09-13c §3/§4 (A-D) and the SEALED
``governance/response_review_2026-09-13/RESPONSE_FAMILY_OPENING_RULE_
PREDECLARATION.md`` §6, whose thresholds are applied VERBATIM:

  A  "Threshold for 'occupancy-imprinted': KL excess > 3x the sampling-noise
      KL in rows with >= 200 events."
  B  "a conditional estimator's l changes by less than 2 paired SE."
  C  "no pull toward the 2LPT slope beyond the inversion's own conditioning
      error."
  D  "a table of every step and whether its shrinkage depends on N_bsk,
      neighbours, total CDDF, or S/N/z occupancy."

CALIBRATION / MOCK INFORMATION ONLY.  No HBI run, no MCMC, no real data, no
mock dN/dX closure number is read anywhere in this program.

Usage (one line per candidate):
  python validation/absorber_ladder/response_review/antitautology/\
run_antitautology.py --kinds R1d_raw,R1d,R1c,M_true_emp --tests A,B,C,D
  ... --candidate-mg /scratch/.../response_review/candidates/Mg_E_2lpt0.npz
  ... --candidate-builder /path/to/adapter.py   (registers build_<name>)

VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import importlib.util as ilu
import json
import os
import platform
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import opbuild as OB                                          # noqa: E402
import metrics as MT                                          # noqa: E402
import invert as IV                                           # noqa: E402
import toys as TOY                                            # noqa: E402
import audit_occupancy as AUD                                 # noqa: E402

_REPO = OB._REPO
OUT_DEFAULT = os.path.join(OB.SCRATCH, "response_review", "antitautology")
CAND_DIR = os.path.join(OB.SCRATCH, "response_review", "candidates")

ROW_MIN_EVENTS = 200          # sealed §6A
KL_EXCESS_FACTOR = 3.0        # sealed §6A
SE_FACTOR = 2.0               # sealed §6B
REWEIGHTINGS = ("flatter_dg-0.5", "steeper_dg+0.5", "equal_occupancy")
TRAIN_SLOPES = ("native", "flatter_dg-0.5", "steeper_dg+0.5")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=_REPO).decode().strip()
    except Exception:                                        # pragma: no cover
        return "unknown"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


# ===========================================================================
# builder dispatch (native kinds + toys + registered candidates)
# ===========================================================================
def make(kind, ev, w, geom, **kw):
    if kind in TOY.TOY_BUILDERS:
        return TOY.TOY_BUILDERS[kind](ev, w, geom, **kw)
    return OB.build_operator(ev, w, kind, geom, **kw)


def _w(ev, geom, name):
    return OB.WEIGHTINGS[name](ev, geom)


def _build(kind, ev, geom, wname, sel=None, seed=0, force_resample=False):
    kw = dict(sel=sel)
    if kind == "R1c":
        kw.update(seed=seed, force_resample=force_resample)
    return make(kind, ev, _w(ev, geom, wname), geom, **kw)


def _build_boot(kind, ev, geom, wname, rng_seed, sel=None):
    """One bootstrap replicate.  Count-based operators take the multiplicity
    vector as weights (an exact multinomial bootstrap); R1c goes through its
    own fixed-seed resample path so both sides of every R1c comparison carry
    the same Monte-Carlo noise."""
    if kind == "R1c":
        return _build(kind, ev, geom, wname, sel=sel, seed=rng_seed,
                      force_resample=True)
    rng = np.random.default_rng(rng_seed)
    n = ev["n"]
    mult = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
    return make(kind, ev, _w(ev, geom, wname) * mult, geom, sel=sel)


# ===========================================================================
# TEST A — CDDF reweighting invariance
# ===========================================================================
def test_A(kinds, ev, geom, n_boot=8, verbose=True):
    out = {}
    for kind in kinds:
        force = (kind == "R1c")
        nat = _build(kind, ev, geom, "native", seed=1000,
                     force_resample=force)
        rb = nat.row_b()
        n_row = nat.row_n
        keep = n_row >= ROW_MIN_EVENTS
        lk_nat = MT.row_leakage(nat.P, geom, rb)

        boots = [_build_boot(kind, ev, geom, "native", 2000 + i)
                 for i in range(n_boot)]
        kl_noise = np.mean([MT.row_kl(nat.P, b.P) for b in boots], axis=0)
        lk_noise_sd = {k: np.std([MT.row_leakage(b.P, geom, rb)[k]
                                  for b in boots], axis=0)
                       for k in lk_nat}

        res = dict(kind=kind, n_rows=int(nat.n_rows),
                   n_rows_ge200=int(keep.sum()),
                   row_grid=nat.grid,
                   events_total=float(n_row.sum()),
                   kl_noise_wmean_ge200=MT.wmean(kl_noise[keep],
                                                 n_row[keep]),
                   n_boot=int(n_boot), weightings={})
        for wname in REWEIGHTINGS:
            rw = _build(kind, ev, geom, wname, seed=1001,
                        force_resample=force)
            kl = MT.row_kl(nat.P, rw.P)
            excess = kl - kl_noise
            flag_row = excess > KL_EXCESS_FACTOR * kl_noise
            klw = MT.wmean(kl[keep], n_row[keep])
            klnw = MT.wmean(kl_noise[keep], n_row[keep])
            lk_rw = MT.row_leakage(rw.P, geom, rb)
            leak = {}
            for k in lk_nat:
                d = lk_rw[k] - lk_nat[k]
                leak[k] = dict(
                    d_wmean=MT.wmean(d[keep], n_row[keep]),
                    d_absmax=float(np.max(np.abs(d[keep]))) if keep.any()
                    else None,
                    noise_sd_wmean=MT.wmean(lk_noise_sd[k][keep],
                                            n_row[keep]),
                    n_rows_gt3sigma=int(np.sum(
                        np.abs(d[keep]) > 3.0 * np.maximum(
                            lk_noise_sd[k][keep], 1e-12))))
            res["weightings"][wname] = dict(
                kl_rw_wmean_ge200=klw,
                kl_noise_wmean_ge200=klnw,
                kl_ratio=float(klw / klnw) if klnw > 0 else None,
                aggregate_excess_over_3x=bool(
                    (klw - klnw) > KL_EXCESS_FACTOR * klnw),
                frac_rows_flagged=float(np.mean(flag_row[keep]))
                if keep.any() else None,
                n_rows_flagged=int(np.sum(flag_row[keep])),
                kl_rw_max_ge200=float(np.max(kl[keep])) if keep.any() else None,
                leakage=leak)
            if verbose:
                print(f"[A] {kind:14s} {wname:16s} KL_rw={klw:.3e} "
                      f"KL_noise={klnw:.3e} ratio="
                      f"{(klw/klnw if klnw>0 else float('nan')):7.2f} "
                      f"flagged={res['weightings'][wname]['frac_rows_flagged']}",
                      flush=True)
        res["VERDICT_occupancy_imprinted"] = bool(any(
            v["aggregate_excess_over_3x"] for v in res["weightings"].values()))
        res["verdict_rule"] = (
            "sealed §6A applied to the count-weighted mean over rows with "
            f">= {ROW_MIN_EVENTS} native events: flagged iff "
            "(KL_rw - KL_noise) > 3 x KL_noise, i.e. KL_rw > 4 x KL_noise; "
            "the per-row flagged fraction is reported alongside as the "
            "fine-grained diagnostic")
        out[kind] = res
    return out


# ===========================================================================
# TEST B — train on one slope, evaluate on another
# ===========================================================================
def test_B(kinds, ev, geom, verbose=True):
    """Crossed train-slope / eval-slope matrix.

    ``native_mc`` is a Monte-Carlo control arm: the SAME native weighting with
    a different resample seed.  It is an exact no-op for the count-based
    operators (d = 0) and, for R1c, measures how much of a crossed-arm shift
    is merely the weighted-resampling noise the missing weighted path in
    ``respfit`` forces on us.
    """
    out = {}
    folds = (("A", ev["fold"], ~ev["fold"]), ("B", ~ev["fold"], ev["fold"]))
    arms = [("native", "native", 0), ("native_mc", "native", 5)] + [
        (w, w, i + 1) for i, w in enumerate(TRAIN_SLOPES[1:])]
    for kind in kinds:
        force = (kind == "R1c")
        ops = {}
        for fi, (tag, fitm, _) in enumerate(folds):
            for arm, wname, so in arms:
                ops[(tag, arm)] = _build(kind, ev, geom, wname, sel=fitm,
                                         seed=3000 + 100 * fi + so,
                                         force_resample=force)
        mat, paired = {}, {}
        for ew in TRAIN_SLOPES:
            evw = _w(ev, geom, ew)
            for arm, _wn, _so in arms:
                lls, ns = [], []
                for tag, _, evm in folds:
                    sel = evm & ev["in_grid"]
                    lls.append(MT.per_event_loglik(ops[(tag, arm)], ev, sel,
                                                   evw))
                    ns.append(float(evw[sel].sum()))
                mat[f"eval={ew}|train={arm}"] = float(
                    np.average(lls, weights=ns))
                if arm == "native":
                    continue
                ds, ses = [], []
                for tag, _, evm in folds:
                    sel = evm & ev["in_grid"]
                    _, _, dm, se = MT.paired_loglik(
                        ops[(tag, arm)], ops[(tag, "native")], ev, sel, evw)
                    ds.append(dm); ses.append(se)
                d = float(np.mean(ds))
                se = float(np.sqrt(np.sum(np.square(ses))) / len(ses))
                paired[f"eval={ew}|train={arm}-native"] = dict(
                    d_loglik=d, paired_se=se,
                    n_se=(float(abs(d) / se) if se > 0
                          else (0.0 if d == 0.0 else float("inf"))),
                    conditional_pass=bool(abs(d) <= SE_FACTOR * se))
        real = {k: v for k, v in paired.items() if "native_mc" not in k}
        mc = {k: v for k, v in paired.items() if "native_mc" in k}
        worst = max(v["n_se"] for v in real.values()) if real else 0.0
        out[kind] = dict(
            kind=kind, loglik_matrix=mat, paired=real, mc_control=mc,
            max_n_se=float(worst),
            mc_control_max_n_se=float(max((v["n_se"] for v in mc.values()),
                                          default=0.0)),
            VERDICT_conditional=bool(all(v["conditional_pass"]
                                         for v in real.values())),
            rule=("sealed §6B: a conditional estimator's held-out per-event "
                  "multinomial loglik changes by < 2 paired SE "
                  "(sightline-clustered) when the TRAINING population slope "
                  "is changed"))
        if verbose:
            print(f"[B] {kind:14s} max |d|/SE = {worst:.2f} "
                  f"(MC control {out[kind]['mc_control_max_n_se']:.2f}) "
                  f"conditional={out[kind]['VERDICT_conditional']}", flush=True)
    return out


# ===========================================================================
# TEST C — forward-fold stress test
# ===========================================================================
def test_C(ops, geom, ops_path, verbose=True, reweight_ops=None,
           n_iter=8000):
    Nc, dN, dX, kz2K = geom["Nc"], geom["dN"], geom["dX"], geom["kz2K"]
    # ---- CONVENTION (a), the fold's own (coordinator note 2026-09-13) -----
    # C_true_bKs already carries the in-grid fraction phi (its numerator counts
    # only in-grid detections), while the parametric Mg rows sum to the frozen
    # adopted_phi_ref -- C_true x Mg_parametric would apply phi TWICE.  Test C
    # therefore folds with C_det = N_det_all / truth_counts (NO phi) and gives
    # EVERY operator's unit-sum conditional rows the MEASURED phi(b, K, s).
    cp = OB.completeness_and_phi(geom, os.path.basename(ops_path)
                                 .split("_")[2])
    C_bKs = cp["C_det"]
    phi_m = cp["phi"]
    shapes = IV.synthetic_shapes(Nc)
    f_nat = IV.native_truth_f(ops_path, dN, dX)
    shapes_bk = {k: np.repeat(v[:, None], geom["Kf"], axis=1)
                 for k, v in shapes.items()}
    shapes_bk["native_2lpt_truth"] = f_nat
    usable = (C_bKs.sum(axis=(1, 2)) > 0) & (Nc >= geom["nhat"][0] - 1e-9)
    A_of = {k: IV.build_A(OB.gather_with_phi(o.P, o.grid, geom, phi_m),
                          C_bKs, dX, dN, kz2K) for k, o in ops.items()}
    dXk = dX.sum(axis=1)

    def collapse(f_bk):
        return (np.asarray(f_bk, float) * dXk[None, :]).sum(axis=1) \
            / dXk.sum()

    out = dict(shapes=list(shapes_bk),
               slope_window=[20.0, 21.5],
               em_arms=dict(
                   forensics=("4000 EM iterations started at the truth — the "
                              "convention of analyze_fold.py:545-556 that the "
                              "sealed §6C names ('the deterministic "
                              "Poisson-MLE inversion of the forensics'); the "
                              "early stopping regularises the oscillating "
                              "mode"),
                   flat=(f"{int(n_iter)} EM iterations from a FLAT start — "
                         "the genuinely unpenalised limit, reported as the "
                         "harsher arm")),
               slope_robustness=("a latent bin the unpenalised MLE has "
                                 "COLLAPSED (f_hat < 1e-6 x f_true) is "
                                 "dropped from the slope fit and counted in "
                                 "n_bins_collapsed_in_window; the per-bin "
                                 "ratio table is the primary product"),
               phi_convention=dict(
                   which="(a) the fold's convention",
                   completeness="C_det = sum_k N_det_all_bks_true_z / "
                                "sum_k truth_counts_bks (NO in-grid factor)",
                   rows="every operator's unit-sum conditional rows x the "
                        "MEASURED phi(b,K,s) = sum_k N_det_bks_true_z / "
                        "sum_k N_det_all_bks_true_z",
                   identity_check_max_abs=cp["identity_max_abs"],
                   n_cells_C_det_gt1=cp["n_cells_C_det_gt1"],
                   n_cells_supported=cp["n_cells_supported"],
                   note="C_true_bKs = C_det * phi is verified exactly "
                        "(fail-closed); the frozen adopted_phi_ref is NOT "
                        "used in test C, it is audited in test D"),
               native_2lpt_slope=IV.fit_slope(collapse(f_nat), Nc),
               self=dict(), cross=dict(), reweight_built=dict())
    def summarise(f, fh):
        """One inversion, reported robustly: the per-bin ratio is primary; the
        slope drops any bin the unpenalised MLE has collapsed."""
        ft, fr = collapse(f), collapse(fh)
        ratio = np.divide(fr, np.where(ft > 0, ft, np.nan))
        keep = fr > 1e-6 * np.where(ft > 0, ft, np.inf)
        win = (Nc >= 20.0) & (Nc <= 21.5) & usable
        s_in = IV.fit_slope(ft, Nc)
        s_out = IV.fit_slope(fr, Nc, keep=keep)
        return dict(
            slope_in=s_in, slope_out=s_out, slope_error=s_out - s_in,
            n_bins_collapsed_in_window=int(np.sum(win & ~keep)),
            n_bins_in_window=int(np.sum(win)),
            ratio_by_bin={f"[{geom['ntrue'][b]:.1f},"
                          f"{geom['ntrue'][b+1]:.1f})": float(ratio[b])
                          for b in range(geom["B"]) if usable[b]},
            max_abs_dev_usable=float(np.nanmax(np.abs(ratio[usable] - 1.0))))

    def invert_both(mu, A, f):
        """Two labelled inversion arms.

        ``forensics`` is the convention the sealed §6C names — the
        deterministic Poisson-MLE inversion of the forensics
        (``analyze_fold.py:545-556``: 4000 EM iterations STARTED AT THE TRUTH,
        so early stopping regularises the oscillating mode).  ``flat`` is the
        harsher arm: an uninformative flat start and 8000 iterations, i.e. the
        genuinely unpenalised limit.
        """
        return dict(
            forensics=summarise(f, IV.em_solve(mu, A, n_iter=4000, f0=f)),
            flat=summarise(f, IV.em_solve(mu, A, n_iter=n_iter)))

    for name, op in ops.items():
        A = A_of[name]
        rec = {s: invert_both(IV.fold(A, f), A, f)
               for s, f in shapes_bk.items()}
        out["self"][name] = rec
        if verbose:
            print(f"[C-self] {name:14s} " + " ".join(
                f"{s.split('_')[-1]}:{rec[s]['forensics']['slope_error']:+.2e}"
                for s in rec), flush=True)

    if "M_true_frozen" in ops:
        A_true = A_of["M_true_frozen"]
        for name, op in ops.items():
            if name == "M_true_frozen":
                continue
            rec = {s: invert_both(IV.fold(A_true, f), A_of[name], f)
                   for s, f in shapes_bk.items()}
            out["cross"][name] = rec
            if verbose:
                w = max(r["forensics"]["max_abs_dev_usable"]
                        for r in rec.values())
                print(f"[C-cross] {name:14s} (fold with M_true, invert with "
                      f"candidate) worst |ratio-1| = {w:.3f}", flush=True)
    # ---- C3: how far the CONSTRUCTION-side occupancy dependence measured by
    # test A propagates into a recovered slope.  Truth generates the counts;
    # the inverting operator is the SAME family REBUILT on a reweighted
    # calibration population.  This converts test A's row KL into dex^-1.
    if reweight_ops and "M_true_frozen" in ops:
        A_true = A_of["M_true_frozen"]
        for name, per_w in reweight_ops.items():
            rec = {}
            for wname, op in per_w.items():
                Aw = IV.build_A(OB.gather_with_phi(op.P, op.grid, geom,
                                                   phi_m),
                                C_bKs, dX, dN, kz2K)
                rec[wname] = {s: invert_both(IV.fold(A_true, f), Aw, f)
                              for s, f in shapes_bk.items()}
            # the tautology signature: does the RECOVERED slope move toward
            # the calibration slope when the operator is rebuilt on a
            # differently-sloped calibration population?
            base = rec.get("native", {})
            drift = {}
            for wname, r2 in rec.items():
                if wname == "native":
                    continue
                drift[wname] = {
                    s: (r2[s]["forensics"]["slope_out"]
                        - base[s]["forensics"]["slope_out"])
                    for s in r2 if s in base}
            out["reweight_built"][name] = dict(by_weighting=rec,
                                               slope_drift_vs_native=drift)
            if verbose:
                vals = [abs(v) for d in drift.values() for v in d.values()
                        if v is not None and np.isfinite(v)]
                w = max(vals) if vals else float("nan")
                print(f"[C-rebuilt] {name:14s} max |slope drift| vs native "
                      f"build = {w:.4f} dex^-1", flush=True)
    out["note"] = (
        "SELF = fold and invert with the SAME operator: this is the "
        "tautology probe (any non-zero slope error is the deterministic "
        "inversion's own CONDITIONING error, quantified by the "
        "native_2lpt_truth row).  CROSS = fold with the forensic M_true and "
        "invert with the candidate: that measures REPRESENTATION error, not "
        "tautology; the two are never mixed.  All operators are given the "
        "SAME measured phi(b,K,s), so the only difference between them is "
        "row SHAPE.  "
        "REWEIGHT_BUILT = truth generates, an operator REBUILT on a "
        "reweighted calibration population inverts: the operational "
        "translation of test A into dex^-1 of recovered slope.")
    return out


# ===========================================================================
# main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="R1d_raw,R1d,R1c,M_true_emp")
    ap.add_argument("--with-toys", action="store_true", default=True)
    ap.add_argument("--no-toys", dest="with_toys", action="store_false")
    ap.add_argument("--tests", default="A,B,C,D")
    ap.add_argument("--n-boot", type=int, default=8)
    ap.add_argument("--events", default=OB.EVENTS)
    ap.add_argument("--family", default="2lpt0")
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--merge", action="store_true",
                    help="update only the sections computed in THIS run, "
                         "keeping the rest of an existing results JSON "
                         "(so a later candidate can be pushed through test C "
                         "without re-running A/B/D)")
    ap.add_argument("--candidate-mg", default="",
                    help="comma-separated Mg_<cand>_<fam>.npz paths (test C "
                         "only, unless a builder is registered)")
    ap.add_argument("--candidate-glob", default="",
                    help=f"glob under {CAND_DIR} (e.g. 'Mg_*_2lpt0.npz')")
    ap.add_argument("--candidate-builder", default="",
                    help="comma-separated .py files defining "
                         "BUILDERS = {name: fn(ev, w, geom, **kw)->Operator}")
    a = ap.parse_args(argv)

    for path in [p for p in a.candidate_builder.split(",") if p]:
        spec = ilu.spec_from_file_location("cand_builder", path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k, fn in getattr(mod, "BUILDERS", {}).items():
            OB.register_builder(k, fn)
            print(f"[reg] candidate builder {k} <- {path}")

    geom = OB.load_geometry(a.family)
    ev = OB.load_events(geom, a.events)
    kinds = [k for k in a.kinds.split(",") if k]
    if a.with_toys:
        kinds = kinds + list(TOY.TOY_BUILDERS)
    tests = set(a.tests.split(","))
    os.makedirs(a.out_dir, exist_ok=True)

    ops_path = OB.OPS_TPL.format(fam=a.family)
    rep = dict(
        schema="absorber_ladder/response_review/antitautology/v1",
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        host=platform.node(), git=_git(),
        authority=("PI ruling 2026-09-13c §3/§4; sealed "
                   "RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION.md §6"),
        inputs=dict(events=dict(path=a.events, sha256=_sha256(a.events),
                                n=int(ev["n"]),
                                n_uniq_tid=int(len(np.unique(ev["tid"])))),
                    empirical_ops=dict(path=ops_path,
                                       sha256=_sha256(ops_path)),
                    scanpack=dict(path=geom["pack_path"],
                                  sha256=_sha256(geom["pack_path"]))),
        geometry=dict(B=geom["B"], C=geom["C"], S=geom["S"], KK=geom["KK"],
                      Kf=geom["Kf"]),
        thresholds=dict(row_min_events=ROW_MIN_EVENTS,
                        kl_excess_factor=KL_EXCESS_FACTOR,
                        se_factor=SE_FACTOR),
        kinds=kinds)

    if "A" in tests:
        rep["test_A_reweighting_invariance"] = test_A(kinds, ev, geom,
                                                      a.n_boot)
    if "B" in tests:
        rep["test_B_train_one_slope_eval_another"] = test_B(kinds, ev, geom)
    if "C" in tests:
        ckinds = [k for k in kinds if k not in TOY.TOY_BUILDERS]
        ops = {k: _build(k, ev, geom, "native", seed=1000,
                         force_resample=(k == "R1c")) for k in ckinds}
        ops["M_true_frozen"] = OB.build_mtrue_frozen(geom, a.family)
        rw = {k: {w: _build(k, ev, geom, w, seed=1001,
                            force_resample=(k == "R1c"))
                  for w in ("native",) + REWEIGHTINGS} for k in ckinds}
        cands = [p for p in a.candidate_mg.split(",") if p]
        if a.candidate_glob:
            cands += sorted(glob.glob(os.path.join(CAND_DIR,
                                                   a.candidate_glob)))
        for p in cands:
            nm = os.path.basename(p).replace(".npz", "")
            ops[nm] = OB.operator_from_mg(p, geom, kind=nm)
            print(f"[cand] loaded {nm} from {p}")
        rep["test_C_forward_fold"] = test_C(ops, geom, ops_path,
                                            reweight_ops=rw)
    if "D" in tests:
        rep["test_D_occupancy_audit"] = AUD.run_audit(kinds, ev, geom)

    out_json = os.path.join(a.out_dir, "antitautology_results.json")
    if a.merge and os.path.exists(out_json):
        prev = json.load(open(out_json))
        prev.setdefault("merged_from", []).append(
            dict(created=rep["created"], tests=sorted(tests),
                 kinds=kinds, git=rep["git"]))
        prev.update({k: v for k, v in rep.items()
                     if k.startswith("test_") or k in ("created", "git",
                                                       "kinds")})
        rep = prev
        print(f"[merge] updated {sorted(tests)} in the existing results JSON")
    rep["MUTATION_CONTROLS"] = AUD.mutation_verdicts(rep)
    with open(out_json, "w") as fh:
        json.dump(_jsonable(rep), fh, indent=1, sort_keys=False)
    print(f"[out] {out_json}")
    return rep


if __name__ == "__main__":
    main()
