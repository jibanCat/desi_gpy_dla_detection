#!/usr/bin/env python
"""build_s6_draws.py — the 8 sealed S6 completeness-calibration draw tables.

CLASSIFICATION: **VALIDATION / PREPARATION ONLY.**  No likelihood, no sampler,
no posterior, no real data.  This script re-evaluates the FROZEN C1nsadd
completeness functional form at 8 PRE-EXISTING bootstrap realisations of its
coefficient vector and writes one ``C_fixed`` table per realisation, in the
schema ``validation/real_c1/run_real_c1.py --c-fixed-file`` consumes.

Authority: PI ruling 2026-09-14b §9 (S6 completeness covariance: propagation
APPROVED, no refit) and the sealed predeclaration
``governance/final_campaign_2026-09-13/S6_COMPLETENESS_COVARIANCE_PROPAGATION_PREDECLARATION.md``.

HARD RULES implemented here (each one is a fail-closed gate):
  * the frozen beta-hat, the functional form, the pivots and the production
    clamp are UNCHANGED — the released standalone evaluator
    ``release/completeness/evaluate_completeness.py`` is imported and used
    verbatim, and gate (1) asserts that it reproduces the frozen table
    ``C_C1nsadd_2lpt0.npz`` at beta-hat to <= 1e-12 on the live strata;
  * the draws are ``beta_bootstrap`` rows 0..7 IN STORED ORDER — no selection,
    no reordering, no replacement (gate: the stored rows are re-read from the
    released covariance product and their sha256 is stamped);
  * beta is NEVER updated with real data; nothing here reads the real pack.

The evaluation convention is the frozen one: the latent-bin centres ``x_b`` and
the live-stratum log10 S/N medians ``u_live`` carried by the covariance product
(the top live stratum's median IS the clamp, so the clamp is on its boundary),
dead strata are exactly 0.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

L_DEFAULT = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
N_DRAWS = 8                      # sealed: indices 0..7, stored order
EQ_TOL = 1e-12                   # gate (1) tolerance on the live strata


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_evaluator(ladder_root: str = L_DEFAULT):
    """Import the RELEASED standalone evaluator (no project code)."""
    rel = os.path.join(ladder_root, "release", "completeness")
    if rel not in sys.path:
        sys.path.insert(0, rel)
    import evaluate_completeness as ec           # noqa: E402
    return ec


def build_table(beta, x_b, u_live, live_idx, n_s, ec, N0=20.0):
    """The (S, B) completeness table at coefficient vector ``beta``.

    Exactly the frozen convention: C[s, b] = evaluate_completeness(
    logN = 20 + x_b[b], log10_snr = U0 + u_live[i], coef=beta) on the live
    strata (production clamp ACTIVE), 0 on the dead strata.
    """
    beta = np.asarray(beta, float).ravel()
    if beta.size != 6:
        raise ValueError("C1nsadd takes exactly 6 coefficients")
    x_b = np.asarray(x_b, float).ravel()
    u_live = np.asarray(u_live, float).ravel()
    live_idx = np.asarray(live_idx, int).ravel()
    if u_live.size != live_idx.size:
        raise ValueError("u_live and live_idx disagree in length")
    C = np.zeros((int(n_s), x_b.size), float)
    C[live_idx, :] = ec.evaluate_completeness(
        float(N0) + x_b[None, :],
        log10_snr=(u_live[:, None] + ec.U0_DEFAULT),
        coef=beta, N0=float(N0), clamp=True)
    return C


def verify_frozen(ladder_root: str = L_DEFAULT, fam: str = "2lpt0"):
    """GATE (1): the released evaluator at beta-hat rebuilds the frozen table.

    Returns a dict with the max abs deviation on the live strata and on the
    whole table.  Raises if the live-strata deviation exceeds ``EQ_TOL``.
    """
    ec = load_evaluator(ladder_root)
    cov_p = os.path.join(ladder_root, "completeness",
                         f"C1nsadd_covariance_{fam}.npz")
    tab_p = os.path.join(ladder_root, "completeness", f"C_C1nsadd_{fam}.npz")
    cv = np.load(cov_p, allow_pickle=True)
    ct = np.load(tab_p, allow_pickle=True)
    C_frozen = np.asarray(ct["C_fixed"], float)
    live = np.asarray(ct["live_strata_mask"], bool)
    C_hat = build_table(cv["beta"], cv["x_b"], cv["u_live"], cv["live_idx"],
                        C_frozen.shape[0], ec)
    dev_live = float(np.abs(C_hat[live] - C_frozen[live]).max())
    dev_all = float(np.abs(C_hat - C_frozen).max())
    if not (dev_live <= EQ_TOL):
        raise SystemExit(
            f"FAIL-CLOSED: the released evaluator does not reproduce the frozen "
            f"{fam} C1nsadd table at beta-hat (max abs dev on live strata "
            f"{dev_live:.3e} > {EQ_TOL:.0e}); refusing to build any S6 draw.")
    # the coefficients stamped into the frozen table must be the beta of record
    prov = json.loads(str(ct["provenance"]))
    coef_tab = np.asarray(prov["coefficients"], float)
    dev_beta = float(np.abs(coef_tab - np.asarray(cv["beta"], float)).max())
    if dev_beta > 0:
        raise SystemExit("FAIL-CLOSED: the covariance product's beta is not the "
                         "frozen table's coefficient vector")
    return dict(max_abs_dev_live=dev_live, max_abs_dev_all=dev_all,
                tolerance=EQ_TOL, beta_identical_to_frozen_table=True,
                covariance_product=cov_p, covariance_sha256=sha256_file(cov_p),
                frozen_table=tab_p, frozen_table_sha256=sha256_file(tab_p),
                live_strata=[int(i) for i in np.where(live)[0]],
                clamp_log10_snr_hi=float(ec.LOG10_SNR_CLAMP_HI),
                u_pivot_log10_snr=float(ec.U0_DEFAULT))


def _git_head(repo):
    try:
        h = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                    text=True).strip()
        d = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"],
                                    text=True).strip()
        return dict(commit=h, dirty=bool(d))
    except Exception:                                        # pragma: no cover
        return dict(commit=None, dirty=None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-root", default=L_DEFAULT)
    ap.add_argument("--family", default="2lpt0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    a = ap.parse_args(argv)
    LR = a.ladder_root
    os.makedirs(a.out_dir, exist_ok=True)

    gate = verify_frozen(LR, a.family)
    print("GATE 1 (released evaluator reproduces the frozen table at beta-hat): "
          f"max |dev| on live strata = {gate['max_abs_dev_live']:.3e} "
          f"<= {EQ_TOL:.0e}  PASS", flush=True)

    ec = load_evaluator(LR)
    cov_p = os.path.join(LR, "completeness", f"C1nsadd_covariance_{a.family}.npz")
    tab_p = os.path.join(LR, "completeness", f"C_C1nsadd_{a.family}.npz")
    real_tab = os.path.join(LR, "real_c1_inputs", "C_C1nsadd_real.npz")
    cv = np.load(cov_p, allow_pickle=True)
    ct = np.load(tab_p, allow_pickle=True)
    bb = np.asarray(cv["beta_bootstrap"], float)
    if bb.ndim != 2 or bb.shape[1] != 6:
        raise SystemExit("beta_bootstrap has an unexpected shape")
    if a.n_draws > bb.shape[0]:
        raise SystemExit("asked for more draws than the released bootstrap has")

    # the real-pack table is a byte-copy of the calibration table; the draws are
    # therefore applicable to the real pack unchanged (gate, fail-closed)
    rt = np.load(real_tab, allow_pickle=True)
    for k in ("C_fixed", "live_strata_mask", "ntrue_edges", "snr_edges",
              "kz_to_K", "b_to_cell"):
        if not np.array_equal(np.asarray(rt[k]), np.asarray(ct[k])):
            raise SystemExit(f"FAIL-CLOSED: the real-pack frozen table differs "
                             f"from the calibration table in '{k}'")

    common = dict(
        schema="absorber_ladder/C_fixed/v1",
        variant="C1nsadd",
        role=("S6 CALIBRATION-UNCERTAINTY DRAW of the FROZEN C1nsadd "
              "completeness table: the same functional form, pivots and "
              "production clamp evaluated at one PRE-EXISTING bootstrap "
              "realisation of the coefficient vector. Not a refit, not a new "
              "completeness surface, not fitted to real data."),
        authority=("PI ruling 2026-09-14b §9; sealed predeclaration "
                   "S6_COMPLETENESS_COVARIANCE_PROPAGATION_PREDECLARATION.md "
                   "(e36cac98)"),
        fitted_to_real_data=False,
        beta_updated_with_real_data=False,
        calibration_family=a.family,
        draw_source=("C1nsadd_covariance_2lpt0.npz['beta_bootstrap'] rows 0..%d "
                     "IN STORED ORDER (sealed a priori; no selection by any "
                     "outcome)" % (a.n_draws - 1)),
        covariance_product=cov_p,
        covariance_product_sha256=gate["covariance_sha256"],
        frozen_table=tab_p,
        frozen_table_sha256=gate["frozen_table_sha256"],
        real_pack_table=real_tab,
        real_pack_table_sha256=sha256_file(real_tab),
        real_pack_table_equals_calibration_table=True,
        evaluator=os.path.join(LR, "release", "completeness",
                               "evaluate_completeness.py"),
        evaluator_sha256=sha256_file(os.path.join(
            LR, "release", "completeness", "evaluate_completeness.py")),
        evaluator_gate=gate,
        evaluation_convention=("C[s, b] = evaluate_completeness(logN = 20 + "
                               "x_b[b], log10_snr = U0 + u_live[i], coef = "
                               "beta_bootstrap[draw], clamp ACTIVE); dead "
                               "strata exactly 0"),
        x_pivot=20.0,
        u_pivot_log10_snr=float(ec.U0_DEFAULT),
        log10_snr_clamp_hi=float(ec.LOG10_SNR_CLAMP_HI),
        consumer="--c-fixed-file <this> (2-D (S,B) branch: keeps g_bk)",
        builder="validation/s6_propagation/build_s6_draws.py",
        built_utc=datetime.datetime.utcnow().isoformat() + "Z",
        git=_git_head(os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."))),
        python=platform.python_version(), numpy=np.__version__,
        source_provenance=json.loads(str(ct["provenance"])))

    written = []
    C_frozen = np.asarray(ct["C_fixed"], float)
    for i in range(a.n_draws):
        beta_i = bb[i]
        C = build_table(beta_i, cv["x_b"], cv["u_live"], cv["live_idx"],
                        C_frozen.shape[0], ec)
        if not np.all(np.isfinite(C)) or C.min() < 0.0 or C.max() > 1.0:
            raise SystemExit(f"draw {i}: C is not a finite probability table")
        live = np.asarray(ct["live_strata_mask"], bool)
        if np.abs(C[~live]).max(initial=0.0) != 0.0:
            raise SystemExit(f"draw {i}: dead strata are not exactly zero")
        p = dict(common)
        p.update(draw_index=int(i),
                 beta=[float(x) for x in beta_i],
                 beta_hat=[float(x) for x in np.asarray(cv["beta"], float)],
                 delta_beta=[float(x) for x in
                             (beta_i - np.asarray(cv["beta"], float))],
                 shape=list(C.shape),
                 max_abs_delta_C_vs_frozen=float(np.abs(C - C_frozen).max()))
        out = os.path.join(a.out_dir, f"C_S6_draw{i}.npz")
        np.savez_compressed(
            out, C_fixed=C, C_fixed_sd=np.zeros_like(C),
            live_strata_mask=np.asarray(ct["live_strata_mask"], bool),
            ntrue_edges=np.asarray(ct["ntrue_edges"]),
            snr_edges=np.asarray(ct["snr_edges"]),
            kz_to_K=np.asarray(ct["kz_to_K"]),
            b_to_cell=np.asarray(ct["b_to_cell"]),
            beta=np.asarray(beta_i, float),
            provenance=np.array(json.dumps(p, indent=1, default=str),
                                dtype=object))
        written.append(out)
        print(f"  draw {i}: max |C - C_frozen| = "
              f"{p['max_abs_delta_C_vs_frozen']:.5f}  -> {out}", flush=True)

    sums = os.path.join(a.out_dir, "SHA256SUMS")
    with open(sums, "w") as fh:
        for w in written:
            fh.write(f"{sha256_file(w)}  {os.path.basename(w)}\n")
    # a parent-level manifest too (the campaign root)
    root = os.path.abspath(os.path.join(a.out_dir, ".."))
    with open(os.path.join(root, "SHA256SUMS"), "w") as fh:
        for w in written:
            fh.write(f"{sha256_file(w)}  {os.path.relpath(w, root)}\n")
        for extra in (cov_p, tab_p, real_tab,
                      os.path.join(LR, "release", "completeness",
                                   "evaluate_completeness.py")):
            fh.write(f"{sha256_file(extra)}  {extra}\n")
    print(f"wrote {len(written)} draw tables + SHA256SUMS under {a.out_dir}")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
