#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Layer-B threshold operating study (PI rulings 2026-08-06 §14.3).

Evaluates candidate Layer-B thresholds (p < 0.01 vs p < 0.05; no third
threshold — none had a frozen rationale) using INDEPENDENTLY SIMULATED
operating characteristics — never the currently observed failures. The
current Phase-B verdict is out of scope by construction (the observed
failures sit far beyond either threshold; nothing here revises them).

THE SIMULATED PROCEDURE IS THE DEPLOYED ONE. Universe: a schema-conformant
`synthetic_pack` at the production 29×15×8 geometry with a KNOWN truth and
a production-regime FP calibration (realized: 94 events carrying a
12.3% mu_FP share — the 89-event/1.5e4-count regime; fp_regime in the
artifact records the realized values). Per
replicate r:

    n0_r ~ Poisson(n0_true)                    # realized finite calibration
    y_r  ~ Poisson(mu_true · (1 + ε·s(N̂)))    # data; ε=0 = healthy null
    Ĉ_r  = E_cov at n0_r  (B=2000, seed 41001) # the frozen deployed recipe
    T_r, p_r = Layer-B gate at (y_r, n0_r)     # null B=2000, seed 43001

with mu_true = mu_sig(truth) + fp_fold(n0_true_mean) the TRUE data mean
(the analyst never sees n0_true), and s(N̂) the OBSERVED twin fractional
tilt shape (committed `observed_tilt_shape.npz`) — ε=1 is the
observed-scale alternative, ε ∈ {0.5, 0.25} the smaller-material-defect
alternatives (§14.3). Deployed seeds stay FIXED across replicates (that IS
the deployed procedure; replicate randomness enters through y_r, n0_r).

FAITHFULNESS GUARD: once per PACK (κ config), before bulk replication,
the study's reduced observed-statistic path is asserted to reproduce the
committed `gate_covariance.predictive_gate` T_obs EXACTLY given the same
covariance object. (The p-value cannot be bit-compared: the reduced
ensembles consume a different rng stream — statistically identical, not
seed-identical; see the Gate docstring. Review finding F8a.)

Family-wise rates: per replicate, THREE independent y-draws share one
n0_r (the three real mocks share the same calibration events), giving the
measured P(≥1 of 3 healthy fails) and P(all 3 pass) under each threshold
with the calibration-shared correlation included. (Approximation, stated:
the three pseudo-mocks share one template truth; the real mocks differ in
truth but share the signal calibration bit-identically.)

Sensitivity axes: calibration size κ ∈ {1, 4.5, 12.5} (≈ 89 / 400 / 1111
events — n0_true and ell_eff scaled together so λ and mu_FP stay fixed);
finite null size B_null ∈ {500, 2000} on the same replicates.

Usage:
  python run_threshold_study.py --smoke          # 40 reps, sanity
  python run_threshold_study.py --n-rep 2000     # the full study (SLURM)
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)

from CDDF_analysis.hbi_mcmc.pack import synthetic_pack                # noqa: E402
from CDDF_analysis.hbi_mcmc.forward_selftest import (                 # noqa: E402
    RATIO_SPAN_NULL_GEOMETRIES)
from CDDF_analysis.hbi_mcmc import gate_covariance as GC              # noqa: E402

GEOM = RATIO_SPAN_NULL_GEOMETRIES["prod_29x15x8"]
FP_KW = dict(fp_frac=0.15, fp_ell_eff=150.0, fp_w=47.0)
PACK_SEED = 7
THRESHOLDS = (0.01, 0.05)
B_COV = GC.N_COV_DRAWS          # 2000, deployed
B_NULL = GC.N_NULL_DRAWS        # 2000, deployed
SEED_COV = GC.SEED_COV          # 41001, deployed
SEED_NULL = GC.SEED_NULL        # 43001, deployed
B_NULL_SENS = 500
N_MOCKS = 3
EPS_POWER = (1.0, 0.5, 0.25)
KAPPAS = (1.0, 4.5, 12.5)
REP_STREAM = 47001              # study-level replicate randomness (§15.5 stream)


def _make_pack(kappa=1.0):
    pk = synthetic_pack(PACK_SEED, **GEOM, t_true=np.zeros(3), **FP_KW)
    if kappa != 1.0:
        n0 = np.asarray(pk.fp_counts, float)
        rng = np.random.default_rng(REP_STREAM + 900 + int(kappa * 10))
        n0k = rng.poisson(n0 * kappa)
        pk = dataclasses.replace(pk, fp_counts=n0k.astype(np.int64),
                                 fp_ell_eff=float(pk.fp_ell_eff) * kappa)
    return pk


class Gate:
    """Cached deployed-procedure evaluator, EXACT REDUCED FORM.

    Two exact reductions make the study tractable (both verified at
    construction):

    1. The FP fold is LINEAR in n0 (mu_FP = w·(1−η_c)·n0[c,s]·E[k,s] up to
       the committed exp(t)=1 factors), so the GROUP vector of
       fp_fold(n0) is M @ n0.flat with M probed from the COMMITTED
       forward.fold_mu_fp on basis vectors — never a re-typed formula.
       Guard: a random-n0 probe must reproduce the committed fold's group
       vector to 1e-9.
    2. Sums of independent Poisson cells under a fixed 0/1 aggregation
       are Poisson with the aggregated mean, so the E_cov/E_null draws
       reduce from (C,Kf,S)-grid draws to 3-vector draws EXACTLY in
       distribution.

    The reduced ensembles are therefore distributionally IDENTICAL to the
    deployed estimate_covariance/predictive_gate loops; they are not
    BIT-identical to the fixed-seed production runs (the rng stream
    consumes different variates), which is irrelevant for operating
    characteristics and is stated in the artifact."""

    def __init__(self, pack):
        self.pack = pack
        self.A = GC.group_aggregator(pack, GC.PRIMARY_GROUP_EDGES)
        self.mu_sig, self.fp_fold, self.live = GC._fold_parts(pack)
        self.g_mu_sig = GC._group_vector(self.mu_sig, self.A, self.live)
        n0_shape = np.asarray(pack.fp_counts, float).shape
        # probe the committed fold's linearity: M (3, C*S)
        ncs = int(np.prod(n0_shape))
        M = np.empty((self.A.shape[0], ncs))
        eye = np.zeros(n0_shape)
        for i in range(ncs):
            eye.flat[i] = 1.0
            M[:, i] = GC._group_vector(self.fp_fold(eye), self.A, self.live)
            eye.flat[i] = 0.0
        self.M = M
        rng = np.random.default_rng(12345)
        n0_probe = rng.poisson(10.0, size=n0_shape).astype(float)
        g_direct = GC._group_vector(self.fp_fold(n0_probe), self.A, self.live)
        if np.max(np.abs(g_direct - M @ n0_probe.ravel())) > 1e-9:
            raise RuntimeError("FP-fold linearity guard FAILED")
        self.n0_shape = n0_shape

    def g_mu_obs(self, n0_obs):
        return self.g_mu_sig + self.M @ np.asarray(n0_obs, float).ravel()

    def _draw_d(self, n0_obs, b, rng):
        """(b, 3) draws of d = G(y*) − G(mu(n0*)) — the deployed resampling
        unit, reduced exactly."""
        gm = self.g_mu_obs(n0_obs)
        y_g = rng.poisson(np.clip(gm, 0, None), size=(b, len(gm)))
        n0_star = rng.poisson(np.asarray(n0_obs, float),
                              size=(b,) + self.n0_shape)
        mu_g = self.g_mu_sig[None, :] + n0_star.reshape(b, -1) @ self.M.T
        return y_g - mu_g

    def null_T(self, n0_obs, b_null=B_NULL):
        d_cov = self._draw_d(n0_obs, B_COV, np.random.default_rng(SEED_COV))
        C = np.cov(d_cov, rowvar=False)
        Cinv = np.linalg.inv(C)
        d_null = self._draw_d(n0_obs, b_null,
                              np.random.default_rng(SEED_NULL))
        T = np.einsum("bi,ij,bj->b", d_null, Cinv, d_null)
        return C, Cinv, self.g_mu_obs(n0_obs), np.sort(T)

    def p_of_grp(self, y_grp, n0_obs, cache=None, b_null=B_NULL):
        if cache is None:
            cache = self.null_T(n0_obs, b_null)
        C, Cinv, gmu, Tn = cache
        d = np.asarray(y_grp, float) - gmu
        T = float(d @ Cinv @ d)
        n_exceed = int(len(Tn) - np.searchsorted(Tn, T, side="left"))
        return T, (1 + n_exceed) / (len(Tn) + 1), cache


def _verify_against_committed(pack, gate):
    """Cross-check the reduced observed-statistic path against the committed
    predictive_gate: SAME covariance object handed to both, so the T_obs
    formula and grouping must agree exactly (the ensembles themselves are
    seed-stream different by construction; see the Gate docstring)."""
    cov = GC.estimate_covariance(pack)
    ref = GC.predictive_gate(pack, covariance=cov)
    y_grp = GC._group_vector(np.asarray(pack.counts, float), gate.A,
                             gate.live)
    d = y_grp - gate.g_mu_obs(np.asarray(pack.fp_counts, float))
    T = float(d @ np.linalg.inv(cov.matrix) @ d)
    if abs(T - ref.T_obs) > 1e-6 * max(1.0, abs(ref.T_obs)):
        raise RuntimeError(
            f"observed-statistic guard FAILED: reduced T={T} vs committed "
            f"T={ref.T_obs}")
    return {"T_committed": float(ref.T_obs), "T_reduced": T,
            "p_committed_fixed_seed": float(ref.p_value)}


def run_config(pack, *, n_rep, eps, tilt_c, rng, gate=None, n_mocks=N_MOCKS,
               b_null_list=(B_NULL,)):
    gate = gate or Gate(pack)
    n0_true = np.asarray(pack.fp_counts, float)
    mu_true = np.clip(gate.mu_sig + gate.fp_fold(n0_true), 0, None)
    factor = 1.0 + eps * tilt_c[:, None, None]
    # the tilted TRUE data mean, aggregated to groups once (Poisson closure)
    mu_alt = np.clip(mu_true * factor, 0, None)
    g_mu_alt = GC._group_vector(mu_alt, gate.A, gate.live)
    ps = {b: np.empty((n_rep, n_mocks)) for b in b_null_list}
    for r in range(n_rep):
        n0_r = rng.poisson(n0_true).astype(float)
        caches = {b: gate.null_T(n0_r, b) for b in b_null_list}
        for m in range(n_mocks):
            y_grp = rng.poisson(g_mu_alt)
            for b in b_null_list:
                _, p, _ = gate.p_of_grp(y_grp, n0_r, cache=caches[b],
                                        b_null=b)
                ps[b][r, m] = p
    return ps


def _rates(p_mat, alpha):
    rej = p_mat <= alpha            # p is the calibrated p-value; reject if <= alpha
    per_mock = float(rej.mean())
    any3 = float(rej.any(axis=1).mean())
    all_pass = float((~rej).all(axis=1).mean())
    n = p_mat.shape[0]
    return {"per_mock_rate": per_mock,
            "per_mock_mc_halfwidth_95": 1.96 * float(
                np.sqrt(max(per_mock * (1 - per_mock), 1e-12) / p_mat.size)),
            "family_any_of_3": any3, "family_all_pass": all_pass,
            "n_replicates": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-rep", type=int, default=2000)
    ap.add_argument("--n-rep-power", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(_HERE,
                                                  "threshold_study.json"))
    a = ap.parse_args()
    n_rep = 40 if a.smoke else a.n_rep
    n_pw = 20 if a.smoke else a.n_rep_power

    tilt = np.load(os.path.join(_HERE, "observed_tilt_shape.npz"))
    tilt_c = np.asarray(tilt["frac_resid_c"], float)

    t0 = time.time()
    out = {"schema": "phaseC_threshold_study/v1",
           "label": ("independently simulated operating characteristics; "
                     "the observed Phase-B failures are NOT inputs except as "
                     "the ε=1 alternative SHAPE (power target, §14.3)"),
           "routine": "diagnostics_phaseC/threshold_study/run_threshold_study.py",
           "date": time.strftime("%Y-%m-%d"),
           "code_commit": subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
           "deployed_recipe": {"B_cov": B_COV, "B_null": B_NULL,
                               "seed_cov": SEED_COV, "seed_null": SEED_NULL,
                               "seeds_fixed_across_replicates": True},
           "geometry": "prod_29x15x8 synthetic_pack seed 7",
           "fp_regime": None, "faithfulness_guard": None,
           "smoke": bool(a.smoke), "configs": {}}

    pack = _make_pack(1.0)
    out["fp_regime"] = {
        "fp_events": int(np.asarray(pack.fp_counts).sum()),
        "counts_total": int(np.asarray(pack.counts).sum()),
        "mu_fp_share": float(
            GC._fold_parts(pack)[1](np.asarray(pack.fp_counts, float)).sum()
            / np.asarray(pack.counts).sum())}
    gate0 = Gate(pack)
    out["faithfulness_guard"] = _verify_against_committed(pack, gate0)
    print(f"[guard] committed-gate agreement OK: {out['faithfulness_guard']}",
          flush=True)

    # 1) healthy null, kappa=1, both B_null values, 3 shared-calibration mocks
    rng = np.random.default_rng(REP_STREAM)
    ps = run_config(pack, n_rep=n_rep, eps=0.0, tilt_c=tilt_c, rng=rng,
                    gate=gate0, b_null_list=(B_NULL, B_NULL_SENS))
    cfg = {}
    for b in (B_NULL, B_NULL_SENS):
        cfg[f"B_null={b}"] = {f"alpha={al}": _rates(ps[b], al)
                              for al in THRESHOLDS}
    out["configs"]["healthy_kappa1"] = cfg
    print(f"[healthy k=1] {time.time()-t0:.0f}s", flush=True)

    # 2) power vs the observed-scale tilt and smaller defects, kappa=1
    for eps in EPS_POWER:
        rng = np.random.default_rng(REP_STREAM + int(eps * 100))
        pse = run_config(pack, n_rep=n_pw, eps=eps, tilt_c=tilt_c, rng=rng,
                         gate=gate0, n_mocks=1)
        out["configs"][f"power_eps{eps}"] = {
            f"alpha={al}": _rates(pse[B_NULL], al) for al in THRESHOLDS}
        print(f"[power eps={eps}] {time.time()-t0:.0f}s", flush=True)

    # 3) calibration-size sensitivity (healthy + observed-scale power)
    for kap in KAPPAS[1:]:
        pk = _make_pack(kap)
        gk = Gate(pk)
        _verify_against_committed(pk, gk)
        for eps, tag in ((0.0, "healthy"), (1.0, "power")):
            rng = np.random.default_rng(REP_STREAM + 500 + int(kap * 10)
                                        + int(eps))
            psk = run_config(pk, n_rep=n_pw, eps=eps, tilt_c=tilt_c, rng=rng,
                             gate=gk, n_mocks=1)
            out["configs"][f"{tag}_kappa{kap}"] = {
                f"alpha={al}": _rates(psk[B_NULL], al) for al in THRESHOLDS}
        print(f"[kappa={kap}] {time.time()-t0:.0f}s", flush=True)

    out["wall_seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["configs"], indent=1)[:2000])
    print("wrote", a.out, f"({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
