#!/usr/bin/env python
"""save_coef_covariance.py — preserve the C1nsadd coefficient-uncertainty
information for a LATER category-3 propagation scheme (PI ruling 2026-09-13c §6,
last sentence: "use the fixed calibration mean for now; preserve the full
covariance/bootstrap information").

VALIDATION-ONLY.  This writes ONE new product,
``C1nsadd_covariance_<fam>.npz``, next to the delivered calibration objects.
**The covariance is NOT used anywhere**: no fold, no figure, no gate and no run
reads it; the fixed calibration mean ``C_fixed`` remains the only thing any
consumer sees.  No delivered product is modified and ``SHA256SUMS`` of the
existing objects is untouched.

Three independent uncertainty statements are stored, because they answer
different questions and disagree by a factor that a propagation scheme must
know about:

``cov_fisher``      inverse observed Fisher information of the full-table IRLS
                    fit (the binomial-model sampling covariance of the six
                    coefficients, assuming independent trials);
``cov_bootstrap``   covariance over 200 refits on binomial resamples drawn
                    INDEPENDENTLY WITHIN each TARGETID-parity sightline half
                    and then pooled (same seed every run);
``cov_halfsplit``   the rank-1, 1-degree-of-freedom estimate
                    ``(1/4)(beta_E - beta_O)(beta_E - beta_O)^T`` built from
                    the two independent sightline halves.  This is the only one
                    of the three that sees SIGHTLINE-LEVEL clustering (several
                    absorbers share a spectrum, so the binomial trials are not
                    independent); its diagonal over the Fisher diagonal is the
                    over-dispersion factor recorded in ``overdispersion``.

Run (seconds):
    conda activate gpdla-hbi   # or gpdla; no jax is needed here
    PYTHONPATH=/home/mfho/wt_abs_diag_2026-09 python \
      validation/absorber_ladder/completeness_review/save_coef_covariance.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from validation.absorber_ladder.completeness.cal_fit import (        # noqa: E402
    design_2d_additive, irls_binomial)

ROOT = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/completeness"
X0, DEG_N, DEG_U = 20.0, 3, 2
SEED = 20260913


def _git_head():
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=True,
                           cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, check=True,
                           cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        return dict(commit=c, dirty=bool(d))
    except Exception:                                                # pragma: no cover
        return dict(commit=None, dirty=None)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zpool(a):
    return np.asarray(a, float).sum(axis=1).T


def build(fam="2lpt0", n_boot=200, out=None):
    cal = np.load(f"{ROOT}/cal_table_{fam}.npz", allow_pickle=True)
    obj = np.load(f"{ROOT}/C_C1nsadd_{fam}.npz", allow_pickle=True)
    prov = json.loads(str(obj["provenance"].item()))
    ne = np.asarray(cal["ntrue_edges"], float)
    Nc = 0.5 * (ne[:-1] + ne[1:])
    x_b = Nc - X0
    tot_full = zpool(cal["truth_bks"])
    live = tot_full.sum(axis=1) > 0
    li = np.where(live)[0]
    logmed = np.asarray(cal["snr_logmed_s"], float)
    w = tot_full.sum(axis=1)[li]
    u0 = float((logmed[li] * w).sum() / w.sum())
    u_live = logmed[li] - u0
    if abs(u0 - float(prov["u_pivot_log10_snr"])) > 1e-12:
        raise SystemExit("pivot mismatch against the delivered object (fail-closed)")

    X = design_2d_additive(x_b, u_live, DEG_N, DEG_U)
    det = zpool(cal["det_bks"])[li]
    tot = tot_full[li]
    dE, tE = zpool(cal["det_bks_E"])[li], zpool(cal["truth_bks_E"])[li]
    dO, tO = zpool(cal["det_bks_O"])[li], zpool(cal["truth_bks_O"])[li]

    st = irls_binomial(X, det.ravel(), tot.ravel(), ridge=1e-6)
    beta = st["beta"]
    delivered = np.asarray(prov["coefficients"], float)
    dmax = float(np.max(np.abs(beta - delivered)))
    if dmax > 1e-8:
        raise SystemExit(f"refit does not reproduce the delivered coefficients "
                         f"(max |d| = {dmax:.3e}); refusing to write (fail-closed)")
    cov_fisher = np.asarray(st["cov"], float)

    stE = irls_binomial(X, dE.ravel(), tE.ravel(), ridge=1e-6)
    stO = irls_binomial(X, dO.ravel(), tO.ravel(), ridge=1e-6)
    dbeta = stE["beta"] - stO["beta"]
    cov_halfsplit = 0.25 * np.outer(dbeta, dbeta)

    rng = np.random.default_rng(SEED)
    boots = np.empty((int(n_boot), X.shape[1]), float)
    pE = np.where(tE > 0, dE / np.maximum(tE, 1.0), 0.0)
    pO = np.where(tO > 0, dO / np.maximum(tO, 1.0), 0.0)
    for i in range(int(n_boot)):
        bE = rng.binomial(tE.astype(np.int64), pE)
        bO = rng.binomial(tO.astype(np.int64), pO)
        sb = irls_binomial(X, (bE + bO).ravel(), tot.ravel(), ridge=1e-6,
                           beta0=beta)
        boots[i] = sb["beta"]
    cov_boot = np.cov(boots, rowvar=False)

    sd_f = np.sqrt(np.diag(cov_fisher))
    sd_b = np.sqrt(np.diag(cov_boot))
    sd_h = np.sqrt(np.diag(cov_halfsplit))
    prov_out = dict(
        role=("coefficient-uncertainty information for the CANDIDATE completeness "
              "object C1nsadd; PRESERVED FOR A LATER CATEGORY-3 PROPAGATION SCHEME "
              "ONLY — not used by any fold, figure, gate or run"),
        status="C1nsadd is a CANDIDATE, NOT ADOPTED",
        calibration_family=fam, fitted_to_real_data=False,
        representation="logit C = poly_3(x) + poly_2(u), x = N - 20, u = log10(SNR_med) - u0",
        coefficient_order=["b0", "b_x", "b_x2", "b_x3", "b_u", "b_u2"],
        x_pivot=X0, u_pivot_log10_snr=u0, ridge=1e-6,
        n_bootstrap=int(n_boot), bootstrap_seed=SEED,
        bootstrap_scheme=("binomial resample of every live (b, s) cell within each "
                          "TARGETID-parity sightline half, halves then pooled and "
                          "refitted; it therefore reproduces the binomial (Fisher) "
                          "scale and does NOT see sightline-level clustering"),
        halfsplit_scheme=("rank-1, 1-dof estimate 0.25 (beta_E - beta_O) outer "
                          "(beta_E - beta_O); the ONLY one of the three that sees "
                          "sightline-level clustering"),
        overdispersion_sd_halfsplit_over_fisher=[float(a / b) for a, b in zip(sd_h, sd_f)],
        gate_refit_reproduces_delivered_coefficients=dict(max_abs_diff=dmax, atol=1e-8),
        source_cal_table=f"{ROOT}/cal_table_{fam}.npz",
        source_cal_table_sha256=_sha256(f"{ROOT}/cal_table_{fam}.npz"),
        source_object=f"{ROOT}/C_C1nsadd_{fam}.npz",
        source_object_sha256=_sha256(f"{ROOT}/C_C1nsadd_{fam}.npz"),
        recipe="validation/absorber_ladder/completeness_review/save_coef_covariance.py",
        git=_git_head(), python=sys.version.split()[0], numpy=np.__version__,
        caveat=("cov_fisher and cov_bootstrap assume independent binomial trials; "
                "cov_halfsplit does not.  A category-3 scheme must decide which "
                "scale it propagates and must also carry the REPRESENTATION error "
                "(the transfer residual to London-0/Saclay-0, -1.2 / -1.6 % in the "
                "reported window), which no coefficient covariance contains."))
    out = out or f"{ROOT}/C1nsadd_covariance_{fam}.npz"
    np.savez(out, beta=beta, cov_fisher=cov_fisher, cov_bootstrap=cov_boot,
             cov_halfsplit=cov_halfsplit, beta_bootstrap=boots,
             beta_half_even=stE["beta"], beta_half_odd=stO["beta"],
             sd_fisher=sd_f, sd_bootstrap=sd_b, sd_halfsplit=sd_h,
             x_b=x_b, u_live=u_live, live_idx=li,
             provenance=np.array(json.dumps(prov_out, indent=1), dtype=object))
    print(f"wrote {out}")
    print("  beta      ", np.array2string(beta, precision=5))
    print("  sd Fisher ", np.array2string(sd_f, precision=5))
    print("  sd boot   ", np.array2string(sd_b, precision=5))
    print("  sd half   ", np.array2string(sd_h, precision=5))
    print("  over-disp ", np.array2string(sd_h / sd_f, precision=3))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="2lpt0")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    build(a.family, a.n_boot, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
