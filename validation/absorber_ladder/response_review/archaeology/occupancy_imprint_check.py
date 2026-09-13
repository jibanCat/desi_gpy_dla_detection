#!/usr/bin/env python
"""Deterministic occupancy-imprint check for the R1d smoother and the R0/R1c
moment surfaces (response-representation review, 2026-09-13).

EXPERIMENT 1 (deterministic; NO sampling noise):
  hold the per-row CONDITIONAL data fixed at its measured value
  yhat[k,b] = A[k,b]/N_b  (the raw, exactly-conditional row estimator)
  and change ONLY the weight vector the smoother uses,
      W  = diag(N_b)            (as implemented)
      W' = diag(N_b * t_b),  t_b = 10^(g (centre_b - 20.5)),  g = +-0.5
  i.e. exactly what a calibration mock with a CDDF slope shifted by g dex^-1
  would hand the smoother.  Any change in the fitted row is PURE BIAS of the
  smoother -- the conditional inputs are byte-identical.

EXPERIMENT 2: sampling-noise floor -- sightline bootstrap of the unmodified
  unit-weight fit, same metric.

VALIDATION-ONLY, read-only on all tracked products.
"""
import json
import os
import sys

import numpy as np

RESP = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/response"
sys.path.insert(0, "/home/mfho/wt_abs_diag_2026-09/validation/absorber_ladder/response")
import r1d_empirical as R1D          # noqa: E402
import respfit as RF                 # noqa: E402

TB = np.array([19.0, 19.2, 19.5, 19.7, 19.9, 20.1, 20.3, 20.5, 20.7, 20.9,
               21.1, 21.3, 21.5, 21.7, 21.9, 22.1, 22.4])
OB = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 3)
NC = 0.5 * (TB[:-1] + TB[1:])


def kl(p, q, eps=1e-12):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = p > eps
    return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], eps, None))))


def main():
    d = np.load(os.path.join(RESP, "response_R1d.npz"), allow_pickle=True)
    A = np.asarray(d["raw_counts"], float)        # (3,3,2J+1,B)
    tot = np.asarray(d["row_totals"], float)      # (3,3,B)
    c0 = np.asarray(d["c0"], int)
    deg = int(d["deg"]); lam = float(d["ridge_lambda"])
    N_ref = float(d["N_ref"])
    C = len(OB) - 1
    out = dict(deg=deg, ridge_lambda=lam, N_ref=N_ref,
               n_events=float(tot.sum()))

    p0, edof0 = R1D.smooth_along_N(A, tot, NC, N_ref, deg, lam)
    m0, _ = R1D.to_masses(p0, c0, C)
    out["edof_unit"] = float(edof0)

    # raw (unsmoothed) conditional rows, for the lack-of-fit diagnostic
    y_raw = np.where(tot[:, :, None, :] > 0,
                     A / np.maximum(tot[:, :, None, :], 1e-12), 0.0)

    # ---------------- EXPERIMENT 1: weight profile only -------------------
    out["exp1"] = {}
    for g in (-0.5, 0.5, -1.0, 1.0):
        t = 10.0 ** (g * (NC - 20.5))
        totp = tot * t[None, None, :]
        Ap = A * t[None, None, None, :]           # y = A/tot is UNCHANGED
        p1, edof1 = R1D.smooth_along_N(Ap, totp, NC, N_ref, deg, lam)
        m1, _ = R1D.to_masses(p1, c0, C)
        rows = []
        for i in range(3):
            for j in range(3):
                for b in range(len(NC)):
                    if tot[i, j, b] <= 0:
                        continue
                    rows.append(dict(cell=f"{i}{j}", b=b, Nc=float(NC[b]),
                                     n=float(tot[i, j, b]),
                                     kl=kl(m0[i, j, :, b], m1[i, j, :, b]),
                                     dmean=float(
                                         np.sum(m1[i, j, :, b] * (OB[:-1] + 0.05))
                                         - np.sum(m0[i, j, :, b] * (OB[:-1] + 0.05)))))
        k = np.array([r["kl"] for r in rows])
        out["exp1"][f"tilt{g:+.1f}"] = dict(
            edof=float(edof1),
            kl_median=float(np.median(k)), kl_p90=float(np.percentile(k, 90)),
            kl_max=float(k.max()),
            dmean_max_abs=float(max(abs(r["dmean"]) for r in rows)),
            worst=sorted(rows, key=lambda r: -r["kl"])[:6])

    # lack of fit of the deg-2 basis on the CONDITIONAL rows (unit weights)
    lof = []
    for i in range(3):
        for j in range(3):
            for b in range(len(NC)):
                if tot[i, j, b] <= 0:
                    continue
                lof.append(kl(y_raw[i, j, :, b] /
                              max(y_raw[i, j, :, b].sum(), 1e-12),
                              p0[i, j, :, b]))
    out["lack_of_fit_raw_vs_smoothed_KL"] = dict(
        median=float(np.median(lof)), p90=float(np.percentile(lof, 90)),
        max=float(np.max(lof)))

    # ---------------- EXPERIMENT 2: bootstrap noise floor -----------------
    ev = RF.load_events(os.path.join(RESP, "calib_events_2lpt0.npz"))
    isr, izr = RF.cell_index(ev["snr"], ev["zqso"])
    b_i = np.clip(np.digitize(ev["N_true"], TB) - 1, 0, len(NC) - 1)
    tids, inv = np.unique(ev["tid"], return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(tids)))
    ends = np.append(starts[1:], len(order))
    rng = np.random.default_rng(20260913)
    kls = []
    for _ in range(int(os.environ.get("NBOOT", 12))):
        pick = rng.integers(0, len(tids), len(tids))
        idx = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        Ab, totb, _ = R1D.raw_masses(ev["N_true"][idx], ev["xhat"][idx],
                                     isr[idx], izr[idx], b_i[idx], TB, OB, 15)
        pb, _ = R1D.smooth_along_N(Ab, totb, NC, N_ref, deg, lam)
        mb, _ = R1D.to_masses(pb, c0, C)
        for i in range(3):
            for j in range(3):
                for b in range(len(NC)):
                    if tot[i, j, b] > 0 and totb[i, j, b] > 0:
                        kls.append(kl(m0[i, j, :, b], mb[i, j, :, b]))
    kls = np.array(kls)
    out["bootstrap_noise_floor_KL"] = dict(
        n=int(kls.size), median=float(np.median(kls)),
        p90=float(np.percentile(kls, 90)), max=float(kls.max()))

    # ------- R0/R1c moment surfaces: same deterministic weight swap -------
    mask = np.ones(len(ev["dx"]), bool)
    rows_pc = RF._rows_for_cells(ev["N_true"][mask], ev["dx"][mask],
                                 isr[mask], izr[mask], RF.SPEC_R0)
    s0, rng0, _ = RF.surfaces_shared(rows_pc, N_ref, 3, deg_cell=2)
    r0m = {}
    for g in (-0.5, 0.5):
        rows_t = [[[dict(r, n=r["n"] * 10.0 ** (g * (r["c"] - 20.5)))
                    for r in rows_pc[i][j]] for j in range(3)] for i in range(3)]
        s1, rng1, _ = RF.surfaces_shared(rows_t, N_ref, 3, deg_cell=2)
        Ngrid = np.arange(19.6, 21.4, 0.1)
        dmu = []; dsg = []; dsk = []
        for i in range(3):
            for j in range(3):
                mu0, sg0, sk0 = RF.eval_moments(
                    s0, rng0, N_ref, Ngrid, np.full(len(Ngrid), i),
                    np.full(len(Ngrid), j), skew_ramp=(21.0, 0.5))
                mu1, sg1, sk1 = RF.eval_moments(
                    s1, rng1, N_ref, Ngrid, np.full(len(Ngrid), i),
                    np.full(len(Ngrid), j), skew_ramp=(21.0, 0.5))
                dmu.append(np.abs(mu1 - mu0)); dsg.append(np.abs(sg1 - sg0))
                dsk.append(np.abs(sk1 - sk0))
        r0m[f"tilt{g:+.1f}"] = dict(
            max_abs_dmu_dex=float(np.max(dmu)),
            max_abs_dsigma_dex=float(np.max(dsg)),
            max_abs_dskew=float(np.max(dsk)))
    out["R0_moment_surface_weight_swap"] = r0m

    print(json.dumps(out, indent=1, default=float))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "occupancy_imprint_result.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
