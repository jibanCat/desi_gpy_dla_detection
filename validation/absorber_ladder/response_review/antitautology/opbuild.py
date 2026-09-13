#!/usr/bin/env python
"""opbuild.py — the GENERIC operator-build interface for the anti-tautology
stress tests (PI ruling 2026-09-13c §3/§4; sealed
``RESPONSE_FAMILY_OPENING_RULE_PREDECLARATION.md`` §6).

The object under test is always the CONDITIONAL response row

    P(c | b, s, K)  =  P(observed N-hat cell c | latent bin b, S/N stratum s,
                         coarse-z block K, detected),

represented internally as a matrix ``P[r, c]`` over an operator's OWN native
row space ``r`` (its native resolution), plus the (S, Kf, C, B) gathered tensor
``Mg`` the production fold consumes, built under the ratified
count-conservation convention (unit in-grid row x the pack's frozen
``adopted_phi_ref``) exactly as
``build_variants.py:620-634`` / ``cc_posterior_validation.build_cc_tensors``.

Row spaces
----------
``resp`` : (i_snr, i_z, b) — 3 x 3 x 16 = 144 rows.  The native space of the
    production response object (R0..R1c) and of R1d; the S/N and z cells are
    ``respfit.SNR_EDGES`` / ``respfit.Z_EDGES`` on (snr, z_qso).
``sKb``  : (s, K, b) — 8 x 3 x 16 = 384 rows.  The native space of the
    forensic empirical operator ``M_true_sKcb`` (S/N stratum on ``snr_edges``,
    coarse-z block on ``zc_edges`` applied to z_DLA), i.e. the (b, s, K) row
    space the sealed protocol §3 names.

Everything here is CALIBRATION-SIDE ONLY: the sole data are the 73,845 2LPT-0
natural-pair matched calibration events.  No HBI run, no real data, no mock
closure number is read anywhere in this package.

VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_RESP = os.path.join(_REPO, "validation", "absorber_ladder", "response")
for _p in (_RESP, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import respfit as R                                              # noqa: E402
import opmetrics as M                                            # noqa: E402
import r1d_empirical as R1D                                      # noqa: E402

SCRATCH = ("/scratch/cavestru_root/cavestru0/mfho/"
           "absorber_ladder_2026-09-13")
EVENTS = os.path.join(SCRATCH, "response", "calib_events_2lpt0.npz")
OPS_TPL = os.path.join(SCRATCH, "support", "empirical_ops_{fam}_A0.npz")
SCANPACK_TPL = ("/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/"
                "packs/scanpack_{fam}_b300.npz")
ADOPTED_NPZ = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
               "track_c/stage0/adopted_response_v1p1.npz")

# R1c and R1d hyper-parameters AS SELECTED by the first ladder's CV
# (variants_report.json: R1c_model_selection -> {"estimator": "ml",
# "marg_deg": 3};  R1d_model_selection -> {"deg": 2, "ridge_lambda": 0.0}).
R1C_SPEC = dict(name="R1c", edges="adaptive", lo=19.0, hi=22.4, step=0.1,
                min_n=50, hard_min_n=25, max_width=0.3, estimator="ml",
                deg_cell=2, deg_shared=3, fit_rng="full", ramp=None,
                marginalise=dict(n_quad=17, deg=3, weight="flat"), defect="")
R1D_DEG = 2
R1D_LAM = 0.0
R1D_J = 15

EPS_FLOOR = 1e-6        # uniform floor mixed into a row before any log/KL


# ===========================================================================
# geometry + events
# ===========================================================================
def load_geometry(fam="2lpt0"):
    """Bin edges and gather maps, cross-checked between the A0 empirical-ops
    product and the scan pack (fail-closed)."""
    ops = np.load(OPS_TPL.format(fam=fam), allow_pickle=True)
    pk = np.load(SCANPACK_TPL.format(fam=fam), allow_pickle=True)
    g = dict(
        ntrue=np.asarray(ops["ntrue_edges"], float),
        nhat=np.asarray(ops["nhat_edges"], float),
        snr=np.asarray(ops["snr_edges"], float),
        zc=np.asarray(ops["zc_edges"], float),
        zf=np.asarray(ops["zf_edges"], float),
        kz2K=np.asarray(ops["kz_to_K"], int),
        dX=np.asarray(pk["dX"], float),
        phi_ref=np.asarray(pk["adopted_phi_ref"], float),
        sig_floor=float(pk["resp_sig_floor"]),
        N_ref=float(np.load(ADOPTED_NPZ, allow_pickle=True)["N_ref"]),
        ops_path=OPS_TPL.format(fam=fam),
        pack_path=SCANPACK_TPL.format(fam=fam))
    for key, pkey in (("ntrue", "ntrue_edges"), ("nhat", "nhat_edges"),
                      ("snr", "snr_edges"), ("zc", "zc_edges"),
                      ("zf", "zf_edges")):
        if not np.array_equal(g[key], np.asarray(pk[pkey], float)):
            raise SystemExit(f"GEOMETRY GATE FAILED: {key} differs between "
                             f"{g['ops_path']} and {g['pack_path']}")
    if not np.array_equal(g["kz2K"], np.asarray(pk["kz_to_K"], int)):
        raise SystemExit("GEOMETRY GATE FAILED: kz_to_K differs")
    rse = np.asarray(pk["resp_snr_edges"], float)
    rze = np.asarray(pk["resp_z_edges"], float)
    g["s2sr"] = np.clip(np.searchsorted(rse, g["snr"][:-1] + 1e-9, "right")
                        - 1, 0, 2)
    g["K2zr"] = np.searchsorted(rze, 0.5 * (g["zc"][:-1] + g["zc"][1:]),
                                "right") - 1
    g["B"] = len(g["ntrue"]) - 1
    g["C"] = len(g["nhat"]) - 1
    g["S"] = len(g["snr"]) - 1
    g["Kf"] = len(g["kz2K"])
    g["KK"] = int(g["kz2K"].max()) + 1
    g["Nc"] = 0.5 * (g["ntrue"][:-1] + g["ntrue"][1:])
    g["cen"] = 0.5 * (g["nhat"][:-1] + g["nhat"][1:])
    g["dN"] = np.diff(g["ntrue"])
    return g


def load_events(geom, path=EVENTS):
    """Calibration events + every index the row spaces need."""
    ev = R.load_events(path)
    ev["isr"], ev["izr"] = R.cell_index(ev["snr"], ev["zqso"])
    ev["fold"] = R.parity_fold(ev["tid"])
    ev["b_i"] = np.clip(np.digitize(ev["N_true"], geom["ntrue"]) - 1, 0,
                        geom["B"] - 1)
    ev["s_i"] = np.clip(np.digitize(ev["snr"], geom["snr"]) - 1, 0,
                        geom["S"] - 1)
    ev["K_i"] = np.clip(np.digitize(ev["zdla"], geom["zc"]) - 1, 0,
                        geom["KK"] - 1)
    c = np.digitize(ev["xhat"], geom["nhat"]) - 1
    ev["c_i"] = c
    ev["in_grid"] = (c >= 0) & (c < geom["C"])
    ev["n"] = len(ev["dx"])
    return ev


# ===========================================================================
# weightings (§6A: "weights enter the estimator wherever counts do")
# ===========================================================================
def weights_slope(N_true, dgamma, N_piv=20.5):
    """w(N_true) proportional to 10^{dgamma (N_true - N_piv)} — a population
    flatter (dgamma < 0) or steeper (dgamma > 0) than the calibration mock's
    own f(N).  Normalised to mean 1 so weighted counts stay comparable."""
    w = 10.0 ** (float(dgamma) * (np.asarray(N_true, float) - float(N_piv)))
    return w / w.mean()


def weights_equal_occupancy(b_i, B, min_n=1):
    """w proportional to 1 / N_b over the usable true-N rows: every latent bin
    carries the same total weight.  Bins with < ``min_n`` events get w = 0
    (they cannot be equalised)."""
    n_b = np.bincount(np.asarray(b_i, int), minlength=B).astype(float)
    inv = np.where(n_b >= min_n, 1.0 / np.maximum(n_b, 1.0), 0.0)
    w = inv[b_i]
    m = w.mean()
    return w / m if m > 0 else w


WEIGHTINGS = {
    "native": lambda ev, g: np.ones(ev["n"]),
    "flatter_dg-0.5": lambda ev, g: weights_slope(ev["N_true"], -0.5),
    "steeper_dg+0.5": lambda ev, g: weights_slope(ev["N_true"], +0.5),
    "equal_occupancy": lambda ev, g: weights_equal_occupancy(ev["b_i"],
                                                             g["B"]),
}


# ===========================================================================
# the Operator container
# ===========================================================================
class Operator:
    """A fixed conditional response object.

    ``P``  : (n_rows, C) conditional rows, each summing to 1 over the observed
             grid (the count-conservation "had mass" convention: the in-grid
             fraction is carried separately by ``phi_ref``).
    ``Mg`` : (S, Kf, C, B) gathered tensor with rows summing to
             ``phi_ref[sr, zr, b]`` — the runner's ``--mg-fixed`` shape.
    """

    def __init__(self, kind, P, grid, geom, row_n=None, meta=None):
        self.kind = kind
        self.grid = grid
        self.geom = geom
        self.P = np.asarray(P, float)
        self.row_n = (np.zeros(self.P.shape[0]) if row_n is None
                      else np.asarray(row_n, float))
        self.meta = dict(meta or {})
        self.n_rows, self.C = self.P.shape

    # -- row space -------------------------------------------------------
    def row_index(self, ev, sel=None):
        return row_index(self.grid, ev, self.geom, sel)

    def row_b(self):
        return row_b(self.grid, self.geom)

    # -- gather ----------------------------------------------------------
    @property
    def Mg(self):
        return gather(self.P, self.grid, self.geom)

    # -- per-event predictive --------------------------------------------
    def logp_events(self, ev, sel=None, eps=EPS_FLOOR):
        r = self.row_index(ev, sel)
        c = ev["c_i"] if sel is None else ev["c_i"][sel]
        Pf = floor_rows(self.P, eps)
        return np.log(Pf[r, np.clip(c, 0, self.C - 1)])


def floor_rows(P, eps=EPS_FLOOR):
    """Mix a uniform floor into every row so logs and KLs are finite.
    p~ = (1-eps) p + eps / C."""
    P = np.asarray(P, float)
    C = P.shape[-1]
    s = P.sum(axis=-1, keepdims=True)
    Pn = np.divide(P, np.where(s > 0, s, 1.0))
    Pn = np.where(s > 0, Pn, 1.0 / C)
    return (1.0 - eps) * Pn + eps / C


def row_shape(grid, geom):
    if grid == "resp":
        return (3, 3, geom["B"])
    if grid == "sKb":
        return (geom["S"], geom["KK"], geom["B"])
    raise ValueError(grid)


def row_index(grid, ev, geom, sel=None):
    """Flat native-row index of each event."""
    sh = row_shape(grid, geom)
    if grid == "resp":
        a, b, c = ev["isr"], ev["izr"], ev["b_i"]
    else:
        a, b, c = ev["s_i"], ev["K_i"], ev["b_i"]
    if sel is not None:
        a, b, c = a[sel], b[sel], c[sel]
    return np.ravel_multi_index((a, b, c), sh)


def row_b(grid, geom):
    """Latent bin index of every native row."""
    sh = row_shape(grid, geom)
    return np.tile(np.arange(sh[2]), sh[0] * sh[1])


def row_meta(grid, geom):
    """(i0, i1, b) triples of every native row."""
    sh = row_shape(grid, geom)
    return np.array(list(np.ndindex(*sh)), int)


def gather(P, grid, geom):
    """(n_rows, C) conditional rows -> (S, Kf, C, B) Mg under the ratified
    count-conservation convention (row x adopted_phi_ref)."""
    sh = row_shape(grid, geom)
    Q = P.reshape(sh[0], sh[1], sh[2], geom["C"])          # (A, Bx, B, C)
    Q = np.transpose(Q, (0, 1, 3, 2))                      # (A, Bx, C, B)
    if grid == "resp":
        g = Q[geom["s2sr"][:, None], geom["K2zr"][geom["kz2K"]][None, :],
              :, :]
        phi = geom["phi_ref"][geom["s2sr"][:, None],
                              geom["K2zr"][geom["kz2K"]][None, :], None, :]
    else:
        g = Q[np.arange(geom["S"])[:, None], geom["kz2K"][None, :], :, :]
        sr = geom["s2sr"][:, None]
        zr = geom["K2zr"][geom["kz2K"]][None, :]
        phi = geom["phi_ref"][sr, zr, None, :]
    return g * phi


# ===========================================================================
# the builders
# ===========================================================================
def _weighted_counts(ev, w, grid, geom, sel=None):
    """(n_rows, C) weighted in-grid counts and (n_rows,) weighted row totals."""
    sh = row_shape(grid, geom)
    n_rows = int(np.prod(sh))
    ok = ev["in_grid"] if sel is None else (ev["in_grid"] & sel)
    r = row_index(grid, ev, geom, ok)
    c = ev["c_i"][ok]
    A = np.zeros((n_rows, geom["C"]))
    np.add.at(A, (r, c), np.asarray(w, float)[ok])
    n_raw = np.zeros(n_rows)
    np.add.at(n_raw, r, 1.0)
    return A, n_raw


def build_r1d_raw(ev, w, geom, sel=None, grid="resp"):
    """R1d-raw — the SATURATED empirical conditional row, smoothed only by the
    Jeffreys +1/2 (sealed §3).  No pooling across rows of any kind."""
    A, n_raw = _weighted_counts(ev, w, grid, geom, sel)
    P = A + 0.5
    P = P / P.sum(axis=1, keepdims=True)
    return Operator("R1d_raw", P, grid, geom, row_n=n_raw,
                    meta=dict(prior="Jeffreys +1/2 on all %d observed cells"
                              % geom["C"]))


def build_mtrue_emp(ev, w, geom, sel=None, grid="sKb"):
    """M_true-like: the pure empirical conditional row on the (b, s, K) grid
    with NO regularisation at all (the construction of the forensic
    ``M_true_sKcb``, build_matched_ops.py:467-478).  Empty rows are left at
    zero and are excluded by the >= 200-event row gate."""
    A, n_raw = _weighted_counts(ev, w, grid, geom, sel)
    s = A.sum(axis=1, keepdims=True)
    P = np.divide(A, np.where(s > 0, s, 1.0))
    P = np.where(s > 0, P, 1.0 / geom["C"])
    return Operator("M_true_emp", P, grid, geom, row_n=n_raw,
                    meta=dict(prior="none (raw counts, rows renormalised)"))


def build_r1d(ev, w, geom, sel=None, grid="resp", deg=R1D_DEG, lam=R1D_LAM,
              J=R1D_J):
    """R1d AS IMPLEMENTED — smoothed empirical offset masses.

    Reuses ``r1d_empirical.smooth_along_N`` and ``r1d_empirical.to_masses``
    UNMODIFIED; only the raw-count accumulation is generalised to carry event
    weights (``r1d_empirical.raw_masses`` hard-codes 1.0 at line 62).  With
    w == 1 this reproduces ``r1d_empirical.fit_r1d`` exactly (tested,
    atol = 0)."""
    nt, ne = geom["ntrue"], geom["nhat"]
    Nc, C, B = geom["Nc"], geom["C"], geom["B"]
    m = np.ones(ev["n"], bool) if sel is None else np.asarray(sel, bool)
    c_idx = ev["c_i"]
    c0 = np.clip(np.digitize(Nc, ne) - 1, 0, C - 1)
    j = c_idx - c0[ev["b_i"]]
    ok = m & (c_idx >= 0) & (c_idx < C) & (np.abs(j) <= J)
    A = np.zeros((3, 3, 2 * J + 1, B))
    np.add.at(A, (ev["isr"][ok], ev["izr"][ok], j[ok] + J, ev["b_i"][ok]),
              np.asarray(w, float)[ok])
    tot = A.sum(axis=2)
    p, edof = R1D.smooth_along_N(A, tot, Nc, geom["N_ref"], deg, lam)
    masses, _phi = R1D.to_masses(p, c0, C)                 # (3, 3, C, B)
    P = np.transpose(masses, (0, 1, 3, 2)).reshape(9 * B, C)
    n_raw = np.zeros(9 * B)
    rr = row_index("resp", ev, geom, m & ev["in_grid"])
    np.add.at(n_raw, rr, 1.0)
    return Operator("R1d", P, "resp", geom, row_n=n_raw,
                    meta=dict(deg=deg, ridge_lambda=lam, J=J,
                              effective_dof=float(edof)))


def build_r1c(ev, w, geom, sel=None, grid="resp", seed=0, spec=None,
              force_resample=False):
    """R1c — the parametric skew-normal moment family.

    ``respfit`` has NO weighted path end-to-end (``subbin_moments`` /
    ``subbin_moments_weighted_centre`` / ``adaptive_edges`` all take raw
    counts; only the inner ``fit_subbin_ml`` accepts ``w``), and this package
    may not modify tracked files, so a weighting is realised by WEIGHTED
    RESAMPLING with a fixed seed: n events drawn with replacement with
    probability proportional to w.  The native build is drawn the same way
    with w == 1 wherever a resampled comparison is needed, so the Monte-Carlo
    noise is matched on both sides of every comparison."""
    sp = dict(spec or R1C_SPEC)
    m = np.ones(ev["n"], bool) if sel is None else np.asarray(sel, bool)
    idx = np.flatnonzero(m)
    ww = np.asarray(w, float)[idx]
    if (not force_resample) and np.allclose(ww, ww[0], atol=0.0, rtol=0.0):
        take = idx                                          # exact, no MC
        resampled = False
    else:
        rng = np.random.default_rng(int(seed))
        p = ww / ww.sum()
        take = idx[rng.choice(len(idx), size=len(idx), replace=True, p=p)]
        resampled = True
    obj = R.fit_variant(ev["N_true"][take], ev["dx"][take], ev["isr"][take],
                        ev["izr"][take], geom["N_ref"], sp,
                        ntrue_edges=geom["ntrue"])
    masses, phi = M.model_masses(obj, geom["ntrue"], geom["nhat"],
                                 sig_floor=geom["sig_floor"])
    # normalise by the object's OWN in-grid fraction phi, exactly as
    # build_variants.py:620-624 does, so the gathered tensor is bit-identical
    # to the delivered Mg_R1c_<fam>.npz.
    masses = masses / np.maximum(phi, 1e-12)[:, :, None, :]
    P = np.transpose(masses, (0, 1, 3, 2)).reshape(9 * geom["B"], geom["C"])
    n_raw = np.zeros(9 * geom["B"])
    rr = row_index("resp", ev, geom, m & ev["in_grid"])
    np.add.at(n_raw, rr, 1.0)
    return Operator("R1c", P, "resp", geom, row_n=n_raw,
                    meta=dict(resampled=resampled, seed=int(seed),
                              spec=sp, obj=obj))


# ===========================================================================
# the FOLD's completeness / in-grid-fraction convention (coordinator note,
# 2026-09-13: C_true_bKs ALREADY carries phi, so C_true x a phi_ref-scaled
# parametric row applies phi TWICE)
# ===========================================================================
def completeness_and_phi(geom, fam="2lpt0"):
    """Split the A0 truth completeness into the DETECTION probability and the
    measured IN-GRID fraction — convention (a), the fold's own convention.

        C_det[b,K,s] = sum_k N_det_all_bks_true_z / sum_k truth_counts_bks
        phi [b,K,s]  = sum_k N_det_bks_true_z    / sum_k N_det_all_bks_true_z
        C_true       = C_det * phi                      (checked, fail-closed)

    ``C_true_bKs`` (build_matched_ops.py:460) counts only IN-GRID detections in
    its numerator, so it already carries phi; the parametric Mg rows sum to the
    frozen ``adopted_phi_ref``, so ``C_true x Mg_parametric`` double-counts the
    in-grid fraction.  Test C therefore folds with ``C_det`` and rows scaled by
    the MEASURED phi, for every operator alike.
    """
    z = np.load(OPS_TPL.format(fam=fam), allow_pickle=True)
    kz = np.asarray(geom["kz2K"], int)
    KK = geom["KK"]

    def agg(X):
        X = np.asarray(X, float)
        return np.stack([X[:, kz == K, :].sum(axis=1) for K in range(KK)],
                        axis=1)                              # (B, KK, S)

    din = agg(z["N_det_bks_true_z"])
    dall = agg(z["N_det_all_bks_true_z"])
    tc = agg(z["truth_counts_bks"])
    C_det = np.divide(dall, np.where(tc > 0, tc, 1.0))
    phi = np.divide(din, np.where(dall > 0, dall, 1.0))
    C_true = np.asarray(z["C_true_bKs"], float)
    resid = float(np.max(np.abs(C_det * phi - C_true)))
    if resid > 1e-12:
        raise SystemExit("PHI CONVENTION GATE FAILED: C_det * phi != C_true "
                         f"(max |diff| = {resid:g})")
    return dict(C_det=C_det, phi=phi, C_true=C_true,
                identity_max_abs=resid,
                n_cells_C_det_gt1=int(np.sum(C_det > 1.0 + 1e-12)),
                n_cells_supported=int(np.sum(tc > 0)),
                truth_counts_bKs=tc, n_det_all_bKs=dall, n_det_ingrid_bKs=din,
                source=OPS_TPL.format(fam=fam))


def gather_with_phi(P, grid, geom, phi_bKs):
    """(n_rows, C) unit-sum conditional rows -> (S, Kf, C, B) with the MEASURED
    in-grid fraction phi[b, K, s] instead of the frozen ``adopted_phi_ref``."""
    sh = row_shape(grid, geom)
    Q = np.transpose(P.reshape(sh[0], sh[1], sh[2], geom["C"]), (0, 1, 3, 2))
    if grid == "resp":
        g = Q[geom["s2sr"][:, None], geom["K2zr"][geom["kz2K"]][None, :], :, :]
    else:
        g = Q[np.arange(geom["S"])[:, None], geom["kz2K"][None, :], :, :]
    ph = np.transpose(np.asarray(phi_bKs, float), (2, 1, 0))  # (S, KK, B)
    ph = ph[:, geom["kz2K"], :][:, :, None, :]                # (S, Kf, 1, B)
    return g * ph


def phi_ref_vs_measured(geom, phi_bKs):
    """The audit the coordinator asked for: the frozen parametric
    ``adopted_phi_ref`` against the MEASURED in-grid fraction."""
    B, KK, S = phi_bKs.shape
    ref = geom["phi_ref"][geom["s2sr"][:, None], geom["K2zr"][None, :], :]
    ref = np.transpose(ref, (2, 1, 0))                        # (B, KK, S)
    ok = phi_bKs > 0
    r = np.divide(ref, np.where(ok, phi_bKs, 1.0))
    by_b, by_s = {}, {}
    nt = geom["ntrue"]
    for b in range(B):
        m = ok[b]
        if m.any():
            by_b[f"[{nt[b]:.1f},{nt[b+1]:.1f})"] = dict(
                min=float(r[b][m].min()), median=float(np.median(r[b][m])),
                max=float(r[b][m].max()), n_cells=int(m.sum()))
    se = geom["snr"]
    for s in range(S):
        m = ok[:, :, s]
        if m.any():
            by_s[f"S/N[{se[s]:.0f},{se[s+1]:.0f})"] = dict(
                min=float(r[:, :, s][m].min()),
                median=float(np.median(r[:, :, s][m])),
                max=float(r[:, :, s][m].max()), n_cells=int(m.sum()))
    return dict(ratio_phi_ref_over_measured_by_true_N_bin=by_b,
                ratio_phi_ref_over_measured_by_snr=by_s,
                overall=dict(min=float(r[ok].min()),
                             median=float(np.median(r[ok])),
                             max=float(r[ok].max())),
                note=("phi_ref is the DEPLOYED parametric kernel's in-grid "
                      "fraction, frozen in the pack; the measured phi is "
                      "N_det_bks_true_z / N_det_all_bks_true_z.  A ratio != 1 "
                      "means the deployed row puts a different fraction of "
                      "its mass on the >= 19.5 observed grid than the "
                      "calibration mock actually does."))


def build_mtrue_frozen(geom, fam="2lpt0"):
    """The FROZEN forensic operator ``M_true_sKcb`` read straight off
    ``empirical_ops_<fam>_A0.npz`` (not rebuilt).  Reference for tests C/D."""
    z = np.load(OPS_TPL.format(fam=fam), allow_pickle=True)
    Mt = np.asarray(z["M_true_sKcb"], float)                # (S, KK, C, B)
    n = np.asarray(z["M_counts_sKcb"], float).sum(axis=2)   # (S, KK, B)
    P = np.transpose(Mt, (0, 1, 3, 2)).reshape(-1, geom["C"])
    return Operator("M_true_frozen", P, "sKb", geom, row_n=n.reshape(-1),
                    meta=dict(source=OPS_TPL.format(fam=fam),
                              note="empty rows backfilled by the model kernel "
                                   "(build_matched_ops.py:478)"))


BUILDERS = {
    "R1d_raw": build_r1d_raw,
    "R1d": build_r1d,
    "R1c": build_r1c,
    "M_true_emp": build_mtrue_emp,
}


def build_operator(events, weights, kind, geom=None, **kw):
    """THE generic interface.  ``build_operator(events, weights, kind)`` ->
    ``Operator`` (``.Mg`` is the (S, Kf, C, B) tensor; ``.P`` the conditional
    rows).  ``weights`` may be None (native), an array of per-event weights,
    or a key of ``WEIGHTINGS``."""
    geom = geom or load_geometry()
    if weights is None:
        w = np.ones(events["n"])
    elif isinstance(weights, str):
        w = WEIGHTINGS[weights](events, geom)
    else:
        w = np.asarray(weights, float)
    if kind in BUILDERS:
        return BUILDERS[kind](events, w, geom, **kw)
    if kind in EXTRA_BUILDERS:
        return EXTRA_BUILDERS[kind](events, w, geom, **kw)
    raise KeyError(f"unknown operator kind {kind!r}; known: "
                   f"{sorted(BUILDERS) + sorted(EXTRA_BUILDERS)}")


EXTRA_BUILDERS = {}


def register_builder(name, fn):
    """Hook for a NEW low-DOF candidate: a candidate that registers
    ``fn(events, w, geom, sel=None, **kw) -> Operator`` is pushed through
    tests A and B unchanged.  A candidate that only ships an
    ``Mg_<cand>_2lpt0.npz`` can still be pushed through test C by
    ``operator_from_mg``."""
    EXTRA_BUILDERS[name] = fn


def operator_from_mg(path, geom, grid=None, kind=None):
    """Load a candidate delivered as ``Mg_<cand>_<fam>.npz``
    (schema absorber_ladder/Mg_fixed/v1, Mg of shape (S, Kf, C, B)) and
    de-gather it to conditional rows.  The row space is inferred: ``resp`` if
    the tensor is constant within every (s2sr, K2zr) block, else ``sKb``."""
    z = np.load(path, allow_pickle=True)
    Mg = np.asarray(z["Mg"], float)                         # (S, Kf, C, B)
    S, Kf, C, B = Mg.shape
    if grid is None:
        Q = np.transpose(Mg, (0, 1, 3, 2))
        grid = "resp"
        for sr in range(3):
            ss = np.flatnonzero(geom["s2sr"] == sr)
            if len(ss) > 1 and not np.allclose(Q[ss[0]], Q[ss[1:]], atol=0.0):
                grid = "sKb"
                break
    sh = row_shape(grid, geom)
    P = np.zeros((int(np.prod(sh)), C))
    meta = row_meta(grid, geom)
    for r, (i0, i1, b) in enumerate(meta):
        if grid == "resp":
            ss = np.flatnonzero(geom["s2sr"] == i0)
            kk = np.flatnonzero(geom["K2zr"][geom["kz2K"]] == i1)
        else:
            ss = np.array([i0])
            kk = np.flatnonzero(geom["kz2K"] == i1)
        if len(ss) == 0 or len(kk) == 0:
            P[r] = 1.0 / C
            continue
        row = Mg[ss[0], kk[0], :, b]
        t = row.sum()
        P[r] = row / t if t > 0 else 1.0 / C
    return Operator(kind or os.path.basename(path), P, grid, geom,
                    meta=dict(source=path, degathered=True))
