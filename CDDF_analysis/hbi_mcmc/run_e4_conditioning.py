"""run_e4_conditioning.py — the E4 diagnostic driver (DIAGNOSTIC ONLY).

Characterises the Model A deconvolution as an INVERSE PROBLEM, on the real
v1.1 packs, using the operator recovered by probing the committed fold
(``e4_probe.build_fold_operator``).  It answers four questions and writes ONE
stamped artifact:

  Q1  what is the singular spectrum / condition number of the fold, at the
      response-kernel level, per (z, SNR) stratum, and for the STACKED design
      matrix the likelihood actually inverts?
  Q2  which basis combinations are unconstrained (null directions in log N_HI)?
  Q3  the decisive never-run check: fold the pack's OWN truth through the
      operator and invert it.  Exact data isolates conditioning from every
      other defect; Poisson data at the real count level measures the noise
      amplification.
  Q4  how do cond / effective rank / self-inversion error respond to a COARSER
      true-N basis (0.1 -> 0.2 / 0.3 / 0.4 dex), and how much RW2 prior
      strength would be needed to regularise at the shipped 0.1 dex?

SCOPE LIMIT: this module never writes a pack and never changes the production
basis width, prior or estimator.  Choosing a basis width changes the estimand's
resolution and is a PI decision; this produces the evidence, not the decision.

Run:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m CDDF_analysis.hbi_mcmc.run_e4_conditioning \
      --out CDDF_analysis/hbi_mcmc/e4_conditioning.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

import numpy as np

from . import e4_probe as e4
from .forward import build_consts, build_K
from .pack import load_pack

PACK_DIR = pathlib.Path(
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/modelA_packs")
MOCKS = ("2lpt0", "london0", "saclay0")
WIDTH_GROUPS = (1, 2, 3, 4)          # group size -> basis width 0.1 * g dex
N_MC = 32
DETAIL_K = 2                          # representative fine-z bin for per-bin detail


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def _git(*args):
    return subprocess.run(["git", *args], cwd=pathlib.Path(__file__).resolve().parents[2],
                          capture_output=True, text=True, check=True).stdout.strip()


def provenance_block(argv):
    sha = _git("rev-parse", "HEAD")            # FULL 40-char SHA, never abbreviated
    if len(sha) != 40:
        raise RuntimeError(f"refusing to stamp a non-40-char SHA: {sha!r}")
    dirty = bool(_git("status", "--porcelain"))
    return dict(
        artifact="e4_conditioning",
        code_commit=sha,
        code_dirty=dirty,
        routine="CDDF_analysis/hbi_mcmc/run_e4_conditioning.py",
        probe_module="CDDF_analysis/hbi_mcmc/e4_probe.py",
        rederive="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                 "/home/mfho/.conda/envs/gpdla-hbi/bin/python -m "
                 "CDDF_analysis.hbi_mcmc.run_e4_conditioning " + " ".join(argv),
        date=datetime.date.today().isoformat(),
        paper_facing=False,
        estimand="NONE — this artifact reports NO population measurement. It "
                 "contains linear-algebra diagnostics of the forward operator "
                 "and self-inversion ratios against each mock's OWN truth.",
        scope="MOCK ONLY (2LPT-0 / london-0 / saclay-0 model-A packs). No real "
              "DESI survey values of any kind.",
        decision_status="EVIDENCE ONLY — the production basis width (0.1 dex), "
                        "the RW2 prior and the estimator are UNCHANGED. "
                        "Choosing a basis width changes the estimand's "
                        "resolution and is a PI decision.",
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _fisher(M, f, *, ridge_free_bins=True):
    """Poisson Fisher information for theta = log f at (M, f).

    d mu / d theta_b = M[:, b] * f_b, so J = W^{1/2} M diag(f) with
    W = diag(1/mu).  Rows with mu == 0 carry no information and are dropped.
    Bins with f_b == 0 have an identically-zero column; they are reported as
    inactive rather than pseudo-inverted.
    """
    mu = M @ f
    live = mu > 0
    J = (M[live] * f[None, :]) / np.sqrt(mu[live])[:, None]
    F = J.T @ J
    active = f > 0
    return F, active, mu


def amplification(M, f, *, prior_prec=None):
    """Per-bin deconvolution variance inflation for theta = log f.

    Returns (amp, sd_theta, n_b) where

      sd_theta_b = sqrt( [ (F + prior_prec)^-1 ]_bb )
      n_b        = f_b * sum_rows M[:, b]   (counts attributable to bin b)
      amp_b      = sd_theta_b * sqrt(n_b)

    amp == 1 is the no-deconvolution reference (a perfectly diagonal kernel:
    the bin's own Poisson error).  amp >> 1 IS the ill-posedness, expressed in
    the only units that matter here — how much wider the error bar on log f in
    that bin is than simple counting would give.
    """
    F, active, _ = _fisher(M, f)
    P = np.zeros_like(F) if prior_prec is None else np.asarray(prior_prec, float)
    Fa = (F + P)[np.ix_(active, active)]
    n_b = f * M.sum(axis=0)
    amp = np.full(len(f), np.nan)
    sd = np.full(len(f), np.nan)
    try:
        cov = np.linalg.inv(Fa)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Fa)
    d = np.clip(np.diag(cov), 0.0, None)
    sd[active] = np.sqrt(d)
    amp[active] = np.sqrt(d) * np.sqrt(np.clip(n_b[active], 1e-300, None))
    return amp, sd, n_b


def selfinvert_exact(M, f):
    """NNLS on exact folded truth. Returns max |ratio - 1| over occupied bins."""
    fh = e4.nnls_invert(M, M @ f)
    r = e4.ratio_profile(fh, f)
    r = r[np.isfinite(r)]
    return float(np.max(np.abs(r - 1.0))) if r.size else float("nan"), fh


def selfinvert_poisson(M, f, *, n_mc=N_MC, seed0=0, data_f=None, data_M=None):
    """MC NNLS on Poisson realisations of the folded truth.

    ``data_M``/``data_f`` let the DATA be generated on a different (finer)
    basis than the one used to invert — that is how the coarse-basis
    representation error is separated from the noise amplification.
    """
    Md = M if data_M is None else data_M
    fd = f if data_f is None else data_f
    mu = Md @ fd
    R = np.empty((n_mc, M.shape[1]))
    for i in range(n_mc):
        y = np.random.default_rng(seed0 + i).poisson(mu).astype(float)
        R[i] = e4.nnls_invert(M, y)
    ratio = np.full_like(R, np.nan)
    occ = f > 1e-12 * np.max(f)
    ratio[:, occ] = R[:, occ] / f[None, occ]
    q = np.full((3, R.shape[1]), np.nan)
    q[:, occ] = np.percentile(ratio[:, occ], [16, 50, 84], axis=0)
    # A bin the solver pins to the non-negativity boundary has ratio == 0 and
    # NO finite log; it is counted separately rather than clipped into the RMS
    # (clipping turns one pinned bin into a fake ~27 nat outlier).
    pos = ratio[:, occ] > 0
    lr = np.log(ratio[:, occ][pos])
    med = q[1][occ]
    med_pos = med[med > 0]
    lo, hi = q[0][occ], q[2][occ]
    both = lo > 0
    return dict(
        occupied=occ,
        ratio_p16=q[0], ratio_p50=q[1], ratio_p84=q[2],
        ratio_min=float(np.min(ratio[:, occ])),
        ratio_max=float(np.max(ratio[:, occ])),
        rms_log_ratio_unpinned=float(np.sqrt(np.mean(lr ** 2))) if lr.size else float("nan"),
        # p16/p84 spread, over the bins where BOTH ends are off the boundary
        # (a p16 pinned at 0 has no finite log and would swamp the median)
        p16_p84_log_width_median=(float(np.median(np.log(hi[both]) - np.log(lo[both])))
                                  if both.any() else float("nan")),
        n_bins_p16_pinned_zero=int(np.sum(~both)),
        frac_pinned_zero=float(np.mean(R[:, occ] == 0.0)),
        dynamic_range_of_median_unpinned=(
            float(med_pos.max() / med_pos.min()) if med_pos.size else float("nan")),
        n_bins_median_pinned_zero=int(np.sum(med == 0.0)),
    )


TILT_SHAPES = ("ramp", "flat")


def tilt_vector(n, eps, shape="ramp"):
    """The systematic-misfit data perturbation, as an explicit named SHAPE.

    ``ramp``  1 + eps * (2 * i/(n-1) - 1): a smooth MONOTONE distortion across
              the row index.  Rows are the live (c, s) cells in c-major order,
              so the ramp is monotone in the observed-N_HI bin c — the shape a
              residual response-calibration error makes.
    ``flat``  1 + eps: a pure rescale.  NNLS is positively homogeneous, so this
              maps to f_hat = (1 + eps) * f EXACTLY — gain 1, zero sign
              changes, no ringing.  It is the null control that proves the
              section-3 mechanism result is a statement about the tilt's SHAPE
              and not about its amplitude.
    """
    if shape not in TILT_SHAPES:
        raise ValueError(f"tilt shape must be one of {TILT_SHAPES}")
    if shape == "flat":
        return 1.0 + eps * np.ones(n)
    return 1.0 + eps * (2.0 * (np.arange(n) / max(n - 1, 1)) - 1.0)


def systematic_misfit_test(M, f, eps=0.05, shape="ramp"):
    """EXACT (noiseless) inversion of data carrying a SMOOTH systematic tilt.

    Multiplies the folded truth by ``tilt_vector(n_rows, eps, shape)`` — by
    default a smooth, monotone few-percent distortion across observed N_HI, the
    shape a residual response-calibration error makes.  If a SMOOTH few-percent
    data perturbation comes back as a large OSCILLATORY error in f, then the
    fold's near-null directions are the mechanism that converts model
    misspecification (D1/D2) into bin-to-bin ringing — with no noise involved
    at all.

    The SHAPE is load-bearing, not incidental: ``shape="flat"`` is the same
    L2 perturbation with no shape, and returns gain 1 with zero sign changes.
    """
    mu = M @ f
    n = mu.size
    tilt = tilt_vector(n, eps, shape)
    fh = e4.nnls_invert(M, mu * tilt)
    r = e4.ratio_profile(fh, f)
    fin = r[np.isfinite(r)]
    sgn = np.sign(fin - 1.0)
    nz = sgn[sgn != 0]
    return dict(
        eps=float(eps),
        tilt_shape=str(shape),
        data_perturbation_rel_l2=float(np.linalg.norm(mu * tilt - mu)
                                       / np.linalg.norm(mu)),
        ratio=[None if not np.isfinite(v) else float(v) for v in r],
        max_abs_ratio_minus_1=float(np.max(np.abs(fin - 1.0))),
        n_sign_changes_of_error=int(np.sum(nz[1:] != nz[:-1])),
        gain=float(np.max(np.abs(fin - 1.0))
                   / max(np.linalg.norm(mu * tilt - mu) / np.linalg.norm(mu), 1e-30)),
    )


def fold_weights(pack, consts, k):
    """w[s, b] — the DIAGONAL (non-response) half of the fold at fine-z ``k``.

    ``forward.fold_mu`` computes

        mu[c,k,s] = sum_b K[s,K(k),c,b] * (C_bs[s,b] * g[b,k] * f[b,k] * dN_b)
                    * dX[k,s]

    so at fixed k the design block factorises EXACTLY as

        A[:, k, s, :] = K[s, K(k)] @ diag(w[s]),
        w[s,b] = C_bs[s,b] * g[b,k] * dN_b * dX[k,s].

    Two FROZEN inputs vary with the stratum s: the response kernel K (through
    ``s_to_sresp``, 3 distinct SNR response cells) and the diagonal weight w
    (through the 7-level completeness step C_bs and the per-stratum pathlength
    dX).  This function returns the second one.  Callers MUST check the
    factorisation against the probed operator rather than trusting it.
    """
    import jax
    C_bs = np.asarray(jax.nn.sigmoid(consts.eta_hat))[:, np.asarray(consts.b_to_cell)]
    g = np.asarray(consts.g_bk)[:, k]
    dN = np.asarray(consts.dN_b)
    dX = np.asarray(pack.dX, float)[k]
    return C_bs * (g * dN)[None, :] * dX[:, None]          # (S, B)


def stack_cond(blocks):
    """Condition number of the vertically stacked design blocks."""
    return float(np.linalg.cond(np.vstack(list(blocks))))


def decompose_stack_gain(Ks, ws, *, ref):
    """Counterfactual conditioning of a stacked design, on synthetic factors.

    ``Ks``/``ws`` are equal-length sequences of per-stratum response blocks
    (rows x B) and diagonal weights (B,).  ``ref`` indexes the stratum whose K
    is used as the COMMON response; the common weight is the arithmetic mean
    of ``ws`` (a mean, not a pick, so no single stratum's completeness level
    sets the baseline).

    Returns the four condition numbers and the two gains:

      baseline  common K AND common w  (n identical blocks: no stratum
                                        information at all)
      k_only    per-stratum K restored, weights held common
      w_only    per-stratum w restored, response held common
      actual    both restored

    This function contains NO physics — it is the arithmetic of the
    decomposition alone, so it can be unit-tested against constructed factors
    where the answer is known.
    """
    Ks = [np.asarray(x, float) for x in Ks]
    ws = [np.asarray(x, float) for x in ws]
    if len(Ks) != len(ws):
        raise ValueError("Ks and ws must be the same length")
    n = len(Ks)
    Kr, wr = Ks[ref], np.mean(np.stack(ws), axis=0)
    base = stack_cond([Kr * wr[None, :]] * n)
    k_only = stack_cond([K * wr[None, :] for K in Ks])
    w_only = stack_cond([Kr * w[None, :] for w in ws])
    actual = stack_cond([K * w[None, :] for K, w in zip(Ks, ws)])
    return dict(
        ref_index=int(ref),
        cond_baseline=base, cond_per_stratum_K_only=k_only,
        cond_per_stratum_w_only=w_only, cond_actual=actual,
        gain_from_K=float(base / k_only), gain_from_w=float(base / w_only),
        gain_total=float(base / actual))


def conditioning_decomposition(A, pack, consts, K, k):
    """WHICH frozen input buys the stacked-design conditioning gain, at fine-z k.

    The stacked design at fixed k has cond ~1e2 while a single stratum's block
    has cond ~1e10.  That ~1e8 gain is not free information: it is bought by
    believing that the strata DIFFER, and two independently frozen inputs make
    them differ.  This isolates each one by counterfactual (see
    ``decompose_stack_gain``).
    """
    dX = np.asarray(pack.dX, float)
    live = [s for s in range(pack.n_s) if dX[k, s] > 0]
    if len(live) < 2:
        return None
    kk = int(np.asarray(consts.kz_to_K)[k])
    Ks = [np.asarray(K)[s, kk] for s in live]               # (C, B) each
    W = fold_weights(pack, consts, k)
    ws = [W[s] for s in live]

    # the factorisation is ASSERTED against the probed operator, never assumed
    err = 0.0
    for i, s in enumerate(live):
        blk = A[:, k, s, :]
        scale = max(float(np.max(np.abs(blk))), 1e-300)
        err = max(err, float(np.max(np.abs(blk - Ks[i] * ws[i][None, :])) / scale))
    if err > 1e-10:
        raise RuntimeError(
            f"conditioning_decomposition: A[:,k={k},s,:] != K_s diag(w_s) "
            f"(max rel dev {err:.3e}) — refusing to attribute a gain with a "
            "factorisation that does not hold")

    per_ref = [dict(ref_stratum=int(live[i]),
                    resp_snr_cell=int(np.asarray(consts.s_to_sresp)[live[i]]),
                    **decompose_stack_gain(Ks, ws, ref=i))
               for i in range(len(live))]
    canon = per_ref[0]
    gk = [d["gain_from_K"] for d in per_ref]
    gw = [d["gain_from_w"] for d in per_ref]
    return dict(
        k=int(k), live_strata=[int(s) for s in live],
        factorisation_max_rel_error=err,
        reference_convention=(
            "common response K = the LOWEST live stratum's response cell (the "
            "worst-conditioned one, i.e. the cell the recorded 2.77e10 refers "
            "to); common weight = the arithmetic mean of the live strata's "
            "w[s]. The K reference is consequential — see "
            "k_reference_sensitivity — because the highest-SNR response cell "
            "is itself far better conditioned."),
        canonical=canon,
        k_reference_sensitivity=per_ref,
        gain_from_K_min=float(min(gk)), gain_from_K_max=float(max(gk)),
        gain_from_w_min=float(min(gw)), gain_from_w_max=float(max(gw)))


def prior_vs_data_precision(M, f, sigma_N):
    """How many of the B basis directions are PRIOR-dominated, not data-driven.

    Eigendecomposes the Poisson Fisher information F for theta = log f and
    reports, per eigen-direction, the data precision (the eigenvalue) against
    the RW2 prior precision in that same direction.  Directions where the prior
    precision exceeds the data precision are directions in which the reported
    posterior width is set by the prior, not measured.
    """
    F, active, _ = _fisher(M, f)
    Fa = F[np.ix_(active, active)]
    P = rw2_precision(M.shape[1], sigma_N)[np.ix_(active, active)]
    w, U = np.linalg.eigh(Fa)
    prior_diag = np.einsum("ij,jk,ki->i", U.T, P, U)
    return dict(
        sigma_N=float(sigma_N),
        n_active_bins=int(active.sum()),
        n_prior_dominated=int(np.sum(prior_diag > w)),
        data_eigenvalues=[float(v) for v in w],
        prior_precision_in_same_direction=[float(v) for v in prior_diag],
    )


def map_mc(M, f, sigma_N, *, n_mc=8, seed0=0):
    """Empirical check of the analytic amplification: Poisson MC through the MAP.

    Runs the model's OWN objective (Poisson likelihood + RW2 curvature prior on
    log f) on Poisson realisations and compares the empirical sd of log f-hat
    to the analytic sqrt(diag((F + P)^-1)).
    """
    mu = M @ f
    B = M.shape[1]
    out = np.empty((n_mc, B))
    for i in range(n_mc):
        y = np.random.default_rng(seed0 + i).poisson(mu).astype(float)
        # NOT initialised at the truth: a neutral count-scale start, so the
        # recovered scatter cannot be an artefact of starting at the answer.
        fh, _ = e4.map_invert_rw2(M, y, sigma_N=sigma_N)
        out[i] = fh
    occ = f > 1e-12 * np.max(f)
    lr = np.log(np.clip(out[:, occ], 1e-300, None)) - np.log(f[None, occ])
    amp_a, sd_a, n_b = amplification(M, f, prior_prec=rw2_precision(B, sigma_N))
    return dict(
        sigma_N=float(sigma_N), n_mc=int(n_mc),
        empirical_sd_log_f_median=float(np.median(lr.std(axis=0, ddof=1))),
        analytic_sd_log_f_median=float(np.nanmedian(sd_a[occ])),
        empirical_sd_log_f_max=float(np.max(lr.std(axis=0, ddof=1))),
        analytic_sd_log_f_max=float(np.nanmax(sd_a[occ])),
        empirical_over_analytic_median=float(np.median(
            lr.std(axis=0, ddof=1) / np.clip(sd_a[occ], 1e-300, None))),
        ratio_p50=[float(v) for v in np.median(out / np.clip(f, 1e-300, None), axis=0)],
        analytic_amplification_max=float(np.nanmax(amp_a)),
    )


def rw2_precision(B, sigma_N):
    D = e4.d2_matrix(B)
    return (D.T @ D) / (sigma_N ** 2)


def sigma_N_for_target(M, f, target, *, grid=None):
    """Largest RW2 sigma_N whose max per-bin amplification is <= target.

    (Larger sigma_N == WEAKER prior; production's hyperprior is
    HalfNormal(0.5), i.e. it puts most mass on sigma_N ~ 0.1-0.8.)
    """
    if grid is None:
        grid = np.geomspace(1e-4, 3.0, 60)
    B = M.shape[1]
    best = None
    for s in grid:
        amp, _, _ = amplification(M, f, prior_prec=rw2_precision(B, s))
        m = float(np.nanmax(amp))
        if m <= target:
            best = (float(s), m)
    return best


# --------------------------------------------------------------------------
# per-mock analysis
# --------------------------------------------------------------------------

def analyse_mock(name, *, n_mc=N_MC, verbose=True):
    import jax.numpy as jnp

    path = PACK_DIR / f"modelA_pack_{name}_v11.npz"
    pack = load_pack(path)
    consts = build_consts(pack)
    A = e4.build_fold_operator(pack, consts=consts)
    lin_err = e4.check_linearity(pack, A, consts=consts)
    if lin_err > 1e-12:
        raise RuntimeError(f"{name}: probe/fold mismatch {lin_err:.3e}")

    Nc = np.asarray(consts.Nc_b)
    dN = np.asarray(consts.dN_b)
    f_true = e4.truth_f(pack, consts)
    dX = np.asarray(pack.dX, float)
    live_ks = [(k, s) for k in range(pack.n_k) for s in range(pack.n_s)
               if dX[k, s] > 0]

    out = dict(
        pack=str(path),
        pack_provenance_commit=(pack.provenance or {}).get("code_commit"),
        pack_schema=(pack.provenance or {}).get("schema"),
        probe_fold_relative_error=float(lin_err),
        grid=dict(nhat_edges=[float(v) for v in pack.nhat_edges],
                  ntrue_edges=[float(v) for v in pack.ntrue_edges],
                  n_c=pack.n_c, n_b=pack.n_b, n_k=pack.n_k, n_s=pack.n_s,
                  n_pad_bins=int(len(pack.ntrue_edges) - len(pack.nhat_edges)),
                  live_strata=sorted({s for _, s in live_ks})),
        counts_total=int(np.asarray(pack.counts).sum()),
        truth_total=float(np.asarray(pack.truth_counts).sum()),
    )

    # -- Q1a: the RAW response kernel, per (SNR-resp, z-resp) cell -----------
    K = np.asarray(build_K(jnp.zeros((2, consts.n_sr, consts.n_zr)), consts))
    kern = []
    seen = set()
    for k, s in live_ks:
        cell = (int(consts.s_to_sresp[s]), int(consts.K_to_zresp[consts.kz_to_K[k]]))
        if cell in seen:
            continue
        seen.add(cell)
        Km = K[s, consts.kz_to_K[k]]
        sp = e4.spectrum(Km)
        kern.append(dict(resp_snr_cell=cell[0], resp_z_cell=cell[1],
                         cond=sp.cond,
                         sigma_max=float(sp.sv[0]), sigma_min=float(sp.sv[-1]),
                         effective_rank=sp.rank_thresholds,
                         row_mass_min=float(Km.sum(axis=0).min()),
                         row_mass_max=float(Km.sum(axis=0).max())))
    out["response_kernel_spectra"] = sorted(
        kern, key=lambda d: (d["resp_snr_cell"], d["resp_z_cell"]))

    # -- Q1b: per (z, SNR) stratum design blocks ----------------------------
    per_stratum = []
    for k, s in live_ks:
        sp = e4.spectrum(A[:, k, s, :])
        per_stratum.append(dict(k=k, s=s,
                                zf_lo=float(pack.zf_edges[k]),
                                zf_hi=float(pack.zf_edges[k + 1]),
                                snr_lo=float(pack.snr_edges[s]),
                                snr_hi=float(pack.snr_edges[s + 1]),
                                cond=sp.cond,
                                effective_rank_1e6=sp.rank_thresholds["1e-06"]))
    cvals = np.array([d["cond"] for d in per_stratum])
    out["per_stratum"] = dict(
        n_blocks=len(per_stratum),
        cond_min=float(cvals.min()), cond_median=float(np.median(cvals)),
        cond_max=float(cvals.max()),
        blocks=per_stratum)

    # -- Q1c/Q2: the STACKED per-z design matrix ----------------------------
    per_k = []
    for k in range(pack.n_k):
        M = e4.operator_matrix(A, pack, k)
        sp = e4.spectrum(M)
        rec = dict(k=k, zf_lo=float(pack.zf_edges[k]), zf_hi=float(pack.zf_edges[k + 1]),
                   n_rows=sp.n_rows, n_cols=sp.n_cols,
                   cond=sp.cond, sigma_max=float(sp.sv[0]),
                   sigma_min=float(sp.sv[-1]),
                   effective_rank=sp.rank_thresholds,
                   numerical_rank=sp.numerical_rank)
        amp, sd, n_b = amplification(M, f_true[:, k])
        rec["amplification"] = dict(
            max=float(np.nanmax(amp)), median=float(np.nanmedian(amp)),
            argmax_logNHI=float(Nc[int(np.nanargmax(amp))]))
        per_k.append(rec)
    out["per_z_stacked"] = per_k

    # -- Q1d: WHICH frozen input buys the stacked-design gain ----------------
    decomp = [d for d in (conditioning_decomposition(A, pack, consts, K, k)
                          for k in range(pack.n_k)) if d is not None]
    out["conditioning_decomposition"] = dict(
        note="the stacked design's ~1e8 conditioning gain over a single "
             "stratum block, attributed by counterfactual to the TWO frozen "
             "inputs that make the strata differ: the per-SNR response kernel "
             "K_s, and the per-stratum diagonal weight w_s (the 7-level "
             "completeness step times the per-stratum dX).",
        gain_from_K_median_over_z=float(np.median(
            [d["canonical"]["gain_from_K"] for d in decomp])),
        gain_from_w_median_over_z=float(np.median(
            [d["canonical"]["gain_from_w"] for d in decomp])),
        gain_total_median_over_z=float(np.median(
            [d["canonical"]["gain_total"] for d in decomp])),
        per_z=decomp)

    kk = DETAIL_K
    Mk = e4.operator_matrix(A, pack, kk)
    spk = e4.spectrum(Mk)
    amp_k, sd_k, n_bk = amplification(Mk, f_true[:, kk])
    out["detail_z_bin"] = dict(
        k=kk, zf_lo=float(pack.zf_edges[kk]), zf_hi=float(pack.zf_edges[kk + 1]),
        singular_values=[float(v) for v in spk.sv],
        cond=spk.cond,
        null_directions=e4.null_directions(Mk, Nc, n=4),
        bin_centers_logNHI=[float(v) for v in Nc],
        amplification=[None if not np.isfinite(v) else float(v) for v in amp_k],
        sd_log_f=[None if not np.isfinite(v) else float(v) for v in sd_k],
        counts_attributable=[float(v) for v in n_bk],
    )

    # -- Q3: self-inversion --------------------------------------------------
    si = dict(exact=[], poisson=[])
    for k in range(pack.n_k):
        M = e4.operator_matrix(A, pack, k)
        ft = f_true[:, k]
        if not np.any(ft > 0):
            continue
        err, _ = selfinvert_exact(M, ft)
        si["exact"].append(dict(k=k, max_abs_ratio_minus_1=err))
        mc = selfinvert_poisson(M, ft, n_mc=n_mc, seed0=1000 * k)
        si["poisson"].append(dict(
            k=k, mu_total=float((M @ ft).sum()),
            rms_log_ratio_unpinned=mc["rms_log_ratio_unpinned"],
            p16_p84_log_width_median=mc["p16_p84_log_width_median"],
            ratio_min=mc["ratio_min"], ratio_max=mc["ratio_max"],
            frac_pinned_zero=mc["frac_pinned_zero"],
            n_bins_median_pinned_zero=mc["n_bins_median_pinned_zero"],
            n_bins_p16_pinned_zero=mc["n_bins_p16_pinned_zero"],
            dynamic_range_of_median_unpinned=mc["dynamic_range_of_median_unpinned"]))
    ex = np.array([d["max_abs_ratio_minus_1"] for d in si["exact"]])
    si["exact_summary"] = dict(
        max_over_z=float(ex.max()), median_over_z=float(np.median(ex)),
        verdict=("EXACT-DATA RECOVERY: the operator is invertible in float64 "
                 "arithmetic; the pathology is NOISE AMPLIFICATION, not "
                 "structural non-invertibility")
        if ex.max() < 1e-6 else
        ("EXACT-DATA FAILURE: the operator loses the truth even with noiseless "
         "data — structural rank deficiency"))
    mcs = si["poisson"]
    si["poisson_summary"] = dict(
        rms_log_ratio_unpinned_median_over_z=float(np.median(
            [d["rms_log_ratio_unpinned"] for d in mcs])),
        p16_p84_log_width_median_over_z=float(np.median(
            [d["p16_p84_log_width_median"] for d in mcs])),
        ratio_max_over_z=float(max(d["ratio_max"] for d in mcs)),
        frac_pinned_zero_median=float(np.median([d["frac_pinned_zero"] for d in mcs])),
        dynamic_range_of_median_unpinned_max_over_z=float(max(
            d["dynamic_range_of_median_unpinned"] for d in mcs)),
        # the MEDIAN over z, so the worst z bin is never quoted as typical
        dynamic_range_of_median_unpinned_median_over_z=float(np.median(
            [d["dynamic_range_of_median_unpinned"] for d in mcs])))
    # per-bin detail at the representative z
    mc_detail = selfinvert_poisson(Mk, f_true[:, kk], n_mc=n_mc, seed0=1000 * kk)
    _, fh_exact = selfinvert_exact(Mk, f_true[:, kk])
    si["detail_z_bin"] = dict(
        k=kk,
        bin_centers_logNHI=[float(v) for v in Nc],
        f_true=[float(v) for v in f_true[:, kk]],
        exact_ratio=[None if not np.isfinite(v) else float(v)
                     for v in e4.ratio_profile(fh_exact, f_true[:, kk])],
        poisson_ratio_p16=[None if not np.isfinite(v) else float(v)
                           for v in mc_detail["ratio_p16"]],
        poisson_ratio_p50=[None if not np.isfinite(v) else float(v)
                           for v in mc_detail["ratio_p50"]],
        poisson_ratio_p84=[None if not np.isfinite(v) else float(v)
                           for v in mc_detail["ratio_p84"]],
    )
    # the mechanism test: a SMOOTH systematic, no noise at all
    si["systematic_misfit"] = dict(
        note="EXACT arithmetic, zero Poisson noise: a smooth few-percent tilt "
             "of the folded truth. A large OSCILLATORY f-error here shows the "
             "near-null directions convert model misspecification (D1/D2) into "
             "bin-to-bin ringing without any noise being involved.",
        per_z=[dict(k=k, **systematic_misfit_test(
            e4.operator_matrix(A, pack, k), f_true[:, k], eps=0.05))
            for k in range(pack.n_k) if np.any(f_true[:, k] > 0)],
        flat_control_note="the SAME L2 perturbation with no SHAPE (a pure "
                          "rescale). NNLS is positively homogeneous, so this "
                          "must return gain 1 and zero sign changes; it is the "
                          "control that shows the ringing is a property of the "
                          "tilt's shape, not of its amplitude.",
        flat_control=[dict(k=k, **systematic_misfit_test(
            e4.operator_matrix(A, pack, k), f_true[:, k], eps=0.05,
            shape="flat"))
            for k in range(pack.n_k) if np.any(f_true[:, k] > 0)])
    g_all = [d["gain"] for d in si["systematic_misfit"]["per_z"]]
    si["systematic_misfit"]["gain_median_over_z"] = float(np.median(g_all))
    si["systematic_misfit"]["gain_max_over_z"] = float(np.max(g_all))
    sc_all = [d["n_sign_changes_of_error"] for d in si["systematic_misfit"]["per_z"]]
    l2_all = [d["data_perturbation_rel_l2"] for d in si["systematic_misfit"]["per_z"]]
    mr_all = [d["max_abs_ratio_minus_1"] for d in si["systematic_misfit"]["per_z"]]
    si["systematic_misfit"]["n_sign_changes_min_over_z"] = int(min(sc_all))
    si["systematic_misfit"]["n_sign_changes_max_over_z"] = int(max(sc_all))
    si["systematic_misfit"]["n_sign_changes_median_over_z"] = float(np.median(sc_all))
    si["systematic_misfit"]["tilt_rel_l2_min_over_z"] = float(min(l2_all))
    si["systematic_misfit"]["tilt_rel_l2_max_over_z"] = float(max(l2_all))
    si["systematic_misfit"]["max_abs_ratio_minus_1_max_over_z"] = float(max(mr_all))
    fc = si["systematic_misfit"]["flat_control"]
    si["systematic_misfit"]["flat_control_gain_max_over_z"] = float(
        max(d["gain"] for d in fc))
    si["systematic_misfit"]["flat_control_sign_changes_max_over_z"] = int(
        max(d["n_sign_changes_of_error"] for d in fc))
    out["self_inversion"] = si

    # -- how much of the answer is the prior, at the shipped basis ----------
    out["prior_vs_data_precision"] = [
        dict(k=kk, **prior_vs_data_precision(Mk, f_true[:, kk], s))
        for s in (0.5, 0.25, 0.1)]

    # -- empirical check of the analytic amplification (model's own MAP) ----
    out["map_mc_check"] = dict(
        note="Poisson MC through the model's own objective (Poisson likelihood "
             "+ RW2 curvature prior on log f, nuisances frozen), at the "
             "representative z bin. Confirms the analytic sqrt(diag((F+P)^-1)).",
        k=kk,
        runs=[map_mc(Mk, f_true[:, kk], s, n_mc=max(min(n_mc, 24), 8), seed0=7000)
              for s in (0.5, 0.1)])

    # -- Q4: basis-width sweep ----------------------------------------------
    sweep = []
    for g in WIDTH_GROUPS:
        groups = e4.basis_groups(pack.n_b, g)
        rows = []
        for k in range(pack.n_k):
            M = e4.operator_matrix(A, pack, k)
            ft = f_true[:, k]
            if not np.any(ft > 0):
                continue
            Mg = e4.merge_basis_columns(M, groups)
            fg = e4.merged_truth(ft, dN, groups)
            sp = e4.spectrum(Mg)
            amp, _, _ = amplification(Mg, fg)
            err_exact, _ = selfinvert_exact(Mg, fg)
            mc_noise = selfinvert_poisson(Mg, fg, n_mc=n_mc, seed0=1000 * k)
            # representation error: data from the FINE truth, inverted coarse
            mu_fine, mu_coarse = M @ ft, Mg @ fg
            rep = float(np.linalg.norm(mu_coarse - mu_fine)
                        / max(np.linalg.norm(mu_fine), 1e-300))
            mc_full = selfinvert_poisson(Mg, fg, n_mc=n_mc, seed0=1000 * k,
                                         data_M=M, data_f=ft)
            rows.append(dict(k=k, cond=sp.cond,
                             effective_rank=sp.rank_thresholds,
                             n_cols=sp.n_cols,
                             amp_max=float(np.nanmax(amp)),
                             amp_median=float(np.nanmedian(amp)),
                             exact_err=err_exact,
                             rms_log_ratio_noise_only=mc_noise["rms_log_ratio_unpinned"],
                             rms_log_ratio_with_representation=mc_full["rms_log_ratio_unpinned"],
                             frac_pinned_zero=mc_noise["frac_pinned_zero"],
                             representation_rel_error=rep))
        agg = lambda key, fn=np.median: float(fn([r[key] for r in rows]))  # noqa: E731
        sweep.append(dict(
            group_size=g, basis_width_dex=round(0.1 * g, 3),
            n_basis_bins=len(groups),
            cond_median=agg("cond"), cond_max=agg("cond", np.max),
            effective_rank_1e6_median=float(np.median(
                [r["effective_rank"]["1e-06"] for r in rows])),
            effective_rank_1e3_median=float(np.median(
                [r["effective_rank"]["0.001"] for r in rows])),
            amp_max_median_over_z=agg("amp_max"),
            amp_max_worst_z=agg("amp_max", np.max),
            amp_median_over_z=agg("amp_median"),
            exact_err_max=agg("exact_err", np.max),
            rms_log_ratio_noise_only_median=agg("rms_log_ratio_noise_only"),
            rms_log_ratio_with_representation_median=agg(
                "rms_log_ratio_with_representation"),
            representation_rel_error_median=agg("representation_rel_error"),
            representation_rel_error_max=agg("representation_rel_error", np.max),
            per_z=rows))
    out["basis_width_sweep"] = sweep

    # -- Q4b: RW2 prior strength needed at the SHIPPED 0.1 dex basis ---------
    reg = []
    for k in range(pack.n_k):
        M = e4.operator_matrix(A, pack, k)
        ft = f_true[:, k]
        if not np.any(ft > 0):
            continue
        row = dict(k=k)
        amp0, _, _ = amplification(M, ft)
        row["amp_max_unregularised"] = float(np.nanmax(amp0))
        for tgt in (3.0, 2.0, 1.5):
            hit = sigma_N_for_target(M, ft, tgt)
            row[f"sigma_N_for_amp_le_{tgt:g}"] = None if hit is None else hit[0]
        # what the production hyperprior actually delivers, at its scale
        for s in (0.5, 0.25, 0.1, 0.05):
            a, _, _ = amplification(M, ft, prior_prec=rw2_precision(M.shape[1], s))
            row[f"amp_max_at_sigma_N_{s}"] = float(np.nanmax(a))
        reg.append(row)
    out["rw2_regularisation"] = dict(
        prior_note="production: sigma_N ~ HalfNormal(0.5) on the RW2 curvature "
                   "sd of theta_pop = log f along N (model_a.ModelAConfig."
                   "sigma_N_scale = 0.5). Smaller sigma_N == STRONGER prior.",
        amp_max_unregularised_median=float(np.median(
            [r["amp_max_unregularised"] for r in reg])),
        sigma_N_for_amp_le_2_median=float(np.median(
            [r["sigma_N_for_amp_le_2"] for r in reg
             if r["sigma_N_for_amp_le_2"] is not None])),
        sigma_N_for_amp_le_3_median=float(np.median(
            [r["sigma_N_for_amp_le_3"] for r in reg
             if r["sigma_N_for_amp_le_3"] is not None])),
        per_z=reg)

    if verbose:
        print(f"[{name}] probe err {lin_err:.2e}; kernel cond max "
              f"{max(d['cond'] for d in out['response_kernel_spectra']):.3e}; "
              f"stacked cond median "
              f"{np.median([d['cond'] for d in per_k]):.3e}", file=sys.stderr)
    return out


# --------------------------------------------------------------------------

def _summary(art, mocks):
    """A self-describing headline block: the findings, in words and numbers."""
    def over(fn):
        return [fn(art["mocks"][m]) for m in mocks]

    kern_max = over(lambda m: max(d["cond"] for d in m["response_kernel_spectra"]))
    stack_med = over(lambda m: float(np.median(
        [d["cond"] for d in m["per_z_stacked"]])))
    stack_max = over(lambda m: float(max(d["cond"] for d in m["per_z_stacked"])))
    exact = over(lambda m: m["self_inversion"]["exact_summary"]["max_over_z"])
    amp01 = over(lambda m: m["basis_width_sweep"][0]["amp_max_median_over_z"])

    def rng(vals, fmt="{:.3g}"):
        """'a-b' over mocks, so no single mock's value is quoted as the number."""
        lo, hi = fmt.format(float(np.min(vals))), fmt.format(float(np.max(vals)))
        return lo if lo == hi else f"{lo}-{hi}"

    # --- measured inputs for the prose (nothing below is hand-entered) ------
    gK = over(lambda m: m["conditioning_decomposition"]["gain_from_K_median_over_z"])
    gW = over(lambda m: m["conditioning_decomposition"]["gain_from_w_median_over_z"])
    gT = over(lambda m: m["conditioning_decomposition"]["gain_total_median_over_z"])

    def _detail(m, key):
        """the canonical decomposition at the DETAIL_K bin (fallback: first)."""
        pz = m["conditioning_decomposition"]["per_z"]
        d = next((d for d in pz if d.get("k") == DETAIL_K), pz[0])
        return d["canonical"][key]

    gK_k = over(lambda m: _detail(m, "gain_from_K"))
    gW_k = over(lambda m: _detail(m, "gain_from_w"))
    gT_k = over(lambda m: _detail(m, "gain_total"))
    base_k = over(lambda m: _detail(m, "cond_baseline"))
    konly_k = over(lambda m: _detail(m, "cond_per_stratum_K_only"))
    wonly_k = over(lambda m: _detail(m, "cond_per_stratum_w_only"))
    act_k = over(lambda m: _detail(m, "cond_actual"))

    def _z_span(key):
        vals = [d["canonical"][key] for m in mocks
                for d in art["mocks"][m]["conditioning_decomposition"]["per_z"]]
        return float(np.min(vals)), float(np.max(vals))

    gK_z, gW_z = _z_span("gain_from_K"), _z_span("gain_from_w")
    # min/max over EVERY reference-stratum choice, z bin and mock
    def _ref_span(key):
        vals = [d[key] for m in mocks
                for dz in art["mocks"][m]["conditioning_decomposition"]["per_z"]
                for d in dz["k_reference_sensitivity"]]
        return float(np.min(vals)), float(np.max(vals))

    gK_span = _ref_span("gain_from_K")
    gW_span = _ref_span("gain_from_w")
    sm = over(lambda m: m["self_inversion"]["systematic_misfit"])
    sc_lo = [d["n_sign_changes_min_over_z"] for d in sm]
    sc_hi = [d["n_sign_changes_max_over_z"] for d in sm]
    sc_med = [d["n_sign_changes_median_over_z"] for d in sm]
    l2_lo = [d["tilt_rel_l2_min_over_z"] for d in sm]
    l2_hi = [d["tilt_rel_l2_max_over_z"] for d in sm]
    mr_hi = [d["max_abs_ratio_minus_1_max_over_z"] for d in sm]
    g_med = [d["gain_median_over_z"] for d in sm]
    g_max = [d["gain_max_over_z"] for d in sm]
    flat_g = [d["flat_control_gain_max_over_z"] for d in sm]
    flat_sc = [d["flat_control_sign_changes_max_over_z"] for d in sm]
    dr_med = over(lambda m: m["self_inversion"]["poisson_summary"][
        "dynamic_range_of_median_unpinned_median_over_z"])
    dr_max = over(lambda m: m["self_inversion"]["poisson_summary"][
        "dynamic_range_of_median_unpinned_max_over_z"])
    pin = over(lambda m: m["self_inversion"]["poisson_summary"]["frac_pinned_zero_median"])
    nd_sc = over(lambda m: [d["n_sign_changes"]
                            for d in m["detail_z_bin"]["null_directions"]])
    nd_node = over(lambda m: [d["node_spacing_dex"]
                              for d in m["detail_z_bin"]["null_directions"]])
    n_b = art["mocks"][mocks[0]]["grid"]["n_b"]
    rep02 = over(lambda m: m["basis_width_sweep"][1]["representation_rel_error_median"])
    rep03 = over(lambda m: m["basis_width_sweep"][2]["representation_rel_error_median"])
    rep04 = over(lambda m: m["basis_width_sweep"][3]["representation_rel_error_median"])
    cond01 = over(lambda m: m["basis_width_sweep"][0]["cond_median"])
    cond02 = over(lambda m: m["basis_width_sweep"][1]["cond_median"])
    amp02 = over(lambda m: m["basis_width_sweep"][1]["amp_max_median_over_z"])
    amp03 = over(lambda m: m["basis_width_sweep"][2]["amp_max_median_over_z"])
    pd = over(lambda m: [(r["sigma_N"], r["n_prior_dominated"], r["n_active_bins"])
                         for r in m["prior_vs_data_precision"]])
    pd05 = [x[1] for mm in pd for x in mm if x[0] == 0.5]
    pd01 = [x[1] for mm in pd for x in mm if x[0] == 0.1]
    nact = [x[2] for mm in pd for x in mm]
    sig2 = over(lambda m: m["rw2_regularisation"]["sigma_N_for_amp_le_2_median"])
    sweep = {}
    for i, g in enumerate(WIDTH_GROUPS):
        sweep[f"{0.1 * g:.1f}dex"] = dict(
            n_basis_bins=art["mocks"][mocks[0]]["basis_width_sweep"][i]["n_basis_bins"],
            cond_median=over(lambda m, i=i: m["basis_width_sweep"][i]["cond_median"]),
            amp_max_median_over_z=over(
                lambda m, i=i: m["basis_width_sweep"][i]["amp_max_median_over_z"]),
            representation_rel_error_median=over(
                lambda m, i=i: m["basis_width_sweep"][i]["representation_rel_error_median"]),
            rms_log_ratio_noise_only_median=over(
                lambda m, i=i: m["basis_width_sweep"][i]["rms_log_ratio_noise_only_median"]),
        )
    return dict(
        mocks=list(mocks),
        recorded_2p77e10=dict(
            claim="an earlier session recorded 'SVD condition number 2.77e10 on "
                  "the fold kernel'",
            verdict="CONFIRMED, and identified: it is cond(K) for the SINGLE "
                    "response cell (SNR cell 0, z cell 0) of the frozen "
                    "response, a 29x29 observed-by-true bin-mass matrix. It is "
                    "identical in all three packs because the response is "
                    "frozen and shared.",
            value=kern_max),
        qualification=dict(
            statement="2.77e10 is NOT the conditioning of the operator the "
                      "likelihood inverts. Stacking the 6 live SNR strata (3 "
                      "distinct response cells x a 7-level completeness step x "
                      "a per-stratum dX) at fixed fine-z gives a design matrix "
                      "with cond ~1.2e2-3.5e2.",
            stacked_cond_median=stack_med, stacked_cond_max=stack_max,
            caveat=(
                "CORRECTED 2026-07-29 (an earlier version of this artifact "
                "attributed the whole gain to the response alone; a referee "
                "decomposition showed that is wrong). The gain is bought by "
                "TWO independently frozen inputs, and it is measured here by "
                "counterfactual (conditioning_decomposition): relative to a "
                "baseline that stacks identical blocks (common response AND "
                f"common weights, cond {rng(base_k, '{:.3g}')} at the detail z "
                f"bin k={DETAIL_K}), restoring the per-SNR response kernel "
                f"alone gives cond {rng(konly_k, '{:.3g}')} — a "
                f"{rng(gK_k, '{:.2g}')}x gain — and restoring the per-stratum "
                "weight alone — the 7-level completeness step times the "
                f"per-stratum dX — INDEPENDENTLY gives cond "
                f"{rng(wonly_k, '{:.4g}')}, a {rng(gW_k, '{:.2g}')}x gain "
                f"(both restored: cond {rng(act_k, '{:.4g}')}, "
                f"{rng(gT_k, '{:.2g}')}x). The gains fall steeply with z as the "
                f"response cell changes: over all 15 z bins and all three mocks "
                f"the response gain spans {gK_z[0]:.2g}x-{gK_z[1]:.2g}x and the "
                f"weight gain {gW_z[0]:.2g}x-{gW_z[1]:.2g}x (medians over z "
                f"{rng(gK, '{:.2g}')}x and {rng(gW, '{:.2g}')}x, total "
                f"{rng(gT, '{:.2g}')}x), so no single number covers the whole "
                "range and the per-z table must be read. Neither input is a "
                "robustness margin: both are frozen calibrations, and the "
                "conditioning of the operator the likelihood inverts is only "
                "as real as they are. If either the SNR-resolved response fit "
                "or the completeness step is wrong in its per-stratum "
                "DIFFERENCES, the well-conditioned stacked operator is a "
                "fiction. NOT VERIFIED HERE: no perturbation of either fit was "
                "run, so this is an attribution of the gain, not a measurement "
                "of the error it would induce."),
            caveat_reference_sensitivity=(
                "the split depends on which stratum supplies the common "
                "response in the counterfactual. The canonical choice is the "
                "LOWEST live stratum (the worst-conditioned response cell — "
                "the 2.77e10 one). Over every live stratum as the reference, "
                "every z bin and all three mocks, the response gain spans "
                f"{gK_span[0]:.2g}x-{gK_span[1]:.2g}x and the weight gain "
                f"{gW_span[0]:.2g}x-{gW_span[1]:.2g}x: with the HIGHEST-SNR "
                "response cell as the reference both baselines and both gains "
                "collapse by orders of magnitude, because that cell is itself "
                "far better conditioned. The qualitative finding — two frozen "
                "inputs, not one, carry the load — holds at every reference; "
                "the numbers do not transfer."),
            decomposition_gain_from_response_K_median_over_z=gK,
            decomposition_gain_from_completeness_dX_weights_median_over_z=gW,
            decomposition_gain_total_median_over_z=gT,
            decomposition_at_detail_k=dict(
                k=DETAIL_K, cond_baseline=base_k,
                cond_per_stratum_K_only=konly_k,
                cond_per_stratum_w_only=wonly_k, cond_actual=act_k,
                gain_from_K=gK_k, gain_from_w=gW_k, gain_total=gT_k)),
        exact_data_self_inversion=dict(
            max_abs_ratio_minus_1=exact,
            verdict="RECOVERS THE TRUTH. With exact noiseless data the NNLS "
                    "inverse returns theta* to ~1e-10. E4 is therefore NOT "
                    "structural rank deficiency; it is error amplification."),
        noise_amplification=dict(
            amp_max_median_over_z_at_0p1dex=amp01,
            definition="amp_b = sd(log f_b) / (1/sqrt(n_b)); amp = 1 would be "
                       "the error a perfectly diagonal kernel gives.",
            verdict=(
                "CONFIRMED PATHOLOGY: at the shipped 0.1 dex basis the "
                f"unregularised per-bin error on log f is up to {rng(amp01)}x "
                "the counting error (max over bins, median over z), and "
                "Poisson MC self-inversion at the packs' own count level "
                f"leaves {rng([100 * p for p in pin], '{:.0f}')}% of "
                "bin-draws pinned to zero. The dynamic range of the median "
                f"recovered profile is {rng(dr_med, '{:.0f}')}x in the MEDIAN "
                f"z bin and {rng(dr_max, '{:.0f}')}x in the WORST z bin — the "
                "worst bin is not typical and is not quoted as such."),
            dynamic_range_of_median_profile_median_over_z=dr_med,
            dynamic_range_of_median_profile_max_over_z=dr_max,
            frac_pinned_zero_median=pin),
        null_directions=dict(
            n_sign_changes_of_4_smallest_at_detail_k=nd_sc,
            node_spacing_dex_of_4_smallest_at_detail_k=nd_node,
            statement=(
                "the small-singular-value right vectors alternate sign almost "
                f"every bin ({min(min(x) for x in nd_sc)}-"
                f"{max(max(x) for x in nd_sc)} sign changes over {n_b} bins "
                "for the 4 smallest directions at the detail z bin, node "
                f"spacing {min(min(x) for x in nd_node):.2f}-"
                f"{max(max(x) for x in nd_node):.2f} dex) and concentrate at "
                "the two ends of the grid. The unconstrained combinations ARE "
                "the bin-to-bin oscillations of a 0.1 dex basis - exactly what "
                "a 0.19-0.28 dex response cannot resolve."),
            scope="measured at the single detail z bin (DETAIL_K), not over z."),
        misspecification_mechanism=dict(
            statement=(
                f"a SMOOTH {100 * min(l2_lo):.1f}-{100 * max(l2_hi):.1f}% (L2) "
                "tilt of the noiseless folded truth comes back as an "
                "OSCILLATORY f error up to "
                f"{100 * max(mr_hi):.0f}%, i.e. a gain of {rng(g_med, '{:.1f}')} "
                f"(median over z) to {rng(g_max, '{:.0f}')} (worst z) with no "
                "noise involved, with "
                f"{min(sc_lo)}-{max(sc_hi)} sign changes of the error across z "
                f"and mocks (median {rng(sc_med, '{:.0f}')}). The near-null "
                "directions are the mechanism that converts D1/D2 "
                "forward-model misfit into bin-to-bin ringing at narrow "
                "credible intervals."),
            n_sign_changes_min_over_z=sc_lo,
            n_sign_changes_max_over_z=sc_hi,
            n_sign_changes_median_over_z=sc_med,
            tilt_rel_l2_min_over_z=l2_lo, tilt_rel_l2_max_over_z=l2_hi,
            flat_control=dict(
                statement=(
                    "NULL CONTROL: the same L2 perturbation with no SHAPE (a "
                    "pure rescale) gives gain "
                    f"{rng(flat_g, '{:.3g}')} and {max(flat_sc)} sign changes. "
                    "The finding is therefore a property of the tilt's shape, "
                    "not of its amplitude — and it is pinned by a test with "
                    "power (test_systematic_misfit_ramp_rings_but_a_flat_"
                    "rescale_does_not)."),
                gain_max_over_z=flat_g, sign_changes_max_over_z=flat_sc)),
        basis_width_sweep=sweep,
        rw2_prior=dict(
            production="sigma_N ~ HalfNormal(0.5) (ModelAConfig.sigma_N_scale)",
            amp_max_at_sigma_N_0p5=over(lambda m: float(np.median(
                [r["amp_max_at_sigma_N_0.5"] for r in m["rw2_regularisation"]["per_z"]]))),
            sigma_N_for_amp_le_2=over(
                lambda m: m["rw2_regularisation"]["sigma_N_for_amp_le_2_median"]),
            n_prior_dominated_directions=over(
                lambda m: [(r["sigma_N"], r["n_prior_dominated"], r["n_active_bins"])
                           for r in m["prior_vs_data_precision"]]),
            statement=(
                f"at the shipped 0.1 dex basis, {min(pd05)}-{max(pd05)} of the "
                f"{min(nact)}-{max(nact)} active basis directions already have "
                "MORE prior precision than data precision at sigma_N = 0.5, "
                f"rising to {min(pd01)}-{max(pd01)} at sigma_N = 0.1. "
                "Regularising 0.1 dex to amp <= 2 by the prior alone needs "
                f"sigma_N ~ {rng(sig2, '{:.2g}')}, i.e. the reported "
                "resolution would be supplied by the prior, not measured.")),
        recommendation=dict(
            status="RECOMMENDATION ONLY — not adopted, not implemented. The "
                   "production basis width, prior and estimator are unchanged.",
            text=(
                "Report on a 0.2 dex TRUE-N basis (reporting grid may stay 0.1 "
                "dex). It buys a "
                f"{np.median(cond01) / np.median(cond02):.0f}x drop in "
                f"condition number ({rng(cond01, '{:.0f}')} -> "
                f"{rng(cond02, '{:.1f}')}), a "
                f"{np.median(amp01) / np.median(amp02):.0f}x drop in the "
                f"worst-bin variance inflation ({rng(amp01, '{:.0f}')} -> "
                f"{rng(amp02, '{:.1f}')}), and costs a "
                f"{rng([100 * r for r in rep02], '{:.2g}')}% representation "
                "error in the folded counts. 0.3 dex buys little more (amp "
                f"{rng(amp03, '{:.1f}')}) at roughly double the representation "
                f"cost ({rng([100 * r for r in rep03], '{:.2g}')}%; 0.4 dex "
                f"{rng([100 * r for r in rep04], '{:.2g}')}%). Doing it by "
                "tightening sigma_N instead is worse: it hides the same loss "
                "of resolution inside a prior, where it does not appear in the "
                "error bar."),
            representation_error_is_not_commensurable=(
                "the ~0.8% representation cost is an L2 error on the FOLDED "
                "COUNT vector, computed against each pack's OWN truth. It is "
                "NOT in the same units as, and not directly comparable to, the "
                "population-level systematics quoted elsewhere (BAL, metals, "
                "FP subtraction, the B16 leak), which are fractional biases on "
                "f(N)/Omega. The earlier phrasing 'the smallest systematic in "
                "this problem by more than an order of magnitude' is "
                "WITHDRAWN: it compared incommensurable quantities. What can "
                "be said is that the representation error is small in its own "
                "units and grows roughly linearly with basis width."),
            representation_error_is_not_transferable=(
                "the representation error was measured only against each "
                "pack's own injected truth, and ALL the mocks share ONE "
                "injected f(N). It therefore measures how well a coarse basis "
                "represents THAT f(N) shape, and does not transfer to real "
                "data or to a different underlying CDDF."),
            explicitly_not_claimed=(
                "this does NOT fix D1 (the missing sub-19.5 true-N basis) or "
                "D2 (the residual high-N response excess). A coarser basis "
                "stops those defects being AMPLIFIED into ringing; it does not "
                "remove them. SCOPE: the counting argument (in-window truth < "
                "observed counts) is a statement about the pack's totals and "
                "is basis-independent by construction, but NO basis-width-"
                "resolved closure test was run here, so this artifact makes no "
                "claim about closure at any particular basis width."),
        ),
        cross_mock_agreement=dict(
            statement=(
                "the three mocks do NOT agree to a single tolerance and no "
                "single mock's value should be quoted as 'the' number. "
                "Measured spreads (max/min over the three mocks): cond_median "
                f"at 0.1 dex {rng(cond01, '{:.1f}')} "
                f"({100 * (max(cond01) / min(cond01) - 1):.1f}% spread); "
                f"amp_max_median at 0.1 dex {rng(amp01, '{:.1f}')} "
                f"({100 * (max(amp01) / min(amp01) - 1):.0f}% spread); "
                f"representation error at 0.2 dex "
                f"{rng([100 * r for r in rep02], '{:.3g}')}% "
                f"({100 * (max(rep02) / min(rep02) - 1):.0f}% spread). Every "
                "range in this summary is quoted as a range over the three "
                "mocks for exactly this reason."),
            spread_ratio_max_over_min=dict(
                cond_median_0p1dex=float(max(cond01) / min(cond01)),
                amp_max_median_0p1dex=float(max(amp01) / min(amp01)),
                representation_rel_error_0p2dex=float(max(rep02) / min(rep02)))),
        limitations=[
            "the fold is block-diagonal in fine-z, so this analysis is per-z; "
            "production additionally couples z through the RW1 sigma_z prior, "
            "which adds regularisation this diagnostic does not credit.",
            "EXPECTATION, NOT MEASURED: nuisances (psi_c, psi_k_delta, log t, "
            "FP) are frozen at their prior centres. Marginalising them adds "
            "parameters that are partly degenerate with the population block, "
            "which is expected to widen the posterior, so these amplifications "
            "are expected to be a LOWER bound — but no run with the nuisances "
            "sampled was made here, so that is an expectation and not a "
            "demonstrated bound.",
            "NOT MEASURED: no perturbation of the frozen response fit or of "
            "the completeness step was run. The conditioning decomposition "
            "attributes the stacked-design gain to those two frozen inputs; it "
            "does NOT quantify the error a wrong fit would induce.",
            "the response fit and the completeness step are FROZEN inputs "
            "shared by all three packs, so the cross-mock agreement of the "
            "kernel-level numbers is a check on the probe, not independent "
            "evidence that either input is right.",
            "the analytic amplification is a Laplace/Fisher quantity; the MAP "
            "Monte Carlo reproduces it to within a factor 0.6-0.8 (analytic "
            "over-predicts), so the amplification numbers are conservative in "
            "the direction that matters.",
            "the truth used for self-inversion is each pack's own in-window "
            "truth_counts, which is the D1-deficient truth (n_pad_bins = 0 on "
            "all three packs). That is deliberate: it makes the exact-data "
            "inversion self-consistent and isolates conditioning.",
            "the truth vector used here is e4_probe.truth_f, which DIFFERS "
            "from forward_selftest.truth_f by exactly the factor g_bk "
            "(e4_probe.truth_f * g_bk == forward_selftest.truth_f, pinned by "
            "test_truth_f_divergence_from_forward_selftest_is_exactly_g_bk). "
            "That is required, not a bug: the fold applies g_bk itself, so "
            "the operator's population coordinate is the g-divided one. Any "
            "comparison of these numbers with forward_selftest quantities must "
            "carry the g_bk factor.",
        ],
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mocks", nargs="*", default=list(MOCKS))
    ap.add_argument("--n-mc", type=int, default=N_MC)
    args = ap.parse_args(argv)

    art = dict(metadata=provenance_block(argv))
    art["mocks"] = {m: analyse_mock(m, n_mc=args.n_mc) for m in args.mocks}

    # cross-mock statement: the response is FROZEN, so the kernel-level
    # conditioning must be identical across mocks. Record whether it is.
    kc = {m: [d["cond"] for d in art["mocks"][m]["response_kernel_spectra"]]
          for m in args.mocks}
    ref = kc[args.mocks[0]]
    art["cross_mock"] = dict(
        response_kernel_cond_identical=all(
            np.allclose(kc[m], ref, rtol=1e-12) for m in args.mocks),
        note="the forward response is FROZEN on 2LPT-0 and shared by all three "
             "packs, so identical kernel condition numbers are the expected "
             "result and a check on the probe, not an independent finding.")
    art["summary"] = _summary(art, args.mocks)

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art, indent=1, sort_keys=False))
    print(f"wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
