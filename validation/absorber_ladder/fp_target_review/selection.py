#!/usr/bin/env python
"""selection.py — PURE, unit-testable selection / classification primitives for
the FP CALIBRATION TARGET DEFINITION REVIEW (PI ruling 2026-09-13c §7).

Nothing here reads a file, imports jax, or touches the survey.  Every function
is a deterministic array transform so the event definitions of the two
estimands can be tested on a synthetic catalogue
(``tests/test_fp_target_review.py``).

Two estimands are defined here, and ONLY here:

``loa0_op_mask``
    The loa-0 HCD-FREE TWIN "false positive" event.  In the twin there are no
    absorbers at all, so *every* detection is a false positive by construction;
    the estimand is therefore entirely a SELECTION statement.  The block of
    record (`fp_counts`, 89 events) is
    ``S/N > 2 & P_DLA > 0.99 & lam_rest >= 1025 & 2.0 <= Z_DLA < 3.5``
    binned on ``NHAT_EDGES`` (19.5 .. 22.4) x ``SNR_EDGES`` — verified against
    ``CDDF_analysis/hbi_mcmc/extract_pack.build_fp_block``.  NO proximity
    collar, NO z_qso window, NO BAL veto, NO quality flag (all inert or absent
    on the twin except the collar; see ``collar_window``).

``classify_host_association``
    The mock "hostless" census event (A0v2).  A detection is *hostless* iff the
    greedy one-to-one matcher assigned it no truth host.  This function splits
    that class into the three mechanisms that produce it, which is the whole
    question the memo answers.
"""
from __future__ import annotations

import numpy as np

LYA_REST = 1215.67
C_KMS = 299792.458

#: the fp_counts / census observed-N grid (extract_pack.NHAT_EDGES)
NHAT_EDGES = np.round(np.arange(19.5, 22.4 + 1e-9, 0.1), 3)
#: the molly S/N strata (extract_pack.SNR_EDGES)
SNR_EDGES = np.array([0., 1., 2., 3., 4., 5., 6., 7., np.inf])
#: the fp_counts z window (extract_pack.ZF_EDGES end points)
ZF_LO, ZF_HI = 2.0, 3.5

#: the three reporting N-hat groups of the memo
NHAT_GROUPS = (("19.5-20.0", 19.5, 20.0),
               ("20.0-20.3", 20.0, 20.3),
               (">=20.3", 20.3, np.inf))

#: mutually exclusive, exhaustive labels for a detection's host association
HOST_CLASSES = (
    "hostless_no_absorber",      # no truth absorber of ANY N in the z window
    "hostless_unresolvable",     # the only candidate(s) were dropped pre-match
    "hostless_taken",            # candidate(s) existed but were claimed by
                                 #   another detection (greedy one-to-one)
    "host_17p2_19p0",            # P6b sub-floor host
    "host_19p0_19p5",
    "host_19p5_19p7",
    "host_19p7_21p6",
    "host_ge_21p6",
)

HOST_SLOTS = (("host_17p2_19p0", 17.2, 19.0),
              ("host_19p0_19p5", 19.0, 19.5),
              ("host_19p5_19p7", 19.5, 19.7),
              ("host_19p7_21p6", 19.7, 21.6),
              ("host_ge_21p6", 21.6, np.inf))
SLOT_EPS = 1e-9


# ---------------------------------------------------------------------------
# loa-0 twin selection
# ---------------------------------------------------------------------------
def lam_rest(z_dla, z_qso):
    """Rest-frame forest position of a detection (A)."""
    return LYA_REST * (1.0 + np.asarray(z_dla, float)) / (1.0 + np.asarray(z_qso, float))


def loa0_op_mask(snr, p_dla, z_dla, z_qso, *, snr_min=2.0, p_dla_min=0.99,
                 lam_rf_min=1025.0, z_lo=ZF_LO, z_hi=ZF_HI, nhi=None,
                 nhat_lo=None):
    """The loa-0 FP event selection.

    ``lam_rf_min=None`` drops the Lya-only cut (the legacy full-forest
    product); ``z_lo/z_hi=None`` drops the fine-grid z window; ``nhat_lo=None``
    drops the observed-N floor (the 2,378-event product) — pass 19.5 for the
    89-event ``fp_counts`` block.  Strict inequalities exactly as the committed
    code writes them.
    """
    snr = np.asarray(snr, float)
    m = (snr > snr_min) & (np.asarray(p_dla, float) > p_dla_min)
    if lam_rf_min is not None:
        m &= lam_rest(z_dla, z_qso) >= lam_rf_min
    if z_lo is not None:
        m &= np.asarray(z_dla, float) >= z_lo
    if z_hi is not None:
        m &= np.asarray(z_dla, float) < z_hi
    if nhat_lo is not None:
        if nhi is None:
            raise ValueError("nhat_lo needs nhi")
        m &= np.asarray(nhi, float) >= nhat_lo
    return m


def collar_window(z_qso, *, collar_kms, lam_rf_min=1025.0, lam_rf_max=1216.0):
    """(z_lo, z_hi) of the proximity-collared forest window — the mock packs'
    geometry (``build_scan_packs`` / ``make_lambda_z_BAL_cuts``).  The loa-0
    ``fp_counts`` block does NOT apply it; this is here to measure what it
    would cost (the 89 -> 87 effect, R-015)."""
    zq = np.asarray(z_qso, float)
    coll = float(collar_kms) / C_KMS
    z_lo = np.maximum(3600.0 / LYA_REST - 1.0,
                      lam_rf_min * (1 + zq) / LYA_REST - 1.0 + coll)
    z_hi = np.minimum(zq - coll,
                      lam_rf_max * (1 + zq) / LYA_REST - 1.0 - coll)
    return z_lo, z_hi


# ---------------------------------------------------------------------------
# gridding
# ---------------------------------------------------------------------------
def bin_index(edges, x):
    """extract_pack._idx convention: half-open [lo, hi), right-searchsorted."""
    return np.searchsorted(np.asarray(edges, float),
                           np.asarray(x, float), side="right") - 1


def grid_cs(nhat, snr, *, nhat_edges=NHAT_EDGES, snr_edges=SNR_EDGES):
    """(C, S) integer counts on the fp_counts / census grid.  Rows outside the
    N-hat grid are DROPPED (the grid is the support); S is clipped exactly as
    ``extract_pack.build_fp_block`` clips it."""
    C = len(nhat_edges) - 1
    S = len(snr_edges) - 1
    c = bin_index(nhat_edges, nhat)
    s = np.clip(bin_index(snr_edges, snr), 0, S - 1)
    ok = (c >= 0) & (c < C)
    out = np.zeros((C, S), dtype=np.int64)
    np.add.at(out, (c[ok], s[ok]), 1)
    return out


def group_rows(nhat_edges=NHAT_EDGES, groups=NHAT_GROUPS):
    """{name: boolean mask over the C grid rows} for the reporting N-hat groups."""
    lo = np.asarray(nhat_edges[:-1], float)
    return {name: (lo >= g_lo - SLOT_EPS) & (lo < g_hi - SLOT_EPS)
            for name, g_lo, g_hi in groups}


# ---------------------------------------------------------------------------
# the Perks log-share template (re-implemented; tested against the committed
# CDDF_analysis/hbi_mcmc/fp_ladder.perks_log_share)
# ---------------------------------------------------------------------------
def perks_log_share(fp_counts, live):
    """m_cs = log((n_cs + a0) / (N + K a0)), a0 = 1/K, K = C * n_live_strata.
    Off-live entries are 0 and must never be used."""
    fpc = np.asarray(fp_counts, float)
    live = np.asarray(live, bool)
    C, S = fpc.shape
    K = int(C * live.sum())
    a0 = 1.0 / K
    n_fp = float(fpc[:, live].sum())
    m = np.zeros((C, S), float)
    m[:, live] = np.log((fpc[:, live] + a0) / (n_fp + K * a0))
    return m


def template_mu_cs(fp_counts, live, *, lam_total, fp_w_ell_eff, eta_c):
    """The FP block's (c, s) expectation at t_K = 0, summed over z:

        mu_cs = fp_w*ell_eff * (1 - eta_c) * Lambda * pi_cs,  pi = exp(m) on live

    (``forward.fold_mu_fp`` with ``sum_k fp_E[k,s] = 1`` per live stratum)."""
    m = perks_log_share(fp_counts, live)
    pi = np.where(np.asarray(live, bool)[None, :], np.exp(m), 0.0)
    return (float(fp_w_ell_eff) * float(lam_total)
            * (1.0 - np.asarray(eta_c, float))[:, None] * pi)


# ---------------------------------------------------------------------------
# mock host-association classification
# ---------------------------------------------------------------------------
def host_slot_of(nhi_true):
    """Slot name per row for rows with a finite matched host (else '')."""
    n = np.asarray(nhi_true, float)
    out = np.full(n.shape, "", dtype=object)
    host = np.isfinite(n)
    for name, lo, hi in HOST_SLOTS:
        m = host & (n >= lo - SLOT_EPS)
        if np.isfinite(hi):
            m &= n < hi - SLOT_EPS
        out[m] = name
    return out


def classify_host_association(nhi_true, n_cand_matchpool, n_cand_rawtruth):
    """Assign every detection exactly one of ``HOST_CLASSES``.

    Parameters
    ----------
    nhi_true : (n,)
        The greedy one-to-one matcher's assigned host N_HI (NaN = hostless).
    n_cand_matchpool : (n,) int
        Number of truth rows IN THE MATCHER'S POOL (floored, resolvable) with
        the same TARGETID and |dz|/(1+z_truth) < dz_rel — i.e. the number of
        hosts an ``any-host-within-window`` definition would have accepted.
    n_cand_rawtruth : (n,) int
        The same count against the RAW truth catalogue, before the
        floor/resolvability filters.  ``n_cand_rawtruth >= n_cand_matchpool``.

    The three hostless mechanisms:
      * ``hostless_taken``        — a host was in the window but the greedy
        one-to-one rule gave it to another (closer-in-N) detection: a BLEND /
        matching-definition artefact, NOT a false positive;
      * ``hostless_unresolvable`` — the only host(s) in the window were removed
        from the matcher's pool (no S/N / Z_QSO lookup, or below the truth
        floor): a bookkeeping artefact;
      * ``hostless_no_absorber``  — no truth absorber of ANY N_HI within the
        window: the ONLY class the HCD-free twin can produce.
    """
    n = np.asarray(nhi_true, float)
    cm = np.asarray(n_cand_matchpool, int)
    cr = np.asarray(n_cand_rawtruth, int)
    if cm.shape != n.shape or cr.shape != n.shape:
        raise ValueError("shape mismatch")
    if np.any(cr < cm):
        raise ValueError("n_cand_rawtruth < n_cand_matchpool: pools inverted")
    host = np.isfinite(n)
    if np.any(host & (cm < 1)):
        raise ValueError("a matched detection with no candidate in the pool")
    out = np.empty(n.shape, dtype=object)
    out[~host & (cm >= 1)] = "hostless_taken"
    out[~host & (cm == 0) & (cr >= 1)] = "hostless_unresolvable"
    out[~host & (cm == 0) & (cr == 0)] = "hostless_no_absorber"
    slots = host_slot_of(n)
    out[host] = slots[host]
    if np.any(out == None) or np.any(out == ""):          # noqa: E711
        raise ValueError("classification is not exhaustive")
    return out.astype("<U24")


def class_grids(labels, binner, *arrays):
    """{class: grid} for every label in ``HOST_CLASSES``, plus ``_total``.

    ``binner(*[a[mask] for a in arrays]) -> ndarray`` is injected so the SAME
    committed binner (``extract_pack.bin_counts_cks``) that built the census can
    be used here; the function asserts the class grids PARTITION the total.
    """
    labels = np.asarray(labels)
    arrays = [np.asarray(a) for a in arrays]
    out = {}
    for c in HOST_CLASSES:
        m = labels == c
        out[c] = np.asarray(binner(*[a[m] for a in arrays]))
    total = np.asarray(binner(*arrays))
    got = sum(out.values())
    if not np.array_equal(got, total):
        raise AssertionError("class grids do not partition the total grid")
    out["_total"] = total
    return out
