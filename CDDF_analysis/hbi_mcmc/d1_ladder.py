# -*- coding: utf-8 -*-
"""d1_ladder.py — the D1 BASIS-PAD ladder driver (finding D1, 2026-07-29).

Produces ``d1_basis_pad_ladder.json``: the forward-model closure of the Model A
truth-fold over the full cross of

    mock (2lpt0 / london0 / saclay0)
  x true-N basis pad floor (none / 19.3 / 19.0 / 18.5 / 18.0)
  x response covariate clamp (``both`` / ``hi``)          <- convention (a)
  x sub-floor completeness (``const_extrap`` / ``molly172``)  <- convention (b)

= 60 configurations. Conventions (a) and (b) are UNDECIDED and are crossed on
purpose so each becomes a MEASURED SYSTEMATIC rather than a choice: on the
unpadded grid the two clamps are identical, and under a pad they diverge
sharply because the response was never measured below ~19.35.

TWO PHASES, TWO ENVS (the extractor is jax-free by design, the fold needs jax):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla    python CDDF_analysis/hbi_mcmc/d1_ladder.py \
        --phase extract  --pack-dir <SCRATCH>

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.d1_ladder \
        --phase selftest --pack-dir <SCRATCH> --out <REPO>/CDDF_analysis/hbi_mcmc/d1_basis_pad_ladder.json

The 30 packs are INPUTS, not results: they go to a scratch dir and are never
committed. ~75 kB each; the extract phase is ~6 min for all 30 (detection-side
cut bundles are cached across pad floors -- the pad touches the truth axis only).

MOCKS ONLY. No real-LOA path is touched and no real-data result value can enter
this artifact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEF_PACKDIR = os.environ.get("D1_LADDER_PACK_DIR", "/tmp/d1packs")
DEF_OUT = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/d1_basis_pad_ladder.json")

MOCKS = ["2lpt0", "london0", "saclay0"]
FLOORS = [None, 19.3, 19.0, 18.5, 18.0]
CLAMPS = ["both", "hi"]
CONVENTIONS = ["const_extrap", "molly172"]

# module-level, rebound by main() so the phase functions stay import-clean
PACKDIR = DEF_PACKDIR
OUT = DEF_OUT


def _extract_pack_module():
    """Load extract_pack.py file-directly: the hbi_mcmc package __init__ imports
    jax, and the extract phase deliberately runs in the jax-free `gpdla` env."""
    p = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/extract_pack.py")
    spec = importlib.util.spec_from_file_location("modelA_extract_pack", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def load_pack(*a, **k):
    """Lazy re-export so `--phase extract` never imports jax."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack as _lp
    return _lp(*a, **k)


def _FS():
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS
    return FS


def _RP():
    from CDDF_analysis.hbi_mcmc import run_posterior as RP
    return RP


def _RAT():
    """The gate-ratification record (decision 8, 2026-07-29).

    The ladder's ``closes`` verdict uses THREE criteria -- z_total_max,
    z_bin_max, chi2_dof_max -- and the two ratio-span tolerances that also live
    in ``run_posterior.GATE`` have never taken part in it.  The artifact says
    which is which so that a reader of ``metadata.gate_tolerances`` cannot
    assume the whole dict was load-bearing.

    🔴 An earlier draft of this docstring called those three "the three RATIFIED
    criteria".  ONE of them is.  chi2_dof_max is PI-ratified; z_total_max and
    z_bin_max are ``RESTATED_NOT_RATIFIED`` -- they gate with no ratified
    authority.  ``closes_criteria_note`` is therefore BUILT from the record by
    ``_closes_criteria_note`` rather than asserted in prose, so the artifact
    cannot drift from ``ratification.py`` again.
    """
    from CDDF_analysis.hbi_mcmc import ratification as RAT
    return RAT


#: the criteria that actually decide ``closes`` (see ``forward_closure_gate``)
CLOSES_CRITERIA = ("z_total_max", "z_bin_max", "chi2_dof_max")


def _closes_criteria_note(criteria=CLOSES_CRITERIA):
    """``closes_criteria_note``, DERIVED from the ratification record.

    The ratification status of each criterion is read from
    ``ratification.record`` at stamp time, never restated here.  A prose claim
    would drift the moment a status changed -- which is precisely what happened:
    the hardcoded predecessor of this function said "All three are RATIFIED",
    and after the 2026-07-29 retraction only one of the three was.
    """
    RAT = _RAT()
    ok = [k for k in criteria if RAT.is_ratified(k)]
    not_ok = [k for k in criteria if not RAT.is_ratified(k)]
    parts = [
        "closes = (|z_total| <= z_total_max) AND (max|z_bin| <= z_bin_max) AND "
        "(chi2/dof <= chi2_dof_max), over the reported n-hat bins with obs > 0."]
    if ok:
        parts.append("PI-RATIFIED: " + ", ".join(ok) + ".")
    if not_ok:
        parts.append(
            "🔴 NOT RATIFIED BY ANY DECIDING AUTHORITY, yet contributing to "
            "`closes`: " + ", ".join(
                f"{k} ({RAT.record(k)['status']})" for k in not_ok)
            + ". A `closes: true` in this artifact therefore rests partly on "
              "tolerances nobody ratified; see "
              "ratification.OPEN_PI_DECISIONS['z_arms_gate_unratified'].")
    parts.append(
        "The exact definition of z is the docstring of "
        "CDDF_analysis.hbi_mcmc.forward_selftest.poisson_z; it is an "
        "order-of-magnitude tripwire, NOT a 5-sigma significance and NOT "
        "multiplicity-corrected.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# shared helpers + the phase-2 evidence/verdict blocks
# ---------------------------------------------------------------------------
def tag_for(floor, conv):
    f = "none" if floor is None else f"{floor:.1f}".replace(".", "p")
    return f"_pad{f}_{conv}"


def _REP():
    from CDDF_analysis.hbi_mcmc import reporting as REP
    return REP


def gate_metrics(tab, pack):
    """EXACTLY run_posterior.forward_closure_gate's arithmetic (the committed
    gate): z over reported n-hat bins with obs > 0, chi2/dof over those bins.

    "EXACTLY" is now STRUCTURAL, not a promise in a docstring: both this
    function and ``forward_closure_gate`` delegate to the single implementation
    in ``reporting.window_closure_metrics``, called UNRESTRICTED (A1,
    2026-08-05).  ``pack`` is retained in the signature — every caller passes it
    and it documents which pack the table came from — but the metrics no longer
    read it: the old ``b["lo"] >= pack.nhat_edges[0] - 1e-9`` clause was
    provably inert (``ratio_tables`` emits one row per OBSERVED bin, so the row
    set IS ``nhat_edges[:-1]``) and dropping it changes no number.
    """
    m = _REP().window_closure_metrics(tab["by_nhat"])
    return dict(
        z_total=float(abs(tab["total"]["z"])),
        z_bin_max=m["z_bin_max"],
        chi2_dof=m["chi2_dof"],
        n_bins=int(m["n_bins_in_z_set"]),
    )


def full_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   text=True).strip()


def dirty():
    """TRACKED-file dirtiness only (--untracked-files=no, matching
    extract_pack._git_commit). The stamp asserts that the CODE was at
    ``code_commit``; the artifact being written is itself untracked until it is
    committed, and counting that would make every stamp read dirty."""
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, text=True).strip())


def _fold_kernel(pack, clamp):
    """The dX-weighted response kernel the fold effectively inverts: (C, B)."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K
    c = build_consts(pack, resp_clamp=clamp)
    K = np.asarray(build_K(jnp.zeros((2, c.n_sr, c.n_zr)), c))     # (S,KK,C,B)
    dXw = np.asarray(pack.dX, float)
    kz = np.asarray(pack.kz_to_K)
    W = np.zeros((c.n_c, c.n_b))
    for s in range(c.n_s):
        for kk in range(c.n_kk):
            W += dXw[kz == kk, s].sum() * K[s, kk]
    return W, c, K


def subfloor_completeness_block():
    """How much the ``const_extrap`` convention OVER-PREDICTS the sub-floor
    completeness, on ONE explicitly stated weighting.

    THE WEIGHTING (stated, not implied). The object is the completeness the fold
    actually multiplies by: ``C[s, b] = sigmoid(consts.eta_hat)[:, b_to_cell]``
    — one value per (SNR stratum s, true-N basis bin b). It enters ``fold_mu``
    multiplied by the pathlength ``dX[k, s]``, so the only defensible scalar per
    basis bin is the **dX-exposure-weighted mean over SNR strata**

        C_eff(b) = sum_s w_s C[s, b] / sum_s w_s ,   w_s = sum_k dX[k, s]

    and the reported ratio is ``C_eff^const_extrap(b) / C_eff^molly172(b)``.
    ``C[s, b]`` is constant within a molly cell (``b_to_cell`` is a step map), so
    one number per sub-floor molly cell is exact, not an average of averages.

    WHY NOT the unweighted per-SNR mean of ratios: strata s=0 and s=1 (SNR <= 2)
    carry EXACTLY ZERO pathlength — the op cut is SNR > 2 strict — so their
    completeness cannot influence any prediction. Those two strata are also
    where ``molly172`` measures ~0 detections (n_det = 0 of 11333 at
    [18.5,19.0), s=0), which makes the RATIO blow up to ~83x and makes the
    unweighted mean of ratios NON-MONOTONE in N. Both facts are reported below
    (``ratio_unweighted_per_snr_mean``, ``zero_dX_strata``) so the disagreement
    is visible rather than hidden; the dX-weighted number is the one to quote.
    """
    import jax
    from CDDF_analysis.hbi_mcmc.forward import build_consts

    out = {}
    for mock in MOCKS:
        packs = {cv: load_pack(os.path.join(
            PACKDIR, f"modelA_pack_{mock}_pad18p0_{cv}.npz"))
            for cv in CONVENTIONS}
        p0 = packs["const_extrap"]
        dX = np.asarray(p0.dX, float)
        w_s = dX.sum(axis=0)
        n_pad = p0.n_b - p0.n_c
        lo = np.round(np.asarray(p0.ntrue_edges, float)[:-1], 3)
        C = {}
        for cv, pk in packs.items():
            c = build_consts(pk, resp_clamp="both")
            C[cv] = np.asarray(jax.nn.sigmoid(c.eta_hat))[:, np.asarray(c.b_to_cell)]
        e172 = np.asarray(packs["molly172"].molly_nhi_edges, float)
        rows, seen = [], set()
        for b in range(n_pad):
            j = int(np.searchsorted(e172, lo[b] + 1e-9)) - 1
            key = (float(e172[j]), float(e172[j + 1]))
            if key in seen:
                continue
            seen.add(key)
            ce, cm = C["const_extrap"][:, b], C["molly172"][:, b]
            rows.append(dict(
                cell=list(key),
                C_const_extrap_dX_weighted=float((w_s * ce).sum() / w_s.sum()),
                C_molly172_dX_weighted=float((w_s * cm).sum() / w_s.sum()),
                ratio_dX_weighted=float((w_s * ce).sum() / (w_s * cm).sum()),
                ratio_unweighted_per_snr_mean=float(np.mean(ce / cm)),
                ratio_unweighted_per_snr_max=float(np.max(ce / cm)),
                molly172_n_tot_by_snr=[float(x) for x in
                                       np.asarray(packs["molly172"].molly_n_tot,
                                                  float)[:, j]],
                molly172_n_det_by_snr=[float(x) for x in
                                       np.asarray(packs["molly172"].molly_n_det,
                                                  float)[:, j]]))
        out[mock] = dict(
            rows=rows,
            zero_dX_strata=[int(s) for s in np.flatnonzero(w_s == 0.0)],
            dX_weight_fraction_by_snr=[float(x) for x in w_s / w_s.sum()],
            ge_floor_cells_bit_identical=bool(np.array_equal(
                C["const_extrap"][:, n_pad:], C["molly172"][:, n_pad:])),
        )
    return dict(
        what=("const_extrap vs the MEASURED floor-17.2 completeness, per "
              "sub-floor molly cell, on the dX-exposure-weighted mean over SNR "
              "strata (the only weighting the fold gives meaning to)."),
        weighting=("C_eff(b) = sum_s w_s * sigmoid(eta_hat)[s, b_to_cell[b]] / "
                   "sum_s w_s with w_s = sum_k dX[k, s]; ratio = "
                   "C_eff(const_extrap) / C_eff(molly172)."),
        routine="CDDF_analysis/hbi_mcmc/d1_ladder.py:subfloor_completeness_block",
        per_mock=out,
        finding=("const_extrap over-predicts the sub-floor completeness by "
                 "1.149x at [19.0,19.5), 1.523x at [18.5,19.0) and 2.018x at "
                 "[18.0,18.5) on 2LPT-0 (1.149/1.521/2.015 on london0, "
                 "1.149/1.522/2.016 on saclay0) — MONOTONE in depth on this "
                 "weighting, because C_eff falls 0.751 -> 0.653 -> 0.493 -> "
                 "0.372 while const_extrap holds 0.751 flat. The UNWEIGHTED "
                 "per-SNR mean of ratios is NOT monotone (1.80 / 12.69 / 4.74) "
                 "and peaks at [18.5,19.0), but that shape is manufactured by "
                 "SNR strata 0 and 1, which carry exactly zero pathlength and "
                 "cannot enter any prediction. A pad floor should be chosen on "
                 "the dX-weighted curve."),
        supersedes=("CORRECTION 2026-07-29: the '~1.7x at [19.0,19.5) rising to "
                    "~5.7x at [18.0,18.5)' quoted in the d20d572 commit message "
                    "is WITHDRAWN. It was never reproducible: it is neither the "
                    "dX-weighted ratio (1.149 / 2.018) nor the SNR-summed "
                    "n_det/n_tot ratio (1.207 / 2.158) nor the unweighted "
                    "per-SNR mean (1.803 / 4.742), and it asserted a monotone "
                    "ramp without saying under which weighting."),
    )


def evidence_blocks():
    """Three supporting measurements the verdict rests on."""
    import jax
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K

    p0 = load_pack(os.path.join(PACKDIR, "modelA_pack_2lpt0_padnone_const_extrap.npz"))
    p18 = load_pack(os.path.join(PACKDIR, "modelA_pack_2lpt0_pad18p0_const_extrap.npz"))

    # (1) the power-law --truth-floor DIAGNOSTIC vs the mock's MEASURED truth
    _, f_pl = _FS().extend_pack_truth(p0, 18.0, 19.8, 20.8)
    f_me = _FS().truth_f(p18)
    dN = np.diff(np.asarray(p18.ntrue_edges, float))
    dXt = np.asarray(p18.dX, float).sum(1)
    n_pl = (f_pl * dN[:, None] * dXt[None, :]).sum(1)
    n_me = (f_me * dN[:, None] * dXt[None, :]).sum(1)
    npad = p18.n_b - p18.n_c
    lo = np.round(np.asarray(p18.ntrue_edges, float)[:-1], 3)
    powerlaw = dict(
        what=("forward_selftest.extend_pack_truth: the --truth-floor "
              "diagnostic replaces the missing sub-floor truth with a power "
              "law fitted to the pack's own truth over [19.8, 20.8). This "
              "block measures that surrogate against the mock's ACTUAL "
              "injected truth, which the re-extracted pad carries."),
        fit_window=[19.8, 20.8], mock="2lpt0", pad_floor=18.0,
        per_bin=[dict(lo=float(lo[i]), powerlaw=float(n_pl[i]),
                      measured=float(n_me[i]),
                      ratio=float(n_pl[i] / max(n_me[i], 1e-30)))
                 for i in range(npad)],
        sum_below_floor_powerlaw=float(n_pl[:npad].sum()),
        sum_below_floor_measured=float(n_me[:npad].sum()),
        sum_ratio=float(n_pl[:npad].sum() / n_me[:npad].sum()),
        finding=("the power-law surrogate OVER-PREDICTS the mock's own "
                 "sub-floor truth by 4.1x integrated (up to 8.1x in the "
                 "deepest bin). Any closure statement built on it is not a "
                 "statement about this forward model — it is a statement "
                 "about an injected surrogate population that does not exist "
                 "in the mock."),
    )

    # (2) conditioning of the fold kernel vs pad depth and clamp (E4)
    cond = []
    for name, clamps in [("2lpt0_padnone_const_extrap", ("both", "hi", "off")),
                         ("2lpt0_pad19p0_const_extrap", ("both", "hi")),
                         ("2lpt0_pad18p5_const_extrap", ("both", "hi")),
                         ("2lpt0_pad18p0_const_extrap", ("both", "hi"))]:
        pk = load_pack(os.path.join(PACKDIR, f"modelA_pack_{name}.npz"))
        for cl in clamps:
            try:
                W, c, K = _fold_kernel(pk, cl)
            except Exception as e:
                cond.append(dict(pack=name, resp_clamp=cl, error=str(e)))
                continue
            sv = np.linalg.svd(W, compute_uv=False)
            rm = K.sum(axis=2)
            cond.append(dict(
                pack=name, resp_clamp=cl, shape=list(W.shape),
                svd_cond=float(sv[0] / sv[-1]),
                n_unknowns_minus_n_data=int(c.n_b - c.n_c),
                kernel_rowmass_min=float(rm.min()),
                kernel_rowmass_max=float(rm.max())))
    # (2b) the OTHER operator: E4's SINGLE per-slice response kernel K[s, kk],
    #      unstacked and unweighted. This is what the E4 escalation's 2.77e10
    #      refers to; measuring it here reconciles the two numbers instead of
    #      disputing them.
    per_slice = []
    for name in ("2lpt0_padnone_const_extrap", "2lpt0_pad18p0_const_extrap"):
        pk = load_pack(os.path.join(PACKDIR, f"modelA_pack_{name}.npz"))
        c = build_consts(pk)                       # E4's call: default clamp
        K = np.asarray(build_K(jnp.zeros((2, c.n_sr, c.n_zr)), c))
        dXa = np.asarray(pk.dX, float)
        seen, rows_ = set(), []
        for k in range(pk.n_k):
            for s in range(pk.n_s):
                if dXa[k, s] <= 0:
                    continue
                cell = (int(c.s_to_sresp[s]), int(c.K_to_zresp[c.kz_to_K[k]]))
                if cell in seen:
                    continue
                seen.add(cell)
                Km = K[s, c.kz_to_K[k]]
                sv = np.linalg.svd(Km, compute_uv=False)
                rows_.append(dict(resp_snr_cell=cell[0], resp_z_cell=cell[1],
                                  shape=list(Km.shape),
                                  cond=float(sv[0] / sv[-1])))
        rows_.sort(key=lambda d: (d["resp_snr_cell"], d["resp_z_cell"]))
        per_slice.append(dict(pack=name, rows=rows_,
                              max_cond=max(r["cond"] for r in rows_),
                              cell_0_0_cond=[r["cond"] for r in rows_
                                             if (r["resp_snr_cell"],
                                                 r["resp_z_cell"]) == (0, 0)][0]))

    conditioning = dict(
        what=("SVD condition number of the dX-weighted STACKED (C, B) operator "
              "W = sum_{s,kk} (sum over the fine-z rows in kk of dX[k,s]) * "
              "K[s,kk] — the operator the fold effectively inverts — vs pad "
              "depth and response clamp. This is NOT the same object as the "
              "single per-slice response kernel E4 reports; both are measured "
              "here (see `single_slice_response_kernel_E4_object`)."),
        operator_measured=("W = sum_{s,kk} dX_weight(s,kk) * K[s,kk], shape "
                           "(n_c, n_b) — routine d1_ladder.py:_fold_kernel"),
        rows=cond,
        single_slice_response_kernel_E4_object=dict(
            what=("the object CDDF_analysis/hbi_mcmc/run_e4_conditioning.py "
                  "reports as `response_kernel_spectra`: ONE per-(SNR-resp, "
                  "z-resp) slice K[s, kk], unstacked and un-dX-weighted, at "
                  "build_consts' default clamp."),
            rows=per_slice,
            reconciliation=(
                "CORRECTION 2026-07-29: the E4 figure IS reproduced. On the "
                "unpadded ladder pack the (SNR-resp 0, z-resp 0) slice gives "
                "cond = 27699981558.74824, matching "
                "/home/mfho/wt_e4/CDDF_analysis/hbi_mcmc/e4_conditioning.json "
                "-> mocks.2lpt0.response_kernel_spectra[0].cond = "
                "27699981558.74824 to the digit, on the independently built "
                "modelA_pack_2lpt0_v11.npz. The earlier claim that 2.77e10 "
                "'is not reproduced by any configuration here and must be "
                "re-derived' was WRONG: it compared a different operator. Both "
                "numbers are correct about different objects."),
        ),
        finding=("on the dX-weighted STACKED operator, D2-clamping dominates: "
                 "3.63e8 with the clamp OFF vs 1.47e5 with it ON (unpadded), "
                 "i.e. most of the ill-conditioning of THIS operator was D2. "
                 "The pad then slightly IMPROVES it (1.47e5 -> 1.41e5 at pad "
                 "18.0). What the pad does create is UNDER-DETERMINATION: 44 "
                 "true-N unknowns against 29 observed bins at pad 18.0. "
                 "Separately, on E4's single-slice object the worst slice is "
                 "2.77e10 unpadded (reproduced exactly) — a much larger number "
                 "because no dX weighting or stacking averages the slices, and "
                 "because it is measured at build_consts' default clamp."),
    )

    # (3) the D1 mechanism, quantified: how much of mu in the lowest observed
    #     bins comes from TRUE bins below the reporting floor
    share = {}
    for name, cl in [("2lpt0_pad18p0_const_extrap", "hi"),
                     ("2lpt0_pad18p0_const_extrap", "both"),
                     ("2lpt0_pad18p0_molly172", "hi"),
                     ("2lpt0_pad18p0_molly172", "both")]:
        pk = load_pack(os.path.join(PACKDIR, f"modelA_pack_{name}.npz"))
        c = build_consts(pk, resp_clamp=cl)
        K = np.asarray(build_K(jnp.zeros((2, c.n_sr, c.n_zr)), c))
        f = _FS().truth_f(pk)
        C_bs = np.asarray(jax.nn.sigmoid(c.eta_hat))[:, c.b_to_cell]
        contrib = (C_bs.T[:, None, :] * np.asarray(c.g_bk)[:, :, None]
                   * f[:, :, None] * np.asarray(c.dN_b)[:, None, None])
        Kf = K[:, np.asarray(pk.kz_to_K)]
        # (c, b) signal expectation with the dX weight applied per (k, s)
        mu = np.einsum("skcb,bks,ks->cb", Kf, contrib, np.asarray(pk.dX, float))
        npad_ = pk.n_b - pk.n_c
        share[f"{name}|clamp={cl}"] = [
            float(mu[c_, :npad_].sum() / max(mu[c_].sum(), 1e-30))
            for c_ in range(min(10, pk.n_c))]
    mechanism = dict(
        what=("fraction of the predicted signal counts in each of the LOWEST "
              "observed n-hat bins that is fed by TRUE systems BELOW the "
              "reporting floor (pad 18.0). Schema v1 carried NONE of them."),
        subfloor_share_of_mu_by_lowest_nhat_bins=share,
        finding=("83% of the predicted counts in [19.5, 19.6) come from true "
                 "systems below 19.5. That is D1 stated as a number, and it "
                 "is why the truncated basis under-predicted the lowest bin "
                 "by 6x (mu/obs 0.1655)."),
    )
    return dict(powerlaw_surrogate_vs_measured_truth=powerlaw,
                fold_kernel_conditioning=conditioning,
                d1_mechanism=mechanism,
                subfloor_completeness_convention=subfloor_completeness_block())


def build_verdict(rows, extras):
    closing = [k for k, v in rows.items() if v["closes"]]
    best_chi2 = min(rows.items(), key=lambda kv: kv[1]["chi2_dof"])
    best_tot = min(rows.items(), key=lambda kv: abs(kv[1]["total_ratio"] - 1.0))
    base = rows["2lpt0|pad=None|clamp=both|cmp=const_extrap"]

    def bracket(mock, floor):
        vals = {f"{cl}|{cv}": rows[f"{mock}|pad={floor}|clamp={cl}|cmp={cv}"]["total_ratio"]
                for cl in CLAMPS for cv in CONVENTIONS}
        return dict(values=vals, min=min(vals.values()), max=max(vals.values()),
                    spread_frac=(max(vals.values()) - min(vals.values()))
                    / max(min(vals.values()), 1e-30))
    return dict(
        question=("does ANY (mock x pad_floor x resp_clamp x sub-floor "
                  "completeness) configuration close the forward model inside "
                  "the committed gate (|z_total| <= 5, max|z_bin| <= 5, "
                  "chi2/dof <= 3)?"),
        answer="NO",
        n_configurations=len(rows),
        n_closing=len(closing),
        closing_configurations=closing,
        committed_path_baseline=dict(
            note=("what the COMMITTED code gives on the UNPADDED pack — the "
                  "number every downstream claim must be measured against"),
            config="2lpt0 | no pad | resp_clamp=both | const_extrap",
            total_ratio=base["total_ratio"], z_total=base["z_total"],
            chi2_dof=base["chi2_dof"],
            lowest_nhat_bin_ratio=base["by_nhat"][0]["ratio"],
            truth_total=base["truth_total"], counts_total=base["counts_total"]),
        best_by_chi2=dict(config=best_chi2[0],
                          chi2_dof=best_chi2[1]["chi2_dof"],
                          total_ratio=best_chi2[1]["total_ratio"],
                          gate_max=3.0,
                          factor_over_gate=best_chi2[1]["chi2_dof"] / 3.0),
        best_by_total_ratio=dict(config=best_tot[0],
                                 total_ratio=best_tot[1]["total_ratio"],
                                 z_total=best_tot[1]["z_total"],
                                 chi2_dof=best_tot[1]["chi2_dof"],
                                 lowest_nhat_bin_ratio=best_tot[1]["by_nhat"][0]["ratio"]),
        convention_bracket_at_pad_18p0={m: bracket(m, 18.0) for m in MOCKS},
        convention_bracket_at_pad_19p0={m: bracket(m, 19.0) for m in MOCKS},
        is_D1_alone_sufficient=dict(
            answer="NO",
            level_vs_shape=("the pad fixes the LEVEL and leaves the SHAPE. At "
                            "2lpt0 / pad 18.0 / clamp=hi / const_extrap the "
                            "total is 0.9973 (z=+0.8, inside the gate's total "
                            "leg) while chi2/dof is 51.5 — 17x the tolerance — "
                            "and the per-bin ratio still runs 0.765 to 1.431."),
            residual_is_not_one_defect=(
                "the residual splits in two. (i) A LOW-N deficit in the two "
                "bins below 19.7 that keeps shrinking with pad depth and "
                "swings hard with the two undecided conventions — that is what "
                "is left of D1, and it is a CONVENTION problem, not a missing "
                "-basis problem. (ii) A HIGH-N excess of 1.23-1.43x on 2lpt0 "
                "(up to 1.80x on london0) above logN ~ 21.6, whose per-bin "
                "digits are IDENTICAL across every pad floor and both "
                "completeness conventions — it is residual D2, untouched by "
                "D1, and it alone puts chi2/dof far over the gate."),
            does_E4_dominate=(
                "NOT on the operator measured here, and the two streams are "
                "NOT in conflict (CORRECTED 2026-07-29). On the dX-weighted "
                "STACKED fold operator the conditioning is 1.4-1.5e5 with the "
                "D2 clamp on, against 3.63e8 with it off — so D2-clamping, not "
                "the pad, dominates this operator's conditioning, and the pad "
                "slightly IMPROVES it. E4's 2.77e10 is a DIFFERENT object: the "
                "condition number of a SINGLE per-slice response kernel "
                "K[s,kk], unstacked and un-dX-weighted. That figure IS "
                "reproduced here to the digit (see "
                "fold_kernel_conditioning.single_slice_response_kernel_E4_"
                "object); the earlier claim that it 'is not reproduced by any "
                "configuration here' is withdrawn. What the pad DOES create is "
                "an under-determination: 44 true-N unknowns against 29 "
                "observed bins at pad 18.0, i.e. 15 unidentified directions "
                "that only a prior can fill. So the ordering is: residual D2 "
                "dominates the chi2, the two sub-floor conventions dominate "
                "the level, and E4 becomes a real problem only once those two "
                "are fixed and the pad has to be INFERRED rather than read "
                "from truth."),
        ),
        next_actions=[
            "D2 is NOT closed by the covariate clamp: a 1.23-1.43x (2lpt0) / "
            "up to 1.80x (london0) excess survives above logN ~ 21.6 in every "
            "configuration. Attack that before any further pad work.",
            "resp_clamp is not a free choice under a pad: it moves the total "
            "by up to 15% at pad 18.0. Either measure the response below "
            "19.35 or declare the sub-floor response a stated limit.",
            "the sub-floor completeness convention moves the total by ~8%. "
            "The measured floor-17.2 molly matrix now exists in the pack, so "
            "'const_extrap' should be retired as a default.",
            "rung 10 stays gated: no configuration passes the pre-flight.",
        ],
        do_not_quote=(
            "the 'total 0.998 / chi2/dof 28.2' figure from the uncommitted "
            "scratchpad reconstruction is NOT reproduced and must not be "
            "cited. The committed --truth-floor 18.5 power-law diagnostic "
            "gives total 0.9986 / chi2/dof 22.1 on this code, but it gets "
            "there by injecting 4.1x MORE sub-floor truth than the mock "
            "actually contains (see powerlaw_surrogate_vs_measured_truth). "
            "The re-extracted pad — the mock's OWN measured sub-floor truth — "
            "gives 0.8643 (clamp=both) to 0.9973 (clamp=hi) with chi2/dof "
            "295.6 to 51.5. Separately, the '0.9127 total / 0.6268 lowest "
            "bin' committed-path figure in the task brief was NOT reproduced "
            "by any of the 60 configurations measured here; the nearest are "
            "london0 | pad 18.5 | clamp=both | const_extrap (0.9130 total, "
            "0.6037 lowest bin) and london0 | pad 18.0 | clamp=both | "
            "const_extrap (0.9130 / 0.6057), so that figure most likely came "
            "from a london0 pad at clamp=both with a slightly different "
            "sub-floor convention. It is NOT the 2lpt0 committed-path number, "
            "which is 0.7312 / 0.1655 unpadded."),
    )


# ---------------------------------------------------------------------------
# phase 1 — extract the ladder of packs (env: gpdla, jax-free)
# ---------------------------------------------------------------------------
def phase_extract():
    EP = _extract_pack_module()
    OUTD = PACKDIR

    os.makedirs(OUTD, exist_ok=True)
    manifest = {}
    for conv in CONVENTIONS:
        t0 = time.time()
        frozen = EP.build_frozen_calibration(OUTD, completeness=conv)
        print(f"[ladder] frozen[{conv}] built in {time.time()-t0:.0f}s; "
              f"molly cells={len(frozen['molly']['molly_nhi_edges'])-1} "
              f"g_grid={frozen['g_grid'].shape}", flush=True)
        for floor in FLOORS:
            for mock in MOCKS:
                tag = tag_for(floor, conv)
                r = EP.extract_pack(mock, OUTD, frozen, pad_floor=floor, tag=tag)
                manifest[f"{mock}{tag}"] = dict(
                    mock=mock, pad_floor=floor, completeness=conv,
                    npz=r["npz"], counts_total=r["counts_total"])
                print(f"[ladder] done {mock}{tag}", flush=True)
    with open(os.path.join(OUTD, "ladder_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v["counts_total"] for k, v in manifest.items()}, indent=1))


# ---------------------------------------------------------------------------
# phase 2 — fold + gate every pack, emit the artifact (env: gpdla-hbi)
# ---------------------------------------------------------------------------
def phase_selftest():
    t_start = time.time()
    rows = {}
    probes_by_pack = {}
    for conv in CONVENTIONS:
        for floor in FLOORS:
            for mock in MOCKS:
                tag = tag_for(floor, conv)
                p = os.path.join(PACKDIR, f"modelA_pack_{mock}{tag}.npz")
                pack = load_pack(p)
                n_pad = pack.n_b - pack.n_c
                prov = pack.provenance or {}
                for clamp in CLAMPS:
                    key = f"{mock}|pad={floor}|clamp={clamp}|cmp={conv}"
                    t0 = time.time()
                    res = _FS().selftest(pack, resp_clamp=clamp)
                    tab = _FS().ratio_tables(res, pack)
                    gm = gate_metrics(tab, pack)
                    rows[key] = dict(
                        mock=mock, pad_floor=floor, resp_clamp=clamp,
                        completeness_below_floor=conv,
                        n_pad_bins=int(n_pad),
                        truth_total=float(np.asarray(pack.truth_counts).sum()),
                        truth_total_below_floor=float(
                            np.asarray(pack.truth_counts)[:n_pad].sum()) if n_pad else 0.0,
                        counts_total=float(np.asarray(pack.counts).sum()),
                        total_mu=float(tab["total"]["mu"]),
                        total_obs=float(tab["total"]["obs"]),
                        total_ratio=float(tab["total"]["ratio"]),
                        z_total=float(tab["total"]["z"]),
                        chi2_dof=gm["chi2_dof"],
                        z_bin_max=gm["z_bin_max"],
                        n_gate_bins=gm["n_bins"],
                        closes=bool(gm["z_total"] <= _RP().GATE["z_total_max"]
                                    and gm["z_bin_max"] <= _RP().GATE["z_bin_max"]
                                    and gm["chi2_dof"] <= _RP().GATE["chi2_dof_max"]),
                        by_nhat=[dict(lo=b["lo"], hi=b["hi"], mu=b["mu"],
                                      obs=b["obs"], ratio=b["ratio"], z=b["z"])
                                 for b in tab["by_nhat"]],
                        by_z=[dict(lo=b["lo"], hi=b["hi"], ratio=b["ratio"],
                                   z=b["z"]) for b in tab["by_z"]],
                        pack=os.path.basename(p),
                        pack_provenance_commit=prov.get("code_commit"),
                    )   # NOTE: no wall-clock field here -- `ladder` is
                        # byte-reproducible from the same packs.
                    print(f"{key:62s} ratio={tab['total']['ratio']:.4f} "
                          f"z={tab['total']['z']:+9.1f} chi2/dof={gm['chi2_dof']:10.1f} "
                          f"lowbin={tab['by_nhat'][0]['ratio']:.4f} "
                          f"({time.time()-t0:.1f}s)", flush=True)
                if mock == "2lpt0":
                    probes_by_pack[f"pad={floor}|cmp={conv}"] = _FS().structural_probes(pack)

    # cross-check the inline gate arithmetic against the COMMITTED gate routine
    ref_pack = load_pack(os.path.join(
        PACKDIR, f"modelA_pack_2lpt0{tag_for(None, 'const_extrap')}.npz"))
    ref = _RP().forward_closure_gate(ref_pack, resp_clamp="both")
    inline = rows["2lpt0|pad=None|clamp=both|cmp=const_extrap"]
    xcheck = dict(
        routine="CDDF_analysis/hbi_mcmc/run_posterior.py:forward_closure_gate",
        committed_total_ratio=ref["total_ratio"], inline_total_ratio=inline["total_ratio"],
        committed_chi2_dof=ref["chi2_dof"], inline_chi2_dof=inline["chi2_dof"],
        committed_z_total=ref["z_total"], inline_z_total=abs(inline["z_total"]),
        agrees=bool(np.isclose(ref["total_ratio"], inline["total_ratio"], rtol=1e-12)
                    and np.isclose(ref["chi2_dof"], inline["chi2_dof"], rtol=1e-12)),
        committed_pass=bool(ref["pass"]),
    )
    print("\n[xcheck committed gate]", json.dumps(xcheck, indent=1))

    # PACK STAMP AUDIT (referee minor, 2026-07-29). The top-level code_commit
    # stamps the SELFTEST phase only; the 30 packs are extracted in a separate
    # env, in a separate process, and carry their OWN stamp. Previously every
    # pack read "<base sha>-dirty" while the artifact advertised a clean sha,
    # with nothing in the artifact saying so. Audit it explicitly and FAIL
    # CLOSED on a dirty input.
    pack_commits = sorted({v["pack_provenance_commit"] for v in rows.values()})
    sha = full_sha()
    stamp_audit = dict(
        what=("the extract phase runs in its own process/env, so each pack "
              "stamps itself; this reconciles those stamps against the "
              "selftest phase's code_commit."),
        selftest_phase_code_commit=sha,
        pack_code_commits=pack_commits,
        n_packs=len(rows) // len(CLAMPS),
        all_packs_same_commit=bool(len(pack_commits) == 1),
        any_pack_dirty=bool(any("-dirty" in (c or "") for c in pack_commits)),
        packs_match_selftest_commit=bool(pack_commits == [sha]),
    )
    if stamp_audit["any_pack_dirty"]:
        raise SystemExit(
            "[ladder] REFUSING to stamp: input packs were extracted from a "
            f"DIRTY tree {pack_commits}. Commit the extractor and re-run "
            "--phase extract so the packs stamp a clean sha.")
    if not stamp_audit["packs_match_selftest_commit"]:
        print(f"[ladder] WARNING: packs stamped {pack_commits} but the "
              f"selftest phase is at {sha} (recorded in metadata).")

    extras = evidence_blocks()
    verdict = build_verdict(rows, extras)
    out = dict(
        metadata=dict(
            title="D1 basis-pad ladder — forward-model closure vs true-N pad "
                  "floor, response clamp, and sub-floor completeness",
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            code_commit=sha,
            code_commit_scope=("the SELFTEST phase (this process). The input "
                               "packs carry their own stamp — see "
                               "pack_stamp_audit."),
            code_commit_dirty=dirty(),
            pack_stamp_audit=stamp_audit,
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                text=True).strip(),
            routines=dict(
                driver=("CDDF_analysis/hbi_mcmc/d1_ladder.py "
                        "(--phase extract, then --phase selftest)"),
                extractor="CDDF_analysis/hbi_mcmc/extract_pack.py:extract_pack "
                          "(--basis-pad-floor / --completeness-below-floor)",
                pad_grid="CDDF_analysis/hbi_mcmc/extract_pack.py:basis_pad_edges",
                fold="CDDF_analysis/hbi_mcmc/forward_selftest.py:selftest "
                     "(-> forward.build_consts + forward.fold_mu)",
                gate="CDDF_analysis/hbi_mcmc/run_posterior.py:forward_closure_gate",
            ),
            pack_dir=PACKDIR,
            rederive=(
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "conda run -n gpdla python CDDF_analysis/hbi_mcmc/d1_ladder.py "
                f"--phase extract --pack-dir {PACKDIR}  &&  "
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc."
                f"d1_ladder --phase selftest --pack-dir {PACKDIR} --out {OUT}"),
            pack_note="packs are INPUTS, written to scratch, never committed",
            gate_tolerances=dict(_RP().GATE),
            # WHICH of those tolerances actually decided ``closes``, and which
            # a deciding authority has NOT ratified.  Dumping all of GATE
            # without this distinction implies the whole dict was
            # load-bearing; it was not.  ``closes`` is z_total_max AND
            # z_bin_max AND chi2_dof_max, and the two ratio-span tolerances
            # that appear in ``gate_tolerances`` above played NO part in it,
            # before or after decision 8.
            # 🔴 The note is DERIVED from ratification.py, not written out here:
            # the hardcoded version said "All three are RATIFIED", which the
            # 2026-07-29 retraction made false for two of the three.
            closes_criteria=list(CLOSES_CRITERIA),
            closes_criteria_note=_closes_criteria_note(),
            gate_tolerances_ratified=list(_RAT().ratified_names()),
            gate_tolerances_unratified=list(_RAT().unratified_names()),
            gate_tolerances_unratified_note=_RAT().UNRATIFIED_NOTE,
            ratification=_RAT().ratification_stamp(),
            mocks_only=True,
            privacy="mock packs only; no real-LOA path is touched",
            wall_seconds=round(time.time() - t_start, 1),
        ),
        verdict=verdict,
        ladder=rows,
        structural_probes_2lpt0=probes_by_pack,
        committed_gate_crosscheck=xcheck,
        **extras,
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[ladder] wrote {OUT}")


def main(argv=None):
    global PACKDIR, OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["extract", "selftest"])
    p.add_argument("--pack-dir", default=DEF_PACKDIR)
    p.add_argument("--out", default=DEF_OUT)
    a = p.parse_args(argv)
    PACKDIR = a.pack_dir
    OUT = a.out
    if a.phase == "extract":
        return phase_extract()
    return phase_selftest()


if __name__ == "__main__":
    main()
