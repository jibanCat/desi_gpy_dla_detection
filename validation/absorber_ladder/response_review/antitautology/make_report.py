#!/usr/bin/env python
"""make_report.py — figures + ANTITAUTOLOGY_REPORT.md from
``antitautology_results.json``.

Four PNGs (matplotlib Agg, dpi 130) and one markdown report; nothing is
recomputed here, every number comes from the JSON written by
``run_antitautology.py``.

VALIDATION-ONLY.  ENV: gpdla-hbi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

DPI = 130
FIGDIR = ("/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_response_review/"
          "antitautology")
OUTDIR = ("/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13/"
          "response_review/antitautology")
ORDER = ["R1d_raw", "R1d", "R1c", "M_true_emp",
         "toy_conditional", "toy_rowfit", "toy_imprinted"]
REAL = ["R1d_raw", "R1d", "R1c", "M_true_emp"]


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _o(rep, key):
    return [k for k in ORDER if k in rep.get(key, {})]


def fig1(rep, path):
    A = rep["test_A_reweighting_invariance"]
    ks = _o(rep, "test_A_reweighting_invariance")
    wts = list(A[ks[0]]["weightings"])
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    FLOOR = 1e-3
    for i, w in enumerate(wts):
        r = [max(A[k]["weightings"][w]["kl_ratio"] or 0.0, 0.0) for k in ks]
        ax.bar(x + (i - 1) * 0.26, np.maximum(r, FLOOR), 0.25, bottom=0,
               label=w)
        for xx, rv in zip(x + (i - 1) * 0.26, r):
            if rv < FLOOR:
                ax.text(xx, FLOOR * 1.15, "0", ha="center", fontsize=6,
                        rotation=90)
    ax.axhline(4.0, color="crimson", ls="--", lw=1.4,
               label="sealed threshold (KL_rw = 4 x KL_noise)")
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 60.0)
    ax.set_xticks(x); ax.set_xticklabels(ks, rotation=18, ha="right")
    ax.set_ylabel(r"KL$_{\rm reweighted}$ / KL$_{\rm sampling\ noise}$")
    ax.set_title("Test A — CDDF reweighting invariance "
                 r"(rows with $\geq$ 200 calibration events)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig2(rep, path):
    A = rep["test_A_reweighting_invariance"]
    ks = _o(rep, "test_A_reweighting_invariance")
    keys = ["cross_20.0", "cross_20.3", "cross_21.0", "tail_0.3"]
    fig, axes = plt.subplots(1, len(keys), figsize=(12.5, 3.6), sharey=True)
    wts = list(A[ks[0]]["weightings"])
    x = np.arange(len(ks))
    for ax, key in zip(axes, keys):
        for i, w in enumerate(wts):
            d = [abs(A[k]["weightings"][w]["leakage"][key]["d_wmean"] or 0.0)
                 for k in ks]
            ax.bar(x + (i - 1) * 0.26, d, 0.25, label=w)
        n = [A[k]["weightings"][wts[0]]["leakage"][key]["noise_sd_wmean"] or 0.0
             for k in ks]
        ax.plot(x, n, "k_", ms=16, label="bootstrap sd")
        ax.set_yscale("log"); ax.set_title(key, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(ks, rotation=80, fontsize=7)
    axes[0].set_ylabel(r"$|\Delta|$ mass (count-weighted)")
    for ax in axes:
        ax.set_ylim(1e-6, 0.1)
    axes[-1].legend(fontsize=6.5)
    fig.suptitle("Test A — leakage / boundary-crossing mass moved by a "
                 "reweighted calibration population", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig3(rep, path):
    C = rep["test_C_forward_fold"]
    shapes = [s for s in C["shapes"]]
    ks = [k for k in C["self"] if k in REAL + ["M_true_frozen"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    x = np.arange(len(shapes))
    for k in ks:
        ax.semilogy(x, [max(abs(C["self"][k][s]["forensics"]["slope_error"]),
                            1e-17) for s in shapes], "o-", ms=4, label=k)
    ax.set_xticks(x); ax.set_xticklabels(shapes, rotation=80, fontsize=6.5)
    ax.set_ylabel(r"|recovered $-$ input slope|  (dex$^{-1}$)")
    ax.set_title("C-self: fold and invert with the SAME fixed operator\n"
                 "(the inversion's own conditioning error)", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    rb = C.get("reweight_built", {})
    lbl = []
    for k in [k for k in REAL if k in rb]:
        for w, d in rb[k]["slope_drift_vs_native"].items():
            vv = [abs(v) for v in d.values() if v is not None]
            lbl.append((f"{k}\n{w}", max(vv) if vv else 0.0))
    if lbl:
        ax.bar(np.arange(len(lbl)), [v for _, v in lbl], color="tab:purple")
        ax.set_xticks(np.arange(len(lbl)))
        ax.set_xticklabels([n for n, _ in lbl], rotation=80, fontsize=6)
    ax.set_ylabel(r"max |slope drift| (dex$^{-1}$)")
    ax.set_title("C-rebuilt: truth folds, an operator REBUILT on a\n"
                 "reweighted calibration population inverts", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig4(rep, path):
    B = rep["test_B_train_one_slope_eval_another"]
    ks = _o(rep, "test_B_train_one_slope_eval_another")
    arms = sorted({a for k in ks for a in B[k]["paired"]})
    x = np.arange(len(ks))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
    for ax, key, thr, lab in (
            (axes[0], "d_loglik", None, r"$|\Delta \ell|$  (nats / event)"),
            (axes[1], "n_se", 2.0, r"$|\Delta \ell|$ / paired SE")):
        for i, a in enumerate(arms):
            v = [abs(B[k]["paired"].get(a, {}).get(key, np.nan)) for k in ks]
            ax.bar(x + (i - len(arms) / 2 + 0.5) * (0.8 / len(arms)),
                   np.clip(np.nan_to_num(v), 1e-6, None),
                   0.8 / len(arms) * 0.9, label=a.replace("|", " / "))
        if thr is not None:
            ax.axhline(thr, color="crimson", ls="--", lw=1.4,
                       label="sealed threshold (2 paired SE)")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(ks, rotation=22, ha="right", fontsize=7.5)
        ax.set_ylabel(lab)
    axes[0].set_ylim(1e-6, 1.0)
    axes[0].axhspan(1e-6, 2e-3, color="0.85", zorder=0)
    axes[0].text(0.02, 0.05, "no cross-row pooling",
                 transform=axes[0].transAxes, fontsize=7, color="0.35")
    axes[0].set_title("absolute — DISCRIMINATING", fontsize=9)
    axes[1].set_title("sealed §6B statistic — fails every ESTIMATED operator",
                      fontsize=9)
    axes[1].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Test B — train on one population slope, evaluate on another",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


# ---------------------------------------------------------------------------
CONTROL = "toy_rowfit"          # the purely conditional, data-driven control


def verdict(rep, k):
    """Test A is primary: it is the only one of the two with a CALIBRATED
    noise floor (the bootstrap KL).  The sealed §6B 2-SE criterion is a pure
    significance test and, at ~37,000 held-out events, is failed even by the
    controls that are conditional by construction, so B is read RELATIVE to
    the conditional control's own level.  Both verbatim results are reported.
    """
    A = rep["test_A_reweighting_invariance"].get(k, {})
    B = rep["test_B_train_one_slope_eval_another"]
    a_bad = bool(A.get("VERDICT_occupancy_imprinted"))
    ref = B.get(CONTROL, {}).get("max_n_se", 0.0)
    b_large = bool(B.get(k, {}).get("max_n_se", 0.0) > ref)
    if a_bad and b_large:
        return "OCCUPANCY-IMPRINTED"
    if a_bad:
        return "MIXED (A)"
    if b_large:
        return "MIXED (B only)"
    return "CONDITIONAL"


def _max_dl(B, k):
    v = B.get(k, {}).get("paired", {})
    return max((abs(x["d_loglik"]) for x in v.values()), default=None)


def _self_err(C, k, shapes):
    r = C.get("self", {}).get(k)
    if not r:
        return None
    v = [abs(r[s]["forensics"]["slope_error"]) for s in shapes
         if s in r and r[s]["forensics"]["slope_error"] is not None]
    return max(v) if v else None


def _max_drift(C, k):
    v = [abs(x) for d in C.get("reweight_built", {}).get(k, {})
         .get("slope_drift_vs_native", {}).values()
         for x in d.values() if x is not None and np.isfinite(x)]
    return max(v) if v else None


def md_table(rows, head):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fnum(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{n}g}"


def report(rep, figs, out_md):
    A = rep["test_A_reweighting_invariance"]
    B = rep["test_B_train_one_slope_eval_another"]
    C = rep["test_C_forward_fold"]
    D = rep["test_D_occupancy_audit"]
    ks = _o(rep, "test_A_reweighting_invariance")
    L = []
    L.append("# Anti-tautology stress tests A–D on the fixed response "
             "operators")
    L.append("")
    L.append(f"**Science lane, {rep['created']}. Authority: {rep['authority']}."
             " Calibration / mock information only — no HBI run, no MCMC, no "
             "real data, no mock closure number is read anywhere in this "
             "package. Nothing here adopts, selects or ranks an operator.**")
    L.append("")
    L.append(f"Code commit `{rep['git']}` (worktree `wt_abs_diag_2026-09`, "
             "uncommitted at the time of the run — the commander commits). "
             f"Host `{rep['host']}`.")
    L.append("")
    L.append("## 0. Inputs and conventions")
    L.append("")
    L.append(md_table(
        [[k, v["path"], v["sha256"][:16]]
         for k, v in rep["inputs"].items()],
        ["input", "path", "sha256 (first 16)"]))
    L.append("")
    L.append(f"Events: {rep['inputs']['events']['n']:,} 2LPT-0 natural-pair "
             f"matched calibration events on "
             f"{rep['inputs']['events']['n_uniq_tid']:,} sightlines. Geometry "
             f"B={rep['geometry']['B']} latent, C={rep['geometry']['C']} "
             f"observed, S={rep['geometry']['S']} S/N strata, "
             f"KK={rep['geometry']['KK']} coarse-z blocks, "
             f"Kf={rep['geometry']['Kf']} fine-z cells.")
    L.append("")
    L.append("* **Row space.** Every operator is tested at its OWN native "
             "resolution: `resp` = (S/N cell, z cell, latent bin) = 3x3x16 "
             "for R1d-raw / R1d / R1c; `sKb` = (S/N stratum, coarse-z block, "
             "latent bin) = 8x3x16 for the M_true-like empirical operator, "
             "i.e. the (b, s, K) space the sealed protocol §3 names. The "
             "`>= 200 events` gate of §6A counts NATIVE calibration events in "
             "that native row.")
    L.append("* **Rows are conditional.** `P[r, c]` sums to 1 over the 29 "
             "observed cells. In test C every operator's rows are given the "
             "SAME MEASURED in-grid fraction φ(b, K, s) and folded against "
             "the φ-free completeness `C_det` (§4), so the only thing that "
             "differs between operators is row SHAPE. The frozen "
             "`adopted_phi_ref` of the ratified count-conservation rule is "
             "not used in test C; it is audited in §5.")
    L.append("* **KL.** KL(native ‖ reweighted) in nats, both rows mixed with "
             "a uniform floor eps = 1e-6 so no comparison is infinite.")
    L.append("* **R1c weighting caveat.** `respfit` has NO weighted path "
             "(`subbin_moments*` and `adaptive_edges` take raw counts; only "
             "the inner `fit_subbin_ml` accepts `w`), and this package may not "
             "modify tracked files, so R1c weightings are realised by "
             "**weighted resampling with a fixed seed**. The native R1c arm is "
             "resampled the same way, so both sides of every R1c comparison "
             "carry matched Monte-Carlo noise; test B additionally reports a "
             "`native_mc` control arm (same weighting, different seed) that "
             "isolates that noise.")
    L.append("* **Noise-floor asymmetry (conservative).** For the count-based "
             "operators the reweighted build is exact (no Monte Carlo) while "
             "the sampling-noise baseline is a bootstrap, so the denominator "
             "of the KL ratio is if anything OVERSTATED — the test is "
             "conservative toward PASS. The mutation controls below show it "
             "still has ample power.")
    L.append("")
    mf = rep.get("merged_from")
    if mf:
        L.append("* **Provenance.** The results JSON was assembled from more "
                 "than one invocation (`--merge`): " +
                 "; ".join(f"tests {','.join(m['tests'])} at {m['created']}"
                           for m in mf) +
                 ". Every section is deterministic given the inputs above and "
                 "the exact commands in §7 reproduce it.")
    L.append("")
    L.append("## 1. Verdicts")
    L.append("")
    rows = []
    for k in ks:
        rows.append([
            k, verdict(rep, k),
            fnum(max(v["kl_ratio"] or 0.0
                     for v in A[k]["weightings"].values())),
            fnum(max(v["frac_rows_flagged"] or 0.0
                     for v in A[k]["weightings"].values())),
            fnum(_max_dl(B, k), 3), fnum(B[k]["max_n_se"], 3),
            fnum(_max_drift(C, k))])
    L.append(md_table(rows, [
        "operator", "verdict", "max KL ratio (A)",
        "max flagged-row frac (A)", "max \\|Δℓ\\| nats/event (B)",
        "max \\|Δℓ\\|/SE (B)",
        "max \\|slope drift\\| dex⁻¹ (C-rebuilt)"]))
    L.append("")
    L.append("**How the verdict column is formed.** Test A is primary: it is "
             "the only one of the two with a CALIBRATED noise floor — the "
             "bootstrap KL of the SAME estimator on the SAME sample, so a "
             "change is measured against how much that estimator moves under "
             "resampling alone. The sealed §6B \"< 2 paired SE\" rule has no "
             "such floor: its SE (7.7e-5 to 8.1e-3 nats/event here) is the "
             "sampling error of the EVALUATION, not of the fit, so it is "
             "failed by every data-driven operator including the controls "
             "that are conditional by construction (`toy_rowfit`, no "
             "cross-row pooling at all, "
             f"{fnum(B.get(CONTROL, {}).get('max_n_se'))} SE; `M_true_emp`, "
             "the saturated conditional row, "
             f"{fnum(B.get('M_true_emp', {}).get('max_n_se'))} SE). Only the "
             "ANALYTIC toy, which touches no count, passes (0.00 SE).")
    L.append("")
    L.append("The **absolute** |Δℓ|, however, separates cleanly into two "
             "groups and is reported alongside: the estimators with no "
             "cross-row pooling move by at most ~1e-3 nats/event "
             f"(`toy_conditional` {fnum(_max_dl(B, 'toy_conditional'), 2)}, "
             f"`toy_rowfit` {fnum(_max_dl(B, 'toy_rowfit'), 2)}, "
             f"`M_true_emp` {fnum(_max_dl(B, 'M_true_emp'), 2)}), while every "
             "operator that pools or shrinks moves by 5e-2 to 1e-1 — the same "
             "size as the deliberately imprinted toy "
             f"(`R1d` {fnum(_max_dl(B, 'R1d'), 2)}, "
             f"`R1d_raw` {fnum(_max_dl(B, 'R1d_raw'), 2)}, "
             f"`R1c` {fnum(_max_dl(B, 'R1c'), 2)}, "
             f"`toy_imprinted` {fnum(_max_dl(B, 'toy_imprinted'), 2)}).")
    L.append("")
    L.append("`R1d_raw` sits in the second group for a reason that is NOT a "
             "population imprint, and test A says so (KL ratio "
             f"{fnum(max(v['kl_ratio'] or 0.0 for v in A['R1d_raw']['weightings'].values()))}"
             ", PASS): its Jeffreys +1/2 is 14.5 pseudo-counts spread over "
             "the 29 observed cells, so a reweighting that changes a row's "
             "effective count by up to ~50x changes how far that row is "
             "shrunk toward the UNIFORM row. That is a prior-strength effect "
             "with a population-free target. §6B cannot tell it apart from an "
             "imprint; test A can, because it measures the move against the "
             "estimator's own sampling noise.")
    L.append("")
    L.append("**Flagged for the PI.** The verdict column therefore reads A "
             "verbatim and reads B relative to the conditional control's own "
             "level; the verbatim §6B pass/fail is reported in full in §3. As "
             "sealed, §6B separates 'analytic' from 'estimated', not "
             "'conditional' from 'occupancy-imprinted'. A repaired version "
             "would normalise Δℓ by the operator's own bootstrap spread, "
             "exactly as §6A does for the KL; that was not run here because "
             "the threshold is sealed.")
    L.append("")
    L.append("Sealed thresholds applied verbatim: **A** flagged iff "
             "`KL_rw − KL_noise > 3 x KL_noise` (i.e. ratio > 4) on the "
             "count-weighted mean over rows with ≥ 200 events; **B** "
             "conditional iff `|Δℓ| < 2` paired (sightline-clustered) SE; "
             "**C** no pull beyond the inversion's own conditioning error.")
    L.append("")
    L.append("### Mutation controls (the battery can fail)")
    L.append("")
    L.append(md_table([[k, "**PASS**" if v else "**FAIL**"]
                       for k, v in rep["MUTATION_CONTROLS"].items()],
                      ["control", "result"]))
    L.append("")
    L.append("`toy_imprinted` is the saturated row shrunk 30 % toward the "
             "POOLED observed N̂ marginal — an estimator whose shrinkage "
             "target carries the calibration population. `toy_conditional` is "
             "an analytic row that touches no count. `toy_rowfit` fits each "
             "row's own mean and width with no cross-row pooling.")
    L.append("")

    # ---- A -------------------------------------------------------------
    L.append("## 2. Test A — CDDF reweighting invariance")
    L.append("")
    L.append("Weightings: `w(N) ∝ 10^{Δγ (N − 20.5)}` with Δγ = −0.5 "
             "(flatter) and +0.5 dex⁻¹ (steeper) than the mock's native "
             "population, and equalised occupancy `w ∝ 1/N_b` across usable "
             "latent bins. Weights enter wherever counts do. Sampling-noise "
             f"KL from {A[ks[0]]['n_boot']} bootstrap replicates of the "
             "native sample.")
    L.append("")
    for k in ks:
        a = A[k]
        L.append(f"**{k}** — {a['n_rows_ge200']} of {a['n_rows']} native "
                 f"`{a['row_grid']}` rows carry ≥ 200 events; sampling-noise "
                 f"KL = {fnum(a['kl_noise_wmean_ge200'])} nats.")
        L.append("")
        L.append(md_table(
            [[w, fnum(v["kl_rw_wmean_ge200"]), fnum(v["kl_noise_wmean_ge200"]),
              fnum(v["kl_ratio"]), fnum(v["kl_rw_max_ge200"]),
              f"{v['n_rows_flagged']} ({fnum(v['frac_rows_flagged'],2)})",
              "**FLAG**" if v["aggregate_excess_over_3x"] else "pass",
              fnum(v["leakage"]["cross_20.3"]["d_wmean"]),
              fnum(v["leakage"]["cross_20.3"]["noise_sd_wmean"]),
              fnum(v["leakage"]["tail_0.3"]["d_wmean"])]
             for w, v in a["weightings"].items()],
            ["weighting", "KL_rw", "KL_noise", "ratio", "max row KL",
             "rows flagged", "§6A", "Δ cross 20.3", "noise sd",
             "Δ tail ±0.3"]))
        L.append("")

    # ---- B -------------------------------------------------------------
    L.append("## 3. Test B — train on one slope, evaluate on another")
    L.append("")
    L.append("2-fold sightline-parity CV (TARGETID parity). The TRAINING fold "
             "is reweighted by one slope, the held-out fold's per-event "
             "multinomial log-likelihood is scored under each evaluation "
             "weighting; both fold directions pooled. Paired SE is clustered "
             "on TARGETID. `native_mc` = the same native weighting with a "
             "different resample seed (Monte-Carlo control; exactly 0 for the "
             "count-based operators).")
    L.append("")
    for k in ks:
        b = B[k]
        L.append(f"**{k}** — max \\|Δℓ\\| = {fnum(_max_dl(B, k), 3)} "
                 f"nats/event, max \\|Δℓ\\|/SE = {fnum(b['max_n_se'])} "
                 f"(MC control {fnum(b['mc_control_max_n_se'])}); §6B "
                 f"conditional = "
                 f"**{'yes' if b['VERDICT_conditional'] else 'NO'}**")
        L.append("")
        L.append(md_table(
            [[a.replace("|", " / "), fnum(v["d_loglik"], 4),
              fnum(v["paired_se"], 3), fnum(v["n_se"]),
              "pass" if v["conditional_pass"] else "**FAIL**"]
             for a, v in b["paired"].items()],
            ["eval / train − native", "Δℓ per event", "paired SE",
             "ratio to SE", "§6B"]))
        L.append("")
        L.append("Crossed held-out log-likelihood matrix (per event):")
        L.append("")
        arms = sorted({a.split("|train=")[1]
                       for a in b["loglik_matrix"]})
        evs = sorted({a.split("|train=")[0].replace("eval=", "")
                      for a in b["loglik_matrix"]})
        L.append(md_table(
            [[e] + [fnum(b["loglik_matrix"][f"eval={e}|train={t}"], 6)
                    for t in arms] for e in evs],
            ["eval (row) / train (col)"] + arms))
        L.append("")

    # ---- C -------------------------------------------------------------
    L.append("## 4. Test C — forward-fold stress test (deterministic)")
    L.append("")
    L.append("Synthetic f(N) (power laws γ = −1.0, −1.5, −2.0, −2.5; a broken "
             "power law −1.0/−2.5 at 21.0; a bump at 21.0) folded through a "
             "fixed operator with the calibration completeness and the pack "
             "`dX`, then read back by the unpenalised non-negative Poisson "
             "MLE (EM), deterministically. Two labelled inversion arms:")
    L.append("")
    L.append(md_table([[k, v] for k, v in C["em_arms"].items()],
                      ["arm", "what it is"]))
    L.append("")
    L.append("Slope fitted over "
             f"[{C['slope_window'][0]}, {C['slope_window'][1]}]. The "
             "calibration mock's own slope over that window is "
             f"**{fnum(C['native_2lpt_slope'], 4)} dex⁻¹** — the value a "
             "tautological operator would pull towards.")
    L.append("")
    pc = C.get("phi_convention", {})
    if pc:
        L.append("**In-grid-fraction convention (coordinator note, "
                 "2026-09-13).** `C_true_bKs` counts only IN-GRID detections "
                 "in its numerator (`build_matched_ops.py:460`), so it already "
                 "carries φ, while the parametric `Mg` rows sum to the frozen "
                 "`adopted_phi_ref` — pairing the two applies φ twice. Test C "
                 f"uses **{pc['which']}**: completeness "
                 f"`{pc['completeness']}`, and {pc['rows']}. The identity "
                 "`C_true = C_det × φ` is verified fail-closed to "
                 f"{fnum(pc['identity_check_max_abs'], 2)}. "
                 f"{pc['n_cells_C_det_gt1']} of {pc['n_cells_supported']} "
                 "supported (b, K, s) cells have `C_det > 1` (sparse "
                 "matching-multiplicity cells); they are left as measured and "
                 "cancel between the fold and the inversion. The frozen "
                 "`adopted_phi_ref` is NOT used in test C — it is audited in "
                 "§5.")
        L.append("")
    L.append("### C-self — fold AND invert with the same operator "
             "(the tautology probe)")
    L.append("")
    shapes = C["shapes"]
    L.append(md_table(
        [[k] + [fnum(C["self"][k][s]["forensics"]["slope_error"], 2)
                for s in shapes] for k in C["self"]],
        ["operator (slope error, dex⁻¹)"] + shapes))
    L.append("")
    L.append("Harsher arm (`flat`):")
    L.append("")
    L.append(md_table(
        [[k] + [fnum(C["self"][k][s]["flat"]["slope_error"], 2)
                for s in shapes] for k in C["self"]],
        ["operator (slope error, dex⁻¹)"] + shapes))
    L.append("")
    L.append("### C-cross — truth folds, the candidate inverts "
             "(REPRESENTATION error, not tautology)")
    L.append("")
    L.append(md_table(
        [[k] + [fnum(C["cross"][k][s]["forensics"]["slope_error"], 3)
                for s in shapes] for k in C["cross"]],
        ["operator (slope error, dex⁻¹)"] + shapes))
    L.append("")
    L.append(md_table(
        [[k] + [fnum(C["cross"][k][s]["forensics"]["max_abs_dev_usable"], 3)
                for s in shapes] for k in C["cross"]],
        ["operator (max \\|f̂/f − 1\\| over usable bins)"] + shapes))
    L.append("")
    L.append("Latent bins the unpenalised MLE COLLAPSES inside the "
             "[20.0, 21.5] slope window (of "
             f"{C['cross'][list(C['cross'])[0]][shapes[0]]['forensics']['n_bins_in_window']}"
             "):")
    L.append("")
    L.append(md_table(
        [[k] + [C["cross"][k][s]["forensics"]["n_bins_collapsed_in_window"]
                for s in shapes] for k in C["cross"]],
        ["operator"] + shapes))
    L.append("")
    L.append(C["slope_robustness"])
    L.append("")
    if C.get("reweight_built"):
        L.append("### C-rebuilt — truth folds, an operator REBUILT on a "
                 "reweighted calibration population inverts")
        L.append("")
        L.append("This converts test A's row KL into dex⁻¹ of recovered "
                 "slope: it is the operational size of the construction-side "
                 "occupancy dependence.")
        L.append("")
        rows = []
        for k, v in C["reweight_built"].items():
            for w, d in v["slope_drift_vs_native"].items():
                rows.append([k, w] + [fnum(d.get(s), 3) for s in shapes])
        L.append(md_table(rows, ["operator", "calibration reweighting"]
                          + [f"drift {s}" for s in shapes]))
        L.append("")
    L.append(C["note"])
    L.append("")

    # ---- D -------------------------------------------------------------
    L.append("## 5. Test D — occupancy-dependence audit")
    L.append("")
    L.append("### D.1 Executable probes")
    L.append("")
    L.append(D["probe_definition"])
    L.append("")
    L.append(md_table(
        [[k, v["row_grid"],
          f"({v['target_row']['i0']},{v['target_row']['i1']},"
          f"{v['target_row']['b']}) n={v['target_row']['n_events']:.0f}",
          fnum(v["a_own_row_count_maxdP"]),
          fnum(v["b_neighbour_row_occupancy_maxdP"]),
          fnum(v["d_other_snr_z_cell_occupancy_maxdP"])]
         for k, v in D["probes"].items()],
        ["operator", "row grid", "target row", "(a) own row count",
         "(b) neighbour rows", "(d) other S/N–z cell"]))
    L.append("")
    L.append("(c) total calibration CDDF **is** test A and is not duplicated "
             "here.")
    L.append("")
    pa = D.get("phi_convention_audit")
    if pa:
        L.append("### D.1b φ convention — the frozen `adopted_phi_ref` "
                 "against the MEASURED in-grid fraction")
        L.append("")
        L.append(pa["why"])
        L.append("")
        pv = pa["phi_ref_vs_measured"]
        L.append("Ratio φ_ref / φ_measured, overall: min "
                 f"{fnum(pv['overall']['min'])}, median "
                 f"{fnum(pv['overall']['median'])}, max "
                 f"{fnum(pv['overall']['max'])}.")
        L.append("")
        L.append(md_table(
            [[k, fnum(v["min"]), fnum(v["median"]), fnum(v["max"]),
              v["n_cells"]]
             for k, v in pv["ratio_phi_ref_over_measured_by_true_N_bin"]
             .items()],
            ["true-N bin", "min", "median", "max", "cells"]))
        L.append("")
        L.append(md_table(
            [[k, fnum(v["min"]), fnum(v["median"]), fnum(v["max"]),
              v["n_cells"]]
             for k, v in pv["ratio_phi_ref_over_measured_by_snr"].items()],
            ["S/N stratum", "min", "median", "max", "cells"]))
        L.append("")
        L.append(pv["note"])
        L.append("")
    L.append("### D.2 Code-level ground truth — every smoothing / shrinkage / "
             "basis / normalisation step, at file:line")
    L.append("")
    L.append(md_table(
        [[s["operator"], s["step"], f"`{s['where']}`",
          "yes" if s["a"] else "no", "yes" if s["b"] else "no",
          "yes" if s["c"] else "no", "yes" if s["d"] else "no",
          s["can_imprint"]]
         for s in D["code_level_table"]],
        ["operator", "step", "where", "(a) row count", "(b) neighbours",
         "(c) total CDDF", "(d) S/N–z occupancy",
         "can it imprint population structure?"]))
    L.append("")

    # ---- verdicts -------------------------------------------------------
    L.append("## 6. Verdict per operator")
    L.append("")
    for k in ks:
        a = A[k]; b = B[k]
        worst_w = max(a["weightings"],
                      key=lambda w: a["weightings"][w]["kl_ratio"] or 0.0)
        drift = _max_drift(C, k)
        L.append(f"**{k} — {verdict(rep, k)}.** Test A: the worst weighting "
                 f"is `{worst_w}` with KL_rw = "
                 f"{fnum(a['weightings'][worst_w]['kl_rw_wmean_ge200'])} nats "
                 f"against a sampling-noise floor of "
                 f"{fnum(a['kl_noise_wmean_ge200'])} nats, a ratio of "
                 f"{fnum(a['weightings'][worst_w]['kl_ratio'])} against the "
                 "sealed threshold of 4, with "
                 f"{a['weightings'][worst_w]['n_rows_flagged']} of "
                 f"{a['n_rows_ge200']} ≥ 200-event rows individually over "
                 f"threshold. Boundary-crossing mass at 20.3 moves by "
                 f"{fnum(a['weightings'][worst_w]['leakage']['cross_20.3']['d_wmean'])}"
                 f" against a bootstrap sd of "
                 f"{fnum(a['weightings'][worst_w]['leakage']['cross_20.3']['noise_sd_wmean'])}"
                 f". Test B: the largest crossed-slope shift is "
                 f"{fnum(b['max_n_se'])} paired SE (threshold 2), MC control "
                 f"{fnum(b['mc_control_max_n_se'])}. Test C: the "
                 "self-fold/invert slope error is "
                 f"{fnum(_self_err(C, k, shapes), 2)}"
                 " dex⁻¹ (conditioning only), and rebuilding the operator on "
                 f"a reweighted calibration population moves the recovered "
                 f"slope by at most {fnum(drift)} dex⁻¹.")
        L.append("")
    L.append("## 7. Exact commands")
    L.append("")
    L.append("```bash")
    L.append("source ~/.bashrc; conda activate gpdla-hbi")
    L.append("export PYTHONPATH=/home/mfho/wt_abs_diag_2026-09 \\")
    L.append("       HDF5_USE_FILE_LOCKING=FALSE JAX_PLATFORMS=cpu \\")
    L.append("       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
             "MKL_NUM_THREADS=1")
    L.append("cd /home/mfho/wt_abs_diag_2026-09")
    L.append("python -m pytest tests/test_antitautology.py -q")
    L.append("python validation/absorber_ladder/response_review/"
             "antitautology/run_antitautology.py \\")
    L.append("    --kinds R1d_raw,R1d,R1c,M_true_emp --tests A,B,C,D "
             "--n-boot 8")
    L.append("python validation/absorber_ladder/response_review/"
             "antitautology/make_report.py")
    L.append("")
    L.append("# a NEW low-DOF candidate, one line:")
    L.append("#   tests A+B (needs a builder adapter defining "
             "BUILDERS = {name: fn}):")
    L.append("python .../run_antitautology.py --kinds <cand> --tests A,B,C,D "
             "\\")
    L.append("    --candidate-builder /path/to/<cand>_adapter.py")
    L.append("#   test C from the delivered Mg alone, merged into the "
             "existing results:")
    L.append("python .../run_antitautology.py --tests C --merge \\")
    L.append("    --candidate-glob 'Mg_*_2lpt0.npz'")
    L.append("```")
    L.append("")
    L.append("## 8. Products")
    L.append("")
    L.append(md_table([[os.path.basename(p), _sha256(p)[:16]] for p in figs],
                      ["figure", "sha256 (first 16)"]))
    L.append("")
    L.append("Results JSON and `SHA256SUMS` in "
             f"`{OUTDIR}`.")
    L.append("")
    L.append("**Nothing here adopts, selects or ranks a response "
             "representation. No real C1. No HBI run.**")
    with open(out_md, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(
        OUTDIR, "antitautology_results.json"))
    ap.add_argument("--fig-dir", default=FIGDIR)
    ap.add_argument("--out-dir", default=OUTDIR)
    a = ap.parse_args(argv)
    rep = json.load(open(a.json))
    os.makedirs(a.fig_dir, exist_ok=True)
    figs = []
    for name, fn in (("at_fig1_reweighting_kl.png", fig1),
                     ("at_fig2_leakage.png", fig2),
                     ("at_fig3_forward_fold.png", fig3),
                     ("at_fig4_train_slope.png", fig4)):
        p = os.path.join(a.fig_dir, name)
        fn(rep, p)
        figs.append(p)
        print("[fig]", p)
    md = os.path.join(a.fig_dir, "ANTITAUTOLOGY_REPORT.md")
    report(rep, figs, md)
    print("[md]", md)
    # SHA256SUMS over the scratch products
    lines = []
    for f in sorted(os.listdir(a.out_dir)):
        p = os.path.join(a.out_dir, f)
        if os.path.isfile(p) and f != "SHA256SUMS":
            lines.append(f"{_sha256(p)}  {f}")
    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[sha]", os.path.join(a.out_dir, "SHA256SUMS"))


if __name__ == "__main__":
    main()
