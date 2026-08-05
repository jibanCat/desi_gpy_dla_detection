# -*- coding: utf-8 -*-
"""adopted_config.py — closure + coverage under the ADOPTED configuration.

Produces the ONE stamped artifact ``adopted_config_closure.json`` for PI
decisions 1, 3 and 4:

    ADOPTED = 0.2-dex LATENT true-N basis
            + basis pad floor 19.0 with the molly172 sub-floor completeness
            + primary reporting window 19.7 <= log N_HI <= 21.6

and it answers the one question nobody had computed: does the forward model
close inside the REPORTING WINDOW even though it fails globally?

The measured cross is deliberately larger than the adopted point, because three
of the four axes are SYSTEMATICS that must be measured rather than chosen:

    mock            2lpt0 / london0 / saclay0          (cross-mock spread)
  x basis width     0.1 (shipped default) / 0.2 (adopted)
  x pad floor       none (committed baseline) / 19.0 (adopted)
  x completeness    const_extrap / molly172 (adopted)   <- convention (b)
  x resp clamp      both (adopted) / hi                 <- convention (a)

= 24 packs, 48 folds.  Every closure number is reported BOTH over the full
observed grid and restricted to [19.7, 21.6] with the SAME routine
(``reporting.window_closure_metrics``), so the restriction is a filter and never
a different formula.

TWO PHASES, TWO ENVS (the extractor is jax-free by design, the fold needs jax):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla    python CDDF_analysis/hbi_mcmc/adopted_config.py \
        --phase extract --pack-dir <SCRATCH>

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.adopted_config \
        --phase closure --pack-dir <SCRATCH> --out <REPO>/CDDF_analysis/hbi_mcmc/adopted_config_closure.json

The 24 packs are INPUTS, not results: they go to a scratch dir and are never
committed.

MOCKS ONLY.  No real-LOA path is touched and no real-data value can enter this
artifact.
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

DEF_PACKDIR = os.environ.get("ADOPTED_PACK_DIR", "/tmp/adopted_packs")
DEF_OUT = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/adopted_config_closure.json")

MOCKS = ["2lpt0", "london0", "saclay0"]
WIDTHS = [0.1, 0.2]
FLOORS = [None, 19.0]
CONVENTIONS = ["const_extrap", "molly172"]
CLAMPS = ["both", "hi"]

# the ADOPTED point in that cross
ADOPTED_WIDTH = 0.2
ADOPTED_FLOOR = 19.0
ADOPTED_CONV = "molly172"
ADOPTED_CLAMP = "both"
NONIDENT_EDGE_LOCAL = 19.7    # == reporting.NONIDENT_EDGE
CEILING_LOCAL = 21.6          # == reporting.RESPONSE_ANCHOR_CEILING

PACKDIR = DEF_PACKDIR
OUT = DEF_OUT
N_SBC_SIMS = 64      # matched-configuration SBC replicas (0 = skip)
SBC_SEED = 0


def tag_for(width, floor, conv):
    w = f"{width:.1f}".replace(".", "p")
    f = "none" if floor is None else f"{floor:.1f}".replace(".", "p")
    return f"_bw{w}_pad{f}_{conv}"


def key_for(mock, width, floor, conv, clamp):
    return f"{mock}|bw={width}|pad={floor}|cmp={conv}|clamp={clamp}"


def _extract_pack_module():
    """Load extract_pack.py file-directly: the hbi_mcmc package __init__ imports
    jax, and the extract phase deliberately runs in the jax-free `gpdla` env."""
    p = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/extract_pack.py")
    spec = importlib.util.spec_from_file_location("adopted_extract_pack", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def load_pack(*a, **k):
    from CDDF_analysis.hbi_mcmc.pack import load_pack as _lp
    return _lp(*a, **k)


def _FS():
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS
    return FS


def _RP():
    from CDDF_analysis.hbi_mcmc import run_posterior as RP
    return RP


def _REP():
    from CDDF_analysis.hbi_mcmc import reporting as REP
    return REP


def full_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   text=True).strip()


def dirty():
    """TRACKED-file dirtiness only (matching extract_pack._git_commit)."""
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, text=True).strip())


# ---------------------------------------------------------------------------
# pack-stamp reconciliation (fail-closed)
# ---------------------------------------------------------------------------
# Files whose content can change what a PACK contains.  If any of these moved
# between the commit the packs were extracted at and the commit the closure phase
# runs at, the packs are STALE and the artifact must not be stamped -- even
# though nothing looks wrong.  Everything else (this driver, the tests, notes)
# cannot change a pack, so a mismatch confined to those files is provably benign
# and is RECORDED rather than fatal.  A bare "WARNING: packs stamped X but the
# closure phase is at Y" printed to a log is not a guard; this is.
PACK_DETERMINING_FILES = (
    "CDDF_analysis/hbi_mcmc/extract_pack.py",
    "CDDF_analysis/hbi_mcmc/pack.py",
    "CDDF_analysis/hbi_mcmc/reporting.py",
    "CDDF_analysis/hbi/cddf_catalog_hbi.py",
    "CDDF_analysis/hbi/ff_fp_estimator.py",
    "CDDF_analysis/hbi/build_loa0_fp_product.py",
    "CDDF_analysis/hbi/znz_kernel.py",
    "CDDF_analysis/hbi/track_c_tf_loa.py",
    "CDDF_analysis/hbi/track_c_tf_saclay.py",
    "CDDF_analysis/hbi/track_c_tf_london0.py",
    "CDDF_analysis/hbi/ab_loa0_fp_baseline.py",
)


def pack_stamp_verdict(pack_commits, closure_sha, changed_files):
    """Is a pack-stamp / closure-stamp mismatch benign?  Pure function, tested.

    ``changed_files`` is the list of repo-relative paths that differ between the
    pack commit and the closure commit (empty when they are the same commit).

    Returns a dict with ``ok`` (may this artifact be stamped?) and the evidence.
    A DIRTY pack is never ok.  A mismatch is ok ONLY if no
    ``PACK_DETERMINING_FILES`` entry changed.
    """
    pack_commits = list(pack_commits)
    changed = sorted(set(changed_files or ()))
    dirty_pack = any("-dirty" in (c or "") for c in pack_commits)
    match = pack_commits == [closure_sha]
    touched = [f for f in changed if f in PACK_DETERMINING_FILES]
    if dirty_pack:
        ok, why = False, ("input packs were extracted from a DIRTY tree "
                          f"{pack_commits}; commit the extractor and re-run "
                          "--phase extract")
    elif match:
        ok, why = True, "packs and closure phase are at the same commit"
    elif touched:
        ok, why = False, (
            "packs are STALE: they were extracted at "
            f"{pack_commits} but {len(touched)} pack-determining file(s) changed "
            f"before the closure commit {closure_sha}: {touched}. Re-run "
            "--phase extract.")
    else:
        ok, why = True, (
            "packs were extracted at a DIFFERENT commit "
            f"({pack_commits}) than the closure phase ({closure_sha}), but the "
            f"{len(changed)} file(s) that changed in between cannot change a "
            "pack (none is in PACK_DETERMINING_FILES), so the packs are current "
            "in content. Recorded, not waved away.")
    return dict(
        ok=bool(ok), reason=why,
        packs_match_closure_commit=bool(match),
        any_pack_dirty=bool(dirty_pack),
        files_changed_between_pack_and_closure_commit=changed,
        pack_determining_files_changed=touched,
        pack_determining_files=list(PACK_DETERMINING_FILES),
    )


def _files_changed_between(a, b):
    """Repo-relative paths differing between two commits ([] if a == b)."""
    if a == b:
        return []
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", a, b], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None            # unknown -> pack_stamp_verdict cannot clear it
    return [l for l in out.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# phase 1 — extract (env: gpdla, jax-free)
# ---------------------------------------------------------------------------
def phase_extract():
    EP = _extract_pack_module()
    os.makedirs(PACKDIR, exist_ok=True)
    manifest = {}
    for conv in CONVENTIONS:
        t0 = time.time()
        frozen = EP.build_frozen_calibration(PACKDIR, completeness=conv)
        print(f"[adopted] frozen[{conv}] built in {time.time()-t0:.0f}s",
              flush=True)
        for width in WIDTHS:
            for floor in FLOORS:
                for mock in MOCKS:
                    tag = tag_for(width, floor, conv)
                    r = EP.extract_pack(mock, PACKDIR, frozen, pad_floor=floor,
                                        tag=tag, basis_width=width)
                    manifest[f"{mock}{tag}"] = dict(
                        mock=mock, basis_width=width, pad_floor=floor,
                        completeness=conv, npz=r["npz"],
                        counts_total=r["counts_total"])
                    print(f"[adopted] done {mock}{tag}", flush=True)
    with open(os.path.join(PACKDIR, "adopted_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v["counts_total"] for k, v in manifest.items()},
                     indent=1))


# ---------------------------------------------------------------------------
# phase 2 — fold every pack, restrict to the window, emit the artifact
# ---------------------------------------------------------------------------
def _grid_block(pack):
    REP = _REP()
    ne = np.asarray(pack.ntrue_edges, float)
    w_int = REP.window_overlap_weights(ne)
    m_diff = REP.bins_fully_inside(ne)
    return dict(
        n_basis_bins=int(pack.n_b),
        n_observed_bins=int(pack.n_c),
        n_pad_bins=int(pack.n_pad_bins),
        basis_width_nominal_dex=float(pack.basis_width),
        basis_is_uniform=bool(pack.basis_is_uniform),
        ntrue_edges=[float(x) for x in ne],
        bin_widths_dex=[float(x) for x in np.round(np.diff(ne), 8)],
        reporting_window_logN=list(REP.REPORTING_WINDOW),
        n_basis_bins_fully_inside_window=int(m_diff.sum()),
        basis_bins_fully_inside_window=[
            [float(ne[i]), float(ne[i + 1])] for i in np.flatnonzero(m_diff)],
        integrated_overlap_weights_dex=[float(x) for x in w_int],
        n_basis_bins_straddling_a_window_edge=int(
            np.sum((w_int > 0) & (~m_diff))),
        basis_bins_straddling_a_window_edge=[
            [float(ne[i]), float(ne[i + 1]), float(w_int[i])]
            for i in np.flatnonzero((w_int > 0) & (~m_diff))],
        pad_bins_are_latent_nuisance=[
            [float(ne[i]), float(ne[i + 1])] for i in range(pack.n_b)
            if ne[i] < REP.NONIDENT_EDGE - 1e-9],
        subwindow_guard_on_integrated_weights=REP.assert_no_subwindow_bins(
            ne, w_int, where="adopted_config integrated reporting weights"),
    )


def phase_closure():
    t_start = time.time()
    REP = _REP()
    FS = _FS()
    rows = {}
    grids = {}
    for width in WIDTHS:
        for floor in FLOORS:
            for conv in CONVENTIONS:
                for mock in MOCKS:
                    tag = tag_for(width, floor, conv)
                    p = os.path.join(PACKDIR, f"modelA_pack_{mock}{tag}.npz")
                    pack = load_pack(p)
                    gkey = f"bw={width}|pad={floor}"
                    if gkey not in grids:
                        grids[gkey] = _grid_block(pack)
                    prov = pack.provenance or {}
                    for clamp in CLAMPS:
                        t0 = time.time()
                        res = FS.selftest(pack, resp_clamp=clamp)
                        tab = FS.ratio_tables(res, pack)
                        full = REP.window_closure_metrics(
                            tab["by_nhat"], label="full_observed_grid")
                        win = REP.window_closure_metrics(
                            tab["by_nhat"], *REP.REPORTING_WINDOW,
                            label="reporting_window_19p7_21p6")
                        gate = dict(_RP().GATE)
                        rows[key_for(mock, width, floor, conv, clamp)] = dict(
                            mock=mock, basis_width=width, pad_floor=floor,
                            completeness_below_floor=conv, resp_clamp=clamp,
                            n_basis_bins=int(pack.n_b),
                            n_pad_bins=int(pack.n_pad_bins),
                            truth_total=float(np.asarray(pack.truth_counts).sum()),
                            truth_total_below_reporting_floor=float(
                                np.asarray(pack.truth_counts)[
                                    :pack.n_pad_bins].sum())
                            if pack.n_pad_bins else 0.0,
                            counts_total=float(np.asarray(pack.counts).sum()),
                            full_grid=full,
                            window=win,
                            closes_full_grid=bool(
                                abs(full["z_total"]) <= gate["z_total_max"]
                                and full["z_bin_max"] <= gate["z_bin_max"]
                                and full["chi2_dof"] <= gate["chi2_dof_max"]),
                            closes_in_window=bool(
                                abs(win["z_total"]) <= gate["z_total_max"]
                                and win["z_bin_max"] <= gate["z_bin_max"]
                                and win["chi2_dof"] <= gate["chi2_dof_max"]),
                            by_z=[dict(lo=b["lo"], hi=b["hi"], mu=b["mu"],
                                       obs=b["obs"], ratio=b["ratio"], z=b["z"])
                                  for b in tab["by_z"]],
                            pack=os.path.basename(p),
                            pack_provenance_commit=prov.get("code_commit"),
                            pack_basis_width_stamp=(
                                (prov.get("basis_pad") or {}).get("basis_width")
                                or (prov.get("latent_basis") or {}).get(
                                    "basis_width_dex")),
                        )
                        print(f"{key_for(mock, width, floor, conv, clamp):58s} "
                              f"FULL ratio={full['total_ratio']:.4f} "
                              f"chi2/dof={full['chi2_dof']:9.2f} | "
                              f"WINDOW ratio={win['total_ratio']:.4f} "
                              f"z={win['z_total']:+7.1f} "
                              f"zmax={win['z_bin_max']:6.1f} "
                              f"chi2/dof={win['chi2_dof']:8.2f} "
                              f"({time.time()-t0:.1f}s)", flush=True)

    # ---- cross-check the window routine against the COMMITTED gate ----------
    ref_key = key_for("2lpt0", 0.1, None, "const_extrap", "both")
    ref_pack = load_pack(os.path.join(
        PACKDIR, f"modelA_pack_2lpt0{tag_for(0.1, None, 'const_extrap')}.npz"))
    ref = _RP().forward_closure_gate(ref_pack, resp_clamp="both")
    inline = rows[ref_key]["full_grid"]
    xcheck = dict(
        what=("the committed gate's own arithmetic on the committed 0.1-dex "
              "UNPADDED pack, against reporting.window_closure_metrics with NO "
              "window restriction. They must agree to 1e-12: the windowed "
              "numbers are only trustworthy if the unrestricted call reproduces "
              "the committed gate exactly."),
        routine="CDDF_analysis/hbi_mcmc/run_posterior.py:forward_closure_gate",
        config=ref_key,
        committed_total_ratio=ref["total_ratio"],
        inline_total_ratio=inline["total_ratio"],
        committed_chi2_dof=ref["chi2_dof"], inline_chi2_dof=inline["chi2_dof"],
        committed_z_total=ref["z_total"],
        inline_abs_z_total=abs(inline["z_total"]),
        committed_z_bin_max=ref["z_bin_max"],
        inline_z_bin_max=inline["z_bin_max"],
        agrees=bool(np.isclose(ref["total_ratio"], inline["total_ratio"],
                               rtol=1e-12)
                    and np.isclose(ref["chi2_dof"], inline["chi2_dof"],
                                   rtol=1e-12)
                    and np.isclose(ref["z_bin_max"], inline["z_bin_max"],
                                   rtol=1e-12)),
        committed_pass=bool(ref["pass"]),
    )
    print("\n[xcheck committed gate]", json.dumps(xcheck, indent=1), flush=True)
    if not xcheck["agrees"]:
        raise SystemExit(
            "[adopted] REFUSING to stamp: the unrestricted window_closure_"
            "metrics call does not reproduce the committed forward_closure_gate")

    # ---- pack stamp audit (fail closed on a dirty input) -------------------
    pack_commits = sorted({v["pack_provenance_commit"] for v in rows.values()})
    sha = full_sha()
    changed = (_files_changed_between(pack_commits[0], sha)
               if len(pack_commits) == 1 and pack_commits[0] else None)
    verdict_stamp = pack_stamp_verdict(pack_commits, sha,
                                       [] if changed is None else changed)
    if changed is None and not verdict_stamp["packs_match_closure_commit"]:
        verdict_stamp["ok"] = False
        verdict_stamp["reason"] = (
            "could not diff the pack commit against the closure commit, so the "
            "mismatch cannot be cleared. Re-run --phase extract.")
    stamp_audit = dict(
        what=("the extract phase runs in its own process/env, so each pack "
              "stamps itself; this reconciles those stamps against the closure "
              "phase's code_commit AND decides, from the actual file diff, "
              "whether a mismatch could have changed a pack."),
        routine="CDDF_analysis/hbi_mcmc/adopted_config.py:pack_stamp_verdict",
        closure_phase_code_commit=sha,
        pack_code_commits=pack_commits,
        n_packs=len(rows) // len(CLAMPS),
        all_packs_same_commit=bool(len(pack_commits) == 1),
        **verdict_stamp,
    )
    if not stamp_audit["ok"]:
        raise SystemExit("[adopted] REFUSING to stamp: "
                         + stamp_audit["reason"])
    if not stamp_audit["packs_match_closure_commit"]:
        print("[adopted] pack-stamp mismatch CLEARED: "
              + stamp_audit["reason"], flush=True)

    syst = convention_systematic_block(rows)
    resid = residual_decomposition_block(rows)
    verdict = build_verdict(rows, syst, resid)
    out = dict(
        metadata=_metadata(sha, stamp_audit, t_start),
        adopted_configuration=_adopted_block(),
        verdict=verdict,
        convention_systematic=syst,
        residual_decomposition=resid,
        closure=rows,
        latent_basis_grids=grids,
        committed_gate_crosscheck=xcheck,
        plotting_grid_disclosure=REP.plotting_grid_disclosure(ADOPTED_WIDTH),
        z_criterion=REP.Z_CRITERION,
        omega_policy=_omega_block(),
        coverage=(coverage_block(n_sims=N_SBC_SIMS, seed=SBC_SEED)
                  if N_SBC_SIMS else dict(
                      skipped="--sbc-sims 0", reason="coverage not requested")),
        limitations=_limitations(
            verdict.get("extrapolated_response_inside_the_omega_window")),
    )
    # FAIL CLOSED before writing: an artifact may not carry two contradictory
    # statements about the same measured quantity (referee defect 4).
    assert_no_contradictory_chi2_claims(out)
    out["metadata"]["self_consistency_checks"] = {
        "no_contradictory_chi2_claims": True,
        "routine": ("CDDF_analysis/hbi_mcmc/adopted_config.py:"
                    "assert_no_contradictory_chi2_claims"),
        "measured_window_chi2_gain_by_mock": window_chi2_gain(rows),
        "what_it_refuses": (
            "any 'order of magnitude' class claim about the window's chi2 gain "
            "when the MAX measured full-grid/in-window chi2/dof factor is below "
            "10. This gate exists because commit fd60337 corrected "
            "residual_decomposition and left verdict.what_the_window_removes "
            "asserting the opposite."),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[adopted] wrote {OUT}")
    return out


# ---------------------------------------------------------------------------
# artifact blocks
# ---------------------------------------------------------------------------
def _adopted_block():
    REP = _REP()
    blk = dict(REP.ADOPTED_CONFIG)
    blk["nonident_edge_source"] = REP.NONIDENT_EDGE_SOURCE
    blk["nonident_edge_reason"] = REP.NONIDENT_EDGE_REASON
    blk["response_anchor_ceiling_reason"] = REP.RESPONSE_ANCHOR_CEILING_REASON
    blk["extract_command"] = (
        "python CDDF_analysis/hbi_mcmc/extract_pack.py --mocks <mock> "
        f"--basis-width {ADOPTED_WIDTH} --basis-pad-floor {ADOPTED_FLOOR} "
        f"--completeness-below-floor {ADOPTED_CONV}")
    return blk


def _omega_block():
    REP = _REP()
    from CDDF_analysis.hbi_mcmc import model_a as MA
    return dict(
        rule=REP.OMEGA_RULE,
        enforced_in=[
            "CDDF_analysis/hbi_mcmc/reporting.py:omega_decision (the rule)",
            "CDDF_analysis/hbi_mcmc/model_a.py:posterior_summary (the "
            "paper-facing emission point: omega_allz is None and an "
            "omega_REFUSED block carries the reason)",
            "CDDF_analysis/hbi_mcmc/model_a.py:plugin_map_diagnostic",
        ],
        NOT_enforced_in=[
            "model_a.reduce_f_posterior's raw omega_20p0 / omega_20p3 DRAW "
            "arrays and evidence.reported_quantities' omega_* R-hat/ESS "
            "entries: those are open-topped by construction and are retained "
            "as CONVERGENCE and rung-ladder DIAGNOSTICS on mocks. They are not "
            "reported values and no artifact quotes them as Omega_HI. Stated "
            "here so the guard is not mistaken for a universal one.",
        ],
        decisions={t: REP.omega_decision(*MA.TIERS[t]) for t in MA.TIERS},
        no_tail_extrapolation=(
            "NONE is implemented. Extending Omega_HI above 21.6 requires a tail "
            "treatment, and the PI ruled that is a PI decision. This code "
            "refuses rather than invents one."),
    )


def convention_systematic_block(rows):
    """The clamp x completeness 2x2 as a PROPAGATED per-bin + integrated
    systematic at the ADOPTED basis width and pad floor (PI decision 4)."""
    REP = _REP()
    out = {}
    for mock in MOCKS:
        corners_tot, corners_win, corners_bin = {}, {}, {}
        for conv in CONVENTIONS:
            for clamp in CLAMPS:
                k = key_for(mock, ADOPTED_WIDTH, ADOPTED_FLOOR, conv, clamp)
                r = rows[k]
                lbl = f"clamp={clamp}|cmp={conv}"
                corners_tot[lbl] = r["full_grid"]["total_mu"]
                corners_win[lbl] = r["window"]["total_mu"]
                corners_bin[lbl] = [b["mu"] for b in r["window"]["per_bin"]]
        adopted = f"clamp={ADOPTED_CLAMP}|cmp={ADOPTED_CONV}"
        win_bins = [[b["lo"], b["hi"]] for b in
                    rows[key_for(mock, ADOPTED_WIDTH, ADOPTED_FLOOR,
                                 ADOPTED_CONV, ADOPTED_CLAMP)
                         ]["window"]["per_bin"]]
        out[mock] = dict(
            integrated_full_grid=REP.convention_systematic(corners_tot, adopted),
            integrated_reporting_window=REP.convention_systematic(
                corners_win, adopted),
            per_bin_reporting_window=dict(
                nhat_bins=win_bins,
                **REP.convention_systematic(corners_bin, adopted)),
        )
    fr = {m: out[m]["integrated_reporting_window"]["frac_conv"] for m in MOCKS}
    fs_full = {m: out[m]["integrated_full_grid"]["frac_span"] for m in MOCKS}
    fs_win = {m: out[m]["integrated_reporting_window"]["frac_span"] for m in MOCKS}
    return dict(
        what=("the clamp x completeness convention dependence, propagated as a "
              "NAMED per-bin and integrated systematic on the predicted counts "
              "at the ADOPTED basis width and pad floor. Neither convention is "
              "a free choice and neither is being chosen here."),
        quantity=("the PREDICTED COUNTS mu of the truth-fold, integrated over "
                  "the stated grid, and per observed n-hat bin inside the "
                  "reporting window."),
        estimator=REP.CONVENTION_SYSTEMATIC,
        per_mock=out,
        frac_conv_integrated_in_window_by_mock=fr,
        cross_mock_note=("the three mocks do NOT agree to a single tolerance; "
                         "the max over mocks is the number to carry, and it is "
                         "reported per mock so no single mock can be quoted as "
                         "'the' systematic."),
        frac_conv_in_window_max_over_mocks=max(fr.values()),
        frac_span_full_grid_by_mock=fs_full,
        frac_span_in_window_by_mock=fs_win,
        reconciliation_of_the_5p5_percent_figure=(
            "the '~5.5% of the total at pad 19.0' figure in circulation is the "
            "FULL bracket WIDTH on the FULL observed grid, and it is "
            f"reproduced here: max-min over the 2x2 corners divided by the "
            f"adopted corner is "
            f"{100 * max(fs_full.values()):.2f}% "
            f"(min over mocks {100 * min(fs_full.values()):.2f}%) at basis "
            f"width {ADOPTED_WIDTH} and pad {ADOPTED_FLOOR}. Two corrections "
            "to how it should be quoted: (i) the PROPAGATED systematic is the "
            "HALF-span, i.e. "
            f"{100 * max(out[m]['integrated_full_grid']['frac_conv'] for m in MOCKS):.2f}% "
            "on the full grid, not 5.5%; (ii) INSIDE the reporting window the "
            "bracket is smaller: full width "
            f"{100 * max(fs_win.values()):.2f}%, half-span "
            f"{100 * max(fr.values()):.2f}%. Reporting only [19.7, 21.6] "
            "therefore also nearly HALVES the convention systematic, which is a "
            "second, previously unmeasured benefit of the window."),
    )


# ---------------------------------------------------------------------------
# the window's chi2 gain: MEASURED, and its narrative GENERATED from it
# ---------------------------------------------------------------------------
# 2026-07-29 (referee defect 4): `verdict.what_the_window_removes` asserted the
# windowed chi2/dof "falls by more than an order of magnitude" while
# `residual_decomposition.correction`, in the SAME artifact, explicitly refuted
# that reading. Commit fd60337 patched the correction and left the verdict. The
# cure for a hand-written number contradicting a computed one is to stop hand-
# writing it: the statement below is GENERATED from the measured factors, and
# `assert_no_contradictory_chi2_claims` refuses to stamp an artifact whose prose
# claims a >=10x gain the numbers do not support.
_ORDER_OF_MAGNITUDE_PHRASES = (
    "order of magnitude",
    "orders of magnitude",
    "more than 10x",
    "more than 10-fold",
    "tenfold",
)

# narrative fields that are allowed to talk about the chi2 gain at all; each is
# a (block, key) path into the artifact.
_CHI2_CLAIM_FIELDS = (
    ("verdict", "what_the_window_removes"),
    ("verdict", "answer_detail"),
    ("verdict", "residual_inside_the_window"),
    ("residual_decomposition", "correction"),
    ("residual_decomposition", "why_the_ceiling_is_still_right"),
    ("residual_decomposition", "inside_the_window_the_residual_is_a_FLOOR_EFFECT"),
)


def window_chi2_gain(rows):
    """MEASURED full-grid chi2/dof divided by in-window chi2/dof, per mock,
    at the adopted configuration."""
    out = {}
    for m in MOCKS:
        r = rows[key_for(m, ADOPTED_WIDTH, ADOPTED_FLOOR, ADOPTED_CONV,
                         ADOPTED_CLAMP)]
        full = float(r["full_grid"]["chi2_dof"])
        win = float(r["window"]["chi2_dof"])
        out[m] = full / win if win > 0 else float("inf")
    return out


def window_removal_statement(factors):
    """The ``what_the_window_removes`` narrative, GENERATED from the measured
    per-mock chi2/dof gain factors so it cannot contradict them."""
    vals = [float(v) for v in factors.values()]
    lo, hi = min(vals), max(vals)
    body = (
        "the reporting window drops the 2 non-identifiable bins below 19.7 and "
        "the 8 observed bins at/above 21.6, leaving 19 of 29 observed n-hat "
        "bins. MEASURED chi2/dof gain from the restriction: "
        + " / ".join(f"{k} {float(v):.2f}x" for k, v in factors.items())
        + f" (range {lo:.1f}-{hi:.1f}x). ")
    if hi >= 10.0:
        body += ("That IS an order of magnitude or more on at least one mock. ")
    else:
        body += (
            "🔴 THAT IS NOT AN ORDER OF MAGNITUDE, and an earlier version of "
            "this field said it was. It is a factor "
            f"{lo:.1f}-{hi:.1f}. ")
    body += (
        "AND THE GAIN IS NOT THE HIGH-N BINS: see "
        "residual_decomposition.correction — the bins at/above 21.6 carry only "
        "0.2-0.8% of the full-grid chi2 (they are count-starved), while the two "
        "NON-IDENTIFIABLE bins below 19.7 carry 87.4-91.4% of it. The window's "
        "chi2 improvement is almost entirely the sub-19.7 bins. The 21.6 "
        "ceiling is still right, for a reason that is NOT chi2: above it the "
        "per-bin mu/obs runs 1.05-1.81x on an EXTRAPOLATED response. Note "
        "further that capping at 21.6 does NOT put the whole window on measured "
        "response — see verdict."
        "extrapolated_response_inside_the_omega_window.")
    return body


def assert_no_contradictory_chi2_claims(artifact):
    """FAIL CLOSED if the artifact's prose claims a chi2 gain its own numbers refute.

    A stamped artifact must not be able to carry two contradictory statements
    about the same measured quantity.  This scans the narrative fields that are
    allowed to discuss the window's chi2 gain for "order of magnitude"-class
    phrases and checks them against the MAX measured factor in
    ``residual_decomposition.per_mock`` (full_grid chi2/dof over reporting_window
    chi2/dof).  A phrase claiming >=10x with a measured max below 10 raises.
    """
    REP = _REP()
    per_mock = ((artifact.get("residual_decomposition") or {})
                .get("per_mock") or {})
    factors = {}
    for m, blk in per_mock.items():
        try:
            full = float(blk["full_grid"]["chi2_dof"])
            win = float(blk["reporting_window"]["chi2_dof"])
        except (KeyError, TypeError):
            continue
        factors[m] = full / win if win > 0 else float("inf")
    if not factors:
        raise REP.ReportingGuardError(
            "assert_no_contradictory_chi2_claims: no measured chi2/dof factors "
            "in residual_decomposition.per_mock — the scanner cannot vouch for "
            "any claim, so it refuses (fail closed).")
    worst = max(factors.values())
    offenders = []
    for block, key in _CHI2_CLAIM_FIELDS:
        txt = ((artifact.get(block) or {}).get(key) or "")
        if not isinstance(txt, str):
            continue
        low = txt.lower()
        for phrase in _ORDER_OF_MAGNITUDE_PHRASES:
            if phrase not in low:
                continue
            # a field is allowed to say "NOT an order of magnitude"
            i = low.find(phrase)
            before = low[max(0, i - 60):i]
            if "not " in before or "n't " in before:
                continue
            if worst < 10.0:
                offenders.append(f"{block}.{key}: {phrase!r}")
    if offenders:
        raise REP.ReportingGuardError(
            "CONTRADICTORY CHI2 CLAIM: the artifact asserts an order-of-"
            f"magnitude chi2 improvement in {len(offenders)} field(s) "
            + "; ".join(offenders)
            + f" while its own residual_decomposition measures a MAX factor of "
            f"{worst:.2f}x ("
            + ", ".join(f"{k} {v:.2f}x" for k, v in sorted(factors.items()))
            + "). A stamped artifact may not carry two contradictory statements "
              "about the same measured quantity.")
    return True


def extrapolated_response_block(rows):
    """DEFECT 3 (referee, 2026-07-29): how much of the AUTHORIZED Omega window
    still sits on EXTRAPOLATED response, and how much N-weighted Omega that is.

    The dex comes from ``reporting.extrapolated_response_inside_window`` (the
    frozen response's own anchors).  The Omega SHARE is measured here, from the
    in-window per-bin counts of the adopted folds: an N-weighted share, because
    Omega_HI is an N-weighted mass and a plain bin count badly understates the
    top of the window.
    """
    REP = _REP()
    ex = dict(REP.extrapolated_response_inside_window())
    # the sub-interval that is above EVERY response cell's top anchor, snapped
    # DOWN to the observed 0.1-dex grid so it is a union of whole reported bins.
    # round to the observed grid's own precision: np.floor(21.216358 / 0.1) * 0.1
    # is 21.200000000000003 in binary floating point, and that is not a number to
    # put in a PI-facing headline.
    edge = round(float(np.floor(ex["top_anchor_max"] / REP.OBSERVED_STEP + 1e-9)
                       * REP.OBSERVED_STEP), 6)
    share_obs, share_mu = {}, {}
    for m in MOCKS:
        pb = rows[key_for(m, ADOPTED_WIDTH, ADOPTED_FLOOR, ADOPTED_CONV,
                          ADOPTED_CLAMP)]["window"]["per_bin"]
        lo = np.array([float(b["lo"]) for b in pb])
        hi = np.array([float(b["hi"]) for b in pb])
        w = 10.0 ** (0.5 * (lo + hi) - 21.0)          # the Omega weight
        sel = lo >= edge - 1e-9
        for name, key, dest in (("obs", "obs", share_obs), ("mu", "mu", share_mu)):
            v = np.array([float(b[key]) for b in pb])
            tot = float((v * w).sum())
            dest[m] = float((v * w)[sel].sum() / tot) if tot > 0 else float("nan")
    ex.update(
        what=("the fraction of the AUTHORIZED Omega window that sits above the "
              "response's top measured anchor, and the N-weighted Omega share "
              "carried there. MOCK truth-folds only; no population value."),
        routine="CDDF_analysis/hbi_mcmc/adopted_config.py:extrapolated_response_block",
        anchor_source=REP.RESPONSE_ANCHOR_MEASURED,
        subinterval_logN=[edge, REP.RESPONSE_ANCHOR_CEILING],
        subinterval_note=(
            f"[{edge}, {REP.RESPONSE_ANCHOR_CEILING}) is the part of the window "
            "that lies above the BEST-anchored response cell's top anchor "
            f"({ex['top_anchor_max']:.4f}), snapped down to the observed "
            f"{REP.OBSERVED_STEP}-dex grid so it is a union of whole reported "
            "bins. For the WORST-anchored cell the extrapolated part starts "
            f"lower still, at {ex['top_anchor_min']:.4f}."),
        omega_share_of_subinterval_by_mock_truth_counts=share_obs,
        omega_share_of_subinterval_by_mock_predicted_counts=share_mu,
        omega_share_definition=(
            "sum over n-hat bins in the sub-interval of counts * 10^(Nc - 21), "
            "divided by the same sum over ALL bins inside [19.7, 21.6]. "
            "N-weighted, because Omega_HI is an N-weighted mass."),
        headline=(
            f"{100 * min(share_obs.values()):.1f}-{100 * max(share_obs.values()):.1f}% "
            "of the in-window N-weighted Omega comes from "
            f"[{edge}, {REP.RESPONSE_ANCHOR_CEILING}), which is entirely above "
            "the top measured response anchor. THE PI MUST NOT LEARN THIS "
            "LATER."),
    )
    return ex


def build_verdict(rows, syst, resid=None):
    REP = _REP()
    gate = dict(_RP().GATE)
    ad = {m: rows[key_for(m, ADOPTED_WIDTH, ADOPTED_FLOOR, ADOPTED_CONV,
                          ADOPTED_CLAMP)] for m in MOCKS}
    closing_full = [k for k, v in rows.items() if v["closes_full_grid"]]
    closing_win = [k for k, v in rows.items() if v["closes_in_window"]]

    def _leg(v, w):
        d = v[w]
        return dict(
            total_ratio=d["total_ratio"], z_total=d["z_total"],
            abs_z_total=abs(d["z_total"]), z_bin_max=d["z_bin_max"],
            chi2_dof=d["chi2_dof"], n_bins=d["n_bins_in_window"],
            per_bin_ratio_min=min(b["ratio"] for b in d["per_bin"]
                                  if np.isfinite(b["ratio"])),
            per_bin_ratio_max=max(b["ratio"] for b in d["per_bin"]
                                  if np.isfinite(b["ratio"])),
            legs=dict(
                z_total_leg=bool(abs(d["z_total"]) <= gate["z_total_max"]),
                z_bin_leg=bool(d["z_bin_max"] <= gate["z_bin_max"]),
                chi2_leg=bool(d["chi2_dof"] <= gate["chi2_dof_max"]),
            ),
            factor_over_chi2_gate=d["chi2_dof"] / gate["chi2_dof_max"],
        )

    # 🔴 MEASURED, not asserted: does the verdict survive if the ONLY arm
    # allowed to speak is the one the PI actually ratified?  Relabelling the
    # |z| arms as unratified would be an empty gesture if the "NO" depended on
    # them.  This counts, per window, the configurations that fail the
    # RATIFIED chi2/dof <= 3 arm on its own.
    def _chi2_only(win):
        fac = sorted(rows[k][win]["chi2_dof"] / gate["chi2_dof_max"]
                     for k in rows)
        return dict(
            n_configurations=len(rows),
            n_failing_ratified_chi2_arm_alone=sum(1 for f in fac if f > 1.0),
            min_factor_over_ratified_chi2_gate=fac[0],
            max_factor_over_ratified_chi2_gate=fac[-1])

    rests = dict(
        what=("the RATIFIED arm is chi2/dof <= 3 (PI decision 8) and it is the "
              "ONLY ratified numerical closure gate. The four |z| arms are "
              "RESTATED_NOT_RATIFIED and the two ratio-span arms are "
              "UNRATIFIED. This block answers: is the answer 'NO' still 'NO' "
              "with only the ratified arm armed?"),
        full_grid=_chi2_only("full_grid"),
        window=_chi2_only("window"),
    )
    rests["answer"] = (
        "YES -- the verdict rests on the ratified arm alone"
        if (rests["window"]["n_failing_ratified_chi2_arm_alone"]
            == rests["window"]["n_configurations"]
            and rests["full_grid"]["n_failing_ratified_chi2_arm_alone"]
            == rests["full_grid"]["n_configurations"])
        else "NO -- at least one configuration passes the ratified chi2 arm "
             "and is failed only by an UNRATIFIED arm. The relabelling is "
             "then verdict-bearing and a PI must see this line.")

    return dict(
        question=("under the ADOPTED configuration (0.2-dex latent basis, pad "
                  "floor 19.0, molly172 sub-floor completeness), does the "
                  "forward model close inside the PRIMARY REPORTING WINDOW "
                  "19.7 <= logN <= 21.6, even though it fails over the full "
                  "observed grid?"),
        answer="NO",
        answer_detail=(
            "the window restriction is a LARGE, real improvement and it is not "
            "enough. On the adopted configuration the total ratio moves from "
            f"{ad['2lpt0']['full_grid']['total_ratio']:.4f} (full grid) to "
            f"{ad['2lpt0']['window']['total_ratio']:.4f} (window) on 2LPT-0 and "
            f"chi2/dof from {ad['2lpt0']['full_grid']['chi2_dof']:.1f} to "
            f"{ad['2lpt0']['window']['chi2_dof']:.1f}, but the windowed chi2/dof "
            f"is still {ad['2lpt0']['window']['chi2_dof'] / gate['chi2_dof_max']:.0f}x "
            "the RATIFIED tolerance of 3.0 (chi2/dof <= 3, PI decision 8 -- "
            "the only ratified numerical closure gate) and max|z_bin| is still "
            f"{ad['2lpt0']['window']['z_bin_max']:.1f} against a tolerance of 5 "
            "(z_bin_max: RESTATED_NOT_RATIFIED -- it gates, nobody ratified "
            "it; see verdict.gate_authority and "
            "verdict.verdict_rests_on_the_ratified_arm_alone). "
            "No configuration in the 48-fold cross closes in the window. "
            "WHERE THE IMPROVEMENT COMES FROM is NOT what it looks like -- see "
            "residual_decomposition.correction: the bins at/above 21.6 carry "
            "under 1% of the full-grid chi2 (they are count-starved), and it is "
            "the two NON-IDENTIFIABLE bins below 19.7 that carried ~90% of it."),
        gate_tolerances=gate,
        # 🔴 EVERY ONE OF THESE IS READ FROM reporting.GATE_AUTHORITY, THE ONE
        # TABLE.  Until 2026-08-05 this site wrote the LITERAL
        #     gate_tolerances_ratified=["z_total_max","z_bin_max","chi2_dof_max"]
        # into the committed artifact.  That was FABRICATED PI AUTHORITY:
        # decision 8 ratified three things, of which exactly one is a gate
        # tolerance (chi2/dof <= 3), and it called |z| <= 5 MALFORMED AS
        # STATED and sent it back for RESTATEMENT -- the opposite of ratifying
        # it.  A literal typed at a call site is unfalsifiable; a read from a
        # guarded table is not.
        # MERGE (2026-08-05): ``REP.ratified_gate_tolerances`` now DERIVES from
        # CDDF_analysis.hbi_mcmc.ratification, the single source. Spelled out
        # here rather than hidden behind the accessor so the claim and its
        # provenance are readable in one place -- the tree scanner's R7 rule
        # exists because a ratified-name list computed from an unnamed source is
        # not checkable.
        gate_tolerances_ratified=[
            n for n in REP.GATE_AUTHORITY
            if REP._status_from_ratification(n) == REP.RATIFIED],
        gate_tolerances_restated_not_ratified=list(
            REP.restated_not_ratified_gate_tolerances()),
        gate_tolerances_unratified=list(REP.unratified_gate_tolerances()),
        # name means what it says: EVERY tolerance that is not ratified.  The
        # old field carried only the two declined ratio-span numbers, which
        # invited the reading "so the other five are ratified".
        gate_tolerances_not_ratified=list(REP.not_ratified_gate_tolerances()),
        gate_tolerances_unratified_but_gating=list(
            REP.unratified_but_gating_gate_tolerances()),
        gate_authority=REP.gate_authority_stamp(),
        verdict_rests_on_the_ratified_arm_alone=rests,
        n_configurations=len(rows),
        n_closing_full_grid=len(closing_full),
        n_closing_in_window=len(closing_win),
        closing_configurations_full_grid=closing_full,
        closing_configurations_in_window=closing_win,
        adopted_configuration_by_mock={
            m: dict(full_grid=_leg(ad[m], "full_grid"),
                    reporting_window=_leg(ad[m], "window"),
                    counts_total=ad[m]["counts_total"],
                    truth_total=ad[m]["truth_total"],
                    truth_total_below_reporting_floor=ad[m][
                        "truth_total_below_reporting_floor"])
            for m in MOCKS},
        basis_width_effect_on_window_closure={
            m: {f"bw={w}": dict(
                total_ratio=rows[key_for(m, w, ADOPTED_FLOOR, ADOPTED_CONV,
                                         ADOPTED_CLAMP)]["window"]["total_ratio"],
                chi2_dof=rows[key_for(m, w, ADOPTED_FLOOR, ADOPTED_CONV,
                                      ADOPTED_CLAMP)]["window"]["chi2_dof"],
                z_bin_max=rows[key_for(m, w, ADOPTED_FLOOR, ADOPTED_CONV,
                                       ADOPTED_CLAMP)]["window"]["z_bin_max"])
                for w in WIDTHS} for m in MOCKS},
        basis_width_note=(
            "the basis WIDTH is not a closure lever and was never claimed to "
            "be: merging basis columns is an exact statement about the "
            "REPRESENTATION of f, and at the pack's own truth the merged truth "
            "reproduces almost the same folded counts. Decision 3 was taken on "
            "CONDITIONING grounds (E4: 28x lower condition number, 45.7-62x "
            "per-bin noise amplification at 0.1 dex), not to fix closure. The "
            "numbers above are the check that it does not BREAK closure."),
        pad_effect_on_window_closure={
            m: {f"pad={f}": dict(
                total_ratio=rows[key_for(m, ADOPTED_WIDTH, f, ADOPTED_CONV,
                                         ADOPTED_CLAMP)]["window"]["total_ratio"],
                chi2_dof=rows[key_for(m, ADOPTED_WIDTH, f, ADOPTED_CONV,
                                      ADOPTED_CLAMP)]["window"]["chi2_dof"])
                for f in FLOORS} for m in MOCKS},
        what_the_window_removes=window_removal_statement(
            window_chi2_gain(rows)),
        residual_inside_the_window=(
            "what is LEFT inside the window is not D2 and is not the pad; it is "
            "a FLOOR EFFECT plus a floor of its own. The window's per-bin ratio "
            "range on the adopted 2LPT-0 configuration is "
            f"{_leg(ad['2lpt0'], 'window')['per_bin_ratio_min']:.4f} to "
            f"{_leg(ad['2lpt0'], 'window')['per_bin_ratio_max']:.4f}, and "
            + ((f"{100 * min(resid['per_mock'][m]['reporting_window']['chi2_frac_from_the_two_bins_just_above_the_floor'] for m in MOCKS):.0f}"
                f"-{100 * max(resid['per_mock'][m]['reporting_window']['chi2_frac_from_the_two_bins_just_above_the_floor'] for m in MOCKS):.0f}% "
                "of the in-window chi2 sits in the TWO bins immediately above "
                "19.7. Excluding those two bins chi2/dof is "
                f"{min(resid['per_mock'][m]['reporting_window']['chi2_dof_excluding_the_two_bins_just_above_the_floor'] for m in MOCKS):.1f}"
                f"-{max(resid['per_mock'][m]['reporting_window']['chi2_dof_excluding_the_two_bins_just_above_the_floor'] for m in MOCKS):.1f}, "
                "so raising the floor further would buy a factor ~4-5 and STILL "
                "NOT CLOSE. ") if resid else "")
            + "That is a systematic no sampler and no window can repair."),
        convention_systematic_headline=(
            "the clamp x completeness 2x2 half-span is "
            f"{100 * syst['frac_conv_in_window_max_over_mocks']:.2f}% "
            "(max over mocks) of the integrated predicted counts INSIDE the "
            "reporting window at the adopted basis and pad — full bracket width "
            f"{100 * max(syst['frac_span_in_window_by_mock'].values()):.2f}%, "
            "against a full width of "
            f"{100 * max(syst['frac_span_full_grid_by_mock'].values()):.2f}% "
            "over the whole observed grid. It is carried as a SEPARATE LINEAR "
            "ENVELOPE, not in quadrature (see "
            "convention_systematic.estimator)."),
        extrapolated_response_inside_the_omega_window=extrapolated_response_block(
            rows),
        omega_hi=("NOT emitted as a total. Only Omega_HI limited to "
                  "[19.7, 21.6] and labelled as such is emittable "
                  "(omega_policy). 🔴 AND EVEN THAT WINDOW IS NOT ALL MEASURED "
                  "RESPONSE — see extrapolated_response_inside_the_omega_window: "
                  "the top ~0.4 dex of it is EXTRAPOLATED and carries ~28% of "
                  "the N-weighted Omega."),
        rung10=("STAYS GATED. No configuration passes the pre-flight, in the "
                "window or out of it."),
        next_actions=[
            "the binding constraint is now the IN-WINDOW residual, and it "
            "decomposes: ~3/4 of it is the two 0.1-dex bins immediately above "
            "the 19.7 floor (an edge effect one bin higher than the one "
            "NONIDENT_EDGE was created for), and the rest is a ~4x-over-gate "
            "floor that survives removing them. Attack both, in that order, "
            "before any further window or pad work.",
            "residual D2 is NOT closed and is NOT chi2-visible: it is a "
            "1.05-1.81x per-bin PHYSICAL bias in a count-starved tail. Any "
            "future gate arm intended to catch it must be a RATIO-SPAN arm, "
            "which is precisely the arm PI decision 8 left unratified. The "
            "measured spans are recorded in residual_decomposition."
            "prospective_ratio_span_calibration as calibration input.",
            "resp_clamp is still not a free choice; it is carried as half of "
            "the convention systematic and the response below ~19.35 remains "
            "unmeasured.",
            "coverage under the adopted 0.2-dex basis is measured only at the "
            "reduced SBC scale (see coverage block); a matched-configuration "
            "SBC on the production geometry is a compute decision for the PI.",
        ],
    )


def residual_decomposition_block(rows):
    """WHERE the misfit actually lives, per mock, at the ADOPTED configuration.

    Written because the first reading of these numbers was WRONG in a way that
    matters. "The window helps, so the excluded high-N bins were carrying the
    misfit" is a natural inference and it is FALSE here: the >= 21.6 bins carry
    under 1% of the full-grid chi2, because they are count-starved. Almost all of
    the chi2 improvement from the window comes from dropping the two
    NON-IDENTIFIABLE bins below 19.7.

    The 21.6 ceiling is still right, and this block says why in the only terms
    that survive scrutiny: it removes a large PHYSICAL bias (mu/obs 1.05-1.81 per
    bin) that a chi2 cannot see. That is exactly the regime the (UNRATIFIED)
    ratio-span gate arms were invented for, so the measured spans are reported
    here as PROSPECTIVE calibration input for PI decision 8 -- not as a
    threshold, and not as a ratification.
    """
    import numpy as np
    out = {}
    for mock in MOCKS:
        row = rows[key_for(mock, ADOPTED_WIDTH, ADOPTED_FLOOR, ADOPTED_CONV,
                           ADOPTED_CLAMP)]
        leg = {}
        for name, blk in (("full_grid", row["full_grid"]),
                          ("reporting_window", row["window"])):
            pb = [b for b in blk["per_bin"] if b["obs"] > 0]
            z = np.array([b["z"] for b in pb], float)
            lo = np.array([b["lo"] for b in pb], float)
            r = np.array([b["ratio"] for b in pb], float)
            chi2 = float((z ** 2).sum())
            below = lo < NONIDENT_EDGE_LOCAL - 1e-9
            above = lo >= CEILING_LOCAL - 1e-9
            first_two = below if name == "full_grid" else (
                lo < NONIDENT_EDGE_LOCAL + 0.2 - 1e-9)
            leg[name] = dict(
                n_bins=int(len(z)), chi2=chi2,
                chi2_dof=float(chi2 / max(len(z), 1)),
                chi2_frac_from_bins_below_19p7=float(
                    (z[below] ** 2).sum() / chi2) if chi2 > 0 else None,
                chi2_frac_from_bins_at_or_above_21p6=float(
                    (z[above] ** 2).sum() / chi2) if chi2 > 0 else None,
                chi2_frac_from_the_two_bins_just_above_the_floor=float(
                    (z[first_two] ** 2).sum() / chi2) if chi2 > 0 else None,
                chi2_dof_excluding_the_two_bins_just_above_the_floor=(
                    float((z[~first_two] ** 2).sum() / max((~first_two).sum(), 1))
                    if (~first_two).any() else None),
                ratio_span=float(r.max() - r.min()),
                ratio_min=float(r.min()), ratio_max=float(r.max()),
                ratio_by_bin_at_or_above_21p6=[float(x) for x in r[above]],
            )
        zr = [b["ratio"] for b in row["by_z"] if b["obs"] > 0]
        leg["by_z_ratio_span_full_grid"] = float(max(zr) - min(zr))
        out[mock] = leg
    fr = [out[m]["reporting_window"][
        "chi2_frac_from_the_two_bins_just_above_the_floor"] for m in MOCKS]
    ex = [out[m]["reporting_window"][
        "chi2_dof_excluding_the_two_bins_just_above_the_floor"] for m in MOCKS]
    hi = [out[m]["full_grid"]["chi2_frac_from_bins_at_or_above_21p6"]
          for m in MOCKS]
    bl = [out[m]["full_grid"]["chi2_frac_from_bins_below_19p7"] for m in MOCKS]
    return dict(
        what=("where the misfit lives, per mock, at the adopted configuration; "
              "computed from the per-bin z of the same folds."),
        routine=("CDDF_analysis/hbi_mcmc/adopted_config.py:"
                 "residual_decomposition_block"),
        per_mock=out,
        correction=(
            "CORRECTION, stated because the obvious reading is wrong: the "
            "window's chi2 improvement is NOT mostly the high-N bins. On the "
            "full observed grid the bins at/above 21.6 carry only "
            f"{100 * min(hi):.1f}-{100 * max(hi):.1f}% of chi2, while the two "
            "NON-IDENTIFIABLE bins below 19.7 carry "
            f"{100 * min(bl):.1f}-{100 * max(bl):.1f}%. The 21.6 bins are "
            "count-starved, so a 1.05-1.81x per-bin bias there produces almost "
            "no z. Any statement of the form 'the window works because it "
            "removes D2' is unsupported by these numbers."),
        why_the_ceiling_is_still_right=(
            "because a chi2 is the wrong instrument in a count-starved tail. "
            "Above 21.6 the per-bin mu/obs runs 1.05-1.81x -- a 5-81% PHYSICAL "
            "bias on exactly the bins that dominate an N-weighted Omega -- and "
            "the fitted response there is EXTRAPOLATION. Capping the reporting "
            "window removes that bias from what is reported. It does not remove "
            "it from the model."),
        inside_the_window_the_residual_is_a_FLOOR_EFFECT=(
            "of the in-window chi2, "
            f"{100 * min(fr):.0f}-{100 * max(fr):.0f}% comes from the TWO "
            "0.1-dex bins immediately above the reporting floor ([19.7, 19.9)), "
            "consistently across all three mocks. Excluding those two bins the "
            f"in-window chi2/dof is {min(ex):.1f}-{max(ex):.1f} -- still "
            f"{min(ex) / 3.0:.1f}-{max(ex) / 3.0:.1f}x the ratified tolerance of "
            "3.0. So raising the reporting floor further would buy roughly a "
            "factor 4-5 in chi2/dof AND STILL NOT CLOSE. That is the "
            "decision-relevant consequence: the floor is not the remaining "
            "defect, it is only the loudest part of it."),
        prospective_ratio_span_calibration=dict(
            status=("INPUT FOR A FUTURE PI DECISION, NOT A RATIFICATION. PI "
                    "decision 8 declined to ratify ratio_span_by_z_max = 0.10 "
                    "and ratio_span_by_snr_max = 0.15 and asked for them to be "
                    "defined and calibrated prospectively. These are MEASURED "
                    "spans on mocks whose forward model does not close, so they "
                    "cannot be turned into a tolerance yet; they are recorded so "
                    "the calibration starts from numbers."),
            measured_by_nhat_ratio_span_in_window={
                m: out[m]["reporting_window"]["ratio_span"] for m in MOCKS},
            measured_by_nhat_ratio_span_full_grid={
                m: out[m]["full_grid"]["ratio_span"] for m in MOCKS},
            measured_by_z_ratio_span_full_grid={
                m: out[m]["by_z_ratio_span_full_grid"] for m in MOCKS},
            note=("the by-N span is 3-4x smaller inside the window than over the "
                  "full grid; the by-z span is unchanged by an N window, as it "
                  "must be, since the window does not touch the z marginal."),
        ),
    )


def _metadata(sha, stamp_audit, t_start):
    return dict(
        title=("ADOPTED configuration — forward closure + coverage under the "
               "0.2-dex latent basis, pad floor 19.0/molly172, and the "
               "19.7-21.6 primary reporting window"),
        artifact="adopted_config_closure",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        code_commit=sha,
        code_commit_scope=("the CLOSURE phase (this process). The input packs "
                           "carry their own stamp — see pack_stamp_audit."),
        code_commit_dirty=dirty(),
        pack_stamp_audit=stamp_audit,
        branch=subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
            text=True).strip(),
        routines=dict(
            driver=("CDDF_analysis/hbi_mcmc/adopted_config.py "
                    "(--phase extract, then --phase closure)"),
            reporting_rules="CDDF_analysis/hbi_mcmc/reporting.py",
            extractor=("CDDF_analysis/hbi_mcmc/extract_pack.py:extract_pack "
                       "(--basis-width / --basis-pad-floor / "
                       "--completeness-below-floor)"),
            basis_grid="CDDF_analysis/hbi_mcmc/extract_pack.py:basis_pad_edges",
            merge_convention=("CDDF_analysis/hbi_mcmc/reporting.py:basis_groups "
                              "(E4's own; re-exported by e4_probe)"),
            fold=("CDDF_analysis/hbi_mcmc/forward_selftest.py:selftest "
                  "(-> forward.build_consts + forward.fold_mu)"),
            window_metrics=("CDDF_analysis/hbi_mcmc/reporting.py:"
                            "window_closure_metrics"),
            gate="CDDF_analysis/hbi_mcmc/run_posterior.py:forward_closure_gate",
        ),
        pack_dir=PACKDIR,
        rederive=(
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "conda run -n gpdla python CDDF_analysis/hbi_mcmc/adopted_config.py "
            f"--phase extract --pack-dir {PACKDIR}  &&  "
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc."
            f"adopted_config --phase closure --pack-dir {PACKDIR} --out {OUT}"),
        pack_note="packs are INPUTS, written to scratch, never committed",
        gate_tolerances=dict(_RP().GATE),
        estimand=("NONE — this artifact reports NO population measurement. It "
                  "contains forward-model CLOSURE diagnostics (predicted vs "
                  "observed counts at the packs' own truth) and the propagated "
                  "convention systematic on those predictions. No f(N), no "
                  "dN/dX, no Omega_HI value is quoted anywhere in it."),
        scope=("MOCK ONLY (2LPT-0 / london-0 / saclay-0 model-A packs). No real "
               "DESI survey values of any kind."),
        paper_facing=False,
        paper_facing_reason=("the forward model does NOT close inside the "
                             "reporting window (see verdict). Nothing here may "
                             "be promoted."),
        privacy="mock packs only; no real-LOA path is touched",
        mocks_only=True,
        wall_seconds=round(time.time() - t_start, 1),
    )


def coverage_block(n_sims=64, seed=0):
    """MATCHED-CONFIGURATION SBC on the ADOPTED latent geometry, WITH a measured
    detection curve (PI decision 8 + the project's power-check rule).

    Truth-containment is monotone in band width and therefore cannot fail an
    over-wide band -- a 2x over-dispersed posterior has previously passed 24/24
    tests and 8/8 containment on this project.  So the coverage claim is reported
    ONLY alongside the detection curve over ``sbc.DISPERSION_SCALES``: the same
    posterior draws re-scaled in log f about their per-bin median, at no extra
    sampling cost.  If s = 2.0 is not flagged, the s = 1.0 result certifies
    nothing and this block says so.
    """
    from CDDF_analysis.hbi_mcmc import sbc as S
    from CDDF_analysis.hbi_mcmc.evidence import SBC_UNIFORM_P_MIN
    t0 = time.time()
    ranks, meta = S.sbc_run(n_sims, seed=seed, grid=S.SBC_GRID_ADOPTED,
                            dispersion_scales=S.DISPERSION_SCALES,
                            verbose=True, **S.SBC_ADOPTED_BASIS)
    if not ranks:
        return dict(incomplete=["sbc_produced_no_usable_replicas"], meta=meta)
    L = meta["n_ranks_L"]
    curve = {}
    for s, rk_by_q in meta["ranks_by_scale"].items():
        per_q = {}
        worst_p, worst_name = 1.0, None
        for name, rk in sorted(rk_by_q.items()):
            t = S.uniformity_test(rk, L, n_bins=10)
            per_q[name] = dict(p_value=t["p_value"], chi2=t["chi2"],
                               shape=t["shape"],
                               edge_mass_frac=t["edge_mass_frac"],
                               hist=t["hist"]["counts"])
            if t["p_value"] is not None and t["p_value"] < worst_p:
                worst_p, worst_name = t["p_value"], name
        bonf = float(min(1.0, worst_p * max(len(per_q), 1)))
        curve[s] = dict(
            worst_p_value=float(worst_p), worst_quantity=worst_name,
            worst_p_bonferroni=bonf,
            flagged=bool(bonf < SBC_UNIFORM_P_MIN),
            per_quantity=per_q)
    flagged = {s: curve[s]["flagged"] for s in curve}
    detects_2x = bool(flagged.get("2", False))
    detects_0p5x = bool(flagged.get("0.5", False))
    over = sorted(float(s) for s in curve if float(s) > 1.0)
    under = sorted((float(s) for s in curve if float(s) < 1.0), reverse=True)
    det_over = [s for s in over if flagged[f"{s:g}"]]
    und_over = [s for s in over if not flagged[f"{s:g}"]]
    det_under = [s for s in under if flagged[f"{s:g}"]]
    und_under = [s for s in under if not flagged[f"{s:g}"]]
    blind = dict(
        smallest_DETECTED_over_dispersion=(min(det_over) if det_over else None),
        largest_UNDETECTED_over_dispersion=(max(und_over) if und_over else None),
        largest_DETECTED_under_dispersion=(max(det_under) if det_under else None),
        smallest_UNDETECTED_under_dispersion=(min(und_under) if und_under
                                              else None),
        over_dispersion_detection_is_monotone=bool(
            not und_over or not det_over or max(und_over) < min(det_over)),
        note=("these are MEASURED envelope values on the scales actually run "
              "(dispersion_scales), not an interpolation: nothing here says the "
              "test's power is monotone BETWEEN two adjacent scales, and an "
              "undetected scale ABOVE a detected one would show up as "
              "over_dispersion_detection_is_monotone = false."),
    )
    return dict(
        what=("simulation-based calibration of the SAME estimator on the ADOPTED "
              "latent geometry (0.2-dex basis, pad 19.0), with a MEASURED "
              "detection curve rather than a bare pass/fail."),
        routine="CDDF_analysis/hbi_mcmc/sbc.py:sbc_run",
        matched_configuration=dict(
            basis_width_dex=meta["basis_width"], pad_floor=meta["pad_floor"],
            n_basis_bins=meta["n_basis_bins"],
            n_observed_bins=meta["n_observed_bins"],
            n_pad_bins=meta["n_pad_bins"], ntrue_edges=meta["ntrue_edges"],
            matched_on=("the LATENT basis width and the downward pad -- the two "
                        "things decisions 3 and 4 changed"),
            NOT_matched_on=("the grid extent, the sampler scale, the prior "
                            "widths and the response realisation. Those are "
                            "reductions R1-R5, enumerated in meta.reduction_"
                            "note, and they are the reason this is a "
                            "calibration statement about the SAMPLER + FORWARD "
                            "MODEL, not about the production prior."),
        ),
        p_threshold=SBC_UNIFORM_P_MIN,
        detection_curve=curve,
        detection_curve_definition=meta["dispersion_scale_definition"],
        dispersion_scales=meta["dispersion_scales"],
        flagged_by_scale=flagged,
        power_verdict=(
            "USABLE WITH A STATED BLIND SPOT" if detects_2x
            else "NOT USABLE AS A COVERAGE CERTIFICATE"),
        detection_envelope=blind,
        power_statement=(
            f"at n_sims_used = {meta['n_sims_used']} and L = {L} the test "
            f"{'DOES' if detects_2x else 'DOES NOT'} flag a 2.0x over-dispersed "
            f"posterior and {'DOES' if detects_0p5x else 'DOES NOT'} flag a 0.5x "
            "under-dispersed one. THE BLIND SPOT, MEASURED: the largest "
            "over-dispersion this configuration does NOT flag is "
            f"{blind['largest_UNDETECTED_over_dispersion']}x, and the smallest "
            "it DOES flag is "
            f"{blind['smallest_DETECTED_over_dispersion']}x. "
            + ("So the s = 1.0 result is interpretable as 'not mis-scaled by "
               f"{blind['smallest_DETECTED_over_dispersion']}x or more', and it "
               "is NOT evidence that the band is correct to better than "
               f"~{blind['largest_UNDETECTED_over_dispersion']}x. Quote it with "
               "that qualifier or not at all."
               if detects_2x else
               "The s = 1.0 result therefore CERTIFIES NOTHING about band width: "
               "a uniform rank histogram at this power is consistent with a "
               "substantially mis-scaled posterior. Do not quote it as coverage.")),
        worst_quantity_at_s1=(curve.get("1") or {}).get("worst_quantity"),
        worst_quantity_note=(
            "if the worst quantity at s = 1 is the reporting-window functional, "
            "say so: it is the one the paper would quote and it is the one whose "
            "calibration matters, whatever the Bonferroni verdict."),
        s1_result=curve.get("1"),
        conclusion_scope=(
            "MOCK/SYNTHETIC only, REDUCED scale. This is not a coverage "
            "statement about the production 29x15x8 fit, whose forward model "
            "does not close (see verdict) -- a coverage statement about a "
            "non-closing model would be meaningless anyway."),
        wall_seconds=round(time.time() - t0, 1),
        meta=meta,
    )


def _limitations(omega_extrap=None):
    REP = _REP()
    ex = omega_extrap or REP.extrapolated_response_inside_window()
    return [
        # 🔴 FIRST, because it is the one a reader is most likely to get wrong
        # and it was not stated at all before 2026-07-29 (referee defect 3).
        ("🔴 THE AUTHORIZED Omega_HI WINDOW STILL CONTAINS EXTRAPOLATED "
         "RESPONSE. Capping the window at 21.6 does NOT put it on measured "
         f"response: the frozen response's top true-N anchor is at "
         f"{ex['top_anchor_min']:.4f}-{ex['top_anchor_max']:.4f} depending on "
         f"the response cell, so {ex['dex_extrapolated_best_cell']:.2f}-"
         f"{ex['dex_extrapolated_worst_cell']:.2f} dex of EXTRAPOLATED response "
         "sits INSIDE [19.7, 21.6] — the one window where Omega_HI is "
         "authorized. On the adopted packs the N-weighted Omega share of "
         "[21.2, 21.6) alone is 27.5-29.6% of the in-window total, and that "
         "whole sub-interval is above the BEST-anchored cell's top anchor. "
         "21.6 was chosen for a residual-excess reason (finding D2), not "
         "because the response is measured up to it. See "
         "verdict.extrapolated_response_inside_the_omega_window."),
        "the 21.6 ceiling is a REPORTING cap, not a fix. Residual D2 (the "
        "1.23-1.80x high-N excess) is still in the model; it has been moved "
        "outside what is reported. Any statement that D2 is 'closed' is wrong.",
        "the decision-4 sub-window guard is armed on every PAPER-FACING tier, "
        "which is NOT every tier: subdla_195_203 and all_195_up have windows "
        "that START below 19.7 and are therefore REFUSED as paper-facing rather "
        "than guarded (their dN/dX draws w = 0.20 dex on the non-identifiable "
        "[19.5, 19.7) basis bin). Read reporting.SUBWINDOW_GUARD_SCOPE before "
        "quoting the guard; an earlier version of that prose over-claimed.",
        "21.6 - 19.7 = 1.9 dex is an ODD multiple of 0.1, so NO uniform 0.2-dex "
        "basis can carry both window edges. The adopted grid puts 19.5 and 19.7 "
        "on exact basis edges and leaves 21.6 astride the [21.5, 21.7) basis "
        "bin. Integrated reported quantities split that bin by dex overlap, "
        "which is exact under the adopted merging convention; DIFFERENTIAL "
        "per-bin reporting on the 0.2-dex basis therefore stops at 21.5.",
        "the convention systematic is measured on the TRUTH-FOLD (predicted "
        "counts at the pack's own truth), NOT on a posterior. Propagating it to "
        "f(N) assumes a multiplicative mu error maps to the inverse "
        "multiplicative f error, exact only for a diagonal kernel.",
        "all three mocks share ONE injected f(N), so the cross-mock spread here "
        "bounds pipeline/realisation differences, NOT population-model error.",
        "the closure numbers fold the pack's OWN truth. They are a statement "
        "about the forward model, not about the estimator's coverage; coverage "
        "is the separate block and it runs at REDUCED scale.",
        "the 0.2-dex basis is a first-class OPTION, not the default. The shipped "
        "default is still 0.1 dex, because decision 3 was conditioned on "
        "re-running closure and closure still fails.",
        "🔴 GATE AUTHORITY IS NOT UNIFORM. Exactly ONE of the seven "
        "run_posterior.GATE tolerances is ratified: chi2_dof_max (chi2/dof "
        "<= 3, PI decision 8). The two ratio-span tolerances are UNRATIFIED -- "
        "the PI was asked and DECLINED -- and are not used by this artifact's "
        "verdict. The four |z| arms are RESTATED_NOT_RATIFIED: they DO refuse "
        "work and no deciding authority ratified them (decision 8 called "
        "|z| <= 5 MALFORMED AS STATED and sent it back for restatement, which "
        "is not a ratification). Read verdict.gate_authority before quoting "
        "any tolerance as authorised, and "
        "verdict.verdict_rests_on_the_ratified_arm_alone before treating this "
        "artifact's 'NO' as authority-dependent.",
        "the coverage block's SBC is matched on the LATENT BASIS WIDTH and the "
        "PAD only. It is reduced in grid extent, sampler scale, prior width and "
        "response realisation (R1-R5), and its measured power has a stated blind "
        "spot: read coverage.detection_envelope before quoting coverage.detection"
        "_curve['1']. A uniform rank histogram at this power does NOT certify the "
        "band to better than the largest UNDETECTED mis-scaling.",
    ]


# ---------------------------------------------------------------------------
def main(argv=None):
    global PACKDIR, OUT, N_SBC_SIMS, SBC_SEED
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["extract", "closure"])
    p.add_argument("--pack-dir", default=DEF_PACKDIR)
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--sbc-sims", type=int, default=N_SBC_SIMS,
                   help="matched-configuration SBC replicas (0 = skip the "
                        "coverage block). ~6 s each after JIT warmup, MEASURED.")
    p.add_argument("--sbc-seed", type=int, default=SBC_SEED)
    a = p.parse_args(argv)
    PACKDIR = a.pack_dir
    OUT = a.out
    N_SBC_SIMS = a.sbc_sims
    SBC_SEED = a.sbc_seed
    if a.phase == "extract":
        return phase_extract()
    return phase_closure()


if __name__ == "__main__":
    main()
