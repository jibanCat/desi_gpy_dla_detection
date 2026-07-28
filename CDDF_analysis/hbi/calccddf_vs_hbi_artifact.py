# -*- coding: utf-8 -*-
"""Aggregate the three per-mock FF-B closure JSONs into the stamped artifact
``CDDF_analysis/hbi/calccddf_vs_hbi.json``.  MOCK-ONLY, public-OK.

This is THE aggregation entry point for the feed-forward (FF) arm.  The
three-mock closure itself ran at full scale on 2026-07-11 (2LPT-0 1150 files /
London-0 1149 / Saclay-0 1127, 0 skipped); only this reduction — under one
second of compute — had never been invoked, which is why no
``calccddf_vs_hbi.json`` existed.

WHAT THIS ARTIFACT CONTAINS (and deliberately does NOT)
-------------------------------------------------------
* dN/dX (cumulative, per limit) and the differential f(N_HI).  **NO Omega.**
  Omega built from ``f_truth``/``omega`` is B16-contaminated (the truth f(N) is
  assembled with NO z-mask while dX IS masked), so every Omega ratio from these
  inputs is biased.  dN/dX is clean.  The Omega blocks present in the input
  JSONs are dropped here on purpose and the drop is recorded in
  ``metadata['omega_excluded']``.
* Per-mock ROLE.  2LPT-0 is the ON-MOCK CALIBRATION / RECOVERY FLOOR, not a
  held-out validation leg — the GP null model's trainset records mock-0/loa-124,
  ~21.96% of its catalog's sightlines are literally training spectra, and the
  forward kernel / molly C-P matrix / occupancy g(N,z) / loa-0 FP product are
  byte-identical 2LPT-0 strings for EVERY mock.  London-0 (jura-124) and
  Saclay-0 (juraLy8-124) are the genuine transfer tests.  This is stamped into
  ``mocks[<m>]['role']`` and ``metadata['leg_roles']``, not left in a comment.
* An UNCERTAINTY on the FF point (the FF arm previously emitted none) — see
  ``metadata['uncertainty']``.  It is a SAMPLING interval on a PLUG-IN
  estimator, NOT a posterior credible interval.
* ``metadata['estimand']`` — a statement of what the FF number IS, so it can
  never be silently compared against an HBI posterior median as if the two were
  the same object.

Rederive
--------
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    /home/mfho/.conda/envs/gpdla/bin/python \\
        CDDF_analysis/hbi/calccddf_vs_hbi_artifact.py \\
        --in 2lpt0=CDDF_analysis/hbi/calccddf_2lpt0_closure.json \\
             london0=CDDF_analysis/hbi/calccddf_london0_closure.json \\
             saclay0=CDDF_analysis/hbi/calccddf_saclay0_closure.json
"""
import os
import sys
import json
import hashlib
import argparse
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from CDDF_analysis.calc_cddf import DLACatalogue  # noqa: E402

# --------------------------------------------------------------------------- #
# PROCESS-START git stamp.  Captured at import, BEFORE anything can touch the
# tree, so the stamp describes the code that actually produced the numbers.
# 40-char SHA; never a movable tag name.
# --------------------------------------------------------------------------- #
def _git(*args):
    return subprocess.check_output(["git", "-C", HERE] + list(args)).decode().strip()


DEP_PATHS = [
    "CDDF_analysis/hbi/calccddf_vs_hbi_artifact.py",   # this routine
    "CDDF_analysis/hbi/calccddf_vs_hbi.py",            # the producer of the inputs
    "CDDF_analysis/calc_cddf.py",                      # the estimator + CI primitive
    "CDDF_analysis/cddf_forward/window.py",            # the search-window spec
]


def _capture_code_commit():
    """(40-char HEAD, repo-dirty, deps-dirty).

    ``repo_dirty`` covers the WHOLE worktree, which other concurrent agents may
    have dirty for unrelated reasons.  ``deps_dirty`` is the one that matters for
    re-derivability: it is scoped to the files this artifact's numbers actually
    depend on.
    """
    try:
        sha = _git("rev-parse", "HEAD")
        assert len(sha) == 40, sha
        repo_dirty = subprocess.call(["git", "-C", HERE, "diff", "--quiet", "HEAD"]) != 0
        deps_dirty = subprocess.call(
            ["git", "-C", HERE, "diff", "--quiet", "HEAD", "--"] + DEP_PATHS) != 0
        return sha, repo_dirty, deps_dirty
    except Exception as exc:  # pragma: no cover - only outside a git tree
        return "unknown ({}: {})".format(type(exc).__name__, exc), True, True


CODE_COMMIT_AT_START, DIRTY_AT_START, DEPS_DIRTY_AT_START = _capture_code_commit()

# --------------------------------------------------------------------------- #
# HBI reference.  The forward-response ("right object") kernel artifact lives on
# the OTHER branch, lls-subdla-cddf.  RESOLUTION: this file and that branch share
# ONE object store (git worktrees off /home/mfho/desi_gpy_dla_detection/.git), so
# the artifact is read CONTENT-ADDRESSED via ``git show <ref>`` — no checkout, no
# copy into this branch, no hard-coded numbers, and the exact blob SHA is
# recorded in the output.  The RETIRED kappa-kernel artifact
# (subdla_mock_validation.json, metadata.retired=True) must never be the source.
# --------------------------------------------------------------------------- #
HBI_FORWARD_BRANCH = "lls-subdla-cddf"
HBI_FORWARD_PATH = "CDDF_analysis/hbi/subdla_mock_validation_forward.json"

# Which mock the HBI forward artifact is a recovery ratio FOR.
HBI_FORWARD_MOCK = "2lpt0"

MOCK_ROLES = {
    "2lpt0": dict(
        role="ON-MOCK CALIBRATION / RECOVERY FLOOR",
        held_out=False,
        why="2LPT-0 (= loa-124) is the calibration set at four levels: the GP null "
            "model's trainset.h5 records mock-0/loa-124; ~21.96% of its catalog's "
            "sightlines are literally training spectra; and the forward kernel, the "
            "molly completeness/purity matrix, the occupancy g(N,z) and the loa-0 FP "
            "product are byte-identical 2LPT-0 strings for EVERY mock. Quote it as a "
            "recovery FLOOR, never as an independent validation.",
    ),
    "london0": dict(
        role="HELD-OUT TRANSFER",
        held_out=True,
        why="London-0 (jura-124): LyaCoLoRe on a CoLoRe lognormal density field — a "
            "different field recipe from the calibration mock. Genuine transfer test.",
    ),
    "saclay0": dict(
        role="HELD-OUT TRANSFER",
        held_out=True,
        why="Saclay-0 (juraLy8-124): Gaussian field + FGPA, deeper Lyman series — a "
            "different field recipe from the calibration mock. Genuine transfer test.",
    ),
}

LIMIT_KEYS = ["20.3", "20.0", "19.5", "band_195_203"]
N_EDGES = np.round(np.arange(17.2, 22.40001, 0.1), 3)


# --------------------------------------------------------------------------- #
# FF uncertainty
# --------------------------------------------------------------------------- #
def _pb_ref():
    """A bare ``DLACatalogue`` used ONLY as a namespace for the CI-combine.

    ``_count_ci_from_probs_poissons`` (calc_cddf.py) and its helper
    ``_get_combined_levels`` touch no instance state, so we do NOT open an HDF5
    file for them (``loa_literal_calccddf.py`` opens ``files[0]`` purely because
    it already has one to hand).  Same primitive, no I/O.
    """
    return DLACatalogue.__new__(DLACatalogue)


def _poisson_limit_ci(totals):
    """68/95 count intervals for a list of ACCUMULATED expected counts.

    Reuses calc_cddf's OWN Poisson-binomial + Poisson CI-combine
    (``_count_ci_from_probs_poissons``) — the very primitive FF-A
    (``CDDF_analysis/loa_literal_calccddf.py``) calls once on its totals — with
    the per-bin large-p ``probs`` lists EMPTY and the whole accumulated mass
    handed to the Poisson (small-p) channel.

    Why the Poisson limit rather than the exact Poisson-binomial: the additive
    ``(probs, poissons)`` seam IS the right object, but the 2026-07-11 closure
    run collapsed it to mean counts inside ``_counts_dx`` before writing, so the
    per-detection large-p list is not in these artifacts.  Re-running to recover
    it is ~4.5 CPU-hours, which is not warranted for an error bar, because the
    Poisson limit is CONSERVATIVE by construction:

        Var[Poisson-binomial] = sum p_i (1 - p_i)  <=  sum p_i = Var[Poisson]

    with equality only as all p_i -> 0.  So this interval is an UPPER BOUND on
    the true FF sampling width; it over-covers, never under-covers.
    ``calccddf_vs_hbi.py --ci-seam`` persists the exact seam for future runs.
    """
    ref = _pb_ref()
    totals = [float(t) for t in totals]
    ml, l68, l95 = ref._count_ci_from_probs_poissons([[] for _ in totals], totals)
    return ([float(x) for x in ml],
            [[float(a), float(b)] for a, b in l68],
            [[float(a), float(b)] for a, b in l95])


def _cumulative_counts(counts_N, N_cent):
    """Accumulated expected counts for each report limit (dN/dX numerators)."""
    counts_N = np.asarray(counts_N, float)
    N_cent = np.asarray(N_cent, float)
    out = {}
    for lim in (20.3, 20.0, 19.5):
        out[str(lim)] = float(counts_N[N_cent >= lim - 1e-9].sum())
    band = (N_cent >= 19.5 - 1e-9) & (N_cent < 20.3 - 1e-9)
    out["band_195_203"] = float(counts_N[band].sum())
    return out


# --------------------------------------------------------------------------- #
# HBI forward reference
# --------------------------------------------------------------------------- #
def load_hbi_forward():
    """Read + VALIDATE the forward-kernel HBI artifact from the other branch.

    Fail-closed: the artifact must self-declare ``resp_kind == 'forward'`` and
    must NOT carry ``metadata.retired``.  The retired kappa-kernel artifact
    would fail both assertions.
    """
    commit = _git("rev-parse", HBI_FORWARD_BRANCH)
    assert len(commit) == 40, commit
    ref = "{}:{}".format(commit, HBI_FORWARD_PATH)
    blob = _git("rev-parse", ref)
    raw = subprocess.check_output(["git", "-C", HERE, "show", ref])
    d = json.loads(raw.decode())
    md = d.get("metadata", {})

    if md.get("resp_kind") != "forward":
        raise RuntimeError(
            "HBI reference {} has resp_kind={!r}, expected 'forward'. The "
            "kappa/posterior-kernel artifact is RETIRED and must not be the "
            "source of any comparison point.".format(ref, md.get("resp_kind")))
    if "retired" in md:
        raise RuntimeError(
            "HBI reference {} carries metadata.retired={!r} (superseded_by={!r}); "
            "refusing to build a comparison on a retired artifact."
            .format(ref, md["retired"], md.get("superseded_by")))

    return d, dict(
        source="git show {}".format(ref),
        branch=HBI_FORWARD_BRANCH,
        branch_commit=commit,
        path=HBI_FORWARD_PATH,
        blob_sha=blob,
        artifact_code_commit=md.get("code_commit"),
        resp_kind=md.get("resp_kind"),
        retired_key_present=False,
        resolution_note=(
            "subdla_mock_validation_forward.json is committed on branch "
            "lls-subdla-cddf, not on this branch. Both branches are worktrees off "
            "the SAME object store (/home/mfho/desi_gpy_dla_detection/.git), so it "
            "is read content-addressed via `git show <40-char-commit>:<path>`. "
            "Nothing is copied or hard-coded; the blob SHA above pins the exact "
            "bytes that were read."),
    )


def hbi_forward_block(d):
    """dN/dX-only view of the forward HBI artifact (Omega dropped: B16)."""
    integ = d["integrated"]["loa0"]
    per_bin = [dict(blo=b["blo"], bhi=b["bhi"],
                    dndx_est=b["dndx_est"], dndx_tru=b["dndx_tru"],
                    f_est=b["f_est"], f_tru=b["f_tru"], r0=b["r0"])
               for b in d["per_bin"]["loa0"]]
    return dict(
        fp_estimator="loa0",
        R0_dndx={
            "20.3": integ["r0_dndx_203"],
            "20.0": integ["r0_dndx_200"],
            "19.5": None,   # cumulative >=19.5 is derived in build() (band + >=20.3)
            "band_195_203": integ["r0_dndx_195_203"],
        },
        dndx_est={"20.3": integ["dndx_est_203"],
                  "band_195_203": integ["dndx_est_195_203"]},
        dndx_tru={"20.3": integ["dndx_tru_203"],
                  "band_195_203": integ["dndx_tru_195_203"]},
        per_bin_dndx=per_bin,
        n_sightlines=integ["n_sl"],
    )


def _classify_input(doc):
    """Run the repo's OWN provenance guard over one closure artifact.

    KNOWN DEFECT recorded here rather than papered over: the six FF closure
    artifacts stamp under a TOP-LEVEL ``provenance/`` key, while
    ``CDDF_analysis/unblind/provenance.py::_load_metadata`` reads ``metadata/``
    or the bare top level -- so the guard returns NOT_STAMPED on them even though
    they carry a perfectly good stamp.  We therefore classify the ``provenance``
    block EXPLICITLY (``classify`` takes a metadata dict) and record BOTH the
    real verdict and the fact that the default loader would have missed it.
    """
    prov = doc.get("provenance", {})
    out = dict(stamp_key_used="provenance (top level)",
               default_loader_would_find_it=False,
               loader_defect=("unblind/provenance.py::_load_metadata reads 'metadata/' or "
                              "the bare top level; these artifacts stamp under "
                              "'provenance/', so the guard reports NOT_STAMPED unless the "
                              "block is passed explicitly, as here."))
    try:
        from CDDF_analysis.unblind import provenance as _P
        res = _P.classify(prov, routine_path=prov.get("routine"), repo=REPO)
        out.update(status=res.status,
                   stamp_kind=res.stamp_kind,
                   commit_exists=res.commit_exists,
                   contains_routine=res.contains_routine,
                   is_ancestor=res.is_ancestor,
                   routine_drift=res.routine_drift)
        if res.routine_drift:
            out["routine_drift_note"] = (
                "calccddf_vs_hbi.py has changed since this closure was stamped. Both "
                "known changes are PURELY ADDITIVE opt-ins that leave the default code "
                "path untouched: (a) --splits (the C4 per-z / SNR>4 strata, added after "
                "2d013e2), and (b) --ci-seam (the additive Poisson-binomial accumulator, "
                "2026-07-28). Neither alters the 'full'-tag counts these numbers come "
                "from. This is an ASSERTION TO CHECK, not a proof: run "
                "`git diff <stamped_code_commit> HEAD -- "
                "CDDF_analysis/hbi/calccddf_vs_hbi.py` before quoting.")
    except Exception as exc:
        out.update(status="UNCLASSIFIED",
                   error="{}: {}".format(type(exc).__name__, exc))
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tracked(path):
    # NOTE: pass the ABSOLUTE path. ``git -C HERE`` runs in CDDF_analysis/hbi, so
    # a REPO-relative pathspec would not resolve there.
    return subprocess.call(
        ["git", "-C", HERE, "ls-files", "--error-unmatch", os.path.abspath(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


# --------------------------------------------------------------------------- #
def build(inputs, hbi=True):
    mocks = {}
    input_files = []
    total_wall = 0.0

    for m, jp in inputs:
        if not os.path.exists(jp):
            raise SystemExit("missing input: {}".format(jp))
        d = json.load(open(jp))
        if d.get("mock") != m:
            raise SystemExit("input {} declares mock={!r}, labelled {!r}"
                             .format(jp, d.get("mock"), m))
        if d.get("checkpoint"):
            raise SystemExit("{} is a mid-run CHECKPOINT (n_files {} of {}); refusing"
                             .format(jp, d["n_files"], d["n_files_total"]))
        if d.get("n_files_skipped", 0) != 0:
            raise SystemExit("{} skipped {} files; refusing"
                             .format(jp, d["n_files_skipped"]))
        if d.get("second", 0) != 0:
            raise SystemExit("{} has second={!r}: calc_cddf's multi-DLA increment "
                             "path is RETIRED/broken (b00e6e4)".format(jp, d["second"]))

        prov = d.get("provenance", {})
        # Two of the three closures stamp an ABBREVIATED sha (project rule wants
        # 40 chars). Resolve it here so the aggregate is unambiguous, and record
        # whether the object is even reachable in this object store.
        stamped = str(prov.get("code_commit", ""))
        resolved, exists = None, False
        if stamped:
            try:
                resolved = _git("rev-parse", stamped + "^{commit}")
                exists = len(resolved) == 40
            except Exception:
                resolved, exists = None, False
        rel = os.path.relpath(os.path.abspath(jp), REPO)
        input_files.append(dict(
            mock=m, path=rel, sha256=_sha256(jp), git_tracked=_tracked(jp),
            stamped_code_commit=prov.get("code_commit"),
            stamped_code_commit_is_40char=len(stamped) == 40,
            stamped_code_commit_resolved=resolved,
            stamped_code_commit_exists=exists,
            stamped_date=prov.get("date"), stamped_routine=prov.get("routine"),
            stamped_rederive=prov.get("rederive"),
            provenance_classification=_classify_input(d),
            n_files=d["n_files"], n_files_total=d["n_files_total"],
            n_files_skipped=d["n_files_skipped"], n_sightlines=d["n_sightlines"],
        ))
        total_wall += float(d.get("wallclock_s", 0.0))

        N_cent = np.asarray(d["N_centers"], float)
        est_N = np.asarray(d["counts_calccddf_N"], float)
        tru_N = np.asarray(d["counts_truth_N"], float)
        dX = float(d["dX_total"])
        dN_lin = 10.0 ** N_EDGES[1:] - 10.0 ** N_EDGES[:-1]

        # ---- cumulative dN/dX + the FF sampling interval ------------------- #
        cum_est_counts = _cumulative_counts(est_N, N_cent)
        cum_tru_counts = _cumulative_counts(tru_N, N_cent)
        order = LIMIT_KEYS
        ml, l68, l95 = _poisson_limit_ci([cum_est_counts[k] for k in order])

        dndx_est, dndx_tru, r0, dndx68, dndx95 = {}, {}, {}, {}, {}
        for i, k in enumerate(order):
            dndx_est[k] = cum_est_counts[k] / dX
            dndx_tru[k] = cum_tru_counts[k] / dX
            r0[k] = (dndx_est[k] / dndx_tru[k]) if dndx_tru[k] else float("nan")
            dndx68[k] = [l68[i][0] / dX, l68[i][1] / dX]
            dndx95[k] = [l95[i][0] / dX, l95[i][1] / dX]

        # ---- differential f(N) + per-bin sampling interval ---------------- #
        # Prefer the EXACT Poisson-binomial seam when the closure was run with
        # --ci-seam; otherwise fall back to the conservative Poisson limit.
        seam = d.get("ci_seam")
        if seam:
            f68 = [list(map(float, x)) for x in seam["counts_68"]]
            f95 = [list(map(float, x)) for x in seam["counts_95"]]
            fN_ci_method = "exact Poisson-binomial seam (calccddf_vs_hbi.py --ci-seam)"
        else:
            _fml, f68, f95 = _poisson_limit_ci(est_N.tolist())
            fN_ci_method = ("Poisson limit of the Poisson-binomial (conservative upper "
                            "bound; the 2026-07-11 closure did not persist the seam)")
        fN_est = (est_N / dX / dN_lin)
        fN_tru = (tru_N / dX / dN_lin)
        fN68 = np.asarray(f68) / dX / np.vstack([dN_lin, dN_lin]).T
        fN95 = np.asarray(f95) / dX / np.vstack([dN_lin, dN_lin]).T

        mocks[m] = dict(
            role=MOCK_ROLES[m]["role"],
            held_out=MOCK_ROLES[m]["held_out"],
            role_why=MOCK_ROLES[m]["why"],
            n_files=d["n_files"], n_files_total=d["n_files_total"],
            n_files_skipped=d["n_files_skipped"], n_sightlines=d["n_sightlines"],
            dX_total=dX, grid=d["grid"], second=d["second"],
            z_range=d["z_range"], snr_min=d["snr_min"],
            truth_catalog=d["truth"],
            dndx=dict(
                estimand="feed-forward plug-in (see metadata['estimand'])",
                calccddf=dndx_est, truth=dndx_tru, R0_calccddf=r0,
                calccddf_counts=cum_est_counts, truth_counts=cum_tru_counts,
                calccddf_68=dndx68, calccddf_95=dndx95,
            ),
            fN=dict(
                ci_method=fN_ci_method,
                ci_is_credible_interval=False,
                N_centers=N_cent.tolist(), lnhi_edges=N_EDGES.tolist(),
                calccddf=fN_est.tolist(), truth=fN_tru.tolist(),
                calccddf_68_lo=fN68[:, 0].tolist(), calccddf_68_hi=fN68[:, 1].tolist(),
                calccddf_95_lo=fN95[:, 0].tolist(), calccddf_95_hi=fN95[:, 1].tolist(),
                counts_calccddf=est_N.tolist(), counts_truth=tru_N.tolist(),
            ),
            wallclock_s=float(d.get("wallclock_s", 0.0)),
        )

    # ---- HBI forward reference (2LPT-0 only; see note) --------------------- #
    hbi_block, hbi_prov = None, None
    if hbi:
        hd, hbi_prov = load_hbi_forward()
        blk = hbi_forward_block(hd)
        integ = hd["integrated"]["loa0"]
        # 19.5 cumulative R0 is not stored directly in the forward artifact's
        # integrated block; it is the sum of the band and the >=20.3 numerators
        # over the matching truth sum.
        num = integ["dndx_est_195_203"] + integ["dndx_est_203"]
        den = integ["dndx_tru_195_203"] + integ["dndx_tru_203"]
        blk["R0_dndx"]["19.5"] = num / den
        blk["dndx_est"]["19.5"] = num
        blk["dndx_tru"]["19.5"] = den
        hbi_block = {HBI_FORWARD_MOCK: blk}

    out = dict(
        metadata=dict(
            what="Feed-forward (FF-B) LITERAL calc_cddf closure aggregated across three "
                 "mocks, alongside the forward-kernel catalog-HBI reference. MOCK-ONLY, "
                 "public-OK: no real-LOA (loa main-dark) data was read, and every number "
                 "here is a mock recovery quantity.",
            routine="CDDF_analysis/hbi/calccddf_vs_hbi_artifact.py",
            code_commit=CODE_COMMIT_AT_START,
            code_commit_captured="process start (module import), before any file was read",
            code_commit_repo_dirty_at_start=DIRTY_AT_START,
            code_commit_deps_dirty_at_start=DEPS_DIRTY_AT_START,
            code_commit_deps=DEP_PATHS,
            code_commit_dirty_note=("repo_dirty covers the WHOLE worktree (other concurrent "
                                    "workflows may have unrelated files dirty); deps_dirty is "
                                    "scoped to code_commit_deps and is the flag that governs "
                                    "re-derivability of THESE numbers."),
            producing_run="CDDF_analysis/hbi/calccddf_vs_hbi.py --mock {2lpt0,london0,saclay0} "
                          "--second 0 (full scale, 2026-07-11)",
            rederive=(
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; "
                "/home/mfho/.conda/envs/gpdla/bin/python "
                "CDDF_analysis/hbi/calccddf_vs_hbi_artifact.py --in "
                "2lpt0=CDDF_analysis/hbi/calccddf_2lpt0_closure.json "
                "london0=CDDF_analysis/hbi/calccddf_london0_closure.json "
                "saclay0=CDDF_analysis/hbi/calccddf_saclay0_closure.json "
                "(reduction only, <1 s; the per-mock closures themselves are the "
                "2026-07-11 full-scale runs recorded in input_files[])"),
            input_files=input_files,
            estimand=dict(
                ff=("POSTERIOR-WEIGHTED PLUG-IN CDDF. For every sightline the DLA(1) "
                    "model's per-sample posterior over (logN, z) is deposited into "
                    "(logN, z) bins weighted by P(DLA|data); the bin total is the "
                    "expected absorber count, divided by the summed absorption path dX. "
                    "It is a SINGLE point-valued functional of the catalog posterior "
                    "(Bird 2017 recipe) — a plug-in, NOT a draw from any posterior over "
                    "f(N). In the paper's FF arm this point is subsequently multiplied "
                    "by a naive mock correction alpha measured on mocks; alpha is NOT "
                    "applied in this artifact (the numbers here are the RAW plug-in, so "
                    "R0 = est/truth here IS the quantity alpha is fitted to invert)."),
                hbi=("POSTERIOR quantity from a hierarchical Bayesian inference over the "
                     "catalog with a measured forward-response (Eddington) kernel, a "
                     "completeness/purity surface and an FP model. The numbers carried "
                     "here from the forward artifact are its point estimate."),
                matching_rule=("THESE TWO ARE DIFFERENT ESTIMANDS. A plug-in point and a "
                               "posterior median/point are not the same object. Never "
                               "plot or tabulate them on one axis without labelling "
                               "which is which, and never present the FF interval below "
                               "as if it were a posterior credible interval."),
                detected_space=("R0 != 1 is EXPECTED for FF: the FF number lives in "
                                "DETECTED space (no completeness/purity/FP correction "
                                "at all), while truth is the injected catalog. The "
                                "N-dependence of R0 is the (C, K, FP) fingerprint."),
                multi_dla=("second=0. The DLA(1)-model N-shape carries the TOTAL DLA "
                           "posterior once per sightline; ~7-8% of injected DLAs are "
                           "the 2nd/3rd in a sightline and are NOT separately counted, "
                           "so FF R0 is ~7% conservative (LOW) at the DLA tier. "
                           "calc_cddf's multi-DLA increment path is RETIRED and has "
                           "been numerically broken since b00e6e4 (2020-03-31) — see "
                           "the MODULE STATUS block in CDDF_analysis/calc_cddf.py."),
            ),
            uncertainty=dict(
                what="68% and 95% intervals on the FF dN/dX and f(N).",
                kind="SAMPLING INTERVAL ON A PLUG-IN ESTIMATOR",
                is_credible_interval=False,
                honest_label=("This is a SAMPLING interval on a PLUG-IN estimator, NOT a "
                              "posterior credible interval. It propagates ONLY the "
                              "counting (Poisson-binomial) scatter of the deposited "
                              "posterior mass. It contains NO uncertainty from the GP "
                              "model, the completeness/purity surface, the FP model, the "
                              "mock correction alpha, or the choice of mock. It must "
                              "never be compared to, or drawn on the same axis as, an "
                              "HBI credible band without that label."),
                method=("calc_cddf's own Poisson-binomial + Poisson CI-combine "
                        "(DLACatalogue._count_ci_from_probs_poissons), called ONCE on "
                        "the accumulated totals — the same primitive FF-A "
                        "(CDDF_analysis/loa_literal_calccddf.py) uses. The seam is "
                        "additive over sightlines, so combining on the totals equals "
                        "combining on one merged file."),
                poisson_limit=("The 2026-07-11 closure collapsed the (probs, poissons) "
                               "seam to mean counts before writing, so the per-detection "
                               "large-p list is not available here and the whole mass is "
                               "handed to the Poisson channel. Var[Poisson-binomial] = "
                               "sum p(1-p) <= sum p = Var[Poisson], so this interval is "
                               "an UPPER BOUND on the true FF sampling width: it "
                               "OVER-covers, never under-covers. Run "
                               "calccddf_vs_hbi.py --ci-seam to persist the exact seam."),
                figure_caption=("FF band: 68% Poisson-limit sampling interval on the "
                                "plug-in estimator (counting scatter only; conservative "
                                "upper bound). NOT a posterior credible interval, and "
                                "not commensurate with the HBI band."),
            ),
            leg_roles=dict(
                calibration_floor=["2lpt0"],
                held_out_transfer=["london0", "saclay0"],
                statement=("2LPT-0 (= loa-124) is the ON-MOCK CALIBRATION / RECOVERY "
                           "FLOOR, NOT a held-out validation leg. Only London-0 and "
                           "Saclay-0 are transfers. Do not average the three legs into "
                           "a single 'three-mock validation' number as if they were "
                           "exchangeable."),
                detail={m: MOCK_ROLES[m]["why"] for m in MOCK_ROLES},
            ),
            omega_excluded=dict(
                excluded=True,
                reason=("B16: the truth f(N)/Omega in the input closures is built with "
                        "NO z-mask while dX IS masked, so every Omega ratio from these "
                        "inputs is biased. dN/dX is CLEAN. Omega is therefore dropped "
                        "from this artifact rather than carried with a caveat. The "
                        "leak is NOT a scalar rescale — Omega must be RE-DERIVED, "
                        "never rescaled."),
                dropped_keys=["cumulative.calccddf.omega", "cumulative.truth.omega",
                              "cumulative.R0_calccddf.omega", "splits.*.R0_omega"],
            ),
            conventions=dict(
                z_range=[2.0, 3.5],
                snr_cut="SNR_REDSIDE > 2.0 (matches HBI)",
                window=("Lya-only (WindowSpec z_min_lyb=True): blue edge = "
                        "lymanbeta(z_qso); no proximity/tail re-cut (stored "
                        "min/max_z_dla already encode it)"),
                p_dla="posterior-weighted, NO hard P_DLA cut (HBI uses P_DLA>0.99)",
                nan_handling=("DESI processed files store NaN for negligible/invalid DLA "
                              "posteriors & samples; NaN->0 (posteriors, renormalized) / "
                              "NaN->-inf (samples). Literal calc_cddf does NOT do this on "
                              "its DLA(1) path and does not run out-of-the-box on these "
                              "files; NanSafeDLACatalogue in calccddf_vs_hbi.py adds it "
                              "with fail-closed writer-convention guards."),
                filter=("FILTER_LOW_LIKELIHOOD=1 production files — what real production "
                        "actually runs."),
                truth=("injected HCD truth catalog, windowed IDENTICALLY to the estimator "
                       "(same Lyb edge, same stored [min_z_dla,max_z_dla], same SNR>2 "
                       "sightline set, same dX)."),
            ),
            z_resolved_warning=("Per-z FF/HBI splits are deliberately NOT reduced here. The "
                                "estimator manufactures a residual z-tilt (R0(dN/dX>=20.3) "
                                "0.908 -> 1.052 -> 1.189 across z in [2.0,3.5), 0/3 z-bins "
                                "covered at 95% against a ~2% statistical half-width). "
                                "Report z-MARGINALISED (integrated) results unless the tilt "
                                "is separately controlled. The per-z counts live in "
                                "calccddf_{mock}_splits.json."),
            hbi_reference=hbi_prov,
            hbi_coverage=("HBI forward-kernel points are carried for 2LPT-0 ONLY. The "
                          "London-0 / Saclay-0 HBI transfer legs exist only in the "
                          "UNTRACKED crossmock_transfer_loa0.json, whose own "
                          "provenance_note admits there is no committed aggregator "
                          "(build_artifact_loa0.py was never written) and whose legs are "
                          "stamped '-dirty'. Under this project's headline rule those are "
                          "NOT QUOTABLE, so they are NOT hard-coded here. They become "
                          "available the moment a committed aggregator stamps them."),
            wallclock_s_inputs=total_wall,
        ),
        mocks=mocks,
        hbi_forward=hbi_block,
    )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="mock=json ...")
    ap.add_argument("--out", default=os.path.join(HERE, "calccddf_vs_hbi.json"))
    ap.add_argument("--no-hbi", action="store_true",
                    help="skip the cross-branch HBI forward reference")
    args = ap.parse_args(argv)

    inputs = [tuple(s.split("=", 1)) for s in args.inp]
    out = build(inputs, hbi=not args.no_hbi)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)
    print("code_commit (process start):", CODE_COMMIT_AT_START,
          "| deps_dirty=" + str(DEPS_DIRTY_AT_START),
          "| repo_dirty=" + str(DIRTY_AT_START))

    for m, d in out["mocks"].items():
        print("\n=== {}  [{}]  nfiles={} nSL={} dX={:.0f} ==="
              .format(m, d["role"], d["n_files"], d["n_sightlines"], d["dX_total"]))
        hb = (out["hbi_forward"] or {}).get(m)
        for lim in LIMIT_KEYS:
            rc = d["dndx"]["R0_calccddf"][lim]
            lo, hi = d["dndx"]["calccddf_68"][lim]
            tru = d["dndx"]["truth"][lim]
            band = " [68% {:.4f},{:.4f}] R0in68=[{:.3f},{:.3f}]".format(
                lo, hi, lo / tru, hi / tru) if tru else ""
            rh = "   HBI(fwd) R0={:.3f}".format(hb["R0_dndx"][lim]) if hb else "   HBI n/a"
            print("  dNdX {:>12}: FF R0={:.4f}{}{}".format(lim, rc, band, rh))
    print("\nFF interval = SAMPLING interval on a PLUG-IN estimator "
          "(Poisson-limit, conservative). NOT a posterior credible interval.")
    return out


if __name__ == "__main__":
    main()
