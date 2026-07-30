# -*- coding: utf-8 -*-
"""window_study.py — the MATCHED SPECTRAL-WINDOW study (PI decision 2, 2026-07-29).

Produces ``CDDF_analysis/hbi_mcmc/spectral_window_study.json``.

THE TWO DIFFERENT "WINDOWS" — do not conflate them
--------------------------------------------------
(a) FINDER FITTING window — ``gpy_dla_detection/set_parameters.py``
    ``Parameters.min_lambda`` (default **911.75**) / ``max_lambda`` (library
    default 1216.75; **production overrides it to 1250**, see
    ``slurm/greatlakes/production/london0_gl_v1.env:77`` on top of
    ``slurm/configs/_base.env:31-32``). Production passes ``--min_lambda
    "$MIN_LAMBDA"`` through ``desi-DLAGP.py``
    (``submit_desi_mock_gl.sh:181``); ``MIN_LAMBDA`` is exported by
    ``launch_gl.sh:137``. This is the region the GP actually models, so moving
    it changes the LIKELIHOOD and requires **re-running the finder**. ARM 2
    below only PILOTS it.

(b) ANALYSIS window — ``HBIConfig.lam_rf_min``, applied POST-HOC to decide which
    absorbers and which pathlength count. This is what the existing
    ``lya_only`` (1025 A) vs ``lya_lyb`` (911 A) products already vary, and it
    costs no inference. ARM 1 measures it completely.

ARM 1 — the complete matched ANALYSIS-window comparison
-------------------------------------------------------
Both windows, all three mocks, ONE configuration, IDENTICAL absorber
selections / truth definitions / estimands / gates. Every window-DEPENDENT
calibration ingredient is rebuilt per window (registry:
``extract_pack.ANALYSIS_WINDOWS``):

  ingredient            lya_only              lya_lyb
  --------------------  --------------------  --------------------------------
  lam_rf_min            1025.0                911.0
  completeness matrix   figures_molly_nhi195/ figures_molly_nhi195/lya_lyb/
                        lya_only/             (both already on disk; each
                                              stamps its own lam_rf_min)
  sub-floor matrix      figures_molly_nhi172/ figures_molly_nhi172_bywindow/
                        (lam_rf_min 1025)     lya_lyb/  <- BUILT for this study
  forward response      forward_response_     forward_response_2lpt0_lam911
                        2lpt0.npz             .npz     <- BUILT for this study
  loa-0 FP background   product cut @1025     raw dlacat re-cut @911
  dX / counts / truth   from cfg.lam_rf_min   from cfg.lam_rf_min
  g(N, z)               from the bundle       from the bundle

  NOT window-matched (stated limit): ``t_sigma`` — the cross-mock transfer
  PRIOR WIDTHS read from ``CDDF_analysis/hbi/ff_fp_{mock}.json``, built at
  lam_rf_min=1025. They are identical in both arms, so they cannot DRIVE a
  window difference, but they are not re-measured either.

COMPARISON PROTOCOL — FIXED BEFORE ANY OUTCOME WAS LOOKED AT
-------------------------------------------------------------
Written into the artifact as ``protocol`` and reproduced here verbatim:

  P1. BOTH windows are reported in full, whatever the outcome. The nominal
      Molly window (``lya_only``, 1025 A) is the STANDARD REFERENCE and is
      retained and reported unconditionally.
  P2. The primary discriminator is chi2/dof over the PRIMARY REPORTING WINDOW
      19.7 <= log NHI <= 21.6 (PI decision 1), on the pooled Poisson
      standardized residuals defined in ``restated_gate_criteria()``.
  P3. The SECONDARY discriminator, decided in advance because D2 is the
      binding constraint, is the HIGH-N RESIDUAL mu/obs over log NHI >= 21.6 —
      the region the PI's reporting window deliberately EXCLUDES. A window is
      only preferred on this axis if it moves the high-N residual TOWARD 1.
  P4. A window is NOT preferred merely for a better total mu/obs. The total is
      a level; the residual SHAPE is the defect (established by the D1 ladder:
      total 0.9973 with chi2/dof 51.5).
  P5. Direction-of-effect must AGREE ON ALL THREE MOCKS. A 2-of-3 split is
      reported as no effect.
  P6. No configuration is dropped, re-run, or re-tuned after its number is
      seen. The full cross is emitted.

ARM 2 — the FITTING-window pilot (POINTER, NOT A MEASUREMENT)
-------------------------------------------------------------
``--phase pilot`` runs the finder twice on the SAME small set of spectra, at
``--min_lambda 911.75`` and at ``--min_lambda 1025.0``, everything else
identical, and reports (i) the per-spectrum cost and (ii) the paired change in
recovered NHI for large DLAs. A pilot of tens of spectra CANNOT settle the
misfit hypothesis; it can only point, and the artifact says so with a
quantified uncertainty.

USAGE (two envs — the extractor is jax-free by design, the fold needs jax)
-------------------------------------------------------------------------
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla python CDDF_analysis/hbi_mcmc/window_study.py \
        --phase extract --pack-dir <SCRATCH>

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.window_study \
        --phase selftest --pack-dir <SCRATCH> \
        --out CDDF_analysis/hbi_mcmc/spectral_window_study.json

MOCKS ONLY. No real-LOA path is touched; no real-data value can enter this
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

DEF_PACKDIR = os.environ.get(
    "WINDOW_STUDY_PACK_DIR",
    "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/window_study/packs")
DEF_OUT = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/spectral_window_study.json")

MOCKS = ["2lpt0", "london0", "saclay0"]
WINDOWS = ["lya_only", "lya_lyb"]

# --- the configuration the PI adopted (decisions 1, 3, 4) -------------------
PAD_FLOOR = 19.0                    # decision 4: pad floor 19.0 ...
COMPLETENESS = "molly172"           # ... with the molly172 convention
CLAMPS = ["both", "hi"]             # the D2 covariate-clamp bracket (undecided)
# decision 1: the PRIMARY column-density reporting window
REPORT_LO, REPORT_HI = 19.7, 21.6
# decision 1 / finding D2: the excluded high-N tail, reported SEPARATELY
HIGHN_LO = 21.6
# decision 3: 0.2-dex latent basis. NOT IMPLEMENTED HERE -- see
# BASIS_RESOLUTION_STATUS.
BASIS_DEX = 0.1
BASIS_RESOLUTION_STATUS = (
    "PI decision 3 (a 0.2-dex latent true-N basis) is NOT implemented in this "
    "study. It is owned by a sibling stream on another branch and was not "
    "available. Reason it was not implemented locally: a 0.2-dex ntrue grid "
    "violates TWO committed schema rules in "
    "CDDF_analysis/hbi_mcmc/pack.py:validate_pack -- "
    "_check_edges_uniform('ntrue_edges', ..., _N_STEP=0.1) and the rule that "
    "nhat_edges be an exact TAIL SUBSET of ntrue_edges -- and relaxing them "
    "would change the shared pack schema the sibling stream is editing. "
    "CONSEQUENCE FOR THIS STUDY: the window contrast is measured at FIXED "
    "0.1-dex basis on BOTH arms, so it is a valid matched comparison and the "
    "DIRECTION of the window effect is unaffected by basis resolution. The "
    "ABSOLUTE closure numbers WILL move when the 0.2-dex basis lands, so no "
    "absolute closure figure here may be quoted as the PI-adopted "
    "configuration's closure.")

# module-level, rebound by main()
PACKDIR = DEF_PACKDIR
OUT = DEF_OUT

_MU_FLOOR = 1e-12         # the committed ratio_tables Poisson-denominator floor


# ---------------------------------------------------------------------------
# THE COMPARISON PROTOCOL (fixed before any outcome was inspected)
# ---------------------------------------------------------------------------
PROTOCOL = [
    ("P1", "BOTH windows are reported in full, whatever the outcome. The "
           "nominal Molly window (lya_only, 1025 A) is the STANDARD REFERENCE "
           "and is retained and reported unconditionally."),
    ("P2", "PRIMARY discriminator: chi2/dof over the PRIMARY REPORTING WINDOW "
           f"{REPORT_LO} <= log NHI <= {REPORT_HI} (PI decision 1), on the "
           "Poisson standardized residuals defined in "
           "restated_gate_criteria()."),
    ("P3", "SECONDARY discriminator, fixed in advance because D2 is the "
           f"binding constraint: the HIGH-N residual mu/obs over log NHI >= "
           f"{HIGHN_LO}. A window is preferred on this axis ONLY if it moves "
           "that residual TOWARD 1."),
    ("P4", "A window is NOT preferred merely for a better TOTAL mu/obs. The "
           "total is a level; the residual SHAPE is the defect (D1 ladder: "
           "total 0.9973 at chi2/dof 51.5)."),
    ("P5", "Direction of effect must AGREE ON ALL THREE MOCKS. A 2-of-3 split "
           "is reported as no effect."),
    ("P6", "No configuration is dropped, re-run, or re-tuned after its number "
           "is seen. The full cross (2 windows x 3 mocks x 2 clamps) is "
           "emitted."),
]


def restated_gate_criteria():
    """PI decision 8: the EXACT mathematical definition of the closure gate.

    The brief's ``|z| <= 5`` was malformed -- it named neither the object z is
    computed on, nor which marginal, nor the denominator, nor the restriction
    to occupied bins. Restated:

    Let ``mu[c, k, s]`` be the folded expected counts and ``n[c, k, s]`` the
    observed counts, on the observed n-hat grid c, fine-z grid k, SNR strata s.
    Let ``M = {(k, s) : dX[k, s] > 0}`` be the OCCUPIED calibration cells (the
    op cut is SNR > 2 strict, so strata 0 and 1 carry exactly zero pathlength
    and are excluded by construction, not by choice).

    Per-bin marginals over a reporting window W = [lo, hi], summing only over M:

        mu_c = sum_{(k,s) in M} mu[c,k,s]        n_c = sum_{(k,s) in M} n[c,k,s]

    and the bin index set

        B(W) = { c : nhat_lo[c] >= lo - eps  AND  nhat_hi[c] <= hi + eps }

    (fully-contained bins only; eps = 1e-9). The Poisson standardized residual

        z_c = (n_c - mu_c) / sqrt(max(mu_c, 1e-12))

    -- signed, observed MINUS expected, denominator sqrt(EXPECTED) (the
    variance under the model), floored at 1e-12 so an empty prediction cannot
    divide by zero. Then, over the OCCUPIED subset B+(W) = {c in B(W) : n_c > 0}:

        z_total(W)  = (sum_{c in B(W)} n_c - sum_{c in B(W)} mu_c)
                      / sqrt(max(sum_{c in B(W)} mu_c, 1e-12))
        z_bin_max(W)= max_{c in B+(W)} |z_c|
        chi2_dof(W) = ( sum_{c in B+(W)} z_c^2 ) / |B+(W)|

    THE THREE RATIFIED ARMS (PI decision 8) are

        |z_total(W)| <= 5      z_bin_max(W) <= 5      chi2_dof(W) <= 3

    NOTE the exclusion ``n_c > 0`` applies to z_bin_max and chi2_dof but NOT to
    z_total: an empty bin contributes to the summed totals but has no
    standardized residual of its own. This asymmetry is the committed
    ``run_posterior.forward_closure_gate`` behaviour and is stated, not fixed,
    here.

    NOT RATIFIED (PI decision 8, explicit): ``ratio_span_by_z_max = 0.10`` and
    ``ratio_span_by_snr_max = 0.15``. They are reported as MEASUREMENTS in this
    artifact (``ratio_span_by_z`` / ``ratio_span_by_snr``) and are NOT used to
    decide anything.
    """
    return dict(
        z_c=("z_c = (n_c - mu_c) / sqrt(max(mu_c, 1e-12)); signed, observed "
             "MINUS expected, denominator sqrt(EXPECTED)"),
        occupied_cells=("(k, s) with dX[k, s] > 0 only; SNR strata 0 and 1 "
                        "carry exactly zero pathlength under the SNR > 2 "
                        "strict op cut"),
        bin_set=("B(W) = bins FULLY CONTAINED in W: nhat_lo >= lo - 1e-9 AND "
                 "nhat_hi <= hi + 1e-9"),
        occupied_bin_set="B+(W) = {c in B(W) : n_c > 0}",
        z_total=("(sum_B n_c - sum_B mu_c) / sqrt(max(sum_B mu_c, 1e-12)) -- "
                 "over ALL of B(W), empty bins included"),
        z_bin_max="max over B+(W) of |z_c|",
        chi2_dof="sum over B+(W) of z_c^2, divided by |B+(W)|",
        ratified_arms={"abs_z_total_max": 5.0, "z_bin_max": 5.0,
                       "chi2_dof_max": 3.0},
        not_ratified={"ratio_span_by_z_max": 0.10,
                      "ratio_span_by_snr_max": 0.15},
        not_ratified_note=("PI decision 8 declined to ratify these two; they "
                           "are reported as measurements here and decide "
                           "nothing. They must be defined and calibrated "
                           "prospectively."),
        asymmetry_note=("n_c > 0 restricts z_bin_max and chi2_dof but NOT "
                        "z_total. This is the committed "
                        "run_posterior.forward_closure_gate behaviour, stated "
                        "rather than changed."),
    )


# ---------------------------------------------------------------------------
# PURE metric helpers (no I/O, no jax -- unit-tested in tests/test_window_study.py)
# ---------------------------------------------------------------------------
def select_bins(by_nhat, lo=None, hi=None, eps=1e-9):
    """Bins FULLY CONTAINED in [lo, hi]. ``None`` = unbounded on that side.

    Full containment, not midpoint membership: a 0.1-dex bin straddling the
    reporting edge carries counts from OUTSIDE the window, and mixing those in
    is the one-sided-support defect class this project has been burned by four
    times. ``hi`` therefore compares against the bin's UPPER edge.
    """
    out = []
    for b in by_nhat:
        if lo is not None and not (float(b["lo"]) >= lo - eps):
            continue
        if hi is not None and not (float(b["hi"]) <= hi + eps):
            continue
        out.append(b)
    return out


def window_metrics(by_nhat, lo=None, hi=None):
    """Closure metrics over the n-hat bins fully contained in [lo, hi].

    Arithmetic EXACTLY as restated in ``restated_gate_criteria()``. ``by_nhat``
    rows are the committed ``forward_selftest.ratio_tables`` rows, which already
    carry mu/obs summed over the dX > 0 cells only.
    """
    rows = select_bins(by_nhat, lo, hi)
    mu = float(sum(float(b["mu"]) for b in rows))
    obs = float(sum(float(b["obs"]) for b in rows))
    occ = [b for b in rows if float(b["obs"]) > 0]
    z = np.array([(float(b["obs"]) - float(b["mu"]))
                  / np.sqrt(max(float(b["mu"]), _MU_FLOOR)) for b in occ],
                 float)
    return dict(
        window=[lo, hi],
        n_bins=int(len(rows)),
        n_bins_occupied=int(len(occ)),
        mu=mu, obs=obs,
        ratio=(mu / obs if obs > 0 else float("nan")),
        z_total=float((obs - mu) / np.sqrt(max(mu, _MU_FLOOR))),
        z_bin_max=(float(np.abs(z).max()) if len(z) else float("nan")),
        chi2_dof=(float((z ** 2).sum() / len(z)) if len(z) else float("nan")),
        per_bin=[dict(lo=float(b["lo"]), hi=float(b["hi"]),
                      mu=float(b["mu"]), obs=float(b["obs"]),
                      ratio=(float(b["mu"]) / float(b["obs"])
                             if float(b["obs"]) > 0 else float("nan")),
                      z=float((float(b["obs"]) - float(b["mu"]))
                              / np.sqrt(max(float(b["mu"]), _MU_FLOOR))))
                 for b in rows],
    )


def closes(metrics, gate):
    """The three RATIFIED arms, on one ``window_metrics`` dict."""
    fails = []
    if not (abs(metrics["z_total"]) <= gate["abs_z_total_max"]):
        fails.append(f"|z_total|={abs(metrics['z_total']):.2f} > "
                     f"{gate['abs_z_total_max']}")
    zb = metrics["z_bin_max"]
    if np.isfinite(zb) and not (zb <= gate["z_bin_max"]):
        fails.append(f"z_bin_max={zb:.2f} > {gate['z_bin_max']}")
    c2 = metrics["chi2_dof"]
    if np.isfinite(c2) and not (c2 <= gate["chi2_dof_max"]):
        fails.append(f"chi2_dof={c2:.2f} > {gate['chi2_dof_max']}")
    return dict(closes=not fails, failures=fails, gate=dict(gate))


def marginal_block(rows):
    """|z|max + mu/obs span over a z- or SNR-marginal (MEASURED, never gating).

    The span is the UNRATIFIED tolerance's object (PI decision 8), reported so
    the number exists without being used to refuse anything.
    """
    occ = [r for r in (rows or []) if float(r.get("obs", 0.0)) > 0]
    zs = np.array([float(r["z"]) for r in occ], float)
    ratios = np.array([float(r["ratio"]) for r in occ], float)
    ratios = ratios[np.isfinite(ratios)]
    return dict(
        n_occupied=int(len(occ)),
        z_max=(float(np.abs(zs).max()) if len(zs) else float("nan")),
        ratio_min=(float(ratios.min()) if len(ratios) else float("nan")),
        ratio_max=(float(ratios.max()) if len(ratios) else float("nan")),
        ratio_span=(float(ratios.max() - ratios.min())
                    if len(ratios) >= 2 else 0.0),
        rows=[dict(lo=r.get("lo"), hi=r.get("hi"), s=r.get("s"),
                   mu=float(r["mu"]), obs=float(r["obs"]),
                   ratio=float(r["ratio"]), z=float(r["z"])) for r in occ],
    )


def direction_verdict(per_mock_delta, tol=0.0):
    """P5: an effect counts only if its SIGN agrees on all three mocks.

    ``per_mock_delta`` maps mock -> (lya_lyb value - lya_only value).
    Returns the unanimous sign, or 0 with ``unanimous=False``.
    """
    signs = {m: (1 if v > tol else (-1 if v < -tol else 0))
             for m, v in per_mock_delta.items()}
    uniq = set(signs.values())
    unanimous = len(uniq) == 1 and 0 not in uniq
    return dict(per_mock=dict(per_mock_delta), signs=signs,
                unanimous=bool(unanimous),
                sign=(next(iter(uniq)) if unanimous else 0),
                n_mocks=len(per_mock_delta))


# ---------------------------------------------------------------------------
# window matching guards (fail-closed on a window mismatch)
# ---------------------------------------------------------------------------
def read_molly_summary(tsv_path):
    """Parse a molly_summary.tsv's scalar metric block -> dict."""
    summary = os.path.join(os.path.dirname(tsv_path), "molly_summary.tsv")
    out = {}
    with open(summary) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            out[parts[0]] = parts[1]
    return out


def assert_window_matched(window):
    """FAIL CLOSED unless every window-dependent ingredient stamps THIS window.

    The failure this guards against is silent and severe: pairing a 911-A
    absorber/pathlength selection with a 1025-A completeness matrix or a 1025-A
    forward response gives a number that looks fine and means nothing. Each
    ingredient carries its OWN lam_rf_min stamp (molly_summary.tsv for the two
    matrices, a .window.json sidecar for the forward response), and this reads
    those stamps rather than trusting the registry.
    """
    EP = _extract_pack_module()
    w = EP.window_spec(window)
    lam = float(w["lam_rf_min"])
    checked = {}
    for key in ("molly_tsv", "molly_tsv_172"):
        path = w[key]
        if not os.path.exists(path):
            raise SystemExit(f"window {window!r}: {key} missing on disk: {path}")
        s = read_molly_summary(path)
        got = float(s["lam_rf_min"])
        if abs(got - lam) > 1e-9:
            raise SystemExit(
                f"WINDOW MISMATCH: window {window!r} has lam_rf_min={lam} but "
                f"{key}={path} stamps lam_rf_min={got} in its own "
                f"molly_summary.tsv. Refusing to build a mixed-window pack.")
        checked[key] = dict(path=path, lam_rf_min=got, title=s.get("title"),
                            nhi_min=s.get("nhi_min"),
                            n_cat_post_cuts=s.get("n_cat_post_cuts"),
                            n_truth_post_cuts=s.get("n_truth_post_cuts"))
    fpath = w["forward_npz"]
    side = fpath + ".window.json"
    if not os.path.exists(side):
        raise SystemExit(
            f"window {window!r}: forward response {fpath} carries no "
            f"{os.path.basename(side)} sidecar. The frozen NPZ has NO lam_rf "
            "stamp of its own, so an unsided response cannot be vouched for.")
    with open(side) as f:
        sc = json.load(f)
    if abs(float(sc["lam_rf_min"]) - lam) > 1e-9:
        raise SystemExit(
            f"WINDOW MISMATCH: window {window!r} has lam_rf_min={lam} but its "
            f"forward response sidecar {side} stamps "
            f"lam_rf_min={sc['lam_rf_min']}.")
    checked["forward_npz"] = dict(path=fpath, sidecar=side,
                                  lam_rf_min=float(sc["lam_rf_min"]),
                                  provenance=sc.get("provenance"),
                                  n_truth_matched_tp=sc.get("n_truth_matched_tp"))
    return dict(window=window, lam_rf_min=lam, ingredients=checked)


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def _extract_pack_module():
    """Load extract_pack.py file-directly: the hbi_mcmc package __init__ imports
    jax, and the extract phase deliberately runs in the jax-free `gpdla` env."""
    name = "modelA_extract_pack_windowstudy"
    if name in sys.modules:
        return sys.modules[name]
    p = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/extract_pack.py")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_pack(*a, **k):
    from CDDF_analysis.hbi_mcmc.pack import load_pack as _lp
    return _lp(*a, **k)


def _FS():
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS
    return FS


def full_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   text=True).strip()


def dirty():
    """TRACKED-file dirtiness only (matches extract_pack._git_commit scope)."""
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, text=True).strip())


def pack_name(mock, window):
    f = f"{PAD_FLOOR:.1f}".replace(".", "p")
    return f"modelA_pack_{mock}_win{window}_pad{f}_{COMPLETENESS}.npz"


# ---------------------------------------------------------------------------
# phase 1 — extract (env: gpdla, jax-free)
# ---------------------------------------------------------------------------
def phase_extract():
    EP = _extract_pack_module()
    os.makedirs(PACKDIR, exist_ok=True)
    manifest = {}
    for window in WINDOWS:
        m = assert_window_matched(window)
        print(f"[window] {window} matched: "
              f"{json.dumps({k: v['lam_rf_min'] for k, v in m['ingredients'].items()})}",
              flush=True)
        t0 = time.time()
        frozen = EP.build_frozen_calibration(
            PACKDIR, completeness=COMPLETENESS, window=window)
        print(f"[window] frozen[{window}] in {time.time()-t0:.0f}s; "
              f"molly cells={len(frozen['molly']['molly_nhi_edges'])-1} "
              f"g_grid={frozen['g_grid'].shape} "
              f"fp_total={int(frozen['fp_counts'].sum())}", flush=True)
        for mock in MOCKS:
            tag = f"_win{window}_pad{f'{PAD_FLOOR:.1f}'.replace('.', 'p')}_{COMPLETENESS}"
            r = EP.extract_pack(mock, PACKDIR, frozen, pad_floor=PAD_FLOOR,
                                tag=tag)
            manifest[f"{mock}|{window}"] = dict(
                mock=mock, window=window, pad_floor=PAD_FLOOR,
                completeness=COMPLETENESS, npz=r["npz"],
                counts_total=r["counts_total"], dx_gap=r["dx_gap"])
            print(f"[window] done {mock}|{window}", flush=True)
    with open(os.path.join(PACKDIR, "window_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v["counts_total"] for k, v in manifest.items()},
                     indent=1))
    return manifest


# ---------------------------------------------------------------------------
# phase 2 — fold + gate + emit (env: gpdla-hbi)
# ---------------------------------------------------------------------------
def phase_selftest():
    t_start = time.time()
    gate = restated_gate_criteria()["ratified_arms"]
    rows, packmeta = {}, {}
    for window in WINDOWS:
        for mock in MOCKS:
            p = os.path.join(PACKDIR, pack_name(mock, window))
            pack = load_pack(p)
            prov = pack.provenance or {}
            aw = prov.get("analysis_window", {})
            # FAIL CLOSED: the pack must carry the window it is filed under.
            if aw.get("name") != window:
                raise SystemExit(
                    f"{p}: provenance analysis_window.name={aw.get('name')!r} "
                    f"!= {window!r}. Refusing to compare mixed-window packs.")
            packmeta[f"{mock}|{window}"] = dict(
                pack=os.path.basename(p),
                code_commit=prov.get("code_commit"),
                analysis_window=aw,
                n_pad_bins=int(pack.n_b - pack.n_c),
                ntrue_lo=float(np.asarray(pack.ntrue_edges, float)[0]),
                nhat_lo=float(np.asarray(pack.nhat_edges, float)[0]),
                counts_total=float(np.asarray(pack.counts).sum()),
                truth_total=float(np.asarray(pack.truth_counts).sum()),
                dX_total=float(np.asarray(pack.dX, float).sum()),
                molly_counts=prov.get("molly_counts"),
                fp=prov.get("fp"),
            )
            for clamp in CLAMPS:
                key = f"{mock}|{window}|clamp={clamp}"
                t0 = time.time()
                res = _FS().selftest(pack, resp_clamp=clamp)
                tab = _FS().ratio_tables(res, pack)
                full = window_metrics(tab["by_nhat"], None, None)
                rep = window_metrics(tab["by_nhat"], REPORT_LO, REPORT_HI)
                hin = window_metrics(tab["by_nhat"], HIGHN_LO, None)
                low = window_metrics(tab["by_nhat"], None, REPORT_LO)
                rows[key] = dict(
                    mock=mock, window=window, resp_clamp=clamp,
                    pad_floor=PAD_FLOOR, completeness_below_floor=COMPLETENESS,
                    full_grid=full,
                    primary_reporting_window=rep,
                    primary_closes=closes(rep, gate),
                    high_n_above_21p6=hin,
                    below_reporting_floor=low,
                    by_z=marginal_block(tab["by_z"]),
                    by_snr=marginal_block(tab["by_snr"]),
                    pack=os.path.basename(p),
                    wall_seconds=round(time.time() - t0, 2),
                )
                print(f"{key:44s} FULL ratio={full['ratio']:.4f} "
                      f"chi2/dof={full['chi2_dof']:9.1f} | "
                      f"REPORT[{REPORT_LO},{REPORT_HI}] ratio={rep['ratio']:.4f} "
                      f"chi2/dof={rep['chi2_dof']:9.1f} | "
                      f"HIGH-N>={HIGHN_LO} ratio={hin['ratio']:.4f} "
                      f"({time.time()-t0:.1f}s)", flush=True)

    # --- cross-check the inline arithmetic against the COMMITTED gate -------
    from CDDF_analysis.hbi_mcmc import run_posterior as RP
    ref_pack = load_pack(os.path.join(PACKDIR, pack_name("2lpt0", "lya_only")))
    ref = RP.forward_closure_gate(ref_pack, resp_clamp="both")
    inline = rows["2lpt0|lya_only|clamp=both"]["full_grid"]
    xcheck = dict(
        routine="CDDF_analysis/hbi_mcmc/run_posterior.py:forward_closure_gate",
        note=("the committed gate runs on the FULL observed grid [19.5, 22.4). "
              "This cross-check therefore compares against `full_grid`, NOT "
              "the PI's 19.7-21.6 reporting window."),
        committed_total_ratio=ref["total_ratio"],
        inline_total_ratio=inline["ratio"],
        committed_chi2_dof=ref["chi2_dof"], inline_chi2_dof=inline["chi2_dof"],
        committed_z_total=ref["z_total"], inline_abs_z_total=abs(inline["z_total"]),
        committed_n_bins=ref["n_bins"], inline_n_bins_occupied=inline["n_bins_occupied"],
        agrees=bool(np.isclose(ref["total_ratio"], inline["ratio"], rtol=1e-12)
                    and np.isclose(ref["chi2_dof"], inline["chi2_dof"], rtol=1e-12)
                    and np.isclose(ref["z_total"], abs(inline["z_total"]), rtol=1e-12)),
        committed_pass=bool(ref["pass"]),
        committed_failures=ref["failures"],
    )
    print("\n[xcheck committed gate]", json.dumps(xcheck, indent=1))
    if not xcheck["agrees"]:
        raise SystemExit(
            "[window] REFUSING to stamp: the inline full-grid metrics do not "
            "reproduce the committed forward_closure_gate arithmetic.")

    # --- pack stamp audit (fail closed on a dirty input) -------------------
    commits = sorted({v["code_commit"] for v in packmeta.values()})
    sha = full_sha()
    stamp_audit = dict(
        selftest_phase_code_commit=sha, pack_code_commits=commits,
        n_packs=len(packmeta), all_packs_same_commit=bool(len(commits) == 1),
        any_pack_dirty=bool(any("-dirty" in (c or "") for c in commits)),
        packs_match_selftest_commit=bool(commits == [sha]))
    if stamp_audit["any_pack_dirty"]:
        raise SystemExit(
            "[window] REFUSING to stamp: input packs were extracted from a "
            f"DIRTY tree {commits}. Commit, then re-run --phase extract.")
    if not stamp_audit["packs_match_selftest_commit"]:
        print(f"[window] WARNING: packs stamped {commits} but the selftest "
              f"phase is at {sha} (recorded in metadata).")

    verdict = build_verdict(rows, packmeta)
    pilot = load_pilot()
    out = dict(
        metadata=dict(
            title=("Matched spectral-window study — the ANALYSIS window "
                   "(lya_only 1025 A vs lya_lyb 911 A) at the PI-adopted "
                   "pad/completeness/reporting configuration, plus a bounded "
                   "FINDER-FITTING-window pilot"),
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            code_commit=sha,
            code_commit_scope=("the SELFTEST phase (this process). Input packs "
                               "carry their own stamp — see pack_stamp_audit."),
            code_commit_dirty=dirty(),
            pack_stamp_audit=stamp_audit,
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                text=True).strip(),
            routines=dict(
                driver=("CDDF_analysis/hbi_mcmc/window_study.py "
                        "(--phase extract, then --phase selftest, "
                        "and --phase pilot for ARM 2)"),
                window_registry=("CDDF_analysis/hbi_mcmc/extract_pack.py:"
                                 "ANALYSIS_WINDOWS / window_spec"),
                window_guard=("CDDF_analysis/hbi_mcmc/window_study.py:"
                              "assert_window_matched"),
                extractor="CDDF_analysis/hbi_mcmc/extract_pack.py:extract_pack",
                fold="CDDF_analysis/hbi_mcmc/forward_selftest.py:selftest",
                metrics=("CDDF_analysis/hbi_mcmc/window_study.py:"
                         "window_metrics / select_bins / closes"),
                gate_xcheck=("CDDF_analysis/hbi_mcmc/run_posterior.py:"
                             "forward_closure_gate"),
            ),
            pack_dir=PACKDIR,
            rederive=(
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "conda run -n gpdla python CDDF_analysis/hbi_mcmc/"
                f"window_study.py --phase extract --pack-dir {PACKDIR}  &&  "
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc."
                f"window_study --phase selftest --pack-dir {PACKDIR} "
                f"--out {OUT}"),
            pack_note="packs are INPUTS, written to scratch, never committed",
            configuration=dict(
                pad_floor=PAD_FLOOR, completeness_below_floor=COMPLETENESS,
                reporting_window=[REPORT_LO, REPORT_HI],
                high_n_residual_floor=HIGHN_LO,
                resp_clamps=CLAMPS,
                basis_dex=BASIS_DEX,
                basis_resolution_status=BASIS_RESOLUTION_STATUS,
                pi_decisions_implemented=dict(
                    d1_reporting_window=(f"YES — every gate metric is computed "
                                         f"over [{REPORT_LO}, {REPORT_HI}] on "
                                         "fully-contained bins; the >= "
                                         f"{HIGHN_LO} tail is reported "
                                         "SEPARATELY and gates nothing."),
                    d3_basis_0p2dex="NO — see basis_resolution_status",
                    d4_pad_19p0_molly172=("YES — pad floor 19.0 with the "
                                          "molly172 convention on BOTH arms; "
                                          "the lya_lyb sub-floor matrix was "
                                          "BUILT for this study (it did not "
                                          "exist)."),
                    d8_gates=("PARTIAL — the fail-closed framework and the "
                              "chi2/dof <= 3 arm are used; the |z| <= 5 "
                              "criterion is restated exactly in "
                              "restated_gate_criteria(); the two unratified "
                              "ratio-span tolerances are MEASURED and gate "
                              "nothing. Matched-configuration SBC is NOT run "
                              "here (no sampling in this study)."),
                ),
            ),
            gate=restated_gate_criteria(),
            mocks_only=True,
            privacy="mock packs only; no real-LOA path is touched",
            wall_seconds=round(time.time() - t_start, 1),
        ),
        protocol=dict(
            fixed_before_outcome=True,
            rules=[dict(id=i, rule=r) for i, r in PROTOCOL],
            selection_discipline=("The nominal Molly window (lya_only) is the "
                                  "standard reference and is reported "
                                  "unconditionally. No window was preferred "
                                  "for giving a nicer answer; P2/P3/P5 were "
                                  "written before any closure number was "
                                  "inspected."),
        ),
        verdict=verdict,
        arm1_analysis_window=rows,
        pack_metadata=packmeta,
        committed_gate_crosscheck=xcheck,
        window_matching=[assert_window_matched(w) for w in WINDOWS],
        arm2_fitting_window_pilot=pilot,
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[window] wrote {OUT}")
    return out


def build_verdict(rows, packmeta):
    """The recommendation, decided by the pre-registered protocol only."""
    def get(mock, window, clamp):
        return rows[f"{mock}|{window}|clamp={clamp}"]

    per_clamp = {}
    for clamp in CLAMPS:
        d_chi2 = {m: (get(m, "lya_lyb", clamp)["primary_reporting_window"]["chi2_dof"]
                      - get(m, "lya_only", clamp)["primary_reporting_window"]["chi2_dof"])
                  for m in MOCKS}
        # P3: |ratio - 1| in the excluded high-N tail, lya_lyb minus lya_only.
        # NEGATIVE = the wider window moves the high-N residual TOWARD 1.
        d_high = {m: (abs(get(m, "lya_lyb", clamp)["high_n_above_21p6"]["ratio"] - 1.0)
                      - abs(get(m, "lya_only", clamp)["high_n_above_21p6"]["ratio"] - 1.0))
                  for m in MOCKS}
        d_ratio = {m: (get(m, "lya_lyb", clamp)["primary_reporting_window"]["ratio"]
                       - get(m, "lya_only", clamp)["primary_reporting_window"]["ratio"])
                   for m in MOCKS}
        per_clamp[clamp] = dict(
            P2_delta_chi2_dof_reporting_window=direction_verdict(d_chi2),
            P3_delta_abs_highn_residual_from_1=direction_verdict(d_high),
            measured_delta_reporting_ratio=direction_verdict(d_ratio),
        )

    closing = [k for k, v in rows.items() if v["primary_closes"]["closes"]]
    best = min(rows.items(),
               key=lambda kv: kv[1]["primary_reporting_window"]["chi2_dof"])

    def table(field, sub=None):
        out = {}
        for k, v in rows.items():
            x = v[field]
            out[k] = (x if sub is None else x[sub])
        return out

    return dict(
        question=("Should the Lya-only ANALYSIS window (lam_rf_min = 1025 A) "
                  "become the PRIMARY robust configuration, or remain a "
                  "sensitivity test?"),
        n_configurations=len(rows),
        n_closing_primary_window=len(closing),
        closing_configurations=closing,
        best_by_reporting_chi2_dof=dict(
            config=best[0],
            chi2_dof=best[1]["primary_reporting_window"]["chi2_dof"],
            ratio=best[1]["primary_reporting_window"]["ratio"],
            gate_max=3.0,
            factor_over_gate=best[1]["primary_reporting_window"]["chi2_dof"] / 3.0),
        reporting_window_chi2_dof=table("primary_reporting_window", "chi2_dof"),
        reporting_window_ratio=table("primary_reporting_window", "ratio"),
        high_n_above_21p6_ratio=table("high_n_above_21p6", "ratio"),
        full_grid_chi2_dof=table("full_grid", "chi2_dof"),
        full_grid_ratio=table("full_grid", "ratio"),
        by_z_ratio_span=table("by_z", "ratio_span"),
        by_snr_ratio_span=table("by_snr", "ratio_span"),
        protocol_outcomes=per_clamp,
        counts_and_pathlength=({k: dict(counts_total=v["counts_total"],
                                       truth_total=v["truth_total"],
                                       dX_total=v["dX_total"])
                                for k, v in packmeta.items()}),
        arm_scope=dict(
            arm1="COMPLETE — the ANALYSIS window, both arms, all three mocks.",
            arm2=("PILOT ONLY — the FINDER FITTING window. A full re-run is a "
                  "campaign and was NOT authorized. See "
                  "arm2_fitting_window_pilot for the costed decision."),
        ),
    )


def load_pilot():
    """ARM 2's pilot result, written by ``--phase pilot`` to the pack dir."""
    p = os.path.join(PACKDIR, "arm2_pilot.json")
    if not os.path.exists(p):
        return dict(status="NOT RUN", path=p)
    with open(p) as f:
        return json.load(f)


def main(argv=None):
    global PACKDIR, OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True,
                   choices=["extract", "selftest", "check-windows"])
    p.add_argument("--pack-dir", default=DEF_PACKDIR)
    p.add_argument("--out", default=DEF_OUT)
    a = p.parse_args(argv)
    PACKDIR = a.pack_dir
    OUT = a.out
    if a.phase == "check-windows":
        for w in WINDOWS:
            print(json.dumps(assert_window_matched(w), indent=1))
        return None
    if a.phase == "extract":
        return phase_extract()
    return phase_selftest()


if __name__ == "__main__":
    main()
