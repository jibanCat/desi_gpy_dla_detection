#!/usr/bin/env python
"""build_crossmock_transfer_artifact.py -- the MISSING committed aggregator for the
catalog-HBI CROSS-MOCK TRANSFER legs.

WHY THIS FILE EXISTS
--------------------
The cross-mock transfer numbers (the frozen 2LPT-0 recipe applied, with NO refit, to
held-out mocks) have until now lived only in two UNTRACKED scratch JSONs,
``CDDF_analysis/hbi/crossmock_transfer.json`` and ``crossmock_transfer_loa0.json``.
Their own ``metadata.provenance_note`` says so, verbatim:

    "Legs re-derivable from deps; the guard correctly reports ORPHANED because the
     final aggregation step has no committed routine -- that is the real debt, honestly
     recorded, not to be papered over."

Under this project's headline-provenance rule (a headline number must re-derive from a
COMMITTED routine with a GIT-STAMPED output) those files are NOT QUOTABLE: the final
aggregation was done by hand, every per-leg stamp reads ``<sha>-dirty``, and the files
are untracked.  This module is that missing routine.

WHY THIS NAME (and not ``build_artifact_loa0.py``)
--------------------------------------------------
The old note names the never-written aggregator ``build_artifact_loa0.py``.  That name is
rejected here for two concrete reasons:

  1. ``CDDF_analysis/hbi/build_loa0_fp_product.py`` already exists and builds the loa-0
     FALSE-POSITIVE PRODUCT.  ``build_artifact_loa0.py`` sitting beside it reads as a
     second builder of the same object.  It is not -- it CONSUMES that product.
  2. The debt is not loa0-specific.  Exactly the same manual aggregation produced the
     ``purity_mixture`` file.  One routine must own BOTH FP-estimator families or the
     next promotion re-opens the same hole.

Hence: ``build_crossmock_transfer_artifact.py`` -- it says what it builds (the cross-mock
transfer artifact) and is silent about which FP estimator, because it does both.

WHAT IT DOES
------------
* runs (or, with ``--reuse``, ingests) FOUR legs per FP estimator, via the COMMITTED
  per-leg drivers ``track_c_tf_{2lpt1,london0,saclay}.py``;
* aggregates them into ONE artifact with the derived reductions (cumulative R0 at the
  report limits, the differential sub-DLA band [19.5, 20.3), the transfer-specific delta
  against the on-mock self-recovery floor, and the alpha-corrected transfer error);
* stamps a 40-CHARACTER ``code_commit`` CAPTURED AT PROCESS START (never a tag, never a
  short sha), plus ``routine``, ``rederive``, the resolved input list, per-leg
  provenance (full argv + each driver's own stamp), and ``metadata['estimand']``;
* LABELS THE MOCK ROLES, because the four legs are NOT four equivalent tests.

MOCK ROLES (this is a scientific statement, not bookkeeping)
------------------------------------------------------------
  2LPT-0 (loa-124)   ON-MOCK CALIBRATION / RECOVERY FLOOR.  Not a transfer at all: it is
                     the mock the forward-response kernel, the molly completeness/purity
                     matrix, the occupancy g(N,z) and the loa-0 FP product were ALL built
                     on, and ~22% of its sightlines are literally GP training spectra.
                     Subtract it to isolate the cross-recipe part of any leg.
  2LPT-1 (loa-124)   SAME-RECIPE, DIFFERENT-REALIZATION check.  LyaCoLoRe-2LPT v2.8.5
                     mock-1.  It is a REALIZATION test, not a recipe transfer.
  London-0 (jura-124) GENUINE TRANSFER.  LyaCoLoRe on a CoLoRe lognormal density field.
  Saclay-0 (juraLy8-124) GENUINE TRANSFER.  Gaussian field + FGPA, deeper Lyman series
                     (LYB..LY8 vs LYB..LY5 for 2LPT).

ESTIMAND WARNING (read before quoting anything from the output)
---------------------------------------------------------------
Every number here is a PLUG-IN MAP point, ``R0 = MAP / truth``, integrated over z.  It is
NOT a posterior median and it carries NO credible interval (``--point-only``).  Under the
PI's 2026-07 decision, paper-facing bands must be credible intervals from faithful joint
posterior sampling with the point being the median of the SAME posterior; a plug-in MAP
is a DIAGNOSTIC.  Do not table an R0 from this artifact next to a posterior-median number
without labelling them as different estimands.

Omega is emitted but flagged NOT QUOTABLE: at the stamped commit the truth-side Omega is
built from ``truth_reductions``' ``f_truth``, which is the B16 z-leaky object (truth
counts with no z-mask against a z-masked Delta-X).  dN/dX is CLEAN -- it already carries
its own ``t_zidx >= 0`` mask.  Build the FF-vs-HBI comparison on dN/dX.

USAGE
-----
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export HDF5_USE_FILE_LOCKING=FALSE
    /home/mfho/.conda/envs/gpdla/bin/python \\
        CDDF_analysis/hbi/build_crossmock_transfer_artifact.py \\
        --legs-root /path/to/scratch/crossmock_legs \\
        --out CDDF_analysis/hbi/crossmock_transfer_artifact.json

~10 min wall on a login node (8 sequential driver invocations, point-only).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

ROUTINE = "CDDF_analysis/hbi/build_crossmock_transfer_artifact.py"

# ---------------------------------------------------------------------------
# frozen 2LPT-0 calibration artifacts (shared by EVERY leg -- this is exactly why
# 2LPT-0 is the calibration set and not an independent validation)
# ---------------------------------------------------------------------------
FORWARD_MODEL = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                 "track_c/stage0/forward_response_2lpt0.npz")
MOLLY_TSV = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
             "figures_molly_nhi195/lya_only/molly_matrix.tsv")
LOA0_PRODUCT = ("/scratch/cavestru_root/cavestru0/mfho/gl_loa0_fp_v1_20260615/"
                "outputs/loa0_fp_product_lyaonly1025.npz")

_2LPT0_CAT = ("/scratch/cavestru_root/cavestru0/mfho/gl_prod_2lpt0_v1_20260526/"
              "combined_catalog/")
_2LPT0_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                  "qq_desi_y3/v2.8.5/mock-0/loa-124")
_2LPT1_CATDIR = "/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/2lpt1_loa124_v1"
_2LPT1_MOCKDIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/"
                  "qq_desi_y3/v2.8.5/mock-1/loa-124")

# ---------------------------------------------------------------------------
# leg table.  ``driver`` is repo-relative (it is also the provenance dep list).
# ``role`` is the scientific label, NOT decoration -- see the module docstring.
# ---------------------------------------------------------------------------
LEGS = [
    dict(
        key="2lpt0_self",
        mock="2lpt0",
        driver="CDDF_analysis/hbi/track_c_tf_2lpt1.py",
        out_basename="track_c_tf_2lpt1.json",
        role="calibration_floor",
        role_note=(
            "ON-MOCK CALIBRATION / RECOVERY FLOOR -- held-out == the calibration mock "
            "itself, scored through the SAME forward-path driver. NOT a transfer. "
            "Subtract it to isolate the cross-recipe part of every other leg."),
        recipe="LyaCoLoRe-2LPT v2.8.5 mock-0 / loa-124 (the calibration mock)",
        variant="A",
        extra=[
            "--heldout-cat", _2LPT0_CAT,
            "--heldout-truth", _2LPT0_MOCKDIR + "/hcd_truth_cat.fits",
            "--heldout-bal", _2LPT0_MOCKDIR + "/bal_cat.fits",
            "--heldout-mockdir", _2LPT0_MOCKDIR,
        ],
    ),
    dict(
        key="2lpt1",
        mock="2lpt1",
        driver="CDDF_analysis/hbi/track_c_tf_2lpt1.py",
        out_basename="track_c_tf_2lpt1.json",
        role="same_recipe_new_realization",
        role_note=(
            "SAME RECIPE as the calibration mock, DIFFERENT REALIZATION (mock-1). A "
            "realization check, NOT a recipe transfer. It also has NO spectra-16 and NO "
            "processed GP .h5 anywhere on GreatLakes -- see metadata.mock_roles -- so it "
            "runs off the PACKAGED catalog only and cannot be re-processed or re-flagged."),
        recipe="LyaCoLoRe-2LPT v2.8.5 mock-1 / loa-124",
        variant="both",
        extra=[
            # the committed default --heldout-cat points at the .fits FILE, which
            # load_catalog_dir cannot glob.  Pass the DIRECTORY.  Config-only.
            "--heldout-cat", _2LPT1_CATDIR,
            "--heldout-truth", _2LPT1_MOCKDIR + "/hcd_truth_cat.fits",
            "--heldout-bal", _2LPT1_MOCKDIR + "/bal_cat.fits",
            "--heldout-mockdir", _2LPT1_MOCKDIR,
        ],
    ),
    dict(
        key="london0",
        mock="london0",
        driver="CDDF_analysis/hbi/track_c_tf_london0.py",
        out_basename="track_c_tf_london0.json",
        role="genuine_transfer",
        role_note="GENUINE TRANSFER -- LyaCoLoRe on a CoLoRe lognormal density field.",
        recipe="LyaCoLoRe london v9 / jura-124 (london0_jura124_v1)",
        variant="both",
        extra=[],       # committed defaults are already correct for this leg
    ),
    dict(
        key="saclay",
        mock="saclay0",
        driver="CDDF_analysis/hbi/track_c_tf_saclay.py",
        out_basename="track_c_tf_saclay.json",
        role="genuine_transfer",
        role_note=("GENUINE TRANSFER -- Saclay Gaussian density field + FGPA, deeper "
                   "Lyman series (quickquasars --metals LYB..LY8 vs LYB..LY5 for 2LPT)."),
        recipe="Saclay v4.7.5 mock-0 / juraLy8-124 (saclay0_juraLy8124_v1)",
        variant="both",
        extra=[],       # committed defaults are already correct for this leg
    ),
]

FP_ESTIMATORS = ("purity_mixture", "loa0")

REPORT_LIMITS = (19.5, 20.0, 20.3)
ZBINS = "2.0,2.5,3.0,3.5"
FIT_FLOOR = 19.5


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------
def _git(*args, repo=_REPO):
    return subprocess.check_output(["git", *args], cwd=repo,
                                   stderr=subprocess.DEVNULL).decode().strip()


def capture_code_commit(repo=_REPO):
    """40-CHAR HEAD sha + a tracked-dirty flag, captured ONCE at PROCESS START.

    A 40-char sha, never ``--short`` and never a tag: a tag is movable, a short sha can
    collide as the repo grows, and ``CDDF_analysis/unblind/provenance.py`` wants an
    immutable object name it can ``git cat-file`` the routine out of.

    ``dirty`` is the same tracked-only test the per-leg drivers use
    (``git status --porcelain --untracked-files=no``), so the aggregate stamp and the
    per-leg stamps agree by construction.  If it is dirty we APPEND ``-dirty`` -- the
    guard must then refuse the artifact.  Fail loud, never silently launder.
    """
    sha = _git("rev-parse", "HEAD", repo=repo)
    assert len(sha) == 40, f"expected a 40-char sha, got {sha!r}"
    dirty_paths = _git("status", "--porcelain", "--untracked-files=no", repo=repo)
    dirty = bool(dirty_paths)
    return dict(sha=sha, dirty=dirty,
                dirty_paths=[l for l in dirty_paths.splitlines() if l],
                code_commit=(sha + "-dirty") if dirty else sha)


def _stat(path):
    try:
        st = os.stat(path)
        return dict(path=path, exists=True, is_dir=os.path.isdir(path),
                    bytes=int(st.st_size),
                    mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(
                        timespec="seconds"))
    except OSError:
        return dict(path=path, exists=False, is_dir=None, bytes=None, mtime=None)


# ---------------------------------------------------------------------------
# leg execution / ingestion
# ---------------------------------------------------------------------------
def leg_argv(leg, fp, legs_root, python):
    out_dir = os.path.join(legs_root, fp, leg["key"])
    report_out = os.path.join(legs_root, fp, f"{leg['key']}_report.md")
    argv = [
        python, os.path.join(_REPO, leg["driver"]),
        "--variant", leg["variant"],
        "--point-only",
        "--report-limits", ",".join(f"{l:g}" for l in REPORT_LIMITS),
        "--fit-floor", f"{FIT_FLOOR:g}",
        "--zbins", ZBINS,
        "--fp-estimator", fp,
        "--out", out_dir,
        "--report-out", report_out,
    ]
    if fp == "loa0":
        argv += ["--loa0-product", LOA0_PRODUCT]
    argv += list(leg["extra"])
    return argv, out_dir


def run_leg(leg, fp, legs_root, python, reuse=False):
    argv, out_dir = leg_argv(leg, fp, legs_root, python)
    out_json = os.path.join(out_dir, leg["out_basename"])
    if reuse and os.path.exists(out_json):
        with open(out_json) as fh:
            return json.load(fh), argv, out_json, 0.0, "reused"
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HDF5_USE_FILE_LOCKING="FALSE")
    t0 = time.time()
    print(f"\n[agg] === {fp} / {leg['key']} ===\n[agg] {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=_REPO, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    tail = proc.stdout.decode(errors="replace")[-2500:]
    if proc.returncode != 0:
        print(tail, flush=True)
        raise RuntimeError(f"leg {fp}/{leg['key']} FAILED rc={proc.returncode}")
    if not os.path.exists(out_json):
        print(tail, flush=True)
        raise RuntimeError(f"leg {fp}/{leg['key']} produced no {out_json}")
    with open(out_json) as fh:
        d = json.load(fh)
    print(f"[agg] {fp}/{leg['key']} ok in {dt:.0f}s", flush=True)
    return d, argv, out_json, dt, "ran"


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------
def _cum(d, vk, kind, key):
    """cumulative {limit: value} for kind in (dndx, omega), key in (map, truth, R0)."""
    ir = d["variants"][vk]["integrated_R0"][kind]
    return {f"{l:g}": float(ir[str(l)][key]) for l in REPORT_LIMITS}


def subdla_band(cum_map, cum_truth):
    """DIFFERENTIAL sub-DLA band [19.5, 20.3) by differencing the two cumulatives.

    R0_band = (map(>=19.5) - map(>=20.3)) / (truth(>=19.5) - truth(>=20.3)).  This is a
    DIFFERENT estimand from any cumulative R0 and must never be tabled unlabelled beside
    one: it is dominated by the 41%-false-positive-subtraction regime.
    """
    num = cum_map["19.5"] - cum_map["20.3"]
    den = cum_truth["19.5"] - cum_truth["20.3"]
    return dict(R0=(num / den if den else float("nan")), num_map=num, den_truth=den,
                map_lo195=cum_map["19.5"], map_hi203=cum_map["20.3"],
                truth_lo195=cum_truth["19.5"], truth_hi203=cum_truth["20.3"])


def reduce_leg(d, vk):
    out = {}
    for kind in ("dndx", "omega"):
        cm = _cum(d, vk, kind, "map")
        ct = _cum(d, vk, kind, "truth")
        cr = _cum(d, vk, kind, "R0")
        out[kind] = dict(cumulative_map=cm, cumulative_truth=ct, cumulative_R0=cr,
                         subdla_band_19p5_20p3=subdla_band(cm, ct))
    return out


def transfer_deltas(leg_red, self_red):
    """Two reductions of a leg against the on-mock floor, on the SAME estimand.

    ``delta_vs_floor``      = R0_leg - R0_floor  (additive; the raw transfer penalty)
    ``alpha_corrected``     = R0_leg / R0_floor - 1  (multiplicative; what a single
                              scalar mock-correction alpha = 1/R0_floor would leave
                              behind -- i.e. the residual AFTER the naive correction)
    """
    out = {}
    for kind in ("dndx", "omega"):
        out[kind] = dict(cumulative={}, subdla_band_19p5_20p3={})
        for lim, r in leg_red[kind]["cumulative_R0"].items():
            r0 = self_red[kind]["cumulative_R0"][lim]
            out[kind]["cumulative"][lim] = dict(
                delta_vs_floor=r - r0,
                alpha_corrected=(r / r0 - 1.0) if r0 else float("nan"))
        rb = leg_red[kind]["subdla_band_19p5_20p3"]["R0"]
        r0b = self_red[kind]["subdla_band_19p5_20p3"]["R0"]
        out[kind]["subdla_band_19p5_20p3"] = dict(
            delta_vs_floor=rb - r0b,
            alpha_corrected=(rb / r0b - 1.0) if r0b else float("nan"))
    return out


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------
def build(args):
    stamp = capture_code_commit()          # <-- PROCESS START
    t_start = time.time()
    generated = datetime.datetime.now().isoformat(timespec="seconds")

    fps = FP_ESTIMATORS if args.fp_estimator == "both" else (args.fp_estimator,)

    results, per_leg_prov = {}, {}
    for fp in fps:
        raw = {}
        for leg in LEGS:
            d, argv, out_json, dt, how = run_leg(leg, fp, args.legs_root,
                                                 args.python, reuse=args.reuse)
            vks = sorted(d["variants"].keys())
            vk = "A" if "A" in vks else vks[0]
            raw[leg["key"]] = (leg, d, vk, vks)
            per_leg_prov[f"{fp}/{leg['key']}"] = dict(
                driver=leg["driver"],
                argv=argv,
                per_leg_json=out_json,
                how=how,
                variants_run=vks,
                variant_reported=vk,
                driver_stamp_code_commit=d["metadata"].get("code_commit"),
                driver_wallclock_s=d["metadata"].get("wallclock_s"),
                aggregator_wallclock_s=dt,
                point_only=d["metadata"].get("point_only"),
                fp_estimator=d["metadata"].get("fp_estimator"),
                heldout_cat=d["metadata"].get("heldout_cat"),
                heldout_truth=d["metadata"].get("heldout_truth"),
                heldout_mockdir=d["metadata"].get("heldout_mockdir"),
                forward_model=d["metadata"].get("forward_model"),
                molly_tsv=d["metadata"].get("molly_tsv"),
                loa0_product=d["metadata"].get("loa0_product"),
            )

        self_leg, self_d, self_vk, _ = raw["2lpt0_self"]
        self_red = reduce_leg(self_d, self_vk)

        block = {}
        for leg in LEGS:
            _l, d, vk, vks = raw[leg["key"]]
            red = reduce_leg(d, vk)
            entry = dict(
                mock=leg["mock"], recipe=leg["recipe"],
                role=leg["role"], role_note=leg["role_note"],
                variant_reported=vk, variants_run=vks,
                reductions=red,
            )
            # A/B agreement check: point-only -> A and B must coincide exactly (the
            # forward-path POINT is band-independent, and the loa0 FP is
            # molly-count-independent).  Record it, do not assume it.
            if "B" in vks and "A" in vks:
                redB = reduce_leg(d, "B")
                entry["variantB_max_abs_reldiff"] = max(
                    abs(redB[k]["cumulative_R0"][l] / red[k]["cumulative_R0"][l] - 1.0)
                    for k in ("dndx", "omega") for l in red[k]["cumulative_R0"])
            if leg["key"] != "2lpt0_self":
                entry["vs_calibration_floor"] = transfer_deltas(red, self_red)
            block[leg["key"]] = entry
        results[fp] = block

    wallclock = time.time() - t_start

    inputs = [
        dict(role="forward_response_kernel (FROZEN 2LPT-0)", **_stat(FORWARD_MODEL)),
        dict(role="molly C/rho matrix, lya_only nhi195 (FROZEN 2LPT-0)",
             **_stat(MOLLY_TSV)),
        dict(role="loa-0 twin-mock FP product (loa0 estimator only)",
             **_stat(LOA0_PRODUCT)),
        dict(role="2LPT-0 calibration catalog", **_stat(_2LPT0_CAT)),
        dict(role="2LPT-0 mock-0 truth", **_stat(_2LPT0_MOCKDIR + "/hcd_truth_cat.fits")),
        dict(role="2LPT-1 packaged catalog (DIRECTORY)", **_stat(_2LPT1_CATDIR)),
        dict(role="2LPT-1 mock-1 truth", **_stat(_2LPT1_MOCKDIR + "/hcd_truth_cat.fits")),
        dict(role="2LPT-1 mock-1 QSO/snr catalogs (pathlength)",
             **_stat(_2LPT1_MOCKDIR + "/snr_cat.fits")),
        dict(role="London-0 packaged catalog",
             **_stat("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/london0_jura124_v1")),
        dict(role="London-0 staged mockdir (truth dla_cat.fits + zcat + staged snr_cat)",
             **_stat("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
                     "track_c/tf_london0/mockdir")),
        dict(role="Saclay-0 packaged catalog",
             **_stat("/gpfs/accounts/cavestru_root/cavestru0/mfho/"
                     "gl_prod_saclay0_v1_20260630/combined_catalog")),
        dict(role="Saclay-0 native mockdir (truth hcd_truth_cat.fits + native snr_cat)",
             **_stat("/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/"
                     "v4.7.5/mock-0/juraLy8-124")),
    ]

    rederive = (
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "HDF5_USE_FILE_LOCKING=FALSE; "
        f"/home/mfho/.conda/envs/gpdla/bin/python {ROUTINE} "
        f"--legs-root <scratch>/crossmock_legs --fp-estimator both "
        f"--out CDDF_analysis/hbi/crossmock_transfer_artifact.json  "
        "# runs all 8 legs (4 mocks x 2 FP estimators) via the committed "
        "track_c_tf_{2lpt1,london0,saclay}.py drivers; ~10 min on a login node. "
        "MUST be run from a CLEAN tree (no modified tracked files) or the stamp is "
        "suffixed -dirty and the provenance guard will refuse it."
    )

    artifact = dict(
        title=("Catalog-HBI CROSS-MOCK TRANSFER legs -- frozen 2LPT-0 recipe, no refit, "
               "sub-DLA report limits (19.5, 20.0, 20.3). ALL MOCK, public-OK."),
        metadata=dict(
            code_commit=stamp["code_commit"],
            code_commit_sha40=stamp["sha"],
            code_commit_dirty=stamp["dirty"],
            code_commit_dirty_paths=stamp["dirty_paths"],
            routine=ROUTINE,
            deps=sorted({leg["driver"] for leg in LEGS}),
            rederive=rederive,
            generated=generated,
            wallclock_s=wallclock,
            python=args.python,
            estimand=dict(
                quantity="R0 = recovered / truth, INTEGRATED (z-marginalised)",
                point=("PLUG-IN MAP of the catalog-HBI v3 estimator with the MEASURED "
                       "FORWARD-RESPONSE kernel (resp_kind='forward'). NOT a posterior "
                       "median."),
                interval=("NONE. --point-only: no MC band was drawn. Do not attach an "
                          "uncertainty to these numbers and do not table them beside a "
                          "posterior-median number without labelling both."),
                status=("DIAGNOSTIC-grade under the PI's 2026-07 band decision "
                        "(paper-facing bands must be credible intervals from faithful "
                        "joint posterior sampling, point = median of the SAME "
                        "posterior). A plug-in MAP is explicitly a diagnostic."),
                cumulative_vs_band=("cumulative_R0[lim] is R0 for the CUMULATIVE "
                                    "quantity >= lim; subdla_band_19p5_20p3 is the "
                                    "DIFFERENTIAL [19.5, 20.3) band formed by "
                                    "differencing two cumulatives. DIFFERENT ESTIMANDS."),
                calibration=("FROZEN 2LPT-0: forward_response_2lpt0.npz + g(N,z) + the "
                             "lya_only-nhi195 molly C/rho matrix. NO refit on any "
                             "held-out mock. Only the held-out-side FP model changes "
                             "between the two fp_estimator families."),
            ),
            quotability=dict(
                dndx=("QUOTABLE. truth_reductions' dN/dX carries its own z-mask "
                      "(t_zidx >= 0), so it is not B16-contaminated."),
                omega=("NOT QUOTABLE at this commit. Omega is built from f_truth, which "
                       "at the stamped commit counts truth rows with NO z-mask against a "
                       "z-MASKED Delta-X (B16). RE-DERIVE after the B16 fix lands; never "
                       "rescale by a scalar."),
                f_of_N="not emitted by this artifact (point-only, integrated reductions).",
            ),
            mock=("2lpt0 / 2lpt1 / london0 / saclay0 are ALL MOCK (public-OK). The loa-0 "
                  "FP product is the HCD-free TWIN MOCK, not real LoA. No real-DESI "
                  "(loa main-dark) data was read by any leg. Provenance test: mock "
                  "TARGETID is O(1e3-1e8), real DESI is O(1e16)."),
            inputs=inputs,
            per_leg_provenance=per_leg_prov,
            supersedes=dict(
                files=["CDDF_analysis/hbi/crossmock_transfer.json",
                       "CDDF_analysis/hbi/crossmock_transfer_loa0.json"],
                why=("both were UNTRACKED, every per-leg stamp read '<sha>-dirty', and "
                     "their own provenance_note recorded that the FINAL AGGREGATION WAS "
                     "MANUAL with no committed aggregator ('build_artifact_loa0.py was "
                     "never written'). This artifact replaces them: committed routine, "
                     "40-char stamp captured at process start, git-tracked output."),
            ),
            provenance_note=(
                "Legs re-run from a CLEAN checkout of the stamped commit via the "
                "committed drivers; the aggregation step is now itself a committed "
                "routine (metadata.routine). Numbers WILL differ from the retired "
                "untracked JSONs wherever the committed code has moved since d496f42 -- "
                "that difference is the point of re-deriving, not an error."),
        ),
        config=dict(
            report_limits=list(REPORT_LIMITS),
            fit_floor=FIT_FLOOR,
            zbins=[float(x) for x in ZBINS.split(",")],
            point_only=True,
            n_mc=0,
            fp_estimators=list(fps),
            variant_policy=("floor leg: A only (the point is variant-independent under "
                            "--point-only); transfer legs: both A and B, with the "
                            "A-vs-B agreement recorded per leg."),
            forward_model=FORWARD_MODEL,
            molly_tsv=MOLLY_TSV,
            loa0_product=LOA0_PRODUCT,
        ),
        mock_roles={leg["key"]: dict(mock=leg["mock"], role=leg["role"],
                                     note=leg["role_note"], recipe=leg["recipe"])
                    for leg in LEGS},
        results=results,
    )
    return artifact


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legs-root", required=True,
                   help="scratch root for the per-leg driver outputs")
    p.add_argument("--out", default=os.path.join(
        _HERE, "crossmock_transfer_artifact.json"))
    p.add_argument("--fp-estimator", default="both",
                   choices=["purity_mixture", "loa0", "both"])
    p.add_argument("--python", default=sys.executable,
                   help="interpreter used to run the per-leg drivers (needs fitsio)")
    p.add_argument("--reuse", action="store_true",
                   help="ingest an existing per-leg JSON instead of re-running it")
    p.add_argument("--allow-dirty", action="store_true",
                   help="write the artifact even if the tree is dirty (it will be "
                        "stamped '-dirty' and the provenance guard will REFUSE it)")
    args = p.parse_args(argv)

    art = build(args)
    if art["metadata"]["code_commit_dirty"] and not args.allow_dirty:
        raise SystemExit(
            "[agg] REFUSING to write: the tree has modified tracked files, so the stamp "
            "would be '<sha>-dirty' and CDDF_analysis/unblind/provenance.py would class "
            "the artifact DIRTY (not quotable). Modified:\n  "
            + "\n  ".join(art["metadata"]["code_commit_dirty_paths"])
            + "\nRun from a clean checkout (e.g. `git worktree add --detach <dir> <sha>`)"
              " or pass --allow-dirty to write a knowingly-unquotable diagnostic.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(art, fh, indent=2, default=float)
    print(f"\n[agg] artifact -> {args.out}")
    print(f"[agg] code_commit = {art['metadata']['code_commit']}")
    for fp, block in art["results"].items():
        for k, e in block.items():
            r = e["reductions"]["dndx"]
            print(f"[agg] {fp:15s} {k:12s} ({e['role']:26s}) "
                  f"dN/dX R0: >=19.5 {r['cumulative_R0']['19.5']:.4f}  "
                  f">=20.3 {r['cumulative_R0']['20.3']:.4f}  "
                  f"subDLA-band {r['subdla_band_19p5_20p3']['R0']:.4f}")
    return art


if __name__ == "__main__":
    main()
