#!/usr/bin/env python3
"""build_science_manifest.py

COMPONENT CLASS: **PROVENANCE-ONLY**.

Emits a `science_manifest.json` for a Paper-1 low-z freeze, validated against
`SCIENCE_MANIFEST_SCHEMA.json` (phase-0c).  It produces NO science value: every number it
writes is transcribed from a file another lane produced, and every edge it records is read
from a recorded source (a provenance sidecar, a selection contract, a producer's own
`common.emit()` call, a comparison product).  Nothing is inferred; links that cannot be
filled are emitted as explicit `_GAP ...` markers with matching `gaps[]` entries.

Two modes:

  --mode candidate          the surgical C1 candidate (freeze.status = candidate)
  --mode reconstruct-frozen the 2026-08-26 freeze, RECONSTRUCTED from
                            docs/PAPER1_FROZEN_MANIFEST.json + the frozen Turbo archive +
                            the frozen pooled reductions (R-042a reduce.py outputs) + the
                            paper repo working tree.  Lossy; every field that the archive
                            never recorded is emitted as `_GAP` and listed in gaps[], and
                            the manifest is stamped previous_manifest_completeness =
                            "reconstructed" by its consumer.

Also writes, beside the manifest, a machine-readable `ledger_quantities.json`: the quotable
set as NUMBERS (value, interval, printed_precision, print_multiplier, the printed string) so
the Paper lane can read a value instead of hand-editing the systematics ledger.  That file is
a PROPOSAL to the paper lane; this script never writes into the paper or notes repositories.

This script reads (read-only) from the candidate tree, the frozen Turbo archive, the paper
repository and the phase-0a/0b/0c reports.  It writes only to --out-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------- sources
PHASE0 = pathlib.Path("/home/mfho/lowz_clean_work_2026-09-11")
SCHEMA_PATH = PHASE0 / "phase0c/SCIENCE_MANIFEST_SCHEMA.json"
CONSUMER_INVENTORY = PHASE0 / "phase0c/CONSUMER_INVENTORY.json"
PACK_SOURCE_MAP = PHASE0 / "phase0a/PACK_SOURCE_MAP.json"
SHIPPING_AUDIT = PHASE0 / "phase0b/PHASE0B_SHIPPING_AUDIT.json"

HANDOFF = pathlib.Path("/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff")
CANDIDATE = HANDOFF / "LOWZ_CLEAN_C1_CANDIDATE_2026-09-10"
R042A = HANDOFF / "R042a_run_2026-09-10"
PHASE1_REBUILD = HANDOFF / "PHASE1_FULL_REBUILD_2026-09-11"
FROZEN_ARCHIVE = pathlib.Path("/nfs/turbo/lsa-cavestru/mfho/paper1_frozen_2026-08-26")
FROZEN_MANIFEST = pathlib.Path("/home/mfho/wt_forward_2026_08/docs/PAPER1_FROZEN_MANIFEST.json")
PAPER = pathlib.Path("/home/mfho/Latex/gp_dla_desi_y3")
NOTES_LEDGER = pathlib.Path(
    "/home/mfho/desi_gpy_dla_notes/figures/2026-08-21_freeze_pathB/ledger_v2p3_cp3.json")

PAPER_HEAD = "4758dae"                      # read 2026-09-11 (Phase-0b/0c)
PAPER_COMMIT_AT_FIGDATA = "362dc8b785a3124080657f7f00ea6679321624ec"
CODE_COMMIT = "1fd482812dbc459f2073d47617993d02abe55505"   # prov/paper1-freeze-2026-08-26
CODE_TAG = "prov/paper1-freeze-2026-08-26"

HZ2_POOLED_SHA256 = "9e1c2f0e06c290eb4d5e8133df94ed370d733d2dc1ce7787b8891caa90bc2c5c"
HZ2_PACK_SHA256 = "814af27a3d77921f9268446d433411b22eeaee5d909c604f1fea4f00be6357c3"
R034_NPZ_SHA256 = "8fe4a7d4dfc1c93be5fb10c5630386e5b36c6993be24b75adf54007929bd19af"
# the R-037 Omega anatomy, pinned in full at paper_figures/emit_tab_systematics.py:52
R037_SHA256 = "473a0afca360836fd140963aa9eaeb557c0c98ef9ecae14a3182a7e5bce60c2f"
DLA_DATA_COMMIT = "e24f7b924207a7fe3d3c111a43a6f2739eddf955"

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
BINS = ["B1", "B2", "B3", "B4", "B5"]
CDDF_BINS = ["19.7_19.9", "19.9_20.1", "20.1_20.3", "20.3_20.5", "20.5_20.7", "20.7_20.9",
             "20.9_21.1", "21.1_21.3", "21.3_21.5", "21.5_21.7", "21.7_21.9", "21.9_22.1",
             "22.1_22.4"]


# ---------------------------------------------------------------------------- helpers
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def first_sha(text):
    """A recorded sha field may carry trailing prose ('... 2.91 GiB, not re-hashed today')."""
    if not isinstance(text, str):
        return None
    m = HEX64.search(text)
    return m.group(0) if m else None


def utcnow():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Gaps:
    """Collects gaps[] entries so no unfillable link is ever silently dropped."""

    def __init__(self):
        self.items = []
        self._seen = set()

    def add(self, link, gap, severity, fillable_from=None, note=None):
        if link in self._seen:
            return f"_GAP: {gap}"
        self._seen.add(link)
        e = {"link": link, "gap": gap, "severity": severity}
        if fillable_from:
            e["fillable_from"] = fillable_from
        if note:
            e["note"] = note
        self.items.append(e)
        return f"_GAP: {gap}"


class DurableIndex:
    """Resolve an artifact to a DURABLE copy, by sha256, across the three known mirrors.

    The PI requirement is that no chain depends on scratch/home working state.  This index
    is how that requirement is CHECKED rather than asserted: it maps a content hash onto a
    path under /nfs/turbo, using only registries that already exist on disk.
    """

    def __init__(self):
        self.by_sha = {}
        # (1) the frozen Turbo archive, addressed through its own 165-entry manifest
        if FROZEN_MANIFEST.exists():
            for e in load_json(FROZEN_MANIFEST)["entries"]:
                mirrored = FROZEN_ARCHIVE / e["path"].lstrip("/")
                if e.get("sha256") and mirrored.exists():
                    self.by_sha.setdefault(e["sha256"],
                                           (str(mirrored), "paper1_frozen_2026-08-26_archive"))
        # (2) the Phase-1 canonical-inputs durability mirror (OPUS-1, 2026-09-11)
        cim = PHASE1_REBUILD / "CANONICAL_INPUTS_MANIFEST.json"
        if cim.exists():
            for it in load_json(cim).get("items", []):
                sha = it.get("sha256_turbo_copy") or it.get("expect")
                tp = it.get("turbo_path")
                if sha and tp and os.path.exists(tp):
                    self.by_sha.setdefault(sha, (tp, "PHASE1_FULL_REBUILD/canonical_inputs"))
        # (3) the sealed C1 candidate tree
        sums = CANDIDATE / "SHA256SUMS"
        if sums.exists():
            for line in sums.read_text().splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and HEX64.fullmatch(parts[0]):
                    p = CANDIDATE / parts[1].strip()
                    if p.exists():
                        self.by_sha.setdefault(parts[0], (str(p), "LOWZ_CLEAN_C1_CANDIDATE"))

    def resolve(self, sha):
        return self.by_sha.get(sha)


DUR = DurableIndex()


def classify_path(p: str) -> str:
    if p.startswith("/nfs/turbo/"):
        return "durable_turbo"
    if p.startswith("/scratch/"):
        return "scratch"
    if p.startswith("/gpfs/"):
        return "scratch"
    if p.startswith("/home/mfho/Latex/release_staging"):
        return "release_staging"
    if p.startswith("/home/mfho/Latex/") or p.startswith("/home/mfho/desi_gpy_dla"):
        return "repo"
    if p.startswith("/home/"):
        return "home_working"
    if not p.startswith("/"):
        return "repo"          # a path recorded relative to a repository root
    return "external"


def location(path, sha=None, gaps: Gaps = None, link=None):
    """Build a $defs/location.  Prefers a durable copy resolved BY CONTENT HASH."""
    path = str(path)
    lc = classify_path(path)
    if lc in ("scratch", "home_working", "nersc") and sha and HEX64.fullmatch(sha):
        hit = DUR.resolve(sha)
        if hit:
            return {"path": hit[0], "location_class": "durable_turbo",
                    "_recorded_path": path,
                    "_durability": f"durable copy resolved by sha256 in {hit[1]}"}
    if lc in ("scratch", "home_working", "nersc"):
        mirror = FROZEN_ARCHIVE / path.lstrip("/")
        if mirror.exists() and mirror.is_file():
            return {"path": str(mirror), "location_class": "durable_turbo",
                    "_recorded_path": path,
                    "_durability": ("durable copy resolved by PATH in the "
                                    "paper1_frozen_2026-08-26 archive mirror; the source registry "
                                    "records no sha256 for this artifact, so the identity of the "
                                    "mirror is asserted by path, not verified by content")}
        msg = (f"{path} is {lc}-only; no durable copy could be resolved by content hash in "
               f"the frozen Turbo archive, the Phase-1 canonical-inputs mirror or the C1 "
               f"candidate tree")
        ref = gaps.add(link or path, msg, "MAJOR",
                       note="violates the PI requirement of no dependence on scratch state") \
            if gaps else "_GAP: no durable copy"
        return {"path": path, "location_class": lc, "durable_copy_ref": ref}
    return {"path": path, "location_class": lc}


def rel(cls, basis, **kw):
    d = {"class": cls, "basis": basis}
    d.update({k: v for k, v in kw.items() if v is not None})
    return d


AGG = rel("public_release_eligible", "aggregate_reduced")
DOC = rel("public_release_eligible", "documentation")


def fmt(value, precision, multiplier=1):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return f"{value * multiplier:.{precision}f}"


def fmt_band(q, precision, multiplier=1):
    """The TAB-12 / prose form: median [q16, q84]."""
    return (f"{q[2] * multiplier:.{precision}f} "
            f"[{q[1] * multiplier:.{precision}f}, {q[3] * multiplier:.{precision}f}]")


# ------------------------------------------------------------------- figure-array edges
#
# These edge tables are READ FROM THE PRODUCERS, not from any old/new comparison:
#   paper_figures/fig_hbi_dndx.py  common.emit(...)  lines 365-430
#   paper_figures/fig_hbi_cddf.py  common.emit(...)  lines 169-215
# Each npz key is mapped to the manifest entries it is computed from.  manifest_diff then
# derives which keys are stale by walking these edges; the P1 gate checks that the derived
# set equals the independently measured D2 array diff.  If the two disagree the EDGE TABLE
# is wrong and must be corrected here -- never the expectation.
LEDGER = "product:ledger"
ENVP = "product:mock_envelope"
HZ2 = "product:hz2_posterior"
LIT = "product:literature_dla_data"
PYIGM = "product:pyigm_reference_r034"
CONV = "input:reporting_conventions"
DRAWS = "product:pooled_draws"
POST = "product:pooled_posterior"
PACK = "product:pack"

_lowz_dndx00 = [f"dndx_20p0_{b}" for b in BINS]
_lowz_dndx03 = [f"dndx_20p3_{b}" for b in BINS]
_lowz_om03 = [f"omega_20p3_21p6_{b}" for b in BINS]
_cddf_q = [f"cddf_{b}" for b in CDDF_BINS]

FIG41_EDGES = {
    "allz_dndx_20p0": ["dndx_20p0_allz"],
    "allz_dndx_20p3": ["dndx_20p3_allz"],
    "allz_omega_20p3": ["omega_20p3_21p6_allz"],
    "lowz_dndx_20p0": _lowz_dndx00,
    "lowz_dndx_20p3": _lowz_dndx03,
    "lowz_omega_20p3": _lowz_om03,
    "closure_max_rel_diff": [POST, DRAWS],
    "dX_per_bin": [PACK],
    "l15_dndx_20p0": _lowz_dndx00 + [LEDGER],
    "l15_dndx_20p3": _lowz_dndx03 + [LEDGER],
    "cmp_ho21_ratio_to_reference": [LIT] + _lowz_dndx03,
    "l15_ratio_20p0": [LEDGER], "l15_ratio_20p3": [LEDGER],
    "l2_band_20p0": [LEDGER], "l2_band_20p3": [LEDGER],
}
for _k in ("env_dndx_20p0_bias_max", "env_dndx_20p0_bias_min", "env_dndx_20p3_bias_max",
           "env_dndx_20p3_bias_min", "env_dndx_20p3_ratio_hi", "env_dndx_20p3_ratio_lo",
           "env_dndx_families", "env_dndx_n_runs", "env_dndx_n_runs_per_family"):
    FIG41_EDGES[_k] = [ENVP]
for _k in ("highz_code_commit_as_run", "highz_code_commit_note", "highz_code_commit_public",
           "highz_dndx_20p0", "highz_dndx_20p3", "highz_estimand", "highz_n_draws",
           "highz_pack_sha256", "highz_quantile_levels", "highz_quantile_names",
           "highz_role", "highz_seeds_excluded", "highz_seeds_excluded_disclosed",
           "highz_seeds_included"):
    FIG41_EDGES[_k] = [HZ2]
for _k in ("cmp_conventions", "cmp_dla_data_commit", "cmp_ho21_dndx", "cmp_ho21_hi68",
           "cmp_ho21_lo68", "cmp_ho21_ratio_z", "cmp_ho21_z"):
    FIG41_EDGES[_k] = [LIT]
for _k in ("bin_edges", "bin_names", "coverage", "lowz_support", "omega_limits",
           "plot_z_supported_centre_20p3", "quantile_levels", "ratio_reference",
           "ratio_threshold", "z_eff_provenance_note"):
    FIG41_EDGES[_k] = [CONV]

FIG40_EDGES = {
    "closure_max_rel_diff": [POST, DRAWS],
    "cumulative": [DRAWS, PACK],
    "dndx_allz_20p0": ["dndx_20p0_allz"],
    "dndx_allz_20p3": ["dndx_20p3_allz"],
    "omega_20p0": [DRAWS, PACK],
    "omega_20p3": ["omega_20p3_21p6_allz"],
    "f_bin_integral": _cddf_q,
    "f_per_dex": _cddf_q,
    "f_per_linear_N": _cddf_q,
    "stat_ratio_to_median": _cddf_q,
    "env_truth_side_f_per_N_hi": [ENVP] + _cddf_q,
    "env_truth_side_f_per_N_lo": [ENVP] + _cddf_q,
    "l15_f_per_linear_N": [LEDGER] + _cddf_q,
    "ref_spine_binavg_ratio": [PYIGM] + _cddf_q,
}
for _k in ("env_bias_max", "env_bias_min", "env_min_truth_support", "env_ratio_hi",
           "env_ratio_lo", "env_supported"):
    FIG40_EDGES[_k] = [ENVP]
for _k in ("ref_spine_binavg_f_per_N", "ref_spine_citation", "ref_spine_cosmology",
           "ref_spine_f_per_N", "ref_spine_form", "ref_spine_logN", "ref_spine_params",
           "ref_spine_path_outside_pyigm_validity", "ref_spine_product_sha256"):
    FIG40_EDGES[_k] = [PYIGM]
FIG40_EDGES["l15_ratio_to_median"] = [LEDGER]
FIG40_EDGES["cmp_dla_data_commit"] = [LIT]
for _k in ("cumulative_edges", "env_min_support_rule", "nhi_hi", "nhi_lo", "omega_limits",
           "quantile_levels", "ratio_reference", "tier"):
    FIG40_EDGES[_k] = [CONV]

KEY_UNITS = {
    "dndx": "dimensionless (per unit absorption distance dX)",
    "omega": "dimensionless (Omega_DLA; printed x1e4)",
    "f": "cm^2 (f(N,X) per linear N_HI) / per dex where stated",
    "dX": "absorption distance",
    "pct": "per cent", "ratio": "dimensionless (ratio)", "text": "n/a (string field)",
}


def key_meta(key):
    """axes / units / definition for one npz key, from its name and the producer's own use."""
    if key.startswith("highz_"):
        return (["quantile"] if "dndx" in key else ["scalar"], KEY_UNITS["dndx"],
                "high-z (HZ2) arm value, transcribed from the 2026-09-04 freeze")
    if key.startswith("lowz_"):
        return (["z_bin", "quantile"],
                KEY_UNITS["omega"] if "omega" in key else KEY_UNITS["dndx"],
                "per reporting bin B1..B5, posterior quantiles [2.5,16,50,84,97.5]")
    if key.startswith("allz_") or key.startswith("dndx_allz"):
        return (["quantile"],
                KEY_UNITS["omega"] if "omega" in key else KEY_UNITS["dndx"],
                "all-redshift value over the arm's absorber support [2.0,3.5]")
    if key.startswith("omega_2"):
        return ["quantile"], KEY_UNITS["omega"], "Omega_DLA posterior quantiles"
    if key.startswith("cddf") or key.startswith("f_"):
        return ["nhi_bin", "quantile"], KEY_UNITS["f"], "CDDF posterior quantiles per N_HI bin"
    if key == "dX_per_bin":
        return ["z_bin"], KEY_UNITS["dX"], "absorption path per reporting bin (from the pack)"
    if key.startswith("env"):
        return ["nhi_bin"], KEY_UNITS["ratio"], "mock-recovery envelope"
    if key.startswith("ref_spine"):
        return ["nhi"], KEY_UNITS["f"], "pyigm default reference layer (R-034)"
    if key.startswith("cmp_"):
        return ["z"], KEY_UNITS["dndx"], "Ho, Bird & Garnett 2021 comparison layer"
    if key.startswith("l15") or key.startswith("l2_"):
        return ["z_bin"], KEY_UNITS["ratio"], "systematics-ledger layer (L15 / L2)"
    return ["scalar"], KEY_UNITS["ratio"], "figure metadata / convention constant"


def npz_file_schema(path, edges):
    """file_schema for a figure .data.npz: one entry per key, with axes/units/definition."""
    entries = []
    with np.load(path, allow_pickle=True) as z:
        for k in sorted(z.files):
            a = z[k]
            axes, units, definition = key_meta(k)
            e = {"name": k, "axes": axes, "units": units, "definition": definition,
                 "shape": [int(s) for s in a.shape],
                 "dtype": str(a.dtype)}
            if k in edges:
                e["_upstream"] = edges[k]
            entries.append(e)
    return {"format": "npz", "entries": entries}


# ---------------------------------------------------------------------------- inputs
RESTRICTION_TO_REL = {
    "DR2-RESTRICTED": rel("collaboration_restricted", "object_level_real",
                          phase="phase2_after_dr2_public",
                          rationale="one row per DESI object; DESI Publication Policy v2.0 s2.4"),
    "public-eligible": AGG,
}


def build_inputs(gaps: Gaps, side: str):
    psm = load_json(PACK_SOURCE_MAP)
    out = []
    for key, art in psm["upstream_artifacts"].items():
        sha = first_sha(art.get("sha256", "")) or gaps.add(
            f"input:{key}.sha256", f"no full sha256 recorded for {key}", "moderate",
            fillable_from="re-hash the artifact")
        path = art.get("path", "")
        loc = location(path, sha if HEX64.fullmatch(str(sha)) else None, gaps,
                       link=f"input:{key}.durable_copy")
        restriction = art.get("restriction", "")
        if "DR2-RESTRICTED" in restriction:
            re_ = RESTRICTION_TO_REL["DR2-RESTRICTED"]
        elif "public" in restriction:
            re_ = AGG
        else:
            re_ = rel("undetermined", "object_level_mock" if "mock" in key else "aggregate_reduced",
                      rationale=f"restriction recorded as {restriction!r} in PACK_SOURCE_MAP")
        e = {
            "input_id": f"input:{key}",
            "location": loc,
            "sha256": sha,
            "role": art.get("role", ""),
            "origin": art.get("origin", ""),
            "release_eligibility": re_,
            "status": "current",
            "notes": art.get("recoverability", ""),
        }
        if art.get("bytes"):
            e["size_bytes"] = int(art["bytes"])
        out.append(e)

    # the reporting conventions the figure producers read their constants from
    red = PAPER / "paper_figures/hbi_reduction.py"
    out.append({
        "input_id": CONV,
        "location": {"path": str(red), "location_class": "repo"},
        "sha256": sha256_file(red),
        "role": ("reporting conventions of record: bin edges, the absorber support [2.0,3.5], "
                 "the quantile order, the N_HI reporting window, h=0.70 / Omega_m=0.279. "
                 "Every figure-array key that is a pure convention constant is an edge to this."),
        "origin": "paper repository, paper_figures/hbi_reduction.py",
        "release_eligibility": rel("public_release_eligible", "code"),
        "status": "current",
    })

    # the injection substrates the systematics campaigns selected on.  Their hashes were
    # first recorded by OPUS-E (VALIDATION_CLEAN_C1.json::inputs_sha256) and are recorded
    # here for the first time as manifest entries.  On the FROZEN side they are unrecorded.
    if side == "c1":
        try:
            val = load_json(CANDIDATE / "systematics/validation/VALIDATION_CLEAN_C1.json")
            for name, iid, role in (
                ("h2m_sightlines.csv", "input:h2m_substrate",
                 "H2-M injection substrate (540 real sightlines); carries the MEDIAN RED_SNR column"),
                ("h2mc_sightlines.csv", "input:h2mc_substrate",
                 "H2-M clean-arm substrate (525 real sightlines)"),
                ("l8real_sightlines.csv", "input:l8real_substrate",
                 "L8 extended-injection substrate (1,206 real sightlines)"),
            ):
                rec = val["inputs_sha256"].get(name)
                if not rec:
                    continue
                out.append({
                    "input_id": iid,
                    "location": location(rec["path"], rec["sha256"], gaps,
                                         link=f"{iid}.durable_copy"),
                    "sha256": rec["sha256"],
                    "role": role,
                    "origin": "systematics injection campaign, ckpt-10.5",
                    "release_eligibility": rel(
                        "collaboration_restricted", "object_level_real",
                        restricted_through=["TARGETID", "RED_SNR"],
                        phase="phase2_after_dr2_public",
                        rationale="per-object identifiers; the aggregate built from it is eligible"),
                    "status": "current",
                })
        except FileNotFoundError:
            pass
    else:
        gaps.add("frozen:campaign substrate hashes",
                 "The 2026-08-26 freeze records no hash, path or manifest entry for the "
                 "H2-M / clean-arm / L8 injection substrates; their identities were first "
                 "pinned by OPUS-E on 2026-09-10. Old-side campaign edges are therefore "
                 "old_side_unrecorded, not 'unchanged'.", "MAJOR",
                 fillable_from="VALIDATION_CLEAN_C1.json::inputs_sha256 (C1 side only)")
    return out


# ---------------------------------------------------------------------------- contract
def build_contract(side, inputs_ids, gaps: Gaps):
    c1_sc = load_json(CANDIDATE / "contract/C1.selection_contract.json")
    frozen = side == "frozen"
    estimands = [
        {"estimand_id": "dndx_20p3_allz",
         "definition": ("Posterior incidence per unit absorption distance of absorbers with "
                        "log10 N_HI >= 20.3, integrated over the arm's full absorber support "
                        "z in [2.0,3.5]; observable-only estimator, collar c=3300 km/s; reduced "
                        "by paper_figures/hbi_reduction.py reduce_f_posterior; reported as the "
                        "posterior median with quantiles [2.5,16,50,84,97.5]."),
         "units": "dimensionless (per unit dX)", "print_multiplier": 1,
         "nhi_threshold": 20.3, "z_domain": "[2.0, 3.5]",
         "counting_convention": "observable-only estimator, ADOPTED collar 3300 km/s (PI ckpt-10.8)",
         "reported": True},
        {"estimand_id": "dndx_20p0_allz",
         "definition": "As dndx_20p3_allz at the companion threshold log10 N_HI >= 20.0.",
         "units": "dimensionless (per unit dX)", "print_multiplier": 1,
         "nhi_threshold": 20.0, "z_domain": "[2.0, 3.5]", "reported": True},
        {"estimand_id": "omega_20p3_21p6_allz",
         "definition": ("Posterior cosmological mass density of neutral hydrogen in DLAs with "
                        "log10 N_HI in [20.3,21.6], same z domain and reduction; "
                        "OMEGA_PREFACTOR_CM2 * f-integral, h=0.70, Omega_m=0.279."),
         "units": "dimensionless (Omega_DLA)", "print_multiplier": 10000,
         "nhi_range": [20.3, 21.6], "z_domain": "[2.0, 3.5]",
         "counting_convention": "N_HI,max of record 22.4; Omega closed at 21.6",
         "reported": True},
        {"estimand_id": "cddf_bin",
         "definition": ("Posterior f(N,X) in each of the 13 reported 0.2-dex N_HI bins over "
                        "[19.7, 22.4), in three disclosure tiers "
                        "(calibrated / response_held / ceiling_adjacent)."),
         "units": "cm^2", "z_domain": "[2.0, 3.5]", "reported": True},
        {"estimand_id": "TAB10.L15",
         "definition": ("One-sided per-bin leverage on the science estimand of the SECOND "
                        "stationary posterior configuration found on real data: "
                        "(mirror configuration / pooled candidate - 1), per cent. Never added "
                        "in quadrature with the statistical interval; never double-counted "
                        "with L16."),
         "units": "per cent", "print_multiplier": 1, "z_domain": "all-z and B1..B5",
         "reported": True},
        {"estimand_id": "OMEGA_upper_limit_scan",
         "definition": ("Definition dependence of Omega_DLA on the upper integration limit: "
                        "median shift, in per cent, of Omega integrated to X instead of 21.6. "
                        "Explicitly NOT an uncertainty."),
         "units": "per cent", "reported": True},
        {"estimand_id": "census",
         "definition": "Counting quantities of the selected plane: sightlines, path, rows.",
         "units": "count or absorption distance", "reported": True},
    ]

    if frozen:
        snr = {
            "definition": ("MIXED PLANE. The numerator and every S/N-indexed calibration block "
                           "used SNR_REDSIDE = mean(flux*sqrt(ivar)) over unmasked rest-frame "
                           "[1420,1480] A; the DENOMINATOR was cut and stratified on "
                           "src_archive_catalog.npy::RED_SNR = float32(QSO-cat v3-altbal "
                           "SNR_REDSIDE), the MEDIAN of the same pixels."),
            "statistic": "mean",
            "code_ref": {"repo": "code", "file": "dlasearch.py", "lines": "670-676",
                         "commit": CODE_COMMIT},
            "threshold": 2.0, "strict": True,
            "strata_edges": [0, 1, 2, 3, 4, 5, 6, 7, "Infinity"],
            "legs": [
                {"leg": "row", "source_field": "SNR_REDSIDE (DLA catalogue column)",
                 "source_input_ref": "input:dlacat_real", "statistic": "mean",
                 "equality_gate": "the mean statistic, as the contract names it"},
                {"leg": "path",
                 "source_field": "src_archive_catalog.npy::RED_SNR (float32 of QSO-cat v3-altbal SNR_REDSIDE)",
                 "source_input_ref": "input:src_archive_catalog_npy", "statistic": "median",
                 "equality_gate": ("NONE. This is the 2026-09 defect: the path leg used a "
                                   "DIFFERENT statistic from the leg the contract names. "
                                   "GATE1 rev 2 s4/s5.")},
                {"leg": "calibration", "source_field": "snr_cat.fits (mock)",
                 "source_input_ref": "input:snr_cat_2lpt0", "statistic": "mean",
                 "equality_gate": "mean, consistent with the row leg and inconsistent with the path leg"},
            ],
        }
        population = {
            "definition": ("archive-builder searched population with --spectype QSO, "
                           "INTERSECT z_QSO in (2.0,4.25) strict INTERSECT BAL veto "
                           "(BI_CIV>0, v2-altbal) INTERSECT median RED_SNR > 2 "
                           "INTERSECT non-empty collared Lya window"),
            "n_sightlines": 410767,
            "spectype_cut": "QSO",
            "source_input_ref": "input:src_archive_catalog_npy",
        }
    else:
        snr = {
            "definition": ("SNR_REDSIDE = mean(flux*sqrt(ivar)) over unmasked pixels in "
                           "rest-frame [1420,1480] A inclusive, wave/(1+Z); per-spectrum, "
                           "DLA-uncorrelated red side. ONE statistic on ALL THREE legs."),
            "statistic": "mean",
            "code_ref": {"repo": "code", "file": "dlasearch.py", "lines": "670-676",
                         "commit": CODE_COMMIT},
            "threshold": 2.0, "strict": True,
            "strata_edges": [0, 1, 2, 3, 4, 5, 6, 7, "Infinity"],
            "legs": [
                {"leg": "row", "source_field": "SNR_REDSIDE (DLA catalogue column)",
                 "source_input_ref": "input:dlacat_real", "statistic": "mean",
                 "equality_gate": "bit-identical to the path leg on all 358,835 overlapping sightlines"},
                {"leg": "path", "source_field": "processed-main-dark-<HPX>.h5::snrs",
                 "source_input_ref": "input:searched_population_npz", "statistic": "mean",
                 "equality_gate": ("16,519 h5 files; reviewer re-derived bit-identically; "
                                   "1 unevaluated sentinel (BAL-vetoed), 9 NaN "
                                   "(no red-window pixels, no catalogue rows)")},
                {"leg": "calibration", "source_field": "snr_cat.fits (mock)",
                 "source_input_ref": "input:snr_cat_2lpt0", "statistic": "mean",
                 "equality_gate": "bit-identical to the mock dlacat on all 361,167 mock sightlines"},
            ],
        }
        population = {
            "definition": ("the finder's searched population (942,936 evaluated) INTERSECT "
                           "z_QSO in (2.0,4.25) strict INTERSECT BAL veto (BI_CIV>0, "
                           "v2-altbal) INTERSECT SNR_REDSIDE > 2 INTERSECT non-empty "
                           "collared Lya window"),
            "n_sightlines": 418034,
            "breakdown": {"QSO": 414058, "GALAXY": 3831, "STAR": 145},
            "spectype_cut": None,
            "source_input_ref": "input:searched_population_npz",
        }

    for leg in snr["legs"]:
        if leg["source_input_ref"] not in inputs_ids:
            gaps.add(f"contract.snr.legs[{leg['leg']}].source_input_ref",
                     f"{leg['source_input_ref']} is not an entry of inputs[]", "moderate")

    return {
        "contract_id": "C1" if not frozen else "FROZEN_2026-08-26",
        "contract_name": ("GATE1 s5 clean S/N plane" if not frozen
                          else "frozen mixed S/N plane (median denominator, QSO-only population)"),
        "adjudication": {
            "repo": "science",
            "path": str(CANDIDATE / "contract/GATE1_ADJUDICATION_2026-09-10.md"),
            "sha256": sha256_file(CANDIDATE / "contract/GATE1_ADJUDICATION_2026-09-10.md"),
            "section": "rev 2 s1,s4,s5",
        },
        "sample": "P1_PRIMARY_LYA",
        "source_contract": {
            "repo": "code",
            "path": "docs/CANONICAL_PURITY_COMPLETENESS_CONTRACT.json /sample_contract/P1_PRIMARY_LYA",
            "section": c1_sc.get("source_contract_version", ""),
        },
        "selection": {
            "snr": snr,
            "population": population,
            "cuts": {
                "z_qso_min": 2.0, "z_qso_max": 4.25,
                "p_dla_min": 0.99, "p_dla_strict": True,
                "quality": "DLAFLAG == 0",
                "bal_policy": "drop TARGETIDs with BI_CIV>0 (v2-altbal) + DLAFLAG POTENTIAL_BAL via DLAFLAG==0",
                "lambda_rf_window": c1_sc["lambda_rf_window"],
                "collar_kms": 3300.0,
                "differential_mask": [19.5, 19.7],
                "nhi_support": [19.0, 22.5],
            },
        },
        "estimands": estimands,
        "conventions": [
            {"repo": "notes",
             "path": "h=0.70, Omega_m=0.279, N_HI,max=22.4 (PI 2026-08-26 #42-#49)"},
            {"repo": "code", "path": "observable-only estimator, collar 3300 km/s (PI ckpt-10.8)"},
        ],
    }


# ---------------------------------------------------------------------------- side paths
FROZEN_CP3 = (FROZEN_ARCHIVE /
              "scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/real_pack_v2_20260821")


class Side:
    """Resolved, on-disk locations for one freeze."""

    def __init__(self, name):
        self.name = name
        if name == "frozen":
            self.pack = FROZEN_CP3 / "modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz"
            self.pack_sidecar = FROZEN_CP3 / "modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.selection_contract.json"
            self.pooled = FROZEN_CP3 / "cp3_real/POOLED_ln_real_v2_20260821.json"
            self.draws = FROZEN_CP3 / "cp3_real/POOLED_ln_real_v2_20260821_fdraws.npz"
            self.mirror = FROZEN_CP3 / "cp3_real/REAL_ln_deep_s20260826_fdraws.npz"
            self.confamb = FROZEN_CP3 / "cp3_real/CONFIG_AMBIGUITY_s26mirror_vs_pooled.json"
            self.zdomain = FROZEN_CP3 / "cp3_real/ZDOMAIN_estimands_pooled.json"
            self.reduction = CANDIDATE / "reductions/reduced_FROZEN_pooled.json"
            self.uls = CANDIDATE / "reductions/upper_limit_scan_FROZEN.json"
            self.fig41 = PAPER / "figures/current/data/fig_hbi_dndx.data.npz"
            self.fig40 = PAPER / "figures/current/data/fig_hbi_cddf.data.npz"
            self.frozen_status = FROZEN_CP3 / "cp3_real/FROZEN_STATUS.json"
            self.code_archive = None
            self.manifest_id = "LOWZ_FROZEN_2026-08-26"
            self.contract_id = "FROZEN_2026-08-26"
        else:
            self.pack = CANDIDATE / "pack/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.npz"
            self.pack_sidecar = CANDIDATE / "pack/modelA_pack_REAL_loa50k_c3300_bw0p2_pad19p0_molly172_v2.selection_contract.json"
            self.pooled = CANDIDATE / "inference/POOLED_ln_clean_C1.json"
            self.draws = CANDIDATE / "inference/POOLED_ln_clean_C1_fdraws.npz"
            self.mirror = CANDIDATE / "inference/REAL_ln_deep_s20260826_fdraws.npz"
            self.confamb = CANDIDATE / "systematics/validation/CONFIG_AMBIGUITY_clean_s26mirror_vs_pooled.json"
            self.zdomain = CANDIDATE / "systematics/validation/ZDOMAIN_estimands_pooled_clean.json"
            self.reduction = CANDIDATE / "reductions/reduced_CLEAN_C1_pooled.json"
            self.uls = CANDIDATE / "reductions/upper_limit_scan_CLEAN_C1.json"
            self.fig41 = CANDIDATE / "reductions/figure_data/fig_hbi_dndx.data.npz"
            self.fig40 = CANDIDATE / "reductions/figure_data/fig_hbi_cddf.data.npz"
            self.frozen_status = None
            self.code_archive = CANDIDATE / "code_env/frozen_code_1fd4828.tar.gz"
            self.manifest_id = "LOWZ_CLEAN_C1_2026-09-10"
            self.contract_id = "C1"
        self.ledger = NOTES_LEDGER if NOTES_LEDGER.exists() else (
            FROZEN_ARCHIVE / "home/mfho/desi_gpy_dla_notes/figures/2026-08-21_freeze_pathB/ledger_v2p3_cp3.json")
        self.is_frozen = name == "frozen"


# ---------------------------------------------------------------------------- products
def _json_schema(entries):
    return {"format": "json", "entries": entries}


def build_products(S: Side, gaps: Gaps):
    P = []
    superseded = (lambda pid: {"status": "superseded", "superseded_by": pid}) if S.is_frozen \
        else (lambda pid: {"status": "current"})

    def add(d):
        P.append(d)

    pack_sha = sha256_file(S.pack)
    pooled_sha = sha256_file(S.pooled)
    draws_sha = sha256_file(S.draws)
    mirror_sha = sha256_file(S.mirror)
    confamb_sha = sha256_file(S.confamb)
    red_sha = sha256_file(S.reduction)
    uls_sha = sha256_file(S.uls)
    fig41_sha = sha256_file(S.fig41)
    fig40_sha = sha256_file(S.fig40)
    ledger_sha = sha256_file(S.ledger)

    pack_upstream = ["input:dlacat_real", "input:qsocat_v2altbal", "input:molly_counts_172_npz",
                     "input:molly_tsv_nhi172", "input:adopted_response_npz",
                     "input:loa0_fp_product", "input:kernel_fit_ensemble_npz",
                     "input:ref_pack_2lpt0_v2p2", "input:canonical_contract"]
    if S.is_frozen:
        pack_upstream += ["input:src_archive_catalog_npy", "input:qsocat_v3altbal"]
        pack_producer = {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/extract_pack_real.py --real --stamp-v12",
                         "commit": "0babe21", "slurm_job": "58432277", "cpu_hours": 0.02}
        pack_interp = ("The frozen inference input of record for the 2026-08-26 freeze. Its "
                       "DENOMINATOR was cut and stratified on the archive MEDIAN RED_SNR while "
                       "its numerator and every S/N-indexed calibration block used the finder's "
                       "MEAN SNR_REDSIDE (GATE1 rev 2 s4). SUPERSEDED by the C1 candidate pack.")
    else:
        pack_upstream += ["input:searched_population_npz", "input:processed_store"]
        pack_producer = {"repo": "science", "script": "pack/tools/build_clean_pack.py",
                         "commit": "01a39a4ec5762492a6ae2cd9148341a8c30563b9", "cpu_hours": 0.1}
        pack_interp = ("CANDIDATE inference input of record for C1. Both legs cut and "
                       "stratified on the finder's mean SNR_REDSIDE; no SPECTYPE cut. "
                       "40 of 45 keys byte-identical to the frozen pack; 5 path-derived keys "
                       "rebuilt. Not a freeze until the PI rules.")

    add({"product_id": PACK, "kind": "pack",
         "location": location(S.pack, pack_sha, gaps, link="product:pack"),
         "sha256": pack_sha, "size_bytes": S.pack.stat().st_size,
         "file_schema": {"format": "npz", "entries": [
             {"name": "counts", "axes": ["c (N_HI bin)", "k (z bin)", "s (S/N stratum)"],
              "units": "count", "definition": "catalogue rows per (c,k,s) cell; 100,812 in window, 38,974 >= 20.3"},
             {"name": "dX", "axes": ["k (z bin)", "s (S/N stratum)"], "units": "absorption distance",
              "definition": "path integral per (k,s) cell over the selected sightlines"},
             {"name": "dX_coarse_committed", "axes": ["k", "s"], "units": "absorption distance",
              "definition": "coarse-bin path, committed at pack build"},
             {"name": "fp_E_alloc", "axes": ["c", "k", "s"], "units": "expected count",
              "definition": "false-positive allocation"},
             {"name": "fp_w_sightline_ratio", "axes": ["scalar"], "units": "dimensionless",
              "definition": "FP sightline weight ratio (path-derived)"},
             {"name": "fp_ell_eff", "axes": ["scalar"], "units": "dimensionless",
              "definition": "effective FP path (path-derived)"},
             {"name": "molly_n_det / molly_n_tot", "axes": ["c", "s"], "units": "count",
              "definition": "the mock completeness counts the likelihood consumes"}]},
         "quantity_definition": ("The HBI likelihood's sufficient statistics: counts, path, FP "
                                 "allocation and the frozen mock calibration blocks."),
         "estimand_refs": [], "selection_ref": S.contract_id,
         "producer": pack_producer,
         "config": {"source": str(S.pack_sidecar),
                    "params": {"collar_kms": 3300, "bw": 0.2, "pad": 19.0, "molly": 172,
                               "lya_only": True,
                               "n_sl": 410767 if S.is_frozen else 418034}},
         "upstream": pack_upstream,
         "upstream_hashes": {"dla_catalogue": "9a3f94ea50dd56210ae380668b0ac61e687e17af781218282398b6c8e26682d1"},
         "interpretation": pack_interp,
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="SELF_CONTAINED_BUNDLE_REBUILD",
                                    phase="phase1_paper_reproducibility",
                                    rationale="binned sufficient statistics; no per-object rows"),
         "validation_refs": ["gate:C1_closure", "gate:counts_bit_identity"] if not S.is_frozen else [],
         **superseded(PACK)})

    add({"product_id": "product:pack_sidecar_selection", "kind": "pack_sidecar",
         "location": location(S.pack_sidecar, sha256_file(S.pack_sidecar), gaps,
                              link="product:pack_sidecar_selection"),
         "sha256": sha256_file(S.pack_sidecar), "file_schema": _json_schema([
             {"name": "snr_field", "definition": "the S/N statistic the contract names"},
             {"name": "path_leg_sources_sha256", "definition": "content hashes of every leg source"}]),
         "quantity_definition": "The pack's numeric selection thresholds, as an additive sidecar.",
         "selection_ref": S.contract_id,
         "producer": {"repo": "science",
                      "script": "write_selection_contract_sidecar.py" if S.is_frozen else "pack/tools/write_sidecars.py",
                      "commit": "_GAP" if S.is_frozen else "01a39a4ec5762492a6ae2cd9148341a8c30563b9"},
         "config": {"source": "its own header"},
         "upstream": [PACK],
         "interpretation": "Binds the numeric thresholds to the pack by contract_id + sha256.",
         "release_eligibility": DOC, **superseded("product:pack_sidecar_selection")})

    add({"product_id": POST, "kind": "pooled_posterior",
         "location": location(S.pooled, pooled_sha, gaps, link="product:pooled_posterior"),
         "sha256": pooled_sha, "size_bytes": S.pooled.stat().st_size,
         "file_schema": _json_schema([
             {"name": "estimand", "definition": "POSTERIOR_MEDIAN_CI (committed reduce_f_posterior)"},
             {"name": "selection.included", "definition": "the 6 pooled seeds"},
             {"name": "reporting_bins", "axes": ["k"], "definition": "B1..B5 edges"},
             {"name": "thresholds", "units": "log10 N_HI", "definition": "[20.0, 20.3]"}]),
         "quantity_definition": "Pooled HBI posterior summary over 6 included seeds, 6000 equal-weight draws.",
         "estimand_refs": ["dndx_20p3_allz", "dndx_20p0_allz", "omega_20p3_21p6_allz", "cddf_bin"],
         "selection_ref": S.contract_id,
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/cc_pool_posterior.py select_runs",
                      "commit": "b59e0b5" if S.is_frozen else CODE_COMMIT,
                      "cpu_hours": None if S.is_frozen else 2.9},
         "config": {"source": str(S.pooled), "n_draws": 6000,
                    "params": {"pool": "21d,22,24,25,27,28d", "excluded": "23,26",
                               "rule": "split_rhat<=1.10 with one permitted deep rerun"}},
         "upstream": [PACK], "upstream_hashes": {"pack": pack_sha},
         "interpretation": ("The ADOPTED (dominant) configuration. Reviewer F1 (inherited): "
                            "cc_real_posterior.py:155-170 stores an UNSPLIT 2-chain Gelman-Rubin "
                            "as split_rhat, on BOTH planes."),
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="FROZEN_HASH_PINNED",
                                    phase="phase1_paper_reproducibility"),
         **superseded(POST)})

    add({"product_id": DRAWS, "kind": "pooled_draws",
         "location": location(S.draws, draws_sha, gaps, link="product:pooled_draws"),
         "sha256": draws_sha, "size_bytes": S.draws.stat().st_size,
         "file_schema": {"format": "npz", "entries": [
             {"name": "f", "axes": ["draw", "c (N_HI bin)", "k (z bin)"],
              "shape": [6000, "n_c", "n_k"], "units": "latent f (CDDF amplitude per cell)",
              "definition": "the pooled latent draws every reduction integrates"},
             {"name": "ntrue_edges", "axes": ["c+1"], "units": "log10 N_HI",
              "definition": "latent N_HI bin edges"},
             {"name": "zf_edges", "axes": ["k+1"], "units": "redshift",
              "definition": "latent z bin edges"}]},
         "quantity_definition": "6000 equal-weight pooled latent CDDF draws.",
         "estimand_refs": ["dndx_20p3_allz", "dndx_20p0_allz", "omega_20p3_21p6_allz", "cddf_bin"],
         "selection_ref": S.contract_id,
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/cc_pool_posterior.py",
                      "commit": "b59e0b5" if S.is_frozen else CODE_COMMIT},
         "config": {"source": "POOL_summary.txt", "n_draws": 6000},
         "upstream": [PACK], "upstream_hashes": {"pack": pack_sha},
         "interpretation": "The draws of record for every reduction on this plane.",
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="FROZEN_HASH_PINNED",
                                    phase="phase1_paper_reproducibility"),
         **superseded(DRAWS)})

    add({"product_id": "product:mirror_carrier", "kind": "per_seed_draws",
         "location": location(S.mirror, mirror_sha, gaps, link="product:mirror_carrier"),
         "sha256": mirror_sha, "size_bytes": S.mirror.stat().st_size,
         "file_schema": {"format": "npz", "entries": [
             {"name": "f", "axes": ["draw", "c", "k"], "units": "latent f",
              "definition": "seed-20260826 deep-rerun draws; chain 0 is the mirror mode"}]},
         "quantity_definition": "The L15 mirror carrier. EXCLUDED from the pool (split_rhat 12.404).",
         "estimand_refs": ["TAB10.L15"], "selection_ref": S.contract_id,
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/cc_real_posterior.py",
                      "commit": CODE_COMMIT},
         "config": {"source": str(S.mirror.with_suffix("").as_posix()).replace("_fdraws", "") + ".json",
                    "seed": 20260826, "params": {"deep": True, "n_chains": 2}},
         "upstream": [PACK], "upstream_hashes": {"pack": pack_sha},
         "interpretation": ("Excluded from the science posterior; retained ONLY as the L15 "
                            "upper-bound carrier. Must never be pooled."),
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="FROZEN_HASH_PINNED",
                                    phase="phase1_paper_reproducibility"),
         **superseded("product:mirror_carrier")})

    add({"product_id": "product:l15_config_ambiguity", "kind": "systematics_product",
         "location": location(S.confamb, confamb_sha, gaps, link="product:l15_config_ambiguity"),
         "sha256": confamb_sha, "size_bytes": S.confamb.stat().st_size,
         "file_schema": _json_schema([
             {"name": "result.ge20.3.allz_pct", "units": "per cent",
              "definition": "(mirror/pooled - 1)*100 for dN/dX(>=20.3) all-z",
              "estimand_ref": "TAB10.L15"},
             {"name": "result.ge20.3.bins_pct", "axes": ["z_bin"],
              "axis_values": {"z_bin": BINS}, "units": "per cent", "definition": "same, per bin"},
             {"name": "result.ge20.0.allz_pct", "units": "per cent",
              "definition": "same at the >=20.0 threshold"},
             {"name": "run_diagnostics.split_rhat", "definition": "the mirror run's convergence posture"},
             {"name": "run_diagnostics.mean_potential_energy_per_chain", "units": "nats",
              "definition": "the PE gap that identifies a MODE, not a stuck chain"}]),
         "quantity_definition": "The L15 configuration-ambiguity measurement: mirror carrier vs pooled.",
         "estimand_refs": ["TAB10.L15"], "selection_ref": S.contract_id,
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/cc_config_ambiguity.py",
                      "commit": "7e49c86",
                      "lines": ":66 estimand weights = pk['dX'].sum(axis=1), a path-derived key"},
         "config": {"source": "its own header", "params": {"chain": 0, "n_chains": 2}},
         "upstream": ["product:mirror_carrier", POST, PACK],
         "upstream_hashes": {"mirror_carrier": mirror_sha, "pack": pack_sha},
         "interpretation": ("RECLASSIFY, not a mechanical refresh: the mirror CARRIER is a "
                            "different chain on the two planes, and no chain sits at t ~ 0, so "
                            "the ledger's 't ~ 0' wording is a defect on BOTH planes."),
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="SCIENCE_REDERIVABLE_FROM_PUBLIC_ASSETS",
                                    phase="phase1_paper_reproducibility"),
         "validation_refs": ["gate:mirror_is_a_mode"],
         **superseded("product:l15_config_ambiguity")})

    if S.zdomain.exists():
        z_sha = sha256_file(S.zdomain)
        add({"product_id": "product:zdomain_estimands", "kind": "systematics_product",
             "location": location(S.zdomain, z_sha, gaps, link="product:zdomain_estimands"),
             "sha256": z_sha, "file_schema": _json_schema([
                 {"name": "estimands", "axes": ["z_lo"], "axis_values": {"z_lo": [2.0, 2.3, 2.56]},
                  "units": "dimensionless", "definition": "z-domain-restricted dN/dX"},
                 {"name": "config_leverage_pct", "axes": ["z_lo"], "units": "per cent",
                  "definition": "the mirror leverage restricted to z >= 2.0 / 2.3 / 2.56; the "
                                "field sample631.tex:2253 prints at p=2"}]),
             "quantity_definition": "z-domain restricted estimands from the pooled draws (PI 2026-08-21 #27).",
             "estimand_refs": ["TAB10.L15"], "selection_ref": S.contract_id,
             "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/cc_zdomain_estimands.py",
                          "commit": CODE_COMMIT},
             "config": {"source": "its own header"},
             "upstream": [DRAWS, PACK],
             "interpretation": "The p=2 carrier of the L15 literal at sample631.tex:2253.",
             "release_eligibility": AGG, **superseded("product:zdomain_estimands")})

    red_prod = {"repo": "science",
                "script": "reductions/tools/reduce.py (wraps paper_figures/hbi_reduction.py, unchanged)",
                "commit": gaps.add("product:reduction_pooled.producer.commit",
                                   "reductions/tools/reduce.py is vendored by sha "
                                   "2ce58a2b60009be7a5558134fb1bfbe38fe16076da05a6c1142345d61275c40d "
                                   "and has no git commit of its own; the hbi_reduction.py it "
                                   "wraps is pinned only by the paper-repo commit",
                                   "moderate",
                                   fillable_from="paper repo commit at the R-042a run")}
    add({"product_id": "product:reduction_pooled", "kind": "reduction",
         "location": location(S.reduction, red_sha, gaps, link="product:reduction_pooled"),
         "sha256": red_sha, "size_bytes": S.reduction.stat().st_size,
         "file_schema": _json_schema([
             {"name": "quantities.<estimand_id>", "axes": ["quantile"],
              "axis_values": {"quantile": [2.5, 16.0, 50.0, 84.0, 97.5]}, "shape": [5],
              "dtype": "float64",
              "units": "per estimand (dN/dX dimensionless; Omega dimensionless, printed x1e4)",
              "definition": "posterior quantiles of the named estimand"},
             {"name": "cddf_bins", "definition": "13 bins with tier in {calibrated, response_held, ceiling_adjacent}"},
             {"name": "closure_vs_summary", "units": "relative",
              "definition": "max_rel_diff of the reduction against the pooled summary"}]),
         "quantity_definition": "Every reported low-z estimand, reduced from the pooled draws by the paper's own hbi_reduction.py, UNCHANGED.",
         "estimand_refs": ["dndx_20p3_allz", "dndx_20p0_allz", "omega_20p3_21p6_allz", "cddf_bin"],
         "selection_ref": S.contract_id, "producer": red_prod,
         "config": {"source": str(S.reduction) + "::inputs",
                    "params": {"quantile_order": [2.5, 16.0, 50.0, 84.0, 97.5],
                               "h": 0.70, "Omega_m": 0.279}},
         "upstream": [DRAWS, POST, PACK],
         "upstream_hashes": {"draws": draws_sha, "pack": pack_sha, "summary": pooled_sha},
         "interpretation": (("RECONSTRUCTION ARTIFACT: this reduction of the FROZEN draws was "
                             "produced on 2026-09-10 by the R-042a vehicle, not at the "
                             "2026-08-26 freeze, which emitted no reduction JSON. Its inputs are "
                             "the frozen triple, so the numbers are the frozen numbers.")
                            if S.is_frozen else
                            "The NEW side of the D1 old-vs-new comparison. Closure against the "
                            "pooled summary is at machine precision."),
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="SCIENCE_REDERIVABLE_FROM_PUBLIC_ASSETS",
                                    phase="phase1_paper_reproducibility"),
         "validation_refs": ["gate:closure_vs_summary"], **superseded("product:reduction_pooled")})

    add({"product_id": "product:upper_limit_scan", "kind": "reduction",
         "location": location(S.uls, uls_sha, gaps, link="product:upper_limit_scan"),
         "sha256": uls_sha, "file_schema": _json_schema([
             {"name": "upper_limit_scan.<X>.median_shift_pct", "units": "per cent",
              "definition": "median shift of Omega integrated to X instead of 21.6",
              "estimand_ref": "OMEGA_upper_limit_scan"},
             {"name": "upper_limit_scan.<X>.halfwidth68_pct", "units": "per cent",
              "definition": "fractional 68 per cent posterior half-width at that limit"},
             {"name": "upper_limit_scan.<X>.median_over_adopted", "units": "dimensionless",
              "definition": "ratio to the adopted [20.3,21.6] integral"}]),
         "quantity_definition": "The upper-integration-limit scan, re-run with omega_anatomy.py @fb0a740 imported verbatim.",
         "estimand_refs": ["OMEGA_upper_limit_scan"], "selection_ref": S.contract_id,
         "producer": {"repo": "science", "script": "reductions/tools/upper_limit_scan.py",
                      "commit": "fb0a740"},
         "config": {"source": str(S.uls) + "::inputs"},
         "upstream": [DRAWS, PACK],
         "upstream_hashes": {"draws": draws_sha, "pack": pack_sha},
         "interpretation": ("Reproduces the frozen R-037 upper-limit row to 0.0 per cent on the "
                            "frozen side, which is what licenses it as the old-side carrier."),
         "release_eligibility": AGG, **superseded("product:upper_limit_scan")})

    fig41_prov = (CANDIDATE / "reductions/figure_data/fig_hbi_dndx.provenance.json")
    add({"product_id": "product:fig41_array", "kind": "figure_array",
         "location": location(S.fig41, fig41_sha, gaps, link="product:fig41_array"),
         "sha256": fig41_sha, "size_bytes": S.fig41.stat().st_size,
         "file_schema": npz_file_schema(S.fig41, FIG41_EDGES),
         "quantity_definition": "The backing array of FIG-41 and the sole numeric source of TAB-12.",
         "estimand_refs": ["dndx_20p3_allz", "dndx_20p0_allz", "omega_20p3_21p6_allz"],
         "selection_ref": S.contract_id,
         "producer": {"repo": "paper", "script": "paper_figures/fig_hbi_dndx.py",
                      "commit": "74f2b3378aeaa5fb053549822e256b7a520e83cb",
                      "command": "python3 paper_figures/fig_hbi_dndx.py"},
         "config": {"source": str(fig41_prov) if not S.is_frozen else str(S.fig41.with_suffix("")) + ".provenance.json",
                    "params": {"paper_repo_commit_at_generation": PAPER_COMMIT_AT_FIGDATA}},
         "upstream": [POST, DRAWS, PACK, LEDGER, HZ2, LIT, ENVP, CONV],
         "upstream_hashes": {"hbi_posterior": pooled_sha, "hbi_posterior_draws": draws_sha,
                             "real_pack": pack_sha, "ledger": ledger_sha,
                             "hz2_posterior": HZ2_POOLED_SHA256},
         "interpretation": ("FROZEN" if S.is_frozen else
                            "Stamped status=PROVISIONAL; the rendered figure carries a CANDIDATE "
                            "banner. The 44 high-z / literature / convention arrays are "
                            "byte-identical to the frozen version."),
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="SELF_CONTAINED_BUNDLE_REBUILD",
                                    phase="phase1_paper_reproducibility"),
         "validation_refs": ["gate:D2_array_diff"], **superseded("product:fig41_array")})

    add({"product_id": "product:fig40_array", "kind": "figure_array",
         "location": location(S.fig40, fig40_sha, gaps, link="product:fig40_array"),
         "sha256": fig40_sha, "size_bytes": S.fig40.stat().st_size,
         "file_schema": npz_file_schema(S.fig40, FIG40_EDGES),
         "quantity_definition": "The backing array of FIG-40 (the CDDF figure) and of TAB-13 downstream.",
         "estimand_refs": ["cddf_bin", "omega_20p3_21p6_allz"], "selection_ref": S.contract_id,
         "producer": {"repo": "paper", "script": "paper_figures/fig_hbi_cddf.py",
                      "commit": PAPER_COMMIT_AT_FIGDATA,
                      "command": "python3 paper_figures/fig_hbi_cddf.py"},
         "config": {"source": str(S.fig40.with_suffix("")) + ".provenance.json"},
         "upstream": [POST, DRAWS, PACK, LEDGER, PYIGM, ENVP, LIT, CONV],
         "upstream_hashes": {"hbi_posterior": pooled_sha, "hbi_posterior_draws": draws_sha,
                             "real_pack": pack_sha, "ledger": ledger_sha,
                             "pyigm_reference_r034": R034_NPZ_SHA256},
         "interpretation": "FROZEN" if S.is_frozen else "Stamped status=PROVISIONAL.",
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="SELF_CONTAINED_BUNDLE_REBUILD",
                                    phase="phase1_paper_reproducibility"),
         "validation_refs": ["gate:D2_array_diff"], **superseded("product:fig40_array")})

    # --- products that are NOT expected to move under a low-z freeze -----------------
    add({"product_id": LEDGER, "kind": "ledger",
         "location": location(S.ledger, ledger_sha, gaps, link="product:ledger"),
         "sha256": ledger_sha, "size_bytes": S.ledger.stat().st_size,
         "file_schema": _json_schema([
             {"name": "lines.L15_CONFIG_AMBIGUITY.mirror_vs_pooled_pct", "units": "per cent",
              "definition": "per-threshold, per-bin mirror-vs-pooled leverage (TAB-10 prints p=1)",
              "estimand_ref": "TAB10.L15"},
             {"name": "lines.L15_CONFIG_AMBIGUITY.leverage_by_z_domain_pct", "units": "per cent",
              "definition": "the same leverage restricted to z >= 2.0/2.3/2.56 (sample631.tex:2253 prints p=2)"},
             {"name": "lines.L2_MOCK2REAL_C_TRANSPORT.{ge20.0_band,ge20.3_band}",
              "units": "multiplicative", "definition": "the T_tr transport band"}]),
         "quantity_definition": ("The systematics ledger, revision v2.3r5: 19 rows, 85 fields, "
                                 "20 numeric-value carriers -- all STRINGS holding ranges."),
         "selection_ref": "FROZEN_2026-08-26",
         "producer": {"repo": "notes", "script": "hand-maintained (no producer script)",
                      "commit": gaps.add(
                          "product:ledger.producer",
                          "The systematics ledger has NO producer script. It is hand-edited, its "
                          "20 numeric fields are strings holding ranges, and there is no machine "
                          "link from a ledger row to the posterior it was measured on. This is "
                          "the largest single traceability break in the chain.", "MAJOR",
                          fillable_from="nothing on disk; the fix is the proposed ledger_quantities.json")},
         "config": {"source": "hand-maintained"},
         # The ledger had NO machine link to any posterior until the D1 dependency trace
         # (systematics/TAB10_DEPENDENCY_2026-09-10.json) recorded, per line, which artifact
         # each numeric row was measured on. Those edges are recorded here for the first time;
         # on the frozen side they are old_side_unrecorded, not "none".
         "upstream": ([] if S.is_frozen else
                      ["product:l15_config_ambiguity", "product:h2m_results",
                       "product:mock_envelope", "product:r037_omega_anatomy", PACK]),
         "interpretation": ("The values of record for TAB-10 at the 2026-08-26 freeze."
                            if S.is_frozen else
                            "UNCHANGED BYTES under C1 and therefore STALE: revision must become "
                            "v2.4 and every REAL_POSTERIOR_DEPENDENT row must be refreshed or "
                            "reclassified."),
         "release_eligibility": rel("public_release_eligible", "documentation",
                                    release_class="FROZEN_HASH_PINNED",
                                    scrub_required=["absolute private paths (release_scrub.py / SCRUB_RECORD.json)"]),
         "status": "current"})

    add({"product_id": HZ2, "kind": "pooled_posterior",
         "location": {"path": "/nfs/turbo/lsa-cavestru/mfho/paper1_frozen_2026-09-04_highz "
                              "(HZ2 pooled posterior; pinned in paper_figures/common.py:80)",
                      "location_class": "durable_turbo"},
         "sha256": HZ2_POOLED_SHA256,
         "file_schema": _json_schema([
             {"name": "measurement.{20.0,20.3}.dndx", "axes": ["quantile"], "units": "dimensionless",
              "definition": "the high-z arm's incidence, posterior median + quantiles"},
             {"name": "pack_sha256", "definition": f"the HZ2 pack identity, {HZ2_PACK_SHA256}"}]),
         "quantity_definition": "The high-redshift (BH) arm, frozen 2026-09-04. Not touched by a low-z freeze.",
         "estimand_refs": [], "selection_ref": "HZ2_2026-09-04",
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/cc_pool_posterior.py (high-z arm)",
                      "commit": "2a8652f97ad04dc6f1d4c4d7f15c60469892bc4f"},
         "config": {"source": "paper_figures/common.py:71-81 + common.require_hz2()"},
         "upstream": [],
         "interpretation": ("The BH row of TAB-12 and every highz_* key of FIG-41 rest on this. "
                            "A low-z contract break does NOT reach it: there is no recorded edge."),
         "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                    release_class="FROZEN_HASH_PINNED"),
         "status": "current"})

    add({"product_id": LIT, "kind": "comparison_product",
         "location": {"path": "DLA_data (external, pinned by git commit)", "location_class": "external"},
         "sha256": gaps.add("product:literature_dla_data.sha256",
                            "The Ho+21 / literature comparison layer is pinned by the DLA_data "
                            f"git commit {DLA_DATA_COMMIT}, never by a file hash.", "minor",
                            fillable_from="hash the checked-out DLA_data tree"),
         "file_schema": _json_schema([{"name": "ho21 dndx/lo68/hi68/z", "axes": ["z"],
                                       "units": "dimensionless",
                                       "definition": "Ho, Bird & Garnett 2021 observed-column dN/dX"}]),
         "quantity_definition": "External literature comparison layer.",
         "selection_ref": "n/a (third party)",
         "producer": {"repo": "code", "script": "paper_figures/comparison_data.py",
                      "commit": DLA_DATA_COMMIT},
         "config": {"source": "paper_figures/comparison_data.py pin_inputs()"},
         "upstream": [], "interpretation": "Presentation-only comparison; never a consistency test.",
         "release_eligibility": rel("public_release_eligible", "third_party"),
         "status": "current"})

    add({"product_id": PYIGM, "kind": "comparison_product",
         "location": {"path": "R-034 pyigm default CDDF product (registered in common.ARTIFACTS)",
                      "location_class": "external"},
         "sha256": R034_NPZ_SHA256,
         "file_schema": {"format": "npz", "entries": [
             {"name": "logN / f_per_N", "axes": ["nhi"], "units": "cm^2",
              "definition": "the pyigm default reference spine"}]},
         "quantity_definition": "The R-034 pyigm default reference layer of FIG-40.",
         "selection_ref": "n/a (third party)",
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/pyigm_default_cddf.py",
                      "commit": "2c2032118e33c073c391eef1b8999f42be147dbe"},
         "config": {"source": "paper_figures/pyigm_reference.py:63-65"},
         "upstream": [], "interpretation": "Reference layer only.",
         "release_eligibility": rel("public_release_eligible", "third_party"),
         "status": "current"})

    env_path = (FROZEN_ARCHIVE / "home/mfho/desi_gpy_dla_notes/figures/2026-08-26_sys_viz_preview/cddf_recovery_audit.json")
    add({"product_id": ENVP, "kind": "validation_product",
         "location": location(env_path, sha256_file(env_path) if env_path.exists() else None,
                              gaps, link="product:mock_envelope"),
         "sha256": sha256_file(env_path) if env_path.exists() else "_GAP: not present",
         "file_schema": _json_schema([{"name": "ratio_lo/ratio_hi/bias_min/bias_max",
                                       "axes": ["nhi_bin"], "units": "dimensionless (ratio)",
                                       "definition": "mock-recovery envelope per N_HI bin"}]),
         "quantity_definition": "The mock-recovery envelope drawn on FIG-40/41 (cddf_recovery_audit).",
         "selection_ref": "mock calibration plane",
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/cddf_recovery_audit.py",
                      "commit": "08504c0"},
         "config": {"source": "its own header"},
         "upstream": [], "interpretation": "Mock-only; a low-z real-data freeze does not move it.",
         "release_eligibility": AGG, "status": "current"})

    if S.code_archive:
        ca_sha = sha256_file(S.code_archive)
        add({"product_id": "product:code_archive", "kind": "code_archive",
             "location": location(S.code_archive, ca_sha, gaps, link="product:code_archive"),
             "sha256": ca_sha, "size_bytes": S.code_archive.stat().st_size,
             "file_schema": {"format": "tar.gz"},
             "quantity_definition": f"git archive of the inference code at {CODE_COMMIT[:7]} ({CODE_TAG}).",
             "selection_ref": S.contract_id,
             "producer": {"repo": "science", "script": "git archive", "commit": CODE_COMMIT},
             "config": {"source": "CANDIDATE_IDENTITY.json"},
             "upstream": [],
             "interpretation": "The code of record, vendored so the chain does not depend on repo state.",
             "release_eligibility": rel("public_release_eligible", "code",
                                        phase="phase1_paper_reproducibility"),
             "status": "current"})
    else:
        gaps.add("frozen:code_archive",
                 "The 2026-08-26 freeze vendored no git archive of the inference code; its code "
                 "identity rests on FROZEN_STATUS.code_at_freeze ('hbi/forward-2026-08 @2d4035e') "
                 "and on the tag alone.", "moderate",
                 fillable_from="git archive 1fd4828, as the C1 candidate did")

    # --- comparison / dependency products that exist only on the candidate side -------
    if not S.is_frozen:
        for pid, relpath, kind, qdef in (
            ("product:tab10_dependency", "systematics/TAB10_DEPENDENCY_2026-09-10.json",
             "comparison_product",
             "The dependency and disposition map for all 22 low-z TAB-10 lines. PROVENANCE-ONLY."),
            ("product:d2_figdata_diff", "comparison/D2_figdata_diff.json", "comparison_product",
             "Array-level old-vs-new diff of FIG-40/41 and the 21 TAB-12 cells."),
            ("product:d1_reduction_comparison", "comparison/D1_reduction_comparison.json",
             "comparison_product",
             "Old-vs-new comparison on all 30 reduced low-z quantities."),
            ("product:shift_decomposition", "comparison/SHIFT_DECOMPOSITION.json",
             "comparison_product",
             "Attribution of the C1 shift to the S/N-definition repair and the SPECTYPE inclusion."),
        ):
            p = CANDIDATE / relpath
            s = sha256_file(p)
            add({"product_id": pid, "kind": kind,
                 "location": location(p, s, gaps, link=pid), "sha256": s,
                 "size_bytes": p.stat().st_size,
                 "file_schema": _json_schema([{"name": "(see the document's own header)",
                                               "definition": qdef}]),
                 "quantity_definition": qdef, "selection_ref": "C1",
                 "producer": {"repo": "science", "script": relpath.split("/")[-1] + " (worker product)",
                              "commit": gaps.add(f"{pid}.producer.commit",
                                                 f"{relpath} is a worker product with no git commit; "
                                                 f"it is pinned by sha256 only", "minor")},
                 "config": {"source": "its own header"},
                 "upstream": [PACK, DRAWS, "product:reduction_pooled"],
                 "interpretation": "Diagnostic/provenance product; carries no approved number.",
                 "release_eligibility": DOC, "status": "current"})

    # --- the systematics campaign products (unpinned on BOTH sides) -------------------
    CAMPAIGNS = [
        ("product:h2m_results", "h2m_results.json", ["TAB10.L2", "TAB10.L10"], "input:h2m_substrate",
         "H2-M real-spectrum injection campaign aggregate (540 real sightlines, 900 injections)."),
        ("product:cleanreal_results", "cleanreal_results.json", ["TAB10.L2"], "input:h2mc_substrate",
         "Clean-arm real-spectrum campaign aggregate (875 records; tid/snr/z already scrubbed once)."),
        ("product:mockinj_results", "mockinj_results.json", ["TAB10.L2"], "input:mock_dlacat_2lpt0",
         "Mock-injection campaign aggregate (900 records; tid already dropped)."),
        ("product:rinj_decomposition", "rinj_decomposition.json", ["TAB10.L2"],
         "product:h2m_results",
         "R_inj decomposition of the transport band (ledger L2 basis line: 'H2-M decomposed "
         "by R_inj; iso x1.057, corrected x1.070')."),
        ("product:weight_target", "weight_target.json", ["TAB10.L2"], "input:molly_counts_172_npz",
         "The stratified weight target (MEDIAN plane today; a mean-plane version is required "
         "for any L2 refresh)."),
        ("product:l8ext_results", "l8ext_results.json", ["TAB10.L8"], "input:l8real_substrate",
         "L8 extended-injection aggregate over 20.0-20.3."),
        ("product:cleanarm_z0_failure_census", "cleanarm_z0_failure_census.json", ["TAB10.L2"],
         "product:cleanreal_results", "Clean-arm z0 failure census."),
    ]
    for pid, fname, est, substrate, qdef in CAMPAIGNS:
        # Edges recorded by the Phase-0b shipping audit (section 2e/L8e) and the ledger's own
        # L2 basis line. Unrecorded at the 2026-08-26 freeze, hence old-side empty.
        upstream = [] if S.is_frozen else ([substrate] if substrate else [])
        sha = gaps.add(f"{pid}.sha256",
                       f"{fname} has NO hash pin in the paper repo, NO entry in "
                       f"release_staging/data_v1 and no durable path recorded anywhere, yet "
                       f"shipping TAB-10 rows and shipping figures depend on it.", "MAJOR",
                       fillable_from="the ckpt-10.5 campaign directories on /scratch")
        add({"product_id": pid, "kind": "campaign_product",
             "location": {"path": f"_GAP: /scratch .../h2m_ckpt10p5_20260817/... {fname}",
                          "location_class": "scratch",
                          "durable_copy_ref": "_GAP: no durable copy exists"},
             "sha256": sha,
             "file_schema": _json_schema([
                 {"name": "per", "definition": "per-injection records; carries TARGETID -> object_level_real"},
                 {"name": "by_logN", "axes": ["logN bin"], "units": "dimensionless (ratio)",
                  "definition": "aggregate transport ratio by column density"},
                 {"name": "by_cell", "axes": ["s (S/N stratum)", "z bin"],
                  "units": "dimensionless (ratio)",
                  "definition": "the MEDIAN-stratified cells -- the labelling C1 voids"}]),
             "quantity_definition": qdef, "estimand_refs": est,
             "selection_ref": "FROZEN_2026-08-26 (median plane)",
             "producer": {"script": f"_GAP: builder for {fname} is not pinned in the paper repo",
                          "commit": "_GAP"},
             "config": {"source": "_GAP"},
             "upstream": upstream,
             "interpretation": ("MIXED. The substrate was SELECTED and STRATIFIED on the archive "
                                "MEDIAN RED_SNR against a mean-indexed mock completeness surface, "
                                "so CHECKPOINT10P5.md:53's 'MATCHED (within-SNR-bin)' verdict is "
                                "VOID on the C1 contract. A mean-plane refresh costs >= 50 core-h."),
             "release_eligibility": rel("public_release_eligible", "aggregate_reduced",
                                        restricted_through=["TARGETID"],
                                        release_class="RESEARCH_ENVIRONMENT_ONLY",
                                        phase="phase1_paper_reproducibility",
                                        rationale="the aggregate is releasable; its per-record "
                                                  "substrate is object_level_real and must be scrubbed"),
             "status": "current"})

    add({"product_id": "product:r037_omega_anatomy", "kind": "systematics_product",
         "location": {"path": "_GAP: R037_omega_anatomy.json (frozen product; ASSET-04)"
                              if S.is_frozen else
                              "_GAP: NOT re-run on the C1 plane; the upper-limit scan was, the "
                              "response-family treatments and the L15 Omega leverage were not",
                      "location_class": "scratch",
                      "durable_copy_ref": "_GAP: no durable copy recorded"},
         "sha256": (R037_SHA256 if S.is_frozen else gaps.add(
             "product:r037_omega_anatomy on the C1 plane",
             "R-037 has NOT been re-run on the C1 plane: omega_anatomy.py was imported verbatim "
             "for the upper-limit scan only. The Omega-L15 mirror leverage and the "
             "response-family band (sample631.tex:2257 and the second bracket of :2411) "
             "therefore have a replacing quantity but no value yet. Its frozen counterpart is "
             f"pinned at emit_tab_systematics.py:52 ({R037_SHA256[:12]}) but at no durable path.",
             "MAJOR",
             fillable_from="a 0.2 core-h re-run of CDDF_analysis/hbi_mcmc/omega_anatomy.py @fb0a740 "
                           "on the new draws + new pack, which TAB-10 already classifies "
                           "refresh_mechanical")),
         "file_schema": _json_schema([
             {"name": "real.allz.omega_adopted.halfwidth68_pct", "units": "per cent",
              "definition": "the Omega statistical half-width TAB-10 prints"},
             {"name": "real.allz.treatments.L15_mirror_configuration.median_shift_pct",
              "units": "per cent", "definition": "the Omega leverage of the mirror configuration"},
             {"name": "real.allz.upper_limit_scan.<X>.median_shift_pct", "units": "per cent",
              "definition": "the upper-integration-limit row"}]),
         "quantity_definition": "The Omega anatomy (R-037): the authority for four TAB-10 Omega rows.",
         "estimand_refs": ["OMEGA_upper_limit_scan", "TAB10.L15"],
         "selection_ref": S.contract_id,
         "producer": {"repo": "code", "script": "CDDF_analysis/hbi_mcmc/omega_anatomy.py",
                      "commit": "fb0a740"},
         "config": {"source": "_GAP"},
         "upstream": [DRAWS, PACK, "product:mirror_carrier"],
         "interpretation": ("On the C1 side the upper-limit rows are carried by "
                            "product:upper_limit_scan (omega_anatomy.py imported verbatim); the "
                            "L15 Omega leverage and the response-family band are NOT yet re-run."),
         "release_eligibility": AGG, **superseded("product:r037_omega_anatomy")})

    add({"product_id": "product:l8real_sightlines", "kind": "injection_substrate",
         "location": {"path": "_GAP: .../h2m_ckpt10p5_20260817/l8ext/l8real_sightlines.csv",
                      "location_class": "scratch",
                      "durable_copy_ref": "_GAP: no durable copy exists"}
         if S.is_frozen else location(
             "/scratch/cavestru_root/cavestru0/mfho/h2m_ckpt10p5_20260817/l8ext/l8real_sightlines.csv",
             "8fc4fbc602477d92bb731646f277c1c5594b3ba10b2e26bbe88e30654c5e1718", gaps,
             link="product:l8real_sightlines"),
         "sha256": ("8fc4fbc602477d92bb731646f277c1c5594b3ba10b2e26bbe88e30654c5e1718"
                    if not S.is_frozen else
                    gaps.add("frozen:l8real_sightlines.sha256",
                             "the frozen freeze recorded no hash for the L8 substrate; its "
                             "identity was first pinned by OPUS-E on 2026-09-10", "moderate")),
         "file_schema": {"format": "csv", "entries": [
             {"name": "TARGETID", "definition": "per-object identifier"},
             {"name": "RED_SNR", "units": "dimensionless",
              "definition": "the MEDIAN statistic -- the defect column"}]},
         "quantity_definition": ("The 1,206-sightline L8 injection substrate. Under the mean plane "
                                 "57 rows (4.73%) change stratum and 11 were never eligible."),
         "estimand_refs": ["TAB10.L8"], "selection_ref": "FROZEN_2026-08-26 (median plane)",
         "producer": {"script": "build_l8ext.py real (sha 07e7fe97)", "commit": "_GAP"},
         "config": {"source": "l8ext/l8real_100k.env"},
         "upstream": ["input:dlacat_real"] if not S.is_frozen else [],
         "interpretation": ("The measured statistic is a per-injection Delta logN, not a "
                            "per-stratum ratio, so the only exposure is eligibility."),
         "release_eligibility": rel("collaboration_restricted", "object_level_real",
                                    restricted_through=["TARGETID", "RED_SNR"],
                                    phase="phase2_after_dr2_public",
                                    rationale="per-object DR2 rows; DESI Publication Policy v2.0 s2.4"),
         "status": "current"})
    return P


# ------------------------------------------------------------------- ledger quantities
DNDX_P, OMEGA_P, OMEGA_MULT = 4, 3, 10000

PENDING_LITERALS = [
    (2175, "L1", r"approximately \(+1.4\) to \(+2.3\) per cent",
     ["TAB10.L1.ge20p3_pct_span"], 1),
    (2177, "L1", r"\(\lesssim0.3\) per cent", ["TAB10.L1.ge20p0_pct_span"], 1),
    (2214, "L2 (MIXED)", r"T_tr = 1.00--1.10 (>=20.3) / 1.00--1.13 (>=20.0)",
     ["TAB10.L2.T_tr_ge20p3_band", "TAB10.L2.T_tr_ge20p0_band"], 2),
    (2217, "L4", r"\(4.5\)-\(6.1\) per cent", ["TAB10.L4.tail_pct_span"], 1),
    (2253, "L15", r"+6.66\% (>=20.3) / +12.87\% (>=20.0)",
     ["TAB10.L15.ge20p3_allz", "TAB10.L15.ge20p0_allz"], 2),
    (2257, "Omega-L15", r"+2.57\%", ["TAB10.OMEGA_L15.allz_pct"], 2),
    (2407, "Omega statistical", r"0.94\%", ["TAB10.OMEGA_STAT.halfwidth68_pct"], 2),
    (2411, "Omega anatomy (R-037)", r"about 21.1\% ; +0.12 to +3.17\%",
     ["OMEGA.mass_share_above_21p3_pct", "OMEGA.response_family_band_pct"], 1),
    (2415, "Omega upper limit", r"-21\%, +8\%, +21\%",
     ["OMEGA.upper_limit_21p3_pct", "OMEGA.upper_limit_21p7_pct",
      "OMEGA.upper_limit_22p4_pct"], 0),
]


def _q(reduction, key):
    return reduction["quantities"].get(key)


def build_ledger_quantities(S: Side, gaps: Gaps):
    red = load_json(S.reduction)
    uls = load_json(S.uls)
    ca = load_json(S.confamb)
    Q = []

    def add(qid, product_ref, accessor, value, *, kind="estimand", estimand_ref=None,
            quantiles=None, units=None, mult=1, prec=None, pclass=None, interp=None,
            interval=None, releg=None):
        e = {"quantity_id": qid, "product_ref": product_ref, "value": value, "kind": kind}
        if accessor:
            e["accessor"] = accessor
        if estimand_ref:
            e["estimand_ref"] = estimand_ref
        if quantiles is not None:
            e["interval"] = {"kind": "quantile",
                             "quantile_order": [2.5, 16.0, 50.0, 84.0, 97.5],
                             "values": [float(x) for x in quantiles]}
        elif interval is not None:
            e["interval"] = interval
        if units:
            e["units"] = units
        e["print_multiplier"] = mult
        if prec is not None:
            e["printed_precision"] = prec
        e["status"] = "current"
        if pclass:
            e["provenance_class"] = pclass
        if interp:
            e["interpretation"] = interp
        e["release_eligibility"] = releg or AGG
        Q.append(e)

    # --- the 30 reduced quantities -------------------------------------------------
    for key, q in red["quantities"].items():
        if key.startswith("cddf_"):
            add(key, "product:reduction_pooled", f"/quantities/{key}", float(q[2]),
                estimand_ref="cddf_bin", quantiles=q, units="cm^2 (f(N,X) per linear N_HI)",
                pclass="REAL_POSTERIOR_DEPENDENT",
                interp="figure-only (FIG-40/41); no printed digit in the manuscript")
            continue
        is_om = key.startswith("omega_")
        add(key, "product:reduction_pooled", f"/quantities/{key}", float(q[2]),
            estimand_ref=("omega_20p3_21p6_allz" if is_om else
                          ("dndx_20p3_allz" if "20p3" in key else "dndx_20p0_allz")),
            quantiles=q, units=("dimensionless (Omega_DLA)" if is_om else
                                "dimensionless (per unit dX)"),
            mult=OMEGA_MULT if is_om else 1, prec=OMEGA_P if is_om else DNDX_P,
            pclass="REAL_POSTERIOR_DEPENDENT")

    # --- Omega in B5: emitted by the figure producer, not by the reduction ----------
    with np.load(S.fig41, allow_pickle=True) as z:
        om_b5 = [float(x) for x in z["lowz_omega_20p3"][4]]
        hz00 = [float(x) for x in z["highz_dndx_20p0"]]
        hz03 = [float(x) for x in z["highz_dndx_20p3"]]
    add("omega_20p3_21p6_B5", "product:fig41_array", "npz:lowz_omega_20p3[4]", om_b5[2],
        estimand_ref="omega_20p3_21p6_allz", quantiles=om_b5,
        units="dimensionless (Omega_DLA)", mult=OMEGA_MULT, prec=OMEGA_P,
        pclass="REAL_POSTERIOR_DEPENDENT",
        interp=("The TAB-12 B5 Omega CELL is a DELIBERATE SUPPRESSION in "
                "emit_tab_reporting_architecture.py (OMEGA_SUPPRESSED_BINS); the value the "
                "generator computes still moves, so the cell must still be re-emitted."))
    add("dndx_20p0_BH", HZ2, "npz:highz_dndx_20p0", hz00[2], quantiles=hz00,
        units="dimensionless (per unit dX)", prec=DNDX_P, pclass="MOCK_ONLY",
        interp="the high-z (BH) arm; frozen 2026-09-04 and not touched by a low-z freeze")
    add("dndx_20p3_BH", HZ2, "npz:highz_dndx_20p3", hz03[2], quantiles=hz03,
        units="dimensionless (per unit dX)", prec=DNDX_P, pclass="MOCK_ONLY",
        interp="the high-z (BH) arm; frozen 2026-09-04 and not touched by a low-z freeze")

    # --- census --------------------------------------------------------------------
    add("census.sum_dX", "product:pack", "npz:dX (sum); also upper_limit_scan::dX",
        float(uls["dX"]), kind="census", units="absorption distance", prec=0,
        pclass="REAL_PACK_DEPENDENT",
        interp="printed in prose as an integer with thousands separators (sample631.tex:1701)")
    add("census.n_sightlines", "product:pack", "pack provenance sidecar::n_sl",
        410767 if S.is_frozen else 418034, kind="census", units="count", prec=0,
        pclass="REAL_PACK_DEPENDENT",
        interp="the selected sightline population (sample631.tex:859 and :1701)")
    add("census.counts_in_window", "product:pack", "npz:counts (sum)", 100812,
        kind="census", units="count", prec=0, pclass="REAL_PACK_DEPENDENT",
        interp="BIT-IDENTICAL on both planes: the numerator never changed")
    add("census.counts_ge_20p3", "product:pack", "npz:counts (sum, c >= 20.3)", 38974,
        kind="census", units="count", prec=0, pclass="REAL_PACK_DEPENDENT",
        interp="BIT-IDENTICAL on both planes")

    # --- L15 configuration ambiguity ------------------------------------------------
    for thr, tag in (("ge20.3", "ge20p3"), ("ge20.0", "ge20p0")):
        v = float(ca["result"][thr]["allz_pct"])
        add(f"TAB10.L15.{tag}_allz", "product:l15_config_ambiguity",
            f"/result/{thr}/allz_pct", v, kind="systematic", estimand_ref="TAB10.L15",
            interval={"kind": "one_sided_bound", "lo": 0.0, "hi": v},
            units="per cent", prec=1, pclass="REAL_POSTERIOR_DEPENDENT",
            interp=("One-sided upper bound; the adopted pooled configuration is the LOW side. "
                    "TAB-10 prints it at p=1 from lines.L15_CONFIG_AMBIGUITY.mirror_vs_pooled_pct; "
                    "sample631.tex:2253 prints the SAME quantity at p=2 from "
                    "leverage_by_z_domain_pct. Never added in quadrature with the statistical "
                    "interval; never double-counted with L16."))
        for b in BINS:
            add(f"TAB10.L15.{tag}_{b}", "product:l15_config_ambiguity",
                f"/result/{thr}/bins_pct/{b}", float(ca["result"][thr]["bins_pct"][b]),
                kind="systematic", estimand_ref="TAB10.L15", units="per cent", prec=0,
                pclass="REAL_POSTERIOR_DEPENDENT")

    # --- the Omega systematics rows --------------------------------------------------
    scan = uls["upper_limit_scan"]
    add("TAB10.OMEGA_STAT.halfwidth68_pct", "product:upper_limit_scan",
        "/upper_limit_scan/21.6/halfwidth68_pct", float(scan["21.6"]["halfwidth68_pct"]),
        kind="systematic", units="per cent", prec=2, pclass="REAL_POSTERIOR_DEPENDENT",
        interp="the statistical 68 per cent half-width of Omega over [20.3,21.6]")
    for X, tag in (("21.3", "21p3"), ("21.7", "21p7"), ("22.4", "22p4")):
        add(f"OMEGA.upper_limit_{tag}_pct", "product:upper_limit_scan",
            f"/upper_limit_scan/{X}/median_shift_pct", float(scan[X]["median_shift_pct"]),
            kind="convention", estimand_ref="OMEGA_upper_limit_scan", units="per cent", prec=0,
            pclass="REAL_POSTERIOR_DEPENDENT",
            interp="definition dependence on the upper integration limit; explicitly NOT an uncertainty")
    add("OMEGA.mass_share_above_21p3_pct", "product:upper_limit_scan",
        "/upper_limit_scan/21.3/median_shift_pct (magnitude: truncating at 21.3 removes exactly "
        "the mass above 21.3, so the share above 21.3 is the magnitude of the recorded shift)",
        abs(float(scan["21.3"]["median_shift_pct"])), kind="diagnostic", units="per cent", prec=1,
        pclass="REAL_POSTERIOR_DEPENDENT",
        interp="the high-N mass share printed at sample631.tex:2411")

    omega_l15 = (2.572040412192078 if S.is_frozen else None)
    if omega_l15 is None:
        gaps.add("TAB10.OMEGA_L15.allz_pct on the C1 plane",
                 "The Omega leverage of the mirror configuration has NOT been recomputed on the "
                 "C1 plane: omega_anatomy.py was imported for the upper-limit scan only, and the "
                 "treatment record that carries L15's Omega leverage was not re-run. "
                 "sample631.tex:2257 therefore has a replacing quantity but no value yet.",
                 "MAJOR",
                 fillable_from="the 0.2 core-h omega_anatomy.py re-run TAB-10 already classifies "
                               "refresh_mechanical (OMEGA_STAT); no campaign and no new sampling")
    add("TAB10.OMEGA_L15.allz_pct", "product:r037_omega_anatomy",
        "/real/allz/treatments/L15_mirror_configuration/median_shift_pct", omega_l15,
        kind="systematic", estimand_ref="TAB10.L15",
        interval=({"kind": "one_sided_bound", "lo": 0.0, "hi": omega_l15} if omega_l15 else None),
        units="per cent", prec=2, pclass="REAL_POSTERIOR_DEPENDENT",
        interp=("The Omega leverage of the mirror configuration, set beside its larger dN/dX "
                "leverage." + ("" if S.is_frozen else
                               " NOT YET MEASURED on C1: refresh_mechanical, not a campaign.")))

    # --- the string-valued ledger spans the pending literals quote --------------------
    strings = [
        ("TAB10.L1.ge20p3_pct_span", LEDGER, "/lines/L1_RESPONSE_BIAS/ge20p3_pct_v2p3",
         "+1.4 to +2.3", "MOCK_ONLY",
         "mock-family residual response bias at >=20.3; carries over unchanged under C1"),
        ("TAB10.L1.ge20p0_pct_span", LEDGER, "/lines/L1_RESPONSE_BIAS/ge20p0_pct_v2p3",
         "<=0.3", "MOCK_ONLY", "the same at >=20.0; carries over unchanged under C1"),
        ("TAB10.L4.tail_pct_span", LEDGER, "/lines/L4_XFAMILY_TRANSPORT_SCATTER/pct",
         "4.5 to 6.1", "MOCK_ONLY",
         "cross-family scatter in the high-N tail; carries over unchanged under C1"),
    ]
    for qid, pref, acc, val, pc, interp in strings:
        add(qid, pref, acc, val, kind="systematic", units="per cent", prec=1,
            pclass=pc, interp=interp, releg=DOC)

    for qid, acc, val, thr in (
            ("TAB10.L2.T_tr_ge20p3_band", "/lines/L2_MOCK2REAL_C_TRANSPORT/ge20.3_band",
             "1.00 to 1.10", ">=20.3"),
            ("TAB10.L2.T_tr_ge20p0_band", "/lines/L2_MOCK2REAL_C_TRANSPORT/ge20.0_band",
             "1.00 to 1.13", ">=20.0")):
        add(qid, "product:h2m_results", acc, (val if S.is_frozen else None),
            kind="systematic", units="multiplicative", prec=2, pclass="MIXED", releg=DOC,
            interp=("The transport band T_tr at " + thr + ". " +
                    ("" if S.is_frozen else
                     "UNRESOLVED on C1: the H2-M substrate was stratified on the MEDIAN "
                     "RED_SNR against a mean-indexed completeness surface, so the band cannot "
                     "be carried over and cannot be recomputed without a mean-plane refresh "
                     "(>= 50 core-h). manifest_diff must say UNRESOLVED, not invent a number.")))
    if not S.is_frozen:
        gaps.add("sample631.tex:2214 -> a value",
                 "The L2 T_tr pending literal cannot be cleared from the C1 candidate at all; it "
                 "requires a mean-plane H2-M refresh. Since tools/check_additions.py:1063-1070 "
                 "fails a submission build on ANY pending literal, adopting C1 clears 8 of 9 "
                 "literals but NOT this one -- the submission build stays blocked on L2 "
                 "regardless of the freeze ruling.", "MAJOR",
                 fillable_from="a mean-plane H2-M refresh (>= 50 core-h) plus a predeclaration")

    add("OMEGA.response_family_band_pct", "product:r037_omega_anatomy",
        "/real/allz/treatments/response_family_band_pct",
        ("+0.12 to +3.17" if S.is_frozen else None), kind="systematic", units="per cent",
        prec=2, pclass="REAL_POSTERIOR_DEPENDENT", releg=DOC,
        interp=("The high-N response-family band of the Omega integral (second bracket of "
                "sample631.tex:2411). Recomputed inside the omega_anatomy.py re-run."))
    return Q


# ---------------------------------------------------------------------------- consumers
TAB12 = PAPER / "tables/tab_reporting_architecture.tex"
TAB10 = PAPER / "tables/tab_systematics.tex"
SAMPLE = PAPER / "sample631.tex"
FIG41_REPO = PAPER / "figures/current/data/fig_hbi_dndx.data.npz"
FIG40_REPO = PAPER / "figures/current/data/fig_hbi_cddf.data.npz"

TAB12_ROWS = [
    ("B1", 1, "B1"), ("B2", 2, "B2"), ("B3", 3, "B3"), ("B4", 4, "B4"), ("B5", 5, "B5"),
    ("all redshift", 7, "allz"), ("BH (high-z arm)", 13, "BH"),
]
TAB12_COLS = [("dndx_20p0", "dndx_20p0_{}", DNDX_P),
              ("dndx_20p3", "dndx_20p3_{}", DNDX_P),
              ("omega", "omega_20p3_21p6_{}", OMEGA_P)]

HASH_PINS = [
    ("HASHPIN.emit_tab_systematics.py:50", "paper_figures/emit_tab_systematics.py", 50,
     "LEDGER_SHA", [LEDGER]),
    ("HASHPIN.emit_tab_systematics.py:58", "paper_figures/emit_tab_systematics.py", 58,
     "NPZ_SHA", ["product:fig41_array"]),
    ("HASHPIN.emit_tab_reporting_architecture.py:43", "paper_figures/emit_tab_reporting_architecture.py",
     43, "NPZ_SHA", ["product:fig41_array"]),
    ("HASHPIN.emit_tab_raw_decomposition.py:27", "paper_figures/emit_tab_raw_decomposition.py",
     27, "NPZ_SHA", ["product:fig40_array"]),
    ("HASHPIN.fig_cddf_raw_comparison.py:47", "paper_figures/fig_cddf_raw_comparison.py", 47,
     "ZS_SHA", ["product:fig40_array"]),
    ("HASHPIN.fig_comparison_literature.py:73", "paper_figures/fig_comparison_literature.py", 73,
     "DNDX_SHA", ["product:fig41_array"]),
    ("HASHPIN.fig_comparison_literature.py:75", "paper_figures/fig_comparison_literature.py", 75,
     "CDDF_SHA", ["product:fig40_array"]),
]

BOUND_PROSE = [
    ("PROSE.sample631.tex:859", 859, "$410{,}767$", 0, ["census.n_sightlines"]),
    ("PROSE.sample631.tex:1701#n_sl", 1701, r"$410\,767$ quasar sightlines", 0,
     ["census.n_sightlines"]),
    ("PROSE.sample631.tex:1701#dX", 1701, r"$\Delta X = 550\,966$", 0, ["census.sum_dX"]),
    ("PROSE.sample631.tex:1702", 1702, r"$100\,812$ accepted detections, of which $38\,974$", 0,
     ["census.counts_in_window", "census.counts_ge_20p3"]),
    ("PROSE.sample631.tex:1779", 1779, r"0.0630\,[0.0623,\,0.0638].", DNDX_P,
     ["dndx_20p3_allz"]),
    ("PROSE.sample631.tex:1784", 1784, r"\frac{\mathrm dN}{\mathrm dX}=0.0565", DNDX_P,
     ["dndx_20p3_B1"]),
    ("PROSE.sample631.tex:1786", 1786, "0.0772", DNDX_P, ["dndx_20p3_B4"]),
    ("PROSE.sample631.tex:1808", 1808, r"0.0885\,[0.0878,\,0.0893].", DNDX_P,
     ["dndx_20p0_allz"]),
    ("PROSE.sample631.tex:2005", 2005, r"6.263\,[6.204,\,6.322]\times10^{-4}.", OMEGA_P,
     ["omega_20p3_21p6_allz"]),
]

TAB10_DEPENDS = {
    "MOCK-ONLY": [LEDGER],
    "CONVENTION": [CONV, LEDGER],
    "MIXED": ["product:h2m_results", LEDGER],
    "REAL-PACK-DEPENDENT": [PACK, LEDGER],
}
TAB10_SPECIAL = {
    "L15": ["TAB10.L15.ge20p3_allz", "TAB10.L15.ge20p0_allz", "product:l15_config_ambiguity"],
    "L16": [POST, LEDGER],
    "L14": [POST, LEDGER],
    "L12": [PACK, LEDGER],
    "OMEGA_STAT": ["TAB10.OMEGA_STAT.halfwidth68_pct"],
    "OMEGA_L15": ["TAB10.OMEGA_L15.allz_pct", "TAB10.L15.ge20p3_allz"],
    "OMEGA_UPPER_LIMIT": ["OMEGA.upper_limit_21p3_pct", "OMEGA.upper_limit_21p7_pct",
                          "OMEGA.upper_limit_22p4_pct"],
    "L8": ["product:l8ext_results", "product:l8real_sightlines", LEDGER],
    "L10": ["product:h2m_results", LEDGER],
    "L2": ["product:h2m_results", "product:cleanreal_results", LEDGER],
    "OMEGA_L2": ["product:h2m_results", LEDGER],
}
DISPOSITION_FROM_TAB10 = {
    "changes_NO": "carry_over",
    "changes_YES_MECHANICAL": "refresh_mechanical",
    "changes_YES_CAMPAIGN": "refresh_campaign",
    "changes_RECLASSIFY": "reclassify",
}


def build_dependencies(S: Side, quantities, gaps: Gaps):
    D = []
    qids = {q["quantity_id"] for q in quantities}
    d2 = load_json(CANDIDATE / "comparison/D2_figdata_diff.json")
    tab12_old = {r["bin"]: r for r in d2["tab12"]["rows"]}

    def add(**kw):
        for ref in kw["depends_on"]:
            if not (ref in qids or ref.startswith("product:") or ref.startswith("input:")):
                gaps.add(f"dependency {kw['consumer_id']} -> {ref}",
                         f"{ref} is not an entry of this manifest", "moderate")
        D.append(kw)

    # ---- TAB-12: 21 value cells --------------------------------------------------
    for binname, line, tag in TAB12_ROWS:
        row = tab12_old[binname]
        for col, qfmt, prec in TAB12_COLS:
            qid = qfmt.format(tag)
            if tag == "BH" and col == "omega":
                dep, printed = [HZ2], "---"
            elif tag == "BH":
                dep, printed = [qid.replace("_BH", "_BH")], row[f"{col}_old"]
                dep = [f"{col}_BH"]
            else:
                dep = [qid]
                printed = row["omega_old"] if col == "omega" else row[f"{col}_old"]
                if tag == "B5" and col == "omega":
                    printed = "---"
            add(consumer_id=f"TAB-12.{tag}.{col}", consumer_kind="table_cell",
                location={"repo": "paper", "path": str(TAB12), "line": line,
                          "key": f"{tag}.{col}"},
                emitter={"repo": "paper",
                         "script": "paper_figures/emit_tab_reporting_architecture.py",
                         "commit": PAPER_COMMIT_AT_FIGDATA},
                hand_typed=False, printed_value=printed, printed_precision=prec,
                depends_on=dep, status="current", disposition="refresh_mechanical",
                release_eligibility=DOC,
                submission_visibility={"ships": True, "build_state": "live", "block": None,
                                       "pending_literal": False,
                                       "gate": "tools/check_repro.py:320 check_generated_tables()"},
                **({"_note": "the generator deliberately SUPPRESSES this cell "
                             "(OMEGA_SUPPRESSED_BINS); the value it computes still moves, so the "
                             "cell must still be re-emitted"}
                   if (tag == "B5" and col == "omega") else {}))

    # ---- FIG-41 / FIG-40: one consumer per npz key -------------------------------
    for figid, repo_npz, edges, emitter in (
            ("FIG-41", FIG41_REPO, FIG41_EDGES, "paper_figures/fig_hbi_dndx.py"),
            ("FIG-40", FIG40_REPO, FIG40_EDGES, "paper_figures/fig_hbi_cddf.py")):
        for key in sorted(edges):
            add(consumer_id=f"{figid}.{key}", consumer_kind="figure_array",
                location={"repo": "paper", "path": str(repo_npz), "key": key},
                emitter={"repo": "paper", "script": emitter,
                         "commit": PAPER_COMMIT_AT_FIGDATA},
                hand_typed=False, depends_on=list(edges[key]), status="current",
                disposition="refresh_mechanical", release_eligibility=DOC,
                submission_visibility={"ships": True, "build_state": "live",
                                       "pending_literal": False})

    # ---- TAB-10: the 22 low-z lines ----------------------------------------------
    t10 = load_json(CANDIDATE / "systematics/TAB10_DEPENDENCY_2026-09-10.json")
    disp_of = {}
    for bucket, disp in DISPOSITION_FROM_TAB10.items():
        for lid in t10["summary_counts"].get(bucket, []):
            disp_of[lid] = disp
    for line in t10["lines"]:
        lid = line["id"]
        pcls = line.get("provenance_class", "")
        dep = TAB10_SPECIAL.get(lid)
        if dep is None:
            for prefix, d in TAB10_DEPENDS.items():
                if pcls.startswith(prefix):
                    dep = d
                    break
        if dep is None:
            dep = [LEDGER]
        tex = line.get("tex", "tables/tab_systematics.tex:0")
        ln = int(tex.rsplit(":", 1)[1]) if ":" in tex else None
        suppressed_prose = lid in ("L8", "L10")
        add(consumer_id=f"TAB-10.{lid}", consumer_kind="ledger_row",
            location={"repo": "paper", "path": str(TAB10), "line": ln, "key": lid},
            emitter={"repo": "paper", "script": "paper_figures/emit_tab_systematics.py",
                     "commit": "976cb4151caf93de32467f0e0a9e3880e66b959c"},
            hand_typed=False, printed_value=line.get("row_verbatim"), printed_precision=1,
            depends_on=dep, status="current",
            disposition=disp_of.get(lid, "unchanged"), release_eligibility=DOC,
            submission_visibility={
                "ships": True, "build_state": "live",
                "block": ("TAB-10 float sample631.tex:2151-2162 is LIVE; every row ships, "
                          "including L8 and L10 whose PROSE is inside the suppressed "
                          "additions block 2279-2323" if suppressed_prose else
                          "TAB-10 float sample631.tex:2151-2162 is live; every row ships"),
                "pending_literal": False, "gate": None},
            _provenance_class=pcls)

    # ---- the notes ledger's own rows ---------------------------------------------
    for key, prec, dep in (
            ("lines.L15_CONFIG_AMBIGUITY.mirror_vs_pooled_pct", 1, ["TAB10.L15.ge20p3_allz",
                                                                    "TAB10.L15.ge20p0_allz"]),
            ("lines.L15_CONFIG_AMBIGUITY.leverage_by_z_domain_pct", 2, ["TAB10.L15.ge20p3_allz"]),
            ("lines.L2_MOCK2REAL_C_TRANSPORT.ge20.3_band", 2, ["TAB10.L2.T_tr_ge20p3_band"])):
        add(consumer_id=f"LEDGER.{key}", consumer_kind="ledger_row",
            location={"repo": "notes", "path": str(NOTES_LEDGER), "key": key},
            emitter=None, hand_typed=True, printed_precision=prec, depends_on=dep,
            status="current", disposition="reclassify", release_eligibility=DOC,
            submission_visibility={"ships": True, "build_state": "live",
                                   "pending_literal": False},
            _note="hand-edited; there is no producer script and no machine link to the posterior")

    # ---- bound prose literals ------------------------------------------------------
    for cid, line, printed, prec, dep in BOUND_PROSE:
        add(consumer_id=cid, consumer_kind="prose_literal",
            location={"repo": "paper", "path": str(SAMPLE), "line": line},
            emitter=None, hand_typed=True, printed_value=printed, printed_precision=prec,
            depends_on=dep, status="current", disposition="refresh_mechanical",
            release_eligibility=DOC,
            submission_visibility={"ships": True, "build_state": "live", "block": None,
                                   "pending_literal": False,
                                   "gate": "tools/check_repro.py:696 check_prose_numbers()"},
            _bound_by="paper_figures/emit_prose_numbers.py (derived entry); assertion breaks on a "
                      "freeze change but cannot auto-fix the text")

    # ---- the nine live PENDING CLEAN LOW-Z FREEZE literals -------------------------
    for line, owner, placeholder, dep, prec in PENDING_LITERALS:
        campaign = owner.startswith("L2")
        add(consumer_id=f"PROSE.sample631.tex:{line}", consumer_kind="prose_literal",
            location={"repo": "paper", "path": str(SAMPLE), "line": line},
            emitter=None, hand_typed=True,
            printed_value=f"[PENDING CLEAN LOW-Z FREEZE: {placeholder}]",
            printed_precision=prec, depends_on=dep, status="current",
            disposition="refresh_campaign" if campaign else "refresh_mechanical",
            release_eligibility=DOC,
            submission_visibility={
                "ships": True, "build_state": "live", "block": None, "pending_literal": True,
                "gate": "tools/check_additions.py:1063-1070 (PENDING-LITERAL GUARD) -- a "
                        "submission build FAILS while this renders"},
            _owning_ledger_line=owner)

    # ---- consumers that do NOT ship (inside \begin{additions} or %%-commented) ---------
    for cid, line, printed, dep, block, state in (
            ("PROSE.sample631.tex:2308", 2308,
             "Between the floor and the lower reporting threshold the transport of the "
             "calibration degrades, and we report that degradation as a diagnostic",
             ["product:h2m_results"], "additions 2279-2323 (subsec:sys_family/sys_lown/sys_small)",
             "additions_suppressed"),
            ("PROSE.sample631.tex:2311", 2311,
             "We also tested whether real detections cross the lower reporting threshold "
             "differently from injected ones. An extended measurement found no effect",
             ["product:l8ext_results"], "additions 2279-2323", "additions_suppressed"),
            ("PROSE.sample631.tex:2272", 2272,
             "L15 / L16 / PPC \\addition paragraphs", ["TAB10.L15.ge20p3_allz"],
             "%%-commented 2272-2276", "commented_out")):
        add(consumer_id=cid, consumer_kind="prose_literal",
            location={"repo": "paper", "path": str(SAMPLE), "line": line},
            emitter=None, hand_typed=True, printed_value=printed, printed_precision=1,
            depends_on=dep, status="current", disposition="reclassify",
            release_eligibility=DOC,
            submission_visibility={"ships": False, "build_state": state, "block": block,
                                   "pending_literal": False,
                                   "gate": "sample631.tex:411 \\excludecomment{additions}"})

    # ---- hash pins -----------------------------------------------------------------
    for cid, path, line, key, dep in HASH_PINS:
        add(consumer_id=cid, consumer_kind="hash_pin",
            location={"repo": "paper", "path": str(PAPER / path), "line": line, "key": key},
            emitter=None, hand_typed=True, depends_on=dep, status="current",
            disposition="refresh_mechanical", release_eligibility=DOC,
            submission_visibility={"ships": True, "build_state": "live",
                                   "pending_literal": False},
            _enforced_by="paper_figures/common.py:619 pinned() / table_common.py:88 require()")
    add(consumer_id="HASHPIN.FROZEN_STATUS.triple", consumer_kind="hash_pin",
        location={"repo": "science",
                  "path": str(S.frozen_status) if S.frozen_status else
                          "/scratch/.../cp3_real/FROZEN_STATUS.json",
                  "key": "artifact_sha256 / draws_sha256 / pack_sha256"},
        emitter=None, hand_typed=True,
        printed_value="ea881b5f / e43d9148 / 219c43aa",
        depends_on=[POST, DRAWS, PACK], status="current", disposition="refresh_mechanical",
        release_eligibility=DOC,
        submission_visibility={"ships": True, "build_state": "live", "pending_literal": False},
        _checker="paper_figures/common.py:740 verify_frozen_status (fail-closed; the manifest "
                 "supplies the expected value and never relaxes the check)")

    # ---- release artifacts ----------------------------------------------------------
    for name, dep in (("products/fig_hbi_dndx.data.npz", ["product:fig41_array"]),
                      ("products/fig_hbi_cddf.data.npz", ["product:fig40_array"]),
                      ("frozen_inputs/modelA_pack_REAL_..._v2.npz", [PACK]),
                      ("frozen_inputs/POOLED_ln_real_v2_20260821.json", [POST]),
                      ("frozen_inputs/POOLED_ln_real_v2_20260821_fdraws.npz", [DRAWS])):
        add(consumer_id=f"RELEASE.data_v1/{name}", consumer_kind="release_artifact",
            location={"repo": "release", "path": f"/home/mfho/Latex/release_staging/data_v1/{name}"},
            emitter={"repo": "paper", "script": "tools/build_release.py", "commit": "b58ccf31"},
            hand_typed=False, depends_on=dep, status="current",
            disposition="refresh_mechanical",
            release_eligibility=rel("public_release_eligible", "aggregate_reduced",
                                    release_class="SELF_CONTAINED_BUNDLE_REBUILD",
                                    phase="phase1_paper_reproducibility"),
            submission_visibility={"ships": True, "build_state": "live",
                                   "pending_literal": False})

    # ---- shipping figures backed by the campaign products --------------------------
    for cid, stem, dep, disp, note in (
            ("FIG-46.data", "fig_transport_validation", ["product:h2m_results"],
             "refresh_mechanical", "H2-M campaign, 540 sightlines / 900 injections"),
            ("FIG-48.data", "fig_transport_anatomy", ["product:h2m_results"], "reclassify",
             "its caption IS the median stratification: real-to-mock completeness ratio in "
             "three S/N strata and three z bins"),
            ("FIG-49.data", "fig_blending", ["product:h2m_results"], "refresh_mechanical",
             "same campaign, clean arm, no S/N axis")):
        add(consumer_id=cid, consumer_kind="figure_array",
            location={"repo": "paper",
                      "path": str(PAPER / f"figures/current/data/{stem}.data.npz")},
            emitter={"repo": "paper", "script": f"paper_figures/{stem}.py",
                     "commit": PAPER_COMMIT_AT_FIGDATA},
            hand_typed=False, depends_on=dep, status="current", disposition=disp,
            release_eligibility=AGG,
            submission_visibility={"ships": True, "build_state": "live", "block": None,
                                   "pending_literal": False, "gate": None},
            _note=note)
    return D


# ---------------------------------------------------------------------------- assembly
def build_freeze(S: Side, gaps: Gaps):
    if S.is_frozen:
        fs = load_json(S.frozen_status)
        n_files = len(load_json(FROZEN_MANIFEST)["entries"])
        return {
            "name": "PAPER1_FROZEN_2026-08-26",
            "status": "frozen",
            "sealed_utc": "2026-08-26T21:00:46+00:00",
            "science_lane_authority": {
                "repo": "notes",
                "path": "/home/mfho/desi_gpy_dla_notes/notes/2026-08-21_PI_RULINGS_N1_CHECKPOINT.md",
                "section": "fifth set #32-#37 (freeze declared, Path B)"},
            "pi_ruling": {"adopted": True, "date": "2026-08-26",
                          "ruling_ref": {"repo": "notes",
                                         "path": fs.get("ruling_record", "")}},
            "root": {"path": str(FROZEN_ARCHIVE), "location_class": "durable_turbo"},
            "tree_seal": {"n_files": n_files,
                          "tree_manifest_sha256": sha256_file(FROZEN_MANIFEST)},
            "_reconstruction": (
                "RECONSTRUCTED 2026-09-11 from docs/PAPER1_FROZEN_MANIFEST.json (165 entries, "
                "1:1 onto the 165 files of the Turbo archive), cp3_real/FROZEN_STATUS.json, the "
                "pack sidecars, ledger_v2p3_cp3.json (r5 -- the file, not ARTIFACT_MANIFEST_cp3's "
                "stale 'r3' reference), the R-042a reductions of the frozen draws, and the paper "
                "repository working tree (which is the ONLY source of the frozen figure arrays "
                "and tables: the archive contains neither). Lossy; see gaps[]."),
        }
    ident = load_json(CANDIDATE / "MANIFEST.json")
    return {
        "name": "LOWZ_CLEAN_C1_CANDIDATE_2026-09-10",
        "status": "candidate",
        "sealed_utc": "2026-09-10T22:38:05+00:00",
        "science_lane_authority": {
            "repo": "science",
            "path": str(CANDIDATE / "contract/GATE1_ADJUDICATION_2026-09-10.md"),
            "sha256": sha256_file(CANDIDATE / "contract/GATE1_ADJUDICATION_2026-09-10.md"),
            "section": "rev 2 s1,s4,s5"},
        "pi_ruling": {"adopted": False, "date": "2026-09-11",
                      "ruling_ref": {"repo": "notes",
                                     "path": "PI ruling 2026-09-11 (C1 accept/freeze hold)"}},
        "root": {"path": str(CANDIDATE), "location_class": "durable_turbo"},
        "tree_seal": {"n_files": int(ident.get("n_files", 0)),
                      "total_bytes": int(ident.get("total_bytes", 0)),
                      "tree_manifest_sha256": sha256_file(CANDIDATE / "MANIFEST.json"),
                      "rehash_mismatches": 0},
    }


def build_code(S: Side, gaps: Gaps):
    env = CANDIDATE / "code_env/env_lock_gpdla-hbi_2026-08-17.txt"
    repos = [{"name": "code", "path": "/home/mfho/desi_gpy_dla_detection",
              "commit": CODE_COMMIT, "tag": CODE_TAG, "clean": True}]
    if not S.is_frozen:
        repos[0]["archive"] = "product:code_archive"
    repos.append({"name": "paper", "path": str(PAPER),
                  "commit": PAPER_COMMIT_AT_FIGDATA, "tag": None, "clean": True})
    repos.append({"name": "notes", "path": "/home/mfho/desi_gpy_dla_notes",
                  "commit": gaps.add("code.repos[notes].commit",
                                     "no notes-repo commit is recorded on either side; the "
                                     "ledger's identity rests on its sha256 alone", "minor"),
                  "tag": None})
    code = {"repos": repos,
            "figure_env_lock": {"python": "3.11.15", "matplotlib": "3.10.8", "numpy": "2.2.6",
                                "lock": "paper_figures/ENV_LOCK_2026-08-26.txt",
                                "text_usetex": True}}
    if S.is_frozen:
        gaps.add("frozen:code.env_lock",
                 "The 2026-08-26 archive contains no conda/module/version capture anywhere, so "
                 "no environment diff is possible for the frozen side. Absence here is "
                 "old_side_unrecorded, never 'unchanged'.", "moderate")
        code["env_lock"] = {"name": "gpdla-hbi",
                            "path": "_GAP: not captured at the 2026-08-26 freeze",
                            "method": "conda list --explicit"}
    else:
        code["env_lock"] = {"name": "gpdla-hbi", "path": str(env),
                            "method": "conda list --explicit", "sha256": sha256_file(env)}
    return code


def build_validation(S: Side):
    if S.is_frozen:
        red = load_json(S.reduction)
        return [
            {"gate_id": "gate:closure_vs_summary",
             "description": "the R-042a reduction of the frozen draws vs the frozen pooled summary",
             "verdict": "PASS", "evidence_ref": "product:reduction_pooled",
             "metric": red["closure_vs_summary"]},
            {"gate_id": "gate:frozen_tree_rehash",
             "description": "165 manifest entries map 1:1 onto the 165 files of the Turbo archive",
             "verdict": "PASS", "metric": {"entries": 165, "files": 165}},
            {"gate_id": "gate:C1_closure",
             "description": "mixed-plane closure (rows on unselected sightlines, stratum mismatch)",
             "verdict": "NOT-RUN",
             "metric": {"note": "the defect this gate detects was found in 2026-09; it was never "
                                "run on the frozen plane"}},
        ]
    red = load_json(S.reduction)
    return [
        {"gate_id": "gate:C1_closure",
         "description": ("mixed-plane violations, rows on unselected sightlines, stratum "
                         "mismatch, rows in dX=0 cells, path outside the searched set"),
         "verdict": "PASS", "evidence_ref": "product:pack",
         "metric": {"mixed_plane_violations": 0, "rows_on_unselected_sightlines": 0,
                    "stratum_mismatch": 0, "rows_in_dX0_cells": 0, "path_outside_searched": 0,
                    "migration_matrix": "identity"}},
        {"gate_id": "gate:counts_bit_identity",
         "description": "the counts array is bit-identical to the frozen pack",
         "verdict": "PASS",
         "metric": {"n_rows": 100812, "n_ge_20p3": 38974, "keys_byte_identical": "40/45"}},
        {"gate_id": "gate:closure_vs_summary",
         "description": "the reduction vs the pooled summary", "verdict": "PASS",
         "evidence_ref": "product:reduction_pooled", "metric": red["closure_vs_summary"]},
        {"gate_id": "gate:CP3_pool_posture",
         "description": "pool composition and nuisance posture vs production", "verdict": "PASS",
         "metric": {"pool_composition": "identical to production", "G_A": "PASS on all 12 runs"}},
        {"gate_id": "gate:mirror_is_a_mode",
         "description": "the s26 mirror chain is a MODE, not a stuck or divergent chain",
         "verdict": "PASS", "evidence_ref": "product:l15_config_ambiguity",
         "metric": {"PE_gap_nats": 776, "divergences": 0, "split_rhat_20p0": 12.404}},
        {"gate_id": "gate:D2_array_diff",
         "description": "FIG-40/41 regenerated by the paper's own producers; key sets and shapes identical",
         "verdict": "PASS", "evidence_ref": "product:d2_figdata_diff",
         "metric": {"fig41_keys": 55, "fig41_changed": 11, "fig40_keys": 39,
                    "fig40_changed": 14, "keys_added_or_removed": 0}},
        {"gate_id": "gate:independent_review",
         "description": "independent reviewer re-derived population, dX, counts and reductions",
         "verdict": "NEEDS-PI",
         "metric": {"critical": 0, "major": 4, "minor": 14, "dX_agreement": "7.6e-15"}},
    ]


def build_open_items(S: Side):
    if S.is_frozen:
        return [{"id": "F1", "text": ("cc_real_posterior.py:155-170 stores an UNSPLIT 2-chain "
                                      "Gelman-Rubin as split_rhat. Disclosed, not blocking for "
                                      "this freeze."), "blocking": False}]
    return [
        {"id": "PI-1", "text": "Adopt C1 as the new low-z freeze? The science lane cannot adopt.",
         "blocking": True},
        {"id": "PI-2", "text": ("cc_real_posterior.py:155-170 stores an UNSPLIT 2-chain "
                                "Gelman-Rubin as split_rhat; under a true split-Rhat<=1.10 three "
                                "pooled seeds would fail, on the frozen pool too. Disclose as-is, "
                                "or redefine for both planes?"), "blocking": True},
        {"id": "PI-3", "text": ("Population choice ratification: C1 (no SPECTYPE cut) vs P2 "
                                "(QSO-only on both legs); ~0.1% on dN/dX."), "blocking": True},
        {"id": "PI-4", "text": ("L2/L10/L8 MIXED lines: disclose-and-defer vs a mean-plane H2-M "
                                "redo (>=50 core-h) before the freeze."), "blocking": True},
        {"id": "PI-5", "text": ("Ledger wording: the mirror mode's 't ~ 0' must become a "
                                "per-chain description (a defect on BOTH planes)."),
         "blocking": False},
        {"id": "PI-6", "text": "Adjudication s5 wording nit (355,709 -> 358,835).",
         "blocking": False},
    ]


def build_supersession(prev_path, quantities, prev_quantities, gaps: Gaps):
    prev = load_json(prev_path)
    d1 = load_json(CANDIDATE / "comparison/D1_reduction_comparison.json")
    shift = load_json(CANDIDATE / "comparison/SHIFT_DECOMPOSITION.json")
    newq = {q["quantity_id"]: q for q in quantities}
    oldq = {q["quantity_id"]: q for q in prev_quantities}
    comps = []
    for qid in sorted(set(newq) | set(oldq)):
        o, n = oldq.get(qid), newq.get(qid)
        ov = o["value"] if o else None
        nv = n["value"] if n else None
        c = {"quantity_id": qid, "old": ov, "new": nv}
        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            c["delta"] = nv - ov
            c["delta_pct"] = (nv - ov) / ov * 100 if ov else None
            oi = (o.get("interval") or {}).get("values")
            ni = (n.get("interval") or {}).get("values")
            if oi and ni and len(oi) == 5:
                hw_o = 0.5 * (oi[3] - oi[1])
                hw_n = 0.5 * (ni[3] - ni[1])
                c["old_interval"] = [oi[1], oi[3]]
                c["new_interval"] = [ni[1], ni[3]]
                c["halfwidth_ratio"] = hw_n / hw_o if hw_o else None
                c["delta_over_halfwidth"] = (nv - ov) / hw_o if hw_o else None
            prec = n.get("printed_precision")
            mult = n.get("print_multiplier", 1)
            if prec is not None:
                c["printed_digits_change"] = (round(ov * mult, prec) != round(nv * mult, prec))
        comps.append(c)
    sd = shift["sum_dX_decomposition"]
    nd = shift["naive_dndx_decomposition_multiplicative"]
    return {
        "previous_manifest_id": prev.get("manifest_id"),
        "previous_manifest_sha256": sha256_file(prev_path),
        "previous_manifest_path": (
            f"{prev_path} -- a TWO-SOURCE reconstruction: docs/PAPER1_FROZEN_MANIFEST.json "
            f"(165 entries) + the Turbo archive for the inference chain, and the paper repository "
            f"working tree for the figure arrays and tables, which the archive does not contain."),
        "previous_manifest_completeness": "reconstructed",
        "reason": (
            "GATE1 rev 2: the frozen pack cut and stratified its DENOMINATOR on the archive "
            "MEDIAN RED_SNR while its NUMERATOR and every S/N-indexed calibration block used the "
            "finder's MEAN SNR_REDSIDE; the archive builder additionally applied --spectype QSO, "
            "excluding 847 afterburner-rescued sightlines the finder had searched."),
        "comparisons": comps,
        "decomposition": [
            {"cause": "S/N-definition repair (median -> mean on the path leg)",
             "sum_dX_pct": sd["snr_definition_repair_pct"],
             "naive_dndx_pct": nd["snr_definition_repair_pct"],
             "n_sl": shift["n_sl_decomposition"]["snr_definition_repair"]},
            {"cause": "SPECTYPE-population inclusion (GALAXY/STAR sightlines the finder searched)",
             "sum_dX_pct": sd["spectype_population_pct"],
             "naive_dndx_pct": nd["spectype_population_pct"],
             "n_sl": shift["n_sl_decomposition"]["spectype_population"]},
            {"cause": "TOTAL", "sum_dX_pct": sd["total_pct"], "naive_dndx_pct": nd["total_pct"],
             "n_sl": shift["n_sl_decomposition"]["total"],
             "note": ("The frozen NUMERATOR already contained the GALAXY/STAR rows; only the "
                      "denominator lacked their path. The two steps COMPOSE exactly "
                      "(multiplicative form); compose_check_pct = "
                      f"{nd['compose_check_pct']}."),
             "source": "comparison/SHIFT_DECOMPOSITION.json"},
        ],
    }


def _generator_commit(gaps: Gaps):
    """This script's own commit. Provenance tooling must be as traceable as what it records."""
    env = os.environ.get("PROVENANCE_COMMIT")
    if env and re.fullmatch(r"[0-9a-f]{7,40}", env):
        return env
    try:
        import subprocess
        c = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        if c.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", c.stdout.strip()):
            return c.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return gaps.add("generator.commit",
                    "this manifest was emitted from an uncommitted working tree; re-emit after "
                    "committing provenance/ so the document names its own producer",
                    "minor", fillable_from="git rev-parse HEAD in the code worktree")


HERE = pathlib.Path(__file__).resolve().parent


def build(mode, prev_path=None):
    S = Side("frozen" if mode == "reconstruct-frozen" else "c1")
    gaps = Gaps()
    inputs = build_inputs(gaps, S.name)
    ids = {i["input_id"] for i in inputs}
    contract = build_contract(S.name, ids, gaps)
    products = build_products(S, gaps)
    if S.is_frozen:
        for p in products:
            if p.get("status") == "superseded":
                p["_supersession_state"] = (
                    "superseded-by-candidate LOWZ_CLEAN_C1_2026-09-10 (PROPOSED; the PI has not "
                    "adopted C1, so this remains the freeze of record until a ruling)")
    quantities = build_ledger_quantities(S, gaps)
    deps = build_dependencies(S, quantities, gaps)
    M = {
        "manifest_version": "1.0",
        "manifest_id": S.manifest_id,
        "generated_utc": utcnow(),
        "generator": {"repo": "code", "script": "provenance/build_science_manifest.py",
                      "commit": _generator_commit(gaps),
                      "command": f"build_science_manifest.py --mode {mode}"},
        "_class": "PROVENANCE-ONLY. This document records identities and edges. It produces no "
                  "science value and no consumer of it may compute one.",
        "freeze": build_freeze(S, gaps),
        "contract": contract,
        "inputs": inputs,
        "code": build_code(S, gaps),
        "products": products,
        "ledger_quantities": quantities,
        "dependencies": deps,
        "validation": build_validation(S),
        "open_items": build_open_items(S),
    }
    if not S.is_frozen and prev_path:
        prev = load_json(prev_path)
        M["supersession"] = build_supersession(prev_path, quantities,
                                               prev["ledger_quantities"], gaps)
    M["gaps"] = gaps.items
    return M


def emit_ledger_quantities(M, out):
    """The PROPOSAL to the paper lane: the quotable set as numbers, not as ledger prose."""
    rows = {}
    for q in M["ledger_quantities"]:
        prec, mult = q.get("printed_precision"), q.get("print_multiplier", 1)
        rows[q["quantity_id"]] = {
            "value": q["value"],
            "interval": q.get("interval"),
            "units": q.get("units"),
            "print_multiplier": mult,
            "printed_precision": prec,
            "printed": (fmt(q["value"], prec, mult) if prec is not None else None),
            "kind": q.get("kind"),
            "provenance_class": q.get("provenance_class"),
            "product_ref": q["product_ref"],
            "accessor": q.get("accessor"),
            "status": q.get("status"),
            "interpretation": q.get("interpretation"),
        }
    doc = {
        "_role": ("PROPOSAL, provenance-only. The machine-readable quotable set of "
                  f"{M['manifest_id']}. Offered to the Paper lane so emit_tab_systematics.py can "
                  "read a NUMBER here and the WORDING from the systematics ledger, instead of "
                  "parsing 20 string-valued range fields. Nothing in the notes repository was "
                  "edited to produce this file."),
        "manifest_id": M["manifest_id"],
        "freeze_status": M["freeze"]["status"],
        "blocking_open_items": [o["id"] for o in M.get("open_items", []) if o["blocking"]],
        "generated_utc": M["generated_utc"],
        "quantities": rows,
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["candidate", "reconstruct-frozen"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--previous", default=None,
                    help="candidate mode: the reconstructed old manifest, for supersession[]")
    ap.add_argument("--ledger-out", default=None)
    ap.add_argument("--schema", default=str(SCHEMA_PATH))
    a = ap.parse_args()

    M = build(a.mode, a.previous)

    try:
        import jsonschema
        jsonschema.validate(M, load_json(a.schema))
        valid = "VALID against SCIENCE_MANIFEST_SCHEMA.json"
    except ImportError:
        valid = "NOT VALIDATED (jsonschema unavailable)"
    except Exception as e:  # noqa: BLE001
        print("SCHEMA VALIDATION FAILED:", str(e)[:4000], file=sys.stderr)
        raise SystemExit(2)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(M, f, indent=1)
    lq = a.ledger_out or str(out.parent / "ledger_quantities.json")
    if a.mode == "candidate":
        emit_ledger_quantities(M, lq)
    print(f"{out}  {valid}")
    print(f"  sha256          {sha256_file(out)}")
    print(f"  inputs          {len(M['inputs'])}")
    print(f"  products        {len(M['products'])}")
    print(f"  quantities      {len(M['ledger_quantities'])}")
    print(f"  dependencies    {len(M['dependencies'])}")
    print(f"  gaps            {len(M['gaps'])} "
          f"(MAJOR {sum(1 for g in M['gaps'] if g['severity'] == 'MAJOR')})")
    if a.mode == "candidate":
        print(f"  ledger_quantities -> {lq}")


if __name__ == "__main__":
    main()
