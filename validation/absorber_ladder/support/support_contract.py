#!/usr/bin/env python
"""support_contract.py — the machine-enforced SUPPORT INVARIANT (PI ruling
2026-09-13b §3).

    support_id(N_truth) == support_id(dX) == support_id(counts)
                        == support_id(FP census)

A *support* is the set of rows of the universe that an object counts over.  Two
objects may only be divided, differenced or pinned against one another when they
are built on the SAME support.  The project's recurring bug class
(``feedback_one_sided_support_bug_class``, seven instances) is precisely a
numerator and a denominator carrying different supports; the 2026-09-12 forensics
found instance #7 (``OPERATOR_FORENSICS_REPORT.md`` §7): the mock scan packs of
record carry ``counts``/``dX``/``fp_E_alloc`` rebuilt at a **3300 km/s** proximity
collar while ``truth_counts``/``truth_counts_bks`` were copied byte-identically
from the **3000 km/s** adopted packs.

This module makes the support a first-class, hashable, fail-closed object.

WHAT IS IN THE SUPPORT
----------------------
The schema is CLOSED (``SUPPORT_FIELDS``).  Nine fields are the PI's minimum
list; three more (``lam_rf_max``, ``z_cut_columns``, ``quality_cut``) are
extensions this work found to be genuinely support-defining on these objects
(see ``A0_SUPPORT_REPORT.md`` §2) — in particular ``z_cut_columns``, which
separates the λ/z window applied to ``Z_DLA`` alone (``build_scan_packs.py``)
from the one applied to ``min/max(Z_DLA, Z_TRUE)``
(``molly_faithful_pc_plots.make_lambda_z_BAL_cuts(use_truth_z=True)``).

FAIL-CLOSED RULES (all raise ``SupportContractError``)
------------------------------------------------------
  * an unknown field name                      -> the schema is closed
  * a missing field                            -> no field may be defaulted
  * ``None`` / NaN / empty-string value        -> "unknown" is never a support
  * comparing objects of which any lacks a stamp -> a missing stamp is a FAIL,
    never a pass

NOTHING in this module reads survey (real) data, and nothing under
``CDDF_analysis/`` is imported, modified or executed.  It is pure stdlib +
numpy, so it runs under both ``gpdla`` and ``gpdla-hbi``.

CLI
---
    python validation/absorber_ladder/support/support_contract.py \
        --pack PACK.npz [--census CENSUS.npz] [--ops OPS.npz] [--json]

exits 0 when every stamped object shares one support, 1 otherwise.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import glob
import hashlib
import json
import math
import os
import sys
from typing import Any, Mapping

import numpy as np

__all__ = [
    "SUPPORT_FIELDS",
    "ROW_SELECTION_FIELDS",
    "SUPPORT_SCHEMA_VERSION",
    "SupportContractError",
    "SupportMismatch",
    "SupportID",
    "support_id",
    "support_from_json",
    "stamp",
    "stamp_path",
    "read_stamp",
    "stamp_array",
    "assert_same_support",
    "check_support_consistency",
    "file_sha256",
    "catalogue_identity",
]

#: Bumped whenever the FIELD SET changes.  It is part of the canonical string,
#: so ids never collide across schema versions.
SUPPORT_SCHEMA_VERSION = "support_contract/v1"

#: The CLOSED field schema, in canonical order.
#:
#:   collar_kms             proximity collar applied at BOTH window edges, km/s
#:   snr_min                SNR_REDSIDE > snr_min  (STRICT >; the strictness is
#:                          part of the contract, not of the value)
#:   z_window               (z_qso_min, z_qso_max) QSO admission window
#:   p_dla_min              P_DLA > p_dla_min      (STRICT >)
#:   lya_only_lam_min       blue rest-frame edge of the analysis window, A
#:   lam_rf_max             red  rest-frame edge of the analysis window, A   [ext]
#:   z_cut_columns          which z column(s) the λ/z window is applied to    [ext]
#:   quality_cut            the detection quality flag rule                   [ext]
#:   truth_host_floor       truth/host N_HI floor of the matching bundle, or
#:                          the string 'n/a' for objects with no truth side
#:                          (counts, dX, fp_E_alloc)
#:   bal_policy             BAL veto convention (+ the bal catalogue identity)
#:   catalogue_id           identity of the DETECTION catalogue (sha manifest)
#:   truth_catalogue_sha256 identity of the TRUTH catalogue
SUPPORT_FIELDS = (
    "collar_kms",
    "snr_min",
    "z_window",
    "p_dla_min",
    "lya_only_lam_min",
    "lam_rf_max",
    "z_cut_columns",
    "quality_cut",
    "truth_host_floor",
    "bal_policy",
    "catalogue_id",
    "truth_catalogue_sha256",
)

#: The nine fields the PI ruling names explicitly (the rest are extensions).
PI_MINIMUM_FIELDS = (
    "collar_kms", "snr_min", "z_window", "p_dla_min", "lya_only_lam_min",
    "truth_host_floor", "bal_policy", "catalogue_id", "truth_catalogue_sha256",
)

#: The fields that select WHICH ROWS OF THE UNIVERSE an object counts over.
#: ``truth_host_floor`` is excluded: 19.0 (the latent basis) vs 17.2 (the census)
#: vs 'n/a' (counts / dX) is a legitimate, intended difference BETWEEN objects,
#: whereas every other field must be identical for two objects to be divisible.
#: NOTE (measured, A0_SUPPORT_REPORT.md §5): on the truth-AWARE z window the
#: floor does perturb the detection rows at the 0.02 % level, so this subset is
#: a documented relaxation, never a silent one — the default comparison level
#: stays the full 12-field one.
ROW_SELECTION_FIELDS = tuple(f for f in SUPPORT_FIELDS
                             if f != "truth_host_floor")

#: Value allowed in ``truth_host_floor`` for objects with no truth side.
NO_TRUTH_SIDE = "n/a"

#: The two z-column conventions actually in use on these objects.
Z_CUT_ZDLA_ONLY = "Z_DLA"
Z_CUT_ZDLA_OR_ZTRUE = "minmax(Z_DLA,Z_TRUE)"


class SupportContractError(ValueError):
    """A support declaration violated the closed contract (fail-closed)."""


class SupportMismatch(SupportContractError):
    """Two or more objects carry different supports (readable diff attached)."""


# ---------------------------------------------------------------------------
# canonicalisation
# ---------------------------------------------------------------------------
def _canon_scalar(name: str, v: Any) -> Any:
    """One field value -> a JSON-canonical, round-trip-exact python value."""
    if v is None:
        raise SupportContractError(
            f"support field {name!r} is None — 'unknown' is never a support "
            "(fail-closed). Supply the value or do not build the support.")
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode()
    if isinstance(v, bool):
        raise SupportContractError(
            f"support field {name!r}: bool is not an allowed support value")
    if isinstance(v, (int, float)):
        f = float(v)
        if not math.isfinite(f):
            raise SupportContractError(
                f"support field {name!r} is {v!r} — non-finite is never a "
                "support (fail-closed).")
        # exact round-trip; 3300 and 3300.0 canonicalise identically
        return repr(f)
    if isinstance(v, str):
        if not v.strip():
            raise SupportContractError(
                f"support field {name!r} is empty/blank — fail-closed.")
        return v
    if isinstance(v, (list, tuple, np.ndarray)):
        seq = list(np.asarray(v).tolist()) if isinstance(v, np.ndarray) else list(v)
        if len(seq) == 0:
            raise SupportContractError(
                f"support field {name!r} is an empty sequence — fail-closed.")
        return [_canon_scalar(f"{name}[{i}]", x) for i, x in enumerate(seq)]
    raise SupportContractError(
        f"support field {name!r} has unsupported type {type(v).__name__}; "
        "allowed: number, string, or a sequence of those.")


def _canonical_string(fields: Mapping[str, Any],
                      subset: tuple = SUPPORT_FIELDS) -> str:
    body = {k: _canon_scalar(k, fields[k]) for k in subset}
    head = {"schema": SUPPORT_SCHEMA_VERSION, "fields": body}
    if subset != SUPPORT_FIELDS:
        head["level"] = "+".join(subset)
    return json.dumps(head, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


@dataclasses.dataclass(frozen=True)
class SupportID:
    """A canonical support declaration + its sha256.

    ``str(SupportID)`` is the short human handle used in logs and reports;
    ``.sha256`` is the invariant that must match across objects.
    """

    fields: Mapping[str, Any]
    canonical: str
    sha256: str
    row_sha256: str = ""

    @property
    def short(self) -> str:
        return self.sha256[:16]

    def subset_sha256(self, subset: tuple) -> str:
        """sha256 over a named FIELD SUBSET (e.g. ``ROW_SELECTION_FIELDS``)."""
        if not self.fields:
            raise SupportContractError(
                "this SupportID is sha-only (from an embedded NPZ key); a "
                "field-subset comparison needs the full .support.json stamp")
        return hashlib.sha256(
            _canonical_string(self.fields, tuple(subset)).encode()).hexdigest()

    def __str__(self) -> str:               # pragma: no cover - cosmetic
        return f"support:{self.short}"

    def to_json(self) -> dict:
        return dict(schema=SUPPORT_SCHEMA_VERSION,
                    support_id=self.sha256, support_id_short=self.short,
                    row_support_id=self.row_sha256,
                    canonical=self.canonical,
                    fields={k: _json_safe(self.fields[k]) for k in SUPPORT_FIELDS})

    def __eq__(self, other) -> bool:        # sha256 IS the identity
        return isinstance(other, SupportID) and other.sha256 == self.sha256

    def __hash__(self) -> int:
        return hash(self.sha256)


def _json_safe(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_json_safe(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    return v


def support_id(**fields: Any) -> SupportID:
    """Build a ``SupportID`` from the CLOSED field schema.

    Every field in ``SUPPORT_FIELDS`` is REQUIRED; unknown fields, missing
    fields, ``None``, NaN and blank strings all raise (fail-closed).
    """
    given = set(fields)
    unknown = sorted(given - set(SUPPORT_FIELDS))
    if unknown:
        raise SupportContractError(
            f"unknown support field(s) {unknown}; the schema "
            f"{SUPPORT_SCHEMA_VERSION} is CLOSED. Known: {list(SUPPORT_FIELDS)}")
    missing = [k for k in SUPPORT_FIELDS if k not in given]
    if missing:
        raise SupportContractError(
            f"missing support field(s) {missing} — no support field may be "
            "defaulted (fail-closed). Declare every one of "
            f"{list(SUPPORT_FIELDS)}.")
    canonical = _canonical_string(fields)
    row = _canonical_string(fields, ROW_SELECTION_FIELDS)
    return SupportID(fields=dict(fields), canonical=canonical,
                     sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                     row_sha256=hashlib.sha256(row.encode()).hexdigest())


def support_from_json(obj: Mapping[str, Any]) -> SupportID:
    """Rebuild a ``SupportID`` from a stamp dict, re-deriving (not trusting)
    the sha256, and refusing a stamp whose recorded sha has drifted."""
    if "fields" not in obj:
        raise SupportContractError("support stamp has no 'fields' block")
    if obj.get("schema") not in (None, SUPPORT_SCHEMA_VERSION):
        raise SupportContractError(
            f"support stamp schema {obj.get('schema')!r} != "
            f"{SUPPORT_SCHEMA_VERSION} — refusing to compare across schemas")
    sid = support_id(**{k: obj["fields"][k] for k in SUPPORT_FIELDS
                        if k in obj["fields"]})
    rec = obj.get("support_id")
    if rec is not None and rec != sid.sha256:
        raise SupportContractError(
            "support stamp is INTERNALLY INCONSISTENT: recorded support_id "
            f"{rec} != sha256 of its own fields {sid.sha256}")
    return sid


# ---------------------------------------------------------------------------
# stamping
# ---------------------------------------------------------------------------
def stamp_path(npz_path: str) -> str:
    """The stamp sidecar path for a product: ``<file-without-ext>.support.json``."""
    base = str(npz_path)
    if base.endswith(".npz"):
        base = base[:-4]
    return base + ".support.json"


def file_sha256(path: str, _cache: dict | None = None) -> str:
    """sha256 of a file, with an optional caller-held cache keyed on
    (realpath, size, mtime_ns) — the catalogues are ~120 MB each."""
    rp = os.path.realpath(path)
    st = os.stat(rp)
    key = (rp, st.st_size, st.st_mtime_ns)
    if _cache is not None and key in _cache:
        return _cache[key]
    h = hashlib.sha256()
    with open(rp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    out = h.hexdigest()
    if _cache is not None:
        _cache[key] = out
    return out


def catalogue_identity(catalog_dir: str, pattern: str = "dlacat*.fits",
                       cache: dict | None = None) -> str:
    """Identity of a DETECTION catalogue directory.

    ``sha256`` over the sorted manifest of ``(basename, size, sha256)`` of every
    matching file — so a file added, removed, renamed or rewritten changes the
    identity.  Returned as ``"<n-files>:<sha12>"`` for readability.
    """
    files = sorted(glob.glob(os.path.join(catalog_dir, pattern)))
    if not files:
        raise SupportContractError(
            f"no {pattern} under {catalog_dir!r} — cannot identify the "
            "catalogue (fail-closed)")
    man = [f"{os.path.basename(p)}:{os.path.getsize(os.path.realpath(p))}:"
           f"{file_sha256(p, cache)}" for p in files]
    h = hashlib.sha256("\n".join(man).encode()).hexdigest()
    return f"{len(files)}f:{h[:24]}"


def stamp(npz_path: str, support: SupportID, *, extra: Mapping | None = None,
          hash_product: bool = True) -> str:
    """Write the support stamp sidecar for ``npz_path``; return its path.

    The stamp records the support fields, the canonical string, the sha256, the
    product's own sha256 (so a silently-regenerated product is detectable) and
    anything the caller passes in ``extra``.
    """
    if not isinstance(support, SupportID):
        raise SupportContractError("stamp() takes a SupportID, not "
                                   f"{type(support).__name__}")
    rec = support.to_json()
    rec["product"] = os.path.abspath(npz_path)
    rec["product_sha256"] = (file_sha256(npz_path)
                             if hash_product and os.path.exists(npz_path) else None)
    rec["stamped_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    rec["stamper"] = "validation/absorber_ladder/support/support_contract.py"
    if extra:
        rec["extra"] = _json_safe(dict(extra))
    p = stamp_path(npz_path)
    with open(p, "w") as fh:
        json.dump(rec, fh, indent=1)
    return p


def stamp_array(support: SupportID) -> np.ndarray:
    """The 0-d numpy string array to embed as an NPZ ``support_id`` key.

    NOTE (verified 2026-09-13): ``CDDF_analysis/hbi_mcmc/pack.load_pack``
    rejects any key outside its closed schema, so a Model-A *pack* must NOT
    carry this key — its support lives in the ``.support.json`` sidecar
    instead.  Non-pack products (the census) may carry it.
    """
    return np.array(support.sha256, dtype="<U64")


def read_stamp(path: str, *, required: bool = True) -> SupportID | None:
    """Read the support of a product.

    Resolution order: the ``.support.json`` sidecar, then an embedded
    ``support_id`` NPZ key (sha only — which can be COMPARED but not diffed).
    ``required=True`` (the default) raises when neither is present: a missing
    stamp is a FAILURE, never a pass.
    """
    sp = stamp_path(path)
    if os.path.exists(sp):
        with open(sp) as fh:
            return support_from_json(json.load(fh))
    if str(path).endswith(".npz") and os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            if "support_id" in z.files:
                sha = str(np.asarray(z["support_id"]).item())
                return SupportID(fields={}, canonical="", sha256=sha)
    if required:
        raise SupportContractError(
            f"{path} carries NO support stamp ({sp} absent and no NPZ "
            "'support_id' key) — fail-closed: an unstamped object can never be "
            "certified consistent.")
    return None


# ---------------------------------------------------------------------------
# the invariant
# ---------------------------------------------------------------------------
def _diff_table(objects: Mapping[str, SupportID],
                subset: tuple = SUPPORT_FIELDS) -> str:
    names = list(objects)
    lines = []
    have_fields = {n: bool(objects[n].fields) for n in names}
    for f in subset:
        vals = {n: (objects[n].fields.get(f, "<no-fields-in-stamp>")
                    if have_fields[n] else "<sha-only stamp>") for n in names}
        uniq = {json.dumps(_json_safe(v), sort_keys=True) for v in vals.values()}
        if len(uniq) > 1:
            lines.append(f"  DIFFERS  {f}:")
            for n in names:
                lines.append(f"             {n:<28s} {_json_safe(vals[n])!r}")
    if not lines:
        lines.append("  (no field-level diff available — at least one stamp is "
                     "sha-only; compare the full sidecars)")
    return "\n".join(lines)


def _normalise(objects: Mapping[str, Any]) -> dict:
    norm: dict[str, SupportID] = {}
    for name, v in objects.items():
        if isinstance(v, SupportID):
            norm[name] = v
        elif isinstance(v, Mapping):
            norm[name] = support_from_json(v)
        elif isinstance(v, str) and len(v) == 64:
            norm[name] = SupportID(fields={}, canonical="", sha256=v)
        else:
            raise SupportContractError(
                f"object {name!r}: expected SupportID / stamp dict / sha256, "
                f"got {type(v).__name__}")
    return norm


def assert_same_support(objects: Mapping[str, Any], *,
                        fields: tuple = SUPPORT_FIELDS) -> str:
    """Raise ``SupportMismatch`` unless every named object shares one support.

    ``objects`` maps a readable name (``"counts"``, ``"truth_counts"``,
    ``"dX"``, ``"fp_census"``, ...) to its ``SupportID`` (a raw stamp dict or a
    bare sha256 string is accepted too).  Returns the common sha256.

    ``fields`` selects the comparison level and DEFAULTS to the full 12-field
    schema (strictest).  Pass ``ROW_SELECTION_FIELDS`` only where the PI has
    ruled that a ``truth_host_floor`` difference between objects is acceptable;
    the relaxation is then recorded in the returned/raised text.
    """
    if not objects:
        raise SupportContractError(
            "assert_same_support() got no objects — an empty check is not a "
            "pass (fail-closed).")
    fields = tuple(fields)
    unknown = [f for f in fields if f not in SUPPORT_FIELDS]
    if unknown:
        raise SupportContractError(f"unknown comparison field(s) {unknown}")
    if not fields:
        raise SupportContractError(
            "assert_same_support(fields=()) compares nothing — not a pass.")
    norm = _normalise(objects)
    if fields == SUPPORT_FIELDS:
        shas = {n: s.sha256 for n, s in norm.items()}
    else:
        shas = {n: s.subset_sha256(fields) for n, s in norm.items()}
    if len(set(shas.values())) == 1:
        return next(iter(shas.values()))
    lvl = ("full 12-field schema" if fields == SUPPORT_FIELDS
           else f"RELAXED level [{'+'.join(fields)}]")
    head = "\n".join(f"  {n:<30s} {s[:16]}" for n, s in shas.items())
    raise SupportMismatch(
        "SUPPORT MISMATCH — these objects are NOT on a common support and may "
        "not be divided, differenced or pinned against one another "
        f"(compared at the {lvl}):\n"
        f"{head}\n\nfield-level diff:\n{_diff_table(norm, fields)}\n\n"
        "(PI ruling 2026-09-13b §3: support consistency is a machine-enforced, "
        "fail-closed invariant.)")


def _object_planes(path: str, label: str) -> dict:
    """Every stamped PLANE of one product, as ``{"<label>.<plane>": SupportID}``.

    A stamp may carry ``extra.planes = {plane_name: {fields...}}`` — the A0
    products do, so ``counts``, ``dX`` and ``truth_counts`` inside one pack are
    compared against each other individually (that pairing IS defect #7).  A
    stamp without a ``planes`` block contributes the object as a whole.
    """
    sid = read_stamp(path)
    sp = stamp_path(path)
    planes = {}
    if os.path.exists(sp):
        with open(sp) as fh:
            rec = json.load(fh)
        for name, blk in (rec.get("extra", {}) or {}).get("planes", {}).items():
            planes[f"{label}.{name}"] = support_id(
                **{k: blk[k] for k in SUPPORT_FIELDS})
    return planes or {label: sid}


def check_support_consistency(pack_path: str, census_path: str | None = None,
                              ops_path: str | None = None, *,
                              fields: tuple = SUPPORT_FIELDS) -> dict:
    """THE GATE the ladder runner calls before it does anything else.

    Fail-closed: every supplied path must carry a stamp, every stamped plane is
    entered into the comparison individually, and every one must agree.  Raises
    ``SupportContractError`` / ``SupportMismatch``; on success returns a
    provenance record for the run's JSON.

    A pack whose ``truth_counts`` plane sits on a different support than its
    ``counts``/``dX`` plane is exactly the defect this gate exists to refuse;
    such a pack must be REBUILT (A0), never stamped around.
    """
    objs: dict[str, SupportID] = {}
    paths: dict[str, str] = {"pack": os.path.abspath(pack_path)}
    objs.update(_object_planes(pack_path, "pack"))
    if census_path:
        paths["fp_census"] = os.path.abspath(census_path)
        objs.update(_object_planes(census_path, "fp_census"))
    if ops_path:
        paths["empirical_ops"] = os.path.abspath(ops_path)
        objs.update(_object_planes(ops_path, "empirical_ops"))
    common = assert_same_support(objs, fields=fields)
    any_fields = next((s.fields for s in objs.values() if s.fields), {})
    return dict(status="PASS", schema=SUPPORT_SCHEMA_VERSION,
                level=("full" if tuple(fields) == SUPPORT_FIELDS
                       else "+".join(fields)),
                support_id=common, support_id_short=common[:16],
                paths=paths,
                planes={n: s.sha256 for n, s in objs.items()},
                truth_host_floor={n: _json_safe(s.fields.get("truth_host_floor"))
                                  for n, s in objs.items() if s.fields},
                fields=_json_safe(dict(any_fields)),
                checked_utc=datetime.datetime.utcnow().isoformat() + "Z")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--census", default=None)
    ap.add_argument("--ops", default=None)
    ap.add_argument("--level", choices=("full", "row"), default="full",
                    help="'full' = all 12 fields (default, strictest); 'row' = "
                         "the row-selection subset, i.e. truth_host_floor "
                         "differences tolerated (PI-ruled relaxation only)")
    ap.add_argument("--json", action="store_true", help="machine-readable only")
    a = ap.parse_args(argv)
    lvl = SUPPORT_FIELDS if a.level == "full" else ROW_SELECTION_FIELDS
    try:
        rec = check_support_consistency(a.pack, a.census, a.ops, fields=lvl)
    except SupportContractError as exc:
        if a.json:
            print(json.dumps(dict(status="FAIL", error=str(exc)), indent=1))
        else:
            print("SUPPORT CONTRACT: FAIL\n", file=sys.stderr)
            print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
