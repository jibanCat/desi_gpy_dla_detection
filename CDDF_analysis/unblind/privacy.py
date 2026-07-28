"""privacy.py -- self-check that no real-LOA numerics leak into committed artifacts.

Real-LOA (DESI DR2 Loa) result values (dN/dX, Omega, f(N), per-z tables, over-count
percentages) are PRIVATE.  They belong in the private notes repo, never in the code
repo and never in a commit message.  This module gives the two committed carriers of
a leak their own guard:

  NOTEBOOKS  ``assert_no_outputs`` -- a committed notebook with zero code-cell
             outputs cannot leak a value.  ``scan_notebook_outputs`` is the softer
             heuristic for a notebook someone forgot to clear.

  JSON ARTIFACTS  ``assert_json_artifact_mock_only`` / ``scan_json_artifact`` --
             the three independent tells that a JSON came from real DESI data:
               1. TARGETID magnitude.  Mock TARGETIDs are O(1e3-1e8); real DESI
                  TARGETIDs are O(1e16).  A single large integer is decisive.
               2. Real-data PATH tokens (``main-dark``, ``loa-main``, the altbal
                  VAC, the real catalog directories), matched separator- and
                  case-insensitively.
               3. Real-value CO-OCCURRENCE: a real-data tell appearing in the same
                  document as science-result keys (dndx / omega / f_N / per-z) is
                  not a stray path string, it is a real MEASUREMENT.

THE FALSE-PASS THAT MOTIVATED THE JSON GUARD
--------------------------------------------
``assert_no_outputs`` used to iterate ``nb.get("cells", [])``.  Handed a ``.json``
artifact, that ``.get`` returned ``[]``, both loops were vacuous, and the function
returned 0 -- a clean PASS on a file it had not inspected at all.  It was cited as
privacy clearance for JSON artifacts.  It now REFUSES any non-notebook input: a
guard that cannot inspect its input must ERROR, never pass.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

# decimals that look like a scientific result (a dot with digits either side, or sci-notation).
_NUM = re.compile(r"(?<![\w.])\d+\.\d+(?:[eE][+-]?\d+)?(?![\w.])")


class PrivacyError(RuntimeError):
    """Raised when a privacy guard fails, or cannot inspect what it was given.

    Subclasses ``RuntimeError`` so existing ``pytest.raises(RuntimeError)`` callers
    keep working.
    """


# ---------------------------------------------------------------------------
# notebook guards
# ---------------------------------------------------------------------------
@dataclass
class OutputHit:
    cell_index: int
    output_index: int
    kind: str            # 'stream' / 'execute_result' / 'display_data' / 'error'
    n_numbers: int
    sample: str          # a short, truncated excerpt (may itself contain numbers!)


def _iter_output_text(out):
    """Yield text payloads from a single nbformat output object."""
    t = out.get("output_type")
    if t == "stream":
        yield "".join(out.get("text", []) if isinstance(out.get("text"), list) else [out.get("text", "")])
    elif t in ("execute_result", "display_data"):
        data = out.get("data", {})
        txt = data.get("text/plain", "")
        yield "".join(txt) if isinstance(txt, list) else txt
    elif t == "error":
        yield "\n".join(out.get("traceback", []))


def _load_notebook(nb_path: str) -> dict:
    """Load a path and PROVE it is an nbformat notebook, or raise.

    FAIL-CLOSED.  The historical bug was that a non-notebook produced an empty
    cell list and therefore a vacuous pass; the only safe response to "I cannot
    tell what this is" is to raise.
    """
    if not os.path.exists(nb_path):
        raise PrivacyError(f"{nb_path}: does not exist -- cannot certify it privacy-clean.")
    ext = os.path.splitext(nb_path)[1].lower() or "<no extension>"
    if ext != ".ipynb":
        raise PrivacyError(
            f"{nb_path}: this is a NOTEBOOK guard and the file is not a .ipynb (got {ext}). "
            "It CANNOT certify this file. A JSON artifact goes through "
            "assert_json_artifact_mock_only(); anything else has no guard yet. "
            "(Historical bug: this function used to return a silent PASS on a .json because "
            "nb['cells'] was empty and both loops were vacuous.)"
        )
    with open(nb_path) as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError as exc:
            raise PrivacyError(f"{nb_path}: not valid JSON ({exc}) -- cannot certify.") from exc
    # REQUIRED, never defaulted: the original bug was `nb.get("cells", [])`, whose
    # default turned "this file has no cells" into "this file has no offending
    # cells".  A missing / non-list `cells` is now an ERROR.
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise PrivacyError(
            f"{nb_path}: not an nbformat notebook (no 'cells' list). Refusing to return a "
            "vacuous PASS on a file this guard cannot read."
        )
    bad = [i for i, c in enumerate(nb["cells"])
           if not isinstance(c, dict) or "cell_type" not in c]
    if bad:
        raise PrivacyError(
            f"{nb_path}: 'cells' entries {bad[:5]} are not notebook cells (no 'cell_type'). "
            "Refusing to certify a file this guard cannot read."
        )
    return nb


def assert_no_outputs(nb_path: str) -> int:
    """Hard guarantee: raise if ANY code cell carries outputs.  Returns 0 on success.

    A committed notebook that passes this cannot leak a real-LOA value.  Raises
    :class:`PrivacyError` if ``nb_path`` is not an nbformat notebook -- it is NOT a
    general-purpose privacy check and must never be cited as one for a JSON."""
    nb = _load_notebook(nb_path)
    offenders = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            offenders.append(i)
    if offenders:
        raise PrivacyError(
            f"{nb_path}: {len(offenders)} code cell(s) still carry outputs {offenders}. "
            "Run `jupyter nbconvert --clear-output --inplace <nb>` before committing -- "
            "executed outputs contain real-LOA values."
        )
    return 0


def scan_notebook_outputs(nb_path: str, max_sample: int = 60) -> list:
    """Heuristic: return OutputHit rows for every output containing decimal numbers.

    Use as a soft tripwire when a notebook was executed.  Empty list == clean/cleared.
    Raises :class:`PrivacyError` on a non-notebook (see :func:`_load_notebook`)."""
    nb = _load_notebook(nb_path)
    hits = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for oi, out in enumerate(cell.get("outputs", []) or []):
            for text in _iter_output_text(out):
                nums = _NUM.findall(text or "")
                if nums:
                    hits.append(OutputHit(ci, oi, out.get("output_type", "?"),
                                          len(nums), (text or "")[:max_sample]))
    return hits


# ---------------------------------------------------------------------------
# JSON-artifact guards
# ---------------------------------------------------------------------------
#
# TELL 1 -- TARGETID magnitude.
# The project's own provenance test (nomenclature glossary, 2026-07-27): mock
# TARGETIDs are O(1e3-1e8), real DESI TARGETIDs are O(1e16).  Nothing else in these
# artifacts is a legitimate integer above ~1e12 (counts are O(1e3), wallclock O(1e3),
# seeds O(1)), so one such integer anywhere in the document is decisive.
REAL_TARGETID_MIN = 10 ** 12

# TELL 2 -- real-data path tokens, written in CANONICAL hyphen form.  Matching is
# separator-insensitive and case-insensitive (see _canon), so "main-dark" also
# catches "main_dark", "Main Dark", "main.dark" and "healpix/main/dark".
#
# THE MISS THIS FIXES: the one committed real-data token test scanned for
# "main_dark" (UNDERSCORE) while the artifacts and DESI filenames write "main-dark"
# (HYPHEN) -- `dlacat-loa-main-dark-v1.fits`, `coadd-main-dark-705.fits`,
# `processed-main-dark-*.h5`.  Every hyphenated real path sailed straight through.
REAL_DATA_TOKENS = (
    "main-dark",              # DESI main survey, dark-time program
    "loa-main",               # loa_main_dark_v1 catalog family
    "processed-main-dark",    # real processed-*.h5
    "coadd-main-dark",        # real coadd FITS
    "dlacat-loa",             # real DLA catalog
    "qso-cat-loa",            # real QSO VAC
    "qso-cat-kibo",           # real QSO VAC (kibo)
    "healpix-main-dark",      # real healpix tree
    "altbal",                 # the real BAL VAC suffix
)

# TELL 3 -- science-result keys.  A real-data tell in a document that ALSO carries
# these is not a stray path string, it is a real MEASUREMENT.
SCIENCE_VALUE_KEYS = (
    "dndx", "dn_dx", "omega", "omega_hi", "f_n", "fn", "cddf", "f_truth",
    "r0", "perz", "per_z", "measurement", "integrated", "cumulative", "lx", "ell",
)

# PROSE vs PATH.  A mock artifact legitimately says so in words -- the committed
# forward artifact's metadata.mock literally reads "No real-LOA (loa main-dark) data
# was read".  That is a CLAIM, not data, and failing on it would train people to
# delete the disclaimer.  A leak is a PATH (an input actually read) or a VALUE.  So a
# token inside a path-like string, or inside a KEY name, is a HARD hit; the same
# token in free prose is a SOFT 'prose_token' -- reported, not fatal by default.
_PATHISH_EXT = (".fits", ".h5", ".hdf5", ".npz", ".npy", ".tsv", ".csv", ".env",
                ".json", ".txt", ".dat", ".fits.gz")

_SEP = re.compile(r"[-_./\\\s]+")


def _is_pathish(text: str) -> bool:
    t = str(text).strip().lower()
    if "/" in t or "\\" in t:
        return True
    return any(e in t for e in _PATHISH_EXT)


def _canon(text: str) -> str:
    """Lowercase and collapse every separator to '-' so token matching is robust
    to the underscore/hyphen/slash/dot variants the same dataset is written in."""
    return _SEP.sub("-", str(text).lower())


@dataclass
class RealDataHit:
    kind: str          # 'targetid' | 'path_token' | 'value_cooccurrence'
    where: str         # JSON path, e.g. "metadata.inputs.catalog_dir"
    token: str         # the token / the offending integer, as a string
    excerpt: str = ""  # short context (never a science value)


def _walk(obj, prefix=""):
    """Yield ``(json_path, key_or_None, value)`` for every node in a JSON document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p, k, v
            yield from _walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield p, None, v
            yield from _walk(v, p)


def _load_json(artifact):
    if isinstance(artifact, (dict, list)):
        return artifact, "<in-memory>"
    if not os.path.exists(artifact):
        raise PrivacyError(f"{artifact}: does not exist -- cannot certify it privacy-clean.")
    ext = os.path.splitext(artifact)[1].lower()
    if ext == ".ipynb":
        raise PrivacyError(
            f"{artifact}: this is the JSON-ARTIFACT guard and the file is a notebook. "
            "Use assert_no_outputs()/scan_notebook_outputs() for notebooks."
        )
    with open(artifact) as f:
        try:
            return json.load(f), artifact
        except json.JSONDecodeError as exc:
            raise PrivacyError(f"{artifact}: not valid JSON ({exc}) -- cannot certify.") from exc


def scan_json_artifact(artifact) -> list:
    """Return every :class:`RealDataHit` in a JSON artifact (path or parsed object).

    Empty list == no real-DESI tell found.  This is a POSITIVE-evidence scanner: an
    empty result means "no tell", not "proven mock" -- pair it with a positive mock
    marker (``2lpt`` / ``london`` / ``saclay`` / ``loa-124``) when that matters.
    """
    doc, _label = _load_json(artifact)
    hits: list = []

    # --- tell 1: TARGETID magnitude ---------------------------------------
    for path, key, val in _walk(doc):
        if isinstance(val, bool):
            continue
        if isinstance(val, int) and abs(val) >= REAL_TARGETID_MIN:
            hits.append(RealDataHit(
                "targetid", path, str(val),
                "integer >= 1e12; mock TARGETIDs are O(1e3-1e8), real DESI O(1e16)"))
        elif isinstance(val, float) and abs(val) >= REAL_TARGETID_MIN and float(val).is_integer():
            hits.append(RealDataHit(
                "targetid", path, repr(val),
                "integral float >= 1e12 -- a real DESI TARGETID that lost its int type"))

    # --- tell 2: real-data path tokens ------------------------------------
    for path, key, val in _walk(doc):
        texts = []
        if key is not None:
            texts.append((f"{path} (key)", key, True))
        if isinstance(val, str):
            texts.append((path, val, _is_pathish(val)))
        for where, text, pathish in texts:
            c = _canon(text)
            for tok in REAL_DATA_TOKENS:
                if tok in c:
                    hits.append(RealDataHit(
                        "path_token" if pathish else "prose_token",
                        where, tok, str(text)[:120]))

    # --- tell 3: real-value co-occurrence ---------------------------------
    # Escalate only on a HARD tell (a real path actually read, or a real TARGETID).
    # A prose disclaimer next to mock numbers is not a leak.
    if any(h.kind in ("targetid", "path_token") for h in hits):
        science = sorted({
            str(k).lower() for _, k, _ in _walk(doc)
            if k is not None and str(k).lower() in SCIENCE_VALUE_KEYS
        })
        if science:
            hits.append(RealDataHit(
                "value_cooccurrence", "<document>", ",".join(science),
                "a real-data tell CO-OCCURS with science-result keys: this document "
                "carries real-DESI RESULT VALUES, not just a path string."))
    return hits


HARD_HIT_KINDS = ("targetid", "path_token", "value_cooccurrence")


def assert_json_artifact_mock_only(artifact, allow_tokens=(), strict_prose: bool = False) -> int:
    """Raise :class:`PrivacyError` if a JSON artifact shows any HARD real-DESI tell.

    ``allow_tokens`` whitelists specific path tokens for the rare artifact that must
    legitimately NAME a real dataset without carrying its values (e.g. a config
    echo).  It never whitelists a TARGETID or a value co-occurrence.
    ``strict_prose=True`` additionally fails on a real-data token in free prose
    (a mock artifact's own "no real-LOA data was read" disclaimer trips this).
    """
    doc, label = _load_json(artifact)
    allow = {_canon(t) for t in allow_tokens}
    kinds = HARD_HIT_KINDS + (("prose_token",) if strict_prose else ())
    hits = [h for h in scan_json_artifact(doc)
            if h.kind in kinds
            and not (h.kind in ("path_token", "prose_token") and h.token in allow)]
    if hits:
        lines = [f"  [{h.kind}] {h.where}: {h.token}" for h in hits[:12]]
        more = f"\n  ... and {len(hits) - 12} more" if len(hits) > 12 else ""
        raise PrivacyError(
            f"{label}: {len(hits)} real-DESI privacy hit(s) -- this artifact must NOT be "
            f"committed to the code repo (private notes repo only):\n" + "\n".join(lines) + more
        )
    return 0


CLEAR_INSTRUCTION = (
    "Before committing this notebook, STRIP all outputs (they embed real-LOA values):\n"
    "    jupyter nbconvert --clear-output --inplace notebooks/UNBLIND_00_guards_and_data.ipynb\n"
    "and keep executed copies out of git (see the .gitignore note in the handoff)."
)
