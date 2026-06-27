"""Shared commit-stamp + provenance-emit layer for the CDDF intermediate-results
store (implements ``CDDF_analysis/RESULTS_STORE_PLAN.md`` §2 + §4).

Pure stdlib (``subprocess``, ``hashlib``, ``json``, ``datetime``, ``pathlib``,
``os``). Generalizes ``track_c_tf_loa.py::_git_commit`` into a full
``git_stamp`` (short+long SHA, branch, dirty flag, dirty-diff fingerprint), and
adds the config hashing / slugging / privacy-contagion / atomic provenance
writer that ``CDDF_analysis/results_store.py`` builds on.

Design rules (from the plan):
  * A clean tree is trustworthy + fully reproducible from its SHA.
  * A dirty tree is LOUD, not fatal: ``dirty:true`` + ``diff_sha256`` fingerprint,
    a ``⚠ DIRTY`` banner in the README. Research moves fast; never block, but
    never print a misleading clean SHA.
  * ``commit:"unknown"`` (git failure / non-repo) is a hard smell, recorded safely.
  * Privacy is contagious DOWNWARD: a result is real-LOA iff any input in its
    chain is real-LOA. Can't be laundered.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "git_stamp",
    "config_hash",
    "make_slug",
    "privacy_class",
    "write_provenance",
]

# This module lives at CDDF_analysis/hbi/provenance.py; the repo root is three
# levels up. Used as the default cwd for git when no repo_root is given.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROVENANCE_SCHEMA_VERSION = "cddf-provenance/1"


# --------------------------------------------------------------------------- #
# git commit stamping                                                          #
# --------------------------------------------------------------------------- #
def _git(args: list[str], cwd: str) -> str:
    """Run ``git <args>`` in ``cwd``, return stripped stdout. Raises on failure
    (caller catches). stderr is swallowed so git noise never leaks."""
    return subprocess.check_output(
        ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
    ).decode().strip()


def git_stamp(repo_root: str | None = None) -> dict:
    """Return a reproducibility stamp for the working tree at ``repo_root``.

    Generalizes ``track_c_tf_loa._git_commit``: that returned only the short SHA
    (or ``"unknown"``); this returns the full record the provenance contract needs.

    Returns a dict with keys::

        commit_short : str   abbreviated SHA, or "unknown" on git failure
        commit_long  : str   full 40-char SHA, or "unknown"
        branch       : str   current branch (abbrev-ref), or "unknown"
        dirty        : bool   True iff tracked files differ from HEAD
        diff_sha256  : str|None  sha256 of `git diff HEAD` when dirty, else None

    On ANY git failure (not a repo, git not installed, detached/empty) every
    field degrades to its safe default and no exception escapes.
    """
    cwd = repo_root or _REPO
    out = {
        "commit_short": "unknown",
        "commit_long": "unknown",
        "branch": "unknown",
        "dirty": False,
        "diff_sha256": None,
    }
    try:
        out["commit_short"] = _git(["rev-parse", "--short", "HEAD"], cwd)
        out["commit_long"] = _git(["rev-parse", "HEAD"], cwd)
    except Exception:
        # No HEAD / not a repo: everything stays at the safe defaults.
        return out

    # branch (best-effort; failure leaves "unknown" but keeps the SHA).
    try:
        out["branch"] = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    except Exception:
        pass

    # dirty = any tracked change vs HEAD (ignore untracked files, matching the
    # plan: `git status --porcelain --untracked-files=no`).
    try:
        porcelain = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
        out["dirty"] = bool(porcelain)
        if out["dirty"]:
            diff = subprocess.check_output(
                ["git", "diff", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            out["diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except Exception:
        # Leave dirty=False / diff_sha256=None on failure: never claim dirty
        # state we couldn't verify.
        pass

    return out


# --------------------------------------------------------------------------- #
# config hashing                                                               #
# --------------------------------------------------------------------------- #
def config_hash(config: dict) -> str:
    """Stable 8-hex-char id for a config dict.

    ``sha1(json.dumps(config, sort_keys=True, default=str))[:8]``. Idempotent
    (same config → same hash regardless of key order); ``default=str`` lets
    non-JSON-native values (Path, tuple-via-list, numpy scalars) hash without
    raising; distinct configs → distinct hashes (8 hex = 32 bits, ample for a
    per-dataset/stage namespace).
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# human-readable slug                                                          #
# --------------------------------------------------------------------------- #
def _fmt_num(v: float) -> str:
    """Compact, filesystem-safe rendering of a number: drop a trailing ``.0``,
    replace the decimal point with nothing for integers and keep it otherwise,
    so 2.0 -> '2', 2.5 -> '2.5', 4.25 -> '4.25'."""
    if isinstance(v, bool):  # bool is an int subclass — guard first.
        return "1" if v else "0"
    if isinstance(v, (int,)):
        return str(v)
    f = float(v)
    if f.is_integer():
        return str(int(f))
    # strip trailing zeros but keep the meaningful decimals.
    return ("%g" % f)


def _slug_token(key: str, value: Any) -> str:
    """Render one differing (key, value) pair into a short slug token.

    Conventions tuned to the science knobs in the plan:
      * bool True  -> bare key with a leading 'no' kept verbatim if already
        phrased so (``no_bal`` True -> ``nobal``); otherwise ``<key>``.
      * bool False -> empty (handled by caller: a False that differs from a True
        default renders as ``no<key>`` only if the key reads positively — kept
        simple here: emit ``no<key>`` token).
      * numeric ``snr_min`` -> ``snr<n>`` (drop the ``_min`` suffix).
      * list (e.g. ``zbins``) -> ``<key-prefix><first>-<last>`` (z2-3.5).
      * other scalars -> ``<key><value>``.
    """
    # snr_min -> snr<value>
    if key == "snr_min":
        return "snr" + _fmt_num(value)

    # boolean flags: collapse no_bal/no_* into a single readable token.
    if isinstance(value, bool):
        base = key
        if base.startswith("no_"):
            tok = "no" + base[len("no_"):]
        else:
            tok = base
        tok = tok.replace("_", "")
        return tok if value else ("no" + tok)

    # list/tuple of numbers (zbins, edges): render as <prefix><first>-<last>.
    if isinstance(value, (list, tuple)) and value:
        prefix = "z" if key == "zbins" else key.replace("_", "")
        return f"{prefix}{_fmt_num(value[0])}-{_fmt_num(value[-1])}"

    # generic scalar.
    return f"{key.replace('_', '')}{_fmt_num(value) if isinstance(value, (int, float)) else value}"


def make_slug(config: dict, producer_defaults: dict) -> str:
    """Short, readable, deterministic, filesystem-safe slug from only the config
    keys whose value differs from ``producer_defaults``.

    Example (from the plan)::

        cfg      = {"snr_min": 2.0, "no_bal": True, "zbins": [2.0,2.5,3.0,3.5]}
        defaults = {"snr_min": 0.0, "no_bal": False, "zbins": [2.0,2.5,3.0,3.5,4.0]}
        make_slug(cfg, defaults) == "snr2_nobal_z2-3.5"

    Keys equal to their default contribute nothing. When nothing differs, returns
    a stable non-empty base token (``"base"``) so a leaf dir is never just the hash.
    Token order follows the config dict's insertion order (deterministic for a
    given producer, which builds the config in a fixed order); this preserves the
    natural ``snr2_nobal_z2-3.5`` reading rather than an alphabetical reshuffle.
    """
    tokens: list[str] = []
    for key in config:
        val = config[key]
        if key in producer_defaults and producer_defaults[key] == val:
            continue
        tokens.append(_slug_token(key, val))
    slug = "_".join(t for t in tokens if t)
    if not slug:
        slug = "base"
    # final filesystem-safety pass: no spaces/slashes/control chars.
    return _fs_safe(slug)


def _fs_safe(s: str) -> str:
    """Replace any character that is not [A-Za-z0-9._-] with '-'."""
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


# --------------------------------------------------------------------------- #
# privacy contagion                                                            #
# --------------------------------------------------------------------------- #
def privacy_class(input_provs: list[dict]) -> dict:
    """Derive a result's privacy class from its inputs' privacy (contagious down).

    A result is ``real-LOA`` iff ANY input's ``privacy.class`` is ``"real-LOA"``;
    otherwise ``mock``. Only ``mock`` results are shareable. With no inputs the
    result is ``mock`` (e.g. a pure-mock seed leaf).

    Each input is a provenance-like dict carrying at least
    ``{"privacy": {"class": ...}}``.
    """
    is_real = any(
        (inp.get("privacy") or {}).get("class") == "real-LOA"
        for inp in (input_provs or [])
    )
    cls = "real-LOA" if is_real else "mock"
    return {"class": cls, "shareable": cls == "mock"}


# --------------------------------------------------------------------------- #
# atomic provenance writer (README.md + provenance.json)                       #
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (write tmp in same dir + os.replace)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _render_readme(rec: dict) -> str:
    """Generalize ``tutorial_data/README.md``: a one-line what-this-is, a status/
    privacy header, the commit stamp (+ ⚠ DIRTY banner when dirty), a
    ``file | what it is`` outputs table, the inputs, and the regen command."""
    cc = rec["code_commit"]
    dirty = bool(cc.get("dirty"))
    lines: list[str] = []
    lines.append(f"# {rec['id']}")
    lines.append("")
    lines.append(rec["what"])
    lines.append("")
    if dirty:
        lines.append("> ## ⚠ DIRTY")
        lines.append(">")
        lines.append(
            "> Built from a **dirty** working tree — the recorded commit does "
            "NOT fully describe this result. See `diff_sha256` in "
            "`provenance.json` for the uncommitted-change fingerprint."
        )
        lines.append("")

    # status / privacy / producer / commit / date header.
    priv = rec["privacy"]
    share = "yes" if priv.get("shareable") else "no"
    lines.append(f"- **status:** {rec['status']}")
    lines.append(f"- **privacy:** {priv.get('class')} (shareable: {share})")
    lines.append(f"- **producer:** `{rec['producer']}`")
    commit_line = f"- **code commit:** `{cc.get('commit_short')}`"
    if cc.get("commit_long") and cc.get("commit_long") != "unknown":
        commit_line += f" (`{cc.get('commit_long')}`)"
    if cc.get("branch") and cc.get("branch") != "unknown":
        commit_line += f" on `{cc.get('branch')}`"
    if dirty:
        commit_line += "  ⚠ DIRTY"
    lines.append(commit_line)
    lines.append(f"- **date (UTC):** {rec['date_utc']}")
    lines.append(f"- **config hash:** `{rec['config_hash']}`  (slug `{rec['slug']}`)")
    lines.append("")

    # inputs (transitive provenance: each input id + privacy).
    lines.append("## Inputs")
    lines.append("")
    if rec["inputs"]:
        for inp in rec["inputs"]:
            iid = inp.get("id", inp.get("path", "<external>"))
            ipriv = (inp.get("privacy") or {}).get("class", "?")
            lines.append(f"- `{iid}` — privacy: {ipriv}")
    else:
        lines.append("- (none)")
    lines.append("")

    # outputs table: file | what it is.
    lines.append("## Outputs")
    lines.append("")
    lines.append("| file | what it is |")
    lines.append("|------|------------|")
    for fname, desc in rec["outputs"]:
        lines.append(f"| `{fname}` | {desc} |")
    lines.append("")

    # exact CLI invocation.
    lines.append("## CLI")
    lines.append("")
    lines.append("```bash")
    lines.append(rec["cli"])
    lines.append("```")
    lines.append("")

    # regenerate command.
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append(rec["regen_cmd"])
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_provenance(
    leaf_dir: str,
    *,
    what: str,
    status: str,
    privacy: dict,
    producer: str,
    config: dict,
    inputs: list[dict],
    cli: str,
    outputs: list[tuple],
    regen_cmd: str,
    code_commit: dict | None = None,
) -> dict:
    """Atomically emit ``README.md`` (human) + ``provenance.json`` (machine) into
    ``leaf_dir``, returning the machine record.

    ``code_commit`` defaults to ``git_stamp()`` of the repo. When the stamp is
    dirty the README carries a ``⚠ DIRTY`` banner. Both files are written via a
    same-dir tmp + ``os.replace`` so a reader never sees a half-written file and
    the pair cannot drift (one call writes both).

    ``provenance.json`` schema (``schema_version="cddf-provenance/1"``)::

        {schema_version, id, dataset, stage, producer, config_hash, config, slug,
         code_commit, date_utc, inputs, outputs, cli, regen_cmd, status, privacy,
         supersedes:null, superseded_by:null}

    ``id`` is the leaf-dir basename (the stable handle the manifest also keys on);
    ``dataset``/``stage`` are inferred from the leaf path when it is laid out as
    ``.../{dataset}/{stage}/{slug}__{hash}/`` (the store layout), else "".
    """
    leaf = Path(leaf_dir)
    leaf.mkdir(parents=True, exist_ok=True)

    if code_commit is None:
        code_commit = git_stamp()

    cfg_hash = config_hash(config)

    # Infer dataset / stage from the store layout when present.
    dataset, stage = _infer_dataset_stage(leaf)

    rec = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "id": leaf.name,
        "what": what,
        "dataset": dataset,
        "stage": stage,
        "producer": producer,
        "config_hash": cfg_hash,
        "config": config,
        "slug": _slug_from_leafname(leaf.name, cfg_hash),
        "code_commit": code_commit,
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": [list(o) for o in outputs],  # JSON has no tuples.
        "cli": cli,
        "regen_cmd": regen_cmd,
        "status": status,
        "privacy": privacy,
        "supersedes": None,
        "superseded_by": None,
    }

    # README needs the original tuple-form outputs; pass the record but render
    # from rec["outputs"] (list-of-list, iterable the same way).
    _atomic_write(leaf / "provenance.json", json.dumps(rec, indent=2, sort_keys=False))
    _atomic_write(leaf / "README.md", _render_readme(rec))
    return rec


def _infer_dataset_stage(leaf: Path) -> tuple:
    """Best-effort (dataset, stage) from a store-layout leaf path
    ``.../{privacy}/{dataset}/{stage}/{leafname}``. Returns ("","") if the path
    is not laid out that way (e.g. a bare tmp dir in a test)."""
    parts = leaf.parts
    if len(parts) >= 4:
        # leaf is parts[-1]; stage parts[-2]; dataset parts[-3]; privacy parts[-4].
        return parts[-3], parts[-2]
    return "", ""


def _slug_from_leafname(name: str, cfg_hash: str) -> str:
    """A store leaf dir is ``{slug}__{hash}``; recover the slug. If the name is
    not in that form (bare tmp dir), fall back to the name itself."""
    suffix = "__" + cfg_hash
    if name.endswith(suffix):
        return name[: -len(suffix)]
    if "__" in name:
        return name.rsplit("__", 1)[0]
    return name
