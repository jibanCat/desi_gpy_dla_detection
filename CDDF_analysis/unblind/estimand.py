"""estimand.py -- the ESTIMAND vocabulary + a classifier for stamped artifacts.

WHY THIS EXISTS
---------------
The PI decision (2026-07-28) is that paper-facing uncertainty bands must be CREDIBLE
INTERVALS of a faithful joint posterior, with the reported POINT being the MEDIAN of
that SAME posterior.  Historically this repo emitted several *different* objects under
the single word "band", and nothing on the artifact said which one it was:

  * quantiles of a joint posterior whose median is the point           (admissible)
  * quantiles of an MC/bootstrap ensemble whose CENTRE IS NOT the point (not admissible
    as a credible interval: the point is a plug-in MAP, the cloud is elsewhere)
  * the same MC cloud RIGIDLY SLID onto the plug-in point so that q50 == point by
    construction                                                        (retired)
  * per-bin / per-limit marginal intervals combined as if independent
    (quadrature, np.hypot) without their covariance                     (retired)

Every band-writing path must now stamp ``metadata['estimand']`` with one of the strings
below, plus ``metadata['paper_facing']``.  Legacy artifacts that predate the stamp are
still classifiable -- see :func:`classify_estimand`, which infers the class from the
band structure itself (a recentered band is detectable: ``q50 == MAP`` to machine
precision, and/or a recorded non-zero ``jensen_shift``).

VOCABULARY
----------
POSTERIOR_MEDIAN_CI
    The band is a set of quantiles of a joint posterior, and the reported point IS the
    median of that same posterior.  The ONLY class admissible as a paper-facing
    uncertainty.  Requires sampler diagnostics on the artifact
    (``metadata['sampler']`` with r_hat / ESS / divergences).
PLUGIN_MAP_MC
    Point is a plug-in optimum (MAP / in-data integral); band is an MC or bootstrap
    ensemble around a DIFFERENT centre.  Honest but NOT a credible interval: the point
    and the band are different estimands.  Diagnostic only.
DIAGNOSTIC_RECENTERED
    A PLUGIN_MAP_MC cloud additively slid so its median sits on the point.  RETIRED for
    paper-facing use (PI, 2026-07-28).  Diagnostic only.
MARGINAL_COMBINED
    Per-bin / per-channel marginal intervals combined without their covariance
    (quadrature / np.hypot / independent-sigma sums).  Diagnostic only.
POINT_ONLY
    No band on the artifact.
UNKNOWN
    Could not be classified -- treat as NOT paper-facing.

This module is stdlib-only (json + os) so it can be imported from anywhere, including
``cddf_catalog_hbi`` (no numpy/scipy import cost, no circular import).
"""

from __future__ import annotations

import json
import os

# --- the vocabulary ---------------------------------------------------------
POSTERIOR_MEDIAN_CI = "POSTERIOR_MEDIAN_CI"
PLUGIN_MAP_MC = "PLUGIN_MAP_MC"
DIAGNOSTIC_RECENTERED = "DIAGNOSTIC_RECENTERED"
MARGINAL_COMBINED = "MARGINAL_COMBINED"
POINT_ONLY = "POINT_ONLY"
UNKNOWN = "UNKNOWN"

ESTIMAND_VOCABULARY = frozenset(
    {POSTERIOR_MEDIAN_CI, PLUGIN_MAP_MC, DIAGNOSTIC_RECENTERED,
     MARGINAL_COMBINED, POINT_ONLY, UNKNOWN}
)

#: the ONLY estimand class a paper-facing band may carry (PI, 2026-07-28).
PAPER_FACING_ESTIMANDS = frozenset({POSTERIOR_MEDIAN_CI})

#: classes that ASSERT THE EXISTENCE OF A BAND.  An artifact stamped with one of these
#: while exposing zero band records is making a claim its own body does not support, and
#: :func:`classify_estimand` refuses to certify it (referee defect 2, 2026-07-29).
BAND_BEARING_ESTIMANDS = frozenset(
    {POSTERIOR_MEDIAN_CI, PLUGIN_MAP_MC, DIAGNOSTIC_RECENTERED, MARGINAL_COMBINED})

#: stamped on any artifact whose band class was demoted by the 2026-07-28 retirement.
RETIRED_REASON_RECENTERED = (
    "RETIRED 2026-07-28 (PI): band is an MC/bootstrap cloud rigidly recentered on a "
    "plug-in MAP point (recenter_band_on_point). q50==point by construction, so the "
    "band is NOT a credible interval of the posterior whose median is the point. "
    "Diagnostic use only; artifact retained, never deleted."
)
RETIRED_REASON_PLUGIN_MC = (
    "RETIRED 2026-07-28 (PI): point is a plug-in optimum and the band is an MC ensemble "
    "around a DIFFERENT centre -- point and band are different estimands. "
    "Diagnostic use only; artifact retained, never deleted."
)
RETIRED_REASON_MARGINAL = (
    "RETIRED 2026-07-28 (PI): interval combines per-bin/per-channel marginals as if "
    "independent (quadrature) without their covariance. "
    "Diagnostic use only; artifact retained, never deleted."
)

_RETIRED_REASON_BY_CLASS = {
    DIAGNOSTIC_RECENTERED: RETIRED_REASON_RECENTERED,
    PLUGIN_MAP_MC: RETIRED_REASON_PLUGIN_MC,
    MARGINAL_COMBINED: RETIRED_REASON_MARGINAL,
}


class RecenteredBandRetired(RuntimeError):
    """Raised when recentering is requested on a path that is not explicitly
    declared diagnostic-only.  See :mod:`CDDF_analysis.unblind.estimand`."""


#: metadata key that carries the free-text / structured DESCRIPTION accompanying the
#: one-word ``estimand`` label.  See :func:`normalize_estimand_stamp`.
ESTIMAND_DETAIL_KEY = "estimand_detail"

#: metadata key recording WHY certification was refused, so the refusal survives to disk.
PAPER_FACING_REFUSED_KEY = "paper_facing_refused"

# --- how a producer's paper_facing declaration is READ ----------------------
DECLARATION_ABSENT = "ABSENT"
DECLARATION_TRUE = "TRUE"
DECLARATION_FALSE = "FALSE"
DECLARATION_UNPARSEABLE = "UNPARSEABLE"

_TRUE_TOKENS = frozenset({"true", "yes", "y", "t", "1"})
_FALSE_TOKENS = frozenset({"false", "no", "n", "f", "0"})

#: sentinel for "the key is not present at all" (distinct from a present ``None``).
ABSENT = type("_Absent", (), {"__repr__": lambda s: "<ABSENT>"})()


def parse_paper_facing_declaration(value):
    """Read a producer's ``metadata['paper_facing']`` into one of four verdicts.

    The veto used to be ``declared_pf is False`` -- an IDENTITY test against the single
    ``False`` object.  Nothing on disk was mis-certified by that (``json`` decodes
    ``false`` to the singleton), but a producer writing ``0``, ``''``, ``'false'`` or
    ``'no'`` was silently CERTIFIED paper-facing.  The last line of defence has to
    parse, not to compare identities.

    Returns :data:`DECLARATION_TRUE` / :data:`DECLARATION_FALSE` /
    :data:`DECLARATION_UNPARSEABLE` / :data:`DECLARATION_ABSENT`.

    DELIBERATE DECISIONS:

    * ``None`` and a MISSING key are both ABSENT.  JSON ``null`` is indistinguishable
      from an absent key to every reader in this repo, and "no declaration" must mean
      the same thing however it is spelled.  ABSENT is NOT a veto -- the classifier
      then infers the class from structure, as it always did.
    * anything we cannot read as a yes or a no is UNPARSEABLE, and the CALLER refuses
      to certify on it.  We do not know what the producer meant, and guessing in the
      permissive direction is exactly how a fail-open ships.
    * ``bool`` is checked before ``int`` (``isinstance(True, int)`` is True in Python),
      and NaN is UNPARSEABLE rather than "not 1 and not 0".
    """
    if value is ABSENT or value is None:
        return DECLARATION_ABSENT
    if isinstance(value, bool):
        return DECLARATION_TRUE if value else DECLARATION_FALSE
    if isinstance(value, (int, float)):
        if value != value:                       # NaN
            return DECLARATION_UNPARSEABLE
        if value == 1:
            return DECLARATION_TRUE
        if value == 0:
            return DECLARATION_FALSE
        return DECLARATION_UNPARSEABLE
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return DECLARATION_TRUE
        if token in _FALSE_TOKENS:
            return DECLARATION_FALSE
        return DECLARATION_UNPARSEABLE
    return DECLARATION_UNPARSEABLE

#: keys inside a dict-valued ``estimand`` stamp that may carry the vocabulary word.
_DETAIL_LABEL_KEYS = ("estimand", "class", "label", "kind", "vocabulary")


def normalize_estimand_stamp(value, detail=None):
    """Accept EITHER ``metadata['estimand']`` schema; return ``(label, detail)``.

    Two incompatible schemas are in circulation in this repo:

      * SCHEMA A (canonical) -- a single vocabulary STRING, e.g. ``"PLUGIN_MAP_MC"``.
      * SCHEMA B (descriptive) -- a DICT of free-text prose keyed by aspect
        (``quantity`` / ``point`` / ``interval`` / ``ff`` / ``hbi`` / ...), written by
        ``crossmock_transfer_artifact.json`` and ``calccddf_vs_hbi.json``.

    Schema B carries no vocabulary word, so it cannot be compared against
    ``ESTIMAND_VOCABULARY`` -- and doing so literally crashed
    (``TypeError: unhashable type: 'dict'``), which is worse than misclassifying:
    the artifact could not be audited at all.

    ``label`` is a member of :data:`ESTIMAND_VOCABULARY`, or ``None`` when the stamp
    carries no vocabulary word (the caller must then INFER the class).  ``detail`` is
    the descriptive dict, or ``None``.  Nothing is ever discarded: a non-vocabulary
    string is preserved under ``detail['legacy_estimand_text']``.
    """
    out_detail = dict(detail) if isinstance(detail, dict) else {}
    if isinstance(value, dict):
        for k in _DETAIL_LABEL_KEYS:
            v = value.get(k)
            if isinstance(v, str) and v in ESTIMAND_VOCABULARY:
                label = v
                break
        else:
            label = None
        out_detail.update(value)
        return label, (out_detail or None)
    if value is None:
        return None, (out_detail or None)
    if isinstance(value, str):
        if value in ESTIMAND_VOCABULARY:
            return value, (out_detail or None)
        out_detail.setdefault("legacy_estimand_text", value)
        return None, out_detail
    out_detail.setdefault("legacy_estimand_text", repr(value))
    return None, out_detail


def normalize_estimand_metadata(metadata: dict, *, label: str = None) -> dict:
    """EMIT ONE SCHEMA: rewrite ``metadata`` in place so ``metadata['estimand']`` is a
    vocabulary STRING and any descriptive dict lives under
    ``metadata[ESTIMAND_DETAIL_KEY]``.  Returns ``metadata``.  Idempotent.

    ``label`` overrides the label read from the stamp; when neither is available the
    label becomes :data:`UNKNOWN` (explicitly unclassified, never silently admissible).
    """
    read_label, detail = normalize_estimand_stamp(
        metadata.get("estimand"), metadata.get(ESTIMAND_DETAIL_KEY))
    metadata["estimand"] = label or read_label or UNKNOWN
    if detail:
        metadata[ESTIMAND_DETAIL_KEY] = detail
    return metadata


def is_paper_facing(estimand) -> bool:
    """True only for the estimand classes admissible as a paper-facing band.

    Tolerates a dict-valued (schema B) stamp: it is normalised first, and a stamp with
    no vocabulary word is NOT paper-facing.
    """
    if not isinstance(estimand, str):
        estimand, _ = normalize_estimand_stamp(estimand)
    return estimand in PAPER_FACING_ESTIMANDS


# ---------------------------------------------------------------------------
# stamping
# ---------------------------------------------------------------------------
def band_estimand(*, band_recenter: bool, posterior_sampled: bool = False,
                  marginal_combined: bool = False, has_band: bool = True) -> str:
    """Resolve the estimand class from the three facts that determine it.

    ``posterior_sampled`` means the band quantiles and the reported point come from
    the SAME joint posterior draws (point == posterior median).  It is NOT enough that
    an MC ensemble exists: the plug-in-MAP + bootstrap-cloud path is PLUGIN_MAP_MC.
    """
    if not has_band:
        return POINT_ONLY
    if band_recenter:
        return DIAGNOSTIC_RECENTERED
    if marginal_combined:
        return MARGINAL_COMBINED
    if posterior_sampled:
        return POSTERIOR_MEDIAN_CI
    return PLUGIN_MAP_MC


def stamp_band_estimand(metadata: dict, *, band_recenter: bool,
                        posterior_sampled: bool = False,
                        marginal_combined: bool = False,
                        has_band: bool = True) -> dict:
    """Stamp ``metadata`` in place with ``estimand`` + ``paper_facing`` (+
    ``retired_reason`` when the class is not paper-facing).  Returns ``metadata``.

    Every band-writing path must call this before the artifact is serialized, so that
    the artifact SELF-DECLARES which estimand it is.
    """
    est = band_estimand(band_recenter=band_recenter,
                        posterior_sampled=posterior_sampled,
                        marginal_combined=marginal_combined,
                        has_band=has_band)
    # migrate any pre-existing descriptive (schema B) stamp into ``estimand_detail``
    # instead of silently clobbering it -- that prose is the only record of WHAT the
    # number is on several artifacts.
    normalize_estimand_metadata(metadata, label=est)
    metadata["band_paper_facing"] = bool(is_paper_facing(est))
    # ``paper_facing`` is a CONJUNCTION of every admissibility gate on the artifact
    # (the resp_kind/kernel gate in CDDF_analysis.unblind.resp_kind already writes it),
    # so ANDing is the only safe update: a gate may veto, never re-authorize.
    #
    # The prior must be PARSED, not truthiness-tested: ``bool('false')`` is True, so a
    # prior veto spelled as a string was silently re-authorized here (the writer-side
    # twin of the classifier's ``is False`` defect).  Only an explicit YES or NO
    # DECLARATION may pass; ABSENT means no prior gate has spoken, and UNPARSEABLE is
    # refused like a veto.
    prior_decl = parse_paper_facing_declaration(
        metadata.get("paper_facing", ABSENT))
    prior_ok = prior_decl in (DECLARATION_TRUE, DECLARATION_ABSENT)
    metadata["paper_facing"] = bool(prior_ok) and bool(is_paper_facing(est))
    metadata["band_recenter"] = bool(band_recenter)
    if est in _RETIRED_REASON_BY_CLASS:
        metadata.setdefault("retired_reason", _RETIRED_REASON_BY_CLASS[est])
    return metadata


def assert_paper_facing(metadata: dict, where: str = "") -> None:
    """Fail loudly if ``metadata`` claims paper_facing=True while carrying an estimand
    class that the PI decision retired.  Cheap guard for headline writers."""
    label, _detail = normalize_estimand_stamp(metadata.get("estimand"))
    est = label or UNKNOWN     # a descriptive-only (schema B) stamp is UNKNOWN, not OK
    raw = metadata.get("paper_facing", ABSENT)
    decl = parse_paper_facing_declaration(raw)
    if decl == DECLARATION_UNPARSEABLE:
        # ``bool('maybe')`` is True and ``bool('false')`` is True: truthiness cannot be
        # trusted here.  A headline writer that cannot state cleanly whether it is
        # paper-facing is refused rather than waved through in either direction.
        raise RecenteredBandRetired(
            f"{where or 'artifact'}: metadata['paper_facing']={raw!r} is UNPARSEABLE "
            "-- state it as a JSON boolean. Refusing to guess.")
    if decl == DECLARATION_TRUE and not is_paper_facing(est):
        raise RecenteredBandRetired(
            f"{where or 'artifact'}: metadata['paper_facing']=True but "
            f"metadata['estimand']={est!r}, which is NOT a faithful joint-posterior "
            f"credible interval. Admissible: {sorted(PAPER_FACING_ESTIMANDS)}.")


# ---------------------------------------------------------------------------
# classification of EXISTING artifacts (legacy, unstamped)
# ---------------------------------------------------------------------------
_BAND_KEYS = ("q16", "q84", "q025", "q975", "band68", "band95",
              "lo68", "hi68", "lo95", "hi95", "f68_lo", "f68_hi")
# NOTE: ``point_q50`` / ``q50`` MUST be here.  A faithful posterior artifact
# (``run_posterior.py``) keys its reported point as ``point_q50`` beside q16/q84 and
# carries NO ``MAP``/``point`` key anywhere.  While they were missing, such an artifact
# yielded ZERO band records, took the POINT_ONLY early-return, and every structural
# check was silently skipped -- the classifier then trusted the free-text stamp.
_POINT_KEYS = ("MAP", "point", "f", "MAP_R0", "point_q50", "q50")


def _walk(node, path=""):
    """Yield (path, dict) for every dict in a nested JSON structure."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _band_records(art):
    """Every dict that looks like a band record: has >=1 band key and >=1 point key."""
    out = []
    for path, node in _walk(art):
        if any(k in node for k in _BAND_KEYS) and any(k in node for k in _POINT_KEYS):
            out.append((path, node))
    return out


def _looks_recentered(rec):
    """A recentered band is detectable WITHOUT the stamp:
    q50 == MAP exactly (the rigid shift makes it exact), or a recorded non-zero
    ``jensen_shift`` / ``raw_median`` that differs from the point."""
    if "jensen_shift" in rec:
        try:
            if abs(float(rec["jensen_shift"])) > 0.0:
                return True, "non-zero jensen_shift recorded"
        except (TypeError, ValueError):
            pass
    pt = rec.get("MAP", rec.get("point"))
    q50 = rec.get("q50")
    if pt is not None and q50 is not None:
        try:
            pt, q50 = float(pt), float(q50)
        except (TypeError, ValueError):
            return False, ""
        if pt == q50:
            return True, "q50 == MAP exactly (rigid recenter signature)"
    return False, ""


def classify_estimand(artifact, name: str = "") -> dict:
    """Classify a loaded artifact dict (or a path to a JSON file).

    Returns ``dict(name, estimand, paper_facing, stamped, evidence, n_band_records)``.
    The STAMP wins when present and self-consistent; otherwise the class is INFERRED
    from the band structure.  When the stamp and the structure disagree, the structural
    evidence wins and it is reported in ``evidence`` (a stamp cannot launder a band).
    """
    if isinstance(artifact, str):
        name = name or artifact
        with open(artifact) as fh:
            artifact = json.load(fh)
    meta = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    evidence = []
    raw_stamp = meta.get("estimand")
    stamped, detail = normalize_estimand_stamp(raw_stamp, meta.get(ESTIMAND_DETAIL_KEY))
    if isinstance(raw_stamp, dict):
        evidence.append(
            f"stamped estimand is a DESCRIPTIVE DICT (schema B, keys "
            f"{sorted(raw_stamp)}); vocabulary label={stamped!r}")
    elif raw_stamp is not None:
        evidence.append(f"stamped estimand={raw_stamp!r} -> label={stamped!r}")

    # The PRODUCER'S VETO.  ``metadata['paper_facing']`` is written by the routine that
    # made the artifact and knows things the file's shape cannot show (here: that the
    # molly calibration set IS the same synthetic mock, so the run is a self-
    # calibration smoke).  Inference may only ever REMOVE paper-facing status, never
    # restore it, so this is ANDed into every return path below.
    declared_pf = meta.get("paper_facing")

    declaration = parse_paper_facing_declaration(declared_pf)

    def _finish(est, n_records):
        # Every refusal is collected and RECORDED even when the class alone already
        # disqualifies the artifact: "unclassifiable" must never be a SILENT pass, and
        # a reader needs to know which gate closed.
        refusals = []
        if declaration == DECLARATION_FALSE:
            refusals.append(
                "producer stamped metadata['paper_facing']=False -- the producer's "
                "veto WINS over the inferred class (a gate may veto, never "
                "re-authorize)")
        elif declaration == DECLARATION_UNPARSEABLE:
            refusals.append(
                f"producer's metadata['paper_facing']={declared_pf!r} is UNPARSEABLE "
                "(neither a recognisable yes nor no): refusing to certify on a "
                "declaration we cannot read")
        if is_paper_facing(est) and n_records == 0 and est in BAND_BEARING_ESTIMANDS:
            refusals.append(
                f"artifact declares the band-bearing estimand {est!r} but exposes ZERO "
                "band records: a free-text stamp cannot substantiate a band that is "
                "not in the file (band keys "
                f"{list(_BAND_KEYS[:4])}... beside a point key {list(_POINT_KEYS)})")
        pf = is_paper_facing(est) and not refusals
        evidence.extend(r for r in refusals if r not in evidence)
        return dict(name=name, estimand=est, paper_facing=pf, stamped=stamped,
                    estimand_detail=detail, producer_paper_facing=declared_pf,
                    producer_declaration=declaration, refusals=refusals,
                    evidence=evidence, n_band_records=n_records)

    recs = _band_records(artifact)
    if not recs:
        est = POINT_ONLY
        evidence.append("no band record found (point-only artifact)")
        if stamped is not None and stamped != POINT_ONLY:
            est = stamped
        return _finish(est, 0)

    n_rec = 0
    for path, rec in recs:
        hit, why = _looks_recentered(rec)
        if hit:
            n_rec += 1
            if len(evidence) < 8:
                evidence.append(f"{path or '/'}: {why}")
    # Recorded band_recenter flags ANYWHERE in the artifact (top-level metadata, or the
    # per-variant ``coverage/_meta`` blocks the cross-mock drivers write).
    flags = [(p, n["band_recenter"]) for p, n in _walk(artifact)
             if isinstance(n, dict) and "band_recenter" in n]
    for p, v in flags:
        if v is True:
            n_rec = max(n_rec, 1)
            if len(evidence) < 10:
                evidence.append(f"{p or '/metadata'}: band_recenter is True")

    # PROVENANCE RULE for legacy track_c_tf_* artifacts. That writer emits per-limit
    # bands as {MAP,q16,q84,q025,q975,std} -- NO q50 -- so the "q50 == MAP" structural
    # signature is unavailable, and it stamped no band_recenter key. But until
    # 2026-07-28 its CLI default WAS band_recenter=True, and its per-z Omega and
    # per-(logN,z) f(N|z) bands called recenter_band_on_point UNCONDITIONALLY (ignoring
    # the flag entirely). So every artifact of this shape without an explicit
    # band_recenter=False stamp carries at least one recentered band.
    if (not n_rec and not flags and isinstance(artifact, dict)
            and {"measurement", "zbins"} <= set(artifact)):
        n_rec = 1
        evidence.append(
            "legacy track_c_tf_loa shape with no band_recenter flag anywhere: that "
            "writer's CLI defaulted to band_recenter=True and its per-z Omega / "
            "f(N|z) bands recentered UNCONDITIONALLY before 2026-07-28")

    if n_rec:
        est = DIAGNOSTIC_RECENTERED
    elif stamped is not None:
        est = stamped
    else:
        # unstamped (or descriptive-only), not recentered: the historical default on
        # every band writer in this repo is a plug-in MAP point with a bootstrap/MC
        # cloud around it.
        est = PLUGIN_MAP_MC
        evidence.append("unstamped, not recentered -> historical plug-in MAP + MC cloud")

    return _finish(est, len(recs))


def _sniff_indent(raw: str, default: int = 2) -> int:
    """Best-effort original indent width, so re-stamping an artifact does not reformat
    the whole file into a giant meaningless diff."""
    for line in raw.split("\n", 40)[1:40]:
        stripped = line.lstrip(" ")
        n = len(line) - len(stripped)
        if n and stripped:
            return n
    return default


def mark_retired(path: str, *, dry_run: bool = True,
                 skip_point_only: bool = True) -> dict:
    """Classify the artifact at ``path`` and, unless ``dry_run``, stamp its metadata
    with ``estimand`` / ``paper_facing`` / ``retired_reason`` IN PLACE.

    The artifact is NEVER deleted and no science value is touched -- only the
    ``metadata`` block is written, at the file's original indent width.

    ``skip_point_only`` (default) leaves band-free artifacts untouched on disk: they
    carry no uncertainty band, so there is nothing to reclassify and rewriting them
    would only churn committed files.
    """
    with open(path) as fh:
        raw = fh.read()
    art = json.loads(raw)
    res = classify_estimand(art, name=path)
    res["written"] = False
    if not isinstance(art, dict):
        return res
    if skip_point_only and res["estimand"] == POINT_ONLY:
        return res
    meta = art.setdefault("metadata", {})
    # normalise to ONE schema on write: vocabulary string in ``estimand``, any
    # descriptive dict preserved verbatim under ``estimand_detail``.
    normalize_estimand_metadata(meta, label=res["estimand"])
    meta["paper_facing"] = bool(res["paper_facing"])
    # Persist WHY certification was refused.  A refusal that lives only in the
    # classifier's return value is re-litigated by the next reader of the file.
    if res.get("refusals"):
        meta[PAPER_FACING_REFUSED_KEY] = list(res["refusals"])
    if res["estimand"] in _RETIRED_REASON_BY_CLASS:
        meta["retired_reason"] = _RETIRED_REASON_BY_CLASS[res["estimand"]]
        meta.setdefault("retired_on", "2026-07-28")
    if not dry_run:
        with open(path, "w") as fh:
            json.dump(art, fh, indent=_sniff_indent(raw))
        res["written"] = True
    return res


def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Classify (and optionally retire-stamp) band artifacts.")
    p.add_argument("paths", nargs="+", help="JSON artifact paths")
    p.add_argument("--mark-retired", action="store_true",
                   help="write metadata estimand/paper_facing/retired_reason in place")
    p.add_argument("--include-point-only", action="store_true",
                   help="also rewrite band-free (POINT_ONLY) artifacts")
    a = p.parse_args(argv)
    rows = []
    for path in a.paths:
        if not os.path.exists(path):
            print(f"{path}: MISSING")
            continue
        try:
            r = mark_retired(path, dry_run=not a.mark_retired,
                             skip_point_only=not a.include_point_only)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"{path}: UNREADABLE ({e})")
            continue
        rows.append(r)
        flag = "written" if r.get("written") else "dry-run"
        print(f"{os.path.basename(path):48s} {r['estimand']:22s} "
              f"paper_facing={str(r['paper_facing']):5s} [{flag}]")
        for e in r["evidence"][:3]:
            print(f"      - {e}")
    return rows


if __name__ == "__main__":  # pragma: no cover
    _main()
