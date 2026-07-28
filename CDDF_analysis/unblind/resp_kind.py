"""resp_kind.py -- NO-DEFAULT response-kernel resolution + artifact self-declaration.

WHY THIS EXISTS (2026-07-28)
----------------------------
``HBIConfig.resp_kind`` (``CDDF_analysis/hbi/cddf_catalog_hbi.py``) defaults to
``"kappa"`` -- the GP-POSTERIOR kernel -- which Track-C established is the WRONG OBJECT
for the catalog-HBI CDDF.  ``provenance.assert_forward_kernel`` is the right guard, but it
is OPT-IN: a routine that never calls it silently inherits the kappa default.

An audit of BOTH worktrees (``desi_gpy_dla_detection`` @ lls-subdla-cddf and
``hbi_mcmc_wt`` @ hbi-mcmc-threeroute) found that **not one** of the 22
``HBIConfig(...)`` construction sites passes ``resp_kind``.  The forward kernel is NEVER a
construction-time choice; it is ALWAYS a post-hoc mutation (``cfg.resp_kind = "forward"``,
normally via ``track_c_perz_band._set_forward_cfg``).  Therefore:

    "forgot the mutation"  ==  "silently kappa"

and that is exactly how the RETIRED sub-DLA anchor (0.883/0.899, superseded by the forward
0.849/0.822) was produced.

WHY NOT JUST FLIP THE DATACLASS DEFAULT
---------------------------------------
Two verified reasons, both structural:

1. ``cddf_catalog_hbi.py`` is byte-pinned at commit ``8816e1e`` by the committed
   acceptance test ``tests/test_subdla_forward_headline.py::
   test_frozen_files_unchanged_by_forward_switch`` (AC14) -- the invariant being that the
   forward switch is CONFIG-ONLY and touches no estimator code.  Editing line ~172 turns a
   green invariant red for a reason the invariant does not protect against.
2. A default of ``"forward"`` with ``kernel_forward_model=None`` makes ``build_A_ib``
   raise at all 22 construction sites.  It would break every legitimate kappa DIAGNOSTIC
   while making zero paper-facing paths more correct -- none of them rely on the default;
   they all mutate.

The accident that actually matters is **a PAPER-FACING artifact built on kappa**.  That is
closed here, at the entry-point / stamp layer, which is not frozen:

    resolve_resp_kind(...)      NO DEFAULT.  Missing/None raises.  'kappa' raises when
                                paper_facing=True.  Every public entry point must state
                                the kernel explicitly.
    kernel_metadata(...)        Builds the metadata block so an artifact SELF-DECLARES
                                metadata['resp_kind'] (+ paper_facing + kernel_note).
                                Resolution runs first, so it is also the write-side guard.
    assert_artifact_kernel(...) Fail-closed READ side: an artifact that does not
                                self-declare resp_kind is REJECTED (silence is not
                                consent), and paper_facing=True on kappa is REJECTED.

A LABELLED kappa diagnostic is legitimate and stays runnable -- an UNLABELLED one does not.
"""

from __future__ import annotations

from CDDF_analysis.unblind.provenance import ProvenanceError

__all__ = [
    "RESP_KIND_FORWARD", "RESP_KIND_KAPPA", "KNOWN_RESP_KINDS",
    "PAPER_FACING_RESP_KIND", "KAPPA_WHY",
    "resolve_resp_kind", "kernel_metadata", "assert_artifact_kernel",
]

RESP_KIND_FORWARD = "forward"
RESP_KIND_KAPPA = "kappa"
KNOWN_RESP_KINDS = (RESP_KIND_FORWARD, RESP_KIND_KAPPA)

#: The only kernel a paper-facing artifact may be built on (Track-C "right object").
PAPER_FACING_RESP_KIND = RESP_KIND_FORWARD

KAPPA_WHY = (
    "the GP-POSTERIOR ('kappa') kernel is the WRONG OBJECT for the catalog-HBI CDDF "
    "(Track-C: it over-recovers the steep high-N tail; DLA-tier R0>=20.3 ~1.16 posterior "
    "vs ~1.04 forward, and it produced the RETIRED sub-DLA anchor 0.883/0.899 vs the "
    "forward 0.849/0.822)"
)

# Sentinel for "the caller did not state a kernel at all" -- deliberately distinct from
# None so that passing None explicitly is ALSO rejected (and with a different message).
_UNSET = object()


def resolve_resp_kind(cfg_or_resp_kind=_UNSET, *, context, paper_facing):
    """FAIL-CLOSED, NO-DEFAULT resolution of a response-kernel selection.

    Unlike ``getattr(cfg, "resp_kind", "kappa")`` -- the idiom this function exists to
    replace -- there is **no default**: a caller that does not state a kernel gets a
    ``ProvenanceError``, never a silent ``"kappa"``.

    Parameters
    ----------
    cfg_or_resp_kind : HBIConfig-like or str
        An object carrying ``.resp_kind`` (read) or a bare kernel string.
    context : str
        Required.  Names the call site in the error (e.g. "sub-DLA headline stamp").
    paper_facing : bool
        Required, and it is the whole point.  ``True`` => only
        ``PAPER_FACING_RESP_KIND`` ('forward') is accepted and ``'kappa'`` raises.
        ``False`` => a LABELLED kappa diagnostic is allowed (and returned).

    Returns
    -------
    str
        The validated resp_kind: ``'forward'``, or ``'kappa'`` when
        ``paper_facing=False``.
    """
    if cfg_or_resp_kind is _UNSET:
        raise ProvenanceError(
            f"[resp_kind] {context}: no response kernel was stated. resp_kind has NO "
            f"default at a public entry point -- pass resp_kind='forward' (the measured "
            f"forward response, the only paper-facing kernel) or, for a LABELLED "
            f"diagnostic, resp_kind='kappa' with paper_facing=False."
        )
    rk = getattr(cfg_or_resp_kind, "resp_kind", cfg_or_resp_kind)
    if rk is None or rk is _UNSET:
        raise ProvenanceError(
            f"[resp_kind] {context}: resp_kind is unset/None. It is NOT defaulted to "
            f"'kappa' here (that silent default is the defect this guard removes) -- "
            f"state the kernel explicitly. Note {KAPPA_WHY}."
        )
    rk = str(rk)
    if rk not in KNOWN_RESP_KINDS:
        raise ProvenanceError(
            f"[resp_kind] {context}: unknown resp_kind={rk!r}; expected one of "
            f"{KNOWN_RESP_KINDS!r}."
        )
    if paper_facing and rk != PAPER_FACING_RESP_KIND:
        raise ProvenanceError(
            f"[resp_kind] {context}: refusing to produce a PAPER-FACING artifact under "
            f"resp_kind={rk!r}. A paper-facing number MUST use "
            f"resp_kind={PAPER_FACING_RESP_KIND!r}, because {KAPPA_WHY}. Either set "
            f"cfg.resp_kind='forward' (see track_c_perz_band._set_forward_cfg) or stamp "
            f"this run as a labelled diagnostic (paper_facing=False)."
        )
    return rk


def kernel_metadata(cfg_or_resp_kind=_UNSET, *, context, paper_facing, extra_note=None):
    """Build the kernel self-declaration block for an artifact's ``metadata``.

    Every artifact this project stamps must SELF-DECLARE which response kernel produced
    it, so a reader never has to infer it from the routine.  Merge the result into the
    artifact metadata::

        metadata = dict(what=...,
                        **RK.kernel_metadata(cfg, context="sub-DLA headline",
                                             paper_facing=True))

    Resolution is delegated to :func:`resolve_resp_kind`, so this is ALSO the fail-closed
    WRITE-side guard: it raises before a single byte is written.
    """
    rk = resolve_resp_kind(cfg_or_resp_kind, context=context, paper_facing=paper_facing)
    if rk == RESP_KIND_FORWARD:
        note = ("FORWARD-response kernel (the measured forward response "
                "p(x_hat | N_true, SNR, z); Track-C 'right object').")
    else:
        note = ("GP-POSTERIOR ('kappa') kernel -- LABELLED DIAGNOSTIC, NOT PAPER-FACING. "
                + KAPPA_WHY[0].upper() + KAPPA_WHY[1:] + ".")
    if extra_note:
        note = f"{note} {extra_note}"
    return dict(resp_kind=rk, paper_facing=bool(paper_facing), kernel_note=note)


def assert_artifact_kernel(metadata, *, context="artifact", require_paper_facing=None):
    """FAIL-CLOSED READ side: an artifact must self-declare a legitimate kernel.

    Rejects
      (a) an artifact with no ``metadata['resp_kind']`` -- silence is not consent; most
          unstamped artifacts in this repo are kappa (e.g. the retired
          ``subdla_mock_validation.json``, whose metadata carries no ``resp_kind`` at all);
      (b) an unknown kernel;
      (c) ``paper_facing=True`` declared on anything other than ``'forward'``.

    ``require_paper_facing=True`` additionally demands the artifact declare itself
    paper-facing -- use it where a consumer is about to put the number in the paper.

    Returns the validated resp_kind.
    """
    md = metadata or {}
    if "resp_kind" not in md or md.get("resp_kind") in (None, ""):
        raise ProvenanceError(
            f"[resp_kind] {context}: artifact does not self-declare "
            f"metadata['resp_kind']. An unstamped artifact is REJECTED rather than "
            f"assumed forward -- most unstamped artifacts in this repo are kappa (e.g. "
            f"the retired subdla_mock_validation.json). Re-stamp it with "
            f"resp_kind.kernel_metadata(...)."
        )
    declared_pf = bool(md.get("paper_facing", False))
    rk = resolve_resp_kind(md.get("resp_kind"), context=context,
                           paper_facing=declared_pf)
    if require_paper_facing and not declared_pf:
        raise ProvenanceError(
            f"[resp_kind] {context}: artifact declares paper_facing="
            f"{md.get('paper_facing')!r} but the consumer requires a PAPER-FACING "
            f"artifact. Refusing to promote a diagnostic (resp_kind={rk!r}) into a "
            f"paper-facing number."
        )
    if require_paper_facing and rk != PAPER_FACING_RESP_KIND:
        raise ProvenanceError(
            f"[resp_kind] {context}: paper-facing consumer got resp_kind={rk!r}; "
            f"{KAPPA_WHY}."
        )
    return rk
