"""Tests for the fail-closed forward-response kernel guard.

``CDDF_analysis.unblind.provenance.assert_forward_kernel`` enforces the Track-C
invariant that a HEADLINE catalog-HBI run must use the forward-response kernel
(``resp_kind='forward'``), never the GP-posterior ('kappa') kernel that
``HBIConfig`` defaults to.  The guard must FAIL LOUDLY (raise), not warn.

These tests are pure (no GP inference, no data, no git) — they exercise the guard
contract directly:

  * accepts ``resp_kind='forward'`` (string and via a cfg-like object),
  * REJECTS ``resp_kind='kappa'``,
  * REJECTS a missing / ``None`` resp_kind (HBIConfig default is 'kappa' -> fail-closed),
  * REJECTS an unknown resp_kind,
  * with ``require_kernel_model=True``, rejects forward-without-a-kernel-model.
"""
import pytest

from CDDF_analysis.unblind.provenance import assert_forward_kernel, ProvenanceError


class _Cfg:
    """Minimal HBIConfig stand-in: only the attributes the guard reads."""
    def __init__(self, resp_kind=None, kernel_forward_model=None):
        if resp_kind is not None:
            self.resp_kind = resp_kind
        else:
            # mimic HBIConfig's real default so a "forgot to set forward" cfg is tested
            self.resp_kind = "kappa"
        self.kernel_forward_model = kernel_forward_model


def test_accepts_forward_string():
    assert assert_forward_kernel("forward", context="unit") == "forward"


def test_accepts_forward_cfg():
    cfg = _Cfg(resp_kind="forward", kernel_forward_model="/path/to/frm.npz")
    assert assert_forward_kernel(cfg, context="unit") == "forward"


def test_rejects_kappa_string():
    with pytest.raises(ProvenanceError, match="forward-kernel guard"):
        assert_forward_kernel("kappa", context="unit")


def test_rejects_kappa_cfg():
    cfg = _Cfg(resp_kind="kappa")
    with pytest.raises(ProvenanceError, match="WRONG OBJECT"):
        assert_forward_kernel(cfg, context="unit")


def test_rejects_missing_resp_kind_string():
    # a bare None (no resp_kind at all) is treated as the HBIConfig default -> fail-closed
    with pytest.raises(ProvenanceError):
        assert_forward_kernel(None, context="unit")


def test_rejects_default_cfg():
    # a cfg that never had resp_kind flipped to forward defaults to 'kappa' -> reject
    cfg = _Cfg()
    with pytest.raises(ProvenanceError):
        assert_forward_kernel(cfg, context="unit")


def test_rejects_unknown_resp_kind():
    with pytest.raises(ProvenanceError):
        assert_forward_kernel("posterior", context="unit")


def test_require_kernel_model_rejects_forward_without_model():
    cfg = _Cfg(resp_kind="forward", kernel_forward_model=None)
    with pytest.raises(ProvenanceError, match="kernel_forward_model is unset"):
        assert_forward_kernel(cfg, context="unit", require_kernel_model=True)


def test_require_kernel_model_accepts_forward_with_model():
    cfg = _Cfg(resp_kind="forward", kernel_forward_model="/path/to/frm.npz")
    assert assert_forward_kernel(cfg, context="unit", require_kernel_model=True) == "forward"
