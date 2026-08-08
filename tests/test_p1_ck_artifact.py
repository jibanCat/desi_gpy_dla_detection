# -*- coding: utf-8 -*-
"""Guard tests for the atomic P1 (C, K) artifact loader (fail-loud).

Synthetic-artifact tests run everywhere (no scratch access needed).
The integration test against the real artifact is opt-in:
RUN_P1_CK_SCRATCH=1 (GPFS read, matches the committed build hash).
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "injection"))

from p1_ck_loader import P1CKGuardError, load_p1_ck  # noqa: E402

EID = "p1_natpair_ck/v1"


def _mini(tmp_path, **override):
    """A minimal internally-consistent artifact (2 SNR rows x 3 N cols,
    identity region = cols >= 1)."""
    det = np.array([[3, 4, 5], [2, 6, 7]], np.int64)
    tot = np.array([[6, 5, 6], [4, 8, 9]], np.int64)
    j195 = 1
    kid = np.full((2, 3, 5), np.nan)
    kid[:, j195:, 0] = det[:, j195:]
    kid[:, j195:, 1] = 0.01
    fields = dict(
        estimand_id=np.array(EID), version=np.array([1]),
        provenance_json=np.array(json.dumps({"synthetic": True})),
        C_molly_n_det=det, C_molly_n_tot=tot,
        C_snr_edges=np.array([0.0, 2.0, np.inf]),
        C_nhi_edges=np.array([17.2, 19.5, 20.0, np.inf]),
        C_live_row=np.array([False, True]),
        K_id_grid=kid, K_id_j195=np.array([j195]),
        miss_subfloor=np.zeros_like(det),
        miss_lowP=(tot - det) // 2,
        miss_flag=np.zeros_like(det),
        miss_unmatched=(tot - det) - (tot - det) // 2,
        K_rep=np.concatenate(
            [np.full((2, 3, 3, 1), 30.0), np.zeros((2, 3, 3, 4))], axis=3),
        K_rep_n_edges=np.array([19.5, 19.7, 19.9]),
        K_rep_z_edges=np.array([2.56, 2.96]),
        K_rep_s_edges=np.array([3.5, 6.5]),
        K_marginal=np.zeros((2, 5)), K_sparse_flag=np.zeros((2, 3, 3), bool),
        composition=np.zeros((2, 4)), n_out_support=np.array([0]))
    fields.update(override)
    p = str(tmp_path / "mini_ck.npz")
    np.savez(p, **fields)
    return p


def test_valid_artifact_loads_and_is_readonly(tmp_path):
    art = load_p1_ck(_mini(tmp_path))
    assert art["provenance"]["synthetic"] is True
    with pytest.raises(ValueError):
        art["C_molly_n_det"][0, 0] = 99


def test_estimand_id_mismatch_fails_loud(tmp_path):
    p = _mini(tmp_path, estimand_id=np.array("someone_elses_kernel/v9"))
    with pytest.raises(P1CKGuardError, match="estimand mismatch"):
        load_p1_ck(p)


def test_version_mismatch_fails_loud(tmp_path):
    p = _mini(tmp_path, version=np.array([2]))
    with pytest.raises(P1CKGuardError, match="estimand mismatch"):
        load_p1_ck(p)


def test_tampered_kernel_count_breaks_identity(tmp_path):
    kid = np.full((2, 3, 5), np.nan)
    kid[:, 1:, 0] = [[5, 5], [6, 7]]     # true det is [[4,5],[6,7]] — off by one
    kid[:, 1:, 1] = 0.01
    p = _mini(tmp_path, K_id_grid=kid)
    with pytest.raises(P1CKGuardError, match="IDENTITY"):
        load_p1_ck(p)


def test_broken_miss_closure_fails(tmp_path):
    p = _mini(tmp_path, miss_unmatched=np.zeros((2, 3), np.int64))
    with pytest.raises(P1CKGuardError, match="miss closure"):
        load_p1_ck(p)


def test_negative_class_fails(tmp_path):
    det = np.array([[3, 4, 5], [2, 6, 7]], np.int64)
    tot = np.array([[6, 5, 6], [4, 8, 9]], np.int64)
    bad = tot - det
    bad[0, 1] += 1
    unm = np.zeros_like(det)
    unm[0, 1] = -1
    p = _mini(tmp_path, miss_lowP=bad, miss_unmatched=unm)
    with pytest.raises(P1CKGuardError):
        load_p1_ck(p)


def test_det_exceeding_tot_fails(tmp_path):
    det = np.array([[7, 4, 5], [2, 6, 7]], np.int64)   # 7 > tot 6
    p = _mini(tmp_path, C_molly_n_det=det)
    with pytest.raises(P1CKGuardError, match="normalization"):
        load_p1_ck(p)


def test_nan_mean_in_populated_unflagged_cell_fails(tmp_path):
    rep = np.concatenate(
        [np.full((2, 3, 3, 1), 30.0), np.zeros((2, 3, 3, 4))], axis=3)
    rep[0, 0, 0, 1] = np.nan
    p = _mini(tmp_path, K_rep=rep)
    with pytest.raises(P1CKGuardError, match="NaN kernel mean"):
        load_p1_ck(p)


def test_missing_field_fails(tmp_path):
    import numpy as _np
    p = str(tmp_path / "broken.npz")
    _np.savez(p, estimand_id=_np.array(EID))
    with pytest.raises(P1CKGuardError, match="missing fields"):
        load_p1_ck(p)


@pytest.mark.skipif(os.environ.get("RUN_P1_CK_SCRATCH") != "1",
                    reason="scratch integration opt-in (RUN_P1_CK_SCRATCH=1)")
def test_real_artifact_identity_and_hash():
    build = json.load(open(os.path.join(
        _REPO, "diagnostics_phaseC/p1_completeness/p1_ck_build.json")))
    path = build["artifact"]
    art = load_p1_ck(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    assert h.hexdigest() == build["artifact_sha256"]
    assert art["provenance"]["schema"] == EID
