#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-loud loader for the atomic P1 (C, K) artifact (`p1_natpair_ck/v1`).

Guards (rulings §25: the production code must be UNABLE to double-apply
or omit completeness, mix estimands, silently renormalize the miss
state, or combine incompatible versions):

  * estimand ID + version must match the caller's expectation exactly;
  * C and K live in ONE atomic file — there is no code path that loads
    them separately, so version skew is structurally impossible;
  * internal identity re-verified AT LOAD: per-cell kernel counts
    (K_id_grid n) == C_molly_n_det on every ≥19.5 column, integers;
  * miss closure re-verified AT LOAD: det + subfloor + lowP + flag +
    unmatched == tot per cell, and no negative class anywhere;
  * normalization: 0 ≤ det ≤ tot in every cell;
  * no NaN kernel mean in any populated, unflagged reporting cell;
  * arrays are returned READ-ONLY; there is no renormalization helper
    and none may be added to this module.

Any violation raises P1CKGuardError. Nothing is auto-corrected.
"""
import json

import numpy as np


class P1CKGuardError(RuntimeError):
    pass


def load_p1_ck(path, expect_estimand_id="p1_natpair_ck/v1",
               expect_version=1):
    z = np.load(path, allow_pickle=False)
    need = ["estimand_id", "version", "provenance_json",
            "C_molly_n_det", "C_molly_n_tot", "C_snr_edges", "C_nhi_edges",
            "C_live_row", "K_id_grid", "K_id_j195", "miss_subfloor",
            "miss_lowP", "miss_flag", "miss_unmatched", "K_rep",
            "K_rep_n_edges", "K_marginal", "K_sparse_flag"]
    missing = [k for k in need if k not in z]
    if missing:
        raise P1CKGuardError(f"artifact missing fields: {missing}")
    eid = str(z["estimand_id"])
    ver = int(np.asarray(z["version"]).ravel()[0])
    if eid != expect_estimand_id or ver != expect_version:
        raise P1CKGuardError(
            f"estimand mismatch: artifact ({eid!r}, v{ver}) != expected "
            f"({expect_estimand_id!r}, v{expect_version})")

    det = np.asarray(z["C_molly_n_det"], np.int64)
    tot = np.asarray(z["C_molly_n_tot"], np.int64)
    if det.shape != tot.shape:
        raise P1CKGuardError("C shape mismatch det vs tot")
    if np.any(det < 0) or np.any(tot < 0) or np.any(det > tot):
        raise P1CKGuardError("normalization violated: need 0 <= det <= tot")

    j195 = int(np.asarray(z["K_id_j195"]).ravel()[0])
    kid_n = np.asarray(z["K_id_grid"])[:, :, 0]
    kn = np.nan_to_num(kid_n[:, j195:], nan=-1).astype(np.int64)
    if not np.array_equal(kn, det[:, j195:]):
        bad = int(np.sum(kn != det[:, j195:]))
        raise P1CKGuardError(
            f"IDENTITY violated at load: kernel n != C numerator in {bad} "
            f"cells — refusing to serve an incoherent (C, K) pair")

    closure = (det + np.asarray(z["miss_subfloor"], np.int64)
               + np.asarray(z["miss_lowP"], np.int64)
               + np.asarray(z["miss_flag"], np.int64)
               + np.asarray(z["miss_unmatched"], np.int64))
    if not np.array_equal(closure[:, j195:], tot[:, j195:]):
        raise P1CKGuardError("miss closure violated at load")
    for k in ("miss_subfloor", "miss_lowP", "miss_flag", "miss_unmatched"):
        if np.any(np.asarray(z[k]) < 0):
            raise P1CKGuardError(f"negative miss class {k}")

    rep = np.asarray(z["K_rep"], float)
    sparse = np.asarray(z["K_sparse_flag"], bool)
    populated = rep[:, :, :, 0] > 0
    bad_mean = populated & ~sparse & ~np.isfinite(rep[:, :, :, 1])
    if np.any(bad_mean):
        raise P1CKGuardError("NaN kernel mean in populated unflagged cell")

    out = {k: np.asarray(z[k]) for k in need if k != "provenance_json"}
    out["provenance"] = json.loads(str(z["provenance_json"]))
    for v in out.values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return out
