#!/usr/bin/env python
"""upgrade_packs_v2.py — stamp the adopted response + count-conservation
contract onto the adopted Model A packs (PI ruling 2026-08-17 item 6).

Per pack: copy every v1 array byte-identically; add the schema-v1.2 stamp
group — TP-convention ID, contract ID, adopted response version + surfaces
+ fit range, the deployed kernel's in-grid fraction phi_ref, the 96-draw
sightline-bootstrap carrier ensemble — and populate ``resp_fitcov_diag``
with the frozen kernel's bootstrap order-0 variances (retiring the
0.02/0.10 placeholder on the frozen path too, per the checkpoint-3 ruling).

Verification per pack (fail-loud): every v1 key byte-identical; reload
passes the v1.2 all-or-none validation; the fail-closed adopted fold runs
and restores the deployed level to 1e-6.

Env: gpdla-hbi. Usage:
  python -m CDDF_analysis.hbi_mcmc.upgrade_packs_v2 [--outdir DIR]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f
from CDDF_analysis.hbi_mcmc.count_conserving_fold import (
    phi_from_surfaces, cc_fold_adopted, cc_fold_cmarginal)

SRCDIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
          "adopted_packs_20260816")
DEF_OUT = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
           "adopted_packs_v2_20260817")
ADOPTED = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
           "track_c/stage0/adopted_response_v1.npz")
KFE = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
       "track_c/stage0/kernel_fit_ensemble_v1.npz")
TAG = "bw0p2_pad19p0_molly172"
MOCKS = ("2lpt0", "london0", "saclay0")

TP_ID = "tp_natpair_tilthost_op/v1"
CONTRACT_ID = "ckfp_lown_contract/v1"
TP_DEF = ("TP = natural-pair truth match, host_col=NHI_TILT_HOST, op-cut "
          "S2N_RED>2 & P_DLA>0.99 & good_mask, xhat>=19.5; C = deployed "
          "molly two-chain splice (calibrated jointly with the deployed "
          "kernel; C_op = C_molly*phi_ref); FP = loa-0 HCD-free twin "
          "(purity_mixture, (1-eta_band)); counting domain = the observed "
          "grid [19.5, 22.4]")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=DEF_OUT)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ad = np.load(ADOPTED, allow_pickle=True)
    kfe = np.load(KFE, allow_pickle=True)
    fitcov = np.stack([kfe["mu_coef"][..., 0].std(axis=0, ddof=1) ** 2,
                       kfe["sig_coef"][..., 0].std(axis=0, ddof=1) ** 2])
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))).decode().strip()
    except Exception:
        commit = "unknown"

    for mock in MOCKS:
        src = os.path.join(SRCDIR, f"modelA_pack_{mock}_{TAG}.npz")
        dst = os.path.join(a.outdir, f"modelA_pack_{mock}_{TAG}_v2.npz")
        pk = load_pack(src)
        phi_ref = phi_from_surfaces(pk)
        raw = dict(np.load(src, allow_pickle=False))
        assert "resp_fitcov_diag" not in raw
        raw["resp_fitcov_diag"] = fitcov
        raw["tp_convention_id"] = np.array(TP_ID)
        raw["contract_id"] = np.array(CONTRACT_ID)
        raw["adopted_resp_version"] = np.array("adopted_response/v1")
        raw["adopted_resp_mu_coef"] = np.asarray(ad["mu_coef"], float)
        raw["adopted_resp_sig_coef"] = np.asarray(ad["sig_coef"], float)
        raw["adopted_resp_skew_coef"] = np.asarray(ad["skew_coef"], float)
        raw["adopted_resp_fit_range"] = np.asarray(ad["fit_rng"], float)
        raw["adopted_phi_ref"] = phi_ref
        raw["adopted_carrier_mu"] = np.asarray(ad["carrier_mu"], float)
        raw["adopted_carrier_sig"] = np.asarray(ad["carrier_sig"], float)
        raw["adopted_carrier_skew"] = np.asarray(ad["carrier_skew"], float)
        raw["adopted_carrier_shared3"] = np.asarray(ad["carrier_shared3"],
                                                    float)
        np.savez_compressed(dst, **raw)

        # ---- verification (fail-loud) ---------------------------------
        with np.load(src) as z1, np.load(dst) as z2:
            for k in z1.files:
                if k == "resp_fitcov_diag":
                    continue
                if not np.array_equal(z1[k], z2[k]):
                    raise SystemExit(f"{mock}: v1 key {k} NOT byte-identical")
        pk2 = load_pack(dst)                       # v1.2 validation
        theta = np.log(np.clip(np.asarray(truth_f(pk2), float),
                               1e-300, None))
        lam = np.asarray(pk2.fp_counts, float) / float(pk2.fp_ell_eff)
        obs = float(np.asarray(pk2.counts, float).sum())
        mu_dep, _ = cc_fold_cmarginal(pk2, theta, lam)
        mu_ad, _ = cc_fold_adopted(pk2, theta, lam)
        d_level = abs(mu_ad.sum() / obs - mu_dep.sum() / obs)
        if d_level > 1e-6:
            raise SystemExit(f"{mock}: adopted-CC level identity FAILED "
                             f"({d_level:.2e})")
        prov = dict(schema="modelA_pack_schema v1.2 (adopted-contract stamp)",
                    upgraded_from=src, src_sha256=_sha(src),
                    adopted_response=ADOPTED, adopted_sha256=_sha(ADOPTED),
                    kernel_fit_ensemble=KFE, kfe_sha256=_sha(KFE),
                    tp_convention_id=TP_ID, tp_definition=TP_DEF,
                    contract_id=CONTRACT_ID,
                    contract_doc="docs/JOINT_CKFP_LOWN_CONTRACT_PROPOSAL.md "
                                 "(ratified in principle 2026-08-17)",
                    resp_fitcov_diag_source=("frozen-kernel sightline "
                                             "bootstrap order-0 variances "
                                             "(kernel_fit_ensemble_v1)"),
                    code_commit=commit)
        prov_path = dst[:-4] + ".provenance.json"
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
        print(f"[v2] {mock}: wrote {os.path.basename(dst)} "
              f"(byte-identity OK; adopted-CC level identity {d_level:.1e}; "
              f"level {mu_ad.sum()/obs:.4f})", flush=True)
    print("[v2] done")


if __name__ == "__main__":
    main()
