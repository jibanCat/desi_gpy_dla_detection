#!/usr/bin/env python3
"""write_selection_contract_sidecar.py — R-016 (PI hygiene approval 2026-08-26).

The frozen pack carries its contract IDENTIFIERS (contract_id, tp_convention_id,
adopted_resp_version) but no machine-readable numeric selection threshold, so the
paper-side completeness figures hard-coded the SNR floor. This writes ONE small JSON
BESIDE the pack (never inside it; the .npz is opened read-only), generated from the
sources of record rather than retyped:

  * docs/CANONICAL_PURITY_COMPLETENESS_CONTRACT.json  /sample_contract/<sample>
  * CDDF_analysis/hbi/cddf_catalog_hbi.HBIConfig (`CatalogCuts` below) defaults (snr_min, p_dla_min, z_qso window)
  * CDDF_analysis/hbi_mcmc/model_a._THRESHOLDS and the differential mask
  * the pack's own contract_id and sha256 (binds the sidecar to the pack; consumers
    fail closed if either disagrees)

    python3 CDDF_analysis/hbi_mcmc/write_selection_contract_sidecar.py <pack.npz> [--sample P1_PRIMARY_LYA]
"""
import argparse, hashlib, json, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--sample", default="P1_PRIMARY_LYA")
    a = ap.parse_args()
    import numpy as np
    from CDDF_analysis.hbi.cddf_catalog_hbi import HBIConfig as CatalogCuts
    from CDDF_analysis.hbi_mcmc import model_a
    pack = pathlib.Path(a.pack)
    with np.load(pack, allow_pickle=False) as d:
        contract_id = str(d["contract_id"])
        tp_id = str(d["tp_convention_id"])
        resp_id = str(d["adopted_resp_version"])
    contract = json.loads((REPO / "docs/CANONICAL_PURITY_COMPLETENESS_CONTRACT.json").read_text())
    # The collar OF RECORD is the one the frozen real pack was built with (PI ruling
    # checkpoint 10.8, observable-only estimator, c = 3300 km/s), read from the pack
    # builder -- NOT the 3000 km/s that the contract JSON's v1.1 text still carries
    # (documentation lag; recorded in the JSON's collar_kms_of_record field).
    from CDDF_analysis.hbi_mcmc import extract_pack_real
    collar_kms = float(extract_pack_real.COLLAR_KMS)
    sc = contract["sample_contract"][a.sample]
    cuts = CatalogCuts.__dataclass_fields__
    snr_min = cuts["snr_min"].default
    p_min = cuts["p_dla_min"].default
    # the contract text must agree with the code defaults; refuse to write otherwise
    assert re.search(r">\s*%g\b" % snr_min, sc["snr_cut"]), (sc["snr_cut"], snr_min)
    assert re.search(r">\s*%g\b" % p_min, sc["p_dla_cut"]), (sc["p_dla_cut"], p_min)
    ct_collar = contract.get("collar_kms_of_record")
    assert ct_collar is None or float(ct_collar) == collar_kms, (ct_collar, collar_kms)
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        head = "UNKNOWN"
    out = {
        "role": "selection-contract sidecar (R-016): numeric thresholds of the frozen pack's contract, generated from the sources of record; additive, the pack is untouched",
        # NOTE: the pack's contract_id names the joint C/K/FP low-N CALIBRATION contract, not
        # the sample selection; the selection contract gets its own identifier here.
        "selection_contract_id": "p1_primary_lya_selection/v1.2 (v1.1 thresholds; collar 3300 km/s per PI ruling ckpt-10.8)",
        "pack_contract_id": contract_id, "tp_convention_id": tp_id, "adopted_resp_version": resp_id,
        "pack": str(pack), "pack_sha256": sha256(pack),
        "sample": a.sample,
        "source_contract": "docs/CANONICAL_PURITY_COMPLETENESS_CONTRACT.json /sample_contract/" + a.sample,
        "source_contract_version": contract.get("version"),
        "snr_field": sc["snr_field"], "snr_min": float(snr_min), "snr_strict": True,
        "p_dla_field": sc["p_dla_field"], "p_dla_min": float(p_min), "p_dla_strict": True,
        "quality": sc["quality"], "bal_policy": sc["bal_policy"],
        "lambda_rf_window": "lya_only: lam_rf(Z_DLA) in [1025, 1216] A, symmetric collar, 3600 A observed floor; strict "
                            "(extract_pack_real.contract_row_mask)",
        "collar_kms": collar_kms,
        "collar_note": ("collar of record = %g km/s (frozen real pack, extract_pack_real.COLLAR_KMS; PI ruling ckpt-10.8); "
                        "the v1.1 contract text's 3000 km/s is SUPERSEDED for the HBI real-data plane" % collar_kms),
        "fp_block_window_note": ("the pack's FP calibration block (fp_counts) is built under a hard lam_rf >= 1025 A blue edge "
                                 "with NO collar (extract_pack.build_fp_block); 89 events on support vs 87 under the "
                                 "c=3300 collar -- recorded 2026-08-26 (R-015), disclosure pending PI"),
        "z_qso_min": cuts["z_qso_min"].default, "z_qso_max": cuts["z_qso_max"].default,
        "reporting_thresholds": list(model_a._THRESHOLDS),
        "differential_mask": [model_a._MASK_LO, model_a._MASK_HI],
        "generator": "CDDF_analysis/hbi_mcmc/write_selection_contract_sidecar.py", "code_commit": head,
    }
    dst = pack.with_suffix(".selection_contract.json")
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dst)


if __name__ == "__main__":
    main()
