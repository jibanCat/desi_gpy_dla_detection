"""REVIEW-ONLY (Phase A) — build and cache the PRODUCTION fold operator.

This is the MODEL DEFINITION (the estimand under review), built by probing the
committed production fold — reusing it is explicitly allowed and required so
the review tests the same estimand. Everything downstream (fitting, truth
injection, calibration) is a fresh, independent implementation.

Run once with the gpdla-hbi python; caches A[c,k,s,b] and truth_f[b,k] as .npy
so the calibration driver never has to import JAX (fork-safety for the
multiprocessing bootstrap).
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")

PACK = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
        "PRESERVED_2026-07-28_small_artifacts/modelA_packs/modelA_pack_2lpt0_v11.npz")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def main():
    os.makedirs(CACHE, exist_ok=True)
    from CDDF_analysis.hbi_mcmc.pack import load_pack, coarsen_basis
    from CDDF_analysis.hbi_mcmc.forward import build_consts
    from CDDF_analysis.hbi_mcmc import e4_probe as E4

    t0 = time.time()
    pk = coarsen_basis(load_pack(PACK), 0.1, pad_floor=19.0)
    consts = build_consts(pk, resp_clamp="both")
    A = np.asarray(E4.build_fold_operator(pk, resp_clamp="both", consts=consts),
                   float)                                   # (C, Kf, S, B)
    lin = float(E4.check_linearity(pk, A))
    ftru = np.asarray(E4.truth_f(pk), float)                # (B, Kf), pad rows 0
    np.save(os.path.join(CACHE, "A_2lpt0_both.npy"), A)
    np.save(os.path.join(CACHE, "truthf_2lpt0.npy"), ftru)
    meta = dict(pack=PACK, resp_clamp="both", basis_width=0.1, pad_floor=19.0,
                A_shape=list(A.shape), linearity_max_rel_err=lin,
                A_sum=float(A.sum()), truthf_sum=float(ftru.sum()),
                build_seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(CACHE, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
