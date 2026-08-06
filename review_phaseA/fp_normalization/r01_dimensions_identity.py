# REVIEW-ONLY (Phase A) — does not alter production behavior.
"""r01 — Independent dimensional derivation + numerical identity checks of the
FP normalization defect/repair (commit 7707c8e), from first principles.

THE COUNTING ARGUMENT (independent of the fold implementation)
--------------------------------------------------------------
Observed facts:
  * loa-0 is the HCD-free twin; every op-passing loa-0 detection is a forest FP.
  * n0[c,s]  = loa-0 FP detections binned on the pack support
               (N-hat in [19.5,22.4), z in [2.0,3.5), SNR>2, P_DLA>0.99, lya-only).
  * ns0      = searched loa-0 sightlines with SNR>2  [sightlines]
  * nsm      = mock-m searched sightlines with SNR>2 [sightlines]
Define the empirical per-sightline FP rate
      r_hat[c,s] = n0[c,s] / ns0        [FP counts per searched sightline]
Under the transport hypothesis (same forest-FP rate per searched sightline in
the mock as in loa-0), the expected FP count in the mock's data census is
      MU_FP[c,s] = r_hat[c,s] * nsm * (1 - eta)                    ... (*)
where (1 - eta) is the host-occlusion survival the loa-0 product itself defines
(build_loa0_fp_product.py:34-39): forest FPs can only be found in forest not
already occupied by a real HCD, and loa-0 has none.

THE MODEL'S PARAMETERIZATION
  calibration: fp_counts[c,s] ~ Poisson(ell * lam[c,s]),  ell = ns0^2/nsm
  fold:        mu_FP[c,k,s]   = w * ell * exp(t_K) * lam[c,s] * E[k,s],
               w = nsm/ns0,  sum_k E[k,s] = 1  per populated stratum.
At the point calibration lam_hat = n0/ell and t=0:
      sum_k mu_FP = w * ell * lam_hat = w * n0 = (nsm/ns0) * n0 = r_hat*nsm.
So the repaired fold reproduces the counting answer (*) EXACTLY ONCE — except
for the (1 - eta) factor, which appears NOWHERE in the Model A chain (checked
in r02): the pack carries no eta field and the fold has no eta term, while the
commit message equates the fold to an expression that contains (1 - eta_bar).

Units:  lam [FP counts per unit ell-exposure]; ell [deflated sightlines];
w [dimensionless ratio]; E [dimensionless z-allocation, sums to 1];
exp(t) [dimensionless transfer]; mu_FP [counts].  w*ell = ns0 [sightlines]
identically in exact arithmetic: (nsm/ns0)*(ns0^2/nsm) = ns0 for ANY nsm.
NOTE this means the "identity fp_w*fp_ell_eff == n_sl_loa0" is an algebraic
tautology of the extractor's two definitions — verifying it numerically checks
float round-trip and pack integrity, not the physics.

WHY ell = ns0^2/nsm AND NOT ns0: the likelihood is invariant to the choice
(any positive constant c in "n0 ~ Poisson(c*lam)" folded back as "w*c*lam"
gives the same mu_FP posterior; see r03).  The ns0^2/nsm convention makes the
GAMMA-POSTERIOR VARIANCE of the folded total equal the frequentist variance of
the production-extrapolated count (Var[w*n0] = w^2*n0), i.e. it is the
'production-extrapolation variance' bookkeeping inherited from the Q2 Loa0FP
product where lam itself (not the fold) carried the posterior.

This script verifies, on every available pack (v11 and non-v11) and on the
coarsened basis the review brief specifies:
  1. fp_w * fp_ell_eff == 2255.0 exactly (and == the product's n_sl_loa0);
  2. sum_k fp_E_alloc[k,s] == 1 for populated strata, 0 for empty;
  3. my counting-argument total (nsm/ns0)*sum(n0)  ==  the repaired
     forward.fold_mu_fp total at t=0, lam=n0/ell  (to float precision);
  4. the pre-repair fold is low by exactly ell (the defect);
  5. the loa-0 product's eta values and the size of the (1-eta) omission;
  6. the Poisson relative uncertainty of the calibrated FP total (1/sqrt(n0)).
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from CDDF_analysis.hbi_mcmc.pack import load_pack, coarsen_basis  # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts, fold_mu_fp  # noqa: E402

PACK_DIR = ("/nfs/turbo/lsa-cavestru/mfho/DESI/gpdla_catalogs/"
            "PRESERVED_2026-07-28_small_artifacts/modelA_packs")
PRODUCTS = {
    "full_window": "/scratch/cavestru_root/cavestru0/mfho/"
                   "gl_loa0_fp_v1_20260615/outputs/loa0_fp_product.npz",
    "lyaonly1025": "/scratch/cavestru_root/cavestru0/mfho/"
                   "gl_loa0_fp_v1_20260615/outputs/loa0_fp_product_lyaonly1025.npz",
}

out = {"packs": {}, "products": {}}

# ---- loa-0 products: eta, n_sl, totals --------------------------------------
for tag, path in PRODUCTS.items():
    if not os.path.exists(path):
        out["products"][tag] = {"missing": True}
        continue
    d = np.load(path, allow_pickle=True)
    logN_lo = np.asarray(d["logN_lo"], float)
    fine = np.asarray(d["n_fp_fine"], float)
    b195 = int(np.searchsorted(np.round(logN_lo, 3), 19.5))
    out["products"][tag] = dict(
        n_sl_loa0=float(d["n_sl_loa0"]), n_sl_prod=float(d["n_sl_prod"]),
        n_fp_total=float(d["n_fp_total"]), ell_eff_product=float(d["ell_eff"]),
        eta_lls=float(d["eta_lls"]), eta_subdla=float(d["eta_subdla"]),
        eta_dla=float(d["eta_dla"]),
        n_fp_fine_ge195_zwin=float(fine[b195:, :].sum()),
        n_fp_fine_all=float(fine.sum()),
    )

# ---- packs ------------------------------------------------------------------
PACKS = {
    "2lpt0_v11": f"{PACK_DIR}/modelA_pack_2lpt0_v11.npz",
    "london0_v11": f"{PACK_DIR}/modelA_pack_london0_v11.npz",
    "saclay0_v11": f"{PACK_DIR}/modelA_pack_saclay0_v11.npz",
    "2lpt0_nonv11": f"{PACK_DIR}/modelA_pack_2lpt0.npz",
    "london0_nonv11": f"{PACK_DIR}/modelA_pack_london0.npz",
    "saclay0_nonv11": f"{PACK_DIR}/modelA_pack_saclay0.npz",  # expected ABSENT
}

for name, path in PACKS.items():
    if not os.path.exists(path):
        out["packs"][name] = {"missing": True}
        continue
    p = load_pack(path)
    variants = {"raw": p}
    if name.endswith("_v11"):
        # the review brief's working basis: 0.1-dex, pad floor 19.0
        variants["coarsen_0.1_pad19.0"] = coarsen_basis(p, 0.1, pad_floor=19.0)
    rec = {}
    for vtag, pk in variants.items():
        w = float(pk.fp_w_sightline_ratio)
        ell = float(pk.fp_ell_eff)
        n0 = np.asarray(pk.fp_counts, float)
        E = np.asarray(pk.fp_E_alloc, float)
        ns0 = w * ell                      # implied loa-0 sightlines
        nsm = w * ns0                      # implied mock sightlines
        # (1) identity
        ident = dict(fp_w=w, fp_ell_eff=ell, w_times_ell=w * ell,
                     equals_2255=bool(w * ell == 2255.0),
                     implied_nsm=nsm)
        # (2) E allocation
        colsum = E.sum(axis=0)
        ident["E_colsums_populated_all_one"] = bool(
            np.allclose(colsum[colsum > 0], 1.0, rtol=0, atol=1e-12))
        ident["E_n_empty_strata"] = int((colsum == 0).sum())
        # (3) my counting total vs the repaired fold
        my_total = (nsm / ns0) * n0.sum()          # counting argument, no fold
        lam_hat = n0 / ell
        # resp_clamp is irrelevant to the FP term; legacy (non-v11) packs lack
        # resp_N_fit_range, so admit them in the documented diagnostic mode.
        has_rng = pk.resp_N_fit_range is not None
        consts = build_consts(pk, resp_clamp="both" if has_rng else "off",
                              allow_unclamped_response=not has_rng)
        mufp = np.asarray(fold_mu_fp(jnp.zeros(consts.n_kk),
                                     jnp.asarray(lam_hat), consts))
        fold_total = float(mufp.sum())
        # (4) the defective (pre-repair) expression
        defect_total = float((w * lam_hat[:, None, :]
                              * E[None, :, :]).sum())
        rec[vtag] = dict(
            **ident,
            n0_total=float(n0.sum()),
            counting_mu_fp_total=float(my_total),
            fold_mu_fp_total=fold_total,
            agree_rel=float(abs(fold_total - my_total) / my_total),
            defective_total=defect_total,
            defect_ratio=float(fold_total / defect_total),
            defect_ratio_equals_ell=bool(
                abs(fold_total / defect_total - ell) < 1e-9),
            poisson_rel_sd_of_total=float(1.0 / np.sqrt(n0.sum() + 0.5)),
            t_sigma=np.asarray(pk.t_sigma, float).tolist(),
            n0_by_nhat_nonzero={
                f"{pk.nhat_edges[c]:.1f}": int(n0[c].sum())
                for c in range(n0.shape[0]) if n0[c].sum() > 0},
        )
    out["packs"][name] = rec

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "r01_out.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
