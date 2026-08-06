# REVIEW-ONLY (Phase A) — does not alter production behavior.
"""r04 — Post-repair closure overshoot, re-measured at the reviewed tip
(9d73365) on freshly extracted ADOPTED-config packs (bw 0.2, pad 19.0,
molly172), and decomposed against the r03b physical forest-FP supply.

Quantities per mock (all on dX>0 cells):
    ratio_full   = sum(mu)/sum(obs) over the full n-hat grid  (selftest point:
                   theta at the pack truth, psi=0, t=0, lam=fp_counts/ell)
    ratio_win    = same over the reporting window [19.7, 21.6)
    mu_fp_tot    = the folded FP total (repaired convention)
    required_fp  = obs_tot - mu_sig_tot  (what the FP term would have to
                   supply for exact total closure, holding the signal fold)
    S_r03b       = chance-corrected empirical forest-FP supply (transport-free
                   census, r03b)
The overshoot attribution test: if mu_fp_tot ~ S on the twin (2lpt0) but
mu_fp_tot >> S on the cross mocks, the residual is the loa-0 -> mock
TRANSPORT, not the ell_eff normalization.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

from CDDF_analysis.hbi_mcmc.pack import load_pack  # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PACK_DIR = ("/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
            "b10b5e23-575d-487e-811d-479f51611f63/scratchpad/r04_packs")
WIN_LO, WIN_HI = 19.7, 21.6

S_R03B = {"2lpt0": 14738.821017020087, "london0": 10167.737758670737,
          "saclay0": 11251.175418088838}

out = {}
for mock in ("2lpt0", "london0", "saclay0"):
    path = os.path.join(PACK_DIR, f"modelA_pack_{mock}_bw0p2_pad19p0_molly172.npz")
    if not os.path.exists(path):
        out[mock] = {"missing": True}
        continue
    pack = load_pack(path)
    res = FS.selftest(pack, resp_clamp="both")
    obs = np.asarray(pack.counts, float)
    m3 = np.broadcast_to((np.asarray(pack.dX, float) > 0)[None, :, :],
                         res["mu"].shape)
    mu = np.where(m3, res["mu"], 0.0)
    mu_sig = np.where(m3, res["mu_sig"], 0.0)
    mu_fp = np.where(m3, res["mu_fp"], 0.0)
    obs = np.where(m3, obs, 0.0)

    nhat_lo = np.asarray(pack.nhat_edges, float)[:-1]
    win = (nhat_lo >= WIN_LO - 1e-9) & (nhat_lo < WIN_HI - 1e-9)

    obs_tot, mu_tot = float(obs.sum()), float(mu.sum())
    required_fp = obs_tot - float(mu_sig.sum())
    rec = dict(
        ratio_full=mu_tot / obs_tot,
        ratio_win=float(mu[win].sum() / obs[win].sum()),
        obs_tot=obs_tot, mu_tot=mu_tot,
        mu_sig_tot=float(mu_sig.sum()), mu_fp_tot=float(mu_fp.sum()),
        required_fp_for_total_closure=required_fp,
        mu_fp_over_required=float(mu_fp.sum()) / required_fp,
        S_forest_fp_r03b=S_R03B[mock],
        mu_fp_over_S=float(mu_fp.sum()) / S_R03B[mock],
        required_over_S=required_fp / S_R03B[mock],
        overshoot_counts=mu_tot - obs_tot,
        fp_excess_vs_S_counts=float(mu_fp.sum()) - S_R03B[mock],
    )
    out[mock] = rec
    print(json.dumps({mock: rec}, indent=1, default=float), flush=True)

with open(os.path.join(HERE, "r04_out.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote r04_out.json")
