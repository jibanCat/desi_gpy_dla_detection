#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-B bounded diagnosis (exploratory) — H7: below-support promotion.

PRE-STATED PREDICTED SIGNATURE (spec s7.7), written BEFORE folding: extending
the truth fold's pad floor 19.0 -> 17.2 with the EXISTING molly172 sub-floor
completeness promotes more sub-floor truth into the low observed bins, so
G1's residual should CHANGE (become more negative: more promoted mu).  If G3
is untouched, promotion cannot explain the leading (G3) violation.

The pad-17.2 pack was produced by the COMMITTED extractor at the Phase-B tip
(extract_pack.py --mocks 2lpt0 --basis-pad-floor 17.2
--completeness-below-floor molly172 --basis-width 0.2 --analysis-window
lya_only --tag _pad17p2_diag; log: h07_extract.log).  No new model freedom.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/mfho/wt_repair_phaseB")

from CDDF_analysis.hbi_mcmc.pack import load_pack                     # noqa: E402
from CDDF_analysis.hbi_mcmc import forward_selftest as FS             # noqa: E402
from CDDF_analysis.hbi_mcmc.gate_covariance import (                  # noqa: E402
    group_aggregator, PRIMARY_GROUP_EDGES)

OUT = "/home/mfho/wt_repair_phaseB/diagnostics_phaseB/twin_nhat"
P17 = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/phaseB_packs/"
       "modelA_pack_2lpt0_pad17p2_diag.npz")
WIN = (19.7, 21.6)

pk = load_pack(P17)
st = FS.selftest(pk, resp_clamp="both")
live = (np.asarray(pk.dX, float) > 0)[None, :, :]
mu_c = np.where(live, st["mu"], 0.0).sum(axis=(1, 2))
obs_c = np.where(live, st["counts"], 0.0).sum(axis=(1, 2))
ne = np.asarray(pk.nhat_edges, float)
win = (ne[:-1] >= WIN[0] - 1e-9) & (ne[1:] <= WIN[1] + 1e-9)
A = group_aggregator(pk, PRIMARY_GROUP_EDGES)
d17 = A @ (obs_c - mu_c)
z17 = (obs_c - mu_c)[win] / np.sqrt(np.maximum(mu_c[win], 1e-12))

b19 = np.load(f"{OUT}/base_2lpt0_both.npz")
d19 = b19["d_grp"]
z19 = b19["z_c"][b19["win_mask"].astype(bool)]

res = {"label": "exploratory (prespecified discriminant, spec s7.7)",
       "pack": P17, "n_pad_bins_17p2": int(pk.n_pad_bins),
       "ntrue_floor": float(pk.ntrue_edges[0]),
       "group_residual_pad19p0": d19.tolist(),
       "group_residual_pad17p2": d17.tolist(),
       "delta_group_residual": (d17 - d19).tolist(),
       "per_bin_z_pad19p0": z19.tolist(),
       "per_bin_z_pad17p2": z17.tolist(),
       "chi2_dof_win_pad19p0": float(np.sum(z19 ** 2) / len(z19)),
       "chi2_dof_win_pad17p2": float(np.sum(z17 ** 2) / len(z17)),
       "window_mu_pad17p2": float(mu_c[win].sum()),
       "full_mu_pad17p2": float(mu_c.sum()),
       "full_obs": float(obs_c.sum())}
with open(f"{OUT}/h07_pad17p2_fold.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items()
                  if not k.startswith("per_bin")}, indent=1))
print("z pad19.0:", " ".join("%+.1f" % v for v in z19))
print("z pad17.2:", " ".join("%+.1f" % v for v in z17))
