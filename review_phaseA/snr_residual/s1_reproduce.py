# REVIEW-ONLY (Phase A)
"""s1 — reproduce the committed PI-checkpoint §10 numbers at commit 9d73365.

Target claims (docs/PI_CHECKPOINT_2026-08-05_kernel_fp_identifiability.md §9-10):
  by_snr chi2/dof = 36.61 / 62.91 / 54.61   (2lpt0 / london0 / saclay0)
  by_snr z: SNR [2,3) +8.14/+9.67/+10.93 ; SNR >= 5 in -4..-9.8
  window chi2/dof = 22.22 / 28.39 / 25.77
  full mu/obs     = 1.0016 / 1.0501 / 1.0281

Config: adopted config (bw=0.2, pad_floor=19.0, molly172, window=lya_only),
packs rebuilt at 9d73365 in this session's scratchpad (r04_packs).
Fold: production selftest (repaired fold_mu / fold_mu_fp), resp_clamp="both".
"""
import sys, os, json
import numpy as np

sys.path.insert(0, "/home/mfho/wt_review_phaseA")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax; jax.config.update("jax_enable_x64", True)

from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc import forward_selftest as FS
from CDDF_analysis.hbi_mcmc import reporting as REP

PACKDIR = ("/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
           "b10b5e23-575d-487e-811d-479f51611f63/scratchpad/r04_packs")
PACKS = {m: os.path.join(PACKDIR, f"modelA_pack_{m}_bw0p2_pad19p0_molly172.npz")
         for m in ("2lpt0", "london0", "saclay0")}

TARGETS = {"2lpt0": dict(by_snr=36.61, win=22.22, z23=8.14, ratio=1.0016),
           "london0": dict(by_snr=62.91, win=28.39, z23=9.67, ratio=1.0501),
           "saclay0": dict(by_snr=54.61, win=25.77, z23=10.93, ratio=1.0281)}

out = {"command": "python s1_reproduce.py", "code_commit": "9d73365",
       "packs": PACKS, "resp_clamp": "both", "mocks": {}}

for mock, pth in PACKS.items():
    pack = load_pack(pth)
    res = FS.selftest(pack, resp_clamp="both")       # repaired production fold
    tab = FS.ratio_tables(res, pack)
    full = REP.window_closure_metrics(tab["by_nhat"], label="full")
    win = REP.window_closure_metrics(tab["by_nhat"], *REP.REPORTING_WINDOW,
                                     label="win")
    snr_rows = tab["by_snr"]
    zs = np.array([r["z"] for r in snr_rows if r["obs"] > 0])
    chi2_snr = float((zs ** 2).sum() / max(len(zs), 1))
    zsn = np.array([r["z"] for r in tab["by_z"] if r["obs"] > 0])
    chi2_byz = float((zsn ** 2).sum() / max(len(zsn), 1))
    print(f"\n===== {mock} =====")
    print(f"  full  mu/obs = {full['total_ratio']:.4f}   "
          f"(target {TARGETS[mock]['ratio']})")
    print(f"  win   chi2/dof = {win['chi2_dof']:.2f}   "
          f"(target {TARGETS[mock]['win']})   z_tot = {win['z_total']:+.2f}")
    print(f"  by_snr chi2/dof = {chi2_snr:.2f}   (target {TARGETS[mock]['by_snr']})"
          f"   n_rows = {len(zs)}")
    print(f"  by_z  chi2/dof = {chi2_byz:.2f}")
    for r in snr_rows:
        lo = pack.snr_edges[r["s"]]; hi = pack.snr_edges[r["s"] + 1]
        if r["obs"] > 0:
            print(f"    SNR [{lo:.0f},{hi:.0f})  mu={r['mu']:9.1f} "
                  f"obs={r['obs']:7.0f}  ratio={r['ratio']:.4f}  z={r['z']:+7.2f}")
    out["mocks"][mock] = dict(
        full_ratio=full["total_ratio"], full_chi2=full["chi2_dof"],
        win_chi2=win["chi2_dof"], win_ztot=win["z_total"],
        win_ratio=win["total_ratio"],
        by_snr_chi2=chi2_snr, by_z_chi2=chi2_byz,
        by_snr_rows=[dict(s=int(r["s"]),
                          snr_lo=float(pack.snr_edges[r["s"]]),
                          snr_hi=float(pack.snr_edges[r["s"] + 1]),
                          mu=r["mu"], obs=r["obs"], ratio=r["ratio"], z=r["z"])
                     for r in snr_rows],
        by_nhat_window=[b for b in win["per_bin"]],
        targets=TARGETS[mock])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "s1_reproduce.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("\nwrote s1_reproduce.json")
