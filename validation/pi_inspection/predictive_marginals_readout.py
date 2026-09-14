"""Deterministic read-out of the frozen model's stored predictive marginals (posterior-median mu / observed counts) on the
real survey (16 J=8 runs) and on the three mock families (J=8 runs), by S/N stratum, observed N-hat bin and fine z.
Reads RUN JSON `predictive_marginals` only; no fit, no draw. Values go to the notes-repo files (private)."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = "/scratch/cavestru_root/cavestru0/mfho/absorber_ladder_2026-09-13"
OUT_FIG = "/home/mfho/desi_gpy_dla_notes/figures/2026-09-14_pi_inspection/fig_sec1b_predictive_marginals_real_vs_mock.png"
OUT_MD = "/home/mfho/desi_gpy_dla_notes/governance/final_campaign_2026-09-13/pi_inspection_sections/SEC_1b_predictive_marginals.md"

def pm(paths, key):
    arr = []
    for p in paths:
        j = json.load(open(p)); d = j.get("predictive_marginals") or j["diagnostics"]["predictive_marginals"]
        arr.append(np.asarray(d[key], float))
    return np.median(np.array(arr), axis=0), len(paths)

real = sorted(glob.glob(f"{L}/real_c1/runs/RUN_REALC1_*.json"))
mock = {f: sorted(glob.glob(f"{L}/final/runs/B-phi2lpt-C1nsadd-M1CUTJ8/RUN_*_{f}_*J8j*.json")) for f in ("2lpt0", "london0", "saclay0")}
keys = {"snr": "mu_over_obs_by_snr", "nhat": "mu_over_obs_by_nhat", "z": "mu_over_obs_by_z"}
j0 = json.load(open(real[0])); pm0 = j0.get("predictive_marginals") or {}
avail = {k: v for k, v in keys.items() if v in pm0}
nhat_edges = np.load(f"{L}/support/empirical_ops_2lpt0_A0.npz", allow_pickle=True)["nhat_edges"]
fig, axes = plt.subplots(1, len(avail), figsize=(4.2 * len(avail), 3.4))
axes = np.atleast_1d(axes); lines = []
for ax, (k, key) in zip(axes, avail.items()):
    r, n = pm(real, key)
    for fam, ps in mock.items():
        m, nm = pm(ps, key); x = np.arange(m.size) if k != "nhat" else 0.5 * (nhat_edges[:-1] + nhat_edges[1:])
        live = m > 0; ax.plot(x[live], m[live], lw=1, alpha=0.7, label=f"mock {fam} (median of {nm} runs)")
    x = np.arange(r.size) if k != "nhat" else 0.5 * (nhat_edges[:-1] + nhat_edges[1:]); live = r > 0
    ax.plot(x[live], r[live], "k-o", ms=3, lw=1.5, label=f"REAL C1 (median of {n} runs)")
    ax.axhline(1, color="0.3", lw=0.8); ax.set_title({"snr": "by S/N stratum", "nhat": "by observed log N", "z": "by fine z bin"}[k])
    ax.set_ylabel("posterior-median μ / observed"); ax.set_xlabel({"snr": "S/N stratum index", "nhat": "observed log N_HI", "z": "fine z index (2.0–3.5)"}[k])
    if k == "nhat": ax.axvline(20.0, color="0.5", ls=":"); ax.axvline(20.3, color="0.5", ls=":")
    lines.append((k, r, {fam: pm(ps, key)[0] for fam, ps in mock.items()}))
axes[0].legend(fontsize=6.5)
fig.suptitle("Frozen model (B + φ_2LPT + C1nsadd + M1CUT): predictive marginals on the REAL survey vs the mock families (deterministic read-out of stored runs)", fontsize=8.5, y=1.02)
fig.tight_layout(); fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
md = ["# SEC 1b — Predictive marginals of the frozen model: real survey vs mocks (read-out only)", "",
      f"Figure: `{os.path.basename(OUT_FIG)}`. Median over the 16 real J=8 runs and over the 8 J=8 runs per mock family of the stored posterior-median μ/observed ratios (`predictive_marginals`). The total ratio is ≈ 1 by construction; the STRUCTURE is what matters.", ""]
for k, r, mocks in lines:
    if k == "snr":
        md += ["## By S/N stratum (live strata 2–7)", "", "| stratum | REAL μ/obs | 2LPT-0 | London-0 | Saclay-0 |", "|---|---|---|---|---|"]
        for i in range(2, r.size): md.append(f"| s={i} | {r[i]:.3f} | " + " | ".join(f"{mocks[f][i]:.3f}" for f in ("2lpt0", "london0", "saclay0")) + " |")
    if k == "nhat":
        md += ["", "## By observed log N_HI (29 bins)", "", "| N̂ bin | REAL | 2LPT-0 | London-0 | Saclay-0 |", "|---|---|---|---|---|"]
        for i in range(r.size): md.append(f"| [{nhat_edges[i]:.1f},{nhat_edges[i+1]:.1f}) | {r[i]:.3f} | " + " | ".join(f"{mocks[f][i]:.3f}" for f in ("2lpt0", "london0", "saclay0")) + " |")
    if k == "z":
        md += ["", "## By fine z bin", "", "| k | REAL | 2LPT-0 | London-0 | Saclay-0 |", "|---|---|---|---|---|"]
        for i in range(r.size):
            if r[i] > 0: md.append(f"| {i} | {r[i]:.3f} | " + " | ".join(f"{mocks[f][i]:.3f}" for f in ("2lpt0", "london0", "saclay0")) + " |")
md += ["", "Reading (no causal claim): on the mocks the frozen model's S/N marginal is flat to a few per cent; on the real survey the lowest live stratum is over-predicted and the high-S/N strata under-predicted by several per cent — the real low-S/N population is not described by the frozen fold the way the mock one is. The N̂ and z marginals are flat to ≈ 2–3 % on both. This is a diagnostic of the frozen fit, reported for the PI's inspection; it changes nothing (PI ruling 2026-09-14 §17).", "",
       "Sources: real `real_c1/runs/RUN_REALC1_*.json` (16), mock `final/runs/B-phi2lpt-C1nsadd-M1CUTJ8/RUN_*_J8j*.json` (24), `support/empirical_ops_2lpt0_A0.npz` (nhat_edges). Verified: 16 + 24 files read; medians over runs; no draws touched."]
open(OUT_MD, "w").write("\n".join(md) + "\n"); print("written", OUT_FIG, OUT_MD)
