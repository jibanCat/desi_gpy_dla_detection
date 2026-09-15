"""make_wg_figure.py — the Lya-WG S/N predictive diagnostic figure + the internal tables.

PI ruling 2026-09-14b §5 / §14.  READ-ONLY on every frozen object: the only computation is
the deterministic evaluation of the FROZEN fold at the stored posterior-median draw
(wg_predictive_breakdown.fold_at_median, a verbatim copy of the runners' own
`predictive_marginals` code path), gated by an exact reproduction of each run's STORED
`mu_over_obs_by_snr`.  Nothing is fitted, sampled or corrected.

Outputs (ratios and fractions only — no counts, no dN/dX):
  figures/2026-09-15_wg_diagnostic/wg_fig1_snr_predictive_real_vs_mock.png
  <cache>/wg_ratio_tables.json   (the ratio tables, for the internal appendix)
  stdout: the markdown tables for the internal appendix
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wg_predictive_breakdown as W

FIGDIR = "/home/mfho/desi_gpy_dla_notes/figures/2026-09-15_wg_diagnostic"
FIG = os.path.join(FIGDIR, "wg_fig1_snr_predictive_real_vs_mock.png")
CACHE = os.path.join(W.SCRATCH, "wg_diagnostic", "wg_ratio_tables.json")

SNR_LABELS = {2: "2–3", 3: "3–4", 4: "4–5", 5: "5–6", 6: "6–7", 7: ">7"}
FAM_LABEL = {"2lpt0": "2LPT-0", "london0": "London-0", "saclay0": "Saclay-0"}
K_LABEL = ["K0: 2.0 < z < 2.5", "K1: 2.5 < z < 3.0", "K2: 3.0 < z < 3.5"]
G_LABEL = {"19.5_20.0": r"$19.5 \leq \hat{N} < 20.0$", "20.0_20.3": r"$20.0 \leq \hat{N} < 20.3$",
           "20.3_22.4": r"$\hat{N} \geq 20.3$"}
PAL = ["#3b528b", "#21918c", "#b5367a"]


def style():
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.family": "serif",
                         "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
                         "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.2,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
                         "lines.linewidth": 1.2, "axes.linewidth": 0.6, "savefig.bbox": "tight"})


def _jsonable(d):
    return {k: (_jsonable(v) if isinstance(v, dict)
                else (np.asarray(v).tolist() if hasattr(v, "__len__") else v))
            for k, v in d.items()}


def compute(cache=CACHE, use_cache=True):
    if use_cache and os.path.exists(cache):
        return json.load(open(cache))
    ver = {}
    res = {}
    rr = W.real_run_jsons()
    ver["real"] = [W.verify_stored_snr(p, W.real_fixed()) for p in rr]
    res["real"] = _jsonable(W.median_over_runs(rr, lambda p: W.real_fixed()))
    res["real"]["runs"] = [os.path.basename(p) for p in rr]
    for fam in W.MOCK_FAMILIES:
        ms = W.mock_run_jsons(fam)
        ver[fam] = [W.verify_stored_snr(p, W.mock_fixed(p)) for p in ms]
        res[fam] = _jsonable(W.median_over_runs(ms, W.mock_fixed))
        res[fam]["runs"] = [os.path.basename(p) for p in ms]
    res["verification"] = {k: dict(n_runs=len(v), max_abs_diff_vs_stored=max(x[0] for x in v),
                                   n_strata_compared=v[0][1], tolerance=1e-10)
                           for k, v in ver.items()}
    W.assert_ratios_only({k: {kk: vv for kk, vv in v.items()
                              if kk in ("by_snr", "by_snr_K", "by_snr_nhat_group", "by_nhat")}
                          for k, v in res.items() if k != "verification"})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(res, open(cache, "w"), indent=1)
    return res


def _live(res):
    return np.asarray(res["real"]["live_snr"], bool)


def make_figure(res, out=FIG):
    style()
    live = _live(res)
    sidx = np.where(live)[0]
    x = np.arange(len(sidx))
    xt = [SNR_LABELS[int(i)] for i in sidx]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))

    # --- (a) all-K S/N marginal: real vs the three mocks, mock-to-mock spread as a band
    ax = axes[0, 0]
    M = np.stack([np.asarray(res[f]["by_snr"], float)[sidx] for f in W.MOCK_FAMILIES])
    ax.fill_between(x, M.min(0), M.max(0), color="0.55", alpha=0.25, lw=0,
                    label="mock family spread (min–max of 3)")
    for f, c in zip(W.MOCK_FAMILIES, ["#5ec962", "#21918c", "#3b528b"]):
        ax.plot(x, np.asarray(res[f]["by_snr"], float)[sidx], lw=1.0, color=c, alpha=0.85,
                label=f"{FAM_LABEL[f]} (median of {res[f]['n_runs']} runs)")
    ax.plot(x, np.asarray(res["real"]["by_snr"], float)[sidx], "k-o", ms=3.5, lw=1.8,
            label=f"REAL survey (median of {res['real']['n_runs']} runs)")
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set_title("(a) predictive ratio by S/N stratum: real vs mocks")
    ax.set_ylabel(r"posterior-median $\mu$ / observed")
    ax.legend(loc="upper right", ncol=1)

    # --- (b) real, S/N x coarse-z block K (mock envelope per K, faint, for reference)
    ax = axes[0, 1]
    for K in range(3):
        mk = np.stack([np.asarray(res[f]["by_snr_K"], float)[K][sidx] for f in W.MOCK_FAMILIES])
        ax.fill_between(x, mk.min(0), mk.max(0), color=PAL[K], alpha=0.13, lw=0)
        ax.plot(x, np.asarray(res["real"]["by_snr_K"], float)[K][sidx], "-o", ms=3.2,
                color=PAL[K], label=K_LABEL[K])
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set_title("(b) REAL: S/N $\\times$ coarse-$z$ block (bands = mock spread, same blocks)")
    ax.legend(loc="upper right")

    # --- (c) real, S/N x observed N-hat group
    ax = axes[1, 0]
    for i, (g, _, _) in enumerate(W.NHAT_GROUPS):
        mg = np.stack([np.asarray(res[f]["by_snr_nhat_group"][g], float)[sidx] for f in W.MOCK_FAMILIES])
        ax.fill_between(x, mg.min(0), mg.max(0), color=PAL[i], alpha=0.13, lw=0)
        ax.plot(x, np.asarray(res["real"]["by_snr_nhat_group"][g], float)[sidx], "-o", ms=3.2,
                color=PAL[i], label=G_LABEL[g])
    ax.axhline(1.0, color="0.3", lw=0.8)
    ax.set_title(r"(c) REAL: S/N $\times$ observed $\hat{N}$ group (bands = mock spread)")
    ax.set_ylabel(r"posterior-median $\mu$ / observed")
    ax.legend(loc="upper right")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(x); ax.set_xticklabels(xt); ax.set_xlabel("spectrum S/N stratum")

    # --- (d) the N-hat marginal, for context
    ax = axes[1, 1]
    cc = np.asarray(res["real"]["nhat_centres"], float)
    keep = cc <= 21.65                      # rows above this carry too few objects to read
    Mn = np.stack([np.asarray(res[f]["by_nhat"], float)[keep] for f in W.MOCK_FAMILIES])
    ax.fill_between(cc[keep], Mn.min(0), Mn.max(0), color="0.55", alpha=0.25, lw=0,
                    label="mock family spread (min–max of 3)")
    ax.plot(cc[keep], np.asarray(res["real"]["by_nhat"], float)[keep], "k-o", ms=3.0, lw=1.5,
            label="REAL survey")
    ax.axhline(1.0, color="0.3", lw=0.8)
    for v in (20.0, 20.3):
        ax.axvline(v, color="0.5", ls=":", lw=0.8)
    ax.set_title(r"(d) context: predictive ratio by observed $\log N_{\rm HI}$")
    ax.set_xlabel(r"observed $\log N_{\rm HI}$ (rows to 21.6 shown)")
    ax.set_ylabel(r"posterior-median $\mu$ / observed")
    ax.legend(loc="upper left")

    fig.suptitle("Frozen Paper-1 low-$z$ model (B + $\\varphi_{2\\rm LPT}$ + C1nsadd + M1CUT): "
                 "posterior-predictive ratios, real survey vs three independent mock families\n"
                 "deterministic read-out at the stored posterior-median draw — no fit, no correction "
                 "(PI ruling 2026-09-14b §5, §14)", fontsize=8.0, y=1.045)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    return out


def _row(name, v, sidx):
    return "| " + name + " | " + " | ".join(f"{np.asarray(v,float)[i]:.3f}" for i in sidx) + " |"


def tables_md(res):
    live = _live(res); sidx = np.where(live)[0]
    hdr = "| | " + " | ".join(f"S/N {SNR_LABELS[int(i)]}" for i in sidx) + " |"
    sep = "|---" * (len(sidx) + 1) + "|"
    L = ["## T1 — mu/obs by S/N stratum (all z, all N-hat)", "", hdr, sep,
         _row("**REAL** (median of %d runs)" % res["real"]["n_runs"], res["real"]["by_snr"], sidx)]
    for f in W.MOCK_FAMILIES:
        L.append(_row(f"{FAM_LABEL[f]} (median of {res[f]['n_runs']})", res[f]["by_snr"], sidx))
    L += ["", "## T2 — REAL mu/obs by S/N x coarse-z block", "", hdr, sep]
    for K in range(3):
        L.append(_row(K_LABEL[K], np.asarray(res["real"]["by_snr_K"], float)[K], sidx))
    L += ["", "### T2b — the same decomposition on the mocks", "", hdr, sep]
    for f in W.MOCK_FAMILIES:
        for K in range(3):
            L.append(_row(f"{FAM_LABEL[f]} {K_LABEL[K].split(':')[0]}",
                          np.asarray(res[f]["by_snr_K"], float)[K], sidx))
    L += ["", "## T3 — REAL mu/obs by S/N x observed N-hat group", "", hdr, sep]
    for g, _, _ in W.NHAT_GROUPS:
        L.append(_row(g.replace("_", "–"), res["real"]["by_snr_nhat_group"][g], sidx))
    L += ["", "### T3b — the same decomposition on the mocks", "", hdr, sep]
    for f in W.MOCK_FAMILIES:
        for g, _, _ in W.NHAT_GROUPS:
            L.append(_row(f"{FAM_LABEL[f]} {g.replace('_','–')}", res[f]["by_snr_nhat_group"][g], sidx))
    L += ["", "## T4 — mu/obs by observed log N_HI (context)", "",
          "| row | REAL | " + " | ".join(FAM_LABEL[f] for f in W.MOCK_FAMILIES) + " |",
          "|---|---|---|---|---|"]
    cc = np.asarray(res["real"]["nhat_centres"], float)
    for i, c in enumerate(cc):
        L.append(f"| {c-0.05:.1f}–{c+0.05:.1f} | {np.asarray(res['real']['by_nhat'],float)[i]:.3f} | "
                 + " | ".join(f"{np.asarray(res[f]['by_nhat'],float)[i]:.3f}" for f in W.MOCK_FAMILIES) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    res = compute(use_cache="--recompute" not in sys.argv)
    print("VERIFICATION:", json.dumps(res["verification"], indent=1))
    print("figure:", make_figure(res))
    print()
    print(tables_md(res))
