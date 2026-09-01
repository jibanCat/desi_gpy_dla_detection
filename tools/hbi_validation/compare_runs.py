#!/usr/bin/env python3
"""compare_runs.py — cross-run science comparison of the 2026-09-02 validation matrix (kickoff §33–36; request §8):
for each run's geometry JSON, the science outputs (dN/dX ≥20.0 / ≥20.3 all-z + coarse blocks, Ω[20.3,21.6], the 10 reporting
bins) as absolute / fractional / baseline-σ shifts vs R0; the catalogue decomposition (FP shares, Λ, t_K, A_K, C̄); and the
low-z (2.0 ≤ z < 2.5, 19.7 ≤ log N < 20.3) common-bin figure. PRIVATE outputs.

    python tools/hbi_validation/compare_runs.py --baseline geometry_R0.json --runs R1=geometry_R1.json R2=... --out-dir DIR
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.hbi_validation.viz_common import plt, SLOTS, INK2   # noqa: E402

RUN_SLOT = {"R0": 0, "R1": 1, "R2": 2, "R2B": 3, "R3": 4, "R4": 5, "R2b": 6}


def sig(q):
    """posterior sigma from the 16–84 half-width of a 5-quantile list."""
    return 0.5 * (q[3] - q[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True); ap.add_argument("--runs", nargs="+", required=True, help="RUN=geometry.json")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    G0 = json.load(open(a.baseline)); runs = {"R0": G0}
    for spec in a.runs:
        k, p = spec.split("="); runs[k] = json.load(open(p))
    rows = []; md = ["# Science-output comparison vs R0 (absolute / fractional / shift in BASELINE posterior σ = half the 16–84 range)\n",
                     "| run | quantity | R0 median | run median | Δ abs | Δ frac | Δ/σ_R0 |", "|---|---|---:|---:|---:|---:|---:|"]
    def add(run, name, q0, q):
        d = q[2] - q0[2]; s0 = sig(q0)
        rows.append(dict(run=run, quantity=name, baseline_median=q0[2], run_median=q[2], delta_abs=d, delta_frac=d / q0[2], delta_over_sigma0=d / s0 if s0 > 0 else None, run_q=q, baseline_q=q0))
        md.append(f"| {run} | {name} | {q0[2]:.5g} | {q[2]:.5g} | {d:+.3g} | {100*d/q0[2]:+.2f} % | {d/s0:+.2f} |")
    for run, G in runs.items():
        if run == "R0":
            continue
        s0, s = G0["science"], G["science"]
        add(run, "dN/dX(≥20.0) all-z", s0["dndx_ge20p0_allz"], s["dndx_ge20p0_allz"])
        add(run, "dN/dX(≥20.3) all-z", s0["dndx_ge20p3_allz"], s["dndx_ge20p3_allz"])
        add(run, "Ω_HI[20.3,21.6] all-z (h=0.70)", s0["omega_20p3_21p6_allz_h0p70"], s["omega_20p3_21p6_allz_h0p70"])
        for b0, b in zip(s0["reporting_bins"], s["reporting_bins"]):
            add(run, f"f [{b0['bin'][0]},{b0['bin'][1]})", b0["f_post"], b["f_post"])
        for K in range(3):
            # coarse medians only carry the median: use the all-z sigma scaled by sqrt(1/path share) is NOT available -> report abs/frac only
            m0, m = s0["dndx_ge20p3_coarse_median"][K], s["dndx_ge20p3_coarse_median"][K]
            rows.append(dict(run=run, quantity=f"dN/dX(≥20.3) z-block {K}", baseline_median=m0, run_median=m, delta_abs=m - m0, delta_frac=(m - m0) / m0, delta_over_sigma0=None))
            md.append(f"| {run} | dN/dX(≥20.3) z-block {K} | {m0:.5g} | {m:.5g} | {m-m0:+.3g} | {100*(m-m0)/m0:+.2f} % | — |")
            m0, m = s0["dndx_ge20p0_coarse_median"][K], s["dndx_ge20p0_coarse_median"][K]
            rows.append(dict(run=run, quantity=f"dN/dX(≥20.0) z-block {K}", baseline_median=m0, run_median=m, delta_abs=m - m0, delta_frac=(m - m0) / m0, delta_over_sigma0=None))
            md.append(f"| {run} | dN/dX(≥20.0) z-block {K} | {m0:.5g} | {m:.5g} | {m-m0:+.3g} | {100*(m-m0)/m0:+.2f} % | — |")
    # largest CDDF shift per run
    md.append("\n## Largest reporting-bin shift per run\n\n| run | bin | Δ frac | Δ/σ_R0 |\n|---|---|---:|---:|")
    for run in runs:
        if run == "R0":
            continue
        rr = [r for r in rows if r["run"] == run and r["quantity"].startswith("f [")]
        w = max(rr, key=lambda r: abs(r["delta_over_sigma0"] or 0))
        md.append(f"| {run} | {w['quantity']} | {100*w['delta_frac']:+.2f} % | {w['delta_over_sigma0']:+.2f} |")
    # catalogue decomposition
    md.append("\n## Catalogue decomposition per run\n\n| run | FP share all | FP share 19.7–20.3 | FP share ≥20.3 | FP share sub-DLA z0 | Λ median | t₀ / t₁ / t₂ | A₀ / A₁ / A₂ | C̄_subDLA | d̄ | f_subDLA all-z |\n|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for run, G in runs.items():
        S = G["fp_shares"]; t = G["t"]; A = G["A_K"]; C = G["completeness"]
        fsub = G["corr_t_vs"].get("f_subdla_19p7_20p3_allz")
        md.append(f"| {run} | {S['all_19p5_22p4']['allz']['fp_share_of_mu_cat'][1]:.4f} | {S['subdla_19p7_20p3']['allz']['fp_share_of_mu_cat'][1]:.4f} | {S['dla_ge20p3']['allz']['fp_share_of_mu_cat'][1]:.4f} | {S['subdla_19p7_20p3']['z0_2.0_2.5']['fp_share_of_mu_cat'][1]:.4f} | {G['log_Lambda']['Lambda_median']:.3f} | {t['K0']['median']:+.3f} / {t['K1']['median']:+.3f} / {t['K2']['median']:+.3f} | {A['K0']['median']:.3f} / {A['K1']['median']:.3f} / {A['K2']['median']:.3f} | {C['C_bar'][2]:.4f} | {C['d_bar'][2]:+.4f} | — |")
    md.append("\n## Convergence per run (per arm: Λ split-R̂, t R̂, sites with R̂>1.10)\n")
    for run, G in runs.items():
        md.append(f"- **{run}**: " + "; ".join(f"{k}: PE {[round(x) for x in v['pe_chain_mean']]}, Λ R̂ {v['lam_rhat']:.2f}, t R̂ {[round(x,2) for x in v['t_rhat']]}, R̂>1.10: {v['site_rhat_n_gt_1p10']}" for k, v in G["per_arm"].items()))
    open(os.path.join(a.out_dir, "SCIENCE_COMPARISON.md"), "w").write("\n".join(md) + "\n")
    json.dump(rows, open(os.path.join(a.out_dir, "science_comparison.json"), "w"), indent=1)
    # low-z decomposition figure: observed, μ_TP, μ_FP (sub-DLA z0), C̄, f_subDLA — one panel per run, identical binning/axes
    names = list(runs); fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    x = np.arange(len(names))
    obs = G0["fp_shares"]["subdla_19p7_20p3"]["z0_2.0_2.5"]["obs"]
    tp = [runs[r]["fp_shares"]["subdla_19p7_20p3"]["z0_2.0_2.5"]["mu_TP"] for r in names]; fp = [runs[r]["fp_shares"]["subdla_19p7_20p3"]["z0_2.0_2.5"]["mu_FP"] for r in names]
    axs[0].bar(x - 0.18, [t[1] for t in tp], 0.34, color=SLOTS[0], label="posterior TP"); axs[0].bar(x + 0.18, [f[1] for f in fp], 0.34, color=SLOTS[1], label="posterior FP")
    axs[0].errorbar(x - 0.18, [t[1] for t in tp], yerr=[[t[1] - t[0] for t in tp], [t[2] - t[1] for t in tp]], fmt="none", ecolor=INK2, lw=0.8)
    axs[0].errorbar(x + 0.18, [f[1] for f in fp], yerr=[[f[1] - f[0] for f in fp], [f[2] - f[1] for f in fp]], fmt="none", ecolor=INK2, lw=0.8)
    axs[0].axhline(obs, color=INK2, lw=0.8, ls="--"); axs[0].text(len(names) - 0.5, obs, f"observed {obs:.0f}", fontsize=6, va="bottom", ha="right", color=INK2)
    axs[0].set_xticks(x); axs[0].set_xticklabels(names); axs[0].set_ylabel("counts, 19.7 ≤ log N < 20.3, 2.0 ≤ z < 2.5"); axs[0].legend(fontsize=6); axs[0].set_title("TP / FP decomposition", fontsize=8, loc="left")
    cb = [runs[r]["completeness"]["C_bar"] for r in names]
    axs[1].errorbar(x, [c[2] for c in cb], yerr=[[c[2] - c[1] for c in cb], [c[3] - c[2] for c in cb]], fmt="o", color=SLOTS[0], ms=4, lw=0.8)
    axs[1].axhline(G0["completeness"]["C_bar_central"], color=INK2, lw=0.8, ls="--"); axs[1].set_xticks(x); axs[1].set_xticklabels(names); axs[1].set_ylabel("C̄ subDLA (effective completeness)"); axs[1].set_title("effective completeness (dashed = central calibration)", fontsize=8, loc="left")
    fs = [next((rw for rw in rows if rw["run"] == r and rw["quantity"] == "f [19.9,20.1)"), None) for r in names]
    q = [G0["science"]["reporting_bins"][1]["f_post"]] + [runs[r]["science"]["reporting_bins"][1]["f_post"] for r in names[1:]]
    axs[2].errorbar(x, [v[2] for v in q], yerr=[[v[2] - v[1] for v in q], [v[3] - v[2] for v in q]], fmt="o", color=SLOTS[0], ms=4, lw=0.8)
    axs[2].set_xticks(x); axs[2].set_xticklabels(names); axs[2].set_ylabel("f [19.9,20.1) all-z (per unit N per dX)"); axs[2].set_title("latent population in the window", fontsize=8, loc="left")
    fig.suptitle("Low-z sub-DLA decomposition across runs (identical binning; internal diagnostic)", fontsize=8, x=0.02, ha="left"); fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "lowz_decomposition_runs.png"), dpi=140); plt.close(fig)
    print("\n".join(md[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
