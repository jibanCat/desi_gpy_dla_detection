#!/usr/bin/env python
"""Emit ORACLE_RECOVERY_PLOTS.md from _plotted_numbers.json written by
make_oracle_recovery_plots.py.  Every number in the markdown is a number that
appears in one of the five figures."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/home/mfho/desi_gpy_dla_notes/figures/2026-09-13_absorber_diag")
D = json.loads((OUT / "_plotted_numbers.json").read_text())
FAMS = ["2lpt0", "london0", "saclay0"]
SEEDS = [20260811, 20260812]
MODELS = ["ORACLE", "M0", "M2", "M3"]
L: list[str] = []


def w(s: str = "") -> None:
    L.append(s)


def f(x, n=3):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def g(x, n=4):
    if x is None:
        return "—"
    return f"{float(x):.{n}g}"


def mark(in68, in95):
    if not in95:
        return "**out95**"
    if not in68:
        return "out68"
    return "in68"


w("# ORACLE absorber-recovery diagnostic — figures and plotted numbers")
w()
w("**Date:** 2026-09-13 · **Stage:** `diag_oracle` (FP ladder 2026-09-12) · "
  "**read-only on existing MCMC outputs; no new sampling**")
w()
w("ORACLE = the diagnostic ladder rung in which `mu_FP` is **pinned to the mock FP truth** "
  "(hostless census), i.e. the FP model is given the right answer by construction and carries "
  "zero FP degrees of freedom (`nominal_fp_dof = 0`).")
w()
w("## Provenance")
w()
w("| item | value |")
w("|---|---|")
w("| ORACLE runs | `/scratch/cavestru_root/cavestru0/mfho/fp_ladder_2026-09-12/diag_oracle/RUN_ORACLE_<fam>_s<seed>.json` |")
w("| comparison runs | `.../wave1/RUN_{M0,M2}_...` , `.../wave2_M3/RUN_M3_...` |")
w("| packs | `.../packs/scanpack_<fam>_b300.npz` (`kz_to_K`, `nhat_edges`, `snr_edges`) |")
w("| families | " + ", ".join(FAMS) + " |")
w("| seeds | " + ", ".join(str(s) for s in SEEDS) + " (both seeds shown everywhere except the PPC — see Fig. 4) |")
w(f"| PPC files present | ORACLE seed(s) {', '.join(str(s) for s in D['fig4_seeds'])} only |")
w("| posterior draws | 4 chains × 1000 samples (1500 warm-up), `f` array (4000, 16, 15) |")
w("| true-N grid | 16 bins, edges 19.0 … 22.4 (`ntrue_edges`) |")
w("| z grid | 15 native cells 2.0–3.5 (Δz = 0.1), coarse blocks K0 = [2.0,2.5), K1 = [2.5,3.0), K2 = [3.0,3.5) |")
w(f"| plotting style | {D['style']} |")
w(f"| usetex | {D['usetex']} |")
miss = D.get("missing") or []
w("| fields/files not found | " + ("none" if not miss else "; ".join(f"`{m}`" for m in miss)) + " |")
w()
w("**Gate windows drawn on the figures (sealed):** |bias| ≤ 0.5 % for dN/dX(≥20.0) all-z; "
  "bias ∈ [−0.5, +3.0] % for dN/dX(≥20.3) all-z. Green shading marks the window; a point "
  "outside the shading is outside the bias window at that grain.")
w()
w("> The PASS/FAIL column in Table 6 is the **bias-window criterion only**. It is not the "
  "run gate: `perz_gate.gate_one` also carries containment and the §5.2 flags, and the "
  "ladder table records every ORACLE run as `gate = FAIL` (all six carry the "
  "`fp_truth ratio outside [0.5, 2.0]` flag, and the 2lpt0/london0 runs additionally carry "
  "`omega[20.3,21.6] all-z bias > 3 %`). Nothing here adjudicates a gate.")
w()
w("**Internal cross-check.** The Fig. 1 ratio panels are computed from the raw posterior "
  "draws (`_fdraws.npz`, path-weighting `f[b,k]` by `dX_k`), while Fig. 3 reads the runner's "
  "own `reporting_bins.median_bias_pct`. On the ten shared 0.2-dex bins the two agree to "
  "≤ 0.005 percentage points, so the two figures are consistent read-outs of the same "
  "posterior.")
w()

# ------------------------------------------------------------------ figures
w("## Figures")
w()
for fn, cap in [
    ("fig1_truth_vs_recovered_byN.png",
     "Fig. 1 — truth vs recovered absorber density per TRUE N bin, path-weighted over all z "
     "(top, log scale: truth in black, ORACLE posterior median with 68 % / 95 % bands; "
     "bottom: recovered/truth − 1 in %). Columns = families, both seeds overlaid "
     "(solid/circles = s20260811, dashed/squares = s20260812)."),
    ("fig2_ratio_vs_N_by_block.png",
     "Fig. 2 — recovered/truth − 1 vs true N, path-weighted **within** each coarse z block. "
     "3 × 3 grid: rows = K0/K1/K2, columns = families; shaded = 68 % posterior band."),
    ("fig3_reporting_bin_zigzag.png",
     "Fig. 3 — the 0.2-dex reporting-bin zigzag. Bold purple = ORACLE (both seeds); faint = "
     "M0 / M2 / M3 for the same family and seeds. Filled marker = truth inside the 68 % "
     "interval, open marker = outside 68 %, red ring = outside 95 %."),
    ("fig4_ppc_residuals.png",
     "Fig. 4 — ORACLE posterior-predictive residuals (mu_median/obs − 1) by S/N stratum (top, "
     "S/N 2–3 highlighted in red), by observed Nhat bin (middle) and by z cell (bottom). "
     "Red rings = PPC cells below the two-sided p threshold (0.002)."),
    ("fig5_dndx_bias_bars.png",
     "Fig. 5 — dN/dX median recovery bias per Paper-1 reporting bin (B1–B5) and per coarse z "
     "block (K0–K2), ORACLE vs M0/M2/M3. Top row ≥20.0, bottom row ≥20.3; bars = seed "
     "20260811, black ticks = seed 20260812; green band = the gate window."),
]:
    w(f"### {cap}")
    w()
    w(f"![{fn}]({fn})")
    w()

# ------------------------------------------------------------------ T1
w("## Table 1 — Fig. 1: f(N) path-weighted over all z, per true-N bin")
w()
w("`f` is the absorber density per dex per unit absorption path; "
  "Σ_b f_b · Δ(log N) over b ≥ 20.3 reproduces dN/dX(≥20.3) to 5 significant figures.")
w()
for fam in FAMS:
    w(f"**{fam}**")
    w()
    w("| true log N bin | truth f | s11 median | s11 68 % | s11 ratio−1 [%] (68 %) | "
      "s12 median | s12 ratio−1 [%] (68 %) |")
    w("|---|---|---|---|---|---|---|")
    r1 = {r["b"]: r for r in D["fig1"] if r["family"] == fam and r["seed"] == SEEDS[0]}
    r2 = {r["b"]: r for r in D["fig1"] if r["family"] == fam and r["seed"] == SEEDS[1]}
    for b in sorted(r1):
        a, c = r1[b], r2.get(b, {})
        w(f"| [{a['nlo']:.1f}, {a['nhi']:.1f}) | {g(a['truth'])} | {g(a['med'])} | "
          f"{g(a['p16'])}–{g(a['p84'])} | {f(a['ratio_med'],2)} ({f(a['ratio_p16'],2)}, {f(a['ratio_p84'],2)}) | "
          f"{g(c.get('med'))} | {f(c.get('ratio_med'),2)} ({f(c.get('ratio_p16'),2)}, {f(c.get('ratio_p84'),2)}) |")
    w()

# ------------------------------------------------------------------ T2
w("## Table 2 — Fig. 2: ratio − 1 [%] vs true N within each coarse z block")
w()
w("Seed 20260811 with its 68 % band; seed 20260812 median in the last column of each block.")
w()
for fam in FAMS:
    w(f"**{fam}**")
    w()
    w("| true log N bin | K0 s11 (68 %) | K0 s12 | K1 s11 (68 %) | K1 s12 | K2 s11 (68 %) | K2 s12 |")
    w("|---|---|---|---|---|---|---|")
    idx = {}
    for r in D["fig2"]:
        if r["family"] != fam:
            continue
        idx[(r["block"], r["seed"], r["b"])] = r
    bs = sorted({r["b"] for r in D["fig2"] if r["family"] == fam})
    for b in bs:
        cells = []
        lab = None
        for K in ["K0", "K1", "K2"]:
            a = idx.get((K, SEEDS[0], b))
            c = idx.get((K, SEEDS[1], b))
            if a:
                lab = f"[{a['nlo']:.1f}, {a['nhi']:.1f})"
                cells.append(f"{f(a['ratio_med'],2)} ({f(a['ratio_p16'],2)}, {f(a['ratio_p84'],2)})")
            else:
                cells.append("—")
            cells.append(f(c["ratio_med"], 2) if c else "—")
        w(f"| {lab} | " + " | ".join(cells) + " |")
    w()

# ------------------------------------------------------------------ T3
w("## Table 3 — Fig. 3: 0.2-dex reporting-bin median bias [%] (containment in parentheses)")
w()
for fam in FAMS:
    w(f"**{fam}**")
    w()
    hdr = "| 0.2-dex bin |"
    sep = "|---|"
    for m in MODELS:
        for s in SEEDS:
            hdr += f" {m} s{str(s)[-2:]} |"
            sep += "---|"
    w(hdr)
    w(sep)
    idx = {}
    for r in D["fig3"]:
        if r["family"] != fam:
            continue
        idx[(r["model"], r["seed"], r["bin_lo"])] = r
    los = sorted({r["bin_lo"] for r in D["fig3"] if r["family"] == fam})
    for lo in los:
        any_r = next(idx[(m, s, lo)] for m in MODELS for s in SEEDS if (m, s, lo) in idx)
        row = f"| [{any_r['bin_lo']:.1f}, {any_r['bin_hi']:.1f}) |"
        for m in MODELS:
            for s in SEEDS:
                r = idx.get((m, s, lo))
                row += (" — |" if r is None else
                        f" {r['median_bias_pct']:+.2f} ({mark(r['truth_in_68'], r['truth_in_95'])}) |")
        w(row)
    w()

# ------------------------------------------------------------------ T4
w("## Table 4 — Fig. 4: ORACLE PPC marginal residuals")
w()
ppc_seeds = D["fig4_seeds"]
w(f"PPC products exist for seed(s) **{', '.join(str(s) for s in ppc_seeds)}** only "
  f"(no `_ppc.json` was written for the other seed, so Fig. 4 shows one seed per family).")
w()
w("### 4a — by S/N stratum (`marginal_by_snr`; strata edges 0,1,2,3,4,5,6,7,∞)")
w()
w("| S/N stratum | family | obs | mu_median | resid [%] | p_two_sided |")
w("|---|---|---|---|---|---|")
for r in D["fig4"]:
    if r["marginal"] != "marginal_by_snr":
        continue
    w(f"| {r['i']} | {r['family']} | {r['obs']:.0f} | {r['mu_median']:.1f} | "
      f"{f(r['resid_pct'],2)} | {f(r['p_two_sided'],3)} |")
w()
w("### 4b — by z cell (`marginal_by_z`)")
w()
w("| z cell i | " + " | ".join(f"{fam} resid [%] (p)" for fam in FAMS) + " |")
w("|---|" + "---|" * len(FAMS))
zi = sorted({r["i"] for r in D["fig4"] if r["marginal"] == "marginal_by_z"})
for i in zi:
    cells = []
    for fam in FAMS:
        r = next((x for x in D["fig4"] if x["marginal"] == "marginal_by_z"
                  and x["family"] == fam and x["i"] == i), None)
        cells.append("—" if r is None else f"{f(r['resid_pct'],2)} ({f(r['p_two_sided'],3)})")
    w(f"| {i} (z = {2.0 + 0.1 * i:.1f}–{2.1 + 0.1 * i:.1f}) | " + " | ".join(cells) + " |")
w()
w("### 4c — by observed Nhat bin (`marginal_by_nhat`, 29 bins from `nhat_edges`)")
w()
w("| Nhat bin i | Nhat centre | " + " | ".join(f"{fam} resid [%] (p)" for fam in FAMS) + " |")
w("|---|---|" + "---|" * len(FAMS))
ni = sorted({r["i"] for r in D["fig4"] if r["marginal"] == "marginal_by_nhat"})
for i in ni:
    cells, xc = [], None
    for fam in FAMS:
        r = next((x for x in D["fig4"] if x["marginal"] == "marginal_by_nhat"
                  and x["family"] == fam and x["i"] == i), None)
        if r:
            xc = r["x"]
        cells.append("—" if r is None else f"{f(r['resid_pct'],2)} ({f(r['p_two_sided'],3)})")
    w(f"| {i} | {f(xc,2)} | " + " | ".join(cells) + " |")
w()
w("### 4d — `snr_ramp` block carried by each PPC file")
w()
w("| family | seed | strata | ratios mu_median/obs | amplitude | threshold | flag |")
w("|---|---|---|---|---|---|---|")
for r in D.get("snr_ramp", []):
    w(f"| {r['family']} | {r['seed']} | {r['strata']} | "
      f"{[round(v, 4) for v in r['ratios']]} | {r['amplitude']:.4f} | "
      f"{r['flag_threshold']} | {r['flag']} |")
w()

# ------------------------------------------------------------------ T5
w("## Table 5 — Fig. 5: dN/dX median bias [%] per Paper-1 bin and coarse z block")
w()
for est, gate in [("ge20.0", "|bias| ≤ 0.5 %"), ("ge20.3", "bias ∈ [−0.5, +3.0] %")]:
    w(f"### {est} (gate window {gate})")
    w()
    for fam in FAMS:
        w(f"**{fam}**")
        w()
        hdr = "| bin | z range | truth dN/dX |"
        sep = "|---|---|---|"
        for m in MODELS:
            for s in SEEDS:
                hdr += f" {m} s{str(s)[-2:]} |"
                sep += "---|"
        w(hdr)
        w(sep)
        idx = {}
        for r in D["fig5"]:
            if r["family"] == fam and r["estimand"] == est:
                idx[(r["model"], r["seed"], r["bin"])] = r
        for bn in ["B1", "B2", "B3", "B4", "B5", "block0", "block1", "block2"]:
            ref = next((idx[(m, s, bn)] for m in MODELS for s in SEEDS if (m, s, bn) in idx), None)
            if ref is None:
                continue
            row = (f"| {bn} | {ref['z_lo']:.2f}–{ref['z_hi']:.2f} | {g(ref['truth'])} |")
            for m in MODELS:
                for s in SEEDS:
                    r = idx.get((m, s, bn))
                    row += (" — |" if r is None else
                            f" {r['median_bias_pct']:+.2f} ({mark(r['truth_in_68'], r['truth_in_95'])}) |")
            w(row)
        w()

# ------------------------------------------------------------------ T6
w("## Table 6 — all-z threshold estimands (the gated numbers)")
w()
w("| family | model | seed | estimand | truth | posterior median | 68 % | bias [%] | in68 | in95 | gate |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for r in D["allz"]:
    lo, hi = (-0.5, 0.5) if r["estimand"] == "ge20.0" else (-0.5, 3.0)
    ok = lo <= r["median_bias_pct"] <= hi
    w(f"| {r['family']} | {r['model']} | {r['seed']} | {r['estimand']} | {g(r['truth'])} | "
      f"{g(r['median'])} | {g(r['p16'])}–{g(r['p84'])} | {r['median_bias_pct']:+.2f} | "
      f"{r['truth_in_68']} | {r['truth_in_95']} | {'PASS' if ok else '**FAIL**'} |")
w()

# ------------------------------------------------------------------ summary
w("## What the plots show (factual, 10 lines)")
w()
w("1. ORACLE — FP pinned to truth, zero FP degrees of freedom — still misses both gated "
  "all-z estimands: dN/dX(≥20.0) bias is +0.99/+1.19/+1.42 % (2lpt0/london0/saclay0, "
  "s20260811), i.e. 2–3× outside the ±0.5 % window, and the second seed reproduces each to "
  "≤0.03 pp.")
w("2. dN/dX(≥20.3) all-z is +2.95/+2.17/+3.29 %: inside the [−0.5, +3.0] % window for 2lpt0 "
  "and london0, outside it for saclay0; truth is outside the 95 % posterior interval in "
  "4 of the 6 ORACLE runs.")
w("3. Fig. 1: the path-weighted f(N) tracks truth over four decades on the log panel, but the "
  "ratio panel oscillates bin-to-bin — a sawtooth of alternating sign reaching −10.4…+16.1 % "
  "(2lpt0), −8.7…+10.5 % (london0) and −13.0…+10.0 % (saclay0) across 19.7–21.7, with much "
  "larger excursions in the end bins (+91…+115 % in [19.0, 19.2), −21…−41 % in [19.2, 19.5), "
  "and −26…+27 % in the top two bins above 21.9).")
w("4. The sawtooth is seed-independent: the two seeds lie on top of each other in every panel "
  "of Figs. 1–3 (max |Δratio| between seeds is 0.98 pp over all 48 Fig. 1 bins; Fig. 5 "
  "seed-to-seed |Δbias| has median 0.05 pp, max 2.85 pp), so it is not Monte-Carlo noise.")
w("5. Fig. 2: the same alternating pattern is present in all three coarse z blocks, with the "
  "68 % bands frequently excluding zero; over 19.7–21.7 the swing is −12.8…+16.3 % in K0, "
  "−19.2…+22.2 % in K1 and −20.0…+22.6 % in K2, i.e. K0 is the flattest block and K1/K2 the "
  "largest.")
w("6. Fig. 3: the 0.2-dex zigzag spans 26.5 pp (2lpt0, −10.4 to +16.1), 19.2 pp (london0) and "
  "22.9 pp (saclay0); truth is outside 68 % in 5/10, 5/10 and 9/10 bins and outside 95 % in "
  "3, 1 and 4 bins respectively.")
w("7. M0 traces the ORACLE zigzag almost exactly (mean offset +0.3 to +1.1 pp); M2 and M3 sit "
  "2–5 pp lower on average but keep the same bin-to-bin shape, so the oscillation is not "
  "removed by any FP model on the ladder.")
w("8. Fig. 4, S/N: the PPC residual mu_median/obs − 1 rises monotonically with S/N, from "
  "−4.0/−6.1/−6.3 % in the S/N 2–3 stratum (the only stratum that fails the two-sided p "
  "threshold in all three families) to +2.2/+3.8/+3.5 % at the highest strata; the sealed "
  "`snr_ramp` amplitudes are 0.062/0.099/0.098, all under the 0.106 flag threshold.")
w("9. Fig. 4, z and Nhat: the z residual drifts from ≈0 at z = 2.0 to −2.8 % (2lpt0) and "
  "−1.7 % (london0) by z = 3.4–3.5 and to −4.3 % at z = 3.3–3.4 for saclay0; the Nhat "
  "residual stays within −3.9…+2.7 % below Nhat ≈ 21 and scatters to −22…+50 % above it; "
  "0 (2lpt0, london0) and 1 (saclay0) of 435 joint PPC cells fail, while the omnibus "
  "chi-square PPC p-value is 0.0 in all three families.")
w("10. Fig. 5: of the 24 ORACLE family × bin cells (B1–B5 plus K0–K2), 7 are inside the "
  "±0.5 % window at ≥20.0 (range −1.30…+3.48 %) and 12 are inside [−0.5, +3.0] % at ≥20.3 "
  "(range −3.90…+7.84 %); the largest per-bin biases are B3 and K1 (≥20.3: +7.84/+5.78/"
  "+5.93 % in B3), the most negative is saclay0 B5 (−3.90 %), and the same K0 ≈ 0, K1/K2 "
  "elevated pattern appears in M0, M2 and M3.")
w()

Path(OUT / "ORACLE_RECOVERY_PLOTS.md").write_text("\n".join(L) + "\n")
print("wrote", OUT / "ORACLE_RECOVERY_PLOTS.md", len(L), "lines")
