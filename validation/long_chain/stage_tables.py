"""VALIDATION-ONLY: render a stage's sealed-criteria evaluation as markdown.

Usage: stage_tables.py <STAGE_x_DIAGNOSTICS.json> <out.md> [<previous_stage.json>]
The optional third argument enables the predeclared stability test (sec.4) between the two
stages, in units of the CURRENT C1 POOL half-width.
"""
import json
import sys

HEAD = ["dndx_20p3_allz", "dndx_20p0_allz", "omega_20p3_21p6_allz"]
C1_REDUCED = ("/nfs/turbo/lsa-cavestru/mfho/paper1_science_handoff/"
              "LOWZ_CLEAN_C1_DURABLE_CANDIDATE_2026-09-11/reductions/"
              "reduced_CLEAN_C1_pooled.json")
STAB_TOL = 0.05


def hw_frozen():
    q = json.load(open(C1_REDUCED))["quantities"]
    return {k: 0.5 * (v[3] - v[1]) for k, v in q.items()}


def main():
    src, out = sys.argv[1], sys.argv[2]
    prev = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    J = json.load(open(src))
    L = J["criteria_literals"]
    hw = hw_frozen()
    R = J["runs"]
    seeds = sorted(R)
    W = []
    a = W.append
    a(f"# STAGE {list(R.values())[0]['stage']} — sealed-criteria evaluation\n")
    a("**VALIDATION-ONLY / CANDIDATE.** Criteria literals are the sealed ones and are not "
      "adjustable: `rhat_true <= %.2f`, `ess_bulk >= %.0f`, `ess_tail >= %.0f`, "
      "`divergences == 0`, `E-BFMI >= %.1f` on every chain, evaluated on every reported scalar "
      "AND every sampled nuisance component.\n" % (L["rhat_true_max"], L["ess_bulk_min"],
                                                   L["ess_tail_min"], L["ebfmi_min"]))
    r0 = R[seeds[0]]
    a(f"Design: {r0['chains']} chains x {r0['samples']} retained draws, warmup {r0['warmup']}, "
      f"{r0['n_draws']} draws/run, {r0['criteria']['n_quantities']} quantities gated per run.\n")
    a("## Per-seed verdict\n")
    a("| seed | class | max R&#770;_true | worst R&#770; quantity | min ESS-bulk | worst | "
      "min ESS-tail | div | min E-BFMI | ΔPE (nats) | PASS |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        r, c = R[s], R[s]["criteria"]
        a(f"| {s} | **{r['classification']}** | {c['max_rhat_true']:.4f} | "
          f"`{c['max_rhat_true_quantity']}` | {c['min_ess_bulk']:.0f} | "
          f"`{c['min_ess_bulk_quantity']}` | {c['min_ess_tail']:.0f} | {r['divergences']} | "
          f"{r['ebfmi_min']:.4f} | {r['mode']['pe_gap']:.1f} | {'YES' if r['PASS'] else 'no'} |")
    a("\n## Criterion-by-criterion (which criterion fails, per seed)\n")
    a("| seed | R&#770;<=1.01 | ESS-bulk>=400 | ESS-tail>=400 | 0 divergences | E-BFMI>=0.3 |")
    a("|---|---|---|---|---|---|")
    for s in seeds:
        c = R[s]["criteria"]
        f = lambda k: "PASS" if c[k] else "**FAIL**"          # noqa: E731
        a(f"| {s} | {f('pass_rhat')} | {f('pass_ess_bulk')} | {f('pass_ess_tail')} | "
          f"{f('pass_divergences')} | {f('pass_ebfmi')} |")
    a("\n## Mode diagnostics (per-chain potential energy)\n")
    a("| seed | chain mean PE | ΔPE | lower-cluster chains | upper | crossings (blocks of 100) | "
      "genuine between-mode mixing |")
    a("|---|---|---|---|---|---|---|")
    for s in seeds:
        m = R[s]["mode"]
        a(f"| {s} | {m['pe_chain_means']} | {m['pe_gap']:.1f} | {m['lower_cluster_chains']} | "
          f"{m['upper_cluster_chains']} | {m['crossings_block100_per_chain']} | "
          f"{'**YES**' if m['genuine_between_mode_mixing'] else 'no'} |")
    a("\n## Nuisance block — the worst-mixing sampled site per seed\n")
    a("| seed | worst nuisance R&#770;_true | site | `t` per-chain medians |")
    a("|---|---|---|---|")
    for s in seeds:
        nd = R[s]["nuisance"]
        k = max(nd, key=lambda x: nd[x]["rhat_true"])
        tm = R[s]["t_per_chain_medians"]
        a(f"| {s} | {nd[k]['rhat_true']:.4f} | `{k}` | " +
          "; ".join(f"{n}={[round(v, 3) for v in vv]}" for n, vv in tm.items()) + " |")
    a("\n## Headline quantities per seed (q16 / q50 / q84)\n")
    a("| seed | " + " | ".join(HEAD) + " |")
    a("|---|" + "---|" * len(HEAD))
    for s in seeds:
        cells = []
        for h in HEAD:
            v = R[s]["headline_q16_q50_q84"][h]
            sc = 1e4 if h.startswith("omega") else 1.0
            cells.append(f"{v[1]*sc:.6g} [{v[0]*sc:.6g}, {v[2]*sc:.6g}]")
        a(f"| {s} | " + " | ".join(cells) + " |")
    a("\n(Ω is shown x10^4.)\n")
    if prev:
        a(f"\n## Predeclared stability test vs the previous stage "
          f"(|Δ| < {STAB_TOL} x the CURRENT C1 POOL half-width, on q16/q50/q84)\n")
        a("| seed | quantity | Δq16 (hw) | Δq50 (hw) | Δq84 (hw) | stable |")
        a("|---|---|---|---|---|---|")
        for s in seeds:
            if s not in prev["runs"]:
                continue
            for h in HEAD:
                n = R[s]["headline_q16_q50_q84"][h]
                o = prev["runs"][s]["headline_q16_q50_q84"][h]
                d = [(n[i] - o[i]) / hw[h] for i in range(3)]
                ok = max(abs(x) for x in d) < STAB_TOL
                a(f"| {s} | `{h}` | {d[0]:+.3f} | {d[1]:+.3f} | {d[2]:+.3f} | "
                  f"{'yes' if ok else '**no**'} |")
    a("\n## Roll-up\n")
    a(f"* PASS seeds: {J['PASS_seeds'] or 'none'}")
    a(f"* MULTIMODAL-DISCONNECTED seeds: {J['MULTIMODAL_DISCONNECTED_seeds'] or 'none'}")
    a(f"* MULTIMODAL-MIXING seeds (PI ruling §4 evidence, reported, not acted on): "
      f"{J['MULTIMODAL_MIXING_seeds'] or 'none'}")
    open(out, "w").write("\n".join(W) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()
