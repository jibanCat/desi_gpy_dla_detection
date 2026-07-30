# -*- coding: utf-8 -*-
"""run_evidence.py -- ONE command, ONE artifact: ``inference_evidence.json``.

The paper cites this file and nothing else for "is this posterior citable?".
It carries, in one stamped object:

  blocks.convergence   split-R-hat / ESS_bulk / ESS_tail PER REPORTED QUANTITY
                       and per latent, divergences + their location in
                       parameter space, tree-depth saturation, E-BFMI
  blocks.ppc           posterior predictive replicated counts vs the observed
                       counts, per (N, z) cell and in marginals, Bayesian
                       p-values, and the cells the model cannot reproduce
  blocks.closure       posterior median / pack truth with credible interval,
                       per reported quantity and per z-bin, z-scores, 68/95
                       coverage of the truth
  blocks.coverage_sbc  reduced-dimension simulation-based calibration: rank
                       histograms and their uniformity test
  blocks.ztilt         the manufactured redshift tilt and the verdict on
                       whether an integrated (z-marginalised) result is the
                       only defensible product
  gate                 FAIL-CLOSED. ``stampable`` is the AND of every check
                       AND of every required block being present and complete.

Three modes:

  --mode fit       run Model A on a pack, then all five blocks (full evidence)
  --mode artifact  read a saved run_rung9 JSON: everything the artifact can
                   still support, and an HONEST list of what it cannot
                   (this always lands NOT STAMPABLE, by design)
  --mode sbc       the SBC block alone (it is the expensive one)

MOCK ONLY: refuses any pack or artifact whose provenance mentions the real
survey.  Real-survey result values must never enter this repo.

Examples
--------
  conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.run_evidence \\
      --mode fit --pack <modelA_pack_*.npz> --out inference_evidence.json \\
      --warmup 1000 --samples 1000 --chains 4

  conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.run_evidence \\
      --mode artifact --result rung9_2lpt0_v2.json --pack <pack.npz> \\
      --out evidence_rung9_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from CDDF_analysis.hbi_mcmc import evidence as EV

_REAL_TOKENS = ("main_dark", "loa_main_dark", "matterhorn", "dr3")


def _refuse_real(obj, what):
    s = json.dumps(obj) if not isinstance(obj, str) else obj
    low = s.lower()
    for tok in _REAL_TOKENS:
        if tok in low:
            raise SystemExit(f"REAL-SURVEY GUARD: {what} mentions {tok!r}; "
                             f"this harness is MOCK ONLY")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _print_verdict(ev):
    g = ev["gate"]
    print("\n=== inference evidence ===")
    print(f"  stampable = {g['stampable']}   "
          f"paper_facing = {g['paper_facing']}   estimand = {g['estimand']}")
    if g.get("bypasses"):
        for k, v in sorted(g["bypasses"].items()):
            print(f"    BYPASS IN FORCE: {k} = {v!r} -> NOT paper-facing")
    print(f"  checks: {g['n_checks'] - g['n_failed']}/{g['n_checks']} pass")
    for r in g["reasons"]:
        print(f"    REFUSED: {r}")
    c = (ev["blocks"].get("convergence") or {}).get("summary")
    if c:
        print(f"  reported r_hat_max={c['reported_r_hat_max']:.5f} "
              f"({c['reported_r_hat_argmax']})  "
              f"ess_bulk_min={c['reported_ess_bulk_min']:.0f} "
              f"({c['reported_ess_bulk_argmin']})")
    p = ev["blocks"].get("ppc") or {}
    if "frac_cells_failed" in p:
        print(f"  ppc: {p['n_cells_failed']}/{p['n_cells']} cells "
              f"unreproducible; omnibus ppp="
              f"{p['omnibus_chi2_discrepancy']['posterior_predictive_p']:.4f}")
    cl = ev["blocks"].get("closure") or {}
    if "coverage95" in cl:
        print(f"  closure: cover68={cl['coverage68']:.2f} "
              f"cover95={cl['coverage95']:.2f} worst|z|={cl['worst_z']} "
              f"({cl['worst_quantity']})")
    zt = ev["blocks"].get("ztilt") or {}
    if "R0_span" in zt:
        print(f"  ztilt: R0 span={zt['R0_span']} slope={zt['R0_slope_per_unit_z']} "
              f"z-bins in95={zt['n_z_in95']}/{zt['n_z_bins']} "
              f"integrated_only={zt.get('integrated_only_defensible')}")
    sb = ev["blocks"].get("coverage_sbc") or {}
    if "worst_p_bonferroni" in sb:
        print(f"  sbc: worst uniformity p={sb['worst_p_value']:.4f} "
              f"(bonf {sb['worst_p_bonferroni']:.4f}) on {sb['worst_quantity']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fit", "artifact", "sbc"),
                    default="fit")
    ap.add_argument("--pack")
    ap.add_argument("--result", help="a saved run_rung9 JSON (--mode artifact)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resp-clamp", default="both")
    ap.add_argument("--allow-low-farr", metavar="REASON", default=None,
                    help="bypass the Farr headroom gate. RECORDED in the "
                         "artifact and forces paper_facing=False.")
    ap.add_argument("--allow-open-forward-model", metavar="REASON",
                    default=None,
                    help="bypass the forward-model closure gate. RECORDED in "
                         "the artifact and forces paper_facing=False.")
    ap.add_argument("--ppc-draws", type=int, default=300)
    ap.add_argument("--sbc-sims", type=int, default=48)
    ap.add_argument("--sbc-seed", type=int, default=0)
    ap.add_argument("--no-sbc", action="store_true",
                    help="skip the SBC block; the gate then REFUSES the stamp "
                         "(a missing block is a failure, never a pass)")
    ap.add_argument("--synthetic-grid", action="store_true",
                    help="smoke: build a small synthetic pack instead of "
                         "loading --pack")
    a = ap.parse_args(argv)

    t_start = time.time()
    blocks, prov = {}, {"date": time.strftime("%Y-%m-%d"), "mode": a.mode}
    # every gate-bypass flag actually in force, recorded in the artifact; any
    # one of them makes the artifact permanently non-paper-facing.
    bypasses = {}
    if a.allow_low_farr is not None:
        bypasses["allow_low_farr"] = a.allow_low_farr
    if a.allow_open_forward_model is not None:
        bypasses["allow_open_forward_model"] = a.allow_open_forward_model

    if a.mode == "sbc":
        from CDDF_analysis.hbi_mcmc import sbc as _sbc
        # NO run_config: --mode sbc has no run to be matched against, so the
        # block reports sbc_configuration_matches_run=False and the artifact is
        # not stampable.  That is the ratified behaviour (decision 8), not an
        # oversight -- an SBC standing alone certifies nothing.
        blocks["coverage_sbc"] = _sbc.sbc_block(a.sbc_sims, seed=a.sbc_seed)
        prov["rederive"] = (f"python -m CDDF_analysis.hbi_mcmc.run_evidence "
                            f"--mode sbc --sbc-sims {a.sbc_sims} "
                            f"--sbc-seed {a.sbc_seed} --out <out>")
        # NOTE: NO ``required=`` narrowing.  This used to pass
        # ``required=("coverage_sbc",)``, which made ``gate`` blind to the
        # four absent blocks: the call returned stampable=True /
        # paper_facing=True / n_checks=2, so anything this branch wrote would
        # have carried that verdict.  (No such file was located on disk when
        # this was audited on 2026-07-29; the defect is in the CODE PATH, and
        # that is what is claimed here.)  An SBC-only run is
        # a PARTIAL evidence set: reportable, never stampable.  (``gate`` now
        # also unions REQUIRED_BLOCKS in, so this is belt AND braces.)
        prov["partial_evidence"] = True
        prov["blocks_computed"] = ["coverage_sbc"]
        ev = EV.assemble_evidence(blocks, provenance=prov, bypasses=bypasses)
        with open(a.out, "w") as fh:
            json.dump(ev, fh, indent=1, default=_json_default)
        _print_verdict(ev)
        return ev

    # --- load / build the pack
    from CDDF_analysis.hbi_mcmc.pack import load_pack, synthetic_pack
    if a.synthetic_grid:
        grid = dict(
            nhat_edges=np.round(np.arange(19.9, 20.4 + 1e-9, 0.1), 10),
            zf_edges=np.round(np.arange(2.0, 2.4 + 1e-9, 0.1), 10),
            zc_edges=np.array([2.0, 2.2, 2.4]),
            snr_edges=np.array([0.0, 3.0, np.inf]),
            n_molly_cells=3)
        pack = synthetic_pack(a.seed, **grid, fp_frac=0.15,
                              t_true=np.array([0.2, -0.15]))
        prov["pack"] = "synthetic_pack (smoke)"
        prov["pack_grid"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in grid.items()}
    else:
        if not a.pack:
            raise SystemExit("--pack is required unless --synthetic-grid")
        _refuse_real(os.path.basename(a.pack), "pack filename")
        pack = load_pack(a.pack)
        _refuse_real(pack.provenance or {}, "pack provenance")
        prov["pack"] = os.path.basename(a.pack)
        prov["pack_provenance_commit"] = (pack.provenance or {}).get(
            "code_commit")

    from CDDF_analysis.hbi_mcmc.forward import build_consts
    consts = build_consts(pack, resp_clamp=a.resp_clamp,
                          allow_unclamped_response=(a.resp_clamp == "off"))

    # The configuration the SBC would have to MATCH in order to certify this
    # run (decision 8, ratified 2026-07-29).  Built for --mode fit, where the
    # ModelAConfig is in hand; left None for --mode artifact, because a saved
    # result artifact does not record the prior scales and an unverifiable
    # configuration must not be asserted (fail closed).
    run_config = None
    if a.mode == "artifact":
        if not a.result:
            raise SystemExit("--mode artifact needs --result")
        with open(a.result) as fh:
            result = json.load(fh)
        _refuse_real(result.get("provenance") or {}, "result provenance")
        run = EV.posterior_run_from_artifact(result, pack)
        prov["result"] = os.path.basename(a.result)
        prov["result_provenance"] = result.get("provenance")
        prov["saved_diagnostics"] = result.get("diagnostics")
        prov["rederive"] = (f"python -m CDDF_analysis.hbi_mcmc.run_evidence "
                            f"--mode artifact --result {a.result} "
                            f"--pack {a.pack} --out <out>")
    else:
        from CDDF_analysis.hbi_mcmc.model_a import ModelAConfig, run_model_a
        cfg = ModelAConfig(num_warmup=a.warmup, num_samples=a.samples,
                           num_chains=a.chains, seed=a.seed,
                           resp_clamp=a.resp_clamp,
                           enforce_farr_gate=(a.allow_low_farr is None))
        from CDDF_analysis.hbi_mcmc import sbc as _sbc_cfgmod
        run_config = _sbc_cfgmod.run_configuration(pack, cfg)
        mcmc, red = run_model_a(pack, cfg)
        run = EV.posterior_run_from_mcmc(mcmc, pack,
                                         max_tree_depth=cfg.max_tree_depth)
        prov["sampler"] = dict(warmup=a.warmup, samples=a.samples,
                               chains=a.chains, seed=a.seed,
                               resp_clamp=a.resp_clamp)
        prov["legacy_summarize"] = red.get("diagnostics")
        prov["farr_ratio"] = red.get("farr_ratio")
        prov["rederive"] = (
            f"python -m CDDF_analysis.hbi_mcmc.run_evidence --mode fit "
            f"--pack {a.pack} --warmup {a.warmup} --samples {a.samples} "
            f"--chains {a.chains} --seed {a.seed} --out <out>")

    blocks["convergence"] = EV.convergence_block(run, pack)
    blocks["ppc"] = EV.ppc_block(run, pack, consts, n_rep_draws=a.ppc_draws,
                                 seed=a.seed)
    if pack.truth_counts is not None:
        blocks["closure"] = EV.closure_block(run, pack)
        blocks["ztilt"] = EV.ztilt_block(run, pack, resp_clamp=a.resp_clamp)
    else:
        blocks["closure"] = {"incomplete": ["pack_has_no_truth_counts"],
                             "checks": {"closure_cover95_ok": False}}
        blocks["ztilt"] = {"incomplete": ["pack_has_no_truth_counts"],
                           "checks": {"ztilt_has_a_defensible_product": False}}
    if not a.no_sbc:
        from CDDF_analysis.hbi_mcmc import sbc as _sbc
        blocks["coverage_sbc"] = _sbc.sbc_block(
            a.sbc_sims, seed=a.sbc_seed, run_config=run_config)
        prov["sbc_configuration_match"] = (
            (blocks["coverage_sbc"].get("configuration_match") or {})
            .get("matched"))

    prov["wallclock_s"] = float(time.time() - t_start)
    ev = EV.assemble_evidence(blocks, provenance=prov, bypasses=bypasses)
    with open(a.out, "w") as fh:
        json.dump(ev, fh, indent=1, default=_json_default)
    print(f"[run_evidence] wrote {a.out}")
    _print_verdict(ev)
    return ev


if __name__ == "__main__":
    main()
