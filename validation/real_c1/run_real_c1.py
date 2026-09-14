#!/usr/bin/env python
"""run_real_c1.py — REAL-MODE production runner for the FROZEN Paper-1 low-z model.

Frozen configuration (PI ruling 2026-09-14 §16, governance/final_campaign_2026-09-13/
PAPER1_LOWZ_MODEL_FREEZE_2026-09-14.md): B response (Q_B x phi_2LPT-measured) +
C1nsadd completeness + M1CUT FP block (a0 = 1/K, Lambda ~ Gamma(89.5, ell_eff) with
J = 8 stratified cut imputations, t_K ~ N(0,1)) + the TRANSPORTED sub-floor-host
term, on the v3 support contract, NUTS target_accept 0.95, 4 x (1500 + 1000).

THIS IS THE SAME SCIENCE DRIVER AS ``validation/fp_ladder/run_ladder.py`` MINUS
THE MOCK-ONLY MACHINERY.  Specifically:
  * the model call, the fixed-object loading, the Lambda imputation and the
    MCMC/NUTS construction are COPIED VERBATIM from run_ladder.py (the copied
    blocks are marked "VERBATIM run_ladder.py lines NN-MM" below);
  * run_ladder's MOCK GATE (``pack.truth_counts.sum() > 0``) is INVERTED: this
    runner requires the all-zero ``ZEROS_NO_TRUTH`` real sentinel and fails
    closed on a pack that carries truth, unless the operator passes the
    explicit ``--allow-mock-for-identity-test`` flag (used only by
    tests/test_run_real_c1.py to prove the two drivers are bit-identical);
  * every truth-comparison read-out (``forward_selftest.truth_f``,
    ``perz_recovery``, the bias tables, the FP-truth census flag, the ORACLE
    member and the ``--fix`` / ``--ops`` diagnostics) is DELETED, not disabled.
    ``--fix``, ``--ops`` and ``--ladder ORACLE`` are refused outright: real mode
    has no truth to pin anything to.
  * the reductions are the committed ones, truth-free
    (``validation/real_c1/reduce_truthfree.py``).

PRIVACY.  The runner writes the JSON to ``--out`` and prints ONLY sampler health,
nuisance diagnostics and array shapes.  No estimand value is ever printed
(``--quiet-values`` defaults to True and CANNOT be switched off in real mode).
No truth field is written anywhere; a fail-closed scan of the output object
(``reduce_truthfree.assert_no_truth``) runs before the file is written.

Usage (the production command; see validation/real_c1/run_real_c1.sbatch):
  python validation/real_c1/run_real_c1.py --pack $I/C1_pack.npz --ladder M1CUT \
      --seed 20260811 --chains 4 --warmup 1500 --samples 1000 --target-accept 0.95 \
      --require-support --mg-fixed-file $I/Mg_B_real.npz --mg-fixed-key Mg \
      --c-fixed-file $I/C_C1nsadd_real.npz --extra-fixed-file $I/mu_extra_P6bcal_real.npz \
      --lam-imputations 8 --lam-imputation $j --stage REAL_C1_BLIND --out $OUT/....json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import numpy as np
import jax, jax.numpy as jnp
import numpyro

from CDDF_analysis.hbi_mcmc.provenance_util import run_config, sha256
from CDDF_analysis.hbi_mcmc.pack import load_pack
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
from CDDF_analysis.hbi_mcmc.fp_ladder import model_cc_ladder, FP_SITES, live_mask, NOMINAL_FP_DOF
from CDDF_analysis.hbi_mcmc.model_a import reduce_f_posterior

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reduce_truthfree as RT

POP_SITES = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z", "psi_c")
NHAT_GROUPS = (("19.5_20.0", 19.5, 20.0), ("20.0_20.3", 20.0, 20.3), ("20.3_22.4", 20.3, 22.4))

ROLE_REAL = ("REAL-MODE production run of the FROZEN Paper-1 low-z model "
             "(PI ruling 2026-09-14 §16/§17). No truth object exists or is read.")
ROLE_IDENT = ("IDENTITY TEST on a MOCK pack (--allow-mock-for-identity-test): "
              "proves this driver is bit-identical to validation/fp_ladder/run_ladder.py. "
              "NOT a science product and NOT a mock closure result.")


class _RefusedAction(argparse.Action):
    """A flag that exists only so its refusal is explicit and testable."""

    def __call__(self, parser, ns, values, option_string=None):
        raise SystemExit(
            f"REAL MODE: {option_string} is a MOCK-ONLY truth-pinning diagnostic "
            "(it reads the mock FP-truth census / the matched-truth operators). "
            "Real data has no truth side; refusing. The survey-facing sub-floor "
            "term is passed as --extra-fixed-file (the TRANSPORTED rate).")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--ladder", required=True,
                    help="the frozen member is M1CUT; ORACLE is REFUSED (mock-only diagnostic)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--t-sd", type=float, default=1.0)
    ap.add_argument("--tau-scale", type=float, default=0.5)
    ap.add_argument("--calib-weight", type=float, default=1.0)
    ap.add_argument("--stage", default="")
    ap.add_argument("--mg-fixed-file", default=None, help="npz with Mg (S,Kf,C,B) = the FIXED frozen response")
    ap.add_argument("--mg-fixed-key", default="Mg", help="key inside --mg-fixed-file; real mode allows 'Mg' only")
    ap.add_argument("--c-fixed-file", default=None, help="npz with C_fixed (S,B) or C_fixed_bks (B,Kf,S)")
    ap.add_argument("--extra-fixed-file", default=None, help="npz with mu_extra (C,Kf,S) = the TRANSPORTED sub-floor-host term")
    ap.add_argument("--lam-imputations", type=int, default=1)
    ap.add_argument("--lam-imputation", type=int, default=0)
    ap.add_argument("--fp-a0", type=float, default=None,
                    help="M1CUT Perks pseudo-count a0; DEFAULT None = the record 1/K. "
                         "Passing a value is the a0 sensitivity battery, not the run of record.")
    ap.add_argument("--require-support", action="store_true",
                    help="fail closed unless the pack carries a stamped, self-consistent "
                         "row-level support (support_contract.ROW_SELECTION_FIELDS)")
    ap.add_argument("--allow-mock-for-identity-test", action="store_true",
                    help="TESTING ONLY: accept a pack that carries truth_counts, so the "
                         "driver can be compared bit-for-bit against run_ladder.py. "
                         "Never used for a science run; stamped into the JSON.")
    ap.add_argument("--skip-contract-guards", action="store_true",
                    help="TESTING ONLY: skip the contract-guard subprocess (smoke tests)")
    ap.add_argument("--quiet-values", action=argparse.BooleanOptionalAction, default=True,
                    help="never print estimand values to stdout (default True; the "
                         "--no- form is honoured ONLY under --allow-mock-for-identity-test)")
    # --- refused outright: the mock-only truth-pinning diagnostics -----------
    ap.add_argument("--fix", action=_RefusedAction, nargs="?", default=None,
                    help="REFUSED in real mode (mock-only truth pinning)")
    ap.add_argument("--ops", action=_RefusedAction, nargs="?", default=None,
                    help="REFUSED in real mode (matched-truth operators)")
    ap.add_argument("--census", action=_RefusedAction, nargs="?", default=None,
                    help="REFUSED in real mode (mock FP-truth census)")
    ap.add_argument("--out", required=True)
    return ap


def contract_guards(pack_path):
    """The contract-guard subprocess, with cc_real_posterior.main's rule.

    G_A_partition's truth-point form is UNDEFINED on a truth-less real pack (the
    all-zero sentinel makes it FAIL); that is the fail-closed behaviour working,
    and its REAL-mode semantics are carried by the predictive level ratio in
    ``predictive_marginals``.  Every OTHER guard must hard-pass.
    """
    r = subprocess.run([sys.executable, "-m",
                        "CDDF_analysis.hbi_mcmc.contract_guards_check",
                        "--pack", pack_path], capture_output=True, text=True)
    try:
        greport = json.loads(r.stdout[r.stdout.index("{"):])
    except Exception:
        raise SystemExit(f"guards output unparseable:\n{r.stdout}\n{r.stderr}")
    other_fail = [k for k, v in greport.items()
                  if isinstance(v, dict) and v.get("status") == "FAIL"
                  and k != "G_A_partition"]
    if other_fail:
        raise SystemExit(f"contract guards FAILED (non-G_A): {other_fail}\n{r.stdout}")
    return {k: (v.get("status") if isinstance(v, dict) else v)
            for k, v in greport.items()}


def _stamp(path):
    """path@sha8 — the fixed-object stamp (run_ladder records the bare path;
    the blind real run records the content hash as well)."""
    if not path:
        return None
    return f"{os.path.abspath(path)}@{sha256(path)[:8]}"


def main(argv=None):
    a = build_parser().parse_args(argv)

    # ---- what real mode categorically refuses ------------------------------
    if a.ladder == "ORACLE":
        raise SystemExit("REAL MODE: --ladder ORACLE pins mu_FP to the MOCK FP-truth "
                         "census. Real data has no such object; refusing.")
    if a.ladder not in FP_SITES:
        raise SystemExit(f"unknown ladder member {a.ladder!r}")
    if a.mg_fixed_key != "Mg":
        raise SystemExit("REAL MODE: --mg-fixed-key must be 'Mg'. 'Mg_phi_family' is the "
                         "ORACLE response diagnostic (a mock family's own measured phi); "
                         "refusing.")
    if not a.quiet_values and not a.allow_mock_for_identity_test:
        raise SystemExit("REAL MODE: --no-quiet-values is only honoured under "
                         "--allow-mock-for-identity-test; refusing to print real values.")
    numpyro.set_host_device_count(a.chains)

    # ---- the support gate (VERBATIM run_ladder.py lines 65-71, row level) ---
    support_record = None
    if a.require_support:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "absorber_ladder", "support"))
        import support_contract as SC
        # REAL_C1_READINESS.md §2: check_support_consistency(pack, None, None) over the
        # row-selection subset. There is no census and no ops object on real data.
        support_record = SC.check_support_consistency(
            a.pack, None, None, fields=SC.ROW_SELECTION_FIELDS)   # raises on mismatch
        print("SUPPORT GATE PASS (row level):", support_record["support_id_short"],
              "host floors:", support_record["truth_host_floor"], flush=True)

    pk = load_pack(a.pack)

    # ---- THE REAL GATE — run_ladder's MOCK GATE, INVERTED ------------------
    # run_ladder.py lines 73-75:
    #     if tc.size == 0 or tc.sum() <= 0: raise SystemExit("MOCK GATE: ...")
    # Real mode requires exactly the refused condition: the all-zero
    # ZEROS_NO_TRUTH sentinel that extract_pack_real.py stamps.
    tc = np.asarray(pk.truth_counts)
    is_real_sentinel = bool(tc.size > 0 and not np.any(tc != 0))
    real_gate = None
    if not is_real_sentinel:
        if not a.allow_mock_for_identity_test:
            raise SystemExit(
                "REAL GATE: pack carries a NONZERO truth_counts plane — this is a MOCK "
                "pack. The real-mode runner refuses it (use validation/fp_ladder/"
                "run_ladder.py for mocks, or --allow-mock-for-identity-test for the "
                "bit-identity test).")
        print("IDENTITY-TEST MODE: a MOCK pack was accepted under "
              "--allow-mock-for-identity-test; this run is NOT a science product.",
              flush=True)
    else:
        from CDDF_analysis.hbi_mcmc.cc_real_posterior import _real_mode_gate
        prov = _real_mode_gate(a.pack, pk)       # provenance real_data + sentinel + all-zero
        real_gate = dict(real_data=bool(prov.get("real_data")),
                         truth_counts_sentinel=str(prov.get("truth_counts_sentinel")),
                         truth_counts_all_zero=True,
                         truth_counts_shape=[int(x) for x in tc.shape])
    if is_real_sentinel and a.allow_mock_for_identity_test:
        raise SystemExit("--allow-mock-for-identity-test was passed with a REAL pack; "
                         "refusing (the flag exists only to accept a mock).")

    # ---- the contract guards (real mode only; cc_real_posterior's subprocess) ----
    guards = None
    if is_real_sentinel and not a.skip_contract_guards:
        guards = contract_guards(a.pack)
        print("CONTRACT GUARDS:", json.dumps(guards), flush=True)

    # ======================================================================
    # VERBATIM run_ladder.py lines 76-113 (the ORACLE / --fix / --ops branches
    # are ABSENT by construction — they are refused above).
    # ======================================================================
    consts, Mg = build_cc_tensors(pk)
    counts = jnp.asarray(np.asarray(pk.counts, float))
    fpc = jnp.asarray(np.asarray(pk.fp_counts, float))

    mu_extra = C_fixed = Mg_fixed = None
    lam_fixed = None
    if a.ladder == "M1CUT":
        from CDDF_analysis.hbi_mcmc.fp_ladder import lambda_calibration_posterior_quantiles
        qs = lambda_calibration_posterior_quantiles(pk.fp_counts, consts.fp_ell_eff, a.lam_imputations)
        lam_fixed = float(qs[a.lam_imputation])
    if a.mg_fixed_file:
        _mgf = np.load(a.mg_fixed_file, allow_pickle=True)
        Mg_fixed = np.asarray(_mgf[a.mg_fixed_key], float)
    if a.c_fixed_file:
        cf = np.load(a.c_fixed_file, allow_pickle=True)
        C_fixed = np.asarray(cf["C_fixed_bks"], float) if "C_fixed_bks" in cf.files else np.asarray(cf["C_fixed"], float)
    if a.extra_fixed_file:
        ef = np.load(a.extra_fixed_file, allow_pickle=True)
        key = "mu_extra" if "mu_extra" in ef.files else "mu_P6b_cks"
        mu_extra = (mu_extra if mu_extra is not None else 0.0) + np.asarray(ef[key], float)   # P6b-cal (transported rate)
    kz_np = np.asarray(consts.kz_to_K); KK = consts.n_kk

    # ======================================================================
    # VERBATIM run_ladder.py lines 139-151 — the sampler.  mu_fp_fixed and
    # E_fixed are passed as None (their only producers were the ORACLE /
    # --fix diagnostics), so the model call is argument-for-argument the same.
    # ======================================================================
    from numpyro.infer import MCMC, NUTS
    kern = NUTS(model_cc_ladder, target_accept_prob=a.target_accept)
    mcmc = MCMC(kern, num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(a.seed), consts, Mg, counts=counts, fp_counts=fpc,
             ladder=a.ladder, t_sd=a.t_sd, tau_scale=a.tau_scale, calib_weight=a.calib_weight,
             mu_fp_fixed=None, mu_extra_fixed=mu_extra, C_fixed=C_fixed, Mg_fixed=Mg_fixed, E_fixed=None,
             lam_fixed=lam_fixed, fp_a0=a.fp_a0,
             extra_fields=("potential_energy", "energy", "diverging"))
    sam = mcmc.get_samples(group_by_chain=False)
    sam_g = mcmc.get_samples(group_by_chain=True)
    xf_g = mcmc.get_extra_fields(group_by_chain=True)
    f_draws = np.asarray(sam["f"])

    # ======================================================================
    # THE REDUCTION — the committed estimands, truth-free
    # ======================================================================
    ntrue = np.asarray(pk.ntrue_edges, float)
    dX_k = np.asarray(pk.dX, float).sum(axis=1)
    red = reduce_f_posterior(f_draws, pk)
    estimands = dict(
        estimand="POSTERIOR_MEDIAN_CI",
        thresholds_allz=RT.thresholds_allz(f_draws, pk, red=red),
        reporting_bins_0p2dex=RT.reporting_bins_0p2dex(f_draws, pk),
        perz_posterior=RT.perz_posterior(f_draws, pk),
        omega_20p3_21p6_allz=RT.omega_20p3_21p6_allz(
            f_draws, ntrue, np.asarray(pk.zf_edges, float), dX_k),
    )

    # ======================================================================
    # SAMPLER HEALTH (run_ladder's diagnostics, plus rank-normalised R-hat and
    # bulk/tail ESS on the two headline estimands and on t_K)
    # ======================================================================
    div_g = np.asarray(xf_g["diverging"]); pe_g = np.asarray(xf_g["potential_energy"], float)
    en_g = np.asarray(xf_g["energy"], float)
    ebfmi = [float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2)) for e in en_g]
    fg = np.asarray(sam_g["f"])
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    try:
        import arviz as _az
    except Exception:                                        # pragma: no cover
        _az = None

    def _rank_stats(cs):
        """cs: (chains, draws). Rank-normalised split-Rhat + bulk/tail ESS."""
        cs = np.asarray(cs, float)
        if _az is None or cs.shape[0] < 2:                   # pragma: no cover
            return dict(rank_split_rhat=None, ess_bulk=None, ess_tail=None,
                        note="arviz unavailable or single chain")
        return dict(rank_split_rhat=round(float(_az.rhat(cs, method="rank")), 5),
                    ess_bulk=round(float(_az.ess(cs, method="bulk")), 1),
                    ess_tail=round(float(_az.ess(cs, method="tail")), 1))

    mixing = {}
    for key in ("dndx_dla_20p0_allz", "dndx_dla_20p3_allz"):
        cs = np.stack([np.asarray(reduce_f_posterior(fg[ci], pk)[key]) for ci in range(fg.shape[0])])
        W = cs.var(axis=1, ddof=1).mean(); Bv = cs.mean(axis=1).var(ddof=1) * cs.shape[1]
        rh = float(np.sqrt(((cs.shape[1] - 1) / cs.shape[1] * W + Bv / cs.shape[1]) / W)) if cs.shape[0] > 1 else None
        m = dict(split_rhat=(round(rh, 4) if rh else None),
                 split_rhat_numpyro=round(float(split_gelman_rubin(cs)), 4),
                 ess=round(float(effective_sample_size(cs)), 1))
        m.update(_rank_stats(cs))
        mixing[key] = m                       # NOTE: no per-chain medians (values)
    t_g = np.asarray(sam_g["t"]) if "t" in sam_g else None
    t_draws = np.asarray(sam["t"]) if "t" in sam else np.zeros((f_draws.shape[0], consts.n_kk))
    t_mixing = ([_rank_stats(t_g[:, :, K]) for K in range(t_g.shape[2])]
                if t_g is not None else None)
    sampler_health = dict(
        chains=int(a.chains), warmup=int(a.warmup), samples=int(a.samples),
        target_accept=float(a.target_accept),
        divergences=int(div_g.sum()), divergences_per_chain=[int(x) for x in div_g.sum(axis=1)],
        ebfmi_per_chain=[round(x, 4) for x in ebfmi],
        mean_potential_energy_per_chain=[round(float(e.mean()), 1) for e in pe_g],
        estimand_mixing=mixing, t_mixing_per_K=t_mixing,
        nominal_fp_dof=NOMINAL_FP_DOF[a.ladder])
    t_posterior = dict(
        mean=[float(x) for x in t_draws.mean(axis=0)],
        sd=[float(x) for x in t_draws.std(axis=0)],
        per_chain_median=([[float(np.median(t_g[ci, :, K])) for K in range(t_g.shape[2])]
                           for ci in range(t_g.shape[0])] if t_g is not None else None),
        in_record_prior_sd=[float(x) for x in (t_draws.mean(axis=0) / np.asarray(consts.t_sigma))],
        e_t_median=[float(x) for x in np.exp(np.median(t_draws, axis=0))])

    # ======================================================================
    # FP TOTALS — VERBATIM run_ladder.py lines 202-221 (fp_by_block) with the
    # fp_truth census comparison removed.
    # ======================================================================
    lam_draws = np.asarray(sam["lam_fp"])                       # (D, C, S)
    naive = float(np.asarray(pk.fp_counts, float).sum() / consts.fp_ell_eff)
    kz = np.asarray(consts.kz_to_K); E = np.asarray(consts.fp_E, float)
    EK = np.stack([E[kz == K].sum(axis=0) for K in range(consts.n_kk)])          # (KK, S)
    pref = float(consts.fp_w * consts.fp_ell_eff) * (1.0 - np.asarray(consts.fp_eta_c))  # (C,)
    mu_cKs = pref[None, :, None, None] * np.exp(t_draws)[:, None, :, None] * lam_draws[:, :, None, :] * EK[None, None, :, :]
    mu_cK = mu_cKs.sum(axis=3)                                   # (D, C, KK)
    nh = np.asarray(pk.nhat_edges, float); cc = 0.5 * (nh[:-1] + nh[1:])
    obs_cK = np.stack([np.asarray(pk.counts, float)[:, kz == K, :].sum(axis=(1, 2)) for K in range(consts.n_kk)], axis=1)
    fp_totals = dict(
        mu_fp_total_p16_50_84=[float(x) for x in np.percentile(mu_cK.sum(axis=(1, 2)), [16, 50, 84])],
        mu_fp_block_p16_50_84=[[float(x) for x in np.percentile(mu_cK[:, :, K].sum(axis=1), [16, 50, 84])]
                               for K in range(consts.n_kk)],
        mu_fp_nhat_group_p16_50_84={name: [float(x) for x in np.percentile(mu_cK[:, (cc >= lo) & (cc < hi), :].sum(axis=(1, 2)), [16, 50, 84])]
                                    for name, lo, hi in NHAT_GROUPS},
        mu_fp_cK_median=np.median(mu_cK, axis=0).tolist(),
        counts_block=[float(obs_cK[:, K].sum()) for K in range(consts.n_kk)],
        counts_nhat_group={name: float(obs_cK[(cc >= lo) & (cc < hi), :].sum()) for name, lo, hi in NHAT_GROUPS},
        fp_frac_by_c_median=[float(x) for x in np.median(mu_cK.sum(axis=2), axis=0) / np.maximum(obs_cK.sum(axis=1), 1)],
        fp_lam_total_over_naive=[round(float(x), 4) for x in np.percentile(lam_draws.sum(axis=(1, 2)) / naive, [16, 50, 84])],
        note=("posterior mu_FP = w * ell_eff * (1-eta_c) * e^{t_K} * lam * E, summed over "
              "fine z within each coarse block; expected COUNTS. VERBATIM "
              "run_ladder.fp_by_block minus its fp_truth census comparison. None of "
              "this is printed to stdout."))

    # ======================================================================
    # PREDICTIVE MARGINALS — VERBATIM run_ladder.py lines 254-282.  These are
    # RATIOS mu/obs (allowed by the blinding rule), never counts.
    # ======================================================================
    idx_med = int(np.argsort(np.asarray(sam["theta_level"]))[len(sam["theta_level"]) // 2])
    th_med = jnp.asarray(np.asarray(sam["theta_pop"])[idx_med]); pc_med = jnp.asarray(np.asarray(sam["psi_c"])[idx_med])
    t_med = jnp.asarray(t_draws[idx_med]); lf_med = jnp.asarray(lam_draws[idx_med])
    f_med = jnp.exp(th_med)
    Mg_use = Mg if Mg_fixed is None else jnp.asarray(Mg_fixed)
    if C_fixed is None:
        Cc = jax.nn.sigmoid(consts.eta_hat + pc_med)[:, consts.b_to_cell]
        tpx = jnp.einsum("skcb,sb,bk->cks", Mg_use, Cc, consts.g_bk * f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
    elif np.asarray(C_fixed).ndim == 2:
        tpx = jnp.einsum("skcb,sb,bk->cks", Mg_use, jnp.asarray(C_fixed), consts.g_bk * f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
    else:
        tpx = jnp.einsum("skcb,bks,bk->cks", Mg_use, jnp.asarray(C_fixed), f_med * consts.dN_b[:, None]) * consts.dX[None, :, :]
    fpx = (consts.fp_w * consts.fp_ell_eff * (1.0 - consts.fp_eta_c)[:, None, None]
           * jnp.exp(t_med[consts.kz_to_K])[None, :, None] * lf_med[:, None, :] * consts.fp_E[None, :, :])
    if mu_extra is not None:
        fpx = fpx + jnp.asarray(mu_extra)
    obs3 = np.asarray(pk.counts, float); mu3 = np.asarray(tpx) + np.asarray(fpx)
    tot_obs = float(obs3.sum())

    def _marg(ax):
        o = obs3.sum(axis=ax); m = mu3.sum(axis=ax); return [float(x) for x in (m / np.maximum(o, 1.0))]

    pred_marg = dict(mu_over_obs_by_nhat=_marg((1, 2)), mu_over_obs_by_z=_marg((0, 2)), mu_over_obs_by_snr=_marg((0, 1)),
                     mu_over_obs_by_nhat_K=[[float(x) for x in (mu3[:, kz_np == K, :].sum((1, 2)) / np.maximum(obs3[:, kz_np == K, :].sum((1, 2)), 1.0))] for K in range(KK)],
                     tp_over_obs_by_nhat=[float(x) for x in (np.asarray(tpx).sum((1, 2)) / np.maximum(obs3.sum((1, 2)), 1.0))],
                     predictive_total_ratio=round(float(mu3.sum() / tot_obs), 4),
                     predictive_fp_share=round(float(np.asarray(fpx).sum() / mu3.sum()), 4),
                     note="posterior-median draw (by theta_level); RATIOS only, no counts")

    # ======================================================================
    # OUTPUT
    # ======================================================================
    out = dict(
        pack=os.path.abspath(a.pack),
        pack_sha256=sha256(a.pack),
        ladder=a.ladder, stage=a.stage,
        mode=("REAL" if is_real_sentinel else "MOCK_IDENTITY_TEST"),
        identity_test=bool(a.allow_mock_for_identity_test),
        n_draws=int(f_draws.shape[0]), chains=a.chains, warmup=a.warmup, samples=a.samples,
        shapes=dict(f=[int(x) for x in f_draws.shape], counts=[int(x) for x in obs3.shape],
                    Mg=[int(x) for x in np.asarray(Mg).shape],
                    n_b=int(consts.n_b), n_k=int(consts.n_k), n_c=int(consts.n_c),
                    n_s=int(consts.n_s), n_kk=int(consts.n_kk),
                    n_live_strata=int(live_mask(consts).sum())),
        run_config=run_config(a),
        support_gate=support_record,
        real_gate=real_gate,
        contract_guards=guards,
        fixed_files=dict(mg=_stamp(a.mg_fixed_file), c=_stamp(a.c_fixed_file),
                         extra=_stamp(a.extra_fixed_file), mg_key=a.mg_fixed_key),
        lam_cut=(dict(J=a.lam_imputations, j=a.lam_imputation, lam_fixed=lam_fixed,
                      fp_a0=a.fp_a0) if a.ladder == "M1CUT" else None),
        model=dict(t_sd=a.t_sd, tau_scale=a.tau_scale, calib_weight=a.calib_weight,
                   sigma_N_post=[float(x) for x in np.percentile(sam["sigma_N"], [16, 50, 84])],
                   sigma_z_post=[float(x) for x in np.percentile(sam["sigma_z"], [16, 50, 84])]),
        sampler_health=sampler_health,
        t_posterior=t_posterior,
        fp_totals=fp_totals,
        estimands=estimands,
        predictive_marginals=pred_marg,
        role=(ROLE_REAL if is_real_sentinel else ROLE_IDENT))

    RT.assert_no_truth(out)                     # fail-closed BEFORE anything is written
    json.dump(out, open(a.out, "w"), indent=1)
    base = a.out[:-5] if a.out.endswith(".json") else a.out
    np.savez(base + "_fdraws.npz", f=f_draws, ntrue_edges=ntrue,
             zf_edges=np.asarray(pk.zf_edges), dX_k=dX_k)
    keep = {k: np.asarray(sam_g[k]) for k in POP_SITES + tuple(FP_SITES[a.ladder]) + ("lam_fp", "t") if k in sam_g}
    np.savez(base + "_bychain.npz", seed=a.seed, chains=a.chains, warmup=a.warmup, samples=a.samples,
             ladder=a.ladder, potential_energy=pe_g, energy=en_g, diverging=div_g, **keep)

    # ---- stdout: sampler health + shapes ONLY ------------------------------
    print(json.dumps(dict(mode=out["mode"], ladder=a.ladder, out=a.out,
                          shapes=out["shapes"], lam_cut=out["lam_cut"],
                          fixed_files=out["fixed_files"]), indent=1, default=str))
    print(json.dumps(dict(sampler_health=sampler_health,
                          t_posterior={k: t_posterior[k] for k in ("mean", "sd", "in_record_prior_sd")}),
                     indent=1, default=str))
    if not a.quiet_values:                      # identity test only (guarded above)
        print(json.dumps(dict(estimands=estimands["thresholds_allz"]), indent=1, default=str))
    else:
        print("VALUES WITHHELD (--quiet-values): the estimands are in", a.out)
    return out


if __name__ == "__main__":
    main()
