#!/usr/bin/env python
"""HZ2 FIDUCIAL GENERATIVE CLOSURE pack (gate MAX4_HZ2_HBI_CLOSURE_GATE_2026-09-03.md §2): an injection-grounded, forward-generated, truth-known
high-z population whose catalogue is produced under the SAME observation model HZ2 uses on real data (real-spectrum A_shared completeness + Candidate E).

Construction: f_syn(N) ∝ N^-beta on the latent basis (19.0–22.4), amplitude set so that the expected TP rows ≈ n_tp_target (NOT from the real P0 diagnostic value), spread over the
real population's sightline windows (dX per fine z bin × S/N row) — the number of systems per (latent bin b, fine z bin k, S/N row s) is Poisson with mean
f_syn(b) ΔN_b dX[k, s]. Each system takes the OUTCOME (detected; x̂) of a randomly chosen real-spectrum injection event of the same stratum, same emulated z block
and nearest design point in N (|ΔN| ≤ 0.15 dex; latent mass below 19.5 uses the 19.5-point events); detected x̂ inside the observed grid become counts. FP rows are
Poisson from the FP block expectation (fp_w × fp_counts, the HZ1 real-pack convention). Calibration blocks (molly, g, FP) and the Candidate-E override are the
real-arm ones (identical to the HZ2 real pack). Writes modelA_pack_HZ2_fidclosure_s<seed>.npz + provenance.
"""
import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.abspath(os.path.join(_HERE, "..", "..")); sys.path.insert(0, _REPO)
from CDDF_analysis.hbi_mcmc.extract_pack_hz import (NHAT_EDGES, ZF_EDGES, ZC_EDGES, SNR_EDGES, RESP_SNR_EDGES, T_SIGMA_FLOOR, ROOT_MAX4, ROOT_R041,  # noqa: E402
                                                    basis_pad_edges_19p0_0p2, dX_from_windows, molly_blocks, fp_block, _idx, _sha, stratum_of_row)
REPAIR = os.environ.get("REPAIR_REPO", "/home/mfho/wt_highz_repair"); sys.path.insert(0, os.path.join(REPAIR, "tools"))
from r041_response_population_study import load_events  # noqa: E402
from r041_response_estimator import build_E, TB  # noqa: E402

STRATA = [2.0, 3.0, 4.0, 5.0, 7.0, np.inf]
DESIGN = np.array([19.5, 19.75, 20.0, 20.15, 20.3, 20.4, 20.5, 20.65, 20.8, 21.0, 21.25, 21.5, 22.0])


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--beta", type=float, default=1.7); ap.add_argument("--n-tp-target", type=float, default=1500.0)
    ap.add_argument("--generator", choices=["events", "model"], default="events",
                    help="events: outcomes resampled from real-spectrum injection events (nearest design point; frozen gate §2 as written); "
                         "model: outcomes drawn from the deployed observation model itself (count_conserving_fold.outcome_probabilities; gate Amendment 1)")
    ap.add_argument("--tag", default="", help="output-name tag, e.g. 'model' -> modelA_pack_HZ2_fidclosure_model_s<seed>.npz")
    ap.add_argument("--shape", choices=["powerlaw", "A", "B"], default="powerlaw",
                    help="truth shape (replacement gate MAX4_HZ2_HBI_REPLACEMENT_CLOSURE_GATE_2026-09-03.md §1): 'A' = frozen Paper-1 low-z pooled "
                         "posterior-median f(N) shape on the identical latent basis (pad bins continued flat); 'B' = single power law N^-beta "
                         "(identical to 'powerlaw', kept as an explicit name)")
    a = ap.parse_args(argv); os.makedirs(a.out_dir, exist_ok=True); rng = np.random.default_rng(a.seed)
    pop_csv = f"{ROOT_R041}/population/r041_population.csv"; pop = list(csv.DictReader(open(pop_csv)))
    dX, xc, n_sl = dX_from_windows(pop, lambda r: float(r["snr"]))                                   # (Kf, S)
    fid_json = f"{ROOT_MAX4}/fid_max4/analysis/analysis_fid_MAX4.json"; fid_pi = f"{ROOT_MAX4}/fid_max4/analysis/analysis_fid_MAX4_per_injection.csv"
    n_det, n_tot, medges, g, occ = molly_blocks(fid_json)
    mock_pi = f"{ROOT_MAX4}/p1/reductions/analysis_mock_2lpt_random_MAX4_per_injection.csv"; mock_pop = f"{ROOT_MAX4}/p1/mock/truth_r041/2lpt_random_population.csv"
    fp, n_sl_mock, n_extra = fp_block(mock_pi, mock_pop); fp_w = n_sl / n_sl_mock; fp_ell = n_sl_mock * (n_sl_mock / n_sl)
    ntrue = basis_pad_edges_19p0_0p2(); assert np.allclose(ntrue, TB); Nc = 0.5 * (ntrue[:-1] + ntrue[1:]); dN = np.diff(ntrue)
    # --- real-spectrum injection events, indexed by (stratum, z block, design point)
    ev = load_events("IR", f"{ROOT_MAX4}/response_study")
    zb = (np.searchsorted(ZC_EDGES, ev["z"], side="right") - 1).clip(0, 2); dp = np.array([int(np.argmin(np.abs(DESIGN - n))) for n in ev["logN"]])
    pools = {}
    for i in range(len(ev["logN"])):
        pools.setdefault((int(ev["stratum"][i]), int(zb[i]), int(dp[i])), []).append(i)
    # --- synthetic population: Poisson systems per (b, k, s)
    # f_syn(N) dN ∝ N^-beta dN (N linear, cm^-2): the expected number in latent bin b is ∝ ∫_bin N^-beta dN = (N_lo^(1-beta) - N_hi^(1-beta))/(beta-1);
    # expressed per dex so that shape[b] * dN[b] is that integral (dN in dex). 2026-09-03 FIX: the first build used Nc ** (-beta) with Nc = log10 N,
    # which is flat in log N (disclosed in the ledger; those packs are superseded).
    lo, hi = 10.0 ** ntrue[:-1], 10.0 ** ntrue[1:]
    shape = (lo ** (1.0 - a.beta) - hi ** (1.0 - a.beta)) / (a.beta - 1.0) / dN; shape = shape / (shape * dN).sum()
    if a.shape == "A":
        # SHAPE A: per-dex density of the frozen low-z pooled posterior median (hbi_validation_2026-09-02/R0/POOLED_ln_R0_fdraws.npz,
        # sha e43d9148…; dX-weighted z-average per draw, median over 6,000 draws), normalized at [20.3,20.5) = 1, on the basis
        # 19.0,19.2,19.5,19.7,...,22.1,22.4. Pad bins [19.0,19.2) and [19.2,19.5) (uncalibrated latent nuisance at low z) are continued
        # flat at the [19.5,19.7) value per the frozen replacement gate §1.
        SHAPE_A = np.array([1.345, 1.345, 1.345, 1.027, 1.341, 0.966, 1.000, 0.984, 0.848, 0.524, 0.368, 0.172, 0.102, 0.030, 0.0125, 0.0029])
        assert SHAPE_A.size == len(Nc)
        shape = SHAPE_A / (SHAPE_A * dN).sum()
    # expected TP count per unit amplitude: sum_b,k,s f(b) dN_b dX[k,s] C(b,s)  with C from the injection completeness of the nearest design point / stratum
    def C_of(b, s):
        idx = [i for st in [s] for i in pools.get((st, 0, int(np.argmin(np.abs(DESIGN - Nc[b])))), [])]
        return float(np.mean([ev["matched"][i] for i in idx])) if idx else 0.0
    exp_unit = sum(shape[b] * dN[b] * dX[k, s] * C_of(b, min(4, max(0, stratum_of_row(s)))) for b in range(len(Nc)) for k in range(dX.shape[0]) for s in range(dX.shape[1]))
    A = a.n_tp_target / max(exp_unit, 1e-12)
    truth_counts = np.zeros((len(Nc), len(ZF_EDGES) - 1), np.int64); counts = np.zeros((len(NHAT_EDGES) - 1, len(ZF_EDGES) - 1, len(SNR_EDGES) - 1), np.int64)
    n_sys = 0; n_tp = 0; n_pool_miss = 0
    nsys = np.zeros((len(Nc), len(ZF_EDGES) - 1, len(SNR_EDGES) - 1), np.int64)   # per-(b,k,s) truth draws (needed by the model generator)
    zmid = 0.5 * (ZF_EDGES[:-1] + ZF_EDGES[1:])
    for b in range(len(Nc)):
        for k in range(len(ZF_EDGES) - 1):
            for s in range(len(SNR_EDGES) - 1):
                mu = A * shape[b] * dN[b] * dX[k, s]
                if mu <= 0:
                    continue
                n = rng.poisson(mu); truth_counts[b, k] += n; n_sys += n; nsys[b, k, s] += n
                if n == 0 or a.generator == "model":
                    continue
                st = min(4, max(0, stratum_of_row(s))); zblk = int(np.searchsorted(ZC_EDGES, zmid[k], side="right") - 1); dpt = int(np.argmin(np.abs(DESIGN - Nc[b])))
                if abs(DESIGN[dpt] - Nc[b]) > 0.15 and Nc[b] > 19.5:
                    dpt = int(np.argmin(np.abs(DESIGN - Nc[b])))
                pool = pools.get((st, zblk, dpt)) or pools.get((st, 0, dpt)) or [i for i in range(len(ev["logN"])) if dp[i] == dpt]
                if not pool:
                    n_pool_miss += n; continue
                pick = rng.choice(pool, n)
                for i in pick:
                    if ev["matched"][i] and np.isfinite(ev["Nhat"][i]):
                        c = _idx(NHAT_EDGES, ev["Nhat"][i])
                        if 0 <= c < len(NHAT_EDGES) - 1:
                            counts[c, k, s] += 1; n_tp += 1
    # FP rows: Poisson(fp_w * fp[c, s]) spread over fine z bins by the exposure allocation
    col = dX.sum(axis=0); fp_E = np.zeros_like(dX); nz = col > 0; fp_E[:, nz] = dX[:, nz] / col[nz]
    n_fp = 0
    for c in range(fp.shape[0]):
        for s in range(fp.shape[1]):
            n = rng.poisson(fp_w * fp[c, s]); n_fp += n
            if n:
                ks = rng.choice(len(ZF_EDGES) - 1, n, p=(fp_E[:, s] / fp_E[:, s].sum()) if fp_E[:, s].sum() > 0 else None)
                for k in ks:
                    counts[c, k, s] += 1
    # --- Candidate E on the same injections (identical to the HZ2 real pack's override)
    E = build_E(ev); M = np.asarray(E["M"], float)
    kz_to_K = (np.searchsorted(ZC_EDGES, zmid, side="right") - 1).astype(np.int64)
    masked = np.zeros(len(NHAT_EDGES) - 1, bool); masked[(NHAT_EDGES[:-1] >= 19.5 - 1e-9) & (NHAT_EDGES[1:] <= 19.7 + 1e-9)] = True
    # deployed/adopted surfaces: reuse the real HZ2 pack's stamps if present (identical calibration arm), else fit the fid arm directly
    from CDDF_analysis.hbi_mcmc.extract_pack_hz import response_fit
    inj = list(csv.DictReader(open(fid_pi)))
    frm, cm, cs, ck, n_tp_r, n_uniq = response_fit(inj, os.path.join(a.out_dir, f"forward_response_hz2_fidclosure_s{a.seed}.npz"), 96)
    from CDDF_analysis.hbi_mcmc.extract_pack_hz import load_forward_response_pack_local
    fwd, fwd_meta = load_forward_response_pack_local(os.path.join(a.out_dir, f"forward_response_hz2_fidclosure_s{a.seed}.npz"))
    B = len(ntrue) - 1
    pack = dict(nhat_edges=NHAT_EDGES, ntrue_edges=ntrue, zf_edges=ZF_EDGES, zc_edges=ZC_EDGES, kz_to_K=kz_to_K, snr_edges=SNR_EDGES, nhat_masked_bins=masked,
                counts=counts, dX=dX, dX_coarse_committed=xc, molly_n_det=n_det, molly_n_tot=n_tot, molly_nhi_edges=medges, molly_snr_edges=SNR_EDGES, g_grid=g, g_occupancy=occ,
                **fwd, fp_counts=fp, fp_eta_c=np.zeros(len(NHAT_EDGES) - 1), fp_ell_eff=np.float64(fp_ell), fp_w_sightline_ratio=np.float64(fp_w), fp_E_alloc=fp_E,
                t_sigma=np.full(len(ZC_EDGES) - 1, T_SIGMA_FLOOR), truth_counts=truth_counts,
                tp_convention_id="hz_injection_nearest_dz_0.01", contract_id="hz2-2026-09-03", adopted_resp_version="candidate_E_realspectrum_v1",
                adopted_resp_mu_coef=fwd["resp_mu_coef"], adopted_resp_sig_coef=fwd["resp_sig_coef"], adopted_resp_skew_coef=fwd["resp_skew_coef"], adopted_resp_fit_range=fwd["resp_N_fit_range"],
                adopted_phi_ref=M.sum(axis=2), adopted_masses_override=M, adopted_carrier_mu=cm, adopted_carrier_sig=cs, adopted_carrier_skew=ck, adopted_carrier_shared3=np.zeros((cm.shape[0], 3)))
    tag = f"_{a.tag}" if a.tag else ""
    npz = os.path.join(a.out_dir, f"modelA_pack_HZ2_fidclosure{tag}_s{a.seed}.npz"); np.savez(npz, **pack)
    if a.generator == "model":
        # gate Amendment 1 (2026-09-03): outcomes drawn from the deployed observation model itself. Reload the pack exactly as the sampler
        # will, take P[c,b,k,s] from the fold's own index maps, verify the fold identity, then draw the TP counts multinomially per (b,k,s)
        # from the SAME truth draws; FP rows stay as drawn above.
        from CDDF_analysis.hbi_mcmc.pack import load_pack as _lp
        from CDDF_analysis.hbi_mcmc.count_conserving_fold import outcome_probabilities, cc_fold_adopted
        pk0 = _lp(npz, allow_nonstandard_grid=True); P = outcome_probabilities(pk0)
        f_true = nsys.sum(axis=2) / np.maximum(dN[:, None] * dX.sum(axis=1)[None, :], 1e-12)   # (B, Kf) density implied by the draws
        # identity: tp[c] = Σ_{b,k,s} dX[k,s] f[b,k] dN[b] P[c,b,k,s] must equal cc_fold_adopted at theta = log f
        lamz = np.asarray(fp, float) / float(fp_ell)
        mu_ref, parts_ref = cc_fold_adopted(pk0, np.log(np.clip(f_true, 1e-300, None)), lamz)
        tp_P = np.einsum("ks,bk,b,cbks->c", dX, f_true, dN, P)
        dev = float(np.max(np.abs(tp_P - parts_ref["tp"]) / np.maximum(np.abs(parts_ref["tp"]), 1e-9)))
        assert dev < 1e-8, f"outcome_probabilities does not reproduce the fold (max rel dev {dev:.2e})"
        n_tp = 0
        for b in range(P.shape[1]):
            for k in range(P.shape[2]):
                for s in range(P.shape[3]):
                    n = int(nsys[b, k, s])
                    if n == 0:
                        continue
                    pv = P[:, b, k, s]; pv = np.append(pv, max(0.0, 1.0 - pv.sum()))
                    draw = rng.multinomial(n, pv / pv.sum())[:-1]
                    counts[:, k, s] += draw; n_tp += int(draw.sum())
        pack["counts"] = counts; np.savez(npz, **pack)
        print(f"model generator: fold identity max rel dev {dev:.1e}; TP rows {n_tp}; expected TP at truth {float(parts_ref['tp'].sum()):.1f}")
    prov = dict(real_data=False, mode="hz2_fidclosure", seed=a.seed, gate="MAX4_HZ2_HBI_CLOSURE_GATE_2026-09-03.md §2" + (" + Amendment 1 (model generator)" if a.generator == "model" else ""),
                generator=a.generator, truth_shape=a.shape, beta=a.beta, n_tp_target=a.n_tp_target, amplitude=A,
                truth_shape_per_dex_normalized=(shape / shape[int(np.argmin(np.abs(Nc - 20.4)))]).round(5).tolist(),
                n_systems=int(n_sys), n_tp_rows=int(n_tp), n_fp_rows=int(n_fp), n_pool_miss=int(n_pool_miss), counts_total=int(counts.sum()), truth_total=int(truth_counts.sum()),
                code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip(), repair_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPAIR).decode().strip(),
                built=_dt.datetime.now().astimezone().isoformat(),
                hz2_observation_model=dict(response_representation="Candidate E", response_calibration_arm="real-spectrum A_shared DLA-only injections (fid_max4, 2,900)",
                                           completeness="real-spectrum A_shared injection calibration (molly block)", fp="2LPT loa-0 random arm extra rows (fp_w scaled)",
                                           associated_absorption_validation="AA (<= ~1.5 %)", meanflux_response_validation="bounded", native_response="stress only"),
                inputs=dict(population=pop_csv, population_sha256=_sha(pop_csv), fid_analysis=fid_json, fid_analysis_sha256=_sha(fid_json), fid_per_injection=fid_pi, fid_per_injection_sha256=_sha(fid_pi),
                            fp_mock_per_injection=mock_pi, fp_mock_population=mock_pop, pack_sha256=_sha(npz)))
    json.dump(prov, open(npz[:-4] + ".provenance.json", "w"), indent=1)
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    pk = load_pack(npz, allow_nonstandard_grid=True); consts, Mg = build_cc_tensors(pk)
    print(f"pack {npz} sha {_sha(npz)[:16]}… systems {n_sys} TP rows {n_tp} FP rows {n_fp} counts {int(counts.sum())} pool-miss {n_pool_miss} amplitude {A:.4g}")


if __name__ == "__main__":
    main()
