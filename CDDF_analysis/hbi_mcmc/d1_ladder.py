# -*- coding: utf-8 -*-
"""d1_ladder.py — the D1 BASIS-PAD ladder driver (finding D1, 2026-07-29).

Produces ``d1_basis_pad_ladder.json``: the forward-model closure of the Model A
truth-fold over the full cross of

    mock (2lpt0 / london0 / saclay0)
  x true-N basis pad floor (none / 19.3 / 19.0 / 18.5 / 18.0)
  x response covariate clamp (``both`` / ``hi``)          <- convention (a)
  x sub-floor completeness (``const_extrap`` / ``molly172``)  <- convention (b)

= 60 configurations. Conventions (a) and (b) are UNDECIDED and are crossed on
purpose so each becomes a MEASURED SYSTEMATIC rather than a choice: on the
unpadded grid the two clamps are identical, and under a pad they diverge
sharply because the response was never measured below ~19.35.

TWO PHASES, TWO ENVS (the extractor is jax-free by design, the fold needs jax):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla    python CDDF_analysis/hbi_mcmc/d1_ladder.py \
        --phase extract  --pack-dir <SCRATCH>

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc.d1_ladder \
        --phase selftest --pack-dir <SCRATCH> --out <REPO>/CDDF_analysis/hbi_mcmc/d1_basis_pad_ladder.json

The 30 packs are INPUTS, not results: they go to a scratch dir and are never
committed. ~75 kB each; the extract phase is ~6 min for all 30 (detection-side
cut bundles are cached across pad floors -- the pad touches the truth axis only).

MOCKS ONLY. No real-LOA path is touched and no real-data result value can enter
this artifact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEF_PACKDIR = os.environ.get("D1_LADDER_PACK_DIR", "/tmp/d1packs")
DEF_OUT = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/d1_basis_pad_ladder.json")

MOCKS = ["2lpt0", "london0", "saclay0"]
FLOORS = [None, 19.3, 19.0, 18.5, 18.0]
CLAMPS = ["both", "hi"]
CONVENTIONS = ["const_extrap", "molly172"]

# module-level, rebound by main() so the phase functions stay import-clean
PACKDIR = DEF_PACKDIR
OUT = DEF_OUT


def _extract_pack_module():
    """Load extract_pack.py file-directly: the hbi_mcmc package __init__ imports
    jax, and the extract phase deliberately runs in the jax-free `gpdla` env."""
    p = os.path.join(REPO, "CDDF_analysis/hbi_mcmc/extract_pack.py")
    spec = importlib.util.spec_from_file_location("modelA_extract_pack", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def load_pack(*a, **k):
    """Lazy re-export so `--phase extract` never imports jax."""
    from CDDF_analysis.hbi_mcmc.pack import load_pack as _lp
    return _lp(*a, **k)


def _FS():
    from CDDF_analysis.hbi_mcmc import forward_selftest as FS
    return FS


def _RP():
    from CDDF_analysis.hbi_mcmc import run_posterior as RP
    return RP


# ---------------------------------------------------------------------------
# shared helpers + the phase-2 evidence/verdict blocks
# ---------------------------------------------------------------------------
def tag_for(floor, conv):
    f = "none" if floor is None else f"{floor:.1f}".replace(".", "p")
    return f"_pad{f}_{conv}"


def gate_metrics(tab, pack):
    """EXACTLY run_posterior.forward_closure_gate's arithmetic (the committed
    gate): z over reported n-hat bins with obs > 0, chi2/dof over those bins."""
    floor = float(np.asarray(pack.nhat_edges, float)[0])
    rows = [b for b in tab["by_nhat"] if b["obs"] > 0 and b["lo"] >= floor - 1e-9]
    z = np.array([b["z"] for b in rows], float)
    return dict(
        z_total=float(abs(tab["total"]["z"])),
        z_bin_max=float(np.abs(z).max()) if len(z) else float("nan"),
        chi2_dof=float((z ** 2).sum() / max(len(z), 1)),
        n_bins=int(len(z)),
    )


def full_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   text=True).strip()


def dirty():
    """TRACKED-file dirtiness only (--untracked-files=no, matching
    extract_pack._git_commit). The stamp asserts that the CODE was at
    ``code_commit``; the artifact being written is itself untracked until it is
    committed, and counting that would make every stamp read dirty."""
    return bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, text=True).strip())


def _fold_kernel(pack, clamp):
    """The dX-weighted response kernel the fold effectively inverts: (C, B)."""
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K
    c = build_consts(pack, resp_clamp=clamp)
    K = np.asarray(build_K(jnp.zeros((2, c.n_sr, c.n_zr)), c))     # (S,KK,C,B)
    dXw = np.asarray(pack.dX, float)
    kz = np.asarray(pack.kz_to_K)
    W = np.zeros((c.n_c, c.n_b))
    for s in range(c.n_s):
        for kk in range(c.n_kk):
            W += dXw[kz == kk, s].sum() * K[s, kk]
    return W, c, K


def evidence_blocks():
    """Three supporting measurements the verdict rests on."""
    import jax
    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.forward import build_consts, build_K

    p0 = load_pack(os.path.join(PACKDIR, "modelA_pack_2lpt0_padnone_const_extrap.npz"))
    p18 = load_pack(os.path.join(PACKDIR, "modelA_pack_2lpt0_pad18p0_const_extrap.npz"))

    # (1) the power-law --truth-floor DIAGNOSTIC vs the mock's MEASURED truth
    _, f_pl = _FS().extend_pack_truth(p0, 18.0, 19.8, 20.8)
    f_me = _FS().truth_f(p18)
    dN = np.diff(np.asarray(p18.ntrue_edges, float))
    dXt = np.asarray(p18.dX, float).sum(1)
    n_pl = (f_pl * dN[:, None] * dXt[None, :]).sum(1)
    n_me = (f_me * dN[:, None] * dXt[None, :]).sum(1)
    npad = p18.n_b - p18.n_c
    lo = np.round(np.asarray(p18.ntrue_edges, float)[:-1], 3)
    powerlaw = dict(
        what=("forward_selftest.extend_pack_truth: the --truth-floor "
              "diagnostic replaces the missing sub-floor truth with a power "
              "law fitted to the pack's own truth over [19.8, 20.8). This "
              "block measures that surrogate against the mock's ACTUAL "
              "injected truth, which the re-extracted pad carries."),
        fit_window=[19.8, 20.8], mock="2lpt0", pad_floor=18.0,
        per_bin=[dict(lo=float(lo[i]), powerlaw=float(n_pl[i]),
                      measured=float(n_me[i]),
                      ratio=float(n_pl[i] / max(n_me[i], 1e-30)))
                 for i in range(npad)],
        sum_below_floor_powerlaw=float(n_pl[:npad].sum()),
        sum_below_floor_measured=float(n_me[:npad].sum()),
        sum_ratio=float(n_pl[:npad].sum() / n_me[:npad].sum()),
        finding=("the power-law surrogate OVER-PREDICTS the mock's own "
                 "sub-floor truth by 4.1x integrated (up to 8.1x in the "
                 "deepest bin). Any closure statement built on it is not a "
                 "statement about this forward model — it is a statement "
                 "about an injected surrogate population that does not exist "
                 "in the mock."),
    )

    # (2) conditioning of the fold kernel vs pad depth and clamp (E4)
    cond = []
    for name, clamps in [("2lpt0_padnone_const_extrap", ("both", "hi", "off")),
                         ("2lpt0_pad19p0_const_extrap", ("both", "hi")),
                         ("2lpt0_pad18p5_const_extrap", ("both", "hi")),
                         ("2lpt0_pad18p0_const_extrap", ("both", "hi"))]:
        pk = load_pack(os.path.join(PACKDIR, f"modelA_pack_{name}.npz"))
        for cl in clamps:
            try:
                W, c, K = _fold_kernel(pk, cl)
            except Exception as e:
                cond.append(dict(pack=name, resp_clamp=cl, error=str(e)))
                continue
            sv = np.linalg.svd(W, compute_uv=False)
            rm = K.sum(axis=2)
            cond.append(dict(
                pack=name, resp_clamp=cl, shape=list(W.shape),
                svd_cond=float(sv[0] / sv[-1]),
                n_unknowns_minus_n_data=int(c.n_b - c.n_c),
                kernel_rowmass_min=float(rm.min()),
                kernel_rowmass_max=float(rm.max())))
    conditioning = dict(
        what=("SVD condition number of the dX-weighted (C, B) response kernel "
              "the fold effectively inverts, vs pad depth and response clamp."),
        rows=cond,
        finding=("the pad does NOT degrade conditioning (1.47e5 unpadded -> "
                 "1.41e5 at pad 18.0 with the D2 clamp on; 3.63e8 with the "
                 "clamp OFF, i.e. most of the ill-conditioning WAS D2). What "
                 "the pad does instead is make the system UNDERDETERMINED: 44 "
                 "true-N unknowns against 29 observed bins at pad 18.0. Note "
                 "this is NOT the 2.77e10 quoted in the E4 escalation — that "
                 "number is not reproduced by any configuration measured here "
                 "and must be re-derived before it is used."),
    )

    # (3) the D1 mechanism, quantified: how much of mu in the lowest observed
    #     bins comes from TRUE bins below the reporting floor
    share = {}
    for name, cl in [("2lpt0_pad18p0_const_extrap", "hi"),
                     ("2lpt0_pad18p0_const_extrap", "both"),
                     ("2lpt0_pad18p0_molly172", "hi"),
                     ("2lpt0_pad18p0_molly172", "both")]:
        pk = load_pack(os.path.join(PACKDIR, f"modelA_pack_{name}.npz"))
        c = build_consts(pk, resp_clamp=cl)
        K = np.asarray(build_K(jnp.zeros((2, c.n_sr, c.n_zr)), c))
        f = _FS().truth_f(pk)
        C_bs = np.asarray(jax.nn.sigmoid(c.eta_hat))[:, c.b_to_cell]
        contrib = (C_bs.T[:, None, :] * np.asarray(c.g_bk)[:, :, None]
                   * f[:, :, None] * np.asarray(c.dN_b)[:, None, None])
        Kf = K[:, np.asarray(pk.kz_to_K)]
        # (c, b) signal expectation with the dX weight applied per (k, s)
        mu = np.einsum("skcb,bks,ks->cb", Kf, contrib, np.asarray(pk.dX, float))
        npad_ = pk.n_b - pk.n_c
        share[f"{name}|clamp={cl}"] = [
            float(mu[c_, :npad_].sum() / max(mu[c_].sum(), 1e-30))
            for c_ in range(min(10, pk.n_c))]
    mechanism = dict(
        what=("fraction of the predicted signal counts in each of the LOWEST "
              "observed n-hat bins that is fed by TRUE systems BELOW the "
              "reporting floor (pad 18.0). Schema v1 carried NONE of them."),
        subfloor_share_of_mu_by_lowest_nhat_bins=share,
        finding=("83% of the predicted counts in [19.5, 19.6) come from true "
                 "systems below 19.5. That is D1 stated as a number, and it "
                 "is why the truncated basis under-predicted the lowest bin "
                 "by 6x (mu/obs 0.1655)."),
    )
    return dict(powerlaw_surrogate_vs_measured_truth=powerlaw,
                fold_kernel_conditioning=conditioning,
                d1_mechanism=mechanism)


def build_verdict(rows, extras):
    closing = [k for k, v in rows.items() if v["closes"]]
    best_chi2 = min(rows.items(), key=lambda kv: kv[1]["chi2_dof"])
    best_tot = min(rows.items(), key=lambda kv: abs(kv[1]["total_ratio"] - 1.0))
    base = rows["2lpt0|pad=None|clamp=both|cmp=const_extrap"]

    def bracket(mock, floor):
        vals = {f"{cl}|{cv}": rows[f"{mock}|pad={floor}|clamp={cl}|cmp={cv}"]["total_ratio"]
                for cl in CLAMPS for cv in CONVENTIONS}
        return dict(values=vals, min=min(vals.values()), max=max(vals.values()),
                    spread_frac=(max(vals.values()) - min(vals.values()))
                    / max(min(vals.values()), 1e-30))
    return dict(
        question=("does ANY (mock x pad_floor x resp_clamp x sub-floor "
                  "completeness) configuration close the forward model inside "
                  "the committed gate (|z_total| <= 5, max|z_bin| <= 5, "
                  "chi2/dof <= 3)?"),
        answer="NO",
        n_configurations=len(rows),
        n_closing=len(closing),
        closing_configurations=closing,
        committed_path_baseline=dict(
            note=("what the COMMITTED code gives on the UNPADDED pack — the "
                  "number every downstream claim must be measured against"),
            config="2lpt0 | no pad | resp_clamp=both | const_extrap",
            total_ratio=base["total_ratio"], z_total=base["z_total"],
            chi2_dof=base["chi2_dof"],
            lowest_nhat_bin_ratio=base["by_nhat"][0]["ratio"],
            truth_total=base["truth_total"], counts_total=base["counts_total"]),
        best_by_chi2=dict(config=best_chi2[0],
                          chi2_dof=best_chi2[1]["chi2_dof"],
                          total_ratio=best_chi2[1]["total_ratio"],
                          gate_max=3.0,
                          factor_over_gate=best_chi2[1]["chi2_dof"] / 3.0),
        best_by_total_ratio=dict(config=best_tot[0],
                                 total_ratio=best_tot[1]["total_ratio"],
                                 z_total=best_tot[1]["z_total"],
                                 chi2_dof=best_tot[1]["chi2_dof"],
                                 lowest_nhat_bin_ratio=best_tot[1]["by_nhat"][0]["ratio"]),
        convention_bracket_at_pad_18p0={m: bracket(m, 18.0) for m in MOCKS},
        convention_bracket_at_pad_19p0={m: bracket(m, 19.0) for m in MOCKS},
        is_D1_alone_sufficient=dict(
            answer="NO",
            level_vs_shape=("the pad fixes the LEVEL and leaves the SHAPE. At "
                            "2lpt0 / pad 18.0 / clamp=hi / const_extrap the "
                            "total is 0.9973 (z=+0.8, inside the gate's total "
                            "leg) while chi2/dof is 51.5 — 17x the tolerance — "
                            "and the per-bin ratio still runs 0.765 to 1.431."),
            residual_is_not_one_defect=(
                "the residual splits in two. (i) A LOW-N deficit in the two "
                "bins below 19.7 that keeps shrinking with pad depth and "
                "swings hard with the two undecided conventions — that is what "
                "is left of D1, and it is a CONVENTION problem, not a missing "
                "-basis problem. (ii) A HIGH-N excess of 1.23-1.43x on 2lpt0 "
                "(up to 1.80x on london0) above logN ~ 21.6, whose per-bin "
                "digits are IDENTICAL across every pad floor and both "
                "completeness conventions — it is residual D2, untouched by "
                "D1, and it alone puts chi2/dof far over the gate."),
            does_E4_dominate=(
                "NOT as stated. The measured conditioning of the fold kernel "
                "is 1.4-1.5e5 with the D2 clamp on and is IMPROVED, not "
                "degraded, by the pad; the 2.77e10 figure in the E4 "
                "escalation is not reproduced here and should be re-derived "
                "before use. What the pad DOES create is an "
                "under-determination: 44 true-N unknowns against 29 observed "
                "bins at pad 18.0, i.e. 15 unidentified directions that only "
                "a prior can fill. So the ordering is: residual D2 dominates "
                "the chi2, the two sub-floor conventions dominate the level, "
                "and E4 becomes a real problem only once those two are fixed "
                "and the pad has to be INFERRED rather than read from truth."),
        ),
        next_actions=[
            "D2 is NOT closed by the covariate clamp: a 1.23-1.43x (2lpt0) / "
            "up to 1.80x (london0) excess survives above logN ~ 21.6 in every "
            "configuration. Attack that before any further pad work.",
            "resp_clamp is not a free choice under a pad: it moves the total "
            "by up to 15% at pad 18.0. Either measure the response below "
            "19.35 or declare the sub-floor response a stated limit.",
            "the sub-floor completeness convention moves the total by ~8%. "
            "The measured floor-17.2 molly matrix now exists in the pack, so "
            "'const_extrap' should be retired as a default.",
            "rung 10 stays gated: no configuration passes the pre-flight.",
        ],
        do_not_quote=(
            "the 'total 0.998 / chi2/dof 28.2' figure from the uncommitted "
            "scratchpad reconstruction is NOT reproduced and must not be "
            "cited. The committed --truth-floor 18.5 power-law diagnostic "
            "gives total 0.9986 / chi2/dof 22.1 on this code, but it gets "
            "there by injecting 4.1x MORE sub-floor truth than the mock "
            "actually contains (see powerlaw_surrogate_vs_measured_truth). "
            "The re-extracted pad — the mock's OWN measured sub-floor truth — "
            "gives 0.8643 (clamp=both) to 0.9973 (clamp=hi) with chi2/dof "
            "295.6 to 51.5. Separately, the '0.9127 total / 0.6268 lowest "
            "bin' committed-path figure in the task brief was NOT reproduced "
            "by any of the 60 configurations measured here; the nearest are "
            "london0 | pad 18.5 | clamp=both | const_extrap (0.9130 total, "
            "0.6037 lowest bin) and london0 | pad 18.0 | clamp=both | "
            "const_extrap (0.9130 / 0.6057), so that figure most likely came "
            "from a london0 pad at clamp=both with a slightly different "
            "sub-floor convention. It is NOT the 2lpt0 committed-path number, "
            "which is 0.7312 / 0.1655 unpadded."),
    )


# ---------------------------------------------------------------------------
# phase 1 — extract the ladder of packs (env: gpdla, jax-free)
# ---------------------------------------------------------------------------
def phase_extract():
    EP = _extract_pack_module()
    OUTD = PACKDIR

    os.makedirs(OUTD, exist_ok=True)
    manifest = {}
    for conv in CONVENTIONS:
        t0 = time.time()
        frozen = EP.build_frozen_calibration(OUTD, completeness=conv)
        print(f"[ladder] frozen[{conv}] built in {time.time()-t0:.0f}s; "
              f"molly cells={len(frozen['molly']['molly_nhi_edges'])-1} "
              f"g_grid={frozen['g_grid'].shape}", flush=True)
        for floor in FLOORS:
            for mock in MOCKS:
                tag = tag_for(floor, conv)
                r = EP.extract_pack(mock, OUTD, frozen, pad_floor=floor, tag=tag)
                manifest[f"{mock}{tag}"] = dict(
                    mock=mock, pad_floor=floor, completeness=conv,
                    npz=r["npz"], counts_total=r["counts_total"])
                print(f"[ladder] done {mock}{tag}", flush=True)
    with open(os.path.join(OUTD, "ladder_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v["counts_total"] for k, v in manifest.items()}, indent=1))


# ---------------------------------------------------------------------------
# phase 2 — fold + gate every pack, emit the artifact (env: gpdla-hbi)
# ---------------------------------------------------------------------------
def phase_selftest():
    t_start = time.time()
    rows = {}
    probes_by_pack = {}
    for conv in CONVENTIONS:
        for floor in FLOORS:
            for mock in MOCKS:
                tag = tag_for(floor, conv)
                p = os.path.join(PACKDIR, f"modelA_pack_{mock}{tag}.npz")
                pack = load_pack(p)
                n_pad = pack.n_b - pack.n_c
                prov = pack.provenance or {}
                for clamp in CLAMPS:
                    key = f"{mock}|pad={floor}|clamp={clamp}|cmp={conv}"
                    t0 = time.time()
                    res = _FS().selftest(pack, resp_clamp=clamp)
                    tab = _FS().ratio_tables(res, pack)
                    gm = gate_metrics(tab, pack)
                    rows[key] = dict(
                        mock=mock, pad_floor=floor, resp_clamp=clamp,
                        completeness_below_floor=conv,
                        n_pad_bins=int(n_pad),
                        truth_total=float(np.asarray(pack.truth_counts).sum()),
                        truth_total_below_floor=float(
                            np.asarray(pack.truth_counts)[:n_pad].sum()) if n_pad else 0.0,
                        counts_total=float(np.asarray(pack.counts).sum()),
                        total_mu=float(tab["total"]["mu"]),
                        total_obs=float(tab["total"]["obs"]),
                        total_ratio=float(tab["total"]["ratio"]),
                        z_total=float(tab["total"]["z"]),
                        chi2_dof=gm["chi2_dof"],
                        z_bin_max=gm["z_bin_max"],
                        n_gate_bins=gm["n_bins"],
                        closes=bool(gm["z_total"] <= _RP().GATE["z_total_max"]
                                    and gm["z_bin_max"] <= _RP().GATE["z_bin_max"]
                                    and gm["chi2_dof"] <= _RP().GATE["chi2_dof_max"]),
                        by_nhat=[dict(lo=b["lo"], hi=b["hi"], mu=b["mu"],
                                      obs=b["obs"], ratio=b["ratio"], z=b["z"])
                                 for b in tab["by_nhat"]],
                        by_z=[dict(lo=b["lo"], hi=b["hi"], ratio=b["ratio"],
                                   z=b["z"]) for b in tab["by_z"]],
                        pack=os.path.basename(p),
                        pack_provenance_commit=prov.get("code_commit"),
                    )   # NOTE: no wall-clock field here -- `ladder` is
                        # byte-reproducible from the same packs.
                    print(f"{key:62s} ratio={tab['total']['ratio']:.4f} "
                          f"z={tab['total']['z']:+9.1f} chi2/dof={gm['chi2_dof']:10.1f} "
                          f"lowbin={tab['by_nhat'][0]['ratio']:.4f} "
                          f"({time.time()-t0:.1f}s)", flush=True)
                if mock == "2lpt0":
                    probes_by_pack[f"pad={floor}|cmp={conv}"] = _FS().structural_probes(pack)

    # cross-check the inline gate arithmetic against the COMMITTED gate routine
    ref_pack = load_pack(os.path.join(
        PACKDIR, f"modelA_pack_2lpt0{tag_for(None, 'const_extrap')}.npz"))
    ref = _RP().forward_closure_gate(ref_pack, resp_clamp="both")
    inline = rows["2lpt0|pad=None|clamp=both|cmp=const_extrap"]
    xcheck = dict(
        routine="CDDF_analysis/hbi_mcmc/run_posterior.py:forward_closure_gate",
        committed_total_ratio=ref["total_ratio"], inline_total_ratio=inline["total_ratio"],
        committed_chi2_dof=ref["chi2_dof"], inline_chi2_dof=inline["chi2_dof"],
        committed_z_total=ref["z_total"], inline_z_total=abs(inline["z_total"]),
        agrees=bool(np.isclose(ref["total_ratio"], inline["total_ratio"], rtol=1e-12)
                    and np.isclose(ref["chi2_dof"], inline["chi2_dof"], rtol=1e-12)),
        committed_pass=bool(ref["pass"]),
    )
    print("\n[xcheck committed gate]", json.dumps(xcheck, indent=1))

    extras = evidence_blocks()
    verdict = build_verdict(rows, extras)
    out = dict(
        metadata=dict(
            title="D1 basis-pad ladder — forward-model closure vs true-N pad "
                  "floor, response clamp, and sub-floor completeness",
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            code_commit=full_sha(),
            code_commit_dirty=dirty(),
            branch=subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                text=True).strip(),
            routines=dict(
                driver=("CDDF_analysis/hbi_mcmc/d1_ladder.py "
                        "(--phase extract, then --phase selftest)"),
                extractor="CDDF_analysis/hbi_mcmc/extract_pack.py:extract_pack "
                          "(--basis-pad-floor / --completeness-below-floor)",
                pad_grid="CDDF_analysis/hbi_mcmc/extract_pack.py:basis_pad_edges",
                fold="CDDF_analysis/hbi_mcmc/forward_selftest.py:selftest "
                     "(-> forward.build_consts + forward.fold_mu)",
                gate="CDDF_analysis/hbi_mcmc/run_posterior.py:forward_closure_gate",
            ),
            pack_dir=PACKDIR,
            rederive=(
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "conda run -n gpdla python CDDF_analysis/hbi_mcmc/d1_ladder.py "
                f"--phase extract --pack-dir {PACKDIR}  &&  "
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "conda run -n gpdla-hbi python -m CDDF_analysis.hbi_mcmc."
                f"d1_ladder --phase selftest --pack-dir {PACKDIR} --out {OUT}"),
            pack_note="packs are INPUTS, written to scratch, never committed",
            gate_tolerances=dict(_RP().GATE),
            mocks_only=True,
            privacy="mock packs only; no real-LOA path is touched",
            wall_seconds=round(time.time() - t_start, 1),
        ),
        verdict=verdict,
        ladder=rows,
        structural_probes_2lpt0=probes_by_pack,
        committed_gate_crosscheck=xcheck,
        **extras,
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[ladder] wrote {OUT}")


def main(argv=None):
    global PACKDIR, OUT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["extract", "selftest"])
    p.add_argument("--pack-dir", default=DEF_PACKDIR)
    p.add_argument("--out", default=DEF_OUT)
    a = p.parse_args(argv)
    PACKDIR = a.pack_dir
    OUT = a.out
    if a.phase == "extract":
        return phase_extract()
    return phase_selftest()


if __name__ == "__main__":
    main()
