#!/usr/bin/env python
"""analyze_fp_target.py — tables, shape comparisons and figures for the FP
CALIBRATION TARGET DEFINITION REVIEW (PI ruling 2026-09-13c §7).

Consumes the products written by ``build_fp_target_review.py`` plus the A0
packs, and writes ``FP_TARGET_REVIEW_TABLES.json`` + up to four figures.
Calibration / mock information only; no sampler; nothing adopted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import gamma

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SUPPORT = os.path.join(_REPO, "validation", "absorber_ladder", "support")
for _p in (_HERE, _SUPPORT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import selection as SEL                                          # noqa: E402
from support_contract import read_stamp                          # noqa: E402

FAMILIES = ("2lpt0", "london0", "saclay0")
A0_SUPPORT_DIR = ("/scratch/cavestru_root/cavestru0/mfho/"
                  "absorber_ladder_2026-09-13/support")
LIVE_S = (2, 3, 4, 5, 6, 7)
SNR_LABEL = {2: "2-3", 3: "3-4", 4: "4-5", 5: "5-6", 6: "6-7", 7: ">=7"}
#: LaTeX-safe versions for figure tick labels (usetex is on)
SNR_TEX = {2: "2--3", 3: "3--4", 4: "4--5", 5: "5--6", 6: "6--7",
           7: r"$\geq 7$"}


# ---------------------------------------------------------------------------
def kl(p, q, eps=1e-300):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], eps))))


def shape_compare(obs_cs, tmpl_cs, live):
    """Compare two (C,S) shapes after renormalising the template to the
    observed total (so this is a SHAPE test, not a scale test)."""
    obs = np.asarray(obs_cs, float)[:, live]
    tm = np.asarray(tmpl_cs, float)[:, live]
    n = obs.sum()
    exp = tm * (n / tm.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        pear = (obs - exp) / np.sqrt(np.maximum(exp, 1e-12))
    # multinomial deviance, 2 * sum obs log(obs/exp)
    m = obs > 0
    dev = 2.0 * float(np.sum(obs[m] * np.log(obs[m] / exp[m])))
    dof = int(obs.size - 1)
    return dict(
        n_obs=float(n), kl_row_Nhat=kl(obs.sum(1), tm.sum(1)),
        kl_col_SNR=kl(obs.sum(0), tm.sum(0)),
        kl_full_cell=kl(obs.ravel(), tm.ravel()),
        multinomial_deviance=dev, cells=dof + 1,
        deviance_per_cell=dev / (dof + 1),
        max_abs_pearson=float(np.nanmax(np.abs(pear))),
        n_cells_pearson_gt3=int(np.nansum(np.abs(pear) > 3.0)),
        pearson=pear.tolist())


def two_sample_g(n1, n2):
    """2 x K G-test between two multinomial samples (the 89 calibration events
    and the mock census), which — unlike a chi2 against a fixed template —
    PROPAGATES the calibration sample's own Poisson noise.  Returns G, dof, p."""
    from scipy.stats import chi2 as _chi2
    n1 = np.asarray(n1, float); n2 = np.asarray(n2, float)
    keep = (n1 + n2) > 0
    n1, n2 = n1[keep], n2[keep]
    N1, N2, N = n1.sum(), n2.sum(), n1.sum() + n2.sum()
    col = n1 + n2
    e1, e2 = N1 * col / N, N2 * col / N
    g = 0.0
    for o, e in ((n1, e1), (n2, e2)):
        m = o > 0
        g += 2.0 * float(np.sum(o[m] * np.log(o[m] / e[m])))
    dof = int(keep.sum() - 1)
    return dict(G=g, dof=dof, p_value=float(_chi2.sf(g, dof)),
                n_calib=float(N1), n_census=float(N2),
                expected_calib_counts=(N1 * n2 / N2).round(2).tolist(),
                observed_calib_counts=n1.astype(int).tolist())


def template_from_counts(fp_counts, pack):
    live = np.asarray(pack["dX"], float).sum(axis=0) > 0
    ell = float(pack["fp_ell_eff"])
    fpw = float(pack["fp_w_sightline_ratio"])
    eta = np.asarray(pack["fp_eta_c"], float)
    n = float(np.asarray(fp_counts).sum())
    lam = float(gamma.ppf(0.5, n + 0.5, scale=1.0 / ell))
    mu = SEL.template_mu_cs(fp_counts, live, lam_total=lam,
                            fp_w_ell_eff=fpw * ell, eta_c=eta)
    return mu, live, dict(ell_eff=ell, fp_w=fpw, lam_median=lam,
                          n_calib_events=n)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", required=True)
    p.add_argument("--figdir", default=None)
    a = p.parse_args(argv)

    L = np.load(os.path.join(a.products, "loa0_fp_target.npz"),
                allow_pickle=True)
    g89 = np.asarray(L["fp_counts_89"], np.int64)
    ladder = json.loads(str(L["selection_ladder"]))
    cfsup = json.loads(str(L["counterfactual_support"]))
    groups = SEL.group_rows()
    T = dict(loa0_selection_ladder=ladder, loa0_counterfactual_support=cfsup,
             loa0_support_id=str(L["support_id"]))

    # ---- the 89 vs 2,378 support match -----------------------------------
    nhi2378 = np.asarray(L["nhi_2378"], float)
    z2378 = np.asarray(L["z_2378"], float)
    snr2378 = np.asarray(L["snr_2378"], float)
    on_floor = nhi2378 >= 19.5
    in_z = (z2378 >= 2.0) & (z2378 < 3.5)
    T["product_89_vs_2378"] = dict(
        n_2378=int(len(nhi2378)),
        n_2378_with_Nhat_ge_19p5_no_z_window=int(on_floor.sum()),
        n_2378_with_Nhat_ge_19p5_and_z_window=int((on_floor & in_z).sum()),
        n_89=int(g89.sum()),
        identical_on_the_fp_counts_support=bool(
            (on_floor & in_z).sum() == g89.sum()),
        events_below_Nhat_19p5=int((~on_floor).sum()),
        note=("the two products are the SAME selection; the only difference is "
              "the observed-N floor (and the fine-grid z window). Restricted "
              "to the fp_counts support the 2,378-event product IS the "
              "89-event block, event for event."),
        Nhat_histogram_0p1dex_from_17p2=np.histogram(
            nhi2378, bins=np.round(np.arange(17.2, 22.5, 0.1), 3))[0].tolist(),
        snr_stratum_counts_2378=np.histogram(
            np.clip(snr2378, 0, 1e9), bins=[2, 3, 4, 5, 6, 7, 1e9])[0].tolist(),
        snr_stratum_counts_89=g89.sum(axis=0)[2:].tolist())

    # log-linear tail fit on the 2,378 product (19.0 -> 20.1), for the record
    edges = np.round(np.arange(17.2, 22.5, 0.1), 3)
    h = np.histogram(nhi2378, bins=edges)[0]
    lo = edges[:-1]
    fitm = (lo >= 19.0 - 1e-9) & (lo < 20.1 - 1e-9)
    x = lo[fitm] + 0.05
    y = h[fitm].astype(float)
    # Poisson (log-link) fit, IRLS — the convention of the Stage-0 design memo
    beta = np.array([np.log(max(y.mean(), 1e-9)), -1.5])
    X = np.column_stack([np.ones_like(x), x - 19.55])
    for _ in range(60):
        mu_ = np.exp(X @ beta)
        W = np.diag(mu_)
        z = X @ beta + (y - mu_) / np.maximum(mu_, 1e-12)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
    cov = np.linalg.inv(X.T @ np.diag(np.exp(X @ beta)) @ X)
    slope10 = float(beta[1] / np.log(10.0))
    T["product_89_vs_2378"]["poisson_loglinear_slope_dex_per_dex_19p0_20p1"] = slope10
    T["product_89_vs_2378"]["poisson_loglinear_slope_sd"] = float(
        np.sqrt(cov[1, 1]) / np.log(10.0))
    Xall = np.column_stack([np.ones_like(lo), lo + 0.05 - 19.55])
    pred = np.exp(Xall @ beta)
    T["product_89_vs_2378"]["loglinear_extrapolated_events_ge_20p3"] = float(
        pred[lo >= 20.3 - 1e-9].sum())
    T["product_89_vs_2378"]["loglinear_extrapolated_events_ge_20p0"] = float(
        pred[lo >= 20.0 - 1e-9].sum())
    T["product_89_vs_2378"]["observed_events_ge_20p3"] = int(
        h[lo >= 20.3 - 1e-9].sum())
    T["product_89_vs_2378"]["observed_events_ge_20p0"] = int(
        h[lo >= 20.0 - 1e-9].sum())

    # ---- per family -------------------------------------------------------
    T["families"] = {}
    figdata = {}
    for fam in FAMILIES:
        pack = np.load(os.path.join(A0_SUPPORT_DIR,
                                    f"scanpack_{fam}_b300_A0.npz"),
                       allow_pickle=True)
        mu_cs, live, const = template_from_counts(g89, pack)
        E = np.asarray(pack["fp_E_alloc"], float)
        kzK = np.asarray(pack["kz_to_K"], int)
        d = np.load(os.path.join(a.products,
                                 f"fp_target_decomposition_{fam}.npz"),
                    allow_pickle=True)
        prov = json.loads(str(d["provenance"]))

        cls = {k[4:]: np.asarray(d[k], np.int64) for k in d.files
               if k.startswith("cks_")}
        hostless = cls["hostless"]
        no_abs = cls["hostless_no_absorber"]
        taken = cls["hostless_taken"]
        unres = cls["hostless_unresolvable"]
        total = cls["total"]

        # partition check (also unit-tested)
        part = (no_abs + taken + unres + cls["host_17p2_19p0"]
                + cls["host_19p0_19p5"] + cls["host_19p5_19p7"]
                + cls["host_19p7_21p6"] + cls["host_ge_21p6"])
        assert np.array_equal(part, total), "decomposition is not a partition"

        mu_cks = mu_cs[:, None, :] * E[None, :, :]
        rec = dict(constants=const,
                   support_id=str(d["support_id"]),
                   support_equals_A0v2_census=prov[
                       "support_gate_equals_A0v2_census"],
                   fidelity_gates=prov["fidelity_gates_vs_A0v2_census"],
                   meta=prov["meta"])

        def g(x, m=None):
            return int(x.sum() if m is None else x[m].sum(axis=(0, 2)).sum())

        rec["totals"] = {k: int(v.sum()) for k, v in cls.items()}
        rec["template_total_t0"] = float(mu_cks.sum())

        # by N-hat group
        rec["by_Nhat_group"] = {}
        for name, rows in groups.items():
            rec["by_Nhat_group"][name] = dict(
                template_t0=float(mu_cs[rows].sum()),
                hostless=int(hostless.sum(axis=1)[rows].sum()),
                hostless_no_absorber=int(no_abs.sum(axis=1)[rows].sum()),
                hostless_taken=int(taken.sum(axis=1)[rows].sum()),
                host_17p2_19p0_P6b=int(
                    cls["host_17p2_19p0"].sum(axis=1)[rows].sum()),
                counts=int(total.sum(axis=1)[rows].sum()))
        # by S/N stratum
        rec["by_SNR_stratum"] = {}
        for s in LIVE_S:
            rec["by_SNR_stratum"][SNR_LABEL[s]] = dict(
                template_t0=float(mu_cs[:, s].sum()),
                hostless=int(hostless[:, :, s].sum()),
                hostless_no_absorber=int(no_abs[:, :, s].sum()),
                hostless_taken=int(taken[:, :, s].sum()),
                counts=int(total[:, :, s].sum()))
        # by coarse z block
        rec["by_coarse_z"] = {}
        for K in range(3):
            m = kzK == K
            rec["by_coarse_z"][f"K{K}"] = dict(
                template_t0=float(mu_cks[:, m, :].sum()),
                hostless=int(hostless[:, m, :].sum()),
                hostless_no_absorber=int(no_abs[:, m, :].sum()))

        # shapes
        # --- calibration-noise-aware two-sample tests (the 89 events vs the
        # census, on the COARSE cells where the 89 have counts) -------------
        grp_list = list(groups)
        hl_cs = hostless.sum(axis=1)
        na_cs = no_abs.sum(axis=1)
        cal_grp = np.array([g89[groups[g]].sum() for g in grp_list], float)
        cen_grp = np.array([hl_cs[groups[g]].sum() for g in grp_list], float)
        cal_snr = g89.sum(axis=0)[list(LIVE_S)].astype(float)
        cen_snr = hl_cs.sum(axis=0)[list(LIVE_S)].astype(float)
        cal_cell = np.array([[g89[groups[g], s].sum() for s in LIVE_S]
                             for g in grp_list], float).ravel()
        cen_cell = np.array([[hl_cs[groups[g], s].sum() for s in LIVE_S]
                             for g in grp_list], float).ravel()
        rec["two_sample_tests_89_vs_census"] = dict(
            Nhat_group_marginal=two_sample_g(cal_grp, cen_grp),
            SNR_marginal=two_sample_g(cal_snr, cen_snr),
            coarse_3x6_cells=two_sample_g(cal_cell, cen_cell),
            Nhat_group_marginal_no_absorber=two_sample_g(
                cal_grp, np.array([na_cs[groups[g]].sum() for g in grp_list],
                                  float)),
            SNR_marginal_no_absorber=two_sample_g(
                cal_snr, na_cs.sum(axis=0)[list(LIVE_S)].astype(float)))
        # what the 89-event sample can and cannot resolve
        share203 = float(hl_cs[groups[">=20.3"]].sum() / hl_cs.sum())
        rec["calibration_power"] = dict(
            live_cells=int(g89.shape[0] * live.sum()),
            nonempty_calibration_cells=int((g89[:, live] > 0).sum()),
            calibration_events=int(g89.sum()),
            census_share_ge_20p3=share203,
            expected_calib_events_ge_20p3_if_census_shape=89.0 * share203,
            observed_calib_events_ge_20p3=int(g89[groups[">=20.3"]].sum()),
            poisson_p_obs0=float(np.exp(-89.0 * share203)),
            one_sided_95pc_upper_limit_on_share=3.0 / 89.0,
            template_share_ge_20p3_all_perks_pseudocount=float(
                mu_cs[groups[">=20.3"]].sum() / mu_cs.sum()))

        # a0 sensitivity of the >=20.3 FP prediction (it is 100 % pseudo-count)
        rows203 = groups[">=20.3"]
        n_cells_203 = int(rows203.sum() * live.sum())
        n_live_cells = int(g89.shape[0] * live.sum())
        tot = float(mu_cks.sum())
        rec["perks_a0_sensitivity_ge_20p3"] = {
            f"a0={a0:g}": float(tot * (n_cells_203 * a0)
                                / (89.0 + n_live_cells * a0))
            for a0 in (0.0, 1.0 / n_live_cells, 0.05, 0.5)}
        rec["perks_a0_sensitivity_ge_20p3"]["a0_of_record"] = 1.0 / n_live_cells
        rec["perks_a0_sensitivity_ge_20p3"]["mock_census_hostless_ge_20p3"] = int(
            hostless.sum(axis=1)[rows203].sum())
        rec["perks_a0_sensitivity_ge_20p3"]["calibration_events_ge_20p3"] = int(
            g89[rows203].sum())

        rec["shape_vs_template"] = dict(
            hostless=shape_compare(hostless.sum(axis=1), mu_cs, live),
            hostless_no_absorber=shape_compare(no_abs.sum(axis=1), mu_cs, live))

        # window-size probe restricted to the grid
        hn = np.asarray(d["hostless_nhat"], float)
        hz = np.asarray(d["hostless_zobs"], float)
        hdz = np.asarray(d["hostless_dz_near"], float)
        on = (hn >= 19.5) & (hn < 22.4) & (hz >= 2.0) & (hz < 3.5)
        probe = dict(n_on_grid=int(on.sum()),
                     no_truth_in_sightline=int((~np.isfinite(hdz) & on).sum()))
        for thr in (0.015, 0.02, 0.03, 0.05):
            probe[f"nearest_truth_within_dz_rel_{thr}"] = int(
                ((hdz < thr) & on).sum())
        hi = on & (hn >= 20.3)
        probe["ge_20p3"] = dict(
            n=int(hi.sum()),
            no_truth_in_sightline=int((hi & ~np.isfinite(hdz)).sum()),
            nearest_within_0p02=int((hi & (hdz < 0.02)).sum()),
            nearest_within_0p05=int((hi & (hdz < 0.05)).sum()))
        rec["window_size_probe"] = probe
        T["families"][fam] = rec
        figdata[fam] = dict(mu_cs=mu_cs, live=live, hostless=hostless,
                            no_abs=no_abs, taken=taken, cls=cls)

    T["verdict_inputs"] = dict(
        note=("hostless_no_absorber / hostless is the fraction of the mock "
              "census that the HCD-free twin's event definition can also "
              "produce; hostless_taken is the greedy one-to-one matching "
              "artefact; hostless_unresolvable is the bookkeeping artefact."),
        fraction_no_absorber={
            f: T["families"][f]["totals"]["hostless_no_absorber"]
               / T["families"][f]["totals"]["hostless"] for f in FAMILIES})

    out = os.path.join(a.products, "FP_TARGET_REVIEW_TABLES.json")
    with open(out, "w") as fh:
        json.dump(T, fh, indent=1, default=str)
    print("[tables]", out)

    if a.figdir:
        make_figures(a.figdir, g89, figdata, L, T)
    return T


# ---------------------------------------------------------------------------
def make_figures(figdir, g89, figdata, L, T):
    os.makedirs(figdir, exist_ok=True)
    sys.path.insert(0, "/home/mfho/Latex/gp_dla_desi_y3/paper_figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    note = "paper_figures/common.py style()"
    try:
        import common
        common.style()
        f = plt.figure(); plt.plot([0, 1], [0, 1])
        plt.xlabel(r"$\log N_{\rm HI}$"); f.savefig("/tmp/_probe_fptr.png")
        plt.close(f)
    except Exception as exc:
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm",
                             "font.family": "serif", "figure.dpi": 150,
                             "savefig.dpi": 220, "savefig.bbox": "tight"})
        note = f"FALLBACK mathtext ({type(exc).__name__})"
    print("[style]", note)
    nh = SEL.NHAT_EDGES
    ctr = 0.5 * (nh[:-1] + nh[1:])
    live = figdata["2lpt0"]["live"]
    mu = figdata["2lpt0"]["mu_cs"]

    # Fig 1 — N-hat marginal shapes
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    tm = mu.sum(1); tm = tm / tm.sum()
    ax[0].step(ctr, tm, where="mid", color="#440154", lw=2,
               label="loa-0 Perks template (89 events)")
    for fam, c in zip(FAMILIES, ("#31688e", "#35b779", "#e16462")):
        h = figdata[fam]["hostless"].sum(axis=(1, 2)).astype(float)
        n = figdata[fam]["no_abs"].sum(axis=(1, 2)).astype(float)
        ax[0].step(ctr, h / h.sum(), where="mid", color=c, lw=1.2, label=fam)
        ax[0].step(ctr, n / n.sum(), where="mid", color=c, lw=1.0, ls=":")
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"observed $\log N_{\rm HI}$")
    ax[0].set_ylabel("share of the FP population")
    ax[0].set_title(r"$\hat N$ marginal (solid: hostless; dotted: no absorber)",
                    fontsize=9)
    ax[0].legend(fontsize=7)
    # S/N marginal
    ss = np.arange(len(LIVE_S))
    tms = mu.sum(0)[list(LIVE_S)]; tms = tms / tms.sum()
    ax[1].plot(ss, tms, "o-", color="#440154", lw=2, label="loa-0 template")
    for fam, c in zip(FAMILIES, ("#31688e", "#35b779", "#e16462")):
        h = figdata[fam]["hostless"].sum(axis=(0, 1))[list(LIVE_S)].astype(float)
        ax[1].plot(ss, h / h.sum(), "s--", color=c, lw=1.2, label=fam)
    ax[1].set_xticks(ss); ax[1].set_xticklabels([SNR_TEX[s] for s in LIVE_S])
    ax[1].set_xlabel("S/N stratum"); ax[1].set_ylabel("share")
    ax[1].set_title("S/N marginal")
    ax[1].legend(fontsize=7)
    fig.suptitle("FP calibration target: loa-0 twin template vs mock hostless "
                 "census", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(figdir, "fig1_shape_marginals.png")
    fig.savefig(p1); plt.close(fig); print("[fig]", p1)

    # Fig 2 — the four (N-hat x S/N) shapes side by side (2LPT-0)
    nhat_full = np.asarray(L["nhat_edges_full"], float)
    g2378 = np.asarray(L["grid_2378_lyaonly"], float)
    panels = [
        (r"loa-0 Perks template""\n"r"(89 events, $\hat N\geq 19.5$)",
         mu / mu.sum(), nh),
        ("mock hostless census\n(2LPT-0, A0v2)",
         figdata["2lpt0"]["hostless"].sum(1) / figdata["2lpt0"]["hostless"].sum(),
         nh),
        ("mock: NO absorber of any $N$\nin the window (2LPT-0)",
         figdata["2lpt0"]["no_abs"].sum(1) / figdata["2lpt0"]["no_abs"].sum(),
         nh),
        ("loa-0 2,378-event product\n(same selection, no $N$ floor)",
         g2378 / g2378.sum(), nhat_full)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for axp, (ttl, G, edges) in zip(axes, panels):
        Gp = np.where(G > 0, G, np.nan)
        im = axp.pcolormesh(edges, np.arange(9), np.log10(Gp).T,
                            cmap="viridis", vmin=-6, vmax=-1)
        axp.set_title(ttl, fontsize=8)
        axp.set_xlabel(r"observed $\log N_{\rm HI}$")
        axp.set_yticks(np.arange(8) + 0.5)
        axp.set_yticklabels([r"$<1$", "1--2", "2--3", "3--4", "4--5", "5--6",
                             "6--7", r"$\geq 7$"], fontsize=6)
        axp.set_ylim(2, 8)
        fig.colorbar(im, ax=axp, label=r"$\log_{10}$ share")
    axes[0].set_ylabel("S/N stratum")
    fig.tight_layout()
    p2 = os.path.join(figdir, "fig2_Nhat_x_SNR_shapes.png")
    fig.savefig(p2); plt.close(fig); print("[fig]", p2)

    # Fig 3 — decomposition bars per N-hat group
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    gnames = list(SEL.group_rows().keys())
    gtex = [r"$19.5$--$20.0$", r"$20.0$--$20.3$", r"$\geq 20.3$"]
    for axp, fam in zip(axes, FAMILIES):
        rec = T["families"][fam]["by_Nhat_group"]
        x = np.arange(len(gnames)); w = 0.38
        na = [rec[g]["hostless_no_absorber"] for g in gnames]
        tk = [rec[g]["hostless_taken"] for g in gnames]
        tp = [rec[g]["template_t0"] for g in gnames]
        axp.bar(x - w / 2, na, w, color="#31688e", label="no absorber (twin-like)")
        axp.bar(x - w / 2, tk, w, bottom=na, color="#e16462",
                label="host taken (matching artefact)")
        axp.bar(x + w / 2, tp, w, color="#440154",
                label=r"loa-0 template, $t_K=0$")
        axp.set_xticks(x); axp.set_xticklabels(gtex, fontsize=7)
        axp.set_yscale("log"); axp.set_title(fam, fontsize=9)
        axp.set_ylabel("counts on the survey grid")
    axes[0].legend(fontsize=6)
    fig.suptitle("Mock hostless census decomposition vs the loa-0 template "
                 "(no adoption; no fitted $t_K$)", fontsize=10)
    fig.tight_layout()
    p3 = os.path.join(figdir, "fig3_decomposition_bars.png")
    fig.savefig(p3); plt.close(fig); print("[fig]", p3)

    # Fig 4 — the 2,378-event N-hat tail and what it says above 19.5
    fig, axp = plt.subplots(figsize=(6.2, 4.0))
    edges = np.round(np.arange(17.2, 22.5, 0.1), 3)
    nhi2378 = np.asarray(L["nhi_2378"], float)
    h = np.histogram(nhi2378, bins=edges)[0]
    c = 0.5 * (edges[:-1] + edges[1:])
    axp.step(c, np.maximum(h, 1e-1), where="mid", color="#31688e",
             label="loa-0 2,378-event product")
    slope = T["product_89_vs_2378"][
        "poisson_loglinear_slope_dex_per_dex_19p0_20p1"]
    sd = T["product_89_vs_2378"]["poisson_loglinear_slope_sd"]
    m = (edges[:-1] >= 19.0 - 1e-9) & (edges[:-1] < 20.1 - 1e-9)
    # re-draw the SAME Poisson (log-link) fit the tables report
    x = c[m]; y = h[m].astype(float)
    X = np.column_stack([np.ones_like(x), x - 19.55])
    beta = np.array([np.log(max(y.mean(), 1e-9)), -1.5 * np.log(10.0)])
    for _ in range(60):
        mu_ = np.exp(X @ beta)
        beta = np.linalg.solve(X.T @ np.diag(mu_) @ X,
                               X.T @ np.diag(mu_) @ (X @ beta
                                                     + (y - mu_) / mu_))
    Xall = np.column_stack([np.ones_like(c), c - 19.55])
    axp.plot(c, np.exp(Xall @ beta), "--", color="#440154",
             label=("Poisson log-linear fit 19.0--20.1 "
                    rf"(slope ${slope:.2f}\pm{sd:.2f}$/dex)"))
    axp.axvline(19.5, color="k", lw=0.8, ls=":")
    axp.text(19.56, 30, r"\texttt{fp\_counts} floor", fontsize=7, rotation=90)
    axp.set_yscale("log"); axp.set_ylim(0.1, 1e3)
    axp.set_xlabel(r"observed $\log N_{\rm HI}$")
    axp.set_ylabel("loa-0 FP events per 0.1 dex")
    axp.legend(fontsize=7)
    axp.set_title("The 2,378-event product = the 89-event block "
                  r"$\oplus$ the sub-19.5 tail", fontsize=9)
    fig.tight_layout()
    p4 = os.path.join(figdir, "fig4_loa0_2378_tail.png")
    fig.savefig(p4); plt.close(fig); print("[fig]", p4)
    return [p1, p2, p3, p4]


if __name__ == "__main__":
    main()
