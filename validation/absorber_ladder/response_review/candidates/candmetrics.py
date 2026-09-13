#!/usr/bin/env python
"""candmetrics.py -- the section 3 metric battery and the section 5 opening
criterion of the sealed response-family rule.

Every metric is computed on the DELIVERED FIXED OBJECT rows P[b,s,K,c] (which
is what the fold consumes) and, where the sealed rule asks for a per-event
score, also on the conditional model at the held-out event's own covariates.
No metric reads or produces a closure number.

ENV: gpdla-hbi (numpy only).
"""
from __future__ import annotations

import numpy as np

import candlib as CL

LOGFLOOR = 1e-300


# ===========================================================================
# per-event and per-row predictive scores
# ===========================================================================
def event_logp_from_rows(rows, ev, mask):
    """log P(c_i | b_i, s_i, K_i) of the FIXED object for the masked events."""
    idx = np.where(mask)[0]
    p = rows[ev["b_i"][idx], ev["s_i"][idx], ev["K_i"][idx], ev["c_i"][idx]]
    return np.log(np.maximum(p, LOGFLOOR))


def paired_diff(a, b):
    """Paired mean difference of two per-event score vectors, with its SE."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(n)),
                n=int(n), sum=float(d.sum()))


def row_paired_diff(a, b, rid, n_rows):
    """Per-row-cell paired difference, aggregated weighted by held-out counts
    (the sealed rule's 'per-row-cell version weighted by held-out counts')."""
    d = np.asarray(a, float) - np.asarray(b, float)
    s = np.bincount(rid, weights=d, minlength=n_rows)
    n = np.bincount(rid, minlength=n_rows).astype(float)
    ok = n > 0
    m = s[ok] / n[ok]
    w = n[ok]
    wm = float(np.sum(w * m) / np.sum(w))
    # SE of the weighted mean across rows (between-row variability)
    var = float(np.sum(w * (m - wm) ** 2) / np.sum(w))
    se = float(np.sqrt(var * np.sum(w ** 2)) / np.sum(w))
    return dict(wmean=wm, se_between_rows=se, n_rows=int(ok.sum()),
                row_mean=m, row_n=w, row_sum=s[ok])


# ===========================================================================
# row deviance / KL
# ===========================================================================
def row_table(rows, counts, geom, min_events=20):
    """Per-(b,s,K) held-out deviance, KL, moment and leakage residuals."""
    import opmetrics as OM
    ccen = geom["ccen"]; nt = geom["ntrue"]
    recs = []
    B, S, K, C = counts.shape
    for b in range(B):
        lo, hi = nt[b], nt[b + 1]
        for s in range(S):
            for k in range(K):
                n = float(counts[b, s, k].sum())
                if n < min_events:
                    continue
                ph = counts[b, s, k] / n
                pm = rows[b, s, k]
                pm = pm / max(pm.sum(), LOGFLOOR)
                nz = ph > 0
                kl = float(np.sum(ph[nz] * np.log(
                    ph[nz] / np.maximum(pm[nz], LOGFLOOR))))
                dev = 2.0 * n * kl
                em = OM.row_stats(counts[b, s, k], ccen, lo, hi)
                mm = OM.row_stats(pm, ccen, lo, hi)
                tail_e = float(ph[np.abs(ccen - geom["bcen"][b]) > 0.3].sum())
                tail_m = float(pm[np.abs(ccen - geom["bcen"][b]) > 0.3].sum())
                recs.append(dict(
                    b=b, s=s, k=k, n=n, b_lo=float(lo), b_hi=float(hi),
                    kl=kl, deviance=dev, dev_per_dof=dev / max(C - 1, 1),
                    mean_emp=em[0], mean_mod=mm[0], sd_emp=em[1],
                    sd_mod=mm[1], skew_emp=em[2], skew_mod=mm[2],
                    d_mean=mm[0] - em[0],
                    r_sd=(mm[1] / em[1] - 1.0) if em[1] > 0 else np.nan,
                    d_skew=mm[2] - em[2], d_down=mm[3] - em[3],
                    d_in=mm[4] - em[4], d_up=mm[5] - em[5],
                    tail_emp=tail_e, tail_mod=tail_m,
                    d_tail=tail_m - tail_e))
    return recs


def wmean(recs, key, wkey="n", sel=None):
    rr = recs if sel is None else [r for r in recs if sel(r)]
    if not rr:
        return float("nan")
    v = np.array([r[key] for r in rr], float)
    w = np.array([r[wkey] for r in rr], float)
    g = np.isfinite(v)
    if not g.any():
        return float("nan")
    return float(np.sum(w[g] * v[g]) / np.sum(w[g]))


# ===========================================================================
# boundary-crossing masses (sealed section 3 / section 5(ii))
# ===========================================================================
def boundary_table(rows, counts, geom, boundaries=CL.BOUNDARIES,
                   min_events=200):
    """Per-row crossing mass at each boundary: predicted - held-out, with the
    binomial SE.  A row whose latent bin centre lies BELOW the boundary has
    its crossing mass = the predicted mass at observed bins ABOVE it, and
    vice versa (predeclaration I5(ii)).  The SE uses the Jeffreys-adjusted
    p~ = (k + 1/2)/(n + 1) so that a row with k = 0 still has a finite SE
    (numerical amendment, disclosed)."""
    ccen = geom["ccen"]; bcen = geom["bcen"]
    B, S, K, C = counts.shape
    out = {}
    for beta in boundaries:
        recs = []
        up = ccen >= beta
        for b in range(B):
            sel = up if bcen[b] < beta else ~up
            for s in range(S):
                for k in range(K):
                    n = float(counts[b, s, k].sum())
                    if n < min_events:
                        continue
                    kcnt = float(counts[b, s, k][sel].sum())
                    p_e = kcnt / n
                    p_m = float(rows[b, s, k][sel].sum())
                    pt = (kcnt + 0.5) / (n + 1.0)
                    se = float(np.sqrt(pt * (1.0 - pt) / n))
                    recs.append(dict(b=b, s=s, k=k, n=n, p_emp=p_e,
                                     p_mod=p_m, d=p_m - p_e, se=se,
                                     z=(p_m - p_e) / max(se, 1e-12)))
        nbad = sum(1 for r in recs if abs(r["z"]) > 3.0)
        out[f"{beta:.1f}"] = dict(
            n_rows=len(recs), n_bad_gt3se=nbad,
            frac_bad=(nbad / len(recs)) if recs else float("nan"),
            wmean_d=wmean(recs, "d") if recs else float("nan"),
            max_abs_z=float(max((abs(r["z"]) for r in recs), default=np.nan)),
            records=recs)
    return out


# ===========================================================================
# PIT / CDF calibration
# ===========================================================================
def pit_hist(rows, ev, mask, nbin=20, seed=20260913):
    """Randomised PIT of the held-out events under the fixed object."""
    idx = np.where(mask)[0]
    p = rows[ev["b_i"][idx], ev["s_i"][idx], ev["K_i"][idx]]
    cdf = np.cumsum(p, axis=1)
    ci = ev["c_i"][idx]
    lo = np.where(ci > 0, cdf[np.arange(len(ci)), np.maximum(ci - 1, 0)], 0.0)
    pc = p[np.arange(len(ci)), ci]
    rs = np.random.RandomState(seed)
    u = lo + rs.uniform(size=len(ci)) * pc
    h, edges = np.histogram(np.clip(u, 0, 1), bins=nbin, range=(0, 1))
    exp = len(u) / nbin
    return dict(hist=h.tolist(), edges=edges.tolist(),
                chi2=float(np.sum((h - exp) ** 2 / exp)), dof=int(nbin - 1),
                ks=float(np.max(np.abs(np.sort(u) -
                                       (np.arange(len(u)) + 0.5) / len(u)))))


# ===========================================================================
# transport to the London-0 / Saclay-0 measured operators
# ===========================================================================
def transfer_kl(rows, ops_path, geom, min_events=20):
    """Weighted row KL( M_true || P_cand ) against a family's measured
    operator, weighted by that family's N_match row counts."""
    z = np.load(ops_path, allow_pickle=True)
    Mt = np.asarray(z["M_true_sKcb"], float)            # (S, K, C, B)
    Nm = np.asarray(z["N_match_cksb"], float)           # (C, kf, S, B)
    kz = np.asarray(z["kz_to_K"], int)
    S, K, C, B = Mt.shape
    nK = np.zeros((S, K, B))
    for kf in range(len(kz)):
        nK[:, kz[kf], :] += Nm[:, kf, :, :].sum(axis=0)
    tot_w = 0.0; acc = 0.0; recs = []
    for b in range(B):
        for s in range(S):
            for k in range(K):
                n = float(nK[s, k, b])
                q = Mt[s, k, :, b]
                if n < min_events or q.sum() <= 0:
                    continue
                q = q / q.sum()
                p = rows[b, s, k]; p = p / max(p.sum(), LOGFLOOR)
                nz = q > 0
                kl = float(np.sum(q[nz] * np.log(q[nz] /
                                                 np.maximum(p[nz], LOGFLOOR))))
                acc += n * kl; tot_w += n
                recs.append(dict(b=b, s=s, k=k, n=n, kl=kl))
    return dict(wmean_kl=(acc / tot_w) if tot_w > 0 else float("nan"),
                n_rows=len(recs), n_events=tot_w,
                median_kl=float(np.median([r["kl"] for r in recs]))
                if recs else float("nan"), records=recs)


def transfer_moments(rows, ops_path, geom, min_events=20):
    """Moment residuals of a fixed object against a family's MEASURED
    operator ``M_true_sKcb``, weighted by that family's N_match row counts.

    This is the SAME comparison basis the first absorber ladder used to quote
    R1c's row skew -0.25 at b >= 21.3, so the sealed section 5(iii) tolerance
    can be read on a comparable footing alongside the held-out version.
    """
    import opmetrics as OM
    z = np.load(ops_path, allow_pickle=True)
    Mt = np.asarray(z["M_true_sKcb"], float)
    Nm = np.asarray(z["N_match_cksb"], float)
    kz = np.asarray(z["kz_to_K"], int)
    S, K, C, B = Mt.shape
    nK = np.zeros((S, K, B))
    for kf in range(len(kz)):
        nK[:, kz[kf], :] += Nm[:, kf, :, :].sum(axis=0)
    ccen = geom["ccen"]; nt = geom["ntrue"]
    recs = []
    for b in range(B):
        lo, hi = nt[b], nt[b + 1]
        for s in range(S):
            for k in range(K):
                n = float(nK[s, k, b])
                if n < min_events or Mt[s, k, :, b].sum() <= 0:
                    continue
                em = OM.row_stats(Mt[s, k, :, b], ccen, lo, hi)
                mm = OM.row_stats(rows[b, s, k], ccen, lo, hi)
                recs.append(dict(b=b, s=s, k=k, n=n, b_lo=float(lo),
                                 b_hi=float(hi),
                                 d_mean=mm[0] - em[0],
                                 r_sd=(mm[1] / em[1] - 1.0) if em[1] > 0
                                 else np.nan,
                                 d_skew=mm[2] - em[2],
                                 d_up=mm[5] - em[5], d_down=mm[3] - em[3]))
    hi_sel = (lambda r: r["b_lo"] >= 21.3 - 1e-9)
    lo_sel = (lambda r: r["b_hi"] <= 19.7 + 1e-9)
    return dict(n_rows=len(recs),
                d_mean=wmean(recs, "d_mean"), r_sd=wmean(recs, "r_sd"),
                d_skew=wmean(recs, "d_skew"),
                d_skew_b_ge_21p3=wmean(recs, "d_skew", sel=hi_sel),
                d_mean_b_ge_21p3=wmean(recs, "d_mean", sel=hi_sel),
                r_sd_b_ge_21p3=wmean(recs, "r_sd", sel=hi_sel),
                d_skew_b_lt_19p7=wmean(recs, "d_skew", sel=lo_sel),
                d_up=wmean(recs, "d_up"), d_down=wmean(recs, "d_down"))


# ===========================================================================
# the sealed section 5 criterion
# ===========================================================================
def criterion_section5(ll_cand, ll_ref, rid, n_rows, fold_data, geom,
                       min_big=200, skew_tol=0.10, skew_min_events=20):
    """Verbatim implementation of the sealed section 5 failure criterion.

    ``fold_data`` is a list of (rows, counts) pairs, one per CV fold: each
    fold's HELD-OUT counts are scored against the object fitted on that fold's
    TRAINING half, and the per-row records of both folds are pooled before the
    thresholds are applied (the sealed rule's 'both folds, averaged').

    Returns dict(fail=bool, which=[...], detail={...}).
    """
    pd = paired_diff(ll_cand, ll_ref)
    d = np.asarray(ll_cand, float) - np.asarray(ll_ref, float)
    rsum = np.bincount(rid, weights=d, minlength=n_rows)
    rn = np.bincount(rid, minlength=n_rows).astype(float)
    neg = np.minimum(rsum, 0.0)
    tot_def = float(neg.sum())
    big = rn >= min_big
    frac_big = float(neg[big].sum() / tot_def) if tot_def < 0 else float("nan")
    i_fail = bool(pd["mean"] < -2.0 * pd["se"] and tot_def < 0 and
                  frac_big >= 0.5)

    bt = {}
    for rows, counts in fold_data:
        b1 = boundary_table(rows, counts, geom, min_events=min_big)
        for k, v in b1.items():
            bt.setdefault(k, []).extend(v["records"])
    bsum = {}
    for k, recs in bt.items():
        nbad = sum(1 for r in recs if abs(r["z"]) > 3.0)
        bsum[k] = dict(n_rows=len(recs), n_bad_gt3se=nbad,
                       frac_bad=(nbad / len(recs)) if recs else float("nan"),
                       wmean_d=wmean(recs, "d") if recs else float("nan"),
                       max_abs_z=float(max((abs(r["z"]) for r in recs),
                                           default=np.nan)))
    ii_which = [k for k, v in bsum.items()
                if v["n_rows"] > 0 and v["frac_bad"] > 0.20]
    ii_fail = bool(ii_which)

    recs = []
    for rows, counts in fold_data:
        recs += row_table(rows, counts, geom, min_events=skew_min_events)
    hi = [r for r in recs if r["b_lo"] >= 21.3 - 1e-9]
    sk = wmean(hi, "d_skew") if hi else float("nan")
    iii_fail = bool(not np.isfinite(sk) or abs(sk) > skew_tol)

    which = ([f"(i) dll={pd['mean']:+.4f} +- {pd['se']:.4f}, "
              f"{100*frac_big:.0f}% of the deficit in rows >= {min_big}"]
             if i_fail else [])
    if ii_fail:
        which.append("(ii) boundaries " + ",".join(ii_which) +
                     f" mispredicted > 3 SE on > 20% of rows >= {min_big}")
    if iii_fail:
        which.append(f"(iii) row skew at b >= 21.3 = {sk:+.3f} "
                     f"(tolerance +-{skew_tol})")
    return dict(fail=bool(i_fail or ii_fail or iii_fail), which=which,
                detail=dict(dll_mean=pd["mean"], dll_se=pd["se"],
                            deficit_total=tot_def, deficit_frac_big=frac_big,
                            i_fail=i_fail, ii_fail=ii_fail,
                            ii_boundaries=ii_which, iii_fail=iii_fail,
                            skew_ge_21p3=sk, n_rows_skew=len(hi)),
                boundary=bsum, row_records=recs)
