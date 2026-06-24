"""track_c_eddington_verify.py — REDUCE-ONLY Eddington-bias falsifier for Track-C.

Decides whether the kernel column's strong LEFT skew (observed-conditional
p(x_true|xhat) skew −1.5..−1.9) is EDDINGTON bias (steep f ⊛ symmetric forward
response) rather than a per-system N_HI over-estimation.

Inference is byte-frozen: this reads ONLY the truth-match (xhat, xtrue, z, SNR,
z_qso) from the SAME op-set the znz cache uses, via build_ingredients +
measure_znz_response.  No commits.  Prints a structured report to stdout.
"""
from __future__ import annotations

import os
import sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from CDDF_analysis.hbi.ab_loa0_fp_baseline import (
    build_ingredients, DEF_CAT, DEF_TRUTH, DEF_BAL, DEF_KERNEL, DEF_LOA0_PRODUCT,
)


def _skew(a, w=None):
    """Standardized third moment (Fisher skewness), optional weights."""
    a = np.asarray(a, float)
    if w is None:
        w = np.ones_like(a)
    w = np.asarray(w, float)
    sw = w.sum()
    if sw <= 0 or len(a) < 3:
        return np.nan
    m1 = np.sum(w * a) / sw
    d = a - m1
    m2 = np.sum(w * d * d) / sw
    if m2 <= 0:
        return np.nan
    m3 = np.sum(w * d * d * d) / sw
    return float(m3 / m2 ** 1.5)


class _Args:
    """Mimic the argparse namespace build_ingredients expects (ab_loa0 defaults)."""
    catalog_dir = DEF_CAT
    truth = DEF_TRUTH
    bal_cat = DEF_BAL
    molly_tsv = None
    kernel = DEF_KERNEL
    loa0_product = DEF_LOA0_PRODUCT
    out = "/tmp/track_c_eddington"
    mockdir = None
    zbins = "2.0,2.5,3.0,3.5"
    report_limits = "20.0,20.3,20.6"
    family = "bspbody"
    fit_floor = 19.5
    fit_ceil = 99.0
    lambda_bspbody = 30.0
    lam_rf_min = 1025.0
    edge_slope_lam = 40.0
    gl_nodes = 1
    host_truth_floor = 19.0


def main():
    args = _Args()
    print("[eddington] building ingredients (frozen op-set)...", flush=True)
    ing = build_ingredients(args, fp_estimator="purity_mixture")
    cfg = ing["cfg"]
    cat_cut = ing["cat_cut"]
    good_mask = ing["good_mask"]

    # ---- exact op-set from measure_znz_response (lya-only, S2N>2, P_DLA>0.99) ----
    s2n_all = np.asarray(cat_cut["S2N_RED"], float)
    pdla_all = np.asarray(cat_cut["P_DLA"], float)
    op = (s2n_all > cfg.snr_min) & (pdla_all > cfg.p_dla_min) & good_mask

    xhat = np.asarray(cat_cut["NHI"], float)[op]
    host_col = "NHI_TILT_HOST" if "NHI_TILT_HOST" in cat_cut.colnames else "NHI_TRUE"
    xtrue = np.asarray(cat_cut[host_col], float)[op]
    z_dla = np.asarray(cat_cut["Z_DLA"], float)[op]
    z_qso = np.asarray(cat_cut["Z_QSO"], float)[op] if "Z_QSO" in cat_cut.colnames else np.full(op.sum(), np.nan)
    s2n = s2n_all[op]

    tp = np.isfinite(xtrue)
    xhat = xhat[tp]; xtrue = xtrue[tp]; z_dla = z_dla[tp]; z_qso = z_qso[tp]; s2n = s2n[tp]
    # The znz fit floor is xhat>=19.5; replicate so we describe the SAME population.
    keep = xhat >= 19.5
    xhat = xhat[keep]; xtrue = xtrue[keep]; z_dla = z_dla[keep]; z_qso = z_qso[keep]; s2n = s2n[keep]

    dx = xhat - xtrue          # xhat - xtrue (znz convention)
    print(f"[eddington] N (TP, xhat>=19.5) = {len(xhat):,}", flush=True)
    print(f"[eddington] xhat range {xhat.min():.2f}..{xhat.max():.2f}; "
          f"xtrue range {xtrue.min():.2f}..{xtrue.max():.2f}", flush=True)
    print(f"[eddington] host_col = {host_col}; SNR med={np.median(s2n):.2f}; "
          f"z_qso med={np.nanmedian(z_qso):.3f}", flush=True)

    # ================================================================
    # PART 0: the observed-conditional left-skew (the claim to reproduce)
    #   p(x_true | xhat-cell): bin in OBSERVED xhat, skew of x_true.
    # ================================================================
    print("\n=== PART 0: OBSERVED-CONDITIONAL skew  p(x_true | xhat) ===")
    print("  (bin in OBSERVED xhat; report skew of x_true within the cell)")
    obs_cells = [(19.5, 19.7), (19.7, 20.0), (20.0, 20.3), (20.3, 20.5),
                 (20.5, 21.0), (21.0, 23.0)]
    print(f"  {'xhat-cell':<16}{'N':>8}{'skew(x_true)':>14}{'skew(dx)':>12}")
    for lo, hi in obs_cells:
        m = (xhat >= lo) & (xhat < hi)
        n = int(m.sum())
        sk_xt = _skew(xtrue[m]) if n >= 30 else np.nan
        sk_dx = _skew(dx[m]) if n >= 30 else np.nan
        print(f"  [{lo:.1f},{hi:.1f}){'':<6}{n:>8}{sk_xt:>14.3f}{sk_dx:>12.3f}")

    # ================================================================
    # PART 1: FORWARD response  p(xhat | N_true) binned in TRUE N
    #   For fixed-true-N subsamples, skew of xhat (and of dx).
    #   Orientation: dx = xhat - xtrue. Positive skew of xhat = right tail (UP).
    # ================================================================
    print("\n=== PART 1: FORWARD response  p(xhat | N_true)  binned in TRUE N ===")
    print("  (bin in TRUE N_HI; report skew of xhat and of dx within the cell)")
    print("  EDDINGTON signature: forward (true-binned) ~symmetric |skew|<~0.5")
    fwd_cells = [(20.0, 20.1), (20.3, 20.4), (20.5, 20.6), (21.0, 21.1),
                 (20.0, 20.3), (20.3, 20.5), (20.5, 21.0)]
    print(f"  {'N_true-cell':<16}{'N':>8}{'skew(xhat)':>12}{'skew(dx)':>11}"
          f"{'mean(dx)':>10}{'std(dx)':>9}")
    fwd_summary = {}
    for lo, hi in fwd_cells:
        m = (xtrue >= lo) & (xtrue < hi)
        n = int(m.sum())
        sk_xh = _skew(xhat[m]) if n >= 30 else np.nan
        sk_dx = _skew(dx[m]) if n >= 30 else np.nan
        mdx = float(np.mean(dx[m])) if n else np.nan
        sdx = float(np.std(dx[m])) if n else np.nan
        fwd_summary[(lo, hi)] = (n, sk_xh, sk_dx, mdx, sdx)
        print(f"  [{lo:.1f},{hi:.1f}){'':<6}{n:>8}{sk_xh:>12.3f}{sk_dx:>11.3f}"
              f"{mdx:>10.3f}{sdx:>9.3f}")

    # ASCII histogram at one forward cell (N_true in [20.3,20.4))
    print("\n  -- histogram of xhat at fixed N_true in [20.3,20.4) (forward kernel) --")
    m = (xtrue >= 20.3) & (xtrue < 20.4)
    if m.sum() >= 30:
        vals = xhat[m]
        edges = np.linspace(19.5, 22.0, 26)
        h, _ = np.histogram(vals, bins=edges)
        hmax = max(h.max(), 1)
        for k in range(len(h)):
            c = 0.5 * (edges[k] + edges[k + 1])
            bar = "#" * int(round(40 * h[k] / hmax))
            print(f"    xhat={c:5.2f} |{bar} {h[k]}")
        print(f"    [forward cell N_true∈[20.3,20.4): N={int(m.sum())}, "
              f"mean(xhat)={vals.mean():.3f}, median(xhat)={np.median(vals):.3f}, "
              f"skew(xhat)={_skew(vals):.3f}]")

    # ================================================================
    # PART 2: EDDINGTON CLOSURE
    #   Take the steep truth f(N) on 2LPT-0; forward-scatter each true system's
    #   xhat by RESAMPLING from the MEASURED forward response (true-N-binned dx
    #   bootstrap); bin the synthetic xhat by OBSERVED xhat; measure the
    #   observed-conditional skew of x_true. Does it reproduce the −1.5..−1.9?
    # ================================================================
    print("\n=== PART 2: EDDINGTON CLOSURE (symmetric-forward + steep-f) ===")
    rng = np.random.default_rng(0)
    # Build the empirical forward response in fine TRUE-N bins (dx | N_true).
    Ntrue_edges = np.arange(19.5, 23.01, 0.1)
    Ntrue_mid = 0.5 * (Ntrue_edges[:-1] + Ntrue_edges[1:])
    # For each truth system, draw a synthetic dx from the dx-pool of its true-N bin.
    jtrue_data = np.searchsorted(Ntrue_edges, xtrue, side="right") - 1
    jtrue_data = np.clip(jtrue_data, 0, len(Ntrue_mid) - 1)
    dx_pools = {j: dx[jtrue_data == j] for j in range(len(Ntrue_mid))}

    # The "steep truth f": use the TRUTH N_HI population itself (these xtrue ARE
    # drawn from the steep mock f over the DLA regime). Synthesize xhat by adding a
    # resampled dx from the SAME true-N bin (so the per-N forward response is exactly
    # the measured one, by construction symmetric-or-not as measured).
    syn_xhat = np.full(len(xtrue), np.nan)
    for j in range(len(Ntrue_mid)):
        sel = np.where(jtrue_data == j)[0]
        pool = dx_pools[j]
        if len(sel) == 0:
            continue
        if len(pool) == 0:
            # borrow nearest non-empty pool
            for off in range(1, len(Ntrue_mid)):
                if j - off >= 0 and len(dx_pools[j - off]) > 0:
                    pool = dx_pools[j - off]; break
                if j + off < len(Ntrue_mid) and len(dx_pools[j + off]) > 0:
                    pool = dx_pools[j + off]; break
        draws = rng.choice(pool, size=len(sel), replace=True)
        syn_xhat[sel] = xtrue[sel] + draws

    # Now measure the OBSERVED-conditional skew of x_true under the SYNTHETIC xhat.
    print("  Synthetic xhat = xtrue + resampled dx(from true-N bin).")
    print("  Bin by SYNTHETIC observed xhat; skew of x_true should match PART 0.")
    print(f"  {'syn-xhat-cell':<16}{'N':>8}{'skew(x_true)|obs':>18}")
    closure = {}
    for lo, hi in obs_cells:
        m = (syn_xhat >= lo) & (syn_xhat < hi)
        n = int(m.sum())
        sk = _skew(xtrue[m]) if n >= 30 else np.nan
        closure[(lo, hi)] = (n, sk)
        print(f"  [{lo:.1f},{hi:.1f}){'':<6}{n:>8}{sk:>18.3f}")

    # ----- second closure variant: PURE-SYMMETRIC forward response -----
    # Replace the measured per-N dx by a SYMMETRIZED Gaussian (mean=measured per-N
    # mean dx, std=measured per-N std, skew forced to 0). If THIS still reproduces
    # the observed left-skew, the asymmetry is purely from steep-f Eddington.
    print("\n  -- variant: FORCE symmetric (Gaussian) forward response per true-N --")
    syn_xhat_sym = np.full(len(xtrue), np.nan)
    for j in range(len(Ntrue_mid)):
        sel = np.where(jtrue_data == j)[0]
        pool = dx_pools[j]
        if len(sel) == 0:
            continue
        if len(pool) >= 5:
            mu_j = float(np.mean(pool)); sd_j = float(np.std(pool))
        else:
            mu_j = float(np.mean(dx)); sd_j = float(np.std(dx))
        sd_j = max(sd_j, 1e-3)
        syn_xhat_sym[sel] = xtrue[sel] + rng.normal(mu_j, sd_j, size=len(sel))
    print(f"  {'syn-xhat-cell':<16}{'N':>8}{'skew(x_true)|obs(sym)':>22}")
    for lo, hi in obs_cells:
        m = (syn_xhat_sym >= lo) & (syn_xhat_sym < hi)
        n = int(m.sum())
        sk = _skew(xtrue[m]) if n >= 30 else np.nan
        print(f"  [{lo:.1f},{hi:.1f}){'':<6}{n:>8}{sk:>22.3f}")

    # ================================================================
    # PART 3: SNR + z dependence of the forward response
    # ================================================================
    print("\n=== PART 3: FORWARD-response width/shape vs SNR and z_qso ===")
    print("  (fixed TRUE-N band [20.3,20.5); split by SNR tertile and z_qso tertile)")
    band = (xtrue >= 20.3) & (xtrue < 20.5)
    sb = s2n[band]; zb = z_qso[band]; dxb = dx[band]; xhb = xhat[band]
    if band.sum() >= 90:
        # SNR tertiles
        snr_q = np.quantile(sb, [0, 1/3, 2/3, 1.0])
        print(f"  SNR tertiles edges: {np.round(snr_q,2)}")
        print(f"  {'SNR-bin':<18}{'N':>7}{'mean(dx)':>10}{'std(dx)':>9}{'skew(xhat)':>12}")
        for t in range(3):
            mm_ = (sb >= snr_q[t]) & (sb <= snr_q[t + 1] if t == 2 else sb < snr_q[t + 1])
            n = int(mm_.sum())
            print(f"  [{snr_q[t]:.1f},{snr_q[t+1]:.1f}]{'':<3}{n:>7}"
                  f"{np.mean(dxb[mm_]):>10.3f}{np.std(dxb[mm_]):>9.3f}"
                  f"{_skew(xhb[mm_]):>12.3f}")
        # z_qso tertiles
        zfin = np.isfinite(zb)
        if zfin.sum() >= 60:
            zq = np.quantile(zb[zfin], [0, 1/3, 2/3, 1.0])
            print(f"  z_qso tertiles edges: {np.round(zq,3)}")
            print(f"  {'z_qso-bin':<18}{'N':>7}{'mean(dx)':>10}{'std(dx)':>9}{'skew(xhat)':>12}")
            for t in range(3):
                mm_ = zfin & (zb >= zq[t]) & (zb <= zq[t + 1] if t == 2 else zb < zq[t + 1])
                n = int(mm_.sum())
                print(f"  [{zq[t]:.2f},{zq[t+1]:.2f}]{'':<2}{n:>7}"
                      f"{np.mean(dxb[mm_]):>10.3f}{np.std(dxb[mm_]):>9.3f}"
                      f"{_skew(xhb[mm_]):>12.3f}")
    else:
        print(f"  too few in band ({int(band.sum())})")

    # ================================================================
    # PART 4: kernel (kappa2d) vs measured forward response (width/shape)
    # ================================================================
    print("\n=== PART 4: kappa2d kernel width vs measured forward-response width ===")
    print("  Is the GP SIR posterior kernel kappa2d consistent with the measured")
    print("  forward response? (too narrow => under-corrects Eddington up-scatter)")
    kpath = args.kernel
    try:
        d = np.load(kpath, allow_pickle=True)
        kappa = d["kappa"]  # (n_obs, n_nbins, n_zf) posterior mass per fine bin
        print(f"  kappa shape = {kappa.shape}; sum per obj (mean) "
              f"= {kappa.reshape(kappa.shape[0],-1).sum(1).mean():.4f}")
        # The fine logN grid: reconstruct from cfg
        from CDDF_analysis.hbi.cddf_catalog_hbi import build_fine_grid, _fine_z_grid
        logN_lo, logN_hi, N_b, dN_b = build_fine_grid(cfg)
        mids = 0.5 * (np.asarray(logN_lo, float) + np.asarray(logN_hi, float))
        n_obs_k = kappa.shape[0]
        # per-object posterior over N (marginalize z): mass-weighted mean + std
        kN = kappa.reshape(n_obs_k, len(mids), -1).sum(axis=2)  # (n_obs, n_nbins)
        tot = kN.sum(axis=1)
        good = tot > 0
        mu_post = np.where(good, (kN * mids[None, :]).sum(1) / np.where(good, tot, 1), np.nan)
        var_post = np.where(good,
                            (kN * (mids[None, :] ** 2)).sum(1) / np.where(good, tot, 1) - mu_post ** 2,
                            np.nan)
        sd_post = np.sqrt(np.clip(var_post, 0, None))
        print(f"  per-object POSTERIOR width (mass-wtd std over N): "
              f"median={np.nanmedian(sd_post):.3f} dex, "
              f"mean={np.nanmean(sd_post):.3f}")
        print(f"  [compare to measured FORWARD response std(dx) ~ "
              f"{fwd_summary[(20.3,0.4+20.0)][4] if (20.3,20.4) in fwd_summary else 'n/a'}]")
        for cell in [(20.0,20.3),(20.3,20.5),(20.5,21.0)]:
            if cell in fwd_summary:
                print(f"    forward std(dx) | N_true {cell} = {fwd_summary[cell][4]:.3f}")
        # posterior skewness per object (is the kernel itself skewed?)
        m3 = np.where(good, (kN * ((mids[None,:]-mu_post[:,None])**3)).sum(1)/np.where(good,tot,1), np.nan)
        sk_post = np.where(sd_post>0, m3/np.where(sd_post>0,sd_post,1)**3, np.nan)
        print(f"  per-object POSTERIOR skew (kernel's own): "
              f"median={np.nanmedian(sk_post):.3f}, mean={np.nanmean(sk_post):.3f}")
    except Exception as e:
        print(f"  [kernel load/analysis failed: {e}]")

    print("\n[eddington] DONE.", flush=True)


if __name__ == "__main__":
    main()
