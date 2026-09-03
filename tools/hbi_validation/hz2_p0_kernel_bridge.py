#!/usr/bin/env python
"""Bridge between the P0 recentered forward-MAP estimator and the HZ2 HBI at the DLA threshold (PI ruling 2026-09-03 evening §7, §10).

Two response kernels are folded through two latent populations on the SAME sightline population (dX[k,s] of the HZ2 real pack, which
equals P0's X_tot) with the SAME completeness (HZ2's molly block sigmoid(eta_hat) · g_bk; psi_c ≈ 0) so that only the kernel and the
population differ:
  kernels     E      : real-spectrum Candidate E (pack `adopted_masses_override`, P(x̂ in c | detected, b, sr, zr)), in-grid fraction phi
              2LPT-0 : the frozen low-z forward kernel of the Track-C/P0 estimator (`forward_response_2lpt0.npz`, smoothed-empirical family,
                       its z>2.96 block), P(x̂ >= T | N, snr) = ∫_{x̂>=T} p(x̂|N,snr,z) dx̂ (density integrated numerically on an x̂ grid)
  populations HZ2    : pooled posterior-median f(b,k) on the 16-bin latent basis 19.0–22.4
              P0     : the archived MAP map_fbk (52 fine bins 17.2–22.4 × 3 coarse z) of the P0 product
For each (kernel, population) at T = 20.3 and 20.0: stay, up, down (expected detected rows), the migration factor R = (stay+up)/(stay+down)
and the implied 'de-migration' of a completeness-corrected count. The 2×2 table is order-independent (two factors, both orders shown).
Also: P0's patched molly C cells vs HZ2's sigmoid(eta_hat) per (row, cell).
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from CDDF_analysis.hbi_mcmc.pack import load_pack  # noqa: E402
from CDDF_analysis.hbi_mcmc.forward import build_consts  # noqa: E402

REPAIR = "/home/mfho/wt_highz_repair"


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--pooled-fdraws", required=True); ap.add_argument("--p0-npz", required=True)
    ap.add_argument("--p0-json", required=True); ap.add_argument("--forward-npz", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pk = load_pack(a.pack, allow_nonstandard_grid=True); consts = build_consts(pk, resp_clamp="both")
    eta = np.asarray(consts.eta_hat); b2c = np.asarray(consts.b_to_cell); g = np.asarray(consts.g_bk); dX = np.asarray(consts.dX)
    nt = np.asarray(pk.ntrue_edges, float); ne = np.asarray(pk.nhat_edges, float); dN = np.diff(nt); Nc = 0.5 * (nt[:-1] + nt[1:])
    snr = np.asarray(pk.snr_edges, float); S = len(snr) - 1; Kf = dX.shape[0]; zf = np.asarray(pk.zf_edges, float); zc = 0.5 * (zf[:-1] + zf[1:])
    s2sr = np.asarray(consts.s_to_sresp); K2zr = np.asarray(consts.K_to_zresp); kz = np.asarray(consts.kz_to_K)
    M = np.asarray(pk.adopted_masses_override, float)                       # (SR, ZR, C, B)
    phi = M.sum(axis=2)                                                      # (SR, ZR, B)
    # ---- population HZ2: pooled posterior median f(b,k)
    fz = np.load(a.pooled_fdraws); fH = np.median(fz["f"], axis=0)          # (B, Kf)
    # ---- population P0: MAP map_fbk (52, 3) on 17.2–22.4 fine bins; map to the 16-bin basis by dex-overlap (N-integrated) and to fine z by coarse block
    p0 = np.load(a.p0_npz, allow_pickle=True); lo, hi, fbk = p0["logN_lo"], p0["logN_hi"], p0["map_fbk"]; zb = p0["zbins"]
    dNp = 10.0 ** hi - 10.0 ** lo                                            # linear-N widths (P0's f is per unit N)
    fP = np.zeros((len(Nc), Kf)); fP_below = np.zeros(Kf)                    # P0 systems below the HZ2 basis floor (17.2–19.0) kept separately
    kblock = np.clip(np.searchsorted(zb, zc, side="right") - 1, 0, len(zb) - 2)
    for b in range(len(Nc)):
        ov = np.clip(np.minimum(hi, nt[b + 1]) - np.maximum(lo, nt[b]), 0, None) / (hi - lo)   # dex overlap fraction of each fine bin with basis bin b
        for k in range(Kf):
            fP[b, k] = (fbk[:, kblock[k]] * dNp * ov).sum() / dN[b]                              # per-dex density on the basis bin
    for k in range(Kf):
        fP_below[k] = (fbk[:, kblock[k]] * dNp * (hi <= nt[0] + 1e-9)).sum()                     # systems per unit X below 19.0 (per k)
    # ---- kernel E crossing probabilities per (s, k, b): P(x̂ >= T | detected, b) and in-grid phi
    def E_cross(T):
        cm = ne[:-1] >= T - 1e-9; out = np.zeros((S, Kf, len(Nc))); ph = np.zeros((S, Kf, len(Nc)))
        for s in range(S):
            for k in range(Kf):
                m = M[s2sr[s], K2zr[kz[k]]]; out[s, k] = m[cm].sum(axis=0); ph[s, k] = m.sum(axis=0)
        return out, ph
    # ---- kernel 2LPT-0 (frozen P0 kernel): density integrals on an x̂ grid; z block = the kernel's own z>2.96 block (zqso=4.5 for every k)
    sys.path.insert(0, REPAIR)
    from CDDF_analysis.hbi.znz_kernel import load_forward_response
    frm = load_forward_response(a.forward_npz)
    xg = np.arange(17.0, 23.5, 0.01); snr_c = np.array([0.5 * (snr[i] + (snr[i + 1] if np.isfinite(snr[i + 1]) else snr[i] + 2.0)) for i in range(S)])
    def P0_cross(T, Ncent):
        cm = xg >= T - 1e-9; ing = (xg >= ne[0] - 1e-9) & (xg < ne[-1] + 1e-9)
        out = np.zeros((S, len(Ncent))); ph = np.zeros((S, len(Ncent)))
        for s in range(S):
            for b, N in enumerate(Ncent):
                d = frm.density(xg, np.full(xg.size, N), np.full(xg.size, snr_c[s]), np.full(xg.size, 4.5)); tot = d.sum() * 0.01
                out[s, b] = (d[cm & ing].sum() * 0.01); ph[s, b] = d[ing].sum() * 0.01
                if tot > 0: out[s, b] /= 1.0; ph[s, b] /= 1.0
        return out, ph
    res = {}
    for T in (20.3, 20.0):
        wb = np.clip(nt[1:] - np.maximum(nt[:-1], T), 0.0, None) / dN; wb = np.clip(wb, 0, 1)
        Ecr, Eph = E_cross(T); Pcr, Pph = P0_cross(T, Nc)
        Cc = sig(eta)[:, b2c]                                                # (S, B)
        table = {}
        for kname, cr, ph in (("E", Ecr, Eph), ("2LPT0", np.broadcast_to(Pcr[:, None, :], (S, Kf, len(Nc))), np.broadcast_to(Pph[:, None, :], (S, Kf, len(Nc))))):
            for pname, f in (("HZ2", fH), ("P0", fP)):
                # expected detected in-grid rows from latent bin b: n_det[s,k,b] = C[s,b] g[b,k] f[b,k] dN[b] dX[k,s] * phi ; above-T rows = ... * cr
                base = Cc[:, None, :] * g.T[None, :, :] * f.T[None, :, :] * dN[None, None, :] * dX.T[:, :, None]   # (S, Kf, B)
                det = base * ph; above = base * cr
                stay = (above * wb[None, None, :]).sum(); up = (above * (1 - wb)[None, None, :]).sum(); down = ((det - above) * wb[None, None, :]).sum()
                R = (stay + up) / (stay + down)
                table[f"{kname}|{pname}"] = dict(stay=float(stay), up=float(up), down=float(down), R=float(R), obs_pred_TP=float(stay + up), lat_det=float(stay + down))
        res[f"T{T}"] = table
        print(f"== T = {T}: kernel | population -> stay / up / down ; R = (stay+up)/(stay+down)")
        for k, v in table.items():
            print(f"   {k:10s} stay {v['stay']:7.1f} up {v['up']:7.1f} down {v['down']:6.1f}  R {v['R']:.3f}")
        rE = table["E|HZ2"]["R"] / table["2LPT0|HZ2"]["R"]; rP = table["E|P0"]["R"] / table["2LPT0|P0"]["R"]
        fE = table["E|P0"]["R"] / table["E|HZ2"]["R"]; f2 = table["2LPT0|P0"]["R"] / table["2LPT0|HZ2"]["R"]
        res[f"T{T}"]["kernel_effect_R_E_over_2LPT0"] = dict(at_HZ2_population=rE, at_P0_population=rP); res[f"T{T}"]["population_effect_R_P0_over_HZ2"] = dict(with_E=fE, with_2LPT0=f2)
        print(f"   kernel effect (R_E / R_2LPT0): at HZ2 pop {rE:.3f}, at P0 pop {rP:.3f} | population effect (R_P0pop / R_HZ2pop): with E {fE:.3f}, with 2LPT0 {f2:.3f}")
    # ---- completeness comparison: P0 patched cells vs HZ2 sigmoid(eta_hat)
    J = json.load(open(a.p0_json)); patched = J["metadata"]["calibration"]["r041_patch"]["patched"]; me = np.asarray(pk.molly_nhi_edges, float)
    comp = []
    for c in patched:
        lo_ = float(c["cell"].split("[")[1].split(",")[0]); jc = int(np.argmin(np.abs(me[:-1] - lo_))); row = int(c["snr_row"])
        comp.append(dict(cell_lo=lo_, snr_row=row, C_P0=c["C"], C_HZ2=float(sig(eta)[row, jc]), k=c["k"], n=c["n"]))
    dmax = max(abs(x["C_P0"] - x["C_HZ2"]) for x in comp); res["completeness_compare"] = dict(cells=comp, max_abs_diff=float(dmax))
    print(f"completeness: P0 patched cells vs HZ2 sigmoid(eta_hat): max |ΔC| = {dmax:.4f} over {len(comp)} cells; examples", [(x['cell_lo'], x['snr_row'], round(x['C_P0'],3), round(x['C_HZ2'],3)) for x in comp[:4]])
    res["P0_population_below_19p0_per_X"] = float((fP_below * dX.sum(axis=1)).sum() / dX.sum())
    print("P0 MAP systems per unit X below the HZ2 basis floor (17.2–19.0):", round(res["P0_population_below_19p0_per_X"], 4), "| HZ2 basis latent ≥19.0 per X (HZ2 median):", round(float((fH * dN[:, None] * dX.sum(axis=1)[None, :]).sum() / dX.sum()), 4), "| P0 MAP ≥19.0 per X:", round(float((fP * dN[:, None] * dX.sum(axis=1)[None, :]).sum() / dX.sum()), 4))
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
