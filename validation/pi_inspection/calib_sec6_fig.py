"""SEC 6 figure + tables: the frozen response rows of B (baseline) and E.

Eight representative (b, s) cells -- four latent-N bins x two S/N strata --
each panel showing the HELD-OUT empirical histogram over the 29 observed
N_hat bins (both CV folds pooled, coarse-z blocks pooled), the delivered B
row and the delivered E row, with the 20.0 and 20.3 reporting thresholds
marked.

Read-only: every row is READ from the released / frozen tensors
(rows_unit inside Mg_<cand>_2lpt0.npz), every count is READ from
cv_fold_rows.npz.  No fit, no refit, no sampler.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_common as cc                                        # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402

CAND = os.path.join(cc.LAD, "response_review", "candidates")
SUPP = os.path.join(cc.LAD, "support", "empirical_ops_2lpt0_A0.npz")
OUT = os.path.join(cc.FIGDIR, "fig_sec6_response_rows.png")

BBINS = [5, 8, 11, 12]          # [20.1,20.3) [20.7,20.9) [21.3,21.5) [21.5,21.7)
STRATA = [2, 7]                 # S/N 2-3 and S/N >= 7


def load_rows(cand):
    p = os.path.join(CAND, "Mg_%s_2lpt0.npz" % cand)
    d = np.load(p, allow_pickle=True)
    return d["rows_unit"], d


def main():
    cc.style()
    cv = np.load(os.path.join(CAND, "cv_fold_rows.npz"))
    nt, nh = cv["ntrue_edges"], cv["nhat_edges"]
    held = cv["B__heldout_counts_fold0"] + cv["B__heldout_counts_fold1"]

    # nhat_edges cross-check against the frozen support pack
    if os.path.exists(SUPP):
        sp = np.load(SUPP, allow_pickle=True)
        key = [k for k in sp.files if "nhat" in k.lower() and "edge" in k.lower()]
        if key:
            same = np.allclose(sp[key[0]], nh, atol=0, rtol=0)
            print("nhat_edges identical to %s[%s]: %s" % (SUPP, key[0], same))

    rows = {}
    extras = {}
    for c in ("B", "E", "R0"):
        rows[c], extras[c] = load_rows(c)
    phi = extras["B"]["phi_bsK"]                 # (16, 8, 3) measured
    phi_ref = extras["B"]["phi_ref_pack_gathered"]

    c20 = int(np.where(np.isclose(nh, 20.0))[0][0])
    c203 = int(np.where(np.isclose(nh, 20.3))[0][0])
    ctr = 0.5 * (nh[:-1] + nh[1:])

    fig, axes = plt.subplots(len(STRATA), len(BBINS),
                             figsize=(10.4, 4.9), sharex=True)
    table_rows = []
    for i, s in enumerate(STRATA):
        for j, b in enumerate(BBINS):
            ax = axes[i, j]
            n_k = held[b, s].sum(axis=-1)                # (3,)
            n = float(n_k.sum())
            emp = held[b, s].sum(axis=0)
            emp = emp / max(emp.sum(), 1.0)
            w = n_k / max(n_k.sum(), 1.0)                # display pooling only
            rB = np.einsum("k,kc->c", w, rows["B"][b, s])
            rE = np.einsum("k,kc->c", w, rows["E"][b, s])
            rR = np.einsum("k,kc->c", w, rows["R0"][b, s])
            ax.step(ctr, emp, where="mid", color="0.25", lw=1.0,
                    label="held-out empirical")
            ax.fill_between(ctr, 0, emp, step="mid", color="0.25", alpha=0.13)
            ax.plot(ctr, rB, color=cc.PALETTE[5], lw=1.3, label="B (record)")
            ax.plot(ctr, rE, color=cc.PALETTE[2], lw=0.8, ls="--",
                    label="E (alternate)")
            ax.axvline(20.0, color="0.45", ls=":", lw=0.7)
            ax.axvline(20.3, color="0.10", ls=":", lw=0.8)
            ax.set_xlim(19.5, min(22.4, nt[b + 1] + 0.9))
            ax.set_title(r"$b=[%.1f,%.1f)$, $s=%s$"
                         "\n" r"$n_{\rm held-out}=%d$%s"
                         % (nt[b], nt[b + 1],
                            r"2\!-\!3" if s == 2 else r"\geq 7",
                            n, "" if n >= 200 else r"  (BELOW 200)"),
                         fontsize=7.0,
                         color="k" if n >= 200 else "crimson")
            if j == 0:
                ax.set_ylabel(r"$P(\hat{N}\,|\,b,s)$ (in-grid)")
            if i == len(STRATA) - 1:
                ax.set_xlabel(r"observed $\log_{10}\hat{N}$")
            # log-y inset: the tails, which is where the compression is judged
            ipos = [0.60, 0.60, 0.38, 0.36] if b <= 8 else [0.02, 0.60, 0.38, 0.36]
            axi = ax.inset_axes(ipos, facecolor="w")
            axi.patch.set_alpha(0.95)
            ei = np.where(emp > 0, emp, np.nan)
            axi.step(ctr, ei, where="mid", color="0.25", lw=0.6)
            axi.plot(ctr, rB, color=cc.PALETTE[5], lw=0.8)
            axi.plot(ctr, rE, color=cc.PALETTE[2], lw=0.6, ls="--")
            axi.set_yscale("log")
            axi.set_ylim(1e-5, 1.0)
            axi.set_xlim(19.5, 22.4)
            axi.axvline(20.3, color="0.10", ls=":", lw=0.5)
            axi.tick_params(labelsize=4.5, length=1.5, pad=1)
            axi.set_xticks([19.5, 20.5, 21.5, 22.4])
            axi.set_yticks([1e-4, 1e-2, 1e0])
            axi.grid(alpha=0.2, lw=0.3)
            axi.set_title(r"log $y$: tails", fontsize=4.6, pad=0.8)
            # threshold masses
            rec = dict(b=b, s=s, n=n, n_by_K=n_k.tolist())
            for nm, v in (("emp", emp), ("B", rB), ("E", rE), ("R0", rR)):
                rec["P20_%s" % nm] = float(v[c20:].sum())
                rec["P203_%s" % nm] = float(v[c203:].sum())
            rec["phi"] = phi[b, s].tolist()
            rec["phi_ref"] = phi_ref[b, s].tolist()
            table_rows.append(rec)

    hnd = [plt.Line2D([], [], color="0.25", lw=1.0),
           plt.Line2D([], [], color=cc.PALETTE[5], lw=1.3),
           plt.Line2D([], [], color=cc.PALETTE[2], lw=0.8, ls="--"),
           plt.Line2D([], [], color="0.10", ls=":", lw=0.8)]
    fig.legend(hnd, ["held-out empirical (both CV folds, coarse-z pooled)",
                     "B  (36 coefficients; model of record)",
                     "E  (rank-2 correction on the R1c-form base)",
                     r"reporting thresholds $\hat{N}=20.0$, $20.3$"],
               loc="lower center", ncol=4, frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(pad=0.6, w_pad=0.9, h_pad=1.5)
    os.makedirs(cc.FIGDIR, exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)

    # ---- printed tables ----
    print("\n== threshold-crossing masses (coarse-z pooled by held-out counts) ==")
    hdr = ("cell", "n", "n by K", "P>=20.0 emp/B/E/R0", "P>=20.3 emp/B/E/R0")
    print("%-26s %6s %-18s %-34s %-34s" % hdr)
    for r in table_rows:
        print("%-26s %6d %-18s %-34s %-34s" % (
            "[%.1f,%.1f) s=%d" % (nt[r["b"]], nt[r["b"] + 1], r["s"]),
            r["n"], "/".join("%d" % x for x in r["n_by_K"]),
            " / ".join("%.4f" % r["P20_%s" % k] for k in ("emp", "B", "E", "R0")),
            " / ".join("%.4f" % r["P203_%s" % k] for k in ("emp", "B", "E", "R0"))))

    print("\n== phi(b,s,K) measured vs frozen phi_ref ==")
    for r in table_rows:
        print("[%.1f,%.1f) s=%d  phi=%s  phi_ref=%s" % (
            nt[r["b"]], nt[r["b"] + 1], r["s"],
            "/".join("%.4f" % x for x in r["phi"]),
            "/".join("%.4f" % x for x in r["phi_ref"])))
    print("\n== low-N contrast cells (where the phi defect lives) ==")
    for b in (0, 1, 2):
        for s in (2, 7):
            print("[%.1f,%.1f) s=%d  phi=%s  phi_ref=%s" % (
                nt[b], nt[b + 1], s,
                "/".join("%.4f" % x for x in phi[b, s]),
                "/".join("%.4f" % x for x in phi_ref[b, s])))

    # how many (b,s,K) rows reach 200 at b >= 21.3 (the b index of 21.3 is 11)
    tot = held.sum(axis=-1)
    hi = tot[11:]
    print("\nrows (b,s,K) with >=200 held-out events, all b:",
          int((tot >= 200).sum()))
    print("max held-out events in any (b,s,K) row at b >= 21.3:",
          float(hi.max()), "; cells >= 20:", int((hi >= 20).sum()),
          "; total events:", float(hi.sum()))
    print("max held-out events in any (b,s) cell at b = [21.5,21.7):",
          float(tot[12].sum(axis=-1).max()))

    with open(os.path.join(cc.FIGDIR, "sec6_numbers.json"), "w") as fh:
        json.dump(table_rows, fh, indent=1)


if __name__ == "__main__":
    main()
