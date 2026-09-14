"""dash_figures.py — READ-ONLY dashboards for the PI inspection sections.

dash_systematics.png : per-effect size in units of the real pooled 68 % half-width
                       (no quadrature sum; S6 drawn as an open 'pending' bar).
dash_sampler_health.png : E-BFMI per chain over the 24 mock J=8 runs and the 16
                       real C1 runs, and the per-imputation headline medians in
                       half-widths around the pool.

All inputs are stored products; nothing frozen is written or modified.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCR = ("/tmp/claude-114399728/-home-mfho-desi-gpy-dla-detection/"
       "323b5500-b134-4e5f-8dd1-1292b65f6a96/scratchpad")
FIG = "/home/mfho/desi_gpy_dla_notes/figures/2026-09-14_pi_inspection"

TYPE_COLOR = {"bias": "#440154", "envelope": "#3b528b", "sensitivity": "#21918c",
              "calibration uncertainty": "#5ec962", "disclosure": "#bdbdbd"}


def systematics_fig(hw0, hw3):
    # (label, type, size_pp_ge20.0, size_pp_ge20.3, per-bin flag, outer pp or None)
    eff = [
        ("S1 z-transfer residual", "bias", 3.39, 4.10, True, None),
        ("S2 B-vs-E response form", "envelope", 1.78, 1.07, False, None),
        ("S3 FP scale / decomposition", "sensitivity", 0.66, 0.185, False, None),
        ("S4 high-N FP pseudo-count", "sensitivity", 0.16, 0.20, False, (0.80, 0.86)),
        ("S5 measured-vs-smooth phi", "sensitivity", 0.27, 0.40, False, None),
        ("S6 completeness covariance", "calibration uncertainty", None, None, False, None),
        ("S7 sub-floor transport", "sensitivity", 0.06, 0.03, False, None),
        ("S8 sampler geometry", "disclosure", 0.065, 0.044, False, None),
    ]
    # per-bin half-widths (B4, the widest-residual fully covered bin)
    hwb0, hwb3 = 1.154, 1.709
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for ax, hw, hwb, idx, ttl in ((axes[0], hw0, hwb0, 2, r"dN/dX $\geq$20.0"),
                                  (axes[1], hw3, hwb3, 3, r"dN/dX $\geq$20.3")):
        ys = np.arange(len(eff))[::-1]
        for y, e in zip(ys, eff):
            lab, typ, s0, s3, perbin, outer = e
            v = (s0 if idx == 2 else s3)
            den = hwb if perbin else hw
            if v is None:
                ax.barh(y, 0.30, color="none", edgecolor=TYPE_COLOR[typ],
                        hatch="////", linewidth=1.4, height=0.62)
                ax.text(0.34, y, "PENDING (not propagated)", va="center",
                        fontsize=8, color=TYPE_COLOR[typ])
                continue
            x = v / den
            ax.barh(y, x, color=TYPE_COLOR[typ], height=0.62)
            ax.text(x + 0.06, y, f"{v:.2f} pp = {x:.2f} hw" + (" (per-bin hw)" if perbin else ""),
                    va="center", fontsize=8)
            if outer is not None:
                xo = (outer[0] if idx == 2 else outer[1]) / den
                ax.plot([xo], [y], marker="D", ms=6, mfc="none",
                        mec=TYPE_COLOR[typ], mew=1.4)
                ax.text(xo - 0.10, y - 0.40, "outer (Jeffreys)", fontsize=7, ha="center",
                        color=TYPE_COLOR[typ])
        ax.axvline(1.0, color="0.35", lw=0.9, ls="--")
        ax.set_yticks(ys[::-1])
        ax.set_yticklabels([e[0] for e in eff][::-1], fontsize=9)
        ax.set_xlabel("size / real pooled 68 % half-width")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlim(0, 5.2)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(handles=[Patch(facecolor=c, label=k) for k, c in TYPE_COLOR.items()],
                   fontsize=7.5, loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.02))
    fig.suptitle("Named systematics of the frozen Paper-1 low-z model — sizes are MOCK-DERIVED; "
                 "NO quadrature sum (PI 2026-09-14 §18)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(FIG, "dash_systematics.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def sampler_fig(d):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax = axes[0]
    mock = d["mock_ebfmi_all"]
    real = d["real_ebfmi_all"]
    ax.plot(np.arange(len(mock)), mock, "o", ms=3.4, color="#3b528b",
            label=f"mock J=8 (24 runs, {len(mock)} chains)")
    ax.plot(np.arange(len(real)) + len(mock) + 4, real, "s", ms=3.4,
            color="#21918c", label=f"real C1 (16 runs, {len(real)} chains)")
    ax.axhline(0.3, color="#b02418", lw=1.0, ls="--")
    ax.text(1, 0.32, "E-BFMI = 0.3 advisory floor", fontsize=7.5, color="#b02418")
    ax.axvline(len(mock) + 2, color="0.7", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("chain (grouped by run)")
    ax.set_ylabel("E-BFMI")
    ax.set_title("E-BFMI per chain — one low chain per run, headlines unaffected", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    o = d["real_run_offsets_hw"]
    for seed, mk, col in ((20260811, "o", "#3b528b"), (20260812, "s", "#21918c")):
        js = [r["j"] for r in o if r["seed"] == seed]
        ax.plot(js, [r["d0"] for r in o if r["seed"] == seed], mk, ms=5,
                color=col, label=f"$\\geq$20.0 s{str(seed)[-2:]}")
        ax.plot(js, [r["d3"] for r in o if r["seed"] == seed], mk, ms=5,
                mfc="none", color=col, label=f"$\\geq$20.3 s{str(seed)[-2:]}")
    ax.axhline(0, color="0.3", lw=0.9)
    for y in (-0.5, 0.5):
        ax.axhline(y, color="#b02418", lw=0.9, ls="--")
    ax.text(0.05, 0.52, "sealed 0.5 hw spread rule", fontsize=7.5, color="#b02418")
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel("$\\Lambda$ imputation $j$")
    ax.set_ylabel("headline median $-$ pool  [pooled 68 % half-widths]")
    ax.set_title("Real C1: science marginal vs imputation (max spread 0.14 hw)", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIG, "dash_sampler_health.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    d = json.load(open(os.path.join(SCR, "dash_readout.json")))
    hw0 = d["real_pooled_halfwidths"]["ge20.0"]["hw68_pct"]
    hw3 = d["real_pooled_halfwidths"]["ge20.3"]["hw68_pct"]
    os.makedirs(FIG, exist_ok=True)
    systematics_fig(hw0, hw3)
    sampler_fig(d)
