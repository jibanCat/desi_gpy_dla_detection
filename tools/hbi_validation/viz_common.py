"""viz_common.py — shared plotting conventions for the 2026-09-02 HBI validation diagnostics.
Internal diagnostics (not manuscript figures). Categorical colours in FIXED slot order (validated
reference palette of the dataviz method): slot 1 blue = chain 0 / R0, slot 2 orange = chain 1 / R1,
slot 3 aqua = mirror chain / R2, then yellow, magenta, green, violet, red for further runs. Correlation
maps use a two-hue diverging scale with a neutral grey midpoint. Arms are shown as small multiples,
never as an 8-hue scatter."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e6e3"; SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div_blue_grey_orange", ["#2a78d6", "#e9e9e6", "#eb6834"])
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#9dc1ec", "#2a78d6", "#0d3f7a"])

plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False, "pdf.fonttype": 42})


def short_label(row):
    """physical short label for an atlas panel from a mapping row."""
    import json
    site = row["site"]; ai = json.loads(row["array_index"]) if isinstance(row["array_index"], str) else row["array_index"]
    ce = json.loads(row["cell_edges"]) if isinstance(row["cell_edges"], str) else row["cell_edges"]
    f = lambda x: "∞" if x is None or (isinstance(x, float) and np.isinf(x)) else f"{x:g}"
    if site == "eps_N":
        return f"εN[{ai[0]}]\nN {f(ce[0])}–{f(ce[1])}"
    if site == "eps_z":
        return f"εz[b{ai[0]},k{ai[1]}]\nz→{f(ce[2])}"
    if site == "psi_c":
        return f"ψc[s{ai[0]},m{ai[1]}]\nN {f(ce[2])}–{f(ce[3])}"
    if site == "fp_shape_v":
        return f"v[c{ai[0]},s{ai[1]}]\nN̂ {f(ce[0])}–{f(ce[1])}"
    if site == "t":
        return f"t{ai[0]}\nz {f(ce[0])}–{f(ce[1])}"
    return {"sigma_N": "σN", "sigma_z": "σz", "theta_level": "θ level", "theta_slope": "θ slope", "fp_lam_total": "Λ"}.get(site, site)


def corner(X, labels, chain=None, groups=None, group_labels=None, title=None, bins=30, panel=0.42, contour=True, lines=None):
    """Lower-triangle corner. Diagonal: histogram per chain (slot colours). Off-diagonal: 2-D density (single sequential hue)
    with optional per-group contours (slot colours). `lines`: dict (i,j) -> list of (slope, intercept) reference lines."""
    X = np.asarray(X); n = X.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(max(3.5, panel * n + 0.8), max(3.5, panel * n + 0.8)))
    if n == 1:
        axes = np.array([[axes]])
    lims = [(np.percentile(X[:, i], 0.2), np.percentile(X[:, i], 99.8)) for i in range(n)]
    lims = [(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)) if hi > lo else (lo - 1e-6, hi + 1e-6) for lo, hi in lims]
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False); continue
            if i == j:
                if chain is not None:
                    for c in sorted(set(chain.tolist())):
                        ax.hist(X[chain == c, i], bins=bins, range=lims[i], histtype="step", color=SLOTS[c % len(SLOTS)], lw=0.8, density=True)
                else:
                    ax.hist(X[:, i], bins=bins, range=lims[i], histtype="step", color=SLOTS[0], lw=0.8, density=True)
                ax.set_yticks([])
            else:
                H, xe, ye = np.histogram2d(X[:, j], X[:, i], bins=bins, range=[lims[j], lims[i]])
                ax.imshow(H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto", cmap=SEQ, interpolation="nearest")
                if contour and groups is not None:
                    for gi, g in enumerate(sorted(set(groups.tolist()))):
                        m = groups == g
                        if m.sum() < 50:
                            continue
                        Hg, _, _ = np.histogram2d(X[m, j], X[m, i], bins=bins, range=[lims[j], lims[i]])
                        Hg = Hg.T / Hg.sum(); srt = np.sort(Hg.ravel())[::-1]; cs = np.cumsum(srt)
                        lev = [srt[np.searchsorted(cs, q)] for q in (0.68, 0.95)]
                        lev = sorted(set(lev))
                        if len(lev) >= 1 and lev[-1] > 0:
                            xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
                            ax.contour(xc, yc, Hg, levels=sorted(lev), colors=[SLOTS[gi % len(SLOTS)]], linewidths=0.6)
                if lines and (i, j) in lines:
                    xs = np.linspace(lims[j][0], lims[j][1], 2)
                    for slope, icpt in lines[(i, j)]:
                        ax.plot(xs, slope * xs + icpt, color=INK2, lw=0.5, ls="--")
                ax.set_ylim(lims[i])
            ax.set_xlim(lims[j])
            ax.tick_params(labelsize=5, length=2)
            if i < n - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=6)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=6)
            for s in ax.spines.values():
                s.set_linewidth(0.4)
    if group_labels:
        h = [plt.Line2D([], [], color=SLOTS[k % len(SLOTS)], lw=1) for k in range(len(group_labels))]
        fig.legend(h, group_labels, loc="upper right", fontsize=7, ncol=1)
    if title:
        fig.suptitle(title, fontsize=8, x=0.02, ha="left")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.96, wspace=0.05, hspace=0.05)
    return fig


def heatmap(C, labels_x, labels_y, title, vmax=1.0, figsize=None, tick_every=1):
    C = np.asarray(C)
    fig, ax = plt.subplots(figsize=figsize or (min(12, 0.18 * C.shape[1] + 2), min(12, 0.18 * C.shape[0] + 2)))
    im = ax.imshow(C, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(0, C.shape[1], tick_every)); ax.set_xticklabels(labels_x[::tick_every], rotation=90, fontsize=5)
    ax.set_yticks(range(0, C.shape[0], tick_every)); ax.set_yticklabels(labels_y[::tick_every], fontsize=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("posterior correlation", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    return fig
