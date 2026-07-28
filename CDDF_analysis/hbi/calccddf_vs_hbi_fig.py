# -*- coding: utf-8 -*-
"""Figure: differential f(N) calc_cddf vs HBI vs truth, + ratio panel (est/truth), per mock.
MOCK-ONLY. Writes PNGs to the private notes repo (not the code repo).

RESPONSE KERNEL: every HBI number on this figure -- the per-bin points (HBI_BAND_2LPT0)
AND the cumulative R0 annotation (HBI_CUM) -- is now on the FORWARD-response kernel
(resp_kind='forward').  Until 2026-07-28 the points came from the RETIRED kappa artifact
while the annotation quoted forward; see the HBI_BAND_2LPT0 comment.  The kernel is
printed on the plot so a reader never has to trust this docstring.

NO HARD-CODED HBI NUMBERS (2026-07-28).  The kernel-coherence fix below is now
enforced by CONSTRUCTION rather than by transcription: every HBI value is READ,
at import, from the stamped forward artifact and the loader ASSERTS
``metadata.resp_kind == 'forward'`` and that ``metadata.retired`` is ABSENT, so
the retired kappa artifact can never silently become the source again.

CROSS-BRANCH RESOLUTION.  ``subdla_mock_validation_forward.json`` is committed on
branch ``lls-subdla-cddf``, not on this one.  Both are worktrees off the SAME
object store (/home/mfho/desi_gpy_dla_detection/.git), so it is read
content-addressed via ``git show <40-char-commit>:<path>`` — no checkout, no
copy, and the exact blob SHA is reported on stdout.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root on path (so this runs as a script as well as a module)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from CDDF_analysis.hbi.calccddf_vs_hbi_artifact import load_hbi_forward  # noqa: E402

FIGDIR = "/home/mfho/desi_gpy_dla_notes/notes/figures/2026-07-10_calccddf"

# HBI per-0.1dex differential in the band [19.5,20.3), 2LPT-0 loa0.
#
# KERNEL COHERENCE FIX (2026-07-28).  These points were previously copied from
# CDDF_analysis/hbi/subdla_mock_validation.json -- the RETIRED artifact
# (metadata.retired=True, superseded_by=subdla_mock_validation_forward.json), which was
# produced on the GP-POSTERIOR ('kappa') kernel.  Meanwhile HBI_CUM below and the
# annotation in plot_mock() quote the FORWARD cumulative R0.  The figure therefore
# plotted KAPPA points underneath FORWARD text -- two different estimators in one panel,
# with nothing on the figure saying so.  The old kappa row for reference:
#     19.5: 0.00596205, 19.6: 0.00948884, 19.7: 0.0116299, 19.8: 0.0119525,
#     19.9: 0.0112131,  20.0: 0.0104513,  20.1: 0.0104033, 20.2: 0.0107529
# (the retired kappa band R0 is 0.883/0.899; the forward is 0.849/0.822).
#
# Now taken from CDDF_analysis/hbi/subdla_mock_validation_forward.json per_bin.loa0
# (metadata.resp_kind == "forward"), which is the SAME kernel as HBI_CUM.  Note the
# per-bin SHAPE changes qualitatively, not just the level: the forward kernel puts MORE
# mass in [19.5,19.6) (R0 1.068 vs the kappa 0.454) and LESS in [19.7,19.9), i.e. the
# edge-migration story the ratio panel tells is kernel-dependent.
def _load_hbi_tables():
    """Build the per-bin and cumulative HBI tables by READING the stamped
    forward artifact (assertions live in ``load_hbi_forward``).

    Returns ``(band, resp_kind, cum, prov)`` where ``band`` is
    ``{blo: (dndx_tru, dndx_est_loa0)}`` and ``cum`` is
    ``{mock: {"dndx": {...}}}``.

    Omega is deliberately NOT carried: Omega built from the leaky truth f(N)
    (B16 — truth assembled with no z-mask while dX IS masked) is biased and must
    be re-derived, never rescaled.  dN/dX is clean, and the figure only uses it.

    HBI COVERAGE: 2LPT-0 only.  The London-0 / Saclay-0 forward transfer legs
    exist solely in the UNTRACKED ``crossmock_transfer_loa0.json``, whose own
    provenance_note admits there is no committed aggregator and whose legs are
    stamped ``-dirty`` => NOT QUOTABLE under this project's headline rule.  They
    are therefore absent rather than hard-coded, and the figure prints
    "HBI n/a" on those panels.
    """
    doc, prov = load_hbi_forward()
    band = {round(r["blo"], 3): (r["dndx_tru"], r["dndx_est"])
            for r in doc["per_bin"]["loa0"]}
    integ = doc["integrated"]["loa0"]
    num195 = integ["dndx_est_195_203"] + integ["dndx_est_203"]
    den195 = integ["dndx_tru_195_203"] + integ["dndx_tru_203"]
    cum = {"2lpt0": dict(dndx={
        "20.3": integ["r0_dndx_203"],
        "20.0": integ["r0_dndx_200"],
        "19.5": num195 / den195,
        "band": integ["r0_dndx_195_203"],
    })}
    return band, prov["resp_kind"], cum, prov


HBI_BAND_2LPT0, HBI_BAND_2LPT0_RESP_KIND, HBI_CUM, HBI_PROV = _load_hbi_tables()


def plot_mock(mock, jpath, ax_top, ax_bot, agg=None):
    d = json.load(open(jpath))
    N = np.array(d["N_centers"])
    fN_calc = np.array(d["fN_calccddf"])
    fN_tru = np.array(d["fN_truth"])
    edges = np.round(np.arange(17.2, 22.40001, 0.1), 3)
    dN = 10.0 ** edges[1:] - 10.0 ** edges[:-1]

    # FF sampling interval + leg role, from the stamped aggregate (optional).
    ff68 = None
    role = ""
    if agg is not None and mock in agg.get("mocks", {}):
        am = agg["mocks"][mock]
        ff68 = (np.asarray(am["fN"]["calccddf_68_lo"], float),
                np.asarray(am["fN"]["calccddf_68_hi"], float))
        role = "  [{}]".format(am["role"])

    Nlin = 10.0 ** N
    good = fN_tru > 0
    # top: f(N) log-log
    ax_top.plot(Nlin[good], fN_tru[good], "k-", lw=1.6, label="truth (injected)")
    gc = fN_calc > 0
    ax_top.plot(Nlin[gc], fN_calc[gc], "o-", color="#2166ac", ms=3, lw=1.0,
                label="FF plug-in: calc_cddf (literal Bird-2017)")
    if ff68 is not None:
        ax_top.fill_between(Nlin[gc], ff68[0][gc], ff68[1][gc], color="#2166ac",
                            alpha=0.20, lw=0,
                            label="FF 68% sampling interval (plug-in; NOT credible)")
    # HBI band differential (2lpt0 only)
    if mock == "2lpt0":
        xb, yb = [], []
        for blo, (dt, de) in HBI_BAND_2LPT0.items():
            c = blo + 0.05
            i = np.argmin(np.abs(N - c))
            xb.append(10.0 ** c)
            yb.append(de / dN[i])
        ax_top.plot(xb, yb, "s", color="#b2182b", ms=5, label=f"HBI ({HBI_BAND_2LPT0_RESP_KIND} kernel, band)")
    ax_top.axvspan(10 ** 19.5, 10 ** 20.3, color="orange", alpha=0.08)
    ax_top.axvline(10 ** 20.3, color="grey", ls=":", lw=0.8)
    ax_top.set_yscale("log"); ax_top.set_xscale("log")
    ax_top.set_ylabel(r"$f(N_{\rm HI})$  [cm$^2$]")
    ax_top.set_title(f"{mock}{role}  (MOCK; z$\\in$[2,3.5], SNR>2, Ly$\\alpha$-only)",
                     fontsize=8.5)
    ax_top.legend(fontsize=7.0, loc="upper right")
    ax_top.set_xlim(10 ** 18.8, 10 ** 22.4)
    ymid = np.median(fN_tru[good])
    ax_top.set_ylim(ymid * 1e-4, fN_tru[good].max() * 5)

    # bottom: ratio est/truth
    r_calc = np.where(good, fN_calc / np.where(good, fN_tru, 1), np.nan)
    ax_bot.plot(Nlin[good], r_calc[good], "o-", color="#2166ac", ms=3, lw=1.0,
                label="FF plug-in / truth")
    if ff68 is not None:
        den = np.where(good, fN_tru, 1.0)
        ax_bot.fill_between(Nlin[good], (ff68[0] / den)[good], (ff68[1] / den)[good],
                            color="#2166ac", alpha=0.20, lw=0)
    if mock == "2lpt0":
        xb, rb = [], []
        for blo, (dt, de) in HBI_BAND_2LPT0.items():
            xb.append(10.0 ** (blo + 0.05)); rb.append(de / dt)
        ax_bot.plot(xb, rb, "s-", color="#b2182b", ms=4, lw=0.8, label=f"HBI / truth ({HBI_BAND_2LPT0_RESP_KIND}, band)")
    ax_bot.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_bot.axvspan(10 ** 19.5, 10 ** 20.3, color="orange", alpha=0.08)
    ax_bot.axvline(10 ** 20.3, color="grey", ls=":", lw=0.8)
    ax_bot.set_xscale("log")
    ax_bot.set_ylim(0, 1.6)
    ax_bot.set_xlim(10 ** 18.8, 10 ** 22.4)
    ax_bot.set_ylabel("est / truth")
    ax_bot.set_xlabel(r"$N_{\rm HI}$  [cm$^{-2}$]")
    ax_bot.legend(fontsize=7.0, loc="upper right")

    cum = d["cumulative"]
    hc = HBI_CUM.get(mock)
    lines = [f"HBI kernel: resp_kind={HBI_BAND_2LPT0_RESP_KIND}  (read from artifact)",
             "cumulative R0 (est/truth):"]
    for ffk, hk, lab in [("20.3", "20.3", ">=20.3"), ("20.0", "20.0", ">=20.0"),
                         ("band_195_203", "band", "band[19.5,20.3)")]:
        h = f"  (HBI {hc['dndx'][hk]:.3f})" if hc else "  (HBI n/a: no stamped leg)"
        lines.append(f"  {lab}: {cum['R0_calccddf']['dndx'][ffk]:.3f}{h}")
    lines.append("FF = plug-in point; HBI = posterior-model point. NOT the same estimand.")
    # bottom-left: the FF/truth curve lives near ratio ~1, so keep the annotation
    # out of the panel's mid-band or it hides the very line it describes.
    ax_bot.text(0.02, 0.03, "\n".join(lines), transform=ax_bot.transAxes, fontsize=6.2,
                va="bottom", family="monospace",
                bbox=dict(fc="white", ec="grey", alpha=0.85))


DEFAULT_CAPTION = (
    "FF band: 68% sampling interval on a PLUG-IN estimator (counting scatter only). "
    "NOT a posterior credible interval and not commensurate with an HBI band.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", nargs="+", required=True, help="mock=jsonpath ...")
    ap.add_argument("--aggregate", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "calccddf_vs_hbi.json"),
        help="stamped aggregate (calccddf_vs_hbi_artifact.py) supplying the FF "
             "68%% sampling interval, the leg ROLE and the honest caption")
    ap.add_argument("--figdir", default=FIGDIR)
    args = ap.parse_args(argv)

    print(f"HBI reference: {HBI_PROV['source']}  "
          f"(resp_kind={HBI_PROV['resp_kind']}, metadata.retired absent)")

    agg = json.load(open(args.aggregate)) if os.path.exists(args.aggregate) else None
    caption = (agg["metadata"]["uncertainty"]["figure_caption"]
               if agg else DEFAULT_CAPTION + "  [aggregate not found: no FF band drawn]")

    os.makedirs(args.figdir, exist_ok=True)
    for spec in args.jsons:
        mock, jp = spec.split("=", 1)
        if not os.path.exists(jp):
            print("skip (missing):", jp); continue
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.4, 6.6), height_ratios=[2.2, 1],
                                     sharex=True, gridspec_kw=dict(hspace=0.06))
        plot_mock(mock, jp, a0, a1, agg=agg)
        fig.text(0.01, 0.004, caption, fontsize=5.4, va="bottom", wrap=True)
        out = os.path.join(args.figdir, f"calccddf_vs_hbi_{mock}.png")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
