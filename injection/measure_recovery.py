#!/usr/bin/env python
"""Measure recovered-vs-INJECTED truth from a GP run on an injectable tree:
detection completeness C_det, N_HI bias b_N, and the response matrix R + b_FP,
then render diagnostic figures. Step 3 of the campaign (see injection/README.md).

Non-circular by construction: recovery is scored against the INJECTION manifest
(inj_id → injected logN_true/z_true), never the natural 2LPT truth.
"""
import argparse, os, sys, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from measurements import detection_completeness, nhi_bias, response_matrix


def _load_recovered(processed_dir, manifest):
    """Build recovered records keyed by inj_id by matching GP TARGETID → manifest.

    Recovered-record schema (measurements.py seam): {inj_id, p_dla, logN_rec, z_rec}.
    Single-absorber model: model_posteriors[:,1] = p(absorber); MAP_log_nhis/MAP_z_dlas
    are the recovered N_HI/z. Each injected sightline hosts one injection (pilot).
    """
    import h5py
    # manifest: TARGETID -> inj_id (injections + controls)
    tid2inj = {int(r["target_id"]): int(r["inj_id"]) for r in manifest}
    recovered = {}
    files = sorted(glob.glob(os.path.join(processed_dir, "processed-*-*.h5")))
    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                tids = np.asarray(f["target_ids"][:]).astype(np.int64).ravel()
                mp = f["model_posteriors"][:]
                p_abs = mp[:, 1] if mp.shape[1] >= 2 else mp[:, -1]
                lognhi = np.asarray(f["MAP_log_nhis"][:]).reshape(len(tids), -1)[:, 0]
                zdla = np.asarray(f["MAP_z_dlas"][:]).reshape(len(tids), -1)[:, 0]
        except (OSError, KeyError):
            continue
        for i, tid in enumerate(tids):
            inj_id = tid2inj.get(int(tid))
            if inj_id is None:
                continue
            recovered[inj_id] = dict(inj_id=inj_id, p_dla=float(p_abs[i]),
                                     logN_rec=float(lognhi[i]), z_rec=float(zdla[i]))
    return recovered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True, help="injectable-tree root (has injection_truth.fits)")
    ap.add_argument("--processed", required=True, help="GP processed/ output dir")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--p_thresh", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)

    from astropy.table import Table
    man = Table.read(os.path.join(a.campaign, "injection_truth.fits"))
    manifest = [dict(zip(man.colnames, row)) for row in man]
    recovered = _load_recovered(a.processed, manifest)
    print(f"[measure] {len(recovered)}/{len(manifest)} manifest rows recovered in GP output")

    cdet = detection_completeness(recovered, manifest, p_dla_thresh=a.p_thresh)
    bias = nhi_bias(recovered, manifest, p_dla_thresh=a.p_thresh)
    print("[C_det] per (logN_true, z, SNR) cell:")
    for c in cdet.get("cells", cdet if isinstance(cdet, list) else []):
        print("  ", c)

    # figures: C_det(logN_true) and N_HI bias(logN_true)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # aggregate over z/SNR for the headline curves (cells carry logN_true)
    def _agg(d, key):
        xs, ys, los, his = [], [], [], []
        for c in d.get("cells", []):
            xs.append(c.get("logN_true")); ys.append(c.get(key))
            los.append(c.get(key + "_lo", c.get(key))); his.append(c.get(key + "_hi", c.get(key)))
        o = np.argsort(xs);
        return (np.array(xs)[o], np.array(ys)[o], np.array(los)[o], np.array(his)[o])
    try:
        x, y, lo, hi = _agg(cdet, "C_det")
        ax[0].errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-"); ax[0].set_xlabel("log N_HI (injected)")
        ax[0].set_ylabel("detection completeness C_det"); ax[0].axvline(20.3, ls=":", c="grey")
        ax[0].set_title("C_det (recovered vs INJECTED)"); ax[0].set_ylim(0, 1.05)
    except Exception as e:
        ax[0].set_title(f"C_det N/A: {e}")
    try:
        xb, yb, _, _ = _agg(bias, "b_N")
        ax[1].axhline(0, ls="--", c="grey"); ax[1].plot(xb, yb, "s-")
        ax[1].set_xlabel("log N_HI (injected)"); ax[1].set_ylabel("⟨logN_rec⟩ − logN_true")
        ax[1].set_title("N_HI bias"); ax[1].axvline(20.3, ls=":", c="grey")
    except Exception as e:
        ax[1].set_title(f"bias N/A: {e}")
    fig.suptitle("M3 injection pilot — recovery vs injected truth (non-circular)")
    fig.tight_layout()
    out = os.path.join(a.figdir, "injection_pilot_recovery.png")
    fig.savefig(out, dpi=110)
    print("[save] figure:", out)


if __name__ == "__main__":
    main()
