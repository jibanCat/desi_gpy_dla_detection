#!/usr/bin/env python
"""r029_export_gp_model.py — export a trained GP null model (MATLAB .mat v7.3 / HDF5 or the
phase-2 .h5 schema) to the plot-ready npz the paper's FIG-02 generator reads
(rest_wavelengths, mu, log_omega, M[n_wave, k], n_spectra, n_iters) with a provenance JSON
(source path, sha256, training_release, n_train, grid, hyperparameters). Read-only on the
source; never recreated from a plot (R-029)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import h5py
import numpy as np


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _str(ds):
    a = np.array(ds)
    try:
        return "".join(chr(int(c)) for c in a.ravel())
    except Exception:
        return str(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True, help="output .npz (provenance JSON written alongside)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-iters", type=int, default=None)
    a = ap.parse_args()
    with h5py.File(a.src, "r") as h:
        keys = set(h.keys())
        rw = np.array(h["rest_wavelengths"]).ravel().astype(float)
        mu = np.array(h["mu"]).ravel().astype(float)
        lo = np.array(h["log_omega"]).ravel().astype(float)
        M = np.array(h["M"]).astype(float)
        if M.shape[0] != rw.size and M.shape[1] == rw.size:
            M = M.T                                       # MATLAB stores k x n_wave transposed under HDF5
        meta = {}
        for k in ("log_tau_0", "log_beta", "log_c_0", "max_noise_variance", "log_likelihood"):
            if k in keys:
                v = np.array(h[k]).ravel()
                meta[k] = float(v[-1]) if v.size else None
        if "training_release" in keys:
            meta["training_release"] = _str(h["training_release"])
        n_spectra = None
        if "train_ind" in keys:
            ti = np.array(h["train_ind"]).ravel()
            n_spectra = int(np.sum(ti > 0)); meta["train_ind_total"] = int(ti.size)
        elif "n_spectra" in keys:
            n_spectra = int(np.array(h["n_spectra"]).ravel()[0])
        n_iters = a.n_iters
        if n_iters is None and "minFunc_output" in keys and "iterations" in h["minFunc_output"]:
            n_iters = int(np.array(h["minFunc_output"]["iterations"]).ravel()[0])
        if n_iters is None and "n_iters" in keys:
            n_iters = int(np.array(h["n_iters"]).ravel()[0])
        if "normalization_min_lambda" in keys:
            meta["normalization_min_lambda"] = float(np.array(h["normalization_min_lambda"]).ravel()[0])
            meta["normalization_max_lambda"] = float(np.array(h["normalization_max_lambda"]).ravel()[0])
    np.savez(a.out, rest_wavelengths=rw, mu=mu, log_omega=lo, M=M,
             n_spectra=np.array(n_spectra if n_spectra is not None else -1), n_iters=np.array(n_iters if n_iters is not None else -1),
             label=np.array(a.label))
    prov = dict(label=a.label, source=a.src, source_realpath=os.path.realpath(a.src), source_sha256=_sha(a.src), out=a.out, out_sha256=_sha(a.out),
                grid=dict(n=int(rw.size), min=float(rw.min()), max=float(rw.max()), step=float(np.median(np.diff(rw)))), latent_rank=int(M.shape[1]),
                n_spectra=n_spectra, n_iters=n_iters, hyperparameters=meta,
                generator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip(),
                note="mu / log_omega / M exported verbatim (M transposed to [n_wave, k] if stored k x n_wave); omega = exp(log_omega) is the stored parameterization's own definition")
    json.dump(prov, open(a.out.replace(".npz", ".provenance.json"), "w"), indent=1)
    print(json.dumps({k: prov[k] for k in ("label", "source_sha256", "grid", "latent_rank", "n_spectra", "n_iters", "hyperparameters")}, indent=1))


if __name__ == "__main__":
    main()
