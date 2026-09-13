#!/usr/bin/env python
"""verify_runner_hook.py — smoke test that the delivered C_*.npz / P6b_*.npz
objects are consumable by the ladder runner's fixed-component hook.

VALIDATION-ONLY.  No sampler is run: this evaluates the SAME einsum expressions
``CDDF_analysis/hbi_mcmc/fp_ladder.model_cc_ladder`` uses for ``C_fixed`` (2-D
and 3-D) and ``mu_extra_fixed``, at the pack's own truth f, and checks shapes,
finiteness and the total-count ratio against the realised catalogue.

ENV: ``gpdla-hbi``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

FAMILIES = ("2lpt0", "london0", "saclay0")
PACK_DIR = ("/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata/"
            "adopted_packs_v2p2_20260821")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=("/scratch/cavestru_root/cavestru0/"
                                           "mfho/absorber_ladder_2026-09-13/"
                                           "completeness"))
    ap.add_argument("--variants", nargs="+",
                    default=["C0", "C1g", "C1n", "C1ns", "C1nsadd", "C1gz",
                             "C1nsz"])
    a = ap.parse_args(argv)

    import jax.numpy as jnp
    from CDDF_analysis.hbi_mcmc.pack import load_pack
    from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors
    from CDDF_analysis.hbi_mcmc.forward_selftest import truth_f

    out = {}
    for fam in FAMILIES:
        pk = load_pack(os.path.join(
            PACK_DIR, f"modelA_pack_{fam}_bw0p2_pad19p0_molly172_v2.npz"))
        consts, Mg = build_cc_tensors(pk)
        f = jnp.asarray(np.asarray(truth_f(pk), float))
        counts = np.asarray(pk.counts, float)
        row = {}
        for v in a.variants:
            z = np.load(os.path.join(a.products, f"C_{v}_{fam}.npz"),
                        allow_pickle=True)
            Cf = np.asarray(z["C_fixed"], float)
            assert Cf.shape == (consts.n_s, consts.n_b), \
                f"{v}/{fam}: C_fixed {Cf.shape} != (S, B)"
            # the runner's 2-D branch, verbatim
            w = consts.g_bk * f * consts.dN_b[:, None]
            tp = jnp.einsum("skcb,sb,bk->cks", Mg, jnp.asarray(Cf), w) \
                * consts.dX[None, :, :]
            tp = np.asarray(tp)
            assert tp.shape == counts.shape and np.all(np.isfinite(tp))
            rec = dict(shape_2d=list(Cf.shape),
                       tp_total=float(tp.sum()),
                       tp_over_counts=float(tp.sum() / counts.sum()),
                       dead_strata_are_zero=bool(
                           np.all(Cf[~np.asarray(z["live_strata_mask"],
                                                 bool)] == 0.0)))
            if "C_fixed_bkS" in z.files:
                C3 = np.asarray(z["C_fixed_bkS"], float)
                assert C3.shape == (consts.n_b, consts.n_k, consts.n_s)
                w0 = f * consts.dN_b[:, None]
                tp3 = np.asarray(jnp.einsum(
                    "skcb,bks,bk->cks", Mg, jnp.asarray(C3), w0)
                    * consts.dX[None, :, :])
                C3g = np.asarray(z["C_fixed_bkS_with_g"], float)
                tp3g = np.asarray(jnp.einsum(
                    "skcb,bks,bk->cks", Mg, jnp.asarray(C3g), w0)
                    * consts.dX[None, :, :])
                rec.update(shape_3d=list(C3.shape),
                           tp3_over_counts=float(tp3.sum() / counts.sum()),
                           tp3_with_g_over_counts=float(
                               tp3g.sum() / counts.sum()),
                           note_3d=("the runner's 3-D branch drops g_bk, so "
                                    "C_fixed_bkS_with_g is the like-for-like "
                                    "array for that branch"))
            row[v] = rec
        z = np.load(os.path.join(a.products, f"P6b_rate_{fam}.npz"),
                    allow_pickle=True)
        mu = np.asarray(z["mu_P6b_cks"], float)
        assert mu.shape == counts.shape and np.all(np.isfinite(mu)) \
            and np.all(mu >= 0)
        row["P6b"] = dict(shape=list(mu.shape), total=float(mu.sum()),
                          observed=float(np.asarray(z["obs_P6b_cks"]).sum()))
        out[fam] = row
        print(f"[{fam}] OK  " + "  ".join(
            f"{v}:{row[v]['tp_over_counts']:.4f}" for v in a.variants)
            + f"  P6b mu={row['P6b']['total']:.0f}", flush=True)
    p = os.path.join(a.products, "runner_hook_smoke.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
