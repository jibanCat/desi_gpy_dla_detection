#!/usr/bin/env python3
"""laplace_modes.py — Laplace (Gaussian) estimate of the posterior MASS of separate basins of the frozen model_cc
(2026-09-02 HBI identifiability campaign, kickoff §27 / request §7: 'a mode claim must be supported by posterior geometry';
the potential-energy gap alone ignores volume). For each labelled chain: start from its highest-density draw, refine the
mode of the log joint in numpyro's UNCONSTRAINED space (L-BFGS), evaluate the Hessian there, and report
log mass ≈ log p(z*) + (d/2) log 2π − ½ log det(−H). Differences between basins are what matters. PRIVATE outputs.

    python tools/hbi_validation/laplace_modes.py --pack PACK --chain ROOT/R0/REAL_ln_s20260822_allsites.npz:1:dominant \
        --chain ROOT/R0/REAL_ln_deep_s20260826_allsites.npz:0:mirror [--t-scale 1.0] --out laplace.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from CDDF_analysis.hbi_mcmc.pack import load_pack                                            # noqa: E402
from CDDF_analysis.hbi_mcmc.cc_posterior_validation import build_cc_tensors, model_cc       # noqa: E402
SITES = ("sigma_N", "sigma_z", "theta_level", "theta_slope", "eps_N", "eps_z", "psi_c", "fp_lam_total", "fp_shape_v", "t")


def main(argv=None):
    import jax, jax.numpy as jnp
    from numpyro.infer.util import initialize_model, unconstrain_fn, log_density
    from scipy.optimize import minimize
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True); ap.add_argument("--chain", action="append", required=True, help="allsites.npz:chain:label")
    ap.add_argument("--t-scale", type=float, default=1.0); ap.add_argument("--fix-t", action="store_true"); ap.add_argument("--fix-psi-c", action="store_true")
    ap.add_argument("--out", required=True); ap.add_argument("--maxiter", type=int, default=400)
    a = ap.parse_args(argv)
    pk = load_pack(a.pack); consts, Mg = build_cc_tensors(pk)
    counts = jnp.asarray(np.asarray(pk.counts, float)); fpc = jnp.asarray(np.asarray(pk.fp_counts, float))
    margs = (consts, Mg); mkw = dict(counts=counts, fp_counts=fpc, fp_mode="informative_ln", t_scale=a.t_scale, fix_t=a.fix_t, fix_psi_c=a.fix_psi_c)
    init = initialize_model(jax.random.PRNGKey(0), model_cc, model_args=margs, model_kwargs=mkw)
    potential_fn = init.potential_fn                     # -log joint in unconstrained space (incl. Jacobians)
    sites = [s for s in SITES if not (a.fix_t and s == "t") and not (a.fix_psi_c and s == "psi_c")]
    shapes = {}
    z0 = init.param_info.z
    for s in sites:
        shapes[s] = np.shape(z0[s])
    def flat(zd):
        return jnp.concatenate([jnp.ravel(jnp.asarray(zd[s])) for s in sites])
    def unflat(v):
        out, off = {}, 0
        for s in sites:
            n = int(np.prod(shapes[s])) if shapes[s] else 1
            out[s] = jnp.reshape(v[off:off + n], shapes[s]); off += n
        return out
    U = jax.jit(lambda v: potential_fn(unflat(v))); gU = jax.jit(jax.grad(lambda v: potential_fn(unflat(v))))
    def HU(v, h=1e-4):
        """Hessian by central finite differences of the AD gradient (second-order AD returns NaN columns
        through the masked Poisson / clip; the gradient itself is finite)."""
        v = np.asarray(v, float); n = v.size; H = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n); e[j] = h
            H[:, j] = (np.asarray(gU(jnp.asarray(v + e)), float) - np.asarray(gU(jnp.asarray(v - e)), float)) / (2 * h)
        return 0.5 * (H + H.T)
    d = int(sum(int(np.prod(shapes[s])) if shapes[s] else 1 for s in sites))
    res = dict(pack=a.pack, t_scale=a.t_scale, fix_t=a.fix_t, fix_psi_c=a.fix_psi_c, dim=d, basins={})
    for spec in a.chain:
        p, ch, lab = spec.split(":"); ch = int(ch); z = np.load(p)
        pe = np.asarray(z["potential_energy"])[ch]; i = int(np.argmin(pe))
        params = {s: jnp.asarray(np.asarray(z[s])[ch, i]) for s in sites}
        zu = unconstrain_fn(model_cc, margs, mkw, params)
        v0 = np.asarray(flat(zu), float); U0 = float(U(v0))
        r = minimize(lambda v: float(U(jnp.asarray(v))), v0, jac=lambda v: np.asarray(gU(jnp.asarray(v)), float), method="L-BFGS-B", options=dict(maxiter=a.maxiter))
        v1 = r.x if np.isfinite(r.fun) and r.fun <= U0 else v0
        U1 = float(U(jnp.asarray(v1))); gn = float(np.linalg.norm(np.asarray(gU(jnp.asarray(v1)), float)))
        H = HU(v1)
        if not np.isfinite(H).all():
            raise SystemExit(f"{lab}: non-finite Hessian entries ({int((~np.isfinite(H)).sum())})")
        w = np.linalg.eigvalsh(H)
        npos = int((w > 0).sum()); logdet = float(np.sum(np.log(w[w > 0])))
        logmass = -U1 + 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet
        # constrained-space readout of the mode
        zc = {s: np.asarray(v) for s, v in init.postprocess_fn(unflat(jnp.asarray(v1))).items() if s in sites}
        lam = float(np.sum(zc["fp_lam_total"])) if "fp_lam_total" in zc else None
        res["basins"][lab] = dict(source=p, chain=ch, start_draw=i, U_start=U0, U_mode=U1, converged=bool(r.success), n_iter=int(r.nit), grad_norm_mode=gn,
                                  hessian_min_eig=float(w.min()), hessian_n_nonpositive=int(d - npos), logdet_pos=logdet,
                                  log_laplace_mass=float(logmass), t_mode=(np.asarray(zc["t"]).tolist() if "t" in zc else "fixed 0"),
                                  lam_mode=lam, logL_mode=(float(np.log(lam)) if lam else None))
        print(f"{lab}: U start {U0:.1f} -> mode {U1:.1f} (iter {r.nit}, conv {r.success}); min eig {w.min():.3g}, non-positive {d-npos}; ½logdet {0.5*logdet:.1f}; log Laplace mass {logmass:.1f}; t_mode {res['basins'][lab]['t_mode']}")
    labs = list(res["basins"])
    if len(labs) >= 2:
        b0, b1 = res["basins"][labs[0]], res["basins"][labs[1]]
        res["comparison"] = dict(pair=[labs[0], labs[1]], delta_U_mode=b1["U_mode"] - b0["U_mode"], delta_half_logdet=0.5 * (b1["logdet_pos"] - b0["logdet_pos"]),
                                 log_mass_ratio_first_over_second=b0["log_laplace_mass"] - b1["log_laplace_mass"],
                                 note="log(mass_1/mass_2) = -(U1-U2) - ½(logdet1 - logdet2); positive favours the first basin")
        print("comparison:", res["comparison"])
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
