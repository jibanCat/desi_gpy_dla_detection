"""REVIEW-ONLY (Phase A) — pilot: gradient check, noiseless reproduction of
Δdev≈41, the observed seed-17 draw, and per-fit timing."""
import json, time
import numpy as np
import fitter

P = fitter.Problem()
out = {}

print("dims: C=%d Kf=%d S=%d B=%d CS=%d npad=%d n_live=%d" %
      (P.C, P.Kf, P.S, P.B, P.CS, P.npad, P.n_live))
nf_F1 = int((P.sA > 0).sum())
nf_F3 = int((P.sA[P.npad:] > 0).sum())
n_pad_par = int((P.sA[:P.npad] > 0).sum())
nl = int(P.fp_live.sum())
print("free params: F1 signal=%d (pad=%d, window=%d) + FP=%d ; F3 signal=%d + FP=%d"
      % (nf_F1, n_pad_par, nf_F1 - n_pad_par, nl, nf_F3, nl))
out["dims"] = dict(C=P.C, Kf=P.Kf, S=P.S, B=P.B, CS=P.CS, npad=P.npad,
                   n_live=P.n_live, n_par_F1=nf_F1 + nl, n_par_F3=nf_F3 + nl,
                   n_pad_par=n_pad_par, n_fp_par=nl)

g = fitter.grad_check(P, seed=0)
print("grad check max rel err %.3e" % g)
out["grad_check_max_rel_err"] = g
assert g < 1e-5

# --- truth (I1: absorber-dominated) ---------------------------------------
T_A, T_B = 24000.0, 1086.7
f_t, lam_t, mu_t = P.build_truth(T_A, T_B)
T_W = float((f_t[P.npad:] * P.sA[P.npad:]).sum())
print("truth: T_A=%.1f T_B=%.1f T_W=%.2f (probe: 62589.7) total mu=%.2f"
      % (T_A, T_B, T_W, mu_t.sum()))
out["truth"] = dict(T_A=T_A, T_B=T_B, T_W=T_W, mu_total=float(mu_t.sum()))

# --- noiseless fits --------------------------------------------------------
for tag, padfree in [("F1", True), ("F3", False)]:
    t0 = time.time()
    r = fitter.fit(P, mu_t, pad_free=padfree)
    dt = time.time() - t0
    print("noiseless %s: dev=%.4f T_A=%.1f T_B=%.1f T_W=%.1f nit=%d nfev=%d "
          "success=%s t=%.1fs" % (tag, r["dev"], r["T_A"], r["T_B"], r["T_W"],
                                  r["nit"], r["nfev"], r["success"], dt))
    out["noiseless_" + tag] = dict(dev=r["dev"], T_A=r["T_A"], T_B=r["T_B"],
                                   T_W=r["T_W"], nit=r["nit"], nfev=r["nfev"],
                                   success=r["success"], seconds=dt)
out["noiseless_delta"] = out["noiseless_F3"]["dev"] - out["noiseless_F1"]["dev"]
print("noiseless Delta dev = %.4f  (probe: 41.24 - 0.03 = 41.21)"
      % out["noiseless_delta"])

# --- observed draw (probe's seed 17, identical generation path) ------------
rng = np.random.default_rng(17)
y_obs = rng.poisson(mu_t).astype(float)
out["y_obs_total"] = float(y_obs[P.live].sum())
print("y_obs total (live) = %.0f" % out["y_obs_total"])
for tag, padfree in [("F1", True), ("F3", False)]:
    t0 = time.time()
    r = fitter.fit(P, y_obs, pad_free=padfree)
    dt = time.time() - t0
    print("obs %s: dev=%.4f T_A=%.1f T_B=%.1f T_W=%.1f nit=%d success=%s t=%.1fs"
          % (tag, r["dev"], r["T_A"], r["T_B"], r["T_W"], r["nit"],
             r["success"], dt))
    out["obs_" + tag] = dict(dev=r["dev"], T_A=r["T_A"], T_B=r["T_B"],
                             T_W=r["T_W"], nit=r["nit"], success=r["success"],
                             seconds=dt)
out["obs_delta"] = out["obs_F3"]["dev"] - out["obs_F1"]["dev"]
print("obs Delta dev = %.4f  (probe, iteration-capped: 2206.21-2120.81=85.40)"
      % out["obs_delta"])

# --- second-init agreement check on the observed draw (global-opt evidence)
r2 = fitter.fit(P, y_obs, pad_free=True, n_em=2000)
print("obs F1 re-fit (n_em=2000): dev=%.4f (delta vs first %.2e)"
      % (r2["dev"], r2["dev"] - out["obs_F1"]["dev"]))
out["obs_F1_refit_dev"] = r2["dev"]

json.dump(out, open("out_pilot.json", "w"), indent=1)
print("PILOT DONE")
