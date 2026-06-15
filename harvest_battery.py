import numpy as np, os, glob
BASE = "/scratch/cavestru_root/cavestru0/mfho/cddf_o3_realdata"
TRUTH = {20.0: None, 20.3: 0.0537, 20.6: 0.03007}
dirs = [("baseline", f"{BASE}/phase3d_postkernel_out")] + [
    (os.path.basename(d.rstrip('/')), d) for d in sorted(glob.glob(f"{BASE}/phase3d_experiments/*/"))]
print(f"{'experiment':16} {'dNdX>=20.0':>10} {'dNdX>=20.3':>10} {'R0_20.3':>8} {'PITcov68':>9} {'WALL1':>7}")
for name, d in dirs:
    vp = os.path.join(d, "phase3d_v3_point_kernel.npz")
    pit = os.path.join(d, "phase3d_pit_isolated_tp.npz")
    w = os.path.join(d, "wall1_result.tsv")
    d203 = d200 = r0 = cov = passed = "-"
    if os.path.exists(vp):
        z = np.load(vp, allow_pickle=True)
        for x in z.files:
            if "dndx_total_20.3" in x: v = float(z[x]); d203 = f"{v:.4f}"; r0 = f"{v/0.0537:.2f}"
            if "dndx_total_20.0" in x: d200 = f"{float(z[x]):.4f}"
    if os.path.exists(pit):
        zp = np.load(pit, allow_pickle=True)
        if "cov68" in zp.files: cov = f"{float(zp['cov68']):.3f}"
    if os.path.exists(w):
        t = open(w).read()
        passed = "PASS" if ("passed\tTrue" in t) else ("FAIL" if "passed" in t else "?")
    print(f"{name:16} {d200:>10} {d203:>10} {r0:>8} {cov:>9} {passed:>7}")
