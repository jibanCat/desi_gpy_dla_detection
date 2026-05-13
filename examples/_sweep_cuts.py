"""Sweep (snr_min, P_DLA cut) over multiple London catalogs, full forest."""
import subprocess, itertools, re, sys, os

TRUTH = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/dla_cat.fits"
BAL   = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/bal_cat.fits"
MOCK  = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124"
SCRIPT = os.path.join(os.path.dirname(__file__), "molly_faithful_pc_plots.py")

CATS_NORMAL = [
    ("26k baseline",   "/pscratch/sd/j/jibancat/prod533test-20260511_1333/london0_y3"),
    ("8f PW14 50k",    "/pscratch/sd/j/jibancat/prod533_5k_20260511/london_pw14_50k"),
    ("8f tau_eb",      "/pscratch/sd/j/jibancat/prod533_5k_20260511/london_tau_eb"),
]
CATS_LYB = [
    ("26k base + lyb_veto",    "/pscratch/sd/j/jibancat/prod533test-20260511_1333/london0_y3"),
    ("8f PW14 50k + lyb_veto", "/pscratch/sd/j/jibancat/prod533_5k_20260511/london_pw14_50k"),
    ("8f tau_eb + lyb_veto",   "/pscratch/sd/j/jibancat/prod533_5k_20260511/london_tau_eb"),
]

def run(cat_dir, snr, pdla, lyb_veto=False):
    cmd = ["python3", SCRIPT,
           "--catalog-dir", cat_dir,
           "--truth", TRUTH, "--bal-cat", BAL, "--no-bal", "--truth-nhi-min", "20.3",
           "--mockdir", MOCK,
           "--snr-min", str(snr), "--gp-conf", str(pdla),
           "--lam-rf-min", "911",
           "--out", "/tmp/sweep_%s_%s_%d" % (str(snr), str(pdla), int(lyb_veto)),
           "--title", "test"]
    if lyb_veto:
        cmd.append("--lyb-veto")
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"\[lya_lyb \].*P=([0-9.]+)\s+C=([0-9.]+)", out)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

print()
print("## Full forest 911-1216, BAL excluded, truth NHI>=20.3 — sweep across (snr_min, P_DLA cut)")
print()
hdr = "%-25s %5s %12s %8s %8s" % ("config", "snr", "P_DLA_cut", "Purity", "Compl")
print(hdr)
print("-" * len(hdr))
for label, cd in CATS_NORMAL:
    for snr, pdla in itertools.product([1.0, 2.0, 4.0, 6.0], [0.99, 0.999, 0.99999, 0.9999999]):
        p, c = run(cd, snr, pdla)
        mark = " <- 85/85 PASS" if (p is not None and p >= 0.85 and c >= 0.85) else ""
        if p is not None:
            print("%-25s %5.1f %12g %8.4f %8.4f%s" % (label, snr, pdla, p, c, mark))

print()
print("## + lyb_veto stacking")
print(hdr)
print("-" * len(hdr))
for label, cd in CATS_LYB:
    for snr, pdla in itertools.product([1.0, 2.0, 4.0, 6.0], [0.99, 0.999, 0.99999, 0.9999999]):
        p, c = run(cd, snr, pdla, lyb_veto=True)
        mark = " <- 85/85 PASS" if (p is not None and p >= 0.85 and c >= 0.85) else ""
        if p is not None:
            print("%-25s %5.1f %12g %8.4f %8.4f%s" % (label, snr, pdla, p, c, mark))
