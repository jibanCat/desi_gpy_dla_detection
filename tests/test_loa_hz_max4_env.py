"""slurm/greatlakes/production/loa_hz_MAX4_gl_v1.env (MAX4 repair cycle): sourcing the flavour
yields the certified MAX4 finder configuration on top of the unchanged hz chain, and leaves OUTDIR
empty for the wave env. Shell-only (no launch)."""
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(REPO, "slurm/greatlakes/production/loa_hz_MAX4_gl_v1.env")
KEYS = ["MAX_DLAS", "SINGLE_ABSORBER_MODEL", "FILTER_LOW_LIKELIHOOD", "NUM_SAMPLES", "NUM_DLA_SAMPLES", "NUM_SUBDLA_SAMPLES",
        "DLA_SAMPLES_FILE", "SUB_DLA_SAMPLES_FILE", "GPDLA_ZMIN_QSO", "GPDLA_ZMAX_QSO", "MODE", "OUTDIR", "RUN_TAG",
        "LEARNED_FILE", "MAX_LAMBDA", "ENABLE_TAU_EB", "NERSC_SLURM_ACCOUNT"]


def _source(extra=""):
    script = f'source "{ENV}"\n{extra}\n' + "\n".join(f'echo "{k}=${{{k}:-}}"' for k in KEYS)
    env = dict(os.environ, REPO_ROOT=REPO)
    for k in ("OUTDIR", "OUTDIR_OVERRIDE"):
        env.pop(k, None)
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)


def test_max4_flavour_applies_the_seven_overrides_on_the_hz_chain():
    v = _source()
    assert v["MAX_DLAS"] == "4" and v["FILTER_LOW_LIKELIHOOD"] == "1" and v["SINGLE_ABSORBER_MODEL"] == "1"
    assert v["NUM_SAMPLES"] == "50000" and v["NUM_DLA_SAMPLES"] == "50000" and v["NUM_SUBDLA_SAMPLES"] == "50000"
    assert v["DLA_SAMPLES_FILE"] == f"{REPO}/data/dr12q/processed/pw_samples_a3_172_225_50000.mat"
    assert v["SUB_DLA_SAMPLES_FILE"] == f"{REPO}/data/dr12q/processed/subdla_samples_a03_191_200_50000.mat"
    # the hz chain underneath is unchanged
    assert v["GPDLA_ZMIN_QSO"] == "4.25" and v["GPDLA_ZMAX_QSO"] == "7.0" and v["MODE"] == "loa"
    assert v["LEARNED_FILE"].endswith("DEPLOYED_phase2_2lpt_loa124_nohcd_nobal_wide_m/phase2_result.h5")
    assert v["MAX_LAMBDA"] == "1250" and v["ENABLE_TAU_EB"] == "1" and v["NERSC_SLURM_ACCOUNT"] == "cavestru0"
    assert v["RUN_TAG"] == "loa_hz_MAX4_v1"
    assert v["OUTDIR"] == ""                                       # a direct launch is refused by launch_nersc.sh


def test_wave_env_sets_outdir_after_sourcing(tmp_path):
    v = _source(f'OUTDIR="{tmp_path}/wave_outputs"')
    assert v["OUTDIR"] == f"{tmp_path}/wave_outputs" and v["MAX_DLAS"] == "4"


def test_max1_hz_flavour_is_untouched():
    script = f'source "{os.path.join(REPO, "slurm/greatlakes/production/loa_cddf_hz_gl_v1.env")}"\n' \
             'echo "MAX_DLAS=$MAX_DLAS"; echo "FILTER_LOW_LIKELIHOOD=$FILTER_LOW_LIKELIHOOD"; echo "NUM_DLA_SAMPLES=$NUM_DLA_SAMPLES"'
    r = subprocess.run(["bash", "-c", script], env=dict(os.environ, REPO_ROOT=REPO), capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split() == ["MAX_DLAS=1", "FILTER_LOW_LIKELIHOOD=0", "NUM_DLA_SAMPLES=100000"]
