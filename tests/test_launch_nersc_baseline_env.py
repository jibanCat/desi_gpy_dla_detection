"""launch_nersc.sh (MAX4 repair cycle, 2026-09-01): the emitted BASELINE.env records the exported
archive-route / input-window variables (GPDLA_SPECTRA_ARCHIVE, EXTERNAL_HPX_LIST, GPDLA_ZMIN_QSO,
GPDLA_ZMAX_QSO), empty when unset. Runs the launcher against a stub `sbatch` on PATH (prints and
exits 0) — no scheduler is ever contacted."""
import os
import subprocess
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(REPO, "slurm/nersc/production/launch_nersc.sh")


def _run(tmp_path, exports):
    out = tmp_path / "out"
    for name in ("qso.fits", "model.h5", "samples.mat"):
        (tmp_path / name).write_text("x")
    stub = tmp_path / "bin"; stub.mkdir()
    (stub / "sbatch").write_text("#!/bin/bash\necho \"stub sbatch $*\" >> \"$STUB_LOG\"\necho 'Submitted batch job 0'\n")
    os.chmod(stub / "sbatch", 0o755)
    env_file = tmp_path / "synthetic.env"
    env_file.write_text(textwrap.dedent(f'''
        MODE=loa
        QSOCAT="{tmp_path}/qso.fits"
        OUTDIR="{out}"
        LEARNED_FILE="{tmp_path}/model.h5"
        CATALOG_NAME=x; LOS_CATALOG=x; DLA_CATALOG=x
        DLA_SAMPLES_FILE="{tmp_path}/samples.mat"; SUB_DLA_SAMPLES_FILE="{tmp_path}/samples.mat"
        OUTER_MAX_INDEX=1
        NERSC_ALLOWED_OUTPUT_PREFIXES=("{os.path.realpath(tmp_path)}/")
        NERSC_SLURM_CONSTRAINT=""; NERSC_NTASKS=1; MAX_WORKERS=1; NERSC_ENV_SETUP="true"
        LOADING_MIN_LAMBDA=910; LOADING_MAX_LAMBDA=1550; NORMALIZATION_MIN_LAMBDA=1425; NORMALIZATION_MAX_LAMBDA=1475
        MIN_LAMBDA=911.75; MAX_LAMBDA=1250; DLAMBDA=0.15; K=30; NUM_FOREST_LINES=31; NUM_LINES=3
        MIN_Z_SEPARATION=3000.0; MAX_NOISE_VARIANCE=9; MAX_DLAS=4; SINGLE_ABSORBER_MODEL=1; FILTER_LOW_LIKELIHOOD=1
        NUM_DLA_SAMPLES=50000; NUM_SUBDLA_SAMPLES=50000; BATCH_SIZE=1250; BALMASK=false; RELEASE=loa
        PREV_TAU_0=0.00246; PREV_BETA=3.62
    ''') + "".join(f'export {k}="{v}"\n' for k, v in exports.items()))
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_LOG=str(tmp_path / "stub.log"))
    for k in ("GPDLA_SPECTRA_ARCHIVE", "EXTERNAL_HPX_LIST", "GPDLA_ZMIN_QSO", "GPDLA_ZMAX_QSO"):
        env.pop(k, None)
    r = subprocess.run(["bash", LAUNCHER, str(env_file), "--no-sleep"], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "submitted 1 sbatch job(s)" in r.stdout and (tmp_path / "stub.log").exists()   # the stub, not a scheduler
    return dict(l.split("=", 1) for l in (out / "BASELINE.env").read_text().splitlines() if l and not l.startswith("#"))


def test_baseline_env_records_the_exported_route_variables(tmp_path):
    b = _run(tmp_path, {"GPDLA_SPECTRA_ARCHIVE": "/x/archive_v1.h5", "EXTERNAL_HPX_LIST": "/x/hpx.txt",
                        "GPDLA_ZMIN_QSO": "4.25", "GPDLA_ZMAX_QSO": "7.0"})
    assert b["GPDLA_SPECTRA_ARCHIVE"] == "/x/archive_v1.h5" and b["EXTERNAL_HPX_LIST"] == "/x/hpx.txt"
    assert b["GPDLA_ZMIN_QSO"] == "4.25" and b["GPDLA_ZMAX_QSO"] == "7.0"
    # the pre-existing record is intact
    assert b["MAX_DLAS"] == "4" and b["FILTER_LOW_LIKELIHOOD"] == "1" and b["NUM_DLA_SAMPLES"] == "50000"
    assert b["MODE"] == "loa" and b["PACKING"] == "N1xW1" and b["PAIR_PRIOR_MODE"] == "(unset)"


def test_baseline_env_records_empty_when_unset(tmp_path):
    b = _run(tmp_path, {})
    for k in ("GPDLA_SPECTRA_ARCHIVE", "EXTERNAL_HPX_LIST", "GPDLA_ZMIN_QSO", "GPDLA_ZMAX_QSO"):
        assert k in b and b[k] == ""
