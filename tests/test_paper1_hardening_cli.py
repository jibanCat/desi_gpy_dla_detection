"""Pre-push hardening (2026-08-26): production builders must NOT default their outputs to
frozen or superseded artifact paths, and the real-pack extractor must not silently pick a
superseded reference pack. Each CLI must refuse BEFORE touching any file."""
import os, subprocess, sys
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args):
    return subprocess.run([sys.executable] + args, cwd=REPO, capture_output=True, text=True,
                          env=dict(os.environ, OMP_NUM_THREADS="1"))


def test_kernel_fit_ensemble_requires_explicit_out():
    r = _run([os.path.join(REPO, "CDDF_analysis", "hbi_mcmc", "build_kernel_fit_ensemble.py")])
    assert r.returncode == 2 and "--out" in r.stderr


def test_znz_build_forward_cache_requires_explicit_out():
    from CDDF_analysis.hbi.znz_kernel import build_forward_cache
    with pytest.raises(SystemExit) as e:
        build_forward_cache([])
    assert e.value.code == 2


def test_extract_pack_real_requires_out_dir_and_ref_pack(tmp_path):
    script = os.path.join(REPO, "CDDF_analysis", "hbi_mcmc", "extract_pack_real.py")
    r = _run([script, "--real"])
    assert r.returncode == 2 and "--out-dir" in r.stderr
    r = _run([script, "--real", "--out-dir", str(tmp_path)])
    assert r.returncode != 0 and "--ref-pack" in (r.stderr + r.stdout)
    assert not any(tmp_path.iterdir()), "must refuse before writing anything"
