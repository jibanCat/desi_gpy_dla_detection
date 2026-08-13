"""Tests for the fail-loud C-Voigt preflight (tools/voigt_preflight.py).

Engineering/reproducibility contract only (repair handoff 2026-08-12 §7):
a production-like run must not silently fall back to the pure-Python Voigt.
No science module is touched by the preflight; these tests exercise its
exit-code contract via subprocess.
"""
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "tools", "voigt_preflight.py")
SO = os.path.join(REPO_ROOT, "gpy_dla_detection", "_voigt.so")


def _run(repo_root):
    return subprocess.run([sys.executable, SCRIPT, "--repo-root", repo_root],
                          capture_output=True, text=True, timeout=120)


def test_preflight_fails_loud_without_extension(tmp_path):
    fake = tmp_path / "repo"
    (fake / "gpy_dla_detection").mkdir(parents=True)
    r = _run(str(fake))
    assert r.returncode == 1
    assert "VOIGT PREFLIGHT FAIL" in r.stderr
    assert "_voigt.so not found" in r.stderr


@pytest.mark.skipif(not os.path.isfile(SO),
                    reason="_voigt.so not built in this checkout "
                           "(bash tools/build_voigt.sh)")
def test_preflight_passes_with_built_extension():
    r = _run(REPO_ROOT)
    assert r.returncode == 0, r.stderr
    assert "VOIGT PREFLIGHT PASS" in r.stdout
    assert "so_sha256" in r.stdout


@pytest.mark.skipif(not os.path.isfile(SO),
                    reason="_voigt.so not built in this checkout")
def test_preflight_fails_on_unloadable_so(tmp_path):
    fake = tmp_path / "repo"
    pkg = fake / "gpy_dla_detection"
    pkg.mkdir(parents=True)
    (pkg / "_voigt.so").write_bytes(b"not an ELF shared object")
    # minimal package so the import path resolves to the broken .so
    (pkg / "__init__.py").write_text("")
    shutil.copy(os.path.join(REPO_ROOT, "gpy_dla_detection", "voigt_fast.py"),
                pkg / "voigt_fast.py")
    r = _run(str(fake))
    assert r.returncode == 1
    assert "VOIGT PREFLIGHT FAIL" in r.stderr
