#!/usr/bin/env bash
# Canonical build for the production C Voigt extension (_voigt.so).
#
# Formalizes the README "Compilation and Installation Guide" step so a fresh
# checkout has ONE documented, verifiable path to the compiled extension
# (2026-08-12 NERSC execution report §3.2 / repair handoff §7 follow-up):
#   - compiles gpy_dla_detection/ctypes_voigt.c against libcerf;
#   - bakes an RPATH to the libcerf lib dir so the extension resolves
#     WITHOUT requiring LD_LIBRARY_PATH in the run environment;
#   - finishes with the fail-loud preflight (tools/voigt_preflight.py),
#     so a broken build cannot be mistaken for a working one.
#
# libcerf itself is a separate install (README Step 1: build from
# https://jugit.fz-juelich.de/mlz/libcerf.git, `make install DESTDIR=~/.local`).
# Point LIBCERF_PREFIX at that install prefix if it is not the default.
#
# Usage:
#   bash tools/build_voigt.sh            # default LIBCERF_PREFIX=$HOME/.local/usr/local
#   LIBCERF_PREFIX=/opt/libcerf bash tools/build_voigt.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="$REPO_ROOT/gpy_dla_detection"
LIBCERF_PREFIX="${LIBCERF_PREFIX:-$HOME/.local/usr/local}"

INC="$LIBCERF_PREFIX/include"
LIBDIR=""
for d in "$LIBCERF_PREFIX/lib64" "$LIBCERF_PREFIX/lib"; do
    if compgen -G "$d/libcerf.so*" > /dev/null; then LIBDIR="$d"; break; fi
done
if [ -z "$LIBDIR" ]; then
    echo "FATAL: no libcerf.so under $LIBCERF_PREFIX/{lib64,lib}." >&2
    echo "Install libcerf first (README Step 1) or set LIBCERF_PREFIX." >&2
    exit 1
fi
if [ ! -f "$INC/cerf.h" ]; then
    echo "FATAL: cerf.h not found under $INC." >&2
    exit 1
fi

echo "building _voigt.so  (libcerf: $LIBDIR, headers: $INC)"
cc -fPIC -shared -o "$PKG_DIR/_voigt.so" "$PKG_DIR/ctypes_voigt.c" \
   -I"$INC" -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" -lcerf

sha256sum "$PKG_DIR/_voigt.so" "$PKG_DIR/ctypes_voigt.c"
python "$REPO_ROOT/tools/voigt_preflight.py" --repo-root "$REPO_ROOT"
