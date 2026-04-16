#!/bin/bash
# rsync_mock_catalogs.sh — Download mock truth-catalog files from NERSC
#
# Downloads the three key catalog files for each mock into a structured
# local directory that mirrors the remote hierarchy:
#
#   data/mocks/{suite}/{version}/{mock-N}/
#       zcat.fits            QSO redshift catalog
#       bal_cat.fits         BAL catalog
#       dla_cat.fits         DLA truth catalog  (London mocks)
#       hcd_truth_cat.fits   HCD truth catalog  (Saclay mocks)
#
# Usage:
#   bash data/scripts/rsync_mock_catalogs.sh [OPTIONS]
#
# Options:
#   --dry-run            Print rsync commands without executing them
#   --host HOST          NERSC SSH host (default: dtn01.nersc.gov)
#   --local-base DIR     Local root for all mocks (default: data/mocks)
#   --no-bal             Skip bal_cat.fits (useful if the file doesn't exist yet)
#   -h, --help           Show this help message
#
# To add a new mock, append one sync_mock call to the "Mock inventory" section.
# The function signature is:
#   sync_mock  SUITE  VERSION  MOCK  REMOTE_DIR  TRUTH_CAT
#
# Requirements:
#   - SSH key for NERSC already configured (~/.ssh/config or ssh-agent)
#   - rsync available locally
#
# Example (dry run first, then real):
#   bash data/scripts/rsync_mock_catalogs.sh --dry-run
#   bash data/scripts/rsync_mock_catalogs.sh

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────

NERSC_HOST="dtn01.nersc.gov"
# LOCAL_BASE is set relative to the repo root.  The script resolves it to an
# absolute path so it works regardless of where you call it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_BASE="${REPO_ROOT}/data/mocks"

DRY_RUN=0
SKIP_BAL=0

# ── Argument parsing ─────────────────────────────────────────────────────────

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --no-bal)    SKIP_BAL=1; shift ;;
        --host)      NERSC_HOST="$2"; shift 2 ;;
        --local-base) LOCAL_BASE="$2"; shift 2 ;;
        -h|--help)   usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Logging setup ────────────────────────────────────────────────────────────

LOG_DIR="${LOCAL_BASE}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/rsync_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG_FILE}"; }
log_ok()   { log "  ✓ $*"; }
log_skip() { log "  - $*"; }
log_err()  { log "  ✗ ERROR: $*"; }

RSYNC_FLAGS=(-avP --checksum)
if [[ $DRY_RUN -eq 1 ]]; then
    RSYNC_FLAGS+=(--dry-run)
    log "DRY-RUN mode — no files will be transferred"
fi

# ── Transfer counters ────────────────────────────────────────────────────────

N_OK=0
N_SKIP=0
N_ERR=0

# ── Core helper ──────────────────────────────────────────────────────────────
#
# rsync_one SRC_REMOTE LOCAL_DIR FILENAME
#   SRC_REMOTE  : full remote path including filename
#   LOCAL_DIR   : local destination directory (created if absent)
#   FILENAME    : display name for log messages
#
rsync_one() {
    local src="$1"
    local local_dir="$2"
    local fname="$3"

    mkdir -p "${local_dir}"
    local dst="${local_dir}/${fname}"

    log "  rsync ${NERSC_HOST}:${src}  →  ${dst}"
    if rsync "${RSYNC_FLAGS[@]}" "${NERSC_HOST}:${src}" "${dst}" >> "${LOG_FILE}" 2>&1; then
        log_ok "${fname}"
        (( N_OK++ )) || true
    else
        log_err "${fname} (rsync exit $?)"
        (( N_ERR++ )) || true
    fi
}

# ── Per-mock orchestrator ─────────────────────────────────────────────────────
#
# sync_mock SUITE VERSION MOCK REMOTE_DIR TRUTH_CAT
#   SUITE      : "london" | "saclay" | any future suite name
#   VERSION    : e.g. "v5.9.5"
#   MOCK       : e.g. "mock-0"
#   REMOTE_DIR : full remote directory path (no trailing slash)
#   TRUTH_CAT  : truth catalog filename, e.g. "dla_cat.fits" or "hcd_truth_cat.fits"
#
sync_mock() {
    local suite="$1"
    local version="$2"
    local mock="$3"
    local remote_dir="$4"
    local truth_cat="$5"

    local local_dir="${LOCAL_BASE}/${suite}/${version}/${mock}"

    log ""
    log "── ${suite}/${version}/${mock} ─────────────────────────────────"
    log "   remote : ${remote_dir}"
    log "   local  : ${local_dir}"

    # QSO redshift catalog — always present
    rsync_one "${remote_dir}/zcat.fits"     "${local_dir}" "zcat.fits"

    # BAL catalog — may not exist for all mocks
    if [[ $SKIP_BAL -eq 0 ]]; then
        rsync_one "${remote_dir}/bal_cat.fits"  "${local_dir}" "bal_cat.fits"
    else
        log_skip "bal_cat.fits (--no-bal)"
        (( N_SKIP++ )) || true
    fi

    # DLA / HCD truth catalog (name differs by suite)
    rsync_one "${remote_dir}/${truth_cat}"  "${local_dir}" "${truth_cat}"

    # Write a sidecar provenance file so you always know where a mock came from
    if [[ $DRY_RUN -eq 0 ]]; then
        cat > "${local_dir}/SOURCE.txt" <<PROV
suite        : ${suite}
version      : ${version}
mock         : ${mock}
remote_host  : ${NERSC_HOST}
remote_dir   : ${remote_dir}
truth_catalog: ${truth_cat}
synced_at    : $(date --iso-8601=seconds)
PROV
    fi
}

# ── Mock inventory ────────────────────────────────────────────────────────────
#
# Add one sync_mock line per mock.  Columns:
#   SUITE    VERSION   MOCK     REMOTE_DIR                                                                                              TRUTH_CAT
#
# London mocks — v5.9.5 — DLA truth file is "dla_cat.fits"
sync_mock \
    london  v5.9.5  mock-0 \
    /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124 \
    dla_cat.fits

sync_mock \
    london  v5.9.5  mock-1 \
    /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-1/jura-124 \
    dla_cat.fits

# Saclay mocks — v4.7.5 — HCD truth file is "hcd_truth_cat.fits"
sync_mock \
    saclay  v4.7.5  mock-0 \
    /global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124 \
    hcd_truth_cat.fits

sync_mock \
    saclay  v4.7.5  mock-1 \
    /global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-1/juraLy8-124 \
    hcd_truth_cat.fits

# ── To add a new mock, copy one of the blocks above and adjust the four fields.
# Example (London v6):
#
# sync_mock \
#     london  v6.0.0  mock-0 \
#     /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v6.0.0/mock-0/jura-124 \
#     dla_cat.fits

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log "  Done.  OK=${N_OK}  skipped=${N_SKIP}  errors=${N_ERR}"
log "  Log: ${LOG_FILE}"
log "═══════════════════════════════════════════════════"

[[ $N_ERR -eq 0 ]]   # exit non-zero if any transfer failed
