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
# SSH multiplexing is used so you authenticate (password + MFA) only once
# at the start; all rsync transfers reuse the same open connection.
#
# Usage:
#   bash data/scripts/rsync_mock_catalogs.sh [OPTIONS]
#
# Options:
#   --dry-run            Print rsync commands without executing them
#   --host HOST          NERSC SSH host (default: jibancat@dtn01.nersc.gov)
#   --local-base DIR     Local root for all mocks (default: data/mocks)
#   --no-bal             Skip bal_cat.fits (useful if the file doesn't exist yet)
#   -h, --help           Show this help message
#
# To add a new mock, append one sync_mock call to the "Mock inventory" section.
# The function signature is:
#   sync_mock  SUITE  VERSION  MOCK  REMOTE_DIR  TRUTH_CAT
#
# Requirements:
#   - rsync and ssh available locally
#   - NERSC credentials (password + MFA) — entered once at startup
#
# Example (dry run first, then real):
#   bash data/scripts/rsync_mock_catalogs.sh --dry-run
#   bash data/scripts/rsync_mock_catalogs.sh

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────

NERSC_HOST="jibancat@dtn01.nersc.gov"
# LOCAL_BASE is resolved to an absolute path so the script works regardless
# of where you call it from.
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
        --dry-run)    DRY_RUN=1; shift ;;
        --no-bal)     SKIP_BAL=1; shift ;;
        --host)       NERSC_HOST="$2"; shift 2 ;;
        --local-base) LOCAL_BASE="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Logging setup ────────────────────────────────────────────────────────────

LOG_DIR="${LOCAL_BASE}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/rsync_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

log()      { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG_FILE}"; }
log_ok()   { log "  ✓ $*"; }
log_skip() { log "  - $*"; }
log_err()  { log "  ✗ ERROR: $*"; }

# ── SSH multiplexing — authenticate once, reuse for all transfers ─────────────
#
# ControlMaster=auto  : first connection becomes the master
# ControlPersist=10m  : master stays open 10 min after last use
# ControlPath         : socket file shared by all rsync calls

CTRL_SOCKET="/tmp/nersc_rsync_${TIMESTAMP}_$$.sock"
SSH_MUX_OPTS="-o ControlMaster=auto -o ControlPath=${CTRL_SOCKET} -o ControlPersist=10m"

if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN mode — printing commands only, no network connection made."
else
    log "Opening SSH master connection to ${NERSC_HOST} ..."
    log "(You will be prompted for your password + MFA once.)"
    # -N : no remote command; -f : go to background after auth
    ssh ${SSH_MUX_OPTS} -N -f "${NERSC_HOST}"
    log "SSH master connection established — all transfers will reuse it."
fi

# Close the master connection on exit (normal or error)
cleanup() {
    if [[ $DRY_RUN -eq 0 ]]; then
        log "Closing SSH master connection..."
        ssh -o "ControlPath=${CTRL_SOCKET}" -O exit "${NERSC_HOST}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── rsync flags (used only in live mode) ─────────────────────────────────────

RSYNC_FLAGS=(-avP --checksum -e "ssh ${SSH_MUX_OPTS}")

# ── Transfer counters ────────────────────────────────────────────────────────

N_OK=0
N_SKIP=0
N_ERR=0

# ── Core helper ──────────────────────────────────────────────────────────────
#
# rsync_one REMOTE_PATH LOCAL_DIR FILENAME
#
rsync_one() {
    local src="$1"
    local local_dir="$2"
    local fname="$3"

    mkdir -p "${local_dir}"
    local dst="${local_dir}/${fname}"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [would rsync] ${NERSC_HOST}:${src}  →  ${dst}"
        (( N_SKIP++ )) || true
        return
    fi

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

    rsync_one "${remote_dir}/zcat.fits"      "${local_dir}" "zcat.fits"

    if [[ $SKIP_BAL -eq 0 ]]; then
        rsync_one "${remote_dir}/bal_cat.fits"   "${local_dir}" "bal_cat.fits"
    else
        log_skip "bal_cat.fits (--no-bal)"
        (( N_SKIP++ )) || true
    fi

    rsync_one "${remote_dir}/${truth_cat}"   "${local_dir}" "${truth_cat}"

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
#   SUITE    VERSION   MOCK     REMOTE_DIR                                   TRUTH_CAT
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

# ── To add a new mock, copy one block above and adjust the fields.
# Example (London v6):
#
# sync_mock \
#     london  v6.0.0  mock-0 \
#     /global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v6.0.0/mock-0/jura-124 \
#     dla_cat.fits

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
if [[ $DRY_RUN -eq 1 ]]; then
    log "  Done (dry run).  would_transfer=${N_SKIP}"
else
    log "  Done.  OK=${N_OK}  skipped=${N_SKIP}  errors=${N_ERR}"
fi
log "  Log: ${LOG_FILE}"
log "═══════════════════════════════════════════════════"

[[ $N_ERR -eq 0 ]]
