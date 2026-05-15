#!/bin/bash
# Deploy rooftop_tmp/ to the web server via SFTP
#
# Usage:
#   bash deploy.sh
#
# Password handling:
#   RADWATCH_SFTP_PASS is read from the environment first. If unset,
#   ~/.radwatch_env is sourced as a fallback (this is the durable
#   location written by setup.sh). To change the password later, edit
#   ~/.radwatch_env directly — it's a normal shell file with one line:
#     export RADWATCH_SFTP_PASS='...'
#   For a one-off override (e.g. testing a different account), export
#   the variable in the calling shell before invoking this script.
#   See docs/runbook.md §5 for the full setup and troubleshooting story.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/rooftop_tmp"
REMOTE_DIR="/test/"
SFTP_USER="coeradwatch-RADWATCH"
SFTP_HOST="coeradwatch.sftp.wpengine.com"
SFTP_PORT=2222

# Fall back to ~/.radwatch_env only if the var isn't already set.
# This preserves any one-off override from the calling shell or crontab.
if [ -z "${RADWATCH_SFTP_PASS:-}" ] && [ -f "$HOME/.radwatch_env" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$HOME/.radwatch_env"
fi

if [ -z "${RADWATCH_SFTP_PASS:-}" ]; then
    echo "ERROR: RADWATCH_SFTP_PASS is not set and ~/.radwatch_env did not define it."
    echo "  Run 'bash setup.sh' from the repo root to be prompted for the password,"
    echo "  or see docs/runbook.md §5 for manual setup."
    exit 1
fi

if [ ! -d "$LOCAL_DIR" ]; then
    echo "ERROR: ${LOCAL_DIR} does not exist"
    exit 1
fi

echo "Deploying ${LOCAL_DIR} -> ${SFTP_HOST}:${REMOTE_DIR}"
echo "Files to upload:"
ls -1 "${LOCAL_DIR}"/*.png "${LOCAL_DIR}"/*.csv 2>/dev/null | wc -l

lftp <<EOF
set sftp:auto-confirm yes
open -u ${SFTP_USER},${RADWATCH_SFTP_PASS} sftp://${SFTP_HOST}:${SFTP_PORT}
mirror -Rnv --no-perms ${LOCAL_DIR} ${REMOTE_DIR}
quit
EOF

echo "Deploy exit code: $?"
