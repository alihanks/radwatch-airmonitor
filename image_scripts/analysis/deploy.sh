#!/bin/bash
# Deploy rooftop_tmp/ to the web server via SFTP
#
# Usage:
#   bash deploy.sh
#
# Requires RADWATCH_SFTP_PASS environment variable, or set it in crontab.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/rooftop_tmp"
REMOTE_DIR="/test/"
SFTP_USER="coeradwatch-RADWATCH"
SFTP_HOST="coeradwatch.sftp.wpengine.com"
SFTP_PORT=2222

if [ -z "$RADWATCH_SFTP_PASS" ]; then
    echo "ERROR: RADWATCH_SFTP_PASS not set"
    echo "  export RADWATCH_SFTP_PASS='your-password'"
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
mirror -Rnv ${LOCAL_DIR} ${REMOTE_DIR}
quit
EOF

echo "Deploy exit code: $?"
