#!/bin/bash

TERM=xterm
SHELL=/bin/bash
XDG_SESSION_COOKIE=8342d40a26d074e931c6e4e600000004-1393211401.179579-245733898
USER=dosenet/radwatch-airmonitor
MAIL=/var/mail/dosenet/radwatch-airmonitor
#PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games
PATH=/home/dosenet/anaconda3/bin:/home/dosenet/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
PWD=/home/dosenet/radwatch-airmonitor

# Activate the radwatch conda environment
eval "$(conda shell.bash hook)"
conda activate radwatch
LANG=en_US.UTF-8
SHLVL=1
HOME=/home/dosenet/radwatch-airmonitor
LOGNAME=dosenet/radwatch-airmonitor
DISPLAY=localhost:10.0
#_=/usr/bin/env

export PYTHONUNBUFFERED=1

DATA_DIR=/home/dosenet/radwatch-airmonitor/data
LOGFILE=${DATA_DIR}/pipeline.log

# Rotate log if over 1MB
if [ -f "$LOGFILE" ] && [ $(stat -f%z "$LOGFILE" 2>/dev/null || stat -c%s "$LOGFILE" 2>/dev/null) -gt 1048576 ]; then
    mv "$LOGFILE" "${LOGFILE}.old"
fi

exec >> "$LOGFILE" 2>&1

echo ""
echo "========================================"
echo "Pipeline run: $(date)"
echo "========================================"

cd /home/dosenet/radwatch-airmonitor/image_scripts/analysis/

echo "--- weather_gatherer.py ---"
python3 /home/dosenet/radwatch-airmonitor/image_scripts/weather_gatherer.py
WGATHER_EXIT=$?
echo "weather_gatherer.py exit code: $WGATHER_EXIT"

echo ""
echo "--- raw_analysis.py ---"
python3 /home/dosenet/radwatch-airmonitor/image_scripts/analysis/raw_analysis.py
RAW_EXIT=$?
echo "raw_analysis.py exit code: $RAW_EXIT"

echo ""
echo "--- h5_analysis.py ---"
python3 /home/dosenet/radwatch-airmonitor/image_scripts/analysis/h5_analysis.py
H5_EXIT=$?
echo "h5_analysis.py exit code: $H5_EXIT"

# Check for generated output before proceeding
echo ""
if [ $RAW_EXIT -ne 0 ]; then
    echo "WARNING: raw_analysis.py failed (exit $RAW_EXIT), plots may be stale"
fi
if [ $H5_EXIT -ne 0 ]; then
    echo "WARNING: h5_analysis.py failed (exit $H5_EXIT), skipping image deploy"
fi

echo "--- Dropbox status ---"
ls -d "/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/"*/ 2>/dev/null | tail -5
echo "Total date dirs: $(ls -d '/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/'*/ 2>/dev/null | wc -l)"
echo "last_processed.txt: $(cat ${DATA_DIR}/last_processed.txt 2>/dev/null || echo 'MISSING')"
echo "rebin.h5 size: $(ls -lh ${DATA_DIR}/rebin.h5 2>/dev/null | awk '{print $5}' || echo 'MISSING')"
echo "Generated PNGs: $(ls ${DATA_DIR}/*.png 2>/dev/null | wc -l)"

# Stage output files for deployment
mkdir -p "rooftop_tmp"
convert -geometry 300x220+0+0 ${DATA_DIR}/iso_One_Day.png ${DATA_DIR}/iso_One_Day_small.png 2>/dev/null || echo "WARNING: convert failed (imagemagick)"

# Copy (not move) PNGs and weather CSV to staging directory
cp ${DATA_DIR}/*.png ./rooftop_tmp/ 2>/dev/null
cp ${DATA_DIR}/weather_sorted.csv ./rooftop_tmp/ 2>/dev/null

echo ""
echo "--- Deploying via SFTP ---"
if [ -z "$RADWATCH_SFTP_PASS" ]; then
    echo "WARNING: RADWATCH_SFTP_PASS not set, skipping SFTP upload"
    echo "  Set it with: export RADWATCH_SFTP_PASS='your-password'"
    echo "  Or add to crontab: RADWATCH_SFTP_PASS=your-password"
else
    lftp -e "set sftp:auto-confirm yes; mirror -Rnv /home/dosenet/radwatch-airmonitor/image_scripts/analysis/rooftop_tmp /test/; quit;" -u coeradwatch-RADWATCH,"${RADWATCH_SFTP_PASS}" sftp://coeradwatch.sftp.wpengine.com:2222
fi