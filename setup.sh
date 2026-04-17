#!/bin/bash
# RadWatch Air Monitor - Server Setup Script
#
# Run this after a fresh git clone on a new server to set up the
# analysis pipeline. Assumes Ubuntu/Debian with Anaconda3 installed.
#
# Usage:
#   bash setup.sh
#
# After running this script you still need to:
#   1. Set up Dropbox sync for the spectral data directory
#   2. Install the crontab (see step at end of script output)
#   3. Verify weatherhawk.csv exists or run weather_gatherer.py manually

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${REPO_DIR}/data"
ANALYSIS_DIR="${REPO_DIR}/image_scripts/analysis"
CONDA_ENV="radwatch"

echo "========================================"
echo "RadWatch Air Monitor - Server Setup"
echo "========================================"
echo "Repository: ${REPO_DIR}"
echo ""

# ---- 1. Check conda ----
echo "--- Checking conda ---"
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Anaconda3 or Miniconda."
    exit 1
fi
echo "Found: $(conda --version)"

# ---- 2. Create / update conda environment ----
echo ""
echo "--- Setting up conda environment '${CONDA_ENV}' ---"
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Environment '${CONDA_ENV}' already exists, updating..."
    conda env update -f "${REPO_DIR}/environment.yml" --prune
else
    echo "Creating environment '${CONDA_ENV}'..."
    conda env create -f "${REPO_DIR}/environment.yml"
fi

# Activate the environment for the rest of setup
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
echo "Activated: ${CONDA_ENV}"

PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python: ${PYTHON_VERSION}"

echo ""
echo "--- Installed package versions ---"
python3 -c "
import numpy; print(f'  numpy:      {numpy.__version__}')
import matplotlib; print(f'  matplotlib: {matplotlib.__version__}')
import h5py; print(f'  h5py:       {h5py.__version__}')
import pandas; print(f'  pandas:     {pandas.__version__}')
import pytz; print(f'  pytz:       {pytz.__version__}')
try:
    import scipy; print(f'  scipy:      {scipy.__version__} (optional)')
except ImportError:
    print('  scipy:      not installed (optional, for notebooks)')
"

# ---- 3. Create data directory ----
echo ""
echo "--- Setting up data directory ---"
mkdir -p "${DATA_DIR}"
echo "Created: ${DATA_DIR}"

# ---- 4. Verify Dropbox spectral data directory ----
echo ""
echo "--- Checking Dropbox spectral data ---"
SPEC_DIR="/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/"
if [ -d "$SPEC_DIR" ]; then
    CNF_COUNT=$(find "$SPEC_DIR" -name "*.CNF" 2>/dev/null | wc -l)
    DIR_COUNT=$(find "$SPEC_DIR" -maxdepth 1 -type d -name "20*" 2>/dev/null | wc -l)
    echo "Spectral data directory exists: ${SPEC_DIR}"
    echo "  Date directories: ${DIR_COUNT}"
    echo "  CNF files: ${CNF_COUNT}"
else
    echo "WARNING: Spectral data directory not found: ${SPEC_DIR}"
    echo "  You need to set up Dropbox sync before the pipeline can process data."
    echo "  Install Dropbox and sync: 'UCB Air Monitor/Data/Roof/current/'"
fi

# ---- 5. Check weather data ----
echo ""
echo "--- Checking weather data ---"
WEATHER_CSV="${REPO_DIR}/weatherhawk.csv"
if [ -f "$WEATHER_CSV" ]; then
    LINE_COUNT=$(wc -l < "$WEATHER_CSV")
    echo "Weather CSV exists: ${WEATHER_CSV} (${LINE_COUNT} lines)"
else
    echo "WARNING: weatherhawk.csv not found at ${WEATHER_CSV}"
    echo "  The weather_gatherer.py script will create it on first run."
    echo "  You can also copy it from a backup if available."
fi

# ---- 6. Check calibration files ----
echo ""
echo "--- Checking calibration files ---"
CAL_FILE="${REPO_DIR}/image_scripts/calibration/calibration_coefficients.txt"
ROI_FILE="${ANALYSIS_DIR}/roi_energy.dat"
ECC_FILE="${ANALYSIS_DIR}/roof.ecc"

for f in "$CAL_FILE" "$ROI_FILE" "$ECC_FILE"; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename $f)"
    else
        echo "  MISSING: $f"
    fi
done

# ---- 7. Check system tools ----
echo ""
echo "--- Checking system tools ---"
for cmd in convert lftp; do
    if command -v $cmd &>/dev/null; then
        echo "  OK: $cmd"
    else
        echo "  MISSING: $cmd"
        case $cmd in
            convert) echo "    Install with: sudo apt-get install imagemagick" ;;
            lftp)    echo "    Install with: sudo apt-get install lftp" ;;
        esac
    fi
done

# ---- 8. Make cron script executable ----
echo ""
echo "--- Setting permissions ---"
chmod +x "${ANALYSIS_DIR}/cron_job.sh"
echo "Made cron_job.sh executable"

# ---- 9. Test import of core modules ----
echo ""
echo "--- Testing Python imports (in '${CONDA_ENV}' env) ---"
cd "${ANALYSIS_DIR}"
python3 -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
sys.path.insert(0, '${REPO_DIR}/image_scripts')
sys.path.insert(0, '..')
sys.path.insert(0, '.')

errors = []
for mod_name, import_str in [
    ('sample_collection', 'from image_scripts import sample_collection'),
    ('weather_utils',     'from image_scripts import weather_utils'),
    ('time_utils',        'from image_scripts import time_utils'),
    ('spectra_utils',     'import spectra_utils'),
    ('cnf_parser',        'import cnf_parser_standalone'),
    ('spectrum_calibration', 'from image_scripts.spectrum_calibration import read_calibration_file'),
]:
    try:
        exec(import_str)
        print(f'  OK: {mod_name}')
    except Exception as e:
        print(f'  FAIL: {mod_name} — {e}')
        errors.append(mod_name)

if errors:
    print(f'\nWARNING: {len(errors)} module(s) failed to import.')
    print('The pipeline may not work until these are resolved.')
else:
    print('\nAll modules imported successfully.')
"

# ---- 10. Summary ----
echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Ensure Dropbox is syncing spectral data to:"
echo "     ${SPEC_DIR}"
echo ""
echo "  2. Install the crontab (runs pipeline hourly):"
echo "     crontab ${REPO_DIR}/crontab.txt"
echo ""
echo "  3. Test the pipeline manually:"
echo "     cd ${ANALYSIS_DIR}"
echo "     bash cron_job.sh"
echo "     tail -50 ${DATA_DIR}/pipeline.log"
echo ""
echo "  4. Check that plots were generated:"
echo "     ls -la ${ANALYSIS_DIR}/rooftop_tmp/*.png"
echo ""
echo "  For crontab, ensure the conda env is activated:"
echo "     The cron_job.sh script handles this automatically."
echo ""
echo "See docs/architecture.md for full system documentation."
