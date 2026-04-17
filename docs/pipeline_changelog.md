# Analysis Pipeline Changelog

## 2026-04-16: Dead Code Cleanup, Lazy Imports, and Conda Environment

### Problem
The codebase had accumulated dead code from testing and development: commented-out
becquerel and xylib implementations inside `'''` blocks, unused imports, old debugging
prints, and legacy collection-building code. Several top-level imports pulled in
packages (`becquerel`, `scipy`, `matplotlib`) that were only used by dead code or
interactive notebooks, not the production pipeline. The setup script used bare
`pip install` which risked conflicting with other projects on the server.

### Changes

#### `image_scripts/spectra_utils.py`
- Removed `import becquerel as bq` and `#import xylib` -- all usage was in dead code
- Removed helper functions `_meta_to_dict()` and `_read_first_block_xy()` -- only
  used by the removed xylib/becquerel code paths
- Removed two `'''...'''` blocks: old becquerel implementations of `parse_spectra`
  and `load_xy` (the active versions use `cnf_parser_standalone`)

#### `image_scripts/sample_collection.py`
- Removed `from itertools import chain` -- never used anywhere
- Removed old `'''...'''` `write_hdf` block with hardcoded `'2014'` group name
  (superseded by the current `write_hdf` that uses a `'data'` group)

#### `image_scripts/analysis/raw_analysis.py`
- Removed three blocks of commented-out code:
  - Old `build_collection`/`rebin`/`write_hdf` calls (superseded by incremental pipeline)
  - Diagnostic print blocks (used during channel standardization debugging)
  - Old `write_spe`/`write_last_update_image` calls (now use `DATA_DIR` paths)

#### `image_scripts/analysis/h5_analysis.py`
- Removed five small commented-out blocks: old color palette file loading, old
  efficiency correction loop, old weather plotting loop, old gridspec line,
  old windrose call

#### `image_scripts/spectrum_calibration.py`
- Moved `scipy.signal.find_peaks`, `scipy.optimize.curve_fit`, and
  `matplotlib.pyplot` from top-level imports to lazy imports inside the functions
  that actually use them
- Pipeline functions (`read_calibration_file`, `apply_calibration`,
  `energy_to_channel`) no longer require scipy or matplotlib at import time
- Result: `scipy` is now optional (only needed for interactive calibration notebooks)

#### New files
- **`environment.yml`**: Conda environment spec (name: `radwatch`) with required
  packages (numpy, h5py, matplotlib, pandas, Pillow, pytz). scipy listed as optional.
- **`requirements.txt`**: Pip fallback with the same package list.

#### `setup.sh`
- Rewritten to create/update the `radwatch` conda env from `environment.yml`
  instead of bare `pip install`
- Import tests now run inside the activated conda env

#### `image_scripts/analysis/cron_job.sh`
- Added `conda activate radwatch` after PATH setup to isolate the pipeline from
  the system Python

#### Documentation
- Updated `README.md` with proper project overview, setup instructions, and
  project structure
- Updated `docs/architecture.md` with conda env references, lazy import notes,
  and new files in the file reference table
- Updated `LOGBOOK.md` with session entry

### Verification
- `python3 -c "import spectra_utils"` -- succeeds without becquerel/xylib
- `python3 -c "from image_scripts.spectrum_calibration import read_calibration_file"` --
  succeeds without scipy
- Full pipeline: `bash cron_job.sh`

---

## 2026-02-23: Data Quality Checks, ROI Storage, and CSV Expansion

### Problem
Time series plots showed anomalous near-zero K-40 count rates at several points
(visible around Jan 25 - Feb 5). The existing NaN filter in `h5_analysis.py` only
caught samples with `live_time <= 0` or completely empty spectra -- it did not
catch samples with real data but anomalously low counts in specific peaks.

Additionally, ROI counts were computed on-the-fly every time `h5_analysis.py` ran
but were never stored persistently, and the CSV output only included 4 of 7 isotopes.

### Root Cause
Detector artifacts (e.g., brief power interruptions, gain shifts) can produce
spectra with valid total counts but anomalously low counts in individual peaks.
The previous filter (`live_time <= 0 or sum(spectra) == 0`) was too coarse to
detect these per-peak anomalies.

### Changes

#### `image_scripts/analysis/raw_analysis.py`
- Added a **two-layer QA filter** on K-40 count rate (ROI index 4):
  1. **Absolute floor** (`K40_MIN_RATE = 0.05` counts/sec): catches obviously bad
     data regardless of rolling median state. This is critical for the start of a
     fresh rebuild where the rolling median has no good baseline yet — without it,
     bad early samples would establish a low median and pass through undetected.
     The absolute check runs first and skips the flagged sample so it does not
     pollute the rolling median window for subsequent samples.
  2. **Rolling median** (50% threshold, 24-sample window): catches subtler
     anomalies where the rate is above the absolute floor but is a sudden drop
     relative to recent history. Uses an expanding window at edges.
  - Appends flagged samples to `qa_flagged.csv` with columns:
    `timestamp, k40_rate, rolling_median, reason`.
- For the incremental path: uses the last 24 existing samples' K-40 rates to
  seed the rolling median baseline before filtering new data.
- **Computes ROI counts** for all 7 peaks after rebinning and passes them to
  `write_hdf()` for persistent storage.

#### `image_scripts/sample_collection.py`
- **`write_hdf()`** extended with `roi_data` and `roi_labels` parameters:
  - Creates `data/roi_counts` dataset (shape `(N, 7, 2)`) in HDF5.
  - Creates `data/roi_labels` dataset with isotope name strings.
- **`read_hdf()`** extended to load `roi_counts` and `roi_labels` when present,
  storing them on the collection object for use during incremental merges.

#### `image_scripts/analysis/h5_analysis.py`
- **Uses stored ROI data** from HDF5 when available (`data/roi_counts`), falling
  back to on-the-fly computation if the dataset is absent.
- **Expanded CSV output**: removed the `if y == 0 or y == 2 or y == 3: continue`
  filter. Both `weather.csv` and `weather_bq.csv` now include all 7 isotopes:
  `Time, Pb214, Pb214_err, Bi214, Bi214_err, Pb212, Pb212_err, Tl208, Tl208_err, K40, K40_err, Cs134, Cs134_err, Cs137, Cs137_err`

---

## 2026-02-22: Channel Standardization Fix

### Problem
After the detector firmware was updated, new spectral files contained 4096
channels instead of 8192, causing shape mismatches during rebinning (array
concatenation failed with incompatible shapes).

### Root Cause
The detector's MCA was reconfigured from 8192 to 4096 channels. Raw `.CNF` files
from before and after the change had different channel counts, and the pipeline
assumed a uniform channel count across all samples.

### Solution
Added `SampleCollection.standardize_channel_counts()` in `sample_collection.py`:
- Detects all unique channel counts in the collection.
- Upsamples smaller spectra (e.g., 4096 -> 8192) by splitting each bin's counts
  across sub-bins, preserving total counts.
- Adjusts the energy calibration slope (`bin_cal[1]`) proportionally.
- Called in `raw_analysis.py` after building the collection but before rebinning.

### Verification
- Delete `rebin.h5` and `last_processed.txt`, rebuild from scratch.
- All samples should have uniform 8192-channel spectra after standardization.
