# Analysis Pipeline Changelog

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
