# RadWatch Air Monitor - System Architecture

This document describes how the RadWatch rooftop air monitor pipeline works, from data acquisition through to the published plots on the website.

## System Overview

The RadWatch air monitor is a gamma-ray spectroscopy system on the UC Berkeley rooftop. A detector continuously collects gamma spectra (saved as `.CNF` files by the MCA hardware), while a nearby WeatherHawk station records weather conditions. A server (`dosenet`) runs a cron job every hour that processes new spectral data, correlates it with weather, and publishes time-series plots to a website via SFTP. The pipeline runs inside an isolated `radwatch` conda environment (see `environment.yml`).

```
  Detector (MCA)          WeatherHawk Station
       |                         |
       v                         v
  .CNF files              Weather Underground
  (via Dropbox)             (web scraping)
       |                         |
       v                         v
  +-----------+           weatherhawk.csv
  | Dropbox   |                  |
  | sync      |                  |
  +-----------+                  |
       |                         |
       +------------+------------+
                    |
              cron_job.sh (hourly)
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
   weather_     raw_         h5_
   gatherer   analysis     analysis
     .py        .py          .py
        |           |           |
        v           v           v
   weatherhawk  rebin.h5    iso_*.png
   .csv         (HDF5)      rose_*.png
                            weather.csv
                                |
                                v
                          SFTP upload
                          to website
```

## Data Flow in Detail

### Stage 1: Data Acquisition (continuous, external)

**Gamma Spectra:**
- The MCA hardware saves one `.CNF` file per collection interval (~5 minutes)
- Files are organized in date-stamped directories: `YYYY-MM-DD_HH-MM-SS/filename.CNF`
- Dropbox syncs these from the detector PC to the server at:
  `/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/`

**Weather Data:**
- `weather_gatherer.py` scrapes today's weather from WeatherUnderground (station `KCABERKE272`)
- Appends rows to `/home/dosenet/radwatch-airmonitor/weatherhawk.csv`
- Columns: temperature, humidity, pressure, solar radiation, wind speed/direction, rain

### Stage 2: Raw Analysis (`raw_analysis.py`)

This script turns raw `.CNF` files + weather CSV into a clean, rebinned HDF5 database.

```
weatherhawk.csv -----> resort_weather_timestamps() ----> weather_sorted.csv
                                                              |
Dropbox/current/ ----> build_collection_incremental() --------+
  *.CNF files              |                                   |
                           v                                   |
                    Parse each CNF file              Match weather by
                    (cnf_parser_standalone)           timestamp range
                           |                                   |
                           +-----------------------------------+
                           |
                    standardize_channel_counts()
                    (upsample 4096 -> 8192 channels)
                           |
                    estimate_livetimes_from_k40()
                    (correct short-collection livetimes)
                           |
                    rebin(1 hour)
                    (average spectra into hourly bins)
                           |
                    qa_filter_k40()
                    (remove anomalous samples)
                           |
                    compute_roi_counts()
                    (extract isotope count rates)
                           |
                    write_hdf() ----> data/rebin.h5
```

**Incremental processing:**
- On first run (no `rebin.h5`): processes all available CNF files
- On subsequent runs: reads `last_processed.txt` to find only new files
- New data is merged into the existing HDF5 collection
- ROI counts are recomputed for the full dataset

**Quality assurance (QA):**
- **K-40 livetime correction:** Uses K-40 gross counts as a proxy for true collection time. If a spectrum has fewer K-40 counts than expected for its preset time, the livetime is scaled down proportionally.
- **Absolute floor:** Samples with K-40 rate below 0.15 counts/sec are flagged (steady-state is ~0.33)
- **Rolling median:** Samples below 50% of the trailing 24-hour K-40 median are flagged
- Flagged samples are logged to `data/qa_flagged.csv` and excluded from the dataset

### Stage 3: Plot Generation (`h5_analysis.py`)

This script reads `rebin.h5` and produces all visualization outputs.

```
data/rebin.h5 -------> Load timestamps, spectra, weather, ROI counts
                             |
data/weather_sorted.csv --> build_merged_weather()
                             |  (fill radiation gaps with continuous
                             |   weather CSV data)
                             |
                +------------+------------+
                |            |            |
                v            v            v
          Most recent   Isotope time   Wind roses
          spectrum      series plots   (per time window)
          plot          (per time window)
                |            |            |
                v            v            v
          most_recent_  iso_One_Day   rose_One_Day
          spectra.png   iso_One_Week  rose_One_Week
                        iso_One_Month rose_One_Month
                        iso_One_Year  rose_One_Year
```

**Isotope time-series plots (`iso_*.png`):**
Each plot has 3 panels:
1. **Top:** Temperature + barometric pressure (weather timeline)
2. **Middle:** Isotope count rates with error bars (radiation timeline with NaN gap breaks)
3. **Bottom:** Rainfall + solar radiation (weather timeline)

Weather and radiation use independent timelines: weather is plotted continuously (merged HDF5 + CSV data), while radiation shows gaps where the detector was offline.

**Tracked isotopes (7 ROIs defined in `roi_energy.dat`):**
Pb-214, Bi-214, Pb-212, Tl-208, K-40, Cs-134, Cs-137

### Stage 4: Deployment (cron_job.sh)

```
data/*.png ---------> rooftop_tmp/ ---------> SFTP upload
data/weather_sorted.csv --> rooftop_tmp/      to website
```

- PNGs are moved (not copied) from `data/` to `rooftop_tmp/`
- `lftp` mirrors `rooftop_tmp/` to the web host via SFTP
- All pipeline output is logged to `data/pipeline.log`

## HDF5 Schema (`rebin.h5`)

All data is stored under a single group named `data`:

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `timestamps` | (N,) | uint32 | UTC epoch seconds |
| `spectra` | (N, 8192) | uint32 | Raw channel counts per hourly bin |
| `spectra_meta` | (N, 4) | float | [real_time, live_time, cal_offset, cal_slope] |
| `weather_data` | (N, 7) | float | [temp, pressure, solar, RH, wind_dir_weighted, wind_speed, rain] |
| `roi_counts` | (N, 7, 2) | float | [net_counts, error] per ROI per sample |
| `roi_labels` | (7,) | string | Isotope names |
| `k40_median_counts` | scalar | float | Baseline K-40 gross counts for livetime estimation |

## Cron Schedule

From `crontab.txt`:
```
# Every hour at :00 — run the full pipeline
0 * * * * /home/dosenet/radwatch-airmonitor/image_scripts/analysis/cron_job.sh

# Every hour at :00 — clean up mail
0 * * * * /home/dosenet/radwatch-airmonitor/admin_stuff/mail/del_mail.sh

# Every hour at :18 — check Dropbox is alive
18 * * * * /home/dosenet/radwatch-airmonitor/admin_stuff/dropbox_alive/alive.sh
```

## Key File Reference

### Pipeline scripts (run order)
| File | Purpose |
|------|---------|
| `image_scripts/analysis/cron_job.sh` | Orchestrates the pipeline: activates `radwatch` conda env, runs scripts, logs to `data/pipeline.log` |
| `image_scripts/weather_gatherer.py` | Scrapes WeatherUnderground, appends to `weatherhawk.csv` |
| `image_scripts/analysis/raw_analysis.py` | Processes CNF files, builds/updates `rebin.h5` |
| `image_scripts/analysis/h5_analysis.py` | Reads `rebin.h5`, generates all plots |

### Core libraries
| File | Purpose |
|------|---------|
| `image_scripts/sample_collection.py` | HDF5 I/O, rebinning, channel standardization, incremental processing |
| `image_scripts/spectra_utils.py` | CNF file parsing (via `cnf_parser_standalone`), ROI parsing |
| `image_scripts/spectrum_calibration.py` | Energy calibration from coefficients file; scipy/matplotlib lazy-imported for notebook use |
| `image_scripts/weather_utils.py` | Weather CSV parsing, wind rose drawing |
| `image_scripts/time_utils.py` | Time window definitions (1 day, 1 week, 1 month, 1 year) |
| `image_scripts/cnf_parser_standalone.py` | Standalone CNF binary file reader |

### Configuration & environment files
| File | Purpose |
|------|---------|
| `image_scripts/analysis/roi_energy.dat` | ROI windows in keV (energy-based, current) |
| `image_scripts/analysis/roi.dat` | ROI windows in channels (legacy) |
| `image_scripts/analysis/roof.ecc` | Detector efficiency calibration |
| `image_scripts/calibration/calibration_coefficients.txt` | Energy calibration: offset + slope (keV/channel) |
| `environment.yml` | Conda environment spec (env name: `radwatch`) |
| `requirements.txt` | Pip fallback dependency list |
| `setup.sh` | Server setup script: creates conda env, checks prerequisites, tests imports |

### Data files
| File | Purpose |
|------|---------|
| `data/rebin.h5` | Master HDF5 database of all processed data |
| `data/last_processed.txt` | Marker: path of last CNF file processed |
| `data/pipeline.log` | Cron job output log |
| `data/weather_sorted.csv` | Time-sorted weather data (regenerated each run) |
| `data/qa_flagged.csv` | Log of QA-rejected samples |
| `weatherhawk.csv` | Raw weather data (appended by weather_gatherer.py) |

## Calibration

The energy calibration converts channel numbers to keV:

```
energy_keV = cal_offset + cal_slope * (channel + 1)
```

Coefficients are stored in `image_scripts/calibration/calibration_coefficients.txt`. The current calibration has `cal_offset = 1.13` keV and `cal_slope = 0.332` keV/channel, giving a range of 0-2722 keV across 8192 channels.

ROI windows are defined in keV in `roi_energy.dat` and converted to channels at parse time using this calibration. This means ROIs automatically adjust if the calibration changes.
