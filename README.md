# RadWatch Air Monitor

Server-side analysis pipeline for the UC Berkeley rooftop gamma-ray spectroscopy system. Processes raw CNF spectral files and weather station data into time-series plots of isotope count rates, published hourly to the RadWatch website.

A detailed [workflow diagram](https://github.com/alihanks/radwatch-airmonitor/wiki/Plot-creation-work%E2%80%90flow) with specifics for the raw spectrum file processing is provided on the wiki.

## What It Does

- Parses Canberra `.CNF` gamma spectra (via `cnf_parser_standalone`)
- Scrapes weather data from WeatherUnderground
- Rebins spectra into hourly time bins with weather correlation
- Applies QA filtering using K-40 count rate as a stability proxy
- Generates isotope time-series plots, wind roses, and spectrum snapshots
- Deploys output to the web dashboard via SFTP

## Quick Start

```bash
# 1. Clone the repo on the dosenet server
git clone https://github.com/alihanks/radwatch-airmonitor.git
cd radwatch-airmonitor

# 2. Run the setup script (creates conda env, checks dependencies)
bash setup.sh

# 3. Activate the environment
conda activate radwatch

# 4. Test the pipeline
cd image_scripts/analysis
bash cron_job.sh
tail -50 ../../data/pipeline.log
```

## Prerequisites

- **Anaconda3** or **Miniconda** (conda is used to manage the `radwatch` environment)
- **Dropbox** syncing spectral data to `/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/`
- **ImageMagick** (`convert`) for thumbnail generation
- **lftp** for SFTP deployment to the website

## Project Structure

```
radwatch-airmonitor/
  image_scripts/
    analysis/
      raw_analysis.py       # CNF -> HDF5 pipeline (Stage 2)
      h5_analysis.py        # HDF5 -> plots (Stage 3)
      cron_job.sh           # Hourly cron orchestrator
      roi_energy.dat        # ROI windows (keV)
    sample_collection.py    # HDF5 I/O, rebinning, channel standardization
    spectra_utils.py        # CNF parsing, ROI parsing
    spectrum_calibration.py # Energy calibration (lazy scipy for notebooks)
    weather_utils.py        # Weather CSV parsing, wind roses
    weather_gatherer.py     # WeatherUnderground scraper
    cnf_parser_standalone.py # Binary CNF file reader
    calibration/
      calibration_coefficients.txt  # Energy calibration coefficients
  data/                     # Pipeline output (rebin.h5, PNGs, logs)
  docs/
    architecture.md         # Full system architecture and data flow
    runbook.md              # Operations reference (rebuilds, gap fills, diagnostics)
    todo.md                 # Known open items / deferred work
  environment.yml           # Conda environment specification
  requirements.txt          # Pip fallback dependencies
  setup.sh                  # Server setup script
  logbook.md                # Development logbook
```

## Environment

The pipeline runs in an isolated conda environment called `radwatch`. Required packages:

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| h5py | HDF5 read/write |
| matplotlib | Plot generation |
| pandas | Weather data handling |
| Pillow | Image generation (last_update.png) |
| pytz | Timezone handling |

**Optional** (for interactive calibration notebooks only):
- `scipy` -- peak finding and curve fitting

## Documentation

- **[System Architecture](docs/architecture.md)** -- data flow diagrams, HDF5 schema, file reference
- **[Operations Runbook](docs/runbook.md)** -- rebuilds, weather backfill, K-40 diagnostic, deploy, QA review
- **[Open Items](docs/todo.md)** -- known flagged issues that have been deferred
- **[Development Logbook](logbook.md)** -- session-by-session work log

## Cron Schedule

The pipeline runs hourly via crontab:

```
0 * * * *  .../image_scripts/analysis/cron_job.sh
```

Pipeline output is logged to `data/pipeline.log`.
