# RadWatch Air Monitor - Development Logbook

This logbook documents the ongoing development and maintenance of the RadWatch air monitor analysis pipeline. It serves as a reference when context is lost between sessions and as a historical record of what was changed and why.

## Project Overview

The RadWatch air monitor is a rooftop gamma-ray spectroscopy system at UC Berkeley. It continuously collects gamma spectra (CNF files), pairs them with weather station data, and produces time-series plots of isotope count rates and weather conditions. The pipeline runs on a cron job on the `dosenet` server.

**Key files:**
- `image_scripts/analysis/raw_analysis.py` - Ingests raw CNF spectral files + weather CSV, applies QA, writes `rebin.h5`
- `image_scripts/analysis/h5_analysis.py` - Reads `rebin.h5`, produces all plots (`iso_*.png`, wind roses, spectra)
- `image_scripts/sample_collection.py` - HDF5 read/write, spectral rebinning, channel standardization
- `image_scripts/weather_utils.py` - Weather CSV parsing, wind rose drawing
- `image_scripts/spectrum_calibration.py` - Energy calibration from coefficients file
- `cron_job.sh` - Orchestrates the pipeline: raw_analysis -> h5_analysis -> file conversion
- `data/` - All output: `rebin.h5`, CSVs, PNGs, `last_processed.txt`

---

## Work Log

### 2026-02-17: Initial Pipeline Fixes (`899e295` .. `9ed68de`)

**Problem:** The analysis pipeline had accumulated several bugs preventing it from running cleanly on the server.

**Changes:**
- **`899e295` - Fix multiple bugs in analysis pipeline**
  - `cron_job.sh`: replaced broken `if/mkdir` with idempotent `mkdir -p`
  - `sample_collection.py`: fixed `read_hdf` to check `data` group before legacy `2014` group
  - `h5_analysis.py`: clamped weather CSV write range to prevent negative indexing when fewer than 360 timestamps exist
  - `h5_analysis.py`: fixed NaN propagation in cumulative rain plot (replace NaN with 0.0 before `cumsum`)

- **`9ed68de` - Add guard for empty HDF5 data group**
  - `h5_analysis.py` and `sample_collection.py` now check that required datasets exist before accessing them, giving clear error messages instead of cryptic HDF5 exceptions

### 2026-02-18: Calibration & Energy-Based ROIs (`f0dc3dc` .. `fa73c03`)

**Problem:** ROI windows were hardcoded as channel numbers, which broke whenever calibration changed. Two conflicting calibration files existed (one correct, one wrong).

**Changes:**
- **`f0dc3dc` - Consolidate calibration files, add ROI verification notebook**
  - Moved correct calibration (`c0=1.13`) to `image_scripts/calibration/` and deleted incorrect root-level file (`c0=-0.026`)
  - Added `roi_analysis_verification.ipynb` for step-by-step ROI checks
  - Fixed `h5_analysis.py` to use NaN instead of zero for invalid data points (gaps show as line breaks, not false dips)

- **`468947c` - Add energy-based ROI overlay plot to verification notebook**
  - Visual comparison cell showing energy-based vs old channel-based ROI windows

- **`fa73c03` - Use energy-based ROI windows for isotope analysis**
  - ROI windows defined in keV in `roi_energy.dat` instead of channel numbers in `roi.dat`
  - Channels computed at parse time via `energy_to_channel()`, automatically adapting to calibration
  - `roi.dat` still exists for legacy/comparison use

### 2026-02-19: Channel Standardization Fix (`119acd1`)

**Problem:** Spectra from newer hardware had 4096 channels while the calibration and ROI system expected 8192. The original "standardization" was downsampling 8192->4096, which put K-40 (channel ~4394) out of range.

**Changes:**
- **`119acd1` - Fix channel standardization to upsample 4096->8192**
  - Reversed the logic: upsample short spectra instead of downsampling long ones
  - Used external calibration file for energy axis in `h5_analysis.py`
  - Fixed y-axis auto-scale to use `nanmax` on count rates only (not errors)
  - **Deploy note:** requires deleting `rebin.h5` and `last_processed.txt` for fresh rebuild

### 2026-02-23: QA Filtering & 7-Isotope Output (`5a88c66`)

**Problem:** Bad spectra (detector glitches, very short collections) caused spurious dips in count rate plots. CSV output only covered 4 of 7 tracked isotopes.

**Changes:**
- **`5a88c66` - Add K-40 QA filter, persistent ROI storage, full 7-isotope CSV**
  - Two-layer QA in `raw_analysis.py`:
    1. Absolute floor (0.05 counts/sec K-40) catches bad data at dataset start
    2. Rolling-median check (50% of 24-sample window) catches sudden drops mid-dataset
  - ROI counts computed during rebinning and stored in HDF5 (`roi_counts`, `roi_labels`), so `h5_analysis.py` loads them directly
  - CSV output expanded from 4 to all 7 isotopes (Pb214, Bi214, Pb212, Tl208, K40, Cs134, Cs137)

### 2026-03-02: Output Directory Consolidation (`cbfc4f7`, `fabf661`)

**Problem:** Output files were written relative to the working directory, causing path bugs depending on where scripts were invoked.

**Changes:**
- **`cbfc4f7` - Use absolute paths for all output files via `data/` directory**
  - All outputs (`rebin.h5`, CSVs, plots, `last_spectrum/`) now write to `/home/dosenet/radwatch-airmonitor/data/`

- **`fabf661` - Fix cron job to read output files from `data/` directory**
  - Updated `cron_job.sh` convert and mv commands to match new output location

### 2026-03-03: K-40 Floor Tuning & Plot Gap Handling (`9a62f58`)

**Problem:** K-40 floor of 0.05 was too permissive (steady-state is ~0.33). Also, time gaps from detector outages showed as misleading straight lines connecting data points across multi-hour gaps.

**Changes:**
- **`9a62f58` - Raise K-40 floor to 0.15, insert NaN breaks for time gaps**
  - K40_MIN_RATE raised from 0.05 to 0.15 counts/sec
  - NaN rows inserted into time series wherever gaps exceed 2 hours, so matplotlib draws gaps
  - CSV output skips NaN break rows; baseline calculation uses `nanargmin`

### 2026-03-10: Stale Data & Windrose Path Fix (`4d66987`)

**Problem:** Test folders (e.g., `temp_test_folder/`) sorted lexicographically after all real date-formatted directories, poisoning `last_processed.txt` and preventing new data from being processed. Wind rose PNGs were written to cwd instead of `data/`.

**Changes:**
- **`4d66987` - Fix file list filtering, windrose output paths, stale data**
  - Glob results filtered to only include date-formatted directories (`YYYY-MM-DD*`)
  - `draw_windrose()` gained `output_dir` parameter; `h5_analysis.py` passes `DATA_DIR`
  - **Deploy note:** delete `data/rebin.h5` and `data/last_processed.txt` to force clean rebuild

### 2026-04-08: Livetime Estimation for Short Spectra (`d2aba7f`)

**Problem:** Newer CNF files store a preset time (300s) that doesn't reflect actual collection duration when the detector is stopped early. This caused artificial dips in count rates for short-collection spectra.

**Changes:**
- **`d2aba7f` - Add K-40 based livetime estimation**
  - Uses K-40 gross counts as a proxy for true collection time: if a spectrum has fewer K-40 counts than expected for 300s, its livetime is scaled down proportionally
  - Scaling is one-directional (only down, never above preset) to avoid amplifying noise
  - K-40 median baseline computed on first run and stored in HDF5 for reuse on incremental runs
  - Fixed bug in `read_hdf` where the HDF5 file was closed before reading ROI data

### 2026-04-09: Decouple Weather from Radiation Timeline (`e182566`)

**Problem:** Weather subplots in isotope time-series plots (`iso_*.png`) showed the same gaps as radiation data, because weather was stored 1:1 with radiation samples in the HDF5. The weather station records continuously, so these gaps were artificial.

**Changes:**
- **`e182566` - Decouple weather plotting from radiation data timeline**
  - Added `build_merged_weather()` function that merges HDF5 weather (aligned to radiation timestamps) with continuous CSV weather data (`weather_sorted.csv`) to fill gaps
  - NaN-break insertion now only affects `timestamps` and `roi_array` (radiation), not weather
  - Weather plotted once on its own merged timeline (moved out of the ROI loop where it was redundantly re-plotted per isotope)
  - Graceful fallback: if `weather_sorted.csv` doesn't exist, uses HDF5-only weather on original radiation timestamps

### 2026-04-15: Add Pipeline Logging & Fix Marker Bug

**Problem:** The cron job had no logging — all Python stdout/stderr was lost. When the pipeline failed or produced stale output, there was no way to diagnose why without being at the server. Additionally, `last_processed.txt` was being updated even when all CNF file processing failed, causing the pipeline to skip those files permanently on subsequent runs.

**Changes:**
- **`cron_job.sh` — Add comprehensive logging**
  - All output (stdout + stderr) redirected to `data/pipeline.log` via `exec >> "$LOGFILE" 2>&1`
  - Log rotation: old log moved to `pipeline.log.old` when over 1MB
  - Each pipeline step (weather_gatherer, raw_analysis, h5_analysis) logs its exit code
  - Diagnostic summary after each run: Dropbox directory count, last_processed marker, rebin.h5 size, PNG count
  - Warnings printed when scripts fail

- **`sample_collection.py` — Fix `last_processed.txt` marker bug**
  - Marker now only updated when `files_processed > 0` (previously updated whenever `fil_list` was non-empty, even if every file errored)
  - Prevents the marker from advancing past files that failed to process

**Diagnostic checklist (when pipeline isn't updating):**
1. SSH in and check `data/pipeline.log` for errors
2. Verify Dropbox is syncing: `ls '/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/' | tail`
3. Check `data/last_processed.txt` — if it points to a file in a removed directory, delete it
4. Check `data/rebin.h5` exists and has recent mtime
5. Check `data/*.png` exist after a manual run

### 2026-04-16: Add Architecture Docs and Setup Script

**Motivation:** Document the system end-to-end for long-term maintainability, and provide a setup script so the pipeline can be stood up on a new server from a fresh git clone.

**New files:**
- **`docs/architecture.md`** - Complete system documentation: data flow diagrams (ASCII), HDF5 schema, cron schedule, calibration details, and a file reference table for every script, config, and data file in the pipeline.
- **`setup.sh`** - Server setup script that: installs Python dependencies, creates the data directory, checks for Dropbox/weather/calibration files, tests all module imports, and prints next steps (crontab install, manual test run).

### 2026-04-16: Dead Code Cleanup & Conda Environment

**Problem:** The codebase had accumulated dead code from testing/development (becquerel and xylib implementations inside `'''` blocks, old debugging prints, legacy collection-building code). Several top-level imports pulled in packages only used by dead code. The setup script used bare `pip install` which risks conflicts with the other project on the server.

**Changes:**

- **`spectra_utils.py`**: Removed `import becquerel`, `#import xylib` comment, two `'''...'''` dead code blocks (old becquerel `parse_spectra` and `load_xy`), and helper functions `_meta_to_dict` / `_read_first_block_xy` that were only used by the removed code paths.

- **`sample_collection.py`**: Removed unused `from itertools import chain` import and old `'''...'''` `write_hdf` block with hardcoded `'2014'` group.

- **`raw_analysis.py`**: Removed three blocks of commented-out legacy code: old `build_collection`/`rebin`/`write_hdf` calls, diagnostic print blocks, and old `write_spe`/`write_last_update_image` calls.

- **`h5_analysis.py`**: Removed five small commented-out blocks: old color palette file loading, old efficiency correction loop, old weather plotting loop, old gridspec line, old windrose call.

- **`spectrum_calibration.py`**: Moved `scipy.signal.find_peaks`, `scipy.optimize.curve_fit`, and `matplotlib.pyplot` behind lazy imports inside the functions that use them. Pipeline functions (`read_calibration_file`, `apply_calibration`, `energy_to_channel`) no longer require scipy or matplotlib at import time.

- **`environment.yml`** (new): Conda environment spec (`radwatch`) with required packages (numpy, h5py, matplotlib, pandas, Pillow, pytz). scipy listed as optional for notebooks.

- **`requirements.txt`** (new): Pip fallback with the same package list.

- **`setup.sh`**: Rewritten to create/update the `radwatch` conda env from `environment.yml` instead of bare `pip install`. Import tests now run inside the env.

- **`cron_job.sh`**: Added `conda activate radwatch` after PATH setup to isolate the pipeline from the system Python.

### 2026-04-17: Environment Polish, Weather Gatherer Overhaul, Deploy Split

**Context:** After standing up the conda env on the server, several rough edges surfaced: import warnings, pandas-version incompatibilities in the weather gatherer, a hardcoded SFTP password in `cron_job.sh`, and no way to deploy without running the full pipeline. This session cleaned all of that up and pinned dependency versions to the known-working server env.

**Conda environment:**
- **`d3de7fc` - Add `lxml`** to `environment.yml` / `requirements.txt`. Silences pandas's optional-XML-parser import warning in `pipeline.log`.
- **`6f4796b` - Pin Python `<3.13`** (server had landed on 3.14 which was too new for stable pandas/lxml). Targets 3.12.x.
- **`27915eb` - Pin all dependency versions** to minor-version ranges (e.g. `>=2.4,<2.5`) based on `conda list` output from the running `radwatch` env on the server. Allows patch updates, blocks major/minor bumps that could reintroduce the issues fixed this session.
- **`ef81836` - Remove `conda_list.txt`** after transcribing its versions into `environment.yml`. The raw dump is no longer needed.

**`image_scripts/weather_gatherer.py`:**
- **`0db29fb` - Remove deprecated `pd.set_option('future.no_silent_downcasting', True)`**. It was a pandas 2.1 transitional flag removed in newer versions; its presence raised on import, which silently prevented wind-direction conversion from running and caused type errors downstream in the pipeline.
- **`6cbe8bd`, `74a3d86` - Fix wind direction NaN handling.** Replaced `.replace()` (which failed on NaN in newer pandas) with `.map()` + `pd.notna()` guard. Unrecognized values now map to NaN instead of silently passing through.
- **`84f95d6` - Automatic backfill on every run.** The gatherer now reads the last date in `weatherhawk.csv` and scrapes all missing days from WeatherUnderground — not just today. Handles server-downtime gaps without manual intervention. Refactored into smaller functions; renamed the local `dict` to avoid shadowing the builtin.
- **`179a87e` - Add `--fill-gaps` flag** for scanning `weatherhawk.csv` for internal date gaps and backfilling them from WU (with a 2-second delay between requests to be polite). Default behavior (no flag) is unchanged: catch up from last entry to today. Appended gap rows are out of chronological order, but `resort_weather_timestamps()` in the pipeline handles that.

**`cron_job.sh` / deploy:**
- **`49d062e` - `cp` instead of `mv`, remove hardcoded SFTP password.** PNGs and the weather CSV are now copied (not moved) out of `data/` so the originals remain and don't trigger missing-file errors on the next run. `convert` errors are now suppressed rather than failing the job. The SFTP password moved from plaintext in the script to the `RADWATCH_SFTP_PASS` env var — to be set via crontab or `~/.bashrc` on the server. **Deploy note:** must export `RADWATCH_SFTP_PASS` in the server env before the next cron run or the upload will fail.
- **`3c28d61` - Split out `deploy.sh`.** SFTP upload moved into its own script for easier manual testing and debugging. Uses `lftp` with a heredoc instead of the `-e` one-liner (which had connection issues). `cron_job.sh` now just calls `deploy.sh`.

### 2026-04-19: Cleanup

- **`f1d14f8` - Remove stray `~` file** at the repo root. Leftover no-op script, likely from a shell redirection to `~` during an earlier `rtpavlovsk21 → dosenet` find-and-replace. Harmless but confusing.

### 2026-04-20: Fix K-40 Baseline Bias in Livetime Correction

**Problem:** Residual non-physical K-40 dips (0.17–0.25 cts/s vs steady-state ~0.33) remained in the month plot even though `estimate_livetimes_from_k40` was running and scaling livetimes down. Tracing a specific dip bin showed the correction *was* applied — the livetime dropped from 3600s to ~1200s — but the K-40 rate still came out below nominal.

**Root cause:** The baseline reference `k40_median_counts` (used as the denominator of the K-40 ratio that drives the correction) was computed as `np.median` over *all* raw spectra, including the short-livetime truncated ones the correction is designed to handle. The distribution is bimodal (full-livetime cluster near ~100 counts, truncated tail trailing toward 0), so the median sits in the valley (~75) instead of on the full-livetime peak. That biases the baseline ~30% low, which makes the correction systematically *under-correct* by the same factor.

**Evidence from `rebin.h5`:**
- Stored `k40_median_counts = 74.86`, implying "nominal" rate `= 74.86/300 ≈ 0.25` cts/s.
- Observed hourly median K-40 rate: **0.342** cts/s.
- Ratio 0.342 / 0.25 = **1.37** ≈ ratio by which the baseline needs to rise (i.e. true baseline ≈ 103 counts, not 75).
- Every dip bin I inspected had `current_rate / 0.33 < 1`, consistent with over-assigned livetime from the too-low baseline.

**Change:** In `raw_analysis.py` (both first-run and incremental-fallback paths), replace `np.median(k40_counts)` with `np.percentile(k40_counts, 75)` for the baseline, and log both values so the first rebuild confirms the bias number. The HDF5 key name (`k40_median_counts`) is kept for backward compatibility — it's still the livetime-correction baseline, just computed differently. No QA threshold changes: the plan is for the corrected livetimes to be close enough to truth that the existing `K40_MIN_RATE = 0.15` / `QA_THRESHOLD = 0.5` gates stop firing on the previously-residual dips.

**New file:** `image_scripts/analysis/diagnose_k40_baseline.py` — standalone diagnostic that reads a recent window of raw CNFs (default last 500), computes K-40 gross counts per spectrum, and prints median / p75 / p90 / a text histogram / interpretation. No HDF5 writes. Purpose: empirically confirm the bimodality and the ~30% gap between median and p75 before paying for a full rebuild.

**Deploy sequence:**
1. On server: `python3 image_scripts/analysis/diagnose_k40_baseline.py` — verify p75/median ratio ≈ 1.3–1.4.
2. Optionally `python3 image_scripts/weather_gatherer.py --fill-gaps` if `weatherhawk.csv` has internal date gaps (default cron run only fills trailing gaps).
3. Delete `data/rebin.h5` and `data/last_processed.txt` to force full rebuild with the new baseline.
4. Let cron run, or invoke `raw_analysis.py` + `h5_analysis.py` manually.
5. Inspect new `iso_One_Month.png` — residual K-40 dips should flatten to the ~0.33 steady-state line.

### 2026-04-20: `weather_gatherer --since` for Gap Filling

**Problem:** `weather_gatherer.py --fill-gaps` used the CSV's first timestamp as the start of the gap-scan range. `weatherhawk.csv` contains at least one stale entry from 2014 (pre-dating the actual weather station install), so `--fill-gaps` was attempting to scrape every missing day back to 2014 from Weather Underground — most of which returns nothing useful since the station didn't exist yet, and with a 2-second per-request delay that means thousands of wasted requests.

**Change (`weather_gatherer.py`):** Add `--since YYYY-MM-DD` to cap how far back the gap scan goes. Passed as `since_date` into `fill_gaps()`; when provided and later than the CSV's first date, scan starts from `since_date` instead. Prints both the full CSV date range and the actual scan range so the effect is visible in logs. If the user passes `--since` without `--fill-gaps`, a note is printed and the flag is ignored.

**Usage:**
```
python3 image_scripts/weather_gatherer.py --fill-gaps --since 2024-01-01
```

### 2026-04-21: Runbook Touch-Up, Add `docs/todo.md`

**Runbook updates (`docs/runbook.md`):**
- §1 (Full rebuild, "What to look for in the log") now documents the new K-40 per-sample log lines (`K40 livetime correction:`, `K40 spectrum drop:`) and the summary `estimate_livetimes_from_k40: corrected N, dropped M, of K spectra`.
- §3 (K-40 baseline diagnostic) now includes the `K40_LIVETIME_THRESHOLD = 0.90` / `K40_LIVETIME_MIN_RATIO = 0.70` band explanation so readers understand the connection between the diagnostic numbers and the pipeline's decision logic.
- §6 split into §6a (raw-spectrum drops, pre-rebin) and §6b (hourly QA, post-rebin). §6a explains these drops only appear in `pipeline.log` and are not in `qa_flagged.csv`. §6b notes that heavy hourly QA activity is a signal to revisit the raw-drop threshold.

**New: `docs/todo.md`** — open-items list for things flagged during recent sessions and deferred: weather parser NaN-vs-0.0 semantics, missing 2017–2024 weather history, unverified backwards-compatibility with pre-2018 CNFs, and a reminder to re-evaluate the K-40 correction thresholds after some accumulated data. Each entry says what/why/how-a-fix-would-look. Linked from `README.md`'s Documentation section and listed in the project-structure block.

### 2026-04-21: Operational Cleanup — SFTP Log Noise, Stale Marker Self-Heal, Ignore Generated Outputs

Three small operational fixes, each committed separately.

**`deploy.sh` — suppress SFTP chmod noise.** WPEngine's SFTP server doesn't grant chmod to the uploading user; lftp's default behavior is to attempt chmod after every transferred file, generating ~17 "Access failed: permission denied" lines per run in `pipeline.log`. Transfers themselves succeed. Added `--no-perms` to the `mirror` invocation so lftp doesn't attempt remote chmods at all.

**`sample_collection.build_collection_incremental` — self-heal stale marker.** When `data/last_processed.txt` pointed at a file no longer in the filtered CNF list (the current case: a file under `temp_test_folder/` that the date-dir regex now excludes), the pipeline was falling through to `start_index = 0` and reprocessing every CNF from the list origin on every hourly cron run. Works correctly thanks to HDF5 dedup on merge, but wastes I/O. Now, on stale-marker detection, the code deletes the marker file and falls into the same "no marker → last 10000 files" path used on initial runs. The next successful pipeline run writes a fresh valid marker and subsequent runs resume incremental behavior.

**`.gitignore` — stop tracking generated pipeline outputs.** Every rebuild was adding ~600k lines to the repo (`data/rebin.h5` 15MB binary, `weather_sorted.csv` 364k rows, `weatherhawk.csv` 107k rows, all the regenerated PNGs). Ignored going forward: `data/`, `image_scripts/analysis/rooftop_tmp/`, `weatherhawk.csv`, `weather_sorted.csv`, and the two `.backup` CSVs. Used `git rm --cached` to untrack the current snapshots without deleting them from disk — the files stay on the server, and the last committed snapshot still lives in history if anyone needs it. Both directories (`data/`, `rooftop_tmp/`) are created at pipeline start via `os.makedirs(exist_ok=True)` and `mkdir -p`, so no placeholder `.gitkeep` needed.

**Server pull note:** `git rm --cached` only removes from the index, not the working tree. When you pull on the server, git sees the files as newly-untracked (matched by .gitignore) and leaves them in place. Your `weatherhawk.csv` (with the 107k rows of scraped history) stays safely on disk. If you want to be paranoid, back it up before pulling (`cp weatherhawk.csv weatherhawk.csv.safety`) — it's not a file we'd want to lose.

### 2026-04-21: Drop Raw Spectra Below K-40 Ratio 0.70

**Problem:** The p75 baseline fix collapsed the wide K-40 dips in the month plot (steady state restored to ~0.33), but narrow residual dips to ~0.15–0.25 remained at the start/end of continuous data runs. These are likely raw spectra that are bad in ways beyond just short livetime (gain shift, partial readout, electronics issue during stop/start) — the linear `livetime = preset × ratio` model assumes K-40 deficit is purely a livetime problem, and that assumption stops being trustworthy much below a ratio of ~0.70.

**Change (`raw_analysis.py`):** Introduce `K40_LIVETIME_MIN_RATIO = 0.70`. `estimate_livetimes_from_k40` now returns a set of indices to drop:
- `ratio >= 0.90`: no correction (unchanged).
- `0.70 <= ratio < 0.90`: scale livetime down proportionally (unchanged behavior, new lower bound).
- `ratio < 0.70`: **drop the raw spectrum** before rebinning. Logged per-sample with `K40 spectrum drop: timestamp | K40 ratio=X.XXX`.

Both call sites (first-run and incremental) filter `col(_new).collection` by the returned drop set before calling `rebin()`. The summary log line now reports `corrected N, dropped M, of K spectra` for visibility.

**Why drop at the raw-spectrum level (not the hourly QA level):** Dropping a raw spectrum only removes its ~300s of contribution from one hourly bin. The surviving raw spectra in that same hour still contribute their good data for all 7 isotopes. Compare to tightening hourly QA (`K40_MIN_RATE` or `QA_THRESHOLD`), which would drop the entire hour including the perfectly-good radon/thoron/cesium data from the other ~11 spectra in that bin. The surgical approach keeps more good data.

**Expected effect on plots:** Residual edge dips to ~0.15–0.25 on K-40 should either flatten (if the constituent raw spectra are in the 0.70–0.90 range and get a stronger correction thanks to the higher p75 baseline) or leave NaN gaps (if too many in that bin are below 0.70 and get dropped). Both are honest outcomes.

**Deploy:** Pull, let cron run. No HDF5 rebuild needed — the change applies going forward on every incremental run. If you want the existing historical data reprocessed under the new rules, delete `data/rebin.h5` and `data/last_processed.txt` to trigger a full rebuild.

### 2026-04-20: Handle Non-Numeric Weather Values (`--` Placeholder)

**Problem:** After the `--fill-gaps` backfill brought in days where Weather Underground's table had `'--'` placeholder entries for missing readings, every subsequent pipeline run failed at `weather_utils.parse_weather_data` with `ValueError: could not convert string to float: '--'`. Both `raw_analysis.py` and `h5_analysis.py` exited with code 1; plots stopped updating (deploy still ran, but on stale PNGs).

**Change (`weather_utils.py`, `parse_weather_data`):** Wrap the float coercion in a try/except that catches `ValueError`/`TypeError` and falls back to the same 0.0 default the existing code used for empty cells. This is the minimal unblock — handles `'--'`, `'N/A'`, `'nan'`, and anything else non-numeric that WU might return, while preserving the prior empty-cell semantics. Also collapsed the previous pattern of calling `float(row[order[k]])` multiple times on the same value into a single `tmp` local.

**Known separate concerns (not fixed here):**
- Treating missing values as 0.0 is historically what this parser does, but it's not physically correct (0.0 solar is very different from "no reading"). Converting missing to NaN would be more right, but the downstream plotting code isn't known to handle NaN in every weather field yet, so leaving that as-is for now.
- `last_processed.txt` in the pushed `data/` snapshot points at a file inside `temp_test_folder/`. That path is excluded by the date-dir regex in `build_collection_incremental`, so the marker is effectively dead — every run "starts from the beginning" of the filtered file list, which isn't the incremental-path behavior the marker is meant to provide. Worth a separate cleanup pass.

### 2026-04-20: Docs — Add Runbook, Retire Pipeline Changelog

**Problem:** Operational details for running tools in isolation (full rebuild, weather gap fill, K-40 diagnostic, manual deploy, stale-plot diagnosis) lived only in logbook prose and oral tradition. There was no runbook-style reference. Separately, `docs/pipeline_changelog.md` and `logbook.md` had substantial overlap — all three changelog entries (2026-02-22, 2026-02-23, 2026-04-16) were also in the logbook, and the changelog stopped being updated after 2026-04-16 while the logbook kept going.

**Changes:**
- **New: `docs/runbook.md`** — six-section operations reference:
  1. Full rebuild of `rebin.h5` (when, steps, log signals, expected runtime)
  2. Weather backfill (trailing catch-up vs `--fill-gaps --since`, re-sort helper)
  3. K-40 baseline diagnostic (when to run, how to interpret median/p75/p90 ratios)
  4. Diagnosing stale plots (top-down walk through cron → Dropbox → `last_processed.txt` → `rebin.h5` → PNGs → deploy)
  5. Manual deploy / SFTP failure (`RADWATCH_SFTP_PASS` env var, `deploy.sh` invocation, target host)
  6. QA flagged samples (`data/qa_flagged.csv` columns, reason strings, interpretation)
- **Runbook includes a historical-data note** under the Full Rebuild section: the current pipeline only processes CNFs under the Dropbox `current/` folder (2025–2026 data). Pre-2018 data was processed with an older version pointing at the `RPAVLOVSKY` folder using different analysis procedures — that data should not be re-run through the current pipeline. The current code is still capable of parsing older CNFs if someone ever wanted to recover those results, but the pre-2018 published numbers aren't reproduced by the `current/`-only rebuild.
- **Deleted: `docs/pipeline_changelog.md`** — redundant with `logbook.md`, stale after 2026-04-16. Going forward the logbook is the single source of truth; deep treatment (Problem / Root Cause / Change / Verify) happens inside logbook entries when warranted (e.g. today's K-40 baseline fix entry).
- **`README.md`:** updated the project-structure block and the Documentation section to point at `runbook.md`, drop the `pipeline_changelog.md` reference, and fix the case on `logbook.md` (was `LOGBOOK.md`, a Linux-case-sensitive broken link).

### 2026-04-20: Fix Isotope Labels on "Most Recent Spectrum" Plot

**Problem:** The `most_recent_spectra.png` plot showed isotope-name labels pointing to the wrong energies — several labels ("Pb212", "Pb214", "Be7", "Tl208", "Cs134", "Bi214", "Cs137") clustered in the 200–700 keV range regardless of where the actual peaks sit. Symptom of calibration drift: labels were anchored to hardcoded channel numbers, not energies.

**Root cause:** `h5_analysis.py` was loading ROIs from *two* files — the current energy-based `roi_energy.dat` (into `col`) and the legacy channel-based `roi.dat` (into `col_comp`). The ROI highlight overlay used `col` (correct), but the `annotate` loop that places the isotope text labels iterated over `col_comp` (wrong — channel numbers are stale when calibration changes). This was a migration leftover from the Feb 18 `fa73c03` switch to energy-based ROIs — the logbook from that date noted "`roi.dat` still exists for legacy/comparison use," and this plot was silently depending on it.

**Change (`h5_analysis.py`):** Drop `col_comp` and the `roi.dat` load. Merge the annotation logic into the existing `col.rois` loop so labels and highlight overlay share the same (calibration-aware) ROI channels. Also drops the redundant yellow `roi_comp` overlay — it was visual noise duplicating the pink `roi_` overlay at wrong positions.

**Scope:** Only affects `most_recent_spectra.png`. No impact on isotope time-series plots (`iso_*.png`), QA, HDF5 schema, or the K-40 livetime correction. `roi.dat` is still present in the repo for the verification notebook; not removed.

---

## Known Issues & Future Work

- **Windrose data** still comes from HDF5 only (wind direction/speed not yet merged with CSV). This is lower priority since wind roses are aggregated over time windows, not plotted as time series.
- **Cumulative rain plot** (`out.png`) now uses merged weather timeline, but the differential rain values from HDF5 vs CSV may use slightly different accumulation logic (CSV rain is computed as deltas from yearly totals in `parse_weather_data`).
- The `weather_sorted.csv` is generated by `raw_analysis.py` calling `resort_weather_timestamps()`. If the weather gatherer hasn't run or the CSV is missing, weather falls back to HDF5-only (with radiation gaps).
- **K-40 dips** occasionally still show up in the count-rate plots despite the two-layer QA (absolute floor + rolling-median). Root cause not yet identified — may require tightening the rolling-median window or adding a third check against the long-term baseline.
- **SFTP password** is now expected in `RADWATCH_SFTP_PASS` env var rather than in the script. Must be set on the server (via crontab line or `~/.bashrc`) for the deploy step to succeed. Not yet committed to any durable config outside the server's shell env.

---

## Deployment Notes

When making changes that affect the HDF5 schema or channel count:
1. SSH to `dosenet` server
2. Delete `data/rebin.h5` and `data/last_processed.txt`
3. Run `raw_analysis.py` to do a full rebuild
4. Run `h5_analysis.py` to regenerate all plots

The cron job (`cron_job.sh`) runs both scripts and converts output PNGs for the web dashboard.

**Logs:** All pipeline output goes to `data/pipeline.log`. Check this first when diagnosing issues.

**Key paths on server:**
- Spectral data (Dropbox): `/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/`
- Weather station CSV: `/home/dosenet/radwatch-airmonitor/weatherhawk.csv`
- Pipeline output: `/home/dosenet/radwatch-airmonitor/data/`
- Cron job: `/home/dosenet/radwatch-airmonitor/image_scripts/analysis/cron_job.sh`
- SFTP staging: `/home/dosenet/radwatch-airmonitor/image_scripts/analysis/rooftop_tmp/`
