# spectra_utils.py
# Uses the pip package "xylib-py" (import name: xylib)
# No ctypes, no manual .so loading — just the official SWIG bindings.

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import datetime

#import xylib  # provided by pip install xylib-py

import becquerel as bq


# ---- helpers ---------------------------------------------------------------

def _meta_to_dict(meta) -> Dict[str, str]:
    """
    Convert xylib.MetaData to a plain dict using only public methods:
      - size(), get_key(i), get(key)  (and has_key(key) for safety)
    """
    out: Dict[str, str] = {}
    n = meta.size()                  # number of metadata entries
    for i in range(n):
        k = meta.get_key(i)
        if meta.has_key(k):          # guard against stale keys
            out[k] = meta.get(k)
    return out


def _read_first_block_xy(ds) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Extract X and Y arrays from the first block.
    Columns in xylib are 1-based (0 is a pseudo index column).  :contentReference[oaicite:1]{index=1}
    """
    block = ds.get_block(0)
    nrows = block.get_point_count()
    ncols = block.get_column_count()

    # Column 1: X (e.g., energy or 2theta)
    # Column 2: Y (counts). If only one real column, treat it as Y and
    # synthesize X as 0..N-1.
    if ncols >= 1:
        x_col = block.get_column(1)
        xs = [x_col.get_value(i) for i in range(nrows)]
    else:
        xs = list(range(nrows))

    if ncols >= 2:
        y_col = block.get_column(2)
        ys = [y_col.get_value(i) for i in range(nrows)]
    else:
        ys = []

    meta = {
        "dataset": _meta_to_dict(ds.meta),
        "block": _meta_to_dict(block.meta),
    }
    return xs, ys, meta


def parse_spectra(file_name, sample, format_hint: str = ""):
    """
    Load a spectrum file with standalone parser and populate `sample`.

    Args:
        file_name: str | Path - path to spectrum file (typically .CNF)
        sample: Sample object to populate with spectrum data
        format_hint: str - not used with standalone parser

    Returns:
        sample: Sample object with populated spectrum and metadata

    Notes:
        - using standalone parser for now for new CNF format 
            - will go back to becquerel after patch is implemented
        - Tries to extract rich metadata including timestamps, calibration, live/real time
            - currently using external calibration because stored values are not being found or are incorrect
        - Much simpler than xylib implementation

    Old version of method:
        sample.x, sample.y, sample.meta (dict with dataset/block metadata)
        - we still do this for backward compatibility, but also set spectrum values explicitly
    """
    from cnf_parser_standalone import read_cnf
    from spectrum_calibration import read_calibration_file, apply_calibration

    path = Path(str(file_name))
    if not path.is_file():
        print(f"File does not exist: {path}")
        return sample

    try:
        # Load spectrum with standalone parser
        data = read_cnf(str(path), verbose=False)

        # Load external calibration
        cal_path = Path('/home/dosenet/radwatch-airmonitor/image_scripts/calibration/calibration_coefficients.txt')
        if cal_path.exists():
            cal = read_calibration_file(str(cal_path))
        else:
            # Fallback to calibration from file (may be placeholder)
            cal = data['calibration_coefficients']
            print(f"Warning: Using calibration from file (may be placeholder): {cal}")

        # Extract spectrum data
        counts = data['counts']  # numpy array
        n_channels = len(counts)

        # Apply calibration to get energy values
        energy = apply_calibration(counts, cal)

        # Extract timestamp (from file or filename)
        if data['start_time'] is not None:
            timestamp = data['start_time']
        else:
            # Fallback: use file modification time
            timestamp = datetime.datetime.fromtimestamp(path.stat().st_mtime)

        # Extract times (in seconds)
        real_time_sec = data['realtime'] if data['realtime'] is not None else 0.0
        live_time_sec = data['livetime'] if data['livetime'] is not None else 0.0

        # Calibration as [offset, slope] for set_spectra
        bin_cal = np.array([cal[0], cal[1]])  # [c0, c1] - linear terms only

        # Bin limits (first and last channel indices)
        bin_lim = np.array([0, n_channels - 1])

        # Populate the sample object using set_spectra method
        sample.set_spectra(
            timestamps=timestamp,
            real_times=real_time_sec,
            live_times=live_time_sec,
            bin_lims=bin_lim,
            bin_cals=bin_cal,
            energys=energy,
            counts_=counts
        )

        # Create metadata dict for compatibility
        sample.meta = {
            "filename": str(path),
            "channels": n_channels,
            "livetime": live_time_sec,
            "realtime": real_time_sec,
            "start_time": str(timestamp),
            "calibration": f"{cal[0]:.6f} + {cal[1]:.9f} * channel",
            "format": data.get('format', 'unknown'),
            "total_counts": int(np.sum(counts))
        }

        # Backward compatibility: attach x, y directly
        sample.x = energy
        sample.y = counts

    except Exception as e:
        print(f"Failed to load spectrum for {path}: {e}")
        import traceback
        traceback.print_exc()
        return sample

    return sample


'''
# ---- public API ------------------------------------------------------------

def parse_spectra(file_name, sample, format_hint: str = ""):
    """
    Load a spectrum file with becquerel and populate `sample`.

    Args:
        file_name: str | Path - path to spectrum file (typically .CNF)
        sample: Sample object to populate with spectrum data
        format_hint: str - not used with becquerel (auto-detects CNF)

    Returns:
        sample: Sample object with populated spectrum and metadata

    Notes:
        - becquerel automatically detects and parses Canberra CNF files
        - Extracts rich metadata including timestamps, calibration, live/real time
        - Much simpler than xylib implementation

    Old version of method:
        sample.x, sample.y, sample.meta (dict with dataset/block metadata)
        - we still do this for backward compatibility, but also set spectrum values explicitly

        Notes:
          * Uses xylib.load_file(path, format_name="", options="") which auto-guesses
            the format when `format_name` is "" (empty).  :contentReference[oaicite:2]{index=2}
          * If you know the exact format (e.g. "canberra_cnf"), pass it via
            `format_hint` for more robust parsing. (Canberra CNF is explicitly
            supported by xylib.) :contentReference[oaicite:3]{index=3}
    """
    path = Path(str(file_name))
    if not path.is_file():
        print(f"File does not exist: {path}")
        return sample

    try:
        #ds = xylib.load_file(str(path), format_hint or "", "")
        # Load spectrum with becquerel (one line!)
        spec = bq.Spectrum.from_file(str(path))

        # Extract timestamp
        # becquerel provides start_time as datetime object
        if spec.start_time is not None:
            timestamp = spec.start_time
        else:
            # Fallback: use file modification time
            timestamp = datetime.datetime.fromtimestamp(path.stat().st_mtime)

        # Extract times (becquerel provides these in seconds)
        real_time_sec = spec.realtime if spec.realtime is not None else 0.0
        live_time_sec = spec.livetime if spec.livetime is not None else 0.0

        # Extract energy calibration
        # becquerel stores calibration as a Calibration object
        # For linear calibration: E = offset + slope * channel
        if spec.energy_cal is not None:
            # Get calibration coefficients
            cal_coeffs = spec.energy_cal.coeffs
            if len(cal_coeffs) >= 2:
                bin_cal = np.array([cal_coeffs[0], cal_coeffs[1]])
            else:
                # No calibration - use default
                bin_cal = np.array([0.0, 1.0])
        else:
            # No calibration available
            bin_cal = np.array([0.0, 1.0])

        # Extract spectrum data
        counts = spec.counts  # Already a numpy array
        n_channels = len(counts)

        # Calculate energy values per channel
        if spec.energies is not None:
            energy = spec.energies
        else:
            # Calculate from calibration: E = offset + slope * channel
            energy = bin_cal[0] + bin_cal[1] * np.arange(n_channels)

        # Bin limits (first and last channel indices)
        bin_lim = np.array([0, n_channels - 1])

        # Populate the sample object using set_spectra method
        sample.set_spectra(
            timestamps=timestamp,
            real_times=real_time_sec,
            live_times=live_time_sec,
            bin_lims=bin_lim,
            bin_cals=bin_cal,
            energys=energy,
            counts_=counts
        )

        # Create metadata dict for compatibility (optional, if downstream code uses it)
        sample.meta = {
            "filename": str(path),
            "channels": n_channels,
            "livetime": live_time_sec,
            "realtime": real_time_sec,
            "start_time": str(timestamp),
            "calibration": f"{bin_cal[0]} + {bin_cal[1]} * channel"
        }

    except Exception as e:
        print(f"Failed to load spectrum for {path}: {e}")
        import traceback
        traceback.print_exc()
        return sample

    #xs, ys, meta = _read_first_block_xy(ds)

    # Attach results to your `sample` object.
    try:
        setattr(sample, "x", energy)
        setattr(sample, "y", counts)
        # merge into existing meta if present
        if hasattr(sample, "meta") and isinstance(getattr(sample, "meta"), dict):
            sample.meta.update(meta)
        else:
            setattr(sample, "meta", meta)
    except Exception as e:
        print(f"Warning: could not attach data to sample: {e}")

    return sample
'''


def load_xy(file_name: str | Path, format_hint: str = "") -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Convenience: read a file and return (x, y, metadata) without mutating anything.

    Args:
        file_name: path to spectrum file
        format_hint: unused (kept for API compatibility)

    Returns:
        x: energy array (keV)
        y: counts array
        metadata: dict with spectrum metadata

    Notes:
        - using standalone parser for now for new CNF format
        - uses external calibration file if available
    """
    from cnf_parser_standalone import read_cnf
    from spectrum_calibration import read_calibration_file, apply_calibration

    path = Path(str(file_name))

    # Load spectrum with standalone parser
    data = read_cnf(str(path), verbose=False)

    # Load external calibration
    cal_path = Path('/home/dosenet/radwatch-airmonitor/calibration_coefficients.txt')
    if cal_path.exists():
        cal = read_calibration_file(str(cal_path))
    else:
        # Fallback to calibration from file (may be placeholder)
        cal = data['calibration_coefficients']

    # Y values (counts)
    y = data['counts']

    # X values (energies) - apply calibration
    x = apply_calibration(y, cal)

    # Metadata
    metadata = {
        "filename": str(path),
        "channels": len(y),
        "livetime": data['livetime'],
        "realtime": data['realtime'],
        "start_time": data['start_time'],
        "energy_calibration": cal,
        "format": data.get('format', 'unknown'),
        "total_counts": int(np.sum(y))
    }

    return x, y, metadata


'''
def load_xy(file_name: str | Path, format_hint: str = "") -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Convenience: read a file and return (x, y, metadata) without mutating anything.

    Args:
        file_name: path to spectrum file
        format_hint: unused (kept for API compatibility)

    Returns:
        x: energy array (keV)
        y: counts array
        metadata: dict with spectrum metadata

    Old version of method: Uses the same xylib API as parse_spectra().
    """
    path = Path(str(file_name))
    #ds = xylib.load_file(str(path), format_hint or "", "")

    path = Path(str(file_name))
    spec = bq.Spectrum.from_file(str(path))

    # X values (energies)
    if spec.energies is not None:
        x = spec.energies
    else:
        x = np.arange(len(spec.counts))

    # Y values (counts)
    y = spec.counts

    # Metadata
    metadata = {
        "filename": str(path),
        "channels": len(spec.counts),
        "livetime": spec.livetime,
        "realtime": spec.realtime,
        "start_time": spec.start_time,
        "energy_calibration": spec.energy_cal
    }

    return x, y, metadata

    #return _read_first_block_xy(ds)
'''

# ---- kept from your original module ---------------------------------------


def parse_eff(file_name):
    """
    Parse a plain-text efficiency file in the form used by your pipeline.
    (This block is retained from your original script.)
    """
    with open(file_name) as f:
        content = f.readlines()

    k = 0
    while not ("kev_eff_%err_effw" in content[k].replace('\n', '')):
        k += 1
    k += 1
    num_points = int(content[k])
    k += 1
    eff = []
    for x in range(0, num_points):
        # Use list comprehension to filter out empty items after split
        tmp_str = [item for item in content[x + k].split(' ') if item]
        tmp = [float(tmp_str[0]), float(tmp_str[1]), float(tmp_str[2]), float(tmp_str[3])]
        eff.append(tmp)
    return eff

def parse_roi_energy(file_name, calibration):
    """Parse energy-based ROI file ($ROI_ENERGY: format). Converts keV windows
    to channels using the given calibration. Returns list of ROI objects."""
    import sample_collection
    from spectrum_calibration import energy_to_channel

    roi_col = []
    with open(file_name) as f:
        lines = f.readlines()

    k = 0
    # Skip to header
    while k < len(lines) and '$ROI_ENERGY' not in lines[k]:
        k += 1
    k += 1

    while k < len(lines):
        line = lines[k].strip()
        if not line or line.startswith('#'):
            k += 1
            continue
        parts = line.split()
        n_bkg = int(parts[0])
        peak_lo_keV = float(parts[1])
        peak_hi_keV = float(parts[2])
        isotope = parts[3]
        energy = float(parts[4].replace('keV', ''))
        origin = parts[5].replace('_', ' ') if len(parts) > 5 else ''

        # Convert peak window from keV to channels
        peak_lo_ch = int(round(energy_to_channel(peak_lo_keV, calibration)))
        peak_hi_ch = int(round(energy_to_channel(peak_hi_keV, calibration)))

        # Convert background windows from keV to channels
        bkg_channels = []
        for _ in range(n_bkg):
            k += 1
            bkg_parts = lines[k].strip().split()
            bkg_lo_ch = int(round(energy_to_channel(float(bkg_parts[1]), calibration)))
            bkg_hi_ch = int(round(energy_to_channel(float(bkg_parts[2]), calibration)))
            bkg_channels.append([bkg_lo_ch, bkg_hi_ch])

        r = sample_collection.ROI([peak_lo_ch, peak_hi_ch], bkg_channels)
        r.isotope = isotope
        r.energy = energy
        r.origin = origin
        roi_col.append(r)
        k += 1

    return roi_col


def parse_roi(file_name):
    # This function doesn't use xylib, so it remains unchanged.
    # Import only when needed to avoid circular dependency
    import sample_collection

    roi_col = []
    with open(file_name) as roi_file:
        content = roi_file.readlines()
    
    k = 0
    while k < len(content) - 1:
        bkg = []
        if k == 0:
            while not ('$ROI' in content[k]):
                k = k + 1
        k = k + 1
        sp_line = content[k].split(" ")
        iso = sp_line[3]
        energy = sp_line[4]
        origin = sp_line[5]
        spec_roi = [int(sp_line[1]), int(sp_line[2])]
        for x in range(0, int(sp_line[0])):
            k = k + 1
            sp_line = content[k].split(" ")
            bkg.append([int(sp_line[1]), int(sp_line[2])])
        r = sample_collection.ROI(spec_roi, bkg)
        r.origin = origin.replace('\n', '').replace("_", " ")
        r.isotope = iso.replace('\n', '')
        r.energy = float(energy.split('k')[0])
        roi_col.append(r)
    
    return roi_col