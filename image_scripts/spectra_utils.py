# spectra_utils.py
# Spectrum file parsing utilities for the RadWatch air-monitor pipeline.
# Uses cnf_parser_standalone for CNF files and external calibration.

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import datetime


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
        - Much simpler than previous xylib/becquerel implementations

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