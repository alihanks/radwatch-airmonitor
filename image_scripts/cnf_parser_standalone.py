"""Updated CNF parser that handles both legacy (with PHA keyword) and newer formats.

This is based on becquerel's cnf.py but modified to handle the newer CNF format
that doesn't include the "PHA" ASCII keyword marker.
"""

import datetime
import struct
import re
from pathlib import Path

import numpy as np


class BecquerelParserError(Exception):
    """Custom exception for parser errors."""
    pass


def _extract_timestamp_from_filename(filename):
    """
    Extract acquisition start time from CNF filename.

    Parses filenames in format: YYYY-MM-DD_HH-MM-SS-microseconds.CNF

    Parameters
    ----------
    filename : str or Path
        CNF filename or full path

    Returns
    -------
    timestamp : datetime.datetime or None
        Parsed timestamp with UTC timezone, or None if parsing fails
    """
    filename = Path(filename).name
    pattern = r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{6})'

    match = re.search(pattern, filename)
    if match:
        try:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
            microsecond = int(match.group(7))

            return datetime.datetime(year, month, day, hour, minute, second,
                                     microsecond, tzinfo=datetime.timezone.utc)
        except ValueError:
            return None
    return None


def _from_little_endian(data, index, n_bytes):
    """Convert bytes starting from index from little endian to an integer."""
    return sum(data[index + j] << 8 * j for j in range(n_bytes))


def _convert_date(data, index):
    """Convert 64-bit number starting at index into a date."""
    d = _from_little_endian(data, index, 8)
    if d == 0:  # Handle missing date data
        return None
    t = (d / 10000000.0) - 3506716800
    try:
        return datetime.datetime.fromtimestamp(t, datetime.UTC)
    except (OSError, ValueError, OverflowError):
        # Handle invalid timestamps (e.g., before Unix epoch on Windows)
        return None


def _convert_time(data, index):
    """Convert 64-bit number starting at index into a time."""
    d = _from_little_endian(data, index, 8)
    if d == 0:  # Handle missing time data
        return None
    d = (pow(2, 64) - 1) & ~d
    result = d * 1.0e-7
    # Check for unreasonable values
    if result <= 0 or result > 1e12:  # More than ~30,000 years is unreasonable
        return None
    return result


def _from_pdp11(data, index):
    """Convert 32-bit floating point in DEC PDP-11 format to a double."""
    if (data[index + 1] & 0x80) == 0:
        sign = 1
    else:
        sign = -1
    exb = ((data[index + 1] & 0x7F) << 1) + ((data[index] & 0x80) >> 7)
    if exb == 0:
        if sign == -1:
            return np.nan
        else:
            return 0.0
    h = (
        data[index + 2] / 256.0 / 256.0 / 256.0
        + data[index + 3] / 256.0 / 256.0
        + (128 + (data[index] & 0x7F)) / 256.0
    )
    return sign * h * pow(2.0, exb - 128.0)


def _read_energy_calibration(data, index):
    """Read the four energy calibration coefficients."""
    coeff = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        coeff[i] = _from_pdp11(data, index + 2 * 4 + 28 + 4 * i)
    if coeff[1] == 0.0:
        return None
    return coeff


def read_cnf(filename, verbose=False):
    """Parse a CNF file and return spectrum data.

    This version handles both legacy format (with PHA keyword) and newer format
    (without PHA keyword).

    Parameters
    ----------
    filename : str | pathlib.Path
        The filename of the CNF file to read.
    verbose : bool (optional)
        Whether to print out debugging information. By default False.

    Returns
    -------
    data : dict
        Dictionary of spectrum data including:
        - counts: numpy array of channel counts
        - livetime: live time in seconds
        - realtime: real time in seconds
        - start_time: acquisition start time
        - sample_name, sample_id, etc.: sample information
    """
    filename = Path(filename)
    if verbose:
        print(f"Reading CNF file {filename}")

    ext = filename.suffix
    if ext.lower() != ".cnf":
        raise BecquerelParserError("File extension is incorrect: " + ext)

    # read all of the file into memory
    file_bytes = []
    with Path(filename).open("rb") as f:
        byte = f.read(1)
        while byte:
            byte_int = struct.unpack("1B", byte)
            file_bytes.append(byte_int[0])
            byte = f.read(1)
    if verbose:
        print("file size (bytes):", len(file_bytes))

    # initialize a dictionary of spectrum data to populate as we parse
    data = {}

    # scan for offsets starting at byte 112
    offset_acq = 0
    offset_sam = 0
    offset_eff = 0
    offset_enc = 0
    offset_chan = 0
    for i in range(112, 128 * 1024, 48):
        offset = _from_little_endian(file_bytes, i + 10, 4)
        if (
            (file_bytes[i + 1] == 0x20 and file_bytes[i + 2] == 0x01)
            or (file_bytes[i + 1] == 0)
            or (file_bytes[i + 2] == 0)
        ):
            if file_bytes[i] == 0:
                if offset_acq == 0:
                    offset_acq = offset
                else:
                    offset_enc = offset
            elif file_bytes[i] == 1:
                if offset_sam == 0:
                    offset_sam = offset
            elif file_bytes[i] == 2:
                if offset_eff == 0:
                    offset_eff = offset
            elif file_bytes[i] == 5:
                if offset_chan == 0:
                    offset_chan = offset
            else:
                pass
            if (
                offset_acq != 0
                and offset_sam != 0
                and offset_eff != 0
                and offset_chan != 0
            ):
                break
    if offset_enc == 0:
        offset_enc = offset_acq

    if verbose:
        print(f"Offsets - ACQ: {hex(offset_acq)}, SAM: {hex(offset_sam)}, "
              f"EFF: {hex(offset_eff)}, ENC: {hex(offset_enc)}, CHAN: {hex(offset_chan)}")

    # extract sample information
    if (
        (offset_sam + 48 + 80) >= len(file_bytes)
        or file_bytes[offset_sam] != 1
        or file_bytes[offset_sam + 1] != 0x20
    ):
        if verbose:
            print(offset_sam + 48 + 80, len(file_bytes))
            print(file_bytes[offset_sam], file_bytes[offset_sam + 1])
        raise BecquerelParserError("Sample information not found")
    else:
        sample_name = ""
        for j in range(offset_sam + 48, offset_sam + 48 + 64):
            sample_name += chr(file_bytes[j])
        if verbose:
            print("sample name: ", sample_name)
        sample_id = ""
        for j in range(offset_sam + 112, offset_sam + 112 + 64):
            sample_id += chr(file_bytes[j])
        if verbose:
            print("sample id:   ", sample_id)
        sample_type = ""
        for j in range(offset_sam + 176, offset_sam + 176 + 16):
            sample_type += chr(file_bytes[j])
        if verbose:
            print("sample type: ", sample_type)
        sample_unit = ""
        for j in range(offset_sam + 192, offset_sam + 192 + 64):
            sample_unit += chr(file_bytes[j])
        if verbose:
            print("sample unit: ", sample_unit)
        user_name = ""
        for j in range(offset_sam + 0x02D6, offset_sam + 0x02D6 + 32):
            user_name += chr(file_bytes[j])
        if verbose:
            print("user name:   ", user_name)
        sample_desc = ""
        for j in range(offset_sam + 0x036E, offset_sam + 0x036E + 256):
            sample_desc += chr(file_bytes[j])
        if verbose:
            print("sample desc: ", sample_desc)
        data["sample_name"] = sample_name
        data["sample_id"] = sample_id
        data["sample_type"] = sample_type
        data["sample_unit"] = sample_unit
        data["user_name"] = user_name
        data["sample_description"] = sample_desc

    # extract acquisition information
    if (
        (offset_acq + 48 + 128 + 10 + 4) >= len(file_bytes)
        or file_bytes[offset_acq] != 0
        or file_bytes[offset_acq + 1] != 0x20
    ):
        if verbose:
            print(offset_acq + 48 + 128 + 10 + 4, len(file_bytes))
            print(file_bytes[offset_acq], file_bytes[offset_acq + 1])
        raise BecquerelParserError("Acquisition information not found")
    else:
        offset1 = _from_little_endian(file_bytes, offset_acq + 34, 2)
        offset2 = _from_little_endian(file_bytes, offset_acq + 36, 2)
        offset_pha = offset_acq + 48 + 128

        # Check for legacy format with PHA keyword
        has_pha_keyword = (
            chr(file_bytes[offset_pha + 0]) == "P"
            and chr(file_bytes[offset_pha + 1]) == "H"
            and chr(file_bytes[offset_pha + 2]) == "A"
        )

        if has_pha_keyword:
            if verbose:
                print("Legacy format detected (PHA keyword found)")
            # Original parsing for legacy format
            num_channels = 256 * _from_little_endian(file_bytes, offset_pha + 10, 2)
        else:
            if verbose:
                print("New format detected (no PHA keyword)")
            # New format: channel count is at offset_pha + 8 (where it would be
            # at offset_pha + 10 in legacy format, but shifted due to missing "PHA+")
            # The pattern we see is: 04 00 20 00 at offset_pha+8
            # The channel count appears to be at offset_pha + 10 as a 2-byte value
            num_channels = 256 * _from_little_endian(file_bytes, offset_pha + 10, 2)

        if num_channels < 256 or num_channels > 16384:
            raise BecquerelParserError(f"Unexpected number of channels: {num_channels}")
        if verbose:
            print("Number of channels: ", num_channels)

    # extract date and time information
    # MODIFICATION: Handle new format (no PHA) differently
    if not has_pha_keyword:
        # New format: timing stored as IEEE 754 float at offset_acq + 0xe5c
        NEW_FORMAT_REALTIME_OFFSET = 0x0e5c
        realtime_offset = offset_acq + NEW_FORMAT_REALTIME_OFFSET

        if realtime_offset + 4 <= len(file_bytes):
            # Read realtime as IEEE 754 float32
            realtime_bytes = bytes([file_bytes[i] for i in range(realtime_offset, realtime_offset + 4)])
            realtime = struct.unpack('<f', realtime_bytes)[0]

            # Livetime: Not reliably stored in new format
            # The file may not contain separate livetime information
            # As a fallback, use realtime (assuming minimal deadtime)
            livetime = realtime

            # Collection start: try to extract from filename
            collection_start = _extract_timestamp_from_filename(filename)
            if collection_start and verbose:
                print(f"Extracted start time from filename: {collection_start:%Y-%m-%d %H:%M:%S}")
            elif verbose:
                print("Could not extract start time from filename")

            if verbose:
                print(f"New format timing: realtime={realtime:.2f}s")
                print(f"Note: Livetime not found in file, using realtime as approximation")
                print(f"      (Actual livetime may be slightly less if detector had deadtime)")
        else:
            raise BecquerelParserError("New format: realtime offset beyond file size")
    else:
        # Original format
        offset_date = offset_acq + 48 + offset2 + 1
        if offset_date + 24 >= len(file_bytes):
            raise BecquerelParserError("Problem with date offset")
        collection_start = _convert_date(file_bytes, offset_date)
        realtime = _convert_time(file_bytes, offset_date + 8)
        livetime = _convert_time(file_bytes, offset_date + 16)

    # Handle missing/invalid timing data
    if realtime is None or realtime <= 0.0:
        if verbose:
            print("Warning: Invalid realtime, setting to 1.0s")
        realtime = 1.0
    if livetime is None or livetime <= 0.0:
        if verbose:
            print("Warning: Invalid livetime, setting to realtime")
        livetime = realtime
    if collection_start is None:
        collection_start = datetime.datetime.now(datetime.UTC)

    # Log timing information
    if verbose:
        print("realtime: ", realtime)
        print("livetime: ", livetime)
        if collection_start:
            print(f"{collection_start:%Y-%m-%d %H:%M:%S}")

    # Check timing validity
    if livetime > realtime:
        if verbose:
            print(f"Warning: Livetime > realtime ({livetime} > {realtime}), setting livetime = realtime")
        livetime = realtime

    # extract energy calibration information
    offset_cal = offset_enc + 48 + 32 + offset1
    if offset_cal >= len(file_bytes):
        raise BecquerelParserError("Problem with energy calibration offset")
    cal_coeff = _read_energy_calibration(file_bytes, offset_cal)
    if verbose:
        print("calibration coefficients:", cal_coeff)
    if cal_coeff is None:
        if verbose:
            print("Energy calibration - second try")
        cal_coeff = _read_energy_calibration(file_bytes, offset_cal - offset1)
        if verbose:
            print("calibration coefficients:", cal_coeff)
    if cal_coeff is None:
        raise BecquerelParserError("Energy calibration not found")

    # extract channel count data
    if (
        offset_chan + 512 + 4 * num_channels > len(file_bytes)
        or file_bytes[offset_chan] != 5
        or file_bytes[offset_chan + 1] != 0x20
    ):
        raise BecquerelParserError("Channel data not found")
    channels = np.array([], dtype=float)
    counts = np.array([], dtype=float)
    for i in range(2):
        y = _from_little_endian(file_bytes, offset_chan + 512 + 4 * i, 4)
        if y == int(realtime) or y == int(livetime):
            y = 0
        counts = np.append(counts, y)
        channels = np.append(channels, i)
    for i in range(2, num_channels):
        y = _from_little_endian(file_bytes, offset_chan + 512 + 4 * i, 4)
        counts = np.append(counts, y)
        channels = np.append(channels, i)

    # finish populating data dict
    data["realtime"] = realtime
    data["livetime"] = livetime
    data["start_time"] = collection_start
    data["counts"] = counts
    data["calibration_coefficients"] = cal_coeff

    # clean up null characters in any strings
    for key, value in data.items():
        if isinstance(data[key], str):
            data[key] = value.replace("\x00", " ").replace("\x01", " ").strip()

    return data


if __name__ == "__main__":
    # Test on both files
    import sys
    from pathlib import Path

    file1 = Path('/mnt/user-data/uploads/2017-01-17_15-23-07-523000.CNF')
    file2 = Path('/mnt/user-data/uploads/2025-04-15_15-17-23-043954.CNF')

    print("=" * 80)
    print("Testing 2017 file (legacy format):")
    print("=" * 80)
    try:
        data1 = read_cnf(file1, verbose=True)
        print("\n✓ Successfully parsed!")
        print(f"  Channels: {len(data1['counts'])}")
        print(f"  Live time: {data1['livetime']:.2f} s")
        print(f"  Real time: {data1['realtime']:.2f} s")
        print(f"  Total counts: {sum(data1['counts']):.0f}")
        print(f"  Calibration: {data1['calibration_coefficients']}")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Testing 2025 file (new format):")
    print("=" * 80)
    try:
        data2 = read_cnf(file2, verbose=True)
        print("\n✓ Successfully parsed!")
        print(f"  Channels: {len(data2['counts'])}")
        print(f"  Live time: {data2['livetime']:.2f} s")
        print(f"  Real time: {data2['realtime']:.2f} s")
        print(f"  Total counts: {sum(data2['counts']):.0f}")
        print(f"  Calibration: {data2['calibration_coefficients']}")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
