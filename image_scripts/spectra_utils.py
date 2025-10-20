# spectra_utils.py
# Uses the pip package "xylib-py" (import name: xylib)
# No ctypes, no manual .so loading — just the official SWIG bindings.

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import xylib  # provided by pip install xylib-py


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


# ---- public API ------------------------------------------------------------

def parse_spectra(file_name, sample, format_hint: str = ""):
    """
    Load a spectrum file with xylib and populate `sample`.

    - `file_name` : str | Path
    - `sample`    : object you already use downstream; we attach/overwrite:
        sample.x, sample.y, sample.meta (dict with dataset/block metadata)

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
        ds = xylib.load_file(str(path), format_hint or "", "")
    except Exception as e:
        print(f"Failed to load dataset for {path}: {e}")
        return sample

    xs, ys, meta = _read_first_block_xy(ds)

    # Attach results to your `sample` object.
    try:
        setattr(sample, "x", xs)
        setattr(sample, "y", ys)
        # merge into existing meta if present
        if hasattr(sample, "meta") and isinstance(getattr(sample, "meta"), dict):
            sample.meta.update(meta)
        else:
            setattr(sample, "meta", meta)
    except Exception as e:
        print(f"Warning: could not attach data to sample: {e}")

    return sample


def load_xy(file_name: str | Path, format_hint: str = "") -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Convenience: read a file and return (x, y, metadata) without mutating anything.
    Uses the same xylib API as parse_spectra().
    """
    path = Path(str(file_name))
    ds = xylib.load_file(str(path), format_hint or "", "")
    return _read_first_block_xy(ds)


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
        tmp_str = [item for item in content[x+k].split(' ') if item]
        tmp = [float(tmp_str[0]), float(tmp_str[1]), float(tmp_str[2]), float(tmp_str[3])]
        eff.append(tmp)
    return eff