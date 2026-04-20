"""Diagnose the K-40 baseline bias in the livetime correction.

Reads a recent window of raw CNF spectra, computes K-40 gross counts per
spectrum using the same ROI + calibration the pipeline uses, and reports
summary statistics (median, 75th percentile, 90th percentile, plus a
text histogram). No HDF5 writes; no state touched.

Usage:
    python3 diagnose_k40_baseline.py             # last 500 CNFs
    python3 diagnose_k40_baseline.py --n 2000    # last 2000 CNFs
    python3 diagnose_k40_baseline.py --n 200     # quick spot check

What to look for:
- If `median` and `p75` are close (within ~5%), the distribution is
  unimodal and the current median-based baseline is fine.
- If `p75` is noticeably higher than `median` (e.g. 25-40% higher), the
  distribution is bimodal (well-collected + truncated spectra) and the
  current median is biased low by the truncated tail — which is the
  hypothesis driving the planned fix.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/')
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/image_scripts')

from image_scripts import sample_collection, spectra_utils
from image_scripts.spectrum_calibration import read_calibration_file

PROJECT_ROOT = '/home/dosenet/radwatch-airmonitor'
SPEC_DIR = '/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/'
CAL_FILE = os.path.join(PROJECT_ROOT, 'image_scripts', 'calibration', 'calibration_coefficients.txt')
ROI_FILE = os.path.join(PROJECT_ROOT, 'image_scripts', 'analysis', 'roi_energy.dat')
K40_ROI_INDEX = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=500, help='Number of most-recent CNFs to sample')
    args = p.parse_args()

    calibration = read_calibration_file(CAL_FILE)
    rois = spectra_utils.parse_roi_energy(ROI_FILE, calibration)
    k40_roi = rois[K40_ROI_INDEX]
    print(f"K-40 ROI: isotope={k40_roi.isotope}, peak channels={k40_roi.peak}")

    date_dir_re = re.compile(r'/\d{4}-\d{2}-\d{2}')
    fil_list = glob.glob(SPEC_DIR + '*/*.CNF')
    fil_list = [f for f in fil_list if date_dir_re.search(f)]
    fil_list.sort()
    fil_list = fil_list[-args.n:]
    print(f"Sampling {len(fil_list)} most recent CNFs (from {fil_list[0]} to {fil_list[-1]})")

    k40_counts = []
    live_times = []
    skipped = 0
    for f in fil_list:
        sample = sample_collection.Sample()
        spectra_utils.parse_spectra(f, sample)
        if sample.counts is None or len(sample.counts) == 0:
            skipped += 1
            continue
        spec = np.asarray(sample.counts)
        # Upsample 4096 -> 8192 if needed (matches pipeline's standardize_channel_counts)
        if len(spec) == 4096:
            spec = np.repeat(spec, 2) / 2.0
        counts, _ = k40_roi.get_counts(spec)
        k40_counts.append(float(counts))
        lt = sample.live_time.total_seconds() if hasattr(sample.live_time, 'total_seconds') else float(sample.live_time)
        live_times.append(lt)

    if not k40_counts:
        print("ERROR: No spectra parsed; check SPEC_DIR and ROI setup.")
        sys.exit(1)

    k40 = np.asarray(k40_counts)
    lt = np.asarray(live_times)
    print(f"\nParsed {len(k40)} spectra, skipped {skipped}")
    print(f"\n--- K-40 gross counts per raw spectrum ---")
    print(f"  min:     {k40.min():.1f}")
    print(f"  median:  {np.median(k40):.2f}   <-- what raw_analysis.py currently uses")
    print(f"  p75:     {np.percentile(k40, 75):.2f}   <-- proposed replacement")
    print(f"  p90:     {np.percentile(k40, 90):.2f}")
    print(f"  max:     {k40.max():.1f}")
    print(f"  p75/median ratio: {np.percentile(k40, 75)/np.median(k40):.3f}  (expect ~1.0 for a clean unimodal distribution)")

    print(f"\n--- Preset live_time stats ---")
    print(f"  median:  {np.median(lt):.1f}s   (nominal 300)")
    print(f"  frac <250s: {np.mean(lt < 250):.1%}")

    print(f"\n--- Histogram of K-40 counts (counts per raw spectrum) ---")
    # Bin from 0 to max with 25 bins
    edges = np.linspace(0, max(k40.max(), 1), 26)
    hist, _ = np.histogram(k40, bins=edges)
    peak = hist.max() if hist.max() > 0 else 1
    for i, c in enumerate(hist):
        bar = '#' * int(40 * c / peak)
        print(f"  [{edges[i]:6.1f}, {edges[i+1]:6.1f}) | {c:4d} | {bar}")

    print(f"\n--- Interpretation ---")
    ratio = np.percentile(k40, 75) / np.median(k40)
    if ratio > 1.15:
        print(f"p75 is {ratio:.2f}x the median -> median is BIASED LOW by the truncated tail.")
        print(f"Switching the baseline to p75 will raise it by ~{(ratio-1)*100:.0f}% and deepen")
        print(f"the livetime correction applied to short-livetime spectra.")
    else:
        print(f"p75/median = {ratio:.2f} -> distribution looks roughly unimodal;")
        print(f"baseline bias is smaller than hypothesized.")


if __name__ == '__main__':
    main()
