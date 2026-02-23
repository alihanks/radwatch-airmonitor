import sys
import os
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/')
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/image_scripts')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from image_scripts import weather_utils
from image_scripts import sample_collection
from image_scripts.spectrum_calibration import read_calibration_file
import datetime
import numpy as np

spec_dir = r'/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/'
weat_csv = r'/home/dosenet/radwatch-airmonitor/weatherhawk.csv'
weat_csv_sorted = r'weather_sorted.csv'
roi_dat = r'/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi.dat'

weather_utils.resort_weather_timestamps(weat_csv, weat_csv_sorted)
print("resorted the weather data")
#col = sample_collection.SampleCollection()
#print("Collection made")
#col.build_collection(spec_dir, weat_csv_sorted)

# Add this line:
#col.standardize_channel_counts()

#col.rebin(datetime.timedelta(hours=1.0))
#print('rebinned')
#col.write_hdf('rebin.h5')
#print('db written')
#col.rebin(datetime.timedelta(hours=8.))
#col.write_hdf('short.h5')

## NEW: Added configuration constants at the top
# Lines 11-14 in raw_analysis_incremental.py
hdf5_file = 'rebin.h5'
last_processed_marker = 'last_processed.txt'
qa_flagged_csv = 'qa_flagged.csv'

# Calibration and ROI setup (same files h5_analysis.py uses)
calibration = read_calibration_file('/home/dosenet/radwatch-airmonitor/image_scripts/calibration/calibration_coefficients.txt')
roi_energy_file = '/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi_energy.dat'

K40_ROI_INDEX = 4  # K-40 is the 5th ROI (0-indexed) in roi_energy.dat
QA_WINDOW = 24     # Rolling median window (24 hourly bins = 1 day)
QA_THRESHOLD = 0.5    # Flag if K-40 rate < 50% of rolling median
K40_MIN_RATE = 0.05   # Absolute minimum K-40 rate (counts/sec); catches bad data even at start of dataset


def load_rois_for_qa():
    """Load ROI definitions using the energy-based ROI file and calibration."""
    import spectra_utils
    return spectra_utils.parse_roi_energy(roi_energy_file, calibration)


def compute_roi_counts_for_collection(collection, rois):
    """Compute ROI counts for each sample in the collection.

    Returns
    -------
    roi_array : ndarray of shape (N, num_rois, 2)
        [:,:,0] = net count rate (counts/sec), [:,:,1] = error.
    roi_labels : list of str
        Isotope label for each ROI.
    """
    n_samples = len(collection)
    n_rois = len(rois)
    roi_array = np.zeros((n_samples, n_rois, 2))
    for i, sample in enumerate(collection):
        live_sec = sample.live_time.total_seconds() if hasattr(sample.live_time, 'total_seconds') else float(sample.live_time)
        if live_sec <= 0:
            roi_array[i, :, :] = np.nan
            continue
        spec = np.asarray(sample.counts)
        for j, roi in enumerate(rois):
            counts, error = roi.get_counts(spec)
            roi_array[i, j, 0] = counts / live_sec
            roi_array[i, j, 1] = error / live_sec
    roi_labels = [roi.isotope for roi in rois]
    return roi_array, roi_labels


def qa_filter_k40(collection, roi_array, baseline_k40_rates=None):
    """Apply rolling-median QA filter on K-40 count rate.

    Parameters
    ----------
    collection : list of Sample
        Rebinned samples.
    roi_array : ndarray (N, num_rois, 2)
        ROI count-rate array.
    baseline_k40_rates : list or None
        Previous K-40 rates to seed the rolling median (incremental path).

    Returns
    -------
    keep_mask : ndarray of bool (N,)
    flagged_records : list of dict
        Records for qa_flagged.csv.
    """
    k40_rates = roi_array[:, K40_ROI_INDEX, 0].copy()
    n = len(k40_rates)
    keep_mask = np.ones(n, dtype=bool)
    flagged_records = []

    # Prepend baseline rates if available (incremental path)
    if baseline_k40_rates is not None and len(baseline_k40_rates) > 0:
        all_rates = np.concatenate([np.asarray(baseline_k40_rates), k40_rates])
        offset = len(baseline_k40_rates)
    else:
        all_rates = k40_rates
        offset = 0

    total = len(all_rates)

    for i in range(n):
        if np.isnan(k40_rates[i]):
            continue

        # Absolute floor — catches bad data regardless of rolling median state
        if k40_rates[i] < K40_MIN_RATE:
            keep_mask[i] = False
            flagged_records.append({
                'timestamp': collection[i].timestamp,
                'k40_rate': k40_rates[i],
                'rolling_median': np.nan,
                'reason': f'K40 rate {k40_rates[i]:.4f} < absolute minimum {K40_MIN_RATE}'
            })
            continue

        idx = i + offset  # index into all_rates
        # Window: expanding at edges, capped at QA_WINDOW
        win_start = max(0, idx - QA_WINDOW + 1)
        window = all_rates[win_start:idx + 1]
        median_val = np.nanmedian(window)

        if np.isnan(median_val) or median_val <= 0:
            continue

        if k40_rates[i] < QA_THRESHOLD * median_val:
            keep_mask[i] = False
            flagged_records.append({
                'timestamp': collection[i].timestamp,
                'k40_rate': k40_rates[i],
                'rolling_median': median_val,
                'reason': f'K40 rate {k40_rates[i]:.4f} < {QA_THRESHOLD*100:.0f}% of median {median_val:.4f}'
            })

    return keep_mask, flagged_records


def append_flagged_csv(flagged_records, csv_path):
    """Append flagged QA records to CSV."""
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('timestamp,k40_rate,rolling_median,reason\n')
        for rec in flagged_records:
            f.write(f"{rec['timestamp']},{rec['k40_rate']:.6f},{rec['rolling_median']:.6f},{rec['reason']}\n")

# ----------------------------------------
# Initialize collection
col = sample_collection.SampleCollection()
print("Collection created")

# Try to load existing HDF5 data
existing_data_loaded = col.read_hdf(hdf5_file)

if existing_data_loaded:
    # INCREMENTAL PATH: Load existing data, process only new files
    print(f"Loaded existing data: {len(col.collection)} rebinned samples")
    print(f"Date range: {col.collection[0].timestamp} to {col.collection[-1].timestamp}")
    
    # Create new collection for incremental data
    col_new = sample_collection.SampleCollection()
    print("\nBuilding collection from new files...")
    col_new.build_collection_incremental(spec_dir, weat_csv_sorted, last_processed_marker)
    
    if len(col_new.collection) > 0:
        print(f"\nNew collection built: {len(col_new.collection)} raw samples")

        # Standardize channel counts for new data
        col_new.standardize_channel_counts()

        # Rebin new data
        print("\nRebinning new data...")
        col_new.rebin(datetime.timedelta(hours=1.0))
        print(f"New data rebinned to {len(col_new.collection)} samples")

        # QA filter on new rebinned data using K-40 rolling median
        print("\nRunning QA filter on new data...")
        rois = load_rois_for_qa()
        new_roi_array, roi_labels = compute_roi_counts_for_collection(col_new.collection, rois)

        # Seed rolling median baseline from last QA_WINDOW existing samples
        baseline_k40 = []
        if col.roi_counts is not None and col.roi_counts.shape[0] > 0:
            baseline_k40 = col.roi_counts[-QA_WINDOW:, K40_ROI_INDEX, 0].tolist()
        elif len(col.collection) > 0:
            # Compute K-40 rates from existing collection tail
            tail_start = max(0, len(col.collection) - QA_WINDOW)
            tail_roi, _ = compute_roi_counts_for_collection(col.collection[tail_start:], rois)
            baseline_k40 = tail_roi[:, K40_ROI_INDEX, 0].tolist()

        keep_mask, flagged = qa_filter_k40(col_new.collection, new_roi_array, baseline_k40)
        if len(flagged) > 0:
            print(f"QA flagged {len(flagged)} anomalous samples")
            append_flagged_csv(flagged, qa_flagged_csv)
            col_new.collection = [s for s, keep in zip(col_new.collection, keep_mask) if keep]
            new_roi_array = new_roi_array[keep_mask]
            print(f"After QA: {len(col_new.collection)} samples retained")
        else:
            print("QA filter: all samples passed")

        # Merge new rebinned data into existing collection
        col.merge_collection(col_new)

        # Recompute full ROI array for merged collection
        print("\nComputing ROI counts for full collection...")
        full_roi_array, roi_labels = compute_roi_counts_for_collection(col.collection, rois)

        # Write updated HDF5 with ROI data
        print("\nWriting updated database...")
        col.write_hdf(hdf5_file, roi_data=full_roi_array, roi_labels=roi_labels)
        print('Database updated')
    else:
        print("\nNo new files to process")
else:
    # FIRST RUN PATH: No existing data, process from scratch
    print("No existing data found, processing all files...")
    
    col.build_collection_incremental(spec_dir, weat_csv_sorted, last_processed_marker)
    print("Collection built")
    print(f"Size: {len(col.collection)}")
    
    if len(col.collection) > 0:
        # Standardize channel counts
        col.standardize_channel_counts()

        # Rebin
        print("\nRebinning data...")
        col.rebin(datetime.timedelta(hours=1.0))
        print('Rebinned')

        # QA filter using K-40 rolling median
        print("\nRunning QA filter...")
        rois = load_rois_for_qa()
        roi_array, roi_labels = compute_roi_counts_for_collection(col.collection, rois)

        keep_mask, flagged = qa_filter_k40(col.collection, roi_array)
        if len(flagged) > 0:
            print(f"QA flagged {len(flagged)} anomalous samples")
            append_flagged_csv(flagged, qa_flagged_csv)
            col.collection = [s for s, keep in zip(col.collection, keep_mask) if keep]
            roi_array = roi_array[keep_mask]
            print(f"After QA: {len(col.collection)} samples retained")
        else:
            print("QA filter: all samples passed")

        # Write HDF5 with ROI data
        print("\nWriting database...")
        col.write_hdf(hdf5_file, roi_data=roi_array, roi_labels=roi_labels)
        print('Database written')
    else:
        print("No data to process")
        sys.exit(0)

print("Collection built")
print("Size ", len(col.collection))
#col.rebin(datetime.timedelta(minutes=20));
#col.write_hdf('mod_rebin.h5');

# Add this before the rebin call to diagnose the issue
#print(f"Collection size: {len(col.collection)}")
#for i, sample in enumerate(col.collection[:5]):  # Check first 5
#    print(f"Sample {i}: counts shape = {sample.counts.shape if hasattr(sample.counts, 'shape') else len(sample.counts)}")

# Check the samples that will be grouped together
#print(f"\nChecking samples 9962-9998:")
#for i in range(9962, min(9998, len(col.collection))):
#    print(f"  Sample {i}: shape={col.collection[i].counts.shape}, timestamp={col.collection[i].timestamp}")

# ----------------------------------------
# Generate output files from the latest data
if len(col.collection) > 0:
    print("\nGenerating output files...")
    
    # Create last_spectrum directory if needed
    os.makedirs('./last_spectrum', exist_ok=True)
    
    # Write spectrum and image from most recent sample
    col.collection[-1].write_spe('./last_spectrum/rep.spe')
    col.collection[-1].write_last_update_image("last_update.png")
    
    print(f"Processing complete!")
    print(f"Total samples in database: {len(col.collection)}")
    print(f"Date range: {col.collection[0].timestamp} to {col.collection[-1].timestamp}")
else:
    print("No data available to generate outputs")

#os.makedirs('./last_spectrum', exist_ok=True)
#col.collection[-1].write_spe('./last_spectrum/rep.spe')
#col.collection[-1].write_last_update_image("last_update.png")
