import sys
import os
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/')
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/image_scripts')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from image_scripts import weather_utils
from image_scripts import sample_collection
import datetime

spec_dir = r'/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/'
weat_csv = r'/home/dosenet/radwatch-airmonitor/image_scripts/analysis/weather.csv'
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
        
        # Merge new rebinned data into existing collection
        col.merge_collection(col_new)
        
        # Write updated HDF5
        print("\nWriting updated database...")
        col.write_hdf(hdf5_file)
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
        
        # Write HDF5
        print("\nWriting database...")
        col.write_hdf(hdf5_file)
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
