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
weat_csv = r'/home/dosenet/radwatch-airmonitor/weatherhawk.csv'
weat_csv_sorted = r'weather_sorted.csv'
roi_dat = r'/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi.dat'

weather_utils.resort_weather_timestamps(weat_csv, weat_csv_sorted)
print("resorted the weather data")
col = sample_collection.SampleCollection()
print("Collection made")
col.build_collection(spec_dir, weat_csv_sorted)
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

# Add this line:
col.standardize_channel_counts()

col.rebin(datetime.timedelta(hours=1.0))
print('rebinned')
col.write_hdf('rebin.h5')
print('db written')
#col.rebin(datetime.timedelta(hours=8.));
#col.write_hdf('short.h5');

os.makedirs('./last_spectrum', exist_ok=True)
col.collection[-1].write_spe('./last_spectrum/rep.spe')
col.collection[-1].write_last_update_image("last_update.png")
