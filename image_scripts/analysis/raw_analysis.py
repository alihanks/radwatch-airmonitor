import sys
import datetime
sys.path.insert(0,'..')
from image_scripts import sample_collection
from image_scripts import weather_utils

spec_dir='/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/current/'
weat_csv='/home/dosenet/Dropbox/UCB Air Monitor/Data/Weather/weatherhawk.csv'
weat_csv_sorted='weather_sorted.csv'
roi_dat='/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi.dat'

weather_utils.resort_weather_timestamps(weat_csv,weat_csv_sorted)
print("resorted the weather data")
col = sample_collection.SampleCollection()
print("Collection made")
col.build_collection(spec_dir,weat_csv_sorted)
print("Collection built")
print("Size ", len(col.collection))
#col.rebin(datetime.timedelta(minutes=20));
#col.write_hdf('mod_rebin.h5');
col.rebin(datetime.timedelta(hours=1.0))
print('rebinned')
col.write_hdf('rebin.h5')
print('db written')
#col.rebin(datetime.timedelta(hours=8.));
#col.write_hdf('short.h5');

col.collection[-1].write_spe('./last_spectrum/rep.spe')
col.collection[-1].write_last_update_image("last_update.png")
