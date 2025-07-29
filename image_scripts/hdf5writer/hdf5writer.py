import sys
sys.path.insert(0,'..')
from image_scripts import  sample_collection

spec_dir='/home/dosenet/radwatch-airmonitor/Dropbox/UCB Air Monitor/Data/Roof/PAVLOVSKY/'
weat_csv='/home/dosenet/radwatch-airmonitor/Dropbox/UCB Air Monitor/Data/Weather/weatherhawk.csv'

col = sample_collection.SampleCollection()
col.build_collection(spec_dir,weat_csv)
col.write_hdf('out.h5')
