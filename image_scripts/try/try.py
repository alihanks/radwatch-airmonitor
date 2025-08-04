import sys
import h5py
import datetime
sys.path.insert(0,'..')
from image_scripts import spectra_utils
from image_scripts import weather_utils
from image_scripts import sample_collection
import calendar
from matplotlib.pyplot import *
import numpy as np
import glob
import os
#import code

def weighted_mean(first,second):
    sum_=[0,0]
    for n in range(0,len(first)):
            sum_[0]=sum_[0]+first[n]*second[n]
            sum_[1]=sum_[1]+first[n]
    if(sum_[1]==0 or 0==len(first)):
        return 0
    else:
        return float(sum_[0])/float(sum_[1])

spec_dir='/home/dosenet/radwatch-airmonitor/Dropbox/UCB Air Monitor/Data/Roof/PAVLOVSKY/'

timestamps_index=1
temperature_index=2
rh_index=7
pres_index=8
sola_index=13
winds_index=17
windd_index=18

list_of_lists,units,label,is_time_str,order=weather_utils.parse_weather_data(sys.argv[1])

k=0
col= sample_collection.SampleCollection()
for fil in glob.glob(spec_dir+'*/*.CNF'):
    k=0
    sa = sample_collection.Sample()
    sa= spectra_utils.parse_spectra(fil,sa)
    p=weather_utils.discard_data_before_time(list_of_lists[timestamps_index],sa.get_timestamp(),k)
    k=weather_utils.discard_data_before_time(list_of_lists[timestamps_index],sa.get_timestamp()+sa.get_real_time(),p)
    sa.set_weather_params(list_of_lists[timestamps_index][p:k-1],list_of_lists[temperature_index][p:k-1],list_of_lists[sola_index][p:k-1],list_of_lists[rh_index][p:k-1],list_of_lists[winds_index][p:k-1],list_of_lists[windd_index][p:k-1])
    col.add_sample(sa)
    
print(len(col.collection))

out_file=h5py.File('test_database.hdf5','w')
ths_yr=out_file.create_group('2014')
timstmps=[]
specs=[]
weather_list=[]#weather_el=np.zeros(5)
for stmp in col.collection:
    timstmps.append(np.uint32(calendar.timegm(stmp.get_timestamp().utctimetuple())))
    specs.append( stmp.counts )
    if(stmp.bool_weather_set()):
        weather_el=[ np.mean(stmp.temp), np.mean(stmp.solar), np.mean(stmp.relh), np.mean(stmp.wind_speed), weighted_mean(stmp.wind_speed,stmp.wind_dir)]
    else:
        weather_el=np.zeros(5)
        weather_el[:]=np.nan
    weather_list.append(weather_el)
print(weather_list)
ths_yr.create_dataset('timestamps',data=timstmps)
ths_yr.create_dataset('spectra',data=specs, dtype=np.dtype('uint32'))
ths_yr.create_dataset('weather_data',data=weather_list,dtype=float)
#code.interact(local=locals())
