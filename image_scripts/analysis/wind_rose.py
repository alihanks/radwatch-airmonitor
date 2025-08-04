import h5py
import datetime
import time
import sys
sys.path.insert(0,'..')
from image_scripts import sample_collection
from image_scripts import weather_utils
from matplotlib.pyplot import *

file = h5py.File('./rebin.h5', 'r')
tmstmps=file['/2014/timestamps']
tm_meta=file['/2014/spectra_meta']
#spectra=file['/2014/spectra']
#cals=file['/2014/spectra_meta']
weather=file['/2014/weather_data']

#fix timestamps                                                                                                                      
timestamps=[]
for el in tmstmps:
    timestamps.append( datetime.datetime.fromtimestamp( time.mktime( time.gmtime(el) ) ) )

#rebin the wind data
p=0
k=0
wind_time_delta=datetime.timedelta(hours=6)
n_weather_dir=[]
n_weather_speed=[]
n_timestamps=[]
while( p < len(timestamps) ):
    tmp_stmp = timestamps[p]+wind_time_delta
    while( (k+1<len(timestamps)) and (timestamps[k] < tmp_stmp) ):
        k=k+1
    if(p==k):
        break
    n_weather_dir.append( weather[p:k-1,3] )
    n_weather_speed.append(weather[p:k-1,4])
    tm_diff = timestamps[k-1]-timestamps[p]
    n_timestamps.append( timestamps[p]+tm_diff/2 )
    p = k

#make all the windrose plots
for x in range(0,len(n_weather_dir)):
    if(len(n_weather_dir[x])<2):
        continue
    weather_utils.draw_windrose(n_weather_dir[x],n_weather_speed[x],str(n_timestamps[x]))
    
