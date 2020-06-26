import matplotlib;
from matplotlib.pyplot import *;
import h5py;
import datetime;
import time;
import numpy as np;

def datetime_from_tmstmps(tmstmps):
    timestmps=[];
    for el in tmstmps:
        timestmps.append( datetime.datetime.fromtimestamp( time.mktime( time.gmtime(el))));
    return timestmps;

def return_inds_between_dates(datestmps,begin_date,end_date):
    inds=[];
    counter=0;
    while( begin_date>datestmps[counter] ):
        counter+=1;
    while( end_date>datestmps[counter] and counter<len(datestmps)-1):
        inds.append(counter);
        counter+=1;
    return inds;

def between_dates(datestmps,beg_date,end_date):
    between_dates=[];
    index=0;
    while( beg_date>datestmps[index]):
        index+=1;
    while( end_date>datestmps[index]):
        between_dates.append(index);
        index+=1;
    return between_dates;

#open h5 file, make sure this is in your directory
file = h5py.File('./rebin.h5','r');

#get the timestamps
tmstmp=file['/2014/timestamps'];

#get the metadata for the spectra
tmmeta=file['/2014/spectra_meta'];

#get the spectra
spectra=file['/2014/spectra'];

#get the weather
weather=file['/2014/weather_data'];

#look at timedelta between timestamps
datestmps=datetime_from_tmstmps(tmstmp);
print "oldest timestamp",datestmps[0];# oldest timestmp
print "newest timestamp",datestmps[-1];# newest timestamp
print "difference in time",datestmps[-1]-datestmps[0]; # the difference in time
diff=datestmps[-1]-datestmps[0];
print "difference in time, seconds",diff.total_seconds();

#plot the first spectrum, corresponding to the oldest timestamp
#plot(spectra[0,:]);
#show();

#compute total time of acquisition
total_time_s=np.sum(tmmeta[0:9,1]);
print total_time_s;

#lets sum 10 spectra together
sum_1=np.sum(spectra[0:9,:],axis=0);
#plot(sum_/total_time_s);
#show()

date1=datetime.datetime(2014,2,26,8);
date2=datetime.datetime(2014,4,4,12);
print date1,date2;

#return 1's for dates between_dates
datestmp_mask=between_dates(datestmps,date1,date2);
print "time difference between stmps",datestmps[1]-datestmps[0];
print date2-date1;
print len(datestmp_mask);

energy_params=np.asarray(tmmeta[0,2:4]);
print energy_params;
energy_axis=[ energy_params[0]+energy_params[1]*x for x in range(0,len(spectra[0,:])) ];

sum_2=np.sum(spectra[datestmp_mask[0]:datestmp_mask[-1],:],axis=0);
semilogy(energy_axis,spectra[0,:]);
semilogy(energy_axis,sum_1);
semilogy(energy_axis,sum_2);
show();
