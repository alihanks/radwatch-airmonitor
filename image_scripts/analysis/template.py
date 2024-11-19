import matplotlib;
from matplotlib.pyplot import *;
import h5py;
import datetime;
import time;
import numpy as np;
import base64;
import sys;
sys.path.insert(0,'..');
import sample_collection;

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
print datestmps[0];# oldest timestmp
print datestmps[-1];# newest timestamp
di=datestmps[-1]-datestmps[0]; # the difference in time
print di;

out_file=open('out.csv','w');
for x in range(0,len(datestmps)):
    out_file.write(str(datestmps[x]) +","+str(weather[x,0])+"\n")

#enter a beginning date
begin_date=datetime.datetime(year=2014,month=2,day=1,hour=0);
#end_date=datetime.datetime(year=2014,month=2,day=27,hour=0);

#enter an end date
end_date=datetime.datetime.now()+datetime.timedelta(days=10);
print "time between the two dates entered:",end_date-begin_date;

#alright get all the appropriate spectra
inds=return_inds_between_dates(datestmps,begin_date,end_date);
print np.sum(weather[:,6]);
print np.sum(weather[inds[0]:inds[-1],6]);
plot( datestmps[inds[0]:inds[-1]],weather[inds[0]:inds[-1],6]);
show();
spec=[];
for el in inds:
    spec.append(spectra[el,:]);
spec=np.asarray(spec);
spec_sum=np.sum(spec,0);
#semilogy(spec_sum);
#show();

#lets readin the rois
col_comp=sample_collection.SampleCollection();
col_comp.add_roi('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi.dat');

#make something with dimensions spec, nonzeros
#at rois
roi_comp=np.zeros(len(spec_sum));
for el in col_comp.rois:
    roi_comp[el.peak[0]:el.peak[1]]=spec_sum[el.peak[0]:el.peak[1]];
#semilogy(spec_sum);
#semilogy(roi_comp,'r');
#show()

tmp_smp=sample_collection.Sample();
tmp_smp.set_spectra(datetime.datetime.now(),1,1,[0,len(spec_sum)],[tmmeta[1,3],tmmeta[1,2]],[],spec_sum);
tmp_smp.write_spe('int.spe');
semilogy(spec_sum);
show();
