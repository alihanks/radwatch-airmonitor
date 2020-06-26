import matplotlib;
import matplotlib.pyplot as plt;
import h5py;
import datetime;
import time;
import sys;
import os;
sys.path.insert(0,'..');
sys.path.insert(0,'.');
import sample_collection;
import weather_utils;
import time_utils;
import numpy as np;

file = h5py.File('rebin.h5','r');

spectra=file['/2014/spectra'];

cals=file['/2014/spectra_meta'];

tm_meta=file['/2014/spectra_meta'];

tmstmps=file['/2014/timestamps'];

#fix timestamps
timestamps=[];
for el in tmstmps:
    timestamps.append( datetime.datetime.fromtimestamp( time.mktime( time.gmtime(el) ) ) );


ax=[];
spec=np.sum(spectra[:,:],axis=0);
print len(spec),tm_meta[-1,1]/(24*3600), (3600-np.mean(tm_meta[:-1,1]))/12;
#spec=np.sum(spectra[:12,:],axis=0);

#energy axis generation
for x in range(0,len(spec)):
    ax.append( (x+3)*cals[1,3]+cals[1,2]);
col=sample_collection.SampleCollection();
col_comp=sample_collection.SampleCollection();
col.add_roi('/home/rtpavlovsk21/image_scripts/analysis/roi_simple.dat');
col.set_eff('/home/rtpavlovsk21/image_scripts/analysis/roof.ecc');
col_comp.add_roi('/home/rtpavlovsk21/image_scripts/analysis/roi.dat');

plt.semilogy(ax,spec);
plt.show();

plt.semilogy(timestamps,tm_meta[:,1]/3600);
plt.title('Rebinned Integration Time vs Time');
plt.ylabel('Rebinned Integration Time (1hr Nominal)');
plt.show();

plt.plot(timestamps);
plt.show();
