from builtins import range
import matplotlib.pyplot as plt
import h5py
import numpy as np
import datetime
import time
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
from image_scripts import sample_collection

col_pal=open('../etc/colr/col_scheme.dat').readlines()
col_pal=[el.replace('\n','') for el in col_pal]

file = h5py.File('./rebin.h5', 'r')
tmstmps=file['/2014/timestamps']
tm_meta=file['/2014/spectra_meta']
spectra=file['/2014/spectra']
cals=file['/2014/spectra_meta']
weather=file['/2014/weather_data']

timestamps=[]
for el in tmstmps:
    timestamps.append( datetime.datetime.fromtimestamp( time.mktime( time.gmtime(el) ) ) )

ax=[]
spec=np.sum(spectra[-150:-100,:],0)
spec1=np.sum(spectra[-50:,:],0)
#energy axis generation
for x in range(0,len(spec)):
    #ax.append( (x+1)*cals[1,3]+cals[1,2])
    ax.append(x)
col=sample_collection.SampleCollection()
col_comp=sample_collection.SampleCollection()
col.add_roi('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi_simple.dat')
col.set_eff('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roof.ecc')
col_comp.add_roi('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi.dat')

#roi report
res=np.zeros((len(col.rois),2))
k=0
for el in col.rois:
    res[k,:]=el.get_counts(spec)
    k+=1

#printing rois
roi_=np.zeros(len(spec))
roi_comp=np.zeros(len(spec))
lst_peak=0
lst_peak_height=0
for el in col.rois:
    roi_[el.peak[0]:el.peak[1]]=spec[el.peak[0]:el.peak[1]]

plt.semilogy(ax[30:],spec1[30:],col_pal[5])
plt.semilogy(ax[30:],spec[30:],col_pal[5])
plt.semilogy(ax,roi_,col_pal[8])
plt.show()
