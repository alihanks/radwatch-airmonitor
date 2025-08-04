from past.utils import old_div
plot_most_recent_spec=False
import matplotlib
if(not plot_most_recent_spec):
    matplotlib.use('Agg')
from matplotlib.pyplot import *
from matplotlib.gridspec import GridSpec
import h5py
import datetime
import time
import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'.')
from image_scripts import sample_collection
from image_scripts import weather_utils
from image_scripts import time_utils

def time_indicies(timestamps,timed):
    p=0; k=1; ind=[]
    while( p < len(timestamps)):
        tmstmp=timestamps[p]+timed
        while( k+1 < len(timestamps) and timestamps[k]<tmstmp):
            k+=1
        if(p==k):
            break
        ind.append([p,k-1])
        p=k
    return ind


file = h5py.File('./rebin.h5', 'r')
tmstmps=file['/2014/timestamps']
tm_meta=file['/2014/spectra_meta']
spectra=file['/2014/spectra']
cals=file['/2014/spectra_meta']
weather=file['/2014/weather_data']
s=[len(spectra[:,1]),len(spectra[1,:])]
print('spectra shape', s)
print(os.getcwd())

ax=[]
spec=spectra[-1,:]

#energy axis generation
for x in range(0,len(spec)):
    ax.append( (x+1)*cals[1,3]+cals[1,2])
col=sample_collection.SampleCollection()
col_comp=sample_collection.SampleCollection()
col.add_roi('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi_simple.dat')
col.set_eff('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roof.ecc')
col_comp.add_roi('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi.dat')

#eff=col.get_eff_for_binning(ax)
#for x in range(0,len(spec)):
#    if(eff[x]==0.):
#        spec[x]=0
#        continue
#    spec[x]=spec[x]/eff[x]

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
for el in col_comp.rois:
    roi_comp[el.peak[0]:el.peak[1]]=spec[el.peak[0]:el.peak[1]]
    super_tmp=el.peak[0]+np.argmax(spec[el.peak[0]:el.peak[1]])
    if( el.peak[0]-lst_peak < 100 ):
        if( lst_peak_height+25>spec[super_tmp] ):
            shift=old_div(120*max([lst_peak_height,spec[super_tmp]]),(min([lst_peak_height,spec[super_tmp]])))
            shift=min([150,shift])
            print(el.isotope, lst_peak_height, lst_peak, spec[super_tmp], el.peak[0])
        else:
            shift=80
    else:
        shift=80
    annotate(el.isotope, (ax[super_tmp],spec[super_tmp]),xytext=(0,shift),textcoords='offset points',rotation=90,ha='center',va='center',arrowprops=dict(width=0.25,headwidth=0,shrink=0.05))
    lst_peak= el.peak[1]
    lst_peak_height = spec[super_tmp]

plot(ax[30:],spec[30:],'b')
plot(ax,roi_comp,'y')
plot(ax,roi_,'r')
if( plot_most_recent_spec==True ):
    show()
else:
    xlim([0,3000])
    xlabel('Energy (keV)')
    ylabel('Counts')
    title('Past Hour\'s Gamma Spectrum')
    savefig('most_recent_spectra.png',transparent=True,bbox_inches='tight')
clf()

#make isotope table
cell_txt=[]
col_txt=[ "Isotope","Origin"]
f,tmp_ax=subplots(1)
f.patch.set_visible(False)
f.subplots_adjust(bottom=0.4,top=0.6)
tmp_ax.xaxis.set_visible(False)
tmp_ax.yaxis.set_visible(False)
for el in col.rois:
    cell_txt.append([el.isotope, el.origin])
color=(0.097,0.34,0.44)
table(cellText=cell_txt,colLabels=col_txt,loc='center',colLoc='center',rowLoc='center',cellLoc='center',colColours=[color]*len(col.rois))
#title('Timeseries Isotope Origins')
savefig('isotope_table.png',transparent=True,bbox_inches='tight')
clf()

#fix timestamps
timestamps=[]
for el in tmstmps:
    timestamps.append( datetime.datetime.fromtimestamp( time.mktime( time.gmtime(el) ) ) )

clf()

#make all the time roses
p=0
for x in time_utils.time_wins:
    tmp_timstmp=[]; tmp_win_dir=[]; tmp_win_speed=[]
    k=weather_utils.discard_data_before_time(timestamps,x,0)
    weather_utils.draw_windrose(weather[k:,4],weather[k:,5],weather_utils.time_wins_str[p])
    p+=1
clf(); cla()

#weather_data
#for x in range(0,len(weather[1,:])):
#    wea=4;
#    max_=np.max( weather[:,wea] );
#    tmp=weather[:,wea]/max_;
#    plot(timestamps,tmp);

#count rate array
roi_array=np.zeros((s[0],len(col.rois),2))
for x in range(0,s[0]):
    tmp=np.zeros((len(col.rois),2))
    k=0
    for el in col.rois:
        tmp[k]=old_div(el.get_counts(spectra[x,:]),tm_meta[x,1])
        k+=1
    roi_array[x,:,:]=tmp

#plot all isotopes
gs = GridSpec(2,1,height_ratios=[1,3])
axarr=[subplot2grid((5,4),(0,0),colspan=4),subplot2grid((5,4),(1,0),rowspan=3,colspan=4),subplot2grid((5,4),(4,0),rowspan=1,colspan=4)]
gcf().add_subplot(axarr[0])
gcf().add_subplot(axarr[0])
axarr[0].axes.get_xaxis().set_visible(False)
axarr[1].axes.get_xaxis().set_visible(False)
bar_ax=axarr[0].twinx()
bar_ax0=axarr[2].twinx()
for x in range(0,len(col.rois)):
    axarr[0].plot(timestamps,weather[:,0], color='#097054')
    bar_ax.plot(timestamps,weather[:,1],color='#FF6600')
    eff=col.get_eff_for_binning(col.rois[x].energy)
    axarr[1].errorbar(timestamps,old_div((roi_array[:,x,0]-np.min(roi_array[:,x,0])),eff),yerr=roi_array[:,x,1],marker='+',label=col.rois[x].isotope)
    axarr[2].plot(timestamps,weather[:,6],color='#E96D63')
    bar_ax0.plot(timestamps,weather[:,2],color='#4A789C')

out_file=open('out.csv','w')
for x in range(0,len(timestamps)):
    out_str=str(timestamps[x])
    for y in range(0,len(roi_array[x,:,0])):
        out_str+=","+str(roi_array[x,y,0])+","+str(roi_array[x,y,1])
    out_str+="\n"
    out_file.write(out_str)

for ba_l in bar_ax.get_yticklabels():
    ba_l.set_color('#FF6600')
for ba_l in axarr[0].get_yticklabels():
    ba_l.set_color('#097054')
for ba_l in axarr[2].get_yticklabels():
    ba_l.set_color('#E96D63')
for ba_l in bar_ax0.get_yticklabels():
    ba_l.set_color('#4A789C')
#fix the plots
mx=np.amax(roi_array)
#axarr[1].set_ylim([0,mx*1.05])
legend=axarr[1].legend(loc='upper left',prop={'size':10})
gcf().subplots_adjust(bottom=0.05)

p=0
for x in (time_utils.time_wins):
    lower=max([time_utils.beginning_date,x])
    axarr[0].set_xlim([lower,time_utils.right_now])
    axarr[1].set_xlim([lower,time_utils.right_now])
    axarr[2].set_xlim([lower,time_utils.right_now])
    setp(axarr[2].xaxis.get_majorticklabels(),rotation=45,ha="right")
    axarr[0].set_ylabel('Temp. (deg C)')
    bar_ax.set_ylabel('Pressure (kPa)')
    bar_ax0.set_ylabel('Solar (W/m2)')
    axarr[1].set_ylabel('Count Rate (1/sec)')
    axarr[2].set_ylabel('Rain (mm/hr)')
    axarr[2].set_title('Rain and Solar Data')
    axarr[1].set_title('Count Rate of Select Peaks vs Time')
    axarr[0].set_title('Temp. (Green), Barometric Pres. (Orange)')
    savefig('iso_'+time_utils.time_wins_str[p]+'.png',transparent=True,bbox_inches='tight')
    p+=1

#weather_utils.draw_windrose(weather[:,3],weather[:,4],'hour')

#tmp plot of rainfall
clf()
#make cummulative sum
cum_sum=np.zeros(len(weather[:,6]))
for x in range(1,len(weather[:,6])):
    cum_sum[x]=cum_sum[x-1]+weather[x,6]
#plot(timestamps,weather[:,6])
plot( timestamps,cum_sum)
savefig('out.png')

clf()
hist(roi_array[:,2,0],250)
savefig('tmp.png')
