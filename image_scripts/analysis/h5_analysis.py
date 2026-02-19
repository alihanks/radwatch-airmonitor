plot_most_recent_spec = False
import matplotlib
if not plot_most_recent_spec:
    matplotlib.use('Agg')
from matplotlib.pyplot import *
import h5py
import datetime
import time
import sys
import os
import numpy as np  # Adding numpy import that was implied but not explicit
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/')
sys.path.insert(0, '/home/dosenet/radwatch-airmonitor/image_scripts')
sys.path.insert(0, '..')
sys.path.insert(0, '.')
sys.path.insert(0, '../..')
from image_scripts import sample_collection
from image_scripts import weather_utils
from image_scripts import time_utils
from image_scripts.spectrum_calibration import read_calibration_file

def time_indicies(timestamps,timed):
    p=0
    k=1
    ind=[]
    while p < len(timestamps):
        tmstmp = timestamps[p] + timed
        while k+1 < len(timestamps) and timestamps[k] < tmstmp:
            k += 1
        if p == k:
            break
        ind.append([p, k-1])
        p = k
    return ind
            
# color palette
#with open('../etc/colr/col_scheme.dat', 'r') as f:
#    col_pal = f.readlines()
#col_pal = [el.replace('\n', '') for el in col_pal]
col_pal = ['#00B2A5',
            '#D9661F',
            '#00B0DA',
            '#FDB515',
            '#ED4E33',
            '#2D637F',
            '#9DAD33',
            '#53626F',
            '#EE1F60',
            '#6C3302',
            '#C2B9A7',
            '#CFDD45',
            '#003262']

# get the data
file = h5py.File('./rebin.h5', 'r')

# FIXED: Support both 'data' group (new) and legacy year groups (e.g., '2014')
if 'data' in file:
    data_group = file['data']
    print("Loading from 'data' group")
else:
    # Legacy support
    available_groups = list(file.keys())
    if len(available_groups) == 0:
        print("ERROR: No data groups found in HDF5 file")
        sys.exit(1)
    data_group = file[available_groups[0]]
    print(f"Loading from legacy group '{available_groups[0]}'")

required_datasets = ['timestamps', 'spectra_meta', 'spectra', 'weather_data']
for ds in required_datasets:
    if ds not in data_group:
        print(f"ERROR: Required dataset '{ds}' not found in HDF5 file. The data group is empty.")
        print("This likely means no spectral files were processed. Check that the spectral data directory exists.")
        sys.exit(1)

tmstmps = data_group['timestamps']
tm_meta = data_group['spectra_meta']
spectra = data_group['spectra']
print(spectra)
cals = data_group['spectra_meta']
weather = data_group['weather_data']


# FIXED: Use proper shape access
s = spectra.shape
print('spectra shape', s)
print(os.getcwd())

# Check we have data
if s[0] == 0:
    print("ERROR: No spectra in HDF5 file")
    sys.exit(1)

# Check weather data shape
print(f"Weather data shape: {weather.shape}")
if weather.shape[1] < 7:
    print("ERROR: Weather data doesn't have expected number of columns")
    sys.exit(1)

ax = []
spec = spectra[-1, :]
# print("comment the line after this in order to get 1hr spec")
# spec = np.sum(spectra[:300, :], axis=0)

# FIXED: energy axis generation using last sample's calibration
#last_cal = cals[-1]  # Get calibration for most recent spectrum

# energy axis generation
for x in range(0, len(spec)):
    ax.append((x+1)*cals[1, 3]+cals[1, 2])

col = sample_collection.SampleCollection()
col_comp = sample_collection.SampleCollection()
calibration = read_calibration_file('/home/dosenet/radwatch-airmonitor/image_scripts/calibration/calibration_coefficients.txt')
col.add_roi_energy('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi_energy.dat', calibration)
col.set_eff('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roof.ecc')
col_comp.add_roi('/home/dosenet/radwatch-airmonitor/image_scripts/analysis/roi.dat')



#eff=col.get_eff_for_binning(ax); 
#for x in range(0,len(spec)):
#    if(eff[x]==0.):
#        spec[x]=0;
#        continue;
#    spec[x]=spec[x]/eff[x];

# roi report
res = np.zeros((len(col.rois), 2))
k = 0
for el in col.rois:
    counts = el.get_counts(spec)
    print(f"get_counts output: {counts}")
    res[k, :] = counts
    k += 1

#printing rois
roi_ = np.zeros(len(spec))
roi_comp = np.zeros(len(spec))
lst_peak = 0
lst_peak_height = 0
for el in col.rois:
    roi_[el.peak[0]:el.peak[1]] = spec[el.peak[0]:el.peak[1]]
for el in col_comp.rois:
    roi_comp[el.peak[0]:el.peak[1]] = spec[el.peak[0]:el.peak[1]]
    try:
        super_tmp = el.peak[0] + np.argmax(spec[el.peak[0]:el.peak[1]])
        if el.peak[0] - lst_peak < 100:
            if lst_peak_height + 25 > spec[super_tmp]:
                shift = 120 * max([lst_peak_height, spec[super_tmp]]) / (min([lst_peak_height, spec[super_tmp]]))
                shift = min([150, shift])
                print(el.isotope, lst_peak_height, lst_peak, spec[super_tmp], el.peak[0])
            else:
                shift = 80
        else:
            shift = 80
    except Exception as e:
        print(e)
        print(f"Spectrum object: {el}")
        shift = 80
    annotate(el.isotope, (ax[super_tmp], spec[super_tmp]), xytext=(0, shift), textcoords='offset points', rotation=90, 
             ha='center', va='center', arrowprops=dict(width=0.25, headwidth=0, shrink=0.05))
    lst_peak = el.peak[1]
    lst_peak_height = spec[super_tmp]

semilogy(ax[30:], spec[30:], col_pal[5])
semilogy(ax, roi_comp, col_pal[6])
semilogy(ax, roi_, col_pal[8])
if plot_most_recent_spec:
    show()
else:
    xlim([0, 3000])
    xlabel('Energy (keV)')
    ylabel('Counts')
    title('Past Hour\'s Gamma Spectrum')
    savefig('most_recent_spectra.png', transparent=True, bbox_inches='tight')
clf()

#make isotope table
cell_txt = []
col_txt = ["Isotope", "Origin"]
f, tmp_ax = subplots(1)
f.patch.set_visible(False)
f.subplots_adjust(bottom=0.4, top=0.6)
tmp_ax.xaxis.set_visible(False)
tmp_ax.yaxis.set_visible(False)
for el in col.rois:
    cell_txt.append([el.isotope, el.origin])
color = (0.18, 0.39, 0.50)
table(cellText=cell_txt, colLabels=col_txt, loc='center', colLoc='center', rowLoc='center', 
      cellLoc='center', colColours=[color]*len(col_txt))
# title('Timeseries Isotope Origins')

# savefig('isotope_table.png', transparent=True, bbox_inches='tight')
# savefig('isotope_table.png', bbox_inches='tight')
clf()

#fix timestamps
timestamps = []
for el in tmstmps:
    timestamps.append(datetime.datetime.fromtimestamp(time.mktime(time.gmtime(el))))

clf()

#make all the time roses
p = 0
for x in time_utils.time_wins:
    tmp_timstmp = []
    tmp_win_dir = []
    tmp_win_speed = []
    k = weather_utils.discard_data_before_time(timestamps, x, 0) + 1
    print(x, weather[k:, 4], weather[k:, 5])
    weather_utils.draw_windrose(weather[k:, 4], weather[k:, 5], weather_utils.time_wins_str[p])
    p += 1
clf()
cla()

#weather_data
#for x in range(0,len(weather[1,:])):
#    wea=4;
#    max_=np.max( weather[:,wea] );
#    tmp=weather[:,wea]/max_;
#    plot(timestamps,tmp);

#count rate array
roi_array = np.zeros((s[0], len(col.rois), 2))
for x in range(0, s[0]):
    tmp = np.zeros((len(col.rois), 2))
    k = 0
    for el in col.rois:
        tmp[k] = el.get_counts(spectra[x, :]) / tm_meta[x, 1]
        k += 1
    roi_array[x, :, :] = tmp

# Replace invalid data points with NaN so plots show gaps instead of zeros
for x in range(0, s[0]):
    if tm_meta[x, 1] <= 0 or np.sum(spectra[x, :]) == 0:
        roi_array[x, :, :] = np.nan


#plot all isotopes
#gs=matplotlib.gridspec.GridSpec(2,1,height_ratios=[1,3]);
axarr = [subplot2grid((5, 4), (0, 0), colspan=4), subplot2grid((5, 4), (1, 0), rowspan=3, colspan=4), 
         subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4)]
gcf().add_subplot(axarr[0])
gcf().add_subplot(axarr[0])
axarr[0].axes.get_xaxis().set_visible(False)
axarr[1].axes.get_xaxis().set_visible(False)
bar_ax = axarr[0].twinx()
bar_ax0 = axarr[2].twinx()
for x in range(0, len(col.rois)):
    axarr[0].plot(timestamps, weather[:, 0], color=col_pal[0])
    bar_ax.plot(timestamps, weather[:, 1], color=col_pal[1])
    axarr[1].errorbar(timestamps, roi_array[:, x, 0], yerr=roi_array[:, x, 1], marker='+', 
                      label=col.rois[x].isotope, color=col_pal[4+x])
    axarr[2].plot(timestamps, weather[:, 6], color=col_pal[2])
    bar_ax0.plot(timestamps, weather[:, 2], color=col_pal[3])

with open('weather.csv', 'w') as out_file, open('weather_bq.csv', 'w') as out_file_bq:
    # out_file.write("Time, Pb212, Bi214, Pb212, Tl208, K40, Cs134, Cs137\n")
    out_file.write("Time, Bi214, K40, Cs134, Cs137\n")
    out_file_bq.write("Time, Bi214, K40, Cs134, Cs137\n")

    mins = np.argmin(roi_array[:, :, 0], axis=0)
    print(mins)
    for x in range(max(0, len(timestamps) - 15 * 24), len(timestamps)):
        out_str = str(timestamps[x])
        out_str_bq = str(timestamps[x])
        for y in range(0, len(roi_array[x, :, 0])):
            if y == 0 or y == 2 or y == 3:
                continue
            eff = col.get_eff_for_binning(col.rois[y].energy)
            out_str_bq += "," + str((roi_array[x, y, 0] - roi_array[mins[y], y, 0]) / eff) + "," + \
                         str(np.sqrt(roi_array[x, y, 1]**2 + roi_array[mins[y], y, 1]**2) / eff)
            
            out_str += "," + str((roi_array[x, y, 0] - roi_array[mins[y], y, 0])) + "," + \
                      str(np.sqrt(roi_array[x, y, 1]**2 + roi_array[mins[y], y, 1]**2))
        out_str += "\n"
        out_file.write(out_str)
        out_str_bq += "\n"
        out_file_bq.write(out_str_bq)

for ba_l in bar_ax.get_yticklabels():
    ba_l.set_color(col_pal[1])
for ba_l in axarr[0].get_yticklabels():
    ba_l.set_color(col_pal[0])
for ba_l in axarr[2].get_yticklabels():
    ba_l.set_color(col_pal[2])
for ba_l in bar_ax0.get_yticklabels():
    ba_l.set_color(col_pal[3])

#fix the plots
mx = np.amax(roi_array)
axarr[1].set_ylim([0, mx * 1.05])
legend = axarr[1].legend(loc='upper left', prop={'size': 10})
gcf().subplots_adjust(wspace=2.5, top=1.1)

p = 0
for x in (time_utils.time_wins):
    lower = max([time_utils.beginning_date, x, timestamps[0]])
    axarr[0].set_xlim([lower, time_utils.right_now])
    axarr[1].set_xlim([lower, time_utils.right_now])
    axarr[2].set_xlim([lower, time_utils.right_now])
    setp(axarr[2].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axarr[0].set_ylabel('TEMP (deg C)')
    bar_ax.set_ylabel('PRESSURE (kPa)')
    bar_ax0.set_ylabel('SOLAR (W/m2)')
    axarr[1].set_ylabel('COUNT RATE (1/sec)')
    axarr[2].set_ylabel('RAIN (mm/hr)')
    axarr[2].set_title('Rain and Solar Data', fontweight='bold')
    axarr[1].set_title('Count Rate of Select Peaks vs Time', fontweight='bold')
    axarr[0].set_title('Temperture, Barometric Pressure', fontweight='bold')
    savefig('iso_' + time_utils.time_wins_str[p] + '.png', transparent=True, bbox_inches='tight')
    p += 1

fig = figure()
p = 0
for x in time_utils.time_wins:
    lower = max([time_utils.beginning_date, x, timestamps[0]])
    new_axis = axarr[1]
    new_axis.set_xlim([lower, time_utils.right_now])
    new_axis.axes.get_xaxis().set_visible(True)
    setp(new_axis.xaxis.get_majorticklabels(), rotation=45, ha="right")
    savefig('trimmed_' + time_utils.time_wins_str[p] + '.png', transparent=False, bbox_inches='tight')
    p += 1
#weather_utils.draw_windrose(weather[:,3],weather[:,4],'hour');

# tmp plot of rainfall
clf()
# make cummulative sum
rain_data = np.nan_to_num(weather[:, 6], nan=0.0)
cum_sum = np.cumsum(rain_data)
# plot(timestamps, weather[:, 6])
plot(timestamps, cum_sum)
savefig('out.png')