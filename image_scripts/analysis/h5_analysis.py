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

PROJECT_ROOT = '/home/dosenet/radwatch-airmonitor'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def build_merged_weather(rad_timestamps, hdf5_weather, csv_timestamps, csv_temp,
                         csv_pres, csv_rain, csv_solar, gap_threshold_sec):
    """Build merged weather timeline: HDF5 data where radiation exists, CSV data in gaps."""
    merged_ts = list(rad_timestamps)
    merged_temp = list(hdf5_weather[:, 0])
    merged_pres = list(hdf5_weather[:, 1])
    merged_rain = list(hdf5_weather[:, 6])
    merged_solar = list(hdf5_weather[:, 2])

    for i in range(1, len(rad_timestamps)):
        gap_start = rad_timestamps[i - 1]
        gap_end = rad_timestamps[i]
        dt_sec = (gap_end - gap_start).total_seconds()
        if dt_sec > gap_threshold_sec:
            for j, ct in enumerate(csv_timestamps):
                if ct > gap_start and ct < gap_end:
                    merged_ts.append(ct)
                    merged_temp.append(csv_temp[j])
                    merged_pres.append(csv_pres[j])
                    merged_rain.append(csv_rain[j])
                    merged_solar.append(csv_solar[j])

    sort_idx = sorted(range(len(merged_ts)), key=lambda k: merged_ts[k])
    merged_ts = [merged_ts[k] for k in sort_idx]
    merged_temp = [merged_temp[k] for k in sort_idx]
    merged_pres = [merged_pres[k] for k in sort_idx]
    merged_rain = [merged_rain[k] for k in sort_idx]
    merged_solar = [merged_solar[k] for k in sort_idx]

    return merged_ts, merged_temp, merged_pres, merged_rain, merged_solar

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
file = h5py.File(os.path.join(DATA_DIR, 'rebin.h5'), 'r')

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

# Load continuous weather data from CSV for gap-filling
weat_csv_sorted = os.path.join(DATA_DIR, 'weather_sorted.csv')
csv_weather_available = os.path.exists(weat_csv_sorted)
if csv_weather_available:
    csv_weather_raw, _, _, _, _ = weather_utils.parse_weather_data(weat_csv_sorted)
    csv_weather_timestamps = [t.replace(tzinfo=None) for t in csv_weather_raw[1]]
    csv_weather_temp = csv_weather_raw[2]
    csv_weather_pres = csv_weather_raw[8]
    csv_weather_rain = csv_weather_raw[12]
    csv_weather_solar = csv_weather_raw[13]
    print(f"Loaded {len(csv_weather_timestamps)} weather CSV records for gap-filling")
else:
    print("weather_sorted.csv not found, weather will only use HDF5 data")

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

spec = spectra[-1, :]
# print("comment the line after this in order to get 1hr spec")
# spec = np.sum(spectra[:300, :], axis=0)

col = sample_collection.SampleCollection()
calibration = read_calibration_file(os.path.join(PROJECT_ROOT, 'image_scripts', 'calibration', 'calibration_coefficients.txt'))

# energy axis generation using external calibration file (matches ROI channel mapping)
ax = [calibration[0] + calibration[1] * (x + 1) for x in range(len(spec))]
col.add_roi_energy(os.path.join(PROJECT_ROOT, 'image_scripts', 'analysis', 'roi_energy.dat'), calibration)
col.set_eff(os.path.join(PROJECT_ROOT, 'image_scripts', 'analysis', 'roof.ecc'))

# roi report
res = np.zeros((len(col.rois), 2))
k = 0
for el in col.rois:
    counts = el.get_counts(spec)
    print(f"get_counts output: {counts}")
    res[k, :] = counts
    k += 1

# Build highlighted ROI overlay and place isotope labels at peak positions.
# Channels come from the energy-based ROI definitions, so labels track the
# current calibration rather than the legacy channel-hardcoded roi.dat.
roi_ = np.zeros(len(spec))
lst_peak = 0
lst_peak_height = 0
for el in col.rois:
    roi_[el.peak[0]:el.peak[1]] = spec[el.peak[0]:el.peak[1]]
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
semilogy(ax, roi_, col_pal[8])
if plot_most_recent_spec:
    show()
else:
    xlim([0, 3000])
    xlabel('Energy (keV)')
    ylabel('Counts')
    title('Past Hour\'s Gamma Spectrum')
    savefig(os.path.join(DATA_DIR, 'most_recent_spectra.png'), transparent=True, bbox_inches='tight')
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
    weather_utils.draw_windrose(weather[k:, 4], weather[k:, 5], weather_utils.time_wins_str[p], output_dir=DATA_DIR)
    p += 1
clf()
cla()

#count rate array — use stored ROI data if available, else compute on-the-fly
if 'roi_counts' in data_group:
    roi_array = data_group['roi_counts'][:]
    print(f"Using stored ROI counts: shape {roi_array.shape}")
else:
    print("No stored ROI counts found, computing on-the-fly...")
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


# Insert NaN breaks wherever time gaps exceed 2 hours so radiation plots
# show gaps instead of drawing misleading straight lines across data outages
GAP_THRESHOLD_SEC = 2 * 3600
rad_timestamps = list(timestamps)  # Save pre-NaN-break timestamps for weather merging
gap_indices = []
for i in range(1, len(timestamps)):
    dt = (tmstmps[i] - tmstmps[i-1])
    if dt > GAP_THRESHOLD_SEC:
        gap_indices.append(i)

if len(gap_indices) > 0:
    print(f"Inserting {len(gap_indices)} NaN breaks for time gaps > 2 hours")
    new_timestamps = list(timestamps)
    new_roi = list(roi_array)
    nan_roi_row = np.full((roi_array.shape[1], roi_array.shape[2]), np.nan)
    for idx in reversed(gap_indices):
        mid_time = datetime.datetime.fromtimestamp(
            time.mktime(time.gmtime((tmstmps[idx-1] + tmstmps[idx]) / 2)))
        new_timestamps.insert(idx, mid_time)
        new_roi.insert(idx, nan_roi_row)
    timestamps = new_timestamps
    roi_array = np.array(new_roi)
    print(f"Data expanded from {s[0]} to {len(timestamps)} points (with NaN breaks)")

# Build merged weather timeline: HDF5 data where radiation exists, CSV data in gaps
if csv_weather_available:
    weather_timestamps, weather_temp, weather_pres, weather_rain, weather_solar = build_merged_weather(
        rad_timestamps, weather, csv_weather_timestamps, csv_weather_temp,
        csv_weather_pres, csv_weather_rain, csv_weather_solar, GAP_THRESHOLD_SEC)
    print(f"Merged weather timeline: {len(weather_timestamps)} points")
else:
    weather_timestamps = rad_timestamps
    weather_temp = list(weather[:, 0])
    weather_pres = list(weather[:, 1])
    weather_rain = list(weather[:, 6])
    weather_solar = list(weather[:, 2])

#plot all isotopes
axarr = [subplot2grid((5, 4), (0, 0), colspan=4), subplot2grid((5, 4), (1, 0), rowspan=3, colspan=4),
         subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4)]
gcf().add_subplot(axarr[0])
gcf().add_subplot(axarr[0])
axarr[0].axes.get_xaxis().set_visible(False)
axarr[1].axes.get_xaxis().set_visible(False)
bar_ax = axarr[0].twinx()
bar_ax0 = axarr[2].twinx()
# Plot weather on its own (merged) timeline — continuous through radiation gaps
axarr[0].plot(weather_timestamps, weather_temp, color=col_pal[0])
bar_ax.plot(weather_timestamps, weather_pres, color=col_pal[1])
axarr[2].plot(weather_timestamps, weather_rain, color=col_pal[2])
bar_ax0.plot(weather_timestamps, weather_solar, color=col_pal[3])
# Plot radiation data (with NaN breaks for gaps) for each ROI
for x in range(0, len(col.rois)):
    axarr[1].errorbar(timestamps, roi_array[:, x, 0], yerr=roi_array[:, x, 1], marker='+',
                      label=col.rois[x].isotope, color=col_pal[4+x])

with open(os.path.join(DATA_DIR, 'weather.csv'), 'w') as out_file, open(os.path.join(DATA_DIR, 'weather_bq.csv'), 'w') as out_file_bq:
    csv_header = "Time, Pb214, Pb214_err, Bi214, Bi214_err, Pb212, Pb212_err, Tl208, Tl208_err, K40, K40_err, Cs134, Cs134_err, Cs137, Cs137_err\n"
    out_file.write(csv_header)
    out_file_bq.write(csv_header)

    mins = np.nanargmin(roi_array[:, :, 0], axis=0)
    print(mins)
    for x in range(max(0, len(timestamps) - 15 * 24), len(timestamps)):
        # Skip NaN break rows inserted for plot gaps
        if np.isnan(roi_array[x, 0, 0]):
            continue
        out_str = str(timestamps[x])
        out_str_bq = str(timestamps[x])
        for y in range(0, len(roi_array[x, :, 0])):
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
mx = np.nanmax(roi_array[:, :, 0])
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
    savefig(os.path.join(DATA_DIR, 'iso_' + time_utils.time_wins_str[p] + '.png'), transparent=True, bbox_inches='tight')
    p += 1

fig = figure()
p = 0
for x in time_utils.time_wins:
    lower = max([time_utils.beginning_date, x, timestamps[0]])
    new_axis = axarr[1]
    new_axis.set_xlim([lower, time_utils.right_now])
    new_axis.axes.get_xaxis().set_visible(True)
    setp(new_axis.xaxis.get_majorticklabels(), rotation=45, ha="right")
    savefig(os.path.join(DATA_DIR, 'trimmed_' + time_utils.time_wins_str[p] + '.png'), transparent=False, bbox_inches='tight')
    p += 1

# tmp plot of rainfall
clf()
# make cummulative sum
rain_data = np.nan_to_num(np.array(weather_rain), nan=0.0)
cum_sum = np.cumsum(rain_data)
plot(weather_timestamps, cum_sum)
savefig(os.path.join(DATA_DIR, 'out.png'))