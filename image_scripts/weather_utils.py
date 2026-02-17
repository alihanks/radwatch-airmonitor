import sys
import os
import csv
from time_utils import *
plot_most_recent_spec = False
import matplotlib
if not plot_most_recent_spec:
    matplotlib.use('Agg')
from matplotlib.pyplot import *
import numpy as np
sys.path.insert(0, '../windrose')
sys.path.insert(0, '..')
from windrose import WindroseAxes
import sample_collection
import pytz

def resort_weather_timestamps(in_file,out_file):
    with open(in_file) as fil:
        date_frmt = '%Y-%m-%d %H:%M:%S'
        dims = []
        lines = []
        first_line = ''
        counter = -1
        # extract the dates
        for line in fil:
            counter += 1
            if counter == 0:
                first_line = line
                continue
            else:
                line_split = str.split(line, ',')
                dims.append(datetime.datetime.strptime(line_split[1], date_frmt))
                lines.append(line)
    
    # argsort the dates
    args = np.argsort(dims)
    lines_new = []
    # dims_new = []
    # rearrange for sorted dates in file
    for ar in args:
        lines_new.append(lines[ar])
        # dims_new.append(dims[ar])
    
    with open(out_file, 'w') as o_fil:
        o_fil.write(first_line)
        # write down sorted
        for line in lines_new:
            o_fil.write(line)
    
    return

def parse_date_to_struct(datetime_str):
    [date_str, time_str] = datetime_str.split(" ")
    date_array = date_str.split("-")
    time_array = time_str.split(":")    
    return datetime.datetime(int(date_array[0]), int(date_array[1]), int(date_array[2]), 
                            int(time_array[0]), int(time_array[1]), int(time_array[2]), tzinfo=pytz.timezone('US/Pacific'))

def get_units_label(order):
    unit_list = []
    label = []
    is_time_str = []
    for x in order:
        tmp = x.split("_")
        if len(tmp) == 1:
            unit_list.append(tmp[-1])
        else:
            unit_list.append(tmp[-2])
        is_time_str.append(not(-1 == tmp[0].find("Time")))
        label.append(tmp[0].replace("-", " "))
    return unit_list, label, is_time_str

def discard_data_before_time(x,win_time,k=0):
    l = len(x) - 1
    while (x[k] < win_time) and (k < l):
        k = k + 1
    return k
           
def rebin(x,y,win_str,win_time):
    if time_wins_str[0] == win_str:
        time_bin_size = datetime.timedelta(hours=0.125)
        one_half_bin = datetime.timedelta(hours=0.0625)
    elif time_wins_str[1] == win_str:
        time_bin_size = datetime.timedelta(days=0.125)
        one_half_bin = datetime.timedelta(days=0.0625)
    elif time_wins_str[2] == win_str:
        time_bin_size = datetime.timedelta(days=7./8.)
        one_half_bin = datetime.timedelta(days=7./16.)
    elif time_wins_str[3] == win_str:
        time_bin_size = datetime.timedelta(days=15./4.)
        one_half_bin = datetime.timedelta(days=15./8.)
    else:
        print('NO TIME BINS!')

    newx = []
    newy = []
    sum_ = 0
    k = discard_data_before_time(x, win_time, 0)
    p = k
    while k < len(x):
        while (p < len(x)) and (x[p] < (time_bin_size + x[k])):
            sum_ = sum_ + y[p]
            p = p + 1

        if p == k:
            continue
        newx.append(x[k] + one_half_bin)
        newy.append(sum_ / float(p - k))
        k = p
        sum_ = 0

    return newx, newy

def new_axes():
    fig = figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect, facecolor='w')
    fig.add_axes(ax)
    return ax

def set_legend(ax):
    l = ax.legend(borderaxespad=-0.10)
    setp(l.get_texts(), fontsize=8)

def dot_product(x,y):
    res = np.zeros(len(x))
    for p in range(0, len(x)):
        res[p] = x[p] * y[p]
    return res

def draw_compass(wind_direction,wind_speed,time_win_str):
    dir_ = sum(dot_product(wind_direction, wind_speed)) / sum(wind_speed)
    im = imread('../compass.jpg')
    imshow(im, extent=[-828, 828, -828, 828], origin='lower')

    r = np.arange(0, 500, 5)
    theta = 1./2. * np.pi - dir_ * np.pi / 180.
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plot(x, y, color='r', linewidth=2)
    savefig("compass_" + time_win_str + ".png")

    return
    
def draw_windrose(wind_direction,wind_speed,time_win_str):
    if len(wind_direction) <= 1 or len(wind_speed) <= 1:
        print("Something not right with windrose data at time", time_win_str)
        return
    from matplotlib.colors import LinearSegmentedColormap
    # windrose like a stacked histogram with normed (displayed in percent) results
    # make the colormap
    cdict = {'red': ((0, 1, 0), (1, 1, 0.6475)), 'green': ((0, 1, 0.3378), (1, 0.2131, 1)), 'blue': ((0, 1, 0.6622), (1, 0.1393, 1))}
    cmap = LinearSegmentedColormap("myown", cdict)
    
    ax = new_axes()
    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white', cmap=cmap)
    set_legend(ax)
    savefig("rose_" + time_win_str + ".png")  # , transparent=True, bbox_inches='tight')
    return

def parse_weather_data(f_name):
    order = ['RecordIDNumber', 'RecDateTime', 'Air-Temp-Avg_Temperature (C)_', 'Air-Temp-Min_Temperature (C)_',
             'Air-Temp-TimeMn_Temperature (C)_', 'Air-Temp-Max_Temperature (C)_', 'Air-Temp-TimeMx_Temperature (C)_',
             'RH-Avg_Percent (%)_', 'Barometer-Pressure_Pressure (kPa)_', 'Bat-Volt_V_', 'Bat-Volt-Min_V_',
             'ETo_unknown_', 'Rain-Yearly_mm_', 'Solar-Avg_W/m2_', 'Wind-Speed-Avg_m/s_', 'Wind-Speed-Max_m/s_',
             'Wind-Speed-TimeMx_m/s_', 'Wind-Speed_Wind Speed (m/s)_', 'Wind-Speed-Direction_Wind Direction (Degrees, True North)_']
    
    with open(f_name) as f:
        cvsreader = csv.DictReader(f, order)
        next(cvsreader)  # Skip header row
        list_of_lists = [[] for i in range(len(order))]
        units, label, is_time_str = get_units_label(order)
            
        index = -1
        prev_rain_sample = 0
        for row in cvsreader:
            index += 1
            for k in range(0, len(order)):
                if is_time_str[k]:
                    date_obj = parse_date_to_struct(row[order[k]])
                    list_of_lists[k].append(date_obj)
                elif k == 12:
                    # first rain sample or rain data reset.
                    if row[order[k]] == '':
                        prev_rain_sample = 0
                        list_of_lists[k].append(0.0)
                    else:
                        tmp = float(row[order[k]])
                        if index == 0:
                            prev_rain_sample = tmp
                            list_of_lists[k].append(0.)
                        else:
                            if float(row[order[k]]) - prev_rain_sample > 0:
                                list_of_lists[k].append(float(row[order[k]]) - prev_rain_sample)
                                prev_rain_sample = float(row[order[k]])
                            else:
                                list_of_lists[k].append(0)
                            if float(row[order[k]]) == 0:
                                prev_rain_sample = 0
                else:
                    if row[order[k]] == '':
                        list_of_lists[k].append(0.0)
                    else:
                        list_of_lists[k].append(float(row[order[k]]))
    
    return list_of_lists, units, label, is_time_str, order
