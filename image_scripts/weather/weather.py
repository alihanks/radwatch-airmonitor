import os;
import csv;
import sys;
import math;
import datetime;
import numpy as np;
from matplotlib.dates import DayLocator;
from matplotlib.pyplot import *;
sys.path.insert(0,'..');
from weather_utils import *;


def parse_date_to_struct(datetime_str):
    [date_str, time_str]=datetime_str.split(" ");
    date_array=date_str.split("-");
    time_array=time_str.split(":");
    
    return datetime.datetime(int(date_array[0]),int(date_array[1]),int(date_array[2]),int(time_array[0]),int(time_array[1]),int(time_array[2]));

def get_units_label(order):
    unit_list=[];
    label=[];
    is_time_str=[];
    for x in order:
        tmp=x.split("_");
        if(len(tmp)==1):
            unit_list.append(tmp[-1]);
        else:
            unit_list.append(tmp[-2]);
        is_time_str.append(not(-1==tmp[0].find("Time")));
        label.append(tmp[0].replace("-", " "));
    return unit_list,label,is_time_str;

def discard_data_before_time(x,win_time):
    k=0;
    while( x[k] < win_time):
        k=k+1;
    return k;
            
def rebin(x,y,win_str,win_time):
    if(  time_wins_str[0]==win_str):
        time_bin_size=datetime.timedelta(hours=0.125);
        one_half_bin=datetime.timedelta(hours=0.0625);
    elif(time_wins_str[1]==win_str):
        time_bin_size=datetime.timedelta(days=0.125);
        one_half_bin=datetime.timedelta(days=0.0625);
    elif(time_wins_str[2]==win_str):
        time_bin_size=datetime.timedelta(days=7./8.);
        one_half_bin=datetime.timedelta(days=7./16.);
    elif(time_wins_str[3]==win_str):
        time_bin_size=datetime.timedelta(days=15./4.);
        one_half_bin=datetime.timedelta(days=15./8.);
    else:
        print 'NO TIME BINS!';

    newx=[];
    newy=[];
    sum_=0;
    k=discard_data_before_time(x,win_time);
    p=k;
    while( k < len(x)):
        while( (p<len(x)) and ( x[p] < (time_bin_size+x[k])) ):
            sum_=sum_+y[p];
            p=p+1;

        if(p==k): continue;
        newx.append(x[k]+one_half_bin);
        newy.append(sum_/float(p-k));
        k=p;
        sum_=0;

    return newx,newy;

def new_axes():
    fig = figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w');
    rect = [0.1, 0.1, 0.8, 0.8];
    ax = WindroseAxes(fig, rect, axisbg='w');
    fig.add_axes(ax);
    return ax

def set_legend(ax):
    l = ax.legend(axespad=-0.10);
    setp(l.get_texts(), fontsize=8);

def dot_product(x,y):
    res=np.zeros(len(x));
    for p in range(0,len(x)):
        res[p]=x[p]*y[p];
    return res;

def draw_compass(wind_direction,wind_speed,time_win_str):
    dir_= sum(dot_product(wind_direction,wind_speed))/sum(wind_speed);
    im=imread('../compass.jpg');
    imshow(im,extent=[-828,828,-828,828],origin='lower');

    r=np.arange(0,500,5);
    theta=1./2.*np.pi-dir_*np.pi/180.;
    x=r*np.cos(theta);
    y=r*np.sin(theta);
    plot(x,y,color='r',linewidth=2);
    savefig("compass_"+time_win_str+".png");
    
    #windrose like a stacked histogram with normed (displayed in percent) results                                                              
    ax = new_axes();
    ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white');
    set_legend(ax);
    savefig("rose_"+time_win_str+".png");
    return;

def parse_weather_data(f_name):
    order= ['RecordIDNumber','RecDateTime','Air-Temp-Avg_Temperature (C)_','Air-Temp-Min_Temperature (C)_','Air-Temp-TimeMn_Temperature (C)_','Air-Temp-Max_Temperature (C)_','Air-Temp-TimeMx_Temperature (C)_','RH-Avg_Percent (%)_','Barometer-Pressure_Pressure (kPa)_','Bat-Volt_V_','Bat-Volt-Min_V_','ETo_unknown_','Rain-Yearly_mm_','Solar-Avg_W/m2_','Wind-Speed-Avg_m/s_','Wind-Speed-Max_m/s_','Wind-Speed-TimeMx_m/s_','Wind-Speed_Wind Speed (m/s)_','Wind-Speed-Direction_Wind Direction (Degrees, True North)_'];
    cvsreader=csv.DictReader(open(f_name),order);
    cvsreader.next();
    list_of_lists=[ [] for i in range(len(order)) ];
    units,label,is_time_str=get_units_label(order);
        
    for row in cvsreader:
        for k in range(0,len(order)):
            if( is_time_str[k] ):
                date_obj=parse_date_to_struct( row[order[k]] );
                list_of_lists[k].append(date_obj);
            else:
                list_of_lists[k].append( float(row[order[k]] ) );    
    return list_of_lists,units,label,is_time_str,order;

def makePlots(list_of_lists,units,label,is_time_str,order):
    #units,label,is_time_str=get_units_label(order);
    for p in range( 2, len(order) ):
        if( is_time_str[p]==True ): continue;
        fold_dir='./'+label[p].replace(' ','_');
        if not os.path.exists(fold_dir): os.makedirs(fold_dir);
        os.chdir(fold_dir);
        for x in range(0,len(time_wins)):
            horiz,vert=rebin(list_of_lists[1],list_of_lists[p],time_wins_str[x],time_wins[x]);
            plot_date(horiz, vert,marker='+');
            gcf().subplots_adjust(bottom=0.20);
            xticks(rotation=45,ha='right');
            ylabel(units[p]);
            title(label[p]);
            savefig(time_wins_str[x]+'.png');
            clf();
            if(p==len(order)-1):
                horz_speed,vert_speed=rebin(list_of_lists[1],list_of_lists[p-1],time_wins_str[x],time_wins[x]);
                draw_compass(vert,vert_speed,time_wins_str[x]);
                clf();
        os.chdir('..');

def main():
    f_name=sys.argv[1];
    list_of_lists,units,label,is_time_str,order=parse_weather_data(f_name);
    makePlots(list_of_lists,units,label,is_time_str,order);

if __name__=='__main__':
    main();
