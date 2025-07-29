from builtins import str
from builtins import range
import numpy as np;
import subprocess
import re
from matplotlib.pyplot import *
import sys
sys.path.insert(0,'..')
from image_scripts import time_utils

filename=sys.argv[1]
str_list=open(filename)
energy_cal=np.zeros(3)

mm_dd_yyyy=''
time=''
data=np.zeros((0,2))
for el in str_list:
    if "date and time" in el:
        tmp=el.split(",")
        tmp_date_time=tmp[1].split(" ")
        tmp_yyyy=tmp_date_time[1].split("-")
        #print tmp_yyyy
        mm_dd_yyyy = tmp_yyyy[1]+"/"+tmp_yyyy[2]+"/"+tmp_yyyy[0]
        time = str.replace(tmp_date_time[2],"\n","")
    if "live time" in el:
        tmp=el.split(":")
        live_time=float(tmp[1])
        #print "live time:", live_time
    if "real time" in el:
        tmp=el.split(":")
        real_time=float(tmp[1])
        #print "real time:", real_time
    if "energy calib 0" in el:
        tmp=el.split(":")
        energy_cal[0]=float(tmp[1])
    if "energy calib 1" in el:
        tmp=el.split(":")
        energy_cal[1]=float(tmp[1])
    if "energy calib 2" in el:
        tmp=el.split(":")
        energy_cal[2]=float(tmp[1])
    if not "#" in el:
        if "\t" in el:
            tmp=el.split("\t")
            #print tmp
            data=np.vstack((data,[float(tmp[0]), float(tmp[1])]))

ene=np.zeros(len(data))
diff=np.zeros(len(data))
for x in range(0,len(data)):
    ene[x]=(x+1)*energy_cal[1]+energy_cal[0]
    diff[x]=ene[x]-data[x,0]

out_file_name = sys.argv[2]
out_file=open(out_file_name,'w')

out_file.write("$SPEC_ID:\nNo sample description was entered.\n$SPEC_REM:\nDET# 2\nDETDESC# RoofTopBEGE\nAP# GammaVision Version 6.09\n")
out_file.write("$DATE_MEA:\n")
out_file.write(mm_dd_yyyy+" "+time+"\n")
out_file.write("$MEAS_TIM:\n")
out_file.write(str(live_time)+" "+str(real_time)+"\n")
out_file.write("$DATA:\n")
out_file.write("1 "+str(len(data)) +"\n")
for x in range(0,len(data)):
    out_file.write("{0:8}".format(int(data[x,1]))+"\n")
out_file.write("$ENER_FIT:\n")
out_file.write(str(energy_cal[0])+" "+ str(energy_cal[1]) )

close(filename)
close(out_file_name)
subprocess.call(["rm",filename])
