from builtins import str
import sys
import os
import subprocess
sys.path.insert(0,"..")
from image_scripts.time_utils import *

specs_file="./log_files/spectras_not_converted.dat"
conv_specs="./log_files/converted_spectras.dat"
directory="./spectras/"+str(this_year)

if not os.path.exists(directory):
    os.makedirs(directory)

#find files that haven't been converted. Store them in
#spectras_not_converted.dat
subprocess.call("./compare_spectra_lists.sh",shell=True)

f_specs_to_convert=open(specs_file)
f_conv_specs=open(conv_specs,"a")
for line in f_specs_to_convert:
    line=line.replace(" ","\\ ").replace("\n","")
    fil_in_this_dir="./spectras/"+line.replace("/home/dosenet/radwatch-airmonitor/Dropbox/UCB\\ Air\\ Monitor/Data/Roof/PAVLOVSKY/","")
    cmd1="../xylib-1.3/xyconv " +line + " " + fil_in_this_dir.replace(".CNF",".txt")
    subprocess.call( cmd1 , shell=True)
    cmd2="python txt_to_spe.py " + fil_in_this_dir.replace(".CNF",".txt") + " " + fil_in_this_dir.replace(".CNF",".Spe")
    subprocess.call(cmd2 , shell=True)
    f_conv_specs.write(line.replace("\\ "," ")+"\n")

