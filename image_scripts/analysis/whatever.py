import glob
spec_dir="/home/dosenet/radwatch-airmonitor/Dropbox/UCB Air Monitor/Data/Roof/PAVLOVSKY/"
print(spec_dir)
fil_list=glob.glob(spec_dir+'*/*.CNF'); fil_list.sort()
for k in fil_list:
    print(k)
