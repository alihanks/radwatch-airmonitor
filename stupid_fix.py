import numpy as np;
from matplotlib.pyplot import *;
import datetime;


fil = open('/home/dosenet/Dropbox/UCB Air Monitor/Data/Weather/weatherhawk.csv');
#fil=open('weatherhawk.csv');
frmt='%Y-%m-%d %H:%M:%S';
dims=[];
lines=[];
first_line='';
p=-1;
for line in fil:
    if(p==-1):
        p+=1;
        first_line=line;
        continue;
    else:
        str_split=str.split(line,',');
        dims.append(datetime.datetime.strptime(str_split[1],frmt));
        lines.append(line);

args=np.argsort(dims);
lines_=[];
dims_=[];df=[];
for ar in args:
    lines_.append(lines[ar]);
    dims_.append(dims[ar]);
for el in range(0,len(dims)):
    dif=dims_[el]-dims[el];
    df.append(dif.total_seconds());
plot(df);
show();

out_fil=open('out.csv','w');
out_fil.write(first_line);
for line in lines_:
    out_fil.write(line);
