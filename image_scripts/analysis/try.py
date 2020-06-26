import glob;
import h5py;
import numpy as np;

fil_fil=h5py.File('try.hdf','r+');
fils=fil_fil['/2014/files'];

spec_dir='/home/rtpavlovsk21/Dropbox/UCB Air Monitor/Data/Roof/PAVLOVSKY/';
fil_list=glob.glob(spec_dir+'*/*.CNF'); fil_list.sort();
fil_list=np.asarray(fil_list);
print fils[0]==fil_list[0];
for x in range(len(fil_list)-1,-1,-1):
    if(fil_list[x]==fils[-1]):
        print x;break;
old_size=len(fils);
fils.resize(len(fil_list),axis=0);
print fils[-1];
for x in range(0,len(fil_list)-old_size):
    fils[x]=fil_list[x];

if( False):
    out_file=h5py.File('try.hdf','w');
    this_yr=out_file.create_group('2014');
    print len(fil_list);
    dt=h5py.special_dtype(vlen=bytes);
    this_yr.create_dataset('files',data=np.asarray(fil_list),dtype=dt,chunks=(2,));
