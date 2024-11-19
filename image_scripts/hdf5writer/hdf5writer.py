import sys;
sys.path.insert(0,'..');
import sample_collection
import numpy as np;

spec_dir='/home/dosenet/Dropbox/UCB Air Monitor/Data/Roof/PAVLOVSKY/';
weat_csv='/home/dosenet/Dropbox/UCB Air Monitor/Data/Weather/weatherhawk.csv';

col = sample_collection.SampleCollection();
col.build_collection(spec_dir,weat_csv);
col.write_hdf('out.h5');
