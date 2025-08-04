from ctypes import cdll
import os

# Try to load system libstdc++ first - before ANY other imports
try:
    # Print for debugging
    print("Attempting to load system libstdc++...")
    system_lib = cdll.LoadLibrary("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    print("Successfully loaded system libstdc++")
except Exception as e:
    print(f"Failed to load system libstdc++: {e}")

# Now load xylib
try:
    print("Attempting to load libxy.so.3...")
    xylib = cdll.LoadLibrary("/usr/local/lib/libxy.so.3")
    print("Successfully loaded libxy.so.3")
except Exception as e:
    print(f"Failed to load libxy.so.3: {e}")

from ctypes import c_char_p, c_double
import datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
#import sample_collection
import numpy as np

#easier function names
#cdll.LoadLibrary("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
#xylib = cdll.LoadLibrary("libxy.so.3")

get_version = xylib.xylib_get_version
get_version.restype = c_char_p
load_file = xylib.xylib_load_file
get_block = xylib.xylib_get_block
count_columns = xylib.xylib_count_columns
count_rows = xylib.xylib_count_rows
get_data = xylib.xylib_get_data
get_data.restype = c_double
dataset_metadata = xylib.xylib_dataset_metadata
dataset_metadata.restype = c_char_p
block_metadata = xylib.xylib_block_metadata
block_metadata.restype = c_char_p
free_dataset = xylib.xylib_free_dataset

#file name is ascii path of file, sample is the 
#sample class from sample_colllection.py
def parse_spectra(file_name, sample):
    if hasattr(file_name, 'encode'):
        file_name = file_name.encode('utf-8')
    if not os.path.isfile(file_name):
        print(f"File does not exist: {file_name}")
        return sample  # Return the sample object, not an integer
    
    try:
        dataset = load_file(file_name, "cnf", None)  # Encode filename to bytes
        if not dataset:
            print(f"Failed to load dataset for: {file_name}")
            return sample  # Return the sample object, not an integer

        block = get_block(dataset, 0)
        ncol = count_columns(block)
        for i in range(ncol):
            count_rows(block, i+1)

        n = count_rows(block, 2)
        data = np.zeros((n, 2))
        for i in range(n):
            data[i, 0] = get_data(block, 1, i)
            data[i, 1] = get_data(block, 2, i)

        acq_real_time = float(block_metadata(block, 'real time (s)'))
        acq_live_time = float(block_metadata(block, 'live time (s)'))
        tmp_time = block_metadata(block, 'date and time')
        acq_ener_cal0 = float(block_metadata(block, 'energy calib 0'))
        acq_ener_cal1 = float(block_metadata(block, 'energy calib 1'))
        acq_ener_cal2 = float(block_metadata(block, 'energy calib 2'))
        print(f"Acquired Real Time: {acq_real_time}\n")

        
        # In Python 3, block_metadata returns bytes instead of str
        if isinstance(tmp_time, bytes):
            tmp_time = tmp_time.decode('utf-8')
            
        acq_time = datetime.datetime.strptime(tmp_time, "%a, %Y-%m-%d %H:%M:%S")
        acq_bin_lims = [1, len(data)]
        acq_cals = [acq_ener_cal0, acq_ener_cal1, acq_ener_cal2]
        
        sample.set_spectra(acq_time, acq_real_time, acq_live_time, acq_bin_lims, acq_cals, data[:, 0], data[:, 1])

        free_dataset(dataset)
        return sample

    except Exception as e:
        print(f"Error parsing spectra {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return sample  # Return the sample object, not an integer


def parse_roi(file_name):
    # Import only when needed to avoid circular dependency
    import sample_collection

    roi_col = []
    with open(file_name) as roi_file:
        content = roi_file.readlines()
    
    k = 0
    while k < len(content) - 1:
        bkg = []
        if k == 0:
            while not ('$ROI' in content[k]):
                k = k + 1
        k = k + 1
        sp_line = content[k].split(" ")
        iso = sp_line[3]
        energy = sp_line[4]
        origin = sp_line[5]
        spec_roi = [int(sp_line[1]), int(sp_line[2])]
        for x in range(0, int(sp_line[0])):
            k = k + 1
            sp_line = content[k].split(" ")
            bkg.append([int(sp_line[1]), int(sp_line[2])])
        r = sample_collection.ROI(spec_roi, bkg)
        r.origin = origin.replace('\n', '').replace("_", " ")
        r.isotope = iso.replace('\n', '')
        r.energy = float(energy.split('k')[0])
        roi_col.append(r)
    
    return roi_col

def parse_eff(file_name):
    with open(file_name) as f:
        content = f.readlines()
    
    k = 0
    while not ("kev_eff_%err_effw" in content[k].replace('\n', '')):
        k += 1
    k += 1
    num_points = int(content[k])
    k += 1
    eff = []
    for x in range(0, num_points):
        # Use list comprehension to filter out empty items after split
        tmp_str = [item for item in content[x+k].split(' ') if item]
        tmp = [float(tmp_str[0]), float(tmp_str[1]), float(tmp_str[2]), float(tmp_str[3])]
        eff.append(tmp)
    return eff        
        
