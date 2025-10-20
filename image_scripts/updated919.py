import os
import datetime
import sys
# The pip package handles the underlying C library, so you don't need to do this.
# from ctypes import cdll, c_char_p, c_double
# Try to load system libstdc++ first - before ANY other imports
# try:
#     print("Attempting to load system libstdc++...")
#     system_lib = cdll.LoadLibrary("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
#     print("Successfully loaded system libstdc++")
# except Exception as e:
#     print(f"Failed to load system libstdc++: {e}")
# Now load xylib
# try:
#     print("Attempting to load libxy.so.3...")
#     xylib = cdll.LoadLibrary("/home/dosenet/xylib-1.6/build/libxy.so.4")
#     print("Successfully loaded libxy.so.3")
# except Exception as e:
#     print(f"Failed to load libxy.so.3: {e}")

# The pip package provides the 'xylib' module directly.
import xylib
import numpy as np
# The sample_collection module is still a dependency if used.
# import sample_collection

# The functions below are now built-in to the xylib module's objects.

# file name is ascii path of file, sample is the
# sample class from sample_colllection.py
def parse_spectra(file_name, sample):
    """
    Parses a CANBERRA CNF spectra file and updates a sample object.
    
    Args:
        file_name (str): The path to the spectra file.
        sample (object): An object with a set_spectra method to store the data.
    
    Returns:
        object: The updated sample object.
    """
    if not os.path.isfile(file_name):
        print(f"File does not exist: {file_name}")
        return sample

    try:
        # Load the dataset using the high-level function
        dataset = xylib.load_file(file_name, "canberra_cnf")
        if not dataset:
            print(f"Failed to load dataset for: {file_name}")
            return sample

        # Access the first block in the dataset
        block = dataset.get_block(0)

        # Get data and metadata using the object's methods
        data = np.array(block.get_data())

        acq_real_time = float(block.metadata['real time (s)'])
        acq_live_time = float(block.metadata['live time (s)'])
        acq_time_str = block.metadata['date and time']
        acq_ener_cal0 = float(block.metadata['energy calib 0'])
        acq_ener_cal1 = float(block.metadata['energy calib 1'])
        acq_ener_cal2 = float(block.metadata['energy calib 2'])
        print(f"Acquired Real Time: {acq_real_time}\n")

        # The pip package returns strings, so no decoding is needed.
        acq_time = datetime.datetime.strptime(acq_time_str, "%a, %Y-%m-%d %H:%M:%S")
        acq_bin_lims = [1, data.shape[0]]
        acq_cals = [acq_ener_cal0, acq_ener_cal1, acq_ener_cal2]

        # data is a NumPy array, so you can access columns with slicing
        sample.set_spectra(acq_time, acq_real_time, acq_live_time, acq_bin_lims, acq_cals, data[:, 0], data[:, 1])

        # The Python object handles memory management automatically, no need to call free_dataset
        return sample

    except Exception as e:
        print(f"Error parsing spectra {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return sample

def parse_roi(file_name):
    # This function doesn't use xylib, so it remains unchanged.
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
    # This function also doesn't use xylib, so it remains unchanged.
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